import hashlib
import logging
import os
from decimal import Decimal, InvalidOperation

import boto3
import numpy as np

from src.monitoring.metrics import publish_component_failure

log = logging.getLogger(__name__)

_VELOCITY_KEYS = (
    "tx_count_1h",
    "tx_count_24h",
    "tx_count_7d",
    "time_since_last_tx_seconds",
    "amount_sum_1h",
)

_ZERO_RESULT: dict = {
    "tx_count_1h": 0,
    "tx_count_24h": 0,
    "tx_count_7d": 0,
    "time_since_last_tx_seconds": 0,
    "amount_sum_1h": 0.0,
}

_MAX_HISTORY = 100


def _to_decimal(value: float) -> Decimal:
    try:
        return Decimal(str(round(value, 6)))
    except InvalidOperation:
        return Decimal("0")


class VelocityStore:
    def __init__(self) -> None:
        self._table_name = os.environ.get("VELOCITY_TABLE", "fraud-velocity-store")
        self._dynamodb = boto3.resource(
            "dynamodb",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "eu-central-1"),
        )
        self._table = self._dynamodb.Table(self._table_name)
        log.info("VelocityStore initialised with table=%s", self._table_name)

    @staticmethod
    def _card_hash(feature_vector: np.ndarray) -> str:
        return hashlib.sha256(feature_vector.tobytes()).hexdigest()

    def get_card_features(self, feature_vector: np.ndarray) -> dict:
        card_hash = self._card_hash(feature_vector)
        log.debug("Querying velocity store for card_hash=%.8s...", card_hash)
        try:
            response = self._table.get_item(Key={"card_hash": card_hash})
            item = response.get("Item", {})
            if not item:
                log.debug(
                    "No velocity record for card_hash=%.8s — returning defaults", card_hash
                )
                return dict(_ZERO_RESULT)
            result = {
                "tx_count_1h": int(item.get("tx_count_1h", 0)),
                "tx_count_24h": int(item.get("tx_count_24h", 0)),
                "tx_count_7d": int(item.get("tx_count_7d", 0)),
                "time_since_last_tx_seconds": float(
                    item.get("time_since_last_tx_seconds", 0)
                ),
                "amount_sum_1h": float(item.get("amount_sum_1h", 0.0)),
            }
            return result
        except Exception as exc:
            log.warning("VelocityStore unavailable, returning zeros: %s", exc)
            publish_component_failure("VelocityStore")
            return dict(_ZERO_RESULT)

    def update_card_features(
        self, feature_vector: np.ndarray, amount: float, timestamp: float
    ) -> None:
        card_hash = self._card_hash(feature_vector)
        log.debug("Updating velocity store for card_hash=%.8s...", card_hash)
        try:
            # Read existing record
            response = self._table.get_item(Key={"card_hash": card_hash})
            item = response.get("Item", {})

            # Extract existing history lists (convert Decimals to Python floats)
            tx_timestamps: list[float] = [
                float(ts) for ts in item.get("tx_timestamps", [])
            ]
            amount_history: list[dict] = [
                {"ts": float(e["ts"]), "amount": float(e["amount"])}
                for e in item.get("amount_history", [])
            ]

            # Time since last known transaction (stored for serving path)
            last_updated = item.get("last_updated")
            time_since = (
                float(timestamp) - float(last_updated)
                if last_updated is not None
                else -1.0
            )

            # Append and cap at 100 most recent entries
            tx_timestamps.append(timestamp)
            tx_timestamps = tx_timestamps[-_MAX_HISTORY:]

            amount_history.append({"ts": timestamp, "amount": amount})
            amount_history = amount_history[-_MAX_HISTORY:]

            # Count transactions within each time window
            cutoff_1h = timestamp - 3600
            cutoff_24h = timestamp - 86400
            cutoff_7d = timestamp - 604800

            tx_count_1h = sum(1 for ts in tx_timestamps if ts >= cutoff_1h)
            tx_count_24h = sum(1 for ts in tx_timestamps if ts >= cutoff_24h)
            tx_count_7d = sum(1 for ts in tx_timestamps if ts >= cutoff_7d)
            amount_sum_1h = sum(
                e["amount"] for e in amount_history if e["ts"] >= cutoff_1h
            )

            # TTL = now + 7 days (604800 seconds)
            expires_at = int(timestamp) + 604800

            self._table.put_item(Item={
                "card_hash": card_hash,
                "tx_timestamps": [_to_decimal(ts) for ts in tx_timestamps],
                "amount_history": [
                    {
                        "ts": _to_decimal(e["ts"]),
                        "amount": _to_decimal(e["amount"]),
                    }
                    for e in amount_history
                ],
                "tx_count_1h": tx_count_1h,
                "tx_count_24h": tx_count_24h,
                "tx_count_7d": tx_count_7d,
                "time_since_last_tx_seconds": _to_decimal(time_since),
                "amount_sum_1h": _to_decimal(amount_sum_1h),
                "last_updated": _to_decimal(timestamp),
                "expires_at": expires_at,
            })
            log.debug("Velocity store updated for card_hash=%.8s...", card_hash)
        except Exception as exc:
            log.warning("VelocityStore update failed, skipping: %s", exc)
            publish_component_failure("VelocityStore")
