import hashlib
import logging
import os
from decimal import Decimal

import boto3
import numpy as np

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


class VelocityStore:
    def __init__(self) -> None:
        self._table_name = os.environ.get("VELOCITY_TABLE", "fraud-velocity-store")
        self._dynamodb = boto3.resource("dynamodb")
        self._table = self._dynamodb.Table(self._table_name)
        log.debug("VelocityStore initialised with table=%s", self._table_name)

    @staticmethod
    def _card_hash(feature_vector: np.ndarray) -> str:
        return hashlib.sha256(feature_vector.tobytes()).hexdigest()

    def get_card_features(self, feature_vector: np.ndarray) -> dict:
        card_hash = self._card_hash(feature_vector)
        log.debug("Querying velocity store for card_hash=%.8s...", card_hash)
        try:
            response = self._table.get_item(Key={"card_hash": card_hash})
            item = response.get("Item", {})
            result = {
                "tx_count_1h": int(item.get("tx_count_1h", 0)),
                "tx_count_24h": int(item.get("tx_count_24h", 0)),
                "tx_count_7d": int(item.get("tx_count_7d", 0)),
                "time_since_last_tx_seconds": int(item.get("time_since_last_tx_seconds", 0)),
                "amount_sum_1h": float(item.get("amount_sum_1h", 0.0)),
            }
            log.info("Velocity features retrieved for card_hash=%.8s...", card_hash)
            return result
        except Exception as exc:
            log.warning("VelocityStore unavailable, returning zeros: %s", exc)
            return dict(_ZERO_RESULT)

    def update_card_features(
        self, feature_vector: np.ndarray, amount: float, timestamp: float
    ) -> None:
        card_hash = self._card_hash(feature_vector)
        log.debug("Updating velocity store for card_hash=%.8s...", card_hash)
        try:
            self._table.update_item(
                Key={"card_hash": card_hash},
                UpdateExpression=(
                    "SET tx_count_1h = if_not_exists(tx_count_1h, :zero) + :one, "
                    "tx_count_24h = if_not_exists(tx_count_24h, :zero) + :one, "
                    "tx_count_7d = if_not_exists(tx_count_7d, :zero) + :one, "
                    "amount_sum_1h = if_not_exists(amount_sum_1h, :fzero) + :amount, "
                    "last_tx_timestamp = :ts, "
                    "time_since_last_tx_seconds = :zero"
                ),
                ExpressionAttributeValues={
                    ":one": 1,
                    ":zero": 0,
                    ":fzero": Decimal("0"),
                    ":amount": Decimal(str(round(amount, 6))),
                    ":ts": Decimal(str(round(timestamp, 6))),
                },
            )
            log.info("Velocity store updated for card_hash=%.8s...", card_hash)
        except Exception as exc:
            log.warning("VelocityStore update failed, skipping: %s", exc)
