"""Audit logger — writes prediction records to DynamoDB for compliance and replay."""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from decimal import Decimal

import boto3
import msgpack

from src.explainability.shap_explainer import get_shap_values
from src.monitoring.metrics import publish_component_failure

log = logging.getLogger(__name__)

_AUDIT_TABLE = os.environ.get("AUDIT_TABLE", "fraud-audit-log")
_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

_REVIEW_PROB_LO = 0.3
_REVIEW_PROB_HI = 0.7
_REVIEW_TTL_SECONDS = 30 * 24 * 3600


def _to_decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _from_dynamo(obj):
    """Recursively convert DynamoDB Decimal back to float."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _from_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_dynamo(v) for v in obj]
    return obj


def extract_v_features(input_dict: dict) -> dict:
    """Pull V1-V28 from input_dict and return them as a plain {Vi: float} dict."""
    return {
        key: float(val)
        for key, val in input_dict.items()
        if key.startswith("V") and key[1:].isdigit()
    }


class AuditLogger:
    """Writes prediction records to DynamoDB for compliance and explainability."""

    def __init__(self) -> None:
        self._dynamo = boto3.resource("dynamodb", region_name=_REGION)
        self._audit_table = self._dynamo.Table(_AUDIT_TABLE)
        log.info("AuditLogger initialised  table=%s  region=%s", _AUDIT_TABLE, _REGION)

    async def write(
        self,
        prediction_id: str,
        prediction_hash: str,
        input_features: dict,
        fraud_probability: float,
        is_fraud: bool,
        shap_values: dict,
        model_version: str,
        confidence_score: float,
        request_ip: str,
        latency_ms: float,
        threshold_used: float,
    ) -> None:
        """PutItem to DynamoDB. Logs error on failure but never raises."""
        requires_review = _REVIEW_PROB_LO <= fraud_probability <= _REVIEW_PROB_HI

        # V1-V28 compressed with msgpack
        v_dict = extract_v_features(input_features)
        v_compressed = bytes(msgpack.packb(v_dict))

        # SHAP lookup — non-blocking; degrades to empty list on failure
        shap_top3: list = []
        try:
            shap_dict = get_shap_values(prediction_hash)
            if shap_dict is not None:
                sorted_items = sorted(
                    shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True
                )
                shap_top3 = [
                    {"feature": k, "value": Decimal(str(v))}
                    for k, v in sorted_items[:3]
                ]
                log.debug("SHAP lookup OK for prediction_hash=%s", prediction_hash)
            else:
                log.debug("SHAP not found for prediction_hash=%s", prediction_hash)
        except Exception as exc:
            log.warning("SHAP lookup failed for prediction_hash=%s: %s", prediction_hash, exc)
            publish_component_failure("AuditLogger")

        item: dict = {
            "prediction_id": prediction_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction_hash": prediction_hash,
            # Named engineered features
            "Amount": _to_decimal(input_features.get("Amount", 0)),
            "amount_log": _to_decimal(input_features.get("amount_log", 0)),
            "amount_zscore": _to_decimal(input_features.get("amount_zscore", 0)),
            "hour_of_day": _to_decimal(input_features.get("hour_of_day", 0)),
            # V1-V28 compressed
            "v_features_msgpack": v_compressed,
            # Prediction
            "fraud_probability": _to_decimal(fraud_probability),
            "is_fraud": is_fraud,
            "shap_top3": shap_top3,
            "model_version": model_version,
            "confidence_score": _to_decimal(confidence_score),
            "request_ip": request_ip,
            "latency_ms": _to_decimal(latency_ms),
            "threshold_used": _to_decimal(threshold_used),
            # Review flag (stored as string for GSI compatibility)
            "requires_review": "true" if requires_review else "false",
        }

        if requires_review:
            item["review_expires_at"] = int(time.time()) + _REVIEW_TTL_SECONDS

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._audit_table.put_item(Item=item))
        except Exception as exc:
            log.error("AuditLogger.write failed (silent): %s", exc)
            publish_component_failure("AuditLogger")

    async def fetch(self, prediction_id: str) -> dict | None:
        """GetItem by prediction_id. Returns None if not found or on error."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._audit_table.get_item(Key={"prediction_id": prediction_id}),
            )
            item = response.get("Item")
            if item is None:
                return None
            record = _from_dynamo(item)
            # Reconstruct input_features dict from structured fields for consumers
            if "Amount" in record:
                v_features = {}
                if "v_features_msgpack" in item:
                    v_features = msgpack.unpackb(bytes(item["v_features_msgpack"]), raw=False)
                record["input_features"] = {
                    "Amount": float(record.get("Amount", 0)),
                    "amount_log": float(record.get("amount_log", 0)),
                    "amount_zscore": float(record.get("amount_zscore", 0)),
                    "hour_of_day": float(record.get("hour_of_day", 0)),
                    **v_features,
                }
            return record
        except Exception as exc:
            log.error("AuditLogger.fetch failed (silent): %s", exc)
            publish_component_failure("AuditLogger")
            return None

    async def fetch_recent(self, limit: int = 500) -> list[dict]:
        """Scan the audit table and return up to limit records for drift computation."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self._audit_table.scan(Limit=limit)
            )
            records: list[dict] = []
            for item in response.get("Items", []):
                record = _from_dynamo(item)
                # Reconstruct input_features for drift monitor
                if "Amount" in record:
                    v_features = {}
                    if "v_features_msgpack" in item:
                        v_features = msgpack.unpackb(bytes(item["v_features_msgpack"]), raw=False)
                    record["input_features"] = {
                        "Amount": float(record.get("Amount", 0)),
                        "amount_log": float(record.get("amount_log", 0)),
                        "amount_zscore": float(record.get("amount_zscore", 0)),
                        "hour_of_day": float(record.get("hour_of_day", 0)),
                        **v_features,
                    }
                records.append(record)
            return records
        except Exception as exc:
            log.error("AuditLogger.fetch_recent failed (silent): %s", exc)
            publish_component_failure("AuditLogger")
            return []

    async def fetch_pending_reviews(self) -> list[dict]:
        """Query the requires-review-index GSI and return records with requires_review='true'."""
        from boto3.dynamodb.conditions import Key

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._audit_table.query(
                    IndexName="requires-review-index",
                    KeyConditionExpression=Key("requires_review").eq("true"),
                ),
            )
            records: list[dict] = []
            for item in response.get("Items", []):
                record = _from_dynamo(item)
                records.append(
                    {
                        "prediction_id": record.get("prediction_id"),
                        "fraud_probability": record.get("fraud_probability"),
                        "shap_top3": record.get("shap_top3", []),
                        "amount": float(record.get("Amount", 0)),
                        "requires_review": record.get("requires_review"),
                        "timestamp": record.get("timestamp"),
                    }
                )
            return records
        except Exception as exc:
            log.error("AuditLogger.fetch_pending_reviews failed: %s", exc)
            publish_component_failure("AuditLogger")
            return []
