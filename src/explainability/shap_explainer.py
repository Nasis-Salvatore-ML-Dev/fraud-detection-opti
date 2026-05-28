"""SHAP serving utilities — reads pre-computed SHAP values from DynamoDB.

shap is never imported here. SHAP values are stored offline by
scripts/compute_shap.py and retrieved at serving time via get_shap_values().
"""

import logging
import os

import boto3
import msgpack

from src.monitoring.metrics import publish_component_failure

log = logging.getLogger(__name__)

_SHAP_TABLE = os.environ.get("SHAP_TABLE", "fraud-shap-store")
_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")


def get_shap_values(prediction_hash: str, table_name: str = "") -> dict | None:
    """Fetch pre-computed SHAP values from DynamoDB by prediction_hash.

    Returns a dict {feature_name: shap_value} or None if not found or on error.
    table_name defaults to the SHAP_TABLE env var (default: fraud-shap-store).
    """
    if not table_name:
        table_name = os.environ.get("SHAP_TABLE", "fraud-shap-store")

    try:
        region = os.environ.get("AWS_DEFAULT_REGION", _REGION)
        dynamo = boto3.resource("dynamodb", region_name=region)
        table = dynamo.Table(table_name)

        response = table.get_item(Key={"prediction_hash": prediction_hash})
        item = response.get("Item")
        if item is None:
            return None

        return msgpack.unpackb(bytes(item["shap_values"]), raw=False)

    except Exception as exc:
        log.warning("get_shap_values failed (non-fatal): %s", exc)
        publish_component_failure("ShapExplainer")
        return None
