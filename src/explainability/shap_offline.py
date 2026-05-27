# This module must never be imported from src/api/ or src/utils/.
#  It runs offline at training time only. shap is a dev dependency.

import hashlib
import importlib
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal

import boto3
import msgpack
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_BATCH_SIZE = 500
_TTL_SECONDS = 90 * 24 * 3600


def compute_and_store_shap(
    model,
    X: pd.DataFrame,
    background: pd.DataFrame,
    table_name: str,
) -> int:
    """Compute SHAP values for all rows in X and store them in DynamoDB.

    Uses the given background dataset to initialise a TreeExplainer.
    Processes X in batches of 500 rows. Returns total number of items stored.
    """
    shap = importlib.import_module("shap")

    region = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
    dynamo = boto3.resource("dynamodb", region_name=region)
    table = dynamo.Table(table_name)

    explainer = shap.TreeExplainer(model, background)

    feature_names = list(X.columns)
    now = datetime.now(timezone.utc)
    expires_at = int(now.timestamp()) + _TTL_SECONDS
    computed_at = now.isoformat()

    total_stored = 0

    for batch_start in range(0, len(X), _BATCH_SIZE):
        batch = X.iloc[batch_start : batch_start + _BATCH_SIZE]
        shap_vals = explainer.shap_values(batch)

        # Binary classifiers may return a list [class_0_vals, class_1_vals]
        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        else:
            sv = shap_vals

        with table.batch_writer() as writer:
            for row_idx, (_, row) in enumerate(batch.iterrows()):
                try:
                    feature_vector = row.values.astype(np.float64).tobytes()
                    prediction_hash = hashlib.sha256(feature_vector).hexdigest()

                    shap_dict = {
                        name: float(val)
                        for name, val in zip(feature_names, sv[row_idx])
                    }

                    sorted_items = sorted(
                        shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True
                    )
                    shap_top3 = [
                        {"feature": k, "value": Decimal(str(v))}
                        for k, v in sorted_items[:3]
                    ]

                    compressed = bytes(msgpack.packb(shap_dict))

                    writer.put_item(
                        Item={
                            "prediction_hash": prediction_hash,
                            "shap_values": compressed,
                            "shap_top3": shap_top3,
                            "computed_at": computed_at,
                            "expires_at": expires_at,
                        }
                    )
                    total_stored += 1
                except Exception as exc:
                    log.warning(
                        "Failed to write SHAP item (row %d in batch starting %d): %s",
                        row_idx,
                        batch_start,
                        exc,
                    )

        log.info(
            "SHAP batch complete: rows %d-%d  batch_stored=%d  total_stored=%d",
            batch_start,
            batch_start + len(batch) - 1,
            len(batch),
            total_stored,
        )

    log.info("SHAP computation complete: total_stored=%d", total_stored)
    return total_stored
