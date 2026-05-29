"""Transform a PredictionRequest into a model-ready DataFrame.

Feature engineering is delegated to src.features.engineering so training and
serving share a single canonical implementation.  Amount normalisation stats
are read from the loaded model bundle — call set_amount_stats() before
build_feature_dataframe().
"""

import logging

import numpy as np
import pandas as pd

from src.api.schemas import PredictionRequest
from src.features.engineering import engineer_features
from src.features.velocity import VelocityStore

log = logging.getLogger(__name__)

_amount_stats: dict | None = None

# Single VelocityStore instance shared for the lifetime of the process.
velocity_store = VelocityStore()


def set_amount_stats(stats: dict) -> None:
    """Register amount normalisation stats from the loaded model bundle."""
    global _amount_stats
    _amount_stats = stats
    log.debug(
        "Amount stats configured: mean=%.4f std=%.4f", stats["mean"], stats["std"]
    )


def build_feature_dataframe(
    request: PredictionRequest,
    feature_names: list[str],
) -> pd.DataFrame:
    """Convert a PredictionRequest into a single-row DataFrame for inference.

    Raises:
        RuntimeError: If set_amount_stats() has not been called yet.
        ValueError: If the constructed feature set does not match feature_names.
    """
    if _amount_stats is None:
        raise RuntimeError(
            "Amount stats are not configured. Call set_amount_stats() with the "
            "model bundle's 'amount_stats' dict before serving predictions."
        )

    # Extract V1-V28 before engineering — used as the card identity for velocity lookup.
    v_feature_vector = np.array(
        [float(getattr(request, f"v{i}")) for i in range(1, 29)],
        dtype=np.float64,
    )

    raw: dict[str, float] = {
        "V1": request.v1,
        "V2": request.v2,
        "V3": request.v3,
        "V4": request.v4,
        "V5": request.v5,
        "V6": request.v6,
        "V7": request.v7,
        "V8": request.v8,
        "V9": request.v9,
        "V10": request.v10,
        "V11": request.v11,
        "V12": request.v12,
        "V13": request.v13,
        "V14": request.v14,
        "V15": request.v15,
        "V16": request.v16,
        "V17": request.v17,
        "V18": request.v18,
        "V19": request.v19,
        "V20": request.v20,
        "V21": request.v21,
        "V22": request.v22,
        "V23": request.v23,
        "V24": request.v24,
        "V25": request.v25,
        "V26": request.v26,
        "V27": request.v27,
        "V28": request.v28,
        "Amount": float(request.amount),
        "Time": float(request.time),
    }

    df = pd.DataFrame([raw])
    df = engineer_features(df, _amount_stats)

    # Append velocity features (5 columns) after the engineered base features.
    vel = velocity_store.get_card_features(v_feature_vector)
    all_defaults = all(v == 0 or v == 0.0 for v in vel.values())
    if all_defaults:
        log.debug(
            "Velocity features returned defaults for card_hash=%.8s...",
            __import__("hashlib").sha256(v_feature_vector.tobytes()).hexdigest(),
        )
    for key, value in vel.items():
        df[key] = value

    # Select and reorder to match the exact sequence the model expects
    try:
        ordered = df[feature_names]
    except KeyError as missing:
        raise ValueError(
            f"Engineered features are missing column(s) expected by the model: {missing}. "
            f"Model expects: {feature_names}"
        ) from missing

    if list(ordered.columns) != feature_names:
        raise ValueError(
            f"Feature mismatch after construction.\n"
            f"  Got:      {list(ordered.columns)}\n"
            f"  Expected: {feature_names}"
        )

    return ordered
