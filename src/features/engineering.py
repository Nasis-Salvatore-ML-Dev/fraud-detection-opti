import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Add amount_log, amount_zscore, and hour_of_day to df using provided stats."""
    log.debug("Engineering features on DataFrame with %d rows", len(df))
    df = df.copy()
    df["amount_log"] = np.log1p(df["Amount"])
    df["amount_zscore"] = (df["Amount"] - stats["mean"]) / stats["std"]
    df["hour_of_day"] = (df["Time"] % 86400) // 3600
    log.info("Feature engineering complete: %d rows, %d columns", len(df), len(df.columns))
    return df
