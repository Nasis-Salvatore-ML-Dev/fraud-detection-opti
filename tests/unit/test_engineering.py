import os

import numpy as np
import pandas as pd
import pytest

# Fake AWS credentials so boto3 / moto don't complain
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

from moto import mock_aws  # noqa: E402

from src.features.engineering import engineer_features  # noqa: E402
from src.features.velocity import VelocityStore  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Amount": [100.0, 200.0, 50.0, 150.0, 75.0],
            "Time": [0.0, 3600.0, 7200.0, 43200.0, 86399.0],
        }
    )


def _stats_from(df: pd.DataFrame) -> dict:
    return {
        "mean": float(df["Amount"].mean()),
        "std": float(df["Amount"].std()),
    }


# ---------------------------------------------------------------------------
# engineer_features tests
# ---------------------------------------------------------------------------

def test_engineer_features_produces_correct_columns():
    df = _sample_df()
    result = engineer_features(df, _stats_from(df))
    assert "amount_log" in result.columns
    assert "amount_zscore" in result.columns
    assert "hour_of_day" in result.columns
    # original columns must survive
    assert "Amount" in result.columns
    assert "Time" in result.columns


def test_engineer_features_does_not_mutate_input():
    df = _sample_df()
    original_cols = list(df.columns)
    engineer_features(df, _stats_from(df))
    assert list(df.columns) == original_cols


def test_amount_log_is_log1p_of_amount():
    df = _sample_df()
    result = engineer_features(df, _stats_from(df))
    expected = np.log1p(df["Amount"].values)
    np.testing.assert_allclose(result["amount_log"].values, expected)


def test_amount_zscore_zero_mean_on_training_stats():
    df = _sample_df()
    stats = _stats_from(df)
    result = engineer_features(df, stats)
    assert abs(result["amount_zscore"].mean()) < 1e-10


def test_hour_of_day_in_range():
    df = _sample_df()
    result = engineer_features(df, _stats_from(df))
    assert result["hour_of_day"].between(0, 23).all()


def test_hour_of_day_wraps_correctly():
    df = pd.DataFrame(
        {
            "Amount": [50.0, 50.0],
            "Time": [0.0, 86400.0],  # midnight of day 0 and day 1
        }
    )
    result = engineer_features(df, {"mean": 50.0, "std": 1.0})
    # both map to hour 0
    np.testing.assert_allclose(result["hour_of_day"].values, [0.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# VelocityStore tests
# ---------------------------------------------------------------------------

@mock_aws
def test_velocity_store_returns_zeros_when_table_missing():
    """Table is not created inside the mock — simulates an unavailable store."""
    store = VelocityStore()
    vec = np.array([1.0, -0.5, 2.3], dtype=np.float64)
    result = store.get_card_features(vec)

    expected_keys = {
        "tx_count_1h",
        "tx_count_24h",
        "tx_count_7d",
        "time_since_last_tx_seconds",
        "amount_sum_1h",
    }
    assert set(result.keys()) == expected_keys
    for key, val in result.items():
        assert val == 0 or val == 0.0, f"{key} expected 0, got {val}"


@mock_aws
def test_velocity_store_get_returns_all_required_keys():
    """Even when DynamoDB is unavailable, the dict must have all five keys."""
    store = VelocityStore()
    vec = np.zeros(10, dtype=np.float64)
    result = store.get_card_features(vec)
    assert "tx_count_1h" in result
    assert "tx_count_24h" in result
    assert "tx_count_7d" in result
    assert "time_since_last_tx_seconds" in result
    assert "amount_sum_1h" in result


@mock_aws
def test_velocity_store_update_does_not_raise_when_table_missing():
    """update_card_features must never propagate an exception."""
    store = VelocityStore()
    vec = np.array([0.1, 0.2], dtype=np.float64)
    store.update_card_features(vec, amount=99.5, timestamp=1_700_000_000.0)
    # reaching here without exception is the pass condition
