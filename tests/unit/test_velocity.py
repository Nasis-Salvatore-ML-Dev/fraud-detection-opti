"""Unit tests for src/features/velocity.py and velocity integration in preprocessing."""

import hashlib
import os
from decimal import Decimal
from unittest.mock import MagicMock, patch

import boto3
import numpy as np
import pytest
from moto import mock_aws

# Minimal AWS credentials for moto
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ["VELOCITY_TABLE"] = "fraud-velocity-store"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_table(region: str = "us-east-1") -> "boto3.resource":
    dynamodb = boto3.resource("dynamodb", region_name=region)
    dynamodb.create_table(
        TableName="fraud-velocity-store",
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[{"AttributeName": "card_hash", "AttributeType": "S"}],
        KeySchema=[{"AttributeName": "card_hash", "KeyType": "HASH"}],
    )
    return dynamodb


def _store():
    from src.features.velocity import VelocityStore
    return VelocityStore()


def _fv(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(28).astype(np.float64)


# ---------------------------------------------------------------------------
# 1. get_card_features defaults when DynamoDB unavailable
# ---------------------------------------------------------------------------

@mock_aws
def test_get_card_features_returns_all_5_keys_with_defaults_when_unavailable():
    # Table is NOT created → get_item raises ResourceNotFoundException
    store = _store()
    result = store.get_card_features(_fv())

    assert set(result.keys()) == {
        "tx_count_1h", "tx_count_24h", "tx_count_7d",
        "time_since_last_tx_seconds", "amount_sum_1h",
    }
    assert result["tx_count_1h"] == 0
    assert result["tx_count_24h"] == 0
    assert result["tx_count_7d"] == 0
    assert result["time_since_last_tx_seconds"] == 0
    assert result["amount_sum_1h"] == 0.0


# ---------------------------------------------------------------------------
# 2. get_card_features returns correct values when record exists
# ---------------------------------------------------------------------------

@mock_aws
def test_get_card_features_returns_correct_values_when_record_exists():
    dynamodb = _make_table()
    store = _store()
    fv = _fv(seed=1)
    card_hash = hashlib.sha256(fv.tobytes()).hexdigest()

    dynamodb.Table("fraud-velocity-store").put_item(Item={
        "card_hash": card_hash,
        "tx_count_1h": 3,
        "tx_count_24h": 7,
        "tx_count_7d": 15,
        "time_since_last_tx_seconds": Decimal("120.5"),
        "amount_sum_1h": Decimal("456.78"),
    })

    result = store.get_card_features(fv)

    assert result["tx_count_1h"] == 3
    assert result["tx_count_24h"] == 7
    assert result["tx_count_7d"] == 15
    assert result["time_since_last_tx_seconds"] == pytest.approx(120.5)
    assert result["amount_sum_1h"] == pytest.approx(456.78)


# ---------------------------------------------------------------------------
# 3. update_card_features writes correct record structure
# ---------------------------------------------------------------------------

@mock_aws
def test_update_card_features_writes_correct_record_structure():
    dynamodb = _make_table()
    store = _store()
    fv = _fv(seed=2)
    ts = 1_700_000_000.0

    store.update_card_features(fv, amount=250.0, timestamp=ts)

    card_hash = hashlib.sha256(fv.tobytes()).hexdigest()
    item = dynamodb.Table("fraud-velocity-store").get_item(
        Key={"card_hash": card_hash}
    )["Item"]

    assert item["card_hash"] == card_hash
    assert "tx_timestamps" in item
    assert "amount_history" in item
    assert "tx_count_1h" in item
    assert "tx_count_24h" in item
    assert "tx_count_7d" in item
    assert "time_since_last_tx_seconds" in item
    assert "amount_sum_1h" in item
    assert "last_updated" in item
    assert "expires_at" in item
    assert int(item["expires_at"]) == int(ts) + 604800


# ---------------------------------------------------------------------------
# 4. tx_count_1h counts only transactions within last 3600 s
# ---------------------------------------------------------------------------

@mock_aws
def test_tx_count_1h_counts_only_transactions_within_last_3600s():
    _make_table()
    store = _store()
    fv = _fv(seed=3)

    now = 1_700_000_000.0
    store.update_card_features(fv, amount=10.0, timestamp=now - 7200)  # outside 1h
    store.update_card_features(fv, amount=20.0, timestamp=now - 1800)  # inside 1h
    store.update_card_features(fv, amount=30.0, timestamp=now)         # inside 1h

    result = store.get_card_features(fv)
    assert result["tx_count_1h"] == 2


# ---------------------------------------------------------------------------
# 5. tx_count_24h counts only transactions within last 86400 s
# ---------------------------------------------------------------------------

@mock_aws
def test_tx_count_24h_counts_only_transactions_within_last_86400s():
    _make_table()
    store = _store()
    fv = _fv(seed=4)

    now = 1_700_000_000.0
    store.update_card_features(fv, amount=10.0, timestamp=now - 90000)  # outside 24h
    store.update_card_features(fv, amount=20.0, timestamp=now - 3600)   # inside 24h
    store.update_card_features(fv, amount=30.0, timestamp=now)          # inside 24h

    result = store.get_card_features(fv)
    assert result["tx_count_24h"] == 2


# ---------------------------------------------------------------------------
# 6. time_since_last_tx_seconds computed correctly
# ---------------------------------------------------------------------------

@mock_aws
def test_time_since_last_tx_seconds_computed_correctly():
    _make_table()
    store = _store()
    fv = _fv(seed=5)

    t1 = 1_700_000_000.0
    t2 = t1 + 300.0  # 5 minutes later

    store.update_card_features(fv, amount=50.0, timestamp=t1)
    store.update_card_features(fv, amount=75.0, timestamp=t2)

    result = store.get_card_features(fv)
    assert result["time_since_last_tx_seconds"] == pytest.approx(300.0, abs=0.01)


# ---------------------------------------------------------------------------
# 7. amount_sum_1h sums only amounts within last 3600 s
# ---------------------------------------------------------------------------

@mock_aws
def test_amount_sum_1h_sums_only_amounts_within_last_3600s():
    _make_table()
    store = _store()
    fv = _fv(seed=6)

    now = 1_700_000_000.0
    store.update_card_features(fv, amount=100.0, timestamp=now - 7200)  # outside 1h
    store.update_card_features(fv, amount=200.0, timestamp=now - 1800)  # inside 1h
    store.update_card_features(fv, amount=50.0,  timestamp=now)         # inside 1h

    result = store.get_card_features(fv)
    assert result["amount_sum_1h"] == pytest.approx(250.0, abs=0.01)


# ---------------------------------------------------------------------------
# 8. card_hash is SHA-256 of feature_vector.tobytes()
# ---------------------------------------------------------------------------

def test_card_hash_is_sha256_of_feature_vector_tobytes():
    from src.features.velocity import VelocityStore

    fv = np.array([1.0, -2.5] + [0.0] * 26, dtype=np.float64)
    expected = hashlib.sha256(fv.tobytes()).hexdigest()
    assert VelocityStore._card_hash(fv) == expected


# ---------------------------------------------------------------------------
# 9. update_card_features never raises on DynamoDB failure
# ---------------------------------------------------------------------------

@mock_aws
def test_update_card_features_never_raises_on_dynamodb_failure():
    # Table not created → DynamoDB call will fail
    store = _store()
    fv = _fv(seed=7)
    # Must not raise
    store.update_card_features(fv, amount=99.0, timestamp=1_700_000_000.0)


# ---------------------------------------------------------------------------
# 10. preprocessing.py appends 5 velocity features to the DataFrame
# ---------------------------------------------------------------------------

def test_preprocessing_appends_5_velocity_features_to_dataframe():
    from src.api import preprocessing as prep
    from src.api.schemas import PredictionRequest

    prep.set_amount_stats({"mean": 90.0, "std": 50.0})

    mock_vel = MagicMock()
    mock_vel.get_card_features.return_value = {
        "tx_count_1h": 2,
        "tx_count_24h": 8,
        "tx_count_7d": 25,
        "time_since_last_tx_seconds": 300.0,
        "amount_sum_1h": 450.0,
    }

    feature_names = [f"V{i}" for i in range(1, 29)] + [
        "Amount", "amount_log", "amount_zscore", "hour_of_day",
        "tx_count_1h", "tx_count_24h", "tx_count_7d",
        "time_since_last_tx_seconds", "amount_sum_1h",
    ]

    request = PredictionRequest(
        **{f"v{i}": float(i) * 0.1 for i in range(1, 29)},
        amount=150.0,
        time=3600.0,
    )

    with patch.object(prep, "velocity_store", mock_vel):
        df = prep.build_feature_dataframe(request, feature_names)

    assert df.shape[1] == len(feature_names)
    assert "tx_count_1h" in df.columns
    assert df["tx_count_1h"].iloc[0] == 2
    assert df["amount_sum_1h"].iloc[0] == pytest.approx(450.0)
    mock_vel.get_card_features.assert_called_once()


# ---------------------------------------------------------------------------
# 11. Training and serving feature counts are identical
# ---------------------------------------------------------------------------

def test_training_and_serving_feature_counts_are_identical():
    from src.features.velocity import _VELOCITY_KEYS
    from src.api import preprocessing as prep
    from src.api.schemas import PredictionRequest

    prep.set_amount_stats({"mean": 90.0, "std": 50.0})

    # Build the feature_names list exactly as train.py does
    v_cols = [f"V{i}" for i in range(1, 29)]
    base_engineered = ["Amount", "amount_log", "amount_zscore", "hour_of_day"]
    training_feature_names = v_cols + base_engineered + list(_VELOCITY_KEYS)

    mock_vel = MagicMock()
    mock_vel.get_card_features.return_value = {k: 0 for k in _VELOCITY_KEYS}

    request = PredictionRequest(
        **{f"v{i}": float(i) * 0.1 for i in range(1, 29)},
        amount=100.0,
        time=0.0,
    )

    with patch.object(prep, "velocity_store", mock_vel):
        df = prep.build_feature_dataframe(request, training_feature_names)

    assert list(df.columns) == training_feature_names, (
        f"Serving columns do not match training feature_names.\n"
        f"  Serving:  {list(df.columns)}\n"
        f"  Training: {training_feature_names}"
    )
    assert len(df.columns) == len(v_cols) + len(base_engineered) + len(_VELOCITY_KEYS)
