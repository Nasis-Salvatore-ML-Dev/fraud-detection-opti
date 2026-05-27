"""Tests for offline SHAP computation and DynamoDB storage / retrieval."""

import hashlib
import os
from decimal import Decimal
from unittest.mock import MagicMock, patch

import boto3
import msgpack
import numpy as np
import pandas as pd
import pytest
from moto import mock_aws

# Fake AWS credentials so boto3 never hits a real endpoint
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

from src.explainability.shap_offline import compute_and_store_shap
from src.explainability.shap_explainer import get_shap_values

_TABLE = "test-shap-store"
_REGION = "us-east-1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_table(region: str = _REGION) -> None:
    dynamo = boto3.resource("dynamodb", region_name=region)
    dynamo.create_table(
        TableName=_TABLE,
        KeySchema=[{"AttributeName": "prediction_hash", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "prediction_hash", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )


def _make_data(n: int = 10, n_features: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    cols = [f"V{i}" for i in range(1, n_features + 1)]
    X = pd.DataFrame(rng.standard_normal((n, n_features)), columns=cols)
    background = X.iloc[:3]
    return X, background


def _make_fake_shap(shap_values: np.ndarray) -> MagicMock:
    """Return a fake shap module whose TreeExplainer returns the given values."""
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = shap_values
    fake_shap = MagicMock()
    fake_shap.TreeExplainer.return_value = mock_explainer
    return fake_shap


def _row_hash(row: pd.Series) -> str:
    return hashlib.sha256(row.values.astype(np.float64).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. compute_and_store_shap stores correct number of items
# ---------------------------------------------------------------------------

@mock_aws
def test_compute_and_store_correct_number(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    _create_table()

    X, background = _make_data(n=10)
    fake_shap_vals = np.random.default_rng(0).standard_normal((10, 5))
    fake_shap = _make_fake_shap(fake_shap_vals)

    with patch.dict("sys.modules", {"shap": fake_shap}):
        stored = compute_and_store_shap(object(), X, background, _TABLE)

    assert stored == 10


# ---------------------------------------------------------------------------
# 2. Each stored item contains prediction_hash, shap_top3, expires_at
# ---------------------------------------------------------------------------

@mock_aws
def test_stored_items_have_required_fields(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    _create_table()

    X, background = _make_data(n=3)
    fake_shap_vals = np.random.default_rng(1).standard_normal((3, 5))
    fake_shap = _make_fake_shap(fake_shap_vals)

    with patch.dict("sys.modules", {"shap": fake_shap}):
        compute_and_store_shap(object(), X, background, _TABLE)

    dynamo = boto3.resource("dynamodb", region_name=_REGION)
    items = dynamo.Table(_TABLE).scan()["Items"]

    assert len(items) == 3
    for item in items:
        assert "prediction_hash" in item, "missing prediction_hash"
        assert "shap_top3" in item, "missing shap_top3"
        assert "expires_at" in item, "missing expires_at"


# ---------------------------------------------------------------------------
# 3. shap_top3 contains exactly 3 items sorted by absolute value descending
# ---------------------------------------------------------------------------

@mock_aws
def test_shap_top3_sorted_by_absolute_value(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    _create_table()

    X, background = _make_data(n=1, n_features=5)
    # Known values: |V1|=3.0, |V2|=2.0, |V3|=1.0, |V4|=0.5, |V5|=0.1
    known_shap = np.array([[3.0, -2.0, 1.0, 0.5, -0.1]])
    fake_shap = _make_fake_shap(known_shap)

    with patch.dict("sys.modules", {"shap": fake_shap}):
        compute_and_store_shap(object(), X, background, _TABLE)

    item = boto3.resource("dynamodb", region_name=_REGION).Table(_TABLE).scan()["Items"][0]
    top3 = item["shap_top3"]

    assert len(top3) == 3, f"expected 3 items, got {len(top3)}"

    abs_vals = [abs(float(t["value"])) for t in top3]
    assert abs_vals == sorted(abs_vals, reverse=True), (
        f"shap_top3 not sorted by absolute value: {abs_vals}"
    )
    assert top3[0]["feature"] == "V1", f"expected V1 as top feature, got {top3[0]['feature']}"


# ---------------------------------------------------------------------------
# 4. get_shap_values returns None when item not found
# ---------------------------------------------------------------------------

@mock_aws
def test_get_shap_values_returns_none_when_not_found(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    _create_table()

    result = get_shap_values("nonexistent_hash_abc123", table_name=_TABLE)
    assert result is None


# ---------------------------------------------------------------------------
# 5. get_shap_values returns correct dict when item exists
# ---------------------------------------------------------------------------

@mock_aws
def test_get_shap_values_returns_correct_dict(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    _create_table()

    X, background = _make_data(n=1, n_features=5)
    known_shap = np.array([[1.5, -0.8, 2.1, 0.3, -1.0]])
    fake_shap = _make_fake_shap(known_shap)

    with patch.dict("sys.modules", {"shap": fake_shap}):
        compute_and_store_shap(object(), X, background, _TABLE)

    prediction_hash = _row_hash(X.iloc[0])
    result = get_shap_values(prediction_hash, table_name=_TABLE)

    assert result is not None, "expected a dict, got None"
    assert isinstance(result, dict)
    assert "V1" in result
    assert abs(result["V1"] - 1.5) < 1e-6, f"V1 shap value mismatch: {result['V1']}"


# ---------------------------------------------------------------------------
# 6. "import shap" does not appear in any file under src/
# ---------------------------------------------------------------------------

def test_import_shap_not_in_src():
    import pathlib

    src_root = pathlib.Path(__file__).resolve().parents[2] / "src"
    violations = [
        str(py_file)
        for py_file in src_root.rglob("*.py")
        if "import shap" in py_file.read_text()
    ]
    assert not violations, (
        f"'import shap' must not appear in any src/ file. Found in: {violations}"
    )
