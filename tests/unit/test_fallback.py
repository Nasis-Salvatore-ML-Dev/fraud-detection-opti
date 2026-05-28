"""Tests for the rule-based fallback scorer."""

import os
import time as _time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

# Training-set normalisation constants (must match app.py / bias_tester.py)
_AMOUNT_MEAN = 90.8249
_AMOUNT_STD = 250.5032

_FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["amount_log", "amount_zscore", "hour_of_day"]


def _payload(**overrides):
    base = {"time": 43200.0, "amount": 100.0, **{f"v{i}": 0.0 for i in range(1, 29)}}
    base.update(overrides)
    return base


def _mock_features(p, fns):
    data = {col: [0.0] for col in fns}
    data["hour_of_day"] = [12.0]
    data["amount_log"] = [float(np.log1p(p.amount))]
    data["amount_zscore"] = [0.0]
    return pd.DataFrame(data)


def _make_mock_bundle():
    bundle = MagicMock()
    bundle.version = "v1"
    bundle.threshold = 0.5
    bundle.feature_names = _FEATURE_NAMES
    bundle.amount_stats = {}
    bundle.psi_baseline = {}
    bundle.calibrator = None
    bundle.model.predict_proba.return_value = np.array([[0.8, 0.2]])
    return bundle


@pytest.fixture
def client_fallback(monkeypatch):
    """TestClient where model bundle always raises — fallback mode active."""
    import src.api.app as app_module
    import src.utils.model_loader as ml

    mock_audit = MagicMock()
    mock_audit.write = AsyncMock()
    mock_audit.fetch_pending_reviews = AsyncMock(return_value=[])
    mock_audit.fetch_recent = AsyncMock(return_value=[])

    monkeypatch.setattr(app_module, "_FALLBACK_MODE", False)
    monkeypatch.setattr("src.api.app.load_model_bundle", MagicMock(side_effect=RuntimeError("no model")))
    monkeypatch.setattr("src.api.app.set_amount_stats", lambda x: None)
    monkeypatch.setattr("src.api.app.register_validation_stats", lambda x: None)
    monkeypatch.setattr("src.api.app.AuditLogger", lambda: mock_audit)
    monkeypatch.setattr("src.api.app.DriftMonitor", MagicMock)
    monkeypatch.setattr("src.api.app.BiasTestSuite", MagicMock)
    monkeypatch.setattr("src.api.app.publish_component_failure", lambda name: None)
    monkeypatch.setattr(ml, "_load_threshold_from_dynamodb", lambda x: None)
    monkeypatch.setattr(_time, "sleep", lambda n: None)

    from src.api.app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_normal(monkeypatch):
    """TestClient with a working model bundle — normal mode."""
    import src.api.app as app_module
    import src.utils.model_loader as ml

    mock_bundle = _make_mock_bundle()
    mock_audit = MagicMock()
    mock_audit.write = AsyncMock()
    mock_audit.fetch_pending_reviews = AsyncMock(return_value=[])
    mock_audit.fetch_recent = AsyncMock(return_value=[])

    monkeypatch.setattr(app_module, "_FALLBACK_MODE", False)
    monkeypatch.setattr("src.api.app.load_model_bundle", lambda: mock_bundle)
    monkeypatch.setattr("src.api.app.set_amount_stats", lambda x: None)
    monkeypatch.setattr("src.api.app.register_validation_stats", lambda x: None)
    monkeypatch.setattr("src.api.app.AuditLogger", lambda: mock_audit)
    monkeypatch.setattr("src.api.app.DriftMonitor", MagicMock)
    monkeypatch.setattr("src.api.app.BiasTestSuite", MagicMock)
    monkeypatch.setattr("src.api.app.build_feature_dataframe", _mock_features)
    monkeypatch.setattr(ml, "_load_threshold_from_dynamodb", lambda x: None)
    monkeypatch.setattr(ml, "_active_threshold", 0.5)
    monkeypatch.setattr(ml, "_threshold_source", "dynamodb")

    from src.api.app import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. /predict returns 200 in fallback mode
# ---------------------------------------------------------------------------

def test_predict_returns_200_in_fallback_mode(client_fallback):
    resp = client_fallback.post("/predict", json=_payload())
    assert resp.status_code == 200
    assert resp.json()["fallback_mode"] is True


# ---------------------------------------------------------------------------
# 2. is_fraud=True when amount_zscore > 3 (high amount, non-risk hour)
# ---------------------------------------------------------------------------

def test_fallback_is_fraud_true_when_amount_zscore_above_3(client_fallback):
    # amount_zscore = (843 - 90.8249) / 250.5032 ≈ 3.002 > 3
    # hour_of_day=12 is not in {1..5}
    amount = round(_AMOUNT_MEAN + 3.01 * _AMOUNT_STD, 2)  # ≈ 844.06
    resp = client_fallback.post("/predict", json=_payload(amount=amount, time=43200.0))
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_fraud"] is True
    assert abs(data["fraud_probability"] - 0.75) < 1e-9


# ---------------------------------------------------------------------------
# 3. is_fraud=True when hour_of_day=3 (low amount, risk hour)
# ---------------------------------------------------------------------------

def test_fallback_is_fraud_true_when_hour_of_day_3(client_fallback):
    # time=10800 → hour_of_day=3 ∈ {1,2,3,4,5}; amount_zscore ≈ 0.037 (not > 3)
    resp = client_fallback.post("/predict", json=_payload(amount=100.0, time=10800.0))
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_fraud"] is True
    assert abs(data["fraud_probability"] - 0.75) < 1e-9


# ---------------------------------------------------------------------------
# 4. is_fraud=False for normal hour and low amount
# ---------------------------------------------------------------------------

def test_fallback_is_fraud_false_normal(client_fallback):
    # time=43200 → hour_of_day=12; amount_zscore ≈ 0.037
    resp = client_fallback.post("/predict", json=_payload(amount=100.0, time=43200.0))
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_fraud"] is False
    assert abs(data["fraud_probability"] - 0.05) < 1e-9


# ---------------------------------------------------------------------------
# 5. fallback_mode=False in normal operation
# ---------------------------------------------------------------------------

def test_fallback_mode_false_in_normal_operation(client_normal):
    resp = client_normal.post("/predict", json=_payload())
    assert resp.status_code == 200
    assert resp.json()["fallback_mode"] is False
