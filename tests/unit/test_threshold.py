"""Tests for runtime threshold management, calibrated confidence, and time-aware tightening."""

import os
from unittest.mock import AsyncMock, MagicMock

import boto3
import numpy as np
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

# Fake credentials so boto3 never hits a real endpoint
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

_CONFIG_TABLE = "fraud-config"
_REGION = "us-east-1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_config_table(region: str = _REGION) -> None:
    boto3.resource("dynamodb", region_name=region).create_table(
        TableName=_CONFIG_TABLE,
        KeySchema=[{"AttributeName": "config_key", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "config_key", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )


def _seed_threshold(value: float = 0.5, region: str = _REGION) -> None:
    boto3.resource("dynamodb", region_name=region).Table(_CONFIG_TABLE).put_item(
        Item={"config_key": "threshold", "value": str(value)}
    )


def _make_mock_bundle():
    mock_calibrator = MagicMock()
    mock_calibrator.transform.return_value = np.array([0.75])

    bundle = MagicMock()
    bundle.version = "v1"
    bundle.threshold = 0.5
    bundle.feature_names = [f"V{i}" for i in range(1, 29)] + ["amount_log", "amount_zscore", "hour_of_day"]
    bundle.amount_stats = {}
    bundle.psi_baseline = {}
    bundle.calibrator = mock_calibrator
    return bundle


@pytest.fixture
def client(monkeypatch):
    """TestClient with all external services mocked so the lifespan completes safely."""
    import src.utils.model_loader as ml

    mock_bundle = _make_mock_bundle()
    mock_audit = MagicMock()
    mock_audit.write = AsyncMock()
    mock_audit.fetch_pending_reviews = AsyncMock(return_value=[])
    mock_audit.fetch_recent = AsyncMock(return_value=[])

    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    monkeypatch.setattr("src.api.app.load_model_bundle", lambda: mock_bundle)
    monkeypatch.setattr("src.api.app.set_amount_stats", lambda x: None)
    monkeypatch.setattr("src.api.app.register_validation_stats", lambda x: None)
    monkeypatch.setattr("src.api.app.AuditLogger", lambda: mock_audit)
    monkeypatch.setattr("src.api.app.DriftMonitor", MagicMock)
    monkeypatch.setattr("src.api.app.BiasTestSuite", MagicMock)
    monkeypatch.setattr(ml, "_load_threshold_from_dynamodb", lambda x: None)
    monkeypatch.setattr(ml, "_active_threshold", 0.5)
    monkeypatch.setattr(ml, "_threshold_source", "dynamodb")

    from src.api.app import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. GET /config returns current threshold and source
# ---------------------------------------------------------------------------

def test_get_config_returns_threshold_and_source(client):
    resp = client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["threshold"] == 0.5
    assert data["source"] == "dynamodb"


# ---------------------------------------------------------------------------
# 2. POST /config updates threshold in DynamoDB
# ---------------------------------------------------------------------------

@mock_aws
def test_post_config_updates_threshold_in_dynamodb(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    monkeypatch.setenv("CONFIG_TABLE", _CONFIG_TABLE)
    monkeypatch.setenv("CONFIG_API_KEY", "test-secret")

    _create_config_table()
    _seed_threshold(0.5)

    import src.utils.model_loader as ml

    mock_audit = MagicMock()
    mock_audit.write = AsyncMock()
    mock_audit.fetch_pending_reviews = AsyncMock(return_value=[])
    mock_audit.fetch_recent = AsyncMock(return_value=[])

    mock_bundle = _make_mock_bundle()
    monkeypatch.setattr("src.api.app.load_model_bundle", lambda: mock_bundle)
    monkeypatch.setattr("src.api.app.set_amount_stats", lambda x: None)
    monkeypatch.setattr("src.api.app.register_validation_stats", lambda x: None)
    monkeypatch.setattr("src.api.app.AuditLogger", lambda: mock_audit)
    monkeypatch.setattr("src.api.app.DriftMonitor", MagicMock)
    monkeypatch.setattr("src.api.app.BiasTestSuite", MagicMock)
    monkeypatch.setattr(ml, "_load_threshold_from_dynamodb", lambda x: None)
    monkeypatch.setattr(ml, "_active_threshold", 0.5)
    monkeypatch.setattr(ml, "_threshold_source", "dynamodb")

    from src.api.app import app
    with TestClient(app) as c:
        resp = c.post(
            "/config",
            json={"threshold": 0.7},
            headers={"X-Config-Api-Key": "test-secret"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["updated"] is True
    assert abs(data["threshold"] - 0.7) < 1e-9

    # Verify the item was actually written to DynamoDB
    item = boto3.resource("dynamodb", region_name=_REGION).Table(_CONFIG_TABLE).get_item(
        Key={"config_key": "threshold"}
    )["Item"]
    assert item["value"] == "0.7"


# ---------------------------------------------------------------------------
# 3. POST /config returns 403 with wrong API key
# ---------------------------------------------------------------------------

def test_post_config_returns_403_with_wrong_key(client, monkeypatch):
    monkeypatch.setenv("CONFIG_API_KEY", "correct-secret")
    resp = client.post(
        "/config",
        json={"threshold": 0.6},
        headers={"X-Config-Api-Key": "wrong-key"},
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# 4. POST /config returns 422 when threshold outside (0, 1)
# ---------------------------------------------------------------------------

def test_post_config_returns_422_when_threshold_out_of_range(client, monkeypatch):
    monkeypatch.setenv("CONFIG_API_KEY", "secret")

    resp = client.post(
        "/config",
        json={"threshold": 0.0},
        headers={"X-Config-Api-Key": "secret"},
    )
    assert resp.status_code == 422, f"Expected 422 for threshold=0.0, got {resp.status_code}"

    resp = client.post(
        "/config",
        json={"threshold": 1.5},
        headers={"X-Config-Api-Key": "secret"},
    )
    assert resp.status_code == 422, f"Expected 422 for threshold=1.5, got {resp.status_code}"


# ---------------------------------------------------------------------------
# 5. POST /config returns 503 when CONFIG_API_KEY not set
# ---------------------------------------------------------------------------

def test_post_config_returns_503_when_api_key_not_set(client, monkeypatch):
    monkeypatch.delenv("CONFIG_API_KEY", raising=False)
    resp = client.post(
        "/config",
        json={"threshold": 0.6},
        headers={"X-Config-Api-Key": "any-key"},
    )
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# 6. Calibrated confidence score differs from raw probability
# ---------------------------------------------------------------------------

def test_calibrated_confidence_differs_from_raw_probability():
    """Isotonic calibrator must map raw probability to a different output."""
    from sklearn.isotonic import IsotonicRegression

    cal = IsotonicRegression(out_of_bounds="clip")
    # raw 0.3 maps to calibrated 0.7 via this monotone fit
    cal.fit([0.0, 0.3, 1.0], [0.0, 0.7, 1.0])

    raw_prob = 0.3
    calibrated = float(cal.transform([raw_prob])[0])

    assert abs(calibrated - 0.7) < 1e-6, f"Expected 0.7, got {calibrated}"
    assert calibrated != raw_prob


# ---------------------------------------------------------------------------
# 7. effective_threshold = _active_threshold * 0.85 when hour_of_day = 3
# ---------------------------------------------------------------------------

def test_effective_threshold_tightened_when_hour_3():
    from src.api.app import _effective_threshold

    result = _effective_threshold(0.5, 3)
    expected = round(0.5 * 0.85, 4)
    assert result == expected, f"Expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# 8. effective_threshold = _active_threshold when hour_of_day = 12
# ---------------------------------------------------------------------------

def test_effective_threshold_unchanged_when_hour_12():
    from src.api.app import _effective_threshold

    result = _effective_threshold(0.5, 12)
    assert result == 0.5, f"Expected 0.5, got {result}"


# ---------------------------------------------------------------------------
# 9. Old 0.9/0.4 band proxy not present in src/api/app.py
# ---------------------------------------------------------------------------

def test_old_band_proxy_not_in_app():
    import pathlib

    app_path = pathlib.Path(__file__).resolve().parents[2] / "src" / "api" / "app.py"
    content = app_path.read_text()

    assert "0.9 if" not in content, "Old 0.9 band proxy still present in app.py"
    assert "else 0.4" not in content, "Old 0.4 band proxy still present in app.py"
