"""Tests for high-value SNS escalation and ComponentFailure CloudWatch metric."""

import os
from unittest.mock import AsyncMock, MagicMock

import boto3
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

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


def _make_normal_client(monkeypatch, fraud_probability: float = 0.7):
    """Build a TestClient in normal mode with a configurable model probability."""
    import src.api.app as app_module
    import src.utils.model_loader as ml

    bundle = MagicMock()
    bundle.version = "v1"
    bundle.threshold = 0.5
    bundle.feature_names = _FEATURE_NAMES
    bundle.amount_stats = {}
    bundle.psi_baseline = {}
    bundle.calibrator = None
    bundle.model.predict_proba.return_value = np.array([[1 - fraud_probability, fraud_probability]])

    mock_audit = MagicMock()
    mock_audit.write = AsyncMock()
    mock_audit.fetch_pending_reviews = AsyncMock(return_value=[])
    mock_audit.fetch_recent = AsyncMock(return_value=[])

    monkeypatch.setattr(app_module, "_FALLBACK_MODE", False)
    monkeypatch.setattr("src.api.app.load_model_bundle", lambda: bundle)
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
    return TestClient(app)


def _sync_threading_module():
    """Return a namespace that replaces threading in app.py with a sync Thread.

    Using a module-level replacement (not patching threading.Thread globally)
    avoids breaking anyio's WorkerThread in other tests.
    """
    class _SyncThread:
        def __init__(self, target, daemon=False, **kwargs):
            self._target = target

        def start(self):
            self._target()

    class _FakeThreading:
        Thread = _SyncThread

    return _FakeThreading()


# ---------------------------------------------------------------------------
# 1. high_value_alert=True when amount > 1000 AND fraud_probability > 0.3
# ---------------------------------------------------------------------------

def test_high_value_alert_true_when_conditions_met(monkeypatch):
    # Suppress the actual SNS call — this test only checks the response field.
    monkeypatch.setattr("src.api.app._publish_sns_alert", lambda *a, **kw: None)

    with _make_normal_client(monkeypatch, fraud_probability=0.7) as c:
        resp = c.post("/predict", json=_payload(amount=1500.0))

    assert resp.status_code == 200
    assert resp.json()["high_value_alert"] is True


# ---------------------------------------------------------------------------
# 2. high_value_alert=False when amount <= 1000
# ---------------------------------------------------------------------------

def test_high_value_alert_false_when_low_amount(monkeypatch):
    monkeypatch.setattr("src.api.app._publish_sns_alert", lambda *a, **kw: None)

    with _make_normal_client(monkeypatch, fraud_probability=0.7) as c:
        resp = c.post("/predict", json=_payload(amount=500.0))

    assert resp.status_code == 200
    assert resp.json()["high_value_alert"] is False


# ---------------------------------------------------------------------------
# 3. high_value_alert=False when fraud_probability <= 0.3
# ---------------------------------------------------------------------------

def test_high_value_alert_false_when_low_probability(monkeypatch):
    monkeypatch.setattr("src.api.app._publish_sns_alert", lambda *a, **kw: None)

    with _make_normal_client(monkeypatch, fraud_probability=0.25) as c:
        resp = c.post("/predict", json=_payload(amount=1500.0))

    assert resp.status_code == 200
    assert resp.json()["high_value_alert"] is False


# ---------------------------------------------------------------------------
# 4. SNS publish is called when high_value_alert conditions are met
# ---------------------------------------------------------------------------

def test_high_value_alert_publishes_to_sns(monkeypatch):
    import src.api.app as app_module

    published = []
    mock_sns_client = MagicMock()
    mock_sns_client.publish.side_effect = lambda **kw: published.append(kw)

    # Patch threading in app.py's namespace so _do_publish runs synchronously.
    # This avoids touching the global threading.Thread used by anyio/WorkerThread.
    monkeypatch.setattr(app_module, "threading", _sync_threading_module())

    # Patch boto3.client in app.py's namespace for the SNS call.
    real_boto3_client = app_module.boto3.client

    def _patched_client(service, **kw):
        if service == "sns":
            return mock_sns_client
        return real_boto3_client(service, **kw)

    monkeypatch.setattr(app_module.boto3, "client", _patched_client)
    monkeypatch.setenv("HIGH_VALUE_SNS_ARN", "arn:aws:sns:us-east-1:123456789012:fraud-alerts")

    with _make_normal_client(monkeypatch, fraud_probability=0.7) as c:
        resp = c.post("/predict", json=_payload(amount=1500.0))

    assert resp.status_code == 200
    assert resp.json()["high_value_alert"] is True
    assert len(published) == 1
    assert published[0]["TopicArn"] == "arn:aws:sns:us-east-1:123456789012:fraud-alerts"


# ---------------------------------------------------------------------------
# 5. No exception when HIGH_VALUE_SNS_ARN is not set
# ---------------------------------------------------------------------------

def test_no_exception_without_sns_arn(monkeypatch):
    monkeypatch.delenv("HIGH_VALUE_SNS_ARN", raising=False)
    monkeypatch.setattr("src.api.app._publish_sns_alert", lambda *a, **kw: None)

    with _make_normal_client(monkeypatch, fraud_probability=0.7) as c:
        resp = c.post("/predict", json=_payload(amount=1500.0))

    assert resp.status_code == 200
    assert resp.json()["high_value_alert"] is True


# ---------------------------------------------------------------------------
# 6. publish_component_failure never raises
# ---------------------------------------------------------------------------

def test_publish_component_failure_never_raises():
    from src.monitoring.metrics import publish_component_failure

    try:
        publish_component_failure("TestComponent")
    except Exception as exc:
        pytest.fail(f"publish_component_failure raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# 7. publish_component_failure publishes the ComponentFailure metric
# ---------------------------------------------------------------------------

@mock_aws
def test_publish_component_failure_publishes_metric(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")

    from src.monitoring.metrics import publish_component_failure

    publish_component_failure("TestComponent")

    cw = boto3.client("cloudwatch", region_name="us-east-1")
    stats = cw.get_metric_statistics(
        Namespace="FraudDetection",
        MetricName="ComponentFailure",
        Dimensions=[{"Name": "ComponentName", "Value": "TestComponent"}],
        StartTime="2000-01-01T00:00:00Z",
        EndTime="2100-01-01T00:00:00Z",
        Period=3600,
        Statistics=["Sum"],
    )
    assert stats["Datapoints"], "Expected at least one datapoint for ComponentFailure metric"
    assert stats["Datapoints"][0]["Sum"] == 1.0
