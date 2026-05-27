"""Tests for AuditLogger: full-feature storage, review flags, and SHAP integration."""

import asyncio
import os
import time
from decimal import Decimal

import boto3
import msgpack
import pytest
from moto import mock_aws

# Fake credentials so boto3 never hits a real endpoint
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

from src.monitoring.audit_logger import AuditLogger, extract_v_features

_AUDIT_TABLE = "fraud-audit-log"
_SHAP_TABLE = "fraud-shap-store"
_REGION = "us-east-1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_audit_table(region: str = _REGION) -> None:
    boto3.resource("dynamodb", region_name=region).create_table(
        TableName=_AUDIT_TABLE,
        KeySchema=[{"AttributeName": "prediction_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "prediction_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )


def _create_shap_table(region: str = _REGION) -> None:
    boto3.resource("dynamodb", region_name=region).create_table(
        TableName=_SHAP_TABLE,
        KeySchema=[{"AttributeName": "prediction_hash", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "prediction_hash", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )


def _make_input_features() -> dict:
    return {
        "Amount": 100.0,
        "amount_log": 4.615,
        "amount_zscore": 0.5,
        "hour_of_day": 10.0,
        **{f"V{i}": float(i) * 0.1 for i in range(1, 29)},
    }


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _write(audit: AuditLogger, **overrides):
    defaults = dict(
        prediction_id="pred-001",
        prediction_hash="hash-abc-001",
        input_features=_make_input_features(),
        fraud_probability=0.1,
        is_fraud=False,
        shap_values={},
        model_version="v1",
        confidence_score=0.9,
        request_ip="127.0.0.1",
        latency_ms=5.0,
        threshold_used=0.5,
    )
    defaults.update(overrides)
    await audit.write(**defaults)


def _scan_item(region: str = _REGION) -> dict:
    items = boto3.resource("dynamodb", region_name=region).Table(_AUDIT_TABLE).scan()["Items"]
    assert len(items) == 1
    return items[0]


# ---------------------------------------------------------------------------
# 1. Audit record contains all 31 features
# ---------------------------------------------------------------------------

@mock_aws
def test_audit_record_contains_all_31_features(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    monkeypatch.setenv("AUDIT_TABLE", _AUDIT_TABLE)
    _create_audit_table()

    _run(_write(AuditLogger()))
    item = _scan_item()

    for field in ("Amount", "amount_log", "amount_zscore", "hour_of_day"):
        assert field in item, f"named field missing: {field}"

    assert "v_features_msgpack" in item
    v_dict = msgpack.unpackb(bytes(item["v_features_msgpack"]), raw=False)
    assert len(v_dict) == 28, f"expected 28 V features, got {len(v_dict)}"
    for i in range(1, 29):
        assert f"V{i}" in v_dict, f"V{i} missing from msgpack dict"


# ---------------------------------------------------------------------------
# 2. v_features_msgpack deserialises back to correct V1-V28 values
# ---------------------------------------------------------------------------

@mock_aws
def test_v_features_msgpack_deserialises_correctly(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    monkeypatch.setenv("AUDIT_TABLE", _AUDIT_TABLE)
    _create_audit_table()

    _run(_write(AuditLogger()))
    item = _scan_item()
    v_dict = msgpack.unpackb(bytes(item["v_features_msgpack"]), raw=False)

    for i in range(1, 29):
        expected = float(i) * 0.1
        actual = v_dict[f"V{i}"]
        assert abs(actual - expected) < 1e-6, f"V{i}: expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# 3. requires_review True when fraud_probability = 0.5
# ---------------------------------------------------------------------------

@mock_aws
def test_requires_review_true_when_fraud_probability_05(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    monkeypatch.setenv("AUDIT_TABLE", _AUDIT_TABLE)
    _create_audit_table()

    _run(_write(AuditLogger(), fraud_probability=0.5))
    item = _scan_item()

    assert item["requires_review"] == "true"


# ---------------------------------------------------------------------------
# 4. requires_review False when fraud_probability = 0.1
# ---------------------------------------------------------------------------

@mock_aws
def test_requires_review_false_when_fraud_probability_01(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    monkeypatch.setenv("AUDIT_TABLE", _AUDIT_TABLE)
    _create_audit_table()

    _run(_write(AuditLogger(), fraud_probability=0.1))
    item = _scan_item()

    assert item["requires_review"] == "false"


# ---------------------------------------------------------------------------
# 5. review_expires_at present when requires_review True
# ---------------------------------------------------------------------------

@mock_aws
def test_review_expires_at_present_when_requires_review(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    monkeypatch.setenv("AUDIT_TABLE", _AUDIT_TABLE)
    _create_audit_table()

    _run(_write(AuditLogger(), fraud_probability=0.5))
    item = _scan_item()

    assert "review_expires_at" in item
    expected_approx = int(time.time()) + 30 * 24 * 3600
    assert abs(int(item["review_expires_at"]) - expected_approx) < 30


# ---------------------------------------------------------------------------
# 6. review_expires_at absent when requires_review False
# ---------------------------------------------------------------------------

@mock_aws
def test_review_expires_at_absent_when_not_requires_review(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    monkeypatch.setenv("AUDIT_TABLE", _AUDIT_TABLE)
    _create_audit_table()

    _run(_write(AuditLogger(), fraud_probability=0.1))
    item = _scan_item()

    assert "review_expires_at" not in item


# ---------------------------------------------------------------------------
# 7. shap_top3 is empty list when SHAP lookup returns None
# ---------------------------------------------------------------------------

@mock_aws
def test_shap_top3_empty_when_shap_not_found(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    monkeypatch.setenv("AUDIT_TABLE", _AUDIT_TABLE)
    # No shap table → lookup returns None (exception caught internally)
    _create_audit_table()

    _run(_write(AuditLogger(), prediction_hash="hash-with-no-shap"))
    item = _scan_item()

    assert item.get("shap_top3") == []


# ---------------------------------------------------------------------------
# 8. shap_top3 populated when SHAP lookup succeeds
# ---------------------------------------------------------------------------

@mock_aws
def test_shap_top3_populated_when_shap_found(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", _REGION)
    monkeypatch.setenv("AUDIT_TABLE", _AUDIT_TABLE)
    monkeypatch.setenv("SHAP_TABLE", _SHAP_TABLE)
    _create_audit_table()
    _create_shap_table()

    # Pre-populate the SHAP store (V1=5.0 is top, then others)
    shap_dict = {f"V{i}": float(i) for i in range(1, 29)}
    compressed = bytes(msgpack.packb(shap_dict))
    boto3.resource("dynamodb", region_name=_REGION).Table(_SHAP_TABLE).put_item(
        Item={"prediction_hash": "hash-with-shap", "shap_values": compressed}
    )

    _run(_write(AuditLogger(), prediction_hash="hash-with-shap"))
    item = _scan_item()

    top3 = item.get("shap_top3", [])
    assert len(top3) == 3, f"expected 3 shap_top3 items, got {len(top3)}"
    # V28 has the largest absolute value (28.0)
    assert top3[0]["feature"] == "V28"


# ---------------------------------------------------------------------------
# 9. No reference to fraud-override-queue in any src/ file
# ---------------------------------------------------------------------------

def test_no_reference_to_override_queue_in_src():
    import pathlib

    src_root = pathlib.Path(__file__).resolve().parents[2] / "src"
    violations = [
        str(py_file)
        for py_file in src_root.rglob("*.py")
        if "fraud-override-queue" in py_file.read_text()
    ]
    assert not violations, (
        f"'fraud-override-queue' must not appear in any src/ file. Found in: {violations}"
    )
