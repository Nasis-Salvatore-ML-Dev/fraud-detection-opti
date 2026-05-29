"""Tests for nearest-neighbour similar cases, enriched /override, and /warmup."""

import asyncio
import os
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import boto3
import msgpack
import numpy as np
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ["AUDIT_TABLE"] = "fraud-audit-log"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_table():
    """Create fraud-audit-log with the requires-review-index GSI."""
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    dynamodb.create_table(
        TableName="fraud-audit-log",
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[
            {"AttributeName": "prediction_id", "AttributeType": "S"},
            {"AttributeName": "requires_review", "AttributeType": "S"},
        ],
        KeySchema=[{"AttributeName": "prediction_id", "KeyType": "HASH"}],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "requires-review-index",
                "KeySchema": [
                    {"AttributeName": "requires_review", "KeyType": "HASH"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
    )
    return dynamodb


def _make_resolved_item(prediction_id: str, v_seed: int, amount: float = 100.0) -> dict:
    """A resolved (non-review) audit record with V1-V28 packed via msgpack."""
    rng = np.random.default_rng(v_seed)
    v_dict = {f"V{i}": float(v) for i, v in enumerate(rng.standard_normal(28), start=1)}
    return {
        "prediction_id": prediction_id,
        "v_features_msgpack": msgpack.packb(v_dict),
        "Amount": Decimal(str(amount)),
        "fraud_probability": Decimal("0.2"),
        "is_fraud": False,
        "requires_review": "false",
        "shap_top3": [],
        "confidence_score": Decimal("0.5"),
        "timestamp": "2025-01-01T00:00:00+00:00",
    }


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


# ---------------------------------------------------------------------------
# 1. find_similar_cases returns empty list when fewer than 10 records
# ---------------------------------------------------------------------------

@mock_aws
def test_find_similar_cases_returns_empty_when_fewer_than_10_records():
    from src.monitoring.audit_logger import find_similar_cases
    dynamodb = _make_table()
    table = dynamodb.Table("fraud-audit-log")

    for i in range(5):
        table.put_item(Item=_make_resolved_item(f"pid-{i}", v_seed=i))

    v_features = {f"V{i}": 1.0 for i in range(1, 29)}
    result = find_similar_cases(v_features, "current-id", "fraud-audit-log")

    assert result == []


# ---------------------------------------------------------------------------
# 2. find_similar_cases returns top 3 sorted by cosine similarity
# ---------------------------------------------------------------------------

@mock_aws
def test_find_similar_cases_returns_top_3_by_cosine_similarity():
    from src.monitoring.audit_logger import find_similar_cases
    dynamodb = _make_table()
    table = dynamodb.Table("fraud-audit-log")

    for i in range(15):
        table.put_item(Item=_make_resolved_item(f"pid-{i}", v_seed=i))

    v_features = {f"V{i}": 1.0 for i in range(1, 29)}
    result = find_similar_cases(v_features, "current-id", "fraud-audit-log", n=3)

    assert len(result) == 3
    scores = [r["similarity_score"] for r in result]
    assert scores == sorted(scores, reverse=True), "Results not sorted by similarity descending"
    assert all(isinstance(r["prediction_id"], str) for r in result)
    assert all(isinstance(r["is_fraud"], bool) for r in result)


# ---------------------------------------------------------------------------
# 3. find_similar_cases excludes current_prediction_id from results
# ---------------------------------------------------------------------------

@mock_aws
def test_find_similar_cases_excludes_current_prediction_id():
    from src.monitoring.audit_logger import find_similar_cases
    dynamodb = _make_table()
    table = dynamodb.Table("fraud-audit-log")

    for i in range(14):
        table.put_item(Item=_make_resolved_item(f"pid-{i}", v_seed=i))
    # Also store a resolved record matching the "current" prediction
    table.put_item(Item=_make_resolved_item("current-id", v_seed=99))

    v_features = {f"V{i}": 1.0 for i in range(1, 29)}
    result = find_similar_cases(v_features, "current-id", "fraud-audit-log", n=5)

    returned_ids = [r["prediction_id"] for r in result]
    assert "current-id" not in returned_ids


# ---------------------------------------------------------------------------
# 4. find_similar_cases returns empty list on DynamoDB failure
# ---------------------------------------------------------------------------

@mock_aws
def test_find_similar_cases_returns_empty_on_dynamodb_failure():
    from src.monitoring.audit_logger import find_similar_cases

    # Table not created → query raises ResourceNotFoundException
    v_features = {f"V{i}": 1.0 for i in range(1, 29)}
    result = find_similar_cases(v_features, "current-id", "fraud-audit-log-missing")

    assert result == []


# ---------------------------------------------------------------------------
# 5. similarity_score rounded to 4 decimal places
# ---------------------------------------------------------------------------

def test_similarity_score_rounded_to_4_decimal_places():
    from src.monitoring.audit_logger import find_similar_cases

    items = []
    for j in range(15):
        v_dict = {f"V{i}": float(i + j * 0.3) for i in range(1, 29)}
        items.append({
            "prediction_id": f"pid-{j}",
            "v_features_msgpack": msgpack.packb(v_dict),
            "Amount": Decimal("100.0"),
            "fraud_probability": Decimal("0.2"),
            "is_fraud": False,
            "requires_review": "false",
            "shap_top3": [],
            "confidence_score": Decimal("0.5"),
            "timestamp": "2025-01-01T00:00:00",
        })

    with patch("src.monitoring.audit_logger.boto3") as mock_boto3:
        mock_table = MagicMock()
        mock_table.query.return_value = {"Items": items}
        mock_boto3.resource.return_value.Table.return_value = mock_table

        v_features = {f"V{i}": float(i) * 1.7 for i in range(1, 29)}
        result = find_similar_cases(v_features, "current-id", "fraud-audit-log", n=3)

    assert len(result) == 3
    for r in result:
        score = r["similarity_score"]
        assert score == round(score, 4), f"similarity_score {score!r} has more than 4 dp"


# ---------------------------------------------------------------------------
# 6. similar_cases = [] when requires_review = False
# ---------------------------------------------------------------------------

@mock_aws
def test_similar_cases_empty_when_requires_review_false():
    _make_table()
    from src.monitoring.audit_logger import AuditLogger

    audit = AuditLogger()

    with patch("src.monitoring.audit_logger.find_similar_cases") as mock_fsc:
        _run(audit.write(
            prediction_id="test-false",
            prediction_hash="hash-false",
            input_features=_make_input_features(),
            fraud_probability=0.1,  # outside [0.3, 0.7] → requires_review=False
            is_fraud=False,
            shap_values={},
            model_version="v1",
            confidence_score=0.1,
            request_ip="127.0.0.1",
            latency_ms=5.0,
            threshold_used=0.5,
        ))
        mock_fsc.assert_not_called()

    item = boto3.resource("dynamodb", region_name="us-east-1") \
        .Table("fraud-audit-log").get_item(Key={"prediction_id": "test-false"})["Item"]
    assert item["similar_cases"] == []


# ---------------------------------------------------------------------------
# 7. similar_cases populated when requires_review = True
# ---------------------------------------------------------------------------

@mock_aws
def test_similar_cases_populated_when_requires_review_true():
    _make_table()
    from src.monitoring.audit_logger import AuditLogger

    audit = AuditLogger()
    fake_cases = [
        {
            "prediction_id": "sim-1",
            "fraud_probability": 0.4,
            "is_fraud": False,
            "shap_top3": [],
            "amount": 140.0,
            "similarity_score": 0.9876,
        }
    ]

    with patch("src.monitoring.audit_logger.find_similar_cases", return_value=fake_cases):
        _run(audit.write(
            prediction_id="test-true",
            prediction_hash="hash-true",
            input_features=_make_input_features(),
            fraud_probability=0.5,  # inside [0.3, 0.7] → requires_review=True
            is_fraud=False,
            shap_values={},
            model_version="v1",
            confidence_score=0.5,
            request_ip="127.0.0.1",
            latency_ms=8.0,
            threshold_used=0.5,
        ))

    item = boto3.resource("dynamodb", region_name="us-east-1") \
        .Table("fraud-audit-log").get_item(Key={"prediction_id": "test-true"})["Item"]

    assert len(item["similar_cases"]) == 1
    assert item["similar_cases"][0]["prediction_id"] == "sim-1"


# ---------------------------------------------------------------------------
# 8. /override response includes similar_cases field
# ---------------------------------------------------------------------------

def test_override_response_includes_similar_cases():
    import src.api.app as app_module

    mock_audit = MagicMock()
    mock_audit.fetch_pending_reviews = AsyncMock(return_value=[
        {
            "prediction_id": "review-1",
            "fraud_probability": 0.5,
            "confidence_score": 0.5,
            "shap_top3": [],
            "similar_cases": [
                {
                    "prediction_id": "sim-1",
                    "fraud_probability": 0.4,
                    "is_fraud": False,
                    "shap_top3": [],
                    "amount": 120.0,
                    "similarity_score": 0.9512,
                }
            ],
            "amount": 150.0,
            "requires_review": True,
            "timestamp": "2025-01-01T00:00:00+00:00",
            "anomaly_flags": [],
            "high_value_alert": False,
        }
    ])
    app_module._audit = mock_audit

    client = TestClient(app_module.app)
    response = client.get("/override")

    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert "similar_cases" in items[0]
    assert items[0]["similar_cases"][0]["prediction_id"] == "sim-1"


# ---------------------------------------------------------------------------
# 9. /override response includes shap_top3 field
# ---------------------------------------------------------------------------

def test_override_response_includes_shap_top3():
    import src.api.app as app_module

    mock_audit = MagicMock()
    mock_audit.fetch_pending_reviews = AsyncMock(return_value=[
        {
            "prediction_id": "review-2",
            "fraud_probability": 0.45,
            "confidence_score": 0.45,
            "shap_top3": [
                {"feature": "V14", "value": -1.2},
                {"feature": "V4", "value": 0.8},
            ],
            "similar_cases": [],
            "amount": 200.0,
            "requires_review": True,
            "timestamp": "2025-01-01T00:00:00+00:00",
            "anomaly_flags": ["v14"],
            "high_value_alert": False,
        }
    ])
    app_module._audit = mock_audit

    client = TestClient(app_module.app)
    response = client.get("/override")

    assert response.status_code == 200
    items = response.json()
    assert "shap_top3" in items[0]
    assert len(items[0]["shap_top3"]) == 2
    features = [e["feature"] for e in items[0]["shap_top3"]]
    assert "V14" in features


# ---------------------------------------------------------------------------
# 10. /warmup returns 200 with model_loaded = True when bundle loaded
# ---------------------------------------------------------------------------

def test_warmup_returns_200_with_model_loaded_true():
    import src.api.app as app_module

    app_module._bundle = MagicMock()  # non-None → model_loaded=True

    client = TestClient(app_module.app)
    response = client.get("/warmup")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "warm"
    assert data["model_loaded"] is True
    assert "timestamp" in data


# ---------------------------------------------------------------------------
# 11. /warmup returns model_loaded = False when bundle not loaded
# ---------------------------------------------------------------------------

def test_warmup_returns_model_loaded_false_when_bundle_not_loaded():
    import src.api.app as app_module

    app_module._bundle = None

    client = TestClient(app_module.app)
    response = client.get("/warmup")

    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is False
