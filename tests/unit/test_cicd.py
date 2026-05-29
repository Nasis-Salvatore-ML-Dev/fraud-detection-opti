"""Unit tests for scripts/latency_check.py and scripts/shadow_eval.py."""

import asyncio
import json
import os
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import numpy as np
import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_audit_item(
    fraud_prob: float = 0.5,
    is_fraud: bool = False,
    hour_of_day: float = 10.0,
    amount: float = 100.0,
) -> dict:
    """DynamoDB item with all required fields."""
    v_dict = {f"V{i}": float(i) * 0.01 for i in range(1, 29)}
    return {
        "prediction_id": "test-id",
        "v_features_msgpack": msgpack.packb(v_dict),
        "Amount": Decimal(str(amount)),
        "hour_of_day": Decimal(str(hour_of_day)),
        "fraud_probability": Decimal(str(fraud_prob)),
        "is_fraud": is_fraud,
    }


def _make_discriminative_items(n: int = 100, fraud_count: int = 10) -> list[dict]:
    """Champion correctly identifies fraud (0.9 for fraud, 0.1 for non-fraud)."""
    items = []
    for i in range(n):
        is_fraud = i < fraud_count
        items.append(_make_audit_item(fraud_prob=0.9 if is_fraud else 0.1, is_fraud=is_fraud))
    return items


def _mock_client(responses: list) -> MagicMock:
    """Build a context-manager-compatible httpx.Client mock with given responses."""
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.post.side_effect = responses
    return client


def _make_resp(prob: float) -> MagicMock:
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = {"fraud_probability": prob}
    return m


# ---------------------------------------------------------------------------
# latency_check tests
# ---------------------------------------------------------------------------

def test_compute_p99_returns_99th_percentile():
    from scripts.latency_check import compute_p99

    times = list(range(1, 101))  # 1..100 ms
    result = compute_p99(times)
    assert result == pytest.approx(np.percentile(times, 99))


def test_run_latency_check_returns_0_when_p99_within_limit():
    from scripts.latency_check import run_latency_check

    with patch("scripts.latency_check._gather_times", new_callable=AsyncMock) as mock_gt:
        mock_gt.return_value = [10.0] * 20
        result = asyncio.run(run_latency_check("http://example.com", 20, 50.0))

    assert result == 0


def test_run_latency_check_returns_1_when_p99_exceeds_limit():
    from scripts.latency_check import run_latency_check

    with patch("scripts.latency_check._gather_times", new_callable=AsyncMock) as mock_gt:
        mock_gt.return_value = [100.0] * 20
        result = asyncio.run(run_latency_check("http://example.com", 20, 50.0))

    assert result == 1


def test_gather_times_sends_n_requests():
    from scripts.latency_check import _gather_times

    call_count = 0

    async def _fake_send(client, endpoint):
        nonlocal call_count
        call_count += 1
        return 5.0

    with patch("scripts.latency_check._send_one", side_effect=_fake_send):
        times = asyncio.run(_gather_times("http://example.com", 15))

    assert len(times) == 15
    assert call_count == 15
    assert all(t == 5.0 for t in times)


# ---------------------------------------------------------------------------
# shadow_eval tests
# ---------------------------------------------------------------------------

def test_compute_metrics_perfect_classifier():
    from scripts.shadow_eval import _compute_metrics

    y_true = [1, 1, 0, 0, 1, 0]
    scores = [0.9, 0.8, 0.1, 0.2, 0.7, 0.15]
    metrics = _compute_metrics(y_true, scores)

    assert metrics["auprc"] == pytest.approx(1.0, abs=0.01)
    assert metrics["recall"] == pytest.approx(1.0, abs=0.01)
    assert metrics["fpr"] == pytest.approx(0.0, abs=0.01)


def test_unpack_record_returns_none_when_v_features_missing():
    from scripts.shadow_eval import _unpack_record

    item = {
        "Amount": Decimal("100.0"),
        "fraud_probability": Decimal("0.5"),
        "is_fraud": False,
    }
    assert _unpack_record(item) is None


def test_unpack_record_returns_correct_fields():
    from scripts.shadow_eval import _unpack_record

    item = _make_audit_item(fraud_prob=0.3, is_fraud=True, hour_of_day=14.0, amount=250.0)
    result = _unpack_record(item)

    assert result is not None
    assert result["champion_score"] == pytest.approx(0.3, abs=1e-6)
    assert result["is_fraud"] is True
    assert result["payload"]["amount"] == pytest.approx(250.0)
    assert result["payload"]["time"] == pytest.approx(14.0 * 3600.0)
    assert len([k for k in result["payload"] if k.startswith("v")]) == 28


def test_run_shadow_eval_skipped_when_fewer_than_100_records(tmp_path):
    from scripts.shadow_eval import run_shadow_eval

    with patch("scripts.shadow_eval._scan_audit", return_value=[]):
        result = run_shadow_eval(
            "http://staging.example.com", 1000, str(tmp_path / "out.json")
        )

    assert result["status"] == "skipped"
    assert result["n_records"] == 0
    assert Path(tmp_path / "out.json").exists()


def test_run_shadow_eval_returns_pass_when_challenger_auprc_sufficient(tmp_path):
    from scripts.shadow_eval import run_shadow_eval

    items = _make_discriminative_items(n=100, fraud_count=10)
    # Challenger also discriminates well
    resps = [_make_resp(0.88 if i < 10 else 0.12) for i in range(100)]
    mock_client = _mock_client(resps)

    with (
        patch("scripts.shadow_eval._scan_audit", return_value=items),
        patch("httpx.Client", return_value=mock_client),
    ):
        result = run_shadow_eval(
            "http://staging.example.com", 1000, str(tmp_path / "out.json")
        )

    assert result["status"] == "pass"
    assert result["challenger"]["auprc"] >= result["min_challenger_auprc"]
    assert Path(tmp_path / "out.json").exists()


def test_run_shadow_eval_returns_fail_when_challenger_auprc_low(tmp_path):
    from scripts.shadow_eval import run_shadow_eval

    items = _make_discriminative_items(n=100, fraud_count=10)
    # Challenger returns constant 0.5 (no discrimination) → AUPRC ≈ fraud_rate
    resps = [_make_resp(0.5) for _ in range(100)]
    mock_client = _mock_client(resps)

    with (
        patch("scripts.shadow_eval._scan_audit", return_value=items),
        patch("httpx.Client", return_value=mock_client),
    ):
        result = run_shadow_eval(
            "http://staging.example.com", 1000, str(tmp_path / "out.json")
        )

    assert result["status"] == "fail"
    assert result["challenger"]["auprc"] < result["min_challenger_auprc"]
