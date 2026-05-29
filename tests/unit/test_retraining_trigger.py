"""Unit tests for src/monitoring/retraining_trigger.py"""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

# Minimal AWS credentials for moto
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

# Ensure the S3 model age guard is skipped unless a test explicitly enables it
os.environ.pop("MODEL_S3_BUCKET", None)
os.environ.pop("MODEL_S3_KEY", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _put_cw(cw, metric_name: str, value: float) -> None:
    """Put a dimensionless metric datapoint 1 hour in the past into moto CloudWatch.

    Using now - 1h ensures the datapoint is clearly inside the 24h query window
    (which runs from now-24h to now), avoiding boundary-exclusion edge cases.
    """
    cw.put_metric_data(
        Namespace="FraudDetection",
        MetricData=[{
            "MetricName": metric_name,
            "Value": value,
            "Unit": "None",
            "Timestamp": datetime.now(timezone.utc) - timedelta(hours=1),
        }],
    )


def _cw():
    return boto3.client("cloudwatch", region_name="us-east-1")


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

@mock_aws
def test_returns_correct_structure():
    from src.monitoring.retraining_trigger import check_and_trigger

    result = check_and_trigger()

    assert "signals" in result
    assert "signals_fired" in result
    assert "retraining_triggered" in result
    assert "reason" in result
    assert set(result["signals"].keys()) == {
        "input_drift", "model_age", "fraud_flag_rate", "concept_drift"
    }
    assert isinstance(result["signals_fired"], int)
    assert isinstance(result["retraining_triggered"], bool)
    assert isinstance(result["reason"], str)


# ---------------------------------------------------------------------------
# Signal evaluation tests (moto CloudWatch)
# ---------------------------------------------------------------------------

@mock_aws
def test_signals_fired_is_2_when_exactly_2_signals_true():
    from src.monitoring.retraining_trigger import check_and_trigger

    cw = _cw()
    _put_cw(cw, "FraudPSI", 0.3)        # > 0.2  → input_drift True
    _put_cw(cw, "ModelAgeDays", 100.0)   # > 90   → model_age True
    # FraudFlagRateDelta and ConceptDriftPSI have no datapoints → False

    result = check_and_trigger()

    assert result["signals"]["input_drift"] is True
    assert result["signals"]["model_age"] is True
    assert result["signals"]["fraud_flag_rate"] is False
    assert result["signals"]["concept_drift"] is False
    assert result["signals_fired"] == 2


@mock_aws
def test_retraining_triggered_true_when_signals_fired_ge_2():
    from src.monitoring.retraining_trigger import check_and_trigger

    cw = _cw()
    _put_cw(cw, "FraudPSI", 0.25)
    _put_cw(cw, "ModelAgeDays", 95.0)

    result = check_and_trigger()

    assert result["retraining_triggered"] is True


@mock_aws
def test_retraining_triggered_false_when_signals_fired_lt_2():
    from src.monitoring.retraining_trigger import check_and_trigger

    cw = _cw()
    _put_cw(cw, "FraudPSI", 0.15)  # ≤ 0.2 → False; all others absent → False

    result = check_and_trigger()

    assert result["retraining_triggered"] is False
    assert result["signals_fired"] < 2


@mock_aws
def test_signal_false_when_cloudwatch_has_no_datapoints():
    from src.monitoring.retraining_trigger import check_and_trigger

    # No metrics put → all signals should evaluate to False
    result = check_and_trigger()

    assert result["signals"]["input_drift"] is False
    assert result["signals"]["model_age"] is False
    assert result["signals"]["fraud_flag_rate"] is False
    assert result["signals"]["concept_drift"] is False
    assert result["signals_fired"] == 0


# ---------------------------------------------------------------------------
# GitHub dispatch — dry_run and signal gate
# ---------------------------------------------------------------------------

@mock_aws
def test_dispatch_not_called_when_dry_run_true():
    from src.monitoring.retraining_trigger import check_and_trigger

    cw = _cw()
    _put_cw(cw, "FraudPSI", 0.3)
    _put_cw(cw, "ModelAgeDays", 100.0)

    with patch("src.monitoring.retraining_trigger.requests") as mock_req:
        result = check_and_trigger(dry_run=True)

    assert result["retraining_triggered"] is True
    mock_req.post.assert_not_called()


@mock_aws
def test_dispatch_not_called_when_signals_fired_lt_2():
    from src.monitoring.retraining_trigger import check_and_trigger

    # No CloudWatch data → 0 signals → no dispatch
    with patch("src.monitoring.retraining_trigger.requests") as mock_req:
        result = check_and_trigger(dry_run=False)

    assert result["retraining_triggered"] is False
    mock_req.post.assert_not_called()


# ---------------------------------------------------------------------------
# GitHub dispatch — in-progress run guard
# ---------------------------------------------------------------------------

@mock_aws
def test_dispatch_skipped_when_in_progress_run_exists():
    from src.monitoring.retraining_trigger import check_and_trigger

    cw = _cw()
    _put_cw(cw, "FraudPSI", 0.3)
    _put_cw(cw, "ModelAgeDays", 100.0)

    mock_get_resp = MagicMock()
    mock_get_resp.json.return_value = {
        "workflow_runs": [{"html_url": "https://github.com/owner/repo/actions/runs/1"}]
    }
    mock_get_resp.raise_for_status.return_value = None

    with patch.dict(os.environ, {"GITHUB_REPO": "owner/repo", "GITHUB_TOKEN": "tok"}):
        with patch("src.monitoring.retraining_trigger.requests.get", return_value=mock_get_resp):
            with patch("src.monitoring.retraining_trigger.requests.post") as mock_post:
                result = check_and_trigger(dry_run=False)

    assert result["retraining_triggered"] is True
    mock_post.assert_not_called()


@mock_aws
def test_dispatch_called_when_no_in_progress_run():
    from src.monitoring.retraining_trigger import check_and_trigger

    cw = _cw()
    _put_cw(cw, "FraudPSI", 0.3)
    _put_cw(cw, "ModelAgeDays", 100.0)

    mock_get_resp = MagicMock()
    mock_get_resp.json.return_value = {"workflow_runs": []}
    mock_get_resp.raise_for_status.return_value = None

    mock_post_resp = MagicMock()
    mock_post_resp.raise_for_status.return_value = None

    with patch.dict(os.environ, {"GITHUB_REPO": "owner/repo", "GITHUB_TOKEN": "tok"}):
        with patch("src.monitoring.retraining_trigger.requests.get", return_value=mock_get_resp):
            with patch(
                "src.monitoring.retraining_trigger.requests.post",
                return_value=mock_post_resp,
            ) as mock_post:
                result = check_and_trigger(dry_run=False)

    assert result["retraining_triggered"] is True
    mock_post.assert_called_once()
    call_url = mock_post.call_args.args[0]
    assert "train.yml/dispatches" in call_url
    json_body = mock_post.call_args.kwargs["json"]
    assert json_body["ref"] == "main"
    assert json_body["inputs"]["trigger"] == "retraining_orchestrator"


# ---------------------------------------------------------------------------
# Model age guard
# ---------------------------------------------------------------------------

@mock_aws
def test_returns_early_with_reason_model_too_recent():
    from src.monitoring.retraining_trigger import check_and_trigger

    recent = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()

    with patch(
        "src.monitoring.retraining_trigger._read_trained_at_from_s3",
        return_value=recent,
    ):
        result = check_and_trigger()

    assert result["retraining_triggered"] is False
    assert result["reason"] == "model too recent"
    assert result["signals_fired"] == 0


@mock_aws
def test_model_age_guard_not_triggered_when_model_old_enough():
    from src.monitoring.retraining_trigger import check_and_trigger

    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    cw = _cw()
    _put_cw(cw, "FraudPSI", 0.3)
    _put_cw(cw, "ModelAgeDays", 100.0)

    with patch(
        "src.monitoring.retraining_trigger._read_trained_at_from_s3",
        return_value=old,
    ):
        result = check_and_trigger()

    assert result["retraining_triggered"] is True
    assert result["reason"] != "model too recent"


# ---------------------------------------------------------------------------
# Missing env vars
# ---------------------------------------------------------------------------

@mock_aws
def test_missing_github_token_dispatch_skipped_no_exception():
    from src.monitoring.retraining_trigger import check_and_trigger

    cw = _cw()
    _put_cw(cw, "FraudPSI", 0.3)
    _put_cw(cw, "ModelAgeDays", 100.0)

    saved_token = os.environ.pop("GITHUB_TOKEN", None)
    saved_repo = os.environ.pop("GITHUB_REPO", None)
    try:
        with patch.dict(os.environ, {"GITHUB_REPO": "owner/repo"}):
            # GITHUB_TOKEN is absent — dispatch should be skipped, no exception raised
            result = check_and_trigger(dry_run=False)
    finally:
        if saved_token is not None:
            os.environ["GITHUB_TOKEN"] = saved_token
        if saved_repo is not None:
            os.environ["GITHUB_REPO"] = saved_repo

    # Signals still evaluated correctly
    assert result["retraining_triggered"] is True
    assert result["signals_fired"] == 2
