"""Retraining orchestrator: reads 4 CloudWatch drift signals and dispatches
GitHub Actions train.yml when 2 or more signals fire.

Designed to run as a scheduled AWS Lambda triggered by EventBridge daily.
"""

import logging
import os
from datetime import datetime, timedelta, timezone

import boto3
import requests

from src.monitoring.metrics import publish_component_failure

log = logging.getLogger(__name__)

_NAMESPACE = "FraudDetection"
_PERIOD = 86400  # 24 hours


def _get_metric_max(cw, metric_name: str) -> float | None:
    """Return the maximum value across all datapoints for metric_name in the last 24h.

    Returns None if there are no datapoints; logs a warning.
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(seconds=_PERIOD)
    try:
        resp = cw.get_metric_statistics(
            Namespace=_NAMESPACE,
            MetricName=metric_name,
            StartTime=start_time,
            EndTime=end_time,
            Period=_PERIOD,
            Statistics=["Maximum"],
        )
        datapoints = resp.get("Datapoints", [])
        if not datapoints:
            log.warning(
                "No CloudWatch datapoints for %s in last 24h — treating signal as False",
                metric_name,
            )
            return None
        return max(d["Maximum"] for d in datapoints)
    except Exception as exc:
        log.warning("CloudWatch read failed for %s: %s", metric_name, exc)
        publish_component_failure("RetrainingTrigger")
        return None


def _read_trained_at_from_s3() -> str | None:
    """Download the model bundle pkl from S3 and return experiment_manifest.trained_at.

    Returns None and logs a warning on any failure so the caller can skip the guard.
    """
    bucket = os.environ.get("MODEL_S3_BUCKET")
    key = os.environ.get("MODEL_S3_KEY")
    if not bucket or not key:
        log.warning(
            "MODEL_S3_BUCKET or MODEL_S3_KEY not set — skipping model age guard"
        )
        return None

    import tempfile

    from joblib import load as joblib_load

    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    s3 = boto3.client("s3", region_name=region)
    try:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        s3.download_file(bucket, key, tmp_path)
        bundle = joblib_load(tmp_path)
        if isinstance(bundle, dict):
            manifest = bundle.get("experiment_manifest", {})
        else:
            manifest = getattr(bundle, "experiment_manifest", {})
        return manifest.get("trained_at")
    except Exception as exc:
        log.warning("S3 model bundle read failed (non-fatal): %s", exc)
        publish_component_failure("RetrainingTrigger")
        return None


def _parse_model_age_days(trained_at_str: str) -> int:
    """Parse a trained_at ISO string and return age in whole days (0 on parse failure)."""
    try:
        trained_at = datetime.fromisoformat(
            str(trained_at_str).replace("Z", "+00:00")
        )
        return int((datetime.now(timezone.utc) - trained_at).days)
    except Exception as exc:
        log.warning("Could not parse trained_at=%r: %s", trained_at_str, exc)
        return 0


def _maybe_dispatch_github() -> None:
    """Check for an in-progress training run and dispatch train.yml if none found."""
    github_repo = os.environ.get("GITHUB_REPO")
    github_token = os.environ.get("GITHUB_TOKEN")

    if not github_repo or not github_token:
        log.warning("GITHUB_REPO or GITHUB_TOKEN not set — skipping GitHub dispatch")
        return

    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    check_url = (
        f"https://api.github.com/repos/{github_repo}"
        "/actions/workflows/train.yml/runs"
    )
    try:
        resp = requests.get(
            check_url, headers=headers, params={"status": "in_progress"}, timeout=10
        )
        resp.raise_for_status()
        runs = resp.json().get("workflow_runs", [])
        if runs:
            log.info(
                "In-progress training run found (%s) — skipping dispatch",
                runs[0].get("html_url", ""),
            )
            return
    except Exception as exc:
        log.error("GitHub API check for in-progress runs failed: %s", exc)
        publish_component_failure("RetrainingTrigger")
        return

    dispatch_url = (
        f"https://api.github.com/repos/{github_repo}"
        "/actions/workflows/train.yml/dispatches"
    )
    try:
        resp = requests.post(
            dispatch_url,
            headers=headers,
            json={"ref": "main", "inputs": {"trigger": "retraining_orchestrator"}},
            timeout=10,
        )
        resp.raise_for_status()
        log.info("GitHub Actions workflow dispatch triggered: %s", dispatch_url)
    except Exception as exc:
        log.error("GitHub Actions dispatch POST failed: %s", exc)
        publish_component_failure("RetrainingTrigger")


def check_and_trigger(dry_run: bool = False) -> dict:
    """Evaluate 4 CloudWatch drift signals and trigger retraining if 2+ fire.

    Args:
        dry_run: When True, evaluates signals but does not dispatch GitHub Actions.

    Returns:
        {
            "signals": {"input_drift": bool, "model_age": bool,
                        "fraud_flag_rate": bool, "concept_drift": bool},
            "signals_fired": int,
            "retraining_triggered": bool,
            "reason": str,
        }
    """
    # Model age guard — skip retraining if model was trained within the last 7 days
    trained_at = _read_trained_at_from_s3()
    if trained_at is not None:
        model_age_days = _parse_model_age_days(trained_at)
        log.info("Model age from S3 manifest: %d days", model_age_days)
        if model_age_days < 7:
            log.info(
                "Model is only %d days old — too recent to retrain", model_age_days
            )
            return {
                "signals": {
                    "input_drift": False,
                    "model_age": False,
                    "fraud_flag_rate": False,
                    "concept_drift": False,
                },
                "signals_fired": 0,
                "retraining_triggered": False,
                "reason": "model too recent",
            }

    # Read all 4 CloudWatch signals
    cw = boto3.client(
        "cloudwatch",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )

    fraud_psi_max = _get_metric_max(cw, "FraudPSI")
    model_age_max = _get_metric_max(cw, "ModelAgeDays")
    fraud_flag_max = _get_metric_max(cw, "FraudFlagRateDelta")
    concept_drift_max = _get_metric_max(cw, "ConceptDriftPSI")

    signals = {
        "input_drift": fraud_psi_max is not None and fraud_psi_max > 0.2,
        "model_age": model_age_max is not None and model_age_max > 90,
        "fraud_flag_rate": fraud_flag_max is not None and fraud_flag_max > 0.1,
        "concept_drift": concept_drift_max is not None and concept_drift_max > 0.2,
    }

    log.info(
        "Retraining signals — input_drift=%s model_age=%s"
        " fraud_flag_rate=%s concept_drift=%s",
        signals["input_drift"],
        signals["model_age"],
        signals["fraud_flag_rate"],
        signals["concept_drift"],
    )

    signals_fired = sum(signals.values())
    retraining_triggered = signals_fired >= 2

    if retraining_triggered:
        fired_names = [name for name, v in signals.items() if v]
        reason = f"signals fired: {', '.join(fired_names)}"
    else:
        reason = f"insufficient signals ({signals_fired}/4 fired)"

    if retraining_triggered and not dry_run:
        _maybe_dispatch_github()

    return {
        "signals": signals,
        "signals_fired": signals_fired,
        "retraining_triggered": retraining_triggered,
        "reason": reason,
    }
