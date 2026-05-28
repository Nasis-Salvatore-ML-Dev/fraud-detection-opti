"""Feature drift monitor using Population Stability Index (PSI).

Compares the distribution of recent prediction features against the
training baseline stored in data/baselines/training_baseline.json.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.monitoring.metrics import publish_component_failure

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BASELINE_PATH = _REPO_ROOT / "data" / "baselines" / "training_baseline.json"

_PSI_FEATURES = ["Amount", "amount_log", "amount_zscore", "hour_of_day"]

_STABLE = "stable"
_MONITOR = "monitor"
_ACTION = "action_required"
_STATUS_RANK = {_STABLE: 0, _MONITOR: 1, _ACTION: 2}


def _psi_status(psi: float) -> str:
    if psi >= 0.2:
        return _ACTION
    if psi >= 0.1:
        return _MONITOR
    return _STABLE


def _worst_status(statuses: list[str]) -> str:
    return max(statuses, key=lambda s: _STATUS_RANK.get(s, 0), default=_STABLE)


_RECOMMENDATIONS = {
    _STABLE: "Feature distributions are stable.",
    _MONITOR: (
        "Moderate drift detected — monitor closely and consider "
        "scheduling a retrain if trend continues."
    ),
    _ACTION: (
        "Significant drift detected — retrain the model or "
        "investigate the upstream data pipeline immediately."
    ),
}


class DriftMonitor:
    """Compute PSI-based drift reports by comparing recent predictions to the training distribution."""

    def __init__(self) -> None:
        env_path = os.environ.get("BASELINE_PATH")
        baseline_path = Path(env_path) if env_path else _DEFAULT_BASELINE_PATH
        log.info("DriftMonitor loading baseline from %s", baseline_path)
        with open(baseline_path) as f:
            self._baseline: dict = json.load(f)
        log.info(
            "DriftMonitor initialised  baseline_features=%s",
            list(self._baseline.keys()),
        )

    def compute_report(self, recent_records: list[dict]) -> dict:
        """Compute PSI for each PSI feature and return a dict matching DriftReportResponse."""
        computed_at = datetime.now(timezone.utc).isoformat()

        if not recent_records:
            return {
                "computed_at": computed_at,
                "n_recent_predictions": 0,
                "overall_status": _STABLE,
                "features": [],
                "recommendation": "No recent predictions available for drift analysis.",
            }

        # Collect feature values from the input_features sub-dict in each record
        raw: dict[str, list[float]] = {f: [] for f in _PSI_FEATURES}
        for record in recent_records:
            feats = record.get("input_features") or {}
            for feat in _PSI_FEATURES:
                val = feats.get(feat)
                if val is not None:
                    raw[feat].append(float(val))

        feature_reports: list[dict] = []
        statuses: list[str] = []

        for feat in _PSI_FEATURES:
            values = raw[feat]
            if not values or feat not in self._baseline:
                continue

            baseline = self._baseline[feat]
            bin_edges = np.array(baseline["bin_edges"])
            expected = np.array(baseline["expected_proportions"])

            counts, _ = np.histogram(values, bins=bin_edges)
            total = counts.sum()
            if total == 0:
                continue

            actual = counts / total
            # Clip to epsilon before log to guard against empty bins
            eps = 1e-6
            actual_c = np.clip(actual, eps, None)
            expected_c = np.clip(expected, eps, None)

            psi = float(np.sum((actual_c - expected_c) * np.log(actual_c / expected_c)))
            status = _psi_status(psi)
            statuses.append(status)

            feature_reports.append(
                {
                    "feature": feat,
                    "psi": round(psi, 6),
                    "status": status,
                    "n_samples": len(values),
                    "baseline_mean": round(baseline["mean"], 4),
                    "baseline_std": round(baseline["std"], 4),
                }
            )

        overall_status = _worst_status(statuses) if statuses else _STABLE

        return {
            "computed_at": computed_at,
            "n_recent_predictions": len(recent_records),
            "overall_status": overall_status,
            "features": feature_reports,
            "recommendation": _RECOMMENDATIONS[overall_status],
        }

    def publish_to_cloudwatch(self, report: dict) -> None:
        """Put FraudPSI metrics to CloudWatch namespace FraudDetection. Fails silently."""
        try:
            import boto3

            cw = boto3.client(
                "cloudwatch",
                region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            )
            metric_data = [
                {
                    "MetricName": "FraudPSI",
                    "Dimensions": [{"Name": "Feature", "Value": f["feature"]}],
                    "Value": f["psi"],
                    "Unit": "None",
                }
                for f in report.get("features", [])
            ]
            if metric_data:
                cw.put_metric_data(Namespace="FraudDetection", MetricData=metric_data)
                log.info("PSI metrics published to CloudWatch (%d features)", len(metric_data))
        except Exception as exc:
            log.warning("CloudWatch publish failed (non-fatal): %s", exc)
            publish_component_failure("DriftMonitor")
