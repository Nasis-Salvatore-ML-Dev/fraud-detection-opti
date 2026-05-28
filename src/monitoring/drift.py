"""Feature drift monitor using Population Stability Index (PSI).

Covers all 31 prediction features (4 engineered + V1-V28), predicted
fraud probability distribution (concept drift), model age, and fraud flag
rate — then emits a RetrainingRequired CloudWatch signal when multiple
conditions fire simultaneously.
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

_MIN_RECORDS_CONCEPT_DRIFT = 50
_MIN_RECORDS_FRAUD_RATE = 100


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


def _compute_psi(values: list[float], baseline: dict) -> float | None:
    """Compute PSI for values against a baseline with bin_edges and expected_proportions.

    Returns None when values is empty or all counts fall outside all bins.
    """
    if not values:
        return None
    bin_edges = np.array(baseline["bin_edges"])
    expected = np.array(baseline["expected_proportions"])
    counts, _ = np.histogram(values, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return None
    actual = counts / total
    eps = 1e-6
    actual_c = np.clip(actual, eps, None)
    expected_c = np.clip(expected, eps, None)
    return float(np.sum((actual_c - expected_c) * np.log(actual_c / expected_c)))


class DriftMonitor:
    """Compute PSI-based drift reports and publish CloudWatch retraining signals."""

    def __init__(self, bundle=None, baseline=None) -> None:
        # ── Engineered-feature baseline (4 features, from JSON file) ─────────
        if baseline is not None:
            self._baseline: dict = baseline
        else:
            env_path = os.environ.get("BASELINE_PATH")
            baseline_path = Path(env_path) if env_path else _DEFAULT_BASELINE_PATH
            log.info("DriftMonitor loading baseline from %s", baseline_path)
            with open(baseline_path) as f:
                self._baseline = json.load(f)

        # ── Model bundle (V-feature PSI, concept drift, experiment manifest) ─
        self._v_psi_baseline: dict = {}
        self._experiment_manifest: dict = {}

        if bundle is None:
            try:
                from src.utils.model_loader import load_model_bundle
                bundle = load_model_bundle()
            except Exception as exc:
                log.warning(
                    "DriftMonitor: model bundle unavailable, extended drift disabled: %s", exc
                )

        if bundle is not None:
            self._v_psi_baseline = getattr(bundle, "psi_baseline", {})
            self._experiment_manifest = getattr(bundle, "experiment_manifest", {})

        log.info(
            "DriftMonitor initialised  baseline_features=%s  v_psi_features=%d",
            list(self._baseline.keys()),
            len(self._v_psi_baseline),
        )

    # ── Core PSI report ───────────────────────────────────────────────────────

    def compute_report(self, recent_records: list[dict]) -> dict:
        """Compute PSI for all features (4 engineered + V1-V28 when present)."""
        computed_at = datetime.now(timezone.utc).isoformat()

        if not recent_records:
            return {
                "computed_at": computed_at,
                "n_recent_predictions": 0,
                "overall_status": _STABLE,
                "features": [],
                "recommendation": "No recent predictions available for drift analysis.",
            }

        # Collect feature values from the input_features sub-dict
        raw: dict[str, list[float]] = {}
        v_found = False
        for record in recent_records:
            feats = record.get("input_features") or {}
            for feat, val in feats.items():
                if val is not None:
                    raw.setdefault(feat, []).append(float(val))
                    if feat.startswith("V") and feat[1:].isdigit():
                        v_found = True

        if not v_found and self._v_psi_baseline:
            log.warning(
                "DriftMonitor: v_features_msgpack absent from audit records — V-feature PSI skipped"
            )

        feature_reports: list[dict] = []
        statuses: list[str] = []

        # 4 engineered features — from JSON baseline (has mean/std)
        for feat in _PSI_FEATURES:
            if feat not in self._baseline:
                continue
            values = raw.get(feat, [])
            if not values:
                continue
            baseline = self._baseline[feat]
            psi = _compute_psi(values, baseline)
            if psi is None:
                continue
            status = _psi_status(psi)
            statuses.append(status)
            log.debug("PSI[%s] = %.6f (%s)", feat, psi, status)
            feature_reports.append({
                "feature": feat,
                "psi": round(psi, 6),
                "status": status,
                "n_samples": len(values),
                "baseline_mean": round(float(baseline.get("mean", 0.0)), 4),
                "baseline_std": round(float(baseline.get("std", 0.0)), 4),
            })

        # V1-V28 — from bundle psi_baseline (bin_edges + expected_proportions only)
        for feat, baseline in self._v_psi_baseline.items():
            if not (feat.startswith("V") and feat[1:].isdigit()):
                continue
            values = raw.get(feat, [])
            if not values:
                continue
            psi = _compute_psi(values, baseline)
            if psi is None:
                continue
            status = _psi_status(psi)
            statuses.append(status)
            log.debug("PSI[%s] = %.6f (%s)", feat, psi, status)
            feature_reports.append({
                "feature": feat,
                "psi": round(psi, 6),
                "status": status,
                "n_samples": len(values),
                "baseline_mean": round(float(baseline.get("mean", 0.0)), 4),
                "baseline_std": round(float(baseline.get("std", 0.0)), 4),
            })

        overall_status = _worst_status(statuses) if statuses else _STABLE
        log.info(
            "Drift check: %d features computed, overall_status=%s",
            len(feature_reports),
            overall_status,
        )

        return {
            "computed_at": computed_at,
            "n_recent_predictions": len(recent_records),
            "overall_status": overall_status,
            "features": feature_reports,
            "recommendation": _RECOMMENDATIONS[overall_status],
        }

    # ── Extended signals ──────────────────────────────────────────────────────

    def compute_concept_drift(self, recent_records: list[dict]) -> tuple[float | None, str]:
        """Compute PSI on the predicted fraud probability distribution.

        Returns (psi, status) or (None, 'stable') when skipped.
        Skips if fewer than 50 records — logs warning.
        """
        if len(recent_records) < _MIN_RECORDS_CONCEPT_DRIFT:
            log.warning(
                "DriftMonitor: fewer than %d records for concept drift (%d) — skipping",
                _MIN_RECORDS_CONCEPT_DRIFT,
                len(recent_records),
            )
            return None, _STABLE

        baseline = self._v_psi_baseline.get("fraud_probability") or self._baseline.get(
            "fraud_probability"
        )
        if baseline is None:
            log.warning(
                "DriftMonitor: no fraud_probability baseline available — concept drift skipped"
            )
            return None, _STABLE

        probs = [
            float(r["fraud_probability"])
            for r in recent_records
            if r.get("fraud_probability") is not None
        ]
        if not probs:
            log.warning(
                "DriftMonitor: no fraud_probability values in records — concept drift skipped"
            )
            return None, _STABLE

        psi = _compute_psi(probs, baseline)
        if psi is None:
            return None, _STABLE

        status = _psi_status(psi)
        log.info("ConceptDriftPSI = %.6f (%s)", psi, status)
        return psi, status

    def compute_model_age(self) -> int:
        """Return model age in whole days from experiment_manifest.trained_at."""
        trained_at_str = self._experiment_manifest.get("trained_at")
        if not trained_at_str:
            log.warning("DriftMonitor: trained_at missing from experiment_manifest — using 0")
            return 0
        try:
            trained_at = datetime.fromisoformat(str(trained_at_str).replace("Z", "+00:00"))
            return int((datetime.now(timezone.utc) - trained_at).days)
        except Exception as exc:
            log.warning(
                "DriftMonitor: could not parse trained_at=%r: %s — using 0", trained_at_str, exc
            )
            return 0

    def compute_fraud_flag_rate_delta(self, recent_records: list[dict]) -> float | None:
        """Return |current_fraud_flag_rate - baseline_fraud_flag_rate|, rounded to 4 dp.

        Returns None when skipped (< 100 records or no baseline).
        """
        if len(recent_records) < _MIN_RECORDS_FRAUD_RATE:
            log.warning(
                "DriftMonitor: fewer than %d records for fraud flag rate (%d) — skipping",
                _MIN_RECORDS_FRAUD_RATE,
                len(recent_records),
            )
            return None

        baseline_rate = self._experiment_manifest.get("metrics", {}).get("fraud_flag_rate")
        if baseline_rate is None:
            log.warning("DriftMonitor: fraud_flag_rate not in experiment_manifest — skipping")
            return None

        is_fraud_vals = [r.get("is_fraud", False) for r in recent_records]
        current_rate = sum(1 for v in is_fraud_vals if v) / len(is_fraud_vals)
        delta = round(abs(current_rate - float(baseline_rate)), 4)
        log.debug(
            "FraudFlagRateDelta: current=%.4f baseline=%.4f delta=%.4f",
            current_rate,
            float(baseline_rate),
            delta,
        )
        return delta

    # ── Full orchestrated check ───────────────────────────────────────────────

    def run_full_drift_check(
        self,
        recent_500: list[dict],
        recent_1000: list[dict] | None = None,
    ) -> dict:
        """Compute all drift signals, publish CloudWatch metrics, return summary dict."""
        if recent_1000 is None:
            recent_1000 = recent_500

        feature_report = self.compute_report(recent_500)
        concept_psi, concept_status = self.compute_concept_drift(recent_500)
        model_age_days = self.compute_model_age()
        fraud_flag_delta = self.compute_fraud_flag_rate_delta(recent_1000)

        # Evaluate retraining signals
        signal_1 = any(f["psi"] > 0.2 for f in feature_report.get("features", []))
        signal_2 = model_age_days > 90
        signal_3 = fraud_flag_delta is not None and fraud_flag_delta > 0.1
        signal_4 = concept_psi is not None and concept_psi > 0.2

        signals: dict[str, bool] = {
            "feature_drift": signal_1,
            "model_age_over_90": signal_2,
            "fraud_flag_rate_delta": signal_3,
            "concept_drift": signal_4,
        }
        n_fired = sum(signals.values())
        retraining_required = n_fired >= 2

        log.debug(
            "Retraining signals: feature_drift=%s model_age=%s fraud_rate=%s concept_drift=%s",
            signal_1,
            signal_2,
            signal_3,
            signal_4,
        )

        if retraining_required:
            fired = [name for name, fired in signals.items() if fired]
            log.warning("Retraining signals fired: %s — RetrainingRequired=1", fired)
        else:
            log.info("No retraining needed (signals fired: %d/4)", n_fired)

        # Publish all CloudWatch metrics
        try:
            import boto3

            cw = boto3.client(
                "cloudwatch",
                region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            )

            # FraudPSI per feature — batched in groups of 20 (CloudWatch API limit)
            psi_metrics = [
                {
                    "MetricName": "FraudPSI",
                    "Dimensions": [{"Name": "Feature", "Value": f["feature"]}],
                    "Value": f["psi"],
                    "Unit": "None",
                }
                for f in feature_report.get("features", [])
            ]
            for i in range(0, max(len(psi_metrics), 1), 20):
                chunk = psi_metrics[i : i + 20]
                if chunk:
                    cw.put_metric_data(Namespace="FraudDetection", MetricData=chunk)
                    log.info("PSI metrics published to CloudWatch (%d features)", len(chunk))

            # ConceptDriftPSI — always published (0 when skipped)
            cw.put_metric_data(
                Namespace="FraudDetection",
                MetricData=[{
                    "MetricName": "ConceptDriftPSI",
                    "Value": float(concept_psi) if concept_psi is not None else 0.0,
                    "Unit": "None",
                }],
            )

            # ModelAgeDays
            cw.put_metric_data(
                Namespace="FraudDetection",
                MetricData=[{
                    "MetricName": "ModelAgeDays",
                    "Value": float(model_age_days),
                    "Unit": "Count",
                }],
            )

            # FraudFlagRateDelta — 0 when skipped
            cw.put_metric_data(
                Namespace="FraudDetection",
                MetricData=[{
                    "MetricName": "FraudFlagRateDelta",
                    "Value": float(fraud_flag_delta) if fraud_flag_delta is not None else 0.0,
                    "Unit": "None",
                }],
            )

            # RetrainingRequired — always 0 or 1
            cw.put_metric_data(
                Namespace="FraudDetection",
                MetricData=[{
                    "MetricName": "RetrainingRequired",
                    "Value": 1.0 if retraining_required else 0.0,
                    "Unit": "Count",
                }],
            )

        except Exception as exc:
            log.warning("CloudWatch publish failed (non-fatal): %s", exc)
            publish_component_failure("DriftMonitor")

        return {
            "feature_report": feature_report,
            "concept_drift_psi": concept_psi,
            "concept_drift_status": concept_status,
            "model_age_days": model_age_days,
            "fraud_flag_rate_delta": fraud_flag_delta,
            "retraining_required": retraining_required,
            "signals": signals,
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
