"""Bias / fairness test suite for the fraud detection model.

Segments predictions by transaction amount and time-of-day, then
computes per-segment AUPRC to flag any meaningful performance gap
relative to the overall model AUPRC.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix

from src.monitoring.metrics import publish_component_failure
from src.utils.model_loader import ModelBundle, load_model_bundle

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALIDATION_PATH = _REPO_ROOT / "data" / "validation" / "validation_set.csv"
_REPORTS_DIR = _REPO_ROOT / "data" / "reports"

# Flag a segment when its AUPRC falls below this fraction of overall AUPRC
_FLAG_RATIO = 0.7

_SEGMENTS: list[tuple[str, str]] = [
    ("high_amount", "Amount > 1000"),
    ("low_amount", "Amount <= 1000"),
    ("high_hour", "hour_of_day >= 18  (evening)"),
    ("low_hour", "hour_of_day < 18   (daytime)"),
]

# Fixed training-set normalisation constants (must match preprocessing.py)
_AMOUNT_MEAN = 90.8249
_AMOUNT_STD = 250.5032


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns that mirror scripts/train.py."""
    df = df.copy()
    if "amount_log" not in df.columns:
        df["amount_log"] = np.log1p(df["Amount"])
    if "amount_zscore" not in df.columns:
        df["amount_zscore"] = (df["Amount"] - _AMOUNT_MEAN) / _AMOUNT_STD
    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = (df["Time"] % 86400) / 3600
    return df


def _segment_mask(df: pd.DataFrame, name: str) -> pd.Series:
    return {
        "high_amount": df["Amount"] > 1000,
        "low_amount": df["Amount"] <= 1000,
        "high_hour": df["hour_of_day"] >= 18,
        "low_hour": df["hour_of_day"] < 18,
    }[name]


class BiasTestSuite:
    """Run per-segment fairness audits and cache the most recent report."""

    def __init__(self) -> None:
        self._validation_df: pd.DataFrame | None = None
        self._bundle: ModelBundle | None = None
        self._report: dict | None = None

        try:
            self._bundle = load_model_bundle()
        except Exception as exc:
            log.warning("BiasTestSuite: model bundle unavailable: %s", exc)
            publish_component_failure("BiasTester")

        if _VALIDATION_PATH.exists():
            try:
                self._validation_df = pd.read_csv(_VALIDATION_PATH)
                log.info(
                    "BiasTestSuite: loaded validation set  shape=%s",
                    self._validation_df.shape,
                )
            except Exception as exc:
                log.warning("BiasTestSuite: failed to load validation set: %s", exc)
                publish_component_failure("BiasTester")
        else:
            log.info("BiasTestSuite: no validation set found at %s", _VALIDATION_PATH)

        log.info("BiasTestSuite initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, validation_df: pd.DataFrame | None = None) -> dict:
        """Compute per-segment AUPRC and return a dict matching BiasReportResponse.

        Uses the provided validation_df if given, otherwise the CSV loaded at init.
        Returns the cached placeholder report if neither is available.
        """
        df = validation_df if validation_df is not None else self._validation_df

        if df is None or self._bundle is None:
            log.warning("BiasTestSuite.run: no data or model — returning cached report")
            return self.cached_report()

        df = _engineer_features(df)

        try:
            X = df[self._bundle.feature_names]
        except KeyError as exc:
            log.error("BiasTestSuite.run: missing feature columns %s", exc)
            return self.cached_report()

        y = df["Class"].astype(int)
        y_proba: np.ndarray = self._bundle.model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= self._bundle.threshold).astype(int)

        # Overall metrics
        overall_auprc = float(average_precision_score(y, y_proba))
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        overall_recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        overall_fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0

        auprc_floor = _FLAG_RATIO * overall_auprc
        fpr_threshold = 2.0 * overall_fpr
        bias_segments: list[dict] = []
        flagged_names: list[str] = []

        for seg_name, description in _SEGMENTS:
            mask = _segment_mask(df, seg_name)
            y_s = y[mask]
            p_s = y_proba[mask]
            n_samples = int(mask.sum())
            n_fraud = int(y_s.sum())

            if n_samples < 10 or n_fraud == 0:
                bias_segments.append(
                    {
                        "segment": seg_name,
                        "description": description,
                        "n_samples": n_samples,
                        "n_fraud": n_fraud,
                        "auprc": None,
                        "fpr": None,
                        "flagged": False,
                        "fpr_flagged": False,
                        "note": "Insufficient fraud samples for AUPRC",
                    }
                )
                continue

            seg_auprc = float(average_precision_score(y_s, p_s))
            y_pred_s = (p_s >= self._bundle.threshold).astype(int)
            cm_s = confusion_matrix(y_s, y_pred_s, labels=[0, 1])
            tn_s, fp_s, fn_s, tp_s = cm_s.ravel()
            seg_fpr = float(fp_s / (fp_s + tn_s)) if (fp_s + tn_s) else 0.0

            auprc_flagged = seg_auprc < auprc_floor
            fpr_flagged = seg_fpr > fpr_threshold
            flagged = auprc_flagged or fpr_flagged
            if flagged:
                flagged_names.append(seg_name)

            bias_segments.append(
                {
                    "segment": seg_name,
                    "description": description,
                    "n_samples": n_samples,
                    "n_fraud": n_fraud,
                    "auprc": round(seg_auprc, 4),
                    "fpr": round(seg_fpr, 6),
                    "flagged": flagged,
                    "fpr_flagged": fpr_flagged,
                }
            )

        if flagged_names:
            recommendation = (
                f"Performance gap detected in segments: {', '.join(flagged_names)}. "
                "Investigate training-data distribution or collect more samples for these groups."
            )
        else:
            recommendation = "No significant performance gaps detected across tested segments."

        report: dict = {
            "model_version": self._bundle.version,
            "overall_auprc": round(overall_auprc, 4),
            "overall_recall": round(overall_recall, 4),
            "overall_fpr": round(overall_fpr, 6),
            "fpr_parity_threshold": round(fpr_threshold, 6),
            "bias_segments": bias_segments,
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "recommendation": recommendation,
        }

        self._report = report
        self._save_report(report)
        return report

    def cached_report(self) -> dict:
        """Return the most recent report, or a placeholder if no run has completed."""
        if self._report is not None:
            return self._report
        return {
            "model_version": self._bundle.version if self._bundle else "unknown",
            "overall_auprc": 0.0,
            "overall_recall": 0.0,
            "overall_fpr": 0.0,
            "fpr_parity_threshold": 0.0,
            "bias_segments": [],
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "recommendation": (
                "No bias report available — call BiasTestSuite.run() with validation data."
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_report(self, report: dict) -> None:
        """Persist the bias report to data/reports/bias_report.json."""
        try:
            _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            path = _REPORTS_DIR / "bias_report.json"
            with open(path, "w") as f:
                json.dump(report, f, indent=2)
            log.info("Bias report saved → %s", path)
        except Exception as exc:
            log.warning("BiasTestSuite._save_report failed: %s", exc)
            publish_component_failure("BiasTester")
