"""Bias / fairness test suite for the fraud detection model.

Segments predictions by transaction amount and time-of-day (config-driven),
then computes per-segment AUPRC, FPR ratio, and directional bias to flag
any meaningful performance gap relative to the overall model.
"""

import json
import logging
import os
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
_DEFAULT_SEGMENTS_PATH = _REPO_ROOT / "data" / "baselines" / "bias_segments.json"

# Fixed training-set normalisation constants (must match preprocessing.py)
_AMOUNT_MEAN = 90.8249
_AMOUNT_STD = 250.5032

# A segment is directional-bias flagged when its mean predicted probability
# exceeds this multiple of the overall mean predicted probability.
_DIRECTIONAL_BIAS_MULTIPLIER = 1.5

# Hardcoded fallback — used only when bias_segments.json cannot be loaded.
_DEFAULT_SEGMENTS: list[dict] = [
    {
        "name": "high_amount",
        "feature": "Amount",
        "threshold": 1000,
        "comparison": "gt",
        "fpr_multiplier_limit": 1.5,
        "auprc_ratio_limit": 0.7,
    },
    {
        "name": "low_amount",
        "feature": "Amount",
        "threshold": 1000,
        "comparison": "lte",
        "fpr_multiplier_limit": 2.0,
        "auprc_ratio_limit": 0.7,
    },
    {
        "name": "evening_hour",
        "feature": "hour_of_day",
        "threshold": 18,
        "comparison": "gte",
        "fpr_multiplier_limit": 2.0,
        "auprc_ratio_limit": 0.7,
    },
    {
        "name": "daytime_hour",
        "feature": "hour_of_day",
        "threshold": 18,
        "comparison": "lt",
        "fpr_multiplier_limit": 2.0,
        "auprc_ratio_limit": 0.7,
    },
]

_OPS = {
    "gt": lambda col, t: col > t,
    "gte": lambda col, t: col >= t,
    "lt": lambda col, t: col < t,
    "lte": lambda col, t: col <= t,
}

_OP_SYMBOLS = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


def load_segments(path=None) -> list[dict]:
    """Load bias segment definitions from a JSON file.

    Resolution order:
      1. ``path`` argument (if given)
      2. ``BIAS_SEGMENTS_PATH`` environment variable
      3. ``data/baselines/bias_segments.json``
      4. Hardcoded ``_DEFAULT_SEGMENTS`` fallback (logs a warning)
    """
    if path is None:
        env_path = os.environ.get("BIAS_SEGMENTS_PATH")
        path = Path(env_path) if env_path else _DEFAULT_SEGMENTS_PATH
    try:
        with open(path) as f:
            segments = json.load(f)
        log.info("BiasTestSuite: loaded %d segments from %s", len(segments), path)
        return segments
    except Exception as exc:
        log.warning(
            "BiasTestSuite: failed to load segments from %s: %s — using defaults", path, exc
        )
        return _DEFAULT_SEGMENTS


def _apply_segment_mask(df: pd.DataFrame, seg: dict) -> pd.Series:
    """Return a boolean Series selecting rows that belong to *seg*."""
    feature = seg["feature"]
    threshold = seg["threshold"]
    comparison = seg["comparison"]
    op = _OPS.get(comparison)
    if op is None:
        raise ValueError(
            f"Unknown comparison operator: {comparison!r}. Supported: gt, gte, lt, lte."
        )
    return op(df[feature], threshold)


def _generate_description(seg: dict) -> str:
    symbol = _OP_SYMBOLS.get(seg["comparison"], seg["comparison"])
    return f"{seg['feature']} {symbol} {seg['threshold']}"


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
        """Compute per-segment bias metrics and return a report dict.

        Uses *validation_df* when provided, otherwise the CSV loaded at init.
        Returns the cached placeholder when neither data nor model is available.
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
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        overall_recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        overall_fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
        overall_mean_prob = float(y_proba.mean())

        segments = load_segments()
        bias_segments: list[dict] = []
        flagged_names: list[str] = []
        any_directional_bias = False

        for seg_config in segments:
            seg_name = seg_config["name"]
            description = _generate_description(seg_config)
            fpr_multiplier_limit = seg_config["fpr_multiplier_limit"]
            auprc_ratio_limit = seg_config["auprc_ratio_limit"]

            try:
                mask = _apply_segment_mask(df, seg_config)
            except Exception as exc:
                log.warning(
                    "BiasTestSuite: segment %s mask error: %s — skipping", seg_name, exc
                )
                continue

            y_s = y[mask]
            p_s = y_proba[mask]
            n_samples = int(mask.sum())
            n_fraud = int(y_s.sum())

            if n_samples < 10 or n_fraud == 0:
                bias_segments.append({
                    "segment": seg_name,
                    "description": description,
                    "n_samples": n_samples,
                    "n_fraud": n_fraud,
                    "auprc": None,
                    "fpr": None,
                    "auprc_ratio": None,
                    "fpr_ratio": None,
                    "auprc_flagged": False,
                    "fpr_flagged": False,
                    "directional_bias_flagged": False,
                    "flagged": False,
                    "note": "Insufficient fraud samples for AUPRC",
                })
                continue

            seg_auprc = float(average_precision_score(y_s, p_s))
            y_pred_s = (p_s >= self._bundle.threshold).astype(int)
            cm_s = confusion_matrix(y_s, y_pred_s, labels=[0, 1])
            tn_s, fp_s, fn_s, tp_s = cm_s.ravel()
            seg_fpr = float(fp_s / (fp_s + tn_s)) if (fp_s + tn_s) else 0.0
            seg_mean_prob = float(p_s.mean())

            auprc_ratio = (seg_auprc / overall_auprc) if overall_auprc > 0 else None
            fpr_ratio = (seg_fpr / overall_fpr) if overall_fpr > 0 else None

            auprc_flagged = auprc_ratio is not None and auprc_ratio < auprc_ratio_limit
            fpr_flagged = fpr_ratio is not None and fpr_ratio > fpr_multiplier_limit
            directional_bias_flagged = (
                overall_mean_prob > 0
                and seg_mean_prob > _DIRECTIONAL_BIAS_MULTIPLIER * overall_mean_prob
            )

            flagged = auprc_flagged or fpr_flagged or directional_bias_flagged
            if flagged:
                flagged_names.append(seg_name)
            if directional_bias_flagged:
                any_directional_bias = True

            log.debug(
                "Segment %s: auprc=%.4f fpr=%.6f auprc_ratio=%s fpr_ratio=%s "
                "directional_bias=%s flagged=%s",
                seg_name,
                seg_auprc,
                seg_fpr,
                f"{auprc_ratio:.4f}" if auprc_ratio is not None else "N/A",
                f"{fpr_ratio:.4f}" if fpr_ratio is not None else "N/A",
                directional_bias_flagged,
                flagged,
            )

            bias_segments.append({
                "segment": seg_name,
                "description": description,
                "n_samples": n_samples,
                "n_fraud": n_fraud,
                "auprc": round(seg_auprc, 4),
                "fpr": round(seg_fpr, 6),
                "auprc_ratio": round(auprc_ratio, 4) if auprc_ratio is not None else None,
                "fpr_ratio": round(fpr_ratio, 4) if fpr_ratio is not None else None,
                "auprc_flagged": auprc_flagged,
                "fpr_flagged": fpr_flagged,
                "directional_bias_flagged": directional_bias_flagged,
                "flagged": flagged,
            })

        n_segments_flagged = len(flagged_names)
        any_flagged = n_segments_flagged > 0

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
            "bias_segments": bias_segments,
            "summary": {
                "n_segments_flagged": n_segments_flagged,
                "directional_bias_detected": any_directional_bias,
                "any_flagged": any_flagged,
            },
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
            "bias_segments": [],
            "summary": {
                "n_segments_flagged": 0,
                "directional_bias_detected": False,
                "any_flagged": False,
            },
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
