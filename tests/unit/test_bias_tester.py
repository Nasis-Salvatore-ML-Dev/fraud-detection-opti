"""Tests for the extended BiasTestSuite: config-driven segments, directional bias."""

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["amount_log", "amount_zscore", "hour_of_day"]
_AMOUNT_MEAN = 90.8249
_AMOUNT_STD = 250.5032


def _make_bundle():
    bundle = MagicMock()
    bundle.version = "v1"
    bundle.threshold = 0.5
    bundle.feature_names = _FEATURE_NAMES
    return bundle


def _make_df(amounts, classes, hours):
    """Build a DataFrame with all columns required by BiasTestSuite.run()."""
    amounts = np.asarray(amounts, dtype=float)
    classes = np.asarray(classes, dtype=int)
    hours = np.asarray(hours, dtype=float)
    n = len(amounts)
    return pd.DataFrame({
        "Amount": amounts,
        "Time": hours * 3600,
        "Class": classes,
        "hour_of_day": hours,
        "amount_log": np.log1p(amounts),
        "amount_zscore": (amounts - _AMOUNT_MEAN) / _AMOUNT_STD,
        **{f"V{i}": np.zeros(n) for i in range(1, 29)},
    })


def _seg(name="high_amount", feature="Amount", threshold=500, comparison="gt",
         fpr_multiplier_limit=2.0, auprc_ratio_limit=0.7):
    return {
        "name": name,
        "feature": feature,
        "threshold": threshold,
        "comparison": comparison,
        "fpr_multiplier_limit": fpr_multiplier_limit,
        "auprc_ratio_limit": auprc_ratio_limit,
    }


# ---------------------------------------------------------------------------
# 1. load_segments returns 4 named entries from the default JSON file
# ---------------------------------------------------------------------------

def test_load_segments_from_default_path():
    from src.monitoring.bias_tester import load_segments
    segs = load_segments()
    assert len(segs) >= 4
    names = {s["name"] for s in segs}
    assert {"high_amount", "low_amount", "evening_hour", "daytime_hour"}.issubset(names)
    for s in segs:
        assert "fpr_multiplier_limit" in s
        assert "auprc_ratio_limit" in s


# ---------------------------------------------------------------------------
# 2. load_segments respects BIAS_SEGMENTS_PATH env var
# ---------------------------------------------------------------------------

def test_load_segments_from_env_path(tmp_path, monkeypatch):
    from src.monitoring.bias_tester import load_segments
    custom = [_seg(name="custom_seg", threshold=100)]
    seg_file = tmp_path / "custom_segs.json"
    seg_file.write_text(json.dumps(custom))
    monkeypatch.setenv("BIAS_SEGMENTS_PATH", str(seg_file))
    segs = load_segments()
    assert len(segs) == 1
    assert segs[0]["name"] == "custom_seg"


# ---------------------------------------------------------------------------
# 3. _apply_segment_mask: gt comparison
# ---------------------------------------------------------------------------

def test_apply_segment_mask_gt_comparison():
    from src.monitoring.bias_tester import _apply_segment_mask
    df = pd.DataFrame({"Amount": [500.0, 1000.0, 1500.0]})
    mask = _apply_segment_mask(df, _seg(comparison="gt", threshold=1000))
    assert list(mask) == [False, False, True]


# ---------------------------------------------------------------------------
# 4. _apply_segment_mask: lte comparison
# ---------------------------------------------------------------------------

def test_apply_segment_mask_lte_comparison():
    from src.monitoring.bias_tester import _apply_segment_mask
    df = pd.DataFrame({"Amount": [500.0, 1000.0, 1500.0]})
    mask = _apply_segment_mask(df, _seg(comparison="lte", threshold=1000))
    assert list(mask) == [True, True, False]


# ---------------------------------------------------------------------------
# 5. Per-segment report includes auprc_ratio and fpr_ratio fields
# ---------------------------------------------------------------------------

def test_per_segment_report_includes_ratios(monkeypatch):
    from src.monitoring.bias_tester import BiasTestSuite

    # Dataset: 100 rows, 50 high_amount / 50 low_amount, 10 fraud each
    n = 100
    amounts = np.array([600.0] * 50 + [400.0] * 50)
    classes = np.array([1] * 10 + [0] * 40 + [1] * 10 + [0] * 40)
    hours = np.full(n, 12.0)
    proba = np.full(n, 0.3)

    bundle = _make_bundle()
    bundle.model.predict_proba.return_value = np.column_stack([1 - proba, proba])
    monkeypatch.setattr("src.monitoring.bias_tester.load_model_bundle", lambda: bundle)
    monkeypatch.setattr(
        "src.monitoring.bias_tester.load_segments",
        lambda: [_seg(name="high_amount", threshold=500, comparison="gt")],
    )

    suite = BiasTestSuite()
    report = suite.run(_make_df(amounts, classes, hours))

    computed = [s for s in report["bias_segments"] if s.get("auprc") is not None]
    assert computed, "Expected at least one segment with computed auprc"
    for s in computed:
        assert "auprc_ratio" in s, "auprc_ratio missing"
        assert "fpr_ratio" in s, "fpr_ratio missing"


# ---------------------------------------------------------------------------
# 6. auprc_flagged=True when segment AUPRC ratio falls below limit
# ---------------------------------------------------------------------------

def test_auprc_flagged_when_ratio_below_limit(monkeypatch):
    from src.monitoring.bias_tester import BiasTestSuite

    # high_amount (Amount > 200): 50 rows, 10 fraud — model gives 0.1 to all
    # low_amount  (Amount <= 200): 50 rows, 10 fraud — fraud get 0.9, rest get 0.1
    # high_amount AUPRC ≈ 0.2 (random), overall AUPRC >> 0.2 → ratio < 0.7 → flagged
    n = 100
    amounts = np.array([500.0] * 50 + [100.0] * 50)
    classes = np.array([1] * 10 + [0] * 40 + [1] * 10 + [0] * 40)
    hours = np.full(n, 12.0)
    proba = np.array([0.1] * 50 + [0.9] * 10 + [0.1] * 40)

    bundle = _make_bundle()
    bundle.model.predict_proba.return_value = np.column_stack([1 - proba, proba])
    monkeypatch.setattr("src.monitoring.bias_tester.load_model_bundle", lambda: bundle)
    monkeypatch.setattr(
        "src.monitoring.bias_tester.load_segments",
        lambda: [_seg(name="high_amount", threshold=200, comparison="gt", auprc_ratio_limit=0.7)],
    )

    suite = BiasTestSuite()
    report = suite.run(_make_df(amounts, classes, hours))

    seg = next(s for s in report["bias_segments"] if s["segment"] == "high_amount")
    assert seg["auprc_flagged"] is True, f"Expected auprc_flagged=True, got {seg}"
    assert seg["auprc_ratio"] < 0.7


# ---------------------------------------------------------------------------
# 7. fpr_flagged=True when segment FPR ratio exceeds the multiplier limit
# ---------------------------------------------------------------------------

def test_fpr_flagged_when_ratio_above_limit(monkeypatch):
    from src.monitoring.bias_tester import BiasTestSuite

    # high_amount (Amount > 500): 20 rows, 5 fraud, 15 non-fraud — all get proba=0.9
    #   → TP=5, FP=15, seg_fpr = 15/15 = 1.0
    # low_amount (Amount <= 500): 80 rows, 5 fraud, 75 non-fraud
    #   → fraud get 0.9, non-fraud get 0.1; FP=0, TN=75
    # overall_fpr = 15/(15+75) ≈ 0.167 → fpr_ratio ≈ 6.0 > 1.5 → fpr_flagged
    n = 100
    amounts = np.array([600.0] * 20 + [400.0] * 80)
    classes = np.array([1] * 5 + [0] * 15 + [1] * 5 + [0] * 75)
    hours = np.full(n, 12.0)
    proba = np.array([0.9] * 20 + [0.9] * 5 + [0.1] * 75)

    bundle = _make_bundle()
    bundle.model.predict_proba.return_value = np.column_stack([1 - proba, proba])
    monkeypatch.setattr("src.monitoring.bias_tester.load_model_bundle", lambda: bundle)
    monkeypatch.setattr(
        "src.monitoring.bias_tester.load_segments",
        lambda: [
            _seg(name="high_amount", threshold=500, comparison="gt", fpr_multiplier_limit=1.5)
        ],
    )

    suite = BiasTestSuite()
    report = suite.run(_make_df(amounts, classes, hours))

    seg = next(s for s in report["bias_segments"] if s["segment"] == "high_amount")
    assert seg["fpr_flagged"] is True, f"Expected fpr_flagged=True, got {seg}"
    assert seg["fpr_ratio"] > 1.5


# ---------------------------------------------------------------------------
# 8. directional_bias_flagged=True when segment mean prob > 1.5x overall
# ---------------------------------------------------------------------------

def test_directional_bias_flagged_when_mean_above_threshold(monkeypatch):
    from src.monitoring.bias_tester import BiasTestSuite

    # high_amount (Amount > 500): 10 rows, 2 fraud — all proba=0.9
    # low_amount  (Amount <= 500): 90 rows, 5 fraud — all proba=0.1
    # overall_mean_prob = (10×0.9 + 90×0.1)/100 = 0.18
    # seg_mean_prob = 0.9 > 1.5×0.18 = 0.27 → directional_bias_flagged=True
    n = 100
    amounts = np.array([600.0] * 10 + [400.0] * 90)
    classes = np.array([1] * 2 + [0] * 8 + [1] * 5 + [0] * 85)
    hours = np.full(n, 12.0)
    proba = np.array([0.9] * 10 + [0.1] * 90)

    bundle = _make_bundle()
    bundle.model.predict_proba.return_value = np.column_stack([1 - proba, proba])
    monkeypatch.setattr("src.monitoring.bias_tester.load_model_bundle", lambda: bundle)
    monkeypatch.setattr(
        "src.monitoring.bias_tester.load_segments",
        lambda: [_seg(name="high_amount", threshold=500, comparison="gt")],
    )

    suite = BiasTestSuite()
    report = suite.run(_make_df(amounts, classes, hours))

    seg = next(s for s in report["bias_segments"] if s["segment"] == "high_amount")
    assert seg["directional_bias_flagged"] is True, f"Expected directional_bias_flagged=True, got {seg}"
    assert report["summary"]["directional_bias_detected"] is True


# ---------------------------------------------------------------------------
# 9. Report contains a summary section with correct structure
# ---------------------------------------------------------------------------

def test_report_summary_section_present(monkeypatch):
    from src.monitoring.bias_tester import BiasTestSuite

    n = 100
    amounts = np.array([600.0] * 50 + [400.0] * 50)
    classes = np.array([1] * 10 + [0] * 40 + [1] * 10 + [0] * 40)
    hours = np.full(n, 12.0)
    proba = np.full(n, 0.3)

    bundle = _make_bundle()
    bundle.model.predict_proba.return_value = np.column_stack([1 - proba, proba])
    monkeypatch.setattr("src.monitoring.bias_tester.load_model_bundle", lambda: bundle)
    monkeypatch.setattr(
        "src.monitoring.bias_tester.load_segments",
        lambda: [_seg(name="high_amount", threshold=500, comparison="gt")],
    )

    suite = BiasTestSuite()
    report = suite.run(_make_df(amounts, classes, hours))

    assert "summary" in report, "summary key missing from report"
    summary = report["summary"]
    assert "n_segments_flagged" in summary
    assert "directional_bias_detected" in summary
    assert "any_flagged" in summary
    assert isinstance(summary["n_segments_flagged"], int)
    assert isinstance(summary["directional_bias_detected"], bool)
    assert isinstance(summary["any_flagged"], bool)


# ---------------------------------------------------------------------------
# 10. directional_bias_flagged=False when segment mean prob equals overall mean
# ---------------------------------------------------------------------------

def test_no_directional_bias_when_mean_equal(monkeypatch):
    from src.monitoring.bias_tester import BiasTestSuite

    # All rows get the same probability → seg_mean == overall_mean → no directional bias
    n = 100
    amounts = np.array([600.0] * 50 + [400.0] * 50)
    classes = np.array([1] * 10 + [0] * 40 + [1] * 10 + [0] * 40)
    hours = np.full(n, 12.0)
    proba = np.full(n, 0.3)

    bundle = _make_bundle()
    bundle.model.predict_proba.return_value = np.column_stack([1 - proba, proba])
    monkeypatch.setattr("src.monitoring.bias_tester.load_model_bundle", lambda: bundle)
    monkeypatch.setattr(
        "src.monitoring.bias_tester.load_segments",
        lambda: [_seg(name="high_amount", threshold=500, comparison="gt")],
    )

    suite = BiasTestSuite()
    report = suite.run(_make_df(amounts, classes, hours))

    seg = next(s for s in report["bias_segments"] if s["segment"] == "high_amount")
    assert seg["directional_bias_flagged"] is False
    assert report["summary"]["directional_bias_detected"] is False
