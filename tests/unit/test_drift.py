"""Tests for the extended DriftMonitor: V-feature PSI, concept drift,
model age, fraud flag rate, and RetrainingRequired signal."""

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import boto3
import pytest
from moto import mock_aws

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_v_psi_baseline() -> dict:
    """V1-V28 PSI baselines with 2 equal bins each."""
    return {
        f"V{i}": {
            "bin_edges": [-5.001, 0.0, 5.001],
            "expected_proportions": [0.5, 0.5],
        }
        for i in range(1, 29)
    }


def _make_full_bundle(trained_days_ago: int = 100, fraud_flag_rate: float = 0.05) -> SimpleNamespace:
    trained_at = (datetime.now(timezone.utc) - timedelta(days=trained_days_ago)).isoformat()
    v_baseline = _make_v_psi_baseline()
    v_baseline["fraud_probability"] = {
        "bin_edges": [0.0, 0.5, 1.0001],
        "expected_proportions": [0.8, 0.2],
    }
    return SimpleNamespace(
        psi_baseline=v_baseline,
        experiment_manifest={
            "trained_at": trained_at,
            "metrics": {"fraud_flag_rate": fraud_flag_rate},
        },
    )


def _make_baseline_dict() -> dict:
    """Minimal 4-feature baseline with equal 2-bin splits."""
    return {
        "Amount": {
            "bin_edges": [0.0, 50.0, 200.001],
            "expected_proportions": [0.5, 0.5],
            "mean": 90.0,
            "std": 50.0,
        },
        "amount_log": {
            "bin_edges": [-0.001, 2.5, 6.001],
            "expected_proportions": [0.5, 0.5],
            "mean": 3.4,
            "std": 1.5,
        },
        "amount_zscore": {
            "bin_edges": [-2.001, 0.0, 5.001],
            "expected_proportions": [0.5, 0.5],
            "mean": 0.0,
            "std": 1.0,
        },
        "hour_of_day": {
            "bin_edges": [-0.001, 12.0, 24.001],
            "expected_proportions": [0.5, 0.5],
            "mean": 12.0,
            "std": 6.9,
        },
    }


def _make_monitor(trained_days_ago: int = 100, fraud_flag_rate: float = 0.05):
    from src.monitoring.drift import DriftMonitor
    return DriftMonitor(
        bundle=_make_full_bundle(trained_days_ago, fraud_flag_rate),
        baseline=_make_baseline_dict(),
    )


# ---------------------------------------------------------------------------
# 1. PSI computed correctly for Amount feature (known input/output)
# ---------------------------------------------------------------------------

def test_amount_psi_is_zero_when_distribution_matches_expected():
    """When actual proportions equal expected, PSI must be exactly 0."""
    monitor = _make_monitor()
    # 5 records in bin-0 [0,50), 5 in bin-1 [50,200] → actual=[0.5,0.5]=expected
    records = [
        {"input_features": {"Amount": 25.0}} for _ in range(5)
    ] + [
        {"input_features": {"Amount": 100.0}} for _ in range(5)
    ]
    report = monitor.compute_report(records)
    amount_entry = next((f for f in report["features"] if f["feature"] == "Amount"), None)
    assert amount_entry is not None, "Amount feature not found in report"
    assert amount_entry["psi"] == pytest.approx(0.0, abs=1e-6)
    assert amount_entry["status"] == "stable"


# ---------------------------------------------------------------------------
# 2. V-feature PSI skipped gracefully when v_features_msgpack absent
# ---------------------------------------------------------------------------

def test_v_feature_psi_skipped_when_no_v_in_input_features():
    """Records without V-keys in input_features must not produce V-feature PSI entries."""
    monitor = _make_monitor()
    # Records only have Amount — no V1..V28
    records = [{"input_features": {"Amount": 25.0}} for _ in range(10)]
    report = monitor.compute_report(records)
    v_entries = [f for f in report["features"] if f["feature"].startswith("V")]
    assert v_entries == [], f"Expected no V-feature entries, got: {v_entries}"


# ---------------------------------------------------------------------------
# 3. ConceptDriftPSI computed from fraud_probability distribution
# ---------------------------------------------------------------------------

def test_concept_drift_psi_computed_from_fraud_probability():
    """compute_concept_drift should return a positive PSI when distribution differs from baseline."""
    monitor = _make_monitor()
    # Baseline: 80% in [0, 0.5), 20% in [0.5, 1.0]
    # All records at 0.8 → 100% in bin-1 → high PSI
    records = [{"fraud_probability": 0.8} for _ in range(60)]
    psi, status = monitor.compute_concept_drift(records)
    assert psi is not None, "Expected PSI value, got None"
    assert psi > 0.0
    assert status in (_STABLE := "stable", _MONITOR := "monitor", _ACTION := "action_required")
    assert status != "stable"  # distributions differ significantly


# ---------------------------------------------------------------------------
# 4. ConceptDriftPSI skipped when fewer than 50 records
# ---------------------------------------------------------------------------

def test_concept_drift_skipped_when_fewer_than_50_records():
    monitor = _make_monitor()
    records = [{"fraud_probability": 0.8} for _ in range(30)]
    psi, status = monitor.compute_concept_drift(records)
    assert psi is None
    assert status == "stable"


# ---------------------------------------------------------------------------
# 5. ModelAgeDays computed correctly from trained_at timestamp
# ---------------------------------------------------------------------------

def test_model_age_computed_from_trained_at():
    monitor = _make_monitor(trained_days_ago=100)
    age = monitor.compute_model_age()
    # Allow ±1 day for test execution across midnight
    assert 99 <= age <= 101, f"Expected age ~100 days, got {age}"


def test_model_age_zero_when_trained_at_missing():
    from src.monitoring.drift import DriftMonitor
    bundle = SimpleNamespace(
        psi_baseline={},
        experiment_manifest={"metrics": {}},  # no trained_at
    )
    monitor = DriftMonitor(bundle=bundle, baseline={})
    assert monitor.compute_model_age() == 0


# ---------------------------------------------------------------------------
# 6. FraudFlagRateDelta computed correctly against baseline
# ---------------------------------------------------------------------------

def test_fraud_flag_rate_delta_computed():
    """Delta should be |current_rate - baseline_rate| rounded to 4 dp."""
    monitor = _make_monitor(fraud_flag_rate=0.05)
    # 3 fraud in 100 records → current_rate = 0.03; baseline = 0.05 → delta = 0.02
    records = [{"is_fraud": True}] * 3 + [{"is_fraud": False}] * 97
    delta = monitor.compute_fraud_flag_rate_delta(records)
    assert delta is not None
    assert delta == pytest.approx(0.02, abs=1e-4)


def test_fraud_flag_rate_delta_skipped_when_fewer_than_100_records():
    monitor = _make_monitor()
    records = [{"is_fraud": False}] * 50
    delta = monitor.compute_fraud_flag_rate_delta(records)
    assert delta is None


# ---------------------------------------------------------------------------
# 7. RetrainingRequired = 1 when 2 or more signals fire
# ---------------------------------------------------------------------------

@mock_aws
def test_retraining_required_1_when_two_signals_fire(monkeypatch):
    """signal_1 (Amount drift) + signal_2 (model age > 90) → RetrainingRequired=1."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    # Model trained 110 days ago → signal_2 = True
    # All Amount in bin-0 (expected 0.5/0.5) → high Amount PSI → signal_1 = True
    monitor = _make_monitor(trained_days_ago=110, fraud_flag_rate=0.05)

    # 200 records: all Amount=25 (bin-0 only → PSI >> 0.2)
    # fraud_probability split 80/20 to match baseline → concept drift PSI ≈ 0
    # is_fraud≈5% to match baseline → fraud flag rate delta ≈ 0
    records_500 = (
        [{"input_features": {"Amount": 25.0}, "fraud_probability": 0.1, "is_fraud": False}] * 160
        + [{"input_features": {"Amount": 25.0}, "fraud_probability": 0.8, "is_fraud": True}] * 40
    )
    records_1000 = records_500 * 5  # 1000 records, same distribution

    result = monitor.run_full_drift_check(records_500, records_1000)

    assert result["retraining_required"] is True
    assert result["signals"]["feature_drift"] is True
    assert result["signals"]["model_age_over_90"] is True

    # Verify CloudWatch metric
    cw = boto3.client("cloudwatch", region_name="us-east-1")
    stats = cw.get_metric_statistics(
        Namespace="FraudDetection",
        MetricName="RetrainingRequired",
        StartTime="2000-01-01T00:00:00Z",
        EndTime="2100-01-01T00:00:00Z",
        Period=3600,
        Statistics=["Sum"],
    )
    assert stats["Datapoints"], "Expected RetrainingRequired datapoint in CloudWatch"
    assert stats["Datapoints"][0]["Sum"] == 1.0


# ---------------------------------------------------------------------------
# 8. RetrainingRequired = 0 when fewer than 2 signals fire
# ---------------------------------------------------------------------------

@mock_aws
def test_retraining_required_0_when_fewer_than_two_signals_fire(monkeypatch):
    """No signals fire → RetrainingRequired=0."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    # Model trained 20 days ago → signal_2 = False
    monitor = _make_monitor(trained_days_ago=20, fraud_flag_rate=0.05)

    # 200 records with equal Amount distribution → Amount PSI = 0 → signal_1 = False
    # fraud_probability split 80/20 matching baseline → concept drift PSI ≈ 0 → signal_4 = False
    # fraud rate 5% matches baseline → signal_3 ≈ False
    records_500 = (
        [{"input_features": {"Amount": 25.0}, "fraud_probability": 0.1, "is_fraud": False}] * 80
        + [{"input_features": {"Amount": 100.0}, "fraud_probability": 0.1, "is_fraud": False}] * 80
        + [{"input_features": {"Amount": 25.0}, "fraud_probability": 0.8, "is_fraud": False}] * 20
        + [{"input_features": {"Amount": 100.0}, "fraud_probability": 0.8, "is_fraud": False}] * 20
    )
    records_1000 = records_500 * 5

    result = monitor.run_full_drift_check(records_500, records_1000)

    assert result["retraining_required"] is False

    cw = boto3.client("cloudwatch", region_name="us-east-1")
    stats = cw.get_metric_statistics(
        Namespace="FraudDetection",
        MetricName="RetrainingRequired",
        StartTime="2000-01-01T00:00:00Z",
        EndTime="2100-01-01T00:00:00Z",
        Period=3600,
        Statistics=["Sum"],
    )
    assert stats["Datapoints"], "Expected RetrainingRequired datapoint in CloudWatch"
    assert stats["Datapoints"][0]["Sum"] == 0.0


# ---------------------------------------------------------------------------
# 9. All CloudWatch metrics published with correct namespace "FraudDetection"
# ---------------------------------------------------------------------------

@mock_aws
def test_all_cloudwatch_metrics_published_with_correct_namespace(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    monitor = _make_monitor(trained_days_ago=50, fraud_flag_rate=0.05)

    # 200 records with Amount split 50/50, fraud_probability 80/20, is_fraud ~5%
    records_500 = (
        [{"input_features": {"Amount": 25.0}, "fraud_probability": 0.1, "is_fraud": False}] * 80
        + [{"input_features": {"Amount": 100.0}, "fraud_probability": 0.1, "is_fraud": False}] * 80
        + [{"input_features": {"Amount": 25.0}, "fraud_probability": 0.8, "is_fraud": True}] * 20
        + [{"input_features": {"Amount": 100.0}, "fraud_probability": 0.8, "is_fraud": True}] * 20
    )
    records_1000 = records_500 * 5

    monitor.run_full_drift_check(records_500, records_1000)

    cw = boto3.client("cloudwatch", region_name="us-east-1")
    all_metrics = cw.list_metrics(Namespace="FraudDetection")
    published_names = {m["MetricName"] for m in all_metrics["Metrics"]}

    assert "FraudPSI" in published_names, f"FraudPSI not published; got: {published_names}"
    assert "ConceptDriftPSI" in published_names
    assert "ModelAgeDays" in published_names
    assert "FraudFlagRateDelta" in published_names
    assert "RetrainingRequired" in published_names
