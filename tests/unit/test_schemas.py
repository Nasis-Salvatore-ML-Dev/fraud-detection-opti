"""Tests for schema-level range checks and outlier detection."""

import pytest

import src.api.schemas as schemas_module
from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    get_validation_stats,
    register_validation_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(**overrides) -> PredictionRequest:
    """Return a valid PredictionRequest with all V features = 0.0."""
    base: dict = {
        "time": 100.0,
        "amount": 50.0,
        **{f"v{i}": 0.0 for i in range(1, 29)},
    }
    base.update(overrides)
    return PredictionRequest(**base)


def _fake_bundle(
    v_lo: float = -10.0,
    v_hi: float = 10.0,
    amount_mean: float = 50.0,
    amount_std: float = 10.0,
) -> dict:
    """Return a minimal bundle dict with PSI baseline and amount_stats."""
    psi_baseline = {
        f"V{i}": {
            "bin_edges": [v_lo, 0.0, v_hi],
            "expected_proportions": [0.5, 0.5],
        }
        for i in range(1, 29)
    }
    return {
        "psi_baseline": psi_baseline,
        "amount_stats": {"mean": amount_mean, "std": amount_std},
    }


@pytest.fixture(autouse=True)
def reset_validation_stats(monkeypatch):
    """Isolate each test: start with empty _VALIDATION_STATS."""
    monkeypatch.setattr(schemas_module, "_VALIDATION_STATS", {})
    yield


# ---------------------------------------------------------------------------
# 1. anomaly_flags populated when V1 exceeds registered bounds
# ---------------------------------------------------------------------------

def test_anomaly_flags_populated_when_v1_out_of_bounds():
    register_validation_stats(_fake_bundle(v_lo=-10.0, v_hi=10.0))
    req = _make_request(v1=999.0)  # far outside [-10, 10]
    assert "v1" in req.anomaly_flags


# ---------------------------------------------------------------------------
# 2. anomaly_flags empty when all V features within bounds
# ---------------------------------------------------------------------------

def test_anomaly_flags_empty_when_all_within_bounds():
    register_validation_stats(_fake_bundle(v_lo=-10.0, v_hi=10.0))
    req = _make_request()  # all V features = 0.0, well inside bounds
    assert req.anomaly_flags == []


# ---------------------------------------------------------------------------
# 3. high_amount_flag True when Amount z-score > 5
# ---------------------------------------------------------------------------

def test_high_amount_flag_true_when_zscore_exceeds_5():
    # mean=50, std=10 → z-score of 151 = (151-50)/10 = 10.1 > 5
    register_validation_stats(_fake_bundle(amount_mean=50.0, amount_std=10.0))
    req = _make_request(amount=151.0)
    assert req.high_amount_flag is True


# ---------------------------------------------------------------------------
# 4. high_amount_flag False when Amount z-score <= 5
# ---------------------------------------------------------------------------

def test_high_amount_flag_false_when_zscore_within_bounds():
    # mean=50, std=10 → z-score of 95 = (95-50)/10 = 4.5 ≤ 5
    register_validation_stats(_fake_bundle(amount_mean=50.0, amount_std=10.0))
    req = _make_request(amount=95.0)
    assert req.high_amount_flag is False


# ---------------------------------------------------------------------------
# 5. Validators skip gracefully when _VALIDATION_STATS is empty
# ---------------------------------------------------------------------------

def test_validators_skip_gracefully_when_stats_not_registered():
    # autouse fixture ensures _VALIDATION_STATS is {} — no register call here
    req = _make_request(v1=9999.0, amount=1_000_000.0)
    assert req.anomaly_flags == []
    assert req.high_amount_flag is False


# ---------------------------------------------------------------------------
# 6. Response schema includes anomaly_flags and high_amount_flag
# ---------------------------------------------------------------------------

def test_response_schema_includes_anomaly_fields():
    response = PredictionResponse(
        prediction_id="test-id",
        is_fraud=False,
        fraud_probability=0.1,
        confidence_score=0.9,
        shap_values={},
        model_version="v1",
        processing_time_ms=10.0,
        flagged_for_review=False,
        threshold_used=0.5,
        anomaly_flags=["v3", "v7"],
        high_amount_flag=True,
    )
    assert response.anomaly_flags == ["v3", "v7"]
    assert response.high_amount_flag is True


def test_response_schema_defaults_to_empty_flags():
    response = PredictionResponse(
        prediction_id="test-id",
        is_fraud=False,
        fraud_probability=0.1,
        confidence_score=0.9,
        shap_values={},
        model_version="v1",
        processing_time_ms=10.0,
        flagged_for_review=False,
        threshold_used=0.5,
    )
    assert response.anomaly_flags == []
    assert response.high_amount_flag is False


# ---------------------------------------------------------------------------
# 7. Valid request passes through unchanged (no false positives)
# ---------------------------------------------------------------------------

def test_valid_request_no_false_positives():
    register_validation_stats(_fake_bundle(v_lo=-10.0, v_hi=10.0))
    # All V features = 0.0 (inside bounds), amount = 50.0 (z-score = 0)
    req = _make_request()
    assert req.anomaly_flags == []
    assert req.high_amount_flag is False
