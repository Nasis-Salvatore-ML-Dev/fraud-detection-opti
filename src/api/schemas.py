"""Pydantic v2 request/response schemas for the fraud detection API."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Validation-stats registry
# Populated at startup by register_validation_stats(); never touched at
# import time. Validators degrade gracefully when the dict is empty.
# ---------------------------------------------------------------------------
_VALIDATION_STATS: dict = {}


def register_validation_stats(bundle: dict) -> None:
    """Extract per-feature bounds and amount stats from the model bundle.

    Reads bundle["psi_baseline"] for V-feature bounds (bin_edges[0] and
    bin_edges[-1] as outer bounds) and bundle["amount_stats"] for the Amount
    z-score baseline.  Stores results in _VALIDATION_STATS.
    """
    global _VALIDATION_STATS

    psi = bundle.get("psi_baseline", {})
    amount_stats = bundle.get("amount_stats", {})

    v_bounds: dict[str, tuple[float, float]] = {}
    for i in range(1, 29):
        key = f"V{i}"
        if key in psi:
            edges = psi[key]["bin_edges"]
            v_bounds[key] = (float(edges[0]), float(edges[-1]))

    _VALIDATION_STATS = {
        "v_bounds": v_bounds,
        "amount_mean": amount_stats.get("mean"),
        "amount_std": amount_stats.get("std"),
    }


def get_validation_stats() -> dict:
    """Return the registered validation stats, or an empty dict if not yet set."""
    return _VALIDATION_STATS


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Input features for a single transaction fraud prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "time": 406.0,
                "amount": 149.62,
                "v1": -1.3598071,
                "v2": -0.0727812,
                "v3": 2.5363467,
                "v4": 1.3781553,
                "v5": -0.3383207,
                "v6": 0.4623879,
                "v7": 0.2395986,
                "v8": 0.0986979,
                "v9": 0.3637870,
                "v10": 0.0907941,
                "v11": -0.5515995,
                "v12": -0.6178009,
                "v13": -0.9913898,
                "v14": -0.3111694,
                "v15": 1.4681770,
                "v16": -0.4704005,
                "v17": 0.2079708,
                "v18": 0.0257905,
                "v19": 0.4039936,
                "v20": 0.2514121,
                "v21": -0.0183068,
                "v22": 0.2778376,
                "v23": -0.1104739,
                "v24": 0.0669280,
                "v25": 0.1285394,
                "v26": -0.1891148,
                "v27": 0.1336559,
                "v28": -0.0210531,
            }
        }
    )

    time: float = Field(description="Seconds elapsed since the first transaction in the dataset")
    amount: float = Field(ge=0, description="Transaction amount in dollars")

    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float

    # Computed by the validator below; never required in the request body.
    anomaly_flags: list[str] = Field(default_factory=list)
    high_amount_flag: bool = False

    @model_validator(mode="after")
    def _check_outliers(self) -> "PredictionRequest":
        """Flag out-of-bounds V features and high-amount transactions.

        Reads _VALIDATION_STATS set at startup.  If the stats are not yet
        registered (e.g. during testing or before startup completes), the
        validator returns self unchanged — never raises ValidationError.
        """
        stats = get_validation_stats()
        if not stats:
            return self

        flags: list[str] = []
        v_bounds = stats.get("v_bounds", {})
        for i in range(1, 29):
            field_name = f"v{i}"
            key = f"V{i}"
            if key in v_bounds:
                value: float = getattr(self, field_name)
                lo, hi = v_bounds[key]
                if value < lo or value > hi:
                    flags.append(field_name)

        object.__setattr__(self, "anomaly_flags", flags)

        amount_mean = stats.get("amount_mean")
        amount_std = stats.get("amount_std")
        if amount_mean is not None and amount_std is not None and amount_std > 0:
            zscore = (self.amount - amount_mean) / amount_std
            if zscore > 5:
                object.__setattr__(self, "high_amount_flag", True)

        return self


class PredictionResponse(BaseModel):
    """Result of a fraud prediction for a single transaction."""

    prediction_id: str
    is_fraud: bool
    fraud_probability: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    shap_values: dict[str, float]
    model_version: str
    processing_time_ms: float
    flagged_for_review: bool = Field(
        description="True when fraud_probability is in [0.3, 0.7] (uncertain region)"
    )
    threshold_used: float
    effective_threshold: float = 0.0
    anomaly_flags: list[str] = Field(default_factory=list)
    high_amount_flag: bool = False


class ConfigUpdateRequest(BaseModel):
    """Request body for updating the decision threshold at runtime."""

    threshold: float = Field(gt=0, lt=1, description="Decision threshold, must be in (0, 1)")


class OverrideRequest(BaseModel):
    """Request to override a model prediction with a human decision."""

    prediction_id: str
    reason: Optional[str] = None
    notes: Optional[str] = None


class OverrideResponse(BaseModel):
    """Confirmation of a prediction override."""

    prediction_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    """API and model health status."""

    status: str
    model_version: str
    model_loaded: bool
    timestamp: str


class BiasReportResponse(BaseModel):
    """Fairness / bias audit report for the deployed model."""

    model_version: str
    overall_auprc: float
    overall_recall: float
    overall_fpr: float
    bias_segments: list[dict]
    computed_at: str
    recommendation: Optional[str] = None


class DriftReportResponse(BaseModel):
    """Feature and prediction drift report."""

    computed_at: str
    n_recent_predictions: int
    overall_status: str = Field(description="One of: stable, monitor, action_required")
    features: list[dict]
    recommendation: str
