"""Fraud detection API — FastAPI application deployed on AWS Lambda via Mangum."""

import hashlib
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import boto3
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from src.api.middleware import LoggingMiddleware
from src.api.preprocessing import build_feature_dataframe, set_amount_stats
from src.api.schemas import (
    BiasReportResponse,
    ConfigUpdateRequest,
    DriftReportResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    register_validation_stats,
)
from src.monitoring.audit_logger import AuditLogger
from src.monitoring.bias_tester import BiasTestSuite
from src.monitoring.drift import DriftMonitor
from src.utils import model_loader as _model_loader
from src.utils.model_loader import ModelBundle, load_model_bundle

# ---------------------------------------------------------------------------
# Logging: INFO → stdout, ERROR+ → stderr
# ---------------------------------------------------------------------------
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setLevel(logging.DEBUG)
_stdout_handler.addFilter(lambda r: r.levelno < logging.ERROR)

_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[_stdout_handler, _stderr_handler],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-container singletons (initialised in lifespan, never None after startup)
# ---------------------------------------------------------------------------
_bundle: ModelBundle | None = None
_audit: AuditLogger | None = None
_drift: DriftMonitor | None = None
_bias: BiasTestSuite | None = None

# SHAP is disabled: the Lambda deployment image uses a stripped Python runtime
# whose C-extension ABI is incompatible with the shap wheel built locally.
# Re-enable by building shap inside the container image and setting SHAP_ENABLED=1.
_SHAP_ENABLED = False


def _effective_threshold(active_threshold: float, hour_of_day: float) -> float:
    """Return tightened threshold during high-risk hours (1 AM to 5 AM inclusive)."""
    if int(hour_of_day) in (1, 2, 3, 4, 5):
        return round(active_threshold * 0.85, 4)
    return active_threshold


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bundle, _audit, _drift, _bias

    log.info("Container cold-start — initialising services")

    # Model bundle (fatal: no point serving without a model)
    try:
        _bundle = load_model_bundle()
        set_amount_stats(_bundle.amount_stats)
        register_validation_stats({
            "psi_baseline": _bundle.psi_baseline,
            "amount_stats": _bundle.amount_stats,
        })
        log.info("Validation stats registered (%d PSI features)", len(_bundle.psi_baseline))
    except Exception as exc:
        log.error("FATAL: model bundle failed to load: %s", exc)
        raise

    # Audit logger (fatal: predictions must be auditable)
    try:
        _audit = AuditLogger()
    except Exception as exc:
        log.error("FATAL: AuditLogger failed to initialise: %s", exc)
        raise

    # Drift monitor (fatal: drift detection is required for production)
    try:
        _drift = DriftMonitor()
    except Exception as exc:
        log.error("FATAL: DriftMonitor failed to initialise: %s", exc)
        raise

    # Bias test suite (non-fatal: degrades gracefully — /metrics returns 503)
    try:
        _bias = BiasTestSuite()
    except Exception as exc:
        log.warning("BiasTestSuite failed to initialise (non-fatal): %s", exc)

    log.info("All services ready — application startup complete")
    yield
    log.info("Application shutdown")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud scoring with XGBoost.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness / readiness check."""
    return HealthResponse(
        status="ok" if _bundle is not None else "degraded",
        model_version=_bundle.version if _bundle else "unknown",
        model_loaded=_bundle is not None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest, req: Request) -> PredictionResponse:
    """Score a single transaction and return a fraud probability."""
    if _bundle is None or _audit is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    t0 = time.perf_counter()

    features = build_feature_dataframe(payload, _bundle.feature_names)
    proba = _bundle.model.predict_proba(features)
    fraud_probability: float = float(proba[:, 1][0])

    processing_time_ms = (time.perf_counter() - t0) * 1000

    # Calibrated confidence score from isotonic regression calibrator
    calibrator = _bundle.calibrator
    if calibrator is not None:
        confidence_score = float(calibrator.transform([fraud_probability])[0])
    else:
        log.warning("No calibrator in bundle; using raw probability as confidence_score")
        confidence_score = fraud_probability

    # Time-aware threshold tightening
    hour_of_day = float(features["hour_of_day"].iloc[0])
    effective_thr = _effective_threshold(_model_loader._active_threshold, hour_of_day)

    is_fraud = fraud_probability >= effective_thr
    flagged_for_review = 0.3 <= fraud_probability <= 0.7

    prediction_id = str(uuid.uuid4())
    log.debug(
        "prediction_id=%s effective_threshold=%.4f hour_of_day=%.0f",
        prediction_id, effective_thr, hour_of_day,
    )
    request_ip = req.client.host if req.client else "unknown"

    # SHA-256 of the ordered feature vector — consistent with shap_offline.py
    feature_vector = features.iloc[0].values.astype(np.float64).tobytes()
    prediction_hash = hashlib.sha256(feature_vector).hexdigest()

    # All 32 features: V1-V28 (from payload) + engineered Amount features
    input_features = {
        **{f"V{i}": float(getattr(payload, f"v{i}")) for i in range(1, 29)},
        "Amount": payload.amount,
        "amount_log": float(features["amount_log"].iloc[0]),
        "amount_zscore": float(features["amount_zscore"].iloc[0]),
        "hour_of_day": float(features["hour_of_day"].iloc[0]),
    }

    await _audit.write(
        prediction_id=prediction_id,
        prediction_hash=prediction_hash,
        input_features=input_features,
        fraud_probability=fraud_probability,
        is_fraud=is_fraud,
        shap_values={},
        model_version=_bundle.version,
        confidence_score=confidence_score,
        request_ip=request_ip,
        latency_ms=processing_time_ms,
        threshold_used=effective_thr,
    )

    if flagged_for_review:
        log.info(
            "prediction_id=%s flagged for review (probability=%.4f)",
            prediction_id,
            fraud_probability,
        )

    if payload.anomaly_flags:
        log.warning(
            "prediction_id=%s anomaly_flags=%s",
            prediction_id,
            payload.anomaly_flags,
        )

    if payload.high_amount_flag:
        log.warning(
            "prediction_id=%s high_amount_flag=True amount=%.2f",
            prediction_id,
            payload.amount,
        )

    return PredictionResponse(
        prediction_id=prediction_id,
        is_fraud=is_fraud,
        fraud_probability=fraud_probability,
        confidence_score=confidence_score,
        shap_values={},
        model_version=_bundle.version,
        processing_time_ms=processing_time_ms,
        flagged_for_review=flagged_for_review,
        threshold_used=effective_thr,
        effective_threshold=effective_thr,
        anomaly_flags=payload.anomaly_flags,
        high_amount_flag=payload.high_amount_flag,
    )


@app.get("/explain/{prediction_id}")
async def explain(prediction_id: str) -> dict:
    """Return the stored audit record for a prediction."""
    if _audit is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    record = await _audit.fetch(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id!r} not found")
    return record


@app.get("/override")
async def list_pending_reviews() -> list[dict]:
    """Return all predictions currently flagged for human review.

    Queries the requires-review-index GSI on fraud-audit-log.
    Each record includes prediction_id, fraud_probability, shap_top3,
    amount, requires_review, and timestamp.
    """
    if _audit is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    return await _audit.fetch_pending_reviews()


@app.get("/config")
def get_config() -> dict:
    """Return the current decision threshold and its source."""
    return {
        "threshold": _model_loader._active_threshold,
        "source": _model_loader._threshold_source,
    }


@app.post("/config")
async def update_config(payload: ConfigUpdateRequest, req: Request) -> dict:
    """Update the decision threshold at runtime (requires X-Config-Api-Key header)."""
    api_key = os.environ.get("CONFIG_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="CONFIG_API_KEY not configured")
    if req.headers.get("X-Config-Api-Key") != api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    config_table = os.environ.get("CONFIG_TABLE", "fraud-config")
    region = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")

    try:
        dynamo = boto3.resource("dynamodb", region_name=region)
        dynamo.Table(config_table).put_item(
            Item={"config_key": "threshold", "value": str(payload.threshold)}
        )
    except Exception as exc:
        log.error("Failed to write threshold to DynamoDB: %s", exc)
        raise HTTPException(status_code=503, detail="Failed to update config")

    _model_loader.refresh_threshold()

    return {"threshold": _model_loader._active_threshold, "updated": True}


@app.get("/drift", response_model=DriftReportResponse)
async def drift() -> DriftReportResponse:
    """Compute a PSI-based feature drift report over the 500 most recent predictions."""
    if _audit is None or _drift is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    recent = await _audit.fetch_recent(limit=500)
    report = _drift.compute_report(recent)
    return DriftReportResponse(**report)


@app.get("/metrics", response_model=BiasReportResponse)
def metrics() -> BiasReportResponse:
    """Return the cached bias / fairness report."""
    if _bias is None:
        raise HTTPException(status_code=503, detail="Bias test suite not available")

    return BiasReportResponse(**_bias.cached_report())


# ---------------------------------------------------------------------------
# Lambda entrypoint
# ---------------------------------------------------------------------------
handler = Mangum(app, lifespan="on")
