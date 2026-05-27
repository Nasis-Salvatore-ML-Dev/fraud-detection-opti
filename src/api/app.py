"""Fraud detection API — FastAPI application deployed on AWS Lambda via Mangum."""

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

from src.api.middleware import LoggingMiddleware
from src.api.preprocessing import build_feature_dataframe, set_amount_stats
from src.api.schemas import (
    BiasReportResponse,
    DriftReportResponse,
    HealthResponse,
    OverrideRequest,
    OverrideResponse,
    PredictionRequest,
    PredictionResponse,
    register_validation_stats,
)
from src.monitoring.audit_logger import AuditLogger
from src.monitoring.bias_tester import BiasTestSuite
from src.monitoring.drift import DriftMonitor
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

    is_fraud = fraud_probability >= _bundle.threshold
    flagged_for_review = 0.3 <= fraud_probability <= 0.7
    confidence_score = 0.9 if (fraud_probability >= 0.7 or fraud_probability <= 0.3) else 0.4

    prediction_id = str(uuid.uuid4())
    request_ip = req.client.host if req.client else "unknown"

    input_features = {
        "Amount": payload.amount,
        "amount_log": float(features["amount_log"].iloc[0]),
        "amount_zscore": float(features["amount_zscore"].iloc[0]),
        "hour_of_day": float(features["hour_of_day"].iloc[0]),
    }

    await _audit.write(
        prediction_id=prediction_id,
        input_features=input_features,
        fraud_probability=fraud_probability,
        is_fraud=is_fraud,
        shap_values={},
        model_version=_bundle.version,
        confidence_score=confidence_score,
        request_ip=request_ip,
        latency_ms=processing_time_ms,
        threshold_used=_bundle.threshold,
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
        threshold_used=_bundle.threshold,
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


@app.post("/override", response_model=OverrideResponse)
async def override(request: OverrideRequest) -> OverrideResponse:
    """Record a human override decision for a flagged prediction."""
    if _audit is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    record = await _audit.fetch(request.prediction_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction {request.prediction_id!r} not found",
        )

    await _audit.flag_override(
        request.prediction_id,
        reason=request.reason or "",
        reviewer_notes=request.notes,
    )
    log.info("Override recorded for prediction_id=%s", request.prediction_id)

    return OverrideResponse(
        prediction_id=request.prediction_id,
        status="overridden",
        message="Prediction flagged for human review and override recorded.",
    )


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
