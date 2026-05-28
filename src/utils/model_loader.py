"""Load the fraud detection model for serving.

Loading order:
  1. Always load the pkl bundle from local path or S3 (provides metadata +
     fallback XGBoost model).
  2. Additionally attempt to load model.onnx from local path or S3 for faster
     ONNX inference.  If model.onnx is unavailable, the XGBoost model from the
     pkl bundle is used instead and a warning is logged.

ModelBundle.predict() uses ONNX when available, XGBoost otherwise.
shap_background.pkl is never loaded by this module.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from joblib import load as _joblib_load
import numpy as np
from xgboost import XGBClassifier

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_MODEL_PATH = "/tmp/xgboost_fraud_v1.pkl"
_DEFAULT_S3_BUCKET = "fraud-model-artifacts-209998132741"
_DEFAULT_S3_KEY = "models/xgboost_fraud_v1.pkl"

_DEFAULT_ONNX_PATH = "/tmp/model.onnx"
_DEFAULT_ONNX_S3_KEY = "models/model.onnx"

_LOCAL_BASELINE_PATH = _REPO_ROOT / "data" / "baselines" / "training_baseline.json"
_TMP_BASELINE_PATH = Path("/tmp/training_baseline.json")
_S3_BASELINE_KEY = "baselines/training_baseline.json"

_REQUIRED_KEYS = {"model", "feature_names", "threshold", "version"}

# Runtime-adjustable decision threshold — updated at startup and via /config.
_active_threshold: float = 0.5
_threshold_source: str = "bundle"


@dataclass
class ModelBundle:
    model: XGBClassifier
    feature_names: list[str]
    threshold: float
    version: str
    amount_stats: dict = field(default_factory=dict)
    psi_baseline: dict = field(default_factory=dict)
    experiment_manifest: dict = field(default_factory=dict)
    calibrator: Any = field(default=None)
    # Set after construction by load_model_bundle(); excluded from __init__
    _onnx_session: Any = field(default=None, init=False, repr=False)
    _onnx_input_name: str = field(default="", init=False, repr=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probabilities as a 1-D array of shape (n,).

        Uses the ONNX runtime when available; falls back to XGBoost otherwise.

        Raises:
            RuntimeError: If neither an ONNX session nor an XGBoost model is loaded.
        """
        if self._onnx_session is not None:
            X_f32 = np.asarray(X, dtype=np.float32)
            outputs = self._onnx_session.run(None, {self._onnx_input_name: X_f32})
            return outputs[1][:, 1]
        if self.model is not None:
            return self.model.predict_proba(X)[:, 1]
        raise RuntimeError(
            "No model available for inference — neither ONNX session nor XGBoost pkl is loaded."
        )


def _load_threshold_from_dynamodb(bundle_threshold: float) -> None:
    """Load threshold from fraud-config DynamoDB table; fall back to bundle value."""
    global _active_threshold, _threshold_source
    import boto3

    config_table = os.environ.get("CONFIG_TABLE", "fraud-config")
    region = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")

    try:
        dynamo = boto3.resource("dynamodb", region_name=region)
        response = dynamo.Table(config_table).get_item(Key={"config_key": "threshold"})
        item = response.get("Item")
        if item is not None:
            _active_threshold = float(item["value"])
            _threshold_source = "dynamodb"
            log.info("Threshold loaded from DynamoDB: %.4f", _active_threshold)
            return
    except Exception as exc:
        log.warning("DynamoDB threshold lookup failed, falling back to bundle: %s", exc)

    _active_threshold = bundle_threshold
    _threshold_source = "bundle"
    log.warning("Using bundle threshold: %.4f", _active_threshold)


def refresh_threshold() -> float:
    """Re-read threshold from DynamoDB and update _active_threshold. Never raises."""
    global _active_threshold, _threshold_source
    import boto3

    config_table = os.environ.get("CONFIG_TABLE", "fraud-config")
    region = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")

    try:
        dynamo = boto3.resource("dynamodb", region_name=region)
        response = dynamo.Table(config_table).get_item(Key={"config_key": "threshold"})
        item = response.get("Item")
        if item is not None:
            _active_threshold = float(item["value"])
            _threshold_source = "dynamodb"
            log.info("Threshold refreshed from DynamoDB: %.4f", _active_threshold)
    except Exception as exc:
        log.warning("refresh_threshold failed (non-fatal): %s", exc)

    return _active_threshold


def _download_from_s3(bucket: str, key: str, dest: str) -> None:
    import boto3

    region = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
    s3 = boto3.client("s3", region_name=region)
    log.info("Downloading s3://%s/%s → %s", bucket, key, dest)
    s3.download_file(bucket, key, dest)
    log.info("Download complete: %s", dest)


def _ensure_baseline() -> None:
    """Download the training baseline from S3 to /tmp/ if not available locally."""
    if os.environ.get("BASELINE_PATH"):
        log.info("BASELINE_PATH already set: %s", os.environ["BASELINE_PATH"])
        return

    if _LOCAL_BASELINE_PATH.exists():
        log.info("Baseline found at repo path: %s", _LOCAL_BASELINE_PATH)
        return

    if _TMP_BASELINE_PATH.exists():
        log.info("Baseline already in /tmp: %s", _TMP_BASELINE_PATH)
        os.environ["BASELINE_PATH"] = str(_TMP_BASELINE_PATH)
        return

    bucket = os.environ.get("MODEL_S3_BUCKET", _DEFAULT_S3_BUCKET)
    try:
        _download_from_s3(bucket, _S3_BASELINE_KEY, str(_TMP_BASELINE_PATH))
        os.environ["BASELINE_PATH"] = str(_TMP_BASELINE_PATH)
        log.info("BASELINE_PATH set to %s", _TMP_BASELINE_PATH)
    except Exception as exc:
        log.warning("Could not download baseline from S3 (non-fatal): %s", exc)


def _try_load_onnx() -> tuple[Any, str]:
    """Attempt to load model.onnx from local path or S3.

    Returns (session, input_name) on success, (None, '') on any failure.
    ONNX input name is inferred from the session at load time — never hardcoded.
    """
    onnx_path = Path(_DEFAULT_ONNX_PATH)

    if not onnx_path.exists():
        bucket = os.environ.get("MODEL_S3_BUCKET", _DEFAULT_S3_BUCKET)
        try:
            _download_from_s3(bucket, _DEFAULT_ONNX_S3_KEY, str(onnx_path))
        except Exception as exc:
            log.debug("ONNX model not available from S3: %s", exc)
            return None, ""

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        log.info("ONNX model loaded from %s  (input=%r)", onnx_path, input_name)
        return session, input_name
    except Exception as exc:
        log.warning("Failed to load ONNX model from %s: %s", onnx_path, exc)
        return None, ""


def load_model_bundle() -> ModelBundle:
    """Load the model bundle and optionally an ONNX session for inference.

    Resolution order for pkl (metadata + fallback inference):
      1. MODEL_PATH env var (if set and file exists)
      2. /tmp/xgboost_fraud_v1.pkl (prior invocation download)
      3. S3 download → /tmp/xgboost_fraud_v1.pkl

    Resolution order for ONNX (preferred inference path):
      1. /tmp/model.onnx (local or prior download)
      2. S3 download → /tmp/model.onnx
      3. Fall back to XGBoost pkl inference (warning logged)

    Raises:
        RuntimeError: If the pkl bundle cannot be loaded from any source.
    """
    _ensure_baseline()

    # ── 1. Load pkl bundle (metadata + XGBoost fallback) ──────────────────
    raw_path = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
    model_path = Path(raw_path)

    if not model_path.is_absolute():
        model_path = _REPO_ROOT / model_path

    if model_path.exists():
        log.info("Loading model bundle from local path: %s", model_path)
    else:
        bucket = os.environ.get("MODEL_S3_BUCKET", _DEFAULT_S3_BUCKET)
        key = os.environ.get("MODEL_S3_KEY", _DEFAULT_S3_KEY)
        dest = _DEFAULT_MODEL_PATH
        try:
            _download_from_s3(bucket, key, dest)
        except Exception as exc:
            raise RuntimeError(
                f"Model not found at {model_path} and S3 download failed: {exc}"
            ) from exc
        model_path = Path(dest)
        os.environ["MODEL_PATH"] = dest

    try:
        bundle: dict = _joblib_load(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to deserialize model bundle from {model_path}: {exc}"
        ) from exc

    missing = _REQUIRED_KEYS - set(bundle.keys())
    if missing:
        raise RuntimeError(
            f"Model bundle at {model_path} is missing required keys: {sorted(missing)}. "
            f"Found keys: {sorted(bundle.keys())}"
        )

    result = ModelBundle(
        model=bundle["model"],
        feature_names=bundle["feature_names"],
        threshold=bundle["threshold"],
        version=bundle["version"],
        amount_stats=bundle.get("amount_stats", {}),
        psi_baseline=bundle.get("psi_baseline", {}),
        experiment_manifest=bundle.get("experiment_manifest", {}),
        calibrator=bundle.get("calibrator"),
    )

    # ── 2. Try ONNX for faster inference ──────────────────────────────────
    onnx_session, onnx_input_name = _try_load_onnx()
    if onnx_session is not None:
        result._onnx_session = onnx_session
        result._onnx_input_name = onnx_input_name
        log.info("ONNX inference enabled")
    else:
        log.warning("ONNX model unavailable; falling back to XGBoost pkl for inference")

    log.info(
        "Model ready: version=%s  features=%d  threshold=%.4f  onnx=%s",
        result.version,
        len(result.feature_names),
        result.threshold,
        result._onnx_session is not None,
    )

    _load_threshold_from_dynamodb(result.threshold)

    return result


def inverse_transform_probability(probability: float) -> float:
    """Return the probability unchanged (identity; kept for API consistency)."""
    return probability
