import os

import joblib
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Fake AWS credentials so boto3 / moto don't attempt real calls
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

import boto3
from moto import mock_aws

import src.utils.model_loader as model_loader
from src.utils.model_loader import ModelBundle, load_model_bundle

_TEST_BUCKET = "test-model-bucket"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_bundle_dict():
    from xgboost import XGBClassifier

    return {
        "model": XGBClassifier(),
        "feature_names": ["f1", "f2", "f3"],
        "threshold": 0.5,
        "version": "test_v1",
        "amount_stats": {"mean": 100.0, "std": 50.0},
        "hyperparameters": {"n_estimators": 100},
    }


def _make_mock_onnx_session(n_outputs=2):
    """Return a MagicMock that resembles an onnxruntime InferenceSession."""
    mock_input = MagicMock()
    mock_input.name = "X"
    session = MagicMock()
    session.get_inputs.return_value = [mock_input]
    return session


# ---------------------------------------------------------------------------
# 1. Loads ONNX session when model.onnx is available (S3 download path)
# ---------------------------------------------------------------------------

@mock_aws
def test_loads_onnx_session_when_available(tmp_path, monkeypatch):
    # Create mock S3 bucket and upload fake onnx bytes
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=_TEST_BUCKET)
    s3.put_object(Bucket=_TEST_BUCKET, Key="models/model.onnx", Body=b"\x00" * 32)

    # Local pkl exists; onnx will be downloaded from mock S3
    fake_pkl = tmp_path / "model.pkl"
    joblib.dump(_fake_bundle_dict(), fake_pkl)
    onnx_dest = tmp_path / "model.onnx"  # does not exist yet

    mock_session = _make_mock_onnx_session()

    monkeypatch.setenv("MODEL_PATH", str(fake_pkl))
    monkeypatch.setenv("MODEL_S3_BUCKET", _TEST_BUCKET)
    monkeypatch.setattr(model_loader, "_DEFAULT_ONNX_PATH", str(onnx_dest))
    monkeypatch.setattr(model_loader, "_DEFAULT_S3_BUCKET", _TEST_BUCKET)
    monkeypatch.setattr(model_loader, "_ensure_baseline", lambda: None)

    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        bundle = load_model_bundle()

    assert bundle._onnx_session is mock_session
    assert bundle._onnx_input_name == "X"


# ---------------------------------------------------------------------------
# 2. Falls back to pkl when ONNX is not available
# ---------------------------------------------------------------------------

@mock_aws
def test_falls_back_to_pkl_when_onnx_missing(tmp_path, monkeypatch):
    # Bucket exists but model.onnx is NOT uploaded → download will fail
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=_TEST_BUCKET)

    fake_pkl = tmp_path / "model.pkl"
    joblib.dump(_fake_bundle_dict(), fake_pkl)
    onnx_dest = tmp_path / "model.onnx"  # does not exist, won't be created

    monkeypatch.setenv("MODEL_PATH", str(fake_pkl))
    monkeypatch.setenv("MODEL_S3_BUCKET", _TEST_BUCKET)
    monkeypatch.setattr(model_loader, "_DEFAULT_ONNX_PATH", str(onnx_dest))
    monkeypatch.setattr(model_loader, "_DEFAULT_S3_BUCKET", _TEST_BUCKET)
    monkeypatch.setattr(model_loader, "_ensure_baseline", lambda: None)

    bundle = load_model_bundle()

    assert bundle._onnx_session is None
    assert bundle.model is not None


# ---------------------------------------------------------------------------
# 3. predict() returns shape (n,) with values in [0, 1]
# ---------------------------------------------------------------------------

def test_predict_returns_correct_shape_and_range():
    n = 4
    bundle = ModelBundle(
        model=None,
        feature_names=["f1", "f2"],
        threshold=0.5,
        version="test",
    )
    # Wire up a mock ONNX session
    proba_matrix = np.array([[0.3, 0.7], [0.6, 0.4], [0.1, 0.9], [0.8, 0.2]])
    labels = np.array([1, 0, 1, 0])
    session = _make_mock_onnx_session()
    session.run.return_value = [labels, proba_matrix]

    bundle._onnx_session = session
    bundle._onnx_input_name = "X"

    X = np.random.randn(n, 2).astype(np.float32)
    result = bundle.predict(X)

    assert result.shape == (n,)
    assert (result >= 0.0).all() and (result <= 1.0).all()


# ---------------------------------------------------------------------------
# 4. predict() raises RuntimeError when neither ONNX nor pkl is available
# ---------------------------------------------------------------------------

def test_predict_raises_when_no_model_available():
    bundle = ModelBundle(
        model=None,
        feature_names=["f1"],
        threshold=0.5,
        version="test",
    )
    # _onnx_session defaults to None; model is also None
    with pytest.raises(RuntimeError, match="No model available"):
        bundle.predict(np.array([[1.0]]))


# ---------------------------------------------------------------------------
# 5. shap_background.pkl is never loaded by model_loader.py
# ---------------------------------------------------------------------------

@mock_aws
def test_shap_background_never_loaded(tmp_path, monkeypatch):
    fake_pkl = tmp_path / "model.pkl"
    joblib.dump(_fake_bundle_dict(), fake_pkl)
    onnx_dest = tmp_path / "model.onnx"  # does not exist

    monkeypatch.setenv("MODEL_PATH", str(fake_pkl))
    monkeypatch.setattr(model_loader, "_DEFAULT_ONNX_PATH", str(onnx_dest))
    monkeypatch.setattr(model_loader, "_ensure_baseline", lambda: None)

    with patch("joblib.load", wraps=joblib.load) as mock_load:
        load_model_bundle()

    for call in mock_load.call_args_list:
        path_arg = str(call.args[0]) if call.args else str(call.kwargs.get("filename", ""))
        assert "shap" not in path_arg.lower(), (
            f"joblib.load was called with a shap path: {path_arg}"
        )
