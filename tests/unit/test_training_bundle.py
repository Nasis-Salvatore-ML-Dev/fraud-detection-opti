"""Tests for train.py bundle enrichment: manifest, PSI, calibrator, model card."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

import scripts.train as train_module
from scripts.train import (
    _REQUIRED_BUNDLE_KEYS,
    _compute_dataset_hash,
    _fit_calibrator,
    main,
    parse_args,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    data["Amount"] = rng.uniform(1, 1000, n)
    data["Time"] = np.arange(n, dtype=float) * 3600
    data["Class"] = 0
    df = pd.DataFrame(data)
    # Fraud spread across both train and test splits (split at row 80)
    for idx in [20, 40, 60, 85, 90]:
        df.at[idx, "Class"] = 1
    return df


def _patch_paths(monkeypatch, tmp_path: Path, fake_csv: Path) -> None:
    monkeypatch.setattr(train_module, "DATA_PATH", fake_csv)
    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(train_module, "MODEL_OUT", tmp_path / "models" / "model.pkl")
    monkeypatch.setattr(train_module, "ONNX_OUT", tmp_path / "models" / "model.onnx")
    monkeypatch.setattr(train_module, "BASELINE_DIR", tmp_path / "baselines")
    monkeypatch.setattr(train_module, "SHAP_OUT", tmp_path / "baselines" / "shap_background.pkl")
    monkeypatch.setattr(train_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(train_module, "MODEL_CARD_PATH", tmp_path / "model_card.json")


def _run_main(monkeypatch, tmp_path: Path, fake_csv: Path) -> dict:
    _patch_paths(monkeypatch, tmp_path, fake_csv)
    with (
        patch.object(train_module, "_export_onnx", return_value=b"fake_onnx"),
        patch.object(train_module, "_upload_onnx_to_s3"),
        patch.object(train_module, "_upload_model_card_to_s3"),
    ):
        main(parse_args([]))
    return joblib.load(tmp_path / "models" / "model.pkl")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_csv(tmp_path: Path) -> Path:
    path = tmp_path / "creditcard.csv"
    _make_fake_df().to_csv(path, index=False)
    return path


@pytest.fixture
def bundle(tmp_path: Path, fake_csv: Path, monkeypatch) -> dict:
    return _run_main(monkeypatch, tmp_path, fake_csv)


# ---------------------------------------------------------------------------
# 1. Bundle contains all 9 required keys
# ---------------------------------------------------------------------------

def test_bundle_contains_all_required_keys(bundle):
    missing = _REQUIRED_BUNDLE_KEYS - set(bundle.keys())
    assert not missing, f"Bundle missing keys: {sorted(missing)}"


# ---------------------------------------------------------------------------
# 2. experiment_manifest contains git_sha, dataset_hash, metrics, trained_at
# ---------------------------------------------------------------------------

def test_experiment_manifest_fields(bundle):
    manifest = bundle["experiment_manifest"]
    assert "git_sha" in manifest
    assert "dataset_hash" in manifest
    assert "trained_at" in manifest
    metrics = manifest["metrics"]
    for key in ("auprc", "recall", "fpr", "threshold"):
        assert key in metrics, f"manifest.metrics missing key: {key}"


# ---------------------------------------------------------------------------
# 3. psi_baseline contains all feature columns + fraud_probability
# ---------------------------------------------------------------------------

def test_psi_baseline_covers_all_features(bundle):
    psi = bundle["psi_baseline"]
    assert "fraud_probability" in psi, "psi_baseline missing fraud_probability"
    feature_names = bundle["feature_names"]
    for feat in feature_names:
        assert feat in psi, f"psi_baseline missing feature: {feat}"


# ---------------------------------------------------------------------------
# 4. calibrator is a fitted IsotonicRegression
# ---------------------------------------------------------------------------

def test_calibrator_is_fitted_isotonic_regression(bundle):
    cal = bundle["calibrator"]
    assert isinstance(cal, IsotonicRegression)
    assert hasattr(cal, "X_thresholds_"), "calibrator does not appear to be fitted"


# ---------------------------------------------------------------------------
# 5. RuntimeError raised if a required bundle key is missing
# ---------------------------------------------------------------------------

def test_missing_bundle_key_raises(tmp_path: Path, fake_csv: Path, monkeypatch):
    monkeypatch.setattr(
        train_module,
        "_REQUIRED_BUNDLE_KEYS",
        _REQUIRED_BUNDLE_KEYS | {"nonexistent_key"},
    )
    with (
        patch.object(train_module, "_export_onnx", return_value=b"fake_onnx"),
        patch.object(train_module, "_upload_onnx_to_s3"),
        patch.object(train_module, "_upload_model_card_to_s3"),
        pytest.raises(RuntimeError, match="missing keys"),
    ):
        _run_main(monkeypatch, tmp_path, fake_csv)


# ---------------------------------------------------------------------------
# 6. dataset_hash changes when file content changes
# ---------------------------------------------------------------------------

def test_dataset_hash_changes_with_content(tmp_path: Path):
    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    f1.write_text("col1,col2\n1,2\n3,4\n")
    f2.write_text("col1,col2\n1,2\n3,5\n")  # one digit different

    h1 = _compute_dataset_hash(f1)
    h2 = _compute_dataset_hash(f2)
    assert h1 != h2
    assert len(h1) == 64  # SHA-256 hex digest length
