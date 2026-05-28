import numpy as np
import optuna
import pandas as pd
import pytest
import joblib
from unittest.mock import patch
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_splits(n_tr=50, n_val=20, n_fraud=3):
    rng = np.random.default_rng(0)
    X_tr = pd.DataFrame(rng.standard_normal((n_tr, 5)), columns=list("ABCDE"))
    y_tr = pd.Series([0] * (n_tr - n_fraud) + [1] * n_fraud)
    X_val = pd.DataFrame(rng.standard_normal((n_val, 5)), columns=list("ABCDE"))
    y_val = pd.Series([0] * (n_val - n_fraud) + [1] * n_fraud)
    return X_tr, y_tr, X_val, y_val


def _make_fake_df(n=100):
    rng = np.random.default_rng(42)
    data = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    data["Amount"] = rng.uniform(1, 500, n)
    data["Time"] = np.arange(n, dtype=float) * 100.0
    data["Class"] = np.zeros(n, dtype=int)
    # Spread fraud across both train (rows 0–79) and test (rows 80–99)
    # so that y_train has positives and scale_pos_weight is well-defined.
    data["Class"][20] = 1
    data["Class"][40] = 1
    data["Class"][60] = 1
    data["Class"][85] = 1
    data["Class"][90] = 1
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 1. Argument parser accepts --tune flag
# ---------------------------------------------------------------------------

def test_parse_args_tune_flag_true():
    from scripts.train import parse_args
    args = parse_args(["--tune"])
    assert args.tune is True


def test_parse_args_tune_flag_defaults_false():
    from scripts.train import parse_args
    args = parse_args([])
    assert args.tune is False


# ---------------------------------------------------------------------------
# 2. Optuna objective returns a float in [0, 1]
# ---------------------------------------------------------------------------

def test_objective_returns_float_in_range():
    from scripts.train import make_objective

    X_tr, y_tr, X_val, y_val = _make_tiny_splits()
    n_val = len(X_val)
    mock_proba = np.column_stack([np.ones(n_val) * 0.6, np.ones(n_val) * 0.4])

    with patch.object(XGBClassifier, "fit"), \
         patch.object(XGBClassifier, "predict_proba", return_value=mock_proba):
        objective = make_objective(X_tr, y_tr, X_val, y_val, scale_pos_weight=15.0)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1, show_progress_bar=False)

    result = study.best_value
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# 3. Model bundle contains "hyperparameters" after a mock training run
# ---------------------------------------------------------------------------

def test_bundle_contains_hyperparameters(tmp_path):
    import scripts.train as train_mod

    fake_df = _make_fake_df(100)
    n_test = 100 - int(100 * 0.8)  # 20 rows in test split
    mock_proba = np.array([[0.8, 0.2]] * n_test)

    model_out = tmp_path / "model.pkl"
    shap_out = tmp_path / "shap.pkl"

    with patch.object(train_mod, "MODEL_OUT", model_out), \
         patch.object(train_mod, "SHAP_OUT", shap_out), \
         patch.object(train_mod, "MODEL_DIR", tmp_path), \
         patch.object(train_mod, "BASELINE_DIR", tmp_path), \
         patch.object(train_mod, "REPORTS_DIR", tmp_path / "reports"), \
         patch.object(train_mod, "MODEL_CARD_PATH", tmp_path / "model_card.json"), \
         patch.object(train_mod, "_compute_dataset_hash", return_value="fakehash"), \
         patch.object(train_mod, "_upload_model_card_to_s3"), \
         patch.object(train_mod, "_upload_onnx_to_s3"), \
         patch("pandas.read_csv", return_value=fake_df), \
         patch.object(XGBClassifier, "fit"), \
         patch.object(XGBClassifier, "predict_proba", return_value=mock_proba):
        args = train_mod.parse_args([])  # no --tune: uses fixed params
        train_mod.main(args)

    bundle = joblib.load(model_out)
    assert "hyperparameters" in bundle


# ---------------------------------------------------------------------------
# 4. scale_pos_weight is not in the Optuna search space
# ---------------------------------------------------------------------------

def test_scale_pos_weight_not_in_search_space():
    from scripts.train import make_objective

    X_tr, y_tr, X_val, y_val = _make_tiny_splits()
    n_val = len(X_val)
    mock_proba = np.column_stack([np.ones(n_val) * 0.6, np.ones(n_val) * 0.4])

    with patch.object(XGBClassifier, "fit"), \
         patch.object(XGBClassifier, "predict_proba", return_value=mock_proba):
        objective = make_objective(X_tr, y_tr, X_val, y_val, scale_pos_weight=15.0)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1, show_progress_bar=False)

    assert "scale_pos_weight" not in study.best_trial.params
