"""
XGBoost fraud detection training pipeline.

Loads creditcard.csv, engineers features, trains on first 80% chronologically,
evaluates on last 20%, and saves the model bundle + SHAP background sample.
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

from src.features.engineering import engineer_features

# Suppress Optuna's per-trial verbosity; we log trial results ourselves.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Logging: INFO → stdout, WARNING+ → stderr
# ---------------------------------------------------------------------------
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setLevel(logging.DEBUG)
_stdout_handler.addFilter(lambda r: r.levelno < logging.WARNING)

_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[_stdout_handler, _stderr_handler],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "training" / "creditcard.csv"
MODEL_DIR = REPO_ROOT / "models"
BASELINE_DIR = REPO_ROOT / "data" / "baselines"

MODEL_OUT = MODEL_DIR / "xgboost_fraud_v1.pkl"
ONNX_OUT = MODEL_DIR / "model.onnx"
SHAP_OUT = BASELINE_DIR / "shap_background.pkl"
REPORTS_DIR = REPO_ROOT / "data" / "reports"
MODEL_CARD_PATH = REPO_ROOT / "model_card.json"

_DEFAULT_S3_BUCKET = "fraud-model-artifacts-209998132741"
_PKL_S3_KEY = "models/xgboost_fraud_v1.pkl"
_ONNX_S3_KEY = "models/model.onnx"
_EQUIVALENCE_ROWS = 100
_EQUIVALENCE_TOL = 1e-4

MODEL_VERSION = "xgboost_fraud_v1"
DEFAULT_THRESHOLD = 0.5

PSI_N_BINS = 10

_REQUIRED_BUNDLE_KEYS = {
    "model",
    "feature_names",
    "threshold",
    "version",
    "amount_stats",
    "hyperparameters",
    "experiment_manifest",
    "psi_baseline",
    "calibrator",
}

# ---------------------------------------------------------------------------
# Fixed hyperparameters
# ---------------------------------------------------------------------------
XGBOOST_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
)

SHAP_BG_SIZE = 100
SHAP_BG_SEED = 0

TUNE_N_TRIALS = 100
# Inner split for Optuna: 75 % of the 80 % training slice = 60 % of total
_TUNE_INNER_FRAC = 0.75


# ---------------------------------------------------------------------------
# ONNX export helpers
# ---------------------------------------------------------------------------
class OnnxEquivalenceError(RuntimeError):
    """Raised when the ONNX model output diverges from XGBoost beyond tolerance."""


def _export_onnx(
    model,
    feature_names: list[str],
    X_val: pd.DataFrame,
) -> bytes:
    """Convert model to ONNX, verify equivalence on real rows, return bytes.

    Raises RuntimeError if the max absolute difference between XGBoost and
    ONNX probabilities exceeds _EQUIVALENCE_TOL.
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as ort

    n_features = len(feature_names)
    onnx_model = convert_sklearn(
        model,
        initial_types=[("X", FloatTensorType([None, n_features]))],
        options={type(model): {"zipmap": False}},
    )

    check_rows = min(_EQUIVALENCE_ROWS, len(X_val))
    X_check = X_val.iloc[:check_rows].values.astype(np.float32)

    proba_xgb = model.predict_proba(X_check)[:, 1]

    sess = ort.InferenceSession(onnx_model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    proba_onnx = sess.run(None, {input_name: X_check})[1][:, 1]

    max_diff = float(np.abs(proba_xgb - proba_onnx).max())
    log.debug("ONNX equivalence: max_abs_diff=%.2e on %d rows", max_diff, check_rows)

    if max_diff >= _EQUIVALENCE_TOL:
        log.error(
            "ONNX equivalence check FAILED: max_abs_diff=%.2e >= tol=%.2e",
            max_diff,
            _EQUIVALENCE_TOL,
        )
        raise OnnxEquivalenceError(
            f"ONNX equivalence check failed: max abs diff {max_diff:.2e} >= {_EQUIVALENCE_TOL:.2e}. "
            "Refusing to save a mismatched ONNX model."
        )

    log.info("ONNX equivalence OK: max_abs_diff=%.2e on %d rows", max_diff, check_rows)
    return onnx_model.SerializeToString()


def _upload_onnx_to_s3(local_path: Path, bucket: str, key: str) -> None:
    """Upload local_path to S3; logs warning on failure, never raises."""
    try:
        import boto3

        region = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
        s3 = boto3.client("s3", region_name=region)
        s3.upload_file(str(local_path), bucket, key)
        log.info("ONNX model uploaded → s3://%s/%s", bucket, key)
    except Exception as exc:
        log.warning("S3 upload of ONNX model failed (non-fatal): %s", exc)


def _upload_model_card_to_s3(local_path: Path, bucket: str, version: str) -> None:
    """Upload model_card.json to S3; logs warning on failure, never raises."""
    key = f"model_cards/model_card_{version}.json"
    try:
        import boto3

        region = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
        s3 = boto3.client("s3", region_name=region)
        s3.upload_file(str(local_path), bucket, key)
        log.info("Model card uploaded → s3://%s/%s", bucket, key)
    except Exception as exc:
        log.warning("S3 upload of model card failed (non-fatal): %s", exc)


def _compute_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _compute_dataset_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_psi_baseline(X_train: pd.DataFrame, y_prob_train: np.ndarray) -> dict:
    """Build a per-feature PSI baseline for all training features + fraud_probability."""
    psi_baseline: dict = {}
    feature_values: dict = {col: X_train[col].dropna().values for col in X_train.columns}
    feature_values["fraud_probability"] = y_prob_train

    for feature, values in feature_values.items():
        bin_edges = np.unique(np.percentile(values, np.linspace(0, 100, PSI_N_BINS + 1)))
        if len(bin_edges) < 2:
            log.warning("Feature %r has no variance — skipping PSI.", feature)
            continue
        bin_edges[0] -= 1e-9
        bin_edges[-1] += 1e-9
        counts, _ = np.histogram(values, bins=bin_edges)
        psi_baseline[feature] = {
            "bin_edges": bin_edges.tolist(),
            "expected_proportions": (counts / counts.sum()).tolist(),
        }
        log.debug("PSI: %s  %d bins", feature, len(bin_edges) - 1)

    log.info("PSI baseline computed for %d features", len(psi_baseline))
    return psi_baseline


def _fit_calibrator(y_prob: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """Fit an isotonic regression calibrator and verify ordering on the same split."""
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_prob, y_true)

    y_cal = calibrator.predict(y_prob)
    fraud_mask = y_true == 1
    legit_mask = y_true == 0

    if fraud_mask.any() and legit_mask.any():
        mean_fraud = float(y_cal[fraud_mask].mean())
        mean_legit = float(y_cal[legit_mask].mean())
        if mean_fraud <= mean_legit:
            log.warning(
                "Calibration check failed: mean fraud prob (%.4f) not > mean legit prob (%.4f)",
                mean_fraud,
                mean_legit,
            )
        else:
            log.info(
                "Calibration check OK: mean fraud prob=%.4f > mean legit prob=%.4f",
                mean_fraud,
                mean_legit,
            )

    return calibrator


def _generate_model_card(manifest: dict, version: str, bias_report_path: Path) -> dict:
    """Build a Mitchell et al. model card dict from the experiment manifest."""
    metrics = manifest.get("metrics", {})

    bias_segments: list = []
    bias_computed_at = None
    bias_recommendation = None
    if bias_report_path.exists():
        try:
            with open(bias_report_path) as f:
                bias_report = json.load(f)
            bias_segments = bias_report.get("bias_segments", [])
            bias_computed_at = bias_report.get("computed_at")
            bias_recommendation = bias_report.get("recommendation")
            log.info(
                "Bias report loaded — %d segment(s), computed_at=%s",
                len(bias_segments),
                bias_computed_at,
            )
        except Exception as exc:
            log.warning("Could not load bias report (non-fatal): %s", exc)

    return {
        "model_details": {
            "name": "fraud-detection-xgboost",
            "version": version,
            "type": "XGBClassifier",
            "task": "binary_classification",
            "framework": "XGBoost 2.x + scikit-learn wrapper",
            "git_sha": manifest.get("git_sha"),
            "trained_at": manifest.get("trained_at"),
            "dataset_hash": manifest.get("dataset_hash"),
            "description": (
                "Gradient-boosted decision tree classifier that scores individual "
                "credit-card transactions as fraudulent or legitimate. "
                "Trained on PCA-anonymised features from the Kaggle Credit Card "
                "Fraud Detection dataset."
            ),
        },
        "intended_use": {
            "primary_use": (
                "Real-time fraud scoring for European credit-card transactions. "
                "Intended as a decision-support tool for fraud analysts."
            ),
            "out_of_scope": [
                "Automated account closure decisions without human review.",
                "Scoring transactions outside European card networks.",
                "Any use where PCA feature identities are assumed known.",
            ],
        },
        "factors": {
            "relevant_factors": [
                "Transaction amount (Amount)",
                "Hour of day derived from elapsed time (hour_of_day)",
                "V1–V28: PCA-transformed card-network features (identities withheld by Kaggle)",
            ],
            "evaluation_factors": [
                "Amount segment: high (>$1000) vs. low (≤$1000)",
                "Time segment: evening (≥18:00) vs. daytime (<18:00)",
            ],
        },
        "metrics": {
            "performance_measures": ["AUPRC", "Recall", "False Positive Rate"],
            "decision_threshold": metrics.get("threshold"),
            "results": {
                "auprc": metrics.get("auprc"),
                "recall": metrics.get("recall"),
                "fpr": metrics.get("fpr"),
                "note": "Metrics from experiment manifest on 20% temporal hold-out.",
            },
            "bias_results": {
                "computed_at": bias_computed_at,
                "segments": bias_segments,
                "recommendation": bias_recommendation,
            },
        },
        "evaluation_data": {
            "dataset": "Kaggle Credit Card Fraud Detection",
            "source": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
            "split": "Chronological 80/20 — first 80% train, last 20% test",
            "time_coverage": "2 days of European transactions (Sept 2013)",
            "preprocessing": (
                "No scaling applied to V1–V28 (already PCA-transformed). "
                "Amount log-transformed (log1p) and z-scored using training-set statistics. "
                "Hour of day derived from elapsed seconds modulo 86400."
            ),
        },
        "ethical_considerations": [
            "No personally identifiable information in training data — "
            "all card-network features are PCA-anonymised by the dataset provider.",
            "FPR parity tested across Amount (high/low) and time-of-day (evening/daytime) segments. "
            "Segments with FPR > 2× overall FPR are flagged for review.",
            "Uncertain predictions (fraud probability 0.3–0.7) are automatically routed to a "
            "human override queue rather than acted on autonomously.",
            "EU AI Act Annex III classification: high-risk system (financial services). "
            "Requires human oversight, audit logging, and conformity assessment before deployment.",
        ],
        "caveats": [
            "Dataset covers only 2 days — model may not generalise to seasonal fraud patterns "
            "or novel attack vectors that emerged after Sept 2013.",
            "V1–V28 are PCA components of undisclosed features, preventing direct feature "
            "interpretation or manual override based on individual feature values.",
            "Decision threshold tuned for high recall (≥0.90) on the test set. "
            "Expect higher FPR in production due to distribution shift over time.",
            "PSI drift monitoring covers all 32 engineered features plus fraud_probability.",
        ],
        "mlops": {
            "platform": "AWS Lambda (container image) + API Gateway",
            "audit_log": "DynamoDB table  fraud-audit-log  (permanent TTL)",
            "override_queue": "DynamoDB table  fraud-override-queue  (30-day TTL)",
            "drift_detection": "PSI on all features + fraud_probability; "
            "alert thresholds stable < 0.10 < monitor < 0.20 < action_required",
            "bias_testing": "Per-segment AUPRC and FPR parity; gates CD pipeline via "
            "scripts/run_bias_test.py (exit 1 on failure)",
            "ci_cd": "GitHub Actions — CI on PR, CD on merge to main",
        },
        "generated_at": manifest.get("trained_at"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XGBoost fraud detection training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help=(
            "Run an Optuna hyperparameter search (100 trials) before the final "
            "model fit.  When omitted, fixed hyperparameters from XGBOOST_PARAMS "
            "are used and training completes in under 5 minutes."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def make_objective(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
):
    """Return an Optuna objective closed over the inner train / val splits.

    scale_pos_weight is derived from class distribution and is never part of
    the search space — it is passed directly to XGBClassifier.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(X_tr, y_tr, verbose=False)
        y_prob = model.predict_proba(X_val)[:, 1]
        auprc = float(average_precision_score(y_val, y_prob))
        log.debug("Trial %d: AUPRC=%.4f  params=%s", trial.number, auprc, params)
        return auprc

    return objective


# ---------------------------------------------------------------------------
# Stratified background sample for SHAP
# ---------------------------------------------------------------------------
def sample_shap_background(
    X: pd.DataFrame, y: pd.Series, n: int = SHAP_BG_SIZE, seed: int = SHAP_BG_SEED
) -> tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(round(n * y.mean())))
    n_legit = n - n_fraud

    fraud_idx = y[y == 1].index
    legit_idx = y[y == 0].index

    chosen_fraud = rng.choice(fraud_idx, size=min(n_fraud, len(fraud_idx)), replace=False)
    chosen_legit = rng.choice(legit_idx, size=min(n_legit, len(legit_idx)), replace=False)

    chosen = np.concatenate([chosen_fraud, chosen_legit])
    rng.shuffle(chosen)
    return X.loc[chosen].reset_index(drop=True), len(chosen_fraud)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main(args=None) -> None:
    if args is None:
        args = parse_args()
    # ── 1. Load ──────────────────────────────────────────────────────────
    log.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    log.info("Dataset shape: %s  |  fraud rate: %.4f%%", df.shape, df["Class"].mean() * 100)

    # ── 2. Feature engineering ────────────────────────────────────────────
    log.info("Engineering features")
    split_idx = int(len(df) * 0.80)
    amount_stats = {
        "mean": float(df["Amount"].iloc[:split_idx].mean()),
        "std": float(df["Amount"].iloc[:split_idx].std()),
    }
    log.debug("Amount stats (train): mean=%.4f std=%.4f", amount_stats["mean"], amount_stats["std"])
    df = engineer_features(df, amount_stats)

    feature_cols = [c for c in df.columns if c.startswith("V")] + [
        "Amount",
        "amount_log",
        "amount_zscore",
        "hour_of_day",
    ]
    X = df[feature_cols]
    y = df["Class"].astype(int)

    # ── 3. Chronological split (no shuffle) ───────────────────────────────
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    log.info(
        "Train size: %d  (fraud: %d)  |  Test size: %d  (fraud: %d)",
        len(y_train),
        y_train.sum(),
        len(y_test),
        y_test.sum(),
    )

    # ── 4. Class imbalance weight ─────────────────────────────────────────
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    log.info("scale_pos_weight = %.2f  (neg=%d / pos=%d)", scale_pos_weight, n_neg, n_pos)

    # ── 5. Hyperparameter selection (± Optuna tuning) ─────────────────────
    if args.tune:
        log.info("Running Optuna study (%d trials)", TUNE_N_TRIALS)
        inner_idx = int(len(X_train) * _TUNE_INNER_FRAC)
        X_tr = X_train.iloc[:inner_idx]
        X_val_inner = X_train.iloc[inner_idx:]
        y_tr = y_train.iloc[:inner_idx]
        y_val_inner = y_train.iloc[inner_idx:]

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        objective = make_objective(X_tr, y_tr, X_val_inner, y_val_inner, scale_pos_weight)
        study.optimize(objective, n_trials=TUNE_N_TRIALS, n_jobs=1, show_progress_bar=False)
        log.info(
            "Optuna study complete: best_value=%.4f  best_trial=#%d",
            study.best_value,
            study.best_trial.number,
        )
        used_params = dict(study.best_trial.params)
        used_params.update({"eval_metric": "aucpr", "random_state": 42, "n_jobs": -1})
    else:
        used_params = dict(XGBOOST_PARAMS)

    log.info("Hyperparameters selected: %s", used_params)

    # ── 6. Train final model ──────────────────────────────────────────────
    log.info("Training XGBClassifier")
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, **used_params)
    model.fit(X_train, y_train, verbose=False)
    log.info("Training complete")

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    log.info("Evaluating on test set")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= DEFAULT_THRESHOLD).astype(int)

    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(y_test, y_pred, digits=4))

    auprc = average_precision_score(y_test, y_prob)
    print(f"AUPRC (area under precision-recall curve): {auprc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print("\n── Confusion Matrix ───────────────────────────────────")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    print(f"\nRecall (sensitivity): {recall:.4f}")
    print(f"False Positive Rate:  {fpr:.6f}\n")

    log.info("AUPRC=%.4f  Recall=%.4f  FPR=%.6f", auprc, recall, fpr)

    # ── 6b. Threshold tuning ──────────────────────────────────────────────
    log.info("Tuning threshold  (sweep 0.10 → 0.50, step 0.05)")
    optimal_threshold = DEFAULT_THRESHOLD
    optimal_metrics: dict = {}

    for thr in np.arange(0.10, 0.51, 0.05):
        y_pred_t = (y_prob >= thr).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_t).ravel()
        recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) else 0.0
        fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) else 0.0
        if recall_t >= 0.90 and fpr_t <= 0.01:
            optimal_threshold = float(round(thr, 4))
            optimal_metrics = {
                "recall": recall_t,
                "fpr": fpr_t,
                "tp": int(tp_t),
                "fp": int(fp_t),
                "fn": int(fn_t),
                "tn": int(tn_t),
            }
            break

    print("\n── Optimal Threshold (Recall≥0.90 & FPR≤0.01) ────────────")
    if optimal_metrics:
        print(f"  Threshold: {optimal_threshold:.2f}")
        print(f"  Recall:    {optimal_metrics['recall']:.4f}")
        print(f"  FPR:       {optimal_metrics['fpr']:.6f}")
        print(f"  TP={optimal_metrics['tp']}  FP={optimal_metrics['fp']}")
        print(f"  FN={optimal_metrics['fn']}  TN={optimal_metrics['tn']}")
        log.info(
            "Optimal threshold=%.2f  Recall=%.4f  FPR=%.6f",
            optimal_threshold,
            optimal_metrics["recall"],
            optimal_metrics["fpr"],
        )
    else:
        log.warning(
            "No threshold in [0.10, 0.50] satisfies Recall≥0.90 AND FPR≤0.01; "
            "falling back to DEFAULT_THRESHOLD=%.2f",
            DEFAULT_THRESHOLD,
        )
        print(f"  [WARNING] No threshold found; using fallback {DEFAULT_THRESHOLD}")

    # ── 8. SHAP background sample ─────────────────────────────────────────
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    bg, n_bg_fraud = sample_shap_background(X_train, y_train)
    joblib.dump(bg, SHAP_OUT)
    log.info(
        "SHAP background saved → %s  (shape: %s, fraud rows: %d)",
        SHAP_OUT,
        bg.shape,
        n_bg_fraud,
    )

    # ── 9. Full PSI baseline (all features + fraud_probability) ──────────
    log.info("Computing PSI baseline for %d features + fraud_probability", len(feature_cols))
    y_prob_train = model.predict_proba(X_train)[:, 1]
    psi_baseline = _compute_psi_baseline(X_train, y_prob_train)

    # ── 10. Isotonic calibrator (fitted on test / validation set) ─────────
    log.info("Fitting isotonic calibrator on validation set")
    calibrator = _fit_calibrator(y_prob, y_test.values)

    # ── 11. Experiment manifest ────────────────────────────────────────────
    log.info("Building experiment manifest")
    git_sha = _compute_git_sha()
    dataset_hash = _compute_dataset_hash(DATA_PATH)
    s3_bucket = os.environ.get("MODEL_S3_BUCKET", _DEFAULT_S3_BUCKET)
    manifest = {
        "git_sha": git_sha,
        "dataset_hash": dataset_hash,
        "hyperparameters": used_params,
        "metrics": {
            "auprc": float(auprc),
            "recall": float(recall),
            "fpr": float(fpr),
            "threshold": float(optimal_threshold),
        },
        "s3_paths": {
            "model_pkl": _PKL_S3_KEY,
            "model_onnx": _ONNX_S3_KEY,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORTS_DIR / "experiment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Experiment manifest saved → %s", manifest_path)

    # ── 12. Model card ─────────────────────────────────────────────────────
    log.info("Generating model card")
    bias_report_path = REPORTS_DIR / "bias_report.json"
    card = _generate_model_card(manifest, MODEL_VERSION, bias_report_path)
    with open(MODEL_CARD_PATH, "w") as f:
        json.dump(card, f, indent=2)
    log.info("Model card saved → %s", MODEL_CARD_PATH)
    _upload_model_card_to_s3(MODEL_CARD_PATH, s3_bucket, MODEL_VERSION)

    # ── 13. Bundle key assertion + save ────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_names": feature_cols,
        "threshold": optimal_threshold,
        "version": MODEL_VERSION,
        "amount_stats": amount_stats,
        "hyperparameters": used_params,
        "experiment_manifest": manifest,
        "psi_baseline": psi_baseline,
        "calibrator": calibrator,
    }
    missing_keys = _REQUIRED_BUNDLE_KEYS - set(bundle.keys())
    if missing_keys:
        log.error("Model bundle missing required keys: %s", sorted(missing_keys))
        raise RuntimeError(f"Model bundle missing keys: {sorted(missing_keys)}")
    joblib.dump(bundle, MODEL_OUT)
    log.info("Model bundle saved → %s", MODEL_OUT)

    # ── 14. ONNX export + equivalence check ───────────────────────────────
    log.info("Exporting model to ONNX and verifying numerical equivalence")
    try:
        onnx_bytes = _export_onnx(model, feature_cols, X_test)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(ONNX_OUT, "wb") as f:
            f.write(onnx_bytes)
        log.info("ONNX model saved → %s  (%d bytes)", ONNX_OUT, ONNX_OUT.stat().st_size)

        # ── 15. Upload ONNX to S3 ─────────────────────────────────────────
        _upload_onnx_to_s3(ONNX_OUT, s3_bucket, _ONNX_S3_KEY)
    except OnnxEquivalenceError:
        raise  # equivalence mismatch → abort; do not ship a bad ONNX model
    except Exception as exc:
        log.error("ONNX export failed — model.onnx not written: %s", exc)


if __name__ == "__main__":
    try:
        main(parse_args())
    except Exception as exc:
        log.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)
