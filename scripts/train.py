"""
XGBoost fraud detection training pipeline.

Loads creditcard.csv, engineers features, trains on first 80% chronologically,
evaluates on last 20%, and saves the model bundle + SHAP background sample.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
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
SHAP_OUT = BASELINE_DIR / "shap_background.pkl"

MODEL_VERSION = "xgboost_fraud_v1"
DEFAULT_THRESHOLD = 0.5

PSI_FEATURES = ["Amount", "amount_log", "amount_zscore", "hour_of_day"]
PSI_N_BINS = 10
BASELINE_OUT = BASELINE_DIR / "training_baseline.json"

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

    # ── 7. Save model bundle ──────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_names": feature_cols,
        "threshold": optimal_threshold,
        "version": MODEL_VERSION,
        "amount_stats": amount_stats,
        "hyperparameters": used_params,
    }
    joblib.dump(bundle, MODEL_OUT)
    log.info("Model bundle saved → %s", MODEL_OUT)

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

    # ── 9. PSI baseline ───────────────────────────────────────────────────
    log.info("Computing PSI baseline for features: %s", PSI_FEATURES)
    psi_baseline: dict = {}
    for feature in PSI_FEATURES:
        values = X_train[feature].dropna().values
        bin_edges = np.unique(np.percentile(values, np.linspace(0, 100, PSI_N_BINS + 1)))
        if len(bin_edges) < 2:
            log.warning("Feature %r has no variance — skipping.", feature)
            continue
        bin_edges[0] = bin_edges[0] - 1e-9
        bin_edges[-1] = bin_edges[-1] + 1e-9
        counts, _ = np.histogram(values, bins=bin_edges)
        proportions = (counts / counts.sum()).tolist()
        psi_baseline[feature] = {
            "bin_edges": bin_edges.tolist(),
            "expected_proportions": proportions,
            "n_samples": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p5": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }
        log.info(
            "  %s: %d bins, mean=%.4f, std=%.4f",
            feature,
            len(bin_edges) - 1,
            psi_baseline[feature]["mean"],
            psi_baseline[feature]["std"],
        )

    with open(BASELINE_OUT, "w") as f:
        json.dump(psi_baseline, f, indent=2)
    log.info("PSI baseline saved → %s", BASELINE_OUT)


if __name__ == "__main__":
    try:
        main(parse_args())
    except Exception as exc:
        log.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)
