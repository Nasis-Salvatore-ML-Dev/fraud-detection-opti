"""CLI: compute SHAP values offline and store them in DynamoDB.

Usage:
    python scripts/compute_shap.py \
        --data data/training/creditcard.csv \
        --bundle models/bundle.pkl \
        --table fraud-shap-store \
        --sample 5000

shap is a dev dependency imported here only; it never appears under src/.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import shap  # noqa: F401 — dev dependency; only imported in scripts/, never src/

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src.explainability.shap_offline import compute_and_store_shap  # noqa: E402
from src.features.engineering import engineer_features  # noqa: E402

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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SHAP values offline and store in DynamoDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, type=Path, help="Path to training CSV")
    parser.add_argument("--bundle", required=True, type=Path, help="Path to model bundle pkl")
    parser.add_argument("--table", default="fraud-shap-store", help="DynamoDB table name")
    parser.add_argument("--sample", type=int, default=5000, help="Number of rows to sample")
    return parser.parse_args(argv)


def main(args=None) -> None:
    if args is None:
        args = parse_args()

    t0 = time.perf_counter()

    log.info("Loading model bundle from %s", args.bundle)
    bundle: dict = joblib.load(args.bundle)
    model = bundle["model"]
    feature_names: list[str] = bundle["feature_names"]
    amount_stats: dict = bundle.get("amount_stats", {})

    log.info("Loading data from %s", args.data)
    df = pd.read_csv(args.data)
    df = engineer_features(df, amount_stats)
    X = df[feature_names]

    if args.sample < len(X):
        X = X.sample(n=args.sample, random_state=42)
        log.info("Sampled %d rows from %d total", args.sample, len(df))

    background = X.iloc[:100]
    log.info(
        "Computing SHAP values for %d rows  background=%d rows  table=%s",
        len(X),
        len(background),
        args.table,
    )

    total = compute_and_store_shap(model, X, background, args.table)

    elapsed = time.perf_counter() - t0
    log.info("Done: total_stored=%d  elapsed=%.1fs", total, elapsed)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("compute_shap failed: %s", exc, exc_info=True)
        sys.exit(1)
