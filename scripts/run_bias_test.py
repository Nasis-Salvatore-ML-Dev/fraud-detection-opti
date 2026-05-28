"""
CD pipeline bias gate — exits 1 if any segment fails AUPRC, FPR, or directional-bias check.

Run:
    PYTHONPATH=. python scripts/run_bias_test.py

Exit codes:
    0 — all segments pass (or no validation data available)
    1 — one or more segments flagged
"""

import logging
import sys
from pathlib import Path

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
VALIDATION_PATH = REPO_ROOT / "data" / "validation" / "validation_set.csv"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ── 1. Check for validation data ──────────────────────────────────────
    if not VALIDATION_PATH.exists():
        log.warning(
            "No validation set found at %s — skipping bias gate (exit 0).\n"
            "  Generate it with:  python scripts/train.py  (or supply externally).",
            VALIDATION_PATH,
        )
        sys.exit(0)

    # ── 2. Load validation data ───────────────────────────────────────────
    import pandas as pd

    log.info("Loading validation set from %s", VALIDATION_PATH)
    try:
        validation_df = pd.read_csv(VALIDATION_PATH)
    except Exception as exc:
        log.error("Failed to load validation set: %s", exc)
        sys.exit(1)

    log.info(
        "Validation set loaded  shape=%s  fraud_rate=%.4f%%",
        validation_df.shape,
        validation_df["Class"].mean() * 100,
    )

    # ── 3. Load model bundle ──────────────────────────────────────────────
    from src.utils.model_loader import load_model_bundle

    log.info("Loading model bundle")
    try:
        load_model_bundle()
    except Exception as exc:
        log.error("Failed to load model bundle: %s", exc)
        sys.exit(1)

    # ── 4. Run bias test ──────────────────────────────────────────────────
    from src.monitoring.bias_tester import BiasTestSuite

    log.info("Running bias test suite")
    suite = BiasTestSuite()
    report = suite.run(validation_df=validation_df)

    overall_auprc = report["overall_auprc"]
    overall_fpr = report["overall_fpr"]
    summary = report.get("summary", {})

    # ── 5. Print segment table ────────────────────────────────────────────
    print()
    print("── Bias Test Results ─────────────────────────────────────────────────────────")
    print(
        f"  overall_auprc={overall_auprc:.4f}   overall_fpr={overall_fpr:.6f}"
    )
    print()

    col_w = [16, 8, 10, 13, 11, 15]
    header = (
        f"  {'segment':<{col_w[0]}}  {'AUPRC':>{col_w[1]}}  "
        f"{'FPR':>{col_w[2]}}  {'auprc_flagged':>{col_w[3]}}  "
        f"{'fpr_flagged':>{col_w[4]}}  {'dir_bias':>{col_w[5]}}"
    )
    print(header)
    print("  " + "-" * (sum(col_w) + 10))

    any_flagged = False
    for seg in report["bias_segments"]:
        auprc_val = seg.get("auprc")
        fpr_val = seg.get("fpr")

        auprc_str = f"{auprc_val:.4f}" if auprc_val is not None else "N/A"
        fpr_str = f"{fpr_val:.6f}" if fpr_val is not None else "N/A"
        auprc_flagged = seg.get("auprc_flagged", False)
        fpr_flagged = seg.get("fpr_flagged", False)
        dir_bias = seg.get("directional_bias_flagged", False)

        row_flagged = seg.get("flagged", False)
        if row_flagged:
            any_flagged = True

        marker = "  * " if row_flagged else "    "
        print(
            f"{marker}{seg['segment']:<{col_w[0]}}  "
            f"{auprc_str:>{col_w[1]}}  {fpr_str:>{col_w[2]}}  "
            f"{str(auprc_flagged):>{col_w[3]}}  {str(fpr_flagged):>{col_w[4]}}  "
            f"{str(dir_bias):>{col_w[5]}}"
        )

    print()
    print("  (* = flagged segment)")
    if summary.get("directional_bias_detected"):
        print("  (!) Directional bias detected in one or more segments.")

    # ── 6. Gate ───────────────────────────────────────────────────────────
    print()
    if any_flagged:
        print("── BIAS GATE: FAIL ───────────────────────────────────────────────────────────")
        print(f"  {report['recommendation']}")
        print()
        log.warning("Bias gate FAILED — one or more segments flagged.")
        sys.exit(1)
    else:
        print("── BIAS GATE: PASS ───────────────────────────────────────────────────────────")
        print(f"  {report['recommendation']}")
        print()
        log.info("Bias gate PASSED — no segments flagged.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        log.error("run_bias_test failed: %s", exc, exc_info=True)
        sys.exit(1)
