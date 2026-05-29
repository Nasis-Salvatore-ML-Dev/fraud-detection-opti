"""CLI entry point for the retraining orchestrator.

Usage:
    python scripts/run_retraining_check.py [--dry-run]

Exit codes:
    0 — always (retraining trigger is not an error condition)
"""

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stderr,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate drift signals and optionally trigger retraining."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate signals but do not dispatch GitHub Actions.",
    )
    args = parser.parse_args()

    from src.monitoring.retraining_trigger import check_and_trigger

    result = check_and_trigger(dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
