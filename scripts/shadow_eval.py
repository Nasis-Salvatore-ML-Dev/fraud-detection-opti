"""Compare champion (stored audit scores) vs challenger (staging) AUPRC.

Usage:
    python scripts/shadow_eval.py --endpoint URL [--limit N] [--output PATH]

Exit codes:
    0 — challenger AUPRC >= 0.95 × champion AUPRC (or < 100 labelled records — skip)
    1 — challenger AUPRC < 0.95 × champion AUPRC
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import boto3
import httpx
import msgpack
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

log = logging.getLogger(__name__)

_AUDIT_TABLE = os.environ.get("AUDIT_TABLE", "fraud-audit-log")
_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
_MIN_LABELLED = 100
_CHALLENGER_RATIO = 0.95


def _from_decimal(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _from_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_decimal(v) for v in obj]
    return obj


def _scan_audit(limit: int) -> list[dict]:
    """Scan fraud-audit-log and return up to limit raw DynamoDB items."""
    dynamodb = boto3.resource("dynamodb", region_name=_REGION)
    table = dynamodb.Table(_AUDIT_TABLE)
    try:
        response = table.scan(Limit=limit)
        return response.get("Items", [])
    except Exception as exc:
        log.error("DynamoDB scan failed: %s", exc)
        return []


def _unpack_record(item: dict) -> dict | None:
    """Decode a DynamoDB item into payload + champion_score + is_fraud.

    Returns None if the item is missing required fields.
    """
    try:
        record = _from_decimal(item)

        raw_bytes = item.get("v_features_msgpack")
        if raw_bytes is None:
            return None
        v_features: dict = msgpack.unpackb(bytes(raw_bytes), raw=False)

        # V1-V28 stored uppercase; /predict expects lowercase keys
        payload_v = {
            f"v{k[1:]}": float(v)
            for k, v in v_features.items()
            if k.startswith("V") and k[1:].isdigit()
        }
        if len(payload_v) != 28:
            return None

        amount = float(record.get("Amount", 0.0))
        hour_of_day = float(record.get("hour_of_day", 0.0))
        # Reconstruct time so (time % 86400) // 3600 == stored hour_of_day
        time_approx = hour_of_day * 3600.0

        champion_score = record.get("fraud_probability")
        is_fraud = record.get("is_fraud")
        if champion_score is None or is_fraud is None:
            return None

        return {
            "payload": {"time": time_approx, "amount": amount, **payload_v},
            "champion_score": float(champion_score),
            "is_fraud": bool(is_fraud),
        }
    except Exception as exc:
        log.warning("Failed to unpack audit record: %s", exc)
        return None


def _send_predict(client: httpx.Client, endpoint: str, payload: dict) -> float | None:
    """POST to /predict and return fraud_probability, or None on error."""
    try:
        resp = client.post(f"{endpoint}/predict", json=payload)
        resp.raise_for_status()
        return float(resp.json()["fraud_probability"])
    except Exception as exc:
        log.warning("Challenger /predict failed: %s", exc)
        return None


def _compute_metrics(y_true: list, scores: list[float]) -> dict:
    """Compute AUPRC, recall, and FPR at a 0.5 decision threshold."""
    y_arr = np.array(y_true, dtype=int)
    s_arr = np.array(scores, dtype=float)

    auprc = float(average_precision_score(y_arr, s_arr))

    preds = (s_arr >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, preds, labels=[0, 1]).ravel()
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {"auprc": round(auprc, 6), "recall": round(recall, 6), "fpr": round(fpr, 6)}


def _write_json(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    log.info("Shadow eval report written to %s", path)


def run_shadow_eval(endpoint: str, limit: int, output_path: str) -> dict:
    """Run shadow evaluation. Returns a result dict with 'status' key."""
    log.info("Scanning audit table (limit=%d)", limit)
    raw_items = _scan_audit(limit)
    log.info("Fetched %d audit records", len(raw_items))

    records = [r for item in raw_items if (r := _unpack_record(item)) is not None]
    log.info("Decoded %d valid records", len(records))

    if len(records) < _MIN_LABELLED:
        log.info(
            "Only %d labelled records (< %d) — skipping shadow eval",
            len(records), _MIN_LABELLED,
        )
        result = {
            "status": "skipped",
            "reason": f"fewer than {_MIN_LABELLED} labelled records",
            "n_records": len(records),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(output_path, result)
        return result

    log.info("Sending %d requests to challenger at %s", len(records), endpoint)
    champion_scores: list[float] = []
    challenger_scores: list[float] = []
    y_true: list[bool] = []
    errors = 0

    with httpx.Client(timeout=5.0) as client:
        for rec in records:
            score = _send_predict(client, endpoint, rec["payload"])
            if score is None:
                errors += 1
                continue
            challenger_scores.append(score)
            champion_scores.append(rec["champion_score"])
            y_true.append(rec["is_fraud"])

    n_evaluated = len(y_true)
    log.info(
        "Challenger complete: %d evaluated, %d errors", n_evaluated, errors
    )

    if n_evaluated < _MIN_LABELLED:
        log.warning(
            "Only %d successful challenger responses (< %d) — skipping",
            n_evaluated, _MIN_LABELLED,
        )
        result = {
            "status": "skipped",
            "reason": f"fewer than {_MIN_LABELLED} successful challenger responses",
            "n_records": len(records),
            "n_evaluated": n_evaluated,
            "n_errors": errors,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(output_path, result)
        return result

    champion_metrics = _compute_metrics(y_true, champion_scores)
    challenger_metrics = _compute_metrics(y_true, challenger_scores)

    min_auprc = round(_CHALLENGER_RATIO * champion_metrics["auprc"], 6)
    passed = challenger_metrics["auprc"] >= min_auprc

    log.info(
        "Champion AUPRC=%.4f  Challenger AUPRC=%.4f  Threshold=%.4f  Pass=%s",
        champion_metrics["auprc"],
        challenger_metrics["auprc"],
        min_auprc,
        passed,
    )

    result = {
        "status": "pass" if passed else "fail",
        "champion": champion_metrics,
        "challenger": challenger_metrics,
        "min_challenger_auprc": min_auprc,
        "n_records": len(records),
        "n_evaluated": n_evaluated,
        "n_errors": errors,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_path, result)
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare champion vs challenger model AUPRC using audit log."
    )
    parser.add_argument("--endpoint", required=True, help="Staging Lambda base URL.")
    parser.add_argument(
        "--limit", type=int, default=1000, help="Max audit records to scan (default: 1000)."
    )
    parser.add_argument(
        "--output", default="data/reports/shadow_eval.json",
        help="Output path for the JSON report.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    _args = parse_args()
    result = run_shadow_eval(_args.endpoint, _args.limit, _args.output)
    sys.exit(0 if result["status"] in ("pass", "skipped") else 1)
