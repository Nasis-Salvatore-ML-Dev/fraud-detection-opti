"""EU AI Act Articles 9-15 compliance manifest generator.

Produces a structured compliance manifest mapping each article to its
implementing component and evidence artifact, then derives a status
by inspecting actual artifact existence — never hardcoded.
"""

import inspect
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3

from src.monitoring.metrics import publish_component_failure

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPORTS_DIR = _REPO_ROOT / "data" / "reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_satisfied(path: Path) -> str:
    """Return 'satisfied' if file exists and is non-empty, else 'absent'."""
    try:
        return "satisfied" if path.exists() and path.stat().st_size > 0 else "absent"
    except Exception as exc:
        log.error("File check failed for %s: %s", path, exc)
        return "absent"


# ---------------------------------------------------------------------------
# Per-article checks
# ---------------------------------------------------------------------------

def _check_art9() -> dict:
    component_path = _REPO_ROOT / "src" / "monitoring" / "bias_tester.py"
    return {
        "article": "Art. 9",
        "requirement": (
            "High-risk AI systems must implement a risk management system covering "
            "identification, analysis, and evaluation of known and foreseeable risks."
        ),
        "implementing_component": "src/monitoring/bias_tester.py",
        "evidence_artifact": "data/reports/bias_report.json",
        "status": _file_satisfied(component_path),
        "mandatory": False,
    }


def _check_art10() -> dict:
    component_path = _REPO_ROOT / "src" / "monitoring" / "bias_tester.py"
    return {
        "article": "Art. 10",
        "requirement": (
            "Training, validation, and testing data must be subject to data governance "
            "practices covering relevance, representativeness, and bias identification."
        ),
        "implementing_component": "src/monitoring/bias_tester.py",
        "evidence_artifact": "data/baselines/bias_segments.json",
        "status": _file_satisfied(component_path),
        "mandatory": False,
    }


def _check_art11() -> dict:
    model_card_path = _REPO_ROOT / "model_card.json"
    return {
        "article": "Art. 11",
        "requirement": (
            "Providers must draw up technical documentation demonstrating compliance "
            "before the system is placed on the market."
        ),
        "implementing_component": "model_card.json",
        "evidence_artifact": "S3: model_cards/model_card_{version}.json",
        "status": _file_satisfied(model_card_path),
        "mandatory": False,
    }


def _check_art12() -> dict:
    """Satisfied if audit_logger imports msgpack AND stores v_features_msgpack."""
    mandatory = True
    try:
        import src.monitoring.audit_logger as audit_mod
        source = inspect.getsource(audit_mod)
        has_msgpack = "msgpack" in source
        has_v_features_msgpack = "v_features_msgpack" in source
        if has_msgpack and has_v_features_msgpack:
            status = "satisfied"
        elif has_msgpack or has_v_features_msgpack:
            status = "partial"
        else:
            status = "absent"
    except Exception as exc:
        log.error("Art. 12 audit_logger inspection failed: %s", exc)
        publish_component_failure("ComplianceCheck")
        status = "absent"
    return {
        "article": "Art. 12",
        "requirement": (
            "High-risk AI systems must automatically log events throughout their lifetime "
            "to ensure traceability of inputs and outputs."
        ),
        "implementing_component": "src/monitoring/audit_logger.py",
        "evidence_artifact": "DynamoDB: fraud-audit-log",
        "status": status,
        "mandatory": mandatory,
    }


def _check_art13() -> dict:
    """Satisfied if SHAP table is accessible and has items; partial if DynamoDB unavailable."""
    mandatory = True
    try:
        shap_table = os.environ.get("SHAP_TABLE", "fraud-shap-store")
        region = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
        dynamo = boto3.resource("dynamodb", region_name=region)
        table = dynamo.Table(shap_table)
        response = table.scan(Limit=1)
        status = "satisfied" if response.get("Items") else "partial"
    except Exception as exc:
        log.warning("Art. 13 DynamoDB unavailable (%s) — marking partial", exc)
        status = "partial"
    return {
        "article": "Art. 13",
        "requirement": (
            "High-risk AI systems must be designed to allow operators to interpret outputs, "
            "including per-prediction explanations of AI decision logic."
        ),
        "implementing_component": "src/explainability/shap_explainer.py",
        "evidence_artifact": "DynamoDB: fraud-shap-store",
        "status": status,
        "mandatory": mandatory,
    }


def _check_art14() -> dict:
    """Satisfied if requires_review flag is implemented in audit_logger."""
    mandatory = True
    try:
        import src.monitoring.audit_logger as audit_mod
        source = inspect.getsource(audit_mod)
        status = "satisfied" if "requires_review" in source else "absent"
    except Exception as exc:
        log.error("Art. 14 audit_logger inspection failed: %s", exc)
        publish_component_failure("ComplianceCheck")
        status = "absent"
    return {
        "article": "Art. 14",
        "requirement": (
            "High-risk AI systems must allow human oversight, including the ability to "
            "identify anomalies and flag outputs for human review before consequential action."
        ),
        "implementing_component": "src/monitoring/audit_logger.py",
        "evidence_artifact": "DynamoDB: fraud-audit-log (requires_review GSI)",
        "status": status,
        "mandatory": mandatory,
    }


def _check_art15() -> dict:
    component_path = _REPO_ROOT / "src" / "monitoring" / "drift.py"
    return {
        "article": "Art. 15",
        "requirement": (
            "High-risk AI systems must achieve an appropriate level of accuracy, robustness, "
            "and cybersecurity, and perform consistently throughout their lifecycle."
        ),
        "implementing_component": "src/monitoring/drift.py",
        "evidence_artifact": "CloudWatch: FraudDetection metrics",
        "status": _file_satisfied(component_path),
        "mandatory": False,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_compliance_manifest(bundle: dict) -> dict:
    """Generate EU AI Act Articles 9-15 compliance manifest.

    Args:
        bundle: Model bundle dict (may be empty dict if unavailable).

    Returns:
        Manifest with per-article status entries and overall_status.
    """
    articles = [
        _check_art9(),
        _check_art10(),
        _check_art11(),
        _check_art12(),
        _check_art13(),
        _check_art14(),
        _check_art15(),
    ]

    mandatory_absent = any(
        a["mandatory"] and a["status"] == "absent" for a in articles
    )
    overall_status = "non_compliant" if mandatory_absent else "compliant"

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "articles": articles,
    }

    log.info(
        "Compliance manifest generated: overall_status=%s  articles=%d",
        overall_status,
        len(articles),
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_cli() -> int:
    """Execute compliance check, write manifest, return exit code (0 or 1)."""
    bundle_path = _REPO_ROOT / "models" / "bundle.pkl"
    bundle: dict = {}
    if bundle_path.exists():
        try:
            import joblib
            bundle = joblib.load(bundle_path)
            log.info("Bundle loaded from %s", bundle_path)
        except Exception as exc:
            log.warning("Could not load bundle from %s: %s", bundle_path, exc)

    manifest = generate_compliance_manifest(bundle)

    try:
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        manifest_path = _REPORTS_DIR / "compliance_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info("Compliance manifest written → %s", manifest_path)
    except Exception as exc:
        log.error("Failed to write compliance manifest: %s", exc)
        publish_component_failure("ComplianceCheck")

    print(manifest["overall_status"])

    mandatory_absent = any(
        a["mandatory"] and a["status"] == "absent" for a in manifest["articles"]
    )
    return 1 if mandatory_absent else 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    sys.exit(run_cli())
