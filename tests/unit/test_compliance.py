"""Tests for generate_compliance_manifest and the compliance CLI."""

import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest

_REQUIRED_ARTICLE_FIELDS = {
    "article",
    "requirement",
    "implementing_component",
    "evidence_artifact",
    "status",
    "mandatory",
}


# ---------------------------------------------------------------------------
# 1. generate_compliance_manifest returns a dict with all 7 articles
# ---------------------------------------------------------------------------

def test_generate_compliance_manifest_returns_dict_with_all_7_articles():
    from src.monitoring.compliance import generate_compliance_manifest

    manifest = generate_compliance_manifest({})

    assert isinstance(manifest, dict)
    assert "articles" in manifest
    assert len(manifest["articles"]) == 7
    article_ids = [a["article"] for a in manifest["articles"]]
    for art in ["Art. 9", "Art. 10", "Art. 11", "Art. 12", "Art. 13", "Art. 14", "Art. 15"]:
        assert art in article_ids, f"{art} missing from manifest"


# ---------------------------------------------------------------------------
# 2. Each article entry contains all required fields
# ---------------------------------------------------------------------------

def test_each_article_entry_contains_all_required_fields():
    from src.monitoring.compliance import generate_compliance_manifest

    manifest = generate_compliance_manifest({})

    for entry in manifest["articles"]:
        missing = _REQUIRED_ARTICLE_FIELDS - set(entry.keys())
        assert not missing, f"Article {entry.get('article')} missing fields: {missing}"
        assert entry["status"] in ("satisfied", "partial", "absent"), (
            f"Invalid status {entry['status']!r} for {entry['article']}"
        )
        assert isinstance(entry["mandatory"], bool)


# ---------------------------------------------------------------------------
# 3. overall_status is "non_compliant" when a mandatory article is "absent"
# ---------------------------------------------------------------------------

def test_overall_status_non_compliant_when_mandatory_article_absent(monkeypatch):
    from src.monitoring.compliance import generate_compliance_manifest

    def _absent_art12():
        return {
            "article": "Art. 12",
            "requirement": "Record-keeping",
            "implementing_component": "src/monitoring/audit_logger.py",
            "evidence_artifact": "DynamoDB: fraud-audit-log",
            "status": "absent",
            "mandatory": True,
        }

    monkeypatch.setattr("src.monitoring.compliance._check_art12", _absent_art12)

    manifest = generate_compliance_manifest({})
    assert manifest["overall_status"] == "non_compliant"


# ---------------------------------------------------------------------------
# 4. overall_status is "compliant" when all mandatory articles are satisfied/partial
# ---------------------------------------------------------------------------

def test_overall_status_compliant_when_mandatory_articles_satisfied_or_partial():
    from src.monitoring.compliance import generate_compliance_manifest

    # Art. 12 and Art. 14 are satisfied (audit_logger has msgpack + requires_review).
    # Art. 13 will be "partial" (DynamoDB unavailable in tests) — that's still compliant.
    manifest = generate_compliance_manifest({})

    mandatory_entries = [a for a in manifest["articles"] if a["mandatory"]]
    for entry in mandatory_entries:
        assert entry["status"] in ("satisfied", "partial"), (
            f"Mandatory {entry['article']} has status {entry['status']!r}"
        )
    assert manifest["overall_status"] == "compliant"


# ---------------------------------------------------------------------------
# 5. CLI run_cli() returns 1 when a mandatory article is "absent"
# ---------------------------------------------------------------------------

def test_cli_returns_1_when_mandatory_article_absent(monkeypatch, tmp_path):
    from src.monitoring.compliance import run_cli

    def _mock_generate(bundle):
        return {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "overall_status": "non_compliant",
            "articles": [
                {
                    "article": "Art. 12",
                    "requirement": "Record-keeping",
                    "implementing_component": "src/monitoring/audit_logger.py",
                    "evidence_artifact": "DynamoDB: fraud-audit-log",
                    "status": "absent",
                    "mandatory": True,
                }
            ],
        }

    monkeypatch.setattr("src.monitoring.compliance.generate_compliance_manifest", _mock_generate)
    monkeypatch.setattr("src.monitoring.compliance._REPORTS_DIR", tmp_path)

    exit_code = run_cli()
    assert exit_code == 1


# ---------------------------------------------------------------------------
# 6. CLI run_cli() returns 0 when all mandatory articles are satisfied
# ---------------------------------------------------------------------------

def test_cli_returns_0_when_mandatory_articles_satisfied(monkeypatch, tmp_path):
    from src.monitoring.compliance import run_cli

    def _mock_generate(bundle):
        return {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "overall_status": "compliant",
            "articles": [
                {
                    "article": "Art. 12",
                    "requirement": "Record-keeping",
                    "implementing_component": "src/monitoring/audit_logger.py",
                    "evidence_artifact": "DynamoDB: fraud-audit-log",
                    "status": "satisfied",
                    "mandatory": True,
                },
                {
                    "article": "Art. 13",
                    "requirement": "Transparency",
                    "implementing_component": "src/explainability/shap_explainer.py",
                    "evidence_artifact": "DynamoDB: fraud-shap-store",
                    "status": "partial",
                    "mandatory": True,
                },
                {
                    "article": "Art. 14",
                    "requirement": "Human oversight",
                    "implementing_component": "src/monitoring/audit_logger.py",
                    "evidence_artifact": "DynamoDB: fraud-audit-log (requires_review GSI)",
                    "status": "satisfied",
                    "mandatory": True,
                },
            ],
        }

    monkeypatch.setattr("src.monitoring.compliance.generate_compliance_manifest", _mock_generate)
    monkeypatch.setattr("src.monitoring.compliance._REPORTS_DIR", tmp_path)

    exit_code = run_cli()
    assert exit_code == 0


# ---------------------------------------------------------------------------
# 7. compliance_manifest.json is written to data/reports/ after generation
# ---------------------------------------------------------------------------

def test_compliance_manifest_json_written_to_data_reports(monkeypatch, tmp_path):
    from src.monitoring.compliance import run_cli

    def _mock_generate(bundle):
        return {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "overall_status": "compliant",
            "articles": [],
        }

    monkeypatch.setattr("src.monitoring.compliance.generate_compliance_manifest", _mock_generate)
    monkeypatch.setattr("src.monitoring.compliance._REPORTS_DIR", tmp_path)

    run_cli()

    manifest_path = tmp_path / "compliance_manifest.json"
    assert manifest_path.exists(), "compliance_manifest.json was not written"
    data = json.loads(manifest_path.read_text())
    assert data["overall_status"] == "compliant"


# ---------------------------------------------------------------------------
# 8. Art. 12 status is derived from audit_logger source inspection, not hardcoded
# ---------------------------------------------------------------------------

def test_art12_status_derived_from_audit_logger_inspection(monkeypatch):
    """Patching inspect.getsource to return source without msgpack changes Art. 12 status."""
    from src.monitoring.compliance import _check_art12

    monkeypatch.setattr(inspect, "getsource", lambda *a, **kw: "def write(): pass")

    result = _check_art12()
    assert result["status"] != "satisfied", (
        f"Expected status to change when msgpack absent from source, got {result['status']!r}"
    )


# ---------------------------------------------------------------------------
# 9. Art. 11 status is "satisfied" when model_card.json exists
# ---------------------------------------------------------------------------

def test_art11_status_satisfied_when_model_card_exists():
    """model_card.json exists in the project root — Art. 11 must be 'satisfied'."""
    from src.monitoring.compliance import _check_art11

    result = _check_art11()
    assert result["status"] == "satisfied", (
        f"Expected 'satisfied' for Art. 11 (model_card.json exists), got {result['status']!r}"
    )


# ---------------------------------------------------------------------------
# 10. Art. 11 status is "absent" when model_card.json is missing
# ---------------------------------------------------------------------------

def test_art11_status_absent_when_model_card_missing(monkeypatch, tmp_path):
    """When _REPO_ROOT points to a temp dir without model_card.json, status is 'absent'."""
    monkeypatch.setattr("src.monitoring.compliance._REPO_ROOT", tmp_path)

    from src.monitoring.compliance import _check_art11

    result = _check_art11()
    assert result["status"] == "absent", (
        f"Expected 'absent' when model_card.json missing, got {result['status']!r}"
    )
