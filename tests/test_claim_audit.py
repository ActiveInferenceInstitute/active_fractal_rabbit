from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import active_fractal_rabbit.claim_audit as claim_audit
from active_fractal_rabbit.claim_audit import CLAIM_AUDIT_SCHEMA, audit_claims


def test_real_claim_audit_passes_and_covers_high_risk_families() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = audit_claims(root)

    # A checkout without the generated release/external surface cannot satisfy
    # the ledger's cited local paths. Skip only when missing cited artifacts are
    # the *sole* class of violation: any numeric or boundary violation still
    # fails, so this cannot mask a genuine claim drift.
    # (removal condition: cited artifacts generated; owner: release lane)
    if payload["status"] != "pass":
        missing_paths = [
            violation
            for violation in payload["violations"]
            if violation.startswith("claim ledger cites missing local path:")
        ]
        other = [
            violation
            for violation in payload["violations"]
            if not violation.startswith("claim ledger cites missing local path:")
        ]
        if missing_paths and not other:
            pytest.skip(
                "generated release/external surface not present; the claim audit "
                "is blocked only by missing cited artifact(s): "
                + "; ".join(missing_paths)
            )

    assert payload["schema"] == CLAIM_AUDIT_SCHEMA
    assert payload["status"] == "pass"
    assert payload["claim_count"] > 200
    assert payload["ledger_row_count"] >= 80
    assert len(payload["families"]) == 9
    assert payload["violations"] == []
    assert payload["unsupported_high_risk_count"] == 0
    assert payload["unsupported_high_risk_claims"] == []
    assert all(row["ledger_hit_count"] > 0 for row in payload["families"])
    assert "numeric_provenance" in payload["category_counts"]
    assert payload["ledger_numeric_content_checks"] >= 1


def test_claim_audit_surfaces_unsupported_high_risk_claims() -> None:
    """The unsupported-high-risk count must be backed by an inspectable listing.

    Regression against the oracle blind spot: previously only a bare
    ``unsupported_high_risk_count`` was emitted, so nobody could see WHICH
    high-risk claims lacked a supporting category. The listing must exist, match
    the count exactly, and every entry must genuinely be high-risk with no
    supporting category — so the number can never again be decorative.
    """

    root = Path(__file__).resolve().parents[1]
    payload = audit_claims(root)

    listing = payload["unsupported_high_risk_claims"]
    assert isinstance(listing, list)
    assert len(listing) == payload["unsupported_high_risk_count"]

    supporting = {
        "numeric_provenance",
        "artifact_backed",
        "citation_backed",
        "formal_reference",
        "limitation",
        "roadmap",
    }
    for record in listing:
        assert record["risk_families"], "unsupported entry must be high-risk"
        assert not set(record["categories"]).intersection(supporting)
        assert record["location"]
        assert record["text"]


def test_claim_audit_detects_missing_high_risk_ledger_family(
    tmp_path: Path, monkeypatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "manuscript", tmp_path / "manuscript")
    (tmp_path / "docs").mkdir()
    ledger = (root / "docs" / "claim_ledger.md").read_text(encoding="utf-8")
    filtered = "\n".join(
        line
        for line in ledger.splitlines()
        if "Factorial pymdp diagnostics expose synthetic movement/reporting factors"
        not in line
    )
    (tmp_path / "docs" / "claim_ledger.md").write_text(
        filtered + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        claim_audit,
        "_verify_ledger_local_paths",
        lambda _root, _rows: {"checked": [], "violations": []},
    )

    payload = audit_claims(tmp_path)

    assert payload["status"] == "fail"
    assert any(
        "factorial_latent_mode_interpretability: no claim-ledger row covers family"
        in item
        for item in payload["violations"]
    )


def test_claim_audit_numeric_content_check_fires_on_injected_mismatch(
    tmp_path: Path,
) -> None:
    """Proof-of-detection: ``_verify_ledger_local_paths`` only checks that a cited
    path exists, never that its content supports the claim's numbers — this is the
    RedTeam-found gap. ``_verify_ledger_numeric_content`` closes it; this test proves
    it actually fires on a claim whose stated number is absent from the cited
    artifact, not merely that the code runs.
    """

    artifact = tmp_path / "metrics.json"
    artifact.write_text('{"accuracy": 0.500, "loss": 1.234}\n', encoding="utf-8")
    rows = [
        {
            "claim": "The model reaches 0.999 accuracy on the holdout split",
            "evidence_type": "artifact",
            "source": "`metrics.json`",
        }
    ]

    report = claim_audit._verify_ledger_numeric_content(tmp_path, rows)

    assert report["claims_checked_for_numeric_content"] == 1
    assert len(report["violations"]) == 1
    assert "0.999" in report["violations"][0]


def test_claim_audit_numeric_content_check_passes_on_matching_value(
    tmp_path: Path,
) -> None:
    """Negative control for the proof-of-detection test above: a claim whose number
    genuinely is in the cited artifact (exactly, or within the 1% tolerance a
    rounded-for-prose claim needs) must not be flagged.
    """

    artifact = tmp_path / "metrics.json"
    artifact.write_text('{"nested": {"accuracy": 0.07667}}\n', encoding="utf-8")
    rows = [
        {
            "claim": "The model reaches ~0.077 accuracy on the holdout split",
            "evidence_type": "artifact",
            "source": "`metrics.json`",
        }
    ]

    report = claim_audit._verify_ledger_numeric_content(tmp_path, rows)

    assert report["claims_checked_for_numeric_content"] == 1
    assert report["violations"] == []


def test_claim_audit_numeric_content_mismatch_blocks_release_mode(
    tmp_path: Path, monkeypatch
) -> None:
    """A known-wrong numeric claim must fail the production release oracle."""

    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "manuscript", tmp_path / "manuscript")
    (tmp_path / "docs").mkdir()
    (tmp_path / "output" / "runs").mkdir(parents=True)
    (tmp_path / "output" / "runs" / "metrics.json").write_text(
        '{"accuracy": 0.5}\n', encoding="utf-8"
    )
    ledger = (root / "docs" / "claim_ledger.md").read_text(encoding="utf-8")
    ledger += (
        "| A synthetic test claim stating the model reaches 0.999 accuracy | artifact | "
        "`output/runs/metrics.json` |\n"
    )
    (tmp_path / "docs" / "claim_ledger.md").write_text(ledger, encoding="utf-8")
    monkeypatch.setattr(
        claim_audit,
        "_verify_ledger_local_paths",
        lambda _root, _rows: {"checked": [], "violations": []},
    )

    payload = audit_claims(tmp_path)

    assert any("0.999" in warning for warning in payload["numeric_content_warnings"])
    assert payload["status"] == "fail"
    assert any("0.999" in violation for violation in payload["violations"])

    sandbox = audit_claims(tmp_path, strict_numeric=False)
    assert sandbox["status"] == "pass"
    assert sandbox["violations"] == []


def test_claim_audit_rejects_upstream_attribution_for_fixture_headline(
    tmp_path: Path, monkeypatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "manuscript", tmp_path / "manuscript")
    (tmp_path / "docs").mkdir()
    shutil.copy2(root / "docs" / "claim_ledger.md", tmp_path / "docs" / "claim_ledger.md")
    abstract = tmp_path / "manuscript" / "00_abstract.md"
    abstract.write_text(
        abstract.read_text(encoding="utf-8")
        + "\nSynthetic FractalRabbit data generated by open-source software from the National Security Agency.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        claim_audit,
        "_verify_ledger_local_paths",
        lambda _root, _rows: {"checked": [], "violations": []},
    )
    monkeypatch.setattr(
        claim_audit,
        "_verify_ledger_numeric_content",
        lambda _root, _rows: {
            "claims_checked_for_numeric_content": 0,
            "violations": [],
        },
    )

    payload = audit_claims(tmp_path)

    assert payload["status"] == "fail"
    assert any("upstream FractalRabbit simulator" in item for item in payload["violations"])


def test_claim_audit_ledger_path_check_allows_test_function_anchors(
    tmp_path: Path,
) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_example.py").write_text(
        "def test_example():\n    pass\n", encoding="utf-8"
    )
    rows = [
        {
            "claim": "example",
            "evidence_type": "test",
            "source": "`tests/test_example.py::test_example`",
        }
    ]

    report = claim_audit._verify_ledger_local_paths(tmp_path, rows)

    assert report["checked"] == ["tests/test_example.py"]
    assert report["violations"] == []
