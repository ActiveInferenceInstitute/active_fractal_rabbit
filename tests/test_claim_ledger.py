from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from active_fractal_rabbit.claim_ledger import (
    cited_generated_artifacts,
    verify_claim_ledger_artifacts,
)


def test_claim_ledger_artifact_parser_and_missing_gate(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    ledger = tmp_path / "docs" / "claim_ledger.md"
    ledger.write_text(
        "| claim | source |\n"
        "| --- | --- |\n"
        "| generated | `output/runs/a.json`, `output/figures/b.png` |\n"
        "| non-generated | `src/package/module.py`, `citation_key` |\n",
        encoding="utf-8",
    )
    (tmp_path / "output" / "runs").mkdir(parents=True)
    (tmp_path / "output" / "runs" / "a.json").write_text("{}\n", encoding="utf-8")

    assert cited_generated_artifacts(ledger) == [
        "output/figures/b.png",
        "output/runs/a.json",
    ]
    with pytest.raises(FileNotFoundError, match="output/figures/b.png"):
        verify_claim_ledger_artifacts(tmp_path)


def test_real_claim_ledger_cites_promoted_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = set(cited_generated_artifacts(root / "docs" / "claim_ledger.md"))

    assert {
        "output/runs/latent_regime_hmm.json",
        "output/runs/neural_baseline.json",
        "output/runs/learned_stay_rate.json",
        "output/runs/heldout_reporting_process.json",
        "output/runs/cross_generator_transfer.json",
        "output/runs/model_conformance.json",
        "output/runs/efe_location_frontier.json",
        "output/runs/linkage_risk_frontier.json",
        "output/runs/poisoning_robustness.json",
        "output/cover/active_fractal_rabbit_cover.png",
    } <= paths


def test_claim_ledger_figure_count_matches_visual_quality_audit() -> None:
    root = Path(__file__).resolve().parents[1]
    ledger = (root / "docs" / "claim_ledger.md").read_text(encoding="utf-8")
    audit_path = root / "output" / "runs" / "visual_quality_audit.json"
    if not audit_path.is_file():
        pytest.skip(
            f"generated visual-quality audit not present: missing {audit_path} "
            "(removal condition: run scripts/07_run_full_study.py --lane "
            "fixture; owner: fixture lane)"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    match = re.search(
        r"default visual registry contains (\d+) source-bound manuscript figures",
        ledger,
    )

    assert match is not None
    assert int(match.group(1)) == int(audit["figure_count"])
