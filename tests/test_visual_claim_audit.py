from __future__ import annotations

import hashlib
import json
from pathlib import Path

import active_fractal_rabbit.visual_claim_audit as visual_claim_audit
from active_fractal_rabbit.visual_claim_audit import (
    VISUAL_CLAIM_AUDIT_SCHEMA,
    VisualExpectation,
    audit_visual_claims,
)


def test_real_visual_claim_audit_passes() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = audit_visual_claims(root)

    assert payload["schema"] == VISUAL_CLAIM_AUDIT_SCHEMA
    assert payload["status"] == "pass"
    assert payload["figure_count"] == 53
    assert payload["brief_count"] == 54
    assert payload["first_page_layers"] == ["cover", "evidence_atlas", "dataflow"]
    assert payload["violations"] == []
    evidence = next(
        row for row in payload["briefs"] if row["visual_layer"] == "evidence_atlas"
    )
    assert "full evidence chain" in evidence["story_job"]
    assert evidence["reader_task"].startswith("Use the visual to orient")
    assert "surveillance readiness" in evidence["claim_boundary"]
    assert "PDF export" in evidence["interaction_or_fallback"]


def test_visual_claim_audit_detects_caption_without_nonclaim_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_visual_project(tmp_path)
    monkeypatch.setattr(
        visual_claim_audit,
        "FIGURE_EXPECTATIONS",
        {
            "test_layer": VisualExpectation(
                "Test story job.",
                "Rows.",
                "Bars.",
                ("source.json",),
                ("not empirical",),
            )
        },
    )

    registry = {
        "entries": [
            {
                "key": "test_layer",
                "filename": "test_layer.png",
                "caption": "Source source.json shows a synthetic diagnostic.",
                "alt_text": "Synthetic diagnostic.",
                "source": "output/runs/source.json",
                "data_derived": True,
            }
        ]
    }
    (tmp_path / "output" / "figures" / "figure_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )

    payload = audit_visual_claims(tmp_path)

    assert payload["status"] == "fail"
    assert any(
        "caption missing evidence/boundary contract" in item
        for item in payload["violations"]
    )


def test_visual_claim_audit_rejects_false_privacy_utility_graceful_caption(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_visual_project(tmp_path)
    expectation = visual_claim_audit.FIGURE_EXPECTATIONS["privacy_utility_frontier"]
    monkeypatch.setattr(
        visual_claim_audit,
        "FIGURE_EXPECTATIONS",
        {"privacy_utility_frontier": expectation},
    )
    (tmp_path / "manuscript" / "01.md").write_text(
        "![caption](../output/figures/privacy_utility_frontier.png){#fig:privacy-utility-frontier}\n",
        encoding="utf-8",
    )
    (tmp_path / "output" / "runs" / "privacy_utility_frontier.json").write_text(
        json.dumps(
            {
                "graceful_degradation_tradeoff": False,
                "min_collection_operating_point": {"found": False},
            }
        ),
        encoding="utf-8",
    )
    registry = {
        "entries": [
            {
                "key": "privacy_utility_frontier",
                "filename": "privacy_utility_frontier.png",
                "caption": (
                    "Source privacy_utility_frontier.json shows minimization and re-identification. "
                    "Points trace fine-to-coarse grids and both axes fall together as a "
                    "graceful-degradation governance tradeoff; no minimum-collection operating point "
                    "is noted. Supports a synthetic governance audit; does not support a "
                    "state-of-the-art claim, association over real people, or a real-mobility claim."
                ),
                "alt_text": "Privacy utility frontier.",
                "source": "output/runs/privacy_utility_frontier.json",
                "data_derived": True,
            }
        ]
    }
    (tmp_path / "output" / "figures" / "figure_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )

    payload = audit_visual_claims(tmp_path)

    assert payload["status"] == "fail"
    assert any(
        "caption claims graceful degradation although source gate is false" in item
        for item in payload["violations"]
    )


def test_visual_claim_audit_rejects_stale_privacy_utility_failed_caption_when_gate_clears(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_visual_project(tmp_path)
    expectation = visual_claim_audit.FIGURE_EXPECTATIONS["privacy_utility_frontier"]
    monkeypatch.setattr(
        visual_claim_audit,
        "FIGURE_EXPECTATIONS",
        {"privacy_utility_frontier": expectation},
    )
    (tmp_path / "manuscript" / "01.md").write_text(
        "![caption](../output/figures/privacy_utility_frontier.png){#fig:privacy-utility-frontier}\n",
        encoding="utf-8",
    )
    (tmp_path / "output" / "runs" / "privacy_utility_frontier.json").write_text(
        json.dumps(
            {
                "graceful_degradation_tradeoff": True,
                "min_collection_operating_point": {"found": True, "regime": "grid_2x2"},
            }
        ),
        encoding="utf-8",
    )
    registry = {
        "entries": [
            {
                "key": "privacy_utility_frontier",
                "filename": "privacy_utility_frontier.png",
                "caption": (
                    "Source privacy_utility_frontier.json shows minimization. "
                    "The full-grid ladder is non-monotone and has no minimum-collection operating point. "
                    "Supports a synthetic governance audit; does not support a state-of-the-art claim, "
                    "association over real people, or a real-mobility claim."
                ),
                "alt_text": "Privacy utility frontier.",
                "source": "output/runs/privacy_utility_frontier.json",
                "data_derived": True,
            }
        ]
    }
    (tmp_path / "output" / "figures" / "figure_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )

    payload = audit_visual_claims(tmp_path)

    assert payload["status"] == "fail"
    assert any(
        "caption reports failed or non-monotone privacy-utility gate although source gate is true"
        in item
        for item in payload["violations"]
    )
    assert any(
        "caption reports no minimum-collection operating point although source found one"
        in item
        for item in payload["violations"]
    )


def test_visual_claim_audit_rejects_false_uniform_phase_caption(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_visual_project(tmp_path)
    expectation = visual_claim_audit.FIGURE_EXPECTATIONS[
        "regime_switch_phase_diagnostic"
    ]
    monkeypatch.setattr(
        visual_claim_audit,
        "FIGURE_EXPECTATIONS",
        {"regime_switch_phase_diagnostic": expectation},
    )
    (tmp_path / "manuscript" / "01.md").write_text(
        "![caption](../output/figures/regime_switch_phase_diagnostic.png)"
        "{#fig:regime-switch-phase-diagnostic}\n",
        encoding="utf-8",
    )
    source = tmp_path / "output" / "runs" / "regime_switch_phase_diagnostic.json"
    source.write_text(
        json.dumps(
            {
                "uniform_disadvantage": False,
                "worst_to_best_magnitude_ratio": None,
                "segments": [
                    {
                        "segment": "commute",
                        "clustered_ci_low": -0.04,
                        "clustered_ci_high": 0.03,
                    },
                    {
                        "segment": "explore",
                        "clustered_ci_low": -0.16,
                        "clustered_ci_high": -0.05,
                    },
                    {
                        "segment": "return_stabilization",
                        "clustered_ci_low": -0.31,
                        "clustered_ci_high": -0.19,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = {
        "entries": [
            {
                "key": "regime_switch_phase_diagnostic",
                "filename": "regime_switch_phase_diagnostic.png",
                "caption": (
                    "Source regime_switch_phase_diagnostic.json shows commute, explore, and "
                    "return_stabilization; all three currently sit below zero with a CI "
                    "excluding zero. Supports a structural audit; does not support a "
                    "state-of-the-art claim or real-mobility claim."
                ),
                "alt_text": "Phase diagnostic.",
                "source": "output/runs/regime_switch_phase_diagnostic.json",
                "data_derived": True,
            }
        ]
    }
    (tmp_path / "output" / "figures" / "figure_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )

    payload = audit_visual_claims(tmp_path)

    assert payload["status"] == "fail"
    assert any("intervals exclude zero" in item for item in payload["violations"])
    assert any("source ratio is null" in item for item in payload["violations"])


def test_visual_claim_audit_rejects_swapped_registered_figure(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_visual_project(tmp_path)
    monkeypatch.setattr(
        visual_claim_audit,
        "FIGURE_EXPECTATIONS",
        {
            "test_layer": VisualExpectation(
                "Test a source-bound visual.",
                "Rows.",
                "Bars.",
                ("source.json",),
                ("not empirical",),
            )
        },
    )
    (tmp_path / "manuscript" / "01.md").write_text(
        "![caption](../output/figures/test_layer.png){#fig:test-layer}\n",
        encoding="utf-8",
    )
    source = tmp_path / "output" / "runs" / "source.json"
    source.write_text('{"value": 1}\n', encoding="utf-8")
    figure = tmp_path / "output" / "figures" / "test_layer.png"
    figure.write_bytes(b"first-image")
    caption = (
        "Source source.json provides a synthetic diagnostic. Supports a bounded test; "
        "does not support not empirical conclusions."
    )
    image_sha = hashlib.sha256(figure.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    binding = hashlib.sha256()
    for value in ("test_layer", caption, image_sha, source_sha):
        binding.update(value.encode("utf-8"))
    registry = {
        "schema": "active-fractal-rabbit-figure-registry-v2",
        "entries": [
            {
                "key": "test_layer",
                "filename": "test_layer.png",
                "caption": caption,
                "alt_text": caption,
                "source": "output/runs/source.json",
                "data_derived": True,
                "image_sha256": image_sha,
                "source_sha256": source_sha,
                "binding_sha256": binding.hexdigest(),
            }
        ],
    }
    (tmp_path / "output" / "figures" / "figure_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )
    figure.write_bytes(b"swapped-image")

    payload = audit_visual_claims(tmp_path)

    assert payload["status"] == "fail"
    assert any("image checksum differs" in item for item in payload["violations"])


def _seed_visual_project(root: Path) -> None:
    (root / "output" / "figures").mkdir(parents=True)
    (root / "output" / "runs" / "default").mkdir(parents=True)
    (root / "output" / "cover").mkdir(parents=True)
    (root / "manuscript").mkdir()
    (root / "manuscript" / "01.md").write_text(
        "![caption](../output/figures/test_layer.png){#fig:test-layer}\n",
        encoding="utf-8",
    )
    (root / "output" / "runs" / "source.json").write_text("{}\n", encoding="utf-8")
    for rel in (
        "output/runs/default/metrics.json",
        "output/runs/spatial_resolution_frontier.json",
        "output/runs/forward_gates_summary.json",
        "output/runs/efe_location_frontier.json",
        "output/runs/linkage_risk_frontier.json",
        "output/runs/poisoning_robustness.json",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (root / "output" / "cover" / "active_fractal_rabbit_cover.png").write_bytes(
        b"x" * 1200
    )
