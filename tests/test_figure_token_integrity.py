"""Deterministic integrity checks added in session 24: every figure is actually cited,
every manuscript token resolves to a real (non-placeholder) value, and every figure
registry entry carries complete, placeholder-free metadata. These close gaps a wrong
artifact or a dangling figure could otherwise slip through (caption boundary-terms are
already gated by ``visual_claim_audit``)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from active_fractal_rabbit.config import DEFAULT_FIGURES
from active_fractal_rabbit.figures import _registry_entry, _set_metadata
from active_fractal_rabbit.manuscript_variables import compute_variables

ROOT = Path(__file__).resolve().parents[1]
_PLACEHOLDERS = ("{{", "}}", "[todo]", "todo", "placeholder", "tbd", "xxx", "fixme")


def _manuscript_text() -> str:
    excluded = {"AGENTS.md", "README.md", "SYNTAX.md"}
    return "\n".join(
        p.read_text(encoding="utf-8")
        for p in sorted((ROOT / "manuscript").glob("*.md"))
        if p.name not in excluded
    )


def test_every_default_figure_is_cited_in_the_manuscript() -> None:
    """Call-site enforcement: every figure the pipeline renders must be referenced by its
    PNG path in the manuscript (no rendered-but-orphan figures, no cited-but-missing keys).

    Uses EXACT cited-filename set membership rather than substring containment, so a key
    that is a suffix of another (e.g. ``anomaly_detection`` vs ``hardened_anomaly_detection``)
    can never be falsely counted as cited (RedTeam-hardened, session 24)."""
    cited = set(re.findall(r"output/figures/([A-Za-z0-9_]+)\.png", _manuscript_text()))
    missing = [key for key in DEFAULT_FIGURES if key not in cited]
    assert not missing, f"rendered figures never cited in the manuscript: {missing}"


def test_no_manuscript_token_renders_as_placeholder() -> None:
    """Token integrity: every {{TOKEN}} used in the manuscript is emitted by
    compute_variables and renders to a concrete value, never a leftover placeholder.
    ('unavailable' is a legitimate CI-safe value for an optional gitignored artifact.)"""
    variables = compute_variables(ROOT)
    used = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", _manuscript_text()))
    missing = sorted(t for t in used if t not in variables)
    assert not missing, f"manuscript tokens not emitted by compute_variables: {missing}"
    placeholdery = sorted(
        t
        for t in used
        if any(bad in str(variables[t]).lower() for bad in _PLACEHOLDERS)
    )
    assert not placeholdery, (
        f"manuscript tokens render as placeholder text: {placeholdery}"
    )


@pytest.fixture(scope="module")
def figure_registry() -> dict:
    # Read-only validator of the SHIPPED registry. Regenerating here would
    # either churn production figures (the pre-sandbox behavior) or require
    # rebuilding the full artifact surface, which the sandboxed
    # test_figures_and_manuscript_variables_are_generated already covers.
    # The full study (scripts/07_run_full_study.py) owns generating this file.
    registry_path = ROOT / "output" / "figures" / "figure_registry.json"
    if not registry_path.is_file():
        pytest.skip(
            "output/figures/figure_registry.json not generated on this checkout; "
            "run scripts/07_run_full_study.py --lane fixture first"
        )
    return json.loads(registry_path.read_text(encoding="utf-8"))


def test_figure_registry_metadata_is_complete(figure_registry: dict) -> None:
    """Metadata completeness: every registry entry has every required field populated with
    a non-empty, placeholder-free string (data_derived is boolean)."""
    required_text = ("key", "filename", "caption", "alt_text", "source")
    offenders: list[str] = []
    for entry in figure_registry["entries"]:
        for field in required_text:
            value = entry.get(field, "")
            if not isinstance(value, str) or not value.strip():
                offenders.append(f"{entry.get('key', '?')}.{field}: empty")
            elif any(
                bad in value.lower()
                for bad in ("[todo]", "placeholder", "tbd", "fixme")
            ):
                offenders.append(f"{entry.get('key', '?')}.{field}: placeholder")
        if not isinstance(entry.get("data_derived"), bool):
            offenders.append(f"{entry.get('key', '?')}.data_derived: not bool")
    assert not offenders, f"incomplete figure metadata: {offenders}"


def test_default_registry_entries_have_source_and_boundary_contracts() -> None:
    context: dict[str, object] = {
        "n_observations": 2400,
        "n_features": 2400,
        "first_observations": 120,
        "smoothing": 0.05,
        "speed_thresholds": "2.0/8.0",
        "control_count": 1,
        "n_travellers": 120,
        "n_inference": 2400,
        "n_candidates": 2,
        "negative_controls_passed": 5,
        "negative_controls_total": 6,
        "pymdp_exact_delta": 1e-12,
        "pymdp_loop_delta": 0.01,
        "pymdp_mean_vfe": 0.3,
        "pymdp_median_step_ms": 2.5,
        "recovery_reference_mass": 0.9,
        "recovery_reference_top1": 0.8,
    }
    privacy_summary = {
        "graceful_degradation_tradeoff": True,
        "min_collection_operating_point": {"found": True, "regime": "coarse"},
    }
    fractal_summary = {"dimension_control_active": True}

    entries = [
        _registry_entry(
            key,
            f"{key}.png",
            "default",
            context,
            privacy_summary,
            fractal_summary,
        )
        for key in DEFAULT_FIGURES
    ]

    assert [entry.key for entry in entries] == list(DEFAULT_FIGURES)
    assert len(entries) == 53
    for entry in entries:
        assert entry.filename == f"{entry.key}.png"
        assert entry.alt_text == entry.caption
        assert " Evidence: " in entry.caption
        assert " Boundary: " in entry.caption
        if entry.key == "dataflow":
            assert entry.source == "source pipeline modules"
            assert entry.data_derived is False
        else:
            assert entry.source.startswith("output/")
            assert Path(entry.source).suffix in {".csv", ".json"}
            assert entry.data_derived is True

    phase = next(
        entry for entry in entries if entry.key == "regime_switch_phase_diagnostic"
    )
    assert "one synthetic regime" in phase.caption
    assert "Boundary: a state-of-the-art claim" in phase.caption

    phase_context = dict(context)
    phase_context["regime_switch_phase_summary"] = {
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
        ],
    }
    phase_bound = _registry_entry(
        "regime_switch_phase_diagnostic",
        "regime_switch_phase_diagnostic.png",
        "default",
        phase_context,
        privacy_summary,
        fractal_summary,
    )
    assert "CI-separated negative segments: explore" in phase_bound.caption
    assert "zero-crossing unresolved segments: commute" in phase_bound.caption
    assert "uniform disadvantage: False" in phase_bound.caption
    assert "worst-to-best magnitude ratio: n/a" in phase_bound.caption
    assert "all three" not in phase_bound.caption

    privacy = next(
        entry for entry in entries if entry.key == "privacy_utility_frontier"
    )
    assert "minimum-collection operating point coarse" in privacy.caption

    replay_fractal = _registry_entry(
        "fractal_dimension_sweep",
        "fractal_dimension_sweep.png",
        "default",
        context,
        privacy_summary,
        {"dimension_control_active": False},
    )
    assert "fixture-replay invariance check" in replay_fractal.caption

    blocked_privacy = _registry_entry(
        "privacy_utility_frontier",
        "privacy_utility_frontier.png",
        "default",
        context,
        {
            "graceful_degradation_tradeoff": False,
            "min_collection_operating_point": {"found": False},
        },
        fractal_summary,
    )
    assert "non-monotone minimization audit" in blocked_privacy.caption


def test_set_metadata_rejects_invalid_dpi(tmp_path: Path) -> None:
    path = tmp_path / "probe.png"
    _set_metadata(path, 220)
    with pytest.raises(ValueError, match="dpi must be positive"):
        _set_metadata(path, 0)
