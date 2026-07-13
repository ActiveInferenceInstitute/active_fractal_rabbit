from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from active_fractal_rabbit.config import FRACTALRABBIT_PINNED_COMMIT
from active_fractal_rabbit.sparsity import SPARSITY_SCHEMA, run_sparsity_analysis

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (ROOT / "data" / "fixtures" / "fractalrabbit_sample_large.csv").resolve()


def _config(tmp_path: Path, seed: int = 7) -> Path:
    path = tmp_path / f"sparsity_{seed}.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            run_id: sparsity_{seed}
            seed: {seed}
            fractalrabbit:
              upstream_commit: {FRACTALRABBIT_PINNED_COMMIT}
              execution: fixture
              fixture_csv: {FIXTURE}
            observation:
              spatial_bins: 3
              time_bins_per_day: 4
            model:
              smoothing: 0.05
            closed_loop:
              iterations: 1
              candidates: []
            """
        ),
        encoding="utf-8",
    )
    return path


def test_sparsity_analysis_schema_and_monotone_evidence(tmp_path) -> None:
    summary_path = run_sparsity_analysis(_config(tmp_path))
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert payload["schema"] == SPARSITY_SCHEMA
    assert payload["n_levels"] == 4
    levels = sorted(payload["levels"], key=lambda item: item["fraction"], reverse=True)
    assert [level["fraction"] for level in levels] == [1.0, 0.75, 0.5, 0.25]
    # report counts strictly decrease with the fraction
    counts = [level["n_reports"] for level in levels]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == 2400.0
    for level in levels:
        # full model dominates the deep-latent model at every density
        assert (
            level["full_mean_mass_on_true_cell"]
            >= level["latent_mean_mass_on_true_cell"]
        )
        # pymdp matched-prior validation holds at every level
        assert level["pymdp_exact_inference_max_abs_delta"] < 1e-4
        assert level["pymdp_validation_rows"] == 32.0
        assert level["pymdp_source_observation_rows"] == level["n_reports"]
        assert 0.0 < level["pymdp_validation_fraction"] <= 1.0
        assert 0.0 <= level["latent_top1_accuracy"] <= 1.0
    assert "not empirical mobility findings" in payload["caveat"]
    assert "deterministic sentinel rows" in payload["caveat"]


def test_sparsity_is_deterministic_per_seed_and_flips_with_seed(tmp_path) -> None:
    first = json.loads(
        run_sparsity_analysis(_config(tmp_path, seed=7)).read_text(encoding="utf-8")
    )
    second = json.loads(
        run_sparsity_analysis(_config(tmp_path, seed=7)).read_text(encoding="utf-8")
    )
    assert first["levels"] == second["levels"]

    other = json.loads(
        run_sparsity_analysis(_config(tmp_path, seed=8)).read_text(encoding="utf-8")
    )
    assert first["levels"] != other["levels"]


def test_sparsity_rejects_external_configs_and_bad_fractions(tmp_path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="fractions"):
        run_sparsity_analysis(config, fractions=(1.5,))
    external = tmp_path / "external.yaml"
    external.write_text(
        textwrap.dedent(
            f"""
            run_id: ext
            fractalrabbit:
              upstream_commit: {FRACTALRABBIT_PINNED_COMMIT}
              execution: external
            observation:
              spatial_bins: 3
              time_bins_per_day: 4
            model:
              smoothing: 0.05
            closed_loop:
              iterations: 1
              candidates: []
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fixture"):
        run_sparsity_analysis(external)


def test_large_fixture_contract() -> None:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "ID,Days,x(km),y(km)"
    assert len(lines) == 2401
    travellers = {line.split(",")[0] for line in lines[1:]}
    assert len(travellers) == 5
    days = [float(line.split(",")[1]) for line in lines[1:]]
    assert max(days) <= 36.0
    assert max(days) > 24.0


def test_sparsity_levels_emit_actual_fraction_and_calibration(tmp_path) -> None:
    """actual_fraction is the honest sparsity level; Brier keys follow H2 per level."""

    summary_path = run_sparsity_analysis(_config(tmp_path, seed=11))
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    levels = sorted(payload["levels"], key=lambda item: item["fraction"], reverse=True)
    for level in levels:
        assert 0.0 < level["actual_fraction"] <= 1.0
        # 480 reports/traveller divide evenly, so requested == actual on this fixture.
        assert level["actual_fraction"] == pytest.approx(level["fraction"])
        assert 0.0 <= level["full_brier"] <= 2.0
        assert 0.0 <= level["latent_brier"] <= 2.0
        # H2 at every density: observing location calibrates strictly better.
        assert level["full_brier"] < level["latent_brier"]
        assert level["full_brier_skill_score"] > level["latent_brier_skill_score"]
    assert "actual_fraction" in payload["caveat"]
