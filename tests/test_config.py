from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from active_fractal_rabbit.config import (
    FRACTALRABBIT_PINNED_COMMIT,
    load_config,
    parameter_rows,
)


def test_load_default_config() -> None:
    config = load_config("configs/default.yaml")

    assert config.run_id == "default"
    assert config.fractalrabbit.upstream_commit == FRACTALRABBIT_PINNED_COMMIT
    assert config.fractalrabbit.execution == "fixture"
    assert config.inference.spatial_bins == 4
    assert config.inference.observation.spatial_discretizer == "grid"
    assert config.inference.model.controls == (
        "empirical_continuation",
        "high_entropy_exploration",
    )
    assert config.inference.model.factorization == "single"
    assert config.inference.model.hidden_factors == (
        "cell",
        "movement_mode",
        "cognitive_proxy_mode",
    )
    assert config.inference.model.factorial_controls == (
        "mode_persistence",
        "mode_reset",
    )
    assert config.experiment.objective_direction == "minimize"
    assert "posterior_entropy" in config.figures.enabled
    assert "spatial_resolution_frontier" in config.figures.enabled
    assert "factorial_pymdp_modes" in config.figures.enabled
    # candidates bracket the declared mean report count (240.0 in the 10x default)
    assert config.closed_loop.candidates[0].params.number_of_reports == 320.0
    assert (
        config.closed_loop.candidates[0].params.number_of_reports
        < config.fractalrabbit.params.number_of_reports
        < config.closed_loop.candidates[1].params.number_of_reports
    )
    assert config.closed_loop.candidates[1].params.number_of_reports == 960.0
    assert config.robustness.external_realization_fractal_dimensions == (
        1.21,
        1.45,
        1.7,
    )


def test_config_schema_doc_mentions_groups_and_artifacts() -> None:
    schema = Path("docs/config_schema.md").read_text(encoding="utf-8")

    for group in [
        "fractalrabbit",
        "observation",
        "model",
        "experiment",
        "figures",
        "closed_loop",
        "robustness",
    ]:
        assert f"`{group}`" in schema
    for artifact in [
        "config_snapshot.json",
        "provenance_manifest.json",
        "style_provenance.json",
    ]:
        assert artifact in schema


def test_parameter_csv_order_matches_fractalrabbit() -> None:
    params = load_config("configs/default.yaml").fractalrabbit.params

    assert [label for label, _ in parameter_rows(params)] == [
        "space dimension",
        "fractal dimension",
        "number of points",
        "number of travellers",
        "number of co-travellers",
        "number of days",
        "number of reports",
    ]


def test_rejects_unpinned_fractalrabbit_commit(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "bad.yaml"
    config.write_text(
        textwrap.dedent(
            """
            run_id: bad
            fractalrabbit:
              upstream_commit: not-the-pin
              execution: fixture
              fixture_csv: fixture.csv
              params: {}
            observation:
              spatial_bins: 2
              time_bins_per_day: 2
            model:
              smoothing: 0.1
            closed_loop:
              iterations: 1
              candidates: []
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must stay pinned"):
        load_config(config)


def test_accepts_factorial_model_config(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    fixture = Path("data/fixtures/fractalrabbit_sample.csv").resolve()
    config = config_dir / "factorial.yaml"
    config.write_text(
        textwrap.dedent(
            f"""
            run_id: factorial
            fractalrabbit:
              upstream_commit: {FRACTALRABBIT_PINNED_COMMIT}
              execution: fixture
              fixture_csv: {fixture}
            observation:
              spatial_bins: 2
              time_bins_per_day: 2
            model:
              smoothing: 0.1
              factorization: factorial
              hidden_factors: [cell, movement_mode, cognitive_proxy_mode]
              factorial_controls: [mode_persistence, mode_reset]
              preference_prior: diagnostic_uncertainty
            closed_loop:
              iterations: 1
              candidates: []
            """
        ),
        encoding="utf-8",
    )

    loaded = load_config(config)

    assert loaded.inference.model.factorization == "factorial"
    assert loaded.inference.model.preference_prior == "diagnostic_uncertainty"
    assert loaded.inference.model.factorial_controls == (
        "mode_persistence",
        "mode_reset",
    )
