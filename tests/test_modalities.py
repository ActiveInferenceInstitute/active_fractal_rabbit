from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from active_fractal_rabbit.config import FRACTALRABBIT_PINNED_COMMIT, load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.inference import run_latent_inference
from active_fractal_rabbit.observations import discretize_observations
from active_fractal_rabbit.pipeline import run_pipeline_from_path
from active_fractal_rabbit.pomdp import build_pomdp

FIXTURE = Path("data/fixtures/fractalrabbit_sample.csv").resolve()


def _sequence():
    config = load_config("configs/default.yaml")
    return discretize_observations(parse_waypoint_csv(FIXTURE), config.inference)


def _modalities_config(tmp_path: Path, modalities: str) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    path = config_dir / "modalities.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            run_id: modalities_test
            fractalrabbit:
              upstream_commit: {FRACTALRABBIT_PINNED_COMMIT}
              execution: fixture
              fixture_csv: {FIXTURE}
            observation:
              spatial_bins: 3
              time_bins_per_day: 4
            model:
              smoothing: 0.05
              modalities: {modalities}
            closed_loop:
              iterations: 1
              candidates: []
            """
        ),
        encoding="utf-8",
    )
    return path


def test_dropping_location_modality_raises_posterior_uncertainty() -> None:
    """Deep-latent flip test: without direct location evidence the filter must be less certain."""

    sequence = _sequence()
    full = run_latent_inference(build_pomdp(sequence, 0.05), sequence)
    latent = run_latent_inference(
        build_pomdp(
            sequence, 0.05, modalities=("time_of_day", "speed", "report_burst")
        ),
        sequence,
    )

    assert (
        latent.metrics["mean_posterior_entropy"]
        > 2 * full.metrics["mean_posterior_entropy"]
    )


def test_modalities_knob_flows_through_pipeline(tmp_path) -> None:
    result = run_pipeline_from_path(
        _modalities_config(tmp_path, "[time_of_day, speed, report_burst]")
    )

    assert result.metadata["modalities"] == ("time_of_day", "speed", "report_burst")
    assert result.metadata["observation_modalities"] == (
        "time_of_day",
        "speed",
        "report_burst",
    )
    assert result.metrics["mean_posterior_entropy"] > 0.5


def test_modalities_canonical_order_is_enforced_regardless_of_config_order(
    tmp_path,
) -> None:
    result = run_pipeline_from_path(_modalities_config(tmp_path, "[speed, location]"))
    assert result.metadata["observation_modalities"] == ("location", "speed")


def test_build_pomdp_rejects_unknown_or_empty_modalities() -> None:
    sequence = _sequence()
    with pytest.raises(ValueError, match="unknown observation modalities"):
        build_pomdp(sequence, 0.05, modalities=("location", "weather"))
    with pytest.raises(ValueError, match="at least one"):
        build_pomdp(sequence, 0.05, modalities=())


def test_config_validation_rejects_bad_modalities_and_engine(tmp_path) -> None:
    with pytest.raises(ValueError, match="modalities entries"):
        load_config(_modalities_config(tmp_path, "[location, weather]"))

    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    bad_engine = config_dir / "bad_engine.yaml"
    bad_engine.write_text(
        textwrap.dedent(
            f"""
            run_id: bad_engine
            fractalrabbit:
              upstream_commit: {FRACTALRABBIT_PINNED_COMMIT}
              execution: fixture
              fixture_csv: {FIXTURE}
            observation:
              spatial_bins: 3
              time_bins_per_day: 4
            model:
              smoothing: 0.05
              engine: quantum
            closed_loop:
              iterations: 1
              candidates: []
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="engine"):
        load_config(bad_engine)

    duplicate = _modalities_config(tmp_path, "[speed, speed]")
    with pytest.raises(ValueError, match="duplicates"):
        load_config(duplicate)
