from __future__ import annotations

from pathlib import Path

import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.neural_baseline import run_and_write_neural_baseline
from active_fractal_rabbit.observation_sources import resolve_observations_path
from active_fractal_rabbit.poisoning_robustness import (
    run_and_write_poisoning_robustness,
)


def test_explicit_missing_observation_path_fails_loudly(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError,
        match="requested neural-baseline observations file does not exist",
    ):
        run_and_write_neural_baseline(
            "configs/default.yaml", tmp_path / "neural.json", observations_path=missing
        )

    with pytest.raises(
        FileNotFoundError,
        match="requested poisoning-robustness observations file does not exist",
    ):
        run_and_write_poisoning_robustness(
            "configs/default.yaml", missing, tmp_path / "poison.json"
        )


def test_configured_fixture_resolution_still_defaults_to_fixture() -> None:
    config = load_config("configs/default.yaml")
    path = resolve_observations_path(config)

    assert path.name == "fractalrabbit_sample_large.csv"
    assert path.is_file()
