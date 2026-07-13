from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.pipeline import run_pipeline_from_path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONFIGS = sorted((ROOT / "configs" / "fixtures").glob("*.yaml"))


def test_fixture_config_roster_is_complete() -> None:
    names = {path.name for path in FIXTURE_CONFIGS}
    assert names == {
        "fixed_bounds_grid.yaml",
        "quantile_bins.yaml",
        "identity_ablation.yaml",
        "fixed_burst.yaml",
        "deep_latent.yaml",
    }


@pytest.mark.parametrize("config_path", FIXTURE_CONFIGS, ids=lambda path: path.stem)
def test_fixture_configs_load_and_run(config_path: Path, repo_sandbox: Path) -> None:
    # Run against the sandbox COPY of the config so the pipeline's project
    # root (and therefore output/runs/fixture_*) anchors in the sandbox, not
    # in the production output tree. Canonical fixture_* runs are owned by
    # scripts/07_run_full_study.py.
    config_path = repo_sandbox / "configs" / "fixtures" / config_path.name
    config = load_config(config_path)
    result = run_pipeline_from_path(config_path)

    assert result.metrics["n_observations"] == 2400.0
    snapshot = json.loads(
        (result.output_dir / "config_snapshot.json").read_text(encoding="utf-8")
    )
    observation = snapshot["inference"]["observation"]
    model = snapshot["inference"]["model"]

    if config_path.stem == "fixed_bounds_grid":
        assert observation["spatial_discretizer"] == "fixed_bounds"
        assert observation["spatial_bounds"] == [0.0, 100.0, 0.0, 50.0]
    if config_path.stem == "quantile_bins":
        assert observation["spatial_discretizer"] == "quantile"
        # quantile bins must spread the fixture across more than one occupied state
        with (result.output_dir / "discrete_features.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            states = {row["spatial_state"] for row in csv.DictReader(handle)}
        assert len(states) > 1
    if config_path.stem == "identity_ablation":
        assert model["controls"] == [
            "empirical_continuation",
            "high_entropy_exploration",
            "identity",
        ]
        with (result.output_dir / "policy_trace.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            header = handle.readline()
        assert "proxy_score_identity" in header
    if config_path.stem == "fixed_burst":
        assert observation["burst_threshold_rule"] == "fixed_days"
        assert observation["burst_fixed_days"] == 0.5
    if config_path.stem == "deep_latent":
        assert model["modalities"] == ["time_of_day", "speed", "report_burst"]
        assert result.metadata["observation_modalities"] == (
            "time_of_day",
            "speed",
            "report_burst",
        )
        assert result.metrics["mean_posterior_entropy"] > 0.5
    assert config.fractalrabbit.execution == "fixture"
