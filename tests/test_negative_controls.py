from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from active_fractal_rabbit.config import FRACTALRABBIT_PINNED_COMMIT
from active_fractal_rabbit.negative_controls import (
    NEGATIVE_CONTROL_SCHEMA,
    run_negative_controls,
)

FIXTURE = Path("data/fixtures/fractalrabbit_sample.csv").resolve()


def _config(tmp_path: Path, seed: int = 7, execution: str = "fixture") -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    path = config_dir / f"negative_seed_{seed}.yaml"
    fixture_line = f"fixture_csv: {FIXTURE}" if execution == "fixture" else ""
    path.write_text(
        textwrap.dedent(
            f"""
            run_id: negative_seed_{seed}
            seed: {seed}
            fractalrabbit:
              upstream_commit: {FRACTALRABBIT_PINNED_COMMIT}
              execution: {execution}
              {fixture_line}
              params:
                number_of_travellers: 4
                number_of_days: 3.0
                number_of_reports: 6.0
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


def test_negative_controls_summary_schema_and_directional_probes(tmp_path) -> None:
    summary_path = run_negative_controls(_config(tmp_path, seed=7))
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert payload["schema"] == NEGATIVE_CONTROL_SCHEMA
    assert payload["lane"] == "fixture"
    assert payload["seed"] == 7
    assert payload["n_controls"] == 4
    names = [item["name"] for item in payload["controls"]]
    assert names == [
        "shuffled_time",
        "shuffled_ids",
        "degenerate_bounds",
        "misspecified_speed",
    ]
    for item in payload["controls"]:
        assert item["hypothesis"]
        assert item["probe"]
        assert isinstance(item["passed"], bool)
    degenerate = next(
        item for item in payload["controls"] if item["name"] == "degenerate_bounds"
    )
    assert degenerate["occupied_spatial_states"] == 1
    misspecified = next(
        item for item in payload["controls"] if item["name"] == "misspecified_speed"
    )
    assert misspecified["occupied_speed_categories"] == 1
    assert payload["all_passed"] is True
    assert "not empirical mobility evidence" in payload["caveat"]


def test_negative_controls_run_artifacts_exist(tmp_path) -> None:
    config_path = _config(tmp_path, seed=7)
    run_negative_controls(config_path)
    for run_id in [
        "negative_intact",
        "negative_shuffled_time",
        "negative_shuffled_ids",
        "negative_degenerate_bounds",
        "negative_misspecified_speed",
    ]:
        assert (tmp_path / "output" / "runs" / run_id / "metrics.json").exists()
    assert (tmp_path / "output" / "negative_controls" / "shuffled_time.csv").exists()
    assert (tmp_path / "output" / "negative_controls" / "shuffled_ids.csv").exists()


def test_negative_controls_consume_seed_flip(tmp_path) -> None:
    """Flip test: the root seed must change the shuffled mutations."""

    run_negative_controls(_config(tmp_path, seed=7))
    first_time = (
        tmp_path / "output" / "negative_controls" / "shuffled_time.csv"
    ).read_text(encoding="utf-8")
    first_ids = (
        tmp_path / "output" / "negative_controls" / "shuffled_ids.csv"
    ).read_text(encoding="utf-8")

    run_negative_controls(_config(tmp_path, seed=8))
    second_time = (
        tmp_path / "output" / "negative_controls" / "shuffled_time.csv"
    ).read_text(encoding="utf-8")
    second_ids = (
        tmp_path / "output" / "negative_controls" / "shuffled_ids.csv"
    ).read_text(encoding="utf-8")

    assert first_time != second_time
    assert first_ids != second_ids


def test_negative_controls_are_deterministic_for_a_seed(tmp_path) -> None:
    config_path = _config(tmp_path, seed=7)
    first = json.loads(run_negative_controls(config_path).read_text(encoding="utf-8"))
    second = json.loads(run_negative_controls(config_path).read_text(encoding="utf-8"))
    assert first == second


def test_negative_controls_reject_external_configs(tmp_path) -> None:
    config_path = _config(tmp_path, seed=7, execution="external")
    with pytest.raises(ValueError, match="fixture-lane diagnostic"):
        run_negative_controls(config_path)


def test_negative_controls_mutated_csvs_stay_parseable(tmp_path) -> None:
    from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv

    run_negative_controls(_config(tmp_path, seed=7))
    intact_rows = parse_waypoint_csv(FIXTURE)
    for name in ["shuffled_time.csv", "shuffled_ids.csv"]:
        rows = parse_waypoint_csv(tmp_path / "output" / "negative_controls" / name)
        assert len(rows) == len(intact_rows)
        assert {row.traveller_id for row in rows} == {
            row.traveller_id for row in intact_rows
        }
