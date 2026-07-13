from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.spatial_resolution_frontier import (
    SPATIAL_RESOLUTION_CONFIRMATION_SCHEMA,
    SPATIAL_RESOLUTION_FRONTIER_SCHEMA,
    render_spatial_resolution_frontier,
    run_and_write_larger_sample_confirmation,
    run_and_write_spatial_resolution_frontier,
    run_spatial_resolution_frontier,
)


def test_spatial_resolution_frontier_selects_4x4_tradeoff(tmp_path) -> None:
    config = load_config("configs/default.yaml")
    rows = parse_waypoint_csv(config.fractalrabbit.fixture_csv)

    payload = run_spatial_resolution_frontier(
        config,
        rows,
        candidate_bins=(3, 4, 5, 6, 8),
        pymdp_validation_steps=8,
    )

    assert payload["schema"] == SPATIAL_RESOLUTION_FRONTIER_SCHEMA
    assert payload["verdict"] == "default_resolution_not_best_feasible_loss"
    assert [row["spatial_bins"] for row in payload["rows"]] == [3, 4, 5, 6, 8]
    assert payload["default_row"]["spatial_bins"] == 4
    assert payload["default_row"]["state_count"] == 16
    assert payload["default_row"]["occupied_state_count"] == 16
    assert payload["default_row"]["empty_state_count"] == 0
    assert payload["default_row"]["min_rows_per_occupied_state"] >= 3
    assert payload["best_feasible_row"]["spatial_bins"] == 3
    assert payload["default_minus_coarse"]["full_location_mean_predictive_loss"] > 0.0
    assert payload["default_minus_coarse"]["no_location_mean_mass_on_true_cell"] < 0.0
    assert payload["default_row"]["pymdp_validation_rows"] == 8.0
    assert payload["default_row"]["pymdp_exact_inference_max_abs_delta"] < 1e-4

    out = tmp_path / "spatial_resolution_frontier.json"
    run_and_write_spatial_resolution_frontier(
        "configs/default.yaml",
        out,
        candidate_bins=(3, 4),
        pymdp_validation_steps=2,
    )
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["default_row"]["state_count"] == 16
    assert written["n_observations"] == len(rows)

    confirmation_out = tmp_path / "spatial_resolution_larger_sample_confirmation.json"
    confirmation_observations = tmp_path / "larger_sample" / "observations.csv"
    run_and_write_larger_sample_confirmation(
        "configs/default.yaml",
        out,
        confirmation_out,
        confirmation_observations,
        sample_multipliers=(2, 4),
        candidate_bins=(3, 4),
        pymdp_validation_steps=2,
    )
    confirmation = json.loads(confirmation_out.read_text(encoding="utf-8"))
    assert confirmation["schema"] == SPATIAL_RESOLUTION_CONFIRMATION_SCHEMA
    assert confirmation["sample_multiplier"] == 4
    assert confirmation["sample_multipliers"] == [2, 4]
    assert confirmation["larger_sample"]["n_observations"] == 9600
    assert confirmation["larger_sample"]["reports_per_traveller"] == 1920
    assert confirmation["larger_sample"]["strict_feasible_candidate_count"] >= 0
    assert confirmation["comparison"]["status"] in {"confirmed", "changed"}
    assert "larger_best_feasible_is_strict" in confirmation["comparison"]
    assert len(confirmation["sample_size_ladder"]) == 2
    assert confirmation["sample_size_ladder_summary"]["multipliers"] == [2, 4]
    assert (
        confirmation["sample_size_ladder_summary"]["largest_n_observations"] == 9600
    )
    assert confirmation_observations.read_text(encoding="utf-8").count("\n") == 9601
    assert (tmp_path / "larger_sample" / "2x_observations.csv").is_file()
    assert (tmp_path / "larger_sample" / "4x_observations.csv").is_file()
    updated = json.loads(out.read_text(encoding="utf-8"))
    assert updated["larger_sample_confirmation"]["sample_multiplier"] == 4
    assert updated["larger_sample_confirmation"]["n_observations"] == 9600
    assert updated["larger_sample_confirmation"]["sample_multipliers"] == [2, 4]

    fig = tmp_path / "spatial_resolution_frontier.png"
    render_spatial_resolution_frontier(updated, fig)
    image = mpimg.imread(fig)
    assert image.shape[0] >= 300
    assert image.shape[1] >= 400
    assert float(np.ptp(image)) > 0.01


def test_no_obsolete_default_resolution_claims() -> None:
    stale_patterns = [
        re.compile(r"default\s+3x3", re.IGNORECASE),
        re.compile(r"3x3\s+default", re.IGNORECASE),
        re.compile(r"\b9[- ]states?\b", re.IGNORECASE),
        re.compile(r"1/9\s+for\s+the\s+default", re.IGNORECASE),
    ]
    paths = [
        *Path("manuscript").glob("*.md"),
        Path("docs/claim_ledger.md"),
        Path("README.md"),
        Path("TODO.md"),
        Path("tasks.yaml"),
    ]

    offenders: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for pattern in stale_patterns:
            if pattern.search(text):
                offenders.append(f"{path}:{pattern.pattern}")
    assert offenders == []
