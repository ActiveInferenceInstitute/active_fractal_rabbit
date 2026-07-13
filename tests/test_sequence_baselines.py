from __future__ import annotations

import json

import numpy as np
import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.observations import discretize_observations
from active_fractal_rabbit.sequence_baselines import (
    run_and_write_sequence_baselines,
    run_sequence_baseline_analysis,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_sequence_baseline_ladder_reports_best_model() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    sequence = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    payload = run_sequence_baseline_analysis(
        sequence,
        smoothing=config.inference.smoothing,
        reference_loss=3.0,
        pymdp_loss=4.0,
    )

    assert payload["schema"] == "active-fractal-rabbit-sequence-baseline-v1"
    assert payload["status"] == "completed"
    assert payload["n_scored_steps"] == len(sequence.observations) - len(
        {row.traveller_id for row in sequence.observations}
    )
    assert payload["best_model"] in {row["model"] for row in payload["rows"]}
    assert payload["pymdp_performance_gate"] in {
        "stronger_sequence_baseline_beats_pymdp_on_this_run",
        "pymdp_beats_sequence_baselines_on_this_run",
    }


def test_persistence_loss_is_hand_computable() -> None:
    from active_fractal_rabbit.sequence_baselines import _persistence_loss

    cells = np.array([0, 0, 1])
    first = np.array([True, False, False])
    # With smoothing 1 and two states, persistence assigns 2/3 to previous
    # cell and 1/3 to the other cell.
    expected = (-np.log(2 / 3) - np.log(1 / 3)) / 2

    assert _persistence_loss(cells, first, n_states=2, smoothing=1.0) == pytest.approx(
        expected
    )


def test_sequence_baseline_writer_emits_json(tmp_path) -> None:
    out = tmp_path / "sequence_baseline_analysis.json"
    run_and_write_sequence_baselines(ROOT / "configs" / "default.yaml", out)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["schema"] == "active-fractal-rabbit-sequence-baseline-v1"
    assert len(payload["rows"]) >= 6
    assert payload["best_mean_predictive_loss"] > 0
