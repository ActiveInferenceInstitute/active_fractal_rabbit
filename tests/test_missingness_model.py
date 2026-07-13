from __future__ import annotations

import json

import numpy as np
import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.missingness_model import (
    binary_prequential_loss,
    phase_burst_probabilities,
    run_and_write_missingness_model,
    run_missingness_model_analysis,
)
from active_fractal_rabbit.observations import discretize_observations

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_binary_prequential_loss_matches_hand_example() -> None:
    labels = np.array([1, 1])
    # First prediction is 1/2; second is 2/3 after one positive update.
    expected = (-np.log(0.5) - np.log(2 / 3)) / 2

    assert binary_prequential_loss(labels, smoothing=1.0) == pytest.approx(expected)


def test_phase_burst_probability_is_smoothed() -> None:
    labels = np.array([0, 1, 1, 1])
    phases = np.array([0, 0, 1, 1])
    probs = phase_burst_probabilities(labels, phases, n_phases=2, smoothing=1.0)

    assert probs == pytest.approx([0.5, 0.75])


def test_missingness_analysis_reports_reporting_process_gate() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    sequence = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    payload = run_missingness_model_analysis(
        sequence, smoothing=config.inference.smoothing, seed=0
    )

    assert payload["schema"] == "active-fractal-rabbit-missingness-model-v1"
    assert payload["n_intervals"] == len(sequence.observations) - len(
        {row.traveller_id for row in sequence.observations}
    )
    assert payload["verdict"] in {"candidate_reporting_process_model", "honest_null"}
    assert (
        len(payload["phase_short_gap_probability"])
        == config.inference.time_bins_per_day
    )


def test_missingness_writer_emits_json(tmp_path) -> None:
    out = tmp_path / "missingness_model_analysis.json"
    run_and_write_missingness_model(ROOT / "configs" / "default.yaml", out)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["schema"] == "active-fractal-rabbit-missingness-model-v1"
    assert payload["short_gap_rate"] == pytest.approx(0.5, abs=0.002)
