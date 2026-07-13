from __future__ import annotations

import numpy as np
import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.observations import discretize_observations
from active_fractal_rabbit.anticipatory import _brier_on_all
from active_fractal_rabbit.inference import run_latent_inference
from active_fractal_rabbit.pomdp import build_pomdp
from active_fractal_rabbit.return_structure import (
    conditional_entropy_reduction,
    return_conditioned_brier,
    return_conditioned_transition,
    run_return_structure_analysis,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_conditional_entropy_reduction_hand_example() -> None:
    previous = np.array([0, 0, 1, 1])
    visited = np.array([0, 1, 0, 1])
    next_cell = np.array([0, 1, 1, 1])
    reduction = conditional_entropy_reduction(previous, visited, next_cell)

    assert reduction == pytest.approx(0.34657359027997264)


def test_return_conditioned_transition_prefers_visited_cells() -> None:
    cells = np.array([0, 1, 0, 2, 0, 1])
    first = np.array([True, False, False, False, False, False])
    tensor = return_conditioned_transition(cells, first, n_states=3, smoothing=0.01)

    # From previous cell 2, the next observed return is to a visited cell 0.
    assert tensor[0, 2, 1] > tensor[1, 2, 1]
    assert tensor[:, 2, 1].sum() == pytest.approx(1.0)


def test_return_structure_analysis_reports_kill_gate_and_headroom() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    sequence = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    result = run_return_structure_analysis(
        sequence, smoothing=config.inference.smoothing
    )

    assert result.n_move_steps > 0
    assert result.move_step_entropy_reduction >= 0.0
    assert result.persistence_headroom >= 0.0
    assert result.verdict in {"promote_return_conditioned_prior", "honest_null"}


def test_return_conditioned_brier_scores_the_prior_not_the_visited_cell() -> None:
    # The return-conditioned predictor cannot be perfect: it never observes the
    # destination cell, only a visited/not-visited indicator per candidate.
    cells = np.array([0, 1, 0, 2, 0, 1, 2, 0])
    first = np.array([True, False, False, False, False, False, False, False])
    tensor = return_conditioned_transition(cells, first, n_states=3, smoothing=0.01)
    brier = return_conditioned_brier(cells, first, n_states=3, tensor=tensor)

    # A genuine renormalized predictive prior lands well above the ~0 Brier that
    # a location-observed posterior would achieve, and never hits a perfect 0.
    assert brier > 0.05
    assert brier <= 2.0


def test_return_structure_metric_is_not_the_location_observed_posterior() -> None:
    """Regression: the 'return skill' must not be the full-model posterior score.

    The vacuous prior form scored ``_brier_on_all`` of the location-observed
    POMDP posterior, giving a near-perfect Brier (~5e-5) and skill ~1.0 by
    construction. The honest metric scores the return-conditioned prior itself,
    which is materially worse than the observed-location posterior. This test
    fails the moment the vacuous form is reintroduced.
    """

    config = load_config(ROOT / "configs" / "default.yaml")
    sequence = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    cells = np.asarray(sequence.spatial_states, dtype=int)

    # The value the vacuous implementation reported (location-observed posterior).
    model = build_pomdp(sequence, config.inference.smoothing)
    posterior = run_latent_inference(model, sequence).posterior_states
    location_observed_brier = _brier_on_all(posterior, cells)
    assert location_observed_brier < 1e-3  # near-perfect by construction

    result = run_return_structure_analysis(
        sequence, smoothing=config.inference.smoothing
    )

    # The published metric must NOT be the near-perfect observed-location score.
    assert result.return_conditioned_brier > 100.0 * location_observed_brier
    assert result.return_conditioned_brier > 0.05
    # Skill of exactly-1.0-by-construction is impossible for a real prior here.
    assert result.return_skill_vs_persistence < 0.99
    # It must equal an independent recomputation from the return-conditioned tensor.
    tensor = return_conditioned_transition(
        cells,
        np.asarray(sequence.is_first_observation, dtype=bool),
        sequence.num_spatial_states,
        config.inference.smoothing,
    )
    expected = return_conditioned_brier(
        cells,
        np.asarray(sequence.is_first_observation, dtype=bool),
        sequence.num_spatial_states,
        tensor,
    )
    assert result.return_conditioned_brier == pytest.approx(expected)
