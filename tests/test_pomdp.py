from __future__ import annotations

import numpy as np
import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.observations import discretize_observations
from active_fractal_rabbit.pomdp import (
    _transition_tensor,
    build_pomdp,
    tensor_summary,
    to_pymdp_payload,
)


def test_transition_tensor_excludes_cross_traveller_boundaries() -> None:
    """A (last-cell-of-A, first-cell-of-B) pair must NOT be counted as a transition.

    Two travellers: A stays in cell 0 for three reports, B stays in cell 1 for
    three reports. The only consecutive pair that crosses travellers is idx 2->3
    (cell 0 -> cell 1). Without the boundary guard that pair leaks 1/3 of state
    0's outgoing mass into state 1; with it, state 0 stays in state 0.
    """

    hidden = np.array([0, 0, 0, 1, 1, 1])
    boundaries = np.array(
        [True, False, False, True, False, False]
    )  # idx 3 = new traveller
    controls = ("empirical_continuation",)

    # Default (no boundary guard) reproduces the spurious 1/3 boundary mass.
    no_guard = _transition_tensor(hidden, 2, 1e-6, controls)
    assert no_guard[1, 0, 0] == pytest.approx(1 / 3, abs=1e-3)

    # With the guard, the cross-traveller pair is dropped: cell 0 -> cell 0 only.
    guarded = _transition_tensor(
        hidden, 2, 1e-6, controls, first_observation=boundaries
    )
    assert guarded[1, 0, 0] < 1e-3
    assert guarded[0, 0, 0] > 0.999


def test_build_pomdp_drops_boundary_transition_counts() -> None:
    """build_pomdp wires is_first_observation, so B differs from the unguarded count."""

    config = load_config("configs/default.yaml")
    observations = parse_waypoint_csv("data/fixtures/fractalrabbit_sample.csv")
    sequence = discretize_observations(observations, config.inference)
    assert int(sequence.is_first_observation.sum()) >= 2  # the fixture spans travellers

    guarded = build_pomdp(sequence, config.inference.smoothing)
    # Same data, same smoothing, but boundaries counted (old behavior).
    unguarded_B = _transition_tensor(
        sequence.spatial_states,
        sequence.num_spatial_states,
        config.inference.smoothing,
        ("empirical_continuation", "high_entropy_exploration"),
    )
    # The boundary guard must actually change the learned continuation dynamics.
    assert not np.allclose(guarded.B[:, :, 0], unguarded_B[:, :, 0])
    # And both remain valid column-stochastic transition matrices.
    assert np.allclose(guarded.B[:, :, 0].sum(axis=0), 1.0)


def _model():
    config = load_config("configs/default.yaml")
    observations = parse_waypoint_csv("data/fixtures/fractalrabbit_sample.csv")
    sequence = discretize_observations(observations, config.inference)
    return build_pomdp(sequence, config.inference.smoothing)


def test_pomdp_arrays_are_normalized() -> None:
    model = _model()

    for A in model.A:
        assert np.allclose(A.sum(axis=0), 1.0)
    for control in range(model.B.shape[2]):
        assert np.allclose(model.B[:, :, control].sum(axis=0), 1.0)
    assert np.isclose(model.D.sum(), 1.0)
    assert model.B.shape[2] == 2
    assert model.control_names == ("empirical_continuation", "high_entropy_exploration")


def test_pymdp_payload_shapes() -> None:
    model = _model()
    payload = to_pymdp_payload(model)

    assert len(payload["A"]) == 4
    assert payload["B"].shape == model.B.shape
    assert payload["D"].shape == model.D.shape


def test_tensor_summary_matches_model_shapes() -> None:
    model = _model()
    summary = tensor_summary(model)

    assert summary["A_shapes"] == [list(matrix.shape) for matrix in model.A]
    assert summary["B_shape"] == list(model.B.shape)
    assert summary["transition_controls"] == list(model.control_names)
    assert np.isclose(summary["D_sum"], 1.0)
    assert np.allclose(summary["A_column_sum_min"], 1.0)


def test_configurable_identity_control_changes_transition_axis() -> None:
    config = load_config("configs/sweep.yaml")
    observations = parse_waypoint_csv("data/fixtures/fractalrabbit_sample.csv")
    sequence = discretize_observations(observations, config.inference)
    model = build_pomdp(
        sequence, config.inference.smoothing, config.inference.model.controls
    )

    assert model.B.shape[2] == 3
