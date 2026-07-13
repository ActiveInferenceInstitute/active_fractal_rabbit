from __future__ import annotations

import numpy as np
import pytest

from active_fractal_rabbit.circadian_phase import (
    jensen_shannon_divergence,
    phase_cell_distributions,
    phase_predictive_gain,
)


def test_jensen_shannon_divergence_is_zero_for_equal_distributions() -> None:
    assert jensen_shannon_divergence(
        np.array([0.5, 0.5]), np.array([0.5, 0.5])
    ) == pytest.approx(0.0)


def test_phase_conditioned_distribution_is_hand_computable() -> None:
    cells = np.array([0, 0, 1, 1])
    phases = np.array([0, 0, 1, 1])
    pooled, phase_probs = phase_cell_distributions(
        cells, phases, n_states=2, n_phases=2, smoothing=0.0
    )

    assert pooled.tolist() == [0.5, 0.5]
    assert phase_probs.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_phase_predictive_gain_beats_pooled_when_phase_names_cell() -> None:
    cells = np.array([0, 0, 1, 1])
    phases = np.array([0, 0, 1, 1])
    metrics = phase_predictive_gain(
        cells, phases, n_states=2, n_phases=2, smoothing=0.1
    )

    assert metrics["phase_predictive_loss"] < metrics["pooled_predictive_loss"]
    assert metrics["phase_gain_vs_pooled"] > 0
    assert metrics["max_phase_js"] > 0
