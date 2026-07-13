"""Pattern-of-life structure recovery: occupancy + preferred-place set.

Pins the robust facts (recovers structure above uniform; recovers most of the
preferred set; full-location is the sanity ceiling) and the hand-computed
metrics. The dynamics-only comparison is deliberately NOT asserted as a universal
direction — it is regime-dependent (the genuine finding), so binding it to one
direction would be a brittle, dishonest test.
"""

from __future__ import annotations

import numpy as np
import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.observations import discretize_observations
from active_fractal_rabbit.structure_recovery import (
    cross_generator_structure_robustness,
    preferred_set_jaccard,
    run_structure_recovery,
    total_variation,
    true_occupancy,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_total_variation_hand_example() -> None:
    assert total_variation(np.array([0.5, 0.5]), np.array([1.0, 0.0])) == pytest.approx(
        0.5
    )
    assert total_variation(np.array([0.2, 0.8]), np.array([0.2, 0.8])) == pytest.approx(
        0.0
    )


def test_true_occupancy_normalized_counts() -> None:
    occ = true_occupancy(np.array([0, 0, 1, 2]), 3)
    assert list(occ) == pytest.approx([0.5, 0.25, 0.25])


def test_preferred_set_jaccard_hand_example() -> None:
    # top-2 of [0.5,0.3,0.2,0.0] = {0,1}; of [0.4,0.1,0.4,0.1] = {0,2}; overlap {0}, union {0,1,2}.
    inferred = np.array([0.4, 0.1, 0.4, 0.1])
    true = np.array([0.5, 0.3, 0.2, 0.0])
    assert preferred_set_jaccard(inferred, true, 2) == pytest.approx(1 / 3)


def _fixture_sequence():
    config = load_config(ROOT / "configs" / "default.yaml")
    return config, discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )


def test_structure_recovery_beats_uniform_and_recovers_preferred_set() -> None:
    config, sequence = _fixture_sequence()
    r = run_structure_recovery(sequence, smoothing=config.inference.smoothing)
    assert r.deep_latent_beats_uniform  # recovers structure above the uniform null
    assert r.deep_latent_preferred_jaccard >= 0.4  # recovers most of the preferred set
    # Full-location is the sanity ceiling — never worse than the deep-latent lane.
    assert r.full_location_tv <= r.deep_latent_tv + 1e-9


def test_dynamics_only_baseline_is_reported_not_assumed() -> None:
    """The dynamics-only comparison is computed and exposed; its DIRECTION is
    regime-dependent (the honest finding), so we only assert it is a real number,
    not which way it goes."""

    config, sequence = _fixture_sequence()
    r = run_structure_recovery(sequence, smoothing=config.inference.smoothing)
    assert 0.0 <= r.dynamics_only_tv <= 1.0
    assert isinstance(r.deep_latent_beats_dynamics_only, bool)


def test_cross_generator_robustness_paired_fraction_reproducible_and_bounded() -> None:
    """The secondary cross-generator block is a deterministic, additive robustness
    summary: the paired beats-dynamics-only fraction is a probability in [0, 1] and is
    identical across two independent runs on the pinned generator seeds. The paired
    indicator is computed per draw (deep_tv < dyn_tv) before aggregating, so the
    fraction equals the mean of per-draw booleans, not a comparison of marginal means."""

    config = load_config(ROOT / "configs" / "default.yaml")
    first = cross_generator_structure_robustness(
        config.inference, smoothing=config.inference.smoothing
    )
    second = cross_generator_structure_robustness(
        config.inference, smoothing=config.inference.smoothing
    )

    frac = first["beats_dynamics_only_fraction"]
    assert 0.0 <= frac <= 1.0
    assert (
        second["beats_dynamics_only_fraction"] == frac
    )  # deterministic on pinned seeds
    # Fraction is the mean of the paired per-draw indicators, not a marginal-mean compare.
    draws = first["draw_rows"]
    assert len(draws) == first["n_draws"]
    paired_mean = sum(1 for d in draws if d["deep_beats_dynamics_only"]) / len(draws)
    assert frac == pytest.approx(paired_mean)
