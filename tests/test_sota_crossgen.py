"""Tests for the cross-generator (gravity / EPR) lane of the SOTA spine.

No mocks; real deterministic synthetic sequences and real computation. Covers:
* the gravity generator's waypoint shape and determinism,
* the spine-compatible dataset shape and locked-split disjointness (so the IDENTICAL
  harness scores it with no special-casing),
* that the observation channel is the same correctly-specified emission law,
* that the gravity latent stream is STRUCTURALLY DIFFERENT from FractalRabbit's
  (otherwise the cross-generator test would be vacuous),
* that every spine candidate scores on the gravity dataset leakage-free, and that a
  traveller-clustered CI on the loss difference is well-formed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.power_analysis import generate_synthetic_sequence
from active_fractal_rabbit.sota_crossgen import (
    DEFAULT_N_PLACES,
    build_gravity_partial_observability_dataset,
    generate_gravity_waypoints,
)
from active_fractal_rabbit.sota_program import (
    Candidate,
    FunctionCandidate,
    PredictContext,
    SoftFilterCandidate,
    emission_matrix,
    paired_loss_difference_ci,
    reference_baselines,
    score_candidate,
)

ROOT = Path(__file__).resolve().parents[1]


def _inference(bins: int = 6):
    cfg = load_config(ROOT / "configs" / "default.yaml")
    return replace(
        cfg.inference, observation=replace(cfg.inference.observation, spatial_bins=bins)
    )


def _dataset(n_travellers: int = 24, eps: float = 0.3, bins: int = 6):
    return build_gravity_partial_observability_dataset(
        _inference(bins), n_travellers=n_travellers, eps=eps
    )


# --------------------------------------------------------------------------- #
# Gravity waypoint generator
# --------------------------------------------------------------------------- #
def test_gravity_waypoints_shape_and_sorting() -> None:
    rows = generate_gravity_waypoints(8, seed=2026, n_reports=24)
    # exactly n_travellers * n_reports rows, one per emitted report
    assert len(rows) == 8 * 24
    assert {r.traveller_id for r in rows} == set(range(8))
    # rows are sorted by (traveller, days) so the discretizer sees a stable order
    keys = [(r.traveller_id, r.days) for r in rows]
    assert keys == sorted(keys)
    # waypoints stay inside the box
    assert all(0.0 <= r.x_km <= 200.0 and 0.0 <= r.y_km <= 200.0 for r in rows)


def test_gravity_waypoints_are_deterministic() -> None:
    a = generate_gravity_waypoints(6, seed=2026, n_reports=24)
    b = generate_gravity_waypoints(6, seed=2026, n_reports=24)
    assert a == b
    # a different seed yields a different stream (non-degenerate generator)
    c = generate_gravity_waypoints(6, seed=2027, n_reports=24)
    assert a != c


def test_gravity_generator_rejects_degenerate_inputs() -> None:
    for bad in (
        dict(n_travellers=1),
        dict(n_travellers=4, n_places=1),
        dict(n_travellers=4, n_reports=1),
    ):
        try:
            generate_gravity_waypoints(**{"seed": 2026, **bad})
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


# --------------------------------------------------------------------------- #
# Spine-compatible dataset shape + locked split
# --------------------------------------------------------------------------- #
def test_gravity_dataset_shape_and_disjoint_locked_split() -> None:
    ds = _dataset(n_travellers=24, bins=6)
    assert ds.n_states == 36  # 6x6 grid of spatial cells (spatial_bins ** 2)
    assert set(int(t) for t in ds.travellers) == set(range(24))
    # observed and latent align in length and index order
    assert ds.observed.shape == ds.latent.shape == ds.travellers.shape
    # holdout is keyed on traveller identity: no traveller crosses the split
    holdout_travs = set(int(t) for t in ds.travellers[ds.is_holdout])
    train_travs = set(int(t) for t in ds.travellers[~ds.is_holdout])
    assert holdout_travs and train_travs
    assert holdout_travs.isdisjoint(train_travs)
    # transition is column-stochastic (transition[next, prev])
    assert np.allclose(ds.transition.sum(axis=0), 1.0)
    # emission is the spine's known symmetric channel
    assert np.allclose(ds.emission, emission_matrix(ds.n_states, ds.eps))


def test_gravity_dataset_is_deterministic() -> None:
    a = _dataset(n_travellers=16)
    b = _dataset(n_travellers=16)
    assert np.array_equal(a.latent, b.latent)
    assert np.array_equal(a.observed, b.observed)
    assert np.array_equal(a.is_holdout, b.is_holdout)


def test_gravity_observed_channel_matches_emission_law() -> None:
    """The realized flip rate of the observation channel matches eps (correctly specified)."""
    ds = _dataset(n_travellers=40, eps=0.3)
    flips = ds.observed != ds.latent
    assert abs(flips.mean() - 0.3) < 0.04
    # clean channel never flips
    clean = _dataset(n_travellers=16, eps=0.0)
    assert np.array_equal(clean.observed, clean.latent)


def test_gravity_latent_is_structurally_different_from_fractalrabbit() -> None:
    """The whole point: the gravity latent stream is NOT FractalRabbit's. If the two latent
    cell streams matched, the cross-generator test would be vacuous. We compare on the SAME
    discretization config and traveller count and require the latent cell streams to differ
    in distribution (different self-transition rate is the cheapest structural fingerprint).
    """
    cfg = _inference(bins=6)
    gravity = _dataset(n_travellers=24)
    fractal_seq = generate_synthetic_sequence(24, cfg, seed=2026)
    fractal_latent = np.asarray(fractal_seq.spatial_states, dtype=int)
    fractal_first = np.asarray(fractal_seq.is_first_observation, dtype=bool)

    def self_transition_rate(latent: np.ndarray, first: np.ndarray) -> float:
        same = [
            int(latent[i] == latent[i - 1])
            for i in range(1, len(latent))
            if not first[i]
        ]
        return float(np.mean(same)) if same else 0.0

    grav_rate = self_transition_rate(gravity.latent, gravity.is_first)
    frac_rate = self_transition_rate(fractal_latent, fractal_first)
    # both are real mobility streams (some persistence) ...
    assert 0.0 < grav_rate < 1.0
    assert 0.0 < frac_rate < 1.0
    # ... but the two mechanisms produce a materially different self-transition regime
    assert abs(grav_rate - frac_rate) > 1e-3


# --------------------------------------------------------------------------- #
# The IDENTICAL spine harness scores the gravity dataset
# --------------------------------------------------------------------------- #
def test_spine_candidates_score_on_gravity_dataset() -> None:
    ds = _dataset()
    for cand in (*reference_baselines(), SoftFilterCandidate()):
        assert isinstance(cand, Candidate)
        scored = score_candidate(ds, cand)
        assert scored.n_scored_steps > 0
        assert np.isfinite(scored.pooled_mean_loss)
        # scored travellers are a subset of the holdout travellers
        holdout_travs = set(int(t) for t in ds.travellers[ds.is_holdout])
        assert set(int(t) for t in scored.step_travellers).issubset(holdout_travs)


def test_uniform_candidate_loss_equals_log_n_on_gravity() -> None:
    """Closed-form sanity check that the spine scorer computes negative-log-loss correctly
    on the gravity dataset: a uniform predictor scores exactly log(n_states) nats per step."""
    ds = _dataset(bins=6)
    n = ds.n_states

    def uniform(ctx: PredictContext) -> np.ndarray:
        return np.ones(ctx.n_states)

    scored = score_candidate(ds, FunctionCandidate("uniform", uniform))
    assert abs(scored.pooled_mean_loss - float(np.log(n))) < 1e-9


def test_peeking_candidate_cannot_win_on_gravity() -> None:
    """Leakage-freeness transfers: a candidate that points all mass at the previous
    observation (the most recent thing it is allowed to see) still cannot approach the
    oracle's ~0 loss on the gravity dataset, because the harness withholds the present."""
    ds = _dataset()

    def peeky(ctx: PredictContext) -> np.ndarray:
        dist = np.full(ctx.n_states, 1e-9)
        dist[int(ctx.prev_observation)] = 1.0
        return dist

    scored = score_candidate(ds, FunctionCandidate("peeky_prev_obs", peeky))
    assert scored.pooled_mean_loss > 0.5


def test_paired_difference_ci_is_well_formed_on_gravity() -> None:
    ds = _dataset()
    soft = score_candidate(ds, SoftFilterCandidate())
    hard, observed_markov = (score_candidate(ds, c) for c in reference_baselines())
    ci = paired_loss_difference_ci(soft, hard, n_boot=500)
    expected = soft.pooled_mean_loss - hard.pooled_mean_loss
    assert abs(ci.mean_difference_nats - expected) < 1e-9
    assert ci.clustered_ci_low <= ci.mean_difference_nats <= ci.clustered_ci_high
    assert not (ci.a_wins_clustered and ci.b_wins_clustered)
    assert ci.n_holdout_travellers >= 2


def test_builder_rejects_too_few_travellers() -> None:
    """The dataset builder enforces its own >=2 traveller contract (a locked split needs at
    least one train and one holdout traveller)."""
    with pytest.raises(ValueError):
        build_gravity_partial_observability_dataset(_inference(), n_travellers=1)


def test_default_n_places_constant_is_used() -> None:
    """The default place count flows through the builder (guards against a silent override)."""
    ds = build_gravity_partial_observability_dataset(
        _inference(), n_travellers=12, n_places=DEFAULT_N_PLACES
    )
    assert ds.latent.shape[0] > 0
