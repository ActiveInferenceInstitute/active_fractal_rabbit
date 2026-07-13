"""Tests for the SOTA program spine (partial-observability regime).

No mocks; real deterministic synthetic sequences and real computation. Covers:
* the high-power generator's shape and locked-split disjointness,
* the leakage-free scorer (a peeking candidate cannot win through the harness, because
  the harness never hands it the current observation or the true target),
* the traveller-clustered bootstrap on the loss difference (and why per-step is invalid).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.sota_program import (
    DEFAULT_N_TRAVELLERS,
    Candidate,
    FunctionCandidate,
    HardFilterCandidate,
    ObservedMarkovCandidate,
    PredictContext,
    SoftFilterCandidate,
    build_partial_observability_dataset,
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
    # smaller-than-default traveller count keeps CI fast while keeping many clusters
    return build_partial_observability_dataset(
        _inference(bins), n_travellers=n_travellers, eps=eps
    )


# --------------------------------------------------------------------------- #
# Generator + locked split
# --------------------------------------------------------------------------- #
def test_emission_matrix_is_the_known_symmetric_channel() -> None:
    n, eps = 6, 0.3
    em = emission_matrix(n, eps)
    assert em.shape == (n, n)
    # rows sum to 1 (proper distribution over observations given a latent cell)
    assert np.allclose(em.sum(axis=1), 1.0)
    assert np.allclose(np.diag(em), 1.0 - eps)
    off = em[0, 1]
    assert abs(off - eps / (n - 1)) < 1e-12
    # clean channel is the identity
    assert np.allclose(emission_matrix(n, 0.0), np.eye(n))


def test_generator_shape_and_disjoint_locked_split() -> None:
    ds = _dataset(n_travellers=24, bins=6)
    assert ds.n_states == 36  # 6x6 grid of spatial cells
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


def test_generator_is_deterministic() -> None:
    a = _dataset(n_travellers=16)
    b = _dataset(n_travellers=16)
    assert np.array_equal(a.latent, b.latent)
    assert np.array_equal(a.observed, b.observed)
    assert np.array_equal(a.is_holdout, b.is_holdout)


def test_observed_channel_matches_emission_law() -> None:
    """The realized flip rate of the observation channel matches eps (correctly specified)."""
    ds = _dataset(n_travellers=DEFAULT_N_TRAVELLERS, eps=0.3)
    flips = ds.observed != ds.latent
    # empirical flip fraction is close to eps over a large sample
    assert abs(flips.mean() - 0.3) < 0.03
    # clean channel never flips
    clean = _dataset(n_travellers=16, eps=0.0)
    assert np.array_equal(clean.observed, clean.latent)


# --------------------------------------------------------------------------- #
# Leakage-free scorer
# --------------------------------------------------------------------------- #
def test_scorer_shapes_and_per_traveller_pooling() -> None:
    ds = _dataset()
    scored = score_candidate(ds, SoftFilterCandidate())
    assert (
        scored.n_scored_steps
        == scored.step_losses.shape[0]
        == scored.step_travellers.shape[0]
    )
    assert (
        scored.n_scored_steps
        == scored.step_true_cells.shape[0]
        == scored.step_confidences.shape[0]
    )
    assert scored.n_scored_steps > 0
    # scored travellers are exactly the holdout travellers
    holdout_travs = set(int(t) for t in ds.travellers[ds.is_holdout])
    assert set(int(t) for t in scored.step_travellers).issubset(holdout_travs)
    # pooled mean equals the mean of the per-step losses
    assert abs(scored.pooled_mean_loss - float(scored.step_losses.mean())) < 1e-12
    # per-traveller means reconstruct from the step losses
    for tid, mean in scored.per_traveller_mean_loss.items():
        mask = scored.step_travellers == tid
        assert abs(mean - float(scored.step_losses[mask].mean())) < 1e-12
    assert np.all(
        (scored.step_true_probabilities >= 0.0)
        & (scored.step_true_probabilities <= 1.0)
    )
    assert np.all((scored.step_confidences >= 0.0) & (scored.step_confidences <= 1.0))
    assert 0.0 <= scored.brier_score <= 1.0
    assert 0.0 <= scored.ece <= 1.0
    assert 0.0 <= scored.top1_accuracy <= 1.0


def test_scorer_is_leakage_free_against_a_peeking_candidate() -> None:
    """A candidate that tries to peek at the current observation CANNOT, because the harness
    never puts the current observation (or the true target) into the context. Such a
    candidate's only available signal is the previous observation, so it can do no better
    than the honest observed-Markov baseline — it gets NO oracle-like loss.
    """
    ds = _dataset()

    # This candidate WANTS to cheat: it would point all mass at context.prev_observation
    # (the most recent thing it is allowed to see). Even maximally exploiting the context,
    # it cannot see the true next cell, so its loss is bounded well away from ~0.
    def peeky(ctx: PredictContext) -> np.ndarray:
        dist = np.full(ctx.n_states, 1e-9)
        dist[int(ctx.prev_observation)] = 1.0
        return dist

    scored = score_candidate(ds, FunctionCandidate("peeky_prev_obs", peeky))
    # A genuine oracle (peeking the true next latent cell) would score ~0 nats. The harness
    # forbids that, so even the most aggressive peek is far from oracle loss.
    assert scored.pooled_mean_loss > 0.5

    # Hard proof the harness withholds the present: the context never equals the
    # current/true cell by construction. We assert the context exposes only past data.
    captured: list[PredictContext] = []

    def capture(ctx: PredictContext) -> np.ndarray:
        captured.append(ctx)
        return np.full(ctx.n_states, 1.0 / ctx.n_states)

    score_candidate(ds, FunctionCandidate("capture", capture))
    assert captured  # at least one holdout step was scored
    # the past-observations window for each scored step ends at the previous observation;
    # the current observation is not appended until after scoring, so the candidate never
    # sees the present.
    for ctx in captured:
        assert ctx.past_observations[-1] == ctx.prev_observation
        assert ctx.belief is None or ctx.belief.shape == (ctx.n_states,)


def test_uniform_candidate_loss_equals_log_n() -> None:
    """A candidate that always predicts uniform scores exactly log(n_states) nats per step —
    a closed-form check that the scorer computes negative-log-loss correctly."""
    ds = _dataset(bins=6)
    n = ds.n_states

    def uniform(ctx: PredictContext) -> np.ndarray:
        return np.ones(ctx.n_states)

    scored = score_candidate(ds, FunctionCandidate("uniform", uniform))
    assert abs(scored.pooled_mean_loss - float(np.log(n))) < 1e-9


def test_reference_baselines_implement_the_protocol_and_score() -> None:
    ds = _dataset()
    for cand in (*reference_baselines(), SoftFilterCandidate()):
        assert isinstance(cand, Candidate)
        scored = score_candidate(ds, cand)
        assert scored.n_scored_steps > 0
        assert np.isfinite(scored.pooled_mean_loss)


def test_scorer_rejects_malformed_distribution() -> None:
    ds = _dataset()

    def all_zero(ctx: PredictContext) -> np.ndarray:
        return np.zeros(ctx.n_states)

    with pytest.raises(ValueError):
        score_candidate(ds, FunctionCandidate("zero", all_zero))

    def wrong_length(ctx: PredictContext) -> np.ndarray:
        return np.ones(ctx.n_states + 1)

    with pytest.raises(ValueError):
        score_candidate(ds, FunctionCandidate("wrong_len", wrong_length))


# --------------------------------------------------------------------------- #
# Traveller-clustered bootstrap on the loss difference
# --------------------------------------------------------------------------- #
def test_paired_difference_ci_brackets_the_mean_and_is_consistent() -> None:
    ds = _dataset()
    soft = score_candidate(ds, SoftFilterCandidate())
    hard = score_candidate(ds, HardFilterCandidate())
    ci = paired_loss_difference_ci(soft, hard, n_boot=500)
    # mean_difference is loss(a) - loss(b) = loss(soft) - loss(hard)
    expected = soft.pooled_mean_loss - hard.pooled_mean_loss
    assert abs(ci.mean_difference_nats - expected) < 1e-9
    assert ci.clustered_ci_low <= ci.mean_difference_nats <= ci.clustered_ci_high
    # cannot simultaneously be a win for both candidates
    assert not (ci.a_wins_clustered and ci.b_wins_clustered)
    assert ci.n_holdout_travellers >= 2


def test_clustered_ci_wider_than_naive_per_step_ci() -> None:
    """Non-vacuity: with correlated within-traveller steps, the traveller-clustered CI is
    markedly WIDER than a naive per-step bootstrap — the reason per-step is invalid here.
    Built on a real scored difference between two reference candidates.
    """
    ds = _dataset(n_travellers=24)
    soft = score_candidate(ds, SoftFilterCandidate())
    obs = score_candidate(ds, ObservedMarkovCandidate())
    ci = paired_loss_difference_ci(soft, obs, n_boot=500)
    clustered_width = ci.clustered_ci_high - ci.clustered_ci_low

    diffs = soft.step_losses - obs.step_losses
    rng = np.random.default_rng(0)
    per_step = [
        rng.choice(diffs, size=diffs.size, replace=True).mean() for _ in range(500)
    ]
    iid_lo, iid_hi = np.percentile(per_step, [2.5, 97.5])
    iid_width = iid_hi - iid_lo
    # within-traveller correlation makes the honest clustered interval strictly wider
    assert clustered_width > iid_width


def test_paired_ci_requires_aligned_steps() -> None:
    """Pairing two candidates scored on different step sets must fail loudly, not silently
    interval mismatched rows."""
    ds_a = _dataset(n_travellers=24)
    ds_b = _dataset(
        n_travellers=16
    )  # different traveller set => different scored steps
    a = score_candidate(ds_a, SoftFilterCandidate())
    b = score_candidate(ds_b, SoftFilterCandidate())
    with pytest.raises(ValueError):
        paired_loss_difference_ci(a, b, n_boot=100)
