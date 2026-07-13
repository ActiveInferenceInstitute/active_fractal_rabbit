"""Tests for the neural SOTA candidate (sota_neural).

No mocks; real deterministic synthetic data and real training/scoring through the
``sota_program`` spine. Covers:
* the candidate satisfies the spine's Candidate protocol and scores leakage-free,
* training is deterministic and uses TRAIN travellers only (no holdout leakage),
* the trained network is a genuine FAIR competitor — at high power it beats the weak
  observed-Markov null AND the strongest fair baseline (hard_filter) but does NOT beat
  the Bayes-optimal soft_filter (the AIF candidate); at low power it overfits — the
  honest, reported behaviour, never a manufactured win,
* the candidate cannot peek: a network trained on a leakage-free belief stays far from
  oracle loss, and its prediction depends only on the belief carried through step i-1,
* edge cases: empty/None belief falls back to uniform; malformed configs raise.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.sota_program import (
    Candidate,
    HardFilterCandidate,
    ObservedMarkovCandidate,
    PredictContext,
    SoftFilterCandidate,
    build_partial_observability_dataset,
    paired_loss_difference_ci,
    score_candidate,
)
from active_fractal_rabbit.sota_neural import (
    SOTA_NEURAL_SCHEMA,
    MlpNextCellCandidate,
    MlpWeights,
    train_mlp_candidate,
)

ROOT = Path(__file__).resolve().parents[1]


def _inference(bins: int = 6):
    cfg = load_config(ROOT / "configs" / "default.yaml")
    return replace(
        cfg.inference, observation=replace(cfg.inference.observation, spatial_bins=bins)
    )


def _dataset(n_travellers: int = 24, eps: float = 0.3, bins: int = 6):
    return build_partial_observability_dataset(
        _inference(bins), n_travellers=n_travellers, eps=eps
    )


# --------------------------------------------------------------------------- #
# Protocol + basic scoring
# --------------------------------------------------------------------------- #
def test_candidate_implements_protocol_and_scores() -> None:
    ds = _dataset()
    cand = train_mlp_candidate(ds)
    assert isinstance(cand, Candidate)
    assert cand.name == "mlp_next_cell"
    scored = score_candidate(ds, cand)
    assert scored.n_scored_steps > 0
    assert np.isfinite(scored.pooled_mean_loss)
    # only holdout travellers are scored
    holdout_travs = set(int(t) for t in ds.travellers[ds.is_holdout])
    assert set(int(t) for t in scored.step_travellers).issubset(holdout_travs)


def test_schema_constant_is_versioned() -> None:
    assert SOTA_NEURAL_SCHEMA == "active-fractal-rabbit-sota-neural-v1"


def test_predict_returns_a_proper_distribution() -> None:
    ds = _dataset()
    cand = train_mlp_candidate(ds)
    belief = np.full(ds.n_states, 1.0 / ds.n_states)
    ctx = PredictContext(
        n_states=ds.n_states,
        transition=ds.transition,
        emission=ds.emission,
        past_observations=np.array([0, 1], dtype=int),
        prev_observation=1,
        belief=belief,
        traveller_id=0,
        step_index=3,
    )
    dist = cand.predict(ctx)
    assert dist.shape == (ds.n_states,)
    assert np.all(dist >= 0.0)
    assert abs(float(dist.sum()) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Determinism + train-only fitting
# --------------------------------------------------------------------------- #
def test_training_is_deterministic() -> None:
    ds = _dataset(n_travellers=16)
    a = train_mlp_candidate(ds)
    b = train_mlp_candidate(ds)
    assert np.array_equal(a.weights.w1, b.weights.w1)
    assert np.array_equal(a.weights.w2, b.weights.w2)
    assert np.array_equal(a.weights.b1, b.weights.b1)
    assert np.array_equal(a.weights.b2, b.weights.b2)
    assert a.train_mean_loss == b.train_mean_loss
    # identical predictions on the holdout set
    sa = score_candidate(ds, a)
    sb = score_candidate(ds, b)
    assert np.array_equal(sa.step_losses, sb.step_losses)


def test_training_uses_train_travellers_only() -> None:
    """Permuting the LATENT labels of holdout travellers must not change the trained
    weights at all — proof the optimizer never touches holdout data."""
    ds = _dataset(n_travellers=24)
    base = train_mlp_candidate(ds)

    # Corrupt only the holdout travellers' latent labels; train travellers untouched.
    corrupted_latent = ds.latent.copy()
    rng = np.random.default_rng(123)
    holdout_idx = np.flatnonzero(ds.is_holdout)
    corrupted_latent[holdout_idx] = rng.integers(0, ds.n_states, holdout_idx.size)
    ds_corrupt = replace(ds, latent=corrupted_latent)

    corrupt = train_mlp_candidate(ds_corrupt)
    # The training set (TRAIN travellers' beliefs+targets) is unchanged, so weights match.
    assert np.array_equal(base.weights.w1, corrupt.weights.w1)
    assert np.array_equal(base.weights.w2, corrupt.weights.w2)
    assert base.train_mean_loss == corrupt.train_mean_loss


# --------------------------------------------------------------------------- #
# Leakage-freeness
# --------------------------------------------------------------------------- #
def test_candidate_cannot_peek_stays_far_from_oracle() -> None:
    """A genuine oracle (knowing the next latent cell) scores ~0 nats. The trained network
    only sees the leakage-free belief, so its holdout loss is bounded well away from 0."""
    ds = _dataset(n_travellers=24)
    cand = train_mlp_candidate(ds)
    scored = score_candidate(ds, cand)
    assert scored.pooled_mean_loss > 0.5  # nowhere near the oracle's ~0


def test_prediction_depends_only_on_the_belief() -> None:
    """Two contexts with identical belief but different (irrelevant) current-step metadata
    yield identical predictions — the candidate uses no information beyond the belief."""
    ds = _dataset(n_travellers=16)
    cand = train_mlp_candidate(ds)
    belief = ds.emission[3, :] / ds.emission[3, :].sum()
    base_kw = dict(
        n_states=ds.n_states,
        transition=ds.transition,
        emission=ds.emission,
        belief=belief,
    )
    ctx_a = PredictContext(
        past_observations=np.array([3], dtype=int),
        prev_observation=3,
        traveller_id=0,
        step_index=1,
        **base_kw,
    )
    ctx_b = PredictContext(
        past_observations=np.array([5, 7, 3], dtype=int),
        prev_observation=3,
        traveller_id=9,
        step_index=42,
        **base_kw,
    )
    assert np.allclose(cand.predict(ctx_a), cand.predict(ctx_b))


# --------------------------------------------------------------------------- #
# Honest competitiveness (no manufactured win)
# --------------------------------------------------------------------------- #
def test_high_power_beats_fair_baselines_but_not_the_aif_soft_filter() -> None:
    """At high power the learned model beats the weak observed-Markov null AND the strongest
    fair baseline (hard_filter), yet does NOT beat the Bayes-optimal soft_filter (the AIF
    candidate). This is the honest ceiling: a learned predictor cannot out-predict the exact
    posterior of a KNOWN generative model. The soft filter winning is the truthful outcome,
    not a defect of the neural candidate."""
    ds = _dataset(n_travellers=80)
    neural = score_candidate(ds, train_mlp_candidate(ds))
    obs = score_candidate(ds, ObservedMarkovCandidate())
    hard = score_candidate(ds, HardFilterCandidate())
    soft = score_candidate(ds, SoftFilterCandidate())

    # beats the weak null and the strongest fair point-estimate baseline
    assert neural.pooled_mean_loss < obs.pooled_mean_loss
    assert neural.pooled_mean_loss < hard.pooled_mean_loss
    # but the AIF soft filter (exact posterior marginal) is still strictly better
    assert soft.pooled_mean_loss < neural.pooled_mean_loss

    # ... and the soft filter's edge over the neural model is CI-separated under the
    # honest traveller-clustered bootstrap (a=neural, b=soft => b_wins => soft better).
    ci = paired_loss_difference_ci(neural, soft, n_boot=600)
    assert ci.b_wins_clustered
    assert not ci.a_wins_clustered


def test_low_power_overfits_and_is_reported_honestly() -> None:
    """With few travellers the learned model overfits the finite traveller pool: its holdout
    loss is WORSE than the transparent baselines. We assert this honestly rather than tuning
    it away — it documents the real data-hunger of the learned baseline."""
    ds = _dataset(n_travellers=16)
    neural = score_candidate(ds, train_mlp_candidate(ds))
    hard = score_candidate(ds, HardFilterCandidate())
    # the learned model does not clear the fair baseline at low power
    assert neural.pooled_mean_loss > hard.pooled_mean_loss


# --------------------------------------------------------------------------- #
# Edge cases / config validation
# --------------------------------------------------------------------------- #
def test_predict_falls_back_to_uniform_without_a_belief() -> None:
    ds = _dataset(n_travellers=16)
    cand = train_mlp_candidate(ds)
    ctx = PredictContext(
        n_states=ds.n_states,
        transition=ds.transition,
        emission=ds.emission,
        past_observations=np.empty(0, dtype=int),
        prev_observation=0,
        belief=None,
        traveller_id=0,
        step_index=0,
    )
    dist = cand.predict(ctx)
    assert np.allclose(dist, 1.0 / ds.n_states)


def test_predict_rejects_state_count_mismatch() -> None:
    ds = _dataset(n_travellers=16)
    cand = train_mlp_candidate(ds)
    bad = PredictContext(
        n_states=ds.n_states + 1,
        transition=ds.transition,
        emission=ds.emission,
        past_observations=np.array([0], dtype=int),
        prev_observation=0,
        belief=np.full(ds.n_states + 1, 1.0 / (ds.n_states + 1)),
        traveller_id=0,
        step_index=1,
    )
    with pytest.raises(ValueError):
        cand.predict(bad)


def test_invalid_training_configs_raise() -> None:
    ds = _dataset(n_travellers=16)
    with pytest.raises(ValueError):
        train_mlp_candidate(ds, hidden=0)
    with pytest.raises(ValueError):
        train_mlp_candidate(ds, epochs=0)
    with pytest.raises(ValueError):
        train_mlp_candidate(ds, batch_size=0)


def test_training_raises_without_examples() -> None:
    """A dataset whose non-first steps are all holdout has no training examples."""
    ds = _dataset(n_travellers=16)
    all_holdout = replace(ds, is_holdout=np.ones_like(ds.is_holdout, dtype=bool))
    with pytest.raises(ValueError):
        train_mlp_candidate(all_holdout)


def test_weights_initial_shapes() -> None:
    w = MlpWeights.initial(36, 24, seed=0)
    assert w.w1.shape == (24, 36)
    assert w.b1.shape == (24,)
    assert w.w2.shape == (36, 24)
    assert w.b2.shape == (36,)


def test_candidate_dataclass_carries_train_loss() -> None:
    ds = _dataset(n_travellers=16)
    cand = train_mlp_candidate(ds)
    assert isinstance(cand, MlpNextCellCandidate)
    assert np.isfinite(cand.train_mean_loss)
    assert cand.hidden == 24
    assert cand.n_states == ds.n_states
