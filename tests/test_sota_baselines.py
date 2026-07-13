"""Tests for the transparent fair-baseline ladder (no mocks; real deterministic data).

Every baseline is exercised through the real spine harness (``score_candidate``) on real
synthetic partial-observability datasets, with real computation and fixed seeds. The
tests prove:

* each ladder member implements the spine ``Candidate`` protocol and produces a finite,
  proper, leakage-free score on the holdout travellers,
* the inference primitive (the emission posterior) is the correctly-specified single-step
  posterior — a closed-form check, and at a clean channel it collapses to the true cell,
* none of the baselines beats a true oracle (they infer location from noisy observations,
  they never peek), and a peeking baseline cannot win through the harness,
* the return boost is a leakage-free reweighting toward visited cells, strictly distinct
  from the boost-free first-order Markov,
* the phase-conditioned Markov is honestly unavailable against the spine's phase-free
  context.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.sota_baselines import (
    FirstOrderMarkovCandidate,
    HigherOrderBackoffMarkovCandidate,
    MultiHorizonRecencyAttentionCandidate,
    OnlineBaseRateCandidate,
    PersistenceCandidate,
    RecencyAttentionCandidate,
    ReturnBoostedMarkovCandidate,
    _emission_posterior,
    _emission_posteriors,
    phase_conditioning_available,
    transparent_baseline_ladder,
)
from active_fractal_rabbit.sota_program import (
    Candidate,
    FunctionCandidate,
    PredictContext,
    build_partial_observability_dataset,
    emission_matrix,
    paired_loss_difference_ci,
    score_candidate,
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


def _ctx(
    *,
    n_states: int,
    emission: np.ndarray,
    transition: np.ndarray,
    past: np.ndarray,
    prev_obs: int,
    belief: np.ndarray | None = None,
) -> PredictContext:
    return PredictContext(
        n_states=n_states,
        transition=transition,
        emission=emission,
        past_observations=np.asarray(past, dtype=int),
        prev_observation=int(prev_obs),
        belief=belief,
        traveller_id=0,
        step_index=1,
    )


# --------------------------------------------------------------------------- #
# Inference primitive: the emission posterior
# --------------------------------------------------------------------------- #
def test_emission_posterior_is_the_correctly_specified_single_step_posterior() -> None:
    n, eps = 6, 0.3
    em = emission_matrix(n, eps)
    post = _emission_posterior(em, 2)
    assert post.shape == (n,)
    assert abs(post.sum() - 1.0) < 1e-12
    # uniform prior => posterior ∝ emission row; the observed cell is most probable
    assert np.argmax(post) == 2
    assert abs(post[2] - (1.0 - eps)) < 1e-12
    # off-cells share the remaining mass uniformly
    assert abs(post[0] - eps / (n - 1)) < 1e-12


def test_emission_posterior_collapses_to_truth_on_a_clean_channel() -> None:
    n = 5
    em = emission_matrix(n, 0.0)  # identity channel
    for o in range(n):
        post = _emission_posterior(em, o)
        onehot = np.zeros(n)
        onehot[o] = 1.0
        assert np.allclose(post, onehot)


def test_emission_posterior_degenerate_row_falls_back_to_uniform() -> None:
    n = 4
    em = np.zeros((n, n))  # pathological all-zero emission row
    post = _emission_posterior(em, 0)
    assert np.allclose(post, np.full(n, 1.0 / n))
    # batched form is consistent with the scalar form
    batched = _emission_posteriors(em, np.array([0, 1, 2]))
    assert batched.shape == (3, n)
    assert np.allclose(batched, np.full((3, n), 1.0 / n))


def test_emission_posteriors_batches_match_scalar() -> None:
    n, eps = 6, 0.2
    em = emission_matrix(n, eps)
    obs = np.array([0, 3, 5, 1])
    batched = _emission_posteriors(em, obs)
    assert batched.shape == (obs.size, n)
    for i, o in enumerate(obs):
        assert np.allclose(batched[i], _emission_posterior(em, int(o)))
    # empty input yields a well-formed empty stack
    assert _emission_posteriors(em, np.array([], dtype=int)).shape == (0, n)


# --------------------------------------------------------------------------- #
# Candidate-protocol conformance and real scoring through the harness
# --------------------------------------------------------------------------- #
def test_ladder_members_implement_protocol_and_score_finite() -> None:
    ds = _dataset()
    ladder = transparent_baseline_ladder()
    assert len(ladder) == 7
    names = {c.name for c in ladder}
    assert names == {
        "persistence",
        "online_base_rate",
        "first_order_markov",
        "higher_order_backoff_markov",
        "return_boosted_markov",
        "recency_attention",
        "multi_horizon_recency_attention",
    }
    for cand in ladder:
        assert isinstance(cand, Candidate)
        scored = score_candidate(ds, cand)
        assert scored.n_scored_steps > 0
        assert np.isfinite(scored.pooled_mean_loss)
        # proper next-cell loss is non-negative and bounded by the uniform null (log n) only
        # in expectation; here we just assert positivity and finiteness per the scorer.
        assert scored.pooled_mean_loss > 0.0


def test_every_ladder_member_returns_a_proper_distribution() -> None:
    n, eps = 6, 0.3
    em = emission_matrix(n, eps)
    # a real (small) transition: column-stochastic, transition[next, prev]
    rng = np.random.default_rng(0)
    raw = rng.random((n, n)) + 0.1
    transition = raw / raw.sum(axis=0, keepdims=True)
    past = np.array([0, 1, 0, 2, 5])
    ctx = _ctx(n_states=n, emission=em, transition=transition, past=past, prev_obs=5)
    for cand in transparent_baseline_ladder():
        dist = cand.predict(ctx)
        assert dist.shape == (n,)
        assert np.all(np.isfinite(dist))
        assert np.all(dist >= -1e-12)
        # the harness normalises, but every baseline already returns a proper (or
        # proportional, all-non-negative, non-zero) vector
        assert dist.sum() > 0.0


# --------------------------------------------------------------------------- #
# Semantics of individual baselines (closed-form / structural)
# --------------------------------------------------------------------------- #
def test_persistence_predicts_the_inferred_previous_cell_posterior() -> None:
    n, eps = 6, 0.3
    em = emission_matrix(n, eps)
    transition = np.eye(n)  # irrelevant: persistence ignores the transition
    ctx = _ctx(
        n_states=n, emission=em, transition=transition, past=np.array([4]), prev_obs=4
    )
    dist = PersistenceCandidate().predict(ctx)
    assert np.allclose(dist, _emission_posterior(em, 4))


def test_first_order_markov_is_transition_times_inferred_prev() -> None:
    n, eps = 6, 0.25
    em = emission_matrix(n, eps)
    rng = np.random.default_rng(1)
    raw = rng.random((n, n)) + 0.05
    transition = raw / raw.sum(axis=0, keepdims=True)
    ctx = _ctx(
        n_states=n,
        emission=em,
        transition=transition,
        past=np.array([2, 3]),
        prev_obs=3,
    )
    dist = FirstOrderMarkovCandidate().predict(ctx)
    expected = transition @ _emission_posterior(em, 3)
    assert np.allclose(dist, expected)


def test_first_order_markov_ignores_the_running_belief() -> None:
    """Fairness: the boost-free Markov is non-recursive and must NOT use context.belief."""
    n, eps = 5, 0.3
    em = emission_matrix(n, eps)
    rng = np.random.default_rng(2)
    raw = rng.random((n, n)) + 0.05
    transition = raw / raw.sum(axis=0, keepdims=True)
    base = _ctx(
        n_states=n, emission=em, transition=transition, past=np.array([1]), prev_obs=1
    )
    with_belief = replace(base, belief=np.array([0.7, 0.1, 0.1, 0.05, 0.05]))
    assert np.allclose(
        FirstOrderMarkovCandidate().predict(base),
        FirstOrderMarkovCandidate().predict(with_belief),
    )


def test_higher_order_backoff_is_history_sensitive_and_protocol_compatible() -> None:
    n, eps = 6, 0.0
    em = emission_matrix(n, eps)
    transition = np.full((n, n), 1.0 / n)
    past = np.array([2, 4, 2, 5, 2, 4])
    ctx = _ctx(n_states=n, emission=em, transition=transition, past=past, prev_obs=4)
    cand = HigherOrderBackoffMarkovCandidate(
        higher_order_weight=0.70, return_weight=0.10
    )
    dist = cand.predict(ctx)

    assert cand.name == "higher_order_backoff_markov"
    assert dist.shape == (n,)
    assert np.all(np.isfinite(dist))
    assert np.all(dist >= 0.0)
    assert abs(dist.sum() - 1.0) < 1e-9
    assert int(np.argmax(dist)) == 2

    changed = replace(ctx, past_observations=np.array([1, 3, 1, 3, 1, 5]))
    assert not np.allclose(dist, cand.predict(changed))


def test_higher_order_backoff_falls_to_first_order_with_insufficient_history() -> None:
    n, eps = 5, 0.3
    em = emission_matrix(n, eps)
    rng = np.random.default_rng(7)
    raw = rng.random((n, n)) + 0.1
    transition = raw / raw.sum(axis=0, keepdims=True)
    ctx = _ctx(
        n_states=n,
        emission=em,
        transition=transition,
        past=np.array([1, 2]),
        prev_obs=2,
    )

    first = FirstOrderMarkovCandidate().predict(ctx)
    higher = HigherOrderBackoffMarkovCandidate().predict(ctx)
    assert np.allclose(higher, first / first.sum())


def test_online_base_rate_accumulates_soft_posteriors_over_the_past() -> None:
    n, eps = 6, 0.3
    em = emission_matrix(n, eps)
    transition = np.eye(n)  # ignored by base rate
    past = np.array([0, 0, 3, 5])
    ctx = _ctx(n_states=n, emission=em, transition=transition, past=past, prev_obs=5)
    dist = OnlineBaseRateCandidate(smoothing=0.5).predict(ctx)
    expected = np.full(n, 0.5) + _emission_posteriors(em, past).sum(axis=0)
    expected = expected / expected.sum()
    assert np.allclose(dist, expected)
    # with an empty past it is just the smoothed prior (uniform)
    empty_ctx = _ctx(
        n_states=n,
        emission=em,
        transition=transition,
        past=np.array([], dtype=int),
        prev_obs=2,
    )
    assert np.allclose(
        OnlineBaseRateCandidate().predict(empty_ctx), np.full(n, 1.0 / n)
    )


def test_return_boost_reweights_toward_visited_cells_and_differs_from_markov() -> None:
    n, eps = (
        6,
        0.0,
    )  # clean channel => inferred cell == observation (deterministic visited set)
    em = emission_matrix(n, eps)
    rng = np.random.default_rng(3)
    raw = rng.random((n, n)) + 0.05
    transition = raw / raw.sum(axis=0, keepdims=True)
    past = np.array([0, 1, 2])  # visited cells {0,1,2} on a clean channel
    ctx = _ctx(n_states=n, emission=em, transition=transition, past=past, prev_obs=2)

    markov = FirstOrderMarkovCandidate().predict(ctx)
    boosted = ReturnBoostedMarkovCandidate(return_weight=1.0).predict(ctx)
    assert not np.allclose(markov, boosted)  # the boost actually changes the prediction

    # visited cells gain relative mass, unvisited cells lose relative mass after renorm
    visited = np.array([True, True, True, False, False, False])
    # ratio (boosted/markov) is strictly larger on visited than the max over unvisited
    ratio = boosted / np.clip(markov, 1e-12, None)
    assert ratio[visited].min() > ratio[~visited].max()


def test_return_boost_with_zero_weight_equals_first_order_markov() -> None:
    n, eps = 6, 0.3
    em = emission_matrix(n, eps)
    rng = np.random.default_rng(4)
    raw = rng.random((n, n)) + 0.05
    transition = raw / raw.sum(axis=0, keepdims=True)
    ctx = _ctx(
        n_states=n,
        emission=em,
        transition=transition,
        past=np.array([0, 1]),
        prev_obs=1,
    )
    markov = FirstOrderMarkovCandidate().predict(ctx)
    no_boost = ReturnBoostedMarkovCandidate(return_weight=0.0).predict(ctx)
    # markov returns transition @ post (sums to ~1); zero-boost renormalises the same vector
    assert np.allclose(no_boost, markov / markov.sum())


def test_return_boost_falls_back_to_uniform_on_a_degenerate_prediction() -> None:
    """Defensive guard: if the (pathological) transition yields all-zero predictive mass
    and there is nothing to boost, the candidate returns a proper uniform distribution
    rather than dividing by zero."""
    n, eps = 4, 0.3
    em = emission_matrix(n, eps)
    zero_transition = np.zeros(
        (n, n)
    )  # not column-stochastic: forces zero predictive mass
    ctx = _ctx(
        n_states=n,
        emission=em,
        transition=zero_transition,
        past=np.array([], dtype=int),  # empty => no visited boost either
        prev_obs=1,
    )
    dist = ReturnBoostedMarkovCandidate().predict(ctx)
    assert np.allclose(dist, np.full(n, 1.0 / n))


def test_return_boost_with_empty_past_is_a_proper_distribution() -> None:
    n, eps = 5, 0.3
    em = emission_matrix(n, eps)
    transition = np.eye(n)
    ctx = _ctx(
        n_states=n,
        emission=em,
        transition=transition,
        past=np.array([], dtype=int),
        prev_obs=2,
    )
    dist = ReturnBoostedMarkovCandidate().predict(ctx)
    assert abs(dist.sum() - 1.0) < 1e-9
    assert np.all(dist >= 0.0)


def test_recency_attention_is_dependency_light_and_uses_past_only() -> None:
    n, eps = 6, 0.2
    em = emission_matrix(n, eps)
    rng = np.random.default_rng(5)
    raw = rng.random((n, n)) + 0.05
    transition = raw / raw.sum(axis=0, keepdims=True)
    past = np.array([0, 4, 4, 2, 4])
    ctx = _ctx(n_states=n, emission=em, transition=transition, past=past, prev_obs=4)

    cand = RecencyAttentionCandidate(recency_half_life=2.0, return_weight=0.25)
    dist = cand.predict(ctx)
    assert cand.name == "recency_attention"
    assert dist.shape == (n,)
    assert np.all(np.isfinite(dist))
    assert np.all(dist >= 0.0)
    assert abs(dist.sum() - 1.0) < 1e-9

    changed_past = replace(ctx, past_observations=np.array([1, 1, 1, 1, 1]))
    assert not np.allclose(dist, cand.predict(changed_past))


def test_multi_horizon_attention_is_dependency_light_and_history_sensitive() -> None:
    n, eps = 6, 0.2
    em = emission_matrix(n, eps)
    rng = np.random.default_rng(6)
    raw = rng.random((n, n)) + 0.05
    transition = raw / raw.sum(axis=0, keepdims=True)
    ctx = _ctx(
        n_states=n,
        emission=em,
        transition=transition,
        past=np.array([0, 4, 4, 2, 4, 5, 5, 1]),
        prev_obs=1,
    )

    cand = MultiHorizonRecencyAttentionCandidate()
    dist = cand.predict(ctx)
    assert cand.name == "multi_horizon_recency_attention"
    assert dist.shape == (n,)
    assert np.all(np.isfinite(dist))
    assert np.all(dist >= 0.0)
    assert abs(dist.sum() - 1.0) < 1e-9

    short_only = MultiHorizonRecencyAttentionCandidate(
        half_lives=(2.0,), horizon_weights=(1.0,)
    )
    long_only = MultiHorizonRecencyAttentionCandidate(
        half_lives=(32.0,), horizon_weights=(1.0,)
    )
    assert not np.allclose(short_only.predict(ctx), long_only.predict(ctx))


# --------------------------------------------------------------------------- #
# Honesty: baselines infer (no oracle), and the harness blocks peeking
# --------------------------------------------------------------------------- #
def test_baselines_never_beat_a_true_oracle() -> None:
    """Each baseline must infer location from noisy observations; a true oracle that is
    handed the next latent cell scores far lower. (Sanity bound on the honesty invariant —
    the baselines carry a real handicap.)"""
    ds = _dataset(n_travellers=24, eps=0.3)

    # A genuine oracle: place all mass on the TRUE next cell. This is NOT a fair candidate
    # — it is built here only to certify the baselines are nowhere near oracle loss.
    latent = ds.latent

    def oracle(ctx: PredictContext) -> np.ndarray:
        dist = np.full(ctx.n_states, 1e-9)
        dist[int(latent[ctx.step_index])] = 1.0
        return dist

    oracle_loss = score_candidate(
        ds, FunctionCandidate("oracle", oracle)
    ).pooled_mean_loss
    for cand in transparent_baseline_ladder():
        baseline_loss = score_candidate(ds, cand).pooled_mean_loss
        assert baseline_loss > oracle_loss
        # the handicap is large: the baselines are well away from the ~0-nat oracle
        assert baseline_loss > 0.5


def test_clean_channel_first_order_markov_matches_the_shared_filter_prediction() -> (
    None
):
    """Positive control: at a clean emission the inferred prev cell IS the true observation,
    so the first-order Markov prediction equals transition @ onehot(prev_observation) —
    i.e. the spine's ObservedMarkovCandidate. This verifies the inference primitive is
    correctly specified by collapsing to the oracle-of-the-observation at zero noise."""
    ds = _dataset(n_travellers=16, eps=0.0)
    n = ds.n_states

    def observed_markov(ctx: PredictContext) -> np.ndarray:
        onehot = np.zeros(n)
        onehot[int(ctx.prev_observation)] = 1.0
        return ctx.transition @ onehot

    markov = score_candidate(ds, FirstOrderMarkovCandidate())
    observed = score_candidate(
        ds, FunctionCandidate("observed_markov_ref", observed_markov)
    )
    assert abs(markov.pooled_mean_loss - observed.pooled_mean_loss) < 1e-9


def test_paired_ci_against_a_ladder_member_is_well_formed() -> None:
    """The ladder plugs into the clustered-bootstrap difference exactly like reference
    candidates (same holdout steps, valid traveller-clustered interval)."""
    ds = _dataset(n_travellers=24)
    markov = score_candidate(ds, FirstOrderMarkovCandidate())
    boosted = score_candidate(ds, ReturnBoostedMarkovCandidate())
    ci = paired_loss_difference_ci(markov, boosted, n_boot=400)
    expected = markov.pooled_mean_loss - boosted.pooled_mean_loss
    assert abs(ci.mean_difference_nats - expected) < 1e-9
    assert ci.clustered_ci_low <= ci.mean_difference_nats <= ci.clustered_ci_high
    assert not (ci.a_wins_clustered and ci.b_wins_clustered)
    assert ci.n_holdout_travellers >= 2


# --------------------------------------------------------------------------- #
# Phase-conditioned Markov: honestly unavailable against the spine context
# --------------------------------------------------------------------------- #
def test_phase_conditioning_is_unavailable_against_the_spine_context() -> None:
    n, eps = 6, 0.3
    em = emission_matrix(n, eps)
    ctx = _ctx(
        n_states=n, emission=em, transition=np.eye(n), past=np.array([0]), prev_obs=0
    )
    # the spine PredictContext carries no phase / time-of-day factor, so a phase-conditioned
    # Markov cannot be built honestly here — this is recorded explicitly, not silently faked.
    assert phase_conditioning_available(ctx) is False
    assert not hasattr(ctx, "phase")
    assert not hasattr(ctx, "time_bin")
    # the ladder therefore intentionally omits a phase-conditioned candidate
    assert "phase" not in {c.name for c in transparent_baseline_ladder()}


def test_determinism_of_ladder_scores() -> None:
    ds_a = _dataset(n_travellers=16)
    ds_b = _dataset(n_travellers=16)
    for cand_cls in (
        PersistenceCandidate,
        OnlineBaseRateCandidate,
        FirstOrderMarkovCandidate,
        HigherOrderBackoffMarkovCandidate,
        ReturnBoostedMarkovCandidate,
        RecencyAttentionCandidate,
    ):
        a = score_candidate(ds_a, cand_cls())
        b = score_candidate(ds_b, cand_cls())
        assert a.pooled_mean_loss == b.pooled_mean_loss
        assert np.array_equal(a.step_losses, b.step_losses)


def test_scorer_blocks_peeking_through_a_baseline_shaped_candidate() -> None:
    """A candidate that tries to exploit only context-available data (prev observation)
    cannot reach oracle loss — the harness withholds the present, so even the most
    aggressive 'baseline-like' peek stays far above oracle (mirrors the spine's leakage
    test, applied to the ladder's information set)."""
    ds = _dataset(n_travellers=24)

    def peek_prev(ctx: PredictContext) -> np.ndarray:
        dist = np.full(ctx.n_states, 1e-9)
        dist[int(ctx.prev_observation)] = 1.0
        return dist

    scored = score_candidate(ds, FunctionCandidate("peek_prev", peek_prev))
    assert scored.pooled_mean_loss > 0.5
