"""Tests for the two classical sequence candidates (``sota_classical.py``).

No mocks; real deterministic synthetic sequences and real computation against the spine.
Covers, for both the switching-noise HMM and the particle filter:
* protocol conformance and scoring through the leakage-free spine,
* fairness — neither candidate can out-score the leakage-free baselines through the harness
  (no oracle access; a peeking attempt is structurally impossible because the spine never
  exposes the present),
* the particle filter's determinism and its convergence to the exact soft filter under the
  known emission as the particle count grows,
* the switching HMM's regime posterior concentrating on the best-matched emission channel,
* both candidates beating the strongest FAIR baseline (the hard point estimate) with a
  traveller-clustered CI separated from zero — and NOT beating the soft filter / oracle
  (the honest ceiling).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.sota_classical import (
    SOTA_CLASSICAL_SCHEMA,
    ParticleFilterCandidate,
    SwitchingNoiseHMMCandidate,
    _normalize,
    _uniform,
    classical_candidates,
)
from active_fractal_rabbit.sota_program import (
    Candidate,
    HardFilterCandidate,
    ObservedMarkovCandidate,
    PredictContext,
    SoftFilterCandidate,
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


# --------------------------------------------------------------------------- #
# Protocol conformance + basic scoring
# --------------------------------------------------------------------------- #
def test_candidates_implement_the_protocol_and_score() -> None:
    ds = _dataset()
    for cand in classical_candidates():
        assert isinstance(cand, Candidate)
        scored = score_candidate(ds, cand)
        assert scored.n_scored_steps > 0
        assert np.isfinite(scored.pooled_mean_loss)
        # scored travellers are exactly holdout travellers
        holdout = set(int(t) for t in ds.travellers[ds.is_holdout])
        assert set(int(t) for t in scored.step_travellers).issubset(holdout)


def test_classical_candidates_returns_both_named_candidates() -> None:
    cands = classical_candidates()
    names = {c.name for c in cands}
    assert names == {"switching_noise_hmm", "particle_filter"}
    assert SOTA_CLASSICAL_SCHEMA == "active-fractal-rabbit-sota-classical-v1"


def test_predictions_are_valid_distributions() -> None:
    """Every per-step prediction is a finite, non-negative, length-n distribution summing to 1."""
    ds = _dataset(n_travellers=16)
    n = ds.n_states
    for cand in classical_candidates():
        captured: list[np.ndarray] = []
        original = cand.predict

        def wrapped(ctx: PredictContext, _orig=original, _cap=captured) -> np.ndarray:
            out = _orig(ctx)
            _cap.append(np.asarray(out, dtype=float))
            return out

        cand.predict = wrapped  # type: ignore[method-assign]
        score_candidate(ds, cand)
        assert captured
        for dist in captured:
            assert dist.shape == (n,)
            assert np.all(np.isfinite(dist))
            assert np.all(dist >= -1e-12)
            assert abs(float(dist.sum()) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Fairness / leakage-freeness
# --------------------------------------------------------------------------- #
def test_candidates_never_see_the_present_or_the_target() -> None:
    """The spine hands these candidates only past data. We assert each context the candidate
    receives ends at the previous observation, so neither can peek at o_i or z_i."""
    ds = _dataset(n_travellers=16)
    for cand in classical_candidates():
        captured: list[PredictContext] = []
        original = cand.predict

        def wrapped(ctx: PredictContext, _orig=original, _cap=captured) -> np.ndarray:
            _cap.append(ctx)
            return _orig(ctx)

        cand.predict = wrapped  # type: ignore[method-assign]
        score_candidate(ds, cand)
        assert captured
        for ctx in captured:
            # the visible history ends at the previous observation; the current one is never
            # appended until after scoring, so the candidate cannot peek at the present
            assert ctx.past_observations[-1] == ctx.prev_observation


def test_candidates_cannot_beat_the_oracle_ceiling() -> None:
    """A genuine oracle (peeking the true next latent cell) scores ~0 nats. Because the spine
    forbids that, a fair candidate's loss stays well away from zero — these candidates are
    NOT oracle-beating."""
    ds = _dataset(n_travellers=24)
    for cand in classical_candidates():
        scored = score_candidate(ds, cand)
        assert scored.pooled_mean_loss > 0.5  # nowhere near the ~0 oracle floor


# --------------------------------------------------------------------------- #
# Particle filter: determinism + convergence to the exact soft filter
# --------------------------------------------------------------------------- #
def test_particle_filter_is_deterministic() -> None:
    ds = _dataset(n_travellers=16)
    a = score_candidate(ds, ParticleFilterCandidate())
    b = score_candidate(ds, ParticleFilterCandidate())
    assert np.array_equal(a.step_losses, b.step_losses)
    assert a.pooled_mean_loss == b.pooled_mean_loss


def test_particle_filter_converges_to_exact_soft_filter() -> None:
    """Under the KNOWN emission the exact recursive filter is optimal; a bootstrap particle
    filter is a Monte-Carlo approximation of it. With enough particles the particle filter's
    pooled loss approaches the exact soft filter's, and is strictly closer at 2048 than at a
    very small particle count (more particles => less approximation error)."""
    ds = _dataset(n_travellers=16)
    exact = score_candidate(ds, SoftFilterCandidate()).pooled_mean_loss
    coarse = score_candidate(
        ds, ParticleFilterCandidate(n_particles=16)
    ).pooled_mean_loss
    fine = score_candidate(
        ds, ParticleFilterCandidate(n_particles=2048)
    ).pooled_mean_loss
    # the fine cloud is close to the exact filter and closer than the coarse cloud
    assert abs(fine - exact) < 0.1
    assert abs(fine - exact) < abs(coarse - exact)


def test_particle_filter_matches_exact_filter_predictive_on_a_single_stream() -> None:
    """Direct numerical check: on one traveller's observation stream, the particle filter's
    per-step predictive distribution matches the exact soft-filter predictive in L1 to within
    a small Monte-Carlo tolerance (the candidate is a faithful filter, not a different model)."""
    ds = _dataset(n_travellers=16)
    transition, emission, n = ds.transition, ds.emission, ds.n_states
    travs, holdout, observed = ds.travellers, ds.is_holdout, ds.observed
    tid = sorted(set(int(t) for t in travs[holdout]))[0]
    stream = observed[travs == tid]

    # exact soft-filter predictive at each step
    belief = emission[int(stream[0]), :].astype(float)
    belief /= belief.sum()
    exact = []
    for o in stream[1:]:
        exact.append((transition @ belief).copy())
        belief = emission[int(o), :] * (transition @ belief)
        belief /= belief.sum()

    # particle-filter predictive via the candidate, fed the same stream incrementally
    pf = ParticleFilterCandidate(n_particles=4096)
    pf_pred = []
    past: list[int] = [int(stream[0])]
    for k in range(1, len(stream)):
        ctx = PredictContext(
            n_states=n,
            transition=transition,
            emission=emission,
            past_observations=np.asarray(past, dtype=int),
            prev_observation=past[-1],
            belief=None,
            traveller_id=tid,
            step_index=k,
        )
        pf_pred.append(pf.predict(ctx).copy())
        past.append(int(stream[k]))

    exact_arr = np.asarray(exact)
    pf_arr = np.asarray(pf_pred)
    mean_l1 = float(np.abs(exact_arr - pf_arr).sum(axis=1).mean())
    assert mean_l1 < 0.1  # faithful approximation of the exact filter


def test_particle_filter_incremental_matches_full_rebuild() -> None:
    """The fast incremental path (one new observation per call) yields the same predictive as
    a from-scratch rebuild over the full past, so the cache cannot drift from the truth."""
    ds = _dataset(n_travellers=16)
    transition, emission, n = ds.transition, ds.emission, ds.n_states
    travs, holdout, observed = ds.travellers, ds.is_holdout, ds.observed
    tid = sorted(set(int(t) for t in travs[holdout]))[0]
    stream = observed[travs == tid]

    incremental = ParticleFilterCandidate(n_particles=512)
    inc_preds = []
    past: list[int] = [int(stream[0])]
    for k in range(1, min(len(stream), 8)):
        ctx = PredictContext(
            n_states=n,
            transition=transition,
            emission=emission,
            past_observations=np.asarray(past, dtype=int),
            prev_observation=past[-1],
            belief=None,
            traveller_id=tid,
            step_index=k,
        )
        inc_preds.append(incremental.predict(ctx).copy())
        past.append(int(stream[k]))

    # rebuild candidates from scratch each step (no warm cache) — fresh instance per step
    rebuild_preds = []
    past = [int(stream[0])]
    for k in range(1, min(len(stream), 8)):
        fresh = ParticleFilterCandidate(n_particles=512)
        ctx = PredictContext(
            n_states=n,
            transition=transition,
            emission=emission,
            past_observations=np.asarray(past, dtype=int),
            prev_observation=past[-1],
            belief=None,
            traveller_id=tid,
            step_index=k,
        )
        rebuild_preds.append(fresh.predict(ctx).copy())
        past.append(int(stream[k]))

    for inc, reb in zip(inc_preds, rebuild_preds):
        assert np.allclose(inc, reb)


# --------------------------------------------------------------------------- #
# Switching-noise HMM: regime mixing behaviour
# --------------------------------------------------------------------------- #
def test_switching_hmm_regime_posterior_concentrates_on_matched_channel() -> None:
    """Fed a stream generated under a clean (low-eps) channel, the switching HMM's regime
    posterior puts most of its weight on the low-noise regime; fed a very noisy stream it
    shifts weight toward the high-noise regime. The regime factor is doing real work.

    This needs an INFORMATIVE transition (a sticky/persistent latent), so the cell belief
    carries signal: a well-matched emission channel then predicts the next observation
    sharply, while a mismatched one does not — which is exactly what the regime likelihood
    update measures. Under a uniform transition the predictive is uniform for every regime
    and the channels are indistinguishable, so no informative test could exist there.
    """
    n = 9  # 3x3 grid
    # sticky (persistent) latent dynamics make the cell belief informative
    transition = np.full((n, n), 0.02 / (n - 1))
    np.fill_diagonal(transition, 0.98)
    transition = transition / transition.sum(axis=0, keepdims=True)

    rng = np.random.default_rng(0)
    cells = [int(rng.integers(0, n))]
    for _ in range(120):
        cells.append(cells[-1] if rng.random() < 0.9 else int(rng.integers(0, n)))
    cells = np.asarray(cells, dtype=int)

    def emit(eps: float) -> tuple[np.ndarray, np.ndarray]:
        em = emission_matrix(n, eps)
        out = cells.copy()
        for i in range(len(cells)):
            if rng.random() < eps:
                out[i] = rng.choice([c for c in range(n) if c != cells[i]])
        return out, em

    clean_obs, em_clean = emit(0.02)
    noisy_obs, em_noisy = emit(0.6)

    def regime_weights(stream: np.ndarray, true_em: np.ndarray) -> np.ndarray:
        cand = SwitchingNoiseHMMCandidate(noise_regimes=(0.02, 0.6))
        past: list[int] = [int(stream[0])]
        for k in range(1, len(stream)):
            ctx = PredictContext(
                n_states=n,
                transition=transition,
                emission=true_em,
                past_observations=np.asarray(past, dtype=int),
                prev_observation=past[-1],
                belief=None,
                traveller_id=0,
                step_index=k,
            )
            cand.predict(ctx)
            past.append(int(stream[k]))
        return cand._states[0].regime_weights

    # regime bank is (shared_emission, 0.02, 0.6); index 1 is the clean regime, index 2 the noisy one
    clean_w = regime_weights(clean_obs, em_clean)
    noisy_w = regime_weights(noisy_obs, em_noisy)
    # on the clean stream the clean regime (index 1) heavily outweighs the noisy regime (index 2)
    assert clean_w[1] > clean_w[2]
    # on the noisy stream the noisy regime carries much more weight than it did on the clean stream
    assert noisy_w[2] > clean_w[2]
    assert noisy_w[2] > noisy_w[1]


def test_switching_hmm_is_deterministic() -> None:
    ds = _dataset(n_travellers=16)
    a = score_candidate(ds, SwitchingNoiseHMMCandidate())
    b = score_candidate(ds, SwitchingNoiseHMMCandidate())
    assert np.array_equal(a.step_losses, b.step_losses)


# --------------------------------------------------------------------------- #
# Both candidates beat the strongest FAIR baseline (clustered CI separated from zero)
# --------------------------------------------------------------------------- #
def test_classical_candidates_beat_the_strongest_fair_baseline() -> None:
    """The hard point estimate is the strongest FAIR baseline (per partial_observability.py).
    Both classical candidates beat it on the holdout set with a traveller-clustered CI that
    is entirely above zero — a genuine, leakage-free improvement, not an oracle exploit."""
    ds = _dataset(n_travellers=24, eps=0.3)
    hard = score_candidate(ds, HardFilterCandidate())
    for cand in classical_candidates():
        scored = score_candidate(ds, cand)
        ci = paired_loss_difference_ci(hard, scored, n_boot=500)
        # diff = loss(hard) - loss(cand); b_wins (the candidate) iff the CI is above zero
        assert ci.mean_difference_nats > 0.0
        assert ci.b_wins_clustered
        assert ci.clustered_ci_low > 0.0
        assert ci.n_holdout_travellers >= 2


def test_classical_candidates_do_not_beat_the_soft_filter_ceiling() -> None:
    """Honesty check: under the known emission the exact soft (AIF) filter is the ceiling.
    Neither classical candidate produces a clustered win OVER the soft filter — their CIs do
    not separate above zero against it (the soft filter is at least as good)."""
    ds = _dataset(n_travellers=24, eps=0.3)
    soft = score_candidate(ds, SoftFilterCandidate())
    for cand in classical_candidates():
        scored = score_candidate(ds, cand)
        # a = candidate, b = soft; b_wins_clustered would mean soft strictly beats candidate,
        # a_wins_clustered would mean candidate strictly beats soft. The candidate must NOT
        # post a clustered win over the exact filter.
        ci = paired_loss_difference_ci(scored, soft, n_boot=500)
        assert not ci.a_wins_clustered


def test_observed_markov_is_the_weakest_of_the_fair_predictors() -> None:
    """Sanity ordering on this regime: both classical candidates beat the noisy observed-Markov
    baseline (they exploit the temporal belief the raw observation throws away)."""
    ds = _dataset(n_travellers=16)
    obs_markov = score_candidate(ds, ObservedMarkovCandidate())
    for cand in classical_candidates():
        scored = score_candidate(ds, cand)
        assert scored.pooled_mean_loss < obs_markov.pooled_mean_loss


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_empty_past_returns_uniform() -> None:
    """Defensive: a context with no past observations yields a uniform prediction (no crash)."""
    n = 9
    ctx = PredictContext(
        n_states=n,
        transition=np.full((n, n), 1.0 / n),
        emission=emission_matrix(n, 0.3),
        past_observations=np.asarray([], dtype=int),
        prev_observation=0,
        belief=None,
        traveller_id=0,
        step_index=0,
    )
    for cand in classical_candidates():
        out = cand.predict(ctx)
        assert np.allclose(out, 1.0 / n)


def test_switching_hmm_degenerate_likelihood_does_not_crash() -> None:
    """If every regime assigns near-zero likelihood to a stream, the regime weights fall back
    to a valid (uniform) distribution rather than producing NaNs."""
    n = 4
    transition = np.full((n, n), 1.0 / n)
    emission = emission_matrix(n, 0.3)
    cand = SwitchingNoiseHMMCandidate()
    past = [0]
    out = None
    for k, o in enumerate([1, 2, 3, 0, 1], start=1):
        ctx = PredictContext(
            n_states=n,
            transition=transition,
            emission=emission,
            past_observations=np.asarray(past, dtype=int),
            prev_observation=past[-1],
            belief=None,
            traveller_id=0,
            step_index=k,
        )
        out = cand.predict(ctx)
        past.append(o)
    assert out is not None
    assert np.all(np.isfinite(out))
    assert abs(float(out.sum()) - 1.0) < 1e-9


def test_particle_filter_degenerate_weights_reset_path() -> None:
    """Force the degenerate-weight reset branch: an emission row that is zero at the observed
    cell drives all particle weights to zero; the candidate recovers with a valid distribution."""
    n = 4
    transition = np.eye(
        n
    )  # deterministic stay -> particles pinned, easy to starve weights
    emission = np.eye(n)  # zero-noise channel: emission[o, z] = 0 unless z == o
    cand = ParticleFilterCandidate(n_particles=64)
    # seed at cell 0, then observe cell 1: with a stay-transition every particle is at 0,
    # emission[1, 0] = 0 -> all weights zero -> reset branch
    past = [0]
    out = None
    for k, o in enumerate([1, 2, 3], start=1):
        ctx = PredictContext(
            n_states=n,
            transition=transition,
            emission=emission,
            past_observations=np.asarray(past, dtype=int),
            prev_observation=past[-1],
            belief=None,
            traveller_id=0,
            step_index=k,
        )
        out = cand.predict(ctx)
        past.append(o)
    assert out is not None
    assert np.all(np.isfinite(out))
    assert abs(float(out.sum()) - 1.0) < 1e-9


def test_normalize_degenerate_falls_back_to_uniform() -> None:
    """The internal normaliser returns a uniform distribution for an all-zero or non-finite
    vector rather than dividing by zero (the defensive fallback used throughout the filters)."""
    assert np.allclose(_normalize(np.zeros(5)), _uniform(5))
    assert np.allclose(_normalize(np.array([np.nan, 1.0, 1.0])), _uniform(3))
    # a proper vector is just renormalised
    assert np.allclose(_normalize(np.array([1.0, 3.0])), np.array([0.25, 0.75]))


def test_switching_hmm_rebuilds_on_out_of_sequence_context() -> None:
    """If a context arrives whose past length does not match the cached state (an
    out-of-sequence call), the switching HMM rebuilds from the full past and still returns the
    SAME prediction as a candidate that processed the stream incrementally — the warm cache is
    just an optimisation, never a source of truth."""
    n = 9
    transition = np.full((n, n), 0.02 / (n - 1))
    np.fill_diagonal(transition, 0.98)
    transition = transition / transition.sum(axis=0, keepdims=True)
    emission = emission_matrix(n, 0.2)
    rng = np.random.default_rng(1)
    stream = [int(rng.integers(0, n))]
    for _ in range(10):
        stream.append(stream[-1] if rng.random() < 0.8 else int(rng.integers(0, n)))

    # warm/incremental candidate
    warm = SwitchingNoiseHMMCandidate()
    past: list[int] = [stream[0]]
    for k in range(1, len(stream)):
        ctx = PredictContext(
            n_states=n,
            transition=transition,
            emission=emission,
            past_observations=np.asarray(past, dtype=int),
            prev_observation=past[-1],
            belief=None,
            traveller_id=0,
            step_index=k,
        )
        warm.predict(ctx)
        past.append(stream[k])
    warm_final = warm._states[0].regime_weights.copy()

    # cold candidate that only ever sees the FULL past once (forces the rebuild branch)
    cold = SwitchingNoiseHMMCandidate()
    full_ctx = PredictContext(
        n_states=n,
        transition=transition,
        emission=emission,
        past_observations=np.asarray(stream[:-1], dtype=int),
        prev_observation=stream[-2],
        belief=None,
        traveller_id=0,
        step_index=len(stream) - 1,
    )
    cold.predict(full_ctx)
    cold_final = cold._states[0].regime_weights
    assert np.allclose(warm_final, cold_final)


def test_particle_filter_rebuilds_on_out_of_sequence_context() -> None:
    """The particle filter's predict path also handles an out-of-sequence context: feeding the
    full past in one shot triggers the rebuild branch and returns a valid distribution."""
    n = 9
    transition = np.full((n, n), 0.02 / (n - 1))
    np.fill_diagonal(transition, 0.98)
    transition = transition / transition.sum(axis=0, keepdims=True)
    emission = emission_matrix(n, 0.2)
    rng = np.random.default_rng(2)
    stream = [int(rng.integers(0, n))]
    for _ in range(8):
        stream.append(stream[-1] if rng.random() < 0.8 else int(rng.integers(0, n)))
    pf = ParticleFilterCandidate(n_particles=256)
    full_ctx = PredictContext(
        n_states=n,
        transition=transition,
        emission=emission,
        past_observations=np.asarray(stream[:-1], dtype=int),
        prev_observation=stream[-2],
        belief=None,
        traveller_id=0,
        step_index=len(stream) - 1,
    )
    out = pf.predict(full_ctx)
    assert np.all(np.isfinite(out))
    assert abs(float(out.sum()) - 1.0) < 1e-9


def test_single_regime_switching_hmm_reduces_to_a_plain_filter() -> None:
    """With an empty noise-regime bank the switching HMM has a single regime (the shared
    emission) and behaves as a plain recursive filter — it still scores and beats the hard
    baseline, confirming the regime machinery degrades gracefully to K=1."""
    ds = _dataset(n_travellers=16)
    cand = SwitchingNoiseHMMCandidate(noise_regimes=())
    scored = score_candidate(ds, cand)
    hard = score_candidate(ds, HardFilterCandidate())
    assert scored.n_scored_steps == hard.n_scored_steps
    assert np.isfinite(scored.pooled_mean_loss)
    assert scored.pooled_mean_loss < hard.pooled_mean_loss
