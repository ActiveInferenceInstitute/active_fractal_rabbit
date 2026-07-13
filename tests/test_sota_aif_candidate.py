"""Tests for the improved active-inference candidate (precision + return prior).

No mocks; real deterministic synthetic sequences and real computation against the
``sota_program.py`` spine. Covers:
* the precision weighting (identity at gamma=1, point-mass invariance, sharpening),
* the leakage-free visited-cell return boost (built only from past observations),
* the POSITIVE CONTROL: under an identity (clean) emission the candidate collapses onto
  the soft/hard/observed baselines,
* that ``(gamma, boost)`` is selected on the TRAIN travellers ONLY (never the evaluation
  holdout), and the selected pair is the train-internal argmin,
* that the fitted candidate scores leakage-free on the holdout and cannot beat a peeking
  oracle bound.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.sota_aif_candidate import (
    DEFAULT_BOOST_GRID,
    DEFAULT_GAMMA_GRID,
    PrecisionReturnAIFCandidate,
    _precision_weight_belief,
    _online_transition_from_observations,
    _train_only_dataset,
    _visited_from_observations,
    fit_aif_candidate,
    score_fitted_aif_candidate,
)
from active_fractal_rabbit.sota_program import (
    Candidate,
    HardFilterCandidate,
    ObservedMarkovCandidate,
    PredictContext,
    SoftFilterCandidate,
    build_partial_observability_dataset,
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
# Precision weighting
# --------------------------------------------------------------------------- #
def test_precision_weight_identity_at_gamma_one() -> None:
    b = np.array([0.1, 0.2, 0.3, 0.4])
    out = _precision_weight_belief(b, 1.0)
    assert np.allclose(out, b)
    assert abs(out.sum() - 1.0) < 1e-12


def test_precision_weight_invariant_on_point_mass() -> None:
    """A point-mass belief is unchanged for every gamma — the basis of the positive control."""
    b = np.zeros(5)
    b[2] = 1.0
    for gamma in (0.25, 0.5, 1.0, 2.0, 5.0):
        out = _precision_weight_belief(b, gamma)
        assert np.allclose(out, b)


def test_precision_weight_sharpens_and_flattens() -> None:
    b = np.array([0.1, 0.2, 0.3, 0.4])
    sharp = _precision_weight_belief(b, 3.0)
    flat = _precision_weight_belief(b, 0.3)
    # sharpening raises the dominant mass; flattening lowers it toward uniform
    assert sharp[3] > b[3]
    assert flat[3] < b[3]
    assert abs(sharp.sum() - 1.0) < 1e-12
    assert abs(flat.sum() - 1.0) < 1e-12
    # flattening moves the belief closer to uniform than sharpening
    uniform = np.full(4, 0.25)
    assert np.abs(flat - uniform).sum() < np.abs(sharp - uniform).sum()


def test_precision_weight_handles_zero_and_degenerate_input() -> None:
    # all-zero belief => uniform fallback
    out = _precision_weight_belief(np.zeros(4), 2.0)
    assert np.allclose(out, np.full(4, 0.25))
    # belief with zeros keeps them at zero and renormalises the rest
    b = np.array([0.0, 0.5, 0.0, 0.5])
    out = _precision_weight_belief(b, 2.0)
    assert out[0] == 0.0 and out[2] == 0.0
    assert abs(out.sum() - 1.0) < 1e-12


# --------------------------------------------------------------------------- #
# Visited-cell feature (leakage-free)
# --------------------------------------------------------------------------- #
def test_visited_mask_from_past_observations_only() -> None:
    past = np.array([1, 3, 1, 0])
    seen = _visited_from_observations(past, 6)
    assert seen.dtype == bool
    assert set(np.flatnonzero(seen).tolist()) == {0, 1, 3}
    # empty history => nothing visited
    assert not _visited_from_observations(np.array([], dtype=int), 6).any()
    # out-of-range indices are ignored defensively
    seen2 = _visited_from_observations(np.array([0, 99, -1, 5]), 6)
    assert set(np.flatnonzero(seen2).tolist()) == {0, 5}


# --------------------------------------------------------------------------- #
# Candidate equivalence and protocol conformance
# --------------------------------------------------------------------------- #
def test_default_candidate_equals_soft_filter_everywhere() -> None:
    """gamma=1, boost=0 must be byte-identical to the plain soft filter (the AIF reference)."""
    ds = _dataset()
    aif = PrecisionReturnAIFCandidate(gamma=1.0, boost=0.0)
    soft = SoftFilterCandidate()
    a = score_candidate(ds, aif)
    b = score_candidate(ds, soft)
    assert np.allclose(a.step_losses, b.step_losses)
    assert abs(a.pooled_mean_loss - b.pooled_mean_loss) < 1e-12


def test_candidate_implements_protocol_and_scores() -> None:
    ds = _dataset()
    cand = PrecisionReturnAIFCandidate(gamma=2.0, boost=0.5)
    assert isinstance(cand, Candidate)
    scored = score_candidate(ds, cand)
    assert scored.n_scored_steps > 0
    assert np.isfinite(scored.pooled_mean_loss)


def test_online_transition_adaptation_is_normalized_and_history_sensitive() -> None:
    ds = _dataset()
    context = PredictContext(
        n_states=ds.n_states,
        transition=ds.transition,
        emission=ds.emission,
        past_observations=ds.observed[:8],
        prev_observation=int(ds.observed[7]),
        belief=np.full(ds.n_states, 1.0 / ds.n_states),
        traveller_id=0,
        step_index=8,
    )
    fixed = _online_transition_from_observations(context, 0.0)
    adaptive = _online_transition_from_observations(context, 1.0)
    assert np.allclose(fixed, ds.transition)
    assert np.allclose(adaptive.sum(axis=0), 1.0)
    assert np.all(adaptive >= 0.0)
    assert not np.allclose(adaptive, fixed)


# --------------------------------------------------------------------------- #
# POSITIVE CONTROL: identity emission collapses onto the baselines
# --------------------------------------------------------------------------- #
def test_positive_control_identity_emission_collapses_to_baselines() -> None:
    """Under a clean (eps=0) emission the belief becomes a point mass at the observed cell.

    Precision weighting cannot change a point mass, and with a clean channel the observed
    cell IS the latent cell, so the precision-weighted soft predictor coincides exactly with
    the hard (MAP) predictor and the observed-Markov predictor. With boost=0 the improved
    candidate is therefore numerically identical to all three baselines.
    """
    ds = _dataset(eps=0.0)
    # belief is a point mass under the clean channel
    aif = PrecisionReturnAIFCandidate(gamma=3.0, boost=0.0)
    soft = score_candidate(ds, SoftFilterCandidate())
    hard = score_candidate(ds, HardFilterCandidate())
    obs = score_candidate(ds, ObservedMarkovCandidate())
    fitted = score_candidate(ds, aif)
    # precision can't reshape a point mass => identical to soft
    assert np.allclose(fitted.step_losses, soft.step_losses)
    # under a clean channel soft == hard == observed-Markov (all point estimates agree)
    assert np.allclose(soft.step_losses, hard.step_losses)
    assert np.allclose(soft.step_losses, obs.step_losses)


# --------------------------------------------------------------------------- #
# Honest fitting: (gamma, boost) chosen on TRAIN only
# --------------------------------------------------------------------------- #
def test_fit_returns_grid_argmin_and_includes_soft_fallback() -> None:
    ds = _dataset(n_travellers=24, eps=0.3)
    fit = fit_aif_candidate(ds)
    # the grid always contains the plain soft filter (1.0, 0.0)
    grid_points = {
        (g["gamma"], g["boost"], g["adaptation"]) for g in fit.train_grid
    }
    assert (1.0, 0.0, 0.0) in grid_points
    # the selected pair is the train-internal argmin over the recorded grid
    best_loss = min(g["train_val_loss"] for g in fit.train_grid)
    assert abs(fit.train_val_loss - best_loss) < 1e-12
    assert (
        abs(
            next(
                g["train_val_loss"]
                for g in fit.train_grid
                if g["gamma"] == fit.gamma and g["boost"] == fit.boost
                and g["adaptation"] == fit.adaptation
            )
            - fit.train_val_loss
        )
        < 1e-12
    )
    # the fitted candidate never does WORSE than the soft fallback on the train-val split
    assert fit.train_val_loss <= fit.soft_baseline_train_val_loss + 1e-12


def test_fit_only_sees_train_travellers_not_evaluation_holdout() -> None:
    """The inner validation set used to choose (gamma, boost) is a strict subset of the
    TRAIN travellers; the evaluation holdout travellers are NEVER scored during fitting."""
    ds = _dataset(n_travellers=24)
    inner = _train_only_dataset(ds)
    eval_holdout_travs = set(int(t) for t in ds.travellers[ds.is_holdout])
    inner_val_travs = set(int(t) for t in ds.travellers[inner.is_holdout])
    train_travs = set(int(t) for t in ds.travellers[~ds.is_holdout])
    # the inner validation set is non-empty, a strict subset of TRAIN, and disjoint from
    # the evaluation holdout
    assert inner_val_travs
    assert inner_val_travs.issubset(train_travs)
    assert inner_val_travs.isdisjoint(eval_holdout_travs)
    assert inner.split["inner_transition_source"] == "inner_train_travellers_only"


def test_inner_transition_excludes_inner_validation_latents() -> None:
    """Negative control for inner-validation leakage through transition fitting."""

    ds = _dataset(n_travellers=24)
    inner = _train_only_dataset(ds)
    corrupted_latent = ds.latent.copy()
    corrupted_latent[inner.is_holdout] = (
        corrupted_latent[inner.is_holdout] + 1
    ) % ds.n_states
    rebuilt = _train_only_dataset(replace(ds, latent=corrupted_latent))
    assert np.allclose(inner.transition, rebuilt.transition)


def test_fit_selection_is_invariant_to_evaluation_holdout_labels() -> None:
    """Hard proof the selection ignores the evaluation holdout: scrambling the holdout's
    TRUE latent cells leaves the chosen (gamma, boost) unchanged, because the fitter never
    reads holdout data.
    """
    ds = _dataset(n_travellers=24)
    fit_clean = fit_aif_candidate(ds)

    # corrupt the latent + observed cells on the evaluation holdout only
    rng = np.random.default_rng(123)
    latent = ds.latent.copy()
    observed = ds.observed.copy()
    mask = ds.is_holdout
    latent[mask] = rng.integers(0, ds.n_states, int(mask.sum()))
    observed[mask] = rng.integers(0, ds.n_states, int(mask.sum()))
    corrupted = replace(ds, latent=latent, observed=observed)

    fit_corrupt = fit_aif_candidate(corrupted)
    assert (fit_clean.gamma, fit_clean.boost, fit_clean.adaptation) == (
        fit_corrupt.gamma,
        fit_corrupt.boost,
        fit_corrupt.adaptation,
    )
    assert abs(fit_clean.train_val_loss - fit_corrupt.train_val_loss) < 1e-12


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gamma_grid": (0.0,)}, "finite positive"),
        ({"gamma_grid": (float("nan"),)}, "finite positive"),
        ({"boost_grid": (-0.1,)}, "finite nonnegative"),
        ({"adaptation_grid": (1.1,)}, "in \\[0, 1\\]"),
        ({"adaptation_grid": (float("nan"),)}, "in \\[0, 1\\]"),
    ],
)
def test_fit_rejects_invalid_hyperparameter_domains(
    kwargs: dict[str, tuple[float, ...]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_aif_candidate(_dataset(n_travellers=12), **kwargs)


# --------------------------------------------------------------------------- #
# Leakage-freeness of the fitted candidate on the holdout
# --------------------------------------------------------------------------- #
def test_fitted_candidate_scores_and_is_not_oracle() -> None:
    """The fitted candidate scores on the evaluation holdout and stays far from oracle (~0)
    loss — it predicts from past data only, never the true next cell."""
    ds = _dataset(n_travellers=24, eps=0.3)
    scored = score_fitted_aif_candidate(ds)
    assert scored.n_scored_steps > 0
    assert np.isfinite(scored.pooled_mean_loss)
    # a genuine oracle would score ~0; an honest predictor under noise is well above that
    assert scored.pooled_mean_loss > 0.1


def test_candidate_cannot_peek_through_context() -> None:
    """Even an adversarial precision/boost setting cannot reach oracle loss, because the
    context exposes only past observations (the spine's leakage guard applies to us too)."""
    ds = _dataset(n_travellers=24, eps=0.3)
    captured: list[PredictContext] = []
    base = PrecisionReturnAIFCandidate(gamma=5.0, boost=1.0)

    class _Capturing:
        name = "capturing_aif"

        def predict(self, ctx: PredictContext) -> np.ndarray:
            captured.append(ctx)
            return base.predict(ctx)

    scored = score_candidate(ds, _Capturing())
    assert scored.pooled_mean_loss > 0.1
    # the context never contains the current/true cell: past_observations ends at o_{i-1}
    for ctx in captured:
        assert ctx.past_observations[-1] == ctx.prev_observation


def test_grids_contain_baseline_fallback_points() -> None:
    assert 1.0 in DEFAULT_GAMMA_GRID
    assert 0.0 in DEFAULT_BOOST_GRID


def test_fit_requires_enough_train_travellers() -> None:
    ds = _dataset(n_travellers=2)
    # with 2 travellers the split may leave <2 train travellers; the inner split must fail loud
    train = set(int(t) for t in ds.travellers[~ds.is_holdout])
    if len(train) < 2:
        with pytest.raises(ValueError):
            _train_only_dataset(ds)
    else:  # pragma: no cover - depends on the locked split; both branches are valid
        _train_only_dataset(ds)
