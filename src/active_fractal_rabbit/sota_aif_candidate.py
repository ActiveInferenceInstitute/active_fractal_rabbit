"""An improved, fair, leakage-free active-inference candidate for the SOTA spine.

This module builds a stronger active-inference predictor against the
``sota_program.py`` spine. It starts from the genuine soft belief filter of
``partial_observability.py`` — marginalize the running belief through the shared
learned transition, ``T @ b`` — and adds two *fair* improvements that can lower the
prequential next-latent-cell loss under partial observability WITHOUT any peeking:

* **(a) Belief precision-weighting.** Under a noisy emission the running belief is
  diffuse; a single temperature ``gamma`` reshapes it before marginalizing
  (``b ** gamma`` renormalised, i.e. ``softmax(gamma * log b)``). ``gamma > 1`` sharpens a
  belief the agent trusts; ``gamma < 1`` flattens an over-confident one. Crucially
  ``gamma`` is selected on the TRAIN travellers ONLY (:func:`fit_aif_candidate` scores a
  small grid on a train-internal holdout and never touches the evaluation holdout), so the
  knob is honest. At the positive-control identity emission the belief is already a point
  mass and ``gamma`` cannot change a point mass, so the candidate collapses exactly onto
  the soft/hard/observed baselines (they coincide there).

* **(b) Calibrated visited-cell return prior.** Travellers revisit cells (preferential
  return), so a per-traveller prior that up-weights ALREADY-OBSERVED cells helps predict
  the next latent cell. This reuses the exact additive-boost form of
  ``fair_inference._markov_loss`` (``pred + boost * seen * pred.sum() / |seen|``), where
  ``seen`` is built ONLY from the candidate's own past OBSERVED cells (the leakage-free
  ``past_observations`` window ending at ``o_{i-1}``) — never the latent target. The boost
  weight is calibrated on TRAIN travellers ONLY, jointly with ``gamma``.

Both knobs are leakage-free by construction: :class:`PrecisionReturnAIFCandidate.predict`
reads only the fields of :class:`PredictContext` (all of which are information through step
``i - 1``), and the calibration in :func:`fit_aif_candidate` runs a nested train/validation
split *inside the train travellers* — the evaluation holdout the spine scores against is
never seen during fitting. Folding the candidate into the spine's claim gate (the
``sota_benchmark`` phase) is left to the integration phase; this module only provides the
candidate and its honest fitter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sota_program import (
    PartialObservabilityDataset,
    PredictContext,
    ScoredCandidate,
    score_candidate,
)

SOTA_AIF_CANDIDATE_SCHEMA = "active-fractal-rabbit-sota-aif-candidate-v1"

#: Default grids searched by :func:`fit_aif_candidate`. ``gamma == 1`` and ``boost == 0``
#: are included so the fitter can always fall back to the plain soft filter when neither
#: knob helps on the train split (an honest no-improvement outcome).
DEFAULT_GAMMA_GRID: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
DEFAULT_BOOST_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
DEFAULT_ADAPTATION_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
ADAPTATION_WINDOW: int = 16


def _precision_weight_belief(belief: np.ndarray, gamma: float) -> np.ndarray:
    """Apply a precision temperature to a belief: ``b ** gamma`` renormalised.

    Equivalent to ``softmax(gamma * log b)``. ``gamma == 1`` is the identity. A point-mass
    belief is invariant for every ``gamma`` (so the positive-control identity emission, where
    the belief is already a point mass, is untouched). Numerically guarded so a zero entry
    cannot produce ``0 ** 0`` or a non-finite result.
    """

    b = np.clip(np.asarray(belief, dtype=float), 0.0, None)
    total = b.sum()
    if total <= 0.0:
        return np.full(b.shape[0], 1.0 / b.shape[0])
    b = b / total
    if gamma == 1.0:
        return b
    # b ** gamma with a guard: log(0) is handled by leaving zeros at zero.
    out = np.zeros_like(b)
    pos = b > 0.0
    out[pos] = np.exp(gamma * np.log(b[pos]))
    s = out.sum()
    if s <= 0.0:  # underflow guard: collapse back to the (already normalised) belief
        return b
    return out / s


def _visited_from_observations(
    past_observations: np.ndarray, n_states: int
) -> np.ndarray:
    """Boolean mask of cells OBSERVED so far for this traveller (leakage-free).

    Built purely from ``past_observations`` (the observed cells through ``o_{i-1}``), never
    the latent target. Mirrors the visited-cell feature of ``fair_inference._markov_loss``,
    but over the candidate's own observed history available in the context.
    """

    seen = np.zeros(n_states, dtype=bool)
    if past_observations.size:
        idx = np.asarray(past_observations, dtype=int)
        idx = idx[(idx >= 0) & (idx < n_states)]
        seen[idx] = True
    return seen


def _online_transition_from_observations(
    context: PredictContext, adaptation: float
) -> np.ndarray:
    """Shrink a leakage-free online expected-transition estimate toward the train model.

    Past noisy observations are converted to emission posteriors under a uniform prior.
    Consecutive posterior outer products provide fractional transition counts. The result
    is mixed with the train-only transition; ``adaptation=0`` is exactly the fixed model.
    """

    if adaptation <= 0.0 or context.past_observations.size < 2:
        return context.transition
    recent = np.asarray(context.past_observations[-ADAPTATION_WINDOW:], dtype=int)
    rows = context.emission[recent, :]
    totals = rows.sum(axis=1, keepdims=True)
    post = np.divide(
        rows,
        totals,
        out=np.full_like(rows, 1.0 / context.n_states),
        where=totals > 0.0,
    )
    counts = np.zeros_like(context.transition, dtype=float)
    for prev, nxt in zip(post[:-1], post[1:]):
        counts += np.outer(nxt, prev)
    column_totals = counts.sum(axis=0, keepdims=True)
    local = np.divide(
        counts,
        column_totals,
        out=context.transition.copy(),
        where=column_totals > 0.0,
    )
    weight = float(np.clip(adaptation, 0.0, 1.0))
    mixed = (1.0 - weight) * context.transition + weight * local
    return mixed / mixed.sum(axis=0, keepdims=True)


@dataclass
class PrecisionReturnAIFCandidate:
    """Improved AIF predictor: precision-weighted soft belief + calibrated return prior.

    ``predict`` marginalizes a precision-weighted running belief through the shared
    transition (``T @ softmax(gamma*log b)``) and then adds a per-traveller visited-cell
    return boost computed from the leakage-free ``past_observations``. With ``gamma == 1``
    and ``boost == 0`` it is byte-identical to the plain :class:`SoftFilterCandidate` (the
    AIF reference), so any improvement comes only from the two honest knobs.

    The hyperparameters are normally produced by :func:`fit_aif_candidate` on TRAIN
    travellers only; they are stored here as plain attributes so the candidate is a pure,
    self-contained predictor at scoring time.
    """

    gamma: float = 1.0
    boost: float = 0.0
    adaptation: float = 0.0
    name: str = "precision_return_aif"

    def predict(self, context: PredictContext) -> np.ndarray:
        n = context.n_states
        belief = context.belief
        if belief is None:
            return np.full(n, 1.0 / n)
        weighted = _precision_weight_belief(belief, self.gamma)
        transition = _online_transition_from_observations(context, self.adaptation)
        pred = transition @ weighted
        if self.boost > 0.0:
            seen = _visited_from_observations(context.past_observations, n)
            n_seen = int(seen.sum())
            if n_seen > 0:
                pred = pred + self.boost * seen * pred.sum() / n_seen
        pred = np.clip(pred, 0.0, None)
        total = pred.sum()
        if total <= 0.0:
            return np.full(n, 1.0 / n)
        return pred / total


@dataclass(frozen=True)
class AIFFitResult:
    """The outcome of an honest TRAIN-only hyperparameter search.

    ``candidate`` is the fitted :class:`PrecisionReturnAIFCandidate`. ``train_grid`` records
    the train-validation pooled loss for every ``(gamma, boost)`` searched, so the selection
    is fully auditable (and a test can confirm the winner is the train-internal argmin, never
    chosen on the evaluation holdout).
    """

    candidate: PrecisionReturnAIFCandidate
    gamma: float
    boost: float
    adaptation: float
    train_val_loss: float
    soft_baseline_train_val_loss: float
    train_grid: tuple[dict[str, float], ...] = field(default_factory=tuple)


def _train_only_dataset(
    dataset: PartialObservabilityDataset,
) -> PartialObservabilityDataset:
    """A view of ``dataset`` whose holdout is an INNER split of the TRAIN travellers only.

    The evaluation holdout travellers are dropped from scoring entirely (their ``is_holdout``
    becomes ``False`` is not enough — they must not be scored, so we mark only an inner slice
    of the train travellers as the validation holdout and force the original holdout to never
    score). Concretely: take the TRAIN travellers, deterministically reserve a fraction of
    them as the inner validation set, and build a dataset whose ``is_holdout`` is True only on
    that inner validation set. The transition is re-estimated from INNER-training
    travellers only; otherwise inner-validation latent transitions would leak into the
    model used to select hyperparameters.

    This guarantees :func:`fit_aif_candidate` never reads the evaluation holdout: the fitter
    scores candidates on inner-validation travellers that are a strict subset of TRAIN.
    """

    travellers = dataset.travellers
    is_holdout = dataset.is_holdout
    train_travs = np.unique(travellers[~is_holdout])
    if train_travs.size < 2:
        raise ValueError(
            "need at least 2 train travellers to fit on a train-internal split"
        )
    # Deterministic inner validation slice: reserve ~40% of train travellers (at least one),
    # chosen by a fixed-seed permutation so the fit is reproducible and split-stable.
    rng = np.random.default_rng(20260616)
    order = rng.permutation(train_travs)
    n_val = max(1, int(round(0.4 * order.size)))
    val_travs = set(int(t) for t in order[:n_val])
    inner_holdout = np.array([int(t) in val_travs for t in travellers], dtype=bool)
    inner_train = (~is_holdout) & (~inner_holdout)
    counts = np.full(
        (dataset.n_states, dataset.n_states),
        dataset.transition_smoothing,
        dtype=float,
    )
    last: int | None = None
    for i in range(len(dataset.latent)):
        tid = int(travellers[i])
        if dataset.is_first[i] or tid != last:
            last = tid
            continue
        if inner_train[i] and inner_train[i - 1]:
            counts[int(dataset.latent[i]), int(dataset.latent[i - 1])] += 1.0
    inner_transition = counts / counts.sum(axis=0, keepdims=True)
    from dataclasses import replace

    return replace(
        dataset,
        is_holdout=inner_holdout,
        transition=inner_transition,
        split={
            **dataset.split,
            "inner_validation_travellers": sorted(val_travs),
            "inner_transition_source": "inner_train_travellers_only",
        },
    )


def fit_aif_candidate(
    dataset: PartialObservabilityDataset,
    *,
    gamma_grid: tuple[float, ...] = DEFAULT_GAMMA_GRID,
    boost_grid: tuple[float, ...] = DEFAULT_BOOST_GRID,
    adaptation_grid: tuple[float, ...] = DEFAULT_ADAPTATION_GRID,
) -> AIFFitResult:
    """Select ``(gamma, boost)`` on TRAIN travellers ONLY and return the fitted candidate.

    A nested train/validation split is carved out of the TRAIN travellers (see
    :func:`_train_only_dataset`); every ``(gamma, boost)`` on the grid is scored on that
    inner validation set with the leakage-free :func:`score_candidate`, and the lowest-loss
    pair is chosen. The EVALUATION holdout the spine scores against is never read here, so the
    knob is honest. Ties break toward the plain soft filter (``gamma == 1``, ``boost == 0``)
    by grid ordering, so the fitter only departs from the baseline when the train data
    actually rewards it.
    """

    if not gamma_grid or not all(np.isfinite(value) and value > 0 for value in gamma_grid):
        raise ValueError("gamma_grid must contain finite positive values")
    if not boost_grid or not all(np.isfinite(value) and value >= 0 for value in boost_grid):
        raise ValueError("boost_grid must contain finite nonnegative values")
    if not adaptation_grid or not all(
        np.isfinite(value) and 0.0 <= value <= 1.0 for value in adaptation_grid
    ):
        raise ValueError("adaptation_grid values must be finite and in [0, 1]")

    gamma_grid = (1.0, *(value for value in gamma_grid if value != 1.0))
    boost_grid = (0.0, *(value for value in boost_grid if value != 0.0))
    adaptation_grid = (
        0.0,
        *(value for value in adaptation_grid if value != 0.0),
    )

    inner = _train_only_dataset(dataset)

    grid: list[dict[str, float]] = []
    best: tuple[float, float, float, float] | None = None
    soft_val_loss: float | None = None
    # Stage one: select the established precision/return pair with fixed transitions.
    # Stage two: tune only adaptation around that selected pair. This avoids an expensive
    # post-hoc Cartesian search while keeping every choice inside the nested train split.
    for gamma in gamma_grid:
        for boost in boost_grid:
            cand = PrecisionReturnAIFCandidate(
                gamma=float(gamma), boost=float(boost), adaptation=0.0
            )
            loss = float(score_candidate(inner, cand).pooled_mean_loss)
            grid.append(
                {
                    "gamma": float(gamma),
                    "boost": float(boost),
                    "adaptation": 0.0,
                    "train_val_loss": loss,
                }
            )
            if gamma == 1.0 and boost == 0.0:
                soft_val_loss = loss
            if best is None or loss < best[0] - 1e-12:
                best = (loss, float(gamma), float(boost), 0.0)

    assert best is not None
    base_gamma, base_boost = best[1], best[2]
    for adaptation in adaptation_grid:
        if adaptation == 0.0:
            continue
        cand = PrecisionReturnAIFCandidate(
            gamma=base_gamma, boost=base_boost, adaptation=float(adaptation)
        )
        loss = float(score_candidate(inner, cand).pooled_mean_loss)
        grid.append(
            {
                "gamma": base_gamma,
                "boost": base_boost,
                "adaptation": float(adaptation),
                "train_val_loss": loss,
            }
        )
        if loss < best[0] - 1e-12:
            best = (loss, base_gamma, base_boost, float(adaptation))

    assert best is not None  # the grid always contains (1.0, 0.0)
    if soft_val_loss is None:  # pragma: no cover - (1.0, 0.0) is always in the grid
        soft_val_loss = best[0]

    candidate = PrecisionReturnAIFCandidate(
        gamma=best[1], boost=best[2], adaptation=best[3]
    )
    return AIFFitResult(
        candidate=candidate,
        gamma=best[1],
        boost=best[2],
        adaptation=best[3],
        train_val_loss=best[0],
        soft_baseline_train_val_loss=float(soft_val_loss),
        train_grid=tuple(grid),
    )


def score_fitted_aif_candidate(dataset: PartialObservabilityDataset) -> ScoredCandidate:
    """Convenience: fit on TRAIN, then score the fitted candidate on the spine's holdout.

    The fitting and scoring use disjoint traveller sets (TRAIN-internal validation vs the
    evaluation holdout), so the reported holdout loss is leakage-free with respect to the
    hyperparameter selection.
    """

    fit = fit_aif_candidate(dataset)
    return score_candidate(dataset, fit.candidate)
