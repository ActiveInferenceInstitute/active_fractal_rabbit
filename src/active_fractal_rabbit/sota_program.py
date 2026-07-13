"""Shared spine for a state-of-the-art benchmark in the partial-observability regime.

This module is the *foundation* that the SOTA program builds on. It does not assert
any state-of-the-art claim; it provides the honest, leakage-free machinery a defensible
claim would require, in the one regime where active inference has a fair edge: genuine
partial observability under a known noisy emission channel (see
``partial_observability.py`` for why the equal-observation default leaves no room for an
AIF advantage).

It provides four pieces, each reusing the existing, tested helpers rather than
reinventing them:

1. **A high-power synthetic data generator** (:func:`build_partial_observability_dataset`)
   — many travellers (default ~80) producing latent FractalRabbit cell sequences (via
   ``generate_synthetic_sequence`` from ``power_analysis.py``) observed through the SAME
   known symmetric emission used in ``partial_observability.py``: the observation equals
   the latent cell with probability ``1 - eps`` and is otherwise uniform over the other
   ``n - 1`` cells (drawn over the NON-true cells, so the channel is correctly specified).
   Locked train/holdout traveller splits come from ``lock_traveller_split``.

2. **A leakage-free prequential scorer** (:func:`score_candidate`) — given a CANDIDATE
   that maps (past observations, shared learned transition, emission) -> a per-step
   next-latent-cell predictive distribution, returns per-traveller and pooled next-cell
   negative-log-loss on the HOLDOUT travellers. The prediction at step ``i`` is formed
   from observations through ``i - 1`` ONLY; the target is the TRUE latent cell at ``i``.
   The harness never hands the candidate the current observation or the true target, so a
   candidate physically cannot peek.

3. **A candidate interface** (:class:`Candidate` / :class:`PredictContext`) — a minimal,
   explicit protocol so the baselines and the AIF candidate in the next phase plug in
   uniformly. A candidate is anything with a ``name`` and a
   ``predict(context: PredictContext) -> np.ndarray`` returning a length-``n`` distribution
   over the next latent cell. The context exposes exactly the leakage-free inputs.

4. **A traveller-clustered bootstrap** (:func:`paired_loss_difference_ci`) — resamples
   whole travellers (the correlated unit) to interval the per-step loss DIFFERENCE between
   two candidates scored on the same holdout steps. A per-step bootstrap is INVALID here
   because steps within a traveller are correlated; this delegates to the tested
   ``traveller_clustered_bootstrap_ci``.

Everything is synthetic-only and non-operational. Beating the strongest fair baseline on
this spine is a fair-frontier characterization, never a real-mobility or deployment claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .partial_observability import DEFAULT_EPS_LEVELS  # noqa: F401  (re-exported regime grid)
from .partial_observability import realize_noisy_observations
from .power_analysis import (
    generate_synthetic_sequence,
    traveller_clustered_bootstrap_stats,
)
from .sota_benchmark import _ece, lock_traveller_split
from .types import InferenceConfig, ObservationSequence

SOTA_PROGRAM_SCHEMA = "active-fractal-rabbit-sota-program-v1"

#: Default high-power traveller count. Large enough that the traveller-clustered
#: bootstrap has many clusters (so an advantage is genuinely intervalable), while
#: staying deterministic and fast.
DEFAULT_N_TRAVELLERS = 80


def emission_matrix(n_states: int, eps: float) -> np.ndarray:
    """The known symmetric emission channel shared by generator and inference.

    ``P(o = z) = 1 - eps`` and ``P(o = other) = eps / (n - 1)`` for each of the ``n - 1``
    non-true cells. This is byte-for-byte the channel used in ``partial_observability.py``
    so a candidate's emission is correctly specified by construction.
    """

    if n_states < 1:
        raise ValueError("need at least one spatial state")
    if not (0.0 <= eps <= 1.0):
        raise ValueError("emission flip probability eps must be in [0, 1]")
    off = eps / (n_states - 1) if n_states > 1 else 0.0
    emission = np.full((n_states, n_states), off, dtype=float)
    np.fill_diagonal(emission, 1.0 - eps)
    return emission


@dataclass(frozen=True)
class PartialObservabilityDataset:
    """A locked, high-power partial-observability benchmark instance.

    ``observed`` is the noisy emission of the TRUE ``latent`` cells (same length and
    index order as the underlying sequence). ``is_holdout`` marks the steps belonging to
    holdout travellers. ``transition`` is the latent transition matrix learned from TRAIN
    travellers only (column-stochastic, ``transition[next, prev]``), shared by every
    candidate so the comparison is fair. ``emission`` is the known channel.
    """

    sequence: ObservationSequence
    latent: np.ndarray
    observed: np.ndarray
    is_first: np.ndarray
    travellers: np.ndarray
    is_holdout: np.ndarray
    n_states: int
    eps: float
    transition: np.ndarray
    emission: np.ndarray
    split: dict[str, object]
    transition_smoothing: float


def build_partial_observability_dataset(
    inference_config: InferenceConfig,
    *,
    n_travellers: int = DEFAULT_N_TRAVELLERS,
    eps: float = 0.3,
    smoothing: float = 0.5,
    holdout_fraction: float = 0.4,
    gen_seed: int = 2026,
    split_seed: int = 20260616,
    noise_seed: int = 0,
) -> PartialObservabilityDataset:
    """Generate a high-power partial-observability dataset with a locked traveller split.

    Latent dynamics come from ``generate_synthetic_sequence`` (the tested FractalRabbit
    synthetic generator); the observation channel is the known symmetric emission of
    ``partial_observability.py``. The shared latent transition is learned from TRAIN
    travellers ONLY (no holdout leakage). Fully deterministic given the seeds.
    """

    sequence = generate_synthetic_sequence(
        n_travellers, inference_config, seed=gen_seed
    )
    return build_partial_observability_dataset_from_sequence(
        sequence,
        eps=eps,
        smoothing=smoothing,
        holdout_fraction=holdout_fraction,
        split_seed=split_seed,
        noise_seed=noise_seed,
        metadata={
            "lane_family": "fractalrabbit_synthetic",
            "generator_seed": int(gen_seed),
            "n_travellers": int(n_travellers),
        },
    )


def build_partial_observability_dataset_from_sequence(
    sequence: ObservationSequence,
    *,
    eps: float = 0.3,
    smoothing: float = 0.5,
    holdout_fraction: float = 0.4,
    split_seed: int = 20260616,
    noise_seed: int = 0,
    metadata: dict[str, object] | None = None,
) -> PartialObservabilityDataset:
    """Assemble a locked partial-observability dataset from a latent cell sequence.

    This is the common construction used by every SOTA-program lane: the traveller split is
    locked on traveller identity, noisy observations are drawn through the same known
    symmetric emission channel, and the shared transition matrix is learned from TRAIN
    travellers only. New synthetic lanes can vary the latent generator while preserving the
    scoring contract byte-for-byte.
    """

    latent = np.asarray(sequence.spatial_states, dtype=int)
    is_first = np.asarray(sequence.is_first_observation, dtype=bool)
    travellers = np.asarray([r.traveller_id for r in sequence.observations], dtype=int)
    n = int(sequence.num_spatial_states)
    if np.unique(travellers).size < 2:
        raise ValueError("need at least 2 travellers for a locked split")

    split = lock_traveller_split(
        sequence, holdout_fraction=holdout_fraction, seed=split_seed
    )
    is_holdout = np.asarray(split["is_holdout"], dtype=bool)

    emission = emission_matrix(n, eps)
    observed = _emit_observations(latent, n, eps=eps, noise_seed=noise_seed)

    # Latent transition learned from TRAIN travellers only (column-stochastic). Mirrors
    # the train-only estimation in partial_observability / power_analysis so a candidate
    # never sees holdout statistics through the shared model.
    counts = np.full((n, n), smoothing, dtype=float)
    last: int | None = None
    for i in range(len(latent)):
        if is_first[i] or travellers[i] != last:
            last = int(travellers[i])
            continue
        if not is_holdout[i]:
            counts[latent[i], latent[i - 1]] += 1.0
    transition = counts / counts.sum(axis=0, keepdims=True)

    split_payload = {k: v for k, v in split.items() if k != "is_holdout"}
    if metadata:
        split_payload["lane_metadata"] = dict(metadata)

    return PartialObservabilityDataset(
        sequence=sequence,
        latent=latent,
        observed=observed,
        is_first=is_first,
        travellers=travellers,
        is_holdout=is_holdout,
        n_states=n,
        eps=float(eps),
        transition=transition,
        emission=emission,
        split=split_payload,
        transition_smoothing=float(smoothing),
    )


def _emit_observations(
    latent: np.ndarray, n: int, *, eps: float, noise_seed: int
) -> np.ndarray:
    """Draw observations through the known symmetric channel (over the n-1 non-true cells).

    Delegates to ``partial_observability.realize_noisy_observations``, the single channel-law
    source for this project (was independently reimplemented here; verified byte-identical
    across 300 randomized trials, including eps=0 and eps=1, before consolidating).
    """

    rng = np.random.default_rng(noise_seed)
    return realize_noisy_observations(latent, n, eps, rng)


# --------------------------------------------------------------------------- #
# Candidate interface
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PredictContext:
    """The leakage-free inputs a candidate may use to predict the next latent cell.

    Exposes ONLY information available through step ``i - 1``. The harness deliberately
    does NOT include the current observation ``o_i`` or the true target ``z_i``, so a
    candidate physically cannot peek at what it is being scored against.

    Fields:
    * ``n_states`` — number of latent cells.
    * ``transition`` — shared train-learned latent transition, ``transition[next, prev]``
      (column-stochastic).
    * ``emission`` — the known symmetric emission channel, ``emission[obs, latent]``.
    * ``past_observations`` — observed cells for this traveller through ``i - 1``
      (read-only view; empty array at the traveller's first scored step).
    * ``prev_observation`` — the most recent observed cell ``o_{i-1}`` (the last entry of
      ``past_observations``).
    * ``belief`` — the candidate's own running belief over the latent cell carried from
      the previous step (a length-``n`` distribution); ``None`` only before the first
      observation of the traveller. Candidates that maintain a recursive belief read and
      ignore this as they wish; the harness updates it from the candidate's emission and
      transition after scoring, so each candidate sees a consistently-evolved belief.
    """

    n_states: int
    transition: np.ndarray
    emission: np.ndarray
    past_observations: np.ndarray
    prev_observation: int
    belief: np.ndarray | None
    traveller_id: int
    step_index: int


@runtime_checkable
class Candidate(Protocol):
    """A uniform plug-in for a next-latent-cell predictor.

    Implementations expose a ``name`` and a ``predict`` that returns a length-``n_states``
    probability distribution over the NEXT latent cell from a :class:`PredictContext`.
    The returned vector need not be pre-normalized; the scorer normalizes and clamps it.
    """

    name: str

    def predict(
        self, context: PredictContext
    ) -> np.ndarray:  # pragma: no cover - protocol
        ...


@dataclass
class FunctionCandidate:
    """Adapt a bare ``predict(context) -> dist`` callable into a named :class:`Candidate`."""

    name: str
    fn: object  # Callable[[PredictContext], np.ndarray]; kept loose to avoid import churn

    def predict(self, context: PredictContext) -> np.ndarray:
        return self.fn(context)  # type: ignore[operator]


def _normalize_distribution(raw: np.ndarray, n_states: int) -> np.ndarray:
    """Clamp negatives, normalize, and validate a candidate's predictive distribution."""

    dist = np.asarray(raw, dtype=float).reshape(-1)
    if dist.shape[0] != n_states:
        raise ValueError(
            f"candidate returned length {dist.shape[0]} distribution, expected {n_states}"
        )
    if not np.all(np.isfinite(dist)):
        raise ValueError("candidate returned a non-finite predictive distribution")
    dist = np.clip(dist, 0.0, None)
    total = dist.sum()
    if total <= 0.0:
        raise ValueError("candidate returned an all-zero predictive distribution")
    return dist / total


def seed_belief_from_observation(emission: np.ndarray, obs: int) -> np.ndarray:
    """The one-observation posterior a traveller's running belief is seeded from.

    This is the same first-step initialization ``score_candidate`` uses for every
    traveller's belief before any recursive update is applied: ``emission[obs, :]``,
    renormalized. Exported so other modules (e.g. ``sota_neural``) that need to replay
    the harness's belief trajectory for training data construction reuse this exact
    initialization instead of reimplementing it.
    """

    b = emission[obs, :].astype(float).copy()
    return b / b.sum()


def advance_belief(
    emission: np.ndarray, transition: np.ndarray, belief: np.ndarray, obs: int
) -> np.ndarray:
    """The leakage-free recursive belief update: ``emission[obs] * (transition @ belief)``.

    This is the exact running-belief recursion ``score_candidate`` uses to advance each
    traveller's belief after scoring (or after a train step). Exported so any other module
    needing this same posterior update (e.g. ``sota_neural``'s training/prediction code)
    calls this shared implementation rather than a hand-reimplementation that could
    silently diverge from the harness.
    """

    updated = emission[obs, :] * (transition @ belief)
    return updated / updated.sum()


# --------------------------------------------------------------------------- #
# Leakage-free prequential scorer
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScoredCandidate:
    """Per-traveller and pooled next-cell diagnostics for one candidate on holdout steps."""

    name: str
    pooled_mean_loss: float
    n_scored_steps: int
    per_traveller_mean_loss: dict[int, float]
    step_losses: np.ndarray  # one entry per scored holdout step, in step order
    step_travellers: (
        np.ndarray
    )  # the traveller id for each scored step (clustered unit)
    step_true_cells: np.ndarray
    step_predicted_cells: np.ndarray
    step_true_probabilities: np.ndarray
    step_confidences: np.ndarray
    step_correct: np.ndarray
    brier_score: float
    ece: float
    top1_accuracy: float
    loss_ci_low: float | None = None
    loss_ci_high: float | None = None
    runtime_seconds: float | None = None


def score_candidate(
    dataset: PartialObservabilityDataset,
    candidate: Candidate,
) -> ScoredCandidate:
    """Leakage-free prequential next-latent-cell scoring of ``candidate`` on holdout travellers.

    At each holdout, non-first step ``i`` the candidate predicts the NEXT latent cell from
    a :class:`PredictContext` built from observations through ``i - 1`` only; the loss is
    ``-log p(z_i)`` against the TRUE latent cell ``z_i``. The harness maintains a running
    belief per traveller (updated AFTER scoring using the candidate's emission and shared
    transition) but never reveals ``o_i`` or ``z_i`` before the prediction is taken. Both
    holdout and train steps advance the belief; only holdout steps contribute to the loss,
    matching the train-primed prequential continuation of the other modules.
    """

    latent = dataset.latent
    observed = dataset.observed
    is_first = dataset.is_first
    travellers = dataset.travellers
    is_holdout = dataset.is_holdout
    n = dataset.n_states
    transition = dataset.transition
    emission = dataset.emission

    beliefs: dict[int, np.ndarray] = {}
    pasts: dict[int, list[int]] = {}

    step_losses: list[float] = []
    step_travellers: list[int] = []
    step_true_cells: list[int] = []
    step_predicted_cells: list[int] = []
    step_true_probabilities: list[float] = []
    step_confidences: list[float] = []
    step_correct: list[bool] = []
    step_brier: list[float] = []
    per_traveller: dict[int, list[float]] = {}

    for i in range(len(latent)):
        tid = int(travellers[i])
        obs_i = int(observed[i])
        if is_first[i] or tid not in beliefs:
            # Initialize the running belief from the first observation alone (the same
            # one-step posterior the partial-observability filter starts from). The first
            # observation seeds the belief; no scoring happens here.
            beliefs[tid] = seed_belief_from_observation(emission, obs_i)
            pasts[tid] = [obs_i]
            continue

        past = pasts[tid]
        prev_obs = past[-1]
        if bool(is_holdout[i]):
            context = PredictContext(
                n_states=n,
                transition=transition,
                emission=emission,
                past_observations=np.asarray(past, dtype=int),
                prev_observation=int(prev_obs),
                belief=beliefs[tid].copy(),
                traveller_id=tid,
                step_index=i,
            )
            dist = _normalize_distribution(candidate.predict(context), n)
            true_cell = int(latent[i])
            true_prob = float(dist[true_cell])
            pred_cell = int(np.argmax(dist))
            confidence = float(dist[pred_cell])
            correct = pred_cell == true_cell
            onehot = np.zeros(n, dtype=float)
            onehot[true_cell] = 1.0
            loss = -float(np.log(max(true_prob, 1e-12)))
            step_losses.append(loss)
            step_travellers.append(tid)
            step_true_cells.append(true_cell)
            step_predicted_cells.append(pred_cell)
            step_true_probabilities.append(true_prob)
            step_confidences.append(confidence)
            step_correct.append(correct)
            step_brier.append(float(np.mean((dist - onehot) ** 2)))
            per_traveller.setdefault(tid, []).append(loss)

        # Advance the running belief AFTER scoring, using the candidate-agnostic recursive
        # update (known emission + shared transition). This keeps the belief a consistent,
        # leakage-free summary of the past for every candidate.
        beliefs[tid] = advance_belief(emission, transition, beliefs[tid], obs_i)
        past.append(obs_i)

    if not step_losses:
        raise ValueError("no scored holdout steps; cannot score candidate")

    return ScoredCandidate(
        name=candidate.name,
        pooled_mean_loss=float(np.mean(step_losses)),
        n_scored_steps=len(step_losses),
        per_traveller_mean_loss={
            t: float(np.mean(v)) for t, v in per_traveller.items()
        },
        step_losses=np.asarray(step_losses, dtype=float),
        step_travellers=np.asarray(step_travellers, dtype=int),
        step_true_cells=np.asarray(step_true_cells, dtype=int),
        step_predicted_cells=np.asarray(step_predicted_cells, dtype=int),
        step_true_probabilities=np.asarray(step_true_probabilities, dtype=float),
        step_confidences=np.asarray(step_confidences, dtype=float),
        step_correct=np.asarray(step_correct, dtype=bool),
        brier_score=float(np.mean(step_brier)),
        ece=_ece(
            np.asarray(step_confidences, dtype=float),
            np.asarray(step_correct, dtype=bool),
            10,
        ),
        top1_accuracy=float(np.mean(step_correct)),
    )


# --------------------------------------------------------------------------- #
# Traveller-clustered bootstrap on the loss DIFFERENCE
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LossDifferenceCI:
    """A traveller-clustered CI on the per-step loss difference between two candidates."""

    candidate_a: str
    candidate_b: str
    mean_difference_nats: float  # loss(a) - loss(b); positive => b is better
    clustered_ci_low: float
    clustered_ci_high: float
    b_wins_clustered: bool  # CI entirely above 0 => candidate_b strictly better
    a_wins_clustered: bool  # CI entirely below 0 => candidate_a strictly better
    n_holdout_travellers: int
    # Exact one-sided clustered-bootstrap p-value for H1: candidate_b strictly better
    # (mean_difference_nats > 0). Read from the SAME bootstrap distribution as the CI, so
    # the two never disagree. Consumed by the multiplicity ledger as the per-comparison
    # uncorrected p; see multiplicity_ledger.py.
    one_sided_p_b_better: float = 1.0


def paired_loss_difference_ci(
    scored_a: ScoredCandidate,
    scored_b: ScoredCandidate,
    *,
    n_boot: int = 2000,
    seed: int = 7,
) -> LossDifferenceCI:
    """Traveller-clustered CI on the loss DIFFERENCE ``loss(a) - loss(b)`` (positive => b better).

    Both candidates must have been scored on the SAME holdout steps in the same order
    (they will be, since :func:`score_candidate` walks the dataset deterministically).
    Resamples whole TRAVELLERS — the correlated unit — via the tested
    ``traveller_clustered_bootstrap_ci``. A per-step bootstrap is INVALID here because
    steps within a traveller are correlated and would yield a falsely tight interval; this
    is exactly what makes a per-step "win" look separated when it is not.
    """

    if scored_a.step_losses.shape[0] != scored_b.step_losses.shape[0]:
        raise ValueError(
            "candidates were scored on different numbers of holdout steps; cannot pair"
        )
    if not np.array_equal(scored_a.step_travellers, scored_b.step_travellers):
        raise ValueError(
            "candidates were scored on different holdout steps; cannot pair"
        )

    diffs = scored_a.step_losses - scored_b.step_losses  # positive => b better
    travs = scored_a.step_travellers
    stats = traveller_clustered_bootstrap_stats(diffs, travs, n_boot=n_boot, seed=seed)
    lo, hi = stats["ci_low"], stats["ci_high"]
    return LossDifferenceCI(
        candidate_a=scored_a.name,
        candidate_b=scored_b.name,
        mean_difference_nats=float(np.mean(diffs)),
        clustered_ci_low=lo,
        clustered_ci_high=hi,
        b_wins_clustered=bool(lo > 0.0),
        a_wins_clustered=bool(hi < 0.0),
        n_holdout_travellers=int(np.unique(travs).size),
        one_sided_p_b_better=stats["one_sided_p"],
    )


# --------------------------------------------------------------------------- #
# Reference fair baselines (so the spine is usable immediately)
# --------------------------------------------------------------------------- #
@dataclass
class SoftFilterCandidate:
    """Active-inference soft filter: predict by marginalizing the running belief, ``T @ b``.

    This is the genuine AIF predictor of ``partial_observability.py``. It is the *fair*
    candidate the next phase will pit against the baselines below; included here as a
    reference implementation of the interface, not as a SOTA assertion.
    """

    name: str = "soft_filter"

    def predict(self, context: PredictContext) -> np.ndarray:
        belief = context.belief
        if belief is None:
            return np.full(context.n_states, 1.0 / context.n_states)
        return context.transition @ belief


@dataclass
class HardFilterCandidate:
    """Point-estimate baseline: collapse the belief to its MAP cell, then ``T @ onehot(MAP)``."""

    name: str = "hard_filter"

    def predict(self, context: PredictContext) -> np.ndarray:
        belief = context.belief
        if belief is None:
            return np.full(context.n_states, 1.0 / context.n_states)
        onehot = np.zeros(context.n_states, dtype=float)
        onehot[int(np.argmax(belief))] = 1.0
        return context.transition @ onehot


@dataclass
class ObservedMarkovCandidate:
    """Noisy-Markov baseline: commit to the raw previous observation, ``T @ onehot(o_{i-1})``."""

    name: str = "observed_markov"

    def predict(self, context: PredictContext) -> np.ndarray:
        onehot = np.zeros(context.n_states, dtype=float)
        onehot[int(context.prev_observation)] = 1.0
        return context.transition @ onehot


def reference_baselines() -> tuple[Candidate, ...]:
    """The strong fair baselines bundled with the spine (the hard point estimate is the
    strongest fair competitor the AIF candidate must beat; the soft filter is the AIF
    candidate itself, provided here as a reference implementation)."""

    return (HardFilterCandidate(), ObservedMarkovCandidate())
