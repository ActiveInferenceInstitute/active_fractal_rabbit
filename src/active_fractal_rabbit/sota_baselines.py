"""Transparent fair-baseline ladder for the partial-observability SOTA spine.

These are the *bar* the active-inference candidate must clear. Every member here is a
plug-in :class:`~active_fractal_rabbit.sota_program.Candidate`: it implements
``predict(context) -> next-latent-cell distribution`` against the leakage-free
:class:`~active_fractal_rabbit.sota_program.PredictContext`, so it scores through exactly
the same harness as ``SoftFilterCandidate`` / ``HardFilterCandidate`` and competes on the
SAME holdout steps with the SAME train-learned transition.

The honesty rule that shapes every baseline here: **none of them gets the oracle previous
cell.** Under the spine's known noisy emission the latent location is hidden, so each
baseline must INFER location from the same noisy observations the AIF candidate sees. The
single fair inference primitive the context exposes for that is the known emission channel:
``P(latent = c | obs = o) ∝ emission[o, c]`` (uniform latent prior), which is the
correctly-specified single-step posterior used throughout ``partial_observability.py`` and
``fair_inference.py``. A baseline that instead committed to the raw observation as if it
were the true cell is already provided by the spine as ``ObservedMarkovCandidate``; the
ladder here is the *soft-inference* family that marginalizes that uncertainty honestly.

The ladder (mirroring the prequential nulls of ``sequence_baselines.py`` and the matched
Markov of ``fair_inference._markov_loss``):

* :class:`PersistenceCandidate` — predict the inferred current cell persists
  (``emission`` posterior of the previous observation; no transition, no memory).
* :class:`OnlineBaseRateCandidate` — the marginal frequency of inferred cells seen so far
  for this traveller (soft emission-posterior counts over ``past_observations``).
* :class:`FirstOrderMarkovCandidate` — the shared train-learned transition applied to the
  inferred previous-cell posterior, ``transition @ P(cell | o_{i-1})`` (this is the soft
  filter WITHOUT temporal recursion — it ignores the running ``belief``).
* :class:`HigherOrderBackoffMarkovCandidate` — second-order inferred history when the
  traveller's own past contains matching context, backed off to the first-order Markov and
  soft return prior when it does not.
* :class:`ReturnBoostedMarkovCandidate` — first-order Markov plus the per-traveller
  visited-cell return prior of ``fair_inference._markov_loss(return_boost=...)`` /
  ``sequence_baselines._return_boost_loss`` (the strongest transparent fair baseline).
* :class:`RecencyAttentionCandidate` — a dependency-light sequence baseline that uses
  exponentially-decayed attention over this traveller's past inferred cells before applying
  the shared transition and return prior (numpy only, no external weights).
* :class:`MultiHorizonRecencyAttentionCandidate` — a dependency-light attention baseline
  that mixes short-, medium-, and long-memory inferred-cell histories so robustness lanes
  have a stronger transparent sequence competitor without admitting a foundation model.

**Phase-conditioned Markov.** ``sequence_baselines.py`` also offers a phase-conditioned
Markov, but the spine's :class:`PredictContext` deliberately exposes ONLY
``n_states, transition, emission, past_observations, prev_observation, belief,
traveller_id, step_index`` — it carries no time-of-day / circadian phase factor. A
phase-conditioned candidate therefore cannot be built honestly against THIS interface
without fabricating a phase signal, so it is intentionally NOT provided as a spine
Candidate here. :func:`phase_conditioning_available` documents that gap so the harness
phase can detect it explicitly rather than silently shipping a phase-free stand-in.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .sota_program import Candidate, PredictContext

SOTA_BASELINES_SCHEMA = "active-fractal-rabbit-sota-baselines-v1"


def _emission_posterior(emission: np.ndarray, obs: int) -> np.ndarray:
    """Single-step posterior ``P(latent = c | obs)`` under a uniform latent prior.

    ``emission[obs, latent]`` is the known channel; with a uniform prior the posterior is
    just that row renormalised. This is the correctly-specified soft inference of the
    hidden cell from one noisy observation (same primitive as ``fair_inference.post_obs``),
    and it is the ONLY location estimate a fair baseline is allowed to form — never the
    oracle true cell.
    """

    row = np.asarray(emission[int(obs), :], dtype=float)
    total = row.sum()
    if total <= 0.0:
        # A degenerate emission row cannot inform the posterior; fall back to uniform.
        return np.full(row.shape[0], 1.0 / row.shape[0])
    return row / total


def _emission_posteriors(emission: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Stack of single-step posteriors, one row per observation (empty -> shape (0, n))."""

    n = emission.shape[1]
    if observations.size == 0:
        return np.empty((0, n), dtype=float)
    rows = emission[observations.astype(int), :].astype(float)
    totals = rows.sum(axis=1, keepdims=True)
    safe = np.where(totals > 0.0, totals, 1.0)
    posteriors = rows / safe
    # any all-zero emission row becomes uniform rather than NaN
    degenerate = totals.reshape(-1) <= 0.0
    if np.any(degenerate):
        posteriors[degenerate] = 1.0 / n
    return posteriors


@dataclass
class PersistenceCandidate:
    """Inferred-persistence null: predict the inferred *current* cell stays put.

    The fair analogue of ``sequence_baselines._persistence_loss`` under partial
    observability: rather than committing to the raw previous observation, it predicts the
    soft emission posterior of that observation (the inferred previous/current cell). No
    transition and no temporal memory — the weakest member of the ladder.
    """

    name: str = "persistence"

    def predict(self, context: PredictContext) -> np.ndarray:
        return _emission_posterior(context.emission, context.prev_observation)


@dataclass
class OnlineBaseRateCandidate:
    """Online base-rate null: the marginal frequency of inferred cells seen so far.

    Mirrors ``sequence_baselines._base_rate_loss`` but, because the true cell is hidden,
    accumulates SOFT emission-posterior mass over this traveller's ``past_observations``
    instead of hard true-cell counts. Predicts that marginal directly, ignoring the
    previous cell entirely — the "no dynamics" null.
    """

    name: str = "online_base_rate"
    smoothing: float = 0.5

    def predict(self, context: PredictContext) -> np.ndarray:
        n = context.n_states
        posteriors = _emission_posteriors(context.emission, context.past_observations)
        counts = np.full(n, float(self.smoothing), dtype=float)
        if posteriors.shape[0] > 0:
            counts = counts + posteriors.sum(axis=0)
        return counts / counts.sum()


@dataclass
class FirstOrderMarkovCandidate:
    """First-order Markov: the shared transition applied to the inferred previous cell.

    ``transition @ P(cell | o_{i-1})`` with the SAME train-learned, column-stochastic
    transition every spine candidate uses. This is exactly the soft filter WITHOUT temporal
    recursion (it deliberately ignores ``context.belief``), matching the matched-source
    ``fair_inference._markov_loss(posterior=...)`` predictor. The strongest dynamics-only
    fair baseline before any visited-cell prior is added.
    """

    name: str = "first_order_markov"

    def predict(self, context: PredictContext) -> np.ndarray:
        post_prev = _emission_posterior(context.emission, context.prev_observation)
        return context.transition @ post_prev


@dataclass
class HigherOrderBackoffMarkovCandidate:
    """Interpolated second-order Markov baseline with first-order and return backoff."""

    name: str = "higher_order_backoff_markov"
    higher_order_weight: float = 0.35
    return_weight: float = 0.15
    smoothing: float = 0.5

    def predict(self, context: PredictContext) -> np.ndarray:
        n = context.n_states
        first_order = FirstOrderMarkovCandidate().predict(context)
        posteriors = _emission_posteriors(context.emission, context.past_observations)
        if posteriors.shape[0] < 3:
            return _normalise(first_order, n)

        inferred = np.argmax(posteriors, axis=1)
        # Genuine second-order context: the traveller's last TWO inferred cells, not just
        # the single most-recent one -- matching only on `inferred[-1]` would collapse this
        # to first-order conditioning under a different (traveller-local) count source.
        context_prev2 = int(inferred[-2])
        context_prev1 = int(inferred[-1])
        second_order_counts = np.full(n, float(self.smoothing), dtype=float)
        matched = 0
        for idx in range(0, inferred.size - 2):
            if (
                int(inferred[idx]) == context_prev2
                and int(inferred[idx + 1]) == context_prev1
            ):
                second_order_counts[int(inferred[idx + 2])] += 1.0
                matched += 1

        if matched == 0:
            return_prior = np.full(
                n, float(self.smoothing), dtype=float
            ) + posteriors.sum(axis=0)
            return _normalise(
                (1.0 - float(self.return_weight)) * first_order
                + float(self.return_weight) * _normalise(return_prior, n),
                n,
            )

        second_order = _normalise(second_order_counts, n)
        return_prior = np.full(n, float(self.smoothing), dtype=float) + posteriors.sum(
            axis=0
        )
        backoff_weight = max(
            0.0, 1.0 - float(self.higher_order_weight) - float(self.return_weight)
        )
        pred = (
            backoff_weight * first_order
            + float(self.higher_order_weight) * second_order
            + float(self.return_weight) * _normalise(return_prior, n)
        )
        return _normalise(pred, n)


@dataclass
class ReturnBoostedMarkovCandidate:
    """First-order Markov plus the per-traveller visited-cell return prior.

    Adds the visited-cell boost of ``fair_inference._markov_loss(return_boost=...)`` /
    ``sequence_baselines._return_boost_loss`` to the first-order Markov prediction: extra
    predictive mass on cells this traveller has already (inferably) visited, scaled to the
    prediction's own mass so it is a reweighting, not an override. The visited set is the
    inferred-MAP cells of this traveller's ``past_observations`` (no oracle), so the boost
    is itself leakage-free. This is the strongest transparent fair baseline in the ladder —
    the bar the AIF candidate must beat once both carry the visited-cell feature.
    """

    name: str = "return_boosted_markov"
    return_weight: float = 1.0

    def predict(self, context: PredictContext) -> np.ndarray:
        n = context.n_states
        post_prev = _emission_posterior(context.emission, context.prev_observation)
        pred = context.transition @ post_prev

        seen = np.zeros(n, dtype=bool)
        past = context.past_observations
        if past.size > 0:
            posteriors = _emission_posteriors(context.emission, past)
            inferred = np.argmax(posteriors, axis=1)
            seen[inferred] = True

        n_seen = int(seen.sum())
        if self.return_weight > 0.0 and n_seen > 0:
            # Same shape as the existing return boost: distribute weight * total mass over
            # the visited cells (mean-normalised), then renormalise. A reweighting toward
            # revisits, never a hard override.
            pred = pred + self.return_weight * seen * pred.sum() / n_seen
        total = pred.sum()
        if total <= 0.0:
            return np.full(n, 1.0 / n)
        return pred / total


@dataclass
class RecencyAttentionCandidate:
    """Dependency-light sequence attention over past inferred cells.

    This is intentionally not a foundation model: it has no learned weights, no external
    checkpoint, and no data contract beyond the locked split. It gives the SOTA program a
    simple attention-style competitor by weighting this traveller's past emission
    posteriors with an exponential recency kernel, applying the shared train-learned
    transition, then mixing in a soft return prior from the same past. It sees only
    ``past_observations`` and the shared channel/transition exposed by the spine.
    """

    name: str = "recency_attention"
    recency_half_life: float = 8.0
    return_weight: float = 0.35
    smoothing: float = 0.5

    def predict(self, context: PredictContext) -> np.ndarray:
        n = context.n_states
        posteriors = _emission_posteriors(context.emission, context.past_observations)
        if posteriors.shape[0] == 0:
            return np.full(n, 1.0 / n)

        ages = np.arange(posteriors.shape[0] - 1, -1, -1, dtype=float)
        half_life = max(float(self.recency_half_life), 1e-9)
        weights = np.exp(-math.log(2.0) * ages / half_life)
        attended_prev = weights @ posteriors
        attended_prev = attended_prev / attended_prev.sum()
        pred = context.transition @ attended_prev

        return_prior = np.full(n, float(self.smoothing), dtype=float) + posteriors.sum(
            axis=0
        )
        return_prior = return_prior / return_prior.sum()
        pred = (1.0 - float(self.return_weight)) * pred + float(
            self.return_weight
        ) * return_prior
        total = pred.sum()
        if total <= 0.0:
            return np.full(n, 1.0 / n)
        return pred / total


@dataclass
class MultiHorizonRecencyAttentionCandidate:
    """Dependency-light multi-horizon sequence attention.

    This candidate is deliberately transparent: it has no fitted weights and no external
    checkpoint. It blends several exponential attention horizons over this traveller's past
    emission posteriors, pushes each attended history through the shared train-learned
    transition, and mixes the result with the same soft return prior used by the recency
    baseline. The short horizon helps bursty reports, the long horizon helps sparse reporting
    gaps, and the fixed mixture makes the row reproducible and auditable.
    """

    name: str = "multi_horizon_recency_attention"
    half_lives: tuple[float, ...] = (2.0, 8.0, 32.0)
    horizon_weights: tuple[float, ...] = (0.40, 0.35, 0.25)
    return_weight: float = 0.30
    smoothing: float = 0.5

    def predict(self, context: PredictContext) -> np.ndarray:
        n = context.n_states
        posteriors = _emission_posteriors(context.emission, context.past_observations)
        if posteriors.shape[0] == 0:
            return np.full(n, 1.0 / n)

        half_lives = tuple(float(h) for h in self.half_lives if float(h) > 0.0)
        if not half_lives:
            half_lives = (8.0,)
        raw_weights = np.asarray(self.horizon_weights[: len(half_lives)], dtype=float)
        if raw_weights.shape[0] < len(half_lives):
            raw_weights = np.pad(
                raw_weights,
                (0, len(half_lives) - raw_weights.shape[0]),
                constant_values=1.0,
            )
        raw_weights = np.clip(raw_weights, 0.0, None)
        if raw_weights.sum() <= 0.0:
            raw_weights = np.ones(len(half_lives), dtype=float)
        horizon_weights = raw_weights / raw_weights.sum()

        ages = np.arange(posteriors.shape[0] - 1, -1, -1, dtype=float)
        mixture = np.zeros(n, dtype=float)
        for half_life, weight in zip(half_lives, horizon_weights, strict=True):
            kernel = np.exp(-math.log(2.0) * ages / max(float(half_life), 1e-9))
            attended_prev = kernel @ posteriors
            attended_prev = attended_prev / attended_prev.sum()
            mixture += float(weight) * (context.transition @ attended_prev)

        return_prior = np.full(n, float(self.smoothing), dtype=float) + posteriors.sum(
            axis=0
        )
        return_prior = return_prior / return_prior.sum()
        pred = (1.0 - float(self.return_weight)) * mixture + float(
            self.return_weight
        ) * return_prior
        total = pred.sum()
        if total <= 0.0:
            return np.full(n, 1.0 / n)
        return pred / total


def transparent_baseline_ladder() -> tuple[Candidate, ...]:
    """The transparent fair-baseline ladder, weakest -> strongest.

    Every member infers location from the SAME noisy observations (no oracle previous
    cell) and plugs into ``sota_program.score_candidate`` exactly like the bundled
    reference candidates. The attention-style rows are stronger transparent sequence
    competitors, not foundation models.
    """

    return (
        PersistenceCandidate(),
        OnlineBaseRateCandidate(),
        FirstOrderMarkovCandidate(),
        HigherOrderBackoffMarkovCandidate(),
        ReturnBoostedMarkovCandidate(),
        RecencyAttentionCandidate(),
        MultiHorizonRecencyAttentionCandidate(),
    )


def phase_conditioning_available(context: PredictContext) -> bool:
    """Whether a phase-conditioned Markov candidate can be built from THIS context.

    The spine's :class:`PredictContext` exposes no time-of-day / circadian phase factor, so
    a phase-conditioned baseline cannot be formed honestly against the interface. This
    returns ``False`` for the current spine and is the explicit, testable record of that
    gap — a phase-conditioned candidate is intentionally omitted rather than shipped with a
    fabricated phase signal.
    """

    return hasattr(context, "phase") or hasattr(context, "time_bin")


def _normalise(values: np.ndarray, n: int) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0.0:
        return np.full(n, 1.0 / n)
    return np.asarray(values, dtype=float) / total
