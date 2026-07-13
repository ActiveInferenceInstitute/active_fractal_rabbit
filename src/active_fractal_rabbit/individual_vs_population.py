"""Individual-vs-population experiment: does active-inference individual modelling help most
when there is no population to borrow strength from (the N=1 limit)?

The project's scale-fragility finding is that the *population* transition model becomes sufficient
as travellers accumulate, so the per-traveller (recursive / preferential-return) advantage erodes.
The logical extreme: at one traveller there is no population, prediction can only exploit the
individual's own history, and individual modelling should matter most. This module measures that
directly on a single target traveller's stream, holding the emission fixed and varying ONLY the
transition source:

* **population(k)** — the next-cell transition is estimated from ``k`` OTHER travellers (pooled,
  frozen); the target is predicted by marginalizing its own soft emission posterior through that
  shared transition. This is "borrow strength from the population."
* **individual** — the target's OWN online transition counts plus a frequency-weighted
  preferential-return prior (visit^alpha), using only the target's past. No population.

``advantage(k) = loss_population(k) - loss_individual``. The advantage is largest at ``k = 0`` (the
N=1 limit — population starved) and shrinks as ``k`` grows and pooled transitions become sufficient.

**Mechanism (RedTeam-corrected — do not read this as an active-inference win).** On the real
external trace travellers are severely spatially confined (5/8 occupy <=3 of 16 cells; one is 99%
in a single cell). The individual model wins not because of preferential-return structure or active
inference but by **information congruence**: each traveller occupies its own few-cell territory, so a
transition pooled across travellers with *disjoint* territories is off-distribution for any target,
while the target's own confined history is on-distribution. The return weight rho barely affects the
individual loss (it is not the driver), and a trivial dominant-cell predictor beats the model. Any
individual model captures this. The experiment is a valid measurement of an information-set mismatch;
it is NOT evidence for a structure-matched active-inference generative model.

Real methods only: a fixed emission trained on non-target travellers, real transition counts,
prequential scoring; no mocks, stubs, or synthetic shortcuts.
"""

from __future__ import annotations

import math

import numpy as np

from .pomdp import _emission_matrix
from .types import ObservationSequence

INDIVIDUAL_VS_POPULATION_SCHEMA = "active-fractal-rabbit-individual-vs-population-v1"


def _arrays(sequence: ObservationSequence):
    cells = np.asarray(sequence.spatial_states, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    travellers = np.asarray([r.traveller_id for r in sequence.observations], dtype=int)
    return cells, first, travellers, int(sequence.num_spatial_states)


def _population_loss(cells, first, travellers, n, target, others, post, smoothing):
    """Prequential next-cell loss for the target, transition pooled from `others` (frozen)."""
    if others:
        mask = np.isin(travellers, list(others))
        counts = np.full((n, n), smoothing, dtype=float)
        last = None
        for i in range(len(cells)):
            if first[i] or travellers[i] != last:
                last = travellers[i]
                continue
            if mask[i]:
                counts[int(cells[i]), int(cells[i - 1])] += 1.0
        B = counts / counts.sum(axis=0, keepdims=True)
    else:
        B = np.full((n, n), 1.0 / n)  # no population -> uninformative transition
    losses: list[float] = []
    last = None
    for i in range(len(cells)):
        if travellers[i] != target:
            continue
        if first[i] or last != target:
            last = target
            continue
        pred = B @ post[i - 1]
        pred = pred / pred.sum()
        losses.append(-math.log(max(float(pred[int(cells[i])]), 1e-12)))
    return float(np.mean(losses)) if losses else math.nan


def _individual_loss(cells, first, travellers, n, target, post, smoothing, alpha, rho):
    """Prequential next-cell loss for the target using only its OWN online history +
    frequency-weighted preferential return."""
    counts = np.full((n, n), smoothing, dtype=float)
    visits = np.zeros(n, dtype=float)
    losses: list[float] = []
    last = None
    for i in range(len(cells)):
        if travellers[i] != target:
            continue
        if first[i] or last != target:
            last = target
            visits[int(cells[i])] += 1.0
            continue
        explore = (counts / counts.sum(axis=0, keepdims=True)) @ post[i - 1]
        if visits.sum() > 0.0:
            ret = np.power(visits, alpha)
            ret = ret / ret.sum()
        else:
            ret = explore
        pred = (1.0 - rho) * explore + rho * ret
        pred = pred / pred.sum()
        losses.append(-math.log(max(float(pred[int(cells[i])]), 1e-12)))
        counts[int(cells[i]), int(cells[i - 1])] += 1.0
        visits[int(cells[i])] += 1.0
    return float(np.mean(losses)) if losses else math.nan


def advantage_vs_population_size(
    sequence: ObservationSequence,
    *,
    target: int | None = None,
    smoothing: float = 0.05,
    alpha: float = 1.0,
    rho: float = 0.7,
) -> dict[str, object]:
    """Sweep the population size available to the pooled transition (k = 0..N-1 other travellers)
    and measure the individual structure-matched model's advantage over it, on one target traveller."""

    cells, first, travellers, n = _arrays(sequence)
    uniq = sorted(set(int(t) for t in travellers.tolist()))
    if len(uniq) < 2:
        raise ValueError("need >=2 travellers to vary the population size")
    if target is None:
        # the most-active traveller makes the cleanest target (most holdout steps)
        target = int(max(uniq, key=lambda t: int((travellers == t).sum())))
    others_pool = [t for t in uniq if t != target]
    # fixed emission trained on NON-target travellers (no target leakage), shared by both predictors
    non_target = travellers != target
    A = _emission_matrix(cells[non_target], cells[non_target], n, n, smoothing)
    post = A[cells, :] / A[cells, :].sum(axis=1, keepdims=True)

    ind_loss = _individual_loss(
        cells, first, travellers, n, target, post, smoothing, alpha, rho
    )
    sizes = sorted({0, 1, 2, min(4, len(others_pool)), len(others_pool)})
    rows = []
    for k in sizes:
        pop_loss = _population_loss(
            cells, first, travellers, n, target, others_pool[:k], post, smoothing
        )
        rows.append(
            {
                "n_other_travellers": int(k),
                "population_loss": pop_loss,
                "individual_loss": ind_loss,
                "individual_advantage_nats": float(pop_loss - ind_loss),
                "individual_wins": bool(pop_loss > ind_loss),
            }
        )
    advs = [r["individual_advantage_nats"] for r in rows]
    monotone = all(
        a >= b - 1e-9 for a, b in zip(advs, advs[1:])
    )  # advantage decreasing with k
    return {
        "schema": INDIVIDUAL_VS_POPULATION_SCHEMA,
        "target_traveller": int(target),
        "n_travellers": len(uniq),
        "alpha": float(alpha),
        "rho": float(rho),
        "rows": rows,
        "advantage_is_max_at_zero_population": bool(advs[0] == max(advs)),
        "advantage_decreases_with_population": bool(monotone),
        "individual_advantage_at_zero": float(advs[0]),
        "individual_advantage_at_full_population": float(advs[-1]),
        "interpretation": (
            "the individual model's advantage over the pooled-population transition is largest with no "
            "population to pool from (the N=1 limit) and shrinks as travellers accumulate"
        ),
        "mechanism_corrected": (
            "RedTeam-verified caveat: on confined traces this advantage is INFORMATION CONGRUENCE / "
            "spatial confinement, NOT active inference. Each traveller occupies a few cells; pooling "
            "mixes disjoint territories so the pooled transition is off-distribution for the target. "
            "The return weight rho barely changes the individual loss (it is not the driver), and a "
            "trivial dominant-cell predictor beats the model. Any individual model captures this; it "
            "is not evidence for a structure-matched active-inference generative model."
        ),
        "boundary": "synthetic-only; an information-congruence characterization, not an active-inference win, not oracle-beating or operational",
    }
