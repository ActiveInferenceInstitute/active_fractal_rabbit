"""Confound-control battery + mobility check for active-inference model claims.

This project's defining lesson is that every apparent active-inference advantage has reduced to a
SIMPLER effect — oracle access, data density, or spatial confinement. This module makes that
discipline executable: before any "model X is better" claim, X must be scored against a battery of
confound baselines on a *mobility-checked* (non-confined) regime, and is only credited with a win if
it beats the fair confounds. It exists so a confound can never again be mistaken for capability.

The battery (all scored on the same locked traveller split, with the same emission):

* **dominant_cell** — predict the traveller's own online visit-frequency distribution. Captures pure
  spatial confinement; if a model cannot beat this, its "win" is confinement, not inference.
* **first_order_pooled** — a first-order Markov transition pooled over the train travellers,
  marginalized through the target's soft emission posterior (inferred previous cell, fair). Captures
  the data-density null; if a model cannot beat this, pooling is already sufficient.
* **return_boosted_soft** — the strongest fair non-recursive competitor (soft Markov + per-traveller
  visited-cell prior). A model must beat this to claim a genuine fair-frontier advantage.
* **oracle_prev** — the FIXED pooled-population Markov handed the TRUE previous cell. A reference
  point, NOT a strict ceiling: because the pooled transition is fixed (not adapted per traveller), an
  adaptive online fair model can legitimately beat it on some generators, so a model loss below
  oracle_prev is not by itself a leak. Reported as a gap, never required to clear.

A model ``clears_confounds`` iff its loss is below dominant_cell AND first_order_pooled AND
return_boosted_soft. The mobility check (mean distinct cells / occupancy entropy per traveller) gates
whether the regime is non-confined enough for the test to mean anything.
"""

from __future__ import annotations

import math

import numpy as np

from .fair_inference import _markov_loss
from .pomdp import _emission_matrix
from .sota_benchmark import lock_traveller_split
from .types import ObservationSequence

CONFOUND_CONTROLS_SCHEMA = "active-fractal-rabbit-confound-controls-v1"

#: Mobility-classification thresholds (single tunable source of truth, not inline magic numbers).
#: A regime counts as non-confined ("mobile") only if travellers visit at least
#: ``MOBILITY_MIN_MEAN_DISTINCT_CELLS`` distinct cells on average AND at most
#: ``MOBILITY_MAX_CONFINED_FRACTION`` of them are confined to ``<= MOBILITY_CONFINED_CELL_MAX`` cells.
#: Pass overrides to :func:`mobility_summary` to retune without editing the module.
MOBILITY_MIN_MEAN_DISTINCT_CELLS: float = 5.0
MOBILITY_CONFINED_CELL_MAX: int = 3
MOBILITY_MAX_CONFINED_FRACTION: float = 0.5


def mobility_summary(
    sequence: ObservationSequence,
    *,
    min_mean_distinct_cells: float = MOBILITY_MIN_MEAN_DISTINCT_CELLS,
    confined_cell_max: int = MOBILITY_CONFINED_CELL_MAX,
    max_confined_fraction: float = MOBILITY_MAX_CONFINED_FRACTION,
) -> dict[str, object]:
    """Per-traveller spatial mobility — distinct cells and occupancy entropy. A regime where the
    mean distinct-cell count is small (travellers confined to a few cells) makes any individual model
    win by information congruence, so model claims there are meaningless."""

    cells = np.asarray(sequence.spatial_states, dtype=int)
    travellers = np.asarray([r.traveller_id for r in sequence.observations], dtype=int)
    n = int(sequence.num_spatial_states)
    rows = []
    for t in sorted(set(travellers.tolist())):
        c = cells[travellers == t]
        vc = np.bincount(c, minlength=n).astype(float)
        p = vc[vc > 0] / vc.sum()
        rows.append(
            {
                "traveller": int(t),
                "distinct_cells": int((vc > 0).sum()),
                "dominant_share": float(vc.max() / vc.sum()),
                "entropy_bits": float(-(p * np.log2(p)).sum()),
            }
        )
    mean_distinct = float(np.mean([r["distinct_cells"] for r in rows]))
    confined_fraction = float(
        np.mean([r["distinct_cells"] <= confined_cell_max for r in rows])
    )
    # non-confined ("mobile") iff travellers visit enough distinct cells AND most are not confined.
    mean_distinct_meets_threshold = bool(mean_distinct >= min_mean_distinct_cells)
    confined_fraction_within_threshold = bool(
        confined_fraction <= max_confined_fraction
    )
    return {
        "schema": CONFOUND_CONTROLS_SCHEMA,
        "n_cells": n,
        "per_traveller": rows,
        "mean_distinct_cells": mean_distinct,
        "mean_entropy_bits": float(np.mean([r["entropy_bits"] for r in rows])),
        "confined_fraction": confined_fraction,
        "is_mobile": bool(
            mean_distinct_meets_threshold and confined_fraction_within_threshold
        ),
        # provenance: the thresholds used and which side of each this dataset landed on, so a
        # mobile/confined verdict is traceable to the exact cutoffs rather than an opaque bool.
        "mobility_thresholds": {
            "min_mean_distinct_cells": float(min_mean_distinct_cells),
            "confined_cell_max": int(confined_cell_max),
            "max_confined_fraction": float(max_confined_fraction),
        },
        "mean_distinct_meets_threshold": mean_distinct_meets_threshold,
        "confined_fraction_within_threshold": confined_fraction_within_threshold,
    }


def confound_battery(
    sequence: ObservationSequence,
    *,
    smoothing: float = 0.05,
    holdout_fraction: float = 0.5,
    seed: int = 20260616,
) -> dict[str, float]:
    """Score the four confound baselines on the locked split (mean next-cell loss, nats)."""

    cells = np.asarray(sequence.spatial_states, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    travellers = np.asarray([r.traveller_id for r in sequence.observations], dtype=int)
    n = int(sequence.num_spatial_states)
    split = lock_traveller_split(sequence, holdout_fraction=holdout_fraction, seed=seed)
    is_holdout = np.asarray(split["is_holdout"], dtype=bool)
    train = ~is_holdout
    A = _emission_matrix(cells[train], cells[train], n, n, smoothing)
    post = A[cells, :] / A[cells, :].sum(axis=1, keepdims=True)

    # pooled transition from train travellers
    counts = np.full((n, n), smoothing, dtype=float)
    last = None
    for i in range(len(cells)):
        if first[i] or travellers[i] != last:
            last = travellers[i]
            continue
        if train[i]:
            counts[int(cells[i]), int(cells[i - 1])] += 1.0
    B = counts / counts.sum(axis=0, keepdims=True)

    dom: list[float] = []
    fop: list[float] = []
    orc: list[float] = []
    visits: dict[int, np.ndarray] = {}
    last = None
    for i in range(len(cells)):
        tid = int(travellers[i])
        v = visits.setdefault(tid, np.full(n, smoothing))
        if first[i] or tid != last:
            last = tid
            v[int(cells[i])] += 1.0
            continue
        true_cell = int(cells[i])
        if is_holdout[i]:
            # dominant-cell: the traveller's own online visit frequency
            dom.append(-math.log(max(float((v / v.sum())[true_cell]), 1e-12)))
            # first-order pooled, inferred prev (soft)
            soft = B @ post[i - 1]
            fop.append(-math.log(max(float((soft / soft.sum())[true_cell]), 1e-12)))
            # oracle prev (ceiling)
            opred = B[:, int(cells[i - 1])] / B[:, int(cells[i - 1])].sum()
            orc.append(-math.log(max(float(opred[true_cell]), 1e-12)))
        v[int(cells[i])] += 1.0
    # return-boosted soft (strongest fair non-recursive competitor): delegate to
    # fair_inference._markov_loss, the single source of truth for this named
    # quantity, instead of re-deriving it from a fixed train-only transition here.
    # That keeps the two modules' claims about "the strongest fair non-recursive
    # competitor" numerically identical rather than merely identically worded.
    return_boosted_soft = _markov_loss(
        cells,
        first,
        travellers,
        is_holdout,
        n,
        smoothing,
        posterior=post,
        return_boost=1.0,
    )[0]
    return {
        "dominant_cell": float(np.mean(dom)),
        "first_order_pooled": float(np.mean(fop)),
        "return_boosted_soft": float(return_boosted_soft),
        "oracle_prev": float(np.mean(orc)),
    }


def clears_confounds(
    model_loss: float, baselines: dict[str, float]
) -> dict[str, object]:
    """A model genuinely wins only if it beats the dominant-cell, first-order-pooled, and
    return-boosted-soft baselines. The oracle gap is reported, never required to be cleared."""

    fair = ("dominant_cell", "first_order_pooled", "return_boosted_soft")
    beaten = {k: bool(model_loss < baselines[k]) for k in fair}
    return {
        "model_loss": float(model_loss),
        "beats": beaten,
        "clears_confounds": bool(all(beaten.values())),
        "gap_to_oracle_nats": float(model_loss - baselines["oracle_prev"]),
        "verdict": (
            "clears the confound battery (beats dominant-cell, pooled-Markov, and return-boosted soft)"
            if all(beaten.values())
            else "does NOT clear the confounds — any apparent advantage is a confound, not capability"
        ),
    }
