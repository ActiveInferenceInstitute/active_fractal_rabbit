"""Privacy-utility frontier — the dual of the re-identification frontier.

`privacy_frontier.py` measures re-identification RISK (unicity, co-travel) as location is
minimized. This module measures the other half of the governance tradeoff: how much next-cell
PREDICTIVE UTILITY survives each minimization regime, paired with the re-identification risk at
that regime, so a reader can see what useful inference is retained as location is coarsened or
withheld and what privacy is bought in exchange.

Two confounds make a naive utility axis dishonest, and both are controlled here (the controls
were established by a prototype-before-claim pass):

1. **Target cardinality.** Predicting one of a few coarse cells is intrinsically easier than one
   of many fine cells, so skill-vs-base-rate and absolute next-cell loss both *improve* as the
   grid coarsens — saying nothing about retained utility. We therefore report a resolution-fair
   axis, ``resolved_bits = log2(n_states) - loss_bits`` (bits of next-location actually pinned),
   which is comparable across grids.
2. **Static occupancy.** At a fine grid a temporally-shuffled predictor still scores well by
   leaning on a peaked marginal-occupancy distribution. We subtract a label-shuffle floor to get
   the **shuffle-corrected excess utility** — the bits attributable to temporal/predictive
   structure rather than to static occupancy.

The honest test is a kill-gate for graceful degradation: excess utility and re-identification
risk must both fall as location minimization increases before that wording is promoted. A
non-monotone full-grid ladder is therefore a result, not a failure to hide. Everything here is a
synthetic-data software measurement on the pinned generator; nothing is a real-mobility,
operational, or surveillance claim, and the risk metrics are oversight-positive minimization
diagnostics, not association capability over real people.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from .privacy_frontier import co_travel_coincidence_rate, resolve_observations_path
from .sequence_baselines import run_sequence_baseline_analysis
from .sota_program_harness import build_inference_config

PRIVACY_UTILITY_FRONTIER_SCHEMA = "active-fractal-rabbit-privacy-utility-frontier-v1"

#: Spatial-grid minimization regimes, fine -> coarse (square grids of ``bins x bins`` cells).
#: A denser ladder than the original (8, 6, 4, 3, 2) so the minimum-collection operating point
#: is resolved on a finer disclosure axis.
DEFAULT_GRID_REGIMES: tuple[int, ...] = (8, 7, 6, 5, 4, 3, 2)
DEFAULT_SHUFFLE_TRIALS = 5
DEFAULT_SHUFFLE_SEED = 20260620
_LN2 = math.log(2.0)

#: Minimum-collection operating-point thresholds (falsifiable by raising the floor / lowering the
#: ceiling). The operating point is the COARSEST grid (least location disclosed) that still retains
#: useful next-cell utility while holding re-identification risk below the ceiling; if none exists,
#: that is an honest null — location coarsening alone cannot buy that privacy target without
#: destroying utility on this synthetic data.
UTILITY_FLOOR_BITS = 0.10
REID_RISK_CEILING = 0.20


class PrivacyUtilityError(RuntimeError):
    """Raised when the privacy-utility frontier cannot be assembled."""


def _resolved_bits(best_loss_nats: float, n_states: int) -> float:
    """Bits of next-location actually pinned: ``log2(n_states) - loss_in_bits`` (>=0 means better than blind)."""
    return math.log2(max(n_states, 1)) - (best_loss_nats / _LN2)


def event_uniqueness(
    traveller_ids: np.ndarray, cells: np.ndarray, time_bins: np.ndarray
) -> float:
    """Fraction of ``(time_bin, cell)`` events occupied by exactly one distinct traveller.

    A non-degenerate re-identification proxy (``trace_unicity`` saturates at 1.0 once every
    traveller's full signature is unique); this asks, per spatio-temporal event, how singling-out
    it is. Falls as location is coarsened because coarse cells pool travellers.
    """
    traveller_ids = np.asarray(traveller_ids, dtype=int)
    cells = np.asarray(cells, dtype=int)
    time_bins = np.asarray(time_bins, dtype=int)
    if traveller_ids.size == 0:
        return 0.0
    by_event: dict[tuple[int, int], set[int]] = {}
    for t, c, trav in zip(time_bins, cells, traveller_ids):
        by_event.setdefault((int(t), int(c)), set()).add(int(trav))
    if not by_event:
        return 0.0
    singletons = sum(1 for occupants in by_event.values() if len(occupants) == 1)
    return float(singletons / len(by_event))


def _best_loss_nats(sequence, smoothing: float) -> float:
    panel = run_sequence_baseline_analysis(sequence, smoothing=smoothing)
    return float(panel["best_mean_predictive_loss"])


def _shuffle_floor_resolved_bits(
    sequence, n_states: int, smoothing: float, *, n_trials: int, seed: int
) -> tuple[float, float]:
    """Mean and Monte-Carlo SD of resolved-bits of the best predictor on globally-permuted cells.

    A global permutation of the cell labels preserves the marginal occupancy distribution while
    destroying every temporal and per-traveller dependency, so the surviving signal is the
    static-occupancy floor. Subtracting its mean isolates utility from temporal/predictive
    structure; the SD across permutation trials reports how tight that floor estimate is (a small
    SD means the excess-utility correction is Monte-Carlo stable, not a single-draw artifact).
    """
    cells = np.asarray(sequence.spatial_states, dtype=int)
    rng = np.random.default_rng(seed)
    floors: list[float] = []
    for _ in range(n_trials):
        permuted = cells.copy()
        rng.shuffle(permuted)
        shuffled_seq = replace(sequence, spatial_states=tuple(int(c) for c in permuted))
        floors.append(
            _resolved_bits(_best_loss_nats(shuffled_seq, smoothing), n_states)
        )
    if not floors:
        return 0.0, 0.0
    return float(np.mean(floors)), float(np.std(floors, ddof=1)) if len(
        floors
    ) > 1 else 0.0


def privacy_utility_frontier(
    config_path: str | Path,
    observations_path: str | Path,
    *,
    grid_regimes: tuple[int, ...] = DEFAULT_GRID_REGIMES,
    n_shuffle: int = DEFAULT_SHUFFLE_TRIALS,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
) -> dict[str, object]:
    """Compute the paired predictive-utility / re-identification-risk frontier over grids."""

    from .config import load_config
    from .fractalrabbit import parse_waypoint_csv
    from .observations import discretize_observations

    base_config = load_config(config_path)
    try:
        obs_path = resolve_observations_path(
            observations_path, base_config.fractalrabbit.fixture_csv
        )
    except ValueError as exc:
        raise PrivacyUtilityError(str(exc)) from exc
    observations = parse_waypoint_csv(obs_path)
    smoothing = float(base_config.inference.model.smoothing)

    rows: list[dict[str, object]] = []
    for bins in grid_regimes:
        inference_config = build_inference_config(config_path, spatial_bins=bins)
        sequence = discretize_observations(observations, inference_config)
        n_states = int(sequence.num_spatial_states)
        traveller_ids = np.asarray(
            [row.traveller_id for row in sequence.observations], dtype=int
        )
        cells = np.asarray(sequence.spatial_states, dtype=int)
        time_bins = np.asarray(sequence.time_bins, dtype=int)

        resolved = _resolved_bits(_best_loss_nats(sequence, smoothing), n_states)
        shuffle_floor, shuffle_floor_sd = _shuffle_floor_resolved_bits(
            sequence, n_states, smoothing, n_trials=n_shuffle, seed=shuffle_seed
        )
        rows.append(
            {
                "regime": f"grid_{bins}x{bins}",
                "spatial_bins": int(bins),
                "n_states": n_states,
                "resolved_bits": round(resolved, 6),
                "shuffle_floor_bits": round(shuffle_floor, 6),
                "shuffle_floor_sd": round(shuffle_floor_sd, 6),
                "excess_utility_bits": round(resolved - shuffle_floor, 6),
                "reid_event_uniqueness": round(
                    event_uniqueness(traveller_ids, cells, time_bins), 6
                ),
                "co_travel_coincidence": round(
                    co_travel_coincidence_rate(traveller_ids, cells, time_bins), 6
                ),
            }
        )

    # Withheld-location floor: with no location modality there is no next-cell to predict, so the
    # surviving location utility is identically zero — the minimization endpoint.
    withheld = {
        "regime": "location_withheld",
        "spatial_bins": 0,
        "n_states": 0,
        "resolved_bits": 0.0,
        "shuffle_floor_bits": 0.0,
        "shuffle_floor_sd": 0.0,
        "excess_utility_bits": 0.0,
        "reid_event_uniqueness": 0.0,
        "co_travel_coincidence": 0.0,
        "note": "location modality withheld; no next-cell target exists, so location utility is zero by construction",
    }

    excess = [float(r["excess_utility_bits"]) for r in rows]
    risk = [float(r["reid_event_uniqueness"]) for r in rows]
    # rows are fine -> coarse; minimization increases left -> right, so honest degradation is
    # a monotone NON-INCREASING sequence on both axes.
    utility_monotone = all(
        excess[i] >= excess[i + 1] - 1e-9 for i in range(len(excess) - 1)
    )
    risk_monotone = all(risk[i] >= risk[i + 1] - 1e-9 for i in range(len(risk) - 1))
    # Negative-control summary: at the coarsest grid the shuffle floor should account for nearly
    # all resolved bits (excess -> ~0), i.e. little temporal structure remains to launder.
    coarsest_excess = excess[-1] if excess else 0.0
    # Monte-Carlo stability of the shuffle-floor correction: the worst-case SD across grids should
    # be small relative to the excess-utility magnitudes for the correction to be trustworthy.
    max_floor_sd = max((float(r["shuffle_floor_sd"]) for r in rows), default=0.0)

    # Minimum-collection operating point: among location-minimization grids, the COARSEST (least
    # disclosed) whose shuffle-corrected excess utility still clears the floor while re-identification
    # event-uniqueness stays under the ceiling. Iterating fine -> coarse and keeping the last match
    # lands on the most-minimized acceptable grid. None -> honest null (the privacy/utility targets
    # are not jointly reachable by coarsening alone on this synthetic data).
    operating = None
    for r in rows:
        if (
            float(r["excess_utility_bits"]) >= UTILITY_FLOOR_BITS
            and float(r["reid_event_uniqueness"]) < REID_RISK_CEILING
        ):
            operating = r
    min_collection = {
        "found": operating is not None,
        "utility_floor_bits": UTILITY_FLOOR_BITS,
        "reid_risk_ceiling": REID_RISK_CEILING,
        "regime": operating["regime"] if operating else "none",
        "spatial_bins": int(operating["spatial_bins"]) if operating else 0,
        "excess_utility_bits": float(operating["excess_utility_bits"])
        if operating
        else 0.0,
        "reid_event_uniqueness": float(operating["reid_event_uniqueness"])
        if operating
        else 0.0,
        # the binding constraint when no operating point exists: the lowest re-id achieved by any
        # utility-bearing grid (so the honest null can say *why* — re-id floors above the ceiling).
        "min_reid_among_useful_grids": round(
            min(
                (
                    float(r["reid_event_uniqueness"])
                    for r in rows
                    if float(r["excess_utility_bits"]) >= UTILITY_FLOOR_BITS
                ),
                default=0.0,
            ),
            6,
        ),
    }

    return {
        "schema": PRIVACY_UTILITY_FRONTIER_SCHEMA,
        "source": str(obs_path),
        "smoothing": smoothing,
        "n_shuffle_trials": int(n_shuffle),
        "grid_regimes": list(grid_regimes),
        "frontier": rows + [withheld],
        "utility_monotone_decreasing": bool(utility_monotone),
        "risk_monotone_decreasing": bool(risk_monotone),
        "graceful_degradation_tradeoff": bool(utility_monotone and risk_monotone),
        "min_collection_operating_point": min_collection,
        "max_shuffle_floor_sd": round(max_floor_sd, 6),
        "finest_excess_utility_bits": round(excess[0], 6) if excess else 0.0,
        "coarsest_excess_utility_bits": round(coarsest_excess, 6),
        "finest_reid_event_uniqueness": round(risk[0], 6) if risk else 0.0,
        "coarsest_reid_event_uniqueness": round(risk[-1], 6) if risk else 0.0,
        "boundary": (
            "Synthetic-data software measurement on the pinned generator. Predictive utility is "
            "resolution-fair (bits pinned beyond chance) and shuffle-corrected (beyond static "
            "occupancy); re-identification metrics are oversight-positive minimization diagnostics, "
            "not association or tasking capability over real people. No real-mobility, operational, "
            "or state-of-the-art claim."
        ),
    }


def run_and_write_privacy_utility_frontier(
    config_path: str | Path,
    observations_path: str | Path,
    out_path: str | Path,
) -> Path:
    """Build the frontier and write ``privacy_utility_frontier.json``."""

    payload = privacy_utility_frontier(config_path, observations_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out
