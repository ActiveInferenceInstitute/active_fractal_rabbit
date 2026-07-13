"""Return-plus-phase prior gate for synthetic waypoint prediction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import load_config
from .fractalrabbit import parse_waypoint_csv
from .observations import discretize_observations
from .sequence_baselines import _neg_log, _persistence_loss, _skill
from .types import ObservationSequence

RETURN_PHASE_PRIOR_SCHEMA = "active-fractal-rabbit-return-phase-prior-v1"


def run_return_phase_prior_analysis(
    sequence: ObservationSequence,
    *,
    smoothing: float,
    seed: int = 0,
    promotion_threshold: float = 0.01,
) -> dict[str, object]:
    """Score pooled, phase, return, and return-plus-phase next-cell priors."""

    cells = np.asarray(sequence.spatial_states, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    phases = np.asarray(sequence.time_bins, dtype=int)
    travellers = np.asarray(
        [row.traveller_id for row in sequence.observations], dtype=int
    )
    n_states = int(sequence.num_spatial_states)
    n_phases = int(sequence.num_time_bins)
    n_scored = int((~first).sum())
    if n_scored <= 0:
        raise ValueError(
            "return-plus-phase prior needs at least one non-first observation"
        )

    pooled = _context_prior_loss(
        cells, first, np.zeros_like(phases), 1, n_states, smoothing
    )
    phase = _context_prior_loss(cells, first, phases, n_phases, n_states, smoothing)
    return_only = _return_prior_loss(
        cells,
        first,
        travellers,
        np.zeros_like(phases),
        1,
        n_states,
        smoothing,
    )
    return_phase = _return_prior_loss(
        cells,
        first,
        travellers,
        phases,
        n_phases,
        n_states,
        smoothing,
    )
    persistence = _persistence_loss(cells, first, n_states, smoothing)
    rng = np.random.default_rng(seed)
    permuted_phases = rng.permutation(phases)
    permuted_return_phase = _return_prior_loss(
        cells,
        first,
        travellers,
        permuted_phases,
        n_phases,
        n_states,
        smoothing,
    )

    rows = [
        _row("base_rate_null", "pooled online marginal-frequency null", pooled, pooled),
        _row(
            "persistence_null",
            "smoothed previous-cell persistence null",
            persistence,
            pooled,
        ),
        _row("pooled_prior", "pooled categorical next-cell prior", pooled, pooled),
        _row(
            "phase_prior",
            "phase-conditioned categorical next-cell prior",
            phase,
            pooled,
        ),
        _row(
            "return_prior",
            "per-traveller visited-cell return prior",
            return_only,
            pooled,
        ),
        _row(
            "return_plus_phase_prior",
            "phase-conditioned prior with visited-cell return boost",
            return_phase,
            pooled,
        ),
        _row(
            "permuted_return_plus_phase_prior",
            "return-plus-phase prior with phase labels permuted",
            permuted_return_phase,
            pooled,
        ),
    ]
    candidate_names = {
        "pooled_prior",
        "phase_prior",
        "return_prior",
        "return_plus_phase_prior",
    }
    candidate_rows = [row for row in rows if row["model"] in candidate_names]
    best = min(candidate_rows, key=lambda item: float(item["mean_predictive_loss"]))
    gain_vs_persistence = float(persistence - return_phase)
    gain_over_permuted = float(permuted_return_phase - return_phase)
    best_competing_loss = min(
        float(row["mean_predictive_loss"])
        for row in candidate_rows
        if row["model"] != "return_plus_phase_prior"
    )
    verdict = (
        "promoted_return_phase_prior"
        if (
            str(best["model"]) == "return_plus_phase_prior"
            and gain_vs_persistence > promotion_threshold
            and gain_over_permuted > promotion_threshold
            and best_competing_loss - return_phase > promotion_threshold
        )
        else "honest_null"
    )

    return {
        "schema": RETURN_PHASE_PRIOR_SCHEMA,
        "status": "completed",
        "verdict": verdict,
        "n_observations": int(len(cells)),
        "n_scored_steps": n_scored,
        "n_states": n_states,
        "n_phases": n_phases,
        "smoothing": float(smoothing),
        "promotion_threshold": float(promotion_threshold),
        "best_model": str(best["model"]),
        "best_mean_predictive_loss": float(best["mean_predictive_loss"]),
        "return_plus_phase_loss": float(return_phase),
        "return_plus_phase_gain_vs_persistence": gain_vs_persistence,
        "return_plus_phase_gain_over_permuted": gain_over_permuted,
        "return_plus_phase_gain_vs_best_competing_prior": float(
            best_competing_loss - return_phase
        ),
        "rows": rows,
        "caveat": (
            "This is a synthetic prior gate. It promotes a return-plus-phase prior only when it beats "
            "pooled, phase-only, return-only, persistence, and permuted-phase comparators on the current lane."
        ),
    }


def run_and_write_return_phase_prior(
    config_path: str | Path, out_path: str | Path
) -> Path:
    """Run the return-plus-phase prior gate for a project config and write JSON."""

    config = load_config(config_path)
    if config.fractalrabbit.execution == "fixture":
        if config.fractalrabbit.fixture_csv is None:
            raise ValueError("fixture execution requires fixture_csv")
        obs_path = config.fractalrabbit.fixture_csv
    else:
        obs_path = config.output_dir / f"{config.fractalrabbit.output_prefix}XY.csv"
        if not obs_path.exists():
            obs_path = config.output_dir / "observations.csv"
    sequence = discretize_observations(parse_waypoint_csv(obs_path), config.inference)
    payload = run_return_phase_prior_analysis(
        sequence, smoothing=config.inference.smoothing, seed=config.seed
    )
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def unavailable_return_phase_prior(reason: str) -> dict[str, object]:
    """Return an explicit unavailable artifact for a missing lane."""

    return {
        "schema": RETURN_PHASE_PRIOR_SCHEMA,
        "status": "unavailable",
        "verdict": "unavailable",
        "reason": reason,
        "rows": [],
        "caveat": "No numeric return-plus-phase prior result is emitted for an unavailable lane.",
    }


def _context_prior_loss(
    cells: np.ndarray,
    first: np.ndarray,
    contexts: np.ndarray,
    n_contexts: int,
    n_states: int,
    smoothing: float,
) -> float:
    counts = np.full((n_contexts, n_states), smoothing, dtype=float)
    losses: list[float] = []
    for cell, is_first, context in zip(cells, first, contexts):
        probs = counts[int(context)] / counts[int(context)].sum()
        if not is_first:
            losses.append(_neg_log(probs[int(cell)]))
        counts[int(context), int(cell)] += 1.0
    return _mean_loss(losses)


def _return_prior_loss(
    cells: np.ndarray,
    first: np.ndarray,
    travellers: np.ndarray,
    contexts: np.ndarray,
    n_contexts: int,
    n_states: int,
    smoothing: float,
    *,
    return_weight: float = 1.0,
) -> float:
    counts = np.full((n_contexts, n_states), smoothing, dtype=float)
    visited: dict[int, np.ndarray] = {}
    losses: list[float] = []
    for cell, is_first, traveller, context in zip(cells, first, travellers, contexts):
        traveller_id = int(traveller)
        seen = visited.setdefault(traveller_id, np.zeros(n_states, dtype=bool))
        scores = counts[int(context)].copy()
        scores[seen] += return_weight
        probs = scores / scores.sum()
        if not is_first:
            losses.append(_neg_log(probs[int(cell)]))
        counts[int(context), int(cell)] += 1.0
        seen[int(cell)] = True
    return _mean_loss(losses)


def _row(model: str, role: str, loss: float, base_loss: float) -> dict[str, object]:
    return {
        "model": model,
        "current_role": role,
        "mean_predictive_loss": float(loss),
        "skill_vs_base_rate": _skill(base_loss, loss),
    }


def _mean_loss(losses: list[float]) -> float:
    if not losses:
        raise ValueError("no scored observations for return-plus-phase prior")
    return float(np.mean(losses))
