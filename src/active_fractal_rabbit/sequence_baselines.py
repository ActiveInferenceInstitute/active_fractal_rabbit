"""Prequential sequence baselines for the state-of-the-art gate.

The project already compares the active-inference trace with transparent
Bayes-style filtering and persistence nulls. This module adds a stronger
baseline ladder without adding new dependencies: online base-rate, persistence,
first-order Markov, phase-conditioned Markov, metadata-conditioned Markov, and
return-boosted Markov predictors. These are not presented as final state of the
art; they are executable gates that make any future performance claim harder to
overstate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import load_config
from .fractalrabbit import parse_waypoint_csv
from .observations import discretize_observations
from .types import ObservationSequence

SEQUENCE_BASELINE_SCHEMA = "active-fractal-rabbit-sequence-baseline-v1"


def run_sequence_baseline_analysis(
    sequence: ObservationSequence,
    *,
    smoothing: float,
    reference_loss: float | None = None,
    pymdp_loss: float | None = None,
) -> dict[str, object]:
    """Score prequential next-cell baselines on a discretized sequence."""

    cells = np.asarray(sequence.spatial_states, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    phases = np.asarray(sequence.time_bins, dtype=int)
    speed = np.asarray(sequence.speed_categories, dtype=int)
    burst = np.asarray(sequence.burst_categories, dtype=int)
    travellers = np.asarray(
        [row.traveller_id for row in sequence.observations], dtype=int
    )
    n_states = int(sequence.num_spatial_states)
    n_scored = int((~first).sum())
    if n_scored <= 0:
        raise ValueError(
            "sequence baseline analysis needs at least one non-first observation"
        )

    phase_context = phases
    n_phase_contexts = int(sequence.num_time_bins)
    metadata_context = (
        (phases * sequence.num_speed_categories + speed) * sequence.num_burst_categories
    ) + burst
    n_metadata_contexts = int(
        sequence.num_time_bins
        * sequence.num_speed_categories
        * sequence.num_burst_categories
    )

    raw_rows = [
        (
            "base_rate",
            "online marginal-frequency next-cell baseline",
            _base_rate_loss(cells, first, n_states, smoothing),
        ),
        (
            "persistence",
            "smoothed previous-cell persistence null",
            _persistence_loss(cells, first, n_states, smoothing),
        ),
        (
            "first_order_markov",
            "online P(cell_t | cell_{t-1}) sequence baseline",
            _context_markov_loss(
                cells, first, np.zeros_like(cells), 1, n_states, smoothing
            ),
        ),
        (
            "phase_markov",
            "online P(cell_t | cell_{t-1}, time phase) sequence baseline",
            _context_markov_loss(
                cells, first, phase_context, n_phase_contexts, n_states, smoothing
            ),
        ),
        (
            "metadata_markov",
            "online P(cell_t | cell_{t-1}, phase, speed, burst) sequence baseline",
            _context_markov_loss(
                cells, first, metadata_context, n_metadata_contexts, n_states, smoothing
            ),
        ),
        (
            "return_boost_markov",
            "first-order Markov predictor with per-traveller visited-cell return boost",
            _return_boost_loss(
                cells, first, travellers, np.zeros_like(cells), 1, n_states, smoothing
            ),
        ),
        (
            "phase_return_markov",
            "phase-conditioned Markov predictor with per-traveller visited-cell return boost",
            _return_boost_loss(
                cells,
                first,
                travellers,
                phase_context,
                n_phase_contexts,
                n_states,
                smoothing,
            ),
        ),
    ]
    base_rate_loss = raw_rows[0][2]
    rows = [
        {
            "model": model,
            "current_role": role,
            "mean_predictive_loss": loss,
            "skill_vs_base_rate": _skill(base_rate_loss, loss),
            "n_scored_steps": n_scored,
        }
        for model, role, loss in raw_rows
    ]
    best = min(rows, key=lambda item: float(item["mean_predictive_loss"]))
    best_loss = float(best["mean_predictive_loss"])
    reference_delta = _delta(best_loss, reference_loss)
    pymdp_delta = _delta(best_loss, pymdp_loss)
    if pymdp_loss is None:
        gate = "pymdp_unavailable_for_sequence_gate"
    elif best_loss < pymdp_loss:
        gate = "stronger_sequence_baseline_beats_pymdp_on_this_run"
    else:
        gate = "pymdp_beats_sequence_baselines_on_this_run"

    return {
        "schema": SEQUENCE_BASELINE_SCHEMA,
        "status": "completed",
        "n_observations": int(len(cells)),
        "n_scored_steps": n_scored,
        "n_states": n_states,
        "smoothing": float(smoothing),
        "rows": rows,
        "best_model": str(best["model"]),
        "best_mean_predictive_loss": best_loss,
        "best_skill_vs_base_rate": float(best["skill_vs_base_rate"]),
        "best_minus_reference_predictive_loss": reference_delta,
        "best_minus_pymdp_predictive_loss": pymdp_delta,
        "pymdp_performance_gate": gate,
        "caveat": (
            "Prequential sequence baselines are stronger synthetic nulls for performance claims. "
            "They are not full state-of-the-art movement models and do not use empirical mobility traces."
        ),
    }


def run_and_write_sequence_baselines(
    config_path: str | Path, out_path: str | Path
) -> Path:
    """Run the sequence-baseline gate for ``config_path`` and write JSON."""

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
    metrics = _read_json(config.output_dir / "metrics.json", {})
    payload = run_sequence_baseline_analysis(
        sequence,
        smoothing=config.inference.smoothing,
        reference_loss=_optional_float(metrics.get("mean_predictive_loss")),
        pymdp_loss=_optional_float(metrics.get("pymdp_mean_predictive_loss")),
    )
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def _base_rate_loss(
    cells: np.ndarray, first: np.ndarray, n_states: int, smoothing: float
) -> float:
    counts = np.full(n_states, smoothing, dtype=float)
    losses: list[float] = []
    for cell, is_first in zip(cells, first):
        probs = counts / counts.sum()
        if not is_first:
            losses.append(_neg_log(probs[int(cell)]))
        counts[int(cell)] += 1.0
    return _mean_loss(losses)


def _persistence_loss(
    cells: np.ndarray, first: np.ndarray, n_states: int, smoothing: float
) -> float:
    """Config-smoothed previous-cell persistence null (skips first observations).

    This intentionally differs from ``factorial_pymdp._stronger_baseline_metrics``'s
    "hard" persistence loss, which uses a fixed ``1e-6`` smoothing constant
    (not the ``smoothing`` argument here) and scores first observations too
    (via a base-rate fallback instead of skipping them). Both are legitimate
    persistence-null conventions with different intents (config-driven
    smoothed null vs. fixed near-deterministic null); see that function's
    docstring for the full contrast.
    """
    losses: list[float] = []
    for idx, (cell, is_first) in enumerate(zip(cells, first)):
        if is_first:
            continue
        probs = np.full(n_states, smoothing, dtype=float)
        probs[int(cells[idx - 1])] += 1.0
        probs /= probs.sum()
        losses.append(_neg_log(probs[int(cell)]))
    return _mean_loss(losses)


def _context_markov_loss(
    cells: np.ndarray,
    first: np.ndarray,
    contexts: np.ndarray,
    n_contexts: int,
    n_states: int,
    smoothing: float,
) -> float:
    counts = np.full((n_contexts, n_states, n_states), smoothing, dtype=float)
    losses: list[float] = []
    for idx, (cell, is_first) in enumerate(zip(cells, first)):
        if is_first:
            continue
        context = int(contexts[idx])
        previous = int(cells[idx - 1])
        probs = counts[context, previous] / counts[context, previous].sum()
        losses.append(_neg_log(probs[int(cell)]))
        counts[context, previous, int(cell)] += 1.0
    return _mean_loss(losses)


def _return_boost_loss(
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
    counts = np.full((n_contexts, n_states, n_states), smoothing, dtype=float)
    visited: dict[int, np.ndarray] = {}
    losses: list[float] = []
    for idx, (cell, is_first, traveller) in enumerate(zip(cells, first, travellers)):
        traveller_id = int(traveller)
        seen = visited.setdefault(traveller_id, np.zeros(n_states, dtype=bool))
        if is_first:
            seen[int(cell)] = True
            continue
        context = int(contexts[idx])
        previous = int(cells[idx - 1])
        scores = counts[context, previous].copy()
        scores[seen] += return_weight
        probs = scores / scores.sum()
        losses.append(_neg_log(probs[int(cell)]))
        counts[context, previous, int(cell)] += 1.0
        seen[int(cell)] = True
    return _mean_loss(losses)


def _mean_loss(losses: list[float]) -> float:
    if not losses:
        raise ValueError("no scored observations for sequence baseline")
    return float(np.mean(losses))


def _neg_log(probability: float) -> float:
    return float(-math.log(max(float(probability), 1e-12)))


def _skill(null_loss: float, model_loss: float) -> float:
    if null_loss <= 0:
        return 0.0
    return float((null_loss - model_loss) / null_loss)


def _delta(best_loss: float, other_loss: float | None) -> float | None:
    return None if other_loss is None else float(best_loss - other_loss)


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
