"""Return-structured priors over visited cells.

FractalRabbit's trajectory process has long-memory return structure. This
module asks whether a filter-realizable visited-cell feature reduces move-step
uncertainty enough to justify promoting a return-conditioned prior, or whether
the result should stay an honest null.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .objectives import brier_with_base_rate_fallback, persistence_brier, skill_score
from .pomdp import _normalize_columns
from .types import ObservationSequence


def _entropy(labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def _conditional_entropy(target: np.ndarray, *conditions: np.ndarray) -> float:
    if target.size == 0:
        return 0.0
    groups: dict[tuple[int, ...], list[int]] = {}
    for idx, value in enumerate(target):
        key = tuple(int(cond[idx]) for cond in conditions)
        groups.setdefault(key, []).append(int(value))
    total = target.size
    return float(
        sum(
            (len(values) / total) * _entropy(np.asarray(values, dtype=int))
            for values in groups.values()
        )
    )


def conditional_entropy_reduction(
    previous_cell: np.ndarray, visited_feature: np.ndarray, next_cell: np.ndarray
) -> float:
    """Return ``H(next|prev) - H(next|prev, visited)`` in nats."""

    previous_cell = np.asarray(previous_cell, dtype=int)
    visited_feature = np.asarray(visited_feature, dtype=int)
    next_cell = np.asarray(next_cell, dtype=int)
    return max(
        0.0,
        _conditional_entropy(next_cell, previous_cell)
        - _conditional_entropy(next_cell, previous_cell, visited_feature),
    )


def return_conditioned_transition(
    spatial_states: np.ndarray,
    is_first_observation: np.ndarray,
    n_states: int,
    smoothing: float,
) -> np.ndarray:
    """Estimate ``P(next_cell | previous_cell, destination_was_visited)``."""

    cells = np.asarray(spatial_states, dtype=int)
    first = np.asarray(is_first_observation, dtype=bool)
    tensor = np.full((n_states, n_states, 2), smoothing, dtype=float)
    visited: set[int] = set()
    for idx, cell in enumerate(cells):
        if first[idx]:
            visited = {int(cell)}
            continue
        destination_visited = int(int(cell) in visited)
        tensor[int(cell), int(cells[idx - 1]), destination_visited] += 1.0
        visited.add(int(cell))
    for feature in range(2):
        tensor[:, :, feature] = _normalize_columns(tensor[:, :, feature])
    return tensor


def return_conditioned_brier(
    spatial_states: np.ndarray,
    is_first_observation: np.ndarray,
    n_states: int,
    tensor: np.ndarray,
) -> float:
    """Brier score of the filter-realizable return-conditioned predictor.

    At each step this scores the *return-conditioned prior itself*, not the
    location-observed POMDP posterior. For every candidate destination cell
    ``c`` the predictor reads ``P(c | previous_cell, visited=(c already seen in
    this traveller's trace so far))`` from ``tensor`` and renormalizes over
    candidates. The visited indicator is filter-realizable — it depends only on
    the history strictly before the predicted step, matching the feature used to
    estimate ``tensor``. At each traveller's first report (no previous cell) it
    falls back to the base-rate frequency vector, exactly as
    :func:`objectives.persistence_brier` does (both route through the shared
    :func:`objectives.brier_with_base_rate_fallback` helper), so the two
    scores are comparable.
    """

    cells = np.asarray(spatial_states, dtype=int)
    first = np.asarray(is_first_observation, dtype=bool)
    n_steps = cells.shape[0]
    if n_steps == 0:
        return 0.0
    if first.shape[0] != n_steps:
        raise ValueError("first_observation length must match the true-cell trace")
    counts = np.bincount(cells, minlength=n_states).astype(float)
    base_rate = counts / counts.sum()
    candidate_index = np.arange(n_states)
    visited_indicator = np.zeros(n_states, dtype=int)
    non_fallback_prediction = np.zeros((n_steps, n_states), dtype=float)
    for step in range(n_steps):
        if first[step]:
            visited_indicator = np.zeros(n_states, dtype=int)
        if not (first[step] or step == 0):
            previous = int(cells[step - 1])
            weights = tensor[candidate_index, previous, visited_indicator]
            weight_mass = float(weights.sum())
            non_fallback_prediction[step] = (
                weights / weight_mass if weight_mass > 0.0 else base_rate
            )
        visited_indicator[int(cells[step])] = 1
    return brier_with_base_rate_fallback(
        cells, n_states, first, non_fallback_prediction
    )


@dataclass(frozen=True)
class ReturnStructureResult:
    n_move_steps: int
    move_step_entropy_reduction: float
    persistence_headroom: float
    return_conditioned_brier: float
    persistence_brier: float
    return_skill_vs_persistence: float
    verdict: str


def run_return_structure_analysis(
    sequence: ObservationSequence, *, smoothing: float, entropy_gate: float = 0.1
) -> ReturnStructureResult:
    """Measure whether return conditioning clears the cheap promotion gate."""

    cells = np.asarray(sequence.spatial_states, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    previous: list[int] = []
    visited_feature: list[int] = []
    target: list[int] = []
    visited: set[int] = set()
    for idx, cell in enumerate(cells):
        if first[idx]:
            visited = {int(cell)}
            continue
        if int(cell) != int(cells[idx - 1]):
            previous.append(int(cells[idx - 1]))
            visited_feature.append(int(int(cell) in visited))
            target.append(int(cell))
        visited.add(int(cell))
    reduction = conditional_entropy_reduction(
        np.asarray(previous, dtype=int),
        np.asarray(visited_feature, dtype=int),
        np.asarray(target, dtype=int),
    )
    tensor = return_conditioned_transition(
        cells, first, sequence.num_spatial_states, smoothing
    )
    pb = persistence_brier(cells, sequence.num_spatial_states, first)
    rb = return_conditioned_brier(cells, first, sequence.num_spatial_states, tensor)
    skill = skill_score(rb, pb) if pb > 1e-12 else 0.0
    verdict = (
        "promote_return_conditioned_prior"
        if reduction >= entropy_gate and pb > 1e-12
        else "honest_null"
    )
    return ReturnStructureResult(
        n_move_steps=len(target),
        move_step_entropy_reduction=reduction,
        persistence_headroom=max(0.0, pb - rb),
        return_conditioned_brier=rb,
        persistence_brier=pb,
        return_skill_vs_persistence=skill,
        verdict=verdict,
    )


RETURN_STRUCTURE_SCHEMA = "active-fractal-rabbit-return-structure-v1"


def run_and_write_return_structure(config_path, observations_path, out_path):
    from .config import load_config
    from .fractalrabbit import parse_waypoint_csv
    from .observations import discretize_observations

    config = load_config(config_path)
    obs = Path(observations_path)
    if not obs.exists():
        assert config.fractalrabbit.fixture_csv is not None
        obs = Path(config.fractalrabbit.fixture_csv)
    sequence = discretize_observations(parse_waypoint_csv(obs), config.inference)
    result = run_return_structure_analysis(
        sequence, smoothing=config.inference.smoothing
    )
    payload = {
        "schema": RETURN_STRUCTURE_SCHEMA,
        "source": str(obs),
        "n_move_steps": result.n_move_steps,
        "move_step_entropy_reduction": result.move_step_entropy_reduction,
        "persistence_headroom": result.persistence_headroom,
        "return_conditioned_brier": result.return_conditioned_brier,
        "persistence_brier": result.persistence_brier,
        "return_skill_vs_persistence": result.return_skill_vs_persistence,
        "verdict": result.verdict,
        "caveat": (
            "Visited-cell return conditioning is a filter-realizable synthetic-generator "
            "probe. Promotion is decided solely by the move-step entropy gate; otherwise "
            "it remains an honest null about long-memory structure under the current data. "
            "return_conditioned_brier / return_skill_vs_persistence score the "
            "return-conditioned predictive prior itself (P(next|prev,visited) renormalized "
            "over candidate cells), not the location-observed POMDP posterior, so the skill "
            "may be zero or negative even when the entropy gate promotes the feature."
        ),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out
