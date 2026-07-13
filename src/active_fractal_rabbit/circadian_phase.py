"""Circadian-phase occupancy signal for sporadic waypoint states."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import ObservationSequence

CIRCADIAN_PHASE_SCHEMA = "active-fractal-rabbit-circadian-phase-v1"


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """Return Jensen-Shannon divergence in nats for two probability vectors."""

    p = _normalize(np.asarray(left, dtype=float))
    q = _normalize(np.asarray(right, dtype=float))
    midpoint = 0.5 * (p + q)
    return 0.5 * _kl(p, midpoint) + 0.5 * _kl(q, midpoint)


def phase_cell_distributions(
    cells: np.ndarray,
    phases: np.ndarray,
    *,
    n_states: int,
    n_phases: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pooled and phase-conditioned cell distributions."""

    pooled = np.full(n_states, smoothing, dtype=float)
    phase_counts = np.full((n_phases, n_states), smoothing, dtype=float)
    for cell, phase in zip(cells.astype(int), phases.astype(int), strict=True):
        pooled[cell] += 1.0
        phase_counts[phase, cell] += 1.0
    pooled = _normalize(pooled)
    phase_probs = phase_counts / phase_counts.sum(axis=1, keepdims=True)
    return pooled, phase_probs


def phase_predictive_gain(
    cells: np.ndarray,
    phases: np.ndarray,
    *,
    n_states: int,
    n_phases: int,
    smoothing: float,
) -> dict[str, float]:
    """Compare phase-conditioned cell prediction to the pooled occupancy null."""

    pooled, phase_probs = phase_cell_distributions(
        cells, phases, n_states=n_states, n_phases=n_phases, smoothing=smoothing
    )
    cell_idx = cells.astype(int)
    phase_idx = phases.astype(int)
    pooled_loss = float(np.mean(-np.log(np.maximum(pooled[cell_idx], 1e-12))))
    phase_loss = float(
        np.mean(-np.log(np.maximum(phase_probs[phase_idx, cell_idx], 1e-12)))
    )
    phase_js = np.array(
        [
            jensen_shannon_divergence(phase_probs[idx], pooled)
            for idx in range(n_phases)
        ],
        dtype=float,
    )
    weights = np.array(
        [(phase_idx == idx).sum() for idx in range(n_phases)], dtype=float
    )
    weights = weights / weights.sum() if weights.sum() else np.ones(n_phases) / n_phases
    return {
        "pooled_predictive_loss": pooled_loss,
        "phase_predictive_loss": phase_loss,
        "phase_gain_vs_pooled": pooled_loss - phase_loss,
        "weighted_phase_js": float(np.sum(weights * phase_js)),
        "max_phase_js": float(np.max(phase_js)) if phase_js.size else 0.0,
    }


@dataclass(frozen=True)
class CircadianPhaseResult:
    n_steps: int
    n_phases: int
    n_states: int
    pooled_predictive_loss: float
    phase_predictive_loss: float
    phase_gain_vs_pooled: float
    permutation_gain_vs_pooled: float
    gain_over_permutation: float
    weighted_phase_js: float
    max_phase_js: float
    n_permutations: int
    permutation_null_mean: float
    phase_gain_p_value: float
    max_phase_js_p_value: float
    phase_cell_probability: list[list[float]]
    verdict: str


def run_circadian_phase_analysis(
    sequence: ObservationSequence,
    *,
    smoothing: float,
    seed: int,
    gain_gate: float = 0.01,
    js_gate: float = 0.01,
    n_permutations: int = 200,
) -> CircadianPhaseResult:
    """Estimate whether phase-conditioned occupancy clears a simple signal gate."""

    cells = np.asarray(sequence.spatial_states, dtype=int)
    phases = np.asarray(sequence.time_bins, dtype=int)
    metrics = phase_predictive_gain(
        cells,
        phases,
        n_states=sequence.num_spatial_states,
        n_phases=sequence.num_time_bins,
        smoothing=smoothing,
    )
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(phases)
    null = phase_predictive_gain(
        cells,
        permuted,
        n_states=sequence.num_spatial_states,
        n_phases=sequence.num_time_bins,
        smoothing=smoothing,
    )
    # Permutation null distribution (N draws) for a proper floored p-value on
    # both the predictive gain and the max phase JS, replacing the single-draw
    # comparison as the inferential null. p = (1 + #{null >= observed}) / (N + 1)
    # cannot resolve below 1/(N+1) (mirrors the ledger resolution-floor convention).
    null_gains = np.empty(n_permutations, dtype=float)
    null_max_js = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        draw = phase_predictive_gain(
            cells,
            rng.permutation(phases),
            n_states=sequence.num_spatial_states,
            n_phases=sequence.num_time_bins,
            smoothing=smoothing,
        )
        null_gains[i] = draw["phase_gain_vs_pooled"]
        null_max_js[i] = draw["max_phase_js"]
    obs_gain = metrics["phase_gain_vs_pooled"]
    obs_max_js = metrics["max_phase_js"]
    phase_gain_p = float(
        (1 + int(np.sum(null_gains >= obs_gain))) / (n_permutations + 1)
    )
    max_phase_js_p = float(
        (1 + int(np.sum(null_max_js >= obs_max_js))) / (n_permutations + 1)
    )
    permutation_null_mean = float(null_gains.mean())
    _, phase_probs = phase_cell_distributions(
        cells,
        phases,
        n_states=sequence.num_spatial_states,
        n_phases=sequence.num_time_bins,
        smoothing=smoothing,
    )
    gain_over_null = metrics["phase_gain_vs_pooled"] - null["phase_gain_vs_pooled"]
    # The verdict gates on the N-draw permutation null (its mean), not the single
    # noisy draw above -- that single draw is retained only for the descriptive
    # gain_over_permutation/permutation_gain_vs_pooled reporting fields.
    gain_over_null_mean = obs_gain - permutation_null_mean
    verdict = (
        "candidate_phase_conditioned_prior"
        if gain_over_null_mean > gain_gate and metrics["max_phase_js"] > js_gate
        else "honest_null"
    )
    return CircadianPhaseResult(
        n_steps=int(cells.size),
        n_phases=sequence.num_time_bins,
        n_states=sequence.num_spatial_states,
        pooled_predictive_loss=metrics["pooled_predictive_loss"],
        phase_predictive_loss=metrics["phase_predictive_loss"],
        phase_gain_vs_pooled=metrics["phase_gain_vs_pooled"],
        permutation_gain_vs_pooled=null["phase_gain_vs_pooled"],
        gain_over_permutation=gain_over_null,
        weighted_phase_js=metrics["weighted_phase_js"],
        max_phase_js=metrics["max_phase_js"],
        n_permutations=n_permutations,
        permutation_null_mean=permutation_null_mean,
        phase_gain_p_value=phase_gain_p,
        max_phase_js_p_value=max_phase_js_p,
        phase_cell_probability=phase_probs.tolist(),
        verdict=verdict,
    )


def circadian_phase_payload(
    result: CircadianPhaseResult,
    *,
    source: str,
    smoothing: float,
    seed: int,
) -> dict[str, object]:
    return {
        "schema": CIRCADIAN_PHASE_SCHEMA,
        "source": source,
        "smoothing": smoothing,
        "seed": seed,
        "n_steps": result.n_steps,
        "n_phases": result.n_phases,
        "n_states": result.n_states,
        "pooled_predictive_loss": result.pooled_predictive_loss,
        "phase_predictive_loss": result.phase_predictive_loss,
        "phase_gain_vs_pooled": result.phase_gain_vs_pooled,
        "permutation_gain_vs_pooled": result.permutation_gain_vs_pooled,
        "gain_over_permutation": result.gain_over_permutation,
        "weighted_phase_js": result.weighted_phase_js,
        "max_phase_js": result.max_phase_js,
        "n_permutations": result.n_permutations,
        "permutation_null_mean": result.permutation_null_mean,
        "phase_gain_p_value": result.phase_gain_p_value,
        "max_phase_js_p_value": result.max_phase_js_p_value,
        "phase_cell_probability": result.phase_cell_probability,
        "verdict": result.verdict,
        "caveat": (
            "The phase signal is a synthetic-data occupancy diagnostic over configured time-of-day bins. "
            "It tests whether a phase-conditioned prior is worth implementing; it is not evidence of "
            "real circadian behavior in any population."
        ),
    }


def run_and_write_circadian_phase(
    config_path,
    observations_path,
    out_path,
) -> Path:
    """Discretize a waypoint CSV, run phase analysis, and write the JSON artifact."""

    from .config import load_config
    from .fractalrabbit import parse_waypoint_csv
    from .observations import discretize_observations

    config = load_config(config_path)
    obs_path = Path(observations_path)
    if obs_path.exists():
        source = str(observations_path)
    else:
        assert config.fractalrabbit.fixture_csv is not None
        obs_path = Path(config.fractalrabbit.fixture_csv)
        source = f"fixture:{obs_path.name}"
    sequence = discretize_observations(parse_waypoint_csv(obs_path), config.inference)
    result = run_circadian_phase_analysis(
        sequence, smoothing=config.inference.smoothing, seed=config.seed
    )
    payload = circadian_phase_payload(
        result, source=source, smoothing=config.inference.smoothing, seed=config.seed
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0:
        return np.ones_like(values, dtype=float) / values.size
    return values / total


def _kl(left: np.ndarray, right: np.ndarray) -> float:
    mask = left > 0
    return float(
        np.sum(left[mask] * np.log(left[mask] / np.maximum(right[mask], 1e-12)))
    )
