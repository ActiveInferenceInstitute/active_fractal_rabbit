"""Reporting-process and missingness diagnostics for sporadic waypoints."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import load_config
from .fractalrabbit import parse_waypoint_csv
from .observations import discretize_observations
from .types import ObservationSequence

MISSINGNESS_MODEL_SCHEMA = "active-fractal-rabbit-missingness-model-v1"


def binary_prequential_loss(
    labels: np.ndarray,
    contexts: np.ndarray | None = None,
    *,
    n_contexts: int = 1,
    smoothing: float = 0.5,
) -> float:
    """Online Bernoulli log loss with optional discrete context."""

    y = np.asarray(labels, dtype=int)
    if contexts is None:
        ctx = np.zeros(len(y), dtype=int)
        n_contexts = 1
    else:
        ctx = np.asarray(contexts, dtype=int)
    if len(y) == 0:
        raise ValueError("binary prequential loss needs at least one label")
    counts = np.full((n_contexts, 2), smoothing, dtype=float)
    losses: list[float] = []
    for label, context in zip(y, ctx):
        probs = counts[int(context)] / counts[int(context)].sum()
        losses.append(float(-math.log(max(probs[int(label)], 1e-12))))
        counts[int(context), int(label)] += 1.0
    return float(np.mean(losses))


def phase_burst_probabilities(
    labels: np.ndarray,
    phases: np.ndarray,
    *,
    n_phases: int,
    smoothing: float = 0.5,
) -> list[float]:
    """Return smoothed descriptive P(short gap | phase)."""

    y = np.asarray(labels, dtype=int)
    p = np.asarray(phases, dtype=int)
    counts = np.full((n_phases, 2), smoothing, dtype=float)
    for label, phase in zip(y, p):
        counts[int(phase), int(label)] += 1.0
    probs = counts[:, 1] / counts.sum(axis=1)
    return [float(item) for item in probs]


def run_missingness_model_analysis(
    sequence: ObservationSequence,
    *,
    smoothing: float = 0.5,
    seed: int = 0,
) -> dict[str, object]:
    """Assess whether report timing has phase/traveller structure."""

    first = np.asarray(sequence.is_first_observation, dtype=bool)
    intervals = np.asarray(sequence.intervals_days, dtype=float)
    mask = (~first) & (intervals > 0)
    if int(mask.sum()) == 0:
        raise ValueError("missingness analysis needs positive inter-report intervals")
    scored_intervals = intervals[mask]
    threshold = float(np.median(scored_intervals))
    labels = (scored_intervals <= threshold).astype(int)
    phases = np.asarray(sequence.time_bins, dtype=int)[mask]
    traveller_ids = np.asarray(
        [row.traveller_id for row in sequence.observations], dtype=int
    )[mask]
    unique_travellers = {
        traveller: idx
        for idx, traveller in enumerate(sorted(set(int(t) for t in traveller_ids)))
    }
    travellers = np.asarray(
        [unique_travellers[int(t)] for t in traveller_ids], dtype=int
    )
    n_phases = int(sequence.num_time_bins)
    n_travellers = len(unique_travellers)
    combined = phases * n_travellers + travellers

    pooled_loss = binary_prequential_loss(labels, smoothing=smoothing)
    phase_loss = binary_prequential_loss(
        labels, phases, n_contexts=n_phases, smoothing=smoothing
    )
    traveller_loss = binary_prequential_loss(
        labels, travellers, n_contexts=n_travellers, smoothing=smoothing
    )
    combined_loss = binary_prequential_loss(
        labels,
        combined,
        n_contexts=n_phases * n_travellers,
        smoothing=smoothing,
    )
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(phases)
    permuted_phase_loss = binary_prequential_loss(
        labels, permuted, n_contexts=n_phases, smoothing=smoothing
    )
    phase_gain = pooled_loss - phase_loss
    traveller_gain = pooled_loss - traveller_loss
    combined_gain = pooled_loss - combined_loss
    permuted_gain = pooled_loss - permuted_phase_loss
    verdict = (
        "candidate_reporting_process_model"
        if combined_gain > max(0.01, permuted_gain + 0.005)
        else "honest_null"
    )

    return {
        "schema": MISSINGNESS_MODEL_SCHEMA,
        "verdict": verdict,
        "n_intervals": int(len(labels)),
        "n_phases": n_phases,
        "n_travellers": n_travellers,
        "short_gap_threshold_days": threshold,
        "short_gap_rate": float(np.mean(labels)),
        "mean_interval_days": float(np.mean(scored_intervals)),
        "interval_cv": float(np.std(scored_intervals) / np.mean(scored_intervals))
        if float(np.mean(scored_intervals)) > 0
        else 0.0,
        "pooled_log_loss": pooled_loss,
        "phase_log_loss": phase_loss,
        "traveller_log_loss": traveller_loss,
        "phase_traveller_log_loss": combined_loss,
        "phase_gain_vs_pooled": phase_gain,
        "traveller_gain_vs_pooled": traveller_gain,
        "phase_traveller_gain_vs_pooled": combined_gain,
        "permuted_phase_gain_vs_pooled": permuted_gain,
        "phase_short_gap_probability": phase_burst_probabilities(
            labels, phases, n_phases=n_phases, smoothing=smoothing
        ),
        "caveat": (
            "This is a synthetic reporting-process diagnostic. It separates report timing from movement "
            "only inside the observed waypoint stream and does not estimate real missingness behavior."
        ),
    }


def run_and_write_missingness_model(
    config_path: str | Path, out_path: str | Path
) -> Path:
    """Run the missingness/reporting-process gate and write JSON."""

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
    payload = run_missingness_model_analysis(
        sequence, smoothing=config.inference.smoothing, seed=config.seed
    )
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
