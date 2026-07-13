"""Hardened routine-break anomaly probes."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from .anomaly import detection_auc, inject_routine_breaks
from .inference import run_latent_inference
from .pomdp import build_pomdp


def markov_surprise_baseline(
    cells: np.ndarray, is_first_observation: np.ndarray, n_states: int, smoothing: float
) -> np.ndarray:
    """Online one-step Markov surprise baseline, excluding traveller boundaries."""

    cells = np.asarray(cells, dtype=int)
    first = np.asarray(is_first_observation, dtype=bool)
    counts = np.full((n_states, n_states), smoothing, dtype=float)
    scores = np.zeros(cells.shape[0], dtype=float)
    for idx in range(1, cells.shape[0]):
        if first[idx]:
            continue
        probs = counts[:, cells[idx - 1]] / counts[:, cells[idx - 1]].sum()
        scores[idx] = -np.log(max(float(probs[cells[idx]]), 1e-12))
        counts[cells[idx], cells[idx - 1]] += 1.0
    return scores


def operating_point(
    score: np.ndarray, labels: np.ndarray, false_positive_rate: float = 0.1
) -> dict[str, float]:
    """Threshold score at a requested empirical false-positive-rate quantile."""

    score = np.asarray(score, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    negatives = np.sort(score[~labels])
    if negatives.size == 0:
        threshold = float("inf")
    else:
        index = min(
            negatives.size - 1,
            max(0, int(np.floor((1.0 - false_positive_rate) * negatives.size))),
        )
        threshold = float(negatives[index])
    predicted = score >= threshold
    fp = float(np.mean(predicted[~labels])) if (~labels).any() else 0.0
    tp = float(np.mean(predicted[labels])) if labels.any() else 0.0
    return {"threshold": threshold, "true_positive_rate": tp, "false_positive_rate": fp}


def _score_sequence(
    sequence, cells: np.ndarray, smoothing: float
) -> tuple[np.ndarray, np.ndarray]:
    model = build_pomdp(sequence, smoothing)
    anomalous_sequence = dataclasses.replace(
        sequence, spatial_states=np.asarray(cells, dtype=int)
    )
    vfe = np.asarray(run_latent_inference(model, anomalous_sequence).predictive_loss)
    markov = markov_surprise_baseline(
        cells, sequence.is_first_observation, sequence.num_spatial_states, smoothing
    )
    return vfe, markov


def _score_changed_sequence(
    sequence, changed_sequence, smoothing: float
) -> tuple[np.ndarray, np.ndarray]:
    model = build_pomdp(sequence, smoothing)
    vfe = np.asarray(run_latent_inference(model, changed_sequence).predictive_loss)
    markov = markov_surprise_baseline(
        np.asarray(changed_sequence.spatial_states, dtype=int),
        changed_sequence.is_first_observation,
        changed_sequence.num_spatial_states,
        smoothing,
    )
    return vfe, markov


def _selected_non_first_labels(sequence, fraction: float, seed: int) -> np.ndarray:
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    non_first = np.where(~first)[0]
    rng = np.random.default_rng(seed)
    count = max(3, int(fraction * len(non_first)))
    steps = rng.choice(non_first, count, replace=False)
    labels = np.zeros(len(first), dtype=bool)
    labels[steps] = True
    return labels


def _timing_shift_sequence(sequence, fraction: float, seed: int):
    labels = _selected_non_first_labels(sequence, fraction, seed)
    time_bins = np.asarray(sequence.time_bins, dtype=int).copy()
    speed_categories = np.asarray(sequence.speed_categories, dtype=int).copy()
    burst_categories = np.asarray(sequence.burst_categories, dtype=int).copy()
    shift = max(1, int(sequence.num_time_bins) // 2)
    time_bins[labels] = (time_bins[labels] + shift) % int(sequence.num_time_bins)
    speed_categories[labels] = int(max(sequence.num_speed_categories - 1, 0))
    burst_categories[labels] = 1 - burst_categories[labels]
    return (
        dataclasses.replace(
            sequence,
            time_bins=time_bins,
            speed_categories=speed_categories,
            burst_categories=burst_categories,
        ),
        labels,
    )


def _adversarial_missingness_proxy_sequence(sequence, fraction: float, seed: int):
    labels = _selected_non_first_labels(sequence, fraction, seed)
    cells = np.asarray(sequence.spatial_states, dtype=int).copy()
    speed_categories = np.asarray(sequence.speed_categories, dtype=int).copy()
    burst_categories = np.asarray(sequence.burst_categories, dtype=int).copy()
    for step in np.flatnonzero(labels):
        cells[step] = cells[step - 1]
    speed_categories[labels] = 0
    burst_categories[labels] = 0
    return (
        dataclasses.replace(
            sequence,
            spatial_states=cells,
            speed_categories=speed_categories,
            burst_categories=burst_categories,
        ),
        labels,
    )


def _diagnostic_row(
    vfe: np.ndarray, markov: np.ndarray, labels: np.ndarray, mask: np.ndarray
) -> dict[str, object]:
    return {
        "vfe_auc": detection_auc(vfe, labels, mask),
        "markov_auc": detection_auc(markov, labels, mask),
        "vfe_operating_point": operating_point(
            vfe[mask], labels[mask], false_positive_rate=0.1
        ),
        "markov_operating_point": operating_point(
            markov[mask], labels[mask], false_positive_rate=0.1
        ),
    }


def run_hardened_anomaly_analysis(
    sequence, *, smoothing: float, seed: int
) -> dict[str, object]:
    """Score subtle anomaly classes against a fair Markov-surprise baseline."""

    first = np.asarray(sequence.is_first_observation, dtype=bool)
    mask = ~first
    payload: dict[str, object] = {}
    for label, mode in (
        ("speed_regime_shift", "random"),
        ("preferred_set_drift", "rare"),
    ):
        anomalous_cells, labels = inject_routine_breaks(
            sequence, 1 / 12, seed, mode=mode
        )
        vfe, markov = _score_sequence(sequence, anomalous_cells, smoothing)
        payload[label] = _diagnostic_row(vfe, markov, labels, mask)
    timing_sequence, timing_labels = _timing_shift_sequence(
        sequence, 1 / 12, seed + 101
    )
    vfe, markov = _score_changed_sequence(sequence, timing_sequence, smoothing)
    payload["timing_manipulation"] = _diagnostic_row(vfe, markov, timing_labels, mask)

    missing_sequence, missing_labels = _adversarial_missingness_proxy_sequence(
        sequence, 1 / 12, seed + 202
    )
    vfe, markov = _score_changed_sequence(sequence, missing_sequence, smoothing)
    payload["adversarial_missingness_proxy"] = _diagnostic_row(
        vfe, markov, missing_labels, mask
    )
    return payload


HARDENED_ANOMALY_SCHEMA = "active-fractal-rabbit-hardened-anomaly-v1"


def run_and_write_hardened_anomaly(config_path, observations_path, out_path):
    from .config import load_config
    from .fractalrabbit import parse_waypoint_csv
    from .observations import discretize_observations

    config = load_config(config_path)
    obs = Path(observations_path)
    if not obs.exists():
        assert config.fractalrabbit.fixture_csv is not None
        obs = Path(config.fractalrabbit.fixture_csv)
    sequence = discretize_observations(parse_waypoint_csv(obs), config.inference)
    payload = {
        "schema": HARDENED_ANOMALY_SCHEMA,
        "source": str(obs),
        **run_hardened_anomaly_analysis(
            sequence, smoothing=config.inference.smoothing, seed=config.seed
        ),
        "caveat": (
            "VFE surprise is compared with an online Markov-surprise baseline, subtle anomaly "
            "classes, aggregate timing-manipulation and adversarial-missingness proxy diagnostics, "
            "and an operating-point threshold. Synthetic defensive software evidence only; no "
            "operational/evasion procedure or real-mobility claim is emitted."
        ),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out
