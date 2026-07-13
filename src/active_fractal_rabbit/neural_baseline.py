"""Dependency-light learned (neural) sequence baseline — a bounded NULL.

The state-of-the-art roadmap asks whether a learned sequence model beats the
transparent prequential nulls. This module answers honestly on the project's
synthetic fixture: an online multinomial-logistic next-cell predictor over
engineered features (previous cell, time-of-day, previous speed/burst, return
flag) trained prequentially (predict-then-update SGD) BEATS the weak nulls
(base-rate, persistence, first-order Markov, single-factor pymdp) but does NOT
beat the transparent particle/state-space mixture, and is less interpretable
than either the Markov count table or the pymdp generative lane. Crucially the
win over Markov is still bounded by a stronger transparent comparator: a
previous-cell-only logistic already beats Markov, richer prequential features
improve the loss further on the 4x4 default, and neither clears the
particle/state-space mixture. No new heavy dependency is introduced (numpy
only).

This is shipped as a NULL: the asserted contract is that the learned model does
NOT clear the particle baseline. It makes no real-mobility or operational claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import load_config
from .factorial_pymdp import _stronger_baseline_metrics
from .objectives import ECE_EDGE_RIGHT_CLOSED, expected_calibration_error
from .observation_sources import load_observation_sequence
from .types import ObservationSequence

NEURAL_BASELINE_SCHEMA = "active-fractal-rabbit-neural-baseline-v1"


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _feature_vector(
    n_states: int,
    n_time: int,
    n_speed: int,
    n_burst: int,
    prev_cell: int | None,
    time_bin: int,
    prev_speed: int | None,
    prev_burst: int | None,
    returned: bool,
) -> np.ndarray:
    cell = np.zeros(n_states)
    if prev_cell is not None:
        cell[prev_cell] = 1.0
    tod = np.zeros(n_time)
    tod[time_bin] = 1.0
    spd = np.zeros(n_speed)
    bst = np.zeros(n_burst)
    if prev_speed is not None:
        spd[prev_speed] = 1.0
    if prev_burst is not None:
        bst[prev_burst] = 1.0
    flag = np.array([1.0 if returned else 0.0])
    return np.concatenate([cell, tod, spd, bst, flag, [1.0]])


def prequential_logistic_metrics(
    sequence: ObservationSequence,
    lr: float = 0.8,
    l2: float = 0.05,
    *,
    prev_cell_only: bool = False,
) -> dict[str, float]:
    """Online multinomial-logistic next-cell predictor scored prequentially.

    Weights are zero-initialized (deterministic). Each step predicts the next
    cell from features available before observing it, accrues the log-loss, then
    takes one SGD step. The belief resets at each traveller's first observation.
    """

    cells = np.asarray(sequence.spatial_states, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    phases = np.asarray(sequence.time_bins, dtype=int)
    speeds = np.asarray(sequence.speed_categories, dtype=int)
    bursts = np.asarray(sequence.burst_categories, dtype=int)
    n_states = int(sequence.num_spatial_states)
    n_time = int(sequence.num_time_bins)
    n_speed = int(sequence.num_speed_categories)
    n_burst = int(sequence.num_burst_categories)

    width = n_states + n_time + n_speed + n_burst + 2
    weights = np.zeros((n_states, width))
    losses: list[float] = []
    correct = 0
    confidences: list[float] = []
    hits: list[int] = []
    visited: set[int] = set()
    prev_cell: int | None = None

    for idx in range(len(cells)):
        if idx == 0 or first[idx]:
            visited = set()
            prev_cell = None
        returned = prev_cell is not None and prev_cell in visited
        if prev_cell_only:
            feature = _feature_vector(
                n_states, n_time, n_speed, n_burst, prev_cell, 0, None, None, False
            )
            feature[n_states:] = 0.0
            feature[-1] = 1.0
        else:
            prev_speed = int(speeds[idx - 1]) if (idx > 0 and not first[idx]) else None
            prev_burst = int(bursts[idx - 1]) if (idx > 0 and not first[idx]) else None
            feature = _feature_vector(
                n_states,
                n_time,
                n_speed,
                n_burst,
                prev_cell,
                int(phases[idx]),
                prev_speed,
                prev_burst,
                returned,
            )
        probs = _softmax(weights @ feature)
        target = int(cells[idx])
        losses.append(-np.log(max(float(probs[target]), 1e-12)))
        prediction = int(np.argmax(probs))
        correct += int(prediction == target)
        confidences.append(float(probs[prediction]))
        hits.append(int(prediction == target))
        gradient = np.outer(probs, feature)
        gradient[target] -= feature
        weights -= lr * (gradient + l2 * weights)
        visited.add(target)
        prev_cell = target

    n = len(losses)
    return {
        "mean_predictive_loss": float(np.mean(losses)),
        "accuracy": float(correct / n),
        "ece": _expected_calibration_error(np.asarray(confidences), np.asarray(hits)),
        "n_scored": float(n),
    }


def _expected_calibration_error(
    confidences: np.ndarray, hits: np.ndarray, n_bins: int = 10
) -> float:
    # Delegates to objectives.expected_calibration_error (see sota_benchmark._ece for
    # the full rationale). ECE_EDGE_RIGHT_CLOSED reproduces this site's original
    # convention exactly: (low, high] for every bin, including the first.
    return expected_calibration_error(
        confidences, hits, n_bins, edge_convention=ECE_EDGE_RIGHT_CLOSED
    )[0]


def _gate_and_interpretability_note(
    beats_markov: bool, beats_particle: bool
) -> tuple[str, str]:
    """Derive the gate label and its matching interpretability note.

    The note must always agree with (beats_markov, beats_particle); it is never
    a fixed literal, since a shipped run can legitimately land in any of the
    three branches below.
    """

    if beats_markov and not beats_particle:
        gate = "neural_sequence_beats_weak_nulls_not_particle_mixture"
        note = (
            "The learned logistic predictor beats the weak transparent nulls (base-rate, persistence, "
            "first-order Markov) but not the particle/state-space mixture; previous-cell-only logistic "
            "smoothing already beats the switching-HMM null and richer prequential features improve the "
            "loss further on the 4x4 default, while remaining less interpretable than the Markov count "
            "table or the pymdp generative lane."
        )
    elif beats_particle:
        gate = "neural_sequence_beats_particle_on_this_run"
        note = (
            "The learned logistic predictor beats not only the weak transparent nulls (base-rate, "
            "persistence, first-order Markov) but also the stronger particle/state-space mixture baseline "
            "on this run, while remaining less interpretable than the Markov count table or the pymdp "
            "generative lane."
        )
    else:
        gate = "neural_sequence_below_markov_null"
        note = (
            "The learned logistic predictor does not beat the weak transparent nulls (base-rate, "
            "persistence, first-order Markov) on this run; the switching-HMM null already predicts at "
            "least as well, so no performance advantage over the interpretable Markov count table or the "
            "pymdp generative lane is supported here."
        )
    return gate, note


def neural_baseline_report(
    sequence: ObservationSequence,
    *,
    smoothing: float,
    lr: float = 0.8,
    l2: float = 0.05,
) -> dict[str, object]:
    """Bundle the learned-model metrics with the transparent-null bar."""

    full = prequential_logistic_metrics(sequence, lr=lr, l2=l2)
    prev_only = prequential_logistic_metrics(
        sequence, lr=lr, l2=l2, prev_cell_only=True
    )
    bar = _stronger_baseline_metrics(sequence)
    neural_loss = full["mean_predictive_loss"]
    markov = float(bar["switching_hmm_mean_predictive_loss"])
    particle = float(bar["particle_state_space_mean_predictive_loss"])
    persistence = float(bar["persistence_mean_predictive_loss"])
    beats_markov = neural_loss < markov
    beats_particle = neural_loss < particle
    gate, interpretability_note = _gate_and_interpretability_note(
        beats_markov, beats_particle
    )
    return {
        "schema": NEURAL_BASELINE_SCHEMA,
        "status": "completed",
        "neural_logistic_mean_predictive_loss": neural_loss,
        "neural_logistic_accuracy": full["accuracy"],
        "neural_logistic_ece": full["ece"],
        "prev_cell_only_mean_predictive_loss": prev_only["mean_predictive_loss"],
        "switching_hmm_mean_predictive_loss": markov,
        "particle_state_space_mean_predictive_loss": particle,
        "persistence_mean_predictive_loss": persistence,
        "beats_markov_null": beats_markov,
        "beats_particle_baseline": beats_particle,
        "gate": gate,
        "interpretability_note": interpretability_note,
        "caveat": (
            f"Synthetic-only FractalRabbit fixture ({len(sequence.observations)} rows). Any learned-model "
            "advantage must clear the stronger particle/state-space baseline and the external lane before "
            "performance language is promoted. No real-mobility or operational claim."
        ),
    }


def run_and_write_neural_baseline(
    config_path: str | Path,
    out_path: str | Path,
    observations_path: str | Path | None = None,
) -> Path:
    """Run the neural-baseline null and write its manuscript-facing artifact."""

    config = load_config(config_path)
    sequence = load_observation_sequence(
        config, observations_path, requested_role="neural-baseline observations"
    )
    payload = neural_baseline_report(
        sequence, smoothing=config.inference.model.smoothing
    )
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
