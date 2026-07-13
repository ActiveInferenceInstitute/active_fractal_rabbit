"""Held-out reporting-process / missingness model — out-of-sample gate.

The existing in-sample missingness gate is an honest null: phase- and
traveller-conditioned reporting does not beat a pooled Bernoulli reporter. This
module asks the harder, out-of-sample question for POSITION-IN-BURST
(``runlen`` — the number of consecutive short intervals since the last long
gap). It is a pure report-timing feature, computed only from inter-report
INTERVALS, never from x/y displacement, so it does not smuggle the disclosed
speed cue. The verdict is promoted only when leave-one-traveller-out and
temporal gains clear the predeclared thresholds and the permuted-runlen control
collapses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import load_config
from .observation_sources import load_observation_sequence
from .types import ObservationSequence

HELDOUT_REPORTING_PROCESS_SCHEMA = "active-fractal-rabbit-heldout-reporting-process-v1"
_MAX_RUNLEN = 6

# Predeclared promotion thresholds (see module docstring): a runlen gain must
# clear both an absolute floor and a margin over its permuted-control gain.
MIN_ABSOLUTE_GAIN = 0.02
MIN_MARGIN_OVER_PERMUTED_CONTROL = 0.01


def _build_records(
    sequence: ObservationSequence, long_gap_days: float
) -> dict[str, np.ndarray]:
    """Inter-report interval records with a leak-free position-in-burst feature."""

    first = np.asarray(sequence.is_first_observation, dtype=bool)
    intervals = np.asarray(sequence.intervals_days, dtype=float)
    phases = np.asarray(sequence.time_bins, dtype=int)
    travellers = np.asarray(
        [row.traveller_id for row in sequence.observations], dtype=int
    )

    rec_traveller: list[int] = []
    rec_phase: list[int] = []
    rec_interval: list[float] = []
    rec_runlen: list[int] = []
    run_state: dict[int, int] = {}
    for idx in range(len(intervals)):
        traveller = int(travellers[idx])
        if first[idx]:
            run_state[traveller] = 0
            continue
        if intervals[idx] <= 0:
            continue
        # runlen is the state BEFORE this interval (no future leak)
        rec_traveller.append(traveller)
        rec_phase.append(int(phases[idx]))
        rec_interval.append(float(intervals[idx]))
        rec_runlen.append(min(run_state.get(traveller, 0), _MAX_RUNLEN))
        run_state[traveller] = (
            0 if intervals[idx] > long_gap_days else run_state.get(traveller, 0) + 1
        )
    return {
        "traveller": np.asarray(rec_traveller, dtype=int),
        "phase": np.asarray(rec_phase, dtype=int),
        "interval": np.asarray(rec_interval, dtype=float),
        "runlen": np.asarray(rec_runlen, dtype=int),
    }


def _bernoulli_heldout_loss(
    train: np.ndarray,
    test: np.ndarray,
    context: np.ndarray,
    n_contexts: int,
    labels: np.ndarray,
    smoothing: float,
) -> float:
    counts = np.full((n_contexts, 2), smoothing)
    for j in np.where(train)[0]:
        counts[context[j], labels[j]] += 1.0
    losses = []
    for j in np.where(test)[0]:
        row = counts[context[j]]
        probability = row[labels[j]] / row.sum()
        losses.append(-np.log(max(float(probability), 1e-12)))
    return float(np.mean(losses)) if losses else float("nan")


def _contexts(
    records: dict[str, np.ndarray], seed: int
) -> dict[str, tuple[np.ndarray, int]]:
    traveller = records["traveller"]
    phase = records["phase"]
    runlen = records["runlen"]
    traveller_index = {
        value: i for i, value in enumerate(sorted(set(traveller.tolist())))
    }
    traveller_ctx = np.asarray([traveller_index[int(t)] for t in traveller], dtype=int)
    rng = np.random.default_rng(seed)
    permuted_phase = rng.permutation(phase)
    permuted_runlen = rng.permutation(runlen)
    n_phase = int(phase.max()) + 1 if phase.size else 1
    n_run = int(runlen.max()) + 1 if runlen.size else 1
    return {
        "pooled": (np.zeros(len(traveller), dtype=int), 1),
        "phase": (phase, n_phase),
        "traveller": (traveller_ctx, len(traveller_index)),
        "permuted_phase": (permuted_phase, n_phase),
        "runlen": (runlen, n_run),
        "phase_x_runlen": (phase * n_run + runlen, n_phase * n_run),
        "permuted_runlen": (permuted_runlen, n_run),
    }


def _split_losses(
    records: dict[str, np.ndarray],
    contexts: dict[str, tuple[np.ndarray, int]],
    folds: list[tuple[np.ndarray, np.ndarray]],
    smoothing: float,
) -> dict[str, float]:
    interval = records["interval"]
    out: dict[str, list[float]] = {name: [] for name in contexts}
    for train, test in folds:
        threshold = (
            float(np.median(interval[train]))
            if train.any()
            else float(np.median(interval))
        )
        labels = (interval <= threshold).astype(int)
        for name, (context, n_contexts) in contexts.items():
            out[name].append(
                _bernoulli_heldout_loss(
                    train, test, context, n_contexts, labels, smoothing
                )
            )
    return {name: float(np.nanmean(values)) for name, values in out.items()}


def run_heldout_reporting_process(
    sequence: ObservationSequence,
    *,
    smoothing: float = 0.5,
    long_gap_days: float = 1.0,
    seed: int = 0,
) -> dict[str, object]:
    """Held-out reporting-process gate over leave-one-traveller-out and temporal splits."""

    records = _build_records(sequence, long_gap_days)
    n_intervals = int(records["interval"].size)
    if n_intervals < 4:
        return {
            "schema": HELDOUT_REPORTING_PROCESS_SCHEMA,
            "status": "unavailable",
            "reason": "fewer than four scored inter-report intervals",
            "verdict": "unavailable",
        }
    contexts = _contexts(records, seed)
    traveller = records["traveller"]

    loto_folds = []
    for value in sorted(set(traveller.tolist())):
        test = traveller == value
        if test.any() and (~test).any():
            loto_folds.append((~test, test))
    cut = int(round(0.7 * n_intervals))
    order = np.arange(n_intervals)
    temporal_train = order < cut
    temporal_folds = [(temporal_train, ~temporal_train)]

    loto = _split_losses(records, contexts, loto_folds, smoothing) if loto_folds else {}
    temporal = _split_losses(records, contexts, temporal_folds, smoothing)

    # Per-fold runlen-vs-pooled gain — guards against a single held-out traveller
    # swinging an aggregate win (the small-LOTO-fold-count multiplicity concern).
    runlen_fold_gains: list[float] = []
    for train, test in loto_folds:
        threshold = (
            float(np.median(records["interval"][train]))
            if train.any()
            else float(np.median(records["interval"]))
        )
        labels = (records["interval"] <= threshold).astype(int)
        pooled_ctx, pooled_n = contexts["pooled"]
        runlen_ctx, runlen_n = contexts["runlen"]
        pooled_loss = _bernoulli_heldout_loss(
            train, test, pooled_ctx, pooled_n, labels, smoothing
        )
        runlen_loss = _bernoulli_heldout_loss(
            train, test, runlen_ctx, runlen_n, labels, smoothing
        )
        runlen_fold_gains.append(float(pooled_loss - runlen_loss))
    folds_positive = int(sum(1 for gain in runlen_fold_gains if gain > 0.0))

    def gain(table: dict[str, float], name: str) -> float:
        return float(table.get("pooled", float("nan")) - table.get(name, float("nan")))

    runlen_loto_gain = gain(loto, "runlen") if loto else float("nan")
    runlen_temporal_gain = gain(temporal, "runlen")
    permuted_loto_gain = gain(loto, "permuted_runlen") if loto else float("nan")
    permuted_temporal_gain = gain(temporal, "permuted_runlen")

    promote = (
        bool(loto)
        and runlen_loto_gain
        > max(MIN_ABSOLUTE_GAIN, permuted_loto_gain + MIN_MARGIN_OVER_PERMUTED_CONTROL)
        and runlen_temporal_gain
        > max(
            MIN_ABSOLUTE_GAIN, permuted_temporal_gain + MIN_MARGIN_OVER_PERMUTED_CONTROL
        )
    )
    verdict = "candidate_reporting_process_model" if promote else "honest_null"

    # Disclosure (cross-vendor audit): the short-gap rate per runlen value shows
    # WHERE the signal lives — it concentrates in the terminal post-burst class
    # (a long-gap is predicted right after a full burst), so the model is a
    # burst-cadence / long-gap detector rather than a broad missingness model.
    global_threshold = float(np.median(records["interval"]))
    global_labels = (records["interval"] <= global_threshold).astype(int)
    runlen_values = records["runlen"]
    label_rates = {
        int(value): float(np.mean(global_labels[runlen_values == value]))
        for value in sorted(set(runlen_values.tolist()))
    }
    terminal_class = min(label_rates, key=label_rates.get) if label_rates else None

    return {
        "schema": HELDOUT_REPORTING_PROCESS_SCHEMA,
        "status": "completed",
        "n_intervals": n_intervals,
        "long_gap_days": long_gap_days,
        "loto_losses": loto,
        "temporal_losses": temporal,
        "runlen_loto_gain_vs_pooled": runlen_loto_gain,
        "runlen_temporal_gain_vs_pooled": runlen_temporal_gain,
        "runlen_loto_fold_gains": runlen_fold_gains,
        "runlen_loto_folds_positive": folds_positive,
        "runlen_loto_n_folds": len(loto_folds),
        "phase_loto_gain_vs_pooled": gain(loto, "phase") if loto else float("nan"),
        "traveller_loto_gain_vs_pooled": gain(loto, "traveller")
        if loto
        else float("nan"),
        # identity-proxy defense: runlen adds out-of-sample lift BEYOND traveller identity
        "runlen_gain_beyond_traveller": runlen_loto_gain
        - (gain(loto, "traveller") if loto else 0.0),
        "permuted_runlen_loto_gain_vs_pooled": permuted_loto_gain,
        "runlen_short_gap_rates": label_rates,
        "terminal_burst_class": terminal_class,
        "verdict": verdict,
        "caveat": (
            "Synthetic-only. The runlen feature measures burst cadence in the FractalRabbit reporter — dense "
            "within-burst reports then a long quiet gap. On the expanded fixture it does not clear the "
            "leave-one-traveller-out and temporal promotion thresholds, so the verdict is an honest null even "
            "though the terminal post-burst class remains visible in the descriptive rates. The feature is "
            "derived purely from inter-report intervals, never from x/y displacement, so it does not reuse the "
            "disclosed speed cue. No real-mobility or operational missingness claim."
        ),
    }


def run_and_write_heldout_reporting_process(
    config_path: str | Path,
    out_path: str | Path,
    observations_path: str | Path | None = None,
) -> Path:
    """Run the held-out reporting-process gate and write its artifact."""

    config = load_config(config_path)
    sequence = load_observation_sequence(
        config, observations_path, requested_role="held-out reporting observations"
    )
    payload = run_heldout_reporting_process(sequence)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
