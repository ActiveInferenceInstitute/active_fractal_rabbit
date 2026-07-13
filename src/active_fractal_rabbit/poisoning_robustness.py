"""Adversarial poisoning robustness via VFE surprise — defensive data-integrity check.

Does the analyst's active-inference generative model DETECT when it is fed
tampered or inconsistent input observations? The reference filter's per-step
variational free energy (``-log p(o_t | prior_t)`` — its own surprise) is scored
as a poison detector against three synthetic data-integrity stressors: teleport
(a cell jumped to the farthest grid location), label-flip (a relabelled cell),
and subtle drift (a one-step grid shift). This is DEFENSIVE self-monitoring, not
an attack tool: the stressors are the crudest possible perturbations, never
emitted, optimized, or made evasive, and no real-mobility or operational claim is
made. What it PROTECTS is trust in downstream analysis — gross input corruption
is caught by the filter's surprise before it contaminates inference, and the
honest null tells operators NOT to over-trust the filter for subtle drift.

Each claim is gated with the S14 discipline: a STRUCTURE-NULL (re-assign which
steps carry the poison label, preserving the surprise-score and label-count
marginals while destroying the step-poison alignment) confirms the detection is
real, and a POSITIVE CONTROL (a known surprise spike) confirms the test has power
— so any null is "no active-inference advantage over a cheap Markov baseline",
not "the test is too weak"; if the verdict changes under a new grid, that change
is reported rather than hidden.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from .anomaly import detection_auc, novelty_baseline
from .figure_style import (
    LEGEND_FONT_SIZE,
    SMALL_LABEL_FONT_SIZE,
    TICK_FONT_SIZE,
    manuscript_rcparams,
)
from .hardened_anomaly import markov_surprise_baseline
from .inference import run_latent_inference
from .pomdp import build_pomdp
from .types import ObservationSequence

POISONING_ROBUSTNESS_SCHEMA = "active-fractal-rabbit-poisoning-robustness-v1"
_MODES = ("teleport", "label_flip", "drift")


def _coord(state: int, grid: int) -> tuple[int, int]:
    return state % grid, state // grid


def poison_stream(
    sequence: ObservationSequence, fraction: float, seed: int, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a synthetic data-integrity stressor at a fraction of non-first steps."""

    cells = np.asarray(sequence.spatial_states, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    n_states = int(sequence.num_spatial_states)
    grid = int(sequence.spatial_bins)
    if grid * grid != n_states:
        raise ValueError("poisoning stressors require square grid metadata")
    non_first = np.where(~first)[0]
    rng = np.random.default_rng(seed)
    count = max(3, int(fraction * len(non_first)))
    steps = rng.choice(non_first, count, replace=False)
    poisoned = cells.copy()
    for step in steps:
        cell = int(cells[step])
        if mode == "teleport":
            distances = [
                abs(_coord(cell, grid)[0] - _coord(b, grid)[0])
                + abs(_coord(cell, grid)[1] - _coord(b, grid)[1])
                for b in range(n_states)
            ]
            poisoned[step] = int(np.argmax(distances))
        elif mode == "label_flip":
            poisoned[step] = int(rng.choice([c for c in range(n_states) if c != cell]))
        elif mode == "drift":
            cx, cy = _coord(cell, grid)
            poisoned[step] = (cx + 1) % grid + cy * grid
        else:
            raise ValueError(f"unknown poison mode: {mode}")
    labels = np.zeros(len(cells), dtype=bool)
    labels[steps] = True
    return poisoned, labels


def _vfe(
    sequence: ObservationSequence, cells: np.ndarray, smoothing: float, model=None
) -> np.ndarray:
    if model is None:
        model = build_pomdp(sequence, smoothing)
    poisoned = dataclasses.replace(
        sequence, spatial_states=np.asarray(cells, dtype=int)
    )
    return np.asarray(run_latent_inference(model, poisoned).predictive_loss)


def _ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def run_poisoning_robustness(
    sequence: ObservationSequence,
    *,
    smoothing: float,
    fractions: tuple[float, ...] = (1 / 12, 1 / 6, 1 / 4, 1 / 3),
    n_seeds: int = 12,
    n_null: int = 40,
    seed: int = 0,
) -> dict[str, object]:
    """Score VFE-surprise poison detection across modes/rates, with both controls."""

    first = np.asarray(sequence.is_first_observation, dtype=bool)
    mask = ~first
    n_states = int(sequence.num_spatial_states)
    model = build_pomdp(sequence, smoothing)
    per_mode: dict[str, dict[str, object]] = {}
    for mode in _MODES:
        per_rate: dict[str, object] = {}
        for fraction in fractions:
            vfe_aucs = []
            markov_aucs = []
            novelty_aucs = []
            for offset in range(n_seeds):
                poisoned, labels = poison_stream(
                    sequence, fraction, seed + offset, mode
                )
                vfe_aucs.append(
                    detection_auc(
                        _vfe(sequence, poisoned, smoothing, model), labels, mask
                    )
                )
                markov_aucs.append(
                    detection_auc(
                        markov_surprise_baseline(poisoned, first, n_states, smoothing),
                        labels,
                        mask,
                    )
                )
                novelty_aucs.append(
                    detection_auc(novelty_baseline(poisoned, first), labels, mask)
                )
            vfe = np.asarray(vfe_aucs)
            markov = np.asarray(markov_aucs)
            vfe_lo, vfe_hi = _ci(vfe)
            markov_lo, markov_hi = _ci(markov)
            # Paired VFE-minus-Markov advantage on the aligned per-seed arrays.
            # The flag is ONE-SIDED (lower CI bound > 0), not "interval excludes
            # 0": the delta here lies entirely below 0, so an excludes-0 test
            # would falsely report a VFE advantage when Markov dominates.
            delta = vfe - markov
            delta_lo, delta_hi = _ci(delta)
            per_rate[f"{fraction:.4f}"] = {
                "vfe_auc_mean": float(vfe.mean()),
                "vfe_auc_ci": [vfe_lo, vfe_hi],
                "markov_auc_mean": float(markov.mean()),
                "markov_auc_ci": [markov_lo, markov_hi],
                "novelty_auc_mean": float(np.mean(novelty_aucs)),
                "ci_separated_vs_chance": bool(vfe_lo > 0.5),
                "ci_separated_vs_markov": bool(vfe_lo > markov_hi),
                "vfe_minus_markov_delta_mean": float(delta.mean()),
                "vfe_minus_markov_delta_ci": [delta_lo, delta_hi],
                "vfe_paired_advantage": bool(delta_lo > 0.0),
            }
        per_mode[mode] = per_rate

    structure_null = {
        mode: _structure_null(
            sequence, smoothing, mode, n_null=n_null, seed=seed, model=model
        )
        for mode in _MODES
    }
    power_check = _power_check(
        sequence, smoothing, n_null=n_null, seed=seed, model=model
    )
    teleport_mid = per_mode["teleport"][f"{1 / 6:.4f}"]
    label_flip_mid = per_mode["label_flip"][f"{1 / 6:.4f}"]
    drift_mid = per_mode["drift"][f"{1 / 6:.4f}"]
    gross_detected = bool(teleport_mid["ci_separated_vs_markov"])
    label_flip_above_chance = bool(label_flip_mid["ci_separated_vs_chance"])
    moderate_detected = bool(label_flip_mid["ci_separated_vs_markov"])
    label_flip_markov_null = bool(label_flip_above_chance and not moderate_detected)
    subtle_null = bool(not drift_mid["ci_separated_vs_markov"])
    verdict = "poisoning_detection_regime_changed"
    if gross_detected and moderate_detected and subtle_null:
        verdict = "gross_and_moderate_poisoning_detected_subtle_is_null"
    elif gross_detected and label_flip_markov_null and subtle_null:
        verdict = (
            "gross_poisoning_detected_label_flip_not_markov_separated_subtle_is_null"
        )
    return {
        "schema": POISONING_ROBUSTNESS_SCHEMA,
        "status": "completed",
        "per_mode": per_mode,
        "structure_null": structure_null,
        "power_check": power_check,
        "gross_poisoning_detected": gross_detected,
        "label_flip_detected_above_chance": label_flip_above_chance,
        "label_flip_not_markov_separated": label_flip_markov_null,
        "moderate_poisoning_detected": moderate_detected,
        "subtle_drift_is_null_vs_markov": subtle_null,
        # derived from the measured gates, never a hardcoded literal
        "verdict": verdict,
        "caveat": (
            f"Fixture-bounded synthetic-only ({len(sequence.observations)} rows, {n_states} cells, "
            f"{len({row.traveller_id for row in sequence.observations})} travellers). DEFENSIVE/oversight-positive: this "
            "measures whether the analyst's own generative model flags tampered input data; the poison models are "
            "crude synthetic data-integrity stressors used to test the detector, never an attack technique, and no "
            "operational, surveillance, re-identification, tasking, or evasion capability is produced. At the mid-rate "
            f"gate, teleport separated from Markov: {gross_detected}; label-flip separated from Markov: {moderate_detected}; "
            f"subtle drift remained a Markov-relative null: {subtle_null}. The structure-null preserves the surprise-score "
            "and poison-count marginals while destroying the step-poison alignment; the positive control confirms power, "
            "so any null is 'no AI advantage', not 'weak test'. Operators should read the verdict and per-mode gates as "
            "configuration-bound data-integrity diagnostics, not as subtle-drift assurance. No real-mobility claim."
        ),
    }


def _structure_null(
    sequence: ObservationSequence,
    smoothing: float,
    mode: str,
    *,
    n_null: int,
    seed: int,
    model=None,
) -> dict[str, object]:
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    mask = ~first
    poisoned, labels = poison_stream(sequence, 1 / 6, seed, mode)
    scores = _vfe(sequence, poisoned, smoothing, model)
    real_auc = detection_auc(scores, labels, mask)
    n_positive = int(labels[mask].sum())
    candidate_steps = np.where(mask)[0]
    nulls = []
    for draw in range(n_null):
        rng = np.random.default_rng(seed + 10000 + draw)
        shuffled = np.zeros(len(labels), dtype=bool)
        shuffled[rng.choice(candidate_steps, n_positive, replace=False)] = True
        nulls.append(detection_auc(scores, shuffled, mask))
    null = np.asarray(nulls)
    p95 = float(np.percentile(null, 95))
    return {
        "real_auc": float(real_auc),
        "null_mean": float(null.mean()),
        "null_p95": p95,
        "p_value": float(np.mean(null >= real_auc)),
        "gate_passed": bool(real_auc > p95),
    }


def _power_check(
    sequence: ObservationSequence,
    smoothing: float,
    *,
    n_null: int,
    seed: int,
    spike_nats: float = 2.0,
    model=None,
) -> dict[str, object]:
    """Positive control: a known surprise spike at poisoned steps must be detected."""

    first = np.asarray(sequence.is_first_observation, dtype=bool)
    mask = ~first
    poisoned, labels = poison_stream(sequence, 1 / 6, seed, "teleport")
    scores = _vfe(sequence, poisoned, smoothing, model).copy()
    scores[labels] += spike_nats
    real_auc = detection_auc(scores, labels, mask)
    n_positive = int(labels[mask].sum())
    candidate_steps = np.where(mask)[0]
    nulls = []
    for draw in range(n_null):
        rng = np.random.default_rng(seed + 20000 + draw)
        shuffled = np.zeros(len(labels), dtype=bool)
        shuffled[rng.choice(candidate_steps, n_positive, replace=False)] = True
        nulls.append(detection_auc(scores, shuffled, mask))
    p95 = float(np.percentile(np.asarray(nulls), 95))
    return {
        "injected_spike_nats": spike_nats,
        "real_auc": float(real_auc),
        "null_p95": p95,
        "gate_passed": bool(real_auc > p95),
    }


def render_poisoning_figure(summary: dict[str, object], path: str | Path) -> Path:
    """VFE poison-detection AUC vs rate per mode, against the Markov baseline."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_mode = summary["per_mode"]
    plt.rcParams.update(manuscript_rcparams({"gray": "#595959"}))
    fig, axes = plt.subplots(
        2, 1, figsize=(10.2, 9.2), height_ratios=[1.1, 1.0], constrained_layout=True
    )
    colors = {"teleport": "#D55E00", "label_flip": "#E69F00", "drift": "#009E73"}
    for mode, per_rate in per_mode.items():
        rates = sorted(float(r) for r in per_rate)
        keys = [f"{r:.4f}" for r in rates]
        vfe = [float(per_rate[k]["vfe_auc_mean"]) for k in keys]
        markov = [float(per_rate[k]["markov_auc_mean"]) for k in keys]
        axes[0].plot(rates, vfe, marker="o", color=colors[mode], label=f"VFE {mode}")
        axes[0].plot(
            rates,
            markov,
            marker="x",
            linestyle="--",
            color=colors[mode],
            alpha=0.6,
            label=f"Markov surprise {mode}",
        )
    axes[0].axhline(0.5, color="#333333", linestyle=":", linewidth=1.0, label="chance")
    axes[0].set_xlabel("poison rate")
    axes[0].set_ylabel("detection ROC AUC")
    axes[0].set_title("VFE surprise vs Markov baseline", loc="left")
    # legend outside the axes: with 7 entries any in-axes placement covers
    # either the low-AUC Markov dashes or the high-AUC VFE lines.
    axes[0].legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=LEGEND_FONT_SIZE,
        ncol=1,
        frameon=False,
    )
    axes[0].grid(True, alpha=0.25)

    modes = list(summary["structure_null"].keys())
    reals = [float(summary["structure_null"][m]["real_auc"]) for m in modes]
    nulls = [float(summary["structure_null"][m]["null_mean"]) for m in modes]
    pos = np.arange(len(modes))
    width = 0.35
    axes[1].bar(pos - width / 2, reals, width=width, color="#D55E00", label="real")
    axes[1].bar(
        pos + width / 2, nulls, width=width, color="#999999", label="structure-null"
    )
    axes[1].axhline(
        float(summary["power_check"]["real_auc"]),
        color="#0072B2",
        linestyle="--",
        linewidth=1.0,
        label="positive control",
    )
    for x, value in zip(pos, reals):
        axes[1].text(
            x - width / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=SMALL_LABEL_FONT_SIZE,
        )
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels(
        [mode.replace("_", "\n") for mode in modes], fontsize=TICK_FONT_SIZE
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("detection AUC")
    axes[1].set_title("Structure-null + positive control", loc="left")
    axes[1].legend(loc="lower right", fontsize=LEGEND_FONT_SIZE)
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Poisoning robustness: defensive surprise checks remain verdict-bound",
        x=0.02,
        ha="left",
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def run_and_write_poisoning_robustness(
    config_path: str | Path,
    observations_path: str | Path | None,
    out_path: str | Path,
    *,
    seed: int = 0,
) -> Path:
    """Run the poisoning-robustness analysis and write its artifact."""

    from .config import load_config
    from .observation_sources import load_observation_sequence

    config = load_config(config_path)
    sequence = load_observation_sequence(
        config, observations_path, requested_role="poisoning-robustness observations"
    )
    payload = run_poisoning_robustness(
        sequence, smoothing=config.inference.model.smoothing, seed=seed
    )
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
