"""Latent-regime HMM baseline — a bounded NULL with an explicit overfit signature.

The roadmap asks whether a latent-regime hidden Markov model beats the
first-order Markov null. Prototyped on the synthetic fixture, it does not: a
batch-EM categorical HMM looks like a win when scored in-sample, but that is
overfitting — held out (fit on a train prefix, scored on the suffix) it collapses
back above the transparent Markov baseline. With only ~90 observations across a
few traveller segments and nine cells, EM cannot honestly identify regime-
conditioned structure online.

Shipped as a NULL whose asserted contract is the OVERFIT GAP itself: the
in-sample loss is far below the honest held-out loss, and the held-out HMM does
not beat the Markov bar. Synthetic-only; no real-mobility claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import load_config
from .factorial_pymdp import _stronger_baseline_metrics
from .observation_sources import load_observation_sequence
from .types import ObservationSequence

LATENT_REGIME_HMM_SCHEMA = "active-fractal-rabbit-latent-regime-hmm-v1"


def _segments(cells: np.ndarray, first: np.ndarray) -> list[np.ndarray]:
    segments: list[list[int]] = []
    for idx, cell in enumerate(cells):
        if idx == 0 or first[idx] or not segments:
            segments.append([int(cell)])
        else:
            segments[-1].append(int(cell))
    return [np.asarray(seg, dtype=int) for seg in segments]


def _fit_categorical_hmm(
    segments: list[np.ndarray],
    n_states: int,
    k: int,
    *,
    n_iter: int,
    seed: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baum-Welch for a categorical emission HMM (regimes emit cell distributions)."""

    rng = np.random.default_rng(seed)
    pi = np.full(k, 1.0 / k)
    trans = rng.dirichlet(np.ones(k) * 2.0, size=k)
    emit = rng.dirichlet(np.ones(n_states) * 2.0, size=k)
    for _ in range(n_iter):
        trans_acc = np.full((k, k), smoothing)
        emit_acc = np.full((k, n_states), smoothing)
        pi_acc = np.full(k, smoothing)
        for seg in segments:
            t = len(seg)
            if t == 0:
                continue
            alpha = np.zeros((t, k))
            scale = np.zeros(t)
            alpha[0] = pi * emit[:, seg[0]]
            scale[0] = alpha[0].sum() or 1.0
            alpha[0] /= scale[0]
            for i in range(1, t):
                alpha[i] = (alpha[i - 1] @ trans) * emit[:, seg[i]]
                scale[i] = alpha[i].sum() or 1.0
                alpha[i] /= scale[i]
            beta = np.zeros((t, k))
            beta[-1] = 1.0
            for i in range(t - 2, -1, -1):
                beta[i] = (trans @ (emit[:, seg[i + 1]] * beta[i + 1])) / scale[i + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True).clip(1e-12)
            pi_acc += gamma[0]
            for i in range(t):
                emit_acc[:, seg[i]] += gamma[i]
            for i in range(t - 1):
                xi = (
                    alpha[i][:, None]
                    * trans
                    * (emit[:, seg[i + 1]] * beta[i + 1])[None, :]
                ) / scale[i + 1]
                xi /= xi.sum().clip(1e-12)
                trans_acc += xi
        pi = pi_acc / pi_acc.sum()
        trans = trans_acc / trans_acc.sum(axis=1, keepdims=True)
        emit = emit_acc / emit_acc.sum(axis=1, keepdims=True)
    return pi, trans, emit


def _score_segments(
    segments: list[np.ndarray],
    pi: np.ndarray,
    trans: np.ndarray,
    emit: np.ndarray,
    base: np.ndarray,
) -> float:
    """Prequential next-cell loss under frozen HMM params (no refit)."""

    losses: list[float] = []
    for seg in segments:
        belief = pi.copy()
        for i, cell in enumerate(seg):
            if i == 0:
                predictive = base
            else:
                predictive = (belief @ trans) @ emit
                predictive = np.maximum(predictive, 1e-12)
                predictive /= predictive.sum()
                losses.append(-np.log(max(float(predictive[int(cell)]), 1e-12)))
            # belief update on the observed cell
            belief = (
                (belief @ trans) * emit[:, int(cell)]
                if i > 0
                else pi * emit[:, int(cell)]
            )
            belief = np.maximum(belief, 1e-12)
            belief /= belief.sum()
    return float(np.mean(losses)) if losses else float("nan")


def run_latent_regime_hmm_analysis(
    sequence: ObservationSequence,
    *,
    ks: tuple[int, ...] = (2, 3),
    smoothing: float = 0.5,
    n_em_iter: int = 40,
    seed: int = 0,
) -> dict[str, object]:
    """Fit batch-EM HMMs and contrast in-sample vs honest held-out loss."""

    cells = np.asarray(sequence.spatial_states, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    n_states = int(sequence.num_spatial_states)
    base = np.bincount(cells, minlength=n_states).astype(float) + 0.5
    base /= base.sum()
    segments = _segments(cells, first)
    bar = _stronger_baseline_metrics(sequence)
    markov = float(bar["switching_hmm_mean_predictive_loss"])

    # temporal held-out split over segments (fit on the first ~70%, score the rest)
    cut = max(1, int(round(0.7 * len(segments))))
    train_segments = segments[:cut]

    rows = []
    best_honest = float("inf")
    for k in ks:
        pi, trans, emit = _fit_categorical_hmm(
            segments, n_states, k, n_iter=n_em_iter, seed=seed, smoothing=smoothing
        )
        in_sample = _score_segments(segments, pi, trans, emit, base)
        pi_t, trans_t, emit_t = _fit_categorical_hmm(
            train_segments,
            n_states,
            k,
            n_iter=n_em_iter,
            seed=seed,
            smoothing=smoothing,
        )
        holdout_segments = segments[cut:]
        if not holdout_segments:
            raise ValueError(
                "latent_regime_hmm holdout split is degenerate (empty holdout slice): "
                f"len(segments)={len(segments)}, cut={cut}. Refusing to silently fall back to "
                "scoring against the training segments, which would contaminate honest_heldout_loss."
            )
        held_out = _score_segments(holdout_segments, pi_t, trans_t, emit_t, base)
        best_honest = min(best_honest, held_out)
        rows.append(
            {
                "model": "categorical_emission_hmm",
                "K": k,
                "batch_em_insample_loss": in_sample,
                "honest_heldout_loss": held_out,
                "overfit_gap": held_out - in_sample,
            }
        )

    beats_markov = best_honest < markov
    overfit_gap = max(row["overfit_gap"] for row in rows)
    return {
        "schema": LATENT_REGIME_HMM_SCHEMA,
        "status": "completed",
        "rows": rows,
        "best_honest_heldout_loss": best_honest,
        "markov_null_loss": markov,
        "particle_loss": float(bar["particle_state_space_mean_predictive_loss"]),
        "beats_markov_null": bool(beats_markov),
        "max_overfit_gap": float(overfit_gap),
        "gate": (
            "latent_regime_hmm_beats_markov_on_this_run"
            if beats_markov
            else "latent_regime_hmm_does_not_beat_markov_null_em_overfits"
        ),
        "caveat": (
            "Synthetic-only. A batch-EM categorical HMM looks competitive in-sample but overfits: held out it "
            "is scored against the transparent first-order Markov baseline. The current verdict is emitted from "
            "that held-out comparison rather than from fixture size. A more strongly regime-structured generator "
            "could plausibly let a latent HMM win; on this run the overfit gap remains the asserted signature when "
            "the gate is negative."
        ),
    }


def run_and_write_latent_regime_hmm(
    config_path: str | Path,
    out_path: str | Path,
    observations_path: str | Path | None = None,
) -> Path:
    """Run the latent-regime-HMM null and write its artifact."""

    config = load_config(config_path)
    sequence = load_observation_sequence(
        config, observations_path, requested_role="latent-regime HMM observations"
    )
    payload = run_latent_regime_hmm_analysis(sequence)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
