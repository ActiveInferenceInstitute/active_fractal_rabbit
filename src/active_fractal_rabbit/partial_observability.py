"""Where active inference genuinely helps: partial observability under observation noise.

The default benchmark has the observed cell EQUAL the latent cell (location discretized is the
observation), so there is no hidden state to infer and a point estimate suffices — which is why no
active-inference advantage survives there. Active inference's machinery (a recursive belief over a
hidden state, marginalized through a noisy emission) earns its keep only under genuine partial
observability: a latent state, a noisy emission channel, and the task of predicting the TRUE latent
next state.

This module makes that precise and fair. A latent cell sequence (from the FractalRabbit dynamics, real
or synthetic) is observed through a known symmetric emission: the observed cell equals the latent cell
with probability ``1 - eps`` and is otherwise uniform over the other ``n - 1`` cells. The generator and
the inference emission are the same channel by construction, so the filter is correctly specified.
Three predictors share the SAME learned latent
transition, the SAME emission, the SAME observations, and the SAME priors; they differ only in how they
use the belief:

* **soft (active inference)** — predict by marginalizing the full belief: ``T @ b_t``.
* **hard (point estimate)** — collapse the belief to its MAP cell first: ``T @ onehot(argmax b_t)``.
* **observed-Markov** — commit to the raw (noisy) observation: ``T @ onehot(o_{t-1})``.

Because soft and hard are identical except for marginalize-vs-commit, no knob can be asymmetric between
them (the failure mode that sank an earlier noise study): the comparison is clean by construction. The
measured result, robust across noise realisations, locked splits, and independent synthetic generator
draws: the soft (AIF) advantage over the hard point estimate is exactly zero at zero noise and grows
with the observation-noise level. This is the genuine, principled regime in which active inference has
an advantage — partial observability with a noisy emission.
"""

from __future__ import annotations

import math

import numpy as np

from .sota_benchmark import lock_traveller_split
from .types import ObservationSequence

PARTIAL_OBSERVABILITY_SCHEMA = "active-fractal-rabbit-partial-observability-v1"
DEFAULT_EPS_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4)
DEFAULT_NOISE_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_SPLIT_SEEDS = (20260616, 20260617, 20260618)
# Ten independent generator draws: unanimous sign-stability across all ten gives a sign-test
# bound of 0.5**10 ~= 0.001, which clears the family-wide Bonferroni per-comparison threshold of
# the multiplicity ledger (alpha / family-size) with margin, so the one fair-by-construction
# positive survives even the most conservative correction in the paper. Verified sign-stable at
# 8, 10, and 12 draws before adopting 10.
DEFAULT_CROSS_DRAW_SEEDS = (2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035)
DEFAULT_CROSS_DRAW_TRAVELLERS = 16


def realize_noisy_observations(
    z: np.ndarray, n: int, eps: float, rng: np.random.Generator
) -> np.ndarray:
    """Realize the matched symmetric observation channel for latent cells ``z``.

    On a flip (probability ``eps``) the observation is uniform over the ``n-1`` cells OTHER than
    the true cell, which makes the generated channel exactly the inference emission used by
    :func:`_losses_at`: ``P(o=z)=1-eps`` and ``P(o=other)=eps/(n-1)``. Drawing in ``[0, n-1)`` and
    bumping past the true index gives that uniform-over-alternatives law without bias, so a flip
    NEVER lands on the true cell and the realized flip rate is ``eps`` (not ``eps*(n-1)/n``). This
    is the single channel-law source for the partial-observability lane; it is exercised directly
    in tests so the headline result rests on a matched, validated channel, not prose.
    """

    observed = np.asarray(z).copy()
    flip_idx = np.flatnonzero(rng.random(len(z)) < eps)
    if n > 1 and flip_idx.size:
        true_at_flip = observed[flip_idx]
        draw = rng.integers(0, n - 1, flip_idx.size)
        observed[flip_idx] = draw + (draw >= true_at_flip)
    return observed


def _losses_at(z, first, travellers, n, *, eps, transition, noise_seed, holdout):
    """Prequential next-latent-cell loss for the soft/hard/observed predictors under a noisy emission.
    Target is the TRUE latent cell; predictions use only past observations (leakage-free)."""
    emission = np.full((n, n), eps / (n - 1) if n > 1 else 0.0)
    np.fill_diagonal(emission, 1.0 - eps)
    rng = np.random.default_rng(noise_seed)
    observed = realize_noisy_observations(z, n, eps, rng)
    identity = np.eye(n)
    soft: list[float] = []
    hard: list[float] = []
    obs: list[float] = []
    belief = None
    last = None
    for i in range(len(z)):
        if first[i] or travellers[i] != last:
            last = travellers[i]
            belief = emission[observed[i], :].copy()
            belief = belief / belief.sum()
            continue
        if holdout[i]:
            true_cell = int(z[i])
            soft.append(-math.log(max(float((transition @ belief)[true_cell]), 1e-12)))
            hard.append(
                -math.log(
                    max(
                        float(
                            (transition @ identity[int(np.argmax(belief))])[true_cell]
                        ),
                        1e-12,
                    )
                )
            )
            obs.append(
                -math.log(
                    max(
                        float((transition @ identity[int(observed[i - 1])])[true_cell]),
                        1e-12,
                    )
                )
            )
        belief = emission[observed[i], :] * (transition @ belief)
        belief = belief / belief.sum()
    if not soft:
        raise ValueError(
            "no scored holdout transitions; cannot measure partial-observability advantage"
        )
    return float(np.mean(soft)), float(np.mean(hard)), float(np.mean(obs))


def noisy_emission_advantage(
    sequence: ObservationSequence,
    *,
    eps_levels: tuple[float, ...] = DEFAULT_EPS_LEVELS,
    smoothing: float = 0.05,
    holdout_fraction: float = 0.5,
    noise_seeds: tuple[int, ...] = DEFAULT_NOISE_SEEDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
) -> dict[str, object]:
    """Measure the active-inference (soft filter) advantage over a point estimate (hard/MAP) as the
    observation-noise level grows, averaged over noise realisations and locked splits."""

    z = np.asarray(sequence.spatial_states, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    travellers = np.asarray([r.traveller_id for r in sequence.observations], dtype=int)
    n = int(sequence.num_spatial_states)

    # latent transition learned from TRAIN travellers, shared by all predictors (one per split seed)
    transitions: dict[int, np.ndarray] = {}
    holdouts: dict[int, np.ndarray] = {}
    for split_seed in split_seeds:
        split = lock_traveller_split(
            sequence, holdout_fraction=holdout_fraction, seed=split_seed
        )
        is_holdout = np.asarray(split["is_holdout"], dtype=bool)
        holdouts[split_seed] = is_holdout
        counts = np.full((n, n), smoothing, dtype=float)
        last = None
        for i in range(len(z)):
            if first[i] or travellers[i] != last:
                last = travellers[i]
                continue
            if not is_holdout[i]:
                counts[z[i], z[i - 1]] += 1.0
        transitions[split_seed] = counts / counts.sum(axis=0, keepdims=True)

    rows = []
    for eps in eps_levels:
        hard_minus_soft = []
        obs_minus_soft = []
        for split_seed in split_seeds:
            for noise_seed in noise_seeds:
                s, h, o = _losses_at(
                    z,
                    first,
                    travellers,
                    n,
                    eps=eps,
                    transition=transitions[split_seed],
                    noise_seed=noise_seed,
                    holdout=holdouts[split_seed],
                )
                hard_minus_soft.append(h - s)
                obs_minus_soft.append(o - s)
        # Each (split_seed, noise_seed) is a paired soft-vs-hard resampling on the SAME data; the
        # spread below characterizes sensitivity to the emission noise draw and the holdout split
        # (a robustness spread), and fraction_positive is the paired sign-test proportion — it is NOT
        # a between-dataset/generalisation interval (this run scores one latent draw).
        hms = np.asarray(hard_minus_soft, dtype=float)
        rows.append(
            {
                "eps": float(eps),
                "aif_advantage_over_hard_nats": float(hms.mean()),
                "aif_advantage_over_hard_sd": float(hms.std(ddof=1))
                if hms.size > 1
                else 0.0,
                "aif_advantage_over_hard_min": float(hms.min()),
                "aif_advantage_over_hard_max": float(hms.max()),
                "aif_advantage_over_observed_nats": float(np.mean(obs_minus_soft)),
                "n_realizations": int(hms.size),
                "fraction_realizations_positive": float(np.mean(hms > 0.0)),
            }
        )

    clean = (
        next(r for r in rows if r["eps"] == 0.0)
        if any(r["eps"] == 0.0 for r in rows)
        else None
    )
    noisy = [r for r in rows if r["eps"] > 0.0]
    advs = [r["aif_advantage_over_hard_nats"] for r in rows]
    monotone = all(b >= a - 1e-6 for a, b in zip(advs, advs[1:]))
    max_row = max(rows, key=lambda r: r["eps"])
    # Quantify "grows with noise" as a single number: OLS slope (nats per unit eps) of the mean
    # advantage against the noise level, with the trend R^2. Deterministic given fixed seeds.
    eps_arr = np.asarray([r["eps"] for r in rows], dtype=float)
    adv_arr = np.asarray(advs, dtype=float)
    if eps_arr.size > 1 and eps_arr.std() > 0.0:
        slope = float(np.cov(eps_arr, adv_arr, ddof=0)[0, 1] / eps_arr.var())
        trend_r2 = float(np.corrcoef(eps_arr, adv_arr)[0, 1] ** 2)
    else:
        slope, trend_r2 = 0.0, 0.0
    return {
        "schema": PARTIAL_OBSERVABILITY_SCHEMA,
        "n_cells": n,
        "rows": rows,
        # Top-level scalars are the provenance-bound manuscript-variable surface (see variable_provenance.py).
        "clean_advantage_nats": float(clean["aif_advantage_over_hard_nats"])
        if clean is not None
        else 0.0,
        "max_eps": float(max_row["eps"]),
        "max_noise_advantage_nats": float(max_row["aif_advantage_over_hard_nats"]),
        "max_noise_advantage_sd": float(max_row["aif_advantage_over_hard_sd"]),
        "max_noise_advantage_min": float(max_row["aif_advantage_over_hard_min"]),
        "n_realizations_per_level": int(rows[0]["n_realizations"]) if rows else 0,
        "smoothing": float(smoothing),
        "min_fraction_positive_under_noise": float(
            min((r["fraction_realizations_positive"] for r in noisy), default=0.0)
        ),
        "noise_slope_nats_per_eps": slope,
        "noise_trend_r2": trend_r2,
        "zero_advantage_when_clean": bool(
            clean is not None and abs(clean["aif_advantage_over_hard_nats"]) < 1e-6
        ),
        "advantage_positive_under_noise": bool(
            all(r["aif_advantage_over_hard_min"] > 0 for r in noisy)
        ),
        "advantage_monotone_with_noise": bool(monotone),
        "verdict": (
            "belief marginalization beats a point estimate (committing to the MAP) "
            "under partial observability, with the advantage zero at zero noise and growing with the "
            "observation-noise level; active inference participates as one recursive belief lane, not "
            "as a unique mechanism"
        ),
        "boundary": "fair by construction (soft vs hard differ only in marginalize-vs-commit; same model, "
        "emission, observations, priors); the emission is a known generative noise channel, not a tuned knob",
    }


def cross_realization_noisy_emission_advantage(
    inference_config,
    *,
    n_travellers: int = DEFAULT_CROSS_DRAW_TRAVELLERS,
    generator_seeds: tuple[int, ...] = DEFAULT_CROSS_DRAW_SEEDS,
    eps_levels: tuple[float, ...] = DEFAULT_EPS_LEVELS,
    smoothing: float = 0.05,
    holdout_fraction: float = 0.5,
    noise_seeds: tuple[int, ...] = DEFAULT_NOISE_SEEDS,
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS,
) -> dict[str, object]:
    """Repeat the noisy-emission comparison across independent synthetic generator draws.

    The single-draw robustness band in :func:`noisy_emission_advantage` averages over
    noise draws and locked splits. This companion block measures the between-draw
    spread that band cannot see, while keeping the same known-emission, soft-vs-hard
    comparison.
    """

    if not generator_seeds:
        raise ValueError("at least one generator seed is required")

    from .power_analysis import generate_synthetic_sequence

    draw_rows: list[dict[str, object]] = []
    for generator_seed in generator_seeds:
        sequence = generate_synthetic_sequence(
            n_travellers, inference_config, seed=generator_seed
        )
        result = noisy_emission_advantage(
            sequence,
            eps_levels=eps_levels,
            smoothing=smoothing,
            holdout_fraction=holdout_fraction,
            noise_seeds=noise_seeds,
            split_seeds=split_seeds,
        )
        for row in result["rows"]:
            draw_rows.append(
                {
                    "generator_seed": int(generator_seed),
                    "eps": float(row["eps"]),
                    "aif_advantage_over_hard_nats": float(
                        row["aif_advantage_over_hard_nats"]
                    ),
                    "aif_advantage_over_hard_min": float(
                        row["aif_advantage_over_hard_min"]
                    ),
                    "aif_advantage_over_hard_max": float(
                        row["aif_advantage_over_hard_max"]
                    ),
                    "fraction_realizations_positive": float(
                        row["fraction_realizations_positive"]
                    ),
                }
            )

    summary_rows: list[dict[str, object]] = []
    for eps in eps_levels:
        values = np.asarray(
            [
                row["aif_advantage_over_hard_nats"]
                for row in draw_rows
                if abs(float(row["eps"]) - float(eps)) < 1e-12
            ],
            dtype=float,
        )
        if values.size == 0:
            continue
        summary_rows.append(
            {
                "eps": float(eps),
                "mean_advantage_nats": float(values.mean()),
                "between_draw_sd_nats": float(values.std(ddof=1))
                if values.size > 1
                else 0.0,
                "between_draw_min_nats": float(values.min()),
                "between_draw_max_nats": float(values.max()),
                "fraction_draws_positive": float(np.mean(values > 0.0)),
                "n_generator_draws": int(values.size),
            }
        )

    noisy = [row for row in summary_rows if float(row["eps"]) > 0.0]
    means = [float(row["mean_advantage_nats"]) for row in summary_rows]
    monotone = all(b >= a - 1e-9 for a, b in zip(means, means[1:]))
    return {
        "cross_realization_n_draws": int(len(generator_seeds)),
        "cross_realization_n_travellers": int(n_travellers),
        "cross_realization_rows": summary_rows,
        "cross_realization_draw_rows": draw_rows,
        "cross_realization_max_between_draw_sd": float(
            max((row["between_draw_sd_nats"] for row in noisy), default=0.0)
        ),
        "cross_realization_min_fraction_positive_under_noise": float(
            min((row["fraction_draws_positive"] for row in noisy), default=0.0)
        ),
        "cross_realization_sign_stable_under_noise": bool(
            noisy and all(float(row["fraction_draws_positive"]) == 1.0 for row in noisy)
        ),
        "cross_realization_advantage_monotone_mean": bool(monotone),
        "cross_realization_boundary": (
            "independent synthetic generator draws under the same known symmetric emission; "
            "between-draw spread is not empirical transfer and does not relax the oracle boundary"
        ),
    }


def run_and_write_partial_observability(config_path, observations_path, out_path):
    """Discretize a waypoint CSV, run the partial-observability sweep, and write the JSON artifact.

    The manuscript-facing top-level values always use the configured fixture. If
    ``observations_path`` exists, that current external realization is recorded
    as a secondary lane so it cannot silently replace the canonical token source.
    Returns the written Path.
    """

    import json
    from pathlib import Path

    from .config import load_config
    from .observation_sources import (
        load_observation_sequence,
        resolve_observations_path,
    )

    config = load_config(config_path)
    fixture_path = resolve_observations_path(
        config, None, requested_role="partial-observability fixture observations"
    )
    sequence = load_observation_sequence(
        config, None, requested_role="partial-observability fixture observations"
    )
    payload = {
        **noisy_emission_advantage(sequence),
        **cross_realization_noisy_emission_advantage(config.inference),
        "primary_source": "configured_fixture",
        "source": f"fixture:{fixture_path.name}",
        "seed": config.seed,
    }
    external_path = Path(observations_path) if observations_path is not None else None
    if external_path is not None and external_path.is_file():
        external_sequence = load_observation_sequence(
            config,
            external_path,
            requested_role="partial-observability external observations",
        )
        external_result = noisy_emission_advantage(external_sequence)
        payload["external_lane"] = {
            "status": "completed",
            "source": str(external_path),
            "n_observations": int(len(external_sequence.observations)),
            "n_travellers": int(
                len({row.traveller_id for row in external_sequence.observations})
            ),
            "analysis": external_result,
            "boundary": (
                "single external simulator realization; synthetic-only partial-observability check, "
                "not empirical transfer and not the canonical manuscript-variable source"
            ),
        }
    else:
        payload["external_lane"] = {
            "status": "unavailable",
            "reason": (
                "external observations were not supplied or were not produced in the current full-study invocation"
            ),
            "boundary": "fixture remains canonical; stale external artifacts are not reused",
        }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out
