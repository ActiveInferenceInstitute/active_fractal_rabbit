"""Locked-split ablation of fixed versus online-adaptive AIF transitions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .aif_regime_search import build_regime_dataset
from .sota_aif_candidate import fit_aif_candidate
from .sota_program import score_candidate
from .sota_program import paired_loss_difference_ci
from .sota_baselines import OnlineBaseRateCandidate
from .sota_program_harness import build_inference_config

ADAPTIVE_AIF_ABLATION_SCHEMA = "active-fractal-rabbit-adaptive-aif-ablation-v2"
DEFAULT_REALIZATION_SEEDS: tuple[int, ...] = (
    20260616,
    20260625,
    20260704,
    20260713,
    20260722,
)


def run_adaptive_aif_ablation(
    config_path: str | Path = "configs/default.yaml",
    *,
    n_travellers: int = 40,
    realization_seeds: tuple[int, ...] = DEFAULT_REALIZATION_SEEDS,
    spatial_bins: int = 6,
    n_boot: int = 1000,
) -> dict[str, object]:
    """Repeat fixed/adaptive/fair comparisons across switching realizations."""

    if not realization_seeds:
        raise ValueError("at least one realization seed is required")
    if len(set(realization_seeds)) != len(realization_seeds):
        raise ValueError("realization seeds must be unique")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    inference = build_inference_config(config_path, spatial_bins=spatial_bins)
    rows: list[dict[str, object]] = []
    split_seed = 20260616
    for seed in realization_seeds:
        noise_seed = 20261280 + int(seed % 997)
        dataset = build_regime_dataset(
            "latent_regime_switch_035",
            inference,
            n_travellers=n_travellers,
            realization_seed=int(seed),
            split_seed=split_seed,
            noise_seed=noise_seed,
        )
        fixed_fit = fit_aif_candidate(dataset, adaptation_grid=(0.0,))
        adaptive_fit = fit_aif_candidate(dataset)
        fixed = score_candidate(dataset, fixed_fit.candidate)
        adaptive = score_candidate(dataset, adaptive_fit.candidate)
        fair = score_candidate(dataset, OnlineBaseRateCandidate())
        fixed_ci = paired_loss_difference_ci(
            fixed, adaptive, n_boot=n_boot, seed=7
        )
        fair_ci = paired_loss_difference_ci(fair, adaptive, n_boot=n_boot, seed=11)
        rows.append(
            {
                "realization_seed": int(seed),
                "split_seed": split_seed,
                "noise_seed": noise_seed,
                "fixed": {
                    "gamma": fixed_fit.gamma,
                    "boost": fixed_fit.boost,
                    "adaptation": fixed_fit.adaptation,
                    "holdout_loss": float(fixed.pooled_mean_loss),
                },
                "adaptive": {
                    "gamma": adaptive_fit.gamma,
                    "boost": adaptive_fit.boost,
                    "adaptation": adaptive_fit.adaptation,
                    "holdout_loss": float(adaptive.pooled_mean_loss),
                },
                "online_base_rate_loss": float(fair.pooled_mean_loss),
                "fixed_minus_adaptive_nats": float(
                    fixed.pooled_mean_loss - adaptive.pooled_mean_loss
                ),
                "fixed_minus_adaptive_ci": [
                    float(fixed_ci.clustered_ci_low),
                    float(fixed_ci.clustered_ci_high),
                ],
                "adaptive_beats_fixed_ci": bool(fixed_ci.b_wins_clustered),
                "fair_minus_adaptive_nats": float(
                    fair.pooled_mean_loss - adaptive.pooled_mean_loss
                ),
                "fair_minus_adaptive_ci": [
                    float(fair_ci.clustered_ci_low),
                    float(fair_ci.clustered_ci_high),
                ],
                "adaptive_beats_online_base_rate_ci": bool(fair_ci.b_wins_clustered),
            }
        )
    improvements = np.asarray(
        [row["fixed_minus_adaptive_nats"] for row in rows], dtype=float
    )
    fair_gaps = np.asarray([row["fair_minus_adaptive_nats"] for row in rows], dtype=float)
    representative = rows[0]
    fixed = representative["fixed"]
    adaptive = representative["adaptive"]
    return {
        "schema": ADAPTIVE_AIF_ABLATION_SCHEMA,
        "status": "adaptive_repeated_ablation_scored",
        "regime_id": "latent_regime_switch_035",
        "n_travellers": int(n_travellers),
        "realization_seeds": [int(seed) for seed in realization_seeds],
        "realization_count": len(rows),
        "run_parameters": {
            "spatial_bins": int(spatial_bins),
            "n_travellers": int(n_travellers),
            "n_boot": int(n_boot),
            "split_seed": split_seed,
            "noise_seed_rule": "20261280 + realization_seed % 997",
            "comparator": "online_base_rate",
        },
        "fixed": fixed,
        "adaptive": adaptive,
        "adaptive_improvement_nats": float(improvements.mean()),
        "adaptive_improvement_range_nats": [
            float(improvements.min()),
            float(improvements.max()),
        ],
        "adaptive_improves_fixed": bool(np.all(improvements > 0.0)),
        "adaptive_beats_fixed_ci_count": sum(
            bool(row["adaptive_beats_fixed_ci"]) for row in rows
        ),
        "adaptive_beats_online_base_rate_ci_count": sum(
            bool(row["adaptive_beats_online_base_rate_ci"]) for row in rows
        ),
        "mean_fair_minus_adaptive_nats": float(fair_gaps.mean()),
        "predictive_gate_clears": bool(
            rows and all(row["adaptive_beats_online_base_rate_ci"] for row in rows)
        ),
        "rows": rows,
        "selection_contract": "inner-train transition estimation plus inner-validation tuning",
        "boundary": (
            "This repeated synthetic ablation treats realizations and travellers as explicit "
            "robustness units. Improvement over fixed-transition AIF does not imply a win "
            "over the predeclared online-base-rate comparator or global SOTA."
        ),
    }


def run_and_write_adaptive_aif_ablation(
    project_root: str | Path,
    *,
    n_travellers: int = 40,
    realization_seeds: tuple[int, ...] = DEFAULT_REALIZATION_SEEDS,
    spatial_bins: int = 6,
    n_boot: int = 1000,
) -> Path:
    root = Path(project_root).resolve()
    payload = run_adaptive_aif_ablation(
        root / "configs/default.yaml",
        n_travellers=n_travellers,
        realization_seeds=realization_seeds,
        spatial_bins=spatial_bins,
        n_boot=n_boot,
    )
    out = root / "output/runs/adaptive_aif_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
