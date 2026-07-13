"""Recovery-versus-sparsity analysis for the sporadic-observation situation.

FractalRabbit's premise is that trajectories are seen only through sparse,
irregular reports. The operative analytic question is therefore not whether
latent state can be inferred from dense data, but how inference quality
degrades as reports thin out. This lane measures that directly: each
traveller's reports are deterministically subsampled to a fraction of the
full set, the full and deep-latent (no location modality) models are
re-estimated at every level, and both are scored against the exact
ground-truth cell. The pymdp agent runs on the deep-latent model at every
level so its variational free energy is observable as evidence thins.

All numbers are fixture-lane software measurements over synthetic data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import load_config
from .fractalrabbit import parse_waypoint_csv
from .ground_truth import state_recovery
from .inference import run_latent_inference
from .objectives import (
    chance_floors,
    persistence_brier,
    recovery_calibration,
    skill_score,
)
from .observations import discretize_observations
from .pomdp import build_pomdp
from .pymdp_agent import run_pymdp_inference

SPARSITY_SCHEMA = "active-fractal-rabbit-sparsity-v1"
DEFAULT_FRACTIONS = (1.0, 0.75, 0.5, 0.25)
DEEP_LATENT_MODALITIES = ("time_of_day", "speed", "report_burst")
SPARSITY_PYMDP_VALIDATION_MAX_STEPS = 32
# Independent subsample seeds for the additive robustness band. The canonical
# recovery row above is one deterministic subsample; these expose how sensitive
# the deep-latent mass is to WHICH reports survive at the same fraction.
SPARSITY_SUBSAMPLE_SEEDS = (0, 1, 2, 3, 4)


def _deep_latent_mass_at(sub_rows, config) -> float:
    """Deep-latent mean mass on the true cell for one subsample (band member)."""

    sequence = discretize_observations(sub_rows, config.inference)
    model = build_pomdp(
        sequence,
        config.inference.smoothing,
        config.inference.model.controls,
        modalities=DEEP_LATENT_MODALITIES,
    )
    inference = run_latent_inference(model, sequence)
    metrics, _, _ = state_recovery(sequence, inference.posterior_states, "latent")
    return float(metrics["latent_mean_mass_on_true_cell"])


def _persistence_skill(
    brier: float, true_cells: np.ndarray, n_states: int, first_observation: np.ndarray
) -> float:
    """Recovery Brier skill against the temporal-persistence null (nan if degenerate)."""

    null = persistence_brier(true_cells, n_states, first_observation)
    return skill_score(brier, null) if null > 1e-12 else float("nan")


def run_sparsity_analysis(
    config_path: str | Path,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
) -> Path:
    """Run the subsampling sweep and write ``output/runs/sparsity_analysis.json``."""

    config = load_config(config_path)
    if config.fractalrabbit.execution != "fixture":
        raise ValueError(
            "sparsity analysis subsamples a checked fixture; use a fixture config"
        )
    assert config.fractalrabbit.fixture_csv is not None
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise ValueError("sparsity fractions must lie in (0, 1]")
    rows = parse_waypoint_csv(config.fractalrabbit.fixture_csv)
    rng = np.random.default_rng(config.seed)

    levels: list[dict[str, float]] = []
    for fraction_idx, fraction in enumerate(sorted(set(fractions), reverse=True)):
        sub_rows = _subsample(rows, fraction, rng)
        sequence = discretize_observations(sub_rows, config.inference)
        smoothing = config.inference.smoothing
        controls = config.inference.model.controls
        full_model = build_pomdp(sequence, smoothing, controls)
        latent_model = build_pomdp(
            sequence, smoothing, controls, modalities=DEEP_LATENT_MODALITIES
        )
        full = run_latent_inference(full_model, sequence)
        latent = run_latent_inference(latent_model, sequence)
        full_metrics, _, _ = state_recovery(sequence, full.posterior_states, "full")
        latent_metrics, _, _ = state_recovery(
            sequence, latent.posterior_states, "latent"
        )
        pymdp_validation_indices = _sentinel_indices(
            len(sequence.observations), SPARSITY_PYMDP_VALIDATION_MAX_STEPS
        )
        pymdp = run_pymdp_inference(
            latent_model,
            sequence,
            reference_posterior=latent.posterior_states,
            validation_indices=pymdp_validation_indices,
        )
        pymdp_metrics, _, _ = state_recovery(
            sequence, latent.posterior_states, "pymdp_latent"
        )
        # Additive subsample-resampling band on the deep-latent recovery: resample
        # the SAME fraction under independent seeds so the reported single-subsample
        # value can be read against its spread. This does NOT touch the canonical
        # row values below. At fraction 1.0 every seed keeps the full report set,
        # so the band collapses to a point (zero spread) by construction.
        band_masses = [
            _deep_latent_mass_at(
                _subsample(
                    rows,
                    fraction,
                    np.random.default_rng((int(config.seed), fraction_idx, k)),
                ),
                config,
            )
            for k in SPARSITY_SUBSAMPLE_SEEDS
        ]
        band_arr = np.asarray(band_masses, dtype=float)
        n_travellers = len({row.traveller_id for row in sub_rows})
        true_cells = np.asarray(sequence.spatial_states, dtype=int)
        full_calibration, _ = recovery_calibration(full.posterior_states, true_cells)
        latent_calibration, _ = recovery_calibration(
            latent.posterior_states, true_cells
        )
        floors = chance_floors(latent_model)
        levels.append(
            {
                "fraction": float(fraction),
                "actual_fraction": float(len(sub_rows) / len(rows)),
                "n_reports": float(len(sub_rows)),
                "mean_reports_per_traveller": float(len(sub_rows) / n_travellers),
                "full_mean_mass_on_true_cell": full_metrics[
                    "full_mean_mass_on_true_cell"
                ],
                "full_top1_accuracy": full_metrics["full_top1_accuracy"],
                "latent_mean_mass_on_true_cell": latent_metrics[
                    "latent_mean_mass_on_true_cell"
                ],
                "latent_top1_accuracy": latent_metrics["latent_top1_accuracy"],
                "latent_mean_posterior_entropy": latent.metrics[
                    "mean_posterior_entropy"
                ],
                "full_brier": full_calibration["brier"],
                "latent_brier": latent_calibration["brier"],
                "full_ece": full_calibration["ece"],
                "latent_ece": latent_calibration["ece"],
                "full_brier_skill_score": skill_score(
                    full_calibration["brier"], floors["chance_floor_recovery_brier"]
                ),
                "latent_brier_skill_score": skill_score(
                    latent_calibration["brier"], floors["chance_floor_recovery_brier"]
                ),
                # Honest nulls across the ladder: how recovery skill against the
                # tighter base-rate and persistence nulls degrades as reports thin.
                "latent_brier_skill_vs_base_rate": (
                    skill_score(
                        latent_calibration["brier"],
                        latent_calibration["base_rate_brier"],
                    )
                    if latent_calibration["base_rate_brier"] > 1e-12
                    else float("nan")
                ),
                "latent_brier_skill_vs_persistence": _persistence_skill(
                    latent_calibration["brier"],
                    true_cells,
                    latent_model.D.size,
                    sequence.is_first_observation,
                ),
                "pymdp_latent_mean_mass_on_true_cell": pymdp_metrics[
                    "pymdp_latent_mean_mass_on_true_cell"
                ],
                "pymdp_latent_mean_vfe": pymdp.metrics["pymdp_mean_vfe"],
                "pymdp_exact_inference_max_abs_delta": pymdp.metrics[
                    "pymdp_exact_inference_max_abs_delta"
                ],
                "pymdp_validation_rows": pymdp.metrics["pymdp_inference_rows"],
                "pymdp_source_observation_rows": pymdp.metrics[
                    "pymdp_source_observation_rows"
                ],
                "pymdp_validation_fraction": pymdp.metrics["pymdp_validation_fraction"],
                # Additive subsample-resampling robustness band (not an inferential CI).
                "latent_mean_mass_subsample_mean": float(band_arr.mean()),
                "latent_mean_mass_subsample_sd": (
                    float(band_arr.std(ddof=1)) if band_arr.size > 1 else 0.0
                ),
                "latent_mean_mass_subsample_min": float(band_arr.min()),
                "latent_mean_mass_subsample_max": float(band_arr.max()),
                "n_subsample_seeds": len(SPARSITY_SUBSAMPLE_SEEDS),
            }
        )
    payload = {
        "schema": SPARSITY_SCHEMA,
        "lane": "fixture",
        "seed": config.seed,
        "source_config": str(Path(config_path)),
        "fixture": str(config.fractalrabbit.fixture_csv),
        "deep_latent_modalities": list(DEEP_LATENT_MODALITIES),
        "n_levels": len(levels),
        "levels": levels,
        "caveat": (
            "Recovery-versus-sparsity curves over deterministic subsamples of the checked fixture. "
            "They quantify how this configured model degrades as reports thin; they are software "
            "measurements on synthetic data, not empirical mobility findings. The full-model curve "
            "observes the true cell directly through the location modality, so its near-ceiling "
            "recovery is a sanity bound by construction; only the deep-latent curves test genuine "
            "inference. The per-traveller subsample keeps at least two reports, so actual_fraction "
            "is the honest sparsity level when it diverges from the requested fraction. The pymdp lane "
            "validates the released state updater on deterministic sentinel rows at each sparsity level; "
            "full-row deep-latent recovery uses the matched reference posterior after that agreement check. "
            "The deep-latent recovery band is the min-max spread of the mean mass on the true cell across "
            f"{len(SPARSITY_SUBSAMPLE_SEEDS)} independent subsamples at the same fraction — a subsample-resampling "
            "robustness spread over this one fixture, not an inferential or population confidence interval; at the "
            "densest (full-report) level every subsample keeps all reports, so the band collapses to a point by construction."
        ),
    }
    output = config.project_root / "output" / "runs" / "sparsity_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def _subsample(rows, fraction: float, rng: np.random.Generator):
    """Keep a chronologically ordered random subset of each traveller's reports."""

    by_traveller: dict[int, list] = {}
    for row in rows:
        by_traveller.setdefault(row.traveller_id, []).append(row)
    kept = []
    for traveller_rows in by_traveller.values():
        count = max(2, round(len(traveller_rows) * fraction))
        count = min(count, len(traveller_rows))
        indices = sorted(rng.choice(len(traveller_rows), size=count, replace=False))
        kept.extend(traveller_rows[idx] for idx in indices)
    return tuple(sorted(kept, key=lambda item: (item.traveller_id, item.days)))


def _sentinel_indices(n_steps: int, max_steps: int) -> tuple[int, ...]:
    if n_steps <= 0 or max_steps <= 0:
        return ()
    if n_steps <= max_steps:
        return tuple(range(n_steps))
    return tuple(
        int(item)
        for item in np.unique(np.linspace(0, n_steps - 1, max_steps, dtype=int))
    )
