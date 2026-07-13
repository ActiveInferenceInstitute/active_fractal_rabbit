"""Regime-qualified active-inference SOTA search.

This module deliberately does NOT redefine the global SOTA program.  It searches a
separate synthetic surface for the narrower condition where active-inference belief
marginalization can legitimately lead: latent next-cell prediction under known noisy or
ambiguous emissions, locked traveller splits, train-only transition learning, and the same
fair baseline roster used by the canonical SOTA harness.

The gate is regime-local.  A regime clears only when the best AIF-family candidate beats
the strongest non-AIF fair baseline by mean loss and by traveller-clustered CI in every
configured realization, while staying above the oracle/leakage floor.  A cleared regime is
still synthetic-only and does not clear global, external, empirical, operational, or
foundation-model SOTA claims.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .power_analysis import traveller_clustered_bootstrap_ci
from .sota_aif_candidate import fit_aif_candidate
from .sota_program import (
    PartialObservabilityDataset,
    ScoredCandidate,
    build_partial_observability_dataset,
    paired_loss_difference_ci,
    score_candidate,
)
from .sota_program_harness import (
    AIF_FAMILY_NAMES,
    DEFAULT_BOOT_SEED,
    DEFAULT_SPATIAL_BINS,
    DEPENDENCY_FOOTPRINTS,
    LEAKAGE_FLOOR_NATS,
    _candidate_family,
    _csv_cell,
    _roster,
    build_inference_config,
)
from .sota_stressors import (
    build_regime_stressor_partial_observability_dataset,
    build_reporting_gap_partial_observability_dataset,
)
from .types import InferenceConfig

AIF_REGIME_SEARCH_SCHEMA = "active-fractal-rabbit-aif-regime-search-v1"
AIF_REGIME_GATE_SCHEMA = "active-fractal-rabbit-aif-regime-sota-gate-v1"

DEFAULT_AIF_REGIME_N_TRAVELLERS = 40
# 2000 matches DEFAULT_N_BOOT across the SOTA lanes: the regime family produced the
# benchmark's lone uncorrected clearer, and at B=300 its smallest reportable p was the
# 1/301 resolution floor with a Monte-Carlo SE ~35% of the headline p. Unified 2026-07-01.
DEFAULT_AIF_REGIME_N_BOOT = 2000
DEFAULT_AIF_REGIME_REALIZATION_SEEDS = (20260625, 20260626)
DEFAULT_AIF_REGIME_SPLIT_SEED = 20260616
DEFAULT_AIF_REGIME_NOISE_SEED = 20260701

CANONICAL_AIF_REGIMES = (
    "identity_emission_collapse",
    "symmetric_noise_045",
    "ambiguous_pairs_010",
    "burst_reporting_gap_035",
    "latent_regime_switch_035",
)

AIF_REGIME_AIF_FAMILY_NAMES = frozenset(
    set(AIF_FAMILY_NAMES) | {"learned_factorial_aif"}
)

AIF_REGIME_SEARCH_HEADER = [
    "regime_id",
    "realization_seed",
    "split_seed",
    "noise_seed",
    "latent_cell_count",
    "emission_model",
    "missingness_mode",
    "n_travellers",
    "candidate",
    "family",
    "role",
    "mean_loss",
    "loss_ci_low",
    "loss_ci_high",
    "clustered_ci_low",
    "clustered_ci_high",
    "brier_score",
    "ece",
    "top1_accuracy",
    "runtime_seconds",
    "dependency_footprint",
    "oracle_gap_nats",
    "leakage_status",
    "locked_split_scored",
    "train_only_transition",
    "emission_contract",
    "regime_gate_status",
    "failure_notes",
]


@dataclass(frozen=True)
class RegimeSpec:
    """A deterministic regime definition for the AIF search surface."""

    regime_id: str
    emission_model: str
    missingness_mode: str
    description: str
    eligible_for_regime_sota: bool
    builder: Callable[
        [InferenceConfig, int, int, int, int], PartialObservabilityDataset
    ]
    control_role: str = "candidate_regime"


@dataclass(frozen=True)
class RegimeRealizationResult:
    """Scoring result for one regime and one generator realization."""

    regime_id: str
    realization_seed: int
    split_seed: int
    noise_seed: int
    emission_model: str
    missingness_mode: str
    n_travellers: int
    n_states: int
    eligible_for_regime_sota: bool
    aif_gamma: float
    aif_boost: float
    aif_adaptation: float
    scored: dict[str, ScoredCandidate]
    best_aif_name: str
    best_aif_loss: float
    strongest_fair_name: str
    strongest_fair_loss: float
    mean_difference_nats: float
    clustered_ci_low: float
    clustered_ci_high: float
    one_sided_p: float
    aif_beats_strongest_fair: bool
    leakage_floor_ok: bool
    n_holdout_travellers: int


def build_aif_regime_specs() -> dict[str, RegimeSpec]:
    """Return stable regime definitions for the synthetic search surface."""

    return {
        "identity_emission_collapse": RegimeSpec(
            regime_id="identity_emission_collapse",
            emission_model="identity_known_emission_eps_0",
            missingness_mode="none",
            description=(
                "Negative control: observations equal latent cells. Recursive belief "
                "marginalization should not create a regime-qualified AIF advantage."
            ),
            eligible_for_regime_sota=False,
            control_role="negative_control",
            builder=_build_identity_regime,
        ),
        "symmetric_noise_045": RegimeSpec(
            regime_id="symmetric_noise_045",
            emission_model="known_symmetric_noisy_emission_eps_0.45",
            missingness_mode="none",
            description=(
                "High-noise known emission channel where soft belief marginalization can "
                "beat point estimates if the implemented fair baselines cannot match it."
            ),
            eligible_for_regime_sota=True,
            builder=_build_symmetric_noise_regime,
        ),
        "ambiguous_pairs_010": RegimeSpec(
            regime_id="ambiguous_pairs_010",
            emission_model="known_coarsened_pairs_with_positive_floor_0.01",
            missingness_mode="coarsened_observation",
            description=(
                "Ambiguous emission: observations identify two-cell groups rather than "
                "individual latent cells, with a strictly positive channel floor."
            ),
            eligible_for_regime_sota=True,
            builder=_build_ambiguous_pair_regime,
        ),
        "burst_reporting_gap_035": RegimeSpec(
            regime_id="burst_reporting_gap_035",
            emission_model="known_symmetric_noisy_emission_eps_0.35",
            missingness_mode="deterministic_burst_reporting_gap",
            description=(
                "Sparse-report synthetic lane with deterministic burst gaps and a known "
                "noisy emission channel shared by every candidate."
            ),
            eligible_for_regime_sota=True,
            builder=_build_burst_reporting_gap_regime,
        ),
        "latent_regime_switch_035": RegimeSpec(
            regime_id="latent_regime_switch_035",
            emission_model="known_symmetric_noisy_emission_eps_0.35",
            missingness_mode="short_bursts_plus_rare_long_gaps",
            description=(
                "Larger latent regime-switching trajectory process with return-rate and "
                "movement-mode shifts; synthetic-only."
            ),
            eligible_for_regime_sota=True,
            builder=_build_latent_regime_switch,
        ),
    }


def build_regime_dataset(
    regime_id: str,
    inference_config: InferenceConfig,
    *,
    n_travellers: int = DEFAULT_AIF_REGIME_N_TRAVELLERS,
    realization_seed: int = DEFAULT_AIF_REGIME_REALIZATION_SEEDS[0],
    split_seed: int = DEFAULT_AIF_REGIME_SPLIT_SEED,
    noise_seed: int = DEFAULT_AIF_REGIME_NOISE_SEED,
) -> PartialObservabilityDataset:
    """Build one locked, train-only regime-search dataset by ID."""

    specs = build_aif_regime_specs()
    if regime_id not in specs:
        raise KeyError(
            f"unknown AIF regime {regime_id!r}; expected one of {sorted(specs)}"
        )
    return specs[regime_id].builder(
        inference_config,
        int(n_travellers),
        int(realization_seed),
        int(split_seed),
        int(noise_seed),
    )


def score_regime_realization(
    spec: RegimeSpec,
    dataset: PartialObservabilityDataset,
    *,
    realization_seed: int,
    split_seed: int,
    noise_seed: int,
    n_boot: int = DEFAULT_AIF_REGIME_N_BOOT,
    boot_seed: int = DEFAULT_BOOT_SEED,
) -> RegimeRealizationResult:
    """Score every candidate and evaluate the regime-local AIF-vs-fair comparison."""

    aif_fit = fit_aif_candidate(
        dataset,
        adaptation_grid=(0.0,)
        if spec.control_role == "negative_control"
        else (0.0, 0.25, 0.5, 1.0),
    )
    scored: dict[str, ScoredCandidate] = {}
    for cand in _roster(dataset, aif_candidate=aif_fit.candidate):
        t0 = time.perf_counter()
        raw = score_candidate(dataset, cand)
        runtime_seconds = time.perf_counter() - t0
        try:
            loss_ci_low, loss_ci_high = traveller_clustered_bootstrap_ci(
                raw.step_losses,
                raw.step_travellers,
                n_boot=n_boot,
                seed=boot_seed,
            )
        except ValueError:
            loss_ci_low, loss_ci_high = None, None
        scored[cand.name] = replace(
            raw,
            loss_ci_low=loss_ci_low,
            loss_ci_high=loss_ci_high,
            runtime_seconds=runtime_seconds,
        )

    aif_scored = {
        name: candidate
        for name, candidate in scored.items()
        if name in AIF_REGIME_AIF_FAMILY_NAMES
    }
    if not aif_scored:
        raise ValueError(f"no AIF-family candidates scored for regime {spec.regime_id}")
    fair = {
        name: candidate
        for name, candidate in scored.items()
        if name not in AIF_REGIME_AIF_FAMILY_NAMES
    }
    if not fair:
        raise ValueError(
            f"no non-AIF fair baselines scored for regime {spec.regime_id}"
        )

    best_aif = min(aif_scored.values(), key=lambda row: row.pooled_mean_loss)
    strongest_fair = min(fair.values(), key=lambda row: row.pooled_mean_loss)
    ci = paired_loss_difference_ci(
        strongest_fair, best_aif, n_boot=n_boot, seed=boot_seed
    )
    leakage_floor_ok = bool(best_aif.pooled_mean_loss > LEAKAGE_FLOOR_NATS)
    aif_beats = bool(
        spec.eligible_for_regime_sota
        and leakage_floor_ok
        and best_aif.pooled_mean_loss < strongest_fair.pooled_mean_loss
        and ci.b_wins_clustered
    )
    return RegimeRealizationResult(
        regime_id=spec.regime_id,
        realization_seed=int(realization_seed),
        split_seed=int(split_seed),
        noise_seed=int(noise_seed),
        emission_model=spec.emission_model,
        missingness_mode=spec.missingness_mode,
        n_travellers=int(np.unique(dataset.travellers).size),
        n_states=int(dataset.n_states),
        eligible_for_regime_sota=spec.eligible_for_regime_sota,
        aif_gamma=float(aif_fit.gamma),
        aif_boost=float(aif_fit.boost),
        aif_adaptation=float(aif_fit.adaptation),
        scored=scored,
        best_aif_name=best_aif.name,
        best_aif_loss=float(best_aif.pooled_mean_loss),
        strongest_fair_name=strongest_fair.name,
        strongest_fair_loss=float(strongest_fair.pooled_mean_loss),
        mean_difference_nats=float(ci.mean_difference_nats),
        clustered_ci_low=float(ci.clustered_ci_low),
        clustered_ci_high=float(ci.clustered_ci_high),
        one_sided_p=float(ci.one_sided_p_b_better),
        aif_beats_strongest_fair=aif_beats,
        leakage_floor_ok=leakage_floor_ok,
        n_holdout_travellers=int(ci.n_holdout_travellers),
    )


def run_regime_search(
    inference_config: InferenceConfig,
    *,
    regime_ids: tuple[str, ...] = CANONICAL_AIF_REGIMES,
    realization_seeds: tuple[int, ...] = DEFAULT_AIF_REGIME_REALIZATION_SEEDS,
    n_travellers: int = DEFAULT_AIF_REGIME_N_TRAVELLERS,
    n_boot: int = DEFAULT_AIF_REGIME_N_BOOT,
    boot_seed: int = DEFAULT_BOOT_SEED,
) -> dict[str, object]:
    """Run the configured regime search and return a JSON-serialisable payload."""

    specs = build_aif_regime_specs()
    results: list[RegimeRealizationResult] = []
    for regime_id in regime_ids:
        spec = specs[regime_id]
        for idx, realization_seed in enumerate(realization_seeds):
            split_seed = DEFAULT_AIF_REGIME_SPLIT_SEED + idx
            noise_seed = DEFAULT_AIF_REGIME_NOISE_SEED + int(realization_seed % 997)
            dataset = build_regime_dataset(
                regime_id,
                inference_config,
                n_travellers=n_travellers,
                realization_seed=realization_seed,
                split_seed=split_seed,
                noise_seed=noise_seed,
            )
            results.append(
                score_regime_realization(
                    spec,
                    dataset,
                    realization_seed=realization_seed,
                    split_seed=split_seed,
                    noise_seed=noise_seed,
                    n_boot=n_boot,
                    boot_seed=boot_seed,
                )
            )

    rows = regime_search_rows(results)
    summaries = _regime_summaries(results)
    cleared = [summary for summary in summaries.values() if summary["gate_clears"]]
    best_cleared = _best_cleared_regime(cleared)
    gate_clears = bool(cleared)
    status = (
        "regime_qualified_sota_cleared"
        if gate_clears
        else "regime_qualified_sota_not_cleared"
    )
    best_regime_id = (
        str(best_cleared.get("regime_id", "none")) if best_cleared else "none"
    )
    best_aif_candidates = (
        best_cleared.get("best_aif_candidates", []) if best_cleared else []
    )
    best_strongest_fair = (
        best_cleared.get("strongest_fair_baselines", []) if best_cleared else []
    )
    best_min_advantage = (
        best_cleared.get("min_advantage_nats") if best_cleared else None
    )
    best_min_ci_low = best_cleared.get("min_clustered_ci_low") if best_cleared else None
    run_parameters = {
        "n_travellers": int(n_travellers),
        "spatial_bins": int(inference_config.spatial_bins),
        "n_boot": int(n_boot),
        "boot_seed": int(boot_seed),
        "regime_ids": list(regime_ids),
        "realization_seeds": [int(seed) for seed in realization_seeds],
        "n_realizations": int(len(list(realization_seeds))),
        "canonical": _is_canonical_regime_search(
            n_travellers=n_travellers,
            spatial_bins=inference_config.spatial_bins,
            n_boot=n_boot,
            boot_seed=boot_seed,
            regime_ids=regime_ids,
            realization_seeds=realization_seeds,
        ),
    }
    return {
        "schema": AIF_REGIME_SEARCH_SCHEMA,
        "gate_schema": AIF_REGIME_GATE_SCHEMA,
        "status": status,
        "gate_clears": gate_clears,
        "cleared_regime_count": len(cleared),
        "best_cleared_regime": best_cleared,
        "best_cleared_regime_id": best_regime_id,
        "best_aif_candidates": best_aif_candidates,
        "best_strongest_fair_baselines": best_strongest_fair,
        "best_min_advantage_nats": best_min_advantage,
        "best_min_clustered_ci_low": best_min_ci_low,
        "claim": (
            "Regime-qualified synthetic SOTA clears only for regimes where the best "
            "AIF-family candidate beats every implemented non-AIF fair baseline by mean "
            "loss and traveller-clustered CI across all configured realizations."
        ),
        "global_sota_boundary": (
            "This gate is separate from the canonical global SOTA program. A positive "
            "regime result is limited to the exact synthetic regime IDs, realization seeds, "
            "candidate roster, and observation-channel contract in this artifact; it does "
            "not imply external, empirical, operational, surveillance, foundation-model, "
            "or global SOTA performance."
        ),
        "run_parameters": run_parameters,
        "regime_definitions": {
            regime_id: {
                "emission_model": spec.emission_model,
                "missingness_mode": spec.missingness_mode,
                "description": spec.description,
                "eligible_for_regime_sota": spec.eligible_for_regime_sota,
                "control_role": spec.control_role,
            }
            for regime_id, spec in specs.items()
            if regime_id in regime_ids
        },
        "controls": _control_payload(summaries, results),
        "regime_summaries": summaries,
        "realization_summaries": [_realization_summary(result) for result in results],
        "leaderboard_columns": list(AIF_REGIME_SEARCH_HEADER),
        "leaderboard_rows": rows,
        "blocked_candidate_families": [
            {
                "family": "foundation_mobility",
                "status": "blocked_pending_license_compute_data_provenance_and_adapter_contract",
                "reason": (
                    "Foundation/LLM mobility models remain contextual until license, compute, "
                    "data provenance, and locked-split adapter contracts are locally auditable."
                ),
            },
            {
                "family": "learned_factorial_active_inference",
                "status": "blocked_pending_deterministic_train_only_factorization_contract",
                "reason": (
                    "A learned/factorial AIF candidate is not admitted until its factor learning "
                    "uses train travellers only and has deterministic leakage tests."
                ),
            },
        ],
    }


def regime_search_rows(
    results: list[RegimeRealizationResult],
) -> list[dict[str, object]]:
    """Flatten scored regime results into CSV/JSON leaderboard rows."""

    rows: list[dict[str, object]] = []
    for result in results:
        for name, scored in sorted(
            result.scored.items(), key=lambda item: item[1].pooled_mean_loss
        ):
            is_best_aif = name == result.best_aif_name
            is_strongest_fair = name == result.strongest_fair_name
            is_aif = name in AIF_REGIME_AIF_FAMILY_NAMES
            role = (
                "best_aif_candidate"
                if is_best_aif
                else "aif_reference"
                if is_aif
                else "strongest_fair_baseline"
                if is_strongest_fair
                else "fair_baseline"
            )
            rows.append(
                {
                    "regime_id": result.regime_id,
                    "realization_seed": result.realization_seed,
                    "split_seed": result.split_seed,
                    "noise_seed": result.noise_seed,
                    "latent_cell_count": result.n_states,
                    "emission_model": result.emission_model,
                    "missingness_mode": result.missingness_mode,
                    "n_travellers": result.n_travellers,
                    "candidate": name,
                    "family": _candidate_family(name),
                    "role": role,
                    "mean_loss": round(float(scored.pooled_mean_loss), 6),
                    "loss_ci_low": _round_or_empty(scored.loss_ci_low),
                    "loss_ci_high": _round_or_empty(scored.loss_ci_high),
                    "clustered_ci_low": round(result.clustered_ci_low, 6)
                    if is_best_aif
                    else "",
                    "clustered_ci_high": round(result.clustered_ci_high, 6)
                    if is_best_aif
                    else "",
                    "brier_score": round(float(scored.brier_score), 6),
                    "ece": round(float(scored.ece), 6),
                    "top1_accuracy": round(float(scored.top1_accuracy), 6),
                    "runtime_seconds": _round_or_empty(scored.runtime_seconds),
                    "dependency_footprint": _dependency_footprint(name),
                    "oracle_gap_nats": round(
                        float(scored.pooled_mean_loss - LEAKAGE_FLOOR_NATS), 6
                    ),
                    "leakage_status": (
                        "passes_oracle_floor"
                        if scored.pooled_mean_loss > LEAKAGE_FLOOR_NATS
                        else "fails_oracle_floor"
                    ),
                    "locked_split_scored": True,
                    "train_only_transition": True,
                    "emission_contract": "known_to_all_candidates",
                    "regime_gate_status": (
                        "realization_clears"
                        if result.aif_beats_strongest_fair
                        else "realization_not_cleared"
                    ),
                    "failure_notes": _failure_notes(
                        name=name, role=role, result=result
                    ),
                }
            )
    return rows


def write_regime_search_csv(rows: list[dict[str, object]], path: Path) -> Path:
    """Write the flat regime-search leaderboard CSV."""

    lines = [",".join(AIF_REGIME_SEARCH_HEADER)]
    for row in rows:
        lines.append(
            ",".join(_csv_cell(row[column]) for column in AIF_REGIME_SEARCH_HEADER)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def gate_payload(payload: dict[str, object]) -> dict[str, object]:
    """Extract the smaller manuscript/provenance-facing gate summary."""

    return {
        "schema": AIF_REGIME_GATE_SCHEMA,
        "status": payload["status"],
        "gate_clears": payload["gate_clears"],
        "cleared_regime_count": payload["cleared_regime_count"],
        "best_cleared_regime": payload["best_cleared_regime"],
        "best_cleared_regime_id": payload["best_cleared_regime_id"],
        "best_aif_candidates": payload["best_aif_candidates"],
        "best_strongest_fair_baselines": payload["best_strongest_fair_baselines"],
        "best_min_advantage_nats": payload["best_min_advantage_nats"],
        "best_min_clustered_ci_low": payload["best_min_clustered_ci_low"],
        "claim": payload["claim"],
        "global_sota_boundary": payload["global_sota_boundary"],
        "run_parameters": payload["run_parameters"],
        "controls": payload["controls"],
        "regime_definitions": payload["regime_definitions"],
        "regime_summaries": payload["regime_summaries"],
        "realization_summaries": payload["realization_summaries"],
        "leaderboard_columns": payload["leaderboard_columns"],
        "blocked_candidate_families": payload["blocked_candidate_families"],
    }


def run_and_write_aif_regime_search(
    config_path: str | Path = "configs/default.yaml",
    out_dir: str | Path = "output/runs",
    *,
    n_travellers: int = DEFAULT_AIF_REGIME_N_TRAVELLERS,
    n_boot: int = DEFAULT_AIF_REGIME_N_BOOT,
    boot_seed: int = DEFAULT_BOOT_SEED,
    spatial_bins: int = DEFAULT_SPATIAL_BINS,
    regime_ids: tuple[str, ...] = CANONICAL_AIF_REGIMES,
    realization_seeds: tuple[int, ...] = DEFAULT_AIF_REGIME_REALIZATION_SEEDS,
) -> tuple[dict[str, object], dict[str, Path]]:
    """Build the regime search artifacts in ``out_dir``."""

    inference_config = build_inference_config(config_path, spatial_bins=spatial_bins)
    payload = run_regime_search(
        inference_config,
        regime_ids=tuple(regime_ids),
        realization_seeds=tuple(realization_seeds),
        n_travellers=n_travellers,
        n_boot=n_boot,
        boot_seed=boot_seed,
    )
    output_dir = Path(out_dir)
    csv_path = write_regime_search_csv(
        payload["leaderboard_rows"],  # type: ignore[arg-type]
        output_dir / "aif_regime_search.csv",
    )
    search_path = output_dir / "aif_regime_search.json"
    gate_path = output_dir / "aif_regime_sota_gate.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    search_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate_path.write_text(
        json.dumps(gate_payload(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload, {"leaderboard": csv_path, "search": search_path, "gate": gate_path}


def _build_identity_regime(
    inference_config: InferenceConfig,
    n_travellers: int,
    realization_seed: int,
    split_seed: int,
    noise_seed: int,
) -> PartialObservabilityDataset:
    return build_partial_observability_dataset(
        inference_config,
        n_travellers=n_travellers,
        eps=0.0,
        gen_seed=realization_seed,
        split_seed=split_seed,
        noise_seed=noise_seed,
    )


def _build_symmetric_noise_regime(
    inference_config: InferenceConfig,
    n_travellers: int,
    realization_seed: int,
    split_seed: int,
    noise_seed: int,
) -> PartialObservabilityDataset:
    return build_partial_observability_dataset(
        inference_config,
        n_travellers=n_travellers,
        eps=0.45,
        gen_seed=realization_seed,
        split_seed=split_seed,
        noise_seed=noise_seed,
    )


def _build_ambiguous_pair_regime(
    inference_config: InferenceConfig,
    n_travellers: int,
    realization_seed: int,
    split_seed: int,
    noise_seed: int,
) -> PartialObservabilityDataset:
    base = build_partial_observability_dataset(
        inference_config,
        n_travellers=n_travellers,
        eps=0.0,
        gen_seed=realization_seed,
        split_seed=split_seed,
        noise_seed=noise_seed,
    )
    return _with_coarsened_emission(
        base, group_size=2, channel_noise=0.01, noise_seed=noise_seed
    )


def _build_burst_reporting_gap_regime(
    inference_config: InferenceConfig,
    n_travellers: int,
    realization_seed: int,
    split_seed: int,
    noise_seed: int,
) -> PartialObservabilityDataset:
    return build_reporting_gap_partial_observability_dataset(
        inference_config,
        n_travellers=n_travellers,
        eps=0.35,
        gen_seed=realization_seed,
        split_seed=split_seed,
        noise_seed=noise_seed,
    )


def _build_latent_regime_switch(
    inference_config: InferenceConfig,
    n_travellers: int,
    realization_seed: int,
    split_seed: int,
    noise_seed: int,
) -> PartialObservabilityDataset:
    return build_regime_stressor_partial_observability_dataset(
        inference_config,
        n_travellers=n_travellers,
        eps=0.35,
        gen_seed=realization_seed,
        split_seed=split_seed,
        noise_seed=noise_seed,
    )


def _with_coarsened_emission(
    dataset: PartialObservabilityDataset,
    *,
    group_size: int,
    channel_noise: float,
    noise_seed: int,
) -> PartialObservabilityDataset:
    emission = _coarsened_emission_matrix(
        dataset.n_states, group_size=group_size, channel_noise=channel_noise
    )
    rng = np.random.default_rng(noise_seed)
    observed = np.asarray(
        [
            int(rng.choice(dataset.n_states, p=emission[:, int(latent)]))
            for latent in dataset.latent
        ],
        dtype=int,
    )
    split = dict(dataset.split)
    metadata = (
        dict(split.get("lane_metadata", {}))
        if isinstance(split.get("lane_metadata"), dict)
        else {}
    )
    metadata.update(
        {
            "emission_model": f"coarsened_group_size_{group_size}_channel_noise_{channel_noise:g}",
            "strictly_positive_emission_floor": channel_noise > 0.0,
        }
    )
    split["lane_metadata"] = metadata
    return replace(
        dataset,
        observed=observed,
        eps=float(channel_noise),
        emission=emission,
        split=split,
    )


def _coarsened_emission_matrix(
    n_states: int, *, group_size: int, channel_noise: float
) -> np.ndarray:
    if n_states < 2:
        raise ValueError("coarsened emission needs at least two states")
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    if not (0.0 <= channel_noise < 1.0):
        raise ValueError("channel_noise must be in [0, 1)")
    emission = np.full(
        (n_states, n_states), channel_noise / max(n_states - 1, 1), dtype=float
    )
    for latent in range(n_states):
        anchor = (latent // group_size) * group_size
        emission[anchor, latent] = 1.0 - channel_noise
    emission /= emission.sum(axis=0, keepdims=True)
    return emission


def _regime_summaries(
    results: list[RegimeRealizationResult],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[RegimeRealizationResult]] = {}
    for result in results:
        grouped.setdefault(result.regime_id, []).append(result)
    summaries: dict[str, dict[str, object]] = {}
    for regime_id, regime_results in grouped.items():
        all_clear = all(result.aif_beats_strongest_fair for result in regime_results)
        eligible = all(result.eligible_for_regime_sota for result in regime_results)
        summaries[regime_id] = {
            "regime_id": regime_id,
            "status": "regime_cleared"
            if all_clear and eligible
            else "regime_not_cleared",
            "gate_clears": bool(all_clear and eligible),
            "eligible_for_regime_sota": bool(eligible),
            "n_realizations": len(regime_results),
            "realization_seeds": [result.realization_seed for result in regime_results],
            "best_aif_candidates": sorted(
                {result.best_aif_name for result in regime_results}
            ),
            "strongest_fair_baselines": sorted(
                {result.strongest_fair_name for result in regime_results}
            ),
            "mean_advantage_nats": round(
                float(np.mean([r.mean_difference_nats for r in regime_results])), 6
            ),
            "min_advantage_nats": round(
                float(min(r.mean_difference_nats for r in regime_results)), 6
            ),
            "min_clustered_ci_low": round(
                float(min(r.clustered_ci_low for r in regime_results)), 6
            ),
            "max_clustered_ci_high": round(
                float(max(r.clustered_ci_high for r in regime_results)), 6
            ),
            # Conservative regime-level one-sided p: the WORST (largest) realization p, so a
            # regime "wins" at uncorrected alpha only when even its least-significant
            # realization does — matching the gate's all-realizations-clear logic. Consumed
            # by the multiplicity ledger as this regime's per-comparison uncorrected p.
            "one_sided_p_max": round(
                float(max(r.one_sided_p for r in regime_results)), 6
            ),
            "one_sided_p_min": round(
                float(min(r.one_sided_p for r in regime_results)), 6
            ),
            "all_leakage_floor_ok": all(
                result.leakage_floor_ok for result in regime_results
            ),
            "boundary": (
                "Regime-local synthetic gate only. It cannot be generalized beyond this "
                "regime ID, realization set, candidate roster, and known-emission contract."
            ),
        }
    return summaries


def _realization_summary(result: RegimeRealizationResult) -> dict[str, object]:
    return {
        "regime_id": result.regime_id,
        "realization_seed": result.realization_seed,
        "split_seed": result.split_seed,
        "noise_seed": result.noise_seed,
        "emission_model": result.emission_model,
        "missingness_mode": result.missingness_mode,
        "n_travellers": result.n_travellers,
        "n_holdout_travellers": result.n_holdout_travellers,
        "latent_cell_count": result.n_states,
        "best_aif_candidate": result.best_aif_name,
        "aif_tuning": {
            "gamma": result.aif_gamma,
            "boost": result.aif_boost,
            "online_transition_adaptation": result.aif_adaptation,
            "selection_scope": "inner_train_transition_plus_inner_validation_score",
        },
        "best_aif_loss": round(result.best_aif_loss, 6),
        "strongest_fair_baseline": result.strongest_fair_name,
        "strongest_fair_loss": round(result.strongest_fair_loss, 6),
        "mean_difference_nats_fair_minus_aif": round(result.mean_difference_nats, 6),
        "clustered_ci_low": round(result.clustered_ci_low, 6),
        "clustered_ci_high": round(result.clustered_ci_high, 6),
        "one_sided_p": round(result.one_sided_p, 6),
        "realization_clears": result.aif_beats_strongest_fair,
        "leakage_floor_ok": result.leakage_floor_ok,
    }


def _control_payload(
    summaries: dict[str, dict[str, object]],
    results: list[RegimeRealizationResult],
) -> dict[str, object]:
    identity = summaries.get("identity_emission_collapse", {})
    return {
        "identity_emission_collapse_negative_control_passed": identity.get(
            "gate_clears"
        )
        is False,
        "oracle_ceiling_enforced": all(result.leakage_floor_ok for result in results),
        "locked_split_train_only_contract": all(
            _summary_has_nonempty_train_and_holdout(result) for result in results
        ),
        "claim_boundary": (
            "Controls are used to block overclaiming: identity emission must not be promoted "
            "as an AIF edge, and every candidate must remain above the oracle/leakage floor."
        ),
    }


def _summary_has_nonempty_train_and_holdout(result: RegimeRealizationResult) -> bool:
    return (
        result.n_holdout_travellers > 0
        and result.n_holdout_travellers < result.n_travellers
    )


def _best_cleared_regime(cleared: list[dict[str, object]]) -> dict[str, object] | None:
    if not cleared:
        return None
    return max(cleared, key=lambda row: float(row.get("min_advantage_nats", 0.0)))


def _dependency_footprint(name: str) -> str:
    return DEPENDENCY_FOOTPRINTS.get(name, "project_python_numpy_no_external_weights")


def _round_or_empty(value: float | None) -> object:
    if value is None:
        return ""
    return round(float(value), 6)


def _failure_notes(*, name: str, role: str, result: RegimeRealizationResult) -> str:
    if name == result.best_aif_name:
        if result.aif_beats_strongest_fair:
            return "best_aif_candidate_wins_this_realization_with_ci_separation"
        if not result.eligible_for_regime_sota:
            return "negative_control_not_eligible_for_regime_sota_promotion"
        if not result.leakage_floor_ok:
            return "best_aif_candidate_failed_oracle_leakage_floor"
        return (
            "best_aif_candidate_not_ci_separated_or_not_lower_loss_than_"
            f"{result.strongest_fair_name}_in_this_realization"
        )
    if role == "strongest_fair_baseline":
        return "strongest_fair_reference_bar_for_this_regime_realization"
    if name in AIF_REGIME_AIF_FAMILY_NAMES:
        return "aif_family_candidate_not_best_aif_for_this_realization"
    return "not_strongest_fair_for_this_regime_realization"


def _is_canonical_regime_search(
    *,
    n_travellers: int,
    spatial_bins: int,
    n_boot: int,
    boot_seed: int,
    regime_ids: tuple[str, ...],
    realization_seeds: tuple[int, ...],
) -> bool:
    return bool(
        int(n_travellers) == DEFAULT_AIF_REGIME_N_TRAVELLERS
        and int(spatial_bins) == DEFAULT_SPATIAL_BINS
        and int(n_boot) == DEFAULT_AIF_REGIME_N_BOOT
        and int(boot_seed) == DEFAULT_BOOT_SEED
        and tuple(regime_ids) == tuple(CANONICAL_AIF_REGIMES)
        and tuple(realization_seeds) == tuple(DEFAULT_AIF_REGIME_REALIZATION_SEEDS)
    )


__all__ = [
    "AIF_REGIME_SEARCH_SCHEMA",
    "AIF_REGIME_GATE_SCHEMA",
    "AIF_REGIME_SEARCH_HEADER",
    "CANONICAL_AIF_REGIMES",
    "DEFAULT_AIF_REGIME_N_TRAVELLERS",
    "DEFAULT_AIF_REGIME_N_BOOT",
    "DEFAULT_AIF_REGIME_REALIZATION_SEEDS",
    "RegimeSpec",
    "RegimeRealizationResult",
    "build_aif_regime_specs",
    "build_regime_dataset",
    "score_regime_realization",
    "run_regime_search",
    "regime_search_rows",
    "write_regime_search_csv",
    "gate_payload",
    "run_and_write_aif_regime_search",
]
