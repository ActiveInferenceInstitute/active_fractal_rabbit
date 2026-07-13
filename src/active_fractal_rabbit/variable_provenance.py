"""Fail-loud provenance for manuscript variables.

The manuscript hydrates result numbers from generated artifacts. This module
records where selected result-critical tokens came from and verifies that the
source artifact and rendered value still match before a render is trusted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANUSCRIPT_VARIABLE_PROVENANCE_SCHEMA = (
    "active-fractal-rabbit-manuscript-variable-provenance-v2"
)


class ProvenanceError(RuntimeError):
    """Raised when a manuscript variable provenance contract is violated."""


@dataclass(frozen=True)
class TokenProvenanceSpec:
    """One token-to-artifact binding."""

    token: str
    path: str
    # A ``(key, value)`` element selects the first list row whose ``key``
    # field equals ``value`` instead of a bare positional index, matching the
    # dict-keyed lookups manuscript_variables.py performs on the same
    # frontier/lane lists so the two derivations cannot silently desync if a
    # list's ordering changes.
    value_path: tuple[str | int | tuple[str, str], ...]
    formatter: str = "fmt"
    optional: bool = False


DEFAULT_TOKEN_PROVENANCE: tuple[TokenProvenanceSpec, ...] = (
    TokenProvenanceSpec(
        "OBSERVATION_COUNT", "output/runs/default/metrics.json", ("n_observations",)
    ),
    TokenProvenanceSpec(
        "HOLDOUT_COUNT", "output/runs/default/metrics.json", ("holdout_n_holdout",)
    ),
    TokenProvenanceSpec(
        "PYMDP_EXACT_INFERENCE_DELTA",
        "output/runs/default/metrics.json",
        ("pymdp_exact_inference_max_abs_delta",),
        formatter="sci",
    ),
    TokenProvenanceSpec(
        "ACTIVE_SENSING_BEATEN_FRACTION",
        "output/runs/active_sensing_analysis.json",
        ("lanes", 0, "budgets", 0, "fraction_random_beaten"),
    ),
    TokenProvenanceSpec(
        "ANOMALY_VFE_AUC_RARE", "output/runs/anomaly_analysis.json", ("vfe_auc_rare",)
    ),
    TokenProvenanceSpec(
        "ANOMALY_VFE_AUC_RANDOM",
        "output/runs/anomaly_analysis.json",
        ("vfe_auc_random",),
    ),
    TokenProvenanceSpec(
        "FRONTIER_NO_LOCATION_MASS",
        "output/runs/frontier_analysis.json",
        ("frontier", ("modalities", "no_location"), "mean_mass_on_true_cell"),
    ),
    TokenProvenanceSpec(
        "RETURN_MOVE_ENTROPY_REDUCTION",
        "output/runs/return_structure_analysis.json",
        ("move_step_entropy_reduction",),
    ),
    TokenProvenanceSpec(
        "PRIVACY_FRONTIER_NO_LOCATION_UNICITY",
        "output/runs/privacy_frontier_analysis.json",
        ("frontier", ("modalities", "no_location"), "unicity"),
    ),
    TokenProvenanceSpec(
        "HARDENED_ANOMALY_SPEED_VFE_AUC",
        "output/runs/hardened_anomaly_analysis.json",
        ("speed_regime_shift", "vfe_auc"),
    ),
    TokenProvenanceSpec(
        "FRACTAL_DIMENSION_SWEEP_STATUS",
        "output/runs/fractal_dimension_sweep.json",
        ("status",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "CIRCADIAN_PHASE_GAIN",
        "output/runs/circadian_phase_analysis.json",
        ("phase_gain_vs_pooled",),
    ),
    TokenProvenanceSpec(
        "CIRCADIAN_PHASE_GAIN_P",
        "output/runs/circadian_phase_analysis.json",
        ("phase_gain_p_value",),
    ),
    TokenProvenanceSpec(
        "CIRCADIAN_PHASE_MAXJS_P",
        "output/runs/circadian_phase_analysis.json",
        ("max_phase_js_p_value",),
    ),
    TokenProvenanceSpec(
        "CIRCADIAN_PHASE_N_PERM",
        "output/runs/circadian_phase_analysis.json",
        ("n_permutations",),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_N_CELLS",
        "output/runs/partial_observability_analysis.json",
        ("n_cells",),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_CLEAN_ADVANTAGE",
        "output/runs/partial_observability_analysis.json",
        ("clean_advantage_nats",),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_MAX_EPS",
        "output/runs/partial_observability_analysis.json",
        ("max_eps",),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_MAX_NOISE_ADVANTAGE",
        "output/runs/partial_observability_analysis.json",
        ("max_noise_advantage_nats",),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_MAX_NOISE_SD",
        "output/runs/partial_observability_analysis.json",
        ("max_noise_advantage_sd",),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_MAX_NOISE_MIN",
        "output/runs/partial_observability_analysis.json",
        ("max_noise_advantage_min",),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_N_REALIZATIONS",
        "output/runs/partial_observability_analysis.json",
        ("n_realizations_per_level",),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_NOISE_SLOPE",
        "output/runs/partial_observability_analysis.json",
        ("noise_slope_nats_per_eps",),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_NOISE_R2",
        "output/runs/partial_observability_analysis.json",
        ("noise_trend_r2",),
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_STATUS",
        "output/runs/sota_program_gate.json",
        ("status",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_GATE_CLEARS",
        "output/runs/sota_program_gate.json",
        ("gate_clears",),
        formatter="yes_no",
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_LANE_COUNT",
        "output/runs/sota_program_gate.json",
        ("robustness_summary", "canonical_lanes"),
        formatter="list_len",
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_CANONICAL_LANES",
        "output/runs/sota_program_gate.json",
        ("robustness_summary", "canonical_lanes"),
        formatter="canonical_lanes",
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_REPORTING_GAP_SCORED",
        "output/runs/sota_program_gate.json",
        ("robustness_summary", "reporting_gap_scored"),
        formatter="yes_no",
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_REGIME_STRESSOR_SCORED",
        "output/runs/sota_program_gate.json",
        ("robustness_summary", "regime_stressor_scored"),
        formatter="yes_no",
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_ROBUSTNESS_BOUNDARY",
        "output/runs/sota_program_gate.json",
        ("robustness_summary", "claim_boundary"),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_N_TRAVELLERS",
        "output/runs/sota_program_gate.json",
        ("lanes", "fractalrabbit", "n_travellers"),
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_N_HOLDOUT",
        "output/runs/sota_program_gate.json",
        ("lanes", "fractalrabbit", "n_holdout_travellers"),
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_FR_AIF_LOSS",
        "output/runs/sota_program_gate.json",
        ("lanes", "fractalrabbit", "aif_loss"),
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_FR_STRONGEST",
        "output/runs/sota_program_gate.json",
        ("lanes", "fractalrabbit", "strongest_fair_baseline"),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_FR_STRONGEST_LOSS",
        "output/runs/sota_program_gate.json",
        ("lanes", "fractalrabbit", "strongest_fair_loss"),
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_FR_CI_LOW",
        "output/runs/sota_program_gate.json",
        ("lanes", "fractalrabbit", "clustered_ci_low"),
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_FR_CI_HIGH",
        "output/runs/sota_program_gate.json",
        ("lanes", "fractalrabbit", "clustered_ci_high"),
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_GR_AIF_LOSS",
        "output/runs/sota_program_gate.json",
        ("lanes", "gravity", "aif_loss"),
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_GR_STRONGEST",
        "output/runs/sota_program_gate.json",
        ("lanes", "gravity", "strongest_fair_baseline"),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "SOTA_PROGRAM_GR_STRONGEST_LOSS",
        "output/runs/sota_program_gate.json",
        ("lanes", "gravity", "strongest_fair_loss"),
    ),
    TokenProvenanceSpec(
        "AIF_REGIME_SOTA_STATUS",
        "output/runs/aif_regime_sota_gate.json",
        ("status",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "AIF_REGIME_SOTA_GATE_CLEARS",
        "output/runs/aif_regime_sota_gate.json",
        ("gate_clears",),
        formatter="yes_no",
    ),
    TokenProvenanceSpec(
        "AIF_REGIME_SOTA_CLEARED_COUNT",
        "output/runs/aif_regime_sota_gate.json",
        ("cleared_regime_count",),
    ),
    TokenProvenanceSpec(
        "AIF_REGIME_SOTA_BEST_REGIME",
        "output/runs/aif_regime_sota_gate.json",
        ("best_cleared_regime_id",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "AIF_REGIME_SOTA_BEST_CANDIDATES",
        "output/runs/aif_regime_sota_gate.json",
        ("best_aif_candidates",),
        formatter="canonical_lanes",
    ),
    TokenProvenanceSpec(
        "AIF_REGIME_SOTA_STRONGEST_FAIR",
        "output/runs/aif_regime_sota_gate.json",
        ("best_strongest_fair_baselines",),
        formatter="canonical_lanes",
    ),
    TokenProvenanceSpec(
        "AIF_REGIME_SOTA_MIN_ADVANTAGE",
        "output/runs/aif_regime_sota_gate.json",
        ("best_min_advantage_nats",),
        formatter="nullable_fmt",
    ),
    TokenProvenanceSpec(
        "AIF_REGIME_SOTA_MIN_CI_LOW",
        "output/runs/aif_regime_sota_gate.json",
        ("best_min_clustered_ci_low",),
        formatter="nullable_fmt",
    ),
    TokenProvenanceSpec(
        "AIF_REGIME_SOTA_BOUNDARY",
        "output/runs/aif_regime_sota_gate.json",
        ("global_sota_boundary",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_ZERO_WHEN_CLEAN",
        "output/runs/partial_observability_analysis.json",
        ("zero_advantage_when_clean",),
        formatter="yes_no",
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_MONOTONE",
        "output/runs/partial_observability_analysis.json",
        ("advantage_monotone_with_noise",),
        formatter="yes_no",
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_CROSS_DRAW_COUNT",
        "output/runs/partial_observability_analysis.json",
        ("cross_realization_n_draws",),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_CROSS_DRAW_TRAVELLERS",
        "output/runs/partial_observability_analysis.json",
        ("cross_realization_n_travellers",),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_CROSS_DRAW_MAX_SD",
        "output/runs/partial_observability_analysis.json",
        ("cross_realization_max_between_draw_sd",),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_CROSS_DRAW_MEAN_ADVANTAGE",
        "output/runs/partial_observability_analysis.json",
        ("cross_realization_rows", 4, "mean_advantage_nats"),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_CROSS_DRAW_MIN_ADVANTAGE",
        "output/runs/partial_observability_analysis.json",
        ("cross_realization_rows", 4, "between_draw_min_nats"),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_CROSS_DRAW_MAX_ADVANTAGE",
        "output/runs/partial_observability_analysis.json",
        ("cross_realization_rows", 4, "between_draw_max_nats"),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_CROSS_DRAW_SIGN_STABLE",
        "output/runs/partial_observability_analysis.json",
        ("cross_realization_sign_stable_under_noise",),
        formatter="yes_no",
    ),
    TokenProvenanceSpec(
        "PYMDP_SOTA_STATUS",
        "output/runs/pymdp_benchmark_matrix.json",
        ("state_of_art_performance_status",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "PYMDP_PERFORMANCE_DELTA",
        "output/runs/pymdp_benchmark_matrix.json",
        ("pymdp_minus_reference_predictive_loss",),
    ),
    TokenProvenanceSpec(
        "PYMDP_ROADMAP_GATES",
        "output/runs/pymdp_benchmark_matrix.json",
        ("roadmap_gates",),
        formatter="roadmap_gates",
    ),
    TokenProvenanceSpec(
        "VARIANCE_GAP_FILL_LOW",
        "output/runs/variance_bands.json",
        ("metrics", "gap_filling_skill", "low"),
    ),
    TokenProvenanceSpec(
        "VARIANCE_GAP_FILL_HIGH",
        "output/runs/variance_bands.json",
        ("metrics", "gap_filling_skill", "high"),
    ),
    TokenProvenanceSpec(
        "SEQUENCE_BASELINE_BEST_MODEL",
        "output/runs/sequence_baseline_analysis.json",
        ("best_model",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "SEQUENCE_BASELINE_BEST_LOSS",
        "output/runs/sequence_baseline_analysis.json",
        ("best_mean_predictive_loss",),
    ),
    TokenProvenanceSpec(
        "SEQUENCE_BASELINE_GATE",
        "output/runs/sequence_baseline_analysis.json",
        ("pymdp_performance_gate",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "MISSINGNESS_VERDICT",
        "output/runs/missingness_model_analysis.json",
        ("verdict",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "MISSINGNESS_PHASE_GAIN",
        "output/runs/missingness_model_analysis.json",
        ("phase_gain_vs_pooled",),
    ),
    TokenProvenanceSpec(
        "MISSINGNESS_COMBINED_GAIN",
        "output/runs/missingness_model_analysis.json",
        ("phase_traveller_gain_vs_pooled",),
    ),
    TokenProvenanceSpec(
        "RETURN_PHASE_VERDICT",
        "output/runs/return_phase_prior_analysis.json",
        ("verdict",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "RETURN_PHASE_BEST_MODEL",
        "output/runs/return_phase_prior_analysis.json",
        ("best_model",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "RETURN_PHASE_BEST_LOSS",
        "output/runs/return_phase_prior_analysis.json",
        ("best_mean_predictive_loss",),
    ),
    TokenProvenanceSpec(
        "RETURN_PHASE_GAIN_VS_PERSISTENCE",
        "output/runs/return_phase_prior_analysis.json",
        ("return_plus_phase_gain_vs_persistence",),
    ),
    TokenProvenanceSpec(
        "RETURN_PHASE_GAIN_OVER_PERMUTED",
        "output/runs/return_phase_prior_analysis.json",
        ("return_plus_phase_gain_over_permuted",),
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_VERDICT",
        "output/runs/spatial_resolution_frontier.json",
        ("verdict",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_DEFAULT_BINS",
        "output/runs/spatial_resolution_frontier.json",
        ("default_spatial_bins",),
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_DEFAULT_STATES",
        "output/runs/spatial_resolution_frontier.json",
        ("default_row", "state_count"),
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_DEFAULT_FULL_LOSS",
        "output/runs/spatial_resolution_frontier.json",
        ("default_row", "full_location_mean_predictive_loss"),
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_DEFAULT_MINUS_COARSE_LOSS",
        "output/runs/spatial_resolution_frontier.json",
        ("default_minus_coarse", "full_location_mean_predictive_loss"),
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_DEFAULT_MINUS_COARSE_NO_LOCATION_MASS",
        "output/runs/spatial_resolution_frontier.json",
        ("default_minus_coarse", "no_location_mean_mass_on_true_cell"),
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_PYMDP_DELTA",
        "output/runs/spatial_resolution_frontier.json",
        ("default_row", "pymdp_exact_inference_max_abs_delta"),
        formatter="sci",
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_STATUS",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("comparison", "status"),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_VERDICT",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("comparison", "verdict"),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_N",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("larger_sample", "n_observations"),
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_REPORTS_PER_TRAVELLER",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("larger_sample", "reports_per_traveller"),
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_BEST_FEASIBLE_BINS",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("larger_sample", "best_feasible_bins"),
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_STRICT_FEASIBLE_COUNT",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("larger_sample", "strict_feasible_candidate_count"),
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_DEFAULT_MINUS_COARSE_LOSS",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("larger_sample", "default_minus_coarse_loss"),
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_DEFAULT_MINUS_COARSE_NO_LOCATION_MASS",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("larger_sample", "default_minus_coarse_no_location_mass"),
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_MULTIPLIERS",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("sample_size_ladder_summary", "multipliers"),
        formatter="int_list",
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_LARGEST_MULTIPLIER",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("sample_size_ladder_summary", "largest_multiplier"),
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_LARGEST_N",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("sample_size_ladder_summary", "largest_n_observations"),
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_LARGEST_REPORTS_PER_TRAVELLER",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("sample_size_ladder_summary", "largest_reports_per_traveller"),
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_LADDER_VERDICT_STABLE",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("sample_size_ladder_summary", "all_same_verdict"),
        formatter="yes_no",
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_GUARDRAIL_STABLE",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("sample_size_ladder_summary", "all_same_no_location_guardrail_direction"),
        formatter="yes_no",
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_STRICT_FEASIBLE_SERIES",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("sample_size_ladder_summary", "strict_feasible_counts"),
        formatter="int_list",
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_SELECTED_BINS_SERIES",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("sample_size_ladder_summary", "selected_bins"),
        formatter="int_list",
    ),
    TokenProvenanceSpec(
        "SPATIAL_RESOLUTION_CONFIRMATION_LADDER_INTERPRETATION",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        ("sample_size_ladder_summary", "interpretation"),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "MODEL_GATE_MATRIX_STATUS",
        "output/runs/model_gate_matrix.json",
        ("status",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "MODEL_GATE_SEQUENCE_EXTERNAL_GATE",
        "output/runs/model_gate_matrix.json",
        ("external_sequence_gate",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "MODEL_GATE_MISSINGNESS_EXTERNAL_VERDICT",
        "output/runs/model_gate_matrix.json",
        ("external_missingness_verdict",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "MODEL_GATE_RETURN_PHASE_EXTERNAL_VERDICT",
        "output/runs/model_gate_matrix.json",
        ("external_return_phase_verdict",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "FORWARD_GATE_STATUS",
        "output/runs/forward_gates_summary.json",
        ("status",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "LATENT_REGIME_HMM_GATE",
        "output/runs/latent_regime_hmm.json",
        ("gate",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "LATENT_REGIME_HMM_BEST_HELDOUT_LOSS",
        "output/runs/latent_regime_hmm.json",
        ("best_honest_heldout_loss",),
    ),
    TokenProvenanceSpec(
        "NEURAL_BASELINE_GATE",
        "output/runs/neural_baseline.json",
        ("gate",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "NEURAL_BASELINE_LOSS",
        "output/runs/neural_baseline.json",
        ("neural_logistic_mean_predictive_loss",),
    ),
    TokenProvenanceSpec(
        "LEARNED_STAY_RATE_VERDICT",
        "output/runs/learned_stay_rate.json",
        ("verdict",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "LEARNED_STAY_RATE_DIFF",
        "output/runs/learned_stay_rate.json",
        ("best_variant_minus_particle_mean",),
    ),
    TokenProvenanceSpec(
        "LEARNED_STAY_RATE_CI_LOW",
        "output/runs/learned_stay_rate.json",
        ("bootstrap_ci_low",),
    ),
    TokenProvenanceSpec(
        "LEARNED_STAY_RATE_CI_HIGH",
        "output/runs/learned_stay_rate.json",
        ("bootstrap_ci_high",),
    ),
    TokenProvenanceSpec(
        "HELDOUT_REPORTING_VERDICT",
        "output/runs/heldout_reporting_process.json",
        ("verdict",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "HELDOUT_REPORTING_RUNLEN_LOTO_GAIN",
        "output/runs/heldout_reporting_process.json",
        ("runlen_loto_gain_vs_pooled",),
    ),
    TokenProvenanceSpec(
        "HELDOUT_REPORTING_RUNLEN_TEMPORAL_GAIN",
        "output/runs/heldout_reporting_process.json",
        ("runlen_temporal_gain_vs_pooled",),
    ),
    TokenProvenanceSpec(
        "CROSS_GENERATOR_PERSISTENCE_VERDICT",
        "output/runs/cross_generator_transfer.json",
        ("transfer_verdict", "persistence_strength"),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "CROSS_GENERATOR_RETURN_VERDICT",
        "output/runs/cross_generator_transfer.json",
        ("transfer_verdict", "return_structure_helps"),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "EFE_LOCATION_VERDICT",
        "output/runs/efe_location_frontier.json",
        ("verdict",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "EFE_LOCATION_SHARE",
        "output/runs/efe_location_frontier.json",
        ("location_joint_share",),
    ),
    TokenProvenanceSpec(
        "EFE_METADATA_NULL_P",
        "output/runs/efe_location_frontier.json",
        ("structure_null", "p_value_null_ge_real"),
    ),
    TokenProvenanceSpec(
        "EFE_RESIDUAL_WITHOUT_LOCATION",
        "output/runs/efe_location_frontier.json",
        ("residual_localized_without_location",),
    ),
    TokenProvenanceSpec(
        "LINKAGE_RISK_VERDICT",
        "output/runs/linkage_risk_frontier.json",
        ("verdict",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "LINKAGE_LOCATION_LINKAGE",
        "output/runs/linkage_risk_frontier.json",
        ("location_real_linkage",),
    ),
    TokenProvenanceSpec(
        "LINKAGE_NO_LOCATION_EXCEEDS_NULL",
        "output/runs/linkage_risk_frontier.json",
        ("frontier", ("regime", "no_location"), "structure_null", "exceeds_null"),
        formatter="yes_no",
    ),
    TokenProvenanceSpec(
        "POISONING_VERDICT",
        "output/runs/poisoning_robustness.json",
        ("verdict",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "POISONING_GROSS_DETECTED",
        "output/runs/poisoning_robustness.json",
        ("gross_poisoning_detected",),
        formatter="yes_no",
    ),
    TokenProvenanceSpec(
        "POISONING_SUBTLE_NULL",
        "output/runs/poisoning_robustness.json",
        ("subtle_drift_is_null_vs_markov",),
        formatter="yes_no",
    ),
    TokenProvenanceSpec(
        "POISONING_TELEPORT_AUC",
        "output/runs/poisoning_robustness.json",
        ("per_mode", "teleport", "0.1667", "vfe_auc_mean"),
    ),
    TokenProvenanceSpec(
        "SOTA_STRONGEST_BASELINE_LOSS",
        "output/runs/sota_benchmark_matrix.json",
        ("strongest_baseline_loss",),
    ),
    TokenProvenanceSpec(
        "SOTA_LEAKAGE_PROBE_LOSS",
        "output/runs/sota_benchmark_matrix.json",
        ("false_claim_oracle", "leakage_probe_loss"),
    ),
    TokenProvenanceSpec(
        "AIF_SWEEP_BEST_LOSS",
        "output/runs/aif_sweep.json",
        ("best_config_loss",),
    ),
    TokenProvenanceSpec(
        "FAIR_ORACLE_LOSS",
        "output/runs/fair_inference.json",
        ("ladder", "losses", "oracle"),
    ),
    TokenProvenanceSpec(
        "FAIR_HARD_LOSS",
        "output/runs/fair_inference.json",
        ("ladder", "losses", "hard_handicapped"),
    ),
    TokenProvenanceSpec(
        "FAIR_SOFT_LOSS",
        "output/runs/fair_inference.json",
        ("ladder", "losses", "soft_handicapped"),
    ),
    TokenProvenanceSpec(
        "FAIR_SOFT_BOOST_LOSS",
        "output/runs/fair_inference.json",
        ("ladder", "losses", "soft_plus_return_boost"),
    ),
    TokenProvenanceSpec(
        "FAIR_AIF_LOSS",
        "output/runs/fair_inference.json",
        ("ladder", "losses", "honest_aif"),
    ),
    TokenProvenanceSpec(
        "FAIR_AIF_BOOST_LOSS",
        "output/runs/fair_inference.json",
        ("ladder", "losses", "honest_aif_plus_boost"),
    ),
    TokenProvenanceSpec(
        "FAIR_SOFT_MARG_GAIN",
        "output/runs/fair_inference.json",
        ("ladder", "decomposition_nats", "soft_marginalization_gain"),
    ),
    TokenProvenanceSpec(
        "FAIR_NOISE_ADV_HIGH",
        "output/runs/fair_inference.json",
        ("noise_sweep", "advantage_range_nats", 1),
    ),
    TokenProvenanceSpec(
        "FAIR_SOTA_AIF_BEST_RATE",
        "output/runs/fair_inference.json",
        ("sota_robustness", "aif_class_best_fair_rate"),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "FAIR_SOTA_DISTINCT_TRAVELLERS",
        "output/runs/fair_inference.json",
        ("sota_robustness", "n_distinct_holdout_travellers_min"),
    ),
    TokenProvenanceSpec(
        "FAIR_SOTA_ADV_MEAN",
        "output/runs/fair_inference.json",
        ("sota_robustness", "aif_advantage_mean_nats"),
    ),
    TokenProvenanceSpec(
        "FAIR_SOTA_GAMMA",
        "output/runs/fair_inference.json",
        ("sota_robustness", "precision"),
    ),
    TokenProvenanceSpec(
        "FAIR_SOTA_LOSS_GAMMA1",
        "output/runs/fair_inference.json",
        ("sota_robustness", "mean_loss_gamma1"),
    ),
    TokenProvenanceSpec(
        "FAIR_SOTA_LOSS_PRECISION",
        "output/runs/fair_inference.json",
        ("sota_robustness", "mean_loss_precision"),
    ),
    TokenProvenanceSpec(
        "FAIR_SOTA_ECE_GAMMA1",
        "output/runs/fair_inference.json",
        ("sota_robustness", "mean_ece_gamma1"),
    ),
    TokenProvenanceSpec(
        "FAIR_SOTA_ECE_PRECISION",
        "output/runs/fair_inference.json",
        ("sota_robustness", "mean_ece_precision"),
    ),
    TokenProvenanceSpec(
        "FAIR_SOTA_PRECISION_LOSS_WINS",
        "output/runs/fair_inference.json",
        ("sota_robustness", "precision_gain_loss_win_rate"),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "POWER_MAX_TRAVELLERS",
        "output/runs/power_analysis.json",
        ("traveller_counts", -1),
    ),
    TokenProvenanceSpec(
        "POWER_SMALL_CLEAN_ADV",
        "output/runs/power_analysis.json",
        ("small_clean_advantage_nats",),
    ),
    TokenProvenanceSpec(
        "POWER_LARGE_CLEAN_ADV",
        "output/runs/power_analysis.json",
        ("large_clean_advantage_nats",),
    ),
    TokenProvenanceSpec(
        "POWER_LARGE_NOISY_ADV",
        "output/runs/power_analysis.json",
        ("large_noisy_advantage_nats",),
    ),
    TokenProvenanceSpec(
        "POWER_LARGE_NOISY_SD",
        "output/runs/power_analysis.json",
        ("large_noisy_advantage_sd",),
    ),
    TokenProvenanceSpec(
        "POWER_GEN_SEEDS",
        "output/runs/power_analysis.json",
        ("n_gen_seeds",),
    ),
    TokenProvenanceSpec(
        "POWER_NOISE_RESTORES",
        "output/runs/power_analysis.json",
        ("noise_restores_advantage_at_scale",),
        formatter="yes_no",
    ),
    TokenProvenanceSpec(
        "VISUAL_QA_STATUS",
        "output/runs/visual_quality_audit.json",
        ("status",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "VISUAL_QA_FIGURE_COUNT",
        "output/runs/visual_quality_audit.json",
        ("figure_count",),
    ),
    TokenProvenanceSpec(
        "VISUAL_QA_CAPTION_SOURCE_COUNT",
        "output/runs/visual_quality_audit.json",
        ("caption_source_language_count",),
    ),
    TokenProvenanceSpec(
        "VISUAL_QA_MIN_PIXEL_STDDEV",
        "output/runs/visual_quality_audit.json",
        ("min_pixel_stddev",),
    ),
    TokenProvenanceSpec(
        "EXTERNAL_OBSERVATION_COUNT",
        "output/runs/fractalrabbit_external/metrics.json",
        ("n_observations",),
        optional=True,
    ),
    TokenProvenanceSpec(
        "SOTA_CLAIM_GATE_STATUS",
        "output/runs/sota_claim_gate.json",
        ("status",),
        formatter="str",
        optional=True,
    ),
    # Multiplicity ledger (Holm / Benjamini-Hochberg over the deterministic comparison family).
    TokenProvenanceSpec(
        "MULTIPLICITY_ALPHA", "output/runs/multiplicity_ledger.json", ("alpha",)
    ),
    TokenProvenanceSpec(
        "MULTIPLICITY_FDR_Q", "output/runs/multiplicity_ledger.json", ("fdr_q",)
    ),
    TokenProvenanceSpec(
        "MULTIPLICITY_N_BOOT",
        "output/runs/multiplicity_ledger.json",
        ("headline", "min_n_boot"),
    ),
    TokenProvenanceSpec(
        "MULTIPLICITY_P_FLOOR",
        "output/runs/multiplicity_ledger.json",
        ("headline", "max_p_resolution_floor"),
        formatter="sci",
    ),
    TokenProvenanceSpec(
        "MULTIPLICITY_MIN_ADVANTAGE_NATS",
        "output/runs/multiplicity_ledger.json",
        ("headline", "family_min_advantage_nats"),
    ),
    TokenProvenanceSpec(
        "MULTIPLICITY_MAX_ADVANTAGE_NATS",
        "output/runs/multiplicity_ledger.json",
        ("headline", "family_max_advantage_nats"),
    ),
    TokenProvenanceSpec(
        "SOTA_LANE_HOLDOUT_TRAVELLERS",
        "output/runs/multiplicity_ledger.json",
        ("headline", "min_lane_holdout_travellers"),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_SIGN_TEST_BOUND",
        "output/runs/multiplicity_ledger.json",
        ("partial_observability_positive", "sign_test_bound"),
        formatter="sci",
    ),
    TokenProvenanceSpec(
        "MULTIPLICITY_BONFERRONI_THRESHOLD",
        "output/runs/multiplicity_ledger.json",
        ("partial_observability_positive", "bonferroni_per_comparison_threshold"),
        formatter="sci",
    ),
    TokenProvenanceSpec(
        "AIF_REGIME_N_REALIZATIONS",
        "output/runs/aif_regime_search.json",
        ("run_parameters", "n_realizations"),
    ),
    TokenProvenanceSpec(
        "AIF_REGIME_N_BOOT",
        "output/runs/aif_regime_search.json",
        ("run_parameters", "n_boot"),
    ),
    TokenProvenanceSpec(
        "AIF_REGIME_TRAVELLER_COUNT",
        "output/runs/aif_regime_search.json",
        ("run_parameters", "n_travellers"),
    ),
    TokenProvenanceSpec(
        "SOTA_LANE_N_BOOT",
        "output/runs/sota_program_gate.json",
        ("run_parameters", "n_boot"),
    ),
    TokenProvenanceSpec(
        "VARIANCE_GAP_FILL_SD",
        "output/runs/variance_bands.json",
        ("metrics", "gap_filling_skill", "sd"),
    ),
    TokenProvenanceSpec(
        "VARIANCE_GAP_FILL_IQR_LOW",
        "output/runs/variance_bands.json",
        ("metrics", "gap_filling_skill", "q25"),
    ),
    TokenProvenanceSpec(
        "VARIANCE_GAP_FILL_IQR_HIGH",
        "output/runs/variance_bands.json",
        ("metrics", "gap_filling_skill", "q75"),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_SMOOTHING",
        "output/runs/partial_observability_analysis.json",
        ("smoothing",),
    ),
    TokenProvenanceSpec(
        "POLICY_PROXY_EXPLORATION_SELECTED_FRACTION",
        "output/runs/default/metrics.json",
        ("policy_exploration_selected_fraction",),
    ),
    TokenProvenanceSpec(
        "POWER_SMALL_N_HOLDOUT",
        "output/runs/power_analysis.json",
        ("cells", 0, "min_n_holdout_travellers"),
    ),
    TokenProvenanceSpec(
        "POWER_SMALL_MIN_ACROSS_SEEDS",
        "output/runs/power_analysis.json",
        ("cells", 0, "advantage_min_across_seeds"),
    ),
    TokenProvenanceSpec(
        "MULTIPLICITY_FAMILY_SIZE",
        "output/runs/multiplicity_ledger.json",
        ("family_size",),
    ),
    TokenProvenanceSpec(
        "MULTIPLICITY_UNCORRECTED_CLEARS",
        "output/runs/multiplicity_ledger.json",
        ("n_uncorrected_significant",),
    ),
    TokenProvenanceSpec(
        "MULTIPLICITY_HOLM_SURVIVORS",
        "output/runs/multiplicity_ledger.json",
        ("holm", "n_survivors"),
    ),
    TokenProvenanceSpec(
        "MULTIPLICITY_BH_SURVIVORS",
        "output/runs/multiplicity_ledger.json",
        ("benjamini_hochberg", "n_survivors"),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_MULTIPLICITY_DRAWS",
        "output/runs/multiplicity_ledger.json",
        ("partial_observability_positive", "n_independent_draws"),
    ),
    TokenProvenanceSpec(
        "PARTIAL_OBS_MIN_DRAWS_BONFERRONI",
        "output/runs/multiplicity_ledger.json",
        ("partial_observability_positive", "min_draws_for_family_bonferroni"),
    ),
    # Confound taxonomy (the map of dissolved advantages).
    TokenProvenanceSpec(
        "CONFOUND_N_DISSOLVED", "output/runs/confound_taxonomy.json", ("n_confounds",)
    ),
    TokenProvenanceSpec(
        "CONFOUND_N_CLASSES",
        "output/runs/confound_taxonomy.json",
        ("n_confound_classes_present",),
    ),
    TokenProvenanceSpec(
        "CONFOUND_CROSS_VENDOR_CAUGHT",
        "output/runs/confound_taxonomy.json",
        ("n_cross_vendor_caught",),
    ),
    # Regime mechanism synthesis (WHERE/HOW MUCH/WHY per-regime map).
    TokenProvenanceSpec(
        "REGIME_MECHANISM_N_REGIMES",
        "output/runs/regime_mechanism_synthesis.json",
        ("n_regimes",),
    ),
    TokenProvenanceSpec(
        "REGIME_MECHANISM_N_NULL",
        "output/runs/regime_mechanism_synthesis.json",
        ("verdict_counts", "null_no_signal"),
        formatter="nullable_fmt",
    ),
    TokenProvenanceSpec(
        "REGIME_MECHANISM_N_LOSS",
        "output/runs/regime_mechanism_synthesis.json",
        ("verdict_counts", "loses_to_fair_baseline"),
        formatter="nullable_fmt",
    ),
    TokenProvenanceSpec(
        "REGIME_MECHANISM_N_ROBUST",
        "output/runs/regime_mechanism_synthesis.json",
        ("verdict_counts", "robust_win"),
        formatter="nullable_fmt",
    ),
    # Regime-switch phase diagnostic (structural WHY: per-movement-phase breakdown).
    TokenProvenanceSpec(
        "PHASE_DIAGNOSTIC_WORST_SEGMENT",
        "output/runs/regime_switch_phase_diagnostic.json",
        ("worst_segment",),
    ),
    TokenProvenanceSpec(
        "PHASE_DIAGNOSTIC_BEST_SEGMENT",
        "output/runs/regime_switch_phase_diagnostic.json",
        ("best_segment",),
    ),
    TokenProvenanceSpec(
        "PHASE_DIAGNOSTIC_UNIFORM",
        "output/runs/regime_switch_phase_diagnostic.json",
        ("uniform_disadvantage",),
    ),
    TokenProvenanceSpec(
        "PHASE_DIAGNOSTIC_SEGMENT_COUNT",
        "output/runs/regime_switch_phase_diagnostic.json",
        ("segments",),
        formatter="list_len",
    ),
    TokenProvenanceSpec(
        "PHASE_DIAGNOSTIC_RATIO",
        "output/runs/regime_switch_phase_diagnostic.json",
        ("worst_to_best_magnitude_ratio",),
        formatter="n_a_fmt",
    ),
    # Privacy-utility frontier (surviving predictive utility vs re-identification risk).
    TokenProvenanceSpec(
        "PRIVACY_UTILITY_FINEST_EXCESS",
        "output/runs/privacy_utility_frontier.json",
        ("finest_excess_utility_bits",),
    ),
    TokenProvenanceSpec(
        "PRIVACY_UTILITY_COARSEST_EXCESS",
        "output/runs/privacy_utility_frontier.json",
        ("coarsest_excess_utility_bits",),
    ),
    TokenProvenanceSpec(
        "PRIVACY_UTILITY_FINEST_REID",
        "output/runs/privacy_utility_frontier.json",
        ("finest_reid_event_uniqueness",),
    ),
    TokenProvenanceSpec(
        "PRIVACY_UTILITY_COARSEST_REID",
        "output/runs/privacy_utility_frontier.json",
        ("coarsest_reid_event_uniqueness",),
    ),
    # Minimum-collection operating point (privacy-clear governance metric).
    TokenProvenanceSpec(
        "MINCOLLECT_FOUND",
        "output/runs/privacy_utility_frontier.json",
        ("min_collection_operating_point", "found"),
        formatter="yes_no",
    ),
    TokenProvenanceSpec(
        "MINCOLLECT_UTILITY_FLOOR",
        "output/runs/privacy_utility_frontier.json",
        ("min_collection_operating_point", "utility_floor_bits"),
    ),
    TokenProvenanceSpec(
        "MINCOLLECT_RISK_CEILING",
        "output/runs/privacy_utility_frontier.json",
        ("min_collection_operating_point", "reid_risk_ceiling"),
    ),
    TokenProvenanceSpec(
        "MINCOLLECT_REID",
        "output/runs/privacy_utility_frontier.json",
        ("min_collection_operating_point", "reid_event_uniqueness"),
    ),
    TokenProvenanceSpec(
        "MINCOLLECT_EXCESS_UTILITY",
        "output/runs/privacy_utility_frontier.json",
        ("min_collection_operating_point", "excess_utility_bits"),
    ),
    TokenProvenanceSpec(
        "PRIVACY_UTILITY_REPEATS_COMPLETED",
        "output/runs/privacy_utility_frontier_repeats.json",
        ("completed_count",),
    ),
    TokenProvenanceSpec(
        "PRIVACY_UTILITY_REPEATS_UNAVAILABLE",
        "output/runs/privacy_utility_frontier_repeats.json",
        ("unavailable_count",),
    ),
    TokenProvenanceSpec(
        "TASK_PRIVACY_RELEASE_UTILITY_ALLOWED",
        "output/runs/synthetic_task_utility_privacy.json",
        ("any_release_utility_claim_allowed",),
        formatter="yes_no",
    ),
    TokenProvenanceSpec(
        "TASK_PRIVACY_JOINT_OPERATING_POINTS",
        "output/runs/synthetic_task_utility_privacy.json",
        ("joint_operating_point_count",),
    ),
    TokenProvenanceSpec(
        "AIF_DEEPENING_PREDICTIVE_GATE_CLEARS",
        "output/runs/aif_deepening_candidates.json",
        ("predictive_gate_clears",),
        formatter="yes_no",
    ),
    TokenProvenanceSpec(
        "AIF_DEEPENING_PROTOCOL_CANDIDATES",
        "output/runs/aif_deepening_candidates.json",
        ("scored_protocol_candidate_count",),
    ),
    TokenProvenanceSpec(
        "ADAPTIVE_AIF_REALIZATIONS",
        "output/runs/adaptive_aif_ablation.json",
        ("realization_count",),
    ),
    TokenProvenanceSpec(
        "ADAPTIVE_AIF_MEAN_GAIN",
        "output/runs/adaptive_aif_ablation.json",
        ("adaptive_improvement_nats",),
    ),
    TokenProvenanceSpec(
        "ADAPTIVE_AIF_GAIN_LOW",
        "output/runs/adaptive_aif_ablation.json",
        ("adaptive_improvement_range_nats", 0),
    ),
    TokenProvenanceSpec(
        "ADAPTIVE_AIF_GAIN_HIGH",
        "output/runs/adaptive_aif_ablation.json",
        ("adaptive_improvement_range_nats", 1),
    ),
    TokenProvenanceSpec(
        "ADAPTIVE_AIF_FIXED_CI_CLEARS",
        "output/runs/adaptive_aif_ablation.json",
        ("adaptive_beats_fixed_ci_count",),
    ),
    TokenProvenanceSpec(
        "ADAPTIVE_AIF_FAIR_CI_CLEARS",
        "output/runs/adaptive_aif_ablation.json",
        ("adaptive_beats_online_base_rate_ci_count",),
    ),
    TokenProvenanceSpec(
        "ADAPTIVE_AIF_GATE_CLEARS",
        "output/runs/adaptive_aif_ablation.json",
        ("predictive_gate_clears",),
        formatter="yes_no",
    ),
    # Within-regime model comparison (where active inference leads, bounded).
    TokenProvenanceSpec(
        "MODEL_COMPARISON_AIF_RANK",
        "output/runs/model_comparison_leaderboard.json",
        ("aif_rank",),
    ),
    TokenProvenanceSpec(
        "MODEL_COMPARISON_CHANCE_LOSS",
        "output/runs/model_comparison_leaderboard.json",
        ("chance_loss_nats",),
    ),
    TokenProvenanceSpec(
        "MODEL_COMPARISON_N_METHODS",
        "output/runs/model_comparison_leaderboard.json",
        ("n_methods",),
    ),
    TokenProvenanceSpec(
        "MODEL_COMPARISON_STRONGEST_NON_AIF",
        "output/runs/model_comparison_leaderboard.json",
        ("strongest_non_aif_method",),
        formatter="str",
    ),
    TokenProvenanceSpec(
        "MODEL_COMPARISON_AIF_VS_NONAIF_GAP",
        "output/runs/model_comparison_leaderboard.json",
        ("aif_vs_strongest_non_aif_gap",),
    ),
    TokenProvenanceSpec(
        "MODEL_COMPARISON_BELIEF_NONBELIEF_GAP",
        "output/runs/model_comparison_leaderboard.json",
        ("belief_vs_nonbelief_gap",),
    ),
    TokenProvenanceSpec(
        "MODEL_COMPARISON_N_BELIEF",
        "output/runs/model_comparison_leaderboard.json",
        ("n_belief_methods",),
    ),
    TokenProvenanceSpec(
        "STRUCTURE_CROSS_DRAW_COUNT",
        "output/runs/structure_recovery_analysis.json",
        ("cross_generator", "n_draws"),
    ),
    TokenProvenanceSpec(
        "STRUCTURE_CROSS_DRAW_TRAVELLERS",
        "output/runs/structure_recovery_analysis.json",
        ("cross_generator", "n_travellers"),
    ),
    TokenProvenanceSpec(
        "STRUCTURE_CROSS_BEATS_DYNAMICS_FRACTION",
        "output/runs/structure_recovery_analysis.json",
        ("cross_generator", "beats_dynamics_only_fraction"),
    ),
    TokenProvenanceSpec(
        "STRUCTURE_CROSS_DEEP_LATENT_TV_MEAN",
        "output/runs/structure_recovery_analysis.json",
        ("cross_generator", "tv_deep_latent", "mean"),
    ),
    TokenProvenanceSpec(
        "STRUCTURE_CROSS_DYNAMICS_ONLY_TV_MEAN",
        "output/runs/structure_recovery_analysis.json",
        ("cross_generator", "tv_dynamics_only", "mean"),
    ),
)


def build_token_provenance(
    project_root: str | Path,
    variables: dict[str, str],
    *,
    specs: tuple[TokenProvenanceSpec, ...] = DEFAULT_TOKEN_PROVENANCE,
) -> dict[str, object]:
    """Build a provenance payload, failing on missing sources or value drift."""

    root = Path(project_root).resolve()
    entries: list[dict[str, object]] = []
    for spec in specs:
        if spec.token not in variables:
            raise ProvenanceError(
                f"token {spec.token} is not emitted by manuscript variables"
            )
        artifact = root / spec.path
        if not artifact.exists():
            if spec.optional and variables[spec.token] == "unavailable":
                entries.append(
                    {
                        "token": spec.token,
                        "path": spec.path,
                        "value_path": list(spec.value_path),
                        "formatter": spec.formatter,
                        "optional": True,
                        "status": "unavailable",
                        "rendered_value": "unavailable",
                    }
                )
                continue
            raise ProvenanceError(f"missing artifact for {spec.token}: {spec.path}")
        data = json.loads(artifact.read_text(encoding="utf-8"))
        raw_value = _extract(data, spec.value_path, spec.token)
        rendered = _format_value(raw_value, spec.formatter)
        if rendered != variables[spec.token]:
            raise ProvenanceError(
                f"rendered value for {spec.token} drifted: artifact={rendered!r} token={variables[spec.token]!r}"
            )
        entries.append(
            {
                "token": spec.token,
                "path": spec.path,
                "value_path": list(spec.value_path),
                "formatter": spec.formatter,
                "optional": spec.optional,
                "status": "ok",
                "artifact_sha256": _sha256(artifact),
                "artifact_bytes": artifact.stat().st_size,
                "artifact_value": raw_value,
                "rendered_value": rendered,
            }
        )
    return {
        "schema": MANUSCRIPT_VARIABLE_PROVENANCE_SCHEMA,
        "project_root": str(root),
        "declared_tokens": [spec.token for spec in specs],
        "entries": entries,
    }


def write_token_provenance(project_root: str | Path, variables: dict[str, str]) -> Path:
    """Write ``output/data/manuscript_variable_provenance.json``."""

    root = Path(project_root).resolve()
    payload = build_token_provenance(root, variables)
    output = root / "output" / "data" / "manuscript_variable_provenance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def verify_token_provenance(
    path: str | Path,
    *,
    expected_specs: tuple[TokenProvenanceSpec, ...] | None = None,
) -> dict[str, list[str]]:
    """Re-read and verify a previously written token provenance payload."""

    provenance_path = Path(path).resolve()
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    if payload.get("schema") != MANUSCRIPT_VARIABLE_PROVENANCE_SCHEMA:
        raise ProvenanceError("unexpected manuscript-variable provenance schema")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ProvenanceError("provenance entries must be a non-empty list")
    entry_tokens = [str(entry.get("token", "")) for entry in entries]
    if not all(entry_tokens) or len(set(entry_tokens)) != len(entry_tokens):
        raise ProvenanceError("provenance tokens must be non-empty and unique")
    if payload.get("declared_tokens") != entry_tokens:
        raise ProvenanceError("provenance declared_tokens do not match entry order")
    if expected_specs is not None:
        expected_tokens = [spec.token for spec in expected_specs]
        if entry_tokens != expected_tokens:
            raise ProvenanceError("provenance entries do not match the required token set")
        for entry, spec in zip(entries, expected_specs):
            if (
                str(entry.get("path")) != spec.path
                or _normalized_value_path(entry.get("value_path", ()))
                != spec.value_path
                or str(entry.get("formatter")) != spec.formatter
                or bool(entry.get("optional", False)) != spec.optional
            ):
                raise ProvenanceError(f"provenance specification drift for {spec.token}")
    root = _verification_root(provenance_path, Path(payload["project_root"]))
    checked: list[str] = []
    for entry in entries:
        token = str(entry["token"])
        if entry.get("status") == "unavailable":
            if entry.get("rendered_value") != "unavailable":
                raise ProvenanceError(
                    f"optional token {token} must render explicit unavailable"
                )
            checked.append(token)
            continue
        artifact = root / str(entry["path"])
        if not artifact.exists():
            raise ProvenanceError(f"missing artifact for {token}: {entry['path']}")
        if _sha256(artifact) != entry["artifact_sha256"]:
            raise ProvenanceError(f"checksum drift for {token}: {entry['path']}")
        data = json.loads(artifact.read_text(encoding="utf-8"))
        value_path = tuple(entry["value_path"])
        raw_value = _extract(data, value_path, token)
        rendered = _format_value(raw_value, str(entry["formatter"]))
        if rendered != entry["rendered_value"]:
            raise ProvenanceError(
                f"rendered value drift for {token}: artifact={rendered!r} provenance={entry['rendered_value']!r}"
            )
        checked.append(token)
    return {"checked": checked}


def _normalized_value_path(value_path: Any) -> tuple[Any, ...]:
    """Restore tuple selectors after their JSON list round-trip."""

    if not isinstance(value_path, list | tuple):
        return ()
    return tuple(tuple(item) if isinstance(item, list) else item for item in value_path)


def _verification_root(provenance_path: Path, payload_root: Path) -> Path:
    """Resolve the artifact root from the provenance file location.

    Release bundles are verified in-place, so a standard
    ``output/data/manuscript_variable_provenance.json`` file must bind to the
    project root containing that file. Otherwise a copied provenance payload
    could continue verifying against the original absolute path.
    """

    if (
        provenance_path.parent.name == "data"
        and provenance_path.parent.parent.name == "output"
    ):
        bundle_root = provenance_path.parent.parent.parent.resolve()
        if payload_root.is_absolute() and payload_root.resolve() != bundle_root:
            raise ProvenanceError(
                "provenance project_root does not match provenance file location; "
                "verify the bundle-local output/data/manuscript_variable_provenance.json"
            )
        if payload_root.is_absolute():
            return bundle_root
        return (bundle_root / payload_root).resolve()
    return (
        payload_root.resolve()
        if payload_root.is_absolute()
        else (provenance_path.parent / payload_root).resolve()
    )


def _select_row(rows: Any, key: str, value: str, token: str) -> dict[str, Any]:
    """Select the first list row whose ``key`` field equals ``value``."""

    if not isinstance(rows, list):
        raise ProvenanceError(f"expected a list to select a row for {token}")
    for row in rows:
        if isinstance(row, dict) and row.get(key) == value:
            return row
    raise ProvenanceError(f"no row with {key}={value!r} for {token}")


def _extract(
    data: Any, value_path: tuple[str | int | tuple[str, str], ...], token: str
) -> Any:
    current = data
    for part in value_path:
        try:
            if isinstance(part, (list, tuple)):
                key, value = part
                current = _select_row(current, str(key), str(value), token)
            else:
                current = current[part]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProvenanceError(
                f"missing key for {token}: {'.'.join(map(str, value_path))}"
            ) from exc
    return current


def format_fmt6(value: Any) -> str:
    """Format a numeric value to six decimals, collapsing whole numbers.

    Shared by manuscript_variables._fmt6 and the "fmt6" TokenProvenanceSpec
    formatter below so the two copies of this logic cannot silently diverge.
    """

    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:.6f}"


def format_fmt(value: Any) -> str:
    """Format a value to three decimals if a non-integer float, else ``str(value)``.

    Shared by manuscript_variables._fmt and the "fmt" TokenProvenanceSpec formatter
    below (the default formatter, previously duplicated verbatim in both places)
    so the two copies of this logic cannot silently diverge.
    """

    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.3f}"
    return str(value)


def _format_value(value: Any, formatter: str) -> str:
    if formatter == "str":
        return str(value)
    if formatter == "sci":
        return "unavailable" if value is None else f"{float(value):.2e}"
    if formatter == "fmt6":
        return format_fmt6(value)
    if formatter == "nullable_fmt":
        if value is None:
            return "unavailable"
        value = float(value)
        return str(int(value)) if value.is_integer() else f"{value:.3f}"
    if formatter == "n_a_fmt":
        return "n/a" if value is None else format_fmt(value)
    if formatter == "yes_no":
        if not isinstance(value, bool):
            raise ProvenanceError("yes_no formatter expects a boolean")
        return "yes" if value else "no"
    if formatter == "roadmap_gates":
        if not isinstance(value, list):
            raise ProvenanceError("roadmap_gates formatter expects a list")
        return "; ".join(str(item) for item in value) if value else "unavailable"
    if formatter == "int_list":
        if not isinstance(value, list):
            raise ProvenanceError("int_list formatter expects a list")
        return ", ".join(str(int(item)) for item in value) if value else "unavailable"
    if formatter == "list_len":
        if not isinstance(value, list):
            raise ProvenanceError("list_len formatter expects a list")
        return str(len(value))
    if formatter == "canonical_lanes":
        if not isinstance(value, list):
            raise ProvenanceError("canonical_lanes formatter expects a list")
        return ", ".join(f"`{item}`" for item in value) if value else "unavailable"
    if formatter != "fmt":
        raise ProvenanceError(f"unknown token formatter: {formatter}")
    return format_fmt(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
