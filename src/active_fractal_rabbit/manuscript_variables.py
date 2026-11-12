"""Hydrated manuscript variables derived from current artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import csv
import math
import os
import subprocess

from .config import load_config
from .sota_program_harness import canonical_run_parameter_violations
from .variable_provenance import format_fmt, format_fmt6, write_token_provenance

PAPER_TITLE = (
    "Active FractalRabbit: A Synthetic Benchmark for Belief Filtering Under "
    "Sparse Waypoint Observations"
)


def _gnn_variables(root: Path) -> dict[str, str]:
    """Return GNN-derived variables when the pinned runtime is available.

    The shared ``gnn.importers`` reader is the only source of the pin facts, so
    the manuscript producer never re-reads ``.github/gnn-pin.json``. The project
    environment does not install the pinned GNN distribution, so the operator
    supplies its interpreter in ``GNN_RUNTIME_PYTHON``. When that is absent the
    keys are omitted; the manuscript does not currently consume them.
    """

    runtime = os.environ.get("GNN_RUNTIME_PYTHON")
    if not runtime:
        return {}
    helper = root / "scripts" / "gnn_importer_variables.py"
    if not helper.is_file():
        return {}
    completed = subprocess.run(
        [
            runtime,
            str(helper),
            "--repo",
            str(root),
            "--job",
            "gnn-conformance",
            "--job-name",
            "gnn conformance (pinned)",
        ],
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {}
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {key: value for key, value in payload.items() if key.startswith("gnn_")}


def compute_variables(project_root: str | Path | None = None) -> dict[str, str]:
    """Compute variables consumed by the template hydration layer."""

    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    metrics_path = root / "output" / "runs" / "default" / "metrics.json"
    metrics = _read_json(metrics_path, {})
    baseline_metrics = _read_json(
        root / "output" / "runs" / "default" / "baseline_metrics.json", {}
    )
    sensitivity = _read_json(root / "output" / "runs" / "sensitivity_summary.json", {})
    negative = _read_json(
        root / "output" / "runs" / "negative_controls_summary.json", {}
    )
    external_metrics = _read_json(
        root / "output" / "runs" / "fractalrabbit_external" / "metrics.json", {}
    )
    external_sensitivity = _read_json(
        root / "output" / "runs" / "sensitivity_summary_external.json", {}
    )
    deep_latent_metrics = _read_json(
        root / "output" / "runs" / "sensitivity_no_location" / "metrics.json", {}
    )
    sparsity = _read_json(root / "output" / "runs" / "sparsity_analysis.json", {})
    anticipatory = _read_json(
        root / "output" / "runs" / "anticipatory_analysis.json", {}
    )
    anticipatory_gap = anticipatory.get("gap_filling", {})
    anticipatory_blind = anticipatory.get("location_blind_limit", {})
    partial_obs = _read_json(
        root / "output" / "runs" / "partial_observability_analysis.json", {}
    )
    _partial_obs_cross_rows = partial_obs.get("cross_realization_rows", [])
    partial_obs_cross_max_eps_row = next(
        (
            row
            for row in _partial_obs_cross_rows
            if isinstance(row, dict) and row.get("eps") == partial_obs.get("max_eps")
        ),
        {},
    )
    sota_program_path = root / "output" / "runs" / "sota_program_gate.json"
    sota_program = _read_json(sota_program_path, {})
    _require_canonical_sota_program(sota_program, sota_program_path)
    aif_regime_gate = _read_json(
        root / "output" / "runs" / "aif_regime_sota_gate.json", {}
    )
    multiplicity = _read_json(root / "output" / "runs" / "multiplicity_ledger.json", {})
    _aif_regime_search = _read_json(
        root / "output" / "runs" / "aif_regime_search.json", {}
    )
    aif_regime_run_parameters = (
        _aif_regime_search.get("run_parameters", {})
        if isinstance(_aif_regime_search.get("run_parameters", {}), dict)
        else {}
    )
    sota_gate_run_parameters = (
        sota_program.get("run_parameters", {})
        if isinstance(sota_program.get("run_parameters", {}), dict)
        else {}
    )
    confound_taxonomy = _read_json(
        root / "output" / "runs" / "confound_taxonomy.json", {}
    )
    regime_mechanism = _read_json(
        root / "output" / "runs" / "regime_mechanism_synthesis.json", {}
    )
    regime_switch_phase = _read_json(
        root / "output" / "runs" / "regime_switch_phase_diagnostic.json", {}
    )
    privacy_utility = _read_json(
        root / "output" / "runs" / "privacy_utility_frontier.json", {}
    )
    privacy_utility_repeats = _read_json(
        root / "output" / "runs" / "privacy_utility_frontier_repeats.json", {}
    )
    synthetic_task_privacy = _read_json(
        root / "output" / "runs" / "synthetic_task_utility_privacy.json", {}
    )
    aif_deepening = _read_json(
        root / "output" / "runs" / "aif_deepening_candidates.json", {}
    )
    adaptive_aif = _read_json(
        root / "output" / "runs" / "adaptive_aif_ablation.json", {}
    )
    model_comparison = _read_json(
        root / "output" / "runs" / "model_comparison_leaderboard.json", {}
    )
    sota_program_lanes = (
        sota_program.get("lanes", {}) if isinstance(sota_program, dict) else {}
    )
    sota_program_robustness = (
        sota_program.get("robustness_summary", {})
        if isinstance(sota_program.get("robustness_summary"), dict)
        else {}
    )
    sota_program_canonical_lanes = sota_program_robustness.get(
        "canonical_lanes",
        sota_program.get("run_parameters", {}).get("lanes", [])
        if isinstance(sota_program.get("run_parameters"), dict)
        else [],
    )
    sota_fr = sota_program_lanes.get("fractalrabbit", {})
    sota_gr = sota_program_lanes.get("gravity", {})
    active_sensing = _read_json(
        root / "output" / "runs" / "active_sensing_analysis.json", {}
    )
    efe_sweep = active_sensing.get("epistemic_share_sweep", [])
    # Prefer the external lane when present; otherwise fall back to the first
    # (fixture) lane. Never silently default the beaten-fraction to 0.0 — a
    # defaulted 0 would masquerade as the strongest possible honest null.
    efe_lanes = active_sensing.get("lanes", [])
    efe_lane = next(
        (lane for lane in efe_lanes if "external" in lane.get("label", "")),
        efe_lanes[0] if efe_lanes else {},
    )
    anomaly = _read_json(root / "output" / "runs" / "anomaly_analysis.json", {})
    structure = _read_json(
        root / "output" / "runs" / "structure_recovery_analysis.json", {}
    )
    structure_ext = next(
        (
            lane
            for lane in structure.get("lanes", [])
            if "external" in lane.get("label", "")
        ),
        structure.get("lanes", [{}])[0] if structure.get("lanes") else {},
    )
    structure_cross = structure.get("cross_generator", {})
    mission = _read_json(
        root / "output" / "runs" / "mission_collection_analysis.json", {}
    )
    frontier_rows = {
        r["modalities"]: r
        for r in _read_json(
            root / "output" / "runs" / "frontier_analysis.json", {}
        ).get("frontier", [])
    }
    return_structure = _read_json(
        root / "output" / "runs" / "return_structure_analysis.json", {}
    )
    variance_bands = _read_json(root / "output" / "runs" / "variance_bands.json", {})
    upstream_sensitivity = _read_json(
        root / "output" / "runs" / "upstream_sensitivity.json", {}
    )
    fractal_dimension = _read_json(
        root / "output" / "runs" / "fractal_dimension_sweep.json", {}
    )
    circadian_phase = _read_json(
        root / "output" / "runs" / "circadian_phase_analysis.json", {}
    )
    sequence_baselines = _read_json(
        root / "output" / "runs" / "sequence_baseline_analysis.json", {}
    )
    missingness_model = _read_json(
        root / "output" / "runs" / "missingness_model_analysis.json", {}
    )
    spatial_resolution = _read_json(
        root / "output" / "runs" / "spatial_resolution_frontier.json", {}
    )
    spatial_resolution_confirmation = _read_json(
        root
        / "output"
        / "runs"
        / "spatial_resolution_larger_sample_confirmation.json",
        {},
    )
    return_phase_prior = _read_json(
        root / "output" / "runs" / "return_phase_prior_analysis.json", {}
    )
    model_gate_matrix = _read_json(
        root / "output" / "runs" / "model_gate_matrix.json", {}
    )
    forward_gates = _read_json(
        root / "output" / "runs" / "forward_gates_summary.json", {}
    )
    latent_regime_hmm = _read_json(
        root / "output" / "runs" / "latent_regime_hmm.json", {}
    )
    neural_baseline = _read_json(root / "output" / "runs" / "neural_baseline.json", {})
    learned_stay_rate = _read_json(
        root / "output" / "runs" / "learned_stay_rate.json", {}
    )
    heldout_reporting = _read_json(
        root / "output" / "runs" / "heldout_reporting_process.json", {}
    )
    cross_generator = _read_json(
        root / "output" / "runs" / "cross_generator_transfer.json", {}
    )
    factorial_pymdp = _read_json(
        root / "output" / "runs" / "factorial_pymdp_analysis_fixture.json", {}
    )
    factorial_metrics = (
        factorial_pymdp.get("metrics", {})
        if isinstance(factorial_pymdp.get("metrics"), dict)
        else {}
    )
    pymdp_benchmark = _read_json(
        root / "output" / "runs" / "pymdp_benchmark_matrix.json", {}
    )
    efe_location = _read_json(
        root / "output" / "runs" / "efe_location_frontier.json", {}
    )
    linkage_risk = _read_json(
        root / "output" / "runs" / "linkage_risk_frontier.json", {}
    )
    poisoning_robustness = _read_json(
        root / "output" / "runs" / "poisoning_robustness.json", {}
    )
    sota_gate = _read_json(root / "output" / "runs" / "sota_claim_gate.json", {})
    sota_matrix = _read_json(
        root / "output" / "runs" / "sota_benchmark_matrix.json", {}
    )
    sota_oracle = (
        sota_matrix.get("false_claim_oracle", {})
        if isinstance(sota_matrix.get("false_claim_oracle"), dict)
        else {}
    )
    aif_sweep = _read_json(root / "output" / "runs" / "aif_sweep.json", {})
    aif_fair = (
        aif_sweep.get("fair_comparison", {})
        if isinstance(aif_sweep.get("fair_comparison"), dict)
        else {}
    )
    fair_inf = _read_json(root / "output" / "runs" / "fair_inference.json", {})
    fair_ladder = (
        fair_inf.get("ladder", {}) if isinstance(fair_inf.get("ladder"), dict) else {}
    )
    fair_losses = (
        fair_ladder.get("losses", {})
        if isinstance(fair_ladder.get("losses"), dict)
        else {}
    )
    fair_decomp = (
        fair_ladder.get("decomposition_nats", {})
        if isinstance(fair_ladder.get("decomposition_nats"), dict)
        else {}
    )
    fair_noise = (
        fair_inf.get("noise_sweep", {})
        if isinstance(fair_inf.get("noise_sweep"), dict)
        else {}
    )
    fair_sota = (
        fair_inf.get("sota_robustness", {})
        if isinstance(fair_inf.get("sota_robustness"), dict)
        else {}
    )
    power = _read_json(root / "output" / "runs" / "power_analysis.json", {})
    privacy_rows = {
        r["modalities"]: r
        for r in _read_json(
            root / "output" / "runs" / "privacy_frontier_analysis.json", {}
        ).get("frontier", [])
    }
    hardened_anomaly = _read_json(
        root / "output" / "runs" / "hardened_anomaly_analysis.json", {}
    )
    visual_quality = _read_json(
        root / "output" / "runs" / "visual_quality_audit.json", {}
    )
    metadata = _read_json(root / "output" / "runs" / "default" / "metadata.json", {})
    candidate_scores = _read_json(
        root / "output" / "runs" / "default" / "candidate_scores.json", []
    )
    tensor_summary = _read_json(
        root / "output" / "runs" / "default" / "tensor_summary.json", {}
    )
    registry = _read_json(
        root / "output" / "figures" / "figure_registry.json", {"entries": []}
    )
    config = load_config(root / "configs" / "default.yaml")
    best_candidate = candidate_scores[0] if candidate_scores else {}
    entries = {entry["key"]: entry for entry in registry.get("entries", [])}
    feature_path = root / "output" / "runs" / "default" / "discrete_features.csv"
    feature_rows = _csv_rows(feature_path)
    artifact_count = len(
        [
            path
            for path in (root / "output" / "runs" / "default").glob("*")
            if path.is_file()
        ]
    )
    control_count = len(config.inference.model.controls)
    mean_policy_entropy = float(metrics.get("mean_policy_entropy", 0.0))
    max_policy_entropy = math.log(control_count) if control_count > 0 else 0.0
    variables = {
        "PROJECT_TITLE": PAPER_TITLE,
        "FRACTALRABBIT_COMMIT": "9933449c4f4fe1b26b6ac7bfdeeac76583085df5",
        "PYMDP_VERSION": "1.0.4",
        "FRACTALRABBIT_PARAMETER_COUNT": "7",
        "FRACTALRABBIT_OUTPUT_COLUMNS": "`ID`, `Days`, `x(km)`, `y(km)`",
        "OBSERVATION_COUNT": _fmt(metrics.get("n_observations", 0)),
        "TRAVELLER_COUNT": _fmt(metrics.get("n_travellers", 0)),
        "LATENT_STATE_COUNT": _fmt(metrics.get("n_latent_states", 0)),
        "MEAN_PREDICTIVE_LOSS": _fmt(metrics.get("mean_predictive_loss", 0.0)),
        "MEAN_POSTERIOR_ENTROPY": _fmt(metrics.get("mean_posterior_entropy", 0.0)),
        "FINAL_POSTERIOR_ENTROPY": _fmt(metrics.get("final_posterior_entropy", 0.0)),
        "MEAN_POLICY_ENTROPY": _fmt(metrics.get("mean_policy_entropy", 0.0)),
        "POLICY_PROXY_EXPLORATION_SELECTED_FRACTION": _fmt(
            metrics.get("policy_exploration_selected_fraction", 0.0)
        ),
        "MAX_POLICY_ENTROPY": _fmt(max_policy_entropy),
        "POLICY_TIE_DIAGNOSTIC": _policy_tie_diagnostic(
            mean_policy_entropy, max_policy_entropy
        ),
        "BASELINE_CONTROL": str(
            baseline_metrics.get("baseline_control", "empirical_continuation")
        ),
        "BASELINE_MEAN_PREDICTIVE_LOSS": _fmt(
            baseline_metrics.get("baseline_mean_predictive_loss", 0.0)
        ),
        "BASELINE_MEAN_POSTERIOR_ENTROPY": _fmt(
            baseline_metrics.get("baseline_mean_posterior_entropy", 0.0)
        ),
        "BASELINE_DELTA_MEAN_PREDICTIVE_LOSS": _fmt(
            metrics.get("baseline_delta_mean_predictive_loss", 0.0)
        ),
        "BASELINE_DELTA_MEAN_POSTERIOR_ENTROPY": _fmt(
            metrics.get("baseline_delta_mean_posterior_entropy", 0.0)
        ),
        "SELECTED_NEXT_CANDIDATE": str(metrics.get("selected_next_candidate", "none")),
        "BEST_CANDIDATE_SCORE": _fmt6(best_candidate.get("score", 0.0)),
        "BEST_CANDIDATE_REPORTING_RATIO": _fmt6(
            best_candidate.get("reporting_ratio", 0.0)
        ),
        "CANDIDATE_SCORE_FORMULA": _candidate_formula(best_candidate),
        "SPATIAL_DISCRETIZER": config.inference.observation.spatial_discretizer,
        "SPATIAL_BIN_COUNT": str(config.inference.observation.spatial_bins),
        "TIME_BIN_COUNT": str(config.inference.observation.time_bins_per_day),
        "SPEED_THRESHOLDS": ", ".join(
            _fmt(item)
            for item in config.inference.observation.speed_thresholds_km_per_day
        ),
        "BURST_RULE": config.inference.observation.burst_threshold_rule,
        "CONTROL_COUNT": str(len(config.inference.model.controls)),
        "CONTROL_NAMES": ", ".join(
            f"`{item}`" for item in config.inference.model.controls
        ),
        "OBSERVATION_MODALITIES": ", ".join(
            f"`{item}`" for item in metadata.get("observation_modalities", [])
        ),
        "FIRST_OBSERVATION_COUNT": str(
            sum(1 for row in feature_rows if row.get("is_first_observation") == "1")
        ),
        "FIRST_OBSERVATION_HANDLING": "zero interval, zero speed, and non-burst sentinel",
        "TENSOR_A_SHAPES": _shape_list(tensor_summary.get("A_shapes", [])),
        "TENSOR_B_SHAPE": _shape(tensor_summary.get("B_shape", [])),
        "FIGURE_COUNT": str(len(entries)),
        "RUN_ARTIFACT_COUNT": str(artifact_count),
        "DISCRETE_FEATURE_COLUMNS": _csv_header(feature_path),
        "CANDIDATE_SCORE_TABLE": _candidate_table(candidate_scores),
        "SENSITIVITY_RUN_COUNT": str(sensitivity.get("n_runs", 0)),
        "SENSITIVITY_CAVEAT": str(
            sensitivity.get("caveat", "fixture sensitivity summary only")
        ),
        "SENSITIVITY_POLICY_ENTROPY_NOTE": _policy_entropy_context(sensitivity),
        "SENSITIVITY_SUMMARY_TABLE": _sensitivity_table(sensitivity),
        "ARTIFACT_PROVENANCE_TABLE": _figure_table(registry.get("entries", [])),
        "HOLDOUT_FRACTION": _fmt(metrics.get("holdout_fraction_configured", 0.0)),
        "HOLDOUT_COUNT": _fmt(metrics.get("holdout_n_holdout", 0.0)),
        "HOLDOUT_TRAIN_COUNT": _fmt(metrics.get("holdout_n_train", 0.0)),
        "HOLDOUT_MEAN_PREDICTIVE_LOSS": _fmt(
            metrics.get("holdout_mean_predictive_loss", 0.0)
        ),
        "HOLDOUT_MEAN_POSTERIOR_ENTROPY": _fmt(
            metrics.get("holdout_mean_posterior_entropy", 0.0)
        ),
        "NEGATIVE_CONTROL_COUNT": str(negative.get("n_controls", 0)),
        "NEGATIVE_CONTROL_PASSED_COUNT": str(
            negative.get(
                "n_passed",
                sum(1 for item in negative.get("controls", []) if item.get("passed")),
            )
        ),
        "NEGATIVE_CONTROL_TABLE": _negative_control_table(negative.get("controls", [])),
        "NEGATIVE_CONTROL_CAVEAT": str(
            negative.get("caveat", "negative-control summary not yet generated")
        ),
        "MODEL_ENGINE": config.inference.model.engine,
        "MODEL_MODALITIES": ", ".join(
            f"`{item}`" for item in config.inference.model.modalities
        ),
        "PYMDP_MEAN_PREDICTIVE_LOSS": _fmt(
            metrics.get("pymdp_mean_predictive_loss", 0.0)
        ),
        "PYMDP_MEAN_POSTERIOR_ENTROPY": _fmt(
            metrics.get("pymdp_mean_posterior_entropy", 0.0)
        ),
        "PYMDP_MEAN_POLICY_ENTROPY": _fmt(
            metrics.get("pymdp_mean_policy_entropy", 0.0)
        ),
        "PYMDP_EXACT_INFERENCE_DELTA": _fmt_sci(
            metrics.get("pymdp_exact_inference_max_abs_delta")
        ),
        "PYMDP_LOOP_POSTERIOR_DELTA": _fmt(
            metrics.get("pymdp_reference_max_abs_posterior_delta", 0.0)
        ),
        "PYMDP_MEAN_VFE": _fmt(metrics.get("pymdp_mean_vfe", 0.0)),
        "PYMDP_TOTAL_RUNTIME_S": _fmt(metrics.get("pymdp_total_runtime_s", 0.0)),
        "PYMDP_MEDIAN_STEP_MS": _fmt(metrics.get("pymdp_median_step_runtime_ms", 0.0)),
        "PYMDP_MAX_STEP_MS": _fmt(metrics.get("pymdp_max_step_runtime_ms", 0.0)),
        "REFERENCE_RUNTIME_S": _fmt(metrics.get("reference_runtime_s", 0.0)),
        "RECOVERY_REFERENCE_MASS": _fmt(
            metrics.get("state_recovery_reference_mean_mass_on_true_cell", 0.0)
        ),
        "RECOVERY_REFERENCE_TOP1": _fmt(
            metrics.get("state_recovery_reference_top1_accuracy", 0.0)
        ),
        "RECOVERY_PYMDP_MASS": _fmt(
            metrics.get("state_recovery_pymdp_mean_mass_on_true_cell", 0.0)
        ),
        "RECOVERY_PYMDP_TOP1": _fmt(
            metrics.get("state_recovery_pymdp_top1_accuracy", 0.0)
        ),
        "CHANCE_FLOOR_JOINT_LOSS": _fmt(metrics.get("chance_floor_joint_loss", 0.0)),
        "REFERENCE_SKILL_SCORE": _fmt(metrics.get("reference_skill_score", 0.0)),
        "PYMDP_SKILL_SCORE": _fmt(metrics.get("pymdp_skill_score", 0.0)),
        "REFERENCE_BRIER": _fmt(metrics.get("reference_calibration_brier", 0.0)),
        "PYMDP_BRIER": _fmt(metrics.get("pymdp_calibration_brier", 0.0)),
        "REFERENCE_BRIER_SKILL_VS_BASE_RATE": _fmt(
            metrics.get("reference_brier_skill_vs_base_rate", 0.0)
        ),
        "PYMDP_BRIER_SKILL_VS_BASE_RATE": _fmt(
            metrics.get("pymdp_brier_skill_vs_base_rate", 0.0)
        ),
        "DEEP_LATENT_BRIER_SKILL_VS_BASE_RATE": _fmt(
            deep_latent_metrics.get("reference_brier_skill_vs_base_rate", 0.0)
        ),
        "REFERENCE_BRIER_SKILL_VS_PERSISTENCE": _fmt(
            metrics.get("reference_brier_skill_vs_persistence", 0.0)
        ),
        "DEEP_LATENT_BRIER_SKILL_VS_PERSISTENCE": _fmt(
            deep_latent_metrics.get("reference_brier_skill_vs_persistence", 0.0)
        ),
        "GAP_FILL_N_STEPS": str(anticipatory.get("n_gap_steps", 0)),
        "GAP_FILL_PERSISTENCE_BRIER": _fmt(
            anticipatory_gap.get("persistence_brier", 0.0)
        ),
        "GAP_FILL_FLAT_SKILL": _fmt(
            anticipatory_gap.get("flat_skill_vs_persistence", 0.0)
        ),
        "GAP_FILL_SCRP_SKILL": _fmt(
            anticipatory_gap.get("scrp_skill_vs_persistence", 0.0)
        ),
        "GAP_FILL_SPEED_EDGE": _fmt(anticipatory_gap.get("speed_clamp_edge", 0.0)),
        "BLIND_LIMIT_FLAT_SKILL": _fmt(
            anticipatory_blind.get("flat_skill_vs_persistence", 0.0)
        ),
        "BLIND_LIMIT_SCRP_SKILL": _fmt(
            anticipatory_blind.get("scrp_skill_vs_persistence", 0.0)
        ),
        "PARTIAL_OBS_N_CELLS": _fmt(partial_obs.get("n_cells", 0)),
        "PARTIAL_OBS_CLEAN_ADVANTAGE": _fmt(
            partial_obs.get("clean_advantage_nats", 0.0)
        ),
        "PARTIAL_OBS_MAX_EPS": _fmt(partial_obs.get("max_eps", 0.0)),
        "PARTIAL_OBS_MAX_NOISE_ADVANTAGE": _fmt(
            partial_obs.get("max_noise_advantage_nats", 0.0)
        ),
        "PARTIAL_OBS_MAX_NOISE_SD": _fmt(
            partial_obs.get("max_noise_advantage_sd", 0.0)
        ),
        "PARTIAL_OBS_MAX_NOISE_MIN": _fmt(
            partial_obs.get("max_noise_advantage_min", 0.0)
        ),
        "PARTIAL_OBS_N_REALIZATIONS": _fmt(
            partial_obs.get("n_realizations_per_level", 0)
        ),
        "PARTIAL_OBS_NOISE_SLOPE": _fmt(
            partial_obs.get("noise_slope_nats_per_eps", 0.0)
        ),
        "PARTIAL_OBS_NOISE_R2": _fmt(partial_obs.get("noise_trend_r2", 0.0)),
        "PARTIAL_OBS_ZERO_WHEN_CLEAN": "yes"
        if partial_obs.get("zero_advantage_when_clean", False)
        else "no",
        "PARTIAL_OBS_SMOOTHING": _fmt(partial_obs.get("smoothing", 0.0)),
        "PARTIAL_OBS_MONOTONE": "yes"
        if partial_obs.get("advantage_monotone_with_noise", False)
        else "no",
        "PARTIAL_OBS_CROSS_DRAW_COUNT": _fmt(
            partial_obs.get("cross_realization_n_draws", 0)
        ),
        "PARTIAL_OBS_CROSS_DRAW_TRAVELLERS": _fmt(
            partial_obs.get("cross_realization_n_travellers", 0)
        ),
        "PARTIAL_OBS_CROSS_DRAW_MAX_SD": _fmt(
            partial_obs.get("cross_realization_max_between_draw_sd", 0.0)
        ),
        "PARTIAL_OBS_CROSS_DRAW_MEAN_ADVANTAGE": _fmt(
            partial_obs_cross_max_eps_row.get("mean_advantage_nats", 0.0)
        ),
        "PARTIAL_OBS_CROSS_DRAW_MIN_ADVANTAGE": _fmt(
            partial_obs_cross_max_eps_row.get("between_draw_min_nats", 0.0)
        ),
        "PARTIAL_OBS_CROSS_DRAW_MAX_ADVANTAGE": _fmt(
            partial_obs_cross_max_eps_row.get("between_draw_max_nats", 0.0)
        ),
        "PARTIAL_OBS_CROSS_DRAW_SIGN_STABLE": (
            "yes"
            if partial_obs.get("cross_realization_sign_stable_under_noise", False)
            else "no"
        ),
        "SOTA_PROGRAM_STATUS": str(sota_program.get("status", "unavailable")),
        "SOTA_PROGRAM_GATE_CLEARS": "yes"
        if sota_program.get("gate_clears", False)
        else "no",
        "SOTA_PROGRAM_LANE_COUNT": str(len(sota_program_canonical_lanes)),
        "SOTA_PROGRAM_CANONICAL_LANES": _lane_list(sota_program_canonical_lanes),
        "SOTA_PROGRAM_REPORTING_GAP_SCORED": (
            "yes"
            if sota_program_robustness.get("reporting_gap_scored", False)
            else "no"
        ),
        "SOTA_PROGRAM_REGIME_STRESSOR_SCORED": (
            "yes"
            if sota_program_robustness.get("regime_stressor_scored", False)
            else "no"
        ),
        "SOTA_PROGRAM_ROBUSTNESS_BOUNDARY": str(
            sota_program_robustness.get(
                "claim_boundary",
                "Synthetic robustness lanes cannot unlock empirical-transfer or operational claims.",
            )
        ),
        "SOTA_PROGRAM_N_TRAVELLERS": _fmt(sota_fr.get("n_travellers", 0)),
        "SOTA_PROGRAM_N_HOLDOUT": _fmt(sota_fr.get("n_holdout_travellers", 0)),
        "SOTA_PROGRAM_FR_AIF_LOSS": _fmt(sota_fr.get("aif_loss", 0.0)),
        "SOTA_PROGRAM_FR_STRONGEST": str(
            sota_fr.get("strongest_fair_baseline", "unavailable")
        ),
        "SOTA_PROGRAM_FR_STRONGEST_LOSS": _fmt(sota_fr.get("strongest_fair_loss", 0.0)),
        "SOTA_PROGRAM_FR_CI_LOW": _fmt(sota_fr.get("clustered_ci_low", 0.0)),
        "SOTA_PROGRAM_FR_CI_HIGH": _fmt(sota_fr.get("clustered_ci_high", 0.0)),
        "SOTA_PROGRAM_GR_AIF_LOSS": _fmt(sota_gr.get("aif_loss", 0.0)),
        "SOTA_PROGRAM_GR_STRONGEST": str(
            sota_gr.get("strongest_fair_baseline", "unavailable")
        ),
        "SOTA_PROGRAM_GR_STRONGEST_LOSS": _fmt(sota_gr.get("strongest_fair_loss", 0.0)),
        "AIF_REGIME_SOTA_STATUS": str(aif_regime_gate.get("status", "unavailable")),
        "AIF_REGIME_SOTA_GATE_CLEARS": "yes"
        if aif_regime_gate.get("gate_clears", False)
        else "no",
        "AIF_REGIME_SOTA_CLEARED_COUNT": str(
            aif_regime_gate.get("cleared_regime_count", 0)
        ),
        "AIF_REGIME_SOTA_BEST_REGIME": str(
            aif_regime_gate.get("best_cleared_regime_id", "none")
        ),
        "AIF_REGIME_SOTA_BEST_CANDIDATES": _lane_list(
            aif_regime_gate.get("best_aif_candidates", [])
        ),
        "AIF_REGIME_SOTA_STRONGEST_FAIR": _lane_list(
            aif_regime_gate.get("best_strongest_fair_baselines", [])
        ),
        "AIF_REGIME_SOTA_MIN_ADVANTAGE": _fmt_nullable(
            aif_regime_gate.get("best_min_advantage_nats")
        ),
        "AIF_REGIME_SOTA_MIN_CI_LOW": _fmt_nullable(
            aif_regime_gate.get("best_min_clustered_ci_low")
        ),
        "AIF_REGIME_SOTA_BOUNDARY": str(
            aif_regime_gate.get(
                "global_sota_boundary",
                "Regime-qualified synthetic AIF evidence does not clear global, external, empirical, or operational SOTA.",
            )
        ),
        "MULTIPLICITY_FAMILY_SIZE": _fmt(multiplicity.get("family_size", 0)),
        "MULTIPLICITY_UNCORRECTED_CLEARS": _fmt(
            multiplicity.get("n_uncorrected_significant", 0)
        ),
        "MULTIPLICITY_HOLM_SURVIVORS": _fmt(
            multiplicity.get("holm", {}).get("n_survivors", 0)
        ),
        "MULTIPLICITY_BH_SURVIVORS": _fmt(
            multiplicity.get("benjamini_hochberg", {}).get("n_survivors", 0)
        ),
        "MULTIPLICITY_ALPHA": _fmt(multiplicity.get("alpha", 0.0)),
        "MULTIPLICITY_FDR_Q": _fmt(multiplicity.get("fdr_q", 0.0)),
        "MULTIPLICITY_N_BOOT": _fmt_nullable(
            multiplicity.get("headline", {}).get("min_n_boot")
        ),
        "MULTIPLICITY_P_FLOOR": _fmt_sci_optional(
            multiplicity.get("headline", {}), "max_p_resolution_floor"
        ),
        "MULTIPLICITY_MIN_ADVANTAGE_NATS": _fmt_nullable(
            multiplicity.get("headline", {}).get("family_min_advantage_nats")
        ),
        "MULTIPLICITY_MAX_ADVANTAGE_NATS": _fmt_nullable(
            multiplicity.get("headline", {}).get("family_max_advantage_nats")
        ),
        "SOTA_LANE_HOLDOUT_TRAVELLERS": _fmt_nullable(
            multiplicity.get("headline", {}).get("min_lane_holdout_travellers")
        ),
        "PARTIAL_OBS_SIGN_TEST_BOUND": _fmt_sci_optional(
            multiplicity.get("partial_observability_positive", {}), "sign_test_bound"
        ),
        "MULTIPLICITY_BONFERRONI_THRESHOLD": _fmt_sci_optional(
            multiplicity.get("partial_observability_positive", {}),
            "bonferroni_per_comparison_threshold",
        ),
        "AIF_REGIME_N_BOOT": _fmt_nullable(aif_regime_run_parameters.get("n_boot")),
        "AIF_REGIME_N_REALIZATIONS": _fmt_nullable(
            aif_regime_run_parameters.get("n_realizations")
        ),
        "AIF_REGIME_TRAVELLER_COUNT": _fmt_nullable(
            aif_regime_run_parameters.get("n_travellers")
        ),
        "SOTA_LANE_N_BOOT": _fmt_nullable(sota_gate_run_parameters.get("n_boot")),
        "PARTIAL_OBS_MULTIPLICITY_DRAWS": _fmt(
            multiplicity.get("partial_observability_positive", {}).get(
                "n_independent_draws", 0
            )
        ),
        "PARTIAL_OBS_MIN_DRAWS_BONFERRONI": _fmt(
            multiplicity.get("partial_observability_positive", {}).get(
                "min_draws_for_family_bonferroni", 0
            )
        ),
        "CONFOUND_N_DISSOLVED": _fmt(confound_taxonomy.get("n_confounds", 0)),
        "CONFOUND_N_CLASSES": _fmt(
            confound_taxonomy.get("n_confound_classes_present", 0)
        ),
        "CONFOUND_CROSS_VENDOR_CAUGHT": _fmt(
            confound_taxonomy.get("n_cross_vendor_caught", 0)
        ),
        "CONFOUND_N_VERIFIED": _fmt(confound_taxonomy.get("n_verified", 0)),
        "CONFOUND_N_ENTRIES": _fmt(confound_taxonomy.get("n_entries", 0)),
        "REGIME_MECHANISM_N_REGIMES": _fmt(regime_mechanism.get("n_regimes", 0)),
        "REGIME_MECHANISM_N_ELIGIBLE": _fmt(
            sum(
                1
                for row in regime_mechanism.get("regimes", [])
                if isinstance(row, dict) and row.get("eligible_for_regime_sota")
            )
        ),
        "REGIME_MECHANISM_N_FAILS_BATTERY": _fmt(
            sum(
                1
                for row in regime_mechanism.get("regimes", [])
                if isinstance(row, dict)
                and row.get("eligible_for_regime_sota")
                and not row.get("confound_battery", {}).get("clears_confounds", False)
            )
        ),
        "REGIME_MECHANISM_N_CLEARS_BATTERY": _fmt(
            sum(
                1
                for row in regime_mechanism.get("regimes", [])
                if isinstance(row, dict)
                and row.get("eligible_for_regime_sota")
                and row.get("confound_battery", {}).get("clears_confounds", False)
            )
        ),
        "REGIME_MECHANISM_N_NULL": _fmt(
            regime_mechanism.get("verdict_counts", {}).get("null_no_signal", 0)
        ),
        "REGIME_MECHANISM_N_LOSS": _fmt(
            regime_mechanism.get("verdict_counts", {}).get("loses_to_fair_baseline", 0)
        ),
        "REGIME_MECHANISM_N_ROBUST": _fmt(
            regime_mechanism.get("verdict_counts", {}).get("robust_win", 0)
        ),
        "PHASE_DIAGNOSTIC_WORST_SEGMENT": _fmt(
            regime_switch_phase.get("worst_segment", "unavailable")
        ),
        "PHASE_DIAGNOSTIC_BEST_SEGMENT": _fmt(
            regime_switch_phase.get("best_segment", "unavailable")
        ),
        "PHASE_DIAGNOSTIC_UNIFORM": _fmt(
            regime_switch_phase.get("uniform_disadvantage", False)
        ),
        "PHASE_DIAGNOSTIC_SEGMENT_COUNT": _fmt(
            len(regime_switch_phase.get("segments", []))
        ),
        "PHASE_DIAGNOSTIC_RATIO": (
            _fmt(float(regime_switch_phase["worst_to_best_magnitude_ratio"]))
            if isinstance(
                regime_switch_phase.get("worst_to_best_magnitude_ratio"), (int, float)
            )
            else "n/a"
        ),
        "PRIVACY_UTILITY_FINEST_EXCESS": _fmt(
            privacy_utility.get("finest_excess_utility_bits", 0.0)
        ),
        "PRIVACY_UTILITY_COARSEST_EXCESS": _fmt(
            privacy_utility.get("coarsest_excess_utility_bits", 0.0)
        ),
        "PRIVACY_UTILITY_FINEST_REID": _fmt(
            privacy_utility.get("finest_reid_event_uniqueness", 0.0)
        ),
        "PRIVACY_UTILITY_COARSEST_REID": _fmt(
            privacy_utility.get("coarsest_reid_event_uniqueness", 0.0)
        ),
        "MINCOLLECT_FOUND": _yes_no(
            privacy_utility.get("min_collection_operating_point", {}).get(
                "found", False
            )
        ),
        "MINCOLLECT_UTILITY_FLOOR": _fmt(
            privacy_utility.get("min_collection_operating_point", {}).get(
                "utility_floor_bits", 0.0
            )
        ),
        "MINCOLLECT_RISK_CEILING": _fmt(
            privacy_utility.get("min_collection_operating_point", {}).get(
                "reid_risk_ceiling", 0.0
            )
        ),
        "MINCOLLECT_REID_FLOOR": _fmt(
            privacy_utility.get("min_collection_operating_point", {}).get(
                "min_reid_among_useful_grids", 0.0
            )
        ),
        "MINCOLLECT_REGIME": str(
            privacy_utility.get("min_collection_operating_point", {}).get(
                "regime", "unavailable"
            )
        ),
        "MINCOLLECT_EXCESS_UTILITY": _fmt(
            privacy_utility.get("min_collection_operating_point", {}).get(
                "excess_utility_bits", 0.0
            )
        ),
        "MINCOLLECT_REID": _fmt(
            privacy_utility.get("min_collection_operating_point", {}).get(
                "reid_event_uniqueness", 0.0
            )
        ),
        "PRIVACY_UTILITY_REPEATS_COMPLETED": _fmt(
            privacy_utility_repeats.get("completed_count", 0)
        ),
        "PRIVACY_UTILITY_REPEATS_UNAVAILABLE": _fmt(
            privacy_utility_repeats.get("unavailable_count", 0)
        ),
        "TASK_PRIVACY_RELEASE_UTILITY_ALLOWED": _yes_no(
            synthetic_task_privacy.get("any_release_utility_claim_allowed", False)
        ),
        "TASK_PRIVACY_JOINT_OPERATING_POINTS": _fmt(
            synthetic_task_privacy.get("joint_operating_point_count", 0)
        ),
        "AIF_DEEPENING_PREDICTIVE_GATE_CLEARS": _yes_no(
            aif_deepening.get("predictive_gate_clears", False)
        ),
        "AIF_DEEPENING_PROTOCOL_CANDIDATES": _fmt(
            aif_deepening.get("scored_protocol_candidate_count", 0)
        ),
        "ADAPTIVE_AIF_REALIZATIONS": _fmt(
            adaptive_aif.get("realization_count", 0)
        ),
        "ADAPTIVE_AIF_MEAN_GAIN": _fmt(
            adaptive_aif.get("adaptive_improvement_nats", 0.0)
        ),
        "ADAPTIVE_AIF_GAIN_LOW": _fmt(
            (adaptive_aif.get("adaptive_improvement_range_nats") or [0.0, 0.0])[0]
        ),
        "ADAPTIVE_AIF_GAIN_HIGH": _fmt(
            (adaptive_aif.get("adaptive_improvement_range_nats") or [0.0, 0.0])[1]
        ),
        "ADAPTIVE_AIF_FIXED_CI_CLEARS": _fmt(
            adaptive_aif.get("adaptive_beats_fixed_ci_count", 0)
        ),
        "ADAPTIVE_AIF_FAIR_CI_CLEARS": _fmt(
            adaptive_aif.get("adaptive_beats_online_base_rate_ci_count", 0)
        ),
        "ADAPTIVE_AIF_GATE_CLEARS": _yes_no(
            adaptive_aif.get("predictive_gate_clears", False)
        ),
        "MODEL_COMPARISON_AIF_RANK": _fmt(model_comparison.get("aif_rank", 0)),
        "MODEL_COMPARISON_CHANCE_LOSS": _fmt_nullable(
            model_comparison.get("chance_loss_nats")
        ),
        "MODEL_COMPARISON_N_METHODS": _fmt(model_comparison.get("n_methods", 0)),
        "MODEL_COMPARISON_AIF_LOSS": _fmt(model_comparison.get("aif_loss", 0.0)),
        "MODEL_COMPARISON_STRONGEST_NON_AIF": str(
            model_comparison.get("strongest_non_aif_method", "unavailable")
        ),
        "MODEL_COMPARISON_AIF_VS_NONAIF_GAP": _fmt(
            model_comparison.get("aif_vs_strongest_non_aif_gap", 0.0)
        ),
        "MODEL_COMPARISON_BELIEF_NONBELIEF_GAP": _fmt(
            model_comparison.get("belief_vs_nonbelief_gap", 0.0)
        ),
        "MODEL_COMPARISON_N_BELIEF": _fmt(model_comparison.get("n_belief_methods", 0)),
        "EFE_SHARE_UNIFORM": _fmt(
            efe_sweep[0]["mean_epistemic_share"] if efe_sweep else 0.0
        ),
        "EFE_SHARE_SHARP": _fmt(
            efe_sweep[-1]["mean_epistemic_share"] if efe_sweep else 0.0
        ),
        "EFE_SHARP_PREFERENCE": _fmt(
            efe_sweep[-1]["preference_strength"] if efe_sweep else 0.0
        ),
        "ACTIVE_SENSING_BEATEN_FRACTION": (
            _fmt(efe_lane["budgets"][0]["fraction_random_beaten"])
            if efe_lane.get("budgets")
            and "fraction_random_beaten" in efe_lane["budgets"][0]
            else "unavailable"
        ),
        "ANOMALY_VFE_AUC_RARE": _fmt(anomaly.get("vfe_auc_rare", 0.0)),
        "ANOMALY_VFE_AUC_RANDOM": _fmt(anomaly.get("vfe_auc_random", 0.0)),
        "ANOMALY_NOVELTY_AUC": _fmt(anomaly.get("novelty_auc", 0.0)),
        "STRUCTURE_DEEP_LATENT_TV": _fmt(structure_ext.get("tv_deep_latent", 0.0)),
        "STRUCTURE_DYNAMICS_ONLY_TV": _fmt(structure_ext.get("tv_dynamics_only", 0.0)),
        "STRUCTURE_PREFERRED_JACCARD": _fmt(
            structure_ext.get("preferred_jaccard", 0.0)
        ),
        "STRUCTURE_CROSS_DRAW_COUNT": _fmt(structure_cross.get("n_draws", 0)),
        "STRUCTURE_CROSS_DRAW_TRAVELLERS": _fmt(structure_cross.get("n_travellers", 0)),
        "STRUCTURE_CROSS_BEATS_DYNAMICS_FRACTION": _fmt(
            structure_cross.get("beats_dynamics_only_fraction", 0.0)
        ),
        "STRUCTURE_CROSS_DEEP_LATENT_TV_MEAN": _fmt(
            structure_cross.get("tv_deep_latent", {}).get("mean", 0.0)
        ),
        "STRUCTURE_CROSS_DYNAMICS_ONLY_TV_MEAN": _fmt(
            structure_cross.get("tv_dynamics_only", {}).get("mean", 0.0)
        ),
        "MISSION_EFE_RESOLVED": _fmt(mission.get("resolved", {}).get("efe", 0.0)),
        "MISSION_PRAGMATIC_RESOLVED": _fmt(
            mission.get("resolved", {}).get("pragmatic", 0.0)
        ),
        "FRONTIER_ALL_MASS": _fmt(
            frontier_rows.get("all", {}).get("mean_mass_on_true_cell", 0.0)
        ),
        "FRONTIER_NO_LOCATION_MASS": _fmt(
            frontier_rows.get("no_location", {}).get("mean_mass_on_true_cell", 0.0)
        ),
        "RETURN_MOVE_ENTROPY_REDUCTION": _fmt(
            return_structure.get("move_step_entropy_reduction", 0.0)
        ),
        "RETURN_PERSISTENCE_HEADROOM": _fmt(
            return_structure.get("persistence_headroom", 0.0)
        ),
        "RETURN_STRUCTURE_VERDICT": str(return_structure.get("verdict", "unavailable")),
        "VARIANCE_REALIZATION_COUNT": str(variance_bands.get("n_realizations", 0)),
        "VARIANCE_GAP_FILL_LOW": _variance_metric(
            variance_bands, "gap_filling_skill", "low"
        ),
        "VARIANCE_GAP_FILL_HIGH": _variance_metric(
            variance_bands, "gap_filling_skill", "high"
        ),
        "VARIANCE_GAP_FILL_SD": _variance_metric(
            variance_bands, "gap_filling_skill", "sd"
        ),
        "VARIANCE_GAP_FILL_IQR_LOW": _variance_metric(
            variance_bands, "gap_filling_skill", "q25"
        ),
        "VARIANCE_GAP_FILL_IQR_HIGH": _variance_metric(
            variance_bands, "gap_filling_skill", "q75"
        ),
        "UPSTREAM_SENSITIVITY_STATUS": str(
            upstream_sensitivity.get("status", "unavailable")
        ),
        "UPSTREAM_SENSITIVITY_ATTEMPTS": str(
            len(upstream_sensitivity.get("attempted_commits", []))
        ),
        "FRACTAL_DIMENSION_SWEEP_STATUS": str(
            fractal_dimension.get("status", "unavailable")
        ),
        "FRACTAL_DIMENSION_BEST_LOSS_D": _fmt_nullable(
            fractal_dimension.get("best_predictive_loss_dimension")
        ),
        "FRACTAL_DIMENSION_REVISIT_D": _fmt_nullable(
            fractal_dimension.get("highest_revisit_dimension")
        ),
        "FRACTAL_DIMENSION_LOSS_LOW": _metric_interval_value(
            fractal_dimension, "mean_predictive_loss", "low"
        ),
        "FRACTAL_DIMENSION_LOSS_HIGH": _metric_interval_value(
            fractal_dimension, "mean_predictive_loss", "high"
        ),
        "CIRCADIAN_PHASE_VERDICT": str(circadian_phase.get("verdict", "unavailable")),
        "CIRCADIAN_PHASE_GAIN": _fmt_nullable(
            circadian_phase.get("phase_gain_vs_pooled")
        ),
        "CIRCADIAN_PHASE_PERMUTED_GAIN": _fmt_nullable(
            circadian_phase.get("permutation_gain_vs_pooled")
        ),
        "CIRCADIAN_PHASE_MAX_JS": _fmt_nullable(circadian_phase.get("max_phase_js")),
        "CIRCADIAN_PHASE_GAIN_P": _fmt_nullable(
            circadian_phase.get("phase_gain_p_value")
        ),
        "CIRCADIAN_PHASE_MAXJS_P": _fmt_nullable(
            circadian_phase.get("max_phase_js_p_value")
        ),
        "CIRCADIAN_PHASE_N_PERM": _fmt_nullable(circadian_phase.get("n_permutations")),
        "SEQUENCE_BASELINE_BEST_MODEL": str(
            sequence_baselines.get("best_model", "unavailable")
        ),
        "SEQUENCE_BASELINE_BEST_LOSS": _fmt_nullable(
            sequence_baselines.get("best_mean_predictive_loss")
        ),
        "SEQUENCE_BASELINE_SKILL": _fmt_nullable(
            sequence_baselines.get("best_skill_vs_base_rate")
        ),
        "SEQUENCE_BASELINE_REFERENCE_DELTA": _fmt_nullable(
            sequence_baselines.get("best_minus_reference_predictive_loss")
        ),
        "SEQUENCE_BASELINE_PYMDP_DELTA": _fmt_nullable(
            sequence_baselines.get("best_minus_pymdp_predictive_loss")
        ),
        "SEQUENCE_BASELINE_GATE": str(
            sequence_baselines.get("pymdp_performance_gate", "unavailable")
        ),
        "SEQUENCE_BASELINE_TABLE": _sequence_baseline_table(
            sequence_baselines.get("rows", [])
        ),
        "MISSINGNESS_VERDICT": str(missingness_model.get("verdict", "unavailable")),
        "MISSINGNESS_INTERVAL_COUNT": _fmt_nullable(
            missingness_model.get("n_intervals")
        ),
        "MISSINGNESS_SHORT_GAP_RATE": _fmt_nullable(
            missingness_model.get("short_gap_rate")
        ),
        "MISSINGNESS_PHASE_GAIN": _fmt_nullable(
            missingness_model.get("phase_gain_vs_pooled")
        ),
        "MISSINGNESS_TRAVELLER_GAIN": _fmt_nullable(
            missingness_model.get("traveller_gain_vs_pooled")
        ),
        "MISSINGNESS_COMBINED_GAIN": _fmt_nullable(
            missingness_model.get("phase_traveller_gain_vs_pooled")
        ),
        "MISSINGNESS_PERMUTED_GAIN": _fmt_nullable(
            missingness_model.get("permuted_phase_gain_vs_pooled")
        ),
        "MISSINGNESS_INTERVAL_CV": _fmt_nullable(missingness_model.get("interval_cv")),
        "MISSINGNESS_TABLE": _missingness_table(missingness_model),
        "SPATIAL_RESOLUTION_VERDICT": str(
            spatial_resolution.get("verdict", "unavailable")
        ),
        "SPATIAL_RESOLUTION_DEFAULT_BINS": _fmt_nullable(
            spatial_resolution.get("default_spatial_bins")
        ),
        "SPATIAL_RESOLUTION_DEFAULT_STATES": _nested_fmt(
            spatial_resolution, ("default_row", "state_count")
        ),
        "SPATIAL_RESOLUTION_DEFAULT_EMPTY_STATES": _nested_fmt(
            spatial_resolution, ("default_row", "empty_state_count")
        ),
        "SPATIAL_RESOLUTION_DEFAULT_FULL_LOSS": _nested_fmt(
            spatial_resolution, ("default_row", "full_location_mean_predictive_loss")
        ),
        "SPATIAL_RESOLUTION_DEFAULT_NO_LOCATION_MASS": _nested_fmt(
            spatial_resolution, ("default_row", "no_location_mean_mass_on_true_cell")
        ),
        "SPATIAL_RESOLUTION_BEST_FEASIBLE_BINS": _nested_fmt(
            spatial_resolution, ("best_feasible_row", "spatial_bins")
        ),
        "SPATIAL_RESOLUTION_DEFAULT_MINUS_COARSE_LOSS": _nested_fmt(
            spatial_resolution,
            ("default_minus_coarse", "full_location_mean_predictive_loss"),
        ),
        "SPATIAL_RESOLUTION_DEFAULT_MINUS_COARSE_NO_LOCATION_MASS": _nested_fmt(
            spatial_resolution,
            ("default_minus_coarse", "no_location_mean_mass_on_true_cell"),
        ),
        "SPATIAL_RESOLUTION_PYMDP_DELTA": _fmt_sci(
            _nested_value(
                spatial_resolution,
                ("default_row", "pymdp_exact_inference_max_abs_delta"),
            )
        ),
        "SPATIAL_RESOLUTION_TABLE": _spatial_resolution_table(
            spatial_resolution.get("rows", [])
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_STATUS": _nested_fmt(
            spatial_resolution_confirmation, ("comparison", "status")
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_VERDICT": _nested_fmt(
            spatial_resolution_confirmation, ("comparison", "verdict")
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_N": _nested_fmt(
            spatial_resolution_confirmation, ("larger_sample", "n_observations")
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_REPORTS_PER_TRAVELLER": _nested_fmt(
            spatial_resolution_confirmation,
            ("larger_sample", "reports_per_traveller"),
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_BEST_FEASIBLE_BINS": _nested_fmt(
            spatial_resolution_confirmation, ("larger_sample", "best_feasible_bins")
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_STRICT_FEASIBLE_COUNT": _nested_fmt(
            spatial_resolution_confirmation,
            ("larger_sample", "strict_feasible_candidate_count"),
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_DEFAULT_MINUS_COARSE_LOSS": _nested_fmt(
            spatial_resolution_confirmation,
            ("larger_sample", "default_minus_coarse_loss"),
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_DEFAULT_MINUS_COARSE_NO_LOCATION_MASS": _nested_fmt(
            spatial_resolution_confirmation,
            ("larger_sample", "default_minus_coarse_no_location_mass"),
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_MULTIPLIERS": _int_list(
            _nested_value(
                spatial_resolution_confirmation,
                ("sample_size_ladder_summary", "multipliers"),
            )
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_LARGEST_MULTIPLIER": _nested_fmt(
            spatial_resolution_confirmation,
            ("sample_size_ladder_summary", "largest_multiplier"),
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_LARGEST_N": _nested_fmt(
            spatial_resolution_confirmation,
            ("sample_size_ladder_summary", "largest_n_observations"),
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_LARGEST_REPORTS_PER_TRAVELLER": _nested_fmt(
            spatial_resolution_confirmation,
            ("sample_size_ladder_summary", "largest_reports_per_traveller"),
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_LADDER_VERDICT_STABLE": _yes_no(
            _nested_value(
                spatial_resolution_confirmation,
                ("sample_size_ladder_summary", "all_same_verdict"),
            )
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_GUARDRAIL_STABLE": _yes_no(
            _nested_value(
                spatial_resolution_confirmation,
                (
                    "sample_size_ladder_summary",
                    "all_same_no_location_guardrail_direction",
                ),
            )
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_STRICT_FEASIBLE_SERIES": _int_list(
            _nested_value(
                spatial_resolution_confirmation,
                ("sample_size_ladder_summary", "strict_feasible_counts"),
            )
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_SELECTED_BINS_SERIES": _int_list(
            _nested_value(
                spatial_resolution_confirmation,
                ("sample_size_ladder_summary", "selected_bins"),
            )
        ),
        "SPATIAL_RESOLUTION_CONFIRMATION_LADDER_INTERPRETATION": _nested_str(
            spatial_resolution_confirmation,
            ("sample_size_ladder_summary", "interpretation"),
        ),
        "RETURN_PHASE_VERDICT": str(return_phase_prior.get("verdict", "unavailable")),
        "RETURN_PHASE_BEST_MODEL": str(
            return_phase_prior.get("best_model", "unavailable")
        ),
        "RETURN_PHASE_BEST_LOSS": _fmt_nullable(
            return_phase_prior.get("best_mean_predictive_loss")
        ),
        "RETURN_PHASE_GAIN_VS_PERSISTENCE": _fmt_nullable(
            return_phase_prior.get("return_plus_phase_gain_vs_persistence")
        ),
        "RETURN_PHASE_GAIN_OVER_PERMUTED": _fmt_nullable(
            return_phase_prior.get("return_plus_phase_gain_over_permuted")
        ),
        "RETURN_PHASE_TABLE": _return_phase_table(return_phase_prior.get("rows", [])),
        "MODEL_GATE_MATRIX_STATUS": str(model_gate_matrix.get("status", "unavailable")),
        "MODEL_GATE_SEQUENCE_EXTERNAL_GATE": str(
            model_gate_matrix.get("external_sequence_gate", "unavailable")
        ),
        "MODEL_GATE_MISSINGNESS_EXTERNAL_VERDICT": str(
            model_gate_matrix.get("external_missingness_verdict", "unavailable")
        ),
        "MODEL_GATE_FACTORIAL_EXTERNAL_GATE": str(
            model_gate_matrix.get("external_factorial_gate", "unavailable")
        ),
        "FACTORIAL_PYMDP_STATUS": str(factorial_pymdp.get("status", "unavailable")),
        "FACTORIAL_PYMDP_FACTOR_COUNT": str(
            len(factorial_pymdp.get("hidden_factors", []) or [])
        ),
        "FACTORIAL_PYMDP_GATE": str(
            factorial_pymdp.get("performance_gate", "unavailable")
        ),
        "FACTORIAL_PYMDP_BEST_MODEL": str(
            factorial_pymdp.get("best_model", "unavailable")
        ),
        "FACTORIAL_PYMDP_BOUNDARY": str(
            factorial_pymdp.get("mode_boundary", "synthetic cognitive-proxy states")
        ),
        "FACTORIAL_PYMDP_MEAN_PREDICTIVE_LOSS": _fmt_nullable(
            factorial_metrics.get("factorial_pymdp_mean_predictive_loss")
        ),
        "FACTORIAL_PYMDP_SWITCHING_HMM_LOSS": _fmt_nullable(
            factorial_metrics.get("switching_hmm_mean_predictive_loss")
        ),
        "FACTORIAL_PYMDP_PARTICLE_LOSS": _fmt_nullable(
            factorial_metrics.get("particle_state_space_mean_predictive_loss")
        ),
        "FACTORIAL_PYMDP_MATCHED_PRIOR_DELTA": _fmt_nullable(
            factorial_metrics.get("factorial_pymdp_matched_prior_max_abs_delta")
        ),
        "FACTORIAL_PYMDP_MEAN_FIELD_EXACT_GAP": _fmt_nullable(
            factorial_metrics.get("factorial_pymdp_mean_field_exact_gap")
        ),
        "FACTORIAL_PYMDP_TABLE": _factorial_table(
            factorial_pymdp.get("model_rows", [])
        ),
        "MODEL_GATE_RETURN_PHASE_EXTERNAL_VERDICT": str(
            model_gate_matrix.get("external_return_phase_verdict", "unavailable")
        ),
        "MODEL_GATE_MATRIX_TABLE": _model_gate_matrix_table(
            model_gate_matrix.get("rows", [])
        ),
        "FORWARD_GATE_STATUS": str(forward_gates.get("status", "unavailable")),
        "FORWARD_GATE_FIXTURE_COMPLETED_COUNT": str(
            forward_gates.get("fixture_completed_count", 0)
        ),
        "FORWARD_GATE_TABLE": _forward_gate_table(forward_gates.get("rows", [])),
        "LATENT_REGIME_HMM_GATE": str(latent_regime_hmm.get("gate", "unavailable")),
        "LATENT_REGIME_HMM_BEST_HELDOUT_LOSS": _fmt_nullable(
            latent_regime_hmm.get("best_honest_heldout_loss")
        ),
        "LATENT_REGIME_HMM_OVERFIT_GAP": _fmt_nullable(
            latent_regime_hmm.get("max_overfit_gap")
        ),
        "NEURAL_BASELINE_GATE": str(neural_baseline.get("gate", "unavailable")),
        "NEURAL_BASELINE_LOSS": _fmt_nullable(
            neural_baseline.get("neural_logistic_mean_predictive_loss")
        ),
        "NEURAL_BASELINE_PARTICLE_LOSS": _fmt_nullable(
            neural_baseline.get("particle_state_space_mean_predictive_loss")
        ),
        "LEARNED_STAY_RATE_VERDICT": str(
            learned_stay_rate.get("verdict", "unavailable")
        ),
        "LEARNED_STAY_RATE_BEST_VARIANT": str(
            learned_stay_rate.get("best_variant", "unavailable")
        ),
        "LEARNED_STAY_RATE_DIFF": _fmt_nullable(
            learned_stay_rate.get("best_variant_minus_particle_mean")
        ),
        "LEARNED_STAY_RATE_CI_LOW": _fmt_nullable(
            learned_stay_rate.get("bootstrap_ci_low")
        ),
        "LEARNED_STAY_RATE_CI_HIGH": _fmt_nullable(
            learned_stay_rate.get("bootstrap_ci_high")
        ),
        "HELDOUT_REPORTING_VERDICT": str(
            heldout_reporting.get("verdict", "unavailable")
        ),
        "HELDOUT_REPORTING_RUNLEN_LOTO_GAIN": _fmt_nullable(
            heldout_reporting.get("runlen_loto_gain_vs_pooled")
        ),
        "HELDOUT_REPORTING_RUNLEN_TEMPORAL_GAIN": _fmt_nullable(
            heldout_reporting.get("runlen_temporal_gain_vs_pooled")
        ),
        "CROSS_GENERATOR_PERSISTENCE_VERDICT": _nested_str(
            cross_generator, ("transfer_verdict", "persistence_strength")
        ),
        "CROSS_GENERATOR_RETURN_VERDICT": _nested_str(
            cross_generator, ("transfer_verdict", "return_structure_helps")
        ),
        "CROSS_GENERATOR_PARTICLE_VERDICT": _nested_str(
            cross_generator, ("transfer_verdict", "particle_beats_hard_persistence")
        ),
        "PYMDP_SOTA_STATUS": str(
            pymdp_benchmark.get("state_of_art_performance_status", "unavailable")
        ),
        "PYMDP_INTERPRETABILITY_STATUS": str(
            pymdp_benchmark.get("interpretability_status", "unavailable")
        ),
        "PYMDP_PERFORMANCE_DELTA": _fmt_nullable(
            pymdp_benchmark.get("pymdp_minus_reference_predictive_loss")
        ),
        "PYMDP_ENTROPY_DELTA": _fmt_nullable(
            pymdp_benchmark.get("pymdp_minus_reference_posterior_entropy")
        ),
        "PYMDP_BENCHMARK_TABLE": _pymdp_benchmark_table(
            pymdp_benchmark.get("rows", [])
        ),
        "PYMDP_ROADMAP_GATES": _roadmap_gate_text(
            pymdp_benchmark.get("roadmap_gates", [])
        ),
        "SOTA_CLAIM_GATE_STATUS": str(sota_gate.get("status", "unavailable")),
        "SOTA_STRONGEST_BASELINE_LOSS": _fmt_nullable(
            sota_matrix.get("strongest_baseline_loss")
        ),
        "SOTA_LEAKAGE_PROBE_LOSS": _fmt_nullable(sota_oracle.get("leakage_probe_loss")),
        "AIF_SWEEP_BEST_LOSS": _fmt_nullable(aif_sweep.get("best_config_loss")),
        "AIF_FAIR_AIF_LOSS": _fmt_nullable(aif_fair.get("honest_aif_loss")),
        "AIF_FAIR_HANDICAPPED_LOSS": _fmt_nullable(
            aif_fair.get("handicapped_markov_loss")
        ),
        "AIF_FAIR_ORACLE_LOSS": _fmt_nullable(aif_fair.get("oracle_markov_loss")),
        "AIF_FAIR_WINS": "yes" if aif_fair.get("aif_wins_fair_comparison") else "no",
        "FAIR_ORACLE_LOSS": _fmt_nullable(fair_losses.get("oracle")),
        "FAIR_HARD_LOSS": _fmt_nullable(fair_losses.get("hard_handicapped")),
        "FAIR_SOFT_LOSS": _fmt_nullable(fair_losses.get("soft_handicapped")),
        "FAIR_SOFT_BOOST_LOSS": _fmt_nullable(
            fair_losses.get("soft_plus_return_boost")
        ),
        "FAIR_AIF_LOSS": _fmt_nullable(fair_losses.get("honest_aif")),
        "FAIR_AIF_BOOST_LOSS": _fmt_nullable(fair_losses.get("honest_aif_plus_boost")),
        "FAIR_SOFT_MARG_GAIN": _fmt_nullable(
            fair_decomp.get("soft_marginalization_gain")
        ),
        "FAIR_NOISE_ADV_HIGH": _fmt_nullable(
            (fair_noise.get("advantage_range_nats") or [None, None])[1]
        ),
        "FAIR_SOTA_AIF_BEST_RATE": str(
            fair_sota.get("aif_class_best_fair_rate", "unavailable")
        ),
        "FAIR_SOTA_DISTINCT_TRAVELLERS": str(
            fair_sota.get("n_distinct_holdout_travellers_min", "unavailable")
        ),
        "FAIR_SOTA_ADV_MEAN": _fmt_nullable(fair_sota.get("aif_advantage_mean_nats")),
        "FAIR_SOTA_GAMMA": _fmt_nullable(fair_sota.get("precision")),
        "FAIR_SOTA_LOSS_GAMMA1": _fmt_nullable(fair_sota.get("mean_loss_gamma1")),
        "FAIR_SOTA_LOSS_PRECISION": _fmt_nullable(fair_sota.get("mean_loss_precision")),
        "FAIR_SOTA_ECE_GAMMA1": _fmt_nullable(fair_sota.get("mean_ece_gamma1")),
        "FAIR_SOTA_ECE_PRECISION": _fmt_nullable(fair_sota.get("mean_ece_precision")),
        "FAIR_SOTA_PRECISION_LOSS_WINS": str(
            fair_sota.get("precision_gain_loss_win_rate", "unavailable")
        ),
        "POWER_MAX_TRAVELLERS": str(
            (power.get("traveller_counts") or ["unavailable"])[-1]
        ),
        "POWER_SMALL_CLEAN_ADV": _fmt_nullable(power.get("small_clean_advantage_nats")),
        "POWER_SMALL_N_HOLDOUT": _fmt_nullable(
            _power_first_cell(power).get("min_n_holdout_travellers")
        ),
        "POWER_SMALL_MIN_ACROSS_SEEDS": _fmt_nullable(
            _power_first_cell(power).get("advantage_min_across_seeds")
        ),
        "POWER_LARGE_CLEAN_ADV": _fmt_nullable(power.get("large_clean_advantage_nats")),
        "POWER_LARGE_NOISY_ADV": _fmt_nullable(power.get("large_noisy_advantage_nats")),
        "POWER_LARGE_NOISY_SD": _fmt_nullable(power.get("large_noisy_advantage_sd")),
        "POWER_GEN_SEEDS": str(power.get("n_gen_seeds", "unavailable")),
        "POWER_NOISE_RESTORES": "yes"
        if power.get("noise_restores_advantage_at_scale")
        else "no",
        "EFE_LOCATION_VERDICT": str(efe_location.get("verdict", "unavailable")),
        "EFE_LOCATION_SHARE": _fmt_nullable(efe_location.get("location_joint_share")),
        "EFE_METADATA_NULL_P": _nested_fmt(
            efe_location, ("structure_null", "p_value_null_ge_real")
        ),
        "EFE_RESIDUAL_WITHOUT_LOCATION": _fmt_nullable(
            efe_location.get("residual_localized_without_location")
        ),
        "LINKAGE_RISK_VERDICT": str(linkage_risk.get("verdict", "unavailable")),
        "LINKAGE_LOCATION_LINKAGE": _fmt_nullable(
            linkage_risk.get("location_real_linkage")
        ),
        "LINKAGE_NO_LOCATION_EXCEEDS_NULL": _yes_no(
            _linkage_regime_exceeds_null(linkage_risk, "no_location")
        ),
        "POISONING_VERDICT": str(poisoning_robustness.get("verdict", "unavailable")),
        "POISONING_GROSS_DETECTED": _yes_no(
            poisoning_robustness.get("gross_poisoning_detected")
        ),
        "POISONING_SUBTLE_NULL": _yes_no(
            poisoning_robustness.get("subtle_drift_is_null_vs_markov")
        ),
        "POISONING_TELEPORT_AUC": _fmt_nullable(
            _poisoning_mid_rate_metric(poisoning_robustness, "teleport", "vfe_auc_mean")
        ),
        "PRIVACY_FRONTIER_ALL_UNICITY": _fmt(
            privacy_rows.get("all", {}).get("unicity", 0.0)
        ),
        "PRIVACY_FRONTIER_NO_LOCATION_UNICITY": _fmt(
            privacy_rows.get("no_location", {}).get("unicity", 0.0)
        ),
        "PRIVACY_FRONTIER_NO_LOCATION_COTRAVEL": _fmt(
            privacy_rows.get("no_location", {}).get("co_travel_coincidence", 0.0)
        ),
        "HARDENED_ANOMALY_SPEED_VFE_AUC": _fmt(
            hardened_anomaly.get("speed_regime_shift", {}).get("vfe_auc", 0.0)
        ),
        "HARDENED_ANOMALY_SPEED_MARKOV_AUC": _fmt(
            hardened_anomaly.get("speed_regime_shift", {}).get("markov_auc", 0.0)
        ),
        "HARDENED_ANOMALY_DRIFT_VFE_AUC": _fmt(
            hardened_anomaly.get("preferred_set_drift", {}).get("vfe_auc", 0.0)
        ),
        "VISUAL_QA_STATUS": str(visual_quality.get("status", "unavailable")),
        "VISUAL_QA_FIGURE_COUNT": _fmt_nullable(visual_quality.get("figure_count")),
        "VISUAL_QA_CAPTION_SOURCE_COUNT": _fmt_nullable(
            visual_quality.get("caption_source_language_count")
        ),
        "VISUAL_QA_MIN_WIDTH": _fmt_nullable(visual_quality.get("min_width_px")),
        "VISUAL_QA_MIN_HEIGHT": _fmt_nullable(visual_quality.get("min_height_px")),
        "VISUAL_QA_MIN_PIXEL_STDDEV": _fmt_nullable(
            visual_quality.get("min_pixel_stddev")
        ),
        "REFERENCE_ECE": _fmt(metrics.get("reference_calibration_ece", 0.0)),
        "PYMDP_ECE": _fmt(metrics.get("pymdp_calibration_ece", 0.0)),
        "PYMDP_MEAN_TOP1_CONFIDENCE": _fmt(
            metrics.get("pymdp_calibration_mean_top1_confidence", 0.0)
        ),
        "REFERENCE_MODALITY_INTERACTION": _fmt(
            metrics.get("reference_modality_interaction_information", 0.0)
        ),
        "PER_MODALITY_LOSS_TABLE": _per_modality_table(
            metrics, metadata.get("observation_modalities", [])
        ),
        "DEEP_LATENT_BRIER": _sparsity_full_level(sparsity, "latent_brier"),
        "DEEP_LATENT_BRIER_SKILL": _sparsity_full_level(
            sparsity, "latent_brier_skill_score"
        ),
        "EXTERNAL_REFERENCE_SKILL_SCORE": _fmt_optional(
            external_metrics, "reference_skill_score"
        ),
        "EXTERNAL_PLACE_COUNT": _fmt_optional(external_metrics, "place_n_unique"),
        "EXTERNAL_PLACE_REVISIT_RATE": _fmt_optional(
            external_metrics, "place_revisit_rate"
        ),
        "EXTERNAL_PLACE_REPORTS_PER_PLACE": _fmt_optional(
            external_metrics, "place_mean_reports_per_place"
        ),
        "EXTERNAL_RECOVERY_REFERENCE_MASS": _fmt_optional(
            external_metrics, "state_recovery_reference_mean_mass_on_true_cell"
        ),
        "DEEP_LATENT_RECOVERY_MASS": _fmt(
            deep_latent_metrics.get(
                "state_recovery_reference_mean_mass_on_true_cell", 0.0
            )
        ),
        "DEEP_LATENT_RECOVERY_TOP1": _fmt(
            deep_latent_metrics.get("state_recovery_reference_top1_accuracy", 0.0)
        ),
        "DEEP_LATENT_ENTROPY": _fmt(
            deep_latent_metrics.get("mean_posterior_entropy", 0.0)
        ),
        "SPARSITY_LEVEL_COUNT": str(sparsity.get("n_levels", 0)),
        "SPARSITY_TABLE": _sparsity_table(sparsity.get("levels", [])),
        "SPARSITY_CAVEAT": str(
            sparsity.get("caveat", "sparsity analysis not yet generated")
        ),
        "SPARSITY_DENSEST_LATENT_MASS": _sparsity_extreme(
            sparsity.get("levels", []), densest=True
        ),
        "SPARSITY_SPARSEST_LATENT_MASS": _sparsity_extreme(
            sparsity.get("levels", []), densest=False
        ),
        "FIG_CAPTION_SPARSITY_RECOVERY": _caption(entries, "sparsity_recovery"),
        "EXTERNAL_RUN_STATUS": (
            "completed on this host with the pinned upstream build"
            if external_metrics
            else "not yet executed on this host"
        ),
        "EXTERNAL_OBSERVATION_COUNT": _fmt_optional(external_metrics, "n_observations"),
        "EXTERNAL_TRAVELLER_COUNT": _fmt_optional(external_metrics, "n_travellers"),
        "EXTERNAL_LATENT_STATE_COUNT": _fmt_optional(
            external_metrics, "n_latent_states"
        ),
        "EXTERNAL_MEAN_PREDICTIVE_LOSS": _fmt_optional(
            external_metrics, "mean_predictive_loss"
        ),
        "EXTERNAL_MEAN_POSTERIOR_ENTROPY": _fmt_optional(
            external_metrics, "mean_posterior_entropy"
        ),
        "EXTERNAL_HOLDOUT_MEAN_PREDICTIVE_LOSS": _fmt_optional(
            external_metrics, "holdout_mean_predictive_loss"
        ),
        "EXTERNAL_PYMDP_EXACT_DELTA": _fmt_sci_optional(
            external_metrics, "pymdp_exact_inference_max_abs_delta"
        ),
        "EXTERNAL_SENSITIVITY_RUN_COUNT": str(external_sensitivity.get("n_runs", 0)),
        "EXTERNAL_REPLICATE_DISPERSION": _replicate_dispersion(external_sensitivity),
        "FIGURE_WIDTH": "100",
        "FIG_CAPTION_EVIDENCE_ATLAS": _caption(entries, "evidence_atlas"),
        "FIG_CAPTION_PYMDP_AGREEMENT": _caption(entries, "pymdp_agreement"),
        "FIG_CAPTION_PYMDP_DIAGNOSTICS": _caption(entries, "pymdp_diagnostics"),
        "FIG_CAPTION_STATE_RECOVERY": _caption(entries, "state_recovery"),
        "FIG_CAPTION_NEGATIVE_CONTROLS": _caption(entries, "negative_controls"),
        "FIG_CAPTION_WAYPOINT_PATHS": _caption(entries, "waypoint_paths"),
        "FIG_CAPTION_SPATIAL_BINS": _caption(entries, "spatial_bins"),
        "FIG_CAPTION_SPATIAL_RESOLUTION_FRONTIER": _caption(
            entries, "spatial_resolution_frontier"
        ),
        "FIG_CAPTION_REPORTING_BURSTS": _caption(entries, "reporting_bursts"),
        "FIG_CAPTION_SPEED_CATEGORIES": _caption(entries, "speed_categories"),
        "FIG_CAPTION_TRANSITION_HEATMAP": _caption(entries, "transition_heatmap"),
        "FIG_CAPTION_LIKELIHOOD_PANELS": _caption(entries, "likelihood_panels"),
        "FIG_CAPTION_POSTERIOR_ENTROPY": _caption(entries, "posterior_entropy"),
        "FIG_CAPTION_POLICY_PROXY": _caption(entries, "policy_proxy"),
        "FIG_CAPTION_CANDIDATE_SCORES": _caption(entries, "candidate_scores"),
        "FIG_CAPTION_CALIBRATION": _caption(entries, "calibration"),
        "FIG_CAPTION_ANTICIPATORY": _caption(entries, "anticipatory"),
        "FIG_CAPTION_EFE_DECOMPOSITION": _caption(entries, "efe_decomposition"),
        "FIG_CAPTION_ANOMALY_DETECTION": _caption(entries, "anomaly_detection"),
        "FIG_CAPTION_RECOVERABILITY_FRONTIER": _caption(
            entries, "recoverability_frontier"
        ),
        "FIG_CAPTION_RETURN_STRUCTURE": _caption(entries, "return_structure"),
        "FIG_CAPTION_VARIANCE_BANDS": _caption(entries, "variance_bands"),
        "FIG_CAPTION_UPSTREAM_SENSITIVITY": _caption(entries, "upstream_sensitivity"),
        "FIG_CAPTION_FRACTAL_DIMENSION_SWEEP": _caption(
            entries, "fractal_dimension_sweep"
        ),
        "FIG_CAPTION_CIRCADIAN_PHASE_SIGNAL": _caption(
            entries, "circadian_phase_signal"
        ),
        "FIG_CAPTION_PARTIAL_OBSERVABILITY": _caption(entries, "partial_observability"),
        "FIG_CAPTION_SEQUENCE_BASELINES": _caption(entries, "sequence_baselines"),
        "FIG_CAPTION_MISSINGNESS_MODEL": _caption(entries, "missingness_model"),
        "FIG_CAPTION_MODEL_GATE_MATRIX": _caption(entries, "model_gate_matrix"),
        "FIG_CAPTION_M10_FORWARD_GATES": _caption(entries, "m10_forward_gates"),
        "FIG_CAPTION_FACTORIAL_PYMDP_MODES": _caption(entries, "factorial_pymdp_modes"),
        "FIG_CAPTION_LATENT_MODE_RECOVERY": _caption(entries, "latent_mode_recovery"),
        "FIG_CAPTION_FACTORIAL_MODEL_GATE_MATRIX": _caption(
            entries, "factorial_model_gate_matrix"
        ),
        "FIG_CAPTION_EFE_LOCATION_FRONTIER": _caption(entries, "efe_location_frontier"),
        "FIG_CAPTION_LINKAGE_RISK_FRONTIER": _caption(entries, "linkage_risk_frontier"),
        "FIG_CAPTION_POISONING_ROBUSTNESS": _caption(entries, "poisoning_robustness"),
        "FIG_CAPTION_SOTA_LEADERBOARD": _caption(entries, "sota_leaderboard"),
        "FIG_CAPTION_AIF_REGIME_MAP": _caption(entries, "aif_regime_map"),
        "FIG_CAPTION_FAIR_INFERENCE": _caption(entries, "fair_inference"),
        "FIG_CAPTION_FAIR_POWER_REGIME": _caption(entries, "fair_power_regime"),
        "FIG_CAPTION_MULTIPLICITY_LEDGER": _caption(entries, "multiplicity_ledger"),
        "FIG_CAPTION_CONFOUND_TAXONOMY": _caption(entries, "confound_taxonomy"),
        "FIG_CAPTION_REGIME_MECHANISM_SYNTHESIS": _caption(
            entries, "regime_mechanism_synthesis"
        ),
        "FIG_CAPTION_REGIME_SWITCH_PHASE_DIAGNOSTIC": _caption(
            entries, "regime_switch_phase_diagnostic"
        ),
        "FIG_CAPTION_PRIVACY_FRONTIER": _caption(entries, "privacy_frontier"),
        "FIG_CAPTION_PRIVACY_UTILITY_FRONTIER": _caption(
            entries, "privacy_utility_frontier"
        ),
        "FIG_CAPTION_PRIVACY_UTILITY_REPEATS": _caption(
            entries, "privacy_utility_repeats"
        ),
        "FIG_CAPTION_SYNTHETIC_TASK_UTILITY_PRIVACY": _caption(
            entries, "synthetic_task_utility_privacy"
        ),
        "FIG_CAPTION_MODEL_COMPARISON_UNDER_NOISE": _caption(
            entries, "model_comparison_under_noise"
        ),
        "FIG_CAPTION_AIF_DEEPENING_CANDIDATES": _caption(
            entries, "aif_deepening_candidates"
        ),
        "FIG_CAPTION_HARDENED_ANOMALY_DETECTION": _caption(
            entries, "hardened_anomaly_detection"
        ),
        "FIG_CAPTION_DATAFLOW": _caption(entries, "dataflow"),
    }
    variables.update(_gnn_variables(root))
    return variables


def write_variables(project_root: str | Path | None = None) -> Path:
    """Write variables to the project output directory for inspection."""

    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    variables = compute_variables(root)
    output = root / "output" / "data" / "manuscript_variables.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(variables, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_token_provenance(root, variables)
    return output


def _fmt(value: object) -> str:
    return format_fmt(value)


def _fmt_optional(mapping: dict[str, object], key: str) -> str:
    if key not in mapping:
        return "unavailable"
    return _fmt(mapping[key])


def _lane_list(value: object) -> str:
    if not isinstance(value, list):
        return "unavailable"
    return ", ".join(f"`{item}`" for item in value) if value else "unavailable"


def _int_list(value: object) -> str:
    if not isinstance(value, list):
        return "unavailable"
    return ", ".join(str(int(item)) for item in value) if value else "unavailable"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _require_canonical_sota_program(payload: dict[str, object], path: Path) -> None:
    """Reject existing SOTA-program artifacts that are not safe for manuscript tokens."""

    if not payload:
        return
    run_parameters = payload.get("run_parameters")
    violations = canonical_run_parameter_violations(
        run_parameters if isinstance(run_parameters, dict) else None
    )
    if violations:
        raise ValueError(
            f"{path} is not a canonical SOTA-program artifact ({'; '.join(violations)}); rerun "
            "scripts/18_run_sota_program.py with default parameters or the full study"
        )
    robustness = payload.get("robustness_summary")
    if not isinstance(robustness, dict):
        raise ValueError(
            f"{path} is not a current SOTA-program artifact (robustness_summary missing); rerun "
            "scripts/18_run_sota_program.py with default parameters or the full study"
        )
    required = {
        "canonical_lanes",
        "all_canonical_lanes_scored",
        "reporting_gap_scored",
        "regime_stressor_scored",
        "claim_boundary",
    }
    missing = sorted(required - set(robustness))
    if missing:
        raise ValueError(
            f"{path} is not a current SOTA-program artifact (robustness_summary missing "
            f"{', '.join(missing)}); rerun scripts/18_run_sota_program.py with default parameters "
            "or the full study"
        )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fmt6(value: object) -> str:
    if isinstance(value, float):
        return format_fmt6(value)
    return str(value)


def _candidate_formula(row: dict[str, object]) -> str:
    if not row:
        return "not available"
    objective = row.get("objective", "objective")
    metric = _fmt6(row.get("metric_value", 0.0))
    ratio = _fmt6(row.get("reporting_ratio", 0.0))
    score = _fmt6(row.get("score", 0.0))
    return f"minimize `{objective}`: score = {metric} x {ratio} = {score}"


def _policy_tie_diagnostic(
    mean_policy_entropy: float, max_policy_entropy: float
) -> str:
    if max_policy_entropy <= 0:
        return "single configured policy"
    if abs(mean_policy_entropy - max_policy_entropy) < 1e-3:
        return (
            "near-maximal policy entropy; under the default uniform preferences the proxy "
            "score is constant across controls by algebra, so the configured policies are "
            "exactly tied rather than empirically close"
        )
    return "policy posterior departs from the uniform maximum under the default fixture"


def _caption(entries: dict[str, dict[str, object]], key: str) -> str:
    entry = entries.get(key, {})
    return str(entry.get("caption", key.replace("_", " ")))


def _shape(value: object) -> str:
    if not isinstance(value, list):
        return "`[]`"
    return "`" + " x ".join(str(item) for item in value) + "`"


def _shape_list(values: object) -> str:
    if not isinstance(values, list):
        return "`[]`"
    return ", ".join(_shape(value) for value in values)


def _csv_header(path: Path) -> str:
    if not path.exists():
        return ""
    header = path.read_text(encoding="utf-8").splitlines()[0]
    return ", ".join(f"`{item}`" for item in header.split(","))


def _candidate_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "| rank | candidate | score |\n| --- | --- | --- |\n"
    lines = [
        "| rank | candidate | metric value | reporting ratio | score |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {rank} | `{name}` | {metric} | {ratio} | {score} |".format(
                rank=row["rank"],
                name=row["name"],
                metric=_fmt6(row["metric_value"]),
                ratio=_fmt6(row["reporting_ratio"]),
                score=_fmt6(row["score"]),
            )
        )
    return "\n".join(lines)


def _sensitivity_table(payload: dict[str, object]) -> str:
    summary = payload.get("metric_summary", {})
    if not isinstance(summary, dict) or not summary:
        return "| metric | n | mean | std | min-max | normal-approx interval |\n| --- | --- | --- | --- | --- | --- |\n"
    rows = [
        "| metric | n | mean | std | min-max | normal-approx interval |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for metric in [
        "mean_predictive_loss",
        "mean_posterior_entropy",
        "mean_policy_entropy",
        "baseline_delta_mean_predictive_loss",
    ]:
        stats = summary.get(metric, {})
        if not isinstance(stats, dict):
            continue
        rows.append(
            "| `{metric}` | {n} | {mean} | {std} | [{minimum}, {maximum}] | [{low}, {high}] |".format(
                metric=metric,
                n=_fmt(stats.get("n", 0.0)),
                mean=_fmt(stats.get("mean", 0.0)),
                std=_fmt(stats.get("std", 0.0)),
                minimum=_fmt(stats.get("min", 0.0)),
                maximum=_fmt(stats.get("max", 0.0)),
                low=_fmt(stats.get("normal_approx_low", 0.0)),
                high=_fmt(stats.get("normal_approx_high", 0.0)),
            )
        )
    return "\n".join(rows)


def _policy_entropy_context(payload: dict[str, object]) -> str:
    runs = payload.get("runs", [])
    if not isinstance(runs, list) or not runs:
        return "Policy entropy context is unavailable because the sensitivity summary has no run records."
    counts: dict[int, int] = {}
    max_values: dict[int, float] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        control_count = int(run.get("control_count", 0))
        if control_count <= 0:
            continue
        counts[control_count] = counts.get(control_count, 0) + 1
        max_values[control_count] = float(
            run.get("max_policy_entropy", math.log(control_count))
        )
    if not counts:
        return "Policy entropy context is unavailable because control counts were not recorded."
    parts = [
        f"{count} run(s) with {control_count} control(s), max entropy {_fmt(max_values[control_count])} nats"
        for control_count, count in sorted(counts.items())
    ]
    return "Policy entropy is summarized across " + "; ".join(parts) + "."


def _fmt_sci(value: object) -> str:
    if value is None:
        return "not available"
    return f"{float(value):.2e}"


def _fmt_sci_optional(mapping: dict[str, object], key: str) -> str:
    if key not in mapping:
        return "unavailable"
    return _fmt_sci(mapping[key])


def _replicate_dispersion(payload: dict[str, object]) -> str:
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        return "not available"
    losses = [
        float(run["metrics"]["mean_predictive_loss"])
        for run in runs
        if isinstance(run, dict) and run.get("factors", {}).get("family") == "replicate"
    ]
    if len(losses) < 2:
        return "not available"
    spread = max(losses) - min(losses)
    return f"{spread:.3f}"


def _variance_metric(summary: dict[str, object], metric: str, field: str) -> str:
    if not isinstance(summary, dict):
        return "unavailable"
    # A single draw is not a spread: render the band endpoints loudly
    # unavailable so the "cross-realization band" sentence can never dress one
    # realization up as an empirical spread (the provenance gate then blocks
    # the render on the token/artifact mismatch instead of shipping it).
    try:
        n_realizations = int(summary.get("n_realizations", 0) or 0)
    except (TypeError, ValueError):
        n_realizations = 0
    if n_realizations < 2:
        return "unavailable"
    metrics = summary.get("metrics", {})
    row = metrics.get(metric, {}) if isinstance(metrics, dict) else {}
    return _fmt(row.get(field, 0.0)) if isinstance(row, dict) else "0"


def _metric_interval_value(summary: dict[str, object], metric: str, field: str) -> str:
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    row = metrics.get(metric, {}) if isinstance(metrics, dict) else {}
    if not isinstance(row, dict) or field not in row:
        return "unavailable"
    return _fmt_nullable(row[field])


def _fmt_nullable(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, int | float):
        if not math.isfinite(float(value)):
            return "unavailable"
        return _fmt(float(value))
    return str(value)


def _factorial_table(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return "| model | implemented | mean predictive loss | interpretability surface |\n| --- | --- | --- | --- |\n"
    lines = [
        "| model | implemented | mean predictive loss (nats) | interpretability surface |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        loss = row.get("mean_predictive_loss")
        loss_text = (
            f"{float(loss):.3f}"
            if isinstance(loss, int | float)
            else _status_cell(row.get("status", "n/a"))
        )
        lines.append(
            "| `{model}` | {implemented} | {loss} | {surface} |".format(
                model=row.get("model", "unknown"),
                implemented="yes" if row.get("implemented") else "planned",
                loss=loss_text,
                surface=str(row.get("interpretability_surface", "")),
            )
        )
    return "\n".join(lines)


def _status_cell(value: object) -> str:
    return str(value).replace("_", " ")


def _pymdp_benchmark_table(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return "| model | implemented | current role | interpretability surface |\n| --- | --- | --- | --- |\n"
    lines = [
        "| model | implemented | current role | interpretability surface |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| `{model}` | {implemented} | {role} | {surface} |".format(
                model=row.get("model", "unknown"),
                implemented="yes" if row.get("implemented") else "planned",
                role=str(row.get("current_role", "")),
                surface=str(
                    row.get(
                        "interpretability_surface", row.get("candidate_families", "")
                    )
                ),
            )
        )
    return "\n".join(lines)


def _sequence_baseline_table(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return "| model | mean predictive loss | skill vs base-rate |\n| --- | --- | --- |\n"
    lines = [
        "| model | mean predictive loss (nats) | skill vs base-rate |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| `{model}` | {loss} | {skill} |".format(
                model=row.get("model", "unknown"),
                loss=_fmt_nullable(row.get("mean_predictive_loss")),
                skill=_fmt_nullable(row.get("skill_vs_base_rate")),
            )
        )
    return "\n".join(lines)


def _missingness_table(payload: object) -> str:
    if not isinstance(payload, dict) or not payload:
        return "| reporting context | log-loss gain vs pooled |\n| --- | --- |\n"
    rows = [
        "| reporting context | log-loss gain vs pooled |",
        "| --- | --- |",
        f"| phase | {_fmt_nullable(payload.get('phase_gain_vs_pooled'))} |",
        f"| traveller | {_fmt_nullable(payload.get('traveller_gain_vs_pooled'))} |",
        f"| phase + traveller | {_fmt_nullable(payload.get('phase_traveller_gain_vs_pooled'))} |",
        f"| permuted phase | {_fmt_nullable(payload.get('permuted_phase_gain_vs_pooled'))} |",
    ]
    return "\n".join(rows)


def _return_phase_table(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return "| prior | mean predictive loss | skill vs base-rate |\n| --- | --- | --- |\n"
    lines = [
        "| prior | mean predictive loss (nats) | skill vs base-rate |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| `{model}` | {loss} | {skill} |".format(
                model=row.get("model", "unknown"),
                loss=_fmt_nullable(row.get("mean_predictive_loss")),
                skill=_fmt_nullable(row.get("skill_vs_base_rate")),
            )
        )
    return "\n".join(lines)


def _model_gate_matrix_table(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return "| lane | gate | status | model | metric | verdict |\n| --- | --- | --- | --- | --- | --- |\n"
    lines = [
        "| lane | gate | status | model | metric | verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| `{lane}` | `{gate}` | {status} | `{model}` | {metric} | `{verdict}` |".format(
                lane=row.get("lane", "unknown"),
                gate=row.get("gate", "unknown"),
                status=row.get("status", "unknown"),
                model=row.get("model", "unknown"),
                metric=_fmt_nullable(row.get("primary_metric")),
                verdict=row.get("verdict", "unknown"),
            )
        )
    return "\n".join(lines)


def _forward_gate_table(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return "| lane | gate | status | metric | verdict | artifact |\n| --- | --- | --- | --- | --- | --- |\n"
    lines = [
        "| lane | gate | status | metric | verdict | artifact |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| `{lane}` | `{gate}` | {status} | {metric} | `{verdict}` | `{artifact}` |".format(
                lane=row.get("lane", "unknown"),
                gate=row.get("gate", "unknown"),
                status=row.get("status", "unknown"),
                metric=_fmt_nullable(row.get("primary_metric")),
                verdict=row.get("verdict", "unknown"),
                artifact=row.get("artifact", ""),
            )
        )
    return "\n".join(lines)


def _spatial_resolution_table(rows: object) -> str:
    if not isinstance(rows, list) or not rows:
        return "| bins | states | occupied | empty | full loss | no-location mass | pymdp delta |\n| --- | --- | --- | --- | --- | --- | --- |\n"
    lines = [
        "| bins | states | occupied | empty | full loss | no-location mass | pymdp delta |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(
        (item for item in rows if isinstance(item, dict)),
        key=lambda item: int(item["spatial_bins"]),
    ):
        lines.append(
            "| {bins} | {states} | {occupied} | {empty} | {loss} | {mass} | {delta} |".format(
                bins=row.get("spatial_bins", "unknown"),
                states=row.get("state_count", "unknown"),
                occupied=row.get("occupied_state_count", "unknown"),
                empty=row.get("empty_state_count", "unknown"),
                loss=_fmt_nullable(row.get("full_location_mean_predictive_loss")),
                mass=_fmt_nullable(row.get("no_location_mean_mass_on_true_cell")),
                delta=_fmt_sci(row.get("pymdp_exact_inference_max_abs_delta")),
            )
        )
    return "\n".join(lines)


def _power_first_cell(power: dict[str, object]) -> dict[str, object]:
    """Smallest traveller-count cell of the power map (cells[0]), {} when absent."""
    cells = power.get("cells", [])
    if isinstance(cells, list) and cells and isinstance(cells[0], dict):
        return cells[0]
    return {}


def _nested_value(payload: dict[str, object], path: tuple[str, ...]) -> object:
    current: object = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _nested_str(payload: dict[str, object], path: tuple[str, ...]) -> str:
    value = _nested_value(payload, path)
    return "unavailable" if value is None else str(value)


def _nested_fmt(payload: dict[str, object], path: tuple[str, ...]) -> str:
    value = _nested_value(payload, path)
    return "unavailable" if value is None else _fmt_nullable(value)


def _yes_no(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return "unavailable"


def _linkage_regime_exceeds_null(payload: dict[str, object], regime: str) -> object:
    rows = payload.get("frontier", [])
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict) or row.get("regime") != regime:
            continue
        null = row.get("structure_null", {})
        if isinstance(null, dict):
            return null.get("exceeds_null")
    return None


def _poisoning_mid_rate_metric(
    payload: dict[str, object], mode: str, metric: str
) -> object:
    per_mode = payload.get("per_mode", {})
    if not isinstance(per_mode, dict):
        return None
    per_rate = per_mode.get(mode, {})
    if not isinstance(per_rate, dict):
        return None
    row = per_rate.get(f"{1 / 6:.4f}", {})
    if not isinstance(row, dict):
        return None
    return row.get(metric)


def _roadmap_gate_text(gates: object) -> str:
    if not isinstance(gates, list) or not gates:
        return "unavailable"
    return "; ".join(str(gate) for gate in gates)


def _sparsity_table(levels: list[dict[str, object]]) -> str:
    if not levels:
        return "| reports/traveller | latent mass | latent top-1 | latent entropy | pymdp VFE |\n| --- | --- | --- | --- | --- |\n"
    rows = [
        "| reports/traveller | full mass | latent mass | latent top-1 | latent entropy (nats) | pymdp VFE (nats) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for level in sorted(
        levels, key=lambda item: -float(item["mean_reports_per_traveller"])
    ):
        rows.append(
            "| {rpt} | {full} | {mass} | {top1} | {entropy} | {vfe} |".format(
                rpt=_fmt(float(level["mean_reports_per_traveller"])),
                full=_fmt(float(level["full_mean_mass_on_true_cell"])),
                mass=_fmt(float(level["latent_mean_mass_on_true_cell"])),
                top1=_fmt(float(level["latent_top1_accuracy"])),
                entropy=_fmt(float(level["latent_mean_posterior_entropy"])),
                vfe=_fmt(float(level["pymdp_latent_mean_vfe"])),
            )
        )
    return "\n".join(rows)


def _sparsity_extreme(levels: list[dict[str, object]], densest: bool) -> str:
    if not levels:
        return "not available"
    ordered = sorted(
        levels,
        key=lambda item: float(item["mean_reports_per_traveller"]),
        reverse=densest,
    )
    return _fmt(float(ordered[0]["latent_mean_mass_on_true_cell"]))


def _negative_control_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "| control | probe | delta mean predictive loss | passed |\n| --- | --- | --- | --- |\n"
    lines = [
        "| control | probe | delta mean predictive loss | passed |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| `{name}` | {probe} | {delta} | {passed} |".format(
                name=row.get("name", "unknown"),
                probe=str(row.get("probe", "")),
                delta=_fmt(float(row.get("delta_mean_predictive_loss", 0.0))),
                passed="yes" if row.get("passed") else "NO",
            )
        )
    return "\n".join(lines)


def _per_modality_table(metrics: dict[str, object], modalities: list[str]) -> str:
    if not modalities:
        return "| modality | reference marginal loss | pymdp marginal loss |\n| --- | --- | --- |\n"
    rows = [
        "| modality | reference marginal loss (nats) | pymdp marginal loss (nats) |",
        "| --- | --- | --- |",
    ]
    for name in modalities:
        rows.append(
            "| `{name}` | {ref} | {pym} |".format(
                name=name,
                ref=_fmt(
                    float(metrics.get(f"reference_per_modality_loss_{name}", 0.0))
                ),
                pym=_fmt(float(metrics.get(f"pymdp_per_modality_loss_{name}", 0.0))),
            )
        )
    rows.append(
        "| joint (all modalities) | {ref} | {pym} |".format(
            ref=_fmt(float(metrics.get("mean_predictive_loss", 0.0))),
            pym=_fmt(float(metrics.get("pymdp_mean_predictive_loss", 0.0))),
        )
    )
    return "\n".join(rows)


def _sparsity_full_level(sparsity: dict[str, object], key: str) -> str:
    levels = sparsity.get("levels", [])
    if not isinstance(levels, list) or not levels:
        return "not available"
    densest = max(levels, key=lambda item: float(item.get("fraction", 0.0)))
    if key not in densest:
        return "not available"
    return _fmt(float(densest[key]))


def _figure_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "| key | source |\n| --- | --- |\n"
    lines = ["| figure key | source artifact |", "| --- | --- |"]
    for row in rows:
        lines.append(f"| `{row['key']}` | `{row['source']}` |")
    return "\n".join(lines)
