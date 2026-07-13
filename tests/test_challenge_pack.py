from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from active_fractal_rabbit.challenge_pack import (
    CHALLENGE_PACK_SCHEMA,
    LITERATURE_REFRESH_DATE,
    NON_OPERATIONAL_USAGE_STATEMENT,
    build_open_challenge_pack,
)
from active_fractal_rabbit.sota_program_harness import (
    CANONICAL_LANES,
    DEFAULT_BOOT_SEED,
    DEFAULT_EPS,
    DEFAULT_N_BOOT,
    DEFAULT_N_TRAVELLERS,
    DEFAULT_SPATIAL_BINS,
    SOTA_PROGRAM_LEADERBOARD_HEADER,
)


def _seed_pack_project(root: Path) -> None:
    runs = root / "output" / "runs"
    data = root / "output" / "data"
    runs.mkdir(parents=True)
    data.mkdir(parents=True)
    (runs / "sota_benchmark_matrix.json").write_text(
        json.dumps(
            {
                "status": "sota_not_cleared",
                "locked_split": {"seed": 20260616, "n_holdout_travellers": 3},
                "leaderboard": [
                    {
                        "model": "return_boost_markov",
                        "family": "transparent_null",
                        "lane": "fixture",
                        "locked_split_scored": True,
                        "leakage": False,
                        "mean_predictive_loss": 1.5,
                        "loss_ci_low": 1.4,
                        "loss_ci_high": 1.6,
                        "brier_score": 0.4,
                        "ece": 0.1,
                        "runtime_seconds": 0.01,
                    }
                ],
                "missing_baseline_families": ["foundation_mobility", "llm_mobility"],
                "claim_gate_reasons": [
                    "required strong-baseline families absent on the locked split"
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (runs / "sota_task_contract.json").write_text(
        json.dumps({"status": "ok"}) + "\n", encoding="utf-8"
    )
    (runs / "sota_claim_gate.json").write_text(
        json.dumps(
            {
                "status": "sota_not_cleared",
                "missing_baseline_families": ["foundation_mobility", "llm_mobility"],
                "claim_gate_reasons": [
                    "required strong-baseline families absent on the locked split"
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (runs / "sota_leaderboard.csv").write_text(
        "model,family,lane,locked_split_scored,leakage,mean_predictive_loss,loss_ci_low,loss_ci_high,brier_score,ece,runtime_seconds\n"
        "return_boost_markov,transparent_null,fixture,true,false,1.5,1.4,1.6,0.4,0.1,0.01\n",
        encoding="utf-8",
    )
    (runs / "sota_program_gate.json").write_text(
        json.dumps(
            {
                "run_parameters": {
                    "canonical": True,
                    "n_travellers": DEFAULT_N_TRAVELLERS,
                    "eps": DEFAULT_EPS,
                    "spatial_bins": DEFAULT_SPATIAL_BINS,
                    "n_boot": DEFAULT_N_BOOT,
                    "boot_seed": DEFAULT_BOOT_SEED,
                    "lanes": list(CANONICAL_LANES),
                },
                "lane_definitions": {
                    name: {
                        "status": "scored",
                        "robustness_role": f"{name} synthetic lane",
                    }
                    for name in CANONICAL_LANES
                },
                "robustness_summary": {
                    "canonical_lanes": list(CANONICAL_LANES),
                    "all_canonical_lanes_scored": True,
                    "reporting_gap_scored": True,
                    "regime_stressor_scored": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (runs / "sota_program_leaderboard.csv").write_text(
        ",".join(SOTA_PROGRAM_LEADERBOARD_HEADER)
        + "\n"
        + ",".join(
            [
                "precision_return_aif",
                "fractalrabbit",
                "aif_candidate",
                "1.0",
                "100",
                "active_inference_candidate",
                "true",
                "false",
                "0.9",
                "1.1",
                "-0.1",
                "0.1",
                "0.2",
                "0.05",
                "0.8",
                "0.01",
                f"scored_on_fractalrabbit; canonical_lanes={'|'.join(CANONICAL_LANES)}",
                "project_python_numpy_no_external_weights",
                "scored_nominal_noisy_emission_partial_observability",
                "aif_gate_candidate_not_ci_separated_or_not_lower_loss_than_switching_noise_hmm_on_fractalrabbit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (runs / "sota_program_robustness_grid.csv").write_text(
        ",".join(SOTA_PROGRAM_LEADERBOARD_HEADER)
        + "\n"
        + ",".join(
            [
                "precision_return_aif",
                "reporting_gap_adversarial",
                "aif_candidate",
                "1.1",
                "90",
                "active_inference_candidate",
                "true",
                "false",
                "1.0",
                "1.2",
                "-0.1",
                "0.1",
                "0.3",
                "0.06",
                "0.75",
                "0.01",
                f"scored_on_reporting_gap_adversarial; canonical_lanes={'|'.join(CANONICAL_LANES)}",
                "project_python_numpy_no_external_weights",
                "scored_adversarial_high_motion_missingness_diagnostic_locked_split",
                "diagnostic_stressor_does_not_clear_sota",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (runs / "sota_program_robustness_grid.json").write_text(
        json.dumps(
            {
                "schema": "active-fractal-rabbit-sota-robustness-diagnostics-v1",
                "status": "diagnostic_stressors_scored",
                "all_diagnostic_lanes_scored": True,
                "lane_statuses": [
                    {"lane": "reporting_gap_adversarial", "status": "scored"},
                    {"lane": "regime_stressor_repeat", "status": "scored"},
                ],
                "boundary": "Diagnostic rows do not by themselves clear SOTA.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    aif_header = [
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
    aif_row = [
        "identity_emission_collapse",
        "20260625",
        "20260616",
        "20260701",
        "36",
        "identity_known_emission_eps_0",
        "none",
        "8",
        "soft_filter",
        "active_inference_reference",
        "best_aif_candidate",
        "1.2",
        "1.1",
        "1.3",
        "-0.1",
        "0.1",
        "0.2",
        "0.05",
        "0.8",
        "0.01",
        "project_python_numpy_no_external_weights",
        "0.7",
        "passes_oracle_floor",
        "true",
        "true",
        "known_to_all_candidates",
        "realization_not_cleared",
        "negative_control_not_eligible_for_regime_sota_promotion",
    ]
    (runs / "aif_regime_search.csv").write_text(
        ",".join(aif_header) + "\n" + ",".join(aif_row) + "\n",
        encoding="utf-8",
    )
    aif_payload = {
        "schema": "active-fractal-rabbit-aif-regime-search-v1",
        "status": "regime_qualified_sota_not_cleared",
        "gate_clears": False,
        "cleared_regime_count": 0,
        "best_cleared_regime": None,
        "best_cleared_regime_id": "none",
        "best_aif_candidates": [],
        "best_strongest_fair_baselines": [],
        "best_min_advantage_nats": None,
        "best_min_clustered_ci_low": None,
        "global_sota_boundary": "Regime-local synthetic evidence does not imply external or global SOTA.",
        "controls": {"identity_emission_collapse_negative_control_passed": True},
        "regime_summaries": {
            "identity_emission_collapse": {
                "status": "regime_not_cleared",
                "gate_clears": False,
            }
        },
    }
    (runs / "aif_regime_search.json").write_text(
        json.dumps(aif_payload) + "\n", encoding="utf-8"
    )
    gate_payload = dict(aif_payload)
    gate_payload["schema"] = "active-fractal-rabbit-aif-regime-sota-gate-v1"
    (runs / "aif_regime_sota_gate.json").write_text(
        json.dumps(gate_payload) + "\n", encoding="utf-8"
    )
    (runs / "fair_inference_external_realizations.json").write_text(
        json.dumps(
            {
                "schema": "active-fractal-rabbit-fair-external-realizations-v1",
                "status": "blocked_external_realizations_incomplete",
                "completed_count": 1,
                "all_required_completed": False,
                "can_promote_beyond_fixture_language": False,
                "boundary": "Multiple current external simulator realizations are required.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (runs / "visual_quality_audit.json").write_text(
        json.dumps({"status": "passed", "figure_count": 43}) + "\n", encoding="utf-8"
    )
    (runs / "visual_claim_audit.json").write_text(
        json.dumps({"status": "passed"}) + "\n", encoding="utf-8"
    )
    (data / "manuscript_variable_provenance.json").write_text(
        json.dumps({"status": "passed", "n_tokens": 12}) + "\n", encoding="utf-8"
    )


def test_challenge_pack_records_non_operational_boundary_and_blocked_foundation_rows(
    tmp_path,
) -> None:
    _seed_pack_project(tmp_path)

    path = build_open_challenge_pack(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == CHALLENGE_PACK_SCHEMA
    assert payload["non_operational_usage_statement"] == NON_OPERATIONAL_USAGE_STATEMENT
    assert payload["locked_splits"]["sota_benchmark"]["seed"] == 20260616
    assert payload["claim_gate"]["status"] == "sota_not_cleared"
    family_status = {
        row["family"]: row["status"] for row in payload["baseline_family_contracts"]
    }
    assert (
        family_status["foundation_mobility"]
        == "blocked_pending_license_compute_data_and_adapter_audit"
    )
    assert (
        family_status["llm_mobility"]
        == "blocked_pending_license_compute_data_and_adapter_audit"
    )
    assert payload["literature_refresh"]["date_checked"] == LITERATURE_REFRESH_DATE
    assert payload["literature_refresh"]["foundation_style_baselines_entered"] is False
    assert payload["sota_program_candidate_table_completeness"]["status"] == "pass"
    assert payload["sota_program_robustness_grid_completeness"]["status"] == "pass"
    assert (
        payload["sota_program_robustness_diagnostics"]["all_diagnostic_lanes_scored"]
        is True
    )
    assert payload["aif_regime_search"]["status"] == "regime_qualified_sota_not_cleared"
    assert payload["aif_regime_sota_gate"]["gate_clears"] is False
    assert payload["aif_regime_search_rows"]
    aif_row = payload["aif_regime_search_rows"][0]
    assert aif_row["regime_id"] == "identity_emission_collapse"
    assert aif_row["candidate"] == "soft_filter"
    assert aif_row["family"] == "active_inference_reference"
    assert aif_row["mean_loss"] == 1.2
    assert aif_row["loss_ci_low"] == 1.1
    assert aif_row["locked_split_scored"] is True
    assert aif_row["leakage_status"] == "passes_oracle_floor"
    assert (
        payload["fair_external_realizations"]["can_promote_beyond_fixture_language"]
        is False
    )
    assert payload["sota_program_leaderboard_rows"]
    assert payload["sota_program_robustness_grid_rows"]
    program_row = payload["sota_program_leaderboard_rows"][0]
    assert program_row["family"] == "active_inference_candidate"
    assert program_row["locked_split_scored"] is True
    assert program_row["leakage"] is False
    assert program_row["loss_ci_low"] == 0.9
    assert program_row["missingness_robustness_status"].startswith("scored_")
    robustness_row = payload["sota_program_robustness_grid_rows"][0]
    assert robustness_row["candidate"] == "precision_return_aif"
    assert robustness_row["lane"] == "reporting_gap_adversarial"
    assert robustness_row["family"] == "active_inference_candidate"
    assert robustness_row["mean_loss"] == 1.1
    assert robustness_row["locked_split_scored"] is True
    assert robustness_row["missingness_robustness_status"] == (
        "scored_adversarial_high_motion_missingness_diagnostic_locked_split"
    )
    assert set(payload["sota_program_lane_definitions"]) == set(CANONICAL_LANES)
    assert (
        payload["sota_program_robustness_summary"]["all_canonical_lanes_scored"] is True
    )
    sources = {row["source"] for row in payload["literature_refresh"]["sources"]}
    assert {
        "https://arxiv.org/abs/2406.02578",
        "https://arxiv.org/abs/2505.18670",
        "https://arxiv.org/abs/2410.23692",
        "https://doi.org/10.1016/j.trc.2026.105562",
        "https://arxiv.org/abs/2410.09129",
        "https://arxiv.org/abs/2606.05130",
        "https://doi.org/10.1007/s10994-023-06386-x",
        "https://arxiv.org/abs/2606.00733",
        "https://arxiv.org/abs/2510.24375",
        "https://doi.org/10.56553/popets-2024-0068",
        "https://arxiv.org/abs/2506.05611",
    } <= sources
    assert payload["empirical_transfer"]["status"] == "blocked"
    assert all(
        "sha256" in item for item in payload["source_artifacts"] if item["exists"]
    )
    # Independently recompute the digest for a real artifact — catches a _sha256()
    # regression that reports a constant/fake digest instead of hashing file bytes.
    artifacts_by_path = {item["path"]: item for item in payload["source_artifacts"]}
    contract_artifact = artifacts_by_path["output/runs/sota_task_contract.json"]
    expected_digest = hashlib.sha256(
        (tmp_path / "output" / "runs" / "sota_task_contract.json").read_bytes()
    ).hexdigest()
    assert contract_artifact["sha256"] == expected_digest


def test_challenge_pack_rejects_noncanonical_sota_program_artifact(tmp_path) -> None:
    _seed_pack_project(tmp_path)
    gate_path = tmp_path / "output" / "runs" / "sota_program_gate.json"
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    payload["run_parameters"]["n_travellers"] = 8
    payload["run_parameters"]["canonical"] = True
    gate_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a canonical SOTA-program artifact"):
        build_open_challenge_pack(tmp_path)


def test_challenge_pack_rejects_malformed_required_artifact(tmp_path) -> None:
    _seed_pack_project(tmp_path)
    (tmp_path / "output" / "runs" / "sota_claim_gate.json").write_text(
        "{not-json\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        build_open_challenge_pack(tmp_path)
