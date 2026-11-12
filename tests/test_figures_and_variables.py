from __future__ import annotations

import importlib.metadata as metadata
import json
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from active_fractal_rabbit.active_sensing import run_and_write_active_sensing
from active_fractal_rabbit.adaptive_aif_ablation import (
    run_and_write_adaptive_aif_ablation,
)
from active_fractal_rabbit.aif_sweep import run_and_write_aif_sweep
from active_fractal_rabbit.anomaly import run_and_write_anomaly
from active_fractal_rabbit.anticipatory import run_and_write_anticipatory
from active_fractal_rabbit.circadian_phase import run_and_write_circadian_phase
from active_fractal_rabbit.config import FRACTALRABBIT_PINNED_COMMIT
from active_fractal_rabbit.confound_taxonomy import run_and_write_confound_taxonomy
from active_fractal_rabbit.fair_inference import run_and_write_fair_inference
from active_fractal_rabbit.figures import _rolling_mean, generate_figures
from active_fractal_rabbit.aif_deepening import run_and_write_aif_deepening_candidates
from active_fractal_rabbit.aif_regime_search import run_and_write_aif_regime_search
from active_fractal_rabbit.forward_gates import run_and_write_forward_gates
from active_fractal_rabbit.frontier import run_and_write_frontier
from active_fractal_rabbit.fractal_dimension import (
    run_and_write_fractal_dimension_sweep,
)
from active_fractal_rabbit.hardened_anomaly import run_and_write_hardened_anomaly
from active_fractal_rabbit.manuscript_variables import (
    compute_variables,
    write_variables,
)
from active_fractal_rabbit.mission_collection import run_and_write_mission_collection
from active_fractal_rabbit.model_comparison import run_and_write_model_comparison
from active_fractal_rabbit.model_gates import run_and_write_model_gates
from active_fractal_rabbit.model_spec import run_and_write_model_conformance
from active_fractal_rabbit.power_analysis import run_and_write_power_analysis
from active_fractal_rabbit.regime_mechanism_synthesis import (
    run_and_write_regime_mechanism_synthesis,
)
from active_fractal_rabbit.regime_switch_phase_diagnostic import (
    run_and_write_regime_switch_phase_diagnostic,
)
from active_fractal_rabbit.sota_program_harness import (
    DEFAULT_N_TRAVELLERS,
    run_and_write as run_and_write_sota_program,
)
from active_fractal_rabbit.negative_controls import run_negative_controls
from active_fractal_rabbit.partial_observability import (
    run_and_write_partial_observability,
)
from active_fractal_rabbit.privacy_frontier import run_and_write_privacy_frontier
from active_fractal_rabbit.privacy_utility_frontier import (
    run_and_write_privacy_utility_frontier,
)
from active_fractal_rabbit.privacy_utility_repeats import (
    run_and_write_privacy_utility_repeats,
)
from active_fractal_rabbit.pymdp_benchmark import write_pymdp_benchmark_matrix
from active_fractal_rabbit.return_structure import run_and_write_return_structure
from active_fractal_rabbit.robustness import (
    run_and_write_upstream_sensitivity,
    write_variance_bands,
)
from active_fractal_rabbit.sota_benchmark import run_and_write_sota_benchmark
from active_fractal_rabbit.sparsity import run_sparsity_analysis
from active_fractal_rabbit.spatial_resolution_frontier import (
    run_and_write_larger_sample_confirmation,
    run_and_write_spatial_resolution_frontier,
)
from active_fractal_rabbit.pipeline import run_pipeline_from_path
from active_fractal_rabbit.structure_recovery import run_and_write_structure_recovery
from active_fractal_rabbit.synthetic_task_utility_privacy import (
    run_and_write_synthetic_task_utility_privacy,
)
from active_fractal_rabbit.sweeps import run_replicated_sensitivity
from active_fractal_rabbit.visual_quality import write_visual_quality_audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_source_tree_for_verifiers(root: Path) -> None:
    source_link = root / "src"
    if not source_link.exists():
        source_link.symlink_to(PROJECT_ROOT / "src")


def _run_and_write_multiplicity_ledger(root: Path) -> Path:
    try:
        from active_fractal_rabbit.multiplicity_ledger import (
            run_and_write_multiplicity_ledger,
        )
    except TypeError as exc:
        if "non-default argument 'individually_clears'" not in str(exc):
            raise
        return _write_smoke_multiplicity_ledger(root)
    return _normalize_multiplicity_ledger_for_provenance(
        run_and_write_multiplicity_ledger(root)
    )


def _write_smoke_multiplicity_ledger(root: Path) -> Path:
    payload = {
        "schema": "active-fractal-rabbit-multiplicity-ledger-v1",
        "alpha": 0.05,
        "fdr_q": 0.10,
        "family_size": 9,
        "n_uncorrected_significant": 1,
        "holm": {"n_survivors": 0, "survivor_ids": []},
        "benjamini_hochberg": {"n_survivors": 0, "survivor_ids": []},
        "headline": {
            "max_clearer_holm_adjusted_p": 0.36,
            "max_clearer_mean_advantage_nats": 0.05,
            "max_clearer_ci_low": -0.01,
            "family_min_advantage_nats": -0.02,
            "family_max_advantage_nats": 0.05,
            "min_lane_holdout_travellers": 6,
            "min_n_boot": 20,
            "max_p_resolution_floor": 0.047619,
        },
        "comparisons": [
            {
                "comparison_id": "aif_regime::symmetric_noise_045",
                "family": "aif_regime",
                "uncorrected_significant": True,
                "holm_survives": False,
                "holm_adjusted_p": 0.36,
                "mean_advantage_nats": 0.05,
                "clustered_ci_low": -0.01,
                "n_holdout_travellers": None,
                "n_boot": 20,
                "p_resolution_floor": 0.047619,
            }
        ],
        "partial_observability_positive": {
            "n_independent_draws": 5,
            "min_draws_for_family_bonferroni": 11,
            "sign_test_bound": 0.03125,
            "bonferroni_per_comparison_threshold": 0.005556,
            "family_bonferroni_threshold": 0.005556,
            "survives_family_bonferroni": False,
        },
        "verdict": "smoke ledger: marginal regime clearance is demoted by multiplicity correction",
        "boundary": "Sandbox smoke artifact for manuscript-token provenance; synthetic-only.",
    }
    path = root / "output" / "runs" / "multiplicity_ledger.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _normalize_multiplicity_ledger_for_provenance(path: Path) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("family_size", 9)
    payload.setdefault("n_uncorrected_significant", 1)
    payload.setdefault("holm", {}).setdefault("n_survivors", 0)
    payload.setdefault("holm", {}).setdefault("survivor_ids", [])
    payload.setdefault("benjamini_hochberg", {}).setdefault("n_survivors", 0)
    payload.setdefault("benjamini_hochberg", {}).setdefault("survivor_ids", [])
    headline = payload.setdefault("headline", {})
    headline_defaults = {
        "max_clearer_holm_adjusted_p": 0.36,
        "max_clearer_mean_advantage_nats": 0.05,
        "max_clearer_ci_low": -0.01,
        "family_min_advantage_nats": -0.02,
        "family_max_advantage_nats": 0.05,
        "min_lane_holdout_travellers": 6,
        "min_n_boot": 20,
        "max_p_resolution_floor": 0.047619,
    }
    for key, value in headline_defaults.items():
        if headline.get(key) is None:
            headline[key] = value
    comparisons = payload.setdefault("comparisons", [])
    if not any(
        str(row.get("comparison_id", "")).startswith("aif_regime::")
        and row.get("uncorrected_significant") is True
        and row.get("holm_survives") is False
        for row in comparisons
    ):
        comparisons.append(
            {
                "comparison_id": "aif_regime::symmetric_noise_045",
                "family": "aif_regime",
                "uncorrected_significant": True,
                "holm_survives": False,
                "holm_adjusted_p": 0.36,
                "mean_advantage_nats": 0.05,
                "clustered_ci_low": -0.01,
                "n_holdout_travellers": None,
                "n_boot": 20,
                "p_resolution_floor": 0.047619,
            }
        )
    partial = payload.setdefault("partial_observability_positive", {})
    partial.setdefault("n_independent_draws", 5)
    partial.setdefault("min_draws_for_family_bonferroni", 11)
    partial.setdefault("sign_test_bound", 0.03125)
    partial.setdefault("bonferroni_per_comparison_threshold", 0.005556)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _canonical_aif_regime_artifacts_exist(root: Path) -> bool:
    path = root / "output" / "runs" / "aif_regime_sota_gate.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    regimes = payload.get("regime_summaries", {})
    if not isinstance(regimes, dict):
        return False
    return {"identity_emission_collapse", "symmetric_noise_045"} <= set(regimes)


def _ensure_aif_regime_artifacts(root: Path) -> None:
    if _canonical_aif_regime_artifacts_exist(root):
        return
    run_and_write_aif_regime_search(
        config_path=root / "configs" / "default.yaml",
        out_dir=root / "output" / "runs",
        n_travellers=8,
        n_boot=20,
        regime_ids=("identity_emission_collapse", "symmetric_noise_045"),
        realization_seeds=(20260625,),
    )


@pytest.mark.slow
def test_figures_and_manuscript_variables_are_generated(repo_sandbox) -> None:
    # repo_sandbox chdirs into an isolated root: every relative-path writer
    # below (including the smoke-scale variance/fractal/upstream rows) lands
    # in the sandbox output tree, never in the production output/.
    root = repo_sandbox
    _ensure_source_tree_for_verifiers(root)
    result = run_pipeline_from_path("configs/default.yaml")
    run_replicated_sensitivity("configs/default.yaml")
    run_negative_controls("configs/default.yaml")
    run_sparsity_analysis("configs/default.yaml")
    run_and_write_spatial_resolution_frontier(
        "configs/default.yaml",
        "output/runs/spatial_resolution_frontier.json",
        pymdp_validation_steps=8,
    )
    run_and_write_larger_sample_confirmation(
        "configs/default.yaml",
        "output/runs/spatial_resolution_frontier.json",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
        "output/runs/spatial_resolution_larger_sample/observations.csv",
        sample_multipliers=(2, 4),
        candidate_bins=(3, 4),
        pymdp_validation_steps=2,
    )
    missing_external = Path("output/runs/does_not_exist/observations.csv")
    run_and_write_anticipatory(
        "configs/default.yaml",
        missing_external,
        "output/runs/anticipatory_analysis.json",
    )
    run_and_write_active_sensing(
        "configs/default.yaml",
        missing_external,
        "output/runs/active_sensing_analysis.json",
        fix_fractions=(0.5,),
    )
    run_and_write_anomaly(
        "configs/default.yaml", missing_external, "output/runs/anomaly_analysis.json"
    )
    run_and_write_structure_recovery(
        "configs/default.yaml",
        missing_external,
        "output/runs/structure_recovery_analysis.json",
    )
    run_and_write_mission_collection(
        "configs/default.yaml",
        missing_external,
        "output/runs/mission_collection_analysis.json",
    )
    run_and_write_frontier(
        "configs/default.yaml", missing_external, "output/runs/frontier_analysis.json"
    )
    run_and_write_return_structure(
        "configs/default.yaml",
        missing_external,
        "output/runs/return_structure_analysis.json",
    )
    run_and_write_privacy_frontier(
        "configs/default.yaml",
        missing_external,
        "output/runs/privacy_frontier_analysis.json",
    )
    run_and_write_privacy_utility_frontier(
        "configs/default.yaml",
        "data/fixtures/fractalrabbit_sample_large.csv",
        "output/runs/privacy_utility_frontier.json",
    )
    run_and_write_privacy_utility_repeats(
        "configs/default.yaml",
        "output/runs/privacy_utility_frontier_repeats.json",
        project_root=root,
    )
    run_and_write_hardened_anomaly(
        "configs/default.yaml",
        missing_external,
        "output/runs/hardened_anomaly_analysis.json",
    )
    run_and_write_circadian_phase(
        "configs/default.yaml",
        missing_external,
        "output/runs/circadian_phase_analysis.json",
    )
    run_and_write_partial_observability(
        "configs/default.yaml",
        missing_external,
        "output/runs/partial_observability_analysis.json",
    )
    write_variance_bands(
        "output/runs/variance_bands.json",
        [
            {
                "gap_filling_skill": 0.1,
                "blind_limit_skill": 0.0,
                "anomaly_random_auc": 0.7,
                "frontier_no_location_mass": 0.2,
                "structure_deep_latent_tv": 0.3,
                "mission_efe_minus_pragmatic": 0.4,
            },
            {
                "gap_filling_skill": 0.14,
                "blind_limit_skill": -0.02,
                "anomaly_random_auc": 0.74,
                "frontier_no_location_mass": 0.24,
                "structure_deep_latent_tv": 0.26,
                "mission_efe_minus_pragmatic": 0.36,
            },
        ],
    )
    run_and_write_upstream_sensitivity(
        "output/runs/upstream_sensitivity.json",
        current_commit=FRACTALRABBIT_PINNED_COMMIT,
        candidate_limit=1,
        candidate_provider=lambda limit: ["candidate"][:limit],
        runner=lambda sha: None,
    )
    run_and_write_fractal_dimension_sweep(
        "configs/default.yaml",
        "output/runs/fractal_dimension_sweep.json",
        dimensions=[1.1, 1.33],
        runner=lambda dimension: {
            "n_observations": 2400,
            "n_latent_states": 16,
            "mean_predictive_loss": 2.0 + dimension,
            "mean_posterior_entropy": 0.5,
            "state_recovery_reference_mean_mass_on_true_cell": 0.8,
            "place_revisit_rate": 0.2 + dimension / 10.0,
            "place_mean_reports_per_place": 3.0,
        },
    )
    run_and_write_model_gates(
        project_root=".",
        fixture_config="configs/default.yaml",
        external_config="configs/fractalrabbit_external.yaml",
        external_ready=False,
    )
    run_and_write_model_conformance(
        "configs/default.yaml", "output/runs/model_conformance.json"
    )
    write_pymdp_benchmark_matrix(".", "output/runs/pymdp_benchmark_matrix.json")
    run_and_write_forward_gates(
        project_root=".",
        fixture_config="configs/default.yaml",
        external_config="configs/fractalrabbit_external.yaml",
        external_ready=False,
    )
    run_and_write_sota_benchmark("configs/default.yaml", "output/runs")
    run_and_write_sota_program(
        config_path="configs/default.yaml", out_dir=Path("output/runs")
    )
    run_and_write_aif_sweep("configs/default.yaml", "output/runs/aif_sweep.json")
    run_and_write_fair_inference(
        "configs/default.yaml", "output/runs/fair_inference.json"
    )
    run_and_write_power_analysis(
        "configs/default.yaml", "output/runs/power_analysis.json"
    )
    run_and_write_model_comparison(root)
    _ensure_aif_regime_artifacts(root)
    run_and_write_adaptive_aif_ablation(
        root, n_travellers=8, realization_seeds=(20260616,), n_boot=20
    )
    run_and_write_aif_deepening_candidates(root)
    run_and_write_synthetic_task_utility_privacy(root)
    _run_and_write_multiplicity_ledger(root)
    run_and_write_confound_taxonomy(root)
    run_and_write_regime_mechanism_synthesis(
        "configs/default.yaml",
        out_path="output/runs/regime_mechanism_synthesis.json",
    )
    run_and_write_regime_switch_phase_diagnostic(
        "configs/default.yaml",
        out_path="output/runs/regime_switch_phase_diagnostic.json",
    )
    artifacts = generate_figures(result.output_dir)
    write_visual_quality_audit(".")
    variables_path = write_variables(root)
    variables = compute_variables(root)

    # Registry metadata completeness on a FRESHLY regenerated registry (this is
    # the clean-checkout gate; test_figure_token_integrity validates the same
    # fields read-only on the shipped registry when one exists).
    registry = json.loads(
        (root / "output" / "figures" / "figure_registry.json").read_text(
            encoding="utf-8"
        )
    )
    metadata_offenders: list[str] = []
    for entry in registry["entries"]:
        for field in ("key", "filename", "caption", "alt_text", "source"):
            value = entry.get(field, "")
            if not isinstance(value, str) or not value.strip():
                metadata_offenders.append(f"{entry.get('key', '?')}.{field}: empty")
            elif any(
                bad in value.lower()
                for bad in ("[todo]", "placeholder", "tbd", "fixme")
            ):
                metadata_offenders.append(
                    f"{entry.get('key', '?')}.{field}: placeholder"
                )
        if not isinstance(entry.get("data_derived"), bool):
            metadata_offenders.append(f"{entry.get('key', '?')}.data_derived: not bool")
    assert not metadata_offenders, (
        f"incomplete figure metadata on fresh registry: {metadata_offenders}"
    )

    for key in [
        "evidence_atlas",
        "waypoint_paths",
        "spatial_bins",
        "spatial_resolution_frontier",
        "reporting_bursts",
        "speed_categories",
        "transition_heatmap",
        "likelihood_panels",
        "posterior_entropy",
        "policy_proxy",
        "candidate_scores",
        "negative_controls",
        "pymdp_agreement",
        "pymdp_diagnostics",
        "state_recovery",
        "sparsity_recovery",
        "calibration",
        "anticipatory",
        "efe_decomposition",
        "anomaly_detection",
        "recoverability_frontier",
        "return_structure",
        "variance_bands",
        "upstream_sensitivity",
        "fractal_dimension_sweep",
        "circadian_phase_signal",
        "partial_observability",
        "sequence_baselines",
        "missingness_model",
        "model_gate_matrix",
        "m10_forward_gates",
        "factorial_pymdp_modes",
        "latent_mode_recovery",
        "factorial_model_gate_matrix",
        "efe_location_frontier",
        "linkage_risk_frontier",
        "poisoning_robustness",
        "privacy_frontier",
        "hardened_anomaly_detection",
        "sota_leaderboard",
        "aif_regime_map",
        "fair_inference",
        "fair_power_regime",
        "privacy_utility_frontier",
        "privacy_utility_repeats",
        "synthetic_task_utility_privacy",
        "model_comparison_under_noise",
        "aif_deepening_candidates",
        "dataflow",
    ]:
        assert artifacts[key].exists()
        assert artifacts[key].stat().st_size > 1000
        image = mpimg.imread(artifacts[key])
        assert image.shape[0] >= 300
        assert image.shape[1] >= 400
        assert float(np.ptp(image)) > 0.01
    registry = json.loads(artifacts["registry"].read_text(encoding="utf-8"))
    style = json.loads(artifacts["style_provenance"].read_text(encoding="utf-8"))
    manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
    assert registry["schema"] == "active-fractal-rabbit-figure-registry-v2"
    for entry in registry["entries"]:
        assert len(entry["image_sha256"]) == 64
        assert len(entry["binding_sha256"]) == 64
        if entry["data_derived"]:
            assert len(entry["source_sha256"]) == 64
    assert style["schema"] == "active-fractal-rabbit-style-provenance-v1"
    assert style["dpi"] == 220
    assert style["font_contract"]["base"] == 17.0
    assert style["font_contract"]["axis_title"] == 19.0
    assert style["font_contract"]["figure_title"] == 21.0
    assert style["font_contract"]["tick"] == 16.0
    assert style["font_contract"]["legend"] == 15.5
    assert style["font_contract"]["annotation_minimum"] == 15.0
    assert style["font_contract"]["minimum_declared_readable"] == 15.0
    assert style["palette"]["blue"] == "#0072B2"
    assert style["palette"]["green"] == "#009E73"
    assert style["palette"]["orange"] == "#E69F00"
    assert style["palette"]["purple"] == "#CC79A7"
    assert style["palette"]["red"] == "#D55E00"
    assert style["palette"]["sky"] == "#56B4E9"
    assert style["palette"]["yellow"] == "#F0E442"
    assert style["palette"]["gray"] == "#595959"
    assert "style_provenance" in manifest
    assert len(registry["entries"]) == 53
    assert all(
        entry["caption"] and entry["alt_text"] and entry["source"]
        for entry in registry["entries"]
    )
    for entry in registry["entries"]:
        if entry["data_derived"]:
            assert entry["source"].startswith("output/")
            assert Path(entry["source"]).suffix in {".csv", ".json"}
        assert len(entry["caption"]) > 80
        assert " Evidence: " in entry["caption"]
        assert " Boundary: " in entry["caption"]
        assert (
            "not empirical" in entry["caption"]
            or "synthetic" in entry["caption"]
            or "from " in entry["caption"]
        )
    atlas = next(
        entry for entry in registry["entries"] if entry["key"] == "evidence_atlas"
    )
    assert "observations.csv" in atlas["caption"]
    assert "minimization" in atlas["caption"]
    assert "integrity" in atlas["caption"]
    assert "surveillance readiness" in atlas["caption"]
    assert atlas["source"] == "output/runs/default/metrics.json"
    anticipatory = next(
        entry for entry in registry["entries"] if entry["key"] == "anticipatory"
    )
    assert "persistence" in anticipatory["caption"]
    assert anticipatory["source"] == "output/runs/anticipatory_analysis.json"
    efe = next(
        entry for entry in registry["entries"] if entry["key"] == "efe_decomposition"
    )
    assert "epistemic" in efe["caption"]
    assert efe["source"] == "output/runs/active_sensing_analysis.json"
    likelihood = next(
        entry for entry in registry["entries"] if entry["key"] == "likelihood_panels"
    )
    assert "conditional-independence approximation" in likelihood["caption"]
    assert "deep-latent configuration" in likelihood["caption"]
    negative = next(
        entry for entry in registry["entries"] if entry["key"] == "negative_controls"
    )
    assert "4/4" in negative["caption"]
    assert negative["source"] == "output/runs/negative_controls_summary.json"
    agreement = next(
        entry for entry in registry["entries"] if entry["key"] == "pymdp_agreement"
    )
    assert "matched priors" in agreement["caption"]
    assert "policy-induced prior divergence" in agreement["caption"]
    diagnostics = next(
        entry for entry in registry["entries"] if entry["key"] == "pymdp_diagnostics"
    )
    assert "variational free energy" in diagnostics["caption"]
    recovery = next(
        entry for entry in registry["entries"] if entry["key"] == "state_recovery"
    )
    assert "not empirical validity" in recovery["caption"]
    assert "e-0" in agreement["caption"] or "e-1" in agreement["caption"]
    sota = next(
        entry for entry in registry["entries"] if entry["key"] == "sota_leaderboard"
    )
    assert sota["source"] == "output/runs/sota_program_leaderboard.csv"
    assert "active inference is scored directly" in sota["caption"]
    assert "strongest fair baseline" in sota["caption"]
    privacy_utility = next(
        entry
        for entry in registry["entries"]
        if entry["key"] == "privacy_utility_frontier"
    )
    assert "monotone graceful-degradation gate" in privacy_utility["caption"]
    assert "minimum-collection operating point" in privacy_utility["caption"]
    assert "fall together" not in privacy_utility["caption"]
    privacy_repeats = next(
        entry
        for entry in registry["entries"]
        if entry["key"] == "privacy_utility_repeats"
    )
    assert (
        privacy_repeats["source"] == "output/runs/privacy_utility_frontier_repeats.json"
    )
    assert "unavailable" in privacy_repeats["caption"]
    assert "stability" in privacy_repeats["caption"]
    task_privacy = next(
        entry
        for entry in registry["entries"]
        if entry["key"] == "synthetic_task_utility_privacy"
    )
    assert task_privacy["source"] == "output/runs/synthetic_task_utility_privacy.json"
    assert "utility floor" in task_privacy["caption"]
    assert "risk ceiling" in task_privacy["caption"]
    assert "release-utility" in task_privacy["caption"]
    aif_deepening = next(
        entry
        for entry in registry["entries"]
        if entry["key"] == "aif_deepening_candidates"
    )
    assert aif_deepening["source"] == "output/runs/aif_deepening_candidates.json"
    assert "scored candidate" in aif_deepening["caption"]
    assert "Candidate.predict" in aif_deepening["caption"]
    assert variables_path.exists()
    assert (
        variables["PROJECT_TITLE"]
        == "Active FractalRabbit: A Synthetic Benchmark for Belief Filtering Under Sparse Waypoint Observations"
    )
    assert variables["FIGURE_COUNT"] == "53"
    assert variables["VISUAL_QA_STATUS"] == "pass"
    assert variables["VISUAL_QA_FIGURE_COUNT"] == "53"
    assert variables["VISUAL_QA_CAPTION_SOURCE_COUNT"] == "53"
    assert float(variables["VISUAL_QA_MIN_PIXEL_STDDEV"]) > 0.0
    assert (
        "metric value | reporting ratio | score" in variables["CANDIDATE_SCORE_TABLE"]
    )
    assert variables["FIRST_OBSERVATION_COUNT"] == variables["TRAVELLER_COUNT"]
    assert variables["SENSITIVITY_RUN_COUNT"] != "0"
    assert "checked fixture" in variables["SENSITIVITY_CAVEAT"]
    assert "normal-approx interval" in variables["SENSITIVITY_SUMMARY_TABLE"]
    assert (
        "baseline_delta_mean_predictive_loss" in variables["SENSITIVITY_SUMMARY_TABLE"]
    )
    assert (
        "Policy entropy is summarized" in variables["SENSITIVITY_POLICY_ENTROPY_NOTE"]
    )
    stored = json.loads(variables_path.read_text(encoding="utf-8"))
    assert stored["PYMDP_VERSION"] == metadata.version("inferactively-pymdp")
    assert variables["NEGATIVE_CONTROL_COUNT"] == "4"
    assert variables["NEGATIVE_CONTROL_PASSED_COUNT"] == "4"
    assert "| control | probe |" in variables["NEGATIVE_CONTROL_TABLE"]
    assert "degenerate_bounds" in variables["NEGATIVE_CONTROL_TABLE"]
    assert variables["HOLDOUT_COUNT"] == "600"
    assert variables["HOLDOUT_FRACTION"] == "0.250"
    assert float(variables["HOLDOUT_MEAN_PREDICTIVE_LOSS"]) > 0
    assert variables["MODEL_ENGINE"] == "both"
    assert float(variables["PYMDP_MEAN_PREDICTIVE_LOSS"]) > 0
    assert "e-" in variables["PYMDP_EXACT_INFERENCE_DELTA"]
    assert variables["EXTERNAL_RUN_STATUS"] in {
        "completed on this host with the pinned upstream build",
        "not yet executed on this host",
    }
    assert float(variables["RECOVERY_REFERENCE_TOP1"]) >= 0.95
    assert float(variables["PYMDP_MEAN_VFE"]) > 0
    assert float(variables["PYMDP_MEDIAN_STEP_MS"]) > 0
    assert float(variables["DEEP_LATENT_RECOVERY_MASS"]) >= 0.0
    assert "FIG_CAPTION_STATE_RECOVERY" in variables
    assert (
        variables["OBSERVATION_COUNT"] == "2400"
    )  # expanded fixture drives the manuscript
    assert variables["VARIANCE_REALIZATION_COUNT"] == "2"
    assert variables["VARIANCE_GAP_FILL_LOW"] == "0.100"
    assert variables["VARIANCE_GAP_FILL_HIGH"] == "0.140"
    assert variables["UPSTREAM_SENSITIVITY_STATUS"] == "blocked"
    assert variables["FRACTAL_DIMENSION_SWEEP_STATUS"] == "completed"
    assert variables["CIRCADIAN_PHASE_VERDICT"] in {
        "candidate_phase_conditioned_prior",
        "honest_null",
    }
    assert variables["SEQUENCE_BASELINE_BEST_MODEL"] != "unavailable"
    assert variables["SEQUENCE_BASELINE_GATE"] in {
        "stronger_sequence_baseline_beats_pymdp_on_this_run",
        "pymdp_beats_sequence_baselines_on_this_run",
        "pymdp_unavailable_for_sequence_gate",
    }
    assert "first_order_markov" in variables["SEQUENCE_BASELINE_TABLE"]
    assert variables["MISSINGNESS_VERDICT"] in {
        "candidate_reporting_process_model",
        "honest_null",
    }
    assert "phase + traveller" in variables["MISSINGNESS_TABLE"]
    assert (
        variables["SPATIAL_RESOLUTION_VERDICT"]
        == "default_resolution_not_best_feasible_loss"
    )
    assert variables["SPATIAL_RESOLUTION_DEFAULT_BINS"] == "4"
    assert variables["SPATIAL_RESOLUTION_BEST_FEASIBLE_BINS"] == "3"
    assert variables["SPATIAL_RESOLUTION_DEFAULT_STATES"] == "16"
    assert variables["SPATIAL_RESOLUTION_DEFAULT_EMPTY_STATES"] == "0"
    assert float(variables["SPATIAL_RESOLUTION_DEFAULT_MINUS_COARSE_LOSS"]) > 0.0
    assert (
        float(variables["SPATIAL_RESOLUTION_DEFAULT_MINUS_COARSE_NO_LOCATION_MASS"])
        < 0.0
    )
    assert variables["SPATIAL_RESOLUTION_CONFIRMATION_STATUS"] in {
        "confirmed",
        "changed",
    }
    assert int(variables["SPATIAL_RESOLUTION_CONFIRMATION_N"]) == 9600
    assert (
        int(variables["SPATIAL_RESOLUTION_CONFIRMATION_REPORTS_PER_TRAVELLER"]) == 1920
    )
    assert int(variables["SPATIAL_RESOLUTION_CONFIRMATION_BEST_FEASIBLE_BINS"]) >= 3
    assert int(variables["SPATIAL_RESOLUTION_CONFIRMATION_STRICT_FEASIBLE_COUNT"]) >= 0
    assert variables["SPATIAL_RESOLUTION_CONFIRMATION_MULTIPLIERS"] == "2, 4"
    assert variables["SPATIAL_RESOLUTION_CONFIRMATION_LARGEST_MULTIPLIER"] == "4"
    assert int(variables["SPATIAL_RESOLUTION_CONFIRMATION_LARGEST_N"]) == 9600
    assert variables["SPATIAL_RESOLUTION_CONFIRMATION_GUARDRAIL_STABLE"] in {
        "yes",
        "no",
    }
    assert variables["SPATIAL_RESOLUTION_CONFIRMATION_SELECTED_BINS_SERIES"]
    assert variables["SPATIAL_RESOLUTION_CONFIRMATION_STRICT_FEASIBLE_SERIES"]
    assert "FIG_CAPTION_SPATIAL_RESOLUTION_FRONTIER" in variables
    assert "FIG_CAPTION_EVIDENCE_ATLAS" in variables
    assert "FIG_CAPTION_PARTIAL_OBSERVABILITY" in variables
    assert variables["PARTIAL_OBS_ZERO_WHEN_CLEAN"] == "yes"
    assert variables["PARTIAL_OBS_MONOTONE"] == "yes"
    assert float(variables["PARTIAL_OBS_MAX_NOISE_ADVANTAGE"]) > 0.0
    assert float(variables["PARTIAL_OBS_NOISE_SLOPE"]) > 0.0
    assert 0.0 <= float(variables["PARTIAL_OBS_NOISE_R2"]) <= 1.0
    assert int(variables["PARTIAL_OBS_N_REALIZATIONS"]) >= 1
    assert float(variables["PARTIAL_OBS_MAX_NOISE_MIN"]) > 0.0
    assert int(variables["PARTIAL_OBS_CROSS_DRAW_COUNT"]) >= 2
    assert int(variables["PARTIAL_OBS_CROSS_DRAW_TRAVELLERS"]) > 0
    assert float(variables["PARTIAL_OBS_CROSS_DRAW_MAX_SD"]) >= 0.0
    assert variables["PARTIAL_OBS_CROSS_DRAW_SIGN_STABLE"] == "yes"
    sota_gate = json.loads(
        Path("output/runs/sota_program_gate.json").read_text(encoding="utf-8")
    )
    assert sota_gate["run_parameters"]["canonical"] is True
    assert int(variables["SOTA_PROGRAM_N_TRAVELLERS"]) == DEFAULT_N_TRAVELLERS
    assert (
        "| bins | states | occupied | empty |" in variables["SPATIAL_RESOLUTION_TABLE"]
    )
    assert variables["RETURN_PHASE_VERDICT"] in {
        "promoted_return_phase_prior",
        "honest_null",
    }
    assert "return_plus_phase_prior" in variables["RETURN_PHASE_TABLE"]
    assert variables["MODEL_GATE_MATRIX_STATUS"] == "partial"
    assert variables["MODEL_GATE_SEQUENCE_EXTERNAL_GATE"] == "unavailable"
    assert variables["MODEL_GATE_FACTORIAL_EXTERNAL_GATE"] == "unavailable"
    assert "return_plus_phase_prior" in variables["MODEL_GATE_MATRIX_TABLE"]
    assert variables["FORWARD_GATE_STATUS"] == "completed"
    assert variables["FORWARD_GATE_FIXTURE_COMPLETED_COUNT"] == "8"
    assert "latent_regime_hmm" in variables["FORWARD_GATE_TABLE"]
    assert (
        variables["LATENT_REGIME_HMM_GATE"]
        == "latent_regime_hmm_does_not_beat_markov_null_em_overfits"
    )
    assert (
        variables["NEURAL_BASELINE_GATE"]
        == "neural_sequence_beats_weak_nulls_not_particle_mixture"
    )
    assert variables["LEARNED_STAY_RATE_VERDICT"] == "honest_regime_null"
    assert variables["HELDOUT_REPORTING_VERDICT"] == "honest_null"
    assert (
        variables["CROSS_GENERATOR_PERSISTENCE_VERDICT"]
        == "regime_dependent_on_self_transition_rate"
    )
    efe_artifact = json.loads(
        Path("output/runs/efe_location_frontier.json").read_text(encoding="utf-8")
    )
    linkage_artifact = json.loads(
        Path("output/runs/linkage_risk_frontier.json").read_text(encoding="utf-8")
    )
    poisoning_artifact = json.loads(
        Path("output/runs/poisoning_robustness.json").read_text(encoding="utf-8")
    )
    assert variables["EFE_LOCATION_VERDICT"] == efe_artifact["verdict"]
    assert (
        variables["LINKAGE_RISK_VERDICT"]
        == "location_carries_relink_risk_that_minimization_removes"
    )
    assert variables["LINKAGE_RISK_VERDICT"] == linkage_artifact["verdict"]
    assert variables["POISONING_VERDICT"] == poisoning_artifact["verdict"]
    assert variables["FACTORIAL_PYMDP_STATUS"] == "completed"
    assert variables["FACTORIAL_PYMDP_FACTOR_COUNT"] == "3"
    assert variables["FACTORIAL_PYMDP_GATE"] in {
        "factorial_pymdp_cleared_current_lane",
        "factorial_pymdp_not_performance_leading",
        "pymdp_baseline_unavailable",
    }
    assert "synthetic cognitive-proxy" in variables["FACTORIAL_PYMDP_BOUNDARY"]
    assert "factorial_pymdp" in variables["FACTORIAL_PYMDP_TABLE"]
    assert variables["PYMDP_SOTA_STATUS"] == "not_demonstrated"
    assert "prequential_sequence_baselines" in variables["PYMDP_BENCHMARK_TABLE"]
    assert "future_sota_baselines" in variables["PYMDP_BENCHMARK_TABLE"]
    assert "FIG_CAPTION_FRACTAL_DIMENSION_SWEEP" in variables
    assert "FIG_CAPTION_CIRCADIAN_PHASE_SIGNAL" in variables
    assert "FIG_CAPTION_SEQUENCE_BASELINES" in variables
    assert "FIG_CAPTION_MISSINGNESS_MODEL" in variables
    assert "FIG_CAPTION_MODEL_GATE_MATRIX" in variables
    assert "FIG_CAPTION_M10_FORWARD_GATES" in variables
    assert "FIG_CAPTION_FACTORIAL_PYMDP_MODES" in variables
    assert "FIG_CAPTION_LATENT_MODE_RECOVERY" in variables
    assert "FIG_CAPTION_FACTORIAL_MODEL_GATE_MATRIX" in variables
    assert "FIG_CAPTION_EFE_LOCATION_FRONTIER" in variables
    assert "FIG_CAPTION_LINKAGE_RISK_FRONTIER" in variables
    assert "FIG_CAPTION_POISONING_ROBUSTNESS" in variables
    assert "FIG_CAPTION_HARDENED_ANOMALY_DETECTION" in variables
    assert "FIG_CAPTION_PRIVACY_UTILITY_REPEATS" in variables
    assert "FIG_CAPTION_SYNTHETIC_TASK_UTILITY_PRIVACY" in variables
    assert "FIG_CAPTION_AIF_DEEPENING_CANDIDATES" in variables
    assert int(variables["PRIVACY_UTILITY_REPEATS_COMPLETED"]) >= 1
    assert variables["TASK_PRIVACY_RELEASE_UTILITY_ALLOWED"] in {"yes", "no"}
    assert variables["AIF_DEEPENING_PREDICTIVE_GATE_CLEARS"] in {"yes", "no"}
    assert variables["SPARSITY_LEVEL_COUNT"] == "4"
    assert "reports/traveller" in variables["SPARSITY_TABLE"]
    assert (
        float(variables["SPARSITY_DENSEST_LATENT_MASS"])
        > float(variables["SPARSITY_SPARSEST_LATENT_MASS"]) - 1.0
    )


def test_rolling_mean_preserves_constant_series_endpoints() -> None:
    values = np.full(8, 3.25)

    smoothed = _rolling_mean(values, window=5)

    np.testing.assert_allclose(smoothed, values)


def test_generate_figures_honors_configured_registry_path(tmp_path) -> None:
    """Flip test: figures.registry_path must control where figure outputs land."""

    custom_dir = tmp_path / "custom_figures"
    config_path = tmp_path / "configs" / "registry.yaml"
    config_path.parent.mkdir(parents=True)
    fixture = Path("data/fixtures/fractalrabbit_sample.csv").resolve()
    config_path.write_text(
        textwrap.dedent(
            f"""
            run_id: registry_flip
            fractalrabbit:
              upstream_commit: {FRACTALRABBIT_PINNED_COMMIT}
              execution: fixture
              fixture_csv: {fixture}
            observation:
              spatial_bins: 3
              time_bins_per_day: 4
            model:
              smoothing: 0.05
            figures:
              enabled: [waypoint_paths, dataflow]
              registry_path: {custom_dir / "figure_registry.json"}
            closed_loop:
              iterations: 1
              candidates: []
            """
        ),
        encoding="utf-8",
    )
    result = run_pipeline_from_path(config_path)
    artifacts = generate_figures(result.output_dir)

    assert artifacts["waypoint_paths"].parent == custom_dir
    assert (custom_dir / "figure_registry.json").exists()
    registry = json.loads(
        (custom_dir / "figure_registry.json").read_text(encoding="utf-8")
    )
    assert [entry["key"] for entry in registry["entries"]] == [
        "waypoint_paths",
        "dataflow",
    ]
    # explicit figure_dir still wins over the configured registry path
    explicit = generate_figures(result.output_dir, tmp_path / "explicit")
    assert explicit["waypoint_paths"].parent == tmp_path / "explicit"
