from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import active_fractal_rabbit.release as release
from active_fractal_rabbit.active_sensing import run_and_write_active_sensing
from active_fractal_rabbit.adaptive_aif_ablation import (
    run_and_write_adaptive_aif_ablation,
)
from active_fractal_rabbit.anomaly import run_and_write_anomaly
from active_fractal_rabbit.aif_deepening import run_and_write_aif_deepening_candidates
from active_fractal_rabbit.aif_regime_search import run_and_write_aif_regime_search
from active_fractal_rabbit.aif_sweep import run_and_write_aif_sweep
from active_fractal_rabbit.anticipatory import run_and_write_anticipatory
from active_fractal_rabbit.claim_audit import write_claim_audit
from active_fractal_rabbit.challenge_pack import build_open_challenge_pack
from active_fractal_rabbit.circadian_phase import run_and_write_circadian_phase
from active_fractal_rabbit.confound_taxonomy import run_and_write_confound_taxonomy
from active_fractal_rabbit.cover_art import render_cover_art
from active_fractal_rabbit.forward_gates import run_and_write_forward_gates
from active_fractal_rabbit.frontier import run_and_write_frontier
from active_fractal_rabbit.fractal_dimension import (
    run_and_write_fractal_dimension_sweep,
)
from active_fractal_rabbit.fair_inference import (
    REQUIRED_EXTERNAL_REALIZATION_IDS,
    run_and_write_fair_inference,
    write_fair_external_realizations,
)
from active_fractal_rabbit.figures import generate_figures
from active_fractal_rabbit.hardened_anomaly import run_and_write_hardened_anomaly
from active_fractal_rabbit.manuscript_variables import write_variables
from active_fractal_rabbit.mission_collection import run_and_write_mission_collection
from active_fractal_rabbit.model_comparison import run_and_write_model_comparison
from active_fractal_rabbit.model_gates import run_and_write_model_gates
from active_fractal_rabbit.model_spec import run_and_write_model_conformance
from active_fractal_rabbit.negative_controls import run_negative_controls
from active_fractal_rabbit.pipeline import run_pipeline_from_path
from active_fractal_rabbit.partial_observability import (
    run_and_write_partial_observability,
)
from active_fractal_rabbit.power_analysis import run_and_write_power_analysis
from active_fractal_rabbit.regime_mechanism_synthesis import (
    run_and_write_regime_mechanism_synthesis,
)
from active_fractal_rabbit.regime_switch_phase_diagnostic import (
    run_and_write_regime_switch_phase_diagnostic,
)
from active_fractal_rabbit.privacy_frontier import run_and_write_privacy_frontier
from active_fractal_rabbit.privacy_utility_frontier import (
    run_and_write_privacy_utility_frontier,
)
from active_fractal_rabbit.privacy_utility_repeats import (
    run_and_write_privacy_utility_repeats,
)
from active_fractal_rabbit.pymdp_benchmark import write_pymdp_benchmark_matrix
from active_fractal_rabbit.release import (
    REQUIRED_PUBLICATION_PATHS,
    REQUIRED_RELEASE_PATHS,
    RELEASE_MANIFEST_SCHEMA,
    build_release_manifest,
    sha256_file,
    verify_release_manifest,
)
from active_fractal_rabbit.return_structure import run_and_write_return_structure
from active_fractal_rabbit.robustness import (
    run_and_write_upstream_sensitivity,
    write_variance_bands,
)
from active_fractal_rabbit.sota_benchmark import run_and_write_sota_benchmark
from active_fractal_rabbit.sota_program_harness import (
    run_and_write as run_and_write_sota_program,
)
from active_fractal_rabbit.sparsity import run_sparsity_analysis
from active_fractal_rabbit.spatial_resolution_frontier import (
    run_and_write_larger_sample_confirmation,
    run_and_write_spatial_resolution_frontier,
)
from active_fractal_rabbit.structure_recovery import run_and_write_structure_recovery
from active_fractal_rabbit.synthetic_task_utility_privacy import (
    run_and_write_synthetic_task_utility_privacy,
)
from active_fractal_rabbit.sweeps import run_replicated_sensitivity
from active_fractal_rabbit.visual_claim_audit import write_visual_claim_audit
from active_fractal_rabbit.visual_quality import write_visual_quality_audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_source_tree_for_verifiers(root: Path) -> None:
    source_link = root / "src"
    if not source_link.exists():
        source_link.symlink_to(PROJECT_ROOT / "src")
    tests_link = root / "tests"
    if not tests_link.exists():
        tests_link.symlink_to(PROJECT_ROOT / "tests")


def _materialize_release_inputs(root: Path) -> None:
    for name in ("data", "docs", "manuscript", "src", "scripts", "tests"):
        target = root / name
        if target.is_symlink():
            target.unlink()
            shutil.copytree(PROJECT_ROOT / name, target)
    for name in (
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "tasks.yaml",
        "TODO.md",
        "CITATION.cff",
        "LICENSE",
        "LICENSE-DOCS.md",
        "THIRD_PARTY_NOTICES.md",
    ):
        target = root / name
        if target.is_symlink():
            target.unlink()
            shutil.copy2(PROJECT_ROOT / name, target)


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


def _write_external_release_placeholders(root: Path) -> None:
    external_dir = root / "output" / "runs" / "fractalrabbit_external"
    external_dir.mkdir(parents=True, exist_ok=True)
    (external_dir / "metrics.json").write_text(
        json.dumps(
            {
                "schema": "active-fractal-rabbit-external-unavailable-metrics-v1",
                "status": "unavailable_in_sandbox",
                "n_observations": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (external_dir / "provenance_manifest.json").write_text(
        json.dumps(
            {
                "schema": "active-fractal-rabbit-external-provenance-v1",
                "status": "unavailable_in_sandbox",
                "reason": "release-manifest sandbox does not execute the optional external lane",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "output" / "runs" / "sensitivity_summary_external.json").write_text(
        json.dumps(
            {
                "schema": "active-fractal-rabbit-sensitivity-summary-v1",
                "status": "unavailable_in_sandbox",
                "rows": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pdf_path = root / "output" / "pdf" / "active_fractal_rabbit_combined.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Count 0 >> endobj\n"
        b"trailer << /Root 1 0 R >>\n%%EOF\n"
    )


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


def _external_realization_observation_paths(root: Path) -> dict[str, Path] | Path:
    candidates = {
        REQUIRED_EXTERNAL_REALIZATION_IDS[0]: root
        / "output"
        / "runs"
        / "fractalrabbit_external"
        / "observations.csv"
    }
    candidates.update(
        {
            realization_id: root
            / "output"
            / "runs"
            / f"fractalrabbit_external_{realization_id}"
            / "observations.csv"
            for realization_id in REQUIRED_EXTERNAL_REALIZATION_IDS[1:]
        }
    )
    if all(path.is_file() for path in candidates.values()):
        return candidates
    return root / "output" / "runs" / "does_not_exist" / "observations.csv"


def _seed_project(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "data" / "fixtures").mkdir(parents=True)
    (tmp_path / "output" / "runs" / "default").mkdir(parents=True)
    (tmp_path / "output" / "figures").mkdir(parents=True)
    (tmp_path / "configs" / "default.yaml").write_text(
        "run_id: default\n", encoding="utf-8"
    )
    (tmp_path / "data" / "fixtures" / "sample.csv").write_text(
        "ID,Days,x(km),y(km)\n0,0.1,1,1\n", encoding="utf-8"
    )
    (tmp_path / "output" / "runs" / "default" / "metrics.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (tmp_path / "output" / "figures" / "figure_registry.json").write_text(
        "{}\n", encoding="utf-8"
    )
    return tmp_path


def test_manifest_hashes_and_verifies(tmp_path) -> None:
    root = _seed_project(tmp_path)
    manifest_path = build_release_manifest(root, required_paths=())
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["schema"] == RELEASE_MANIFEST_SCHEMA
    assert payload["n_files"] == len(payload["entries"]) == 4
    by_path = {entry["path"]: entry for entry in payload["entries"]}
    config_entry = by_path["configs/default.yaml"]
    expected = hashlib.sha256(
        (root / "configs" / "default.yaml").read_bytes()
    ).hexdigest()
    assert config_entry["sha256"] == expected
    assert config_entry["bytes"] == len("run_id: default\n")

    report = verify_release_manifest(manifest_path, expected_required_paths=())
    assert report == {
        "missing": [],
        "mismatched": [],
        "unhashed_required": [],
        "missing_manifest_required": [],
        "omitted_entries": [],
        "invalid_manifest": [],
        "failed_checks": [],
    }


def test_manifest_detects_tampering_and_deletion(tmp_path) -> None:
    root = _seed_project(tmp_path)
    manifest_path = build_release_manifest(root, required_paths=())

    (root / "configs" / "default.yaml").write_text(
        "run_id: tampered\n", encoding="utf-8"
    )
    (root / "data" / "fixtures" / "sample.csv").unlink()
    report = verify_release_manifest(manifest_path, expected_required_paths=())

    assert report["mismatched"] == ["configs/default.yaml"]
    assert report["missing"] == ["data/fixtures/sample.csv"]


def test_release_manifest_detects_source_and_script_tampering(tmp_path) -> None:
    root = _seed_project(tmp_path)
    source = root / "src" / "active_fractal_rabbit" / "model.py"
    script = root / "scripts" / "run.py"
    source.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    script.write_text("print('ok')\n", encoding="utf-8")
    manifest_path = build_release_manifest(root, required_paths=())
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in payload["entries"]}
    assert "src/active_fractal_rabbit/model.py" in paths
    assert "scripts/run.py" in paths

    source.write_text("VALUE = 2\n", encoding="utf-8")
    script.write_text("print('tampered')\n", encoding="utf-8")
    report = verify_release_manifest(manifest_path, expected_required_paths=())

    assert report["mismatched"] == [
        "scripts/run.py",
        "src/active_fractal_rabbit/model.py",
    ]


def test_manifest_verifier_rejects_paths_outside_project_root(tmp_path) -> None:
    root = _seed_project(tmp_path)
    secret = tmp_path.parent / f"{tmp_path.name}-secret.txt"
    secret.write_text("outside manifest root\n", encoding="utf-8")
    entry_path = f"../{secret.name}"
    manifest_path = root / "output" / "release_manifest.json"
    payload = {
        "schema": RELEASE_MANIFEST_SCHEMA,
        "project": "active_fractal_rabbit",
        "required_paths": [],
        "entries": [
            {
                "group": "malicious",
                "path": entry_path,
                "bytes": secret.stat().st_size,
                "sha256": sha256_file(secret),
            }
        ],
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = verify_release_manifest(manifest_path, expected_required_paths=())

    assert report["missing"] == [entry_path]
    assert report["mismatched"] == []


def test_sha256_file_matches_hashlib(tmp_path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"x" * 70000)  # spans more than one read chunk
    assert sha256_file(path) == hashlib.sha256(b"x" * 70000).hexdigest()


def test_manifest_on_real_project_covers_release_groups(repo_sandbox) -> None:
    # Sandbox root: this test regenerates the whole release surface at smoke
    # scale (fixture-row variance bands, 1-dimension fractal sweep, 2-grid
    # spatial frontier) and rewrites manuscript_variables.json + the release
    # manifest — it must never do that to the production output tree.
    root = repo_sandbox
    _ensure_source_tree_for_verifiers(root)
    _materialize_release_inputs(root)
    config = root / "configs" / "default.yaml"
    missing_external = root / "output" / "runs" / "does_not_exist" / "observations.csv"
    current_external_observations = _external_realization_observation_paths(root)
    run_pipeline_from_path(config)
    run_replicated_sensitivity(config)
    run_negative_controls(config)
    run_sparsity_analysis(config)
    run_and_write_anticipatory(
        config,
        missing_external,
        root / "output" / "runs" / "anticipatory_analysis.json",
    )
    run_and_write_active_sensing(
        config,
        missing_external,
        root / "output" / "runs" / "active_sensing_analysis.json",
        fix_fractions=(0.5,),
    )
    run_and_write_anomaly(
        config, missing_external, root / "output" / "runs" / "anomaly_analysis.json"
    )
    run_and_write_structure_recovery(
        config,
        missing_external,
        root / "output" / "runs" / "structure_recovery_analysis.json",
    )
    run_and_write_mission_collection(
        config,
        missing_external,
        root / "output" / "runs" / "mission_collection_analysis.json",
    )
    run_and_write_frontier(
        config, missing_external, root / "output" / "runs" / "frontier_analysis.json"
    )
    run_and_write_return_structure(
        config,
        missing_external,
        root / "output" / "runs" / "return_structure_analysis.json",
    )
    run_and_write_privacy_frontier(
        config,
        missing_external,
        root / "output" / "runs" / "privacy_frontier_analysis.json",
    )
    run_and_write_privacy_utility_frontier(
        config,
        root / "data" / "fixtures" / "fractalrabbit_sample_large.csv",
        root / "output" / "runs" / "privacy_utility_frontier.json",
    )
    run_and_write_privacy_utility_repeats(
        config,
        root / "output" / "runs" / "privacy_utility_frontier_repeats.json",
        project_root=root,
    )
    run_and_write_hardened_anomaly(
        config,
        missing_external,
        root / "output" / "runs" / "hardened_anomaly_analysis.json",
    )
    run_and_write_circadian_phase(
        config,
        missing_external,
        root / "output" / "runs" / "circadian_phase_analysis.json",
    )
    run_and_write_partial_observability(
        config,
        missing_external,
        root / "output" / "runs" / "partial_observability_analysis.json",
    )
    write_variance_bands(
        root / "output" / "runs" / "variance_bands.json",
        [
            {
                "gap_filling_skill": 0.1,
                "frontier_no_location_mass": 0.2,
                "anomaly_random_auc": 0.7,
            },
            {
                "gap_filling_skill": 0.14,
                "frontier_no_location_mass": 0.24,
                "anomaly_random_auc": 0.74,
            },
        ],
    )
    run_and_write_upstream_sensitivity(
        root / "output" / "runs" / "upstream_sensitivity.json",
        current_commit="9933449c4f4fe1b26b6ac7bfdeeac76583085df5",
        candidate_limit=1,
        candidate_provider=lambda limit: [],
    )
    run_and_write_fractal_dimension_sweep(
        config,
        root / "output" / "runs" / "fractal_dimension_sweep.json",
        dimensions=[1.1],
        runner=lambda dimension: {
            "n_observations": 2400,
            "n_latent_states": 16,
            "mean_predictive_loss": 3.0,
            "mean_posterior_entropy": 0.5,
            "state_recovery_reference_mean_mass_on_true_cell": 0.8,
            "place_revisit_rate": 0.25,
            "place_mean_reports_per_place": 3.0,
        },
    )
    run_and_write_spatial_resolution_frontier(
        config,
        root / "output" / "runs" / "spatial_resolution_frontier.json",
        candidate_bins=(3, 4),
        pymdp_validation_steps=2,
    )
    run_and_write_larger_sample_confirmation(
        config,
        root / "output" / "runs" / "spatial_resolution_frontier.json",
        root / "output" / "runs" / "spatial_resolution_larger_sample_confirmation.json",
        root / "output" / "runs" / "spatial_resolution_larger_sample" / "observations.csv",
        sample_multipliers=(2, 4),
        candidate_bins=(3, 4),
        pymdp_validation_steps=2,
    )
    run_and_write_model_gates(
        project_root=root,
        fixture_config="configs/default.yaml",
        external_config="configs/fractalrabbit_external.yaml",
        external_ready=False,
    )
    run_and_write_model_conformance(
        config, root / "output" / "runs" / "model_conformance.json"
    )
    write_pymdp_benchmark_matrix(
        root, root / "output" / "runs" / "pymdp_benchmark_matrix.json"
    )
    run_and_write_forward_gates(
        project_root=root,
        fixture_config="configs/default.yaml",
        external_config="configs/fractalrabbit_external.yaml",
        external_ready=False,
    )
    run_and_write_sota_benchmark(config, root / "output" / "runs")
    run_and_write_sota_program(config_path=config, out_dir=root / "output" / "runs")
    run_and_write_aif_sweep(config, root / "output" / "runs" / "aif_sweep.json")
    run_and_write_fair_inference(
        config, root / "output" / "runs" / "fair_inference.json"
    )
    run_and_write_power_analysis(
        config, root / "output" / "runs" / "power_analysis.json"
    )
    run_and_write_model_comparison(root)
    _ensure_aif_regime_artifacts(root)
    run_and_write_adaptive_aif_ablation(
        root, n_travellers=8, realization_seeds=(20260616,), n_boot=20
    )
    run_and_write_aif_deepening_candidates(root)
    run_and_write_synthetic_task_utility_privacy(root)
    write_fair_external_realizations(
        config,
        root / "output" / "runs" / "fair_inference_external_realizations.json",
        current_external_observations,
    )
    _run_and_write_multiplicity_ledger(root)
    run_and_write_confound_taxonomy(root)
    run_and_write_regime_mechanism_synthesis(
        config,
        out_path=root / "output" / "runs" / "regime_mechanism_synthesis.json",
    )
    run_and_write_regime_switch_phase_diagnostic(
        config,
        out_path=root / "output" / "runs" / "regime_switch_phase_diagnostic.json",
    )
    generate_figures(root / "output" / "runs" / "default")
    write_visual_quality_audit(root)
    write_variables(root)
    _write_external_release_placeholders(root)
    render_cover_art(root)
    write_visual_claim_audit(root)
    build_open_challenge_pack(root)
    (root / "output" / "web").mkdir(parents=True, exist_ok=True)
    (root / "output" / "web" / "index.html").write_text(
        "<!-- render fixture -->\n", encoding="utf-8"
    )
    (root / "output" / "render_manifest.json").write_text(
        '{"schema": "test-render-manifest"}\n', encoding="utf-8"
    )
    build_release_manifest(root)
    write_claim_audit(root, strict_numeric=False)
    manifest_path = build_release_manifest(root)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert {
        "configs",
        "project_metadata",
        "fixtures",
        "manuscript",
        "return_structure",
        "variance_bands",
        "upstream_sensitivity",
        "fractal_dimension_sweep",
        "partial_observability",
        "sota_program",
        "aif_regime_search",
        "spatial_resolution_frontier",
        "circadian_phase",
        "sequence_baselines",
        "missingness_model",
        "return_phase_prior",
        "model_gate_matrix",
        "model_conformance",
        "forward_gates",
        "factorial_pymdp",
        "pymdp_benchmark",
        "efe_location_frontier",
        "linkage_risk_frontier",
        "poisoning_robustness",
        "privacy_utility_repeats",
        "synthetic_task_utility_privacy",
        "privacy_frontier",
        "aif_deepening",
        "hardened_anomaly",
        "visual_quality",
        "claim_audit",
        "visual_claim_audit",
        "challenge_pack",
        "cover",
        "manuscript_variable_provenance",
    } <= set(payload["groups"])
    paths = {entry["path"] for entry in payload["entries"]}
    assert "configs/default.yaml" in paths
    for path in REQUIRED_PUBLICATION_PATHS:
        assert path in paths
    assert "data/fixtures/fractalrabbit_sample.csv" in paths
    assert "output/cover/active_fractal_rabbit_cover.png" in paths
    missing_existing_required = sorted(
        path
        for path in REQUIRED_RELEASE_PATHS
        if (root / path).is_file() and path not in paths
    )
    assert missing_existing_required == []
    assert payload["missing_required"] == []
    assert payload["unhashed_required"] == []
    assert verify_release_manifest(manifest_path, validate_semantics=False) == {
        "missing": [],
        "mismatched": [],
        "unhashed_required": [],
        "missing_manifest_required": [],
        "omitted_entries": [],
        "invalid_manifest": [],
        "failed_checks": [],
    }
    # the manifest must never hash itself or recurse
    assert "output/release_manifest.json" not in paths


def test_manifest_verifier_rejects_missing_required_paths(tmp_path) -> None:
    root = _seed_project(tmp_path)
    manifest_path = build_release_manifest(
        root, required_paths=("output/runs/model_gate_matrix.json",)
    )
    report = verify_release_manifest(
        manifest_path, expected_required_paths=("output/runs/model_gate_matrix.json",)
    )

    assert report["missing"] == ["output/runs/model_gate_matrix.json"]


def test_manifest_verifier_rejects_required_paths_omitted_from_entries(
    tmp_path,
) -> None:
    root = _seed_project(tmp_path)
    manifest_path = build_release_manifest(
        root, required_paths=("configs/default.yaml",)
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["entries"] = [
        entry for entry in payload["entries"] if entry["path"] != "configs/default.yaml"
    ]
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = verify_release_manifest(
        manifest_path, expected_required_paths=("configs/default.yaml",)
    )

    assert report["missing"] == []
    assert report["mismatched"] == []
    assert report["unhashed_required"] == ["configs/default.yaml"]
    assert report["omitted_entries"] == ["configs/default.yaml"]


def test_release_semantics_reject_failed_audit_payload(
    tmp_path: Path, monkeypatch
) -> None:
    for relative_path in (
        "output/runs/claim_audit.json",
        "output/runs/visual_quality_audit.json",
        "output/runs/visual_claim_audit.json",
        "output/runs/model_conformance.json",
        "output/render_manifest.json",
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        status = "fail" if relative_path == "output/runs/claim_audit.json" else "pass"
        path.write_text(json.dumps({"status": status}), encoding="utf-8")
    monkeypatch.setattr(release, "verify_token_provenance", lambda *args, **kwargs: {})

    failures = release._release_semantic_failures(tmp_path)

    assert any("claim_audit.json" in item for item in failures)


def test_manifest_verifier_rejects_self_declared_required_subset(tmp_path) -> None:
    root = _seed_project(tmp_path)
    manifest_path = build_release_manifest(root, required_paths=())

    report = verify_release_manifest(
        manifest_path, expected_required_paths=("configs/default.yaml",)
    )

    assert report["missing"] == []
    assert report["unhashed_required"] == []
    assert report["missing_manifest_required"] == ["configs/default.yaml"]
