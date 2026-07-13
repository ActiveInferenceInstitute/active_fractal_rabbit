"""Release bundle manifest with content checksums.

The manifest records every release-relevant input and generated artifact with
size and SHA-256 so a reviewer can verify that a shipped bundle matches the
source configs, fixtures, and regenerated outputs byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from .variable_provenance import (
    DEFAULT_TOKEN_PROVENANCE,
    ProvenanceError,
    verify_token_provenance,
)

RELEASE_MANIFEST_SCHEMA = "active-fractal-rabbit-release-manifest-v1"

MANIFEST_GLOBS: tuple[tuple[str, str], ...] = (
    ("project_metadata", "README.md"),
    ("project_metadata", "pyproject.toml"),
    ("project_metadata", "uv.lock"),
    ("project_metadata", "TODO.md"),
    ("project_metadata", "tasks.yaml"),
    ("project_metadata", "LICENSE"),
    ("project_metadata", "LICENSE-DOCS.md"),
    ("project_metadata", "THIRD_PARTY_NOTICES.md"),
    ("project_metadata", "CITATION.cff"),
    ("project_metadata", "docs/ip_license_status.md"),
    ("documentation", "docs/**/*.md"),
    ("source", "src/**/*.py"),
    ("scripts", "scripts/*.py"),
    ("tests", "tests/**/*.py"),
    ("configs", "configs/**/*.yaml"),
    ("fixtures", "data/fixtures/*.csv"),
    ("manuscript", "manuscript/*.md"),
    ("manuscript", "manuscript/*.bib"),
    ("manuscript", "manuscript/config.yaml"),
    ("default_run", "output/runs/default/*"),
    ("negative_controls", "output/runs/negative_controls_summary.json"),
    ("sparsity", "output/runs/sparsity_analysis.json"),
    ("spatial_resolution_frontier", "output/runs/spatial_resolution_frontier.json"),
    (
        "spatial_resolution_frontier",
        "output/runs/spatial_resolution_larger_sample_confirmation.json",
    ),
    (
        "spatial_resolution_frontier",
        "output/runs/spatial_resolution_larger_sample/*",
    ),
    ("anticipatory", "output/runs/anticipatory_analysis.json"),
    ("active_sensing", "output/runs/active_sensing_analysis.json"),
    ("anomaly", "output/runs/anomaly_analysis.json"),
    ("structure_recovery", "output/runs/structure_recovery_analysis.json"),
    ("mission_collection", "output/runs/mission_collection_analysis.json"),
    ("frontier", "output/runs/frontier_analysis.json"),
    ("return_structure", "output/runs/return_structure_analysis.json"),
    ("variance_bands", "output/runs/variance_bands.json"),
    ("upstream_sensitivity", "output/runs/upstream_sensitivity.json"),
    ("fractal_dimension_sweep", "output/runs/fractal_dimension_sweep.json"),
    ("circadian_phase", "output/runs/circadian_phase_analysis.json"),
    ("partial_observability", "output/runs/partial_observability_analysis.json"),
    ("sota_benchmark", "output/runs/sota_task_contract.json"),
    ("sota_benchmark", "output/runs/sota_benchmark_matrix.json"),
    ("sota_benchmark", "output/runs/sota_leaderboard.csv"),
    ("sota_benchmark", "output/runs/sota_claim_gate.json"),
    ("sota_program", "output/runs/sota_program_gate.json"),
    ("sota_program", "output/runs/sota_program_leaderboard.csv"),
    ("sota_program", "output/runs/sota_program_robustness_grid.json"),
    ("sota_program", "output/runs/sota_program_robustness_grid.csv"),
    ("aif_regime_search", "output/runs/aif_regime_search.csv"),
    ("aif_regime_search", "output/runs/aif_regime_search.json"),
    ("aif_regime_search", "output/runs/aif_regime_sota_gate.json"),
    ("aif_sweep", "output/runs/aif_sweep.json"),
    ("fair_inference", "output/runs/fair_inference.json"),
    ("fair_inference", "output/runs/fair_inference_external_realizations.json"),
    ("power_analysis", "output/runs/power_analysis.json"),
    ("multiplicity_ledger", "output/runs/multiplicity_ledger.json"),
    ("confound_taxonomy", "output/runs/confound_taxonomy.json"),
    ("regime_mechanism_synthesis", "output/runs/regime_mechanism_synthesis.json"),
    (
        "regime_switch_phase_diagnostic",
        "output/runs/regime_switch_phase_diagnostic.json",
    ),
    ("privacy_utility_frontier", "output/runs/privacy_utility_frontier.json"),
    ("privacy_utility_repeats", "output/runs/privacy_utility_frontier_repeats.json"),
    (
        "synthetic_task_utility_privacy",
        "output/runs/synthetic_task_utility_privacy.json",
    ),
    ("model_comparison", "output/runs/model_comparison_leaderboard.json"),
    ("aif_deepening", "output/runs/aif_deepening_candidates.json"),
    ("adaptive_aif", "output/runs/adaptive_aif_ablation.json"),
    ("sequence_baselines", "output/runs/sequence_baseline_analysis.json"),
    ("sequence_baselines", "output/runs/sequence_baseline_analysis_*.json"),
    ("missingness_model", "output/runs/missingness_model_analysis.json"),
    ("missingness_model", "output/runs/missingness_model_analysis_*.json"),
    ("return_phase_prior", "output/runs/return_phase_prior_analysis*.json"),
    ("model_gate_matrix", "output/runs/model_gate_matrix.json"),
    ("model_conformance", "output/runs/model_conformance.json"),
    ("forward_gates", "output/runs/forward_gates_summary.json"),
    ("forward_gates", "output/runs/latent_regime_hmm*.json"),
    ("forward_gates", "output/runs/neural_baseline*.json"),
    ("forward_gates", "output/runs/learned_stay_rate*.json"),
    ("forward_gates", "output/runs/heldout_reporting_process*.json"),
    ("forward_gates", "output/runs/cross_generator_transfer*.json"),
    ("factorial_pymdp", "output/runs/factorial_pymdp_analysis*.json"),
    ("pymdp_benchmark", "output/runs/pymdp_benchmark_matrix.json"),
    ("efe_location_frontier", "output/runs/efe_location_frontier*.json"),
    ("linkage_risk_frontier", "output/runs/linkage_risk_frontier*.json"),
    ("poisoning_robustness", "output/runs/poisoning_robustness*.json"),
    ("privacy_frontier", "output/runs/privacy_frontier_analysis.json"),
    ("hardened_anomaly", "output/runs/hardened_anomaly_analysis.json"),
    ("visual_quality", "output/runs/visual_quality_audit.json"),
    ("claim_audit", "output/runs/claim_audit.json"),
    ("visual_claim_audit", "output/runs/visual_claim_audit.json"),
    ("challenge_pack", "output/runs/open_challenge_pack.json"),
    ("sensitivity", "output/runs/sensitivity_summary.*"),
    ("sensitivity", "output/runs/sensitivity_summary_external.*"),
    ("external_run", "output/runs/fractalrabbit_external/*"),
    ("figures", "output/figures/*"),
    ("cover", "output/cover/*"),
    ("variables", "output/data/manuscript_variables.json"),
    (
        "manuscript_variable_provenance",
        "output/data/manuscript_variable_provenance.json",
    ),
    ("pdf", "output/pdf/*.pdf"),
    ("render", "output/render_manifest.json"),
    ("web", "output/web/index.html"),
)

REQUIRED_PUBLICATION_PATHS: tuple[str, ...] = (
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "TODO.md",
    "tasks.yaml",
    "LICENSE",
    "LICENSE-DOCS.md",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "docs/ip_license_status.md",
    "docs/claim_ledger.md",
    "docs/generative_model_card.md",
    "docs/quickstart.md",
)

REQUIRED_ANALYSIS_PATHS: tuple[str, ...] = (
    "configs/default.yaml",
    "configs/fractalrabbit_external.yaml",
    "data/fixtures/fractalrabbit_sample_large.csv",
    "output/runs/default/metrics.json",
    "output/runs/fractalrabbit_external/metrics.json",
    "output/runs/negative_controls_summary.json",
    "output/runs/sparsity_analysis.json",
    "output/runs/spatial_resolution_frontier.json",
    "output/runs/spatial_resolution_larger_sample_confirmation.json",
    "output/runs/spatial_resolution_larger_sample/observations.csv",
    "output/runs/anticipatory_analysis.json",
    "output/runs/active_sensing_analysis.json",
    "output/runs/anomaly_analysis.json",
    "output/runs/structure_recovery_analysis.json",
    "output/runs/mission_collection_analysis.json",
    "output/runs/frontier_analysis.json",
    "output/runs/return_structure_analysis.json",
    "output/runs/variance_bands.json",
    "output/runs/upstream_sensitivity.json",
    "output/runs/fractal_dimension_sweep.json",
    "output/runs/circadian_phase_analysis.json",
    "output/runs/partial_observability_analysis.json",
    "output/runs/sota_task_contract.json",
    "output/runs/sota_benchmark_matrix.json",
    "output/runs/sota_leaderboard.csv",
    "output/runs/sota_claim_gate.json",
    "output/runs/sota_program_gate.json",
    "output/runs/sota_program_leaderboard.csv",
    "output/runs/sota_program_robustness_grid.json",
    "output/runs/sota_program_robustness_grid.csv",
    "output/runs/aif_regime_search.csv",
    "output/runs/aif_regime_search.json",
    "output/runs/aif_regime_sota_gate.json",
    "output/runs/aif_sweep.json",
    "output/runs/fair_inference.json",
    "output/runs/fair_inference_external_realizations.json",
    "output/runs/power_analysis.json",
    "output/runs/multiplicity_ledger.json",
    "output/runs/confound_taxonomy.json",
    "output/runs/regime_mechanism_synthesis.json",
    "output/runs/regime_switch_phase_diagnostic.json",
    "output/runs/privacy_utility_frontier.json",
    "output/runs/privacy_utility_frontier_repeats.json",
    "output/runs/synthetic_task_utility_privacy.json",
    "output/runs/model_comparison_leaderboard.json",
    "output/runs/aif_deepening_candidates.json",
    "output/runs/adaptive_aif_ablation.json",
    "output/runs/sequence_baseline_analysis.json",
    "output/runs/sequence_baseline_analysis_fixture.json",
    "output/runs/sequence_baseline_analysis_external.json",
    "output/runs/missingness_model_analysis.json",
    "output/runs/missingness_model_analysis_fixture.json",
    "output/runs/missingness_model_analysis_external.json",
    "output/runs/return_phase_prior_analysis.json",
    "output/runs/return_phase_prior_analysis_fixture.json",
    "output/runs/return_phase_prior_analysis_external.json",
    "output/runs/model_gate_matrix.json",
    "output/runs/model_conformance.json",
    "output/runs/forward_gates_summary.json",
    "output/runs/latent_regime_hmm.json",
    "output/runs/neural_baseline.json",
    "output/runs/learned_stay_rate.json",
    "output/runs/heldout_reporting_process.json",
    "output/runs/cross_generator_transfer.json",
    "output/runs/efe_location_frontier.json",
    "output/runs/linkage_risk_frontier.json",
    "output/runs/poisoning_robustness.json",
    "output/runs/factorial_pymdp_analysis_fixture.json",
    "output/runs/pymdp_benchmark_matrix.json",
    "output/runs/privacy_frontier_analysis.json",
    "output/runs/hardened_anomaly_analysis.json",
    "output/runs/visual_quality_audit.json",
    "output/runs/claim_audit.json",
    "output/runs/visual_claim_audit.json",
    "output/runs/open_challenge_pack.json",
    "output/figures/figure_registry.json",
    "output/cover/active_fractal_rabbit_cover.png",
    "output/data/manuscript_variables.json",
    "output/data/manuscript_variable_provenance.json",
)

REQUIRED_RELEASE_PATHS: tuple[str, ...] = (
    REQUIRED_PUBLICATION_PATHS
    + REQUIRED_ANALYSIS_PATHS
    + (
        "output/pdf/active_fractal_rabbit_combined.pdf",
        "output/web/index.html",
        "output/render_manifest.json",
    )
)


def build_release_manifest(
    project_root: str | Path,
    *,
    required_paths: tuple[str, ...] = REQUIRED_RELEASE_PATHS,
) -> Path:
    """Hash release-relevant files and write ``output/release_manifest.json``."""

    root = Path(project_root).resolve()
    entries: list[dict[str, object]] = []
    seen: set[Path] = set()
    for group, pattern in MANIFEST_GLOBS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            entries.append(
                {
                    "group": group,
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    entry_paths = {str(entry["path"]) for entry in entries}
    payload = {
        "schema": RELEASE_MANIFEST_SCHEMA,
        "project": "active_fractal_rabbit",
        "python": sys.version.split()[0],
        "n_files": len(entries),
        "groups": sorted({entry["group"] for entry in entries}),
        "required_paths": list(required_paths),
        "missing_required": [
            path for path in required_paths if not (root / path).is_file()
        ],
        "unhashed_required": [
            path
            for path in required_paths
            if (root / path).is_file() and path not in entry_paths
        ],
        "entries": entries,
    }
    output = root / "output" / "release_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def verify_release_manifest(
    manifest_path: str | Path,
    *,
    expected_required_paths: tuple[str, ...] | None = REQUIRED_RELEASE_PATHS,
    validate_semantics: bool | None = None,
) -> dict[str, list[str]]:
    """Re-hash entries and reject missing, omitted, or unhashed required paths."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    root = Path(manifest_path).resolve().parents[1]
    missing: list[str] = []
    mismatched: list[str] = []
    invalid_manifest: list[str] = []
    failed_checks: list[str] = []
    if manifest.get("schema") != RELEASE_MANIFEST_SCHEMA:
        invalid_manifest.append("schema")
    if manifest.get("project") != "active_fractal_rabbit":
        invalid_manifest.append("project")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return {
            "missing": [],
            "mismatched": [],
            "unhashed_required": [],
            "missing_manifest_required": [],
            "omitted_entries": [],
            "invalid_manifest": ["entries"],
            "failed_checks": [],
        }
    entry_path_list = [str(entry.get("path", "")) for entry in entries]
    entry_paths = set(entry_path_list)
    if not all(entry_path_list) or len(entry_paths) != len(entry_path_list):
        invalid_manifest.append("entry paths must be non-empty and unique")
    if manifest.get("n_files") != len(entries):
        invalid_manifest.append("n_files")
    recorded_groups = sorted({str(entry.get("group", "")) for entry in entries})
    if manifest.get("groups") != recorded_groups:
        invalid_manifest.append("groups")
    manifest_required = {str(path) for path in manifest.get("required_paths", [])}
    expected_required = (
        tuple(str(path) for path in expected_required_paths)
        if expected_required_paths is not None
        else tuple(str(path) for path in manifest.get("required_paths", []))
    )
    missing_manifest_required = [
        path for path in expected_required if path not in manifest_required
    ]
    unhashed_required: list[str] = []
    for required in expected_required:
        required_path = _contained_manifest_path(root, required)
        if required_path is None or not required_path.is_file():
            missing.append(required)
        elif required not in entry_paths:
            unhashed_required.append(required)
    for entry in entries:
        entry_path = str(entry.get("path", ""))
        path = _contained_manifest_path(root, entry_path)
        if path is None or not path.is_file():
            if entry_path not in missing:
                missing.append(entry_path)
            continue
        if path.stat().st_size != entry.get("bytes"):
            mismatched.append(entry_path)
            continue
        if sha256_file(path) != entry.get("sha256"):
            mismatched.append(entry_path)
    expected_entry_paths = {
        str(path.relative_to(root))
        for _group, pattern in MANIFEST_GLOBS
        for path in root.glob(pattern)
        if path.is_file()
    }
    omitted_entries = sorted(expected_entry_paths - entry_paths)
    if validate_semantics is None:
        validate_semantics = tuple(expected_required) == REQUIRED_RELEASE_PATHS
    if validate_semantics:
        failed_checks.extend(_release_semantic_failures(root))
    return {
        "missing": sorted(set(missing)),
        "mismatched": sorted(set(mismatched)),
        "unhashed_required": unhashed_required,
        "missing_manifest_required": missing_manifest_required,
        "omitted_entries": omitted_entries,
        "invalid_manifest": invalid_manifest,
        "failed_checks": failed_checks,
    }


def _release_semantic_failures(root: Path) -> list[str]:
    failures: list[str] = []
    required_statuses = {
        "output/runs/claim_audit.json": "pass",
        "output/runs/visual_quality_audit.json": "pass",
        "output/runs/visual_claim_audit.json": "pass",
        "output/runs/model_conformance.json": "pass",
        "output/render_manifest.json": "pass",
    }
    for relative_path, expected_status in required_statuses.items():
        path = root / relative_path
        if not path.is_file():
            failures.append(f"missing semantic release gate: {relative_path}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"unreadable semantic release gate {relative_path}: {exc}")
            continue
        if payload.get("status") != expected_status:
            failures.append(
                f"semantic release gate is not {expected_status}: {relative_path}"
            )
    try:
        from .claim_audit import audit_claims
        from .visual_claim_audit import audit_visual_claims
        from .visual_quality import audit_visual_quality

        live_audits = {
            "claim audit": audit_claims(root),
            "visual quality audit": audit_visual_quality(root),
            "visual claim audit": audit_visual_claims(root),
        }
        for label, payload in live_audits.items():
            if payload.get("status") != "pass":
                failures.append(f"live {label} failed")
    except Exception as exc:  # fail closed around malformed release inputs
        failures.append(f"live release audit failed: {exc}")
    provenance = root / "output/data/manuscript_variable_provenance.json"
    try:
        verify_token_provenance(
            provenance, expected_specs=DEFAULT_TOKEN_PROVENANCE
        )
    except (OSError, json.JSONDecodeError, KeyError, ProvenanceError) as exc:
        failures.append(f"manuscript variable provenance failed: {exc}")
    return failures


def _contained_manifest_path(root: Path, entry_path: str) -> Path | None:
    path = Path(entry_path)
    if path.is_absolute():
        return None
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
