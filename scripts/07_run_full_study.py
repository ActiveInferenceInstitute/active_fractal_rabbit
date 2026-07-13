"""Orchestrate the full study: fixture and/or real external FractalRabbit lanes.

Stage order per lane keeps the documented artifact contract: the external lane
writes its figures to ``output/figures_external/`` so the canonical fixture
figure registry (which the manuscript hydrates from) is never overwritten by a
lane with a different enabled-figure set.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import time
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.active_sensing import run_and_write_active_sensing
from active_fractal_rabbit.aif_deepening import run_and_write_aif_deepening_candidates
from active_fractal_rabbit.adaptive_aif_ablation import (
    run_and_write_adaptive_aif_ablation,
)
from active_fractal_rabbit.anomaly import run_and_write_anomaly
from active_fractal_rabbit.anticipatory import run_and_write_anticipatory
from active_fractal_rabbit.claim_audit import write_claim_audit
from active_fractal_rabbit.claim_ledger import verify_claim_ledger_artifacts
from active_fractal_rabbit.challenge_pack import build_open_challenge_pack
from active_fractal_rabbit.circadian_phase import run_and_write_circadian_phase
from active_fractal_rabbit.cover_art import render_cover_art
from active_fractal_rabbit.frontier import run_and_write_frontier
from active_fractal_rabbit.figures import generate_figures
from active_fractal_rabbit.forward_gates import run_and_write_forward_gates
from active_fractal_rabbit.fractal_dimension import (
    run_and_write_fractal_dimension_sweep,
)
from active_fractal_rabbit.hardened_anomaly import run_and_write_hardened_anomaly
from active_fractal_rabbit.model_gates import run_and_write_model_gates
from active_fractal_rabbit.model_spec import run_and_write_model_conformance
from active_fractal_rabbit.mission_collection import run_and_write_mission_collection
from active_fractal_rabbit.privacy_frontier import run_and_write_privacy_frontier
from active_fractal_rabbit.privacy_utility_frontier import (
    run_and_write_privacy_utility_frontier,
)
from active_fractal_rabbit.privacy_utility_repeats import (
    run_and_write_privacy_utility_repeats,
)
from active_fractal_rabbit.partial_observability import (
    run_and_write_partial_observability,
)
from active_fractal_rabbit.pymdp_benchmark import write_pymdp_benchmark_matrix
from active_fractal_rabbit.return_structure import run_and_write_return_structure
from active_fractal_rabbit.aif_regime_search import run_and_write_aif_regime_search
from active_fractal_rabbit.aif_sweep import run_and_write_aif_sweep
from active_fractal_rabbit.fair_inference import (
    required_external_realization_ids,
    run_and_write_fair_inference,
    write_fair_external_realizations,
)
from active_fractal_rabbit.power_analysis import run_and_write_power_analysis
from active_fractal_rabbit.multiplicity_ledger import run_and_write_multiplicity_ledger
from active_fractal_rabbit.confound_taxonomy import run_and_write_confound_taxonomy
from active_fractal_rabbit.regime_mechanism_synthesis import (
    run_and_write_regime_mechanism_synthesis,
)
from active_fractal_rabbit.regime_switch_phase_diagnostic import (
    run_and_write_regime_switch_phase_diagnostic,
)
from active_fractal_rabbit.model_comparison import run_and_write_model_comparison
from active_fractal_rabbit.sota_benchmark import run_and_write_sota_benchmark
from active_fractal_rabbit.sota_program_harness import (
    run_and_write as run_and_write_sota_program,
)
from active_fractal_rabbit.structure_recovery import run_and_write_structure_recovery
from active_fractal_rabbit.synthetic_task_utility_privacy import (
    run_and_write_synthetic_task_utility_privacy,
)
from active_fractal_rabbit.fractalrabbit import FractalRabbitRunner
from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.eval_contract import run_contract
from active_fractal_rabbit.manuscript_variables import write_variables
from active_fractal_rabbit.negative_controls import run_negative_controls
from active_fractal_rabbit.pipeline import run_pipeline, run_pipeline_from_path
from active_fractal_rabbit.preflight import external_lane_status, print_preflight
from active_fractal_rabbit.release import (
    REQUIRED_ANALYSIS_PATHS,
)
from active_fractal_rabbit.robustness import (
    candidate_commits_from_repo,
    make_upstream_candidate_runner,
    run_and_write_upstream_sensitivity,
    run_and_write_variance_bands,
)
from active_fractal_rabbit.sparsity import run_sparsity_analysis
from active_fractal_rabbit.spatial_resolution_frontier import (
    run_and_write_larger_sample_confirmation,
    run_and_write_spatial_resolution_frontier,
)
from active_fractal_rabbit.sweeps import run_replicated_sensitivity
from active_fractal_rabbit.variable_provenance import verify_token_provenance
from active_fractal_rabbit.visual_claim_audit import write_visual_claim_audit
from active_fractal_rabbit.visual_quality import write_visual_quality_audit

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lane", choices=["fixture", "external", "both"], default="both"
    )
    parser.add_argument("--fixture-config", default="configs/default.yaml")
    parser.add_argument("--sweep-config", default="configs/sweep.yaml")
    parser.add_argument(
        "--external-config", default="configs/fractalrabbit_external.yaml"
    )
    args = parser.parse_args()

    stages: list[tuple[str, float, str]] = []
    external_config = load_config(ROOT / args.external_config)

    def stage(name: str, fn) -> object:
        start = time.perf_counter()
        try:
            value = fn()
        except Exception as exc:
            stages.append((name, time.perf_counter() - start, f"FAIL: {exc}"))
            _report(stages)
            raise
        stages.append((name, time.perf_counter() - start, "ok"))
        return value

    stage("preflight", print_preflight)
    external_ready_this_run = False
    external_observation_paths: dict[str, Path] = {}

    if args.lane in {"external", "both"}:
        ready, reasons = external_lane_status()
        if not ready:
            print("external lane unavailable:")
            for reason in reasons:
                print(f"  - {reason}")
            exit_code = _external_unavailable_exit_code(
                args.lane,
                stop_on_external_unavailable=external_config.robustness.stop_on_external_unavailable,
            )
            if exit_code is not None:
                return exit_code
            print("continuing with fixture lane only")
        else:
            stage(
                "external:prepare",
                lambda: FractalRabbitRunner(external_config.fractalrabbit).prepare(),
            )
            external_pipeline_config = replace(
                external_config,
                inference=replace(
                    external_config.inference,
                    model=replace(external_config.inference.model, engine="reference"),
                ),
                figures=replace(
                    external_config.figures,
                    enabled=tuple(
                        name
                        for name in external_config.figures.enabled
                        if name not in {"pymdp_agreement", "pymdp_diagnostics"}
                    ),
                ),
            )
            external_result = stage(
                "external:pipeline",
                lambda: run_pipeline(
                    external_pipeline_config, ROOT / args.external_config
                ),
            )
            external_realization_ids = required_external_realization_ids(
                external_config.robustness.external_realization_fractal_dimensions
            )
            external_observation_paths[external_realization_ids[0]] = (
                external_result.observations_path
            )
            external_observation_paths |= stage(
                "external:fair_realization_repeats",
                lambda: _run_external_realization_repeats(
                    external_config, ROOT / args.external_config
                ),
            )
            stage(
                "external:sweep",
                lambda: run_replicated_sensitivity(ROOT / args.external_config),
            )
            stage(
                "external:figures",
                lambda: generate_figures(
                    external_result.output_dir, ROOT / "output" / "figures_external"
                ),
            )
            external_ready_this_run = True

    if args.lane in {"fixture", "both"}:
        fixture_result = stage(
            "fixture:pipeline",
            lambda: run_pipeline_from_path(ROOT / args.fixture_config),
        )
        # Canonical owner of output/runs/fixture_* (fixed_bounds, quantile_bins,
        # identity_ablation, fixed_burst, deep_latent): these runs feed the
        # DEEP_LATENT_* manuscript tokens and previously existed in production
        # only as a side effect of the test suite writing outside its sandbox.
        for _fixture_config in sorted((ROOT / "configs" / "fixtures").glob("*.yaml")):
            stage(
                f"fixture:config:{_fixture_config.stem}",
                lambda path=_fixture_config: run_pipeline_from_path(path),
            )
        stage(
            "fixture:sweep",
            lambda: run_replicated_sensitivity(ROOT / args.sweep_config),
        )
        stage(
            "fixture:negative_controls",
            lambda: run_negative_controls(ROOT / args.fixture_config),
        )
        stage(
            "fixture:sparsity",
            lambda: run_sparsity_analysis(ROOT / args.fixture_config),
        )
        stage(
            "fixture:spatial_resolution_frontier",
            lambda: run_and_write_spatial_resolution_frontier(
                ROOT / args.fixture_config,
                ROOT / "output" / "runs" / "spatial_resolution_frontier.json",
            ),
        )
        stage(
            "fixture:spatial_resolution_larger_sample",
            lambda: run_and_write_larger_sample_confirmation(
                ROOT / args.fixture_config,
                ROOT / "output" / "runs" / "spatial_resolution_frontier.json",
                ROOT
                / "output"
                / "runs"
                / "spatial_resolution_larger_sample_confirmation.json",
                ROOT
                / "output"
                / "runs"
                / "spatial_resolution_larger_sample"
                / "observations.csv",
            ),
        )
        _ext_obs = _external_observations_path(external_ready_this_run)
        stage(
            "fixture:anticipatory",
            lambda: run_and_write_anticipatory(
                ROOT / args.fixture_config,
                _ext_obs,
                ROOT / "output" / "runs" / "anticipatory_analysis.json",
            ),
        )
        stage(
            "fixture:active_sensing",
            lambda: run_and_write_active_sensing(
                ROOT / args.fixture_config,
                _ext_obs,
                ROOT / "output" / "runs" / "active_sensing_analysis.json",
            ),
        )
        stage(
            "fixture:anomaly",
            lambda: run_and_write_anomaly(
                ROOT / args.fixture_config,
                _ext_obs,
                ROOT / "output" / "runs" / "anomaly_analysis.json",
            ),
        )
        stage(
            "fixture:structure_recovery",
            lambda: run_and_write_structure_recovery(
                ROOT / args.fixture_config,
                _ext_obs,
                ROOT / "output" / "runs" / "structure_recovery_analysis.json",
            ),
        )
        stage(
            "fixture:mission_collection",
            lambda: run_and_write_mission_collection(
                ROOT / args.fixture_config,
                _ext_obs,
                ROOT / "output" / "runs" / "mission_collection_analysis.json",
            ),
        )
        stage(
            "fixture:frontier",
            lambda: run_and_write_frontier(
                ROOT / args.fixture_config,
                _ext_obs,
                ROOT / "output" / "runs" / "frontier_analysis.json",
            ),
        )
        stage(
            "fixture:return_structure",
            lambda: run_and_write_return_structure(
                ROOT / args.fixture_config,
                _ext_obs,
                ROOT / "output" / "runs" / "return_structure_analysis.json",
            ),
        )
        stage(
            "fixture:privacy_frontier",
            lambda: run_and_write_privacy_frontier(
                ROOT / args.fixture_config,
                _ext_obs,
                ROOT / "output" / "runs" / "privacy_frontier_analysis.json",
            ),
        )
        stage(
            "fixture:privacy_utility_frontier",
            lambda: run_and_write_privacy_utility_frontier(
                ROOT / args.fixture_config,
                _fixture_observations_path(ROOT / args.fixture_config),
                ROOT / "output" / "runs" / "privacy_utility_frontier.json",
            ),
        )
        stage(
            "fixture:privacy_utility_repeats",
            lambda: run_and_write_privacy_utility_repeats(
                ROOT / args.fixture_config,
                ROOT / "output" / "runs" / "privacy_utility_frontier_repeats.json",
                project_root=ROOT,
            ),
        )
        stage(
            "fixture:hardened_anomaly",
            lambda: run_and_write_hardened_anomaly(
                ROOT / args.fixture_config,
                _ext_obs,
                ROOT / "output" / "runs" / "hardened_anomaly_analysis.json",
            ),
        )
        stage(
            "fixture:circadian_phase",
            lambda: run_and_write_circadian_phase(
                ROOT / args.fixture_config,
                _ext_obs,
                ROOT / "output" / "runs" / "circadian_phase_analysis.json",
            ),
        )
        stage(
            "fixture:partial_observability",
            lambda: run_and_write_partial_observability(
                ROOT / args.fixture_config,
                _ext_obs,
                ROOT / "output" / "runs" / "partial_observability_analysis.json",
            ),
        )
        variance_config = ROOT / (
            args.external_config if external_ready_this_run else args.fixture_config
        )
        variance_cfg = load_config(variance_config)
        stage(
            "fixture:variance_bands",
            lambda: run_and_write_variance_bands(
                variance_config,
                ROOT / "output" / "runs" / "variance_bands.json",
                n_realizations=variance_cfg.robustness.n_realizations,
            ),
        )
        stage(
            "fixture:fractal_dimension_sweep",
            lambda: run_and_write_fractal_dimension_sweep(
                variance_config,
                ROOT / "output" / "runs" / "fractal_dimension_sweep.json",
            ),
        )
        runner = FractalRabbitRunner(external_config.fractalrabbit)
        upstream_runner = (
            make_upstream_candidate_runner(
                ROOT / args.external_config,
                ROOT / "output" / "runs" / "fractalrabbit_external" / "metrics.json",
            )
            if external_ready_this_run
            else None
        )
        stage(
            "fixture:upstream_sensitivity",
            lambda: run_and_write_upstream_sensitivity(
                ROOT / "output" / "runs" / "upstream_sensitivity.json",
                current_commit=external_config.fractalrabbit.upstream_commit,
                candidate_limit=external_config.robustness.upstream_candidate_limit,
                candidate_provider=lambda limit: candidate_commits_from_repo(
                    runner.source_dir,
                    external_config.fractalrabbit.upstream_commit,
                    limit,
                )
                if runner.source_dir.exists()
                else [],
                runner=upstream_runner,
            ),
        )
        stage(
            "fixture:model_gates",
            lambda: run_and_write_model_gates(
                project_root=ROOT,
                fixture_config=args.fixture_config,
                external_config=args.external_config,
                external_ready=external_ready_this_run,
            ),
        )
        stage(
            "fixture:model_conformance",
            lambda: run_and_write_model_conformance(
                ROOT / args.fixture_config,
                ROOT / "output" / "runs" / "model_conformance.json",
            ),
        )
        stage(
            "fixture:pymdp_benchmark",
            lambda: write_pymdp_benchmark_matrix(
                ROOT, ROOT / "output" / "runs" / "pymdp_benchmark_matrix.json"
            ),
        )
        stage(
            "fixture:forward_gates",
            lambda: run_and_write_forward_gates(
                project_root=ROOT,
                fixture_config=args.fixture_config,
                external_config=args.external_config,
                external_ready=external_ready_this_run,
            ),
        )
        stage(
            "fixture:sota_benchmark",
            lambda: run_and_write_sota_benchmark(
                ROOT / args.fixture_config, ROOT / "output" / "runs"
            ),
        )
        stage(
            "fixture:sota_program",
            lambda: run_and_write_sota_program(
                config_path=ROOT / args.fixture_config, out_dir=ROOT / "output" / "runs"
            ),
        )
        stage(
            "fixture:aif_regime_search",
            lambda: run_and_write_aif_regime_search(
                config_path=ROOT / args.fixture_config, out_dir=ROOT / "output" / "runs"
            ),
        )
        stage(
            "fixture:adaptive_aif_ablation",
            lambda: run_and_write_adaptive_aif_ablation(ROOT),
        )
        stage(
            "fixture:aif_deepening",
            lambda: run_and_write_aif_deepening_candidates(ROOT),
        )
        stage(
            "fixture:aif_sweep",
            lambda: run_and_write_aif_sweep(
                ROOT / args.fixture_config, ROOT / "output" / "runs" / "aif_sweep.json"
            ),
        )
        stage(
            "fixture:fair_inference",
            lambda: run_and_write_fair_inference(
                ROOT / args.fixture_config,
                ROOT / "output" / "runs" / "fair_inference.json",
                external_observation_paths or _ext_obs,
            ),
        )
        stage(
            "fixture:fair_external_realizations",
            lambda: write_fair_external_realizations(
                ROOT / args.fixture_config,
                ROOT / "output" / "runs" / "fair_inference_external_realizations.json",
                external_observation_paths or _ext_obs,
            ),
        )
        stage(
            "fixture:power_analysis",
            lambda: run_and_write_power_analysis(
                ROOT / args.fixture_config,
                ROOT / "output" / "runs" / "power_analysis.json",
            ),
        )
        # Multiplicity correction reads the regime + SOTA comparison artifacts; the confound
        # taxonomy reads the multiplicity ledger, power analysis, and partial-observability
        # artifacts. Both must run after those analyses and before figures/variables consume them.
        stage(
            "fixture:multiplicity_ledger",
            lambda: run_and_write_multiplicity_ledger(ROOT),
        )
        stage(
            "fixture:confound_taxonomy", lambda: run_and_write_confound_taxonomy(ROOT)
        )
        # Regime mechanism synthesis joins the AIF regime search and multiplicity ledger
        # artifacts above with a fresh per-regime confound-battery run; must run after both.
        stage(
            "fixture:regime_mechanism_synthesis",
            lambda: run_and_write_regime_mechanism_synthesis(
                ROOT / args.fixture_config,
                out_path=ROOT / "output" / "runs" / "regime_mechanism_synthesis.json",
                regime_search_path=ROOT / "output" / "runs" / "aif_regime_search.json",
                multiplicity_ledger_path=ROOT
                / "output"
                / "runs"
                / "multiplicity_ledger.json",
            ),
        )
        # Structural phase diagnostic re-slices the same regime-search per-step losses;
        # only depends on the regime-search dataset/scoring path, not on the two artifacts
        # above, but ordered alongside them since it belongs to the same mechanism-map family.
        stage(
            "fixture:regime_switch_phase_diagnostic",
            lambda: run_and_write_regime_switch_phase_diagnostic(
                ROOT / args.fixture_config,
                out_path=ROOT
                / "output"
                / "runs"
                / "regime_switch_phase_diagnostic.json",
            ),
        )
        # Within-regime model comparison ranks the SOTA-program candidate losses (where AIF leads);
        # must run after the SOTA program and before figures/variables consume the leaderboard.
        stage("fixture:model_comparison", lambda: run_and_write_model_comparison(ROOT))
        stage(
            "fixture:synthetic_task_utility_privacy",
            lambda: run_and_write_synthetic_task_utility_privacy(ROOT),
        )
        stage("fixture:figures", lambda: generate_figures(fixture_result.output_dir))
        stage("fixture:visual_quality", lambda: write_visual_quality_audit(ROOT))
        stage("fixture:cover_art", lambda: render_cover_art(ROOT))
        stage("fixture:visual_claim_audit", lambda: write_visual_claim_audit(ROOT))

    stage("manuscript_variables", lambda: write_variables(ROOT))
    stage(
        "variable_provenance",
        lambda: verify_token_provenance(
            ROOT / "output" / "data" / "manuscript_variable_provenance.json"
        ),
    )
    stage("challenge_pack", lambda: build_open_challenge_pack(ROOT))
    stage("eval_contract", _check_eval_contract)
    stage("claim_audit", lambda: write_claim_audit(ROOT))
    # The release manifest is deliberately deferred until after the sibling
    # template has rendered PDF and HTML. Building it here used to certify a
    # pre-render bundle whose embedded cover could already be stale.
    stage(
        "release_manifest_deferred",
        lambda: {
            "required_analysis_paths": len(
                REQUIRED_ANALYSIS_PATHS if args.lane != "external" else ()
            ),
            "next": "scripts/29_finalize_publication_package.py after template render",
        },
    )
    stage("claim_ledger_artifacts", lambda: verify_claim_ledger_artifacts(ROOT))

    _report(stages)
    return 0


def _report(stages: list[tuple[str, float, str]]) -> None:
    print("\nfull-study stages:")
    for name, seconds, status in stages:
        print(f"  {name:28s} {seconds:7.2f}s  {status}")


def _external_unavailable_exit_code(
    lane: str, *, stop_on_external_unavailable: bool
) -> int | None:
    if lane == "external":
        return 2
    if lane == "both" and stop_on_external_unavailable:
        return 2
    return None


def _external_observations_path(external_ready_this_run: bool) -> Path:
    if external_ready_this_run:
        return ROOT / "output" / "runs" / "fractalrabbit_external" / "observations.csv"
    return (
        ROOT
        / "output"
        / "runs"
        / "external_not_completed_this_run"
        / "observations.csv"
    )


def _fixture_observations_path(config_path: Path) -> Path:
    fixture_csv = load_config(config_path).fractalrabbit.fixture_csv
    if fixture_csv is None:
        raise SystemExit(
            f"fixture config {config_path} does not define fractalrabbit.fixture_csv"
        )
    path = Path(fixture_csv)
    return path if path.is_absolute() else ROOT / path


def _run_external_realization_repeats(
    external_config, external_config_path: Path
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    variants = tuple(
        (f"pinned_external_repeat_{idx}", fractal_dimension)
        for idx, fractal_dimension in enumerate(
            external_config.robustness.external_realization_fractal_dimensions,
            start=1,
        )
    )
    required_realization_ids = required_external_realization_ids(
        external_config.robustness.external_realization_fractal_dimensions
    )
    for realization_id, fractal_dimension in variants:
        if realization_id not in required_realization_ids:
            continue
        run_id = f"fractalrabbit_external_{realization_id}"
        params = replace(
            external_config.fractalrabbit.params, fractal_dimension=fractal_dimension
        )
        fractalrabbit = replace(
            external_config.fractalrabbit, params=params, output_prefix=run_id
        )
        config = replace(
            external_config,
            run_id=run_id,
            output_dir=ROOT / "output" / "runs" / run_id,
            seed=external_config.seed + len(paths) + 1,
            fractalrabbit=fractalrabbit,
            inference=replace(
                external_config.inference,
                model=replace(external_config.inference.model, engine="reference"),
            ),
        )
        result = run_pipeline(config, external_config_path)
        paths[realization_id] = result.observations_path
    return paths


def _check_eval_contract() -> dict[str, object]:
    report = run_contract(ROOT / "manuscript")
    if not report.passed:
        violations = [f"{item.guard}: {item.sentence}" for item in report.violations]
        raise RuntimeError("evaluation contract failed: " + "; ".join(violations))
    return {"passed": True}


if __name__ == "__main__":
    raise SystemExit(main())
