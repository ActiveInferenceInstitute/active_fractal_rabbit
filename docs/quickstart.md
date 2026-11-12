# Quickstart: Fixture-First Commands Rebuild the Study Surface {#sec:quickstart}

## Fixture Lane Commands Produce Canonical Local Artifacts

Run the fixture lane first. This is the complete manuscript-facing path when
the external simulator is not part of the current evidence claim:

```bash
uv sync --extra dev
uv run python scripts/00_preflight.py
uv run python scripts/07_run_full_study.py --lane fixture
uv run python scripts/04_generate_figures.py --run output/runs/default
uv run python scripts/13_generate_cover_art.py
uv run python scripts/z_generate_manuscript_variables.py
uv run python scripts/10_check_eval_contract.py
uv run python scripts/11_check_manuscript_variable_provenance.py
uv run python scripts/12_check_claim_ledger_artifacts.py
uv run python scripts/14_check_visual_quality.py
uv run python scripts/15_check_claim_audit.py
uv run python scripts/16_check_visual_claim_audit.py
./run.sh test-full
export TEMPLATE_REPO_ROOT=/path/to/template
uv run python scripts/z_generate_manuscript_variables.py
cd "$TEMPLATE_REPO_ROOT"
uv run python -m infrastructure.orchestration link-projects
uv run python scripts/pipeline/stage_03_render.py --project working/active_fractal_rabbit --skip-manuscript-hydration
cd projects/working/active_fractal_rabbit
uv run python scripts/29_finalize_publication_package.py
uv run python scripts/29_finalize_publication_package.py --verify-only
```

The release-manifest step also verifies the root publication metadata:
`LICENSE`, `LICENSE-DOCS.md`, `THIRD_PARTY_NOTICES.md`, `CITATION.cff`,
`pyproject.toml`, `README.md`, and `docs/ip_license_status.md`.

For a smaller development loop that does not build every roadmap and
safety-boundary artifact, run the component stages directly:

```bash
uv run python scripts/02_run_pipeline.py --config configs/default.yaml
uv run python scripts/03_run_sweep.py --config configs/sweep.yaml
uv run python scripts/05_run_negative_controls.py --config configs/default.yaml
uv run python scripts/08_run_sparsity_analysis.py --config configs/default.yaml
uv run python scripts/04_generate_figures.py --run output/runs/default
uv run python scripts/14_check_visual_quality.py
uv run python scripts/z_generate_manuscript_variables.py
uv run python scripts/11_check_manuscript_variable_provenance.py
uv run python scripts/15_check_claim_audit.py
uv run python scripts/16_check_visual_claim_audit.py
```

## Manuscript-Facing Outputs Carry Provenance

The default run writes the following manuscript-facing artifacts under
`output/runs/default/`: `observations.csv`, `discrete_features.csv`,
`tensor_summary.json`, `posterior_states.csv`, `inference_trace.csv`,
`policy_trace.csv`, `baseline_trace.csv`, `baseline_metrics.json`,
`candidate_scores.json`, `config_snapshot.json`, and `provenance_manifest.json`.
The sweep command also writes `output/runs/sensitivity_summary.json` and
`output/runs/sensitivity_summary.csv`. The negative-control lane writes
`output/runs/negative_controls_summary.json` plus mutated fixture CSVs under
`output/negative_controls/`; run it before figure generation because the
`negative_controls` figure reads its summary. The full-study command also writes
the roadmap and safety-boundary artifacts consumed by the default 53-figure
registry: `spatial_resolution_frontier.json`, `anticipatory_analysis.json`,
`active_sensing_analysis.json`, `anomaly_analysis.json`,
`structure_recovery_analysis.json`, `mission_collection_analysis.json`,
`frontier_analysis.json`, `return_structure_analysis.json`,
`variance_bands.json`, `upstream_sensitivity.json`,
`fractal_dimension_sweep.json`, `circadian_phase_analysis.json`,
`partial_observability_analysis.json`, canonical
`sequence_baseline_analysis.json`, lane-qualified
`sequence_baseline_analysis_*.json`, canonical `missingness_model_analysis.json`,
and lane-qualified `missingness_model_analysis_*.json`,
`return_phase_prior_analysis*.json`, `model_gate_matrix.json`,
`model_conformance.json`, `pymdp_benchmark_matrix.json`,
`forward_gates_summary.json`, `efe_location_frontier.json`,
`linkage_risk_frontier.json`, `poisoning_robustness.json`,
`privacy_frontier_analysis.json`, `hardened_anomaly_analysis.json`,
`sota_task_contract.json`, `sota_benchmark_matrix.json`,
`sota_leaderboard.csv`, `sota_claim_gate.json`, `sota_program_gate.json`,
`sota_program_leaderboard.csv`, `sota_program_robustness_grid.json`,
`sota_program_robustness_grid.csv`, `aif_regime_search.csv`,
`aif_regime_search.json`, `aif_regime_sota_gate.json`, `aif_sweep.json`,
`aif_deepening_candidates.json`, `multiplicity_ledger.json`,
`confound_taxonomy.json`, `regime_mechanism_synthesis.json`,
`regime_switch_phase_diagnostic.json`,
`model_comparison_leaderboard.json`,
`privacy_utility_frontier.json`, `privacy_utility_frontier_repeats.json`,
`synthetic_task_utility_privacy.json`, `fair_inference.json`,
`fair_inference_external_realizations.json`, `power_analysis.json`,
`open_challenge_pack.json`, `visual_quality_audit.json`,
`claim_audit.json`, `visual_claim_audit.json`, and
`output/data/manuscript_variable_provenance.json`.

Figures, `figure_registry.json`, `figure_manifest.json`, and
`style_provenance.json` are written under `output/figures/`. The current default
registry contains 53 figures, beginning with `evidence_atlas.png` and ending
with `dataflow.png`. The cover generator writes
`output/cover/active_fractal_rabbit_cover.png`. Figure captions in the manuscript
are hydrated from the registry, so rerun figure generation and
`scripts/z_generate_manuscript_variables.py` after changing any visualization
writer, caption, or source artifact. Run `scripts/14_check_visual_quality.py`
before rendering when figure dimensions, aspect ratios, caption source language,
style provenance, or data-derived source paths change; its audit names the
contact sheet and worst-case figures for manual legibility review. The
release-manifest script writes checksums to `output/release_manifest.json`.
The finalization script must run after the sibling-template render: it records
`output/render_manifest.json`, checks that the PDF embeds the canonical cover,
hashes the PDF and web output, and only then verifies the release manifest.
Run `scripts/15_check_claim_audit.py` and `scripts/16_check_visual_claim_audit.py`
when manuscript claims, claim-ledger rows, captions, or the cover/figure
sequence changes. The visual-claim audit records each figure's story job,
reader task, data shape, encoding, and explicit non-claim boundary.

## Full-Study Orchestration Keeps Lanes Explicit

Run both fixture and external lanes when the host has the JDK 21 + Maven
toolchain and the current claim needs same-run external simulator evidence:

```bash
uv run python scripts/07_run_full_study.py --lane both
```

## Optional External Lane Requires JDK and Maven

Run the external FractalRabbit lane after installing JDK 21 and Maven
(`brew install openjdk@21 maven` — the boundary layer resolves the keg-only JDK
automatically; see `docs/external_lane.md`):

```bash
uv run python scripts/01_prepare_fractalrabbit.py --config configs/fractalrabbit_external.yaml
uv run python scripts/02_run_pipeline.py --config configs/fractalrabbit_external.yaml
```

## Template Rendering Happens from the Sibling Checkout

Render through the sibling template checkout:

```bash
# Point $TEMPLATE_REPO_ROOT at the sibling template checkout.
export TEMPLATE_REPO_ROOT=/path/to/template
uv run python scripts/z_generate_manuscript_variables.py
cd "$TEMPLATE_REPO_ROOT"
uv run python -m infrastructure.orchestration link-projects
uv run python scripts/pipeline/stage_03_render.py --project working/active_fractal_rabbit --skip-manuscript-hydration
cd projects/working/active_fractal_rabbit
uv run python scripts/29_finalize_publication_package.py
uv run python scripts/29_finalize_publication_package.py --verify-only
```
