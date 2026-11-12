# Supplementary S2 - Reproducibility Runs Through Artifacts and Render Checks {#sec:reproducibility}

## Fixture-First Commands Rebuild the Study Surface {#sec:reproducibility-commands}

The reproducibility design is artifact-first and executable, following computational-research and software-citation guidance that favors versioned inputs, retained intermediates, scriptable workflows, plain-text project structure, and specific software identifiers [@sandve2013reproducible; @wilson2017good_enough; @smith2016software_citation]. The default reproducibility lane is fixture-first:

```bash
uv run python scripts/00_preflight.py
uv run python scripts/02_run_pipeline.py --config configs/default.yaml
uv run python scripts/03_run_sweep.py --config configs/sweep.yaml
uv run python scripts/05_run_negative_controls.py --config configs/default.yaml
uv run python scripts/08_run_sparsity_analysis.py --config configs/default.yaml
uv run python scripts/04_generate_figures.py --run output/runs/default
uv run python scripts/14_check_visual_quality.py
uv run python scripts/20_run_aif_regime_search.py
uv run python scripts/z_generate_manuscript_variables.py
uv run python scripts/11_check_manuscript_variable_provenance.py
uv run python scripts/15_check_claim_audit.py
uv run python scripts/16_check_visual_claim_audit.py
uv run python scripts/12_check_claim_ledger_artifacts.py
./run.sh test-full

# or run both lanes end to end (external lane runs when JDK 21 + Maven resolve):
uv run python scripts/07_run_full_study.py --lane both
```

The render lane runs from the sibling template checkout:

```bash
export TEMPLATE_REPO_ROOT=/path/to/template
uv run python scripts/z_generate_manuscript_variables.py
cd "$TEMPLATE_REPO_ROOT"
uv run python -m infrastructure.orchestration link-projects
uv run python scripts/pipeline/stage_03_render.py --project working/active_fractal_rabbit --skip-manuscript-hydration

# back in the project checkout, bind the rendered outputs to their exact inputs:
cd "$OLDPWD"
uv run python scripts/29_finalize_publication_package.py
uv run python scripts/29_finalize_publication_package.py --verify-only
```

## Code and Data Availability {#sec:reproducibility-availability}

The full source tree, checked fixtures, manuscript source, and generated release manifest are public at [github.com/ActiveInferenceInstitute/active_fractal_rabbit](https://github.com/ActiveInferenceInstitute/active_fractal_rabbit) under the licenses recorded in `LICENSE` (code, MIT) and `LICENSE-DOCS.md` (documentation and scholarly outputs, `CC-BY-4.0`). Third-party notices for the pinned external simulator and installed dependencies are recorded in `THIRD_PARTY_NOTICES.md`. A tagged GitHub release and an archival Zenodo deposit provide a versioned, citable snapshot; citation metadata is recorded in `CITATION.cff` and, once minted, the archival digital object identifier appears on the manuscript cover and in the Zenodo record linked from the repository.

## Inventories and Checksums Bind the Release Surface {#sec:reproducibility-artifacts}

The primary generated artifacts are grouped by role:

- **Run traces and tensors:** `observations.csv`, `discrete_features.csv`,
  `tensor_summary.json`, `posterior_states.csv`, `inference_trace.csv`,
  `policy_trace.csv`, `baseline_trace.csv`, `baseline_metrics.json`,
  `candidate_scores.json`, `pymdp_trace.csv`, `ground_truth.csv`, and
  `provenance_manifest.json`.
- **Fixture, robustness, and diagnostic analyses:** `sensitivity_summary.json`,
  `sensitivity_summary_external.json`, `negative_controls_summary.json`,
  `sparsity_analysis.json`, `spatial_resolution_frontier.json`,
  `anticipatory_analysis.json`, `active_sensing_analysis.json`,
  `anomaly_analysis.json`, `structure_recovery_analysis.json`,
  `mission_collection_analysis.json`, `frontier_analysis.json`,
  `partial_observability_analysis.json`, `return_structure_analysis.json`,
  `variance_bands.json`, `upstream_sensitivity.json`,
  `fractal_dimension_sweep.json`, `circadian_phase_analysis.json`,
  canonical `sequence_baseline_analysis.json`, lane-qualified
  `sequence_baseline_analysis_*.json`, canonical
  `missingness_model_analysis.json`, and lane-qualified
  `missingness_model_analysis_*.json`, `return_phase_prior_analysis*.json`,
  `model_gate_matrix.json`, `model_conformance.json`, `forward_gates_summary.json`,
  `latent_regime_hmm.json`, `neural_baseline.json`, `learned_stay_rate.json`,
  `heldout_reporting_process.json`, and `cross_generator_transfer.json`.
- **Safety, minimization, benchmark, and challenge surfaces:**
  `efe_location_frontier.json`, `linkage_risk_frontier.json`,
  `poisoning_robustness.json`, `pymdp_benchmark_matrix.json`,
  `sota_task_contract.json`, `sota_benchmark_matrix.json`,
  `sota_leaderboard.csv`, `sota_claim_gate.json`, `sota_program_gate.json`,
  `sota_program_leaderboard.csv`, `sota_program_robustness_grid.json`,
  `sota_program_robustness_grid.csv`, `aif_regime_search.csv`,
  `aif_regime_search.json`, `aif_regime_sota_gate.json`, `aif_sweep.json`,
  `aif_deepening_candidates.json`, `fair_inference.json`,
  `fair_inference_external_realizations.json`, `power_analysis.json`,
  `open_challenge_pack.json`, `multiplicity_ledger.json`,
  `confound_taxonomy.json`, `model_comparison_leaderboard.json`,
  `privacy_frontier_analysis.json`, `privacy_utility_frontier.json`,
  `privacy_utility_frontier_repeats.json`,
  `synthetic_task_utility_privacy.json`, and `hardened_anomaly_analysis.json`.
- **Manuscript, visual, and release contracts:** `figure_registry.json`,
  `active_fractal_rabbit_cover.png`, `visual_quality_audit.json`,
  `claim_audit.json`, `visual_claim_audit.json`, `manuscript_variables.json`,
  `manuscript_variable_provenance.json`, and `release_manifest.json`.

The manuscript values in [@sec:results] are hydrated from these artifacts. If a
metric changes, the rendered manuscript changes with it, and the provenance check
fails if a token source is missing, stale, incomplete, or byte-mismatched. The release
manifest additionally records a SHA-256 checksum and byte size for source modules,
scripts, tests, documentation, configs, fixtures, default-run artifacts, cover art,
figures, variable files, required release artifacts, and rendered outputs, so a
shipped bundle can be verified byte-for-byte and fails when a required or discovered
release entry is absent. The resolved manuscript embeds a content-addressed input
fingerprint into both PDF and HTML; final verification recomputes that fingerprint,
so changing a manuscript or visual input and merely updating output timestamps cannot
re-certify an unchanged render.
For reviewers who want to rebuild the headline scalars without rerunning the
whole simulator, `scripts/z_generate_manuscript_variables.py` rehydrates every
manuscript token from the retained run artifacts and
`scripts/11_check_manuscript_variable_provenance.py` verifies the source path,
JSON key, checksum, and rendered value behind each token.

The visual-quality audit reports `{{VISUAL_QA_STATUS}}` for `{{VISUAL_QA_FIGURE_COUNT}}` registered figures. Its minimum rendered extent is `{{VISUAL_QA_MIN_WIDTH}} x {{VISUAL_QA_MIN_HEIGHT}}` pixels, its minimum normalized pixel standard deviation is `{{VISUAL_QA_MIN_PIXEL_STDDEV}}`, and `{{VISUAL_QA_CAPTION_SOURCE_COUNT}}` captions carry source or boundary language. The audit also records aspect-ratio extremes and names the shortest, narrowest, lowest-variance, and widest/tallest-aspect figures so the contact sheet can be manually inspected where static layout risk is highest. This is not an aesthetic proof, but it prevents blank, undersized, caption-orphaned, source-orphaned, or silently hard-to-read figures from entering the manuscript unnoticed.

The figure layer treats visual design as part of the reproducibility contract rather than decoration. Quantitative comparisons use aligned axes, common-position encodings, and generated source tables where possible; comparison patterns are graphed only when the plot clarifies a relation that would be harder to read in a dense table; and uncertainty marks are paired with source artifacts plus explicit non-claim boundaries [@cleveland1984graphical_perception; @gelman2002tables_graphs; @munzner2014visualization_design; @hullman2020uncertainty; @spiegelhalter2011visualizing_uncertainty]. These sources justify the visual audit's reader-task, data-shape, encoding, and uncertainty checks. They do not make a figure self-validating: the manuscript claim still has to survive the artifact, variable-provenance, claim-ledger, and visual-claim audits.

The claim and visual-claim audits add the RedTeam layer above those mechanical checks. `claim_audit.json` inventories manuscript claims by section and verifies high-risk families against the claim ledger. `visual_claim_audit.json` records a mini-brief for every registered figure plus the cover, checking story job, reader task, source artifact, data shape, encoding, caption support, and explicit non-claim boundary before render.

{{ARTIFACT_PROVENANCE_TABLE}}

: Figure registry sources for the default generated figure set. {#tbl:figure-registry}

## The External Toolchain Boundary Stays Stochastic and Pinned {#sec:reproducibility-external-boundary}

External FractalRabbit runs require the toolchain boundary documented by preflight: Maven plus a JDK 21 runtime, which the boundary layer resolves automatically from the environment, PATH, Homebrew keg locations, or `java_home`. The external integration tests skip with explicit named reasons when that resolution fails, and run for real when it succeeds. Because the upstream simulator exposes no seed, external waypoint outputs are stochastic; the release manifest checksums pin the exact artifacts behind any given rendered manuscript.
