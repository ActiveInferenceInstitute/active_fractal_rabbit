# Configuration Schema: Fixture and Model Groups Own the Artifact Surface {#sec:config-schema}

This page documents the current YAML surface consumed by `active_fractal_rabbit.config.load_config`.
Configuration uses grouped `observation:` and `model:` blocks; the keys below are the required contract. A top-level `inference:` block is rejected.

## Config Groups Determine Defaults and Artifacts

| group | key | default or source | validation | affected artifacts |
| --- | --- | --- | --- | --- |
| root | `run_id` | config stem | string | run directory names, metadata |
| root | `seed` | `0` | integer | config snapshot, negative-control shuffles (`output/negative_controls/*.csv`) |
| root | `output_dir` | `output/runs/<run_id>` | path | all run artifacts |
| `authors[]` | `orcid` | none | ORCID iD text, e.g. `0000-0001-6232-9096` | PDF cover metadata |
| `paper` | `date` | empty | empty string for render-day `\today`, or explicit date text | PDF cover date |
| `paper.cover` | `image` | none | generated image path such as `output/cover/active_fractal_rabbit_cover.png` | PDF cover visual |
| `publication` | `doi` | none | assigned DOI string; supersedes `doi_status` | PDF cover metadata |
| `publication` | `doi_status` | none | `forthcoming` when a DOI will be assigned later | PDF cover metadata |
| `fractalrabbit` | `upstream_repo` | NSA FractalRabbit URL | string | provenance manifest |
| `fractalrabbit` | `upstream_commit` | pinned project commit | must equal project pin | provenance manifest, preflight boundary |
| `fractalrabbit` | `cache_dir` | `.cache/fractalrabbit` | path | external lane only |
| `fractalrabbit` | `execution` | `external` | `fixture` or `external` | source CSV resolution |
| `fractalrabbit` | `output_prefix` | `run_id` | string | external-lane simulator output file naming |
| `fractalrabbit` | `fixture_csv` | none | required for `fixture` | `observations.csv` |
| `fractalrabbit.params` | seven simulator parameters | `FractalRabbitParams` defaults | positive/range checks; 2D only | `parameters.csv`, candidate scoring |
| `observation` | `spatial_discretizer` | `grid` | `grid`, `fixed_bounds`, or `quantile` | `discrete_features.csv`, figures, tensors |
| `observation` | `spatial_bins` | none (required) | positive integer | latent-state count, A/B/D shapes, `spatial_resolution_frontier.json` |
| `observation` | `spatial_bounds` | none | four values with min < max; required for `fixed_bounds` | spatial bins |
| `observation` | `time_bins_per_day` | required positive integer | positive integer | time modality |
| `observation` | `speed_thresholds_km_per_day` | `[10, 50, 150]` default | exactly three ascending values | speed category and figure thresholds |
| `observation` | `burst_threshold_rule` | `median_positive` | `median_positive` or `fixed_days` | burst category and figure cutoff |
| `observation` | `burst_fixed_days` | none | positive when `fixed_days` | burst category |
| `model` | `smoothing` | required positive float | positive float | A/B/D smoothing |
| `model` | `controls` | empirical + high entropy | known control names only | B tensor, policy trace |
| `model` | `preference_prior` | `uniform` | `uniform`, `diagnostic_uncertainty`, or `diagnostic_gap` | C vectors (non-uniform values are consumed by the factorial pymdp lane's preference-vector builder) |
| `model` | `modalities` | all four (`location`, `time_of_day`, `speed`, `report_burst`) | nonempty unique subset, canonical order enforced | A/C tensors, observation indices, deep-latent lane |
| `model` | `engine` | `reference` | `reference` or `both` | `both` adds the real pymdp lane: `pymdp_trace.csv` + `pymdp_*` metrics |
| `model` | `factorization` | `single` | `single` or `factorial` | factorial pymdp lane selection |
| `model` | `hidden_factors` | `["cell", "movement_mode", "cognitive_proxy_mode"]` | must equal exactly that tuple | factorial pymdp latent-factor layout |
| `model` | `factorial_controls` | `["mode_persistence", "mode_reset"]` | nonempty subset of `mode_persistence`, `mode_reset`, `high_entropy_exploration`, `identity` | factorial pymdp control set |
| `experiment` | `repetitions` | `1` | positive integer | sensitivity replicate-family size (`sensitivity_summary.json` `n_runs`) |
| `experiment` | `holdout_fraction` | `0.0` | 0 to 0.95 | additive `holdout_*` keys in `metrics.json` (per-traveller chronological split) |
| `experiment` | `objective_direction` | `minimize` | `minimize` or `maximize` | candidate ordering |
| `experiment` | `tie_breaker` | `declaration_order` | `declaration_order` or `name` | candidate ordering |
| `figures` | `enabled` | default figure list | nonempty known keys; `negative_controls` requires `negative_controls_summary.json`; `pymdp_agreement` requires `pymdp_trace.csv` (`model.engine: both`) | `output/figures/*.png`, `figure_registry.json` |
| `figures` | `dpi` | `220` | positive integer | PNG render metadata |
| `figures` | `palette` | default semantic palette | nonempty mapping with `blue`, `green`, `orange`, `purple`, `red`, `sky`, `yellow`, and `gray` roles | PNG style and `style_provenance.json` |
| `figures` | `registry_path` | `output/figures/figure_registry.json` | path | figure output directory and registry location (explicit `figure_dir` argument still wins) |
| `closed_loop` | `iterations` | required | positive integer | closed-loop summary |
| `closed_loop` | `objective` | `predictive_loss` | `predictive_loss` or `posterior_entropy` | candidate scores |
| `closed_loop` | `objective_direction` | experiment direction | `minimize` or `maximize` | candidate ordering |
| `closed_loop` | `tie_breaker` | experiment tie-breaker | `declaration_order` or `name` | candidate ordering |
| `closed_loop.candidates` | `name`, partial `params` | none | candidate names plus valid simulator params | `candidate_scores.json` |
| `robustness` | `n_realizations` | `10` | positive integer | `variance_bands.json` realization count |
| `robustness` | `upstream_candidate_limit` | `5` | positive integer | `upstream_sensitivity.json` attempted prior commits |
| `robustness` | `external_realization_fractal_dimensions` | `[1.21, 1.45, 1.7]` | nonempty positive-float sequence | pinned external fair-ladder repeats in `fair_inference_external_realizations.json` |
| `robustness` | `stop_on_external_unavailable` | `true` | boolean | final-study external-lane confirmation policy |

## Run Artifacts Feed the Figure Registry Contract

The default fixture lane writes `config_snapshot.json` for the fully resolved dataclass state and `provenance_manifest.json` for the source CSV, upstream pin, and artifact map. Every run writes `ground_truth.csv` (per-step true cell, posterior mass on the
true cell, and top-1 correctness for each enabled inference lane; plus
`true_place_id` when the simulator's `*PLACES.csv` ground-truth stream is
present, which the pipeline auto-detects next to the XY output). The
`pymdp_trace.csv` artifact carries per-step variational free energy and wall
time alongside policy posteriors and per-policy expected free energy; runtime
keys (`*_runtime_*`) are the only intentionally nondeterministic values in
`metrics.json`. The figure keys `pymdp_diagnostics` and `state_recovery`
require `pymdp_trace.csv` and `ground_truth.csv` respectively.

The figure registry is part of the public configuration contract. Each enabled
figure must write a nonblank PNG, a registry entry with caption, alt text, source
artifact, and data-derived flag, and a style-provenance record with backend,
DPI, enabled-key list, palette, and readable font-size contract. The current
default figure list has 53 keys. It begins with `evidence_atlas`, which binds
the opening visual orientation to `observations.csv`, `discrete_features.csv`,
`metrics.json`, and gate artifacts; it ends with `dataflow`, the artifact
contract diagram. The visual-quality audit then records the registered figure
count, minimum rendered dimensions, aspect-ratio range, nonblank pixel variance,
caption source-language coverage, font contract, data-derived CSV/JSON
source-path existence, and a `quality_summary` naming the shortest, narrowest,
lowest-variance, widest-aspect, and tallest-aspect figures for contact-sheet
review in `output/runs/visual_quality_audit.json`. It also records a
`layout_critical_figures` roster for claim-bearing plots that require explicit
post-regeneration inspection for label collisions, clipping, legends, and title
stacking. The visual-claim audit adds
one mini-brief per figure plus the cover in
`output/runs/visual_claim_audit.json`, binding each visual to a story job,
reader task, source artifact, data shape, encoding, caption support statement,
and explicit non-claim boundary.
Manuscript Markdown should reference figures only through `{{FIG_CAPTION_*}}`
caption variables and `[@fig:...]` labels; hard numbered figure prose is
rejected by tests.

The default manuscript-facing figure keys are:

`evidence_atlas`, `waypoint_paths`, `spatial_bins`,
`spatial_resolution_frontier`, `reporting_bursts`, `speed_categories`,
`transition_heatmap`, `likelihood_panels`, `posterior_entropy`, `policy_proxy`,
`candidate_scores`, `negative_controls`, `pymdp_agreement`,
`pymdp_diagnostics`, `state_recovery`, `sparsity_recovery`, `calibration`,
`anticipatory`, `efe_decomposition`, `anomaly_detection`,
`recoverability_frontier`, `return_structure`, `variance_bands`,
`upstream_sensitivity`, `fractal_dimension_sweep`, `circadian_phase_signal`,
`partial_observability`, `sequence_baselines`, `missingness_model`,
`model_gate_matrix`, `m10_forward_gates`, `factorial_pymdp_modes`,
`latent_mode_recovery`, `factorial_model_gate_matrix`,
`efe_location_frontier`, `linkage_risk_frontier`, `poisoning_robustness`,
`privacy_frontier`, `hardened_anomaly_detection`, `sota_leaderboard`,
`aif_regime_map`, `fair_inference`, `fair_power_regime`, `multiplicity_ledger`,
`confound_taxonomy`, `regime_mechanism_synthesis`, `regime_switch_phase_diagnostic`,
`privacy_utility_frontier`, `privacy_utility_repeats`,
`synthetic_task_utility_privacy`, `model_comparison_under_noise`,
`aif_deepening_candidates`, and `dataflow`.

## Lane-Suffixed Artifacts Preserve Fixture and External Provenance

Sensitivity summaries are lane-suffixed: the fixture lane writes
`sensitivity_summary.json`/`.csv` and the external lane writes
`sensitivity_summary_external.json`/`.csv`, so the lanes never overwrite each
other. Robustness, safety-boundary, and roadmap stages add
`spatial_resolution_frontier.json`, `anticipatory_analysis.json`,
`active_sensing_analysis.json`, `anomaly_analysis.json`,
`structure_recovery_analysis.json`, `mission_collection_analysis.json`,
`frontier_analysis.json`, `return_structure_analysis.json`,
`variance_bands.json`, `upstream_sensitivity.json`,
`fractal_dimension_sweep.json`, `circadian_phase_analysis.json`,
`partial_observability_analysis.json`, canonical `sequence_baseline_analysis.json`,
lane-qualified `sequence_baseline_analysis_fixture.json` /
`sequence_baseline_analysis_external.json`, canonical `missingness_model_analysis.json`,
lane-qualified `missingness_model_analysis_fixture.json` /
`missingness_model_analysis_external.json`, `return_phase_prior_analysis*.json`,
`model_gate_matrix.json`, `model_conformance.json`, `pymdp_benchmark_matrix.json`,
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
`fair_inference_external_realizations.json`, `power_analysis.json`, `open_challenge_pack.json`,
`visual_quality_audit.json`, `claim_audit.json`, `visual_claim_audit.json`, and
`output/data/manuscript_variable_provenance.json`; the last file records
token-to-artifact checksums before rendering.

## Nested Config Resolution Keeps Overrides Explicit

Nested config folders are supported: `load_config` walks upward from the config
file to the directory containing `pyproject.toml` to resolve the project root,
so the checked variants under `configs/fixtures/` (fixed-bound grid, quantile
bins, identity-control ablation, fixed burst threshold, deep latent) resolve fixture and
output paths exactly like top-level configs.
