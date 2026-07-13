# Supplementary S6 - Fixture Diagnostics Bound Sensitivity and pymdp Agreement {#sec:results-sensitivity-pymdp}

This supplement audits the experimental fixture — sensitivity sweeps, negative controls, and pymdp implementation agreement — verifying the setup without advancing the headline claim ([@sec:results]).

## Sensitivity and Negative Controls Falsify Easy Wins {#sec:results-sensitivity-falsification}

The fixture sensitivity lane emits {{SENSITIVITY_RUN_COUNT}} runs across deterministic repetitions, report-count variants, spatial-bin settings, burst-rule settings, smoothing, and transition-control choices. This lane runs the sweep configuration (`configs/sweep.yaml`), which deliberately differs from the canonical default configuration: among other declared differences it configures a third `identity` transition control alongside the two canonical controls and uses a different spatial discretizer, seed, and holdout setting. Its policy-entropy and loss summaries therefore audit the sweep configuration and are not directly comparable to the canonical run's values reported later in this supplement. {{SENSITIVITY_CAVEAT}} The interval column in [@tbl:sensitivity-summary] is a normal-approximation interval over deterministic variants, not an inferential population confidence interval. {{SENSITIVITY_POLICY_ENTROPY_NOTE}}

{{SENSITIVITY_SUMMARY_TABLE}}

: Sensitivity summary emitted by `scripts/03_run_sweep.py`. {#tbl:sensitivity-summary}

The default run also reports per-traveller chronological holdout scoring with configured fraction {{HOLDOUT_FRACTION}}: {{HOLDOUT_COUNT}} of {{OBSERVATION_COUNT}} waypoint rows are scored by tensors estimated only from the remaining {{HOLDOUT_TRAIN_COUNT}} training rows, yielding holdout mean predictive loss {{HOLDOUT_MEAN_PREDICTIVE_LOSS}} nats and holdout mean posterior entropy {{HOLDOUT_MEAN_POSTERIOR_ENTROPY}} nats. Because the holdout pass is additive, the headline full-data metrics above are identical whether or not the knob is enabled.

![{{FIG_CAPTION_NEGATIVE_CONTROLS}}](../output/figures/negative_controls.png){#fig:negative-controls width={{FIGURE_WIDTH}}%}

The negative-control lane intentionally destroys one structural property at a time and checks a directional expectation against the resulting artifacts. [@fig:negative-controls] and [@tbl:negative-controls] report the current outcome: {{NEGATIVE_CONTROL_PASSED_COUNT}} of {{NEGATIVE_CONTROL_COUNT}} controls behaved as their falsification hypotheses require. {{NEGATIVE_CONTROL_CAVEAT}}

{{NEGATIVE_CONTROL_TABLE}}

: Negative-control outcomes emitted by `scripts/05_run_negative_controls.py`. {#tbl:negative-controls}

## pymdp Agreement Is an Implementation Check, Not a Win {#sec:results-pymdp-diagnostics}

![{{FIG_CAPTION_PYMDP_AGREEMENT}}](../output/figures/pymdp_agreement.png){#fig:pymdp-agreement width={{FIGURE_WIDTH}}%}

The real `pymdp` agent lane ([@fig:pymdp-agreement]) reports mean predictive loss {{PYMDP_MEAN_PREDICTIVE_LOSS}} nats and mean posterior entropy {{PYMDP_MEAN_POSTERIOR_ENTROPY}} nats on the default run. The matched-prior validation gives a maximum absolute state-posterior difference of {{PYMDP_EXACT_INFERENCE_DELTA}} against the exact Bayes posterior, so the two independent inference implementations agree at numerical precision; the closed-loop traces nevertheless diverge (maximum absolute posterior difference {{PYMDP_LOOP_POSTERIOR_DELTA}} on this run) through the policy-induced prior divergence mechanism described in [@sec:formal-model]. The matched-prior control localizes that divergence entirely to action selection.

![{{FIG_CAPTION_PYMDP_DIAGNOSTICS}}](../output/figures/pymdp_diagnostics.png){#fig:pymdp-diagnostics width={{FIGURE_WIDTH}}%}

[@fig:pymdp-diagnostics] exposes the full per-step diagnostic surface of the pinned agent: variational free energy (mean {{PYMDP_MEAN_VFE}} nats on the default run), wall-clock runtime (median {{PYMDP_MEDIAN_STEP_MS}} ms per step, maximum {{PYMDP_MAX_STEP_MS}} ms, {{PYMDP_TOTAL_RUNTIME_S}} s total against {{REFERENCE_RUNTIME_S}} s for the reference filter), and the expected free energy of every policy at every timestep. These numbers make the computational cost and the policy-evaluation dynamics of the active-inference layer auditable artifacts.

The performance-versus-interpretability answer from [@sec:pymdp-performance-interpretability] is summarized in [@tbl:pymdp-benchmark]. The current state-of-the-art performance status is `{{PYMDP_SOTA_STATUS}}`: `pymdp` changes predictive loss by {{PYMDP_PERFORMANCE_DELTA}} nats and posterior entropy by {{PYMDP_ENTROPY_DELTA}} nats relative to the transparent reference filter, so the current contribution is not a predictive-performance win. The diagnostic status is `{{PYMDP_INTERPRETABILITY_STATUS}}`, because the library lane exposes VFE, policy posterior, expected free energy, and action-conditioned prior dynamics as inspectable artifacts. The roadmap gates are: {{PYMDP_ROADMAP_GATES}}.

{{PYMDP_BENCHMARK_TABLE}}

: Current `pymdp` performance and interpretability matrix, emitted by `pymdp_benchmark_matrix.json`. {#tbl:pymdp-benchmark}

## Spatial-Resolution Diagnostics Choose the Reporting Grid {#sec:results-fixture-diagnostics}

The remaining sections are the supporting battery — the diagnostics, null hierarchy, gate verifications, and safety boundaries against which the headline result was tested. They begin with the fixture and discretization diagnostics that every later lane depends on.

The default fixture run writes {{RUN_ARTIFACT_COUNT}} run artifacts and {{FIGURE_COUNT}} registered figures. From this baseline, the measured summary is: {{OBSERVATION_COUNT}} observations, {{LATENT_STATE_COUNT}} latent states, mean predictive loss {{MEAN_PREDICTIVE_LOSS}}, final posterior entropy {{FINAL_POSTERIOR_ENTROPY}}, and mean policy entropy {{MEAN_POLICY_ENTROPY}}. The fixed-transition baseline filter uses `{{BASELINE_CONTROL}}`, yielding mean predictive loss {{BASELINE_MEAN_PREDICTIVE_LOSS}} and an active-minus-baseline loss delta of {{BASELINE_DELTA_MEAN_PREDICTIVE_LOSS}}. On this fixture, the fixed-transition baseline has lower mean predictive loss than the active policy-proxy trace, so the current active layer should be read as an auditable diagnostic rather than a performance improvement — and the delta has a mechanical explanation: with the proxy exactly tied under uniform preferences ([@sec:formal-policy-proxy]), tie-breaking selects the uniform `high_entropy_exploration` transition on a fraction {{POLICY_PROXY_EXPLORATION_SELECTED_FRACTION}} of steps, flattening the temporal prior that the fixed `{{BASELINE_CONTROL}}` baseline keeps sharp.

![{{FIG_CAPTION_WAYPOINT_PATHS}}](../output/figures/waypoint_paths.png){#fig:waypoint-paths width={{FIGURE_WIDTH}}%}

[@fig:waypoint-paths] shows that the fixture preserves the FractalRabbit-style sparse waypoint contract while remaining small enough for deterministic tests.

![{{FIG_CAPTION_SPATIAL_BINS}}](../output/figures/spatial_bins.png){#fig:spatial-bins width={{FIGURE_WIDTH}}%}

[@fig:spatial-bins] audits the spatial discretization used in [@eq:spatial-state]. The plotted cells are analysis bins, not FractalRabbit internal point-process states.

![{{FIG_CAPTION_SPATIAL_RESOLUTION_FRONTIER}}](../output/figures/spatial_resolution_frontier.png){#fig:spatial-resolution-frontier width={{FIGURE_WIDTH}}%}

[@fig:spatial-resolution-frontier] makes the discretization choice first-class. In [@fig:spatial-resolution-frontier], the artifact evaluates candidate square grids against occupancy, transition sparsity, full-location predictive loss, location-withheld recovery, and bounded real-`pymdp` matched-prior agreement. The default grid is {{SPATIAL_RESOLUTION_DEFAULT_BINS}}x{{SPATIAL_RESOLUTION_DEFAULT_BINS}}: {{SPATIAL_RESOLUTION_DEFAULT_STATES}} states with {{SPATIAL_RESOLUTION_DEFAULT_EMPTY_STATES}} empty cells, full-location loss {{SPATIAL_RESOLUTION_DEFAULT_FULL_LOSS}}, and default-vs-coarse loss change {{SPATIAL_RESOLUTION_DEFAULT_MINUS_COARSE_LOSS}}. On the canonical fixture the emitted verdict is `{{SPATIAL_RESOLUTION_VERDICT}}`, with best feasible bins {{SPATIAL_RESOLUTION_BEST_FEASIBLE_BINS}}. A deterministic sample-size ladder then regenerates {{SPATIAL_RESOLUTION_CONFIRMATION_MULTIPLIERS}} times the canonical reports per traveller from the same fixture generator family, with the largest {{SPATIAL_RESOLUTION_CONFIRMATION_LARGEST_MULTIPLIER}}x lane at n={{SPATIAL_RESOLUTION_CONFIRMATION_LARGEST_N}} ({{SPATIAL_RESOLUTION_CONFIRMATION_LARGEST_REPORTS_PER_TRAVELLER}} per traveller); the parallel largest-sample tokens resolve to n={{SPATIAL_RESOLUTION_CONFIRMATION_N}} and {{SPATIAL_RESOLUTION_CONFIRMATION_REPORTS_PER_TRAVELLER}} reports per traveller for the same lane. The largest lane reports `{{SPATIAL_RESOLUTION_CONFIRMATION_VERDICT}}`: status `{{SPATIAL_RESOLUTION_CONFIRMATION_STATUS}}`, strict feasible candidate count {{SPATIAL_RESOLUTION_CONFIRMATION_STRICT_FEASIBLE_COUNT}}, selected bins {{SPATIAL_RESOLUTION_CONFIRMATION_BEST_FEASIBLE_BINS}}, default-vs-coarse loss change {{SPATIAL_RESOLUTION_CONFIRMATION_DEFAULT_MINUS_COARSE_LOSS}}, and default-vs-coarse no-location mass change {{SPATIAL_RESOLUTION_CONFIRMATION_DEFAULT_MINUS_COARSE_NO_LOCATION_MASS}}. Across the ladder, verdict stability is {{SPATIAL_RESOLUTION_CONFIRMATION_LADDER_VERDICT_STABLE}}, no-location guardrail stability is {{SPATIAL_RESOLUTION_CONFIRMATION_GUARDRAIL_STABLE}}, selected bins are {{SPATIAL_RESOLUTION_CONFIRMATION_SELECTED_BINS_SERIES}}, strict feasible candidate counts are {{SPATIAL_RESOLUTION_CONFIRMATION_STRICT_FEASIBLE_SERIES}}, and the generated interpretation is `{{SPATIAL_RESOLUTION_CONFIRMATION_LADDER_INTERPRETATION}}`. When a strict feasible count is zero, the selected bins are the best measured loss among the candidate grids rather than a strict occupancy claim. The ladder refines the observed-location fit verdict at larger synthetic report counts, while the no-location guardrail keeps the same direction against fine-cell recovery (canonical default-vs-coarse mass change {{SPATIAL_RESOLUTION_DEFAULT_MINUS_COARSE_NO_LOCATION_MASS}}). It is not evidence that time, speed, and reporting-burst metadata recover fine spatial cells better, and it is no longer phrased as a universal best-fit grid.

That no-location guardrail is a *change* between the default and coarse discretizers. It is not the same scalar as the absolute deep-latent recovery mass in [@fig:state-recovery] or the densest sparsity recovery mass in [@fig:sparsity-recovery], so those values should not be compared as if they measured one regime.

{{SPATIAL_RESOLUTION_TABLE}}

: Spatial-resolution frontier emitted by `spatial_resolution_frontier.json`; the pymdp column is the matched-prior agreement check on deterministic sentinel rows. The sample-size ladder is emitted separately as `spatial_resolution_larger_sample_confirmation.json` so it can test the qualitative decision without changing the canonical fixture rows in this table. {#tbl:spatial-resolution-frontier}

![{{FIG_CAPTION_REPORTING_BURSTS}}](../output/figures/reporting_bursts.png){#fig:reporting-bursts width={{FIGURE_WIDTH}}%}

![{{FIG_CAPTION_SPEED_CATEGORIES}}](../output/figures/speed_categories.png){#fig:speed-categories width={{FIGURE_WIDTH}}%}

[@fig:reporting-bursts] and [@fig:speed-categories] inspect the derived temporal and motion categories from [@eq:interval-speed]. They are category diagnostics, not empirical claims about real movement speed.

![{{FIG_CAPTION_TRANSITION_HEATMAP}}](../output/figures/transition_heatmap.png){#fig:transition-heatmap width={{FIGURE_WIDTH}}%}

![{{FIG_CAPTION_LIKELIHOOD_PANELS}}](../output/figures/likelihood_panels.png){#fig:likelihood-panels width={{FIGURE_WIDTH}}%}

[@fig:transition-heatmap] and [@fig:likelihood-panels] expose the empirical structure that feeds [@eq:transition] and [@eq:likelihood]. These diagnostics are useful because normalization bugs in `A` or `B` can otherwise produce plausible-looking posterior traces.

![{{FIG_CAPTION_POSTERIOR_ENTROPY}}](../output/figures/posterior_entropy.png){#fig:posterior-entropy width={{FIGURE_WIDTH}}%}

![{{FIG_CAPTION_POLICY_PROXY}}](../output/figures/policy_proxy.png){#fig:policy-proxy width={{FIGURE_WIDTH}}%}

[@fig:posterior-entropy] reports the measured loss and uncertainty sequences from [@eq:predictive-loss] and [@eq:posterior-entropy]. [@fig:policy-proxy] reports the configured-control diagnostics from [@eq:policy-proxy] and [@eq:policy-posterior]. For the default fixture, {{POLICY_TIE_DIAGNOSTIC}}.

![{{FIG_CAPTION_CANDIDATE_SCORES}}](../output/figures/candidate_scores.png){#fig:candidate-scores width={{FIGURE_WIDTH}}%}

[@fig:candidate-scores] visualizes [@tbl:candidate-scores]. The selected next candidate is `{{SELECTED_NEXT_CANDIDATE}}`; the displayed formula is {{CANDIDATE_SCORE_FORMULA}}. It is deliberately a score explanation figure: it supports auditability of the next-run choice, not an external performance benchmark.
