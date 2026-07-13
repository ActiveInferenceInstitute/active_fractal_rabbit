# Major Revision IMRaD Restructure: Benchmark-First, Results-Driven Movement

> **IMPLEMENTATION STATUS (shipped 2026-06-21, commits `0f71244` + `fe88e0e`).** This plan was
> executed and extended into a full main-line/supplement partition. The mapping tables further down
> describe an earlier, more conservative draft ("UNCHANGED EXCEPT TITLE CLARITY") and are kept as
> historical record; the structure **as shipped on 2026-06-21** was:
>
> - **Main-line IMRAD spine** (filename-ordered): `00_abstract`, `01_introduction`,
>   `02_methodology` (Methods), `02b_formal_model` (generative model + partial-observability
>   protocol + policy-proxy), `03_results`, `03a_results_partial_observability_and_model_comparison`
>   (the positive regime + within-regime model comparison), `03b_results_null_hierarchy_and_multiplicity`,
>   `04_discussion`, `05_limitations`, `07_roadmap`, `08_conclusion`, `99_references`.
> - **Supplements S01–S08**: S01 Validation & Threat Model · S02 Reproducibility · S03 Simulator
>   Boundary (was `02a`) · S04 Formal Diagnostics — robustness/provenance + pymdp boundary (was `02ce`
>   + carved `02c` tail) · S05 Closed-Loop Extensions (was `02d`) · S06 Fixture Diagnostics (carved
>   `03a` fixture tail + was `03b` sensitivity) · S07 Program Extensions (carved `03c`
>   anticipatory/EFE/program + was `03d`) · S08 Safety/Privacy/External (was `03e`).
> - Every `{#sec:}` label was preserved (cross-refs resolve by label); titles were sharpened to be
>   scope-accurate. Validated by `manuscript_integrity` (render + label-set + dangling-ref +
>   every-figure-referenced + required-substrings) and `docs_contract`.
>
> The manuscript has evolved since this snapshot (e.g. supplement `S09_confound_taxonomy` was added
> later); the `manuscript/` directory itself is authoritative for the current structure. This
> banner and the plan below are a historical record, not a description of the current manuscript.

# Manuscript Restructuring Plan: IMRaD Reorientation with Results-Bearing Titles

## Executive Summary

This plan restructures *active_fractal_rabbit* from a diagnostic-focused layout into a crisp **IMRaD spine** (Introduction / Methods / Results / Discussion) that repositions the work as a **benchmark/evaluation paper first, active-inference second**. Key moves:

1. **Reposition Abstract/Introduction**: End abstract with benchmark contribution; open introduction with "why synthetic benchmarking for sparse-mobility inference?"; close introduction with active-inference as one instrument within a broader evaluation framework.
2. **Promote fair-oracle-stripped comparison + locked-split logic to Results *immediately***—move before ``sec:results-fixture-diagnostics``.
3. **Add early Results comparison-contract TABLE** showing every model's information set, task, oracle status, loss, calibration, verdict.
4. **Move diagnostic/robustness results to Supplementary S03, S04** to reduce main-text figures (~4 core remain: benchmark schematic, model-comparison table/figure, fair-predictor ladder, one minimization/privacy frontier).
5. **Reorganize Discussion** (current 05_limitations + 07_roadmap + 08_conclusion → new Discussion spine): Honest boundaries → roadmap gates → final claim boundary.
6. **Rename section titles** throughout to be result-bearing (encode direction, not just topic).

---

## Current Structure & Mapping

### Current Backbone
- **00_abstract.md**: Mentions both persists-oracle limitation AND fair-comparison win; scattered.
- **01_introduction.md** (~34 lines; 3 subsections): Synthetic benchmark → variational diagnostics → contribution & boundary. Good scaffolding; needs re-emphasis on benchmark as primary.
- **02_methodology.md** (+ 02a–02d subsections): Artifact loop, configuration, roadmap-as-stages. Stays mostly unchanged; clarity improvement needed.
- **03_results.md** (VERY LONG): 11 subsections mixing fixture diagnostics, nulls, pymdp, recovery, sensitivity, roadmap gates, forwards gates, factorial, safety, privacy, external lane. **CORE MOVE SOURCE**.
- **05_limitations.md**: Honest nulls and synthetic boundaries. **MOVES to Discussion tier**.
- **07_roadmap.md**: Forward gates, empirical validation, AIF deepening, fair-inference ladder, power analysis. **SPLITS: key results to 03_results, forward-only to Discussion/Supplements**.
- **08_conclusion.md**: Wraps pymdp, recovery, roadmap, boundary. **INTEGRATES into Discussion**.

### Supplementaries (Existing)
- **S01_validation_and_threat_model.md**: Stays as S01; possibly renamed to encode "Evaluation Contract & Safety Bounds."
- **S02_reproducibility.md**: Stays as S02; possibly renamed to encode "Artifact Provenance & Reproducibility."

---

## Proposed New Structure with Informative Titles

### MAIN BODY (IMRaD)

#### **00_abstract.md** (REVISED)
- **OLD**: "Central result is *where active inference helps*…"
- **NEW FRAMING**: Rewrite to end on benchmark contribution + safety-boundary innovation rather than split-personality message.
  - Opening: sparse waypoints, FractalRabbit synthetic testbed, need for explicit nulls & minimization.
  - Core: **fair oracle-stripped comparison** reveals soft Bayesian marginalization as dominant gain; AIF lowest-loss fair predictor but does not beat oracle-fed persistence baseline (honest frontier).
  - Close: benchmark gate (not_demonstrated on oracle baseline) BUT artifact-bound honesty discipline and minimization frontier are the signature contribution.

#### **01_introduction.md** (MINOR REORGANIZATION & RETITLED SUBSECTIONS)
- H2 1: ~"Building Synthetic Benchmarks for Sparse-Mobility Inference" (was "Synthetic Benchmarking Without Surveillance Claims")
  - Stronger emphasis: benchmarking *first*, not surveillance-avoidance *first*. Surveillance-avoidance is a consequence of artifact-first design.
- H2 2: ~"Variational Diagnostics Expose Uncertainty and Minimization" (was "Why Variational Diagnostics Matter Here")
  - Reword to emphasize active-inference as an audit tool within a broader evaluation frame, not the primary claim engine.
- H2 3: ~"Artifact-Bound Contribution: Honest-Null Discipline and Minimization Frontiers" (was "Project Contribution and Claim Boundaries")
  - **New explicit subsection** emphasizing: benchmark gate (not_demonstrated), locked-split logic, fair-oracle comparison, minimization frontier, safety gates, evaluation contract.
  - Reference the comparison-contract TABLE (to be inserted near Results).

#### **02_methodology.md** (MINOR RENAMING, STRUCTURE CLEAR)
- H2 1: "Artifact-Driven Research Loop and Dataflow" (same; already good) {#sec:methodology-artifact-loop}
- H2 2: "Configuration, Tensors, and Roadmap-as-Executable-Stages" (rename; merge clarity) {#sec:methodology-default-config}
  - Keep subsection on spatial discretizer, bin counts, tensor shapes.
  - Clarify: roadmap items are now executable gates, not prose promises.

#### **02a_simulator_boundary.md** (UNCHANGED EXCEPT TITLE CLARITY)
- H2 1: "Pinned FractalRabbit Interface and Fixture Fallback" (slight clarity) {#sec:simulator-pinned-contract}
- H2 2: "Fractal-Dimension Control as Generator Knob" (same) {#sec:fractal-dimension-control}
- H2 3: "CSV Parameters vs. Fixed Internals" (clarify scope) {#sec:simulator-exposed-fixed}
- H2 4: "Waypoint Parser Contract" (minor) {#sec:simulator-waypoint-contract}

#### **02b_observation_model.md** (UNCHANGED EXCEPT TITLE CLARITY)
- H2 1: "Categorical Observation Surface and Finite Evidence" (minor) {#sec:observation-categorical-surface}
- H2 2: "Spatial Discretization into Analysis Cells" (minor) {#sec:observation-spatial-state}
- H2 3: "Temporal, Kinematic, and Reporting-Burst Features" (same) {#sec:observation-temporal-kinematic}
- H2 4: "Fractal-Dimension Sweep Implications" (same) {#sec:observation-fractal-implications}

#### **02c_formal_model.md** (UNCHANGED EXCEPT TITLE CLARITY)
- H2 1: "Discrete Hidden-State POMDP and Likelihood" (minor) {#sec:formal-discrete-model}
- H2 2: "Policy-Proxy Diagnostic (Not Full EFE)" (same) {#sec:formal-policy-proxy}

#### **02d_closed_loop_experiment.md** (UNCHANGED EXCEPT TITLE CLARITY)
- H2 1: "Candidate-Ranking Heuristic and Non-Controller Boundary" (same) {#sec:closed-loop-candidate-ranking}
- H2 2: "Sporadic-Observation Mission Fit" (same) {#sec:closed-loop-mission-fit}
- H2 3: "Recovery-Versus-Sparsity Lane" (same) {#sec:closed-loop-sparsity-lane}
- H2 4: "Holdout Scoring Under Fixed Discretization" (same) {#sec:closed-loop-holdout}

---

#### **03_results.md** (MAJOR RESTRUCTURE & RETITLING)

**NEW STRUCTURE** (11 sections → reorganized into 8, with strong result-bearing titles):

##### **H2 1: Comparison-Contract Table and Oracle Bounds (NEW, MOVED FROM INTRO/ROADMAP)**
- **Anchor**: {#sec:results-model-comparison-contract}
- **Content**: 
  - New table (`tbl:model-comparison-contract`) showing:
    | Model | Information Set | Task | Oracle? | Mean Loss (nats) | Calibration (ECE) | Verdict |
    | --- | --- | --- | --- | --- | --- | --- |
    | Persistence-oracle | true prev cell | predict next cell | yes (oracle) | ORACLE_LOSS | — | upper bound |
    | Reference filter (full location) | all modalities | next cell prediction | no | REFERENCE_FULL_LOSS | REFERENCE_ECE | strong baseline |
    | Reference filter (deep-latent) | time+speed+burst | next cell prediction | no | DEEP_LATENT_LOSS | DEEP_LATENT_ECE | fair baseline |
    | pymdp (single-factor) | same as reference | next cell prediction | no | PYMDP_LOSS | PYMDP_ECE | interpretable, non-leading |
    | Fair soft-marginal | emission posterior | previous cell only | no | FAIR_SOFT_LOSS | — | best single-step |
    | Fair AIF filter | filtered belief | previous cell recursion | no | FAIR_AIF_LOSS | — | **lowest-loss fair predictor** |
    | Fair oracle-stripped Markov | returned-place boost | previous cell + prior | no | FAIR_SOFT_BOOST_LOSS | — | strong fair baseline |
  - Text: Explains oracle bound (information asymmetry), defines "fair" (no one sees true prev cell), locates this manuscript's central honest result (AIF is lowest-loss **fair** predictor but does not beat oracle-handed persistence).
  - Cites [@tbl:pymdp-benchmark] and [@tbl:factorial-pymdp] as detailed model lanes.

##### **H2 2: Fixture Diagnostics and Spatial Resolution (OLD NAME, UNCHANGED MOSTLY)**
- **Anchor**: {#sec:results-fixture-diagnostics}
- Subsections: 
  - Default run artifact count, observation count, loss, entropy.
  - Waypoint paths ([@fig:waypoint-paths]) and spatial bins ([@fig:spatial-bins]).
  - **Spatial-resolution frontier as a fixture-choice gate** ([@fig:spatial-resolution-frontier], [@tbl:spatial-resolution-frontier]): {{SPATIAL_RESOLUTION_DEFAULT_BINS}}x{{SPATIAL_RESOLUTION_DEFAULT_BINS}} promoted because full-location loss improves; no-location guardrail ensures not claimed as metadata-recovery win.
  - Reporting bursts and speed categories ([@fig:reporting-bursts], [@fig:speed-categories]).

##### **H2 3: Reference Filter, Baselines, and Per-Modality Calibration (RENAMED FROM "Recovery Calibration")**
- **Anchor**: {#sec:results-recovery-calibration}
- **Retitle**: "Recovery, Calibration, and Honest-Null Hierarchy"
- Content (KEEP CURRENT STRUCTURE):
  - Ground-truth recovery ([@fig:state-recovery]): location-observed near-perfect; deep-latent recovers {{DEEP_LATENT_RECOVERY_MASS}} from metadata alone.
  - Calibration decomposition ([@fig:calibration], [@tbl:per-modality]): joint loss, per-modality marginals, skill against uniform/base-rate/persistence nulls.
  - Sparsity-versus-recovery ([@fig:sparsity-recovery], [@tbl:sparsity]): how recovery degrades as reports thin; filter vs. closed-loop divergence.
  - **Gap-filling against persistence oracle** ([@fig:anticipatory], [@sec:anticipatory]): location-blind limit vs. gap-filling regime; honest null result (speed-conditioned dynamics do not beat persistence oracle).
  - **EFE epistemic value** ([@fig:efe-decomposition]): decompose into epistemic and pragmatic; test active-sensing value (beats only fraction {{ACTIVE_SENSING_BEATEN_FRACTION}} of blind random; honest null).
  - **Program extensions as honest nulls** ([@fig:anomaly-detection], [@fig:recoverability-frontier]): routine-break detection (clean signal), structure recovery (no-location guardrail), mission-weighted selection (honest null). Recoverability frontier ([@fig:recoverability-frontier]): all modalities {{FRONTIER_ALL_MASS}}, no-location {{FRONTIER_NO_LOCATION_MASS}}.

##### **H2 4: Fair-Oracle-Stripped Comparison and Active-Inference Advantage (MOVED FROM ROADMAP, PROMOTED)**
- **Anchor**: {#sec:results-fair-comparison}
- **Retitle**: "Fair Predictor Ladder: Soft Marginalization, Recursion, and Regime-Bound Advantage"
- **NEW STRUCTURED MOVE FROM 07_roadmap.md** (the entire [@sec:roadmap-aif-deepening] ladder subsection):
  - Oracle-fed persistence baseline (upper bound).
  - Hard point estimate from emission posterior.
  - Soft marginalization (dominant single-step gain: {{FAIR_SOFT_MARG_GAIN}} nats).
  - Recursive active-inference belief.
  - Return-boosted soft Markov (strong baseline).
  - AIF + return-boosted prior ({{FAIR_AIF_BOOST_LOSS}} best fair loss).
  - Figure: [@fig:fair-inference] showing the ladder.
  - **Power analysis result** ([@fig:fair-power-regime], [@sec:roadmap-aif-deepening] end):
    - Few travellers: AIF advantage seed-robust ({{POWER_SMALL_CLEAN_ADV}} nats at clean emission).
    - Large travellers, clean emission: advantage vanishes ({{POWER_LARGE_CLEAN_ADV}} nats, not seed-robust); population transition model sufficient.
    - Large travellers, noisy emission: advantage re-emerges ({{POWER_LARGE_NOISY_ADV}} ± {{POWER_LARGE_NOISY_SD}} nats, seed-robust); noise restores {{POWER_NOISE_RESTORES}}.
    - **Honest conclusion**: Regime-bound result. Soft Bayesian marginalization dominant; recursion conditional on observation noise. Recursive belief earns its keep when emission uninformative (target sparse-waypoint condition), not in general. Never beats oracle.
  - Status: Best honest-inference loss {{AIF_SWEEP_BEST_LOSS}} nats on locked split; claim gate `{{SOTA_CLAIM_GATE_STATUS}}` (not_demonstrated). **Lowest-loss fair predictor in {{FAIR_SOTA_AIF_BEST_RATE}} seeds at powered holdout; {{FAIR_SOTA_DISTINCT_TRAVELLERS}} distinct held-out travellers per split; provisional, not established SOTA.**

##### **H2 5: PyMDP Benchmarking: Implementation, Agreement, and Interpretability (RENAMED FROM "pymdp Diagnostics")**
- **Anchor**: {#sec:results-pymdp-performance-interpretability}
- **Retitle**: "Active-Inference Instrumentation: PyMDP Validity, Cost, and Diagnostic Gain"
- Content:
  - Matched-prior agreement ([@fig:pymdp-agreement]): maximum absolute state-posterior difference {{PYMDP_EXACT_INFERENCE_DELTA}} against exact Bayes (numerical precision validation); closed-loop divergence {{PYMDP_LOOP_POSTERIOR_DELTA}} from policy-induced prior divergence.
  - Diagnostic surface ([@fig:pymdp-diagnostics]): variational free energy {{PYMDP_MEAN_VFE}}, runtime {{PYMDP_MEDIAN_STEP_MS}} ms per step, {{PYMDP_TOTAL_RUNTIME_S}} s total vs. {{REFERENCE_RUNTIME_S}} s reference.
  - **Performance vs. interpretability matrix** ([@tbl:pymdp-benchmark], formerly [@tbl:pymdp-benchmark]):
    - Performance status: `{{PYMDP_SOTA_STATUS}}` (not_demonstrated); delta {{PYMDP_PERFORMANCE_DELTA}} nats worse than reference.
    - Interpretability status: `{{PYMDP_INTERPRETABILITY_STATUS}}` (interpretable_but_nonleading_on_loss); exposes VFE, policy posterior, EFE, action-conditioned prior dynamics as auditable artifacts.
    - Roadmap gates: {{PYMDP_ROADMAP_GATES}}.
  - **Result**: PyMDP is **interpretability-positive, performance-modest**. Active-inference as an audit tool within benchmark, not a performance-leading claim.

##### **H2 6: Factorial PyMDP and Multi-Factor Interpretability (RENAMED FROM "Factorial pymdp Latent-Mode Diagnostics")**
- **Anchor**: {#sec:results-factorial-pymdp-latent-modes}
- **Retitle**: "Factorial Active Inference: Movement and Reporting Modes Under Factorized Posterior"
- Content (KEEP CURRENT):
  - Factorization across {{FACTORIAL_PYMDP_FACTOR_COUNT}} factors: spatial cell, movement mode, synthetic reporting mode.
  - Per-factor entropy ([@fig:factorial-pymdp-modes]).
  - Matched-prior validation (pymdp vs. analytic mean-field: {{FACTORIAL_PYMDP_MATCHED_PRIOR_DELTA}}); mean-field-vs-exact gap {{FACTORIAL_PYMDP_MEAN_FIELD_EXACT_GAP}}.
  - Recovery sharpness ([@fig:latent-mode-recovery]).
  - Model comparison ([@fig:factorial-model-gate-matrix], [@tbl:factorial-pymdp]): factorial {{FACTORIAL_PYMDP_MEAN_PREDICTIVE_LOSS}} vs. switching HMM {{FACTORIAL_PYMDP_SWITCHING_HMM_LOSS}} vs. particle {{FACTORIAL_PYMDP_PARTICLE_LOSS}}. Best model `{{FACTORIAL_PYMDP_BEST_MODEL}}`; gate `{{FACTORIAL_PYMDP_GATE}}` (non-leading).
  - **Result**: Interpretable latent-mode diagnostics; non-leading performance gate.

##### **H2 7: Robustness, Roadmap Gates, and Model Comparisons (RENAMED & CONSOLIDATED)**
- **Anchor**: {#sec:results-robustness-roadmap-gates}
- **Retitle**: "Robustness Artifacts, Sequence Baselines, and Locked-Split Model Gates"
- Content (MOVE FROM [@sec:results-roadmap-analyses] + [@sec:results-sequence-reporting-gates] + [@sec:results-model-gate-matrix] + [@sec:results-forward-gates]):
  - **Robustness artifacts** ([@fig:return-structure], [@fig:variance-bands], [@fig:upstream-sensitivity]): move-step entropy {{RETURN_MOVE_ENTROPY_REDUCTION}} nats, persistence headroom, return verdict `{{RETURN_STRUCTURE_VERDICT}}`; variance band skill {{VARIANCE_GAP_FILL_LOW}} to {{VARIANCE_GAP_FILL_HIGH}}; upstream sensitivity status.
  - **Sequence-baseline gate** ([@fig:sequence-baselines], [@tbl:sequence-baselines]): prequential next-cell ladder; best model `{{SEQUENCE_BASELINE_BEST_MODEL}}`; gate `{{SEQUENCE_BASELINE_GATE}}`; compares reference and pymdp fairly.
  - **Reporting-process gate** ([@fig:missingness-model], [@tbl:missingness-model]): phase and traveller structure in inter-report gaps; verdict `{{MISSINGNESS_VERDICT}}`; phase gain {{MISSINGNESS_PHASE_GAIN}}, combined gain {{MISSINGNESS_COMBINED_GAIN}}.
  - **Lane-resolved model-gate matrix** ([@fig:model-gate-matrix], [@tbl:model-gate-matrix]): fixture + external evidence side-by-side for sequence, missingness, return-plus-phase gates.
  - **Return-plus-phase prior** ([@tbl:return-phase-prior]): M9 gate comparing pooled, phase-only, return-only, return-plus-phase priors; best `{{RETURN_PHASE_BEST_MODEL}}` with gain {{RETURN_PHASE_GAIN_VS_PERSISTENCE}}; promoted if clear win, honest null otherwise.
  - **Forward-gate summary** ([@fig:m10-forward-gates], [@tbl:forward-gates]): latent-regime HMM `{{LATENT_REGIME_HMM_GATE}}`, neural baseline `{{NEURAL_BASELINE_GATE}}`, learned stay-rate `{{LEARNED_STAY_RATE_VERDICT}}`, held-out reporting `{{HELDOUT_REPORTING_VERDICT}}`, cross-generator persistence `{{CROSS_GENERATOR_PERSISTENCE_VERDICT}}`.

##### **H2 8: Location-Risk Frontier, Integrity Gates, and External Lane Evidence (RENAMED)**
- **Anchor**: {#sec:results-location-risk-integrity}
- **Retitle**: "Safety Boundaries: Location Minimization, Linkage Risk, and Data-Integrity Probes"
- Content (MOVE FROM [@sec:results-location-risk-integrity] + [@sec:results-privacy-external]):
  - **EFE location frontier** ([@fig:efe-location-frontier]): location share {{EFE_LOCATION_SHARE}}, metadata null p-value {{EFE_METADATA_NULL_P}}, residual without location {{EFE_RESIDUAL_WITHOUT_LOCATION}}; verdict `{{EFE_LOCATION_VERDICT}}`.
  - **Linkage-risk frontier** ([@fig:linkage-risk-frontier]): full-location linkage {{LINKAGE_LOCATION_LINKAGE}}; no-location exceeds null: {{LINKAGE_NO_LOCATION_EXCEEDS_NULL}}; verdict `{{LINKAGE_RISK_VERDICT}}` (defensive minimization stat, not matcher).
  - **Poisoning robustness** ([@fig:poisoning-robustness]): gross teleport detected {{POISONING_GROSS_DETECTED}}, subtle-drift null {{POISONING_SUBTLE_NULL}}, teleport AUC {{POISONING_TELEPORT_AUC}}; verdict `{{POISONING_VERDICT}}`.
  - **Privacy frontier** ([@fig:privacy-frontier]): unicity {{PRIVACY_FRONTIER_ALL_UNICITY}} (all modalities) vs. {{PRIVACY_FRONTIER_NO_LOCATION_UNICITY}} (no location); co-travel coincidence {{PRIVACY_FRONTIER_NO_LOCATION_COTRAVEL}}.
  - **Hardened anomaly** ([@fig:hardened-anomaly-detection]): VFE vs. Markov baseline on speed-shift ({{HARDENED_ANOMALY_SPEED_VFE_AUC}}) and preferred-set drift ({{HARDENED_ANOMALY_DRIFT_VFE_AUC}}).
  - **External lane evidence**: {{EXTERNAL_RUN_STATUS}}; {{EXTERNAL_OBSERVATION_COUNT}} waypoints, {{EXTERNAL_TRAVELLER_COUNT}} travellers, {{EXTERNAL_LATENT_STATE_COUNT}} cells; mean loss {{EXTERNAL_MEAN_PREDICTIVE_LOSS}}, entropy {{EXTERNAL_MEAN_POSTERIOR_ENTROPY}}; pymdp agreement {{EXTERNAL_PYMDP_EXACT_DELTA}}; replicate dispersion {{EXTERNAL_REPLICATE_DISPERSION}}; reference recovery {{EXTERNAL_RECOVERY_REFERENCE_MASS}}, skill {{EXTERNAL_REFERENCE_SKILL_SCORE}}; place count {{EXTERNAL_PLACE_COUNT}}, revisit {{EXTERNAL_PLACE_REVISIT_RATE}}.

---

### DISCUSSION (NEW SPINE: RENAMED FROM 05/07/08)

#### **04_discussion.md** (NEW FILE, COMBINES 05/07/08 MATERIAL WITH RESULT-BEARING TITLES)

##### **H2 1: Synthetic Evidence Boundaries and Honest Limits (FROM 05_limitations.md §1–2)**
- **Anchor**: {#sec:discussion-synthetic-boundaries}
- **Retitle**: "What This Benchmark Settles and Doesn't: Synthetic-Only Evidence"
- Content:
  - Generator boundary: FractalRabbit is software benchmark, not empirical validation.
  - Reproducibility limits: external lanes not bit-reproducible (statistical rather than exact).
  - Latent state is discretized cell, not internal FractalRabbit state or real semantic.
  - Spatial-resolution frontier narrowed the claim (location-observed fit vs. metadata-only recovery).
  - No claims about real mobility regularities (returns, scaling, predictability).

##### **H2 2: Performance, State-of-the-Art, and the Persistence Oracle (FROM 05_limitations.md §3 + 07_roadmap.md SOTA section)**
- **Anchor**: {#sec:discussion-sota-boundary}
- **Retitle**: "Performance Gates and the Oracle Baseline: Why We Don't Claim SOTA"
- Content:
  - PyMDP is not state-of-the-art on predictive loss ({{PYMDP_SOTA_STATUS}}, delta {{PYMDP_PERFORMANCE_DELTA}}).
  - Persistence oracle is **unbeatable** when location is never observed (information-asymmetry bound); this is honest, not a limitation to fix.
  - Fair comparison (no one sees true previous cell) shows soft marginalization {{FAIR_SOFT_MARG_GAIN}} nats dominant; recursion conditional on noise; AIF lowest-loss fair predictor but not oracle-beating.
  - Locked-split SOTA harness defined ([@sec:roadmap-sota-gates]): prequential next-cell loss, calibration, persistence-null skill, runtime, minimization/integrity regressions. Claim gate `{{SOTA_CLAIM_GATE_STATUS}}` (not_demonstrated); leakage probe reaches {{SOTA_LEAKAGE_PROBE_LOSS}} but excluded; no eligible candidate beats {{SOTA_STRONGEST_BASELINE_LOSS}}.
  - Future performance bar: better next-cell assignment under sparse reports, calibrated recovery, locked-split performance, runtime, interpretability, minimization/integrity preservation.

##### **H2 3: Honest Results and Regime Bounds (FROM 05_limitations.md §4–8 + 07_roadmap.md "Honest Nulls")**
- **Anchor**: {#sec:discussion-regime-bounds}
- **Retitle**: "Regime-Bound Results: Gap-Filling, Structure Recovery, and Sensing Value"
- Content:
  - Deep-latent recovery does not beat persistence oracle on this fixture (skill {{DEEP_LATENT_BRIER_SKILL_VS_PERSISTENCE}} vs. {{REFERENCE_BRIER_SKILL_VS_PERSISTENCE}}).
  - Gap-filling (location hidden in sparse intervals): speed-conditioned dynamics do not clear persistence oracle null (skill {{GAP_FILL_FLAT_SKILL}}, {{GAP_FILL_SCRP_SKILL}}) — honest null, not bug; FractalRabbit speed is derived from displacement.
  - Structure recovery (occupancy, preferred places) does not beat dynamics-only propagation on external run (deep-latent TV {{STRUCTURE_DEEP_LATENT_TV}} vs. dynamics {{STRUCTURE_DYNAMICS_ONLY_TV}}).
  - Active-sensing value (budget-matched expected-free-energy selection): beats only {{ACTIVE_SENSING_BEATEN_FRACTION}} of blind random schedules; speed-conditioned dynamics already anticipate cell changes.
  - **Fair-comparison power analysis** (the regime result): advantage regime-bound. Large with few travellers ({{POWER_SMALL_CLEAN_ADV}} nats), vanishes at scale under clean emission ({{POWER_LARGE_CLEAN_ADV}} nats, not seed-robust), re-emerges under noise ({{POWER_LARGE_NOISY_ADV}} nats, seed-robust). Implication: recursive belief earns its keep when emission uninformative (sparse-waypoint condition), not universally.
  - All honest nulls reported as nulls, not reframed as windows for future work.

##### **H2 4: Two-Tier Honesty Stack and Evaluation Contract (FROM 08_conclusion.md §3 + 05_limitations.md §8)**
- **Anchor**: {#sec:discussion-honesty-stack}
- **Retitle**: "Honesty Discipline: Execution Harness and Claim-Level Evaluation Contract"
- Content:
  - **Execution tier** (harness): checked fixtures, negative controls, holdout scoring, release manifest checksums, tokenized quantities from local artifacts, source provenance for result-critical numbers.
  - **Claim tier** (evaluation contract): machine-checkable static linter blocking wrong-regime skill claims and undisclosed speed-cue leakage.
  - Contract reach: statically catches regime and cue-disclosure errors; does not catch performance underfit or fabrication (caught only by execution).
  - Honest read: two-tier stack is the smaller claim-level tier; execution-level failures still possible and documented.

##### **H2 5: Roadmap: Executed Gates and Remaining Work (FROM 07_roadmap.md §1–5, CONDENSED)**
- **Anchor**: {#sec:discussion-roadmap}
- **Retitle**: "Executed Roadmap Gates and What Remains for Performance Leadership"
- Content (BULLETED SUMMARY):
  - **Delivered in this revision**:
    - Expanded fixture (4→6 travellers, 240→480 reports, 18→36 days); traveller-clustered CIs now viable.
    - Spatial-resolution frontier (default {{SPATIAL_RESOLUTION_DEFAULT_BINS}}, best {{SPATIAL_RESOLUTION_BEST_FEASIBLE_BINS}}).
    - Return-structure gate (entropy {{RETURN_MOVE_ENTROPY_REDUCTION}} nats, verdict `{{RETURN_STRUCTURE_VERDICT}}`).
    - Fractal-dimension sweep (status `{{FRACTAL_DIMENSION_SWEEP_STATUS}}`, best loss D = {{FRACTAL_DIMENSION_BEST_LOSS_D}}`).
    - Circadian-phase gate (verdict `{{CIRCADIAN_PHASE_VERDICT}}`, gain {{CIRCADIAN_PHASE_GAIN}}`).
    - Sequence-baseline ladder (best `{{SEQUENCE_BASELINE_BEST_MODEL}}`, gate `{{SEQUENCE_BASELINE_GATE}}`).
    - Reporting-process gate (verdict `{{MISSINGNESS_VERDICT}}`).
    - Lane-resolved model-gate matrix ({{MODEL_GATE_MATRIX_STATUS}}`).
    - Return-plus-phase prior ({{RETURN_PHASE_VERDICT}}`).
    - Forward-gate summary ({{FORWARD_GATE_STATUS}}`).
    - PyMDP benchmark matrix ({{PYMDP_SOTA_STATUS}}`).
    - Location-risk, linkage, poisoning, privacy gates.
    - Evaluation contract (claim-level honesty).
    - Fair-inference ladder and power analysis.
  - **Near-term** (would strengthen performance claim):
    - External-realization model gates (promote fixture-positive results only after external lane current).
    - Multiscale spatial state (replace single-grid with hierarchy).
    - Continuous/particle-filter dynamics (with same null hierarchy and interpretability audit).
  - **State-of-the-art prerequisites** ([@sec:roadmap-sota-gates]):
    - Locked-split {{SOTA_STRONGEST_BASELINE_LOSS}}-beater with no minimization regression.
    - Calibrated recovery and proper-score diagnostics.
    - Consented empirical validation under separate protocol.
    - Cross-vendor audit beyond this vendor's implementation.

##### **H2 6: Final Boundary: What This Paper Is and Isn't (FROM 08_conclusion.md §4 + 07_roadmap.md §end)**
- **Anchor**: {#sec:discussion-final-boundary}
- **Retitle**: "Final Claim Boundary: Honest-Benchmark Discipline, Not Operational Surveillance"
- Content:
  - {{PROJECT_TITLE}} is **synthetic-data analysis and software-validation project** on pinned FractalRabbit, **not empirical mobility account**.
  - Signature contribution: **artifact-bound honesty discipline** (explicit nulls, locked-split logic, fair oracle-stripped comparison, minimization frontier, evaluation contract).
  - PyMDP adds **interpretability, not performance leadership**.
  - Recovery and calibration results are **regime-bound and fair-frontier**, never oracle-beating.
  - Any **real-waypoint claim** remains behind separate empirical-validation protocol (consent, provenance, sampling, minimization endpoints, privacy review).
  - **Non-operational boundary**: no surveillance, tasking, or evasion surface. Oversight-positive, location-minimization-framed.
  - Each roadmap item is falsifiable with a kill-gate, not intended capability claim.

---

### SUPPLEMENTARY RESULTS & TECHNICAL DETAIL (NEW S03, S04)

#### **S03_diagnostic_and_robustness_detail.md** (NEW)
- **Purpose**: Move secondary/exploratory figures and detailed robustness checks from main Results.
- **Content**:
  - Sensitivity summary ([@tbl:sensitivity-summary]): deterministic variants, report-count, spatial-bin, burst-rule, smoothing, control sweeps.
  - Negative-control lane ([@fig:negative-controls], [@tbl:negative-controls]): {{NEGATIVE_CONTROL_PASSED_COUNT}}/{{NEGATIVE_CONTROL_COUNT}}.
  - Fixture diagnostics beyond core (transition heatmap, likelihood panels, policy-proxy trace).
  - Posterior entropy and policy proxy figures.
  - Candidate scores ([@fig:candidate-scores]) and ranking explanation.
  - **Anchor names**: {#sec:s03-sensitivity}, {#sec:s03-negative-controls}, {#sec:s03-diagnostic-surface}, etc.

#### **S04_advanced_model_comparisons_and_transfer.md** (NEW)
- **Purpose**: Advanced model lanes and cross-generator transfer.
- **Content**:
  - Latent-regime HMM details and gate ({{LATENT_REGIME_HMM_GATE}}, {{LATENT_REGIME_HMM_BEST_HELDOUT_LOSS}}, {{LATENT_REGIME_HMM_OVERFIT_GAP}}).
  - Neural baseline ({{NEURAL_BASELINE_GATE}}, {{NEURAL_BASELINE_LOSS}}).
  - Learned stay-rate ({{LEARNED_STAY_RATE_VERDICT}}, best {{LEARNED_STAY_RATE_BEST_VARIANT}}).
  - Held-out reporting process ({{HELDOUT_REPORTING_VERDICT}}, leave-one-traveller-out + temporal split).
  - Cross-generator transfer (persistence, particle filter, return structure across two generators).
  - **Anchor names**: {#sec:s04-latent-regime-hmm}, {#sec:s04-neural-baseline}, {#sec:s04-learned-priors}, {#sec:s04-cross-generator}, etc.

#### **S01_validation_and_threat_model.md** (EXISTING; MINOR RETITLE)
- **Rename** to: "Supplementary S1 — Evaluation Contract, Threat Model, and Safety Gates"
- Existing content unchanged; renaming makes scope clearer.

#### **S02_reproducibility.md** (EXISTING; MINOR RETITLE)
- **Rename** to: "Supplementary S2 — Artifact Provenance, Release Manifest, and Reproducibility"
- Existing content unchanged; renaming makes scope clearer.

---

### REFERENCES & APPENDIX

#### **99_references.md**
- Unchanged; 62 entries already cited.
- **NEW CITATIONS TO ADD** (from peer review):
  - Rubin 1976 (missing data / MAR / MNAR).
  - Talts et al. 2018 (Simulation-Based Calibration).
  - Yao et al. 2018 (VI diagnostics / PSIS).
  - Roberts 2017 (cross-validation for dependent data).
  - Rocher 2019 (re-identification of incomplete data).
  - Dwork & Roth 2014 (differential privacy foundations).

---

## File-Level Mapping: Old → New Organization

| Current File | Location After Restructure | Notes |
| --- | --- | --- |
| 00_abstract.md | 00_abstract.md (REVISED) | Benchmark contribution to fore; fair comparison; honest frontier. |
| 01_introduction.md | 01_introduction.md (RETITLED H2s) | Same structure; emphasis shift to benchmark first, AIF as audit tool. |
| 02_methodology.md | 02_methodology.md (RETITLED H2) | Minimal change; clarity. |
| 02a_simulator_boundary.md | 02a_simulator_boundary.md | Same; part of Methods. |
| 02b_observation_model.md | 02b_observation_model.md | Same; part of Methods. |
| 02c_formal_model.md | 02c_formal_model.md | Same; part of Methods. |
| 02d_closed_loop_experiment.md | 02d_closed_loop_experiment.md | Same; part of Methods. |
| **03_results.md** | **03_results.md (MAJOR RESTRUCTURE)** | **8 new H2s; reorganized from 11. Promoted fair-ladder, forward gates, safety gates from 07_roadmap. Moved diagnostic detail to S03.** |
| 05_limitations.md | 04_discussion.md §1–3 (merged + retitled) | Honest limits → Discussion tier. Synthetic boundaries, SOTA boundary, regime bounds. |
| 07_roadmap.md | **SPLIT**: <br/>03_results.md §4 (fair-ladder, power) <br/>04_discussion.md §2, §5 (SOTA, roadmap summary) | Fair-inference content promoted to Results; forward-only roadmap to Discussion. |
| 08_conclusion.md | 04_discussion.md §4, §6 (merged + retitled) | Honesty stack and final boundary integrated into Discussion. |
| (NEW) | S03_diagnostic_and_robustness_detail.md | Sensitivity, negative controls, fixture diagnostics beyond core. |
| (NEW) | S04_advanced_model_comparisons_and_transfer.md | HMM, neural, learned priors, cross-generator transfer. |
| S01_validation_and_threat_model.md | S01_validation_and_threat_model.md (RETITLED) | Minor: clarify scope. |
| S02_reproducibility.md | S02_reproducibility.md (RETITLED) | Minor: clarify scope. |
| 99_references.md | 99_references.md | Add 6 new citations (Rubin, Talts, Yao, Roberts, Rocher, Dwork & Roth). |

---

## Core Figures Remaining in Main Results (Per Peer Review: Keep ~4)

**Proposed 4–5 core figures to retain in main body (03_results.md)**:

1. **Benchmark Schematic** (generative model card diagram or evidence atlas → renamed / restructured for clarity).
2. **Model-Comparison Contract Table/Figure** (tbl:model-comparison-contract; new, showing oracle bounds and fair predictor ladder).
3. **Fair Predictor Ladder Figure** ([@fig:fair-inference] showing hard estimate → soft → recursive → oracle).
4. **Location-Minimization Frontier** ([@fig:efe-location-frontier] or [@fig:recoverability-frontier]; oversight-positive statement of what location protects).
5. *(Optional 5th)*: Sequence-Baseline Ladder ([@fig:sequence-baselines]) if space permits; otherwise move to S04.

**Secondary figures move to S03/S04**:
- Fixture diagnostics (waypoint paths, spatial bins, transition heatmap, likelihood panels, posterior entropy traces, policy proxy).
- Sensitivity and robustness (variance bands, upstream sensitivity, return structure, fractal-dimension sweep, circadian phase).
- Advanced comparisons (factorial pymdp modes, latent-mode recovery, HMM/neural/learned-prior gates, cross-generator transfer).
- Safety/privacy (linkage risk, poisoning, privacy frontier, hardened anomaly).

---

## Detailed Section-by-Section Anchor (`{#sec:...}`) Mapping

| Old Anchor | New Anchor (If Different) | New Location | Reasoning |
| --- | --- | --- | --- |
| {#sec:introduction} | (same) | 01_introduction | Intro unchanged. |
| {#sec:intro-synthetic-benchmark} | (same) | 01_introduction H2:1 | Unchanged. |
| {#sec:intro-variational-diagnostics} | (same) | 01_introduction H2:2 | Unchanged. |
| {#sec:intro-contribution} | (same) | 01_introduction H2:3 | Unchanged. |
| {#sec:methodology} | (same) | 02_methodology | Methods spine unchanged. |
| {#sec:methodology-artifact-loop} | (same) | 02_methodology H2:1 | Unchanged. |
| {#sec:methodology-default-config} | (renamed to clarity) | 02_methodology H2:2 | Merge/clarify. |
| {#sec:methodology-roadmap-stages} | (merged into H2:2 or moved to Discussion) | 02_methodology H2:2 or 04_discussion | Roadmap-as-stages is methodology framing; kept in Methods. |
| {#sec:simulator-boundary} | (same) | 02a_simulator_boundary | Unchanged. |
| {#sec:simulator-pinned-contract} | (same) | 02a H2:1 | Unchanged. |
| {#sec:fractal-dimension-control} | (same) | 02a H2:2 | Unchanged. |
| {#sec:simulator-exposed-fixed} | (same) | 02a H2:3 | Unchanged. |
| {#sec:simulator-waypoint-contract} | (same) | 02a H2:4 | Unchanged. |
| {#sec:observation-model} | (same) | 02b_observation_model | Unchanged. |
| {#sec:observation-categorical-surface} | (same) | 02b H2:1 | Unchanged. |
| {#sec:observation-spatial-state} | (same) | 02b H2:2 | Unchanged. |
| {#sec:observation-temporal-kinematic} | (same) | 02b H2:3 | Unchanged. |
| {#sec:observation-fractal-implications} | (same) | 02b H2:4 | Unchanged. |
| {#sec:formal-model} | (same) | 02c_formal_model | Unchanged. |
| {#sec:formal-discrete-model} | (same) | 02c H2:1 | Unchanged. |
| {#sec:formal-policy-proxy} | (same) | 02c H2:2 | Unchanged. |
| {#sec:closed-loop} | (same) | 02d_closed_loop_experiment | Unchanged. |
| {#sec:closed-loop-candidate-ranking} | (same) | 02d H2:1 | Unchanged. |
| {#sec:closed-loop-mission-fit} | (same) | 02d H2:2 | Unchanged. |
| {#sec:closed-loop-sparsity-lane} | (same) | 02d H2:3 | Unchanged. |
| {#sec:closed-loop-holdout} | (same) | 02d H2:4 | Unchanged. |
| **{#sec:results}** | **{#sec:results}** | **03_results (RESTRUCTURED)** | **Spine anchor stays; subsection anchors below.** |
| {#sec:results-model-comparison-contract} | (NEW) | 03_results H2:1 | New section added early. |
| {#sec:results-fixture-diagnostics} | (same) | 03_results H2:2 | Moved to position 2 (after contract). |
| {#sec:results-recovery-calibration} | {#sec:results-recovery-calibration} | 03_results H2:3 | Retitled "Recovery, Calibration, Honest-Null Hierarchy"; content stays. |
| {#sec:anticipatory} | (same) | 03_results H2:3 subsub | Nested under H2:3. |
| {#sec:efe} | (same) | 03_results H2:3 subsub | Nested under H2:3. |
| {#sec:program-extensions} | (same) | 03_results H2:3 subsub | Nested under H2:3. |
| **{#sec:results-fair-comparison}** | **(NEW)** | **03_results H2:4** | **Moved from 07_roadmap; entire fair-ladder + power analysis.** |
| {#sec:pymdp-performance-interpretability} | (same) | 03_results H2:5 | Retitled "Active-Inference Instrumentation...". |
| {#sec:results-pymdp-diagnostics} | (same) | 03_results H2:5 | Nested under new H2:5. |
| {#sec:results-sensitivity-falsification} | → S03 | Moved to S03 | Moved to supplement. |
| {#sec:results-pymdp-diagnostics} | (same) | 03_results H2:5 | Retained in main (performance-and-interpretability). |
| {#sec:factorial-pymdp-latent-modes} | (same) | 03_results H2:6 | Retitled "Factorial Active Inference...". |
| {#sec:results-roadmap-analyses} | → S03/S04 (split) | Moved/split | Sensitivity, robustness → S03; forward gates → H2:7. |
| {#sec:results-sequence-reporting-gates} | (same) | 03_results H2:7 | Consolidated under "Robustness, Sequence, Model Gates". |
| {#sec:results-model-gate-matrix} | (same) | 03_results H2:7 | Nested under new H2:7. |
| {#sec:results-forward-gates} | (same) | 03_results H2:7 | Nested under new H2:7. |
| {#sec:results-location-risk-integrity} | (same) | 03_results H2:8 | Retitled "Safety Boundaries...". |
| {#sec:results-privacy-external} | (same) | 03_results H2:8 | Nested under new H2:8. |
| **{#sec:discussion}** (NEW) | (NEW) | **04_discussion.md** | **New file merging 05/07/08.** |
| {#sec:discussion-synthetic-boundaries} | (NEW) | 04_discussion H2:1 | From 05_limitations §1–2. |
| {#sec:discussion-sota-boundary} | (NEW) | 04_discussion H2:2 | From 05_limitations §3 + 07_roadmap SOTA. |
| {#sec:discussion-regime-bounds} | (NEW) | 04_discussion H2:3 | From 05_limitations §4–8 + 07_roadmap "Honest Nulls". |
| {#sec:discussion-honesty-stack} | (NEW) | 04_discussion H2:4 | From 08_conclusion §3 + 05_limitations §8. |
| {#sec:discussion-roadmap} | (NEW) | 04_discussion H2:5 | From 07_roadmap §1–5 (condensed). |
| {#sec:discussion-final-boundary} | (NEW) | 04_discussion H2:6 | From 08_conclusion §4 + 07_roadmap end. |
| {#sec:limitations} | → merged into 04_discussion | Removed as separate | Content distributed into Discussion H2s. |
| {#sec:limitations-synthetic-reproducibility} | → {#sec:discussion-synthetic-boundaries} | 04_discussion H2:1 | Merged. |
| {#sec:limitations-sota} | → {#sec:discussion-sota-boundary} | 04_discussion H2:2 | Merged. |
| {#sec:limitations-scoring-null} | → {#sec:discussion-regime-bounds} | 04_discussion H2:3 | Merged. |
| {#sec:limitations-roadmap-artifacts} | → {#sec:discussion-roadmap} | 04_discussion H2:5 | Merged. |
| {#sec:roadmap} | → distributed | Merged into Discussion | Sections redistribute. |
| {#sec:roadmap-implemented-now} | → {#sec:discussion-roadmap} | 04_discussion H2:5 | Condensed summary. |
| {#sec:roadmap-near-term-gates} | → {#sec:discussion-roadmap} | 04_discussion H2:5 | Condensed summary. |
| {#sec:roadmap-sota-gates} | → {#sec:discussion-sota-boundary} | 04_discussion H2:2 | SOTA machinery in boundary section. |
| {#sec:roadmap-aif-deepening} | → {#sec:results-fair-comparison} | 03_results H2:4 | **PROMOTED TO RESULTS; NOT ROADMAP.** |
| {#sec:roadmap-empirical-gate} | → {#sec:discussion-final-boundary} | 04_discussion H2:6 | Brief reference to separate protocol. |
| {#sec:conclusion} | → distributed | Merged into Discussion | Content redistributed. |
| {#sec:conclusion-pymdp-now} | → {#sec:discussion-sota-boundary} or {#sec:results-pymdp-performance-interpretability} | Mixed | PyMDP performance clarity in Results H2:5; conclusion narrative removed. |
| {#sec:conclusion-recovery-minimization} | → {#sec:discussion-synthetic-boundaries}, {#sec:discussion-regime-bounds} | 04_discussion H2:1, H2:3 | Boundaries narrative distributed. |
| {#sec:conclusion-roadmap-gates} | → {#sec:discussion-roadmap} | 04_discussion H2:5 | Roadmap summary. |
| {#sec:conclusion-boundary} | → {#sec:discussion-final-boundary} | 04_discussion H2:6 | Honesty stack and final boundary. |
| {#sec:validation} | (same) | S01_validation_and_threat_model | Existing supplement unchanged. |
| {#sec:reproducibility} | (same) | S02_reproducibility | Existing supplement unchanged. |
| {#sec:s03-sensitivity}} | (NEW) | S03_diagnostic_and_robustness_detail | New supplement. |
| {#sec:s03-negative-controls}} | (NEW) | S03_diagnostic_and_robustness_detail | New supplement. |
| {#sec:s04-latent-regime-hmm}} | (NEW) | S04_advanced_model_comparisons_and_transfer | New supplement. |
| {#sec:s04-neural-baseline}} | (NEW) | S04_advanced_model_comparisons_and_transfer | New supplement. |
| {#sec:s04-cross-generator}} | (NEW) | S04_advanced_model_comparisons_and_transfer | New supplement. |
| {#sec:references}} | (same) | 99_references | Unchanged. |

---

## Implementation Checklist

### Phase 1: Structural Reorganization (Read-Only Plan, No Edits)
- [ ] **Read all current section files** (00–08, S01–S02, 99) — ✓ DONE
- [ ] **Identify token dependencies**: Every {{TOKEN}} in old sections must map to new sections.
  - Example: {{PYMDP_SOTA_STATUS}} cited in 08_conclusion becomes cited in 03_results H2:5 and 04_discussion H2:2.
  - Validation: Search manuscript/ for all {{...}} tokens; cross-reference with `manuscript_variables.json` provenance.
- [ ] **Map anchor cross-references**: Every [@sec:old-anchor] must either stay or redirect.
  - Example: 08_conclusion cites [@sec:results-fixture-diagnostics] — stays valid.
  - Example: 08_conclusion cites [@sec:roadmap-aif-deepening] → becomes [@sec:results-fair-comparison] once moved.
  - Validation: Run `grep -r "@sec:" manuscript/` and update all internal refs.

### Phase 2: File-Level Edits (When Execution Begins)
- [ ] **Merge 05_limitations + 07_roadmap + 08_conclusion into new 04_discussion.md**
  - Preserve all tokens; reorganize into 6 H2 sections with new titles.
  - Preserve all anchor IDs; create new ones for new sections (e.g., {#sec:discussion-synthetic-boundaries}).
  - Update internal references (@sec:) throughout.
- [ ] **Restructure 03_results.md** into 8 H2 sections (from current 11).
  - **NEW H2:1**: Comparison-contract table (source: new synthetic, using existing tokens from pymdp_benchmark_matrix, fair_inference figures).
  - **NEW H2:4**: Promote fair-ladder + power from 07_roadmap [@sec:roadmap-aif-deepening]; update anchors.
  - **Consolidate H2:7**: Merge 4 subsections (robustness, sequence, reporting, model-gates, forward-gates) under result-bearing title.
  - Move sensitivity, negative controls, other diagnostics to S03.
  - Move HMM, neural, learned priors, cross-generator to S04.
  - Update all internal cross-references.
- [ ] **Create S03_diagnostic_and_robustness_detail.md**
  - Extract from 03_results: sensitivity summary, negative controls, fixture diagnostics, robustness artifacts.
  - Maintain all {{TOKEN}} bindings; test renderability.
- [ ] **Create S04_advanced_model_comparisons_and_transfer.md**
  - Extract from 03_results and 07_roadmap: HMM, neural, learned priors, cross-generator.
  - Maintain all {{TOKEN}} bindings; test renderability.
- [ ] **Revise 00_abstract.md**
  - Reframe to end on benchmark contribution + fair comparison + honesty discipline.
  - Mention oracle limitation but pivot to fair frontier.
  - Reword final sentence to underscore safety-boundary innovation.
- [ ] **Revise 01_introduction.md**
  - Retitle H2 subsections per plan (benchmarking → diagnostic → artifact-bound contribution).
  - Strengthen emphasis on benchmark as primary; active-inference as audit tool within it.
  - Add cross-reference to new comparison-contract table in Results.
- [ ] **Minor: 02_methodology.md**
  - Retitle H2 subsections for clarity (no content changes).
  - Ensure "roadmap-as-stages" framing is clear.
- [ ] **Minor: 02a–02d subsections**
  - Retitle H2 subsections for result-bearing clarity.
  - No major content changes.

### Phase 3: Cross-File Validation
- [ ] **Anchor correctness**: All {{...}} tokens render; all [@sec:...] anchors resolve.
- [ ] **Narrative continuity**: New structure reads as coherent IMRaD spine (Intro → Methods → Results → Discussion).
- [ ] **Figure placement**: ~4–5 core figures remain in main Results; others moved to S03/S04.
- [ ] **Token/claim consistency**: Every headline claim in abstract matches Results statement; no new claims introduced in Discussion.
- [ ] **Evaluation contract alignment**: Every claim in main text auditable against eval_contract.py guards.

### Phase 4: Rendering & Final Check
- [ ] Run `scripts/z_generate_manuscript_variables.py` → verify all tokens render.
- [ ] Run `scripts/11_check_manuscript_variable_provenance.py` → verify token provenance.
- [ ] Run `scripts/03_render_pdf.py` → check rendered PDF structure and figure placement.
- [ ] Spot-check cross-references in rendered PDF (click through [@sec:...] links).
- [ ] Verify that new section anchors (Discussion, S03, S04) appear in rendered PDF table of contents.

---

## Result-Bearing Title Examples (Old → New)

| OLD TITLE | NEW TITLE | ENCODING |
| --- | --- | --- |
| Results: Artifact-Gated Diagnostics and Null Hierarchy | Results: Benchmark Comparison, Fair Predictor Ladder, and Safety Boundaries | Prioritizes comparison table, fair result, safety. |
| Fixture Diagnostics and Spatial-Resolution Frontier | Fixture Diagnostics and Spatial Resolution | Tighter title. |
| Recovery Calibration Against Honest Nulls | Recovery, Calibration, and Honest-Null Hierarchy | Emphasizes hierarchy of nulls. |
| pymdp Agreement, Cost, and Diagnostic Surface | Active-Inference Instrumentation: PyMDP Validity, Cost, Diagnostic Gain | Frames AIF as tool; encodes agreement + clarity. |
| Robustness Artifacts and Roadmap Gate Results | Robustness Artifacts, Sequence Baselines, and Locked-Split Model Gates | Names key comparisons; signals rigor. |
| Factorial pymdp Latent-Mode Diagnostics | Factorial Active Inference: Movement and Reporting Modes | Specifies factors; encodes interpretability angle. |
| Location-Minimization, Linkage Risk, Data-Integrity Gates | Safety Boundaries: Location Minimization, Linkage Risk, Data-Integrity Probes | "Safety boundaries" frames as defensive measure. |
| Limitations: Synthetic Evidence and Null-Regime Boundaries | Synthetic Evidence Boundaries and Honest Limits | **Becomes Discussion H2:1.** |
| — | Performance Gates and Oracle Baseline | **New Discussion H2:2, from Limitations SOTA section.** |
| — | Regime-Bound Results: Gap-Filling, Structure Recovery, Sensing Value | **New Discussion H2:3, from Limitations honest-nulls sections.** |
| Roadmap: Forward Gates After Implemented Artifacts | Roadmap: Executed Gates and Remaining Work | **New Discussion H2:5, condensed from 07_roadmap.** |
| Conclusion: Honest Benchmarking on Pinned Synthetic Data | Final Claim Boundary: Honest-Benchmark Discipline | **New Discussion H2:6, from 08_conclusion boundary sections.** |

---

## Risk Mitigation

### Anchor/Reference Integrity
- **Risk**: Moving sections breaks cross-references.
- **Mitigation**: Grep for all [@sec:...] patterns; validate in rendered PDF.

### Token Provenance
- **Risk**: Tokens in moved sections point to old artifact paths that no longer align with new structure.
- **Mitigation**: Run provenance checks post-restructure; re-run variable generation if needed.

### Narrative Continuity
- **Risk**: Jumping from Methods → Results H2:1 (comparison contract) feels abrupt if introduction doesn't set it up.
- **Mitigation**: Revise 01_introduction H2:3 to explicitly mention the comparison-contract table coming in Results.

### Supplement Integration
- **Risk**: Readers skip supplements; key robustness results hidden.
- **Mitigation**: Main Results text explicitly directs readers to S03/S04 for detail (e.g., "Full sensitivity analysis in [@sec:s03-sensitivity]").

### Figure Captions
- **Risk**: Moving figures to supplements breaks caption narrative if main text assumes figure is nearby.
- **Mitigation**: Audit all [@fig:...] references in main text; ensure they align with figure location (main or supplement).

---

## Expected Outcome

**After restructuring:**
- **IMRaD spine crystal clear**: Intro (benchmark motivation & active-inference-as-audit frame) → Methods (simulator, obs model, POMDP, closed loop) → Results (comparison contract, fair ladder, pymdp benchmarking, safety bounds) → Discussion (honest limits, SOTA gate, regime bounds, honesty stack, roadmap, final boundary).
- **Main Results ~20–25 pages** (down from ~40 currently): 4–5 core figures, comparison-contract table, fair-predictor ladder, minimization frontier.
- **Supplementary S03 & S04 ~15–20 pages total**: Sensitivity, negative controls, advanced models, cross-generator transfer.
- **Manuscript reads benchmark-first**: Active inference is one instrument within a larger honest-evaluation discipline.
- **Every headline claim is result-bearing**: Titles encode what changed, what's bounded, what's novel.
- **All anchors, tokens, references validated** before final render.
"

## Top risks
- Anchor and cross-reference integrity: Moving large sections (especially 07_roadmap → 03_results H2:4) risks breaking @sec: and @fig: citations throughout manuscript. Mitigation requires exhaustive grep and rendered PDF validation.
- Token provenance misalignment: Tokens in relocated subsections may have provenance tuples pointing to old artifact or section paths. Must revalidate manuscript_variable_provenance.json post-restructure and re-run variable generation if needed.
- Narrative continuity at Results entry point: Jumping directly from intro to comparison-contract table (Results H2:1) could feel unmotivated if abstract and intro don't clearly signal it's coming. Requires careful intro rewording and explicit forward-reference to comparison contract.
- Figure caption and narrative coherence: Moving figures (e.g., sensitivity, factorial pymdp) to S03/S04 risks breaking narrative flow if main-text Results still assumes figure is nearby. Requires explicit supplementary cross-references in main text ([@sec:s03-...], [@fig:...]) and audit of all caption dependencies.
- Section file merge complexity: Merging 05_limitations + 07_roadmap + 08_conclusion into single 04_discussion.md requires careful preservation of all anchors, tokens, and hierarchical structure while re-organizing into 6 new H2s. High risk of losing content or breaking references during merge.
- Claim-level honesty contract alignment: New structure reorders when claims are introduced (e.g., fair-comparison result now in main Results instead of Roadmap). Must verify that eval_contract.py guards (G1 null-regime, G2 cue-disclosure) still apply correctly and are not invalidated by new placement.
- Roadmap reduction and reader expectation: Moving most of 07_roadmap (forward gates, AIF deepening, power analysis) into Results and Discussion (rather than keeping as separate Roadmap section) changes how readers perceive the research program. Requires explicit Discussion narrative explaining what remains open vs. delivered.
- PDF rendering and visual layout: Restructuring changes page breaks, figure placement, and table layout. New 04_discussion section may not render cleanly if 05+07+08 content is dense. Requires testing full PDF render after edits and possible caption trimming to manage page flow.

## Acceptance criteria
- All 14 existing manuscript sections (00, 01, 02a–d, 03, 05, 07, 08, 99, S01–S02) are successfully merged/restructured into 11 sections (00, 01, 02a–d, 03, 04, S01–S04, 99) with all {{TOKEN}} bindings intact and all [@sec:..., @fig:...] references resolving correctly in rendered PDF.
- New 03_results.md has exactly 8 H2 sections with result-bearing titles, new comparison-contract table inserted at H2:1, fair-inference ladder moved from 07_roadmap to 03_results H2:4 with all anchors and figures updated, and secondary diagnostics/robustness moved to S03/S04.
- New 04_discussion.md has exactly 6 H2 sections merging content from 05_limitations + 07_roadmap + 08_conclusion with new titles encoding result directions (synthetic-boundaries, SOTA-boundary, regime-bounds, honesty-stack, roadmap-summary, final-boundary); all anchors preserved/remapped.
- New S03_diagnostic_and_robustness_detail.md and S04_advanced_model_comparisons_and_transfer.md created with content extracted from 03_results and 07_roadmap; all {{TOKEN}} and [@sec:...] bindings functional; figures render with correct captions.
- Abstract (00_abstract.md) rewritten to position benchmark contribution first, fair-comparison result second, oracle limitation honestly, and concludes on honesty-discipline innovation rather than split message; tone shifts from 'where AIF helps' to 'how synthetic benchmarking with artifact-bound discipline reshapes evaluation'.
- Introduction (01_introduction.md) H2 subsections retitled to emphasize benchmarking first and active-inference-as-audit-tool second; explicit forward-reference to comparison-contract table and evaluation contract added; tone shift validated in rendered PDF.
- All internal cross-references (@sec:, @fig:, @tbl:) verified to resolve in rendered PDF; grep search confirms no dangling references; if any reference must change (e.g., [@sec:roadmap-aif-deepening] → [@sec:results-fair-comparison]), all occurrences updated globally.
- Evaluation contract (src/active_fractal_rabbit/eval_contract.py) still passes on rendered manuscript; no new claim-level violations introduced by restructuring; guards G1 (null-regime) and G2 (cue-disclosure) remain operative and do not contradict new result placements.
- Rendered PDF shows ~4–5 core figures in main Results body (benchmark schematic, model-comparison table/figure, fair-predictor ladder, location-minimization frontier, optionally sequence baselines), with secondary figures moved to S03/S04; Table of Contents reflects new structure (Discussion instead of Limitations+Roadmap+Conclusion); page count and visual layout acceptable (no extreme orphans/widows).
- All {{TOKEN}} variables referenced in restructured sections render correctly via scripts/z_generate_manuscript_variables.py; manuscript_variable_provenance.json passes validation checks; if any token provenance must be remapped due to new section placement, changes are documented and validated by scripts/11_check_manuscript_variable_provenance.py.
- Fair-comparison section (new 03_results H2:4) includes full fair-inference ladder (hard → soft → recursive → oracle-fed baseline), power-analysis regime map (traveller count × noise dimension), and honest conclusion (soft marginalization dominant, recursion conditional on noise, advantage regime-bound, never oracle-beating); all {{FAIR_*}} tokens render.
- Discussion section (new 04_discussion.md) explicitly states what this paper is and isn't: synthetic-data benchmark (not empirical mobility), honest-evaluation discipline (not SOTA claim), artifact-bound honesty stack (execution harness + claim contract), and required empirical-validation protocol gate for any real-trace claim; tone consistent throughout.
- Acceptance gates: (i) NO tokens unreferenced or orphaned; (ii) NO anchors duplicated, broken, or missing in rendered PDF; (iii) EVERY claim in Results matches introduction setup and Discussion honest boundary; (iv) EVERY [@sec:...] cross-ref resolves; (v) Evaluation contract passes cleanly; (vi) PDF renders without errors and reads as coherent IMRaD; (vii) Core narrative reads benchmark-first, AIF-as-audit-tool second, honest-limits-and-safety-boundaries prominent.
