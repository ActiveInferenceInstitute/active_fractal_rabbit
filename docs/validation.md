# Validation: Tests, Claim Gates, and Render QA Preserve the Boundary {#sec:validation}

## The Gate Has Separate Arms

`./run.sh` is the single gate entry point: `lint`, `test`, `test-full`, `docs`,
and `gnn` are separately runnable, and CI calls the same arms. `test-full` is
the documented readiness gate described below (full suite, 90% coverage floor).
The arm list, fast/full split, and measured wall times are in
[`RUN_GUIDE.md`](RUN_GUIDE.md) and
[`_generated/gate-timings.md`](_generated/gate-timings.md).

## Core Tests Protect Scaffold Interfaces

The core tests cover:

- parameter CSV ordering for FractalRabbit's published interface;
- strict parsing of the `ID,Days,x(km),y(km)` output contract;
- expanded config validation for grouped observation/model/experiment/figure/robustness settings;
- deterministic waypoint discretization and feature-table emission;
- normalized POMDP `A/B/C/D` arrays and JAX payload shapes for pymdp;
- tensor-summary consistency with generated matrices;
- first-observation sentinel handling for interval-derived features;
- fixture-based pipeline execution without Java or Maven;
- fixed-transition baseline traces and finite baseline metrics;
- policy-proxy component sums in `policy_trace.csv`;
- closed-loop candidate scoring from measured metrics, report-density scaling, and deterministic tie-breaks;
- fixture sensitivity summary statistics with mean, standard deviation, ranges, and normal-approximation intervals over deterministic variants;
- figure registry, style-provenance, and visual-quality audit generation with captions, alt text, source artifacts, palette/DPI/backend metadata, readable font-size contract, non-empty dimension-stable PNGs, caption source language, and data-derived source-file existence checks;
- visual-claim mini-brief generation for every registered figure plus the cover, including story job, source artifact, caption support/non-support contract, manuscript call site, and PDF-export fallback;
- manuscript token coverage, cross-reference style, figure-file existence, source-bound captions, and rendered-output scans for unresolved tokens or dangling `??` references;
- primary-source scholarship ledger coverage, verifying that every manuscript citation key is present in `manuscript/references.bib`, source-backed by DOI/arXiv/official URLs or official publisher/project pages, and mapped to allowed wording plus forbidden overreach before manuscript citation;
- manuscript claim inventory and claim-ledger coverage for high-risk active-inference, minimization, privacy, integrity, external-lane, gap-filling, factorial-latent, and empirical-boundary claim families;
- fail-loud manuscript-variable provenance for result-critical tokens, including missing artifact, missing key, checksum drift, rendered-value mismatch, and explicit unavailable handling;
- additive per-traveller chronological holdout split and scoring, including the
  flip test proving the `experiment.holdout_fraction` knob changes emitted metrics;
- seeded negative-control mutations (shuffled time, shuffled traveller IDs,
  degenerate spatial bounds, misspecified speed thresholds) with directional
  pass/fail probes and a seed flip test;
- sensitivity replicate counts driven by `experiment.repetitions` (flip-tested);
- figure output location driven by `figures.registry_path` (flip-tested);
- the five checked config fixtures under `configs/fixtures/` loading and running
  end-to-end;
- cover-art generation from canonical run artifacts, including fail-loud missing-artifact behavior and nonblank rendered-image checks;
- release-manifest checksum construction, verification, tamper detection, deletion detection, required publication metadata (`LICENSE`, `LICENSE-DOCS.md`, `THIRD_PARTY_NOTICES.md`, `CITATION.cff`, `docs/ip_license_status.md`), and coverage of the current robustness, safety-boundary, cover, and rendered-PDF artifacts;
- explicit external-lane skip reasons naming the missing tool or Java version;
- JDK 21 resolution order (env override, PATH, Homebrew kegs, java_home) with a
  live host probe;
- real pymdp agent construction, matched-prior agreement with the exact Bayes
  posterior (max abs delta < 1e-5 on the fixture), and the `model.engine` flip test
  proving the lane is knob-controlled and never replaces reference metrics;
- deep-latent modality selection: dropping `location` raises mean posterior
  entropy by more than 2x (directional flip test), with name-aligned observation
  indices for modality-subset models;
- ground-truth ingestion: PLACES parsing, strict XY/PLACES row-alignment (misalignment
  raises, never tolerated), place revisit statistics, and per-step state-recovery
  metrics for both inference lanes (hand-computed oracle cases);
- the directional deep-latent recovery probe: removing the location modality must
  strictly reduce mean posterior mass on the true cell;
- pymdp diagnostic emission: per-step variational free energy (via `return_info`),
  per-step and total runtimes, and per-policy expected free energy columns in
  `pymdp_trace.csv` (runtime keys are the only intentionally nondeterministic
  metrics in `metrics.json`);
- external FractalRabbit preparation and execution against the pinned commit,
  exercised for real when the toolchain resolves (it does on this host).

The tests prove the scaffold and interface boundaries. They do not prove that a
latent state recovered from synthetic waypoints is a real-world behavioral state.

## Objective and Calibration Gates Bound Program Extensions

This section records the objective-decomposition and calibration gates; the
later program-extension lanes (gap-filling anticipation, EFE decomposition,
routine-break detection, structure recovery, mission-conditioned collection,
the recoverability frontier, return-structure gating, variance bands,
upstream sensitivity, fractal-dimension sensitivity, circadian-phase signal,
  sequence-baseline gates, reporting-process/missingness gates, return-plus-phase prior gates,
  lane-resolved model-gate matrix construction, pymdp benchmarking,
privacy-frontier unicity, and hardened anomaly probes)
and the machine-checkable evaluation contract
are gated by their own test modules (`test_anticipatory`, `test_active_sensing`,
`test_anomaly`, `test_structure_recovery`, `test_mission_collection`,
`test_frontier`, `test_return_structure`, `test_robustness_artifacts`,
`test_fractal_dimension`, `test_circadian_phase`, `test_sequence_baselines`,
`test_missingness_model`, `test_return_phase_prior`, `test_model_gates`,
`test_pymdp_benchmark`, `test_sota_stressors`, `test_sota_program_harness`,
`test_challenge_pack`,
`test_privacy_frontier`, `test_hardened_anomaly`, `test_eval_contract`,
`test_claim_audit`, and `test_visual_claim_audit`) and reported in the manuscript's
program-extensions and validation sections.

- `objectives.py` decomposes the joint predictive loss into per-modality marginal losses under the lane's reconstructed prior trace; a binding gate refuses to emit numbers when the reconstruction does not reproduce the lane's own recorded loss (reference: exact; pymdp: within float32/variational tolerance).
- The marginal-vs-joint bound is theorem-grade and enforced per step in tests; the summed-marginal-minus-joint gap is reported as modality interaction information.
- Calibration (multiclass Brier, top-1 ECE with per-bin counts) is scored against exact ground-truth cells for every lane; skill scores are anchored to explicit uniform-chance floors so one is perfect and zero is chance.
- The holdout pass re-discretizes with training-only statistics (quantile edges, auto bounds, median burst cutoff) via `discretize_observations(..., estimation_mask)`; a flip test proves holdout outliers can no longer move training bins.
- Fixture-lane sensitivity replicates are flagged `deterministic_replicates: true` with an explicit bit-identical caveat; replicate spread in that lane is a determinism check, never sampling uncertainty.
- Sparsity levels emit `actual_fraction` (achieved subsample fraction under the two-report floor) alongside the requested fraction, plus per-level Brier and Brier-skill for the full and deep-latent lanes.
- The spatial-resolution frontier writes a deterministic sample-size ladder confirmation artifact. It checks whether the selected grid, best feasible grid, occupancy guardrail, and no-location guardrail direction reproduce at higher report counts; it is not an independent population interval.
- `tests/test_inference_integrity.py` pins filter semantics analytically: hand-computed Bayes posteriors, likelihood-product structure, transition column direction, softmax sign, and an independent mirror filter that must reproduce production posteriors exactly.
- `return_structure.py`, `fractal_dimension.py`, `circadian_phase.py`, `sequence_baselines.py`, `missingness_model.py`, `return_phase_prior.py`, `model_gates.py`, `pymdp_benchmark.py`, `privacy_frontier.py`, `hardened_anomaly.py`, and `robustness.py` add hand-computed tests for entropy reduction, dimension-sweep intervals, phase-conditioned occupancy, online sequence loss, reporting-process log loss, return-plus-phase promotion, lane-unavailable matrix rows, benchmark status, unicity, co-travel coincidence, Markov surprise, operating points, empirical intervals, and blocked upstream-sensitivity artifacts.
- `sota_stressors.py`, `sota_program_harness.py`, `aif_regime_search.py`, and `challenge_pack.py` now test deterministic reporting-gap, high-motion adversarial-missingness, canonical regime-stressor, repeated regime-stressor, regime-qualified AIF search controls, locked-split candidate scoring, diagnostic robustness-grid emission, blocked foundation rows, and the non-operational challenge-pack boundary.
- `release.py` now verifies required release artifacts in addition to checksums, so a manifest fails when required analysis JSON, provenance, or rendered PDF paths are absent rather than merely when listed files drift.
- `visual_quality.py` re-reads `output/figures/figure_registry.json` and every registered PNG, enforcing image dimensions, aspect-ratio sanity, nonblank pixel variance, source-bound captions, alt text, unique keys, style-provenance font floors, and existence of every data-derived CSV/JSON source before the manuscript is rendered. Its `quality_summary` names the shortest, narrowest, lowest-variance, and most extreme-aspect figures plus the contact-sheet surface for manual review, while `layout_critical_figures` names the claim-bearing plots that must be manually checked for clipping, label collisions, legends, and title stacking after regeneration. The current default registry contains 53 manuscript figures, with `evidence_atlas` as the opening orientation figure and `dataflow` as the artifact-contract diagram.
- The visual layer follows the same scholarship-backed convention used in the manuscript: align quantitative comparisons on shared axes where possible, show uncertainty with an explicit source artifact and boundary, and keep tables/JSON ledgers as the source of record when a plot is only the reader-facing view.
- `claim_audit.py` inventories manuscript claims by source section, classifies evidence markers, verifies high-risk claim families against `docs/claim_ledger.md`, checks that ledger-cited local paths exist, and — for ledger claims citing a JSON artifact and stating a decimal number — checks that number is actually findable in that artifact (within tolerance), not merely that the cited path exists. `visual_claim_audit.py` turns the visualization critique into one mini-brief per figure plus the cover, failing if a caption can support a claim without a corresponding non-claim boundary and recording each visual's reader task, data shape, encoding, and claim boundary.

## Figure and Reference QA Protect Rendered Claims

The manuscript uses Pandoc-crossref labels rather than hand-numbered prose. The
test suite rejects hand-numbered figure, equation, and table references,
verifies that every `[@fig:...]`, `[@eq:...]`, `[@tbl:...]`, and `[@sec:...]`
reference resolves to a declared label, and checks that generated figure
captions remain populated with source artifacts. Formal displays use `{#eq:...}`
labels on display math blocks; `\tag{...}` and raw LaTeX equation counters are
not part of the manuscript contract.
