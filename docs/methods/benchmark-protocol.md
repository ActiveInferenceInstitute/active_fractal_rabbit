# Benchmark Protocol: Locked Splits Keep SOTA Claims Blocked {#sec:benchmark-protocol}

This document specifies the state-of-the-art benchmark harness and claim-discipline framework. It covers the locked traveller/realization split, the prequential next-cell loss primary metric, the false-claim oracle, and the four claim gates that prevent illegitimate performance-leading language.

## The Task Contract Uses Locked Splits and Prequential Scoring

The benchmark harness establishes a *locked, leakage-resistant task contract* before scoring any candidate:

- **Traveller-disjoint partition:** Traveller identities are partitioned disjointly into train and holdout sets via `lock_traveller_split()` (seed deterministic, holdout fraction parameterized). No traveller appears in both the train prime and the holdout scoring set; train and holdout are composed of completely different travellers. This prevents traveller-identity leakage where a model's memorization of a specific traveller's transition statistics would inflate loss on nominally held-out steps from the same traveller.

- **Realization recorded per row:** Each observation row carries metadata recording whether it came from a fixture or external realization. This metadata is recorded in the task contract but does not affect split partitioning; it is available for auditing and lane-resolved analysis downstream.

- **Prequential scoring:** Online predictors are primed on the full train set to estimate population statistics (state-marginal frequencies, transition counts), then scored sequentially (prequentially) on holdout steps with online updates. This matches the real-time sequential structure: the model does not re-fit on holdout steps but does learn and adapt as it scores.

- **Primary metric:** Prequential next-cell predictive loss, measured in nats (negative log probability of the true next cell under the model's predictive distribution).

- **Secondary metrics:** Brier score (calibration), expected calibration error (ECE), persistence-relative gap-filling skill, runtime, location-minimization behavior, and poisoning/integrity robustness.

### Baseline Families Must Be Auditable

The harness defines required strong-baseline families. Implemented baselines are scored on the locked split in `sota_benchmark.py`; declared-unavailable families keep the claim gate closed:

| Family | Members |
| --- | --- |
| `transparent_null` | base-rate, persistence, first-order/phase/metadata Markov, return-boosted Markov |
| `classical_sequence` | switching/latent-regime HMM, state-space/particle mixtures |
| `neural_sequence` | dependency-light neural/logistic sequence learner |
| `foundation_mobility` | pretrained mobility transformer/mixture-of-experts (declared, unavailable) |
| `llm_mobility` | instruction-tuned/fine-tuned mobility language model (declared, unavailable) |

### Win Margins Need Confidence Intervals

A candidate must beat the strongest eligible baseline by at least `MIN_WIN_MARGIN_NATS = 0.02` nats. More stringently, the candidate's loss must fall *below* the strongest baseline's lower loss confidence interval edge (95% CI, bootstrap percentile), not merely beat the point estimate. This CI-separation guard rejects wins that sit inside the baseline's uncertainty band (an overlap within normal sampling variation).

## The Candidate Ladder Stays Locked

The `_LOCKED_CANDIDATES` tuple specifies baseline models scored entirely in-harness on the locked split:

1. **base_rate**: Marginal-frequency baseline, no context
2. **persistence**: Previous-cell, smoothed
3. **first_order_markov**: Markov model, no context
4. **phase_markov**: Markov model, phase-conditioned
5. **metadata_markov**: Markov model, phase/speed/burst-conditioned
6. **return_boost_markov**: First-order Markov with per-traveller visited-cell boost

All are deterministic, dependency-free, and transparent (source readable in `sota_benchmark.py`). Stronger baselines (neural, classical HMM, state-space) are marked as `locked_split_scored: False` and loaded from their own artifact analyses; they remain ineligible for a SOTA claim until re-scored under this contract.

## The False-Claim Oracle Exposes Leakage

The oracle runs an adversarial probe that *peeks at the true next cell* on every step (`mode="leakage_oracle"`). This probe is guaranteed to achieve the lowest possible loss on the held-out data because it has access to information that no real predictor can have. The oracle's mean predictive loss is included in the leaderboard but flagged with `leakage: True`.

The oracle verdict is derived, never asserted:

- If the leakage probe beats every honest baseline **and** is excluded by the leakage guard → "leakage probe achieves lower loss yet is excluded, so a leakage win cannot clear the gate."
- If the leakage probe does **not** beat the honest baselines → "negative control did not fire; a real baseline matched or beat the leakage probe, so the leakage exclusion was not exercised."
- If the leakage probe beats the baselines **and** is not excluded → "leakage guard failure" (system error).

This oracle proves that the leakage exclusion is not vacuous: it actively rejects a candidate that would otherwise dominate, making the guard's presence legible rather than merely asserted.

## Evaluation Guards Block Null, Cue, and Performance Drift

The module `eval_contract.py` implements three claim-level guards that apply to manuscript prose. These guards complement the harness execution-level controls and are *not familywise-error-controlled* (they are exploratory, claim-discipline machinery):

### G1 - Null-Regime Guard

A positive persistence-skill claim must cite a token bound to the gap-filling regime (`GAP_FILL_*`), never a location-blind regime token. Persistence is an oracle null in the location-blind regime because it receives the true previous cell while the location-blind filter does not; that is a benchmark-specific information-set asymmetry, not a universal impossibility result.

- **Violation:** A sentence claiming `{{BLIND_LIMIT_SCRP_SKILL}}` (location-blind token) as a positive skill without an explicit limit-framing marker (e.g., "oracle-bound", "weaker information set", "location-blind limit").
- **Fix:** Use `{{GAP_FILL_SCRP_SKILL}}` for positive claims or add limit-framing prose.

### G2 - Cue-Disclosure Guard

Any recovery/skill claim that leans on the speed cue requires the manuscript to disclose that speed is derived from inter-report displacement (a strong but disclosed kinematic cue). This prevents recovery claims that rest on an undisclosed feature.

- **Violation:** Token `GAP_FILL_SCRP_SKILL` appears in the manuscript but none of ("derived from inter-report displacement", "inter-report displacement", "displacement-derived") appears anywhere in the corpus.
- **Fix:** Add disclosure language explaining the source of the speed feature.

### G3 - Performance-Claim Guard

Positive state-of-the-art or performance-leading language (or active paraphrases like "outperform", "best-in-class", "benchmark record") must be explicitly negated, future-scoped, or bound to a gate-status token. Gate-token presence alone does not launder active leadership verbs; they require explicit limit-framing.

- **Violation:** "outperforms all baselines" without negation, future-scoping, or gate-bound status token.
- **Fix:** Negate ("does not outperform"), future-scope ("would outperform if..."), or bind to a status token ("the gate reports {{SOTA_CLAIM_GATE_STATUS}}, which currently blocks performance-leading language").

Status-style markers ("state-of-the-art") may be gate-token-bound; assertive leadership verbs may not.

## The Claim Gate Remains `sota_not_cleared`

The gate collects reasons that block performance-leading language. It is **cleared only when all of the following hold:**

1. All required baseline families are present on the locked split (no missing families).
2. Current-run external-lane evidence exists (no fixture-only wins).
3. An eligible candidate beats the strongest baseline by at least `MIN_WIN_MARGIN_NATS` nats and is CI-separated.
4. The candidate does not leak (not the leakage oracle).
5. The candidate is locked-split-scored (not a full-stream artifact from an external analysis).
6. Calibration does not regress (candidate Brier ≤ best-baseline Brier).
7. Location-minimization is satisfied or unverified → fails closed (`minimization_ok` is `True`).
8. Poisoning/integrity robustness is satisfied or unverified → fails closed (`integrity_ok` is `True`).

If any condition fails, a reason is appended and `status = "sota_not_cleared"`. The manuscript token `{{SOTA_CLAIM_GATE_STATUS}}` is provenance-bound to the gate artifact and automatically rejects SOTA language when the gate is not cleared.

### Current Status Blocks Promotion

As of the roadmap milestone (§[Roadmap](../../manuscript/07_roadmap.md)), the gate reports `sota_not_cleared` because:

- Required foundation-model and language-model baseline families are declared unavailable.
- No implemented candidate beats the strongest honest baseline on the locked split.
- The transparent baselines dominate because they receive the true previous cell as an oracle; no presently-implemented non-oracle method improves on them.

This is not a reflection of effort; it is the intended outcome of a design that makes defeats legible rather than hidden.

## Artifact Outputs Carry the Gate

The harness emits four artifacts via `_write_artifacts()`:

1. **`sota_task_contract.json`**: Task specification (primary/secondary metrics, locked split, baseline families, win margin, boundary caveats).

2. **`sota_benchmark_matrix.json`**: Full benchmark matrix including locked-split baseline rows, full-stream candidate rows, leakage probe, candidate evaluation detail, and claim-gate reasoning.

3. **`sota_leaderboard.csv`**: Human-readable leaderboard sorted by mean predictive loss, with columns: `model`, `family`, `lane`, `locked_split_scored`, `leakage`, `mean_predictive_loss`, `loss_ci_low`, `loss_ci_high`, `brier_score`, `ece`, `runtime_seconds`.

4. **`sota_claim_gate.json`**: Gate status, reasons, candidate evaluation failures, minimization/integrity verdicts, and false-claim oracle summary.

The separate SOTA-program harness (`sota_program_harness.py`, run as its own full-study stage) emits two further artifacts:

5. **`sota_program_leaderboard.csv`**: Four-lane fair-frontier table with one row per
   candidate and lane. The row contract includes predictive loss, traveller-clustered
   loss intervals, Brier/ECE, top-1 accuracy, runtime, lane coverage, dependency
   footprint, scored-or-explicitly-blocked robustness status, and failure-mode notes. The
   paired AIF-vs-strongest-fair interval remains separate from per-candidate loss
   intervals and is populated on the AIF rows.

6. **`sota_program_robustness_grid.json` / `.csv`**: Diagnostic stressor suite scored
   through the same candidate interface, currently covering high-motion adversarial
   missingness and a repeated regime-stressor realization. These rows can block
   overclaiming and guide future work, but they are synthetic-only diagnostics and cannot
   clear SOTA, empirical-transfer, or operational claims by themselves.

## Honest Reach Ends at Synthetic Evidence

The harness catches:

- **Leakage** (peking at the true outcome): Actively rejected by the oracle guard.
- **Fixture-only wins** (no external-lane confirmation): Blocked by requiring `external_available`.
- **Single-metric wins** (loss improvement with calibration regression): Blocked by the calibration guard.
- **Minimization regression** (win depends on direct location): Blocked by `minimization_ok` verdict from the location-minimization frontier artifact.
- **Integrity regression** (win fails poisoning diagnostics): Blocked by `integrity_ok` verdict from the poisoning-robustness artifact.

The harness **does not** execute-level-catch (those are caught only by running the full pipeline):

- Hardcoded-number fabrication.
- Overfitting via grid-search on the holdout set.
- Incorrect feature engineering or preprocessing.

The harness is a claim-level and gate-level discipline, not an execution-level replacement. It works alongside the full evaluation pipeline, execution-level negative controls, and the claim-level linter (`eval_contract`).

## Related Documents Carry Adjacent Contracts

- [../manuscript/07_roadmap.md](../../manuscript/07_roadmap.md): Forward gates and state-of-the-art decision boundary.
- [validation.md](../validation.md): Test surface including the SOTA benchmark test suite.
- [empirical_validation_protocol.md](../empirical_validation_protocol.md): Real-waypoint validation gate (separate, prerequisite for real-trace claims).

## References Anchor the Protocol

- `sota_benchmark.py`: Lock traveller split, score locked baselines, run false-claim oracle, emit leaderboard and gate status.
- `eval_contract.py`: Machine-checkable claims guards (G1/null-regime, G2/cue-disclosure, G3/performance-claim).
- `sota_benchmark.run_and_write_sota_benchmark()`: Top-level entry point for running the benchmark and writing artifacts.
