# Fair-Frontier Robustness: Oracle-Stripped Comparisons Stay Regime-Bound

## Oracle-Fed Baselines Define the Information Asymmetry

The transparent baseline ladder (base-rate, persistence, first-order Markov, return-boosted Markov) achieves its lowest prequential next-cell loss by receiving the true previous cell as an oracle at prediction time. This is a *leaderboard predictor*, not a *deployable* predictor: a system that sees only sparse location observations must *infer* the previous cell before predicting the next. The "SOTA" comparison of oracle-fed baselines vs. any inference-based candidate is therefore misleading — it measures what oracle knowledge is worth, not whether principled inference helps.

The honest question is different: **given the emission alone, how much does Bayesian inference of the latent cell improve prediction over a hard point estimate?** This requires a fair comparison where every predictor faces the same deployable information — no oracle — and differ only in how they estimate the previous cell from the observed location signal.

See `fair_inference.py:fair_inference_ladder()` and `sota_benchmark.py` for implementation.

## Fair Comparisons Share the Transition Source

The fair ladder (`fair_inference.py:fair_inference_ladder()`) scores six predictors on the same locked train/holdout split, all learning the transition source from the same true observed cells (the leakage guard). The only thing that varies is how the previous-cell estimate enters the prediction:

1. **oracle** — the true previous cell (the cheat; reference for how much oracle knowledge is worth)
2. **hard_handicapped** — argmax of P(cell | location obs) (hard point estimate; the deployable baseline)
3. **soft_handicapped** — full emission posterior P(cell | obs), marginalized through the transition (single-step Bayesian inference)
4. **soft_plus_return_boost** — soft Markov plus per-traveller visited-cell prior (non-recursive fair competitor)
5. **honest_aif** — recursive active-inference belief (soft inference + temporal accumulation)
6. **honest_aif_plus_boost** — honest AIF with the same return boost

All are scored under prequential next-cell `-log loss` on the locked holdout steps, with bootstrap confidence intervals (noting the per-step bootstrap overstates precision; see [Regime Analysis](#regime-analysis) for traveller-clustered intervals).

**Key controls:**
- The handicap is real: every handicapped predictor loses to the oracle-fed baseline (positive control `handicap_is_real`).
- An identity emission collapses the handicap: hard inference under perfect observations matches the oracle (positive control `identity_emission_collapses_handicap`).
- None beat the oracle: the fair comparison is never an oracle-beating or SOTA claim (`aif_still_loses_to_oracle`).

## Soft Marginalization Carries Most of the Gain

From the regenerated ladder fixture (parameters and all values below are recorded in `output/runs/fair_inference.json`; the ladder itself runs at the cleanest emission, smoothing=0.05):

- **Soft marginalization dominates recursion.** The hard→soft single-step gain (`ladder.decomposition_nats.soft_marginalization_gain`) is the dominant inference gain; on the regenerated fixture the soft→recursive gain (`ladder.decomposition_nats.recursive_belief_gain`) is *negative*, so recursion adds no further gain at this clean setting.
- **Recursion's gain is noise-conditional.** At the cleanest emission (smoothing=0.05), recursion is *slightly worse* than single-step soft Markov. The advantage becomes positive and grows as the observation channel degrades (`noise_sweep.points`; ≈0.36 nats at the noisiest setting, smoothing=2.0, at time of writing — `noise_sweep.magnitude_is_seed_sensitive` is true, so the magnitude is illustrative, not frozen).
- **Return-boosted soft does not beat bare AIF on the current fixture.** The non-recursive `soft_plus_return_boost` predictor (shared population transition + per-traveller visited cells) loses to `honest_aif` without the boost (`ladder.return_boosted_soft_beats_bare_aif` is false): on this regenerated fixture the visited-cell boost worsens both predictors rather than substituting for recursion.
- **Matched features are the honest comparison.** The visited-cell return prior is scored on both sides (`soft_plus_return_boost`, `honest_aif_plus_boost`); whether it helps either predictor is a regenerated-artifact fact recorded in `fair_inference.json` (`ladder.aif_wins_at_matched_features`), not a frozen doc claim — on the current fixture it worsens both predictors.

**Verdict from `fair_inference_ladder()` (current artifact):**
> Fair-comparison result, not SOTA. With the transition source held identical, soft Bayesian marginalization is the dominant inference gain over a hard point estimate; recursive temporal belief does not add a gain on this regenerated fixture. In the return-feature comparison, the return-boosted soft Markov does not beat the boost-free recursive filter on this regenerated fixture, and the active-inference filter does not lead after matching the visited-cell feature. Every predictor that must infer the cell loses to the oracle-fed baseline.

See `fair_inference.py` lines 143–164 for the decomposition metrics.

## Multi-Seed Robustness Remains Fixture-Bounded

The single-split result above rests on ~1–2 holdout travellers, making any CI-separated claim underpowered. The multi-seed verdict (`fair_inference.py:fair_sota_robustness()`) aggregates across 8 seeds (default `DEFAULT_SOTA_SEEDS = tuple(range(20260616, 20260616 + 8))`) to check directional robustness. The sample size is deliberately kept small to expose the limitation: each seed reshuffles the same few travellers, so the signal is *directional and provisional*, not a formal power claim.

**Key findings (DEFAULT_SOTA_SEEDS across holdout_fraction=0.5; values read from the regenerated `fair_inference.json` `sota_robustness` block, illustrative at time of writing):**

| Metric | Result |
|--------|--------|
| AIF-class best-fair rate (`aif_class_best_fair_rate`) | 0/8 seeds — the honest null: no AIF predictor is the lowest-loss fair predictor on any seed |
| AIF+boost best-fair rate (`aif_plus_boost_best_fair_rate`) | 0/8 seeds |
| Mean AIF advantage (`aif_advantage_mean_nats`) | negative (≈ −0.35 nats vs. the strongest non-AIF fair baseline — the fair baseline wins on average) |
| Per-step CI separation rate (`step_level_ci_separated_rate`) | 0/8 seeds (and per-step overstates precision) |
| Distinct holdout travellers (min) | ~2 travellers per split |
| Oracle still lower (positive control) | 8/8 seeds (handicap is real) |

The multi-seed result is therefore an honest null, not a best-fair-predictor claim: per the artifact verdict, "an active-inference predictor is the lowest-loss fair predictor in 0/8 seeds; it is NOT unanimous (0/8) and does not establish a best-fair-predictor result."

**Sample-size caveat:**
Per-step bootstrap treats within-traveller steps as i.i.d. and overstates precision. A traveller-clustered bootstrap is degenerate at ~2 distinct travellers (resampling 2 clusters cannot estimate between-cluster variance), so **no valid uncertainty interval exists — CI-separation is withheld, not merely wide**. This is a directional, hypothesis-generating result whose prerequisite for a precision claim is more holdout travellers and larger synthetic regimes, not a cleverer filter.

**Cross-vendor status:** Cross-vendor audit is constraining evidence rather than a promotion path. The project record now marks multiple apparent advantages as caught only by cross-vendor review, so the fair-frontier result remains fixture-bound, directional, and non-SOTA unless a future artifact proves a stronger claim.

See `fair_inference.py:fair_sota_robustness()` lines 220–387 for the full verdict generation.

## Precision Self-Improvement Is In-Sample Only

Within the fair frontier, the AIF belief can be sharpened with a predictive precision parameter γ (inverse temperature). The baseline setting is γ=1.0 (uniform temperature); a precision sweep identifies γ=1.5 as the fixture-optimized sweet spot.

**Paired comparison (same split, γ=1.0 vs. γ=1.5, across DEFAULT_SOTA_SEEDS):**

| Metric | γ=1.0 (baseline) | γ=1.5 (sharp) | Improvement |
|--------|-----------------|---------------|-------------|
| Mean loss (nats) | `sota_robustness.mean_loss_gamma1` in `fair_inference.json` | `sota_robustness.mean_loss_precision` | Lower in 8/8 seeds (`precision_gain_loss_win_rate`) |
| Mean ECE | `sota_robustness.mean_ece_gamma1` | `sota_robustness.mean_ece_precision` | Lower or equal in 8/8 seeds (`precision_ece_win_rate`) |

(Docs are not token-hydrated; read the numeric values from `output/runs/fair_inference.json` or the rendered manuscript.)

This is an **in-sample inference-internal improvement, not a generalization claim**. The γ parameter is fixture-selected and not held out, so it cannot be used to artificially inflate the between-predictor comparison. The improvement is deliberately excluded from the fair-frontier verdict above: precision tuning helps AIF's own calibration but does not feed into the margin over non-AIF fair predictors.

See `fair_inference.py:fair_sota_robustness()` lines 292–303 for the paired analysis.

## Regime Analysis Maps Scale and Noise Separately

The small-traveller limitation is testable. The power analysis (`power_analysis.py:fair_power_regime_map()`) sweeps two axes on deterministic synthetic data:

- **Traveller count** (4, 8, 16, 32 travellers on a fixed 4×4 grid)
- **Observation noise** (emission smoothing 0.05 [clean], 0.5 [default], 2.0 [noisy])
- **Generator robustness** (5 independent draws per cell, seeded 2026–2030)

The advantage is intervalled with a **traveller-clustered bootstrap** (resampling whole travellers, the correct correlated unit), and a cell is a **seed-robust win** only if its clustered CI separates from zero in *every* generator draw — the across-draw spread is the between-dataset variance the bootstrap cannot see.

### Regime-Map Findings Stay Local

Values below are read from the regenerated `output/runs/power_analysis.json` and are illustrative at time of writing:

| Regime | Advantage (nats) | Seed-Robust? | Interpretation |
|--------|------------------|--------------|-----------------|
| Few travellers (4) + clean (0.05) | ~+0.41 ± 0.03 | **Yes** | AIF wins decisively when emission is clean but data per cell is sparse |
| Many travellers (32) + clean (0.05) | ~−0.005 | **No** | Advantage vanishes at scale under clean emission; shared transition model becomes sufficient |
| Many travellers (32) + noisy (2.0) | ~+0.05 ± 0.003 | **Yes** | Advantage re-emerges as a monotone gradient with noise; AIF's temporal belief earns its keep |
| Noise gradient at 32 travellers | 0.05 → 0.5 → 2.0 | Monotone positive | Advantage grows steadily with observation noise (the honest lever) |

### Key Insights Do Not Promote SOTA

1. **Small-scale region:** With few travellers and clean observations, AIF's recursive temporal belief is a decisive advantage (~0.4 nats). This regime dominates the target sparse-waypoint condition where data per cell is deliberately limited.

2. **Vanishing at scale (clean):** As traveller density increases on a fixed grid, the shared population transition model becomes sufficient. The advantage shrinks to noise and slightly reverses (~−0.005 nats at 32 travellers, clean). This is a **density effect**, not a count-intrinsic law: at fixed grid resolution, traveller count and per-cell occupancy density are deliberately coupled. A finer grid delays the onset; constant-anchor-density checks confirm it is not a fixed-box artifact.

3. **Restoration under noise:** Even at scale (32 travellers), the advantage re-emerges as observation noise increases. At the noisiest setting (smoothing=2.0), the AIF filter achieves a seed-robust win (≈+0.05 nats on the current artifact). The noise axis is **monotone** — the gradient is consistent across all tested levels, making it a structural finding, not a cherry-picked cell.

4. **Magnitude hierarchy:** The few-traveller clean effect (~0.4 nats) is roughly an order of magnitude larger than the at-scale noisy one (~0.05 nats). Cells are drawn from a 2D sweep, so the monotone **gradient is the claim**, not any single cell clearing zero.

### Deployment Claims Remain Out of Scope

The regime map reveals an honest scope for recursive temporal belief under sparse-waypoint tracking:

- **Where AIF helps:** High uncertainty per cell (few observations, noisy emission), the intended partial-observability condition in this synthetic benchmark.
- **Where a boosted Markov suffices:** High data density (many travellers, clean observations), where the population transition model alone carries enough signal.
- **Operational boundary:** None of this beats the oracle-fed baseline, so it is a fair-frontier regime characterization, never an oracle-beating or operational claim.

See `power_analysis.py:fair_power_regime_map()` lines 179–305 for the full sweep and verdict.

## Regime-Qualified AIF Search Is Separate from Global SOTA

The separate `aif_regime_search.py` surface asks a narrower question than the
global SOTA gate: are there synthetic regimes where the best active-inference
family candidate beats every implemented non-AIF fair baseline under the same
locked traveller split, train-only transition learning, known observation
channel, and traveller-clustered CI rule?

It scores identity-emission collapse, symmetric noisy emission, ambiguous
coarsened emission, burst reporting gaps, and latent regime-switching
transitions across configured realization seeds. The identity-emission row is a
negative control and cannot promote; oracle-fed rows remain ceilings, not fair
baselines. A regime clears only if the best AIF-family candidate has lower mean
loss than the strongest fair baseline and the paired clustered CI separates
from zero in every configured realization while staying above the
oracle/leakage floor. The output can support only "regime-qualified synthetic
SOTA against implemented fair baselines" wording. It cannot clear external,
empirical, operational, foundation-model, or global SOTA claims.

## SOTA Claim Gate Status Remains Blocked

The claim gate in `sota_benchmark.py:run_sota_benchmark()` requires:

1. **Leakage guard:** Candidate must beat the leakage oracle (peeks true next cell) or the guard excludes it. ✓ The leakage probe is near the oracle floor and lower than the strongest honest baseline; excluded rather than treated as a competitor.
2. **CI-separation:** Candidate loss must beat the strongest baseline's lower CI edge, not just the point estimate. ✗ No fair-comparison AIF configuration beats the strongest baseline's CI.
3. **Calibration:** Candidate must not regress on Brier score or ECE. ✓ AIF with precision improves calibration.
4. **No location minimization regression:** Must pass the location-withholding frontier test. ✓ (Separate artifact; provisional.)
5. **No integrity regression:** Must survive poisoning diagnostics. ✗ On the regenerated fixture the poisoning artifact's verdict is `poisoning_detection_regime_changed` (gross and moderate teleport detection did not clear against the Markov baseline), so `integrity_ok` is false in `sota_claim_gate.json`.
6. **Gate-specific external evidence:** Must show the current-run external evidence consumed by this locked-split SOTA gate. ✗ The broader pipeline has external simulator artifacts, but this gate-specific field remains uncleared.
7. **Required baseline families:** Four strong-baseline families are declared but unavailable on the locked split (`sota_claim_gate.json` `missing_baseline_families`: classical-sequence, foundation-mobility, LLM-mobility, neural-sequence). ✗ Missing families keep the gate closed.

**Current status: `sota_not_cleared`**

The gate is not vacuous — a clean, calibrated, externally confirmed candidate with no regressions would clear it — but no current method meets that condition. The oracle-fed baseline and fair-frontier result are permanently non-clearing by design: fair comparison is never SOTA.

See `sota_benchmark.py` lines 258–425 for the gate logic and `manuscript/07_roadmap.md` section "State-of-the-Art Gates Still Not Cleared" for the forward path.

## Related Analyses Preserve the Same Boundary

- **Factorial POMDP comparison** (`factorial_pymdp.py`): Experiments with multi-factor latent states; returns honest nulls on improvement.
- **Locked-split AIF sweep** (`aif_sweep.py:run_aif_sweep()`): Searches configuration space on the standard baseline. Best loss still above baseline CI edge.
- **Leakage oracle** (`sota_benchmark.py:_score_predictor()` mode="leakage_oracle"): Proves the leakage guard is not vacuous; peeks the true cell and is excluded.
- **Minimization frontier** (`efe_location_frontier.py`): Tests whether location dominates the information budget; provisional oversight-positive verdict.
- **Poisoning robustness** (hardened anomaly probes): Tests whether the method detects stream tampering; the current artifact verdict is `poisoning_detection_regime_changed`, which blocks the SOTA gate's integrity check.

## References and Artifacts

- Implementation: `/src/active_fractal_rabbit/{fair_inference.py, aif_sweep.py, power_analysis.py, sota_benchmark.py}`
- Specification: `../generative_model_card.md`
- Forward gates: `../manuscript/07_roadmap.md` sections "State-of-the-Art Gates Still Not Cleared" and "Active-Inference Lane Deepening and Honest Nulls"
- SOTA contract: `../../output/runs/sota_task_contract.json`
- Locked-split leaderboard: `../../output/runs/sota_leaderboard.csv`
- Four-lane SOTA-program gate: `../../output/runs/sota_program_gate.json`
- Four-lane SOTA-program leaderboard: `../../output/runs/sota_program_leaderboard.csv`
- Diagnostic SOTA robustness grid: `../../output/runs/sota_program_robustness_grid.json`
  and `../../output/runs/sota_program_robustness_grid.csv`
- Regime-qualified AIF search: `../../output/runs/aif_regime_search.csv`,
  `../../output/runs/aif_regime_search.json`, and
  `../../output/runs/aif_regime_sota_gate.json`
- Claim gate artifact: `../../output/runs/sota_claim_gate.json`
- Fair-inference artifact: `../../output/runs/fair_inference.json`
- External fair-realization manifest: `../../output/runs/fair_inference_external_realizations.json`
- Power-analysis artifact: `../../output/runs/power_analysis.json`

The four-lane SOTA-program leaderboard is intentionally wider than the canonical locked-split
CSV: each candidate/lane row carries predictive loss, traveller-clustered loss intervals,
Brier/ECE, top-1 accuracy, runtime, lane coverage, dependency footprint, explicit
robustness status, and failure-mode notes. The current canonical lanes are recorded in
`sota_program_gate.json`: FractalRabbit, gravity/preferential-return, reporting-gap
burst-missingness, and larger regime-stressor. These fields support only synthetic
robustness wording; they do not clear SOTA, empirical-transfer, or operational claims.
The diagnostic robustness grid uses the same row contract for high-motion adversarial
missingness and repeated regime-stressor checks. It is a stricter synthetic audit surface,
not an expanded manuscript claim gate. The external fair-realization manifest likewise
blocks promotion beyond fixture language until every required current external realization
is present and agrees on the oracle-boundary controls.

## Boundary Statement Blocks Promotion

**Synthetic-only.** Fair-comparison advantage, not oracle-beating or operational. No real waypoint, deployment, surveillance-readiness, or operational tasking claims. Empirical validation requires separate consent and provenance protocol (`../empirical_validation_protocol.md`).
