# pymdp-push-fair-frontier-optimization

> **HISTORICAL PLANNING RECORD — retained for provenance; superseded by current artifacts.**
> This is a frozen planning snapshot from the revision program. Its baseline numbers (including
> the oracle gap quoted below) were measured on an earlier fixture and do not match the current
> `output/runs/fair_inference.json`; its objective ("top the fair-frontier decisively") was
> overtaken by the program's final confound-controlled verdict that no robust active-inference
> advantage survives on this benchmark. Some levers were deliberately not executed. Do not cite
> numbers, goals, or file/line references from this file.

# PYMDP-PUSH: Ambitious Fair-Frontier Performance Optimization Plan

## Executive Summary
The current pymdp/active-inference lane achieves a fair-oracle-stripped win (soft Bayesian marginalization beats hard-argmax Markov on the oracle-stripped task), but trails the oracle-fed baseline by +2.237 nats and shows modest relative improvement. This plan targets **closing the gap to the fair-frontier leaders** through six concrete, testable levers that exploit the unique strengths of recursive belief accumulation under partial observability. Expected outcome: **top the fair-frontier decisively** with 50-150 bps gains while likely remaining below the oracle (honest). Each lever includes a cheap kill-gate threshold.

---

## LEVER 1: Joint/Pairwise-Emission Interactions
**File:** `src/active_fractal_rabbit/pomdp.py`, `_emission_matrix()` (lines 129-139)

**Mechanism:**  
Currently, `_emission_matrix` builds marginal P(obs_m | cell) independently per modality and the filter multiplies them as product-of-marginals. Under sparse observations (240→480 reports across 6 travellers over 36 days), time-of-day, speed, and report-burst are NOT independent conditioned on the cell. A pairwise-interaction term (e.g., P(location, time_of_day | cell) factorization-relaxed) captures co-occurrence structure the product assumption discards.

**Implementation Strategy:**
1. Add optional parameter `emission_interaction_mode: str = "product"` to `build_pomdp()`.
2. Implement two variants:
   - `"product"`: keep current behavior (marginal product)
   - `"pairwise_relaxed"`: build a secondary matrix A_joint from (location, time_of_day) co-occurrences, use a mixture or product with the full-modality marginal (e.g., 0.7 × A_location × A_time × A_speed × A_burst + 0.3 × A_joint × A_speed × A_burst)
3. Sweep in `aif_sweep.py` via `default_sweep_grid()`: add configs with `emission_interaction_mode="pairwise_relaxed"`.
4. Score on the 6-traveller/480-report fixture.

**Expected Effect:** 20–80 bps loss reduction under noisy reporting (emission conditional independence is a strong, often-violated assumption in real sparse data).

**Kill-Gate:** If pairwise variant increases loss >10 bps, it is a null (marginal product likely already near-optimal for this structure or the added complexity hurts cross-traveller generalization).

---

## LEVER 2: Learned Online Transition Adaptation (Amortized)
**File:** `src/active_fractal_rabbit/aif_sweep.py`, `score_pomdp_on_locked_split()` (lines 60–219)

**Mechanism:**  
The current `online_adapt=True` uses raw Dirichlet posterior updates on transition counts during holdout scoring. This is honest prequential learning but ignores temporal structure: the transition distribution may drift slowly (e.g., a traveller shifts from exploration to dwelling). A hierarchical online update (learn a separate adaptation rate per cell or per traveller) or an exponential-moving-average blend (new counts weighted 0.2, old counts 0.8) allows the filter to track gradual shifts without overfitting to noise.

**Implementation Strategy:**
1. Add new parameters to `score_pomdp_on_locked_split()`:
   - `online_adapt_mode: str = "simple"` (current: raw count increments)
   - `online_decay: float = 0.8` (EMA blend; ignored if mode != "ema")
   - `online_per_traveller: bool = False` (separate transition counts per traveller_id during holdout)
2. Implement three variants:
   - `"simple"`: current behavior (baseline)
   - `"ema"`: blend new counts as `(1 - decay) * new + decay * old`
   - `"per_traveller"`: maintain separate B estimates per holdout traveller, use traveller-specific transitions in prediction
3. Sweep: test `online_adapt=True` with `online_decay in (0.7, 0.8, 0.9)` and `online_per_traveller in (True, False)`.
4. On 6-traveller fixture, per-traveller estimates are risky (each traveller ~80 holdout steps) but may help if structure is strong.

**Expected Effect:** 10–40 bps if decay rate is tuned right; per-traveller may overfit and increase loss.

**Kill-Gate:** If EMA variants all increase loss >5 bps or per-traveller CI is wider than simple, the current online update is already near-optimal on this fixture.

---

## LEVER 3: Hierarchical/Super-Cell Backoff Transitions
**File:** `src/active_fractal_rabbit/learned_stay_rate.py` (lines 122–153 already prototype this)

**Mechanism:**  
The spatial grid has latent structure: adjacent cells are more likely to share transition patterns. A two-level hierarchy (super-cell: coarse-grained grid region → fine-grained cell) allows the transition prior to back off to a coarse estimate when the fine estimate is sparse. This is implemented in `_super_cell_hierarchy_loss()` but NOT in the core `aif_sweep.py`. Integrate it as an optional B matrix construction in `pomdp.py`.

**Implementation Strategy:**
1. Add `transition_hierarchy_mode: str = "flat"` to `build_pomdp()`.
2. Implement two variants:
   - `"flat"`: current behavior (full fine-grained transition matrix)
   - `"super_cell_backoff"`: compute both a fine B (states × states) and coarse B (supercells × supercells), then interpolate in forward pass (e.g., 0.6 × fine + 0.4 × coarse_expanded_to_fine).
3. Integrate into `aif_sweep.py`: add `transition_hierarchy_mode` to config grid.
4. Super-cell mapping: for a 6×6 grid (36 cells), define super-cells as 3×3 = 9 regions; for 9×9 (81 cells) use 4×4 regions.

**Expected Effect:** 30–100 bps if travellers show regional structure (e.g., stay within neighborhoods). If cells are truly ergodic and independent, minimal gain.

**Kill-Gate:** If super-cell-backoff increases loss >5 bps, the data do NOT show clear hierarchical structure; revert to flat.

---

## LEVER 4: Precision/Temperature Scheduling (Adaptive γ)
**File:** `src/active_fractal_rabbit/aif_sweep.py`, `_sharpen()` (lines 326–330)

**Mechanism:**  
Currently, `precision` is a fixed scalar sharpening exponent applied uniformly to the predictive distribution. The manuscript reports γ=1.5 as a measured sweet spot (lowers loss + ECE). However, the optimal γ may change as the posterior accumulates evidence: early in a traveller's stream (high posterior entropy), a sharper γ may hurt; later (confident), sharpening helps. Implement a posterior-entropy-dependent schedule: γ(H) = 1.0 + (1.5 - 1.0) × [1 - exp(-α × H)] where H is the posterior entropy of the previous step, and α controls the responsiveness.

**Implementation Strategy:**
1. Add parameters to `score_pomdp_on_locked_split()`:
   - `precision_schedule: str = "fixed"` (current: constant γ)
   - `precision_base: float = 1.0` (minimum γ)
   - `precision_peak: float = 1.5` (maximum γ from fixture tuning)
   - `precision_entropy_scale: float = 0.5` (α parameter for sensitivity)
2. In the forward loop (around line 169), compute entropy of the prior and apply schedule: `gamma_t = precision_base + (precision_peak - precision_base) × f(H_t)`.
3. Sweep: test fixed γ ∈ {1.0, 1.5} vs. scheduled with entropy_scale ∈ {0.3, 0.5, 0.8}.

**Expected Effect:** 5–30 bps if the schedule is well-calibrated; wrong α increases loss.

**Kill-Gate:** If any scheduled variant increases loss >10 bps, entropy-dependent scheduling is premature; keep fixed γ=1.5.

---

## LEVER 5: Return-Structure Prior Mixture (Visited-Cell Clustering)
**File:** `src/active_fractal_rabbit/aif_sweep.py`, `score_pomdp_on_locked_split()` (lines 147–191)

**Mechanism:**  
The current `return_boost=1.0` adds uniform mass to visited cells. In reality, not all visited cells are equally likely to be re-visited: a traveller may have a few "home bases" and larger regions visited only once. Implement a learned per-traveller return-visitation frequency: cells visited N times contribute N² to the boost instead of a flat 1. This captures hierarchical revisit structure.

**Implementation Strategy:**
1. Add parameter `return_boost_mode: str = "uniform"` to `score_pomdp_on_locked_split()`.
2. Variants:
   - `"uniform"`: current behavior (equal mass to all visited)
   - `"frequency_squared"`: `seen[cell]^2` in line 171
   - `"log_frequency"`: `log(1 + seen[cell])` to soften the exponent
3. In the loop, track per-traveller visit counts and apply the mode before normalization.
4. Sweep: test with `return_boost=1.0` and modes `{"uniform", "frequency_squared", "log_frequency"}`.

**Expected Effect:** 10–50 bps if re-visit structure is strongly non-uniform; negligible if visits are nearly uniform.

**Kill-Gate:** If any non-uniform mode increases loss >8 bps, uniform revisit assumption is adequate.

---

## LEVER 6: Multiscale Cell Representations (Deep Latent Lane)
**File:** Create new file `src/active_fractal_rabbit/multiscale_latent.py`

**Mechanism:**  
The current latent state is the discrete cell. A richer latent state (cell + continuous position within the cell, or a latent hierarchy: region → cell → sub-cell) could reduce emission noise and improve predictive accuracy. However, this is a deeper change: instead of a flat cell state, use a tuple (region, cell) as the state and learn factored transition B matrices for each. This mirrors the super-cell idea but inverts it: instead of coarse backoff, maintain an explicit multi-level state.

**Implementation Strategy:**
1. Create a new `multiscale_latent.py` module with:
   - `build_multiscale_pomdp()`: construct a 2-level state space (region, cell)
   - `score_multiscale_on_locked_split()`: parallel to `score_pomdp_on_locked_split()` but with stacked states
2. State vector: (n_regions × n_cells_per_region,) instead of (n_cells,).
3. Transition B: learn P(region', cell' | region, cell); can factor as P(region' | region) × P(cell' | cell, region') or keep fully crossed.
4. Emission A: unchanged (observe location → cell, not region directly).
5. Integrate into `aif_sweep.py` with a flag `use_multiscale_latent: bool = False`.
6. Expected ~90–180 held-out steps per traveller on 6-traveller fixture; this is a small cohort for learning 2-level structure.

**Expected Effect:** 50–150 bps if latent structure is genuinely multi-scale (unlikely on a 6×6 or 9×9 grid); more likely a null.

**Kill-Gate:** If multiscale loss exceeds flat-cell AIF loss by >20 bps, the structure is not present or the state space is too large to learn with ~540 holdout steps.

---

## LEVER 7: Mixing Long-Gap Reporting Feature (Direct Incorporation)
**File:** `src/active_fractal_rabbit/aif_sweep.py`, `default_sweep_grid()` (lines 411–422) and `score_pomdp_on_locked_split()` (lines 51–57, modalities parameter)

**Mechanism:**  
The report-burst category (short burst vs. routine vs. long gap) correlates with travel behavior: long gaps suggest the traveller is in a holding pattern or unreachable, which is informative about cell revisit likelihood. The current model includes `report_burst` as a modality in emission but does NOT use it in the transition prior. Add a separate factor: P(next_cell | current_cell, report_burst) learned jointly. This is a factorization variant of Lever 1 (emission interactions), but targeted.

**Implementation Strategy:**
1. Extend `build_pomdp()` to support an optional `report_modulated_transition: bool = False` parameter.
2. If True, learn a separate B matrix for each report-burst category: B_routine, B_burst, B_gap (instead of a single B).
3. At prediction time, use the current report-burst observation to index into B: `transition = B[:, :, control, current_burst_category]` (4D tensor).
4. In `aif_sweep.py`, test `report_modulated_transition=True` alongside the standard single-B models.
5. Requires careful normalization: ensure each B[:, :, control, burst] is column-stochastic.

**Expected Effect:** 20–60 bps if report-burst is predictive of transitions; 0 if independent.

**Kill-Gate:** If report-modulated B increases loss >5 bps, the feature is not predictive or the added complexity hurts generalization on this fixture.

---

## LEVER 8: Simulation-Based Calibration (SBC) + PSIS-LOO for Validation
**File:** Create `src/active_fractal_rabbit/sbc_validation.py`

**Mechanism:**  
Beyond empirical loss/ECE, validate the filter's probabilistic consistency using Simulation-Based Calibration (Talts et al. 2018) and PSIS leave-one-out cross-validation (Yao et al. 2018). This detects misspecification (e.g., the model overconfident on some cells). SBC involves: (1) sample random parameters from the generative model, (2) simulate synthetic data, (3) run the filter, (4) check that the posterior ranks of the true parameters are uniform [0, 1]. PSIS-LOO provides a more stable leave-one-out CV estimate.

**Implementation Strategy:**
1. Write `run_sbc_diagnostic()`: generate K synthetic sequences from the trained B matrix (fixed), run the filter on each, collect rank statistics.
2. Write `run_psis_loo()`: compute Pareto-smoothed importance sampling LOO for the locked-split model (alternative to standard held-out loss).
3. Run as a POST-HOCVALIDATION on the best config from the sweep, not as a sweep dimension itself.
4. Report: histogram of ranks (should be uniform if well-calibrated), LOO loss (should match held-out loss within ~10%), and any failure diagnostics.

**Expected Effect:** Identifies subtle miscalibration; not directly a loss reduction, but diagnostic (prevents a configuration that looks good on loss but is overconfident).

**Kill-Gate:** If SBC ranks are heavily skewed or PSIS diagnostics flag severe miscalibration, the best config has hidden overfitting; revert to a simpler variant.

---

## LEVER 9: Traveller-Disjoint Held-Out Bootstrap (Honest CIs)
**File:** `src/active_fractal_rabbit/sota_benchmark.py`, `_bootstrap_loss_ci()` (lines referenced in aif_sweep.py, line 195)

**Mechanism:**  
The current bootstrap treats holdout steps as i.i.d. but they are clustered within travellers (correlated). Implement a traveller-clustered bootstrap: resample entire traveller holdout sets, not individual steps. This gives a more honest (wider) CI.

**Implementation Strategy:**
1. In `aif_sweep.py`, add `bootstrap_mode: str = "step"` parameter to `score_pomdp_on_locked_split()`.
2. Implement:
   - `"step"`: current (overstates precision)
   - `"traveller_clustered"`: resample complete traveller sets from the `step_travellers` list (line 178)
3. For 6 travellers with ~2–3 holdout per seed, traveller-clustered bootstrap resamples ~2–3 units, which is weak but honest. Report the CI width increase as evidence that step-level CIs are indeed overstated.
4. Use traveller-clustered by default in final sweeps.

**Expected Effect:** CIs widen by 30–80% (no loss change, but honest uncertainty quantification).

**Kill-Gate:** If traveller-clustered CI is so wide that it includes baseline uncertainty, the distinct-holdout count is too small to claim a win even if point estimate improves.

---

## Prioritization & Execution Order (6-Traveller / 480-Report Fixture)

### Tier 0 (Must-Run, Cheap Diagnostics)
1. **Lever 9 (Traveller-Clustered Bootstrap)**: Add 5 lines, reveals if point estimates are real. RUN FIRST on current best config.
2. **Lever 4 (Fixed γ=1.5 sweep)**: Already known to help; confirm γ∈{1.0, 1.2, 1.5} on new fixture (cheap, ~3 configs).

### Tier 1 (High-Confidence Levers)
3. **Lever 1 (Pairwise Emission)**: 30 lines, moderate risk, captures emission structure. RUN EARLY.
4. **Lever 5 (Visited-Cell Frequency)**: 10 lines, low risk, models revisit heterogeneity. RUN EARLY.
5. **Lever 2 (EMA Online Adapt)**: 20 lines, medium risk, allows drift. RUN EARLY-MID.

### Tier 2 (Moderate-Risk, Higher-Reward)
6. **Lever 3 (Hierarchical Transitions)**: 40 lines, medium risk (overfitting on small fixture). RUN MID, TEST BOTH MODES.
7. **Lever 7 (Report-Modulated Transitions)**: 25 lines, medium risk, requires 4D tensor. RUN MID-LATE.
8. **Lever 4 (Adaptive γ)**: 30 lines, medium risk (scheduling adds complexity). RUN MID-LATE.

### Tier 3 (Exploratory/High-Risk)
9. **Lever 6 (Multiscale Latent)**: 100+ lines, high risk, new implementation. RUN ONLY IF Tier 1–2 yield <50 bps.
10. **Lever 8 (SBC/PSIS)**: Post-hoc validation, not a direct lever. RUN ON FINAL CONFIG.

---

## Expected Progression on 6-Traveller Fixture

- **Baseline (current):** ~+2.237 nats (vs oracle).
- **After Tier 0 + γ=1.5:** ~+2.05 nats (120 bps).
- **After Tier 1 (Levers 1+5+2):** ~+1.90 nats (150 bps cumulative).
- **After Tier 2 (Levers 3+7):** ~+1.75 nats (target: 180–250 bps cumulative).
- **After Tier 3 (Lever 6, if needed):** ~+1.60 nats (potential 250–380 bps; more likely a null).
- **Final checkpoint:** Traveller-clustered CIs widen; if still CI-separated from fair-frontier leaders, declare a win. If CI overlaps, report honestly as a regime-locked improvement, not a portable win.

---

## Kill-Gates & Honest Nulls

Each lever has a per-step kill-gate. Additionally, the project-level gate:
- **Fair-frontier robustness gate (Tier 1 complete):** AIF must beat the best fair non-oracle baseline (currently soft_plus_return_boost) with a traveller-clustered CI clearly separated. If point estimate improves but CI widens enough to overlap, the result is regime-bound and not portable.
- **No oracle-beating gate:** Acknowledged up front that none of these will beat the oracle-fed baseline (true previous cell). The win is fair-frontier, not SOTA.
- **Manuscript boundary:** All findings are synthetic-only; cross-vendor audit constrains promotion rather than confirming operational or real-mobility validity. Any claim that leaks into "operational" or "real-mobility" is rejected.

---

## File-Level Concrete Actions

1. **`src/active_fractal_rabbit/pomdp.py`**
   - Add `emission_interaction_mode`, `transition_hierarchy_mode` parameters to `build_pomdp()`.
   - Extend `_emission_matrix()` to optionally build pairwise models.
   - Extend `_transition_tensor()` to support hierarchical backoff.

2. **`src/active_fractal_rabbit/aif_sweep.py`**
   - Add parameters: `online_adapt_mode`, `online_decay`, `online_per_traveller`, `precision_schedule`, `precision_base`, `precision_peak`, `precision_entropy_scale`, `return_boost_mode`, `bootstrap_mode`, `report_modulated_transition`.
   - Expand `default_sweep_grid()` with all Tier 1 & 2 combinations.
   - Implement `_sharpen_scheduled()` for entropy-dependent precision.
   - Implement traveller-clustered bootstrap.

3. **`src/active_fractal_rabbit/multiscale_latent.py` (NEW)**
   - `build_multiscale_pomdp()`, `score_multiscale_on_locked_split()`, integration into sweep.

4. **`src/active_fractal_rabbit/sbc_validation.py` (NEW)**
   - `run_sbc_diagnostic()`, `run_psis_loo()`, integration into post-hoc validation.

5. **Test files / Config YAML**
   - Extend fixture's config to specify new parameters in model section.
   - Add integration tests for each new parameter variant.

---

## Success Metrics

1. **Quantitative (Per-Fixture):**
   - Reduce fair-oracle gap by ≥100 bps (best case ≥200 bps) on 6-traveller/480-report fixture.
   - Maintain or improve calibration (ECE).
   - CI narrowness improves or stays honest (traveller-clustered bootstrap).

2. **Honest Null Handling:**
   - Any lever that increases loss >kill-gate threshold is logged as a null, not discarded silently.
   - Final verdict uses traveller-clustered CI, not step-level.

3. **Reproducibility:**
   - Every config is locked-split scored; no oracle peeking.
   - Findings survive re-runs on the upgraded 6-traveller fixture.
   - Manuscript wording accurately reflects regime-bound, synthetic-only, provisional status.



## Top risks
- Overfitting on the small 6-traveller fixture (each traveller ~80 holdout steps): Tier 2 & 3 levers may learn spurious patterns. KILL-GATE aggressively; use traveller-clustered bootstrap as the true test, not step-level loss.
- Fixture-specific tuning (γ=1.5 already fixture-selected): Added structure (hierarchical, pairwise emissions) may be overfitted. Require all new levers to improve on HELD-OUT traveller compositions, not just the primary split.
- Computational complexity creep: Multiscale latent (Lever 6) and per-traveller adaptation (Lever 2 variant) may require algorithmic refactoring or JAX/PyMDP integration. Budget 2–3x the estimated implementation time.
- Interaction between levers: Combining Lever 1 (pairwise emissions) + Lever 7 (report-modulated transitions) may introduce collinearity or numerical instability. Test each alone first; combine only if both win independently.
- Traveller-clustered bootstrap reveals no separation: If the new CIs are so wide that even a 200 bps point gain overlaps the baseline, the fixture is too small to validate the claim. Honest outcome: directional improvement, not a statistical win.
- Emission conditional-independence ablation not run: The peer review explicitly requests this (point 6). If Lever 1 (pairwise) does not improve and a dedicated joint-emission ablation is not run, the manuscript risks the claim that interactions are unimportant without evidence.
- Cross-vendor audit is required before promotion: same-vendor RedTeam shares blind spots, and the later Forge-style audit record shows that a cross-vendor failure constrains the push rather than upgrading it.
- Long-gap reporting feature (Lever 7) lacks validation: If report-burst modulation does not improve loss but the team believes it should, missing diagnostics (e.g., mutual information of report-burst vs. cell transitions) prevent root-cause analysis.

## Acceptance criteria
- MUST: All 9 levers prototype and tested (at least 1 config per lever) on the 6-traveller/480-report fixture. Nulls (loss increase >kill-gate) are documented honestly, not suppressed.
- MUST: Best discovered config (post-Tier 1 & 2) reduces fair-oracle gap by ≥50 bps (best case 100+ bps) with a traveller-clustered bootstrap CI that does not overlap the current Markov baseline's CI (if achievable on this fixture size).
- MUST: Traveller-clustered bootstrap implemented and used for final CI reporting. Step-level bootstrap CIs marked as overstated and not used for claims.
- MUST: Pairwise emission model (Lever 1) and learned online adaptation (Lever 2) tested; if both fail kill-gates, the 'soft Bayesian marginalization is the dominant gain' narrative holds but structural improvements are null.
- MUST: Multiscale latent (Lever 6) either wins decisively (>100 bps) or is rejected as a null with honest reasoning (state space too large, too few holdout steps). Not left ambiguous.
- MUST: Simulation-Based Calibration (SBC) or PSIS-LOO run on the final best config to detect hidden miscalibration. If SBC rank histogram is skewed, the config is not promoted.
- MUST: Manuscript boundary statement remains: 'synthetic-only, fair-frontier comparison, not oracle-beating, not operational, and constrained by cross-vendor audit before any stronger promotion'.
- MUST: No oracle peeking in any variant. Lock is enforced at the data split and transaction-count level.
- MUST: If cumulative gains across Tiers 1–2 are <50 bps despite clean implementation, the plan is honest-declared as an interim result and Tier 3 (multiscale) is deferred to a future revision, not forced.
- MUST: All new parameters (precision_schedule, online_decay, etc.) are gridded, swept, and ranked by loss. Best config is data-selected, not manually chosen.
