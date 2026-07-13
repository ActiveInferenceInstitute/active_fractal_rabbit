# AIF Generative-Model Designs (FirstPrinciples + BeCreative /workflows)

> **HISTORICAL PLANNING RECORD — retained for provenance; superseded by current artifacts.**
> This is a frozen design-exploration snapshot from the revision program. Quantitative values
> quoted below are snapshots of earlier runs and are superseded by the current `output/runs/`
> artifacts and the rendered manuscript; internal cross-references (e.g. numbered "claim" ledger
> ids) refer to a ledger revision that no longer uses those numbers. Some designs were
> deliberately not executed. Do not cite numbers or ledger ids from this file.



## Family: Active Inference Filter (Bayesian Belief Tracking)

**Test bed:** Fine-grid high-mobility synthetic regime: 32 travellers on 8x8 spatial grid (64 cells) with fractal_dimension=1.33; must pass mobility gate (mean distinct cells ≥6, entropy ≥1.5 nats, Gini ≥0.5 per traveller; confinement_ratio <0.4). Parallel validation: external fine-grid rerun (re-discretize fractalrabbit_externalXY.csv at 8x8 resolution, same holdout split).

**Confound controls:** dominant-cell baseline (pure spatial confinement pure confinement); first-order pooled-population Markov (data density sufficiency null); oracle-previous-cell baseline (asymptotic ceiling; asymmetry bound); return-boosted soft Markov (strong fair non-recursive competitor)

**Kill-gates:** mobility_gate.pass (mean_distinct_cells ≥6, entropy ≥1.5 nats, Gini ≥0.5) — if fail, confound-control verdict withheld; aif_beats_soft_at_fixed_features — must be true to claim recursive gain is not Markov misspecification; return_boosted_soft_markov_ci_separation — AIF must beat return-boosted soft Markov CI-separated on locked split to claim fair advantage; external_aif_vs_soft_margin_nonzero (external simulator lane required for persistence claim) — synthetic-only advantage does not unlock operational or empirical claims; confinement_ratio <0.5 — high confinement (top-1 cell >50% observations) invalidates structure-learning claims

# FirstPrinciples Deconstruction: FractalRabbit Generative Structure and Confound Control

## 1. IRREDUCIBLE GENERATIVE STRUCTURE (What Markov Cannot Represent)

### A. Three Tiers of Confound Risk
The FractalRabbit generator encodes a three-tier waypoint process:
1. **AgoraphobicPoints (spatial confinement)**: Each of ~8 travellers anchors to a home point with probabilistic returns; at a coarse 4×4 discretization (16 cells), the simulator-backed external lane shows **5/8 travellers occupy ≤3 cells**, and traveller 0 spends **~99% of its time in a single cell**.
2. **RETRO-PREFERENTIAL (long-memory return)**: The middle tier mixes exploration (empirical continuation P(s_t | s_{t-1})) with preferential return to visited cells weighted by visitation frequency^α, forming a Pólya-urn return law.
3. **SPORADIC reporting**: Bursty, non-uniform observation gaps introduce missing-not-at-random (MNAR) structure.

### B. What First-Order Markov Cannot Capture (Stated as Structural Facts)
- **Confinement asymmetry**: A first-order Markov learned on a confined traveller's trajectory will reflect that confinement as learned transition probabilities, but it assigns NO causal role to the per-traveller spatial anchor. Any prediction advantage is purely a **density confound**—not a proof of long-memory inference.
- **Frequency-reinforced return**: A binary visited-cell boost (uniform over visited) is first-order Markov misspecified against the retro-preferential structure; frequency-weighted return (visit^α) is genuinely non-Markovian. The measured advantage is **+0.4 to +0.9 nats** (claim 96 in ledger) — but only on synthetic data with few travellers; at scale, the population transition model absorbs the return structure.
- **Multiscale spatial structure**: FractalRabbit's Brownian-motion-with-preferential-return path traces a fractal curve (controllable dimension 1.1–1.9); a coarse grid collapses multiscale clustering into apparent confinement, while a fine grid would expose the structure but at the cost of data sparsity.

### C. The Core Confound (Why Prior Models Won Falsely)
**SPATIAL CONFINEMENT IS THE DOMINANT EFFECT.** The prior finding (claim 96) that soft Bayesian marginalization beats a hard-argmax Markov by **~0.25 nats** reduces to:
- At n_travellers = 4 on a 4×4 grid, per-cell occupancy density is **~0.25 visits/cell**. The true previous cell is known; any predictor using that cell (Markov) has nearly enough information.
- A soft belief (marginalizing P(cell|obs) through the transition) appears to win because it **naturally tracks the confined state** via the emission posterior, while a hard point estimate sometimes picks the wrong cell.
- **This is not active inference advantage; it is confinement-aided soft inference.** It vanishes at scale (n=32 travellers, same grid) because the population transition model becomes sufficient—per-cell density rises, and the hard baseline catches up (claim 96, fair_power_regime_map).

---

## 2. CLASSIFICATION: LIKELY-REAL-SIGNAL vs LIKELY-CONFOUND

| Structure | Evidence Type | Real Signal or Confound? | Kill-Gate Proof |
|-----------|---------------|--------------------------|-----------------|
| **Frequency-weighted return (alpha=1)** | +0.4–0.9 nats vs binary return, 8 seeds, synthetic | LIKELY-REAL, structure-matched | Beats binary return UNIVERSALLY; reverses at scale (n=32) ⟹ density effect. |
| **Soft marginalization vs hard argmax** | ~0.25 nats, confined regime | LIKELY-CONFOUND (soft inference + confinement) | Vanishes at scale on same grid; identity-emission collapse proves handicap (claim 96). |
| **Recursive belief temporal accumulation** | ~0.05 nats vs soft, only under noise | LIKELY-REAL but tiny, observation-noise driven | Negative at clean emission; no independent external evidence. |
| **Visitation-frequency^alpha return** | Synthetic-only, not tested on the external simulator lane | UNVALIDATED | Return structure is NOT confirmed beyond the external simulator trace, which is sparse. |
| **Per-traveller vs population transition model** | Population pool N=4→32 → advantage erodes | CONFOUND: data density / asymptotic sufficiency | At N=32, per-cell occupancy sufficient; individual model irrelevant. |
| **MNAR reporting gaps** | Synthetic burst structure + simulator-backed external gaps | UNKNOWN | External simulator trace too sparse (8 travellers, 5700 points ≈ 712 pts/traveller) for honest test. |
| **Multiscale fractal structure** | Coarse 4×4 grid; fractal_dimension=1.33 sweep | UNEXAMINED at grid level | Finer grids (6×6, 8×8) would test; none implemented. |

---

## 3. MANDATORY CONFOUND-CONTROL BATTERY

Every model must be scored against **all four of these baselines** on the SAME holdout split:

### 1. **Dominant-Cell Baseline (Pure Confinement)**
```
pred_t = argmax over all visited cells of (visits[cell])
         normalized by total visits
```
- **Why**: Proves a soft/recursive model is not winning purely because "the agent learns the cage."
- **Kill gate**: If any model loses to dominant-cell on a multi-cell holdout set, it is genuinely defeated by confinement and cannot claim structure learning.
- **Measured on external**: dominant-cell loss would be very low for traveller 0 (~0 nats, ~1 bit/step) since it revisits the same cell constantly.

### 2. **First-Order Pooled-Population Markov (Data Density Null)**
```
P(s_t | s_{t-1}) learned on ALL train travellers (shared transition),
scored prequentially on holdout travellers.
```
- **Why**: Tests whether the pool-sufficient regime is reached. If population Markov beats individual/recursive on holdout, density is sufficient.
- **Kill gate**: A recursive filter claiming "my individual history is more predictive" must lose to pooled Markov on the SAME held-out steps AND the same train/test split.
- **Measured on external**: With 8 travellers and 5700 points, per-cell density is ~22 points per cell (assuming uniform). Pooled transition learns fast; individual advantage would be small.

### 3. **Oracle-Previous-Cell Baseline (Benchmark Ceiling)**
```
P(s_t | s_{t-1}=TRUE) — the agent is handed the true previous cell.
```
- **Why**: This is a deliberately stronger-information benchmark baseline because it is handed the true previous cell. If the claimed model is close, it means the emission identification problem is hard and inference gains are real within this task contract. If there is a large gap, the model is leaving information on the table.
- **Kill gate**: Mandatory positive control. If a model loses to oracle by <0.05 nats, inference matters little; if the gap is >0.5 nats, inference is a secondary effect.
- **Measured on external**: Expected gap ~0.3–0.4 nats given noisy location observations.

### 4. **Return-Boosted Soft Markov (Strong Fair Competitor)**
```
P(s_t | marginal P(s_t | obs_t)), plus per-traveller visited-cell frequency weighting,
learned and scored on the locked split.
```
- **Why**: Claim 96 proves this beats the recursive filter when both carry the frequency prior. It is the strongest non-recursive fair baseline.
- **Kill gate**: A recursive model claiming advantage must beat this return-boosted soft Markov CI-separated on the LOCKED split, not overall.
- **Measured on external**: Return-boosted soft Markov loss ~0.35–0.4 nats (from fixture data); an AIF model must get <0.35 nats to claim a win.

---

## 4. NON-CONFINED TEST BED PROTOCOL

### Problem: The Default Confined Regime Invalidates Claims
- **Fixture (4×4 grid, n=4 travellers)**: Each traveller ~4 obs per cell; heavily confined. Any soft model appears to win because confinement aids posterior tracking.
- **External (4×4 grid, n=8 travellers)**: Traveller 0 is ~99% in one cell. Markov-on-confinement dominates.
- **Solution**: Synthetic multi-cell regime or finer grid + many travellers so each is inherently mobile.

### Prescribed Non-Confined Test Bed (Two Parallel Paths)

**Path A: Synthetic Fine-Grid High-Mobility Regime (Recommended)**
- Generate synthetic waypoints with n=32 travellers, **8×8 spatial grid** (64 cells), fractal_dimension=1.33.
- Spatial bounds: same 200 km box, but **finer discretization** forces each traveller to traverse ~6–10 distinct cells.
- **Mobility check (must pass before any model claim)**:
  - Mean distinct cells per traveller: **≥ 6** (not ≤3)
  - Entropy of cell visit distribution per traveller: **≥ 1.5 nats** (not concentrated in one cell)
  - Gini coefficient of cell occupancy: **≥ 0.5** (not uniform, not dominated)
  - **If these fail, discard run and regenerate with finer grid.**
- Run power_analysis.fair_power_regime_map() on this grid with n_traveller_counts=(8, 16, 32).
- **Expected pattern**: If AIF has a genuine structural advantage, the AIF-vs-soft-boost margin should remain stable or grow even at n=32 on a fine grid (the density effect is attenuated). If it reverses to favor soft-boost, the advantage was density-driven confinement.

**Path B: External Fine-Grid Rerun (If Generator Available)**
- Re-run FractalRabbit external lane with **8×8 fixed_bounds spatial discretization** (instead of 4×4).
- Discretize same external_XY.csv waypoints at finer resolution.
- Run locked-split fair inference & AIF sweep on fine-grid external.
- **Expected**: If external spatial structure is multiscale-fractal (e.g., traveller 0 actually visits 4–6 distinct refined cells), confinement confound is reduced and any remaining AIF advantage is more credible.

### Mobility Verification Checklist (Apply to Any Test Bed)
```
mobility_ok = (
    mean_distinct_cells_per_traveller >= 6
    AND median_entropy_visit_dist >= 1.5
    AND min_traveller_gini >= 0.4
)
if not mobility_ok:
    REJECT model claim; invalid test bed.
else:
    proceed_to_model_scoring()
```

---

## 5. MODEL FAMILY RANKING BY EXPECTED GENUINE VALUE

| Rank | Family | Expected Genuine Win (Non-Confound) | Mechanism | Test Bed | Risk |
|------|--------|-------------------------------------|-----------|----------|------|
| **1** | **Frequency-Weighted Preferential Return (Markov + Visit^α)** | ~+0.4 nats (multi-seed, synthetic) | Structure-matched generative form (matches retro-preferential tier) | Fine-grid multi-traveller | **MEDIUM**: Reverses at scale; external simulator lane not tested. Will likely fail a mobile test bed unless α tuning per traveller. |
| **2** | **Soft Marginalization (Bayesian Belief + Return Boost)** | ~+0.2 nats (confinement-aided) | Soft inference tracks confined state; returns add per-traveller structure. | Confined (default 4×4) | **HIGH**: Not genuine inference gain; density confound. Expect to LOSE on fine-grid mobile test bed. |
| **3** | **Recursive Active-Inference Filter (Soft Belief + Temporal Accumulation)** | ~+0.05 nats (observation-noise driven only) | Temporal belief refinement under noisy emission; orthogonal to return structure. | Confined + high-noise regime | **HIGH**: Negligible magnitude; negative at clean emission; no external simulator-lane validation. Expect to disappear on a mobile test bed. |
| **4** | **MNAR Reporting-Process Inference** | UNKNOWN (unvalidated on the external simulator lane) | Gap structure + burst timing encode cell uncertainty; modeling MNAR helps. | Sparse external simulator trace | **CRITICAL**: External simulator output is too sparse to estimate; synthetic burst structure is not confirmed beyond the fixture. Conjecture only. |
| **5** | **Multiscale Fractal Spatial Inference** | UNKNOWN (untested) | Fractal dimension controls path autocorrelation; multiscale grid could capture. | Fine grids (8×8, 16×16) | **CRITICAL**: No existing fine-grid results; would require new config + figures. Pure prototype tier. |

---

## 6. DESIGN DISCIPLINE: SPECIFICATION BEFORE IMPLEMENTATION

### Non-Negotiable Machinery

1. **Confinement Quantifier** (add to fair_inference.py)
   ```python
   def spatial_confinement_ratio(sequence: ObservationSequence) -> dict[str, float]:
       """Per-traveller fraction of observations in top-1 cell; population mean.
       If > 0.5, declare high confinement and mark test-bed regime as INVALID
       for any active-inference structural claim.
       """
   ```
   - **Deployed**: Run before any model is scored.
   - **Trigger**: If confinement_ratio > 0.6, append caveat: "High confinement regime; confound-control battery required before claim."

2. **Fair Baseline Battery** (expand fair_inference.py)
   ```python
   def fair_baseline_ladder(sequence, modalities=("location",), online_adapt=True, return_boost=1.0):
       """Score the four mandatory baselines on the locked split:
       - dominant_cell (per-traveller visit mode)
       - first_order_pooled_markov (population transition)
       - oracle_previous_cell (hand the true previous cell)
       - return_boosted_soft_markov (soft + frequency prior)
       Return dict with all four + measured AIF, CI-separated verdict.
       """
   ```

3. **Mobility Gate** (new module: mobility.py)
   ```python
   def check_mobility_gate(sequence: ObservationSequence) -> dict[str, object]:
       """Compute mean distinct cells, entropy, Gini; return pass/fail + diagnostics.
       If fail, append to result: "Test bed mobility insufficient; confound-control
       verdict withheld."
       """
   ```

4. **Prototype-Before-Claim Contract** (in evaluation.py)
   ```python
   CLAIM_GATES = {
       "frequency_return_structure": {
           "prototype_artifact": "output/runs/return_structure_analysis.json",
           "gates": [
               "move_step_entropy_reduction >= 0.15",
               "return_skill_vs_persistence >= 0.99",
           ],
           "external_required": False,  # synthetic-only claim is OK if gates clear
       },
       "soft_marginalization_inference_gain": {
           "prototype_artifact": "output/runs/fair_inference.json",
           "gates": [
               "mobility_gate.pass == True",
               "aif_beats_soft_at_fixed_features == False",  # if true, gain is Markov-misspec, not inference
               "confinement_ratio < 0.5",
           ],
           "external_required": True,  # real claim requires external validation
       },
       "aif_recursive_advantage": {
           "prototype_artifact": "output/runs/fair_inference.json",
           "gates": [
               "mobility_gate.pass == True",
               "recursion_gain > soft_gain",  # recursive > soft, not the other way
               "external_aif_vs_soft_margin > 0.01",
           ],
           "external_required": True,
       },
   }
   ```

---

## 7. HONEST EXPECTATIONS & PREDICTED OUTCOMES

### Most Likely Scenario (>70% confidence)

1. **Fine-grid high-mobility test bed is run** (n=32, 8×8 spatial bins).
   - **Confinement ratio drops** from 0.6+ (4×4) to ~0.25 (8×8).
   - **Mobility checks pass**: mean cells/traveller ~8, entropy ~1.8 nats, Gini ~0.6.

2. **Confound-control battery results on fine-grid**:
   - **Dominant-cell baseline**: ~0.35–0.4 nats (still strong, but beatable).
   - **First-order pooled Markov**: ~0.33–0.35 nats (population transition becomes *very* strong with n=32).
   - **Oracle previous-cell**: ~0.25–0.28 nats (emission identification still has cost).
   - **Return-boosted soft Markov**: ~0.32–0.34 nats (soft inference catches up).
   - **AIF + return boost**: ??? (Will likely be ~0.32–0.34 nats, matching soft + boost.)

3. **Verdict on fine-grid**: AIF does NOT beat return-boosted soft Markov CI-separated. The ~0.25-nat advantage on 4×4 confined vanishes. **Claim is retracted to: "Soft inference (not recursion) is the inference gain on sparse waypoints; recursive temporal accumulation adds negligible value when the previous cell is well-inferred."**

4. **Frequency-weighted return claim persists** because it is structure-matched and measured orthogonally (alpha=1 vs alpha=0 binary return), but with caveat: "Advantage reverses at scale (n=32); applicable only in low-population regimes."

---

## 8. BUILD PLAN & DELIVERABLES

### File-Level Modules (New + Modified)

**New Modules:**
- `src/active_fractal_rabbit/mobility.py` — Mobility check (mean cells, entropy, Gini).
- `src/active_fractal_rabbit/confound_controls.py` — Dominant-cell, pooled Markov, oracle baselines.
- `src/active_fractal_rabbit/claim_gates.py` — Machine-checkable promotion gates for each claim family.

**Modified Modules:**
- `src/active_fractal_rabbit/fair_inference.py` — Add confinement_ratio, expand fair_baseline_ladder to include all four baselines.
- `src/active_fractal_rabbit/power_analysis.py` — Add fine-grid test bed option (8×8, 16×16 spatial_bins).
- `src/active_fractal_rabbit/evaluation.py` — Integrate claim_gates, block promotions unless gates clear.
- `configs/` — Add `fine_grid_high_mobility.yaml` (4×4 → 8×8, n_travellers → 32).

**Orchestration:**
- `scripts/03_run_sweep.py` — Add option `--test-bed {confined_default, fine_grid_mobile}`.
- `scripts/07_run_full_study.py` — Run power_analysis on both test beds, compare margins.

### Analysis Artifacts

1. **Confinement Audit** (`output/runs/confinement_audit.json`)
   - Per-traveller top-1-cell fraction.
   - Population mean confinement_ratio.
   - Regime classification (confined/mobile).

2. **Confound-Control Matrix** (`output/runs/confound_control_battery.json`)
   - All four baselines on locked split.
   - Per-baseline CI.
   - AIF loss vs each.
   - Verdict: "AIF beats [baseline] CI-separated: YES/NO."

3. **Mobility Gate Audit** (`output/runs/mobility_gate_audit.json`)
   - Mean distinct cells per traveller.
   - Entropy of visit distribution.
   - Gini coefficient.
   - Pass/fail: mobility_ok boolean.

4. **Multi-Test-Bed Power Analysis** (`output/runs/power_analysis_multibed.json`)
   - 4×4 confined (default).
   - 8×8 mobile (fine-grid).
   - Same comparisons: AIF vs soft+boost margin.
   - Gradient tables showing vanishing at scale.

5. **Claim Ledger Update** (`output/runs/claim_gates_status.json`)
   - Every claim family with gate status (clear/blocked, reason).
   - Provenance: which artifact gates each claim.

---

## 9. HONEST BOUNDARY STATEMENT

**Synthetic-only; fair-comparison advantage, not SOTA or operational.** The FractalRabbit default 4×4 fixture is a **confined, coarse-resolution regime** where any soft-inference predictor appears to win against naive baselines. A **non-confined, fine-grid test bed** is mandatory to distinguish:
- Genuine structure learning (retains advantage under mobility).
- Confound-aided soft inference (vanishes when confinement is reduced).
- Data-density sufficiency (population Markov becomes best).

The honest research claim is: **"On synthetic waypoints with controlled structure, soft Bayesian marginalization of the previous cell is a meaningful inference gain only in confined regimes; under higher spatial mobility the gain vanishes. Frequency-weighted preferential return is a structure-matched improvement over binary return, but only in low-population limits."** This is not a SOTA claim; it is a regime-characterization of inference mechanisms.

---

## 10. APPENDIX: FirstPrinciples Deconstruction Summary

### Constituent Parts (Irreducible Truth)
1. **Spatial confinement** (per-traveller anchor + probabilistic returns).
2. **Emission noise** (observed location ≠ true cell; requires inference).
3. **Transition dynamics** (Markov or preferential-return mixture).
4. **Data density** (observations per cell; determines sufficiency of pooled transition).
5. **Observation gaps** (MNAR reporting; creates state uncertainty).

### Hard Constraints (Immutable)
- Prequential evaluation is mandatory (no looking ahead).
- Holdout travellers must be disjoint from training (no data leak).
- Oracle-fed baseline is the strong benchmark ceiling under this task contract.
- First-order Markov is a strong baseline on confined data (density effect dominates).

### Soft Constraints (Can Be Relaxed)
- 4×4 grid resolution is a choice, not a law (finer grids possible, computational cost higher).
- 8 travellers is a fixture choice (power analysis can sweep count).
- Frequency-weighted return formula (alpha=1) is a tuning choice (alpha in [0, 2] can be swept).

### Unvalidated Assumptions (Must Be Tested)
- **Assumption**: Recursive temporal belief adds value beyond soft Bayesian marginalization.
  - **Reality check**: Negative under clean emission; positive only under noise. Likely confound: soft inference learns the emission, recursion is redundant.
  - **Test**: Run under low-noise (clean cell identification) vs high-noise (diffuse emission). If negative disappears at low noise, recursion is observation-noise driven, not a structural advantage.
- **Assumption**: Frequency-weighted return (visit^alpha) beats binary return on the external simulator lane.
  - **Reality check**: Tested on fixture (synthetic, +0.4–0.9 nats); NOT tested on the external simulator trace (too sparse to estimate alpha per traveller).
  - **Test**: The external simulator trace has 8 travellers × ~5700 points ≈ 712 points per traveller. Can estimate alpha if return structure is strong; early results suggest it is not (sparse simulator-backed output, return effect is statistical noise).

---

## 11. Recommended Next Action (One Decision Path)

**DO THIS FIRST (2–3 hours of compute):**
1. Run `power_analysis.fair_power_regime_map()` on a **fine-grid (8×8) high-mobility synthetic regime** (n_travellers=32, same fractal_dimension=1.33).
2. Measure mobility gate on result.
3. Run confound-control battery on fine-grid holdout.
4. Compare AIF vs soft+boost margin: confined (4×4, n=4) vs mobile (8×8, n=32).
5. **If margin vanishes on mobile**: Recursion is a confined-regime artifact, not a structural advantage. Claim is demoted to soft-inference-on-noisy-emission.
6. **If margin persists on mobile**: Proceed to external-lane fine-grid rerun (10+ hours, requires FractalRabbit generator).

This decision tree avoids months of wasted work on unpromising claims and focuses resources on the most honest direction.



## Family: MNAR-Reporting Joint Motion-and-Reporting Model Family

**Test bed:** Non-Confined Mobile Regime: 3x3 cell grid (9 cells), 4 travellers (fixture) or 16/32 travellers (power analysis). Mobility gate: mean distinct cells per traveller ≥ 5 (out of 9). Entropy of gap duration conditional on traveller > 0.5 nats (structure present). Use existing 48-row fixture for baseline (confirmed mobile: mean 7.2 cells/traveller, entropy 1.8 nats); synthetic power analysis at same spatial resolution with β_gap ∈ {0, 0.2, 0.5} and traveller counts {16, 32} to measure regime dependence.

**Confound controls:** Dominant-Cell Baseline (confinement oracle): single occupied spatial state; Brier loss on reporting events in degenerate single-cell regime; First-Order Pooled-Population Markov (data-density baseline): gap-blind Bernoulli reporter trained globally; no cell/traveller/phase conditioning; Oracle Reporting-Regime Posterior (asymmetry bound): perfect knowledge of true reporting state one-hot-encoded; clairvoyant reporting predictor ceiling; Fixed-Gap Baseline (spuriousness test): all inter-report intervals set to population median; tests whether gap signal is spurious correlate vs real information; Permuted-Gap Control (MNAR authenticity gate): shuffle inter-report intervals across travellers, breaking traveller-specific gap/motion link; tests for MNAR vs MAR artifact

**Kill-gates:** GATE-1 (Gap Coefficient Significance): Fit logit(p(r_t)) ~ β₀ + β_gap·gap + β_cell·cell_type on training split. Wald test: β_gap ≠ 0 at α=0.05 two-tailed on simulator-backed external-lane data. KILL if p > 0.05 or sign counterintuitive. Honest expectation: gap coefficient significant on fixture (β_gap ≈ ±0.1–0.3), may vanish on the external simulator lane (FractalRabbit reporting may not be MNAR).; GATE-2 (Brier Improvement Over Markov, Non-Vacuity): Held-out Brier Δ = Markov_brier - gap_aware_brier. Permutation test (1000 shuffles): Δ must exceed 95th percentile of null. KILL if Δ < null p=0.95. Honest expectation: Δ > 0 on fixture (strong β_gap → ~0.05–0.15 nats), weaker on external.; GATE-3 (Permuted-Gap MNAR Authenticity): Compare observed-gap advantage vs permuted-gap advantage (intervals shuffled globally). KILL if observed-gap advantage ≤ permuted-gap advantage (spurious/MAR, not MNAR). Honest expectation: observed-gap advantage > 50% larger than permuted on fixture; may be much smaller on external if reporting is random (MAR).

# MNAR-Reporting Model Family Design

## Generative Form & Theoretical Foundation

**Joint Motion-and-Reporting (MNAR) Model**: A latent-state-conditioned missing-not-at-random (MNAR) reporting process paired with spatial motion dynamics. The key innovation is modeling the inter-report GAP as jointly informative of both latent state and reporting behavior, rather than treating gaps as exogenous MCAR missingness.

**Generative Decomposition**:
```
p(x_t, r_t | x_{t-1}, obs_{t-1}, θ) = p(x_t | x_{t-1}, obs_{t-1}) · p(r_t | x_t, x_{t-1}, Δ_{t-1})
```

where:
- `x_t` = latent cell state at time t
- `r_t` ∈ {0,1} = reporting indicator (1 = location observed, 0 = missing)
- `Δ_{t-1}` = inter-report interval since the last observation
- **Key asymmetry**: reporting depends on BOTH latent state (cell-dependent dwell/travel likelihood) AND the hidden gap duration, making the missingness NOT MCAR.

The MNAR structure: longer gaps inform the predictor that *displacement likely occurred* (if moving), allowing the model to use gap duration as a proxy for unobserved motion. This is the signal a first-order Markov model, which treats gaps as random missingness, cannot exploit.

**Three-Tier Component Hierarchy** (orthogonal to existing modalities):

1. **Reporting-Markov Tier** (baseline): Bernoulli report probability conditioned on traveller phase + runlen (position-in-burst). Honest null; reproduces existing `heldout_reporting_process.py` finding that phase/runlen do not beat pooled baseline out-of-sample.

2. **Motion-Informed-Reporting Tier** (MNAR signal): Bernoulli report probability conditioned on:
   - Latent state posterior estimate `qs_t` (from the current observation modalities)
   - Previous cell type (dwell vs transit) inferred from speed history
   - Gap duration (inter-report interval)
   - Interaction: `p(r_t = 1 | gap_duration, speed_regime, cell_type)`
   
   Hypothesis: longer gaps in transit states → reporting less likely (motion continues unobserved). Longer gaps in dwell states → reporting more likely (stationary person more likely to report when returning from forced silence). This OPPOSITE direction for dwell vs transit is load-bearing: it means the model learns cell-type-specific reporting and uses gap as a discriminator.

3. **Full-POMDP-Joint Tier** (integrated inference): Embed reporting as a hidden factor in the factorial POMDP alongside movement_mode and cognitive_proxy_mode. Allow the agent to infer reporting state and use it to account for observation scarcity. The agent's policy can trade off between accepting sparse observations (low report probability) vs actions that increase report likelihood (e.g., stay in cell that encourages contact).

## What Signal Does MNAR Exploit That Markov Ignores?

A first-order Markov predictor treats each gap as a random event: `p(next-cell | last-cell, transition)`, with the gap duration ignored in prediction. The MNAR model conditions the *belief update* on gap duration *before* moving forward in time:

1. **Before observing the next report**, the model updates its belief that a large gap implies: 
   - Higher probability of displacement IF in a high-speed state (travel regimes continue)
   - Higher probability of stationary IF in a dwell state (person stayed put)
   
2. **This is an anticipatory belief adjustment** via the reporting model, not a new transition. It sharpens the prior before the next observation lands.

3. **Markov baseline cannot do this** because it does not model reporting; gap duration is exogenous. By design, the first-order predictor has a uniform prior over time between reports.

**Rubin Framework Classification**:
- **MAR** (Missing At Random): report probability `p(r_t | obs_{1:t}, x_{1:t}, known-parameters)` — the observation process depends on observed data and latent state, but given those, not on the missing data itself.
- **MNAR** (Missing Not At Random): report probability `p(r_t | obs_{1:t}, x_{1:t}, x_{t+1:T})` — the reporting mechanism depends on the *unobserved* future or on latent structure not in the observations. Here, gap duration is a signal of unobserved motion, making it MNAR.

## File-Level Build Plan

### New Module: `mnar_reporting.py`
- **Core class**: `MNARReportingModel(sequence, inference_config)` 
  - Input: `ObservationSequence` + discretized posterior estimates from reference lane + speed/regime history
  - Outputs: 
    - `reporting_mnar_brier` — Bernoulli prediction loss on held-out reporting events
    - `gap_duration_regressed_on_displacement` — linear regression: `Δ_{t+1} ~ gap_duration + latent_cell_type`, R²
    - `gap_as_anticipatory_prior_strength` — mutual information `I(Δ_t ; x_{t+1} | last_cell)`, Nats
    
- **Key methods**:
  - `_build_reporting_records(sequence, min_interval_hours)` → DataFrame of (traveller, gap_duration, cell_type, observed_next_reporting, speed_regime)
  - `_cell_type_from_speed_history(speed_categories, window=3)` → dwell/transit/fast classification per step
  - `_fit_reporting_models(train_records)` → fit 3 predictors: (a) pooled Bernoulli, (b) gap+cell-type logistic, (c) gap+speed+phase logistic
  - `_score_held_out(test_records, models)` → Brier loss per model on held-out reporting events
  - `_gap_anticipatory_simulation(posterior_sequence, reporting_model, gap_mask)` → filter that updates belief *given gap duration* before next observation, then score

### Integration Points
- **Pipeline stage**: `src/active_fractal_rabbit/pipeline.py` — add `run_mnar_analysis(sequence, config)` call after holdout/inference, before figures
- **Config knob**: `model.reporting_model: mnar | mar | null` — allow selecting reporting family
- **Figure integration**: `src/active_fractal_rabbit/figures.py` — new figure `reporting_gap_analysis`:
  - Panel A: Brier loss (pooled vs gap-aware reporting models), with 95% bootstrap CI
  - Panel B: Gap duration distribution by cell type (dwell/transit/fast), violin plot
  - Panel C: Mutual information `I(Δ_t ; Δ_{t+1})` and `I(Δ_t ; x_{t+1} | last_x_t)` — the MNAR signal magnitude
- **Artifact**: `mnar_reporting_analysis.json` with schema:
  ```json
  {
    "schema": "active-fractal-rabbit-mnar-reporting-v1",
    "reporting_null": { "name": "pooled_bernoulli", "mean_brier": X },
    "reporting_gap_mar": { "name": "gap_duration_aware", "mean_brier": Y, "gap_coeff": Z, "ci_low": ..., "ci_high": ... },
    "reporting_gap_cell_mnar": { "name": "gap_cell_type_interaction", "mean_brier": W, "dwell_coeff": ..., "transit_coeff": ..., ... },
    "anticipatory_skill": { "flat_blind_brier": X, "gap_anticipatory_brier": Y, "skill": Y-X, "ci": [...] },
    "gap_displacement_correlation": { "r_pearson": X, "p_value": Y, "interpretation": "... under cell type control ..." },
    "information_measures_nats": {
      "h_reporting_unconditional": X,
      "h_gap_marginal": Y,
      "mutual_info_gap_reporting": Z,
      "conditional_gap_future_displacement": W
    }
  }
  ```

### Test Module: `tests/test_mnar_reporting.py`
- **Determinism tests**: gap duration computed identically on re-runs (seed-locked)
- **Data shape tests**: `_build_reporting_records()` returns df with n_intervals rows, no NaN
- **Model fit tests**: logistic coefficients have expected signs (gap→higher reporting likelihood in dwell, opposite in transit)
- **Held-out gate tests**: gap-aware model beats pooled on >50% of bootstrap replicates (directional, not absolute)
- **Anticipatory filter test**: gap-duration-informed belief shift moves posterior mass away from last-cell in >80% of gap steps (proof of concept)
- **Control tests**: 
  - Permuted gap (shuffled inter-report intervals) destroys the advantage (MNAR signal)
  - Constant-gap baseline (all gaps set to median) reduces but doesn't eliminate the effect
- **Anti-vacuity**: gap-aware coefficient is statistically distinguishable from 0 at α=0.05 on simulator-backed external-lane data

## Confound Controls (REQUIRED, per discipline)

Every model must be scored against ALL of:

1. **Dominant-Cell Baseline** (confinement confound): Single occupied spatial cell (collapse all states). Brier loss on reporting events in this degenerate case. Reports this as a floor to ensure the model doesn't just leverage cell occupancy.

2. **First-Order Pooled-Population Markov** (data-density confound): Gap-blind Bernoulli reporter trained on all observed inter-report intervals globally, no cell/traveller/phase conditioning. This is the Markov baseline — it has zero knowledge of gap structure beyond raw frequency.

3. **Oracle Reporting-Regime Posterior** (asymmetry bound): The true reporting state one-hot-encoded (perfect knowledge of `r_t` at every step). This is what a clairvoyant predictor could achieve if reporting were fully observed. The MNAR model cannot beat this, and if it approaches it, the model has learned the *structure of reporting itself*, not just pattern-matched gaps.

4. **Fixed-Gap Baseline** (test for gap spuriousness): All inter-report intervals set to the population median. If reporting Brier stays nearly identical, the gap signal is a spurious correlate (reporting driven by other confounds). If Brier *worsens*, gaps carry real information.

5. **Permuted-Gap Control** (MNAR signal test): Shuffle inter-report intervals across travellers, breaking the link between an individual's motion and their own gap pattern. If the advantage disappears (Brier reverts to pooled), the signal comes from the traveller-specific gap/motion linkage (MNAR), not a global gap/reporting correlation (MAR).

## Test Bed (NON-CONFINED, Mobile Regime)

The fixture uses a 3x3 cell grid (9 cells) with 4 travellers over 12 timepoints. **Confinement check**: mean cells per traveller must be ≥ 5 (out of 9). If confined (mean << 5), the trajectory model is trivial and any reporting model wins by accident.

**Specification for MNAR study**:
- **Use the existing 48-row fixture** (already mobile: mean 7.2 cells per traveller, entropy 1.8 nats)
- **Synthetic data generation** (power/regime analysis):
  - Generate 16/32-traveller sequences at SAME spatial resolution (9-cell grid) so mobility remains > 5 cells/traveller
  - Vary reporting probability: `p(r_t) = exp(β₀ + β_gap · gap_duration) / (1 + ...)` with β_gap ∈ {0, 0.2, 0.5}
    - β_gap = 0: reporting independent of gap (MAR, null case)
    - β_gap = 0.2: weak gap effect (should be detectable)
    - β_gap = 0.5: strong gap effect (should be obvious)
  - Verify: gap duration ranges 0.05–5 days across all regimes, entropy(gap | traveller) > 0.5 nats (structure present)

**Mobility gate**: Assert that mean distinct cells per traveller in test bed ≥ 5. Reject the test bed if mobility is too confined.

## Honest Kill-Gates (Cheap, Pre-Build Probes)

1. **Gap Coefficient Significance** (MNAR formality gate):
   - Fit `logit(p(r_t)) ~ β₀ + β_gap · gap_duration + β_cell · cell_type` on training split
   - Wald test: `β_gap ≠ 0` at α=0.05 (two-tailed) on the *simulator-backed external-lane data*
   - **Kill gate**: If β_gap p-value > 0.05 OR β_gap sign is counterintuitive (negative for dwell), STOP — gap signal does not exist in the external simulator lane; mark as null
   - Honest expectation: gap coefficient will be significant on fixture (β_gap ≈ ±0.1 ~ ±0.3, depending on regime), but may vanish on simulator-backed sparse data if FractalRabbit's reporting is not MNAR

2. **Brier Improvement Over Markov** (Non-Vacuity Gate):
   - Fit gap-aware model and Markov baseline on same training split
   - Held-out Brier: `Δ = Markov_brier - gap_aware_brier`
   - Permutation test: shuffle reporting labels 1000 times, refit, compute null distribution of Δ
   - **Kill gate**: If Δ < 95th percentile of null distribution, the improvement is within chance; mark as null
   - Honest expectation: Δ > 0 on fixture (strong β_gap → ~0.05–0.15 nats improvement), weaker on the external simulator lane if simulator reporting is close to MAR

3. **Permuted-Gap Control Direction Check** (MNAR Authenticity Gate):
   - Compute Brier advantage with observed gaps and with permuted gaps (inter-report intervals shuffled globally)
   - **Kill gate**: If `advantage(observed gaps) ≤ advantage(permuted gaps)`, the signal is spurious (global, not traveller-specific); mark as MAR artifact, not MNAR
   - Honest expectation: observed-gap advantage > permuted-gap advantage by >50% on fixture; may be much smaller on the external simulator lane if simulator reporting is random (MAR)

## Honest Expectation & Framing

**Strong Positive (Unlikely)**:
- Gap duration + cell type jointly predict held-out reporting with Brier 2×–3× lower than pooled baseline
- Anticipatory filter (belief update given gap) beats flat-filter on gap-filling synthetic data by >0.2 nats
- Permuted-gap control collapses the advantage to <10% of the observed-gap improvement
- **Claim (if strong positive holds)**: "The FractalRabbit synthetic reporting mechanism encodes cell-specific MNAR structure; longer gaps in mobile states predict continued absence; models exploiting this joint structure outperform gap-blind Markov."

**Modest/Regime Conditional (Most Likely)**:
- Gap-aware Brier beats Markov on fixture (β_gap significant) but margin is <0.1 nats
- Permuted-gap control shows partial degradation (confound partly structural, partly spurious)
- Simulator-backed external lane shows smaller or zero gap effect (simulator reporting may be closer to MAR)
- **Claim**: "On the fixture, inter-report gaps carry regime-dependent information about reporting and motion; on the simulator-backed external lane, the signal is weak/absent, suggesting the fixture mechanism is more structured than the sparse simulator trace."

**Null (Expected Given Project History)**:
- Gap coefficient not significant (p > 0.05) even on fixture
- Permuted-gap control shows no degradation (gap signal is spurious)
- Anticipatory filter scores worse than flat filter (trying to use unobserved information)
- **Claim**: "Reporting events in FractalRabbit do not exhibit MNAR structure; inter-report gaps are exogenous (MCAR) relative to motion dynamics. First-order Markov models capture the conditional distribution correctly."

**Framing (Regardless of Outcome)**:
- Every result is labeled **synthetic software measurement on the pinned FractalRabbit simulator**, never empirical evidence
- The model never claims to recover real missingness mechanisms in human mobility (out-of-scope)
- The honest null is valuable: it clarifies what structure the synthetic generator *does not* encode and what assumptions Markov filtering can safely make
- The gap signal, if present, is a synthetic-generator pattern; human sparse-mobility reporting is outside this package and would require a separate empirical study with appropriate trace metadata

## Buildability & Integration

- **New source files**: `src/active_fractal_rabbit/mnar_reporting.py` (~400 lines), `tests/test_mnar_reporting.py` (~200 lines, 12–15 unit tests)
- **Modified source files**:
  - `src/active_fractal_rabbit/pipeline.py` — add `run_mnar_analysis()` call + config knob validation
  - `src/active_fractal_rabbit/figures.py` — add `_make_reporting_gap_analysis()` figure generator
  - `src/active_fractal_rabbit/config.py` — add `model.reporting_model` field (default "null")
- **Modified test files**: Add imports in `tests/test_pipeline.py` for the new config knob flip test
- **Modified docs/artifact files**:
  - `docs/config_schema.md` — document `model.reporting_model` field
  - `docs/claim_ledger.md` — +1 row for the MNAR reporting gate
  - `manuscript/` — new section with hydrated tokens for Brier improvements and MI measures
- **Output artifacts**: `output/runs/{default,external}/mnar_reporting_analysis.json` + `figures/reporting_gap_analysis.png`
- **Estimated effort**: ~4–6 hours to build, test, and integrate into the pipeline; ~2 hours for manuscript reporting.

---

## Prototype-Before-Build Quick Probe (Do This First)

Before building the full module, run a throwaway analysis on the existing fixture:

1. Load the fixture observations and reference-lane posterior (`qs`)
2. Compute inter-report intervals and cell-type classification from speed history
3. Fit a simple logistic: `logit(p(r_t)) ~ β₀ + β_gap · gap + β_cell · cell_type`
4. **Quick questions to answer**:
   - Is β_gap significantly different from 0? (z > 1.96)
   - Does the sign match the hypothesis (longer gaps → higher report prob in dwell, lower in transit)?
   - On held-out 25% of intervals, does the gap-aware model beat pooled Bernoulli?
   - **If answers are mostly no → mark as unlikely to clear the kill gates; DEFER or REFRAME**
   - **If answers are mixed → proceed with caution; expect a modest/regime result**
   - **If answers are yes → build with confidence; expect to report a positive finding (with all caveats)**




## Family: Multiscale-Hierarchical + Continuous-State Particle Inference

**Test bed:** Non-confined fine-grid test bed: 8x8 = 64 cells (vs. 4x4 = 16 in default), same waypoints/travellers/reports. Mobility check: mean_distinct_cells_per_traveller must increase >= 2x (e.g., 2.5 -> 5.5+). Entropy of cell occupancy must exceed 1.0; occupancy_ratio > 0.4.

**Confound controls:** dominant_cell_baseline: always predict most-visited cell (pure confinement floor); first_order_pooled_markov: population transition matrix, no per-traveller structure (data density null); oracle_persistence: previous true cell (asymptotic upper bound, models must not beat)

**Kill-gates:** Gate 1 (Confinement Unresolved): mean_distinct_cells_per_traveller does not improve >= 2x in fine grid -> stop claim; refactor regime; Gate 2 (State-Space Explosion): # observed fine-cell pairs > 512 AND sparsity < 0.1 -> increase backoff_threshold or coarsen; Gate 3 (Particle Degeneracy): ESS < n_particles/4 unrecovered for 3+ steps on >= 20% of stream -> increase n_particles or tighten emission scale; Gate 4 (Noise Floor): |loss(hierarchical) - loss(soft_boost)| < 0.01 nats (95% CI overlap) -> do not claim win; report comparable with trade-offs

# Multiscale/Continuous-State Model Family Design for FractalRabbit

## Generative Form: What Markov Cannot Capture

The current single-resolution discrete grid (4×4 = 16 states in default) suffers from a hard confinement trap: when individual travellers occupy ≤3 of 16 cells over their entire trajectory, any "sophisticated" model appears to win simply because each traveller's confined history is on-distribution. The population pooled model mixing disjoint territories appears weak only by contrast.

**What Markov chain (any order) cannot capture:**
1. **Fractal self-similarity across scales** — FractalRabbit generates trajectories with Hölder-continuous displacement at fractal dimension 1.33, meaning motion exhibits structure at multiple temporal/spatial scales. A single fixed grid discards this.
2. **Continuous-state long-memory return patterns** — the true process is continuous; discretization washes out revisit-to-old-places structure when revisits land in the *same* quantized cell (high false-negative revisit rate in coarse grids).
3. **Information geometry of sparse reporting** — observations arrive sporadically; the true path is continuous but only sampled. A learned hierarchical hierarchy allows the model to express "coarse confidence" before "fine disambiguation," matching how humans resolve ambiguity under sparse cues.

**What model family captures this:**
- **Coarse-to-fine cell hierarchy with backoff**: a multi-resolution grid (e.g., 4×4 coarse cells, each recursively subdivided into 4×4 = 256 fine cells in a 16×16 fine overlay). Sparse fine-cell transitions fall back to coarse-cell transitions learned on the same data but pooled across fine sub-cells — capturing fractal self-similarity, reducing parameter explosion, and allowing the filter to express uncertainty hierarchically.
- **Continuous-position particle/state-space model**: maintain belief over continuous (x, y) coordinates via a learned fractional-motion kernel (matching FractalRabbit's motion model), discretized *only for scoring* against discrete cells. The filter works in continuous space (richer expressiveness), but is scored comparably to discrete baselines.

---

## Module-Level Build Plan (File Structure)

### 1. `src/active_fractal_rabbit/hierarchical_cells.py` (NEW)
**Purpose**: Coarse-to-fine cell lattice and backoff transition logic.

**Exports**:
- `HierarchicalGrid(spatial_bounds, coarse_bins, fine_bins_per_coarse)` — creates a 2-level grid hierarchy.
- `cell_to_coarse(fine_cell_id, fine_bins_per_coarse)` → coarse cell id
- `coarse_to_fine_range(coarse_cell_id, fine_bins_per_coarse)` → [fine_cell_id_start, fine_cell_id_end]
- `build_hierarchical_transition(sequence, coarse_bins, fine_bins_per_coarse, smoothing)` → `(T_coarse, T_fine_fallback)` where `T_coarse` is learned on pooled coarse transitions and `T_fine_fallback` selects coarse when fine-cell pair count < threshold (2 observations by default).

**Key design commitment**:
- Do NOT learn separate fine-cell transitions naively (parameter explosion 16×16² + sparse data → overfitting). Instead: fine-cell transitions are queried; if a pair appears < threshold times, emit coarse transition (hierarchical backoff).
- The "fractal" claim is: coarse transitions exhibit learned structure; fine cells that rarely co-occur inherit coarse structure, reducing per-cell estimation variance while preserving place-revisit expressiveness.

**Test**:
- `test_hierarchical_cell_backoff()` — hand-constructed sequence with 2 fine cells in 1 coarse cell, verify backoff fires when needed, coarse transition learned correctly.
- `test_coarse_to_fine_range()` — verify bijection.

---

### 2. `src/active_fractal_rabbit/continuous_particle_model.py` (NEW)
**Purpose**: Continuous-position particle filter with learned fractional motion kernel.

**Exports**:
- `FractionalMotionKernel(fractal_dim, dt, smoothing)` — a 2D displacement kernel parameterized by fractal dimension (learned or fixed). At fractal_dim ≈ 1.33, displacement PDF has longer tails than Brownian but shorter than pure Lévy; approximated via a mixture of isotropic Gaussians (4-component EM-fitted) or a Cauchy-heavy-tail approximation.
- `ContinuousParticleFilter(n_particles, motion_kernel, emission_scale)` — Sequential importance resampling:
  - Predict: each particle's x,y drifts by kernel sample.
  - Update: weight by Gaussian likelihood centered on observed cell's center.
  - Resample when effective sample size < threshold.
- `ParticleFilterTrace` — stores per-step particle positions, weights, effective sample size, and resampling events.
- `discretize_particles_for_scoring(particle_positions, grid_cells)` → discrete cell indices for prequential loss (so scoring is comparable to discrete baselines).

**Key design commitment**:
- The kernel is **learned offline on the training split** via EM on observed displacements (continuous coordinates given → inter-report displacement samples → fit 4-component isotropic Gaussian mixture, or fit a stable-law tail exponent).
- Particles live in continuous [0, 200]² km space, matching FractalRabbit's coordinate system. No separate discretization step.
- Emission is isotropic Gaussian in continuous space, centered on observed cell's rectangle center, with scale learned from observation noise (per-cell empirical variance of observed vs. grid-center position).

**Test**:
- `test_particle_filter_determinism()` — fixed seed, verify same particle trajectory, weights, ESS.
- `test_motion_kernel_fractal_fit()` — hand-constructed Brownian (fractal_dim=2) and Lévy (fractal_dim<<2) samples, verify EM fits recover fractal_dim in expected range.
- `test_discretization_consistency()` — particles in known cells, verify discretized cell indices match.

---

### 3. `src/active_fractal_rabbit/multiscale_inference.py` (NEW)
**Purpose**: Integrate hierarchical and continuous models into the fair-inference ladder and locked-split scoring.

**Exports**:
- `score_hierarchical_on_locked_split(sequence, coarse_bins, fine_bins_per_coarse, smoothing, holdout_fraction, seed)` → `{mean_loss, ci, losses_per_step}` (matching fair_inference.py output shape).
- `score_continuous_particle_on_locked_split(sequence, n_particles, motion_kernel, holdout_fraction, seed)` → same.
- Both functions:
  1. Lock train/holdout on traveller id (reuse `lock_traveller_split`).
  2. Fit the model (hierarchical backoff, particle kernel EM) **only on train** data.
  3. Score prequentially on holdout, with **discretized output** for direct loss comparison to single-grid baselines.
  4. Return per-step losses (so bootstrap CI matches fair_inference.py).

**Key design commitment**:
- **No oracle access** — both models infer cell from observations (like fair_inference soft/aif), never see true cell at prediction time.
- **Discretization is scoring-only** — particles are marginalized to cell-level before loss computation, so uncertainty in continuous space does not artificially inflate loss.
- **Matched feature set** — both hierarchical and continuous get the same optional return-boost (visited-cell prior) as the soft/aif baselines, so gains are not confounded by asymmetric side information.

---

### 4. `src/active_fractal_rabbit/spatial_resolution_probe.py` (UPDATED/EXTENDED)
**Purpose**: Confirm the confinement risk and verify non-confined test bed.

**New probe**:
- `mobility_metrics(sequence)` → `{mean_distinct_cells_per_traveller, entropy_of_cell_occupancy, occupancy_ratio}` where:
  - `mean_distinct_cells_per_traveller` = avg # of distinct cells visited per traveller (must be > 3 for multiscale claim).
  - `entropy_of_cell_occupancy` = Shannon entropy of aggregate traveller × cell histogram (must be > 1.0 for good exploration).
  - `occupancy_ratio` = (# cells with >=1 traveller step) / (# total cells) — should be << 1 for confined regime risk, > 0.4 for non-confined.

**Test bed validation**:
- Default config (4×4 grid, 4 travellers, 240 reports): measure mobility_metrics on both fixture and external.
- Fine-grid config (8×8 or 16×16 grid, same travellers/reports): measure mobility_metrics again.
- **Kill gate 1** (non-confined test): mean_distinct_cells_per_traveller must increase by >= 2× when finer grid is used (proof that confinement risk exists and the finer grid addresses it).

---

### 5. `configs/multiscale_hierarchical.yaml` (NEW)
**Purpose**: Configuration snapshot for hierarchical model experiments.

```yaml
run_id: multiscale_hierarchical_probe
seed: 7
output_dir: output/runs/multiscale_hierarchical

fractalrabbit:
  # (same as default.yaml)

observation:
  spatial_bins: 8  # Fine-grid test bed (non-confined: 8×8 = 64 states vs 4×4 = 16)
  spatial_discretizer: grid
  # ... other fields as default

model:
  # (same as default)

multiscale:
  enabled: true
  coarse_bins: 4  # 4×4 coarse cells (matches default single-grid)
  fine_bins_per_coarse: 2  # 2×2 fine cells per coarse → 8×8 fine = 64 total (matches fine-grid observation.spatial_bins)
  backoff_threshold: 2  # If a fine-cell pair appears < 2 times, fall back to coarse

continuous:
  enabled: false  # (toggle for future continuous particle runs)
  n_particles: 256
  fractal_dim_fixed: 1.33  # Can be 'learned' in future

experiment:
  # (same as default)
```

---

### 6. `scripts/multiscale_inference_sweep.py` (NEW)
**Purpose**: Drive multiscale + continuous model family scoring against confound controls.

**Workflow**:
```python
def run_multiscale_sweep(config_path, output_dir):
    """Sweep hierarchical and continuous models on locked splits."""
    
    # 1. Load sequence from fixture/external.
    # 2. Compute mobility_metrics(sequence); emit to a probe artifact.
    # 3. Score baselines (oracle, hard, soft, soft+boost, aif, aif+boost) on locked split.
    # 4. Score CONFOUND CONTROLS:
    #    a. dominant_cell_baseline() — always predict the most-visited cell (pure confinement).
    #    b. first_order_pooled_markov() — population first-order Markov (data density).
    #    c. oracle_persistence() — last-true-cell (asymmetry bound).
    # 5. Score hierarchical_on_locked_split(coarse_bins, fine_bins_per_coarse).
    # 6. Score continuous_particle_on_locked_split(n_particles, kernel_type).
    # 7. Emit comparison_metrics.json:
    #    {
    #      "mobility_probe": {mean_distinct_cells, entropy, occupancy_ratio},
    #      "confound_controls": {dominant_cell_loss, first_order_markov_loss, oracle_persistence_loss},
    #      "baselines": {oracle, hard, soft, soft_boost, aif, aif_boost},
    #      "candidates": {hierarchical, continuous_particle},
    #      "model_wins": {
    #        "hierarchical_beats_controls": hierarchical_loss < max(dominant_cell, first_order) AND < soft_boost,
    #        "hierarchical_beats_aif": ...,
    #        "continuous_beats_controls": ...,
    #      }
    #    }
    # 8. Report: does the model capture structure Markov cannot, AND beat confound controls?
```

---

## Confound Controls (Required, Non-Negotiable)

Every proposed model MUST be scored against ALL of:

1. **`dominant_cell_baseline`** — predict the single most-visited cell across all training data. Loss is -log P(true cell | most-visited). This isolates **pure confinement**: if a model beats this, it's not just inheriting a confinement artifact.

2. **`first_order_pooled_markov`** — learn a population-wide first-order transition matrix P(cell_t | cell_{t-1}) on training data, apply it to holdout steps. This tests **data density**: if pooled Markov suffices, long-memory or per-traveller structure adds nothing.

3. **`oracle_persistence`** — always predict the previous true cell (last-cell-seen baseline). This is the strong benchmark ceiling under this task contract because it receives a more informative state input than location-blind methods.

**Passing criterion**: A model only "wins" if:
- `loss(model) < loss(dominant_cell)` AND
- `loss(model) < loss(first_order_markov)` AND
- `loss(model) < loss(oracle_persistence)` is FALSE (not even trying).

If the model only beats one, it's a confound artifact, not a genuine inference win.

---

## Non-Confined Test Bed Specification

The current default (4×4 grid, 4 travellers, 240 reports over 18 days) will likely show 5/8 travellers in ≤3 cells (confinement risk). To build a fair test:

**Test Bed A: Fine-Grid Regime**
- Config: 8×8 = 64 cells (double resolution per axis).
- Same waypoints, observations, travellers, reports as default.
- Mobility check: mean_distinct_cells_per_traveller must increase by >=2× (from ~2-3 to ~4-6).
- **Result**: individual models can no longer trivially win; confinement risk reduced.

**Test Bed B: Structured Mobile Regime (Future)**
- Higher traveller count (16+) with **intentional inter-traveller mixing** (co-traveller parameter > 1, or synthetic scenario with aggregation points).
- Mobility check: occupancy_ratio > 0.5 (more cells visited), entropy > 1.5 (flatter occupancy).
- This regime forces the model to learn shared structure (not just confine per-traveller).

---

## Honest Expectation + Kill Gates

**Kill Gate 1: Confinement Risk Unresolved**
- If `mean_distinct_cells_per_traveller` does not improve when finer grid is used (stays ~2-3 in 8×8), the test bed is still confined.
- Action: stop claiming multiscale advantage; refactor synthetic regime for higher mobility or use a different generator.
- **Evidence**: mobility_metrics emitted to artifact; test pins threshold.

**Kill Gate 2: Model Exceeds State-Space Complexity**
- Hierarchical fine-cell transitions: if # observed fine-cell pairs > 512 and sparsity (distinct pairs / max possible) < 0.1, the parameter space is too large relative to data.
- Action: increase backoff_threshold or coarsen fine grid.
- **Evidence**: transition-sparsity metric emitted; test pins threshold (sparsity must be > 0.15 or backoff_threshold auto-increases).

**Kill Gate 3: Continuous Model Particle Degeneracy**
- If effective sample size (ESS) drops below n_particles / 4 and does not recover within 3 steps, resampling has collapsed particle diversity.
- Action: increase n_particles or tighten emission scale (model is overconfident).
- **Evidence**: per-step ESS trace emitted; test requires min ESS > n_particles / 10 on >= 80% of steps.

**Kill Gate 4: Structured Gain Below Noise Floor**
- If `|loss(hierarchical) - loss(soft_boost)| < 0.01` nats (within measurement noise of 100-step bootstrap), the difference is not meaningful.
- Action: do not claim a win; instead report "comparable, with trade-offs in interpretability" and investigate why the hierarchical structure did not help.
- **Evidence**: loss CI bands emitted; 95% CI must not overlap.

---

## Scoring vs. Confound Controls (Discipline Summary)

| Baseline | Role | Must Model Beat? | Why |
|----------|------|------------------|-----|
| dominant_cell | Pure confinement floor | YES | Proof structure beyond "stuck in one cell" |
| first_order_markov | Data-density null | YES | Proof per-traveller/long-memory matters |
| oracle_persistence | Asymptotic upper bound | NO (must lose) | Proof model doesn't cheat |
| soft_boost | Fair non-recursive benchmark | YES (recommended) | Proof recursive structure helps |
| aif + boost | Strong fair recursive baseline | MAYBE | Competitive comparison (not required for win) |

---

## File-Level Build Summary

| File | Status | Purpose |
|------|--------|---------|
| `hierarchical_cells.py` | NEW | Coarse-to-fine grid, backoff transitions |
| `continuous_particle_model.py` | NEW | Particle filter, fractional motion kernel |
| `multiscale_inference.py` | NEW | Locked-split scoring, integration |
| `spatial_resolution_probe.py` | EXTENDED | Mobility metrics, kill-gate checks |
| `multiscale_hierarchical.yaml` | NEW | Config snapshot for experiments |
| `multiscale_inference_sweep.py` | NEW | CLI orchestrator, comparison_metrics emission |
| `tests/test_hierarchical_cells.py` | NEW | Unit tests (backoff, bijection) |
| `tests/test_continuous_particle.py` | NEW | Unit tests (kernel fit, discretization) |
| `tests/test_multiscale_confound_controls.py` | NEW | Confound-control validation, kill-gate tests |
| `tests/test_mobility_metrics.py` | NEW | Mobility probe validation, confinement risk test |

---

## Orchestration Stage (in `scripts/07_run_full_study.py`)

```python
if "multiscale" in lanes:
    print("Running multiscale inference sweep...")
    multiscale_output = run_multiscale_sweep(
        config_path=config_path,
        output_dir=Path(config.output_dir) / "multiscale"
    )
    # Emit: comparison_metrics.json, mobility_probe.json, kill_gate_report.json
    print(f"  Confound controls: {multiscale_output['confound_controls']}")
    print(f"  Model wins: {multiscale_output['model_wins']}")
    if not multiscale_output['kill_gates_passed']:
        print(f"  WARNING: Kill gates failed — {multiscale_output['kill_gate_reasons']}")
```

---

## Analysis Stage (in `outputs/` artifact generation)

1. **mobility_probe.json** — {mean_distinct_cells, entropy, occupancy_ratio} for default + fine-grid.
2. **comparison_metrics.json** — {confound_controls, baselines, candidates, model_wins, ci_bands}.
3. **kill_gate_report.json** — {gate_1, gate_2, gate_3, gate_4, overall_pass}.

---

## Manuscript Integration

When results are ready:

- **Methods**: brief paragraph on hierarchical backoff (self-similarity) and continuous particle model (fractal motion kernel fit).
- **Results**: "On the fine-grid non-confined test bed (mean_distinct_cells increased 2.3×), the hierarchical model achieved {loss} nats [95% CI], beating dominant-cell {diff} nats and first-order Markov {diff} nats, while remaining below oracle-persistence {diff} nats. The continuous particle model achieved {loss} nats [95% CI], capturing fractal motion structure and offering comparable loss with increased interpretability."
- **Limitations**: "Multiscale models tested only on synthetic FractalRabbit data at a single fractal dimension (1.33); generalization to other generators and real trajectories remains future work. Continuous particle filter requires careful EM initialization and ESS monitoring; computational cost not compared."

---

## Honest Null & Fallback

If all kill gates fail or models underperform:
- **Report honestly**: "The hierarchical and continuous models did not outperform matched fair baselines (soft+boost, aif+boost) on the FractalRabbit fixture test bed. Confinement risk was not fully resolved even with finer grids; the synthetic regime may not exercise sufficient inter-traveller mixing or revisit structure to benefit from hierarchical discretization."
- **Next steps**: build a higher-mobility synthetic regime or admit that the single grid with return-boosting suffices for this domain.



## Family: Variable-order / reinforced-transition Markov with long-memory return structure

**Test bed:** ["Regime A: Default fixture (4 travellers, 4x4 grid, 48 steps/traveller), mobility=0.59, locked-split holdout_fraction=0.5, first-order baseline ~0.6 nats (reference)", "Regime B: Fine-grid synthetic (same waypoints, 8x8 grid, 64 cells), mobility target >=0.40, holdout_fraction=0.5 (falsification check: if advantage reverses, confinement artifact)", "Cross-regime validation: report mobility bars for both; reject if either < 0.30 or if regime B reverses the win"]

**Confound controls:** dominant_cell_baseline (marginal-frequency mode predictor); first_order_pooled_population_markov (online-learned transition, no per-traveller boost); oracle_persistence (true_previous_cell, asymptotic upper bound); mobility_check (mean_distinct_cells_per_traveller >= 0.40 to exclude confinement); structure_check (return_gap_distribution bimodal test; exponential-only rejects); revisit_decomposition (loss on revisit_steps < loss on novel_steps required for signal); power_gate (holdout_traveller_count >= 4 minimum; otherwise directional-hypothesis-generating)

**Kill-gates:** Confinement gate: mean_distinct_cells_per_traveller < 0.30 in any regime → reject; Structure gate: return_gap_distribution is exponential-only (no multimodal peaks) → honest null; Baseline beating gate: model_loss >= first_order_markov_loss on holdout → reject; Revisit gate: loss_on_revisit_steps >= loss_on_novel_steps → reject (no predictive structure detected); Power gate: holdout_traveller_count < 4 → directional/hypothesis-generating only (no CI claim); Cross-regime gate: if regime A wins but regime B reverses → confinement artifact, reject main claim

# Long-Memory Family Design: Variable-Order Markov with Reinforced Return

## Generative Form

The long-memory predictor family captures **retro-preferential return to cells visited long ago** (not just last-cell dominance). Three candidate forms are designed to match the FractalRabbit generator's RETRO-PREFERENTIAL three-tier process:

### Candidate 1: Variable-Order Markov (Context Tree, α ∈ {0.5, 1.0, 1.5})
- **Core idea**: The transition probability depends on the **recent history** (prior k steps), not just the immediate previous cell.  
- **Generative form**: For each traveller, maintain **visited-cell suffixes** in a history buffer (e.g., last 2–4 steps). The transition is:
  ```
  P(next | history) = counts[next, history_state] / Z(history_state)
  ```
  where `history_state` is a quantized context (e.g., "was in cell A then B then C").
- **Scoring**: Online prequential next-cell loss, learning the history-conditioned counts from true observed cells (same leakage guard as baselines).
- **Key parameter**: History depth (k=1 recovers first-order Markov; k≥2 captures long-range revisit bias).

### Candidate 2: Polya-Urn Reinforced Transition (Pitman-Yor, α ∈ {0.5, 1.0, 1.5})
- **Core idea**: Each transition's strength is **reinforced by prior use**, biasing future steps toward cells that have been productive in similar contexts.
- **Generative form**: Start with a smoothed transition matrix; at each step, increment a **reinforcement count** for the observed (previous, next) pair. The prediction marginalizes both the base transition and the accumulated reinforcement:
  ```
  pred_t = (1 - α) * base_transition @ posterior + α * normalize(reinforcement_counts)
  ```
  where `α` is the reinforcement fraction and `reinforcement_counts[i]` tracks how often cell i has been transitioned to.
- **Scoring**: Online prequential, with counts and reinforcement state reset per traveller (fair comparison).
- **Key parameter**: `α` (mixing weight; α=0 is pure first-order, α=1 is pure reinforcement).

### Candidate 3: Recency-Weighted Visit-History Kernel (Exponential decay, τ ∈ {0.5, 1.0, 2.0})
- **Core idea**: Recent visits to a cell are worth more than old visits; exponential decay weights them.
- **Generative form**: For each traveller at step t, compute a **recency-biased visit prior**:
  ```
  visit_prior[cell] = sum_{visit times t'} exp(-(t - t') / τ)
  pred = (1 - β) * (B @ posterior) + β * normalize(visit_prior)
  ```
  where `τ` is the decay timescale (small τ = only recent visits count; large τ = all visits matter equally).
- **Scoring**: Online prequential next-cell loss; the visit timeseries is per-traveller, maintained during the forward pass.
- **Key parameter**: `τ` (decay timescale in report units).

---

## Scoring vs Confound Controls: Locked-Split Harness

Every proposed model is measured against **three simultaneously run** baselines on the **same holdout split**:

1. **Dominant-Cell Baseline**: The marginal mode cell at each step (ignores all dynamics). Loss = `-log P(true_cell)` where P is the empirical mode occupancy.
   - **Reason**: Confinement trivially makes any model win if the agent stays in ≤3 cells; this captures that freebie.
   - **Measurement**: Run once per dataset (deterministic).

2. **First-Order Pooled-Population Markov** (the transparent baseline):
   - Transition matrix learned online from true observed cells (shared transition source with the proposed model).
   - No per-traveller visited-cell boost (the fair baseline for comparison).
   - **Reason**: Data density (more travellers = sufficient pooled population transition) is a real confound; if the proposed model beats this, it is not just density.
   - **Measurement**: Prequential next-cell loss, holdout fraction = 0.5, shared smoothing.

3. **Oracle Persistence** (asymptotic upper bound):
   - Prediction = "stay in the true previous cell" (cheats by using the ground-truth cell).
   - **Reason**: No model should beat this; it proves the prediction handicap is real and the comparison is fair.
   - **Measurement**: Prequential, same holdout split.

**Decision rule**: A model "wins" only if **loss(model) < loss(first_order_markov)** AND the holdout steps are from ≥ 4 distinct travellers (low power guard). Publish directional result + confound note if power is low.

---

## Non-Confined Test Bed: Mobile Regime

All models are scored on **two regimes** to prove the advantage is not an artifact of confinement:

### Test Regime A: Default Synthetic (Fixture-Scale, 4 Travellers × 48 Steps)
- Grid: 4×4 cells (16 total).
- Mobility check: Mean distinct cells per traveller = **9.4 / 16 = 0.59** (well-mixed, not confined).
- **Measurement**: Standard locked-split scoring, all three baselines.

### Test Regime B: Fine-Grid Synthetic (Finer Spatial Resolution, 8×8 = 64 Cells)
- Re-discretize the same waypoint data on a finer grid (8×8 instead of 4×4).
- Mobility check: Mean distinct cells per traveller = **target ≥ 0.40** (ensures travellers move through multiple cells even at finer resolution).
- **Reason**: If the advantage shrinks or reverses at finer spatial resolution, the original win was a spatial-confinement artifact, not long-memory structure.
- **Measurement**: Same locked-split protocol, all baselines.

**Validation gate**: Report mobility metrics (mean distinct cells, entropy per traveller) for both regimes. Claim is null if mobility is below threshold in either regime.

---

## Structural Validation: What Must Be Present + Predictive

The long-memory model claims to capture **long-memory return-to-old-places structure**. This structure must:

1. **Be measurable in the data**: Compute **recency-adjusted autocorrelation** of return times within each traveller.
   - For each traveller, compute time gaps between consecutive returns to the same cell.
   - Hypothesis: Gap distribution is multi-modal (spikes at recent times due to reruns, older peaks due to preferential return).
   - **Measurement**: Plot distribution of return-gap times; if roughly exponential-only, long-memory is absent.

2. **Be predictive**: The long-memory model predicts better on **high-revisit steps** (steps where the destination was visited long ago) than on **low-revisit steps** (novel cells or immediate repeats).
   - Partition holdout steps into "revisit" (destination in visit history, time gap > threshold) vs "novel" (destination is a first visit).
   - **Measurement**: Report loss on revisit steps vs novel steps; if revisit loss is lower for the long-memory model, the structure is present + predictive.
   - **Validation gate**: If revisit loss is not lower, claim is null.

---

## Honest Expectation + Kill-Gate

### Expected Outcome (No Guarantee)
On the **default fixture** (4 travellers, 4×4 grid, 48 steps per traveller):
- The long-memory model **may beat first-order Markov by 0.2–0.6 nats** (on revisit steps specifically) if the synthetic generator is structured to favor historical returns.
- **BUT** the advantage will **likely vanish at scale** (e.g., 32 travellers) because the pooled transition matrix becomes sufficient when data density is high.
- Under **clean emissions** (high observation quality), the advantage is small because last-cell information alone (first-order) is already strong; advantage emerges under **noisy emissions**.

### Kill-Gate (Cheap Falsification)

A model is rejected (honest null) if ANY of these hold:

1. **Confined travellers**: Mean distinct cells per traveller < 0.30 (in either test regime).
2. **No structure in data**: Recency-adjusted return-gap distribution is exponential-only (no bimodal peaks).
3. **Does not beat first-order Markov** on the locked split, on holdout steps, with ≥4 distinct holdout travellers.
4. **No revisit advantage**: Loss on revisit steps is not lower than on novel steps.
5. **Negative cross-regime**: If Test Regime A wins but Test Regime B reverses (confinement artifact), reject.

**Publication rule**: Report directional finding + confound controls + mobility metrics. If power < 4 distinct holdout travellers, say "hypothesis-generating, underpowered." Never assert a win under kill-gate conditions.

---

## File-Level Build Plan

### New Module: `src/active_fractal_rabbit/longmemory.py`

**Functions to implement**:

1. **`history_context_markov_loss(sequence, depth=2, smoothing=0.5, holdout_fraction=0.5, seed=20260616)`**
   - Builds a (n_states, n_states, n_contexts) transition tensor indexed by history state.
   - Scores prequential next-cell loss.
   - Returns: loss, CI, per-step loss array for clustered bootstrap.

2. **`polya_urn_reinforced_loss(sequence, alpha=0.5, smoothing=0.5, holdout_fraction=0.5, seed=20260616)`**
   - Maintains reinforcement counts per traveller.
   - Scores prequential next-cell loss with mixing weight α.
   - Returns: loss, CI, per-step loss array.

3. **`recency_weighted_kernel_loss(sequence, tau=1.0, beta=0.5, smoothing=0.5, holdout_fraction=0.5, seed=20260616)`**
   - Computes exponential-decayed visit prior at each step.
   - Scores prequential next-cell loss.
   - Returns: loss, CI, per-step loss array.

4. **`structure_check_return_gaps(sequence)`**
   - Compute per-traveller distribution of time gaps between revisits to the same cell.
   - Return: histogram, KL divergence from exponential, modal peaks.
   - **Validation**: Return a boolean flag indicating bimodal structure (or None if weak).

5. **`revisit_vs_novel_decomposition(sequence, model_loss_per_step, holdout_mask)`**
   - Partition holdout steps into revisits (destination in history, gap > threshold) vs novel.
   - Report mean loss on each partition.
   - Return: {"revisit_loss": float, "novel_loss": float, "revisit_count": int, "novel_count": int}.

6. **`longmemory_sweep(sequence, confound_controls_dict)`**
   - Run all three candidate forms (variable-order, Polya-urn, recency-kernel) at multiple hyperparameter settings.
   - Score each against the three baselines (dominant-cell, first-order, oracle).
   - Return: comprehensive result dict with all losses, CIs, mobility checks, structure checks, revisit decompositions, and honest verdict.

### Orchestration: `scripts/14_run_longmemory_analysis.py`
- Load fixture or external sequence.
- Call `longmemory_sweep()`.
- Write artifact to `output/runs/{run_id}/longmemory_analysis.json`.
- Surface in `scripts/07_run_full_study.py` as a new study stage.

### Figures: `src/active_fractal_rabbit/figures.py`
- New figure: **`longmemory_comparison`** (4-panel: loss landscape for each candidate form, mobility bars, structure check, revisit decomposition).
- Register in the figure manifest.

### Manuscript Integration
- New section: "Long-Memory / Preferential-Return Structure" (∼1 page).
- Hydrated tokens: {{LONGMEMORY_BEST_FORM}}, {{LONGMEMORY_ADVANTAGE_NATS}}, {{LONGMEMORY_REVISIT_ADVANTAGE}}, {{LONGMEMORY_MOBILITY_CHECK}}, {{LONGMEMORY_VERDICT}}.
- Confound-control table: baseline losses + proposed loss + holdout traveller count.
- Honest caveat: "This analysis probes long-memory structure on a synthetic generator with designed return bias. On a first-order Markov null, the advantage would vanish; on real human mobility, long-memory structure and its learnability are unknown."

### Tests: `tests/test_longmemory.py`
- Unit tests for history context (hand-computed transition on toy sequence).
- Unit tests for Polya-urn reinforcement (α=0 recovers first-order, α=1 recovers pure reinforcement).
- Unit tests for recency kernel (τ→0 ignores history, τ→∞ treats all visits equally).
- Integration test: default fixture, all three forms, all baselines, mobility/structure checks.
- Kill-gate test: confinement detector (rejects if mean distinct < 0.3).
- Regression test: on the default fixture, the first-order baseline loss is pinned to a reference value (catches silent breakage).

---

## Honest Expectation Summary

- **Most likely**: The variable-order Markov at depth k=2 beats first-order by ~0.3 nats on the fixture **if** the FractalRabbit generator is tuned to produce long-memory returns. On revisit steps only, advantage may be ~0.5–0.7 nats.
- **Power caveat**: 4 travellers × holdout fraction 0.5 = ~2 holdout travellers. This is underpowered for a CI-separated claim. Publish as directional + design-truth confound disclosure.
- **Scale caveat**: The advantage will likely shrink to near-zero at 32 travellers (population transition model becomes sufficient). This is **not** a model failure — it is honest regime-scoped inference.
- **Noise caveat**: The advantage is largest under noisy emissions (emission_smoothing > 0.5). On clean emissions (sharp location inference), first-order Markov is already close to optimal.
- **Confinement caveat**: If mobility metrics show confinement, reject the result.

## Build Feasibility

✓ **Buildable**: All components fit the existing `fair_inference` + `aif_sweep` harness. The locked-split machinery is reusable; only the predictor logic is new.

✓ **Testable**: Unit and integration test scaffold exists (`tests/test_aif_sweep.py`, etc.). Hand-computed reference values (k=1 → first-order Markov) validate the implementation.

✓ **Honest**: Confound controls are upfront (dominant-cell, first-order, oracle). Kill-gates are checkable (mobility, structure, power). No silent wins.
