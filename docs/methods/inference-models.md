# Inference Models: Predictor Hierarchy and Oracle Boundaries

## The Discrete POMDP Framework Defines Belief State

The inference backbone is a discrete-state POMDP constructed from sparse waypoint observations. The hidden state is the discretized spatial cell *s ∈ {0, ..., S-1}*. The model comprises:

- **Likelihood A(m)** — per-modality observation matrices: P(o(m) | s), normalized columns.
- **Transition B(u)** — control-conditioned dynamics: P(s' | s, u), learned from within-traveller cell transitions, normalized columns.
- **Preferences C(m)** — logged preference distributions over observations (zeros by default).
- **Initial prior D** — P(s₀), seeded from the first observed cell on the train travellers.

The model is deterministic once constructed and frozen during evaluation. A single control u is selected at each step (nominally `empirical_continuation`, which learns the observed transition distribution; alternatively `high_entropy_exploration` or `identity`). All arrays are built by `pomdp.py` using Dirichlet pseudocounts, with configurable smoothing.

## The Reference Filter Implements Bayesian Forward Filtering

The reference filter is a pure Bayesian state-space recursion, implemented in the active-inference diagnostic lane and the fair-comparison analyses:

```
prior[0] = D
posterior[i] = σ(log D[i] + Σ_m log A(m)[o[i](m), :])
D[i+1] = normalize(B[u[i-1], :] @ posterior[i])
```

where σ normalizes onto the simplex. The predictive loss at step i is:
```
ℓ[i] = -log P(o[i] | D[i]) = -log(Σ_s P(o[i] | s) D[i](s))
```

This recursion has no approximation; it computes exact posterior probabilities. All online-adaptation and modality-weighting levers operate within this same forward-pass structure — no re-inference, no variational fixed-point iteration. Emission smoothing is a dominant control: a sharp emission (low smoothing) makes P(s | obs) peaky, approaching identity when the location observation *is* the cell itself; a diffuse emission leaves the agent uncertain, forcing temporal belief to carry more weight.

## The Evaluation Harness Uses Locked Splits

All scoring follows a traveller-stratified locked split (from `sota_benchmark.py` and `aif_sweep.py`):
- **Train travellers:** used once to estimate A, B, D from observed cell transitions.
- **Holdout travellers:** scored prequentially, never seen during estimation.
- **No within-traveller leakage:** each traveller's stream is entirely in either train or holdout; the split is deterministic on seed and sorted traveller IDs.

Online predictors are primed on train statistics and then updated (counting) as they score holdout steps. This matches the prequential scheme the transparent baselines use, ensuring a fair comparison task contract.

## The Predictor Ladder Separates Information Sets

The core fair-comparison finding decomposes inference gains by holding the transition source identical (learned from true observed cells) and varying only the previous-cell representation used for prediction. All six predictors share the same transition training data; the handicap is purely at prediction time.

### Oracle Baseline Is Leakage by Design
```
pred[i] = counts[:, true_prev_cell] / sum
```
The oracle predictor receives the true previous cell — the leaderboard predictor and adversarial probe both use it. This is the transparent SOTA baseline's advantage: it receives oracle knowledge, so it sets the minimum achievable loss on the locked split.

### Hard Predictor Commits to a Point Estimate
```
inferred_cell[i] = argmax_s P(s | obs[i])
pred[i] = counts[:, inferred_cell] / sum
```
A hard argmax over the emission posterior. The inferred cell is a point estimate with no uncertainty quantification. This handicapped Markov predictor faces the same emission-identification cost the active-inference filter faces, but lacks the temporal belief and inference formalism.

### Soft Predictor Marginalizes the Belief
```
post_obs[i] = A[obs[i], :] / sum_s A[obs[i], s]
pred[i] = (counts / counts.sum(axis=0)) @ post_obs[i]
```
The full emission posterior is marginalized through the transition. A single step of Bayesian inference with no temporal memory. This soft Markov predictor is structurally neutral — it averages over cell uncertainty rather than committing to a point estimate — and measures the pure inference gain from soft vs. hard belief representation without temporal accumulation.

### Soft+Boost Adds the Return Feature
```
visited[tid] = cells visited by traveller tid so far
pred[i] = soft[i] + return_boost * visited / max(count_visited, 1)
```
The soft predictor with per-traveller visited-cell mass added. Implements a simple form of memory: cells the traveller has visited before are upweighted. Whether this fair non-recursive baseline beats the boost-free recursive filter is a regenerated-artifact fact recorded per run in `fair_inference.json` (`ladder.return_boosted_soft_beats_bare_aif`); on the current fixture it does not — the return boost worsens both predictors.

### AIF Adds Recursive Belief Filtering
```
posterior[i] = σ(log D[i] + log A[obs[i], :])
D[i+1] = B @ posterior[i]
pred[i] = D[i+1]
visited[tid] = tracked per-traveller
```
The Bayesian forward filter with online adaptation optionally enabled. Maintains a full belief distribution over cells, accumulates temporal evidence, and optionally learns transition counts online. Two configuration variants:
- **Bare AIF** (`online_adapt=True, return_boost=0.0`): pure temporal recursion.
- **AIF+Boost** (`online_adapt=True, return_boost=1.0`): recursive belief + visited-cell feature, matching the soft+boost feature set and the locked-split comparison in fair-inference tests.

## Online Adaptation and Return Boosts Are Candidate Levers

### Online Adaptation Updates Within the Stream
When `online_adapt=True`, the reference filter maintains online Dirichlet counts during holdout scoring:
```
trans_counts[i] = trans_counts[i-1] + observed_transition
B[u] = trans_counts / trans_counts.sum(axis=0)
```
This is the direct analogue of the prequential Markov baseline: both learn on holdout data as they score. Without online adaptation, the reference filter holds B fixed at its train-derived value. The fair comparison explicitly records whether this lever closes or enlarges the gap to the strongest baseline.

### Return Boost Encodes Visited-Cell Structure
When `return_boost > 0`, the predictive is re-weighted by a return-conditioned prior (per-traveller visited-cell indicator):
```
predictive[i] = trans_result[i] + return_boost * visited[tid] * trans_result[i].sum() / count_visited
```
This is a lightweight model of return-to-base behavior: cells the traveller has already visited are upweighted. The boost is applied uniformly to all five predictors in the ladder, so it becomes a constant feature that neither explains nor reverses ranking differences. Its presence in the AIF+Boost configuration ensures a fair matched-feature comparison with soft+boost.

## Precision Weighting Is an Inference-Internal Lever

An optional precision (inverse-temperature γ) sharpens the predictive distribution:
```
predictive_sharp = predictive^γ / sum
```
When γ > 1, this raises high-probability cells and suppresses low ones, trading entropy for confidence. When γ = 1.5 (the fixture-selected sweet spot), it lowers both loss and ECE on the default fixture. This is a purely post-hoc confidence calibration lever — it is never fed back into the fair-comparison verdict, which is computed at γ = 1. The precision gain is reported separately as a paired self-improvement finding, not as evidence that the AIF beats baselines.

## Emission Smoothing and Deep-Latent Configs Bound Uncertainty

The emission matrix A is built from per-modality counts with a configurable smoothing parameter:
```
A[obs, state] = count[obs, state] + smoothing
A_normalized[obs, :] = A[obs, :] / sum_s A[obs, s]
```

When location is the discretization of true waypoint positions and the spatial-state cell is the same discretization, the location emission is nearly identity (P(cell obs | cell) ≈ 1). Other modalities (time-of-day, speed, report-burst) are indirect evidence, observed through the same cell but carrying less discriminative power.

A **deep-latent configuration** drops the direct location modality, leaving the cell observable only through time, speed, and burst data. This increases the emission-identification burden and tests whether the filter can infer cells from indirect evidence alone. The fair comparison includes a modality-subset probe (`modalities=("location",)` vs. full) to measure this effect.

## Oracle Previous-Cell Baseline Bounds Information Asymmetry

The persistence comparator is an oracle baseline in this benchmark because it receives the true previous cell at prediction time. A location-blind predictor instead receives only non-location evidence and must infer any previous-cell information before propagating through the transition model. The measured comparison therefore prices a benchmark-specific information-set asymmetry:

- The oracle predictor scores P(s[i] | s[i-1] = true_prev).
- A location-blind predictor scores from P(s[i] | non-location obs[i]) and any inferred belief over s[i-1].
- On the current synthetic generator and locked split, that weaker information set does not close the gap to the oracle.

This is not a universal impossibility theorem about all location-blind predictors or all mobility processes. The honest question is whether soft inference or temporal recursion beats hard point estimates *given the same handicap*. That is what the predictor ladder measures.

## The Null Hierarchy Sets the Comparison Floor

### Uniform Floor Is the Weakest Null
The lowest null: a predictor that assigns equal probability to all states at every step.
```
loss_uniform = sum_m log(num_categories[m])
brier_uniform = (S - 1) / S
```
Serves as the trivial baseline for skill scoring. Skill = 1 - loss / floor; 0 is chance, negative is worse than chance.

### Base-Rate Null Uses Marginal Frequency
A predictor that ignores observations and timing, and predicts the empirical cell-frequency vector at every step:
```
base_rate[s] = count[s] / total
loss_base_rate = -log(Σ_s base_rate[s] I[cell[i] = s])
brier_base_rate = 1 - Σ_s base_rate[s]²  (Gini impurity)
```
Tighter than uniform whenever cells are not uniformly occupied. Measures how much observing *anything* beats already knowing the base rate. Reported via `objectives.base_rate_brier()`.

### Persistence Null Uses Temporal Stay-Put Structure
A predictor that ignores observations and always predicts the previous true cell (or base-rate at traveller-first steps):
```
pred[i] = [1 @ s[i-1] if i > 0 else base_rate]
brier_persistence = mean_step(||pred - cell||²)
```
For autocorrelated mobility (high self-transition rates), persistence is much stronger than base-rate, so skill against it is the honest measure of how much observations add beyond "assume they didn't move." On anti-correlated data where transitions dominate, the ordering flips. Reported via `objectives.persistence_brier()` and computed on synthetic waypoint data. This is a regime-dependent null, not a universal constant.

## Fair-Comparison Orchestration Holds Inputs Fixed

The `fair_inference.py` module implements the locked-split predictor ladder via `fair_inference_ladder()`:
1. Locks the traveller split (fixed seed, reproducible).
2. Builds the emission A from train travellers at specified smoothing.
3. Computes all six predictors (oracle, hard, soft, soft+boost, AIF, AIF+boost).
4. All predictors learn transitions from the same true observed cells (leakage guard).
5. Holds transition B fixed (no online adaptation at this stage; online adaptation is a separate lever).
6. Scores prequentially on holdout steps.
7. Returns metrics (loss, CI, Brier, ECE) and decomposition (soft gain, recursive gain, total gain).

The fair-comparison result is **synthetic-only** and **directional**. No predictor beats the oracle-fed baseline; the finding is whether soft and recursive inference beat hard and static baselines when both face the same emission identification cost. This decomposes the inference contribution to the AIF's loss from confounds like the return-feature or online adaptation.

## Locked-Split Sweeps Decide Claim-Gate Eligibility

The `aif_sweep.py` module scores POMDP configurations on the locked split via `score_pomdp_on_locked_split()`:
1. Trains A/B/D on the train travellers' observations.
2. Forward-filters each holdout traveller's stream.
3. Scores -log P(next cell | prior, obs) at each step.
4. Returns mean loss, bootstrap CI, Brier, and ECE.

The sweep grid explores structural levers (online_adapt, return_boost, precision, modalities) and ranks configurations by loss against the strongest transparent baseline. A configuration is **CI-separated** if its upper CI bound is below the baseline's lower CI bound — a measured verdict. No assertion is made about whether any configuration wins; the verdict is artifact-measured on the synthetic lane. An honest null is explicitly returned if no configuration beats the baseline.

The module also orchestrates the `fair_comparison_on_locked_split()` finding: an oracle-stripped comparison where Markov predictor and AIF predictor both infer the previous cell from emissions (neither receives the true cell), measuring whether AIF's soft Bayesian belief beats a hard-argmax Markov under fair conditions.

## Calibration and Skill Scores Stay Secondary

Per-lane objectives are computed by `objectives.py::lane_objectives()`:
- **Multiclass Brier score** against the true cell.
- **Top-1 Expected Calibration Error (ECE)** with configurable bin count (default 5).
- **Per-modality loss decomposition:** the joint loss -log P(o | s) is split into marginal losses -log P(o(m) | s) per observation factor. The gap is the modality interaction information. This is binding-checked against the filter's own recorded loss to prevent silent drift.
- **Skill scores:** against uniform, base-rate, and persistence nulls. Skill = 1 is perfect, 0 is chance, negative is worse than chance.

The binding check refuses to emit decomposition when the reconstructed prior trace disagrees with the filter's recorded loss (max allowed delta: 1e-9 for reference, 5e-3 for pymdp). This ensures the objectives surface stays synchronized with the inference code.

## Multi-Seed Robustness Needs Cluster Honesty

The `fair_sota_robustness()` function evaluates the fair-frontier robustness verdict across multiple seeds (default: 8 seeds, `range(20260616, 20260616 + 8)`):
1. Repeats the full fair-inference ladder on distinct (seeded) locked splits.
2. Reports the AIF-class-best rate across seeds (e.g., "6/8 seeds").
3. Records the handicap-is-real positive control: oracle loss < AIF loss on every seed.
4. Measures per-step bootstrap CI separation (which overstates precision due to within-traveller correlation).
5. Records distinct holdout traveller count: the real resolution limit of the fixture.

The verdict is **directional and provisional**, not a stable CI-separated performance claim. The key caveat: per-step bootstrap treats within-traveller-correlated holdout steps as i.i.d. and overstates precision. A traveller-clustered bootstrap on ~2 distinct holdout travellers is degenerate (cannot estimate between-traveller variance), so no valid uncertainty interval exists. The fixture has as few as 1–2 distinct holdout travellers depending on seed, so the result is best-among-tested-predictors on this fixture, not an established state-of-the-art. Cross-vendor audit is treated as a confound-finding constraint, not as confirmation of this fixture-bound result.

## Related Modules Own the Implementation

- `pomdp.py` — POMDP tensor construction, modality selection, control definitions.
- `fair_inference.py` — Fair-comparison predictor ladder, emission-noise sweeps, multi-seed robustness.
- `aif_sweep.py` — POMDP sweep on locked split, oracle-stripped fair-comparison, grid scoring.
- `objectives.py` — Calibration, skill scoring, base-rate and persistence nulls, per-modality decomposition.
- `sota_benchmark.py` — Locked traveller split, transparent baseline ladder (base-rate, persistence, Markov, return-boost), false-claim oracle probes.
