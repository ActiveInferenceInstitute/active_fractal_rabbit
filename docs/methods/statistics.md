# Statistics: Uncertainty Diagnostics Stay Cluster-Aware {#sec:statistics}

## Overview: Intervals Are Descriptive, Not External Validity

The statistical toolkit decomposes uncertainty from two sources: within-generator-draw uncertainty, captured by resampling-based bootstrap intervals, and between-generator-draw variance, captured by sweeping multiple random-seed draws and assessing robustness across them. Directional model comparisons are paired predictive-score comparisons: the same held-out synthetic reports are scored by two predictors, then the per-report loss differential is summarized. This follows the forecast-comparison intuition behind paired predictive-accuracy work while retaining this project's own traveller-clustered bootstrap rather than adding a Diebold-Mariano test [@diebold1995predictive_accuracy].

Calibration is treated as part of the scoring surface rather than as a decorative diagnostic. Brier score supplies a proper multiclass probability score, expected calibration error (ECE) supplies a binning-sensitive descriptive check, and the prose reads both in the probabilistic-forecasting frame of calibration plus sharpness [@brier1950forecasts; @gneiting2007scoring; @gneiting2014probabilistic_forecasting]. All intervals and p-values are software measurements on pinned synthetic artifacts. They do not estimate real-world mobility performance, population effects, or deployment uncertainty.

## Per-Step Bootstrap Intervals Overstate Precision

The row-level percentile bootstrap, implemented in `sota_benchmark.py:_bootstrap_loss_ci`, resamples *individual steps* with replacement across `n_boot` iterations, computes the mean loss per bootstrap sample, and reports the 2.5th and 97.5th percentiles as the lower and upper bounds. It remains useful as a compact leaderboard diagnostic, especially for single-row artifacts that need a deterministic visual interval, but it is not the claim-bearing interval for active-inference-versus-fair directional results.

**Why it overstates precision:** Within-traveller steps are correlated. A traveller's sequence of cells tends to persist or return to a small set of locations; consecutive predictions on the same traveller are not independent. A per-step bootstrap treats each step as an exchangeable unit, which is false when steps belong to the same traveller. This causes the resampled-mean distribution to be artificially tight, so the resulting CI underestimates true uncertainty.

Any row-level CI-separation display is therefore a screening or visualization device. Claim prose must defer to the family-level clustered-bootstrap and multiplicity artifacts when it says whether a directional active-inference comparison cleared. This is why the manuscript treats the SOTA-program leaderboard as a locked-split software comparison while reading the multiplicity ledger and regime-gate artifacts as the claim boundary.

## Traveller-Clustered Bootstrap Uses the Correct Unit

The correct resampling unit is the traveller cluster. Clustered-data bootstrap literature supports resampling the dependent unit rather than the repeated rows inside it [@field2007clustered_bootstrap; @cameron2015cluster]. In this project, `power_analysis.py:traveller_clustered_bootstrap_ci` resamples *whole travellers* (all their held-out steps) with replacement. The procedure:

1. Partition the held-out steps by traveller ID.
2. For each bootstrap iteration, draw traveller IDs with replacement (a resample of size equal to the number of unique travellers).
3. Pool the held-out steps for the drawn travellers and compute the mean paired difference (soft Markov loss minus AIF loss, in the power-analysis context).
4. Estimate the 95% percentile CI as the 2.5th and 97.5th percentiles of the bootstrap means across 2000 iterations (default).

This is the honest uncertainty summary because steps within a traveller are correlated; resampling the cluster as a unit preserves the observed within-traveller dependence instead of pretending every step is exchangeable.

The paired interval and the one-sided achieved-significance p-value are read from the same bootstrap distribution, so the interval and p-value cannot tell incompatible stories. The p-value is descriptive: it is used to bound synthetic comparison-family overreading, not to infer a population parameter.

**Degeneracy limit:** With very few travellers in the holdout set, the bootstrap is degenerate. A resample of size 2 from {traveller 1, traveller 2} has only 4 possible outcomes, and the resulting bootstrap distribution cannot capture the true between-cluster variance well. `power_analysis.py` raises a `ValueError` if fewer than 2 distinct travellers are present. The power-analysis pipeline withholds within-draw CI point estimates when clusters are too few and instead reports a seed-robust verdict: *does the effect separate from zero in every generator draw where the interval is available?* This uses across-draw robustness as a stricter descriptive filter, not as a substitute for enough independent clusters.

## Across-Generator Draws Estimate Robustness, Not Population Variance

A single bootstrap resample, even with the correct traveller-clustered unit, captures only within-draw (within-generator realization) uncertainty. Two different random seeds that control the synthetic data generation produce different sequences, and the two estimates of the treatment effect (AIF advantage) will differ. This between-draw variance is not visible to the bootstrap.

The power-analysis regime map (`power_analysis.py:fair_power_regime_map`) sweeps multiple generator seeds (default: `DEFAULT_GEN_SEEDS = (2026, 2027, 2028, 2029, 2030)`, five draws). For each cell of the 2D traveller-count × observation-noise grid:

1. Generate `len(gen_seeds)` independent synthetic sequences (same config, different random seed).
2. Run the traveller-clustered bootstrap AIF-vs-soft comparison on each sequence.
3. Collect the mean advantage and CI bounds per draw.
4. Declare a cell a *seed-robust win* if the clustered CI separates from zero in **every** draw (lines 213–223).

Across-seed statistics are also reported: mean advantage, SD of advantage across seeds, and minimum advantage. These summarize how stable the regime claim is to generator variability. A regime that wins in 4/5 draws is less robust than one that wins in 5/5; a regime with high SD across seeds is noisier than one with low SD.

The interpretation: A seed-robust win is credible for a small-magnitude regime characterization (the fair-frontier analysis); a win in only some seeds suggests the advantage is regime-dependent or marginal. The 2D regime map reports these outcomes per cell and derives high-level verdicts (lines 236–272) about where the AIF advantage emerges and when it vanishes—conditioned on the traveller count and emission noise.

## Brier Score Measures Multiclass Calibration

The Brier score is the mean squared error between the predicted probability vector and a one-hot true-cell vector. For cell recovery on state space $S$, it is:

$$
\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} \sum_{s=1}^{S} (p_s^{(i)} - \mathbb{1}[s = c_i])^2
$$ {#eq:brier-score}

where $p_s^{(i)}$ is the predicted probability of cell $s$ at step $i$ and $c_i$ is the true cell. Values range from 0 (perfect) to 2 (maximally wrong). The uniform-random floor is $(S-1)/S$; the empirical base-rate (marginal-frequency predictor) floor is $1 - \sum_s f_s^2$ where $f_s$ is the observed cell frequency.

In `objectives.py:recovery_calibration`, the Brier score is computed as `float(np.mean(np.sum((posterior - one_hot) ** 2, axis=1)))`. It is reported for every inference lane as a secondary metric complementing predictive loss. A model with low loss but high Brier is miscalibrated, overconfident or underconfident; the SOTA gate fails a candidate if its Brier exceeds the best baseline's Brier, even if it wins on loss. This prevents "single-metric wins" that trade calibration for loss.

The Brier score is particularly useful because it does not require choosing a binning scheme (unlike ECE) and is directly interpretable as the mean squared distance from the true class.

## Expected Calibration Error Remains Binning-Sensitive

Expected calibration error (ECE) measures the weighted absolute difference between per-bin confidence and per-bin accuracy:

$$
\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} \left| \text{acc}_b - \text{conf}_b \right|
$$ {#eq:expected-calibration-error}

where $B_b$ is the set of predictions with confidence in bin $b$, $\text{acc}_b$ is the top-1 accuracy in that bin, and $\text{conf}_b$ is the mean model confidence in that bin. Ranges from 0 (perfectly calibrated) to 1 (maximally miscalibrated).

In `sota_benchmark.py:_ece`, the ECE is computed with a fixed 10-bin discretization (`n_calib_bins`, default 10). This is the benchmark-gate ECE used for the SOTA calibration sub-gate.

In `objectives.py:recovery_calibration`, ECE is computed with a configurable bin count (default 5 via `DEFAULT_ECE_BINS`). The function also reports per-bin counts and statistics, so sparse bins are visible: a bin with only 1 or 2 samples carries high variance and should not be trusted as a calibration signal.

**Critical limitation — binning sensitivity:** ECE is sensitive to the choice of bin count and boundaries. With 10 bins, rare high-confidence predictions are grouped; with 5 bins, the granularity is coarser. A model that is well-calibrated under 5-bin ECE may appear miscalibrated under 10-bin ECE, and vice versa. ECE is best understood as a **descriptive diagnostic** — useful for spotting gross miscalibration (e.g., a lane that predicts 90% confidence but is right only 50% of the time) — rather than a definitive metric for model comparison. The per-bin counts in `objectives.py` output make this limitation visible: if the highest-confidence bin has only 2 examples, the bin-level accuracy is noisy and should be read skeptically.

## Calibration Baselines Keep Scores Interpretable

Beyond the uniform floor, `objectives.py` computes two additional baselines for skill scoring:

- **Base-rate (marginal-frequency) Brier** (`base_rate_brier`, lines 131–152): The Brier score of a predictor that ignores observations and predicts the empirical cell-frequency vector at every step. This is tighter than the uniform floor whenever cells are not uniformly occupied. Skill against this floor (lines 273–276) answers: "How much does observing anything beat already knowing the base rate?"

- **Persistence Brier** (`persistence_brier`, lines 155–192): The Brier score of a predictor that predicts the previous report's true cell (one-hot) at each step, or the base rate at the first report of each traveller. For mobility data with high autocorrelation (travellers tend to stay or return), persistence is a far stronger null than base rate. Skill against persistence (lines 281–287) answers: "How much do the observations add beyond 'assume no movement'?"

Both are reported in `lane_objectives` and write to the calibration artifact. A lane's Brier skill against persistence is the honest measure of observation utility for highly autocorrelated data.

## Multiple-Comparison Caution Blocks Post-Hoc Sweep Claims

The power-analysis regime map is a 2D sweep (traveller count × observation noise × generator seed). With $n_t$ traveller levels, $n_s$ smoothing levels, and $n_g$ generator seeds, the total number of comparison cells is $n_t \times n_s \times n_g$. Default: 4 × 3 × 5 = 60 cells.

A region in a 2D parameter space will show apparent effects by random chance if multiple cells are compared post-hoc. The power analysis addresses this by:

1. **Pre-specifying three hypothesis-like verdicts** (lines 236–243) derived from the regime structure, not selected after inspection: "AIF wins few travellers clean," "advantage vanishes at scale clean," "advantage persists under noise at scale."
2. **Using seed robustness as a filter**: A cell is claimed as a win only if the CI separates in *all* seeds, not just on average. This is a stricter gate than a single hypothesis test on the pooled advantage.
3. **Reporting gradients, not single cells**: The claim about noise restoring the advantage at scale (line 242) is about the *monotone gradient* of advantage across smoothing levels, not any single cell. A single non-monotone cell does not falsify the gradient claim.
4. **Stating the map as discriminative, not optimized**: The map is valid only if it contains both wins and non-wins (line 286, `map_is_discriminative`). If every cell were a win, the 2D structure would be vacuous. If the map is discriminative, the contrast itself (where the advantage appears and disappears) is the evidence, not any single cell's p-value.

In summary: The power analysis is a **regime characterization**, not a hypothesis-test battery. The multiple-comparisons structure is acknowledged in the verdict prose with an explicit reminder that the gradient is the claim, not any single cell, and that the synthetic-only boundary remains in place.

The manuscript-level comparison family is stricter still: every AIF-versus-fair row that reports a one-sided clustered-bootstrap p-value enters the generated multiplicity ledger, where Holm-Bonferroni is the primary family-wise correction and Benjamini-Hochberg is a secondary false-discovery sensitivity report. These corrections do not upgrade the measurements into population inference. They only prevent a synthetic benchmark with many chances to win from presenting isolated raw p-values as results.

## Data and Artifacts Carry the Statistical Surface

The three modules work together:

- `sota_benchmark.py` uses `_bootstrap_loss_ci` to gate the SOTA leaderboard on per-step CI separation. The candidate must beat the strongest baseline's loss CI lower edge to pass the CI-separation sub-gate.
- `power_analysis.py` uses `traveller_clustered_bootstrap_ci` to interval the per-regime-cell AIF-vs-soft advantage. The result is reported per draw and summarized into seed-robust verdicts.
- `multiplicity_ledger.py` and the regime-search artifacts enumerate the AIF-versus-fair comparison family, keep uncorrected p-values paired with Holm-adjusted p-values, and require intervals as well as p-values before a clearance is readable.
- `objectives.py` uses `recovery_calibration` to compute Brier, ECE, and skill scores for every inference lane. These are secondary metrics that catch miscalibration and are recorded in the calibration artifact.

The corresponding generated artifacts are `output/runs/sota_benchmark_matrix.json`,
`output/runs/power_analysis.json`, `output/runs/multiplicity_ledger.json`, and
the default-run calibration/metrics artifacts consumed by the figure layer and
manuscript claim-audit layer.

## Related Docs Share the Same Statistical Boundary

- Architecture and pipeline structure: see [`architecture.md`](../architecture.md).
- Claim-ledger mappings: see [`claim_ledger.md`](../claim_ledger.md).
- Validation gates: see [`validation.md`](../validation.md).
- Empirical extension protocol: see [`empirical_validation_protocol.md`](../empirical_validation_protocol.md).
