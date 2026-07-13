# Supplementary S5 - Closed-Loop Ranking Remains a Non-Controller {#sec:closed-loop}

This supplement describes the closed-loop candidate-ranking heuristic, the recovery-versus-sparsity lane, and the additive holdout-scoring protocol — secondary experimental lanes extending the main-line evaluation.

## Candidate Ranking Is a Heuristic, Not a Controller {#sec:closed-loop-candidate-ranking}

The closed loop is an experiment-level feedback loop: simulate or load waypoints, infer latent-state beliefs, score the run, then rank declared candidate simulator configurations. It is not an online controller acting on a real mobility system.

For a configured objective $J$ and candidate $k$, the selector records:

$$
\mathrm{score}_k
= J(\mathrm{run}) \cdot
\frac{r_{\mathrm{current}}}{r_k},
$$ {#eq:candidate-score}

when the objective direction is minimization. Here $r$ is the configured mean report count. [@eq:candidate-score] is a transparent ranking heuristic for comparing declared simulator candidates under the current objective: denser reporting is favored when the measured inference objective is high. Every candidate score is written to `candidate_scores.json` and `candidate_scores.csv`. It is not presented as an optimal experimental-design rule; formal active-inference treatments connect expected free energy to Bayesian optimal design and expected utility under assumptions that this scalar fixture diagnostic does not satisfy [@sajid2021active_inference_design].

The current default ranking is:

{{CANDIDATE_SCORE_TABLE}}

: Candidate scores emitted by the default closed-loop run. {#tbl:candidate-scores}

The selected next candidate is `{{SELECTED_NEXT_CANDIDATE}}`, with score {{BEST_CANDIDATE_SCORE}} and reporting ratio {{BEST_CANDIDATE_REPORTING_RATIO}}. Because [@tbl:candidate-scores] is generated from run artifacts, it changes when the objective, candidate set, or simulator parameters change.

## Sporadic-Observation Fit Stays Synthetic {#sec:closed-loop-mission-fit}

The modeled situation deserves explicit statement. FractalRabbit was published
by NSA Research as a sparse-mobility synthetic generator; this project uses it
only as a pinned software benchmark. Its traces are observed through sporadic,
irregular reports, where most of a trajectory is never seen
[@darling2018fractalrabbit; @nsa2026fractalrabbit]. The analytic question that
situation poses — how much latent movement structure is
recoverable from sparse reports, with what calibrated uncertainty, and which
next data-collection configuration would be most informative — is exactly the
shape of problem a discrete active-inference agent addresses: state estimation
under partial observability, uncertainty quantified per step as posterior
entropy and variational free energy, and candidate evaluation through expected
free energy [@dacosta2020discrete_active_inference; @sajid2021active_inference_design].
The ground-truth recovery evaluation in [@sec:results] makes this concrete: it
measures, against the simulator's own latent structure, how much of that
structure each configured model actually recovers. Two boundaries hold
throughout. The evaluation is synthetic-only software evidence, and the same
recoverability that makes sparse analytics tractable is what makes coarse
spatiotemporal data re-identifiable — which is why the privacy threat model of
[@sec:validation] and the separate empirical protocol are load-bearing parts of
this project rather than disclaimers [@demontjoye2013unique; @kondor2018trace_matching].

## Recovery Is Measured Across the Sparsity Lane {#sec:closed-loop-sparsity-lane}

A dedicated sparsity lane turns the question into a quantitative check. Each
traveller's reports are deterministically subsampled (seeded by the root
`seed`) to a configured fraction of the full set; at every level the full and
deep-latent models are re-estimated from scratch and scored against the exact
ground-truth cell, and the pymdp lane validates the released state updater on
deterministic sentinel rows at each sparsity level rather than running a full
deep-latent agent pass. The result is a recovery-versus-sparsity curve rather
than a single operating point — and the curve's direction is read from the
artifact, not assumed: under this deterministic subsampling the deep-latent
recovery mass does not fall monotonically as reports thin (densest-level mean
mass on the true cell {{SPARSITY_DENSEST_LATENT_MASS}} against
{{SPARSITY_SPARSEST_LATENT_MASS}} at the sparsest level, which currently
scores highest), so the lane measures how sparsity reprices the available
evidence rather than demonstrating degradation.

## Holdout Scoring Remains Additive Under Fixed Discretization {#sec:closed-loop-holdout}

The experiment layer also supports holdout scoring as an additional pass. When `experiment.holdout_fraction` is positive, the last configured fraction of each traveller's chronologically ordered waypoints is held out (each traveller always keeps at least two training rows), the `A`/`B`/`D` tensors are re-estimated from the training rows only, and the filter is re-run over the full ordered sequence while aggregating predictive loss and posterior entropy only at holdout indices. The primary full-data metrics are never replaced by this pass; the holdout keys are additive. Discretization bin edges remain shared with the full lane, so the holdout numbers measure out-of-sample tensor estimation under a fixed discretization rather than a fully leakage-free protocol; this boundary is restated in [@sec:limitations].
