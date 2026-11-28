# Generative Model Card: Machine-Checkable POMDP Boundaries {#sec:model-card}

Machine-checkable specification of the active-inference generative model, emitted by
`src/active_fractal_rabbit/model_spec.py` and verified by
`tests/test_model_spec.py` (a de-normalized tensor, stale cardinality, or drifted
dependency is caught). Model structure is declared in the Generalized Notation
Notation (GNN) specs under `models/` and held equal to the in-code declarations by
`tests/test_gnn_spec.py`. This card describes the current default 4x4 fixture surface:
sixteen spatial cells, four time bins, four speed categories, and two burst
categories. Synthetic-only; no real-mobility claim.

## GNN Specs Declare the Model Structure

The committed GNN specs under `models/` are the single source of truth for model
*structure*: hidden factors and their cardinalities, observation modalities and
their cardinalities, the `A`/`B` factor-dependency structure, the control
cardinalities, and whether the conditional-independence approximation applies.
`src/active_fractal_rabbit/gnn_spec.py` parses those files into the same
`GenerativeModelSpec` dataclass used below, and `tests/test_gnn_spec.py` holds the
GNN-derived declarations equal to `single_factor_spec(...)` and
`factorial_spec(...)` field by field, so editing one side without the other fails
the suite. When the pinned upstream GNN checkout (sibling
`Generalized_Notation_Notation`, or `$GNN_CHECKOUT`) is present, the same suite
re-parses both files with the upstream
`gnn.parsers.markdown_parser.MarkdownGNNParser` and compares the parsed factor,
observation, and control cardinalities plus the `A`/`B` dependency edges and the
conditional-independence flag against the in-code declarations; it skips with an
explicit reason when the checkout or its virtualenv is absent.

The `A`/`B`/`D` *values* are learned from declared training rows by the model
builders, so the specs do not declare fitted constants. Each spec's
`## InitialParameterization` block states the exact initialization defaults the
builders start from (the smoothing floor plus the uniform or degenerate slices),
and learning overrides them. The upstream revision the specs are validated
against is pinned in `.github/gnn-pin.json` and `models/README.md`; the pin is
GNN v5.1.0 (`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`), advanced from v5.0.3
(`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`) when the shared reader was adopted.

## `single_factor_reference` Defines the Baseline POMDP

- **Hidden factors:** `cell`
- **Observation modalities:** `location`, `time_of_day`, `speed`, `report_burst`
- **A (emission) dependencies:** ((0,), (0,), (0,), (0,))
- **B (transition) dependencies:** ((0,),)
- **Control cardinalities:** (2,)
- **Conditional-independence approximation:** yes
- **Estimation contract:** A/B/D use declared training rows only; evaluation-holdout rows do not update tensors.
- **Tuning contract:** Precision/return selection follows inner-train transition fitting; adaptation is a second predeclared inner-validation stage.

| modality | cardinality | depends on factors |
| --- | --- | --- |
| `location` | 16 | cell |
| `time_of_day` | 4 | cell |
| `speed` | 4 | cell |
| `report_burst` | 2 | cell |

Single hidden factor (spatial cell). Each A_m is column-stochastic of shape (obs_size, num_states); the joint likelihood P(o|s) = prod_m A_m[o_m, s] uses the conditional-independence approximation across modalities. B is (S, S, n_controls), each control slice column-stochastic; D is a length-S simplex.

## `aif_regime_search` Bounds Regime-Qualified Emission Wins

The regime-qualified AIF search keeps the same latent-cell prediction target but
varies the observation channel to test when recursive belief marginalization has
measurable value over point estimates and implemented fair baselines. Its
canonical artifacts are `output/runs/aif_regime_search.csv`,
`output/runs/aif_regime_search.json`, and
`output/runs/aif_regime_sota_gate.json`.

The configured regime families are:

| regime family | observation contract | purpose |
| --- | --- | --- |
| identity-emission collapse | deterministic observation of the latent cell | negative control; AIF advantage must not be promoted when the handicap disappears |
| symmetric noisy emission | known column-stochastic noisy cell channel with strict positive floor | positive stressor for belief marginalization under diffuse observations |
| ambiguous/coarsened emission | known many-to-one cell-pair observation channel with strict positive floor | tests whether soft posterior mass over indistinguishable cells is useful |
| burst reporting gap | deterministic sparse-report thinning with the same locked split | tests whether missing reports change candidate robustness without changing fairness |
| latent regime switching | synthetic transition-regime stressor with the same scoring contract | tests whether transition nonstationarity changes the AIF-vs-baseline ordering |

Every eligible candidate receives the same observation-channel contract: the
emission matrix is either known to all candidates or must be learned from
training travellers only for all candidates in the row. Transitions are learned
from the training split only, traveller splits are locked and disjoint, and all
reported log scores use strictly positive smoothing so posterior updates cannot
win by zero-probability artifacts. The gate is local to a regime ID and
realization seed set: it can support only regime-qualified synthetic wording
against the implemented fair baseline roster, never global SOTA, external SOTA,
empirical transfer, or operational readiness.

## Online Transition Adaptation Extends the Scored Candidate

The `precision_return_aif` candidate now has three train-only selected components:
belief precision, a visited-cell return prior, and an online transition adaptation
weight. The adaptive transition is estimated only from the candidate's past noisy
observations. Those observations are converted to emission posteriors, consecutive
posterior outer products provide fractional transition counts over a bounded recent
window, and the resulting column-stochastic transition is shrunk toward the transition
learned from inner-training travellers. Setting the adaptation weight to zero exactly
recovers the fixed-transition candidate.

Hyperparameter selection uses a genuinely nested, staged contract: the transition used
during tuning is re-estimated without the inner-validation travellers; precision and
return weight are selected with fixed transitions, then adaptation is tuned around that
selected pair; and the outer evaluation holdout remains untouched.
`output/runs/adaptive_aif_ablation.json` compares separately tuned fixed and adaptive
candidates across repeated latent-regime-switching realizations, with traveller-clustered
paired intervals against both fixed AIF and the predeclared online-base-rate comparator.
The adaptive candidate improves fixed AIF consistently but does not clear that fair comparator.
It is therefore a scored, repeated generative-model improvement, not a promotion result.

## `factorial_latent_mode` Separates Movement and Reporting Modes

- **Hidden factors:** `cell`, `movement_mode`, `cognitive_proxy_mode`
- **Observation modalities:** `location`, `time_of_day`, `speed`, `report_burst`, `cell_delta`, `return_flag`
- **A (emission) dependencies:** ((0,), (0, 2), (1, 0), (2,), (1,), (1,))
- **B (transition) dependencies:** ((0, 1), (1, 2), (2,))
- **Control cardinalities:** (2, 2, 2)
- **Conditional-independence approximation:** no (coupled factors)
- **Estimation contract:** Factorial tensors use declared included rows and remain diagnostic rather than a locked predictive candidate.
- **Tuning contract:** No predictive hyperparameter promotion; tensor and mean-field diagnostics only.

| modality | cardinality | depends on factors |
| --- | --- | --- |
| `location` | 16 | cell |
| `time_of_day` | 4 | cell, cognitive_proxy_mode |
| `speed` | 4 | movement_mode, cell |
| `report_burst` | 2 | cognitive_proxy_mode |
| `cell_delta` | 2 | movement_mode |
| `return_flag` | 2 | movement_mode |

Three hidden factors. A_dependencies couple factors (e.g. time_of_day depends on cell and cognitive-proxy), so the exact posterior is non-factorized and mean-field inference is an approximation reported with its gap. Each A_m is column-stochastic over its observation axis; each B factor tensor is column-stochastic over its target axis; each D factor is a simplex. The cognitive-proxy factor is a synthetic reporting-process state, not real cognition or intent.
