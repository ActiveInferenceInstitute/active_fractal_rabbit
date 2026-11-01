# GNN model specifications

This directory holds the Generalized Notation Notation (GNN) specifications for
the active-inference generative models declared by this repository. The GNN
files are the **single source of truth for model structure**: hidden factors
and their cardinalities, observation modalities and their cardinalities, the
`A` (likelihood) and `B` (transition) factor-dependency structure, the control
cardinalities, and whether the conditional-independence approximation applies.

## Files

| File | Model | Scope |
| --- | --- | --- |
| `single_factor_reference.gnn.md` | `single_factor_reference` | One hidden factor (`cell`, 16 states) and four modalities (`location` 16, `time_of_day` 4, `speed` 4, `report_burst` 2); every `A` depends on `cell`; `B` on `cell`; 2 controls. |
| `factorial_latent_mode.gnn.md` | `factorial_latent_mode` | Three hidden factors (`cell` 16, `movement_mode` 3, `cognitive_proxy_mode` 3) and six modalities (`location` 16, `time_of_day` 4, `speed` 4, `report_burst` 2, `cell_delta` 2, `return_flag` 2); coupled `A`/`B` dependencies; controls (2, 2, 2). |

Both files describe the default 4x4 fixture surface (16 spatial cells, 4 time
bins, 4 speed categories, 2 burst categories).

## Tensors are learned, not declared

The `A`/`B`/`D` tensors are estimated from declared training rows by the
in-repo model builders (`src/active_fractal_rabbit/pomdp.py` and
`src/active_fractal_rabbit/factorial_pymdp.py`). They are not fixed numeric
constants, so these specs intentionally carry **no** `## InitialParameterization`
block. Where a normalization contract must be stated, the declaration uses a
`default=` hint (for example `D_cell[16,type=float,default=uniform]`). The
`## ModelAnnotation` section records the estimation and tuning contracts, and
the `## ModelParameters` section records `model_id` and
`conditional_independence`.

## Loader and drift guard

`src/active_fractal_rabbit/gnn_spec.py` parses these files into the
`GenerativeModelSpec` dataclass from `model_spec.py`. `gnn_spec.load_specs()`
returns the parsed specs; `single_factor_spec()` and `factorial_spec()` keep
working unchanged for existing callers. `tests/test_gnn_spec.py` asserts the
GNN-derived specs equal the in-code declarations field by field, so editing one
side without the other fails the test suite. When the pinned upstream checkout
is available, the same suite also re-parses each file with the upstream
`gnn.parsers.markdown_parser.MarkdownGNNParser` and compares the parsed factor,
observation, and control cardinalities plus the `A`/`B` dependency edges against
the in-code declarations; it skips with an explicit reason when the checkout or
its virtualenv is absent.

## Upstream pin

- Upstream repository:
  `https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation.git`
- Pinned tag: `v5.0.3`
- Pinned revision: `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1` (the commit the
  `v5.0.3` tag dereferences to)
- Upstream version: `5.0.3`
- Pinned on: `2026-11-01`

The machine-readable copy of this pin is `.github/gnn-pin.json`; when the
checkout is present, `tests/test_gnn_spec.py` re-verifies that the recorded tag
still dereferences to the recorded revision and that the revision is present.
When the checkout stands at that revision it also compares the parser source in
use against the pinned source byte for byte; a checkout parked on another
revision skips that comparison with a reason naming the observed `HEAD`, since a
moved checkout cannot prove which parser its virtualenv imports. The syntax
authority is `docs/gnn/gnn_syntax.md` in that checkout. Re-validation runs the upstream validator against these files:

```bash
cd <Generalized_Notation_Notation checkout>
.venv/bin/python -m gnn.cli validate <active_fractal_rabbit>/models/single_factor_reference.gnn.md --strict
.venv/bin/python -m gnn.cli validate <active_fractal_rabbit>/models/factorial_latent_mode.gnn.md --strict
```

`tests/test_gnn_spec.py` runs those commands when the checkout is present
(sibling directory `Generalized_Notation_Notation`, or `$GNN_CHECKOUT`) and
skips cleanly with an explicit reason when it is absent.

## Claim boundary

Synthetic-only. These specs describe the structure of models scored on the
fixture-first synthetic FractalRabbit lane; they are not evidence of
real-world behavioral validation, and the factorial latent-mode model remains a
diagnostic surface rather than a locked predictive candidate. The
cognitive-proxy factor is a synthetic reporting-process state, not real
cognition or intent.
