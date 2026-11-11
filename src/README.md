# src/

Source root for the `active_fractal_rabbit` package (a `src/`-layout project). All
study, audit, release, and figure logic lives under
[`active_fractal_rabbit/`](active_fractal_rabbit/AGENTS.md); the `scripts/` wrappers
stay thin. This package owns the structural model contracts and the synthetic-only
claim boundary.

## Layout

```text
src/
└── active_fractal_rabbit/
    ├── __init__.py
    ├── config.py              # config loading and validation
    ├── fractalrabbit.py       # simulator CSV ingest and lane contracts
    ├── observations.py        # feature tables and tensor summaries
    ├── pomdp.py / inference.py# discrete POMDP filter and predictor ladder
    ├── model_spec.py / gnn_spec.py  # generative-model contract + GNN specs
    ├── manuscript_variables.py      # the only source of manuscript numbers
    ├── figures.py / figure_style.py # figure data transforms and provenance
    ├── claim_audit.py / visual_claim_audit.py / visual_quality.py
    ├── aif_regime_search.py / sota_program_harness.py
    └── release.py / render_freshness.py / variable_provenance.py
```

## Conventions

- Keep behavior here, not in `scripts/`.
- Manuscript numbers flow from `manuscript_variables.py` or generated `output/runs/`
  artifacts; never hand-author result values.
- Preserve provenance fields when transforming artifacts, and keep generated-output
  schemas stable unless tests and consumers change in the same patch.
- Do not vendor FractalRabbit, pymdp, jars, or caches into this tree.

Agent rules, the module-ownership map, and the claim boundary are in
[`active_fractal_rabbit/AGENTS.md`](active_fractal_rabbit/AGENTS.md); design context is
in [`../docs/architecture.md`](../docs/architecture.md).
