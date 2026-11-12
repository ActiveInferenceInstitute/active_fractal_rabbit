# Manuscript Rendering and Cross-Reference Contract

This manuscript is rendered through a sibling checkout of the research template
repository, with this project linked as `working/active_fractal_rabbit`. First
hydrate the variables and the token-resolved tree from this checkout:

```bash
export TEMPLATE_REPO_ROOT=/path/to/template
uv run python scripts/z_generate_manuscript_variables.py
```

Then, from the template repo root, run
`uv run python scripts/pipeline/stage_03_render.py --project working/active_fractal_rabbit --skip-manuscript-hydration`
(after `uv run python -m infrastructure.orchestration link-projects`). The
`--skip-manuscript-hydration` flag is required because the template's hydration
subprocess rejects a symlinked project slot; the explicit hydration command
above writes the `output/manuscript/` tree the render reads.
Do not hand-author generated numeric claims; add them to
`src/active_fractal_rabbit/manuscript_variables.py`.

Equations, figures, and tables use Pandoc-crossref labels such as
`{#eq:belief-update}`, `{#fig:policy-proxy}`, and `{#tbl:candidate-scores}`.
Manuscript prose should refer to `[@eq:...]`, `[@fig:...]`, and `[@tbl:...]`
rather than hardcoded numbers.
