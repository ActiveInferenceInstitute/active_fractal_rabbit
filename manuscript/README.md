# Manuscript Rendering and Cross-Reference Contract

This manuscript is rendered through a sibling checkout of the research template
repository, with this project linked as `working/active_fractal_rabbit`; from
the template repo root run
`uv run python scripts/03_render_pdf.py --project working/active_fractal_rabbit`
(after `uv run python -m infrastructure.orchestration link-projects`).
Do not hand-author generated numeric claims; add them to
`src/active_fractal_rabbit/manuscript_variables.py`.

Equations, figures, and tables use Pandoc-crossref labels such as
`{#eq:belief-update}`, `{#fig:policy-proxy}`, and `{#tbl:candidate-scores}`.
Manuscript prose should refer to `[@eq:...]`, `[@fig:...]`, and `[@tbl:...]`
rather than hardcoded numbers.
