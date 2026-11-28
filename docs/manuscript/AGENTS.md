# AGENTS.md - manuscript

- Top-level chapter files use one H1 with a stable `{#sec:...}` label; sub-part
  files (`02b`, `03a`, `03b`) deliberately start at H2 under their parent
  chapter, each with a stable `{#sec:...}` label.
- Cite with Pandoc keys from `references.bib`.
- Reference generated figures from `../output/figures/`.
- Use `[@eq:...]`, `[@fig:...]`, and `[@tbl:...]` references; do not hardcode
  equation, figure, or table numbers.
- Keep figure captions aligned with `output/figures/figure_registry.json`.
- Do not type result values directly when they can come from `{{TOKEN}}`
  manuscript variables.
- Keep every empirical claim synthetic-only unless a generated artifact proves a
  stronger scope.
- Frame AIF findings as regime-qualified; do not call them global SOTA.
