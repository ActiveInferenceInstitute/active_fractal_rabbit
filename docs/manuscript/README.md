# Manuscript Rendering and Cross-Reference Contract

The numbered sections, supplements, and bibliography in this directory are the
publication source. Generate the fixture study, figures, and cover using the
[run guide](../guides/quickstart.md) before hydration.

Provision a separate renderer checkout at revision
`87704feae0a4490ef7dad9c931d268e8c30273c2`. Install Pandoc, Pandoc-crossref,
and XeLaTeX, then run from this repository root:

```bash
project_root="$(pwd -P)"
export TEMPLATE_REPO_ROOT=/path/to/template
git -C "$TEMPLATE_REPO_ROOT" checkout --detach 87704feae0a4490ef7dad9c931d268e8c30273c2
uv run python scripts/z_generate_manuscript_variables.py
cd "$TEMPLATE_REPO_ROOT"
uv sync --locked
mkdir -p projects/working
ln -s "$project_root" projects/working/active_fractal_rabbit
TEMPLATE_SKIP_LINK_SYNC=1 uv run --no-sync python scripts/pipeline/stage_03_render.py \
  --project working/active_fractal_rabbit --skip-manuscript-hydration
TEMPLATE_SKIP_LINK_SYNC=1 uv run --no-sync python scripts/pipeline/stage_04_validate.py \
  --project working/active_fractal_rabbit
```

The explicit hydration step writes the generated `output/manuscript/` tree.
Render and validation results are written under `output/` in the project
checkout. Check the combined PDF, the HTML edition, and the validation report
before finalizing a publication package.

Do not hand-author generated numeric claims; add them to
`src/active_fractal_rabbit/manuscript_variables.py`.

Equations, figures, and tables use Pandoc-crossref labels such as
`{#eq:belief-update}`, `{#fig:policy-proxy}`, and `{#tbl:candidate-scores}`.
Manuscript prose should refer to `[@eq:...]`, `[@fig:...]`, and `[@tbl:...]`
rather than hardcoded numbers.
