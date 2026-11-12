# Run Guide: One Gate, Separate Arms {#sec:run-guide}

`./run.sh` is the single gate entry point. Every arm is one command, CI calls
the same arm instead of repeating the command, and each arm can run alone.
Measured wall times live in the generated
[`_generated/gate-timings.md`](_generated/gate-timings.md); do not copy them
into prose.

## Arms

| Arm | Command | What it runs | CI step |
| --- | --- | --- | --- |
| `lint` | `./run.sh lint` | `ruff check .` with the pinned dev dependency | `gate` → Lint |
| `test` | `./run.sh test` | Fast suite: `pytest tests/ -m "not requires_fractalrabbit and not slow" --no-cov -q` | `gate` → Fast tests |
| `test-full` | `./run.sh test-full` | Documented gate: `pytest tests/ --cov=src --cov-fail-under=90` | `gate` → Full tests with the coverage floor |
| `docs` | `./run.sh docs` | Relative-link check plus the generated-timings contract | `gate` → Documentation links and generated timing contract |
| `gnn` | `./run.sh gnn` | Pinned GNN strict validation and parser tests, zero skips | `gnn-conformance` → Run the pinned GNN conformance gate |
| `all` | `./run.sh all` | `lint`, `test`, `test-full`, `docs`, `gnn` in order | — |
| `bench` | `./run.sh bench` | Re-measures every arm and rewrites the generated timing document | — |

`./run.sh --list` prints the measurable arm names; `./run.sh --help` prints the
arm list and the absent-arm rationale.

CI uses two jobs: `gate` (display name `gate (lint, tests, docs)`) runs the
first four arms after one `uv sync --frozen --extra dev`, and `gnn-conformance`
(display name `gnn conformance (pinned)`) runs `./run.sh gnn` in its own job
because it prepares the pinned parser virtualenv.

## Fast arm versus full arm

`test` and `test-full` share one test tree. The fast arm drops coverage
instrumentation and excludes two markers:

- `requires_fractalrabbit` — the external-simulator tests
  (`tests/test_fractalrabbit_external.py`, `tests/test_java_resolution.py`).
- `slow` — artifact-surface regeneration tests that run the full writer set.
  `tests/test_figures_and_variables.py::test_figures_and_manuscript_variables_are_generated`
  is the current member.

The full arm runs every test, including both marker groups, and keeps the 90%
coverage floor; it is the arm to run before landing.

## Arms that are deliberately absent

- **`types`.** This repository configures no type checker: there is no mypy or
  pyright configuration or dependency in `pyproject.toml`. Adding a `types` arm
  would introduce a new gate, not expose an existing one.
- **`format`.** `ruff format --check` currently reports files that would be
  reformatted, and formatting is not part of the documented gate. `lint` is
  `ruff check` only, so the arm matches the existing code state.

## Documentation arm

`./run.sh docs` runs two checks:

1. `scripts/check_docs_links.py` resolves every relative Markdown link in the
   checkout. Links into the generated `output/` tree are skipped because that
   tree is produced by the pipeline; links that leave the checkout are
   external. The repository has no mkdocs or markdown-link-check
   configuration, so this script is the link gate.
2. `scripts/gate_report.py --check` verifies that
   [`_generated/gate-timings.md`](_generated/gate-timings.md) still lists every
   arm with a measured duration.

During `./run.sh bench`, the docs arm runs the link check but skips the
generated-report check (`GATE_REPORT_BENCH=1`), because that pass is writing
the report it would otherwise check.

## GNN conformance arm

`./run.sh gnn` reads `.github/gnn-pin.json`, requires a checkout at the pinned
revision, installs it into its own virtualenv, validates both spec files with
the upstream strict validator, and runs `tests/test_gnn_spec.py` with zero
skips. See [`.github/GNN.md`](../.github/GNN.md) for the pin, the hosted job
key and display name, and the difference between the hosted
`gnn conformance (pinned)` job and the required `gnn conformance (local)`
commit status.

## Manuscript numbers and render path

Manuscript result values are not typed into prose. `scripts/z_generate_manuscript_variables.py`
writes `output/data/manuscript_variables.json` from the generated run artifacts,
and the manuscript uses `{{TOKEN}}` placeholders that the template render
hydrates from that file. If a number has no token, add it to
`src/active_fractal_rabbit/manuscript_variables.py` rather than editing the
manuscript.

Rendering is a lifecycle step, not a gate arm, because it needs the sibling
template checkout:

```bash
export TEMPLATE_REPO_ROOT=/path/to/template
uv run python scripts/z_generate_manuscript_variables.py
cd "$TEMPLATE_REPO_ROOT"
uv run python -m infrastructure.orchestration link-projects
uv run python scripts/pipeline/stage_03_render.py --project working/active_fractal_rabbit --skip-manuscript-hydration
cd projects/working/active_fractal_rabbit
uv run python scripts/29_finalize_publication_package.py
uv run python scripts/29_finalize_publication_package.py --verify-only
```

The explicit hydration command writes `output/data/manuscript_variables.json`
and the token-resolved `output/manuscript/` tree. The render reads that tree
with `--skip-manuscript-hydration` because the template's hydration subprocess
rejects a symlinked project slot.

`scripts/04_generate_figures.py` writes `output/figures/figure_registry.json`
with both the repository's `entries` array and the template's label-keyed
`figures` array (`fig:<hyphenated-key>`, `filename`, `alt_text`). The template
reads the `figures` array for per-section HTML and PDF accessibility; do not
hand-edit either array.

## Refreshing the measured numbers

```bash
./run.sh bench
```

This runs every arm once on the current host and rewrites
[`_generated/gate-timings.md`](_generated/gate-timings.md) with the commit,
host, Python version, load average, exit code, wall time, and a short evidence
line per arm.
The numbers are single-run observations, not a benchmark; a new host or a new
dependency set needs a fresh run.
