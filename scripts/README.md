# scripts/

Thin command-line entrypoints for Active FractalRabbit. Each script parses arguments,
resolves repository-relative paths, and calls into the `active_fractal_rabbit` package;
domain logic lives under [`../src/`](../src/README.md). Agent rules are in
[`AGENTS.md`](AGENTS.md).

## Common entry points

| Script | Purpose |
| --- | --- |
| `00_preflight.py` | Environment and fixture preflight. |
| `01_prepare_fractalrabbit.py` | Prepare the optional pinned external simulator. |
| `02_run_pipeline.py` | Run the fixture (or external) pipeline for a config. |
| `03_run_sweep.py` | Sensitivity sweep. |
| `04_generate_figures.py` | Render figures for a run. |
| `05_run_negative_controls.py` | Seeded negative controls. |
| `07_run_full_study.py` | One-command full study across both lanes. |
| `11_check_manuscript_variable_provenance.py` | Fail-loud manuscript-number provenance check. |
| `18_run_sota_program.py` | SOTA benchmark program. |
| `20_run_aif_regime_search.py` | Regime-qualified active-inference search. |
| `29_finalize_publication_package.py` | Finalize/verify the publication package (template render). |
| `z_generate_manuscript_variables.py` | Regenerate manuscript variables. |

Focused checks (`10`–`16`) validate the evaluation contract, claim ledger,
cover art, visual quality, claim audit, and visual-claim audit.

## Running

```bash
uv run python scripts/00_preflight.py
uv run python scripts/07_run_full_study.py --lane both
./run.sh test-full
```

Gate scripts:

| Script | Purpose |
| --- | --- |
| `check_docs_links.py` | Relative-link gate for every Markdown file (`./run.sh docs`). |
| `check_gnn_conformance.py` | Pinned GNN strict validation and parser tests (`./run.sh gnn`). |
| `gate_report.py` | Measures every arm and refreshes `docs/_generated/gate-timings.md` (`./run.sh bench`). |

Outputs are written under `output/runs/` and `output/figures/`; treat `output/runs`
as shared mutable state and run output-mutating jobs serially.
