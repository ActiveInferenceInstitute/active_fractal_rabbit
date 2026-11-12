# AGENTS.md - Active Fractal Rabbit

If working from a sidecar monorepo checkout that links this project in, read
the parent `working/AGENTS.md` first. This project renders only through the
sibling `template` repository checkout (its path given by
`$TEMPLATE_REPO_ROOT`, or the sibling `template/` directory) as
`working/active_fractal_rabbit`; that render-time dependency is internal
tooling and is unrelated to this repository's own public/private status.

## PROJECT KNOWLEDGE BASE

Active Fractal Rabbit studies a fixture-first synthetic FractalRabbit waypoint
lane and optional external simulator lane. Treat all current evidence as
synthetic-only unless a future artifact explicitly proves otherwise. Do not turn
regime-qualified AIF search into global SOTA language.

## Overview

- Default lane: `data/fixtures/fractalrabbit_sample_large.csv`.
- Small contract fixture: `data/fixtures/fractalrabbit_sample.csv`.
- Optional external lane: `configs/fractalrabbit_external.yaml`, JDK 21, and
  Maven are required.
- FractalRabbit source, jars, Maven caches, and pymdp source are not vendored
  here. Use only the pinned external cache under `.cache/fractalrabbit/`.
- Every manuscript number must come from
  `src/active_fractal_rabbit/manuscript_variables.py` or generated artifacts.
- Generated outputs belong under `output/runs/`, `output/figures/`,
  `output/cover/`, `output/data/`, and `output/release_manifest.json`.
- `TODO.md` is open work only. `tasks.yaml` carries historical/task status.
- Keep scripts thin; put reusable logic under `src/active_fractal_rabbit/`.

## Structure

- `configs/` - lane configuration, including optional external simulator config.
- `data/fixtures/` - deterministic checked fixtures for no-Java development.
- `scripts/` - orchestration entrypoints; no heavy domain logic.
- `src/active_fractal_rabbit/` - simulator ingestion, analysis, figures,
  manuscript variables, release, and audit logic.
- `manuscript/` - Pandoc source that consumes variables and generated figures.
- `docs/` - lifecycle docs for fixture-first and optional external workflows.
- `output/runs/` - JSON, CSV, manifests, gates, and run directories.
- `output/figures/` - generated figures and figure registries.
- `output/cover/`, `output/data/`, `output/release_manifest.json` - generated
  cover art, variable/provenance payloads, and release manifest surfaces.

## Where To Look

| Need | Look here | Rule |
| --- | --- | --- |
| Fixture-first analysis | `data/fixtures/fractalrabbit_sample_large.csv` | Default lane; no Java required. |
| Parser regression contract | `data/fixtures/fractalrabbit_sample.csv` | Keep small and deterministic. |
| External simulator lane | `configs/fractalrabbit_external.yaml` | Optional; requires JDK 21 and Maven. |
| Manuscript numbers | `src/active_fractal_rabbit/manuscript_variables.py`, `output/data/` | Never hand-author result values. |
| Run evidence | `output/runs/` | Generated artifacts only; preserve synthetic-only boundary. |
| Figure evidence | `output/figures/`, `output/figures/figure_registry.json` | Captions must match registry/provenance. |
| Release evidence | `output/release_manifest.json` | Generated release manifest, not a prose substitute. |
| Current open work | `TODO.md` | Forward-looking work only. |
| Historical/task status | `tasks.yaml` | Do not duplicate completed history in TODO prose. |

## CODE MAP

| Path | Role | Claim boundary |
| --- | --- | --- |
| `scripts/07_run_full_study.py` | Full-study orchestration for fixture/external lanes and downstream artifacts. | Prefer fixture-first; external requires config, JDK 21, Maven. |
| `src/active_fractal_rabbit/fractalrabbit.py` | FractalRabbit CSV parsing, contracts, and lane data loading. | Do not vendor upstream simulator code or caches. |
| `models/*.gnn.md`, `src/active_fractal_rabbit/gnn_spec.py` | Generalized Notation Notation specs for the generative models and the loader that checks them against `model_spec.py`. | Structural source of truth; `A`/`B`/`D` values stay learned, not declared. |
| `src/active_fractal_rabbit/manuscript_variables.py` | Source-owned manuscript variable generation. | All manuscript numbers must flow from here or generated artifacts. |
| `src/active_fractal_rabbit/figures.py` | Figure generation and registries. | Figures must stay tied to output artifacts and synthetic scope. |
| `src/active_fractal_rabbit/release.py` | Release manifest assembly and generated release surfaces. | Local artifacts are not public publication readiness by themselves. |
| `src/active_fractal_rabbit/claim_audit.py` | Text and artifact claim-boundary checks. | Blocks unsupported real-world, global SOTA, or stale claim wording. |
| `src/active_fractal_rabbit/visual_claim_audit.py` | Visual/caption claim audit over generated figures. | Captions must not exceed artifact evidence. |
| `scripts/18_run_sota_program.py` | SOTA program runner. | Use SOTA wording only for the exact cleared synthetic lane/contract. |
| `src/active_fractal_rabbit/sota_program.py` | SOTA task/gate implementation. | No global SOTA claim without a global benchmark contract. |
| `scripts/20_run_aif_regime_search.py` | AIF regime search runner. | Regime-qualified search is local synthetic evidence, not global SOTA. |
| `src/active_fractal_rabbit/aif_regime_search.py` | Regime map/search and gate logic. | Say "regime-qualified AIF" and name the cleared regime/contract. |

## Claim Rules

- The present project boundary is synthetic-only. Do not claim real-world
  behavioral validation from waypoint fixtures or simulator runs.
- "Regime-qualified AIF" means a bounded search over named synthetic regimes.
  It is not global SOTA and must not be summarized as global dominance.
- Use SOTA wording only for the exact cleared synthetic lane, benchmark
  contract, and artifact gate. Otherwise use "candidate", "local regime", or
  "synthetic benchmark" wording.
- Terms such as `output/runs`, `output/figures`, `output/cover`, `output/data`,
  and `output/release_manifest.json` refer to generated local artifacts.

## Verification

- Green project gate: `./run.sh all`. Each arm is separately runnable
  (`lint`, `test`, `test-full`, `docs`, `gnn`); CI calls the same arms. See
  [`docs/RUN_GUIDE.md`](docs/RUN_GUIDE.md).
- Required PR status on `main` is `gnn conformance (local)`. Follow the
  [local GNN run and publication instructions](README.md#required-local-gnn-check).
  The gate is `./run.sh gnn`, which materializes the parser checkout bound by
  [the GNN pin](.github/gnn-pin.json), runs the adopted shared reader
  (`python -m gnn.importers check --repo . --require-pointer --workflow
  .github/workflows/ci.yml --job gnn-conformance`) and `tests/test_gnn_spec.py`
  offline on the clean PR head, and fails on a reader finding or a skipped
  parser test. The pin is GNN v5.1.0
  (`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`), advanced from v5.0.3
  (`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`). Retain the successful
  source-bound receipt and zero-skip report.
  Publish through the reviewed GitVault status plan using the mapped
  Institute identity. Recheck the live PR head before publishing and rerun
  for a new head. The hosted `gnn conformance (pinned)` job is optional; its
  pin, key, and display name are recorded in [`.github/GNN.md`](.github/GNN.md).
- Render from the sibling `template` checkout (`$TEMPLATE_REPO_ROOT`):

```bash
export TEMPLATE_REPO_ROOT=/path/to/template
uv run python scripts/z_generate_manuscript_variables.py
cd "$TEMPLATE_REPO_ROOT"
uv run python -m infrastructure.orchestration link-projects
uv run python scripts/pipeline/stage_03_render.py --project working/active_fractal_rabbit --skip-manuscript-hydration
```
