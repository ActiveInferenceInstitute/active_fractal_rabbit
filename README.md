# Active FractalRabbit: Synthetic Waypoint Active-Inference Scaffold

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21330636.svg)](https://doi.org/10.5281/zenodo.21330636)
[![GitHub release](https://img.shields.io/badge/release-v0.2.0-blue)](https://github.com/ActiveInferenceInstitute/active_fractal_rabbit/releases/tag/v0.2.0)

Source, fixtures, manuscript, and the release manifest are public at
[github.com/ActiveInferenceInstitute/active_fractal_rabbit](https://github.com/ActiveInferenceInstitute/active_fractal_rabbit),
archived at Zenodo under [10.5281/zenodo.21330636](https://doi.org/10.5281/zenodo.21330636).

Active FractalRabbit is a closed-loop research scaffold for active-inference
analysis of synthetic sporadic waypoint data. It keeps the external simulator
boundary explicit: FractalRabbit is pinned to commit
`9933449c4f4fe1b26b6ac7bfdeeac76583085df5`, fetched/built into `.cache/`, and
not committed into this repository. The external lane runs the real
stochastic simulator end to end (JDK 21 + Maven, resolved automatically), and
the inference layer runs the real pinned `pymdp` JAX agent alongside the
transparent reference filter, with matched-prior agreement validated at every
step.

The default config uses the checked 2,400-row
`data/fixtures/fractalrabbit_sample_large.csv` fixture so the Python pipeline,
manuscript variables, figures, audits, and tests run without Java or Maven. The
smaller `data/fixtures/fractalrabbit_sample.csv` fixture remains the regression
and parser-contract surface. Use `configs/fractalrabbit_external.yaml` for real
pinned simulator runs.

The current scaffold emits:

- parsed observations, discrete feature tables, normalized tensor summaries,
  inference traces, policy proxy diagnostics, fixed-transition baseline traces,
  additive per-traveller holdout metrics, and candidate-score records;
- seeded negative controls, fixture/external sensitivity summaries, sparsity,
  calibration, ground-truth state-recovery, `pymdp` diagnostics, and the
  `pymdp` performance/interpretability benchmark matrix;
- partial-observability, fair-inference, external fair-realization, fair power-regime,
  SOTA task-contract, SOTA leaderboard, SOTA program gate, SOTA program leaderboard,
  SOTA robustness-grid, regime-qualified AIF search/gate, AIF sweep, and
  open challenge-pack artifacts;
- a machine-checked single/factorial generative-model specification declared in
  Generalized Notation Notation (`models/*.gnn.md`) with an upstream-parser drift
  guard, a within-regime model-comparison leaderboard, and a repeated
  online-transition adaptation ablation whose completed evidence is separated
  from forward work;
- return-structure, variance-band, upstream-sensitivity, fractal-dimension,
  circadian-phase, sequence-baseline, reporting-process/missingness,
  return-plus-phase, lane-resolved model-gate, privacy-frontier,
  linkage-risk, poisoning-robustness, and hardened-anomaly artifacts;
- figure registry/style metadata, cover art, visual-quality audit,
  claim audit, visual-claim audit, fail-loud manuscript-variable provenance,
  manuscript variables, a primary-source scholarship ledger that covers every
  manuscript citation key with allowed/forbidden claim language, and a
  render-freshness proof and checksummed release manifest covering the final PDF
  and combined HTML.

## Citation and License Bound the Release

Original project code is licensed under MIT in `LICENSE`. Project-authored
documentation, manuscript source, figures, fixture descriptions, and other
scholarly outputs are licensed under CC-BY-4.0 in `LICENSE-DOCS.md` unless a
file states otherwise. Citation metadata, including the Zenodo DOI, lives in
`CITATION.cff`; third-party simulator and dependency notices live in
`THIRD_PARTY_NOTICES.md`; the current IP/license and publication-readiness
status is recorded in `docs/ip_license_status.md`.

This repository reached public release as `v0.2.0` (2026-07-12): a current
template render, full test gate (658 tests, 91.33% coverage), claim and
visual audits, and a regenerated, verified `output/release_manifest.json` all
passed before the [GitHub release](https://github.com/ActiveInferenceInstitute/active_fractal_rabbit/releases/tag/v0.2.0)
and [Zenodo deposit](https://doi.org/10.5281/zenodo.21330636).

## Quick Start Runs the Fixture-First Lane

```bash
uv sync --extra dev
uv run python scripts/00_preflight.py
uv run python scripts/02_run_pipeline.py --config configs/default.yaml
uv run python scripts/03_run_sweep.py --config configs/sweep.yaml
uv run python scripts/05_run_negative_controls.py --config configs/default.yaml
uv run python scripts/08_run_sparsity_analysis.py --config configs/default.yaml
uv run python scripts/04_generate_figures.py --run output/runs/default
uv run python scripts/z_generate_manuscript_variables.py
uv run python scripts/11_check_manuscript_variable_provenance.py
uv run python scripts/07_run_full_study.py --lane both   # one-command full study
uv run pytest tests/ --cov=src --cov-fail-under=90
cd ../../../template
uv run python -m infrastructure.orchestration link-projects
uv run python scripts/03_render_pdf.py --project working/active_fractal_rabbit
cd ../projects/working/active_fractal_rabbit
uv run python scripts/29_finalize_publication_package.py
uv run python scripts/29_finalize_publication_package.py --verify-only
```

External FractalRabbit preparation:

```bash
uv run python scripts/01_prepare_fractalrabbit.py --config configs/fractalrabbit_external.yaml
uv run python scripts/02_run_pipeline.py --config configs/fractalrabbit_external.yaml
```

## Repository Layout Assigns Artifact Ownership

| Path | Purpose |
| --- | --- |
| `configs/` | Default fixture, external FractalRabbit, and `fixtures/` variant configs |
| `data/fixtures/` | Default 2,400-row fixture plus small regression/parser-contract CSV |
| `models/` | Generalized Notation Notation specs for the generative models (structural source of truth) plus the pinned upstream revision |
| `src/active_fractal_rabbit/` | Config, simulator boundary, observation, POMDP, inference, and loop code |
| `scripts/` | Thin CLI entrypoints |
| `tests/` | Unit, fixture-pipeline, and optional external integration tests |
| `manuscript/` | Template-renderable research write-up |
| `output/` | Generated runs, figures, and manuscript variables |
| `LICENSE`, `LICENSE-DOCS.md`, `CITATION.cff`, `THIRD_PARTY_NOTICES.md` | Code, scholarly-output, citation, and third-party notice surfaces |
| `TODO.md` | Project-scale roadmap and acceptance milestones |

## Configuration Groups Define Fixture, Model, and Artifact Lanes

The preferred config groups are:

- `fractalrabbit`: upstream pin, fixture/external execution, simulator parameters.
- `observation`: spatial discretization, bounds, time bins, speed thresholds, burst rule.
- `model`: smoothing, controls, preference prior, observation `modalities` (deep-latent lane), inference `engine` (`reference` / `both` with the real pymdp agent), and factorization options.
- `experiment`: sensitivity repetitions, additive holdout fraction, objective direction, tie-break policy.
- `figures`: enabled diagnostics (including `negative_controls`), DPI, palette, output/registry path.
- `closed_loop`: objective, iterations, and candidate simulator parameter overrides.
- `robustness`: realization count, upstream candidate attempts, pinned external
  fair-ladder repeat dimensions, and external-lane confirmation policy.

Top-level `inference:` keys are rejected; use the grouped `observation:` and `model:` blocks.

## Synthetic-Only Evidence Blocks Real-Trace Claims

This project analyzes synthetic waypoint data. It does not claim that inferred
latent states are real human intent, that FractalRabbit reproduces any specific
population, or that active-inference metrics validate operational decisions on
real mobility traces without separate empirical validation.
