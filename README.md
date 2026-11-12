# Active FractalRabbit: Synthetic Waypoint Active-Inference Scaffold

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21330636.svg)](https://doi.org/10.5281/zenodo.21330636)
[![GitHub release](https://img.shields.io/badge/release-v0.2.0-blue)](https://github.com/ActiveInferenceInstitute/active_fractal_rabbit/releases/tag/v0.2.0)

Source, fixtures, manuscript, and the release manifest are public at
[github.com/ActiveInferenceInstitute/active_fractal_rabbit](https://github.com/ActiveInferenceInstitute/active_fractal_rabbit),
archived at Zenodo under [10.5281/zenodo.21330636](https://doi.org/10.5281/zenodo.21330636).

## Documentation

Full, category-organized documentation lives in [`docs/`](docs/README.md) — start at the
[documentation hub](docs/README.md); the flat file list is
[`docs/documentation-index.md`](docs/documentation-index.md).

## Table of contents

- [Citation and license](#citation-and-license-bound-the-release)
- [Quick start](#quick-start-runs-the-fixture-first-lane)
- [Required local GNN check](#required-local-gnn-check)
- [Repository layout](#repository-layout-assigns-artifact-ownership)
- [Configuration groups](#configuration-groups-define-fixture-model-and-artifact-lanes)
- [Synthetic-only evidence](#synthetic-only-evidence-blocks-real-trace-claims)
- [InstituteOS sidecar](#instituteos-sidecar)

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
  from forward work. GNN maintains a [downstream adoption
  guide](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/gnn/integration/downstream_adoption.md)
  for repositories that consume the notation;
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
./run.sh test-full
export TEMPLATE_REPO_ROOT=/path/to/template
uv run python scripts/z_generate_manuscript_variables.py
cd "$TEMPLATE_REPO_ROOT"
uv run python -m infrastructure.orchestration link-projects
uv run python scripts/pipeline/stage_03_render.py --project working/active_fractal_rabbit --skip-manuscript-hydration
cd projects/working/active_fractal_rabbit
uv run python scripts/29_finalize_publication_package.py
uv run python scripts/29_finalize_publication_package.py --verify-only
```

`$TEMPLATE_REPO_ROOT` is the sibling template checkout. The hydration command
writes `output/data/manuscript_variables.json` and the token-resolved
`output/manuscript/` tree; the render then reads that tree with
`--skip-manuscript-hydration` because the template's hydration subprocess
rejects a symlinked project slot. `./run.sh` is the
single gate entry point; each arm is one command and CI calls the same arm. See
[`docs/RUN_GUIDE.md`](docs/RUN_GUIDE.md) for the arm list and
[`docs/_generated/gate-timings.md`](docs/_generated/gate-timings.md) for the
measured wall time per arm on the current host.

External FractalRabbit preparation:

```bash
uv run python scripts/01_prepare_fractalrabbit.py --config configs/fractalrabbit_external.yaml
uv run python scripts/02_run_pipeline.py --config configs/fractalrabbit_external.yaml
```

## Required local GNN check

The `main` ruleset requires `gnn conformance (local)` on the current pull
request head. The hosted `gnn conformance (pinned)` job in
[CI](.github/workflows/ci.yml) is a separate, optional signal. The local
gate runs offline when `GNN_CHECKOUT` names a prepared checkout at the pin;
otherwise it clones the pinned tag. Publishing the result uses GitHub's
commit-status API without starting an Actions job.

Start in a clean checkout of the PR head. The gate is one arm:

```bash
./run.sh gnn
```

The arm reads [`.github/gnn-pin.json`](.github/gnn-pin.json) (now GNN
v5.1.0, `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`; previously v5.0.3,
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`), requires a checkout whose `HEAD`
is the recorded revision and whose worktree is clean, and installs that
checkout with its own uv version and lock. It then runs the shared reader
(`python -m gnn.importers check --repo . --require-pointer --workflow
.github/workflows/ci.yml --job gnn-conformance`), which owns pin and pointer
validation and spec syntax, followed by `tests/test_gnn_spec.py` with a
nonempty, zero-skip requirement for the model-declaration semantics the reader
does not check. Set `GNN_CHECKOUT` to an already prepared checkout at the pin
to run offline; otherwise the arm clones the pinned revision into `GNN_SCRATCH`
(or a temporary directory). The arm's exit status is the gate: a reader
finding or a skipped parser test fails it. See [`.github/GNN.md`](.github/GNN.md)
for the pin, the hosted job key and display name, and what the job parses.

In GitVault, run this arm through `execute --classification test` as described
in GitVault's `docs/EXECUTE.md`. Bind the clean Rabbit source and pin file in
the intent and set `GNN_CHECKOUT` inside the child. Use its
`GITVAULT_EXECUTION_OUTPUT` for the test report. The arm writes
`gnn-pytest.xml` into `GNN_SCRATCH` when that variable is set and into a
temporary directory otherwise.

Keep the source-bound GitVault receipt and the test output together. A skipped
parser test means the required check is incomplete even if pytest exits zero.

### Publish the local GNN result

Use the GitVault CLI with its `state` write class enabled and its mapped
Institute identity. Set `PR_NUMBER` to the pull request number, `VAULT` to the
launcher path, and `GNN_EXIT_RECEIPT` to the successful governed run's
`exit.json`. Review that receipt and the
zero-skip report, then re-read the PR head. Plan a success only if the live
head is the commit tested. A new head needs a fresh run.

```bash
set -euo pipefail
repo=ActiveInferenceInstitute/active_fractal_rabbit
tested_head=$(jq -er --arg repo "$repo" '
  .payload | select(.classification == "test" and .outcome == "SUCCEEDED"
    and .returncode == 0 and .source_unchanged == true and .binding.source == $repo
    and .descendants.sample_errors == 0 and .descendants.remaining == [])
  | .binding.revision | select(test("^[0-9a-f]{40}$"))
' "$GNN_EXIT_RECEIPT")
test "$(gh pr view "$PR_NUMBER" --repo "$repo" --json headRefOid --jq .headRefOid)" = "$tested_head"

digest=$(shasum -a 256 "$GNN_EXIT_RECEIPT")
request=$(jq -cn --arg repo "$repo" --arg head "$tested_head" --arg digest "${digest%% *}" '{
  method:"plan", arguments:{action:"github-status", parameters:{
    repository:$repo, expected_commit:$head, context:"gnn conformance (local)",
    state:"success", description:("Local pinned GNN checks passed. Receipt SHA-256 " + $digest[:16])
  }}
}')
plan=$("$VAULT" api --json "$request" | jq -er 'select(.ok == true) | .result.github')
"$VAULT" github-apply --plan "$plan"
```

After reviewing the exact repository, commit, and context, apply and verify
that specific status on GitHub.

```bash
set -euo pipefail
test "$(gh pr view "$PR_NUMBER" --repo "$repo" --json headRefOid --jq .headRefOid)" = "$tested_head"
"$VAULT" github-apply --plan "$plan" --apply --confirm 'REPORT 1 COMMIT STATUS'
gh api --method GET "repos/$repo/commits/$tested_head/status" |
  jq -e --arg head "$tested_head" '.sha == $head and
    (.statuses | any(.context == "gnn conformance (local)" and .state == "success"))'
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
| `run.sh` | Single gate entry point: `lint`, `test`, `test-full`, `docs`, `gnn` arms |
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

## InstituteOS sidecar

This repository carries a committed [`.aii/`](.aii/README.md) sidecar with curated
InstituteOS metadata (`aii-sidecar/v1`): category, maturity, ecosystem relations,
key artifacts, provenance, and portable tasks. It is read by
[InstituteOS](https://github.com/ActiveInferenceInstitute/InstituteOS) tooling and
does not affect the project's build or runtime.
