# GNN conformance: pin, job, and what it parses

This repository consumes the upstream Generalized Notation Notation (GNN)
parser as a pinned external tool. The pin is the gate; it is never a
convenience input.

## Canonical pin

- File: [`.github/gnn-pin.json`](gnn-pin.json)
- Schema: `gnn-importer-pin/v1`
- Upstream repository:
  `https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation.git`
- Tag: `v5.0.3`
- Revision: `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1` (the commit the tag
  dereferences to)
- Upstream version: `5.0.3`
- Pinned on: `2026-11-01`; recorded in the current schema on `2026-11-12`
- Spec directory: `models/`
- Claim boundary: `synthetic-only`

`tests/test_gnn_spec.py` re-verifies the tag-to-revision binding and the
in-repo declarations whenever the pinned checkout is available. The pin file is
the only machine-readable copy; do not restate the revision in workflow YAML.

## What the conformance job parses

The job validates exactly the files named in `spec_files`:

- `models/single_factor_reference.gnn.md`
- `models/factorial_latent_mode.gnn.md`

It runs the upstream strict validator
(`python -m gnn.cli validate <spec> --strict`) from the pinned checkout, then
runs `tests/test_gnn_spec.py` with `GNN_CHECKOUT` bound to that checkout. The
test module re-parses both files with the upstream
`gnn.parsers.markdown_parser.MarkdownGNNParser` and compares the parsed factor,
observation, and control cardinalities plus the `A`/`B` dependency edges against
the in-code declarations in `src/active_fractal_rabbit/model_spec.py`. A
nonempty suite with zero skips is required; a skipped parser test is an
incomplete gate.

## Hosted job

- Workflow: [`.github/workflows/ci.yml`](workflows/ci.yml)
- Job key: `gnn-conformance`
- Display name: `gnn conformance (pinned)`
- Step: `./run.sh gnn`, the same arm documented in
  [`docs/RUN_GUIDE.md`](../docs/RUN_GUIDE.md)

The job is an optional hosted signal. The required status on `main` is the
commit status `gnn conformance (local)`, published from a source-bound local run
as described in [`README.md`](../README.md#required-local-gnn-check); the two
names are deliberately distinct.

## Local arm

```bash
./run.sh gnn
```

The arm reads the pin, requires a checkout whose `HEAD` is the pinned revision
and whose worktree is clean, installs that checkout into its own virtualenv,
validates both specs, and runs the parser tests with zero skips. Set
`GNN_CHECKOUT` to a prepared checkout at the pin to run offline; otherwise the
arm clones the pinned tag into `GNN_SCRATCH` (or a temporary directory).

## CI setup action

The workflow prepares Python and uv with the pinned third-party action
`astral-sh/setup-uv@68559e99cc807c1eb7e9238fd43a15bba460eec2` (v10.3.3), uv
`0.13.7`, Python `3.14`. It does not use the Institute composite action
`ActiveInferenceInstitute/.github/.github/actions/setup-python-uv@7159e67315330aac8e414aebae0ad1e999a41448`.
