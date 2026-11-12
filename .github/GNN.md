# GNN conformance: adopted shared reader

This repository consumes the upstream Generalized Notation Notation (GNN)
parser as a pinned external tool. It has **adopted** the
[GNN–FEP shared-reader contract](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/workflows/gnn-fep.md):
the `./run.sh gnn` arm runs the reader the pin names, inside the pinned parser's
environment. The pin is the gate; it is never a convenience input.

## Canonical pin

- File: [`.github/gnn-pin.json`](gnn-pin.json)
- Schema: `gnn-importer-pin/v1`
- Upstream repository:
  `https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation.git`
- Tag: `v5.1.0`
- Revision: `75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf` (the commit the tag
  dereferences to)
- Upstream version: `5.1.0`
- Pinned on: `2026-11-12`; recorded in the current schema on `2026-11-12`
- Spec directory: `models/`
- Claim boundary: `synthetic-only`

The pin advanced from GNN v5.0.3
(`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`) to v5.1.0
(`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`), the first release that ships
`src/gnn/importers.py`. The old revision carried the local reader this change
deletes; the new revision supplies the shared reader the arm calls.

## The shared reader

After materializing the pinned revision, the arm runs:

```bash
python -m gnn.importers check --repo . --require-pointer \
    --workflow .github/workflows/ci.yml --job gnn-conformance
```

The reader owns pin discovery and shape, the pointer check, and the syntax
validation of every specification the pin names. It reports `pin-missing`,
`pin-unreadable`, `pin-commit-shape`, `pin-commit-mismatch`,
`pin-tag-mismatch`, `pin-repository-mismatch`, `pin-schema-unversioned`,
`spec-missing`, `spec-invalid`, and the pointer codes. This repository does not
carry a second pin reader.

The reader does not check model-declaration semantics. The same arm runs
`tests/test_gnn_spec.py` against the pinned checkout with a nonempty, zero-skip
requirement; those tests hold the parsed factor, observation, and control
cardinalities plus the `A`/`B` dependency edges equal to the in-code
declarations.

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

The arm reads only the revision from the pin so it can materialize the pinned
parser. It requires a checkout whose `HEAD` is the pinned revision and whose
worktree is clean, installs that checkout into its own virtualenv, runs the
shared reader, then runs the parser tests with zero skips. Set `GNN_CHECKOUT`
to a prepared checkout at the pin to run offline; otherwise the arm clones the
pinned revision into `GNN_SCRATCH` (or a temporary directory).

## CI setup action

The workflow prepares Python and uv with the pinned third-party action
`astral-sh/setup-uv@68559e99cc807c1eb7e9238fd43a15bba460eec2` (v10.3.3), uv
`0.13.7`, Python `3.14`. It does not use the Institute composite action
`ActiveInferenceInstitute/.github/.github/actions/setup-python-uv@7159e67315330aac8e414aebae0ad1e999a41448`.
