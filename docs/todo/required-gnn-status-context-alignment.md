# Required GNN Status Context Does Not Match The Live Ruleset

Source task: `afr-45`
Status: blocked
Top-level signpost phrase: GNN status context mismatch

## Current Evidence

`README.md` ("Required local GNN check") states that "the `main` ruleset
requires `gnn conformance (local)` on the current pull request head" and that
the hosted `gnn conformance (pinned)` job is "a separate, optional signal".
`.github/GNN.md` ("Hosted job") repeats the same split: "The job is an optional
hosted signal. The required status on `main` is the commit status `gnn
conformance (local)`".

The live ruleset says otherwise. `gh api
repos/ActiveInferenceInstitute/active_fractal_rabbit/rulesets/26570052` returns
the active branch ruleset `require-pr-check` whose single rule is
`required_status_checks` with context `gnn conformance (pinned)` — the hosted
Actions job, not the local commit status. That context is a job in
`.github/workflows/ci.yml`, so it can only report while GitHub Actions runs
jobs for this repository.

It does not. Run `43280673571` for the merge commit `3803a29` (the GNN cutover,
PR #24) concluded `failure` with no jobs created, and `gh api
repos/ActiveInferenceInstitute/active_fractal_rabbit/commits/main/status`
reports `total_count: 0` — the tree merged with neither the required hosted
context nor the documented local one present. `branches/main/protection`
returns `Branch not protected`; the enforcement lives entirely in the ruleset.

The consequence is concrete: the check the repository enforces cannot turn
green, while the check the repository documents is not enforced.

## Blocked And Unblocked Conditions

Blocked on Institute GitHub organization administration. Changing a repository
ruleset requires admin rights on the repository; PR authors cannot edit
`ruleset 26570052`.

Unblocks when an administrator sets the `require-pr-check` required context to
`gnn conformance (local)`, or when the organization re-enables Actions billing
and the maintainers instead decide the hosted job is the required signal.

The in-repository-only alternative — rewording `README.md` and `.github/GNN.md`
to say the hosted job is required — is not an acceptable substitute: with
Actions billing-locked org-wide, it would document a check that cannot run and
would discard the offline, source-bound signal the pin exists to protect.

## Implementation Scope

After the ruleset edit, confirm the live context with `gh api
repos/ActiveInferenceInstitute/active_fractal_rabbit/rulesets/26570052 --jq
'.rules[].parameters.required_status_checks[].context'`, publish
`gnn conformance (local)` from a source-bound local `./run.sh gnn` run as
`README.md` describes under "Publish the local GNN result", and re-read the
`main` branch rule statements in `README.md`, `.github/GNN.md`, and
`docs/RUN_GUIDE.md` so all three name the same enforced context.

## Non-Goals

Do not reintroduce a hosted-only evidence path, treat an Actions run as the
standard gate, relax the zero-skip parser requirement, or advance the GNN pin.
Do not edit `docs/_generated/gate-timings.md` by hand; `./run.sh bench` owns it.

## Acceptance Gates

The ruleset's required context reads `gnn conformance (local)`; the merged PR
head carries that context as `success`; `./run.sh gnn` still fails on a reader
finding or a skipped parser test; `./run.sh docs` and
`pytest tests/test_docs_contract.py` pass.

## Artifact And Provenance Surfaces

Primary surfaces are `.github/gnn-pin.json`, `.github/GNN.md`, `README.md`,
`docs/RUN_GUIDE.md`, the GitVault execution receipt plus the zero-skip
`gnn-pytest.xml` report, and `docs/_generated/gate-timings.md`.

## Claim Boundary

This is repository governance and does not move the synthetic-only evidence
boundary. No pin, artifact, or manuscript claim changes because a status
context is renamed.
