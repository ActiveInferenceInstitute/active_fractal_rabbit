# Multi-Realization External-Lane Fair Ladder

Source task: `afr-32`
Status: in_progress
Top-level signpost phrase: Multi-realization external-lane fair ladder

## Current Evidence

The full-study runner preflights JDK, Maven, and the pinned external cache. It
runs the current pinned external lane plus three deterministic pinned external
repeats when prerequisites pass, then writes completed, unavailable, and failed
row accounting to `output/runs/fair_inference_external_realizations.json`.

The current artifact has all required rows completed and no unavailable rows,
but promotion remains blocked because the completed rows do not agree on the
fair-ladder oracle-boundary controls.

## Blocked And Unblocked Conditions

The task remains in progress until external realizations agree on the
oracle-boundary controls and `can_promote_beyond_fixture_language` is true.
Unavailable or failed rows must remain explicit and cannot be imputed.

## Implementation Scope

Add pinned external realization rows only through documented config and
full-study orchestration. Keep fixture and external artifacts lane-qualified,
and keep failure modes visible in generated JSON and manuscript variables.

## Non-Goals

Do not promote fixture-positive wording from external rows that disagree. Do
not treat external simulator output as real waypoint data.

## Acceptance Gates

`output/runs/fair_inference_external_realizations.json` reports all required
rows, boundary controls agree, release manifest verification passes, and docs
and manuscript prose remain lane-qualified.

## Artifact And Provenance Surfaces

Primary surfaces are `fair_inference_external_realizations.json`,
`fair_inference.json`, external run directories, manuscript variables,
`docs/external_lane.md`, `docs/claim_ledger.md`, and the release manifest.

## Claim Boundary

External simulator agreement can strengthen simulator-lane language only. It
does not unlock real-world validation, empirical transfer, deployment, or global
SOTA wording.
