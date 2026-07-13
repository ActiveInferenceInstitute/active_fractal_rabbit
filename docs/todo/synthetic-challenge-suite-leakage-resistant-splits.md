# Publish The Challenge Pack

Source task: `afr-37`
Status: in_progress
Top-level signpost phrase: Publish the challenge pack

## Current Evidence

The local challenge pack ingests the four-lane SOTA-program table, diagnostic
robustness grid, external fair-ladder realization manifest, lane definitions,
robustness summary, repeated diagnostic stressor rows, stronger fair-baseline
metadata, refreshed primary-source literature candidates, publication metadata,
license notices, privacy/task utility artifacts, and AIF-deepening audit
artifacts.

## Blocked And Unblocked Conditions

External publication remains open until a destination, release posture, fresh
render, release manifest, notices, challenge artifacts, and non-operational
usage statement are all current. Additional pinned stressor and realization
lanes remain future expansion, not prerequisites for local dry-run validity.

## Implementation Scope

Keep the local pack complete, leakage-resistant, and fail-loud against stale or
noncanonical required artifacts. When publishing, regenerate the pack after all
docs, manuscript, license, and artifact changes.

## Non-Goals

Do not publish stale outputs. Do not imply that the challenge pack is an
empirical benchmark or operational deployment surface.

## Acceptance Gates

`tests/test_challenge_pack.py` passes, release-manifest verification passes,
the sibling-template render is fresh, and stale or malformed required artifacts
are rejected.

## Artifact And Provenance Surfaces

Primary surfaces include `open_challenge_pack.json`, locked split artifacts,
leaderboard rows, claim gates, robustness diagnostics, external realization
accounting, literature-refresh candidates, license notices, and release
manifest entries.

## Claim Boundary

Challenge-pack publication can expose a synthetic benchmark. It cannot imply
empirical, operational, surveillance, or global SOTA readiness.
