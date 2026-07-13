# Stronger Baselines Re-Scored On The Locked Split

Source task: `afr-38`
Status: in_progress
Top-level signpost phrase: Stronger baselines re-scored on the locked split

## Current Evidence

The four-lane SOTA-program leaderboard reports loss, calibration, top-1
accuracy, traveller-clustered uncertainty intervals, runtime, dependency
footprint, failure modes, lane coverage, scored-or-blocked robustness status,
and blocked-row rationale for foundation or Transformer-style candidates. A
protocol-compatible higher-order backoff Markov fair baseline is in the
transparent baseline ladder.

## Blocked And Unblocked Conditions

Remaining work is admissible stronger or foundation rows and broader repeated
stressor grids. A row is admissible only when it returns a locked-split
next-cell distribution and has auditable adapter, dependency, license, compute,
data, failure-mode, and leak-check contracts.

## Implementation Scope

Extend the strong-baseline harness with dependency-light sequence/state-space
models first, then foundation-style candidates only when adapter and licensing
contracts are complete.

## Non-Goals

Do not admit token-only, prose-only, health-only, or unavailable baselines as
scored rows. Do not weaken baselines to manufacture an AIF win.

## Acceptance Gates

Every admitted row reports loss, uncertainty, Brier/ECE, top-1 accuracy,
runtime, dependency footprint, lane coverage, failure modes, and leak checks;
missing families keep the SOTA gate closed.

## Artifact And Provenance Surfaces

Primary surfaces include `sota_program_leaderboard.csv`,
`sota_program_gate.json`, `sota_task_contract.json`,
`sota_benchmark_matrix.json`, challenge-pack rows, manuscript variables, and
claim-ledger entries.

## Claim Boundary

The strongest implemented fair baseline sets the local benchmark threshold.
No global SOTA wording is allowed until the full claim gate clears.
