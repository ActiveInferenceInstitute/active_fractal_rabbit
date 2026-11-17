# Regime-Stressor Lane

Source task: `afr-34`
Status: in_progress
Top-level signpost phrase: Regime-stressor lane

## Current Evidence

The canonical SOTA-program harness scores reporting-gap and larger
regime-stressor lanes. The diagnostic robustness grid adds repeated
reporting-gap, high-motion adversarial-missingness, high-motion adversarial
repeat, and two regime-stressor repeat rows while keeping those diagnostics
outside the canonical four-lane SOTA gate.

The diagnostic generator supports the finite reporting-gap, high-motion
missingness, regime-switch, and long regime-switch profile grid with explicit
realization identities. Profile selection preserves existing realizations and
does not change the canonical four-lane split schedule, train-only tuning
rules, or existing negative claim locks. This describes source/CLI support,
not a regenerated diagnostic result or a cleared gate.

## Blocked And Unblocked Conditions

The task remains open until broader stressor grids and enough repeated
realizations exist to revisit negative HMM, learned-sequence, learned-prior,
reporting-process, and AIF regime gates without weakening current bounded-null
artifacts.

## Implementation Scope

Expand synthetic stressors for reporting gaps, burst missingness, high-motion
missingness, regime switching, and repeated stressor realizations. Regenerate
SOTA robustness, SOTA program, AIF regime, and multiplicity artifacts together.

`scripts/18_run_sota_program.py` exposes `--diagnostic-lane` and
`--diagnostic-n-boot`; `scripts/20_run_aif_regime_search.py` exposes
`--diagnostic-realization-seed`. These select separate diagnostic work, not
replacement canonical lanes. Diagnostic CSV/JSON, SOTA-program/regime artifacts,
and the joint multiplicity ledger must remain one bound family, including all
declared candidates and non-wins. The source repair binds the emitted AIF
diagnostic CSV to its search and the regime gate to those exact search bytes;
shared consumers also verify the existing SOTA CSV/grid bindings. Missing or
changed CSVs and missing or mismatched regime gates cannot supply current joint
evidence. Deepening withholds stale regime/repetition fields, and challenge/release
coverage requires `aif_regime_diagnostic_repetitions.csv` under the same family
validation. This is implemented source behavior awaiting independent parent
qualification, not a regenerated result or closure of `afr-34`. The complete-family
contract is documented in [`statistics.md`](../methods/statistics.md).

## Non-Goals

Do not move diagnostic stressor wins into the canonical SOTA gate unless the
full gate clears. Do not make diagnostic lanes imply external simulator SOTA or
real mobility behavior.

Diagnostic repetitions count synthetic stressor realizations. They are not
online-transition adaptation-trial counts, independently observed empirical
trials, or evidence that an external simulator run completed.

## Acceptance Gates

The SOTA robustness grid, SOTA program gate, AIF regime gate, and multiplicity
ledger all regenerate together; stronger wording stays lane-bound unless the
full claim gate clears.

## Artifact And Provenance Surfaces

Primary surfaces include `sota_program_robustness_grid.json`,
`sota_program_robustness_grid.csv`, `sota_program_gate.json`,
`aif_regime_search.json`, `aif_regime_diagnostic_repetitions.csv`,
`aif_regime_sota_gate.json`, `multiplicity_ledger.json`, figures, manuscript
variables, and release manifest entries.

## Claim Boundary

Stressor evidence is synthetic-only. Local wins must name the exact regime,
seed set, candidate roster, fair baseline, uncertainty, and correction status.
