# Empirical Validation Protocol: Real Waypoint Claims Need a Separate Gate {#sec:empirical-protocol}

This protocol is a prerequisite for any future use of real waypoint or mobility
data with this project. It is deliberately separate from the synthetic-data
manuscript: nothing in the current fixture or external simulator lanes
satisfies any step below, and no current manuscript claim depends on this
document. Status: **draft — not yet executed**.

## Scope and Claim Boundaries Precede Data Access

- Define, in advance, the single empirical claim under test (for example:
  "the discrete filter's holdout predictive loss on dataset D beats the
  fixed-transition baseline"). One claim per study.
- Pre-register the claim, the metric, the comparison, and the stopping rule
  before any real data are processed.
- Any claim about intent, behavior semantics, or operational value remains
  out of scope regardless of results.

## Consent and Repository Separation Gate Real Data

- Real waypoint data may only enter through a documented agreement covering
  consent basis, permitted analyses, retention period, and deletion procedure.
- No co-mingling: empirical data live outside this repository; only aggregate,
  non-reidentifiable summary artifacts may be written into `output/`.
- A named data steward signs off before ingestion.

## Privacy Threat Assessment Runs Before Analysis

- Treat coarse spatiotemporal reduction as insufficient anonymization by
  default: small numbers of spatiotemporal points can uniquely identify
  individuals, and cross-dataset trace matching remains feasible after
  aggregation.
- Run an explicit re-identification assessment (uniqueness profile, linkage
  surface) before analysis; document residual risk and mitigations.
- Synthetic generation is not a privacy guarantee; if synthetic surrogates are
  released, evaluate their privacy leakage explicitly.

## Sampling Design Owns External Validity

- Document the population, sampling frame, observation process, and known
  selection biases of the empirical source.
- Define which FractalRabbit parameters the empirical source is supposed to
  correspond to, and how that mapping is checked.

## Frozen Plans and Negative Controls Prevent Retrofitting

- Freeze the config (`configs/`) and the pinned code commit before unblinding.
- Use per-entity chronological holdout as in the synthetic lane, with bin
  edges and burst statistics estimated on training data only, matching the
  current synthetic holdout leak guard.
- Report negative controls (shuffled time, shuffled identity) on the empirical
  data with the same directional expectations as the synthetic lane.

## Reporting Must Preserve Deviations and Nulls

- Publish the claim ledger row(s), the pre-registration, the release manifest
  of the analysis bundle, and all deviations from this protocol.
- Negative or null results are reported with the same prominence as positive
  results.
