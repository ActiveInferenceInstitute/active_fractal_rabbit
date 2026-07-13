# Architecture: Seven Layers Keep Synthetic Evidence Traceable {#sec:architecture}

## Pipeline Layers Write Artifact-Bound Evidence

Active FractalRabbit has seven explicit layers.

1. **Simulator boundary.** `fractalrabbit.py` writes FractalRabbit's seven-row
   parameter CSV and either uses a checked fixture or builds/runs the pinned
   upstream Java project from `.cache/fractalrabbit/`.
2. **Observation layer.** `observations.py` validates `ID,Days,x(km),y(km)` rows,
   sorts by traveller and time, and discretizes location, time-of-day, speed,
   and report-burst factors using the `observation:` config group.
3. **POMDP layer.** `pomdp.py` estimates normalized `A/B/C/D` arrays from the
   observation sequence, supports named transition controls, exports JAX arrays
   compatible with modern pymdp, and writes tensor summaries.
4. **Inference layer.** `inference.py` runs Bayesian state filtering and reports
   predictive loss, posterior entropy, policy posterior, policy-proxy
   components, and a fixed-transition baseline filter over the same observation
   factors.
5. **Artifact and figure layer.** `pipeline.py`, `figures.py`,
   `figure_style.py`, and `cover_art.py` write discrete feature tables,
   inference traces, baseline traces, candidate scores, provenance manifests,
   sensitivity summaries, robustness and roadmap analyses, variable provenance,
   the data-bound cover image, and a figure registry with captions, alt text,
   source artifacts, and a reproducible style record. The current default
   registry contains 53 manuscript figures: `evidence_atlas` opens the reader's
   path through sparse reports, belief updates, null gates, minimization, and
   integrity checks; `dataflow` closes the registry as the artifact-contract
   diagram. `visual_quality.py` re-opens the registry and registered PNGs to
   check readable dimensions, aspect-ratio extremes, nonblank pixel variance,
   caption source language, alt text, unique registry keys, existing
   data-derived source files, and the worst-case figures that need manual
   contact-sheet review. The visual convention is intentionally statistical:
   quantitative comparisons use shared axes or common signed scales when
   possible, uncertainty marks carry source artifacts and non-claim boundaries,
   and artifact tables remain the source of record behind the reader-facing
   plots. Its `layout_critical_figures` roster names the claim-bearing plots
   that require explicit post-regeneration inspection for clipping, label
   collisions, legends, and title stacking. `visual_claim_audit.py` adds a semantic mini-brief for
   every figure and the cover, checking that the story job, reader task,
   artifact source, data shape, encoding, support statement, and non-claim
   boundary still match the manuscript role.
   Manuscript captions are hydrated from this registry, so visual claims stay
   tied to generated artifact paths rather than handwritten figure prose.
6. **Claim-audit layer.** `claim_audit.py`, `eval_contract.py`,
   `variable_provenance.py`, and `docs/scholarship_ledger.md` attack
   false-certification paths above the analysis code: wrong-regime skill claims,
   unsupported performance language, stale result tokens, orphan high-risk claim
   families, claim-ledger rows that point at missing local evidence, and
   bibliography entries that could otherwise launder external literature into
   unsupported SOTA, privacy, or real-mobility claims.
7. **Closed-loop layer.** `closed_loop.py` selects the next declared simulator
   candidate from measured predictive loss or posterior entropy using transparent
   candidate score records and deterministic tie-breaks.

## Dependency Boundaries Prevent Vendored Simulator Drift

The project does not copy FractalRabbit or pymdp source. It stores only the
pinned upstream commit, configs, fixtures, and analysis code.
`THIRD_PARTY_NOTICES.md` records the pinned FractalRabbit Apache-2.0/NOTICE
boundary and the `pymdp` MIT dependency boundary; `docs/ip_license_status.md`
records the current private release-candidate publication status.

## Grouped Config Keeps Lanes Explicit

The grouped config surface is:

| group | purpose |
| --- | --- |
| `fractalrabbit` | upstream pin, fixture/external execution, simulator parameters |
| `observation` | spatial bins, bounds, time bins, speed thresholds, burst rule |
| `model` | smoothing, controls, preferences, modalities, engine, factorization / hidden factors / factorial controls |
| `experiment` | repetitions, holdout fraction, objective direction, tie-breaks |
| `figures` | enabled diagnostics, DPI, semantic palette, registry path, caption/source contract, visual-quality audit thresholds |
| `closed_loop` | iterations, objective, candidates |
| `robustness` | realization count, upstream commit attempts, external-lane confirmation |
