# Active FractalRabbit — Major-Revision Program (master plan)

> **HISTORICAL PLANNING RECORD — retained for provenance; superseded by current artifacts.**
> This is a frozen planning snapshot from the revision program. All quantitative values quoted
> below (fair rates, nats gaps, traveller/report counts, fixture sizes) are snapshots of the runs
> current when this plan was written and are superseded by the current `output/runs/` artifacts,
> `docs/claim_ledger.md`, and the rendered manuscript — several headline numbers here (e.g. the
> "organizing fact" fair-rate and soft-marginalization gain, and the 6-traveller/480-report
> fixture narrative) no longer match the shipped fixture or artifacts. Some planned steps were
> deliberately not executed (e.g. the comparison-contract table/module and the discussion-collapse
> restructure). Do not cite numbers or structure from this file.

Synthesizes three deep `/workflows` plans ([Pass A — peer review](pass-A-peer-review.md),
[Pass B — IMRaD restructure](pass-B-restructure-imrad.md), [Pass C — pymdp push](pass-C-pymdp-push.md))
into one sequenced, checkpointed program. Each stage ends suite-green + committed.

## Organizing fact (the 6-traveller re-run finding)

The user-requested re-run (4→6 travellers, 240→480 reports, broader sweep) **empirically refuted the
fair-frontier AIF advantage at scale**: `aif_class_best_fair_rate` fell 8/8 → **1/8**, soft-marginalization
gain collapsed to ~0.017 nats once the emission posterior is dense. The power-analysis regime prediction
(advantage vanishes as travellers/data grow, re-emerges only under noise) is now realized on the *default*
fixture. This **mandates** the peer review's core recommendation: reposition as a **benchmark + evaluation
+ governance paper**; active inference is one instrument, valued for *inspectability*, not performance.
All abstract/roadmap prose was corrected this stage to state the eroded result honestly.

## Staged sequence (dependencies respected)

- **Stage 0 — Re-run landed (THIS STAGE):** 6-traveller fixture + broader sweep regenerated (all 20 stages ok);
  abstract + roadmap rewritten to the honest eroded result; scale-pinned tests updated; suite green; PDF clean.
- **Stage 1 — Repositioning + comparison contract (Pass A items 1–4,12):** benchmark-first abstract/intro;
  **auto-generated comparison-contract table** (oracle/hard/soft/soft+boost/aif — info set, task, oracle status,
  loss, verdict) emitted from `fair_inference.py`, embedded in Intro + early Results; move the fair-comparison
  ladder into a new **Results §0 before** the fixture-diagnostics; rename opaque verdict labels to encode
  direction. Citations: Rubin, Talts, Yao, Roberts, Rocher, Dwork-Roth. Dual-use statement.
- **Stage 2 — Methods rigor (Pass A items 5–9) + pymdp push (Pass C):** leakage-resistant defaults
  (train-only discretization/thresholds, traveller- AND realization-disjoint locked splits) with firing tests;
  **traveller-clustered bootstrap as default** (now viable at 3 clusters); emission conditional-independence
  ablation; SBC (Talts) + VI diagnostics (Yao); MAR/MNAR reporting-mechanism sensitivity (Rubin). In parallel,
  the **9 pymdp levers** (pairwise emission, EMA/per-traveller online adapt, super-cell backoff, multiscale,
  precision schedule, return priors, long-gap feature) — prototype-before-claim on the 6-traveller fixture,
  honest that at-scale a recovered advantage is unlikely; document nulls. Each lever has a kill-gate.
- **Stage 3 — IMRaD restructure + titles (Pass B):** move diagnostic/robustness/forward-gate results to new
  supplements **S03 (diagnostics & robustness)** and **S04 (advanced model comparisons & transfer)**; collapse
  05_limitations + 07_roadmap + 08_conclusion into a single **04_discussion.md** (boundaries → gates →
  final-claim); reduce main-text figures to ~4–5 core; **result-bearing section titles** throughout (anchors
  preserved). Done last so the restructure reflects settled content.
- **Stage 4 — Adversarial close:** `/workflows` RedTeam of the repositioned claims + new methods; Forge
  cross-vendor audit as a constraint on promotion; full suite + render; commit.

## Cross-cutting invariants (every stage)
- eval_contract G1/G2/G3 + no-hardcoded-numerals + token provenance stay green; numbers via {{TOKENS}} only.
- Prototype-before-claim; every new check proven to FIRE on an injected defect; honest nulls documented.
- The comparison-contract table and all headline numbers are auto-generated from artifacts (never hand-transcribed).
- Anchors `{#sec:/#fig:}` preserved across any move; 0 dangling refs; visual gates pass.

## Acceptance (program-level)
Benchmark-first framing structural (not decorative); fair-comparison + regime finding lead the Results;
leakage-resistant + traveller-clustered defaults for all headline claims; main text ~4–5 figures with the rest
in S01–S04; result-bearing titles; pymdp levers swept with honest verdicts; suite green; PDF clean; committed.

---

## EMPIRICAL FOUNDATION — the 6-traveller validation run (2026-06-17)

The user-requested re-run (4→6 travellers, 240→480 reports, 36 days, broader sweep) was executed
end-to-end. Two outcomes:

1. **Pipeline confirmed at scale.** All 20 fixture-lane stages ran "ok" — every figure, the
   visual-quality + visual-claim audits, manuscript-variable provenance, eval_contract, claim
   audit, release manifest and verification all passed at the larger scale. The visualizations
   and output validation/injection paths work perfectly under 3× data.

2. **The findings are scale-fragile (the decisive result).** Scaling refuted or shifted **22
   result-level assertions**. The headline collapse: the fair-frontier AIF advantage fell from
   8/8 → **1/8** seeds; soft-marginalization gain went from ~+2.1 → **−0.04 nats** (a dense
   emission posterior makes the hard point estimate as good); the return-boost feature, which
   "helped" at four travellers, **hurts** at six; and structure-recovery, poisoning-detection,
   EFE-metadata-share, the m10 forward gates, and sparsity assertions all moved. The mission-
   collection result inverted (EFE now beats pragmatic in both regimes, beats epistemic only when
   unresolved).

**Interpretation.** The active-inference / inference "advantages" reported on the canonical
four-traveller fixture are largely **small-fixture artifacts that do not survive scaling** — the
power-analysis regime prediction realized on the default fixture. This is the strongest possible
*empirical* case for the peer review's benchmark-first repositioning: the durable contribution is
the **benchmark + the scale-fragility characterization**, not active-inference performance.

**Decision.** Adopting six travellers as the canonical fixture requires re-validating all ~22
findings against their new measured reality AND a benchmark-first manuscript rewrite — i.e. the
whole of Stage 1 below. To avoid shipping a 22-test-red tree mid-revalidation, the canonical
4-traveller green state was restored; this finding is the empirical mandate for Stage 1, whose
**first action** is therefore: re-adopt the 6-traveller fixture, re-validate every flipped finding
to its measured value with honest framing, and rewrite the manuscript as a benchmark paper whose
central empirical result is *the inference advantages are scale-fragile*.

---

## REAL-SIMULATOR FINDING — individual vs population (2026-06-17, external lane)

Verified on a **real FractalRabbit external-simulator trace** (Java/Maven build of the pinned
commit; `fractalrabbit_externalXY.csv`, 5703 reports, 8 travellers — genuine sim output, not the
fixture, no stub/mock/fallback):

- **The N=1 numbers are real, but the mechanism is NOT active inference — RedTeam-corrected.** For
  all 8/8 real targets the individual model's advantage over the pooled-population transition is
  maximal at zero population (+1.9 to +2.1 nats) and collapses as travellers accumulate. BUT on the
  real trace travellers are severely spatially confined (5/8 occupy <=3 of 16 cells; one is 99% in a
  single cell). The win is **information congruence / spatial confinement**: each traveller's own
  confined history is on-distribution, while a transition pooled across travellers with *disjoint*
  territories is off-distribution for any target. The return weight rho barely changes the individual
  loss, and a trivial dominant-cell predictor (0.145 nats) beats the model (0.44) — an information-set
  mismatch ANY individual model captures, not a structure-matched active-inference win.
- **The frequency-vs-binary structure match is real but small** (+0.22 to +0.66 nats on the real
  trace) and secondary, not the driver.
- **Once again an apparent active-inference advantage reduces to a simpler effect** (here: spatial
  confinement) — the project's defining pattern and the strongest empirical case for benchmark-first.

**Stage-1 / Pass-C implication.** The honest open question — whether a structure-matched
active-inference generative model genuinely beats the dominant-cell AND first-order baselines on a
**non-confined, mobility-checked** regime — is NOT established and is the proper target of the
AIF-model-development program (FirstPrinciples/BeCreative designs with a mandatory confound battery).
Shipped: `preferential_return.py`, `individual_vs_population.py` (+ CI-safe tests; real-trace result
reproduced via the external lane). Real methods throughout — no stub/mock/fake.

---

## CONFOUND-CONTROLLED POSITIVE — long-memory clears the battery on a mobile regime (2026-06-17)

Built the mandatory confound battery as `confound_controls.py` (dominant-cell, first-order pooled
Markov, return-boosted soft, oracle-prev) + a mobility check. Then tested the long-memory
frequency-return model against it on the REAL external trace at FINER grids (so confined travellers
become genuinely mobile — a non-confined, mobility-checked regime):

- **8x8** (mean_distinct=5.4, confined_frac=0.12, mobile=True): long-memory **clears the full battery
  in 5/6 seeds** (beats dominant-cell, pooled-Markov, AND return-boosted-soft), gap-to-oracle +0.52.
- **10x10** (mean_distinct=6.0, confined_frac=0.00, mobile=True): **clears 6/6 seeds**, gap +0.26.

This is the project's **first genuine, confound-controlled active-inference positive**: on a
mobility-checked regime it is NOT confinement (confined_frac 0-0.12), NOT data density (beats the
pooled Markov), NOT oracle access (honest gap reported). The discipline that dissolved every prior
claim now *certifies* this one. Caveat: it holds on a finer-grid discretization of the real trace,
not the default 4x4; the honest open work is to make this a first-class manuscript result (Stage 1)
and to run the MNAR and multiscale families through the same battery (designs in
`aif-model-designs.md`). Shipped: `confound_controls.py` + CI-safe tests.

---

## AIF MODEL FAMILY PASS — battery results (2026-06-17)

Ran all three designed families through the confound battery on the mobility-checked finer-grid real
regime (bins=8,10, >=6 seeds). Independent fair (inferred-prev, no oracle access) re-verification:

| Family | Fair battery result | Verdict |
|--------|--------------------|---------|
| **Long-memory / preferential-return** | clears 5/6 (bins=8), 6/6 (bins=10) | **CLEARS** — the genuine positive |
| MNAR joint motion-reporting | clears 0/6 (both grids) | NULL — agent's "win" was an oracle-prev LEAK (used true previous cell); fair version fails |
| Multiscale / hierarchical backoff | clears 0/6 (both grids) | NULL — essentially first-order-pooled + coarse smoothing; adds no return signal |

The MNAR prototype illustrates the project's defining trap a fifth time: a /workflows agent reintroduced
the oracle-access confound (predicting from the TRUE previous cell), which "beat" the fair baselines
trivially. RedTeam caught it (model loss BELOW the oracle is impossible for a fair model); the fair
rebuild is a clean null. The battery rejects nulls (2/3 families) and certifies the one real effect —
working as designed. Only long-memory earns promotion to the manuscript; the two nulls are recorded
here, not built (no module for a null).

---

## CROSS-VENDOR AUDIT (Forge / GPT-5.4) — the noise frontier is a confound; advantage retracted (2026-06-18)

Built an observation-noise advantage frontier (smoothing as the noise axis) that appeared to show a
clean, monotone, noise-driven active-inference advantage across all densities. **Forge cross-vendor
audit refuted it, and I independently confirmed the mechanism:** the noise knob (emission smoothing)
is wired into the BASELINES' visit prior (`confound_controls.py` `np.full(n, smoothing)`) but NOT the
model's return prior (`np.zeros(n)`). Cranking "noise" flattens/handicaps the baselines while leaving
the model untouched; the "advantage" is the baseline degrading, not the model winning. Decoupling the
baseline prior from the smoothing axis: dominant-cell goes dead flat (1.895 at every level) and BEATS
the model throughout — the advantage vanishes. Retracted `advantage_frontier.py` + test + artifact.

Forge also showed the **manuscript long-memory positive (f6694cd) was overclaimed**: at a FIXED
return-rate (no in-sample selection) it clears the battery only at the finest grid (bins=10, 5-6/6)
and just 1/6 one step coarser (bins=8, a razor-thin tie with the pooled Markov). The earlier "5/6 at
bins=8" used a 3-way in-sample rho min — the selection bias Forge flagged. Corrected the manuscript
subsection to the honest negative verdict. Also fixed the `oracle_prev` docstring (it is a fixed
pooled-Markov reference, NOT a strict ceiling; fair models can beat it — no leak).

**THE COMPREHENSIVE, RIGOROUS ANSWER to "where does active inference have an advantage?":** after
confound-controlled testing AND cross-vendor audit, **no robust active-inference advantage survives
on this benchmark.** Every candidate reduces to a confound — oracle access, data density, spatial
confinement, in-sample tuning, or a measurement asymmetry. That rigorous negative result is the map,
and the strongest possible vindication of the benchmark-first repositioning. NO oracle leak remained
(Forge verified the fair paths infer the previous cell). Lesson: a confound can re-enter through the
FAIRNESS OF A KNOB (here the noise axis), not only through oracle access — cross-vendor audit is what
caught it.
