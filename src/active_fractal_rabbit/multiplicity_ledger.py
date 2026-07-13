"""Multiplicity ledger — family-wise and false-discovery correction across the benchmark.

The benchmark runs many directional "does active inference beat the strongest fair
baseline here?" comparisons — one per synthetic regime in the AIF regime search and one per
lane in the cross-generator SOTA program. Each is individually scored at an uncorrected
threshold (a traveller-clustered bootstrap CI that clears zero). With that many comparisons,
a single clearance is exactly the situation multiplicity correction exists to interrogate:
how many clearances would survive once the size of the comparison family is priced in?

This module answers that question without introducing a new researcher degree of freedom.
The comparison family is enumerated by a DETERMINISTIC RULE from the live artifacts — every
record that already reports a one-sided clustered-bootstrap p-value for an AIF-vs-fair
directional comparison — never hand-picked. Selecting the family by hand would itself be the
kind of confound this project has repeatedly had to retract; the enumeration rule removes
that freedom. Two corrections are reported:

* **Holm–Bonferroni** (step-down) is the PRIMARY control. It bounds the family-wise error
  rate under arbitrary dependence between the comparisons, which is the honest assumption
  here: the regimes and lanes share generators, candidate code, and the AIF fit, so positive
  dependence cannot be assumed.
* **Benjamini–Hochberg** (step-up) is reported as a SECONDARY, more powerful control of the
  false-discovery rate. BH is valid under independence or positive regression dependence
  (PRDS); that assumption is stated explicitly and not claimed to hold.

The one fair-by-construction positive — the partial-observability soft-marginalization
advantage — is established by a DIFFERENT and stricter criterion (unanimous sign-stability
across independent generator draws), so it is reported separately rather than folded into the
bootstrap family. A replication criterion that requires every one of ``D`` independent draws
to agree is multiplicity-robust by construction; its sign-test bound ``0.5 ** D`` is compared
against the Bonferroni per-comparison threshold for the whole family.

Nothing here is a real-mobility claim, a state-of-the-art claim, or an oracle-beating claim;
it is a multiplicity audit of synthetic software measurements on the pinned generator.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

MULTIPLICITY_LEDGER_SCHEMA = "active-fractal-rabbit-multiplicity-ledger-v1"

DEFAULT_ALPHA = 0.05
DEFAULT_FDR_Q = 0.10

# Pinned methodological commitment: a regime's per-comparison p is the WORST (largest) p over
# its realizations, so a regime "wins" at the uncorrected threshold only when even its least-
# significant realization does — matching the gate's all-realizations-clear logic. This is the
# conservative (anti-win) choice. Switching to "one_sided_p_min" would silently lower the lone
# clearer's p and could MANUFACTURE a Holm survivor, flipping the headline; the constant + the
# test that pins it make any such change a visible, deliberate methodology edit, never a typo.
REGIME_P_FIELD = "one_sided_p_max"


class MultiplicityLedgerError(RuntimeError):
    """Raised when a required comparison artifact is missing or malformed."""


@dataclass(frozen=True)
class Comparison:
    """One directional AIF-vs-fair comparison drawn from the live artifacts."""

    comparison_id: str
    family: str  # "aif_regime" | "sota_lane"
    description: str
    mean_advantage_nats: (
        float  # loss(strongest_fair) - loss(aif); positive => AIF better
    )
    clustered_ci_low: float
    one_sided_p: float  # exact one-sided clustered-bootstrap p for H1: AIF better
    individually_clears: bool  # the artifact's own uncorrected directional verdict
    n_holdout_travellers: int | None
    n_boot: int | None = (
        None  # bootstrap replicates behind that p (None if artifact predates the field)
    )
    # Additive completeness (never gates a verdict). ``clustered_ci_high`` is the coherent
    # same-comparison bootstrap upper edge for a SOTA lane (paired with clustered_ci_low, both
    # from one lane's own resample). ``min_advantage_nats`` is a regime's worst-realization
    # advantage — the honest conjunction floor of the all-realizations-clear gate. Both default
    # None so an artifact predating the upstream field cannot crash enumeration.
    clustered_ci_high: float | None = None  # sota_lane only
    min_advantage_nats: float | None = None  # aif_regime only


def holm_adjusted_pvalues(pvalues: list[float]) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values, returned in the INPUT order.

    For ``m`` p-values sorted ascending ``p_(1) <= ... <= p_(m)``, the adjusted value at sorted
    rank ``i`` (1-based) is ``max_{k<=i} min(1, (m - k + 1) * p_(k))`` — the running maximum of
    the step-down multipliers, which enforces the required monotonicity. A hypothesis is
    rejected at level ``alpha`` iff its adjusted p-value ``<= alpha``; this controls the
    family-wise error rate under ARBITRARY dependence. Idempotent on an empty list.
    """

    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted_sorted: list[float] = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):  # rank is 0-based
        multiplier = m - rank  # = m - (rank+1) + 1
        value = min(1.0, multiplier * pvalues[idx])
        running_max = max(running_max, value)
        adjusted_sorted[rank] = running_max
    result = [0.0] * m
    for rank, idx in enumerate(order):
        result[idx] = adjusted_sorted[rank]
    return result


def benjamini_hochberg_adjusted_pvalues(pvalues: list[float]) -> list[float]:
    """Benjamini–Hochberg step-up adjusted p-values (q-values), in the INPUT order.

    For ``m`` p-values sorted ascending, the adjusted value at sorted rank ``i`` (1-based) is
    ``min_{k>=i} min(1, (m / k) * p_(k))`` — the running minimum taken from the largest rank
    downward, enforcing the monotone non-decreasing shape BH requires. A hypothesis is rejected
    at FDR level ``q`` iff its adjusted value ``<= q``. Valid under independence or positive
    regression dependence (PRDS); that assumption is the caller's to justify.
    """

    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted_sorted: list[float] = [0.0] * m
    running_min = 1.0
    for rank in range(m - 1, -1, -1):  # 0-based, from largest p downward
        idx = order[rank]
        value = min(1.0, (m / (rank + 1)) * pvalues[idx])
        running_min = min(running_min, value)
        adjusted_sorted[rank] = running_min
    result = [0.0] * m
    for rank, idx in enumerate(order):
        result[idx] = adjusted_sorted[rank]
    return result


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise MultiplicityLedgerError(f"missing comparison artifact: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise MultiplicityLedgerError(f"malformed artifact {path}: {exc}") from exc
    if not isinstance(data, dict):  # pragma: no cover - defensive
        raise MultiplicityLedgerError(f"artifact {path} is not a JSON object")
    return data


def enumerate_comparison_family(runs_dir: str | Path) -> list[Comparison]:
    """Deterministically enumerate every AIF-vs-fair comparison that reports a bootstrap p.

    The rule (no hand-picking): one comparison per regime in ``aif_regime_search.json`` that
    exposes ``one_sided_p_max``, plus one comparison per lane in ``sota_program_gate.json``
    that exposes ``one_sided_p``. A regime or lane that does not carry the p-value field is
    omitted with no silent default — the enumerator must FIRE (drop the comparison) so a stale
    artifact cannot inflate the family with placeholder significance.
    """

    runs = Path(runs_dir)
    comparisons: list[Comparison] = []

    regime = _read_json(runs / "aif_regime_search.json")
    regime_run_parameters = regime.get("run_parameters", {})
    regime_n_boot = (
        int(regime_run_parameters["n_boot"])
        if isinstance(regime_run_parameters, dict) and "n_boot" in regime_run_parameters
        else None
    )
    regime_summaries = regime.get("regime_summaries", {})
    if not isinstance(regime_summaries, dict):  # pragma: no cover - defensive
        raise MultiplicityLedgerError(
            "aif_regime_search.json regime_summaries malformed"
        )
    for regime_id in sorted(regime_summaries):
        summary = regime_summaries[regime_id]
        if not isinstance(summary, dict) or REGIME_P_FIELD not in summary:
            continue  # enumerator fires: no bootstrap p for this regime
        comparisons.append(
            Comparison(
                comparison_id=f"aif_regime::{regime_id}",
                family="aif_regime",
                description=f"AIF beats every fair baseline in synthetic regime {regime_id}",
                mean_advantage_nats=float(summary.get("mean_advantage_nats", 0.0)),
                clustered_ci_low=float(summary.get("min_clustered_ci_low", 0.0)),
                one_sided_p=float(summary[REGIME_P_FIELD]),
                individually_clears=bool(summary.get("gate_clears", False)),
                n_holdout_travellers=None,
                n_boot=regime_n_boot,
                min_advantage_nats=(
                    float(summary["min_advantage_nats"])
                    if "min_advantage_nats" in summary
                    else None
                ),
            )
        )

    sota = _read_json(runs / "sota_program_gate.json")
    sota_run_parameters = sota.get("run_parameters", {})
    sota_n_boot = (
        int(sota_run_parameters["n_boot"])
        if isinstance(sota_run_parameters, dict) and "n_boot" in sota_run_parameters
        else None
    )
    lanes = sota.get("lanes", {})
    if not isinstance(lanes, dict):  # pragma: no cover - defensive
        raise MultiplicityLedgerError("sota_program_gate.json lanes malformed")
    for lane_id in sorted(lanes):
        lane = lanes[lane_id]
        if not isinstance(lane, dict) or "one_sided_p" not in lane:
            continue  # enumerator fires: no bootstrap p for this lane
        comparisons.append(
            Comparison(
                comparison_id=f"sota_lane::{lane_id}",
                family="sota_lane",
                description=f"AIF beats the strongest fair baseline on SOTA lane {lane_id}",
                mean_advantage_nats=float(
                    lane.get("mean_difference_nats_fair_minus_aif", 0.0)
                ),
                clustered_ci_low=float(lane.get("clustered_ci_low", 0.0)),
                one_sided_p=float(lane["one_sided_p"]),
                individually_clears=bool(lane.get("aif_beats_strongest_fair", False)),
                n_holdout_travellers=(
                    int(lane["n_holdout_travellers"])
                    if "n_holdout_travellers" in lane
                    else None
                ),
                n_boot=sota_n_boot,
                clustered_ci_high=(
                    float(lane["clustered_ci_high"])
                    if "clustered_ci_high" in lane
                    else None
                ),
            )
        )

    if not comparisons:
        raise MultiplicityLedgerError(
            "no AIF-vs-fair comparisons exposed a one-sided bootstrap p-value; "
            "regenerate aif_regime_search.json and sota_program_gate.json"
        )
    return comparisons


def _partial_observability_entry(
    runs_dir: Path, *, family_size: int, alpha: float
) -> dict:
    """The fair-by-construction positive, evaluated under its own replication criterion."""

    path = runs_dir / "partial_observability_analysis.json"
    data = _read_json(path)
    n_draws = int(data.get("cross_realization_n_draws", 0))
    sign_stable = bool(data.get("cross_realization_sign_stable_under_noise", False))
    min_fraction_positive = float(
        data.get("cross_realization_min_fraction_positive_under_noise", 0.0)
    )
    max_noise_advantage = float(data.get("max_noise_advantage_nats", 0.0))
    # Sign-test bound for unanimous direction across D independent draws under H0 (each draw
    # 50/50): 0.5 ** D. This is conservative — it ignores the within-draw resampling evidence.
    sign_test_bound = 0.5**n_draws if n_draws > 0 else 1.0
    # The Bonferroni per-comparison threshold for a family this size. If the replication bound
    # clears it, the positive survives even the most conservative family-wide control.
    bonferroni_threshold = alpha / family_size if family_size > 0 else alpha
    # How many unanimous draws the sign-test alone would need to clear that threshold —
    # an actionable, derived target, not a claim that the result currently requires it.
    min_draws = (
        int(math.ceil(math.log(bonferroni_threshold) / math.log(0.5)))
        if 0.0 < bonferroni_threshold < 1.0
        else None
    )
    return {
        "in_inferential_bootstrap_family": False,
        "criterion": "unanimous sign-stability across independent generator draws",
        "n_independent_draws": n_draws,
        "sign_stable_under_noise": sign_stable,
        "min_within_draw_fraction_positive": round(min_fraction_positive, 6),
        "max_noise_advantage_nats": round(max_noise_advantage, 6),
        "sign_test_bound": round(sign_test_bound, 8),
        "bonferroni_per_comparison_threshold": round(bonferroni_threshold, 8),
        "survives_family_bonferroni": bool(
            sign_stable and n_draws > 0 and sign_test_bound <= bonferroni_threshold
        ),
        "min_draws_for_family_bonferroni": min_draws,
        "validity_note": (
            "This positive is fair BY CONSTRUCTION — soft marginalization is mathematically at "
            "least as good as a hard point estimate under a known noisy emission, with a strictly "
            "positive advantage as the channel degrades — so its validity is a mechanism-level "
            "property, not a frequentist sign test. The sign-test bound is a conservative cross-"
            "check; clearing the family-wide Bonferroni threshold would take "
            f"{min_draws if min_draws is not None else 'NA'} unanimous draws."
        ),
        "boundary": (
            "Established by replication across independent synthetic draws under a known "
            "emission; multiplicity-robust by construction, but not an oracle-beating, "
            "state-of-the-art, or real-mobility claim."
        ),
    }


def build_multiplicity_ledger(
    project_root: str | Path,
    *,
    alpha: float = DEFAULT_ALPHA,
    fdr_q: float = DEFAULT_FDR_Q,
) -> dict:
    """Assemble the multiplicity-corrected ledger payload from live artifacts."""

    root = Path(project_root)
    runs = root / "output" / "runs"
    comparisons = enumerate_comparison_family(runs)
    comparisons = sorted(comparisons, key=lambda c: c.comparison_id)

    pvalues = [c.one_sided_p for c in comparisons]
    holm = holm_adjusted_pvalues(pvalues)
    bh = benjamini_hochberg_adjusted_pvalues(pvalues)

    rows: list[dict] = []
    for comp, holm_p, bh_p in zip(comparisons, holm, bh):
        rows.append(
            {
                "comparison_id": comp.comparison_id,
                "family": comp.family,
                "description": comp.description,
                "mean_advantage_nats": round(comp.mean_advantage_nats, 6),
                "clustered_ci_low": round(comp.clustered_ci_low, 6),
                "uncorrected_p": round(comp.one_sided_p, 6),
                "individually_clears": comp.individually_clears,
                # Single source of truth for "clears before correction": the pipeline's own
                # CI-based gate verdict (comp.individually_clears), NOT an independently
                # recomputed one_sided_p < alpha check. The two constructs are not guaranteed
                # to agree — clustered_ci_low is a percentile bound (effectively a stricter
                # one-sided alpha here) while one_sided_p < alpha uses a looser threshold — so
                # a comparison whose own CI never separated from zero could still register as
                # an "uncorrected clearance" under the old definition. Bug found + fixed
                # 2026-07-02 (session 27): confirmed on live data that aif_regime::
                # symmetric_noise_045 had clustered_ci_low < 0 (individually_clears=False) yet
                # scored uncorrected_significant=True, manufacturing a clearance the pipeline's
                # own decision rule never produced.
                "uncorrected_significant": bool(comp.individually_clears),
                "holm_adjusted_p": round(holm_p, 6),
                # A comparison can only "survive correction" if it cleared to begin with —
                # correction promotes from an already-cleared uncorrected result, it cannot
                # resurrect one that never cleared. Gating on individually_clears here closes
                # the mirror-image of the uncorrected_significant bug above (Forge HIGH-1,
                # session 27): without this gate, a comparison whose CI never separated from
                # zero could still register holm_survives=True purely from a small raw p-value,
                # producing the self-contradictory "survives multiplicity correction but never
                # cleared uncorrected" state. Regression-locked by
                # test_holm_bh_survival_requires_individually_clears.
                "holm_survives": bool(holm_p <= alpha and comp.individually_clears),
                "bh_adjusted_p": round(bh_p, 6),
                "bh_survives": bool(bh_p <= fdr_q and comp.individually_clears),
                "n_holdout_travellers": comp.n_holdout_travellers,
                "n_boot": comp.n_boot,
                # the smallest reportable Davison-Hinkley p at this B; documents
                # the granularity behind six-decimal adjusted p-values.
                "p_resolution_floor": (
                    round(1.0 / (comp.n_boot + 1), 6) if comp.n_boot else None
                ),
            }
        )
        # Additive per-row completeness (never a verdict input). SOTA lanes carry a coherent
        # same-comparison bootstrap upper edge; regimes carry the worst-realization advantage
        # (the honest conjunction floor). A regime clustered_ci_high is DELIBERATELY omitted:
        # min_clustered_ci_low and max_clustered_ci_high come from DIFFERENT realizations, so
        # pairing them would imply a false single-comparison interval.
        if comp.family == "sota_lane":
            rows[-1]["clustered_ci_high"] = (
                round(comp.clustered_ci_high, 6)
                if comp.clustered_ci_high is not None
                else None
            )
        if comp.family == "aif_regime":
            rows[-1]["min_advantage_nats"] = (
                round(comp.min_advantage_nats, 6)
                if comp.min_advantage_nats is not None
                else None
            )

    family_size = len(comparisons)
    n_individually_clearing = sum(1 for r in rows if r["individually_clears"])
    n_uncorrected_significant = sum(1 for r in rows if r["uncorrected_significant"])
    n_holm_survivors = sum(1 for r in rows if r["holm_survives"])
    n_bh_survivors = sum(1 for r in rows if r["bh_survives"])
    holm_survivor_ids = [r["comparison_id"] for r in rows if r["holm_survives"]]
    bh_survivor_ids = [r["comparison_id"] for r in rows if r["bh_survives"]]

    # Keypath-addressable headline summary for manuscript tokens. The "clearer" is a comparison
    # that clears the UNCORRECTED threshold; its Holm-adjusted p is the punchline (does the lone
    # apparent win survive multiplicity?). max over clearers is the most conservative.
    clearer_rows = [r for r in rows if r["uncorrected_significant"]]
    headline = {
        "n_uncorrected_clearers": len(clearer_rows),
        "any_holm_survivor": bool(n_holm_survivors > 0),
        "max_clearer_holm_adjusted_p": (
            round(max(r["holm_adjusted_p"] for r in clearer_rows), 6)
            if clearer_rows
            else None
        ),
        "clearer_ids": [r["comparison_id"] for r in clearer_rows],
        # effect sizes travel with verdicts: the magnitude of the lone apparent
        # win (and the family's range) is part of the honest story, derived
        # here so prose never hand-transcribes it.
        "max_clearer_mean_advantage_nats": (
            round(max(r["mean_advantage_nats"] for r in clearer_rows), 6)
            if clearer_rows
            else None
        ),
        "max_clearer_ci_low": (
            round(max(r["clustered_ci_low"] for r in clearer_rows), 6)
            if clearer_rows
            else None
        ),
        "family_min_advantage_nats": round(
            min(r["mean_advantage_nats"] for r in rows), 6
        ),
        "family_max_advantage_nats": round(
            max(r["mean_advantage_nats"] for r in rows), 6
        ),
        "min_lane_holdout_travellers": (
            min(
                r["n_holdout_travellers"]
                for r in rows
                if r["n_holdout_travellers"] is not None
            )
            if any(r["n_holdout_travellers"] is not None for r in rows)
            else None
        ),
        "min_n_boot": (
            min(r["n_boot"] for r in rows if r["n_boot"])
            if any(r["n_boot"] for r in rows)
            else None
        ),
        "max_p_resolution_floor": (
            max(r["p_resolution_floor"] for r in rows if r["p_resolution_floor"])
            if any(r["p_resolution_floor"] for r in rows)
            else None
        ),
    }

    partial_obs = _partial_observability_entry(
        runs, family_size=family_size, alpha=alpha
    )

    # Honest, derived verdict — never hand-transcribed. Describes only what the corrections show.
    if n_holm_survivors == 0:
        bootstrap_clause = (
            f"none of the {family_size} clustered-bootstrap directional comparisons survive "
            f"Holm correction at alpha={alpha}"
        )
    else:
        bootstrap_clause = (
            f"{n_holm_survivors} of {family_size} clustered-bootstrap directional comparisons "
            f"survive Holm correction at alpha={alpha} ({', '.join(holm_survivor_ids)})"
        )
    partial_clause = (
        "the partial-observability positive is established by unanimous sign-stability across "
        f"{partial_obs['n_independent_draws']} independent draws and "
        + (
            "survives"
            if partial_obs["survives_family_bonferroni"]
            else "does not clear"
        )
        + " the family-wide Bonferroni per-comparison threshold"
    )
    verdict = (
        f"Under multiplicity correction across the benchmark, {bootstrap_clause}; "
        f"{n_individually_clearing} comparison(s) clear at the uncorrected threshold. "
        f"Separately, {partial_clause}. All quantities are synthetic-generator software "
        "measurements; none is an oracle-beating, state-of-the-art, or real-mobility claim."
    )

    return {
        "schema": MULTIPLICITY_LEDGER_SCHEMA,
        "alpha": alpha,
        "fdr_q": fdr_q,
        # Factual descriptors of how each comparison's clustered_ci_low is constructed: a
        # two-sided 95% percentile bootstrap interval whose LOWER edge is the 2.5th percentile.
        # Documentation only — no verdict reads these; the clearance rule is the pipeline's own
        # gate verdict (individually_clears), unchanged.
        "ci_level": 0.95,
        "ci_lower_percentile": 2.5,
        "family_definition": (
            "deterministic enumeration of every AIF-vs-fair comparison that reports a one-sided "
            "clustered-bootstrap p-value (regimes in aif_regime_search.json + lanes in "
            "sota_program_gate.json); the family is not hand-selected"
        ),
        "family_size": family_size,
        "n_individually_clearing": n_individually_clearing,
        "n_uncorrected_significant": n_uncorrected_significant,
        "holm": {
            "control": "family-wise error rate under arbitrary dependence",
            "n_survivors": n_holm_survivors,
            "survivor_ids": holm_survivor_ids,
        },
        "benjamini_hochberg": {
            "control": "false-discovery rate; valid under independence or PRDS (assumption stated, not claimed)",
            "q": fdr_q,
            "n_survivors": n_bh_survivors,
            "survivor_ids": bh_survivor_ids,
        },
        "headline": headline,
        "comparisons": rows,
        "partial_observability_positive": partial_obs,
        "verdict": verdict,
        "boundary": (
            "Multiplicity audit of synthetic software measurements on the pinned generator. "
            "Correction is over the directional AIF-vs-fair comparison family only; the "
            "partial-observability positive uses a stricter replication criterion reported "
            "separately. No oracle-beating, state-of-the-art, or real-mobility claim."
        ),
    }


def run_and_write_multiplicity_ledger(
    project_root: str | Path,
    out_dir: str | Path | None = None,
    *,
    alpha: float = DEFAULT_ALPHA,
    fdr_q: float = DEFAULT_FDR_Q,
) -> Path:
    """Build the ledger and write ``multiplicity_ledger.json`` under ``out_dir``."""

    root = Path(project_root)
    payload = build_multiplicity_ledger(root, alpha=alpha, fdr_q=fdr_q)
    target_dir = Path(out_dir) if out_dir is not None else root / "output" / "runs"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "multiplicity_ledger.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
