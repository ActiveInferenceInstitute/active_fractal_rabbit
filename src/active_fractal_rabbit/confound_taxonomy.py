"""Confound taxonomy — the project's defining finding, made machine-readable and claim-bound.

Across its development this benchmark repeatedly found an apparent active-inference advantage,
and repeatedly dissolved it to a confound under confound-controlled testing and cross-vendor
audit. That pattern IS the contribution: a map of *why* naive active-inference wins on sparse
synthetic mobility do not survive, organised by the mechanism that produced each false win.

This module turns that map from prose scattered across the revision log into a structured,
verifiable registry. Each entry records the apparent claim, the confound class that explains
it, the diagnostic that caught it, whether a cross-vendor (GPT-5.4 / Forge) audit was the
catcher, and a STATUS. Crucially, each entry carries a verification hook that re-derives its
standing from live repository state — a retracted module's absence, a gate artifact's blocked
status, a code invariant, or (where the event is purely historical) an explicit
documentation-only check that is labelled as weaker. A taxonomy of past mistakes is only
honest if it cannot drift away from the artifacts; the hooks are what bind it.

Nothing here asserts a positive capability. The one surviving fair-by-construction positive
(partial-observability soft marginalization) is recorded as the single non-confound, with the
boundary that it still does not beat the oracle and is not a real-mobility claim.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

CONFOUND_TAXONOMY_SCHEMA = "active-fractal-rabbit-confound-taxonomy-v1"

# Verification kinds, strongest to weakest. The taxonomy reports the breakdown so a reader can
# see how much of the map is re-derived from live state versus recorded from history.
KIND_LIVE_ARTIFACT = "live_artifact"  # re-read a generated gate/analysis artifact
KIND_FILESYSTEM_ABSENCE = "filesystem_absence"  # a retracted module must stay deleted
KIND_CODE_INVARIANT = "code_invariant"  # a guard/property still present in source
KIND_DOCUMENTED = "documented"  # historical event recorded in the revision log only

CONFOUND_CLASSES = (
    "data-density",
    "spatial-confinement",
    "in-sample-selection",
    "knob-measurement-asymmetry",
    "no-signal",
    "family-misclassification",
    "instrumentation-bug",
)


class ConfoundTaxonomyError(RuntimeError):
    """Raised when the taxonomy cannot be assembled (e.g. required artifact missing)."""


@dataclass(frozen=True)
class ConfoundEntry:
    """One dissolved/retracted apparent advantage, or the one surviving positive."""

    entry_id: str
    apparent_claim: str
    confound_class: str  # one of CONFOUND_CLASSES, or "none" for the surviving positive
    diagnostic: str
    status: str
    cross_vendor_caught: bool
    verify: Callable[[Path], dict]


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ConfoundTaxonomyError(f"missing artifact for verification: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Verification hooks — each returns {verified, kind, evidence}.
# --------------------------------------------------------------------------- #
def _verify_noise_frontier_retracted(root: Path) -> dict:
    """The smoothing-as-noise advantage frontier was retracted; its module must stay deleted."""
    module = root / "src" / "active_fractal_rabbit" / "advantage_frontier.py"
    absent = not module.exists()
    return {
        "verified": absent,
        "kind": KIND_FILESYSTEM_ABSENCE,
        "evidence": (
            "advantage_frontier.py is absent (retracted)"
            if absent
            else "advantage_frontier.py is PRESENT — retraction violated"
        ),
    }


def _verify_data_density_at_scale(root: Path) -> dict:
    """At scale on a fixed grid the clean-emission advantage collapses toward/below zero."""
    power = _read_json(root / "output" / "runs" / "power_analysis.json")
    adv = power.get("large_clean_advantage_nats")
    # Primary criterion is the artifact's OWN seed-robust verdict, not a fixed magnitude band:
    # a small positive residual that is not a seed-robust win still counts as "vanished".
    vanishes = bool(power.get("aif_advantage_vanishes_at_scale_clean", False))
    verified = vanishes
    return {
        "verified": bool(verified),
        "kind": KIND_LIVE_ARTIFACT,
        "evidence": (
            f"aif_advantage_vanishes_at_scale_clean={vanishes} (seed-robust verdict; "
            f"large_clean_advantage_nats={adv} nats reported for context)"
        ),
    }


def _verify_individual_vs_population_confinement(root: Path) -> dict:
    """Re-derive the confinement diagnostic from the module that actually produced it.

    The prior hook only checked that ``confound_controls.py`` exists on disk — that module is
    never called by ``individual_vs_population.py``, so its presence proves nothing about this
    entry's claim. The real evidence is (a) the real external-simulator trace is severely
    spatially confined, and (b) on that SAME target traveller and SAME prequential protocol, a
    trivial dominant-cell (pure visit-frequency, ``rho=1.0``) predictor is at least as good as the
    project's individual model (``rho=1.0`` is a strict submodel of the default blend, so this is
    an apples-to-apples same-traveller comparison rather than mixing in ``confound_controls.py``'s
    differently-split, multi-traveller aggregate baseline). If the cached real external-simulator
    trace is not present on this machine (the external lane is optional and requires a local
    JDK/Maven build), this reports unverified rather than reading an unrelated file's mere
    presence as evidence.
    """
    from .config import load_config
    from .confound_controls import mobility_summary
    from .fractalrabbit import parse_waypoint_csv
    from .individual_vs_population import advantage_vs_population_size
    from .observations import discretize_observations

    config_path = root / "configs" / "fractalrabbit_external.yaml"
    if not config_path.is_file():
        return {
            "verified": False,
            "kind": KIND_LIVE_ARTIFACT,
            "evidence": f"external-lane config absent ({config_path}); cannot resolve the real trace",
        }
    cfg = load_config(config_path)
    trace = cfg.output_dir / f"{cfg.fractalrabbit.output_prefix}XY.csv"
    if not trace.is_file():
        return {
            "verified": False,
            "kind": KIND_LIVE_ARTIFACT,
            "evidence": (
                f"real external-simulator trace absent ({trace}); the confinement/dominant-cell "
                "claim cannot be re-derived without a cached external-lane run"
            ),
        }
    sequence = discretize_observations(parse_waypoint_csv(trace), cfg.inference)
    mobility = mobility_summary(sequence)
    confined = not bool(mobility["is_mobile"])
    dominant = advantage_vs_population_size(sequence, smoothing=0.05, rho=1.0)
    model = advantage_vs_population_size(sequence, smoothing=0.05)
    dominant_rows = cast("list[dict[str, Any]]", dominant["rows"])
    model_rows = cast("list[dict[str, Any]]", model["rows"])
    dominant_loss = float(dominant_rows[0]["individual_loss"])
    model_loss = float(model_rows[0]["individual_loss"])
    dominant_cell_beats_model = bool(dominant_loss <= model_loss)
    verified = confined and dominant_cell_beats_model
    return {
        "verified": bool(verified),
        "kind": KIND_LIVE_ARTIFACT,
        "evidence": (
            f"real trace, target traveller {model['target_traveller']}: "
            f"mean_distinct_cells={mobility['mean_distinct_cells']:.2f}, "
            f"confined_fraction={mobility['confined_fraction']:.2f} (is_mobile="
            f"{mobility['is_mobile']}); dominant-cell-only loss={dominant_loss:.4f} nats vs "
            f"individual-model loss={model_loss:.4f} nats (dominant-cell beats model="
            f"{dominant_cell_beats_model})"
        ),
    }


def _verify_documented_in_revision_log(marker: str) -> Callable[[Path], dict]:
    """A historical confound recorded in the revision program; weaker documentation-only check."""

    def _hook(root: Path) -> dict:
        doc = root / "docs" / "revision" / "REVISION_PROGRAM.md"
        text = doc.read_text(encoding="utf-8") if doc.is_file() else ""
        present = marker.lower() in text.lower()
        return {
            "verified": present,
            "kind": KIND_DOCUMENTED,
            "evidence": f"revision log records {marker!r}"
            if present
            else f"{marker!r} not in revision log",
        }

    return _hook


def _verify_partial_obs_surviving_positive(root: Path) -> dict:
    """The one non-confound: a positive, sign-stable advantage under noisy emission."""
    po = _read_json(root / "output" / "runs" / "partial_observability_analysis.json")
    positive = bool(po.get("advantage_positive_under_noise", False))
    sign_stable = bool(po.get("cross_realization_sign_stable_under_noise", False))
    return {
        "verified": positive and sign_stable,
        "kind": KIND_LIVE_ARTIFACT,
        "evidence": (
            f"advantage_positive_under_noise={positive}, "
            f"cross_realization_sign_stable_under_noise={sign_stable}"
        ),
    }


def _verify_regime_clearance_was_an_instrumentation_bug(root: Path) -> dict:
    """The apparent regime-qualified clearance was never a genuine model-level result.

    Session 27 (2026-07-02) root-caused the apparent clearance: ``multiplicity_ledger.py``
    computed ``uncorrected_significant`` from an independently recomputed ``one_sided_p < alpha``
    check instead of reusing the pipeline's own CI-based ``individually_clears`` verdict. On live
    data, ``aif_regime::symmetric_noise_045`` had ``clustered_ci_low = -0.001306`` (the pipeline's
    own gate never cleared it) yet ``one_sided_p_max = 0.033983 < 0.05``, so the ledger reported an
    "uncorrected clearance" the rest of the pipeline never certified. This was not a confound in
    the model-comparison sense (oracle access, data density, spatial confinement, ...) — it was a
    bug in the reporting instrument itself, manufacturing an apparent result with no underlying
    model-level event to explain away. Fixed by making ``uncorrected_significant`` track
    ``individually_clears`` directly (single source of truth); the invariant is regression-locked
    by ``tests/test_multiplicity_ledger.py::test_uncorrected_significant_always_equals_individually_clears``.

    Forge cross-vendor HIGH-2 (session 27): an earlier version of this hook compared the ledger's
    own ``uncorrected_significant``/``individually_clears`` fields against EACH OTHER — but since
    the ledger now derives one directly from the other, that comparison is true by construction on
    any freshly generated ledger and can only ever fire on a hand-tampered file, never on a
    genuinely mis-derived one. This version instead independently RE-DERIVES ``gate_clears`` from
    the raw upstream artifacts (``aif_regime_search.json`` / ``sota_program_gate.json`` — the same
    sources ``enumerate_comparison_family`` reads) and compares that fresh recomputation against
    the ledger's reported ``uncorrected_significant``, so a regression in either the ledger's
    derivation OR the upstream artifact's own ``gate_clears``/``aif_beats_strongest_fair`` field
    would be caught. Does not (and should not) require zero uncorrected clearers — a future
    genuine win that correctly clears both the pipeline gate and the ledger is not a bug.
    """
    ledger_path = root / "output" / "runs" / "multiplicity_ledger.json"
    regime_path = root / "output" / "runs" / "aif_regime_search.json"
    sota_path = root / "output" / "runs" / "sota_program_gate.json"
    if not ledger_path.is_file():
        return {
            "verified": False,
            "kind": KIND_LIVE_ARTIFACT,
            "evidence": "multiplicity_ledger.json absent — instrumentation-bug fix not adjudicated",
        }
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    comparisons = ledger.get("comparisons", [])
    if not comparisons:
        return {
            "verified": False,
            "kind": KIND_LIVE_ARTIFACT,
            "evidence": "multiplicity_ledger.json has zero comparisons — cannot re-derive the invariant",
        }
    independent_gate_clears: dict[str, bool] = {}
    if regime_path.is_file():
        regime = json.loads(regime_path.read_text(encoding="utf-8"))
        for regime_id, summary in regime.get("regime_summaries", {}).items():
            if isinstance(summary, dict):
                independent_gate_clears[f"aif_regime::{regime_id}"] = bool(
                    summary.get("gate_clears", False)
                )
    if sota_path.is_file():
        sota = json.loads(sota_path.read_text(encoding="utf-8"))
        for lane_id, lane in sota.get("lanes", {}).items():
            if isinstance(lane, dict):
                independent_gate_clears[f"sota_lane::{lane_id}"] = bool(
                    lane.get("aif_beats_strongest_fair", False)
                )
    checked = 0
    mismatches = []
    for c in comparisons:
        cid = c.get("comparison_id")
        if cid not in independent_gate_clears:
            continue  # upstream artifact missing this id — not adjudicable, not a mismatch
        checked += 1
        if bool(c.get("uncorrected_significant")) != independent_gate_clears[cid]:
            mismatches.append(cid)
    verified = checked > 0 and not mismatches
    return {
        "verified": verified,
        "kind": KIND_LIVE_ARTIFACT,
        "evidence": (
            f"{len(comparisons)} ledger comparison(s), {checked} independently re-derived from "
            f"raw upstream artifacts; {len(mismatches)} mismatch(es) between the ledger's "
            f"uncorrected_significant and a fresh gate_clears/aif_beats_strongest_fair "
            f"recomputation ({mismatches})"
        ),
    }


def _verify_tempered_filter_is_aif_family(root: Path) -> dict:
    """The 'tempered soft filter' is provably an AIF-family predictor, not a fair baseline.

    A `/workflows` prototype proposed a TRAIN-fit tempered soft filter, predicting
    ``T @ normalize(belief ** beta)``, as a *fair* baseline that beats the active-inference
    candidate. But ``normalize(b ** gamma) == softmax(gamma * log b)`` is exactly the precision
    weighting the AIF candidate applies (``_precision_weight_belief``), so the tempered filter is
    that candidate with ``boost == 0`` — same family. The catch rests on a live code identity:
    confirm ``sota_aif_candidate.py`` precision-weights the belief as ``b ** gamma`` renormalised.
    """
    src = (root / "src" / "active_fractal_rabbit" / "sota_aif_candidate.py").read_text(
        encoding="utf-8"
    )
    has_identity = "b ** gamma" in src and "softmax(gamma * log b)" in src
    return {
        "verified": bool(has_identity),
        "kind": KIND_CODE_INVARIANT,
        "evidence": (
            "sota_aif_candidate precision-weights belief as b**gamma == softmax(gamma*log b); "
            "a tempered soft filter T @ normalize(b**beta) is therefore the AIF candidate with "
            f"boost=0, not a fair baseline (identity present={has_identity})"
        ),
    }


def build_confound_registry() -> list[ConfoundEntry]:
    """The curated, claim-bound registry of dissolved advantages plus the one survivor."""

    return [
        ConfoundEntry(
            entry_id="noise_frontier",
            apparent_claim="A clean, monotone observation-noise frontier showed an active-inference "
            "advantage growing with emission smoothing across all densities.",
            confound_class="knob-measurement-asymmetry",
            diagnostic="The smoothing 'noise' knob was wired into the baselines' visit prior but not "
            "the model's return prior; decoupling it flattens the baselines and the advantage vanishes "
            "(dominant-cell goes flat and beats the model throughout).",
            status="retracted",
            cross_vendor_caught=True,
            verify=_verify_noise_frontier_retracted,
        ),
        ConfoundEntry(
            entry_id="long_memory_in_sample",
            apparent_claim="A long-memory / preferential-return model cleared the confound battery "
            "5/6 seeds on a mobility-checked finer grid.",
            confound_class="in-sample-selection",
            diagnostic="The 5/6 used an in-sample 3-way return-rate minimum; at a fixed return rate it "
            "clears only at the finest grid and ties the pooled Markov one step coarser.",
            status="corrected-to-negative",
            cross_vendor_caught=True,
            verify=_verify_documented_in_revision_log("long-memory"),
        ),
        ConfoundEntry(
            entry_id="multiscale_backoff",
            apparent_claim="A multiscale / hierarchical backoff model would add return signal.",
            confound_class="no-signal",
            diagnostic="It reduces to a first-order pooled Markov plus coarse smoothing and adds no "
            "return signal; clears 0/6 on both grids.",
            status="null",
            cross_vendor_caught=False,
            verify=_verify_documented_in_revision_log("Multiscale"),
        ),
        ConfoundEntry(
            entry_id="individual_vs_population_confinement",
            apparent_claim="On a real external-simulator trace an individual model beat the pooled "
            "population transition by up to ~2 nats.",
            confound_class="spatial-confinement",
            diagnostic="Travellers are severely spatially confined; a trivial dominant-cell predictor "
            "beats the model. The win is information-set congruence, not active inference.",
            status="reduced-to-confound",
            cross_vendor_caught=False,
            verify=_verify_individual_vs_population_confinement,
        ),
        ConfoundEntry(
            entry_id="fair_frontier_scale_fragility",
            apparent_claim="The fair-frontier active-inference advantage held 8/8 seeds on the canonical "
            "fixture.",
            confound_class="data-density",
            diagnostic="Scaling 4->6 travellers (denser per-cell occupancy) collapses the advantage to "
            "the noise floor as the dense emission posterior makes the hard point estimate as good.",
            status="scale-fragile",
            cross_vendor_caught=False,
            verify=_verify_data_density_at_scale,
        ),
        ConfoundEntry(
            entry_id="regime_qualified_clearance",
            apparent_claim="The AIF regime search clears a regime-qualified synthetic SOTA in one "
            "symmetric-noise regime.",
            confound_class="instrumentation-bug",
            diagnostic="Not a model-level confound: the multiplicity ledger's own "
            "uncorrected_significant field was computed from an independently re-derived p<alpha "
            "check instead of the pipeline's CI-based individually_clears verdict, so a regime "
            "whose own confidence interval never separated from zero (symmetric_noise_045, "
            "clustered_ci_low=-0.001306) was still reported as an 'uncorrected clearance'. Fixed "
            "2026-07-02 (session 27) by making the two fields a single source of truth; the "
            "corrected family shows zero uncorrected clearers, not one demoted by Holm.",
            status="bug-fixed",
            cross_vendor_caught=False,
            verify=_verify_regime_clearance_was_an_instrumentation_bug,
        ),
        ConfoundEntry(
            entry_id="tempered_filter_relabeled_fair",
            apparent_claim="A train-fit tempered soft filter, scored as a fair baseline, beats the "
            "active-inference candidate on the FractalRabbit lane (a stronger fair baseline dethrones AIF).",
            confound_class="family-misclassification",
            diagnostic="The tempered soft filter predicts T @ normalize(belief**beta) = T @ "
            "softmax(beta*log belief) — precision-weighted recursive belief, identical to the AIF "
            "candidate with boost=0. It is an AIF-family variant, not a fair non-AIF baseline; the honest "
            "residual is AIF-internal (the return boost overfits on that lane), not a fair-baseline win. "
            "A /workflows prototype proposed it; the code identity dissolved it before any benchmark claim.",
            status="caught-pre-build",
            cross_vendor_caught=False,
            verify=_verify_tempered_filter_is_aif_family,
        ),
        ConfoundEntry(
            entry_id="partial_observability_positive",
            apparent_claim="Soft Bayesian marginalization beats a hard point estimate under a noisy "
            "emission and genuine partial observability.",
            confound_class="none",
            diagnostic="Fair by construction (shared transition/emission/observations/priors; the only "
            "difference is marginalize-vs-commit) and sign-stable across independent draws. The single "
            "non-confound — but it does not beat the oracle and is not a real-mobility claim.",
            status="surviving-positive",
            cross_vendor_caught=False,
            verify=_verify_partial_obs_surviving_positive,
        ),
    ]


def build_confound_taxonomy(project_root: str | Path) -> dict:
    """Run every verification hook and assemble the taxonomy payload from live state."""

    root = Path(project_root)
    registry = build_confound_registry()

    rows: list[dict] = []
    for entry in registry:
        result = entry.verify(root)
        rows.append(
            {
                "entry_id": entry.entry_id,
                "apparent_claim": entry.apparent_claim,
                "confound_class": entry.confound_class,
                "diagnostic": entry.diagnostic,
                "status": entry.status,
                "cross_vendor_caught": entry.cross_vendor_caught,
                "verification_kind": result["kind"],
                "verified": bool(result["verified"]),
                "verification_evidence": result["evidence"],
            }
        )

    confound_rows = [r for r in rows if r["confound_class"] != "none"]
    classes_present = sorted({r["confound_class"] for r in confound_rows})
    n_cross_vendor = sum(1 for r in rows if r["cross_vendor_caught"])
    n_verified = sum(1 for r in rows if r["verified"])
    kind_counts: dict[str, int] = {}
    for r in rows:
        kind_counts[r["verification_kind"]] = (
            kind_counts.get(r["verification_kind"], 0) + 1
        )
    # Split the verified tally so a documentation-only grep is never read as live verification.
    _live_kinds = {KIND_LIVE_ARTIFACT, KIND_FILESYSTEM_ABSENCE, KIND_CODE_INVARIANT}
    n_verified_live_state = sum(
        1 for r in rows if r["verified"] and r["verification_kind"] in _live_kinds
    )
    n_verified_documented = sum(
        1 for r in rows if r["verified"] and r["verification_kind"] == KIND_DOCUMENTED
    )
    n_unverified = len(rows) - n_verified

    verdict = (
        f"Of {len(confound_rows)} apparent active-inference advantages catalogued, "
        f"{len(confound_rows) - n_unverified} are currently verified as dissolved to one of "
        f"{len(classes_present)} confound classes "
        f"({', '.join(classes_present)}); {n_cross_vendor} were caught only by cross-vendor audit. "
        f"Of the {len(rows)} entries, {n_verified_live_state} are verified against live state "
        f"(artifacts, code invariants, or a retracted module's absence) and {n_verified_documented} "
        "against the revision log only (documentation-grade, the weaker kind); "
        f"{n_unverified} remain explicitly unverified until their required current artifacts exist. "
        "Exactly one fair-by-construction positive survives (partial-observability soft "
        "marginalization), and it neither beats the oracle nor makes a real-mobility claim."
    )

    return {
        "schema": CONFOUND_TAXONOMY_SCHEMA,
        "confound_classes": list(CONFOUND_CLASSES),
        "n_entries": len(rows),
        "n_confounds": len(confound_rows),
        "n_confound_classes_present": len(classes_present),
        "confound_classes_present": classes_present,
        "n_cross_vendor_caught": n_cross_vendor,
        "n_verified": n_verified,
        "n_verified_live_state": n_verified_live_state,
        "n_verified_documented": n_verified_documented,
        "n_unverified": n_unverified,
        "verification_kind_counts": kind_counts,
        "entries": rows,
        "verdict": verdict,
        "boundary": (
            "A map of why naive active-inference wins on sparse synthetic mobility do not survive. "
            "Verification hooks bind each entry to live state; documentation-only entries are "
            "labelled as such. No real-mobility, operational, or surveillance validity is implied."
        ),
    }


def run_and_write_confound_taxonomy(
    project_root: str | Path,
    out_dir: str | Path | None = None,
) -> Path:
    """Build the taxonomy and write ``confound_taxonomy.json`` under ``out_dir``."""

    root = Path(project_root)
    payload = build_confound_taxonomy(root)
    target_dir = Path(out_dir) if out_dir is not None else root / "output" / "runs"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "confound_taxonomy.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
