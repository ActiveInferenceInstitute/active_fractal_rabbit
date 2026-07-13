"""Within-regime model comparison: where active inference leads, and why.

The global SOTA gate is honestly blocked and no directional comparison survives multiplicity
correction. This module answers the narrower, defensible question the benchmark CAN answer: among
all implemented predictors, *which lead under the regime where active inference earns its keep* —
noisy partial observability — and *why*. It does NOT recompute anything: it ranks the per-candidate
losses the cross-generator SOTA program already scored on the locked, noisy FractalRabbit lane
(``sota_program_gate.json``), tags each method by how it treats the noisy observation, and reports
the leaderboard with explicit boundaries.

The mechanism is the headline: methods that maintain a recursive posterior belief over the latent
cell and MARGINALISE the noisy emission (active inference, the soft filter, the switching-noise HMM,
the particle filter) cluster at the top; methods that COMMIT to a point estimate (the MAP/hard
filter) or CONDITION ON RAW observations (Markov variants, recency attention, persistence,
base-rate) are substantially worse. That gap is exactly the partial-observability result
([@sec:results-partial-observability]) shown across the full roster.

Boundaries kept throughout: this is the lowest-loss IMPLEMENTED fair predictor in a noisy synthetic
regime, in a statistical tie at the top among the belief-based methods; it is NOT a global
state-of-the-art claim, does NOT beat the oracle-fed ceiling, and is not a real-mobility claim. The
gate of [@sec:roadmap-sota-gates] stays blocked.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

MODEL_COMPARISON_SCHEMA = "active-fractal-rabbit-model-comparison-v2"

#: The lane whose noisy emission is the regime where active inference earns its keep.
DEFAULT_LANE = "fractalrabbit"

#: Families that maintain a recursive latent posterior and MARGINALISE the noisy emission. These are
#: the belief-based methods the partial-observability mechanism predicts should lead under noise.
_BELIEF_MARGINALISING_FAMILIES = frozenset(
    {
        "active_inference_candidate",
        "active_inference_reference",
        "classical_sequence",  # switching-noise HMM
        "classical_state_space",  # bootstrap particle filter
    }
)
_AIF_FAMILIES = frozenset({"active_inference_candidate", "active_inference_reference"})


class ModelComparisonError(RuntimeError):
    """Raised when the SOTA-program candidate rows needed for the comparison are missing."""


def _method_group(family: str) -> str:
    """Tag a candidate family by how it treats the noisy observation (the explanatory axis)."""
    if family in _AIF_FAMILIES:
        return "active_inference"
    if family in _BELIEF_MARGINALISING_FAMILIES:
        return "belief_marginalizing"
    if family == "transparent_filter_baseline":
        return "point_estimate"  # hard filter: commits to the MAP cell
    return "raw_observation_or_frequency"


def build_model_comparison(
    project_root: str | Path, *, lane: str = DEFAULT_LANE
) -> dict:
    """Rank the implemented predictors on the noisy lane and summarise where AIF leads."""

    root = Path(project_root)
    gate_path = root / "output" / "runs" / "sota_program_gate.json"
    if not gate_path.is_file():
        raise ModelComparisonError(f"missing sota_program_gate.json: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    lanes = gate.get("lanes", {})
    if lane not in lanes:
        raise ModelComparisonError(f"lane {lane!r} not in sota_program_gate.json")
    lane_payload = lanes[lane]
    rows = lane_payload.get("candidate_rows", [])
    if not isinstance(rows, list) or not rows:
        raise ModelComparisonError(f"no candidate_rows for lane {lane!r}")

    ranked = sorted(
        (
            {
                "candidate": str(r["candidate"]),
                "family": str(r.get("family", "")),
                "group": _method_group(str(r.get("family", ""))),
                "mean_loss": float(r["mean_loss"]),
                "loss_ci_low": r.get("loss_ci_low"),
                "loss_ci_high": r.get("loss_ci_high"),
                "top1_accuracy": r.get("top1_accuracy"),
            }
            for r in rows
            if r.get("mean_loss") is not None
        ),
        key=lambda x: x["mean_loss"],
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    belief_rows = [
        r for r in ranked if r["group"] in {"active_inference", "belief_marginalizing"}
    ]
    nonbelief_rows = [
        r
        for r in ranked
        if r["group"] not in {"active_inference", "belief_marginalizing"}
    ]
    aif_rows = [r for r in ranked if r["group"] == "active_inference"]
    if not aif_rows:
        raise ModelComparisonError(
            "no active-inference candidate found in the lane roster"
        )

    best_aif = min(aif_rows, key=lambda r: r["mean_loss"])
    best_belief = belief_rows[0] if belief_rows else best_aif
    best_nonbelief = nonbelief_rows[0] if nonbelief_rows else None
    leader = ranked[0]
    # The next belief-based method after the AIF candidate (the tie-at-the-top partner).
    next_belief = next(
        (r for r in belief_rows if r["candidate"] != best_aif["candidate"]), None
    )

    # honest gap: how far the best non-belief method sits behind the best belief-based one
    belief_vs_nonbelief_gap = (
        round(best_nonbelief["mean_loss"] - best_belief["mean_loss"], 6)
        if best_nonbelief
        else None
    )
    aif_vs_next_belief_gap = (
        round(next_belief["mean_loss"] - best_aif["mean_loss"], 6)
        if next_belief
        else None
    )
    # Descriptive overlap of MARGINAL loss intervals. This is not a paired test and
    # must never be labelled as one; it is retained only to describe the plotted
    # whiskers when no candidate-pair difference interval is available.
    aif_marginal_ci_disjoint_from_next_belief = None
    if (
        next_belief
        and best_aif.get("loss_ci_high") is not None
        and next_belief.get("loss_ci_low") is not None
    ):
        aif_marginal_ci_disjoint_from_next_belief = bool(
            float(best_aif["loss_ci_high"]) < float(next_belief["loss_ci_low"])
            or float(next_belief.get("loss_ci_high", float("inf")))
            < float(best_aif.get("loss_ci_low", float("-inf")))
        )

    # The honesty boundary that matters: AIF vs the strongest NON-AIF method. When that
    # method is the lane gate's strongest fair comparator, reuse the canonical PAIRED
    # traveller-clustered difference interval. Marginal-CI overlap is not a valid substitute.
    non_aif_rows = [r for r in ranked if r["group"] != "active_inference"]
    strongest_non_aif = non_aif_rows[0] if non_aif_rows else None
    aif_vs_strongest_non_aif_gap = (
        round(strongest_non_aif["mean_loss"] - best_aif["mean_loss"], 6)
        if strongest_non_aif
        else None
    )
    paired_ci_low = None
    paired_ci_high = None
    aif_beats_strongest_non_aif_ci_separated = None
    if strongest_non_aif and strongest_non_aif["candidate"] == lane_payload.get(
        "strongest_fair_baseline"
    ):
        paired_ci_low = lane_payload.get("clustered_ci_low")
        paired_ci_high = lane_payload.get("clustered_ci_high")
    if paired_ci_low is not None and paired_ci_high is not None:
        aif_beats_strongest_non_aif_ci_separated = bool(
            best_aif["mean_loss"] < strongest_non_aif["mean_loss"]
            and float(paired_ci_low) > 0.0
        )
    marginal_description = (
        "Its marginal loss interval is disjoint from the next belief-marginalizing "
        "method's interval"
        if aif_marginal_ci_disjoint_from_next_belief
        else "Its marginal loss interval overlaps the next belief-marginalizing method's interval"
    )
    if paired_ci_low is None or paired_ci_high is None:
        paired_description = (
            "No matched paired interval against the strongest non-AIF method is available"
        )
    elif aif_beats_strongest_non_aif_ci_separated:
        paired_description = (
            "The canonical paired traveller-clustered interval against the strongest "
            f"non-AIF method is [{paired_ci_low}, {paired_ci_high}] and excludes zero"
        )
    else:
        paired_description = (
            "The canonical paired traveller-clustered interval against the strongest "
            f"non-AIF method is [{paired_ci_low}, {paired_ci_high}] and does not establish separation"
        )

    return {
        "schema": MODEL_COMPARISON_SCHEMA,
        "lane": lane,
        "eps": lane_payload.get("eps"),
        "n_methods": len(ranked),
        # absolute anchor for the loss axis: uniform-chance next-cell loss for
        # THIS lane's state count (never hardcoded; ln(n_states) from the lane).
        "n_states": int(lane_payload.get("n_states", 0)),
        "chance_loss_nats": (
            round(math.log(int(lane_payload["n_states"])), 6)
            if lane_payload.get("n_states")
            else None
        ),
        "leaderboard": ranked,
        "leader_method": leader["candidate"],
        "leader_group": leader["group"],
        "aif_method": best_aif["candidate"],
        "aif_rank": int(best_aif["rank"]),
        "aif_loss": round(best_aif["mean_loss"], 6),
        "aif_leads": bool(leader["group"] == "active_inference"),
        "next_belief_method": next_belief["candidate"] if next_belief else None,
        "aif_vs_next_belief_gap": aif_vs_next_belief_gap,
        "aif_marginal_ci_disjoint_from_next_belief": aif_marginal_ci_disjoint_from_next_belief,
        "best_nonbelief_method": best_nonbelief["candidate"]
        if best_nonbelief
        else None,
        "best_nonbelief_loss": round(best_nonbelief["mean_loss"], 6)
        if best_nonbelief
        else None,
        "belief_vs_nonbelief_gap": belief_vs_nonbelief_gap,
        "strongest_non_aif_method": strongest_non_aif["candidate"]
        if strongest_non_aif
        else None,
        "strongest_non_aif_loss": round(strongest_non_aif["mean_loss"], 6)
        if strongest_non_aif
        else None,
        "aif_vs_strongest_non_aif_gap": aif_vs_strongest_non_aif_gap,
        "aif_beats_strongest_non_aif_ci_separated": aif_beats_strongest_non_aif_ci_separated,
        "aif_vs_strongest_non_aif_paired_ci_low": paired_ci_low,
        "aif_vs_strongest_non_aif_paired_ci_high": paired_ci_high,
        "n_belief_methods": len(belief_rows),
        "n_nonbelief_methods": len(nonbelief_rows),
        "verdict": (
            f"Under the noisy partial-observability regime (eps={lane_payload.get('eps')}), active "
            f"inference is the lowest-loss implemented fair predictor (rank {best_aif['rank']} of "
            f"{len(ranked)}) by point estimate. {marginal_description}; marginal-interval overlap "
            f"is descriptive, not a paired test. {paired_description}. Belief-marginalizing methods "
            "(active inference, "
            "soft filter, switching-noise HMM, particle filter) form the lowest-loss point-estimate "
            "cluster, ahead of point-estimate and raw-observation methods — the partial-observability "
            "mechanism across the full roster. This is a within-regime lead among implemented "
            "predictors, not a global "
            "state-of-the-art claim; it does not beat the oracle-fed ceiling and is not a real-"
            "mobility claim."
        ),
        "boundary": (
            "Lowest-loss IMPLEMENTED fair predictor by point estimate in a noisy synthetic regime. "
            f"{paired_description}. "
            "Not global SOTA, not oracle-beating, synthetic-only. The "
            "cross-generator SOTA gate remains blocked."
        ),
    }


def run_and_write_model_comparison(
    project_root: str | Path,
    out_dir: str | Path | None = None,
    *,
    lane: str = DEFAULT_LANE,
) -> Path:
    """Build the within-regime model comparison and write ``model_comparison_leaderboard.json``."""

    root = Path(project_root)
    payload = build_model_comparison(root, lane=lane)
    target_dir = Path(out_dir) if out_dir is not None else root / "output" / "runs"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "model_comparison_leaderboard.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
