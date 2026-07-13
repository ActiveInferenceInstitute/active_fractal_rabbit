from __future__ import annotations

from math import isfinite


SOTA_PROGRAM_FIGURE_SCHEMA = "active-fractal-rabbit-sota-program-figure-adapter-v1"


def _maybe_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _csv_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def sota_program_family_bucket(family: str) -> str:
    if family in {"active_inference_candidate", "active_inference_reference"}:
        return "active_inference"
    if family in {"classical_sequence", "classical_state_space"}:
        return "classical_state_space"
    if family in {"neural_sequence", "dependency_light_sequence_attention"}:
        return "neural_or_attention"
    if family in {"transparent_filter_baseline", "transparent_markov_baseline"}:
        return "transparent_baseline"
    return family


def sota_program_display_name(candidate: str) -> str:
    labels = {
        "precision_return_aif": "precision return AIF",
        "soft_filter": "soft filter (AIF ref)",
        "switching_noise_hmm": "switching noise HMM",
        "particle_filter": "particle filter",
        "mlp_next_cell": "MLP next cell",
        "hard_filter": "hard filter",
        "multi_horizon_recency_attention": "multi-horizon recency attention",
        "recency_attention": "recency attention",
        "first_order_markov": "first-order Markov",
        "return_boosted_markov": "return-boosted Markov",
        "observed_markov": "observed Markov",
        "online_base_rate": "online base rate",
    }
    return labels.get(candidate, candidate.replace("_", " "))


def sota_program_leaderboard_summary(
    rows: list[dict[str, str]], gate: dict[str, object]
) -> dict[str, object]:
    converted: list[dict[str, object]] = []
    strongest_baseline_loss: float | None = None
    for row in rows:
        if row.get("lane") != "fractalrabbit":
            continue
        if not _csv_bool(row.get("locked_split_scored")):
            continue
        loss = _maybe_float(row.get("mean_loss"))
        if loss is None:
            continue
        if row.get("role") == "strongest_fair_baseline":
            strongest_baseline_loss = loss
        candidate = str(row.get("candidate", "unknown"))
        converted.append(
            {
                "model": sota_program_display_name(candidate),
                "candidate": candidate,
                "family": sota_program_family_bucket(str(row.get("family", ""))),
                "raw_family": row.get("family", ""),
                "current_role": row.get("role", ""),
                "lane": row.get("lane", "fractalrabbit"),
                "locked_split_scored": True,
                "leakage": _csv_bool(row.get("leakage")),
                "mean_predictive_loss": loss,
                "loss_ci_low": _maybe_float(row.get("loss_ci_low")),
                "loss_ci_high": _maybe_float(row.get("loss_ci_high")),
                "clustered_ci_low": _maybe_float(row.get("clustered_ci_low")),
                "clustered_ci_high": _maybe_float(row.get("clustered_ci_high")),
                "brier_score": _maybe_float(row.get("brier_score")),
                "ece": _maybe_float(row.get("ece")),
                "top1_accuracy": _maybe_float(row.get("top1_accuracy")),
                "runtime_seconds": _maybe_float(row.get("runtime_seconds")),
            }
        )
    if not converted:
        return {}
    if strongest_baseline_loss is None:
        fair_rows = [r for r in converted if r.get("family") != "active_inference"]
        strongest_baseline_loss = min(
            (float(r["mean_predictive_loss"]) for r in fair_rows), default=None
        )
    lanes = gate.get("lanes", {}) if isinstance(gate.get("lanes"), dict) else {}
    fractalrabbit_gate = (
        lanes.get("fractalrabbit", {}) if isinstance(lanes, dict) else {}
    )
    if not isinstance(fractalrabbit_gate, dict):
        fractalrabbit_gate = {}
    return {
        "schema": SOTA_PROGRAM_FIGURE_SCHEMA,
        "status": str(gate.get("status", "sota_program_not_cleared")),
        "strongest_baseline_loss": strongest_baseline_loss,
        "fractalrabbit_gate": {
            "n_holdout_travellers": fractalrabbit_gate.get("n_holdout_travellers"),
            "strongest_fair_baseline": fractalrabbit_gate.get(
                "strongest_fair_baseline"
            ),
            "mean_difference_nats_fair_minus_aif": _maybe_float(
                fractalrabbit_gate.get("mean_difference_nats_fair_minus_aif")
            ),
            "clustered_ci_low": _maybe_float(
                fractalrabbit_gate.get("clustered_ci_low")
            ),
            "clustered_ci_high": _maybe_float(
                fractalrabbit_gate.get("clustered_ci_high")
            ),
            "one_sided_p": _maybe_float(fractalrabbit_gate.get("one_sided_p")),
            "aif_beats_strongest_fair": bool(
                fractalrabbit_gate.get("aif_beats_strongest_fair", False)
            ),
        },
        "leaderboard": converted,
        "headline": (
            "Direct locked-split comparison: active inference is scored; "
            "SOTA program gate stays closed"
        ),
    }
