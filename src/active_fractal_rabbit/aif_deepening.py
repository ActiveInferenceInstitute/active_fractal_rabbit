"""Audit artifact for modular active-inference deepening candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

AIF_DEEPENING_SCHEMA = "active-fractal-rabbit-aif-deepening-candidates-v1"


def aif_deepening_candidates(project_root: str | Path) -> dict[str, Any]:
    """Summarize AIF deepening surfaces without promoting audit-only rows."""

    root = Path(project_root).resolve()
    runs = root / "output" / "runs"
    factorial = _read_json(runs / "factorial_pymdp_analysis_fixture.json")
    regime_gate = _read_json(runs / "aif_regime_sota_gate.json")
    sota_gate = _read_json(runs / "sota_program_gate.json")
    robustness = _read_json(runs / "sota_program_robustness_grid.json")
    adaptive = _read_json(runs / "adaptive_aif_ablation.json")

    rows = [
        _factorial_row(factorial),
        _multiscale_row(regime_gate, sota_gate),
        _online_adaptation_row(robustness, adaptive),
    ]
    scored = sum(row["protocol_status"] == "scored_predictive_candidate" for row in rows)
    return {
        "schema": AIF_DEEPENING_SCHEMA,
        "status": (
            "scored_candidate_no_predictive_promotion"
            if scored
            else "audit_only_no_predictive_promotion"
        ),
        "predictive_gate_clears": False,
        "scored_protocol_candidate_count": scored,
        "audit_surface_count": len(rows),
        "rows": rows,
        "locked_protocol": "Candidate.predict(PredictContext) -> np.ndarray",
        "boundary": (
            "Factorial tensors, EFE/VFE diagnostics, multiscale beliefs, and adaptation surfaces "
            "are audit evidence until they enter the locked predictive Candidate protocol and "
            "improve holdout loss. They do not substitute for predictive improvement or unlock "
            "global, external, empirical, or operational SOTA wording."
        ),
    }


def run_and_write_aif_deepening_candidates(
    project_root: str | Path,
    out_path: str | Path | None = None,
) -> Path:
    """Write ``aif_deepening_candidates.json``."""

    root = Path(project_root).resolve()
    out = (
        Path(out_path)
        if out_path
        else root / "output" / "runs" / "aif_deepening_candidates.json"
    )
    payload = aif_deepening_candidates(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def _factorial_row(factorial: dict[str, Any]) -> dict[str, Any]:
    tensor_summary = factorial.get("tensor_summary", {})
    metrics = factorial.get("metrics", {})
    return {
        "candidate_id": "factorial_matched_prior_tensor_provenance",
        "surface": "factorial_pymdp",
        "status": str(factorial.get("status", "unavailable")),
        "protocol_status": "audit_only_tensor_surface",
        "matched_prior_validation": bool(metrics.get("matched_prior_validation", False))
        or "A_shapes" in tensor_summary,
        "tensor_provenance_present": bool(tensor_summary),
        "predictive_gate_clears": False,
        "evidence_artifact": "output/runs/factorial_pymdp_analysis_fixture.json",
        "next_required_step": "wrap tensor factorization as a locked-split predictive Candidate before scoring",
    }


def _multiscale_row(
    regime_gate: dict[str, Any], sota_gate: dict[str, Any]
) -> dict[str, Any]:
    return {
        "candidate_id": "multiscale_soft_belief_backoff",
        "surface": "aif_regime_search",
        "status": str(regime_gate.get("status", "unavailable")),
        "protocol_status": "audit_only_until_candidate_predicts",
        "local_regime_clears": bool(regime_gate.get("gate_clears")),
        "global_sota_gate_clears": bool(sota_gate.get("gate_clears")),
        "predictive_gate_clears": False,
        "evidence_artifact": "output/runs/aif_regime_sota_gate.json",
        "next_required_step": "enter the SOTA roster as a Candidate and beat the strongest fair baseline on locked splits",
    }


def _online_adaptation_row(
    robustness: dict[str, Any], adaptive: dict[str, Any]
) -> dict[str, Any]:
    lane_statuses = robustness.get("lane_statuses", [])
    all_diagnostics = bool(robustness.get("all_diagnostic_lanes_scored"))
    return {
        "candidate_id": "online_adaptation",
        "surface": "diagnostic_stressor_rows",
        "status": str(robustness.get("status", "unavailable")),
        "protocol_status": (
            "scored_predictive_candidate"
            if adaptive.get("status") == "adaptive_repeated_ablation_scored"
            else "audit_only_until_online_candidate_scores"
        ),
        "diagnostic_lanes_scored": all_diagnostics,
        "diagnostic_lane_count": len(lane_statuses)
        if isinstance(lane_statuses, list)
        else 0,
        "predictive_gate_clears": False,
        "improves_fixed_aif": bool(adaptive.get("adaptive_improves_fixed")),
        "adaptive_improvement_nats": adaptive.get("adaptive_improvement_nats"),
        "realization_count": adaptive.get("realization_count", 0),
        "beats_fixed_ci_count": adaptive.get("adaptive_beats_fixed_ci_count", 0),
        "beats_fair_ci_count": adaptive.get(
            "adaptive_beats_online_base_rate_ci_count", 0
        ),
        "evidence_artifact": "output/runs/adaptive_aif_ablation.json",
        "next_required_step": "broaden switching regimes and close the remaining adaptive fair-baseline gap",
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}
