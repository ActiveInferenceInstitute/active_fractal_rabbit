"""Canonical forward-gate artifacts for the M10 and safety-roadmap analyses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cross_generator import run_cross_generator_transfer
from .efe_location_frontier import run_and_write_efe_location_frontier
from .heldout_reporting_process import run_and_write_heldout_reporting_process
from .latent_regime_hmm import run_and_write_latent_regime_hmm
from .learned_stay_rate import run_and_write_learned_stay_rate
from .linkage_risk_frontier import run_and_write_linkage_risk_frontier
from .neural_baseline import run_and_write_neural_baseline
from .poisoning_robustness import run_and_write_poisoning_robustness

FORWARD_GATES_SCHEMA = "active-fractal-rabbit-forward-gates-v1"


def run_and_write_forward_gates(
    project_root: str | Path,
    *,
    fixture_config: str | Path = "configs/default.yaml",
    external_config: str | Path = "configs/fractalrabbit_external.yaml",
    external_ready: bool = False,
) -> Path:
    """Run and summarize the promoted forward-gate artifact family."""

    root = Path(project_root).resolve()
    runs = root / "output" / "runs"
    fixture_cfg = root / fixture_config
    external_cfg = root / external_config
    external_observations = (
        root / "output" / "runs" / "fractalrabbit_external" / "observations.csv"
    )

    fixture_specs: tuple[tuple[str, str, Any], ...] = (
        (
            "latent_regime_hmm",
            "latent_regime_hmm.json",
            run_and_write_latent_regime_hmm,
        ),
        ("neural_baseline", "neural_baseline.json", run_and_write_neural_baseline),
        (
            "learned_stay_rate",
            "learned_stay_rate.json",
            run_and_write_learned_stay_rate,
        ),
        (
            "heldout_reporting_process",
            "heldout_reporting_process.json",
            run_and_write_heldout_reporting_process,
        ),
        (
            "cross_generator_transfer",
            "cross_generator_transfer.json",
            run_cross_generator_transfer,
        ),
        (
            "efe_location_frontier",
            "efe_location_frontier.json",
            run_and_write_efe_location_frontier,
        ),
        (
            "linkage_risk_frontier",
            "linkage_risk_frontier.json",
            run_and_write_linkage_risk_frontier,
        ),
    )
    rows: list[dict[str, object]] = []
    canonical_paths: dict[str, str] = {}
    for gate, filename, writer in fixture_specs:
        output = writer(fixture_cfg, runs / filename)
        canonical_paths[gate] = _relative(root, output)
        rows.append(_row_from_payload(root, gate, "fixture", output))

    poison_output = run_and_write_poisoning_robustness(
        fixture_cfg, None, runs / "poisoning_robustness.json"
    )
    canonical_paths["poisoning_robustness"] = _relative(root, poison_output)
    rows.append(
        _row_from_payload(root, "poisoning_robustness", "fixture", poison_output)
    )

    external_specs: tuple[tuple[str, str, Any], ...] = (
        (
            "latent_regime_hmm",
            "latent_regime_hmm_external.json",
            run_and_write_latent_regime_hmm,
        ),
        (
            "neural_baseline",
            "neural_baseline_external.json",
            run_and_write_neural_baseline,
        ),
        (
            "learned_stay_rate",
            "learned_stay_rate_external.json",
            run_and_write_learned_stay_rate,
        ),
        (
            "heldout_reporting_process",
            "heldout_reporting_process_external.json",
            run_and_write_heldout_reporting_process,
        ),
        (
            "cross_generator_transfer",
            "cross_generator_transfer_external.json",
            run_cross_generator_transfer,
        ),
        (
            "efe_location_frontier",
            "efe_location_frontier_external.json",
            run_and_write_efe_location_frontier,
        ),
        (
            "linkage_risk_frontier",
            "linkage_risk_frontier_external.json",
            run_and_write_linkage_risk_frontier,
        ),
    )
    if external_ready:
        for gate, filename, writer in external_specs:
            if gate == "cross_generator_transfer":
                output = writer(
                    external_cfg,
                    runs / filename,
                    reference_observations_path=external_observations,
                )
            else:
                output = writer(
                    external_cfg,
                    runs / filename,
                    observations_path=external_observations,
                )
            rows.append(_row_from_payload(root, gate, "external", output))
        output = run_and_write_poisoning_robustness(
            external_cfg,
            external_observations,
            runs / "poisoning_robustness_external.json",
        )
        rows.append(_row_from_payload(root, "poisoning_robustness", "external", output))
    else:
        for gate, filename, _writer in external_specs:
            output = _write_unavailable(
                runs / filename,
                gate=gate,
                lane="external",
                reason="external lane was not completed in this run",
            )
            rows.append(_row_from_payload(root, gate, "external", output))
        output = _write_unavailable(
            runs / "poisoning_robustness_external.json",
            gate="poisoning_robustness",
            lane="external",
            reason="external lane was not completed in this run",
        )
        rows.append(_row_from_payload(root, "poisoning_robustness", "external", output))

    payload = {
        "schema": FORWARD_GATES_SCHEMA,
        "status": "completed"
        if all(row["status"] == "completed" for row in rows if row["lane"] == "fixture")
        else "partial",
        "canonical_fixture_artifacts": canonical_paths,
        "rows": rows,
        "fixture_completed_count": sum(
            1
            for row in rows
            if row["lane"] == "fixture" and row["status"] == "completed"
        ),
        "external_completed_count": sum(
            1
            for row in rows
            if row["lane"] == "external" and row["status"] == "completed"
        ),
        "caveat": (
            "Fixture rows are canonical manuscript evidence. External rows are current-lane evidence only when "
            "the external simulator completed in this run; otherwise they are explicit unavailable records."
        ),
    }
    output = runs / "forward_gates_summary.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def _write_unavailable(path: Path, *, gate: str, lane: str, reason: str) -> Path:
    payload = {
        "schema": f"active-fractal-rabbit-{gate.replace('_', '-')}-unavailable-v1",
        "status": "unavailable",
        "gate": gate,
        "lane": lane,
        "reason": reason,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _row_from_payload(
    root: Path, gate: str, lane: str, path: Path
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    primary_metric = _primary_metric(gate, payload)
    return {
        "lane": lane,
        "gate": gate,
        "status": payload.get("status", "unknown"),
        "artifact": _relative(root, path),
        "verdict": _verdict(gate, payload),
        "primary_metric": primary_metric,
        "primary_metric_label": _metric_label(gate),
    }


def _verdict(gate: str, payload: dict[str, object]) -> str:
    if payload.get("status") == "unavailable":
        return str(payload.get("reason", "unavailable"))
    if "verdict" in payload:
        return str(payload["verdict"])
    if "gate" in payload:
        return str(payload["gate"])
    if gate == "cross_generator_transfer":
        verdict = payload.get("transfer_verdict", {})
        if isinstance(verdict, dict):
            persistence = verdict.get("persistence_strength", "unknown")
            particle = verdict.get("particle_beats_hard_persistence", "unknown")
            returns = verdict.get("return_structure_helps", "unknown")
            return f"persistence {persistence}; particle {particle}; return {returns}"
    return "completed"


def _primary_metric(gate: str, payload: dict[str, object]) -> float | None:
    if payload.get("status") == "unavailable":
        return None
    mapping = {
        "latent_regime_hmm": "best_honest_heldout_loss",
        "neural_baseline": "neural_logistic_mean_predictive_loss",
        "learned_stay_rate": "best_variant_minus_particle_mean",
        "heldout_reporting_process": "runlen_loto_gain_vs_pooled",
        "efe_location_frontier": "location_joint_share",
        "linkage_risk_frontier": "location_real_linkage",
    }
    key = mapping.get(gate)
    if key and isinstance(payload.get(key), int | float):
        return float(payload[key])
    if gate == "cross_generator_transfer":
        regimes = payload.get("regimes", {})
        if isinstance(regimes, dict) and isinstance(regimes.get("shortrun"), dict):
            return float(
                regimes["shortrun"].get("particle_beats_hard_persistence_count", 0.0)
            )
    if gate == "poisoning_robustness":
        per_mode = payload.get("per_mode", {})
        if isinstance(per_mode, dict) and isinstance(per_mode.get("teleport"), dict):
            row = per_mode["teleport"].get(f"{1 / 6:.4f}", {})
            if isinstance(row, dict) and isinstance(
                row.get("vfe_auc_mean"), int | float
            ):
                return float(row["vfe_auc_mean"])
    return None


def _metric_label(gate: str) -> str:
    return {
        "latent_regime_hmm": "best honest held-out loss",
        "neural_baseline": "logistic mean predictive loss",
        "learned_stay_rate": "best variant minus particle",
        "heldout_reporting_process": "runlen LOTO gain vs pooled",
        "cross_generator_transfer": "short-run particle-beats-hard count",
        "efe_location_frontier": "location joint info share",
        "linkage_risk_frontier": "full-location linkage risk",
        "poisoning_robustness": "teleport VFE AUC at mid-rate",
    }.get(gate, "metric")


def _relative(root: Path, path: Path) -> str:
    return str(Path(path).resolve().relative_to(root))
