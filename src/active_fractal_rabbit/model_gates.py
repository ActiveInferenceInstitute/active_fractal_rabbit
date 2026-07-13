"""Lane-resolved model-gate matrix for stronger benchmark claims."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .factorial_pymdp import (
    run_and_write_factorial_analysis,
    unavailable_factorial_analysis,
)
from .missingness_model import MISSINGNESS_MODEL_SCHEMA, run_and_write_missingness_model
from .return_phase_prior import (
    run_and_write_return_phase_prior,
    unavailable_return_phase_prior,
)
from .sequence_baselines import (
    SEQUENCE_BASELINE_SCHEMA,
    run_and_write_sequence_baselines,
)

MODEL_GATE_MATRIX_SCHEMA = "active-fractal-rabbit-model-gate-matrix-v1"


def run_and_write_model_gates(
    *,
    project_root: str | Path,
    fixture_config: str | Path,
    external_config: str | Path,
    external_ready: bool,
) -> dict[str, Path]:
    """Run fixture/external model gates and write the matrix plus lane artifacts."""

    root = Path(project_root).resolve()
    runs = root / "output" / "runs"
    artifacts = {
        "sequence_fixture": runs / "sequence_baseline_analysis_fixture.json",
        "sequence_external": runs / "sequence_baseline_analysis_external.json",
        "missingness_fixture": runs / "missingness_model_analysis_fixture.json",
        "missingness_external": runs / "missingness_model_analysis_external.json",
        "return_phase_fixture": runs / "return_phase_prior_analysis_fixture.json",
        "return_phase_external": runs / "return_phase_prior_analysis_external.json",
        "return_phase": runs / "return_phase_prior_analysis.json",
        "factorial_fixture": runs / "factorial_pymdp_analysis_fixture.json",
        "factorial_external": runs / "factorial_pymdp_analysis_external.json",
        "matrix": runs / "model_gate_matrix.json",
        "sequence_canonical": runs / "sequence_baseline_analysis.json",
        "missingness_canonical": runs / "missingness_model_analysis.json",
    }

    fixture_cfg = root / fixture_config
    external_cfg = root / external_config
    run_and_write_sequence_baselines(fixture_cfg, artifacts["sequence_fixture"])
    run_and_write_missingness_model(fixture_cfg, artifacts["missingness_fixture"])
    run_and_write_return_phase_prior(fixture_cfg, artifacts["return_phase_fixture"])
    run_and_write_factorial_analysis(fixture_cfg, artifacts["factorial_fixture"])
    shutil.copyfile(artifacts["sequence_fixture"], artifacts["sequence_canonical"])
    shutil.copyfile(
        artifacts["missingness_fixture"], artifacts["missingness_canonical"]
    )
    shutil.copyfile(artifacts["return_phase_fixture"], artifacts["return_phase"])

    unavailable_reason = (
        "external lane did not complete in the current full-study invocation"
    )
    if external_ready:
        run_and_write_sequence_baselines(external_cfg, artifacts["sequence_external"])
        run_and_write_missingness_model(external_cfg, artifacts["missingness_external"])
        run_and_write_return_phase_prior(
            external_cfg, artifacts["return_phase_external"]
        )
        run_and_write_factorial_analysis(external_cfg, artifacts["factorial_external"])
    else:
        _write_json(
            artifacts["sequence_external"], _unavailable_sequence(unavailable_reason)
        )
        _write_json(
            artifacts["missingness_external"],
            _unavailable_missingness(unavailable_reason),
        )
        _write_json(
            artifacts["return_phase_external"],
            unavailable_return_phase_prior(unavailable_reason),
        )
        _write_json(
            artifacts["factorial_external"],
            unavailable_factorial_analysis(unavailable_reason),
        )

    payload = build_model_gate_matrix(
        {
            "fixture": {
                "sequence": _read_json(artifacts["sequence_fixture"]),
                "missingness": _read_json(artifacts["missingness_fixture"]),
                "return_phase": _read_json(artifacts["return_phase_fixture"]),
                "factorial": _factorial_summary(
                    _read_json(artifacts["factorial_fixture"])
                ),
            },
            "external": {
                "sequence": _read_json(artifacts["sequence_external"]),
                "missingness": _read_json(artifacts["missingness_external"]),
                "return_phase": _read_json(artifacts["return_phase_external"]),
                "factorial": _factorial_summary(
                    _read_json(artifacts["factorial_external"])
                ),
            },
        }
    )
    _write_json(artifacts["matrix"], payload)
    return artifacts


def build_model_gate_matrix(
    lanes: dict[str, dict[str, dict[str, object]]],
) -> dict[str, object]:
    """Build the manuscript-facing lane/model comparison matrix."""

    rows: list[dict[str, object]] = []
    lane_summaries: list[dict[str, object]] = []
    for lane, artifacts in lanes.items():
        sequence = artifacts["sequence"]
        missingness = artifacts["missingness"]
        return_phase = artifacts["return_phase"]
        factorial = artifacts.get("factorial", {"status": "unavailable"})
        # Check every expected gate explicitly (including factorial). Relying on
        # artifacts.values() would silently exclude a missing factorial key, and
        # an empty/malformed payload must never green-wash the lane.
        lane_status = (
            "completed"
            if all(
                _completed(item)
                for item in (sequence, missingness, return_phase, factorial)
            )
            else "unavailable"
        )
        lane_summaries.append(
            {
                "lane": lane,
                "status": lane_status,
                "sequence_gate": sequence.get("pymdp_performance_gate", "unavailable"),
                "sequence_best_model": sequence.get("best_model", "unavailable"),
                "missingness_verdict": missingness.get("verdict", "unavailable"),
                "return_phase_verdict": return_phase.get("verdict", "unavailable"),
                "factorial_gate": factorial.get("performance_gate", "unavailable"),
                "factorial_best_model": factorial.get("best_model", "unavailable"),
            }
        )
        rows.extend(
            [
                _matrix_row(
                    lane,
                    "sequence_baseline_ladder",
                    sequence.get("status", "unavailable"),
                    sequence.get("best_model", "unavailable"),
                    sequence.get("best_mean_predictive_loss"),
                    sequence.get("pymdp_performance_gate", "unavailable"),
                    "best prequential next-cell null",
                ),
                _matrix_row(
                    lane,
                    "reporting_process_missingness",
                    missingness.get(
                        "status",
                        "completed" if "verdict" in missingness else "unavailable",
                    ),
                    "phase_traveller_reporter",
                    missingness.get("phase_traveller_gain_vs_pooled"),
                    missingness.get("verdict", "unavailable"),
                    "log-loss gain over pooled reporter",
                ),
                _matrix_row(
                    lane,
                    "return_plus_phase_prior",
                    return_phase.get("status", "unavailable"),
                    return_phase.get("best_model", "unavailable"),
                    return_phase.get("return_plus_phase_gain_vs_persistence"),
                    return_phase.get("verdict", "unavailable"),
                    "gain over persistence for combined prior",
                ),
                _matrix_row(
                    lane,
                    "factorial_pymdp_latent_modes",
                    factorial.get("status", "unavailable"),
                    factorial.get("best_model", "unavailable"),
                    factorial.get("factorial_pymdp_mean_predictive_loss"),
                    factorial.get("performance_gate", "unavailable"),
                    "factorial mean next-observation loss vs single-factor pymdp",
                ),
            ]
        )

    status = (
        "completed"
        if all(item["status"] == "completed" for item in lane_summaries)
        else "partial"
    )
    return {
        "schema": MODEL_GATE_MATRIX_SCHEMA,
        "status": status,
        "lanes": lane_summaries,
        "rows": rows,
        "external_sequence_gate": _lane_value(
            lane_summaries, "external", "sequence_gate"
        ),
        "external_missingness_verdict": _lane_value(
            lane_summaries, "external", "missingness_verdict"
        ),
        "external_return_phase_verdict": _lane_value(
            lane_summaries, "external", "return_phase_verdict"
        ),
        "external_factorial_gate": _lane_value(
            lane_summaries, "external", "factorial_gate"
        ),
        "future_candidate_families": [
            "hidden_markov_models",
            "state_space_particle_filters",
            "neural_sequence_models",
            "learned_hierarchical_active_inference",
        ],
        "caveat": (
            "This matrix is a lane-resolved evidence gate for model claims. Unavailable lanes render explicit "
            "unavailable statuses and never contribute numeric zeroes."
        ),
    }


def _matrix_row(
    lane: str,
    gate: str,
    status: object,
    model: object,
    primary_metric: object,
    verdict: object,
    metric_role: str,
) -> dict[str, object]:
    return {
        "lane": lane,
        "gate": gate,
        "status": str(status),
        "model": str(model),
        "primary_metric": primary_metric
        if isinstance(primary_metric, int | float)
        else None,
        "metric_role": metric_role,
        "verdict": str(verdict),
    }


def _factorial_summary(payload: dict[str, object]) -> dict[str, object]:
    """Flatten a factorial-pymdp artifact into the lane comparison shape.

    Completed artifacts nest the predictive loss under ``metrics``; the
    no-number unavailable payload carries a flat ``"unavailable"`` sentinel.
    Both map to the same flat keys the matrix consumes.
    """

    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "status": payload.get("status", "unavailable"),
        "performance_gate": payload.get("performance_gate", "unavailable"),
        "best_model": payload.get("best_model", "unavailable"),
        "factorial_pymdp_mean_predictive_loss": metrics.get(
            "factorial_pymdp_mean_predictive_loss",
            payload.get("factorial_pymdp_mean_predictive_loss", "unavailable"),
        ),
        "factorial_pymdp_gain_vs_single_pymdp": metrics.get(
            "factorial_pymdp_gain_vs_single_pymdp"
        ),
    }


def _completed(payload: dict[str, object]) -> bool:
    """Fail-closed lane-completion test.

    An empty or non-dict payload is NOT completed (a missing/malformed artifact
    must never green-wash a lane). When a ``status`` field is present it is
    authoritative. The missingness artifact signals completion through a
    ``verdict`` rather than a ``status``; for it, any non-``unavailable`` verdict
    counts as completed. Anything else is treated as unavailable.
    """

    if not isinstance(payload, dict) or not payload:
        return False
    status = payload.get("status")
    if status is not None:
        return status == "completed"
    verdict = payload.get("verdict")
    if verdict is not None:
        return verdict != "unavailable"
    return False


def _lane_value(rows: list[dict[str, object]], lane: str, key: str) -> str:
    for row in rows:
        if row["lane"] == lane:
            return str(row.get(key, "unavailable"))
    return "unavailable"


def _unavailable_sequence(reason: str) -> dict[str, object]:
    return {
        "schema": SEQUENCE_BASELINE_SCHEMA,
        "status": "unavailable",
        "reason": reason,
        "rows": [],
        "pymdp_performance_gate": "unavailable",
        "best_model": "unavailable",
        "caveat": "No numeric sequence-baseline result is emitted for an unavailable lane.",
    }


def _unavailable_missingness(reason: str) -> dict[str, object]:
    return {
        "schema": MISSINGNESS_MODEL_SCHEMA,
        "status": "unavailable",
        "reason": reason,
        "verdict": "unavailable",
        "phase_short_gap_probability": [],
        "caveat": "No numeric missingness result is emitted for an unavailable lane.",
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
