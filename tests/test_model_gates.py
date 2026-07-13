from __future__ import annotations

import json
from pathlib import Path

from active_fractal_rabbit.model_gates import (
    build_model_gate_matrix,
    run_and_write_model_gates,
)

def test_model_gate_matrix_reports_lane_rows_and_unavailable_without_zeroes() -> None:
    payload = build_model_gate_matrix(
        {
            "fixture": {
                "sequence": {
                    "status": "completed",
                    "best_model": "persistence",
                    "best_mean_predictive_loss": 0.75,
                    "pymdp_performance_gate": "stronger_sequence_baseline_beats_pymdp_on_this_run",
                },
                "missingness": {
                    "status": "completed",
                    "verdict": "honest_null",
                    "phase_traveller_gain_vs_pooled": -0.1,
                },
                "return_phase": {
                    "status": "completed",
                    "verdict": "honest_null",
                    "best_model": "phase_prior",
                    "return_plus_phase_gain_vs_persistence": -0.2,
                },
                "factorial": {
                    "status": "completed",
                    "performance_gate": "factorial_pymdp_not_performance_leading",
                    "best_model": "switching_hmm",
                    "factorial_pymdp_mean_predictive_loss": 1.1,
                    "factorial_pymdp_gain_vs_single_pymdp": -0.3,
                },
            },
            "external": {
                "sequence": {
                    "status": "unavailable",
                    "pymdp_performance_gate": "unavailable",
                },
                "missingness": {"status": "unavailable", "verdict": "unavailable"},
                "return_phase": {"status": "unavailable", "verdict": "unavailable"},
                "factorial": {
                    "status": "unavailable",
                    "performance_gate": "unavailable",
                    "factorial_pymdp_mean_predictive_loss": "unavailable",
                },
            },
        }
    )

    assert payload["schema"] == "active-fractal-rabbit-model-gate-matrix-v1"
    assert payload["status"] == "partial"
    assert payload["external_sequence_gate"] == "unavailable"
    assert payload["external_factorial_gate"] == "unavailable"
    assert any(row["gate"] == "factorial_pymdp_latent_modes" for row in payload["rows"])
    external_rows = [row for row in payload["rows"] if row["lane"] == "external"]
    assert external_rows
    assert all(row["primary_metric"] is None for row in external_rows)


def test_run_model_gates_writes_lane_qualified_and_canonical_artifacts(
    repo_sandbox: Path,
) -> None:
    # This end-to-end writer regenerates canonical-named artifacts. Keep it in
    # the repository sandbox so a test run cannot invalidate the production
    # figure registry's source bindings.
    artifacts = run_and_write_model_gates(
        project_root=repo_sandbox,
        fixture_config="configs/default.yaml",
        external_config="configs/fractalrabbit_external.yaml",
        external_ready=False,
    )

    for key in [
        "sequence_fixture",
        "sequence_external",
        "missingness_fixture",
        "missingness_external",
        "return_phase_fixture",
        "return_phase_external",
        "factorial_fixture",
        "factorial_external",
        "return_phase",
        "matrix",
        "sequence_canonical",
        "missingness_canonical",
    ]:
        assert artifacts[key].exists()

    matrix = json.loads(artifacts["matrix"].read_text(encoding="utf-8"))
    sequence_external = json.loads(
        artifacts["sequence_external"].read_text(encoding="utf-8")
    )
    missing_external = json.loads(
        artifacts["missingness_external"].read_text(encoding="utf-8")
    )
    factorial_external = json.loads(
        artifacts["factorial_external"].read_text(encoding="utf-8")
    )

    assert matrix["status"] == "partial"
    assert sequence_external["status"] == "unavailable"
    assert missing_external["verdict"] == "unavailable"
    assert factorial_external["status"] == "unavailable"
    assert factorial_external["factorial_pymdp_mean_predictive_loss"] == "unavailable"
    assert "current full-study invocation" in sequence_external["reason"]


def test_empty_or_malformed_factorial_artifact_does_not_greenwash_lane() -> None:
    """Fail-closed regression: a missing/empty factorial payload must mark the
    lane unavailable, never silently complete it (the _completed default-True
    green-wash caught by cross-vendor audit)."""

    from active_fractal_rabbit.model_gates import _completed

    assert _completed({}) is False
    assert _completed({"status": "completed"}) is True
    assert _completed({"verdict": "honest_null"}) is True
    assert _completed({"verdict": "unavailable"}) is False

    payload = build_model_gate_matrix(
        {
            "fixture": {
                "sequence": {
                    "status": "completed",
                    "best_model": "persistence",
                    "best_mean_predictive_loss": 0.7,
                    "pymdp_performance_gate": "pymdp_beats_sequence_baselines_on_this_run",
                },
                "missingness": {
                    "status": "completed",
                    "verdict": "honest_null",
                    "phase_traveller_gain_vs_pooled": -0.1,
                },
                "return_phase": {
                    "status": "completed",
                    "verdict": "honest_null",
                    "best_model": "phase_prior",
                    "return_plus_phase_gain_vs_persistence": -0.2,
                },
                "factorial": {},  # malformed/empty must not green-wash
            }
        }
    )
    fixture_lane = next(lane for lane in payload["lanes"] if lane["lane"] == "fixture")
    assert fixture_lane["status"] == "unavailable"
    assert payload["status"] == "partial"
