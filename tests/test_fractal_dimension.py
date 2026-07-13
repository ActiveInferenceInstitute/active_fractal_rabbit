from __future__ import annotations

import json
from pathlib import Path

from active_fractal_rabbit.fractal_dimension import (
    run_and_write_fractal_dimension_sweep,
    summarize_fractal_dimension_sweep,
)


def test_fractal_dimension_summary_selects_best_loss_and_revisit_dimension() -> None:
    payload = summarize_fractal_dimension_sweep(
        [
            {
                "fractal_dimension": 1.1,
                "mean_predictive_loss": 2.0,
                "place_revisit_rate": 0.3,
            },
            {
                "fractal_dimension": 1.7,
                "mean_predictive_loss": 1.5,
                "place_revisit_rate": 0.8,
            },
        ],
        attempted_dimensions=[1.1, 1.7],
    )

    assert payload["schema"] == "active-fractal-rabbit-fractal-dimension-sweep-v1"
    assert payload["status"] == "completed"
    assert payload["best_predictive_loss_dimension"] == 1.7
    assert payload["highest_revisit_dimension"] == 1.7
    assert payload["metrics"]["mean_predictive_loss"]["low"] == 1.5
    assert payload["metrics"]["place_revisit_rate"]["high"] == 0.8


def test_fractal_dimension_inactive_control_reports_fixture_replay_invariance() -> None:
    """When the dimension control is inactive the caveat must not claim sensitivity."""

    flat_rows = [
        {
            "fractal_dimension": dim,
            "mean_predictive_loss": 5.72,
            "mean_posterior_entropy": 0.046,
        }
        for dim in (1.1, 1.33, 1.67, 1.9)
    ]
    payload = summarize_fractal_dimension_sweep(
        flat_rows,
        attempted_dimensions=[1.1, 1.33, 1.67, 1.9],
        dimension_control_active=False,
    )

    assert payload["status"] == "fixture_control_unvaried"
    assert payload["dimension_control_active"] is False
    assert payload["verdict"] == "fixture_only_dimension_not_resimulated"
    # Zero spread by construction — an invariance check, not sensitivity evidence.
    assert payload["metrics"]["mean_predictive_loss"]["sd"] == 0.0
    caveat = payload["caveat"].lower()
    assert "invariance check" in caveat
    assert "no generator-knob sensitivity" in caveat
    assert "replayed" in caveat
    # The active-control caveat's sensitivity language must be absent here.
    assert "varied as a synthetic-generator control" not in caveat


def test_fractal_dimension_active_control_keeps_sensitivity_caveat() -> None:
    payload = summarize_fractal_dimension_sweep(
        [
            {"fractal_dimension": 1.1, "mean_predictive_loss": 2.0},
            {"fractal_dimension": 1.7, "mean_predictive_loss": 1.5},
        ],
        attempted_dimensions=[1.1, 1.7],
        dimension_control_active=True,
    )

    assert payload["status"] == "completed"
    assert "varied as a synthetic-generator control" in payload["caveat"]
    assert "invariance check" not in payload["caveat"].lower()


def test_fractal_dimension_sweep_records_runner_failures(tmp_path) -> None:
    def runner(dimension: float):
        if dimension > 1.2:
            return None
        return {
            "n_observations": 10,
            "n_latent_states": 4,
            "mean_predictive_loss": 2.0,
            "mean_posterior_entropy": 0.5,
            "state_recovery_reference_mean_mass_on_true_cell": 0.8,
            "place_revisit_rate": 0.25,
        }

    out = run_and_write_fractal_dimension_sweep(
        "configs/fractalrabbit_external.yaml",
        tmp_path / "fractal_dimension_sweep.json",
        dimensions=[1.1, 1.3],
        runner=runner,
    )
    payload = json.loads(Path(out).read_text(encoding="utf-8"))

    assert payload["status"] == "completed_with_failures"
    assert payload["rows"][0]["fractal_dimension"] == 1.1
    assert payload["errors"] == {"1.30": "runner returned no metrics"}
