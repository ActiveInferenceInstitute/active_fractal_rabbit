from __future__ import annotations

import json
from pathlib import Path

import pytest

from active_fractal_rabbit.pymdp_benchmark import (
    build_pymdp_benchmark_matrix,
    write_pymdp_benchmark_matrix,
)


def test_pymdp_benchmark_matrix_reports_no_sota_performance_claim(tmp_path) -> None:
    run_dir = tmp_path / "output" / "runs" / "default"
    run_dir.mkdir(parents=True)
    run_dir.joinpath("metrics.json").write_text(
        json.dumps(
            {
                "mean_predictive_loss": 2.0,
                "mean_posterior_entropy": 0.5,
                "pymdp_mean_predictive_loss": 2.4,
                "pymdp_mean_posterior_entropy": 0.7,
                "pymdp_exact_inference_max_abs_delta": 1e-7,
                "pymdp_source_observation_rows": 2400.0,
                "pymdp_inference_rows": 2400.0,
                "pymdp_validation_fraction": 1.0,
                "reference_calibration_persistence_brier": 0.3,
                "reference_calibration_base_rate_brier": 0.8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    hardened = tmp_path / "output" / "runs" / "hardened_anomaly_analysis.json"
    hardened.write_text(
        json.dumps(
            {
                "speed_regime_shift": {"markov_auc": 0.6},
                "preferred_set_drift": {"markov_auc": 0.55},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    factorial = tmp_path / "output" / "runs" / "factorial_pymdp_analysis_fixture.json"
    factorial.write_text(
        json.dumps(
            {
                "status": "completed",
                "performance_gate": "factorial_pymdp_not_performance_leading",
                "metrics": {
                    "factorial_pymdp_mean_predictive_loss": 2.2,
                    "factorial_pymdp_mean_posterior_entropy": 0.6,
                    "factorial_pymdp_matched_prior_max_abs_delta": 1e-6,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_pymdp_benchmark_matrix(tmp_path)

    assert payload["schema"] == "active-fractal-rabbit-pymdp-benchmark-matrix-v1"
    assert payload["state_of_art_performance_status"] == "not_demonstrated"
    assert payload["interpretability_status"] == "diagnostic_gain_demonstrated"
    assert payload["pymdp_minus_reference_predictive_loss"] == pytest.approx(0.4)
    assert payload["inferactively_pymdp_version"] == "1.0.4"
    assert payload["pymdp_agent_module"].endswith(str(Path("pymdp") / "agent.py"))
    assert payload["pymdp_package_path"].endswith(str(Path("pymdp") / "__init__.py"))
    assert payload["source_observation_rows"] == 2400.0
    assert payload["inference_rows"] == 2400.0
    assert payload["validation_fraction"] == 1.0
    assert {row["model"] for row in payload["rows"]} >= {
        "reference_filter",
        "pymdp_agent",
        "factorial_pymdp_agent",
        "future_sota_baselines",
    }
    factorial_row = next(
        row for row in payload["rows"] if row["model"] == "factorial_pymdp_agent"
    )
    assert (
        factorial_row["current_role"]
        == "multi-factor latent-mode active-inference diagnostic"
    )
    assert (
        factorial_row["performance_gate"] == "factorial_pymdp_not_performance_leading"
    )
    pymdp_row = next(row for row in payload["rows"] if row["model"] == "pymdp_agent")
    assert (
        pymdp_row["inferactively_pymdp_version"]
        == payload["inferactively_pymdp_version"]
    )
    assert pymdp_row["inference_rows"] == pymdp_row["source_observation_rows"] == 2400.0


def test_pymdp_benchmark_matrix_writes_artifact(tmp_path) -> None:
    run_dir = tmp_path / "output" / "runs" / "default"
    run_dir.mkdir(parents=True)
    run_dir.joinpath("metrics.json").write_text("{}\n", encoding="utf-8")

    out = write_pymdp_benchmark_matrix(
        tmp_path, tmp_path / "output" / "runs" / "pymdp_benchmark_matrix.json"
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["state_of_art_performance_status"] == "not_demonstrated"
