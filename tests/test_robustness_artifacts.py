from __future__ import annotations

import json
from pathlib import Path

import pytest

from active_fractal_rabbit.robustness import (
    compare_upstream_metrics,
    empirical_interval,
    run_and_write_upstream_sensitivity,
    summarize_variance_bands,
)


def test_empirical_interval_uses_order_statistics() -> None:
    summary = empirical_interval([1.0, 2.0, 3.0, 4.0, 100.0])

    assert summary["n"] == 5
    assert summary["median"] == 3.0
    assert summary["low"] == 1.0
    assert summary["high"] == 100.0


def test_variance_bands_summarize_requested_metrics() -> None:
    bands = summarize_variance_bands(
        [
            {"gap_filling_skill": 0.1, "frontier_no_location_mass": 0.2},
            {"gap_filling_skill": 0.3, "frontier_no_location_mass": 0.4},
        ]
    )

    assert bands["schema"] == "active-fractal-rabbit-variance-bands-v1"
    assert bands["metrics"]["gap_filling_skill"]["median"] == 0.2
    assert bands["metrics"]["frontier_no_location_mass"]["n"] == 2


def test_upstream_sensitivity_blocked_artifact_when_no_candidate_builds(
    tmp_path,
) -> None:
    out = run_and_write_upstream_sensitivity(
        tmp_path / "upstream_sensitivity.json",
        current_commit="current",
        candidate_limit=2,
        candidate_provider=lambda limit: ["a", "b"][:limit],
        runner=lambda sha: None,
    )
    payload = json.loads(Path(out).read_text(encoding="utf-8"))

    assert payload["schema"] == "active-fractal-rabbit-upstream-sensitivity-v1"
    assert payload["status"] == "blocked"
    assert payload["attempted_commits"] == ["a", "b"]
    assert set(payload["attempt_errors"]) == {"a", "b"}


def test_upstream_sensitivity_completed_artifact_from_first_success(tmp_path) -> None:
    out = run_and_write_upstream_sensitivity(
        tmp_path / "upstream_sensitivity.json",
        current_commit="current",
        candidate_limit=3,
        candidate_provider=lambda limit: ["bad", "good", "ignored"][:limit],
        runner=lambda sha: None
        if sha == "bad"
        else {"predictive_loss_abs_delta": 0.25},
    )
    payload = json.loads(Path(out).read_text(encoding="utf-8"))

    assert payload["status"] == "completed"
    assert payload["selected_commit"] == "good"
    assert payload["attempted_commits"] == ["bad", "good", "ignored"]
    assert payload["metrics"] == {"predictive_loss_abs_delta": 0.25}
    assert "ignored" not in payload["attempt_errors"]


def test_upstream_sensitivity_records_runner_exceptions(tmp_path) -> None:
    def runner(sha: str):
        raise RuntimeError(f"{sha} failed to build")

    out = run_and_write_upstream_sensitivity(
        tmp_path / "upstream_sensitivity.json",
        current_commit="current",
        candidate_limit=1,
        candidate_provider=lambda limit: ["broken"][:limit],
        runner=runner,
    )
    payload = json.loads(Path(out).read_text(encoding="utf-8"))

    assert payload["status"] == "blocked"
    assert "failed to build" in payload["attempt_errors"]["broken"]


def test_compare_upstream_metrics_reports_requested_absolute_deltas() -> None:
    current = {
        "mean_predictive_loss": 2.0,
        "mean_posterior_entropy": 0.5,
        "state_recovery_reference_mean_mass_on_true_cell": 0.8,
        "n_latent_states": 16,
    }
    candidate = {
        "mean_predictive_loss": 2.25,
        "mean_posterior_entropy": 0.4,
        "state_recovery_reference_mean_mass_on_true_cell": 0.7,
        "n_latent_states": 25,
    }

    assert compare_upstream_metrics(current, candidate) == pytest.approx(
        {
            "predictive_loss_abs_delta": 0.25,
            "posterior_entropy_abs_delta": 0.1,
            "reference_recovery_mass_abs_delta": 0.1,
            "latent_state_count_abs_delta": 9.0,
        }
    )
