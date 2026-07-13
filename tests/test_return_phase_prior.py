from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from active_fractal_rabbit.return_phase_prior import (
    run_and_write_return_phase_prior,
    run_return_phase_prior_analysis,
)
from active_fractal_rabbit.types import ObservationSequence, WaypointObservation

ROOT = Path(__file__).resolve().parents[1]


def _sequence(cells: list[int], phases: list[int]) -> ObservationSequence:
    n = len(cells)
    return ObservationSequence(
        observations=tuple(
            WaypointObservation(0, float(idx), float(cell), 0.0)
            for idx, cell in enumerate(cells)
        ),
        is_first_observation=np.array([True] + [False] * (n - 1)),
        x_bins=np.asarray(cells),
        y_bins=np.zeros(n, dtype=int),
        spatial_states=np.asarray(cells),
        time_bins=np.asarray(phases),
        speed_categories=np.zeros(n, dtype=int),
        burst_categories=np.zeros(n, dtype=int),
        speeds_km_per_day=np.zeros(n),
        intervals_days=np.ones(n),
        spatial_bins=2,
        num_spatial_states=2,
        num_time_bins=2,
        num_speed_categories=1,
        num_burst_categories=1,
        spatial_bounds=(0.0, 1.0, 0.0, 1.0),
    )


def test_return_phase_prior_promotes_only_when_combined_prior_clears_nulls() -> None:
    sequence = _sequence([0, 0, 1, 0, 1], [0, 1, 1, 1, 1])

    payload = run_return_phase_prior_analysis(
        sequence, smoothing=0.1, seed=0, promotion_threshold=0.01
    )

    assert payload["schema"] == "active-fractal-rabbit-return-phase-prior-v1"
    assert payload["verdict"] == "promoted_return_phase_prior"
    assert payload["best_model"] == "return_plus_phase_prior"
    assert payload["return_plus_phase_gain_vs_persistence"] > 0.01
    assert payload["return_plus_phase_gain_over_permuted"] > 0.01


def test_return_phase_prior_honest_null_under_strict_threshold() -> None:
    sequence = _sequence([0, 0, 1], [0, 0, 1])

    payload = run_return_phase_prior_analysis(
        sequence, smoothing=1.0, seed=0, promotion_threshold=99.0
    )

    assert payload["verdict"] == "honest_null"
    assert {row["model"] for row in payload["rows"]} >= {
        "pooled_prior",
        "phase_prior",
        "return_prior",
        "return_plus_phase_prior",
        "permuted_return_plus_phase_prior",
    }


def test_return_phase_writer_emits_fixture_json(tmp_path) -> None:
    out = tmp_path / "return_phase_prior_analysis.json"
    run_and_write_return_phase_prior(ROOT / "configs" / "default.yaml", out)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["schema"] == "active-fractal-rabbit-return-phase-prior-v1"
    assert payload["status"] == "completed"
    assert payload["n_scored_steps"] == 2395
    assert payload["best_mean_predictive_loss"] == pytest.approx(
        min(
            row["mean_predictive_loss"]
            for row in payload["rows"]
            if row["model"]
            in {
                "pooled_prior",
                "phase_prior",
                "return_prior",
                "return_plus_phase_prior",
            }
        )
    )
