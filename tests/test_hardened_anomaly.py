from __future__ import annotations

import numpy as np
import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.hardened_anomaly import (
    markov_surprise_baseline,
    operating_point,
    run_hardened_anomaly_analysis,
)
from active_fractal_rabbit.observations import discretize_observations

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_markov_surprise_baseline_hand_example() -> None:
    cells = np.array([0, 0, 1, 1, 0])
    first = np.array([True, False, False, False, False])
    score = markov_surprise_baseline(cells, first, n_states=2, smoothing=0.5)

    assert score[0] == pytest.approx(0.0)
    assert score[1] < score[2]


def test_operating_point_returns_threshold_metrics() -> None:
    score = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([False, False, True, True])
    metrics = operating_point(score, labels, false_positive_rate=0.5)

    assert metrics["threshold"] == pytest.approx(0.2)
    assert metrics["true_positive_rate"] == pytest.approx(1.0)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)


def test_hardened_anomaly_analysis_reports_fair_baseline() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    sequence = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    result = run_hardened_anomaly_analysis(
        sequence, smoothing=config.inference.smoothing, seed=0
    )

    assert "speed_regime_shift" in result
    assert "preferred_set_drift" in result
    assert "timing_manipulation" in result
    assert "adversarial_missingness_proxy" in result
    assert 0.0 <= result["speed_regime_shift"]["vfe_auc"] <= 1.0
    assert 0.0 <= result["speed_regime_shift"]["markov_auc"] <= 1.0
    assert 0.0 <= result["timing_manipulation"]["vfe_auc"] <= 1.0
    assert 0.0 <= result["adversarial_missingness_proxy"]["markov_auc"] <= 1.0
    assert "vfe_operating_point" in result["preferred_set_drift"]
    assert "markov_operating_point" in result["preferred_set_drift"]
