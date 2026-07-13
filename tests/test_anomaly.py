"""Routine-break detection via variational free energy (surprise).

Honest evaluation: a clean-trained generative model makes injected routine-breaks
surprising, and surprise separates anomalies from normal steps (ROC AUC) far above
a first-occurrence novelty baseline. Detectability is reported across anomaly
strengths (clear rare-cell vs subtler random-cell) so no single near-perfect
number is overclaimed.
"""

from __future__ import annotations

import numpy as np
import pytest

from active_fractal_rabbit.anomaly import (
    detection_auc,
    novelty_baseline,
    run_anomaly_detection,
)
from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.observations import discretize_observations

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _fixture_sequence():
    config = load_config(ROOT / "configs" / "default.yaml")
    sequence = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    return config, sequence


def test_detection_auc_hand_example() -> None:
    """Mann-Whitney AUC on a worked example: 2 positives clearly above 2 negatives."""

    score = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([False, False, True, True])
    mask = np.array([True, True, True, True])
    assert detection_auc(score, labels, mask) == pytest.approx(1.0)
    # A reversed score gives AUC 0.
    assert detection_auc(-score, labels, mask) == pytest.approx(0.0)


def test_detection_auc_uses_average_ranks_for_ties() -> None:
    score = np.array([0.0, 1.0, 1.0, 2.0])
    labels = np.array([False, True, False, True])
    mask = np.ones(4, dtype=bool)
    assert detection_auc(score, labels, mask) == pytest.approx(0.875)


def test_detection_auc_handles_expanded_arrays_without_pairwise_loop() -> None:
    score = np.tile(np.arange(25, dtype=float), 400)
    labels = np.zeros(score.size, dtype=bool)
    labels[::11] = True
    mask = np.ones(score.size, dtype=bool)
    auc = detection_auc(score, labels, mask)
    assert 0.0 <= auc <= 1.0


def test_novelty_baseline_marks_first_occurrence() -> None:
    cells = np.array([0, 0, 1, 0, 2])
    first = np.zeros(5, dtype=bool)
    novelty = novelty_baseline(cells, first)
    assert list(novelty) == [1.0, 0.0, 1.0, 0.0, 1.0]


def test_surprise_detects_rare_cell_breaks_above_novelty() -> None:
    config, sequence = _fixture_sequence()
    result = run_anomaly_detection(
        sequence, smoothing=config.inference.smoothing, seed=0, mode="rare"
    )
    assert result.n_anomalies >= 3
    assert result.mean_surprise_anomaly > result.mean_surprise_normal
    assert result.vfe_auc > 0.6
    assert result.vfe_auc > result.novelty_auc + 0.1


def test_surprise_detects_subtler_random_cell_breaks() -> None:
    config, sequence = _fixture_sequence()
    result = run_anomaly_detection(
        sequence, smoothing=config.inference.smoothing, seed=0, mode="random"
    )
    # A random-cell break is subtler, but surprise should still beat novelty.
    assert result.vfe_auc > result.novelty_auc
    assert result.vfe_auc > 0.6


def test_anomaly_detection_deterministic() -> None:
    config, sequence = _fixture_sequence()
    a = run_anomaly_detection(sequence, smoothing=config.inference.smoothing, seed=1)
    b = run_anomaly_detection(sequence, smoothing=config.inference.smoothing, seed=1)
    assert a.vfe_auc == pytest.approx(b.vfe_auc)
