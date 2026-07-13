from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import numpy as np
import pytest

from active_fractal_rabbit.config import FRACTALRABBIT_PINNED_COMMIT, load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.inference import holdout_split, run_holdout_scoring
from active_fractal_rabbit.observations import discretize_observations
from active_fractal_rabbit.pipeline import run_pipeline_from_path
from active_fractal_rabbit.pomdp import build_pomdp

FIXTURE = Path("data/fixtures/fractalrabbit_sample.csv").resolve()


def _config(tmp_path: Path, holdout_fraction: float) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    path = config_dir / f"holdout_{holdout_fraction}.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            run_id: holdout_{holdout_fraction}
            output_dir: {tmp_path / "output" / "runs" / f"holdout_{holdout_fraction}"}
            fractalrabbit:
              upstream_commit: {FRACTALRABBIT_PINNED_COMMIT}
              execution: fixture
              fixture_csv: {FIXTURE}
              params:
                number_of_travellers: 4
                number_of_days: 3.0
                number_of_reports: 6.0
            observation:
              spatial_bins: 3
              time_bins_per_day: 4
            model:
              smoothing: 0.05
            experiment:
              holdout_fraction: {holdout_fraction}
            closed_loop:
              iterations: 1
              candidates: []
            """
        ),
        encoding="utf-8",
    )
    return path


def _sequence():
    config = load_config("configs/default.yaml")
    observations = parse_waypoint_csv(FIXTURE)
    return discretize_observations(observations, config.inference)


def test_holdout_split_marks_last_rows_per_traveller() -> None:
    sequence = _sequence()
    mask = holdout_split(sequence, 0.25)

    assert mask.dtype == bool
    assert mask.sum() == 4  # one row per 4-row traveller
    by_traveller: dict[int, list[int]] = {}
    for idx, row in enumerate(sequence.observations):
        by_traveller.setdefault(row.traveller_id, []).append(idx)
    for indices in by_traveller.values():
        assert mask[indices[-1]]
        assert not mask[indices[:-1]].any()


def test_holdout_split_zero_fraction_and_bounds() -> None:
    sequence = _sequence()
    assert not holdout_split(sequence, 0.0).any()
    with pytest.raises(ValueError, match="between 0 and 0.95"):
        holdout_split(sequence, 0.96)


def test_holdout_split_protects_small_travellers() -> None:
    sequence = _sequence()
    mask = holdout_split(sequence, 0.5)
    by_traveller: dict[int, list[int]] = {}
    for idx, row in enumerate(sequence.observations):
        by_traveller.setdefault(row.traveller_id, []).append(idx)
    for indices in by_traveller.values():
        kept = sum(1 for idx in indices if not mask[idx])
        assert kept >= 2


def test_build_pomdp_include_mask_validation() -> None:
    sequence = _sequence()
    with pytest.raises(ValueError, match="length"):
        build_pomdp(sequence, 0.05, include=np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="at least one"):
        build_pomdp(
            sequence, 0.05, include=np.zeros(len(sequence.observations), dtype=bool)
        )


def test_masked_pomdp_differs_from_full_estimation() -> None:
    sequence = _sequence()
    mask = holdout_split(sequence, 0.25)
    full = build_pomdp(sequence, 0.05)
    masked = build_pomdp(sequence, 0.05, include=~mask)

    assert any(not np.allclose(a, b) for a, b in zip(full.A, masked.A))
    for control in range(full.B.shape[2]):
        assert np.allclose(full.B[:, :, control].sum(axis=0), 1.0)
        assert np.allclose(masked.B[:, :, control].sum(axis=0), 1.0)


def test_holdout_scoring_returns_finite_metrics() -> None:
    sequence = _sequence()
    mask = holdout_split(sequence, 0.25)
    model = build_pomdp(sequence, 0.05, include=~mask)
    scored = run_holdout_scoring(model, sequence, mask)

    assert scored["holdout_n_holdout"] == 4.0
    assert scored["holdout_n_train"] == 12.0
    assert math.isfinite(scored["holdout_mean_predictive_loss"])
    assert scored["holdout_mean_predictive_loss"] > 0


def test_holdout_scoring_empty_mask_is_honest() -> None:
    sequence = _sequence()
    model = build_pomdp(sequence, 0.05)
    scored = run_holdout_scoring(
        model, sequence, np.zeros(len(sequence.observations), dtype=bool)
    )

    assert scored["holdout_n_holdout"] == 0.0
    assert math.isnan(scored["holdout_mean_predictive_loss"])


def test_pipeline_holdout_knob_flip_changes_metrics(tmp_path) -> None:
    """Flip test: the knob must change emitted artifacts, not just parse."""

    disabled = run_pipeline_from_path(_config(tmp_path, 0.0))
    quarter = run_pipeline_from_path(_config(tmp_path, 0.25))
    half = run_pipeline_from_path(_config(tmp_path, 0.5))

    assert "holdout_mean_predictive_loss" not in disabled.metrics
    assert "holdout_mean_predictive_loss" in quarter.metrics
    assert quarter.metrics["holdout_n_holdout"] == 4.0
    assert half.metrics["holdout_n_holdout"] == 8.0
    assert (
        quarter.metrics["holdout_mean_predictive_loss"]
        != half.metrics["holdout_mean_predictive_loss"]
    )
    # additional-pass guarantee: primary metrics identical with and without holdout
    for key in [
        "mean_predictive_loss",
        "mean_posterior_entropy",
        "mean_policy_entropy",
    ]:
        assert disabled.metrics[key] == quarter.metrics[key]
    metrics_on_disk = json.loads(
        (quarter.output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics_on_disk["holdout_fraction_configured"] == 0.25


def test_quantile_and_burst_statistics_estimated_from_training_rows_only() -> None:
    """Leak gate: holdout rows must not shape quantile edges, auto bounds, or burst cutoff.

    Construct rows where the holdout tail contains extreme outliers. With
    estimation_mask excluding them, the training rows' bin assignments must
    match a training-only discretization; estimating on all rows instead
    must move at least one training bin (proving the mask is load-bearing).
    """

    from dataclasses import replace as dc_replace

    from active_fractal_rabbit.types import WaypointObservation

    config = load_config(Path("configs/default.yaml").resolve())
    quantile_obs = dc_replace(
        config.inference.observation,
        spatial_discretizer="quantile",
        spatial_bounds=None,
    )
    quantile_config = dc_replace(config.inference, observation=quantile_obs)
    train_rows = tuple(
        WaypointObservation(
            traveller_id=0,
            days=0.1 * (idx + 1),
            x_km=float(5 + idx),
            y_km=float(5 + idx),
        )
        for idx in range(6)
    )
    outliers = (
        WaypointObservation(traveller_id=0, days=0.7, x_km=100.0, y_km=100.0),
        WaypointObservation(traveller_id=0, days=0.8, x_km=200.0, y_km=200.0),
    )
    all_rows = train_rows + outliers
    mask = np.array([True] * 6 + [False] * 2)

    masked = discretize_observations(all_rows, quantile_config, estimation_mask=mask)
    train_only = discretize_observations(train_rows, quantile_config)
    leaky = discretize_observations(all_rows, quantile_config)

    # Training rows keep their training-only bins under the mask...
    assert np.array_equal(masked.x_bins[:6], train_only.x_bins)
    assert np.array_equal(masked.y_bins[:6], train_only.y_bins)
    # ...whereas estimating on all rows shifts the training bins (the leak).
    assert not np.array_equal(leaky.x_bins[:6], train_only.x_bins)
    # Outlier rows are still transformed (assigned to the top bin).
    assert masked.x_bins[6:].min() >= masked.x_bins[:6].max()


def test_holdout_pipeline_headline_metrics_still_unchanged_with_leak_fix(
    tmp_path,
) -> None:
    """The training-only re-discretization is internal to the holdout pass."""

    disabled = run_pipeline_from_path(_config(tmp_path, 0.0))
    enabled = run_pipeline_from_path(_config(tmp_path, 0.25))
    for key in [
        "mean_predictive_loss",
        "mean_posterior_entropy",
        "mean_policy_entropy",
    ]:
        assert disabled.metrics[key] == enabled.metrics[key]
    assert enabled.metrics["holdout_n_holdout"] > 0


def test_estimation_mask_validation() -> None:
    config = load_config(Path("configs/default.yaml").resolve())
    rows = parse_waypoint_csv(FIXTURE)
    with pytest.raises(ValueError, match="length"):
        discretize_observations(
            rows, config.inference, estimation_mask=np.array([True])
        )
    with pytest.raises(ValueError, match="at least two"):
        discretize_observations(
            rows, config.inference, estimation_mask=np.zeros(len(rows), dtype=bool)
        )
