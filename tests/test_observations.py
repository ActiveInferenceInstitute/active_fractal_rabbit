from __future__ import annotations

import numpy as np

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.observations import discretize_observations


def test_discretization_is_deterministic() -> None:
    config = load_config("configs/default.yaml").inference
    rows = parse_waypoint_csv("data/fixtures/fractalrabbit_sample.csv")

    first = discretize_observations(rows, config)
    second = discretize_observations(tuple(reversed(rows)), config)

    assert np.array_equal(first.spatial_states, second.spatial_states)
    assert np.array_equal(first.time_bins, second.time_bins)
    assert first.spatial_bins == 4
    assert first.num_spatial_states == 16
    assert set(first.burst_categories.tolist()) <= {0, 1}
    assert np.array_equal(first.is_first_observation, second.is_first_observation)


def test_speed_categories_include_motion() -> None:
    config = load_config("configs/default.yaml").inference
    rows = parse_waypoint_csv("data/fixtures/fractalrabbit_sample.csv")
    sequence = discretize_observations(rows, config)

    assert sequence.speed_categories.max() > 0
    assert sequence.speed_categories.min() == 0


def test_first_observations_use_interval_sentinel() -> None:
    config = load_config("configs/default.yaml").inference
    rows = parse_waypoint_csv("data/fixtures/fractalrabbit_sample.csv")
    sequence = discretize_observations(rows, config)

    assert int(sequence.is_first_observation.sum()) == 4
    assert np.all(sequence.intervals_days[sequence.is_first_observation] == 0.0)
    assert np.all(sequence.speeds_km_per_day[sequence.is_first_observation] == 0.0)
    assert np.all(sequence.burst_categories[sequence.is_first_observation] == 0)


def test_grouped_discretizers_are_deterministic() -> None:
    quantile_config = load_config("configs/sweep.yaml").inference
    fixed_config = load_config("configs/fractalrabbit_external.yaml").inference
    rows = parse_waypoint_csv("data/fixtures/fractalrabbit_sample.csv")

    quantile = discretize_observations(rows, quantile_config)
    fixed = discretize_observations(rows, fixed_config)

    assert quantile.spatial_bins == 4
    assert quantile.num_spatial_states == 16
    assert fixed.num_spatial_states == 16
    assert fixed.spatial_bounds == (-250.0, 250.0, -250.0, 250.0)
    assert set(quantile.x_bins.tolist()) <= {0, 1, 2, 3}
    assert set(fixed.y_bins.tolist()) <= {0, 1, 2, 3}
