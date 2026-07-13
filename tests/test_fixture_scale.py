from __future__ import annotations

from collections import Counter

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.observations import discretize_observations


def test_large_fixture_is_canonical_doubled_analysis_series() -> None:
    config = load_config("configs/default.yaml")
    rows = parse_waypoint_csv(config.fractalrabbit.fixture_csv)
    counts = Counter(row.traveller_id for row in rows)

    assert len(rows) == 2400
    assert len(counts) == 5
    assert set(counts.values()) == {480}
    assert config.fractalrabbit.params.number_of_days == 36.0
    assert config.fractalrabbit.params.number_of_reports == 480.0
    assert max(row.days for row in rows) <= 36.0
    assert min(row.days for row in rows) >= 0.0

    sequence = discretize_observations(rows, config.inference)
    assert sequence.spatial_bins == 4
    assert sequence.num_spatial_states == 16
    assert len(set(sequence.spatial_states.tolist())) == 16
    assert set(sequence.speed_categories.tolist()) == {0, 1, 2, 3}
    assert set(sequence.burst_categories.tolist()) == {0, 1}


def test_config_brackets_10x_default_and_exposes_robustness() -> None:
    config = load_config("configs/default.yaml")
    external = load_config("configs/fractalrabbit_external.yaml")

    assert [
        candidate.params.number_of_reports
        for candidate in config.closed_loop.candidates
    ] == [
        320.0,
        960.0,
    ]
    assert [
        candidate.params.number_of_reports
        for candidate in external.closed_loop.candidates
    ] == [
        360.0,
        1440.0,
    ]
    assert config.robustness.n_realizations == 10
    assert config.robustness.upstream_candidate_limit == 5
    assert config.robustness.stop_on_external_unavailable is True
