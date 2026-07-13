"""Recoverability / minimization frontier across modality subsets.

Pins the oversight-relevant facts: full location is the recovery ceiling, and
withholding location collapses point recovery — quantifying how protective
location-minimization is and how little the specific cell is recoverable from
metadata alone.
"""

from __future__ import annotations

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.frontier import recoverability_frontier
from active_fractal_rabbit.observations import discretize_observations

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _frontier():
    config = load_config(ROOT / "configs" / "default.yaml")
    sequence = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    rows = recoverability_frontier(sequence, smoothing=config.inference.smoothing)
    return {row["modalities"]: row for row in rows}


def test_full_location_is_the_recovery_ceiling() -> None:
    f = _frontier()
    masses = {k: v["mean_mass_on_true_cell"] for k, v in f.items()}
    assert all(0.0 <= m <= 1.0 for m in masses.values())
    assert masses["all"] == max(masses.values())  # location-bearing set recovers best


def test_withholding_location_collapses_point_recovery() -> None:
    f = _frontier()
    # Dropping location substantially reduces mean mass on the true cell — the
    # oversight-positive statement that location-minimization is protective.
    assert (
        f["no_location"]["mean_mass_on_true_cell"]
        < 0.6 * f["all"]["mean_mass_on_true_cell"]
    )


def test_every_subset_is_present_and_well_formed() -> None:
    f = _frontier()
    for name in (
        "all",
        "no_location",
        "metadata_time_burst",
        "speed_only",
        "time_only",
        "burst_only",
    ):
        assert name in f
        assert 0.0 <= f[name]["occupancy_tv"] <= 1.0
