"""Tests for the confound-control battery + mobility check. Locks the battery mechanics and the
clears-confounds gate. The headline confound-controlled positive (the long-memory model clears the
battery on a mobility-checked finer-grid real regime, 5-6/6 seeds) is verified on the external trace
and documented; CI checks the machinery on the fixture (real data, no mocks)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.confound_controls import (
    clears_confounds,
    confound_battery,
    mobility_summary,
)
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.observations import discretize_observations

ROOT = Path(__file__).resolve().parents[1]


def _sequence(spatial_bins: int | None = None):
    cfg = load_config(ROOT / "configs" / "default.yaml")
    inf = cfg.inference
    if spatial_bins is not None:
        inf = replace(
            cfg.inference,
            observation=replace(cfg.inference.observation, spatial_bins=spatial_bins),
        )
    return discretize_observations(
        parse_waypoint_csv(cfg.fractalrabbit.fixture_csv), inf
    )


def test_battery_returns_four_positive_baselines() -> None:
    """All four baselines are scored and positive. (Note: the oracle-prev baseline uses the POOLED
    transition with the true previous cell, so on confined data where travellers differ it is not
    universally the lowest — a same-distribution dominant-cell baseline can beat it.)"""
    b = confound_battery(_sequence(), smoothing=0.05)
    assert set(b) == {
        "dominant_cell",
        "first_order_pooled",
        "return_boosted_soft",
        "oracle_prev",
    }
    assert all(v > 0 for v in b.values())


def test_clears_confounds_gate_logic() -> None:
    """The gate credits a win only when the model beats dominant-cell AND first-order AND
    return-boosted-soft; one loss fails it. Oracle gap is reported, never required to clear."""
    import pytest

    base = {
        "dominant_cell": 1.0,
        "first_order_pooled": 0.9,
        "return_boosted_soft": 0.8,
        "oracle_prev": 0.3,
    }
    assert clears_confounds(0.7, base)["clears_confounds"] is True
    assert (
        clears_confounds(0.85, base)["clears_confounds"] is False
    )  # loses to return_boosted_soft
    assert clears_confounds(0.7, base)["gap_to_oracle_nats"] == pytest.approx(0.4)


def test_mobility_summary_flags_confinement() -> None:
    """The fixture at a fine grid is sparse/confined; mobility_summary reports the per-traveller
    distinct-cell structure and the is_mobile flag used to gate model claims."""
    m = mobility_summary(_sequence())
    assert m["mean_distinct_cells"] > 0
    assert 0.0 <= m["confined_fraction"] <= 1.0
    assert isinstance(m["is_mobile"], bool)
    assert len(m["per_traveller"]) == len(
        {r.traveller_id for r in _sequence().observations}
    )


def test_mobility_provenance_fields_echo_thresholds() -> None:
    """The provenance block must faithfully echo the exact thresholds used, so a mobile/confined
    verdict is traceable to the cutoffs rather than an opaque bool. Guards against a wrong or dropped
    provenance field."""
    m = mobility_summary(
        _sequence(),
        min_mean_distinct_cells=4.0,
        confined_cell_max=2,
        max_confined_fraction=0.25,
    )
    assert m["mobility_thresholds"] == {
        "min_mean_distinct_cells": 4.0,
        "confined_cell_max": 2,
        "max_confined_fraction": 0.25,
    }
    assert isinstance(m["mean_distinct_meets_threshold"], bool)
    assert isinstance(m["confined_fraction_within_threshold"], bool)


def test_mobility_meets_threshold_comparator_both_sides() -> None:
    """mean_distinct_meets_threshold must be True when mean distinct cells clear the floor and False
    when they do not. Two calls straddle the fixture's mean (~10.6 distinct cells) so a flipped
    comparator (``<=`` vs ``>=``) or a mislabelled field is caught. is_mobile must follow suit."""
    seq = _sequence()
    met = mobility_summary(seq, min_mean_distinct_cells=1.0)
    assert met["mean_distinct_meets_threshold"] is True
    # confined_fraction is 0 at this grid, so clearing the distinct floor makes the regime mobile.
    assert met["is_mobile"] is True

    not_met = mobility_summary(seq, min_mean_distinct_cells=1000.0)
    assert not_met["mean_distinct_meets_threshold"] is False
    # a failed distinct-cell floor forces is_mobile False regardless of confinement.
    assert not_met["is_mobile"] is False


def test_mobility_confined_fraction_comparator_both_sides() -> None:
    """confined_fraction_within_threshold must be True when the confined share sits at/below the cap
    and False when it exceeds it. Raising confined_cell_max above the cell count marks every traveller
    confined (fraction 1.0), exceeding the 0.5 cap; the small default keeps it within. Guards the
    ``<=`` direction and that is_mobile ANDs both gates."""
    seq = _sequence()
    within = mobility_summary(seq, confined_cell_max=3, max_confined_fraction=0.5)
    assert within["confined_fraction"] <= 0.5
    assert within["confined_fraction_within_threshold"] is True

    exceeded = mobility_summary(
        seq, confined_cell_max=10_000, max_confined_fraction=0.5
    )
    assert exceeded["confined_fraction"] == 1.0
    assert exceeded["confined_fraction_within_threshold"] is False
    # every traveller counted confined ⇒ not mobile even though the distinct-cell floor is cleared.
    assert exceeded["is_mobile"] is False
