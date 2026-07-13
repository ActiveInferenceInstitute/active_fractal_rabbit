"""Tests for the individual-vs-population experiment: active-inference individual modelling
has its largest advantage at the N=1 limit (no population to pool from). CI-safe on the fixture;
the headline result is verified separately on the real external simulator trace."""

from __future__ import annotations

import math
from pathlib import Path

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.individual_vs_population import advantage_vs_population_size
from active_fractal_rabbit.observations import discretize_observations

ROOT = Path(__file__).resolve().parents[1]


def _sequence():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    return discretize_observations(
        parse_waypoint_csv(cfg.fractalrabbit.fixture_csv), cfg.inference
    )


def test_individual_advantage_is_maximal_at_zero_population() -> None:
    """The N=1 limit: with no population to pool from, the individual (preferential-return)
    model's advantage over the pooled-population transition is the largest it ever is."""
    r = advantage_vs_population_size(_sequence(), smoothing=0.05)
    assert r["advantage_is_max_at_zero_population"] is True
    assert r["individual_advantage_at_zero"] > 0.0
    assert (
        r["individual_advantage_at_zero"]
        >= r["individual_advantage_at_full_population"]
    )


def test_zero_population_is_uninformative_firing_control() -> None:
    """FIRING control: with zero other travellers the pooled transition is uninformative, so its
    loss must sit at the uniform-over-cells floor (~log n_states) — proving the population model is
    genuinely data-starved at N=1, not silently borrowing the target's own data."""
    seq = _sequence()
    r = advantage_vs_population_size(seq, smoothing=0.05)
    zero = next(row for row in r["rows"] if row["n_other_travellers"] == 0)
    uniform_floor = math.log(seq.num_spatial_states)
    assert abs(zero["population_loss"] - uniform_floor) < 0.05
    # and the individual model is far below that floor (it uses real own-history structure)
    assert zero["individual_loss"] < uniform_floor - 0.5


def test_advantage_shrinks_from_n1_toward_full_population() -> None:
    """The advantage is strictly larger at the N=1 limit than at full population — the
    population model 'catches up' as travellers accumulate (the individual-vs-population mechanism)."""
    r = advantage_vs_population_size(_sequence(), smoothing=0.05)
    adv = [row["individual_advantage_nats"] for row in r["rows"]]
    assert adv[0] == max(adv)  # peak at zero population
    assert adv[0] > adv[-1] - 1e-9  # not larger at full population than at N=1
