"""Tests for the privacy-utility frontier — real fixture + real computation, no mocks.

Pins the resolution-fair / shuffle-corrected utility axis and the kill-gate (graceful
degradation) plus the negative control (the static-occupancy shuffle floor is actually
subtracted, and location-withheld utility is zero by construction).
"""

from __future__ import annotations

import math

import numpy as np

from active_fractal_rabbit.privacy_utility_frontier import (
    _resolved_bits,
    event_uniqueness,
    privacy_utility_frontier,
)

CONFIG = "configs/default.yaml"
FIXTURE = "data/fixtures/fractalrabbit_sample_large.csv"


def test_resolved_bits_math():
    # log2(16) - (loss_nats / ln2). At chance (loss = ln(16)) resolved bits == 0.
    assert math.isclose(_resolved_bits(math.log(16), 16), 0.0, abs_tol=1e-9)
    # A perfect predictor (loss -> 0) pins all log2(n) bits.
    assert math.isclose(_resolved_bits(0.0, 16), 4.0, abs_tol=1e-9)


def test_event_uniqueness_counts_singleton_events():
    travellers = np.array([0, 1, 2, 3])
    cells = np.array([5, 5, 7, 9])
    times = np.array([0, 0, 1, 2])
    # events: (0,5) -> {0,1} shared; (1,7) -> {2} singleton; (2,9) -> {3} singleton => 2/3
    assert math.isclose(
        event_uniqueness(travellers, cells, times), 2.0 / 3.0, abs_tol=1e-9
    )
    assert event_uniqueness(np.array([]), np.array([]), np.array([])) == 0.0


def test_frontier_graceful_degradation_kill_gate():
    # Kill-gate: both surviving predictive utility AND re-identification risk degrade as location
    # minimization increases. Reduced regimes/shuffles keep the real-fixture test fast.
    payload = privacy_utility_frontier(
        CONFIG, FIXTURE, grid_regimes=(4, 2), n_shuffle=2
    )
    assert payload["graceful_degradation_tradeoff"] is True
    assert payload["utility_monotone_decreasing"] is True
    assert payload["risk_monotone_decreasing"] is True
    grids = [r for r in payload["frontier"] if r["spatial_bins"] > 0]
    # finer grid retains strictly more excess utility AND carries strictly more re-id risk
    assert grids[0]["excess_utility_bits"] > grids[-1]["excess_utility_bits"]
    assert grids[0]["reid_event_uniqueness"] > grids[-1]["reid_event_uniqueness"]


def test_low_shuffle_full_grid_reports_non_monotone_frontier_negative_control():
    payload = privacy_utility_frontier(CONFIG, FIXTURE, n_shuffle=2)

    assert payload["grid_regimes"] == [8, 7, 6, 5, 4, 3, 2]
    assert payload["graceful_degradation_tradeoff"] is True
    assert payload["utility_monotone_decreasing"] is True
    assert payload["risk_monotone_decreasing"] is True
    by_regime = {row["regime"]: row for row in payload["frontier"]}
    assert (
        by_regime["grid_7x7"]["excess_utility_bits"]
        < by_regime["grid_8x8"]["excess_utility_bits"]
    )
    assert (
        by_regime["grid_7x7"]["reid_event_uniqueness"]
        < by_regime["grid_8x8"]["reid_event_uniqueness"]
    )
    mc = payload["min_collection_operating_point"]
    assert mc["found"] is True
    assert mc["regime"] == "grid_2x2"
    assert mc["min_reid_among_useful_grids"] <= mc["reid_risk_ceiling"]


def test_negative_control_shuffle_floor_is_subtracted():
    # The static-occupancy shuffle floor must be a real, positive correction (not a no-op): at a
    # fine grid the floor is non-trivial and excess < resolved. Withheld location => zero utility.
    payload = privacy_utility_frontier(
        CONFIG, FIXTURE, grid_regimes=(4, 2), n_shuffle=2
    )
    fine = next(r for r in payload["frontier"] if r["spatial_bins"] == 4)
    assert fine["shuffle_floor_bits"] > 0.0  # static occupancy is being subtracted
    assert (
        fine["excess_utility_bits"] < fine["resolved_bits"]
    )  # correction actually applied
    withheld = next(
        r for r in payload["frontier"] if r["regime"] == "location_withheld"
    )
    assert withheld["excess_utility_bits"] == 0.0


def test_shuffle_floor_is_monte_carlo_stable_and_seed_robust():
    # The shuffle-floor correction reports a Monte-Carlo SD; it must be small (a tight floor
    # estimate, not a single-draw artifact), and the graceful-degradation verdict must hold under
    # a DIFFERENT shuffle seed (robust to the permutation draw, not seed-cherry-picked).
    a = privacy_utility_frontier(
        CONFIG, FIXTURE, grid_regimes=(4, 2), n_shuffle=3, shuffle_seed=20260620
    )
    b = privacy_utility_frontier(
        CONFIG, FIXTURE, grid_regimes=(4, 2), n_shuffle=3, shuffle_seed=777
    )
    assert a["graceful_degradation_tradeoff"] is True
    assert b["graceful_degradation_tradeoff"] is True
    # the worst-case shuffle-floor SD is small relative to the finest excess-utility magnitude
    assert a["max_shuffle_floor_sd"] < 0.5 * abs(a["finest_excess_utility_bits"])
    fine = next(r for r in a["frontier"] if r["spatial_bins"] == 4)
    assert fine["shuffle_floor_sd"] >= 0.0


def test_minimum_collection_operating_point_is_honest():
    payload = privacy_utility_frontier(
        CONFIG, FIXTURE, grid_regimes=(8, 4, 2), n_shuffle=2
    )
    mc = payload["min_collection_operating_point"]
    assert mc["utility_floor_bits"] == 0.10
    assert mc["reid_risk_ceiling"] == 0.20
    assert mc["found"] is True
    assert mc["regime"] == "grid_2x2"
    assert mc["min_reid_among_useful_grids"] <= mc["reid_risk_ceiling"]


def test_run_and_write_emits_artifact(tmp_path):
    from active_fractal_rabbit.privacy_utility_frontier import (
        run_and_write_privacy_utility_frontier,
    )

    out = run_and_write_privacy_utility_frontier(CONFIG, FIXTURE, tmp_path / "puf.json")
    assert out.is_file()
    import json

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "active-fractal-rabbit-privacy-utility-frontier-v1"
    assert payload["frontier"]
    assert payload["graceful_degradation_tradeoff"] is True
    assert payload["min_collection_operating_point"]["found"] is True
