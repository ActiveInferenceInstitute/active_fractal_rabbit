"""Tests for the regime-switch phase diagnostic — real data, no mocks.

Covers the position-within-traveller phase reconstruction (the invariant this whole
module depends on: that ``score_candidate`` emits steps for a held-out traveller
contiguously and in generation order, so the k-th occurrence of a traveller id in
``step_travellers`` is observation index k+1) and the end-to-end per-segment breakdown.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from active_fractal_rabbit.regime_switch_phase_diagnostic import (
    REGIME_SWITCH_PHASE_DIAGNOSTIC_SCHEMA,
    _step_phase_segments,
    run_and_write_regime_switch_phase_diagnostic,
    run_regime_switch_phase_diagnostic,
)
from active_fractal_rabbit.sota_stressors import REGIME_STRESSOR_PHASE_SEGMENTS

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "default.yaml"


def test_step_phase_segments_maps_position_to_named_segment_by_generator_formula() -> (
    None
):
    """Direct proof of the reconstruction formula: for n_reports=11 (10 scored steps per
    traveller, k=0..9), the segment for scored step k must be attributed by
    phase=k/(n_reports-1) - the phase of the TRANSITION THAT PRODUCED that observation
    (generate_regime_stressor_waypoints computes phase from the row it just appended,
    THEN uses it to advance to the next row, so row idx>=1's producing phase is
    (idx-1)/(n_reports-1), i.e. k/(n_reports-1) for the k-th occurrence / idx=k+1).
    A prior version of this table (and of the implementation) used (k+1)/(n_reports-1)
    instead - a real off-by-one a cross-vendor audit caught; this table is the fix,
    not the original bug."""

    # traveller 0 contributes 10 steps (k=0..9), traveller 1 contributes 10 steps too.
    step_travellers = np.array([0] * 10 + [1] * 10)
    segments = _step_phase_segments(step_travellers, n_reports=11)
    # k=0..9 -> phase=0.0..0.9; boundaries at 0.34, 0.68
    expected_single_traveller = [
        "commute",  # k=0 phase .00
        "commute",  # k=1 phase .10
        "commute",  # k=2 phase .20
        "commute",  # k=3 phase .30
        "explore",  # k=4 phase .40
        "explore",  # k=5 phase .50
        "explore",  # k=6 phase .60
        "return_stabilization",  # k=7 phase .70
        "return_stabilization",  # k=8 phase .80
        "return_stabilization",  # k=9 phase .90
    ]
    assert list(segments[:10]) == expected_single_traveller
    assert list(segments[10:]) == expected_single_traveller
    assert set(segments) <= set(REGIME_STRESSOR_PHASE_SEGMENTS)


def test_step_phase_segments_matches_generator_at_the_production_boundary() -> None:
    """Regression test for the exact off-by-one a cross-vendor audit found at the
    production n_reports=260 scale, tied to the real generator constants rather than a
    hand-typed table: the transition FROM generator row idx=88 TO idx=89 is governed by
    phase(88)=88/259=0.33977<0.34 ('commute' params), so the scored step whose loss
    measures the difficulty of arriving at row 89 (the k=88-th occurrence, since idx=k+1)
    must be attributed to 'commute', not 'explore' (which phase(89)=89/259=0.34363 would
    wrongly suggest if bucketed by the arriving row's own index instead of the departing
    one that produced it)."""

    from active_fractal_rabbit.sota_stressors import DEFAULT_REGIME_REPORTS

    # A single traveller's full 259 scored steps (k=0..258); only the boundary matters.
    step_travellers = np.zeros(DEFAULT_REGIME_REPORTS - 1, dtype=int)
    segments = _step_phase_segments(step_travellers, n_reports=DEFAULT_REGIME_REPORTS)
    assert segments[88] == "commute"  # k=88 (idx=89): producing phase(88)=88/259<0.34
    assert segments[89] == "explore"  # k=89 (idx=90): producing phase(89)=89/259>=0.34


def test_step_phase_segments_raises_on_step_count_mismatch() -> None:
    """Proof-of-detection: if a traveller's step count doesn't match n_reports-1, the
    reconstruction must fail loudly, not silently mislabel phases."""

    step_travellers = np.array([0, 0, 0])  # only 3 steps, but n_reports implies 10
    with pytest.raises(ValueError, match="expected"):
        _step_phase_segments(step_travellers, n_reports=11)


def test_regime_switch_phase_diagnostic_runs_and_covers_all_segments() -> None:
    payload = run_regime_switch_phase_diagnostic(CONFIG, n_boot=25)

    assert payload["schema"] == REGIME_SWITCH_PHASE_DIAGNOSTIC_SCHEMA
    assert payload["status"] == "completed"
    assert payload["regime_id"] == "latent_regime_switch_035"
    segments = {row["segment"]: row for row in payload["segments"]}
    assert set(segments) == set(REGIME_STRESSOR_PHASE_SEGMENTS)
    for row in segments.values():
        assert row["n_steps"] > 0
        assert row["n_travellers"] >= 2
        assert (
            row["clustered_ci_low"]
            <= row["mean_advantage_nats"]
            <= row["clustered_ci_high"]
        )
    assert payload["worst_segment"] in segments
    assert payload["best_segment"] in segments
    assert isinstance(payload["uniform_disadvantage"], bool)


def test_writer_emits_valid_json(tmp_path: Path) -> None:
    out_path = tmp_path / "regime_switch_phase_diagnostic.json"
    written = run_and_write_regime_switch_phase_diagnostic(
        CONFIG, out_path=out_path, n_boot=25
    )

    assert written == out_path
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == REGIME_SWITCH_PHASE_DIAGNOSTIC_SCHEMA
    assert len(payload["segments"]) == 3
