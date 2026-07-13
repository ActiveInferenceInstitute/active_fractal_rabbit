from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.power_analysis import generate_synthetic_waypoints
from active_fractal_rabbit.sota_program import score_candidate
from active_fractal_rabbit.sota_stressors import (
    DEFAULT_REPORTING_GAP_MIN_KEEP,
    DIAGNOSTIC_STRESSOR_LANES,
    all_lane_definitions,
    build_adversarial_missingness_partial_observability_dataset,
    build_regime_stressor_partial_observability_dataset,
    build_regime_stressor_repeat_partial_observability_dataset,
    build_reporting_gap_partial_observability_dataset,
    diagnostic_lane_definitions,
    generate_regime_stressor_waypoints,
    lane_definitions,
    phase_segment_for_phase,
    thin_adversarial_missingness_waypoints,
    thin_reporting_gap_waypoints,
)
from active_fractal_rabbit.sota_baselines import RecencyAttentionCandidate

ROOT = Path(__file__).resolve().parents[1]


def _inference(bins: int = 6):
    cfg = load_config(ROOT / "configs" / "default.yaml")
    return replace(
        cfg.inference, observation=replace(cfg.inference.observation, spatial_bins=bins)
    )


def test_reporting_gap_thinning_is_deterministic_and_sparse() -> None:
    dense = generate_synthetic_waypoints(6, seed=2026, n_reports=120, n_days=10.0)
    a = thin_reporting_gap_waypoints(dense, min_keep_per_traveller=32)
    b = thin_reporting_gap_waypoints(dense, min_keep_per_traveller=32)

    assert a == b
    assert len(a) < len(dense)
    by_traveller: dict[int, list[float]] = {}
    for row in a:
        by_traveller.setdefault(row.traveller_id, []).append(row.days)
    assert set(by_traveller) == set(range(6))
    assert all(len(days) >= 32 for days in by_traveller.values())
    assert all(days == sorted(days) for days in by_traveller.values())


def test_adversarial_missingness_thinning_is_deterministic_and_hides_motion() -> None:
    dense = generate_synthetic_waypoints(6, seed=2026, n_reports=140, n_days=12.0)
    a = thin_adversarial_missingness_waypoints(dense, min_keep_per_traveller=36)
    b = thin_adversarial_missingness_waypoints(dense, min_keep_per_traveller=36)

    assert a == b
    assert len(a) < len(dense)
    assert {row.traveller_id for row in a} == set(range(6))
    by_traveller: dict[int, list[float]] = {}
    for row in a:
        by_traveller.setdefault(row.traveller_id, []).append(row.days)
    assert all(len(days) >= 36 for days in by_traveller.values())


def test_regime_stressor_generator_is_deterministic_with_reporting_gaps() -> None:
    a = generate_regime_stressor_waypoints(5, seed=99, n_reports=80)
    b = generate_regime_stressor_waypoints(5, seed=99, n_reports=80)

    assert a == b
    assert len(a) == 400
    first = a[0]
    assert first.traveller_id == 0
    assert first.days == 0.0573573445738264
    assert first.x_km == 100.98903062593669
    assert first.y_km == 110.67502761265608
    intervals_by_traveller: dict[int, list[float]] = {}
    for previous, current in zip(a, a[1:]):
        if previous.traveller_id == current.traveller_id:
            intervals_by_traveller.setdefault(current.traveller_id, []).append(
                current.days - previous.days
            )
    assert intervals_by_traveller
    assert max(max(v) for v in intervals_by_traveller.values()) > 0.3
    assert min(min(v) for v in intervals_by_traveller.values()) < 0.04


def test_regime_stressor_phase_segment_boundaries_are_canonical() -> None:
    assert phase_segment_for_phase(0.0) == "commute"
    assert phase_segment_for_phase(0.3399) == "commute"
    assert phase_segment_for_phase(0.34) == "explore"
    assert phase_segment_for_phase(0.6799) == "explore"
    assert phase_segment_for_phase(0.68) == "return_stabilization"
    assert phase_segment_for_phase(1.0) == "return_stabilization"


def test_reporting_gap_and_regime_datasets_keep_locked_split_contract() -> None:
    inf = _inference()
    builders = (
        build_reporting_gap_partial_observability_dataset,
        build_adversarial_missingness_partial_observability_dataset,
        build_regime_stressor_partial_observability_dataset,
        build_regime_stressor_repeat_partial_observability_dataset,
    )
    for builder in builders:
        ds = builder(inf, n_travellers=12, eps=0.3)
        assert ds.n_states == 36
        assert ds.observed.shape == ds.latent.shape == ds.travellers.shape
        assert ds.transition.shape == (36, 36)
        assert np.allclose(ds.transition.sum(axis=0), 1.0)
        holdout = set(int(t) for t in ds.travellers[ds.is_holdout])
        train = set(int(t) for t in ds.travellers[~ds.is_holdout])
        assert holdout and train and holdout.isdisjoint(train)
        scored = score_candidate(ds, RecencyAttentionCandidate())
        assert scored.n_scored_steps > 0
        assert np.isfinite(scored.pooled_mean_loss)


def test_lane_definitions_are_current_for_all_canonical_robustness_lanes() -> None:
    definitions = lane_definitions()
    assert set(definitions) == {
        "fractalrabbit",
        "gravity",
        "reporting_gap",
        "regime_stressor",
    }
    assert definitions["reporting_gap"]["status"] == "scored"
    assert "burst" in str(definitions["reporting_gap"]["robustness_role"])
    assert definitions["regime_stressor"]["status"] == "scored"
    assert "regime" in str(definitions["regime_stressor"]["robustness_role"])
    assert DEFAULT_REPORTING_GAP_MIN_KEEP >= 24

    diagnostic = diagnostic_lane_definitions()
    assert tuple(diagnostic) == DIAGNOSTIC_STRESSOR_LANES
    assert set(diagnostic) == {
        "reporting_gap_adversarial",
        "reporting_gap_repeat",
        "high_motion_adversarial_repeat",
        "regime_stressor_repeat",
        "regime_stressor_repeat_2",
    }
    assert all(row["status"] == "scored_diagnostic" for row in diagnostic.values())
    assert set(all_lane_definitions()) == set(definitions) | set(diagnostic)
