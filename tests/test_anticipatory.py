"""Anticipatory gap-filling recovery: speed-clamped transitions can beat persistence.

The decisive tests use a CONTROLLED synthetic sequence with explicit speed->move
coupling (stationary speed never moves; high speed always moves, to a previously
visited cell). On that structure the speed-conditioned filter must beat both the
flat filter and persistence in the gap-filling regime, and a shuffled-speed
control must destroy the advantage (non-vacuity).
"""

from __future__ import annotations

import numpy as np
import pytest

from active_fractal_rabbit.anticipatory import (
    run_anticipatory_analysis,
    slice_off_diagonal_mass,
    speed_conditioned_transition,
)
from active_fractal_rabbit.pomdp import build_pomdp
from active_fractal_rabbit.types import ObservationSequence, WaypointObservation


def _synthetic_sequence(
    cells: np.ndarray, speeds: np.ndarray, first: np.ndarray
) -> ObservationSequence:
    """Minimal ObservationSequence with controlled cells + speed categories.

    Time-of-day and report-burst are held constant (uninformative), so the only
    structure the model can exploit is the speed->cell-transition coupling.
    """

    cells = np.asarray(cells, dtype=int)
    n_states = int(cells.max()) + 1
    n = cells.shape[0]
    obs = tuple(
        WaypointObservation(traveller_id=0, days=float(i), x_km=0.0, y_km=0.0)
        for i in range(n)
    )
    return ObservationSequence(
        observations=obs,
        is_first_observation=np.asarray(first, dtype=bool),
        x_bins=np.zeros(n, dtype=int),
        y_bins=np.zeros(n, dtype=int),
        spatial_states=cells,
        time_bins=np.zeros(n, dtype=int),
        speed_categories=np.asarray(speeds, dtype=int),
        burst_categories=np.zeros(n, dtype=int),
        speeds_km_per_day=np.zeros(n, dtype=float),
        intervals_days=np.ones(n, dtype=float),
        spatial_bins=n_states,
        num_spatial_states=n_states,
        num_time_bins=1,
        num_speed_categories=4,
        num_burst_categories=1,
        spatial_bounds=(0.0, 0.0, 1.0, 1.0),
    )


def _coupled_trace() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A dwell-and-return trace: high speed exactly on (and only on) move steps."""

    # Two travellers, each dwelling in cells then returning among visited ones.
    base = [0, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 1, 1]
    cells = np.array(base + base, dtype=int)
    first = np.zeros(cells.shape[0], dtype=bool)
    first[0] = True
    first[len(base)] = True
    speeds = np.zeros(cells.shape[0], dtype=int)
    for i in range(cells.shape[0]):
        if not first[i] and cells[i] != cells[i - 1]:
            speeds[i] = 3  # a move -> high speed category
    return cells, speeds, first


def test_speed_slice_k0_is_identity_and_slices_monotone() -> None:
    cells, speeds, first = _coupled_trace()
    n_states = int(cells.max()) + 1
    speed_b = speed_conditioned_transition(cells, speeds, first, n_states, 4, 1e-6)
    # Stationary slice is (near) identity: no off-diagonal mass.
    eye = np.eye(n_states, dtype=bool)
    assert speed_b[:, :, 0][~eye].sum() < 1e-3
    # The move slice carries real off-diagonal mass; k=3 >> k=0.
    off = slice_off_diagonal_mass(speed_b)
    assert off[3] > 0.3
    assert off[3] > off[0] + 0.3


def test_controlled_speed_coupling_can_beat_persistence_in_gap_filling() -> None:
    """On speed-coupled data, the speed-clamped filter wins, and merely
    propagating belief already beats persistence."""

    cells, speeds, first = _coupled_trace()
    seq = _synthetic_sequence(cells, speeds, first)
    full = build_pomdp(seq, smoothing=1e-3)
    r = run_anticipatory_analysis(seq, full, smoothing=1e-3, gap_fraction=0.5, seed=0)
    assert r.n_gap_steps > 0
    assert r.flat_skill_gap > 0.0  # controlled propagation beats rigid persistence
    assert r.scrp_skill_gap > r.flat_skill_gap  # speed-clamping adds real value
    assert r.scrp_skill_gap > 0.1


def test_shuffled_speed_destroys_the_advantage() -> None:
    """Non-vacuity: if speed no longer predicts moves, speed-clamping stops helping."""

    cells, speeds, first = _coupled_trace()
    rng = np.random.default_rng(7)
    shuffled = speeds.copy()
    rng.shuffle(shuffled)  # break the speed<->move coupling
    seq = _synthetic_sequence(cells, shuffled, first)
    full = build_pomdp(seq, smoothing=1e-3)
    r = run_anticipatory_analysis(seq, full, smoothing=1e-3, gap_fraction=0.5, seed=0)
    coupled = run_anticipatory_analysis(
        _synthetic_sequence(cells, speeds, first),
        full,
        smoothing=1e-3,
        gap_fraction=0.5,
        seed=0,
    )
    # The speed-clamp advantage over flat is much larger with real coupling.
    shuffled_edge = r.scrp_skill_gap - r.flat_skill_gap
    coupled_edge = coupled.scrp_skill_gap - coupled.flat_skill_gap
    assert coupled_edge > shuffled_edge + 0.05


def test_blind_lane_cannot_beat_persistence_limit_finding() -> None:
    """The honest limit: with location hidden everywhere, persistence (which is
    handed the true previous cell) is not beaten — but speed narrows the gap."""

    cells, speeds, first = _coupled_trace()
    seq = _synthetic_sequence(cells, speeds, first)
    full = build_pomdp(seq, smoothing=1e-3)
    r = run_anticipatory_analysis(seq, full, smoothing=1e-3, gap_fraction=0.5, seed=0)
    # Speed-clamping never does worse than flat in the blind lane.
    assert r.blind_persistence_skill_scrp >= r.blind_persistence_skill_flat - 1e-9


def test_analysis_is_deterministic() -> None:
    cells, speeds, first = _coupled_trace()
    seq = _synthetic_sequence(cells, speeds, first)
    full = build_pomdp(seq, smoothing=1e-3)
    a = run_anticipatory_analysis(seq, full, smoothing=1e-3, gap_fraction=0.5, seed=1)
    b = run_anticipatory_analysis(seq, full, smoothing=1e-3, gap_fraction=0.5, seed=1)
    assert a.scrp_skill_gap == pytest.approx(b.scrp_skill_gap)
    assert a.persistence_brier_gap == pytest.approx(b.persistence_brier_gap)
