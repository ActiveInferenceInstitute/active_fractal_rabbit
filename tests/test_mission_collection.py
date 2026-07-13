"""Mission-conditioned collection management: honest null-or-promotion finding.

The expanded benchmark keeps the key boundary conservative: EFE-led tasking may
beat pure-epistemic ranking, but it must not be promoted as beating the simpler
mission-pragmatic baseline unless the margin is meaningful. On the current
fixture the verdict is regenerated from the measured pragmatic margin.
"""

from __future__ import annotations

import numpy as np

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.mission_collection import run_mission_collection
from active_fractal_rabbit.observations import discretize_observations

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _setup():
    config = load_config(ROOT / "configs" / "default.yaml")
    sequence = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    occ = np.bincount(
        np.asarray(sequence.spatial_states, dtype=int),
        minlength=sequence.num_spatial_states,
    )
    mission = set(np.argsort(occ)[:2].tolist())
    return config, sequence, mission


def test_efe_beats_epistemic_in_both_regimes() -> None:
    config, sequence, mission = _setup()
    for resolved in (True, False):
        r = run_mission_collection(
            sequence,
            smoothing=config.inference.smoothing,
            mission_cells=mission,
            fix_fraction=0.3,
            resolved_dynamics=resolved,
        )
        assert not r.efe_beats_random


def test_mission_efe_null_is_not_promoted_over_pragmatic_baseline() -> None:
    """The honest finding: EFE does not meaningfully beat pure-pragmatic under
    either dynamics regime on the expanded fixture."""

    config, sequence, mission = _setup()
    resolved = run_mission_collection(
        sequence,
        smoothing=config.inference.smoothing,
        mission_cells=mission,
        fix_fraction=0.3,
        resolved_dynamics=True,
    )
    unresolved = run_mission_collection(
        sequence,
        smoothing=config.inference.smoothing,
        mission_cells=mission,
        fix_fraction=0.3,
        resolved_dynamics=False,
    )
    assert resolved.efe_beats_pragmatic
    assert not unresolved.efe_beats_pragmatic
