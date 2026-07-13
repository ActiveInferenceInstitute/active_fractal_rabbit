"""Regenerate the checked large fixture (deterministic, seed 2026).

The large fixture is the canonical analysis lane: 480 timepoints per
traveller over 36 days across five synthetic travellers. It preserves the FractalRabbit output contract and the
qualitative structure the figures exercise: bursty inter-report intervals,
speed segments spanning all four speed categories, and revisited regions.
"""

from __future__ import annotations

from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - standalone script execution
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.fixture_generation import (
    generate_waypoint_observations,
    write_waypoint_csv,
)

ROOT = Path(__file__).resolve().parents[1]

SEED = 2026
N_TRAVELLERS = 5
N_REPORTS = 480
N_DAYS = 36.0


def main() -> int:
    observations = generate_waypoint_observations(
        seed=SEED,
        n_travellers=N_TRAVELLERS,
        n_reports_per_traveller=N_REPORTS,
        n_days=N_DAYS,
    )
    output = ROOT / "data" / "fixtures" / "fractalrabbit_sample_large.csv"
    write_waypoint_csv(observations, output)
    print(f"wrote {output} ({len(observations)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
