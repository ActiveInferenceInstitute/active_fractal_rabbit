"""Deterministic synthetic waypoint fixture generation.

The checked fixture and larger confirmation lane both use this helper so the
CSV contract and generator family stay identical while sample size varies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import WaypointObservation

DEFAULT_FIXTURE_SEED = 2026
DEFAULT_FIXTURE_ANCHORS: tuple[tuple[float, float], ...] = (
    (10.0, 10.0),
    (120.0, 30.0),
    (40.0, 150.0),
    (170.0, 160.0),
    (105.0, 105.0),
)


def generate_waypoint_observations(
    *,
    seed: int = DEFAULT_FIXTURE_SEED,
    n_travellers: int = 5,
    n_reports_per_traveller: int = 480,
    n_days: float = 36.0,
    anchors: tuple[tuple[float, float], ...] = DEFAULT_FIXTURE_ANCHORS,
) -> tuple[WaypointObservation, ...]:
    """Generate deterministic sporadic waypoint reports in FractalRabbit CSV shape."""

    if n_travellers <= 0:
        raise ValueError("n_travellers must be positive")
    if n_reports_per_traveller <= 0:
        raise ValueError("n_reports_per_traveller must be positive")
    if n_reports_per_traveller % 4 != 0:
        raise ValueError("n_reports_per_traveller must be divisible by 4")
    if n_days <= 0:
        raise ValueError("n_days must be positive")

    rng = np.random.default_rng(seed)
    traveller_anchors = _anchors_for_count(n_travellers, anchors)
    rows: list[WaypointObservation] = []
    for traveller, anchor in enumerate(traveller_anchors):
        times: list[float] = []
        for burst in range(n_reports_per_traveller // 4):
            start = (burst * n_days / (n_reports_per_traveller // 4)) + rng.uniform(
                0.02, 0.08
            )
            times.extend(start + np.cumsum(rng.uniform(0.003, 0.035, size=4)))
        times = sorted(times)[:n_reports_per_traveller]
        x, y = anchor
        home = (x, y)
        for idx, day in enumerate(times):
            rows.append(
                WaypointObservation(
                    traveller_id=traveller,
                    days=float(day),
                    x_km=float(x),
                    y_km=float(y),
                )
            )
            if idx == len(times) - 1:
                break
            dt = times[idx + 1] - day
            regime = rng.choice(["dwell", "local", "fast"], p=[0.4, 0.4, 0.2])
            if regime == "dwell":
                speed = rng.uniform(0.5, 8.0)
            elif regime == "local":
                speed = rng.uniform(15.0, 120.0)
            else:
                speed = rng.uniform(160.0, 400.0)
            distance = speed * dt
            if rng.uniform() < 0.5:
                direction = np.arctan2(home[1] - y, home[0] - x) + rng.uniform(
                    -0.5, 0.5
                )
            else:
                direction = rng.uniform(0, 2 * np.pi)
            x = float(np.clip(x + distance * np.cos(direction), 0.0, 200.0))
            y = float(np.clip(y + distance * np.sin(direction), 0.0, 200.0))
    return tuple(rows)


def write_waypoint_csv(
    observations: tuple[WaypointObservation, ...], destination: str | Path
) -> Path:
    """Write observations to the `ID,Days,x(km),y(km)` waypoint CSV contract."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ID,Days,x(km),y(km)"]
    lines.extend(
        f"{row.traveller_id},{row.days:.4f},{row.x_km:.4f},{row.y_km:.4f}"
        for row in observations
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _anchors_for_count(
    n_travellers: int, anchors: tuple[tuple[float, float], ...]
) -> tuple[tuple[float, float], ...]:
    if n_travellers <= len(anchors):
        return anchors[:n_travellers]

    generated = list(anchors)
    side = int(np.ceil(np.sqrt(n_travellers)))
    xs = np.linspace(20.0, 180.0, side)
    ys = np.linspace(20.0, 180.0, side)
    for y in ys:
        for x in xs:
            if len(generated) >= n_travellers:
                return tuple(generated)
            candidate = (float(x), float(y))
            if candidate not in generated:
                generated.append(candidate)
    return tuple(generated[:n_travellers])
