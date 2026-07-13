"""Waypoint validation and discretization."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .types import InferenceConfig, ObservationSequence, WaypointObservation


def discretize_observations(
    observations: tuple[WaypointObservation, ...],
    config: InferenceConfig,
    estimation_mask: np.ndarray | None = None,
) -> ObservationSequence:
    """Convert continuous sporadic waypoints into categorical observation factors.

    ``estimation_mask`` (aligned to the sorted row order) restricts every
    data-driven discretization statistic — quantile bin edges, auto spatial
    bounds, and the median burst cutoff — to the masked (training) rows.
    All rows are still transformed, but holdout rows can no longer shape the
    boundaries that score them.
    """

    observation_config = config.observation
    if not observations:
        raise ValueError("at least one waypoint observation is required")
    ordered = tuple(
        sorted(observations, key=lambda item: (item.traveller_id, item.days))
    )
    _validate_ordered(ordered)
    if estimation_mask is None:
        est = np.ones(len(ordered), dtype=bool)
    else:
        est = np.asarray(estimation_mask, dtype=bool)
        if est.shape[0] != len(ordered):
            raise ValueError("estimation mask length must match the observation count")
        if int(est.sum()) < 2:
            raise ValueError("estimation mask must keep at least two rows")

    x = np.array([row.x_km for row in ordered], dtype=float)
    y = np.array([row.y_km for row in ordered], dtype=float)
    x_bins, y_bins, bounds = _spatial_bins(x, y, observation_config, est)
    spatial_states = (y_bins * observation_config.spatial_bins + x_bins).astype(int)
    time_bins = np.array(
        [
            min(
                observation_config.time_bins_per_day - 1,
                int((row.days % 1.0) * observation_config.time_bins_per_day),
            )
            for row in ordered
        ],
        dtype=int,
    )
    speeds, intervals, is_first_observation = _speed_intervals_and_first_observation(
        ordered
    )
    speed_categories = np.digitize(
        speeds, np.array(observation_config.speed_thresholds_km_per_day), right=False
    ).astype(int)
    burst_cutoff = _burst_cutoff(
        intervals[est],
        observation_config.burst_threshold_rule,
        observation_config.burst_fixed_days,
    )
    burst_categories = np.where(
        (intervals > 0) & (intervals <= burst_cutoff), 1, 0
    ).astype(int)

    return ObservationSequence(
        observations=ordered,
        is_first_observation=is_first_observation,
        x_bins=x_bins,
        y_bins=y_bins,
        spatial_states=spatial_states,
        time_bins=time_bins,
        speed_categories=speed_categories,
        burst_categories=burst_categories,
        speeds_km_per_day=speeds,
        intervals_days=intervals,
        spatial_bins=observation_config.spatial_bins,
        num_spatial_states=observation_config.spatial_bins
        * observation_config.spatial_bins,
        num_time_bins=observation_config.time_bins_per_day,
        num_speed_categories=4,
        num_burst_categories=2,
        spatial_bounds=bounds,
    )


def _validate_ordered(observations: tuple[WaypointObservation, ...]) -> None:
    previous_by_traveller: dict[int, float] = {}
    for row in observations:
        if row.days < 0:
            raise ValueError("observation days cannot be negative")
        previous = previous_by_traveller.get(row.traveller_id)
        if previous is not None and row.days < previous:
            raise ValueError(
                f"traveller {row.traveller_id} observations are not time sorted"
            )
        previous_by_traveller[row.traveller_id] = row.days


def _spatial_bins(
    x: np.ndarray,
    y: np.ndarray,
    config,
    estimation_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    est = np.ones(x.shape, dtype=bool) if estimation_mask is None else estimation_mask
    if config.spatial_discretizer == "quantile":
        return (
            _one_axis_quantile_bins(x, config.spatial_bins, est),
            _one_axis_quantile_bins(y, config.spatial_bins, est),
            _bounds(x[est], y[est]),
        )
    if config.spatial_bounds is not None:
        xmin, xmax, ymin, ymax = config.spatial_bounds
        return (
            _one_axis_bins(x, config.spatial_bins, xmin, xmax),
            _one_axis_bins(y, config.spatial_bins, ymin, ymax),
            config.spatial_bounds,
        )
    bounds = _bounds(x[est], y[est])
    xmin, xmax, ymin, ymax = bounds
    return (
        _one_axis_bins(x, config.spatial_bins, xmin, xmax),
        _one_axis_bins(y, config.spatial_bins, ymin, ymax),
        bounds,
    )


def _one_axis_bins(values: np.ndarray, bins: int, lo: float, hi: float) -> np.ndarray:
    if np.isclose(lo, hi):
        return np.zeros(values.shape, dtype=int)
    clipped = np.clip(values, lo, hi)
    scaled = (clipped - lo) / (hi - lo)
    return np.minimum(bins - 1, np.floor(scaled * bins)).astype(int)


def _one_axis_quantile_bins(
    values: np.ndarray, bins: int, estimation_mask: np.ndarray | None = None
) -> np.ndarray:
    estimation_values = values if estimation_mask is None else values[estimation_mask]
    if np.allclose(estimation_values, estimation_values[0]):
        return np.zeros(values.shape, dtype=int)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(estimation_values, quantiles)
    edges = np.maximum.accumulate(edges)
    interior = edges[1:-1]
    return np.searchsorted(interior, values, side="right").astype(int)


def _bounds(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    return (float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y)))


def _burst_cutoff(intervals: np.ndarray, rule: str, fixed_days: float | None) -> float:
    if rule == "fixed_days":
        if fixed_days is None:
            raise ValueError("fixed_days burst threshold requires burst_fixed_days")
        return fixed_days
    return float(np.median(intervals[intervals > 0])) if np.any(intervals > 0) else 1.0


def _speed_intervals_and_first_observation(
    observations: tuple[WaypointObservation, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_traveller: dict[int, list[tuple[int, WaypointObservation]]] = defaultdict(list)
    for idx, row in enumerate(observations):
        by_traveller[row.traveller_id].append((idx, row))

    speeds = np.zeros(len(observations), dtype=float)
    intervals = np.zeros(len(observations), dtype=float)
    is_first_observation = np.ones(len(observations), dtype=bool)
    for rows in by_traveller.values():
        for (_, previous), (idx, current) in zip(rows, rows[1:]):
            is_first_observation[idx] = False
            dt = max(current.days - previous.days, 0.0)
            intervals[idx] = dt
            if dt > 0:
                distance = float(
                    np.hypot(current.x_km - previous.x_km, current.y_km - previous.y_km)
                )
                speeds[idx] = distance / dt
    return speeds, intervals, is_first_observation
