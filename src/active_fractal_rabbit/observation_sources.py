"""Observation-source resolution for generated analysis artifacts."""

from __future__ import annotations

from pathlib import Path

from .fractalrabbit import parse_waypoint_csv
from .observations import discretize_observations
from .types import ObservationSequence, ProjectConfig


def resolve_observations_path(
    config: ProjectConfig,
    observations_path: str | Path | None = None,
    *,
    requested_role: str = "observations",
) -> Path:
    """Resolve the waypoint CSV used by manuscript-facing analyses.

    Explicit paths are fail-loud: if a caller asks for current external
    observations, a missing file is a blocking condition, never a fixture
    fallback. With no explicit path, fixture configs use ``fixture_csv`` and
    external configs use the current lane's ``output_dir/observations.csv``.
    """

    if observations_path is not None:
        source = Path(observations_path)
        if not source.is_file():
            raise FileNotFoundError(
                f"requested {requested_role} file does not exist: {source}. "
                "Run the producing lane first or pass observations_path=None for the configured fixture."
            )
        return source

    if config.fractalrabbit.execution == "fixture":
        source = config.fractalrabbit.fixture_csv
        if source is None:
            raise FileNotFoundError(
                "fixture execution requires fractalrabbit.fixture_csv"
            )
        if not source.is_file():
            raise FileNotFoundError(
                f"configured fixture observations file does not exist: {source}"
            )
        return source

    source = config.output_dir / "observations.csv"
    if not source.is_file():
        raise FileNotFoundError(
            f"current external-lane observations file does not exist: {source}. "
            "Run the external pipeline before requesting external evidence."
        )
    return source


def load_observation_sequence(
    config: ProjectConfig,
    observations_path: str | Path | None = None,
    *,
    requested_role: str = "observations",
) -> ObservationSequence:
    """Resolve, parse, and discretize one observation source."""

    source = resolve_observations_path(
        config, observations_path, requested_role=requested_role
    )
    return discretize_observations(parse_waypoint_csv(source), config.inference)
