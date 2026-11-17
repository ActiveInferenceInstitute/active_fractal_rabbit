"""Observation-source resolution for generated analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from .fractalrabbit import file_sha256, parse_waypoint_csv
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
    manifest_path = config.output_dir / "provenance_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Current external-lane completion receipt missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    build = manifest.get("observation_bundle", {}).get("build") or {}
    if (
        manifest.get("execution") != "external"
        or manifest.get("run_id") != config.run_id
        or manifest.get("configured_upstream_commit")
        != config.fractalrabbit.upstream_commit
        or build.get("requested_revision") != config.fractalrabbit.upstream_commit
        or build.get("upstream_repo") != config.fractalrabbit.upstream_repo
        or not build.get("commit")
        or build["commit"] != manifest.get("upstream_commit")
    ):
        raise ValueError(
            f"External completion receipt does not match configured run: {manifest_path}"
        )
    for key, path in (
        ("observations", source),
        ("metrics", config.output_dir / "metrics.json"),
    ):
        if not path.is_file() or manifest.get("artifact_sha256", {}).get(
            key
        ) != file_sha256(path):
            raise ValueError(
                f"External completion receipt does not bind current {key}: {path}"
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
