"""Negative-control falsification lane for the fixture pipeline.

Each control intentionally destroys one structural property of the waypoint
data or the observation configuration, then checks a directional expectation
on the resulting artifacts. A pipeline that passed these checks vacuously
(i.e. produced the same outputs under destroyed structure) would be
green-washed; this lane exists to falsify that failure mode.

The lane is fixture-only by design: controls mutate a checked CSV or the
observation config, never the external simulator.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import load_config
from .pipeline import run_pipeline
from .types import PipelineResult, ProjectConfig, WaypointObservation
from .fractalrabbit import parse_waypoint_csv

NEGATIVE_CONTROL_SCHEMA = "active-fractal-rabbit-negative-controls-v1"
DEGENERATE_BOUNDS = (-1_000_000.0, 1_000_000.0, -1_000_000.0, 1_000_000.0)
MISSPECIFIED_SPEED_THRESHOLDS = (100_000.0, 200_000.0, 300_000.0)

CONTROL_HYPOTHESES = {
    "shuffled_time": "Permuting report times within each traveller must change interval-derived speed features.",
    "shuffled_ids": "Permuting traveller identities across rows must change traveller-conditional speed features.",
    "degenerate_bounds": "Spatial bounds far wider than the data must collapse all rows into one occupied spatial cell.",
    "misspecified_speed": "Speed thresholds far above all derived speeds must collapse all rows into one occupied speed category.",
}


def run_negative_controls(config_path: str | Path) -> Path:
    """Run the four negative controls and write the falsification summary."""

    base = load_config(config_path)
    if base.fractalrabbit.execution != "fixture":
        raise ValueError(
            "negative controls are a fixture-lane diagnostic; use a fixture config"
        )
    assert base.fractalrabbit.fixture_csv is not None
    rng = np.random.default_rng(base.seed)
    mutated_dir = base.project_root / "output" / "negative_controls"
    mutated_dir.mkdir(parents=True, exist_ok=True)

    intact = run_pipeline(_control_config(base, "negative_intact"), config_path)
    intact_speeds = _feature_column(intact, "speed_km_per_day")

    controls: list[dict[str, object]] = []
    rows = parse_waypoint_csv(base.fractalrabbit.fixture_csv)

    shuffled_time_csv = _write_waypoints(
        mutated_dir / "shuffled_time.csv", _shuffle_time(rows, rng)
    )
    result = run_pipeline(
        _control_config(base, "negative_shuffled_time", fixture_csv=shuffled_time_csv),
        config_path,
    )
    controls.append(
        _control_record(
            "shuffled_time",
            result,
            intact,
            passed=_feature_column(result, "speed_km_per_day") != intact_speeds,
            probe="speed_km_per_day column differs from the intact run",
        )
    )

    shuffled_ids_csv = _write_waypoints(
        mutated_dir / "shuffled_ids.csv", _shuffle_ids(rows, rng)
    )
    result = run_pipeline(
        _control_config(base, "negative_shuffled_ids", fixture_csv=shuffled_ids_csv),
        config_path,
    )
    controls.append(
        _control_record(
            "shuffled_ids",
            result,
            intact,
            passed=_feature_column(result, "speed_km_per_day") != intact_speeds,
            probe="speed_km_per_day column differs from the intact run",
        )
    )

    degenerate_observation = replace(
        base.inference.observation,
        spatial_discretizer="fixed_bounds",
        spatial_bounds=DEGENERATE_BOUNDS,
    )
    result = run_pipeline(
        _control_config(
            base, "negative_degenerate_bounds", observation=degenerate_observation
        ),
        config_path,
    )
    occupied_spatial = _occupied(result, "spatial_state")
    controls.append(
        _control_record(
            "degenerate_bounds",
            result,
            intact,
            passed=occupied_spatial == 1,
            probe=f"occupied spatial states == 1 (observed {occupied_spatial})",
        )
    )

    misspecified_observation = replace(
        base.inference.observation,
        speed_thresholds_km_per_day=MISSPECIFIED_SPEED_THRESHOLDS,
    )
    result = run_pipeline(
        _control_config(
            base, "negative_misspecified_speed", observation=misspecified_observation
        ),
        config_path,
    )
    occupied_speed = _occupied(result, "speed_category")
    controls.append(
        _control_record(
            "misspecified_speed",
            result,
            intact,
            passed=occupied_speed == 1,
            probe=f"occupied speed categories == 1 (observed {occupied_speed})",
        )
    )

    payload = {
        "schema": NEGATIVE_CONTROL_SCHEMA,
        "lane": "fixture",
        "seed": base.seed,
        "source_config": str(Path(config_path)),
        "intact_run_id": intact.run_id,
        "intact_metrics": {
            "mean_predictive_loss": float(intact.metrics["mean_predictive_loss"]),
            "n_observations": float(intact.metrics["n_observations"]),
        },
        "controls": controls,
        "n_controls": len(controls),
        "n_passed": sum(1 for item in controls if bool(item["passed"])),
        "all_passed": all(bool(item["passed"]) for item in controls),
        "caveat": (
            "Negative controls falsify pipeline insensitivity to destroyed structure. "
            "They are software diagnostics on the fixture lane, not empirical mobility evidence."
        ),
    }
    summary_path = (
        base.project_root / "output" / "runs" / "negative_controls_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary_path


def _control_config(
    base: ProjectConfig,
    run_id: str,
    *,
    fixture_csv: Path | None = None,
    observation=None,
) -> ProjectConfig:
    fractalrabbit = replace(base.fractalrabbit, output_prefix=run_id)
    if fixture_csv is not None:
        fractalrabbit = replace(fractalrabbit, fixture_csv=fixture_csv)
    inference = base.inference
    if observation is not None:
        inference = replace(inference, observation=observation)
    inference = replace(inference, model=replace(inference.model, engine="reference"))
    return replace(
        base,
        run_id=run_id,
        output_dir=base.project_root / "output" / "runs" / run_id,
        fractalrabbit=fractalrabbit,
        inference=inference,
    )


def _control_record(
    name: str,
    result: PipelineResult,
    intact: PipelineResult,
    *,
    passed: bool,
    probe: str,
) -> dict[str, object]:
    return {
        "name": name,
        "hypothesis": CONTROL_HYPOTHESES[name],
        "probe": probe,
        "run_id": result.run_id,
        "passed": bool(passed),
        "mean_predictive_loss": float(result.metrics["mean_predictive_loss"]),
        "delta_mean_predictive_loss": float(result.metrics["mean_predictive_loss"])
        - float(intact.metrics["mean_predictive_loss"]),
        "occupied_spatial_states": _occupied(result, "spatial_state"),
        "occupied_speed_categories": _occupied(result, "speed_category"),
    }


def _shuffle_time(
    rows: tuple[WaypointObservation, ...], rng: np.random.Generator
) -> list[WaypointObservation]:
    by_traveller: dict[int, list[WaypointObservation]] = {}
    for row in rows:
        by_traveller.setdefault(row.traveller_id, []).append(row)
    mutated: list[WaypointObservation] = []
    for traveller_rows in by_traveller.values():
        days = [row.days for row in traveller_rows]
        if len(set(days)) < 2:
            mutated.extend(traveller_rows)
            continue
        permuted = _non_identity_permutation(days, rng)
        mutated.extend(
            replace(row, days=new_days)
            for row, new_days in zip(traveller_rows, permuted)
        )
    return mutated


def _shuffle_ids(
    rows: tuple[WaypointObservation, ...], rng: np.random.Generator
) -> list[WaypointObservation]:
    ids = [row.traveller_id for row in rows]
    permuted = _non_identity_permutation(ids, rng)
    return [replace(row, traveller_id=new_id) for row, new_id in zip(rows, permuted)]


def _non_identity_permutation(values: list, rng: np.random.Generator) -> list:
    if len(set(values)) < 2:
        raise ValueError(
            "cannot build a non-identity permutation over identical values"
        )
    while True:
        order = rng.permutation(len(values))
        permuted = [values[idx] for idx in order]
        if permuted != list(values):
            return permuted


def _write_waypoints(path: Path, rows: list[WaypointObservation]) -> Path:
    lines = ["ID,Days,x(km),y(km)"]
    for row in sorted(rows, key=lambda item: (item.traveller_id, item.days)):
        lines.append(
            f"{row.traveller_id},{row.days:.12g},{row.x_km:.12g},{row.y_km:.12g}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _feature_column(result: PipelineResult, column: str) -> list[str]:
    import csv

    with (result.output_dir / "discrete_features.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return [row[column] for row in csv.DictReader(handle)]


def _occupied(result: PipelineResult, column: str) -> int:
    return len(set(_feature_column(result, column)))
