"""Fixture-vs-external tensor-summary comparison.

The comparison is deliberately descriptive: it reports structural agreement
and normalization deltas between two ``tensor_summary.json`` payloads without
claiming that either lane validates the other. The external lane only exists
when the pinned FractalRabbit toolchain (JDK 21 + Maven) has produced a run.
"""

from __future__ import annotations

import json
from pathlib import Path

COMPARED_SCALARS = (
    "A_column_sum_min",
    "A_column_sum_max",
    "B_column_sum_min",
    "B_column_sum_max",
)


def compare_tensor_summaries(
    fixture_summary: dict, external_summary: dict
) -> dict[str, object]:
    """Compare two tensor summaries structurally and numerically."""

    report: dict[str, object] = {
        "schema": "active-fractal-rabbit-tensor-comparison-v1",
        "same_modalities": fixture_summary.get("observation_modalities")
        == external_summary.get("observation_modalities"),
        "same_controls": fixture_summary.get("transition_controls")
        == external_summary.get("transition_controls"),
        "fixture_shapes": {
            "A": fixture_summary.get("A_shapes"),
            "B": fixture_summary.get("B_shape"),
            "D": fixture_summary.get("D_shape"),
        },
        "external_shapes": {
            "A": external_summary.get("A_shapes"),
            "B": external_summary.get("B_shape"),
            "D": external_summary.get("D_shape"),
        },
        "caveat": (
            "Structural agreement between fixture and external tensor summaries shows interface "
            "compatibility only; it is not evidence of external validity."
        ),
    }
    deltas: dict[str, list[float]] = {}
    for key in COMPARED_SCALARS:
        fixture_values = fixture_summary.get(key)
        external_values = external_summary.get(key)
        if (
            isinstance(fixture_values, list)
            and isinstance(external_values, list)
            and len(fixture_values) == len(external_values)
        ):
            deltas[key] = [
                float(b) - float(a) for a, b in zip(fixture_values, external_values)
            ]
    report["normalization_deltas"] = deltas
    report["d_sum_delta"] = float(external_summary.get("D_sum", 0.0)) - float(
        fixture_summary.get("D_sum", 0.0)
    )
    return report


def compare_tensor_summary_files(
    fixture_path: str | Path, external_path: str | Path
) -> dict[str, object]:
    """Load two tensor-summary files and compare them."""

    fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    external = json.loads(Path(external_path).read_text(encoding="utf-8"))
    return compare_tensor_summaries(fixture, external)
