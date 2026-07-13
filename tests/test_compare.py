from __future__ import annotations

import json

from active_fractal_rabbit.compare import (
    compare_tensor_summaries,
    compare_tensor_summary_files,
)


def _summary(a_min: float = 1.0) -> dict:
    return {
        "observation_modalities": ["location", "time_of_day", "speed", "report_burst"],
        "transition_controls": ["empirical_continuation", "high_entropy_exploration"],
        "A_shapes": [[9, 9], [4, 9], [4, 9], [2, 9]],
        "B_shape": [9, 9, 2],
        "D_shape": [9],
        "A_column_sum_min": [a_min, 1.0, 1.0, 1.0],
        "A_column_sum_max": [1.0, 1.0, 1.0, 1.0],
        "B_column_sum_min": [1.0, 1.0],
        "B_column_sum_max": [1.0, 1.0],
        "D_sum": 1.0,
    }


def test_compare_identical_summaries() -> None:
    report = compare_tensor_summaries(_summary(), _summary())

    assert report["same_modalities"] is True
    assert report["same_controls"] is True
    assert report["normalization_deltas"]["A_column_sum_min"] == [0.0, 0.0, 0.0, 0.0]
    assert report["d_sum_delta"] == 0.0
    assert "not evidence of external validity" in str(report["caveat"])


def test_compare_reports_numeric_deltas_and_mismatched_lengths() -> None:
    external = _summary(a_min=0.5)
    report = compare_tensor_summaries(_summary(), external)
    assert report["normalization_deltas"]["A_column_sum_min"][0] == -0.5

    external_short = _summary()
    external_short["A_column_sum_min"] = [1.0]
    report = compare_tensor_summaries(_summary(), external_short)
    assert "A_column_sum_min" not in report["normalization_deltas"]


def test_compare_tensor_summary_files(tmp_path) -> None:
    fixture = tmp_path / "fixture.json"
    external = tmp_path / "external.json"
    fixture.write_text(json.dumps(_summary()), encoding="utf-8")
    external.write_text(json.dumps(_summary(a_min=0.75)), encoding="utf-8")

    report = compare_tensor_summary_files(fixture, external)
    assert report["same_modalities"] is True
    assert report["normalization_deltas"]["A_column_sum_min"][0] == -0.25
