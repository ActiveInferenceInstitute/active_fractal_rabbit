from __future__ import annotations

import importlib
from pathlib import Path

from active_fractal_rabbit.types import PipelineResult

full_study = importlib.import_module("scripts.07_run_full_study")


def test_external_lane_unavailable_policy_blocks_final_both_run() -> None:
    assert (
        full_study._external_unavailable_exit_code(
            "both", stop_on_external_unavailable=True
        )
        == 2
    )
    assert (
        full_study._external_unavailable_exit_code(
            "external", stop_on_external_unavailable=False
        )
        == 2
    )


def test_external_lane_unavailable_policy_allows_fixture_development() -> None:
    assert (
        full_study._external_unavailable_exit_code(
            "fixture", stop_on_external_unavailable=True
        )
        is None
    )
    assert (
        full_study._external_unavailable_exit_code(
            "both", stop_on_external_unavailable=False
        )
        is None
    )


def test_scope23_contract_external_path_uses_completed_result_only(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(full_study, "ROOT", tmp_path)
    stale = tmp_path / "output/runs/fractalrabbit_external/observations.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("old")
    current = tmp_path / "owned/current/observations.csv"
    current.parent.mkdir(parents=True)
    current.write_text("current")
    result = PipelineResult(
        "custom", Path("external.yaml"), current.parent, current, {}, None, (), {}, {}
    )
    assert full_study._external_observations_path(result).read_text() == "current"
    unavailable = full_study._external_observations_path(None)
    assert unavailable != stale
    assert not unavailable.exists()


def test_privacy_utility_canonical_path_uses_configured_fixture_not_external_lane() -> (
    None
):
    fixture_path = full_study._fixture_observations_path(
        full_study.ROOT / "configs" / "default.yaml"
    )
    external_path = full_study._external_observations_path(None)

    assert (
        fixture_path
        == full_study.ROOT / "data" / "fixtures" / "fractalrabbit_sample_large.csv"
    )
    assert fixture_path != external_path
