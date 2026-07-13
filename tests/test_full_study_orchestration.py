from __future__ import annotations

import importlib


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


def test_fixture_only_external_observation_path_cannot_reuse_stale_external_artifact() -> (
    None
):
    stale_path = (
        full_study.ROOT
        / "output"
        / "runs"
        / "fractalrabbit_external"
        / "observations.csv"
    )

    assert full_study._external_observations_path(True) == stale_path
    assert full_study._external_observations_path(False) != stale_path
    assert "external_not_completed_this_run" in str(
        full_study._external_observations_path(False)
    )


def test_privacy_utility_canonical_path_uses_configured_fixture_not_external_lane() -> (
    None
):
    fixture_path = full_study._fixture_observations_path(
        full_study.ROOT / "configs" / "default.yaml"
    )
    external_path = full_study._external_observations_path(True)

    assert (
        fixture_path
        == full_study.ROOT / "data" / "fixtures" / "fractalrabbit_sample_large.csv"
    )
    assert fixture_path != external_path
