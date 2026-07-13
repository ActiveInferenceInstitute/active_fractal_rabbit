from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import (
    parse_places_csv,
    parse_waypoint_csv,
    places_csv_for,
)
from active_fractal_rabbit.ground_truth import (
    align_places,
    place_statistics,
    state_recovery,
)
from active_fractal_rabbit.inference import run_latent_inference
from active_fractal_rabbit.observations import discretize_observations
from active_fractal_rabbit.pomdp import build_pomdp
from active_fractal_rabbit.types import PlaceObservation, WaypointObservation

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "fractalrabbit_sample.csv"


def test_parse_places_csv_roundtrip(tmp_path) -> None:
    path = tmp_path / "runPLACES.csv"
    path.write_text(
        "Traveler ID,Days,Place ID\n1,0.5,42\n0,0.25,7\n0,0.75,7\n",
        encoding="utf-8",
    )
    rows = parse_places_csv(path)

    assert rows[0] == PlaceObservation(traveller_id=0, days=0.25, place_id=7)
    assert [row.place_id for row in rows] == [7, 7, 42]

    bad = tmp_path / "badPLACES.csv"
    bad.write_text("ID,Days,Place\n0,0.1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain header"):
        parse_places_csv(bad)
    empty = tmp_path / "emptyPLACES.csv"
    empty.write_text("Traveler ID,Days,Place ID\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no place rows"):
        parse_places_csv(empty)


def test_places_csv_for_sibling_detection(tmp_path) -> None:
    xy = tmp_path / "myrunXY.csv"
    xy.write_text("x", encoding="utf-8")
    assert places_csv_for(xy) is None  # sibling absent
    places = tmp_path / "myrunPLACES.csv"
    places.write_text("x", encoding="utf-8")
    assert places_csv_for(xy) == places
    assert places_csv_for(tmp_path / "fixture.csv") is None  # not an XY output


def test_align_places_strictness() -> None:
    obs = (
        WaypointObservation(0, 0.25, 1.0, 1.0),
        WaypointObservation(0, 0.75, 2.0, 2.0),
    )
    places = (
        PlaceObservation(0, 0.25, 7),
        PlaceObservation(0, 0.75, 9),
    )
    assert align_places(obs, places) == [7, 9]

    with pytest.raises(ValueError, match="row counts differ"):
        align_places(obs, places[:1])
    shifted = (PlaceObservation(0, 0.30, 7), PlaceObservation(0, 0.75, 9))
    with pytest.raises(ValueError, match="misalignment"):
        align_places(obs, shifted)


def test_place_statistics_hand_computed() -> None:
    stats = place_statistics([5, 5, 9, 3, 3, 3])

    assert stats["place_n_reports"] == 6.0
    assert stats["place_n_unique"] == 3.0
    assert stats["place_n_revisited"] == 2.0
    assert stats["place_revisit_rate"] == pytest.approx(2 / 3)
    assert stats["place_mean_reports_per_place"] == 2.0


def test_state_recovery_hand_computed() -> None:
    config = load_config("configs/default.yaml")
    sequence = discretize_observations(parse_waypoint_csv(FIXTURE), config.inference)
    n = len(sequence.observations)
    n_states = sequence.num_spatial_states
    # synthetic posterior: 0.6 on the true cell everywhere, argmax on true cell
    posterior = np.full((n, n_states), 0.4 / (n_states - 1))
    posterior[np.arange(n), sequence.spatial_states] = 0.6
    metrics, mass, top1 = state_recovery(sequence, posterior, "probe")

    assert metrics["probe_mean_mass_on_true_cell"] == pytest.approx(0.6)
    assert metrics["probe_top1_accuracy"] == 1.0
    assert mass.shape == (n,)
    assert top1.min() == 1.0

    with pytest.raises(ValueError, match="length"):
        state_recovery(sequence, posterior[:-1], "probe")


def test_deep_latent_recovery_strictly_below_location_observing() -> None:
    """Directional probe: removing location evidence must reduce true-cell mass."""

    config = load_config("configs/default.yaml")
    sequence = discretize_observations(parse_waypoint_csv(FIXTURE), config.inference)
    full = run_latent_inference(build_pomdp(sequence, 0.05), sequence)
    latent = run_latent_inference(
        build_pomdp(
            sequence, 0.05, modalities=("time_of_day", "speed", "report_burst")
        ),
        sequence,
    )
    full_metrics, _, _ = state_recovery(sequence, full.posterior_states, "full")
    latent_metrics, _, _ = state_recovery(sequence, latent.posterior_states, "latent")

    assert (
        latent_metrics["latent_mean_mass_on_true_cell"]
        < full_metrics["full_mean_mass_on_true_cell"]
    )
    # and yet substantially above chance for the configured grid.
    assert (
        latent_metrics["latent_mean_mass_on_true_cell"]
        > 3.0 / sequence.num_spatial_states
    )


def test_pipeline_emits_ground_truth_artifact(tmp_path) -> None:
    import textwrap

    from active_fractal_rabbit.config import FRACTALRABBIT_PINNED_COMMIT
    from active_fractal_rabbit.pipeline import run_pipeline_from_path

    config_path = tmp_path / "gt.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            run_id: gt
            fractalrabbit:
              upstream_commit: {FRACTALRABBIT_PINNED_COMMIT}
              execution: fixture
              fixture_csv: {FIXTURE}
            observation:
              spatial_bins: 3
              time_bins_per_day: 4
            model:
              smoothing: 0.05
            closed_loop:
              iterations: 1
              candidates: []
            """
        ),
        encoding="utf-8",
    )
    result = run_pipeline_from_path(config_path)

    gt_path = result.output_dir / "ground_truth.csv"
    assert gt_path.exists()
    header = gt_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "t,true_cell,reference_mass_on_true,reference_top1"
    assert result.metrics["state_recovery_reference_top1_accuracy"] == 1.0
    assert result.metadata["places_available"] is False
    assert "place_n_unique" not in result.metrics


def test_external_run_ground_truth_with_places_when_present() -> None:
    run_dir = ROOT / "output" / "runs" / "fractalrabbit_external"
    gt = run_dir / "ground_truth.csv"
    if not gt.exists():
        pytest.skip("no external run artifacts present; run the external lane first")
    header = gt.read_text(encoding="utf-8").splitlines()[0]
    assert "true_place_id" in header
    import json

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["place_n_unique"] > 0
    assert 0.0 <= metrics["place_revisit_rate"] <= 1.0
    assert metrics["state_recovery_reference_mean_mass_on_true_cell"] > 0.5
