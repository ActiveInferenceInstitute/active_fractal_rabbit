"""The EFE location-minimization frontier: an oversight-positive boundary.

Location dominates the joint information gain. On the 4x4 expanded fixture the
metadata bundle remains small but no longer clears the structure-destroyed null,
so withholding location still collapses localizability while metadata-only
recovery is kept as a bounded null.
"""

from __future__ import annotations

import json

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.efe_location_frontier import (
    efe_location_frontier,
    run_and_write_efe_location_frontier,
)
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.observations import discretize_observations
from active_fractal_rabbit.pomdp import build_pomdp

CONFIG = "configs/default.yaml"


def _frontier():
    config = load_config(CONFIG)
    sequence = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    model = build_pomdp(
        sequence, config.inference.model.smoothing, config.inference.model.controls
    )
    return efe_location_frontier(
        sequence, model, smoothing=config.inference.model.smoothing
    ), config


def test_location_dominates_the_information_gain_frontier() -> None:
    payload, _ = _frontier()
    assert payload["location_joint_share"] > 0.95
    assert payload["frontier_cost_nats"] > 1.0
    mi = payload["mutual_information"]
    assert mi["location"] > 10 * max(mi["time_of_day"], mi["speed"], mi["report_burst"])


def test_metadata_bundle_is_small_and_not_above_the_structure_null() -> None:
    """The 4x4 expanded fixture demotes the previous metadata-positive claim."""

    payload, _ = _frontier()
    assert 0.02 <= payload["metadata_bundle_mean"] <= 0.03
    assert payload["fraction_recovered_by_metadata"] < 0.05
    null = payload["structure_null"]
    assert null["gate_passed"] is False
    assert 0.0 < null["p_value_null_ge_real"] <= 1.0
    assert payload["verdict"] == "metadata_recovery_not_above_structure_null"
    # POSITIVE CONTROL (the decisive non-vacuity): the same test REJECTS when a
    # genuine cell-coupled metadatum is injected, so a null verdict would not
    # mean "test too weak".
    assert payload["structure_null_power_check"]["gate_passed"] is True
    assert payload["residual_localized_without_location"] == 0.0


def test_time_to_confident_localization_collapses_without_location() -> None:
    payload, _ = _frontier()
    localize = payload["time_to_localize"]
    assert localize["with_location"]["frac_localized"] == 1.0
    assert localize["without_location"]["frac_localized"] <= 0.5  # near-total collapse


def test_frontier_artifact_writes(tmp_path) -> None:
    _, config = _frontier()
    out = run_and_write_efe_location_frontier(CONFIG, tmp_path / "frontier.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "active-fractal-rabbit-efe-location-frontier-v1"
    assert payload["status"] == "completed"


def test_frontier_figure_renders_nonblank(tmp_path) -> None:
    import matplotlib.image as mpimg
    import numpy as np

    from active_fractal_rabbit.efe_location_frontier import render_frontier_figure

    payload, _ = _frontier()
    out = render_frontier_figure(payload, tmp_path / "frontier.png")
    assert out.exists() and out.stat().st_size > 1000
    image = mpimg.imread(out)
    assert image.shape[0] >= 300 and image.shape[1] >= 400
    assert float(np.ptp(image)) > 0.01
