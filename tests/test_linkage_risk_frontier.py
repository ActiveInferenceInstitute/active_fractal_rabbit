"""Re-identification linkage-risk frontier: oversight-positive, structure-null + positive-control gated."""

from __future__ import annotations

import json

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.linkage_risk_frontier import (
    linkage_risk_frontier,
    render_linkage_risk_figure,
    run_and_write_linkage_risk_frontier,
)
from active_fractal_rabbit.observations import discretize_observations

CONFIG = "configs/default.yaml"
_REPEATS = 120
_NULL = 60


def _frontier():
    config = load_config(CONFIG)
    sequence = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    return linkage_risk_frontier(
        sequence,
        synthetic_provenance="fractalrabbit_fixture",
        n_repeats=_REPEATS,
        n_structure_null=_NULL,
    ), config


def _row(payload, regime):
    return next(r for r in payload["frontier"] if r["regime"] == regime)


def test_location_carries_relink_risk_above_the_structure_null() -> None:
    payload, _ = _frontier()
    location = _row(payload, "all")
    assert (
        location["real_linkage"] > location["structure_null"]["p95"]
    )  # genuine linkable structure
    assert location["structure_null"]["p_value_null_ge_real"] == 0.0
    # null is calibrated near chance (no built-in apparatus bias)
    for row in payload["frontier"]:
        assert 0.18 <= row["structure_null"]["mean"] <= 0.30


def test_minimization_collapses_relink_risk_toward_the_null() -> None:
    payload, _ = _frontier()
    # withholding the cell label drops re-linkability inside the null band
    assert _row(payload, "no_location")["structure_null"]["exceeds_null"] is False
    assert _row(payload, "time_burst")["structure_null"]["exceeds_null"] is False
    assert _row(payload, "time_only")["structure_null"]["exceeds_null"] is False
    assert payload["minimization_protects"] is True
    assert (
        payload["verdict"] == "location_carries_relink_risk_that_minimization_removes"
    )


def test_positive_control_proves_the_test_has_power() -> None:
    """An injected unique synthetic signature must be detected — so a minimized
    regime reading as null means 'minimization protects', not 'test too weak'."""

    payload, _ = _frontier()
    control = payload["positive_control"]
    assert control["gate_passed"] is True
    assert control["target_real_linkage"] > 0.6
    assert control["target_real_linkage"] > control["structure_null_p95"]


def test_frontier_is_deterministic_and_writes_and_renders(tmp_path) -> None:
    config = load_config(CONFIG)
    sequence = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    a = linkage_risk_frontier(
        sequence,
        synthetic_provenance="fractalrabbit_fixture",
        n_repeats=60,
        n_structure_null=20,
        seed=0,
    )
    b = linkage_risk_frontier(
        sequence,
        synthetic_provenance="fractalrabbit_fixture",
        n_repeats=60,
        n_structure_null=20,
        seed=0,
    )
    assert json.dumps(a, sort_keys=True) == json.dumps(
        b, sort_keys=True
    )  # deterministic

    out = run_and_write_linkage_risk_frontier(CONFIG, tmp_path / "linkage.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "active-fractal-rabbit-linkage-risk-frontier-v1"
    assert "oversight-positive" in payload["caveat"]

    import matplotlib.image as mpimg
    import numpy as np

    figure = render_linkage_risk_figure(a, tmp_path / "linkage.png")
    assert figure.exists() and figure.stat().st_size > 1000
    image = mpimg.imread(figure)
    assert (
        image.shape[0] >= 300 and image.shape[1] >= 400 and float(np.ptp(image)) > 0.01
    )


def test_linkage_refuses_non_synthetic_provenance() -> None:
    """The boundary is enforced in CODE: the re-id RISK measure refuses to run on
    anything not marked as a recognized synthetic generator."""

    import pytest

    config = load_config(CONFIG)
    sequence = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    with pytest.raises(ValueError, match="refuses non-synthetic provenance"):
        linkage_risk_frontier(
            sequence,
            synthetic_provenance="real_field_collection",
            n_repeats=20,
            n_structure_null=10,
        )
