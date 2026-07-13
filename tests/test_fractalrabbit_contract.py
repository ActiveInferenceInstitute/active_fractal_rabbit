from __future__ import annotations

import csv

import pytest

from active_fractal_rabbit.config import load_config, parameter_rows
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv, write_parameter_csv


def test_write_parameter_csv_uses_upstream_labels(tmp_path) -> None:
    params = load_config("configs/default.yaml").fractalrabbit.params
    output = write_parameter_csv(params, tmp_path / "parameters.csv")

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows == [list(row) for row in parameter_rows(params)]


def test_parse_fractalrabbit_waypoints_fixture() -> None:
    rows = parse_waypoint_csv("data/fixtures/fractalrabbit_sample.csv")

    assert len(rows) == 16
    assert rows[0].traveller_id == 0
    assert rows[0].days == pytest.approx(0.10)
    assert rows[-1].traveller_id == 3


def test_parse_rejects_wrong_header(tmp_path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("id,t,x,y\n1,0,0,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="header"):
        parse_waypoint_csv(bad)
