from __future__ import annotations

from pathlib import Path

import pytest

from active_fractal_rabbit.config import FRACTALRABBIT_PINNED_COMMIT, load_config


def _write(tmp_path: Path, extra: str) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    path = config_dir / "case.yaml"
    path.write_text(extra, encoding="utf-8")
    return path


def _base(**params: object) -> str:
    merged = {
        "space_dimension": 2,
        "fractal_dimension": 1.33,
        "number_of_points": 10,
        "number_of_travellers": 2,
        "number_of_co_travellers": 1,
        "number_of_days": 2.0,
        "number_of_reports": 3.0,
    } | params
    param_lines = "\n".join(
        f"                {key}: {value}" for key, value in merged.items()
    )
    return f"""
run_id: case
fractalrabbit:
  upstream_commit: {FRACTALRABBIT_PINNED_COMMIT}
  execution: fixture
  fixture_csv: data.csv
  params:
{param_lines}
observation:
  spatial_bins: 2
  time_bins_per_day: 2
  speed_thresholds_km_per_day: [1, 2, 3]
model:
  smoothing: 0.1
closed_loop:
  iterations: 1
  objective: predictive_loss
  candidates: []
"""


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"space_dimension": 3}, "2D"),
        ({"fractal_dimension": 0}, "fractal_dimension"),
        ({"number_of_points": 1}, "number_of_points"),
        ({"number_of_travellers": 0}, "number_of_travellers"),
        ({"number_of_co_travellers": -1}, "co_travellers"),
        ({"number_of_co_travellers": 3}, "cannot exceed"),
        ({"number_of_days": 0}, "number_of_days"),
        ({"number_of_reports": 0}, "number_of_reports"),
    ],
)
def test_rejects_bad_fractalrabbit_params(tmp_path, params, message) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(_write(tmp_path, _base(**params)))


def test_rejects_bad_top_level_shapes(tmp_path) -> None:
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(_write(tmp_path, "- bad\n"))

    with pytest.raises(ValueError, match="fractalrabbit"):
        load_config(
            _write(
                tmp_path / "a",
                """
run_id: bad
fractalrabbit: []
observation:
  spatial_bins: 2
  time_bins_per_day: 2
model:
  smoothing: 0.1
closed_loop:
  iterations: 1
""",
            )
        )


def test_rejects_removed_top_level_inference_block(tmp_path) -> None:
    config = (
        _base()
        .replace("observation:", "inference:", 1)
        .replace("model:\n  smoothing: 0.1\n", "")
    )

    with pytest.raises(ValueError, match="top-level inference block has been removed"):
        load_config(_write(tmp_path, config))


def test_rejects_bad_execution_objective_and_thresholds(tmp_path) -> None:
    with pytest.raises(ValueError, match="execution"):
        load_config(
            _write(
                tmp_path / "b",
                _base().replace("execution: fixture", "execution: other"),
            )
        )

    with pytest.raises(ValueError, match="objective"):
        load_config(
            _write(
                tmp_path / "c",
                _base().replace("objective: predictive_loss", "objective: unknown"),
            )
        )

    with pytest.raises(ValueError, match="three thresholds"):
        load_config(_write(tmp_path / "d", _base().replace("[1, 2, 3]", "[1, 2]")))

    with pytest.raises(ValueError, match="sorted"):
        load_config(_write(tmp_path / "e", _base().replace("[1, 2, 3]", "[2, 1, 3]")))

    with pytest.raises(ValueError, match="spatial_bins"):
        load_config(
            _write(
                tmp_path / "f", _base().replace("spatial_bins: 2", "spatial_bins: 0")
            )
        )

    with pytest.raises(ValueError, match="smoothing"):
        load_config(
            _write(tmp_path / "g", _base().replace("smoothing: 0.1", "smoothing: 0"))
        )


def test_fixture_execution_requires_fixture_csv(tmp_path) -> None:
    with pytest.raises(ValueError, match="fixture_csv"):
        load_config(_write(tmp_path, _base().replace("  fixture_csv: data.csv\n", "")))


def test_rejects_bad_grouped_config_values(tmp_path) -> None:
    with pytest.raises(ValueError, match="spatial_discretizer"):
        load_config(
            _write(
                tmp_path / "obs_a",
                _base().replace(
                    "  time_bins_per_day: 2",
                    "  time_bins_per_day: 2\n  spatial_discretizer: diagonal",
                ),
            )
        )

    with pytest.raises(ValueError, match="spatial_bounds"):
        load_config(
            _write(
                tmp_path / "obs_b",
                _base().replace(
                    "  time_bins_per_day: 2",
                    "  time_bins_per_day: 2\n  spatial_discretizer: fixed_bounds",
                ),
            )
        )

    with pytest.raises(ValueError, match="unknown control"):
        load_config(
            _write(
                tmp_path / "model",
                _base().replace(
                    "  smoothing: 0.1", "  smoothing: 0.1\n  controls: [teleport]"
                ),
            )
        )

    with pytest.raises(ValueError, match="objective_direction"):
        load_config(
            _write(
                tmp_path / "experiment",
                _base() + "\nexperiment:\n  objective_direction: sideways\n",
            )
        )

    with pytest.raises(ValueError, match="figures.enabled"):
        load_config(
            _write(tmp_path / "figures", _base() + "\nfigures:\n  enabled: []\n")
        )

    with pytest.raises(ValueError, match="external_realization_fractal_dimensions"):
        load_config(
            _write(
                tmp_path / "robustness",
                _base()
                + "\nrobustness:\n  external_realization_fractal_dimensions: [1.21, 0]\n",
            )
        )
