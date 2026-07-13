from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from active_fractal_rabbit.cover_art import REQUIRED_COVER_ARTIFACTS, render_cover_art


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _seed_cover_project(root: Path) -> None:
    run_dir = root / "output" / "runs" / "default"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = ["ID,Days,x(km),y(km)"]
    features = ["ID,step,x(km),y(km),x_bin,y_bin,spatial_state"]
    step = 0
    for traveller in range(4):
        for y_bin in range(4):
            for x_bin in range(4):
                x = 10.0 + x_bin * 4.0 + traveller * 0.2
                y = 10.0 + y_bin * 4.0 + traveller * 0.2
                rows.append(f"{traveller},{step / 10:.1f},{x:.2f},{y:.2f}")
                features.append(
                    f"{traveller},{step},{x:.2f},{y:.2f},{x_bin},{y_bin},{y_bin * 4 + x_bin}"
                )
                step += 1
    (run_dir / "observations.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (run_dir / "discrete_features.csv").write_text(
        "\n".join(features) + "\n", encoding="utf-8"
    )
    _write_json(
        run_dir / "metrics.json", {"pymdp_exact_inference_max_abs_delta": 2.4e-7}
    )
    _write_json(
        root / "output" / "runs" / "spatial_resolution_frontier.json",
        {
            "status": "completed",
            "default_spatial_bins": 4,
            "default_state_count": 16,
            "default_row": {
                "spatial_bins": 4,
                "full_location_mean_predictive_loss": 3.84,
                "no_location_mean_mass_on_true_cell": 0.076,
            },
            "rows": [
                {
                    "spatial_bins": 3,
                    "full_location_mean_predictive_loss": 3.96,
                    "empty_state_count": 0,
                },
                {
                    "spatial_bins": 4,
                    "full_location_mean_predictive_loss": 3.84,
                    "empty_state_count": 0,
                },
                {
                    "spatial_bins": 5,
                    "full_location_mean_predictive_loss": 4.43,
                    "empty_state_count": 2,
                },
            ],
        },
    )
    _write_json(
        root / "output" / "runs" / "model_gate_matrix.json",
        {
            "status": "completed",
            "lanes": [{"status": "completed"}, {"status": "completed"}],
        },
    )
    _write_json(
        root / "output" / "runs" / "forward_gates_summary.json",
        {
            "status": "completed",
            "fixture_completed_count": 8,
            "external_completed_count": 8,
        },
    )
    _write_json(
        root / "output" / "runs" / "efe_location_frontier.json",
        {
            "status": "completed",
            "verdict": "metadata_recovery_not_above_structure_null",
        },
    )
    _write_json(
        root / "output" / "runs" / "linkage_risk_frontier.json",
        {
            "status": "completed",
            "verdict": "location_carries_relink_risk_that_minimization_removes",
        },
    )
    _write_json(
        root / "output" / "runs" / "poisoning_robustness.json",
        {"status": "completed", "verdict": "poisoning_detection_regime_changed"},
    )
    _write_json(
        root / "output" / "runs" / "pymdp_benchmark_matrix.json",
        {"status": "completed"},
    )


def test_cover_art_fails_when_required_artifact_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=REQUIRED_COVER_ARTIFACTS[0]):
        render_cover_art(tmp_path)


def test_cover_art_renders_nonblank_artifact(tmp_path: Path) -> None:
    _seed_cover_project(tmp_path)
    output = render_cover_art(tmp_path)

    assert output == tmp_path / "output" / "cover" / "active_fractal_rabbit_cover.png"
    assert output.stat().st_size > 20_000
    image = mpimg.imread(output)
    assert image.shape[0] >= 1900
    assert image.shape[1] >= 3200
    assert float(np.std(image[..., :3])) > 0.03
    assert float(np.mean(image[:20, :, :3])) > 0.90
    assert float(np.mean(image[-20:, :, :3])) > 0.90
    image = mpimg.imread(output)
    assert image.shape[0] >= 1000
    assert image.shape[1] >= 1800
    assert image.shape[1] > image.shape[0]
    assert float(np.ptp(image)) > 0.05
