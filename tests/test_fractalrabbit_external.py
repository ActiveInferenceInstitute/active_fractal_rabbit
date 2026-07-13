from __future__ import annotations

from pathlib import Path

import pytest

from active_fractal_rabbit.compare import compare_tensor_summary_files
from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import FractalRabbitRunner
from active_fractal_rabbit.preflight import external_lane_status

ROOT = Path(__file__).resolve().parents[1]


def test_external_lane_smoke_reports_explicit_skip_reasons() -> None:
    """The external lane must always explain exactly why it is unavailable."""

    ready, reasons = external_lane_status()
    if ready:
        assert reasons == []
    else:
        assert reasons
        for reason in reasons:
            assert any(token in reason for token in ("git", "Maven", "Java", "JDK"))


@pytest.mark.requires_fractalrabbit
def test_external_fractalrabbit_prepare_is_available_when_toolchain_exists() -> None:
    ready, reasons = external_lane_status()
    if not ready:
        pytest.skip("external FractalRabbit lane unavailable: " + "; ".join(reasons))

    config = load_config("configs/fractalrabbit_external.yaml")
    jar = FractalRabbitRunner(config.fractalrabbit).prepare()

    assert jar.exists()


def test_fixture_vs_external_tensor_comparison_when_artifacts_exist() -> None:
    """Compare lanes descriptively; skip honestly when no external run exists."""

    fixture_summary = ROOT / "output" / "runs" / "default" / "tensor_summary.json"
    external_summary = (
        ROOT / "output" / "runs" / "fractalrabbit_external" / "tensor_summary.json"
    )
    if not fixture_summary.exists():
        pytest.skip(
            "default fixture run artifacts missing; run scripts/02_run_pipeline.py first"
        )
    if not external_summary.exists():
        pytest.skip(
            "no external FractalRabbit run artifacts present; run scripts/02_run_pipeline.py "
            "--config configs/fractalrabbit_external.yaml after installing JDK 21 and Maven"
        )
    report = compare_tensor_summary_files(fixture_summary, external_summary)
    assert report["same_modalities"] is True
    assert "not evidence of external validity" in str(report["caveat"])
