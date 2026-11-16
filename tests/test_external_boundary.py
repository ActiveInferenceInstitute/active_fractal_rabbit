from __future__ import annotations

import subprocess

import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import (
    FractalRabbitRunner,
    FractalRabbitUnavailable,
    _java_major_version,
)


def test_runner_missing_toolchain_is_unavailable(monkeypatch) -> None:
    config = load_config("configs/fractalrabbit_external.yaml")
    runner = FractalRabbitRunner(config.fractalrabbit)

    monkeypatch.setattr(
        "active_fractal_rabbit.fractalrabbit.shutil.which", lambda _: None
    )

    with pytest.raises(FractalRabbitUnavailable, match="Missing external"):
        runner.prepare()


def test_java_major_version_handles_missing_and_present_java(monkeypatch) -> None:
    def raises(*args, **kwargs):
        raise OSError

    monkeypatch.setattr("active_fractal_rabbit.fractalrabbit.subprocess.run", raises)
    assert _java_major_version() is None

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stderr='openjdk version "21.0.1"\n')

    monkeypatch.setattr("active_fractal_rabbit.fractalrabbit.subprocess.run", fake_run)
    assert _java_major_version() == 21
