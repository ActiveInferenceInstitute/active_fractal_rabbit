from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import (
    FractalRabbitRunner,
    FractalRabbitUnavailable,
    _java_major_version,
    resolve_observation_csv,
)


def test_runner_properties_and_missing_toolchain(monkeypatch) -> None:
    config = load_config("configs/fractalrabbit_external.yaml")
    runner = FractalRabbitRunner(config.fractalrabbit)

    assert runner.source_dir.name.startswith("fractalrabbit-9933449c4f4f")
    assert runner.jar_path.name.endswith("jar-with-dependencies.jar")
    monkeypatch.setattr(
        "active_fractal_rabbit.fractalrabbit.shutil.which", lambda _: None
    )

    with pytest.raises(FractalRabbitUnavailable, match="Missing external"):
        runner.prepare()


def test_runner_prepare_and_run_are_subprocess_boundaries(
    tmp_path, monkeypatch
) -> None:
    config = load_config("configs/fractalrabbit_external.yaml")
    fr_config = config.fractalrabbit.__class__(
        upstream_repo=config.fractalrabbit.upstream_repo,
        upstream_commit=config.fractalrabbit.upstream_commit,
        cache_dir=tmp_path / "cache",
        execution="external",
        params=config.fractalrabbit.params,
        output_prefix="mocked",
    )
    runner = FractalRabbitRunner(fr_config)
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "active_fractal_rabbit.fractalrabbit.shutil.which", lambda name: f"/bin/{name}"
    )
    monkeypatch.setattr(
        "active_fractal_rabbit.fractalrabbit._java_major_version", lambda *args: 21
    )

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:2] == ["git", "clone"]:
            runner.source_dir.mkdir(parents=True)
        if cmd[:3] == ["mvn", "-q", "-DskipTests"]:
            runner.jar_path.parent.mkdir(parents=True, exist_ok=True)
            runner.jar_path.write_text("jar", encoding="utf-8")
        if str(cmd[0]).endswith("java") and cmd[1:3] == ["-cp", str(runner.jar_path)]:
            Path(str(cmd[-1]) + "XY.csv").write_text(
                "ID,Days,x(km),y(km)\n1,0,0,0\n", encoding="utf-8"
            )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("active_fractal_rabbit.fractalrabbit.subprocess.run", fake_run)

    assert runner.prepare() == runner.jar_path
    output = runner.run(tmp_path / "params.csv", tmp_path / "result")

    assert output.exists()
    assert ["git", "checkout", fr_config.upstream_commit] in calls
    assert any(
        str(call[0]).endswith("java") and call[1:3] == ["-cp", str(runner.jar_path)]
        for call in calls
    )


def test_runner_run_rebuilds_existing_cached_jar_before_execution(
    tmp_path, monkeypatch
) -> None:
    config = load_config("configs/fractalrabbit_external.yaml")
    fr_config = config.fractalrabbit.__class__(
        upstream_repo=config.fractalrabbit.upstream_repo,
        upstream_commit=config.fractalrabbit.upstream_commit,
        cache_dir=tmp_path / "cache",
        execution="external",
        params=config.fractalrabbit.params,
        output_prefix="mocked",
    )
    runner = FractalRabbitRunner(fr_config)
    runner.jar_path.parent.mkdir(parents=True)
    runner.jar_path.write_text("stale jar", encoding="utf-8")
    prepare_called = False

    def fake_prepare():
        nonlocal prepare_called
        prepare_called = True
        runner.jar_path.write_text("rebuilt jar", encoding="utf-8")
        return runner.jar_path

    def fake_run(cmd, **kwargs):
        if str(cmd[0]).endswith("java"):
            Path(str(cmd[-1]) + "XY.csv").write_text(
                "ID,Days,x(km),y(km)\n1,0,0,0\n", encoding="utf-8"
            )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner, "prepare", fake_prepare)
    monkeypatch.setattr(
        runner,
        "_check_external_requirements",
        lambda: (Path("/bin/java"), Path("/tmp/java-home")),
    )
    monkeypatch.setattr("active_fractal_rabbit.fractalrabbit.subprocess.run", fake_run)

    output = runner.run(tmp_path / "params.csv", tmp_path / "result")

    assert output.exists()
    assert prepare_called is True
    assert runner.jar_path.read_text(encoding="utf-8") == "rebuilt jar"


def test_runner_prepare_reports_missing_jar(tmp_path, monkeypatch) -> None:
    config = load_config("configs/fractalrabbit_external.yaml")
    fr_config = config.fractalrabbit.__class__(
        upstream_repo=config.fractalrabbit.upstream_repo,
        upstream_commit=config.fractalrabbit.upstream_commit,
        cache_dir=tmp_path / "cache",
        execution="external",
        params=config.fractalrabbit.params,
        output_prefix="mocked",
    )
    runner = FractalRabbitRunner(fr_config)
    monkeypatch.setattr(
        "active_fractal_rabbit.fractalrabbit.shutil.which", lambda name: f"/bin/{name}"
    )
    monkeypatch.setattr(
        "active_fractal_rabbit.fractalrabbit._java_major_version", lambda *args: 21
    )

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "clone"]:
            runner.source_dir.mkdir(parents=True)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("active_fractal_rabbit.fractalrabbit.subprocess.run", fake_run)

    with pytest.raises(FractalRabbitUnavailable, match="no jar"):
        runner.prepare()


def test_resolve_observation_csv_external_calls_runner(tmp_path, monkeypatch) -> None:
    config = load_config("configs/fractalrabbit_external.yaml")
    expected = tmp_path / "externalXY.csv"
    expected.write_text("ID,Days,x(km),y(km)\n1,0,0,0\n", encoding="utf-8")

    class FakeRunner:
        def __init__(self, _config):
            pass

        def run(self, parameter_csv, output_prefix):
            assert Path(parameter_csv).exists()
            assert Path(output_prefix).name == "fractalrabbit_external"
            return expected

    monkeypatch.setattr(
        "active_fractal_rabbit.fractalrabbit.FractalRabbitRunner", FakeRunner
    )

    assert resolve_observation_csv(config.fractalrabbit, tmp_path) == expected


def test_java_major_version_handles_missing_and_present_java(monkeypatch) -> None:
    def raises(*args, **kwargs):
        raise OSError

    monkeypatch.setattr("active_fractal_rabbit.fractalrabbit.subprocess.run", raises)
    assert _java_major_version() is None

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stderr='openjdk version "21.0.1"\n')

    monkeypatch.setattr("active_fractal_rabbit.fractalrabbit.subprocess.run", fake_run)
    assert _java_major_version() == 21
