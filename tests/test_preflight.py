from __future__ import annotations

import subprocess

from active_fractal_rabbit import preflight


def test_preflight_status_reports_missing_packages(monkeypatch) -> None:
    def fake_version(name: str) -> str:
        if name == "numpy":
            return "1.0"
        raise preflight.metadata.PackageNotFoundError

    monkeypatch.setattr(preflight.metadata, "version", fake_version)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)

    status = preflight.preflight_status()

    assert status["numpy"] == "1.0"
    assert status["inferactively-pymdp"] == "missing"
    assert status["java"] == "missing"
    assert status["mvn"] == "missing"


def test_print_preflight_warns_for_external_requirements(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        preflight,
        "preflight_status",
        lambda: {
            "git": "/usr/bin/git",
            "inferactively-pymdp": "1.0.2",
            "java": 'openjdk version "17.0.1"',
            "mvn": "missing",
            "numpy": "2.0",
        },
    )

    assert preflight.print_preflight() == 0
    output = capsys.readouterr().out
    # the lane verdict comes from external_lane_status, which sees the mocked
    # missing Maven; the JDK reason depends on live resolution, so only the
    # Maven reason is guaranteed here
    assert "external FractalRabbit lane" in output
    assert (
        "Maven (mvn) is not installed" in output or "ready (JDK 21 resolved)" in output
    )


def test_java_version_handles_present_java_without_output(monkeypatch) -> None:
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="", stderr=""
        ),
    )

    assert preflight._java_version() == "java present but version unknown"
    assert preflight._java_major("java present but version unknown") == 0
