"""Environment checks for the project and optional FractalRabbit toolchain."""

from __future__ import annotations

import importlib.metadata as metadata
import shutil
import subprocess


def preflight_status() -> dict[str, str]:
    """Return status strings without mutating the environment."""

    status: dict[str, str] = {}
    for package in ("numpy", "inferactively-pymdp"):
        try:
            status[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            status[package] = "missing"
    status["git"] = shutil.which("git") or "missing"
    status["mvn"] = shutil.which("mvn") or "missing"
    status["java"] = _java_version()
    return status


def print_preflight() -> int:
    """Print a concise readiness report."""

    status = preflight_status()
    for key in sorted(status):
        print(f"{key}: {status[key]}")
    ready, reasons = external_lane_status()
    if ready:
        print("external FractalRabbit lane: ready (JDK 21 resolved)")
    else:
        for reason in reasons:
            print(f"external FractalRabbit lane blocked: {reason}")
        print("Fixture-mode tests do not need the external toolchain.")
    return 0


def external_lane_status() -> tuple[bool, list[str]]:
    """Return readiness of the external FractalRabbit lane with explicit reasons."""

    from .fractalrabbit import REQUIRED_JAVA_MAJOR, resolve_java

    status = preflight_status()
    reasons: list[str] = []
    if status["git"] == "missing":
        reasons.append("git is not installed")
    if status["mvn"] == "missing":
        reasons.append(
            "Maven (mvn) is not installed; install with `brew install maven`"
        )
    if resolve_java() is None:
        if status["java"] == "missing":
            reasons.append(
                f"Java is not installed; the upstream POM targets JDK {REQUIRED_JAVA_MAJOR} "
                "(`brew install openjdk@21`)"
            )
        else:
            major = _java_major(status["java"])
            reasons.append(
                f"PATH Java is {major} and no JDK {REQUIRED_JAVA_MAJOR} was found via AFR_JAVA_HOME, "
                "Homebrew kegs, or java_home; install with `brew install openjdk@21`"
            )
    return (not reasons, reasons)


def _java_version() -> str:
    java = shutil.which("java")
    if not java:
        return "missing"
    completed = subprocess.run(
        ["java", "-version"], capture_output=True, text=True, check=False
    )
    lines = (completed.stderr or completed.stdout).splitlines()
    first = lines[0] if lines else "java present but version unknown"
    return first


def _java_major(text: str) -> int:
    for token in text.replace('"', " ").split():
        if token and token[0].isdigit():
            parts = token.split(".")
            if parts[0] == "1" and len(parts) > 1:
                # Pre-JEP223 (pre-Java-9) versioning: "1.8.0_292" -> major 8.
                return int(parts[1])
            return int(parts[0])
    return 0
