"""Boundary layer for the pinned Java FractalRabbit simulator."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

from .config import parameter_rows
from .types import (
    FractalRabbitConfig,
    FractalRabbitParams,
    PlaceObservation,
    WaypointObservation,
)

REQUIRED_JAVA_MAJOR = 21

HOMEBREW_JDK_CANDIDATES = (
    Path("/opt/homebrew/opt/openjdk@21/bin/java"),
    Path("/opt/homebrew/opt/openjdk/bin/java"),
    Path("/usr/local/opt/openjdk@21/bin/java"),
)


class FractalRabbitUnavailable(RuntimeError):
    """Raised when the external FractalRabbit toolchain is not available."""


def resolve_java(required_major: int = REQUIRED_JAVA_MAJOR) -> tuple[Path, Path] | None:
    """Locate a Java executable with the required major version.

    Resolution order: ``AFR_JAVA_HOME`` env override, the PATH ``java``,
    Homebrew keg-only JDK locations (which are deliberately not on PATH),
    then ``/usr/libexec/java_home``. Returns ``(java_executable, java_home)``
    or ``None`` when no suitable JDK exists.
    """

    candidates: list[Path] = []
    env_home = os.environ.get("AFR_JAVA_HOME")
    if env_home:
        candidates.append(Path(env_home) / "bin" / "java")
    path_java = shutil.which("java")
    if path_java:
        candidates.append(Path(path_java))
    candidates.extend(HOMEBREW_JDK_CANDIDATES)
    java_home_lookup = _java_home_lookup(required_major)
    if java_home_lookup is not None:
        candidates.append(java_home_lookup / "bin" / "java")
    for candidate in candidates:
        resolved = Path(candidate)
        if not resolved.exists():
            continue
        major = _java_major_version(str(resolved))
        if major is not None and major >= required_major:
            home = resolved.resolve().parent.parent
            return resolved, home
    return None


def _java_home_lookup(required_major: int) -> Path | None:
    tool = Path("/usr/libexec/java_home")
    if not tool.exists():
        return None
    completed = subprocess.run(
        [str(tool), "-v", str(required_major)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    home = (completed.stdout or "").strip()
    return Path(home) if home else None


def write_parameter_csv(params: FractalRabbitParams, destination: str | Path) -> Path:
    """Write FractalRabbit's seven-row parameter CSV in upstream order."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(parameter_rows(params))
    return path


def parse_waypoint_csv(path: str | Path) -> tuple[WaypointObservation, ...]:
    """Parse FractalRabbit's `ID,Days,x(km),y(km)` output contract."""

    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"ID", "Days", "x(km)", "y(km)"}
        if set(reader.fieldnames or []) != required:
            raise ValueError(f"{csv_path} must contain header {sorted(required)}")
        rows = tuple(
            WaypointObservation(
                traveller_id=int(row["ID"]),
                days=float(row["Days"]),
                x_km=float(row["x(km)"]),
                y_km=float(row["y(km)"]),
            )
            for row in reader
        )
    if not rows:
        raise ValueError(f"{csv_path} contains no waypoint rows")
    return tuple(sorted(rows, key=lambda item: (item.traveller_id, item.days)))


def parse_places_csv(path: str | Path) -> tuple[PlaceObservation, ...]:
    """Parse FractalRabbit's `Traveler ID,Days,Place ID` ground-truth output.

    The upstream simulator writes this file row-aligned with the `*XY.csv`
    waypoint output: same traveller, same report time, plus the identity of
    the AgoraphobicPoints place generating the report.
    """

    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"Traveler ID", "Days", "Place ID"}
        if set(reader.fieldnames or []) != required:
            raise ValueError(f"{csv_path} must contain header {sorted(required)}")
        rows = tuple(
            PlaceObservation(
                traveller_id=int(row["Traveler ID"]),
                days=float(row["Days"]),
                place_id=int(row["Place ID"]),
            )
            for row in reader
        )
    if not rows:
        raise ValueError(f"{csv_path} contains no place rows")
    return tuple(sorted(rows, key=lambda item: (item.traveller_id, item.days)))


def places_csv_for(waypoint_csv: str | Path) -> Path | None:
    """Return the sibling PLACES file for a simulator XY output, if present."""

    xy = Path(waypoint_csv)
    if not xy.name.endswith("XY.csv"):
        return None
    candidate = xy.with_name(xy.name[: -len("XY.csv")] + "PLACES.csv")
    return candidate if candidate.exists() else None


class FractalRabbitRunner:
    """Fetch, build, and execute FractalRabbit outside the committed project tree."""

    def __init__(self, config: FractalRabbitConfig) -> None:
        self.config = config

    @property
    def source_dir(self) -> Path:
        return (
            self.config.cache_dir / f"fractalrabbit-{self.config.upstream_commit[:12]}"
        )

    @property
    def jar_path(self) -> Path:
        return (
            self.source_dir / "target" / "fractalrabbit-1.0-jar-with-dependencies.jar"
        )

    def prepare(self) -> Path:
        """Clone/check out the pinned source and build the executable jar."""

        java_exe, java_home = self._check_external_requirements()
        env = os.environ | {"JAVA_HOME": str(java_home)}
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.source_dir.exists():
            subprocess.run(
                ["git", "clone", self.config.upstream_repo, str(self.source_dir)],
                check=True,
                text=True,
            )
        subprocess.run(
            ["git", "fetch", "--all", "--tags"],
            cwd=self.source_dir,
            check=True,
            text=True,
        )
        subprocess.run(
            ["git", "checkout", self.config.upstream_commit],
            cwd=self.source_dir,
            check=True,
            text=True,
        )
        subprocess.run(
            ["mvn", "-q", "-DskipTests", "package"],
            cwd=self.source_dir,
            check=True,
            text=True,
            env=env,
        )
        if not self.jar_path.exists():
            candidates = sorted(
                (self.source_dir / "target").glob("*jar-with-dependencies*.jar")
            )
            if not candidates:
                raise FractalRabbitUnavailable(
                    "Maven completed but no jar-with-dependencies artifact was produced"
                )
            return candidates[0]
        return self.jar_path

    def run(self, parameter_csv: str | Path, output_prefix: str | Path) -> Path:
        """Run the simulator and return the generated `*XY.csv` file."""

        jar = self.prepare()
        java_exe, java_home = self._check_external_requirements()
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(java_exe),
                "-cp",
                str(jar),
                "fractalRabbitGenerator.MainClassFR",
                str(parameter_csv),
                str(prefix),
            ],
            check=True,
            text=True,
            env=os.environ | {"JAVA_HOME": str(java_home)},
        )
        output = Path(str(prefix) + "XY.csv")
        if not output.exists():
            raise FractalRabbitUnavailable(
                f"FractalRabbit did not produce expected output {output}"
            )
        return output

    def _check_external_requirements(self) -> tuple[Path, Path]:
        mvn = shutil.which("mvn")
        git = shutil.which("git")
        missing = [
            name for name, value in {"mvn": mvn, "git": git}.items() if not value
        ]
        if missing:
            raise FractalRabbitUnavailable(
                "Missing external FractalRabbit toolchain: "
                + ", ".join(missing)
                + ". Install JDK 21 and Maven, then rerun prepare."
            )
        resolved = resolve_java()
        if resolved is None:
            raise FractalRabbitUnavailable(
                f"FractalRabbit's Maven build targets Java {REQUIRED_JAVA_MAJOR} and no suitable JDK was found "
                "(checked AFR_JAVA_HOME, PATH, Homebrew kegs, and java_home). "
                "Install with `brew install openjdk@21`."
            )
        return resolved


def resolve_observation_csv(config: FractalRabbitConfig, run_dir: Path) -> Path:
    """Return a simulator output CSV, running FractalRabbit or using a fixture."""

    if config.execution == "fixture":
        assert config.fixture_csv is not None
        return config.fixture_csv
    params_csv = write_parameter_csv(config.params, run_dir / "parameters.csv")
    runner = FractalRabbitRunner(config)
    return runner.run(params_csv, run_dir / config.output_prefix)


def _java_major_version(java_executable: str = "java") -> int | None:
    try:
        completed = subprocess.run(
            [java_executable, "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    text = completed.stderr or completed.stdout
    for token in text.replace('"', " ").split():
        if token[0].isdigit():
            return int(token.split(".")[0])
    return None
