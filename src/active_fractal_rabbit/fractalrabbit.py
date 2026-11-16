"""Boundary layer for the pinned Java FractalRabbit simulator."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from .config import parameter_rows
from .types import (
    FractalRabbitBuild,
    FractalRabbitConfig,
    FractalRabbitParams,
    ObservationBundle,
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
    if any(
        not all(math.isfinite(value) for value in (row.days, row.x_km, row.y_km))
        for row in rows
    ):
        raise ValueError(f"{csv_path} contains nonfinite waypoint values")
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
    if any(not math.isfinite(row.days) for row in rows):
        raise ValueError(f"{csv_path} contains nonfinite place times")
    return tuple(sorted(rows, key=lambda item: (item.traveller_id, item.days)))


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _validated_bundle(
    xy: Path,
    places: Path | None,
    parameters: Path | None = None,
    build: FractalRabbitBuild | None = None,
) -> ObservationBundle:
    from .ground_truth import align_places

    for path in (xy, places, parameters):
        if path is not None and (path.is_symlink() or not path.is_file()):
            raise FractalRabbitUnavailable(f"Expected a regular bundle member: {path}")
    observations = parse_waypoint_csv(xy)
    if places is not None:
        align_places(observations, parse_places_csv(places))
    return ObservationBundle(
        xy_path=xy,
        xy_sha256=file_sha256(xy),
        places_path=places,
        places_sha256=file_sha256(places) if places is not None else None,
        parameters_path=parameters,
        parameters_sha256=file_sha256(parameters) if parameters is not None else None,
        build=build,
    )


class FractalRabbitRunner:
    """Fetch, build, and execute FractalRabbit outside the committed project tree."""

    def __init__(self, config: FractalRabbitConfig) -> None:
        self.config = config

    @property
    def source_dir(self) -> Path:
        return (
            self.config.cache_dir / f"fractalrabbit-{self.config.upstream_commit[:12]}"
        )

    def _git(self, *args: str, cwd: Path | None = None) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.source_dir if cwd is None else cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _admit_source(self) -> None:
        if (
            Path(self._git("rev-parse", "--show-toplevel")).resolve()
            != self.source_dir.resolve()
        ):
            raise FractalRabbitUnavailable(
                "Cached source is not its own Git repository"
            )
        if self._git("remote", "get-url", "origin") != self.config.upstream_repo:
            raise FractalRabbitUnavailable(
                "Cached source origin does not match configured upstream"
            )
        if self._git("status", "--porcelain=v1", "--untracked-files=all"):
            raise FractalRabbitUnavailable(
                "Cached source is dirty; preserve and resolve local changes"
            )

    def prepare(self) -> FractalRabbitBuild:
        """Build a fresh private checkout and return its observed executable identity."""

        java_exe, java_home = self._check_external_requirements()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.source_dir.exists():
            subprocess.run(
                ["git", "clone", self.config.upstream_repo, str(self.source_dir)],
                check=True,
                text=True,
            )
        self._admit_source()
        self._git("fetch", "--all", "--tags")
        requested = self._git(
            "rev-parse", "--verify", f"{self.config.upstream_commit}^{{commit}}"
        )
        self._git("checkout", "--detach", requested)
        self._admit_source()
        commit = self._git("rev-parse", "HEAD")
        tree = self._git("rev-parse", "HEAD^{tree}")
        if commit != requested:
            raise FractalRabbitUnavailable(
                "Checked-out HEAD differs from requested revision"
            )
        maven_argv = ("mvn", "-q", "-DskipTests", "clean", "package")
        with tempfile.TemporaryDirectory(
            prefix="fractalrabbit-build-", dir=self.config.cache_dir
        ) as temporary:
            checkout = Path(temporary) / "source"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--no-hardlinks",
                    "--no-checkout",
                    str(self.source_dir.resolve()),
                    str(checkout),
                ],
                check=True,
                text=True,
            )
            self._git("checkout", "--detach", commit, cwd=checkout)
            if self._git("rev-parse", "HEAD", cwd=checkout) != commit:
                raise FractalRabbitUnavailable(
                    "Private build checkout has the wrong HEAD"
                )
            subprocess.run(
                list(maven_argv),
                cwd=checkout,
                check=True,
                text=True,
                env=os.environ | {"JAVA_HOME": str(java_home)},
            )
            if (
                self._git("rev-parse", "HEAD", cwd=checkout) != commit
                or self._git("rev-parse", "HEAD^{tree}", cwd=checkout) != tree
                or self._git(
                    "status", "--porcelain=v1", "--untracked-files=no", cwd=checkout
                )
            ):
                raise FractalRabbitUnavailable(
                    "Build changed its tracked source identity"
                )
            candidates = list((checkout / "target").glob("*jar-with-dependencies*.jar"))
            if len(candidates) != 1:
                raise FractalRabbitUnavailable(
                    f"Expected one freshly built dependency jar, found {len(candidates)}"
                )
            built = candidates[0]
            if built.is_symlink() or not built.is_file() or built.stat().st_size == 0:
                raise FractalRabbitUnavailable(
                    "Build did not produce a regular nonempty jar"
                )
            digest = file_sha256(built)
            artifacts = self.config.cache_dir / "executables"
            artifacts.mkdir(exist_ok=True)
            jar = artifacts / f"{commit}-{uuid.uuid4().hex}.jar"
            shutil.copyfile(built, jar)
            if file_sha256(jar) != digest:
                jar.unlink()
                raise FractalRabbitUnavailable("Published executable digest changed")
        return FractalRabbitBuild(
            self.config.upstream_repo,
            self.config.upstream_commit,
            commit,
            tree,
            jar.resolve(),
            digest,
            java_exe.resolve(),
            java_home.resolve(),
            maven_argv,
        )

    def run(self, output_prefix: str | Path) -> ObservationBundle:
        """Execute in an empty directory; retain only this invocation's valid bundle."""

        build = self.prepare()
        if file_sha256(build.jar_path) != build.jar_sha256:
            raise FractalRabbitUnavailable(
                "Prepared executable digest changed before execution"
            )
        prefix = Path(output_prefix).resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        generation = f"{prefix.name}-{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory(
            prefix=".simulator-", dir=prefix.parent
        ) as temporary:
            staging = Path(temporary)
            staged_prefix = staging / generation
            parameters = write_parameter_csv(
                self.config.params, staging / f"{generation}-parameters.csv"
            )
            parameter_digest = file_sha256(parameters)
            subprocess.run(
                [
                    str(build.java_executable),
                    "-cp",
                    str(build.jar_path),
                    "fractalRabbitGenerator.MainClassFR",
                    str(parameters),
                    str(staged_prefix),
                ],
                cwd=staging,
                check=True,
                text=True,
                env=os.environ | {"JAVA_HOME": str(build.java_home)},
            )
            if file_sha256(parameters) != parameter_digest:
                raise FractalRabbitUnavailable(
                    "Simulator changed its parameter snapshot"
                )
            if file_sha256(build.jar_path) != build.jar_sha256:
                raise FractalRabbitUnavailable(
                    "Executable digest changed during execution"
                )
            xy = Path(str(staged_prefix) + "XY.csv")
            companion = Path(str(staged_prefix) + "PLACES.csv")
            places = companion if companion.exists() or companion.is_symlink() else None
            bundle = _validated_bundle(xy, places, parameters, build)
            published: list[Path] = []
            try:
                for member in (parameters, places, xy):
                    if member is not None:
                        target = prefix.parent / member.name
                        member.rename(target)
                        published.append(target)
            except BaseException:
                for target in published:
                    target.unlink()
                raise
        return ObservationBundle(
            xy_path=prefix.parent / xy.name,
            xy_sha256=bundle.xy_sha256,
            places_path=prefix.parent / places.name if places is not None else None,
            places_sha256=bundle.places_sha256,
            parameters_path=prefix.parent / parameters.name,
            parameters_sha256=bundle.parameters_sha256,
            build=build,
        )

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


def resolve_observations(
    config: FractalRabbitConfig, run_dir: Path
) -> ObservationBundle:
    """Resolve a validated fixture or one fresh simulator invocation."""

    if config.execution == "fixture":
        assert config.fixture_csv is not None
        xy = config.fixture_csv
        candidate = (
            xy.with_name(xy.name[: -len("XY.csv")] + "PLACES.csv")
            if xy.name.endswith("XY.csv")
            else None
        )
        places = candidate if candidate is not None and candidate.exists() else None
        return _validated_bundle(xy, places)
    return FractalRabbitRunner(config).run(run_dir / config.output_prefix)


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
