"""Run the pinned GNN conformance gate.

The gate is the one recorded in ``.github/gnn-pin.json``:

1. resolve a checkout whose ``HEAD`` is exactly the recorded revision,
2. install that checkout into its own virtualenv,
3. run the upstream strict validator on every spec named in the pin,
4. run ``tests/test_gnn_spec.py`` against that checkout and require a nonempty
   suite with zero skips, failures, or errors.

A missing checkout, a revision mismatch, an unimportable parser, or a skipped
parser test is a gate failure, never a silent pass. Set ``GNN_CHECKOUT`` to a
prepared checkout at the pinned revision to run offline; otherwise the gate
clones the pinned tag into ``GNN_SCRATCH`` (or a temporary directory).

Exit codes:
    0: every pinned check passed
    1: the gate ran and failed
    2: the pin file itself is missing or malformed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
PIN_PATH = ROOT / ".github" / "gnn-pin.json"
PIN_SCHEMA = "gnn-importer-pin/v1"
REQUIRED_KEYS = (
    "schema",
    "upstream_repository",
    "tag",
    "revision",
    "version",
    "spec_files",
)
CLONE_ATTEMPTS = 3


class PinError(RuntimeError):
    """The recorded pin cannot be used as a gate."""


def load_pin(path: Path = PIN_PATH) -> dict[str, object]:
    """Load and validate the recorded pin."""

    try:
        pin = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PinError(f"pin file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise PinError(f"pin file is not valid JSON: {path}: {error}") from error
    if not isinstance(pin, dict):
        raise PinError(f"pin file must contain a JSON object: {path}")
    missing = [key for key in REQUIRED_KEYS if key not in pin]
    if missing:
        raise PinError(f"pin file is missing keys {missing}: {path}")
    if pin["schema"] != PIN_SCHEMA:
        raise PinError(f"pin schema is {pin['schema']!r}, expected {PIN_SCHEMA!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(pin["revision"])):
        raise PinError(f"pin revision is not a 40-character commit id: {pin['revision']!r}")
    specs = pin["spec_files"]
    if not isinstance(specs, list) or not specs:
        raise PinError("pin spec_files must be a nonempty list")
    for spec in specs:
        if not isinstance(spec, str) or not (ROOT / spec).is_file():
            raise PinError(f"pinned spec file is missing: {spec}")
    return pin


def _git(checkout: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _head(checkout: Path) -> str:
    result = _git(checkout, "rev-parse", "HEAD")
    if result.returncode != 0:
        raise PinError(f"cannot read HEAD of {checkout}: {result.stderr.strip()}")
    return result.stdout.strip()


def verify_checkout(checkout: Path, pin: dict[str, object]) -> None:
    """Require the checkout to stand exactly at the pinned revision."""

    revision = str(pin["revision"])
    head = _head(checkout)
    if head != revision:
        raise PinError(
            f"checkout {checkout} is at {head}, not the pinned revision {revision}; "
            "check out the pin or unset GNN_CHECKOUT to let the gate clone it"
        )
    dirty = _git(checkout, "status", "--porcelain")
    if dirty.returncode != 0:
        raise PinError(f"cannot read status of {checkout}: {dirty.stderr.strip()}")
    if dirty.stdout.strip():
        raise PinError(f"checkout {checkout} has local changes; the gate requires a clean pin")


def clone_pin(pin: dict[str, object], scratch: Path) -> Path:
    """Clone the pinned tag into *scratch* and verify the revision."""

    checkout = scratch / "gnn"
    revision = str(pin["revision"])
    tag = str(pin["tag"])
    upstream = str(pin["upstream_repository"])
    for attempt in range(1, CLONE_ATTEMPTS + 1):
        shutil.rmtree(checkout, ignore_errors=True)
        result = subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--depth",
                "1",
                "--branch",
                tag,
                "--",
                upstream,
                str(checkout),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0 and (checkout / ".git").exists():
            if _head(checkout) == revision:
                tag_target = _git(checkout, "rev-parse", f"refs/tags/{tag}^{{commit}}")
                if tag_target.returncode == 0 and tag_target.stdout.strip() == revision:
                    return checkout
        print(
            f"attempt {attempt}: pinned tag {tag} did not resolve to {revision}",
            file=sys.stderr,
        )
        time.sleep(attempt * 5)
    raise PinError(f"pinned GNN revision {revision} could not be resolved after {CLONE_ATTEMPTS} attempts")


def _venv_python(checkout: Path) -> Path:
    return checkout / ".venv" / "bin" / "python"


def _required_uv(checkout: Path) -> str | None:
    """Return the uv version the pinned checkout pins for itself, if any."""

    for path, keys in (
        (checkout / "uv.toml", ("required-version",)),
        (checkout / "pyproject.toml", ("tool", "uv", "required-version")),
    ):
        if not path.is_file():
            continue
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as error:
            raise PinError(f"cannot parse {path}: {error}") from error
        value: object = payload
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, str):
            return value.removeprefix("==")
    return None


def _sync_command(checkout: Path) -> list[str]:
    """Build the frozen sync command for the checkout's own environment."""

    required = _required_uv(checkout)
    prefix = ["uvx", "--from", f"uv=={required}", "uv"] if required else ["uv"]
    return [
        *prefix,
        "sync",
        "--project",
        str(checkout),
        "--frozen",
        "--no-default-groups",
        "--python",
        "3.14",
    ]


def ensure_parser(checkout: Path, pin: dict[str, object]) -> Path:
    """Return the checkout's interpreter, installing the parser when needed."""

    python = _venv_python(checkout)

    def probe() -> subprocess.CompletedProcess[str] | None:
        if not python.exists():
            return None
        return subprocess.run(
            [str(python), "-c", "import gnn; print(gnn.__version__)"],
            cwd=str(checkout),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=300,
        )

    result = probe()
    if result is None or result.returncode != 0:
        command = _sync_command(checkout)
        print(f"installing the pinned parser: {' '.join(command)}")
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if completed.returncode != 0:
            raise PinError(
                f"command failed: {' '.join(command)}\n{completed.stderr.strip()[-2000:]}"
            )
        result = probe()
    if result is None or result.returncode != 0:
        stderr = "" if result is None else result.stderr.strip()[-2000:]
        raise PinError(f"pinned parser is not importable: {stderr}")
    observed = result.stdout.strip()
    expected = str(pin["version"])
    if observed != expected:
        raise PinError(f"parser version is {observed!r}, expected pinned {expected!r}")
    return python


def validate_specs(checkout: Path, python: Path, pin: dict[str, object]) -> list[str]:
    """Run the upstream strict validator on every pinned spec."""

    failures: list[str] = []
    for spec in pin["spec_files"]:
        spec_path = (ROOT / str(spec)).resolve()
        result = subprocess.run(
            [str(python), "-B", "-m", "gnn.cli", "validate", str(spec_path), "--strict"],
            cwd=str(checkout),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = (result.stdout + result.stderr).strip()
        print(output)
        if result.returncode != 0:
            failures.append(f"{spec}: strict validation exited {result.returncode}")
    return failures


def run_parser_tests(checkout: Path, evidence: Path) -> list[str]:
    """Run the focused parser tests and require a nonempty, zero-skip suite."""

    junit = evidence / "gnn-pytest.xml"
    env = dict(os.environ)
    env["GNN_CHECKOUT"] = str(checkout)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "tests/test_gnn_spec.py",
            "-q",
            "-ra",
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit}",
        ],
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    print((result.stdout + result.stderr).strip())
    if result.returncode != 0:
        return [f"tests/test_gnn_spec.py exited {result.returncode}"]
    if not junit.is_file():
        return [f"pytest did not write {junit}"]
    report = ElementTree.parse(junit)
    cases = report.findall(".//testcase")
    if not cases:
        return ["the pinned parser suite reported no test cases"]
    blocked = [
        tag
        for tag in ("skipped", "failure", "error")
        if report.findall(f".//{tag}")
    ]
    if blocked:
        return [f"the pinned parser suite reported {blocked}; zero skips are required"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch",
        type=Path,
        default=None,
        help="directory for the cloned pin (default: $GNN_SCRATCH or a temp dir)",
    )
    args = parser.parse_args(argv)

    try:
        pin = load_pin()
    except PinError as error:
        print(f"gnn conformance: {error}", file=sys.stderr)
        return 2

    scratch_override = args.scratch or (
        Path(os.environ["GNN_SCRATCH"]).expanduser() if os.environ.get("GNN_SCRATCH") else None
    )
    scratch = scratch_override or Path(tempfile.mkdtemp(prefix="afr-gnn-"))
    scratch.mkdir(parents=True, exist_ok=True)
    created_scratch = scratch_override is None
    try:
        override = os.environ.get("GNN_CHECKOUT")
        if override:
            checkout = Path(override).expanduser().resolve()
            if not checkout.is_dir():
                raise PinError(f"GNN_CHECKOUT is not a directory: {checkout}")
            verify_checkout(checkout, pin)
        else:
            checkout = clone_pin(pin, scratch)
        python = ensure_parser(checkout, pin)
        failures = validate_specs(checkout, python, pin)
        if not failures:
            failures = run_parser_tests(checkout, scratch)
    except (PinError, OSError, subprocess.SubprocessError) as error:
        print(f"gnn conformance: {error}", file=sys.stderr)
        return 1
    finally:
        if created_scratch and not os.environ.get("GNN_CHECKOUT"):
            shutil.rmtree(scratch, ignore_errors=True)

    if failures:
        for failure in failures:
            print(f"gnn conformance: {failure}", file=sys.stderr)
        return 1
    print(
        f"gnn conformance: {pin['version']} at {pin['revision']} "
        f"validated {len(pin['spec_files'])} spec(s) with zero skips"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
