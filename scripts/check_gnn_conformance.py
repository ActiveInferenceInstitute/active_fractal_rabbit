"""Run the pinned GNN conformance check with the shared reader.

The gate is the one recorded in ``.github/gnn-pin.json``:

1. materialize a checkout whose ``HEAD`` is exactly the recorded revision,
2. install that checkout's parser into its own virtualenv,
3. run the shared reader from that environment::

       python -m gnn.importers check --repo . --require-pointer \\
           --workflow .github/workflows/ci.yml --job gnn-conformance

   The reader owns pin discovery and shape, the ``.github/GNN.md`` pointer,
   and the syntax validation of every ``*.gnn.md`` specification the pin
   names.
4. run ``tests/test_gnn_spec.py`` against that checkout and require a
   nonempty suite with zero skips, failures, or errors. Those tests check the
   model-declaration semantics the reader does not.

Only the revision key is read from the pin here: the pinned parser has to be
materialized before the reader that validates the pin can run. Every other
pin question belongs to ``gnn.importers``. A missing checkout, a revision
mismatch, an unimportable parser, a reader finding, or a skipped parser test
is a gate failure, never a silent pass.

Set ``GNN_CHECKOUT`` to a prepared checkout at the pinned revision to run
offline; otherwise the gate clones the pinned revision into ``GNN_SCRATCH``
(or a temporary directory).

Exit codes:
    0: the shared reader and the parser tests passed
    1: the gate ran and failed
    2: the pin file is missing or unreadable
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
UPSTREAM = "https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation.git"
WORKFLOW = ".github/workflows/ci.yml"
JOB = "gnn-conformance"
CLONE_ATTEMPTS = 3


class PinError(RuntimeError):
    """The recorded pin cannot be used as a gate."""


def read_revision(path: Path = PIN_PATH) -> str:
    """Return the revision the environment must materialize.

    This is the environment's input, not a second reader: the installed parser
    cannot validate a pin before it exists, so one key is taken from the file
    and the shared reader judges everything else about it afterwards.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PinError(f"pin file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise PinError(f"pin file is not valid JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PinError(f"pin file must contain a JSON object: {path}")
    revision = payload.get("commit") or payload.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise PinError(
            f"pin records no 40-character commit/revision: {revision!r}"
        )
    return revision


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


def verify_checkout(checkout: Path, revision: str) -> None:
    """Require the checkout to stand exactly at the pinned revision."""

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


def clone_pin(revision: str, scratch: Path) -> Path:
    """Clone the pinned revision into *scratch* and verify the checkout."""

    checkout = scratch / "gnn"
    for attempt in range(1, CLONE_ATTEMPTS + 1):
        shutil.rmtree(checkout, ignore_errors=True)
        clone = subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--filter=blob:none",
                "--no-checkout",
                "--",
                UPSTREAM,
                str(checkout),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=600,
        )
        fetched = False
        if clone.returncode == 0 and (checkout / ".git").exists():
            fetch = _git(checkout, "fetch", "--quiet", "--depth", "1", "origin", revision)
            if fetch.returncode == 0:
                result = _git(checkout, "checkout", "--quiet", revision)
                fetched = result.returncode == 0 and _head(checkout) == revision
        if fetched:
            return checkout
        print(
            f"attempt {attempt}: pinned GNN revision {revision} could not be materialized",
            file=sys.stderr,
        )
        time.sleep(attempt * 5)
    raise PinError(
        f"pinned GNN revision {revision} could not be resolved after "
        f"{CLONE_ATTEMPTS} attempts"
    )


def _required_uv(checkout: Path) -> str | None:
    """Return the uv version constraint the pinned checkout records, if any."""

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
            return value
    return None


def _sync_command(checkout: Path) -> list[str]:
    """Build the frozen sync command for the checkout's own environment."""

    required = _required_uv(checkout)
    # An exact pin needs its own uv; a range is satisfied by the host's uv.
    if required and required.startswith("=="):
        prefix = ["uvx", "--from", f"uv=={required[2:]}", "uv"]
    else:
        prefix = ["uv"]
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


def ensure_parser(checkout: Path) -> Path:
    """Return the checkout's interpreter, installing the parser when needed."""

    python = checkout / ".venv" / "bin" / "python"

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
    return python


def run_reader(python: Path) -> tuple[list[str], int]:
    """Run the shared reader and return its failure messages and spec count."""

    command = [
        str(python),
        "-B",
        "-m",
        "gnn.importers",
        "check",
        "--repo",
        ".",
        "--require-pointer",
        "--workflow",
        WORKFLOW,
        "--job",
        JOB,
        "--json",
    ]
    result = subprocess.run(
        command,
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=600,
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        report = None
    if not isinstance(report, dict):
        output = (result.stdout + result.stderr).strip()
        return [f"the shared reader produced no JSON report: {output[-2000:]}"], 0

    pin = report.get("pin") or {}
    specs = report.get("specs") or {}
    count = int(specs.get("count", 0) or 0)
    pointer = report.get("pointer") or {}
    print(f"importer: {report.get('repo')}")
    if pin:
        party = f" [{pin['party']}]" if pin.get("party") else ""
        print(
            f"  pin: {pin.get('path')}{party} -> {pin.get('tag') or '(no tag)'} "
            f"@ {str(pin.get('commit', ''))[:10]} "
            f"(schema {pin.get('schema') or 'unversioned'})"
        )
    else:
        print("  pin: none")
    print(f"  specs: {specs.get('count', 0)}")
    if pointer:
        print(
            f"  pointer: {pointer.get('path')} -> "
            f"{'ok' if pointer.get('ok') else 'FAILED'}"
        )
    for finding in report.get("findings", []):
        print(
            f"  [{finding.get('level')}] {finding.get('code')}: "
            f"{finding.get('message')}"
        )
    print(f"  verdict: {'ok' if report.get('ok') else 'FAILED'}")

    failures: list[str] = []
    if result.returncode != 0 or not report.get("ok"):
        errors = [
            finding for finding in report.get("findings", []) if finding.get("level") == "error"
        ]
        if errors:
            failures.extend(
                f"{finding.get('code')}: {finding.get('message')}" for finding in errors
            )
        else:
            failures.append("the shared reader reported a failure")
    return failures, count


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
        revision = read_revision()
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
            verify_checkout(checkout, revision)
        else:
            checkout = clone_pin(revision, scratch)
        python = ensure_parser(checkout)
        failures, spec_count = run_reader(python)
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
        f"gnn conformance: {revision[:12]} validated {spec_count} spec(s) "
        "with zero pointer findings and zero skips"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
