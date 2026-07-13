"""Pytest configuration for Active Fractal Rabbit."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _purge_stale_bytecode_cache(package_root: Path) -> None:
    """Delete every ``__pycache__`` under ``package_root`` before any project module is imported.

    CPython's default bytecode-cache invalidation is source-mtime-based: a ``.pyc`` is trusted as
    fresh whenever its embedded source mtime matches the ``.py`` file's mtime on disk, to the
    second. A bulk filesystem operation (a restore, a checkout that stamps every file with the same
    timestamp, an rsync with ``-a``) can make an EDITED source file collide with an OLDER compiled
    cache's recorded mtime, and Python will then silently execute the stale bytecode with no error,
    warning, or version mismatch of any kind — a source-level bugfix can be invisibly reverted at
    import time while the file on disk reads correctly. Caught live in session 27 (2026-07-02):
    ``confound_controls.mobility_summary``'s comparator read ``>=`` in the source but disassembled
    to ``<=`` at runtime, both files timestamped to the identical second, causing a real (but
    already-fixed-in-source) test failure that vanished the moment the cache was purged. Since the
    project's coverage/correctness gates trust ``pytest``'s pass/fail as ground truth, a stale cache
    here is a silent-failure class the whole rigor discipline (R8 GENERATOR-CHECK, no-mocks, real
    computation) cannot see through source review alone — it has to be purged, not merely avoided.
    """
    for cache_dir in package_root.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)


_purge_stale_bytecode_cache(SRC)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: Read-only inputs a surface-regeneration test may need, exposed into the
#: sandbox as symlinks. output/ is deliberately NOT linked — each sandbox
#: gets a fresh, empty output tree; configs/ is copied (see _build_sandbox).
_SANDBOX_LINKS = (
    "data",
    "manuscript",
    "docs",
    "pyproject.toml",
    "uv.lock",
    "tasks.yaml",
    "README.md",
    "TODO.md",
    "CITATION.cff",
    "LICENSE",
    "LICENSE-DOCS.md",
    "THIRD_PARTY_NOTICES.md",
)


def fixture_csv() -> Path:
    return ROOT / "data" / "fixtures" / "fractalrabbit_sample.csv"


def _build_sandbox(base: Path) -> Path:
    """Construct an isolated project root under ``base``.

    ``configs/`` is COPIED, not symlinked: ``load_config`` resolves the config
    path (following symlinks) before walking up to the ``pyproject.toml``
    project root, so a symlinked config would silently anchor ``output_dir``
    back at the production repo and defeat the sandbox. The remaining
    read-only inputs are symlinks; ``pyproject.toml`` is linked at the sandbox
    root so ``_find_project_root`` stops there.
    """
    sandbox = base / "repo"
    sandbox.mkdir()
    shutil.copytree(ROOT / "configs", sandbox / "configs")
    for name in _SANDBOX_LINKS:
        source = ROOT / name
        if source.exists():
            (sandbox / name).symlink_to(source)
    (sandbox / "output").mkdir()
    return sandbox


@pytest.fixture
def repo_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated project root for tests that regenerate the artifact surface.

    Surface-regeneration tests run the real generators end-to-end, but several
    of them do so at smoke scale or with fixture-valued rows (variance bands,
    fractal-dimension stub runners, upstream-sensitivity no-ops). Without
    isolation those writes land in the production ``output/`` tree via
    relative or repo-root paths, silently replacing canonical multi-realization
    artifacts with test constants — which then render into the manuscript and
    release manifest unless a full-study regeneration happens before the next
    render. This fixture gives such tests a sandbox root with copied configs,
    symlinked read-only inputs, and a fresh ``output/`` so production
    artifacts are never touched by the suite.
    """
    sandbox = _build_sandbox(tmp_path)
    monkeypatch.chdir(sandbox)
    return sandbox


@pytest.fixture(scope="module")
def repo_sandbox_module(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped repo sandbox for module-scoped surface fixtures.

    Same isolation contract as ``repo_sandbox``; the working directory is
    switched for the duration of the module and restored afterwards.
    """
    sandbox = _build_sandbox(tmp_path_factory.mktemp("repo_sandbox"))
    previous = os.getcwd()
    os.chdir(sandbox)
    yield sandbox
    os.chdir(previous)
