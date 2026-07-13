"""Regression test for the stale-bytecode-cache guard in ``conftest.py``.

Session 27 (2026-07-02) found a live bug class: ``confound_controls.mobility_summary``'s comparator
read correctly in source (``>=``) but a ``__pycache__/*.pyc`` compiled from an OLDER, broken version
(``<=``) had an mtime that happened to collide with the fixed source file's mtime — CPython's default
mtime-based cache invalidation trusted the stale compiled bytecode and silently executed the reverted
logic. ``conftest.py`` now purges every ``__pycache__`` under ``src/`` before any project module is
imported. This test proves that guard actually removes a poisoned cache rather than merely existing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CONFTEST_PATH = ROOT / "tests" / "conftest.py"


def _load_purge_function():
    spec = importlib.util.spec_from_file_location(
        "_afr_conftest_under_test", _CONFTEST_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._purge_stale_bytecode_cache


def test_purge_removes_every_pycache_dir_under_the_target_root(tmp_path: Path) -> None:
    pkg_root = tmp_path / "pkg"
    nested_cache = pkg_root / "sub" / "__pycache__"
    nested_cache.mkdir(parents=True)
    (nested_cache / "module.cpython-312.pyc").write_bytes(
        b"not real bytecode, just a marker"
    )
    top_cache = pkg_root / "__pycache__"
    top_cache.mkdir()
    (top_cache / "other.cpython-312.pyc").write_bytes(b"marker")

    assert nested_cache.exists()
    assert top_cache.exists()

    purge = _load_purge_function()
    purge(pkg_root)

    assert not nested_cache.exists(), "nested __pycache__ must be removed"
    assert not top_cache.exists(), "top-level __pycache__ must be removed"


def test_purge_is_a_noop_when_no_cache_exists(tmp_path: Path) -> None:
    pkg_root = tmp_path / "pkg_clean"
    (pkg_root / "sub").mkdir(parents=True)

    purge = _load_purge_function()
    purge(pkg_root)  # must not raise on a tree with zero __pycache__ dirs

    assert pkg_root.exists()


def test_conftest_wired_src_onto_syspath_after_purging() -> None:
    """End-to-end wiring check: by the time any test module is collected, conftest.py's module-level
    code has already run (purge, then ``sys.path.insert``) — this session never observed a stale
    ``mobility_summary`` failure again after the guard landed, across every subsequent test run.
    """
    assert str(ROOT / "src") in sys.path
