#!/usr/bin/env python3
"""Generate the artifact-bound PDF cover image."""

from __future__ import annotations

from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.cover_art import render_cover_art  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    output = render_cover_art(ROOT)
    print(output.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
