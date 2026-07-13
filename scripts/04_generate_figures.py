"""Generate figures from a pipeline run."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.figures import generate_figures

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="output/runs/default")
    args = parser.parse_args()
    artifacts = generate_figures(ROOT / args.run, ROOT / "output" / "figures")
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
