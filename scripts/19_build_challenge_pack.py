"""Build the synthetic open challenge-pack artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.challenge_pack import build_open_challenge_pack

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="output/runs/open_challenge_pack.json")
    args = parser.parse_args()
    out = build_open_challenge_pack(ROOT, ROOT / args.out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
