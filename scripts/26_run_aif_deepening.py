from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.aif_deepening import run_and_write_aif_deepening_candidates

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize source-bound AIF diagnostic evidence.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    print(run_and_write_aif_deepening_candidates(
        args.project_root, args.out, runs_dir=args.runs_dir
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
