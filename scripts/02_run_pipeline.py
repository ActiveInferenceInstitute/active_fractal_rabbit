"""Run one simulate/observe/infer/score pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.pipeline import run_pipeline_from_path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    result = run_pipeline_from_path(ROOT / args.config)
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "metrics": result.metrics,
                "selected_next_candidate": result.selected_next_candidate,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
