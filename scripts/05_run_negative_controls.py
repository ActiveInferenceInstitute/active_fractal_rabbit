"""Run the fixture negative-control falsification lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.negative_controls import run_negative_controls

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    summary_path = run_negative_controls(ROOT / args.config)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    print(str(summary_path))
    print(
        json.dumps(
            {
                "n_controls": payload["n_controls"],
                "all_passed": payload["all_passed"],
                "controls": [
                    {
                        "name": item["name"],
                        "passed": item["passed"],
                        "probe": item["probe"],
                    }
                    for item in payload["controls"]
                ],
            },
            indent=2,
        )
    )
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
