"""Run the closed-loop candidate-selection sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.closed_loop import run_closed_loop
from active_fractal_rabbit.sweeps import run_replicated_sensitivity

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sweep.yaml")
    args = parser.parse_args()
    results = run_closed_loop(ROOT / args.config)
    summary_artifacts = run_replicated_sensitivity(ROOT / args.config)
    print(
        json.dumps(
            {
                "closed_loop": [
                    {
                        "run_id": item.run_id,
                        "selected_next_candidate": item.selected_next_candidate,
                    }
                    for item in results
                ],
                "sensitivity_summary": {
                    key: str(path) for key, path in summary_artifacts.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
