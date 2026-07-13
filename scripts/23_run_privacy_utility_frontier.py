from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.privacy_utility_frontier import (  # noqa: E402
    DEFAULT_GRID_REGIMES,
    run_and_write_privacy_utility_frontier,
)

ROOT = Path(__file__).resolve().parents[1]

DESCRIPTION = (
    "Privacy-utility frontier: surviving next-cell predictive utility (resolution-fair, "
    "shuffle-corrected) paired with re-identification risk across location-minimization regimes. "
    "Defensive, synthetic-only governance diagnostic."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--observations", default="data/fixtures/fractalrabbit_sample_large.csv"
    )
    parser.add_argument("--out", default="output/runs/privacy_utility_frontier.json")
    args = parser.parse_args()

    out = run_and_write_privacy_utility_frontier(
        ROOT / args.config, ROOT / args.observations, ROOT / args.out
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    print(str(out))
    print()
    print(
        f"graceful_degradation_tradeoff={payload['graceful_degradation_tradeoff']}  "
        f"(grids {list(DEFAULT_GRID_REGIMES)})"
    )
    for row in payload["frontier"]:
        print(
            f"  {row['regime']}: excess_utility_bits={row['excess_utility_bits']} "
            f"reid_event_uniqueness={row['reid_event_uniqueness']} "
            f"co_travel={row['co_travel_coincidence']}"
        )
    print()
    print(payload["boundary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
