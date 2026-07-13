"""Run the partial-observability noisy-emission benchmark — the genuine AIF-advantage regime.

Measures the active-inference (soft belief-marginalization) advantage over a point estimate
(hard MAP-commit) as observation noise grows under a known symmetric emission. Fair by
construction: soft and hard differ only in marginalize-vs-commit (same model, emission,
observations, priors). Reads a run's observations.csv (default: the external run) so it scores
the richer stochastic data; falls back to the checked fixture when that run is absent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.partial_observability import (
    run_and_write_partial_observability,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--observations",
        default="output/runs/fractalrabbit_external/observations.csv",
        help="waypoint CSV to score; falls back to the config fixture if absent",
    )
    parser.add_argument(
        "--out", default="output/runs/partial_observability_analysis.json"
    )
    args = parser.parse_args()

    out_path = run_and_write_partial_observability(
        ROOT / args.config, ROOT / args.observations, ROOT / args.out
    )
    print(str(out_path))
    import json

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    print(
        f"  n_cells={payload['n_cells']}  clean advantage {payload['clean_advantage_nats']:+.3f} nats; "
        f"advantage at eps={payload['max_eps']:.2f} is {payload['max_noise_advantage_nats']:+.3f} nats "
        f"(zero-when-clean={payload['zero_advantage_when_clean']}, monotone={payload['advantage_monotone_with_noise']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
