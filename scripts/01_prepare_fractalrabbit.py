"""Prepare a pinned FractalRabbit source cache for external simulator runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import (
    FractalRabbitRunner,
    FractalRabbitUnavailable,
    write_parameter_csv,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    prepared = config.project_root / "output" / "prepared" / "parameters.csv"
    write_parameter_csv(config.fractalrabbit.params, prepared)
    print(f"wrote {prepared}")
    if config.fractalrabbit.execution == "fixture":
        print("fixture execution selected; external FractalRabbit cache build skipped")
        return 0
    try:
        jar = FractalRabbitRunner(config.fractalrabbit).prepare()
    except FractalRabbitUnavailable as exc:
        print(f"FractalRabbit unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"prepared {jar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
