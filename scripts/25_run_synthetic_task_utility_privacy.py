from __future__ import annotations

from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.synthetic_task_utility_privacy import (
    run_and_write_synthetic_task_utility_privacy,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    run_and_write_synthetic_task_utility_privacy(ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
