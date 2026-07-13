from __future__ import annotations

from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.privacy_utility_repeats import (
    run_and_write_privacy_utility_repeats,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    run_and_write_privacy_utility_repeats(
        ROOT / "configs" / "default.yaml",
        ROOT / "output" / "runs" / "privacy_utility_frontier_repeats.json",
        project_root=ROOT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
