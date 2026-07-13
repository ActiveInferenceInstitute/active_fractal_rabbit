"""Print project and optional FractalRabbit toolchain readiness."""

from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.preflight import print_preflight

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(print_preflight())
