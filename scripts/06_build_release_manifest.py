"""Build (and optionally verify) the checksummed release bundle manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.release import (
    build_release_manifest,
    verify_release_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the existing manifest instead of rebuilding",
    )
    args = parser.parse_args()
    manifest_path = ROOT / "output" / "release_manifest.json"
    if not args.verify_only:
        manifest_path = build_release_manifest(ROOT)
        print(str(manifest_path))
    report = verify_release_manifest(manifest_path)
    print(json.dumps(report, indent=2))
    return 0 if all(not values for values in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
