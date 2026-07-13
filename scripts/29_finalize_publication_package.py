"""Finalize and verify the publication package after sibling-template rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from active_fractal_rabbit.render_freshness import (
    verify_render_freshness,
    write_render_freshness,
)
from active_fractal_rabbit.release import (
    build_release_manifest,
    verify_release_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _render_report(root: Path, *, write: bool) -> dict[str, list[str]]:
    manifest = root / "output" / "render_manifest.json"
    if write:
        write_render_freshness(root)
    return verify_render_freshness(manifest)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify rendered PDF/web/cover freshness and the final release manifest."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing manifests without rebuilding either one",
    )
    args = parser.parse_args()

    root = PROJECT_ROOT.resolve()
    render_report = _render_report(root, write=not args.verify_only)
    if args.verify_only:
        release_manifest = root / "output" / "release_manifest.json"
        if not release_manifest.is_file():
            release_report = {
                "missing": [str(release_manifest.relative_to(root))],
                "mismatched": [],
                "unhashed_required": [],
                "missing_manifest_required": [],
                "omitted_entries": [],
                "invalid_manifest": [],
                "failed_checks": [],
            }
        else:
            release_report = verify_release_manifest(release_manifest)
    else:
        release_manifest = build_release_manifest(root)
        release_report = verify_release_manifest(release_manifest)
    payload = {
        "render_manifest": str((root / "output" / "render_manifest.json").relative_to(root)),
        "render_report": render_report,
        "release_manifest": str(release_manifest.relative_to(root)),
        "release_report": release_report,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    render_failed = any(render_report.values())
    release_failed = any(release_report.values())
    return 1 if render_failed or release_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
