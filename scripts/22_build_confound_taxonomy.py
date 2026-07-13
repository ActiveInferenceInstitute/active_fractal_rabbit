from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.confound_taxonomy import (  # noqa: E402
    build_confound_taxonomy,
    run_and_write_confound_taxonomy,
)

ROOT = Path(__file__).resolve().parents[1]

DESCRIPTION = (
    "Assemble the confound taxonomy: every apparent active-inference advantage this benchmark "
    "raised and dissolved, organised by confound class, each entry re-verified against live "
    "repository state."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument(
        "--out-dir", default=None, help="defaults to <root>/output/runs"
    )
    args = parser.parse_args()

    root = Path(args.project_root)
    payload = build_confound_taxonomy(root)
    path = run_and_write_confound_taxonomy(root, out_dir=args.out_dir)

    print(str(path))
    print()
    print(
        f"confounds={payload['n_confounds']}  classes={payload['n_confound_classes_present']}  "
        f"cross_vendor_caught={payload['n_cross_vendor_caught']}  "
        f"verified={payload['n_verified']}/{payload['n_entries']}"
    )
    for row in payload["entries"]:
        flag = "ok" if row["verified"] else "UNVERIFIED"
        print(
            f"  [{row['entry_id']}] {row['confound_class']} / {row['status']} "
            f"({row['verification_kind']}: {flag})"
        )
    print()
    print(payload["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
