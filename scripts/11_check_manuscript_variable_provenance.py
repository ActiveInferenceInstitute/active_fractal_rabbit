"""Verify manuscript-variable provenance against current artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.manuscript_variables import compute_variables
from active_fractal_rabbit.variable_provenance import (
    DEFAULT_TOKEN_PROVENANCE,
    ProvenanceError,
    verify_token_provenance,
    write_token_provenance,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.project_root)
    try:
        variables = compute_variables(root)
        provenance = write_token_provenance(root, variables)
        report = verify_token_provenance(
            provenance, expected_specs=DEFAULT_TOKEN_PROVENANCE
        )
    except ProvenanceError as exc:
        print(f"manuscript variable provenance: FAIL — {exc}", file=sys.stderr)
        return 1
    print(
        f"manuscript variable provenance: PASS — {len(report['checked'])} token source(s) checked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
