"""Verify generated artifacts cited by docs/claim_ledger.md exist."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.claim_ledger import verify_claim_ledger_artifacts

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    report = verify_claim_ledger_artifacts(ROOT)
    print(
        json.dumps(
            {"checked": len(report["checked"]), "missing": report["missing"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
