"""Run semantic mini-brief checks over manuscript figures and cover art."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.visual_claim_audit import (
    VisualClaimAuditError,
    write_visual_claim_audit,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        output = write_visual_claim_audit(ROOT)
    except VisualClaimAuditError as exc:
        print(f"visual claim audit: FAIL - {exc}")
        return 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    print(
        "visual claim audit: {status} - {figure_count} figure(s), {brief_count} mini-brief(s)".format(
            status=str(payload["status"]).upper(),
            figure_count=payload["figure_count"],
            brief_count=payload["brief_count"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
