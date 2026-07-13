"""Run manuscript claim inventory and claim-ledger coverage audit."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.claim_audit import ClaimAuditError, write_claim_audit

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        output = write_claim_audit(ROOT)
    except ClaimAuditError as exc:
        print(f"claim audit: FAIL - {exc}")
        return 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    print(
        "claim audit: {status} - {claim_count} claim(s), {ledger_rows} ledger row(s), {families} family gate(s)".format(
            status=str(payload["status"]).upper(),
            claim_count=payload["claim_count"],
            ledger_rows=payload["ledger_row_count"],
            families=len(payload["families"]),
        )
    )
    # Surface the high-risk oracle blind spot: high-risk claims that carry no
    # supporting category (artifact/citation/numeric/formal/limitation/roadmap).
    # The gate verdict stays green, but these are printed and listed so the
    # coverage gap is visible for human triage instead of hiding behind a count.
    unsupported = payload.get("unsupported_high_risk_claims", [])
    count = payload.get("unsupported_high_risk_count", len(unsupported))
    print(
        f"claim audit: unsupported high-risk claim(s) with NO supporting category: {count}"
    )
    for record in unsupported:
        families = ", ".join(record.get("risk_families", []))
        text = " ".join(str(record.get("text", "")).split())
        if len(text) > 160:
            text = text[:157] + "..."
        print(f"  - [{record.get('location', '?')}] ({families}) {text}")
    # Production numeric-content mismatches are fail-closed by ``write_claim_audit``;
    # retain the explicit count so the failing evidence is visible in CI logs.
    numeric_warnings = payload.get("numeric_content_warnings", [])
    print(f"claim audit: numeric-content mismatches: {len(numeric_warnings)}")
    for warning in numeric_warnings:
        print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
