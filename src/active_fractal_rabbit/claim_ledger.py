"""Sanity checks for generated artifact paths cited by the claim ledger."""

from __future__ import annotations

import re
from pathlib import Path

CLAIM_LEDGER_PATH_RE = re.compile(r"`(output/[^`]+)`")
GENERATED_SUFFIXES = (".csv", ".json", ".png", ".pdf")


def cited_generated_artifacts(claim_ledger_path: str | Path) -> list[str]:
    """Return generated artifact paths cited in ``docs/claim_ledger.md``."""

    text = Path(claim_ledger_path).read_text(encoding="utf-8")
    paths = {
        match.group(1)
        for match in CLAIM_LEDGER_PATH_RE.finditer(text)
        if match.group(1).endswith(GENERATED_SUFFIXES)
    }
    return sorted(paths)


def verify_claim_ledger_artifacts(project_root: str | Path) -> dict[str, list[str]]:
    """Fail if any generated artifact cited by the ledger is absent."""

    root = Path(project_root).resolve()
    paths = cited_generated_artifacts(root / "docs" / "claim_ledger.md")
    missing = [path for path in paths if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(
            "claim ledger cites generated artifacts that are missing: "
            + ", ".join(missing)
        )
    return {"checked": paths, "missing": []}
