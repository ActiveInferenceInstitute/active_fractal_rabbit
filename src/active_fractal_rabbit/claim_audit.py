"""RedTeam-style manuscript claim inventory and ledger coverage checks.

This module is intentionally static. It does not decide whether a result is
true; it checks whether reader-facing manuscript claims are covered by the
project's evidence ledger and whether high-risk framing remains bounded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .eval_contract import _any_marker_at_word_boundary

CLAIM_AUDIT_SCHEMA = "active-fractal-rabbit-claim-audit-v1"

_TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
_CITATION_RE = re.compile(r"@([A-Za-z][A-Za-z0-9_:-]*)")
_XREF_RE = re.compile(r"\[@(fig|eq|tbl|sec):([A-Za-z0-9_-]+)\]")
_BACKTICK_PATH_RE = re.compile(r"`([^`]+)`")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "et al.",
    "vs.",
    "cf.",
    "Fig.",
    "Eq.",
    "etc.",
    "Dr.",
    "No.",
    "Inc.",
    "approx.",
)
_ABBREV_SENTINEL = "\x00"

_MANUSCRIPT_EXCLUDED = {
    "AGENTS.md",
    "README.md",
    "SYNTAX.md",
    "99_references.md",
    "preamble.md",
}

_BOUNDARY_MARKERS = (
    "not ",
    "no ",
    "never",
    "without",
    "rather than",
    "does not",
    "do not",
    "cannot",
    "limitation",
    "boundary",
    "out of scope",
    "synthetic",
    "empirical protocol",
    "requires",
    "require",
    "oversight",
    "minimization",
    "null",
)

_NATIONAL_SECURITY_GUARDS = (
    "responsible",
    "uncertainty",
    "minimization",
    "integrity",
    "oversight",
    "synthetic",
    "pre-empirical",
    "not as a surveillance",
    "without making surveillance claims",
)


@dataclass(frozen=True)
class ClaimFamily:
    """One high-risk claim family that must be represented in the ledger."""

    family: str
    manuscript_markers: tuple[str, ...]
    ledger_markers: tuple[str, ...]
    required_ledger_terms: tuple[str, ...]
    required_boundary_terms: tuple[str, ...] = ()


CLAIM_FAMILIES: tuple[ClaimFamily, ...] = (
    ClaimFamily(
        "active_inference_pymdp_performance",
        (
            "pymdp",
            "state-of-the-art",
            "performance",
            "expected free energy",
            "variational free energy",
        ),
        ("pymdp", "performance", "expected free energy", "variational free energy"),
        (
            "pymdp_benchmark",
            "tests/test_pymdp",
            "output/runs/pymdp_benchmark_matrix.json",
        ),
        ("not", "gate", "interpretability", "matched priors"),
    ),
    ClaimFamily(
        "efe_tasking_and_collection_boundary",
        (
            "expected free energy",
            "tasking",
            "collection",
            "mission-conditioned",
            "sensing",
        ),
        ("expected free energy", "mission-conditioned", "tasking", "active sensing"),
        (
            "active_sensing",
            "mission_collection",
            "tests/test_active_sensing",
            "tests/test_mission_collection",
        ),
        (
            "not operational",
            "recommendation",
            "not a collection",
            "not operational tasking",
        ),
    ),
    ClaimFamily(
        "minimization_privacy_linkage",
        (
            "minimization",
            "privacy",
            "linkage",
            "re-identification",
            "unicity",
            "co-travel",
        ),
        ("minimization", "privacy", "linkage", "unicity", "co-travel"),
        (
            "privacy_frontier",
            "linkage_risk_frontier",
            "demontjoye2013unique",
            "dwork2006differential_privacy",
        ),
        ("synthetic", "not", "oversight", "real"),
    ),
    ClaimFamily(
        "poisoning_integrity_boundary",
        ("poisoning", "integrity", "tampered", "corrupted", "vfe surprise"),
        ("poisoning", "integrity", "vfe surprise", "hardened anomaly"),
        ("poisoning_robustness", "hardened_anomaly", "tests/test_poisoning_robustness"),
        ("synthetic", "not", "defensive", "null"),
    ),
    ClaimFamily(
        "external_lane_evidence",
        (
            "external lane",
            "external stochastic",
            "full stochastic simulator",
            "pinned simulator build",
        ),
        ("external", "stochastic", "simulator"),
        (
            "fractalrabbit_external",
            "tests/test_fractalrabbit_external",
            "sensitivity_summary_external",
        ),
        ("synthetic", "not empirical", "unavailable"),
    ),
    ClaimFamily(
        "gap_filling_persistence_null",
        ("gap-filling", "gap filling", "persistence", "oracle null"),
        ("gap-filling", "persistence", "oracle"),
        ("anticipatory_analysis", "tests/test_anticipatory", "GAP_FILL"),
        ("null", "oracle", "does not beat", "bounded"),
    ),
    ClaimFamily(
        "factorial_latent_mode_interpretability",
        ("factorial", "latent-mode", "movement mode", "reporting mode", "mean-field"),
        ("factorial", "latent-mode", "mean-field"),
        (
            "factorial_pymdp",
            "tests/test_factorial_pymdp",
            "jordan1999variational_methods",
        ),
        ("not", "synthetic", "interpretability", "performance"),
    ),
    ClaimFamily(
        "empirical_validation_boundary",
        (
            "real waypoint",
            "real mobility",
            "empirical validation",
            "operational",
            "surveillance",
        ),
        ("empirical", "real mobility", "operational", "surveillance"),
        (
            "docs/empirical_validation_protocol.md",
            "demontjoye2013unique",
            "kondor2018trace_matching",
        ),
        ("not", "requires", "separate", "synthetic"),
    ),
    ClaimFamily(
        "governance_policy_boundary",
        (
            "governance",
            "commercially available",
            "automation bias",
            "sensitive location",
            "policy relevance",
            "responsible interpretation",
            "human oversight",
        ),
        (
            "governance",
            "commercially available",
            "automation bias",
            "sensitive location",
            "policy relevance",
            "responsible interpretation",
            "human oversight",
        ),
        (
            "haggerty2000surveillant_assemblage",
            "horowitz2024automation_bias",
            "nist2023ai_rmf",
            "nist2024csf",
            "dod2021responsible_ai",
            "odni2022cai_report",
            "odni2024cai_framework",
            "ftc2024xmode_outlogic",
            "ftc2026gm_onstar",
        ),
        (
            "synthetic",
            "not operational",
            "non-operational",
            "boundary",
            "conceptual",
            "not a",
        ),
    ),
)


class ClaimAuditError(RuntimeError):
    """Raised when claim-audit violations are present."""


def audit_claims(
    project_root: str | Path, *, strict_numeric: bool = True
) -> dict[str, Any]:
    """Return the manuscript claim inventory and deterministic audit results.

    ``strict_numeric=False`` exists only for deliberately reduced-scale sandbox
    fixtures whose generated values are not intended to certify a release.
    Production and release callers use the fail-closed default.
    """

    root = Path(project_root).resolve()
    ledger_rows = _parse_claim_ledger(root / "docs" / "claim_ledger.md")
    records = _extract_manuscript_claims(root)
    violations: list[str] = []
    family_rows: list[dict[str, Any]] = []

    ledger_blob = "\n".join(
        f"{row['claim']} {row['evidence_type']} {row['source']}" for row in ledger_rows
    ).lower()
    manuscript_blob = "\n".join(record["text"] for record in records).lower()
    violations.extend(_lane_provenance_violations(root))

    for family in CLAIM_FAMILIES:
        manuscript_hits = [
            record
            for record in records
            if any(
                marker in record["text"].lower() for marker in family.manuscript_markers
            )
        ]
        ledger_hits = [
            row
            for row in ledger_rows
            if any(
                marker
                in f"{row['claim']} {row['evidence_type']} {row['source']}".lower()
                for marker in family.ledger_markers
            )
        ]
        missing_terms = [
            term
            for term in family.required_ledger_terms
            if term.lower() not in ledger_blob
        ]
        boundary_hits = [
            record
            for record in manuscript_hits
            if any(
                term in record["text"].lower()
                for term in family.required_boundary_terms
            )
        ]
        if not manuscript_hits:
            violations.append(f"{family.family}: no manuscript claim family hit")
        if not ledger_hits:
            violations.append(f"{family.family}: no claim-ledger row covers family")
        if missing_terms:
            violations.append(
                f"{family.family}: claim ledger missing required support term(s): {', '.join(missing_terms)}"
            )
        if family.required_boundary_terms and manuscript_hits and not boundary_hits:
            violations.append(
                f"{family.family}: manuscript family lacks required boundary framing"
            )
        family_rows.append(
            {
                "family": family.family,
                "manuscript_hit_count": len(manuscript_hits),
                "ledger_hit_count": len(ledger_hits),
                "missing_required_ledger_terms": missing_terms,
                "boundary_hit_count": len(boundary_hits),
            }
        )

    if "national-security" in manuscript_blob or "national security" in manuscript_blob:
        ns_records = [
            record
            for record in records
            if "national-security" in record["text"].lower()
            or "national security" in record["text"].lower()
        ]
        for record in ns_records:
            lower = record["text"].lower()
            if not any(marker in lower for marker in _NATIONAL_SECURITY_GUARDS):
                violations.append(
                    f"{record['location']}: national-security framing lacks defensive guardrail wording"
                )

    unsupported_high_risk = [
        record
        for record in records
        if record["risk_families"]
        and not set(record["categories"]).intersection(
            {
                "numeric_provenance",
                "artifact_backed",
                "citation_backed",
                "formal_reference",
                "limitation",
                "roadmap",
            }
        )
    ]

    # Surface the identities of the unsupported high-risk claims, not just a
    # bare count. The count alone was decorative: it was computed but never made
    # inspectable, so a reviewer could not see WHICH high-risk claims carry no
    # supporting category (artifact/citation/numeric/formal/limitation/roadmap).
    # This listing is deliberately load-bearing for human review even though the
    # gate verdict below stays derived from ``violations`` (an unsupported
    # high-risk claim is a coverage gap to triage, not an automatic hard fail).
    unsupported_high_risk_claims = [
        {
            "location": record["location"],
            "section": record["section"],
            "risk_families": record["risk_families"],
            "categories": record["categories"],
            "text": record["text"],
        }
        for record in unsupported_high_risk
    ]

    path_report = _verify_ledger_local_paths(root, ledger_rows)
    violations.extend(path_report["violations"])

    numeric_report = _verify_ledger_numeric_content(root, ledger_rows)
    if strict_numeric:
        violations.extend(numeric_report["violations"])

    payload: dict[str, Any] = {
        "schema": CLAIM_AUDIT_SCHEMA,
        "status": "pass" if not violations else "fail",
        "claim_count": len(records),
        "ledger_row_count": len(ledger_rows),
        "families": family_rows,
        "category_counts": _category_counts(records),
        "unsupported_high_risk_count": len(unsupported_high_risk),
        "unsupported_high_risk_claims": unsupported_high_risk_claims,
        "ledger_path_checks": path_report["checked"],
        "ledger_numeric_content_checks": numeric_report[
            "claims_checked_for_numeric_content"
        ],
        "numeric_content_warnings": numeric_report["violations"],
        "strict_numeric": strict_numeric,
        "violations": violations,
        "claims": records,
    }
    return payload


def write_claim_audit(
    project_root: str | Path,
    *,
    out_path: str | Path = "output/runs/claim_audit.json",
    fail_on_violations: bool = True,
    strict_numeric: bool = True,
) -> Path:
    """Write the claim audit payload and optionally fail on violations."""

    root = Path(project_root).resolve()
    payload = audit_claims(root, strict_numeric=strict_numeric)
    output = root / out_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if fail_on_violations and payload["violations"]:
        raise ClaimAuditError("; ".join(payload["violations"]))
    return output


def _lane_provenance_violations(root: Path) -> list[str]:
    """Reject wording that launders fixture/local-generator results as upstream runs."""

    abstract = root / "manuscript" / "00_abstract.md"
    if not abstract.is_file():
        return ["missing manuscript/00_abstract.md for lane-provenance audit"]
    text = " ".join(abstract.read_text(encoding="utf-8").lower().split())
    upstream_misattributions = (
        "synthetic fractalrabbit data generated by open-source software from the national security agency",
        "headline lane uses the upstream fractalrabbit simulator",
        "aif advantage was generated by the upstream fractalrabbit simulator",
    )
    return [
        "abstract attributes fixture/local-generator evidence to the upstream FractalRabbit simulator"
        for phrase in upstream_misattributions
        if phrase in text
    ]


def _extract_manuscript_claims(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((root / "manuscript").glob("*.md")):
        if path.name in _MANUSCRIPT_EXCLUDED:
            continue
        text = path.read_text(encoding="utf-8")
        section = ""
        for line_no, sentence in _sentence_spans(text):
            stripped = sentence.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                section = stripped.lstrip("#").split("{#", 1)[0].strip()
                continue
            if stripped.startswith("|") or stripped.startswith(": "):
                continue
            if not _is_claim_like(stripped):
                continue
            categories = _classify_sentence(stripped)
            risks = _risk_families(stripped)
            records.append(
                {
                    "location": f"{path.relative_to(root)}:{line_no}",
                    "section": section,
                    "text": stripped,
                    "categories": categories,
                    "tokens": sorted(set(_TOKEN_RE.findall(stripped))),
                    "citations": sorted(
                        {
                            key
                            for key in _CITATION_RE.findall(stripped)
                            if not key.startswith(("fig:", "eq:", "tbl:", "sec:"))
                        }
                    ),
                    "xrefs": sorted(
                        {f"{kind}:{name}" for kind, name in _XREF_RE.findall(stripped)}
                    ),
                    "risk_families": risks,
                }
            )
    return records


def _sentence_spans(text: str) -> list[tuple[int, str]]:
    protected = text
    for abbrev in _ABBREVIATIONS:
        protected = protected.replace(abbrev, abbrev.replace(".", _ABBREV_SENTINEL))
    output: list[tuple[int, str]] = []
    cursor = 0
    for part in _SENTENCE_RE.split(protected):
        restored = part.replace(_ABBREV_SENTINEL, ".")
        if restored.strip():
            line_no = text.count("\n", 0, cursor) + 1
            output.append((line_no, restored))
        cursor += len(part) + 1
    return output


def _is_claim_like(sentence: str) -> bool:
    lower = sentence.lower()
    return (
        bool(_TOKEN_RE.search(sentence))
        or bool(_XREF_RE.search(sentence))
        or bool(_CITATION_RE.search(sentence))
        or "output/" in lower
        or any(
            marker in lower
            for family in CLAIM_FAMILIES
            for marker in family.manuscript_markers
        )
    )


def _classify_sentence(sentence: str) -> list[str]:
    lower = sentence.lower()
    categories: list[str] = []
    if _TOKEN_RE.search(sentence):
        categories.append("numeric_provenance")
    if "output/" in lower or "[@fig:" in lower or "[@tbl:" in lower:
        categories.append("artifact_backed")
    if _CITATION_RE.search(sentence):
        categories.append("citation_backed")
    if "[@eq:" in lower or "[@sec:" in lower:
        categories.append("formal_reference")
    if _any_marker_at_word_boundary(lower, _BOUNDARY_MARKERS):
        categories.append("limitation")
    if any(
        marker in lower
        for marker in ("future", "roadmap", "remaining", "gate", "verdict", "status")
    ):
        categories.append("roadmap")
    return categories or ["unsupported"]


def _risk_families(sentence: str) -> list[str]:
    lower = sentence.lower()
    return [
        family.family
        for family in CLAIM_FAMILIES
        if any(marker in lower for marker in family.manuscript_markers)
    ]


def _parse_claim_ledger(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] in {"claim", "---"} or set(cells[0]) == {"-"}:
            continue
        rows.append({"claim": cells[0], "evidence_type": cells[1], "source": cells[2]})
    return rows


def _verify_ledger_local_paths(
    root: Path, ledger_rows: list[dict[str, str]]
) -> dict[str, Any]:
    checked: list[str] = []
    violations: list[str] = []
    for row in ledger_rows:
        for raw in _BACKTICK_PATH_RE.findall(row["source"]):
            candidate = raw.strip()
            if "::" in candidate:
                candidate = candidate.split("::", 1)[0]
            if "*" in candidate or " " in candidate:
                continue
            if not any(
                candidate.startswith(prefix)
                for prefix in (
                    "configs/",
                    "data/",
                    "docs/",
                    "manuscript/",
                    "output/",
                    "scripts/",
                    "src/",
                    "tests/",
                )
            ):
                continue
            checked.append(candidate)
            if not (root / candidate).exists():
                violations.append(f"claim ledger cites missing local path: {candidate}")
    return {"checked": sorted(set(checked)), "violations": violations}


#: Matches decimals only (requires a literal ``.``) in claim prose - deliberately
#: excludes plain integers, which turned out on the real ledger to be dominated by
#: derived counts ("51" manuscript figures, i.e. ``len(registry)``, not a stored
#: field) and external constants ("JDK 21") rather than artifact-stored results, so
#: checking them produced false positives against a currently-honest ledger. A
#: decimal like "0.077" is much more likely to be an actual measured value copied
#: from an artifact field.
_CLAIM_NUMBER_RE = re.compile(r"(?<![\w.:/-])(-?\d+\.\d+)(?![\w-])")


def _extract_claim_numbers(text: str) -> list[float]:
    """Best-effort extraction of literal numeric results asserted in claim prose."""

    numbers: list[float] = []
    for match in _CLAIM_NUMBER_RE.finditer(text):
        try:
            numbers.append(float(match.group(1)))
        except ValueError:  # pragma: no cover - regex guarantees a parseable numeral
            continue
    return numbers


def _flatten_json_numbers(payload: Any) -> set[float]:
    """Recursively collect every numeric leaf value in a JSON structure, rounded to
    6 decimals so float-noise doesn't prevent a legitimate match."""

    numbers: set[float] = set()
    if isinstance(payload, bool):
        return numbers
    if isinstance(payload, (int, float)):
        numbers.add(round(float(payload), 6))
    elif isinstance(payload, dict):
        for value in payload.values():
            numbers |= _flatten_json_numbers(value)
    elif isinstance(payload, list):
        for value in payload:
            numbers |= _flatten_json_numbers(value)
    return numbers


def _verify_ledger_numeric_content(
    root: Path, ledger_rows: list[dict[str, str]]
) -> dict[str, Any]:
    """Best-effort content check that ``_verify_ledger_local_paths`` cannot do: every
    literal number a ledger claim asserts should be findable (within tolerance)
    somewhere in the JSON artifact(s) it cites, not merely a path that exists on disk.

    This does not catch purely qualitative overclaims (a "substantially" vs
    "marginally" mismatch with no numeral in the claim text) - that class needs a
    richer per-row field-path schema this ledger does not carry. It does catch a
    claim that states a specific number absent from every artifact it cites, which
    the path-existence-only check above is structurally blind to.
    """

    claims_checked = 0
    violations: list[str] = []
    for row in ledger_rows:
        claim_numbers = _extract_claim_numbers(row["claim"])
        if not claim_numbers:
            continue
        json_paths = []
        for raw in _BACKTICK_PATH_RE.findall(row["source"]):
            candidate = raw.strip().split("::", 1)[0]
            if candidate.endswith(".json") and (root / candidate).is_file():
                json_paths.append(candidate)
        if not json_paths:
            continue
        claims_checked += 1
        artifact_numbers: set[float] = set()
        for json_path in json_paths:
            try:
                payload = json.loads((root / json_path).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            artifact_numbers |= _flatten_json_numbers(payload)
        for number in claim_numbers:
            tolerance = max(1e-3, abs(number) * 0.01)
            if not any(
                abs(number - candidate) < tolerance for candidate in artifact_numbers
            ):
                violations.append(
                    f"claim ledger row {row['claim']!r} states {number!r}, not found in any "
                    f"cited artifact ({', '.join(json_paths)})"
                )
    return {
        "claims_checked_for_numeric_content": claims_checked,
        "violations": violations,
    }


def _category_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for category in record["categories"]:
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))
