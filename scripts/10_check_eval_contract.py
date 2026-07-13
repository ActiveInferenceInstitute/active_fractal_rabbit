"""Run the honest-benchmark evaluation contract over the manuscript.

A claim-level linter: it checks that every persistence-skill claim cites a
regime-correct token (G1) and that speed-cue recovery claims disclose the cue
(G2). Exits non-zero on any violation, so the contract can gate a benchmark
submission the same way the negative controls gate the pipeline.
"""

from __future__ import annotations

from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.eval_contract import run_contract

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    report = run_contract(ROOT / "manuscript")
    if report.passed:
        print("eval contract: PASS — all claim-level guards satisfied")
        return 0
    print(f"eval contract: FAIL — {len(report.violations)} violation(s)")
    for violation in report.violations:
        print(f"  [{violation.guard}] {violation.token}: {violation.reason}")
        print(f"    > {violation.sentence}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
