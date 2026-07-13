from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.multiplicity_ledger import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_FDR_Q,
    build_multiplicity_ledger,
    run_and_write_multiplicity_ledger,
)

ROOT = Path(__file__).resolve().parents[1]

DESCRIPTION = (
    "Family-wise (Holm) and false-discovery (Benjamini-Hochberg) correction across the "
    "benchmark's AIF-vs-fair directional comparison family. The family is enumerated "
    "deterministically from live artifacts; nothing is hand-picked."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument(
        "--out-dir", default=None, help="defaults to <root>/output/runs"
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--fdr-q", type=float, default=DEFAULT_FDR_Q)
    args = parser.parse_args()

    root = Path(args.project_root)
    payload = build_multiplicity_ledger(root, alpha=args.alpha, fdr_q=args.fdr_q)
    path = run_and_write_multiplicity_ledger(
        root, out_dir=args.out_dir, alpha=args.alpha, fdr_q=args.fdr_q
    )

    print(str(path))
    print()
    print(
        f"family_size={payload['family_size']}  "
        f"uncorrected_significant={payload['n_uncorrected_significant']}  "
        f"holm_survivors={payload['holm']['n_survivors']}  "
        f"bh_survivors={payload['benjamini_hochberg']['n_survivors']}"
    )
    for row in payload["comparisons"]:
        marker = "SURVIVES-HOLM" if row["holm_survives"] else "."
        print(
            f"  [{row['comparison_id']}] p={row['uncorrected_p']} "
            f"holm={row['holm_adjusted_p']} bh={row['bh_adjusted_p']} {marker}"
        )
    print()
    print(payload["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
