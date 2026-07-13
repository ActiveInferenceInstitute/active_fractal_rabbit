from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.aif_regime_search import (  # noqa: E402
    CANONICAL_AIF_REGIMES,
    DEFAULT_AIF_REGIME_N_BOOT,
    DEFAULT_AIF_REGIME_N_TRAVELLERS,
    DEFAULT_AIF_REGIME_REALIZATION_SEEDS,
    run_and_write_aif_regime_search,
)
from active_fractal_rabbit.sota_program_harness import (
    DEFAULT_BOOT_SEED,
    DEFAULT_SPATIAL_BINS,
)  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

DESCRIPTION = "Run the separate regime-qualified synthetic AIF SOTA search."


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="output/runs")
    parser.add_argument(
        "--n-travellers", type=int, default=DEFAULT_AIF_REGIME_N_TRAVELLERS
    )
    parser.add_argument("--n-boot", type=int, default=DEFAULT_AIF_REGIME_N_BOOT)
    parser.add_argument("--boot-seed", type=int, default=DEFAULT_BOOT_SEED)
    parser.add_argument("--spatial-bins", type=int, default=DEFAULT_SPATIAL_BINS)
    parser.add_argument("--regime", action="append", choices=CANONICAL_AIF_REGIMES)
    parser.add_argument("--realization-seed", action="append", type=int)
    args = parser.parse_args()

    regime_ids = tuple(args.regime) if args.regime else CANONICAL_AIF_REGIMES
    realization_seeds = (
        tuple(args.realization_seed)
        if args.realization_seed
        else DEFAULT_AIF_REGIME_REALIZATION_SEEDS
    )
    payload, paths = run_and_write_aif_regime_search(
        config_path=ROOT / args.config,
        out_dir=ROOT / args.out_dir,
        n_travellers=args.n_travellers,
        n_boot=args.n_boot,
        boot_seed=args.boot_seed,
        spatial_bins=args.spatial_bins,
        regime_ids=regime_ids,
        realization_seeds=realization_seeds,
    )

    print(str(paths["leaderboard"]))
    print(str(paths["search"]))
    print(str(paths["gate"]))
    print()
    print(f"STATUS: {payload['status']}  (gate_clears={payload['gate_clears']})")
    for regime_id, summary in payload["regime_summaries"].items():
        print(
            f"  [{regime_id}] {summary['status']}; "
            f"best AIF={','.join(summary['best_aif_candidates'])}; "
            f"strongest fair={','.join(summary['strongest_fair_baselines'])}; "
            f"min advantage={summary['min_advantage_nats']}"
        )
    print()
    print(payload["global_sota_boundary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
