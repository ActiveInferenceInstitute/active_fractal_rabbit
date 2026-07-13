from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

import json  # noqa: E402

from active_fractal_rabbit.regime_mechanism_synthesis import (  # noqa: E402
    DEFAULT_MECHANISM_BOOT,
    run_and_write_regime_mechanism_synthesis,
)

ROOT = Path(__file__).resolve().parents[1]

DESCRIPTION = (
    "WHERE/HOW MUCH/WHY mechanism map across the AIF regime grid: joins "
    "aif_regime_search.json's per-regime magnitude/CI, multiplicity_ledger.json's "
    "Holm-survival status, and a per-regime run of the confound-control battery "
    "(never previously applied per-regime) into one mechanism verdict per regime."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--out-dir", default=str(ROOT / "output" / "runs"))
    parser.add_argument("--n-boot", type=int, default=DEFAULT_MECHANISM_BOOT)
    args = parser.parse_args()

    out_path = Path(args.out_dir) / "regime_mechanism_synthesis.json"
    written = run_and_write_regime_mechanism_synthesis(
        args.config, out_path=out_path, n_boot=args.n_boot
    )
    payload = json.loads(written.read_text(encoding="utf-8"))

    print(str(written))
    print()
    print(
        f"n_regimes={payload['n_regimes']}  verdict_counts={payload['verdict_counts']}"
    )
    for row in payload["regimes"]:
        print(
            f"  [{row['regime_id']}] verdict={row['mechanism_verdict']} "
            f"advantage={row['mean_advantage_nats']:.4f} nats "
            f"CI=[{row['clustered_ci_low']:.4f}, {row['clustered_ci_high']:.4f}] "
            f"holm_survives={row['holm_survives']} "
            f"clears_confound_battery={row['confound_battery']['clears_confounds']} "
            f"mobile={row['mobility']['is_mobile']}"
        )
    print()
    print(payload["caveat"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
