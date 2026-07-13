from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.regime_switch_phase_diagnostic import (  # noqa: E402
    DEFAULT_PHASE_BOOT,
    run_and_write_regime_switch_phase_diagnostic,
)

ROOT = Path(__file__).resolve().parents[1]

DESCRIPTION = (
    "Structural WHY diagnostic for latent_regime_switch_035: re-slices the same "
    "AIF-vs-strongest-fair per-step losses aif_regime_search.py already computes by "
    "movement-mode phase segment (commute/explore/return_stabilization), with a "
    "traveller-clustered bootstrap CI per segment, to show WHERE within the "
    "trajectory the loss to the fair baseline concentrates."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--out-dir", default=str(ROOT / "output" / "runs"))
    parser.add_argument("--n-boot", type=int, default=DEFAULT_PHASE_BOOT)
    args = parser.parse_args()

    out_path = Path(args.out_dir) / "regime_switch_phase_diagnostic.json"
    written = run_and_write_regime_switch_phase_diagnostic(
        args.config, out_path=out_path, n_boot=args.n_boot
    )
    payload = json.loads(written.read_text(encoding="utf-8"))

    print(str(written))
    print()
    print(
        f"regime={payload['regime_id']} best_aif={payload['best_aif_name']} "
        f"strongest_fair={payload['strongest_fair_name']} "
        f"uniform_disadvantage={payload['uniform_disadvantage']} "
        f"worst_to_best_ratio={payload['worst_to_best_magnitude_ratio']}"
    )
    for row in payload["segments"]:
        print(
            f"  [{row['segment']}] n={row['n_steps']} advantage={row['mean_advantage_nats']:.4f} nats "
            f"CI=[{row['clustered_ci_low']:.4f}, {row['clustered_ci_high']:.4f}] "
            f"p_aif_better={row['one_sided_p_aif_better']:.4f}"
        )
    print()
    print(payload["caveat"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
