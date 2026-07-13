"""Run the canonical SOTA program: score every candidate, gate the AIF claim honestly.

Thin orchestrator. All business logic lives in
``active_fractal_rabbit.sota_program_harness``: this script only parses arguments, calls the
harness, writes the leaderboard + gate artifacts, and prints the honest verdict.

It builds four canonical synthetic lanes plus a separate diagnostic stressor grid, scores
EVERY candidate (the bundled references, the transparent baseline ladder, switching-noise HMM,
particle filter, pure-numpy neural MLP, and fitted active-inference candidate) with the
leakage-free prequential next-latent-cell loss, and computes the traveller-CLUSTERED bootstrap
CI on the AIF-vs-strongest-fair loss difference per lane. The claim gate clears ONLY if the AIF
candidate beats the strongest FAIR baseline on every canonical lane with the clustered CI
separated from zero; diagnostic stressor rows are stricter synthetic evidence, not a shortcut.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.sota_program_harness import (  # noqa: E402
    DEFAULT_EPS,
    DEFAULT_BOOT_SEED,
    DEFAULT_N_BOOT,
    DEFAULT_N_TRAVELLERS,
    DEFAULT_SPATIAL_BINS,
    run_and_write,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="output/runs")
    parser.add_argument("--n-travellers", type=int, default=DEFAULT_N_TRAVELLERS)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--spatial-bins", type=int, default=DEFAULT_SPATIAL_BINS)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--boot-seed", type=int, default=DEFAULT_BOOT_SEED)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    gate, paths = run_and_write(
        config_path=ROOT / args.config,
        out_dir=out_dir,
        n_travellers=args.n_travellers,
        eps=args.eps,
        spatial_bins=args.spatial_bins,
        n_boot=args.n_boot,
        boot_seed=args.boot_seed,
    )

    print(str(paths["leaderboard"]))
    print(str(paths["gate"]))
    print(str(paths["robustness_grid"]))
    print(str(paths["robustness_diagnostics"]))
    print()
    print(f"AIF candidate under test: {gate.aif_candidate_name}")
    for lane_name, lr in gate.lanes.items():
        verdict = "WINS" if lr.aif_beats_strongest_fair else "does NOT win"
        print(
            f"  [{lane_name}] strongest fair = {lr.strongest_fair_name} "
            f"({lr.strongest_fair_loss:.4f}); AIF = {lr.aif_loss:.4f}; "
            f"diff(fair-aif) mean {lr.mean_difference_nats:+.4f} "
            f"CI=[{lr.clustered_ci_low:+.4f}, {lr.clustered_ci_high:+.4f}] -> AIF {verdict}"
        )
    print()
    print(f"STATUS: {gate.status}  (gate_clears={gate.gate_clears})")
    if not gate.gate_clears:
        print("RESIDUAL GAP (gate stays BLOCKED — honest report):")
        for reason in gate.reasons:
            print(f"  - {reason}")
    else:
        print(
            "Gate CLEARED: AIF beats strongest fair baseline on BOTH lanes, CI-separated."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
