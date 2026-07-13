"""Run the anticipatory gap-filling recovery analysis.

Measures when speed-conditioned latent inference beats the persistence null:
the location-blind limit (persistence is near-unbeatable) and the realistic
gap-filling regime (it wins, and speed-clamped transitions win more). Reads a
run's observations.csv (default: the external run) so it scores the richer
stochastic data; falls back to the checked fixture when that run is absent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.anticipatory import (
    anticipatory_payload,
    run_anticipatory_analysis,
)
from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.observations import discretize_observations
from active_fractal_rabbit.pomdp import build_pomdp

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--observations",
        default="output/runs/fractalrabbit_external/observations.csv",
        help="waypoint CSV to score; falls back to the config fixture if absent",
    )
    parser.add_argument("--gap-fraction", type=float, default=0.5)
    parser.add_argument("--out", default="output/runs/anticipatory_analysis.json")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    obs_path = ROOT / args.observations
    if obs_path.exists():
        source = args.observations
    else:
        assert config.fractalrabbit.fixture_csv is not None
        obs_path = Path(config.fractalrabbit.fixture_csv)
        source = f"fixture:{obs_path.name}"

    sequence = discretize_observations(parse_waypoint_csv(obs_path), config.inference)
    full_model = build_pomdp(sequence, config.inference.smoothing)
    result = run_anticipatory_analysis(
        sequence,
        full_model,
        smoothing=config.inference.smoothing,
        gap_fraction=args.gap_fraction,
        seed=config.seed,
    )
    payload = anticipatory_payload(
        result, source=source, gap_fraction=args.gap_fraction, seed=config.seed
    )
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(str(out_path))
    gf = payload["gap_filling"]
    print(
        f"  gap-filling ({payload['n_gap_steps']} steps): persistence Brier {gf['persistence_brier']:.3f}; "
        f"flat skill {gf['flat_skill_vs_persistence']:+.3f}; SCRP skill {gf['scrp_skill_vs_persistence']:+.3f} "
        f"(speed-clamp edge {gf['speed_clamp_edge']:+.3f})"
    )
    bl = payload["location_blind_limit"]
    print(
        f"  location-blind limit: flat {bl['flat_skill_vs_persistence']:+.3f}, "
        f"SCRP {bl['scrp_skill_vs_persistence']:+.3f} (persistence near-unbeatable)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
