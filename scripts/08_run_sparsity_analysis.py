"""Run the recovery-versus-sparsity analysis over the checked fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.sparsity import run_sparsity_analysis

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    summary_path = run_sparsity_analysis(ROOT / args.config)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    print(str(summary_path))
    for level in payload["levels"]:
        print(
            f"  fraction={level['fraction']:.2f} reports/traveller={level['mean_reports_per_traveller']:.1f} "
            f"latent_mass={level['latent_mean_mass_on_true_cell']:.3f} "
            f"latent_entropy={level['latent_mean_posterior_entropy']:.3f} "
            f"pymdp_vfe={level['pymdp_latent_mean_vfe']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
