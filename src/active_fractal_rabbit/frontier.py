"""Recoverability / minimization frontier: what each modality subset recovers.

The oversight-facing capstone. For each subset of observation modalities, it
measures how much latent structure is recoverable — mean posterior mass on the
true cell (point recovery) and total-variation distance of the inferred occupancy
to the truth (structure recovery). The result is a principled map of what CAN and
CANNOT be recovered from a given modality set, which informs data-minimization
decisions: e.g., how much pattern-of-life survives when location is withheld and
only time/speed/burst metadata remain.

This is an oversight-positive contribution — it quantifies limits — not a
surveillance how-to. Synthetic-data software measurement on the pinned
FractalRabbit generator; recoverability of the generator's own planted latent
structure is not evidence about any real population.
"""

from __future__ import annotations

import numpy as np

from .inference import run_latent_inference
from .pomdp import build_pomdp
from .structure_recovery import total_variation, true_occupancy
from .types import ObservationSequence

# Modality subsets spanning full-location to metadata-only.
MODALITY_SUBSETS: dict[str, tuple[str, ...]] = {
    "all": ("location", "time_of_day", "speed", "report_burst"),
    "no_location": ("time_of_day", "speed", "report_burst"),
    "metadata_time_burst": ("time_of_day", "report_burst"),
    "speed_only": ("speed",),
    "time_only": ("time_of_day",),
    "burst_only": ("report_burst",),
}


def _mean_mass_on_true_cell(posterior: np.ndarray, cells: np.ndarray) -> float:
    cells = np.asarray(cells, dtype=int)
    return float(np.mean(posterior[np.arange(posterior.shape[0]), cells]))


def recoverability_frontier(
    sequence: ObservationSequence, *, smoothing: float
) -> list[dict[str, object]]:
    """Per-modality-subset recoverability: point recovery + occupancy structure."""

    cells = np.asarray(sequence.spatial_states, dtype=int)
    n_states = sequence.num_spatial_states
    occ_true = true_occupancy(cells, n_states)
    rows: list[dict[str, object]] = []
    for name, modalities in MODALITY_SUBSETS.items():
        model = build_pomdp(sequence, smoothing, modalities=modalities)
        posterior = run_latent_inference(model, sequence).posterior_states
        rows.append(
            {
                "modalities": name,
                "modality_list": list(modalities),
                "mean_mass_on_true_cell": _mean_mass_on_true_cell(posterior, cells),
                "occupancy_tv": total_variation(posterior.mean(axis=0), occ_true),
            }
        )
    return rows


FRONTIER_SCHEMA = "active-fractal-rabbit-frontier-v1"


def run_and_write_frontier(config_path, observations_path, out_path):
    import json
    from pathlib import Path

    from .config import load_config
    from .fractalrabbit import parse_waypoint_csv
    from .observations import discretize_observations

    config = load_config(config_path)
    obs = Path(observations_path)
    if obs.exists():
        source = str(observations_path)
    else:
        assert config.fractalrabbit.fixture_csv is not None
        obs = Path(config.fractalrabbit.fixture_csv)
        source = f"fixture:{obs.name}"
    sequence = discretize_observations(parse_waypoint_csv(obs), config.inference)
    rows = recoverability_frontier(sequence, smoothing=config.inference.smoothing)
    payload = {
        "schema": FRONTIER_SCHEMA,
        "source": source,
        "frontier": rows,
        "caveat": (
            "Recoverability of the synthetic generator's own latent cell and occupancy structure "
            "as a function of the observed modality set — an oversight-positive map of what each "
            "modality contributes and what withholding location leaves recoverable from metadata. "
            "Not evidence about any real population; synthetic-data software measurement only."
        ),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out
