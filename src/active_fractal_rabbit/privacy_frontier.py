"""Privacy-facing recoverability frontier: unicity and co-travel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def resolve_observations_path(
    observations_path: str | Path, fixture_csv: str | Path | None
) -> Path:
    """Resolve the observations CSV: the given path if it exists, else the config's
    ``fixture_csv`` fallback. Raises ``ValueError`` - not a bare ``assert``, which
    ``python -O`` strips entirely - if neither is available.

    Shared by ``run_and_write_privacy_frontier`` (this module) and
    ``privacy_utility_frontier.privacy_utility_frontier``, which independently
    reimplemented the same path-or-fixture-fallback resolution with different
    validation styles before consolidating here.
    """

    obs_path = Path(observations_path)
    if obs_path.exists():
        return obs_path
    if fixture_csv is None:
        raise ValueError("no observations path and no configured fixture csv")
    return Path(fixture_csv)


def trace_unicity(
    traveller_ids: np.ndarray, cells: np.ndarray, time_bins: np.ndarray
) -> float:
    """Fraction of distinct coarsened signatures per traveller."""

    signatures: dict[int, tuple[tuple[int, int], ...]] = {}
    for traveller in sorted(set(np.asarray(traveller_ids, dtype=int))):
        mask = np.asarray(traveller_ids, dtype=int) == traveller
        signature = tuple(
            sorted(
                (int(t), int(c))
                for t, c in zip(
                    np.asarray(time_bins, dtype=int)[mask],
                    np.asarray(cells, dtype=int)[mask],
                )
            )
        )
        signatures[int(traveller)] = signature
    return len(set(signatures.values())) / max(len(signatures), 1)


def co_travel_coincidence_rate(
    traveller_ids: np.ndarray, cells: np.ndarray, time_bins: np.ndarray
) -> float:
    """Fraction of traveller pairs sharing at least one coarse cell/time event."""

    ids = sorted(set(np.asarray(traveller_ids, dtype=int)))
    if len(ids) < 2:
        return 0.0
    events = {
        traveller: {
            (int(t), int(c))
            for t, c in zip(
                np.asarray(time_bins, dtype=int)[
                    np.asarray(traveller_ids, dtype=int) == traveller
                ],
                np.asarray(cells, dtype=int)[
                    np.asarray(traveller_ids, dtype=int) == traveller
                ],
            )
        }
        for traveller in ids
    }
    pairs = 0
    coincident = 0
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            pairs += 1
            coincident += int(bool(events[left] & events[right]))
    return coincident / pairs


def signature_regimes(sequence) -> dict[str, np.ndarray]:
    """Per-step coarsened signatures under each minimization regime.

    Single source of truth for the modality-minimization coarsening scheme
    shared by ``privacy_frontier`` and ``linkage_risk_frontier``, so the two
    frontiers cannot silently diverge on what "no_location"/"time_burst"/
    "time_only" mean.
    """

    return {
        "all": np.asarray(sequence.spatial_states, dtype=int),
        "no_location": np.asarray(
            sequence.time_bins * 10
            + sequence.speed_categories * 2
            + sequence.burst_categories,
            dtype=int,
        ),
        "time_burst": np.asarray(
            sequence.time_bins * 2 + sequence.burst_categories, dtype=int
        ),
        "time_only": np.asarray(sequence.time_bins, dtype=int),
    }


def privacy_frontier(sequence) -> list[dict[str, object]]:
    """Unicity/co-travel under modality minimization regimes."""

    traveller_ids = np.asarray(
        [row.traveller_id for row in sequence.observations], dtype=int
    )
    rows = []
    regimes = signature_regimes(sequence)
    for name, values in regimes.items():
        rows.append(
            {
                "modalities": name,
                "unicity": trace_unicity(traveller_ids, values, sequence.time_bins),
                "co_travel_coincidence": co_travel_coincidence_rate(
                    traveller_ids, values, sequence.time_bins
                ),
            }
        )
    return rows


PRIVACY_FRONTIER_SCHEMA = "active-fractal-rabbit-privacy-frontier-v1"


def run_and_write_privacy_frontier(config_path, observations_path, out_path):
    from .config import load_config
    from .fractalrabbit import parse_waypoint_csv
    from .observations import discretize_observations

    config = load_config(config_path)
    obs = resolve_observations_path(observations_path, config.fractalrabbit.fixture_csv)
    sequence = discretize_observations(parse_waypoint_csv(obs), config.inference)
    payload = {
        "schema": PRIVACY_FRONTIER_SCHEMA,
        "source": str(obs),
        "frontier": privacy_frontier(sequence),
        "caveat": (
            "Synthetic unicity and co-travel coincidence are reported as oversight-positive "
            "data-minimization risk metrics on synthetic traces, not as association "
            "or tasking capability over real people."
        ),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out
