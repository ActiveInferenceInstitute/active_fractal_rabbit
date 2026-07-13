"""Ground-truth evaluation against the simulator's own latent structure.

Two ground-truth surfaces exist. First, the discretized true spatial cell of
every report is always known exactly (it is computed from the reported
coordinates), so any posterior over latent cells can be scored against it.
This evaluation is trivially near-perfect when ``location`` is an observation
modality and becomes a genuine recovery test in the deep-latent configuration.
Second, the external simulator's ``*PLACES.csv`` output names the true
AgoraphobicPoints place behind every report, which exposes revisit structure
finer than any spatial cell.

Nothing here is empirical validity: recovery of a synthetic simulator's
latent structure is a software/methods evaluation only.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .types import ObservationSequence, PlaceObservation, WaypointObservation


def align_places(
    observations: tuple[WaypointObservation, ...],
    places: tuple[PlaceObservation, ...],
) -> list[int]:
    """Return per-row place ids aligned to the sorted waypoint observations.

    Both inputs are sorted by (traveller, days); the upstream simulator emits
    the two files row-aligned. Misalignment is an error, never silently
    tolerated.
    """

    if len(observations) != len(places):
        raise ValueError(
            f"waypoint and place row counts differ: {len(observations)} vs {len(places)}"
        )
    aligned: list[int] = []
    for row, place in zip(observations, places):
        if row.traveller_id != place.traveller_id or abs(row.days - place.days) > 1e-9:
            raise ValueError(
                f"waypoint/place misalignment at traveller {row.traveller_id} day {row.days}"
            )
        aligned.append(place.place_id)
    return aligned


def place_statistics(place_ids: list[int]) -> dict[str, float]:
    """Revisit-structure summary of the ground-truth place stream."""

    counts = Counter(place_ids)
    n_reports = len(place_ids)
    n_unique = len(counts)
    revisited = sum(1 for value in counts.values() if value > 1)
    return {
        "place_n_reports": float(n_reports),
        "place_n_unique": float(n_unique),
        "place_n_revisited": float(revisited),
        "place_revisit_rate": float(revisited / n_unique) if n_unique else 0.0,
        "place_mean_reports_per_place": float(n_reports / n_unique)
        if n_unique
        else 0.0,
    }


def state_recovery(
    sequence: ObservationSequence,
    posterior: np.ndarray,
    prefix: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Score a posterior trace against the exact true latent cell per report.

    Returns (metrics, mass_on_true, top1_correct). ``mass_on_true[t]`` is the
    posterior probability assigned to the true cell at step ``t``;
    ``top1_correct[t]`` is 1.0 when the posterior argmax equals the true cell.
    """

    true_cells = np.asarray(sequence.spatial_states, dtype=int)
    if posterior.shape[0] != true_cells.shape[0]:
        raise ValueError("posterior length does not match the observation count")
    if posterior.shape[1] <= int(true_cells.max()):
        raise ValueError(
            "posterior state dimension smaller than the largest true cell index"
        )
    mass_on_true = posterior[np.arange(true_cells.shape[0]), true_cells]
    top1 = (np.argmax(posterior, axis=1) == true_cells).astype(float)
    metrics = {
        f"{prefix}_mean_mass_on_true_cell": float(np.mean(mass_on_true)),
        f"{prefix}_top1_accuracy": float(np.mean(top1)),
        f"{prefix}_final_mass_on_true_cell": float(mass_on_true[-1]),
    }
    return metrics, mass_on_true, top1
