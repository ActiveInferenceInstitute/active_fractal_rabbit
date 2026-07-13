"""Cross-generator lane for the SOTA spine — a STRUCTURALLY DIFFERENT mobility process.

The honesty risk this module guards against is that any win on the spine
(``sota_program.py``) is a FractalRabbit-specific scoring artifact rather than a
genuine model capability. The spine's :func:`build_partial_observability_dataset`
draws its latent cell stream from FractalRabbit's fractal / agoraphobic-point
mechanism (via ``generate_synthetic_sequence``). This module provides a SECOND,
structurally unrelated synthetic generator that produces a
:class:`~active_fractal_rabbit.sota_program.PartialObservabilityDataset` of the
SAME shape — so the IDENTICAL harness (the same
:func:`~active_fractal_rabbit.sota_program.score_candidate`,
:func:`~active_fractal_rabbit.sota_program.emission_matrix`, and
:func:`~active_fractal_rabbit.sota_program.paired_loss_difference_ci`) can score
every candidate on it. A finding that holds on BOTH generators is mechanism-robust;
one that holds only on FractalRabbit is a generator-specific artifact.

The generator here is a **gravity / explore-and-return (EPR) hop process**:

* Each traveller has a fixed set of ``places`` with random ``masses`` scattered in a
  square box. This is a spatial-interaction mobility model (Zipf/gravity family),
  NOT a fractal random walk: there is no agoraphobic-point construction, no
  home-drift Brownian step, and no speed-regime mixture.
* At each step the traveller either RETURNS to a previously-visited place (with
  preferential-return probability proportional to past visit frequency) or EXPLORES
  a new place chosen by a **gravity law** — probability proportional to
  ``mass_j / distance(current, j) ** 2`` (Newtonian inverse-square spatial
  interaction). The waypoint is the chosen place's coordinate plus small Gaussian
  jitter, and reporting times are bursty (Pareto pauses).

The resulting continuous waypoints are discretized through the SAME tested
``discretize_observations`` into an ``ObservationSequence``; the latent cell stream,
the known symmetric emission channel, the train-only learned transition, and the
locked traveller split are assembled byte-for-byte the way the spine assembles
them (reusing ``emission_matrix``, the spine's noise channel, and
``lock_traveller_split``). The only thing that differs from the spine is the
LATENT DYNAMICS generator — exactly the variable a cross-generator test must vary.

This is a SECOND SYNTHETIC generator, not a real-mobility dataset. Scoring a
candidate here certifies nothing about real traces; it only separates
mechanism-robust findings from FractalRabbit-specific ones across two synthetic
generators.
"""

from __future__ import annotations

import numpy as np

from .observations import discretize_observations
from .sota_program import (
    DEFAULT_N_TRAVELLERS,
    PartialObservabilityDataset,
    build_partial_observability_dataset_from_sequence,
)
from .types import InferenceConfig, WaypointObservation

CROSS_GENERATOR_SCHEMA = "active-fractal-rabbit-sota-crossgen-gravity-v1"

#: Default knobs for the gravity / EPR generator. Chosen so that, on a 6x6 grid,
#: the discretized stream revisits cells (return structure) and also hops to
#: distant cells (gravity exploration) — i.e. it exercises the same partial-
#: observability machinery the spine needs without copying FractalRabbit's law.
DEFAULT_N_PLACES = 12
DEFAULT_N_REPORTS = 240
DEFAULT_BOX_KM = 200.0


def generate_gravity_waypoints(
    n_travellers: int,
    *,
    seed: int = 2026,
    n_places: int = DEFAULT_N_PLACES,
    n_reports: int = DEFAULT_N_REPORTS,
    box_km: float = DEFAULT_BOX_KM,
    rho: float = 0.6,
    gamma: float = 0.21,
    jitter_km: float = 1.5,
) -> tuple[WaypointObservation, ...]:
    """Deterministic gravity / explore-and-return waypoint generator (numpy only).

    A STRUCTURALLY DIFFERENT mobility process from ``generate_synthetic_waypoints``
    (FractalRabbit). Each traveller has ``n_places`` mass-weighted locations in a
    square box. At each step the traveller either RETURNS to a previously-visited
    place (preferential return, probability proportional to past visit frequency)
    or EXPLORES a new place by a gravity law (probability proportional to
    ``mass_j / distance(current, j) ** 2``). The emitted waypoint is the chosen
    place plus Gaussian jitter; report times are bursty with Pareto pauses.

    Fully deterministic given ``seed``. Returns rows sorted by ``(traveller, days)``
    so the downstream discretizer sees a stable order.
    """

    if n_travellers < 2:
        raise ValueError("need at least 2 travellers for a locked split")
    if n_places < 2:
        raise ValueError("need at least 2 places for explore/return dynamics")
    if n_reports < 2:
        raise ValueError("need at least 2 reports per traveller")

    rng = np.random.default_rng(seed)
    rows: list[WaypointObservation] = []
    for traveller in range(n_travellers):
        places = rng.uniform(0.0, box_km, size=(n_places, 2))
        # masses follow a heavy-tailed (Zipf-like) law: a few dominant attractors,
        # the gravity hallmark — distinct from FractalRabbit's uniform agoraphobic points.
        masses = 1.0 / (rng.permutation(n_places) + 1.0)
        visit_counts = np.zeros(n_places, dtype=float)
        current = int(rng.integers(0, n_places))
        visit_counts[current] += 1.0
        days = float(rng.uniform(0.0, 1.0))
        for _ in range(n_reports):
            place = places[current]
            x = float(np.clip(place[0] + rng.normal(0.0, jitter_km), 0.0, box_km))
            y = float(np.clip(place[1] + rng.normal(0.0, jitter_km), 0.0, box_km))
            rows.append(
                WaypointObservation(traveller_id=traveller, days=days, x_km=x, y_km=y)
            )
            # bursty inter-report time: short dwells punctuated by Pareto pauses
            days += float(0.02 + rng.pareto(1.8) * 0.05)

            distinct = float((visit_counts > 0).sum())
            p_explore = min(1.0, rho * distinct ** (-gamma))
            unvisited = np.where(visit_counts == 0.0)[0]
            if rng.random() < p_explore and unvisited.size > 0:
                # gravity law: prefer massive AND nearby unvisited places (inverse-square)
                dist = np.linalg.norm(places[unvisited] - places[current], axis=1) + 1.0
                weights = masses[unvisited] / dist**2
                current = int(rng.choice(unvisited, p=weights / weights.sum()))
            else:
                # preferential return: revisit weighted by accumulated visit frequency
                weights = visit_counts.copy()
                weights[current] = 0.0  # advance to a different place
                if weights.sum() > 0.0:
                    current = int(rng.choice(n_places, p=weights / weights.sum()))
            visit_counts[current] += 1.0
    return tuple(sorted(rows, key=lambda r: (r.traveller_id, r.days)))


def build_gravity_partial_observability_dataset(
    inference_config: InferenceConfig,
    *,
    n_travellers: int = DEFAULT_N_TRAVELLERS,
    eps: float = 0.3,
    smoothing: float = 0.5,
    holdout_fraction: float = 0.4,
    gen_seed: int = 2026,
    split_seed: int = 20260616,
    noise_seed: int = 0,
    n_places: int = DEFAULT_N_PLACES,
) -> PartialObservabilityDataset:
    """Build a spine-compatible dataset from the GRAVITY/EPR generator.

    Drop-in analogue of
    :func:`~active_fractal_rabbit.sota_program.build_partial_observability_dataset`,
    differing ONLY in the latent-dynamics generator: latent cells come from
    :func:`generate_gravity_waypoints` (discretized through the same tested
    ``discretize_observations``) instead of FractalRabbit. Everything else — the
    known symmetric emission channel (``emission_matrix``), the noisy-observation
    draw (the spine's ``_emit_observations``), the train-only learned transition,
    and the locked traveller split — is byte-for-byte the spine's, so the returned
    ``PartialObservabilityDataset`` is scored by the IDENTICAL
    :func:`~active_fractal_rabbit.sota_program.score_candidate` with no special
    casing. Fully deterministic given the seeds.
    """

    if n_travellers < 2:
        raise ValueError("need at least 2 travellers for a locked split")

    waypoints = generate_gravity_waypoints(
        n_travellers, seed=gen_seed, n_places=n_places
    )
    sequence = discretize_observations(waypoints, inference_config)

    return build_partial_observability_dataset_from_sequence(
        sequence,
        eps=eps,
        smoothing=smoothing,
        holdout_fraction=holdout_fraction,
        split_seed=split_seed,
        noise_seed=noise_seed,
        metadata={
            "lane_family": "gravity_explore_return",
            "generator_seed": int(gen_seed),
            "n_places": int(n_places),
            "n_travellers": int(n_travellers),
        },
    )
