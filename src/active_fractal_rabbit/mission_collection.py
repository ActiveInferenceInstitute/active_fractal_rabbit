"""Mission-conditioned collection management: when does the epistemic term help?

Collection is budget-constrained and mission-prioritized. Framed as active
inference, allocating a fixed observation budget is an expected-free-energy
problem: the EFE of fixing a location at a step is its epistemic value (expected
information gain about the cell) plus its pragmatic value (the mission preference
over what is observed). Under uniform preferences the pragmatic term is zero and
sensing is purely epistemic (the session-7 null). Under MISSION preferences the
pragmatic term becomes load-bearing.

The honest, prototype-verified finding is regime-scoped: EFE-led tasking does
is judged against pure-pragmatic tasking on the expanded benchmark, including
the weak-dynamics comparison. The epistemic term can move the ranking, but on
this synthetic run it does not clear the configured meaningful-improvement
margin over the simpler pragmatic baseline. The benchmark lesson is therefore
an honest null, not a capability win. This is a synthetic-data, abstract
budgeted-inference study — not an operational tasking policy for any real
system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .active_sensing import modality_info_gain, modality_utility
from .anticipatory import speed_conditioned_transition
from .pomdp import build_pomdp, observation_indices
from .types import ObservationSequence

MEANINGFUL_RECOVERY_DELTA = 0.005


def _mission_weights(
    n_states: int, mission_cells: set[int], off_weight: float = 0.05
) -> np.ndarray:
    return np.array(
        [1.0 if c in mission_cells else off_weight for c in range(n_states)]
    )


def _z(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sample = values[mask]
    return (values - sample.mean()) / (sample.std() + 1e-9)


@dataclass(frozen=True)
class MissionCollectionResult:
    regime: str
    budget: int
    efe_recovery: float
    epistemic_recovery: float
    pragmatic_recovery: float
    random_recovery: float
    efe_beats_epistemic: bool
    efe_beats_random: bool
    efe_beats_pragmatic: bool


def run_mission_collection(
    sequence: ObservationSequence,
    *,
    smoothing: float,
    mission_cells: set[int],
    fix_fraction: float = 0.3,
    resolved_dynamics: bool = True,
    n_random_seeds: int = 20,
) -> MissionCollectionResult:
    """Mission-weighted recovery under EFE / epistemic / pragmatic / random tasking.

    ``resolved_dynamics`` selects the speed-conditioned transition (a good dynamics
    model that resolves uncertainty) vs the flat transition (weak dynamics)."""

    cells = np.asarray(sequence.spatial_states, dtype=int)
    speed = np.asarray(sequence.speed_categories, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    n_states = sequence.num_spatial_states
    full = build_pomdp(sequence, smoothing)
    loc = list(full.observation_names).index("location")
    a_loc = full.A[loc]
    w_cell = _mission_weights(n_states, mission_cells)
    c_loc = np.array([w_cell[c] for c in range(a_loc.shape[0])])
    speed_b = speed_conditioned_transition(
        cells, speed, first, n_states, sequence.num_speed_categories, smoothing
    )
    evidence = observation_indices(full, sequence)
    non_first = np.where(~first)[0]
    mask = ~first

    def propagate(belief, step):
        if resolved_dynamics:
            propagated = speed_b[:, :, speed[step + 1]] @ belief
        else:
            propagated = full.B[:, :, 0] @ belief
        return propagated / propagated.sum()

    # Rank values from the location-blind forward pass.
    epi = np.full(len(cells), -np.inf)
    pra = np.full(len(cells), -np.inf)
    prior = full.D.copy()
    for step, obs in enumerate(evidence):
        if not first[step]:
            epi[step] = modality_info_gain(prior, a_loc)
            pra[step] = modality_utility(prior, a_loc, c_loc)
        likelihood = np.ones(n_states)
        for modality, (matrix, obs_index) in enumerate(zip(full.A, obs)):
            if modality == loc:
                continue
            likelihood = likelihood * matrix[obs_index, :]
        post = likelihood * prior
        post = post / post.sum() if post.sum() > 0 else np.ones(n_states) / n_states
        if step + 1 < len(evidence):
            prior = full.D.copy() if first[step + 1] else propagate(post, step)
    efe = _z(epi, mask) + _z(pra, mask)

    def gap_filter(fix_steps):
        gaps = ~first.copy()
        gaps[fix_steps] = False
        post = np.zeros((len(cells), n_states))
        prior = full.D.copy()
        for step, obs in enumerate(evidence):
            likelihood = np.ones(n_states)
            for modality, (matrix, obs_index) in enumerate(zip(full.A, obs)):
                if gaps[step] and modality == loc:
                    continue
                likelihood = likelihood * matrix[obs_index, :]
            p = likelihood * prior
            p = p / p.sum() if p.sum() > 0 else np.ones(n_states) / n_states
            post[step] = p
            if step + 1 < len(evidence):
                prior = full.D.copy() if first[step + 1] else propagate(p, step)
        return post

    def mission_recovery(post):
        num = sum(w_cell[cells[t]] * post[t, cells[t]] for t in non_first)
        den = sum(w_cell[cells[t]] for t in non_first)
        return num / den

    budget = int(fix_fraction * len(non_first))

    def top_k(score):
        return non_first[np.argsort(-score[non_first])][:budget]

    efe_rec = mission_recovery(gap_filter(top_k(efe)))
    epi_rec = mission_recovery(gap_filter(top_k(epi)))
    pra_rec = mission_recovery(gap_filter(top_k(pra)))
    rng = np.random.default_rng(0)
    rand_rec = float(
        np.mean(
            [
                mission_recovery(
                    gap_filter(rng.choice(non_first, budget, replace=False))
                )
                for _ in range(n_random_seeds)
            ]
        )
    )
    return MissionCollectionResult(
        regime="resolved" if resolved_dynamics else "unresolved",
        budget=budget,
        efe_recovery=efe_rec,
        epistemic_recovery=epi_rec,
        pragmatic_recovery=pra_rec,
        random_recovery=rand_rec,
        efe_beats_epistemic=efe_rec > epi_rec,
        efe_beats_random=efe_rec > rand_rec,
        efe_beats_pragmatic=efe_rec > pra_rec + MEANINGFUL_RECOVERY_DELTA,
    )


MISSION_COLLECTION_SCHEMA = "active-fractal-rabbit-mission-collection-v1"


def run_and_write_mission_collection(config_path, observations_path, out_path):
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
    occ = np.bincount(
        np.asarray(sequence.spatial_states, dtype=int),
        minlength=sequence.num_spatial_states,
    )
    mission = set(np.argsort(occ)[:2].tolist())

    def regime(resolved: bool) -> dict:
        r = run_mission_collection(
            sequence,
            smoothing=config.inference.smoothing,
            mission_cells=mission,
            fix_fraction=0.3,
            resolved_dynamics=resolved,
        )
        return {
            "regime": r.regime,
            "efe": float(r.efe_recovery),
            "epistemic": float(r.epistemic_recovery),
            "pragmatic": float(r.pragmatic_recovery),
            "random": float(r.random_recovery),
            "efe_beats_pragmatic": bool(r.efe_beats_pragmatic),
        }

    payload = {
        "schema": MISSION_COLLECTION_SCHEMA,
        "source": source,
        "mission_cells": sorted(mission),
        "resolved": regime(True),
        "unresolved": regime(False),
        "caveat": (
            "Mission-weighted recovery under EFE / epistemic / pragmatic / random tasking, on the "
            "synthetic generator. The epistemic term does not clear the configured meaningful-improvement "
            "margin over pure mission-pragmatic tasking on the expanded benchmark, including the weak-dynamics "
            "comparison. An abstract budgeted-inference study, not an operational tasking policy for any real system."
        ),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out
