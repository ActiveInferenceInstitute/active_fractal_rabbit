"""Pattern-of-life as structure recovery: the preferred-place occupancy.

The operational product of sparse-mobility analysis is not next-cell prediction
but the traveller's *structure* — where they spend time (the occupancy
distribution over cells) and which cells are preferred (the AgoraphobicPoint
set). This is a structure-recovery question, and it needs a structure metric
(total-variation distance of the inferred occupancy to the truth, and set
overlap of the top-k preferred cells), not the next-step Brier that the
session-7 candidates were wrongly judged on.

The honest baseline is the load-bearing choice. The generative model's transition
tensor and initial prior already encode empirical structure, so "the filter
recovers occupancy" is only a real claim if the inferred occupancy — which uses
the time/speed/burst observations — beats a DYNAMICS-ONLY belief that propagates
the prior through the transitions with NO observation updates. Beating uniform is
necessary but not sufficient; beating dynamics-only is the genuine test that the
non-location observations carry occupancy information. All synthetic-data software
measurements on the pinned FractalRabbit generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .inference import run_latent_inference
from .pomdp import build_pomdp, observation_indices
from .types import ObservationSequence, PomdpModel

DEEP_LATENT_MODALITIES = ("time_of_day", "speed", "report_burst")


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(p) - np.asarray(q))))


def true_occupancy(cells: np.ndarray, n_states: int) -> np.ndarray:
    counts = np.bincount(np.asarray(cells, dtype=int), minlength=n_states).astype(float)
    return counts / counts.sum()


def dynamics_only_occupancy(
    model: PomdpModel, sequence: ObservationSequence
) -> np.ndarray:
    """Mean belief if the filter propagates the prior through the transitions but
    never updates on an observation — the honest 'no evidence' baseline."""

    n_states = model.D.size
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    n_steps = len(observation_indices(model, sequence))
    beliefs = np.zeros((n_steps, n_states), dtype=float)
    belief = model.D.copy()
    for step in range(n_steps):
        beliefs[step] = belief
        if step + 1 < n_steps:
            if first[step + 1]:
                belief = model.D.copy()
            else:
                propagated = model.B[:, :, 0] @ belief
                belief = propagated / propagated.sum()
    return beliefs.mean(axis=0)


def preferred_set_jaccard(
    inferred_occ: np.ndarray, true_occ: np.ndarray, k: int
) -> float:
    inferred_top = set(np.argsort(-np.asarray(inferred_occ))[:k].tolist())
    true_top = set(np.argsort(-np.asarray(true_occ))[:k].tolist())
    return len(inferred_top & true_top) / len(inferred_top | true_top)


@dataclass(frozen=True)
class StructureRecoveryResult:
    n_states: int
    top_k: int
    true_occupancy_tv_uniform: float
    deep_latent_tv: float
    dynamics_only_tv: float
    full_location_tv: float
    deep_latent_beats_uniform: bool
    deep_latent_beats_dynamics_only: bool
    deep_latent_preferred_jaccard: float


def run_structure_recovery(
    sequence: ObservationSequence, *, smoothing: float, top_k: int = 3
) -> StructureRecoveryResult:
    """Recover the occupancy distribution / preferred-place set from non-location
    evidence and score it against uniform, dynamics-only, and full-location."""

    cells = np.asarray(sequence.spatial_states, dtype=int)
    n_states = sequence.num_spatial_states
    occ_true = true_occupancy(cells, n_states)
    uniform = np.ones(n_states) / n_states

    deep_model = build_pomdp(sequence, smoothing, modalities=DEEP_LATENT_MODALITIES)
    deep_occ = run_latent_inference(deep_model, sequence).posterior_states.mean(axis=0)
    full_model = build_pomdp(sequence, smoothing)
    full_occ = run_latent_inference(full_model, sequence).posterior_states.mean(axis=0)
    dyn_occ = dynamics_only_occupancy(deep_model, sequence)

    deep_tv = total_variation(deep_occ, occ_true)
    dyn_tv = total_variation(dyn_occ, occ_true)
    uni_tv = total_variation(uniform, occ_true)
    return StructureRecoveryResult(
        n_states=n_states,
        top_k=top_k,
        true_occupancy_tv_uniform=uni_tv,
        deep_latent_tv=deep_tv,
        dynamics_only_tv=dyn_tv,
        full_location_tv=total_variation(full_occ, occ_true),
        deep_latent_beats_uniform=deep_tv < uni_tv,
        deep_latent_beats_dynamics_only=deep_tv < dyn_tv,
        deep_latent_preferred_jaccard=preferred_set_jaccard(deep_occ, occ_true, top_k),
    )


STRUCTURE_RECOVERY_SCHEMA = "active-fractal-rabbit-structure-recovery-v1"


def cross_generator_structure_robustness(
    inference_config, *, smoothing: float, top_k: int = 3
) -> dict[str, object]:
    """Repeat the same structure-recovery computation across independent synthetic
    generator draws to expose the between-draw spread the single-fixture lane cannot see.

    Mirrors :func:`partial_observability.cross_realization_noisy_emission_advantage`:
    it loops :func:`power_analysis.generate_synthetic_sequence` over
    ``DEFAULT_CROSS_DRAW_SEEDS`` at ``DEFAULT_CROSS_DRAW_TRAVELLERS`` travellers, runs the
    identical deep-latent-vs-dynamics-only recovery on each draw, and reports the
    between-draw distribution. ``beats_dynamics_only_fraction`` is the mean over draws of
    the PAIRED per-draw indicator ``deep_tv < dyn_tv`` (computed per draw before
    aggregating), NOT a comparison of the two marginal means.
    """

    from .partial_observability import (
        DEFAULT_CROSS_DRAW_SEEDS,
        DEFAULT_CROSS_DRAW_TRAVELLERS,
    )
    from .power_analysis import generate_synthetic_sequence

    generator_seeds = DEFAULT_CROSS_DRAW_SEEDS
    n_travellers = DEFAULT_CROSS_DRAW_TRAVELLERS
    if not generator_seeds:
        raise ValueError("at least one generator seed is required")

    draw_rows: list[dict[str, object]] = []
    for generator_seed in generator_seeds:
        sequence = generate_synthetic_sequence(
            n_travellers, inference_config, seed=int(generator_seed)
        )
        r = run_structure_recovery(sequence, smoothing=smoothing, top_k=top_k)
        # Paired per-draw indicator: computed on THIS draw's own two TVs before any aggregation.
        beats = bool(r.deep_latent_tv < r.dynamics_only_tv)
        draw_rows.append(
            {
                "generator_seed": int(generator_seed),
                "tv_deep_latent": float(r.deep_latent_tv),
                "tv_dynamics_only": float(r.dynamics_only_tv),
                "preferred_jaccard": float(r.deep_latent_preferred_jaccard),
                "deep_beats_dynamics_only": beats,
            }
        )

    def _between_draw(field: str) -> dict[str, float]:
        values = np.asarray([row[field] for row in draw_rows], dtype=float)
        return {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }

    paired = np.asarray(
        [1.0 if row["deep_beats_dynamics_only"] else 0.0 for row in draw_rows]
    )
    return {
        "n_draws": int(len(generator_seeds)),
        "n_travellers": int(n_travellers),
        "top_k": int(top_k),
        "draw_rows": draw_rows,
        "tv_deep_latent": _between_draw("tv_deep_latent"),
        "tv_dynamics_only": _between_draw("tv_dynamics_only"),
        "preferred_jaccard": _between_draw("preferred_jaccard"),
        "beats_dynamics_only_fraction": float(paired.mean()),
        "boundary": (
            "independent synthetic FractalRabbit generator draws under the same pinned model; "
            "the between-draw spread is synthetic-generator variability, not empirical mobility "
            "transfer, and does not relax the oracle boundary — the canonical fixture lane rows "
            "remain the sole manuscript-bearing values"
        ),
    }


def run_and_write_structure_recovery(config_path, observations_path, out_path):
    import json
    from pathlib import Path

    from .config import load_config
    from .fractalrabbit import parse_waypoint_csv
    from .observations import discretize_observations

    config = load_config(config_path)

    def lane(path: Path, label: str) -> dict:
        sequence = discretize_observations(parse_waypoint_csv(path), config.inference)
        r = run_structure_recovery(sequence, smoothing=config.inference.smoothing)
        return {
            "label": label,
            "n_states": r.n_states,
            "tv_uniform": r.true_occupancy_tv_uniform,
            "tv_dynamics_only": r.dynamics_only_tv,
            "tv_deep_latent": r.deep_latent_tv,
            "tv_full_location": r.full_location_tv,
            "deep_latent_beats_uniform": r.deep_latent_beats_uniform,
            "deep_latent_beats_dynamics_only": r.deep_latent_beats_dynamics_only,
            "preferred_jaccard": r.deep_latent_preferred_jaccard,
        }

    lanes = []
    ext = Path(observations_path)
    if ext.exists():
        lanes.append(lane(ext, f"external:{ext.parent.name}"))
    assert config.fractalrabbit.fixture_csv is not None
    lanes.append(lane(Path(config.fractalrabbit.fixture_csv), "fixture"))
    payload = {
        "schema": STRUCTURE_RECOVERY_SCHEMA,
        "lanes": lanes,
        # SECONDARY robustness block: additive between-draw spread across independent
        # synthetic generator seeds. The `lanes` rows above stay the sole canonical values.
        "cross_generator": cross_generator_structure_robustness(
            config.inference, smoothing=config.inference.smoothing
        ),
        "caveat": (
            "Preferred-place occupancy recovery scored by total-variation distance to the true "
            "occupancy and top-k preferred-set overlap. The honest baseline is dynamics-only "
            "(propagate the trained prior with no observation updates): the non-location evidence "
            "beats it only in some regimes and can degrade recovery in others, because a trained "
            "transition model already encodes the preferred-place structure. Synthetic-data software "
            "measurement on the pinned FractalRabbit generator, not empirical mobility validation."
        ),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out
