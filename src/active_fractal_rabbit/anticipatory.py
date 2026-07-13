"""Anticipatory gap-filling recovery: when can latent inference beat persistence?

The persistence null (predict the previous true cell) is a deceptively strong
oracle: it is *handed* the true previous cell. A fully location-blind filter
never observes location, so its belief stays diffuse under a weaker information
set; this module reports that benchmark-specific information asymmetry rather
than papering over it.

The realistic scenario to test is **gap-filling**: in sporadic mobility
reporting, location is observed at sparse reports and missing in between. Both a
filter and persistence then know the last *reported* cell; the question is what
happens across a gap. Persistence rigidly holds the last reported cell. A
controlled harness proves that a filter can beat persistence when speed-to-move
coupling is planted, while the current FractalRabbit artifact is allowed to be a
null if propagation or speed conditioning fails to clear the oracle baseline.

Two honesty boundaries are load-bearing here:

- Speed is derived from inter-report displacement, so it is a strong kinematic
  cue for *whether* the cell changed. We treat it as a legitimate observation
  (a speed sensor independent of a position fix), but we report the limit lane,
  the gap-filling lane, a speed-free ablation, and the sign of the measured
  skill so the cue's contribution is never hidden or promoted by prose alone.
- Every number is a synthetic-data software measurement on the pinned simulator,
  never empirical mobility validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objectives import persistence_brier, skill_score
from .pomdp import _normalize_columns, observation_indices
from .types import ObservationSequence, PomdpModel


def speed_conditioned_transition(
    spatial_states: np.ndarray,
    speed_categories: np.ndarray,
    is_first_observation: np.ndarray,
    n_states: int,
    n_speed: int,
    smoothing: float,
) -> np.ndarray:
    """Empirical cell->cell transition tensor with one slice per speed category.

    ``B[:, :, k]`` is estimated from consecutive within-traveller pairs whose
    destination step falls in speed category ``k`` (boundary pairs excluded via
    ``is_first_observation``, mirroring the core transition guard). Low-speed
    slices come out near-identity (stationary reports do not move); high-speed
    slices are diffuse. Columns are normalized per slice.
    """

    spatial_states = np.asarray(spatial_states, dtype=int)
    speed_categories = np.asarray(speed_categories, dtype=int)
    first = np.asarray(is_first_observation, dtype=bool)
    tensor = np.full((n_states, n_states, n_speed), smoothing, dtype=float)
    for idx in range(spatial_states.shape[0] - 1):
        if first[idx + 1]:
            continue
        tensor[
            spatial_states[idx + 1], spatial_states[idx], speed_categories[idx + 1]
        ] += 1.0
    for k in range(n_speed):
        tensor[:, :, k] = _normalize_columns(tensor[:, :, k])
    return tensor


def slice_off_diagonal_mass(speed_b: np.ndarray) -> list[float]:
    """Mean off-diagonal column mass per speed slice (the learned move propensity).

    Monotone increase across speed bins is the honest signature that the model
    learned 'higher speed -> more likely to move'.
    """

    n_states = speed_b.shape[0]
    eye = np.eye(n_states, dtype=bool)
    return [
        float(speed_b[:, :, k][~eye].sum() / n_states) for k in range(speed_b.shape[2])
    ]


def gap_mask(
    n: int, is_first_observation: np.ndarray, gap_fraction: float, seed: int
) -> np.ndarray:
    """Deterministic gap mask: hide location at a fraction of non-first steps."""

    first = np.asarray(is_first_observation, dtype=bool)
    rng = np.random.default_rng(seed)
    mask = (rng.random(n) < gap_fraction) & ~first
    return mask


def _brier_on(posterior: np.ndarray, true_cells: np.ndarray, mask: np.ndarray) -> float:
    selected = posterior[mask]
    cells = np.asarray(true_cells, dtype=int)[mask]
    one_hot = np.zeros_like(selected)
    one_hot[np.arange(selected.shape[0]), cells] = 1.0
    return float(np.mean(np.sum((selected - one_hot) ** 2, axis=1)))


def forward_filter_core(
    model: PomdpModel,
    evidence,
    first: np.ndarray,
    speed: np.ndarray,
    *,
    speed_b: np.ndarray | None,
    location_index: int | None,
    gap_mask: np.ndarray | None,
    mask_location_unconditionally: bool = False,
    track_prior_entropy: bool = False,
    emit_priors: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Shared speed-conditioned forward-filter core.

    Consolidates three previously-independent forward-filter loops - this
    module's own ``_filter``, and ``active_sensing._gap_filling_filter`` /
    ``active_sensing._location_blind_priors`` - which reimplemented the same
    ~30-line loop three times with small but real behavioral differences:

    - ``gap_mask``: hide the location modality only at steps where the mask is
      True (matches ``_filter`` and ``_gap_filling_filter``'s conditional case).
    - ``mask_location_unconditionally``: hide the location modality at every
      step regardless of ``gap_mask`` (matches ``_location_blind_priors``).
    - ``track_prior_entropy``: also return each step's PRE-update prior entropy
      (matches ``_gap_filling_filter``).
    - ``emit_priors``: record each step's PRE-update prior instead of its
      POST-update posterior (matches ``_location_blind_priors``).
    - ``speed_b=None`` falls back to the flat ``model.B[:, :, 0]`` transition
      (matches ``_filter``'s default-lane behavior; the two active_sensing call
      sites always pass a real ``speed_b``).

    Verified byte-identical against all three original implementations across
    randomized trials (including the gap/no-gap and speed_b/no-speed_b
    branches) before consolidating; see the module test suites.
    """

    n_steps = len(evidence)
    n_states = model.D.size
    output = np.zeros((n_steps, n_states), dtype=float)
    prior_entropy = np.zeros(n_steps, dtype=float) if track_prior_entropy else None
    prior = model.D.copy()
    for step, observation in enumerate(evidence):
        if track_prior_entropy:
            prior_entropy[step] = float(-np.sum(prior * np.log(prior + 1e-12)))
        if emit_priors:
            output[step] = prior
        likelihood = np.ones(n_states, dtype=float)
        for modality, (matrix, obs_index) in enumerate(zip(model.A, observation)):
            if mask_location_unconditionally:
                hide = modality == location_index
            else:
                hide = bool(
                    gap_mask is not None
                    and location_index is not None
                    and gap_mask[step]
                    and modality == location_index
                )
            if hide:
                continue
            likelihood = likelihood * matrix[obs_index, :]
        post = likelihood * prior
        total = post.sum()
        post = post / total if total > 0 else np.ones(n_states) / n_states
        if not emit_priors:
            output[step] = post
        if step + 1 < n_steps:
            if first[step + 1]:
                prior = model.D.copy()
            elif speed_b is not None:
                propagated = speed_b[:, :, speed[step + 1]] @ post
                prior = propagated / propagated.sum()
            else:
                propagated = model.B[:, :, 0] @ post
                prior = propagated / propagated.sum()
    return output, prior_entropy


def _filter(
    model: PomdpModel,
    sequence: ObservationSequence,
    *,
    gaps: np.ndarray | None = None,
    location_index: int | None = None,
    speed_b: np.ndarray | None = None,
) -> np.ndarray:
    """Forward filter. Optionally hides the location modality at gap steps and
    propagates with a speed-conditioned transition slice (clamped to the observed
    speed category) instead of the flat transition.
    """

    evidence = observation_indices(model, sequence)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    speed = np.asarray(sequence.speed_categories, dtype=int)
    posterior, _ = forward_filter_core(
        model,
        evidence,
        first,
        speed,
        speed_b=speed_b,
        location_index=location_index,
        gap_mask=gaps,
    )
    return posterior


def _persistence_posterior(
    sequence: ObservationSequence, n_states: int, gaps: np.ndarray
) -> np.ndarray:
    """Persistence carries the last *observed* (non-gap) true cell forward."""

    cells = np.asarray(sequence.spatial_states, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    posterior = np.zeros((cells.shape[0], n_states), dtype=float)
    last = cells[0]
    for step in range(cells.shape[0]):
        if first[step] or not gaps[step]:
            last = cells[step]  # location observed -> known cell updates
        posterior[step, last] = 1.0
    return posterior


@dataclass(frozen=True)
class AnticipatoryResult:
    persistence_brier_gap: float
    flat_skill_gap: float
    scrp_skill_gap: float
    flat_brier_gap: float
    scrp_brier_gap: float
    n_gap_steps: int
    blind_persistence_skill_flat: float
    blind_persistence_skill_scrp: float
    speed_slice_off_diagonal: list[float]
    skill_by_speed_cat: dict[int, float]


def run_anticipatory_analysis(
    sequence: ObservationSequence,
    full_model: PomdpModel,
    *,
    smoothing: float,
    gap_fraction: float,
    seed: int,
) -> AnticipatoryResult:
    """Compare persistence, flat, and speed-clamped filters in two regimes.

    ``full_model`` must include the location modality (gap-filling needs to
    observe location at non-gap steps). The location-blind 'limit' numbers are
    computed by hiding location at *every* non-first step.
    """

    cells = np.asarray(sequence.spatial_states, dtype=int)
    n_states = full_model.D.size
    n = cells.shape[0]
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    speed = np.asarray(sequence.speed_categories, dtype=int)
    location_index = list(full_model.observation_names).index("location")
    speed_b = speed_conditioned_transition(
        cells, speed, first, n_states, sequence.num_speed_categories, smoothing
    )

    # Gap-filling regime: location observed except at gap steps; score on gaps.
    gaps = gap_mask(n, first, gap_fraction, seed)
    flat_post = _filter(full_model, sequence, gaps=gaps, location_index=location_index)
    scrp_post = _filter(
        full_model, sequence, gaps=gaps, location_index=location_index, speed_b=speed_b
    )
    persist_post = _persistence_posterior(sequence, n_states, gaps)
    pb = _brier_on(persist_post, cells, gaps)
    flat_b = _brier_on(flat_post, cells, gaps)
    scrp_b = _brier_on(scrp_post, cells, gaps)

    # Limit regime: location hidden everywhere (non-first). Persistence is the
    # oracle null over all non-first steps.
    blind = ~first
    blind_flat = _filter(
        full_model, sequence, gaps=blind, location_index=location_index
    )
    blind_scrp = _filter(
        full_model, sequence, gaps=blind, location_index=location_index, speed_b=speed_b
    )
    blind_pb = persistence_brier(cells, n_states, first)
    blind_flat_skill = skill_score(_brier_on_all(blind_flat, cells), blind_pb)
    blind_scrp_skill = skill_score(_brier_on_all(blind_scrp, cells), blind_pb)

    # Skill vs persistence per speed category on the gap steps (where the win lives).
    skill_by_cat: dict[int, float] = {}
    for k in range(sequence.num_speed_categories):
        cat_mask = gaps & (speed == k)
        if cat_mask.sum() == 0:
            continue
        cat_pb = _brier_on(persist_post, cells, cat_mask)
        cat_scrp = _brier_on(scrp_post, cells, cat_mask)
        skill_by_cat[k] = (
            skill_score(cat_scrp, cat_pb) if cat_pb > 1e-12 else float("nan")
        )

    return AnticipatoryResult(
        persistence_brier_gap=pb,
        flat_skill_gap=skill_score(flat_b, pb) if pb > 1e-12 else float("nan"),
        scrp_skill_gap=skill_score(scrp_b, pb) if pb > 1e-12 else float("nan"),
        flat_brier_gap=flat_b,
        scrp_brier_gap=scrp_b,
        n_gap_steps=int(gaps.sum()),
        blind_persistence_skill_flat=blind_flat_skill,
        blind_persistence_skill_scrp=blind_scrp_skill,
        speed_slice_off_diagonal=slice_off_diagonal_mass(speed_b),
        skill_by_speed_cat=skill_by_cat,
    )


def _brier_on_all(posterior: np.ndarray, true_cells: np.ndarray) -> float:
    cells = np.asarray(true_cells, dtype=int)
    one_hot = np.zeros_like(posterior)
    one_hot[np.arange(posterior.shape[0]), cells] = 1.0
    return float(np.mean(np.sum((posterior - one_hot) ** 2, axis=1)))


ANTICIPATORY_SCHEMA = "active-fractal-rabbit-anticipatory-v1"


def run_and_write_anticipatory(
    config_path,
    observations_path,
    out_path,
    *,
    gap_fraction: float = 0.5,
):
    """Discretize a waypoint CSV, run the analysis, and write the JSON artifact.

    ``observations_path`` is scored if it exists (the external run's richer data);
    otherwise the config's checked fixture is used. Returns the written Path.
    """

    import json
    from pathlib import Path

    from .config import load_config
    from .fractalrabbit import parse_waypoint_csv
    from .observations import discretize_observations
    from .pomdp import build_pomdp

    config = load_config(config_path)
    obs_path = Path(observations_path)
    if obs_path.exists():
        source = str(observations_path)
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
        gap_fraction=gap_fraction,
        seed=config.seed,
    )
    payload = anticipatory_payload(
        result, source=source, gap_fraction=gap_fraction, seed=config.seed
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def anticipatory_payload(
    result: AnticipatoryResult, *, source: str, gap_fraction: float, seed: int
) -> dict[str, object]:
    """JSON-safe artifact summarizing the limit finding and the gap-filling win."""

    return {
        "schema": ANTICIPATORY_SCHEMA,
        "source": source,
        "gap_fraction": gap_fraction,
        "seed": seed,
        "n_gap_steps": result.n_gap_steps,
        "gap_filling": {
            "persistence_brier": result.persistence_brier_gap,
            "flat_brier": result.flat_brier_gap,
            "scrp_brier": result.scrp_brier_gap,
            "flat_skill_vs_persistence": result.flat_skill_gap,
            "scrp_skill_vs_persistence": result.scrp_skill_gap,
            "speed_clamp_edge": result.scrp_skill_gap - result.flat_skill_gap,
        },
        "location_blind_limit": {
            "flat_skill_vs_persistence": result.blind_persistence_skill_flat,
            "scrp_skill_vs_persistence": result.blind_persistence_skill_scrp,
            "note": (
                "Persistence is handed the true previous cell; a location-blind filter "
                "cannot match that oracle, so skill here is typically negative. Speed "
                "narrows but does not close the gap."
            ),
        },
        "speed_slice_off_diagonal_mass": result.speed_slice_off_diagonal,
        "gap_skill_vs_persistence_by_speed_cat": {
            str(k): v for k, v in result.skill_by_speed_cat.items()
        },
        "caveat": (
            "Gap-filling recovery measured on the pinned synthetic simulator. Speed is "
            "derived from inter-report displacement, so it is a strong kinematic cue for "
            "whether the cell changed; this is disclosed, not hidden. All numbers are "
            "synthetic-data software measurements, not empirical mobility validation."
        ),
    }
