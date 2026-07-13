"""Deterministic robustness lanes for the SOTA-program harness.

The lanes here are synthetic-only stressors that preserve the same scoring contract as the
partial-observability spine: locked traveller split, train-only transition learning, known
noisy-emission channel, and leakage-free prequential scoring. They vary only the latent
waypoint process and reporting pattern.
"""

from __future__ import annotations

import math

import numpy as np

from .observations import discretize_observations
from .power_analysis import generate_synthetic_waypoints
from .sota_program import (
    DEFAULT_N_TRAVELLERS,
    PartialObservabilityDataset,
    build_partial_observability_dataset_from_sequence,
)
from .types import InferenceConfig, WaypointObservation

SOTA_STRESSORS_SCHEMA = "active-fractal-rabbit-sota-stressors-v1"

REPORTING_GAP_SEED = 20260619
REGIME_STRESSOR_SEED = 20260620
ADVERSARIAL_MISSINGNESS_SEED = 20260621
REGIME_STRESSOR_REPEAT_SEED = 20260622
REPORTING_GAP_REPEAT_SEED = 20260623
ADVERSARIAL_REPEAT_SEED = 20260624
REGIME_STRESSOR_SECOND_REPEAT_SEED = 20260625
DEFAULT_REPORTING_GAP_MIN_KEEP = 64
DEFAULT_REGIME_REPORTS = 260
DEFAULT_REGIME_DAYS = 21.0
DEFAULT_BOX_KM = 200.0
DIAGNOSTIC_STRESSOR_LANES = (
    "reporting_gap_adversarial",
    "reporting_gap_repeat",
    "high_motion_adversarial_repeat",
    "regime_stressor_repeat",
    "regime_stressor_repeat_2",
)


def thin_reporting_gap_waypoints(
    rows: tuple[WaypointObservation, ...],
    *,
    seed: int = REPORTING_GAP_SEED,
    min_keep_per_traveller: int = DEFAULT_REPORTING_GAP_MIN_KEEP,
) -> tuple[WaypointObservation, ...]:
    """Thin dense synthetic waypoints into deterministic sparse-report sequences.

    Each traveller keeps first/last reports, a few short observed bursts, and a sparse
    background sample while deterministic hidden windows remove consecutive reports. The
    operation is traveller-local and seed-controlled; it never uses target outcomes or the
    eventual train/holdout assignment.
    """

    if min_keep_per_traveller < 2:
        raise ValueError("min_keep_per_traveller must be at least 2")
    by_traveller: dict[int, list[WaypointObservation]] = {}
    for row in sorted(rows, key=lambda item: (item.traveller_id, item.days)):
        by_traveller.setdefault(int(row.traveller_id), []).append(row)
    if len(by_traveller) < 2:
        raise ValueError("need at least 2 travellers for a reporting-gap lane")

    thinned: list[WaypointObservation] = []
    for traveller, seq in sorted(by_traveller.items()):
        n = len(seq)
        if n < 2:
            raise ValueError("each traveller needs at least 2 reports")
        rng = np.random.default_rng(seed + traveller * 9973)
        keep = np.zeros(n, dtype=bool)
        keep[0] = True
        keep[-1] = True

        n_gaps = max(2, min(6, n // 42))
        gap_start_max = max(2, n - 12)
        gap_starts = rng.choice(np.arange(2, gap_start_max), size=n_gaps, replace=False)
        suppressed = np.zeros(n, dtype=bool)
        for start in np.sort(gap_starts):
            gap_len = int(rng.integers(max(6, n // 36), max(10, n // 12)))
            suppressed[start : min(n - 1, start + gap_len)] = True

        n_bursts = max(2, min(7, n // 35))
        burst_centres = rng.choice(np.arange(1, n - 1), size=n_bursts, replace=False)
        for centre in burst_centres:
            width = int(rng.integers(2, 6))
            lo = max(1, int(centre) - width)
            hi = min(n - 1, int(centre) + width + 1)
            keep[lo:hi] = True

        background = rng.random(n) < 0.16
        cadence = (np.arange(n) % max(9, n // 20)) == 0
        keep |= (background | cadence) & ~suppressed

        target = min(n, int(min_keep_per_traveller))
        if int(keep.sum()) < target:
            candidates = np.flatnonzero(~keep)
            if candidates.size:
                needed = min(target - int(keep.sum()), candidates.size)
                fill_positions = np.linspace(0, candidates.size - 1, needed, dtype=int)
                keep[candidates[fill_positions]] = True
        thinned.extend(seq[idx] for idx in np.flatnonzero(keep))

    return tuple(sorted(thinned, key=lambda item: (item.traveller_id, item.days)))


def thin_adversarial_missingness_waypoints(
    rows: tuple[WaypointObservation, ...],
    *,
    seed: int = ADVERSARIAL_MISSINGNESS_SEED,
    min_keep_per_traveller: int = DEFAULT_REPORTING_GAP_MIN_KEEP,
) -> tuple[WaypointObservation, ...]:
    """Preferentially hide high-movement synthetic reports without using split outcomes.

    The variant is defensive and synthetic-only: it masks reports whose preceding
    displacement is high for that traveller, adds deterministic background sampling, and
    then fills to a minimum count. It never uses train/holdout membership, candidate losses,
    or target labels, so it is a reporting-process stressor rather than an adaptive attack.
    """

    if min_keep_per_traveller < 2:
        raise ValueError("min_keep_per_traveller must be at least 2")
    by_traveller: dict[int, list[WaypointObservation]] = {}
    for row in sorted(rows, key=lambda item: (item.traveller_id, item.days)):
        by_traveller.setdefault(int(row.traveller_id), []).append(row)
    if len(by_traveller) < 2:
        raise ValueError(
            "need at least 2 travellers for an adversarial-missingness lane"
        )

    thinned: list[WaypointObservation] = []
    for traveller, seq in sorted(by_traveller.items()):
        n = len(seq)
        if n < 3:
            raise ValueError("each traveller needs at least 3 reports")
        rng = np.random.default_rng(seed + traveller * 7919)
        coords = np.asarray([(row.x_km, row.y_km) for row in seq], dtype=float)
        displacements = np.zeros(n, dtype=float)
        displacements[1:] = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        high_motion_cut = float(np.quantile(displacements[1:], 0.70))
        high_motion = displacements >= high_motion_cut

        keep = np.zeros(n, dtype=bool)
        keep[0] = True
        keep[-1] = True
        background = rng.random(n) < 0.18
        cadence = (np.arange(n) % max(11, n // 18)) == 0
        keep |= background | cadence
        keep &= ~(high_motion & (rng.random(n) < 0.72))

        n_windows = max(2, min(5, n // 55))
        candidates = np.flatnonzero(
            high_motion & (np.arange(n) > 1) & (np.arange(n) < n - 1)
        )
        if candidates.size:
            starts = rng.choice(
                candidates, size=min(n_windows, candidates.size), replace=False
            )
            for start in starts:
                width = int(rng.integers(3, 9))
                keep[int(start) : min(n - 1, int(start) + width)] = False
        keep[0] = True
        keep[-1] = True

        target = min(n, int(min_keep_per_traveller))
        if int(keep.sum()) < target:
            low_motion_candidates = np.flatnonzero(~keep & ~high_motion)
            all_candidates = np.flatnonzero(~keep)
            fill_pool = (
                low_motion_candidates if low_motion_candidates.size else all_candidates
            )
            needed = min(target - int(keep.sum()), fill_pool.size)
            if needed:
                fill_positions = np.linspace(0, fill_pool.size - 1, needed, dtype=int)
                keep[fill_pool[fill_positions]] = True
        thinned.extend(seq[idx] for idx in np.flatnonzero(keep))

    return tuple(sorted(thinned, key=lambda item: (item.traveller_id, item.days)))


def _validate_regime_stressor_inputs(n_travellers: int, n_reports: int) -> None:
    if n_travellers < 2:
        raise ValueError("need at least 2 travellers for a locked split")
    if n_reports < 24:
        raise ValueError("regime stressor needs at least 24 reports per traveller")


def _regime_stressor_anchors(
    rng: np.random.Generator,
    *,
    box_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    home = rng.uniform(18.0, box_km - 18.0, size=2)
    commute_angle = rng.uniform(0.0, 2.0 * math.pi)
    commute_dist = rng.uniform(18.0, 55.0)
    work = home + commute_dist * np.array(
        [math.cos(commute_angle), math.sin(commute_angle)]
    )
    work = np.clip(work, 5.0, box_km - 5.0)
    alt = rng.uniform(10.0, box_km - 10.0, size=2)
    anchors = np.vstack([home, work, alt])
    return anchors, home.copy()


#: Phase boundaries partitioning a traveller's trajectory into 3 movement-mode segments
#: (see ``_regime_stressor_phase_parameters``). Named here, rather than left as inline
#: literals, so any downstream analysis that needs to know WHICH segment a given
#: observation falls in (e.g. ``regime_switch_phase_diagnostic.py``) reads the exact
#: same thresholds the generator itself uses, instead of a second hardcoded copy that
#: could silently drift out of sync.
REGIME_STRESSOR_PHASE_BOUNDARIES: tuple[float, float] = (0.34, 0.68)

#: Human-readable names for the 3 segments ``REGIME_STRESSOR_PHASE_BOUNDARIES`` splits
#: a trajectory into, in phase order. Matches the movement_scale/return_rate jump at
#: each boundary: commute is low-variance/high-return, explore is high-variance/
#: low-return, return_stabilization is intermediate on both.
REGIME_STRESSOR_PHASE_SEGMENTS: tuple[str, str, str] = (
    "commute",
    "explore",
    "return_stabilization",
)


def _regime_stressor_phase_parameters(phase: float) -> tuple[float, float, np.ndarray]:
    low, high = REGIME_STRESSOR_PHASE_BOUNDARIES
    if phase < low:
        return 0.82, 2.0, np.array([0.62, 0.31, 0.07])
    if phase < high:
        return 0.42, 12.0, np.array([0.40, 0.22, 0.38])
    return 0.72, 5.0, np.array([0.52, 0.39, 0.09])


def phase_segment_for_phase(phase: float) -> str:
    """Return the movement-mode segment name (see ``REGIME_STRESSOR_PHASE_SEGMENTS``)
    a given phase fraction (``idx / (n_reports - 1)``, as computed by
    ``generate_regime_stressor_waypoints``) falls into. Single source of truth for
    segment boundaries, shared with ``_regime_stressor_phase_parameters`` above."""

    low, high = REGIME_STRESSOR_PHASE_BOUNDARIES
    if phase < low:
        return REGIME_STRESSOR_PHASE_SEGMENTS[0]
    if phase < high:
        return REGIME_STRESSOR_PHASE_SEGMENTS[1]
    return REGIME_STRESSOR_PHASE_SEGMENTS[2]


def _advance_regime_stressor_position(
    rng: np.random.Generator,
    current: np.ndarray,
    anchors: np.ndarray,
    *,
    return_rate: float,
    movement_scale: float,
    anchor_probs: np.ndarray,
    box_km: float,
) -> np.ndarray:
    if rng.random() < return_rate:
        target = anchors[int(rng.choice(3, p=anchor_probs))]
        return target + rng.normal(0.0, movement_scale, size=2)
    jump = rng.normal(0.0, movement_scale * 2.5, size=2)
    return 0.78 * current + 0.22 * rng.uniform(0.0, box_km, size=2) + jump


def _sample_regime_stressor_interval(rng: np.random.Generator) -> float:
    if rng.random() < 0.12:
        return float(rng.uniform(0.35, 1.25))
    if rng.random() < 0.45:
        return float(rng.uniform(0.003, 0.035))
    return float(rng.uniform(0.045, 0.18))


def generate_regime_stressor_waypoints(
    n_travellers: int,
    *,
    seed: int = REGIME_STRESSOR_SEED,
    n_reports: int = DEFAULT_REGIME_REPORTS,
    n_days: float = DEFAULT_REGIME_DAYS,
    box_km: float = DEFAULT_BOX_KM,
) -> tuple[WaypointObservation, ...]:
    """Generate a larger synthetic lane with controlled regime changes.

    The process combines high-return home/work periods, lower-return exploration windows,
    bursty reporting gaps, and movement-mode shifts. It is intentionally synthetic and
    deterministic; the purpose is robustness scoring, not realism or operational transfer.
    """

    _validate_regime_stressor_inputs(n_travellers, n_reports)
    rng = np.random.default_rng(seed)
    rows: list[WaypointObservation] = []

    for traveller in range(n_travellers):
        anchors, current = _regime_stressor_anchors(rng, box_km=box_km)
        days = float(rng.uniform(0.0, 0.2))

        for idx in range(n_reports):
            rows.append(
                WaypointObservation(
                    traveller_id=traveller,
                    days=days,
                    x_km=float(np.clip(current[0], 0.0, box_km)),
                    y_km=float(np.clip(current[1], 0.0, box_km)),
                )
            )
            if idx == n_reports - 1:
                break

            phase = idx / max(n_reports - 1, 1)
            return_rate, movement_scale, anchor_probs = (
                _regime_stressor_phase_parameters(phase)
            )
            current = _advance_regime_stressor_position(
                rng,
                current,
                anchors,
                return_rate=return_rate,
                movement_scale=movement_scale,
                anchor_probs=anchor_probs,
                box_km=box_km,
            )
            days += _sample_regime_stressor_interval(rng)
            if days > n_days:
                days = n_days + float(idx) * 1e-5

    return tuple(sorted(rows, key=lambda item: (item.traveller_id, item.days)))


def build_reporting_gap_partial_observability_dataset(
    inference_config: InferenceConfig,
    *,
    n_travellers: int = DEFAULT_N_TRAVELLERS,
    eps: float = 0.3,
    smoothing: float = 0.5,
    holdout_fraction: float = 0.4,
    gen_seed: int = 2026,
    split_seed: int = 20260616,
    noise_seed: int = 0,
) -> PartialObservabilityDataset:
    """Build the deterministic reporting-gap lane on the shared SOTA spine."""

    dense = generate_synthetic_waypoints(
        n_travellers, seed=gen_seed, n_reports=240, n_days=18.0
    )
    sparse = thin_reporting_gap_waypoints(dense)
    sequence = discretize_observations(sparse, inference_config)
    return build_partial_observability_dataset_from_sequence(
        sequence,
        eps=eps,
        smoothing=smoothing,
        holdout_fraction=holdout_fraction,
        split_seed=split_seed,
        noise_seed=noise_seed,
        metadata={
            "lane_family": "reporting_gap_burst_missingness",
            "generator_seed": int(gen_seed),
            "thinning_seed": REPORTING_GAP_SEED,
            "min_keep_per_traveller": DEFAULT_REPORTING_GAP_MIN_KEEP,
            "n_travellers": int(n_travellers),
        },
    )


def build_adversarial_missingness_partial_observability_dataset(
    inference_config: InferenceConfig,
    *,
    n_travellers: int = DEFAULT_N_TRAVELLERS,
    eps: float = 0.3,
    smoothing: float = 0.5,
    holdout_fraction: float = 0.4,
    gen_seed: int = 2026,
    split_seed: int = 20260616,
    noise_seed: int = 0,
) -> PartialObservabilityDataset:
    """Build a deterministic high-motion missingness stressor on the shared SOTA spine."""

    dense = generate_synthetic_waypoints(
        n_travellers, seed=gen_seed, n_reports=260, n_days=20.0
    )
    sparse = thin_adversarial_missingness_waypoints(dense)
    sequence = discretize_observations(sparse, inference_config)
    return build_partial_observability_dataset_from_sequence(
        sequence,
        eps=eps,
        smoothing=smoothing,
        holdout_fraction=holdout_fraction,
        split_seed=split_seed,
        noise_seed=noise_seed,
        metadata={
            "lane_family": "adversarial_high_motion_missingness",
            "generator_seed": int(gen_seed),
            "thinning_seed": ADVERSARIAL_MISSINGNESS_SEED,
            "min_keep_per_traveller": DEFAULT_REPORTING_GAP_MIN_KEEP,
            "n_travellers": int(n_travellers),
        },
    )


def build_regime_stressor_partial_observability_dataset(
    inference_config: InferenceConfig,
    *,
    n_travellers: int = DEFAULT_N_TRAVELLERS,
    eps: float = 0.3,
    smoothing: float = 0.5,
    holdout_fraction: float = 0.4,
    gen_seed: int = REGIME_STRESSOR_SEED,
    split_seed: int = 20260616,
    noise_seed: int = 0,
) -> PartialObservabilityDataset:
    """Build the larger regime-stressor lane on the shared SOTA spine."""

    waypoints = generate_regime_stressor_waypoints(n_travellers, seed=gen_seed)
    sequence = discretize_observations(waypoints, inference_config)
    return build_partial_observability_dataset_from_sequence(
        sequence,
        eps=eps,
        smoothing=smoothing,
        holdout_fraction=holdout_fraction,
        split_seed=split_seed,
        noise_seed=noise_seed,
        metadata={
            "lane_family": "larger_regime_stressor",
            "generator_seed": int(gen_seed),
            "return_rate_schedule": "0.82_home_work|0.42_exploration|0.72_return",
            "movement_mode_shifts": "low_variance_commute|high_variance_explore|return_stabilization",
            "reporting_gap_process": "short_bursts_plus_rare_long_gaps",
            "n_travellers": int(n_travellers),
        },
    )


def build_regime_stressor_repeat_partial_observability_dataset(
    inference_config: InferenceConfig,
    *,
    n_travellers: int = DEFAULT_N_TRAVELLERS,
    eps: float = 0.3,
    smoothing: float = 0.5,
    holdout_fraction: float = 0.4,
    gen_seed: int = REGIME_STRESSOR_REPEAT_SEED,
    split_seed: int = 20260616,
    noise_seed: int = 0,
) -> PartialObservabilityDataset:
    """Build an independent repeated regime-stressor realization on the shared spine."""

    waypoints = generate_regime_stressor_waypoints(
        n_travellers,
        seed=gen_seed,
        n_reports=300,
        n_days=28.0,
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
            "lane_family": "larger_regime_stressor_repeated_realization",
            "generator_seed": int(gen_seed),
            "return_rate_schedule": "0.82_home_work|0.42_exploration|0.72_return",
            "movement_mode_shifts": "low_variance_commute|high_variance_explore|return_stabilization",
            "reporting_gap_process": "short_bursts_plus_rare_long_gaps",
            "n_travellers": int(n_travellers),
        },
    )


def lane_definitions() -> dict[str, dict[str, object]]:
    """Stable reader-facing definitions for the canonical SOTA lanes."""

    return {
        "fractalrabbit": {
            "status": "scored",
            "latent_process": "FractalRabbit synthetic spine",
            "robustness_role": "canonical partial-observability lane",
        },
        "gravity": {
            "status": "scored",
            "latent_process": "gravity / explore-and-return synthetic generator",
            "robustness_role": "cross-generator mechanism check",
        },
        "reporting_gap": {
            "status": "scored",
            "latent_process": "FractalRabbit-style synthetic waypoints after deterministic reporting-gap thinning",
            "robustness_role": "burst-missingness and sparse-report robustness lane",
        },
        "regime_stressor": {
            "status": "scored",
            "latent_process": "larger return-rate, burst-process, reporting-gap, and movement-mode stressor",
            "robustness_role": "larger synthetic regime-shift robustness lane",
        },
    }


def diagnostic_lane_definitions() -> dict[str, dict[str, object]]:
    """Stable reader-facing definitions for scored diagnostic stressor lanes."""

    return {
        "reporting_gap_adversarial": {
            "status": "scored_diagnostic",
            "latent_process": "FractalRabbit-style synthetic waypoints after high-motion-biased missingness",
            "robustness_role": "diagnostic adversarial-missingness variant; synthetic-only",
        },
        "reporting_gap_repeat": {
            "status": "scored_diagnostic",
            "latent_process": "independent repeated FractalRabbit-style reporting-gap thinning",
            "robustness_role": "diagnostic repeated burst-missingness/reporting-gap lane; synthetic-only",
        },
        "high_motion_adversarial_repeat": {
            "status": "scored_diagnostic",
            "latent_process": "independent high-motion-biased missingness realization",
            "robustness_role": "diagnostic repeated high-motion adversarial-missingness lane; synthetic-only",
        },
        "regime_stressor_repeat": {
            "status": "scored_diagnostic",
            "latent_process": "independent repeated larger regime-stressor realization",
            "robustness_role": "diagnostic repeated-realization regime stressor; synthetic-only",
        },
        "regime_stressor_repeat_2": {
            "status": "scored_diagnostic",
            "latent_process": "second independent repeated larger regime-stressor realization",
            "robustness_role": "diagnostic second repeated-realization regime stressor; synthetic-only",
        },
    }


def all_lane_definitions() -> dict[str, dict[str, object]]:
    """Canonical and diagnostic SOTA-program lane definitions."""

    return {**lane_definitions(), **diagnostic_lane_definitions()}
