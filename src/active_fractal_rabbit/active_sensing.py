"""Expected-free-energy decomposition and budgeted active sensing.

This module opens the active-inference value function into its two standard
components and uses them to schedule observations. Expected free energy of a
candidate observation decomposes as ``neg_efe = info_gain + utility`` (Friston
et al.; the same additive split pymdp computes internally):

- **Epistemic value (info gain / salience).** The expected reduction in
  uncertainty about the latent cell from making an observation in a modality:
  ``H(prior) - E_o[H(posterior | o)]``. Always non-negative.
- **Pragmatic value (utility).** The expected log-preference of the predicted
  observation under the preference vector ``C``: ``E_o[C(o)]``. Zero when ``C``
  is uniform (no preferences), which is the project's default.

Because the project runs with uniform ``C``, the agent has no pragmatic drive,
so its sensing value is *purely epistemic* — the epistemic share is 1 by the
construction of a preference-free generative model. A preference sweep (sharpen
``C``) makes the pragmatic term grow and the epistemic share fall, which is the
non-vacuous check that the decomposition genuinely separates the two drives
rather than reporting a constant.

Budgeted active sensing then ranks non-first steps by epistemic value and spends
a fixed observation budget on the highest-value (most uncertain-and-resolvable)
steps. Whether that beats a budget-matched blind schedule is structure
dependent — it pays off on the richer external resimulation but not on the tiny
fixture — and is always scored on the FULL trajectory (every schedule covers the
identical evaluation set), never on each schedule's own gap steps (a metric that
flatters whichever schedule fixed the hardest steps). All numbers are
synthetic-data software measurements, never empirical mobility validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anticipatory import forward_filter_core, speed_conditioned_transition
from .pomdp import observation_indices
from .types import ObservationSequence, PomdpModel


def _entropy(distribution: np.ndarray) -> float:
    return float(-np.sum(distribution * np.log(distribution + 1e-12)))


def modality_info_gain(prior: np.ndarray, emission: np.ndarray) -> float:
    """Expected reduction in cell-entropy from observing one modality.

    ``emission`` is the modality's ``A_m`` matrix of shape (observations, states).
    Returns ``H(prior) - E_o[H(posterior|o)] >= 0``.
    """

    prior = np.asarray(prior, dtype=float)
    obs_marginal = emission @ prior  # P(o) = sum_s A[o,s] prior[s]
    expected_posterior_entropy = 0.0
    for obs_index in range(emission.shape[0]):
        joint = emission[obs_index, :] * prior
        total = float(obs_marginal[obs_index])
        if total <= 0:
            continue
        posterior = joint / total
        expected_posterior_entropy += total * _entropy(posterior)
    return _entropy(prior) - expected_posterior_entropy


def modality_utility(
    prior: np.ndarray, emission: np.ndarray, preference: np.ndarray
) -> float:
    """Expected log-preference (pragmatic value) of a modality's predicted obs."""

    prior = np.asarray(prior, dtype=float)
    obs_marginal = emission @ prior
    return float(np.sum(obs_marginal * np.asarray(preference, dtype=float)))


def efe_components(
    prior: np.ndarray, emission: np.ndarray, preference: np.ndarray
) -> dict[str, float]:
    """Epistemic, pragmatic, and the epistemic share for one observation."""

    epistemic = modality_info_gain(prior, emission)
    pragmatic = modality_utility(prior, emission, preference)
    denom = abs(epistemic) + abs(pragmatic)
    share = abs(epistemic) / denom if denom > 1e-12 else float("nan")
    return {
        "epistemic": epistemic,
        "pragmatic": pragmatic,
        "neg_efe": epistemic + pragmatic,
        "epistemic_share": share,
    }


def _gap_filling_filter(
    model: PomdpModel,
    sequence: ObservationSequence,
    evidence,
    speed_b: np.ndarray,
    location_index: int,
    gaps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Speed-conditioned forward filter; returns (posterior, per-step prior entropy)."""

    first = np.asarray(sequence.is_first_observation, dtype=bool)
    speed = np.asarray(sequence.speed_categories, dtype=int)
    posterior, prior_entropy = forward_filter_core(
        model,
        evidence,
        first,
        speed,
        speed_b=speed_b,
        location_index=location_index,
        gap_mask=gaps,
        track_prior_entropy=True,
    )
    return posterior, prior_entropy


def _location_blind_priors(
    model: PomdpModel,
    evidence,
    speed_b: np.ndarray,
    location_index: int,
    first: np.ndarray,
    speed: np.ndarray,
) -> list[np.ndarray]:
    """Speed-conditioned forward filter that always masks the location modality.

    Shared by the info-gain ranking, the epistemic-share report, and the
    preference sweep, all of which need the identical location-blind belief
    (so a location fix never leaks into the ranking/reporting belief itself).
    Returns the prior *before* each step's update, one array per step.
    """

    priors, _ = forward_filter_core(
        model,
        evidence,
        first,
        speed,
        speed_b=speed_b,
        location_index=location_index,
        gap_mask=None,
        mask_location_unconditionally=True,
        emit_priors=True,
    )
    return list(priors)


def _full_trajectory_brier(
    posterior: np.ndarray, true_cells: np.ndarray, first: np.ndarray
) -> float:
    """Brier over ALL non-first steps (the common evaluation set for every schedule)."""

    mask = ~np.asarray(first, dtype=bool)
    cells = np.asarray(true_cells, dtype=int)[mask]
    selected = posterior[mask]
    one_hot = np.zeros_like(selected)
    one_hot[np.arange(selected.shape[0]), cells] = 1.0
    return float(np.mean(np.sum((selected - one_hot) ** 2, axis=1)))


@dataclass(frozen=True)
class ActiveSensingResult:
    fix_fraction: float
    budget: int
    info_gain_led_brier: float
    random_mean_brier: float
    random_std_brier: float
    periodic_brier: float
    fraction_random_beaten: float
    margin_vs_random_mean: float
    info_gain_led_posterior_entropy: float
    random_mean_posterior_entropy: float
    median_epistemic_share: float


def run_active_sensing(
    sequence: ObservationSequence,
    model: PomdpModel,
    *,
    smoothing: float,
    fix_fraction: float,
    n_random_seeds: int = 40,
    preference: np.ndarray | None = None,
) -> ActiveSensingResult:
    """Budgeted active sensing: epistemic-led vs random vs periodic schedules.

    Ranks non-first steps by epistemic value (expected info gain of a location
    fix under the propagated belief), spends ``budget`` fixes on the top steps,
    and scores the full trajectory. The random and periodic schedules spend the
    same budget blindly. Also reports the median epistemic share at the fixed
    steps (≈1 under uniform preferences).
    """

    cells = np.asarray(sequence.spatial_states, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    speed = np.asarray(sequence.speed_categories, dtype=int)
    n_states = model.D.size
    location_index = list(model.observation_names).index("location")
    location_emission = model.A[location_index]
    if preference is None:
        preference = np.zeros(location_emission.shape[0], dtype=float)
    speed_b = speed_conditioned_transition(
        cells, speed, first, n_states, sequence.num_speed_categories, smoothing
    )
    evidence = observation_indices(model, sequence)
    non_first = np.where(~first)[0]
    budget = int(fix_fraction * len(non_first))

    # Rank by epistemic value of a fix at each non-first step, using the
    # location-blind prior belief propagated through the whole trajectory (a
    # fixed, schedule-independent ranking). Reused below for the median
    # epistemic-share report at the fixed steps.
    location_blind_priors = _location_blind_priors(
        model, evidence, speed_b, location_index, first, speed
    )
    info_gain = np.full(len(cells), -np.inf)
    for step in range(len(evidence)):
        if not first[step]:
            info_gain[step] = modality_info_gain(
                location_blind_priors[step], location_emission
            )
    order = non_first[np.argsort(-info_gain[non_first])]

    def brier_for(fix_steps: np.ndarray) -> tuple[float, np.ndarray]:
        gaps = ~first.copy()
        gaps[fix_steps] = False
        posterior, _ = _gap_filling_filter(
            model, sequence, evidence, speed_b, location_index, gaps
        )
        return _full_trajectory_brier(posterior, cells, first), posterior

    led_steps = order[:budget]
    led_brier, led_post = brier_for(led_steps)
    random_briers = []
    for seed in range(n_random_seeds):
        rng = np.random.default_rng(seed)
        random_briers.append(brier_for(rng.choice(non_first, budget, replace=False))[0])
    random_briers = np.array(random_briers)
    stride = max(1, len(non_first) // budget) if budget else 1
    periodic_steps = non_first[::stride][:budget]
    periodic_brier, _ = brier_for(periodic_steps)

    # Median epistemic share at the fixed steps (≈1 under uniform preference),
    # reusing the location-blind priors already propagated above.
    fix_set = sorted(int(s) for s in led_steps)
    shares = [
        efe_components(location_blind_priors[step], location_emission, preference)[
            "epistemic_share"
        ]
        for step in fix_set
    ]
    shares = [s for s in shares if not np.isnan(s)]

    led_entropy = float(np.mean([_entropy(p) for p in led_post[~first]]))
    rng0 = np.random.default_rng(0)
    rand_post = brier_for(rng0.choice(non_first, budget, replace=False))[1]
    rand_entropy = float(np.mean([_entropy(p) for p in rand_post[~first]]))

    return ActiveSensingResult(
        fix_fraction=fix_fraction,
        budget=budget,
        info_gain_led_brier=led_brier,
        random_mean_brier=float(random_briers.mean()),
        random_std_brier=float(random_briers.std()),
        periodic_brier=periodic_brier,
        fraction_random_beaten=float(np.mean(led_brier < random_briers)),
        margin_vs_random_mean=float(random_briers.mean() - led_brier),
        info_gain_led_posterior_entropy=led_entropy,
        random_mean_posterior_entropy=rand_entropy,
        median_epistemic_share=float(np.median(shares)) if shares else float("nan"),
    )


ACTIVE_SENSING_SCHEMA = "active-fractal-rabbit-active-sensing-v1"


def _preference_sweep(
    sequence: ObservationSequence,
    model: PomdpModel,
    strengths=(0.0, 0.5, 1.0, 2.0, 4.0),
) -> list[dict[str, float]]:
    """Mean epistemic share of a location fix as preference strength grows.

    The preference vector rewards location observation bin 0 with weight
    ``strength``. At strength 0 (uniform) the share is 1; as the agent acquires
    a pragmatic drive the share falls — proving the decomposition tracks the two
    components rather than reporting a constant.
    """

    location_index = list(model.observation_names).index("location")
    emission = model.A[location_index]
    n_obs = emission.shape[0]
    speed = np.asarray(sequence.speed_categories, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    n_states = model.D.size
    speed_b = speed_conditioned_transition(
        np.asarray(sequence.spatial_states, dtype=int),
        speed,
        first,
        n_states,
        sequence.num_speed_categories,
        1e-6,
    )
    evidence = observation_indices(model, sequence)
    # Collect the location-blind prior at each non-first step.
    all_priors = _location_blind_priors(
        model, evidence, speed_b, location_index, first, speed
    )
    priors = [p for step, p in enumerate(all_priors) if not first[step]]
    sweep = []
    for strength in strengths:
        preference = np.zeros(n_obs, dtype=float)
        preference[0] = strength
        shares = [
            efe_components(p, emission, preference)["epistemic_share"] for p in priors
        ]
        shares = [s for s in shares if not np.isnan(s)]
        sweep.append(
            {
                "preference_strength": float(strength),
                "mean_epistemic_share": float(np.mean(shares)),
            }
        )
    return sweep


def run_and_write_active_sensing(
    config_path,
    observations_path,
    out_path,
    *,
    fix_fractions=(0.3, 0.4, 0.5),
):
    """Run active sensing on the external run (rich) and the fixture (small), at
    several budgets, and write the JSON artifact. Reports the EFE decomposition
    (epistemic share) and the structure-dependent budgeted-sensing win.
    """

    import json
    from pathlib import Path

    from .config import load_config
    from .fractalrabbit import parse_waypoint_csv
    from .observations import discretize_observations
    from .pomdp import build_pomdp

    config = load_config(config_path)

    def lane(obs_path: Path, label: str):
        sequence = discretize_observations(
            parse_waypoint_csv(obs_path), config.inference
        )
        model = build_pomdp(sequence, config.inference.smoothing)
        budgets = []
        for fraction in fix_fractions:
            r = run_active_sensing(
                sequence,
                model,
                smoothing=config.inference.smoothing,
                fix_fraction=fraction,
            )
            budgets.append(
                {
                    "fix_fraction": r.fix_fraction,
                    "budget": r.budget,
                    "info_gain_led_brier": r.info_gain_led_brier,
                    "random_mean_brier": r.random_mean_brier,
                    "random_std_brier": r.random_std_brier,
                    "fraction_random_beaten": r.fraction_random_beaten,
                    "margin_vs_random_mean": r.margin_vs_random_mean,
                    "info_gain_led_posterior_entropy": r.info_gain_led_posterior_entropy,
                    "random_mean_posterior_entropy": r.random_mean_posterior_entropy,
                    "median_epistemic_share": r.median_epistemic_share,
                }
            )
        return {
            "label": label,
            "n_steps": int(len(sequence.spatial_states)),
            "budgets": budgets,
        }

    ext = Path(observations_path)
    lanes = []
    if ext.exists():
        lanes.append(lane(ext, f"external:{ext.parent.name}"))
    assert config.fractalrabbit.fixture_csv is not None
    lanes.append(lane(Path(config.fractalrabbit.fixture_csv), "fixture"))

    # Preference sweep on the fixture: mean epistemic share of a location fix as
    # the preference vector C is sharpened. Demonstrates the decomposition is a
    # real measurement (share falls from 1 toward 0), not a constant.
    fixture_seq = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    fixture_model = build_pomdp(fixture_seq, config.inference.smoothing)
    epistemic_share_sweep = _preference_sweep(fixture_seq, fixture_model)

    payload = {
        "schema": ACTIVE_SENSING_SCHEMA,
        "lanes": lanes,
        "epistemic_share_sweep": epistemic_share_sweep,
        "caveat": (
            "Budgeted active sensing scored on the full non-first trajectory (every schedule "
            "covers the identical evaluation set). Epistemic-led acquisition ranks steps by the "
            "expected information gain of a location fix; under uniform preferences the EFE value "
            "is purely epistemic (share 1.0). Whether epistemic-led beats a budget-matched blind "
            "schedule is structure dependent. Synthetic-data software measurements, not empirical "
            "mobility validation."
        ),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out
