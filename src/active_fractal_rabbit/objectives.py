"""Richer comparison objectives: decomposition, calibration, and chance floors.

Predictive loss and posterior entropy summarize one number per lane; this
module adds the objectives a methods reviewer asks for next:

- **Per-modality predictive-loss decomposition.** The joint loss
  ``-log p(o_t | prior_t)`` is split into per-modality marginal losses
  ``-log p(o_t^m | prior_t)``. A hard invariant binds them: every marginal
  probability is at least the joint probability, so every per-modality loss
  is bounded above by the joint loss at every step. The gap between the
  summed marginals and the joint loss is the modality interaction
  information under the running prior.
- **Calibration of state recovery.** Multiclass Brier score and top-1
  expected calibration error (ECE) scored against the exact true cell, with
  per-bin counts so small-sample bins are visible rather than hidden.
- **Chance floors and skill scores.** The uniform-predictor joint loss and
  uniform Brier score give the floors; skill = 1 - value / floor, so 1 is
  perfect, 0 is chance, negative is worse than chance.

Priors are reconstructed from an inference result (initial prior ``D``, then
the selected control's transition slice applied to the previous posterior)
rather than re-deriving the filter recursion; a binding check refuses to
emit numbers when the reconstruction disagrees with the lane's own recorded
predictive loss. All numbers are software measurements on synthetic data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pomdp import observation_indices
from .types import ObservationSequence, PomdpModel

CALIBRATION_SCHEMA = "active-fractal-rabbit-calibration-v1"
DEFAULT_ECE_BINS = 5
# Exact float64 filter reproduces its own loss to machine precision; the pymdp
# lane carries float32 + variational fixed-point noise, so its binding gate is
# looser while still catching structural divergence (wrong control ~ O(0.1)).
REFERENCE_BINDING_ATOL = 1e-9
PYMDP_BINDING_ATOL = 5e-3


def reconstruct_prior_trace(
    model: PomdpModel,
    posterior_states: np.ndarray,
    policy_posterior: np.ndarray,
) -> np.ndarray:
    """Rebuild the per-step prior trace a sequential lane used.

    ``prior_0 = D``; ``prior_t = normalize(B[:, :, argmax(q_pi_{t-1})] @
    posterior_{t-1})``. This matches both the reference filter and the pymdp
    loop, which select the argmax control and propagate through ``B``.
    """

    n_steps = posterior_states.shape[0]
    priors = np.zeros_like(posterior_states)
    priors[0] = model.D
    for idx in range(1, n_steps):
        control = int(np.argmax(policy_posterior[idx - 1]))
        propagated = model.B[:, :, control] @ posterior_states[idx - 1]
        total = float(propagated.sum())
        priors[idx] = (
            propagated / total
            if total > 0
            else np.ones_like(propagated) / propagated.size
        )
    return priors


def per_modality_losses(
    model: PomdpModel,
    sequence: ObservationSequence,
    prior_states: np.ndarray,
    recorded_joint_loss: np.ndarray,
    binding_atol: float = REFERENCE_BINDING_ATOL,
) -> dict[str, float]:
    """Decompose the joint predictive loss into per-modality marginal losses.

    Refuses to emit numbers when the reconstructed priors do not reproduce
    the lane's own recorded joint loss (binding check), so the decomposition
    can never drift silently from the filter it describes.
    """

    evidence = observation_indices(model, sequence)
    n_steps = len(evidence)
    if prior_states.shape[0] != n_steps:
        raise ValueError("prior trace length does not match the observation count")
    joint = np.zeros(n_steps, dtype=float)
    marginal = np.zeros((n_steps, len(model.A)), dtype=float)
    for idx, observation in enumerate(evidence):
        prior = prior_states[idx]
        likelihood = np.ones(model.D.size, dtype=float)
        for m, (matrix, obs_index) in enumerate(zip(model.A, observation)):
            marginal_prob = max(float(matrix[obs_index, :] @ prior), 1e-12)
            marginal[idx, m] = -np.log(marginal_prob)
            likelihood *= matrix[obs_index, :]
        joint[idx] = -np.log(max(float(likelihood @ prior), 1e-12))
    max_delta = float(np.max(np.abs(joint - recorded_joint_loss)))
    if max_delta > binding_atol:
        raise ValueError(
            f"prior reconstruction does not reproduce the recorded joint loss (max delta {max_delta:.3g})"
        )
    metrics: dict[str, float] = {
        "per_modality_binding_max_delta": max_delta,
        "modality_interaction_information": float(
            np.mean(marginal.sum(axis=1) - joint)
        ),
    }
    for m, name in enumerate(model.observation_names):
        metrics[f"per_modality_loss_{name}"] = float(np.mean(marginal[:, m]))
    return metrics


def chance_floors(model: PomdpModel) -> dict[str, float]:
    """Uniform-predictor floors for the joint loss and the recovery Brier score."""

    joint_floor = float(sum(np.log(matrix.shape[0]) for matrix in model.A))
    n_states = int(model.D.size)
    uniform_brier = float((n_states - 1) / n_states)
    return {
        "chance_floor_joint_loss": joint_floor,
        "chance_floor_recovery_brier": uniform_brier,
    }


def skill_score(mean_loss: float, floor: float) -> float:
    """1 is perfect, 0 is the uniform-chance floor, negative is worse than chance."""

    if floor <= 0:
        raise ValueError("chance floor must be positive")
    return float(1.0 - mean_loss / floor)


def base_rate_brier(true_cells: np.ndarray, n_states: int) -> float:
    """Brier score of the empirical base-rate (marginal-frequency) predictor.

    The base-rate predictor ignores the observations and predicts the constant
    empirical cell-frequency vector ``f`` at every step. Its multiclass Brier
    score has the closed form ``1 - sum_k f_k**2`` (the Gini impurity of the
    true-cell distribution), which is a strictly tighter null than the uniform
    floor ``(S-1)/S`` whenever the cells are not uniformly occupied. Skill
    against this floor answers the honest question the uniform floor cannot:
    how much does observing anything beat already knowing the base rate?
    Returns 0.0 for a degenerate single-cell trace (the task carries no
    base-rate uncertainty to beat).
    """

    true_cells = np.asarray(true_cells, dtype=int)
    if true_cells.size and int(true_cells.max()) >= n_states:
        raise ValueError(
            f"true cell index {int(true_cells.max())} exceeds n_states={n_states}"
        )
    counts = np.bincount(true_cells, minlength=n_states).astype(float)
    frequencies = counts / counts.sum()
    return float(1.0 - np.sum(frequencies**2))


def brier_with_base_rate_fallback(
    true_cells: np.ndarray,
    n_states: int,
    fallback: np.ndarray,
    non_fallback_prediction: np.ndarray,
) -> float:
    """Shared base-rate-fallback + Brier-accumulation loop.

    At every step, predicts the empirical base-rate (marginal-frequency)
    vector wherever ``fallback[step]`` is True or ``step == 0`` (the "no
    usable prior report yet" case at the start of any trace); otherwise
    predicts ``non_fallback_prediction[step]``. Accumulates the multiclass
    Brier score against the true one-hot cell and returns the per-step mean.
    :func:`persistence_brier` and
    :func:`return_structure.return_conditioned_brier` both route through this
    helper so their documented base-rate-fallback equivalence is enforced by
    shared code rather than by hand-matched duplicated loops.
    """

    true_cells = np.asarray(true_cells, dtype=int)
    fallback = np.asarray(fallback, dtype=bool)
    n_steps = true_cells.shape[0]
    if n_steps == 0:
        return 0.0
    counts = np.bincount(true_cells, minlength=n_states).astype(float)
    base_rate = counts / counts.sum()
    total = 0.0
    for step in range(n_steps):
        one_hot = np.zeros(n_states, dtype=float)
        one_hot[true_cells[step]] = 1.0
        prediction = (
            base_rate
            if (fallback[step] or step == 0)
            else non_fallback_prediction[step]
        )
        total += float(np.sum((prediction - one_hot) ** 2))
    return float(total / n_steps)


def persistence_brier(
    true_cells: np.ndarray, n_states: int, first_observation: np.ndarray
) -> float:
    """Brier score of the temporal-persistence (last-true-cell) recovery null.

    The persistence predictor ignores the observations and predicts the
    previous report's true cell (a one-hot), falling back to the base-rate
    frequency vector at each traveller's first report (where there is no
    previous report). For autocorrelated mobility — reports tend to stay in or
    return to the same cell — persistence is a much stronger null than the
    order-ignoring base rate, so skill against it is the honest measure of how
    much the *observations* add beyond "assume they didn't move". The ordering
    base_rate >= persistence is NOT universal: it flips on anti-correlated data
    where moving is the norm. Returns 0.0 for a single-report degenerate trace.
    """

    true_cells = np.asarray(true_cells, dtype=int)
    first_observation = np.asarray(first_observation, dtype=bool)
    n_steps = true_cells.shape[0]
    if first_observation.shape[0] != n_steps:
        raise ValueError("first_observation length must match the true-cell trace")
    if true_cells.size and int(true_cells.max()) >= n_states:
        raise ValueError(
            f"true cell index {int(true_cells.max())} exceeds n_states={n_states}"
        )
    non_fallback_prediction = np.zeros((n_steps, n_states), dtype=float)
    for step in range(1, n_steps):
        non_fallback_prediction[step, true_cells[step - 1]] = 1.0
    return brier_with_base_rate_fallback(
        true_cells, n_states, first_observation, non_fallback_prediction
    )


#: Bin edge-inclusion conventions for :func:`expected_calibration_error`. Three call
#: sites in this package (here, sota_benchmark._ece, neural_baseline._expected_calibration_error)
#: independently reimplemented the same equal-width binning loop with three different,
#: individually valid choices of which bin boundary is inclusive. Consolidated into one
#: parameterized core so the loop exists once; each convention is preserved exactly
#: (verified byte-identical against each site's pre-consolidation output) rather than
#: silently unified, since a boundary-inclusion change can shift which bin an exact-valued
#: confidence lands in and change a computed number this project treats as load-bearing.
ECE_EDGE_LAST_CLOSED = "last_closed"  # [low, high) except final bin [low, high]
ECE_EDGE_FIRST_CLOSED = "first_closed"  # (low, high] except first bin [low, high]
ECE_EDGE_RIGHT_CLOSED = (
    "right_closed"  # (low, high] always (confidence==0.0 falls in no bin)
)


def expected_calibration_error(
    confidences: np.ndarray,
    hits: np.ndarray,
    n_bins: int,
    *,
    edge_convention: str = ECE_EDGE_LAST_CLOSED,
) -> tuple[float, list[dict[str, float]]]:
    """Count-weighted |accuracy - confidence| top-1 ECE over ``n_bins`` equal-width bins.

    Returns (ece, bins); bins carry per-bin low/high/count/mean_confidence/accuracy so
    sparse bins stay visible rather than hidden. See the module-level ``ECE_EDGE_*``
    constants for what ``edge_convention`` controls and why it exists.
    """

    confidences = np.asarray(confidences, dtype=float)
    hits = np.asarray(hits, dtype=float)
    total = confidences.size
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float]] = []
    ece = 0.0
    if total == 0:
        return 0.0, bins
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        if edge_convention == ECE_EDGE_LAST_CLOSED:
            is_last = i == len(edges) - 2
            in_bin = (confidences >= low) & (
                confidences <= high if is_last else confidences < high
            )
        elif edge_convention == ECE_EDGE_FIRST_CLOSED:
            is_first = i == 0
            in_bin = (confidences >= low if is_first else confidences > low) & (
                confidences <= high
            )
        elif edge_convention == ECE_EDGE_RIGHT_CLOSED:
            in_bin = (confidences > low) & (confidences <= high)
        else:
            raise ValueError(f"unknown edge_convention: {edge_convention!r}")
        count = int(in_bin.sum())
        if count == 0:
            bins.append(
                {
                    "low": float(low),
                    "high": float(high),
                    "count": 0.0,
                    "mean_confidence": float("nan"),
                    "accuracy": float("nan"),
                }
            )
            continue
        bin_confidence = float(np.mean(confidences[in_bin]))
        bin_accuracy = float(np.mean(hits[in_bin]))
        ece += (count / total) * abs(bin_accuracy - bin_confidence)
        bins.append(
            {
                "low": float(low),
                "high": float(high),
                "count": float(count),
                "mean_confidence": bin_confidence,
                "accuracy": bin_accuracy,
            }
        )
    return float(ece), bins


def recovery_calibration(
    posterior: np.ndarray,
    true_cells: np.ndarray,
    n_bins: int = DEFAULT_ECE_BINS,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Brier score and top-1 ECE of a posterior trace against the true cells.

    Returns (metrics, bins). Bins carry per-bin counts so sparse bins are
    visible. Brier is the multiclass score in [0, 2]; ECE is the
    count-weighted |accuracy - confidence| over occupied bins, in [0, 1].
    """

    true_cells = np.asarray(true_cells, dtype=int)
    n_steps, n_states = posterior.shape
    if true_cells.shape[0] != n_steps:
        raise ValueError("true cell trace length does not match the posterior")
    one_hot = np.zeros_like(posterior)
    one_hot[np.arange(n_steps), true_cells] = 1.0
    brier = float(np.mean(np.sum((posterior - one_hot) ** 2, axis=1)))

    confidence = np.max(posterior, axis=1)
    correct = (np.argmax(posterior, axis=1) == true_cells).astype(float)
    ece, bins = expected_calibration_error(
        confidence, correct, n_bins, edge_convention=ECE_EDGE_LAST_CLOSED
    )
    metrics = {
        "brier": brier,
        "ece": float(ece),
        "mean_top1_confidence": float(np.mean(confidence)),
        "top1_accuracy": float(np.mean(correct)),
        "base_rate_brier": base_rate_brier(true_cells, n_states),
        "n_steps": float(n_steps),
        "n_states": float(n_states),
    }
    return metrics, bins


def lane_objectives(
    model: PomdpModel,
    sequence: ObservationSequence,
    posterior_states: np.ndarray,
    policy_posterior: np.ndarray,
    predictive_loss: np.ndarray,
    prefix: str,
) -> tuple[dict[str, float], dict[str, object]]:
    """All objectives for one inference lane, metric keys prefixed by lane."""

    binding_atol = (
        PYMDP_BINDING_ATOL if prefix.startswith("pymdp") else REFERENCE_BINDING_ATOL
    )
    priors = reconstruct_prior_trace(model, posterior_states, policy_posterior)
    decomposition = per_modality_losses(
        model, sequence, priors, predictive_loss, binding_atol=binding_atol
    )
    floors = chance_floors(model)
    calibration_metrics, calibration_bins = recovery_calibration(
        posterior_states, np.asarray(sequence.spatial_states, dtype=int)
    )
    metrics: dict[str, float] = {}
    for key, value in decomposition.items():
        metrics[f"{prefix}_{key}"] = value
    metrics[f"{prefix}_skill_score"] = skill_score(
        float(np.mean(predictive_loss)), floors["chance_floor_joint_loss"]
    )
    metrics[f"{prefix}_brier_skill_score"] = skill_score(
        calibration_metrics["brier"], floors["chance_floor_recovery_brier"]
    )
    # Skill against the tighter base-rate (marginal-frequency) null: 0 means the
    # lane does no better than predicting the constant base rate, so the
    # by-construction-trivial location lane no longer scores a free high skill.
    base_brier = calibration_metrics["base_rate_brier"]
    metrics[f"{prefix}_brier_skill_vs_base_rate"] = (
        skill_score(calibration_metrics["brier"], base_brier)
        if base_brier > 1e-12
        else float("nan")
    )
    # Skill against the temporal-persistence null. Mobility reports are highly
    # autocorrelated (stay-put / return), so persistence is a far stronger null
    # than the order-ignoring base rate; skill against it is the honest measure
    # of how much the observations add beyond "assume no movement".
    persist_brier = persistence_brier(
        np.asarray(sequence.spatial_states, dtype=int),
        model.D.size,
        sequence.is_first_observation,
    )
    calibration_metrics["persistence_brier"] = persist_brier
    metrics[f"{prefix}_brier_skill_vs_persistence"] = (
        skill_score(calibration_metrics["brier"], persist_brier)
        if persist_brier > 1e-12
        else float("nan")
    )
    for key, value in calibration_metrics.items():
        metrics[f"{prefix}_calibration_{key}"] = value
    payload: dict[str, object] = {
        "prefix": prefix,
        "calibration_bins": calibration_bins,
        "calibration": calibration_metrics,
        "per_modality": decomposition,
        "per_modality_losses": {
            name: decomposition[f"per_modality_loss_{name}"]
            for name in model.observation_names
        },
        "mean_joint_loss": float(np.mean(predictive_loss)),
        "floors": floors,
        "skill_score": metrics[f"{prefix}_skill_score"],
        "brier_skill_score": metrics[f"{prefix}_brier_skill_score"],
        "brier_skill_vs_base_rate": metrics[f"{prefix}_brier_skill_vs_base_rate"],
        "brier_skill_vs_persistence": metrics[f"{prefix}_brier_skill_vs_persistence"],
    }
    return metrics, payload


def write_calibration_artifact(
    path: Path, lanes: list[dict[str, object]], floors: dict[str, float]
) -> None:
    """Write the calibration/objectives artifact consumed by the figure layer."""

    payload = {
        "schema": CALIBRATION_SCHEMA,
        "floors": floors,
        "lanes": lanes,
        "caveat": (
            "Calibration and skill scores are software measurements against the synthetic "
            "simulator's own latent structure; the location-observing lane is sharply "
            "calibrated largely by construction. Nothing here is empirical mobility validation."
        ),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
