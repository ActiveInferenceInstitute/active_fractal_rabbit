"""Latent-state filtering and policy-proxy diagnostics."""

from __future__ import annotations

import numpy as np

from .pomdp import observation_indices, to_pymdp_payload
from .types import BaselineResult, InferenceResult, ObservationSequence, PomdpModel


def run_latent_inference(
    model: PomdpModel, sequence: ObservationSequence
) -> InferenceResult:
    """Filter latent states and compute transparent policy-proxy diagnostics."""

    import time

    loop_start = time.perf_counter()
    evidence = observation_indices(model, sequence)
    qs = np.zeros((len(sequence.observations), model.D.size), dtype=float)
    losses = np.zeros(len(sequence.observations), dtype=float)
    entropy = np.zeros(len(sequence.observations), dtype=float)
    policy_posterior = np.zeros(
        (len(sequence.observations), model.B.shape[2]), dtype=float
    )
    policy_proxy_score = np.zeros_like(policy_posterior)
    policy_predictive_entropy = np.zeros_like(policy_posterior)
    policy_preference_divergence = np.zeros_like(policy_posterior)

    prior = model.D.copy()
    for idx, observation in enumerate(evidence):
        likelihood = _likelihood(model, observation)
        predictive_probability = max(float(likelihood @ prior), 1e-12)
        posterior = _normalize(likelihood * prior)
        qs[idx] = posterior
        losses[idx] = -np.log(predictive_probability)
        entropy[idx] = _entropy(posterior)
        predictive_entropy, preference_divergence, proxy_score = (
            _policy_proxy_components(model, posterior)
        )
        policy_predictive_entropy[idx] = predictive_entropy
        policy_preference_divergence[idx] = preference_divergence
        policy_proxy_score[idx] = proxy_score
        policy_posterior[idx] = _softmax(-proxy_score)
        prior = model.B[:, :, int(np.argmax(policy_posterior[idx]))] @ posterior
        prior = _normalize(prior)

    payload_shape_ok = _pymdp_payload_shape_check(model)
    metrics = {
        "n_observations": float(len(sequence.observations)),
        "n_travellers": float(len({row.traveller_id for row in sequence.observations})),
        "n_latent_states": float(model.D.size),
        "mean_predictive_loss": float(np.mean(losses)),
        "mean_posterior_entropy": float(np.mean(entropy)),
        "final_posterior_entropy": float(entropy[-1]),
        "mean_policy_entropy": float(
            np.mean([_entropy(row) for row in policy_posterior])
        ),
        "policy_exploration_selected_fraction": (
            float(
                np.mean(
                    np.argmax(policy_posterior, axis=1)
                    == model.control_names.index("high_entropy_exploration")
                )
            )
            if "high_entropy_exploration" in model.control_names
            else 0.0
        ),
        "pymdp_payload_shape_ok": float(payload_shape_ok),
        "reference_runtime_s": float(time.perf_counter() - loop_start),
    }
    return InferenceResult(
        posterior_states=qs,
        predictive_loss=losses,
        posterior_entropy=entropy,
        policy_posterior=policy_posterior,
        policy_proxy_score=policy_proxy_score,
        policy_predictive_entropy=policy_predictive_entropy,
        policy_preference_divergence=policy_preference_divergence,
        metrics=metrics,
    )


def run_fixed_transition_baseline(
    model: PomdpModel,
    sequence: ObservationSequence,
    control_name: str = "empirical_continuation",
) -> BaselineResult:
    """Run a fixed-transition baseline filter over the same observation factors."""

    evidence = observation_indices(model, sequence)
    qs = np.zeros((len(sequence.observations), model.D.size), dtype=float)
    losses = np.zeros(len(sequence.observations), dtype=float)
    entropy = np.zeros(len(sequence.observations), dtype=float)
    control_idx = (
        model.control_names.index(control_name)
        if control_name in model.control_names
        else 0
    )
    transition = model.B[:, :, control_idx]

    prior = model.D.copy()
    for idx, observation in enumerate(evidence):
        likelihood = _likelihood(model, observation)
        predictive_probability = max(float(likelihood @ prior), 1e-12)
        posterior = _normalize(likelihood * prior)
        qs[idx] = posterior
        losses[idx] = -np.log(predictive_probability)
        entropy[idx] = _entropy(posterior)
        prior = _normalize(transition @ posterior)

    metrics = {
        "baseline_control": control_name
        if control_name in model.control_names
        else model.control_names[control_idx],
        "baseline_mean_predictive_loss": float(np.mean(losses)),
        "baseline_mean_posterior_entropy": float(np.mean(entropy)),
        "baseline_final_posterior_entropy": float(entropy[-1]),
    }
    return BaselineResult(
        posterior_states=qs,
        predictive_loss=losses,
        posterior_entropy=entropy,
        metrics=metrics,
    )


def holdout_split(sequence: ObservationSequence, fraction: float) -> np.ndarray:
    """Return a boolean holdout mask marking the last ``fraction`` of each traveller's rows.

    Travellers contribute holdout rows only when they keep at least two
    training rows, so the training lane always retains within-traveller
    transition structure. Rows are already ordered by (traveller, days).
    """

    if not 0.0 <= fraction <= 0.95:
        raise ValueError("holdout fraction must be between 0 and 0.95")
    mask = np.zeros(len(sequence.observations), dtype=bool)
    if fraction == 0.0:
        return mask
    indices_by_traveller: dict[int, list[int]] = {}
    for idx, row in enumerate(sequence.observations):
        indices_by_traveller.setdefault(row.traveller_id, []).append(idx)
    for indices in indices_by_traveller.values():
        count = int(len(indices) * fraction)
        count = min(count, len(indices) - 2)
        if count > 0:
            mask[indices[-count:]] = True
    return mask


def run_holdout_scoring(
    train_model: PomdpModel,
    sequence: ObservationSequence,
    holdout_mask: np.ndarray,
) -> dict[str, float]:
    """Score holdout rows with a model estimated from training rows only.

    The filter runs over the full ordered sequence (so the holdout rows are
    scored in their real temporal context) but losses and entropies are
    aggregated only at holdout indices. Callers must pass a sequence whose
    data-driven discretization statistics (quantile edges, auto bounds,
    median burst cutoff) were estimated from training rows only — the
    pipeline does this via ``discretize_observations(..., estimation_mask)``
    — and a model whose A/B/D estimation excluded the holdout rows.
    """

    holdout_mask = np.asarray(holdout_mask, dtype=bool)
    if holdout_mask.shape[0] != len(sequence.observations):
        raise ValueError("holdout mask length must match the observation count")
    n_holdout = int(holdout_mask.sum())
    if n_holdout == 0:
        return {
            "holdout_n_holdout": 0.0,
            "holdout_n_train": float(len(sequence.observations)),
            "holdout_mean_predictive_loss": float("nan"),
            "holdout_mean_posterior_entropy": float("nan"),
        }
    inference = run_latent_inference(train_model, sequence)
    return {
        "holdout_n_holdout": float(n_holdout),
        "holdout_n_train": float(len(sequence.observations) - n_holdout),
        "holdout_mean_predictive_loss": float(
            np.mean(inference.predictive_loss[holdout_mask])
        ),
        "holdout_mean_posterior_entropy": float(
            np.mean(inference.posterior_entropy[holdout_mask])
        ),
    }


def _likelihood(model: PomdpModel, observation: tuple[int, ...]) -> np.ndarray:
    likelihood = np.ones(model.D.size, dtype=float)
    for modality, index in zip(model.A, observation):
        likelihood *= modality[index, :]
    return likelihood


def _policy_proxy_components(
    model: PomdpModel, posterior: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictive_entropy_values = np.zeros(model.B.shape[2], dtype=float)
    preference_divergence_values = np.zeros(model.B.shape[2], dtype=float)
    for control in range(model.B.shape[2]):
        predicted_state = _normalize(model.B[:, :, control] @ posterior)
        predictive_entropy = 0.0
        preference_divergence = 0.0
        for A_m, C_m in zip(model.A, model.C):
            predicted_obs = _normalize(A_m @ predicted_state)
            predictive_entropy += _entropy(predicted_obs)
            preferred = _softmax(C_m)
            preference_divergence += float(
                np.sum(
                    predicted_obs
                    * (np.log(predicted_obs + 1e-12) - np.log(preferred + 1e-12))
                )
            )
        predictive_entropy_values[control] = predictive_entropy
        preference_divergence_values[control] = preference_divergence
    return (
        predictive_entropy_values,
        preference_divergence_values,
        predictive_entropy_values + preference_divergence_values,
    )


def _pymdp_payload_shape_check(model: PomdpModel) -> bool:
    try:
        payload = to_pymdp_payload(model)
    except Exception:
        return False
    return bool(
        len(payload["A"]) == len(model.A)
        and payload["B"].shape == model.B.shape
        and payload["D"].shape == model.D.shape
    )


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0:
        return np.ones_like(values, dtype=float) / values.size
    return values / total


def _entropy(values: np.ndarray) -> float:
    probs = _normalize(values)
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()
