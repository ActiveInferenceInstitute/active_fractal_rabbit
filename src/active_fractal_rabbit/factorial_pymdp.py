"""Factorial pymdp model for synthetic latent-mode analysis.

The factors here are analysis constructs over generated waypoints. In
particular, the ``cognitive_proxy_mode`` factor is a synthetic reporting-process
proxy derived from gaps and bursts; it is not a claim about real cognition,
intent, or mental state.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .fractalrabbit import parse_waypoint_csv
from .inference import _entropy, run_fixed_transition_baseline
from .observations import discretize_observations
from .pomdp import _normalize_columns, build_pomdp
from .pymdp_agent import _configure_jax_runtime
from .types import (
    FactorialInferenceResult,
    FactorialPomdpModel,
    LatentModeSequence,
    ModelConfig,
    ObservationSequence,
)

FACTORIAL_PYMDP_SCHEMA = "active-fractal-rabbit-factorial-pymdp-analysis-v1"
FACTOR_NAMES = ("cell", "movement_mode", "cognitive_proxy_mode")
MOVEMENT_NAMES = ("stationary", "return_or_dwell", "exploratory_move")
COGNITIVE_PROXY_NAMES = ("routine_report", "report_burst", "report_gap")
FACTORIAL_OBSERVATIONS = (
    "location",
    "time_of_day",
    "speed",
    "report_burst",
    "cell_delta",
    "return_flag",
)
MODE_BOUNDARY = "synthetic cognitive-proxy states, not real cognition or intent"
FACTORIAL_FULL_POLICY_MAX_STEPS = 256
FACTORIAL_STATE_VALIDATION_MAX_STEPS = 96
FULL_POLICY_MODE = "full_policy"
STATE_TRANSITION_POLICY_MODE = "state_transition_large_sample"


def derive_latent_modes(sequence: ObservationSequence) -> LatentModeSequence:
    """Derive synthetic movement and reporting/cognitive-proxy labels.

    Movement labels describe generated waypoint transitions. Cognitive-proxy
    labels describe report timing texture only; they deliberately avoid any
    real-world mental-state interpretation.
    """

    n = len(sequence.observations)
    movement = np.zeros(n, dtype=int)
    cognitive = np.zeros(n, dtype=int)
    cell_delta = np.zeros(n, dtype=int)
    return_flags = np.zeros(n, dtype=int)
    positive_intervals = sequence.intervals_days[sequence.intervals_days > 0]
    long_gap_cutoff = (
        float(np.quantile(positive_intervals, 0.75)) if positive_intervals.size else 0.0
    )
    visited_by_traveller: dict[int, set[int]] = {}
    previous_cell_by_traveller: dict[int, int] = {}
    for idx, row in enumerate(sequence.observations):
        cell = int(sequence.spatial_states[idx])
        traveller = row.traveller_id
        visited = visited_by_traveller.setdefault(traveller, set())
        previous = previous_cell_by_traveller.get(traveller)
        returned = bool(visited and cell in visited)
        return_flags[idx] = int(returned)
        if previous is not None and previous != cell:
            cell_delta[idx] = 1

        if sequence.is_first_observation[idx] or previous == cell:
            movement[idx] = 0
        elif returned:
            movement[idx] = 1
        elif int(sequence.speed_categories[idx]) >= 2:
            movement[idx] = 2
        else:
            movement[idx] = 1

        if sequence.is_first_observation[idx]:
            cognitive[idx] = 0
        elif int(sequence.burst_categories[idx]) == 1:
            cognitive[idx] = 1
        elif (
            long_gap_cutoff > 0
            and float(sequence.intervals_days[idx]) >= long_gap_cutoff
        ):
            cognitive[idx] = 2
        else:
            cognitive[idx] = 0

        visited.add(cell)
        previous_cell_by_traveller[traveller] = cell
    return LatentModeSequence(
        movement_modes=movement,
        cognitive_proxy_modes=cognitive,
        cell_delta=cell_delta,
        return_flags=return_flags,
        movement_names=MOVEMENT_NAMES,
        cognitive_proxy_names=COGNITIVE_PROXY_NAMES,
    )


def build_factorial_pomdp(
    sequence: ObservationSequence,
    model_config: ModelConfig,
    include: np.ndarray | None = None,
) -> FactorialPomdpModel:
    """Build a three-factor generative model over cell, movement, and report mode."""

    modes = derive_latent_modes(sequence)
    if include is None:
        include = np.ones(len(sequence.observations), dtype=bool)
    include = np.asarray(include, dtype=bool)
    if include.shape[0] != len(sequence.observations):
        raise ValueError("include mask length must match the observation count")
    if not include.any():
        raise ValueError("include mask must keep at least one observation")
    factor_values = (
        np.asarray(sequence.spatial_states, dtype=int),
        modes.movement_modes,
        modes.cognitive_proxy_modes,
    )
    factor_sizes = (
        int(sequence.num_spatial_states),
        len(MOVEMENT_NAMES),
        len(COGNITIVE_PROXY_NAMES),
    )
    obs_values = (
        np.asarray(sequence.spatial_states, dtype=int),
        np.asarray(sequence.time_bins, dtype=int),
        np.asarray(sequence.speed_categories, dtype=int),
        np.asarray(sequence.burst_categories, dtype=int),
        modes.cell_delta,
        modes.return_flags,
    )
    obs_sizes = (
        int(sequence.num_spatial_states),
        int(sequence.num_time_bins),
        int(sequence.num_speed_categories),
        int(sequence.num_burst_categories),
        2,
        2,
    )
    a_dependencies = ((0,), (0, 2), (1, 0), (2,), (1,), (1,))
    b_dependencies = ((0, 1), (1, 2), (2,))
    A = tuple(
        _emission_tensor(
            obs,
            obs_size,
            factor_values,
            factor_sizes,
            deps,
            model_config.smoothing,
            include,
        )
        for obs, obs_size, deps in zip(obs_values, obs_sizes, a_dependencies)
    )
    cell_controls = model_config.controls
    mode_controls = model_config.factorial_controls
    B = (
        _transition_tensor(
            factor_values,
            factor_sizes,
            0,
            b_dependencies[0],
            cell_controls,
            model_config.smoothing,
            include,
            sequence.is_first_observation,
        ),
        _transition_tensor(
            factor_values,
            factor_sizes,
            1,
            b_dependencies[1],
            mode_controls,
            model_config.smoothing,
            include,
            sequence.is_first_observation,
        ),
        _transition_tensor(
            factor_values,
            factor_sizes,
            2,
            b_dependencies[2],
            mode_controls,
            model_config.smoothing,
            include,
            sequence.is_first_observation,
        ),
    )
    C = tuple(
        _preference_vector(size, name, model_config.preference_prior)
        for size, name in zip(obs_sizes, FACTORIAL_OBSERVATIONS)
    )
    D = tuple(
        _initial_prior(values[include], size, model_config.smoothing)
        for values, size in zip(factor_values, factor_sizes)
    )
    return FactorialPomdpModel(
        A=A,
        B=B,
        C=C,
        D=D,
        observation_names=FACTORIAL_OBSERVATIONS,
        factor_names=FACTOR_NAMES,
        factor_sizes=factor_sizes,
        A_dependencies=a_dependencies,
        B_dependencies=b_dependencies,
        control_names=(cell_controls, mode_controls, mode_controls),
        latent_modes=modes,
    )


def build_factorial_pymdp_agent(model: FactorialPomdpModel):
    """Construct the real pymdp Agent for the factorial model."""

    _configure_jax_runtime()
    from jax import numpy as jnp
    from pymdp.agent import Agent

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A JAX array is being set as static")
        return Agent(
            A=[jnp.asarray(matrix) for matrix in model.A],
            B=[jnp.asarray(tensor) for tensor in model.B],
            C=[jnp.asarray(vector) for vector in model.C],
            D=[jnp.asarray(vector) for vector in model.D],
            A_dependencies=[list(item) for item in model.A_dependencies],
            B_dependencies=[list(item) for item in model.B_dependencies],
            num_controls=[len(item) for item in model.control_names],
            control_fac_idx=list(range(len(model.factor_names))),
        )


def run_factorial_pymdp_inference(
    model: FactorialPomdpModel,
    sequence: ObservationSequence,
    *,
    policy_mode: str = FULL_POLICY_MODE,
) -> FactorialInferenceResult:
    """Run sequential multi-factor pymdp inference and diagnostic scoring."""

    _configure_jax_runtime()
    from jax import numpy as jnp

    if policy_mode not in {FULL_POLICY_MODE, STATE_TRANSITION_POLICY_MODE}:
        raise ValueError(f"unknown factorial policy_mode: {policy_mode}")
    agent = build_factorial_pymdp_agent(model)
    evidence = observation_indices(model, sequence)
    n_steps = len(evidence)
    posterior_by_factor = {
        name: np.zeros((n_steps, size), dtype=float)
        for name, size in zip(model.factor_names, model.factor_sizes)
    }
    n_policies = int(agent.policies.num_policies)
    policy_posterior = np.zeros((n_steps, n_policies), dtype=float)
    expected_free_energy = np.zeros((n_steps, n_policies), dtype=float)
    losses = np.zeros(n_steps, dtype=float)
    entropies = np.zeros(n_steps, dtype=float)
    vfe = np.zeros(n_steps, dtype=float)
    runtimes = np.zeros(n_steps, dtype=float)
    selected_actions = np.zeros((n_steps, len(model.factor_names)), dtype=int)

    policy_evaluation_steps = 0
    state_validation_indices = set(
        _sentinel_indices(n_steps, FACTORIAL_STATE_VALIDATION_MAX_STEPS)
    )
    state_validation_steps = 0
    uniform_policy = np.full(n_policies, 1.0 / n_policies, dtype=float)
    prior = [jnp.asarray(vector)[None] for vector in model.D]
    matched_delta = 0.0
    mean_field_exact_gap = 0.0
    loop_start = time.perf_counter()
    for idx, obs in enumerate(evidence):
        prior_np = [
            np.asarray(item).reshape(size)
            for item, size in zip(prior, model.factor_sizes)
        ]
        losses[idx] = -np.log(
            max(_prior_predictive_probability(model, prior_np, obs), 1e-12)
        )
        # Two distinct references, never conflated:
        #   * the analytic mean-field fixed point validates the released pymdp
        #     variational engine against an independent re-derivation of the SAME
        #     update equations (they must agree to numerical tolerance);
        #   * the exact joint marginals quantify the cost of factorizing the
        #     posterior under coupled likelihoods, reported honestly not hidden.
        mean_field_marginals = _mean_field_marginals_given_factorized_prior(
            model, prior_np, obs
        )
        exact_marginals = _exact_marginals_given_factorized_prior(model, prior_np, obs)
        mean_field_exact_gap = max(
            mean_field_exact_gap,
            max(
                float(np.max(np.abs(mf - ex)))
                for mf, ex in zip(mean_field_marginals, exact_marginals)
            ),
        )
        start = time.perf_counter()
        obs_arrays = [jnp.array([[value]], dtype=jnp.int32) for value in obs]
        if policy_mode == FULL_POLICY_MODE:
            qs, info = agent.infer_states(obs_arrays, prior, return_info=True)
            q_arrays = [
                np.asarray(q).reshape(size) for q, size in zip(qs, model.factor_sizes)
            ]
            q_pi, neg_efe = agent.infer_policies(qs)
            action = agent.sample_action(q_pi)
            prior = agent.update_empirical_prior(action, qs)
            policy_posterior[idx] = np.asarray(q_pi).reshape(n_policies)
            expected_free_energy[idx] = np.asarray(neg_efe).reshape(n_policies)
            selected_actions[idx] = np.asarray(action).reshape(-1)[
                : len(model.factor_names)
            ]
            matched_delta = max(
                matched_delta,
                max(
                    float(np.max(np.abs(q - mf)))
                    for q, mf in zip(q_arrays, mean_field_marginals)
                ),
            )
            vfe[idx] = float(np.asarray(info["vfe"]).reshape(-1)[-1])
            policy_evaluation_steps += 1
        else:
            q_arrays = [np.asarray(q, dtype=float) for q in mean_field_marginals]
            if idx in state_validation_indices:
                qs, info = agent.infer_states(obs_arrays, prior, return_info=True)
                pymdp_q_arrays = [
                    np.asarray(q).reshape(size)
                    for q, size in zip(qs, model.factor_sizes)
                ]
                matched_delta = max(
                    matched_delta,
                    max(
                        float(np.max(np.abs(q - mf)))
                        for q, mf in zip(pymdp_q_arrays, mean_field_marginals)
                    ),
                )
                vfe[idx] = float(np.asarray(info["vfe"]).reshape(-1)[-1])
                state_validation_steps += 1
            else:
                vfe[idx] = losses[idx]
            action_np = np.zeros(len(model.factor_names), dtype=int)
            next_prior = _predict_next_factor_priors(model, q_arrays, action_np)
            prior = [jnp.asarray(vector)[None] for vector in next_prior]
            policy_posterior[idx] = uniform_policy
            selected_actions[idx] = action_np
        runtimes[idx] = (time.perf_counter() - start) * 1000.0
        for name, q in zip(model.factor_names, q_arrays):
            posterior_by_factor[name][idx] = q
        entropies[idx] = sum(_entropy(q) for q in q_arrays)
    total_runtime = time.perf_counter() - loop_start

    mode_metrics = _mode_metrics(model, posterior_by_factor)
    metrics = {
        "factorial_pymdp_n_factors": float(len(model.factor_names)),
        "factorial_pymdp_n_joint_states": float(np.prod(model.factor_sizes)),
        "factorial_pymdp_n_policies": float(n_policies),
        "factorial_pymdp_inference_rows": float(n_steps),
        "factorial_pymdp_state_validation_steps": float(state_validation_steps),
        "factorial_pymdp_policy_evaluation_steps": float(policy_evaluation_steps),
        "factorial_pymdp_state_transition_prior_steps": float(
            n_steps - policy_evaluation_steps
        ),
        "factorial_pymdp_mean_predictive_loss": float(np.mean(losses)),
        "factorial_pymdp_mean_posterior_entropy": float(np.mean(entropies)),
        "factorial_pymdp_final_posterior_entropy": float(entropies[-1]),
        "factorial_pymdp_mean_policy_entropy": float(
            np.mean([_entropy(row) for row in policy_posterior])
        ),
        "factorial_pymdp_mean_vfe": float(np.mean(vfe)),
        "factorial_pymdp_matched_prior_max_abs_delta": float(matched_delta),
        "factorial_pymdp_mean_field_exact_gap": float(mean_field_exact_gap),
        "factorial_pymdp_total_runtime_s": float(total_runtime),
        "factorial_pymdp_median_step_runtime_ms": float(np.median(runtimes)),
    } | mode_metrics
    return FactorialInferenceResult(
        factor_posteriors=posterior_by_factor,
        predictive_loss=losses,
        posterior_entropy=entropies,
        policy_posterior=policy_posterior,
        expected_free_energy=expected_free_energy,
        variational_free_energy=vfe,
        step_runtime_ms=runtimes,
        selected_actions=selected_actions,
        metrics=metrics,
    )


def run_and_write_factorial_analysis(
    config_path: str | Path,
    out_path: str | Path,
    *,
    observations_path: str | Path | None = None,
) -> Path:
    """Run the factorial lane and write the manuscript-facing artifact."""

    config = load_config(config_path)
    obs_path = (
        Path(observations_path)
        if observations_path
        else _resolved_existing_observations(config, Path(config_path))
    )
    observations = parse_waypoint_csv(obs_path)
    sequence = discretize_observations(observations, config.inference)
    model = build_factorial_pomdp(sequence, config.inference.model)
    policy_mode = (
        FULL_POLICY_MODE
        if len(sequence.observations) <= FACTORIAL_FULL_POLICY_MAX_STEPS
        else STATE_TRANSITION_POLICY_MODE
    )
    result = run_factorial_pymdp_inference(model, sequence, policy_mode=policy_mode)
    single_metrics = _read_json(config.output_dir / "metrics.json", {})
    baseline = _transparent_baseline_metrics(
        sequence, config.inference.model.smoothing, config.inference.model.controls
    )
    stronger = _stronger_baseline_metrics(sequence)
    metrics = (
        result.metrics
        | baseline
        | stronger
        | _gain_metrics(result.metrics, single_metrics)
    )
    gate = _performance_gate(metrics, single_metrics)
    payload = {
        "schema": FACTORIAL_PYMDP_SCHEMA,
        "status": "completed",
        "lane": config.fractalrabbit.execution,
        "source_observations": str(obs_path),
        "hidden_factors": list(model.factor_names),
        "factor_sizes": list(model.factor_sizes),
        "observation_modalities": list(model.observation_names),
        "A_dependencies": [list(item) for item in model.A_dependencies],
        "B_dependencies": [list(item) for item in model.B_dependencies],
        "movement_mode_names": list(model.latent_modes.movement_names),
        "cognitive_proxy_mode_names": list(model.latent_modes.cognitive_proxy_names),
        "mode_boundary": MODE_BOUNDARY,
        "policy_mode": policy_mode,
        "policy_evaluation_steps": int(
            metrics["factorial_pymdp_policy_evaluation_steps"]
        ),
        "performance_gate": gate,
        "best_model": _best_model_name(metrics),
        "metrics": metrics,
        "tensor_summary": tensor_summary(model),
        "model_rows": _model_rows(metrics, single_metrics, gate),
        "caveat": _factorial_caveat(policy_mode),
    }
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def unavailable_factorial_analysis(reason: str) -> dict[str, object]:
    """Return a no-number unavailable payload for a lane that did not run."""

    return {
        "schema": FACTORIAL_PYMDP_SCHEMA,
        "status": "unavailable",
        "reason": reason,
        "performance_gate": "unavailable",
        "factorial_pymdp_mean_predictive_loss": "unavailable",
        "mode_boundary": MODE_BOUNDARY,
        "model_rows": [],
        "caveat": "No numeric factorial-pymdp result is emitted for an unavailable lane.",
    }


def observation_indices(
    model: FactorialPomdpModel, sequence: ObservationSequence
) -> list[tuple[int, ...]]:
    """Per-step observation tuples for the factorial model."""

    modes = model.latent_modes
    arrays = {
        "location": sequence.spatial_states,
        "time_of_day": sequence.time_bins,
        "speed": sequence.speed_categories,
        "report_burst": sequence.burst_categories,
        "cell_delta": modes.cell_delta,
        "return_flag": modes.return_flags,
    }
    return [
        tuple(int(arrays[name][idx]) for name in model.observation_names)
        for idx in range(len(sequence.observations))
    ]


def tensor_summary(model: FactorialPomdpModel) -> dict[str, object]:
    """Compact shape and normalization summary for factored tensors."""

    return {
        "factor_names": list(model.factor_names),
        "factor_sizes": list(model.factor_sizes),
        "joint_state_count": int(np.prod(model.factor_sizes)),
        "observation_modalities": list(model.observation_names),
        "A_shapes": [list(matrix.shape) for matrix in model.A],
        "B_shapes": [list(tensor.shape) for tensor in model.B],
        "D_shapes": [list(vector.shape) for vector in model.D],
        "A_dependencies": [list(item) for item in model.A_dependencies],
        "B_dependencies": [list(item) for item in model.B_dependencies],
        "A_column_sum_min": [float(np.min(matrix.sum(axis=0))) for matrix in model.A],
        "A_column_sum_max": [float(np.max(matrix.sum(axis=0))) for matrix in model.A],
        "B_column_sum_min": [float(np.min(tensor.sum(axis=0))) for tensor in model.B],
        "B_column_sum_max": [float(np.max(tensor.sum(axis=0))) for tensor in model.B],
        "D_sum": [float(vector.sum()) for vector in model.D],
    }


def _emission_tensor(
    observations: np.ndarray,
    obs_size: int,
    factor_values: tuple[np.ndarray, ...],
    factor_sizes: tuple[int, ...],
    dependencies: tuple[int, ...],
    smoothing: float,
    include: np.ndarray,
) -> np.ndarray:
    shape = (obs_size,) + tuple(factor_sizes[idx] for idx in dependencies)
    tensor = np.full(shape, smoothing, dtype=float)
    for row_idx, observed in enumerate(observations):
        if include[row_idx]:
            key = (int(observed),) + tuple(
                int(factor_values[idx][row_idx]) for idx in dependencies
            )
            tensor[key] += 1.0
    return _normalize_columns(tensor)


def _transition_tensor(
    factor_values: tuple[np.ndarray, ...],
    factor_sizes: tuple[int, ...],
    target_factor: int,
    dependencies: tuple[int, ...],
    controls: tuple[str, ...],
    smoothing: float,
    include: np.ndarray,
    first_observation: np.ndarray,
) -> np.ndarray:
    shape = (
        (factor_sizes[target_factor],)
        + tuple(factor_sizes[idx] for idx in dependencies)
        + (len(controls),)
    )
    tensor = np.full(shape, smoothing, dtype=float)
    target_values = factor_values[target_factor]
    for control_idx, control in enumerate(controls):
        if control in {"empirical_continuation"}:
            for idx in range(len(target_values) - 1):
                if include[idx] and include[idx + 1] and not first_observation[idx + 1]:
                    key = (
                        (int(target_values[idx + 1]),)
                        + tuple(int(factor_values[dep][idx]) for dep in dependencies)
                        + (control_idx,)
                    )
                    tensor[key] += 1.0
        elif control in {"identity", "mode_persistence"}:
            if target_factor not in dependencies:
                tensor[..., control_idx] += 1.0
                continue
            dep_axis = dependencies.index(target_factor)
            for dep_state in np.ndindex(*(factor_sizes[idx] for idx in dependencies)):
                target_state = dep_state[dep_axis]
                tensor[(target_state,) + dep_state + (control_idx,)] += 1.0
        elif control in {"high_entropy_exploration", "mode_reset"}:
            tensor[..., control_idx] += 1.0
        else:
            raise ValueError(f"unknown factorial control: {control}")
    return _normalize_columns(tensor)


def _initial_prior(values: np.ndarray, size: int, smoothing: float) -> np.ndarray:
    prior = np.full(size, smoothing, dtype=float)
    for value in values:
        prior[int(value)] += 1.0
    return prior / prior.sum()


def _preference_vector(
    size: int, observation_name: str, preference_prior: str
) -> np.ndarray:
    vector = np.zeros(size, dtype=float)
    if preference_prior == "diagnostic_uncertainty":
        vector[:] = 0.0
    elif preference_prior == "diagnostic_gap" and observation_name == "report_burst":
        vector[min(size - 1, 1)] = 0.25
    return vector


def _prior_predictive_probability(
    model: FactorialPomdpModel,
    priors: list[np.ndarray],
    observation: tuple[int, ...],
) -> float:
    probability = 0.0
    for state in np.ndindex(*model.factor_sizes):
        p_state = float(
            np.prod([priors[idx][state[idx]] for idx in range(len(priors))])
        )
        probability += p_state * _likelihood_at_state(model, observation, state)
    return probability


def _predict_next_factor_priors(
    model: FactorialPomdpModel,
    posterior_by_factor: list[np.ndarray],
    actions: np.ndarray,
) -> list[np.ndarray]:
    """Roll a factorized posterior forward through the declared B tensors.

    The large-sample factorial lane uses this after real pymdp state inference
    to avoid per-step policy enumeration. It keeps the sequential prior tied to
    the same transition tensors used by the full policy lane.
    """

    next_priors: list[np.ndarray] = []
    for factor_idx, (tensor, deps, size) in enumerate(
        zip(model.B, model.B_dependencies, model.factor_sizes)
    ):
        action = int(actions[factor_idx])
        if action < 0 or action >= len(model.control_names[factor_idx]):
            raise ValueError(
                f"action index out of range for factor {factor_idx}: {action}"
            )
        transition = np.asarray(tensor[..., action], dtype=float)
        prior = np.zeros(size, dtype=float)
        dep_sizes = tuple(model.factor_sizes[dep] for dep in deps)
        for dep_state in np.ndindex(*dep_sizes):
            weight = 1.0
            for position, dep in enumerate(deps):
                weight *= float(posterior_by_factor[dep][dep_state[position]])
            prior += transition[(slice(None),) + dep_state] * weight
        next_priors.append(_safe_normalize(prior))
    return next_priors


def _sentinel_indices(n_steps: int, max_steps: int) -> tuple[int, ...]:
    if n_steps <= 0 or max_steps <= 0:
        return ()
    if n_steps <= max_steps:
        return tuple(range(n_steps))
    return tuple(
        int(item)
        for item in np.unique(np.linspace(0, n_steps - 1, max_steps, dtype=int))
    )


def _exact_marginals_given_factorized_prior(
    model: FactorialPomdpModel,
    priors: list[np.ndarray],
    observation: tuple[int, ...],
) -> list[np.ndarray]:
    joint = np.zeros(model.factor_sizes, dtype=float)
    for state in np.ndindex(*model.factor_sizes):
        p_state = float(
            np.prod([priors[idx][state[idx]] for idx in range(len(priors))])
        )
        joint[state] = p_state * _likelihood_at_state(model, observation, state)
    total = float(joint.sum())
    if total <= 0:
        joint = np.ones(model.factor_sizes, dtype=float) / np.prod(model.factor_sizes)
    else:
        joint /= total
    marginals = []
    for factor_idx in range(len(model.factor_sizes)):
        axes = tuple(
            axis for axis in range(len(model.factor_sizes)) if axis != factor_idx
        )
        marginals.append(joint.sum(axis=axes))
    return marginals


def _mean_field_marginals_given_factorized_prior(
    model: FactorialPomdpModel,
    priors: list[np.ndarray],
    observation: tuple[int, ...],
    iters: int = 64,
) -> list[np.ndarray]:
    """Independent coordinate-ascent mean-field fixed point for the posterior.

    This is a dependency-light re-derivation of the SAME mean-field variational
    update that pymdp's VANILLA ``infer_states`` performs: each factor marginal
    is set to ``prior_i * exp(E_{q_{-i}}[log A])`` and the factors are cycled to
    convergence. It exists to validate the released pymdp engine against an
    independent implementation — the two must agree to numerical tolerance.

    It is deliberately distinct from :func:`_exact_marginals_given_factorized_prior`.
    Under coupled likelihoods (e.g. an observation depending on two latent
    factors) the exact posterior is non-factorized, so the mean-field fixed point
    differs from the exact marginals by a measurable gap. That gap is the honest
    cost of factorization and is reported as ``factorial_pymdp_mean_field_exact_gap``,
    never folded into the pymdp-agreement check.
    """

    log_potentials = [
        (np.log(np.asarray(matrix)[int(observed)] + 1e-300), deps)
        for matrix, deps, observed in zip(model.A, model.A_dependencies, observation)
    ]
    log_priors = [np.log(np.asarray(p, dtype=float) + 1e-300) for p in priors]
    q = [_safe_normalize(np.asarray(p, dtype=float)) for p in priors]
    n_factors = len(model.factor_sizes)
    for _ in range(iters):
        for factor in range(n_factors):
            log_q = log_priors[factor].copy()
            for log_matrix, deps in log_potentials:
                if factor not in deps:
                    continue
                contracted = log_matrix
                offset = 0
                for position, dep in enumerate(deps):
                    if dep == factor:
                        continue
                    contracted = np.tensordot(
                        contracted, q[dep], axes=([position - offset], [0])
                    )
                    offset += 1
                log_q = log_q + contracted
            q[factor] = _safe_normalize(np.exp(log_q - log_q.max()))
    return q


def _safe_normalize(vector: np.ndarray) -> np.ndarray:
    total = float(vector.sum())
    if total <= 0:
        return np.ones_like(vector) / vector.size
    return vector / total


def _likelihood_at_state(
    model: FactorialPomdpModel,
    observation: tuple[int, ...],
    state: tuple[int, ...],
) -> float:
    likelihood = 1.0
    for matrix, deps, observed in zip(model.A, model.A_dependencies, observation):
        likelihood *= float(
            matrix[(int(observed),) + tuple(state[idx] for idx in deps)]
        )
    return likelihood


def _mode_metrics(
    model: FactorialPomdpModel, posteriors: dict[str, np.ndarray]
) -> dict[str, float]:
    movement = posteriors["movement_mode"]
    cognitive = posteriors["cognitive_proxy_mode"]
    cell = posteriors["cell"]
    return {
        "factorial_pymdp_cell_mean_entropy": float(
            np.mean([_entropy(row) for row in cell])
        ),
        "factorial_pymdp_movement_mode_mean_entropy": float(
            np.mean([_entropy(row) for row in movement])
        ),
        "factorial_pymdp_cognitive_proxy_mean_entropy": float(
            np.mean([_entropy(row) for row in cognitive])
        ),
        "factorial_pymdp_movement_mode_stability": _stability(
            np.argmax(movement, axis=1)
        ),
        "factorial_pymdp_cognitive_proxy_stability": _stability(
            np.argmax(cognitive, axis=1)
        ),
        "factorial_pymdp_mean_movement_mode_confidence": float(
            np.mean(np.max(movement, axis=1))
        ),
        "factorial_pymdp_mean_cognitive_proxy_confidence": float(
            np.mean(np.max(cognitive, axis=1))
        ),
    }


def _transparent_baseline_metrics(
    sequence: ObservationSequence,
    smoothing: float,
    controls: tuple[str, ...],
) -> dict[str, float]:
    single = build_pomdp(sequence, smoothing, controls)
    baseline = run_fixed_transition_baseline(single, sequence)
    return {
        "single_reference_fixed_transition_loss": float(
            baseline.metrics["baseline_mean_predictive_loss"]
        ),
        "single_reference_fixed_transition_entropy": float(
            baseline.metrics["baseline_mean_posterior_entropy"]
        ),
    }


def _stronger_baseline_metrics(sequence: ObservationSequence) -> dict[str, float]:
    """Score a fixed, near-deterministic persistence/Markov ladder.

    This intentionally duplicates the *shape* of
    ``sequence_baselines._persistence_loss`` / ``_context_markov_loss`` with a
    different, deliberately harder-to-beat convention: it uses a fixed
    ``1e-6`` smoothing constant (not the caller-supplied
    ``config.inference.model.smoothing``) and it scores every observation,
    including first-observations-per-traveller (via ``base_rate`` rather than
    skipping them as ``sequence_baselines`` does). The two are a "hard"
    near-deterministic persistence null here vs. a config-driven smoothed
    persistence null there; they are expected to disagree and neither is a
    bug in the other. See ``sequence_baselines._persistence_loss`` for the
    config-smoothed counterpart.
    """
    states = np.asarray(sequence.spatial_states, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    n_states = int(sequence.num_spatial_states)
    base_counts = np.bincount(states, minlength=n_states).astype(float) + 0.5
    base_rate = base_counts / base_counts.sum()
    persistence_losses = []
    switching_losses = []
    particle_losses = []
    transition_counts = np.ones((n_states, n_states), dtype=float) * 0.5
    for idx, state in enumerate(states):
        if idx == 0 or first[idx]:
            persistence = base_rate
        else:
            persistence = np.full(n_states, 1e-6, dtype=float)
            persistence[int(states[idx - 1])] = 1.0
            persistence /= persistence.sum()
        persistence_losses.append(-np.log(max(float(persistence[int(state)]), 1e-12)))
        if idx == 0 or first[idx]:
            markov = base_rate
        else:
            previous = int(states[idx - 1])
            markov = (
                transition_counts[:, previous] / transition_counts[:, previous].sum()
            )
        switching_losses.append(-np.log(max(float(markov[int(state)]), 1e-12)))
        particle = 0.65 * persistence + 0.35 * markov
        particle /= particle.sum()
        particle_losses.append(-np.log(max(float(particle[int(state)]), 1e-12)))
        if idx > 0 and not first[idx]:
            transition_counts[int(state), int(states[idx - 1])] += 1.0
    return {
        "switching_hmm_mean_predictive_loss": float(np.mean(switching_losses)),
        "particle_state_space_mean_predictive_loss": float(np.mean(particle_losses)),
        "persistence_mean_predictive_loss": float(np.mean(persistence_losses)),
    }


def _gain_metrics(
    metrics: dict[str, float], single_metrics: dict[str, Any]
) -> dict[str, float]:
    pymdp_loss = single_metrics.get("pymdp_mean_predictive_loss")
    reference_loss = single_metrics.get("mean_predictive_loss")
    result: dict[str, float] = {}
    if isinstance(pymdp_loss, int | float):
        result["factorial_pymdp_gain_vs_single_pymdp"] = (
            float(pymdp_loss) - metrics["factorial_pymdp_mean_predictive_loss"]
        )
    if isinstance(reference_loss, int | float):
        result["factorial_pymdp_gain_vs_reference"] = (
            float(reference_loss) - metrics["factorial_pymdp_mean_predictive_loss"]
        )
    return result


def _performance_gate(metrics: dict[str, float], single_metrics: dict[str, Any]) -> str:
    pymdp_loss = single_metrics.get("pymdp_mean_predictive_loss")
    if not isinstance(pymdp_loss, int | float):
        return "pymdp_baseline_unavailable"
    if metrics["factorial_pymdp_mean_predictive_loss"] < float(pymdp_loss) - 0.01:
        return "factorial_pymdp_cleared_current_lane"
    return "factorial_pymdp_not_performance_leading"


def _best_model_name(metrics: dict[str, float]) -> str:
    losses = {
        "factorial_pymdp": metrics["factorial_pymdp_mean_predictive_loss"],
        "switching_hmm": metrics["switching_hmm_mean_predictive_loss"],
        "particle_state_space": metrics["particle_state_space_mean_predictive_loss"],
        "persistence": metrics["persistence_mean_predictive_loss"],
    }
    return min(losses, key=losses.get)


def _model_rows(
    metrics: dict[str, float], single_metrics: dict[str, Any], gate: str
) -> list[dict[str, object]]:
    rows = [
        {
            "model": "reference_filter",
            "implemented": True,
            "mean_predictive_loss": _maybe_number(
                single_metrics.get("mean_predictive_loss")
            ),
            "interpretability_surface": "posterior trace and tensor provenance",
        },
        {
            "model": "single_factor_pymdp",
            "implemented": "pymdp_mean_predictive_loss" in single_metrics,
            "mean_predictive_loss": _maybe_number(
                single_metrics.get("pymdp_mean_predictive_loss")
            ),
            "interpretability_surface": "VFE, EFE, policy posterior",
        },
        {
            "model": "factorial_pymdp",
            "implemented": True,
            "mean_predictive_loss": metrics["factorial_pymdp_mean_predictive_loss"],
            "mean_posterior_entropy": metrics["factorial_pymdp_mean_posterior_entropy"],
            "performance_gate": gate,
            "interpretability_surface": "cell, movement-mode, cognitive-proxy posteriors plus VFE/EFE",
        },
        {
            "model": "switching_hmm",
            "implemented": True,
            "mean_predictive_loss": metrics["switching_hmm_mean_predictive_loss"],
            "interpretability_surface": "online transition regime baseline",
        },
        {
            "model": "particle_state_space",
            "implemented": True,
            "mean_predictive_loss": metrics[
                "particle_state_space_mean_predictive_loss"
            ],
            "interpretability_surface": "transparent persistence/Markov mixture state-space baseline",
        },
        {
            "model": "neural_sequence",
            "implemented": False,
            "status": "blocked_until_explicit_prequential_training_gate",
            "interpretability_surface": "future comparator only",
        },
    ]
    return rows


def _factorial_caveat(policy_mode: str) -> str:
    base = (
        "Factorial active inference adds interpretable synthetic latent-mode diagnostics. "
        "It is not evidence of real cognition, intent, or state-of-the-art predictive performance unless the gate clears."
    )
    if policy_mode == STATE_TRANSITION_POLICY_MODE:
        return (
            base
            + " Expanded-sample runs score every row with the independent mean-field solver and declared transition "
            "tensors, validate real pymdp state inference on deterministic sentinel rows, and reserve exhaustive "
            "factorial policy/EFE enumeration for the small-sample oracle because the promoted factorial claim is a "
            "latent-state diagnostic."
        )
    return base


def _resolved_existing_observations(config, config_path: Path) -> Path:
    if config.fractalrabbit.execution == "fixture":
        assert config.fractalrabbit.fixture_csv is not None
        return config.fractalrabbit.fixture_csv
    existing = config.output_dir / "observations.csv"
    if existing.exists():
        return existing
    raise FileNotFoundError(
        f"external observations for {config_path} are unavailable; run the external lane first"
    )


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _stability(labels: np.ndarray) -> float:
    if labels.size <= 1:
        return 1.0
    return float(np.mean(labels[1:] == labels[:-1]))


def _maybe_number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None
