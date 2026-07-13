"""Real pymdp (JAX) agent lane.

This module runs the actual ``inferactively-pymdp`` ``Agent`` over the same
discretized observation sequence consumed by the reference filter in
``inference.py``. Both lanes are emitted side by side: the reference filter
remains the transparent, dependency-light implementation, while this lane
provides the canonical pymdp state inference, policy posterior, and expected
free energy from the released library rather than a re-derivation.

The agreement between the two state-posterior traces is itself a validation
artifact: with a single latent factor and shared tensors, fixed-point
iteration in pymdp should converge to the same exact posterior the reference
filter computes analytically.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from .pomdp import observation_indices
from .types import ObservationSequence, PomdpModel


@dataclass(frozen=True)
class PymdpResult:
    """Per-step outputs of the sequential pymdp agent loop."""

    posterior_states: np.ndarray
    policy_posterior: np.ndarray
    expected_free_energy: np.ndarray
    predictive_loss: np.ndarray
    posterior_entropy: np.ndarray
    variational_free_energy: np.ndarray
    step_runtime_ms: np.ndarray
    metrics: dict[str, float]


def build_pymdp_agent(model: PomdpModel):
    """Construct a batched JAX pymdp Agent from the project's POMDP arrays."""

    import warnings

    _configure_jax_runtime()
    from jax import numpy as jnp
    from pymdp.agent import Agent

    with warnings.catch_warnings():
        # pymdp 1.0.2 stores C/E as equinox static fields and warns when they
        # are JAX arrays. Benign for this read-only agent: the matched-prior
        # agreement test pins inference correctness regardless.
        warnings.filterwarnings("ignore", message="A JAX array is being set as static")
        return Agent(
            A=[jnp.asarray(matrix)[None] for matrix in model.A],
            B=[jnp.asarray(model.B)[None]],
            C=[jnp.asarray(vector)[None] for vector in model.C],
            D=[jnp.asarray(model.D)[None]],
        )


def run_pymdp_inference(
    model: PomdpModel,
    sequence: ObservationSequence,
    reference_posterior: np.ndarray | None = None,
    *,
    validation_indices: tuple[int, ...] | None = None,
) -> PymdpResult:
    """Run the sequential pymdp agent loop over the full observation sequence.

    Per step: variational state inference under the current empirical prior,
    policy inference (policy posterior and expected free energy), deterministic
    action selection from the policy posterior, and empirical-prior update
    through the selected control's transition slice. Predictive loss uses the
    same formula as the reference filter (negative log prior predictive of the
    actual observation) and is identical at step 0; the traces diverge
    thereafter by construction, because each lane evolves its own empirical
    prior through its own action-selection rule.
    """

    import time

    _configure_jax_runtime()
    from jax import numpy as jnp

    agent = build_pymdp_agent(model)
    full_evidence = observation_indices(model, sequence)
    source_rows = len(full_evidence)
    if validation_indices is None:
        selected_indices = tuple(range(source_rows))
    else:
        selected_indices = tuple(sorted(set(int(item) for item in validation_indices)))
        if not selected_indices:
            raise ValueError("validation_indices must select at least one row")
        if selected_indices[0] < 0 or selected_indices[-1] >= source_rows:
            raise ValueError("validation_indices contain an out-of-range row")
    evidence = [full_evidence[idx] for idx in selected_indices]
    n_steps = len(evidence)
    n_states = model.D.size
    n_controls = model.B.shape[2]

    posterior = np.zeros((n_steps, n_states), dtype=float)
    policy_posterior = np.zeros((n_steps, n_controls), dtype=float)
    expected_free_energy = np.zeros((n_steps, n_controls), dtype=float)
    losses = np.zeros(n_steps, dtype=float)
    entropies = np.zeros(n_steps, dtype=float)
    vfe = np.zeros(n_steps, dtype=float)
    step_runtime_ms = np.zeros(n_steps, dtype=float)

    exact_inference_delta = 0.0
    loop_start = time.perf_counter()
    prior = [jnp.asarray(model.D)[None]]
    for idx, obs in enumerate(evidence):
        prior_np = np.asarray(prior[0]).reshape(n_states)
        likelihood = np.ones(n_states, dtype=float)
        for matrix, index in zip(model.A, obs):
            likelihood *= matrix[index, :]
        losses[idx] = -np.log(max(float(likelihood @ prior_np), 1e-12))
        joint = likelihood * prior_np
        exact_posterior = (
            joint / joint.sum() if joint.sum() > 0 else np.ones(n_states) / n_states
        )

        step_start = time.perf_counter()
        obs_arrays = [jnp.array([[index]], dtype=jnp.int32) for index in obs]
        qs, info = agent.infer_states(obs_arrays, prior, return_info=True)
        q_pi, neg_efe = agent.infer_policies(qs)
        action = jnp.argmax(q_pi, axis=-1, keepdims=True).astype(jnp.int32)
        new_prior = agent.update_empirical_prior(action, qs)
        prior = [jnp.asarray(np.asarray(new_prior[0]).reshape(1, n_states))]
        step_runtime_ms[idx] = (time.perf_counter() - step_start) * 1000.0

        vfe[idx] = float(np.asarray(info["vfe"]).reshape(-1)[-1])
        qs_np = np.asarray(qs[0]).reshape(n_states)
        exact_inference_delta = max(
            exact_inference_delta, float(np.max(np.abs(qs_np - exact_posterior)))
        )
        posterior[idx] = qs_np
        policy_posterior[idx] = np.asarray(q_pi).reshape(n_controls)
        expected_free_energy[idx] = np.asarray(neg_efe).reshape(n_controls)
        entropies[idx] = float(-np.sum(qs_np * np.log(qs_np + 1e-12)))
    total_runtime_s = time.perf_counter() - loop_start

    metrics = {
        # Matched-prior validation: pymdp variational inference vs the exact
        # Bayes posterior under the SAME prior and tensors at every step.
        "pymdp_exact_inference_max_abs_delta": exact_inference_delta,
        "pymdp_source_observation_rows": float(source_rows),
        "pymdp_inference_rows": float(n_steps),
        "pymdp_validation_fraction": float(n_steps / source_rows)
        if source_rows
        else 0.0,
        "pymdp_mean_predictive_loss": float(np.mean(losses)),
        "pymdp_mean_posterior_entropy": float(np.mean(entropies)),
        "pymdp_final_posterior_entropy": float(entropies[-1]),
        "pymdp_mean_policy_entropy": float(
            np.mean([-np.sum(row * np.log(row + 1e-12)) for row in policy_posterior])
        ),
        # Variational free energy reported by pymdp itself, per step (final
        # fixed-point iterate). Runtime is emitted rather than hidden because
        # the real pymdp lane is deliberately heavier than the reference filter.
        "pymdp_mean_vfe": float(np.mean(vfe)),
        "pymdp_final_vfe": float(vfe[-1]),
        "pymdp_total_runtime_s": float(total_runtime_s),
        "pymdp_median_step_runtime_ms": float(np.median(step_runtime_ms)),
        "pymdp_max_step_runtime_ms": float(np.max(step_runtime_ms)),
    }
    if reference_posterior is not None:
        if reference_posterior.shape[0] == source_rows:
            reference_posterior = reference_posterior[list(selected_indices)]
        if reference_posterior.shape != posterior.shape:
            raise ValueError(
                "reference posterior shape does not match pymdp posterior shape"
            )
        metrics["pymdp_reference_max_abs_posterior_delta"] = float(
            np.max(np.abs(reference_posterior - posterior))
        )
    return PymdpResult(
        posterior_states=posterior,
        policy_posterior=policy_posterior,
        expected_free_energy=expected_free_energy,
        predictive_loss=losses,
        posterior_entropy=entropies,
        variational_free_energy=vfe,
        step_runtime_ms=step_runtime_ms,
        metrics=metrics,
    )


def _configure_jax_runtime() -> None:
    """Keep the small synthetic benchmark reproducible on workstation CPUs.

    The project compares pymdp's inference semantics, not accelerator
    compilation throughput. Disabling JIT by default avoids repeated
    multi-minute XLA compilation in the expanded fixture and test suite while
    still running the released pymdp/JAX implementation. Developers can opt
    back into JIT explicitly for runtime experiments.
    """

    if os.environ.get("ACTIVE_FRACTAL_RABBIT_ENABLE_JAX_JIT") == "1":
        return
    from jax import config as jax_config

    jax_config.update("jax_disable_jit", True)
