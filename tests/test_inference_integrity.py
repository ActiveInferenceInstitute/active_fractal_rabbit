"""Analytic filter-semantics tests.

The integration suite validates shapes, normalization, and artifact
structure; a wrong likelihood rule or transition direction can hide behind
downstream normalization and still pass those gates. These tests pin the
filter's computational semantics directly: an independent mirror
implementation written in this file must reproduce the production filter's
posteriors exactly, and each primitive (likelihood product, transition
direction, softmax sign, entropy anchors) is checked against hand-computed
values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.inference import (
    _entropy,
    _likelihood,
    _softmax,
    run_latent_inference,
)
from active_fractal_rabbit.objectives import reconstruct_prior_trace
from active_fractal_rabbit.observations import discretize_observations
from active_fractal_rabbit.pomdp import build_pomdp, observation_indices
from active_fractal_rabbit.types import PomdpModel

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "fractalrabbit_sample.csv"


def _model_sequence():
    config = load_config(ROOT / "configs" / "default.yaml")
    rows = parse_waypoint_csv(FIXTURE)
    sequence = discretize_observations(rows, config.inference)
    model = build_pomdp(
        sequence, config.inference.smoothing, config.inference.model.controls
    )
    return model, sequence


def _toy_model() -> PomdpModel:
    A0 = np.array([[0.9, 0.2], [0.1, 0.8]])
    A1 = np.array([[0.6, 0.3], [0.4, 0.7]])
    B = np.zeros((2, 2, 1))
    B[:, 0, 0] = [0.3, 0.7]
    B[:, 1, 0] = [0.5, 0.5]
    return PomdpModel(
        A=(A0, A1),
        B=B,
        C=(np.zeros(2), np.zeros(2)),
        D=np.array([0.5, 0.5]),
        observation_names=("m0", "m1"),
        control_names=("only",),
    )


def test_likelihood_is_product_of_modality_rows() -> None:
    """The joint likelihood is the elementwise product of A_m[o_m, :] rows."""

    model = _toy_model()
    observation = (0, 1)
    expected = np.array([0.9 * 0.4, 0.2 * 0.7])
    assert np.allclose(_likelihood(model, observation), expected)


def test_posterior_matches_hand_computed_bayes() -> None:
    """One full Bayes step on the toy model, posterior computed by hand."""

    model = _toy_model()
    likelihood = _likelihood(model, (0, 1))  # [0.36, 0.14]
    prior = model.D  # [0.5, 0.5]
    joint = likelihood * prior  # [0.18, 0.07]
    posterior = joint / joint.sum()
    assert posterior[0] == pytest.approx(0.18 / 0.25)
    assert posterior[1] == pytest.approx(0.07 / 0.25)


def test_transition_applies_b_columns_not_rows() -> None:
    """prior_next = B[:, :, a] @ posterior: a delta posterior selects a COLUMN."""

    model = _toy_model()
    delta_on_state_0 = np.array([[1.0, 0.0]])
    uniform_policy = np.array([[1.0]])
    priors = reconstruct_prior_trace(
        model,
        np.vstack([delta_on_state_0, delta_on_state_0]),
        np.vstack([uniform_policy, uniform_policy]),
    )
    # Step 1 prior is B[:, 0, 0] = [0.3, 0.7] — column 0, not row 0 ([0.3, 0.5]).
    assert np.allclose(priors[1], [0.3, 0.7])


def test_softmax_negation_makes_lowest_score_most_probable() -> None:
    scores = np.array([1.0, 3.0])
    policy = _softmax(-scores)
    assert policy[0] > policy[1]
    assert policy.sum() == pytest.approx(1.0)


def test_policy_posterior_argmax_is_proxy_argmin_on_fixture() -> None:
    model, sequence = _model_sequence()
    inference = run_latent_inference(model, sequence)
    for step in range(inference.policy_posterior.shape[0]):
        assert int(np.argmax(inference.policy_posterior[step])) == int(
            np.argmin(inference.policy_proxy_score[step])
        )


def test_entropy_anchors() -> None:
    assert _entropy(np.ones(4) / 4) == pytest.approx(np.log(4), abs=1e-6)
    assert _entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-6)


def test_mirror_filter_reproduces_production_posteriors() -> None:
    """An independent forward-filter implementation must match exactly.

    The mirror recursion below is written from the model definition, not
    copied from inference.py; agreement to float64 precision on the real
    fixture pins likelihood structure, normalization, action selection, and
    transition direction simultaneously.
    """

    model, sequence = _model_sequence()
    production = run_latent_inference(model, sequence)
    evidence = observation_indices(model, sequence)

    prior = model.D.copy()
    for idx, observation in enumerate(evidence):
        likelihood = np.ones(model.D.size)
        for matrix, obs_index in zip(model.A, observation):
            likelihood = likelihood * matrix[obs_index, :]
        predictive = max(float(likelihood @ prior), 1e-12)
        posterior = likelihood * prior
        posterior = posterior / posterior.sum()
        assert np.allclose(posterior, production.posterior_states[idx], atol=1e-12)
        assert -np.log(predictive) == pytest.approx(
            production.predictive_loss[idx], abs=1e-12
        )
        chosen = int(np.argmax(production.policy_posterior[idx]))
        prior = model.B[:, :, chosen] @ posterior
        prior = prior / prior.sum()
