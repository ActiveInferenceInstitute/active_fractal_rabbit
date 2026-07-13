"""Objectives lane: decomposition invariants, calibration math, chance floors.

The directional expectations here are pre-registered hypotheses:
H1 (theorem): every per-modality marginal loss <= the joint loss per step.
H2: the location-observing model is strictly better calibrated (lower Brier)
    than the deep-latent model on the same data.
H3: both joint-loss skill scores beat the uniform-chance floor on the fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.inference import run_latent_inference
from active_fractal_rabbit.objectives import (
    base_rate_brier,
    chance_floors,
    lane_objectives,
    per_modality_losses,
    persistence_brier,
    reconstruct_prior_trace,
    recovery_calibration,
    skill_score,
)
from active_fractal_rabbit.observations import discretize_observations
from active_fractal_rabbit.pipeline import run_pipeline_from_path
from active_fractal_rabbit.pomdp import build_pomdp, observation_indices

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "fractalrabbit_sample.csv"
DEEP_LATENT = ("time_of_day", "speed", "report_burst")


def _model_sequence(modalities=None):
    config = load_config(ROOT / "configs" / "default.yaml")
    rows = parse_waypoint_csv(FIXTURE)
    sequence = discretize_observations(rows, config.inference)
    kwargs = {} if modalities is None else {"modalities": modalities}
    model = build_pomdp(
        sequence, config.inference.smoothing, config.inference.model.controls, **kwargs
    )
    return model, sequence


def test_prior_reconstruction_binds_exactly_to_reference_loss() -> None:
    """The reconstructed prior trace must reproduce the filter's own loss."""

    model, sequence = _model_sequence()
    inference = run_latent_inference(model, sequence)
    metrics, _ = lane_objectives(
        model,
        sequence,
        inference.posterior_states,
        inference.policy_posterior,
        inference.predictive_loss,
        "reference",
    )
    assert metrics["reference_per_modality_binding_max_delta"] < 1e-9


def test_marginal_loss_never_exceeds_joint_loss_per_step() -> None:
    """H1 theorem: p(o^m) >= p(o_joint), so marginal loss <= joint loss, every step."""

    model, sequence = _model_sequence()
    inference = run_latent_inference(model, sequence)
    priors = reconstruct_prior_trace(
        model, inference.posterior_states, inference.policy_posterior
    )
    evidence = observation_indices(model, sequence)
    for idx, observation in enumerate(evidence):
        prior = priors[idx]
        joint_likelihood = np.ones(model.D.size)
        for matrix, obs_index in zip(model.A, observation):
            joint_likelihood *= matrix[obs_index, :]
        joint = -np.log(max(float(joint_likelihood @ prior), 1e-12))
        for matrix, obs_index in zip(model.A, observation):
            marginal = -np.log(max(float(matrix[obs_index, :] @ prior), 1e-12))
            assert marginal <= joint + 1e-9


def test_binding_gate_rejects_wrong_priors() -> None:
    """Negative control: a uniform prior trace must be refused, not silently used."""

    model, sequence = _model_sequence()
    inference = run_latent_inference(model, sequence)
    wrong = np.full_like(inference.posterior_states, 1.0 / model.D.size)
    with pytest.raises(ValueError, match="does not reproduce"):
        per_modality_losses(model, sequence, wrong, inference.predictive_loss)


def test_per_modality_split_certified_term_by_term() -> None:
    """The decomposition is verified per modality, not only through the joint binding.

    The binding gate certifies the reconstructed prior reproduces the JOINT loss;
    this independently recomputes each modality's mean marginal loss from
    A_m[o_m, :] @ prior and matches the emitted per-modality term, closing the
    'joint-only certification' gap.
    """

    model, sequence = _model_sequence()
    inference = run_latent_inference(model, sequence)
    priors = reconstruct_prior_trace(
        model, inference.posterior_states, inference.policy_posterior
    )
    decomposition = per_modality_losses(
        model, sequence, priors, inference.predictive_loss
    )
    evidence = observation_indices(model, sequence)
    independent: dict[str, list[float]] = {name: [] for name in model.observation_names}
    for idx, observation in enumerate(evidence):
        prior = priors[idx]
        for m, (matrix, obs_index) in enumerate(zip(model.A, observation)):
            p = max(float(matrix[obs_index, :] @ prior), 1e-12)
            independent[model.observation_names[m]].append(-np.log(p))
    for name in model.observation_names:
        assert decomposition[f"per_modality_loss_{name}"] == pytest.approx(
            float(np.mean(independent[name]))
        )


def test_chance_floor_formulas() -> None:
    model, _ = _model_sequence()
    floors = chance_floors(model)
    expected_joint = sum(np.log(matrix.shape[0]) for matrix in model.A)
    assert floors["chance_floor_joint_loss"] == pytest.approx(expected_joint)
    n_states = model.D.size
    assert floors["chance_floor_recovery_brier"] == pytest.approx(
        (n_states - 1) / n_states
    )


def test_skill_score_anchors() -> None:
    assert skill_score(0.0, 2.0) == pytest.approx(1.0)
    assert skill_score(2.0, 2.0) == pytest.approx(0.0)
    assert skill_score(3.0, 2.0) < 0.0
    with pytest.raises(ValueError, match="positive"):
        skill_score(1.0, 0.0)


def test_recovery_calibration_hand_example() -> None:
    """Brier and ECE on a worked two-step example, computed by hand."""

    posterior = np.array([[0.8, 0.2], [0.6, 0.4]])
    true_cells = np.array([0, 1])
    metrics, bins = recovery_calibration(posterior, true_cells, n_bins=5)
    # Brier: step 1 = (0.8-1)^2 + 0.2^2 = 0.08; step 2 = 0.6^2 + (0.4-1)^2 = 0.72.
    assert metrics["brier"] == pytest.approx(0.4)
    # ECE: conf 0.8 -> bin [0.8,1.0], acc 1, |1-0.8|*0.5 = 0.1;
    #      conf 0.6 -> bin [0.6,0.8), acc 0, |0-0.6|*0.5 = 0.3.
    assert metrics["ece"] == pytest.approx(0.4)
    assert metrics["top1_accuracy"] == pytest.approx(0.5)
    assert metrics["mean_top1_confidence"] == pytest.approx(0.7)
    occupied = [b for b in bins if b["count"] > 0]
    assert len(occupied) == 2
    assert sum(b["count"] for b in occupied) == 2


def test_base_rate_brier_is_gini_impurity_and_tighter_than_uniform() -> None:
    """Base-rate Brier = 1 - sum f_k^2, and is <= the uniform floor when skewed."""

    # Uniform two-class: f = [0.5, 0.5] -> 1 - 0.5 = 0.5, equal to (S-1)/S = 0.5.
    assert base_rate_brier(np.array([0, 1]), 2) == pytest.approx(0.5)
    # Skewed three-class: f = [2/3, 0, 1/3] -> 1 - (4/9 + 1/9) = 4/9 ~ 0.4444,
    # strictly below the uniform floor (S-1)/S = 2/3.
    skewed = base_rate_brier(np.array([0, 0, 2]), 3)
    assert skewed == pytest.approx(4.0 / 9.0)
    assert skewed < (3 - 1) / 3
    # Degenerate single-cell trace carries no base-rate uncertainty.
    assert base_rate_brier(np.array([1, 1, 1]), 3) == pytest.approx(0.0)


def test_brier_skill_vs_base_rate_hand_example() -> None:
    """The reported skill-vs-base-rate matches a hand computation."""

    posterior = np.array([[0.8, 0.2], [0.6, 0.4]])
    true_cells = np.array([0, 1])
    metrics, _ = recovery_calibration(posterior, true_cells, n_bins=5)
    # model Brier 0.4 (see hand example above); base-rate Brier 0.5.
    assert metrics["base_rate_brier"] == pytest.approx(0.5)
    assert skill_score(metrics["brier"], metrics["base_rate_brier"]) == pytest.approx(
        0.2
    )


def test_base_rate_skill_emitted_and_no_freebie_for_trivial_lane() -> None:
    """The lane payload carries skill-vs-base-rate; it is <= the uniform-floor skill.

    Because the base-rate null is tighter than the uniform null, skill against
    it can never exceed skill against uniform — the by-construction location
    lane stops earning a free near-1 score it did not work for.
    """

    model, sequence = _model_sequence()
    inference = run_latent_inference(model, sequence)
    metrics, payload = lane_objectives(
        model,
        sequence,
        inference.posterior_states,
        inference.policy_posterior,
        inference.predictive_loss,
        "reference",
    )
    assert "reference_brier_skill_vs_base_rate" in metrics
    assert payload["brier_skill_vs_base_rate"] == pytest.approx(
        metrics["reference_brier_skill_vs_base_rate"], nan_ok=True
    )
    vs_base = metrics["reference_brier_skill_vs_base_rate"]
    vs_uniform = metrics["reference_brier_skill_score"]
    if not np.isnan(vs_base):
        assert vs_base <= vs_uniform + 1e-9


def test_persistence_brier_is_tighter_null_on_autocorrelated_trace() -> None:
    """On a stay-put trace persistence is a smaller (tighter) null than base rate."""

    true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    first = np.array([True, False, False, False, False, False, False, False])
    # Hand: step 0 (first) scores base-rate Brier 0.5; the single 0->1 move at
    # step 4 scores 2; every other step persists correctly at 0 -> 2.5/8.
    assert persistence_brier(true, 2, first) == pytest.approx(2.5 / 8)
    assert base_rate_brier(true, 2) == pytest.approx(0.5)
    assert persistence_brier(true, 2, first) < base_rate_brier(true, 2)


def test_persistence_can_be_weaker_null_on_anticorrelated_trace() -> None:
    """The base-rate >= persistence ordering FLIPS when moving is the norm."""

    true = np.array([0, 1, 0, 1, 0, 1])
    first = np.zeros(6, dtype=bool)
    first[0] = True
    # Persistence predicts the wrong cell every non-first step (Brier 2 each).
    assert persistence_brier(true, 2, first) == pytest.approx((0.5 + 2 * 5) / 6)
    assert persistence_brier(true, 2, first) > base_rate_brier(true, 2)


def test_lane_emits_persistence_skill_consistent_with_its_brier() -> None:
    """The persistence skill is emitted and equals 1 - model_brier/persistence_brier.

    The base-rate vs persistence ORDERING is deliberately not asserted here — it
    is data-dependent (pinned on hand traces above), so binding it to whichever
    direction this fixture happens to show would be a vacuous, brittle test.
    """

    model, sequence = _model_sequence()
    inference = run_latent_inference(model, sequence)
    metrics, payload = lane_objectives(
        model,
        sequence,
        inference.posterior_states,
        inference.policy_posterior,
        inference.predictive_loss,
        "reference",
    )
    assert "reference_brier_skill_vs_persistence" in metrics
    assert payload["brier_skill_vs_persistence"] == pytest.approx(
        metrics["reference_brier_skill_vs_persistence"], nan_ok=True
    )
    persist_brier = persistence_brier(
        np.asarray(sequence.spatial_states, dtype=int),
        model.D.size,
        sequence.is_first_observation,
    )
    expected = skill_score(metrics["reference_calibration_brier"], persist_brier)
    assert metrics["reference_brier_skill_vs_persistence"] == pytest.approx(expected)


def test_full_model_better_calibrated_than_deep_latent() -> None:
    """H2: location-observing Brier strictly below deep-latent Brier."""

    full_model, sequence = _model_sequence()
    latent_model, _ = _model_sequence(DEEP_LATENT)
    true_cells = np.asarray(sequence.spatial_states, dtype=int)
    full = run_latent_inference(full_model, sequence)
    latent = run_latent_inference(latent_model, sequence)
    full_cal, _ = recovery_calibration(full.posterior_states, true_cells)
    latent_cal, _ = recovery_calibration(latent.posterior_states, true_cells)
    assert full_cal["brier"] < latent_cal["brier"]


def test_joint_skill_positive_and_ordered() -> None:
    """H3: the fitted models beat the uniform-chance floor; location helps."""

    full_model, sequence = _model_sequence()
    latent_model, _ = _model_sequence(DEEP_LATENT)
    full = run_latent_inference(full_model, sequence)
    latent = run_latent_inference(latent_model, sequence)
    full_skill = skill_score(
        float(np.mean(full.predictive_loss)),
        chance_floors(full_model)["chance_floor_joint_loss"],
    )
    latent_skill = skill_score(
        float(np.mean(latent.predictive_loss)),
        chance_floors(latent_model)["chance_floor_joint_loss"],
    )
    assert full_skill > 0.0
    assert full_skill > latent_skill


def test_objectives_deterministic() -> None:
    model, sequence = _model_sequence()
    inference = run_latent_inference(model, sequence)
    first, _ = lane_objectives(
        model,
        sequence,
        inference.posterior_states,
        inference.policy_posterior,
        inference.predictive_loss,
        "reference",
    )
    second, _ = lane_objectives(
        model,
        sequence,
        inference.posterior_states,
        inference.policy_posterior,
        inference.predictive_loss,
        "reference",
    )
    assert first == second


def test_pipeline_emits_objectives_and_calibration_artifact(tmp_path) -> None:
    """Integration: reference-engine pipeline emits objective metrics + calibration.json."""

    config_path = tmp_path / "objectives.yaml"
    config_path.write_text(
        f"""
run_id: objectives_probe
output_dir: {tmp_path / "run"}
fractalrabbit:
  execution: fixture
  fixture_csv: {FIXTURE.resolve()}
observation:
  spatial_bins: 3
  time_bins_per_day: 4
model:
  smoothing: 0.05
closed_loop:
  iterations: 1
  candidates: []
""",
        encoding="utf-8",
    )
    result = run_pipeline_from_path(config_path)
    metrics = result.metrics
    assert "reference_skill_score" in metrics
    assert "reference_calibration_brier" in metrics
    assert "reference_calibration_ece" in metrics
    assert "chance_floor_joint_loss" in metrics
    for name in result.metadata["observation_modalities"]:
        assert f"reference_per_modality_loss_{name}" in metrics
        # H1 at the artifact level: each emitted marginal mean <= joint mean + interaction slack
        assert (
            metrics[f"reference_per_modality_loss_{name}"]
            <= metrics["mean_predictive_loss"] + 1e-9
        )
    # Reference engine only: no pymdp objective keys.
    assert not any(key.startswith("pymdp_calibration") for key in metrics)
    payload = json.loads(result.artifacts["calibration"].read_text(encoding="utf-8"))
    assert payload["schema"] == "active-fractal-rabbit-calibration-v1"
    assert [lane["prefix"] for lane in payload["lanes"]] == ["reference"]
    assert (
        "not empirical" in payload["caveat"]
        or "Nothing here is empirical" in payload["caveat"]
    )
    bins = payload["lanes"][0]["calibration_bins"]
    assert sum(b["count"] for b in bins) == metrics["n_observations"]
