from __future__ import annotations

import json

import numpy as np

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.factorial_pymdp import (
    FACTORIAL_PYMDP_SCHEMA,
    FULL_POLICY_MODE,
    STATE_TRANSITION_POLICY_MODE,
    build_factorial_pomdp,
    build_factorial_pymdp_agent,
    derive_latent_modes,
    run_and_write_factorial_analysis,
    run_factorial_pymdp_inference,
    tensor_summary,
    unavailable_factorial_analysis,
)
from active_fractal_rabbit.fractalrabbit import WaypointObservation, parse_waypoint_csv
from active_fractal_rabbit.observations import discretize_observations


def _fixture_sequence():
    config = load_config("configs/default.yaml")
    observations = parse_waypoint_csv("data/fixtures/fractalrabbit_sample.csv")
    return discretize_observations(observations, config.inference), config


def test_derive_latent_modes_separates_movement_and_synthetic_cognitive_proxy() -> None:
    config = load_config("configs/default.yaml")
    observations = (
        WaypointObservation(0, 0.0, 0.0, 0.0),
        WaypointObservation(0, 1.0, 0.0, 0.0),
        WaypointObservation(0, 1.1, 2.0, 0.0),
        WaypointObservation(0, 4.0, 0.0, 0.0),
    )
    sequence = discretize_observations(observations, config.inference)

    modes = derive_latent_modes(sequence)

    assert modes.movement_names == ("stationary", "return_or_dwell", "exploratory_move")
    assert modes.cognitive_proxy_names == (
        "routine_report",
        "report_burst",
        "report_gap",
    )
    assert modes.movement_modes.tolist() == [0, 0, 1, 1]
    assert modes.cell_delta.tolist() == [0, 0, 1, 1]
    assert modes.return_flags.tolist() == [0, 1, 0, 1]
    assert modes.cognitive_proxy_modes[0] == 0
    assert 1 in modes.cognitive_proxy_modes
    assert 2 in modes.cognitive_proxy_modes


def test_derive_latent_modes_low_speed_new_cell_is_not_exploratory() -> None:
    """A slow move to a never-visited cell must not be misclassified via the
    (previous != cell) tautology; it should fall through to return_or_dwell
    unless speed_categories actually indicates a fast move. This reuses the
    fixture above, where idx 2 is exactly such a case."""
    config = load_config("configs/default.yaml")
    observations = (
        WaypointObservation(0, 0.0, 0.0, 0.0),
        WaypointObservation(0, 1.0, 0.0, 0.0),
        WaypointObservation(0, 1.1, 2.0, 0.0),
        WaypointObservation(0, 4.0, 0.0, 0.0),
    )
    sequence = discretize_observations(observations, config.inference)

    modes = derive_latent_modes(sequence)

    assert bool(sequence.is_first_observation[2]) is False
    assert modes.return_flags[2] == 0
    assert int(sequence.speed_categories[2]) < 2
    assert modes.movement_modes[2] == 1


def test_derive_latent_modes_high_speed_new_cell_is_exploratory() -> None:
    """The exploratory branch must still be reachable for a genuinely fast
    move to a never-visited cell."""
    config = load_config("configs/default.yaml")
    observations = (
        WaypointObservation(0, 0.0, 0.0, 0.0),
        WaypointObservation(0, 1.0, 0.0, 0.0),
        WaypointObservation(0, 1.01, 40.0, 0.0),
        WaypointObservation(0, 4.0, 0.0, 0.0),
    )
    sequence = discretize_observations(observations, config.inference)

    modes = derive_latent_modes(sequence)

    assert bool(sequence.is_first_observation[2]) is False
    assert modes.return_flags[2] == 0
    assert int(sequence.speed_categories[2]) >= 2
    assert modes.movement_modes[2] == 2


def test_factorial_pomdp_shapes_dependencies_and_normalization() -> None:
    sequence, config = _fixture_sequence()
    model = build_factorial_pomdp(sequence, config.inference.model)
    summary = tensor_summary(model)

    assert model.factor_names == ("cell", "movement_mode", "cognitive_proxy_mode")
    assert model.factor_sizes == (sequence.num_spatial_states, 3, 3)
    assert model.observation_names == (
        "location",
        "time_of_day",
        "speed",
        "report_burst",
        "cell_delta",
        "return_flag",
    )
    assert model.A_dependencies == ((0,), (0, 2), (1, 0), (2,), (1,), (1,))
    assert model.B_dependencies == ((0, 1), (1, 2), (2,))
    assert summary["joint_state_count"] == sequence.num_spatial_states * 9
    for matrix in model.A:
        assert np.allclose(matrix.sum(axis=0), 1.0)
    for tensor in model.B:
        assert np.allclose(tensor.sum(axis=0), 1.0)
    for prior in model.D:
        assert np.isclose(prior.sum(), 1.0)


def test_factorial_pymdp_agent_builds_real_multifactor_agent() -> None:
    sequence, config = _fixture_sequence()
    model = build_factorial_pomdp(sequence, config.inference.model)
    agent = build_factorial_pymdp_agent(model)

    assert agent.num_factors == 3
    assert agent.num_states == list(model.factor_sizes)
    assert agent.A_dependencies == [list(item) for item in model.A_dependencies]
    assert agent.B_dependencies == [list(item) for item in model.B_dependencies]
    assert agent.policies.policy_arr.shape[-1] == 3


def test_factorial_pymdp_matched_prior_and_mode_metrics() -> None:
    sequence, config = _fixture_sequence()
    model = build_factorial_pomdp(sequence, config.inference.model)
    result = run_factorial_pymdp_inference(
        model, sequence, policy_mode=FULL_POLICY_MODE
    )

    assert result.metrics["factorial_pymdp_matched_prior_max_abs_delta"] < 1e-4
    assert result.metrics["factorial_pymdp_n_factors"] == 3.0
    assert result.metrics["factorial_pymdp_n_joint_states"] == float(
        np.prod(model.factor_sizes)
    )
    assert result.metrics["factorial_pymdp_state_validation_steps"] == 0.0
    assert result.metrics["factorial_pymdp_policy_evaluation_steps"] == float(
        len(sequence.observations)
    )
    assert result.metrics["factorial_pymdp_state_transition_prior_steps"] == 0.0
    assert result.factor_posteriors["cell"].shape[0] == len(sequence.observations)
    assert result.factor_posteriors["movement_mode"].shape[1] == 3
    assert result.factor_posteriors["cognitive_proxy_mode"].shape[1] == 3
    assert np.allclose(result.factor_posteriors["cell"].sum(axis=1), 1.0, atol=1e-5)
    assert np.all(np.isfinite(result.predictive_loss))
    assert np.all(np.isfinite(result.expected_free_energy))


def test_factorial_pymdp_large_sample_state_transition_mode_is_explicit() -> None:
    sequence, config = _fixture_sequence()
    model = build_factorial_pomdp(sequence, config.inference.model)
    result = run_factorial_pymdp_inference(
        model, sequence, policy_mode=STATE_TRANSITION_POLICY_MODE
    )

    assert result.metrics["factorial_pymdp_matched_prior_max_abs_delta"] < 1e-4
    assert result.metrics["factorial_pymdp_state_validation_steps"] == float(
        len(sequence.observations)
    )
    assert result.metrics["factorial_pymdp_policy_evaluation_steps"] == 0.0
    assert result.metrics["factorial_pymdp_state_transition_prior_steps"] == float(
        len(sequence.observations)
    )
    assert np.allclose(result.policy_posterior.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(result.policy_posterior, result.policy_posterior[0])
    assert np.allclose(result.expected_free_energy, 0.0)
    assert np.all(result.selected_actions == 0)
    assert np.all(np.isfinite(result.predictive_loss))


def test_run_and_write_factorial_analysis_artifact_and_unavailable_payload(
    tmp_path,
) -> None:
    out = tmp_path / "factorial_pymdp_analysis_test.json"
    path = run_and_write_factorial_analysis("configs/default.yaml", out)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema"] == FACTORIAL_PYMDP_SCHEMA
    assert payload["status"] == "completed"
    assert payload["lane"] == "fixture"
    assert payload["hidden_factors"] == [
        "cell",
        "movement_mode",
        "cognitive_proxy_mode",
    ]
    assert (
        payload["mode_boundary"]
        == "synthetic cognitive-proxy states, not real cognition or intent"
    )
    assert payload["policy_mode"] == STATE_TRANSITION_POLICY_MODE
    assert payload["policy_evaluation_steps"] == 0
    assert payload["performance_gate"] in {
        "factorial_pymdp_cleared_current_lane",
        "factorial_pymdp_not_performance_leading",
        "pymdp_baseline_unavailable",
    }
    assert payload["metrics"]["factorial_pymdp_mean_predictive_loss"] > 0
    assert payload["metrics"]["factorial_pymdp_state_validation_steps"] == 96.0
    assert payload["metrics"]["factorial_pymdp_state_transition_prior_steps"] == 2400.0
    assert payload["metrics"]["switching_hmm_mean_predictive_loss"] > 0
    assert payload["metrics"]["particle_state_space_mean_predictive_loss"] > 0
    assert any(row["model"] == "factorial_pymdp" for row in payload["model_rows"])

    unavailable = unavailable_factorial_analysis("external lane not completed")
    assert unavailable["status"] == "unavailable"
    assert unavailable["factorial_pymdp_mean_predictive_loss"] == "unavailable"


def test_mean_field_solver_matches_exact_marginals_without_factor_coupling() -> None:
    """Non-circularity anchor for the matched-prior validation.

    The matched-prior check validates pymdp against an independent analytic
    mean-field solver. That is only meaningful if the mean-field solver itself
    is correct — otherwise two copies of the same error could agree. When the
    likelihood factorizes (every observation depends on exactly one latent
    factor) mean-field inference is provably EXACT, so the independent solver
    must match the exact joint marginals to machine precision. This pins the
    reference, breaking the circularity an agreement number alone cannot.
    """

    import numpy as np

    from active_fractal_rabbit.factorial_pymdp import (
        _exact_marginals_given_factorized_prior,
        _mean_field_marginals_given_factorized_prior,
        _normalize_columns,
    )
    from active_fractal_rabbit.types import FactorialPomdpModel, LatentModeSequence

    rng = np.random.default_rng(0)
    factor_sizes = (4, 3, 2)
    a_dependencies = (
        (0,),
        (1,),
        (2,),
        (0,),
        (1,),
    )  # NO observation couples two factors
    obs_sizes = (5, 3, 2, 4, 6)
    A = tuple(
        _normalize_columns(rng.random((obs_size, factor_sizes[deps[0]])) + 0.05)
        for obs_size, deps in zip(obs_sizes, a_dependencies)
    )
    dummy_modes = LatentModeSequence(
        movement_modes=np.zeros(1, dtype=int),
        cognitive_proxy_modes=np.zeros(1, dtype=int),
        cell_delta=np.zeros(1, dtype=int),
        return_flags=np.zeros(1, dtype=int),
        movement_names=("a", "b", "c"),
        cognitive_proxy_names=("x", "y", "z"),
    )
    model = FactorialPomdpModel(
        A=A,
        B=tuple(np.zeros((size, size, 1)) for size in factor_sizes),
        C=tuple(np.zeros(size) for size in obs_sizes),
        D=tuple(
            _normalize_columns(rng.random((size,)) + 0.05) for size in factor_sizes
        ),
        observation_names=("o0", "o1", "o2", "o3", "o4"),
        factor_names=("cell", "movement_mode", "cognitive_proxy_mode"),
        factor_sizes=factor_sizes,
        A_dependencies=a_dependencies,
        B_dependencies=((0,), (1,), (2,)),
        control_names=(("c",), ("c",), ("c",)),
        latent_modes=dummy_modes,
    )
    priors = [_normalize_columns(rng.random((size,)) + 0.05) for size in factor_sizes]

    worst = 0.0
    for observation in [(0, 0, 0, 0, 0), (4, 2, 1, 3, 5), (1, 1, 0, 2, 4)]:
        mean_field = _mean_field_marginals_given_factorized_prior(
            model, priors, observation
        )
        exact = _exact_marginals_given_factorized_prior(model, priors, observation)
        worst = max(
            worst,
            max(float(np.max(np.abs(mf - ex))) for mf, ex in zip(mean_field, exact)),
        )
    assert worst < 1e-9  # mean-field == exact under no coupling: the solver is correct


def test_factorial_mean_field_incurs_real_coupling_gap_vs_exact() -> None:
    """The mean-field-vs-exact diagnostic is a genuine, non-zero coupling cost.

    On the real factor-coupled fixture model the independent mean-field solver
    must differ measurably from the exact marginals (a tautological self-compare
    would read zero). This guards against the diagnostic silently collapsing to
    a decorative zero and proves the matched-prior reference is a distinct object
    from the exact marginals.
    """

    import numpy as np

    from active_fractal_rabbit.factorial_pymdp import (
        _exact_marginals_given_factorized_prior,
        _mean_field_marginals_given_factorized_prior,
        observation_indices,
    )

    sequence, config = _fixture_sequence()
    model = build_factorial_pomdp(sequence, config.inference.model)
    priors = [np.asarray(prior, dtype=float) for prior in model.D]
    evidence = observation_indices(model, sequence)
    gap = 0.0
    for observation in evidence[:24]:
        mean_field = _mean_field_marginals_given_factorized_prior(
            model, priors, observation
        )
        exact = _exact_marginals_given_factorized_prior(model, priors, observation)
        gap = max(
            gap,
            max(float(np.max(np.abs(mf - ex))) for mf, ex in zip(mean_field, exact)),
        )
    assert gap > 1e-3  # coupling is real; the honest diagnostic is not decorative
    assert gap < 1.0


def test_factorial_helpers_and_error_paths() -> None:
    """Direct coverage of pure helpers and fail-loud error branches."""

    import numpy as np
    import pytest

    from active_fractal_rabbit import factorial_pymdp as F

    sequence, config = _fixture_sequence()

    # include-mask validation is fail-loud
    with pytest.raises(ValueError, match="include mask length"):
        F.build_factorial_pomdp(
            sequence, config.inference.model, include=np.ones(3, dtype=bool)
        )
    with pytest.raises(ValueError, match="at least one observation"):
        F.build_factorial_pomdp(
            sequence,
            config.inference.model,
            include=np.zeros(len(sequence.observations), dtype=bool),
        )

    # normalization refuses zero-probability columns
    with pytest.raises(ValueError, match="zero-probability"):
        F._normalize_columns(np.zeros((3, 2)))

    # safe-normalize falls back to uniform on a zero-sum vector
    assert np.allclose(F._safe_normalize(np.zeros(4)), np.full(4, 0.25))

    # preference vectors: uniform (default), diagnostic_uncertainty, diagnostic_gap
    assert np.allclose(F._preference_vector(3, "report_burst", "uniform"), 0.0)
    assert np.allclose(F._preference_vector(3, "speed", "diagnostic_uncertainty"), 0.0)
    gap_vector = F._preference_vector(3, "report_burst", "diagnostic_gap")
    assert gap_vector[1] > 0.0

    # gate, best-model, and gain helpers
    metrics = {
        "factorial_pymdp_mean_predictive_loss": 1.0,
        "switching_hmm_mean_predictive_loss": 2.0,
        "particle_state_space_mean_predictive_loss": 3.0,
        "persistence_mean_predictive_loss": 4.0,
    }
    assert F._best_model_name(metrics) == "factorial_pymdp"
    assert F._performance_gate(metrics, {}) == "pymdp_baseline_unavailable"
    assert (
        F._performance_gate(metrics, {"pymdp_mean_predictive_loss": 2.0})
        == "factorial_pymdp_cleared_current_lane"
    )
    assert (
        F._performance_gate(metrics, {"pymdp_mean_predictive_loss": 1.0})
        == "factorial_pymdp_not_performance_leading"
    )
    gains = F._gain_metrics(
        metrics, {"pymdp_mean_predictive_loss": 2.5, "mean_predictive_loss": 1.8}
    )
    assert gains["factorial_pymdp_gain_vs_single_pymdp"] == pytest.approx(1.5)
    assert gains["factorial_pymdp_gain_vs_reference"] == pytest.approx(0.8)
    assert F._gain_metrics(metrics, {}) == {}

    # number/stability helpers
    assert F._maybe_number(2) == 2.0
    assert F._maybe_number("unavailable") is None
    assert F._stability(np.array([0])) == 1.0
    assert F._stability(np.array([1, 1, 2])) == pytest.approx(0.5)
    assert F._sentinel_indices(5, 10) == (0, 1, 2, 3, 4)
    assert F._sentinel_indices(10, 4) == (0, 3, 6, 9)
    assert F._sentinel_indices(0, 4) == ()

    # unavailable payload carries no numbers
    unavailable = F.unavailable_factorial_analysis("external lane not built")
    assert unavailable["status"] == "unavailable"
    assert unavailable["factorial_pymdp_mean_predictive_loss"] == "unavailable"
    assert unavailable["model_rows"] == []


def test_resolved_observations_external_lane_is_fail_loud(tmp_path) -> None:
    """An external-lane run with no produced observations must fail loudly."""

    import textwrap

    import pytest

    from active_fractal_rabbit import factorial_pymdp as F
    from active_fractal_rabbit.config import FRACTALRABBIT_PINNED_COMMIT, load_config

    config_path = tmp_path / "external.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            run_id: external_probe
            output_dir: {tmp_path / "out"}
            fractalrabbit:
              upstream_commit: {FRACTALRABBIT_PINNED_COMMIT}
              execution: external
            observation:
              spatial_bins: 3
              time_bins_per_day: 4
            model:
              smoothing: 0.05
            closed_loop:
              iterations: 1
              candidates: []
            """
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    with pytest.raises(FileNotFoundError, match="external observations"):
        F._resolved_existing_observations(config, config_path)


def test_factorial_pymdp_rejects_unknown_policy_mode() -> None:
    import pytest

    sequence, config = _fixture_sequence()
    model = build_factorial_pomdp(sequence, config.inference.model)
    with pytest.raises(ValueError, match="unknown factorial policy_mode"):
        run_factorial_pymdp_inference(model, sequence, policy_mode="maybe")
