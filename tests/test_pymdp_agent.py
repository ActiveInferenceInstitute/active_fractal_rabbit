from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from active_fractal_rabbit.config import FRACTALRABBIT_PINNED_COMMIT, load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.inference import run_latent_inference
from active_fractal_rabbit.observations import discretize_observations
from active_fractal_rabbit.pipeline import run_pipeline_from_path
from active_fractal_rabbit.pomdp import build_pomdp, observation_indices
from active_fractal_rabbit.pymdp_agent import build_pymdp_agent, run_pymdp_inference

FIXTURE = Path("data/fixtures/fractalrabbit_sample.csv").resolve()


def _sequence():
    config = load_config("configs/default.yaml")
    return discretize_observations(parse_waypoint_csv(FIXTURE), config.inference)


def _engine_config(tmp_path: Path, engine: str) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    path = config_dir / f"engine_{engine}.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            run_id: engine_{engine}
            fractalrabbit:
              upstream_commit: {FRACTALRABBIT_PINNED_COMMIT}
              execution: fixture
              fixture_csv: {FIXTURE}
            observation:
              spatial_bins: 3
              time_bins_per_day: 4
            model:
              smoothing: 0.05
              engine: {engine}
            closed_loop:
              iterations: 1
              candidates: []
            """
        ),
        encoding="utf-8",
    )
    return path


def test_pymdp_agent_builds_with_batched_arrays() -> None:
    sequence = _sequence()
    model = build_pomdp(sequence, 0.05)
    agent = build_pymdp_agent(model)

    assert agent.num_modalities == len(model.A)
    assert agent.num_states == [model.D.size]
    assert agent.num_controls == [model.B.shape[2]]


def test_pymdp_matched_prior_inference_agrees_with_exact_bayes() -> None:
    """Cross-implementation validation: pymdp FPI vs exact posterior, same priors."""

    sequence = _sequence()
    model = build_pomdp(sequence, 0.05)
    reference = run_latent_inference(model, sequence)
    result = run_pymdp_inference(
        model, sequence, reference_posterior=reference.posterior_states
    )

    assert result.metrics["pymdp_exact_inference_max_abs_delta"] < 1e-5
    # the closed loops legitimately diverge (different action selection); both recorded
    assert "pymdp_reference_max_abs_posterior_delta" in result.metrics
    assert result.posterior_states.shape == reference.posterior_states.shape
    assert np.allclose(result.posterior_states.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(result.policy_posterior.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(np.isfinite(result.expected_free_energy))
    assert np.all(np.isfinite(result.predictive_loss))
    assert np.all(np.isfinite(result.variational_free_energy))
    assert np.all(result.step_runtime_ms > 0)
    assert (
        result.metrics["pymdp_max_step_runtime_ms"]
        >= result.metrics["pymdp_median_step_runtime_ms"]
    )


def test_pymdp_validation_indices_bound_trace_without_changing_model() -> None:
    sequence = _sequence()
    model = build_pomdp(sequence, 0.05)
    reference = run_latent_inference(model, sequence)

    result = run_pymdp_inference(
        model,
        sequence,
        reference_posterior=reference.posterior_states,
        validation_indices=(0, 3, 5),
    )

    assert result.posterior_states.shape[0] == 3
    assert result.metrics["pymdp_source_observation_rows"] == float(
        len(sequence.observations)
    )
    assert result.metrics["pymdp_inference_rows"] == 3.0
    assert 0.0 < result.metrics["pymdp_validation_fraction"] < 1.0
    assert result.metrics["pymdp_exact_inference_max_abs_delta"] < 1e-5
    with pytest.raises(ValueError, match="out-of-range"):
        run_pymdp_inference(
            model, sequence, validation_indices=(len(sequence.observations),)
        )


def test_pymdp_reference_posterior_shape_mismatch_raises() -> None:
    sequence = _sequence()
    model = build_pomdp(sequence, 0.05)
    with pytest.raises(ValueError, match="shape"):
        run_pymdp_inference(model, sequence, reference_posterior=np.zeros((2, 2)))


def test_observation_indices_align_by_name_for_modality_subsets() -> None:
    sequence = _sequence()
    full = build_pomdp(sequence, 0.05)
    subset = build_pomdp(
        sequence, 0.05, modalities=("time_of_day", "speed", "report_burst")
    )

    full_indices = observation_indices(full, sequence)
    subset_indices = observation_indices(subset, sequence)

    assert len(full_indices[0]) == 4
    assert len(subset_indices[0]) == 3
    # subset tuples must equal the full tuples with the location entry removed
    assert [row[1:] for row in full_indices] == subset_indices


def test_pipeline_engine_knob_flip(tmp_path) -> None:
    """Flip test: model.engine controls whether the real pymdp lane runs."""

    reference_only = run_pipeline_from_path(_engine_config(tmp_path, "reference"))
    both = run_pipeline_from_path(_engine_config(tmp_path, "both"))

    assert "pymdp_mean_predictive_loss" not in reference_only.metrics
    assert not (reference_only.output_dir / "pymdp_trace.csv").exists()

    assert "pymdp_mean_predictive_loss" in both.metrics
    assert both.metrics["pymdp_exact_inference_max_abs_delta"] < 1e-5
    trace_path = both.output_dir / "pymdp_trace.csv"
    assert trace_path.exists()
    header = trace_path.read_text(encoding="utf-8").splitlines()[0]
    assert "pymdp_q_pi_empirical_continuation" in header
    assert "pymdp_neg_efe_empirical_continuation" in header
    assert "pymdp_vfe" in header
    assert "pymdp_step_runtime_ms" in header
    assert both.metrics["pymdp_total_runtime_s"] > 0
    assert both.metrics["pymdp_median_step_runtime_ms"] > 0
    assert np.isfinite(both.metrics["pymdp_mean_vfe"])
    metrics_on_disk = json.loads(
        (both.output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert "pymdp_mean_posterior_entropy" in metrics_on_disk
    # the pymdp lane never replaces reference metrics — both present, prefixed
    assert (
        metrics_on_disk["mean_predictive_loss"]
        == reference_only.metrics["mean_predictive_loss"]
    )
