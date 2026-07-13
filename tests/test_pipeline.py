from __future__ import annotations

import json
import csv
import math
import textwrap
from pathlib import Path

import pytest

from active_fractal_rabbit.closed_loop import run_closed_loop
from active_fractal_rabbit.config import FRACTALRABBIT_PINNED_COMMIT, load_config
from active_fractal_rabbit.pipeline import run_pipeline_from_path, score_candidates
from active_fractal_rabbit.sweeps import run_replicated_sensitivity


def _config(tmp_path: Path, fixture: Path, objective: str = "predictive_loss") -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "test.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            run_id: test
            output_dir: {tmp_path / "output" / "runs" / "test"}
            fractalrabbit:
              upstream_repo: https://github.com/NationalSecurityAgency/fractalrabbit.git
              upstream_commit: {FRACTALRABBIT_PINNED_COMMIT}
              cache_dir: {tmp_path / ".cache" / "fractalrabbit"}
              execution: fixture
              fixture_csv: {fixture}
              output_prefix: test
              params:
                space_dimension: 2
                fractal_dimension: 1.33
                number_of_points: 5000
                number_of_travellers: 4
                number_of_co_travellers: 1
                number_of_days: 3.0
                number_of_reports: 6.0
            observation:
              spatial_bins: 3
              time_bins_per_day: 4
            model:
              smoothing: 0.05
              speed_thresholds_km_per_day: [10.0, 50.0, 150.0]
            closed_loop:
              iterations: 2
              objective: {objective}
              candidates:
                - name: low
                  params:
                    number_of_reports: 4.0
                - name: high
                  params:
                    number_of_reports: 8.0
            """
        ),
        encoding="utf-8",
    )
    return path


def test_fixture_pipeline_writes_metrics(tmp_path) -> None:
    config = _config(tmp_path, Path("data/fixtures/fractalrabbit_sample.csv").resolve())
    result = run_pipeline_from_path(config)

    assert result.metrics["n_observations"] == 16.0
    assert result.metrics["pymdp_payload_shape_ok"] == 1.0
    assert result.selected_next_candidate == "high"
    assert result.candidate_scores[0].name == "high"
    for artifact in [
        "metrics.json",
        "baseline_metrics.json",
        "baseline_trace.csv",
        "discrete_features.csv",
        "inference_trace.csv",
        "policy_trace.csv",
        "tensor_summary.json",
        "candidate_scores.json",
        "candidate_scores.csv",
        "config_snapshot.json",
        "provenance_manifest.json",
    ]:
        assert (result.output_dir / artifact).exists()
    metrics = json.loads(
        (result.output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["mean_predictive_loss"] > 0
    assert math.isfinite(metrics["baseline_delta_mean_predictive_loss"])
    baseline = json.loads(
        (result.output_dir / "baseline_metrics.json").read_text(encoding="utf-8")
    )
    assert baseline["baseline_control"] == "empirical_continuation"
    assert baseline["baseline_mean_predictive_loss"] > 0
    with (result.output_dir / "baseline_trace.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        baseline_rows = list(csv.DictReader(handle))
    assert len(baseline_rows) == int(metrics["n_observations"])
    assert all(
        math.isfinite(float(row["baseline_predictive_loss"])) for row in baseline_rows
    )
    with (result.output_dir / "discrete_features.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        feature_rows = list(csv.DictReader(handle))
    assert "is_first_observation" in feature_rows[0]
    assert sum(row["is_first_observation"] == "1" for row in feature_rows) == 4
    with (result.output_dir / "policy_trace.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        policy_rows = list(csv.DictReader(handle))
    for row in policy_rows:
        for control in ["empirical_continuation", "high_entropy_exploration"]:
            proxy = float(row[f"proxy_score_{control}"])
            predictive_entropy = float(row[f"predictive_entropy_{control}"])
            preference = float(row[f"preference_divergence_{control}"])
            assert math.isclose(
                proxy, predictive_entropy + preference, rel_tol=1e-9, abs_tol=1e-9
            )


def test_pipeline_validates_fixture_before_copying_to_output(tmp_path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP_SECRET_LOCAL_CONTENT\n", encoding="utf-8")
    config = _config(tmp_path, secret)
    observations_path = tmp_path / "output" / "runs" / "test" / "observations.csv"

    with pytest.raises(ValueError, match="must contain header"):
        run_pipeline_from_path(config)

    assert not observations_path.exists()


def test_closed_loop_selection_uses_measured_metrics(tmp_path) -> None:
    config = _config(
        tmp_path,
        Path("data/fixtures/fractalrabbit_sample.csv").resolve(),
        "posterior_entropy",
    )
    results = run_closed_loop(config)

    assert len(results) == 2
    assert results[0].selected_next_candidate in {"low", "high"}
    assert results[1].run_id == "test_iter_2"
    summary = tmp_path / "output" / "runs" / "test_closed_loop.json"
    assert summary.exists()
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload[0]["candidate_scores"][0]["rank"] == 1


def test_candidate_scores_are_deterministic_and_metric_based(tmp_path) -> None:
    config_path = _config(
        tmp_path, Path("data/fixtures/fractalrabbit_sample.csv").resolve()
    )
    result = run_pipeline_from_path(config_path)
    config_metrics = {"mean_predictive_loss": result.metrics["mean_predictive_loss"]}
    scores = score_candidates(load_config(config_path), config_metrics)

    assert [score.name for score in scores] == ["high", "low"]
    assert scores[0].score < scores[1].score


def test_sensitivity_report_count_variants_follow_scaled_config_candidates() -> None:
    from active_fractal_rabbit.sweeps import _variant_configs

    base = load_config("configs/default.yaml")
    variants = {
        factors["variant"]: config for factors, config in _variant_configs(base)
    }

    assert variants["lower_reports"].fractalrabbit.params.number_of_reports == 320.0
    assert variants["higher_reports"].fractalrabbit.params.number_of_reports == 960.0


def test_sensitivity_replicates_follow_experiment_repetitions(tmp_path) -> None:
    """Flip test: experiment.repetitions must control the replicate family size."""

    base = _config(tmp_path, Path("data/fixtures/fractalrabbit_sample.csv").resolve())
    one = json.loads(
        run_replicated_sensitivity(base)["json"].read_text(encoding="utf-8")
    )

    boosted = base.read_text(encoding="utf-8") + "\nexperiment:\n  repetitions: 3\n"
    boosted_path = base.parent / "boosted.yaml"
    boosted_path.write_text(boosted, encoding="utf-8")
    three = json.loads(
        run_replicated_sensitivity(boosted_path)["json"].read_text(encoding="utf-8")
    )

    assert three["n_runs"] == one["n_runs"] + 2
    replicate_runs = [
        run for run in three["runs"] if run["factors"]["family"] == "replicate"
    ]
    assert len(replicate_runs) == 3
    # Fixture-lane replicates are bit-identical reruns; the payload must say so
    # and the replicate metrics must actually be identical (zero spread).
    assert three["deterministic_replicates"] is True
    losses = {run["metrics"]["mean_predictive_loss"] for run in replicate_runs}
    assert len(losses) == 1
    assert "bit-identical" in three["caveat"]
    assert "canonical lane carries the full pymdp trace" in three["caveat"]
    assert {run["engine"] for run in three["runs"]} == {"reference"}


def test_replicated_sensitivity_writes_summary_artifacts(tmp_path) -> None:
    config = _config(tmp_path, Path("data/fixtures/fractalrabbit_sample.csv").resolve())
    artifacts = run_replicated_sensitivity(config)

    payload = json.loads(artifacts["json"].read_text(encoding="utf-8"))
    assert payload["schema"] == "active-fractal-rabbit-sensitivity-v1"
    assert payload["lane"] == "fixture"
    assert "checked fixture" in payload["caveat"]
    assert payload["n_runs"] >= 6
    assert payload["source_config"].endswith("test.yaml")
    families = {run["factors"]["family"] for run in payload["runs"]}
    assert "latent_uncertainty" in families
    latent_run = next(
        run
        for run in payload["runs"]
        if run["factors"]["family"] == "latent_uncertainty"
    )
    assert latent_run["metrics"]["mean_posterior_entropy"] > 0.5
    assert "mean_predictive_loss" in payload["metric_summary"]
    assert all(
        "control_count" in run and "max_policy_entropy" in run
        for run in payload["runs"]
    )
    assert all(run["engine"] == "reference" for run in payload["runs"])
    stats = payload["metric_summary"]["mean_predictive_loss"]
    assert {
        "mean",
        "std",
        "min",
        "max",
        "normal_approx_low",
        "normal_approx_high",
    } <= set(stats)
    assert (
        artifacts["csv"]
        .read_text(encoding="utf-8")
        .startswith("metric,n,mean,std,min,max,normal_approx_low,normal_approx_high")
    )
