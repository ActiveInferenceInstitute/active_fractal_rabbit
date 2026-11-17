from __future__ import annotations

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from active_fractal_rabbit import forward_gates, pipeline, robustness
from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractal_dimension import run_fractal_dimension_sweep
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.negative_controls import run_negative_controls
from active_fractal_rabbit.types import PipelineResult

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "data/fixtures/fractalrabbit_sample.csv"
LARGE = ROOT / "data/fixtures/fractalrabbit_sample_large.csv"
full_study = importlib.import_module("scripts.07_run_full_study")


def _config(tmp_path, *, engine="reference", fixture=SMALL):
    directory = tmp_path / "configs"
    directory.mkdir()
    path = directory / "custom.yaml"
    raw = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    raw["output_dir"] = str(tmp_path / "owned_collection" / "base")
    raw["fractalrabbit"]["fixture_csv"] = str(fixture)
    raw["fractalrabbit"]["cache_dir"] = str(tmp_path / "cache")
    raw["model"]["engine"] = engine
    path.write_text(yaml.safe_dump(raw))
    return path, load_config(path)


def _record_real_pipeline(monkeypatch):
    original = pipeline.run_pipeline
    records = []

    def recording(config, path):
        result = original(config, path)
        records.append((config, result))
        return result

    monkeypatch.setattr(pipeline, "run_pipeline", recording)
    return records


def _assert_owned(records, base):
    for config, result in records:
        assert result.output_dir == base.output_dir.parent / config.run_id
        assert config.fractalrabbit.output_prefix == config.run_id
        assert result.observations_path.is_file()
        assert result.artifacts["metrics"].is_file()
        assert config.seed == base.seed
    assert not (base.project_root / "output/runs").exists()


def test_scope23_contract_upstream_uses_explicit_current_metrics_and_owned_output(
    tmp_path, monkeypatch
):
    path, base = _config(tmp_path)
    baseline = pipeline.run_pipeline(base, path)
    current = dict(baseline.metrics)
    current["mean_predictive_loss"] = float(current["mean_predictive_loss"]) + 1.25
    current_path = tmp_path / "actual-current-metrics.json"
    current_path.write_text(json.dumps(current))
    records = _record_real_pipeline(monkeypatch)
    candidate = robustness.make_upstream_candidate_runner(path, current_path)
    delta = candidate("b" * 40)
    assert delta["predictive_loss_abs_delta"] == pytest.approx(1.25)
    assert delta["posterior_entropy_abs_delta"] == 0
    assert records[0][0].fractalrabbit.upstream_commit == "b" * 40
    assert (
        load_config(path).fractalrabbit.upstream_commit
        == base.fractalrabbit.upstream_commit
    )
    assert records[0][0].fractalrabbit.params == base.fractalrabbit.params
    _assert_owned(records, base)


def test_scope23_contract_variance_realizations_stay_in_custom_collection(
    tmp_path, monkeypatch
):
    path, base = _config(tmp_path, engine="both")
    records = _record_real_pipeline(monkeypatch)
    output = tmp_path / "summaries/variance.json"
    robustness.run_and_write_variance_bands(path, output, n_realizations=2)
    payload = json.loads(output.read_text())
    assert payload["n_realizations"] == 2
    assert "frontier_no_location_mass" in payload["metrics"]
    assert {config.run_id for config, _ in records} == {
        "variance_realization_1",
        "variance_realization_2",
    }
    assert all(config.inference.model.engine == "reference" for config, _ in records)
    assert all(
        config.fractalrabbit.params == base.fractalrabbit.params
        for config, _ in records
    )
    _assert_owned(records, base)


def test_scope23_contract_fractal_sweep_preserves_inactive_fixture_control(
    tmp_path, monkeypatch
):
    path, base = _config(tmp_path, engine="both")
    records = _record_real_pipeline(monkeypatch)
    payload = run_fractal_dimension_sweep(path, dimensions=(1.1, 1.9))
    assert payload["status"] == "fixture_control_unvaried"
    assert payload["dimension_control_active"] is False
    assert not payload["errors"]
    assert [config.fractalrabbit.params.fractal_dimension for config, _ in records] == [
        1.1,
        1.9,
    ]
    assert all(config.inference.model.engine == "reference" for config, _ in records)
    assert (
        records[0][1].observations_path.read_bytes()
        == records[1][1].observations_path.read_bytes()
    )
    _assert_owned(records, base)


def test_scope23_contract_negative_controls_own_results_not_mutation_scratch(tmp_path):
    path, base = _config(tmp_path, fixture=LARGE)
    output = run_negative_controls(path)
    payload = json.loads(output.read_text())
    assert output == base.output_dir.parent / "negative_controls_summary.json"
    assert payload["n_controls"] == 4
    assert payload["all_passed"] is True
    assert (
        base.output_dir.parent / "negative_intact/observations.csv"
    ).read_bytes() == LARGE.read_bytes()
    assert (base.project_root / "output/negative_controls/shuffled_time.csv").is_file()
    assert (base.project_root / "output/negative_controls/shuffled_ids.csv").is_file()
    assert not (base.project_root / "output/runs").exists()
    for row in payload["controls"]:
        assert (base.output_dir.parent / row["run_id"] / "metrics.json").is_file()


def test_scope23_contract_external_repeats_return_actual_paths_and_preserve_axes(
    tmp_path, monkeypatch
):
    path, fixture = _config(tmp_path)
    base = replace(
        fixture, fractalrabbit=replace(fixture.fractalrabbit, execution="external")
    )
    records = []

    def producer(config, config_path):
        actual = tmp_path / "producer_return" / config.run_id / "observations.csv"
        actual.parent.mkdir(parents=True)
        actual.write_bytes(SMALL.read_bytes())
        records.append(config)
        return PipelineResult(
            config.run_id, config_path, actual.parent, actual, {}, None, (), {}, {}
        )

    monkeypatch.setattr(full_study, "run_pipeline", producer)
    result = full_study._run_external_realization_repeats(base, path)
    assert [c.fractalrabbit.params.fractal_dimension for c in records] == list(
        base.robustness.external_realization_fractal_dimensions
    )
    assert [c.seed for c in records] == [base.seed + i + 1 for i in range(len(records))]
    for config in records:
        assert config.output_dir == base.output_dir.parent / config.run_id
        assert config.fractalrabbit.output_prefix == config.run_id
        assert config.fractalrabbit.execution == "external"
        assert (
            config.fractalrabbit.upstream_commit == base.fractalrabbit.upstream_commit
        )
        assert config.inference.model.engine == "reference"
    assert set(result.values()) == {
        tmp_path / "producer_return" / c.run_id / "observations.csv" for c in records
    }


def test_scope23_contract_forward_gate_receivers_never_reuse_stale_external(
    tmp_path, monkeypatch
):
    current = tmp_path / "owned/current.csv"
    current.parent.mkdir()
    current.write_bytes(SMALL.read_bytes())
    stale = tmp_path / "output/runs/fractalrabbit_external/observations.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("not the completed run")

    def reader(*args, **kwargs):
        output = Path(args[-1])
        observed = kwargs.get(
            "observations_path", kwargs.get("reference_observations_path")
        )
        if len(args) == 3:
            observed = args[1]
        sequence = parse_waypoint_csv(observed if observed is not None else LARGE)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"status": "completed", "n_source_rows": len(sequence)})
        )
        return output

    for name in (
        "run_and_write_latent_regime_hmm",
        "run_and_write_neural_baseline",
        "run_and_write_learned_stay_rate",
        "run_and_write_heldout_reporting_process",
        "run_cross_generator_transfer",
        "run_and_write_efe_location_frontier",
        "run_and_write_linkage_risk_frontier",
        "run_and_write_poisoning_robustness",
    ):
        monkeypatch.setattr(forward_gates, name, reader)
    summary_path = forward_gates.run_and_write_forward_gates(
        tmp_path, external_observations=current
    )
    summary = json.loads(summary_path.read_text())
    assert summary["external_completed_count"] == 8
    for row in summary["rows"]:
        data = json.loads((tmp_path / row["artifact"]).read_text())
        expected = SMALL if row["lane"] == "external" else LARGE
        assert data["n_source_rows"] == len(parse_waypoint_csv(expected))
    forward_gates.run_and_write_forward_gates(tmp_path, external_observations=None)
    summary = json.loads(summary_path.read_text())
    assert summary["external_completed_count"] == 0
    assert summary["fixture_completed_count"] == 8
    assert all(
        row["status"] == "unavailable"
        for row in summary["rows"]
        if row["lane"] == "external"
    )
    assert stale.read_text() == "not the completed run"


@pytest.mark.parametrize("lane", ["fixture", "both"])
def test_scope23_contract_full_orchestration_preserves_current_result_custody(
    tmp_path, monkeypatch, lane
):
    _, config = _config(tmp_path)
    monkeypatch.setattr(full_study, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["07_run_full_study.py", "--lane", lane])
    external = tmp_path / "returned_by_producer/observations.csv"
    external.parent.mkdir()
    external.write_text("current-external")
    metrics = external.with_name("metrics.json")
    metrics.write_text('{"mean_predictive_loss": 0.125}')
    stale = tmp_path / "output/runs/fractalrabbit_external/observations.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale-external")
    stale.with_name("metrics.json").write_text('{"mean_predictive_loss": 900}')
    result = PipelineResult(
        "custom",
        Path("external.yaml"),
        external.parent,
        external,
        {},
        None,
        (),
        {"metrics": metrics},
        {},
    )
    readers, forward_sources, current_losses = {}, [], []
    reader_names = {
        "run_and_write_anticipatory",
        "run_and_write_active_sensing",
        "run_and_write_anomaly",
        "run_and_write_structure_recovery",
        "run_and_write_mission_collection",
        "run_and_write_frontier",
        "run_and_write_return_structure",
        "run_and_write_privacy_frontier",
        "run_and_write_hardened_anomaly",
        "run_and_write_circadian_phase",
        "run_and_write_partial_observability",
    }

    def boundary(name):
        def call(*args, **kwargs):
            if name in reader_names:
                source = Path(args[1])
                readers[name] = (
                    source.read_text() if source.is_file() else "unavailable"
                )
            return None

        return call

    for name, value in list(vars(full_study).items()):
        if callable(value) and name.startswith(
            ("run_", "write_", "verify_", "generate_", "render_", "build_open_")
        ):
            monkeypatch.setattr(full_study, name, boundary(name))
    external_config = replace(
        config, fractalrabbit=replace(config.fractalrabbit, execution="external")
    )
    monkeypatch.setattr(
        full_study,
        "load_config",
        lambda path: (
            external_config
            if Path(path).name == "fractalrabbit_external.yaml"
            else config
        ),
    )
    monkeypatch.setattr(full_study, "print_preflight", lambda: None)
    monkeypatch.setattr(full_study, "external_lane_status", lambda: (True, []))
    monkeypatch.setattr(
        full_study,
        "FractalRabbitRunner",
        lambda config: SimpleNamespace(
            prepare=lambda: None, source_dir=tmp_path / "absent-cache"
        ),
    )
    monkeypatch.setattr(full_study, "run_pipeline", lambda *args: result)
    monkeypatch.setattr(full_study, "run_pipeline_from_path", lambda *args: result)
    monkeypatch.setattr(
        full_study, "_run_external_realization_repeats", lambda *args: {}
    )
    monkeypatch.setattr(full_study, "_check_eval_contract", lambda: {"passed": True})

    def current_metrics(config_path, metrics_path):
        current_losses.append(
            json.loads(Path(metrics_path).read_text())["mean_predictive_loss"]
        )
        return None

    def forward(**kwargs):
        source = kwargs["external_observations"]
        forward_sources.append(
            Path(source).read_text() if source is not None else "unavailable"
        )

    monkeypatch.setattr(full_study, "make_upstream_candidate_runner", current_metrics)
    monkeypatch.setattr(full_study, "run_and_write_forward_gates", forward)
    assert full_study.main() == 0
    expected = "current-external" if lane == "both" else "unavailable"
    assert readers == dict.fromkeys(reader_names, expected)
    assert forward_sources == [expected]
    assert current_losses == ([0.125] if lane == "both" else [])
    assert stale.read_text() == "stale-external"
