"""Offline boundary regressions: real files/parsers/publication, recorded transports only."""

from __future__ import annotations

import importlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from active_fractal_rabbit import fractalrabbit as boundary
from active_fractal_rabbit import pipeline
from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.observation_sources import resolve_observations_path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data/fixtures/fractalrabbit_sample.csv"


def _config(tmp_path):
    directory = tmp_path / "configs"
    directory.mkdir()
    path = directory / "external.yaml"
    raw = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    raw["output_dir"] = str(tmp_path / "owned" / "current")
    raw["fractalrabbit"].update(execution="external", cache_dir=str(tmp_path / "cache"))
    raw["model"]["engine"] = "reference"
    path.write_text(yaml.safe_dump(raw))
    return path, load_config(path)


def _transport(
    monkeypatch,
    config,
    *,
    dirty="",
    origin=None,
    wrong_root=False,
    wrong_head=False,
    build_dirty=False,
    jar_count=1,
    xy=None,
    emit=True,
    places=None,
    process_failure=False,
    build_failure=False,
    mutate_input=None,
    resolved_commit=None,
):
    runner = boundary.FractalRabbitRunner(config.fractalrabbit)
    commit = resolved_commit or config.fractalrabbit.upstream_commit
    runner.source_dir.mkdir(parents=True, exist_ok=True)
    old_jar = runner.source_dir / "target/fractalrabbit-1.0-jar-with-dependencies.jar"
    old_jar.parent.mkdir(exist_ok=True)
    old_jar.write_bytes(b"old cached jar")
    (runner.source_dir / "ignored.java").write_text(
        "ignored source must not enter build"
    )
    calls = []
    monkeypatch.setattr(
        boundary.FractalRabbitRunner,
        "_check_external_requirements",
        lambda self: (Path("/test/java"), Path("/test/jdk")),
    )

    def run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        cwd = Path(kwargs.get("cwd", runner.source_dir))
        stdout = ""
        if argv[:2] == ["git", "clone"]:
            Path(argv[-1]).mkdir(parents=True)
        elif argv[:3] == ["git", "rev-parse", "--show-toplevel"]:
            stdout = str(runner.source_dir.parent if wrong_root else runner.source_dir)
        elif argv[:3] == ["git", "remote", "get-url"]:
            stdout = (
                origin if origin is not None else config.fractalrabbit.upstream_repo
            )
        elif argv[:2] == ["git", "status"]:
            stdout = (
                dirty
                if cwd == runner.source_dir
                else (" M pom.xml" if build_dirty else "")
            )
        elif argv[:3] == ["git", "rev-parse", "--verify"]:
            stdout = commit
        elif argv[:3] == ["git", "rev-parse", "HEAD"]:
            stdout = "0" * 40 if wrong_head else commit
        elif argv[:3] == ["git", "rev-parse", "HEAD^{tree}"]:
            stdout = "a" * 40
        elif argv[0] == "mvn":
            assert cwd != runner.source_dir
            assert not (cwd / "ignored.java").exists()
            assert not (cwd / "target").exists()
            if build_failure:
                raise subprocess.CalledProcessError(17, argv)
            (cwd / "target").mkdir()
            for index in range(jar_count):
                (
                    cwd / "target" / f"historical-{index}-jar-with-dependencies.jar"
                ).write_bytes(b"fresh built jar")
        elif argv[0] == "/test/java":
            assert boundary.file_sha256(Path(argv[2])) != boundary.file_sha256(old_jar)
            assert Path(argv[-2]).is_file()
            assert not Path(argv[-1] + "XY.csv").exists()
            if emit:
                Path(argv[-1] + "XY.csv").write_text(
                    FIXTURE.read_text() if xy is None else xy
                )
            if places is not None:
                Path(argv[-1] + "PLACES.csv").write_text(places)
            if mutate_input == "parameters":
                Path(argv[-2]).write_text("modified parameter snapshot")
            if mutate_input == "jar":
                Path(argv[2]).write_bytes(b"modified executable")
            if process_failure:
                raise subprocess.CalledProcessError(23, argv)
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(boundary.subprocess, "run", run)
    return runner, old_jar, calls


@pytest.mark.parametrize("dirty", [" M pom.xml", "M  pom.xml", "?? new.java"])
def test_scope23_contract_dirty_sources_refuse_before_build(
    tmp_path, monkeypatch, dirty
):
    _, config = _config(tmp_path)
    runner, old, calls = _transport(monkeypatch, config, dirty=dirty)
    with pytest.raises(boundary.FractalRabbitUnavailable, match="dirty"):
        runner.prepare()
    assert old.read_bytes() == b"old cached jar"
    assert not any(
        argv[0] == "mvn" or argv[:2] == ["git", "checkout"] for argv, _ in calls
    )


@pytest.mark.parametrize(
    "failure", ["origin", "root", "head", "build_dirty", "zero_jars", "many_jars"]
)
def test_scope23_contract_unverifiable_build_never_reuses_old_jar(
    tmp_path, monkeypatch, failure
):
    _, config = _config(tmp_path)
    options = {
        "origin": {"origin": "wrong-repository"},
        "root": {"wrong_root": True},
        "head": {"wrong_head": True},
        "build_dirty": {"build_dirty": True},
        "zero_jars": {"jar_count": 0},
        "many_jars": {"jar_count": 2},
    }[failure]
    runner, old, calls = _transport(monkeypatch, config, **options)
    with pytest.raises(boundary.FractalRabbitUnavailable):
        runner.prepare()
    assert old.read_bytes() == b"old cached jar"
    assert not any(argv[0] == "/test/java" for argv, _ in calls)


def test_scope23_contract_alternate_revision_and_actual_executable_are_bound(
    tmp_path, monkeypatch
):
    path, base = _config(tmp_path)
    config = replace(
        base, fractalrabbit=replace(base.fractalrabbit, upstream_commit="b" * 40)
    )
    _, old, calls = _transport(monkeypatch, config)
    result = pipeline.run_pipeline(config, path)
    receipt = json.loads(result.artifacts["provenance_manifest"].read_text())
    build = receipt["observation_bundle"]["build"]
    assert (
        load_config(path).fractalrabbit.upstream_commit
        != config.fractalrabbit.upstream_commit
    )
    assert build["commit"] == receipt["upstream_commit"] == "b" * 40
    assert build["tree"] == "a" * 40
    assert Path(build["jar_path"]).read_bytes() == b"fresh built jar"
    assert build["jar_sha256"] == boundary.file_sha256(Path(build["jar_path"]))
    assert old.read_bytes() == b"old cached jar"
    assert result.metadata["fractalrabbit_commit"] == "b" * 40
    assert resolve_observations_path(config) == result.observations_path
    assert any(
        argv[:3] == ["git", "checkout", "--detach"] and argv[-1] == "b" * 40
        for argv, _ in calls
    )


def test_scope23_contract_each_execution_rebuilds_and_retains_identical_fresh_output(
    tmp_path, monkeypatch
):
    _, config = _config(tmp_path)
    runner, old, calls = _transport(monkeypatch, config)
    prefix = tmp_path / "result"
    Path(str(prefix) + "XY.csv").write_text(FIXTURE.read_text())
    Path(str(prefix) + "PLACES.csv").write_text("stale invalid companion")
    first, second = runner.run(prefix), runner.run(prefix)
    assert first.xy_path != second.xy_path != Path(str(prefix) + "XY.csv")
    assert first.xy_sha256 == second.xy_sha256 == boundary.file_sha256(FIXTURE)
    assert first.places_path is second.places_path is None
    assert first.build.jar_path != second.build.jar_path
    assert sum(argv[0] == "mvn" for argv, _ in calls) == 2
    assert old.read_bytes() == b"old cached jar"


@pytest.mark.parametrize(
    "options",
    [
        {"emit": False},
        {"process_failure": True},
        {"xy": "wrong header\n"},
        {"build_failure": True},
        {"mutate_input": "parameters"},
        {"mutate_input": "jar"},
        {"xy": "ID,Days,x(km),y(km)\n"},
        {"xy": "ID,Days,x(km),y(km)\n0,nan,0,0\n"},
        {"places": "wrong header\n"},
        {"places": "Traveler ID,Days,Place ID\n"},
        {"places": "Traveler ID,Days,Place ID\n0,nan,7\n"},
        {"places": "Traveler ID,Days,Place ID\n9,0,7\n"},
    ],
)
def test_scope23_contract_failed_bundle_preserves_previous_completed_run(
    tmp_path, monkeypatch, options
):
    path, config = _config(tmp_path)
    config.output_dir.mkdir(parents=True)
    previous = {
        name: (name + " old").encode()
        for name in [
            "observations.csv",
            "parameters.csv",
            "metrics.json",
            "provenance_manifest.json",
            config.fractalrabbit.output_prefix + "XY.csv",
            config.fractalrabbit.output_prefix + "PLACES.csv",
        ]
    }
    for name, content in previous.items():
        (config.output_dir / name).write_bytes(content)
    _transport(monkeypatch, config, **options)
    with pytest.raises(
        (ValueError, boundary.FractalRabbitUnavailable, subprocess.CalledProcessError)
    ):
        pipeline.run_pipeline(config, path)
    assert {
        name: (config.output_dir / name).read_bytes() for name in previous
    } == previous
    assert set(p.name for p in config.output_dir.iterdir()) == set(previous)


def test_scope23_contract_aligned_companion_and_explicit_absence_survive_pipeline(
    tmp_path, monkeypatch
):
    path, config = _config(tmp_path)
    rows = boundary.parse_waypoint_csv(FIXTURE)
    places = "Traveler ID,Days,Place ID\n" + "".join(
        f"{r.traveller_id},{r.days},7\n" for r in rows
    )
    _transport(monkeypatch, config, places=places)
    with_places = pipeline.run_pipeline(config, path)
    assert with_places.metadata["places_available"] is True
    assert with_places.metrics["place_n_unique"] == 1
    _transport(monkeypatch, config)
    absent = pipeline.run_pipeline(config, path)
    assert absent.metadata["places_available"] is False
    assert absent.metadata["observation_bundle"]["places_path"] is None
    assert "place_n_unique" not in absent.metrics
    assert (
        "true_place_id"
        not in absent.artifacts["ground_truth"].read_text().splitlines()[0]
    )
    assert Path(with_places.metadata["observation_bundle"]["places_path"]).exists()


def test_scope23_contract_aliased_collection_publishes_live_bundle_paths(
    tmp_path, monkeypatch
):
    path, base = _config(tmp_path)
    collection = tmp_path / "collection"
    collection.mkdir()
    alias = tmp_path / "collection-alias"
    alias.symlink_to(collection, target_is_directory=True)
    config = replace(base, output_dir=alias / "current")
    rows = boundary.parse_waypoint_csv(FIXTURE)
    places = "Traveler ID,Days,Place ID\n" + "".join(
        f"{row.traveller_id},{row.days},7\n" for row in rows
    )
    _transport(monkeypatch, config, places=places)

    result = pipeline.run_pipeline(config, path)

    receipt = json.loads(result.artifacts["provenance_manifest"].read_text())
    bundle = receipt["observation_bundle"]
    for member in ("xy_path", "places_path", "parameters_path"):
        published = Path(bundle[member])
        assert published.is_file()
        assert published.resolve().is_relative_to(config.output_dir.resolve())
        assert not any(part.startswith(".pipeline-") for part in published.parts)
        digest_key = member.removesuffix("_path") + "_sha256"
        assert boundary.file_sha256(published) == bundle[digest_key]
    assert Path(receipt["source_csv"]) == Path(bundle["xy_path"])
    assert result.metadata["observation_bundle"] == bundle
    assert all(artifact.is_file() for artifact in result.artifacts.values())


def test_scope23_contract_modified_prepared_jar_is_not_executed(tmp_path, monkeypatch):
    _, config = _config(tmp_path)
    runner, _, calls = _transport(monkeypatch, config)
    build = runner.prepare()
    build.jar_path.write_bytes(b"changed")
    monkeypatch.setattr(runner, "prepare", lambda: build)
    with pytest.raises(boundary.FractalRabbitUnavailable, match="digest"):
        runner.run(tmp_path / "result")
    assert not any(argv[0] == "/test/java" for argv, _ in calls)


def test_scope23_contract_publication_rolls_back_before_failed_completion(
    tmp_path, monkeypatch
):
    staging, destination = tmp_path / "stage", tmp_path / "run"
    staging.mkdir()
    destination.mkdir()
    for name in ["metrics.json", "observations.csv", "provenance_manifest.json"]:
        (staging / name).write_text("new " + name)
        (destination / name).write_text("old " + name)
    (destination / "historical.txt").write_text("untouched")
    original = pipeline.os.replace
    failed = False

    def fail_completion(source, target):
        nonlocal failed
        if Path(target).name == "provenance_manifest.json" and not failed:
            failed = True
            raise OSError("publication refused")
        return original(source, target)

    monkeypatch.setattr(pipeline.os, "replace", fail_completion)
    with pytest.raises(OSError, match="publication refused"):
        pipeline._publish_run(staging, destination)
    for name in ["metrics.json", "observations.csv", "provenance_manifest.json"]:
        assert (destination / name).read_text() == "old " + name
    assert (destination / "historical.txt").read_text() == "untouched"


@pytest.mark.parametrize(
    "module,writer",
    [
        ("sequence_baselines", "run_and_write_sequence_baselines"),
        ("missingness_model", "run_and_write_missingness_model"),
        ("return_phase_prior", "run_and_write_return_phase_prior"),
    ],
)
def test_scope23_contract_receivers_use_bound_canonical_not_stale_raw(
    tmp_path, monkeypatch, module, writer
):
    path, config = _config(tmp_path)
    _transport(monkeypatch, config)
    result = pipeline.run_pipeline(config, path)
    old_raw = config.output_dir / f"{config.fractalrabbit.output_prefix}XY.csv"
    old_raw.write_text("stale invalid raw CSV")
    fn = getattr(importlib.import_module(f"active_fractal_rabbit.{module}"), writer)
    output = tmp_path / "analysis.json"
    fn(path, output)
    expected = output.read_bytes()
    old_raw.write_text("another stale raw CSV")
    fn(path, output)
    assert output.read_bytes() == expected
    result.observations_path.unlink()
    with pytest.raises(FileNotFoundError):
        fn(path, output)
    assert output.read_bytes() == expected


def test_scope23_contract_changed_canonical_bytes_invalidate_completion(
    tmp_path, monkeypatch
):
    path, config = _config(tmp_path)
    _transport(monkeypatch, config)
    result = pipeline.run_pipeline(config, path)
    result.artifacts["metrics"].write_text("{}")
    with pytest.raises(ValueError, match="metrics"):
        resolve_observations_path(config)


@pytest.mark.parametrize("failed_stage", ["analysis", "writer"])
def test_scope23_contract_failed_analysis_keeps_last_completed_bundle(
    tmp_path, monkeypatch, failed_stage
):
    path, config = _config(tmp_path)
    _transport(monkeypatch, config)
    pipeline.run_pipeline(config, path)
    previous = {p.name: p.read_bytes() for p in config.output_dir.iterdir()}
    if failed_stage == "analysis":

        def fail_analysis(*args, **kwargs):
            raise RuntimeError("analysis refused")

        monkeypatch.setattr(pipeline, "_objective_evaluation", fail_analysis)
    else:
        original = pipeline._write_json

        def fail_metadata(path, payload):
            if path.name == "metadata.json":
                raise RuntimeError("writer refused")
            return original(path, payload)

        monkeypatch.setattr(pipeline, "_write_json", fail_metadata)
    with pytest.raises(RuntimeError, match="refused"):
        pipeline.run_pipeline(config, path)
    assert {p.name: p.read_bytes() for p in config.output_dir.iterdir()} == previous


def test_scope23_contract_configured_revision_is_not_misreported_as_observed_commit(
    tmp_path, monkeypatch
):
    path, base = _config(tmp_path)
    config = replace(
        base,
        fractalrabbit=replace(base.fractalrabbit, upstream_commit="selected-revision"),
    )
    _transport(monkeypatch, config, resolved_commit="c" * 40)
    result = pipeline.run_pipeline(config, path)
    receipt = json.loads(result.artifacts["provenance_manifest"].read_text())
    assert receipt["configured_upstream_commit"] == "selected-revision"
    assert (
        receipt["upstream_commit"]
        == result.metadata["fractalrabbit_commit"]
        == "c" * 40
    )
    assert (
        receipt["observation_bundle"]["build"]["requested_revision"]
        == "selected-revision"
    )
    assert resolve_observations_path(config) == result.observations_path
    wrong = replace(
        config,
        fractalrabbit=replace(config.fractalrabbit, upstream_commit="other-revision"),
    )
    with pytest.raises(ValueError, match="configured run"):
        resolve_observations_path(wrong)
