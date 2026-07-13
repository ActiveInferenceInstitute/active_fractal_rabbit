"""Artifact writers for the roadmap-extension phases.

Each writer falls back to the checked fixture when the external observations are
absent, so these run without Java/Maven. They exercise the writer orchestration
(lane assembly, JSON serialization, fixture-first path resolution) that the core-function
tests do not, and assert the artifact schema and key structure on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from active_fractal_rabbit.anomaly import run_and_write_anomaly
from active_fractal_rabbit.frontier import run_and_write_frontier
from active_fractal_rabbit.hardened_anomaly import run_and_write_hardened_anomaly
from active_fractal_rabbit.mission_collection import run_and_write_mission_collection
from active_fractal_rabbit.privacy_frontier import run_and_write_privacy_frontier
from active_fractal_rabbit.return_structure import run_and_write_return_structure
from active_fractal_rabbit.structure_recovery import run_and_write_structure_recovery

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "default.yaml"
MISSING = ROOT / "output" / "runs" / "does_not_exist" / "observations.csv"


def test_anomaly_writer_falls_back_to_fixture_and_writes_schema(tmp_path) -> None:
    out = run_and_write_anomaly(CONFIG, MISSING, tmp_path / "anomaly.json")
    payload = json.loads(Path(out).read_text())
    assert payload["schema"] == "active-fractal-rabbit-anomaly-v1"
    assert payload["source"].startswith("fixture:")
    assert (
        "vfe_auc_rare" in payload
        and "vfe_auc_random" in payload
        and "novelty_auc" in payload
    )
    assert payload["vfe_auc"] > payload["novelty_auc"]


def test_structure_recovery_writer_reports_both_baselines(tmp_path) -> None:
    out = run_and_write_structure_recovery(CONFIG, MISSING, tmp_path / "structure.json")
    payload = json.loads(Path(out).read_text())
    assert payload["schema"] == "active-fractal-rabbit-structure-recovery-v1"
    lane = payload["lanes"][0]
    for key in (
        "tv_uniform",
        "tv_dynamics_only",
        "tv_deep_latent",
        "tv_full_location",
        "preferred_jaccard",
    ):
        assert key in lane


def test_mission_collection_writer_reports_both_regimes(tmp_path) -> None:
    out = run_and_write_mission_collection(CONFIG, MISSING, tmp_path / "mission.json")
    payload = json.loads(Path(out).read_text())
    assert payload["schema"] == "active-fractal-rabbit-mission-collection-v1"
    assert "resolved" in payload and "unresolved" in payload
    assert isinstance(payload["resolved"]["efe_beats_pragmatic"], bool)
    assert payload["mission_cells"]


def test_frontier_writer_spans_modality_subsets(tmp_path) -> None:
    out = run_and_write_frontier(CONFIG, MISSING, tmp_path / "frontier.json")
    payload = json.loads(Path(out).read_text())
    assert payload["schema"] == "active-fractal-rabbit-frontier-v1"
    names = {row["modalities"] for row in payload["frontier"]}
    assert {"all", "no_location"} <= names


def test_return_structure_writer_reports_promote_or_null_verdict(tmp_path) -> None:
    out = run_and_write_return_structure(CONFIG, MISSING, tmp_path / "return.json")
    payload = json.loads(Path(out).read_text())
    assert payload["schema"] == "active-fractal-rabbit-return-structure-v1"
    assert payload["verdict"] in {"promote_return_conditioned_prior", "honest_null"}
    assert "move_step_entropy_reduction" in payload


def test_privacy_frontier_writer_reports_minimization_rows(tmp_path) -> None:
    out = run_and_write_privacy_frontier(CONFIG, MISSING, tmp_path / "privacy.json")
    payload = json.loads(Path(out).read_text())
    assert payload["schema"] == "active-fractal-rabbit-privacy-frontier-v1"
    names = {row["modalities"] for row in payload["frontier"]}
    assert {"all", "no_location"} <= names
    assert payload["caveat"].startswith("Synthetic")


def test_hardened_anomaly_writer_reports_operating_points(tmp_path) -> None:
    out = run_and_write_hardened_anomaly(CONFIG, MISSING, tmp_path / "hardened.json")
    payload = json.loads(Path(out).read_text())
    assert payload["schema"] == "active-fractal-rabbit-hardened-anomaly-v1"
    for probe in ("speed_regime_shift", "preferred_set_drift"):
        assert "vfe_auc" in payload[probe]
        assert "markov_operating_point" in payload[probe]
