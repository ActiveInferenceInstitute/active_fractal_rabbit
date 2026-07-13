from __future__ import annotations

import json
from pathlib import Path

from active_fractal_rabbit.privacy_utility_repeats import (
    PRIVACY_UTILITY_REPEATS_SCHEMA,
    privacy_utility_frontier_repeats,
    run_and_write_privacy_utility_repeats,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "default.yaml"


def test_privacy_utility_repeats_emit_unavailable_rows_without_fabrication(
    tmp_path: Path,
) -> None:
    project_root = _seed_repeat_project(tmp_path)
    payload = privacy_utility_frontier_repeats(
        project_root / "configs" / "default.yaml",
        project_root=project_root,
        realization_count=2,
        grid_regimes=(4, 2),
        n_shuffle=2,
    )

    assert payload["schema"] == PRIVACY_UTILITY_REPEATS_SCHEMA
    assert payload["status"] == "partial_repeats_unavailable"
    assert payload["completed_count"] == 1
    assert payload["unavailable_count"] == 2
    assert payload["all_required_completed"] is False
    assert payload["can_claim_graceful_repeated_tradeoff"] is False
    unavailable = [row for row in payload["rows"] if row["status"] == "unavailable"]
    assert unavailable
    assert all(
        "missing variance realization observations.csv" in row["reason"]
        for row in unavailable
    )


def test_repeat_gate_blocks_byte_identical_reruns(tmp_path: Path) -> None:
    """Bit-identical realizations are determinism reruns, not independent repeats."""

    project_root = _seed_repeat_project(tmp_path)
    runs = project_root / "output" / "runs"
    source = (
        project_root / "data" / "fixtures" / "fractalrabbit_sample_large.csv"
    ).read_text(encoding="utf-8")
    for idx in (1, 2):
        realization = runs / f"variance_realization_{idx}"
        realization.mkdir(parents=True)
        (realization / "observations.csv").write_text(source, encoding="utf-8")

    payload = privacy_utility_frontier_repeats(
        project_root / "configs" / "default.yaml",
        project_root=project_root,
        realization_count=2,
        grid_regimes=(4, 2),
        n_shuffle=2,
    )

    assert payload["completed_count"] == 3
    assert payload["distinct_draws"] == 1
    assert payload["duplicate_of_canonical_count"] == 2
    assert payload["independent_repeats"] is False
    assert payload["can_claim_graceful_repeated_tradeoff"] is False
    assert payload["can_claim_min_collection_replicates"] is False
    assert payload["status"] == "complete_duplicate_reruns_only"


def test_distinct_draws_unlock_independent_repeat_claim(tmp_path: Path) -> None:
    """A green repeat claim requires input files that are NOT all byte-identical."""

    project_root = _seed_repeat_project(tmp_path)
    runs = project_root / "output" / "runs"
    source = (
        project_root / "data" / "fixtures" / "fractalrabbit_sample_large.csv"
    ).read_text(encoding="utf-8")
    header, *rows = source.splitlines()

    identical = runs / "variance_realization_1"
    identical.mkdir(parents=True)
    (identical / "observations.csv").write_text(source, encoding="utf-8")

    distinct = runs / "variance_realization_2"
    distinct.mkdir(parents=True)
    kept = rows[: max(2, len(rows) // 2)]
    (distinct / "observations.csv").write_text(
        "\n".join([header, *kept]) + "\n", encoding="utf-8"
    )

    payload = privacy_utility_frontier_repeats(
        project_root / "configs" / "default.yaml",
        project_root=project_root,
        realization_count=2,
        grid_regimes=(4, 2),
        n_shuffle=2,
    )

    assert payload["completed_count"] == 3
    assert payload["distinct_draws"] >= 2
    assert payload["duplicate_of_canonical_count"] == 1
    assert payload["independent_repeats"] is True
    # Binding invariant: the gate can only report a repeat claim when draws differ.
    if payload["can_claim_graceful_repeated_tradeoff"]:
        assert payload["distinct_draws"] > 1


def test_privacy_utility_repeats_writer(tmp_path: Path) -> None:
    out = run_and_write_privacy_utility_repeats(
        CONFIG,
        tmp_path / "privacy_utility_frontier_repeats.json",
        project_root=ROOT,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == PRIVACY_UTILITY_REPEATS_SCHEMA
    assert payload["rows"][0]["realization_id"] == "canonical_fixture"


def _seed_repeat_project(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "data" / "fixtures").mkdir(parents=True)
    source_fixture = ROOT / "data" / "fixtures" / "fractalrabbit_sample_large.csv"
    target_fixture = tmp_path / "data" / "fixtures" / "fractalrabbit_sample_large.csv"
    target_fixture.write_text(
        source_fixture.read_text(encoding="utf-8"), encoding="utf-8"
    )
    config_text = CONFIG.read_text(encoding="utf-8").replace(
        "data/fixtures/fractalrabbit_sample_large.csv",
        "data/fixtures/fractalrabbit_sample_large.csv",
    )
    (tmp_path / "configs" / "default.yaml").write_text(config_text, encoding="utf-8")
    return tmp_path
