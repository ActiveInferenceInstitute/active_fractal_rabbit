from __future__ import annotations

import json
from pathlib import Path

from active_fractal_rabbit.synthetic_task_utility_privacy import (
    SYNTHETIC_TASK_UTILITY_PRIVACY_SCHEMA,
    run_and_write_synthetic_task_utility_privacy,
    synthetic_task_utility_privacy,
)

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_task_utility_privacy_denies_release_claim_when_joint_gate_fails() -> (
    None
):
    payload = synthetic_task_utility_privacy(ROOT)

    assert payload["schema"] == SYNTHETIC_TASK_UTILITY_PRIVACY_SCHEMA
    assert payload["status"] == "release_utility_operating_point_found"
    assert payload["any_release_utility_claim_allowed"] is True
    assert payload["best_minimized_operating_regime"] == "grid_2x2"
    assert payload["rows"]
    assert any(row["release_utility_claim_allowed"] for row in payload["rows"])
    assert "no empirical" in payload["claim_boundary"]


def test_synthetic_task_utility_privacy_writer(tmp_path: Path) -> None:
    out = run_and_write_synthetic_task_utility_privacy(
        ROOT,
        tmp_path / "synthetic_task_utility_privacy.json",
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == SYNTHETIC_TASK_UTILITY_PRIVACY_SCHEMA
    assert payload["regime_count"] == len(payload["rows"])
