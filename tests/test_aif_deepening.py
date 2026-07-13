from __future__ import annotations

import json
from pathlib import Path

from active_fractal_rabbit.aif_deepening import (
    AIF_DEEPENING_SCHEMA,
    aif_deepening_candidates,
    run_and_write_aif_deepening_candidates,
)

ROOT = Path(__file__).resolve().parents[1]


def test_aif_deepening_admits_only_scored_online_adaptation_without_promotion() -> None:
    payload = aif_deepening_candidates(ROOT)

    assert payload["schema"] == AIF_DEEPENING_SCHEMA
    assert payload["status"] == "scored_candidate_no_predictive_promotion"
    assert payload["predictive_gate_clears"] is False
    assert payload["scored_protocol_candidate_count"] == 1
    assert {row["candidate_id"] for row in payload["rows"]} == {
        "factorial_matched_prior_tensor_provenance",
        "multiscale_soft_belief_backoff",
        "online_adaptation",
    }
    online = next(
        row for row in payload["rows"] if row["candidate_id"] == "online_adaptation"
    )
    assert online["protocol_status"] == "scored_predictive_candidate"
    assert online["predictive_gate_clears"] is False
    assert online["beats_fair_ci_count"] == 0
    assert all(
        row["protocol_status"].startswith("audit_only")
        for row in payload["rows"]
        if row["candidate_id"] != "online_adaptation"
    )
    assert "do not substitute for predictive improvement" in payload["boundary"]


def test_aif_deepening_writer(tmp_path: Path) -> None:
    out = run_and_write_aif_deepening_candidates(
        ROOT, tmp_path / "aif_deepening_candidates.json"
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == AIF_DEEPENING_SCHEMA
    assert (
        payload["locked_protocol"] == "Candidate.predict(PredictContext) -> np.ndarray"
    )
