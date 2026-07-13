"""Tests for the within-regime model comparison (where AIF leads) — real dicts, no mocks.

Locks the honest claim shape: active inference is the lowest-loss IMPLEMENTED predictor under the
noisy regime, the belief-marginalizing methods dominate the raw-observation methods, and the lead
over the strongest non-AIF method is a non-CI-separated TIE (not a clean win, not global SOTA).
"""

from __future__ import annotations

import json
import math

import pytest

from active_fractal_rabbit.model_comparison import (
    ModelComparisonError,
    build_model_comparison,
    run_and_write_model_comparison,
)


def _write_gate(root, rows, *, n_states=None):
    runs = root / "output" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    lane_payload = {
        "eps": 0.3,
        "candidate_rows": rows,
        "strongest_fair_baseline": "switching_noise_hmm",
        "clustered_ci_low": -0.01,
        "clustered_ci_high": 0.02,
    }
    if n_states is not None:
        lane_payload["n_states"] = n_states
    (runs / "sota_program_gate.json").write_text(
        json.dumps({"lanes": {"fractalrabbit": lane_payload}}),
        encoding="utf-8",
    )


# A realistic noisy-lane roster: belief-marginalizing methods lead, raw-obs/point methods trail.
_ROWS = [
    {
        "candidate": "precision_return_aif",
        "family": "active_inference_candidate",
        "mean_loss": 1.139,
        "loss_ci_low": 1.10,
        "loss_ci_high": 1.18,
    },
    {
        "candidate": "soft_filter",
        "family": "active_inference_reference",
        "mean_loss": 1.140,
        "loss_ci_low": 1.10,
        "loss_ci_high": 1.18,
    },
    {
        "candidate": "switching_noise_hmm",
        "family": "classical_sequence",
        "mean_loss": 1.144,
        "loss_ci_low": 1.11,
        "loss_ci_high": 1.18,
    },
    {
        "candidate": "particle_filter",
        "family": "classical_state_space",
        "mean_loss": 1.165,
        "loss_ci_low": 1.12,
        "loss_ci_high": 1.21,
    },
    {
        "candidate": "hard_filter",
        "family": "transparent_filter_baseline",
        "mean_loss": 1.40,
        "loss_ci_low": 1.36,
        "loss_ci_high": 1.44,
    },
    {
        "candidate": "first_order_markov",
        "family": "transparent_markov_baseline",
        "mean_loss": 2.07,
        "loss_ci_low": 2.0,
        "loss_ci_high": 2.1,
    },
    {
        "candidate": "online_base_rate",
        "family": "transparent_null",
        "mean_loss": 2.44,
        "loss_ci_low": 2.4,
        "loss_ci_high": 2.5,
    },
]


def test_aif_leads_in_regime_but_tie_is_not_ci_separated(tmp_path):
    # n_states is fixture-only metadata (not derived from _ROWS) added specifically to
    # exercise chance_loss_nats = round(math.log(n_states), 6) against an independently
    # computed expected value — a log -> log2 mutation in the source would flip this.
    _write_gate(tmp_path, _ROWS, n_states=20)
    d = build_model_comparison(tmp_path)
    assert d["aif_leads"] is True
    assert d["aif_rank"] == 1
    assert d["leader_group"] == "active_inference"
    assert d["chance_loss_nats"] == round(math.log(20), 6)
    # The lead over the strongest non-AIF method (switching HMM) is a TIE, not a clean win.
    assert d["strongest_non_aif_method"] == "switching_noise_hmm"
    assert d["aif_beats_strongest_non_aif_ci_separated"] is False
    assert d["aif_vs_strongest_non_aif_paired_ci_low"] == -0.01
    assert d["aif_vs_strongest_non_aif_paired_ci_high"] == 0.02
    assert 0.0 < d["aif_vs_strongest_non_aif_gap"] < 0.02
    # precision_return_aif (1.139) is best_aif; soft_filter (1.140) is the *next* belief-
    # marginalizing row after it (family active_inference_reference is also tagged group
    # "active_inference", so it is the tie partner, distinct from strongest_non_aif above).
    assert d["next_belief_method"] == "soft_filter"
    assert d["aif_vs_next_belief_gap"] == round(1.140 - 1.139, 6)
    assert d["aif_marginal_ci_disjoint_from_next_belief"] is False
    assert "descriptive, not a paired test" in d["verdict"]
    # hard_filter (1.40) is the lowest-loss row outside the belief-marginalizing groups.
    assert d["best_nonbelief_method"] == "hard_filter"
    assert d["best_nonbelief_loss"] == round(1.40, 6)


def test_belief_methods_dominate_non_belief(tmp_path):
    _write_gate(tmp_path, _ROWS)
    d = build_model_comparison(tmp_path)
    # the four belief-marginalizing methods occupy the top of the leaderboard
    top4 = [r["group"] for r in d["leaderboard"][:4]]
    assert all(g in {"active_inference", "belief_marginalizing"} for g in top4)
    assert d["n_belief_methods"] == 4
    # there is a clear positive gap from the best belief method to the best non-belief method
    assert d["belief_vs_nonbelief_gap"] > 0.1
    # the verdict keeps the boundaries (not global SOTA, not oracle-beating)
    assert "not a global" in d["verdict"].lower()
    assert "does not beat the oracle" in d["verdict"].lower()


def test_aif_separation_uses_paired_lane_interval_not_marginal_overlap(tmp_path):
    _write_gate(tmp_path, _ROWS)
    gate_path = tmp_path / "output" / "runs" / "sota_program_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    lane = gate["lanes"]["fractalrabbit"]
    lane["clustered_ci_low"] = 0.001
    lane["clustered_ci_high"] = 0.009
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    result = build_model_comparison(tmp_path)
    # Marginal intervals still overlap in _ROWS, but the paired difference clears.
    assert result["aif_beats_strongest_non_aif_ci_separated"] is True


def test_missing_gate_raises(tmp_path):
    with pytest.raises(ModelComparisonError):
        build_model_comparison(tmp_path)


def test_run_and_write_emits_artifact(tmp_path):
    _write_gate(tmp_path, _ROWS)
    path = run_and_write_model_comparison(tmp_path)
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "active-fractal-rabbit-model-comparison-v2"
    assert payload["aif_rank"] == 1
