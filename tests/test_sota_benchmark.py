"""Tests for the SOTA benchmark harness, locked task contract, and false-claim oracle.

These are the negative and positive controls that make the claim gate trustworthy:
the leakage probe must be excluded, every illegitimate path to a "performance
leading" claim must keep the gate closed, and at least one legitimate path must
open it (non-vacuity). No mocks -- all on the real fixture sequence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.observations import discretize_observations
from active_fractal_rabbit.sota_benchmark import (
    lock_traveller_split,
    run_and_write_sota_benchmark,
    run_sota_benchmark,
)

ROOT = Path(__file__).resolve().parents[1]


def _sequence():
    config = load_config(ROOT / "configs" / "default.yaml")
    sequence = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    return sequence, config.inference.smoothing


def _full_family_rows(loss: float = 9.9, brier: float = 1.9):
    """Locked-split rows covering the classical/neural/foundation/LLM families."""
    return [
        {
            "model": f"{fam}_provided",
            "family": fam,
            "mean_predictive_loss": loss,
            "brier_score": brier,
            "ece": 0.5,
            "runtime_seconds": 0.0,
        }
        for fam in (
            "classical_sequence",
            "neural_sequence",
            "foundation_mobility",
            "llm_mobility",
        )
    ]


def test_locked_split_is_traveller_disjoint() -> None:
    sequence, _ = _sequence()
    split = lock_traveller_split(sequence, holdout_fraction=0.4, seed=7)
    import numpy as np

    travellers = np.asarray([row.traveller_id for row in sequence.observations])
    is_holdout = split["is_holdout"]
    holdout_ids = set(int(t) for t in travellers[is_holdout])
    train_ids = set(int(t) for t in travellers[~is_holdout])
    assert holdout_ids.isdisjoint(train_ids)  # no traveller identity crosses the split
    assert split["n_holdout_travellers"] >= 1
    assert split["n_train_steps"] >= 1 and split["n_holdout_steps"] >= 1


def test_leakage_oracle_beats_baselines_but_is_excluded() -> None:
    """Negative control that FIRES: a peeking probe wins on loss yet cannot clear."""
    sequence, smoothing = _sequence()
    payload = run_sota_benchmark(sequence, smoothing=smoothing)
    oracle = payload["false_claim_oracle"]
    assert oracle["leakage_probe_beats_baselines"] is True
    assert oracle["leakage_probe_excluded"] is True
    # the lowest-loss row in the whole leaderboard is the leakage probe...
    best = payload["leaderboard"][0]
    assert best["model"] == "leakage_oracle_probe"
    # ...yet the gate is not cleared by it.
    assert payload["status"] == "sota_not_cleared"


def test_baselines_only_gate_is_not_vacuously_cleared() -> None:
    sequence, smoothing = _sequence()
    payload = run_sota_benchmark(sequence, smoothing=smoothing)
    assert payload["status"] == "sota_not_cleared"
    reasons = " ".join(payload["claim_gate_reasons"]).lower()
    assert "no eligible candidate" in reasons
    assert "fixture-only" in reasons or "external" in reasons
    # foundation and LLM families are absent -> they appear as a blocker.
    assert "foundation_mobility" in payload["missing_baseline_families"]
    assert "llm_mobility" in payload["missing_baseline_families"]


def test_positive_control_legitimate_candidate_clears_gate() -> None:
    """Non-vacuity: a clean, calibrated, externally-confirmed win opens the gate."""
    sequence, smoothing = _sequence()
    baseline_only = run_sota_benchmark(sequence, smoothing=smoothing)
    strongest = baseline_only["strongest_baseline_loss"]
    candidate = {
        "model": "synthetic_positive_control",
        "family": "active_inference",
        "mean_predictive_loss": strongest - 0.5,  # clear margin over strongest baseline
        "brier_score": 0.0,  # better calibration than any baseline
        "ece": 0.0,
        "runtime_seconds": 0.1,
    }
    payload = run_sota_benchmark(
        sequence,
        smoothing=smoothing,
        external_available=True,
        minimization_ok=True,
        integrity_ok=True,
        injected_candidate=candidate,
        injected_locked_rows=_full_family_rows(),
    )
    assert payload["candidate_evaluation"]["clears"] is True
    assert payload["missing_baseline_families"] == []
    assert payload["status"] == "sota_cleared", payload["claim_gate_reasons"]


def _clearing_kwargs(sequence, smoothing):
    baseline_only = run_sota_benchmark(sequence, smoothing=smoothing)
    strongest = baseline_only["strongest_baseline_loss"]
    candidate = {
        "model": "candidate",
        "family": "active_inference",
        "mean_predictive_loss": strongest - 0.5,
        "brier_score": 0.0,
        "ece": 0.0,
        "runtime_seconds": 0.1,
    }
    return dict(
        external_available=True,
        minimization_ok=True,
        integrity_ok=True,
        injected_candidate=candidate,
        injected_locked_rows=_full_family_rows(),
    )


def test_fixture_only_win_is_blocked() -> None:
    sequence, smoothing = _sequence()
    kwargs = _clearing_kwargs(sequence, smoothing)
    kwargs["external_available"] = False
    payload = run_sota_benchmark(sequence, smoothing=smoothing, **kwargs)
    assert payload["status"] == "sota_not_cleared"
    assert any("fixture-only" in r for r in payload["claim_gate_reasons"])


def test_minimization_regression_is_blocked() -> None:
    sequence, smoothing = _sequence()
    kwargs = _clearing_kwargs(sequence, smoothing)
    kwargs["minimization_ok"] = False
    payload = run_sota_benchmark(sequence, smoothing=smoothing, **kwargs)
    assert payload["status"] == "sota_not_cleared"
    assert any(
        "minimization" in r.lower() for r in payload["candidate_evaluation"]["failures"]
    )


def test_integrity_regression_is_blocked() -> None:
    sequence, smoothing = _sequence()
    kwargs = _clearing_kwargs(sequence, smoothing)
    kwargs["integrity_ok"] = False
    payload = run_sota_benchmark(sequence, smoothing=smoothing, **kwargs)
    assert payload["status"] == "sota_not_cleared"
    assert any(
        "integrity" in r.lower() for r in payload["candidate_evaluation"]["failures"]
    )


def test_single_metric_win_loses_on_calibration() -> None:
    sequence, smoothing = _sequence()
    baseline_only = run_sota_benchmark(sequence, smoothing=smoothing)
    strongest = baseline_only["strongest_baseline_loss"]
    candidate = {
        "model": "loss_only_winner",
        "family": "active_inference",
        "mean_predictive_loss": strongest - 0.5,
        "brier_score": 2.0,  # worse calibration than every baseline
        "ece": 0.9,
        "runtime_seconds": 0.1,
    }
    payload = run_sota_benchmark(
        sequence,
        smoothing=smoothing,
        external_available=True,
        minimization_ok=True,
        integrity_ok=True,
        injected_candidate=candidate,
        injected_locked_rows=_full_family_rows(),
    )
    assert payload["status"] == "sota_not_cleared"
    assert any(
        "calibration" in r.lower() for r in payload["candidate_evaluation"]["failures"]
    )


def test_determinism_same_seed_same_losses() -> None:
    sequence, smoothing = _sequence()
    a = run_sota_benchmark(sequence, smoothing=smoothing, seed=42)
    b = run_sota_benchmark(sequence, smoothing=smoothing, seed=42)
    la = [(r["model"], r["mean_predictive_loss"]) for r in a["leaderboard"]]
    lb = [(r["model"], r["mean_predictive_loss"]) for r in b["leaderboard"]]
    assert la == lb


def test_locked_rows_report_ordered_deterministic_bootstrap_ci() -> None:
    sequence, smoothing = _sequence()
    a = run_sota_benchmark(sequence, smoothing=smoothing, seed=3)
    b = run_sota_benchmark(sequence, smoothing=smoothing, seed=3)
    locked = [
        r
        for r in a["leaderboard"]
        if r["locked_split_scored"] and not r.get("injected")
    ]
    assert locked, "expected in-harness locked rows"
    for r in locked:
        assert r["loss_ci_low"] <= r["mean_predictive_loss"] <= r["loss_ci_high"]
    # bootstrap CI is deterministic across runs
    ci_a = [(r["model"], r["loss_ci_low"], r["loss_ci_high"]) for r in a["leaderboard"]]
    ci_b = [(r["model"], r["loss_ci_low"], r["loss_ci_high"]) for r in b["leaderboard"]]
    assert ci_a == ci_b


def test_ci_separation_guard_blocks_overlapping_win() -> None:
    """A candidate that beats the point loss but sits inside the baseline's CI
    band is not a separated win and must not clear."""
    from active_fractal_rabbit.sota_benchmark import _evaluate_candidate

    baseline_rows = [
        {
            "brier_score": 0.5,
            "mean_predictive_loss": 1.0,
            "locked_split_scored": True,
            "leakage": False,
        }
    ]
    common = dict(
        strongest_baseline_loss=1.0,
        strongest_baseline_ci_low=0.9,
        baseline_rows=baseline_rows,
        external_available=True,
        minimization_ok=True,
        integrity_ok=True,
    )
    overlapping = {
        "model": "c",
        "mean_predictive_loss": 0.95,
        "brier_score": 0.1,
        "locked_split_scored": True,
        "leakage": False,
    }
    ev = _evaluate_candidate(overlapping, **common)
    assert ev["ci_separated"] is False
    assert ev["clears"] is False
    assert any("CI-separated" in f for f in ev["failures"])
    separated = {**overlapping, "mean_predictive_loss": 0.5}
    ev2 = _evaluate_candidate(separated, **common)
    assert ev2["ci_separated"] is True
    assert ev2["clears"] is True


def test_injected_rows_are_flagged_and_production_path_uses_none() -> None:
    """Trust boundary: a clear that leans on caller-injected rows is recorded;
    the production writer never injects, so its gate is driven only by real rows."""
    sequence, smoothing = _sequence()
    # production-style: no injection
    plain = run_sota_benchmark(sequence, smoothing=smoothing)
    assert plain["used_injected_rows"] is False
    # positive control clears but is honestly flagged as injection-dependent
    payload = run_sota_benchmark(
        sequence, smoothing=smoothing, **_clearing_kwargs(sequence, smoothing)
    )
    assert payload["status"] == "sota_cleared"
    assert payload["used_injected_rows"] is True
    assert any("injected" in r.lower() for r in payload["claim_gate_reasons"])


def test_non_finite_candidate_metrics_cannot_clear_the_gate() -> None:
    """RedTeam CRITICAL: NaN/Inf loss or Brier must fail closed, not bypass the
    comparison guards. A positive control must still clear (non-vacuity)."""
    sequence, smoothing = _sequence()
    kwargs = _clearing_kwargs(sequence, smoothing)
    base_candidate = kwargs.pop("injected_candidate")
    for bad in (float("nan"), float("-inf"), float("inf")):
        payload = run_sota_benchmark(
            sequence,
            smoothing=smoothing,
            injected_candidate={**base_candidate, "mean_predictive_loss": bad},
            **kwargs,
        )
        assert payload["status"] == "sota_not_cleared", (
            f"non-finite loss {bad} cleared the gate"
        )
    # NaN Brier must also fail closed (calibration guard)
    payload = run_sota_benchmark(
        sequence,
        smoothing=smoothing,
        injected_candidate={**base_candidate, "brier_score": float("nan")},
        **kwargs,
    )
    assert payload["status"] == "sota_not_cleared"
    # control: the finite, calibrated, separated candidate still clears
    payload = run_sota_benchmark(
        sequence, smoothing=smoothing, injected_candidate=base_candidate, **kwargs
    )
    assert payload["status"] == "sota_cleared"


def test_oracle_verdict_is_derived_not_hardcoded() -> None:
    """RedTeam MEDIUM: the leakage-oracle verdict text must flip with the measured
    boolean (no green-by-construction prose)."""
    from active_fractal_rabbit.sota_benchmark import _oracle_summary

    fired = _oracle_summary({"mean_predictive_loss": 0.0, "leakage": True}, 0.37)
    assert fired["leakage_probe_beats_baselines"] is True
    assert "excluded by the leakage guard" in fired["verdict"]
    # negative control did NOT fire: a baseline matched/beat the probe
    not_fired = _oracle_summary({"mean_predictive_loss": 0.5, "leakage": True}, 0.37)
    assert not_fired["leakage_probe_beats_baselines"] is False
    assert "DID NOT FIRE" in not_fired["verdict"]


def test_figure_headline_tracks_gate_status() -> None:
    """RedTeam MEDIUM: the leaderboard headline must be derived from status, not a
    fixed literal that could contradict a cleared gate."""
    from active_fractal_rabbit.figures import sota_leaderboard_headline

    # not-cleared + no eligible candidate beats the strongest baseline (0.50 > 0.37)
    not_cleared_rows = [
        {
            "model": "return_boost_markov",
            "family": "transparent_null",
            "mean_predictive_loss": 0.37,
            "locked_split_scored": True,
            "leakage": False,
        },
        {
            "model": "cand",
            "family": "active_inference",
            "mean_predictive_loss": 0.50,
            "locked_split_scored": True,
            "leakage": False,
        },
    ]
    assert (
        "SOTA gate closed: no eligible candidate beats the strongest honest baseline"
        == sota_leaderboard_headline(not_cleared_rows, "sota_not_cleared", 0.37)
    )
    # cleared status yields the cleared headline, never the fixed "no candidate" literal
    assert "SOTA gate cleared" in sota_leaderboard_headline(
        not_cleared_rows, "sota_cleared", 0.37
    )
    # faithful refusal: status says not-cleared but a candidate beats -> neither false headline
    contradictory = [
        {
            "model": "c",
            "family": "active_inference",
            "mean_predictive_loss": 0.10,
            "locked_split_scored": True,
            "leakage": False,
        }
    ]
    h = sota_leaderboard_headline(contradictory, "sota_not_cleared", 0.37)
    assert "no eligible candidate beats" not in h


def test_full_stream_rows_cannot_assert_locked_split_eligibility() -> None:
    """RedTeam hardening: a caller-supplied full-stream row cannot claim
    locked_split_scored to satisfy a baseline family."""
    sequence, smoothing = _sequence()
    sneaky = [
        {
            "model": "fake",
            "family": "foundation_mobility",
            "mean_predictive_loss": 0.1,
            "brier_score": 0.1,
            "locked_split_scored": True,
        }
    ]
    payload = run_sota_benchmark(
        sequence, smoothing=smoothing, extra_full_stream_rows=sneaky
    )
    fake_row = next(r for r in payload["leaderboard"] if r["model"] == "fake")
    assert fake_row["locked_split_scored"] is False
    assert "foundation_mobility" in payload["missing_baseline_families"]


def test_minimization_and_integrity_readers_match_real_artifacts() -> None:
    """The production readers must be able to return True, not silently None."""
    from active_fractal_rabbit.sota_benchmark import _integrity_ok, _minimization_ok

    runs = ROOT / "output" / "runs"
    # These artifacts exist after a study run; skip cleanly if a bare checkout.
    if (runs / "efe_location_frontier.json").is_file():
        assert _minimization_ok(runs / "efe_location_frontier.json") is True
    if (runs / "poisoning_robustness.json").is_file():
        assert _integrity_ok(runs / "poisoning_robustness.json") is False
    # missing / malformed -> None (unverified), never a silent True
    assert _minimization_ok(runs / "does_not_exist.json") is None
    assert _integrity_ok(runs / "does_not_exist.json") is None


def test_writer_emits_four_artifacts(tmp_path) -> None:
    paths = run_and_write_sota_benchmark(ROOT / "configs" / "default.yaml", tmp_path)
    assert set(paths) == {
        "task_contract",
        "benchmark_matrix",
        "leaderboard",
        "claim_gate",
    }
    for p in paths.values():
        assert p.is_file()
    gate = json.loads(paths["claim_gate"].read_text(encoding="utf-8"))
    assert gate["status"] in {"sota_cleared", "sota_not_cleared"}
    assert (
        gate["status"] == "sota_not_cleared"
    )  # honest current state on the fixture lane
    contract = json.loads(paths["task_contract"].read_text(encoding="utf-8"))
    assert "prequential next-cell" in contract["primary_metric"]
    with paths["leaderboard"].open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert any(r["model"] == "leakage_oracle_probe" for r in rows)
    assert all("mean_predictive_loss" in r for r in rows)
