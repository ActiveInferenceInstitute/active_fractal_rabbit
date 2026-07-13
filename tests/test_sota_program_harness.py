"""Tests for the canonical SOTA program harness.

No mocks: real deterministic synthetic lanes, real scoring, real traveller-clustered
bootstrap. Covers lane construction, the full candidate roster, the strongest-fair-baseline
selection (oracle/AIF excluded), the clustered-CI gate decision, the honest "still blocked"
verdict whenever a lane fails, the leakage sanity floor, and the emitted artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from active_fractal_rabbit.sota_program import (
    PartialObservabilityDataset,
)
from active_fractal_rabbit.sota_program_harness import (
    AIF_CANDIDATE_NAME,
    AIF_FAMILY_NAMES,
    DEFAULT_BOOT_SEED,
    DEFAULT_EPS,
    DEFAULT_N_BOOT,
    DEFAULT_N_TRAVELLERS,
    DEFAULT_SPATIAL_BINS,
    DIAGNOSTIC_STRESSOR_LANES,
    LEAKAGE_FLOOR_NATS,
    CANONICAL_LANES,
    SOTA_PROGRAM_ROBUSTNESS_DIAGNOSTICS_SCHEMA,
    SOTA_PROGRAM_LEADERBOARD_HEADER,
    GateResult,
    LaneResult,
    build_inference_config,
    build_diagnostic_lanes,
    build_lanes,
    evaluate_gate,
    gate_payload,
    leaderboard_rows,
    robustness_diagnostic_payload,
    run_and_write,
    run_diagnostic_robustness_suite,
    run_program,
    score_lane,
    write_gate_json,
    write_leaderboard_csv,
    _is_canonical_run,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "default.yaml"

# Small-but-real lanes keep the bootstrap and MLP training fast while preserving many
# travellers (clusters) so the clustered CI is meaningful.
N_TRAVELLERS = 24
N_BOOT = 400


def _inference(bins: int = 6):
    return build_inference_config(CONFIG, spatial_bins=bins)


def _lanes():
    return build_lanes(_inference(), n_travellers=N_TRAVELLERS, eps=0.3)


# --------------------------------------------------------------------------- #
# Lane construction
# --------------------------------------------------------------------------- #
def test_build_inference_config_sets_grid() -> None:
    inf = _inference(bins=6)
    assert inf.spatial_bins == 6  # n_states == 36


def test_build_lanes_four_distinct_synthetic_lanes_same_contract() -> None:
    lanes = _lanes()
    assert tuple(lanes) == CANONICAL_LANES
    fr = lanes["fractalrabbit"]
    gv = lanes["gravity"]
    for ds in lanes.values():
        assert isinstance(ds, PartialObservabilityDataset)
        assert ds.n_states == 36
        assert ds.observed.shape == ds.latent.shape == ds.travellers.shape
        # locked split keyed on traveller identity: disjoint train/holdout
        holdout = set(int(t) for t in ds.travellers[ds.is_holdout])
        train = set(int(t) for t in ds.travellers[~ds.is_holdout])
        assert holdout and train and holdout.isdisjoint(train)
    # the two lanes are genuinely different processes: their latent streams differ
    assert not np.array_equal(fr.latent, gv.latent)


def test_build_diagnostic_lanes_are_scored_stressor_variants() -> None:
    lanes = build_diagnostic_lanes(_inference(), n_travellers=12, eps=0.3)
    assert tuple(lanes) == DIAGNOSTIC_STRESSOR_LANES
    for ds in lanes.values():
        assert isinstance(ds, PartialObservabilityDataset)
        holdout = set(int(t) for t in ds.travellers[ds.is_holdout])
        train = set(int(t) for t in ds.travellers[~ds.is_holdout])
        assert holdout and train and holdout.isdisjoint(train)


# --------------------------------------------------------------------------- #
# Scoring a lane: full roster + strongest-fair selection
# --------------------------------------------------------------------------- #
def test_score_lane_includes_full_roster_and_excludes_aif_family_from_fair() -> None:
    lanes = _lanes()
    lr = score_lane("fractalrabbit", lanes["fractalrabbit"], n_boot=N_BOOT)
    names = set(lr.scored)
    # every expected candidate is present
    expected = {
        "hard_filter",
        "observed_markov",
        "soft_filter",
        "persistence",
        "online_base_rate",
        "first_order_markov",
        "higher_order_backoff_markov",
        "return_boosted_markov",
        "recency_attention",
        "multi_horizon_recency_attention",
        "switching_noise_hmm",
        "particle_filter",
        "mlp_next_cell",
        AIF_CANDIDATE_NAME,
    }
    assert expected <= names
    # the strongest fair baseline is never the AIF candidate or its soft-filter reference
    assert lr.strongest_fair_name not in AIF_FAMILY_NAMES
    # the strongest fair baseline really is the min-loss non-AIF candidate
    fair_losses = {
        n: s.pooled_mean_loss for n, s in lr.scored.items() if n not in AIF_FAMILY_NAMES
    }
    assert lr.strongest_fair_loss == pytest.approx(min(fair_losses.values()))
    assert lr.strongest_fair_name == min(fair_losses, key=fair_losses.get)


def test_score_lane_all_losses_finite_and_positive() -> None:
    lanes = _lanes()
    for name, ds in lanes.items():
        lr = score_lane(name, ds, n_boot=N_BOOT)
        for cand_name, scored in lr.scored.items():
            assert np.isfinite(scored.pooled_mean_loss)
            assert scored.pooled_mean_loss > 0.0
        assert scored.n_scored_steps > 0
        assert scored.loss_ci_low is not None
        assert scored.loss_ci_high is not None
        assert scored.loss_ci_low <= scored.pooled_mean_loss <= scored.loss_ci_high
        assert scored.runtime_seconds is not None
        assert scored.runtime_seconds >= 0.0
        assert 0.0 <= scored.brier_score <= 1.0
        assert 0.0 <= scored.ece <= 1.0


def test_clustered_ci_matches_direct_paired_call() -> None:
    # The lane's reported CI must be exactly the paired clustered CI on the AIF vs
    # strongest-fair difference (no hidden per-step shortcut).
    from active_fractal_rabbit.sota_program import paired_loss_difference_ci

    lanes = _lanes()
    ds = lanes["fractalrabbit"]
    lr = score_lane("fractalrabbit", ds, n_boot=N_BOOT)
    strongest = lr.scored[lr.strongest_fair_name]
    aif = lr.scored[AIF_CANDIDATE_NAME]
    ci = paired_loss_difference_ci(strongest, aif, n_boot=N_BOOT, seed=7)
    assert lr.clustered_ci_low == pytest.approx(ci.clustered_ci_low)
    assert lr.clustered_ci_high == pytest.approx(ci.clustered_ci_high)
    # aif_beats requires BOTH a lower mean loss AND the CI separated from zero
    expect_beats = (
        ci.b_wins_clustered and aif.pooled_mean_loss < strongest.pooled_mean_loss
    )
    assert lr.aif_beats_strongest_fair == expect_beats


def test_leakage_floor_holds_no_candidate_peeks() -> None:
    # The AIF candidate's loss must stay well above the oracle floor (a peeking candidate
    # would sit near zero). This is the structural leakage sanity check.
    lanes = _lanes()
    for name, ds in lanes.items():
        lr = score_lane(name, ds, n_boot=N_BOOT)
        assert lr.aif_loss > LEAKAGE_FLOOR_NATS
        assert lr.leakage_floor_ok


# --------------------------------------------------------------------------- #
# Gate decision logic
# --------------------------------------------------------------------------- #
def _fake_lane(
    name: str,
    *,
    aif_loss: float,
    fair_loss: float,
    ci_low: float,
    ci_high: float,
    aif_beats: bool,
    floor_ok: bool = True,
) -> LaneResult:
    """A minimal LaneResult for exercising the pure gate logic (no scoring needed).

    The ``scored`` map is left empty because ``evaluate_gate`` reads only the summary fields;
    this keeps the gate-logic tests fast and isolated from the heavy scoring path. (The
    full-scoring path is exercised separately by the integration tests below.)
    """

    return LaneResult(
        lane=name,
        n_states=36,
        n_travellers=24,
        eps=0.3,
        aif_gamma=1.5,
        aif_boost=0.0,
        scored={},
        strongest_fair_name="switching_noise_hmm",
        strongest_fair_loss=fair_loss,
        aif_name=AIF_CANDIDATE_NAME,
        aif_loss=aif_loss,
        mean_difference_nats=fair_loss - aif_loss,
        clustered_ci_low=ci_low,
        clustered_ci_high=ci_high,
        aif_beats_strongest_fair=aif_beats,
        leakage_floor_ok=floor_ok,
        n_holdout_travellers=10,
    )


def test_gate_clears_only_when_aif_wins_both_lanes() -> None:
    lanes = {
        "fractalrabbit": _fake_lane(
            "fractalrabbit",
            aif_loss=1.24,
            fair_loss=1.25,
            ci_low=0.005,
            ci_high=0.02,
            aif_beats=True,
        ),
        "gravity": _fake_lane(
            "gravity",
            aif_loss=2.50,
            fair_loss=2.60,
            ci_low=0.05,
            ci_high=0.15,
            aif_beats=True,
        ),
    }
    gate = evaluate_gate(lanes)
    assert gate.gate_clears
    assert gate.status == "sota_program_cleared"
    assert gate.reasons == []


def test_gate_blocked_when_one_lane_loses_on_mean() -> None:
    lanes = {
        "fractalrabbit": _fake_lane(
            "fractalrabbit",
            aif_loss=1.24,
            fair_loss=1.25,
            ci_low=0.005,
            ci_high=0.02,
            aif_beats=True,
        ),
        # AIF loses on the gravity lane (higher loss than the strongest fair baseline)
        "gravity": _fake_lane(
            "gravity",
            aif_loss=3.50,
            fair_loss=2.76,
            ci_low=-0.81,
            ci_high=-0.66,
            aif_beats=False,
        ),
    }
    gate = evaluate_gate(lanes)
    assert not gate.gate_clears
    assert gate.status == "sota_program_not_cleared"
    assert any("does NOT beat the strongest fair baseline" in r for r in gate.reasons)
    # residual gap records the positive (AIF-worse) shortfall on the failing lane
    assert gate.residual_gap["gravity"]["aif_minus_strongest_fair_nats"] > 0.0


def test_gate_blocked_when_ci_overlaps_zero_despite_lower_mean() -> None:
    # AIF has the lower mean on both lanes, but on one lane the clustered CI straddles zero
    # (not separated), so the gate must stay blocked.
    lanes = {
        "fractalrabbit": _fake_lane(
            "fractalrabbit",
            aif_loss=1.24,
            fair_loss=1.25,
            ci_low=0.005,
            ci_high=0.02,
            aif_beats=True,
        ),
        "gravity": _fake_lane(
            "gravity",
            aif_loss=2.55,
            fair_loss=2.60,
            ci_low=-0.02,
            ci_high=0.12,
            aif_beats=False,
        ),
    }
    gate = evaluate_gate(lanes)
    assert not gate.gate_clears
    assert any("NOT clustered-CI separated from zero" in r for r in gate.reasons)


def test_gate_blocked_on_leakage_floor_violation() -> None:
    lanes = {
        "fractalrabbit": _fake_lane(
            "fractalrabbit",
            aif_loss=0.10,
            fair_loss=1.25,
            ci_low=0.5,
            ci_high=0.9,
            aif_beats=True,
            floor_ok=False,
        ),
        "gravity": _fake_lane(
            "gravity",
            aif_loss=2.50,
            fair_loss=2.60,
            ci_low=0.05,
            ci_high=0.15,
            aif_beats=True,
        ),
    }
    gate = evaluate_gate(lanes)
    assert not gate.gate_clears
    assert any("leakage sanity failed" in r for r in gate.reasons)


def test_evaluate_gate_rejects_empty_lanes() -> None:
    with pytest.raises(ValueError):
        evaluate_gate({})


# --------------------------------------------------------------------------- #
# End-to-end program + artifacts (the REAL measured outcome)
# --------------------------------------------------------------------------- #
def test_run_program_real_measured_outcome_and_honesty() -> None:
    gate = run_program(_inference(), n_travellers=N_TRAVELLERS, eps=0.3, n_boot=N_BOOT)
    assert isinstance(gate, GateResult)
    assert tuple(gate.lanes) == CANONICAL_LANES
    assert gate.run_parameters["canonical"] is False
    assert gate.run_parameters["n_travellers"] == N_TRAVELLERS
    # Honesty invariant: gate_clears is True iff every lane's AIF candidate beats its
    # strongest fair baseline with a separated CI. This must be consistent with per-lane flags.
    expected_clears = all(
        lr.aif_beats_strongest_fair and lr.leakage_floor_ok
        for lr in gate.lanes.values()
    )
    assert gate.gate_clears == expected_clears
    # if blocked, there must be at least one honest reason; if cleared, none.
    assert (len(gate.reasons) == 0) == gate.gate_clears
    # the oracle is never selected as the strongest fair baseline
    for lr in gate.lanes.values():
        assert lr.strongest_fair_name not in AIF_FAMILY_NAMES
        assert "oracle" not in lr.strongest_fair_name


def test_leaderboard_rows_one_per_candidate_per_lane() -> None:
    gate = run_program(_inference(), n_travellers=N_TRAVELLERS, eps=0.3, n_boot=N_BOOT)
    rows = leaderboard_rows(gate)
    required = set(SOTA_PROGRAM_LEADERBOARD_HEADER)
    # one row per (candidate, lane)
    per_lane = {}
    for r in rows:
        assert required <= set(r)
        for field in (
            "family",
            "loss_ci_low",
            "loss_ci_high",
            "brier_score",
            "ece",
            "top1_accuracy",
            "runtime_seconds",
            "lane_coverage",
            "dependency_footprint",
            "missingness_robustness_status",
            "failure_modes",
        ):
            assert r[field] != ""
        assert r["locked_split_scored"] is True
        assert r["leakage"] is False
        assert str(r["missingness_robustness_status"]).startswith("scored_")
        per_lane.setdefault(r["lane"], set()).add(r["candidate"])
    for lane, cands in per_lane.items():
        assert AIF_CANDIDATE_NAME in cands
        # exactly one AIF-candidate row per lane carries the CI columns
        aif_rows = [
            r
            for r in rows
            if r["lane"] == lane and r["candidate"] == AIF_CANDIDATE_NAME
        ]
        assert len(aif_rows) == 1
        assert aif_rows[0]["clustered_ci_low"] != ""
        assert aif_rows[0]["role"] == "aif_candidate"
    # the strongest-fair row is labelled as such
    sf_roles = {r["role"] for r in rows}
    assert "strongest_fair_baseline" in sf_roles


def test_run_and_write_emits_csv_and_gate_json(tmp_path: Path) -> None:
    gate, paths = run_and_write(
        config_path=CONFIG,
        out_dir=tmp_path,
        n_travellers=N_TRAVELLERS,
        eps=0.3,
        n_boot=N_BOOT,
    )
    lb = paths["leaderboard"]
    gj = paths["gate"]
    rb = paths["robustness_grid"]
    rd = paths["robustness_diagnostics"]
    assert lb.exists() and gj.exists() and rb.exists() and rd.exists()
    # CSV has a header and at least one row per lane for every candidate
    csv_text = lb.read_text(encoding="utf-8").strip().splitlines()
    assert csv_text[0].startswith("candidate,lane,role,mean_loss")
    assert csv_text[0].split(",") == SOTA_PROGRAM_LEADERBOARD_HEADER
    assert len(csv_text) - 1 == len(leaderboard_rows(gate))
    # gate JSON round-trips and carries the honest status + per-lane residual gap
    payload = json.loads(gj.read_text(encoding="utf-8"))
    assert payload["gate_clears"] == gate.gate_clears
    assert payload["status"] == gate.status
    assert payload["run_parameters"]["canonical"] is False
    assert payload["run_parameters"]["n_boot"] == N_BOOT
    assert tuple(payload["lanes"]) == CANONICAL_LANES
    for lane in payload["lanes"].values():
        assert lane["strongest_fair_baseline"] not in AIF_FAMILY_NAMES
        assert lane["candidate_rows"]
        for row in lane["candidate_rows"]:
            assert row["loss_ci_low"] != ""
            assert row["loss_ci_high"] != ""
            assert row["dependency_footprint"]
            assert row["missingness_robustness_status"].startswith("scored_")
    diag = json.loads(rd.read_text(encoding="utf-8"))
    assert diag["schema"] == SOTA_PROGRAM_ROBUSTNESS_DIAGNOSTICS_SCHEMA
    assert diag["all_diagnostic_lanes_scored"] is True
    assert {row["lane"] for row in diag["lane_statuses"]} == set(
        DIAGNOSTIC_STRESSOR_LANES
    )
    assert set(payload["residual_gap"]) == set(CANONICAL_LANES)
    assert (
        payload["candidate_table_contract"]["columns"]
        == SOTA_PROGRAM_LEADERBOARD_HEADER
    )
    assert payload["readiness"]["missingness_robustness"]["status"] == "scored"
    assert payload["readiness"]["diagnostic_stressors"]["lanes"] == list(
        DIAGNOSTIC_STRESSOR_LANES
    )
    assert payload["robustness_summary"]["all_canonical_lanes_scored"] is True
    assert payload["robustness_summary"]["reporting_gap_scored"] is True
    assert payload["robustness_summary"]["regime_stressor_scored"] is True
    assert {
        row["lane"] for row in payload["readiness"]["regime_stressor_lanes"]
    } == set(CANONICAL_LANES)


def test_reduced_run_cannot_overwrite_canonical_sota_program_artifacts() -> None:
    with pytest.raises(
        ValueError, match="noncanonical SOTA-program run cannot write canonical"
    ):
        run_and_write(
            config_path=CONFIG,
            out_dir=ROOT / "output" / "runs",
            n_travellers=N_TRAVELLERS,
            eps=0.3,
            n_boot=N_BOOT,
        )


def test_default_parameters_are_the_canonical_manuscript_run() -> None:
    assert _is_canonical_run(
        n_travellers=DEFAULT_N_TRAVELLERS,
        eps=DEFAULT_EPS,
        spatial_bins=DEFAULT_SPATIAL_BINS,
        n_boot=DEFAULT_N_BOOT,
        boot_seed=DEFAULT_BOOT_SEED,
        lanes=list(CANONICAL_LANES),
    )
    assert not _is_canonical_run(
        n_travellers=12,
        eps=DEFAULT_EPS,
        spatial_bins=DEFAULT_SPATIAL_BINS,
        n_boot=DEFAULT_N_BOOT,
        boot_seed=DEFAULT_BOOT_SEED,
        lanes=list(CANONICAL_LANES),
    )


def test_gate_payload_blocked_status_carries_reasons_when_not_cleared() -> None:
    # Construct a deterministically-blocked gate from fake lanes and confirm the payload is honest.
    lanes = {
        "fractalrabbit": _fake_lane(
            "fractalrabbit",
            aif_loss=1.24,
            fair_loss=1.25,
            ci_low=0.005,
            ci_high=0.02,
            aif_beats=True,
        ),
        "gravity": _fake_lane(
            "gravity",
            aif_loss=3.50,
            fair_loss=2.76,
            ci_low=-0.81,
            ci_high=-0.66,
            aif_beats=False,
        ),
    }
    gate = evaluate_gate(lanes)
    payload = gate_payload(gate)
    assert payload["gate_clears"] is False
    assert payload["claim_gate_reasons"]  # non-empty honest reasons
    assert "Synthetic-only" in payload["caveat"]


def test_diagnostic_robustness_payload_is_claim_bounded() -> None:
    gate = run_diagnostic_robustness_suite(
        _inference(), n_travellers=12, eps=0.3, n_boot=120
    )
    payload = robustness_diagnostic_payload(gate)
    assert payload["schema"] == SOTA_PROGRAM_ROBUSTNESS_DIAGNOSTICS_SCHEMA
    assert payload["all_diagnostic_lanes_scored"] is True
    assert payload["row_count"] > 0
    assert "do not by themselves clear SOTA" in payload["boundary"]


def test_write_helpers_create_parent_dirs(tmp_path: Path) -> None:
    lanes = {
        "fractalrabbit": _fake_lane(
            "fractalrabbit",
            aif_loss=1.24,
            fair_loss=1.25,
            ci_low=0.005,
            ci_high=0.02,
            aif_beats=True,
        ),
        "gravity": _fake_lane(
            "gravity",
            aif_loss=2.50,
            fair_loss=2.60,
            ci_low=0.05,
            ci_high=0.15,
            aif_beats=True,
        ),
    }
    gate = evaluate_gate(lanes)
    nested = tmp_path / "a" / "b"
    csv_path = write_leaderboard_csv(gate, nested / "lb.csv")
    json_path = write_gate_json(gate, nested / "gate.json")
    assert csv_path.exists() and json_path.exists()
