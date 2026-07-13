"""Tests for the principled-inference path: the fair, oracle-stripped ladder.

These lock the HONEST, RedTeam-corrected claims: a matched transition source
(leakage guard), measured soft-vs-hard and recursion terms, return-feature
comparisons, and the fair-comparison-not-SOTA framing. No mocks — real fixture
sequence.
"""

from __future__ import annotations

import json
from pathlib import Path

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fair_inference import (
    FAIR_EXTERNAL_REALIZATIONS_SCHEMA,
    REQUIRED_EXTERNAL_REALIZATION_IDS,
    emission_noise_sweep,
    fair_external_realization_manifest,
    fair_inference_ladder,
    fair_sota_robustness,
    run_and_write_fair_inference,
    write_fair_external_realizations,
)
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.observations import discretize_observations

ROOT = Path(__file__).resolve().parents[1]


def _sequence():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    return discretize_observations(
        parse_waypoint_csv(cfg.fractalrabbit.fixture_csv), cfg.inference
    )


def test_ladder_ordering_and_positive_controls() -> None:
    seq = _sequence()
    L = fair_inference_ladder(seq, smoothing=0.5)
    loss = L["losses"]
    assert (
        loss["oracle"]
        < loss["honest_aif"]
        < loss["hard_handicapped"]
        < loss["soft_handicapped"]
    )
    assert L["aif_beats_soft_at_fixed_features"] is True
    # positive controls
    assert L["handicap_is_real"] is True
    assert L["identity_emission_collapses_handicap"] is True
    assert L["aif_still_loses_to_oracle"] is True  # fair-comparison, not SOTA
    assert L["transition_source"].startswith("true_observed_cells")


def test_soft_marginalization_dominates_and_decomposition_is_honest() -> None:
    seq = _sequence()
    L = fair_inference_ladder(seq, smoothing=0.5)
    d = L["decomposition_nats"]
    assert d["soft_marginalization_gain"] < 0
    assert d["recursive_belief_gain"] > 0
    assert d["recursive_belief_gain"] > abs(d["soft_marginalization_gain"])
    assert L["soft_marginalization_dominates_recursion"] is False


def test_return_boosted_soft_is_a_strong_fair_competitor() -> None:
    """The honest, non-overclaimed framing: a fair non-recursive return-boosted soft
    Markov beats the boost-free recursive filter; the filter leads only once BOTH
    carry the visited-cell feature."""
    seq = _sequence()
    L = fair_inference_ladder(seq, smoothing=0.5)
    assert L["return_boosted_soft_beats_bare_aif"] is False
    assert L["aif_wins_at_matched_features"] is True


def test_recursion_advantage_grows_with_observation_noise() -> None:
    """Honest crossover finding (fair matched-source comparator): recursive belief is
    small or negative when observations are clean and grows as the channel gets
    noisier — it earns its keep specifically under observation noise."""
    seq = _sequence()
    s = emission_noise_sweep(seq, smoothing_levels=(0.05, 0.5, 2.0))
    assert s["recursion_advantage_grows_with_noise"] is True
    assert s["recursion_negative_at_lowest_noise"] is True
    assert s["magnitude_is_seed_sensitive"] is True
    adv = [p["aif_advantage_over_soft"] for p in s["points"]]
    assert adv[-1] > adv[0]  # grows from clean -> noisy


def test_writer_emits_artifact(tmp_path) -> None:
    out = run_and_write_fair_inference(
        ROOT / "configs" / "default.yaml", tmp_path / "fair_inference.json"
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "active-fractal-rabbit-fair-inference-v1"
    assert payload["primary_source"] == "configured_fixture"
    assert payload["ladder"]["aif_still_loses_to_oracle"] is True
    assert len(payload["noise_sweep"]["points"]) == 5
    # the SOTA robustness block is embedded in the artifact (ISC-210)
    assert payload["sota_robustness"]["aif_class_best_fair_all_seeds"] is False
    assert payload["external_lane"]["status"] == "unavailable"
    assert (
        payload["external_realizations"]["status"]
        == "blocked_external_realizations_incomplete"
    )


def test_writer_records_external_lane_without_changing_primary(tmp_path) -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    out = run_and_write_fair_inference(
        ROOT / "configs" / "default.yaml",
        tmp_path / "fair_inference.json",
        cfg.fractalrabbit.fixture_csv,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["primary_source"] == "configured_fixture"
    assert payload["external_lane"]["status"] == "completed"
    assert payload["external_lane"]["aif_still_loses_to_oracle"] is True
    assert "not oracle-beating" in payload["external_lane"]["boundary"]
    assert payload["external_realizations"]["completed_count"] == 1
    assert (
        payload["external_realizations"]["can_promote_beyond_fixture_language"] is False
    )


def test_external_realization_manifest_blocks_until_all_required_rows_current(
    tmp_path,
) -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    manifest = fair_external_realization_manifest(
        ROOT / "configs" / "default.yaml",
        current_external_observations=cfg.fractalrabbit.fixture_csv,
    )
    assert manifest["schema"] == FAIR_EXTERNAL_REALIZATIONS_SCHEMA
    assert manifest["completed_count"] == 1
    assert manifest["all_required_completed"] is False
    assert manifest["can_promote_beyond_fixture_language"] is False
    assert any(row["status"] == "unavailable" for row in manifest["rows"])

    out = write_fair_external_realizations(
        ROOT / "configs" / "default.yaml",
        tmp_path / "fair_inference_external_realizations.json",
        cfg.fractalrabbit.fixture_csv,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked_external_realizations_incomplete"


def test_external_realization_manifest_promotes_only_when_all_required_rows_current(
    tmp_path,
) -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    current_rows = {
        realization_id: cfg.fractalrabbit.fixture_csv
        for realization_id in REQUIRED_EXTERNAL_REALIZATION_IDS
    }
    manifest = fair_external_realization_manifest(
        ROOT / "configs" / "default.yaml",
        current_external_observations=current_rows,
    )
    assert manifest["completed_count"] == len(REQUIRED_EXTERNAL_REALIZATION_IDS)
    assert manifest["unavailable_count"] == 0
    assert manifest["all_required_completed"] is True
    assert manifest["all_completed_agree_on_boundary_controls"] is True
    assert manifest["can_promote_beyond_fixture_language"] is True
    assert manifest["status"] == "promotion_ready"

    current_rows["pinned_external_repeat_2"] = tmp_path / "missing.csv"
    blocked = fair_external_realization_manifest(
        ROOT / "configs" / "default.yaml",
        current_external_observations=current_rows,
    )
    assert blocked["completed_count"] == len(REQUIRED_EXTERNAL_REALIZATION_IDS) - 1
    assert blocked["unavailable_count"] == 1
    assert blocked["can_promote_beyond_fixture_language"] is False


def test_writer_embeds_multi_realization_manifest_without_changing_primary(
    tmp_path,
) -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    current_rows = {
        realization_id: cfg.fractalrabbit.fixture_csv
        for realization_id in REQUIRED_EXTERNAL_REALIZATION_IDS
    }
    out = run_and_write_fair_inference(
        ROOT / "configs" / "default.yaml",
        tmp_path / "fair_inference.json",
        current_rows,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["primary_source"] == "configured_fixture"
    assert (
        payload["external_lane"]["realization_id"]
        == REQUIRED_EXTERNAL_REALIZATION_IDS[0]
    )
    assert payload["external_realizations"]["status"] == "promotion_ready"
    assert (
        payload["external_realizations"]["can_promote_beyond_fixture_language"] is True
    )


def test_fair_sota_robustness_verdict_and_positive_controls() -> None:
    """Fair-frontier SOTA verdict (computed at gamma=1, no tuning knob) + the precision
    self-improvement. Establishes AIF as the best deployable-information predictor across
    every seed, with the data-power caveat, and the honest gamma=1.5 gain."""
    seq = _sequence()
    r = fair_sota_robustness(seq, smoothing=0.5, holdout_fraction=0.5)
    assert r["aif_class_best_fair_all_seeds"] is False
    assert (
        r["handicap_real_all_seeds"] is True
    )  # oracle < AIF every seed: comparison is non-degenerate
    assert r["aif_advantage_mean_nats"] < 0.0
    assert r["aif_advantage_min_nats"] < 0.0
    # finding 2 — precision gamma=1.5 lowers loss AND improves calibration, paired
    assert r["precision_improves_calibration"] is True
    assert r["mean_loss_precision"] < r["mean_loss_gamma1"]
    assert r["mean_ece_precision"] <= r["mean_ece_gamma1"]
    # honesty boundary present in the verdict (ISC-214)
    assert "oracle" in r["verdict"].lower() and "fair-frontier" in r["verdict"].lower()


def test_fair_sota_verdict_is_derived_not_hardcoded() -> None:
    """RedTeam CRITICAL fix: the verdict must TRACK the measured outcome, never a
    hardcoded victory string. At a powered holdout it is unanimous; at an underpowered
    holdout the same machinery reports NOT-unanimous. In NO regime does it claim
    CI-separation as evidence (the per-step bootstrap overstates precision)."""
    seq = _sequence()
    powered = fair_sota_robustness(seq, smoothing=0.5, holdout_fraction=0.5)
    assert powered["aif_class_best_fair_all_seeds"] is False
    assert powered["aif_class_best_fair_rate"] in powered["verdict"]
    assert "NOT unanimous" in powered["verdict"]
    assert "does not establish a best-fair-predictor result" in powered["verdict"]
    assert "CI-separation is NOT claimed" in powered["verdict"]
    assert powered["step_ci_overstates_precision"] is True

    under = fair_sota_robustness(seq, smoothing=0.5, holdout_fraction=0.25)
    # underpowered holdout is genuinely NOT unanimous on this fixture -> verdict must say so
    assert under["aif_class_best_fair_all_seeds"] is False
    assert under["aif_class_best_fair_rate"] in under["verdict"]
    assert "NOT unanimous" in under["verdict"]
    assert "directionally unanimous across every" not in under["verdict"]


def test_fair_sota_robustness_non_vacuity_firing_controls() -> None:
    """The metrics are computed from the probe, not green-by-construction: two controls
    that FIRE when the lever is neutralized or reversed."""
    seq = _sequence()
    # FIRING control A: gamma=1.0 is a no-op => zero loss wins and identical means
    flat = fair_sota_robustness(seq, smoothing=0.5, precision=1.0)
    assert flat["precision_gain_loss_win_rate"].startswith("0/")
    assert flat["mean_loss_precision"] == flat["mean_loss_gamma1"]
    # FIRING control B: under-sharpening (gamma<1) flattens an already under-confident
    # belief => calibration gets WORSE, so the improvement flag must turn False
    reversed_lever = fair_sota_robustness(seq, smoothing=0.5, precision=0.5)
    assert reversed_lever["precision_improves_calibration"] is False
    assert reversed_lever["mean_ece_precision"] > reversed_lever["mean_ece_gamma1"]
