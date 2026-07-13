"""Tests for the multiplicity ledger — real computation, no mocks.

Covers the Holm and Benjamini–Hochberg cores against textbook values, the deterministic
family enumerator (and its FIRE behaviour on a stale artifact), and positive/negative
significance controls proving a marginal lone win does not survive correction while a strong
win does.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from active_fractal_rabbit.multiplicity_ledger import (
    REGIME_P_FIELD,
    MultiplicityLedgerError,
    benjamini_hochberg_adjusted_pvalues,
    build_multiplicity_ledger,
    enumerate_comparison_family,
    holm_adjusted_pvalues,
    run_and_write_multiplicity_ledger,
)


def test_regime_p_aggregation_is_pinned_to_conservative_max(tmp_path):
    # Forge M1: the per-regime p must be the conservative MAX over realizations. A silent swap to
    # the MIN would lower the lone clearer's p and could manufacture a Holm survivor. Pin it.
    assert REGIME_P_FIELD == "one_sided_p_max"
    runs = tmp_path / "output" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    # One regime whose worst realization is non-significant (0.30) but best is tiny (0.001).
    # Using the pinned MAX, it must NOT clear; the family must report zero uncorrected clearers.
    regime_summaries = {
        "r_split": {
            "regime_id": "r_split",
            "gate_clears": False,
            "mean_advantage_nats": 0.02,
            "min_clustered_ci_low": -0.01,
            "one_sided_p_max": 0.30,
            "one_sided_p_min": 0.001,
        }
    }
    (runs / "aif_regime_search.json").write_text(
        json.dumps({"regime_summaries": regime_summaries}), encoding="utf-8"
    )
    (runs / "sota_program_gate.json").write_text(
        json.dumps({"lanes": {}}), encoding="utf-8"
    )
    (runs / "partial_observability_analysis.json").write_text(
        json.dumps(
            {
                "cross_realization_n_draws": 5,
                "cross_realization_sign_stable_under_noise": True,
                "cross_realization_min_fraction_positive_under_noise": 1.0,
                "max_noise_advantage_nats": 0.77,
            }
        ),
        encoding="utf-8",
    )
    payload = build_multiplicity_ledger(tmp_path)
    row = next(
        r for r in payload["comparisons"] if r["comparison_id"] == "aif_regime::r_split"
    )
    assert row["uncorrected_p"] == 0.30  # the MAX, not the MIN
    assert row["uncorrected_significant"] is False
    assert payload["holm"]["n_survivors"] == 0


def test_holm_matches_textbook_values():
    # p = [.01,.02,.03,.04,.05], m=5 -> Holm adj = [.05,.08,.09,.09,.09]
    adj = holm_adjusted_pvalues([0.01, 0.02, 0.03, 0.04, 0.05])
    assert [round(v, 6) for v in adj] == [0.05, 0.08, 0.09, 0.09, 0.09]


def test_holm_preserves_input_order_and_monotonicity():
    adj = holm_adjusted_pvalues([0.04, 0.01, 0.03])
    assert [round(v, 6) for v in adj] == [0.06, 0.03, 0.06]
    # sorted adjusted values are monotone non-decreasing
    pvals = [0.2, 0.001, 0.04, 0.5, 0.03]
    adj = holm_adjusted_pvalues(pvals)
    sorted_pairs = sorted(zip(pvals, adj), key=lambda t: t[0])
    sorted_adj = [a for _, a in sorted_pairs]
    assert all(
        sorted_adj[i] <= sorted_adj[i + 1] + 1e-12 for i in range(len(sorted_adj) - 1)
    )


def test_bh_matches_textbook_values():
    # p = [.01,.02,.03,.04,.05], m=5 -> BH adj all = .05
    adj = benjamini_hochberg_adjusted_pvalues([0.01, 0.02, 0.03, 0.04, 0.05])
    assert all(abs(v - 0.05) < 1e-9 for v in adj)


def test_holm_and_bh_handle_tied_pvalues() -> None:
    """Ties are the classic step-down/step-up implementation trap: without the
    running max/min monotonicity enforcement, tied hypotheses can receive
    different adjusted values. Hand-derived vectors pin the behavior (the
    production family itself carries tied p=1.0 rows, exercised untested
    before this lock)."""
    assert holm_adjusted_pvalues([0.01, 0.01, 0.04, 0.04, 1.0]) == [
        0.05,
        0.05,
        0.12,
        0.12,
        1.0,
    ]
    assert benjamini_hochberg_adjusted_pvalues([0.01, 0.01, 0.04, 0.04, 1.0]) == [
        0.025,
        0.025,
        0.05,
        0.05,
        1.0,
    ]
    # all-ties: Holm collapses to the Bonferroni of the full family at rank 1
    # (running max then propagates it), BH to the raw p at the last rank.
    assert holm_adjusted_pvalues([0.02] * 5) == [0.1] * 5
    assert benjamini_hochberg_adjusted_pvalues([0.02] * 5) == [0.02] * 5
    # tied inputs must always receive identical adjusted values
    for fn in (holm_adjusted_pvalues, benjamini_hochberg_adjusted_pvalues):
        adjusted = fn([0.03, 0.007, 0.03, 0.2, 0.007])
        assert adjusted[0] == adjusted[2]
        assert adjusted[1] == adjusted[4]


def test_bh_is_never_more_conservative_than_holm():
    pvals = [0.001, 0.02, 0.2, 0.04, 0.5, 0.009]
    holm = holm_adjusted_pvalues(pvals)
    bh = benjamini_hochberg_adjusted_pvalues(pvals)
    assert all(b <= h + 1e-12 for b, h in zip(bh, holm))


def test_corrections_empty_list():
    assert holm_adjusted_pvalues([]) == []
    assert benjamini_hochberg_adjusted_pvalues([]) == []


def _write_artifacts(
    runs: Path,
    *,
    regime_pvalues: dict[str, float],
    lane_pvalues: dict[str, float],
    n_draws: int = 5,
    sign_stable: bool = True,
    include_regime_p: bool = True,
) -> None:
    runs.mkdir(parents=True, exist_ok=True)
    regime_summaries = {}
    for regime_id, p in regime_pvalues.items():
        summary = {
            "regime_id": regime_id,
            "gate_clears": p < 0.05,
            "mean_advantage_nats": 0.05,
            "min_clustered_ci_low": 0.001 if p < 0.05 else -0.01,
        }
        if include_regime_p:
            summary["one_sided_p_max"] = p
        regime_summaries[regime_id] = summary
    (runs / "aif_regime_search.json").write_text(
        json.dumps({"regime_summaries": regime_summaries}), encoding="utf-8"
    )
    lanes = {
        lane_id: {
            "one_sided_p": p,
            "mean_difference_nats_fair_minus_aif": 0.03,
            "clustered_ci_low": 0.001 if p < 0.05 else -0.02,
            "aif_beats_strongest_fair": p < 0.05,
            "n_holdout_travellers": 6,
        }
        for lane_id, p in lane_pvalues.items()
    }
    (runs / "sota_program_gate.json").write_text(
        json.dumps({"lanes": lanes}), encoding="utf-8"
    )
    (runs / "partial_observability_analysis.json").write_text(
        json.dumps(
            {
                "cross_realization_n_draws": n_draws,
                "cross_realization_sign_stable_under_noise": sign_stable,
                "cross_realization_min_fraction_positive_under_noise": 1.0,
                "max_noise_advantage_nats": 0.7745,
            }
        ),
        encoding="utf-8",
    )


def test_enumerator_fires_when_regime_lacks_pvalue(tmp_path):
    runs = tmp_path / "output" / "runs"
    _write_artifacts(
        runs,
        regime_pvalues={"r_a": 0.01, "r_b": 0.2},
        lane_pvalues={"lane_x": 0.3},
        include_regime_p=False,  # regimes carry NO bootstrap p
    )
    family = enumerate_comparison_family(runs)
    ids = {c.comparison_id for c in family}
    # the regimes are dropped (no p); only the lane survives
    assert ids == {"sota_lane::lane_x"}


def test_enumerator_raises_when_family_empty(tmp_path):
    runs = tmp_path / "output" / "runs"
    _write_artifacts(runs, regime_pvalues={}, lane_pvalues={}, include_regime_p=False)
    with pytest.raises(MultiplicityLedgerError):
        enumerate_comparison_family(runs)


def test_missing_artifact_raises(tmp_path):
    runs = tmp_path / "output" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    with pytest.raises(MultiplicityLedgerError):
        enumerate_comparison_family(runs)


def test_negative_control_marginal_lone_win_does_not_survive_holm(tmp_path):
    # One regime marginally significant (p=0.04) among a family of 9; under Holm it should be
    # multiplied toward ~0.36 and NOT survive. A spurious extra comparison must not flip a real
    # win's verdict in either direction.
    runs = tmp_path / "output" / "runs"
    regimes = {f"r{i}": 0.6 for i in range(5)}
    regimes["r0"] = 0.04  # the lone marginal "win"
    lanes = {f"lane{i}": 0.7 for i in range(4)}
    _write_artifacts(runs, regime_pvalues=regimes, lane_pvalues=lanes)
    payload = build_multiplicity_ledger(tmp_path)
    assert payload["family_size"] == 9
    assert payload["n_uncorrected_significant"] == 1
    assert (
        payload["holm"]["n_survivors"] == 0
    )  # marginal lone win dissolves under correction
    row = next(
        r for r in payload["comparisons"] if r["comparison_id"] == "aif_regime::r0"
    )
    assert row["uncorrected_significant"] is True
    assert row["holm_survives"] is False
    assert math.isclose(row["holm_adjusted_p"], 0.36, abs_tol=1e-6)


def test_positive_control_strong_win_survives_holm(tmp_path):
    # A genuinely strong win (p≈1e-5) survives Holm across the same family size.
    runs = tmp_path / "output" / "runs"
    regimes = {f"r{i}": 0.6 for i in range(5)}
    regimes["r0"] = 1e-5
    lanes = {f"lane{i}": 0.7 for i in range(4)}
    _write_artifacts(runs, regime_pvalues=regimes, lane_pvalues=lanes)
    payload = build_multiplicity_ledger(tmp_path)
    assert payload["holm"]["n_survivors"] == 1
    assert payload["holm"]["survivor_ids"] == ["aif_regime::r0"]


def test_uncorrected_significant_follows_pipeline_gate_not_raw_pvalue(tmp_path):
    # Bug found live 2026-07-02 (session 27): aif_regime::symmetric_noise_045 had
    # gate_clears=False (clustered_ci_low < 0, the pipeline's own CI-based verdict never
    # cleared) yet scored one_sided_p_max=0.033983 < 0.05, and the ledger's
    # uncorrected_significant was independently recomputed from the p-value alone,
    # manufacturing a clearance the pipeline itself never produced. Reproduce that exact
    # shape directly: a regime whose p clears alpha=0.05 but whose CI does not.
    runs = tmp_path / "output" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    regime_summaries = {
        "r_ci_never_cleared": {
            "regime_id": "r_ci_never_cleared",
            "gate_clears": False,  # the pipeline's own CI-based verdict: never cleared
            "mean_advantage_nats": 0.02,
            "min_clustered_ci_low": -0.001306,  # CI straddles zero
            "one_sided_p_max": 0.033983,  # yet the raw p clears a naive alpha=0.05 check
        }
    }
    (runs / "aif_regime_search.json").write_text(
        json.dumps({"regime_summaries": regime_summaries}), encoding="utf-8"
    )
    (runs / "sota_program_gate.json").write_text(
        json.dumps({"lanes": {}}), encoding="utf-8"
    )
    (runs / "partial_observability_analysis.json").write_text(
        json.dumps(
            {
                "cross_realization_n_draws": 5,
                "cross_realization_sign_stable_under_noise": True,
                "cross_realization_min_fraction_positive_under_noise": 1.0,
                "max_noise_advantage_nats": 0.77,
            }
        ),
        encoding="utf-8",
    )
    payload = build_multiplicity_ledger(tmp_path)
    row = next(
        r
        for r in payload["comparisons"]
        if r["comparison_id"] == "aif_regime::r_ci_never_cleared"
    )
    assert row["individually_clears"] is False
    assert (
        row["uncorrected_p"] == 0.033983
    )  # p is still reported, just not the gate source
    # The whole point of the fix: uncorrected_significant must track individually_clears,
    # not "uncorrected_p < alpha" — this assertion FAILS pre-fix (would read True).
    assert row["uncorrected_significant"] is False
    assert payload["n_uncorrected_significant"] == 0
    assert row["comparison_id"] not in payload["headline"]["clearer_ids"]


def test_uncorrected_significant_always_equals_individually_clears(tmp_path):
    # General invariant across a mixed family (some clear the CI gate, some don't, at varying
    # p): the two fields must never disagree, for every comparison, every time.
    runs = tmp_path / "output" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    regime_summaries = {
        "clears_low_p": {
            "regime_id": "clears_low_p",
            "gate_clears": True,
            "one_sided_p_max": 0.01,
            "min_clustered_ci_low": 0.02,
        },
        "clears_high_p": {
            "regime_id": "clears_high_p",
            "gate_clears": True,
            "one_sided_p_max": 0.049,
            "min_clustered_ci_low": 0.001,
        },
        "misses_low_p": {
            "regime_id": "misses_low_p",
            "gate_clears": False,
            "one_sided_p_max": 0.033,
            "min_clustered_ci_low": -0.001,
        },
        "misses_high_p": {
            "regime_id": "misses_high_p",
            "gate_clears": False,
            "one_sided_p_max": 0.9,
            "min_clustered_ci_low": -0.5,
        },
    }
    (runs / "aif_regime_search.json").write_text(
        json.dumps({"regime_summaries": regime_summaries}), encoding="utf-8"
    )
    (runs / "sota_program_gate.json").write_text(
        json.dumps({"lanes": {}}), encoding="utf-8"
    )
    (runs / "partial_observability_analysis.json").write_text(
        json.dumps(
            {
                "cross_realization_n_draws": 5,
                "cross_realization_sign_stable_under_noise": True,
                "cross_realization_min_fraction_positive_under_noise": 1.0,
                "max_noise_advantage_nats": 0.77,
            }
        ),
        encoding="utf-8",
    )
    payload = build_multiplicity_ledger(tmp_path)
    assert len(payload["comparisons"]) == 4
    for row in payload["comparisons"]:
        assert row["uncorrected_significant"] == row["individually_clears"], row


def test_holm_bh_survival_requires_individually_clears(tmp_path):
    # Forge cross-vendor HIGH-1 (session 27, 2026-07-02): correction can only ever DEMOTE an
    # already-cleared comparison, never resurrect one that never cleared. A comparison with
    # gate_clears=False but a very small raw p (e.g. from a noisy bootstrap draw) must NOT
    # register holm_survives/bh_survives=True just because its p-value happens to correct well —
    # that would produce the self-contradictory "survives correction but never cleared
    # uncorrected" state. Family size 1 (Holm/BH threshold == alpha exactly at rank 1) makes this
    # the sharpest possible test: without the individually_clears gate, p=1e-6 trivially survives.
    runs = tmp_path / "output" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    regime_summaries = {
        "never_cleared_tiny_p": {
            "regime_id": "never_cleared_tiny_p",
            "gate_clears": False,  # CI never separated from zero
            "one_sided_p_max": 1e-6,  # yet the raw p is minuscule
            "min_clustered_ci_low": -0.02,
        }
    }
    (runs / "aif_regime_search.json").write_text(
        json.dumps({"regime_summaries": regime_summaries}), encoding="utf-8"
    )
    (runs / "sota_program_gate.json").write_text(
        json.dumps({"lanes": {}}), encoding="utf-8"
    )
    (runs / "partial_observability_analysis.json").write_text(
        json.dumps(
            {
                "cross_realization_n_draws": 5,
                "cross_realization_sign_stable_under_noise": True,
                "cross_realization_min_fraction_positive_under_noise": 1.0,
                "max_noise_advantage_nats": 0.77,
            }
        ),
        encoding="utf-8",
    )
    payload = build_multiplicity_ledger(tmp_path)
    row = payload["comparisons"][0]
    assert row["individually_clears"] is False
    assert row["uncorrected_significant"] is False
    # Without the fix, holm_adjusted_p == 1e-6 <= alpha would make this True — the exact bug.
    assert row["holm_adjusted_p"] <= 0.05
    assert row["holm_survives"] is False
    assert row["bh_survives"] is False
    assert payload["holm"]["n_survivors"] == 0
    assert payload["benjamini_hochberg"]["n_survivors"] == 0


def test_family_size_is_independent_of_clearance_not_derived_from_it(tmp_path):
    # Advisor gate (session 27, 2026-07-02): multiplicity correction is only honest if the family
    # size (denominator) is an independent count of every test ATTEMPTED, not derived from the
    # cleared subset (numerator) — else correction is vacuous (correcting a subset by its own
    # size). enumerate_comparison_family includes every comparison with a p-value field
    # regardless of gate_clears/individually_clears; assert this holds even when EVERY comparison
    # in a large family individually fails to clear (family_size must still equal the full count,
    # not shrink toward the 0 that cleared).
    runs = tmp_path / "output" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    regime_summaries = {
        f"r{i}": {
            "regime_id": f"r{i}",
            "gate_clears": False,
            "one_sided_p_max": 0.9,
            "min_clustered_ci_low": -0.5,
        }
        for i in range(20)
    }
    (runs / "aif_regime_search.json").write_text(
        json.dumps({"regime_summaries": regime_summaries}), encoding="utf-8"
    )
    (runs / "sota_program_gate.json").write_text(
        json.dumps({"lanes": {}}), encoding="utf-8"
    )
    (runs / "partial_observability_analysis.json").write_text(
        json.dumps(
            {
                "cross_realization_n_draws": 5,
                "cross_realization_sign_stable_under_noise": True,
                "cross_realization_min_fraction_positive_under_noise": 1.0,
                "max_noise_advantage_nats": 0.77,
            }
        ),
        encoding="utf-8",
    )
    payload = build_multiplicity_ledger(tmp_path)
    # zero comparisons clear, yet the family size must still be the full attempted count (20),
    # not 0 — proving family_size is not derived from the cleared subset.
    assert payload["n_uncorrected_significant"] == 0
    assert payload["family_size"] == 20
    assert len(payload["comparisons"]) == 20


def test_partial_observability_replication_survives_family_bonferroni(tmp_path):
    # D=5 draws -> sign-test bound 0.5**5 = 0.03125; family of 9 -> Bonferroni threshold
    # 0.05/9 = 0.00556. 0.03125 > 0.00556, so 5 draws is NOT enough for Bonferroni here.
    runs = tmp_path / "output" / "runs"
    regimes = {f"r{i}": 0.6 for i in range(5)}
    lanes = {f"lane{i}": 0.7 for i in range(4)}
    _write_artifacts(runs, regime_pvalues=regimes, lane_pvalues=lanes, n_draws=5)
    payload = build_multiplicity_ledger(tmp_path)
    po = payload["partial_observability_positive"]
    assert po["n_independent_draws"] == 5
    assert math.isclose(po["sign_test_bound"], 0.03125, abs_tol=1e-9)
    assert po["survives_family_bonferroni"] is False
    # With more draws (D=12 -> 0.000244 < 0.00556) it clears.
    _write_artifacts(runs, regime_pvalues=regimes, lane_pvalues=lanes, n_draws=12)
    payload2 = build_multiplicity_ledger(tmp_path)
    assert (
        payload2["partial_observability_positive"]["survives_family_bonferroni"] is True
    )


def test_additive_completeness_fields_are_family_specific_and_non_gating(tmp_path):
    # Additive completeness: top-level CI descriptors document how clustered_ci_low is built,
    # SOTA lanes carry a coherent same-comparison clustered_ci_high, and regimes carry the
    # worst-realization min_advantage_nats (the honest conjunction floor). None of these may
    # alter any verdict/count, and a regime must NEVER receive a clustered_ci_high (its
    # min_clustered_ci_low and max_clustered_ci_high come from different realizations).
    runs = tmp_path / "output" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    regime_summaries = {
        "r_clear": {
            "regime_id": "r_clear",
            "gate_clears": True,
            "mean_advantage_nats": 0.05,
            "min_advantage_nats": 0.011,
            "min_clustered_ci_low": 0.002,
            "one_sided_p_max": 0.01,
        }
    }
    (runs / "aif_regime_search.json").write_text(
        json.dumps({"regime_summaries": regime_summaries}), encoding="utf-8"
    )
    lanes = {
        "lane_x": {
            "one_sided_p": 0.3,
            "mean_difference_nats_fair_minus_aif": 0.03,
            "clustered_ci_low": -0.02,
            "clustered_ci_high": 0.041,
            "aif_beats_strongest_fair": False,
            "n_holdout_travellers": 6,
        }
    }
    (runs / "sota_program_gate.json").write_text(
        json.dumps({"lanes": lanes}), encoding="utf-8"
    )
    (runs / "partial_observability_analysis.json").write_text(
        json.dumps(
            {
                "cross_realization_n_draws": 5,
                "cross_realization_sign_stable_under_noise": True,
                "cross_realization_min_fraction_positive_under_noise": 1.0,
                "max_noise_advantage_nats": 0.77,
            }
        ),
        encoding="utf-8",
    )
    payload = build_multiplicity_ledger(tmp_path)
    # top-level factual CI descriptors
    assert payload["ci_level"] == 0.95
    assert payload["ci_lower_percentile"] == 2.5
    # no effective-alpha field was introduced (explicitly out of scope)
    assert "clearance_effective_one_sided_alpha" not in payload
    regime_row = next(r for r in payload["comparisons"] if r["family"] == "aif_regime")
    lane_row = next(r for r in payload["comparisons"] if r["family"] == "sota_lane")
    # regime carries the worst-realization floor, and NO clustered_ci_high
    assert regime_row["min_advantage_nats"] == 0.011
    assert "clustered_ci_high" not in regime_row
    # lane carries the coherent same-comparison upper edge, and NO min_advantage_nats
    assert lane_row["clustered_ci_high"] == 0.041
    assert "min_advantage_nats" not in lane_row
    # additive fields never touch the verdict machinery: the clearing regime's real Holm
    # outcome (family of 2, p=0.01 -> adj 0.02 <= alpha, and it individually cleared) stands.
    assert payload["n_uncorrected_significant"] == 1
    assert payload["holm"]["n_survivors"] == 1
    assert payload["holm"]["survivor_ids"] == ["aif_regime::r_clear"]


def test_additive_fields_guard_missing_upstream_keys(tmp_path):
    # A lane without clustered_ci_high and a regime without min_advantage_nats must not crash;
    # the guarded .get path emits the additive key as None rather than raising.
    runs = tmp_path / "output" / "runs"
    _write_artifacts(
        runs,
        regime_pvalues={"r0": 0.01},  # helper omits min_advantage_nats
        lane_pvalues={"lane0": 0.3},  # helper omits clustered_ci_high
    )
    payload = build_multiplicity_ledger(tmp_path)
    regime_row = next(r for r in payload["comparisons"] if r["family"] == "aif_regime")
    lane_row = next(r for r in payload["comparisons"] if r["family"] == "sota_lane")
    assert regime_row["min_advantage_nats"] is None
    assert lane_row["clustered_ci_high"] is None


def test_run_and_write_emits_artifact(tmp_path):
    runs = tmp_path / "output" / "runs"
    _write_artifacts(
        runs,
        regime_pvalues={"r0": 0.01, "r1": 0.5},
        lane_pvalues={"lane0": 0.3},
    )
    path = run_and_write_multiplicity_ledger(tmp_path)
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "active-fractal-rabbit-multiplicity-ledger-v1"
    assert payload["family_size"] == 3
    assert "verdict" in payload and payload["verdict"]
