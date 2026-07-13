"""Tests for the regime mechanism synthesis — real data, no mocks.

Covers the pure classifier (`_mechanism_verdict`, every branch) and the end-to-end
synthesis on a small regime subset with reduced travellers/bootstrap draws for speed,
matching the convention in test_aif_regime_search.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from active_fractal_rabbit.regime_mechanism_synthesis import (
    REGIME_MECHANISM_SCHEMA,
    _mechanism_verdict,
    run_and_write_regime_mechanism_synthesis,
    run_regime_mechanism_synthesis,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "default.yaml"

_BASE_KWARGS = dict(
    eligible_for_regime_sota=True,
    mean_advantage_nats=0.05,
    ci_low=0.01,
    ci_high=0.09,
    holm_survives=True,
    clears_battery=True,
    is_mobile=True,
)


def test_mechanism_verdict_negative_control_is_trivial_regardless_of_numbers() -> None:
    verdict, explanation = _mechanism_verdict(
        **{**_BASE_KWARGS, "eligible_for_regime_sota": False}
    )
    assert verdict == "trivial_by_construction"
    assert "construction" in explanation


def test_mechanism_verdict_confined_regime_is_uninterpretable() -> None:
    verdict, _ = _mechanism_verdict(**{**_BASE_KWARGS, "is_mobile": False})
    assert verdict == "confined_uninterpretable"


def test_mechanism_verdict_ci_straddling_zero_is_null() -> None:
    verdict, _ = _mechanism_verdict(
        **{**_BASE_KWARGS, "ci_low": -0.02, "ci_high": 0.03}
    )
    assert verdict == "null_no_signal"


def test_mechanism_verdict_ci_touching_zero_is_null() -> None:
    verdict, explanation = _mechanism_verdict(
        **{**_BASE_KWARGS, "ci_low": 0.0, "ci_high": 0.03}
    )
    assert verdict == "null_no_signal"
    assert "touches" in explanation


def test_mechanism_verdict_negative_advantage_loses_to_fair_baseline() -> None:
    verdict, _ = _mechanism_verdict(
        **{
            **_BASE_KWARGS,
            "mean_advantage_nats": -0.1,
            "ci_low": -0.2,
            "ci_high": -0.02,
        }
    )
    assert verdict == "loses_to_fair_baseline"


def test_mechanism_verdict_positive_but_fails_confound_battery() -> None:
    """Proof-of-detection: a nominally positive, non-straddling win must still be
    downgraded to confound_explained when clears_battery is False — this is exactly
    the gap the RedTeam finding identified (aif_regime_search.py never checked this).
    """

    verdict, explanation = _mechanism_verdict(
        **{**_BASE_KWARGS, "clears_battery": False}
    )
    assert verdict == "confound_explained"
    assert "confound" in explanation.lower()


def test_mechanism_verdict_clears_battery_but_fails_multiplicity() -> None:
    verdict, explanation = _mechanism_verdict(
        **{**_BASE_KWARGS, "holm_survives": False}
    )
    assert verdict == "nominal_win_not_multiplicity_robust"
    assert "multiplicity" in explanation.lower() or "holm" in explanation.lower()


def test_mechanism_verdict_robust_win_requires_every_gate_to_pass() -> None:
    verdict, explanation = _mechanism_verdict(**_BASE_KWARGS)
    assert verdict == "robust_win"
    assert "regime-local synthetic" in explanation


def test_mechanism_verdict_precedence_confined_beats_positive_signal() -> None:
    """A confined regime is flagged uninterpretable even if every other gate would
    otherwise pass — mobility is checked before the win is even considered."""

    verdict, _ = _mechanism_verdict(
        **{**_BASE_KWARGS, "is_mobile": False, "clears_battery": False}
    )
    assert verdict == "confined_uninterpretable"


def test_regime_mechanism_synthesis_runs_on_small_subset_and_classifies_identity_control() -> (
    None
):
    payload = run_regime_mechanism_synthesis(
        CONFIG,
        regime_ids=("identity_emission_collapse", "ambiguous_pairs_010"),
        n_travellers=10,
        n_boot=20,
    )

    assert payload["schema"] == REGIME_MECHANISM_SCHEMA
    assert payload["status"] == "completed"
    assert payload["n_regimes"] == 2
    rows = {row["regime_id"]: row for row in payload["regimes"]}
    assert set(rows) == {"identity_emission_collapse", "ambiguous_pairs_010"}

    identity = rows["identity_emission_collapse"]
    assert identity["mechanism_verdict"] == "trivial_by_construction"
    assert identity["mean_advantage_nats"] == 0.0

    ambiguous = rows["ambiguous_pairs_010"]
    assert ambiguous["mechanism_verdict"] in {
        "null_no_signal",
        "loses_to_fair_baseline",
        "confound_explained",
        "nominal_win_not_multiplicity_robust",
        "robust_win",
    }
    # Every row carries the full mechanism surface, not just the verdict label.
    for row in rows.values():
        assert "mobility" in row and "is_mobile" in row["mobility"]
        assert "confound_battery" in row
        battery = row["confound_battery"]
        assert set(battery["beats"]) == {
            "dominant_cell",
            "first_order_pooled",
            "return_boosted_soft",
        }
        assert isinstance(battery["clears_confounds"], bool)
        assert row["mechanism_explanation"]


def test_writer_emits_valid_json(tmp_path: Path) -> None:
    out_path = tmp_path / "regime_mechanism_synthesis.json"
    written = run_and_write_regime_mechanism_synthesis(
        CONFIG,
        out_path=out_path,
        regime_ids=("identity_emission_collapse",),
        n_travellers=10,
        n_boot=20,
    )

    assert written == out_path
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == REGIME_MECHANISM_SCHEMA
    assert payload["n_regimes"] == 1


def test_synthesis_reads_live_multiplicity_and_regime_search_when_present(
    tmp_path: Path,
) -> None:
    """When aif_regime_search.json / multiplicity_ledger.json exist, the synthesis
    cross-references their live values (magnitude, CI, holm_survives) rather than
    re-deriving them independently — proving the join actually happens."""

    regime_search = tmp_path / "aif_regime_search.json"
    regime_search.write_text(
        json.dumps(
            {
                "regime_summaries": {
                    "identity_emission_collapse": {
                        "mean_advantage_nats": 0.0,
                        "min_clustered_ci_low": 0.0,
                        "max_clustered_ci_high": 0.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    ledger = tmp_path / "multiplicity_ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "comparisons": [
                    {
                        "comparison_id": "aif_regime::identity_emission_collapse",
                        "holm_survives": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = run_regime_mechanism_synthesis(
        CONFIG,
        regime_ids=("identity_emission_collapse",),
        n_travellers=10,
        n_boot=20,
        regime_search_path=regime_search,
        multiplicity_ledger_path=ledger,
    )

    row = payload["regimes"][0]
    # holm_survives=True was injected via the live ledger file; the negative-control
    # regime is still classified trivial_by_construction regardless (eligibility gate
    # fires first), but the join itself is proven by not raising/defaulting silently.
    assert row["holm_survives"] is True
    assert row["mechanism_verdict"] == "trivial_by_construction"
