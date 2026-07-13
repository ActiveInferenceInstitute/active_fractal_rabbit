from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from active_fractal_rabbit.aif_regime_search import (
    AIF_REGIME_GATE_SCHEMA,
    AIF_REGIME_SEARCH_HEADER,
    AIF_REGIME_SEARCH_SCHEMA,
    build_aif_regime_specs,
    build_regime_dataset,
    run_and_write_aif_regime_search,
    run_regime_search,
    score_regime_realization,
)
from active_fractal_rabbit.sota_program_harness import (
    DEFAULT_BOOT_SEED,
    LEAKAGE_FLOOR_NATS,
    build_inference_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "default.yaml"


def _inference():
    return build_inference_config(CONFIG, spatial_bins=6)


def test_ambiguous_regime_is_deterministic_locked_and_strictly_positive() -> None:
    kwargs = {
        "n_travellers": 10,
        "realization_seed": 20260625,
        "split_seed": 20260616,
        "noise_seed": 20260701,
    }
    first = build_regime_dataset("ambiguous_pairs_010", _inference(), **kwargs)
    second = build_regime_dataset("ambiguous_pairs_010", _inference(), **kwargs)

    assert np.array_equal(first.latent, second.latent)
    assert np.array_equal(first.observed, second.observed)
    assert np.array_equal(first.is_holdout, second.is_holdout)
    assert np.min(first.emission) > 0.0
    holdout = set(int(t) for t in first.travellers[first.is_holdout])
    train = set(int(t) for t in first.travellers[~first.is_holdout])
    assert holdout and train and holdout.isdisjoint(train)


def test_identity_emission_collapse_is_negative_control_not_promotion() -> None:
    specs = build_aif_regime_specs()
    spec = specs["identity_emission_collapse"]
    dataset = build_regime_dataset(
        "identity_emission_collapse",
        _inference(),
        n_travellers=8,
        realization_seed=20260625,
        split_seed=20260616,
        noise_seed=20260701,
    )
    result = score_regime_realization(
        spec,
        dataset,
        realization_seed=20260625,
        split_seed=20260616,
        noise_seed=20260701,
        n_boot=20,
        boot_seed=DEFAULT_BOOT_SEED,
    )

    assert result.eligible_for_regime_sota is False
    assert result.aif_beats_strongest_fair is False
    assert result.best_aif_loss > LEAKAGE_FLOOR_NATS
    assert result.strongest_fair_name not in {"soft_filter", "precision_return_aif"}

    # Mechanism-level rejection (NOT the eligibility flag): on identity-emission-collapse data the
    # observation equals the latent cell, so recursive belief marginalization collapses to the same
    # point estimate a fair Markov baseline already has. The best AIF candidate's loss must therefore
    # NOT be better than the strongest fair baseline's loss — the spurious "win" is rejected by the
    # scoring itself, independent of the SOTA-eligibility flag being False by construction.
    assert result.best_aif_loss >= result.strongest_fair_loss

    # Prove the negative control does not rely SOLELY on the eligibility flag: flip the flag to True
    # and re-score. `aif_beats_strongest_fair` must still be False, because the loss comparison
    # (best_aif >= strongest_fair) and the traveller-clustered CI both independently reject the win.
    forced_eligible = replace(spec, eligible_for_regime_sota=True)
    forced_result = score_regime_realization(
        forced_eligible,
        dataset,
        realization_seed=20260625,
        split_seed=20260616,
        noise_seed=20260701,
        n_boot=20,
        boot_seed=DEFAULT_BOOT_SEED,
    )
    assert forced_result.eligible_for_regime_sota is True
    assert forced_result.best_aif_loss >= forced_result.strongest_fair_loss
    assert forced_result.aif_beats_strongest_fair is False
    # the clustered CI is on the fair side of zero (AIF does not win the paired difference)
    assert not forced_result.clustered_ci_low > 0.0


def test_regime_search_payload_has_complete_rows_and_boundaries() -> None:
    payload = run_regime_search(
        _inference(),
        regime_ids=("identity_emission_collapse", "symmetric_noise_045"),
        realization_seeds=(20260625,),
        n_travellers=8,
        n_boot=20,
    )

    assert payload["schema"] == AIF_REGIME_SEARCH_SCHEMA
    assert payload["gate_schema"] == AIF_REGIME_GATE_SCHEMA
    assert payload["status"] in {
        "regime_qualified_sota_cleared",
        "regime_qualified_sota_not_cleared",
    }
    assert (
        payload["controls"]["identity_emission_collapse_negative_control_passed"]
        is True
    )
    assert payload["controls"]["locked_split_train_only_contract"] is True
    assert "does not imply external" in payload["global_sota_boundary"]
    rows = payload["leaderboard_rows"]
    assert rows
    assert set(AIF_REGIME_SEARCH_HEADER) <= set(rows[0])
    assert {row["regime_id"] for row in rows} == {
        "identity_emission_collapse",
        "symmetric_noise_045",
    }
    assert all(row["locked_split_scored"] is True for row in rows)
    assert all(row["train_only_transition"] is True for row in rows)
    assert all(row["emission_contract"] == "known_to_all_candidates" for row in rows)


def test_writer_emits_csv_search_json_and_gate_json(tmp_path: Path) -> None:
    payload, paths = run_and_write_aif_regime_search(
        config_path=CONFIG,
        out_dir=tmp_path,
        n_travellers=8,
        n_boot=20,
        regime_ids=("identity_emission_collapse",),
        realization_seeds=(20260625,),
    )

    assert paths["leaderboard"].is_file()
    assert paths["search"].is_file()
    assert paths["gate"].is_file()
    with paths["leaderboard"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and set(AIF_REGIME_SEARCH_HEADER) <= set(rows[0])
    search = json.loads(paths["search"].read_text(encoding="utf-8"))
    gate = json.loads(paths["gate"].read_text(encoding="utf-8"))
    assert search["schema"] == AIF_REGIME_SEARCH_SCHEMA
    assert gate["schema"] == AIF_REGIME_GATE_SCHEMA
    assert gate["status"] == payload["status"]
    assert gate["best_cleared_regime_id"] == "none"
