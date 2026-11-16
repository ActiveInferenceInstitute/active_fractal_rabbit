from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from active_fractal_rabbit.aif_deepening import run_and_write_aif_deepening_candidates
from active_fractal_rabbit.aif_regime_search import (
    AIF_REGIME_GATE_SCHEMA,
    AIF_REGIME_SEARCH_SCHEMA,
    _regime_summaries,
    gate_payload,
    run_and_write_aif_regime_search,
    run_regime_search,
)
from active_fractal_rabbit.confound_taxonomy import (
    _verify_regime_clearance_was_an_instrumentation_bug,
)
from active_fractal_rabbit.multiplicity_ledger import (
    MultiplicityLedgerError,
    diagnostic_family_evidence,
    enumerate_comparison_family,
    run_and_write_multiplicity_ledger,
)
from active_fractal_rabbit.power_analysis import DEFAULT_GEN_SEEDS
from active_fractal_rabbit.sota_program import (
    build_partial_observability_dataset_from_sequence,
    paired_loss_difference_ci,
)
from active_fractal_rabbit.sota_program_harness import (
    CANONICAL_LANES,
    build_diagnostic_lanes,
    build_inference_config,
    robustness_diagnostic_payload,
    run_diagnostic_robustness_suite,
    write_robustness_diagnostics,
)
from active_fractal_rabbit.sota_stressors import (
    DIAGNOSTIC_SPLIT_SEED,
    build_adversarial_missingness_partial_observability_dataset,
    build_regime_stressor_repeat_partial_observability_dataset,
    build_reporting_gap_partial_observability_dataset,
)

ROOT = Path(__file__).resolve().parents[1]


def _inference():
    return build_inference_config(ROOT / "configs/default.yaml", spatial_bins=3)


def test_scope23_contract_full_existing_profile_grid_is_order_independent_and_train_only():
    inference = _inference()
    lanes = build_diagnostic_lanes(inference, n_travellers=8)
    expected_new = {
        f"{profile}_draw_{seed}"
        for profile in (
            "reporting_gap",
            "high_motion",
            "regime_switch",
            "regime_switch_long",
        )
        for seed in DEFAULT_GEN_SEEDS
    }
    legacy = {
        "reporting_gap_adversarial",
        "reporting_gap_repeat",
        "high_motion_adversarial_repeat",
        "regime_stressor_repeat",
        "regime_stressor_repeat_2",
    }
    assert set(lanes) == expected_new | legacy
    assert set(lanes).isdisjoint(CANONICAL_LANES)
    selected = ("high_motion_draw_2027", "reporting_gap_draw_2026")
    reordered = build_diagnostic_lanes(
        inference, n_travellers=8, lane_ids=selected[::-1]
    )
    for name, dataset in lanes.items():
        train = set(dataset.travellers[~dataset.is_holdout])
        holdout = set(dataset.travellers[dataset.is_holdout])
        assert train and holdout and train.isdisjoint(holdout)
        metadata = dataset.split["lane_metadata"]
        assert metadata["split_seed"] == DIAGNOSTIC_SPLIT_SEED
        assert sum(metadata["retained_reports_per_traveller"].values()) == len(
            dataset.latent
        )
        changed = dataset.latent.copy()
        changed[dataset.is_holdout] = (
            changed[dataset.is_holdout] + 1
        ) % dataset.n_states
        changed_sequence = replace(dataset.sequence, spatial_states=changed)
        rebuilt = build_partial_observability_dataset_from_sequence(
            changed_sequence,
            eps=dataset.eps,
            smoothing=dataset.transition_smoothing,
            split_seed=DIAGNOSTIC_SPLIT_SEED,
            noise_seed=metadata["noise_seed"],
        )
        np.testing.assert_array_equal(dataset.transition, rebuilt.transition)
        if name in reordered:
            np.testing.assert_array_equal(dataset.latent, reordered[name].latent)
            np.testing.assert_array_equal(dataset.observed, reordered[name].observed)
            np.testing.assert_array_equal(
                dataset.is_holdout, reordered[name].is_holdout
            )
    assert not np.array_equal(
        lanes["high_motion_draw_2026"].latent, lanes["high_motion_draw_2027"].latent
    )


def test_scope23_contract_legacy_diagnostic_draws_keep_exact_generator_and_noise_axes():
    inference = _inference()
    builders = {
        "reporting_gap_adversarial": (
            build_adversarial_missingness_partial_observability_dataset,
            2026,
            0,
        ),
        "reporting_gap_repeat": (
            build_reporting_gap_partial_observability_dataset,
            20260623,
            11,
        ),
        "high_motion_adversarial_repeat": (
            build_adversarial_missingness_partial_observability_dataset,
            20260624,
            12,
        ),
        "regime_stressor_repeat": (
            build_regime_stressor_repeat_partial_observability_dataset,
            20260622,
            0,
        ),
        "regime_stressor_repeat_2": (
            build_regime_stressor_repeat_partial_observability_dataset,
            20260625,
            13,
        ),
    }
    lanes = build_diagnostic_lanes(inference, n_travellers=6, lane_ids=tuple(builders))
    for name, (builder, seed, noise) in builders.items():
        original = builder(inference, n_travellers=6, gen_seed=seed, noise_seed=noise)
        np.testing.assert_array_equal(lanes[name].latent, original.latent)
        np.testing.assert_array_equal(lanes[name].observed, original.observed)
        np.testing.assert_array_equal(lanes[name].transition, original.transition)


@pytest.mark.parametrize(
    "ids", [(), ("unknown",), ("reporting_gap_repeat", "reporting_gap_repeat")]
)
def test_scope23_contract_invalid_diagnostic_selection_refuses(ids):
    with pytest.raises(ValueError):
        build_diagnostic_lanes(_inference(), n_travellers=6, lane_ids=ids)


def test_scope23_contract_scored_diagnostic_carries_real_p_ci_and_separate_m10_protocol(
    tmp_path,
):
    from active_fractal_rabbit.neural_baseline import neural_baseline_report

    inference = _inference()
    ids = ("high_motion_draw_2026",)
    gate = run_diagnostic_robustness_suite(
        inference, n_travellers=6, n_boot=31, boot_seed=9, lane_ids=ids
    )
    lane = gate.lanes[ids[0]]
    payload = robustness_diagnostic_payload(gate)
    reported = payload["diagnostic_lanes"][ids[0]]
    paired = paired_loss_difference_ci(
        lane.scored[lane.strongest_fair_name],
        lane.scored[lane.aif_name],
        n_boot=31,
        seed=9,
    )
    assert reported["one_sided_p"] == paired.one_sided_p_b_better
    assert reported["clustered_ci_low"] == paired.clustered_ci_low
    assert reported["clustered_ci_high"] == paired.clustered_ci_high
    assert (
        reported["mean_difference_nats_fair_minus_aif"] == paired.mean_difference_nats
    )
    assert reported["provenance"]["p_resolution_floor"] == 1 / 32
    assert reported["provenance"]["n_boot"] == 31
    assert payload["expected_cell_count"] == payload["scored_cell_count"] == 1
    assert gate.gate_clears is False
    dataset = build_diagnostic_lanes(inference, n_travellers=6, lane_ids=ids)[ids[0]]
    m10 = reported["bounded_m10_analyses"]
    assert m10["source_rows"] == len(dataset.sequence.observations)
    assert m10["neural_baseline"] == neural_baseline_report(
        dataset.sequence, smoothing=inference.smoothing
    )
    for key in (
        "latent_regime_hmm",
        "neural_baseline",
        "learned_stay_rate",
        "heldout_reporting_process",
    ):
        assert m10[key]["status"] == "completed"
    assert m10["protocol"] != reported["provenance"]["protocol"]
    paths = write_robustness_diagnostics(gate, tmp_path)
    stored = json.loads(paths["robustness_diagnostics"].read_text())
    assert (
        stored["leaderboard_csv_sha256"]
        == hashlib.sha256(paths["robustness_grid"].read_bytes()).hexdigest()
    )
    assert (
        stored["diagnostic_lanes"][ids[0]]["one_sided_p"] == paired.one_sided_p_b_better
    )


def test_scope23_contract_aif_diagnostics_keep_fixed_split_separate_from_canonical_schedule():
    payload = run_regime_search(
        _inference(),
        regime_ids=("identity_emission_collapse",),
        realization_seeds=(20260625, 20260626),
        diagnostic_realization_seeds=(2027, 2026),
        n_travellers=6,
        n_boot=31,
    )
    assert [row["split_seed"] for row in payload["realization_summaries"]] == [
        20260616,
        20260617,
    ]
    diagnostic = payload["diagnostic_repetitions"]
    rows = diagnostic["realization_summaries"]
    assert [row["realization_seed"] for row in rows] == [2027, 2026]
    assert [row["split_seed"] for row in rows] == [20260616, 20260616]
    assert [row["noise_seed"] for row in rows] == [
        20260701 + seed % 997 for seed in (2027, 2026)
    ]
    assert (
        rows[0]["provenance"]["split"]["holdout_traveller_ids"]
        == rows[1]["provenance"]["split"]["holdout_traveller_ids"]
    )
    summary = diagnostic["regime_summaries"]["identity_emission_collapse"]
    assert summary["one_sided_p_max"] == max(row["one_sided_p"] for row in rows)
    assert summary["gate_clears"] is False
    assert payload["gate_clears"] is False
    assert (
        diagnostic["expected_realization_count"]
        == diagnostic["scored_realization_count"]
        == 2
    )


def test_scope23_contract_one_bad_repeat_blocks_conjunction_and_keeps_worst_p():
    common = dict(
        regime_id="repeated",
        eligible_for_regime_sota=True,
        best_aif_name="aif",
        strongest_fair_name="fair",
        leakage_floor_ok=True,
        clustered_ci_high=0.2,
    )
    winning = SimpleNamespace(
        **common,
        realization_seed=1,
        aif_beats_strongest_fair=True,
        mean_difference_nats=0.1,
        clustered_ci_low=0.01,
        one_sided_p=0.001,
    )
    losing = SimpleNamespace(
        **common,
        realization_seed=2,
        aif_beats_strongest_fair=False,
        mean_difference_nats=-0.1,
        clustered_ci_low=-0.2,
        one_sided_p=0.9,
    )
    summary = _regime_summaries([winning, losing])["repeated"]
    assert summary["gate_clears"] is False
    assert summary["one_sided_p_max"] == 0.9
    assert summary["min_advantage_nats"] == -0.1


def _dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2))


def _bind_grid(runs):
    path = runs / "sota_program_gate.json"
    sota = json.loads(path.read_text())
    sota["run_parameters"]["diagnostic_artifact_sha256"] = hashlib.sha256(
        (runs / "sota_program_robustness_grid.json").read_bytes()
    ).hexdigest()
    _dump(path, sota)


def _bind_regime(runs):
    path = runs / "aif_regime_search.json"
    payload = json.loads(path.read_text())
    _dump(
        runs / "aif_regime_sota_gate.json",
        gate_payload(
            payload,
            search_artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ),
    )


def _comparison_inputs(runs):
    ids = ["reporting_gap_draw_2026", "high_motion_draw_2027"]
    rows = {
        name: {
            "one_sided_p": p,
            "mean_difference_nats_fair_minus_aif": -0.02,
            "clustered_ci_low": -0.1,
            "clustered_ci_high": 0.1,
            "aif_beats_strongest_fair": False,
            "n_holdout_travellers": 3,
        }
        for name, p in zip(ids, (0.001, 0.9))
    }
    sota_csv = runs / "sota_program_robustness_grid.csv"
    sota_csv.parent.mkdir(parents=True, exist_ok=True)
    sota_csv.write_text(
        "lane,one_sided_p\n"
        + "".join(f"{name},{row['one_sided_p']}\n" for name, row in rows.items())
    )
    _dump(
        runs / "sota_program_robustness_grid.json",
        {
            "run_parameters": {"lanes": ids, "n_boot": 2000},
            "diagnostic_lanes": rows,
            "expected_cell_count": 2,
            "scored_cell_count": 2,
            "all_diagnostic_lanes_scored": True,
            "profile_count": 2,
            "profile_draw_identity_count": 2,
            "lane_statuses": [{"lane": name, "status": "scored"} for name in ids],
            "leaderboard_csv_sha256": hashlib.sha256(sota_csv.read_bytes()).hexdigest(),
        },
    )
    _dump(
        runs / "sota_program_gate.json",
        {
            "run_parameters": {"n_boot": 2000, "diagnostic_lanes": ids},
            "lanes": {"canonical_loser": {**rows[ids[0]], "one_sided_p": 0.2}},
        },
    )
    summary = {
        "gate_clears": False,
        "eligible_for_regime_sota": True,
        "n_realizations": 2,
        "realization_seeds": [2026, 2027],
        "one_sided_p_max": 0.6,
        "mean_advantage_nats": -0.02,
        "min_advantage_nats": -0.1,
        "min_clustered_ci_low": -0.2,
    }
    diagnostic = {
        "run_parameters": {
            "regime_ids": ["repeated"],
            "realization_seeds": [2026, 2027],
            "n_boot": 2000,
        },
        "regime_summaries": {"repeated": summary},
        "realization_summaries": [
            {
                "regime_id": "repeated",
                "realization_seed": 2026,
                "one_sided_p": 0.001,
                "realization_clears": True,
            },
            {
                "regime_id": "repeated",
                "realization_seed": 2027,
                "one_sided_p": 0.6,
                "realization_clears": False,
            },
        ],
        "expected_realization_count": 2,
        "scored_realization_count": 2,
        "all_realizations_scored": True,
    }
    diagnostic_csv = runs / "aif_regime_diagnostic_repetitions.csv"
    diagnostic_csv.write_text(
        "regime_id,realization_seed,one_sided_p,realization_clears\n"
        + "".join(
            f"{row['regime_id']},{row['realization_seed']},{row['one_sided_p']},{row['realization_clears']}\n"
            for row in diagnostic["realization_summaries"]
        )
    )
    diagnostic["leaderboard_csv_sha256"] = hashlib.sha256(
        diagnostic_csv.read_bytes()
    ).hexdigest()
    canonical = {
        "one_sided_p_max": 0.04,
        "gate_clears": True,
        "mean_advantage_nats": 0.02,
        "min_clustered_ci_low": 0.001,
    }
    regime = {
        "schema": AIF_REGIME_SEARCH_SCHEMA,
        "gate_schema": AIF_REGIME_GATE_SCHEMA,
        "status": "regime_qualified_sota_cleared",
        "gate_clears": True,
        "cleared_regime_count": 1,
        "best_cleared_regime": canonical,
        "best_cleared_regime_id": "canonical_win",
        "best_aif_candidates": [],
        "best_strongest_fair_baselines": [],
        "best_min_advantage_nats": canonical["mean_advantage_nats"],
        "best_min_clustered_ci_low": canonical["min_clustered_ci_low"],
        "claim": "deterministic comparison-family fixture only",
        "global_sota_boundary": "No scientific or publication qualification.",
        "controls": {},
        "regime_definitions": {},
        "realization_summaries": [],
        "leaderboard_columns": [],
        "blocked_candidate_families": [],
        "run_parameters": {"n_boot": 2000},
        "diagnostic_repetitions": diagnostic,
        "regime_summaries": {"canonical_win": canonical},
    }
    _dump(runs / "aif_regime_search.json", regime)
    _bind_regime(runs)
    _dump(
        runs / "partial_observability_analysis.json",
        {
            "cross_realization_n_draws": 5,
            "cross_realization_sign_stable_under_noise": True,
            "cross_realization_min_fraction_positive_under_noise": 1.0,
            "max_noise_advantage_nats": 0.7,
        },
    )
    _bind_grid(runs)


def test_scope23_contract_joint_family_includes_losers_without_resurrection_and_owns_input_root(
    tmp_path,
):
    runs = tmp_path / "custom-input"
    _comparison_inputs(runs)
    output = run_and_write_multiplicity_ledger(
        tmp_path / "project", tmp_path / "owned-output", runs_dir=runs
    )
    payload = json.loads(output.read_text())
    by_id = {row["comparison_id"]: row for row in payload["comparisons"]}
    assert set(by_id) == {
        "aif_regime::canonical_win",
        "sota_lane::canonical_loser",
        "aif_regime_diagnostic::repeated",
        "sota_diagnostic::reporting_gap_draw_2026",
        "sota_diagnostic::high_motion_draw_2027",
    }
    assert payload["family_size"] == 5
    assert by_id["aif_regime::canonical_win"]["holm_adjusted_p"] == 0.16
    assert by_id["aif_regime_diagnostic::repeated"]["uncorrected_p"] == 0.6
    for row in by_id.values():
        if not row["individually_clears"]:
            assert not row["uncorrected_significant"]
            assert not row["holm_survives"]
            assert not row["bh_survives"]
    assert diagnostic_family_evidence(runs, payload)["current"] is True
    assert set(payload["source_artifact_sha256"]) == {
        "aif_regime_search.json",
        "aif_regime_sota_gate.json",
        "aif_regime_diagnostic_repetitions.csv",
        "sota_program_gate.json",
        "sota_program_robustness_grid.json",
        "sota_program_robustness_grid.csv",
    }
    assert not (tmp_path / "project/output").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "missing_grid",
        "stale_grid",
        "missing_p",
        "missing_cell",
        "missing_repeat",
        "best_p_substitution",
    ],
)
def test_scope23_contract_incomplete_diagnostic_family_refuses_instead_of_old_nine(
    tmp_path, damage
):
    runs = tmp_path / "inputs"
    _comparison_inputs(runs)
    grid_path = runs / "sota_program_robustness_grid.json"
    regime_path = runs / "aif_regime_search.json"
    if damage == "missing_grid":
        grid_path.unlink()
    elif damage == "stale_grid":
        grid_path.write_text(grid_path.read_text() + "\n")
    elif damage in {"missing_p", "missing_cell"}:
        grid = json.loads(grid_path.read_text())
        if damage == "missing_p":
            del grid["diagnostic_lanes"]["reporting_gap_draw_2026"]["one_sided_p"]
        else:
            del grid["diagnostic_lanes"]["reporting_gap_draw_2026"]
        _dump(grid_path, grid)
        _bind_grid(runs)
    else:
        regime = json.loads(regime_path.read_text())
        diagnostic = regime["diagnostic_repetitions"]
        if damage == "missing_repeat":
            diagnostic["realization_summaries"].pop()
        else:
            diagnostic["regime_summaries"]["repeated"]["one_sided_p_max"] = 0.001
        _dump(regime_path, regime)
        _bind_regime(runs)
    with pytest.raises(MultiplicityLedgerError):
        enumerate_comparison_family(runs)


def test_scope23_contract_audit_rederives_new_diagnostic_clearances(tmp_path):
    runs = tmp_path / "output/runs"
    _comparison_inputs(runs)
    ledger_path = run_and_write_multiplicity_ledger(tmp_path)
    assert (
        _verify_regime_clearance_was_an_instrumentation_bug(tmp_path)["verified"]
        is True
    )
    ledger = json.loads(ledger_path.read_text())
    diagnostic = next(
        row
        for row in ledger["comparisons"]
        if row["comparison_id"] == "sota_diagnostic::reporting_gap_draw_2026"
    )
    diagnostic["uncorrected_significant"] = True
    _dump(ledger_path, ledger)
    assert (
        _verify_regime_clearance_was_an_instrumentation_bug(tmp_path)["verified"]
        is False
    )
    ledger["comparisons"].remove(diagnostic)
    _dump(ledger_path, ledger)
    assert (
        _verify_regime_clearance_was_an_instrumentation_bug(tmp_path)["verified"]
        is False
    )


def test_scope23_contract_deepening_keeps_correction_custody_and_separate_adaptation_count(
    tmp_path,
):
    runs = tmp_path / "inputs"
    _comparison_inputs(runs)
    _dump(
        runs / "adaptive_aif_ablation.json",
        {"status": "adaptive_repeated_ablation_scored", "realization_count": 7},
    )
    run_and_write_multiplicity_ledger(tmp_path, runs, runs_dir=runs)
    output = run_and_write_aif_deepening_candidates(
        tmp_path, tmp_path / "owned-output/deepening.json", runs_dir=runs
    )
    payload = json.loads(output.read_text())
    assert payload["diagnostic_correction"]["current"] is True
    assert payload["diagnostic_correction"]["diagnostic_comparison_count"] == 3
    row = next(
        row for row in payload["rows"] if row["candidate_id"] == "online_adaptation"
    )
    assert row["diagnostic_profile_count"] == 2
    assert row["diagnostic_scored_cell_count"] == 2
    assert row["realization_count"] == 7
    assert row["diagnostics_count_as_adaptation_trials"] is False
    assert payload["predictive_gate_clears"] is False
    path = runs / "aif_regime_search.json"
    path.write_text(path.read_text() + "\n")
    run_and_write_aif_deepening_candidates(tmp_path, output, runs_dir=runs)
    assert json.loads(output.read_text())["diagnostic_correction"]["current"] is False


def test_scope23_contract_real_regime_producer_binds_csv_search_and_gate(tmp_path):
    payload, paths = run_and_write_aif_regime_search(
        ROOT / "configs/default.yaml",
        tmp_path,
        n_travellers=6,
        n_boot=31,
        boot_seed=9,
        spatial_bins=3,
        regime_ids=("identity_emission_collapse",),
        realization_seeds=(20260625,),
        diagnostic_realization_seeds=(2026, 2027),
    )
    stored = json.loads(paths["search"].read_text())
    gate = json.loads(paths["gate"].read_text())
    assert stored == payload
    assert (
        stored["diagnostic_repetitions"]["leaderboard_csv_sha256"]
        == hashlib.sha256(paths["diagnostic_leaderboard"].read_bytes()).hexdigest()
    )
    digest = hashlib.sha256(paths["search"].read_bytes()).hexdigest()
    assert gate == gate_payload(stored, search_artifact_sha256=digest)
    assert gate["diagnostic_repetitions"] == stored["diagnostic_repetitions"]
    assert stored["run_parameters"]["realization_seeds"] == [20260625]
    assert stored["diagnostic_repetitions"]["run_parameters"]["realization_seeds"] == [
        2026,
        2027,
    ]
    assert gate["gate_clears"] is False


@pytest.mark.parametrize(
    "name,damage",
    [
        ("sota_program_robustness_grid.csv", "missing"),
        ("sota_program_robustness_grid.csv", "changed"),
        ("aif_regime_diagnostic_repetitions.csv", "missing"),
        ("aif_regime_diagnostic_repetitions.csv", "changed"),
        ("aif_regime_sota_gate.json", "missing"),
        ("aif_regime_sota_gate.json", "mismatched"),
    ],
)
def test_scope23_contract_companion_damage_refuses_ledger_and_deepening(
    tmp_path, name, damage
):
    runs = tmp_path / "inputs"
    _comparison_inputs(runs)
    _dump(
        runs / "adaptive_aif_ablation.json",
        {
            "status": "adaptive_repeated_ablation_scored",
            "realization_count": 7,
        },
    )
    ledger_path = run_and_write_multiplicity_ledger(tmp_path, runs, runs_dir=runs)
    ledger = json.loads(ledger_path.read_text())
    assert diagnostic_family_evidence(runs, ledger)["current"] is True
    path = runs / name
    if damage == "missing":
        path.unlink()
    elif damage == "mismatched":
        gate = json.loads(path.read_text())
        gate["gate_clears"] = not gate["gate_clears"]
        _dump(path, gate)
    else:
        path.write_text(path.read_text() + "\n")
    with pytest.raises(MultiplicityLedgerError):
        enumerate_comparison_family(runs)
    with pytest.raises(MultiplicityLedgerError):
        run_and_write_multiplicity_ledger(tmp_path, runs, runs_dir=runs)
    assert json.loads(ledger_path.read_text()) == ledger
    output = run_and_write_aif_deepening_candidates(
        tmp_path,
        tmp_path / "owned-output/deepening.json",
        runs_dir=runs,
    )
    payload = json.loads(output.read_text())
    assert payload["diagnostic_correction"]["current"] is False
    assert payload["diagnostic_regime_repetitions"] == {}
    multiscale = next(
        row
        for row in payload["rows"]
        if row["candidate_id"] == "multiscale_soft_belief_backoff"
    )
    assert multiscale["status"] == "unavailable"
    assert multiscale["local_regime_clears"] is False
    assert multiscale["global_sota_gate_clears"] is False
    online = next(
        row for row in payload["rows"] if row["candidate_id"] == "online_adaptation"
    )
    assert (
        online["diagnostic_profile_count"]
        == online["diagnostic_scored_cell_count"]
        == 0
    )
    assert online["realization_count"] == 7
    assert payload["predictive_gate_clears"] is False
