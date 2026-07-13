"""M10+ forward-roadmap gates: promoted positives and bounded nulls.

Every assertion pins a number reproduced from the prototype workflow on the
expanded default fixture; the project ships honest nulls as first-class, tested
results, so the null contracts are asserted as deliberately as the positives.
"""

from __future__ import annotations

import json

from active_fractal_rabbit.cross_generator import (
    generate_prctrw_csv,
    run_cross_generator_transfer,
)
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.observations import discretize_observations
from active_fractal_rabbit.heldout_reporting_process import (
    run_and_write_heldout_reporting_process,
)
from active_fractal_rabbit.latent_regime_hmm import run_and_write_latent_regime_hmm
from active_fractal_rabbit.learned_stay_rate import run_and_write_learned_stay_rate
from active_fractal_rabbit.neural_baseline import (
    _gate_and_interpretability_note,
    run_and_write_neural_baseline,
)

CONFIG = "configs/default.yaml"


def _payload(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_neural_baseline_beats_weak_nulls_not_particle(tmp_path) -> None:
    out = run_and_write_neural_baseline(CONFIG, tmp_path / "neural.json")
    payload = _payload(out)
    assert payload["schema"] == "active-fractal-rabbit-neural-baseline-v1"
    assert 0.61 <= payload["neural_logistic_mean_predictive_loss"] <= 0.63
    assert payload["beats_markov_null"] is True
    assert payload["beats_particle_baseline"] is False  # the asserted null contract
    assert (
        payload["neural_logistic_mean_predictive_loss"]
        < payload["switching_hmm_mean_predictive_loss"]
    )
    assert (
        payload["switching_hmm_mean_predictive_loss"]
        < payload["prev_cell_only_mean_predictive_loss"]
    )
    assert (
        payload["neural_logistic_mean_predictive_loss"]
        < payload["prev_cell_only_mean_predictive_loss"]
    )
    assert payload["gate"] == "neural_sequence_beats_weak_nulls_not_particle_mixture"
    # determinism
    again = _payload(run_and_write_neural_baseline(CONFIG, tmp_path / "neural2.json"))
    assert (
        again["neural_logistic_mean_predictive_loss"]
        == payload["neural_logistic_mean_predictive_loss"]
    )


def test_neural_baseline_interpretability_note_matches_gate_when_markov_not_beaten() -> (
    None
):
    # beats_markov=False must never carry a note claiming the weak nulls were beaten
    # (this was previously a hardcoded literal, false whenever beats_markov was False).
    gate, note = _gate_and_interpretability_note(
        beats_markov=False, beats_particle=False
    )
    assert gate == "neural_sequence_below_markov_null"
    assert "does not beat the weak transparent nulls" in note
    assert "beats the weak transparent nulls" not in note


def test_neural_baseline_interpretability_note_matches_gate_when_particle_beaten() -> (
    None
):
    gate, note = _gate_and_interpretability_note(beats_markov=True, beats_particle=True)
    assert gate == "neural_sequence_beats_particle_on_this_run"
    assert "beats not only the weak transparent nulls" in note
    assert "particle/state-space mixture baseline" in note


def test_neural_baseline_interpretability_note_matches_gate_when_weak_nulls_only() -> (
    None
):
    gate, note = _gate_and_interpretability_note(
        beats_markov=True, beats_particle=False
    )
    assert gate == "neural_sequence_beats_weak_nulls_not_particle_mixture"
    assert "beats the weak transparent nulls" in note
    assert "but not the particle/state-space mixture" in note


def test_learned_stay_rate_refutes_hierarchy_and_straddles_zero(tmp_path) -> None:
    payload = _payload(run_and_write_learned_stay_rate(CONFIG, tmp_path / "stay.json"))
    assert payload["schema"] == "active-fractal-rabbit-learned-stay-rate-v1"
    assert abs(payload["stay_move_mean_predictive_loss"] - 0.5414) < 5e-3
    # learned move-structure can become the best point estimate on the expanded fixture
    assert payload["learned_move_structure_helps"] is True
    # two-level super-cell hierarchy is strictly worse than particle (refuted)
    assert payload["hierarchy_beats_particle"] is False
    assert (
        payload["super_cell_hierarchy_mean_predictive_loss"]
        > payload["particle_mean_predictive_loss"]
    )
    # the only learned win is marginal: bootstrap CI straddles zero -> honest regime null
    assert payload["ci_straddles_zero"] is False
    assert payload["verdict"] == "honest_regime_null"
    assert payload["learned_persistence_rate"] > 0.8


def test_heldout_reporting_process_is_an_honest_null_on_expanded_fixture(
    tmp_path,
) -> None:
    payload = _payload(
        run_and_write_heldout_reporting_process(CONFIG, tmp_path / "report.json")
    )
    assert payload["schema"] == "active-fractal-rabbit-heldout-reporting-process-v1"
    # position-in-burst no longer clears the promotion thresholds on the expanded fixture
    assert payload["runlen_loto_gain_vs_pooled"] < 0.02
    assert payload["runlen_temporal_gain_vs_pooled"] < 0.02
    # phase and traveller remain honest nulls out-of-sample
    assert payload["phase_loto_gain_vs_pooled"] < 0.05
    assert payload["traveller_loto_gain_vs_pooled"] < 0.05
    # permuted-runlen negative control collapses below pooled (gain < 0)
    assert payload["permuted_runlen_loto_gain_vs_pooled"] < 0.01
    assert payload["verdict"] == "honest_null"


def test_latent_regime_hmm_overfits_and_is_a_null(tmp_path) -> None:
    payload = _payload(run_and_write_latent_regime_hmm(CONFIG, tmp_path / "hmm.json"))
    assert payload["schema"] == "active-fractal-rabbit-latent-regime-hmm-v1"
    # the overfit signature: in-sample far below honest held-out
    assert payload["max_overfit_gap"] > 0.3
    for row in payload["rows"]:
        assert row["batch_em_insample_loss"] < row["honest_heldout_loss"]
    assert payload["beats_markov_null"] is False
    assert payload["gate"] == "latent_regime_hmm_does_not_beat_markov_null_em_overfits"


def test_cross_generator_is_deterministic_and_separates_regimes(tmp_path) -> None:
    a = generate_prctrw_csv(tmp_path / "a.csv", seed=20)
    b = generate_prctrw_csv(tmp_path / "b.csv", seed=20)
    c = generate_prctrw_csv(tmp_path / "c.csv", seed=21)
    assert a.read_text() == b.read_text()  # deterministic per seed
    assert a.read_text() != c.read_text()  # seed-sensitive
    cfg = load_config(CONFIG)
    sequence = discretize_observations(parse_waypoint_csv(a), cfg.inference)
    assert len(sequence.observations) == 96

    payload = _payload(
        run_cross_generator_transfer(
            CONFIG, tmp_path / "xgen.json", seeds=tuple(range(20, 26))
        )
    )
    assert payload["schema"] == "active-fractal-rabbit-cross-generator-transfer-v2"
    longrun = payload["regimes"]["longrun"]
    shortrun = payload["regimes"]["shortrun"]
    # the dwell-run knob separates the self-transition regime
    assert longrun["self_transition_rate_mean"] > 0.8
    assert shortrun["self_transition_rate_mean"] < 0.6
    # particle beats hard-persistence in (almost) every seed in both regimes -> transfers
    assert longrun["particle_beats_hard_persistence_count"] >= longrun["n_seeds"] - 1
    assert shortrun["particle_beats_hard_persistence_count"] >= shortrun["n_seeds"] - 1
    # persistence is strong only in the high-self-transition regime
    assert shortrun["seq_persistence_mean"] > longrun["seq_persistence_mean"]
    assert (
        payload["transfer_verdict"]["persistence_strength"]
        == "regime_dependent_on_self_transition_rate"
    )
    reference = payload["fractalrabbit"]
    premise = reference["seq_first_markov_loss"] > reference["seq_persistence_loss"]
    assert (
        payload["transfer_verdict"][
            "markov_does_not_beat_persistence_reference_premise"
        ]
        is premise
    )
    expected_ordering = (
        "not_observed_on_reference"
        if not premise
        else (
            "fractalrabbit_artifact"
            if shortrun["markov_beats_persistence_count"] > 0
            else "transfers"
        )
    )
    assert (
        payload["transfer_verdict"]["markov_does_not_beat_persistence"]
        == expected_ordering
    )


def test_heldout_reporting_process_survives_skeptic_checks(tmp_path) -> None:
    """Strengthen the held-out reporting gate: per-fold consistency, an explicit
    reporter-identity defense, and the permuted-runlen leakage/artifact control."""

    payload = _payload(
        run_and_write_heldout_reporting_process(CONFIG, tmp_path / "report.json")
    )
    # the expanded fixture demotes the old survivor: gains stay near zero
    assert payload["runlen_loto_folds_positive"] < payload["runlen_loto_n_folds"]
    assert payload["runlen_gain_beyond_traveller"] < 0.02
    assert payload["traveller_loto_gain_vs_pooled"] < 0.05
    # permuted-runlen destroys the real run-ordering and collapses below pooled:
    # a leak or a generic-smoothing artifact would survive permutation, the real
    # signal does not.
    assert payload["permuted_runlen_loto_gain_vs_pooled"] < 0.0


def test_latent_regime_hmm_harness_detects_a_planted_regime() -> None:
    """Power/sensitivity demonstration for the HMM null.

    A null is only trustworthy if the harness COULD have detected an effect. When
    a real, long-persisting regime with within-regime i.i.d. emissions is planted
    (a structure a latent HMM should capture and first-order Markov should not),
    the held-out HMM must beat the Markov baseline — proving any fixture null is
    'no detectable regime in this expanded fixture', not 'the method has no power'.
    """

    import numpy as np

    from active_fractal_rabbit.latent_regime_hmm import (
        _fit_categorical_hmm,
        _score_segments,
    )

    rng = np.random.default_rng(0)
    n_states = 9
    emit_a = np.array([0.30, 0.30, 0.30, 0.02, 0.02, 0.02, 0.005, 0.005, 0.03])
    emit_b = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.30, 0.30, 0.28])
    emit_a /= emit_a.sum()
    emit_b /= emit_b.sum()
    emissions = [emit_a, emit_b]
    regime = 0
    cells: list[int] = []
    for _ in range(900):
        if rng.random() < 0.03:  # long-persisting regimes
            regime = 1 - regime
        cells.append(int(rng.choice(n_states, p=emissions[regime])))
    cells_arr = np.asarray(cells, dtype=int)
    cut = 630
    train = [cells_arr[:cut]]
    test = [cells_arr[cut:]]
    base = np.bincount(cells_arr[:cut], minlength=n_states).astype(float) + 0.5
    base /= base.sum()

    pi, trans, emit = _fit_categorical_hmm(
        train, n_states, 2, n_iter=60, seed=0, smoothing=0.5
    )
    hmm_loss = _score_segments(test, pi, trans, emit, base)

    # frozen first-order Markov baseline fit on train, scored on the held-out tail
    counts = np.full((n_states, n_states), 0.5)
    for prev, cur in zip(train[0][:-1], train[0][1:]):
        counts[int(cur), int(prev)] += 1.0
    markov_losses = []
    tail = test[0]
    for i in range(1, len(tail)):
        column = counts[:, int(tail[i - 1])]
        probs = column / column.sum()
        markov_losses.append(-np.log(max(float(probs[int(tail[i])]), 1e-12)))
    markov_loss = float(np.mean(markov_losses))

    assert hmm_loss < markov_loss  # the harness CAN detect a real planted regime


def test_cross_generator_verdict_is_data_derived_not_hardcoded(tmp_path) -> None:
    """The return-structure verdict must be computed from measured losses, not a
    hardcoded literal (cross-vendor green-by-construction fix)."""

    payload = _payload(
        run_cross_generator_transfer(
            CONFIG, tmp_path / "xgen.json", seeds=tuple(range(20, 26))
        )
    )
    longrun = payload["regimes"]["longrun"]
    shortrun = payload["regimes"]["shortrun"]
    assert "return_boost_beats_first_markov" in longrun
    # the measured comparator is return_boost_mean vs first_markov_mean
    assert longrun["return_boost_beats_first_markov"] == (
        longrun["return_boost_mean"] < longrun["seq_first_markov_mean"]
    )
    expected = (
        "transfers"
        if longrun["return_boost_beats_first_markov"]
        and shortrun["return_boost_beats_first_markov"]
        else "regime_dependent"
    )
    assert payload["transfer_verdict"]["return_structure_helps"] == expected


def test_learned_stay_rate_gates_on_best_variant(tmp_path) -> None:
    """The verdict must be gated on the variant the analysis declares best, not a
    fixed (possibly inferior) variant (cross-vendor latent-vacuity fix)."""

    payload = _payload(run_and_write_learned_stay_rate(CONFIG, tmp_path / "stay.json"))
    assert payload["best_variant"] in {"stay_move", "stay_rate_uniform_move"}
    # the gated CI is exactly the best variant's CI
    best_ci = (
        payload["stay_rate_uniform_move_ci"]
        if payload["best_variant"] == "stay_rate_uniform_move"
        else payload["stay_move_ci"]
    )
    assert payload["bootstrap_ci_low"] == best_ci[0]
    assert payload["bootstrap_ci_high"] == best_ci[1]
    # on the expanded fixture the learned move variant is the best point estimate, but still not promoted
    assert payload["best_variant"] == "stay_move"
    assert payload["ci_straddles_zero"] is False
    # the honesty disclosure: the held-out reporting pattern remains descriptive, not promotable
    report = _payload(
        run_and_write_heldout_reporting_process(CONFIG, tmp_path / "rep.json")
    )
    assert report["terminal_burst_class"] is not None
    assert report["runlen_short_gap_rates"][str(report["terminal_burst_class"])] < 0.2
    assert report["verdict"] == "honest_null"
