"""EFE decomposition + budgeted active sensing.

The decomposition tests are hand-computed and deterministic. The active-sensing
win is deliberately NOT pinned as "beats random", because it is structure
dependent (it pays off on the richer external resimulation but not on the small
fixture) — pinning a data-dependent inequality would be a brittle, dishonest
test. Instead we pin the load-bearing MECHANISM (epistemic share under uniform
vs sharpened preferences) and the honest scoring contract.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from active_fractal_rabbit.active_sensing import (
    efe_components,
    modality_info_gain,
    modality_utility,
    run_active_sensing,
    run_and_write_active_sensing,
)
from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.observations import discretize_observations
from active_fractal_rabbit.pomdp import build_pomdp

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_info_gain_hand_computed_on_informative_modality() -> None:
    """H(prior) - E_o[H(posterior|o)] on a worked 2x2 example."""

    emission = np.array([[0.9, 0.1], [0.1, 0.9]])
    prior = np.array([0.5, 0.5])
    # H(prior)=ln2=0.6931; each posterior [0.9,0.1] has H=0.3251; expected=0.3251.
    assert modality_info_gain(prior, emission) == pytest.approx(
        0.6931 - 0.3251, abs=1e-3
    )


def test_info_gain_zero_for_uninformative_modality() -> None:
    """A modality whose emission is identical across states yields no info gain."""

    emission = np.array([[0.5, 0.5], [0.5, 0.5]])
    prior = np.array([0.3, 0.7])
    assert modality_info_gain(prior, emission) == pytest.approx(0.0, abs=1e-9)


def test_utility_is_expected_log_preference() -> None:
    emission = np.array([[0.9, 0.1], [0.1, 0.9]])
    prior = np.array([0.5, 0.5])
    # qo = [0.5, 0.5]; C=[2,0] -> 0.5*2 + 0.5*0 = 1.0.
    assert modality_utility(prior, emission, np.array([2.0, 0.0])) == pytest.approx(1.0)
    assert modality_utility(prior, emission, np.array([0.0, 0.0])) == pytest.approx(0.0)


def test_epistemic_share_is_one_under_uniform_preference() -> None:
    """No preferences -> pragmatic term is zero -> sensing value is purely epistemic."""

    emission = np.array([[0.9, 0.1], [0.1, 0.9]])
    prior = np.array([0.5, 0.5])
    comp = efe_components(prior, emission, np.array([0.0, 0.0]))
    assert comp["pragmatic"] == pytest.approx(0.0)
    assert comp["epistemic_share"] == pytest.approx(1.0)


def test_epistemic_share_falls_with_sharpened_preference() -> None:
    """Non-vacuity: sharpening C grows the pragmatic term and lowers the share."""

    emission = np.array([[0.9, 0.1], [0.1, 0.9]])
    prior = np.array([0.5, 0.5])
    uniform = efe_components(prior, emission, np.array([0.0, 0.0]))["epistemic_share"]
    sharp = efe_components(prior, emission, np.array([2.0, 0.0]))["epistemic_share"]
    assert sharp < uniform
    # info_gain 0.368, utility 1.0 -> 0.368 / 1.368.
    assert sharp == pytest.approx(0.368 / 1.368, abs=1e-2)


def _fixture_model_sequence():
    config = load_config(ROOT / "configs" / "default.yaml")
    sequence = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    model = build_pomdp(sequence, config.inference.smoothing)
    return config, model, sequence


def test_active_sensing_reports_purely_epistemic_acquisition_under_uniform_C() -> None:
    """The agent has no preferences here, so its sensing value is all epistemic."""

    config, model, sequence = _fixture_model_sequence()
    result = run_active_sensing(
        sequence,
        model,
        smoothing=config.inference.smoothing,
        fix_fraction=0.4,
        n_random_seeds=20,
    )
    assert result.budget > 0
    assert result.median_epistemic_share == pytest.approx(1.0, abs=1e-9)


def test_active_sensing_full_trajectory_brier_is_well_formed() -> None:
    config, model, sequence = _fixture_model_sequence()
    result = run_active_sensing(
        sequence,
        model,
        smoothing=config.inference.smoothing,
        fix_fraction=0.4,
        n_random_seeds=20,
    )
    # Brier of a categorical posterior against a one-hot lies in [0, 2].
    for value in (
        result.info_gain_led_brier,
        result.random_mean_brier,
        result.periodic_brier,
    ):
        assert 0.0 <= value <= 2.0
    assert 0.0 <= result.fraction_random_beaten <= 1.0


def test_active_sensing_deterministic() -> None:
    config, model, sequence = _fixture_model_sequence()
    a = run_active_sensing(
        sequence,
        model,
        smoothing=config.inference.smoothing,
        fix_fraction=0.4,
        n_random_seeds=20,
    )
    b = run_active_sensing(
        sequence,
        model,
        smoothing=config.inference.smoothing,
        fix_fraction=0.4,
        n_random_seeds=20,
    )
    assert a.info_gain_led_brier == pytest.approx(b.info_gain_led_brier)
    assert a.fraction_random_beaten == pytest.approx(b.fraction_random_beaten)


def test_active_sensing_writer_emits_fixture_artifact(tmp_path) -> None:
    config_payload = yaml.safe_load(
        (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    )
    config_payload["output_dir"] = str(tmp_path / "run")
    config_payload["fractalrabbit"]["fixture_csv"] = str(
        ROOT / "data" / "fixtures" / "fractalrabbit_sample.csv"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8"
    )

    out_path = tmp_path / "active_sensing.json"
    written = run_and_write_active_sensing(
        config_path,
        tmp_path / "missing-external" / "observations.csv",
        out_path,
        fix_fractions=(0.5,),
    )
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert written == out_path
    assert payload["schema"] == "active-fractal-rabbit-active-sensing-v1"
    assert payload["caveat"].endswith(
        "Synthetic-data software measurements, not empirical mobility validation."
    )
    assert [lane["label"] for lane in payload["lanes"]] == ["fixture"]
    assert payload["lanes"][0]["n_steps"] == 16
    assert len(payload["lanes"][0]["budgets"]) == 1
    assert payload["lanes"][0]["budgets"][0]["fix_fraction"] == pytest.approx(0.5)
    assert payload["lanes"][0]["budgets"][0]["budget"] > 0
    assert payload["epistemic_share_sweep"][0]["preference_strength"] == pytest.approx(
        0.0
    )
    assert payload["epistemic_share_sweep"][0]["mean_epistemic_share"] == pytest.approx(
        1.0
    )
