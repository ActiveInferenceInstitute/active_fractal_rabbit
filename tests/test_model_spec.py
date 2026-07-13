"""The generative-model specification conforms, and corruption is caught."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.factorial_pymdp import build_factorial_pomdp
from active_fractal_rabbit.fractalrabbit import parse_waypoint_csv
from active_fractal_rabbit.model_spec import (
    MODEL_SPEC_SCHEMA,
    check_conformance,
    factorial_spec,
    model_card_markdown,
    run_and_write_model_conformance,
    single_factor_spec,
)
from active_fractal_rabbit.observations import discretize_observations
from active_fractal_rabbit.pomdp import build_pomdp

CONFIG = "configs/default.yaml"


def _sequence():
    config = load_config(CONFIG)
    sequence = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    return sequence, config


def _sizes(sequence):
    return dict(
        num_spatial_states=sequence.num_spatial_states,
        num_time_bins=sequence.num_time_bins,
        num_speed_categories=sequence.num_speed_categories,
        num_burst_categories=sequence.num_burst_categories,
    )


def test_single_and_factorial_models_conform_to_spec() -> None:
    sequence, config = _sequence()
    sizes = _sizes(sequence)
    single = build_pomdp(
        sequence, config.inference.model.smoothing, config.inference.model.controls
    )
    factorial = build_factorial_pomdp(sequence, config.inference.model)

    single_report = check_conformance(
        single,
        single_factor_spec(
            **sizes,
            observation_names=single.observation_names,
            n_controls=len(single.control_names),
        ),
    )
    factorial_report = check_conformance(factorial, factorial_spec(**sizes))

    assert single_report["conforms"] is True
    assert single_report["n_failed"] == 0
    assert single_report["conditional_independence"] is True
    assert factorial_report["conforms"] is True
    assert factorial_report["n_failed"] == 0
    assert factorial_report["conditional_independence"] is False  # coupled factors
    assert "training rows only" in single_report["estimation_contract"]
    assert "inner-validation" in single_report["tuning_contract"]
    assert "diagnostic" in factorial_report["estimation_contract"]


def test_conformance_catches_a_denormalized_emission_column() -> None:
    """Non-vacuity: corruption of a generative tensor must be caught."""

    sequence, config = _sequence()
    sizes = _sizes(sequence)
    single = build_pomdp(
        sequence, config.inference.model.smoothing, config.inference.model.controls
    )
    spec = single_factor_spec(
        **sizes,
        observation_names=single.observation_names,
        n_controls=len(single.control_names),
    )

    bad_emissions = list(single.A)
    corrupted = np.array(bad_emissions[0])
    corrupted[:, 0] *= 2.0  # break column-stochasticity
    bad_model = dataclasses.replace(
        single, A=tuple(bad_emissions[:0]) + (corrupted,) + tuple(bad_emissions[1:])
    )

    report = check_conformance(bad_model, spec)
    assert report["conforms"] is False
    assert any(not c["passed"] and "stochastic" in c["check"] for c in report["checks"])


def test_conformance_catches_a_wrong_dependency() -> None:
    """A factorial model whose dependency structure drifts from the spec is caught."""

    sequence, config = _sequence()
    factorial = build_factorial_pomdp(sequence, config.inference.model)
    wrong_spec = dataclasses.replace(
        factorial_spec(**_sizes(sequence)),
        a_dependencies=(
            (0,),
            (0,),
            (1, 0),
            (2,),
            (1,),
            (1,),
        ),  # second modality dependency altered
    )
    report = check_conformance(factorial, wrong_spec)
    assert report["conforms"] is False
    assert any(
        not c["passed"] and c["check"] == "a_dependencies" for c in report["checks"]
    )


def test_model_card_and_artifact_render(tmp_path: Path) -> None:
    sequence, config = _sequence()
    sizes = _sizes(sequence)
    single_card = model_card_markdown(single_factor_spec(**sizes))
    factorial_card = model_card_markdown(factorial_spec(**sizes))
    assert (
        "Generative model" in factorial_card
        and "cognitive_proxy_mode" in factorial_card
    )
    assert "coupled factors" in factorial_card
    assert "Estimation contract" in single_card
    assert "Tuning contract" in factorial_card
    assert "| `location` | 16 | cell |" in single_card
    assert "| `location` | 16 | cell |" in factorial_card

    docs_card = Path("docs/generative_model_card.md").read_text(encoding="utf-8")
    assert "| `location` | 16 | cell |" in docs_card
    assert "| `location` | 9 | cell |" not in docs_card

    out = run_and_write_model_conformance(
        CONFIG, tmp_path / "model_conformance.json"
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == MODEL_SPEC_SCHEMA
    assert payload["status"] == "pass"
    assert payload["conforms"] is True
    assert payload["single_factor"]["conforms"] is True
    assert payload["factorial"]["conforms"] is True


def test_conformance_fails_closed_on_out_of_range_dependency() -> None:
    """A corrupt spec with an out-of-range factor index must FAIL CLOSED with a
    reported check, never raise (cross-vendor fail-closed hardening)."""

    sequence, config = _sequence()
    factorial = build_factorial_pomdp(sequence, config.inference.model)
    corrupt_spec = dataclasses.replace(
        factorial_spec(**_sizes(sequence)),
        a_dependencies=(
            (0,),
            (0, 9),
            (1, 0),
            (2,),
            (1,),
            (1,),
        ),  # factor index 9 does not exist
    )
    report = check_conformance(factorial, corrupt_spec)  # must not raise
    assert report["conforms"] is False
    assert any(
        not c["passed"] and c["check"] == "A[1]_dependency_range"
        for c in report["checks"]
    )

    corrupt_b_spec = dataclasses.replace(
        factorial_spec(**_sizes(sequence)),
        b_dependencies=((0, 1), (1, 9), (2,)),
    )
    b_report = check_conformance(factorial, corrupt_b_spec)
    assert b_report["conforms"] is False
    assert any(
        not c["passed"] and c["check"] == "B[1]_dependency_range"
        for c in b_report["checks"]
    )


def test_conformance_fails_closed_on_malformed_spec_counts() -> None:
    sequence, config = _sequence()
    factorial = build_factorial_pomdp(sequence, config.inference.model)
    malformed = dataclasses.replace(
        factorial_spec(**_sizes(sequence)),
        observation_sizes=(sequence.num_spatial_states,),
        b_dependencies=((0, 1),),
        control_sizes=(),
    )
    report = check_conformance(factorial, malformed)
    failed = {item["check"] for item in report["checks"] if not item["passed"]}
    assert report["conforms"] is False
    assert {
        "spec_observation_cardinality_count",
        "spec_B_dependency_count",
        "spec_control_count",
    } <= failed


def test_conformance_rejects_negative_nan_and_wrong_preference_shape() -> None:
    """Negative/NaN probabilities and drifted C shapes must fail certification."""

    sequence, config = _sequence()
    single = build_pomdp(
        sequence, config.inference.model.smoothing, config.inference.model.controls
    )
    spec = single_factor_spec(
        **_sizes(sequence),
        observation_names=single.observation_names,
        n_controls=len(single.control_names),
    )
    bad_a = [np.array(item, copy=True) for item in single.A]
    bad_a[0][0, 0] = -0.1
    bad_a[0][1, 0] += 0.1
    bad_c = list(single.C)
    bad_c[0] = np.append(bad_c[0], np.nan)
    report = check_conformance(
        dataclasses.replace(single, A=tuple(bad_a), C=tuple(bad_c)), spec
    )
    failed = {c["check"] for c in report["checks"] if not c["passed"]}
    assert report["conforms"] is False
    assert {"A[0]_stochastic", "C[0]_shape_finite"} <= failed


def test_factorial_conformance_rejects_wrong_transition_shape() -> None:
    sequence, config = _sequence()
    factorial = build_factorial_pomdp(sequence, config.inference.model)
    bad_b = list(factorial.B)
    bad_b[0] = bad_b[0][..., :1]
    report = check_conformance(
        dataclasses.replace(factorial, B=tuple(bad_b)),
        factorial_spec(**_sizes(sequence)),
    )
    assert report["conforms"] is False
    assert any(
        not c["passed"] and c["check"] == "B[0]_shape"
        for c in report["checks"]
    )


def test_conformance_rejects_missing_tensors_and_control_drift() -> None:
    """A coordinated or shortened model must not certify against itself."""

    sequence, config = _sequence()
    single = build_pomdp(
        sequence, config.inference.model.smoothing, config.inference.model.controls
    )
    single_specification = single_factor_spec(
        **_sizes(sequence),
        observation_names=single.observation_names,
        n_controls=len(single.control_names),
    )
    missing_c = check_conformance(
        dataclasses.replace(single, C=single.C[:-1]), single_specification
    )
    drifted_controls = check_conformance(
        dataclasses.replace(
            single,
            B=single.B[:, :, :1],
            control_names=single.control_names[:1],
        ),
        single_specification,
    )
    assert missing_c["conforms"] is False
    assert any(
        not item["passed"] and item["check"] == "C_count"
        for item in missing_c["checks"]
    )
    assert drifted_controls["conforms"] is False
    assert {
        item["check"] for item in drifted_controls["checks"] if not item["passed"]
    } >= {"control_count", "B_shape"}

    factorial = build_factorial_pomdp(sequence, config.inference.model)
    factorial_report = check_conformance(
        dataclasses.replace(
            factorial,
            B=factorial.B[:-1],
            D=factorial.D[:-1],
            C=factorial.C[:-1],
        ),
        factorial_spec(**_sizes(sequence)),
    )
    assert factorial_report["conforms"] is False
    failed = {
        item["check"]
        for item in factorial_report["checks"]
        if not item["passed"]
    }
    assert {"B_count", "C_count", "D_count"} <= failed


@pytest.mark.parametrize(
    "factory",
    [
        lambda: single_factor_spec(
            num_spatial_states=0,
            num_time_bins=4,
            num_speed_categories=4,
            num_burst_categories=2,
        ),
        lambda: single_factor_spec(
            num_spatial_states=16,
            num_time_bins=4,
            num_speed_categories=4,
            num_burst_categories=2,
            n_controls=0,
        ),
        lambda: factorial_spec(
            num_spatial_states=16,
            num_time_bins=4,
            num_speed_categories=4,
            num_burst_categories=2,
            n_mode_controls=0,
        ),
    ],
)
def test_model_specs_reject_vacuous_cardinalities(factory) -> None:
    with pytest.raises(ValueError, match="cardinalities must be positive"):
        factory()
