"""Machine-checkable specification of the active-inference generative model.

This module makes the pymdp generative model an auditable artifact rather than an
implicit consequence of construction code. It states, for both the single-factor
reference model and the factorial latent-mode model, exactly what the generative
tensors must be — factor and observation names and cardinalities, the
``A``/``B``/``C``/``D`` shapes, the factor-dependency structure, the
column-stochastic normalization invariants, and the conditional-independence
approximation the likelihood uses — and then verifies that a built model conforms.

The conformance report is the methodology contribution: a reviewer can confirm
that the implemented active-inference model is the one the manuscript describes,
and a deliberately corrupted tensor (a de-normalized column, a wrong dependency,
a wrong cardinality) is caught rather than silently scored. Everything here is a
specification and a check; no mobility claim is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .pomdp import CANONICAL_MODALITIES
from .types import FactorialPomdpModel, PomdpModel

MODEL_SPEC_SCHEMA = "active-fractal-rabbit-generative-model-spec-v2"
_NORMALIZATION_TOL = 1e-9


@dataclass(frozen=True)
class GenerativeModelSpec:
    """Declarative contract for a discrete active-inference generative model."""

    name: str
    factor_names: tuple[str, ...]
    factor_sizes: tuple[int, ...]
    observation_names: tuple[str, ...]
    observation_sizes: tuple[int, ...]
    a_dependencies: tuple[tuple[int, ...], ...]
    b_dependencies: tuple[tuple[int, ...], ...]
    control_sizes: tuple[int, ...]
    conditional_independence: bool
    estimation_contract: str
    tuning_contract: str
    notes: str


def single_factor_spec(
    *,
    num_spatial_states: int,
    num_time_bins: int,
    num_speed_categories: int,
    num_burst_categories: int,
    observation_names: tuple[str, ...] = CANONICAL_MODALITIES,
    n_controls: int = 2,
) -> GenerativeModelSpec:
    """Spec for the single-latent-factor reference model.

    The single hidden factor is the discretized spatial cell. Every observation
    modality depends on that cell alone, so the joint likelihood factorizes as a
    product of per-modality emissions — the conditional-independence approximation.
    """

    cardinalities = (
        num_spatial_states,
        num_time_bins,
        num_speed_categories,
        num_burst_categories,
        n_controls,
    )
    if any(value < 1 for value in cardinalities):
        raise ValueError("all state, observation, and control cardinalities must be positive")

    sizes = {
        "location": num_spatial_states,
        "time_of_day": num_time_bins,
        "speed": num_speed_categories,
        "report_burst": num_burst_categories,
    }
    observation_sizes = tuple(sizes[name] for name in observation_names)
    return GenerativeModelSpec(
        name="single_factor_reference",
        factor_names=("cell",),
        factor_sizes=(num_spatial_states,),
        observation_names=observation_names,
        observation_sizes=observation_sizes,
        a_dependencies=tuple((0,) for _ in observation_names),
        b_dependencies=((0,),),
        control_sizes=(n_controls,),
        conditional_independence=True,
        estimation_contract=(
            "A/B/D estimated from declared training rows only; smoothing is explicit; "
            "evaluation-holdout rows do not update tensors"
        ),
        tuning_contract=(
            "precision/return selected after inner-train transition fitting; online "
            "adaptation selected in a second predeclared inner-validation stage"
        ),
        notes=(
            "Single hidden factor (spatial cell). Each A_m is column-stochastic of shape "
            "(obs_size, num_states); the joint likelihood P(o|s) = prod_m A_m[o_m, s] uses the "
            "conditional-independence approximation across modalities. B is (S, S, n_controls), "
            "each control slice column-stochastic; D is a length-S simplex."
        ),
    )


def factorial_spec(
    *,
    num_spatial_states: int,
    num_time_bins: int,
    num_speed_categories: int,
    num_burst_categories: int,
    n_cell_controls: int = 2,
    n_mode_controls: int = 2,
) -> GenerativeModelSpec:
    """Spec for the three-factor latent-mode model (cell, movement, cognitive proxy)."""

    cardinalities = (
        num_spatial_states,
        num_time_bins,
        num_speed_categories,
        num_burst_categories,
        n_cell_controls,
        n_mode_controls,
    )
    if any(value < 1 for value in cardinalities):
        raise ValueError("all state, observation, and control cardinalities must be positive")

    return GenerativeModelSpec(
        name="factorial_latent_mode",
        factor_names=("cell", "movement_mode", "cognitive_proxy_mode"),
        factor_sizes=(num_spatial_states, 3, 3),
        observation_names=(
            "location",
            "time_of_day",
            "speed",
            "report_burst",
            "cell_delta",
            "return_flag",
        ),
        observation_sizes=(
            num_spatial_states,
            num_time_bins,
            num_speed_categories,
            num_burst_categories,
            2,
            2,
        ),
        a_dependencies=((0,), (0, 2), (1, 0), (2,), (1,), (1,)),
        b_dependencies=((0, 1), (1, 2), (2,)),
        control_sizes=(n_cell_controls, n_mode_controls, n_mode_controls),
        conditional_independence=False,
        estimation_contract=(
            "factorial A/B/D tensors estimated from declared included rows; the current "
            "factorial surface is diagnostic, not a locked predictive candidate"
        ),
        tuning_contract=(
            "no predictive hyperparameter promotion; tensor and mean-field diagnostics only"
        ),
        notes=(
            "Three hidden factors. A_dependencies couple factors (e.g. time_of_day depends on cell and "
            "cognitive-proxy), so the exact posterior is non-factorized and mean-field inference is an "
            "approximation reported with its gap. Each A_m is column-stochastic over its observation axis; "
            "each B factor tensor is column-stochastic over its target axis; each D factor is a simplex. "
            "The cognitive-proxy factor is a synthetic reporting-process state, not real cognition or intent."
        ),
    )


def _check(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def _columns_stochastic(tensor: np.ndarray) -> bool:
    tensor = np.asarray(tensor, dtype=float)
    if not np.all(np.isfinite(tensor)) or np.any(tensor < 0.0):
        return False
    totals = tensor.sum(axis=0)
    return bool(np.all(np.abs(totals - 1.0) < _NORMALIZATION_TOL))


def _simplex(vector: np.ndarray) -> bool:
    vector = np.asarray(vector, dtype=float)
    return bool(
        np.all(np.isfinite(vector))
        and np.all(vector >= 0.0)
        and abs(float(vector.sum()) - 1.0) < _NORMALIZATION_TOL
    )


def _finite_vector(vector: np.ndarray, expected_size: int) -> bool:
    vector = np.asarray(vector, dtype=float)
    return bool(vector.shape == (expected_size,) and np.all(np.isfinite(vector)))


def check_conformance(
    model: PomdpModel | FactorialPomdpModel, spec: GenerativeModelSpec
) -> dict[str, object]:
    """Verify a built model conforms to its declared specification.

    Returns a structured report whose ``conforms`` flag is the conjunction of all
    checks. Handles both the single-factor ``PomdpModel`` (one ``B`` array,
    implicit single-factor dependencies) and the factorial model (per-factor ``B``
    tuple, explicit dependencies).
    """

    checks: list[dict[str, object]] = []
    is_factorial = isinstance(model, FactorialPomdpModel)

    checks.extend(
        [
            _check(
                "spec_factor_cardinality_count",
                len(spec.factor_sizes) == len(spec.factor_names),
                f"{len(spec.factor_sizes)} sizes for {len(spec.factor_names)} factors",
            ),
            _check(
                "spec_observation_cardinality_count",
                len(spec.observation_sizes) == len(spec.observation_names),
                f"{len(spec.observation_sizes)} sizes for {len(spec.observation_names)} modalities",
            ),
            _check(
                "spec_A_dependency_count",
                len(spec.a_dependencies) == len(spec.observation_names),
                f"{len(spec.a_dependencies)} dependencies for {len(spec.observation_names)} modalities",
            ),
            _check(
                "spec_B_dependency_count",
                len(spec.b_dependencies) == len(spec.factor_names),
                f"{len(spec.b_dependencies)} dependencies for {len(spec.factor_names)} factors",
            ),
            _check(
                "spec_control_count",
                len(spec.control_sizes) == len(spec.factor_names),
                f"{len(spec.control_sizes)} control sizes for {len(spec.factor_names)} factors",
            ),
        ]
    )

    checks.append(
        _check(
            "observation_names",
            tuple(model.observation_names) == spec.observation_names,
            f"{tuple(model.observation_names)} vs {spec.observation_names}",
        )
    )
    checks.append(
        _check(
            "emission_count",
            len(model.A) == len(spec.observation_names),
            f"{len(model.A)} emissions for {len(spec.observation_names)} modalities",
        )
    )

    if is_factorial:
        checks.append(
            _check(
                "B_count",
                len(model.B) == len(spec.factor_names),
                f"{len(model.B)} transitions for {len(spec.factor_names)} factors",
            )
        )
        checks.append(
            _check(
                "C_count",
                len(model.C) == len(spec.observation_names),
                f"{len(model.C)} preferences for {len(spec.observation_names)} modalities",
            )
        )
        checks.append(
            _check(
                "D_count",
                len(model.D) == len(spec.factor_names),
                f"{len(model.D)} priors for {len(spec.factor_names)} factors",
            )
        )
        checks.append(
            _check(
                "control_factor_count",
                len(model.control_names) == len(spec.control_sizes),
                f"{len(model.control_names)} control sets vs {len(spec.control_sizes)}",
            )
        )
        for idx, expected_size in enumerate(spec.control_sizes):
            actual = (
                len(model.control_names[idx]) if idx < len(model.control_names) else 0
            )
            checks.append(
                _check(
                    f"control[{idx}]_count",
                    actual == expected_size,
                    f"{actual} vs {expected_size}",
                )
            )
        checks.append(
            _check(
                "factor_names",
                tuple(model.factor_names) == spec.factor_names,
                f"{tuple(model.factor_names)} vs {spec.factor_names}",
            )
        )
        checks.append(
            _check(
                "factor_sizes",
                tuple(model.factor_sizes) == spec.factor_sizes,
                f"{tuple(model.factor_sizes)} vs {spec.factor_sizes}",
            )
        )
        checks.append(
            _check(
                "a_dependencies",
                tuple(tuple(d) for d in model.A_dependencies) == spec.a_dependencies,
                f"{tuple(tuple(d) for d in model.A_dependencies)} vs {spec.a_dependencies}",
            )
        )
        checks.append(
            _check(
                "b_dependencies",
                tuple(tuple(d) for d in model.B_dependencies) == spec.b_dependencies,
                f"{tuple(tuple(d) for d in model.B_dependencies)} vs {spec.b_dependencies}",
            )
        )
        n_factors = len(spec.factor_sizes)
        for idx, (matrix, deps, obs_size) in enumerate(
            zip(
                model.A[: len(spec.observation_names)],
                spec.a_dependencies[: len(spec.observation_names)],
                spec.observation_sizes[: len(spec.observation_names)],
            )
        ):
            if any(d >= n_factors or d < 0 for d in deps):
                # A drifted/corrupt spec — exactly what this validator exists to catch.
                # Fail closed with a reported check rather than raising on the index.
                checks.append(
                    _check(
                        f"A[{idx}]_dependency_range",
                        False,
                        f"dependency {deps} out of range for {n_factors} factors",
                    )
                )
                continue
            expected = (obs_size,) + tuple(spec.factor_sizes[d] for d in deps)
            checks.append(
                _check(
                    f"A[{idx}]_shape",
                    tuple(matrix.shape) == expected,
                    f"{tuple(matrix.shape)} vs {expected}",
                )
            )
            checks.append(
                _check(
                    f"A[{idx}]_stochastic",
                    _columns_stochastic(np.asarray(matrix)),
                    "columns sum to 1",
                )
            )
        safe_b_count = min(
            len(model.B),
            len(spec.factor_names),
            len(spec.b_dependencies),
            len(spec.control_sizes),
            len(spec.factor_sizes),
        )
        for idx, tensor in enumerate(model.B[:safe_b_count]):
            deps = spec.b_dependencies[idx]
            if any(d >= n_factors or d < 0 for d in deps):
                checks.append(
                    _check(
                        f"B[{idx}]_dependency_range",
                        False,
                        f"dependency {deps} out of range for {n_factors} factors",
                    )
                )
                continue
            expected = (spec.factor_sizes[idx],) + tuple(
                spec.factor_sizes[d] for d in deps
            ) + (spec.control_sizes[idx],)
            checks.append(
                _check(
                    f"B[{idx}]_shape",
                    tuple(tensor.shape) == expected,
                    f"{tuple(tensor.shape)} vs {expected}",
                )
            )
            checks.append(
                _check(
                    f"B[{idx}]_stochastic",
                    _columns_stochastic(np.asarray(tensor)),
                    "columns sum to 1",
                )
            )
        for idx, prior in enumerate(
            model.D[: min(len(spec.factor_names), len(spec.factor_sizes))]
        ):
            checks.append(
                _check(
                    f"D[{idx}]_simplex",
                    np.asarray(prior).shape == (spec.factor_sizes[idx],)
                    and _simplex(np.asarray(prior)),
                    "has factor cardinality, finite nonnegative entries, and sums to 1",
                )
            )
        for idx, preference in enumerate(
            model.C[
                : min(len(spec.observation_names), len(spec.observation_sizes))
            ]
        ):
            checks.append(
                _check(
                    f"C[{idx}]_shape_finite",
                    _finite_vector(preference, spec.observation_sizes[idx]),
                    f"{np.asarray(preference).shape} vs {(spec.observation_sizes[idx],)}; finite",
                )
            )
    else:
        n_states = int(model.D.size)
        expected_control_size = spec.control_sizes[0] if spec.control_sizes else 0
        checks.append(
            _check(
                "C_count",
                len(model.C) == len(spec.observation_names),
                f"{len(model.C)} preferences for {len(spec.observation_names)} modalities",
            )
        )
        checks.append(
            _check(
                "control_count",
                len(spec.control_sizes) == 1
                and len(model.control_names) == expected_control_size,
                f"{len(model.control_names)} vs {expected_control_size}",
            )
        )
        checks.append(
            _check(
                "factor_sizes",
                (n_states,) == spec.factor_sizes,
                f"{(n_states,)} vs {spec.factor_sizes}",
            )
        )
        checks.append(
            _check(
                "conditional_independence",
                spec.conditional_independence,
                "single-factor likelihood is a product of per-modality emissions",
            )
        )
        for idx, (matrix, obs_size) in enumerate(
            zip(
                model.A[: len(spec.observation_names)],
                spec.observation_sizes[: len(spec.observation_names)],
            )
        ):
            checks.append(
                _check(
                    f"A[{idx}]_shape",
                    tuple(matrix.shape) == (obs_size, n_states),
                    f"{tuple(matrix.shape)} vs {(obs_size, n_states)}",
                )
            )
            checks.append(
                _check(
                    f"A[{idx}]_stochastic",
                    _columns_stochastic(np.asarray(matrix)),
                    "columns sum to 1",
                )
            )
        b_array = np.asarray(model.B)
        checks.append(
            _check(
                "B_shape",
                b_array.shape == (n_states, n_states, expected_control_size),
                f"{b_array.shape} vs {(n_states, n_states, expected_control_size)}",
            )
        )
        controls_ok = bool(
            b_array.ndim == 3
            and all(
                _columns_stochastic(b_array[:, :, c])
                for c in range(b_array.shape[2])
            )
        )
        checks.append(
            _check("B_stochastic", controls_ok, "each control slice column-stochastic")
        )
        checks.append(
            _check(
                "D_simplex",
                np.asarray(model.D).shape == (n_states,) and _simplex(model.D),
                "has state cardinality, finite nonnegative entries, and sums to 1",
            )
        )
        for idx, preference in enumerate(
            model.C[
                : min(len(spec.observation_names), len(spec.observation_sizes))
            ]
        ):
            checks.append(
                _check(
                    f"C[{idx}]_shape_finite",
                    _finite_vector(preference, spec.observation_sizes[idx]),
                    f"{np.asarray(preference).shape} vs {(spec.observation_sizes[idx],)}; finite",
                )
            )

    conforms = all(item["passed"] for item in checks)
    return {
        "schema": MODEL_SPEC_SCHEMA,
        "spec_name": spec.name,
        "conforms": conforms,
        "conditional_independence": spec.conditional_independence,
        "estimation_contract": spec.estimation_contract,
        "tuning_contract": spec.tuning_contract,
        "control_sizes": spec.control_sizes,
        "n_checks": len(checks),
        "n_failed": sum(1 for item in checks if not item["passed"]),
        "checks": checks,
        "notes": spec.notes,
    }


def model_card_markdown(spec: GenerativeModelSpec) -> str:
    """Render a human-readable model-card table for the manuscript/docs."""

    lines = [
        f"### Generative model: `{spec.name}`",
        "",
        f"- **Hidden factors:** {', '.join(f'`{n}`' for n in spec.factor_names)}",
        f"- **Observation modalities:** {', '.join(f'`{n}`' for n in spec.observation_names)}",
        f"- **A (emission) dependencies:** {spec.a_dependencies}",
        f"- **B (transition) dependencies:** {spec.b_dependencies}",
        f"- **Control cardinalities:** {spec.control_sizes}",
        f"- **Conditional-independence approximation:** {'yes' if spec.conditional_independence else 'no (coupled factors)'}",
        f"- **Estimation contract:** {spec.estimation_contract}",
        f"- **Tuning contract:** {spec.tuning_contract}",
        "",
        "| modality | cardinality | depends on factors |",
        "| --- | --- | --- |",
    ]
    for name, size, deps in zip(
        spec.observation_names, spec.observation_sizes, spec.a_dependencies
    ):
        factors = ", ".join(spec.factor_names[d] for d in deps)
        lines.append(f"| `{name}` | {size} | {factors} |")
    lines.append("")
    lines.append(spec.notes)
    return "\n".join(lines)


def run_and_write_model_conformance(
    config_path: str | Path, out_path: str | Path
) -> Path:
    """Build both models from a config and write their conformance report."""

    from .config import load_config
    from .factorial_pymdp import build_factorial_pomdp
    from .fractalrabbit import parse_waypoint_csv
    from .observations import discretize_observations
    from .pomdp import build_pomdp

    config = load_config(config_path)
    assert config.fractalrabbit.fixture_csv is not None
    sequence = discretize_observations(
        parse_waypoint_csv(config.fractalrabbit.fixture_csv), config.inference
    )
    sizes = dict(
        num_spatial_states=sequence.num_spatial_states,
        num_time_bins=sequence.num_time_bins,
        num_speed_categories=sequence.num_speed_categories,
        num_burst_categories=sequence.num_burst_categories,
    )
    single = build_pomdp(
        sequence, config.inference.model.smoothing, config.inference.model.controls
    )
    factorial = build_factorial_pomdp(sequence, config.inference.model)
    single_spec = single_factor_spec(
        **sizes,
        observation_names=single.observation_names,
        n_controls=len(single.control_names),
    )
    factorial_spec_obj = factorial_spec(**sizes)
    single_report = check_conformance(single, single_spec)
    factorial_report = check_conformance(factorial, factorial_spec_obj)
    conforms = bool(single_report["conforms"] and factorial_report["conforms"])
    payload = {
        "schema": MODEL_SPEC_SCHEMA,
        "status": "pass" if conforms else "fail",
        "conforms": conforms,
        "single_factor": single_report,
        "factorial": factorial_report,
        # Human-readable model-card tables (model_card_markdown) alongside the
        # machine-checked conformance verdicts above, so a reviewer confirming
        # "is the implemented model the one the manuscript describes" has both
        # the pass/fail check and the readable factor/modality/dependency table
        # in the one artifact this script writes, not just an untested render
        # path only exercised by its own unit test.
        "model_cards": {
            "single_factor": model_card_markdown(single_spec),
            "factorial": model_card_markdown(factorial_spec_obj),
        },
    }
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not conforms:  # pragma: no cover - built-in models are expected to conform
        raise RuntimeError(f"generative-model conformance failed; see {output}")
    return output
