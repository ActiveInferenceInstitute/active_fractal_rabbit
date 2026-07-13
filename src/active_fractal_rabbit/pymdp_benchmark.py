"""pymdp performance and interpretability benchmark matrix."""

from __future__ import annotations

import json
import importlib.metadata as metadata
import inspect
from pathlib import Path

PYMDP_BENCHMARK_SCHEMA = "active-fractal-rabbit-pymdp-benchmark-matrix-v1"


def build_pymdp_benchmark_matrix(project_root: str | Path) -> dict[str, object]:
    """Build a manuscript-facing matrix from current run artifacts."""

    root = Path(project_root)
    metrics = _read_json(root / "output" / "runs" / "default" / "metrics.json", {})
    hardened = _read_json(
        root / "output" / "runs" / "hardened_anomaly_analysis.json", {}
    )
    sequence = _read_json(
        root / "output" / "runs" / "sequence_baseline_analysis.json", {}
    )
    model_gate = _read_json(root / "output" / "runs" / "model_gate_matrix.json", {})
    return_phase = _read_json(
        root / "output" / "runs" / "return_phase_prior_analysis.json", {}
    )
    factorial = _read_json(
        root / "output" / "runs" / "factorial_pymdp_analysis_fixture.json", {}
    )
    reference_loss = _number(metrics, "mean_predictive_loss")
    pymdp_loss = _number(metrics, "pymdp_mean_predictive_loss")
    reference_entropy = _number(metrics, "mean_posterior_entropy")
    pymdp_entropy = _number(metrics, "pymdp_mean_posterior_entropy")
    loss_delta = _delta(pymdp_loss, reference_loss)
    entropy_delta = _delta(pymdp_entropy, reference_entropy)
    provenance = _pymdp_provenance()
    source_observation_rows = _number(metrics, "pymdp_source_observation_rows")
    inference_rows = _number(metrics, "pymdp_inference_rows")
    validation_fraction = _number(metrics, "pymdp_validation_fraction")
    rows = [
        {
            "model": "reference_filter",
            "implemented": True,
            "current_role": "transparent Bayes-style performance baseline",
            "mean_predictive_loss": reference_loss,
            "mean_posterior_entropy": reference_entropy,
            "interpretability_surface": "posterior trace, tensor provenance, calibration",
        },
        {
            "model": "pymdp_agent",
            "implemented": True,
            "current_role": "variational active-inference implementation and policy diagnostic",
            "mean_predictive_loss": pymdp_loss,
            "mean_posterior_entropy": pymdp_entropy,
            "exact_inference_delta": _number(
                metrics, "pymdp_exact_inference_max_abs_delta"
            ),
            "source_observation_rows": source_observation_rows,
            "inference_rows": inference_rows,
            "validation_fraction": validation_fraction,
            "inferactively_pymdp_version": provenance["inferactively_pymdp_version"],
            "pymdp_agent_module": provenance["pymdp_agent_module"],
            "pymdp_package_path": provenance["pymdp_package_path"],
            "interpretability_surface": "VFE, policy posterior, expected free energy, action-conditioned prior",
        },
        {
            "model": "factorial_pymdp_agent",
            "implemented": bool(factorial),
            "current_role": "multi-factor latent-mode active-inference diagnostic",
            "performance_gate": factorial.get("performance_gate")
            if isinstance(factorial, dict)
            else None,
            "mean_predictive_loss": _nested_number(
                factorial, ("metrics", "factorial_pymdp_mean_predictive_loss")
            ),
            "mean_posterior_entropy": _nested_number(
                factorial, ("metrics", "factorial_pymdp_mean_posterior_entropy")
            ),
            "matched_prior_max_abs_delta": _nested_number(
                factorial, ("metrics", "factorial_pymdp_matched_prior_max_abs_delta")
            ),
            "mean_field_exact_gap": _nested_number(
                factorial, ("metrics", "factorial_pymdp_mean_field_exact_gap")
            ),
            "interpretability_surface": "cell, movement-mode, and cognitive-proxy posteriors plus VFE/EFE per step",
        },
        {
            "model": "persistence_and_base_rate_nulls",
            "implemented": True,
            "current_role": "honesty floor for autocorrelated waypoint recovery",
            "persistence_brier": _number(
                metrics, "reference_calibration_persistence_brier"
            ),
            "base_rate_brier": _number(
                metrics, "reference_calibration_base_rate_brier"
            ),
            "interpretability_surface": "explicit null hierarchy",
        },
        {
            "model": "markov_surprise_baseline",
            "implemented": True,
            "current_role": "transition-aware anomaly comparator",
            "speed_shift_auc": _nested_number(
                hardened, ("speed_regime_shift", "markov_auc")
            ),
            "preferred_drift_auc": _nested_number(
                hardened, ("preferred_set_drift", "markov_auc")
            ),
            "interpretability_surface": "online transition surprise",
        },
        {
            "model": "prequential_sequence_baselines",
            "implemented": bool(sequence),
            "current_role": "stronger next-cell predictive-performance gate",
            "best_model": sequence.get("best_model")
            if isinstance(sequence, dict)
            else None,
            "best_mean_predictive_loss": _number(sequence, "best_mean_predictive_loss")
            if isinstance(sequence, dict)
            else None,
            "best_minus_pymdp_predictive_loss": _number(
                sequence, "best_minus_pymdp_predictive_loss"
            )
            if isinstance(sequence, dict)
            else None,
            "interpretability_surface": "named online nulls and per-baseline loss/skill rows",
        },
        {
            "model": "lane_resolved_model_gate_matrix",
            "implemented": bool(model_gate),
            "current_role": "fixture/external evidence gate for model claims",
            "matrix_status": model_gate.get("status")
            if isinstance(model_gate, dict)
            else None,
            "external_sequence_gate": model_gate.get("external_sequence_gate")
            if isinstance(model_gate, dict)
            else None,
            "external_return_phase_verdict": model_gate.get(
                "external_return_phase_verdict"
            )
            if isinstance(model_gate, dict)
            else None,
            "interpretability_surface": "lane status, unavailable handling, and per-gate verdict rows",
        },
        {
            "model": "return_plus_phase_prior",
            "implemented": bool(return_phase),
            "current_role": "first promoted-or-null prior extension after return and phase gates",
            "verdict": return_phase.get("verdict")
            if isinstance(return_phase, dict)
            else None,
            "best_mean_predictive_loss": _number(
                return_phase, "best_mean_predictive_loss"
            )
            if isinstance(return_phase, dict)
            else None,
            "gain_vs_persistence": _number(
                return_phase, "return_plus_phase_gain_vs_persistence"
            )
            if isinstance(return_phase, dict)
            else None,
            "interpretability_surface": "pooled, phase, return, combined, persistence, and permuted comparator rows",
        },
        {
            "model": "future_sota_baselines",
            "implemented": False,
            "current_role": "required before any state-of-the-art performance claim",
            "candidate_families": "HMM, particle/state-space, neural sequence, learned hierarchical active inference",
            "interpretability_surface": "to be compared under the same null hierarchy",
        },
    ]
    return {
        "schema": PYMDP_BENCHMARK_SCHEMA,
        "state_of_art_performance_status": "not_demonstrated",
        "interpretability_status": "diagnostic_gain_demonstrated",
        "pymdp_minus_reference_predictive_loss": loss_delta,
        "pymdp_minus_reference_posterior_entropy": entropy_delta,
        "inferactively_pymdp_version": provenance["inferactively_pymdp_version"],
        "pymdp_agent_module": provenance["pymdp_agent_module"],
        "pymdp_package_path": provenance["pymdp_package_path"],
        "source_observation_rows": source_observation_rows,
        "inference_rows": inference_rows,
        "validation_fraction": validation_fraction,
        "rows": rows,
        "roadmap_gates": [
            "use the lane-resolved model gate matrix before performance-leading language",
            "learn hierarchical or continuous-space dynamics instead of fixed cells only",
            "calibrate missingness and reporting processes explicitly",
            "validate on consented empirical traces before real-world claims",
        ],
        "active_inference_benefits": [
            "inspectable variational free energy per step",
            "policy posterior and expected-free-energy diagnostics",
            "uncertainty-aware active-sensing experiments",
            "oversight-friendly minimization and null-regime reporting",
        ],
        "caveat": (
            "Current pymdp results validate the variational implementation and add interpretability surfaces. "
            "They do not establish state-of-the-art predictive performance beyond the transparent reference filter."
        ),
    }


def write_pymdp_benchmark_matrix(
    project_root: str | Path, out_path: str | Path
) -> Path:
    """Write the benchmark matrix artifact."""

    payload = build_pymdp_benchmark_matrix(project_root)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _pymdp_provenance() -> dict[str, str]:
    import pymdp
    from pymdp.agent import Agent

    return {
        "inferactively_pymdp_version": metadata.version("inferactively-pymdp"),
        "pymdp_agent_module": str(inspect.getmodule(Agent).__file__),
        "pymdp_package_path": str(Path(pymdp.__file__).resolve()),
    }


def _number(mapping: dict[str, object], key: str) -> float | None:
    value = mapping.get(key)
    return float(value) if isinstance(value, int | float) else None


def _nested_number(mapping: dict[str, object], path: tuple[str, ...]) -> float | None:
    current: object = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return float(current) if isinstance(current, int | float) else None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right
