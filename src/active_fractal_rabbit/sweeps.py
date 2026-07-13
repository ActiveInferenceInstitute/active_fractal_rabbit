"""Small fixture-based sensitivity sweeps for manuscript diagnostics."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .config import load_config
from .pipeline import run_pipeline
from .types import PipelineResult, ProjectConfig

SUMMARY_METRICS = (
    "mean_predictive_loss",
    "mean_posterior_entropy",
    "mean_policy_entropy",
    "baseline_delta_mean_predictive_loss",
    "baseline_delta_mean_posterior_entropy",
)


def run_replicated_sensitivity(config_path: str | Path) -> dict[str, Path]:
    """Run deterministic fixture variants and write aggregate summary artifacts."""

    base = load_config(config_path)
    results: list[tuple[dict[str, str], PipelineResult]] = []
    for factors, config in _variant_configs(base):
        results.append((factors, run_pipeline(config, config_path)))
    payload = _summary_payload(results, Path(config_path), base.fractalrabbit.execution)
    suffix = (
        ""
        if base.fractalrabbit.execution == "fixture"
        else f"_{base.fractalrabbit.execution}"
    )
    json_path = (
        base.project_root / "output" / "runs" / f"sensitivity_summary{suffix}.json"
    )
    csv_path = (
        base.project_root / "output" / "runs" / f"sensitivity_summary{suffix}.csv"
    )
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_path.write_text(_summary_csv(payload), encoding="utf-8")
    return {"json": json_path, "csv": csv_path}


def _variant_configs(base: ProjectConfig) -> list[tuple[dict[str, str], ProjectConfig]]:
    variants: list[tuple[dict[str, str], ProjectConfig]] = []
    replicates = max(1, base.experiment.repetitions)
    for index in range(replicates):
        variants.append(
            (
                {"family": "replicate", "variant": f"default_{index + 1}"},
                _with_run(base, f"sensitivity_default_{index + 1}"),
            )
        )

    variants.append(
        (
            {"family": "report_count", "variant": "lower_reports"},
            _with_run(
                base,
                "sensitivity_lower_reports",
                fractalrabbit=replace(
                    base.fractalrabbit,
                    params=replace(
                        base.fractalrabbit.params,
                        number_of_reports=_report_count_variant(base, "lower"),
                    ),
                    output_prefix="sensitivity_lower_reports",
                ),
            ),
        )
    )
    variants.append(
        (
            {"family": "report_count", "variant": "higher_reports"},
            _with_run(
                base,
                "sensitivity_higher_reports",
                fractalrabbit=replace(
                    base.fractalrabbit,
                    params=replace(
                        base.fractalrabbit.params,
                        number_of_reports=_report_count_variant(base, "higher"),
                    ),
                    output_prefix="sensitivity_higher_reports",
                ),
            ),
        )
    )

    current_bins = int(base.inference.observation.spatial_bins)
    # Asymmetric step: -1 is clamped to a minimum of 2 bins to avoid a degenerate
    # near-unbinned lower variant, so the higher variant steps by +2 to keep the
    # lower/higher pair spaced roughly symmetrically around current_bins.
    lower_bins = max(2, current_bins - 1)
    higher_bins = current_bins + 2
    variants.append(
        (
            {"family": "spatial_bins", "variant": str(lower_bins)},
            _with_observation(
                base, f"sensitivity_spatial_bins_{lower_bins}", spatial_bins=lower_bins
            ),
        )
    )
    variants.append(
        (
            {"family": "spatial_bins", "variant": str(higher_bins)},
            _with_observation(
                base,
                f"sensitivity_spatial_bins_{higher_bins}",
                spatial_bins=higher_bins,
            ),
        )
    )
    variants.append(
        (
            {"family": "burst_rule", "variant": "fixed_days_0.5"},
            _with_observation(
                base,
                "sensitivity_fixed_burst",
                burst_threshold_rule="fixed_days",
                burst_fixed_days=0.5,
            ),
        )
    )
    variants.append(
        (
            {"family": "smoothing", "variant": "0.10"},
            _with_model(base, "sensitivity_smoothing_0_10", smoothing=0.10),
        )
    )
    variants.append(
        (
            {"family": "controls", "variant": "with_identity"},
            _with_model(
                base,
                "sensitivity_identity_control",
                controls=(
                    "empirical_continuation",
                    "high_entropy_exploration",
                    "identity",
                ),
            ),
        )
    )
    variants.append(
        (
            {"family": "latent_uncertainty", "variant": "no_location_modality"},
            _with_model(
                base,
                "sensitivity_no_location",
                modalities=("time_of_day", "speed", "report_burst"),
            ),
        )
    )
    return variants


def _with_observation(
    base: ProjectConfig, run_id: str, **updates: Any
) -> ProjectConfig:
    observation = replace(base.inference.observation, **updates)
    if "spatial_bins" in updates:
        observation = replace(observation, spatial_bins=int(updates["spatial_bins"]))
    inference = replace(base.inference, observation=observation)
    return _with_run(base, run_id, inference=inference)


def _report_count_variant(base: ProjectConfig, side: str) -> float:
    candidates = [
        float(candidate.params.number_of_reports)
        for candidate in base.closed_loop.candidates
        if float(candidate.params.number_of_reports) > 0
    ]
    current = float(base.fractalrabbit.params.number_of_reports)
    if candidates:
        return min(candidates) if side == "lower" else max(candidates)
    return current * (2.0 / 3.0 if side == "lower" else 2.0)


def _with_model(base: ProjectConfig, run_id: str, **updates: Any) -> ProjectConfig:
    model = replace(base.inference.model, **updates)
    inference = replace(base.inference, model=model)
    return _with_run(base, run_id, inference=inference)


def _with_run(
    base: ProjectConfig,
    run_id: str,
    *,
    inference=None,
    fractalrabbit=None,
) -> ProjectConfig:
    selected_inference = inference if inference is not None else base.inference
    # Sensitivity summaries compare deterministic perturbations of the
    # reference filter. The canonical/default run remains responsible for the
    # full pymdp trace so expanded fixtures do not multiply heavyweight policy
    # inference across every sweep variant.
    selected_inference = replace(
        selected_inference,
        model=replace(selected_inference.model, engine="reference"),
    )
    return replace(
        base,
        run_id=run_id,
        output_dir=base.project_root / "output" / "runs" / run_id,
        inference=selected_inference,
        fractalrabbit=fractalrabbit
        if fractalrabbit is not None
        else replace(base.fractalrabbit, output_prefix=run_id),
    )


def _summary_payload(
    results: list[tuple[dict[str, str], PipelineResult]],
    source_config: Path,
    execution: str = "fixture",
) -> dict[str, object]:
    metric_summary = {
        metric: _stats([float(result.metrics[metric]) for _, result in results])
        for metric in SUMMARY_METRICS
    }
    selected_counts: dict[str, int] = {}
    runs = []
    for factors, result in results:
        selected = str(result.selected_next_candidate or "none")
        selected_counts[selected] = selected_counts.get(selected, 0) + 1
        controls = tuple(result.metadata.get("controls", ()))
        control_count = len(controls)
        runs.append(
            {
                "run_id": result.run_id,
                "factors": factors,
                "factor_metadata": factors,
                "source_config": str(result.config_path),
                "engine": str(result.metadata.get("engine", "unknown")),
                "control_count": control_count,
                "max_policy_entropy": float(math.log(control_count))
                if control_count > 0
                else 0.0,
                "selected_next_candidate": selected,
                "metrics": {
                    metric: float(result.metrics[metric]) for metric in SUMMARY_METRICS
                },
            }
        )
    if execution == "external":
        caveat = (
            "This sensitivity lane resimulates waypoint rows with the pinned external FractalRabbit "
            "build for every variant, so replicate dispersion reflects genuine simulator stochasticity. "
            "It remains synthetic-data evidence, not empirical mobility validation. "
            "Sensitivity variants use the reference engine; the canonical lane carries the full pymdp trace."
        )
    else:
        caveat = (
            "This deterministic sensitivity lane reprocesses the checked fixture. "
            "Simulator parameter variants affect declared run metadata and candidate scoring, "
            "but they do not resimulate waypoint rows unless the external FractalRabbit lane is available. "
            "Fixture-lane replicates are bit-identical reruns: their spread is exactly zero by "
            "construction, so std and normal_approx bounds in this lane are determinism checks, "
            "never sampling uncertainty. Sensitivity variants use the reference engine; the canonical lane carries "
            "the full pymdp trace."
        )
    return {
        "schema": "active-fractal-rabbit-sensitivity-v1",
        "lane": execution,
        "source_config": str(source_config),
        "caveat": caveat,
        "deterministic_replicates": execution != "external",
        "n_runs": len(results),
        "runs": runs,
        "metric_summary": metric_summary,
        "selected_candidate_counts": selected_counts,
    }


def _stats(values: list[float]) -> dict[str, float]:
    n = len(values)
    avg = mean(values)
    spread = stdev(values) if n > 1 else 0.0
    half_width = 1.96 * spread / (n**0.5) if n > 1 else 0.0
    return {
        "n": float(n),
        "mean": float(avg),
        "std": float(spread),
        "min": float(min(values)),
        "max": float(max(values)),
        "normal_approx_low": float(avg - half_width),
        "normal_approx_high": float(avg + half_width),
    }


def _summary_csv(payload: dict[str, object]) -> str:
    summary = payload["metric_summary"]
    assert isinstance(summary, dict)
    rows = ["metric,n,mean,std,min,max,normal_approx_low,normal_approx_high"]
    for metric in SUMMARY_METRICS:
        stats = summary[metric]
        assert isinstance(stats, dict)
        rows.append(
            ",".join(
                [
                    metric,
                    f"{float(stats['n']):.0f}",
                    f"{float(stats['mean']):.12g}",
                    f"{float(stats['std']):.12g}",
                    f"{float(stats['min']):.12g}",
                    f"{float(stats['max']):.12g}",
                    f"{float(stats['normal_approx_low']):.12g}",
                    f"{float(stats['normal_approx_high']):.12g}",
                ]
            )
        )
    return "\n".join(rows) + "\n"
