"""End-to-end pipeline orchestration."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

import numpy as np

from .config import load_config
from .fractalrabbit import (
    parse_places_csv,
    parse_waypoint_csv,
    places_csv_for,
    resolve_observation_csv,
)
from .ground_truth import align_places, place_statistics, state_recovery
from .inference import (
    holdout_split,
    run_fixed_transition_baseline,
    run_holdout_scoring,
    run_latent_inference,
)
from .objectives import lane_objectives, write_calibration_artifact
from .observations import discretize_observations
from .pomdp import build_pomdp, tensor_summary
from .types import (
    BaselineResult,
    CandidateScore,
    InferenceResult,
    ObservationSequence,
    PipelineResult,
    ProjectConfig,
)


def run_pipeline(config: ProjectConfig, config_path: str | Path) -> PipelineResult:
    """Run one configured simulate/observe/infer/score pass."""

    run_dir = config.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    source_csv = resolve_observation_csv(config.fractalrabbit, run_dir)
    observations = parse_waypoint_csv(source_csv)
    observations_path = run_dir / "observations.csv"
    if source_csv.resolve() != observations_path.resolve():
        shutil.copyfile(source_csv, observations_path)

    sequence = discretize_observations(observations, config.inference)
    model = build_pomdp(
        sequence,
        config.inference.smoothing,
        config.inference.model.controls,
        modalities=config.inference.model.modalities,
    )
    inference = run_latent_inference(model, sequence)
    baseline = run_fixed_transition_baseline(model, sequence)
    pymdp_result = None
    if config.inference.model.engine == "both":
        from .pymdp_agent import run_pymdp_inference

        pymdp_result = run_pymdp_inference(
            model, sequence, reference_posterior=inference.posterior_states
        )
    candidate_scores = score_candidates(config, inference.metrics)
    selected = candidate_scores[0].name if candidate_scores else None
    baseline_numeric_metrics = {
        key: value
        for key, value in baseline.metrics.items()
        if isinstance(value, int | float)
    }
    holdout_metrics = _holdout_metrics(config, sequence)
    pymdp_metrics = pymdp_result.metrics if pymdp_result is not None else {}
    ground_truth_metrics, ground_truth_rows = _ground_truth_evaluation(
        source_csv, sequence, inference, pymdp_result
    )
    objective_metrics, objective_lanes = _objective_evaluation(
        model, sequence, inference, pymdp_result
    )
    metrics = (
        inference.metrics
        | baseline_numeric_metrics
        | holdout_metrics
        | pymdp_metrics
        | ground_truth_metrics
        | objective_metrics
        | {
            "baseline_delta_mean_predictive_loss": inference.metrics[
                "mean_predictive_loss"
            ]
            - float(baseline.metrics["baseline_mean_predictive_loss"]),
            "baseline_delta_mean_posterior_entropy": inference.metrics[
                "mean_posterior_entropy"
            ]
            - float(baseline.metrics["baseline_mean_posterior_entropy"]),
            "selected_next_candidate": selected,
        }
    )

    artifacts = {
        "observations": observations_path,
        "metrics": run_dir / "metrics.json",
        "baseline_metrics": run_dir / "baseline_metrics.json",
        "baseline_trace": run_dir / "baseline_trace.csv",
        "posterior": run_dir / "posterior_states.csv",
        "discrete_features": run_dir / "discrete_features.csv",
        "inference_trace": run_dir / "inference_trace.csv",
        "policy_trace": run_dir / "policy_trace.csv",
        "tensor_summary": run_dir / "tensor_summary.json",
        "candidate_scores": run_dir / "candidate_scores.json",
        "candidate_scores_csv": run_dir / "candidate_scores.csv",
        "config_snapshot": run_dir / "config_snapshot.json",
        "provenance_manifest": run_dir / "provenance_manifest.json",
    }
    _write_json(artifacts["metrics"], metrics)
    _write_json(artifacts["baseline_metrics"], baseline.metrics)
    _write_baseline_trace_csv(artifacts["baseline_trace"], baseline)
    _write_posterior_csv(artifacts["posterior"], inference.posterior_states)
    _write_discrete_features_csv(artifacts["discrete_features"], sequence)
    _write_inference_trace_csv(artifacts["inference_trace"], inference)
    _write_policy_trace_csv(
        artifacts["policy_trace"], inference, config.inference.model.controls
    )
    _write_json(artifacts["tensor_summary"], tensor_summary(model))
    _write_candidate_scores(
        artifacts["candidate_scores"],
        artifacts["candidate_scores_csv"],
        candidate_scores,
    )
    if pymdp_result is not None:
        artifacts["pymdp_trace"] = run_dir / "pymdp_trace.csv"
        _write_pymdp_trace_csv(
            artifacts["pymdp_trace"], pymdp_result, config.inference.model.controls
        )
    artifacts["ground_truth"] = run_dir / "ground_truth.csv"
    _write_ground_truth_csv(artifacts["ground_truth"], ground_truth_rows)
    artifacts["calibration"] = run_dir / "calibration.json"
    write_calibration_artifact(
        artifacts["calibration"], objective_lanes, objective_lanes[0]["floors"]
    )
    _write_json(artifacts["config_snapshot"], _jsonable(asdict(config)))
    run_metadata = {
        "fractalrabbit_commit": config.fractalrabbit.upstream_commit,
        "fractalrabbit_execution": config.fractalrabbit.execution,
        "observation_modalities": model.observation_names,
        "config_path": str(config_path),
        "spatial_discretizer": config.inference.observation.spatial_discretizer,
        "controls": config.inference.model.controls,
        "modalities": config.inference.model.modalities,
        "engine": config.inference.model.engine,
        "baseline_control": baseline.metrics["baseline_control"],
        "places_available": ground_truth_rows.get("place_ids") is not None,
    }
    _write_json(run_dir / "metadata.json", run_metadata)
    _write_json(
        artifacts["provenance_manifest"],
        {
            "run_id": config.run_id,
            "source_csv": str(source_csv),
            "config_path": str(config_path),
            "upstream_repo": config.fractalrabbit.upstream_repo,
            "upstream_commit": config.fractalrabbit.upstream_commit,
            "software": {
                "python": sys.version.split()[0],
                "fractalrabbit": {
                    "repo": config.fractalrabbit.upstream_repo,
                    "commit": config.fractalrabbit.upstream_commit,
                    "commit_url": (
                        "https://github.com/NationalSecurityAgency/fractalrabbit/tree/"
                        f"{config.fractalrabbit.upstream_commit}"
                    ),
                    "release_status": "no upstream GitHub releases were available when this manifest schema was added",
                },
                "inferactively-pymdp": {
                    "version": _package_version("inferactively-pymdp"),
                    "citation": "https://doi.org/10.21105/joss.04098",
                },
                "numpy": {"version": _package_version("numpy")},
                "matplotlib": {"version": _package_version("matplotlib")},
                "pyyaml": {"version": _package_version("pyyaml")},
            },
            "execution": config.fractalrabbit.execution,
            "artifacts": {key: str(path) for key, path in artifacts.items()},
        },
    )
    return PipelineResult(
        run_id=config.run_id,
        config_path=Path(config_path).resolve(),
        output_dir=run_dir,
        observations_path=observations_path,
        metrics=metrics,
        selected_next_candidate=selected,
        candidate_scores=candidate_scores,
        artifacts=artifacts,
        metadata=run_metadata,
    )


def run_pipeline_from_path(config_path: str | Path) -> PipelineResult:
    """Load a config path and run the pipeline."""

    config = load_config(config_path)
    return run_pipeline(config, config_path)


def score_candidates(
    config: ProjectConfig, metrics: dict[str, float]
) -> tuple[CandidateScore, ...]:
    """Score declared candidates using the measured objective and transparent report-density scaling."""

    if not config.closed_loop.candidates:
        return ()
    metric_name = _objective_metric(config.closed_loop.objective)
    metric_value = float(metrics[metric_name])
    current_reports = float(config.fractalrabbit.params.number_of_reports)
    scored: list[tuple[int, str, CandidateScore]] = []
    for index, candidate in enumerate(config.closed_loop.candidates):
        reporting_ratio = current_reports / float(candidate.params.number_of_reports)
        if config.closed_loop.objective_direction == "maximize":
            score = metric_value / max(reporting_ratio, 1e-12)
        else:
            score = metric_value * reporting_ratio
        tie_value = (
            candidate.name
            if config.closed_loop.tie_breaker == "name"
            else f"{index:08d}"
        )
        scored.append(
            (
                index,
                tie_value,
                CandidateScore(
                    name=candidate.name,
                    objective=config.closed_loop.objective,
                    metric_value=metric_value,
                    reporting_ratio=reporting_ratio,
                    score=score,
                    rank=0,
                ),
            )
        )
    if config.closed_loop.objective_direction == "maximize":
        scored.sort(key=lambda item: (-item[2].score, item[1]))
    else:
        scored.sort(key=lambda item: (item[2].score, item[1]))
    ranked = []
    for rank, (_, _, item) in enumerate(scored, start=1):
        ranked.append(
            CandidateScore(
                name=item.name,
                objective=item.objective,
                metric_value=item.metric_value,
                reporting_ratio=item.reporting_ratio,
                score=item.score,
                rank=rank,
            )
        )
    return tuple(ranked)


def _holdout_metrics(
    config: ProjectConfig, sequence: ObservationSequence
) -> dict[str, float]:
    """Compute holdout scoring as an additional pass when configured.

    The primary metrics remain estimated on the full sequence; holdout keys
    are additive so enabling the knob never changes the headline numbers.
    The holdout pass re-discretizes with training-only statistics (quantile
    bin edges, auto bounds, median burst cutoff) so holdout rows cannot
    shape the boundaries that score them, and A/B/D estimation excludes the
    holdout rows.
    """

    fraction = config.experiment.holdout_fraction
    if fraction <= 0.0:
        return {}
    mask = holdout_split(sequence, fraction)
    metrics: dict[str, float] = {"holdout_fraction_configured": float(fraction)}
    if not mask.any():
        metrics["holdout_n_holdout"] = 0.0
        metrics["holdout_n_train"] = float(len(sequence.observations))
        return metrics
    train_sequence = discretize_observations(
        sequence.observations, config.inference, estimation_mask=~mask
    )
    train_model = build_pomdp(
        train_sequence,
        config.inference.smoothing,
        config.inference.model.controls,
        include=~mask,
        modalities=config.inference.model.modalities,
    )
    scored = run_holdout_scoring(train_model, train_sequence, mask)
    return metrics | {key: value for key, value in scored.items() if np.isfinite(value)}


def _objective_evaluation(model, sequence, inference, pymdp_result):
    """Per-modality decomposition, calibration, and skill scores for each lane."""

    metrics, reference_payload = lane_objectives(
        model,
        sequence,
        inference.posterior_states,
        inference.policy_posterior,
        inference.predictive_loss,
        "reference",
    )
    metrics |= reference_payload["floors"]
    lanes = [reference_payload]
    if pymdp_result is not None:
        pymdp_metrics, pymdp_payload = lane_objectives(
            model,
            sequence,
            pymdp_result.posterior_states,
            pymdp_result.policy_posterior,
            pymdp_result.predictive_loss,
            "pymdp",
        )
        metrics |= pymdp_metrics
        lanes.append(pymdp_payload)
    return metrics, lanes


def _objective_metric(objective: str) -> str:
    if objective == "posterior_entropy":
        return "mean_posterior_entropy"
    return "mean_predictive_loss"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _write_posterior_csv(path: Path, posterior) -> None:
    rows = ["t," + ",".join(f"state_{i}" for i in range(posterior.shape[1]))]
    for idx, row in enumerate(posterior):
        rows.append(str(idx) + "," + ",".join(f"{value:.12g}" for value in row))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_discrete_features_csv(path: Path, sequence: ObservationSequence) -> None:
    rows = [
        "ID,Days,x(km),y(km),is_first_observation,x_bin,y_bin,spatial_state,time_bin,speed_km_per_day,"
        "interval_days,speed_category,burst_category"
    ]
    for idx, row in enumerate(sequence.observations):
        rows.append(
            ",".join(
                [
                    str(row.traveller_id),
                    f"{row.days:.12g}",
                    f"{row.x_km:.12g}",
                    f"{row.y_km:.12g}",
                    str(int(sequence.is_first_observation[idx])),
                    str(int(sequence.x_bins[idx])),
                    str(int(sequence.y_bins[idx])),
                    str(int(sequence.spatial_states[idx])),
                    str(int(sequence.time_bins[idx])),
                    f"{sequence.speeds_km_per_day[idx]:.12g}",
                    f"{sequence.intervals_days[idx]:.12g}",
                    str(int(sequence.speed_categories[idx])),
                    str(int(sequence.burst_categories[idx])),
                ]
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_inference_trace_csv(path: Path, inference: InferenceResult) -> None:
    rows = ["t,predictive_loss,posterior_entropy"]
    for idx, (loss, entropy) in enumerate(
        zip(inference.predictive_loss, inference.posterior_entropy)
    ):
        rows.append(f"{idx},{loss:.12g},{entropy:.12g}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_baseline_trace_csv(path: Path, baseline: BaselineResult) -> None:
    rows = ["t,baseline_predictive_loss,baseline_posterior_entropy"]
    for idx, (loss, entropy) in enumerate(
        zip(baseline.predictive_loss, baseline.posterior_entropy)
    ):
        rows.append(f"{idx},{loss:.12g},{entropy:.12g}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_pymdp_trace_csv(path: Path, result, controls: tuple[str, ...]) -> None:
    headers = [
        "t",
        "pymdp_predictive_loss",
        "pymdp_posterior_entropy",
        "pymdp_vfe",
        "pymdp_step_runtime_ms",
    ]
    headers += [f"pymdp_q_pi_{name}" for name in controls]
    headers += [f"pymdp_neg_efe_{name}" for name in controls]
    rows = [",".join(headers)]
    for idx in range(result.predictive_loss.shape[0]):
        values = [
            str(idx),
            f"{result.predictive_loss[idx]:.12g}",
            f"{result.posterior_entropy[idx]:.12g}",
            f"{result.variational_free_energy[idx]:.12g}",
            f"{result.step_runtime_ms[idx]:.12g}",
        ]
        values += [f"{value:.12g}" for value in result.policy_posterior[idx]]
        values += [f"{value:.12g}" for value in result.expected_free_energy[idx]]
        rows.append(",".join(values))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _ground_truth_evaluation(source_csv: Path, sequence, inference, pymdp_result):
    """Score posteriors against the exact true cell; join PLACES when emitted."""

    metrics, ref_mass, ref_top1 = state_recovery(
        sequence, inference.posterior_states, "state_recovery_reference"
    )
    rows: dict[str, object] = {
        "true_cells": sequence.spatial_states,
        "reference_mass": ref_mass,
        "reference_top1": ref_top1,
        "pymdp_mass": None,
        "pymdp_top1": None,
        "place_ids": None,
    }
    if pymdp_result is not None:
        pymdp_metrics, pymdp_mass, pymdp_top1 = state_recovery(
            sequence, pymdp_result.posterior_states, "state_recovery_pymdp"
        )
        metrics |= pymdp_metrics
        rows["pymdp_mass"] = pymdp_mass
        rows["pymdp_top1"] = pymdp_top1
    places_path = places_csv_for(source_csv)
    if places_path is not None:
        place_ids = align_places(sequence.observations, parse_places_csv(places_path))
        metrics |= place_statistics(place_ids)
        rows["place_ids"] = place_ids
    return metrics, rows


def _write_ground_truth_csv(path: Path, rows: dict[str, object]) -> None:
    headers = ["t", "true_cell", "reference_mass_on_true", "reference_top1"]
    if rows["pymdp_mass"] is not None:
        headers += ["pymdp_mass_on_true", "pymdp_top1"]
    if rows["place_ids"] is not None:
        headers += ["true_place_id"]
    lines = [",".join(headers)]
    n_rows = len(rows["true_cells"])
    for idx in range(n_rows):
        values = [
            str(idx),
            str(int(rows["true_cells"][idx])),
            f"{rows['reference_mass'][idx]:.12g}",
            f"{rows['reference_top1'][idx]:.0f}",
        ]
        if rows["pymdp_mass"] is not None:
            values += [
                f"{rows['pymdp_mass'][idx]:.12g}",
                f"{rows['pymdp_top1'][idx]:.0f}",
            ]
        if rows["place_ids"] is not None:
            values += [str(int(rows["place_ids"][idx]))]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_policy_trace_csv(
    path: Path, inference: InferenceResult, controls: tuple[str, ...]
) -> None:
    headers = ["t"]
    headers += [f"proxy_score_{name}" for name in controls]
    headers += [f"predictive_entropy_{name}" for name in controls]
    headers += [f"preference_divergence_{name}" for name in controls]
    headers += [f"policy_posterior_{name}" for name in controls]
    rows = [",".join(headers)]
    for idx in range(inference.policy_proxy_score.shape[0]):
        values = [str(idx)]
        values += [f"{value:.12g}" for value in inference.policy_proxy_score[idx]]
        values += [
            f"{value:.12g}" for value in inference.policy_predictive_entropy[idx]
        ]
        values += [
            f"{value:.12g}" for value in inference.policy_preference_divergence[idx]
        ]
        values += [f"{value:.12g}" for value in inference.policy_posterior[idx]]
        rows.append(",".join(values))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_candidate_scores(
    json_path: Path, csv_path: Path, scores: tuple[CandidateScore, ...]
) -> None:
    payload = [asdict(score) for score in scores]
    _write_json(json_path, payload)
    rows = ["rank,name,objective,metric_value,reporting_ratio,score"]
    for score in scores:
        rows.append(
            f"{score.rank},{score.name},{score.objective},{score.metric_value:.12g},"
            f"{score.reporting_ratio:.12g},{score.score:.12g}"
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
