"""Spatial-resolution frontier for the discrete generative model.

The default grid size is a model-surface choice, not a plotting detail. This
analysis evaluates candidate square grids on occupancy, transition sparsity,
observed-location predictive fit, location-withheld recovery, and bounded real
pymdp matched-prior agreement. The result justifies the default as a measured
tradeoff while keeping higher resolutions available as stress settings.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import DEFAULT_PALETTE, load_config
from .figure_style import LEGEND_FONT_SIZE, SMALL_LABEL_FONT_SIZE, manuscript_rcparams
from .fixture_generation import generate_waypoint_observations, write_waypoint_csv
from .fractalrabbit import parse_waypoint_csv
from .inference import run_latent_inference
from .observation_sources import resolve_observations_path
from .observations import discretize_observations
from .pomdp import build_pomdp
from .pymdp_agent import run_pymdp_inference
from .types import InferenceConfig, ProjectConfig, WaypointObservation

SPATIAL_RESOLUTION_FRONTIER_SCHEMA = (
    "active-fractal-rabbit-spatial-resolution-frontier-v1"
)
SPATIAL_RESOLUTION_CONFIRMATION_SCHEMA = (
    "active-fractal-rabbit-spatial-resolution-larger-sample-confirmation-v1"
)
DEFAULT_CANDIDATE_BINS = (3, 4, 5, 6, 8)
NO_LOCATION_MODALITIES = ("time_of_day", "speed", "report_burst")
FULL_LOCATION_MODALITIES = ("location", "time_of_day", "speed", "report_burst")
PYMDP_VALIDATION_MAX_STEPS = 32
DEFAULT_CONFIRMATION_SAMPLE_MULTIPLIER = 8
DEFAULT_CONFIRMATION_SAMPLE_MULTIPLIERS = (2, 4, 8)
DEFAULT_CONFIRMATION_SEED = 20260709


def run_spatial_resolution_frontier(
    config: ProjectConfig,
    observations: tuple[WaypointObservation, ...],
    *,
    candidate_bins: tuple[int, ...] = DEFAULT_CANDIDATE_BINS,
    pymdp_validation_steps: int = PYMDP_VALIDATION_MAX_STEPS,
) -> dict[str, object]:
    """Evaluate candidate spatial grids against model-fit and sparsity metrics."""

    if not candidate_bins:
        raise ValueError("candidate_bins must contain at least one grid size")
    unique_bins = tuple(dict.fromkeys(int(item) for item in candidate_bins))
    if any(item < 2 for item in unique_bins):
        raise ValueError("spatial-resolution candidates must be >= 2")

    rows = [
        _candidate_row(config, observations, bins, pymdp_validation_steps)
        for bins in unique_bins
    ]
    default_bins = int(config.inference.observation.spatial_bins)
    default_row = _row_for_bins(rows, default_bins)
    if default_row is None:
        raise ValueError(
            f"default spatial_bins={default_bins} is not present in candidate_bins"
        )
    coarse_row = _row_for_bins(rows, 3) or next(
        (row for row in rows if int(row["spatial_bins"]) < default_bins), rows[0]
    )
    strict_feasible = [
        row
        for row in rows
        if int(row["empty_state_count"]) == 0
        and int(row["min_rows_per_occupied_state"]) >= 3
    ]
    candidate_pool = strict_feasible or rows
    best_feasible = min(
        candidate_pool, key=lambda row: float(row["full_location_mean_predictive_loss"])
    )
    best_full_location = min(
        rows, key=lambda row: float(row["full_location_mean_predictive_loss"])
    )
    best_no_location = max(
        rows, key=lambda row: float(row["no_location_mean_mass_on_true_cell"])
    )
    default_minus_coarse = {
        "coarse_spatial_bins": int(coarse_row["spatial_bins"]),
        "state_count": int(default_row["state_count"]) - int(coarse_row["state_count"]),
        "full_location_mean_predictive_loss": (
            float(default_row["full_location_mean_predictive_loss"])
            - float(coarse_row["full_location_mean_predictive_loss"])
        ),
        "no_location_mean_mass_on_true_cell": (
            float(default_row["no_location_mean_mass_on_true_cell"])
            - float(coarse_row["no_location_mean_mass_on_true_cell"])
        ),
        "no_location_top1_accuracy": (
            float(default_row["no_location_top1_accuracy"])
            - float(coarse_row["no_location_top1_accuracy"])
        ),
    }
    verdict = (
        "default_resolution_is_best_fully_occupied_loss"
        if int(best_feasible["spatial_bins"]) == default_bins
        else "default_resolution_not_best_feasible_loss"
    )
    sample_summary = _sample_summary(observations)
    selection_summary = _selection_summary(
        default_row=default_row,
        coarse_row=coarse_row,
        best_feasible=best_feasible,
        best_full_location=best_full_location,
        best_no_location=best_no_location,
        strict_feasible_candidate_count=len(strict_feasible),
    )
    return {
        "schema": SPATIAL_RESOLUTION_FRONTIER_SCHEMA,
        "status": "completed",
        "source": str(resolve_observations_path(config)),
        **sample_summary,
        "candidate_bins": list(unique_bins),
        "default_spatial_bins": default_bins,
        "default_state_count": int(default_row["state_count"]),
        "default_row": default_row,
        "coarse_row": coarse_row,
        "best_feasible_row": best_feasible,
        "best_full_location_fit_row": best_full_location,
        "best_no_location_recovery_row": best_no_location,
        "strict_feasible_candidate_count": int(len(strict_feasible)),
        "best_feasible_is_strict": bool(strict_feasible),
        "default_minus_coarse": default_minus_coarse,
        "selection_rules": {
            "feasible_empty_state_count": 0,
            "feasible_min_rows_per_occupied_state": 3,
            "primary_metric": "full_location_mean_predictive_loss",
            "guardrail_metric": "no_location_mean_mass_on_true_cell",
            "fallback_when_no_strict_feasible_candidates": "best measured loss among all candidates",
        },
        "selection_summary": selection_summary,
        "rows": rows,
        "verdict": verdict,
        "caveat": (
            "Fixture-bounded synthetic-only spatial-resolution frontier. A lower full-location predictive "
            "loss means the observed-location generative model fits this synthetic trace better at that grid; "
            "it does not imply metadata-only recovery improves. The no-location columns are the guardrail for "
            "that stronger claim, and higher grids remain stress settings when empty or singleton cells appear."
        ),
    }


def run_and_write_spatial_resolution_frontier(
    config_path: str | Path,
    out_path: str | Path,
    observations_path: str | Path | None = None,
    *,
    candidate_bins: tuple[int, ...] = DEFAULT_CANDIDATE_BINS,
    pymdp_validation_steps: int = PYMDP_VALIDATION_MAX_STEPS,
) -> Path:
    """Write the canonical ``spatial_resolution_frontier.json`` artifact."""

    config = load_config(config_path)
    source = resolve_observations_path(
        config,
        observations_path,
        requested_role="spatial-resolution frontier observations",
    )
    payload = run_spatial_resolution_frontier(
        config,
        parse_waypoint_csv(source),
        candidate_bins=candidate_bins,
        pymdp_validation_steps=pymdp_validation_steps,
    )
    payload["source"] = str(source)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def run_and_write_larger_sample_confirmation(
    config_path: str | Path,
    frontier_path: str | Path,
    out_path: str | Path,
    observations_out_path: str | Path,
    *,
    sample_multiplier: int = DEFAULT_CONFIRMATION_SAMPLE_MULTIPLIER,
    sample_multipliers: tuple[int, ...] | None = None,
    seed: int = DEFAULT_CONFIRMATION_SEED,
    candidate_bins: tuple[int, ...] = DEFAULT_CANDIDATE_BINS,
    pymdp_validation_steps: int = PYMDP_VALIDATION_MAX_STEPS,
) -> Path:
    """Write deterministic larger-sample checks for the frontier decision."""

    multipliers = _confirmation_multipliers(sample_multiplier, sample_multipliers)
    config = load_config(config_path)
    frontier_file = Path(frontier_path)
    if frontier_file.is_file():
        primary_payload = json.loads(frontier_file.read_text(encoding="utf-8"))
    else:
        primary_source = resolve_observations_path(config)
        primary_payload = run_spatial_resolution_frontier(
            config,
            parse_waypoint_csv(primary_source),
            candidate_bins=candidate_bins,
            pymdp_validation_steps=pymdp_validation_steps,
        )
        primary_payload["source"] = str(primary_source)

    source = resolve_observations_path(config)
    primary_observations = parse_waypoint_csv(source)
    primary_summary = _sample_summary(primary_observations)
    largest_multiplier = max(multipliers)
    observations_out = Path(observations_out_path)
    ladder: list[dict[str, object]] = []
    largest_frontier: dict[str, object] | None = None
    largest_observations_path: Path | None = None
    largest_comparison: dict[str, object] | None = None

    for multiplier in multipliers:
        reports_per_traveller = int(
            primary_summary["max_reports_per_traveller"] * multiplier
        )
        confirmation_observations = generate_waypoint_observations(
            seed=seed,
            n_travellers=int(primary_summary["n_travellers"]),
            n_reports_per_traveller=reports_per_traveller,
            n_days=float(config.fractalrabbit.params.number_of_days),
        )
        multiplier_observations_path = observations_out.with_name(
            f"{multiplier}x_{observations_out.name}"
        )
        observations_path = write_waypoint_csv(
            confirmation_observations, multiplier_observations_path
        )
        if multiplier == largest_multiplier:
            observations_path = write_waypoint_csv(
                confirmation_observations, observations_out
            )
        confirmation_frontier = run_spatial_resolution_frontier(
            config,
            confirmation_observations,
            candidate_bins=candidate_bins,
            pymdp_validation_steps=pymdp_validation_steps,
        )
        confirmation_frontier["source"] = str(observations_path)
        comparison = _confirmation_summary(primary_payload, confirmation_frontier)
        summary = {
            **_compact_frontier_summary(confirmation_frontier),
            "sample_multiplier": int(multiplier),
            "source": str(observations_path),
            "comparison": comparison,
            "comparison_status": str(comparison["status"]),
            "comparison_verdict": str(comparison["verdict"]),
            "same_verdict": bool(comparison["same_verdict"]),
            "same_no_location_guardrail_direction": bool(
                comparison["same_no_location_guardrail_direction"]
            ),
            "same_best_feasible_bins": bool(comparison["same_best_feasible_bins"]),
        }
        ladder.append(summary)
        if multiplier == largest_multiplier:
            largest_frontier = confirmation_frontier
            largest_observations_path = observations_path
            largest_comparison = comparison

    if largest_frontier is None or largest_observations_path is None or largest_comparison is None:
        raise ValueError("sample_multipliers must contain at least one multiplier")
    stability = _sample_size_ladder_summary(ladder)
    payload = {
        "schema": SPATIAL_RESOLUTION_CONFIRMATION_SCHEMA,
        "status": "completed",
        "primary_source": str(source),
        "larger_sample_source": str(largest_observations_path),
        "seed": int(seed),
        "sample_multiplier": int(largest_multiplier),
        "sample_multipliers": list(multipliers),
        "candidate_bins": list(candidate_bins),
        "primary": _compact_frontier_summary(primary_payload),
        "larger_sample": _compact_frontier_summary(largest_frontier),
        "larger_sample_rows": largest_frontier["rows"],
        "comparison": largest_comparison,
        "sample_size_ladder": ladder,
        "sample_size_ladder_summary": stability,
        "caveat": (
            "Deterministic larger synthetic samples from the same fixture generator family; "
            "these are qualitative confirmation lanes, not independent population confidence intervals."
        ),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    primary_payload["larger_sample_confirmation"] = {
        **largest_comparison,
        "artifact": str(out),
        "source": str(largest_observations_path),
        "seed": int(seed),
        "sample_multiplier": int(largest_multiplier),
        "sample_multipliers": list(multipliers),
        "sample_size_ladder": ladder,
        "sample_size_ladder_summary": stability,
        "n_observations": int(largest_frontier["n_observations"]),
        "n_travellers": int(largest_frontier["n_travellers"]),
        "reports_per_traveller": int(
            largest_frontier["max_reports_per_traveller"]
        ),
        "best_feasible_bins": int(
            largest_frontier["best_feasible_row"]["spatial_bins"]
        ),
        "verdict": str(largest_frontier["verdict"]),
    }
    frontier_file.parent.mkdir(parents=True, exist_ok=True)
    frontier_file.write_text(
        json.dumps(primary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def render_spatial_resolution_frontier(
    payload: dict[str, object],
    path: str | Path,
    palette: dict[str, str] | None = None,
) -> None:
    """Render the spatial-resolution frontier figure."""

    colors = {**DEFAULT_PALETTE, **(palette or {})}
    plt.rcParams.update(manuscript_rcparams(colors))
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list) or not rows:
        fig, axis = plt.subplots(figsize=(7, 3.2))
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "spatial resolution frontier unavailable",
            ha="center",
            va="center",
        )
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    ordered = sorted(
        (row for row in rows if isinstance(row, dict)),
        key=lambda row: int(row["spatial_bins"]),
    )
    bins = np.asarray([int(row["spatial_bins"]) for row in ordered], dtype=int)
    full_loss = np.asarray(
        [float(row["full_location_mean_predictive_loss"]) for row in ordered],
        dtype=float,
    )
    no_location_mass = np.asarray(
        [float(row["no_location_mean_mass_on_true_cell"]) for row in ordered],
        dtype=float,
    )
    no_location_top1 = np.asarray(
        [float(row["no_location_top1_accuracy"]) for row in ordered], dtype=float
    )
    occupied = np.asarray(
        [int(row["occupied_state_count"]) for row in ordered], dtype=int
    )
    empty = np.asarray([int(row["empty_state_count"]) for row in ordered], dtype=int)
    occupied_fraction = np.asarray(
        [float(row["occupied_state_fraction"]) for row in ordered], dtype=float
    )
    min_rows = np.asarray(
        [int(row["min_rows_per_occupied_state"]) for row in ordered], dtype=int
    )
    transition_sparsity = np.asarray(
        [float(row["transition_edge_sparsity"]) for row in ordered], dtype=float
    )
    default_bins = int(payload.get("default_spatial_bins", bins[0]))

    # constrained_layout + taller canvas (F6-4) so the left y-label is not
    # clipped at the bottom edge and the suptitle/titles get reserved space.
    fig, axes_grid = plt.subplots(2, 2, figsize=(13.6, 8.2), constrained_layout=True)
    axes = axes_grid.ravel()
    axes[0].plot(
        bins, full_loss, marker="o", color=colors["blue"], label="full-location loss"
    )
    default_idx = (
        int(np.where(bins == default_bins)[0][0]) if default_bins in bins else 0
    )
    axes[0].scatter(
        [bins[default_idx]],
        [full_loss[default_idx]],
        s=86,
        color=colors["green"],
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
        label=f"default {default_bins}x{default_bins}",
    )
    axes[0].set_xlabel("spatial bins per axis")
    axes[0].set_ylabel("loss\n(nats)", labelpad=6)
    axes[0].set_title("Observed-location fit\n(lower is better)", pad=8)
    axes[0].legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    axes[0].grid(True, linewidth=0.4, alpha=0.3)

    axes[1].bar(
        bins - 0.16, occupied, width=0.32, color=colors["green"], label="occupied"
    )
    axes[1].bar(bins + 0.16, empty, width=0.32, color=colors["orange"], label="empty")
    occupancy_axis = axes[1]
    sparsity_axis = occupancy_axis.twinx()
    axes[1].plot(
        bins,
        min_rows,
        color=colors["purple"],
        marker="^",
        linestyle="--",
        label="min rows/cell",
    )
    sparsity_axis.plot(
        bins,
        transition_sparsity,
        color=colors["gray"],
        marker="s",
        label="transition edge sparsity",
    )
    axes[1].set_xlabel("spatial bins per axis")
    axes[1].set_ylabel("count\n(cells/reports)", labelpad=6)
    sparsity_axis.set_ylabel("edge sparsity\n(0-1)", labelpad=8)
    axes[1].set_title("Occupancy load", pad=8)
    # Legend below the middle panel (F6-4) instead of loc="best", where it sat
    # directly on top of the occupied/empty bars and occluded the orange series.
    handles, labels = axes[1].get_legend_handles_labels()
    sparsity_handles, sparsity_labels = sparsity_axis.get_legend_handles_labels()
    axes[1].legend(
        handles + sparsity_handles,
        labels + sparsity_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=LEGEND_FONT_SIZE,
        framealpha=0.9,
    )
    axes[1].grid(True, axis="y", linewidth=0.4, alpha=0.3)

    axes[2].plot(
        bins,
        no_location_mass,
        marker="o",
        color=colors["purple"],
        label="mass on true cell",
    )
    axes[2].plot(
        bins,
        no_location_top1,
        marker="s",
        color=colors["orange"],
        label="top-1 accuracy",
    )
    axes[2].set_xlabel("spatial bins per axis")
    axes[2].set_ylabel("recovery fraction\n(0-1)", labelpad=6)
    axes[2].set_ylim(
        0.0,
        min(
            1.0,
            max(float(np.max(no_location_mass)), float(np.max(no_location_top1))) * 1.25
            + 0.02,
        ),
    )
    # Pad the title (F6-4) so it clears the rotated "no-location recovery" y-label.
    axes[2].set_title("Location-withheld guardrail", pad=8)
    axes[2].legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    axes[2].grid(True, linewidth=0.4, alpha=0.3)

    axes[3].scatter(
        full_loss,
        no_location_mass,
        s=70 + occupied_fraction * 100,
        c=empty,
        cmap="Oranges",
        edgecolor="black",
        linewidth=0.45,
        alpha=0.90,
    )
    for bin_count, loss, mass in zip(bins, full_loss, no_location_mass):
        axes[3].annotate(
            f"{bin_count}x{bin_count}",
            xy=(loss, mass),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=SMALL_LABEL_FONT_SIZE,
        )
    best_idx = int(np.argmin(full_loss))
    axes[3].scatter(
        [full_loss[default_idx]],
        [no_location_mass[default_idx]],
        marker="D",
        s=88,
        color=colors["green"],
        edgecolor="black",
        linewidth=0.5,
        zorder=4,
        label="default",
    )
    axes[3].scatter(
        [full_loss[best_idx]],
        [no_location_mass[best_idx]],
        marker="*",
        s=128,
        color=colors["yellow"],
        edgecolor="black",
        linewidth=0.5,
        zorder=4,
        label="lowest loss",
    )
    axes[3].set_xlabel("observed-location loss\n(nats; lower better)")
    axes[3].set_ylabel("true-cell mass\n(0-1)", labelpad=6)
    axes[3].set_title("Fit vs metadata-only guardrail", pad=8)
    axes[3].legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    axes[3].grid(True, linewidth=0.4, alpha=0.3)
    axes[3].axvline(
        full_loss[default_idx], color="black", linestyle="--", linewidth=0.9, alpha=0.55
    )

    for axis in axes[:3]:
        axis.axvline(
            default_bins, color="black", linestyle="--", linewidth=0.9, alpha=0.55
        )
    verdict = str(payload.get("verdict", "unavailable"))
    default_minus = (
        payload.get("default_minus_coarse", {})
        if isinstance(payload.get("default_minus_coarse"), dict)
        else {}
    )
    fit_delta = float(default_minus.get("full_location_mean_predictive_loss", 0.0))
    if verdict == "default_resolution_is_best_fully_occupied_loss":
        verdict_line = "default is best feasible observed-location fit"
    elif fit_delta > 0:
        verdict_line = f"default remains auditable, not best feasible fit (+{fit_delta:.2f} nats vs coarse)"
    else:
        verdict_line = f"default fit delta vs coarse: {fit_delta:+.2f} nats"
    confirmation = (
        payload.get("larger_sample_confirmation", {})
        if isinstance(payload.get("larger_sample_confirmation"), dict)
        else {}
    )
    confirmation_line = ""
    if confirmation:
        status = str(confirmation.get("status", "unavailable"))
        larger_n = int(confirmation.get("n_observations", 0) or 0)
        larger_best = confirmation.get("best_feasible_bins", "n/a")
        larger_strict = bool(confirmation.get("larger_best_feasible_is_strict", True))
        best_label = "best feasible" if larger_strict else "best measured"
        stability = (
            confirmation.get("sample_size_ladder_summary", {})
            if isinstance(confirmation.get("sample_size_ladder_summary"), dict)
            else {}
        )
        multipliers = stability.get("multipliers", [])
        if isinstance(multipliers, list) and multipliers:
            multiplier_label = "/".join(f"{int(item)}x" for item in multipliers)
            guardrail = (
                "guardrail stable"
                if stability.get("all_same_no_location_guardrail_direction") is True
                else "guardrail sensitive"
            )
            confirmation_line = (
                f"; ladder {multiplier_label} {guardrail}, largest {status} "
                f"at n={larger_n}, {best_label} {larger_best}x{larger_best}"
            )
        else:
            multiplier = int(confirmation.get("sample_multiplier", 0) or 0)
            multiplier_label = f"{multiplier}x " if multiplier else ""
            confirmation_line = (
                f"; {multiplier_label}larger-sample check {status} at n={larger_n}, "
                f"{best_label} {larger_best}x{larger_best}"
            )
    fig.suptitle(
        "Spatial-resolution frontier: "
        f"{verdict_line}\n"
        "guardrails bound metadata-only recovery"
        f"{confirmation_line}",
        x=0.02,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    # constrained_layout (set on the figure) handles spacing; an explicit
    # tight_layout here would conflict with it and re-clip the y-label.
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _candidate_row(
    config: ProjectConfig,
    observations: tuple[WaypointObservation, ...],
    bins: int,
    pymdp_validation_steps: int,
) -> dict[str, object]:
    inference_config = _with_spatial_bins(config.inference, bins)
    sequence = discretize_observations(observations, inference_config)
    cells = np.asarray(sequence.spatial_states, dtype=int)
    full_model = build_pomdp(
        sequence,
        inference_config.smoothing,
        inference_config.model.controls,
        modalities=FULL_LOCATION_MODALITIES,
    )
    full = run_latent_inference(full_model, sequence)
    no_location_model = build_pomdp(
        sequence,
        inference_config.smoothing,
        inference_config.model.controls,
        modalities=NO_LOCATION_MODALITIES,
    )
    no_location = run_latent_inference(no_location_model, sequence)
    validation_indices = _sentinel_indices(
        len(sequence.observations), pymdp_validation_steps
    )
    pymdp = run_pymdp_inference(
        full_model,
        sequence,
        reference_posterior=full.posterior_states,
        validation_indices=validation_indices,
    )
    occupancy = np.bincount(cells, minlength=sequence.num_spatial_states)
    occupied_counts = occupancy[occupancy > 0]
    edge_count = _transition_edge_count(cells, sequence.is_first_observation)
    state_count = int(sequence.num_spatial_states)
    no_location_recovery = _recovery(no_location.posterior_states, cells)
    full_recovery = _recovery(full.posterior_states, cells)
    return {
        "spatial_bins": int(bins),
        "state_count": state_count,
        "occupied_state_count": int(occupied_counts.size),
        "empty_state_count": int(state_count - occupied_counts.size),
        "occupied_state_fraction": float(occupied_counts.size / state_count),
        "mean_rows_per_state": float(len(cells) / max(state_count, 1)),
        "min_rows_per_occupied_state": int(np.min(occupied_counts))
        if occupied_counts.size
        else 0,
        "mean_rows_per_occupied_state": float(np.mean(occupied_counts))
        if occupied_counts.size
        else 0.0,
        "median_rows_per_occupied_state": float(np.median(occupied_counts))
        if occupied_counts.size
        else 0.0,
        "transition_edge_count": int(edge_count),
        "transition_edge_density": float(
            edge_count / max(state_count * state_count, 1)
        ),
        "transition_edge_sparsity": float(
            1.0 - edge_count / max(state_count * state_count, 1)
        ),
        "full_location_mean_predictive_loss": float(np.mean(full.predictive_loss)),
        "full_location_mean_posterior_entropy": float(np.mean(full.posterior_entropy)),
        "full_location_mean_mass_on_true_cell": full_recovery["mean_mass_on_true_cell"],
        "full_location_top1_accuracy": full_recovery["top1_accuracy"],
        "no_location_mean_predictive_loss": float(np.mean(no_location.predictive_loss)),
        "no_location_mean_posterior_entropy": float(
            np.mean(no_location.posterior_entropy)
        ),
        "no_location_mean_mass_on_true_cell": no_location_recovery[
            "mean_mass_on_true_cell"
        ],
        "no_location_top1_accuracy": no_location_recovery["top1_accuracy"],
        "pymdp_validation_rows": float(pymdp.metrics["pymdp_inference_rows"]),
        "pymdp_source_observation_rows": float(
            pymdp.metrics["pymdp_source_observation_rows"]
        ),
        "pymdp_validation_fraction": float(pymdp.metrics["pymdp_validation_fraction"]),
        "pymdp_exact_inference_max_abs_delta": float(
            pymdp.metrics["pymdp_exact_inference_max_abs_delta"]
        ),
    }


def _with_spatial_bins(config: InferenceConfig, bins: int) -> InferenceConfig:
    observation = replace(config.observation, spatial_bins=int(bins))
    return replace(config, observation=observation)


def _recovery(posterior: np.ndarray, cells: np.ndarray) -> dict[str, float]:
    rows = np.arange(posterior.shape[0])
    return {
        "mean_mass_on_true_cell": float(np.mean(posterior[rows, cells])),
        "top1_accuracy": float(np.mean(np.argmax(posterior, axis=1) == cells)),
    }


def _transition_edge_count(cells: np.ndarray, first_observation: np.ndarray) -> int:
    edges = {
        (int(previous), int(current))
        for previous, current, is_first in zip(
            cells[:-1], cells[1:], first_observation[1:]
        )
        if not bool(is_first)
    }
    return len(edges)


def _sentinel_indices(n_steps: int, max_steps: int) -> tuple[int, ...]:
    if n_steps <= 0 or max_steps <= 0:
        return ()
    if n_steps <= max_steps:
        return tuple(range(n_steps))
    return tuple(
        int(item)
        for item in np.unique(np.linspace(0, n_steps - 1, max_steps, dtype=int))
    )


def _sample_summary(
    observations: tuple[WaypointObservation, ...]
) -> dict[str, int | float]:
    counts: dict[int, int] = {}
    for row in observations:
        counts[row.traveller_id] = counts.get(row.traveller_id, 0) + 1
    values = tuple(counts.values())
    return {
        "n_observations": len(observations),
        "n_travellers": len(counts),
        "min_reports_per_traveller": int(min(values)) if values else 0,
        "max_reports_per_traveller": int(max(values)) if values else 0,
        "mean_reports_per_traveller": float(np.mean(values)) if values else 0.0,
    }


def _confirmation_multipliers(
    sample_multiplier: int, sample_multipliers: tuple[int, ...] | None
) -> tuple[int, ...]:
    requested = sample_multipliers or DEFAULT_CONFIRMATION_SAMPLE_MULTIPLIERS
    if sample_multipliers is None and sample_multiplier != DEFAULT_CONFIRMATION_SAMPLE_MULTIPLIER:
        requested = (sample_multiplier,)
    multipliers = tuple(dict.fromkeys(int(item) for item in requested))
    if not multipliers:
        raise ValueError("sample_multipliers must contain at least one multiplier")
    if any(item <= 1 for item in multipliers):
        raise ValueError("sample multipliers must be greater than 1")
    return tuple(sorted(multipliers))


def _selection_summary(
    *,
    default_row: dict[str, object],
    coarse_row: dict[str, object],
    best_feasible: dict[str, object],
    best_full_location: dict[str, object],
    best_no_location: dict[str, object],
    strict_feasible_candidate_count: int,
) -> dict[str, object]:
    fit_gap = float(default_row["full_location_mean_predictive_loss"]) - float(
        best_feasible["full_location_mean_predictive_loss"]
    )
    no_location_delta = float(default_row["no_location_mean_mass_on_true_cell"]) - float(
        coarse_row["no_location_mean_mass_on_true_cell"]
    )
    return {
        "default_is_fully_occupied": int(default_row["empty_state_count"]) == 0,
        "default_is_best_feasible_fit": int(default_row["spatial_bins"])
        == int(best_feasible["spatial_bins"]),
        "strict_feasible_candidate_count": int(strict_feasible_candidate_count),
        "best_feasible_is_strict": bool(strict_feasible_candidate_count > 0),
        "default_loss_gap_vs_best_feasible": float(fit_gap),
        "default_no_location_mass_delta_vs_coarse": float(no_location_delta),
        "metadata_only_guardrail_direction": _signed_direction(no_location_delta),
        "best_full_location_fit_bins": int(best_full_location["spatial_bins"]),
        "best_no_location_recovery_bins": int(best_no_location["spatial_bins"]),
    }


def _sample_size_ladder_summary(ladder: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(ladder, key=lambda item: int(item["sample_multiplier"]))
    if not ordered:
        return {
            "multipliers": [],
            "largest_multiplier": 0,
            "largest_n_observations": 0,
            "all_same_verdict": False,
            "all_same_no_location_guardrail_direction": False,
            "strict_feasible_counts": [],
            "selected_bins": [],
            "interpretation": "unavailable",
        }
    all_same_verdict = all(bool(item.get("same_verdict", False)) for item in ordered)
    all_guardrail_same = all(
        bool(item.get("same_no_location_guardrail_direction", False))
        for item in ordered
    )
    strict_feasible_counts = [
        int(item.get("strict_feasible_candidate_count", 0) or 0) for item in ordered
    ]
    selected_bins = [int(item.get("best_feasible_bins", 0) or 0) for item in ordered]
    largest = ordered[-1]
    if all_same_verdict and all_guardrail_same and all(strict_feasible_counts):
        interpretation = "stable_verdict_guardrail_and_strict_feasibility"
    elif all_same_verdict and all_guardrail_same:
        interpretation = "stable_verdict_and_guardrail_with_non_strict_lanes"
    elif all_guardrail_same:
        interpretation = "guardrail_stable_but_fit_selection_changes"
    else:
        interpretation = "sample_size_sensitive"
    return {
        "multipliers": [int(item["sample_multiplier"]) for item in ordered],
        "largest_multiplier": int(largest.get("sample_multiplier", 0) or 0),
        "largest_n_observations": int(largest.get("n_observations", 0) or 0),
        "largest_reports_per_traveller": int(
            largest.get("reports_per_traveller", 0) or 0
        ),
        "all_same_verdict": bool(all_same_verdict),
        "all_same_no_location_guardrail_direction": bool(all_guardrail_same),
        "strict_feasible_counts": strict_feasible_counts,
        "selected_bins": selected_bins,
        "interpretation": interpretation,
    }


def _compact_frontier_summary(payload: dict[str, object]) -> dict[str, object]:
    default = (
        payload.get("default_row", {}) if isinstance(payload.get("default_row"), dict) else {}
    )
    best = (
        payload.get("best_feasible_row", {})
        if isinstance(payload.get("best_feasible_row"), dict)
        else {}
    )
    default_minus = (
        payload.get("default_minus_coarse", {})
        if isinstance(payload.get("default_minus_coarse"), dict)
        else {}
    )
    return {
        "n_observations": int(payload.get("n_observations", 0) or 0),
        "n_travellers": int(payload.get("n_travellers", 0) or 0),
        "reports_per_traveller": int(
            payload.get("max_reports_per_traveller", 0) or 0
        ),
        "verdict": str(payload.get("verdict", "unavailable")),
        "default_spatial_bins": int(payload.get("default_spatial_bins", 0) or 0),
        "default_state_count": int(default.get("state_count", 0) or 0),
        "default_empty_state_count": int(default.get("empty_state_count", 0) or 0),
        "default_full_location_loss": float(
            default.get("full_location_mean_predictive_loss", 0.0) or 0.0
        ),
        "best_feasible_bins": int(best.get("spatial_bins", 0) or 0),
        "strict_feasible_candidate_count": int(
            payload.get("strict_feasible_candidate_count", 0) or 0
        ),
        "best_feasible_is_strict": bool(payload.get("best_feasible_is_strict", False)),
        "default_minus_coarse_loss": float(
            default_minus.get("full_location_mean_predictive_loss", 0.0) or 0.0
        ),
        "default_minus_coarse_no_location_mass": float(
            default_minus.get("no_location_mean_mass_on_true_cell", 0.0) or 0.0
        ),
    }


def _confirmation_summary(
    primary_payload: dict[str, object], confirmation_payload: dict[str, object]
) -> dict[str, object]:
    primary = _compact_frontier_summary(primary_payload)
    larger = _compact_frontier_summary(confirmation_payload)
    same_verdict = primary["verdict"] == larger["verdict"]
    same_best = primary["best_feasible_bins"] == larger["best_feasible_bins"]
    same_guardrail = _signed_direction(
        float(primary["default_minus_coarse_no_location_mass"])
    ) == _signed_direction(float(larger["default_minus_coarse_no_location_mass"]))
    default_occupied = (
        int(primary["default_empty_state_count"]) == 0
        and int(larger["default_empty_state_count"]) == 0
    )
    confirmed = bool(same_verdict and same_best and same_guardrail and default_occupied)
    return {
        "status": "confirmed" if confirmed else "changed",
        "verdict": (
            "larger_sample_confirms_frontier_decision"
            if confirmed
            else "larger_sample_changes_frontier_decision"
        ),
        "same_verdict": bool(same_verdict),
        "same_best_feasible_bins": bool(same_best),
        "same_no_location_guardrail_direction": bool(same_guardrail),
        "default_fully_occupied_in_both": bool(default_occupied),
        "primary_n_observations": int(primary["n_observations"]),
        "larger_n_observations": int(larger["n_observations"]),
        "primary_best_feasible_bins": int(primary["best_feasible_bins"]),
        "larger_best_feasible_bins": int(larger["best_feasible_bins"]),
        "primary_strict_feasible_candidate_count": int(
            primary["strict_feasible_candidate_count"]
        ),
        "larger_strict_feasible_candidate_count": int(
            larger["strict_feasible_candidate_count"]
        ),
        "primary_best_feasible_is_strict": bool(primary["best_feasible_is_strict"]),
        "larger_best_feasible_is_strict": bool(larger["best_feasible_is_strict"]),
        "primary_default_minus_coarse_loss": float(
            primary["default_minus_coarse_loss"]
        ),
        "larger_default_minus_coarse_loss": float(larger["default_minus_coarse_loss"]),
        "primary_default_minus_coarse_no_location_mass": float(
            primary["default_minus_coarse_no_location_mass"]
        ),
        "larger_default_minus_coarse_no_location_mass": float(
            larger["default_minus_coarse_no_location_mass"]
        ),
    }


def _signed_direction(value: float) -> str:
    if value > 0:
        return "increases"
    if value < 0:
        return "decreases"
    return "unchanged"


def _row_for_bins(rows: list[dict[str, object]], bins: int) -> dict[str, object] | None:
    return next((row for row in rows if int(row["spatial_bins"]) == bins), None)
