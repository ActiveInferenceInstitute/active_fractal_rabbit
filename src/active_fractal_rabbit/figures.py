"""Figure generation for pipeline artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patches
from matplotlib import patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

from .config import DEFAULT_FIGURES, DEFAULT_PALETTE
from .figure_style import (
    ANNOTATION_FONT_SIZE,
    CANVAS_TITLE_FONT_SIZE,
    DIVERGING_CMAP,
    LEGEND_FONT_SIZE,
    METRIC_FONT_SIZE,
    PANEL_HEADING_FONT_SIZE,
    SEQUENTIAL_CMAP,
    SMALL_LABEL_FONT_SIZE,
    TICK_FONT_SIZE,
    editorial_palette,
    font_contract_payload,
    gate_status_color as _atlas_status_color,
    gate_verdict_label as _atlas_label,
    manuscript_rcparams,
)
from .sota_program_figures import sota_program_leaderboard_summary
from .types import FigureRegistryEntry


def _apply_manuscript_style(palette: dict[str, str]) -> None:
    """Apply a quiet, print-readable style shared by generated manuscript PNGs."""

    plt.rcParams.update(manuscript_rcparams(palette))


def _clean_label(value: object, *, width: int = 22) -> str:
    text = str(value).replace("_", " ")
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _model_axis_label(value: object, *, width: int = 24) -> str:
    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
    overrides = {
        "higher order backoff markov": "backoff Markov",
        "multi horizon recency": "multi-horizon recency",
        "return boosted markov": "return-boost Markov",
        "first order markov": "first-order Markov",
        "online base rate": "online base rate",
        "switching noise hmm": "switching HMM",
        "precision return aif": "precision-return AIF",
        "soft filter aif ref": "soft filter",
        "mlp next cell": "MLP next-cell",
    }
    return overrides.get(normalized, _clean_label(value, width=width))


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values
    bounded = min(window, int(values.size))
    kernel = np.ones(bounded, dtype=float)
    summed = np.convolve(values, kernel, mode="same")
    counts = np.convolve(np.ones_like(values, dtype=float), kernel, mode="same")
    return summed / np.maximum(counts, 1.0)


def _direct_line_label(axis, x: float, y: float, label: str, color: str) -> None:
    axis.annotate(
        label,
        xy=(x, y),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=SMALL_LABEL_FONT_SIZE,
        color=color,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 1.5},
        clip_on=False,
    )


def _fmt_stat(value: object, digits: int = 3) -> str:
    if not isinstance(value, int | float) or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _status_color(status: object, palette: dict[str, str]) -> str:
    return palette["green"] if str(status) == "completed" else palette["gray"]


# (context_key, filename) pairs for the uniform "runs/<name>.json" artifacts
# consumed by generate_figures. Table-driven so every entry shares one
# path-derivation rule (rooted at runs_dir = run_path.parents[1] / "runs")
# instead of ~30 hand-copied _read_json(run_path.parents[1] / "runs" / ..., {})
# call sites, which is exactly the kind of copy-paste surface that let
# _registry_entry's stray hardcoded "output/runs/..." path go unnoticed.
_RUN_SUMMARY_FILES: tuple[tuple[str, str], ...] = (
    ("negative_summary", "negative_controls_summary.json"),
    ("sparsity_summary", "sparsity_analysis.json"),
    ("anticipatory_summary", "anticipatory_analysis.json"),
    ("active_sensing_summary", "active_sensing_analysis.json"),
    ("anomaly_summary", "anomaly_analysis.json"),
    ("frontier_summary", "frontier_analysis.json"),
    ("return_summary", "return_structure_analysis.json"),
    ("variance_summary", "variance_bands.json"),
    ("upstream_summary", "upstream_sensitivity.json"),
    ("fractal_dimension_summary", "fractal_dimension_sweep.json"),
    ("circadian_phase_summary", "circadian_phase_analysis.json"),
    ("partial_obs_summary", "partial_observability_analysis.json"),
    ("sequence_baseline_summary", "sequence_baseline_analysis.json"),
    ("missingness_summary", "missingness_model_analysis.json"),
    ("spatial_resolution_summary", "spatial_resolution_frontier.json"),
    ("model_gate_summary", "model_gate_matrix.json"),
    ("forward_gate_summary", "forward_gates_summary.json"),
    ("factorial_summary", "factorial_pymdp_analysis_fixture.json"),
    ("efe_location_summary", "efe_location_frontier.json"),
    ("linkage_risk_summary", "linkage_risk_frontier.json"),
    ("poisoning_summary", "poisoning_robustness.json"),
    ("fair_summary", "fair_inference.json"),
    ("power_summary", "power_analysis.json"),
    ("privacy_summary", "privacy_frontier_analysis.json"),
    ("hardened_anomaly_summary", "hardened_anomaly_analysis.json"),
    ("aif_regime_summary", "aif_regime_sota_gate.json"),
    ("multiplicity_summary", "multiplicity_ledger.json"),
    ("confound_taxonomy_summary", "confound_taxonomy.json"),
    ("regime_mechanism_summary", "regime_mechanism_synthesis.json"),
    ("regime_switch_phase_summary", "regime_switch_phase_diagnostic.json"),
    ("privacy_utility_summary", "privacy_utility_frontier.json"),
    ("privacy_utility_repeats_summary", "privacy_utility_frontier_repeats.json"),
    ("synthetic_task_privacy_summary", "synthetic_task_utility_privacy.json"),
    ("model_comparison_summary", "model_comparison_leaderboard.json"),
    ("aif_deepening_summary", "aif_deepening_candidates.json"),
)


def _load_run_summaries(runs_dir: Path) -> dict[str, object]:
    """Load the uniform ``runs/<name>.json`` artifacts listed in ``_RUN_SUMMARY_FILES``.

    Every entry resolves against the same ``runs_dir`` (``run_path.parents[1] /
    "runs"``), so there is exactly one place that derives that path instead of
    one per artifact.
    """
    return {
        key: _read_json(runs_dir / filename, {}) for key, filename in _RUN_SUMMARY_FILES
    }


def generate_figures(
    run_dir: str | Path, figure_dir: str | Path | None = None
) -> dict[str, Path]:
    """Generate deterministic figures and a manuscript-facing figure registry."""

    run_path = Path(run_dir)
    config = _read_json(run_path / "config_snapshot.json", {})
    figures_config = config.get("figures", {}) if isinstance(config, dict) else {}
    configured_registry = figures_config.get("registry_path")
    if figure_dir:
        target = Path(figure_dir)
    elif configured_registry:
        target = Path(str(configured_registry)).parent
    else:
        target = run_path.parents[1] / "figures"
    target.mkdir(parents=True, exist_ok=True)
    _remove_stale_review_artifacts(target)
    features = _read_rows(run_path / "discrete_features.csv")
    observations = _read_rows(run_path / "observations.csv")
    inference = _read_rows(run_path / "inference_trace.csv")
    policy = _read_rows(run_path / "policy_trace.csv")
    candidate_scores = _read_json(run_path / "candidate_scores.json", [])
    runs_dir = run_path.parents[1] / "runs"
    summaries = _load_run_summaries(runs_dir)
    negative_summary = summaries["negative_summary"]
    sparsity_summary = summaries["sparsity_summary"]
    anticipatory_summary = summaries["anticipatory_summary"]
    active_sensing_summary = summaries["active_sensing_summary"]
    anomaly_summary = summaries["anomaly_summary"]
    frontier_summary = summaries["frontier_summary"]
    return_summary = summaries["return_summary"]
    variance_summary = summaries["variance_summary"]
    upstream_summary = summaries["upstream_summary"]
    fractal_dimension_summary = summaries["fractal_dimension_summary"]
    circadian_phase_summary = summaries["circadian_phase_summary"]
    partial_obs_summary = summaries["partial_obs_summary"]
    sequence_baseline_summary = summaries["sequence_baseline_summary"]
    missingness_summary = summaries["missingness_summary"]
    spatial_resolution_summary = summaries["spatial_resolution_summary"]
    model_gate_summary = summaries["model_gate_summary"]
    forward_gate_summary = summaries["forward_gate_summary"]
    factorial_summary = summaries["factorial_summary"]
    efe_location_summary = summaries["efe_location_summary"]
    linkage_risk_summary = summaries["linkage_risk_summary"]
    poisoning_summary = summaries["poisoning_summary"]
    sota_program_path = runs_dir / "sota_program_leaderboard.csv"
    sota_program_rows = (
        _read_rows(sota_program_path) if sota_program_path.exists() else []
    )
    sota_program_gate = _read_json(runs_dir / "sota_program_gate.json", {})
    sota_summary = sota_program_leaderboard_summary(
        sota_program_rows, sota_program_gate
    ) or _read_json(runs_dir / "sota_benchmark_matrix.json", {})
    fair_summary = summaries["fair_summary"]
    power_summary = summaries["power_summary"]
    privacy_summary = summaries["privacy_summary"]
    hardened_anomaly_summary = summaries["hardened_anomaly_summary"]
    aif_regime_summary = summaries["aif_regime_summary"]
    multiplicity_summary = summaries["multiplicity_summary"]
    confound_taxonomy_summary = summaries["confound_taxonomy_summary"]
    regime_mechanism_summary = summaries["regime_mechanism_summary"]
    regime_switch_phase_summary = summaries["regime_switch_phase_summary"]
    privacy_utility_summary = summaries["privacy_utility_summary"]
    privacy_utility_repeats_summary = summaries["privacy_utility_repeats_summary"]
    synthetic_task_privacy_summary = summaries["synthetic_task_privacy_summary"]
    model_comparison_summary = summaries["model_comparison_summary"]
    aif_deepening_summary = summaries["aif_deepening_summary"]
    pymdp_trace = (
        _read_rows(run_path / "pymdp_trace.csv")
        if (run_path / "pymdp_trace.csv").exists()
        else []
    )
    ground_truth = (
        _read_rows(run_path / "ground_truth.csv")
        if (run_path / "ground_truth.csv").exists()
        else []
    )
    calibration_summary = _read_json(run_path / "calibration.json", {})
    run_metrics = _read_json(run_path / "metrics.json", {})
    context = _figure_context(
        observations, features, inference, policy, candidate_scores, config
    )
    context["negative_controls_total"] = negative_summary.get(
        "n_controls", len(negative_summary.get("controls", []))
    )
    context["negative_controls_passed"] = negative_summary.get(
        "n_passed",
        sum(1 for item in negative_summary.get("controls", []) if item.get("passed")),
    )
    context["pymdp_exact_delta"] = run_metrics.get(
        "pymdp_exact_inference_max_abs_delta"
    )
    context["pymdp_loop_delta"] = run_metrics.get(
        "pymdp_reference_max_abs_posterior_delta"
    )
    context["recovery_reference_mass"] = run_metrics.get(
        "state_recovery_reference_mean_mass_on_true_cell"
    )
    context["recovery_reference_top1"] = run_metrics.get(
        "state_recovery_reference_top1_accuracy"
    )
    context["pymdp_median_step_ms"] = run_metrics.get("pymdp_median_step_runtime_ms")
    context["pymdp_mean_vfe"] = run_metrics.get("pymdp_mean_vfe")
    context["regime_switch_phase_summary"] = regime_switch_phase_summary
    enabled = tuple(figures_config.get("enabled", DEFAULT_FIGURES))
    dpi = int(figures_config.get("dpi", 220))
    palette = {**DEFAULT_PALETTE, **figures_config.get("palette", {})}
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["figure.dpi"] = dpi
    _apply_manuscript_style(palette)

    generators = {
        "evidence_atlas": lambda path: _evidence_atlas(
            observations,
            features,
            run_metrics,
            spatial_resolution_summary,
            model_gate_summary,
            forward_gate_summary,
            efe_location_summary,
            linkage_risk_summary,
            poisoning_summary,
            partial_obs_summary,
            multiplicity_summary,
            path,
            palette,
        ),
        "waypoint_paths": lambda path: _waypoint_paths(observations, path, palette),
        "spatial_bins": lambda path: _spatial_bins(features, path, palette),
        "spatial_resolution_frontier": lambda path: _spatial_resolution_frontier(
            spatial_resolution_summary, path, palette
        ),
        "reporting_bursts": lambda path: _reporting_bursts(
            features, path, palette, config
        ),
        "speed_categories": lambda path: _speed_categories(
            features, path, palette, config
        ),
        "transition_heatmap": lambda path: _transition_heatmap(features, path, palette),
        "likelihood_panels": lambda path: _likelihood_panels(features, path, palette),
        "posterior_entropy": lambda path: _posterior_entropy(
            inference, run_metrics, spatial_resolution_summary, path, palette
        ),
        "policy_proxy": lambda path: _policy_proxy(policy, path, palette),
        "candidate_scores": lambda path: _candidate_scores(
            candidate_scores, path, palette
        ),
        "negative_controls": lambda path: _negative_controls(
            negative_summary, path, palette
        ),
        "pymdp_agreement": lambda path: _pymdp_agreement(
            inference, pymdp_trace, run_metrics, path, palette
        ),
        "pymdp_diagnostics": lambda path: _pymdp_diagnostics(
            pymdp_trace, path, palette
        ),
        "state_recovery": lambda path: _state_recovery(ground_truth, path, palette),
        "sparsity_recovery": lambda path: _sparsity_recovery(
            sparsity_summary, path, palette
        ),
        "calibration": lambda path: _calibration(calibration_summary, path, palette),
        "anticipatory": lambda path: _anticipatory(anticipatory_summary, path, palette),
        "efe_decomposition": lambda path: _efe_decomposition(
            active_sensing_summary, path, palette
        ),
        "anomaly_detection": lambda path: _anomaly_detection(
            anomaly_summary, path, palette
        ),
        "recoverability_frontier": lambda path: _recoverability_frontier(
            frontier_summary, path, palette
        ),
        "return_structure": lambda path: _return_structure(
            return_summary, path, palette
        ),
        "variance_bands": lambda path: _variance_bands(variance_summary, path, palette),
        "upstream_sensitivity": lambda path: _upstream_sensitivity(
            upstream_summary, path, palette
        ),
        "fractal_dimension_sweep": lambda path: _fractal_dimension_sweep(
            fractal_dimension_summary, path, palette
        ),
        "circadian_phase_signal": lambda path: _circadian_phase_signal(
            circadian_phase_summary, path, palette
        ),
        "partial_observability": lambda path: _partial_observability(
            partial_obs_summary, path, palette
        ),
        "sequence_baselines": lambda path: _sequence_baselines(
            sequence_baseline_summary, path, palette
        ),
        "missingness_model": lambda path: _missingness_model(
            missingness_summary, path, palette
        ),
        "model_gate_matrix": lambda path: _model_gate_matrix(
            model_gate_summary, path, palette
        ),
        "m10_forward_gates": lambda path: _m10_forward_gates(
            forward_gate_summary, path, palette
        ),
        "factorial_pymdp_modes": lambda path: _factorial_pymdp_modes(
            factorial_summary, path, palette
        ),
        "latent_mode_recovery": lambda path: _latent_mode_recovery(
            factorial_summary, path, palette
        ),
        "factorial_model_gate_matrix": lambda path: _factorial_model_gate_matrix(
            factorial_summary, path, palette
        ),
        "efe_location_frontier": lambda path: _efe_location_frontier(
            efe_location_summary, path
        ),
        "linkage_risk_frontier": lambda path: _linkage_risk_frontier(
            linkage_risk_summary, path
        ),
        "poisoning_robustness": lambda path: _poisoning_robustness(
            poisoning_summary, path
        ),
        "privacy_frontier": lambda path: _privacy_frontier(
            privacy_summary, path, palette
        ),
        "hardened_anomaly_detection": lambda path: _hardened_anomaly_detection(
            hardened_anomaly_summary, path, palette
        ),
        "sota_leaderboard": lambda path: _sota_leaderboard(sota_summary, path, palette),
        "aif_regime_map": lambda path: _aif_regime_map(
            aif_regime_summary, multiplicity_summary, path, palette
        ),
        "fair_inference": lambda path: _fair_inference(fair_summary, path, palette),
        "fair_power_regime": lambda path: _fair_power_regime(
            power_summary, path, palette
        ),
        "multiplicity_ledger": lambda path: _multiplicity_ledger(
            multiplicity_summary, path, palette
        ),
        "confound_taxonomy": lambda path: _confound_taxonomy(
            confound_taxonomy_summary, path, palette
        ),
        "regime_mechanism_synthesis": lambda path: _regime_mechanism_synthesis(
            regime_mechanism_summary, path, palette
        ),
        "regime_switch_phase_diagnostic": lambda path: _regime_switch_phase_diagnostic(
            regime_switch_phase_summary, path, palette
        ),
        "privacy_utility_frontier": lambda path: _privacy_utility_frontier(
            privacy_utility_summary, path, palette
        ),
        "privacy_utility_repeats": lambda path: _privacy_utility_repeats(
            privacy_utility_repeats_summary, path, palette
        ),
        "synthetic_task_utility_privacy": lambda path: _synthetic_task_utility_privacy(
            synthetic_task_privacy_summary, path, palette
        ),
        "model_comparison_under_noise": lambda path: _model_comparison_under_noise(
            model_comparison_summary, path, palette
        ),
        "aif_deepening_candidates": lambda path: _aif_deepening_candidates(
            aif_deepening_summary, path, palette
        ),
        "dataflow": lambda path: _dataflow(path, palette),
    }
    registry_entries: list[FigureRegistryEntry] = []
    artifacts: dict[str, Path] = {}
    for key in enabled:
        if key not in generators:
            raise ValueError(f"unknown figure key: {key}")
        filename = f"{key}.png"
        path = target / filename
        generators[key](path)
        artifacts[key] = path
        registry_entries.append(
            _registry_entry(
                key,
                filename,
                run_path.name,
                context,
                privacy_utility_summary,
                fractal_dimension_summary,
            )
        )
        _set_metadata(path, dpi)
    contact_sheet_path = _write_contact_sheet(target, registry_entries, artifacts)
    artifacts["contact_sheet"] = contact_sheet_path
    return _write_manifest_files(
        target,
        registry_entries,
        artifacts,
        enabled,
        dpi,
        palette,
        project_root=run_path.parents[2],
    )


def _remove_stale_review_artifacts(target: Path) -> None:
    """Remove explicitly temporary review crops before writing a canonical set.

    Human review crops use the ``_tmp_`` prefix by convention.  They are useful
    during inspection but are never manuscript figures; removing only that
    reserved prefix keeps regeneration deterministic without hiding arbitrary
    unregistered PNGs, which the visual-quality audit still reports.
    """

    for path in target.glob("_tmp_*.png"):
        path.unlink()


def _write_manifest_files(
    target: Path,
    registry_entries: list[FigureRegistryEntry],
    artifacts: dict[str, Path],
    enabled: tuple[str, ...],
    dpi: int,
    palette: dict[str, str],
    *,
    project_root: Path,
) -> dict[str, Path]:
    """Write the figure registry, style-provenance, and manifest JSON files.

    Mutates and returns ``artifacts`` with the three written paths added; the
    manifest-writing tail previously lived inline at the end of
    ``generate_figures``.
    """
    registry_path = target / "figure_registry.json"
    manifest_path = target / "figure_manifest.json"
    style_path = target / "style_provenance.json"
    registry_rows: list[dict[str, object]] = []
    for entry in registry_entries:
        row: dict[str, object] = dict(entry.__dict__)
        image_sha = _sha256_file(artifacts[entry.key])
        source_path = project_root / entry.source
        source_sha = _sha256_file(source_path) if source_path.is_file() else None
        binding = hashlib.sha256()
        binding.update(entry.key.encode("utf-8"))
        binding.update(entry.caption.encode("utf-8"))
        binding.update(image_sha.encode("ascii"))
        binding.update((source_sha or "non-file-source").encode("ascii"))
        row.update(
            {
                "image_sha256": image_sha,
                "source_sha256": source_sha,
                "binding_sha256": binding.hexdigest(),
            }
        )
        registry_rows.append(row)
    registry_payload = {
        "schema": "active-fractal-rabbit-figure-registry-v2",
        "entries": registry_rows,
        # The sibling template render reads a label-keyed accessibility array
        # from this same file. Manuscript labels are the hyphenated key with a
        # fig: prefix, so the two surfaces stay bound to one registry.
        "figures": [
            {
                "label": "fig:" + str(row["key"]).replace("_", "-"),
                "filename": row["filename"],
                "alt_text": row["alt_text"],
                "long_description": row["caption"],
            }
            for row in registry_rows
        ],
    }
    registry_path.write_text(
        json.dumps(registry_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    style_path.write_text(
        json.dumps(
            {
                "schema": "active-fractal-rabbit-style-provenance-v1",
                "backend": matplotlib.get_backend(),
                "dpi": dpi,
                "enabled_figures": list(enabled),
                "font_contract": font_contract_payload(),
                "palette": palette,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {key: str(path) for key, path in artifacts.items()}
            | {"registry": str(registry_path), "style_provenance": str(style_path)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts["registry"] = registry_path
    artifacts["manifest"] = manifest_path
    artifacts["style_provenance"] = style_path
    return artifacts


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_contact_sheet(
    target: Path,
    registry_entries: list[FigureRegistryEntry],
    artifacts: dict[str, Path],
) -> Path:
    """Write a deterministic all-figure review surface for manual inspection."""

    entries = [entry for entry in registry_entries if entry.key in artifacts]
    ncols = 3
    nrows = max(1, (len(entries) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.4 * nrows))
    axes_array = np.atleast_1d(axes).ravel()
    for axis, entry in zip(axes_array, entries):
        image = plt.imread(artifacts[entry.key])
        axis.imshow(image)
        axis.set_title(
            _clean_label(entry.key, width=28),
            fontsize=SMALL_LABEL_FONT_SIZE,
            fontweight="bold",
            loc="left",
            pad=8,
        )
        axis.text(
            0.0,
            -0.06,
            _contact_sheet_note(entry),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=SMALL_LABEL_FONT_SIZE,
            color="#344054",
            clip_on=False,
        )
        axis.axis("off")
    for axis in axes_array[len(entries) :]:
        axis.axis("off")
    fig.suptitle(
        "Registered manuscript figures: source-aware manual review sheet",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=2.0, w_pad=1.8)
    contact_sheet_path = target / "figure_contact_sheet.png"
    fig.savefig(contact_sheet_path, dpi=120)
    plt.close(fig)
    return contact_sheet_path


def _contact_sheet_note(entry: FigureRegistryEntry) -> str:
    source_parts = Path(entry.source).parts
    if "runs" in source_parts:
        source = "/".join(source_parts[source_parts.index("runs") + 1 :])
    elif "figures" in source_parts:
        source = "/".join(source_parts[source_parts.index("figures") + 1 :])
    else:
        source = entry.source.replace("output/", "out/")
    boundary = "data-derived" if entry.data_derived else "context"
    return f"{boundary}; source: {_clean_label(source, width=34)}"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _figure_context(
    observations: list[dict[str, str]],
    features: list[dict[str, str]],
    inference: list[dict[str, str]],
    policy: list[dict[str, str]],
    candidate_scores: list[dict[str, object]],
    config: dict[str, object],
) -> dict[str, object]:
    observation_config = _config_group(config, "observation")
    model_config = _config_group(config, "model")
    thresholds = (
        observation_config.get("speed_thresholds_km_per_day", [10.0, 50.0, 150.0])
        if isinstance(observation_config, dict)
        else [10.0, 50.0, 150.0]
    )
    smoothing = (
        model_config.get("smoothing", "unknown")
        if isinstance(model_config, dict)
        else "unknown"
    )
    controls = (
        model_config.get("controls", []) if isinstance(model_config, dict) else []
    )
    return {
        "n_observations": len(observations),
        "n_travellers": len({row["ID"] for row in observations}),
        "n_features": len(features),
        "n_inference": len(inference),
        "n_policy": len(policy),
        "n_candidates": len(candidate_scores),
        "first_observations": sum(
            int(row.get("is_first_observation", "0")) for row in features
        ),
        "burst_rule": observation_config.get("burst_threshold_rule", "median_positive")
        if isinstance(observation_config, dict)
        else "median_positive",
        "speed_thresholds": ", ".join(f"{float(item):.0f}" for item in thresholds),
        "smoothing": smoothing,
        "control_count": len(controls),
    }


def _config_group(config: dict[str, object], name: str) -> dict[str, object]:
    if not isinstance(config, dict):
        return {}
    direct = config.get(name)
    if isinstance(direct, dict):
        return direct
    inference = config.get("inference")
    if isinstance(inference, dict):
        nested = inference.get(name)
        if isinstance(nested, dict):
            return nested
    return {}


def _waypoint_paths(
    rows: list[dict[str, str]], path: Path, palette: dict[str, str]
) -> None:
    """Trajectory overview that stays readable as the report count grows.

    Per-traveller line connects consecutive reports (a visual aid, not
    continuous tracking); marker fill encodes report time on a shared
    colorbar; start and final reports get distinct markers instead of
    per-point index labels, which stop scaling beyond a handful of rows.
    """

    by_id: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_id.setdefault(row["ID"], []).append(row)
    all_days = [float(row["Days"]) for row in rows]
    vmin, vmax = min(all_days), max(all_days)
    line_colors = [
        palette["blue"],
        palette["green"],
        palette["orange"],
        palette["purple"],
        palette["gray"],
    ]
    fig, axis = plt.subplots(figsize=(8.6, 5.6))
    scatter = None
    for index, (traveller_id, traveller_rows) in enumerate(
        sorted(by_id.items(), key=lambda item: int(float(item[0])))
    ):
        ordered = sorted(traveller_rows, key=lambda row: float(row["Days"]))
        x = [float(row["x(km)"]) for row in ordered]
        y = [float(row["y(km)"]) for row in ordered]
        days = [float(row["Days"]) for row in ordered]
        color = line_colors[index % len(line_colors)]
        axis.plot(
            x,
            y,
            color=color,
            linewidth=1.1,
            alpha=0.45,
            zorder=1,
            label=f"traveller {traveller_id}",
        )
        scatter = axis.scatter(
            x,
            y,
            c=days,
            cmap=SEQUENTIAL_CMAP,
            vmin=vmin,
            vmax=vmax,
            s=34,
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        axis.scatter(
            [x[0]],
            [y[0]],
            marker="^",
            s=90,
            facecolor="none",
            edgecolor=color,
            linewidth=1.6,
            zorder=3,
        )
        axis.scatter(
            [x[-1]],
            [y[-1]],
            marker="*",
            s=140,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
    axis.set_xlabel("x (km)")
    axis.set_ylabel("y (km)")
    axis.set_title(
        "Sporadic waypoint reports by traveller (^ first report, * last report)"
    )
    axis.grid(True, linewidth=0.4, alpha=0.3)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.20, 1.0),
        fontsize=LEGEND_FONT_SIZE,
        framealpha=0.88,
    )
    if scatter is not None:
        fig.colorbar(scatter, ax=axis, label="report time (days)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _spatial_bins(
    rows: list[dict[str, str]], path: Path, palette: dict[str, str]
) -> None:
    x = np.array([float(row["x(km)"]) for row in rows])
    y = np.array([float(row["y(km)"]) for row in rows])
    states = np.array([int(row["spatial_state"]) for row in rows])
    x_bins = np.array([int(row["x_bin"]) for row in rows])
    y_bins = np.array([int(row["y_bin"]) for row in rows])
    bins = max(int(x_bins.max()) + 1, int(y_bins.max()) + 1)
    plt.figure(figsize=(6.4, 5.2))
    plt.scatter(
        x, y, c=states, cmap=SEQUENTIAL_CMAP, s=48, edgecolor="white", linewidth=0.4
    )
    x_edges = np.linspace(float(x.min()), float(x.max()), bins + 1)
    y_edges = np.linspace(float(y.min()), float(y.max()), bins + 1)
    for x_edge in x_edges:
        plt.axvline(x_edge, color=palette["gray"], linewidth=0.6, alpha=0.45)
    for y_edge in y_edges:
        plt.axhline(y_edge, color=palette["gray"], linewidth=0.6, alpha=0.45)
    for y_idx in range(bins):
        for x_idx in range(bins):
            cell = y_idx * bins + x_idx
            plt.text(
                (x_edges[x_idx] + x_edges[x_idx + 1]) / 2,
                (y_edges[y_idx] + y_edges[y_idx + 1]) / 2,
                str(cell),
                ha="center",
                va="center",
                fontsize=SMALL_LABEL_FONT_SIZE,
                color="black",
                alpha=0.9,
                # White stroke halo keeps the cell index legible on both the dark
                # and bright ends of viridis without blanking the scatter points.
                path_effects=[
                    path_effects.withStroke(linewidth=2.0, foreground="white")
                ],
            )
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.title("Spatial discretization overlay")
    plt.colorbar(label="latent spatial cell")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _spatial_resolution_frontier(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    from .spatial_resolution_frontier import render_spatial_resolution_frontier

    render_spatial_resolution_frontier(summary, path, palette)


def _reporting_bursts(
    rows: list[dict[str, str]],
    path: Path,
    palette: dict[str, str],
    config: dict[str, object],
) -> None:
    intervals = np.array(
        [float(row["interval_days"]) for row in rows if float(row["interval_days"]) > 0]
    )
    bursts = sum(int(row["burst_category"]) for row in rows)
    non_bursts = len(rows) - bursts
    first_observations = sum(int(row.get("is_first_observation", "0")) for row in rows)
    observation_config = _config_group(config, "observation")
    rule = observation_config.get("burst_threshold_rule", "median_positive")
    fixed = observation_config.get("burst_fixed_days")
    cutoff = (
        float(fixed)
        if rule == "fixed_days" and fixed is not None
        else (float(np.median(intervals)) if intervals.size else 0.0)
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    if cutoff > 0 and intervals.size:
        # Bin edges must resolve the burst cutoff: the caption's central
        # comparison is intervals *against the configured cutoff*, so the
        # cutoff is an explicit bin boundary and below/above mass cannot be
        # merged into one bar (a coarse 8-bin histogram hid the ~50/50 split).
        upper = max(float(intervals.max()), cutoff * 1.01)
        hist_bins = np.unique(
            np.concatenate([[0.0, cutoff], np.linspace(cutoff, upper, 21)[1:]])
        )
    else:
        hist_bins = min(8, max(2, len(intervals)))
    axes[0].hist(
        intervals,
        bins=hist_bins,
        color=palette["blue"],
        edgecolor="white",
    )
    if cutoff > 0:
        axes[0].axvline(
            cutoff,
            color=palette["orange"],
            linestyle="--",
            linewidth=1.2,
            label=f"cutoff {cutoff:.2g} d",
        )
        axes[0].legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    axes[0].set_xlabel("inter-report interval (days)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Intervals", loc="left")
    axes[1].bar(
        ["non-burst", "burst"],
        [non_bursts, bursts],
        color=[palette["gray"], palette["orange"]],
    )
    axes[1].set_ylabel("observations")
    axes[1].set_title("Burst labels", loc="left")
    # Headroom so the count annotation sits in clear space above the bars
    # rather than overprinting the taller of the two.
    axes[1].set_ylim(0, max(non_bursts, bursts) * 1.18)
    axes[1].text(
        0.02,
        0.97,
        f"first obs = {first_observations}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=SMALL_LABEL_FONT_SIZE,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    fig.suptitle("Sporadic reporting diagnostics")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _speed_categories(
    rows: list[dict[str, str]],
    path: Path,
    palette: dict[str, str],
    config: dict[str, object],
) -> None:
    categories = np.array([int(row["speed_category"]) for row in rows])
    speeds = np.array([float(row["speed_km_per_day"]) for row in rows])
    observation_config = _config_group(config, "observation")
    thresholds = observation_config.get(
        "speed_thresholds_km_per_day", [10.0, 50.0, 150.0]
    )
    counts = [int(np.sum(categories == idx)) for idx in range(4)]
    labels = [
        f"< {float(thresholds[0]):.0f}",
        f"{float(thresholds[0]):.0f}-{float(thresholds[1]):.0f}",
        f"{float(thresholds[1]):.0f}-{float(thresholds[2]):.0f}",
        f">= {float(thresholds[2]):.0f}",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    axes[0].bar(labels, counts, color=palette["green"])
    axes[0].set_xlabel("speed category (km/day)")
    axes[0].set_ylabel("observations")
    axes[0].set_title("Categorical speeds", loc="left")
    axes[0].tick_params(axis="x", labelrotation=20)
    axes[1].scatter(
        range(len(speeds)), speeds, c=categories, cmap=SEQUENTIAL_CMAP, s=32
    )
    # Threshold labels moved OUTSIDE the axes entirely (right-margin legend),
    # not inline at each line's y-position: the scatter is densely populated
    # across the full y-range on this fixture, so any inline anchor - above
    # or below its own line, alternated or not - lands on top of scatter
    # points for at least one of the three thresholds. A legend anchored past
    # the axes' right edge can never collide with plotted data.
    threshold_handles = []
    for threshold in thresholds:
        threshold_value = float(threshold)
        axes[1].axhline(
            threshold_value,
            color=palette["gray"],
            linestyle="--",
            linewidth=0.8,
            alpha=0.65,
        )
        threshold_handles.append(
            Line2D(
                [0],
                [0],
                color=palette["gray"],
                linestyle="--",
                linewidth=1.2,
                label=f"{threshold_value:.0f} km/day",
            )
        )
    axes[1].legend(
        handles=threshold_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=SMALL_LABEL_FONT_SIZE,
        frameon=False,
        title="thresholds",
        title_fontsize=SMALL_LABEL_FONT_SIZE,
    )
    axes[1].set_xlabel("ordered waypoint")
    axes[1].set_ylabel("km/day")
    axes[1].set_title("Derived speeds with thresholds", loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _transition_heatmap(
    rows: list[dict[str, str]], path: Path, palette: dict[str, str]
) -> None:
    states = np.array([int(row["spatial_state"]) for row in rows])
    first = np.array([int(row.get("is_first_observation", "0")) for row in rows])
    state_count = int(states.max()) + 1
    counts = np.zeros((state_count, state_count), dtype=float)
    for idx in range(len(states) - 1):
        # First-observation boundary guard, mirroring pomdp._transition_tensor:
        # crossing a traveller boundary would inject a spurious
        # last-cell-of-A -> first-cell-of-B count that the model's B-tensor
        # estimator explicitly excludes.
        if first[idx + 1]:
            continue
        counts[states[idx + 1], states[idx]] += 1.0
    totals = counts.sum(axis=0, keepdims=True)
    matrix = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    fig, ax = plt.subplots(figsize=(8.6, 6.2), constrained_layout=True)
    im = ax.imshow(matrix, cmap=SEQUENTIAL_CMAP, vmin=0.0)
    # Label the diagonal plus only the largest off-diagonal counts: at the
    # font-contract minimum size, labelling every count>=3 cell made adjacent
    # labels run together into false multi-digit strings and clipped at the
    # axes edges.
    off_diagonal = counts.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    positive_off = np.sort(off_diagonal[off_diagonal > 0])[::-1]
    top_k = 8
    off_threshold = (
        float(positive_off[min(top_k, positive_off.size) - 1])
        if positive_off.size
        else np.inf
    )
    for row_idx in range(state_count):
        for col_idx in range(state_count):
            count = int(counts[row_idx, col_idx])
            if count < 3:
                continue
            if row_idx != col_idx and count < off_threshold:
                continue
            # A 3+ digit count ("346") is wider than the fixed per-cell pixel
            # budget at the base font size and overflows into a neighbouring
            # cell's label ("8" reads as concatenated "3468"); shrink the font
            # for wider labels so every count stays inside its own cell.
            digits = len(str(count))
            label_fontsize = (
                SMALL_LABEL_FONT_SIZE
                if digits <= 2
                else max(
                    SMALL_LABEL_FONT_SIZE - 2.0 * (digits - 2), TICK_FONT_SIZE * 0.6
                )
            )
            ax.text(
                col_idx,
                row_idx,
                str(count),
                ha="center",
                va="center",
                color="white",
                fontsize=label_fontsize,
            )
    ticks = np.arange(state_count)
    stride = max(1, int(np.ceil(state_count / 8)))
    ax.set_xticks(ticks[::stride])
    ax.set_yticks(ticks[::stride])
    ax.set_xlabel("previous latent cell")
    ax.set_ylabel("next latent cell")
    ax.set_title(
        "Empirical transition probabilities\n"
        "(column-normalized; counts labelled on the diagonal and largest off-diagonal cells)"
    )
    fig.colorbar(im, ax=ax, label="probability", pad=0.02)
    fig.savefig(path)
    plt.close(fig)


def _likelihood_panels(
    rows: list[dict[str, str]], path: Path, palette: dict[str, str]
) -> None:
    states = np.array([int(row["spatial_state"]) for row in rows])
    state_count = int(states.max()) + 1
    modalities = [
        ("spatial_state", "location"),
        ("time_bin", "time"),
        ("speed_category", "speed"),
        ("burst_category", "burst"),
    ]
    fig, axes = plt.subplots(
        1, len(modalities), figsize=(12.4, 3.8), constrained_layout=True
    )
    for axis, (column, title) in zip(axes, modalities):
        obs = np.array([int(row[column]) for row in rows])
        counts = np.zeros((int(obs.max()) + 1, state_count), dtype=float)
        for state, observed in zip(states, obs):
            counts[observed, state] += 1.0
        totals = counts.sum(axis=0, keepdims=True)
        matrix = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        im = axis.imshow(
            matrix,
            aspect="auto",
            cmap=SEQUENTIAL_CMAP,
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_title(title)
        axis.set_xlabel("state")
        axis.set_ylabel("obs")
    fig.colorbar(
        im, ax=axes.ravel().tolist(), shrink=0.76, pad=0.02, label="P(obs | state)"
    )
    fig.suptitle("Observation likelihood diagnostics by modality (column-normalized)")
    fig.savefig(path)
    plt.close(fig)


def _posterior_entropy(
    rows: list[dict[str, str]],
    metrics: dict[str, object],
    spatial: dict[str, object],
    path: Path,
    palette: dict[str, str],
) -> None:
    t = np.array([int(row["t"]) for row in rows])
    loss = np.array([float(row["predictive_loss"]) for row in rows])
    entropy = np.array([float(row["posterior_entropy"]) for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.2), sharex=True)
    axes[0].plot(t, loss, color=palette["blue"], linewidth=1.0, label="predictive loss")
    window = min(48, int(loss.size)) if loss.size else 1
    if window > 1:
        axes[0].plot(
            t,
            _rolling_mean(loss, window),
            color=palette["green"],
            linewidth=1.4,
            label=f"rolling mean ({window} reports)",
        )
    chance_floor = metrics.get("chance_floor_joint_loss")
    if isinstance(chance_floor, int | float) and np.isfinite(float(chance_floor)):
        axes[0].axhline(
            float(chance_floor),
            color=palette["gray"],
            linestyle="--",
            linewidth=1.0,
            label="uniform-chance joint loss",
        )
        axes[0].annotate(
            "uniform-chance\njoint loss",
            xy=(float(t[-1]) if t.size else 0.0, float(chance_floor)),
            xytext=(-8, 6),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=SMALL_LABEL_FONT_SIZE,
            color=palette["gray"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
    axes[0].set_ylabel("loss (nats)")
    axes[0].set_title("One-step predictive loss", loc="left")
    axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=LEGEND_FONT_SIZE,
        framealpha=0.88,
    )
    axes[1].plot(
        t,
        entropy,
        color=palette["orange"],
        linewidth=1.0,
        label="posterior entropy",
    )
    axes[1].set_xlabel("waypoint index")
    axes[1].set_ylabel("entropy (nats)")
    axes[1].set_title("Posterior uncertainty", loc="left")
    state_count = spatial.get("default_state_count")
    max_entropy = (
        float(np.log(int(state_count)))
        if isinstance(state_count, int | float) and int(state_count) > 1
        else None
    )
    mean_entropy = float(np.mean(entropy)) if entropy.size else float("nan")
    if np.isfinite(mean_entropy):
        ceiling_text = (
            f"\nmax ln {int(state_count)}={max_entropy:.2f}"
            if max_entropy is not None
            else ""
        )
        axes[1].annotate(
            f"posterior nearly deterministic\nmean={mean_entropy:.2f} nats{ceiling_text}",
            xy=(0.02, 0.90),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=SMALL_LABEL_FONT_SIZE,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
        )
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(path)
    plt.close(fig)


def _policy_proxy(
    rows: list[dict[str, str]], path: Path, palette: dict[str, str]
) -> None:
    if not rows:
        raise ValueError("policy trace is empty")
    t = np.array([int(row["t"]) for row in rows])
    proxy_columns = [key for key in rows[0] if key.startswith("proxy_score_")]
    entropy_columns = [key for key in rows[0] if key.startswith("predictive_entropy_")]
    preference_columns = [
        key for key in rows[0] if key.startswith("preference_divergence_")
    ]
    posterior_columns = [key for key in rows[0] if key.startswith("policy_posterior_")]
    fig, axes = plt.subplots(3, 1, figsize=(9.8, 6.8), sharex=True)
    colors = [palette["blue"], palette["green"], palette["orange"], palette["purple"]]
    proxy_values = np.array(
        [[float(row[column]) for column in proxy_columns] for row in rows]
    )
    proxy_spread = proxy_values.max(axis=1) - proxy_values.min(axis=1)
    proxy_tied = bool(
        proxy_values.size
        and np.max(proxy_values.max(axis=1) - proxy_values.min(axis=1)) < 1e-9
    )
    if proxy_tied and proxy_columns:
        axes[0].plot(
            t,
            proxy_values[:, 0],
            label="both policies (identical scores)",
            color=palette["blue"],
            linewidth=1.8,
        )
        axes[0].annotate(
            "max-min = 0 at every step\nEFE proxy cannot distinguish policies",
            xy=(0.02, 0.92),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=SMALL_LABEL_FONT_SIZE,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
        )
    else:
        for idx, column in enumerate(proxy_columns):
            axes[0].plot(
                t,
                [float(row[column]) for row in rows],
                label=column.replace("proxy_score_", ""),
                color=colors[idx % len(colors)],
            )
    axes[1].plot(
        t,
        proxy_spread,
        color=palette["purple"],
        marker="o",
        label="max-min proxy score",
    )
    axes[1].axhline(0.0, color=palette["gray"], linewidth=0.8, linestyle="--")
    if entropy_columns and preference_columns:
        first_entropy = entropy_columns[0]
        first_preference = preference_columns[0]
        first_name = first_entropy.replace("predictive_entropy_", "")
        axes[1].plot(
            t,
            [float(row[first_entropy]) for row in rows],
            label=f"{first_name} predictive entropy",
            color=palette["blue"],
            alpha=0.55,
        )
        axes[1].plot(
            t,
            [float(row[first_preference]) for row in rows],
            label=f"{first_name} preference divergence",
            color=palette["orange"],
            alpha=0.55,
        )
    posterior_values = np.array(
        [[float(row[column]) for column in posterior_columns] for row in rows]
    )
    uniform = 1.0 / len(posterior_columns)
    tied = bool(np.max(np.abs(posterior_values - uniform)) < 1e-3)
    if tied:
        axes[2].plot(
            t,
            posterior_values[:, 0],
            label=f"both policies (uniform {uniform:.1f})",
            color=palette["blue"],
            linewidth=1.8,
        )
        axes[2].annotate(
            f"posterior fixed at uniform {uniform:.1f}",
            xy=(0.02, 0.88),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=SMALL_LABEL_FONT_SIZE,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
        )
    else:
        for idx, column in enumerate(posterior_columns):
            axes[2].plot(
                t,
                [float(row[column]) for row in rows],
                label=column.replace("policy_posterior_", ""),
                color=colors[idx % len(colors)],
            )
    axes[0].set_ylabel("proxy score")
    axes[0].set_title("Total proxy score", loc="left")
    axes[0].legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    axes[1].set_ylabel("nats")
    axes[1].set_title("Score spread and first-control components", loc="left")
    axes[1].legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=LEGEND_FONT_SIZE,
        framealpha=0.88,
    )
    axes[2].set_ylabel("policy posterior")
    axes[2].set_xlabel("waypoint index", labelpad=8)
    axes[2].legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    fig.suptitle("Policy proxy diagnostics" + (" (policies tied)" if tied else ""))
    fig.subplots_adjust(left=0.10, right=0.72, top=0.90, bottom=0.10, hspace=0.48)
    fig.savefig(path)
    plt.close(fig)


def _candidate_scores(
    rows: list[dict[str, object]], path: Path, palette: dict[str, str]
) -> None:
    names = [str(row["name"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    ranks = [int(row["rank"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9.8, 4.8), constrained_layout=True)
    # Colour the selected (rank-1) candidate so visual salience matches the
    # closed-loop choice; the rest stay neutral grey.
    best_rank = min(ranks) if ranks else 1
    bar_colors = [
        palette["green"] if r == best_rank else palette["gray"] for r in ranks
    ]
    bars = ax.bar(names, scores, color=bar_colors, width=0.58)
    if scores:
        top = max(scores)
        ax.set_ylim(0.0, top * 1.18 if top > 0 else 1.0)
    for bar, rank, score in zip(bars, ranks, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(scores) * 0.018,
            f"rank {rank}\n{score:.3f}" + ("\n(selected)" if rank == best_rank else ""),
            ha="center",
            va="bottom",
            fontsize=SMALL_LABEL_FONT_SIZE,
            fontweight="semibold",
        )
    ax.set_ylabel("candidate score")
    if rows:
        objective = str(rows[0].get("objective", "objective"))
        metric = float(rows[0].get("metric_value", 0.0))
        ax.set_xlabel(
            f"score = metric x reporting ratio\ncurrent {objective}={metric:.4g}"
        )
    ax.set_title("Closed-loop candidate ranking", loc="left")
    ax.grid(True, axis="y")
    fig.savefig(path)
    plt.close(fig)


def _negative_controls(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    controls = summary.get("controls")
    if not controls or not isinstance(controls, list):
        raise ValueError(
            "negative_controls figure requires output/runs/negative_controls_summary.json; "
            "run scripts/05_run_negative_controls.py before generating figures"
        )
    intact_metrics = summary.get("intact_metrics", {})
    intact_loss = (
        float(intact_metrics.get("mean_predictive_loss", 0.0))
        if isinstance(intact_metrics, dict)
        else 0.0
    )
    names = ["intact"] + [str(item["name"]) for item in controls]
    losses = [intact_loss] + [float(item["mean_predictive_loss"]) for item in controls]
    passed = [True] + [bool(item["passed"]) for item in controls]
    spatial = [int(item["occupied_spatial_states"]) for item in controls]
    speed = [int(item["occupied_speed_categories"]) for item in controls]
    control_names = [str(item["name"]) for item in controls]

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 4.8), constrained_layout=True)
    colors = [
        palette["blue"]
        if name == "intact"
        else (palette["green"] if ok else palette["orange"])
        for name, ok in zip(names, passed)
    ]
    bars = axes[0].bar(names, losses, color=colors)
    for bar, ok in zip(bars[1:], passed[1:]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            "pass" if ok else "FAIL",
            ha="center",
            va="bottom",
            fontsize=SMALL_LABEL_FONT_SIZE,
        )
    # Reference line at the intact loss so each control's degradation reads as
    # its height above the baseline the controls are meant to destroy.
    axes[0].axhline(
        intact_loss,
        color=palette["gray"],
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        zorder=1.5,
    )
    axes[0].text(
        len(names) - 0.4,
        intact_loss,
        "intact baseline",
        ha="right",
        va="bottom",
        fontsize=SMALL_LABEL_FONT_SIZE,
        color=palette["gray"],
    )
    axes[0].set_ylabel("mean predictive loss (nats)")
    axes[0].set_title("Destroyed-structure loss", loc="left")
    axes[0].tick_params(axis="x", labelrotation=30)
    for label in axes[0].get_xticklabels():
        label.set_ha("right")

    positions = np.arange(len(control_names))
    width = 0.38
    axes[1].bar(
        positions - width / 2,
        spatial,
        width,
        color=palette["purple"],
        label="occupied spatial states",
    )
    axes[1].bar(
        positions + width / 2,
        speed,
        width,
        color=palette["gray"],
        label="occupied speed categories",
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(control_names, rotation=30, ha="right")
    axes[1].set_ylabel("occupied count")
    axes[1].set_title("Occupancy collapse", loc="left")
    axes[1].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.52),
        ncol=1,
        fontsize=LEGEND_FONT_SIZE,
        framealpha=0.9,
    )
    fig.suptitle("Negative-control falsification lane (fixture diagnostics)")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _pymdp_agreement(
    inference: list[dict[str, str]],
    pymdp_trace: list[dict[str, str]],
    run_metrics: dict[str, object],
    path: Path,
    palette: dict[str, str],
) -> None:
    if not pymdp_trace:
        raise ValueError(
            "pymdp_agreement figure requires pymdp_trace.csv; set model.engine to 'both' "
            "and rerun the pipeline before generating figures"
        )
    t = np.array([int(row["t"]) for row in pymdp_trace])
    ref_loss = np.array([float(row["predictive_loss"]) for row in inference])
    ref_entropy = np.array([float(row["posterior_entropy"]) for row in inference])
    pym_loss = np.array([float(row["pymdp_predictive_loss"]) for row in pymdp_trace])
    pym_entropy = np.array(
        [float(row["pymdp_posterior_entropy"]) for row in pymdp_trace]
    )
    q_pi_columns = [key for key in pymdp_trace[0] if key.startswith("pymdp_q_pi_")]
    efe_columns = [key for key in pymdp_trace[0] if key.startswith("pymdp_neg_efe_")]
    colors = [palette["blue"], palette["green"], palette["orange"], palette["purple"]]

    fig, axes = plt.subplots(3, 1, figsize=(12.2, 9.6), sharex=True)
    smooth_window = max(12, min(48, len(t) // 20))
    axes[0].plot(
        t,
        ref_loss,
        color=palette["blue"],
        linewidth=0.6,
        alpha=0.18,
        label="reference raw",
    )
    axes[0].plot(
        t,
        pym_loss,
        color=palette["orange"],
        linewidth=0.6,
        alpha=0.18,
        label="pymdp raw",
    )
    axes[0].plot(
        t,
        _rolling_mean(ref_loss, smooth_window),
        color=palette["blue"],
        linewidth=1.8,
        label=f"reference rolling mean ({smooth_window})",
    )
    axes[0].plot(
        t,
        _rolling_mean(pym_loss, smooth_window),
        color=palette["orange"],
        linewidth=1.8,
        label=f"pymdp rolling mean ({smooth_window})",
    )
    axes[0].set_ylabel("loss")
    axes[0].set_title(
        "Predictive loss (nats): raw trace plus rolling mean",
        loc="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
    )
    axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=LEGEND_FONT_SIZE,
        framealpha=0.86,
        ncol=1,
    )
    axes[1].plot(
        t,
        ref_entropy,
        color=palette["blue"],
        linewidth=0.6,
        alpha=0.18,
        label="reference raw",
    )
    axes[1].plot(
        t,
        pym_entropy,
        color=palette["orange"],
        linewidth=0.6,
        alpha=0.18,
        label="pymdp raw",
    )
    axes[1].plot(
        t,
        _rolling_mean(ref_entropy, smooth_window),
        color=palette["blue"],
        linewidth=1.8,
        label="reference rolling mean",
    )
    axes[1].plot(
        t,
        _rolling_mean(pym_entropy, smooth_window),
        color=palette["orange"],
        linewidth=1.8,
        label="pymdp rolling mean",
    )
    axes[1].set_ylabel("entropy")
    axes[1].set_title(
        "Posterior entropy (nats): closed-loop prior divergence",
        loc="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
    )
    # Print the two headline agreement/divergence deltas on the figure itself so
    # the caption's numbers are depicted (not just described): the matched-prior
    # agreement validates inference; the closed-loop divergence is policy-induced.
    exact_delta = run_metrics.get("pymdp_exact_inference_max_abs_delta")
    loop_delta = run_metrics.get("pymdp_reference_max_abs_posterior_delta")
    if isinstance(exact_delta, int | float) and isinstance(loop_delta, int | float):
        axes[1].annotate(
            "matched-prior agreement: "
            f"max abs delta {float(exact_delta):.2e}\n"
            "closed-loop divergence: "
            f"max abs delta {float(loop_delta):.2e}",
            xy=(0.015, 0.04),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
            color="#222222",
            bbox={
                "facecolor": "white",
                "edgecolor": "#BBBBBB",
                "alpha": 0.88,
                "pad": 2.5,
            },
        )
    # No axes[1] legend: it duplicated axes[0]'s (same labels, same anchor) and
    # its box overprinted the agreement/divergence delta annotations that are the
    # whole point of this panel. The shared series legend on axes[0] covers it.
    for idx, column in enumerate(q_pi_columns):
        axes[2].plot(
            t,
            [float(row[column]) for row in pymdp_trace],
            color=colors[idx % len(colors)],
            linewidth=1.2,
            label=column.replace("pymdp_q_pi_", "q_pi "),
        )
    for idx, column in enumerate(efe_columns):
        axes[2].plot(
            t,
            [float(row[column]) for row in pymdp_trace],
            color=colors[idx % len(colors)],
            linestyle="--",
            alpha=0.62,
            linewidth=1.2,
            label=column.replace("pymdp_neg_efe_", "neg-EFE "),
        )
    axes[2].set_xlabel("waypoint index")
    axes[2].set_ylabel("posterior\n/ neg-EFE")
    axes[2].set_title(
        "Policy posterior and neg-EFE surfaces",
        loc="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
    )
    axes[2].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        fontsize=LEGEND_FONT_SIZE,
        borderaxespad=0.0,
        ncol=2,
        framealpha=0.86,
    )
    fig.suptitle(
        "Real pymdp agent lane vs reference filter",
        x=0.02,
        ha="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.03, 0.14, 0.82, 0.96))
    fig.savefig(path)
    plt.close(fig)


def _pymdp_diagnostics(
    pymdp_trace: list[dict[str, str]], path: Path, palette: dict[str, str]
) -> None:
    if not pymdp_trace:
        raise ValueError(
            "pymdp_diagnostics figure requires pymdp_trace.csv; set model.engine to 'both' "
            "and rerun the pipeline before generating figures"
        )
    t = np.array([int(row["t"]) for row in pymdp_trace])
    vfe = np.array([float(row["pymdp_vfe"]) for row in pymdp_trace])
    runtime = np.array([float(row["pymdp_step_runtime_ms"]) for row in pymdp_trace])
    efe_columns = [key for key in pymdp_trace[0] if key.startswith("pymdp_neg_efe_")]
    colors = [palette["blue"], palette["green"], palette["orange"], palette["purple"]]

    fig, axes = plt.subplots(3, 1, figsize=(12.8, 11.2), sharex=True)
    axes[0].plot(
        t, vfe, color=palette["blue"], linewidth=1.4, label="variational free energy"
    )
    axes[0].set_ylabel("VFE (nats)")
    axes[0].set_title(
        "Per-step variational free energy reported by pymdp",
        loc="left",
        fontsize=PANEL_HEADING_FONT_SIZE,
    )
    axes[0].legend(loc="lower right", fontsize=LEGEND_FONT_SIZE, framealpha=0.88)
    axes[1].semilogy(
        t,
        np.maximum(runtime, 1e-3),
        color=palette["orange"],
        linewidth=1.2,
        label="step runtime",
    )
    axes[1].axhline(
        float(np.median(runtime)),
        color=palette["gray"],
        linestyle="--",
        linewidth=0.9,
        label=f"median {np.median(runtime):.1f} ms",
    )
    axes[1].set_ylabel("runtime (ms, log)")
    axes[1].set_title(
        "Per-step wall time for the real pymdp lane",
        loc="left",
        fontsize=PANEL_HEADING_FONT_SIZE,
    )
    axes[1].legend(loc="upper right", fontsize=LEGEND_FONT_SIZE, framealpha=0.88)
    for idx, column in enumerate(efe_columns):
        axes[2].plot(
            t,
            [float(row[column]) for row in pymdp_trace],
            color=colors[idx % len(colors)],
            linewidth=1.5,
            label=column.replace("pymdp_neg_efe_", ""),
        )
    axes[2].set_xlabel("waypoint index")
    axes[2].set_ylabel("neg-EFE per policy")
    axes[2].set_title(
        "Expected free energy per timestep per policy",
        loc="left",
        fontsize=PANEL_HEADING_FONT_SIZE,
    )
    axes[2].legend(loc="upper right", fontsize=LEGEND_FONT_SIZE, framealpha=0.88)
    fig.suptitle(
        "pymdp agent diagnostics",
        fontsize=CANVAS_TITLE_FONT_SIZE,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.1)
    fig.savefig(path)
    plt.close(fig)


def _state_recovery(
    ground_truth: list[dict[str, str]], path: Path, palette: dict[str, str]
) -> None:
    if not ground_truth:
        raise ValueError(
            "state_recovery figure requires ground_truth.csv; rerun the pipeline before generating figures"
        )
    t = np.array([int(row["t"]) for row in ground_truth])
    ref_mass = np.array([float(row["reference_mass_on_true"]) for row in ground_truth])
    ref_top1 = np.array([float(row["reference_top1"]) for row in ground_truth])
    has_pymdp = "pymdp_mass_on_true" in ground_truth[0]

    # Taller figure + smaller left-anchored panel titles: at 5.8in the
    # centred suptitle, the long panel title, and the rotated y-label
    # overprinted each other in the top-left corner.
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 6.8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )
    axes[0].plot(
        t, ref_mass, color=palette["blue"], marker="o", label="reference filter"
    )
    if has_pymdp:
        pym_mass = np.array([float(row["pymdp_mass_on_true"]) for row in ground_truth])
        axes[0].plot(
            t,
            pym_mass,
            color=palette["orange"],
            marker="s",
            alpha=0.8,
            label="pymdp agent",
        )
    axes[0].set_ylabel("mass on true cell")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title(
        "Ground-truth state recovery (true cell from reported coordinates)",
        loc="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
    )
    axes[0].legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    correct = ref_top1 > 0.5
    axes[1].scatter(
        t[correct],
        ref_top1[correct],
        color=palette["green"],
        s=28,
        label="top-1 correct",
    )
    if (~correct).any():
        axes[1].scatter(
            t[~correct],
            ref_top1[~correct],
            color=palette["orange"],
            s=36,
            marker="x",
            label="top-1 wrong",
        )
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["wrong", "correct"])
    axes[1].set_xlabel("waypoint index")
    axes[1].set_title(
        f"Reference top-1 accuracy = {np.mean(ref_top1):.3f}",
        loc="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
    )
    axes[1].legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    fig.suptitle("Latent-cell recovery against simulator ground truth")
    fig.savefig(path)
    plt.close(fig)


def _sparsity_recovery(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    levels = summary.get("levels")
    if not levels or not isinstance(levels, list):
        raise ValueError(
            "sparsity_recovery figure requires output/runs/sparsity_analysis.json; "
            "run scripts/08_run_sparsity_analysis.py before generating figures"
        )
    ordered = sorted(levels, key=lambda item: float(item["mean_reports_per_traveller"]))
    x = [float(item["mean_reports_per_traveller"]) for item in ordered]
    fig, axes = plt.subplots(
        2, 1, figsize=(9.4, 6.2), sharex=True, constrained_layout=True
    )
    axes[0].plot(
        x,
        [float(item["full_mean_mass_on_true_cell"]) for item in ordered],
        color=palette["blue"],
        marker="o",
        label="full model (location observed)",
    )
    # Subsample-resampling robustness band around the deep-latent reference curve
    # (min-max across independent subsamples at each fraction, clipped to [0,1]).
    # A robustness spread over this one fixture, NOT an inferential CI; it
    # collapses to a point at the densest level by construction.
    if all("latent_mean_mass_subsample_min" in item for item in ordered):
        band_lo = np.clip(
            [float(item["latent_mean_mass_subsample_min"]) for item in ordered],
            0.0,
            1.0,
        )
        band_hi = np.clip(
            [float(item["latent_mean_mass_subsample_max"]) for item in ordered],
            0.0,
            1.0,
        )
        n_seeds = int(ordered[0].get("n_subsample_seeds", 0))
        axes[0].fill_between(
            x,
            band_lo,
            band_hi,
            color=palette["orange"],
            alpha=0.20,
            zorder=1,
            label=f"deep-latent subsample range ({n_seeds} resamples, not a CI)",
        )
    axes[0].plot(
        x,
        [float(item["latent_mean_mass_on_true_cell"]) for item in ordered],
        color=palette["orange"],
        marker="s",
        label="deep-latent reference filter",
    )
    axes[0].plot(
        x,
        [float(item["pymdp_latent_mean_mass_on_true_cell"]) for item in ordered],
        color=palette["purple"],
        marker="^",
        linestyle="--",
        # The pymdp deep-latent recovery reuses the matched reference posterior
        # after the sentinel agreement check, so this curve is identical to the
        # reference filter by construction (it overplots it). Say so in the label
        # so the overlap reads as a validation overlay, not an independent
        # closed-loop recovery curve.
        label="deep-latent pymdp agent (matched to reference by construction)",
    )
    axes[0].set_ylabel("mass on true cell")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Ground-truth recovery vs report sparsity")
    axes[0].grid(True, linewidth=0.4, alpha=0.3)
    axes[0].legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    axes[1].plot(
        x,
        [float(item["latent_mean_posterior_entropy"]) for item in ordered],
        color=palette["green"],
        marker="o",
        label="deep-latent posterior entropy (nats)",
    )
    axes[1].plot(
        x,
        [float(item["pymdp_latent_mean_vfe"]) for item in ordered],
        color=palette["gray"],
        marker="s",
        linestyle="--",
        label="pymdp mean VFE (nats)",
    )
    axes[1].set_xlabel("mean reports per traveller")
    axes[1].set_ylabel("nats")
    axes[1].set_title("Uncertainty diagnostics across sparsity levels")
    axes[1].grid(True, linewidth=0.4, alpha=0.3)
    axes[1].legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    fig.suptitle("Sparsity analysis: latent-state inference under sporadic observation")
    fig.savefig(path)
    plt.close(fig)


def _calibration(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    lanes = summary.get("lanes")
    if not lanes or not isinstance(lanes, list):
        raise ValueError(
            "calibration figure requires calibration.json in the run directory; "
            "rerun the pipeline before generating figures"
        )
    floors = summary.get("floors", {})
    lane_colors = {"reference": palette["blue"], "pymdp": palette["orange"]}
    lane_markers = {"reference": "o", "pymdp": "s"}

    fig = plt.figure(figsize=(10.8, 8.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.02])
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :]),
    ]

    # Panel A: reliability diagram (top-1 confidence vs empirical accuracy per bin).
    axes[0].plot(
        [0, 1],
        [0, 1],
        color=palette["gray"],
        linestyle="--",
        linewidth=0.9,
        label="perfect calibration",
    )
    for lane_index, lane in enumerate(lanes):
        prefix = str(lane["prefix"])
        bins = [b for b in lane["calibration_bins"] if float(b["count"]) > 0]
        confidences = [float(b["mean_confidence"]) for b in bins]
        accuracies = [float(b["accuracy"]) for b in bins]
        counts = [float(b["count"]) for b in bins]
        sizes = [22 + 100 * c / max(counts) for c in counts]
        ece = float(lane["calibration"]["ece"])
        axes[0].scatter(
            confidences,
            accuracies,
            s=sizes,
            facecolors="none",
            edgecolors=lane_colors.get(prefix, palette["purple"]),
            linewidths=1.8,
            marker=lane_markers.get(prefix, "^"),
            alpha=0.95,
            label=f"{prefix} (ECE {ece:.3f})",
        )
        axes[0].plot(
            confidences,
            accuracies,
            color=lane_colors.get(prefix, palette["purple"]),
            linewidth=0.8,
            alpha=0.5,
        )
        for confidence, accuracy, count in zip(confidences, accuracies, counts):
            axes[0].annotate(
                f"n={int(count):,}",
                xy=(confidence, accuracy),
                # 14pt of y-separation between lanes was not enough when two
                # lanes' bins land at nearly the same (confidence, accuracy) -
                # the labels rendered doubled/ghosted on top of each other.
                # 26pt plus a horizontal nudge per lane keeps them legibly
                # apart even when their anchor points coincide.
                xytext=(6 + 34 * lane_index, 4 + 26 * lane_index),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=SMALL_LABEL_FONT_SIZE,
                color="#333333",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.92,
                    "pad": 1.2,
                },
            )
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_xlabel("mean top-1 confidence (bin)")
    axes[0].set_ylabel("empirical top-1 accuracy (bin)")
    axes[0].set_title(
        "Reliability diagram\n(marker size = bin count)",
        loc="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
    )
    # framealpha=1.0 (not 0.86): the "perfect calibration" diagonal runs into
    # the bottom-right corner where this legend sits, and a translucent frame
    # let the line read as visually crossing through the legend text.
    axes[0].legend(loc="lower right", fontsize=LEGEND_FONT_SIZE, framealpha=1.0)

    # Panel B: per-modality marginal loss decomposition vs joint loss and chance floor.
    first = lanes[0]
    modality_names = list(first["per_modality_losses"].keys())
    width = 0.8 / max(len(lanes), 1)
    positions = np.arange(len(modality_names))
    bar_tops: list[float] = []
    line_values: list[float] = []
    for offset, lane in enumerate(lanes):
        prefix = str(lane["prefix"])
        values = [float(lane["per_modality_losses"][name]) for name in modality_names]
        bar_tops.extend(values)
        axes[1].bar(
            positions + offset * width,
            values,
            width=width,
            color=lane_colors.get(prefix, palette["purple"]),
            alpha=0.85,
            label=f"{prefix} marginal",
        )
        joint_loss = float(lane["mean_joint_loss"])
        line_values.append(joint_loss)
        axes[1].axhline(
            joint_loss,
            color=lane_colors.get(prefix, palette["purple"]),
            linestyle="-",
            linewidth=1.1,
            alpha=0.7,
        )
    if "chance_floor_joint_loss" in floors:
        chance_floor = float(floors["chance_floor_joint_loss"])
        line_values.append(chance_floor)
        axes[1].axhline(
            chance_floor,
            color=palette["gray"],
            linestyle=":",
            linewidth=1.2,
            label="uniform-chance joint loss",
        )
    axes[1].set_xticks(positions + width * (len(lanes) - 1) / 2)
    axes[1].set_xticklabels(
        [name.replace("_", "\n") for name in modality_names], fontsize=TICK_FONT_SIZE
    )
    axes[1].set_ylabel("mean predictive loss (nats)")
    axes[1].set_title(
        "Per-modality loss decomposition\n(solid line = lane joint loss)",
        loc="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
    )
    # Anchored above the axes (bbox_to_anchor y=1.02) previously: on this
    # fixture the joint-loss/chance-floor axhlines sit close enough to the
    # panel top that the legend box (which hangs DOWN from an "upper center"
    # anchor) dipped into the axes and crossed those lines. Instead, place
    # the legend in the gap between the tallest bar and the lowest reference
    # line - computed from the actual plotted values, not a hardcoded
    # position, since that gap's location depends on the run's own numbers.
    if bar_tops and line_values:
        gap_y = (max(bar_tops) + min(line_values)) / 2.0
        axes[1].legend(
            loc="center",
            bbox_to_anchor=(0.5, gap_y),
            bbox_transform=axes[1].get_yaxis_transform(),
            ncol=1,
            fontsize=LEGEND_FONT_SIZE,
            framealpha=1.0,
        )
    else:
        axes[1].legend(loc="best", ncol=1, fontsize=LEGEND_FONT_SIZE, framealpha=1.0)

    # Panel C: recovery Brier skill against three increasingly strict nulls —
    # uniform < base-rate < persistence — alongside joint-loss skill. The
    # persistence bar is the honest one for autocorrelated mobility: it strips
    # both the weak uniform floor and the order-ignoring base rate.
    labels = []
    joint_skills = []
    brier_skills = []
    base_rate_skills = []
    persistence_skills = []
    for lane in lanes:
        labels.append(str(lane["prefix"]))
        joint_skills.append(float(lane["skill_score"]))
        brier_skills.append(float(lane["brier_skill_score"]))
        base_rate_skills.append(
            float(lane.get("brier_skill_vs_base_rate", float("nan")))
        )
        persistence_skills.append(
            float(lane.get("brier_skill_vs_persistence", float("nan")))
        )
    bar_positions = np.arange(len(labels))
    joint_bars = axes[2].bar(
        bar_positions - 0.3,
        joint_skills,
        width=0.2,
        color=palette["green"],
        label="joint-loss skill (vs uniform)",
    )
    axes[2].bar(
        bar_positions - 0.1,
        brier_skills,
        width=0.2,
        color=palette["purple"],
        label="recovery skill (vs uniform)",
    )
    axes[2].bar(
        bar_positions + 0.1,
        base_rate_skills,
        width=0.2,
        color=palette["orange"],
        label="recovery skill (vs base rate)",
    )
    axes[2].bar(
        bar_positions + 0.3,
        persistence_skills,
        width=0.2,
        color=palette["blue"],
        label="recovery skill (vs persistence)",
    )
    axes[2].axhline(0.0, color=palette["gray"], linewidth=1.0)
    for bar, value in zip(joint_bars, joint_skills):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.025 if value >= 0 else -0.025),
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=SMALL_LABEL_FONT_SIZE,
            fontweight="bold",
            color="#222222",
        )
    axes[2].set_xticks(bar_positions)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylim(-0.35, 1.05)
    axes[2].set_ylabel("skill")
    axes[2].set_title(
        "Skill vs uniform, base-rate & persistence nulls",
        loc="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
    )
    axes[2].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=LEGEND_FONT_SIZE,
        framealpha=0.9,
    )

    fig.suptitle(
        "Calibration and objective decomposition (synthetic software measurements)",
        x=0.02,
        ha="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def _anticipatory(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    """Speed-slice move propensity + gap-filling skill vs persistence."""

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.8, 4.6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]},
    )
    off = summary.get("speed_slice_off_diagonal_mass") or [0, 0, 0, 0]
    bins = list(range(len(off)))
    axes[0].bar(bins, off, color=palette["blue"])
    axes[0].set_xticks(bins)
    axes[0].set_xlabel("speed category")
    axes[0].set_ylabel("mean off-diagonal column mass")
    # Shortened, left-anchored, smaller title so it no longer overprints this
    # panel's own y-label or bleeds into the middle panel's y-label (F6-8).
    axes[0].set_title(
        "Move propensity per speed slice", loc="left", fontsize=SMALL_LABEL_FONT_SIZE
    )

    gap = summary.get("gap_filling", {})
    blind = summary.get("location_blind_limit", {})
    gap_labels = ["flat", "speed-clamp"]
    gap_values = [
        float(gap.get("flat_skill_vs_persistence", 0.0)),
        float(gap.get("scrp_skill_vs_persistence", 0.0)),
    ]
    blind_labels = ["flat", "speed-clamp"]
    blind_values = [
        float(blind.get("flat_skill_vs_persistence", 0.0)),
        float(blind.get("scrp_skill_vs_persistence", 0.0)),
    ]
    gap_colors = [palette["green"], palette["purple"]]
    axes[1].bar(range(len(gap_values)), gap_values, color=gap_colors)
    axes[1].axhline(0.0, color=palette["gray"], linewidth=1.0)
    axes[1].set_xticks(range(len(gap_values)))
    axes[1].set_xticklabels(gap_labels, fontsize=TICK_FONT_SIZE)
    gap_span = max(0.06, max(abs(value) for value in gap_values) * 1.55)
    axes[1].set_ylim(-gap_span, gap_span)
    axes[1].set_ylabel("gap-filling skill vs persistence")
    axes[1].set_title("Gap-filling skill", loc="left", fontsize=SMALL_LABEL_FONT_SIZE)
    for idx, value in enumerate(gap_values):
        axes[1].text(
            idx,
            value + (0.006 if value >= 0 else -0.006),
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=SMALL_LABEL_FONT_SIZE,
            fontweight="bold",
        )
    axes[2].bar(
        range(len(blind_values)),
        blind_values,
        color=[palette["gray"], palette["orange"]],
    )
    axes[2].axhline(0.0, color=palette["gray"], linewidth=1.0)
    axes[2].set_xticks(range(len(blind_values)))
    axes[2].set_xticklabels(blind_labels, fontsize=TICK_FONT_SIZE)
    axes[2].set_ylabel("blind-lane skill vs persistence")
    axes[2].set_title(
        "Location-blind limit", loc="left", fontsize=SMALL_LABEL_FONT_SIZE
    )
    blind_low = min(blind_values + [0.0])
    axes[2].set_ylim(blind_low * 1.12 if blind_low < 0 else -1.0, 2.0)

    fig.suptitle(
        "Anticipatory recovery (synthetic software measurements)",
        fontsize=SMALL_LABEL_FONT_SIZE,
    )
    fig.savefig(path)
    plt.close(fig)


def _efe_decomposition(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    """EFE epistemic share vs preference strength + the honest active-sensing null."""

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), constrained_layout=True)
    sweep = summary.get("epistemic_share_sweep") or []
    xs = [float(s["preference_strength"]) for s in sweep]
    ys = [float(s["mean_epistemic_share"]) for s in sweep]
    axes[0].plot(xs, ys, marker="o", color=palette["purple"])
    axes[0].set_ylim(0.0, 1.05)
    if xs and ys:
        for x_val, y_val, ha in ((xs[0], ys[0], "left"), (xs[-1], ys[-1], "right")):
            axes[0].annotate(
                f"{y_val:.2f} at C={x_val:g}",
                xy=(x_val, y_val),
                xytext=(8 if ha == "left" else -8, -14),
                textcoords="offset points",
                ha=ha,
                va="top",
                fontsize=SMALL_LABEL_FONT_SIZE,
                color="#333333",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 1.5,
                },
            )
        axes[0].text(
            0.04,
            0.12,
            "epistemic term still dominates\nat the sharpest preference tested",
            transform=axes[0].transAxes,
            ha="left",
            va="bottom",
            fontsize=SMALL_LABEL_FONT_SIZE,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 2.0},
        )
    axes[0].set_xlabel("preference strength (C)")
    axes[0].set_ylabel("mean epistemic share")
    axes[0].set_title("Epistemic share under uniform C", loc="left")

    # Active-sensing margin (info-gain-led minus random mean) per budget, per lane.
    lanes = summary.get("lanes") or []
    width = 0.35
    plotted_margins: list[float] = []
    for li, lane in enumerate(lanes[:2]):
        budgets = lane.get("budgets", [])
        fracs = [float(b["fix_fraction"]) for b in budgets]
        margins = [float(b["margin_vs_random_mean"]) for b in budgets]
        plotted_margins.extend(margins)
        positions = np.arange(len(fracs)) + (li - 0.5) * width
        color = (
            palette["green"]
            if "external" in str(lane.get("label", ""))
            else palette["orange"]
        )
        axes[1].bar(
            positions,
            margins,
            width=width,
            color=color,
            label=str(lane.get("label", ""))
            .replace("fixture", "fixture lane (synthetic)")
            .replace("external:fractalrabbit", "external"),
        )
    axes[1].axhline(0.0, color=palette["gray"], linewidth=1.0)
    if plotted_margins:
        positive_count = sum(1 for margin in plotted_margins if margin > 0.0)
        budget_count = len(plotted_margins)
        verdict = (
            f"info-gain-led sensing beats random in 0/{budget_count} budgets\nhonest null"
            if positive_count == 0
            else f"info-gain-led sensing beats random in {positive_count}/{budget_count} budgets"
        )
        axes[1].annotate(
            verdict,
            # Top-left, not bottom-left: bars for an honest-null verdict
            # scatter near y=0 across the whole x-range (including the first,
            # x=0, budget), so a y=0.10-axes-fraction anchor sat directly on
            # top of the first bar. The margins never approach the axes top,
            # which stays clear.
            xy=(0.03, 0.93),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=SMALL_LABEL_FONT_SIZE,
            fontweight="bold",
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 2.0},
        )
    if lanes:
        axes[1].set_xticks(np.arange(len([b for b in lanes[0].get("budgets", [])])))
        axes[1].set_xticklabels(
            [f"{float(b['fix_fraction']):.1f}" for b in lanes[0].get("budgets", [])]
        )
    axes[1].set_xlabel("sensing budget (fix fraction)")
    axes[1].set_ylabel("Brier margin vs random")
    axes[1].set_title("Active-sensing null", loc="left")
    legend_handles, legend_labels = axes[1].get_legend_handles_labels()
    if legend_handles:
        axes[1].legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            fontsize=LEGEND_FONT_SIZE,
            framealpha=0.9,
        )

    fig.suptitle("Expected-free-energy decomposition (synthetic software measurements)")
    fig.savefig(path)
    plt.close(fig)


def _anomaly_detection(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    """Surprise-based routine-break detection AUC across anomaly strengths vs novelty."""

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    labels = ["VFE\n(rare break)", "VFE\n(random break)", "novelty\nbaseline"]
    values = [
        float(summary.get("vfe_auc_rare", 0.0)),
        float(summary.get("vfe_auc_random", 0.0)),
        float(summary.get("novelty_auc", 0.0)),
    ]
    colors = [palette["purple"], palette["blue"], palette["gray"]]
    ax.bar(range(3), values, color=colors)
    for i, v in enumerate(values):
        ax.text(
            i,
            v + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=SMALL_LABEL_FONT_SIZE,
        )
    ax.axhline(
        0.5, color=palette["gray"], linestyle="--", linewidth=1.0, label="chance"
    )
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=TICK_FONT_SIZE)
    ax.set_ylim(0.0, 1.32)
    ax.set_ylabel("detection AUC")
    ax.set_title("Surprise (VFE) detects routine-breaks above novelty")
    ax.annotate(
        f"n_scored={int(summary.get('n_scored', 0))}, "
        f"injected anomalies={int(summary.get('n_anomalies', 0))}",
        xy=(0.03, 0.99),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=SMALL_LABEL_FONT_SIZE,
        color="#333333",
    )
    ax.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE)
    fig.savefig(path)
    plt.close(fig)


def _recoverability_frontier(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    """What each modality subset recovers: point recovery + occupancy structure."""

    rows = summary.get("frontier") or []
    names = [str(r["modalities"]) for r in rows]
    mass = [float(r["mean_mass_on_true_cell"]) for r in rows]
    occ_tv = [float(r["occupancy_tv"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0), constrained_layout=True)
    pos = np.arange(len(names))
    axes[0].bar(pos, mass, color=palette["green"])
    axes[0].set_xticks(pos)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=TICK_FONT_SIZE)
    axes[0].set_ylabel("mean mass on true cell")
    axes[0].set_title("Point recovery per modality set")
    axes[1].bar(pos, occ_tv, color=palette["orange"])
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=TICK_FONT_SIZE)
    axes[1].set_ylabel("occupancy TV (lower better)")
    axes[1].set_title("Structure recovery per modality set")
    fig.suptitle(
        "Recoverability / minimization frontier (synthetic; oversight-positive)"
    )
    fig.savefig(path)
    plt.close(fig)


def _return_structure(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    labels = [
        "move-step\nentropy gain",
        "persistence\nheadroom",
        "return skill\nvs persistence",
    ]
    values = [
        float(summary.get("move_step_entropy_reduction", 0.0)),
        float(summary.get("persistence_headroom", 0.0)),
        float(summary.get("return_skill_vs_persistence", 0.0)),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(
        range(len(values)),
        values,
        color=[palette["blue"], palette["orange"], palette["green"]],
    )
    ax.axhline(0.0, color=palette["gray"], linewidth=1.0)
    ax.axhline(
        0.1,
        color=palette["gray"],
        linestyle="--",
        linewidth=0.9,
        label="entropy kill-gate",
    )
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=TICK_FONT_SIZE)
    ax.set_ylabel("nats or skill")
    ax.set_title(
        f"Return-structured prior gate: {summary.get('verdict', 'unavailable')}"
    )
    ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _variance_bands(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    names = list(metrics)[:6]
    med = [float(metrics[name].get("median", 0.0)) for name in names]
    low = [float(metrics[name].get("low", 0.0)) for name in names]
    high = [float(metrics[name].get("high", 0.0)) for name in names]
    yerr = np.array(
        [
            [median - lower for median, lower in zip(med, low, strict=True)],
            [upper - median for upper, median in zip(high, med, strict=True)],
        ]
    )
    fig, ax = plt.subplots(figsize=(9.6, 4.9), constrained_layout=True)
    if names:
        ax.bar(range(len(names)), med, yerr=yerr, color=palette["purple"], capsize=4)
        # Print each median and its [min, max] band so the exact spread is
        # readable, not just estimable from the error-bar caps.
        for i, (m, lo, hi) in enumerate(zip(med, low, high, strict=True)):
            above = m >= 0
            # Anchor negative bars' text at the zero line (the bar's near
            # end), not at the bar's far tip, so large-magnitude negative
            # medians keep the annotation inside the bar instead of pushed
            # down into the x-axis tick-label row.
            anchor_y = m if above else 0.0
            ax.annotate(
                f"{m:.2f}\n[{lo:.2f}, {hi:.2f}]",
                (i, anchor_y),
                textcoords="offset points",
                xytext=(0, 8 if above else -8),
                ha="center",
                va="bottom" if above else "top",
                fontsize=SMALL_LABEL_FONT_SIZE,
            )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(
            [name.replace("_", "\n") for name in names],
            rotation=0,
            fontsize=TICK_FONT_SIZE,
        )
        # Give explicit top/bottom headroom sized to the data span, not the
        # autoscaled default: a large-magnitude outlier bar (e.g. a strongly
        # negative median) otherwise compresses near-zero bars' two-line
        # "{median}\n[{lo}, {hi}]" annotations up against the axes title.
        y_top = max(high) if high else 1.0
        y_bottom = min(low + [0.0]) if low else 0.0
        span = (y_top - y_bottom) or 1.0
        ax.set_ylim(y_bottom - span * 0.05, y_top + span * 0.18)
    else:
        ax.text(
            0.5,
            0.5,
            "variance bands unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    n_realizations = int(summary.get("n_realizations", 0) or 0)
    if n_realizations < 2:
        ax.text(
            0.98,
            0.94,
            f"n={n_realizations}: bands degenerate",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=ANNOTATION_FONT_SIZE,
            color=palette.get("orange", "#E69F00"),
            bbox={
                "facecolor": "white",
                "edgecolor": "#BBBBBB",
                "alpha": 0.9,
                "pad": 3.0,
            },
        )
    ax.set_ylabel("median with min-max band")
    ax.set_title("Cross-realization robustness bands", pad=14)
    fig.savefig(path)
    plt.close(fig)


def _upstream_sensitivity(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    status = str(summary.get("status", "unavailable"))
    attempted = summary.get("attempted_commits", []) or []
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    if metrics:
        names = list(metrics)
        values = [float(metrics[name]) for name in names]
        ax.bar(range(len(names)), values, color=palette["green"])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(
            [name.replace("_", "\n") for name in names], fontsize=TICK_FONT_SIZE
        )
        ax.set_ylabel("distance")
    else:
        ax.text(
            0.5,
            0.56,
            f"{status}\n{len(attempted)} attempted prior commit(s)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=ANNOTATION_FONT_SIZE,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Upstream FractalRabbit sensitivity")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fractal_dimension_sweep(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    rows = summary.get("rows") or []
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    if rows:
        dims = np.array([float(row["fractal_dimension"]) for row in rows])
        loss = np.array(
            [_nan_if_missing(row.get("mean_predictive_loss")) for row in rows]
        )
        entropy = np.array(
            [_nan_if_missing(row.get("mean_posterior_entropy")) for row in rows]
        )
        revisit = np.array(
            [_nan_if_missing(row.get("place_revisit_rate")) for row in rows]
        )
        reports_per_place = np.array(
            [_nan_if_missing(row.get("place_reports_per_place")) for row in rows]
        )
        axes[0].plot(
            dims, loss, marker="o", color=palette["blue"], label="predictive loss"
        )
        axes[0].plot(
            dims,
            entropy,
            marker="s",
            color=palette["orange"],
            label="posterior entropy",
        )
        axes[0].set_xlabel("FractalRabbit fractal dimension")
        axes[0].set_ylabel("nats")
        axes[0].set_title("Filter diagnostics under dimension sweep")
        axes[0].legend(loc="best", fontsize=LEGEND_FONT_SIZE)
        axes[1].set_xlabel("FractalRabbit fractal dimension")
        axes[1].set_ylabel("rate or count")
        axes[1].set_title("Place-return texture")
        if np.all(np.isnan(revisit)) and np.all(np.isnan(reports_per_place)):
            # Return-texture columns are null in the current lane (fixture replay
            # does not re-run the stochastic generator's place-return structure);
            # draw an explicit placeholder instead of empty axes + orphan legend.
            axes[1].text(
                0.5,
                0.5,
                "return texture unavailable\n(no revisit data in this lane)",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
                fontsize=SMALL_LABEL_FONT_SIZE,
                color=palette["gray"],
            )
            axes[1].set_xticks([])
            axes[1].set_yticks([])
        else:
            width = 0.07
            axes[1].bar(
                dims - width / 2,
                revisit,
                width=width,
                color=palette["green"],
                label="revisit rate",
            )
            axes[1].bar(
                dims + width / 2,
                reports_per_place,
                width=width,
                color=palette["purple"],
                label="reports/place",
            )
            axes[1].legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    else:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                f"dimension sweep {summary.get('status', 'unavailable')}",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("Fractal-dimension generator sensitivity")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _circadian_phase_signal(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    probs = np.asarray(summary.get("phase_cell_probability") or [], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    if probs.size:
        im = axes[0].imshow(probs, aspect="auto", cmap=SEQUENTIAL_CMAP, vmin=0.0)
        axes[0].set_xlabel("latent cell")
        axes[0].set_ylabel("time-of-day phase bin")
        axes[0].set_title("P(cell | phase)")
        fig.colorbar(im, ax=axes[0], shrink=0.85, label="probability")
        labels = ["phase gain", "permuted gain", "weighted JS", "max JS"]
        values = [
            float(summary.get("phase_gain_vs_pooled", 0.0)),
            float(summary.get("permutation_gain_vs_pooled", 0.0)),
            float(summary.get("weighted_phase_js", 0.0)),
            float(summary.get("max_phase_js", 0.0)),
        ]
        axes[1].bar(
            range(len(values)),
            values,
            color=[
                palette["blue"],
                palette["gray"],
                palette["orange"],
                palette["green"],
            ],
        )
        axes[1].axhline(0.0, color=palette["gray"], linewidth=1.0)
        axes[1].set_xticks(range(len(values)))
        axes[1].set_xticklabels(
            labels, rotation=20, ha="right", fontsize=TICK_FONT_SIZE
        )
        axes[1].set_ylabel("nats")
        axes[1].set_title(f"Phase-prior gate: {summary.get('verdict', 'unavailable')}")
    else:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "circadian phase unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("Circadian-phase occupancy signal")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _partial_observability(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    rows = summary.get("rows") or []
    fig, axis = plt.subplots(figsize=(10.4, 5.4), constrained_layout=True)
    if rows:
        eps = np.array([float(r["eps"]) for r in rows])
        adv_hard = np.array([float(r["aif_advantage_over_hard_nats"]) for r in rows])
        adv_min = np.array([float(r["aif_advantage_over_hard_min"]) for r in rows])
        adv_max = np.array([float(r["aif_advantage_over_hard_max"]) for r in rows])
        adv_obs = np.array([float(r["aif_advantage_over_observed_nats"]) for r in rows])
        n_real = int(rows[0].get("n_realizations", 0))
        axis.axhline(0.0, color=palette["gray"], linewidth=1.0, zorder=1)
        # Full min-max envelope across the resamplings: every eps>0 sits entirely above zero,
        # so the advantage is not a mean artefact — the worst single resampling is still positive.
        axis.fill_between(
            eps,
            adv_min,
            adv_max,
            color=palette["blue"],
            alpha=0.18,
            zorder=1,
            label=f"hard MAP range ({n_real} resamplings)",
        )
        axis.plot(
            eps,
            adv_obs,
            marker="s",
            color=palette["orange"],
            linewidth=1.6,
            label="raw noisy observation",
        )
        axis.plot(
            eps,
            adv_hard,
            marker="o",
            color=palette["blue"],
            linewidth=2.2,
            label="hard MAP mean",
        )
        _direct_line_label(
            axis,
            float(eps[-1]),
            float(adv_hard[-1]),
            "soft belief vs hard MAP",
            palette["blue"],
        )
        _direct_line_label(
            axis,
            float(eps[-1]),
            float(adv_obs[-1]),
            "soft belief vs raw obs.",
            palette["orange"],
        )
        # Overlay the independent-replication design (different N and draw count)
        # so the smaller, more-powered replicated advantage is visible alongside
        # the single-fixture curve and not read as the same experiment.
        cross_rows = summary.get("cross_realization_rows")
        if isinstance(cross_rows, list) and cross_rows:
            c_eps = np.array([float(r["eps"]) for r in cross_rows])
            c_mean = np.array([float(r["mean_advantage_nats"]) for r in cross_rows])
            c_lo = np.array([float(r["between_draw_min_nats"]) for r in cross_rows])
            c_hi = np.array([float(r["between_draw_max_nats"]) for r in cross_rows])
            c_draws = int(summary.get("cross_realization_n_draws", 0))
            c_n = int(summary.get("cross_realization_n_travellers", 0))
            axis.errorbar(
                c_eps,
                c_mean,
                yerr=[c_mean - c_lo, c_hi - c_mean],
                fmt="o",
                markerfacecolor="none",
                color=palette["green"],
                linewidth=1.4,
                capsize=3,
                zorder=2,
            )
            _direct_line_label(
                axis,
                float(c_eps[-1]),
                float(c_mean[-1]),
                f"replicated mean ({c_draws} indep. draws, {c_n} travellers)",
                palette["green"],
            )
        slope = float(summary.get("noise_slope_nats_per_eps", 0.0))
        trend_r2 = float(summary.get("noise_trend_r2", 0.0))
        axis.annotate(
            f"trend: {slope:.2f} nats/$\\epsilon$  ($R^2$={trend_r2:.2f})",
            # Anchored in DATA coords near x=0.02, y=0.42: clear of the y=0
            # baseline and the eps=0 origin-convergence zone (all series meet
            # at (0,0)), clear of the diagonal arrow from the "eps=0:
            # advantage=0" annotation (which runs from (0,0) to (0.11,0.72),
            # i.e. y~0.2 at this x - well below y=0.42), clear of the legend
            # (upper ~20% of the axis), and clear of the "worst of N
            # resamplings" textbox anchored at (0.225, 0.02).
            xy=(0.02, 0.42),
            xycoords="data",
            ha="left",
            va="center",
            fontsize=SMALL_LABEL_FONT_SIZE,
            color=palette["blue"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2.5},
        )
        if bool(summary.get("zero_advantage_when_clean")) and np.isclose(
            float(adv_hard[0]), 0.0
        ):
            # Anchored in data coords, well above the rising eps=0.1 markers
            # (orange/blue/green all sit below y=0.35 there) and clear of the
            # legend box (which occupies the upper-left corner up to
            # roughly x=0.13). The previous offset-points placement put the
            # textbox directly on top of the eps=0.1 raw-observation,
            # hard-MAP, and replicated-mean markers, ghosting them under the
            # translucent bbox and letting the trend lines cut through the text.
            axis.annotate(
                "eps=0: advantage = 0\nsoft belief = hard MAP",
                xy=(float(eps[0]), float(adv_hard[0])),
                xytext=(0.11, 0.72),
                textcoords="data",
                arrowprops={"arrowstyle": "->", "color": palette["gray"], "lw": 0.9},
                ha="left",
                va="bottom",
                fontsize=SMALL_LABEL_FONT_SIZE,
                color="#333333",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.84,
                    "pad": 2.0,
                },
            )
        if adv_min.size:
            # Anchor the arrow on the interior of the "hard MAP range" band's
            # lower edge (midpoint of the last grid segment) rather than
            # exactly at eps[-1]: at eps[-1] the band's lower bound is only
            # ~0.004 nats away from the cross-draw "replicated mean" marker's
            # upper whisker cap at the same x, so an arrow landing there reads
            # as ambiguous between the two distinct statistics. The midpoint
            # still sits exactly on the rendered (piecewise-linear) band edge,
            # away from that marker's column.
            arrow_eps = (
                float(0.5 * (eps[-2] + eps[-1])) if eps.size > 1 else float(eps[-1])
            )
            arrow_adv_min = (
                float(0.5 * (adv_min[-2] + adv_min[-1]))
                if adv_min.size > 1
                else float(adv_min[-1])
            )
            # The textbox anchor sits low (y=0.02) and to the right of the
            # eps=0.2 x-column so it clears the eps=0.2 replicated-mean
            # marker's lower whisker cap (previously the box's top edge sat
            # almost directly on that cap, rendering it faint and crowded
            # against the "worst of" text).
            axis.annotate(
                f"worst of {n_real} resamplings:\n+{float(adv_min[-1]):.2f} nats",
                xy=(arrow_eps, arrow_adv_min),
                xytext=(0.225, 0.02),
                textcoords="data",
                arrowprops={"arrowstyle": "->", "color": palette["gray"], "lw": 0.9},
                ha="left",
                va="bottom",
                fontsize=SMALL_LABEL_FONT_SIZE,
                color="#333333",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.84,
                    "pad": 2.0,
                },
            )
        axis.margins(x=0.10)
        axis.set_ylim(
            min(float(adv_obs.min()), 0.0) - 0.16,
            max(float(adv_obs.max()), float(adv_max.max()), float(adv_hard.max()))
            + 0.18,
        )
        axis.set_xlabel(
            "observation-noise level $\\epsilon$\n(emission flip probability)"
        )
        axis.set_ylabel("soft-belief advantage (nats, lower loss)")
        axis.set_title(
            "Partial observability: advantage rises with emission noise",
            loc="left",
        )
        axis.legend(
            fontsize=LEGEND_FONT_SIZE, loc="upper left", framealpha=0.82, ncol=1
        )
    else:
        axis.text(
            0.5,
            0.5,
            "partial-observability sweep unavailable",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    fig.savefig(path)
    plt.close(fig)


def _sequence_baselines(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    rows = summary.get("rows") or []
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.8), sharey=True)
    if rows:
        names = [str(row["model"]) for row in rows]
        losses = np.array([float(row["mean_predictive_loss"]) for row in rows])
        skills = np.array([float(row["skill_vs_base_rate"]) for row in rows])
        best = str(summary.get("best_model", ""))
        colors = [
            palette["green"] if name == best else palette["blue"] for name in names
        ]
        pos = np.arange(len(names))
        labels = [
            name.replace("first_order", "1st-order")
            .replace("metadata", "meta")
            .replace("markov", "Markov")
            .replace("persistence", "persistence")
            .replace("base_rate", "base rate")
            .replace("_", " ")
            for name in names
        ]
        axes[0].barh(pos, losses, color=colors)
        axes[0].invert_yaxis()
        axes[0].set_yticks(pos)
        axes[0].set_yticklabels(labels, fontsize=TICK_FONT_SIZE)
        axes[0].set_xlabel("mean predictive loss (nats)")
        axes[0].set_title(
            "Prequential sequence-baseline loss",
            loc="left",
            fontsize=PANEL_HEADING_FONT_SIZE,
        )
        for y, value in zip(pos, losses, strict=True):
            axes[0].text(
                value,
                y,
                f" {value:.3f}",
                va="center",
                ha="left",
                fontsize=ANNOTATION_FONT_SIZE,
                color="#1D1D1D",
            )
        axes[1].barh(pos, skills, color=colors)
        axes[1].axvline(0.0, color=palette["gray"], linewidth=0.9)
        axes[1].set_yticks(pos)
        axes[1].set_yticklabels(labels, fontsize=TICK_FONT_SIZE)
        axes[1].set_xlabel("skill vs online base-rate")
        gate = str(summary.get("pymdp_performance_gate", "sequence gate")).replace(
            "_", " "
        )
        axes[1].set_title(gate, loc="left", fontsize=PANEL_HEADING_FONT_SIZE)
        for y, value in zip(pos, skills, strict=True):
            axes[1].text(
                value,
                y,
                f" {value:.3f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=ANNOTATION_FONT_SIZE,
                color="#1D1D1D",
            )
    else:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "sequence baselines unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        "Sequence and reporting-process performance gate",
        fontsize=CANVAS_TITLE_FONT_SIZE,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def sota_leaderboard_headline(rows: list, status: str, strongest: object) -> str:
    """Derive the leaderboard headline from the gate status and the bars, so it can
    never contradict a cleared gate (no green-by-construction literal)."""

    eligible = [
        r
        for r in rows
        if r.get("locked_split_scored")
        and not r.get("leakage")
        and r.get("family") != "transparent_null"
    ]
    best_eligible = min(
        (float(r["mean_predictive_loss"]) for r in eligible), default=None
    )
    no_eligible_beats = (
        not isinstance(strongest, int | float)
        or best_eligible is None
        or best_eligible >= float(strongest)
    )
    if status == "sota_not_cleared" and no_eligible_beats:
        return "SOTA gate closed: no eligible candidate beats the strongest honest baseline"
    if status == "sota_cleared":
        return "SOTA gate cleared: an eligible candidate beats the strongest honest baseline"
    return f"Claim gate: {status}"


def _sota_leaderboard(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    rows = summary.get("leaderboard") or []
    status = str(summary.get("status", "unavailable"))
    strongest = summary.get("strongest_baseline_loss")
    # Taller canvas (F6-21) so the wrapped multi-line y-tick labels (e.g.
    # "multi-horizon recency / attention") get their own vertical space and stop
    # crowding the neighbouring "backoff Markov" tick.
    fig, ax = plt.subplots(figsize=(13.2, 11.2))
    if rows:
        ordered = sorted(rows, key=lambda r: float(r["mean_predictive_loss"]))
        names = [str(r["model"]) for r in ordered]
        losses = np.array([float(r["mean_predictive_loss"]) for r in ordered])
        fam_color = {
            "adversarial_probe": palette.get("red", "#D55E00"),
            "transparent_null": palette.get("blue", "#0072B2"),
            "transparent_baseline": palette.get("sky_blue", "#56B4E9"),
            "classical_sequence": "#E69F00",
            "classical_state_space": "#E69F00",
            "neural_sequence": "#9467BD",
            "neural_or_attention": "#9467BD",
            "active_inference": palette.get("green", "#009E73"),
        }
        fam_label = {
            "active_inference": "active inference",
            "transparent_null": "transparent null",
            "transparent_baseline": "transparent baseline",
            "classical_sequence": "classical sequence",
            "classical_state_space": "classical/state-space",
            "neural_sequence": "neural sequence",
            "neural_or_attention": "neural/attention",
            "adversarial_probe": "leakage probe",
        }
        colors = [
            fam_color.get(str(r.get("family")), palette.get("gray", "#999999"))
            for r in ordered
        ]
        ypos = np.arange(len(names))[::-1]  # lowest loss (best) at the top
        xerr_low, xerr_high = [], []
        for r, loss in zip(ordered, losses):
            lo, hi = r.get("loss_ci_low"), r.get("loss_ci_high")
            if (
                isinstance(lo, int | float)
                and isinstance(hi, int | float)
                and np.isfinite(lo)
                and np.isfinite(hi)
            ):
                xerr_low.append(max(0.0, loss - float(lo)))
                xerr_high.append(max(0.0, float(hi) - loss))
            else:
                xerr_low.append(0.0)
                xerr_high.append(0.0)
        bars = ax.barh(ypos, losses, color=colors, edgecolor="#333333", linewidth=0.6)
        for bar, r in zip(bars, ordered):
            if not r.get("locked_split_scored"):
                bar.set_hatch("//")  # full-stream: not locked-split scored, ineligible
        ax.errorbar(
            losses,
            ypos,
            xerr=[xerr_low, xerr_high],
            fmt="none",
            ecolor="#333333",
            elinewidth=1.0,
            capsize=3,
        )
        ax.set_yticks(ypos)
        ax.set_yticklabels(
            [_model_axis_label(n, width=24) for n in names], fontsize=TICK_FONT_SIZE
        )
        ax.tick_params(axis="y", pad=8)
        for tick_label in ax.get_yticklabels():
            tick_label.set_ha("right")
            tick_label.set_multialignment("right")
        ax.set_ylim(float(ypos.min()) - 0.6, float(ypos.max()) + 6.0)
        ax.set_xlabel(
            "mean next-cell predictive loss (nats, lower is better); whiskers: bootstrap 95% CI"
        )
        xmax = max(float(np.nanmax(losses)) * 1.28, 1.0)
        ax.set_xlim(0, xmax)
        if isinstance(strongest, int | float):
            baseline = float(strongest)
            ax.axvline(
                baseline,
                color=palette.get("gray", "#999999"),
                linestyle="--",
                linewidth=1.0,
            )
        for r, loss, y, xh in zip(ordered, losses, ypos, xerr_high):
            if r.get("leakage"):
                ax.text(
                    loss + xmax * 0.025,
                    y,
                    "EXCLUDED leakage probe",
                    va="center",
                    fontsize=SMALL_LABEL_FONT_SIZE,
                    color=palette.get("red", "#D55E00"),
                    fontweight="bold",
                )
            else:
                # RedTeam fix #5: clear this row's own whisker cap (loss + xh
                # is the CI-high bound) instead of a fixed axis-width offset,
                # so a tight CI's capsize=3 cap never strikes the digits.
                ax.text(
                    loss + max(xh, 0.0) + xmax * 0.012,
                    y,
                    f"{loss:.2f}",
                    va="center",
                    fontsize=SMALL_LABEL_FONT_SIZE,
                    color="#333333",
                )
        gate = summary.get("fractalrabbit_gate", {})
        if isinstance(gate, dict) and gate:
            aif_row = next(
                (r for r in ordered if str(r.get("family")) == "active_inference"),
                {},
            )
            gate_label = "cleared" if gate.get("aif_beats_strongest_fair") else "closed"
            ax.text(
                0.985,
                0.965,
                "\n".join(
                    [
                        f"fair - AIF delta: {_fmt_stat(gate.get('mean_difference_nats_fair_minus_aif'))} nats",
                        (
                            "clustered CI: "
                            f"[{_fmt_stat(gate.get('clustered_ci_low'))}, "
                            f"{_fmt_stat(gate.get('clustered_ci_high'))}]"
                        ),
                        f"one-sided p={_fmt_stat(gate.get('one_sided_p'))}; holdout travellers={gate.get('n_holdout_travellers', 'n/a')}",
                        f"gate status: {gate_label}; strongest fair={gate.get('strongest_fair_baseline', 'n/a')}",
                        (
                            "top AIF: "
                            f"Brier {_fmt_stat(aif_row.get('brier_score'))}; "
                            f"ECE {_fmt_stat(aif_row.get('ece'))}; "
                            f"top-1 {_fmt_stat(aif_row.get('top1_accuracy'))}; "
                            f"runtime {_fmt_stat(aif_row.get('runtime_seconds'))}s"
                        ),
                    ]
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=SMALL_LABEL_FONT_SIZE,
                color="#222222",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#BBBBBB",
                    "alpha": 0.86,
                    "pad": 4.0,
                },
            )
        ax.set_title(
            str(
                summary.get("headline")
                or sota_leaderboard_headline(rows, status, strongest)
            ),
            loc="left",
            fontsize=SMALL_LABEL_FONT_SIZE,
            fontweight="bold",
            pad=8,
        )
        present_families = [str(r.get("family")) for r in ordered]
        legend_order = [
            "active_inference",
            "classical_state_space",
            "classical_sequence",
            "neural_or_attention",
            "neural_sequence",
            "transparent_baseline",
            "transparent_null",
            "adversarial_probe",
        ]
        legend_handles = [
            patches.Patch(
                facecolor=fam_color.get(family, palette.get("gray", "#999999")),
                label=fam_label.get(family, family.replace("_", " ")),
            )
            for family in legend_order
            if family in present_families
        ]
        if any(not r.get("locked_split_scored") for r in ordered):
            legend_handles.append(
                patches.Patch(
                    facecolor="white",
                    edgecolor="#333333",
                    hatch="//",
                    label="full-stream (ineligible)",
                )
            )
        if (
            any(r.get("leakage") for r in ordered)
            and "adversarial_probe" not in present_families
        ):
            legend_handles.append(
                patches.Patch(
                    facecolor=fam_color["adversarial_probe"], label="leakage probe"
                )
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=palette.get("gray", "#999999"),
                linestyle="--",
                linewidth=1.0,
                label="strongest fair baseline",
            )
        )
        ax.legend(
            handles=legend_handles,
            fontsize=LEGEND_FONT_SIZE,
            loc="upper center",
            bbox_to_anchor=(0.52, -0.18),
            ncol=4,
            framealpha=0.82,
            borderpad=0.6,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "sota leaderboard unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(rect=(0, 0.12, 1.0, 1.0))
    fig.savefig(path)
    plt.close(fig)


def _aif_regime_map(
    summary: dict[str, object],
    multiplicity_summary: dict[str, object],
    path: Path,
    palette: dict[str, str],
) -> None:
    realizations = summary.get("realization_summaries") or []
    fig = plt.figure(figsize=(14.6, 6.1))
    grid = fig.add_gridspec(1, 3, width_ratios=[3.25, 0.12, 1.75], wspace=0.55)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    note_ax = fig.add_subplot(grid[0, 2])
    note_ax.axis("off")
    if not isinstance(realizations, list) or not realizations:
        ax.text(
            0.5,
            0.5,
            "AIF regime search unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        cax.axis("off")
        fig.subplots_adjust(left=0.08, right=0.96, bottom=0.12, top=0.90, wspace=0.45)
        fig.savefig(path)
        plt.close(fig)
        return

    regimes = list(
        dict.fromkeys(str(row.get("regime_id", "unknown")) for row in realizations)
    )
    seeds = sorted({int(row.get("realization_seed", 0)) for row in realizations})
    seed_index = {seed: idx for idx, seed in enumerate(seeds)}
    regime_index = {regime: idx for idx, regime in enumerate(regimes)}
    advantages = np.full((len(regimes), len(seeds)), np.nan, dtype=float)
    clears = np.zeros_like(advantages, dtype=bool)
    leakage_ok = np.zeros_like(advantages, dtype=bool)
    fair_names: dict[tuple[int, int], str] = {}
    aif_names: dict[tuple[int, int], str] = {}
    for row in realizations:
        regime = str(row.get("regime_id", "unknown"))
        seed = int(row.get("realization_seed", 0))
        y = regime_index[regime]
        x = seed_index[seed]
        advantages[y, x] = float(row.get("mean_difference_nats_fair_minus_aif", 0.0))
        clears[y, x] = bool(row.get("realization_clears", False))
        leakage_ok[y, x] = bool(row.get("leakage_floor_ok", False))
        fair_names[(y, x)] = str(row.get("strongest_fair_baseline", "fair"))
        aif_names[(y, x)] = str(row.get("best_aif_candidate", "aif"))

    finite = advantages[np.isfinite(advantages)]
    bound = max(float(np.max(np.abs(finite))) if finite.size else 0.0, 0.05)
    image = ax.imshow(
        advantages, cmap=DIVERGING_CMAP, vmin=-bound, vmax=bound, aspect="auto"
    )
    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels([str(seed) for seed in seeds], fontsize=TICK_FONT_SIZE)
    ax.set_yticks(np.arange(len(regimes)))
    ax.set_yticklabels(
        [_clean_label(regime, width=24) for regime in regimes], fontsize=TICK_FONT_SIZE
    )
    ax.set_xlabel("realization seed")
    ax.set_title(
        "Regime-qualified AIF edge vs strongest fair baseline",
        loc="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
        fontweight="bold",
        pad=10,
    )
    for y in range(len(regimes)):
        for x in range(len(seeds)):
            value = advantages[y, x]
            if not np.isfinite(value):
                continue
            cell_color = "#111111" if abs(value) < bound * 0.55 else "white"
            marker = "CI+" if clears[y, x] else "ns"
            if not leakage_ok[y, x]:
                marker = "floor"
            ax.text(
                x,
                y,
                f"{value:+.3f}\n{marker}",
                ha="center",
                va="center",
                fontsize=SMALL_LABEL_FONT_SIZE,
                color=cell_color,
                fontweight="bold" if clears[y, x] else "normal",
            )
            if clears[y, x]:
                ax.add_patch(
                    patches.Rectangle(
                        (x - 0.47, y - 0.47),
                        0.94,
                        0.94,
                        fill=False,
                        edgecolor=palette.get("green", "#009E73"),
                        linewidth=2.2,
                    )
                )
    ax.set_xticks(np.arange(-0.5, len(seeds), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(regimes), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    status = str(summary.get("status", "unavailable")).replace("_", " ")
    status_label = (
        "local regime gate cleared"
        if status == "regime qualified sota cleared"
        else status
    )
    cleared_count = int(summary.get("cleared_regime_count", 0) or 0)
    cleared_regime = summary.get("best_cleared_regime", {})
    if not isinstance(cleared_regime, dict):
        cleared_regime = {}
    comparisons = (
        multiplicity_summary.get("comparisons", [])
        if isinstance(multiplicity_summary, dict)
        else []
    )
    cleared_comparison = None
    best_cleared_id = str(summary.get("best_cleared_regime_id", ""))
    for comparison in comparisons if isinstance(comparisons, list) else []:
        if (
            isinstance(comparison, dict)
            and comparison.get("comparison_id") == f"aif_regime::{best_cleared_id}"
        ):
            cleared_comparison = comparison
            break
    verdict_lines: list[str] | None = None
    if np.any(clears):
        family_size = int(
            multiplicity_summary.get("family_size", len(comparisons) or 0) or 0
        )
        holm_survivors = int(
            multiplicity_summary.get("holm", {}).get("n_survivors", 0)
            if isinstance(multiplicity_summary.get("holm"), dict)
            else 0
        )
        holm_p = (
            cleared_comparison.get("holm_adjusted_p")
            if isinstance(cleared_comparison, dict)
            else None
        )
        if not isinstance(holm_p, int | float):
            headline = (
                multiplicity_summary.get("headline", {})
                if isinstance(multiplicity_summary.get("headline"), dict)
                else {}
            )
            holm_p = headline.get("max_clearer_holm_adjusted_p")
        clear_cell_count = int(np.sum(clears))
        if cleared_count > 0:
            verdict_lines = [
                "clears the LOCAL gate",
                f"Holm p={_fmt_stat(holm_p)}: {holm_survivors}/{family_size} survive correction",
            ]
        else:
            verdict_lines = [
                f"CI+ in {clear_cell_count}/{clears.size} realization cells",
                "local regime gate not cleared",
                f"Holm p={_fmt_stat(holm_p)}: {holm_survivors}/{family_size} survive correction",
            ]
        # NOTE(RedTeam fix #3): render the verdict into the note_ax side panel
        # (below, alongside "Best AIF vs strongest fair") instead of as an
        # ax.annotate box — every regime x seed heatmap cell carries its own
        # value/marker text, so any axes-fraction anchor inside `ax` sits on
        # top of real data. note_ax is already reserved for this kind of
        # side text elsewhere in this function.
    labels = []
    candidate_short_names = {
        "soft_filter": "soft AIF",
        "precision_return_aif": "precision AIF",
        "hard_filter": "hard",
        "first_order_markov": "Markov",
        "online_base_rate": "base-rate",
        "particle_filter": "particle",
        "switching_noise_hmm": "switching HMM",
    }
    for y in range(len(regimes)):
        for x in range(len(seeds)):
            if (y, x) in fair_names:
                regime_label = regimes[y].replace("_", " ").replace("emission ", "")
                pair_label = (
                    f"{candidate_short_names.get(aif_names[(y, x)], aif_names[(y, x)].replace('_', ' '))}"
                    f" vs {candidate_short_names.get(fair_names[(y, x)], fair_names[(y, x)].replace('_', ' '))}"
                )
                labels.append((regime_label, pair_label))
                break
    if labels:
        note_ax.text(
            0.0,
            0.98,
            "Best AIF vs strongest fair",
            transform=note_ax.transAxes,
            ha="left",
            va="top",
            fontsize=SMALL_LABEL_FONT_SIZE,
            fontweight="bold",
            color="#222222",
        )
        y_pos = 0.82
        for regime_label, pair_label in labels[:5]:
            note_ax.text(
                0.0,
                y_pos,
                f"{regime_label}: {pair_label}",
                transform=note_ax.transAxes,
                ha="left",
                va="top",
                fontsize=ANNOTATION_FONT_SIZE,
                color=palette.get("gray", "#595959"),
                linespacing=1.15,
            )
            y_pos -= 0.11
        # Verdict summary (moved here from the ax.annotate box above — see
        # RedTeam fix #3 note) is rendered top-down from the cursor left by the
        # regime/baseline list, and every remaining block below it also flows
        # from that same cursor so nothing can land on top of anything else.
        if verdict_lines:
            y_pos -= 0.03
            note_ax.text(
                0.0,
                y_pos,
                "\n".join(verdict_lines),
                transform=note_ax.transAxes,
                ha="left",
                va="top",
                fontsize=ANNOTATION_FONT_SIZE,
                fontweight="bold",
                color="#222222",
                linespacing=1.15,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#BBBBBB",
                    "alpha": 0.92,
                    "pad": 2.0,
                },
            )
            y_pos -= 0.062 * len(verdict_lines) + 0.05
        if cleared_regime or cleared_comparison:
            note_ax.text(
                0.0,
                y_pos,
                "\n".join(
                    [
                        f"Status: {status_label}; cleared={cleared_count}",
                        (
                            f"Best local regime: {best_cleared_id or 'none'}; "
                            f"p_max={_fmt_stat(cleared_regime.get('one_sided_p_max'))}"
                        ),
                        (
                            f"min CI low={_fmt_stat(cleared_regime.get('min_clustered_ci_low'))}; "
                            f"Holm p={_fmt_stat(cleared_comparison.get('holm_adjusted_p') if cleared_comparison else None)}"
                        ),
                        (
                            "Holm survivor: "
                            f"{'yes' if cleared_comparison and cleared_comparison.get('holm_survives') else 'no'}"
                        ),
                    ]
                ),
                transform=note_ax.transAxes,
                ha="left",
                va="top",
                fontsize=ANNOTATION_FONT_SIZE,
                color="#222222",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#BBBBBB",
                    "alpha": 0.86,
                    "pad": 3.0,
                },
            )
            y_pos -= 0.062 * 4 + 0.05
        else:
            note_ax.text(
                0.0,
                y_pos,
                f"Status: {status_label}; cleared={cleared_count}",
                transform=note_ax.transAxes,
                ha="left",
                va="top",
                fontsize=ANNOTATION_FONT_SIZE,
                color=palette.get("gray", "#595959"),
            )
            y_pos -= 0.07
        note_ax.text(
            0.0,
            y_pos,
            "Local gate only; no global SOTA or empirical-transfer claim.",
            transform=note_ax.transAxes,
            ha="left",
            va="top",
            fontsize=ANNOTATION_FONT_SIZE,
            color=palette.get("gray", "#595959"),
        )
    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.set_title("Delta loss\n(nats)", fontsize=ANNOTATION_FONT_SIZE, pad=8)
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    # Wider inter-column gap (F6-14) so the colorbar tick labels sit in dead
    # space and no longer overprint the side-panel gate/regime text.
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.90, wspace=0.55)
    fig.savefig(path)
    plt.close(fig)


def _fair_inference(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    """Two panels: (left) the oracle-stripped predictor ladder with CI whiskers, and
    (right) the recursion-advantage-vs-observation-noise crossover. Shows that soft
    marginalization carries most of the inference gain and recursion helps only as the
    channel degrades — honestly, not as a SOTA claim."""

    ladder = (
        summary.get("ladder", {}) if isinstance(summary.get("ladder"), dict) else {}
    )
    losses = ladder.get("losses", {}) if isinstance(ladder.get("losses"), dict) else {}
    ci = ladder.get("ci", {}) if isinstance(ladder.get("ci"), dict) else {}
    noise = (
        summary.get("noise_sweep", {})
        if isinstance(summary.get("noise_sweep"), dict)
        else {}
    )
    points = noise.get("points", []) if isinstance(noise.get("points"), list) else []

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.1), gridspec_kw={"wspace": 0.42})
    order = [
        ("oracle", "oracle (handed true cell)", palette.get("gray", "#999999")),
        ("hard_handicapped", "hard argmax", palette.get("red", "#D55E00")),
        ("soft_handicapped", "soft marginalized", palette.get("blue", "#0072B2")),
        ("soft_plus_return_boost", "soft + return-boost", "#56B4E9"),
        ("honest_aif", "recursive AIF", palette.get("green", "#009E73")),
        ("honest_aif_plus_boost", "recursive AIF + boost", "#1B7837"),
    ]
    present = [
        (k, lab, c) for k, lab, c in order if isinstance(losses.get(k), int | float)
    ]
    if present:
        ypos = np.arange(len(present))[::-1]
        vals = np.array([float(losses[k]) for k, _, _ in present])
        lo = np.array(
            [float(ci.get(k, [losses[k], losses[k]])[0]) for k, _, _ in present]
        )
        hi = np.array(
            [float(ci.get(k, [losses[k], losses[k]])[1]) for k, _, _ in present]
        )
        axes[0].barh(
            ypos,
            vals,
            color=[c for _, _, c in present],
            edgecolor="#333333",
            linewidth=0.6,
        )
        axes[0].errorbar(
            vals,
            ypos,
            xerr=[vals - lo, hi - vals],
            fmt="none",
            ecolor="#333333",
            elinewidth=1.0,
            capsize=3,
        )
        axes[0].set_yticks(ypos)
        axes[0].set_yticklabels([lab for _, lab, _ in present], fontsize=TICK_FONT_SIZE)
        # Extra headroom (F6-15) so the "+x.xx vs oracle" delta labels stay inside
        # this panel instead of crossing into the right panel's rotated y-label.
        axes[0].set_xlim(0, max(vals.max() * 1.72, 1.0))
        axes[0].set_xlabel("mean next-cell loss (nats), lower is better")
        axes[0].set_title(
            "Fair ladder: same transition", loc="left", fontsize=SMALL_LABEL_FONT_SIZE
        )
        if "oracle" in losses:
            oracle_loss = float(losses["oracle"])
            axes[0].axvline(
                oracle_loss,
                color=palette.get("gray", "#999999"),
                linestyle=":",
                linewidth=1.1,
            )
            axes[0].annotate(
                "oracle ceiling",
                xy=(oracle_loss, float(ypos.max()) + 0.22),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=SMALL_LABEL_FONT_SIZE,
                color=palette.get("gray", "#595959"),
            )
            for yi, value, (key, _, _) in zip(ypos, vals, present):
                if key == "oracle":
                    continue
                gap = float(value - oracle_loss)
                axes[0].text(
                    float(value) + max(vals.max() * 0.015, 0.01),
                    yi,
                    f"+{gap:.2f} vs oracle",
                    ha="left",
                    va="center",
                    fontsize=ANNOTATION_FONT_SIZE,
                    color="#1D1D1D",
                )
    else:
        axes[0].text(
            0.5,
            0.5,
            "fair-inference ladder unavailable",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_xticks([])
        axes[0].set_yticks([])

    if points:
        sm = [float(p["emission_smoothing"]) for p in points]
        adv = [float(p["aif_advantage_over_soft"]) for p in points]
        axes[1].axhline(
            0.0, color=palette.get("gray", "#999999"), linewidth=1.0, linestyle="--"
        )
        axes[1].plot(
            sm, adv, marker="o", color=palette.get("green", "#009E73"), linewidth=1.8
        )
        axes[1].fill_between(
            sm,
            0.0,
            adv,
            where=[a > 0 for a in adv],
            color=palette.get("green", "#009E73"),
            alpha=0.18,
        )
        _direct_line_label(
            axes[1],
            float(sm[-1]),
            float(adv[-1]),
            "recursive belief helps under noise",
            palette.get("green", "#009E73"),
        )
        crossover = None
        for left_idx in range(len(sm) - 1):
            x0, x1 = float(sm[left_idx]), float(sm[left_idx + 1])
            y0, y1 = float(adv[left_idx]), float(adv[left_idx + 1])
            if y0 == 0.0:
                crossover = x0
                break
            if y0 * y1 < 0.0:
                crossover = x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
                break
        if crossover is not None and np.isfinite(crossover):
            axes[1].axvline(
                float(crossover),
                color=palette.get("gray", "#999999"),
                linestyle=":",
                linewidth=1.1,
            )
            axes[1].annotate(
                f"crossover ≈ {float(crossover):.2f}\nrecursion starts paying around here",
                xy=(float(crossover), 0.0),
                xytext=(10, 28),
                textcoords="offset points",
                arrowprops={
                    "arrowstyle": "->",
                    "color": palette.get("gray", "#777777"),
                    "lw": 0.9,
                },
                ha="left",
                va="bottom",
                fontsize=SMALL_LABEL_FONT_SIZE,
                color="#333333",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 2.0,
                },
            )
        axes[1].set_xscale("log")
        axes[1].set_xlabel("emission smoothing (observation noise →)")
        axes[1].set_ylabel("AIF advantage\n(nats)")
        axes[1].set_title(
            "Recursion helps only under noise",
            loc="left",
            fontsize=SMALL_LABEL_FONT_SIZE,
        )
        _ymin, _ymax = axes[1].get_ylim()
        axes[1].annotate(
            "recursion hurts\nwhen obs are clean",
            xy=(0.06, 0.08),
            xycoords="axes fraction",
            fontsize=SMALL_LABEL_FONT_SIZE,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 2.0},
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "noise sweep unavailable",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_xticks([])
        axes[1].set_yticks([])

    fig.suptitle(
        "Inferring the latent cell: soft marginalization dominates; recursion is noise-conditional\n"
        "Neither beats the oracle-fed baseline (fair-frontier result, not a SOTA claim)",
        x=0.02,
        y=0.97,
        ha="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
    )
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.16, top=0.82, wspace=0.42)
    fig.savefig(path)
    plt.close(fig)


def _fair_power_regime(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    """Heatmap of the traveller-clustered AIF advantage over a return-boosted soft Markov
    across traveller count (x) and observation noise (y). Red = recursive belief better,
    blue = soft better; a green ring marks cells where the clustered CI separates from zero. Shows
    the advantage is regime-bound: it vanishes at scale under a clean emission but re-emerges
    under observation noise."""

    from matplotlib.colors import TwoSlopeNorm

    cells = summary.get("cells", []) if isinstance(summary.get("cells"), list) else []
    counts = (
        summary.get("traveller_counts", [])
        if isinstance(summary.get("traveller_counts"), list)
        else []
    )
    smooths = (
        summary.get("smoothing_levels", [])
        if isinstance(summary.get("smoothing_levels"), list)
        else []
    )

    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    if cells and counts and smooths:
        counts = sorted({int(c) for c in counts})
        smooths = sorted({float(s) for s in smooths}, reverse=True)  # noisy at top
        lut = {(int(c["traveller_count"]), float(c["smoothing"])): c for c in cells}
        adv = np.full((len(smooths), len(counts)), np.nan)
        for i, sm in enumerate(smooths):
            for j, nt in enumerate(counts):
                cell = lut.get((nt, sm))
                if cell is not None:
                    adv[i, j] = float(cell["mean_advantage_nats"])
        lim = float(np.nanmax(np.abs(adv))) or 1e-3
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        im = ax.imshow(adv, cmap=DIVERGING_CMAP, norm=norm, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(
            "AIF advantage (nats):  + = better", fontsize=SMALL_LABEL_FONT_SIZE
        )
        for i, sm in enumerate(smooths):
            for j, nt in enumerate(counts):
                cell = lut.get((nt, sm))
                if cell is None:
                    continue
                wins = bool(cell.get("seed_robust_win"))
                ax.text(
                    j,
                    i,
                    f"{float(cell['mean_advantage_nats']):+.2f}",
                    ha="center",
                    va="center",
                    fontsize=TICK_FONT_SIZE,
                    color="#111111",
                )
                if wins:  # ring marks a seed-robust win across generator draws
                    # A solid ring is interval-backed (the clustered CI separates
                    # from zero AND the holdout cluster count is adequate). Where
                    # ci_cluster_count_adequate is false (e.g. the 4-traveller
                    # column, 2 holdout travellers per draw), the win is only
                    # across-draw sign unanimity, not within-draw interval
                    # precision — draw it dashed to match the Discussion caveat.
                    adequate = bool(cell.get("ci_cluster_count_adequate"))
                    ax.add_patch(
                        plt.Circle(
                            (j, i),
                            0.42,
                            fill=False,
                            edgecolor="#1B7837",
                            linewidth=2.2,
                            linestyle="solid" if adequate else (0, (2, 1.5)),
                        )
                    )
        win_grid = np.array(
            [
                [bool(lut.get((nt, sm), {}).get("seed_robust_win")) for nt in counts]
                for sm in smooths
            ],
            dtype=float,
        )
        if np.any(win_grid) and np.any(win_grid == 0.0):
            ax.contour(
                np.arange(len(counts)),
                np.arange(len(smooths)),
                win_grid,
                levels=[0.5],
                colors=[palette.get("gray", "#595959")],
                linewidths=2.0,
            )
        mechanism_lines = []
        if bool(summary.get("aif_advantage_vanishes_at_scale_clean")):
            mechanism_lines.append(
                "more travellers -> point estimates catch up on clean data"
            )
        if bool(summary.get("noise_restores_advantage_at_scale")):
            mechanism_lines.append("noise keeps marginalization valuable at scale")
        ax.set_xticks(
            range(len(counts)), [str(c) for c in counts], fontsize=TICK_FONT_SIZE
        )
        ax.set_yticks(
            range(len(smooths)), [f"{s:.2f}" for s in smooths], fontsize=TICK_FONT_SIZE
        )
        ax.set_xlabel("number of travellers (fixed cell grid →)")
        ax.set_ylabel("emission smoothing (noise ↑)")
        title = "Regime-bound AIF advantage over boosted soft Markov"
        if mechanism_lines:
            title += "\nclean scale catches up;\nnoise keeps marginalization valuable"
        ax.set_title(
            title,
            loc="left",
            fontsize=SMALL_LABEL_FONT_SIZE,
            fontweight="bold",
            pad=14,
        )
        ring_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                linestyle="none",
                markeredgecolor="#1B7837",
                markerfacecolor="none",
                markeredgewidth=2.0,
                markersize=13,
                label="solid ring: clustered CI separates from zero in every draw (adequate cluster count)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                linestyle="none",
                markeredgecolor="#1B7837",
                markerfacecolor="none",
                markeredgewidth=2.0,
                markersize=13,
                fillstyle="none",
                label="dashed ring: sign-unanimous across draws; cluster count too small for interval reading",
            ),
        ]
        if np.any(win_grid) and np.any(win_grid == 0.0):
            ring_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=palette.get("gray", "#595959"),
                    linewidth=2.0,
                    label="gray line: boundary of the seed-robust-win region",
                )
            )
        ax.legend(
            handles=ring_handles,
            loc="upper center",
            bbox_to_anchor=(0.50, -0.13),
            fontsize=LEGEND_FONT_SIZE,
            framealpha=0.88,
            ncol=1,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "power-regime map unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _missingness_model(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    phase_probs = np.asarray(
        summary.get("phase_short_gap_probability") or [], dtype=float
    )
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 8.4))
    if phase_probs.size:
        axes[0].plot(
            np.arange(len(phase_probs)),
            phase_probs,
            marker="o",
            color=palette["orange"],
        )
        axes[0].axhline(
            float(summary.get("short_gap_rate", 0.0)),
            color=palette["gray"],
            linestyle="--",
            linewidth=0.9,
            label="pooled rate",
        )
        axes[0].set_ylim(0.0, 1.05)
        axes[0].set_xlabel("time-of-day phase bin")
        axes[0].set_ylabel("P(short inter-report gap)")
        axes[0].set_title(
            "Phase-conditioned reporting bursts",
            loc="left",
            fontsize=PANEL_HEADING_FONT_SIZE,
        )
        axes[0].legend(loc="best", fontsize=LEGEND_FONT_SIZE)
        labels = ["phase", "traveller", "phase+traveller", "permuted phase"]
        values = [
            float(summary.get("phase_gain_vs_pooled", 0.0)),
            float(summary.get("traveller_gain_vs_pooled", 0.0)),
            float(summary.get("phase_traveller_gain_vs_pooled", 0.0)),
            float(summary.get("permuted_phase_gain_vs_pooled", 0.0)),
        ]
        pos = np.arange(len(values))
        axes[1].barh(
            pos,
            values,
            color=[
                palette["blue"],
                palette["green"],
                palette["purple"],
                palette["gray"],
            ],
        )
        axes[1].axvline(0.0, color=palette["gray"], linewidth=0.9)
        axes[1].set_yticks(pos)
        axes[1].set_yticklabels(labels, fontsize=TICK_FONT_SIZE)
        axes[1].set_xlabel("log-loss gain vs pooled")
        for y, value in zip(pos, values, strict=True):
            offset = 0.001 if value >= 0 else 0.001
            axes[1].text(
                value + offset,
                y,
                f"{value:+.3f}",
                va="center",
                ha="left",
                fontsize=ANNOTATION_FONT_SIZE,
                color="#1D1D1D",
            )
        axes[1].set_title(
            f"Reporting-process gate: {str(summary.get('verdict', 'unavailable')).replace('_', ' ')}",
            loc="left",
            fontsize=PANEL_HEADING_FONT_SIZE,
        )
    else:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "missingness model unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        "Synthetic reporting-process and missingness structure",
        fontsize=CANVAS_TITLE_FONT_SIZE,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def _model_gate_matrix(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    rows = [row for row in summary.get("rows", []) if isinstance(row, dict)]
    fig, ax = plt.subplots(figsize=(14.2, 5.6))
    if rows:
        lanes = sorted({str(row.get("lane", "unknown")) for row in rows})
        gates = [
            ("sequence_baseline_ladder", "Sequence baseline"),
            ("reporting_process_missingness", "Reporting process"),
            ("return_plus_phase_prior", "Return + phase prior"),
        ]
        lookup = {(str(row.get("gate")), str(row.get("lane"))): row for row in rows}
        ax.set_xlim(0, len(lanes))
        ax.set_ylim(0, len(gates))
        ax.invert_yaxis()
        ax.set_xticks(np.arange(len(lanes)) + 0.5)
        ax.set_xticklabels(
            [lane.title() for lane in lanes],
            fontsize=TICK_FONT_SIZE,
            fontweight="semibold",
        )
        ax.set_yticks(np.arange(len(gates)) + 0.5)
        ax.set_yticklabels(
            [label for _, label in gates],
            fontsize=TICK_FONT_SIZE,
            fontweight="semibold",
        )
        ax.tick_params(length=0)
        for row_idx, (gate, _label) in enumerate(gates):
            for lane_idx, lane in enumerate(lanes):
                row = lookup.get((gate, lane), {})
                status = row.get("status", "unavailable")
                value = row.get("primary_metric")
                verdict = str(row.get("verdict", "unavailable"))
                face = _status_color(status, palette)
                ax.add_patch(
                    plt.Rectangle(
                        (lane_idx + 0.04, row_idx + 0.08),
                        0.92,
                        0.84,
                        facecolor=face,
                        alpha=0.18 if status == "completed" else 0.12,
                        edgecolor=face,
                        linewidth=1.0,
                    )
                )
                metric = (
                    "unavailable"
                    if not isinstance(value, int | float)
                    else f"{float(value):.3g}"
                )
                ax.text(
                    lane_idx + 0.5,
                    row_idx + 0.36,
                    metric,
                    ha="center",
                    va="center",
                    fontsize=METRIC_FONT_SIZE,
                    fontweight="bold",
                    color="#111111",
                )
                ax.text(
                    lane_idx + 0.5,
                    row_idx + 0.67,
                    _clean_label(verdict, width=24),
                    ha="center",
                    va="center",
                    fontsize=SMALL_LABEL_FONT_SIZE,
                    color="#333333",
                )
        for x in range(len(lanes) + 1):
            ax.axvline(x, color=palette["gray"], linewidth=0.5, alpha=0.25)
        for y in range(len(gates) + 1):
            ax.axhline(y, color=palette["gray"], linewidth=0.5, alpha=0.25)
        ax.set_title(
            f"Lane-resolved model gates: {summary.get('status', 'unavailable')}",
            loc="left",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "model gate matrix unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _m10_forward_gates(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    rows = [
        row
        for row in summary.get("rows", [])
        if isinstance(row, dict) and row.get("lane") == "fixture"
    ]
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 10.2), height_ratios=[1.0, 1.05])
    if rows:
        names = [_clean_label(row.get("gate", "unknown"), width=20) for row in rows]
        values = [
            float(row["primary_metric"])
            if isinstance(row.get("primary_metric"), int | float)
            else np.nan
            for row in rows
        ]
        y = np.arange(len(rows))
        axes[0].barh(
            y,
            [1.0] * len(rows),
            color=[_status_color(row.get("status"), palette) for row in rows],
            alpha=0.22,
            edgecolor=[_status_color(row.get("status"), palette) for row in rows],
            linewidth=1.0,
        )
        for pos, row, value in zip(y, rows, values):
            metric = "unavailable" if np.isnan(value) else f"{float(value):.3g}"
            label = str(row.get("primary_metric_label", "metric")).replace("_", " ")
            axes[0].text(
                0.04,
                pos,
                label,
                va="center",
                ha="left",
                fontsize=SMALL_LABEL_FONT_SIZE,
                color="#333333",
            )
            axes[0].text(
                0.96,
                pos,
                metric,
                va="center",
                ha="right",
                fontsize=METRIC_FONT_SIZE,
                fontweight="bold",
            )
        axes[0].set_xlim(0.0, 1.0)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(names, fontsize=TICK_FONT_SIZE)
        axes[0].invert_yaxis()
        axes[0].set_xticks([])
        axes[0].set_title("Canonical fixture metrics", loc="left")

        status_labels = [
            str(row.get("verdict", "unknown")).replace("_", " ") for row in rows
        ]
        axes[1].barh(
            y,
            [1.0] * len(rows),
            color=[_status_color(row.get("status"), palette) for row in rows],
            alpha=0.20,
            edgecolor=[_status_color(row.get("status"), palette) for row in rows],
            linewidth=1.0,
        )
        for pos, label in zip(y, status_labels):
            axes[1].text(
                0.03,
                pos,
                _clean_label(label, width=72),
                va="center",
                fontsize=SMALL_LABEL_FONT_SIZE,
            )
        axes[1].set_xlim(0.0, 1.0)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(names, fontsize=TICK_FONT_SIZE)
        axes[1].invert_yaxis()
        axes[1].set_xticks([])
        axes[1].set_title("Verdicts: nulls stay visible", loc="left")
    else:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "forward gates unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        f"M10+ promoted forward gates: {summary.get('status', 'unavailable')}",
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def _efe_location_frontier(summary: dict[str, object], path: Path) -> None:
    from .efe_location_frontier import render_frontier_figure

    render_frontier_figure(summary, path)


def _linkage_risk_frontier(summary: dict[str, object], path: Path) -> None:
    from .linkage_risk_frontier import render_linkage_risk_figure

    render_linkage_risk_figure(summary, path)


def _poisoning_robustness(summary: dict[str, object], path: Path) -> None:
    from .poisoning_robustness import render_poisoning_figure

    render_poisoning_figure(summary, path)


def _privacy_frontier(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    rows = summary.get("frontier") or []
    names = [str(row["modalities"]) for row in rows]
    unicity = [float(row["unicity"]) for row in rows]
    cotravel = [float(row["co_travel_coincidence"]) for row in rows]
    pos = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    axes[0].bar(pos, unicity, color=palette["orange"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("distinct signatures / travellers")
    axes[0].set_title("Synthetic trace unicity")
    axes[1].bar(pos, cotravel, color=palette["blue"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("traveller-pair coincidence")
    axes[1].set_title("Co-travel coincidence")
    for ax in axes:
        ax.set_xticks(pos)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=TICK_FONT_SIZE)
    for ax, values in zip(axes, [unicity, cotravel], strict=True):
        for x, value in zip(pos, values, strict=True):
            label_inside = value >= 0.90
            ax.text(
                x,
                value - 0.045 if label_inside else value + 0.025,
                f"{value:.2f}",
                ha="center",
                va="top" if label_inside else "bottom",
                fontsize=SMALL_LABEL_FONT_SIZE,
                fontweight="bold",
                color="white" if label_inside else "#111111",
            )
    fig.suptitle("Privacy / minimization frontier (oversight-positive)")
    fig.savefig(path)
    plt.close(fig)


def _hardened_anomaly_detection(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    label_by_key = {
        "speed_regime_shift": "speed\nshift",
        "preferred_set_drift": "preferred\ndrift",
        "timing_manipulation": "timing\nmanipulation",
        "adversarial_missingness_proxy": "adversarial\nmissingness",
    }
    ordered_keys = [
        key
        for key in label_by_key
        if isinstance(summary.get(key), dict)
        and "vfe_auc" in summary[key]
        and "markov_auc" in summary[key]
    ]
    ordered_keys.extend(
        sorted(
            key
            for key, value in summary.items()
            if key not in label_by_key
            and isinstance(value, dict)
            and "vfe_auc" in value
            and "markov_auc" in value
        )
    )
    labels = [label_by_key.get(key, key.replace("_", "\n")) for key in ordered_keys]
    vfe = [float(summary.get(key, {}).get("vfe_auc", 0.0)) for key in ordered_keys]
    markov = [
        float(summary.get(key, {}).get("markov_auc", 0.0)) for key in ordered_keys
    ]
    pos = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)
    ax.bar(
        pos - width / 2, vfe, width=width, color=palette["purple"], label="VFE surprise"
    )
    ax.bar(
        pos + width / 2,
        markov,
        width=width,
        color=palette["gray"],
        label="Markov surprise",
    )
    ax.axhline(0.5, color=palette["gray"], linestyle="--", linewidth=0.9)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Hardened anomaly probes")
    ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    fig.savefig(path)
    plt.close(fig)


def _factorial_metrics(summary: dict[str, object]) -> dict[str, object]:
    metrics = summary.get("metrics") if isinstance(summary, dict) else None
    return metrics if isinstance(metrics, dict) else {}


def _factorial_pymdp_modes(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    metrics = _factorial_metrics(summary)
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.4), constrained_layout=True)
    if metrics:
        factor_labels = ["cell", "movement\nmode", "cognitive\nproxy"]
        entropies = [
            float(metrics.get("factorial_pymdp_cell_mean_entropy", 0.0)),
            float(metrics.get("factorial_pymdp_movement_mode_mean_entropy", 0.0)),
            float(metrics.get("factorial_pymdp_cognitive_proxy_mean_entropy", 0.0)),
        ]
        positions = np.arange(len(factor_labels))
        axes[0].bar(
            positions,
            entropies,
            color=[palette["blue"], palette["green"], palette["purple"]],
        )
        factor_sizes = (
            summary.get("factor_sizes") if isinstance(summary, dict) else None
        )
        if not isinstance(factor_sizes, list):
            factor_sizes = [16, 3, 3]
        max_entropies = [
            float(np.log(max(int(size), 2))) for size in factor_sizes[: len(entropies)]
        ]
        for xpos, max_entropy in zip(positions, max_entropies):
            axes[0].hlines(
                max_entropy,
                xpos - 0.34,
                xpos + 0.34,
                color="#111111",
                linestyle="--",
                linewidth=1.1,
            )
        if max_entropies:
            axes[0].set_ylim(0.0, max(max(entropies), max(max_entropies)) * 1.16)
        if len(entropies) >= 3 and len(max_entropies) >= 3 and max_entropies[2] > 0:
            retained = float(entropies[2]) / float(max_entropies[2])
            axes[0].annotate(
                f"retains {retained:.0%} of max uncertainty",
                xy=(positions[2], entropies[2]),
                xytext=(-12, 26),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": palette["gray"], "lw": 0.9},
                ha="right",
                va="bottom",
                fontsize=SMALL_LABEL_FONT_SIZE,
                color="#333333",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 1.5,
                },
            )
        # Improvement #6: the near-zero "cell"/"movement mode" bars otherwise
        # read as possibly-missing data next to the annotated cognitive-proxy
        # bar — label them explicitly instead of leaving them silent.
        for idx in (0, 1):
            if (
                idx < len(entropies)
                and idx < len(max_entropies)
                and max_entropies[idx] > 0
                and entropies[idx] < max_entropies[idx] * 0.05
            ):
                axes[0].annotate(
                    "≈0, fully resolved",
                    xy=(positions[idx], entropies[idx]),
                    xytext=(0, 18),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=ANNOTATION_FONT_SIZE,
                    color=palette["gray"],
                )
        axes[0].legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#111111",
                    linestyle="--",
                    linewidth=1.1,
                    label="max possible entropy (ln K)",
                )
            ],
            loc="upper right",
            fontsize=LEGEND_FONT_SIZE,
            framealpha=0.86,
        )
        axes[0].set_xticks(positions)
        axes[0].set_xticklabels(factor_labels, fontsize=TICK_FONT_SIZE)
        axes[0].set_ylabel("mean posterior entropy (nats)")
        axes[0].set_title("Per-factor posterior entropy")
        bar_labels = [
            "movement\nconfidence",
            "movement\nstability",
            "cognitive\nconfidence",
            "cognitive\nstability",
        ]
        values = [
            float(metrics.get("factorial_pymdp_mean_movement_mode_confidence", 0.0)),
            float(metrics.get("factorial_pymdp_movement_mode_stability", 0.0)),
            float(metrics.get("factorial_pymdp_mean_cognitive_proxy_confidence", 0.0)),
            float(metrics.get("factorial_pymdp_cognitive_proxy_stability", 0.0)),
        ]
        pos2 = np.arange(len(bar_labels))
        # Colour by FACTOR (movement=green, cognitive=purple) so the 2x2 structure
        # reads at a glance; distinguish confidence vs stability within each pair
        # with a hatch on the stability bars rather than an alpha wash (which
        # desaturates the colourblind-safe hues).
        mode_bars = axes[1].bar(
            pos2,
            values,
            color=[
                palette["green"],
                palette["green"],
                palette["purple"],
                palette["purple"],
            ],
        )
        for stability_idx in (1, 3):
            mode_bars[stability_idx].set_hatch("//")
            mode_bars[stability_idx].set_edgecolor("white")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].set_xticks(pos2)
        axes[1].set_xticklabels(bar_labels, fontsize=TICK_FONT_SIZE)
        axes[1].set_ylabel("posterior confidence / label stability")
        axes[1].set_title("Movement and cognitive-proxy modes")
    else:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "factorial pymdp unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        "Factorial pymdp latent-mode diagnostics (synthetic cognitive-proxy states)"
    )
    fig.savefig(path)
    plt.close(fig)


def _latent_mode_recovery(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    metrics = _factorial_metrics(summary)
    raw_matched = metrics.get("factorial_pymdp_matched_prior_max_abs_delta")
    raw_gap = metrics.get("factorial_pymdp_mean_field_exact_gap")
    # Require both validation metrics to be genuinely present and numeric. A
    # partial artifact missing them must surface "unavailable", never plot a
    # near-perfect 1e-12 floor that reads as a flawless validation.
    have_validation = isinstance(raw_matched, int | float) and isinstance(
        raw_gap, int | float
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    if metrics and have_validation:
        matched = max(float(raw_matched), 1e-12)
        gap = max(float(raw_gap), 1e-12)
        axes[0].bar([0, 1], [matched, gap], color=[palette["green"], palette["orange"]])
        axes[0].set_yscale("log")
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(
            ["pymdp vs\nmean-field", "mean-field\nvs exact"], fontsize=TICK_FONT_SIZE
        )
        axes[0].set_ylabel("max abs marginal delta (log)")
        axes[0].set_title("Validation vs factorization cost")
        factor_names = summary.get("hidden_factors") or [
            "cell",
            "movement_mode",
            "cognitive_proxy_mode",
        ]
        factor_sizes = summary.get("factor_sizes") or [9, 3, 3]
        entropy_keys = (
            "factorial_pymdp_cell_mean_entropy",
            "factorial_pymdp_movement_mode_mean_entropy",
            "factorial_pymdp_cognitive_proxy_mean_entropy",
        )
        sharpness = []
        for key, size in zip(entropy_keys, factor_sizes):
            max_entropy = float(np.log(max(int(size), 2)))
            sharpness.append(max(0.0, 1.0 - float(metrics.get(key, 0.0)) / max_entropy))
        positions = np.arange(len(factor_names))
        axes[1].bar(
            positions,
            sharpness,
            color=[palette["blue"], palette["green"], palette["purple"]],
        )
        axes[1].set_ylim(0.0, 1.05)
        axes[1].set_xticks(positions)
        axes[1].set_xticklabels(
            [str(name).replace("_", "\n") for name in factor_names],
            fontsize=TICK_FONT_SIZE,
        )
        axes[1].set_ylabel("recovery sharpness")
        axes[1].set_title("Per-factor recovery sharpness")
    else:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "factorial pymdp unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("Latent-mode recovery and inference validation")
    fig.savefig(path)
    plt.close(fig)


def _factorial_model_gate_matrix(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    metrics = _factorial_metrics(summary)
    rows = (
        [row for row in summary.get("model_rows", []) if isinstance(row, dict)]
        if isinstance(summary, dict)
        else []
    )
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    losses: list[tuple[str, float]] = []
    for row in rows:
        value = row.get("mean_predictive_loss")
        if isinstance(value, int | float):
            losses.append((str(row.get("model", "model")), float(value)))
    persistence = metrics.get("persistence_mean_predictive_loss")
    if metrics and losses:
        best = str(summary.get("best_model", ""))
        names = [name for name, _ in losses]
        values = [value for _, value in losses]
        colors = [
            palette["green"] if name == best else palette["blue"] for name in names
        ]
        positions = np.arange(len(names))
        ax.bar(positions, values, color=colors)
        if isinstance(persistence, int | float):
            ax.axhline(
                float(persistence),
                color=palette["gray"],
                linestyle="--",
                linewidth=1.0,
                label="persistence null",
            )
            ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [name.replace("_", "\n") for name in names], fontsize=TICK_FONT_SIZE
        )
        ax.set_ylabel("mean next-observation loss (nats)")
        gate_text = str(summary.get("performance_gate", "factorial gate")).replace(
            "_", " "
        )
        best_model = str(summary.get("best_model", "")).replace("_", " ")
        # Self-define the gate so "cleared current lane" is not misread as an
        # overall win: factorial pymdp leads only its own pymdp lane here; the
        # bars show it losing to the stronger baselines, whose best is named.
        subtitle = (
            f"{gate_text}\n(leads its own pymdp lane; does NOT lead overall"
            + (f" — best model: {best_model}" if best_model else "")
            + ")"
        )
        ax.set_title(subtitle, fontsize=SMALL_LABEL_FONT_SIZE)
    else:
        ax.text(
            0.5,
            0.5,
            "factorial model gate unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Factorial pymdp vs stronger baselines")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _nan_if_missing(value: object) -> float:
    return float(value) if isinstance(value, int | float) else float("nan")


def _evidence_atlas(
    observations: list[dict[str, str]],
    features: list[dict[str, str]],
    metrics: dict[str, object],
    spatial: dict[str, object],
    model_gates: dict[str, object],
    forward_gates: dict[str, object],
    efe: dict[str, object],
    linkage: dict[str, object],
    poisoning: dict[str, object],
    partial_observability: dict[str, object],
    multiplicity: dict[str, object],
    path: Path,
    palette: dict[str, str],
) -> None:
    theme = editorial_palette(palette)
    bins = _atlas_bins(features, spatial)
    state_count = int(spatial.get("default_state_count") or bins * bins)
    traveller_count = len({row.get("ID", "") for row in observations})
    default_row = (
        spatial.get("default_row", {})
        if isinstance(spatial.get("default_row"), dict)
        else {}
    )
    fig = plt.figure(figsize=(13.6, 7.5), facecolor=theme["paper"])
    axis = fig.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(patches.Rectangle((0, 0), 1, 1, color=theme["paper"], zorder=0))
    for x in np.linspace(0.045, 0.955, 18):
        axis.plot(
            [x, x],
            [0.11, 0.86],
            color=theme["grid"],
            alpha=0.18,
            linewidth=0.5,
            zorder=0,
        )

    axis.text(
        0.045,
        0.93,
        "Evidence Atlas: From Sparse Reports to Bounded Active-Inference Claims",
        ha="left",
        va="top",
        fontsize=CANVAS_TITLE_FONT_SIZE,
        fontweight="bold",
        color=theme["ink"],
    )
    axis.text(
        0.045,
        0.885,
        "One data-bound frame links waypoint evidence, belief updating, null gates, minimization, and integrity checks.",
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        color=theme["muted"],
    )

    _draw_evidence_lattice(axis, observations, features, bins, theme)
    _draw_atlas_flow(axis, theme)
    _draw_atlas_guardrails(
        axis,
        theme,
        [
            (
                "model matrix",
                _atlas_label(model_gates.get("status", "unknown")),
                _atlas_status_color(model_gates.get("status", "unknown"), theme),
            ),
            (
                "forward gates",
                _atlas_label(forward_gates.get("status", "unknown")),
                _atlas_status_color(forward_gates.get("status", "unknown"), theme),
            ),
            (
                "location EFE",
                _atlas_label(efe.get("verdict", "unknown")),
                theme["amber"],
            ),
            (
                "linkage risk",
                _atlas_label(linkage.get("verdict", "unknown")),
                theme["red"],
            ),
            (
                "poisoning",
                _atlas_label(poisoning.get("verdict", "unknown")),
                theme["violet"],
            ),
        ],
    )
    _draw_atlas_metric_strip(
        axis,
        theme,
        [
            ("reports", str(len(observations))),
            ("travellers", str(traveller_count)),
            ("state surface", f"{bins}x{bins} / {state_count}"),
            ("loss", _fmt_plain(default_row.get("full_location_mean_predictive_loss"))),
            (
                "no-location mass",
                _fmt_plain(default_row.get("no_location_mean_mass_on_true_cell")),
            ),
            (
                "pymdp delta",
                _fmt_delta(metrics.get("pymdp_exact_inference_max_abs_delta")),
            ),
            (
                "PO adv (fair)",
                f"+{_fmt_plain(partial_observability.get('max_noise_advantage_nats'))} nats",
            ),
            (
                "Holm survivors",
                (
                    f"{int(multiplicity.get('holm', {}).get('n_survivors', 0))}/"
                    f"{int(multiplicity.get('family_size', 0) or 0)}"
                )
                if isinstance(multiplicity.get("holm"), dict)
                else "n/a",
            ),
        ],
    )
    axis.text(
        0.045,
        0.018,
        "Synthetic-only orientation: this atlas previews the manuscript evidence chain; it is not a real-map, surveillance, tasking, or empirical mobility-validity claim.",
        ha="left",
        va="center",
        fontsize=ANNOTATION_FONT_SIZE,
        color=theme["muted"],
    )
    fig.savefig(
        path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.02
    )
    plt.close(fig)


def _atlas_bins(features: list[dict[str, str]], spatial: dict[str, object]) -> int:
    configured = spatial.get("default_spatial_bins")
    if isinstance(configured, int | float) and int(configured) > 0:
        return int(configured)
    try:
        return (
            max(
                max(int(float(row.get("x_bin", 0))) for row in features),
                max(int(float(row.get("y_bin", 0))) for row in features),
            )
            + 1
        )
    except ValueError:
        return 4


def _draw_evidence_lattice(
    axis,
    observations: list[dict[str, str]],
    features: list[dict[str, str]],
    bins: int,
    theme: dict[str, str],
) -> None:
    left, bottom, width, height = 0.055, 0.245, 0.35, 0.5
    axis.text(
        left,
        bottom + height + 0.055,
        "1. Sparse reports as finite evidence",
        ha="left",
        fontsize=PANEL_HEADING_FONT_SIZE,
        fontweight="bold",
        color=theme["ink"],
    )
    axis.text(
        left,
        bottom + height + 0.025,
        "Waypoints binned before any belief update.",
        ha="left",
        fontsize=ANNOTATION_FONT_SIZE,
        color=theme["muted"],
    )
    counts = np.zeros((bins, bins), dtype=float)
    for row in features:
        try:
            x_bin = min(bins - 1, max(0, int(float(row.get("x_bin", 0)))))
            y_bin = min(bins - 1, max(0, int(float(row.get("y_bin", 0)))))
        except ValueError:
            continue
        counts[y_bin, x_bin] += 1.0
    max_count = max(float(counts.max()), 1.0)
    cmap = LinearSegmentedColormap.from_list(
        "afr_atlas_grid", ["#FAF6E8", "#DCEADF", "#9CCDBD", theme["teal"]]
    )
    cell_w = width / bins
    cell_h = height / bins
    for y_idx in range(bins):
        for x_idx in range(bins):
            axis.add_patch(
                patches.Rectangle(
                    (left + x_idx * cell_w, bottom + y_idx * cell_h),
                    cell_w,
                    cell_h,
                    facecolor=cmap(counts[y_idx, x_idx] / max_count),
                    edgecolor=theme["grid"],
                    linewidth=1.0,
                    zorder=1,
                )
            )
    if observations:
        xs = np.array([float(row["x(km)"]) for row in observations])
        ys = np.array([float(row["y(km)"]) for row in observations])
        x_span = max(float(xs.max() - xs.min()), 1.0)
        y_span = max(float(ys.max() - ys.min()), 1.0)
        colors = [theme["blue"], theme["green"], theme["amber"], theme["violet"]]
        by_id: dict[str, list[dict[str, str]]] = {}
        for row in observations:
            by_id.setdefault(row.get("ID", ""), []).append(row)
        for idx, (_, rows) in enumerate(sorted(by_id.items())):
            ordered = sorted(rows, key=lambda row: float(row.get("Days", 0.0)))
            x = [
                left + (float(row["x(km)"]) - float(xs.min())) / x_span * width
                for row in ordered
            ]
            y = [
                bottom + (float(row["y(km)"]) - float(ys.min())) / y_span * height
                for row in ordered
            ]
            color = colors[idx % len(colors)]
            axis.plot(x, y, color=color, alpha=0.52, linewidth=1.6, zorder=2)
            stride = max(1, len(x) // 24)
            axis.scatter(
                x[::stride],
                y[::stride],
                s=18,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                alpha=0.82,
                zorder=3,
            )
    axis.add_patch(
        patches.Rectangle(
            (left, bottom),
            width,
            height,
            fill=False,
            edgecolor=theme["ink"],
            linewidth=1.1,
            zorder=4,
        )
    )


def _draw_atlas_flow(axis, theme: dict[str, str]) -> None:
    x0, y0 = 0.455, 0.725
    rows = [
        ("validated reports", "observations.csv", theme["teal"]),
        ("categorical evidence", "location | time | speed | burst", theme["blue"]),
        ("belief update", "Bayes reference + pymdp", theme["green"]),
        ("null hierarchy", "calibration | persistence | controls", theme["amber"]),
        ("safety outputs", "minimization | linkage | integrity", theme["violet"]),
    ]
    axis.text(
        x0,
        y0 + 0.065,
        "2. Analytic loop",
        ha="left",
        fontsize=PANEL_HEADING_FONT_SIZE,
        fontweight="bold",
        color=theme["ink"],
    )
    for idx, (label, detail, color) in enumerate(rows):
        y = y0 - idx * 0.105
        axis.add_patch(
            patches.FancyBboxPatch(
                (x0, y - 0.035),
                0.235,
                0.069,
                boxstyle="round,pad=0.01,rounding_size=0.012",
                facecolor=theme["panel"],
                edgecolor=color,
                linewidth=1.25,
                zorder=3,
            )
        )
        axis.text(
            x0 + 0.016,
            y + 0.012,
            label,
            ha="left",
            va="center",
            fontsize=SMALL_LABEL_FONT_SIZE,
            fontweight="bold",
            color=theme["ink"],
            zorder=4,
        )
        axis.text(
            x0 + 0.016,
            y - 0.014,
            detail,
            ha="left",
            va="center",
            fontsize=SMALL_LABEL_FONT_SIZE,
            color=theme["muted"],
            zorder=4,
        )
        if idx < len(rows) - 1:
            axis.annotate(
                "",
                xy=(x0 + 0.118, y - 0.064),
                xytext=(x0 + 0.118, y - 0.038),
                arrowprops={"arrowstyle": "->", "color": theme["muted"], "lw": 1.0},
                zorder=2,
            )


def _draw_atlas_guardrails(
    axis, theme: dict[str, str], rows: list[tuple[str, str, str]]
) -> None:
    x0, y0 = 0.745, 0.725
    axis.text(
        x0,
        y0 + 0.065,
        "3. Claim gates and guardrails",
        ha="left",
        fontsize=PANEL_HEADING_FONT_SIZE,
        fontweight="bold",
        color=theme["ink"],
    )
    for idx, (label, value, color) in enumerate(rows):
        y = y0 - idx * 0.092
        axis.add_patch(
            patches.Circle(
                (x0 + 0.012, y + 0.004),
                0.009,
                facecolor=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
        )
        axis.text(
            x0 + 0.034,
            y + 0.018,
            label,
            ha="left",
            va="center",
            fontsize=SMALL_LABEL_FONT_SIZE,
            fontweight="bold",
            color=theme["ink"],
        )
        axis.text(
            x0 + 0.034,
            y - 0.012,
            _shorten(value, 42),
            ha="left",
            va="center",
            fontsize=SMALL_LABEL_FONT_SIZE,
            color=theme["muted"],
        )


def _draw_atlas_metric_strip(
    axis, theme: dict[str, str], rows: list[tuple[str, str]]
) -> None:
    left, bottom = 0.055, 0.126
    per_row = 4 if len(rows) > 6 else len(rows)
    step = 0.91 / max(per_row, 1)
    box_width = min(0.195, step - 0.018)
    box_height = 0.066
    row_gap = 0.080
    axis.text(
        left,
        bottom + 0.091,
        "Current run surface",
        ha="left",
        va="center",
        fontsize=PANEL_HEADING_FONT_SIZE,
        fontweight="bold",
        color=theme["ink"],
    )
    for idx, (label, value) in enumerate(rows):
        row_idx = idx // per_row
        col_idx = idx % per_row
        x = left + col_idx * step
        y_bottom = bottom - row_idx * row_gap
        multiline_label = "\n" in label
        axis.add_patch(
            patches.FancyBboxPatch(
                (x - 0.006, y_bottom - 0.004),
                box_width,
                box_height,
                boxstyle="round,pad=0.009,rounding_size=0.01",
                facecolor=theme["panel"],
                edgecolor=theme["grid"],
                linewidth=0.9,
                zorder=1,
            )
        )
        axis.text(
            x + 0.004,
            y_bottom + (0.052 if multiline_label else 0.042),
            label,
            ha="left",
            va="center",
            fontsize=SMALL_LABEL_FONT_SIZE,
            color=theme["muted"],
            zorder=2,
        )
        axis.text(
            x + 0.004,
            y_bottom + (0.006 if multiline_label else 0.014),
            value,
            ha="left",
            va="center",
            fontsize=METRIC_FONT_SIZE,
            fontweight="bold",
            color=theme["ink"],
            zorder=2,
        )


def _shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _multiplicity_ledger(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    """Uncorrected vs Holm-adjusted p for every directional comparison in the family."""

    rows = summary.get("comparisons") or []
    # Height scales with row count: several comparison_id labels exceed
    # _clean_label's 26-char wrap width and render as 2 lines, which at the
    # previous fixed figsize(12.8, 7.8) didn't leave enough per-row vertical
    # budget and adjacent wrapped labels visually collided.
    n_rows = len(rows) if isinstance(rows, list) else 0
    fig, ax = plt.subplots(figsize=(12.8, max(7.8, 1.15 * max(n_rows, 1))))
    if not isinstance(rows, list) or not rows:
        ax.text(
            0.5,
            0.5,
            "multiplicity ledger unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    alpha = float(summary.get("alpha", 0.05))
    ordered = sorted(rows, key=lambda r: float(r.get("uncorrected_p", 1.0)))
    labels = [_clean_label(r.get("comparison_id", "?"), width=26) for r in ordered]
    floor = 1e-3
    raw = [max(float(r.get("uncorrected_p", 1.0)), floor) for r in ordered]
    holm = [max(float(r.get("holm_adjusted_p", 1.0)), floor) for r in ordered]
    y = np.arange(len(ordered))

    ax.hlines(
        y, raw, holm, color=palette.get("gray", "#888888"), lw=2.0, alpha=0.6, zorder=1
    )
    ax.scatter(
        raw,
        y,
        s=70,
        color=palette.get("blue", "#2c6fbb"),
        label="uncorrected p",
        zorder=3,
    )
    holm_colors = [
        palette.get("green", "#2e8b57")
        if bool(r.get("holm_survives"))
        else palette.get("red", "#c0392b")
        for r in ordered
    ]
    ax.scatter(holm, y, s=70, marker="D", color=holm_colors, zorder=3)
    ax.axvline(alpha, color="#111111", ls="--", lw=1.3, zorder=2)
    ax.annotate(
        f"alpha={alpha:g}",
        xy=(alpha, 0.04),
        xycoords=("data", "axes fraction"),
        xytext=(6, 0),
        textcoords="offset points",
        va="bottom",
        ha="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
        color="#111111",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
    )
    for idx, (yi, r) in enumerate(zip(y, ordered)):
        if bool(r.get("individually_clears")):
            ax.text(
                floor * 1.28,
                yi,
                "clears uncorrected",
                ha="left",
                va="center",
                fontsize=SMALL_LABEL_FONT_SIZE,
                color=palette.get("blue", "#2c6fbb"),
                fontweight="bold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 1.4,
                },
                clip_on=False,
            )
        elif float(r.get("uncorrected_p", 1.0)) < alpha:
            # p is left of the alpha line but the clustered CI straddles zero, so
            # it is NOT counted as an uncorrected clearer — label it in place so
            # the p<alpha circle does not read as contradicting "clearers=0".
            # Anchored past this row's own Holm-adjusted marker (holm[idx], which
            # is >= the uncorrected p by construction) rather than past the
            # uncorrected marker, so the label can never occlude either marker or
            # the connecting line for this row (see NOTE(F6-12) below for the
            # same occlusion bug in the sota-lane annotation).
            ax.text(
                holm[idx] * 1.28,
                yi,
                "p<alpha but CI straddles 0",
                ha="left",
                va="center",
                fontsize=ANNOTATION_FONT_SIZE,
                color=palette.get("gray", "#595959"),
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 1.2,
                },
                clip_on=False,
            )
        elif raw[idx] >= 0.98 and holm[idx] >= 0.999:
            # Both markers round to ~1 and overlap almost exactly at this
            # scale, so the dumbbell (circle + line + diamond) visually
            # collapses to a single diamond. Annotate explicitly rather than
            # relying on the reader to infer near-equality from marker
            # occlusion.
            ax.text(
                holm[idx] * 1.28,
                yi,
                "p ≈ p_adj ≈ 1",
                ha="left",
                va="center",
                fontsize=ANNOTATION_FONT_SIZE,
                color=palette.get("gray", "#595959"),
                fontstyle="italic",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 1.2,
                },
                clip_on=False,
            )
    ax.set_xscale("log")
    x_max = max(max(holm), alpha) * 1.7
    ax.set_xlim(floor * 0.72, x_max)
    ax.axvspan(alpha, x_max, color=palette.get("gray", "#888888"), alpha=0.08, zorder=0)
    # NOTE(F6-12): the "Holm penalty" explanation previously lived in an
    # ax.annotate arrow box anchored on the top dumbbell — it occluded the
    # sota-lane Holm diamond and its connector. The explanation now lives as a
    # line in the bottom-left info box (fig.text below), which sits in reserved
    # whitespace and cannot cover any marker.
    ax.set_yticks(y)
    # linespacing=0.85 (matplotlib default 1.2) tightens 2-line wrapped
    # labels so they fit the per-row vertical budget without needing as much
    # figure-height growth; combined with the taller figsize above.
    ax.set_yticklabels(labels, fontsize=TICK_FONT_SIZE, linespacing=0.85)
    ax.set_xlabel(
        "one-sided p-value (log scale; lower = stronger)", fontsize=METRIC_FONT_SIZE
    )
    holm_survivors = int(summary.get("holm", {}).get("n_survivors", 0))
    bh_survivors = int(summary.get("benjamini_hochberg", {}).get("n_survivors", 0))
    family_size = int(summary.get("family_size", len(ordered)))
    headline = (
        summary.get("headline", {}) if isinstance(summary.get("headline"), dict) else {}
    )
    partial = (
        summary.get("partial_observability_positive", {})
        if isinstance(summary.get("partial_observability_positive"), dict)
        else {}
    )
    ax.set_title(
        f"Multiplicity correction: {holm_survivors}/{family_size} comparisons survive Holm",
        loc="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
        fontweight="bold",
        pad=10,
    )
    fig.text(
        0.02,
        0.055,
        "\n".join(
            [
                f"family={family_size}; uncorrected clearers (p<alpha AND clustered CI>0)={headline.get('n_uncorrected_clearers', 0)}",
                "Holm penalty: p multiplied by rank within family",
                f"Holm survivors={holm_survivors}; BH(q={_fmt_stat(summary.get('fdr_q'), 2)}) survivors={bh_survivors}",
                f"top clearer Holm p={_fmt_stat(headline.get('max_clearer_holm_adjusted_p'))}",
                (
                    "partial-observability sign bound="
                    f"{_fmt_stat(partial.get('sign_test_bound'), 6)}; "
                    f"Bonferroni alpha={_fmt_stat(partial.get('bonferroni_per_comparison_threshold'), 6)}"
                ),
            ]
        ),
        ha="left",
        va="bottom",
        fontsize=ANNOTATION_FONT_SIZE,
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.86, "pad": 4.0},
    )
    survivor_label = (
        "Holm survivor" if holm_survivors else "Holm survivor (none in this family)"
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=palette.get("blue", "#2c6fbb"),
            markeredgecolor=palette.get("blue", "#2c6fbb"),
            markersize=8,
            label="uncorrected p",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=palette.get("red", "#D55E00"),
            markeredgecolor=palette.get("red", "#D55E00"),
            markersize=8,
            label="Holm-adjusted p (fails alpha)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=palette.get("green", "#009E73"),
            markeredgecolor=palette.get("green", "#009E73"),
            markersize=8,
            label=survivor_label,
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.70, -0.17),
        ncol=3,
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
    )
    ax.invert_yaxis()
    fig.tight_layout(rect=(0, 0.24, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def _confound_taxonomy(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    """The map of dissolved active-inference advantages, one marker per entry by confound class."""

    rows = summary.get("entries") or []
    # Wider row pitch (F6-25) plus a taller canvas so staggered entry labels have
    # room and stop running into the surviving-positive star / adjacent markers.
    row_spacing = 2.25
    fig, ax = plt.subplots(figsize=(14.2, 11.6))
    if not isinstance(rows, list) or not rows:
        ax.text(
            0.5,
            0.5,
            "confound taxonomy unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    classes = list(summary.get("confound_classes") or [])
    # place the surviving positive ("none") on its own row at the top
    row_labels = classes + ["none (surviving positive)"]
    class_index = {name: idx for idx, name in enumerate(classes)}
    class_index["none"] = len(classes)
    counts: dict[int, int] = {}
    for entry in rows:
        cls = str(entry.get("confound_class", "none"))
        yi = class_index.get(cls, len(classes))
        x = counts.get(yi, 0)
        y = yi * row_spacing
        counts[yi] = x + 1
        verified = bool(entry.get("verified"))
        cross_vendor = bool(entry.get("cross_vendor_caught"))
        is_positive = cls == "none"
        color = (
            palette.get("green", "#2e8b57")
            if is_positive
            else (
                palette.get("blue", "#2c6fbb")
                if verified
                else palette.get("red", "#c0392b")
            )
        )
        marker = "*" if is_positive else ("D" if cross_vendor else "o")
        size = 320 if is_positive else (150 if cross_vendor else 110)
        ax.scatter(
            x,
            y,
            s=size,
            marker=marker,
            color=color,
            edgecolor="#111111",
            linewidth=0.6,
            zorder=3,
        )
        label_x = x + 0.22 if is_positive else x + (0.18 if cross_vendor else 0.0)
        # Three staggered vertical tiers (F6-25) instead of two, with larger
        # offsets, so labels for adjacent markers no longer overlap each other or
        # the row below's marker.
        label_y = y + 0.18 if is_positive else y + 0.40 + (x % 3) * 0.44
        ax.text(
            label_x,
            label_y,
            _clean_label(entry.get("entry_id", "?"), width=18),
            ha="left" if is_positive else "center",
            va="center" if is_positive else "bottom",
            fontsize=SMALL_LABEL_FONT_SIZE,
            color="#222222",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0},
            clip_on=False,
        )

    ax.set_yticks(np.arange(len(row_labels)) * row_spacing)
    ax.set_yticklabels(
        [_clean_label(name, width=26) for name in row_labels], fontsize=TICK_FONT_SIZE
    )
    ax.set_xticks([])
    max_x = max(counts.values()) if counts else 1
    ax.set_xlim(-0.6, max(max_x + 0.8, 2.2))
    ax.set_ylim(-1.0, (len(row_labels) - 1) * row_spacing + 1.85)
    n_confounds = int(summary.get("n_confounds", 0))
    n_classes = int(summary.get("n_confound_classes_present", 0))
    n_cv = int(summary.get("n_cross_vendor_caught", 0))
    n_verified = int(summary.get("n_verified", 0))
    n_unverified = int(summary.get("n_unverified", 0))
    n_entries = int(summary.get("n_entries", len(rows)))
    n_positive = sum(
        1 for entry in rows if str(entry.get("confound_class", "")) == "none"
    )
    ax.set_title(
        f"Confound map: {n_confounds} dissolved advantages over {n_classes} classes "
        f"({n_cv} cross-vendor-caught); {n_positive} surviving positive",
        loc="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
        fontweight="bold",
        pad=10,
    )
    ax.text(
        0.985,
        0.985,
        "\n".join(
            [
                f"entries={n_entries}; verified={n_verified}; unverified={n_unverified}",
                f"cross-vendor caught={n_cv}; confound classes={n_classes}",
                f"dissolved={n_confounds}; surviving positives={n_positive}",
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.86, "pad": 4.0},
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor=palette.get("green", "#2e8b57"),
            markersize=15,
            markeredgecolor="#111111",
            label="surviving positive",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=palette.get("blue", "#2c6fbb"),
            markersize=10,
            markeredgecolor="#111111",
            label="cross-vendor-caught",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette.get("blue", "#2c6fbb"),
            markersize=10,
            markeredgecolor="#111111",
            label="dissolved (live-verified)",
        ),
        # The scatter loop above renders `verified=False` entries in red (any
        # marker shape) but this legend previously had no entry for that
        # colour at all - an unverified entry (e.g. one whose live-artifact
        # trace isn't cached in this environment) rendered as an unexplained
        # marker. Added to keep the legend exhaustive over every colour the
        # loop can actually produce.
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette.get("red", "#c0392b"),
            markersize=10,
            markeredgecolor="#111111",
            label="not yet live-verified",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


#: mechanism_verdict -> palette key. "trivial"/"confined" verdicts share gray (neither is
#: a real mechanism finding); the remaining five verdicts get distinct colors so a reader
#: can scan the plot by color alone.
_MECHANISM_VERDICT_COLORS: dict[str, str] = {
    "trivial_by_construction": "gray",
    "confined_uninterpretable": "gray",
    "null_no_signal": "sky",
    "loses_to_fair_baseline": "red",
    "confound_explained": "orange",
    "nominal_win_not_multiplicity_robust": "purple",
    "robust_win": "green",
}

_MECHANISM_VERDICT_LABELS: dict[str, str] = {
    "trivial_by_construction": "trivial (negative control)",
    "confined_uninterpretable": "confined (uninterpretable)",
    "null_no_signal": "null (CI straddles zero)",
    "loses_to_fair_baseline": "loses to fair baseline",
    "confound_explained": "confound-explained",
    "nominal_win_not_multiplicity_robust": "nominal win, not multiplicity-robust",
    "robust_win": "robust win",
}


def _regime_mechanism_synthesis(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    """WHERE/HOW MUCH/WHY map: per-regime AIF-vs-fair magnitude+CI, colored by mechanism
    verdict, with confound-battery and mobility standing annotated per row."""

    rows = summary.get("regimes") or []
    fig, ax = plt.subplots(figsize=(13.4, max(4.5, 1.15 * max(len(rows), 1) + 1.5)))
    if not isinstance(rows, list) or not rows:
        ax.text(
            0.5,
            0.5,
            "regime mechanism synthesis unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    # Ascending by mean advantage: the regime AIF loses worst sits at the bottom, the
    # regime AIF does best (if any) sits at the top.
    ordered = sorted(rows, key=lambda r: float(r.get("mean_advantage_nats", 0.0)))
    y = np.arange(len(ordered))
    advantages = [float(r.get("mean_advantage_nats", 0.0)) for r in ordered]
    ci_low = [float(r.get("clustered_ci_low", 0.0)) for r in ordered]
    ci_high = [float(r.get("clustered_ci_high", 0.0)) for r in ordered]
    verdicts = [str(r.get("mechanism_verdict", "unknown")) for r in ordered]
    colors = [
        palette.get(
            _MECHANISM_VERDICT_COLORS.get(v, "gray"), palette.get("gray", "#595959")
        )
        for v in verdicts
    ]

    xerr_low = [max(0.0, adv - lo) for adv, lo in zip(advantages, ci_low)]
    xerr_high = [max(0.0, hi - adv) for adv, hi in zip(advantages, ci_high)]
    ax.errorbar(
        advantages,
        y,
        xerr=[xerr_low, xerr_high],
        fmt="none",
        ecolor=palette.get("gray", "#888888"),
        elinewidth=1.4,
        capsize=4,
        zorder=1,
    )
    ax.scatter(
        advantages, y, s=150, c=colors, edgecolor="#111111", linewidth=0.7, zorder=3
    )
    ax.axvline(0.0, color="#111111", linestyle="--", linewidth=1.0, zorder=0)

    labels = [str(r.get("regime_id", "?")).replace("_", "\n") for r in ordered]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=TICK_FONT_SIZE)
    ax.set_xlabel(
        "AIF advantage over strongest fair baseline (nats; negative = AIF loses)"
    )
    ax.set_title("Where, how much, and why: per-regime AIF mechanism map", loc="left")

    # Explicit x-range with generous right-side padding for the per-row battery/mobility
    # annotation text, sized from the data itself rather than a hardcoded guess (memory:
    # figure-caption-audit / repeated annotation-collision fixes earlier this session).
    x_min = min([*ci_low, 0.0])
    x_max = max([*ci_high, 0.0])
    span = (x_max - x_min) or 1.0
    ax.set_xlim(x_min - span * 0.08, x_max + span * 0.85)

    for yi, row, hi in zip(y, ordered, ci_high):
        # Neither the trivial-by-construction negative control nor a confined-and-thus-
        # uninterpretable regime has confound-battery reasoning in its verdict (per
        # _mechanism_verdict's own explanation text) - annotating either with a battery
        # pass/fail reads as a claim the classifier itself does not make for that row.
        if str(row.get("mechanism_verdict")) in {
            "trivial_by_construction",
            "confined_uninterpretable",
        }:
            continue
        battery = (
            row.get("confound_battery", {})
            if isinstance(row.get("confound_battery"), dict)
            else {}
        )
        clears = bool(battery.get("clears_confounds", False))
        mobility = (
            row.get("mobility", {}) if isinstance(row.get("mobility"), dict) else {}
        )
        mobile = bool(mobility.get("is_mobile", True))
        tag = "beats confound battery" if clears else "fails confound battery"
        if not mobile:
            tag += " · confined"
        ax.annotate(
            tag,
            xy=(hi, yi),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=ANNOTATION_FONT_SIZE,
            color=palette.get("gray", "#595959"),
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette.get(color_key, "#595959"),
            markersize=10,
            markeredgecolor="#111111",
            label=_MECHANISM_VERDICT_LABELS[verdict],
        )
        for verdict, color_key in _MECHANISM_VERDICT_COLORS.items()
        if verdict in verdicts
    ]
    # Upper-left, not lower-right: the per-row battery/mobility annotations occupy the
    # right margin at every row including the bottom one, so a lower-right legend would
    # sit on top of that row's annotation text.
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _regime_switch_phase_diagnostic(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    """WHY, structurally: per-movement-phase AIF-vs-fair loss breakdown on the one
    regime whose generator has a named 3-segment movement-mode schedule, showing WHERE
    within the trajectory (not just pooled) the fair baseline's advantage sits."""

    rows = summary.get("segments") or []
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    if not isinstance(rows, list) or not rows:
        ax.text(
            0.5,
            0.5,
            "regime-switch phase diagnostic unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    # Fixed generator order (commute -> explore -> return_stabilization), not sorted by
    # magnitude: the x-axis is the trajectory's own phase sequence, so a reader can see
    # WHERE in time the pattern sits, not just rank segments by effect size.
    order = {"commute": 0, "explore": 1, "return_stabilization": 2}
    ordered = sorted(rows, key=lambda r: order.get(str(r.get("segment")), 99))
    x = np.arange(len(ordered))
    advantages = [float(r.get("mean_advantage_nats", 0.0)) for r in ordered]
    ci_low = [float(r.get("clustered_ci_low", 0.0)) for r in ordered]
    ci_high = [float(r.get("clustered_ci_high", 0.0)) for r in ordered]
    colors = [
        palette.get("green", "#2e7d32")
        if adv > 0.0
        else palette.get("orange", "#d97706")
        for adv in advantages
    ]

    ax.bar(
        x,
        advantages,
        color=colors,
        edgecolor="#111111",
        linewidth=0.7,
        width=0.55,
        zorder=2,
    )
    yerr_low = [max(0.0, adv - lo) for adv, lo in zip(advantages, ci_low)]
    yerr_high = [max(0.0, hi - adv) for adv, hi in zip(advantages, ci_high)]
    ax.errorbar(
        x,
        advantages,
        yerr=[yerr_low, yerr_high],
        fmt="none",
        ecolor="#111111",
        elinewidth=1.4,
        capsize=5,
        zorder=3,
    )
    ax.axhline(0.0, color="#111111", linestyle="--", linewidth=1.0, zorder=0)

    labels = [str(r.get("segment", "?")).replace("_", "\n") for r in ordered]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FONT_SIZE)
    # Short by design: a rotated y-axis label this long clips against the canvas edge
    # regardless of tight_layout/bbox_inches="tight" (verified empirically - neither
    # rescues it). The full "negative = AIF loses" gloss lives in the figure caption.
    ax.set_ylabel("AIF advantage vs. fair baseline (nats)")
    regime_id = str(summary.get("regime_id", "latent_regime_switch_035"))
    ax.set_title(
        f"Where within {regime_id}'s trajectory the loss to the fair baseline sits",
        loc="left",
        fontsize=PANEL_HEADING_FONT_SIZE,
    )

    y_min = min([*ci_low, 0.0])
    y_max = max([*ci_high, 0.0])
    span = (y_max - y_min) or 1.0
    # Extra headroom above zero (not just below the lowest whisker): every bar here is
    # negative, so the region above y=0 is otherwise empty - that is where both the n=
    # counts and the status box live, instead of competing for space below the whiskers
    # where a prior draft placed both and they overlapped.
    ax.set_ylim(y_min - span * 0.10, y_max + span * 0.55)
    for xi, row, lo, hi in zip(x, ordered, ci_low, ci_high):
        n_steps = int(row.get("n_steps", 0))
        # A zero-crossing whisker occupies the same vertical line as the count.
        # Nudge only that label sideways so the unresolved segment remains
        # legible without implying a different y value.
        crosses_zero = lo < 0.0 < hi
        ax.annotate(
            f"n={n_steps}",
            xy=(xi, 0.0),
            xytext=((10 if crosses_zero else 0), 8),
            textcoords="offset points",
            ha="left" if crosses_zero else "center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
            color=palette.get("gray", "#595959"),
        )

    uniform = bool(summary.get("uniform_disadvantage", False))
    ratio = summary.get("worst_to_best_magnitude_ratio")
    ratio_text = f"{float(ratio):.2f}x" if isinstance(ratio, (int, float)) else "n/a"
    status_text = (
        f"uniform disadvantage: {uniform}  |  worst-to-best magnitude ratio: {ratio_text}\n"
        f"worst: {summary.get('worst_segment', '?')}  best: {summary.get('best_segment', '?')}"
    )
    ax.text(
        0.02,
        0.97,
        status_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "#888888",
            "alpha": 0.9,
        },
    )

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _privacy_utility_frontier(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    """The governance tradeoff: surviving predictive utility vs re-identification risk per regime."""

    rows = summary.get("frontier") or []
    grid_rows = [
        r for r in rows if isinstance(r, dict) and int(r.get("spatial_bins", 0)) > 0
    ]
    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    if not grid_rows:
        ax.text(
            0.5,
            0.5,
            "privacy-utility frontier unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    # fine -> coarse order (more minimization to the right/down); x = re-id risk, y = excess utility
    risk = [float(r["reid_event_uniqueness"]) for r in grid_rows]
    util = [float(r["excess_utility_bits"]) for r in grid_rows]
    labels = [str(r["regime"]).replace("grid_", "") for r in grid_rows]
    by_label = {label: (x, y) for label, x, y in zip(labels, risk, util)}
    ax.plot(
        risk,
        util,
        "-",
        color=palette.get("gray", "#888888"),
        lw=1.6,
        alpha=0.7,
        zorder=1,
    )
    sc = ax.scatter(
        risk,
        util,
        c=range(len(grid_rows)),
        cmap=SEQUENTIAL_CMAP,
        s=130,
        zorder=3,
        edgecolor="#111111",
        linewidth=0.6,
    )
    label_offsets = {
        "4x4": (8, 8),
        "5x5": (8, -14),
        "6x6": (-40, 12),
        "7x7": (10, 12),
        "8x8": (12, -14),
        "10x10": (12, -16),
    }
    for x, y, lab in zip(risk, util, labels):
        offset = label_offsets.get(lab, (8, 8))
        ax.annotate(
            lab,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=SMALL_LABEL_FONT_SIZE,
            color="#222222",
            fontweight="bold",
            ha="right" if offset[0] < 0 else "left",
            va="bottom" if offset[1] >= 0 else "top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
            clip_on=False,
        )
    # withheld-location endpoint at the origin (no location -> no utility, pooled risk)
    ax.scatter(
        [0.0], [0.0], marker="x", s=90, color=palette.get("red", "#c0392b"), zorder=3
    )
    # Improvement #8: center the label directly above the origin marker
    # instead of right-aligning at a negative offset (which bled it past the
    # left axis spine) — this also keeps it clear of the "graceful gate:
    # ..." stats box, whose left edge sits further right.
    ax.annotate(
        "location withheld",
        xy=(0.0, 0.0),
        xytext=(0, 24),
        textcoords="offset points",
        fontsize=SMALL_LABEL_FONT_SIZE,
        color=palette.get("red", "#c0392b"),
        ha="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
        clip_on=False,
    )
    if "7x7" in by_label and "8x8" in by_label:
        x7, y7 = by_label["7x7"]
        x8, y8 = by_label["8x8"]
        if y7 > y8 or x7 > x8:
            ax.annotate(
                "7x7/8x8 bump",
                xy=(x7, y7),
                xytext=(-72, 42),
                textcoords="offset points",
                fontsize=SMALL_LABEL_FONT_SIZE,
                color="#222222",
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.9},
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#BBBBBB",
                    "alpha": 0.86,
                    "pad": 2.0,
                },
                clip_on=False,
            )
    ax.margins(x=0.14, y=0.16)
    ax.set_xlabel(
        "re-identification risk (event uniqueness)", fontsize=METRIC_FONT_SIZE
    )
    ax.set_ylabel("surviving utility (excess bits)", fontsize=METRIC_FONT_SIZE)
    graceful = bool(summary.get("graceful_degradation_tradeoff", False))
    utility_monotone = bool(summary.get("utility_monotone_decreasing", False))
    risk_monotone = bool(summary.get("risk_monotone_decreasing", False))
    min_collection = (
        summary.get("min_collection_operating_point", {})
        if isinstance(summary.get("min_collection_operating_point"), dict)
        else {}
    )
    ax.set_title(
        (
            "Privacy-utility frontier: graceful-degradation gate clears"
            if graceful
            else "Privacy-utility frontier: full-grid ladder is non-monotone"
        ),
        loc="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
        fontweight="bold",
        pad=10,
    )
    ax.text(
        0.985,
        0.04,
        "\n".join(
            [
                (
                    "graceful gate: "
                    f"{'clear' if graceful else 'fail'}; "
                    f"utility monotone={utility_monotone}; risk monotone={risk_monotone}"
                ),
                f"max shuffle-floor SD={_fmt_stat(summary.get('max_shuffle_floor_sd'), 6)} bits",
                (
                    f"utility floor={_fmt_stat(min_collection.get('utility_floor_bits'), 2)} bits; "
                    f"risk ceiling={_fmt_stat(min_collection.get('reid_risk_ceiling'), 2)}"
                ),
                (
                    "min-collection op: "
                    + (
                        str(min_collection.get("regime", "found"))
                        if min_collection.get("found")
                        else (
                            "none; min useful risk="
                            f"{_fmt_stat(min_collection.get('min_reid_among_useful_grids'))}"
                        )
                    )
                ),
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=ANNOTATION_FONT_SIZE,
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88, "pad": 4.0},
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "location minimization (fine -> coarse grid)", fontsize=SMALL_LABEL_FONT_SIZE
    )
    cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _privacy_utility_repeats(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    rows = summary.get("rows") if isinstance(summary.get("rows"), list) else []
    completed = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("status") == "completed"
    ]
    unavailable = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("status") != "completed"
    ]
    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(13.2, 6.4),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    if not completed:
        for ax in (ax0, ax1):
            ax.text(
                0.5,
                0.5,
                "repeat privacy-utility audit unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    x = np.arange(len(completed))
    labels = [
        "canonical"
        if str(row.get("realization_id")) == "canonical_fixture"
        else str(row.get("realization_id", "?")).replace(
            "variance_realization_", "var "
        )
        for row in completed
    ]
    finest = [float(row.get("finest_excess_utility_bits", 0.0)) for row in completed]
    coarsest = [
        float(row.get("coarsest_excess_utility_bits", 0.0)) for row in completed
    ]
    ax0.plot(
        x,
        finest,
        marker="o",
        color=palette.get("blue", "#0072B2"),
        lw=1.8,
        label="finest utility",
    )
    ax0.plot(
        x,
        coarsest,
        marker="s",
        color=palette.get("orange", "#E69F00"),
        lw=1.8,
        label="coarsest utility",
    )
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=35, ha="right", fontsize=TICK_FONT_SIZE)
    ax0.set_ylabel("shuffle-corrected excess utility (bits)", fontsize=METRIC_FONT_SIZE)
    ax0.set_title(
        "Repeated privacy-utility spread",
        loc="left",
        fontsize=PANEL_HEADING_FONT_SIZE,
        fontweight="bold",
    )
    ax0.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    ax0.grid(axis="y", alpha=0.25)

    statuses = ["completed", "unavailable"]
    counts = [len(completed), len(unavailable)]
    colors = [palette.get("green", "#009E73"), palette.get("gray", "#595959")]
    ax1.bar(statuses, counts, color=colors, edgecolor="#111111", linewidth=0.5)
    for idx, count in enumerate(counts):
        ax1.text(
            idx,
            count + 0.05,
            str(count),
            ha="center",
            va="bottom",
            fontsize=SMALL_LABEL_FONT_SIZE,
            fontweight="bold",
        )
    ax1.set_ylim(0, max(counts + [1]) * 1.28)
    ax1.set_ylabel("realization rows", fontsize=METRIC_FONT_SIZE)
    unavailable_count = int(summary.get("unavailable_count", len(unavailable)) or 0)
    graceful_repeats = int(summary.get("n_graceful_degradation", 0) or 0)
    completed_count = int(summary.get("completed_count", len(completed)) or 0)
    if unavailable_count > 0:
        repeat_gate_title = "Unavailable repeats block stability wording"
    elif not summary.get("can_claim_graceful_repeated_tradeoff"):
        repeat_gate_title = "Graceful-degradation repeats block stability wording"
    else:
        repeat_gate_title = "Repeated gate clears conditional stability wording"
    ax1.set_title(
        repeat_gate_title,
        loc="left",
        fontsize=PANEL_HEADING_FONT_SIZE,
        fontweight="bold",
    )
    ax1.text(
        0.98,
        0.95,
        "\n".join(
            [
                f"completed={completed_count}",
                f"unavailable={unavailable_count}",
                f"graceful repeats={graceful_repeats}",
                (
                    "repeat claim: "
                    f"{'allowed' if summary.get('can_claim_graceful_repeated_tradeoff') else 'blocked'}"
                ),
            ]
        ),
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88, "pad": 4.0},
    )
    # Derive the suptitle from the same gate state as the right-panel title so
    # the two can never contradict (F6-23). When the conditional gate clears,
    # the repeats are deterministic reruns, not independent-sample stability, so
    # the cleared wording says exactly that rather than claiming "stability".
    if unavailable_count > 0:
        suptitle = "Repeated privacy-utility audit is partial: unavailable repeats block stability wording"
    elif not summary.get("can_claim_graceful_repeated_tradeoff"):
        suptitle = "Repeated privacy-utility audit is partial: graceful-degradation repeats block stability wording"
    else:
        suptitle = (
            "Repeated privacy-utility audit: deterministic reruns clear the conditional gate "
            "(determinism, not independent-sample stability)"
        )
    fig.suptitle(
        suptitle,
        x=0.01,
        ha="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def _synthetic_task_utility_privacy(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    rows = summary.get("rows") if isinstance(summary.get("rows"), list) else []
    rows = [row for row in rows if isinstance(row, dict)]
    # Two panels sharing the regime axis: utility (bits) and re-identification
    # risk (event-uniqueness fraction) live on genuinely different scales, so a
    # single shared x-axis compressed one of them. Splitting them lets each use
    # its own scale and puts the utility_floor and risk_ceiling on the axis they
    # actually gate.
    fig, (ax_u, ax_r) = plt.subplots(
        1, 2, sharey=True, figsize=(12.8, 7.0), gridspec_kw={"width_ratios": [3, 1]}
    )
    if not rows:
        ax_u.text(
            0.5,
            0.5,
            "task-utility privacy gate unavailable",
            ha="center",
            va="center",
            transform=ax_u.transAxes,
        )
        for a in (ax_u, ax_r):
            a.set_xticks([])
            a.set_yticks([])
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    ordered = sorted(
        rows, key=lambda row: int(row.get("spatial_bins", 0)), reverse=True
    )
    y = np.arange(len(ordered))
    utility = [float(row.get("predictive_utility_bits", 0.0)) for row in ordered]
    risk = [float(row.get("reid_event_uniqueness", 0.0)) for row in ordered]
    labels = [str(row.get("regime", "?")).replace("grid_", "") for row in ordered]
    utility_floor = float(summary.get("utility_floor_bits", 0.10))
    risk_ceiling = float(summary.get("reid_risk_ceiling", 0.20))
    blue = palette.get("blue", "#0072B2")
    orange = palette.get("orange", "#E69F00")

    # Left panel: predictive utility on its own scale.
    ax_u.barh(y, utility, height=0.62, color=blue, edgecolor="#111111")
    ax_u.axvline(
        utility_floor,
        color=blue,
        linestyle="--",
        lw=1.2,
        label=f"utility floor={utility_floor:.2g} bits",
    )
    ax_u.set_xlim(0, max(utility + [utility_floor]) * 1.25)
    ax_u.set_xlabel("predictive utility (bits)", fontsize=METRIC_FONT_SIZE)
    ax_u.set_yticks(y)
    ax_u.set_yticklabels(labels, fontsize=TICK_FONT_SIZE)
    ax_u.tick_params(axis="y", pad=8)
    for tick_label in ax_u.get_yticklabels():
        tick_label.set_ha("right")
        tick_label.set_multialignment("right")
    ax_u.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE, frameon=True)
    ax_u.grid(axis="x", alpha=0.24)

    # Right panel: re-identification risk on the (0,1] uniqueness scale with its ceiling.
    ax_r.barh(y, risk, height=0.62, color=orange, edgecolor="#111111")
    ax_r.axvline(
        risk_ceiling,
        color=orange,
        linestyle=":",
        lw=1.2,
        label=f"risk ceiling={risk_ceiling:.2g}",
    )
    ax_r.set_xlim(0, max(risk + [risk_ceiling]) * 1.25)
    ax_r.set_xlabel(
        f"re-id event uniqueness\n(fraction, ceiling={risk_ceiling:.2g})",
        fontsize=METRIC_FONT_SIZE,
    )
    ax_r.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE, frameon=True)
    ax_r.grid(axis="x", alpha=0.24)

    allowed_rows = [row for row in ordered if row.get("release_utility_claim_allowed")]
    # Per-row joint-gate verdict, in the right margin beside each risk bar (the
    # figure's core content); white backing + high zorder keep it un-occluded.
    label_x = max(risk + [risk_ceiling]) * 1.30
    for yi, row in zip(y, ordered):
        row_allowed = bool(row.get("release_utility_claim_allowed"))
        ax_r.text(
            label_x,
            yi,
            "joint gate clear" if row_allowed else "blocked (risk > ceiling)",
            va="center",
            ha="left",
            fontsize=ANNOTATION_FONT_SIZE,
            color=palette.get("green", "#009E73")
            if row_allowed
            else palette.get("red", "#CC3311"),
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.0},
            zorder=6,
            clip_on=False,
        )

    any_allowed = bool(summary.get("any_release_utility_claim_allowed")) or bool(
        allowed_rows
    )
    joint_count = int(
        summary.get("joint_operating_point_count", len(allowed_rows)) or 0
    )
    title_verdict = (
        f"release-utility wording conditional: {joint_count} joint operating point"
        f"{'' if joint_count == 1 else 's'} found"
        if any_allowed
        else "release-utility wording blocked on every regime"
    )
    fig.suptitle(
        f"Synthetic task-utility/privacy gate: {title_verdict}",
        x=0.01,
        ha="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
        fontweight="bold",
    )
    # Figure-level provenance banner below both panels. Anchoring this to the
    # left axes made it collide with the right panel's two-line x label.
    fig.text(
        0.50,
        0.025,
        "   ".join(
            [
                f"status={summary.get('status', 'unavailable')}",
                f"utility floor={_fmt_stat(summary.get('utility_floor_bits'), 2)} bits",
                f"risk ceiling={_fmt_stat(summary.get('reid_risk_ceiling'), 2)}",
                f"best minimized regime={summary.get('best_minimized_operating_regime', 'none')}",
                "joint gate="
                + (
                    "cleared on selected rows"
                    if allowed_rows
                    else "blocked on every regime"
                ),
            ]
        ),
        ha="center",
        va="bottom",
        fontsize=ANNOTATION_FONT_SIZE,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88, "pad": 4.0},
    )
    # Reserve an explicit right margin for the per-row gate labels and a bottom
    # margin for the provenance box. ``tight_layout`` cannot infer the extent of
    # artists intentionally drawn outside ``ax_r`` and emitted a false-clean
    # layout warning even when the saved image happened to remain legible.
    fig.subplots_adjust(
        left=0.13,
        right=0.75,
        bottom=0.25,
        top=0.88,
        wspace=0.26,
    )
    fig.savefig(path)
    plt.close(fig)


def _aif_deepening_candidates(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    rows = summary.get("rows") if isinstance(summary.get("rows"), list) else []
    rows = [row for row in rows if isinstance(row, dict)]
    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    if not rows:
        ax.text(
            0.5,
            0.5,
            "AIF deepening audit unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    rows = sorted(
        rows,
        key=lambda row: str(row.get("protocol_status"))
        != "scored_predictive_candidate",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for x, label in (
        (0.035, "SURFACE"),
        (0.260, "ADMISSION"),
        (0.485, "EVIDENCE"),
        (0.745, "NEXT GATE"),
    ):
        ax.text(
            x,
            0.91,
            label,
            transform=ax.transAxes,
            fontsize=ANNOTATION_FONT_SIZE,
            color=palette.get("gray", "#595959"),
            fontweight="bold",
            va="bottom",
        )
    for idx, row in enumerate(rows):
        y_top = 0.86 - idx * 0.245
        protocol = str(row.get("protocol_status", "unknown"))
        scored = protocol == "scored_predictive_candidate"
        accent = (
            palette.get("blue", "#0072B2")
            if scored
            else palette.get("gray", "#777777")
        )
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.01, y_top - 0.19),
                0.98,
                0.19,
                transform=ax.transAxes,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                facecolor="#F8F8F6",
                edgecolor="#CCCCCC",
                linewidth=0.9,
            )
        )
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.012, y_top - 0.188),
                0.012,
                0.186,
                transform=ax.transAxes,
                boxstyle="round,pad=0.001,rounding_size=0.005",
                facecolor=accent,
                edgecolor=accent,
            )
        )
        candidate_id = str(row.get("candidate_id", "?"))
        candidate = {
            "online_adaptation": "online transition\nadaptation",
            "factorial_matched_prior_tensor_provenance": "factorial tensor\nprovenance",
            "multiscale_soft_belief_backoff": "multiscale belief\nbackoff",
        }.get(candidate_id, _clean_label(candidate_id, width=18))
        ax.text(
            0.035,
            y_top - 0.09,
            candidate,
            transform=ax.transAxes,
            fontsize=SMALL_LABEL_FONT_SIZE,
            fontweight="bold",
            va="center",
        )
        admission = (
            "SCORED\npredictive candidate"
            if scored
            else "AUDIT ONLY\nno predictive score"
        )
        ax.text(
            0.260,
            y_top - 0.09,
            admission,
            transform=ax.transAxes,
            fontsize=ANNOTATION_FONT_SIZE,
            color=accent,
            fontweight="bold",
            va="center",
        )
        if row.get("candidate_id") == "online_adaptation":
            total = int(row.get("realization_count", 0) or 0)
            evidence = (
                f"gain vs fixed: {_fmt_stat(row.get('adaptive_improvement_nats'), 3)} nats\n"
                f"CI clears fixed: {row.get('beats_fixed_ci_count', 0)}/{total}\n"
                f"CI clears base rate: {row.get('beats_fair_ci_count', 0)}/{total}"
            )
        elif row.get("candidate_id") == "multiscale_soft_belief_backoff":
            evidence = "local regime gate: no\nglobal SOTA gate: no"
        else:
            evidence = "provenance: present\nmatched prior: passed"
        ax.text(
            0.485,
            y_top - 0.09,
            evidence,
            transform=ax.transAxes,
            fontsize=ANNOTATION_FONT_SIZE,
            va="center",
        )
        next_gate = {
            "online_adaptation": "Broaden switching regimes;\nclose the base-rate gap",
            "factorial_matched_prior_tensor_provenance": (
                "Implement a locked-split\npredictive Candidate"
            ),
            "multiscale_soft_belief_backoff": (
                "Enter the Candidate roster;\nbeat the strongest\nfair baseline"
            ),
        }.get(
            candidate_id,
            _clean_label(
                row.get("next_required_step", "protocol candidate needed"), width=28
            ),
        )
        ax.text(
            0.745,
            y_top - 0.09,
            next_gate,
            transform=ax.transAxes,
            fontsize=ANNOTATION_FONT_SIZE,
            va="center",
        )
    scored_count = int(summary.get("scored_protocol_candidate_count", 0) or 0)
    ax.set_title(
        (
            "AIF deepening: online adaptation now scores; promotion still requires a fair-baseline win"
            if scored_count
            else "AIF deepening surfaces are audit-only until they score as locked predictive candidates"
        ),
        loc="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.text(
        0.50,
        0.030,
        " | ".join(
            [
                f"status: {str(summary.get('status', 'unavailable')).replace('_', ' ')}",
                f"scored candidates: {summary.get('scored_protocol_candidate_count', 0)}",
                f"promotion gate: {'cleared' if summary.get('predictive_gate_clears') else 'blocked'}",
            ]
        ),
        ha="center",
        va="bottom",
        fontsize=ANNOTATION_FONT_SIZE,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88, "pad": 4.0},
    )
    fig.savefig(path)
    plt.close(fig)


def _model_comparison_under_noise(
    summary: dict[str, object], path: Path, palette: dict[str, str]
) -> None:
    """Where active inference leads: per-method next-cell loss under the noisy lane, ranked."""

    def _model_group(row: dict[str, object]) -> str:
        group = str(row.get("group", ""))
        return "belief_marginalizing" if group == "belief_marginalising" else group

    rows = summary.get("leaderboard") or []
    fig, ax = plt.subplots(figsize=(13.4, 10.0))
    if not isinstance(rows, list) or not rows:
        ax.text(
            0.5,
            0.5,
            "model comparison unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    # best loss at the top: reverse so rank 1 is highest on the y-axis
    ordered = sorted(rows, key=lambda r: float(r.get("mean_loss", 9.0)), reverse=True)
    group_color = {
        "active_inference": palette.get("green", "#2e8b57"),
        "belief_marginalizing": palette.get("blue", "#2c6fbb"),
        "point_estimate": palette.get("orange", "#e08a1e"),
        "raw_observation_or_frequency": palette.get("gray", "#888888"),
    }
    group_label = {
        "active_inference": "active inference (marginalizes belief)",
        "belief_marginalizing": "belief-marginalizing (HMM, particle)",
        "point_estimate": "point estimate (MAP)",
        "raw_observation_or_frequency": "raw observation / frequency",
    }
    y = np.arange(len(ordered))
    losses = [float(r.get("mean_loss", 0.0)) for r in ordered]
    colors = [
        group_color.get(_model_group(r), palette.get("gray", "#888888"))
        for r in ordered
    ]
    labels = [_model_axis_label(r.get("candidate", "?"), width=26) for r in ordered]
    ax.barh(y, losses, color=colors, edgecolor="#111111", linewidth=0.5, zorder=2)
    chance_loss = summary.get("chance_loss_nats")
    if isinstance(chance_loss, int | float) and np.isfinite(float(chance_loss)):
        chance_loss = float(chance_loss)
        chance_cells = int(round(float(np.exp(chance_loss))))
        ax.axvline(
            chance_loss,
            color=palette.get("gray", "#888888"),
            linestyle="--",
            linewidth=1.1,
            zorder=1,
        )
        ax.annotate(
            f"uniform-chance next-cell loss\nln {chance_cells} = {chance_loss:.2f} nats",
            xy=(chance_loss, 0.12),
            xycoords=("data", "axes fraction"),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=SMALL_LABEL_FONT_SIZE,
            color=palette.get("gray", "#595959"),
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
    belief_ys = [
        int(yi)
        for yi, row in zip(y, ordered)
        if _model_group(row) in {"active_inference", "belief_marginalizing"}
    ]
    if belief_ys:
        y_low = min(belief_ys) - 0.45
        y_high = max(belief_ys) + 0.45
        ax.axhspan(
            y_low, y_high, color=palette.get("green", "#009E73"), alpha=0.07, zorder=0
        )
        # RedTeam fix #4: anchor clear of both the belief-family bar tips and
        # their per-row "top-1=" labels (drawn just past each bar tip below,
        # measured at ~belief_bar_max + max(losses)*0.14 wide) and clear of the
        # top-right gap-summary stats box, which occupies the upper portion of
        # this same row span. Biasing the y-anchor toward the lower rows of
        # the span keeps it under the stats box while the x push clears the
        # top-1 label text.
        belief_bar_max = max(losses[yi] for yi in belief_ys)
        label_x = belief_bar_max + max(losses) * 0.22
        label_y = y_low + (y_high - y_low) * 0.28
        ax.text(
            label_x,
            label_y,
            f"belief family\n gap to best non-belief: {_fmt_stat(summary.get('belief_vs_nonbelief_gap'))} nats",
            ha="left",
            va="center",
            fontsize=SMALL_LABEL_FONT_SIZE,
            fontweight="bold",
            color="#222222",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
            zorder=4,
        )
    # CI whiskers where available
    for yi, r in zip(y, ordered):
        lo, hi = r.get("loss_ci_low"), r.get("loss_ci_high")
        if lo is not None and hi is not None:
            ax.plot(
                [float(lo), float(hi)],
                [yi, yi],
                color="#111111",
                lw=1.0,
                alpha=0.7,
                zorder=3,
            )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=TICK_FONT_SIZE)
    ax.set_xlabel(
        "mean next-cell predictive loss, nats (lower = better); whiskers: bootstrap 95% CI",
        fontsize=METRIC_FONT_SIZE,
    )
    x_max = max(
        max(losses) if losses else 1.0,
        float(chance_loss) if isinstance(chance_loss, float) else 0.0,
    )
    ax.set_xlim(0.0, x_max * 1.10)
    best_nonbelief = summary.get("best_nonbelief_loss")
    if isinstance(best_nonbelief, int | float) and np.isfinite(float(best_nonbelief)):
        ax.axvline(
            float(best_nonbelief),
            color=palette.get("gray", "#888888"),
            linestyle=":",
            linewidth=1.2,
            zorder=1,
        )
        # Label the boundary VERTICALLY, riding alongside its own dotted line,
        # instead of a horizontal label that drifted across the bottom bars (F6-26).
        # Anchored a data-space margin to the right of the line itself (rather
        # than a fixed few-point offset) so the rotated text clears the CI
        # whisker caps of the bars nearest the boundary, which can extend past
        # the line on either side (LOW-26: label crossed the "MLP next-cell"
        # and "hard filter" whisker ticks when anchored right on the line).
        boundary_label_x = float(best_nonbelief) + max(losses) * 0.07
        ax.annotate(
            "best non-belief boundary",
            xy=(boundary_label_x, 0.5),
            xycoords=("data", "axes fraction"),
            xytext=(3, 0),
            textcoords="offset points",
            rotation=90,
            ha="left",
            va="center",
            fontsize=SMALL_LABEL_FONT_SIZE,
            color=palette.get("gray", "#595959"),
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
    eps = summary.get("eps")
    aif_rank = summary.get("aif_rank")
    n_methods = summary.get("n_methods", len(ordered))
    ax.set_title(
        "Per-method loss under noisy emission: belief-marginalizing methods lead\n"
        f"(eps={eps}; active inference rank {aif_rank}/{n_methods}; paired tie vs strongest non-AIF; "
        "not oracle-beating, not global SOTA)",
        loc="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
        fontweight="bold",
        pad=10,
    )
    # mark the AIF candidate bar
    for idx, (yi, r) in enumerate(zip(y, ordered)):
        if str(r.get("group")) == "active_inference" and int(r.get("rank", 0)) == 1:
            # Sit the rank marker ABOVE the bar so it no longer overprints the
            # "top-1=" label drawn just below the same bar tip (F6-26).
            ax.text(
                float(r["mean_loss"]),
                yi + 0.30,
                "  #1 point estimate",
                va="bottom",
                ha="left",
                fontsize=SMALL_LABEL_FONT_SIZE,
                color=group_color["active_inference"],
                fontweight="bold",
            )
        rank = (
            int(r.get("rank", idx + 1)) if str(r.get("rank", "")).isdigit() else idx + 1
        )
        if rank <= 4 and isinstance(r.get("top1_accuracy"), int | float):
            bar_tip = float(r.get("mean_loss", 0.0))
            top1_label_x = bar_tip + max(losses) * 0.012
            top1_ha = "left"
            top1_color = "#333333"
            # Guard against straddling the "best non-belief boundary" dotted
            # line and its rotated label: the top-4 rows sit close enough to
            # best_nonbelief that the outside-the-bar placement lets the
            # "top-1=" text overrun the line (F6-27). The corridor between
            # the line and the "belief family" gap-box (~0.37 nats wide) is
            # narrower than the rendered label (~0.50 nats), so no outside
            # position clears both — draw the label inside the bar instead
            # (right-aligned on the bar tip, white on the bar's own color),
            # which always has room since the bar spans 0..bar_tip.
            if (
                isinstance(best_nonbelief, int | float)
                and np.isfinite(float(best_nonbelief))
                and top1_label_x > float(best_nonbelief) - max(losses) * 0.20
            ):
                top1_label_x = bar_tip - max(losses) * 0.01
                top1_ha = "right"
                top1_color = "white"
            ax.text(
                top1_label_x,
                yi - 0.28,
                f"top-1={_fmt_stat(r.get('top1_accuracy'))}",
                va="center",
                ha=top1_ha,
                fontsize=ANNOTATION_FONT_SIZE,
                color=top1_color,
                fontweight="bold" if top1_ha == "right" else "normal",
            )
    ax.text(
        0.985,
        0.98,
        "\n".join(
            [
                f"belief vs non-belief gap={_fmt_stat(summary.get('belief_vs_nonbelief_gap'))} nats",
                f"AIF vs strongest non-AIF gap={_fmt_stat(summary.get('aif_vs_strongest_non_aif_gap'))} nats",
                (
                    "AIF lead CI-separated: "
                    f"{'yes' if summary.get('aif_beats_strongest_non_aif_ci_separated') else 'no'}"
                ),
                (
                    "top-belief marginal CIs disjoint: "
                    f"{'yes' if summary.get('aif_marginal_ci_disjoint_from_next_belief') else 'no'}"
                ),
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.86, "pad": 4.0},
    )
    seen: list[str] = []
    handles = []
    for r in ordered:
        g = _model_group(r)
        if g not in seen:
            seen.append(g)
            handles.append(
                patches.Patch(
                    facecolor=group_color.get(g, "#888888"),
                    edgecolor="#111111",
                    label=group_label.get(g, g),
                )
            )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def _dataflow(path: Path, palette: dict[str, str]) -> None:
    theme = editorial_palette(palette)
    fig, axis = plt.subplots(figsize=(15.6, 6.2), facecolor=theme["paper"])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    lanes = [
        (
            "input boundary",
            [
                ("pinned generator\nor fixture CSV", "no vendored internals"),
                ("observations.csv", "ID | day | x | y"),
            ],
            theme["teal"],
        ),
        (
            "evidence model",
            [
                ("discrete_features.csv", "cell | phase | speed | burst"),
                ("tensor_summary.json", "A/B/C/D normalization"),
            ],
            theme["blue"],
        ),
        (
            "inference lanes",
            [
                ("inference_trace.csv", "Bayes posterior"),
                ("pymdp_trace.csv", "VFE | EFE | policy"),
            ],
            theme["green"],
        ),
        (
            "claim gates",
            [
                ("null artifacts", "calibration | persistence"),
                ("risk artifacts", "minimization | integrity"),
            ],
            theme["amber"],
        ),
        (
            "reader handoff",
            [
                ("figure_registry.json", "captions | sources | alt text"),
                ("variables.json", "token provenance"),
            ],
            theme["violet"],
        ),
    ]
    axis.text(
        0.035,
        0.93,
        "Artifact contract for the synthetic active-inference study",
        ha="left",
        va="top",
        fontsize=CANVAS_TITLE_FONT_SIZE,
        fontweight="bold",
        color=theme["ink"],
    )
    axis.text(
        0.035,
        0.865,
        "Claims move left to right only through generated artifacts; prose receives captions and numbers after provenance checks.",
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        color=theme["muted"],
    )
    lane_width = 0.18
    for idx, (lane, boxes, color) in enumerate(lanes):
        x = 0.035 + idx * 0.19
        axis.add_patch(
            patches.FancyBboxPatch(
                (x, 0.145),
                lane_width,
                0.665,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                facecolor=theme["panel"],
                edgecolor=color,
                linewidth=1.35,
                zorder=1,
            )
        )
        axis.text(
            x + 0.014,
            0.755,
            lane,
            ha="left",
            va="center",
            fontsize=SMALL_LABEL_FONT_SIZE,
            fontweight="bold",
            color=color,
            zorder=2,
        )
        for box_idx, (label, detail) in enumerate(boxes):
            y = 0.585 - box_idx * 0.25
            axis.add_patch(
                patches.FancyBboxPatch(
                    (x + 0.014, y - 0.095),
                    lane_width - 0.028,
                    0.18,
                    boxstyle="round,pad=0.008,rounding_size=0.008",
                    facecolor="#FBFAF5",
                    edgecolor=theme["grid"],
                    linewidth=0.85,
                    zorder=2,
                )
            )
            axis.text(
                x + 0.026,
                y + 0.065,
                _clean_label(label, width=16),
                ha="left",
                va="top",
                fontsize=SMALL_LABEL_FONT_SIZE,
                color=theme["ink"],
                fontweight="bold",
                linespacing=0.88,
                zorder=3,
            )
            axis.text(
                x + 0.026,
                y - 0.015,
                _clean_label(detail, width=20),
                ha="left",
                va="top",
                fontsize=ANNOTATION_FONT_SIZE,
                color=theme["muted"],
                linespacing=0.9,
                zorder=3,
            )
        if idx < len(lanes) - 1:
            axis.annotate(
                "",
                xy=(x + lane_width + 0.026, 0.48),
                xytext=(x + lane_width + 0.002, 0.48),
                arrowprops={"arrowstyle": "->", "color": theme["muted"], "lw": 1.1},
                zorder=4,
            )
    axis.text(
        0.035,
        0.085,
        "The diagram is a contract diagram, not a performance result: generated figures and manuscript tokens must trace back\n"
        "to CSV/JSON artifacts or source modules.",
        ha="left",
        va="center",
        fontsize=ANNOTATION_FONT_SIZE,
        color=theme["muted"],
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _registry_entry(
    key: str,
    filename: str,
    run_name: str,
    context: dict[str, object],
    privacy_utility_summary: dict[str, object],
    fractal_dimension_summary: dict[str, object],
) -> FigureRegistryEntry:
    n_obs = context["n_observations"]
    n_features = context["n_features"]
    first = context["first_observations"]
    smoothing = context["smoothing"]
    speed_thresholds = context["speed_thresholds"]
    control_count = context["control_count"]
    privacy_utility_payload: dict[str, object] = (
        privacy_utility_summary if isinstance(privacy_utility_summary, dict) else {}
    )
    phase_payload = context.get("regime_switch_phase_summary", {})
    phase_payload = phase_payload if isinstance(phase_payload, dict) else {}
    phase_rows = phase_payload.get("segments", [])
    phase_rows = phase_rows if isinstance(phase_rows, list) else []
    resolved_phase_names = [
        str(row.get("segment"))
        for row in phase_rows
        if isinstance(row, dict)
        and isinstance(row.get("clustered_ci_low"), int | float)
        and isinstance(row.get("clustered_ci_high"), int | float)
        and float(row["clustered_ci_low"]) < 0.0 < float(row["clustered_ci_high"])
    ]
    negative_phase_names = [
        str(row.get("segment"))
        for row in phase_rows
        if isinstance(row, dict)
        and isinstance(row.get("clustered_ci_high"), int | float)
        and float(row["clustered_ci_high"]) < 0.0
    ]
    phase_ratio = phase_payload.get("worst_to_best_magnitude_ratio")
    phase_ratio_text = (
        f"{float(phase_ratio):.2f}"
        if isinstance(phase_ratio, int | float) and np.isfinite(float(phase_ratio))
        else "n/a"
    )
    phase_caption = (
        "Source regime_switch_phase_diagnostic.json partitions the AIF-versus-strongest-fair per-step losses from "
        "latent_regime_switch_035, the one synthetic regime with named commute, explore, and return_stabilization "
        "segments. Bars report "
        "AIF advantage with traveller-clustered bootstrap intervals. "
        f"CI-separated negative segments: {', '.join(negative_phase_names) or 'none'}; "
        f"zero-crossing unresolved segments: {', '.join(resolved_phase_names) or 'none'}; "
        f"uniform disadvantage: {phase_payload.get('uniform_disadvantage', 'unavailable')}; "
        f"worst-to-best magnitude ratio: {phase_ratio_text}."
    )
    # Derive the fractal-dimension-sweep support claim from the artifact's own
    # control-active flag: when the external stochastic simulator is unavailable
    # the sweep replays one fixture at every dimension (zero spread by
    # construction), so it is an invariance check, NOT evidence of generator-knob
    # sensitivity. Claiming sensitivity in that state would be a caption overclaim.
    fractal_payload: dict[str, object] = (
        fractal_dimension_summary if isinstance(fractal_dimension_summary, dict) else {}
    )
    fractal_control_active = fractal_payload.get("dimension_control_active") is True
    fractal_support = (
        "generator-knob sensitivity over synthetic support texture"
        if fractal_control_active
        else (
            "a fixture-replay invariance check over the nominal dimension knob "
            "(the external generator was unavailable, so metrics are constant by construction)"
        )
    )

    privacy_graceful = (
        privacy_utility_payload.get("graceful_degradation_tradeoff") is True
    )
    privacy_min_collection = privacy_utility_payload.get(
        "min_collection_operating_point", {}
    )
    privacy_min_collection_found = (
        isinstance(privacy_min_collection, dict)
        and privacy_min_collection.get("found") is True
    )
    privacy_min_collection_text = (
        f"minimum-collection operating point {privacy_min_collection.get('regime', 'found')}"
        if privacy_min_collection_found and isinstance(privacy_min_collection, dict)
        else "no minimum-collection operating point"
    )
    privacy_utility_caption = (
        "Source privacy_utility_frontier.json pairs surviving next-cell utility with per-event uniqueness risk across "
        "location-minimization regimes. Utility is measured as bits beyond chance and corrected for the shuffled "
        "static-occupancy floor, making grids comparable. The current artifact reports a monotone graceful-degradation "
        f"gate and {privacy_min_collection_text}."
        if privacy_graceful
        else (
            "Source privacy_utility_frontier.json pairs surviving next-cell utility with per-event uniqueness risk across "
            "location-minimization regimes. Utility is measured as bits beyond chance and corrected for the shuffled "
            "static-occupancy floor, making grids comparable. This is a non-monotone minimization audit because the "
            f"7x7/8x8 pair rises; it yields {privacy_min_collection_text} and no graceful-degradation claim."
        )
    )
    privacy_utility_support = (
        "a synthetic governance audit showing surviving predictive utility, re-identification risk, "
        "a source-cleared monotone graceful-degradation gate, and the current minimum-collection operating point "
        "as location is minimized"
        if privacy_graceful
        else (
            "a synthetic governance audit showing surviving predictive utility, re-identification risk, "
            "non-monotone full-grid behavior, and the current no-operating-point null as location is minimized"
        )
    )
    captions = {
        "evidence_atlas": (
            "Sources observations.csv, discrete_features.csv, metrics.json, spatial_resolution_frontier.json, "
            "model_gate_matrix.json, forward_gates_summary.json, efe_location_frontier.json, linkage_risk_frontier.json, "
            "and poisoning_robustness.json feed one orientation atlas. It traces sparse synthetic reports through "
            "categorical evidence, belief updating, model gates, minimization, and integrity checks. The chip strip places "
            "the partial-observability positive beside the Holm multiplicity result, so the reader sees the main finding "
            "and its claim discipline together."
        ),
        "waypoint_paths": (
            f"Source observations.csv is shown as {n_obs} synthetic sporadic waypoint reports for "
            f"{context['n_travellers']} travellers. Lines connect reports only in report order; they are a visual index, "
            "not reconstructed continuous tracks. Color encodes report time, the open triangle marks each first report, "
            "and the star marks the final report. The visible clustering and gaps define the sparse-observation regime "
            "that the benchmark scores, not empirical mobility."
        ),
        "spatial_bins": (
            f"Source discrete_features.csv maps all n={n_features} reports from continuous coordinates into the model's "
            "latent spatial cells. Color and printed cell labels show the configured grid that the location modality "
            "actually observes: reports sharing a cell become equivalent evidence even when their coordinates differ. "
            "This figure fixes the analysis resolution before inference and is a synthetic discretization audit, not a "
            "map of real places."
        ),
        "spatial_resolution_frontier": (
            "Sources spatial_resolution_frontier.json and spatial_resolution_larger_sample_confirmation.json compare "
            "candidate square grids on occupancy, transition sparsity, full-location predictive loss, no-location recovery, "
            "real-pymdp agreement, and deterministic sample-size sensitivity. The selected default is the canonical "
            "guardrail survivor. When the larger lane has no strict-feasible grid, the marked bin is explicitly a "
            "best-measured fallback. The two recovery rows keep observed-location fit separate from metadata-only recovery."
        ),
        "reporting_bursts": (
            "Source discrete_features.csv exposes the reporting process rather than treating missingness as background "
            f"noise. The left panel plots positive inter-report intervals against the configured burst cutoff; the right "
            f"panel counts burst labels after the {first} first observations are assigned zero-interval non-burst "
            "sentinels. The heavy tail and short-interval cluster are the synthetic SporadicReporter texture that later "
            "missingness gates must explain."
        ),
        "speed_categories": (
            f"Source discrete_features.csv converts consecutive reports into speed evidence for n={n_features} synthetic "
            f"reports. Bars summarize category counts; the trace shows each derived speed against thresholds "
            f"({speed_thresholds} km/day), with first observations fixed at zero by convention. This is one non-location "
            "modality used by the deep-latent configuration, and it remains derived evidence rather than direct ground "
            "truth about unobserved movement."
        ),
        "transition_heatmap": (
            "Source discrete_features.csv estimates the empirical latent-cell transition surface before tensor smoothing, "
            "excluding traveller-boundary (first-observation) pairs exactly as the model's B-tensor estimator does. "
            "Each column is P(next cell | previous cell), with raw counts printed over the diagonal and the largest "
            "off-diagonal cells; the diagonal makes "
            "dwell-and-return structure visible, while off-diagonal mass carries movement evidence. This panel is the "
            f"normalization and sparsity audit for downstream synthetic B tensors before smoothing={smoothing} is applied."
        ),
        "likelihood_panels": (
            "Source discrete_features.csv builds column-normalized observation likelihoods P(observation | latent cell) "
            "for location, time, speed, and reporting-burst modalities. The near-diagonal location panel is expected by "
            "construction; the time, speed, and burst panels are the evidence channels the deep-latent configuration uses "
            "when location is withheld. The panels are the audit point for the conditional-independence approximation "
            f"before Model A smoothing={smoothing} is applied, not empirical sensor validation."
        ),
        "posterior_entropy": (
            f"Source inference_trace.csv records the transparent reference filter over n={context['n_inference']} reports. "
            "The top panel shows one-step predictive loss, the negative log prior-predictive probability assigned to the "
            "observed report, against the uniform-chance joint-loss floor; the bottom panel shows posterior entropy after updating. Spikes identify synthetic reports "
            "that surprise the model, while low sustained entropy marks confident cell tracking rather than proof of real "
            "trajectory reconstruction."
        ),
        "policy_proxy": (
            f"Source policy_trace.csv exposes the proxy policy diagnostic for the {control_count} configured transition "
            "control(s). Total score, score spread, entropy and preference components, and the softmax policy posterior "
            "show whether candidate controls separate under the current preferences; identical proxy scores are drawn as one "
            "line when the loaded artifact makes the tie exact. Near-uniform posteriors are reported "
            "as a synthetic observational-equivalence finding, not as a controller failure or operational recommendation."
        ),
        "candidate_scores": (
            f"Source candidate_scores.json ranks the n={context['n_candidates']} declared next-run simulator candidates by "
            "the measured objective scaled by relative reporting density. Rank labels and scores make the closed-loop "
            "choice recomputable from artifacts. This is an audit trail for candidate selection inside the synthetic "
            "fixture lane, not an external simulator-performance benchmark."
        ),
        "negative_controls": (
            f"Source negative_controls_summary.json reports {context.get('negative_controls_passed', 0)}/"
            f"{context.get('negative_controls_total', 0)} directional expectations held. Loss changes under shuffled "
            "report times and shuffled identities test destroyed structure, while occupancy-collapse probes test "
            "degenerate bounds and misspecified speed thresholds. Because collapsed modalities can be easier to predict, "
            "the oracle is control-specific rather than a universal loss direction. Fixture-lane software diagnostics, "
            "not empirical mobility evidence."
        ),
        "pymdp_agreement": (
            "Sources pymdp_trace.csv and inference_trace.csv compare the pinned pymdp agent with the transparent reference "
            "on identical tensors. Under matched priors the state posteriors agree to max abs delta "
            f"{_fmt_delta(context.get('pymdp_exact_delta'))}, validating the inference implementation; closed-loop "
            f"posteriors diverge to {_fmt_delta(context.get('pymdp_loop_delta'))} through policy-induced prior divergence "
            "because expected-free-energy action selection and the proxy-score rule propagate different priors. This is "
            "a synthetic action-selection diagnostic, not an inference bug."
        ),
        "pymdp_diagnostics": (
            "Source pymdp_trace.csv makes the pinned agent's internal surface visible. variational free energy per step "
            f"(mean {_fmt_plain(context.get('pymdp_mean_vfe'))} nats) shows fit-plus-complexity surprise, wall time "
            f"(median {_fmt_plain(context.get('pymdp_median_step_ms'))} ms) reports computational cost, and expected free "
            "energy per policy shows the quantity used for policy ranking. The panel supports interpretability claims, "
            "not state-of-the-art predictive performance on synthetic waypoints."
        ),
        "state_recovery": (
            "Sources ground_truth.csv and inference traces score latent-state recovery against the simulator's own truth. "
            "Posterior mass is plotted for both lanes when available, while top-1 correctness is plotted for the reference lane; with location observed, high "
            f"recovery is a sanity bound (reference mean mass {_fmt_plain(context.get('recovery_reference_mass'))}, "
            f"top-1 {_fmt_plain(context.get('recovery_reference_top1'))}), not an achievement. Calibration and "
            "location-withheld recovery carry the substantive burden elsewhere. Synthetic software evaluation, not "
            "empirical validity."
        ),
        "sparsity_recovery": (
            "Source sparsity_analysis.json subsamples each traveller's reports and re-estimates both the full model and "
            "the deep-latent no-location model at each sparsity level. Mean true-cell posterior mass prices direct "
            "location evidence, while entropy and pymdp variational free energy show how uncertainty and closed-loop "
            "cost change as sequences lengthen. These fixture-lane measurements separate filter recovery from "
            "policy-induced prior divergence and are not empirical claims."
        ),
        "calibration": (
            "Source calibration.json decomposes predictive fit, confidence, and null-relative recovery for both inference "
            "lanes on shared axes. The reliability panel compares top-1 confidence with realized top-1 accuracy; the loss panel splits "
            "joint predictive loss into modality margins and their interaction gap; the skill panel moves from a uniform "
            "floor to base-rate and temporal-persistence nulls, with bin counts and tiny joint-loss skills printed directly. Persistence is the stricter null for autocorrelated "
            "synthetic mobility, so apparent by-construction wins are not overpromoted."
        ),
        "anticipatory": (
            "Source anticipatory_analysis.json tests whether belief propagation fills sparse gaps better than persistence. "
            "The left panel shows speed-conditioned off-diagonal transition mass; the middle panel reports gap-filling recovery skill "
            "against the persistence null, and the right panel keeps the location-blind limit on its own scale. Positive bars would support gap filling beyond the "
            "oracle previous-cell baseline, while negative bars are retained as bounded nulls. Speed is derived from "
            "reports, and the result is a synthetic software measurement."
        ),
        "efe_decomposition": (
            "Source active_sensing_analysis.json decomposes expected free energy into epistemic and pragmatic value. As "
            "preferences sharpen, the location-fix epistemic share changes rather than remaining constant; the budgeted "
            "sensing panel compares information-led acquisition with a budget-matched random schedule. Near-zero or "
            "negative margins are reported as an active-sensing null, showing that information gain does not robustly beat "
            "blind acquisition on this synthetic trace."
        ),
        "anomaly_detection": (
            "Source anomaly_analysis.json injects routine breaks into the synthetic observation stream after fitting the "
            "clean routine. Filter surprise, measured by variational free energy, is scored by ROC AUC against injected "
            "labels and compared with first-occurrence novelty for rare-cell and random-cell breaks. The result is a "
            "software diagnostic of synthetic perturbations, not empirical anomaly detection."
        ),
        "recoverability_frontier": (
            "Source frontier_analysis.json compares observation-modality subsets on point recovery and occupancy-structure "
            "recovery. Mean posterior mass on the true cell shows what direct location buys; occupancy total-variation "
            "distance shows which aggregate structure survives minimization. The frontier is an oversight-positive map of "
            "synthetic recoverability, not evidence about real populations."
        ),
        "return_structure": (
            "Source return_structure_analysis.json tests whether a visited-cell return feature lowers move-step next-cell "
            "entropy beyond persistence headroom. The bars show entropy reduction, available null headroom, and the "
            "prototype-before-claim score. A promoted result becomes a candidate prior; a null remains an explicit bound "
            "on this synthetic trace rather than a hidden failure."
        ),
        "variance_bands": (
            "Source variance_bands.json repeats key analyses across synthetic-generator realizations and plots medians with "
            "min-max bands. Gap-filling skill, anomaly separability, and minimization-frontier recovery are shown as "
            "realization-sensitive program diagnostics. These bands describe synthetic reruns, not population confidence "
            "intervals or empirical uncertainty."
        ),
        "upstream_sensitivity": (
            "Source upstream_sensitivity.json records whether a second buildable FractalRabbit commit produced comparable "
            "tensor or recovery distances under the current JDK/Maven toolchain. If no comparable build completes, the "
            "blocked state is rendered explicitly. The captioned artifact prevents silent upstream drift from masquerading "
            "as validated simulator robustness."
        ),
        "fractal_dimension_sweep": (
            "Source fractal_dimension_sweep.json varies the FractalRabbit dimension parameter while holding the expanded "
            "configuration fixed, then reports filter diagnostics (left panel) and, when the lane populates it, place-return "
            "texture (right panel; an explicit 'no revisit data in this lane' marker is drawn when the return texture is "
            "unavailable). The parameter is treated as a synthetic generator knob over support roughness and revisit "
            "structure. It is not estimated fractal dimension for real mobility."
        ),
        "circadian_phase_signal": (
            "Source circadian_phase_analysis.json tests whether phase-conditioned occupancy carries predictive structure. "
            "The heatmap shows cell occupancy by phase, and the bars compare phase-conditioned gain with a permuted-phase "
            "null plus Jensen-Shannon separation. A positive result can become a candidate prior; a null remains bounded "
            "synthetic evidence rather than a behavioral claim."
        ),
        "partial_observability": (
            "Source partial_observability_analysis.json compares soft belief, hard MAP, and observed Markov under the same "
            "known symmetric emission with flip probability epsilon, transition, observations, and priors. The soft-belief advantage is zero for a clean "
            "channel and grows monotonically with noise. The green overlay separates the replicated cross-draw range and mean "
            "from the primary fixture draw, identifying belief marginalization as the mechanism behind the gain."
        ),
        "sequence_baselines": (
            "Source sequence_baseline_analysis.json implements the prequential baseline ladder before any stronger "
            "performance language is allowed. Online base-rate, persistence, Markov, phase-conditioned, metadata-conditioned, "
            "and return-boosted variants are scored on next-cell predictive loss before observing the current report. "
            "pymdp must clear these transparent synthetic sequence nulls before the manuscript can claim predictive leadership."
        ),
        "missingness_model": (
            "Source missingness_model_analysis.json models reporting cadence as its own synthetic process. Phase-conditioned "
            "short-gap probabilities and online log-loss gains for phase, traveller, and combined contexts are compared "
            "with a pooled Bernoulli reporter and a permuted-phase null. A promoted result is a candidate reporting-process "
            "model, not a claim about real nonresponse or missingness behavior."
        ),
        "model_gate_matrix": (
            "Source model_gate_matrix.json puts fixture and external lanes on one readiness contract for the sequence "
            "baseline ladder, reporting-process structure, and return-plus-phase prior. Completed cells show current-lane "
            "metrics and verdicts; unavailable lanes render unavailable rather than zero. The matrix preserves lane "
            "honesty while newer promoted synthetic families move to the forward-gates artifact."
        ),
        "m10_forward_gates": (
            "Source forward_gates_summary.json summarizes the promoted M10+ artifact families: latent-regime HMM, neural "
            "baseline, learned stay-rate, held-out reporting process, cross-generator transfer, EFE location frontier, "
            "linkage-risk frontier, and poisoning robustness. Metrics and verdicts are shown together, including honest "
            "nulls. External rows count only same-run synthetic evidence, not stale simulator results."
        ),
        "factorial_pymdp_modes": (
            "Source factorial_pymdp_analysis_fixture.json exposes the factorial pymdp hidden-factor diagnostics. Posterior "
            "entropy is shown against each factor's computed maximum entropy (all three factors), while mode confidence and label stability are shown for the movement-mode and synthetic "
            "cognitive-proxy reporting factors only. The cognitive-proxy factor is derived from gap and burst texture; it is "
            "not a claim about real cognition, intent, or mental state."
        ),
        "latent_mode_recovery": (
            "Source factorial_pymdp_analysis_fixture.json validates the factorial lane and reports the cost of factorizing "
            "coupled likelihoods. The pymdp variational posterior is checked against an independent analytic mean-field "
            "solver, the mean-field-versus-exact marginal gap is shown separately, and recovery sharpness is plotted per "
            "factor. Synthetic software measurements, not empirical claims."
        ),
        "factorial_model_gate_matrix": (
            "Source factorial_pymdp_analysis_fixture.json compares predictive loss for the transparent reference filter, "
            "single-factor pymdp, factorial pymdp, switching-HMM, and particle/state-space baselines, with persistence "
            "marked as a null. The performance gate reports whether the factorial lane leads; a non-leading verdict is "
            "part of the synthetic result. Multi-factor interpretability does not establish state-of-the-art predictive performance."
        ),
        "efe_location_frontier": (
            "Source efe_location_frontier.json prices location in the model's epistemic currency. The figure compares the "
            "expected cell-information value of direct location with the non-location metadata bundle, checks metadata "
            "against a structure-destroyed null, and includes a positive control for test power. This is an "
            "oversight-positive synthetic boundary, not a collection recommendation."
        ),
        "linkage_risk_frontier": (
            "Source linkage_risk_frontier.json measures aggregate synthetic traveller re-link accuracy under full-location "
            "and minimized-metadata regimes, with structure-null bands and a positive control showing the weak linker can "
            "detect injected identity structure. The output is a defensive minimization-risk statistic, never a per-person "
            "linkage product or real-population claim."
        ),
        "poisoning_robustness": (
            "Source poisoning_robustness.json tests VFE surprise as a defensive data-integrity monitor under crude "
            "synthetic stressors. VFE is compared with a Markov-surprise baseline and structure-null plus positive-control "
            "checks; separated corruptions and remaining nulls are reported by the generated verdict. This is robustness "
            "triage, not a generalized assurance claim."
        ),
        "privacy_frontier": (
            "Source privacy_frontier_analysis.json quantifies what minimization buys back on the synthetic trace. Unicity "
            "under coarsened modality sets and co-travel coincidence rates are shown side by side for the same regimes. "
            "On this fixture unicity remains at the ceiling across every rendered coarsening, and co-travel coincidence "
            "does too except for the full 'all'-modality set, which dips slightly below ceiling (0.89) - an honest, "
            "near-no-relief finding rather than a uniform one. "
            "The figure supports oversight-positive risk accounting, not association tasking or claims about real people."
        ),
        "hardened_anomaly_detection": (
            "Source hardened_anomaly_analysis.json re-tests VFE surprise against an online Markov-surprise baseline on "
            "every hardened synthetic perturbation present in the artifact, including speed-regime shifts, preferred-set drift, timing manipulation, and adversarial missingness. Operating-point "
            "metrics live in the artifact. The figure strengthens the routine-break diagnostic while keeping anomaly "
            "claims inside synthetic software evaluation."
        ),
        "sota_leaderboard": (
            "Source sota_program_leaderboard.csv ranks every locked-split synthetic FractalRabbit row by mean next-cell "
            "predictive loss. Bootstrap whiskers share one loss axis; the annotation adds Brier score, ECE, top-1 accuracy, "
            "runtime, paired delta, one-sided p-value, and holdout-traveller count. Precision-return AIF and soft-filter AIF "
            "are scored beside switching HMM, particle, neural, attention, Markov, persistence, and base-rate comparators. "
            "The comparison is direct: active inference is scored directly, and the strongest fair baseline and blocked program "
            "gate remain visible."
        ),
        "aif_regime_map": (
            "Source aif_regime_sota_gate.json maps the separate regime-qualified synthetic search for active-inference "
            "advantage. Each cell is loss(strongest fair baseline) minus loss(best AIF-family candidate) for one "
            "synthetic regime and realization seed on a common signed scale; positive values favor AIF. Green rings mark traveller-clustered "
            "CI-separated local wins that also clear the oracle/leakage floor, while the side panel reports p/max-CI "
            "clearance and the Holm-adjusted familywise failure from multiplicity_ledger.json."
        ),
        "fair_inference": (
            "Source fair_inference.json scores an oracle-stripped ladder in which every predictor learns the same transition "
            "from the same observed cells. The left panel ranks oracle, hard argmax, soft marginalization, return-boosted "
            "soft prediction, and recursive active inference with bootstrap whiskers. The right panel traces recursion's "
            "incremental advantage as emission noise rises. Most of the gain comes from soft marginalization; recursion adds "
            "value conditionally as the observation channel degrades."
        ),
        "fair_power_regime": (
            "Source power_analysis.json maps active inference minus return-boosted soft Markov across traveller count and "
            "observation noise on deterministic synthetic streams. Color reports signed advantage in nats. Solid rings mark "
            "traveller-clustered interval separation when enough holdout clusters exist; dashed rings mark cross-seed sign "
            "stability where within-draw intervals are underpowered. The map shows large small-sample margins, vanishing "
            "clean-emission margins at scale, and renewed advantage as observation noise rises."
        ),
        "multiplicity_ledger": (
            "Source multiplicity_ledger.json enumerates every synthetic AIF-versus-fair directional comparison with a one-sided "
            "clustered-bootstrap p-value. Each dumbbell joins the raw and Holm-adjusted p-values on a log axis against alpha; "
            "markers identify rows whose raw p-value and clustered interval both clear. The annotation reports family size, "
            "uncorrected clearers, Holm and Benjamini-Hochberg survivors, the best adjusted p-value, and the separate "
            "partial-observability Bonferroni sign-test bound."
        ),
        "confound_taxonomy": (
            "Source confound_taxonomy.json maps every apparent active-inference advantage this synthetic benchmark raised and then "
            "checked onto the confound class that explains it, with each entry marked as live-verified, documentation-verified, "
            "or explicitly unverified when the current external trace needed by the hook is absent. "
            "Markers sit on confound-class rows; diamonds flag confounds caught only by cross-vendor audit, and a star marks "
            "the single fair-by-construction surviving positive. The panel turns the project's pattern of dissolved naive wins "
            "into an auditable map rather than a performance result."
        ),
        "regime_mechanism_synthesis": (
            "Source regime_mechanism_synthesis.json joins three checks per regime: AIF-versus-strongest-fair magnitude and "
            "traveller-clustered interval from aif_regime_search.json, Holm survival from multiplicity_ledger.json, and the "
            "dominant-cell, pooled-Markov, return-boosted-soft, and oracle-previous-cell confound battery. Dot and whisker "
            "encode magnitude and interval; color encodes the mechanism verdict; the margin labels confound clearance and "
            "spatial confinement. Each row names its current blocker, keeping directional loss, confounding, and multiplicity "
            "as separate reasons."
        ),
        "regime_switch_phase_diagnostic": phase_caption,
        "privacy_utility_frontier": privacy_utility_caption,
        "privacy_utility_repeats": (
            "Source privacy_utility_frontier_repeats.json reruns the privacy-utility frontier over the canonical fixture "
            "and any generated variance-realization observation files that are present. Completed rows show finest and "
            "coarsest shuffle-corrected utility spread; unavailable realization rows are counted explicitly rather than "
            "imputed. The figure supports only a repeated-draw audit surface: when required rows are unavailable or the "
            "full repeated gate does not clear, stability and graceful-degradation wording remain blocked."
        ),
        "synthetic_task_utility_privacy": (
            "Source synthetic_task_utility_privacy.json joins predictive utility, event uniqueness, co-travel coincidence, "
            "recoverability, and linkage-minimization status. Bars compare utility with the configured utility floor and risk "
            "with the risk ceiling; each regime states whether both thresholds define a joint operating point. The synthetic "
            "artifact contains such points, so utility wording remains conditional on joint clearance."
        ),
        "model_comparison_under_noise": (
            "Source model_comparison_leaderboard.json ranks every implemented next-cell predictor by fair loss on the locked synthetic "
            "noisy-emission FractalRabbit lane. Active inference, soft filter, switching-noise HMM, and particle filter form "
            "the leading belief-preserving cluster. The plot marks uniform chance, the belief-family band, the best "
            "non-belief boundary, top-row accuracy, the belief-versus-nonbelief gap, and active inference's paired interval "
            "against the strongest non-AIF method. Marginal whiskers are descriptive; the paired interval determines separation."
        ),
        "aif_deepening_candidates": (
            "Source aif_deepening_candidates.json tracks active-inference deepening surfaces by predictive admission: "
            "factorial tensor provenance, multiscale soft-belief backoff, and online adaptation. Rows show whether each "
            "surface has entered the locked Candidate.predict(PredictContext) protocol and whether any predictive gate "
            "clears. Online adaptation is now a scored candidate that improves the fixed AIF ablation but not the "
            "predeclared online-base-rate comparator; audit-only rows remain visible so tensor, EFE, and VFE ideas cannot "
            "substitute for prediction."
        ),
        "dataflow": (
            "From the figure generator and from source pipeline modules, this panel summarizes the rendered artifact dataflow: pinned "
            "simulator or checked fixture, evidence-model artifacts, dual inference lanes, claim-gate artifacts, figure registry, "
            "and manuscript variables. Every drawn box resolves to an auditable local artifact, so the reader can trace claims "
            "without trusting prose alone."
        ),
    }
    sources = {
        "evidence_atlas": f"output/runs/{run_name}/metrics.json",
        "waypoint_paths": f"output/runs/{run_name}/observations.csv",
        "spatial_bins": f"output/runs/{run_name}/discrete_features.csv",
        "spatial_resolution_frontier": "output/runs/spatial_resolution_frontier.json",
        "reporting_bursts": f"output/runs/{run_name}/discrete_features.csv",
        "speed_categories": f"output/runs/{run_name}/discrete_features.csv",
        "transition_heatmap": f"output/runs/{run_name}/discrete_features.csv",
        "likelihood_panels": f"output/runs/{run_name}/discrete_features.csv",
        "posterior_entropy": f"output/runs/{run_name}/inference_trace.csv",
        "policy_proxy": f"output/runs/{run_name}/policy_trace.csv",
        "candidate_scores": f"output/runs/{run_name}/candidate_scores.json",
        "negative_controls": "output/runs/negative_controls_summary.json",
        "pymdp_agreement": f"output/runs/{run_name}/pymdp_trace.csv",
        "pymdp_diagnostics": f"output/runs/{run_name}/pymdp_trace.csv",
        "state_recovery": f"output/runs/{run_name}/ground_truth.csv",
        "sparsity_recovery": "output/runs/sparsity_analysis.json",
        "calibration": f"output/runs/{run_name}/calibration.json",
        "anticipatory": "output/runs/anticipatory_analysis.json",
        "efe_decomposition": "output/runs/active_sensing_analysis.json",
        "anomaly_detection": "output/runs/anomaly_analysis.json",
        "recoverability_frontier": "output/runs/frontier_analysis.json",
        "return_structure": "output/runs/return_structure_analysis.json",
        "variance_bands": "output/runs/variance_bands.json",
        "upstream_sensitivity": "output/runs/upstream_sensitivity.json",
        "fractal_dimension_sweep": "output/runs/fractal_dimension_sweep.json",
        "circadian_phase_signal": "output/runs/circadian_phase_analysis.json",
        "partial_observability": "output/runs/partial_observability_analysis.json",
        "sequence_baselines": "output/runs/sequence_baseline_analysis.json",
        "missingness_model": "output/runs/missingness_model_analysis.json",
        "model_gate_matrix": "output/runs/model_gate_matrix.json",
        "m10_forward_gates": "output/runs/forward_gates_summary.json",
        "factorial_pymdp_modes": "output/runs/factorial_pymdp_analysis_fixture.json",
        "latent_mode_recovery": "output/runs/factorial_pymdp_analysis_fixture.json",
        "factorial_model_gate_matrix": "output/runs/factorial_pymdp_analysis_fixture.json",
        "efe_location_frontier": "output/runs/efe_location_frontier.json",
        "linkage_risk_frontier": "output/runs/linkage_risk_frontier.json",
        "poisoning_robustness": "output/runs/poisoning_robustness.json",
        "privacy_frontier": "output/runs/privacy_frontier_analysis.json",
        "hardened_anomaly_detection": "output/runs/hardened_anomaly_analysis.json",
        "sota_leaderboard": "output/runs/sota_program_leaderboard.csv",
        "aif_regime_map": "output/runs/aif_regime_sota_gate.json",
        "fair_inference": "output/runs/fair_inference.json",
        "fair_power_regime": "output/runs/power_analysis.json",
        "multiplicity_ledger": "output/runs/multiplicity_ledger.json",
        "confound_taxonomy": "output/runs/confound_taxonomy.json",
        "regime_mechanism_synthesis": "output/runs/regime_mechanism_synthesis.json",
        "regime_switch_phase_diagnostic": "output/runs/regime_switch_phase_diagnostic.json",
        "privacy_utility_frontier": "output/runs/privacy_utility_frontier.json",
        "privacy_utility_repeats": "output/runs/privacy_utility_frontier_repeats.json",
        "synthetic_task_utility_privacy": "output/runs/synthetic_task_utility_privacy.json",
        "model_comparison_under_noise": "output/runs/model_comparison_leaderboard.json",
        "aif_deepening_candidates": "output/runs/aif_deepening_candidates.json",
        "dataflow": "source pipeline modules",
    }
    caption_contracts = {
        "evidence_atlas": (
            "front-loaded orientation to the artifact-bound synthetic evidence chain and safety guardrails",
            "surveillance readiness, real-map inference, operational tasking, or empirical mobility validation",
        ),
        "waypoint_paths": (
            "inspection of sparse report ordering and coverage gaps",
            "continuous-track reconstruction, surveillance, or empirical mobility inference",
        ),
        "spatial_bins": (
            "the fixed synthetic analysis-cell boundary used by the tensors",
            "geographic interpretation of places or real-world map validity",
        ),
        "spatial_resolution_frontier": (
            "the grid-selection guardrail for observed-location fit and metadata-only recovery",
            "promotion of no-location recovery beyond the artifact frontier",
        ),
        "reporting_bursts": (
            "the synthetic reporting-process texture used by later missingness gates",
            "a behavioral account of real reporting cadence",
        ),
        "speed_categories": (
            "the derived kinematic evidence channel used by the deep-latent lane",
            "ground-truth claims about unobserved movement between reports",
        ),
        "transition_heatmap": (
            "normalization and sparsity checks for the synthetic B-tensor surface",
            "external movement-model validation",
        ),
        "likelihood_panels": (
            "the likelihood-audit boundary for location, time, speed, and burst modalities",
            "empirical sensor validation or independence claims outside the configured approximation",
        ),
        "posterior_entropy": (
            "reference-filter surprise and uncertainty diagnostics over synthetic reports",
            "proof of real trajectory reconstruction",
        ),
        "policy_proxy": (
            "candidate-control separability under the proxy scoring rule",
            "operational control advice or controller validation",
        ),
        "candidate_scores": (
            "recomputable closed-loop candidate ranking inside the fixture lane",
            "external simulator performance leadership",
        ),
        "negative_controls": (
            "artifact-level falsification checks for destroyed or degenerate structure",
            "empirical mobility evidence",
        ),
        "pymdp_agreement": (
            "implementation agreement and policy-induced prior divergence diagnostics",
            "a state-of-the-art performance claim",
        ),
        "pymdp_diagnostics": (
            "interpretability of the real pymdp lane through VFE, EFE, policy, and runtime surfaces",
            "predictive-performance leadership",
        ),
        "state_recovery": (
            "synthetic-truth recovery sanity checks and calibration handoff",
            "empirical validity",
        ),
        "sparsity_recovery": (
            "how recovery and uncertainty change under fixture-lane report thinning",
            "population generalization across real sampling regimes",
        ),
        "calibration": (
            "calibration floors, null hierarchy, and persistence-relative skill checks",
            "promotion of by-construction location wins as substantive recovery",
        ),
        "anticipatory": (
            "gap-filling skill or bounded nulls against the persistence oracle",
            "claims that propagation predicts real hidden paths",
        ),
        "efe_decomposition": (
            "the separation of epistemic and pragmatic value in synthetic active sensing",
            "a robust collection or sensing recommendation",
        ),
        "anomaly_detection": (
            "VFE surprise behavior under injected synthetic routine breaks",
            "deployment-ready anomaly detection",
        ),
        "recoverability_frontier": (
            "oversight-positive accounting of recovery versus minimization",
            "claims about real populations or individual recoverability",
        ),
        "return_structure": (
            "whether return features merit promotion to a candidate prior",
            "a hidden behavioral law of movement",
        ),
        "variance_bands": (
            "sensitivity of key diagnostics across synthetic-generator reruns",
            "population confidence intervals",
        ),
        "upstream_sensitivity": (
            "explicit blocking or comparison of buildable upstream simulator commits",
            "silent robustness to upstream drift",
        ),
        "fractal_dimension_sweep": (
            fractal_support,
            "estimation of real-mobility fractal dimension",
        ),
        "circadian_phase_signal": (
            "phase-conditioned prior candidacy against a permuted-phase null",
            "a behavioral circadian claim",
        ),
        "sequence_baselines": (
            "the prequential baseline ladder required before performance language",
            "pymdp predictive leadership until the gate is cleared",
        ),
        "missingness_model": (
            "reporting-process structure against pooled and permuted nulls",
            "claims about real nonresponse behavior",
        ),
        "model_gate_matrix": (
            "lane readiness for promoted fixture and external evidence families",
            "treating unavailable external lanes as failures or successes",
        ),
        "m10_forward_gates": (
            "current M10+ synthetic gate outcomes including honest nulls",
            "stale or empirical simulator-validation conclusions",
        ),
        "factorial_pymdp_modes": (
            "factor-specific posterior diagnostics in the factorial pymdp lane",
            "claims about real cognition, intent, or mental state",
        ),
        "latent_mode_recovery": (
            "mean-field agreement, exact-marginal gap, and factor recovery costs",
            "empirical latent-mode validation",
        ),
        "factorial_model_gate_matrix": (
            "performance-gate comparison between factorial, single-factor, and sequence baselines",
            "state-of-the-art predictive performance when the gate is non-leading",
        ),
        "efe_location_frontier": (
            "the epistemic value and positive-control boundary for location minimization",
            "a collection recommendation",
        ),
        "linkage_risk_frontier": (
            "defensive aggregate relink-risk accounting under minimization",
            "per-person linkage, tasking, or real-population inference",
        ),
        "poisoning_robustness": (
            "defensive VFE-integrity triage under crude synthetic corruptions",
            "generalized robustness assurance",
        ),
        "privacy_frontier": (
            "synthetic minimization accounting for unicity and co-travel coincidence risk",
            "association tasking or claims about real people",
        ),
        "hardened_anomaly_detection": (
            "harder synthetic routine-break checks against an online Markov-surprise baseline",
            "deployment-ready anomaly monitoring",
        ),
        "sota_leaderboard": (
            "a direct locked-split SOTA-program comparison that includes active inference and the strongest fair baselines on the same FractalRabbit lane",
            "a global state-of-the-art performance claim, operational tasking, or real-mobility validity",
        ),
        "aif_regime_map": (
            "a regime-local synthetic search for where AIF-family belief marginalization beats the strongest implemented fair baseline with clustered-CI separation",
            "global SOTA, external simulator SOTA, empirical transfer, operational readiness, or real-mobility validation",
        ),
        "fair_inference": (
            "a fair, transition-source-matched comparison of how much latent-cell inference helps, decomposed into soft marginalization and recursion",
            "a state-of-the-art claim, an oracle-access result, or real-mobility validity",
        ),
        "partial_observability": (
            "a fair-by-construction isolation of the genuine partial-observability regime where belief marginalization beats committing to a point estimate under a known noisy emission, zero at zero noise and growing monotonically with it",
            "active-inference-specific superiority, a state-of-the-art claim, an advantage under full observability, or real-mobility validity",
        ),
        "fair_power_regime": (
            "a clustered-bootstrap regime map showing the fair AIF advantage is large with few travellers, vanishes at scale under a clean emission, and re-emerges under observation noise",
            "a blanket state-of-the-art claim, an oracle-access result, or real-mobility validity",
        ),
        "multiplicity_ledger": (
            "a multiplicity audit reporting how many directional comparisons survive Holm and Benjamini-Hochberg correction across the deterministically enumerated family",
            "a state-of-the-art claim, an oracle-beating result, or a real-mobility claim",
        ),
        "confound_taxonomy": (
            "an auditable map of which confound class explains each dissolved active-inference advantage and which catches required cross-vendor review",
            "a state-of-the-art claim, an operational capability, or a real-mobility claim",
        ),
        "regime_mechanism_synthesis": (
            "a per-regime join of AIF-vs-fair magnitude and CI, multiplicity-survival status, and confound-battery/mobility standing into one mechanism verdict per regime",
            "a state-of-the-art claim, a global or operational capability, or a real-mobility claim",
        ),
        "regime_switch_phase_diagnostic": (
            "a structural, per-movement-phase breakdown of an AIF-vs-fair-baseline loss on one regime, showing whether the loss is uniform or concentrated across phase segments",
            "a state-of-the-art claim, an operational capability, or a real-mobility claim",
        ),
        "privacy_utility_frontier": (
            privacy_utility_support,
            "a state-of-the-art claim, association over real people, or a real-mobility claim",
        ),
        "privacy_utility_repeats": (
            "a repeated synthetic privacy-utility audit with explicit unavailable-row accounting",
            "repeat-stability wording when required realization rows are missing, empirical privacy evidence, or real-mobility claims",
        ),
        "synthetic_task_utility_privacy": (
            "a thresholded synthetic task-utility/privacy screen that blocks release-utility wording unless utility and privacy thresholds jointly clear",
            "release utility, operational usefulness, privacy guarantees, or real-world utility claims when the joint gate is blocked",
        ),
        "model_comparison_under_noise": (
            "a within-regime ranking showing belief-marginalizing methods, active inference included, lead the implemented predictors under noisy emission",
            "a global state-of-the-art claim, an oracle-beating result, or a real-mobility claim",
        ),
        "aif_deepening_candidates": (
            "a predictive-admission map where online adaptation is scored while factorial and multiscale surfaces remain audit-only",
            "predictive improvement, SOTA promotion, or operational claims from EFE, VFE, tensor, or adaptation surfaces alone",
        ),
        "dataflow": (
            "traceability from simulator boundary through generated manuscript variables",
            "trust in prose without artifacts",
        ),
    }
    support, boundary = caption_contracts[key]
    caption = f"{captions[key]} Evidence: {support}. Boundary: {boundary}."
    return FigureRegistryEntry(
        key=key,
        filename=filename,
        caption=caption,
        alt_text=caption,
        source=sources[key],
        data_derived=key != "dataflow",
    )


def _fmt_delta(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2e}"


def _fmt_plain(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _set_metadata(path: Path, dpi: int) -> None:
    # Matplotlib receives dpi at save time; this hook keeps the registry path stable for future metadata expansion.
    if dpi <= 0:
        raise ValueError("dpi must be positive")
