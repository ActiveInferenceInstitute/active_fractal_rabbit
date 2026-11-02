"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .pomdp import CANONICAL_MODALITIES
from .types import (
    ClosedLoopCandidate,
    ClosedLoopConfig,
    ExperimentConfig,
    FigureConfig,
    FractalRabbitConfig,
    FractalRabbitParams,
    InferenceConfig,
    ModelConfig,
    ObservationConfig,
    ProjectConfig,
    RobustnessConfig,
)

FRACTALRABBIT_UPSTREAM_REPO = (
    "https://github.com/NationalSecurityAgency/fractalrabbit.git"
)
FRACTALRABBIT_PINNED_COMMIT = "9933449c4f4fe1b26b6ac7bfdeeac76583085df5"

PARAMETER_CSV_ROWS: tuple[tuple[str, str], ...] = (
    ("space dimension", "space_dimension"),
    ("fractal dimension", "fractal_dimension"),
    ("number of points", "number_of_points"),
    ("number of travellers", "number_of_travellers"),
    ("number of co-travellers", "number_of_co_travellers"),
    ("number of days", "number_of_days"),
    ("number of reports", "number_of_reports"),
)

DEFAULT_FIGURES: tuple[str, ...] = (
    "evidence_atlas",
    "waypoint_paths",
    "spatial_bins",
    "spatial_resolution_frontier",
    "reporting_bursts",
    "speed_categories",
    "transition_heatmap",
    "likelihood_panels",
    "posterior_entropy",
    "policy_proxy",
    "candidate_scores",
    "negative_controls",
    "pymdp_agreement",
    "pymdp_diagnostics",
    "state_recovery",
    "sparsity_recovery",
    "calibration",
    "anticipatory",
    "efe_decomposition",
    "anomaly_detection",
    "recoverability_frontier",
    "return_structure",
    "variance_bands",
    "upstream_sensitivity",
    "fractal_dimension_sweep",
    "circadian_phase_signal",
    "partial_observability",
    "sequence_baselines",
    "missingness_model",
    "model_gate_matrix",
    "m10_forward_gates",
    "factorial_pymdp_modes",
    "latent_mode_recovery",
    "factorial_model_gate_matrix",
    "efe_location_frontier",
    "linkage_risk_frontier",
    "poisoning_robustness",
    "privacy_frontier",
    "hardened_anomaly_detection",
    "sota_leaderboard",
    "aif_regime_map",
    "fair_inference",
    "fair_power_regime",
    "multiplicity_ledger",
    "confound_taxonomy",
    "regime_mechanism_synthesis",
    "regime_switch_phase_diagnostic",
    "privacy_utility_frontier",
    "privacy_utility_repeats",
    "synthetic_task_utility_privacy",
    "model_comparison_under_noise",
    "aif_deepening_candidates",
    "dataflow",
)

DEFAULT_PALETTE: dict[str, str] = {
    "blue": "#0072B2",
    "green": "#009E73",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "red": "#D55E00",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "gray": "#595959",
}


def load_config(path: str | Path) -> ProjectConfig:
    """Load a YAML config into stable dataclasses."""

    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    project_root = _find_project_root(config_path)
    run_id = str(raw.get("run_id", config_path.stem))
    seed = int(raw.get("seed", 0))
    output_dir = _project_path(
        project_root, raw.get("output_dir", f"output/runs/{run_id}")
    )

    fr_raw = _mapping(raw, "fractalrabbit")
    params = _params(_mapping(fr_raw, "params"))
    execution = str(fr_raw.get("execution", "external"))
    if execution not in {"external", "fixture"}:
        raise ValueError("fractalrabbit.execution must be 'external' or 'fixture'")
    fixture_csv = fr_raw.get("fixture_csv")
    fr_config = FractalRabbitConfig(
        upstream_repo=str(fr_raw.get("upstream_repo", FRACTALRABBIT_UPSTREAM_REPO)),
        upstream_commit=str(fr_raw.get("upstream_commit", FRACTALRABBIT_PINNED_COMMIT)),
        cache_dir=_project_path(
            project_root, fr_raw.get("cache_dir", ".cache/fractalrabbit")
        ),
        execution=execution,
        params=params,
        fixture_csv=_project_path(project_root, fixture_csv) if fixture_csv else None,
        output_prefix=str(fr_raw.get("output_prefix", run_id)),
    )
    _validate_fractalrabbit(fr_config)

    if "inference" in raw:
        raise ValueError(
            "top-level inference block has been removed; use observation: and model: groups"
        )
    obs_raw = _mapping(raw, "observation")
    model_raw = _mapping(raw, "model")
    experiment_raw = _mapping(raw, "experiment")
    figures_raw = _mapping(raw, "figures")
    robustness_raw = _mapping(raw, "robustness")

    observation = ObservationConfig(
        spatial_bins=_positive_int(obs_raw, "spatial_bins"),
        time_bins_per_day=_positive_int(obs_raw, "time_bins_per_day"),
        speed_thresholds_km_per_day=_speed_thresholds(
            obs_raw.get("speed_thresholds_km_per_day", [10, 50, 150])
        ),
        spatial_discretizer=str(obs_raw.get("spatial_discretizer", "grid")),
        spatial_bounds=_spatial_bounds(obs_raw.get("spatial_bounds")),
        burst_threshold_rule=str(
            obs_raw.get("burst_threshold_rule", "median_positive")
        ),
        burst_fixed_days=_optional_positive_float(
            obs_raw.get("burst_fixed_days"), "burst_fixed_days"
        ),
    )
    _validate_observation(observation)
    model = ModelConfig(
        smoothing=_positive_float(model_raw, "smoothing"),
        controls=tuple(
            str(item)
            for item in model_raw.get(
                "controls", ["empirical_continuation", "high_entropy_exploration"]
            )
        ),
        preference_prior=str(model_raw.get("preference_prior", "uniform")),
        modalities=tuple(
            str(item)
            for item in model_raw.get(
                "modalities", ["location", "time_of_day", "speed", "report_burst"]
            )
        ),
        engine=str(model_raw.get("engine", "reference")),
        factorization=str(model_raw.get("factorization", "single")),
        hidden_factors=tuple(
            str(item)
            for item in model_raw.get(
                "hidden_factors", ["cell", "movement_mode", "cognitive_proxy_mode"]
            )
        ),
        factorial_controls=tuple(
            str(item)
            for item in model_raw.get(
                "factorial_controls", ["mode_persistence", "mode_reset"]
            )
        ),
    )
    _validate_model(model)
    inference = InferenceConfig(observation=observation, model=model)
    experiment = ExperimentConfig(
        repetitions=_positive_int_default(experiment_raw, "repetitions", 1),
        holdout_fraction=_bounded_float(
            experiment_raw.get("holdout_fraction", 0.0), "holdout_fraction", 0.0, 0.95
        ),
        objective_direction=str(experiment_raw.get("objective_direction", "minimize")),
        tie_breaker=str(experiment_raw.get("tie_breaker", "declaration_order")),
    )
    _validate_experiment(experiment)
    figures = FigureConfig(
        enabled=tuple(
            str(item) for item in figures_raw.get("enabled", DEFAULT_FIGURES)
        ),
        dpi=_positive_int_default(figures_raw, "dpi", 220),
        palette={
            str(key): str(value)
            for key, value in figures_raw.get("palette", DEFAULT_PALETTE).items()
        },
        registry_path=_project_path(
            project_root,
            figures_raw.get("registry_path", "output/figures/figure_registry.json"),
        ),
    )
    _validate_figures(figures)
    robustness = RobustnessConfig(
        n_realizations=_positive_int_default(robustness_raw, "n_realizations", 10),
        upstream_candidate_limit=_positive_int_default(
            robustness_raw, "upstream_candidate_limit", 5
        ),
        stop_on_external_unavailable=bool(
            robustness_raw.get("stop_on_external_unavailable", True)
        ),
        external_realization_fractal_dimensions=_positive_float_tuple_default(
            robustness_raw,
            "external_realization_fractal_dimensions",
            (1.21, 1.45, 1.7),
        ),
    )

    loop_raw = _mapping(raw, "closed_loop")
    candidates = tuple(
        ClosedLoopCandidate(
            name=str(item["name"]), params=replace(params, **_mapping(item, "params"))
        )
        for item in loop_raw.get("candidates", [])
    )
    closed_loop = ClosedLoopConfig(
        iterations=_positive_int(loop_raw, "iterations"),
        objective=str(loop_raw.get("objective", "predictive_loss")),
        candidates=candidates,
        objective_direction=str(
            loop_raw.get("objective_direction", experiment.objective_direction)
        ),
        tie_breaker=str(loop_raw.get("tie_breaker", experiment.tie_breaker)),
    )
    if closed_loop.objective not in {"predictive_loss", "posterior_entropy"}:
        raise ValueError(
            "closed_loop.objective must be predictive_loss or posterior_entropy"
        )
    _validate_closed_loop(closed_loop)

    return ProjectConfig(
        run_id=run_id,
        seed=seed,
        project_root=project_root,
        output_dir=output_dir,
        fractalrabbit=fr_config,
        inference=inference,
        experiment=experiment,
        figures=figures,
        robustness=robustness,
        closed_loop=closed_loop,
    )


def parameter_rows(params: FractalRabbitParams) -> list[tuple[str, str]]:
    """Return FractalRabbit parameters in the exact upstream CSV row order."""

    rows: list[tuple[str, str]] = []
    for label, attr in PARAMETER_CSV_ROWS:
        value = getattr(params, attr)
        rows.append((label, str(value)))
    return rows


def with_candidate(
    config: ProjectConfig, candidate: ClosedLoopCandidate, run_id: str
) -> ProjectConfig:
    """Return a copy of a project config with candidate simulator parameters."""

    fr_config = replace(
        config.fractalrabbit, params=candidate.params, output_prefix=run_id
    )
    return replace(
        config,
        run_id=run_id,
        output_dir=config.project_root / "output" / "runs" / run_id,
        fractalrabbit=fr_config,
    )


def _find_project_root(config_path: Path) -> Path:
    """Walk upward to the directory holding this project's pyproject.toml.

    Supports nested config folders such as ``configs/fixtures/``; ad-hoc test
    configs without a project marker fall back to the historical ``parents[1]``.

    A bare ``pyproject.toml`` check is not enough. When a config lives in a
    temporary directory inside an enclosing repository (a vault checkout whose
    ``TMPDIR`` points inside it, for example), the walk would otherwise anchor
    relative paths to that outer repository's root and read or write files in
    the wrong tree. The marker therefore requires the sibling
    ``src/active_fractal_rabbit`` package that belongs to this project.
    """

    for parent in config_path.parents:
        if (parent / "pyproject.toml").exists() and (
            parent / "src" / "active_fractal_rabbit"
        ).is_dir():
            return parent
    return config_path.parents[1]


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _params(raw: Mapping[str, Any]) -> FractalRabbitParams:
    params = FractalRabbitParams(
        space_dimension=int(raw.get("space_dimension", 2)),
        fractal_dimension=float(raw.get("fractal_dimension", 1.33)),
        number_of_points=int(raw.get("number_of_points", 5000)),
        number_of_travellers=int(raw.get("number_of_travellers", 3000)),
        number_of_co_travellers=int(raw.get("number_of_co_travellers", 1)),
        number_of_days=float(raw.get("number_of_days", 14.0)),
        number_of_reports=float(raw.get("number_of_reports", 200.0)),
    )
    if params.space_dimension != 2:
        raise ValueError(
            "This scaffold currently supports 2D FractalRabbit output only"
        )
    if not 0 < params.fractal_dimension <= params.space_dimension:
        raise ValueError("fractal_dimension must be in (0, space_dimension]")
    if params.number_of_points < 2:
        raise ValueError("number_of_points must be at least 2")
    if params.number_of_travellers < 1:
        raise ValueError("number_of_travellers must be positive")
    if params.number_of_co_travellers < 0:
        raise ValueError("number_of_co_travellers cannot be negative")
    if params.number_of_co_travellers > params.number_of_travellers:
        raise ValueError("number_of_co_travellers cannot exceed number_of_travellers")
    if params.number_of_days <= 0:
        raise ValueError("number_of_days must be positive")
    if params.number_of_reports <= 0:
        raise ValueError("number_of_reports must be positive")
    return params


def _project_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def _positive_int(raw: Mapping[str, Any], key: str) -> int:
    value = int(raw.get(key, 0))
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _positive_int_default(raw: Mapping[str, Any], key: str, default: int) -> int:
    value = int(raw.get(key, default))
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _positive_float_tuple_default(
    raw: Mapping[str, Any],
    key: str,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    value = raw.get(key, default)
    if not isinstance(value, list | tuple):
        raise ValueError(f"{key} must be a sequence of positive floats")
    parsed = tuple(float(item) for item in value)
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"{key} must contain positive floats")
    return parsed


def _positive_float(raw: Mapping[str, Any], key: str) -> float:
    value = float(raw.get(key, 0.0))
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _optional_positive_float(value: Any, key: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{key} must be positive")
    return parsed


def _bounded_float(value: Any, key: str, lower: float, upper: float) -> float:
    parsed = float(value)
    if not lower <= parsed <= upper:
        raise ValueError(f"{key} must be between {lower} and {upper}")
    return parsed


def _speed_thresholds(value: Any) -> tuple[float, float, float]:
    thresholds = tuple(float(v) for v in value)
    if len(thresholds) != 3:
        raise ValueError("speed_thresholds_km_per_day must contain three thresholds")
    if tuple(sorted(thresholds)) != thresholds:
        raise ValueError("speed thresholds must be sorted ascending")
    return thresholds


def _spatial_bounds(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    bounds = tuple(float(item) for item in value)
    if len(bounds) != 4:
        raise ValueError("spatial_bounds must contain xmin, xmax, ymin, ymax")
    xmin, xmax, ymin, ymax = bounds
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("spatial_bounds must satisfy xmin < xmax and ymin < ymax")
    return bounds


def _validate_fractalrabbit(config: FractalRabbitConfig) -> None:
    if config.upstream_commit != FRACTALRABBIT_PINNED_COMMIT:
        raise ValueError(
            "FractalRabbit commit must stay pinned to "
            f"{FRACTALRABBIT_PINNED_COMMIT}; update docs/tests before changing it"
        )
    if config.execution == "fixture" and config.fixture_csv is None:
        raise ValueError("fixture execution requires fractalrabbit.fixture_csv")


def _validate_observation(config: ObservationConfig) -> None:
    if config.spatial_discretizer not in {"grid", "fixed_bounds", "quantile"}:
        raise ValueError("spatial_discretizer must be grid, fixed_bounds, or quantile")
    if config.spatial_discretizer == "fixed_bounds" and config.spatial_bounds is None:
        raise ValueError("fixed_bounds spatial discretizer requires spatial_bounds")
    if config.burst_threshold_rule not in {"median_positive", "fixed_days"}:
        raise ValueError("burst_threshold_rule must be median_positive or fixed_days")
    if config.burst_threshold_rule == "fixed_days" and config.burst_fixed_days is None:
        raise ValueError("fixed_days burst threshold requires burst_fixed_days")


def _validate_model(config: ModelConfig) -> None:
    if len(config.controls) < 1:
        raise ValueError("model.controls must contain at least one control")
    allowed_controls = {
        "empirical_continuation",
        "high_entropy_exploration",
        "identity",
    }
    if any(control not in allowed_controls for control in config.controls):
        raise ValueError("model.controls contains an unknown control")
    if config.preference_prior not in {
        "uniform",
        "diagnostic_uncertainty",
        "diagnostic_gap",
    }:
        raise ValueError(
            "preference_prior must be uniform, diagnostic_uncertainty, or diagnostic_gap"
        )
    canonical_modalities = CANONICAL_MODALITIES
    if len(config.modalities) < 1:
        raise ValueError("model.modalities must contain at least one modality")
    if any(name not in canonical_modalities for name in config.modalities):
        raise ValueError(
            f"model.modalities entries must be among {canonical_modalities}"
        )
    if len(set(config.modalities)) != len(config.modalities):
        raise ValueError("model.modalities must not contain duplicates")
    if config.engine not in {"reference", "both"}:
        raise ValueError("model.engine must be 'reference' or 'both'")
    if config.factorization not in {"single", "factorial"}:
        raise ValueError("model.factorization must be single or factorial")
    # This validates against the fixed three-factor structure hardcoded in
    # factorial_pymdp.FACTOR_NAMES (build_factorial_pomdp does not read
    # config.hidden_factors -- it always builds cell/movement_mode/
    # cognitive_proxy_mode). model.hidden_factors is therefore a pinned
    # architecture invariant, not an independently configurable knob; a
    # circular import (factorial_pymdp imports config.load_config) prevents
    # importing FACTOR_NAMES directly here, so the tuple is kept in sync by
    # hand with factorial_pymdp.py.
    expected_factors = ("cell", "movement_mode", "cognitive_proxy_mode")
    if config.hidden_factors != expected_factors:
        raise ValueError(
            f"model.hidden_factors must be {expected_factors} "
            "(fixed to match factorial_pymdp.FACTOR_NAMES; not independently configurable)"
        )
    allowed_factorial_controls = {
        "mode_persistence",
        "mode_reset",
        "high_entropy_exploration",
        "identity",
    }
    if len(config.factorial_controls) < 1:
        raise ValueError("model.factorial_controls must contain at least one control")
    if any(
        control not in allowed_factorial_controls
        for control in config.factorial_controls
    ):
        raise ValueError("model.factorial_controls contains an unknown control")


def _validate_experiment(config: ExperimentConfig) -> None:
    if config.objective_direction not in {"minimize", "maximize"}:
        raise ValueError("objective_direction must be minimize or maximize")
    if config.tie_breaker not in {"declaration_order", "name"}:
        raise ValueError("tie_breaker must be declaration_order or name")


def _validate_closed_loop(config: ClosedLoopConfig) -> None:
    if config.objective_direction not in {"minimize", "maximize"}:
        raise ValueError("closed_loop.objective_direction must be minimize or maximize")
    if config.tie_breaker not in {"declaration_order", "name"}:
        raise ValueError("closed_loop.tie_breaker must be declaration_order or name")


def _validate_figures(config: FigureConfig) -> None:
    if not config.enabled:
        raise ValueError("figures.enabled must contain at least one figure key")
    if not config.palette:
        raise ValueError("figures.palette must contain at least one color")
