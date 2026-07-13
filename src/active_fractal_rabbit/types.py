"""Stable public types for Active Fractal Rabbit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FractalRabbitParams:
    """The seven public parameters consumed by FractalRabbit's CSV interface."""

    space_dimension: int = 2
    fractal_dimension: float = 1.33
    number_of_points: int = 5000
    number_of_travellers: int = 3000
    number_of_co_travellers: int = 1
    number_of_days: float = 14.0
    number_of_reports: float = 200.0


@dataclass(frozen=True)
class FractalRabbitConfig:
    """Configuration for the pinned external FractalRabbit simulator boundary."""

    upstream_repo: str
    upstream_commit: str
    cache_dir: Path
    execution: str
    params: FractalRabbitParams
    fixture_csv: Path | None = None
    output_prefix: str = "fractalrabbit_run"


@dataclass(frozen=True)
class ObservationConfig:
    """Discretization settings for sporadic waypoint observations."""

    spatial_bins: int
    time_bins_per_day: int
    speed_thresholds_km_per_day: tuple[float, float, float]
    spatial_discretizer: str = "grid"
    spatial_bounds: tuple[float, float, float, float] | None = None
    burst_threshold_rule: str = "median_positive"
    burst_fixed_days: float | None = None


@dataclass(frozen=True)
class ModelConfig:
    """Generative-model settings for the discrete active-inference analysis."""

    smoothing: float
    controls: tuple[str, ...] = ("empirical_continuation", "high_entropy_exploration")
    preference_prior: str = "uniform"
    modalities: tuple[str, ...] = ("location", "time_of_day", "speed", "report_burst")
    engine: str = "reference"
    factorization: str = "single"
    hidden_factors: tuple[str, ...] = ("cell", "movement_mode", "cognitive_proxy_mode")
    factorial_controls: tuple[str, ...] = ("mode_persistence", "mode_reset")


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment-level scoring and reproducibility settings."""

    repetitions: int = 1
    holdout_fraction: float = 0.0
    objective_direction: str = "minimize"
    tie_breaker: str = "declaration_order"


@dataclass(frozen=True)
class FigureConfig:
    """Figure generation settings and registry contract."""

    enabled: tuple[str, ...]
    dpi: int
    palette: dict[str, str]
    registry_path: Path


@dataclass(frozen=True)
class RobustnessConfig:
    """Robustness and expansion study settings."""

    n_realizations: int = 10
    upstream_candidate_limit: int = 5
    stop_on_external_unavailable: bool = True
    external_realization_fractal_dimensions: tuple[float, ...] = (1.21, 1.45, 1.7)


@dataclass(frozen=True)
class InferenceConfig:
    """Backward-compatible wrapper for observation and model settings."""

    observation: ObservationConfig
    model: ModelConfig

    @property
    def spatial_bins(self) -> int:
        return self.observation.spatial_bins

    @property
    def time_bins_per_day(self) -> int:
        return self.observation.time_bins_per_day

    @property
    def smoothing(self) -> float:
        return self.model.smoothing

    @property
    def speed_thresholds_km_per_day(self) -> tuple[float, float, float]:
        return self.observation.speed_thresholds_km_per_day


@dataclass(frozen=True)
class ClosedLoopCandidate:
    """One configurable simulator candidate for the next closed-loop iteration."""

    name: str
    params: FractalRabbitParams


@dataclass(frozen=True)
class ClosedLoopConfig:
    """Closed-loop experiment settings."""

    iterations: int
    objective: str
    candidates: tuple[ClosedLoopCandidate, ...]
    objective_direction: str = "minimize"
    tie_breaker: str = "declaration_order"


@dataclass(frozen=True)
class CandidateScore:
    """Transparent candidate score emitted by the closed-loop selector."""

    name: str
    objective: str
    metric_value: float
    reporting_ratio: float
    score: float
    rank: int


@dataclass(frozen=True)
class FigureRegistryEntry:
    """Registered figure metadata used by the manuscript and tests."""

    key: str
    filename: str
    caption: str
    alt_text: str
    source: str
    data_derived: bool


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project configuration."""

    run_id: str
    seed: int
    project_root: Path
    output_dir: Path
    fractalrabbit: FractalRabbitConfig
    inference: InferenceConfig
    experiment: ExperimentConfig
    figures: FigureConfig
    robustness: RobustnessConfig
    closed_loop: ClosedLoopConfig


@dataclass(frozen=True)
class WaypointObservation:
    """One sporadic waypoint observation emitted by FractalRabbit."""

    traveller_id: int
    days: float
    x_km: float
    y_km: float


@dataclass(frozen=True)
class PlaceObservation:
    """One ground-truth place identity from FractalRabbit's PLACES output."""

    traveller_id: int
    days: float
    place_id: int


@dataclass(frozen=True)
class ObservationSequence:
    """Discrete observation factors derived from waypoint rows."""

    observations: tuple[WaypointObservation, ...]
    is_first_observation: np.ndarray
    x_bins: np.ndarray
    y_bins: np.ndarray
    spatial_states: np.ndarray
    time_bins: np.ndarray
    speed_categories: np.ndarray
    burst_categories: np.ndarray
    speeds_km_per_day: np.ndarray
    intervals_days: np.ndarray
    spatial_bins: int
    num_spatial_states: int
    num_time_bins: int
    num_speed_categories: int
    num_burst_categories: int
    spatial_bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class PomdpModel:
    """Discrete POMDP arrays compatible with pymdp-style active inference."""

    A: tuple[np.ndarray, ...]
    B: np.ndarray
    C: tuple[np.ndarray, ...]
    D: np.ndarray
    observation_names: tuple[str, ...]
    control_names: tuple[str, ...]


@dataclass(frozen=True)
class LatentModeSequence:
    """Synthetic movement and reporting/cognitive-proxy factors derived from waypoints."""

    movement_modes: np.ndarray
    cognitive_proxy_modes: np.ndarray
    cell_delta: np.ndarray
    return_flags: np.ndarray
    movement_names: tuple[str, ...]
    cognitive_proxy_names: tuple[str, ...]


@dataclass(frozen=True)
class FactorialPomdpModel:
    """Multi-factor active-inference model for synthetic latent-mode analysis."""

    A: tuple[np.ndarray, ...]
    B: tuple[np.ndarray, ...]
    C: tuple[np.ndarray, ...]
    D: tuple[np.ndarray, ...]
    observation_names: tuple[str, ...]
    factor_names: tuple[str, ...]
    factor_sizes: tuple[int, ...]
    A_dependencies: tuple[tuple[int, ...], ...]
    B_dependencies: tuple[tuple[int, ...], ...]
    control_names: tuple[tuple[str, ...], ...]
    latent_modes: LatentModeSequence


@dataclass(frozen=True)
class FactorialInferenceResult:
    """Sequential outputs from the factorial pymdp lane."""

    factor_posteriors: dict[str, np.ndarray]
    predictive_loss: np.ndarray
    posterior_entropy: np.ndarray
    policy_posterior: np.ndarray
    expected_free_energy: np.ndarray
    variational_free_energy: np.ndarray
    step_runtime_ms: np.ndarray
    selected_actions: np.ndarray
    metrics: dict[str, float]


@dataclass(frozen=True)
class InferenceResult:
    """Latent state inference metrics and trajectories."""

    posterior_states: np.ndarray
    predictive_loss: np.ndarray
    posterior_entropy: np.ndarray
    policy_posterior: np.ndarray
    policy_proxy_score: np.ndarray
    policy_predictive_entropy: np.ndarray
    policy_preference_divergence: np.ndarray
    metrics: dict[str, float]


@dataclass(frozen=True)
class BaselineResult:
    """Fixed-transition baseline filter metrics and trajectories."""

    posterior_states: np.ndarray
    predictive_loss: np.ndarray
    posterior_entropy: np.ndarray
    metrics: dict[str, object]


@dataclass(frozen=True)
class PipelineResult:
    """End-to-end result for one configured run."""

    run_id: str
    config_path: Path
    output_dir: Path
    observations_path: Path
    metrics: dict[str, object]
    selected_next_candidate: str | None
    candidate_scores: tuple[CandidateScore, ...]
    artifacts: dict[str, Path]
    metadata: dict[str, Any]
