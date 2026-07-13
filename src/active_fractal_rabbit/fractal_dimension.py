"""Fractal-dimension sweep artifacts for the pinned synthetic generator.

The FractalRabbit CSV surface exposes a fractal-dimension parameter. This
module treats that parameter as a generator control: vary it, rerun the same
analysis contract, and report how occupancy, returns, and filter diagnostics
move. It does not estimate a fractal dimension from real trajectories.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable

from .robustness import empirical_interval

DEFAULT_FRACTAL_DIMENSIONS: tuple[float, ...] = (1.10, 1.33, 1.67, 1.90)
FRACTAL_DIMENSION_SWEEP_SCHEMA = "active-fractal-rabbit-fractal-dimension-sweep-v1"

SweepRunner = Callable[[float], dict[str, object] | None]


def summarize_fractal_dimension_sweep(
    rows: list[dict[str, object]],
    *,
    attempted_dimensions: Iterable[float],
    errors: dict[str, str] | None = None,
    source_config: str = "",
    dimension_control_active: bool = True,
) -> dict[str, object]:
    """Return a JSON-safe sweep summary from per-dimension metric rows."""

    attempted = [float(value) for value in attempted_dimensions]
    errors = errors or {}
    if not rows:
        status = "blocked"
    elif errors:
        status = "completed_with_failures"
    elif not dimension_control_active:
        status = "fixture_control_unvaried"
    else:
        status = "completed"
    metric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in {"run_id", "source"} and isinstance(value, int | float)
        }
    )
    if not dimension_control_active:
        caveat = (
            "The fractal-dimension control was INACTIVE on this run: the external stochastic "
            "simulator was unavailable, so the same fixture was replayed at every attempted "
            "dimension. Every non-parameter metric is therefore constant across dimensions "
            "(sweep spread is exactly zero by construction). This artifact is a fixture-replay "
            "invariance check — a determinism control — and demonstrates NO generator-knob "
            "sensitivity. It does not estimate a fractal dimension from real movement and does "
            "not validate FractalRabbit as a population mobility model."
        )
    else:
        caveat = (
            "Fractal dimension is varied as a synthetic-generator control over the support "
            "and return texture of generated waypoints. The artifact does not estimate a "
            "fractal dimension from real movement and does not validate FractalRabbit as a "
            "population mobility model."
        )
    return {
        "schema": FRACTAL_DIMENSION_SWEEP_SCHEMA,
        "status": status,
        "source_config": source_config,
        "attempted_dimensions": attempted,
        "dimension_control_active": dimension_control_active,
        "rows": rows,
        "metrics": {
            key: empirical_interval(
                float(row[key]) for row in rows if isinstance(row.get(key), int | float)
            )
            for key in metric_keys
        },
        "best_predictive_loss_dimension": _best_dimension(
            rows, "mean_predictive_loss", minimize=True
        ),
        "highest_revisit_dimension": _best_dimension(
            rows, "place_revisit_rate", minimize=False
        ),
        "errors": errors,
        "verdict": _verdict(rows, status),
        "caveat": caveat,
    }


def run_fractal_dimension_sweep(
    config_path: str | Path,
    *,
    dimensions: Iterable[float] = DEFAULT_FRACTAL_DIMENSIONS,
    runner: SweepRunner | None = None,
) -> dict[str, object]:
    """Run or inject a fractal-dimension sweep and return its artifact payload."""

    dimensions = tuple(float(value) for value in dimensions)
    if runner is not None:
        rows: list[dict[str, object]] = []
        errors: dict[str, str] = {}
        for dimension in dimensions:
            key = _dimension_key(dimension)
            try:
                result = runner(dimension)
            except (
                Exception
            ) as exc:  # pragma: no cover - injected failures are host-specific
                errors[key] = str(exc)
                continue
            if result is None:
                errors[key] = "runner returned no metrics"
                continue
            rows.append(
                _row_from_metrics(
                    dimension, f"fractal_dimension_{_slug(dimension)}", result
                )
            )
        return summarize_fractal_dimension_sweep(
            rows,
            attempted_dimensions=dimensions,
            errors=errors,
            source_config=str(config_path),
            dimension_control_active=True,
        )

    from .config import load_config
    from .pipeline import run_pipeline

    base = load_config(config_path)
    rows = []
    errors = {}
    dimension_control_active = base.fractalrabbit.execution == "external"
    for dimension in dimensions:
        key = _dimension_key(dimension)
        run_id = f"fractal_dimension_{_slug(dimension)}"
        try:
            params = replace(base.fractalrabbit.params, fractal_dimension=dimension)
            fractalrabbit = replace(
                base.fractalrabbit, params=params, output_prefix=run_id
            )
            model = replace(base.inference.model, engine="reference")
            inference = replace(base.inference, model=model)
            cfg = replace(
                base,
                run_id=run_id,
                output_dir=base.project_root / "output" / "runs" / run_id,
                fractalrabbit=fractalrabbit,
                inference=inference,
            )
            result = run_pipeline(cfg, config_path)
            rows.append(_row_from_metrics(dimension, run_id, result.metrics))
        except (
            Exception
        ) as exc:  # pragma: no cover - external failures are host-specific
            errors[key] = str(exc)
    return summarize_fractal_dimension_sweep(
        rows,
        attempted_dimensions=dimensions,
        errors=errors,
        source_config=str(config_path),
        dimension_control_active=dimension_control_active,
    )


def run_and_write_fractal_dimension_sweep(
    config_path: str | Path,
    out_path: str | Path,
    *,
    dimensions: Iterable[float] = DEFAULT_FRACTAL_DIMENSIONS,
    runner: SweepRunner | None = None,
) -> Path:
    """Run the sweep and write ``fractal_dimension_sweep.json``."""

    payload = run_fractal_dimension_sweep(
        config_path, dimensions=dimensions, runner=runner
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def _row_from_metrics(
    dimension: float, run_id: str, metrics: dict[str, object]
) -> dict[str, object]:
    return {
        "fractal_dimension": float(dimension),
        "run_id": run_id,
        "n_observations": _maybe_number(metrics, "n_observations"),
        "n_latent_states": _maybe_number(metrics, "n_latent_states"),
        "mean_predictive_loss": _maybe_number(metrics, "mean_predictive_loss"),
        "mean_posterior_entropy": _maybe_number(metrics, "mean_posterior_entropy"),
        "reference_recovery_mass": _maybe_number(
            metrics, "state_recovery_reference_mean_mass_on_true_cell"
        ),
        "holdout_mean_predictive_loss": _maybe_number(
            metrics, "holdout_mean_predictive_loss"
        ),
        "place_n_unique": _maybe_number(metrics, "place_n_unique"),
        "place_revisit_rate": _maybe_number(metrics, "place_revisit_rate"),
        "place_reports_per_place": _maybe_number(
            metrics, "place_mean_reports_per_place"
        ),
    }


def _maybe_number(metrics: dict[str, object], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, int | float) else None


def _best_dimension(
    rows: list[dict[str, object]], key: str, *, minimize: bool
) -> float | None:
    usable = [row for row in rows if isinstance(row.get(key), int | float)]
    if not usable:
        return None
    selected = (
        min(usable, key=lambda row: float(row[key]))
        if minimize
        else max(usable, key=lambda row: float(row[key]))
    )
    return float(selected["fractal_dimension"])


def _verdict(rows: list[dict[str, object]], status: str) -> str:
    if status == "blocked":
        return "blocked_no_buildable_dimension_runs"
    if status == "fixture_control_unvaried":
        return "fixture_only_dimension_not_resimulated"
    if not rows:
        return "unavailable"
    return "diagnostic_generator_sensitivity"


def _dimension_key(dimension: float) -> str:
    return f"{dimension:.2f}"


def _slug(dimension: float) -> str:
    return f"{dimension:.2f}".replace(".", "p")
