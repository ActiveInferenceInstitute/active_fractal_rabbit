"""Robustness artifacts: variance bands and upstream sensitivity."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Callable, Iterable


def empirical_interval(values: Iterable[float]) -> dict[str, float]:
    sample = sorted(float(value) for value in values)
    if not sample:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
        }
    n = len(sample)
    mean = sum(sample) / n
    # ddof=1 SD and quartiles alongside min-max: over ~10 realizations the
    # min-max band is extreme-value-dominated, so reviewers need a
    # location-scale and a rank-based spread summary next to it.
    sd = (sum((v - mean) ** 2 for v in sample) / (n - 1)) ** 0.5 if n > 1 else 0.0

    def _quantile(q: float) -> float:
        if n == 1:
            return sample[0]
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return sample[lo] * (1.0 - frac) + sample[hi] * frac

    return {
        "n": n,
        "mean": mean,
        "median": float(median(sample)),
        "sd": sd,
        "q25": _quantile(0.25),
        "q75": _quantile(0.75),
        "low": sample[0],
        "high": sample[-1],
    }


def summarize_variance_bands(rows: list[dict[str, float]]) -> dict[str, object]:
    keys = sorted({key for row in rows for key in row})
    return {
        "schema": "active-fractal-rabbit-variance-bands-v1",
        "n_realizations": len(rows),
        "realizations": rows,
        "metrics": {
            key: empirical_interval(row[key] for row in rows if key in row)
            for key in keys
        },
        "caveat": (
            "Empirical bands summarize repeated synthetic-generator realizations. "
            "They calibrate single-draw robustness, not real-world population uncertainty."
        ),
    }


def candidate_commits_from_repo(
    repo: Path, current_commit: str, limit: int
) -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "rev-list",
            "--first-parent",
            "--max-count",
            str(limit + 1),
            current_commit,
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    commits = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return [commit for commit in commits if commit != current_commit][:limit]


def run_and_write_upstream_sensitivity(
    out_path,
    *,
    current_commit: str,
    candidate_limit: int,
    candidate_provider: Callable[[int], list[str]] | None = None,
    runner: Callable[[str], dict[str, float] | None] | None = None,
):
    """Try prior upstream commits and write success or honest blocked status."""

    provider = candidate_provider or (lambda limit: [])
    run_candidate = runner or (lambda sha: None)
    attempted = provider(candidate_limit)
    selected = None
    metrics = None
    attempt_errors: dict[str, str] = {}
    for sha in attempted:
        try:
            result = run_candidate(sha)
        except (
            Exception
        ) as exc:  # pragma: no cover - exact external failures are host-specific
            attempt_errors[sha] = str(exc)
            continue
        if result is not None:
            selected = sha
            metrics = result
            break
        attempt_errors[sha] = "candidate did not produce comparison metrics"
    payload = {
        "schema": "active-fractal-rabbit-upstream-sensitivity-v1",
        "current_commit": current_commit,
        "attempted_commits": attempted,
        "selected_commit": selected,
        "status": "completed" if selected else "blocked",
        "metrics": metrics or {},
        "attempt_errors": attempt_errors,
        "caveat": (
            "A second FractalRabbit revision is compared only when it builds under the current JDK/Maven "
            "toolchain. If no attempted revision builds, this remains an explicit blocked artifact."
        ),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def make_upstream_candidate_runner(config_path, current_metrics_path):
    """Return a runner that builds prior commits and compares them to the current run."""

    from .config import load_config

    base = load_config(config_path)
    current_metrics = _read_json(Path(current_metrics_path))

    def run_candidate(sha: str) -> dict[str, float]:
        from .pipeline import run_pipeline

        sha12 = sha[:12]
        run_id = f"upstream_sensitivity_{sha12}"
        candidate_fractalrabbit = replace(
            base.fractalrabbit,
            upstream_commit=sha,
            output_prefix=run_id,
        )
        candidate_config = replace(
            base,
            run_id=run_id,
            output_dir=base.project_root / "output" / "runs" / run_id,
            fractalrabbit=candidate_fractalrabbit,
        )
        result = run_pipeline(candidate_config, config_path)
        return compare_upstream_metrics(current_metrics, result.metrics)

    return run_candidate


def compare_upstream_metrics(
    current_metrics: dict[str, object], candidate_metrics: dict[str, object]
) -> dict[str, float]:
    """Numeric deltas used by the upstream-sensitivity artifact."""

    return {
        "predictive_loss_abs_delta": _abs_delta(
            current_metrics, candidate_metrics, "mean_predictive_loss"
        ),
        "posterior_entropy_abs_delta": _abs_delta(
            current_metrics, candidate_metrics, "mean_posterior_entropy"
        ),
        "reference_recovery_mass_abs_delta": _abs_delta(
            current_metrics,
            candidate_metrics,
            "state_recovery_reference_mean_mass_on_true_cell",
        ),
        "latent_state_count_abs_delta": _abs_delta(
            current_metrics, candidate_metrics, "n_latent_states"
        ),
    }


def write_variance_bands(out_path, rows: list[dict[str, float]]):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summarize_variance_bands(rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def run_and_write_variance_bands(config_path, out_path, *, n_realizations: int):
    """Run repeated synthetic realizations and summarize key verdict metrics.

    The repeated runs use the same simulator/configuration but force the model
    engine to ``reference`` to keep this robustness pass focused on generator
    variance rather than pymdp runtime.
    """

    from .anomaly import run_anomaly_detection
    from .anticipatory import run_anticipatory_analysis
    from .config import load_config
    from .fractalrabbit import parse_waypoint_csv
    from .frontier import recoverability_frontier
    from .mission_collection import run_mission_collection
    from .observations import discretize_observations
    from .pipeline import run_pipeline
    from .pomdp import build_pomdp
    from .structure_recovery import run_structure_recovery

    base = load_config(config_path)
    rows: list[dict[str, float]] = []
    for index in range(n_realizations):
        run_id = f"variance_realization_{index + 1}"
        model = replace(base.inference.model, engine="reference")
        inference = replace(base.inference, model=model)
        cfg = replace(
            base,
            run_id=run_id,
            output_dir=base.project_root / "output" / "runs" / run_id,
            inference=inference,
            fractalrabbit=replace(base.fractalrabbit, output_prefix=run_id),
        )
        result = run_pipeline(cfg, config_path)
        sequence = discretize_observations(
            parse_waypoint_csv(result.observations_path), cfg.inference
        )
        pomdp = build_pomdp(sequence, cfg.inference.smoothing)
        anticipatory = run_anticipatory_analysis(
            sequence,
            pomdp,
            smoothing=cfg.inference.smoothing,
            gap_fraction=0.5,
            seed=cfg.seed + index,
        )
        frontier = {
            row["modalities"]: row
            for row in recoverability_frontier(
                sequence, smoothing=cfg.inference.smoothing
            )
        }
        structure = run_structure_recovery(sequence, smoothing=cfg.inference.smoothing)
        anomaly = run_anomaly_detection(
            sequence, smoothing=cfg.inference.smoothing, seed=index, mode="random"
        )
        mission_cells = set(range(min(2, sequence.num_spatial_states)))
        mission = run_mission_collection(
            sequence,
            smoothing=cfg.inference.smoothing,
            mission_cells=mission_cells,
            resolved_dynamics=True,
        )
        rows.append(
            {
                "gap_filling_skill": anticipatory.scrp_skill_gap,
                "blind_limit_skill": anticipatory.blind_persistence_skill_scrp,
                "frontier_no_location_mass": float(
                    frontier["no_location"]["mean_mass_on_true_cell"]
                ),
                "structure_deep_latent_tv": structure.deep_latent_tv,
                "anomaly_random_auc": anomaly.vfe_auc,
                "mission_efe_minus_pragmatic": mission.efe_recovery
                - mission.pragmatic_recovery,
            }
        )
    return write_variance_bands(out_path, rows)


def _abs_delta(left: dict[str, object], right: dict[str, object], key: str) -> float:
    return abs(float(right[key]) - float(left[key]))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
