"""Closed-loop experiment runner."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .config import load_config, with_candidate
from .pipeline import run_pipeline
from .types import PipelineResult, ProjectConfig


def run_closed_loop(config_path: str | Path) -> list[PipelineResult]:
    """Run configured iterations and select candidate parameters from metrics."""

    base = load_config(config_path)
    results: list[PipelineResult] = []
    current: ProjectConfig = base
    for iteration in range(base.closed_loop.iterations):
        current = replace(
            current,
            run_id=f"{base.run_id}_iter_{iteration + 1}",
            output_dir=base.project_root
            / "output"
            / "runs"
            / f"{base.run_id}_iter_{iteration + 1}",
        )
        result = run_pipeline(current, config_path)
        results.append(result)
        if result.selected_next_candidate is None:
            break
        candidate = next(
            item
            for item in base.closed_loop.candidates
            if item.name == result.selected_next_candidate
        )
        current = with_candidate(base, candidate, f"{base.run_id}_{candidate.name}")
    _write_summary(
        base.project_root / "output" / "runs" / f"{base.run_id}_closed_loop.json",
        results,
    )
    return results


def _write_summary(path: Path, results: list[PipelineResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "run_id": item.run_id,
            "output_dir": str(item.output_dir),
            "selected_next_candidate": item.selected_next_candidate,
            "candidate_scores": [
                {
                    "rank": score.rank,
                    "name": score.name,
                    "objective": score.objective,
                    "metric_value": score.metric_value,
                    "reporting_ratio": score.reporting_ratio,
                    "score": score.score,
                }
                for score in item.candidate_scores
            ],
            "metrics": item.metrics,
        }
        for item in results
    ]
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
