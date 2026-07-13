"""Repeated-draw audit for the privacy-utility frontier.

The canonical privacy-utility frontier is a single generated synthetic fixture
measurement. This module adds a narrow repeat surface: it recomputes the same
frontier for the canonical fixture and for variance-realization observation CSVs
when they exist. Missing realization files are reported as unavailable rows, not
silently skipped or imputed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .privacy_utility_frontier import (
    DEFAULT_GRID_REGIMES,
    DEFAULT_SHUFFLE_SEED,
    DEFAULT_SHUFFLE_TRIALS,
    privacy_utility_frontier,
)

PRIVACY_UTILITY_REPEATS_SCHEMA = "active-fractal-rabbit-privacy-utility-repeats-v1"
DEFAULT_REQUIRED_REALIZATIONS = 10


def privacy_utility_frontier_repeats(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
    realization_count: int = DEFAULT_REQUIRED_REALIZATIONS,
    grid_regimes: tuple[int, ...] = DEFAULT_GRID_REGIMES,
    n_shuffle: int = DEFAULT_SHUFFLE_TRIALS,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
) -> dict[str, Any]:
    """Return a repeated-draw audit payload for the privacy-utility frontier."""

    cfg_path = Path(config_path)
    root = (
        Path(project_root).resolve()
        if project_root
        else _project_root_for_config(cfg_path)
    )
    cfg_abs = root / cfg_path if not cfg_path.is_absolute() else cfg_path
    cfg = load_config(cfg_abs)
    if cfg.fractalrabbit.fixture_csv is None:
        raise ValueError("privacy utility repeats require a configured fixture_csv")

    canonical_path = _resolve(root, cfg.fractalrabbit.fixture_csv)
    rows: list[dict[str, Any]] = [
        _completed_row(
            "canonical_fixture",
            canonical_path,
            config_path=cfg_abs,
            grid_regimes=grid_regimes,
            n_shuffle=n_shuffle,
            shuffle_seed=shuffle_seed,
        )
    ]

    for idx in range(1, int(realization_count) + 1):
        observations = (
            root
            / "output"
            / "runs"
            / f"variance_realization_{idx}"
            / "observations.csv"
        )
        if observations.is_file():
            rows.append(
                _completed_row(
                    f"variance_realization_{idx}",
                    observations,
                    config_path=cfg_abs,
                    grid_regimes=grid_regimes,
                    n_shuffle=n_shuffle,
                    shuffle_seed=shuffle_seed + idx,
                )
            )
        else:
            rows.append(
                {
                    "realization_id": f"variance_realization_{idx}",
                    "status": "unavailable",
                    "source": str(observations.relative_to(root)),
                    "reason": "missing variance realization observations.csv",
                }
            )

    completed = [row for row in rows if row.get("status") == "completed"]
    unavailable = [row for row in rows if row.get("status") != "completed"]

    # Independence gate: bit-identical realizations are determinism reruns of the
    # same draw, not independent repeats. Content-hash every completed source and
    # flag duplicates of the canonical fixture so a green repeat claim cannot rest
    # on copies. Without >= 2 genuinely distinct draws the repeat lane can only
    # attest determinism, never repeated-draw stability.
    canonical_hash = completed[0].get("content_md5") if completed else None
    for row in completed:
        row["duplicate_of_canonical"] = bool(
            row["realization_id"] != "canonical_fixture"
            and canonical_hash is not None
            and row.get("content_md5") == canonical_hash
        )
    distinct_draws = len(
        {row.get("content_md5") for row in completed if row.get("content_md5")}
    )
    duplicate_of_canonical_count = sum(
        1 for row in completed if row.get("duplicate_of_canonical")
    )
    independent_repeats = distinct_draws > 1

    graceful_count = sum(
        1 for row in completed if row.get("graceful_degradation_tradeoff") is True
    )
    min_collection_count = sum(
        1 for row in completed if row.get("min_collection_found") is True
    )
    all_required_completed = len(completed) == len(rows)
    all_completed_graceful = bool(completed) and graceful_count == len(completed)

    if not all_required_completed:
        status = "partial_repeats_unavailable"
    elif not independent_repeats:
        status = "complete_duplicate_reruns_only"
    else:
        status = "complete"

    return {
        "schema": PRIVACY_UTILITY_REPEATS_SCHEMA,
        "status": status,
        "config": str(cfg_path),
        "canonical_source": str(
            canonical_path.relative_to(root)
            if canonical_path.is_relative_to(root)
            else canonical_path
        ),
        "required_realization_count": int(realization_count),
        "completed_count": len(completed),
        "unavailable_count": len(unavailable),
        "all_required_completed": all_required_completed,
        "distinct_draws": distinct_draws,
        "duplicate_of_canonical_count": duplicate_of_canonical_count,
        "independent_repeats": independent_repeats,
        "n_graceful_degradation": graceful_count,
        "n_min_collection_found": min_collection_count,
        "can_claim_graceful_repeated_tradeoff": bool(
            all_required_completed and all_completed_graceful and independent_repeats
        ),
        "can_claim_min_collection_replicates": bool(
            all_required_completed
            and min_collection_count == len(completed)
            and independent_repeats
        ),
        "ranges": _ranges(completed),
        "rows": rows,
        "boundary": (
            "Synthetic repeated-draw audit over generated fixture/variance observations only. "
            "Unavailable rows block repeat-stability wording; they are not failures, successes, "
            "or empirical privacy evidence. Realizations byte-identical to the canonical fixture "
            "are determinism reruns, not independent draws: they are flagged duplicate_of_canonical "
            "and do not count toward independent repeats, so a repeated-draw claim requires "
            "distinct_draws > 1."
        ),
    }


def run_and_write_privacy_utility_repeats(
    config_path: str | Path,
    out_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Write ``privacy_utility_frontier_repeats.json``."""

    payload = privacy_utility_frontier_repeats(config_path, project_root=project_root)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def _completed_row(
    realization_id: str,
    observations_path: Path,
    *,
    config_path: Path,
    grid_regimes: tuple[int, ...],
    n_shuffle: int,
    shuffle_seed: int,
) -> dict[str, Any]:
    payload = privacy_utility_frontier(
        config_path,
        observations_path,
        grid_regimes=grid_regimes,
        n_shuffle=n_shuffle,
        shuffle_seed=shuffle_seed,
    )
    return {
        "realization_id": realization_id,
        "status": "completed",
        "source": str(observations_path),
        "content_md5": _content_md5(observations_path),
        "graceful_degradation_tradeoff": bool(
            payload.get("graceful_degradation_tradeoff")
        ),
        "utility_monotone_decreasing": bool(payload.get("utility_monotone_decreasing")),
        "risk_monotone_decreasing": bool(payload.get("risk_monotone_decreasing")),
        "min_collection_found": bool(
            (payload.get("min_collection_operating_point") or {}).get("found")
            if isinstance(payload.get("min_collection_operating_point"), dict)
            else False
        ),
        "finest_excess_utility_bits": _float(payload.get("finest_excess_utility_bits")),
        "coarsest_excess_utility_bits": _float(
            payload.get("coarsest_excess_utility_bits")
        ),
        "finest_reid_event_uniqueness": _float(
            payload.get("finest_reid_event_uniqueness")
        ),
        "coarsest_reid_event_uniqueness": _float(
            payload.get("coarsest_reid_event_uniqueness")
        ),
        "max_shuffle_floor_sd": _float(payload.get("max_shuffle_floor_sd")),
        "min_reid_among_useful_grids": _float(
            (payload.get("min_collection_operating_point") or {}).get(
                "min_reid_among_useful_grids"
            )
            if isinstance(payload.get("min_collection_operating_point"), dict)
            else 0.0
        ),
    }


def _ranges(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    fields = (
        "finest_excess_utility_bits",
        "coarsest_excess_utility_bits",
        "finest_reid_event_uniqueness",
        "coarsest_reid_event_uniqueness",
        "max_shuffle_floor_sd",
    )
    ranges: dict[str, dict[str, float]] = {}
    for field in fields:
        values = [_float(row.get(field)) for row in rows]
        ranges[field] = {
            "min": round(min(values), 6) if values else 0.0,
            "max": round(max(values), 6) if values else 0.0,
        }
    return ranges


def _content_md5(path: Path) -> str:
    """Return the md5 of the observations file bytes (draw-identity fingerprint)."""

    digest = hashlib.md5()  # nosec B324 - non-security content-identity fingerprint
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value: object) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _project_root_for_config(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return (
        resolved.parent.parent if resolved.parent.name == "configs" else resolved.parent
    )
