"""Synthetic task-utility/privacy gate assembled from existing artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .privacy_utility_frontier import REID_RISK_CEILING, UTILITY_FLOOR_BITS

SYNTHETIC_TASK_UTILITY_PRIVACY_SCHEMA = (
    "active-fractal-rabbit-synthetic-task-utility-privacy-v1"
)


def synthetic_task_utility_privacy(project_root: str | Path) -> dict[str, Any]:
    """Score local synthetic task utility against privacy ceilings."""

    root = Path(project_root).resolve()
    runs = root / "output" / "runs"
    privacy_utility = _read_json(runs / "privacy_utility_frontier.json")
    recoverability = _read_json(runs / "frontier_analysis.json")
    linkage = _read_json(runs / "linkage_risk_frontier.json")
    privacy_frontier = _read_json(runs / "privacy_frontier_analysis.json")

    utility_rows = [
        row
        for row in privacy_utility.get("frontier", [])
        if isinstance(row, dict) and int(row.get("spatial_bins", 0)) > 0
    ]
    finest = max(
        (_float(row.get("excess_utility_bits")) for row in utility_rows), default=0.0
    )
    recovery_by_modality = {
        str(row.get("modalities")): row
        for row in recoverability.get("frontier", [])
        if isinstance(row, dict)
    }
    # privacy_frontier rows are keyed by modality name ("all", "no_location",
    # ...), not "regime" — keying by the absent "regime" collapsed every row
    # under "None" and silently zeroed the unicity lookups below.
    privacy_by_regime = {
        str(row.get("modalities")): row
        for row in privacy_frontier.get("frontier", [])
        if isinstance(row, dict)
    }
    no_location_recovery = _float(
        recovery_by_modality.get("no_location", {}).get("mean_mass_on_true_cell")
    )
    full_location_recovery = _float(
        recovery_by_modality.get("all", {}).get("mean_mass_on_true_cell")
    )
    linkage_minimization_protects = bool(linkage.get("minimization_protects"))

    rows: list[dict[str, Any]] = []
    for row in utility_rows:
        regime = str(row.get("regime", "unknown"))
        excess = _float(row.get("excess_utility_bits"))
        risk = _float(row.get("reid_event_uniqueness"))
        utility_floor = excess >= UTILITY_FLOOR_BITS
        risk_ceiling = risk < REID_RISK_CEILING
        rows.append(
            {
                "regime": regime,
                "spatial_bins": int(row.get("spatial_bins", 0)),
                "predictive_utility_bits": excess,
                "utility_fraction_of_finest": round(excess / finest, 6)
                if finest > 0.0
                else 0.0,
                "reid_event_uniqueness": risk,
                "co_travel_coincidence": _float(row.get("co_travel_coincidence")),
                "shuffle_floor_sd": _float(row.get("shuffle_floor_sd")),
                "passes_utility_floor": utility_floor,
                "passes_reid_ceiling": risk_ceiling,
                "release_utility_claim_allowed": bool(utility_floor and risk_ceiling),
            }
        )

    allowed = [row for row in rows if row["release_utility_claim_allowed"]]
    best_minimized = min(
        allowed, key=lambda row: int(row["spatial_bins"]), default=None
    )
    status = (
        "release_utility_operating_point_found"
        if best_minimized
        else "no_release_utility_operating_point"
    )
    return {
        "schema": SYNTHETIC_TASK_UTILITY_PRIVACY_SCHEMA,
        "status": status,
        "utility_floor_bits": UTILITY_FLOOR_BITS,
        "reid_risk_ceiling": REID_RISK_CEILING,
        "any_release_utility_claim_allowed": bool(best_minimized),
        "joint_operating_point_count": len(allowed),
        "best_minimized_operating_regime": best_minimized["regime"]
        if best_minimized
        else "none",
        "regime_count": len(rows),
        "rows": rows,
        "task_context": {
            "full_location_recovery_mass": full_location_recovery,
            "no_location_recovery_mass": no_location_recovery,
            "privacy_frontier_all_unicity": _float(
                privacy_by_regime.get("all", {}).get("unicity")
            ),
            "privacy_frontier_no_location_unicity": _float(
                privacy_by_regime.get("no_location", {}).get("unicity")
            ),
            "linkage_minimization_protects": linkage_minimization_protects,
            "linkage_status": str(linkage.get("status", "unavailable")),
        },
        "claim_boundary": (
            "Synthetic downstream task-utility screen assembled from generated local artifacts. "
            "A task/release-utility claim is denied unless utility floors and privacy ceilings "
            "clear together; no empirical, operational, or real-world utility claim follows."
        ),
    }


def run_and_write_synthetic_task_utility_privacy(
    project_root: str | Path,
    out_path: str | Path | None = None,
) -> Path:
    """Write ``synthetic_task_utility_privacy.json``."""

    root = Path(project_root).resolve()
    out = (
        Path(out_path)
        if out_path
        else root / "output" / "runs" / "synthetic_task_utility_privacy.json"
    )
    payload = synthetic_task_utility_privacy(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _float(value: object) -> float:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0
