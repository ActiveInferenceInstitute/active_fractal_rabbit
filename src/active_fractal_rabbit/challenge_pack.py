"""Open synthetic challenge-pack manifest.

The challenge pack is a machine-readable handoff for the benchmark discipline:
locked splits, scored rows, claim-gate status, visual/provenance audits, current
baseline-family candidates, and the non-operational usage boundary. It is not a
data release and it does not make unavailable foundation models look scored.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .release import sha256_file as _sha256
from .sota_benchmark import REQUIRED_BASELINE_FAMILIES
from .sota_program_harness import (
    SOTA_PROGRAM_LEADERBOARD_HEADER,
    canonical_run_parameter_violations,
)

CHALLENGE_PACK_SCHEMA = "active-fractal-rabbit-open-challenge-pack-v3"
LITERATURE_REFRESH_DATE = "2026-06-29"

NON_OPERATIONAL_USAGE_STATEMENT = (
    "Synthetic benchmark package only. It must not be used for operational "
    "tasking, surveillance readiness, real-person mobility inference, evasion, "
    "or deployment claims. Empirical transfer remains blocked until the separate "
    "consent, provenance, minimization, and privacy-review protocol is satisfied."
)

FOUNDATION_BASELINE_REFRESH: tuple[dict[str, str], ...] = (
    {
        "family": "deep_learning_survey",
        "label": "Survey on deep learning for human mobility",
        "source": "https://doi.org/10.1145/3485125",
        "contract_status": "context_only_not_a_scored_baseline",
    },
    {
        "family": "foundation_mobility",
        "label": "Pretrained Mobility Transformer (PMT)",
        "source": "https://arxiv.org/abs/2406.02578",
        "contract_status": "blocked_pending_license_compute_data_and_adapter_audit",
    },
    {
        "family": "foundation_mobility",
        "label": "MoveGPT spatially aware mixture-of-experts",
        "source": "https://arxiv.org/abs/2505.18670",
        "contract_status": "blocked_pending_license_compute_data_and_adapter_audit",
    },
    {
        "family": "llm_mobility",
        "label": "Llama3-8B-Mob instruction-tuned mobility model",
        "source": "https://arxiv.org/abs/2410.23692",
        "contract_status": "blocked_pending_license_compute_data_and_adapter_audit",
    },
    {
        "family": "llm_mobility",
        "label": "MoBLLM open-source LLM mobility predictor",
        "source": "https://doi.org/10.1016/j.trc.2026.105562",
        "contract_status": "blocked_pending_license_compute_data_and_adapter_audit",
    },
    {
        "family": "llm_mobility",
        "label": "NextLocLLM coordinate-regression next-location predictor",
        "source": "https://arxiv.org/abs/2410.09129",
        "contract_status": "blocked_pending_license_compute_data_and_adapter_audit",
    },
    {
        "family": "llm_mobility",
        "label": "AgentMob LLM-driven evidence-gathering mobility predictor",
        "source": "https://arxiv.org/abs/2606.05130",
        "contract_status": "blocked_pending_license_compute_data_and_adapter_audit",
    },
    {
        "family": "benchmark_protocol",
        "label": "Trajectory test-train overlap in next-location prediction datasets",
        "source": "https://doi.org/10.1007/s10994-023-06386-x",
        "contract_status": "context_only_for_locked_split_and_overlap_hygiene",
    },
    {
        "family": "synthetic_mobility_evaluation",
        "label": "Higher-order network analysis of observed versus simulated mobility",
        "source": "https://arxiv.org/abs/2606.00733",
        "contract_status": "context_only_for_synthetic_validity_boundary",
    },
    {
        "family": "synthetic_mobility_evaluation",
        "label": "Representativeness-privacy-utility evaluation for synthetic trip data",
        "source": "https://arxiv.org/abs/2510.24375",
        "contract_status": "context_only_for_task_utility_privacy_boundary",
    },
    {
        "family": "privacy_synthetic_trajectory",
        "label": "SoK on privacy and utility in trajectory generation",
        "source": "https://doi.org/10.56553/popets-2024-0068",
        "contract_status": "context_only_for_privacy_boundary_no_scored_baseline",
    },
    {
        "family": "privacy_synthetic_trajectory",
        "label": "Re-identification stress test for anonymized real-user trajectories",
        "source": "https://arxiv.org/abs/2506.05611",
        "contract_status": "context_only_for_privacy_boundary_no_scored_baseline",
    },
    {
        "family": "neural_sequence",
        "label": "MoBERT / SIGSPATIAL GISCUP-style sparse challenge model",
        "source": "https://www.bgd.ed.tum.de/pdf/2025_short_SIGSPATIAL_giscup.pdf",
        "contract_status": "candidate_family_only_until_locked_split_adapter_exists",
    },
)


def build_open_challenge_pack(
    project_root: str | Path, out_path: str | Path | None = None
) -> Path:
    """Write the open synthetic challenge-pack JSON artifact."""

    root = Path(project_root).resolve()
    runs = root / "output" / "runs"
    output = (
        Path(out_path) if out_path is not None else runs / "open_challenge_pack.json"
    )
    _require_file(runs / "sota_task_contract.json")
    _require_file(runs / "sota_leaderboard.csv")
    _require_file(runs / "sota_program_leaderboard.csv")
    _require_file(runs / "sota_program_robustness_grid.csv")
    _require_file(runs / "aif_regime_search.csv")
    benchmark = _read_required_json(runs / "sota_benchmark_matrix.json")
    gate = _read_required_json(runs / "sota_claim_gate.json")
    sota_program = _read_required_json(runs / "sota_program_gate.json")
    robustness_diagnostics = _read_required_json(
        runs / "sota_program_robustness_grid.json"
    )
    aif_regime_search = _read_required_json(runs / "aif_regime_search.json")
    aif_regime_gate = _read_required_json(runs / "aif_regime_sota_gate.json")
    sota_program_run_parameters = _canonical_sota_program_parameters(
        sota_program, runs / "sota_program_gate.json"
    )
    visual_quality = _read_required_json(runs / "visual_quality_audit.json")
    visual_claim = _read_required_json(runs / "visual_claim_audit.json")
    fair_external = _read_required_json(
        runs / "fair_inference_external_realizations.json"
    )
    provenance = _read_required_json(
        root / "output" / "data" / "manuscript_variable_provenance.json"
    )
    leaderboard_rows = _leaderboard_rows(runs / "sota_leaderboard.csv", benchmark)
    sota_program_rows = _leaderboard_rows(runs / "sota_program_leaderboard.csv", {})
    robustness_rows = _leaderboard_rows(runs / "sota_program_robustness_grid.csv", {})
    aif_regime_rows = _leaderboard_rows(runs / "aif_regime_search.csv", {})
    candidate_table_completeness = _program_candidate_table_status(sota_program_rows)
    robustness_table_completeness = _program_candidate_table_status(robustness_rows)

    payload = {
        "schema": CHALLENGE_PACK_SCHEMA,
        "package": "active_fractal_rabbit_synthetic_challenge",
        "generated_from": "local regenerated artifacts",
        "scope": "synthetic-only sparse-waypoint benchmark on pinned FractalRabbit and companion synthetic generators",
        "non_operational_usage_statement": NON_OPERATIONAL_USAGE_STATEMENT,
        "locked_splits": {
            "sota_benchmark": benchmark.get("locked_split", {}),
            "sota_program_run_parameters": sota_program_run_parameters,
        },
        "source_artifacts": [
            _artifact(root, "output/runs/sota_task_contract.json"),
            _artifact(root, "output/runs/sota_benchmark_matrix.json"),
            _artifact(root, "output/runs/sota_leaderboard.csv"),
            _artifact(root, "output/runs/sota_claim_gate.json"),
            _artifact(root, "output/runs/sota_program_gate.json"),
            _artifact(root, "output/runs/sota_program_leaderboard.csv"),
            _artifact(root, "output/runs/sota_program_robustness_grid.json"),
            _artifact(root, "output/runs/sota_program_robustness_grid.csv"),
            _artifact(root, "output/runs/aif_regime_search.csv"),
            _artifact(root, "output/runs/aif_regime_search.json"),
            _artifact(root, "output/runs/aif_regime_sota_gate.json"),
            _artifact(root, "output/runs/fair_inference_external_realizations.json"),
            _artifact(root, "output/runs/visual_quality_audit.json"),
            _artifact(root, "output/runs/visual_claim_audit.json"),
            _artifact(root, "output/data/manuscript_variable_provenance.json"),
        ],
        "baseline_family_contracts": _baseline_family_contracts(leaderboard_rows, gate),
        "leaderboard_rows": leaderboard_rows,
        "sota_program_leaderboard_rows": sota_program_rows,
        "sota_program_candidate_table_completeness": candidate_table_completeness,
        "sota_program_robustness_grid_rows": robustness_rows,
        "sota_program_robustness_grid_completeness": robustness_table_completeness,
        "aif_regime_search_rows": aif_regime_rows,
        "aif_regime_search": {
            "schema": aif_regime_search.get("schema", "unavailable"),
            "status": aif_regime_search.get("status", "unavailable"),
            "gate_clears": aif_regime_search.get("gate_clears", False),
            "cleared_regime_count": aif_regime_search.get("cleared_regime_count", 0),
            "best_cleared_regime": aif_regime_search.get("best_cleared_regime"),
            "global_sota_boundary": aif_regime_search.get("global_sota_boundary", ""),
            "controls": aif_regime_search.get("controls", {}),
            "regime_summaries": aif_regime_search.get("regime_summaries", {}),
        },
        "aif_regime_sota_gate": {
            "schema": aif_regime_gate.get("schema", "unavailable"),
            "status": aif_regime_gate.get("status", "unavailable"),
            "gate_clears": aif_regime_gate.get("gate_clears", False),
            "cleared_regime_count": aif_regime_gate.get("cleared_regime_count", 0),
            "boundary": aif_regime_gate.get("global_sota_boundary", ""),
        },
        "sota_program_robustness_diagnostics": {
            "schema": robustness_diagnostics.get("schema", "unavailable"),
            "status": robustness_diagnostics.get("status", "unavailable"),
            "all_diagnostic_lanes_scored": robustness_diagnostics.get(
                "all_diagnostic_lanes_scored", False
            ),
            "lane_statuses": robustness_diagnostics.get("lane_statuses", []),
            "boundary": robustness_diagnostics.get("boundary", ""),
        },
        "sota_program_readiness": sota_program.get("readiness", {}),
        "sota_program_lane_definitions": sota_program.get("lane_definitions", {}),
        "sota_program_robustness_summary": sota_program.get("robustness_summary", {}),
        "fair_external_realizations": {
            "schema": fair_external.get("schema", "unavailable"),
            "status": fair_external.get("status", "unavailable"),
            "completed_count": fair_external.get("completed_count", 0),
            "all_required_completed": fair_external.get(
                "all_required_completed", False
            ),
            "can_promote_beyond_fixture_language": fair_external.get(
                "can_promote_beyond_fixture_language", False
            ),
            "boundary": fair_external.get("boundary", ""),
        },
        "candidate_row_requirements": [
            "prequential next-cell predictive loss on the locked split",
            "traveller-clustered loss uncertainty interval or explicit unavailable reason",
            "Brier score and expected calibration error",
            "runtime and dependency footprint",
            "lane coverage and locked-split eligibility",
            "failure modes, minimization behavior, and integrity/poisoning diagnostics",
        ],
        "claim_gate": {
            "status": gate.get("status", benchmark.get("status", "unavailable")),
            "reasons": gate.get(
                "claim_gate_reasons", benchmark.get("claim_gate_reasons", [])
            ),
            "missing_baseline_families": gate.get(
                "missing_baseline_families",
                benchmark.get("missing_baseline_families", []),
            ),
            "false_claim_oracle": gate.get(
                "false_claim_oracle", benchmark.get("false_claim_oracle", {})
            ),
        },
        "visual_audits": {
            "quality_status": visual_quality.get("status", "unavailable"),
            "visual_claim_status": visual_claim.get("status", "unavailable"),
            "figure_count": visual_quality.get("figure_count"),
        },
        "manuscript_variable_provenance": {
            "status": provenance.get("status", "unavailable"),
            "n_tokens": provenance.get(
                "n_tokens", provenance.get("token_count", "unavailable")
            ),
        },
        "literature_refresh": {
            "date_checked": LITERATURE_REFRESH_DATE,
            "sources": list(FOUNDATION_BASELINE_REFRESH),
            "foundation_style_baselines_entered": False,
            "reason": (
                "current foundation and LLM mobility sources are candidate families only; "
                "no row is admitted until license, compute, data provenance, and locked-split adapter "
                "contracts are auditable inside this project"
            ),
        },
        "empirical_transfer": {
            "status": "blocked",
            "reason": "consent/provenance/privacy/minimization protocol has not been satisfied",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def _baseline_family_contracts(
    rows: list[dict[str, object]], gate: dict[str, object]
) -> list[dict[str, object]]:
    missing = set(
        gate.get("missing_baseline_families", []) if isinstance(gate, dict) else []
    )
    contracts: list[dict[str, object]] = []
    for family, description in REQUIRED_BASELINE_FAMILIES.items():
        family_rows = [row for row in rows if row.get("family") == family]
        locked_rows = [
            row
            for row in family_rows
            if row.get("locked_split_scored") in {True, "true", "True"}
        ]
        status = "locked_split_scored" if locked_rows else "blocked"
        if family in missing:
            status = "blocked_missing_locked_split_rows"
        if family in {"foundation_mobility", "llm_mobility"} and not locked_rows:
            status = "blocked_pending_license_compute_data_and_adapter_audit"
        contracts.append(
            {
                "family": family,
                "description": description,
                "status": status,
                "n_rows": len(family_rows),
                "n_locked_split_rows": len(locked_rows),
            }
        )
    return contracts


def _program_candidate_table_status(rows: list[dict[str, object]]) -> dict[str, object]:
    required = set(SOTA_PROGRAM_LEADERBOARD_HEADER)
    missing_columns: list[str] = []
    incomplete_rows: list[dict[str, object]] = []
    if not rows:
        missing_columns = sorted(required)
    else:
        present = set(rows[0])
        missing_columns = sorted(required - present)
        non_empty_fields = {
            "candidate",
            "lane",
            "role",
            "mean_loss",
            "n_scored_steps",
            "family",
            "locked_split_scored",
            "leakage",
            "loss_ci_low",
            "loss_ci_high",
            "brier_score",
            "ece",
            "top1_accuracy",
            "runtime_seconds",
            "lane_coverage",
            "dependency_footprint",
            "missingness_robustness_status",
            "failure_modes",
        }
        for idx, row in enumerate(rows):
            empty = sorted(
                field for field in non_empty_fields if row.get(field) in {"", None}
            )
            if empty:
                incomplete_rows.append(
                    {
                        "row_index": idx,
                        "candidate": row.get("candidate", ""),
                        "lane": row.get("lane", ""),
                        "empty_fields": empty,
                    }
                )
    return {
        "status": "pass"
        if not missing_columns and not incomplete_rows
        else "incomplete",
        "required_columns": list(SOTA_PROGRAM_LEADERBOARD_HEADER),
        "missing_columns": missing_columns,
        "incomplete_rows": incomplete_rows,
        "n_rows": len(rows),
        "boundary": (
            "missingness_robustness_status is a readiness field. Values must either describe "
            "a scored synthetic robustness lane or explicitly state an unscored blocker; neither "
            "form is empirical-transfer or operational evidence."
        ),
    }


def _canonical_sota_program_parameters(
    sota_program: dict[str, object], path: Path
) -> dict[str, object]:
    if not sota_program:
        raise ValueError(f"{path} is required before building the open challenge pack")
    run_parameters = sota_program.get("run_parameters")
    violations = canonical_run_parameter_violations(
        run_parameters if isinstance(run_parameters, dict) else None
    )
    if violations:
        raise ValueError(
            f"{path} is not a canonical SOTA-program artifact ({'; '.join(violations)}); rerun "
            "scripts/18_run_sota_program.py with default parameters or the full study"
        )
    return dict(run_parameters)


def _leaderboard_rows(
    path: Path, benchmark: dict[str, object]
) -> list[dict[str, object]]:
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            return [_coerce_row(row) for row in csv.DictReader(handle)]
    rows = benchmark.get("leaderboard", []) if isinstance(benchmark, dict) else []
    if not isinstance(rows, list):
        return []
    return [_coerce_row(row) for row in rows if isinstance(row, dict)]


def _coerce_row(row: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in row.items():
        if key in {"locked_split_scored", "leakage"}:
            result[key] = (
                value if isinstance(value, bool) else str(value).lower() == "true"
            )
        elif key in {
            "mean_predictive_loss",
            "mean_loss",
            "loss_ci_low",
            "loss_ci_high",
            "clustered_ci_low",
            "clustered_ci_high",
            "brier_score",
            "ece",
            "top1_accuracy",
            "runtime_seconds",
        }:
            result[key] = _float_or_none(value)
        else:
            result[key] = value
    return result


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _artifact(root: Path, rel_path: str) -> dict[str, object]:
    path = root / rel_path
    payload: dict[str, object] = {"path": rel_path, "exists": path.is_file()}
    if path.is_file():
        payload.update({"bytes": path.stat().st_size, "sha256": _sha256(path)})
    return payload


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"required challenge-pack source missing: {path}")


def _read_required_json(path: Path) -> dict[str, object]:
    _require_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"required challenge-pack source is invalid JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"required challenge-pack source is not a JSON object: {path}")
    return payload
