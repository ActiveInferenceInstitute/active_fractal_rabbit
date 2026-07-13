"""Canonical SOTA program harness: score every candidate, gate the AIF claim honestly.

This module wires the candidate roster (the bundled spine references plus the transparent
baseline ladder, the two classical sequence models, the pure-numpy neural baseline, and the
fitted active-inference candidate) into the leakage-free spine of ``sota_program.py`` across
four structurally different synthetic lanes:

* the **FractalRabbit** partial-observability lane (``build_partial_observability_dataset``),
* the **cross-generator** gravity / explore-and-return lane
  (``build_gravity_partial_observability_dataset``),
* the **reporting-gap** lane with deterministic burst-missingness thinning, and
* the **regime-stressor** lane with controlled return rates, reporting gaps, and
  movement-mode shifts.

For each lane it scores every candidate with the prequential, leakage-free next-latent-cell
loss, identifies the STRONGEST FAIR baseline (the lowest-loss candidate that is NOT the AIF
candidate), and computes the traveller-CLUSTERED bootstrap CI on the loss DIFFERENCE between
the AIF candidate and that strongest fair baseline. A win is recorded only when the AIF
candidate has the lower mean loss AND the clustered CI on the difference separates from zero
(entirely below zero, i.e. ``a_wins_clustered`` with ``a = strongest fair baseline``).

The CLAIM GATE clears ONLY if the AIF candidate beats the strongest fair baseline on EVERY
canonical lane with the clustered CI separated from zero on every lane and no leakage.
Otherwise the status stays uncleared and the residual gap is recorded honestly per lane.

THE HONESTY INVARIANT lives here: the AIF candidate is NEVER given oracle access, future
information, or the test split during fitting (``fit_aif_candidate`` calibrates on a
TRAIN-internal split only); baselines are never dropped or weakened; the oracle is a ceiling,
not a competitor and is excluded from the fair-baseline pool; and the CI treats whole
travellers as the resampling unit (correlated within-traveller steps are NOT treated as
independent). A truthful "still blocked, here is why" is the correct output whenever the
evidence does not clear the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Mapping

import numpy as np

from .config import load_config
from .sota_aif_candidate import fit_aif_candidate
from .sota_baselines import transparent_baseline_ladder
from .sota_classical import classical_candidates
from .sota_crossgen import build_gravity_partial_observability_dataset
from .sota_neural import train_mlp_candidate
from .sota_program import (
    Candidate,
    HardFilterCandidate,
    ObservedMarkovCandidate,
    PartialObservabilityDataset,
    ScoredCandidate,
    SoftFilterCandidate,
    build_partial_observability_dataset,
    emission_matrix,
    paired_loss_difference_ci,
    score_candidate,
)
from .sota_stressors import (
    DIAGNOSTIC_STRESSOR_LANES,
    ADVERSARIAL_REPEAT_SEED,
    REGIME_STRESSOR_SECOND_REPEAT_SEED,
    REPORTING_GAP_REPEAT_SEED,
    all_lane_definitions,
    build_adversarial_missingness_partial_observability_dataset,
    build_regime_stressor_partial_observability_dataset,
    build_regime_stressor_repeat_partial_observability_dataset,
    build_reporting_gap_partial_observability_dataset,
    lane_definitions,
)
from .types import InferenceConfig
from .power_analysis import traveller_clustered_bootstrap_ci

SOTA_PROGRAM_HARNESS_SCHEMA = "active-fractal-rabbit-sota-program-harness-v2"
SOTA_PROGRAM_ROBUSTNESS_DIAGNOSTICS_SCHEMA = (
    "active-fractal-rabbit-sota-robustness-diagnostics-v1"
)

#: The name of the AIF candidate under test (its lower-loss-on-both-lanes win is what the
#: claim gate evaluates). The plain ``soft_filter`` is the AIF *reference* and is excluded
#: from the fair-baseline pool alongside the candidate under test, so the AIF family never
#: competes against itself.
AIF_CANDIDATE_NAME = "precision_return_aif"

#: Candidate names that belong to the active-inference family and are therefore NOT eligible
#: as the "strongest fair baseline" the AIF candidate must beat.
AIF_FAMILY_NAMES = frozenset({AIF_CANDIDATE_NAME, "soft_filter"})

#: A peeking probe must NOT be able to out-score this loss; we assert the AIF candidate's
#: loss stays well above the near-zero oracle floor as a structural leakage sanity check.
LEAKAGE_FLOOR_NATS = 0.5

#: Default lane knobs: high-power traveller count on a 6x6 (= 36 cell) grid.
DEFAULT_N_TRAVELLERS = 64
DEFAULT_EPS = 0.3
DEFAULT_SPATIAL_BINS = 6
DEFAULT_N_BOOT = 2000
DEFAULT_BOOT_SEED = 7
CANONICAL_LANES = ("fractalrabbit", "gravity", "reporting_gap", "regime_stressor")
DEFAULT_DIAGNOSTIC_N_TRAVELLERS = 32
DEFAULT_DIAGNOSTIC_N_BOOT = 400

SOTA_PROGRAM_LEADERBOARD_HEADER = [
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
    "clustered_ci_low",
    "clustered_ci_high",
    "brier_score",
    "ece",
    "top1_accuracy",
    "runtime_seconds",
    "lane_coverage",
    "dependency_footprint",
    "missingness_robustness_status",
    "failure_modes",
]

CANDIDATE_FAMILIES = {
    "hard_filter": "transparent_filter_baseline",
    "observed_markov": "transparent_markov_baseline",
    "soft_filter": "active_inference_reference",
    "persistence": "transparent_null",
    "online_base_rate": "transparent_null",
    "first_order_markov": "transparent_markov_baseline",
    "higher_order_backoff_markov": "transparent_markov_baseline",
    "return_boosted_markov": "transparent_markov_baseline",
    "recency_attention": "dependency_light_sequence_attention",
    "multi_horizon_recency_attention": "dependency_light_sequence_attention",
    "switching_noise_hmm": "classical_sequence",
    "particle_filter": "classical_state_space",
    "mlp_next_cell": "neural_sequence",
    AIF_CANDIDATE_NAME: "active_inference_candidate",
}

DEPENDENCY_FOOTPRINTS = {
    "hard_filter": "project_python_numpy_no_external_weights",
    "observed_markov": "project_python_numpy_no_external_weights",
    "soft_filter": "project_python_numpy_no_external_weights",
    "persistence": "project_python_numpy_no_external_weights",
    "online_base_rate": "project_python_numpy_no_external_weights",
    "first_order_markov": "project_python_numpy_no_external_weights",
    "higher_order_backoff_markov": "project_python_numpy_no_external_weights",
    "return_boosted_markov": "project_python_numpy_no_external_weights",
    "recency_attention": "project_python_numpy_no_external_weights",
    "multi_horizon_recency_attention": "project_python_numpy_no_external_weights",
    "switching_noise_hmm": "project_python_numpy_no_external_weights",
    "particle_filter": "project_python_numpy_no_external_weights",
    "mlp_next_cell": "project_python_numpy_no_external_weights",
    AIF_CANDIDATE_NAME: "project_python_numpy_no_external_weights",
}


def build_inference_config(
    config_path: str | Path = "configs/default.yaml",
    *,
    spatial_bins: int = DEFAULT_SPATIAL_BINS,
) -> InferenceConfig:
    """Load the project inference config and override the grid to ``spatial_bins`` per axis.

    ``n_states = spatial_bins ** 2`` (a 6x6 grid is 36 cells), matching the spine caveat.
    """

    cfg = load_config(config_path)
    return replace(
        cfg.inference,
        observation=replace(cfg.inference.observation, spatial_bins=spatial_bins),
    )


@dataclass(frozen=True)
class LaneResult:
    """Scored leaderboard and the AIF-vs-strongest-fair comparison for one lane."""

    lane: str
    n_states: int
    n_travellers: int
    eps: float
    aif_gamma: float
    aif_boost: float
    scored: dict[str, ScoredCandidate]
    strongest_fair_name: str
    strongest_fair_loss: float
    aif_name: str
    aif_loss: float
    mean_difference_nats: (
        float  # loss(strongest_fair) - loss(aif); positive => aif better
    )
    clustered_ci_low: float
    clustered_ci_high: float
    aif_beats_strongest_fair: (
        bool  # CI entirely on the AIF-better side AND lower mean loss
    )
    leakage_floor_ok: bool  # AIF loss stays well above the oracle floor (sanity check)
    n_holdout_travellers: int
    # Exact one-sided clustered-bootstrap p for H1: AIF strictly better on this lane.
    # Consumed by the multiplicity ledger as this lane's per-comparison uncorrected p.
    one_sided_p: float = 1.0
    aif_adaptation: float = 0.0


def _roster(
    dataset: PartialObservabilityDataset,
    *,
    aif_candidate: Candidate | None = None,
) -> tuple[Candidate, ...]:
    """Every candidate scored on a lane: bundled references + ladder + classical + neural + AIF.

    The neural and AIF candidates are trained/fitted on the dataset's TRAIN travellers only
    (leakage-free), so they are constructed per-lane from that lane's dataset. Fresh classical
    instances are required because they carry per-traveller recursive state.
    """

    aif_fit = fit_aif_candidate(dataset) if aif_candidate is None else None
    return (
        HardFilterCandidate(),
        ObservedMarkovCandidate(),
        SoftFilterCandidate(),
        *transparent_baseline_ladder(),
        *classical_candidates(),
        train_mlp_candidate(dataset),
        aif_candidate if aif_candidate is not None else aif_fit.candidate,
    )


def score_lane(
    lane: str,
    dataset: PartialObservabilityDataset,
    *,
    n_boot: int = 2000,
    boot_seed: int = 7,
) -> LaneResult:
    """Score every candidate on ``dataset`` and compute the AIF-vs-strongest-fair clustered CI.

    The strongest fair baseline is the lowest-loss candidate that is NOT in the AIF family
    (the oracle is never a candidate here, so it cannot be selected). The clustered CI is on
    ``loss(strongest_fair) - loss(aif)`` (positive => AIF better); the AIF candidate is judged
    to beat the strongest fair baseline only when it has the lower mean loss AND the CI is
    entirely on the AIF-better side of zero (``a_wins_clustered`` with ``a = strongest_fair``).
    """

    aif_fit = fit_aif_candidate(dataset)
    scored: dict[str, ScoredCandidate] = {}
    for cand in _roster(dataset, aif_candidate=aif_fit.candidate):
        t0 = time.perf_counter()
        raw_scored = score_candidate(dataset, cand)
        runtime_seconds = time.perf_counter() - t0
        try:
            loss_ci_low, loss_ci_high = traveller_clustered_bootstrap_ci(
                raw_scored.step_losses,
                raw_scored.step_travellers,
                n_boot=n_boot,
                seed=boot_seed,
            )
        except ValueError:
            loss_ci_low, loss_ci_high = None, None
        scored[cand.name] = replace(
            raw_scored,
            loss_ci_low=loss_ci_low,
            loss_ci_high=loss_ci_high,
            runtime_seconds=runtime_seconds,
        )

    if AIF_CANDIDATE_NAME not in scored:  # pragma: no cover - roster always includes it
        raise ValueError(
            f"AIF candidate {AIF_CANDIDATE_NAME!r} not scored on lane {lane!r}"
        )
    aif_scored = scored[AIF_CANDIDATE_NAME]

    fair = {n: s for n, s in scored.items() if n not in AIF_FAMILY_NAMES}
    if not fair:  # pragma: no cover - the ladder always supplies fair baselines
        raise ValueError(f"no fair (non-AIF) baseline scored on lane {lane!r}")
    strongest_fair = min(fair.values(), key=lambda s: s.pooled_mean_loss)

    # CI on loss(strongest_fair) - loss(aif): positive mean => AIF better. The strongest fair
    # baseline is candidate_a, the AIF candidate is candidate_b, so b_wins_clustered means the
    # AIF candidate is strictly better with the clustered interval separated from zero.
    ci = paired_loss_difference_ci(
        strongest_fair, aif_scored, n_boot=n_boot, seed=boot_seed
    )
    aif_beats = bool(
        ci.b_wins_clustered
        and aif_scored.pooled_mean_loss < strongest_fair.pooled_mean_loss
    )

    return LaneResult(
        lane=lane,
        n_states=dataset.n_states,
        n_travellers=int(np.unique(dataset.travellers).size),
        eps=float(dataset.eps),
        aif_gamma=float(aif_fit.gamma),
        aif_boost=float(aif_fit.boost),
        aif_adaptation=float(aif_fit.adaptation),
        scored=scored,
        strongest_fair_name=strongest_fair.name,
        strongest_fair_loss=float(strongest_fair.pooled_mean_loss),
        aif_name=AIF_CANDIDATE_NAME,
        aif_loss=float(aif_scored.pooled_mean_loss),
        mean_difference_nats=float(ci.mean_difference_nats),
        clustered_ci_low=float(ci.clustered_ci_low),
        clustered_ci_high=float(ci.clustered_ci_high),
        aif_beats_strongest_fair=aif_beats,
        leakage_floor_ok=bool(aif_scored.pooled_mean_loss > LEAKAGE_FLOOR_NATS),
        n_holdout_travellers=int(ci.n_holdout_travellers),
        one_sided_p=float(ci.one_sided_p_b_better),
    )


def build_lanes(
    inference_config: InferenceConfig,
    *,
    n_travellers: int = DEFAULT_N_TRAVELLERS,
    eps: float = DEFAULT_EPS,
) -> dict[str, PartialObservabilityDataset]:
    """Build the canonical SOTA-program lanes.

    Every lane is a :class:`PartialObservabilityDataset` scored by the IDENTICAL harness.
    They differ in latent dynamics and reporting process only; the split, emission channel,
    train-only transition learning, and prequential scorer stay shared.
    """

    fractalrabbit = build_partial_observability_dataset(
        inference_config, n_travellers=n_travellers, eps=eps
    )
    gravity = build_gravity_partial_observability_dataset(
        inference_config, n_travellers=n_travellers, eps=eps
    )
    reporting_gap = build_reporting_gap_partial_observability_dataset(
        inference_config, n_travellers=n_travellers, eps=eps
    )
    regime_stressor = build_regime_stressor_partial_observability_dataset(
        inference_config, n_travellers=n_travellers, eps=eps
    )
    return {
        "fractalrabbit": fractalrabbit,
        "gravity": gravity,
        "reporting_gap": reporting_gap,
        "regime_stressor": regime_stressor,
    }


def build_diagnostic_lanes(
    inference_config: InferenceConfig,
    *,
    n_travellers: int = DEFAULT_DIAGNOSTIC_N_TRAVELLERS,
    eps: float = DEFAULT_EPS,
) -> dict[str, PartialObservabilityDataset]:
    adversarial = build_adversarial_missingness_partial_observability_dataset(
        inference_config, n_travellers=n_travellers, eps=eps
    )
    reporting_gap_repeat = build_reporting_gap_partial_observability_dataset(
        inference_config,
        n_travellers=n_travellers,
        eps=eps,
        gen_seed=REPORTING_GAP_REPEAT_SEED,
        noise_seed=11,
    )
    high_motion_repeat = build_adversarial_missingness_partial_observability_dataset(
        inference_config,
        n_travellers=n_travellers,
        eps=eps,
        gen_seed=ADVERSARIAL_REPEAT_SEED,
        noise_seed=12,
    )
    regime_repeat = build_regime_stressor_repeat_partial_observability_dataset(
        inference_config, n_travellers=n_travellers, eps=eps
    )
    regime_repeat_2 = build_regime_stressor_repeat_partial_observability_dataset(
        inference_config,
        n_travellers=n_travellers,
        eps=eps,
        gen_seed=REGIME_STRESSOR_SECOND_REPEAT_SEED,
        noise_seed=13,
    )
    return {
        "reporting_gap_adversarial": adversarial,
        "reporting_gap_repeat": reporting_gap_repeat,
        "high_motion_adversarial_repeat": high_motion_repeat,
        "regime_stressor_repeat": regime_repeat,
        "regime_stressor_repeat_2": regime_repeat_2,
    }


@dataclass(frozen=True)
class GateResult:
    """The canonical multi-lane claim-gate verdict, derived from data, never asserted."""

    schema: str
    gate_clears: bool
    status: str
    aif_candidate_name: str
    lanes: dict[str, LaneResult]
    reasons: list[str]  # empty iff gate_clears
    residual_gap: dict[str, dict[str, float]]  # per-lane shortfall when not cleared
    run_parameters: dict[str, object]


def evaluate_gate(
    lanes: dict[str, LaneResult],
    *,
    aif_name: str = AIF_CANDIDATE_NAME,
    run_parameters: dict[str, object] | None = None,
) -> GateResult:
    """Clear ONLY if the AIF candidate beats the strongest fair baseline on EVERY lane.

    The gate is honest by construction: it clears iff, on every lane, the AIF candidate has
    the lower mean loss than the strongest fair baseline AND the traveller-clustered CI on the
    difference separates from zero AND the leakage sanity check holds. Any failing lane keeps
    the gate blocked and records its residual gap (how much loss the AIF candidate still owes
    the strongest fair baseline, and the CI edge nearest zero).
    """

    if not lanes:
        raise ValueError("no lanes to gate")

    reasons: list[str] = []
    residual: dict[str, dict[str, float]] = {}
    for lane_name, lr in lanes.items():
        # mean_difference_nats = loss(strongest_fair) - loss(aif); positive => AIF better.
        gap_nats = float(
            lr.aif_loss - lr.strongest_fair_loss
        )  # positive => AIF still worse
        # The CI edge nearest zero on the AIF-better side: ci_low (the conservative edge of
        # the loss(strongest_fair) - loss(aif) interval). >0 means CI fully on AIF-better side.
        residual[lane_name] = {
            "aif_loss": lr.aif_loss,
            "strongest_fair_loss": lr.strongest_fair_loss,
            "aif_minus_strongest_fair_nats": gap_nats,
            "clustered_ci_low": lr.clustered_ci_low,
            "clustered_ci_high": lr.clustered_ci_high,
        }
        if not lr.leakage_floor_ok:
            reasons.append(
                f"[{lane_name}] leakage sanity failed: AIF loss {lr.aif_loss:.4f} <= floor "
                f"{LEAKAGE_FLOOR_NATS} (implies oracle-like peeking)"
            )
        if lr.aif_loss >= lr.strongest_fair_loss:
            reasons.append(
                f"[{lane_name}] AIF candidate does NOT beat the strongest fair baseline "
                f"'{lr.strongest_fair_name}': AIF loss {lr.aif_loss:.4f} >= "
                f"{lr.strongest_fair_loss:.4f} (gap {gap_nats:+.4f} nats)"
            )
        elif not lr.aif_beats_strongest_fair:
            reasons.append(
                f"[{lane_name}] AIF candidate's lead over '{lr.strongest_fair_name}' is NOT "
                f"clustered-CI separated from zero: CI=[{lr.clustered_ci_low:.4f}, "
                f"{lr.clustered_ci_high:.4f}] does not stay above 0"
            )

    gate_clears = len(reasons) == 0
    status = "sota_program_cleared" if gate_clears else "sota_program_not_cleared"
    return GateResult(
        schema=SOTA_PROGRAM_HARNESS_SCHEMA,
        gate_clears=gate_clears,
        status=status,
        aif_candidate_name=aif_name,
        lanes=lanes,
        reasons=reasons,
        residual_gap=residual,
        run_parameters=run_parameters or {},
    )


def run_program(
    inference_config: InferenceConfig,
    *,
    n_travellers: int = DEFAULT_N_TRAVELLERS,
    eps: float = DEFAULT_EPS,
    n_boot: int = DEFAULT_N_BOOT,
    boot_seed: int = DEFAULT_BOOT_SEED,
) -> GateResult:
    """Build canonical lanes, score every candidate on each, and evaluate the claim gate."""

    datasets = build_lanes(inference_config, n_travellers=n_travellers, eps=eps)
    lanes = {
        name: score_lane(name, ds, n_boot=n_boot, boot_seed=boot_seed)
        for name, ds in datasets.items()
    }
    lane_names = list(lanes)
    run_parameters = {
        "n_travellers": int(n_travellers),
        "eps": float(eps),
        "spatial_bins": int(inference_config.spatial_bins),
        "n_boot": int(n_boot),
        "boot_seed": int(boot_seed),
        "lanes": lane_names,
        "canonical": _is_canonical_run(
            n_travellers=n_travellers,
            eps=eps,
            spatial_bins=inference_config.spatial_bins,
            n_boot=n_boot,
            boot_seed=boot_seed,
            lanes=lane_names,
        ),
    }
    return evaluate_gate(lanes, run_parameters=run_parameters)


def run_diagnostic_robustness_suite(
    inference_config: InferenceConfig,
    *,
    n_travellers: int = DEFAULT_DIAGNOSTIC_N_TRAVELLERS,
    eps: float = DEFAULT_EPS,
    n_boot: int = DEFAULT_DIAGNOSTIC_N_BOOT,
    boot_seed: int = DEFAULT_BOOT_SEED,
) -> GateResult:
    datasets = build_diagnostic_lanes(
        inference_config, n_travellers=n_travellers, eps=eps
    )
    lanes = {
        name: score_lane(name, ds, n_boot=n_boot, boot_seed=boot_seed)
        for name, ds in datasets.items()
    }
    run_parameters = {
        "n_travellers": int(n_travellers),
        "eps": float(eps),
        "spatial_bins": int(inference_config.spatial_bins),
        "n_boot": int(n_boot),
        "boot_seed": int(boot_seed),
        "lanes": list(lanes),
        "canonical": False,
        "diagnostic": True,
    }
    return evaluate_gate(lanes, run_parameters=run_parameters)


def _is_canonical_run(
    *,
    n_travellers: int,
    eps: float,
    spatial_bins: int,
    n_boot: int,
    boot_seed: int,
    lanes: list[str],
) -> bool:
    """Return whether a run is safe to hydrate manuscript SOTA program tokens."""

    return bool(
        int(n_travellers) == DEFAULT_N_TRAVELLERS
        and float(eps) == DEFAULT_EPS
        and int(spatial_bins) == DEFAULT_SPATIAL_BINS
        and int(n_boot) == DEFAULT_N_BOOT
        and int(boot_seed) == DEFAULT_BOOT_SEED
        and lanes == list(CANONICAL_LANES)
    )


def canonical_run_parameter_violations(
    run_parameters: Mapping[str, object] | None,
) -> list[str]:
    """Return human-readable reasons a SOTA-program artifact is not canonical."""

    if not isinstance(run_parameters, Mapping):
        return ["run_parameters missing"]
    violations: list[str] = []
    expected_ints = {
        "n_travellers": DEFAULT_N_TRAVELLERS,
        "spatial_bins": DEFAULT_SPATIAL_BINS,
        "n_boot": DEFAULT_N_BOOT,
        "boot_seed": DEFAULT_BOOT_SEED,
    }
    for key, expected in expected_ints.items():
        try:
            actual = int(run_parameters.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            violations.append(f"{key} missing or non-integer")
            continue
        if actual != expected:
            violations.append(f"{key}={actual} expected {expected}")
    try:
        eps = float(run_parameters.get("eps"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        violations.append("eps missing or non-numeric")
    else:
        if abs(eps - DEFAULT_EPS) > 1e-12:
            violations.append(f"eps={eps:g} expected {DEFAULT_EPS:g}")
    lanes = run_parameters.get("lanes")
    actual_lanes = list(lanes) if isinstance(lanes, (list, tuple)) else []
    if actual_lanes != list(CANONICAL_LANES):
        violations.append(f"lanes={lanes!r} expected {list(CANONICAL_LANES)!r}")
    if run_parameters.get("canonical") is not True:
        violations.append("canonical flag is not true")
    return violations


def _project_root_for_config(config_path: str | Path) -> Path:
    resolved = Path(config_path).resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return (
        resolved.parent.parent if resolved.parent.name == "configs" else resolved.parent
    )


def _assert_noncanonical_run_does_not_target_canonical_output(
    *,
    config_path: str | Path,
    out_dir: Path,
    n_travellers: int,
    eps: float,
    spatial_bins: int,
    n_boot: int,
    boot_seed: int,
) -> None:
    project_root = _project_root_for_config(config_path)
    canonical_out_dir = (project_root / "output" / "runs").resolve()
    if Path(out_dir).resolve() != canonical_out_dir:
        return
    intended_canonical = _is_canonical_run(
        n_travellers=n_travellers,
        eps=eps,
        spatial_bins=spatial_bins,
        n_boot=n_boot,
        boot_seed=boot_seed,
        lanes=list(CANONICAL_LANES),
    )
    if intended_canonical:
        return
    raise ValueError(
        "noncanonical SOTA-program run cannot write canonical output/runs artifacts; "
        "use a temporary directory or run scripts/18_run_sota_program.py with canonical defaults"
    )


# --------------------------------------------------------------------------- #
# Artifact emission
# --------------------------------------------------------------------------- #
def leaderboard_rows(gate: GateResult) -> list[dict[str, object]]:
    """Flat per-(candidate, lane) leaderboard rows with the clustered CI on the AIF row.

    Every candidate appears once per lane with its pooled mean loss and scored-step count.
    The clustered CI columns are populated ONLY on the AIF candidate's row (the CI is the
    AIF-vs-strongest-fair difference for that lane) and on the strongest-fair row (marked as
    the reference); all other rows leave the CI columns blank, since the CI is meaningful only
    for the AIF-vs-strongest-fair comparison.
    """

    rows: list[dict[str, object]] = []
    for lane_name, lr in gate.lanes.items():
        for name, scored in sorted(
            lr.scored.items(), key=lambda kv: kv[1].pooled_mean_loss
        ):
            is_aif = name == lr.aif_name
            is_strongest_fair = name == lr.strongest_fair_name
            role = (
                "aif_candidate"
                if is_aif
                else "strongest_fair_baseline"
                if is_strongest_fair
                else "fair_baseline"
                if name not in AIF_FAMILY_NAMES
                else "aif_reference"
            )
            rows.append(
                {
                    "candidate": name,
                    "lane": lane_name,
                    "role": role,
                    "mean_loss": round(float(scored.pooled_mean_loss), 6),
                    "n_scored_steps": int(scored.n_scored_steps),
                    "family": _candidate_family(name),
                    "locked_split_scored": True,
                    "leakage": False,
                    "loss_ci_low": _round_or_empty(scored.loss_ci_low),
                    "loss_ci_high": _round_or_empty(scored.loss_ci_high),
                    "clustered_ci_low": round(lr.clustered_ci_low, 6) if is_aif else "",
                    "clustered_ci_high": round(lr.clustered_ci_high, 6)
                    if is_aif
                    else "",
                    "brier_score": round(float(scored.brier_score), 6),
                    "ece": round(float(scored.ece), 6),
                    "top1_accuracy": round(float(scored.top1_accuracy), 6),
                    "runtime_seconds": _round_or_empty(scored.runtime_seconds),
                    "lane_coverage": _lane_coverage(lane_name),
                    "dependency_footprint": _dependency_footprint(name),
                    "missingness_robustness_status": _missingness_status(lane_name),
                    "failure_modes": _failure_modes(
                        name=name,
                        role=role,
                        lane_name=lane_name,
                        lane_result=lr,
                    ),
                }
            )
    return rows


def _round_or_empty(value: float | None) -> object:
    if value is None:
        return ""
    return round(float(value), 6)


def _candidate_family(name: str) -> str:
    return CANDIDATE_FAMILIES.get(name, "unclassified_candidate")


def _dependency_footprint(name: str) -> str:
    return DEPENDENCY_FOOTPRINTS.get(name, "unknown_dependency_footprint")


def _lane_coverage(lane_name: str) -> str:
    diagnostic = "|".join(DIAGNOSTIC_STRESSOR_LANES)
    return f"scored_on_{lane_name}; canonical_lanes={'|'.join(CANONICAL_LANES)}; diagnostic_lanes={diagnostic}"


def _missingness_status(lane_name: str) -> str:
    if lane_name == "reporting_gap":
        return "scored_reporting_gap_burst_missingness_locked_split"
    if lane_name == "reporting_gap_adversarial":
        return "scored_adversarial_high_motion_missingness_diagnostic_locked_split"
    if lane_name == "reporting_gap_repeat":
        return "scored_repeated_reporting_gap_burst_missingness_diagnostic_locked_split"
    if lane_name == "high_motion_adversarial_repeat":
        return "scored_repeated_adversarial_high_motion_missingness_diagnostic_locked_split"
    if lane_name == "regime_stressor":
        return "scored_regime_stressor_reporting_gaps_locked_split"
    if lane_name == "regime_stressor_repeat":
        return "scored_repeated_regime_stressor_diagnostic_locked_split"
    if lane_name == "regime_stressor_repeat_2":
        return "scored_second_repeated_regime_stressor_diagnostic_locked_split"
    return "scored_nominal_noisy_emission_partial_observability"


def _failure_modes(
    *,
    name: str,
    role: str,
    lane_name: str,
    lane_result: LaneResult,
) -> str:
    if name == lane_result.aif_name:
        if lane_result.aif_beats_strongest_fair:
            return "aif_gate_candidate_wins_this_lane_but_full_gate_may_still_require_other_lanes"
        return (
            "aif_gate_candidate_not_ci_separated_or_not_lower_loss_than_"
            f"{lane_result.strongest_fair_name}_on_{lane_name}"
        )
    if role == "strongest_fair_baseline":
        return "strongest_fair_reference_bar_for_aif_gate_on_this_lane"
    if name in AIF_FAMILY_NAMES:
        return "aif_family_reference_excluded_from_strongest_fair_pool"
    return "not_strongest_fair_on_this_lane"


def _csv_cell(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    text = "" if value is None else str(value)
    if any(c in text for c in (",", '"', "\n")):
        return '"' + text.replace('"', '""') + '"'
    return text


def write_leaderboard_csv(gate: GateResult, path: Path) -> Path:
    """Write the flat per-(candidate, lane) leaderboard to ``path`` as CSV."""

    rows = leaderboard_rows(gate)
    lines = [",".join(SOTA_PROGRAM_LEADERBOARD_HEADER)]
    for r in rows:
        lines.append(",".join(_csv_cell(r[h]) for h in SOTA_PROGRAM_LEADERBOARD_HEADER))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def robustness_diagnostic_payload(gate: GateResult) -> dict[str, object]:
    definitions = all_lane_definitions()
    rows = leaderboard_rows(gate)
    lane_summaries: dict[str, object] = {}
    for lane_name, lr in gate.lanes.items():
        lane_summaries[lane_name] = {
            "status": "scored",
            "definition": definitions.get(lane_name, {}),
            "n_travellers": lr.n_travellers,
            "n_holdout_travellers": lr.n_holdout_travellers,
            "aif_candidate": lr.aif_name,
            "aif_loss": round(lr.aif_loss, 6),
            "strongest_fair_baseline": lr.strongest_fair_name,
            "strongest_fair_loss": round(lr.strongest_fair_loss, 6),
            "clustered_ci_low": round(lr.clustered_ci_low, 6),
            "clustered_ci_high": round(lr.clustered_ci_high, 6),
            "aif_beats_strongest_fair": lr.aif_beats_strongest_fair,
        }
    lane_statuses = [
        {
            "lane": name,
            "status": "scored" if name in gate.lanes else "missing",
            "boundary": definitions.get(name, {}).get(
                "robustness_role", "diagnostic stressor lane"
            ),
        }
        for name in DIAGNOSTIC_STRESSOR_LANES
    ]
    return {
        "schema": SOTA_PROGRAM_ROBUSTNESS_DIAGNOSTICS_SCHEMA,
        "status": "diagnostic_stressors_scored",
        "gate_clears": gate.gate_clears,
        "claim_gate_reasons": gate.reasons,
        "run_parameters": gate.run_parameters,
        "diagnostic_lanes": lane_summaries,
        "lane_statuses": lane_statuses,
        "all_diagnostic_lanes_scored": all(
            row["status"] == "scored" for row in lane_statuses
        ),
        "candidate_count": len({str(row["candidate"]) for row in rows}),
        "row_count": len(rows),
        "leaderboard_columns": list(SOTA_PROGRAM_LEADERBOARD_HEADER),
        "leaderboard_rows": rows,
        "boundary": (
            "Diagnostic adversarial-missingness and repeated-regime lanes are scored synthetic "
            "robustness evidence. They can block overclaiming and guide future work, but they "
            "do not by themselves clear SOTA, empirical-transfer, or operational claims."
        ),
    }


def write_robustness_diagnostics(gate: GateResult, out_dir: Path) -> dict[str, Path]:
    csv_path = write_leaderboard_csv(gate, out_dir / "sota_program_robustness_grid.csv")
    json_path = out_dir / "sota_program_robustness_grid.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    json_path.write_text(
        json.dumps(robustness_diagnostic_payload(gate), indent=2) + "\n",
        encoding="utf-8",
    )
    return {"robustness_grid": csv_path, "robustness_diagnostics": json_path}


def gate_payload(gate: GateResult) -> dict[str, object]:
    """A JSON-serialisable, audit-complete summary of the canonical gate verdict."""

    lane_summaries: dict[str, object] = {}
    for lane_name, lr in gate.lanes.items():
        lane_summaries[lane_name] = {
            "n_states": lr.n_states,
            "n_travellers": lr.n_travellers,
            "n_holdout_travellers": lr.n_holdout_travellers,
            "eps": lr.eps,
            "aif_gamma": lr.aif_gamma,
            "aif_boost": lr.aif_boost,
            "aif_candidate": lr.aif_name,
            "aif_loss": round(lr.aif_loss, 6),
            "strongest_fair_baseline": lr.strongest_fair_name,
            "strongest_fair_loss": round(lr.strongest_fair_loss, 6),
            "mean_difference_nats_fair_minus_aif": round(lr.mean_difference_nats, 6),
            "clustered_ci_low": round(lr.clustered_ci_low, 6),
            "clustered_ci_high": round(lr.clustered_ci_high, 6),
            "one_sided_p": round(lr.one_sided_p, 6),
            "aif_beats_strongest_fair": lr.aif_beats_strongest_fair,
            "leakage_floor_ok": lr.leakage_floor_ok,
            "candidate_rows": [
                {
                    "candidate": name,
                    "family": _candidate_family(name),
                    "mean_loss": round(float(scored.pooled_mean_loss), 6),
                    "loss_ci_low": _round_or_empty(scored.loss_ci_low),
                    "loss_ci_high": _round_or_empty(scored.loss_ci_high),
                    "brier_score": round(float(scored.brier_score), 6),
                    "ece": round(float(scored.ece), 6),
                    "top1_accuracy": round(float(scored.top1_accuracy), 6),
                    "runtime_seconds": _round_or_empty(scored.runtime_seconds),
                    "dependency_footprint": _dependency_footprint(name),
                    "missingness_robustness_status": _missingness_status(lane_name),
                }
                for name, scored in sorted(
                    lr.scored.items(), key=lambda kv: kv[1].pooled_mean_loss
                )
            ],
        }
    definitions = lane_definitions()
    lane_statuses = [
        {
            "lane": name,
            "status": "scored" if name in gate.lanes else "missing",
            "boundary": definitions.get(name, {}).get(
                "robustness_role", "canonical SOTA-program lane"
            ),
        }
        for name in CANONICAL_LANES
    ]
    return {
        "schema": gate.schema,
        "status": gate.status,
        "gate_clears": gate.gate_clears,
        "aif_candidate": gate.aif_candidate_name,
        "claim": (
            "AIF candidate beats the strongest FAIR baseline on every canonical lane with "
            "the traveller-clustered CI separated from zero, leakage-free."
        ),
        "lanes": lane_summaries,
        "lane_definitions": definitions,
        "robustness_summary": {
            "canonical_lanes": list(CANONICAL_LANES),
            "lane_statuses": lane_statuses,
            "all_canonical_lanes_scored": all(
                row["status"] == "scored" for row in lane_statuses
            ),
            "reporting_gap_scored": "reporting_gap" in gate.lanes,
            "regime_stressor_scored": "regime_stressor" in gate.lanes,
            "claim_boundary": (
                "Reporting-gap and regime-stressor lanes are scored synthetic robustness evidence, "
                "not empirical transfer, operational readiness, or SOTA clearance."
            ),
        },
        "claim_gate_reasons": gate.reasons,
        "residual_gap": gate.residual_gap,
        "run_parameters": gate.run_parameters,
        "candidate_table_contract": {
            "columns": list(SOTA_PROGRAM_LEADERBOARD_HEADER),
            "uncertainty": (
                "loss_ci_low/loss_ci_high are per-candidate traveller-clustered loss "
                "intervals; clustered_ci_low/clustered_ci_high are the AIF-vs-strongest-"
                "fair paired-difference interval and are populated on the AIF row."
            ),
            "calibration": "Brier score, top-label ECE, and top-1 accuracy are computed on the same holdout steps.",
        },
        "readiness": {
            "missingness_robustness": {
                "status": "scored",
                "current_column": "missingness_robustness_status",
                "boundary": (
                    "The canonical SOTA program now includes a deterministic reporting-gap "
                    "lane; this is synthetic robustness evidence only and does not support "
                    "real missing-data or adversarial claims."
                ),
            },
            "diagnostic_stressors": {
                "status": "scored_separately_when_sota_program_runs",
                "lanes": list(DIAGNOSTIC_STRESSOR_LANES),
                "artifacts": [
                    "output/runs/sota_program_robustness_grid.json",
                    "output/runs/sota_program_robustness_grid.csv",
                ],
                "boundary": (
                    "Diagnostic stressor rows expand robustness evidence without changing the "
                    "canonical four-lane manuscript gate."
                ),
            },
            "regime_stressor_lanes": lane_statuses,
            "manuscript_use": (
                "These robustness lanes can support only synthetic robustness wording; they "
                "cannot unlock positive SOTA, empirical-transfer, or operational claims unless "
                "the full claim gate also clears."
            ),
        },
        "caveat": (
            "Synthetic-only, four-lane fair-frontier characterization. A cleared gate "
            "is NOT a real-mobility, deployment, or surveillance-readiness claim. The oracle "
            "is a ceiling, never a competitor; the strongest FAIR baseline is the bar."
        ),
    }


def write_gate_json(gate: GateResult, path: Path) -> Path:
    """Write the gate verdict to ``path`` as pretty-printed JSON."""

    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gate_payload(gate), indent=2) + "\n", encoding="utf-8")
    return path


def run_and_write(
    *,
    config_path: str | Path = "configs/default.yaml",
    out_dir: Path,
    n_travellers: int = DEFAULT_N_TRAVELLERS,
    eps: float = DEFAULT_EPS,
    spatial_bins: int = DEFAULT_SPATIAL_BINS,
    n_boot: int = DEFAULT_N_BOOT,
    boot_seed: int = DEFAULT_BOOT_SEED,
) -> tuple[GateResult, dict[str, Path]]:
    """End-to-end: build config + lanes, score, gate, and write the leaderboard + gate JSON."""

    _assert_noncanonical_run_does_not_target_canonical_output(
        config_path=config_path,
        out_dir=out_dir,
        n_travellers=n_travellers,
        eps=eps,
        spatial_bins=spatial_bins,
        n_boot=n_boot,
        boot_seed=boot_seed,
    )
    inference_config = build_inference_config(config_path, spatial_bins=spatial_bins)
    gate = run_program(
        inference_config,
        n_travellers=n_travellers,
        eps=eps,
        n_boot=n_boot,
        boot_seed=boot_seed,
    )
    leaderboard_path = write_leaderboard_csv(
        gate, out_dir / "sota_program_leaderboard.csv"
    )
    gate_path = write_gate_json(gate, out_dir / "sota_program_gate.json")
    diagnostic_gate = run_diagnostic_robustness_suite(
        inference_config,
        n_travellers=min(int(n_travellers), DEFAULT_DIAGNOSTIC_N_TRAVELLERS),
        eps=eps,
        n_boot=min(int(n_boot), DEFAULT_DIAGNOSTIC_N_BOOT),
        boot_seed=boot_seed,
    )
    diagnostic_paths = write_robustness_diagnostics(diagnostic_gate, out_dir)
    return gate, {
        "leaderboard": leaderboard_path,
        "gate": gate_path,
        **diagnostic_paths,
    }


# Re-export the emission helper so callers wiring custom lanes have a single import surface.
__all__ = [
    "SOTA_PROGRAM_HARNESS_SCHEMA",
    "AIF_CANDIDATE_NAME",
    "AIF_FAMILY_NAMES",
    "LEAKAGE_FLOOR_NATS",
    "DEFAULT_N_TRAVELLERS",
    "DEFAULT_EPS",
    "DEFAULT_SPATIAL_BINS",
    "DEFAULT_N_BOOT",
    "DEFAULT_BOOT_SEED",
    "CANONICAL_LANES",
    "DIAGNOSTIC_STRESSOR_LANES",
    "SOTA_PROGRAM_LEADERBOARD_HEADER",
    "SOTA_PROGRAM_ROBUSTNESS_DIAGNOSTICS_SCHEMA",
    "LaneResult",
    "GateResult",
    "build_inference_config",
    "build_lanes",
    "build_diagnostic_lanes",
    "score_lane",
    "evaluate_gate",
    "run_program",
    "run_diagnostic_robustness_suite",
    "canonical_run_parameter_violations",
    "leaderboard_rows",
    "write_leaderboard_csv",
    "gate_payload",
    "write_gate_json",
    "robustness_diagnostic_payload",
    "write_robustness_diagnostics",
    "run_and_write",
    "emission_matrix",
]
