"""State-of-the-art benchmark harness, locked task contract, and false-claim oracle.

This module delivers the M12 SOTA program's *implementable, honest* core. It does
not assert that any current method is state of the art. Instead it builds the
machinery that would be needed to make such a claim defensibly, and proves --
through a false-claim oracle of adversarial probe rows — that the claim gate
rejects every illegitimate path to a "performance-leading" claim:

* a *leakage* candidate that peeks at the true next cell (lowest loss, excluded),
* a *fixture-only* win with no external-lane evidence,
* a *single-metric* win that loses on calibration,
* a *minimization regression* win that depends on direct location, and
* an *integrity regression* win under a poisoned stream.

The benchmark scores the dependency-free transparent baseline ladder on a
*locked, leakage-resistant split*: travellers are partitioned disjointly into
train and holdout so traveller identity never crosses the split, and the
realization (fixture vs. external generator) is recorded on every row. Online
predictors are primed on the train travellers' population statistics and then
scored prequentially on the held-out travellers' streams.

Heavier candidates (pymdp, factorial, neural, latent-regime HMM) are recorded
from their existing artifacts as full-stream rows that are *not* locked-split
scored, so they remain ineligible for a SOTA claim until re-scored under this
contract — a deliberate, visible gate rather than a silent omission.

Foundation-style and LLM mobility models are declared as required baseline
*categories* that are currently unavailable; their absence alone keeps the claim
gate closed. All claims stay synthetic-only and non-operational.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .objectives import ECE_EDGE_FIRST_CLOSED, expected_calibration_error
from .types import ObservationSequence

SOTA_SCHEMA = "active-fractal-rabbit-sota-v1"

# Required strong-baseline families. Implemented families are scored here;
# declared-unavailable families keep the claim gate closed until provided.
REQUIRED_BASELINE_FAMILIES: dict[str, str] = {
    "transparent_null": "base-rate, persistence, first-order/phase/metadata Markov, return-boosted Markov",
    "classical_sequence": "switching/latent-regime HMM, state-space / particle mixtures",
    "neural_sequence": "dependency-light neural / logistic sequence learner",
    "foundation_mobility": "pretrained mobility transformer / mixture-of-experts (declared, unavailable)",
    "llm_mobility": "instruction-tuned / fine-tuned mobility language model (declared, unavailable)",
}

# A candidate must beat the strongest eligible baseline by at least this margin
# (in nats of mean predictive loss) before the gate will even consider a win.
MIN_WIN_MARGIN_NATS = 0.02


def lock_traveller_split(
    sequence: ObservationSequence,
    *,
    holdout_fraction: float = 0.4,
    seed: int = 20260616,
) -> dict[str, object]:
    """Partition travellers disjointly into train and holdout (no identity leak).

    The split is keyed on traveller id, so no traveller's stream appears in both
    train and holdout. The plan is fully determined by ``seed`` and the sorted
    set of traveller ids, so it is reproducible and locked before scoring.
    """

    travellers = np.asarray(
        [row.traveller_id for row in sequence.observations], dtype=int
    )
    unique = np.unique(travellers)
    if unique.size < 2:
        raise ValueError("locked split needs at least two distinct travellers")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_holdout = max(1, int(round(holdout_fraction * unique.size)))
    n_holdout = min(n_holdout, unique.size - 1)  # keep at least one train traveller
    holdout_ids = set(int(t) for t in shuffled[:n_holdout])
    is_holdout = np.array([int(t) in holdout_ids for t in travellers], dtype=bool)
    return {
        "seed": int(seed),
        "holdout_fraction": float(holdout_fraction),
        "n_travellers": int(unique.size),
        "n_holdout_travellers": int(len(holdout_ids)),
        "holdout_traveller_ids": sorted(holdout_ids),
        "is_holdout": is_holdout,
        "n_train_steps": int((~is_holdout).sum()),
        "n_holdout_steps": int(is_holdout.sum()),
    }


def _neg_log(p: float) -> float:
    return float(-np.log(max(float(p), 1e-12)))


def _context_codes(sequence: ObservationSequence, kind: str) -> tuple[np.ndarray, int]:
    """Return per-step context codes and the context cardinality for a predictor."""

    phases = np.asarray(sequence.time_bins, dtype=int)
    speed = np.asarray(sequence.speed_categories, dtype=int)
    burst = np.asarray(sequence.burst_categories, dtype=int)
    n_phase = int(sequence.num_time_bins)
    n_speed = int(sequence.num_speed_categories)
    n_burst = int(sequence.num_burst_categories)
    if kind == "none":
        return np.zeros(len(phases), dtype=int), 1
    if kind == "phase":
        return phases, n_phase
    if kind == "metadata":
        return (
            (phases * n_speed + speed) * n_burst
        ) + burst, n_phase * n_speed * n_burst
    raise ValueError(f"unknown context kind: {kind}")


def _score_predictor(
    sequence: ObservationSequence,
    is_holdout: np.ndarray,
    *,
    smoothing: float,
    mode: str,
    context: str,
    n_calib_bins: int = 10,
) -> dict[str, float]:
    """Prime an online predictor on train travellers, score it on holdout steps.

    ``mode`` selects the predictor family; ``context`` selects the conditioning
    metadata. Returns mean predictive loss (nats), Brier score, and ECE over the
    held-out, non-first steps. The predictor updates online as it scores
    (prequential continuation past the training prime), but only holdout steps
    contribute to the metrics.
    """

    cells = np.asarray(sequence.spatial_states, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    travellers = np.asarray(
        [row.traveller_id for row in sequence.observations], dtype=int
    )
    n_states = int(sequence.num_spatial_states)
    codes, n_ctx = _context_codes(sequence, context)

    counts = np.full((n_ctx, n_states, n_states), smoothing, dtype=float)
    marginal = np.full((n_ctx, n_states), smoothing, dtype=float)
    visited: dict[int, np.ndarray] = {}

    losses: list[float] = []
    brier: list[float] = []
    confidences: list[float] = []
    hits: list[float] = []

    for idx in range(len(cells)):
        cell = int(cells[idx])
        ctx = int(codes[idx])
        tid = int(travellers[idx])
        seen = visited.setdefault(tid, np.zeros(n_states, dtype=bool))
        if first[idx]:
            seen[cell] = True
            if mode == "base_rate":
                marginal[ctx, cell] += 1.0
            continue
        prev = int(cells[idx - 1])

        if mode == "base_rate":
            probs = marginal[ctx] / marginal[ctx].sum()
        elif mode == "persistence":
            probs = np.full(n_states, smoothing, dtype=float)
            probs[prev] += 1.0
            probs /= probs.sum()
        elif mode == "markov":
            probs = counts[ctx, prev] / counts[ctx, prev].sum()
        elif mode == "return_boost":
            scores = counts[ctx, prev].copy()
            scores[seen] += 1.0
            probs = scores / scores.sum()
        elif mode == "leakage_oracle":
            probs = np.full(n_states, 1e-9, dtype=float)
            probs[cell] = 1.0  # peeks the true next cell — adversarial probe
            probs /= probs.sum()
        else:
            raise ValueError(f"unknown predictor mode: {mode}")

        if bool(is_holdout[idx]):
            p_true = float(probs[cell])
            losses.append(_neg_log(p_true))
            onehot = np.zeros(n_states, dtype=float)
            onehot[cell] = 1.0
            brier.append(float(np.sum((probs - onehot) ** 2)))
            pred = int(np.argmax(probs))
            confidences.append(float(probs[pred]))
            hits.append(1.0 if pred == cell else 0.0)

        # online update (prequential continuation, train-primed)
        counts[ctx, prev, cell] += 1.0
        marginal[ctx, cell] += 1.0
        seen[cell] = True

    if not losses:
        raise ValueError("no held-out scored steps for predictor")
    conf = np.asarray(confidences)
    hit = np.asarray(hits)
    ci_low, ci_high = _bootstrap_loss_ci(np.asarray(losses))
    return {
        "mean_predictive_loss": float(np.mean(losses)),
        "loss_ci_low": ci_low,
        "loss_ci_high": ci_high,
        "brier_score": float(np.mean(brier)),
        "ece": _ece(conf, hit, n_calib_bins),
        "n_scored_steps": int(len(losses)),
    }


def _bootstrap_loss_ci(
    losses: np.ndarray, *, n_boot: int = 2000, seed: int = 12345, alpha: float = 0.05
) -> tuple[float, float]:
    """Deterministic percentile bootstrap CI for the mean predictive loss.

    Makes the win-margin claim defensible: a candidate that beats a baseline's
    point loss but sits inside the baseline's loss CI is not a separated win.
    """

    if losses.size < 2:
        m = float(losses.mean()) if losses.size else 0.0
        return m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, losses.size, size=(n_boot, losses.size))
    means = losses[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _ece(confidences: np.ndarray, hits: np.ndarray, n_bins: int) -> float:
    # Delegates to objectives.expected_calibration_error (single source of truth for
    # the equal-width ECE binning loop, previously reimplemented here, in
    # neural_baseline.py, and in objectives.py's own recovery_calibration with three
    # different edge-inclusion conventions). ECE_EDGE_FIRST_CLOSED reproduces this
    # site's original convention exactly: (lo, hi] except the first bin, [lo, hi].
    return expected_calibration_error(
        confidences, hits, n_bins, edge_convention=ECE_EDGE_FIRST_CLOSED
    )[0]


# Candidate ladder scored on the locked split (all dependency-free, deterministic).
_LOCKED_CANDIDATES: tuple[tuple[str, str, str, str, str], ...] = (
    # (model, family, mode, context, role)
    (
        "base_rate",
        "transparent_null",
        "base_rate",
        "none",
        "online marginal-frequency next-cell baseline",
    ),
    (
        "persistence",
        "transparent_null",
        "persistence",
        "none",
        "smoothed previous-cell persistence null",
    ),
    (
        "first_order_markov",
        "transparent_null",
        "markov",
        "none",
        "online P(cell_t | cell_{t-1})",
    ),
    (
        "phase_markov",
        "transparent_null",
        "markov",
        "phase",
        "online P(cell_t | cell_{t-1}, phase)",
    ),
    (
        "metadata_markov",
        "transparent_null",
        "markov",
        "metadata",
        "online P(cell_t | cell_{t-1}, phase, speed, burst)",
    ),
    (
        "return_boost_markov",
        "transparent_null",
        "return_boost",
        "none",
        "first-order Markov with per-traveller return boost",
    ),
)


def run_sota_benchmark(
    sequence: ObservationSequence,
    *,
    smoothing: float,
    lane: str = "fixture",
    holdout_fraction: float = 0.4,
    seed: int = 20260616,
    external_available: bool = False,
    minimization_ok: bool | None = None,
    integrity_ok: bool | None = None,
    extra_full_stream_rows: list[dict[str, object]] | None = None,
    injected_candidate: dict[str, object] | None = None,
    injected_locked_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Score the locked-split leaderboard, run the false-claim oracle, and gate.

    ``injected_candidate`` lets tests supply a synthetic eligible candidate row
    (the positive control proving the gate is not vacuous). ``minimization_ok``
    and ``integrity_ok`` carry the location-minimization and poisoning verdicts
    from their own artifacts; ``None`` means "not yet evaluated" and is treated
    as a regression for SOTA eligibility.
    """

    split = lock_traveller_split(sequence, holdout_fraction=holdout_fraction, seed=seed)
    is_holdout = split["is_holdout"]  # type: ignore[assignment]

    rows: list[dict[str, object]] = []
    for model, family, mode, context, role in _LOCKED_CANDIDATES:
        t0 = time.perf_counter()
        metrics = _score_predictor(
            sequence, is_holdout, smoothing=smoothing, mode=mode, context=context
        )
        rows.append(
            {
                "model": model,
                "family": family,
                "current_role": role,
                "lane": lane,
                "locked_split_scored": True,
                "leakage": False,
                "runtime_seconds": round(time.perf_counter() - t0, 6),
                **metrics,
            }
        )

    # Optionally injected locked-split baseline rows for families that are
    # otherwise unavailable (e.g. a provided classical/neural/foundation/LLM
    # baseline scored under this contract). Lets the gate become clearable.
    for r in injected_locked_rows or []:
        row = dict(r)
        row.setdefault("locked_split_scored", True)
        row.setdefault("leakage", False)
        row.setdefault("lane", lane)
        row["injected"] = (
            True  # not a genuine in-harness predictor; recorded for honesty
        )
        rows.append(row)

    # Eligible baselines = locked-split, non-leakage, with a finite loss. Filtering
    # non-finite losses here keeps a NaN/Inf row from poisoning the min() selection.
    baseline_rows = [
        r
        for r in rows
        if not r["leakage"]
        and r["locked_split_scored"]
        and math.isfinite(float(r.get("mean_predictive_loss", float("nan"))))
    ]
    if not baseline_rows:
        raise ValueError("no eligible baseline rows with a finite predictive loss")
    strongest_baseline = min(
        baseline_rows, key=lambda r: float(r["mean_predictive_loss"])
    )
    strongest_baseline_loss = float(strongest_baseline["mean_predictive_loss"])
    # Lower edge of the strongest baseline's loss CI: a candidate must beat this
    # (loss below the baseline's optimistic end), not merely the point estimate,
    # to count as a CI-separated win rather than overlap within the baseline band.
    strongest_baseline_ci_low = float(
        strongest_baseline.get("loss_ci_low", strongest_baseline_loss)
    )
    persistence_loss = next(
        float(r["mean_predictive_loss"]) for r in rows if r["model"] == "persistence"
    )

    # ---- False-claim oracle: adversarial probe rows that MUST be rejected ----
    t0 = time.perf_counter()
    leak = _score_predictor(
        sequence, is_holdout, smoothing=smoothing, mode="leakage_oracle", context="none"
    )
    leakage_row = {
        "model": "leakage_oracle_probe",
        "family": "adversarial_probe",
        "current_role": "peeks the true next cell; proves the leakage guard excludes it",
        "lane": lane,
        "locked_split_scored": True,
        "leakage": True,
        "runtime_seconds": round(time.perf_counter() - t0, 6),
        **leak,
    }
    rows.append(leakage_row)

    # Heavier candidates recorded from their own artifacts as full-stream rows.
    # These are NOT scored in-harness, so locked_split_scored is forced False
    # (overwrite, not setdefault): a caller cannot assert eligibility for a row the
    # harness never scored, which would otherwise satisfy a baseline family with junk.
    full_stream_rows = [dict(r) for r in (extra_full_stream_rows or [])]
    for r in full_stream_rows:
        r["locked_split_scored"] = False
        r.setdefault("leakage", False)
        rows.append(r)

    # Optional positive-control / real candidate scored on the locked split.
    candidate_eval: dict[str, object] = {"present": False}
    if injected_candidate is not None:
        cand = dict(injected_candidate)
        cand.setdefault("locked_split_scored", True)
        cand.setdefault("leakage", False)
        cand.setdefault("lane", lane)
        cand.setdefault("family", "active_inference")
        cand["injected"] = (
            True  # supplied by a caller, not scored in-harness; recorded for honesty
        )
        rows.append(cand)
        candidate_eval = _evaluate_candidate(
            cand,
            strongest_baseline_loss=strongest_baseline_loss,
            strongest_baseline_ci_low=strongest_baseline_ci_low,
            baseline_rows=baseline_rows,
            external_available=external_available,
            minimization_ok=minimization_ok,
            integrity_ok=integrity_ok,
        )

    # ---- Claim gate: status derived from data, never asserted ----
    missing_families = sorted(
        fam
        for fam in REQUIRED_BASELINE_FAMILIES
        if not any(
            r.get("family") == fam and r.get("locked_split_scored") for r in rows
        )
    )
    reasons: list[str] = []
    if missing_families:
        reasons.append(
            "required strong-baseline families absent on the locked split: "
            + ", ".join(missing_families)
        )
    if not external_available:
        reasons.append(
            "no current-run external-lane evidence; win would be fixture-only"
        )
    if not candidate_eval.get("present"):
        reasons.append(
            "no eligible candidate beats the strongest baseline on the locked split"
        )
    elif not candidate_eval.get("clears"):
        reasons.extend(str(x) for x in candidate_eval.get("failures", []))

    cleared = len(reasons) == 0
    status = "sota_cleared" if cleared else "sota_not_cleared"
    used_injected_rows = any(bool(r.get("injected")) for r in rows)
    if cleared and used_injected_rows:
        # A clear that depends on caller-injected (non-in-harness) rows is honestly
        # flagged so it can never be read as a clear earned by real predictors.
        reasons.append(
            "NOTE: clearance depended on caller-injected rows (positive control / "
            "supplied baselines), not solely on in-harness locked-split predictors"
        )

    leaderboard = sorted(rows, key=lambda r: float(r["mean_predictive_loss"]))
    return {
        "schema": SOTA_SCHEMA,
        "status": status,
        "lane": lane,
        "task_contract": _task_contract(split, smoothing, lane, external_available),
        "locked_split": {k: v for k, v in split.items() if k != "is_holdout"},
        "strongest_baseline_loss": strongest_baseline_loss,
        "strongest_baseline_ci_low": strongest_baseline_ci_low,
        "persistence_loss": persistence_loss,
        "min_win_margin_nats": MIN_WIN_MARGIN_NATS,
        "candidate_evaluation": candidate_eval,
        "minimization_ok": minimization_ok,
        "integrity_ok": integrity_ok,
        "used_injected_rows": used_injected_rows,
        "missing_baseline_families": missing_families,
        "claim_gate_reasons": reasons,
        "leaderboard": leaderboard,
        "false_claim_oracle": _oracle_summary(leakage_row, strongest_baseline_loss),
        "caveat": (
            "Synthetic-only. A cleared gate would still not unlock real-waypoint, "
            "deployment, surveillance-readiness, or operational tasking claims."
        ),
    }


def _evaluate_candidate(
    candidate: dict[str, object],
    *,
    strongest_baseline_loss: float,
    strongest_baseline_ci_low: float,
    baseline_rows: list[dict[str, object]],
    external_available: bool,
    minimization_ok: bool | None,
    integrity_ok: bool | None,
) -> dict[str, object]:
    """Decide whether a candidate legitimately clears every SOTA sub-gate."""

    loss = float(candidate["mean_predictive_loss"])
    margin = strongest_baseline_loss - loss  # positive => candidate is better
    finite_briers = [
        float(r["brier_score"])
        for r in baseline_rows
        if math.isfinite(float(r["brier_score"]))
    ]
    best_baseline_brier = min(finite_briers) if finite_briers else float("inf")
    failures: list[str] = []

    # Fail closed on non-finite metrics: a NaN/Inf loss or Brier makes every negative
    # comparison below evaluate False, so without this an uncomputable candidate would
    # "clear" every sub-gate. Require finiteness with positive (fail-closed) checks.
    if not math.isfinite(loss):
        failures.append("candidate loss is non-finite (NaN/Inf) and is ineligible")
    if bool(candidate.get("leakage")):
        failures.append("candidate is a leakage probe and is excluded from eligibility")
    if not bool(candidate.get("locked_split_scored")):
        failures.append("candidate is not locked-split scored")
    if not (math.isfinite(margin) and margin >= MIN_WIN_MARGIN_NATS):
        failures.append(
            f"loss margin {margin:.4f} nats below required {MIN_WIN_MARGIN_NATS} nats over strongest baseline"
        )
    # CI-separation: the candidate must beat the strongest baseline's LOWER loss CI
    # edge (its optimistic end), so a win inside the baseline's uncertainty band does
    # not count. Positive-form check so a non-finite loss fails closed.
    if not (math.isfinite(loss) and loss < strongest_baseline_ci_low):
        failures.append(
            f"win not CI-separated: candidate loss {loss:.4f} not below strongest-baseline "
            f"loss CI lower edge {strongest_baseline_ci_low:.4f}"
        )
    # single-metric guard: a loss win that loses on (or cannot compute) calibration does not clear.
    cand_brier = float(candidate.get("brier_score", float("inf")))
    if not math.isfinite(cand_brier) or cand_brier > best_baseline_brier:
        failures.append(
            "calibration regression or non-finite Brier: not better-calibrated than the best baseline (single-metric win)"
        )
    if not external_available:
        failures.append("fixture-only: no external-lane confirmation of the win")
    if minimization_ok is not True:
        failures.append(
            "minimization regression or unverified: win must not depend on direct location"
        )
    if integrity_ok is not True:
        failures.append(
            "integrity regression or unverified: win must survive poisoning diagnostics"
        )

    return {
        "present": True,
        "model": candidate.get("model", "candidate"),
        "loss": loss,
        "loss_margin_over_strongest_baseline": margin,
        "strongest_baseline_ci_low": strongest_baseline_ci_low,
        "ci_separated": loss < strongest_baseline_ci_low,
        "clears": not failures,
        "failures": failures,
    }


def _oracle_summary(
    leakage_row: dict[str, object], strongest_baseline_loss: float
) -> dict[str, object]:
    leak_loss = float(leakage_row["mean_predictive_loss"])
    beats = leak_loss < strongest_baseline_loss
    excluded = bool(leakage_row["leakage"])
    # Verdict is DERIVED from the measured boolean so it cannot assert the negative
    # control fired when it did not (no green-by-construction prose).
    if beats and excluded:
        verdict = (
            "leakage probe achieves a lower loss than every honest baseline yet is "
            "excluded by the leakage guard, so a leakage win cannot clear the SOTA gate"
        )
    elif not beats:
        verdict = (
            "NEGATIVE CONTROL DID NOT FIRE: a real baseline matched or beat the leakage "
            "probe on this run, so the leakage exclusion was not exercised here"
        )
    else:
        verdict = "leakage probe beat the baselines but was NOT excluded — leakage guard failure"
    return {
        "leakage_probe_loss": leak_loss,
        "strongest_baseline_loss": strongest_baseline_loss,
        "leakage_probe_beats_baselines": beats,
        "leakage_probe_excluded": excluded,
        "verdict": verdict,
    }


def _task_contract(
    split: dict[str, object], smoothing: float, lane: str, external_available: bool
) -> dict[str, object]:
    return {
        "primary_metric": "prequential next-cell predictive loss (nats) under sparse reports",
        "secondary_metrics": [
            "Brier score",
            "expected calibration error (ECE)",
            "persistence-relative gap-filling skill",
            "runtime (seconds)",
            "location-minimization behavior",
            "poisoning/integrity robustness",
        ],
        "locked_split": "traveller-disjoint train/holdout; realization recorded per row",
        "smoothing": float(smoothing),
        "lane": lane,
        "external_lane_available": bool(external_available),
        "required_baseline_families": REQUIRED_BASELINE_FAMILIES,
        "win_margin_nats": MIN_WIN_MARGIN_NATS,
        "boundary": "synthetic-only; non-operational; cannot unlock real-mobility or deployment claims",
    }


def run_and_write_sota_benchmark(
    config_path: str | Path,
    out_dir: str | Path,
    observations_path: str | Path | None = None,
) -> dict[str, Path]:
    """Run the SOTA benchmark for ``config_path`` and write the four artifacts."""

    from .config import load_config
    from .observation_sources import load_observation_sequence

    config = load_config(config_path)
    sequence = load_observation_sequence(
        config, observations_path, requested_role="SOTA benchmark observations"
    )
    lane = "external" if config.fractalrabbit.execution != "fixture" else "fixture"

    # Minimization and integrity verdicts come from their own artifacts when present.
    # Sibling minimization/integrity artifacts live in the same runs directory.
    runs = Path(out_dir)
    minimization_ok = _minimization_ok(runs / "efe_location_frontier.json")
    integrity_ok = _integrity_ok(runs / "poisoning_robustness.json")
    full_stream = _full_stream_rows_from_artifacts(runs)

    payload = run_sota_benchmark(
        sequence,
        smoothing=config.inference.smoothing,
        lane=lane,
        external_available=(lane == "external"),
        minimization_ok=minimization_ok,
        integrity_ok=integrity_ok,
        extra_full_stream_rows=full_stream,
    )
    return _write_artifacts(payload, Path(out_dir))


def _write_artifacts(payload: dict[str, object], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    contract_path = out_dir / "sota_task_contract.json"
    matrix_path = out_dir / "sota_benchmark_matrix.json"
    leaderboard_path = out_dir / "sota_leaderboard.csv"
    gate_path = out_dir / "sota_claim_gate.json"

    contract_path.write_text(
        json.dumps(
            {"schema": SOTA_SCHEMA, **payload["task_contract"]},
            indent=2,
            sort_keys=True,
        )
        + "\n",  # type: ignore[index]
        encoding="utf-8",
    )
    matrix_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate_path.write_text(
        json.dumps(
            {
                "schema": SOTA_SCHEMA,
                "status": payload["status"],
                "claim_gate_reasons": payload["claim_gate_reasons"],
                "candidate_evaluation": payload["candidate_evaluation"],
                "minimization_ok": payload["minimization_ok"],
                "integrity_ok": payload["integrity_ok"],
                "used_injected_rows": payload["used_injected_rows"],
                "missing_baseline_families": payload["missing_baseline_families"],
                "false_claim_oracle": payload["false_claim_oracle"],
                "caveat": payload["caveat"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    leaderboard = payload["leaderboard"]  # type: ignore[index]
    header = [
        "model",
        "family",
        "lane",
        "locked_split_scored",
        "leakage",
        "mean_predictive_loss",
        "loss_ci_low",
        "loss_ci_high",
        "brier_score",
        "ece",
        "runtime_seconds",
    ]
    lines = [",".join(header)]
    for row in leaderboard:  # type: ignore[union-attr]
        lines.append(",".join(_csv_cell(row.get(col, "")) for col in header))
    leaderboard_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "task_contract": contract_path,
        "benchmark_matrix": matrix_path,
        "leaderboard": leaderboard_path,
        "claim_gate": gate_path,
    }


def _csv_cell(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _minimization_ok(path: Path) -> bool | None:
    """Location-minimization is satisfied when withholding direct location
    collapses localizability: location dominates the information budget and the
    residual localization without location is near zero (oversight-positive)."""

    data = _load_json(path)
    if data is None:
        return None
    share = data.get("location_joint_share")
    residual = data.get("residual_localized_without_location")
    if not isinstance(share, int | float) or not isinstance(residual, int | float):
        return None
    return bool(float(share) >= 0.5 and float(residual) <= 0.5)


def _integrity_ok(path: Path) -> bool | None:
    """Integrity diagnostics pass when the poisoning monitor detects gross and
    moderate stream tampering (defensive self-monitoring works)."""

    data = _load_json(path)
    if data is None:
        return None
    gross = data.get("gross_poisoning_detected")
    moderate = data.get("moderate_poisoning_detected")
    if not isinstance(gross, bool) or not isinstance(moderate, bool):
        return None
    return bool(gross and moderate)


def _full_stream_rows_from_artifacts(runs: Path) -> list[dict[str, object]]:
    """Collect heavier candidate losses already produced by other analyses."""

    rows: list[dict[str, object]] = []
    # (model, family, artifact, loss_key) — losses already produced and tested
    # by their own analyses; recorded here as full-stream (not locked-split) rows.
    sources = [
        (
            "neural_logistic",
            "neural_sequence",
            "neural_baseline.json",
            "neural_logistic_mean_predictive_loss",
        ),
        (
            "switching_hmm",
            "classical_sequence",
            "neural_baseline.json",
            "switching_hmm_mean_predictive_loss",
        ),
        (
            "particle_state_space",
            "classical_sequence",
            "neural_baseline.json",
            "particle_state_space_mean_predictive_loss",
        ),
        (
            "latent_regime_hmm",
            "classical_sequence",
            "latent_regime_hmm.json",
            "best_honest_heldout_loss",
        ),
    ]
    for model, family, fname, loss_key in sources:
        data = _load_json(runs / fname)
        if data is None:
            continue
        loss = data.get(loss_key)
        if isinstance(loss, int | float):
            rows.append(
                {
                    "model": model,
                    "family": family,
                    "current_role": "full-stream prequential score from its own artifact (not locked-split)",
                    "lane": "full_stream",
                    "locked_split_scored": False,
                    "leakage": False,
                    "mean_predictive_loss": float(loss),
                    "brier_score": float("nan"),
                    "ece": float("nan"),
                    "runtime_seconds": float("nan"),
                }
            )
    return rows
