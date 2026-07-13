"""Structural diagnosis of WHY AIF loses on the latent-regime-switching lane.

``regime_mechanism_synthesis.py`` classifies ``latent_regime_switch_035`` as
``loses_to_fair_baseline``: AIF's best candidate scores worse than ``online_base_rate``
pooled over the whole holdout. That answers WHERE and HOW MUCH, but not WHY the loss
happens where it does within the trajectory.

``latent_regime_switch_035`` (built by ``build_regime_stressor_partial_observability_dataset``
-> ``generate_regime_stressor_waypoints``) is the one regime in the grid whose generator
has an explicit, named 3-phase movement-mode schedule per traveller (see
``sota_stressors.REGIME_STRESSOR_PHASE_BOUNDARIES``/``_regime_stressor_phase_parameters``):
a low-variance, high-return "commute" segment, a high-variance, low-return "explore"
segment (movement_scale jumps 2.0 -> 12.0, a 6x increase), and an intermediate
"return_stabilization" segment. No other regime in the grid has this structure, so this
diagnostic is scoped to this one regime rather than generalized across all five.

This module reconstructs, for every scored holdout step, which movement-mode segment it
falls in (from the step's position within its traveller's held-out sequence, verified
against the generator's own per-traveller report count rather than assumed), and computes
a traveller-clustered-bootstrap CI on the AIF-vs-strongest-fair loss difference within
each segment separately. If AIF's disadvantage concentrates in the high-variance
"explore" segment specifically, that is structural, quantitative evidence that AIF's
fixed generative model (a single learned transition matrix, not segment-aware) is
mis-specified for the more volatile regime rather than uniformly worse everywhere.

Regime-local synthetic evidence only; see ``TODO.md``'s standing claim locks. Never
asserts empirical, operational, or global-SOTA validity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .aif_regime_search import (
    DEFAULT_AIF_REGIME_N_TRAVELLERS,
    DEFAULT_AIF_REGIME_NOISE_SEED,
    DEFAULT_AIF_REGIME_REALIZATION_SEEDS,
    DEFAULT_AIF_REGIME_SPLIT_SEED,
    build_aif_regime_specs,
    build_regime_dataset,
    score_regime_realization,
)
from .power_analysis import traveller_clustered_bootstrap_stats
from .sota_program_harness import DEFAULT_SPATIAL_BINS, build_inference_config
from .sota_stressors import (
    DEFAULT_REGIME_REPORTS,
    REGIME_STRESSOR_PHASE_SEGMENTS,
    phase_segment_for_phase,
)

REGIME_SWITCH_PHASE_DIAGNOSTIC_SCHEMA = (
    "active-fractal-rabbit-regime-switch-phase-diagnostic-v1"
)
REGIME_SWITCH_REGIME_ID = "latent_regime_switch_035"

#: Matches regime_mechanism_synthesis.DEFAULT_MECHANISM_BOOT: a directional diagnostic
#: layered on the canonical gate, not an independent publication-grade CI.
DEFAULT_PHASE_BOOT = 300


def _step_phase_segments(step_travellers: np.ndarray, n_reports: int) -> np.ndarray:
    """Map each scored step to the movement-mode segment that actually produced it.

    ``score_candidate`` walks the dataset in row order (travellers grouped together,
    each traveller's own rows in strict generation order — rows are sorted by
    ``(traveller_id, days)`` and ``days`` only increases within a traveller), and only
    appends a step for non-first observations. So the k-th (0-indexed) occurrence of a
    given traveller id in ``step_travellers`` is exactly the generator's observation
    index ``idx = k + 1`` (idx=0 is the skipped first observation) for that traveller,
    with no need to thread a separate row-index field through ``ScoredCandidate``.

    The phase used to bucket step k is ``k / (n_reports - 1)``, NOT
    ``(k + 1) / (n_reports - 1)``: in ``generate_regime_stressor_waypoints``, the row at
    generator index ``idx`` is appended FIRST (using the position already set by the
    PRIOR iteration's transition), and only afterward is ``phase = idx / (n_reports-1)``
    computed and used to advance the position into row ``idx + 1``. So the movement
    that produced row ``idx`` (idx >= 1) - and hence the difficulty
    ``score_candidate`` actually scores at that step, predicting z_idx from history - was
    governed by ``phase(idx - 1)``, not ``phase(idx)``. Attributing by ``phase(idx)``
    (a prior version of this function did exactly that) buckets every step by the phase
    of the transition LEAVING it rather than the one that produced it, a real off-by-one
    confirmed by a Forge/RedTeam cross-vendor audit and independently re-derived here:
    the row for idx=k+1 was produced using phase((k+1)-1) = phase(k) = k/(n_reports-1).
    """

    travs = np.asarray(step_travellers, dtype=int)
    segments = np.empty(travs.shape[0], dtype=object)
    running_count: dict[int, int] = {}
    for i, tid in enumerate(travs):
        k = running_count.get(int(tid), 0)
        running_count[int(tid)] = k + 1
        phase = k / max(n_reports - 1, 1)
        segments[i] = phase_segment_for_phase(phase)

    # Invariant check, not an assumption: every held-out traveller must contribute
    # exactly n_reports - 1 steps (one per non-first observation). A mismatch would mean
    # the position-from-occurrence-count reconstruction above is wrong for this dataset
    # (e.g. a future generator change dropping observations), and the phase labels this
    # function returns would then be silently wrong rather than loudly wrong.
    expected_steps_per_traveller = n_reports - 1
    for tid, count in running_count.items():
        if count != expected_steps_per_traveller:
            raise ValueError(
                f"traveller {tid} contributed {count} scored steps, expected "
                f"{expected_steps_per_traveller} (n_reports={n_reports}); the "
                "position-within-traveller phase reconstruction assumes every held-out "
                "traveller's steps are contiguous and in generation order, which no "
                "longer holds for this dataset"
            )
    return segments


def run_regime_switch_phase_diagnostic(
    config_path: str | Path = "configs/default.yaml",
    *,
    n_travellers: int = DEFAULT_AIF_REGIME_N_TRAVELLERS,
    realization_seed: int = DEFAULT_AIF_REGIME_REALIZATION_SEEDS[0],
    split_seed: int = DEFAULT_AIF_REGIME_SPLIT_SEED,
    noise_seed: int | None = None,
    n_reports: int = DEFAULT_REGIME_REPORTS,
    n_boot: int = DEFAULT_PHASE_BOOT,
    boot_seed: int = 11,
) -> dict[str, object]:
    """Per-movement-segment AIF-vs-fair loss breakdown on the regime-switching lane.

    Reuses the exact same regime-search scoring path (``build_regime_dataset`` +
    ``score_regime_realization``) so the AIF/fair candidates and their per-step losses
    are identical to what ``aif_regime_search.json``/``regime_mechanism_synthesis.json``
    already report pooled; this module only re-slices those same step losses by
    movement-mode segment instead of re-deriving them.
    """

    resolved_noise_seed = (
        noise_seed
        if noise_seed is not None
        else DEFAULT_AIF_REGIME_NOISE_SEED + int(realization_seed % 997)
    )
    inference_config = build_inference_config(
        config_path, spatial_bins=DEFAULT_SPATIAL_BINS
    )
    specs = build_aif_regime_specs()
    spec = specs[REGIME_SWITCH_REGIME_ID]
    dataset = build_regime_dataset(
        REGIME_SWITCH_REGIME_ID,
        inference_config,
        n_travellers=n_travellers,
        realization_seed=realization_seed,
        split_seed=split_seed,
        noise_seed=resolved_noise_seed,
    )
    result = score_regime_realization(
        spec,
        dataset,
        realization_seed=realization_seed,
        split_seed=split_seed,
        noise_seed=resolved_noise_seed,
        n_boot=n_boot,
    )
    aif_scored = result.scored[result.best_aif_name]
    fair_scored = result.scored[result.strongest_fair_name]
    if not np.array_equal(aif_scored.step_travellers, fair_scored.step_travellers):
        raise ValueError(
            "best AIF candidate and strongest fair baseline were scored on different "
            "holdout steps; cannot pair per-segment losses"
        )

    segments = _step_phase_segments(aif_scored.step_travellers, n_reports)
    aif_losses = np.asarray(aif_scored.step_losses, dtype=float)
    fair_losses = np.asarray(fair_scored.step_losses, dtype=float)
    travs = np.asarray(aif_scored.step_travellers, dtype=int)
    # positive => AIF better, matching sota_program.paired_loss_difference_ci's own
    # documented sign convention (loss(fair) - loss(AIF), b_wins when positive).
    diffs = fair_losses - aif_losses

    segment_rows: list[dict[str, object]] = []
    for segment_index, segment_name in enumerate(REGIME_STRESSOR_PHASE_SEGMENTS):
        mask = segments == segment_name
        n_steps = int(mask.sum())
        # Every held-out traveller appears in every segment (phase is idx-based, not
        # traveller-dependent), so the three segments' traveller universes are
        # identical - reusing one boot_seed would give byte-identical cluster-resample
        # picks across all three calls (a cross-vendor audit finding), making the three
        # CIs paired rather than independent draws. Offsetting by segment_index keeps
        # each segment's bootstrap distribution independent while staying deterministic.
        stats = traveller_clustered_bootstrap_stats(
            diffs[mask], travs[mask], n_boot=n_boot, seed=boot_seed + segment_index
        )
        segment_rows.append(
            {
                "segment": segment_name,
                "n_steps": n_steps,
                "n_travellers": stats["n_travellers"],
                "aif_mean_loss": float(np.mean(aif_losses[mask])),
                "fair_mean_loss": float(np.mean(fair_losses[mask])),
                "mean_advantage_nats": stats["mean"],
                "clustered_ci_low": stats["ci_low"],
                "clustered_ci_high": stats["ci_high"],
                "one_sided_p_aif_better": stats["one_sided_p"],
            }
        )

    worst_segment = min(segment_rows, key=lambda row: row["mean_advantage_nats"])
    best_segment = max(segment_rows, key=lambda row: row["mean_advantage_nats"])
    payload: dict[str, object] = {
        "schema": REGIME_SWITCH_PHASE_DIAGNOSTIC_SCHEMA,
        "status": "completed",
        "regime_id": REGIME_SWITCH_REGIME_ID,
        "best_aif_name": result.best_aif_name,
        "strongest_fair_name": result.strongest_fair_name,
        "pooled_mean_advantage_nats": result.mean_difference_nats,
        "n_reports_per_traveller": int(n_reports),
        "segments": segment_rows,
        "worst_segment": worst_segment["segment"],
        "best_segment": best_segment["segment"],
        # Precise, not a coarse "concentrated in one segment" binary: the real per-segment
        # CIs (verified against this run) all exclude zero on the AIF-worse side, so the
        # honest characterization is "uniform but uneven" - AIF loses everywhere, not just
        # in one segment, but by a different magnitude per segment. A prior draft of this
        # field asked only "is one segment negative and another positive" and would have
        # reported False here despite a real ~1.8x magnitude spread across segments.
        "uniform_disadvantage": (
            uniform_disadvantage := bool(
                all(row["clustered_ci_high"] < 0.0 for row in segment_rows)
            )
        ),
        # Only meaningful (a same-signed magnitude comparison) when every segment shows
        # the same directional disadvantage; otherwise the ratio of two differently-signed
        # numbers wouldn't mean "how much worse", so it's left unreported.
        "worst_to_best_magnitude_ratio": (
            abs(worst_segment["mean_advantage_nats"])
            / abs(best_segment["mean_advantage_nats"])
            if uniform_disadvantage and best_segment["mean_advantage_nats"] != 0.0
            else None
        ),
        "caveat": (
            "Structural diagnostic layered on top of the canonical regime-search gate: a "
            "single generator realization and a smaller bootstrap resample count than "
            "aif_regime_search.json's own multi-realization average. Scoped to "
            f"{REGIME_SWITCH_REGIME_ID} only, the one regime whose generator has a named "
            "movement-mode phase schedule. Regime-local synthetic evidence only; supports "
            "no state-of-the-art, operational, or real-mobility claim."
        ),
    }
    return payload


def run_and_write_regime_switch_phase_diagnostic(
    config_path: str | Path = "configs/default.yaml",
    out_path: str | Path = "output/runs/regime_switch_phase_diagnostic.json",
    *,
    n_travellers: int = DEFAULT_AIF_REGIME_N_TRAVELLERS,
    realization_seed: int = DEFAULT_AIF_REGIME_REALIZATION_SEEDS[0],
    split_seed: int = DEFAULT_AIF_REGIME_SPLIT_SEED,
    noise_seed: int | None = None,
    n_reports: int = DEFAULT_REGIME_REPORTS,
    n_boot: int = DEFAULT_PHASE_BOOT,
    boot_seed: int = 11,
) -> Path:
    payload = run_regime_switch_phase_diagnostic(
        config_path,
        n_travellers=n_travellers,
        realization_seed=realization_seed,
        split_seed=split_seed,
        noise_seed=noise_seed,
        n_reports=n_reports,
        n_boot=n_boot,
        boot_seed=boot_seed,
    )
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
