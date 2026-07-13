"""Regime mechanism synthesis: WHERE active inference wins or loses, HOW MUCH, and WHY.

Three analyses already exist in this project but were never joined into one artifact:

- ``aif_regime_search.py`` reports, per synthetic regime, whether the best AIF-family
  candidate beats the strongest fair baseline, by how much (mean loss difference, nats),
  and with what clustered-bootstrap CI and one-sided p-value. This answers WHERE and part
  of HOW MUCH.
- ``multiplicity_ledger.py`` reports whether that per-regime comparison survives family-wide
  Holm correction across every comparison this benchmark makes. This answers the rest of
  HOW MUCH: is the effect real once multiplicity is accounted for.
- ``confound_controls.py`` implements this project's confound-control battery (dominant-cell,
  first-order-pooled-Markov, return-boosted-soft, oracle-previous-cell) and a mobility check,
  but was applied only to the default fixture, never per-regime. This is the sharpest
  available WHY instrument this project has: does a nominal AIF win survive naive-baseline
  and pooling-baseline scrutiny, or is it a confound?

This module runs the confound battery and mobility check on each regime's own dataset (the
same dataset ``aif_regime_search.py`` builds and scores, with the same realization-0
noise/split seeds ``run_regime_search`` itself derives), cross-references the regime's
existing magnitude/CI/multiplicity standing, and synthesizes one mechanism_verdict + a
one-line plain-language mechanism_explanation per regime - a single artifact that answers
"where does AIF do better or worse, how much, and why" instead of leaving the reader to
manually cross-reference three separate JSON files.

The confound battery runs on the SAME noisy observations the AIF candidate actually sees
(``dataset.observed``), not the true latent cells (``dataset.sequence.spatial_states``) -
using the latter would make ``first_order_pooled``/``return_boosted_soft`` oracle-fed
baselines rather than the "inferred previous cell, fair" comparison confound_controls.py's
own docstring promises, which would make "AIF fails the confound battery" true by
construction on any regime with real emission noise regardless of AIF quality (found by a
Forge cross-vendor audit the first time this module put confound_battery into production
use - it had previously only run against the noise-free default fixture in tests). The
battery's traveller split is also locked to the exact same holdout_fraction/seed the AIF
scoring itself used (read from ``dataset.split``), not confound_battery's own unrelated
default, so both sides of every comparison are computed over the same holdout population.

Every verdict remains regime-local synthetic evidence; see ``TODO.md``'s standing claim
locks. This module never asserts empirical, operational, or global-SOTA validity.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .aif_regime_search import (
    CANONICAL_AIF_REGIMES,
    DEFAULT_AIF_REGIME_N_TRAVELLERS,
    DEFAULT_AIF_REGIME_NOISE_SEED,
    DEFAULT_AIF_REGIME_REALIZATION_SEEDS,
    DEFAULT_AIF_REGIME_SPLIT_SEED,
    build_aif_regime_specs,
    build_regime_dataset,
    score_regime_realization,
)
from .confound_controls import clears_confounds, confound_battery, mobility_summary
from .sota_program_harness import DEFAULT_SPATIAL_BINS, build_inference_config

REGIME_MECHANISM_SCHEMA = "active-fractal-rabbit-regime-mechanism-synthesis-v1"

#: Confound-battery bootstrap resample count for this diagnostic. Deliberately smaller
#: than aif_regime_search's own DEFAULT_AIF_REGIME_N_BOOT (2000): this synthesis needs a
#: directional confound-battery verdict on top of the canonical gate, not an independent
#: publication-grade CI, so it can run across all 5 regimes in seconds, not minutes.
DEFAULT_MECHANISM_BOOT = 300

#: Every verdict class ``_mechanism_verdict`` can return. ``verdict_counts`` in the
#: payload always includes all of these (zero-filled when absent this run) rather than
#: only the classes that occurred, so a manuscript token bound to e.g. ``robust_win``'s
#: count resolves to 0 instead of a missing-key provenance error when no regime
#: currently earns that verdict.
MECHANISM_VERDICT_CLASSES: tuple[str, ...] = (
    "trivial_by_construction",
    "confined_uninterpretable",
    "null_no_signal",
    "loses_to_fair_baseline",
    "confound_explained",
    "nominal_win_not_multiplicity_robust",
    "robust_win",
)


def _mechanism_verdict(
    *,
    eligible_for_regime_sota: bool,
    mean_advantage_nats: float,
    ci_low: float,
    ci_high: float,
    holm_survives: bool,
    clears_battery: bool,
    is_mobile: bool,
) -> tuple[str, str]:
    """Classify one regime's AIF-vs-fair standing into (mechanism_verdict, explanation).

    Precedence, checked in order: a negative-control regime is trivial regardless of its
    numbers; a spatially confined regime makes any win uninterpretable before any
    comparison is even meaningful; a losing or directionally-null regime needs no confound
    check (there is no win to explain away); a winning regime is then split on whether it
    clears the confound battery and, only then, whether it survives family-wise multiplicity
    correction. This ordering matches how this project has always reasoned about its own
    apparent wins (mobility check first, then confound battery, then multiplicity) - see
    confound_controls.py's module docstring and confound_taxonomy.py's registry.
    """

    if not eligible_for_regime_sota:
        return (
            "trivial_by_construction",
            "Negative-control regime (e.g. a zero-noise emission channel): the AIF/fair "
            "distinction collapses by construction, not by a modeling advantage or "
            "disadvantage, so no mechanism claim applies.",
        )
    if not is_mobile:
        return (
            "confined_uninterpretable",
            "Travellers are too spatially confined on this regime for a model comparison "
            "to mean anything - any model wins by information congruence with the confined "
            "occupancy structure alone, not by inference quality.",
        )
    straddles_zero = ci_low <= 0.0 <= ci_high
    if straddles_zero:
        return (
            "null_no_signal",
            "The traveller-clustered confidence interval touches or straddles zero: no directional "
            "AIF advantage or disadvantage is established on this regime.",
        )
    if mean_advantage_nats <= 0.0:
        return (
            "loses_to_fair_baseline",
            "AIF's best candidate scores WORSE than the strongest implemented fair "
            "baseline on this regime; the baseline's inductive bias fits this regime's "
            "structure better than the AIF generative model does.",
        )
    if not clears_battery:
        return (
            "confound_explained",
            "AIF nominally beats the fair-baseline roster, but its winning candidate does "
            "NOT beat this project's confound battery (dominant-cell / pooled-Markov / "
            "return-boosted-soft) - the apparent win reduces to a confound, not capability.",
        )
    if not holm_survives:
        return (
            "nominal_win_not_multiplicity_robust",
            "AIF's best candidate beats both the fair-baseline roster and the confound "
            "battery, but the win does not survive Holm correction across the full "
            "comparison family - a multiplicity artifact, not a robust effect.",
        )
    return (
        "robust_win",
        "AIF's best candidate beats the fair-baseline roster, clears the confound "
        "battery, and survives family-wise Holm correction on this regime - the strongest "
        "standing this benchmark's methodology can currently award, still regime-local "
        "synthetic evidence only.",
    )


def run_regime_mechanism_synthesis(
    config_path: str | Path = "configs/default.yaml",
    *,
    regime_ids: tuple[str, ...] = CANONICAL_AIF_REGIMES,
    n_travellers: int = DEFAULT_AIF_REGIME_N_TRAVELLERS,
    realization_seed: int = DEFAULT_AIF_REGIME_REALIZATION_SEEDS[0],
    split_seed: int = DEFAULT_AIF_REGIME_SPLIT_SEED,
    noise_seed: int | None = None,
    n_boot: int = DEFAULT_MECHANISM_BOOT,
    regime_search_path: str | Path = "output/runs/aif_regime_search.json",
    multiplicity_ledger_path: str | Path = "output/runs/multiplicity_ledger.json",
) -> dict[str, object]:
    """Build the WHERE / HOW MUCH / WHY mechanism map across the AIF regime grid."""

    inference_config = build_inference_config(
        config_path, spatial_bins=DEFAULT_SPATIAL_BINS
    )
    specs = build_aif_regime_specs()

    # Reproduces run_regime_search's own realization-0 noise_seed derivation exactly
    # (aif_regime_search.py: `noise_seed = DEFAULT_AIF_REGIME_NOISE_SEED +
    # int(realization_seed % 997)`) when the caller doesn't override it, so this module's
    # own dataset (and the best_aif_loss/strongest_fair_loss it re-scores) is the SAME
    # dataset the canonical regime-search artifact's numbers came from, not a
    # differently-noised one the module's docstring only claimed to match.
    resolved_noise_seed = (
        noise_seed
        if noise_seed is not None
        else DEFAULT_AIF_REGIME_NOISE_SEED + int(realization_seed % 997)
    )

    regime_search_file = Path(regime_search_path)
    regime_search = (
        json.loads(regime_search_file.read_text(encoding="utf-8"))
        if regime_search_file.is_file()
        else {}
    )
    regime_summaries = (
        regime_search.get("regime_summaries", {})
        if isinstance(regime_search, dict)
        else {}
    )

    ledger_file = Path(multiplicity_ledger_path)
    multiplicity = (
        json.loads(ledger_file.read_text(encoding="utf-8"))
        if ledger_file.is_file()
        else {}
    )
    holm_by_comparison = {
        str(row.get("comparison_id")): bool(row.get("holm_survives"))
        for row in multiplicity.get("comparisons", [])
        if isinstance(multiplicity, dict)
    }

    rows: list[dict[str, object]] = []
    for regime_id in regime_ids:
        if regime_id not in specs:
            raise KeyError(
                f"unknown AIF regime {regime_id!r}; expected one of {sorted(specs)}"
            )
        spec = specs[regime_id]
        dataset = build_regime_dataset(
            regime_id,
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
        mobility = mobility_summary(dataset.sequence)
        # The confound battery must see what the AIF candidate actually sees (the noisy
        # `dataset.observed`), not the true latent cells (`dataset.sequence.spatial_states`)
        # - passing the latter turns first_order_pooled/return_boosted_soft into oracle-fed
        # baselines despite their "inferred previous cell, fair" docstring, which would make
        # AIF fail the battery near-unconditionally on any regime with real emission noise,
        # independent of AIF quality. The battery's split is locked to the exact
        # holdout_fraction/seed the AIF scoring itself used (`dataset.split`), not
        # confound_battery's own unrelated default, so both sides share one holdout population.
        noisy_sequence = replace(dataset.sequence, spatial_states=dataset.observed)
        battery = confound_battery(
            noisy_sequence,
            holdout_fraction=float(dataset.split["holdout_fraction"]),
            seed=int(dataset.split["seed"]),
        )
        battery_verdict = clears_confounds(result.best_aif_loss, battery)

        summary = (
            regime_summaries.get(regime_id, {})
            if isinstance(regime_summaries, dict)
            else {}
        )
        # Fallback values (used only when regime_id is absent from the live regime_search
        # artifact) come from `result`'s own paired_loss_difference_ci(strongest_fair,
        # best_aif, ...), whose docstring is explicit: positive `mean_difference_nats` means
        # candidate B (best_aif) is better - the SAME positive-means-AIF-advantage sign
        # convention regime_summaries' mean_advantage_nats already uses. Negating it (as a
        # prior version of this function did) silently flipped verdicts for any un-cached
        # regime; there's no sign flip here.
        mean_advantage = float(
            summary.get("mean_advantage_nats", result.mean_difference_nats)
        )
        ci_low = float(summary.get("min_clustered_ci_low", result.clustered_ci_low))
        ci_high = float(summary.get("max_clustered_ci_high", result.clustered_ci_high))
        comparison_id = f"aif_regime::{regime_id}"
        holm_survives = holm_by_comparison.get(comparison_id, False)

        verdict, explanation = _mechanism_verdict(
            eligible_for_regime_sota=spec.eligible_for_regime_sota,
            mean_advantage_nats=mean_advantage,
            ci_low=ci_low,
            ci_high=ci_high,
            holm_survives=holm_survives,
            clears_battery=bool(battery_verdict["clears_confounds"]),
            is_mobile=bool(mobility["is_mobile"]),
        )

        rows.append(
            {
                "regime_id": regime_id,
                "description": spec.description,
                "eligible_for_regime_sota": spec.eligible_for_regime_sota,
                "best_aif_name": result.best_aif_name,
                "best_aif_loss": result.best_aif_loss,
                "strongest_fair_name": result.strongest_fair_name,
                "strongest_fair_loss": result.strongest_fair_loss,
                "mean_advantage_nats": mean_advantage,
                "clustered_ci_low": ci_low,
                "clustered_ci_high": ci_high,
                "holm_survives": holm_survives,
                "mobility": {
                    "is_mobile": bool(mobility["is_mobile"]),
                    "mean_distinct_cells": float(mobility["mean_distinct_cells"]),
                    "confined_fraction": float(mobility["confined_fraction"]),
                },
                "confound_battery": {
                    "dominant_cell_loss": float(battery["dominant_cell"]),
                    "first_order_pooled_loss": float(battery["first_order_pooled"]),
                    "return_boosted_soft_loss": float(battery["return_boosted_soft"]),
                    "oracle_prev_loss": float(battery["oracle_prev"]),
                    "beats": battery_verdict["beats"],
                    "clears_confounds": bool(battery_verdict["clears_confounds"]),
                    "gap_to_oracle_nats": float(battery_verdict["gap_to_oracle_nats"]),
                },
                "mechanism_verdict": verdict,
                "mechanism_explanation": explanation,
            }
        )

    verdict_counts: dict[str, int] = dict.fromkeys(MECHANISM_VERDICT_CLASSES, 0)
    for row in rows:
        key = str(row["mechanism_verdict"])
        verdict_counts[key] = verdict_counts.get(key, 0) + 1

    payload: dict[str, object] = {
        "schema": REGIME_MECHANISM_SCHEMA,
        "status": "completed",
        "n_regimes": len(rows),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "regimes": rows,
        "caveat": (
            "Synthetic-only mechanism map over named regimes. A 'robust_win' verdict "
            "would still be regime-local synthetic evidence, never global SOTA, "
            "empirical, or operational validity - see TODO.md's standing claim locks. "
            "The confound-battery and mobility diagnostics here use a single realization "
            "and a smaller bootstrap resample count than aif_regime_search.json's own "
            "multi-realization gate, so this is a mechanism DIAGNOSTIC layered on top of "
            "the canonical regime-search gate, not a replacement for it."
        ),
    }
    return payload


def run_and_write_regime_mechanism_synthesis(
    config_path: str | Path = "configs/default.yaml",
    out_path: str | Path = "output/runs/regime_mechanism_synthesis.json",
    *,
    regime_ids: tuple[str, ...] = CANONICAL_AIF_REGIMES,
    n_travellers: int = DEFAULT_AIF_REGIME_N_TRAVELLERS,
    realization_seed: int = DEFAULT_AIF_REGIME_REALIZATION_SEEDS[0],
    split_seed: int = DEFAULT_AIF_REGIME_SPLIT_SEED,
    noise_seed: int | None = None,
    n_boot: int = DEFAULT_MECHANISM_BOOT,
    regime_search_path: str | Path = "output/runs/aif_regime_search.json",
    multiplicity_ledger_path: str | Path = "output/runs/multiplicity_ledger.json",
) -> Path:
    """Run the synthesis and write it to ``out_path``."""

    payload = run_regime_mechanism_synthesis(
        config_path,
        regime_ids=regime_ids,
        n_travellers=n_travellers,
        realization_seed=realization_seed,
        split_seed=split_seed,
        noise_seed=noise_seed,
        n_boot=n_boot,
        regime_search_path=regime_search_path,
        multiplicity_ledger_path=multiplicity_ledger_path,
    )
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
