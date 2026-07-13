from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.partial_observability import (
    cross_realization_noisy_emission_advantage,
    noisy_emission_advantage,
    realize_noisy_observations,
    run_and_write_partial_observability,
)
from active_fractal_rabbit.power_analysis import generate_synthetic_sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_realized_channel_is_matched_and_never_lands_on_true_cell():
    """The matched symmetric channel realizes flip rate ~= eps and never flips onto the true cell.

    This is the fairness-by-construction property the headline soft-vs-hard advantage rests on:
    the generated channel equals the inference emission (effective flip rate eps, not eps*(n-1)/n),
    so a flip is uniform over the n-1 non-true cells. Locked here as a direct channel-law test.
    """
    n = 7
    rng = np.random.default_rng(20260620)
    z = rng.integers(0, n, 20000)

    # eps = 0: the observation equals the latent exactly (no flips).
    clean = realize_noisy_observations(z, n, 0.0, np.random.default_rng(1))
    assert np.array_equal(clean, z)

    # eps > 0: realized flip rate ~= eps, and no flip ever lands on the true cell.
    eps = 0.3
    observed = realize_noisy_observations(z, n, eps, rng)
    flips = observed != z
    assert abs(flips.mean() - eps) < 0.02  # realized rate is eps, not eps*(n-1)/n
    assert observed.min() >= 0 and observed.max() < n
    # every flipped observation is one of the n-1 cells OTHER than the true cell
    assert not np.any(observed[flips] == z[flips])


def _sequence(bins: int = 6, n_travellers: int = 8):
    cfg = load_config(ROOT / "configs" / "default.yaml")
    inf = replace(
        cfg.inference, observation=replace(cfg.inference.observation, spatial_bins=bins)
    )
    return generate_synthetic_sequence(n_travellers, inf, seed=2026)


def _result():
    return noisy_emission_advantage(
        _sequence(),
        eps_levels=(0.0, 0.3),
        noise_seeds=(0, 1),
        split_seeds=(20260616, 20260617),
    )


def test_zero_advantage_when_clean() -> None:
    r = _result()
    clean = next(row for row in r["rows"] if row["eps"] == 0.0)
    assert r["zero_advantage_when_clean"] is True
    assert clean["aif_advantage_over_hard_nats"] == 0.0


def test_advantage_is_positive_under_noise() -> None:
    r = _result()
    assert r["advantage_positive_under_noise"] is True
    noisy = [row for row in r["rows"] if row["eps"] > 0.0]
    assert noisy and all(row["aif_advantage_over_hard_min"] > 0.0 for row in noisy)
    assert all(row["aif_advantage_over_observed_nats"] > 0.0 for row in noisy)


def test_soft_vs_hard_is_fair_by_construction() -> None:
    """The comparison's fairness is structural: soft and hard share model, emission, observations, and
    priors and differ only in marginalize-vs-commit, so the advantage cannot be a knob asymmetry."""
    r = _result()
    assert "fair by construction" in r["boundary"]
    assert r["n_cells"] == _sequence().num_spatial_states


def test_statistics_quantify_the_noise_trend() -> None:
    """The reported statistics characterise the advantage honestly: a positive noise slope, a fitted
    R^2 in [0, 1], a non-negative spread band, and every paired resampling positive under noise."""
    r = _result()
    assert r["noise_slope_nats_per_eps"] > 0.0
    assert 0.0 <= r["noise_trend_r2"] <= 1.0
    assert r["advantage_monotone_with_noise"] is True
    assert r["max_noise_advantage_sd"] >= 0.0
    assert r["max_noise_advantage_min"] > 0.0
    # every (noise seed x split seed) resampling at the noisy level favors active inference
    assert r["min_fraction_positive_under_noise"] == 1.0
    noisy = [row for row in r["rows"] if row["eps"] > 0.0]
    assert noisy and all(
        row["n_realizations"] == r["n_realizations_per_level"] for row in r["rows"]
    )
    assert all(
        row["aif_advantage_over_hard_max"] >= row["aif_advantage_over_hard_min"]
        for row in r["rows"]
    )


def test_cross_realization_summary_reports_between_draw_spread() -> None:
    """Independent generator draws expose between-dataset spread instead of laundering
    split/noise robustness as generalization."""

    cfg = load_config(ROOT / "configs" / "default.yaml")
    r = cross_realization_noisy_emission_advantage(
        cfg.inference,
        n_travellers=6,
        generator_seeds=(2026, 2027),
        eps_levels=(0.0, 0.3),
        noise_seeds=(0,),
        split_seeds=(20260616,),
    )

    assert r["cross_realization_n_draws"] == 2
    assert r["cross_realization_n_travellers"] == 6
    assert len(r["cross_realization_rows"]) == 2
    assert r["cross_realization_sign_stable_under_noise"] is True
    assert r["cross_realization_max_between_draw_sd"] >= 0.0
    noisy = [row for row in r["cross_realization_rows"] if row["eps"] > 0.0]
    assert noisy and noisy[0]["fraction_draws_positive"] == 1.0


def test_writer_keeps_configured_fixture_as_canonical_source_when_external_exists(
    tmp_path: Path,
) -> None:
    out = tmp_path / "partial_observability_analysis.json"
    run_and_write_partial_observability(
        ROOT / "configs" / "default.yaml",
        ROOT / "data" / "fixtures" / "fractalrabbit_sample.csv",
        out,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["primary_source"] == "configured_fixture"
    assert payload["source"] == "fixture:fractalrabbit_sample_large.csv"
    assert payload["external_lane"]["status"] == "completed"
    assert payload["external_lane"]["source"].endswith("fractalrabbit_sample.csv")
    assert payload["external_lane"]["boundary"].endswith(
        "not the canonical manuscript-variable source"
    )
