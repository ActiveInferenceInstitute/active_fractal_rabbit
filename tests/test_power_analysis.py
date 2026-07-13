"""Tests for the fair-frontier power / regime analysis (session 20).

Locks the honest, clustered-bootstrap result: the fair AIF advantage is regime-bound —
large with few travellers, vanishing at scale under a clean emission, re-emerging under
observation noise at scale. No mocks; real deterministic synthetic sequences.
"""

from __future__ import annotations

import json

import numpy as np

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.power_analysis import (
    aif_vs_soft_clustered,
    fair_power_regime_map,
    generate_synthetic_sequence,
    run_and_write_power_analysis,
    traveller_clustered_bootstrap_ci,
    traveller_clustered_bootstrap_stats,
)

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# reduced grid keeps the test fast while still spanning the regime corners
_COUNTS = (4, 16)
_SM = (0.05, 2.0)


def _inference():
    return load_config(ROOT / "configs" / "default.yaml").inference


def test_generator_is_deterministic_and_scales() -> None:
    inf = _inference()
    a = generate_synthetic_sequence(8, inf, seed=2026)
    b = generate_synthetic_sequence(8, inf, seed=2026)
    assert np.array_equal(np.asarray(a.spatial_states), np.asarray(b.spatial_states))
    travellers = {r.traveller_id for r in a.observations}
    assert travellers == set(range(8))


def test_clustered_bootstrap_wider_than_iid_on_clustered_data() -> None:
    """Non-vacuity: with strong within-traveller correlation, the traveller-clustered CI
    is markedly WIDER than a naive per-step CI — the whole reason it is the honest unit."""
    rng = np.random.default_rng(0)
    # two travellers, each with a near-constant (highly correlated) per-step diff
    diffs = np.concatenate(
        [
            np.full(200, 0.4) + rng.normal(0, 0.01, 200),
            np.full(200, -0.2) + rng.normal(0, 0.01, 200),
        ]
    )
    travs = np.array([0] * 200 + [1] * 200)
    clo, chi = traveller_clustered_bootstrap_ci(diffs, travs, n_boot=500)
    iid_lo, iid_hi = np.percentile(
        [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(500)],
        [2.5, 97.5],
    )
    assert (chi - clo) > 5.0 * (
        iid_hi - iid_lo
    )  # clustering exposes the real (huge) uncertainty


def _planted_aif_advantage_diffs(
    *, n_travellers: int, per_traveller_effect: float, seed: int
):
    """Synthetic paired-difference stream with a KNOWN AIF advantage planted by design.

    ``diffs`` are oriented positive => AIF better (soft - aif), exactly as the real comparator
    produces them in :func:`aif_vs_soft_clustered`. Each traveller (cluster) gets a positive mean
    advantage with small between-traveller spread plus within-traveller step noise, so the planted
    effect is real, clustered, and unambiguously positive."""

    rng = np.random.default_rng(seed)
    diffs, travs = [], []
    for t in range(n_travellers):
        base = per_traveller_effect + rng.normal(
            0.0, 0.05
        )  # between-traveller spread, still positive
        steps = base + rng.normal(
            0.0, 0.10, size=40
        )  # correlated within-traveller step noise
        diffs.append(steps)
        travs.append(np.full(40, t))
    return np.concatenate(diffs), np.concatenate(travs)


def test_clustered_comparator_detects_planted_positive_effect() -> None:
    """POSITIVE CONTROL: the traveller-clustered test the AIF-vs-fair comparator uses must actually
    DETECT a real effect, not merely fail to find one. Plant a known +0.30 nat AIF advantage across
    12 travellers and confirm the shipped clustered bootstrap reports it: CI separated above zero
    (the exact `aif_wins_clustered = ci_low > 0` decision rule) and a small one-sided p-value."""
    diffs, travs = _planted_aif_advantage_diffs(
        n_travellers=12, per_traveller_effect=0.30, seed=20260701
    )
    stats = traveller_clustered_bootstrap_stats(diffs, travs, n_boot=2000, seed=7)
    assert stats["mean"] > 0.2  # recovers the planted magnitude
    assert (
        stats["ci_low"] > 0.0
    )  # this IS the comparator's `aif_wins_clustered` rule firing
    assert stats["ci_high"] > stats["ci_low"]
    assert (
        stats["one_sided_p"] < 0.01
    )  # the directional test rejects the null of no AIF advantage
    assert stats["n_travellers"] == 12

    # Directional-integrity check: flip the sign (fair now genuinely better) and the SAME test must
    # NOT falsely detect an AIF win — proving detection is real, not always-positive by construction.
    flipped = traveller_clustered_bootstrap_stats(-diffs, travs, n_boot=2000, seed=7)
    assert flipped["ci_low"] < 0.0  # aif_wins_clustered would be False
    assert flipped["one_sided_p"] > 0.5


def test_aif_vs_soft_clustered_pairs_same_steps() -> None:
    inf = _inference()
    seq = generate_synthetic_sequence(8, inf, seed=2026)
    res = aif_vs_soft_clustered(seq, smoothing=0.5, n_boot=500)
    assert (
        res["clustered_ci_low"]
        <= res["mean_advantage_nats"]
        <= res["clustered_ci_high"]
    )
    assert res["n_holdout_travellers"] >= 2
    # exactly one of wins/ties — they cannot both be true
    assert not (res["aif_wins_clustered"] and res["soft_wins_clustered"])


def test_regime_map_headline_and_non_vacuity() -> None:
    """The headline regime finding + a firing non-vacuity control: the map must contain
    BOTH a seed-robust win and a non-win (else it is green-by-construction). Wins are
    seed-robust = CI-separated in every generator draw (captures between-draw variance)."""
    inf = _inference()
    r = fair_power_regime_map(
        inf,
        traveller_counts=_COUNTS,
        smoothing_levels=_SM,
        gen_seeds=(2026, 2027),
        n_boot=300,
    )
    # few travellers, clean emission: AIF is a seed-robust win
    assert r["aif_wins_few_travellers_clean"] is True
    # at scale the clean-emission advantage vanishes, but noise restores it as a monotone gradient
    assert r["aif_advantage_vanishes_at_scale_clean"] is True
    assert r["aif_advantage_persists_under_noise_at_scale"] is True
    assert r["noise_gradient_monotone_at_scale"] is True
    assert r["noise_restores_advantage_at_scale"] is True
    # FIRING non-vacuity control: the map is discriminative (not all seed-robust wins)
    assert r["map_is_discriminative"] is True
    wins = [c["seed_robust_win"] for c in r["cells"]]
    assert any(wins) and not all(wins)
    # every cell reports across-seed dispersion (the between-draw variance the bootstrap can't see)
    assert all(
        "advantage_sd_across_seeds" in c and c["n_gen_seeds"] == 2 for c in r["cells"]
    )
    assert "regime-bound" in r["verdict"].lower()
    assert "oracle" in r["verdict"].lower()  # honesty boundary


def test_writer_emits_artifact(tmp_path) -> None:
    out = run_and_write_power_analysis(
        ROOT / "configs" / "default.yaml", tmp_path / "power_analysis.json"
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "active-fractal-rabbit-power-analysis-v1"
    assert len(payload["cells"]) == len(payload["traveller_counts"]) * len(
        payload["smoothing_levels"]
    )
    assert payload["noise_restores_advantage_at_scale"] is True
