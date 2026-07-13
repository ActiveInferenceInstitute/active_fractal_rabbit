from __future__ import annotations

from active_fractal_rabbit.adaptive_aif_ablation import run_adaptive_aif_ablation
import pytest


def test_adaptive_ablation_is_scored_and_honestly_bounded() -> None:
    payload = run_adaptive_aif_ablation(
        n_travellers=24, realization_seeds=(20260616, 20260625), n_boot=100
    )
    assert payload["status"] == "adaptive_repeated_ablation_scored"
    assert payload["fixed"]["adaptation"] == 0.0
    assert 0.0 <= payload["adaptive"]["adaptation"] <= 1.0
    assert payload["fixed"]["holdout_loss"] > 0.0
    assert payload["adaptive"]["holdout_loss"] > 0.0
    assert payload["realization_count"] == 2
    assert len(payload["rows"]) == 2
    assert payload["adaptive_beats_online_base_rate_ci_count"] <= 2
    assert "does not imply" in payload["boundary"]


def test_adaptive_ablation_rejects_vacuous_or_duplicate_replication() -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_adaptive_aif_ablation(realization_seeds=())
    with pytest.raises(ValueError, match="unique"):
        run_adaptive_aif_ablation(realization_seeds=(7, 7))
    with pytest.raises(ValueError, match="positive"):
        run_adaptive_aif_ablation(realization_seeds=(7,), n_boot=0)
