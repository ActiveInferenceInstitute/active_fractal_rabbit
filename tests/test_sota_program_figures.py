from __future__ import annotations

from active_fractal_rabbit.sota_program_figures import (
    SOTA_PROGRAM_FIGURE_SCHEMA,
    sota_program_leaderboard_summary,
)


def test_sota_program_leaderboard_summary_filters_to_locked_fractalrabbit_rows() -> (
    None
):
    rows = [
        {
            "lane": "fractalrabbit",
            "locked_split_scored": "true",
            "candidate": "precision_return_aif",
            "family": "active_inference_candidate",
            "role": "candidate",
            "mean_loss": "1.2",
            "loss_ci_low": "1.1",
            "loss_ci_high": "1.3",
            "clustered_ci_low": "-0.01",
            "clustered_ci_high": "0.02",
            "brier_score": "0.011",
            "ece": "0.027",
            "top1_accuracy": "0.74",
            "runtime_seconds": "0.04",
            "leakage": "false",
        },
        {
            "lane": "fractalrabbit",
            "locked_split_scored": "true",
            "candidate": "switching_noise_hmm",
            "family": "classical_state_space",
            "role": "strongest_fair_baseline",
            "mean_loss": "1.1",
            "loss_ci_low": "1.0",
            "loss_ci_high": "1.2",
            "runtime_seconds": "0.03",
            "leakage": "false",
        },
        {
            "lane": "gravity",
            "locked_split_scored": "true",
            "candidate": "precision_return_aif",
            "family": "active_inference_candidate",
            "role": "candidate",
            "mean_loss": "0.8",
        },
        {
            "lane": "fractalrabbit",
            "locked_split_scored": "false",
            "candidate": "full_stream_row",
            "family": "classical_sequence",
            "role": "candidate",
            "mean_loss": "0.1",
        },
    ]

    summary = sota_program_leaderboard_summary(
        rows,
        {
            "status": "sota_program_not_cleared",
            "lanes": {
                "fractalrabbit": {
                    "n_holdout_travellers": 26,
                    "strongest_fair_baseline": "switching_noise_hmm",
                    "mean_difference_nats_fair_minus_aif": 0.004929,
                    "clustered_ci_low": -0.001616,
                    "clustered_ci_high": 0.0121,
                    "one_sided_p": 0.08046,
                    "aif_beats_strongest_fair": False,
                }
            },
        },
    )

    assert summary["schema"] == SOTA_PROGRAM_FIGURE_SCHEMA
    assert summary["status"] == "sota_program_not_cleared"
    assert summary["strongest_baseline_loss"] == 1.1
    leaderboard = summary["leaderboard"]
    assert [row["candidate"] for row in leaderboard] == [
        "precision_return_aif",
        "switching_noise_hmm",
    ]
    assert leaderboard[0]["model"] == "precision return AIF"
    assert leaderboard[0]["family"] == "active_inference"
    assert leaderboard[0]["brier_score"] == 0.011
    assert leaderboard[0]["ece"] == 0.027
    assert leaderboard[0]["top1_accuracy"] == 0.74
    assert leaderboard[0]["clustered_ci_low"] == -0.01
    assert leaderboard[1]["family"] == "classical_state_space"
    gate = summary["fractalrabbit_gate"]
    assert gate["n_holdout_travellers"] == 26
    assert gate["strongest_fair_baseline"] == "switching_noise_hmm"
    assert gate["one_sided_p"] == 0.08046
    assert gate["aif_beats_strongest_fair"] is False
