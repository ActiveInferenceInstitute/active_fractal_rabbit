"""Routine-break (anomaly) detection via variational free energy / surprise.

The reference filter's per-step predictive loss, ``-log p(o_t | prior_t)``, is
the variational free energy of the observation under the running belief — i.e.
*surprise*. A routine-break (a report from a cell the traveller's learned
dynamics did not expect) should be surprising in a way a naive novelty detector
("have I seen this cell before?") misses, because surprise folds in the
transition model: a jump to a cell that is globally common but unreachable from
the current belief is still surprising.

The honest evaluation is a separability/ROC measurement, never an asserted
detection rate: the generative model is fit on the clean routine, anomalies are
injected into the observation stream, and we report the area under the ROC for
surprise against the injected labels — alongside a novelty baseline on the same
labels. Everything is a synthetic-data software measurement on the pinned
FractalRabbit generator, not empirical anomaly detection on real traces.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .inference import run_latent_inference
from .pomdp import build_pomdp
from .types import ObservationSequence


def inject_routine_breaks(
    sequence: ObservationSequence, fraction: float, seed: int, mode: str = "rare"
) -> tuple[np.ndarray, np.ndarray]:
    """Inject routine-breaks at a fraction of non-first steps.

    ``mode='rare'`` forces the globally rarest cell (a clear break); ``mode='random'``
    forces a random *different* cell (a subtler break). Reporting both honestly
    bounds how detection scales with anomaly strength rather than claiming a single
    near-perfect number. Returns (anomalous_cells, labels).
    """

    cells = np.asarray(sequence.spatial_states, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    n_states = sequence.num_spatial_states
    non_first = np.where(~first)[0]
    rng = np.random.default_rng(seed)
    count = (
        max(3, int(fraction * len(non_first)))
        if fraction < 1
        else max(3, len(non_first) // 12)
    )
    anomaly_steps = rng.choice(non_first, count, replace=False)
    anomalous = cells.copy()
    if mode == "rare":
        rare_cell = int(np.argmin(np.bincount(cells, minlength=n_states)))
        anomalous[anomaly_steps] = rare_cell
    elif mode == "random":
        for step in anomaly_steps:
            choices = [c for c in range(n_states) if c != cells[step]]
            anomalous[step] = int(rng.choice(choices))
    else:
        raise ValueError(f"unknown anomaly mode: {mode}")
    labels = np.zeros(len(cells), dtype=bool)
    labels[anomaly_steps] = True
    return anomalous, labels


def novelty_baseline(cells: np.ndarray, first: np.ndarray) -> np.ndarray:
    """First-occurrence novelty: 1.0 the first time a cell is seen, else 0.0."""

    cells = np.asarray(cells, dtype=int)
    seen: set[int] = set()
    novelty = np.zeros(len(cells), dtype=float)
    for step in range(len(cells)):
        novelty[step] = 0.0 if int(cells[step]) in seen else 1.0
        seen.add(int(cells[step]))
    return novelty


def detection_auc(score: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    """Mann-Whitney AUC: probability a positive scores above a negative."""

    s = np.asarray(score)[mask]
    y = np.asarray(labels)[mask]
    n_pos = int(np.sum(y))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty_like(sorted_scores, dtype=float)
    start = 0
    while start < sorted_scores.size:
        stop = start + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        # Ranks are one-based; tied observations receive the average rank.
        ranks[start:stop] = 0.5 * ((start + 1) + stop)
        start = stop
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    u_statistic = float(np.sum(original_ranks[y]) - n_pos * (n_pos + 1) / 2)
    return u_statistic / (n_pos * n_neg)


@dataclass(frozen=True)
class AnomalyResult:
    n_anomalies: int
    n_scored: int
    vfe_auc: float
    novelty_auc: float
    mean_surprise_anomaly: float
    mean_surprise_normal: float


def run_anomaly_detection(
    sequence: ObservationSequence,
    *,
    smoothing: float,
    fraction: float = 1 / 12,
    seed: int = 0,
    mode: str = "rare",
) -> AnomalyResult:
    """Fit the generative model on the CLEAN routine, score surprise on a stream
    with injected routine-breaks, and report the honest ROC against a novelty
    baseline. Training on the clean routine (not the anomalous stream) is what
    makes a break genuinely surprising rather than partially learned."""

    first = np.asarray(sequence.is_first_observation, dtype=bool)
    clean_model = build_pomdp(sequence, smoothing)  # fit on the clean routine
    anomalous_cells, labels = inject_routine_breaks(sequence, fraction, seed, mode)
    anomalous_sequence = dataclasses.replace(sequence, spatial_states=anomalous_cells)
    surprise = np.asarray(
        run_latent_inference(clean_model, anomalous_sequence).predictive_loss
    )
    novelty = novelty_baseline(anomalous_cells, first)
    mask = ~first
    return AnomalyResult(
        n_anomalies=int(labels.sum()),
        n_scored=int(mask.sum()),
        vfe_auc=detection_auc(surprise, labels, mask),
        novelty_auc=detection_auc(novelty, labels, mask),
        mean_surprise_anomaly=float(surprise[labels & mask].mean()),
        mean_surprise_normal=float(surprise[(~labels) & mask].mean()),
    )


ANOMALY_SCHEMA = "active-fractal-rabbit-anomaly-v1"


def run_and_write_anomaly(config_path, observations_path, out_path, *, seed: int = 0):
    import json
    from pathlib import Path

    from .config import load_config
    from .fractalrabbit import parse_waypoint_csv
    from .observations import discretize_observations

    config = load_config(config_path)
    obs = Path(observations_path)
    if obs.exists():
        source = str(observations_path)
    else:
        assert config.fractalrabbit.fixture_csv is not None
        obs = Path(config.fractalrabbit.fixture_csv)
        source = f"fixture:{obs.name}"
    sequence = discretize_observations(parse_waypoint_csv(obs), config.inference)
    rare = run_anomaly_detection(
        sequence, smoothing=config.inference.smoothing, seed=seed, mode="rare"
    )
    rand = run_anomaly_detection(
        sequence, smoothing=config.inference.smoothing, seed=seed, mode="random"
    )
    payload = {
        "schema": ANOMALY_SCHEMA,
        "source": source,
        "n_anomalies": rare.n_anomalies,
        "n_scored": rare.n_scored,
        "vfe_auc": rare.vfe_auc,
        "vfe_auc_rare": rare.vfe_auc,
        "vfe_auc_random": rand.vfe_auc,
        "novelty_auc": rare.novelty_auc,
        "mean_surprise_anomaly": rare.mean_surprise_anomaly,
        "mean_surprise_normal": rare.mean_surprise_normal,
        "caveat": (
            "Routine-break detection via filter surprise (variational free energy), scored as ROC "
            "AUC against injected synthetic anomalies, with a first-occurrence novelty baseline on the "
            "same labels. Synthetic-data software measurement on the pinned FractalRabbit generator, "
            "not empirical anomaly detection on real traces."
        ),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out
