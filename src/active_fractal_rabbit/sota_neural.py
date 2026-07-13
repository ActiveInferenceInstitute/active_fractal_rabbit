"""A dependency-light NEURAL next-cell candidate for the SOTA spine.

This module supplies ONE learned sequence baseline that plugs into the
``sota_program`` candidate interface (:class:`~active_fractal_rabbit.sota_program.Candidate`):
a compact, pure-numpy two-layer MLP that predicts the next latent cell from the running,
leakage-free *belief* over the latent state. It is a genuinely fair competitor in exactly
the sense the spine requires:

* **Fair (infers from noisy observations only).** The network's only input is the running
  belief — the spine's recursive posterior ``emission[o_{i-1}] * (transition @ belief_{i-1})``
  (renormalized). That belief is a function of the noisy observations through step
  ``i - 1`` ONLY; the network never receives the current observation ``o_i`` or the true
  target ``z_i``. It is therefore subject to the same partial-observability handicap as
  every other fair candidate — no oracle access, no future information. The belief is a
  sufficient statistic of the (Markovian) past, so a feed-forward map over it is the
  natural, honest form for a learned predictor; a recurrence would buy nothing.

* **Trained on training travellers only.** :func:`train_mlp_candidate` walks the TRAIN
  split of a :class:`~active_fractal_rabbit.sota_program.PartialObservabilityDataset`,
  fitting the MLP to predict the next *latent* cell from the belief carried into each
  step. Holdout travellers are never touched during training, so scoring the returned
  candidate on the holdout split is leakage-free.

* **Dependency-light, deterministic.** Pure numpy (no JAX/torch, no model downloads).
  A fixed RNG seed makes the trained weights and every prediction reproducible.

The network is one hidden ReLU layer with a softmax read-out over the ``n_states``
next-cell distribution, trained by mini-batch SGD on cross-entropy. It is deliberately
small: a learned *baseline*, not a heavy framework. It asserts no SOTA claim on its own —
the harness phase decides whether it clears the fair-baseline gate. On the high-power
default fixture it learns to match/beat the point-estimate ``hard_filter`` baseline yet
does NOT beat the Bayes-optimal ``soft_filter`` (the AIF candidate), which is the honest
expected outcome when the generative model is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sota_program import (
    PartialObservabilityDataset,
    PredictContext,
    advance_belief,
    seed_belief_from_observation,
)

SOTA_NEURAL_SCHEMA = "active-fractal-rabbit-sota-neural-v1"

#: Default hidden width of the MLP. Small enough to stay fast and dependency-light,
#: wide enough to model a non-linear refinement of the belief into a next-cell law.
DEFAULT_HIDDEN = 24

#: Default number of mini-batch SGD passes (epochs) over the TRAIN travellers. Kept low
#: on purpose: the belief is already a strong feature, and longer training overfits the
#: finite traveller pool (a real, reported limitation, not a tuning knob to manufacture a
#: win).
DEFAULT_EPOCHS = 6

#: Default SGD learning rate.
DEFAULT_LR = 0.2

#: Default mini-batch size.
DEFAULT_BATCH = 64


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _softmax_vec(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class MlpWeights:
    """The trainable parameters of the two-layer next-cell MLP.

    Shapes are keyed on the input width ``n`` (the belief is length ``n_states``), the
    hidden width ``h``, and the output width ``n``:

    * ``w1`` / ``b1`` — hidden projection ``(h, n)`` / ``(h,)``.
    * ``w2`` / ``b2`` — read-out projection ``(n, h)`` / ``(n,)``.
    """

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @staticmethod
    def initial(n_states: int, hidden: int, *, seed: int) -> "MlpWeights":
        rng = np.random.default_rng(seed)
        return MlpWeights(
            w1=rng.normal(0.0, 0.3, size=(hidden, n_states)),
            b1=np.zeros(hidden),
            w2=rng.normal(0.0, 0.3, size=(n_states, hidden)),
            b2=np.zeros(n_states),
        )


def _train_examples(
    dataset: PartialObservabilityDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (beliefs, targets) training arrays from the TRAIN split.

    Each non-first training step contributes: input = the running leakage-free BELIEF over
    the latent cell carried INTO that step (the same posterior the spine carries), target
    = the TRUE latent cell ``z_i``. Only training travellers are included, so the network
    never fits holdout data, and the input is a strict function of the past.
    """

    latent = dataset.latent
    observed = dataset.observed
    is_first = dataset.is_first
    travellers = dataset.travellers
    is_holdout = dataset.is_holdout
    emission = dataset.emission
    transition = dataset.transition

    beliefs: dict[int, np.ndarray] = {}
    rows: list[np.ndarray] = []
    targets: list[int] = []

    for i in range(len(latent)):
        tid = int(travellers[i])
        obs_i = int(observed[i])
        if is_first[i] or tid not in beliefs:
            beliefs[tid] = seed_belief_from_observation(emission, obs_i)
            continue
        if not bool(is_holdout[i]):  # train travellers only
            rows.append(beliefs[tid].copy())  # belief through i-1 == leakage-free input
            targets.append(int(latent[i]))
        beliefs[tid] = advance_belief(emission, transition, beliefs[tid], obs_i)

    if not targets:
        return np.empty((0, dataset.n_states)), np.empty(0, dtype=int)
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=int)


@dataclass
class MlpNextCellCandidate:
    """A trained two-layer MLP next-cell predictor that plugs into the SOTA spine.

    ``predict`` maps the harness-maintained running belief (``context.belief``) through the
    hidden ReLU layer and softmax read-out to a next-latent-cell distribution. The belief
    is a strict function of observations through step ``i - 1``, so the candidate is
    leakage-free by construction; it never reads ``context``'s (absent) current observation
    or target.
    """

    weights: MlpWeights
    n_states: int
    hidden: int
    name: str = "mlp_next_cell"
    train_mean_loss: float = field(default=float("nan"))

    def predict(self, context: PredictContext) -> np.ndarray:
        if context.n_states != self.n_states:
            raise ValueError(
                f"candidate trained for {self.n_states} states, asked for {context.n_states}"
            )
        belief = context.belief
        if belief is None:
            return np.full(self.n_states, 1.0 / self.n_states)
        hidden = _relu(self.weights.w1 @ belief + self.weights.b1)
        return _softmax_vec(self.weights.w2 @ hidden + self.weights.b2)


def train_mlp_candidate(
    dataset: PartialObservabilityDataset,
    *,
    hidden: int = DEFAULT_HIDDEN,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH,
    l2: float = 0.0,
    seed: int = 20260618,
    name: str = "mlp_next_cell",
) -> MlpNextCellCandidate:
    """Train the MLP next-cell candidate on the TRAIN travellers of ``dataset``.

    Fully deterministic given ``seed``. Uses only training-traveller examples, so the
    returned candidate is leakage-free when scored on the holdout split. Returns an
    :class:`MlpNextCellCandidate` ready to pass to ``sota_program.score_candidate``.
    """

    if hidden < 1:
        raise ValueError("hidden width must be positive")
    if epochs < 1:
        raise ValueError("need at least one training epoch")
    if batch_size < 1:
        raise ValueError("batch size must be positive")

    n = dataset.n_states
    x, y = _train_examples(dataset)
    if x.shape[0] == 0:
        raise ValueError("no training examples; dataset has no non-first train steps")

    weights = MlpWeights.initial(n, hidden, seed=seed)
    rng = np.random.default_rng(seed + 1)
    m = x.shape[0]
    last_mean = float("nan")

    for _ in range(epochs):
        order = rng.permutation(m)
        for start in range(0, m, batch_size):
            idx = order[start : start + batch_size]
            xb = x[idx]
            yb = y[idx]
            b = xb.shape[0]

            h1 = _relu(xb @ weights.w1.T + weights.b1)  # (b, hidden)
            probs = _softmax_rows(h1 @ weights.w2.T + weights.b2)  # (b, n)

            dlogits = probs.copy()
            dlogits[np.arange(b), yb] -= 1.0
            dlogits /= b

            g_w2 = dlogits.T @ h1
            g_b2 = dlogits.sum(axis=0)
            dh = dlogits @ weights.w2
            dh[h1 <= 0.0] = 0.0
            g_w1 = dh.T @ xb
            g_b1 = dh.sum(axis=0)

            weights.w2 -= lr * (g_w2 + l2 * weights.w2)
            weights.b2 -= lr * g_b2
            weights.w1 -= lr * (g_w1 + l2 * weights.w1)
            weights.b1 -= lr * g_b1

        # Full-set mean cross-entropy for reporting (no parameter update).
        h1_all = _relu(x @ weights.w1.T + weights.b1)
        probs_all = _softmax_rows(h1_all @ weights.w2.T + weights.b2)
        picked = np.clip(probs_all[np.arange(m), y], 1e-12, None)
        last_mean = float(-np.log(picked).mean())

    return MlpNextCellCandidate(
        weights=weights,
        n_states=n,
        hidden=hidden,
        name=name,
        train_mean_loss=float(last_mean),
    )
