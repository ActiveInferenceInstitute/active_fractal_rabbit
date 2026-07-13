"""Two stronger classical sequence candidates for the SOTA program spine.

These plug into the leakage-free spine of ``sota_program.py`` (the
:class:`~active_fractal_rabbit.sota_program.Candidate` protocol): each maps a
:class:`~active_fractal_rabbit.sota_program.PredictContext` — information through step
``i - 1`` only — to a length-``n_states`` predictive distribution over the NEXT latent
cell. Both are *fair*: they use only the shared train-learned transition, the known
emission channel, and the candidate's own past observations. Neither is given the current
observation ``o_i``, the true target ``z_i``, or any oracle/future signal, so neither can
out-score the leakage-free baselines through the harness — they can only win by modelling
the latent dynamics better.

Two candidates:

1. :class:`SwitchingNoiseHMMCandidate` — a switching / latent-regime HMM next-cell
   predictor. The latent factor is the cell; the regime factor is the observation-noise
   level (a small bank of emission channels). It runs one recursive cell-filter per regime
   and Bayesian-model-averages them by their *online predictive likelihood* of the observed
   stream under a sticky regime transition. When the realized noise differs from the single
   nominal ``eps`` the bundled soft filter assumes, averaging over a regime bank concentrates
   on the better-matched channel and predicts the next cell more sharply. This is the
   recursive analogue of the batch-EM regime HMM in ``latent_regime_hmm.py`` (regimes index
   emission noise rather than cell-emission distributions), but it filters online and never
   refits on the held-out data, so it carries no overfit gap.

2. :class:`ParticleFilterCandidate` — a bootstrap particle filter over the latent cell under
   the KNOWN emission and the shared transition. Particles are latent-cell hypotheses
   propagated through ``transition``, reweighted by ``emission[o, particle]`` on each
   observed cell, and systematically resampled when the effective sample size collapses. The
   next-cell prediction marginalizes the particle cloud through the transition,
   ``transition @ particle_distribution``. With a fixed RNG seed it is fully deterministic.

Both candidates maintain their own recursive state per traveller, advanced one step at a
time using ``context.past_observations[-1]`` (which the spine guarantees equals
``context.prev_observation`` = ``o_{i-1}``). If a context ever arrives out of sequence the
candidate rebuilds its filter from ``context.past_observations`` (correct in all cases),
so the candidates are robust to the harness scoring order without ever peeking at the
present.

Synthetic-only; nothing here asserts a SOTA claim — the harness phase wires these into the
claim gate (which still requires external-lane evidence, integrity/minimization OK, a
traveller-clustered CI separated from zero, and calibration non-regression before clearing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sota_program import PredictContext, emission_matrix

SOTA_CLASSICAL_SCHEMA = "active-fractal-rabbit-sota-classical-v1"

#: Default bank of emission-noise regimes for the switching HMM. The candidate does not
#: know the realized ``eps``; it averages over this small bank of plausible channels and
#: lets the online predictive likelihood concentrate on the best-matched one.
DEFAULT_NOISE_REGIMES: tuple[float, ...] = (0.05, 0.2, 0.4)

#: Default particle count for the bootstrap particle filter. Enough that the empirical
#: particle distribution over a few-dozen-cell grid is a faithful posterior while staying
#: fast and deterministic.
DEFAULT_N_PARTICLES = 512


def _uniform(n_states: int) -> np.ndarray:
    return np.full(n_states, 1.0 / n_states, dtype=float)


def _normalize(vec: np.ndarray) -> np.ndarray:
    total = vec.sum()
    if total <= 0.0 or not np.isfinite(total):
        return _uniform(vec.shape[0])
    return vec / total


# --------------------------------------------------------------------------- #
# Candidate 1: switching / latent-regime HMM over observation-noise regimes
# --------------------------------------------------------------------------- #
@dataclass
class _SwitchingState:
    """Per-traveller recursive state for the switching-noise HMM.

    ``cell_beliefs[k]`` is the cell posterior of the k-th regime filter; ``regime_weights``
    is the posterior over regimes; ``n_seen`` is how many observations have been folded in
    (used to detect out-of-sequence calls and rebuild)."""

    cell_beliefs: np.ndarray  # shape (K, n_states)
    regime_weights: np.ndarray  # shape (K,)
    n_seen: int


@dataclass
class SwitchingNoiseHMMCandidate:
    """Switching / latent-regime HMM next-cell predictor (regime = observation-noise level).

    A bank of cell-filters, one per assumed emission-noise regime in ``noise_regimes``, is
    run online; the regimes are mixed by their posterior weight under a sticky regime
    transition (``regime_stickiness`` self-weight). Each regime filter is an exact recursive
    cell-filter under ITS OWN emission channel; the regime posterior is updated by the
    online predictive likelihood of each newly observed cell, so the bank concentrates on
    the channel that best explains the stream. The next-cell prediction marginalizes every
    regime's belief through the shared transition and mixes by regime weight:
    ``sum_k w_k * (transition @ belief_k)``.

    Fair by construction: it consumes only the shared transition, the candidate's past
    observations, and a fixed bank of plausible emission channels — never ``o_i``, never the
    target, never an oracle. The shared ``context.emission`` is included as one regime so the
    bank always contains the truly-specified channel.
    """

    name: str = "switching_noise_hmm"
    noise_regimes: tuple[float, ...] = DEFAULT_NOISE_REGIMES
    regime_stickiness: float = 0.95
    _states: dict[int, _SwitchingState] = field(default_factory=dict, repr=False)

    def _emissions(self, context: PredictContext) -> np.ndarray:
        """The bank of emission channels: the shared (correctly specified) channel plus the
        nominal noise-regime bank. Shape ``(K, n_states, n_states)`` indexed ``[regime, obs, latent]``."""

        banks = [np.asarray(context.emission, dtype=float)]
        banks.extend(
            emission_matrix(context.n_states, float(eps)) for eps in self.noise_regimes
        )
        return np.stack(banks, axis=0)

    def _init_state(
        self, context: PredictContext, emissions: np.ndarray
    ) -> _SwitchingState:
        """Seed each regime's belief from the FIRST observation under its own channel and the
        regime posterior from how well each channel explains that first observation."""

        past = np.asarray(context.past_observations, dtype=int)
        k = emissions.shape[0]
        n = context.n_states
        first_obs = int(past[0])
        cell_beliefs = np.empty((k, n), dtype=float)
        for r in range(k):
            cell_beliefs[r] = _normalize(emissions[r, first_obs, :].copy())
        # initial regime weights: how plausible was the first observation under each channel
        # (the marginal likelihood under a uniform cell prior is proportional to the column
        # sum of that observation row); start near-uniform but informed.
        marg = emissions[:, first_obs, :].sum(axis=1)
        regime_weights = _normalize(np.where(np.isfinite(marg) & (marg > 0), marg, 1.0))
        state = _SwitchingState(
            cell_beliefs=cell_beliefs, regime_weights=regime_weights, n_seen=1
        )
        for obs in past[1:]:
            self._advance(state, int(obs), context.transition, emissions)
        return state

    def _advance(
        self,
        state: _SwitchingState,
        obs: int,
        transition: np.ndarray,
        emissions: np.ndarray,
    ) -> None:
        """Fold one newly observed cell into every regime filter and update the regime posterior.

        For each regime the predictive likelihood of ``obs`` is
        ``sum_z emission_r[obs, z] * (transition @ belief_r)_z``; the regime weight is
        multiplied by that likelihood (after a sticky mixing), and each regime's cell belief
        is advanced by the standard filter ``emission_r[obs] * (transition @ belief_r)``.
        """

        k = state.cell_beliefs.shape[0]
        # sticky regime mixing of the prior regime weights
        if k > 1:
            stay = float(self.regime_stickiness)
            off = (1.0 - stay) / (k - 1)
            mixed = stay * state.regime_weights + off * (1.0 - state.regime_weights)
            mixed = _normalize(mixed)
        else:
            mixed = state.regime_weights

        new_beliefs = np.empty_like(state.cell_beliefs)
        likelihoods = np.empty(k, dtype=float)
        for r in range(k):
            predicted_cell = (
                transition @ state.cell_beliefs[r]
            )  # p(z_t | past) per regime
            obs_row = emissions[r, obs, :]
            likelihoods[r] = float(
                obs_row @ predicted_cell
            )  # predictive likelihood of obs
            updated = obs_row * predicted_cell
            new_beliefs[r] = _normalize(updated)
        state.cell_beliefs = new_beliefs
        weights = mixed * np.clip(likelihoods, 0.0, None)
        state.regime_weights = _normalize(weights)
        state.n_seen += 1

    def predict(self, context: PredictContext) -> np.ndarray:
        n = context.n_states
        past = np.asarray(context.past_observations, dtype=int)
        if past.size == 0:
            return _uniform(n)
        emissions = self._emissions(context)
        state = self._states.get(context.traveller_id)
        # incremental fast path: exactly one new observation since last call for this traveller
        if state is not None and state.n_seen == past.size - 1:
            self._advance(state, int(past[-1]), context.transition, emissions)
        elif state is None or state.n_seen != past.size:
            # cold start or out-of-sequence call -> rebuild from the full past (always correct)
            state = self._init_state(context, emissions)
        self._states[context.traveller_id] = state

        # mixture next-cell prediction: sum_k w_k (T @ belief_k)
        pred = np.zeros(n, dtype=float)
        for r in range(state.cell_beliefs.shape[0]):
            pred += state.regime_weights[r] * (
                context.transition @ state.cell_beliefs[r]
            )
        return _normalize(pred)


# --------------------------------------------------------------------------- #
# Candidate 2: bootstrap particle filter over the latent cell
# --------------------------------------------------------------------------- #
@dataclass
class _ParticleState:
    """Per-traveller particle cloud: integer cell hypotheses, their weights, the persistent
    RNG that drives propagation/resampling, and how many observations have been folded in
    (for out-of-sequence detection)."""

    particles: np.ndarray  # shape (n_particles,), integer cells
    weights: np.ndarray  # shape (n_particles,)
    rng: (
        np.random.Generator
    )  # persistent per-traveller stream (NOT reset between steps)
    n_seen: int


@dataclass
class ParticleFilterCandidate:
    """Bootstrap particle filter over the latent cell under the known emission and transition.

    A cloud of ``n_particles`` cell hypotheses is propagated through the shared transition
    (sampling the next cell from ``transition[:, z]`` for each particle), reweighted by the
    known emission likelihood ``emission[o, particle]`` on each observed cell, and
    systematically resampled when the effective sample size falls below ``ess_threshold``
    times the particle count. The next-cell predictive distribution marginalizes the
    (resampled) cloud through the transition: a weighted histogram of particle cells, pushed
    one step through ``transition``.

    Fair by construction: it uses only the shared transition, the known emission, and the
    candidate's own past observations. Deterministic given ``seed`` (a private RNG per
    traveller, seeded from ``seed`` and the traveller id).
    """

    name: str = "particle_filter"
    n_particles: int = DEFAULT_N_PARTICLES
    ess_threshold: float = 0.5
    seed: int = 20260616
    _states: dict[int, _ParticleState] = field(default_factory=dict, repr=False)

    def _rng(self, traveller_id: int) -> np.random.Generator:
        return np.random.default_rng((int(self.seed), int(traveller_id)))

    def _init_state(self, context: PredictContext) -> _ParticleState:
        """Seed the cloud from the first observation's emission posterior (uniform cell prior).

        The RNG is created ONCE here and stored on the state, so propagation and resampling
        advance a single continuing random stream across steps (recreating the RNG every call
        would freeze the particle diffusion and collapse the filter)."""

        past = np.asarray(context.past_observations, dtype=int)
        n = context.n_states
        emission = np.asarray(context.emission, dtype=float)
        rng = self._rng(context.traveller_id)
        first_post = _normalize(emission[int(past[0]), :].copy())
        particles = rng.choice(n, size=self.n_particles, p=first_post)
        weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=float)
        state = _ParticleState(particles=particles, weights=weights, rng=rng, n_seen=1)
        for obs in past[1:]:
            self._advance(state, int(obs), context)
        return state

    def _advance(
        self, state: _ParticleState, obs: int, context: PredictContext
    ) -> None:
        """Propagate the cloud one step (transition), reweight by the emission likelihood of
        the observed cell, normalise, and systematically resample on low effective sample size."""

        transition = np.asarray(context.transition, dtype=float)
        emission = np.asarray(context.emission, dtype=float)
        n = context.n_states
        rng = state.rng
        # propagate: each particle samples its next cell from transition[:, current]
        cols = transition[
            :, state.particles
        ]  # shape (n, n_particles), column-stochastic
        cumulative = np.cumsum(cols, axis=0)
        draws = rng.random(state.particles.shape[0])
        new_cells = (draws[None, :] < cumulative).argmax(axis=0).astype(int)
        # reweight by the known emission likelihood of the observed cell
        likelihood = emission[obs, new_cells]
        weights = state.weights * likelihood
        total = weights.sum()
        if total <= 0.0 or not np.isfinite(total):
            # degenerate weights -> reset to a uniform cloud sampled from the emission posterior
            post = _normalize(emission[obs, :].copy())
            new_cells = rng.choice(n, size=self.n_particles, p=post)
            weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=float)
        else:
            weights = weights / total
        # systematic resampling when effective sample size collapses
        ess = 1.0 / float(np.sum(weights**2))
        if ess < self.ess_threshold * self.n_particles:
            positions = (rng.random() + np.arange(self.n_particles)) / self.n_particles
            cumw = np.cumsum(weights)
            idx = np.searchsorted(cumw, positions)
            idx = np.clip(idx, 0, self.n_particles - 1)
            new_cells = new_cells[idx]
            weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=float)
        state.particles = new_cells
        state.weights = weights
        state.n_seen += 1

    def _cell_distribution(self, state: _ParticleState, n_states: int) -> np.ndarray:
        """Weighted histogram of the particle cloud over cells (the filtered cell posterior)."""

        dist = np.zeros(n_states, dtype=float)
        np.add.at(dist, state.particles, state.weights)
        return _normalize(dist)

    def predict(self, context: PredictContext) -> np.ndarray:
        n = context.n_states
        past = np.asarray(context.past_observations, dtype=int)
        if past.size == 0:
            return _uniform(n)
        state = self._states.get(context.traveller_id)
        if state is not None and state.n_seen == past.size - 1:
            # incremental: fold the single newly revealed observation
            self._advance(state, int(past[-1]), context)
        elif state is None or state.n_seen != past.size:
            state = self._init_state(context)
        self._states[context.traveller_id] = state

        cell_post = self._cell_distribution(state, n)
        return _normalize(context.transition @ cell_post)


def classical_candidates() -> tuple[
    SwitchingNoiseHMMCandidate, ParticleFilterCandidate
]:
    """The two stronger classical sequence candidates bundled by this module.

    Returned as fresh instances (each carries its own per-traveller recursive state, so a
    new instance is required per scoring run)."""

    return (SwitchingNoiseHMMCandidate(), ParticleFilterCandidate())
