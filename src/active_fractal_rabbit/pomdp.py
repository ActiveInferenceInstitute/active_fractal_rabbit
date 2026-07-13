"""POMDP construction for waypoint-derived observation factors."""

from __future__ import annotations

import numpy as np

from .types import ObservationSequence, PomdpModel

CANONICAL_MODALITIES = ("location", "time_of_day", "speed", "report_burst")

SEQUENCE_FIELDS = {
    "location": "spatial_states",
    "time_of_day": "time_bins",
    "speed": "speed_categories",
    "report_burst": "burst_categories",
}

_COUNT_FIELDS = {
    "location": "num_spatial_states",
    "time_of_day": "num_time_bins",
    "speed": "num_speed_categories",
    "report_burst": "num_burst_categories",
}


def build_pomdp(
    sequence: ObservationSequence,
    smoothing: float,
    controls: tuple[str, ...] = ("empirical_continuation", "high_entropy_exploration"),
    include: np.ndarray | None = None,
    modalities: tuple[str, ...] = CANONICAL_MODALITIES,
) -> PomdpModel:
    """Build normalized A/B/C/D arrays from a discretized waypoint sequence.

    When ``include`` is given, only rows with ``include[i] == True`` contribute
    counts to A/B/D estimation (holdout scoring trains on the included subset);
    transition pairs count only when both endpoints are included.

    ``modalities`` selects which observation factors enter the model. Dropping
    ``location`` produces a genuine deep-latent configuration: the latent
    spatial cell is then observed only indirectly through time-of-day, speed,
    and reporting-burst evidence.
    """

    states = sequence.num_spatial_states
    if include is None:
        include = np.ones(len(sequence.observations), dtype=bool)
    include = np.asarray(include, dtype=bool)
    if include.shape[0] != len(sequence.observations):
        raise ValueError("include mask length must match the observation count")
    if not include.any():
        raise ValueError("include mask must keep at least one observation")
    unknown = [name for name in modalities if name not in SEQUENCE_FIELDS]
    if unknown:
        raise ValueError(f"unknown observation modalities: {unknown}")
    if not modalities:
        raise ValueError("at least one observation modality is required")
    ordered_names = tuple(name for name in CANONICAL_MODALITIES if name in modalities)
    selected = tuple(
        (
            name,
            getattr(sequence, SEQUENCE_FIELDS[name]),
            getattr(sequence, _COUNT_FIELDS[name]),
        )
        for name in ordered_names
    )
    A = tuple(
        _emission_matrix(
            sequence.spatial_states[include], obs[include], obs_count, states, smoothing
        )
        for _, obs, obs_count in selected
    )
    B = _transition_tensor(
        sequence.spatial_states,
        states,
        smoothing,
        controls,
        include,
        sequence.is_first_observation,
    )
    C = tuple(np.zeros(obs_count, dtype=float) for _, _, obs_count in selected)
    D = _initial_state_prior(sequence.spatial_states[include], states, smoothing)
    return PomdpModel(
        A=A,
        B=B,
        C=C,
        D=D,
        observation_names=ordered_names,
        control_names=controls,
    )


def observation_indices(
    model: PomdpModel, sequence: ObservationSequence
) -> list[tuple[int, ...]]:
    """Per-step observation index tuples aligned to the model's modality names.

    Alignment is by NAME, not position, so modality-subset models (the
    deep-latent lane) receive correctly ordered evidence.
    """

    arrays = []
    for name in model.observation_names:
        field = SEQUENCE_FIELDS.get(name)
        if field is None:
            raise ValueError(f"unknown observation modality: {name}")
        arrays.append(getattr(sequence, field))
    return [
        tuple(int(array[idx]) for array in arrays)
        for idx in range(len(sequence.observations))
    ]


def to_pymdp_payload(model: PomdpModel) -> dict[str, object]:
    """Return JAX-backed arrays suitable for modern pymdp Agent construction."""

    from jax import numpy as jnp

    return {
        "A": tuple(jnp.asarray(matrix) for matrix in model.A),
        "B": jnp.asarray(model.B),
        "C": tuple(jnp.asarray(vector) for vector in model.C),
        "D": jnp.asarray(model.D),
    }


def tensor_summary(model: PomdpModel) -> dict[str, object]:
    """Return a compact normalization and shape summary for generated tensors."""

    return {
        "observation_modalities": list(model.observation_names),
        "transition_controls": list(model.control_names),
        "A_shapes": [list(matrix.shape) for matrix in model.A],
        "B_shape": list(model.B.shape),
        "C_shapes": [list(vector.shape) for vector in model.C],
        "D_shape": list(model.D.shape),
        "A_column_sum_min": [float(np.min(matrix.sum(axis=0))) for matrix in model.A],
        "A_column_sum_max": [float(np.max(matrix.sum(axis=0))) for matrix in model.A],
        "B_column_sum_min": [
            float(np.min(model.B[:, :, control].sum(axis=0)))
            for control in range(model.B.shape[2])
        ],
        "B_column_sum_max": [
            float(np.max(model.B[:, :, control].sum(axis=0)))
            for control in range(model.B.shape[2])
        ],
        "D_sum": float(model.D.sum()),
    }


def _emission_matrix(
    hidden_states: np.ndarray,
    observations: np.ndarray,
    obs_count: int,
    state_count: int,
    smoothing: float,
) -> np.ndarray:
    matrix = np.full((obs_count, state_count), smoothing, dtype=float)
    for state, observed in zip(hidden_states, observations):
        matrix[int(observed), int(state)] += 1.0
    return _normalize_columns(matrix)


def soft_emission_posterior(
    cells: np.ndarray,
    train: np.ndarray,
    n: int,
    smoothing: float,
) -> np.ndarray:
    """Per-step soft-Markov emission posterior P(cell | obs), learned from the TRAIN
    split only (the leakage guard) and evaluated at every step's observed cell.

    Single shared derivation for ``fair_inference.fair_inference_ladder`` and
    ``power_analysis.aif_vs_soft_clustered`` -- both need the identical posterior and
    must not independently re-derive it.
    """
    emission = _emission_matrix(cells[train], cells[train], n, n, smoothing)
    return emission[cells, :] / emission[cells, :].sum(axis=1, keepdims=True)


def _transition_tensor(
    hidden_states: np.ndarray,
    state_count: int,
    smoothing: float,
    controls: tuple[str, ...],
    include: np.ndarray | None = None,
    first_observation: np.ndarray | None = None,
) -> np.ndarray:
    if include is None:
        include = np.ones(hidden_states.shape[0], dtype=bool)
    # A consecutive (idx, idx+1) pair is a real within-traveller transition only
    # when idx+1 is NOT the first observation of a new traveller. Crossing a
    # traveller boundary (idx+1 is_first) would otherwise inject a spurious
    # last-cell-of-A -> first-cell-of-B count into the learned dynamics. The
    # boundary guard is a structural fact, kept separate from the holdout
    # `include` mask (which is the train/score split).
    if first_observation is None:
        first_observation = np.zeros(hidden_states.shape[0], dtype=bool)
    first_observation = np.asarray(first_observation, dtype=bool)
    tensor = np.full((state_count, state_count, len(controls)), smoothing, dtype=float)
    for control_idx, control in enumerate(controls):
        if control == "empirical_continuation":
            for idx, (previous, current) in enumerate(
                zip(hidden_states, hidden_states[1:])
            ):
                if include[idx] and include[idx + 1] and not first_observation[idx + 1]:
                    tensor[int(current), int(previous), control_idx] += 1.0
        elif control == "high_entropy_exploration":
            tensor[:, :, control_idx] += 1.0
        elif control == "identity":
            tensor[:, :, control_idx] += np.eye(state_count)
        else:
            raise ValueError(f"unknown control: {control}")
    for control in range(tensor.shape[2]):
        tensor[:, :, control] = _normalize_columns(tensor[:, :, control])
    return tensor


def _initial_state_prior(
    hidden_states: np.ndarray, state_count: int, smoothing: float
) -> np.ndarray:
    prior = np.full(state_count, smoothing, dtype=float)
    prior[int(hidden_states[0])] += 1.0
    return prior / prior.sum()


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=0, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("cannot normalize a matrix with zero-probability columns")
    return matrix / totals
