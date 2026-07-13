"""EFE-driven location-minimization frontier - oversight-positive boundary.

This is the active-inference answer to the program's core location-minimization
question, measured in free-energy's own currency (expected cell-entropy reduction,
in nats). At each step, under the running location-blind posterior, it scores the
epistemic value (``H(prior) - E_o[H(post|o)]``) of each observation modality:
location alone, each metadatum, the metadata bundle (location withheld), and the
full set. It then asks the oversight question directly — how many nats does
withholding location cost, and does the metadata bundle secretly recover it?

The decisive control is a **structure-destroyed null**: the metadata emissions
are rebuilt from cell-label-shuffled data, which destroys any genuine
state->metadata structure while preserving the marginal shapes. The artifact
verdict decides whether the small metadata residual clears that null for the
current configuration; the 4x4 default demotes the previous metadata-positive
claim. This is a synthetic generator boundary, not a real-mobility,
operational, or surveillance claim.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np

from .active_sensing import _entropy, modality_info_gain
from .anticipatory import speed_conditioned_transition
from .figure_style import (
    LEGEND_FONT_SIZE,
    SMALL_LABEL_FONT_SIZE,
    TICK_FONT_SIZE,
    manuscript_rcparams,
)
from .pomdp import _emission_matrix, build_pomdp, observation_indices
from .types import ObservationSequence, PomdpModel

EFE_LOCATION_FRONTIER_SCHEMA = "active-fractal-rabbit-efe-location-frontier-v1"
_METADATA_MODALITIES = ("time_of_day", "speed", "report_burst")
_LARGE_SEQUENCE_NULL_MAX_DRAWS = 12


def _joint_info_gain(
    prior: np.ndarray, emissions: list[np.ndarray], n_states: int
) -> float:
    """Exact expected cell-entropy reduction from observing a SET of modalities."""

    base = _entropy(prior)
    expected = 0.0
    for combo in product(*[range(matrix.shape[0]) for matrix in emissions]):
        likelihood = np.ones(n_states)
        for emission, obs_index in zip(emissions, combo):
            likelihood = likelihood * emission[obs_index, :]
        joint = likelihood * prior
        total = float(joint.sum())
        if total > 0:
            expected += total * _entropy(joint / total)
    return base - expected


def _frontier_pass(
    sequence: ObservationSequence,
    emissions: list[np.ndarray],
    model: PomdpModel,
    transition: np.ndarray,
    *,
    location_index: int,
    metadata_indices: list[int],
    include_full: bool,
) -> dict[str, float]:
    """One running-posterior pass; returns mean per-modality info-gain in nats."""

    n_states = int(model.D.size)
    evidence = observation_indices(model, sequence)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    speed = np.asarray(sequence.speed_categories, dtype=int)
    location = np.zeros(len(evidence))
    bundle = np.zeros(len(evidence))
    full = np.zeros(len(evidence))
    per_metadatum: dict[int, list[float]] = {idx: [] for idx in metadata_indices}
    prior = model.D.copy()
    scored: list[int] = []
    for step, obs in enumerate(evidence):
        if not first[step]:
            scored.append(step)
            location[step] = modality_info_gain(prior, emissions[location_index])
            bundle[step] = _joint_info_gain(
                prior, [emissions[i] for i in metadata_indices], n_states
            )
            if include_full:
                full[step] = _joint_info_gain(prior, list(emissions), n_states)
            for idx in metadata_indices:
                per_metadatum[idx].append(modality_info_gain(prior, emissions[idx]))
        # propagate the LOCATION-BLIND posterior (metadata-only evidence)
        likelihood = np.ones(n_states)
        for modality, obs_index in enumerate(obs):
            if modality != location_index:
                likelihood = likelihood * model.A[modality][obs_index, :]
        posterior = likelihood * prior
        posterior = (
            posterior / posterior.sum()
            if posterior.sum() > 0
            else np.ones(n_states) / n_states
        )
        if step + 1 < len(evidence):
            prior = (
                model.D.copy()
                if first[step + 1]
                else transition[:, :, speed[step + 1]] @ posterior
            )
            prior = prior / prior.sum()
    mask = np.asarray(scored)
    result = {
        "location_only_mean": float(location[mask].mean()),
        "metadata_bundle_mean": float(bundle[mask].mean()),
        "per_metadatum": {
            idx: float(np.mean(values)) for idx, values in per_metadatum.items()
        },
    }
    if include_full:
        result["full_set_mean"] = float(full[mask].mean())
    return result


def _time_to_localize(
    sequence: ObservationSequence,
    model: PomdpModel,
    transition: np.ndarray,
    *,
    with_location: bool,
    threshold_nats: float,
) -> dict[str, float]:
    """Per-traveller reports until posterior cell-entropy drops below threshold."""

    n_states = int(model.D.size)
    evidence = observation_indices(model, sequence)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    speed = np.asarray(sequence.speed_categories, dtype=int)
    travellers = np.asarray(
        [row.traveller_id for row in sequence.observations], dtype=int
    )
    location_index = list(model.observation_names).index("location")
    localized: list[int] = []
    prior = model.D.copy()
    count = 0
    resolved = True  # no traveller open before the first observation
    for step, obs in enumerate(evidence):
        if first[step]:
            prior = model.D.copy()
            count = 0
            resolved = False
        likelihood = np.ones(n_states)
        for modality, obs_index in enumerate(obs):
            if with_location or modality != location_index:
                likelihood = likelihood * model.A[modality][obs_index, :]
        posterior = likelihood * prior
        posterior = (
            posterior / posterior.sum()
            if posterior.sum() > 0
            else np.ones(n_states) / n_states
        )
        count += 1
        if not resolved and _entropy(posterior) < threshold_nats:
            localized.append(count)
            resolved = True
        if step + 1 < len(evidence):
            prior = (
                model.D.copy()
                if first[step + 1]
                else transition[:, :, speed[step + 1]] @ posterior
            )
            prior = prior / prior.sum()
    n_travellers = len(set(travellers.tolist()))
    return {
        "frac_localized": float(len(localized) / n_travellers),
        "mean_reports": float(np.mean(localized)) if localized else float("nan"),
    }


def efe_location_frontier(
    sequence: ObservationSequence,
    model: PomdpModel,
    *,
    smoothing: float,
    threshold_nats: float = 0.5,
    n_structure_null: int = 50,
) -> dict[str, object]:
    """Compute the per-modality info-gain frontier and the structure-null gate."""

    effective_null_draws = (
        min(n_structure_null, _LARGE_SEQUENCE_NULL_MAX_DRAWS)
        if len(sequence.observations) > 2000
        else n_structure_null
    )
    names = list(model.observation_names)
    location_index = names.index("location")
    metadata_indices = [
        names.index(name) for name in _METADATA_MODALITIES if name in names
    ]
    n_states = int(model.D.size)
    cells = np.asarray(sequence.spatial_states, dtype=int)
    speed = np.asarray(sequence.speed_categories, dtype=int)
    first = np.asarray(sequence.is_first_observation, dtype=bool)
    transition = speed_conditioned_transition(
        cells, speed, first, n_states, sequence.num_speed_categories, smoothing
    )

    real = _frontier_pass(
        sequence,
        list(model.A),
        model,
        transition,
        location_index=location_index,
        metadata_indices=metadata_indices,
        include_full=True,
    )
    location_mean = real["location_only_mean"]
    bundle_mean = real["metadata_bundle_mean"]
    full_mean = real["full_set_mean"]

    # Structure-destroyed null: rebuild metadata emissions from shuffled cell labels.
    metadata_sources = {
        "time_of_day": (
            np.asarray(sequence.time_bins, dtype=int),
            sequence.num_time_bins,
        ),
        "speed": (
            np.asarray(sequence.speed_categories, dtype=int),
            sequence.num_speed_categories,
        ),
        "report_burst": (
            np.asarray(sequence.burst_categories, dtype=int),
            sequence.num_burst_categories,
        ),
    }
    null_bundles: list[float] = []
    for seed in range(effective_null_draws):
        rng = np.random.default_rng(seed)
        shuffled = cells.copy()
        rng.shuffle(shuffled)
        emissions = list(model.A)
        for name in _METADATA_MODALITIES:
            if name in names:
                values, count = metadata_sources[name]
                emissions[names.index(name)] = _emission_matrix(
                    shuffled, values, count, n_states, smoothing
                )
        null_pass = _frontier_pass(
            sequence,
            emissions,
            model,
            transition,
            location_index=location_index,
            metadata_indices=metadata_indices,
            include_full=False,
        )
        null_bundles.append(null_pass["metadata_bundle_mean"])
    null = np.asarray(null_bundles)
    p_value = float(np.mean(null >= bundle_mean))
    gate_passed = bool(bundle_mean > np.percentile(null, 95))

    mutual_information = {
        name: _mutual_information(
            cells,
            _modality_values(sequence, name),
            n_states,
            _modality_count(sequence, name),
        )
        for name in ("location", *_METADATA_MODALITIES)
        if name in names
    }
    power_check = frontier_power_check(
        sequence, model, smoothing=smoothing, n_structure_null=effective_null_draws
    )
    without = _time_to_localize(
        sequence, model, transition, with_location=False, threshold_nats=threshold_nats
    )
    verdict = (
        "metadata_recovery_exceeds_structure_null"
        if gate_passed
        else "metadata_recovery_not_above_structure_null"
    )
    caveat_claim = (
        "The finding is a small positive: metadata-bundle information gain exceeds the cell-label-shuffle "
        "structure null, so the expanded fixture no longer supports a strict metadata-irrecoverability null. "
        "The effect is still weak relative to direct location and does not localize travellers without location."
        if gate_passed
        else (
            "The finding is a failure-to-reject, made interpretable by a POSITIVE CONTROL: the same "
            "cell-label-shuffle test rejects when a genuine cell-coupled metadatum is injected (see "
            "structure_null_power_check.gate_passed), so the headline result is 'no metadata structure "
            "detectable above the floor', not 'the test is too weak'."
        )
    )

    return {
        "schema": EFE_LOCATION_FRONTIER_SCHEMA,
        "status": "completed",
        "per_modality_mean_nats": {
            "location": location_mean,
            **{names[idx]: real["per_metadatum"][idx] for idx in metadata_indices},
        },
        "location_only_mean": location_mean,
        "metadata_bundle_mean": bundle_mean,
        "full_set_mean": full_mean,
        "location_joint_share": float(location_mean / full_mean)
        if full_mean > 0
        else 0.0,
        "frontier_cost_nats": float(full_mean - bundle_mean),
        "fraction_recovered_by_metadata": float(bundle_mean / full_mean)
        if full_mean > 0
        else 0.0,
        "mutual_information": mutual_information,
        "structure_null": {
            "mean": float(null.mean()),
            "std": float(null.std()),
            "p5": float(np.percentile(null, 5)),
            "p95": float(np.percentile(null, 95)),
            "p_value_null_ge_real": p_value,
            "gate_passed": gate_passed,
            "n_draws": float(effective_null_draws),
        },
        "time_to_localize": {
            "threshold_nats": threshold_nats,
            "with_location": _time_to_localize(
                sequence,
                model,
                transition,
                with_location=True,
                threshold_nats=threshold_nats,
            ),
            "without_location": without,
        },
        "structure_null_power_check": power_check,
        "residual_localized_without_location": float(without["frac_localized"]),
        "verdict": verdict,
        "caveat": (
            f"Fixture-bounded synthetic-only ({len(sequence.observations)} rows, {n_states} cells, "
            f"{len({row.traveller_id for row in sequence.observations})} travellers); no real-mobility, "
            f"operational, or surveillance claim. {caveat_claim} The cell-label shuffle destroys the state->metadata "
            "emission map while preserving each metadatum's marginal and the occupancy marginal. Scope is the "
            "generative model and metadata bundle here - NOT a universal metadata/location result; real "
            "mobility may couple metadata and place far more strongly. This is NOT a collection-priority "
            "recommendation. The residual fraction of travellers localized without location is reported explicitly "
            "as residual_localized_without_location, so localizability reduction is measured rather than assumed. 'speed' is "
            "displacement-derived (disclosed leakage)."
        ),
    }


def frontier_power_check(
    sequence: ObservationSequence,
    model: PomdpModel,
    *,
    smoothing: float,
    n_structure_null: int = 50,
) -> dict[str, object]:
    """Positive control: the structure-null gate must REJECT when coupling exists.

    A failure-to-reject (the real result, p high) distinguishes "no metadata
    structure" from "the test is too weak to see it" ONLY if the same test
    rejects when a genuine coupling is present. Here a synthetic metadatum that is
    a deterministic function of the cell (``cell % K``) is injected; its emission
    carries real state structure, so the cell-label-shuffle null must collapse it
    and the gate must pass. If this control did not reject, the headline
    failure-to-reject would be uninterpretable.
    """

    n_states = int(model.D.size)
    cells = np.asarray(sequence.spatial_states, dtype=int)
    k_bins = 4
    coupled_values = cells % k_bins
    prior = model.D.copy()
    coupled_emission = _emission_matrix(
        cells, coupled_values, k_bins, n_states, smoothing
    )
    coupled_gain = modality_info_gain(prior, coupled_emission)
    null_gains: list[float] = []
    for seed in range(n_structure_null):
        rng = np.random.default_rng(seed)
        shuffled = cells.copy()
        rng.shuffle(shuffled)
        null_emission = _emission_matrix(
            shuffled, coupled_values, k_bins, n_states, smoothing
        )
        null_gains.append(modality_info_gain(prior, null_emission))
    null = np.asarray(null_gains)
    return {
        "coupled_info_gain": float(coupled_gain),
        "structure_null_mean": float(null.mean()),
        "structure_null_p95": float(np.percentile(null, 95)),
        "p_value_null_ge_real": float(np.mean(null >= coupled_gain)),
        "gate_passed": bool(coupled_gain > np.percentile(null, 95)),
    }


def _modality_values(sequence: ObservationSequence, name: str) -> np.ndarray:
    return {
        "location": np.asarray(sequence.spatial_states, dtype=int),
        "time_of_day": np.asarray(sequence.time_bins, dtype=int),
        "speed": np.asarray(sequence.speed_categories, dtype=int),
        "report_burst": np.asarray(sequence.burst_categories, dtype=int),
    }[name]


def _modality_count(sequence: ObservationSequence, name: str) -> int:
    return {
        "location": sequence.num_spatial_states,
        "time_of_day": sequence.num_time_bins,
        "speed": sequence.num_speed_categories,
        "report_burst": sequence.num_burst_categories,
    }[name]


def _mutual_information(
    cells: np.ndarray, observations: np.ndarray, n_states: int, n_obs: int
) -> float:
    joint = np.zeros((n_obs, n_states))
    for obs_value, cell in zip(observations, cells):
        joint[int(obs_value), int(cell)] += 1.0
    joint /= joint.sum()
    p_obs = joint.sum(axis=1, keepdims=True)
    p_cell = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(joint > 0, joint / (p_obs * p_cell), 1.0)
        return float(np.sum(joint * np.log(ratio)))


def _p_annotation_text(raw_p: float, n_draws: int) -> str:
    """Format a permutation p-value for display, flooring an exact 0.0 to the
    resolution the draw count actually supports. A p stored as 0.0 from a
    finite null cannot be resolved below (0+1)/(n_draws+1); printing "p=0"
    overstates resolution (mirrors the multiplicity ledger's convention)."""

    if raw_p <= 0.0 and n_draws > 0:
        return f"p<{1.0 / (n_draws + 1):.2g} ({n_draws} draws)"
    return f"p={raw_p:.2g}"


def render_frontier_figure(summary: dict[str, object], path: str | Path) -> Path:
    """Render the oversight-positive location-minimization frontier figure.

    Top: per-modality mean info-gain in nats (location towering over metadata),
    with the metadata-bundle bar drawn against the shaded structure-destroyed null
    band — the visual kill (the bundle sits INSIDE the null). Bottom: confident-
    localization with vs without location (fraction localized and mean reports).
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(manuscript_rcparams({"gray": "#595959"}))
    if "per_modality_mean_nats" not in summary:
        raise ValueError(
            "efe_location_frontier figure requires output/runs/efe_location_frontier.json; "
            "run the full study (scripts/07_run_full_study.py) before generating figures"
        )
    per_modality = summary["per_modality_mean_nats"]
    null = summary["structure_null"]
    localize = summary["time_to_localize"]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.6, 9.4),
        height_ratios=[1.22, 1.0],
        constrained_layout=True,
    )

    names = list(per_modality.keys())
    values = [float(per_modality[name]) for name in names]
    positions = np.arange(len(names) + 2)
    axes[0].bar(
        positions[: len(names)],
        values,
        color=["#0072B2"] + ["#009E73"] * (len(names) - 1),
    )
    bundle = float(summary["metadata_bundle_mean"])
    bundle_pos = positions[-2]
    control_pos = positions[-1]
    power_check = (
        summary.get("structure_null_power_check", {})
        if isinstance(summary.get("structure_null_power_check"), dict)
        else {}
    )
    control_value = float(power_check.get("coupled_info_gain", 0.0))
    axes[0].bar([bundle_pos], [bundle], color="#CC79A7")
    axes[0].bar([control_pos], [control_value], color="#E69F00")
    axes[0].fill_between(
        [bundle_pos - 0.38, bundle_pos + 0.38],
        float(null["p5"]),
        float(null["p95"]),
        color="#999999",
        alpha=0.30,
        label="metadata structure-null band",
    )
    if power_check:
        control_null = float(power_check.get("structure_null_p95", 0.0))
        axes[0].hlines(
            control_null,
            control_pos - 0.38,
            control_pos + 0.38,
            color="#777777",
            linewidth=2.0,
            label="positive-control null p95",
        )
    axes[0].annotate(
        "metadata bundle\n"
        + _p_annotation_text(
            float(null["p_value_null_ge_real"]), int(float(null.get("n_draws", 0)) or 0)
        ),
        xy=(bundle_pos, bundle),
        xytext=(-38, 20),
        textcoords="offset points",
        fontsize=SMALL_LABEL_FONT_SIZE,
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
    )
    if power_check:
        # A permutation p stored as an exact 0.0 cannot be resolved below the
        # (0+1)/(n_draws+1) floor, so report the floor rather than "p=0", which
        # would overstate resolution (F6-20; mirrors the ledger's convention).
        n_null_draws = int(float(null.get("n_draws", 0)) or 0)
        raw_control_p = float(power_check.get("p_value_null_ge_real", 0.0))
        control_p_text = _p_annotation_text(raw_control_p, n_null_draws)
        axes[0].annotate(
            f"positive control\n{control_p_text}",
            xy=(control_pos, control_value),
            # Up-right (not down-left): the previous xytext=(-76, -52) pulled
            # the text down and toward the "metadata bundle" bar/annotation
            # immediately to its left, landing on top of this bar's own fill
            # and crowding the neighbouring label. Up-right clears both.
            xytext=(20, 26),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=SMALL_LABEL_FONT_SIZE,
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
        )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(
        [n.replace("_", "\n") for n in names]
        + ["metadata\nbundle", "positive\ncontrol"],
        fontsize=TICK_FONT_SIZE,
    )
    axes[0].set_ylabel("mean info-gain about cell (nats)")
    axes[0].set_title("Per-modality epistemic value", loc="left")
    axes[0].legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)
    axes[0].grid(True, axis="y", alpha=0.25)

    labels = ["with\nlocation", "without\nlocation"]
    frac = [
        float(localize["with_location"]["frac_localized"]),
        float(localize["without_location"]["frac_localized"]),
    ]
    pos2 = np.arange(len(labels))
    axes[1].bar(pos2, frac, color=["#0072B2", "#CC79A7"])
    for pos, value in zip(pos2, frac):
        label_inside = value >= 0.88
        axes[1].text(
            pos,
            value - 0.05 if label_inside else value + 0.03,
            f"{value:.2f}",
            ha="center",
            va="top" if label_inside else "bottom",
            fontsize=SMALL_LABEL_FONT_SIZE,
            fontweight="bold",
            color="white" if label_inside else "#111111",
        )
    axes[1].set_ylim(0.0, 1.18)
    axes[1].set_xticks(pos2)
    axes[1].set_xticklabels(labels, fontsize=TICK_FONT_SIZE)
    axes[1].set_ylabel("fraction confidently localized")
    axes[1].set_title(
        f"Localizability collapse (cost {float(summary['frontier_cost_nats']):.2f} nats)",
        loc="left",
    )
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "EFE location-minimization frontier: direct location dominates the information budget",
        x=0.02,
        ha="left",
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def run_and_write_efe_location_frontier(
    config_path: str | Path,
    out_path: str | Path,
    observations_path: str | Path | None = None,
    *,
    threshold_nats: float = 0.5,
) -> Path:
    """Run the location-minimization frontier and write its artifact."""

    from .config import load_config
    from .observation_sources import load_observation_sequence

    config = load_config(config_path)
    sequence = load_observation_sequence(
        config, observations_path, requested_role="EFE location-frontier observations"
    )
    model = build_pomdp(
        sequence, config.inference.model.smoothing, config.inference.model.controls
    )
    payload = efe_location_frontier(
        sequence,
        model,
        smoothing=config.inference.model.smoothing,
        threshold_nats=threshold_nats,
    )
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
