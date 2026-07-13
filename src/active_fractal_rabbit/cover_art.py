"""Artifact-bound cover art generation for the rendered manuscript."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from .figure_style import (
    ANNOTATION_FONT_SIZE,
    METRIC_FONT_SIZE,
    SMALL_LABEL_FONT_SIZE,
    editorial_palette,
    gate_status_color,
    gate_verdict_label,
)

COVER_ART_PATH = Path("output/cover/active_fractal_rabbit_cover.png")

REQUIRED_COVER_ARTIFACTS: tuple[str, ...] = (
    "output/runs/default/observations.csv",
    "output/runs/default/discrete_features.csv",
    "output/runs/default/metrics.json",
    "output/runs/spatial_resolution_frontier.json",
    "output/runs/model_gate_matrix.json",
    "output/runs/forward_gates_summary.json",
    "output/runs/efe_location_frontier.json",
    "output/runs/linkage_risk_frontier.json",
    "output/runs/poisoning_robustness.json",
    "output/runs/pymdp_benchmark_matrix.json",
)


def render_cover_art(
    project_root: str | Path | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Render the evidence-atlas cover image from canonical generated artifacts."""

    root = Path(project_root or ".").resolve()
    output = root / (Path(out_path) if out_path else COVER_ART_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)

    for rel_path in REQUIRED_COVER_ARTIFACTS:
        path = root / rel_path
        if not path.is_file():
            raise FileNotFoundError(f"required cover artifact missing: {rel_path}")

    observations = _read_rows(root / "output/runs/default/observations.csv")
    features = _read_rows(root / "output/runs/default/discrete_features.csv")
    metrics = _read_json(root / "output/runs/default/metrics.json")
    spatial = _read_json(root / "output/runs/spatial_resolution_frontier.json")
    model_gates = _read_json(root / "output/runs/model_gate_matrix.json")
    forward_gates = _read_json(root / "output/runs/forward_gates_summary.json")
    efe = _read_json(root / "output/runs/efe_location_frontier.json")
    linkage = _read_json(root / "output/runs/linkage_risk_frontier.json")
    poisoning = _read_json(root / "output/runs/poisoning_robustness.json")

    bins = int(
        spatial.get("default_spatial_bins")
        or spatial.get("default_row", {}).get("spatial_bins")
        or 4
    )
    state_count = int(spatial.get("default_state_count") or bins * bins)
    n_reports = len(observations)
    n_travellers = len({row.get("ID", "") for row in observations})
    default_row = (
        spatial.get("default_row", {})
        if isinstance(spatial.get("default_row"), dict)
        else {}
    )
    full_loss = _fmt_float(default_row.get("full_location_mean_predictive_loss"))
    no_location_mass = _fmt_float(default_row.get("no_location_mean_mass_on_true_cell"))
    pymdp_delta = _fmt_sci(metrics.get("pymdp_exact_inference_max_abs_delta"))

    palette = editorial_palette()

    fig = plt.figure(figsize=(14.2, 8.4), facecolor=palette["paper"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _background(ax, palette)
    _draw_title_band(ax, palette, n_reports, n_travellers)
    _draw_lattice(ax, observations, features, bins, palette)
    _draw_model_regime_map(ax, palette)
    _draw_evidence_boundary(ax, palette, model_gates, forward_gates, efe, linkage, poisoning)
    _draw_metrics(
        ax,
        palette,
        [
            ("reports", f"{n_reports}"),
            ("travellers", f"{n_travellers}"),
            ("latent cells", f"{bins}x{bins} / {state_count}"),
            ("observed loss", full_loss),
            ("withheld mass", no_location_mass),
            ("pymdp delta", pymdp_delta),
        ],
    )

    ax.text(
        0.055,
        0.035,
        "Canonical synthetic artifacts only  •  no global SOTA, empirical mobility, or operational claim",
        ha="left",
        va="center",
        fontsize=ANNOTATION_FONT_SIZE,
        color=palette["muted"],
    )
    fig.savefig(
        output,
        dpi=260,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    return output


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_float(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_sci(value: object) -> str:
    try:
        return f"{float(value):.1e}"
    except (TypeError, ValueError):
        return "n/a"


def _background(ax, palette: dict[str, str]) -> None:
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, color=palette["paper"], zorder=0))
    bands = [
        (0.84, 0.12, 0.045),
        (0.265, 0.56, 0.030),
        (0.075, 0.16, 0.035),
    ]
    for y, height, alpha in bands:
        ax.add_patch(
            patches.Rectangle(
                (0.035, y),
                0.93,
                height,
                color=palette["ink"],
                alpha=alpha,
                zorder=0,
            )
        )
    for x in np.linspace(0.055, 0.945, 18):
        ax.plot(
            [x, x],
            [0.10, 0.90],
            color=palette["grid"],
            alpha=0.10,
            linewidth=0.55,
            zorder=0,
        )


def _draw_title_band(
    ax, palette: dict[str, str], n_reports: int, n_travellers: int
) -> None:
    ax.text(
        0.055,
        0.94,
        "Active FractalRabbit",
        ha="left",
        va="top",
        fontsize=31,
        fontweight="bold",
        color=palette["ink"],
    )
    ax.text(
        0.055,
        0.89,
        "Synthetic benchmark for belief filtering under sparse waypoint observations",
        ha="left",
        va="top",
        fontsize=15,
        color=palette["muted"],
    )
    ax.text(
        0.055,
        0.862,
        "Primary mechanism: soft marginalization under noisy emissions; AIF diagnostics remain regime-qualified.",
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        color=palette["muted"],
    )
    ax.text(
        0.825,
        0.915,
        f"{n_reports} reports\n{n_travellers} travellers",
        ha="left",
        va="center",
        fontsize=METRIC_FONT_SIZE,
        color=palette["ink"],
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.44,rounding_size=0.010",
            "facecolor": palette["panel"],
            "edgecolor": palette["grid"],
            "linewidth": 1.0,
        },
    )


def _draw_lattice(
    ax,
    observations: list[dict[str, str]],
    features: list[dict[str, str]],
    bins: int,
    palette: dict[str, str],
) -> None:
    left, bottom, width, height = 0.055, 0.315, 0.465, 0.455
    ax.text(
        left,
        bottom + height + 0.030,
        "Dominant synthetic waypoint lattice",
        fontsize=15,
        fontweight="bold",
        color=palette["ink"],
        ha="left",
    )
    ax.text(
        left,
        bottom + height + 0.006,
        "Binned reports define state; paths show order only.",
        fontsize=ANNOTATION_FONT_SIZE,
        color=palette["muted"],
        ha="left",
    )
    grid = np.zeros((bins, bins), dtype=float)
    for row in features:
        try:
            x_bin = min(bins - 1, max(0, int(float(row["x_bin"]))))
            y_bin = min(bins - 1, max(0, int(float(row["y_bin"]))))
        except (KeyError, ValueError):
            continue
        grid[y_bin, x_bin] += 1.0
    max_count = float(grid.max()) if grid.size else 1.0
    if max_count <= 0:
        max_count = 1.0
    cmap = LinearSegmentedColormap.from_list(
        "afr_cover_occupancy",
        ["#f9f5e7", "#d9e8df", "#8ec4b1", palette["teal"]],
    )
    cell_w = width / bins
    cell_h = height / bins
    for y_idx in range(bins):
        for x_idx in range(bins):
            value = grid[y_idx, x_idx] / max_count
            ax.add_patch(
                patches.Rectangle(
                    (left + x_idx * cell_w, bottom + y_idx * cell_h),
                    cell_w,
                    cell_h,
                    facecolor=cmap(value),
                    edgecolor=palette["grid"],
                    linewidth=1.15,
                    zorder=1,
                )
            )
    xs = np.array([float(row["x(km)"]) for row in observations])
    ys = np.array([float(row["y(km)"]) for row in observations])
    x_span = max(float(xs.max() - xs.min()), 1.0)
    y_span = max(float(ys.max() - ys.min()), 1.0)
    colors = [palette["blue"], palette["green"], palette["amber"], palette["violet"]]
    by_id: dict[str, list[dict[str, str]]] = {}
    for row in observations:
        by_id.setdefault(row.get("ID", ""), []).append(row)
    for idx, (traveller_id, rows) in enumerate(sorted(by_id.items())):
        ordered = sorted(rows, key=lambda row: float(row.get("Days", 0.0)))
        x = [
            left + (float(row["x(km)"]) - float(xs.min())) / x_span * width
            for row in ordered
        ]
        y = [
            bottom + (float(row["y(km)"]) - float(ys.min())) / y_span * height
            for row in ordered
        ]
        color = colors[idx % len(colors)]
        ax.plot(x, y, color=color, alpha=0.50, linewidth=1.9, zorder=2)
        ax.scatter(
            x[:: max(1, len(x) // 32)],
            y[:: max(1, len(y) // 32)],
            s=16,
            color=color,
            alpha=0.78,
            zorder=3,
        )
        if ordered:
            ax.scatter(
                [x[0]],
                [y[0]],
                marker="^",
                s=58,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
            )
            ax.scatter(
                [x[-1]],
                [y[-1]],
                marker="*",
                s=82,
                color=color,
                edgecolor=palette["ink"],
                linewidth=0.45,
                zorder=4,
            )
    ax.add_patch(
        patches.Rectangle(
            (left, bottom),
            width,
            height,
            fill=False,
            edgecolor=palette["ink"],
            linewidth=1.15,
            zorder=5,
        )
    )


def _draw_artifact_stack(ax, palette: dict[str, str]) -> None:
    x = 0.56
    y_values = [0.720, 0.660, 0.600, 0.540, 0.480, 0.420]
    labels = [
        "pinned generator",
        "waypoint records",
        "categorical tensors",
        "belief traces",
        "claim gates",
        "release assets",
    ]
    colors = [
        palette["muted"],
        palette["teal"],
        palette["blue"],
        palette["green"],
        palette["amber"],
        palette["violet"],
    ]
    for i, (y, label, color) in enumerate(zip(y_values, labels, colors)):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y - 0.024),
                0.205,
                0.046,
                boxstyle="round,pad=0.008,rounding_size=0.008",
                facecolor=palette["panel"],
                edgecolor=color,
                linewidth=1.45,
                zorder=4,
            )
        )
        ax.text(
            x + 0.016,
            y,
            label,
            ha="left",
            va="center",
            fontsize=SMALL_LABEL_FONT_SIZE,
            color=palette["ink"],
            zorder=5,
        )
        if i < len(y_values) - 1:
            ax.annotate(
                "",
                xy=(x + 0.103, y_values[i + 1] + 0.027),
                xytext=(x + 0.103, y - 0.028),
                arrowprops={
                    "arrowstyle": "->",
                    "color": palette["muted"],
                    "lw": 0.85,
                    "alpha": 0.60,
                },
                zorder=3,
            )
    ax.text(
        x,
        0.776,
        "Artifact pipeline",
        fontsize=14.5,
        fontweight="bold",
        color=palette["ink"],
        ha="left",
    )


def _draw_model_regime_map(ax, palette: dict[str, str]) -> None:
    """Put the paper's model-selection result—not its build pipeline—on the cover."""

    x, width = 0.555, 0.390
    ax.text(
        x,
        0.786,
        "Where each model family earns its keep",
        fontsize=15,
        fontweight="bold",
        color=palette["ink"],
        ha="left",
    )
    ax.text(
        x,
        0.760,
        "The observation regime decides; there is no global winner.",
        fontsize=ANNOTATION_FONT_SIZE,
        color=palette["muted"],
        ha="left",
    )
    rows = [
        (
            "CLEAN / DIRECT",
            "Point estimates suffice",
            "Persistence + Markov",
            "belief adds no general gain",
            palette["blue"],
        ),
        (
            "NOISY / HIDDEN",
            "Preserve uncertainty",
            "Soft · HMM · particle · AIF",
            "paired tie vs strongest non-AIF",
            palette["teal"],
        ),
        (
            "SWITCHING / GAPPED",
            "Adapt or stay simple",
            "Online base rate + controls",
            "fixed AIF does not promote",
            palette["amber"],
        ),
    ]
    for idx, (tag, lead, models, boundary, color) in enumerate(rows):
        y = 0.680 - idx * 0.140
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y - 0.047),
                width,
                0.103,
                boxstyle="round,pad=0.010,rounding_size=0.012",
                facecolor=palette["panel"],
                edgecolor=color,
                linewidth=1.8,
                zorder=3,
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (x, y - 0.047), 0.012, 0.103, facecolor=color, edgecolor="none", zorder=4
            )
        )
        ax.text(x + 0.025, y + 0.031, tag, fontsize=10.5, fontweight="bold", color=color)
        ax.text(x + 0.025, y + 0.006, lead, fontsize=14, fontweight="bold", color=palette["ink"])
        ax.text(x + 0.025, y - 0.021, models, fontsize=11.5, color=palette["ink"])
        ax.text(x + 0.217, y - 0.021, boundary, fontsize=10.5, color=palette["muted"])


def _draw_evidence_boundary(
    ax,
    palette: dict[str, str],
    model_gates: dict[str, Any],
    forward_gates: dict[str, Any],
    efe: dict[str, Any],
    linkage: dict[str, Any],
    poisoning: dict[str, Any],
) -> None:
    """Compact artifact-status strip beneath the conceptual model map."""

    statuses = [
        model_gates.get("status") == "completed",
        forward_gates.get("status") == "completed",
        bool(efe.get("verdict")),
        bool(linkage.get("verdict")),
        bool(poisoning.get("verdict")),
    ]
    color = palette["teal"] if all(statuses) else palette["amber"]
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.555, 0.277),
            0.390,
            0.045,
            boxstyle="round,pad=0.010,rounding_size=0.012",
            facecolor=palette["panel"],
            edgecolor=color,
            linewidth=1.4,
        )
    )
    ax.text(0.575, 0.307, "EVIDENCE BOUNDARY", fontsize=9.5, fontweight="bold", color=color)
    ax.text(
        0.575,
        0.287,
        "Synthetic gates complete  ·  promotion remains blocked",
        fontsize=10.8,
        fontweight="bold",
        color=palette["ink"],
    )


def _draw_frontier(ax, spatial: dict[str, Any], palette: dict[str, str]) -> None:
    inset = ax.inset_axes([0.545, 0.205, 0.245, 0.145])
    rows = [row for row in spatial.get("rows", []) if isinstance(row, dict)]
    secondary = inset.twinx()
    if rows:
        x = [float(row["spatial_bins"]) for row in rows]
        full_loss = [float(row["full_location_mean_predictive_loss"]) for row in rows]
        no_location = [
            float(row.get("no_location_mean_mass_on_true_cell", 0.0)) for row in rows
        ]
        empty = [float(row["empty_state_count"]) for row in rows]
        inset.plot(
            x,
            full_loss,
            color=palette["teal"],
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="loss",
        )
        secondary.plot(
            x,
            no_location,
            color=palette["violet"],
            linewidth=1.55,
            marker="s",
            markersize=3.5,
            linestyle="--",
            label="metadata",
        )
        default_bins = int(spatial.get("default_spatial_bins", 4) or 4)
        inset.axvline(
            default_bins,
            color=palette["ink"],
            linewidth=0.8,
            linestyle="--",
            alpha=0.55,
        )
        flagged = [
            (bins, loss)
            for bins, loss, empty_count in zip(x, full_loss, empty)
            if empty_count > 0
        ]
        if flagged:
            flagged_x, flagged_y = zip(*flagged)
            inset.scatter(
                flagged_x,
                flagged_y,
                marker="*",
                s=42,
                color=palette["amber"],
                edgecolor=palette["ink"],
                linewidth=0.35,
                zorder=5,
            )
            inset.text(
                0.98,
                0.94,
                "* empty state(s)",
                transform=inset.transAxes,
                ha="right",
                va="top",
                fontsize=SMALL_LABEL_FONT_SIZE * 0.8,
                color=palette["ink"],
            )
        confirmation = spatial.get("larger_sample_confirmation", {})
        if isinstance(confirmation, dict) and confirmation:
            status = str(confirmation.get("status", "unavailable"))
            n_observations = int(confirmation.get("n_observations", 0) or 0)
            stability = (
                confirmation.get("sample_size_ladder_summary", {})
                if isinstance(confirmation.get("sample_size_ladder_summary"), dict)
                else {}
            )
            multipliers = stability.get("multipliers", [])
            if isinstance(multipliers, list) and multipliers:
                multiplier_label = "/".join(f"{int(item)}x" for item in multipliers)
                guardrail = (
                    "guardrail stable"
                    if stability.get("all_same_no_location_guardrail_direction") is True
                    else "guardrail sensitive"
                )
                text = f"{multiplier_label}: {guardrail}; max n={n_observations}"
            else:
                multiplier = int(confirmation.get("sample_multiplier", 0) or 0)
                multiplier_label = f"{multiplier}x" if multiplier else "larger"
                text = f"{multiplier_label} check: {status} (n={n_observations})"
            inset.text(
                0.02,
                0.05,
                text,
                transform=inset.transAxes,
                ha="left",
                va="bottom",
                fontsize=SMALL_LABEL_FONT_SIZE * 0.82,
                color=palette["ink"],
                bbox={
                    "facecolor": "#fbfaf5",
                    "edgecolor": "none",
                    "alpha": 0.80,
                    "pad": 1.2,
                },
            )
    inset.set_facecolor("#fbfaf5")
    inset.set_title(
        "resolution frontier",
        loc="left",
        fontsize=SMALL_LABEL_FONT_SIZE,
        color=palette["ink"],
        pad=4,
    )
    inset.set_xlabel("")
    inset.set_ylabel(
        "loss (nats)",
        fontsize=SMALL_LABEL_FONT_SIZE * 0.72,
        color=palette["teal"],
        labelpad=1.5,
    )
    secondary.set_ylabel(
        "mass (0-1)",
        fontsize=SMALL_LABEL_FONT_SIZE * 0.72,
        color=palette["violet"],
        labelpad=1.5,
    )
    inset.tick_params(
        labelsize=SMALL_LABEL_FONT_SIZE * 0.82,
        labelbottom=False,
        colors=palette["teal"],
        pad=1,
    )
    secondary.tick_params(
        labelsize=SMALL_LABEL_FONT_SIZE * 0.82,
        colors=palette["violet"],
        pad=1,
    )
    inset.grid(True, linewidth=0.35, alpha=0.35)
    for spine in inset.spines.values():
        spine.set_color(palette["grid"])
    for spine in secondary.spines.values():
        spine.set_color(palette["grid"])


def _draw_gate_summary(
    ax,
    palette: dict[str, str],
    model_gates: dict[str, Any],
    forward_gates: dict[str, Any],
    efe: dict[str, Any],
    linkage: dict[str, Any],
    poisoning: dict[str, Any],
) -> None:
    x0, y_top = 0.795, 0.720
    ax.text(
        x0,
        y_top + 0.056,
        "Claim guardrails",
        fontsize=14.5,
        fontweight="bold",
        color=palette["ink"],
        ha="left",
    )
    rows = [
        (
            "model matrix",
            str(model_gates.get("status", "unknown")),
            gate_status_color(model_gates.get("status", "unknown"), palette),
        ),
        (
            "forward gates",
            str(forward_gates.get("status", "unknown")),
            gate_status_color(forward_gates.get("status", "unknown"), palette),
        ),
        (
            "minimization",
            gate_verdict_label(efe.get("verdict", "unknown")),
            palette["amber"],
        ),
        (
            "linkage risk",
            gate_verdict_label(linkage.get("verdict", "unknown")),
            palette["red"],
        ),
        (
            "poisoning",
            gate_verdict_label(poisoning.get("verdict", "unknown")),
            palette["violet"],
        ),
    ]
    for idx, (label, value, color) in enumerate(rows):
        y = y_top - idx * 0.074
        ax.add_patch(
            patches.Circle(
                (x0 + 0.009, y), 0.0075, facecolor=color, edgecolor="none", zorder=5
            )
        )
        ax.text(
            x0 + 0.026,
            y + 0.012,
            label,
            fontsize=SMALL_LABEL_FONT_SIZE,
            fontweight="bold",
            color=palette["ink"],
            va="center",
            ha="left",
        )
        ax.text(
            x0 + 0.026,
            y - 0.012,
            _shorten(value, 30),
            fontsize=SMALL_LABEL_FONT_SIZE,
            color=palette["muted"],
            va="center",
            ha="left",
        )


def _draw_metrics(ax, palette: dict[str, str], rows: list[tuple[str, str]]) -> None:
    left, top = 0.055, 0.185
    ax.text(
        left,
        top + 0.055,
        "Artifact-bound surface",
        fontsize=14.5,
        fontweight="bold",
        color=palette["ink"],
        ha="left",
    )
    for idx, (label, value) in enumerate(rows):
        x = left + idx * 0.151
        ax.add_patch(
            patches.FancyBboxPatch(
                (x - 0.006, top - 0.060),
                0.136,
                0.070,
                boxstyle="round,pad=0.008,rounding_size=0.008",
                facecolor=palette["panel"],
                edgecolor=palette["grid"],
                linewidth=0.9,
                zorder=1,
            )
        )
        ax.text(
            x,
            top - 0.008,
            label,
            fontsize=SMALL_LABEL_FONT_SIZE,
            color=palette["muted"],
            ha="left",
            va="center",
            zorder=2,
        )
        ax.text(
            x,
            top - 0.038,
            value,
            fontsize=METRIC_FONT_SIZE,
            fontweight="bold",
            color=palette["ink"],
            ha="left",
            va="center",
            zorder=2,
        )


def _shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."
