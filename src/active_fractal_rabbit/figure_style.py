"""Shared manuscript figure style contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FigureFontContract:
    """Font sizes used by manuscript figures."""

    base: float = 17.0
    axis_title: float = 19.0
    figure_title: float = 21.0
    axis_label: float = 18.0
    tick: float = 16.0
    legend: float = 15.5
    annotation_minimum: float = 15.0


FONT_CONTRACT = FigureFontContract()
MIN_READABLE_FONT_SIZE = FONT_CONTRACT.annotation_minimum
TICK_FONT_SIZE = FONT_CONTRACT.tick
LEGEND_FONT_SIZE = FONT_CONTRACT.legend
ANNOTATION_FONT_SIZE = FONT_CONTRACT.annotation_minimum
SMALL_LABEL_FONT_SIZE = FONT_CONTRACT.annotation_minimum
METRIC_FONT_SIZE = 17.0
PANEL_HEADING_FONT_SIZE = 17.5
CANVAS_TITLE_FONT_SIZE = 19.5

#: Unified colormaps so the figure set reads as one editorial system rather than an
#: ad-hoc mix. SEQUENTIAL for ordinal/magnitude heatmaps; DIVERGING for signed maps
#: centred at zero (e.g. advantage-over-baseline).
#: DIVERGING is ColorBrewer RdBu (reversed: red = positive/better, blue = negative),
#: which is colourblind-safe under deuteranopia/protanopia — red/blue stay separable by
#: hue+luminance where a red/green ramp (RdYlGn) would not. Sign is also carried by the
#: in-cell numeric labels, so the encoding survives greyscale.
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"


def editorial_palette(palette: dict[str, str] | None = None) -> dict[str, str]:
    """Return shared atlas colors for cover and manuscript overview figures."""

    source = palette or {}
    return {
        "ink": "#1D1D1D",
        "muted": "#555555",
        "paper": "#FAFAF7",
        "panel": "#FFFFFF",
        "grid": "#D7D7D7",
        "blue": source.get("blue", "#0072B2"),
        "green": source.get("green", "#009E73"),
        "teal": source.get("teal", "#009E73"),
        "amber": source.get("orange", "#E69F00"),
        "red": source.get("red", "#D55E00"),
        "sky": source.get("sky", "#56B4E9"),
        "yellow": source.get("yellow", "#F0E442"),
        "violet": source.get("purple", "#CC79A7"),
        "gray": source.get("gray", "#595959"),
    }


#: Shared verdict/status -> human-readable label mapping so the cover art and the
#: evidence-atlas figure never describe the same gate outcome with different wording.
GATE_VERDICT_LABELS: dict[str, str] = {
    "metadata_recovery_not_above_structure_null": "metadata below structure null",
    "metadata_recovery_exceeds_structure_null": "metadata exceeds structure null",
    "location_carries_relink_risk_that_minimization_removes": "location risk; minimized",
    "poisoning_detection_regime_changed": "detected regime change",
}


def gate_verdict_label(value: object) -> str:
    """Map a gate verdict/status string to shared human-readable text."""

    raw = str(value)
    return GATE_VERDICT_LABELS.get(raw, raw.replace("_", " "))


def gate_status_color(value: object, palette: dict[str, str]) -> str:
    """Derive a guardrail chip colour from a gate status so a non-completed
    status can never render the same green as a completed one (F6-6)."""

    status = str(value).strip().lower()
    if status == "completed":
        return palette["green"]
    if status in {"partial", "pending", "in_progress"}:
        return palette["amber"]
    if status in {"unavailable", "failed", "error", "blocked", "unknown"}:
        return palette["red"]
    return palette.get("muted", palette["amber"])


def font_contract_payload() -> dict[str, float]:
    """Return the serializable manuscript figure font contract."""

    return asdict(FONT_CONTRACT) | {"minimum_declared_readable": MIN_READABLE_FONT_SIZE}


def manuscript_rcparams(palette: dict[str, str]) -> dict[str, object]:
    """Return print-readable Matplotlib rcParams shared by manuscript PNGs."""

    return {
        "font.family": "DejaVu Sans",
        "font.size": FONT_CONTRACT.base,
        "axes.titlesize": FONT_CONTRACT.axis_title,
        "axes.labelsize": FONT_CONTRACT.axis_label,
        "axes.titleweight": "semibold",
        "figure.titlesize": FONT_CONTRACT.figure_title,
        "figure.titleweight": "semibold",
        "legend.fontsize": FONT_CONTRACT.legend,
        "legend.frameon": False,
        "xtick.labelsize": FONT_CONTRACT.tick,
        "ytick.labelsize": FONT_CONTRACT.tick,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": palette.get("gray", "#595959"),
        "axes.labelcolor": "#1D1D1D",
        "xtick.color": "#1D1D1D",
        "ytick.color": "#1D1D1D",
        "grid.color": palette.get("gray", "#595959"),
        "grid.alpha": 0.18,
        "grid.linewidth": 0.7,
        "lines.linewidth": 2.2,
        "lines.markersize": 7.0,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
