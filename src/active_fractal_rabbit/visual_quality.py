"""Artifact-bound quality checks for generated manuscript figures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np


VISUAL_QUALITY_SCHEMA = "active-fractal-rabbit-visual-quality-audit-v1"
CONTACT_SHEET_PATH = "output/figures/figure_contact_sheet.png"
LAYOUT_CRITICAL_FIGURE_REASONS = {
    "confound_taxonomy": "dense categorical labels and top-row annotation clearance",
    "multiplicity_ledger": "log-scale p-value labels, threshold line, and legend clearance",
    "privacy_utility_frontier": "clustered direct labels on the minimization frontier",
    "linkage_risk_frontier": "bar labels against panel titles and null bands",
    "sota_leaderboard": "long candidate names, CI whiskers, and baseline annotation",
    "model_comparison_under_noise": "ranked bars with long model names and title wrap",
    "aif_regime_map": "per-regime statistical labels and local-versus-Holm gate text",
    "fair_inference": "fair-ladder oracle labels, CI annotations, and boundary text",
    "fair_power_regime": "power-regime confidence labels and synthetic-scope annotation",
    "partial_observability": "noise-regime labels, CI bands, and belief/nonbelief comparison text",
    "spatial_resolution_frontier": "frontier callouts and selected-default annotation",
    "privacy_utility_repeats": "repeated-row status labels and utility-spread annotations",
    "synthetic_task_utility_privacy": "threshold lines, no-operating-point labels, and dual utility/risk axes",
    "aif_deepening_candidates": "long protocol labels and audit-only gate annotations",
}
LAYOUT_CRITICAL_FIGURE_KEYS = tuple(LAYOUT_CRITICAL_FIGURE_REASONS)


class VisualQualityError(RuntimeError):
    """Raised when generated figures fail the visual-quality contract."""


@dataclass(frozen=True)
class VisualQualityThresholds:
    """Minimum acceptable static-figure properties for manuscript delivery."""

    min_width_px: int = 1100
    min_height_px: int = 650
    min_layout_critical_width_px: int = 1600
    min_layout_critical_height_px: int = 850
    min_bytes: int = 1_000
    min_pixel_stddev: float = 0.015
    min_caption_chars: int = 100
    min_alt_chars: int = 40
    min_declared_font_size: float = 15.0
    min_base_font_size: float = 17.0
    min_tick_font_size: float = 16.0
    min_legend_font_size: float = 15.5


def audit_visual_quality(
    project_root: str | Path,
    *,
    figures_dir: str | Path = "output/figures",
    thresholds: VisualQualityThresholds = VisualQualityThresholds(),
) -> dict[str, Any]:
    """Return a fail-loud audit payload for the registered figure set."""

    root = Path(project_root).resolve()
    figure_root = root / figures_dir
    registry_path = figure_root / "figure_registry.json"
    if not registry_path.is_file():
        raise VisualQualityError(
            f"missing figure registry: {registry_path.relative_to(root)}"
        )

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_schema = str(registry.get("schema", ""))
    entries = registry.get("entries")
    if not isinstance(entries, list) or not entries:
        raise VisualQualityError(
            "figure registry must contain a non-empty entries list"
        )
    style_path = figure_root / "style_provenance.json"
    style_payload, style_violations = _audit_style_provenance(
        root, style_path, thresholds
    )

    rows: list[dict[str, Any]] = []
    violations: list[str] = list(style_violations)
    seen_keys: set[str] = set()
    seen_filenames: set[str] = set()

    for entry in entries:
        row = _audit_entry(
            root, figure_root, entry, thresholds, registry_schema=registry_schema
        )
        key = str(row["key"])
        filename = str(row["filename"])
        if key in seen_keys:
            row["violations"].append("duplicate registry key")
        if filename in seen_filenames:
            row["violations"].append("duplicate registry filename")
        seen_keys.add(key)
        seen_filenames.add(filename)
        rows.append(row)
        violations.extend(f"{key}: {item}" for item in row["violations"])

    unexpected_pngs = sorted(
        path.name
        for path in figure_root.glob("*.png")
        if path.name != Path(CONTACT_SHEET_PATH).name
        and path.name not in seen_filenames
    )
    violations.extend(
        f"unregistered figure image: {figure_root.relative_to(root) / filename}"
        for filename in unexpected_pngs
    )

    readable_rows = [row for row in rows if row["image_readable"]]
    widths = [int(row["width_px"]) for row in readable_rows]
    heights = [int(row["height_px"]) for row in readable_rows]
    stddevs = [float(row["pixel_stddev"]) for row in readable_rows]
    aspect_ratios = [
        float(row["width_px"]) / float(row["height_px"])
        for row in readable_rows
        if float(row["height_px"]) > 0
    ]
    source_bound = sum(1 for row in rows if row["caption_has_source_language"])
    contact_sheet, contact_violations = _audit_contact_sheet(
        root, figure_root, readable_rows, thresholds
    )
    violations.extend(contact_violations)
    quality_summary = _quality_summary(readable_rows, contact_sheet)
    layout_critical = _layout_critical_figures(rows, thresholds)
    violations.extend(layout_critical["violations"])
    payload: dict[str, Any] = {
        "schema": VISUAL_QUALITY_SCHEMA,
        "status": "pass" if not violations else "fail",
        "registry_path": str(registry_path.relative_to(root)),
        "style_path": str(style_path.relative_to(root)),
        "font_contract": style_payload.get("font_contract", {}),
        "figure_count": len(rows),
        "unexpected_pngs": unexpected_pngs,
        "caption_source_language_count": source_bound,
        "min_width_px": min(widths) if widths else 0,
        "min_height_px": min(heights) if heights else 0,
        "min_aspect_ratio": min(aspect_ratios) if aspect_ratios else 0.0,
        "max_aspect_ratio": max(aspect_ratios) if aspect_ratios else 0.0,
        "min_pixel_stddev": min(stddevs) if stddevs else 0.0,
        "quality_summary": quality_summary,
        "layout_critical_figures": layout_critical,
        "thresholds": {
            "min_width_px": thresholds.min_width_px,
            "min_height_px": thresholds.min_height_px,
            "min_layout_critical_width_px": thresholds.min_layout_critical_width_px,
            "min_layout_critical_height_px": thresholds.min_layout_critical_height_px,
            "min_bytes": thresholds.min_bytes,
            "min_pixel_stddev": thresholds.min_pixel_stddev,
            "min_caption_chars": thresholds.min_caption_chars,
            "min_alt_chars": thresholds.min_alt_chars,
            "min_declared_font_size": thresholds.min_declared_font_size,
            "min_base_font_size": thresholds.min_base_font_size,
            "min_tick_font_size": thresholds.min_tick_font_size,
            "min_legend_font_size": thresholds.min_legend_font_size,
        },
        "violations": violations,
        "figures": rows,
    }
    return payload


def write_visual_quality_audit(
    project_root: str | Path,
    *,
    out_path: str | Path = "output/runs/visual_quality_audit.json",
    figures_dir: str | Path = "output/figures",
    fail_on_violations: bool = True,
) -> Path:
    """Write the visual-quality audit and optionally raise on failures."""

    root = Path(project_root).resolve()
    payload = audit_visual_quality(root, figures_dir=figures_dir)
    output = root / out_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if fail_on_violations and payload["violations"]:
        raise VisualQualityError("; ".join(payload["violations"]))
    return output


def _audit_style_provenance(
    root: Path,
    style_path: Path,
    thresholds: VisualQualityThresholds,
) -> tuple[dict[str, Any], list[str]]:
    if not style_path.is_file():
        return {}, [f"missing style provenance: {style_path.relative_to(root)}"]

    violations: list[str] = []
    try:
        style_payload = json.loads(style_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, [f"style provenance unreadable: {exc}"]

    font_contract = style_payload.get("font_contract")
    if not isinstance(font_contract, dict):
        return style_payload, ["style provenance missing font_contract"]

    required = {
        "base": thresholds.min_base_font_size,
        "axis_title": thresholds.min_declared_font_size,
        "figure_title": thresholds.min_declared_font_size,
        "axis_label": thresholds.min_declared_font_size,
        "tick": thresholds.min_tick_font_size,
        "legend": thresholds.min_legend_font_size,
        "annotation_minimum": thresholds.min_declared_font_size,
        "minimum_declared_readable": thresholds.min_declared_font_size,
    }
    for field, floor in required.items():
        raw_value = font_contract.get(field)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            violations.append(f"font contract field {field} is not numeric")
            continue
        if value < floor:
            violations.append(f"font contract field {field} below readable floor")

    numeric_values: list[float] = []
    for raw_value in font_contract.values():
        if isinstance(raw_value, (int, float)):
            numeric_values.append(float(raw_value))
    if numeric_values and min(numeric_values) < thresholds.min_declared_font_size:
        violations.append(
            "font contract declares a readable font below the minimum floor"
        )

    return style_payload, violations


def _read_image_pixel_metrics(path: Path) -> tuple[int, int, float]:
    """Read an image file and return ``(width_px, height_px, pixel_stddev)``.

    Single source for the image-loading + RGB-pixel-stddev block previously
    duplicated between ``_audit_entry`` and ``_audit_contact_sheet``. Raises
    whatever ``matplotlib.image.imread`` raises on an unreadable file; each
    caller catches it and reports the violation in its own message format.
    """

    image = mpimg.imread(path)
    image_array = np.asarray(image, dtype=float)
    if image_array.ndim == 2:
        height, width = image_array.shape
        rgb = image_array
    else:
        height, width = image_array.shape[:2]
        rgb = image_array[..., :3]
    return int(width), int(height), float(np.std(rgb))


def _audit_entry(
    root: Path,
    figure_root: Path,
    entry: dict[str, Any],
    thresholds: VisualQualityThresholds,
    *,
    registry_schema: str = "active-fractal-rabbit-figure-registry-v1",
) -> dict[str, Any]:
    key = str(entry.get("key", ""))
    filename = str(entry.get("filename", ""))
    caption = str(entry.get("caption", ""))
    alt_text = str(entry.get("alt_text", ""))
    source = str(entry.get("source", ""))
    data_derived = bool(entry.get("data_derived", True))
    path = figure_root / filename
    violations: list[str] = []

    if not key:
        violations.append("missing key")
    if not filename:
        violations.append("missing filename")
    if len(caption) < thresholds.min_caption_chars:
        violations.append("caption too short")
    if len(alt_text) < thresholds.min_alt_chars:
        violations.append("alt text too short")
    caption_has_source = _has_source_language(caption)
    if not caption_has_source:
        violations.append("caption lacks source or boundary language")
    if not source:
        violations.append("missing source")
    if data_derived:
        if not source.startswith("output/"):
            violations.append("data-derived source is not an output artifact")
        elif Path(source).suffix not in {".csv", ".json"}:
            violations.append("data-derived source is not a CSV or JSON artifact")
        elif not (root / source).is_file():
            violations.append("data-derived source artifact is missing")

    width = height = 0
    pixel_stddev = 0.0
    image_readable = False
    image_bytes = path.stat().st_size if path.is_file() else 0
    if not path.is_file():
        violations.append("image file missing")
    else:
        try:
            width, height, pixel_stddev = _read_image_pixel_metrics(path)
            image_readable = True
        except Exception as exc:  # pragma: no cover - exercised by corrupt files only
            violations.append(f"image unreadable: {exc}")
    if image_readable:
        if width < thresholds.min_width_px:
            violations.append("image width below threshold")
        if height < thresholds.min_height_px:
            violations.append("image height below threshold")
        if pixel_stddev < thresholds.min_pixel_stddev:
            violations.append("image appears visually blank")
    if image_bytes < thresholds.min_bytes:
        violations.append("image bytes below threshold")
    if registry_schema == "active-fractal-rabbit-figure-registry-v2":
        image_sha = _sha256_file(path) if path.is_file() else None
        source_path = root / source
        source_sha = _sha256_file(source_path) if source_path.is_file() else None
        if entry.get("image_sha256") != image_sha:
            violations.append("image checksum differs from generated registry binding")
        if entry.get("source_sha256") != source_sha:
            violations.append("source checksum differs from generated registry binding")
        if image_sha is not None:
            binding = hashlib.sha256()
            binding.update(key.encode("utf-8"))
            binding.update(caption.encode("utf-8"))
            binding.update(image_sha.encode("ascii"))
            binding.update((source_sha or "non-file-source").encode("ascii"))
            if entry.get("binding_sha256") != binding.hexdigest():
                violations.append("figure/source/caption binding checksum mismatch")

    return {
        "key": key,
        "filename": filename,
        "path": str(path.relative_to(root)) if path.is_absolute() else str(path),
        "source": source,
        "data_derived": data_derived,
        "caption_chars": len(caption),
        "alt_chars": len(alt_text),
        "caption_has_source_language": caption_has_source,
        "image_readable": image_readable,
        "bytes": image_bytes,
        "width_px": width,
        "height_px": height,
        "aspect_ratio": float(width) / float(height) if height else 0.0,
        "pixel_stddev": pixel_stddev,
        "violations": violations,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_contact_sheet(
    root: Path,
    figure_root: Path,
    rows: list[dict[str, Any]],
    thresholds: VisualQualityThresholds,
) -> tuple[dict[str, Any], list[str]]:
    path = figure_root / "figure_contact_sheet.png"
    summary: dict[str, Any] = {
        "path": CONTACT_SHEET_PATH,
        "exists": path.is_file(),
        "image_readable": False,
        "current": False,
        "width_px": 0,
        "height_px": 0,
        "bytes": path.stat().st_size if path.is_file() else 0,
        "pixel_stddev": 0.0,
    }
    violations: list[str] = []
    if not path.is_file():
        return summary, [f"missing manual review contact sheet: {CONTACT_SHEET_PATH}"]

    try:
        width, height, pixel_stddev = _read_image_pixel_metrics(path)
        summary.update(
            {
                "image_readable": True,
                "width_px": width,
                "height_px": height,
                "pixel_stddev": pixel_stddev,
            }
        )
    except Exception as exc:  # pragma: no cover - exercised by corrupt files only
        violations.append(f"manual review contact sheet unreadable: {exc}")

    if summary["bytes"] < thresholds.min_bytes:
        violations.append("manual review contact sheet bytes below threshold")
    if summary["image_readable"]:
        if int(summary["width_px"]) < thresholds.min_width_px:
            violations.append("manual review contact sheet width below threshold")
        if int(summary["height_px"]) < thresholds.min_height_px:
            violations.append("manual review contact sheet height below threshold")
        if float(summary["pixel_stddev"]) < thresholds.min_pixel_stddev:
            violations.append("manual review contact sheet appears visually blank")

    registered_paths = [
        root / str(row["path"]) for row in rows if (root / str(row["path"])).is_file()
    ]
    newest_registered_mtime_ns = max(
        (item.stat().st_mtime_ns for item in registered_paths), default=0
    )
    current = path.stat().st_mtime_ns >= newest_registered_mtime_ns
    summary["current"] = current
    if not current:
        violations.append(
            "manual review contact sheet is older than one or more registered figures"
        )
    return summary, violations


def _quality_summary(
    rows: list[dict[str, Any]], contact_sheet: dict[str, Any]
) -> dict[str, Any]:
    if not rows:
        return {
            "shortest_figure": None,
            "narrowest_figure": None,
            "lowest_pixel_stddev_figure": None,
            "widest_aspect_figure": None,
            "tallest_aspect_figure": None,
            "manual_review_surface": CONTACT_SHEET_PATH,
            "manual_review_contact_sheet": contact_sheet,
        }

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "key": row["key"],
            "path": row["path"],
            "width_px": row["width_px"],
            "height_px": row["height_px"],
            "aspect_ratio": row["aspect_ratio"],
            "pixel_stddev": row["pixel_stddev"],
        }

    return {
        "shortest_figure": compact(min(rows, key=lambda row: int(row["height_px"]))),
        "narrowest_figure": compact(min(rows, key=lambda row: int(row["width_px"]))),
        "lowest_pixel_stddev_figure": compact(
            min(rows, key=lambda row: float(row["pixel_stddev"]))
        ),
        "widest_aspect_figure": compact(
            max(rows, key=lambda row: float(row["aspect_ratio"]))
        ),
        "tallest_aspect_figure": compact(
            min(rows, key=lambda row: float(row["aspect_ratio"]))
        ),
        "manual_review_surface": CONTACT_SHEET_PATH,
        "manual_review_contact_sheet": contact_sheet,
    }


def _layout_critical_figures(
    rows: list[dict[str, Any]],
    thresholds: VisualQualityThresholds,
) -> dict[str, Any]:
    by_key = {str(row["key"]): row for row in rows}
    figures = []
    missing_keys = []
    violations = []
    for key in LAYOUT_CRITICAL_FIGURE_KEYS:
        row = by_key.get(key)
        if row is None:
            missing_keys.append(key)
            violations.append(f"{key}: layout-critical figure missing from registry")
            continue
        width_px = int(row["width_px"])
        height_px = int(row["height_px"])
        if width_px < thresholds.min_layout_critical_width_px:
            violations.append(f"{key}: layout-critical width below threshold")
        if height_px < thresholds.min_layout_critical_height_px:
            violations.append(f"{key}: layout-critical height below threshold")
        figures.append(
            {
                "key": key,
                "path": row["path"],
                "source": row["source"],
                "caption_has_source_language": row["caption_has_source_language"],
                "width_px": width_px,
                "height_px": height_px,
                "review_reason": LAYOUT_CRITICAL_FIGURE_REASONS[key],
            }
        )
    return {
        "manual_review_surface": CONTACT_SHEET_PATH,
        "figure_count": len(figures),
        "keys": list(LAYOUT_CRITICAL_FIGURE_KEYS),
        "missing_keys": missing_keys,
        "figures": figures,
        "violations": violations,
    }


def _has_source_language(caption: str) -> bool:
    lowered = caption.lower()
    return any(
        phrase in lowered
        for phrase in (
            "from ",
            "synthetic",
            "not empirical",
            "fixture",
            "external",
            "source",
            "artifact",
        )
    )
