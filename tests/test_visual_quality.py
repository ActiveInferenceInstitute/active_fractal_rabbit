from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
from png_fixtures import synthetic_png_bytes

from active_fractal_rabbit.figure_style import font_contract_payload
from active_fractal_rabbit.visual_quality import (
    LAYOUT_CRITICAL_FIGURE_KEYS,
    VISUAL_QUALITY_SCHEMA,
    VisualQualityError,
    audit_visual_quality,
    write_visual_quality_audit,
)


def test_visual_quality_audit_passes_for_generated_figures() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = root / "output" / "figures" / "figure_registry.json"
    if not registry.is_file():
        pytest.skip(
            f"generated figure surface not present: missing {registry} "
            "(removal condition: run scripts/07_run_full_study.py --lane "
            "fixture and scripts/04_generate_figures.py --run "
            "output/runs/default; owner: fixture lane)"
        )
    payload = audit_visual_quality(root)

    assert payload["schema"] == VISUAL_QUALITY_SCHEMA
    assert payload["status"] == "pass"
    assert payload["figure_count"] == 53
    assert payload["caption_source_language_count"] == 53
    assert payload["font_contract"]["base"] >= 17.0
    assert payload["font_contract"]["tick"] >= 16.0
    assert payload["font_contract"]["legend"] >= 15.5
    assert payload["font_contract"]["minimum_declared_readable"] >= 15.0
    assert payload["min_width_px"] >= 1100
    assert payload["min_height_px"] >= 650
    assert payload["min_aspect_ratio"] > 0.0
    assert payload["max_aspect_ratio"] > payload["min_aspect_ratio"]
    assert (
        float(payload["min_pixel_stddev"]) >= payload["thresholds"]["min_pixel_stddev"]
    )
    assert (
        payload["quality_summary"]["manual_review_surface"]
        == "output/figures/figure_contact_sheet.png"
    )
    layout_critical = payload["layout_critical_figures"]
    assert (
        layout_critical["manual_review_surface"]
        == "output/figures/figure_contact_sheet.png"
    )
    assert tuple(layout_critical["keys"]) == LAYOUT_CRITICAL_FIGURE_KEYS
    assert layout_critical["missing_keys"] == []
    assert layout_critical["figure_count"] == len(LAYOUT_CRITICAL_FIGURE_KEYS)
    layout_by_key = {row["key"]: row for row in layout_critical["figures"]}
    for key in LAYOUT_CRITICAL_FIGURE_KEYS:
        row = layout_by_key[key]
        assert row["path"].startswith("output/figures/")
        assert row["source"].startswith("output/")
        assert row["caption_has_source_language"] is True
        assert row["width_px"] >= payload["thresholds"]["min_layout_critical_width_px"]
        assert (
            row["height_px"] >= payload["thresholds"]["min_layout_critical_height_px"]
        )
        assert row["review_reason"]
    contact_sheet = payload["quality_summary"]["manual_review_contact_sheet"]
    assert contact_sheet["exists"] is True
    assert contact_sheet["image_readable"] is True
    assert contact_sheet["current"] is True
    assert contact_sheet["width_px"] >= payload["thresholds"]["min_width_px"]
    assert contact_sheet["height_px"] >= payload["thresholds"]["min_height_px"]
    assert (
        float(contact_sheet["pixel_stddev"])
        >= payload["thresholds"]["min_pixel_stddev"]
    )
    assert (
        payload["quality_summary"]["shortest_figure"]["height_px"]
        == payload["min_height_px"]
    )
    assert (
        payload["quality_summary"]["narrowest_figure"]["width_px"]
        == payload["min_width_px"]
    )
    assert (
        payload["quality_summary"]["lowest_pixel_stddev_figure"]["pixel_stddev"]
        == payload["min_pixel_stddev"]
    )
    assert "aspect_ratio" in payload["figures"][0]
    assert payload["violations"] == []


def test_contact_sheet_is_review_only_not_manuscript_registered() -> None:
    root = Path(__file__).resolve().parents[1]
    figure_dir = root / "output" / "figures"
    required = (
        figure_dir / "figure_registry.json",
        figure_dir / "figure_manifest.json",
        root / "output" / "release_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        pytest.skip(
            "generated figure/release surface not present; missing "
            + ", ".join(missing)
            + " (removal condition: run the full study and "
            "scripts/29_finalize_publication_package.py; owner: release lane)"
        )
    registry = json.loads(
        (figure_dir / "figure_registry.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (figure_dir / "figure_manifest.json").read_text(encoding="utf-8")
    )
    release_manifest = json.loads(
        (root / "output" / "release_manifest.json").read_text(encoding="utf-8")
    )

    registered_keys = {entry["key"] for entry in registry["entries"]}
    registered_filenames = {entry["filename"] for entry in registry["entries"]}
    release_paths = {entry["path"] for entry in release_manifest["entries"]}

    assert manifest["contact_sheet"].endswith("output/figures/figure_contact_sheet.png")
    assert "output/figures/figure_contact_sheet.png" in release_paths
    assert "contact_sheet" not in registered_keys
    assert "figure_contact_sheet.png" not in registered_filenames


def test_visual_quality_audit_detects_missing_data_source(tmp_path) -> None:
    figure_dir = tmp_path / "output" / "figures"
    figure_dir.mkdir(parents=True)
    source_dir = tmp_path / "output" / "runs"
    source_dir.mkdir(parents=True)
    (source_dir / "source.json").write_text("{}\n", encoding="utf-8")
    png = figure_dir / "figure.png"
    png.write_bytes(synthetic_png_bytes(0.0))
    registry = {
        "schema": "active-fractal-rabbit-figure-registry-v1",
        "entries": [
            {
                "key": "ok",
                "filename": "figure.png",
                "caption": "A source-bound synthetic figure from output/runs/missing.json with enough text.",
                "alt_text": "A readable source-bound visual quality fixture.",
                "source": "output/runs/missing.json",
                "data_derived": True,
            }
        ],
    }
    (figure_dir / "figure_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )
    (figure_dir / "style_provenance.json").write_text(
        json.dumps(
            {
                "schema": "active-fractal-rabbit-style-provenance-v1",
                "dpi": 180,
                "font_contract": font_contract_payload(),
            }
        ),
        encoding="utf-8",
    )

    output = write_visual_quality_audit(tmp_path, fail_on_violations=False)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["status"] == "fail"
    assert any(
        "data-derived source artifact is missing" in item
        for item in payload["violations"]
    )
    with pytest.raises(
        VisualQualityError, match="data-derived source artifact is missing"
    ):
        write_visual_quality_audit(tmp_path)


def test_visual_quality_audit_detects_unregistered_png(tmp_path) -> None:
    figure_dir = tmp_path / "output" / "figures"
    figure_dir.mkdir(parents=True)
    source_dir = tmp_path / "output" / "runs"
    source_dir.mkdir(parents=True)
    (source_dir / "source.json").write_text("{}\n", encoding="utf-8")
    source_png = synthetic_png_bytes(0.0)
    (figure_dir / "figure.png").write_bytes(source_png)
    (figure_dir / "review_crop.png").write_bytes(source_png)
    registry = {
        "schema": "active-fractal-rabbit-figure-registry-v1",
        "entries": [
            {
                "key": "ok",
                "filename": "figure.png",
                "caption": "A source-bound synthetic figure from output/runs/source.json with enough text.",
                "alt_text": "A readable source-bound visual quality fixture.",
                "source": "output/runs/source.json",
                "data_derived": True,
            }
        ],
    }
    (figure_dir / "figure_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )
    (figure_dir / "style_provenance.json").write_text(
        json.dumps(
            {
                "schema": "active-fractal-rabbit-style-provenance-v1",
                "dpi": 180,
                "font_contract": font_contract_payload(),
            }
        ),
        encoding="utf-8",
    )

    payload = audit_visual_quality(tmp_path)

    assert payload["status"] == "fail"
    assert payload["unexpected_pngs"] == ["review_crop.png"]
    assert "unregistered figure image" in " ".join(payload["violations"])


def test_visual_quality_audit_detects_missing_style_provenance(tmp_path) -> None:
    figure_dir = tmp_path / "output" / "figures"
    figure_dir.mkdir(parents=True)
    source_dir = tmp_path / "output" / "runs"
    source_dir.mkdir(parents=True)
    (source_dir / "source.json").write_text("{}\n", encoding="utf-8")
    png = figure_dir / "figure.png"
    png.write_bytes(synthetic_png_bytes(0.0))
    registry = {
        "schema": "active-fractal-rabbit-figure-registry-v1",
        "entries": [
            {
                "key": "ok",
                "filename": "figure.png",
                "caption": "A source-bound synthetic figure from output/runs/source.json with enough boundary text.",
                "alt_text": "A readable source-bound visual quality fixture.",
                "source": "output/runs/source.json",
                "data_derived": True,
            }
        ],
    }
    (figure_dir / "figure_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )

    output = write_visual_quality_audit(tmp_path, fail_on_violations=False)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["status"] == "fail"
    assert any("missing style provenance" in item for item in payload["violations"])


def test_visual_quality_audit_detects_missing_contact_sheet(tmp_path) -> None:
    figure_dir = tmp_path / "output" / "figures"
    figure_dir.mkdir(parents=True)
    source_dir = tmp_path / "output" / "runs"
    source_dir.mkdir(parents=True)
    (source_dir / "source.json").write_text("{}\n", encoding="utf-8")
    png = figure_dir / "figure.png"
    png.write_bytes(synthetic_png_bytes(0.0))
    registry = {
        "schema": "active-fractal-rabbit-figure-registry-v1",
        "entries": [
            {
                "key": "ok",
                "filename": "figure.png",
                "caption": "A source-bound synthetic figure from output/runs/source.json with enough boundary text.",
                "alt_text": "A readable source-bound visual quality fixture.",
                "source": "output/runs/source.json",
                "data_derived": True,
            }
        ],
    }
    (figure_dir / "figure_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )
    (figure_dir / "style_provenance.json").write_text(
        json.dumps(
            {
                "schema": "active-fractal-rabbit-style-provenance-v1",
                "dpi": 180,
                "font_contract": font_contract_payload(),
            }
        ),
        encoding="utf-8",
    )

    output = write_visual_quality_audit(tmp_path, fail_on_violations=False)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["status"] == "fail"
    assert any(
        "missing manual review contact sheet" in item for item in payload["violations"]
    )


def test_visual_quality_rejects_swapped_registered_figure(tmp_path: Path) -> None:
    figure_dir = tmp_path / "output" / "figures"
    source_dir = tmp_path / "output" / "runs"
    figure_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    source = source_dir / "source.json"
    source.write_text('{"value": 1}\n', encoding="utf-8")
    figure = figure_dir / "figure.png"
    figure.write_bytes(synthetic_png_bytes(0.0))
    caption = (
        "Source source.json provides a sufficiently long synthetic figure caption with "
        "an explicit boundary. Supports a source-bound diagnostic; does not support an "
        "empirical or real-mobility claim."
    )
    image_sha = hashlib.sha256(figure.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    binding = hashlib.sha256()
    for value in ("ok", caption, image_sha, source_sha):
        binding.update(value.encode("utf-8"))
    registry = {
        "schema": "active-fractal-rabbit-figure-registry-v2",
        "entries": [
            {
                "key": "ok",
                "filename": "figure.png",
                "caption": caption,
                "alt_text": caption,
                "source": "output/runs/source.json",
                "data_derived": True,
                "image_sha256": image_sha,
                "source_sha256": source_sha,
                "binding_sha256": binding.hexdigest(),
            }
        ],
    }
    (figure_dir / "figure_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )
    (figure_dir / "style_provenance.json").write_text(
        json.dumps(
            {
                "schema": "active-fractal-rabbit-style-provenance-v1",
                "dpi": 220,
                "font_contract": font_contract_payload(),
            }
        ),
        encoding="utf-8",
    )
    contact = figure_dir / "figure_contact_sheet.png"
    contact.write_bytes(figure.read_bytes())

    figure.write_bytes(synthetic_png_bytes(0.5))
    contact.write_bytes(figure.read_bytes())
    payload = audit_visual_quality(tmp_path)

    assert payload["status"] == "fail"
    assert any("image checksum differs" in item for item in payload["violations"])


def test_figure_sources_do_not_reintroduce_tiny_font_literals() -> None:
    root = Path(__file__).resolve().parents[1]
    source_paths = [
        root / "src" / "active_fractal_rabbit" / "figures.py",
        root / "src" / "active_fractal_rabbit" / "efe_location_frontier.py",
        root / "src" / "active_fractal_rabbit" / "linkage_risk_frontier.py",
        root / "src" / "active_fractal_rabbit" / "poisoning_robustness.py",
        root / "src" / "active_fractal_rabbit" / "sota_program_figures.py",
    ]
    violations: list[str] = []
    for source_path in source_paths:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != "fontsize":
                    continue
                value = keyword.value
                if isinstance(value, ast.Constant) and isinstance(
                    value.value, (int, float)
                ):
                    if float(value.value) < 15.0:
                        violations.append(
                            f"{source_path.relative_to(root)}:{value.lineno} fontsize={value.value}"
                        )

    assert violations == []

    figure_source = source_paths[0].read_text(encoding="utf-8")
    partial_section = figure_source.split("def _partial_observability", 1)[1].split(
        "def _sequence_baselines", 1
    )[0]
    assert 'set_ylabel("soft-belief advantage' in partial_section
    assert 'set_ylabel("AIF advantage' not in partial_section
