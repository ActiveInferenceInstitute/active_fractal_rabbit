from __future__ import annotations

import shutil
import os
from pathlib import Path

import numpy as np

import active_fractal_rabbit.render_freshness as freshness
from active_fractal_rabbit.render_freshness import (
    RENDER_INPUT_PATHS,
    RENDER_OUTPUT_PATHS,
    build_render_freshness,
    publication_input_fingerprint,
    verify_render_freshness,
    write_render_freshness,
)


def _seed_render_tree(root: Path) -> None:
    for relative_path in (*RENDER_INPUT_PATHS, *RENDER_OUTPUT_PATHS):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative_path.encode("utf-8"))


def test_render_freshness_reports_missing_and_stale_outputs(tmp_path) -> None:
    payload = build_render_freshness(tmp_path)

    assert payload["status"] == "fail"
    assert set(payload["missing_inputs"]) == set(RENDER_INPUT_PATHS)
    assert set(payload["missing_outputs"]) == set(RENDER_OUTPUT_PATHS)


def test_render_manifest_detects_changed_input(monkeypatch, tmp_path) -> None:
    _seed_render_tree(tmp_path)
    monkeypatch.setattr(
        freshness,
        "_cover_check",
        lambda root: {"status": "pass", "source": "cover", "pdf": "pdf"},
    )
    monkeypatch.setattr(
        freshness,
        "_publication_fingerprint_check",
        lambda root: {"status": "pass", "expected": "test", "outputs": {}},
    )

    manifest_path = write_render_freshness(tmp_path)
    (tmp_path / RENDER_INPUT_PATHS[0]).write_text("changed\n", encoding="utf-8")

    report = verify_render_freshness(manifest_path)

    assert report["missing"] == []
    assert RENDER_INPUT_PATHS[0] in report["mismatched"]
    assert any("older than current inputs" in item for item in report["failed_checks"])


def test_rebuilding_manifest_cannot_certify_stale_render(monkeypatch, tmp_path) -> None:
    _seed_render_tree(tmp_path)
    monkeypatch.setattr(
        freshness,
        "_cover_check",
        lambda root: {"status": "pass", "source": "cover", "pdf": "pdf"},
    )
    monkeypatch.setattr(
        freshness,
        "_publication_fingerprint_check",
        lambda root: {"status": "pass", "expected": "test", "outputs": {}},
    )
    for relative_path in RENDER_OUTPUT_PATHS:
        os.utime(tmp_path / relative_path, ns=(1_000, 1_000))

    manifest_path = write_render_freshness(tmp_path)
    payload = build_render_freshness(tmp_path)
    report = verify_render_freshness(manifest_path)

    assert payload["status"] == "fail"
    assert set(payload["stale_outputs"]) == set(RENDER_OUTPUT_PATHS)
    assert any("older than current inputs" in item for item in report["failed_checks"])


def test_cover_check_rejects_stale_embedded_cover(monkeypatch, tmp_path) -> None:
    cover_path = tmp_path / "cover.png"
    pdf_path = tmp_path / "paper.pdf"
    source_cover = Path(__file__).resolve().parents[1] / "output/cover/active_fractal_rabbit_cover.png"
    shutil.copy2(source_cover, cover_path)
    pdf_path.write_bytes(b"not-used-by-mocked-extractor")
    monkeypatch.setattr(
        freshness,
        "_extract_embedded_cover",
        lambda path: (np.zeros((3, 3, 3), dtype=np.uint8), None),
    )

    monkeypatch.setattr(freshness, "RENDER_INPUT_PATHS", ("cover.png",))
    monkeypatch.setattr(freshness, "RENDER_OUTPUT_PATHS", ("paper.pdf",))
    report = freshness._cover_check(tmp_path)

    assert report["status"] == "fail"
    assert "differ" in report["reason"]


def test_touching_outputs_cannot_certify_unrendered_manuscript_change(
    monkeypatch, tmp_path: Path
) -> None:
    _seed_render_tree(tmp_path)
    manuscript = tmp_path / "manuscript" / "00_abstract.md"
    manuscript.parent.mkdir(parents=True, exist_ok=True)
    manuscript.write_text("original manuscript\n", encoding="utf-8")
    resolved = tmp_path / "output" / "manuscript" / "S02_reproducibility.md"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text("resolved manuscript\n", encoding="utf-8")
    initial_fingerprint = publication_input_fingerprint(tmp_path)
    monkeypatch.setattr(
        freshness,
        "_cover_check",
        lambda root: {"status": "pass", "source": "cover", "pdf": "pdf"},
    )
    monkeypatch.setattr(
        freshness,
        "_rendered_output_text",
        lambda path: (f"{freshness.RENDER_FINGERPRINT_PREFIX} {initial_fingerprint}", None),
    )
    assert freshness._publication_fingerprint_check(tmp_path)["status"] == "pass"

    manuscript.write_text("changed but never rendered\n", encoding="utf-8")
    for relative_path in RENDER_OUTPUT_PATHS:
        os.utime(tmp_path / relative_path, None)

    payload = build_render_freshness(tmp_path)

    assert payload["stale_outputs"] == []
    assert payload["publication_fingerprint"]["status"] == "fail"
    assert payload["status"] == "fail"
