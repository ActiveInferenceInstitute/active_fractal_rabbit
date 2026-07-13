"""Verify that rendered publication outputs match current source artifacts.

The release manifest proves that files exist and are byte-stable.  This module
adds the missing publication-level contract: the PDF and web render must be
produced from the current hydrated manuscript, figure registry, variables, and
cover image.  The first-page cover is checked by extracting the embedded PDF
image and comparing its RGB pixels with the canonical cover artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np


RENDER_FRESHNESS_SCHEMA = "active-fractal-rabbit-render-freshness-v3"
RENDER_FINGERPRINT_PREFIX = "Render input fingerprint:"
RENDER_FINGERPRINT_PATTERN = re.compile(
    rf"^{re.escape(RENDER_FINGERPRINT_PREFIX)}\s*`?([0-9a-f]{{32}})`?\.?\s*$",
    re.MULTILINE,
)
RENDER_INPUT_PATHS: tuple[str, ...] = (
    "manuscript/config.yaml",
    "output/data/manuscript_variables.json",
    "output/data/manuscript_variable_provenance.json",
    "output/figures/figure_registry.json",
    "output/figures/style_provenance.json",
    "manuscript/preamble.md",
    "output/cover/active_fractal_rabbit_cover.png",
)
RENDER_OUTPUT_PATHS: tuple[str, ...] = (
    "output/pdf/active_fractal_rabbit_combined.pdf",
    "output/web/index.html",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        return {"path": relative_path, "exists": False}
    stat = path.stat()
    return {
        "path": relative_path,
        "exists": True,
        "bytes": stat.st_size,
        "sha256": _sha256(path),
        "mtime_ns": stat.st_mtime_ns,
    }


def _render_input_paths(root: Path) -> tuple[str, ...]:
    """Return static contracts plus every source and resolved manuscript file."""

    paths = set(RENDER_INPUT_PATHS)
    paths.update(
        str(path.relative_to(root))
        for pattern in (
            "manuscript/*.md",
            "manuscript/*.bib",
            "output/manuscript/*",
            "output/figures/*",
        )
        for path in root.glob(pattern)
        if path.is_file()
    )
    return tuple(sorted(paths))


def _publication_fingerprint_paths(root: Path) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for pattern in (
        "manuscript/*.md",
        "manuscript/*.bib",
        "manuscript/*.yaml",
        "output/manuscript/*",
        "output/figures/*",
        "output/data/manuscript_variables.json",
        "output/data/manuscript_variable_provenance.json",
        "output/cover/active_fractal_rabbit_cover.png",
    ):
        paths.update(path for path in root.glob(pattern) if path.is_file())
    return tuple(sorted(paths))


def publication_input_fingerprint(project_root: str | Path) -> str:
    """Hash current publication inputs, ignoring the embedded receipt itself."""

    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    for path in _publication_fingerprint_paths(root):
        relative = str(path.relative_to(root))
        content = path.read_bytes()
        if path.suffix.lower() in {".md", ".yaml", ".json", ".bib"}:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                pass
            else:
                text = RENDER_FINGERPRINT_PATTERN.sub("", text).rstrip() + "\n"
                content = text.encode("utf-8")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()[:32]


def write_resolved_render_fingerprint(
    project_root: str | Path, resolved_manuscript_root: str | Path
) -> str:
    """Append a content-addressed render receipt to the resolved S2 manuscript."""

    root = Path(project_root).resolve()
    resolved = Path(resolved_manuscript_root).resolve()
    target = resolved / "S02_reproducibility.md"
    if not target.is_file():
        raise FileNotFoundError(f"missing resolved reproducibility section: {target}")
    text = target.read_text(encoding="utf-8")
    text = RENDER_FINGERPRINT_PATTERN.sub("", text).rstrip() + "\n"
    target.write_text(text, encoding="utf-8")
    fingerprint = publication_input_fingerprint(root)
    target.write_text(
        text + f"\n{RENDER_FINGERPRINT_PREFIX} `{fingerprint}`.\n",
        encoding="utf-8",
    )
    return fingerprint


def _rendered_output_text(path: Path) -> tuple[str | None, str | None]:
    if path.suffix.lower() == ".html":
        try:
            return path.read_text(encoding="utf-8"), None
        except OSError as exc:
            return None, str(exc)
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        return None, "pdftotext executable is unavailable"
    completed = subprocess.run(
        [pdftotext, str(path), "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return None, f"pdftotext failed: {detail or completed.returncode}"
    return completed.stdout, None


def _publication_fingerprint_check(root: Path) -> dict[str, Any]:
    expected = publication_input_fingerprint(root)
    outputs: dict[str, dict[str, Any]] = {}
    for relative_path in RENDER_OUTPUT_PATHS:
        path = root / relative_path
        row: dict[str, Any] = {"contains_expected": False}
        if not path.is_file():
            row["reason"] = "render output is missing"
        else:
            text, error = _rendered_output_text(path)
            row["contains_expected"] = text is not None and expected in text
            if error:
                row["reason"] = error
            elif not row["contains_expected"]:
                row["reason"] = "embedded render fingerprint is missing or stale"
        outputs[relative_path] = row
    status = "pass" if all(row["contains_expected"] for row in outputs.values()) else "fail"
    return {"status": status, "expected": expected, "outputs": outputs}


def _rgb_pixels(path: Path) -> np.ndarray:
    image = np.asarray(mpimg.imread(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0)
    return np.asarray(image, dtype=np.uint8)


def _pixel_digest(pixels: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(pixels.shape).encode("ascii"))
    digest.update(pixels.tobytes())
    return digest.hexdigest()


def _extract_embedded_cover(pdf_path: Path) -> tuple[np.ndarray | None, str | None]:
    pdfimages = shutil.which("pdfimages")
    if pdfimages is None:
        return None, "pdfimages executable is unavailable"
    with tempfile.TemporaryDirectory(prefix="afr-render-cover-") as temp_dir:
        prefix = Path(temp_dir) / "cover"
        completed = subprocess.run(
            [
                pdfimages,
                "-png",
                "-f",
                "1",
                "-l",
                "1",
                str(pdf_path),
                str(prefix),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            return None, f"pdfimages failed: {detail or completed.returncode}"
        candidates = sorted(Path(temp_dir).glob("cover-*.png"))
        if not candidates:
            return None, "PDF first page contains no extractable image"
        try:
            return _rgb_pixels(candidates[0]), None
        except Exception as exc:  # pragma: no cover - corrupt PDF/image path
            return None, f"embedded cover is unreadable: {exc}"


def _cover_check(root: Path) -> dict[str, Any]:
    cover_path = root / RENDER_INPUT_PATHS[-1]
    pdf_path = root / RENDER_OUTPUT_PATHS[0]
    result: dict[str, Any] = {
        "status": "fail",
        "source": RENDER_INPUT_PATHS[-1],
        "pdf": RENDER_OUTPUT_PATHS[0],
    }
    if not cover_path.is_file():
        result["reason"] = "canonical cover is missing"
        return result
    if not pdf_path.is_file():
        result["reason"] = "rendered PDF is missing"
        return result
    try:
        source = _rgb_pixels(cover_path)
        embedded, error = _extract_embedded_cover(pdf_path)
    except Exception as exc:  # pragma: no cover - corrupt source image
        result["reason"] = f"canonical cover is unreadable: {exc}"
        return result
    if embedded is None:
        result["reason"] = error or "embedded cover extraction failed"
        return result
    result["source_shape"] = list(source.shape)
    result["embedded_shape"] = list(embedded.shape)
    result["source_pixel_digest"] = _pixel_digest(source)
    result["embedded_pixel_digest"] = _pixel_digest(embedded)
    if source.shape != embedded.shape:
        result["reason"] = "embedded cover dimensions differ from canonical cover"
        return result
    if not np.array_equal(source, embedded):
        result["reason"] = "embedded PDF cover differs from canonical cover pixels"
        return result
    result["status"] = "pass"
    return result


def build_render_freshness(project_root: str | Path) -> dict[str, Any]:
    """Build a deterministic render-freshness report without writing it."""

    root = Path(project_root).resolve()
    input_paths = _render_input_paths(root)
    inputs = {path: _file_record(root, path) for path in input_paths}
    outputs = {
        path: _file_record(root, path) for path in RENDER_OUTPUT_PATHS
    }
    missing_inputs = [path for path, record in inputs.items() if not record["exists"]]
    missing_outputs = [path for path, record in outputs.items() if not record["exists"]]
    existing_input_mtimes = [
        int(record["mtime_ns"]) for record in inputs.values() if record["exists"]
    ]
    newest_input_mtime_ns = max(existing_input_mtimes, default=None)
    stale_outputs = [
        path
        for path, record in outputs.items()
        if record["exists"]
        and newest_input_mtime_ns is not None
        and int(record["mtime_ns"]) < newest_input_mtime_ns
    ]
    cover = _cover_check(root)
    fingerprint = _publication_fingerprint_check(root)
    status = (
        "pass"
        if not missing_inputs
        and not missing_outputs
        and not stale_outputs
        and cover["status"] == "pass"
        and fingerprint["status"] == "pass"
        else "fail"
    )
    return {
        "schema": RENDER_FRESHNESS_SCHEMA,
        "status": status,
        "project": "active_fractal_rabbit",
        "inputs": inputs,
        "outputs": outputs,
        "missing_inputs": missing_inputs,
        "missing_outputs": missing_outputs,
        "newest_input_mtime_ns": newest_input_mtime_ns,
        "stale_outputs": stale_outputs,
        "cover_embedding": cover,
        "publication_fingerprint": fingerprint,
    }


def write_render_freshness(project_root: str | Path) -> Path:
    """Write ``output/render_manifest.json`` and return its path."""

    root = Path(project_root).resolve()
    output = root / "output" / "render_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_render_freshness(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def verify_render_freshness(manifest_path: str | Path) -> dict[str, list[str]]:
    """Recompute the render proof and compare it with the recorded manifest."""

    path = Path(manifest_path)
    if not path.is_file():
        return {"missing": [str(path)], "mismatched": [], "failed_checks": []}
    root = path.resolve().parents[1]
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"missing": [], "mismatched": [str(path)], "failed_checks": [str(exc)]}
    current = build_render_freshness(root)
    mismatched: list[str] = []
    if recorded.get("schema") != current["schema"]:
        mismatched.append("schema")
    if recorded.get("status") != current["status"]:
        mismatched.append("status")
    if recorded.get("publication_fingerprint") != current["publication_fingerprint"]:
        mismatched.append("publication_fingerprint")
    for section in ("inputs", "outputs"):
        for relative_path, record in current[section].items():
            if record != recorded.get(section, {}).get(relative_path):
                mismatched.append(relative_path)
    failed_checks: list[str] = []
    if current["status"] != "pass":
        if current["stale_outputs"]:
            failed_checks.append(
                "render outputs older than current inputs: "
                + ", ".join(current["stale_outputs"])
            )
        if current["cover_embedding"].get("status") != "pass":
            failed_checks.append(
                current["cover_embedding"].get("reason", "cover freshness failed")
            )
        if current["publication_fingerprint"].get("status") != "pass":
            for output_path, row in current["publication_fingerprint"]["outputs"].items():
                if not row.get("contains_expected"):
                    failed_checks.append(
                        f"{output_path}: {row.get('reason', 'render fingerprint failed')}"
                    )
        if not failed_checks:
            failed_checks.append("render freshness failed")
    return {"missing": [], "mismatched": sorted(set(mismatched)), "failed_checks": failed_checks}
