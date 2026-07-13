"""Check generated manuscript figures for visual and caption integrity."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from . import _bootstrap  # noqa: F401
except ImportError:  # pragma: no cover - only hit when run as a standalone script
    import _bootstrap  # noqa: F401

from active_fractal_rabbit.visual_quality import (
    VisualQualityError,
    write_visual_quality_audit,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        output = write_visual_quality_audit(ROOT)
    except VisualQualityError as exc:
        print(f"visual quality: FAIL — {exc}")
        return 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    print(
        "visual quality: {status} — {count} figure(s), min {width}x{height}px, min pixel stddev {stddev:.3f}".format(
            status=str(payload["status"]).upper(),
            count=payload["figure_count"],
            width=payload["min_width_px"],
            height=payload["min_height_px"],
            stddev=float(payload["min_pixel_stddev"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
