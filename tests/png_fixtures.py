"""Deterministic PNG bytes for tests that exercise image-reading audits.

The generated ``output/figures/*.png`` surface is gitignored, so tests that
only need *a* readable image generate one instead of depending on a developer's
generated tree. The image is a plain sine trace: deterministic, non-uniform
(real pixel variance), and cheap to render without pyplot state.
"""

from __future__ import annotations

import io
import math


def synthetic_png_bytes(
    phase: float, *, width_px: int = 1200, height_px: int = 700
) -> bytes:
    """Return deterministic, non-uniform PNG bytes at the requested size."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width_px / 100.0, height_px / 100.0), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    values = [
        math.sin(2.0 * math.pi * (index / 511.0 + phase)) for index in range(512)
    ]
    axis.plot(range(512), values)
    axis.set_title(f"synthetic audit fixture {phase:g}")
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    return buffer.getvalue()
