from __future__ import annotations

import numpy as np
import pytest

from active_fractal_rabbit.privacy_frontier import (
    co_travel_coincidence_rate,
    trace_unicity,
)


def test_trace_unicity_counts_unique_coarsened_signatures() -> None:
    traveller_ids = np.array([0, 0, 1, 1, 2, 2])
    cells = np.array([1, 2, 1, 2, 2, 1])
    times = np.array([0, 1, 0, 1, 0, 1])

    assert trace_unicity(traveller_ids, cells, times) == pytest.approx(2 / 3)


def test_co_travel_coincidence_rate_separates_coupled_trace() -> None:
    traveller_ids = np.array([0, 0, 1, 1, 2, 2])
    cells = np.array([1, 2, 1, 2, 0, 0])
    times = np.array([0, 1, 0, 1, 0, 1])

    assert co_travel_coincidence_rate(traveller_ids, cells, times) == pytest.approx(
        1 / 3
    )
