# Data Fixtures and Generated-Output Boundary

`fixtures/fractalrabbit_sample.csv` is a small checked CSV using FractalRabbit's
published output contract: `ID,Days,x(km),y(km)`.

Large generated simulator outputs belong under `output/runs/` and are ignored by
git by default.

## Checked Fixture CSVs

- `fixtures/fractalrabbit_sample.csv` — the original 16-row contract fixture
  (4 travellers x 4 reports over 3 days). Kept as the deterministic regression
  substrate; several unit tests pin its exact counts.
- `fixtures/fractalrabbit_sample_large.csv` — the expanded deterministic analysis fixture
  (4 travellers x 240 timepoints over 18 days) driving `configs/default.yaml`
  and the manuscript. Regenerable deterministically (seed 2026) with
  `uv run python scripts/z_generate_large_fixture.py`; covers all four speed
  categories and both burst classes under the current 4x4 default analysis
  surface without treating cell occupancy as a fixed prose claim.
