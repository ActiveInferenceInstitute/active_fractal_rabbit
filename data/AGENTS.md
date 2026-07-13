# AGENTS.md - data

Read the project root `AGENTS.md` first.

- Keep checked data small and deterministic.
- `fixtures/fractalrabbit_sample_large.csv` is the default no-Java analysis
  fixture. `fixtures/fractalrabbit_sample.csv` is the small regression and
  parser-contract fixture.
- Do not commit generated simulator outputs here; they belong under `output/runs/`.
- Preserve the FractalRabbit output contract columns: `ID`, `Days`, `x(km)`, and
  `y(km)`.
- Do not add FractalRabbit source, jars, Maven caches, or pymdp source to this
  tree; the only allowed cache location is `.cache/fractalrabbit/`.
- Treat fixture data as synthetic-only evidence.
