# tests/

Pytest suite for Active FractalRabbit. Tests protect package behavior, artifact
contracts, the documentation contract, and the synthetic-only claim boundary.
Agent rules and marker policy are in [`AGENTS.md`](AGENTS.md).

## Running

```bash
uv sync --extra dev
./run.sh test-full
```

`./run.sh test` is the fast arm (no coverage, excludes the
`requires_fractalrabbit` marker); `./run.sh lint`, `./run.sh docs`, and
`./run.sh gnn` are the other arms. See [`../docs/RUN_GUIDE.md`](../docs/RUN_GUIDE.md).

Focused runs are fine while developing, but the full gate above is the readiness
check; do not weaken the coverage threshold or substitute a narrow subset for it.

## What is here

```text
tests/
├── conftest.py                 # shared fixtures and path setup
├── test_*.py                   # unit and contract tests per package module
├── test_docs_contract.py       # documentation contract (fixtures, figure counts,
│                               #   TODO routing, scholarship ledger, claim wording)
├── test_gnn_spec.py            # GNN spec parsing/model-declaration tests
├── test_fractalrabbit_external.py, test_java_resolution.py
│                               # external-lane tests (use the requires_fractalrabbit mark)
└── test_release_manifest.py, test_render_freshness.py, test_manuscript_integrity.py
```

Tests that need the real simulator, JDK 21, or Maven carry the
`requires_fractalrabbit` marker; artifact-surface regeneration tests that run the
full writer set carry the `slow` marker. Fixture-first tests must stay runnable
without the external cache. Keep fixtures small and deterministic, and do not
commit secrets, caches, jars, or generated bulk data.
