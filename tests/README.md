# tests/

Pytest suite for Active FractalRabbit. Tests protect package behavior, artifact
contracts, the documentation contract, and the synthetic-only claim boundary.
Agent rules and marker policy are in [`AGENTS.md`](AGENTS.md).

## Running

```bash
uv sync --extra dev
uv run pytest tests/ --cov=src --cov-fail-under=90
```

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
`requires_fractalrabbit` marker; fixture-first tests must stay runnable without the
external cache. Keep fixtures small and deterministic, and do not commit secrets,
caches, jars, or generated bulk data.
