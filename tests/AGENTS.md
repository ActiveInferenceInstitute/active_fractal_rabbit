# AGENTS.md - tests/

Tests protect package behavior, artifact contracts, and claim boundaries.

Full gate:
- Run `uv run pytest tests/ --cov=src --cov-fail-under=90` before claiming the
  project is green.
- Do not weaken coverage thresholds to pass a local patch.
- Do not replace the full gate with a narrow subset in final readiness claims.

Focused checks:
- Use focused docs subset checks when editing only AGENTS or documentation
  routing files.
- For package changes, run the nearest focused test first, then the full gate
  when behavior or artifact contracts changed.
- For script wrapper changes, test the wrapper and the package API it calls.
- For manuscript-number changes, test the producer, not just the rendered text.

Artifact protection:
- Protect canonical `output/runs` artifacts from accidental overwrite.
- Tests that write output should use temporary directories unless the scenario
  explicitly verifies canonical artifact handling.
- Do not make tests depend on stale local artifacts without checking freshness
  or provenance.
- Preserve run IDs, manifest paths, and audit evidence expected by consumers.

External simulator lane:
- Tests requiring real FractalRabbit, JDK 21, Maven, or external simulator
  execution must use the `requires_fractalrabbit` marker.
- Keep fixture-first tests runnable without the external simulator cache.
- Do not silently skip external-lane failures unless the marker policy says the
  dependency is unavailable.

Claim boundary:
- Tests must enforce the synthetic-only claim boundary.
- Local fixtures and simulator outputs do not prove real-world validation.
- Audit tests should catch unsupported biological, behavioral, clinical,
  ecological, deployment, or state-of-the-art language.
- When wording changes, update claim-audit fixtures deliberately.

Skip and xfail policy:
- Do not add xfail or skip markers to weaken existing failures.
- Temporary skips need a narrow reason, owner, and removal condition.
- Prefer fixing deterministic fixture behavior over quarantining it.
- Do not mark claim-boundary failures as expected failures.

Test data:
- Keep fixtures small and deterministic.
- Use the documented small regression fixture for contract tests.
- Use the large fixture only when the scenario needs the default analysis lane.
- Do not commit secrets, host-local paths, caches, jars, or generated bulk data.
