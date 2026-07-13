# AGENTS.md - configs/

Configuration files define study lanes and external-run boundaries.

Core files:
- `default.yaml` is the fixture-first default lane.
- `fractalrabbit_external.yaml` is the optional external simulator lane.
- `fixtures/` contains config-local fixture references and support files.

Fixture-first lane:
- Default local work should run against checked-in fixtures.
- The small regression fixture is `data/fixtures/fractalrabbit_sample.csv`.
- The default analysis fixture is
  `data/fixtures/fractalrabbit_sample_large.csv`.
- Keep fixture paths repo-relative.
- Do not require FractalRabbit, JDK 21, or Maven for fixture-first checks.

External lane:
- External simulator runs are optional.
- `fractalrabbit_external.yaml` may require JDK 21, Maven, and the pinned
  external FractalRabbit cache.
- Do not make external-lane requirements part of the default local path.
- Keep external outputs clearly separated from fixture-backed outputs.
- Preserve provenance that distinguishes external simulator runs from fixture
  or synthetic outputs.

Secrets and paths:
- No secrets in config files.
- No tokens, credentials, private URLs, or API keys.
- No host-local absolute paths.
- No user-specific cache paths.
- No machine-specific Maven or JDK paths.
- Prefer environment variables or documented CLI arguments for local overrides.

Claim boundary:
- Config changes must not blur the synthetic-only claim boundary.
- Fixture-backed lanes remain synthetic.
- External simulator lanes are still simulator-backed unless real-world data is
  added with explicit provenance.
- Do not encode labels that imply biological, behavioral, clinical,
  ecological, deployed, or universal state-of-the-art validation.

Maintenance:
- Keep schema changes paired with loader and test updates.
- Keep names stable for scripts and manuscript variable generation.
- Update docs only when the actual config contract changes.
- Avoid duplicate knobs that select the same lane in different ways.
- Prefer explicit lane names over ambiguous booleans.
- Keep config defaults deterministic.
