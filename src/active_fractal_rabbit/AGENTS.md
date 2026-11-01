# AGENTS.md - src/active_fractal_rabbit/

Package logic lives here.

Ownership:
- Keep study, audit, release, and rendering logic under this package.
- Keep top-level scripts thin; scripts should parse args and call package APIs.
- Do not move package behavior into `scripts/` for convenience.
- Do not vendor FractalRabbit, pymdp, jars, or cache payloads here.

Core modules:
- `manuscript_variables.py` owns manuscript-facing numbers and derived values.
- `model_spec.py` owns the declarative generative-model contract and conformance checks.
- `gnn_spec.py` owns GNN spec parsing and the declared model-structure source of truth.
- `aif_regime_search.py` owns active-inference regime search behavior.
- `sota_program_harness.py` owns benchmark program orchestration.
- `release.py` owns release manifest and readiness assembly.
- `figures.py` owns figure data transforms and figure provenance wiring.
- `claim_audit.py` owns textual claim-boundary checks.
- `visual_claim_audit.py` owns visual claim-boundary checks.

Claim boundary:
- This project is synthetic-only unless real external runs are explicitly added
  with auditable provenance.
- Do not imply biological, behavioral, clinical, ecological, or deployed-world
  validation from fixture or simulator outputs.
- Regime results are local artifacts, not universal state-of-the-art claims.
- Phrase evidence as fixture-backed, simulator-backed, or synthetic as
  appropriate.

Result numbers:
- No hand-authored result numbers in prose, tests, or release text.
- Manuscript numbers must come from `manuscript_variables.py` or generated
  artifacts under `output/runs`.
- If a number changes, update the producer and regenerate the dependent
  artifact instead of patching text.
- Treat stale copied numbers as defects.

Canonical artifacts:
- `output/runs` is the canonical location for study outputs consumed by
  manuscripts, audits, figures, and release manifests.
- Prefer reading named artifacts from `output/runs` over ad hoc local files.
- Preserve provenance fields when transforming artifacts.
- Do not make alternate artifact roots unless a caller explicitly requires one.

Implementation rules:
- Keep functions deterministic for fixture-first tests.
- Keep filesystem writes explicit and caller-controlled.
- Keep CLI defaults aligned with the documented fixture lane.
- Add new audit terms to package audits before relying on them in manuscripts.
- Keep generated-output schemas stable unless tests and consumers change in the
  same patch.
- Respect existing synthetic-only wording in all public-facing outputs.
- Use focused package tests for new package behavior.
