# AGENTS.md - scripts/

Scripts are thin CLI wrappers.

Role:
- Parse command-line arguments.
- Resolve repo-relative paths.
- Call `active_fractal_rabbit` package APIs.
- Exit with clear status and concise diagnostics.
- Do not hold business logic here.

Important entry points:
- `07_run_full_study.py` runs the full study lanes and writes canonical outputs.
- `20_run_aif_regime_search.py` runs the active-inference regime search.
- `z_generate_manuscript_variables.py` regenerates manuscript-facing variables.
  `gnn_importer_variables.py` supplies the `gnn_*` keys from the shared reader;
  set `GNN_RUNTIME_PYTHON` to the pinned GNN interpreter when running the
  producer so those keys are included.
- `06_build_release_manifest.py` builds release manifest evidence.
- `check_docs_links.py`, `check_gnn_conformance.py`, and `gate_report.py` are
  the gate arms behind `./run.sh docs`, `./run.sh gnn`, and `./run.sh bench`.
  `check_gnn_conformance.py` is a launch shim: it materializes the pinned
  parser and calls the shared `python -m gnn.importers` reader; it does not read
  the pin itself beyond the revision needed to materialize the environment.
  Keep the commands in `run.sh`; do not add a second copy to CI or docs.

Artifact discipline:
- Write study, regime, release, and manuscript-derived artifacts under
  `output/runs` unless the caller passes an explicit output path.
- Do not introduce competing default output roots.
- Preserve metadata needed by claim audits, visual audits, release manifests,
  and manuscript variables.
- Do not hand-edit generated artifacts after a script run.

Parallelism:
- Do not run competing output-mutating jobs in parallel.
- Treat `output/runs` as shared mutable state.
- Run long output-producing jobs serially when they target the same artifact
  names, run IDs, or manifest files.
- Prefer dry-run, validation, or read-only commands for concurrent checks.

Logic placement:
- If a script needs branching beyond argument handling, move that logic into
  `src/active_fractal_rabbit/`.
- Keep reusable data transforms out of scripts.
- Keep fixture loading, audit rules, release assembly, and figure provenance in
  package modules.
- Keep scripts importable without side effects.

Claim boundary:
- Script output must preserve the synthetic-only claim boundary.
- Do not print state-of-the-art, real-world, or external validation claims from
  local fixture or simulator runs.
- Label fixture-backed results as fixture-backed or synthetic.

Operational notes:
- Prefer repo-relative paths in examples and defaults.
- Do not embed secrets, host-local absolute paths, or machine-specific cache
  paths.
- Keep defaults aligned with the fixture-first lane.
- On failures, report the artifact path that was being produced or validated.
- Add script coverage only where the wrapper behavior itself matters.
