# Documentation Index

Flat index of the long-lived documentation in this repository. The narrative
hub is [`README.md`](README.md); this page is the file list. Generated files are
marked **generated** and are refreshed by their named script.

## Entry points

| File | Purpose |
| --- | --- |
| [`../README.md`](../README.md) | Project overview, quick start, required local GNN check, repository layout. |
| [`README.md`](README.md) | Documentation hub: orientation, methods, governance, publication, reference. |
| [`RUN_GUIDE.md`](RUN_GUIDE.md) | One gate entry point (`./run.sh`), its arms, and the measured-timing contract. |
| [`../AGENTS.md`](../AGENTS.md) | Repository operating contract for agents. |
| [`AGENTS.md`](AGENTS.md) | Documentation-specific agent rules and claim boundaries. |
| [`../TODO.md`](../TODO.md) | Forward-looking open work only; completed history stays in `tasks.yaml`. |
| [`../tasks.yaml`](../tasks.yaml) | Task board with historical and current status. |
| [`../ISA.md`](../ISA.md) | Repository improvement record and decision log. |

## Orientation

| File | Purpose |
| --- | --- |
| [`quickstart.md`](quickstart.md) | Fixture-first commands, generated artifacts, external lane, and render path. |
| [`architecture.md`](architecture.md) | Seven pipeline layers and dependency boundaries. |
| [`validation.md`](validation.md) | Test coverage, claim gates, and figure/reference QA. |
| [`claim_ledger.md`](claim_ledger.md) | High-risk claim families and their evidence. |
| [`ip_license_status.md`](ip_license_status.md) | IP, license, citation, and publication-readiness status. |

## Methods

| File | Purpose |
| --- | --- |
| [`methods/inference-models.md`](methods/inference-models.md) | Discrete POMDP filter, predictor ladder, and null hierarchy. |
| [`methods/fair-frontier-sota.md`](methods/fair-frontier-sota.md) | Fair (oracle-stripped) comparison and the regime map. |
| [`methods/statistics.md`](methods/statistics.md) | Clustered bootstrap intervals, seed sweeps, and calibration. |
| [`methods/benchmark-protocol.md`](methods/benchmark-protocol.md) | Locked splits, prequential loss, claim gate, and eval-contract guards. |
| [`generative_model_card.md`](generative_model_card.md) | Machine-checkable single-factor and factorial POMDP specification surfaces. |

## Governance and boundary

| File | Purpose |
| --- | --- |
| [`governance/privacy-and-minimization.md`](governance/privacy-and-minimization.md) | Synthetic re-identification and minimization proxies, not formal guarantees. |
| [`empirical_validation_protocol.md`](empirical_validation_protocol.md) | Separate protocol required before any real-waypoint claim. |
| [`scholarship_ledger.md`](scholarship_ledger.md) | Primary-source backing and allowed/forbidden language per citation key. |

## Reference

| File | Purpose |
| --- | --- |
| [`config_schema.md`](config_schema.md) | Every supported config key, validation rule, and artifact group. |
| [`external_lane.md`](external_lane.md) | JDK 21 + Maven toolchain for the pinned external simulator. |
| [`../.github/GNN.md`](../.github/GNN.md) | GNN pin, hosted job key and display name, and what the job parses. |
| [`../models/README.md`](../models/README.md) | GNN spec inventory, loader/drift guard, and upstream pin. |
| [`../scripts/README.md`](../scripts/README.md) | Thin CLI entrypoints and gate scripts. |
| [`../tests/README.md`](../tests/README.md) | Test suite layout, markers, and the full gate. |
| [`../manuscript/README.md`](../manuscript/README.md) | Manuscript rendering contract and cross-reference rules. |
| [`../manuscript/SYNTAX.md`](../manuscript/SYNTAX.md) | Citation, label, and numbering syntax. |

## Generated

| File | Producer | Purpose |
| --- | --- | --- |
| [`_generated/gate-timings.md`](_generated/gate-timings.md) | `./run.sh bench` (`scripts/gate_report.py`) | Per-arm wall time, exit code, and evidence for the current host and commit. |

## Process and history

| File | Purpose |
| --- | --- |
| [`todo/README.md`](todo/README.md) | Scoped TODO briefs for open task-board items. |
| [`revision/REVISION_PROGRAM.md`](revision/REVISION_PROGRAM.md) | Frozen planning record; not current results. |
| [`revision/aif-model-designs.md`](revision/aif-model-designs.md) | AIF model design notes from the revision program. |
| [`revision/pass-A-peer-review.md`](revision/pass-A-peer-review.md) | Peer-review pass record. |
| [`revision/pass-B-restructure-imrad.md`](revision/pass-B-restructure-imrad.md) | IMRaD restructure pass record. |
| [`revision/pass-C-pymdp-push.md`](revision/pass-C-pymdp-push.md) | pymdp push pass record. |

## Manuscript

| File | Purpose |
| --- | --- |
| [`../manuscript/00_abstract.md`](../manuscript/00_abstract.md) | Abstract. |
| [`../manuscript/01_introduction.md`](../manuscript/01_introduction.md) | Introduction. |
| [`../manuscript/02_methodology.md`](../manuscript/02_methodology.md) | Methodology. |
| [`../manuscript/02b_formal_model.md`](../manuscript/02b_formal_model.md) | Formal model (H2 sub-part of methodology). |
| [`../manuscript/03_results.md`](../manuscript/03_results.md) | Results spine and reading order. |
| [`../manuscript/03a_results_partial_observability_and_model_comparison.md`](../manuscript/03a_results_partial_observability_and_model_comparison.md) | Partial-observability and model-comparison results. |
| [`../manuscript/03b_results_null_hierarchy_and_multiplicity.md`](../manuscript/03b_results_null_hierarchy_and_multiplicity.md) | Null-hierarchy and multiplicity results. |
| [`../manuscript/04_discussion.md`](../manuscript/04_discussion.md) | Discussion. |
| [`../manuscript/05_limitations.md`](../manuscript/05_limitations.md) | Limitations. |
| [`../manuscript/07_roadmap.md`](../manuscript/07_roadmap.md) | Forward roadmap with kill-gates. |
| [`../manuscript/08_conclusion.md`](../manuscript/08_conclusion.md) | Conclusion. |
| [`../manuscript/S01_validation_and_threat_model.md`](../manuscript/S01_validation_and_threat_model.md) | Validation and threat model supplement. |
| [`../manuscript/S02_reproducibility.md`](../manuscript/S02_reproducibility.md) | Reproducibility supplement. |
| [`../manuscript/S03_simulator_boundary.md`](../manuscript/S03_simulator_boundary.md) | Simulator boundary supplement. |
| [`../manuscript/S04_formal_diagnostics_and_pymdp.md`](../manuscript/S04_formal_diagnostics_and_pymdp.md) | Formal diagnostics and pymdp supplement. |
| [`../manuscript/S05_closed_loop_extensions.md`](../manuscript/S05_closed_loop_extensions.md) | Closed-loop extensions supplement. |
| [`../manuscript/S06_results_fixture_diagnostics.md`](../manuscript/S06_results_fixture_diagnostics.md) | Fixture diagnostics supplement. |
| [`../manuscript/S07_results_program_extensions.md`](../manuscript/S07_results_program_extensions.md) | Program extensions supplement. |
| [`../manuscript/S08_results_safety_privacy_and_external.md`](../manuscript/S08_results_safety_privacy_and_external.md) | Safety, privacy, and external-lane supplement. |
| [`../manuscript/S09_confound_taxonomy.md`](../manuscript/S09_confound_taxonomy.md) | Confound taxonomy supplement. |
| [`../manuscript/99_references.md`](../manuscript/99_references.md) | Reference section. |
| [`../manuscript/references.bib`](../manuscript/references.bib) | Bibliography keys used by the manuscript. |
| [`../manuscript/config.yaml`](../manuscript/config.yaml) | Paper metadata, publication identifiers, and render formats. |
| [`../manuscript/preamble.md`](../manuscript/preamble.md) | LaTeX preamble for the template render. |
