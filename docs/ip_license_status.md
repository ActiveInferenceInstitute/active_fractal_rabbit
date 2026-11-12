# IP and License Status: Publication Remains Release-Gated {#sec:ip-license-status}

This document records the repository's current intellectual-property,
licensing, and publication-readiness posture. It is a release-readiness
signpost, not legal advice; public publication should be reviewed by the
project owner or qualified counsel before the private repository is made public,
archived, or redistributed.

## Current Status Is Public Release

| Area | Status | Release interpretation |
| --- | --- | --- |
| Repository visibility | Public GitHub repository under [`ActiveInferenceInstitute/active_fractal_rabbit`](https://github.com/ActiveInferenceInstitute/active_fractal_rabbit); prior private staging history at `docxology/active_fractal_rabbit` is superseded and no longer used | Public release requires the fresh gates below to be green and the release manifest current. |
| Original code | Licensed under MIT in `LICENSE` | Suitable for permissive source redistribution, subject to owner review. |
| Project-authored docs, manuscript, figures, and scholarly outputs | Licensed under CC-BY-4.0 in `LICENSE-DOCS.md` unless a file says otherwise | Attribution required for public reuse. |
| Third-party notices | Recorded in `THIRD_PARTY_NOTICES.md` | Must stay included in any public or archival bundle. |
| Citation metadata | Recorded in `CITATION.cff`, including the minted Zenodo DOI | Concept DOI [`10.5281/zenodo.21330636`](https://doi.org/10.5281/zenodo.21330636), version DOI [`10.5281/zenodo.21330637`](https://doi.org/10.5281/zenodo.21330637) (v0.2.0). |
| Generated release manifest | `output/release_manifest.json` hashes required source and output surfaces | A public bundle is stale unless this manifest verifies after final render. |

## Third-Party Boundaries Preserve Non-Vendoring

FractalRabbit is used as an external pinned simulator boundary, not as vendored
source. The project configuration pins
`https://github.com/NationalSecurityAgency/fractalrabbit.git` at commit
`9933449c4f4fe1b26b6ac7bfdeeac76583085df5`; the optional external lane clones
and builds that source under `.cache/fractalrabbit/` when JDK 21, Maven, and git
are available. The pinned upstream license is Apache-2.0, and the upstream NOTICE
states that portions of the software are public domain under U.S. copyright law.
Any distribution that includes upstream source, jars, or derivative simulator
code must carry the applicable Apache-2.0 license and upstream NOTICE material.

`inferactively-pymdp==1.0.4` is consumed as an installed package dependency
under the MIT License. The project does not vendor `pymdp` source.

Runtime and development dependencies such as `defusedxml`, `matplotlib`,
`numpy`, `pyyaml`, `pytest`, and `pytest-cov` are package dependencies rather
than vendored source. Container, binary, or archival distributions should
include the notices required by the exact installed dependency versions.

## Generated Outputs Stay Synthetic and Project-Owned

Checked fixtures under `data/fixtures/` are synthetic FractalRabbit-format CSV
fixtures maintained by this project. Generated runs, figures, cover art,
manuscript variables, PDFs, and release manifests live under `output/` and are
not tracked in Git. When distributed as a release bundle, those project-generated
scholarly outputs follow `LICENSE-DOCS.md` unless a more specific notice is
attached.

No fixture, simulator output, figure, or model score in this repository is
evidence of real-person mobility, national-security deployment readiness,
surveillance readiness, operational tasking, formal privacy protection, or
global state-of-the-art performance.

## Publication Readiness Gates: Satisfied for v0.2.0

Public release required all of the following to be true; all six were
satisfied for the `v0.2.0` release published 2026-07-12:

1. `LICENSE`, `LICENSE-DOCS.md`, `THIRD_PARTY_NOTICES.md`, `CITATION.cff`, and
   this document are current and included in the release manifest.
2. Scholarship and claim-boundary tests pass against primary-source citations
   and forbidden-overreach language.
3. The full project gate passes:
   `uv run pytest tests/ --cov=src --cov-fail-under=90`. (658 passed, 91.33%
   coverage.)
4. The manuscript renders through the sibling template repository checkout as
   `working/active_fractal_rabbit`.
5. `uv run python scripts/29_finalize_publication_package.py` records and
   verifies `output/render_manifest.json`, then rebuilds and verifies
   `output/release_manifest.json` after the final render. A second run with
   `--verify-only` reported no stale, missing, mismatched, or unhashed paths.
6. The claim audit, visual audit, visual-claim audit, manuscript-integrity
   checks, and synthetic-only publication boundaries all remained green.

The release destination is public GitHub at
[`ActiveInferenceInstitute/active_fractal_rabbit`](https://github.com/ActiveInferenceInstitute/active_fractal_rabbit)
([`v0.2.0` release](https://github.com/ActiveInferenceInstitute/active_fractal_rabbit/releases/tag/v0.2.0)),
cross-linked with a Zenodo DOI:
[`10.5281/zenodo.21330636`](https://doi.org/10.5281/zenodo.21330636) (concept,
resolves to the latest version) and
[`10.5281/zenodo.21330637`](https://doi.org/10.5281/zenodo.21330637) (v0.2.0).
`CITATION.cff`, this document, the manuscript abstract, and the release
manifest carry the destination-specific URL and DOI; update all four together
if the destination or DOI ever changes (e.g. a future `--new-version` Zenodo
deposit for a subsequent release).

## Archival Redundancy Spans Independent Failure Domains

The `v0.2.0` publication is mirrored across independent archival systems
rather than concentrated on a single vendor:

| System | Reference | Verified |
| --- | --- | --- |
| GitHub (source + release) | [`ActiveInferenceInstitute/active_fractal_rabbit`](https://github.com/ActiveInferenceInstitute/active_fractal_rabbit), [`v0.2.0` release](https://github.com/ActiveInferenceInstitute/active_fractal_rabbit/releases/tag/v0.2.0) | Yes — live commit `fd45579` on `main`. |
| Zenodo (files + metadata + DOI) | Concept [`10.5281/zenodo.21330636`](https://doi.org/10.5281/zenodo.21330636), version [`10.5281/zenodo.21330637`](https://doi.org/10.5281/zenodo.21330637) | Yes — record state `done`, DOI-bearing PDF attached. |
| Software Heritage (source snapshot) | [`swh:1:snp:f6238b4081939a8f2e64185804cfff57b4a2851a`](https://archive.softwareheritage.org/swh:1:snp:f6238b4081939a8f2e64185804cfff57b4a2851a/), [browse origin](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/ActiveInferenceInstitute/active_fractal_rabbit) | Yes — `save_task_status: succeeded`, `visit_status: full`. |
| IPFS (content-addressed release bundle) | CID [`QmUX68wWQ7HQ9HmCqGBsE5XvRtM8ixcYw1KcFHdMbW3y2s`](https://gateway.pinata.cloud/ipfs/QmUX68wWQ7HQ9HmCqGBsE5XvRtM8ixcYw1KcFHdMbW3y2s) via Pinata (release bundle: DOI-bearing PDF + manifest + publication metadata) | Yes — pinned, gateway-resolvable. |

This satisfies the independent-systems redundancy recommendation (four
here). No arXiv deposit exists for this release; add one
manually if desired, since no arXiv credential is configured in this
environment.
