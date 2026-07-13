# Third-Party Notices

Active FractalRabbit does not vendor the FractalRabbit source tree,
FractalRabbit jars, Maven caches, `pymdp` source, or bulk generated `output/`
artifacts in Git. This notice summarizes third-party materials referenced,
fetched, or depended on by the project.

This file is an engineering notice, not legal advice. Public release should be
reviewed by the project owner or qualified counsel.

## Pinned FractalRabbit Simulator

- Upstream repository: https://github.com/NationalSecurityAgency/fractalrabbit.git
- Pinned commit: `9933449c4f4fe1b26b6ac7bfdeeac76583085df5`
- Local cache path when used: `.cache/fractalrabbit/`
- Repository status here: not vendored or tracked by this project
- Upstream license: Apache License 2.0
- Upstream notice: portions of the upstream software are not subject to
  copyright protection under U.S. copyright law and are considered public
  domain by the upstream notice.

The external lane clones and builds the pinned upstream project only when
`configs/fractalrabbit_external.yaml` is selected and the host has JDK 21,
Maven, and git. If an external release bundle distributes upstream source,
compiled jars, or derivative code, it must include the applicable Apache-2.0
license and upstream NOTICE material.

## pymdp

- Package: `inferactively-pymdp==1.0.2`
- Upstream repository: https://github.com/infer-actively/pymdp
- Repository status here: Python package dependency only; source not vendored
- Upstream license: MIT License

## Runtime and Test Dependencies

The current package dependencies are declared in `pyproject.toml` and are
consumed as installed packages, not vendored source:

- `defusedxml` — Python Software Foundation License (PSF-2.0)
- `matplotlib` — Matplotlib License (PSF-based)
- `numpy` — BSD-3-Clause (with some bundled components under 0BSD, MIT, Zlib,
  and CC0-1.0 per the upstream license metadata)
- `pyyaml` — MIT License
- `pytest` (dev) — MIT License
- `pytest-cov` (dev) — MIT License

Public binary, container, or archival bundles should carry the dependency
notices required by their installed versions.

## Generated Artifacts and Fixtures

Checked fixtures under `data/fixtures/` are project-maintained synthetic
FractalRabbit-format CSV fixtures. Generated artifacts under `output/` are not
tracked in Git, but the release manifest hashes the artifacts that support a
rendered manuscript or release candidate.

No current project artifact is evidence of real-person mobility, operational
tasking, surveillance readiness, or empirical behavioral validation.
