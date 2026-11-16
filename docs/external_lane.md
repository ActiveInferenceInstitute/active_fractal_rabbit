# External FractalRabbit Lane: Optional Pinned Simulator Evidence {#sec:external-lane}

The external lane builds and runs the pinned upstream Java simulator instead of
the checked fixture. It is deliberately opt-in: nothing in the default pipeline,
tests, figures, or manuscript requires it.

## Host Prerequisites Explain Every Skip

The pinned FractalRabbit commit's Maven build targets **JDK 21**; the lane also
needs **Maven** and **git**. A saved `output/runs/fractalrabbit_external/`
directory and toolchain availability do not by themselves establish current
qualified completion. When the toolchain is absent, the lane is skipped with
an explicit, named reason rather than failing silently:

```bash
brew install openjdk@21 maven
```

No PATH or `java_home` surgery is required: Homebrew installs `openjdk@21`
keg-only, and the boundary layer (`fractalrabbit.resolve_java`) finds it by
checking, in order, the `AFR_JAVA_HOME` environment override, the PATH `java`,
the Homebrew keg locations, and `/usr/libexec/java_home -v 21`. The resolved
home is exported as `JAVA_HOME` to both the Maven build and the simulator
subprocess.

`scripts/00_preflight.py` reports the toolchain state, and
`active_fractal_rabbit.preflight.external_lane_status()` returns a readiness
boolean plus explicit human-readable reasons. The marked integration test and
the fixture-versus-external comparison test skip with those exact reasons when
the lane is unavailable — a skipped external test always names the missing
tool or Java version, never silently passes.

## External and Full-Study Commands Stay Lane-Qualified

The one-command path is the full-study orchestrator, which runs the external
lane (prepare, pipeline, resimulating sweep, figures to `output/figures_external/`)
and the fixture lane, then regenerates variables and the release manifest:

```bash
uv run python scripts/07_run_full_study.py --lane both
```

The manual path:

```bash
uv run python scripts/01_prepare_fractalrabbit.py --config configs/fractalrabbit_external.yaml
uv run python scripts/02_run_pipeline.py --config configs/fractalrabbit_external.yaml
```

The first command clones the pinned commit into `.cache/fractalrabbit/` and
builds the jar-with-dependencies artifact; the second writes the base run to the
configured `output_dir` (by default `output/runs/fractalrabbit_external/`).
Derived runs use that leaf's parent collection, as described in
[`config_schema.md`](config_schema.md). External **sensitivity** children and
their simulator prefixes are lane-qualified; this does not guarantee isolation
for variance-realization or fractal-dimension children in a shared collection.
Select separate collections or supported explicit destinations for those families.

The FractalRabbit source and jars remain cache-local and are not tracked in this
project. `THIRD_PARTY_NOTICES.md` records the pinned upstream Apache-2.0 license
and NOTICE boundary, and `docs/ip_license_status.md` records what must be carried
forward if a public bundle ever distributes upstream source, jars, or derivative
simulator code.

## Completion Requires Build and Fresh-Bundle Custody

External completion requires observed build inputs and the actual executable,
not just the configured upstream pin. The build admits the checkout's root,
origin, clean source state, revision, and tree, then binds the successfully
built runnable JAR by digest. Dirty or unverifiable inputs, identity mismatches,
failed builds, and zero or multiple candidate JARs are rejected without
resetting or deleting source and without reusing an old executable. An explicitly
selected upstream-sensitivity revision retains its requested and observed
identity rather than being relabelled with the default pin.

Execution must produce a fresh, validated XY bundle and, when present, an
aligned finite PLACES companion, bound to the parameter snapshot and executable
that ran. Companion absence is explicit. Newly produced identical bytes are
admissible, but existing XY or PLACES files cannot fill in missing fresh output.
A failed attempt preserves prior completed output rather than publishing
partial completion. Implicit canonical observation resolution checks the bound
execution receipt; missing or changed bound artifacts fail closed. Explicitly
supplied paths/observations do not perform that receipt verification: the caller
must admit their source, freshness, executable, and companion custody before
passing them to a receiving analysis.

Old saved external artifacts remain historical and are not silently upgraded
into current qualified receipts. Source/CLI acceptance with recorded transports
does not qualify an actual Java/Maven simulator build, a fresh external
realization, or scientific or empirical success. Those require separately
authorized execution and the existing promotion evidence.

## Fixture and External Tensors Are Compared, Not Pooled

Once an external run exists, `active_fractal_rabbit.compare` provides a
descriptive tensor-summary comparison:

```python
from active_fractal_rabbit.compare import compare_tensor_summary_files

report = compare_tensor_summary_files(
    "output/runs/default/tensor_summary.json",
    "output/runs/fractalrabbit_external/tensor_summary.json",
)
```

The report covers modality/control agreement, tensor shapes, and
normalization deltas. It deliberately carries the caveat that structural
agreement shows interface compatibility only — it is not evidence of external
validity, and no manuscript claim may be upgraded on the basis of this
comparison alone.
