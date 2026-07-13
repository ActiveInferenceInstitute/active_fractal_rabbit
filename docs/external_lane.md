# External FractalRabbit Lane: Optional Pinned Simulator Evidence {#sec:external-lane}

The external lane builds and runs the pinned upstream Java simulator instead of
the checked fixture. It is deliberately opt-in: nothing in the default pipeline,
tests, figures, or manuscript requires it.

## Host Prerequisites Explain Every Skip

The pinned FractalRabbit commit's Maven build targets **JDK 21**; the lane also
needs **Maven** and **git**. Where that toolchain is present the lane has been
executed end to end (see `output/runs/fractalrabbit_external/`); where it is
absent the lane is skipped with an explicit, named reason rather than failing
silently:

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
builds the jar-with-dependencies artifact; the second writes a full run under
`output/runs/fractalrabbit_external/` with the same artifact contract as the
fixture lane.

The FractalRabbit source and jars remain cache-local and are not tracked in this
project. `THIRD_PARTY_NOTICES.md` records the pinned upstream Apache-2.0 license
and NOTICE boundary, and `docs/ip_license_status.md` records what must be carried
forward if a public bundle ever distributes upstream source, jars, or derivative
simulator code.

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
