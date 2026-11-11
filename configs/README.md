# configs/

Study-lane configuration for Active FractalRabbit. Each file selects a lane and its
simulator, observation, model, experiment, figure, and robustness settings. The
per-key contract is documented in [`../docs/config_schema.md`](../docs/config_schema.md);
agent rules are in [`AGENTS.md`](AGENTS.md).

## What is here

| File | Lane |
| --- | --- |
| `default.yaml` | Fixture-first default lane (large checked fixture; no Java required). |
| `fractalrabbit_external.yaml` | Optional external FractalRabbit lane (pinned simulator; requires JDK 21 + Maven). |
| `sweep.yaml` | Sensitivity sweep settings. |
| `fixtures/*.yaml` | Config-local variants (deep latent, fixed bounds/burst, identity ablation, quantile bins). |

## Running with a config

```bash
uv run python scripts/02_run_pipeline.py --config configs/default.yaml
uv run python scripts/02_run_pipeline.py --config configs/fractalrabbit_external.yaml
```

Config paths are repository-relative. Keep fixture-backed and external lanes separate,
and never put secrets, tokens, or host-local paths in these files.
