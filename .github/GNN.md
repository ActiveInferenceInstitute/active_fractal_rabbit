# GNN conformance

`gnn conformance (pinned)` is defined in `.github/workflows/ci.yml` and reads
`.github/gnn-pin.json`, which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.0.3`
(`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`).

The job verifies the pin and installed parser, then runs
`tests/test_gnn_spec.py`. That test parses the two committed specifications
`models/single_factor_reference.gnn.md` and
`models/factorial_latent_mode.gnn.md` and validates each against the pinned
upstream `gnn.cli validate --strict`. It does not scan arbitrary repository
`*.gnn.md` files; this is a pin and model conformance baseline for those two
specifications, not a claim that every sketch is covered.
