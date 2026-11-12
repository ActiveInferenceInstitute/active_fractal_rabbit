from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from active_fractal_rabbit.aif_regime_search import run_and_write_aif_regime_search
from active_fractal_rabbit.manuscript_variables import (
    compute_variables,
    _require_canonical_sota_program,
)
from active_fractal_rabbit.robustness import write_variance_bands
from active_fractal_rabbit.variable_provenance import (
    DEFAULT_TOKEN_PROVENANCE,
    ProvenanceError,
    TokenProvenanceSpec,
    build_token_provenance,
    verify_token_provenance,
)


def _canonical_aif_regime_artifacts_exist(root: Path) -> bool:
    path = root / "output" / "runs" / "aif_regime_sota_gate.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    regimes = payload.get("regime_summaries", {})
    if not isinstance(regimes, dict):
        return False
    return {"identity_emission_collapse", "symmetric_noise_045"} <= set(regimes)


def _ensure_aif_regime_artifacts(root: Path) -> None:
    if _canonical_aif_regime_artifacts_exist(root):
        return
    run_and_write_aif_regime_search(
        config_path=root / "configs" / "default.yaml",
        out_dir=root / "output" / "runs",
        n_travellers=8,
        n_boot=20,
        regime_ids=("identity_emission_collapse",),
        realization_seeds=(20260625,),
    )


def test_token_provenance_detects_missing_artifact_and_missing_key(tmp_path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text(json.dumps({"n_observations": 960.0}) + "\n", encoding="utf-8")
    specs = [
        TokenProvenanceSpec("OBSERVATION_COUNT", "metrics.json", ("n_observations",)),
        TokenProvenanceSpec("MISSING_KEY", "metrics.json", ("does_not_exist",)),
    ]

    with pytest.raises(ProvenanceError, match="missing key"):
        build_token_provenance(
            tmp_path,
            {"OBSERVATION_COUNT": "960", "MISSING_KEY": "0"},
            specs=specs,
        )

    with pytest.raises(ProvenanceError, match="missing artifact"):
        build_token_provenance(
            tmp_path,
            {"OBSERVATION_COUNT": "960"},
            specs=[
                TokenProvenanceSpec(
                    "OBSERVATION_COUNT", "absent.json", ("n_observations",)
                )
            ],
        )


def test_token_provenance_detects_stale_checksum_and_value_drift(tmp_path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text(json.dumps({"n_observations": 960.0}) + "\n", encoding="utf-8")
    specs = [
        TokenProvenanceSpec("OBSERVATION_COUNT", "metrics.json", ("n_observations",))
    ]
    payload = build_token_provenance(
        tmp_path, {"OBSERVATION_COUNT": "960"}, specs=specs
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    assert verify_token_provenance(provenance) == {"checked": ["OBSERVATION_COUNT"]}

    artifact.write_text(json.dumps({"n_observations": 97.0}) + "\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="checksum"):
        verify_token_provenance(provenance)

    with pytest.raises(ProvenanceError, match="rendered value"):
        build_token_provenance(tmp_path, {"OBSERVATION_COUNT": "960"}, specs=specs)


def test_provenance_rejects_empty_wrong_schema_duplicate_and_incomplete_entries(
    tmp_path: Path,
) -> None:
    provenance = tmp_path / "provenance.json"
    base = {
        "schema": "bogus",
        "project_root": str(tmp_path),
        "declared_tokens": [],
        "entries": [],
    }
    provenance.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="schema"):
        verify_token_provenance(provenance)

    base["schema"] = "active-fractal-rabbit-manuscript-variable-provenance-v2"
    provenance.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="non-empty"):
        verify_token_provenance(provenance)

    duplicate = {
        **base,
        "declared_tokens": ["X", "X"],
        "entries": [{"token": "X"}, {"token": "X"}],
    }
    provenance.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="unique"):
        verify_token_provenance(provenance)

    artifact = tmp_path / "metrics.json"
    artifact.write_text('{"n_observations": 960}\n', encoding="utf-8")
    specs = (
        TokenProvenanceSpec("OBSERVATION_COUNT", "metrics.json", ("n_observations",)),
        TokenProvenanceSpec("SECOND", "metrics.json", ("n_observations",)),
    )
    payload = build_token_provenance(
        tmp_path, {"OBSERVATION_COUNT": "960"}, specs=specs[:1]
    )
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="required token set"):
        verify_token_provenance(provenance, expected_specs=specs)


def test_standard_provenance_file_must_verify_against_bundle_local_root(
    tmp_path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    artifact = source_root / "metrics.json"
    artifact.write_text(json.dumps({"n_observations": 960.0}) + "\n", encoding="utf-8")
    specs = [
        TokenProvenanceSpec("OBSERVATION_COUNT", "metrics.json", ("n_observations",))
    ]
    payload = build_token_provenance(
        source_root, {"OBSERVATION_COUNT": "960"}, specs=specs
    )

    copied = (
        tmp_path / "copied" / "output" / "data" / "manuscript_variable_provenance.json"
    )
    copied.parent.mkdir(parents=True)
    copied.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ProvenanceError, match="project_root does not match"):
        verify_token_provenance(copied)


def test_sota_program_tokens_require_canonical_artifact(tmp_path) -> None:
    artifact = tmp_path / "sota_program_gate.json"
    artifact.write_text(
        json.dumps({"status": "sota_program_not_cleared"}) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="not a canonical SOTA-program artifact"):
        _require_canonical_sota_program(
            json.loads(artifact.read_text(encoding="utf-8")), artifact
        )

    artifact.write_text(
        json.dumps(
            {
                "run_parameters": {
                    "canonical": True,
                    "n_travellers": 8,
                    "eps": 0.3,
                    "spatial_bins": 6,
                    "n_boot": 2000,
                    "boot_seed": 7,
                    "lanes": ["fractalrabbit", "gravity"],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="n_travellers=8 expected 64"):
        _require_canonical_sota_program(
            json.loads(artifact.read_text(encoding="utf-8")), artifact
        )


def test_claim_critical_sota_fair_aif_power_tokens_are_provenance_declared() -> None:
    root = Path(__file__).resolve().parents[1]
    manuscript_tokens: set[str] = set()
    token_re = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
    for path in sorted((root / "manuscript").glob("*.md")):
        manuscript_tokens.update(token_re.findall(path.read_text(encoding="utf-8")))

    critical = {
        token
        for token in manuscript_tokens
        if token.startswith(("FAIR_", "POWER_", "AIF_"))
        or token in {"SOTA_STRONGEST_BASELINE_LOSS", "SOTA_LEAKAGE_PROBE_LOSS"}
    }
    declared = {spec.token for spec in DEFAULT_TOKEN_PROVENANCE}

    assert sorted(critical - declared) == []


def test_every_provenance_bound_token_is_used_in_manuscript() -> None:
    """Spec/usage drift gate: a provenance-bound token nobody renders protects
    nothing, while its used sibling can silently fall back to a default. If a
    manuscript edit retires a token, retire (or rebind) its spec in the same
    change; if a new result token is published, bind it here."""
    root = Path(__file__).resolve().parents[1]
    token_re = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
    used: set[str] = set()
    for path in sorted((root / "manuscript").glob("*.md")):
        used.update(token_re.findall(path.read_text(encoding="utf-8")))
    declared = {spec.token for spec in DEFAULT_TOKEN_PROVENANCE}

    stale = sorted(declared - used)
    assert stale == [], (
        f"provenance-bound tokens no longer used in the manuscript: {stale}"
    )


def test_project_result_tokens_are_provenance_declared() -> None:
    """Declaration-coverage gate, checked statically against the spec registry.

    Value-level verification (artifact exists, key exists, formatter byte-match)
    is exercised by the unit tests above, the sandboxed release-manifest and
    figures surface tests, and scripts/11_check_manuscript_variable_provenance.py
    against production artifacts. This test only asserts that every project
    result token remains provenance-DECLARED, so it needs no artifact surface.
    """
    declared = {spec.token for spec in DEFAULT_TOKEN_PROVENANCE}
    for token in [
        "OBSERVATION_COUNT",
        "HOLDOUT_COUNT",
        "PYMDP_EXACT_INFERENCE_DELTA",
        "ANOMALY_VFE_AUC_RARE",
        "FRONTIER_NO_LOCATION_MASS",
        "RETURN_MOVE_ENTROPY_REDUCTION",
        "PRIVACY_FRONTIER_NO_LOCATION_UNICITY",
        "HARDENED_ANOMALY_SPEED_VFE_AUC",
        "FRACTAL_DIMENSION_SWEEP_STATUS",
        "CIRCADIAN_PHASE_GAIN",
        "PYMDP_SOTA_STATUS",
        "PYMDP_PERFORMANCE_DELTA",
        "PYMDP_ROADMAP_GATES",
        "VARIANCE_GAP_FILL_LOW",
        "VARIANCE_GAP_FILL_HIGH",
        "SEQUENCE_BASELINE_BEST_MODEL",
        "SEQUENCE_BASELINE_BEST_LOSS",
        "SEQUENCE_BASELINE_GATE",
        "MISSINGNESS_VERDICT",
        "MISSINGNESS_PHASE_GAIN",
        "MISSINGNESS_COMBINED_GAIN",
        "RETURN_PHASE_VERDICT",
        "RETURN_PHASE_BEST_MODEL",
        "RETURN_PHASE_BEST_LOSS",
        "RETURN_PHASE_GAIN_VS_PERSISTENCE",
        "RETURN_PHASE_GAIN_OVER_PERMUTED",
        "SPATIAL_RESOLUTION_VERDICT",
        "SPATIAL_RESOLUTION_DEFAULT_BINS",
        "SPATIAL_RESOLUTION_DEFAULT_STATES",
        "SPATIAL_RESOLUTION_DEFAULT_FULL_LOSS",
        "SPATIAL_RESOLUTION_DEFAULT_MINUS_COARSE_LOSS",
        "SPATIAL_RESOLUTION_DEFAULT_MINUS_COARSE_NO_LOCATION_MASS",
        "SPATIAL_RESOLUTION_PYMDP_DELTA",
        "MODEL_GATE_MATRIX_STATUS",
        "MODEL_GATE_SEQUENCE_EXTERNAL_GATE",
        "MODEL_GATE_MISSINGNESS_EXTERNAL_VERDICT",
        "MODEL_GATE_RETURN_PHASE_EXTERNAL_VERDICT",
        "FORWARD_GATE_STATUS",
        "LATENT_REGIME_HMM_GATE",
        "LATENT_REGIME_HMM_BEST_HELDOUT_LOSS",
        "NEURAL_BASELINE_GATE",
        "NEURAL_BASELINE_LOSS",
        "LEARNED_STAY_RATE_VERDICT",
        "LEARNED_STAY_RATE_DIFF",
        "LEARNED_STAY_RATE_CI_LOW",
        "LEARNED_STAY_RATE_CI_HIGH",
        "HELDOUT_REPORTING_VERDICT",
        "HELDOUT_REPORTING_RUNLEN_LOTO_GAIN",
        "HELDOUT_REPORTING_RUNLEN_TEMPORAL_GAIN",
        "CROSS_GENERATOR_PERSISTENCE_VERDICT",
        "CROSS_GENERATOR_RETURN_VERDICT",
        "EFE_LOCATION_VERDICT",
        "EFE_LOCATION_SHARE",
        "EFE_METADATA_NULL_P",
        "EFE_RESIDUAL_WITHOUT_LOCATION",
        "LINKAGE_RISK_VERDICT",
        "LINKAGE_LOCATION_LINKAGE",
        "LINKAGE_NO_LOCATION_EXCEEDS_NULL",
        "POISONING_VERDICT",
        "POISONING_GROSS_DETECTED",
        "POISONING_SUBTLE_NULL",
        "POISONING_TELEPORT_AUC",
        "VISUAL_QA_STATUS",
        "VISUAL_QA_FIGURE_COUNT",
        "VISUAL_QA_CAPTION_SOURCE_COUNT",
        "VISUAL_QA_MIN_PIXEL_STDDEV",
        "EXTERNAL_OBSERVATION_COUNT",
        "SOTA_STRONGEST_BASELINE_LOSS",
        "SOTA_LEAKAGE_PROBE_LOSS",
        "AIF_SWEEP_BEST_LOSS",
        "FAIR_ORACLE_LOSS",
        "FAIR_SOFT_MARG_GAIN",
        "FAIR_SOTA_AIF_BEST_RATE",
        "POWER_MAX_TRAVELLERS",
        "POWER_LARGE_NOISY_ADV",
        "POWER_NOISE_RESTORES",
        "AIF_REGIME_SOTA_STATUS",
        "AIF_REGIME_SOTA_GATE_CLEARS",
        "AIF_REGIME_SOTA_BOUNDARY",
        "MULTIPLICITY_FAMILY_SIZE",
        "MULTIPLICITY_UNCORRECTED_CLEARS",
        "MULTIPLICITY_HOLM_SURVIVORS",
        "MULTIPLICITY_BH_SURVIVORS",
        "PARTIAL_OBS_MULTIPLICITY_DRAWS",
        "PARTIAL_OBS_MIN_DRAWS_BONFERRONI",
        "PARTIAL_OBS_CROSS_DRAW_MEAN_ADVANTAGE",
        "PARTIAL_OBS_CROSS_DRAW_MIN_ADVANTAGE",
        "PARTIAL_OBS_CROSS_DRAW_MAX_ADVANTAGE",
        "CONFOUND_N_DISSOLVED",
        "CONFOUND_N_CLASSES",
        "CONFOUND_CROSS_VENDOR_CAUGHT",
        "PRIVACY_UTILITY_FINEST_EXCESS",
        "PRIVACY_UTILITY_COARSEST_EXCESS",
        "PRIVACY_UTILITY_FINEST_REID",
        "PRIVACY_UTILITY_COARSEST_REID",
        "MINCOLLECT_FOUND",
        "MINCOLLECT_UTILITY_FLOOR",
        "MINCOLLECT_RISK_CEILING",
        "MINCOLLECT_REID",
        "MINCOLLECT_EXCESS_UTILITY",
        "MODEL_COMPARISON_AIF_RANK",
        "MODEL_COMPARISON_N_METHODS",
        "MODEL_COMPARISON_STRONGEST_NON_AIF",
        "MODEL_COMPARISON_AIF_VS_NONAIF_GAP",
        "MODEL_COMPARISON_BELIEF_NONBELIEF_GAP",
        "MODEL_COMPARISON_N_BELIEF",
    ]:
        assert token in declared, f"result token not provenance-declared: {token}"
    observation_specs = [
        s for s in DEFAULT_TOKEN_PROVENANCE if s.token == "OBSERVATION_COUNT"
    ]
    assert observation_specs and all(s.optional is False for s in observation_specs)


def test_optional_external_token_renders_unavailable_without_artifact(tmp_path) -> None:
    variables = {"EXTERNAL_OBSERVATION_COUNT": "unavailable"}
    specs = (
        TokenProvenanceSpec(
            "EXTERNAL_OBSERVATION_COUNT",
            "output/runs/fractalrabbit_external/metrics.json",
            ("n_observations",),
            optional=True,
        ),
    )
    payload = build_token_provenance(tmp_path, variables, specs=specs)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    assert payload["entries"][0]["status"] == "unavailable"
    assert verify_token_provenance(provenance) == {
        "checked": ["EXTERNAL_OBSERVATION_COUNT"]
    }


def test_optional_external_token_refuses_silent_zero_without_artifact(tmp_path) -> None:
    specs = (
        TokenProvenanceSpec(
            "EXTERNAL_OBSERVATION_COUNT",
            "output/runs/fractalrabbit_external/metrics.json",
            ("n_observations",),
            optional=True,
        ),
    )

    with pytest.raises(ProvenanceError, match="missing artifact"):
        build_token_provenance(
            tmp_path, {"EXTERNAL_OBSERVATION_COUNT": "0"}, specs=specs
        )


def test_compute_variables_renders_missing_external_metrics_as_unavailable(
    tmp_path,
) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "data" / "fixtures").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'tmp-afr'\nversion = '0.0.0'\n", encoding="utf-8"
    )
    (tmp_path / "data" / "fixtures" / "fixture.csv").write_text(
        "ID,Days,x(km),y(km)\n0,0,0,0\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "default.yaml").write_text(
        """
run_id: default
fractalrabbit:
  upstream_commit: 9933449c4f4fe1b26b6ac7bfdeeac76583085df5
  execution: fixture
  fixture_csv: data/fixtures/fixture.csv
  params:
    space_dimension: 2
    fractal_dimension: 1.33
    number_of_points: 2
    number_of_travellers: 1
    number_of_co_travellers: 0
    number_of_days: 1
    number_of_reports: 1
observation:
  spatial_bins: 1
  time_bins_per_day: 1
model:
  smoothing: 0.1
closed_loop:
  iterations: 1
  candidates: []
""",
        encoding="utf-8",
    )

    variables = compute_variables(tmp_path)

    assert variables["EXTERNAL_OBSERVATION_COUNT"] == "unavailable"
    assert variables["EXTERNAL_MEAN_PREDICTIVE_LOSS"] == "unavailable"
    assert variables["EXTERNAL_PYMDP_EXACT_DELTA"] == "unavailable"


def test_single_realization_variance_band_renders_unavailable(tmp_path) -> None:
    """A single draw is not a spread. If variance_bands.json ever ships with
    n_realizations < 2 (the historical test-clobber failure mode), the band
    tokens must render 'unavailable' — which also fails the provenance
    byte-match gate loudly — instead of dressing one run up as an empirical
    cross-realization band."""
    (tmp_path / "output" / "runs").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text(
        "run_id: default\n"
        "fractalrabbit:\n"
        "  upstream_commit: 9933449c4f4fe1b26b6ac7bfdeeac76583085df5\n"
        "  execution: fixture\n"
        "  fixture_csv: data/fixtures/fixture.csv\n"
        "observation:\n"
        "  spatial_bins: 1\n"
        "  time_bins_per_day: 1\n"
        "model:\n"
        "  smoothing: 0.1\n"
        "closed_loop:\n"
        "  iterations: 1\n"
        "  candidates: []\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'tmp-afr'\nversion = '0.0.0'\n", encoding="utf-8"
    )
    write_variance_bands(
        tmp_path / "output" / "runs" / "variance_bands.json",
        [
            {
                "gap_filling_skill": 0.1,
                "frontier_no_location_mass": 0.2,
                "anomaly_random_auc": 0.7,
            }
        ],
    )

    variables = compute_variables(tmp_path)

    assert variables["VARIANCE_REALIZATION_COUNT"] == "1"
    assert variables["VARIANCE_GAP_FILL_LOW"] == "unavailable"
    assert variables["VARIANCE_GAP_FILL_HIGH"] == "unavailable"


def test_gnn_variables_come_from_the_shared_reader_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The producer asks the pinned reader for the gnn_* keys, not the pin JSON."""

    runtime = tmp_path / "python"
    runtime.write_text(
        "#!/bin/sh\n"
        "printf '%s' '{\"gnn_pin_path\": \".github/gnn-pin.json\", "
        "\"gnn_commit\": \"75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf\"}'\n",
        encoding="utf-8",
    )
    runtime.chmod(0o755)
    monkeypatch.setenv("GNN_RUNTIME_PYTHON", str(runtime))

    from active_fractal_rabbit.manuscript_variables import _gnn_variables

    variables = _gnn_variables(Path(__file__).resolve().parents[1])

    assert variables == {
        "gnn_pin_path": ".github/gnn-pin.json",
        "gnn_commit": "75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf",
    }
