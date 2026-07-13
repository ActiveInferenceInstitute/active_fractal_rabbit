"""Tests for the confound taxonomy — real files, no mocks.

Builds a minimal fake project tree in tmp_path whose live state makes every verification hook
pass, then mutates individual artifacts to prove the hooks FIRE (a flipped artifact flips the
entry's ``verified`` flag). This is the teeth: the taxonomy cannot drift away from live state.
"""

from __future__ import annotations

import json
from pathlib import Path

from active_fractal_rabbit.confound_taxonomy import (
    CONFOUND_CLASSES,
    build_confound_registry,
    build_confound_taxonomy,
    run_and_write_confound_taxonomy,
)


def _passing_tree(root: Path) -> None:
    src = root / "src" / "active_fractal_rabbit"
    runs = root / "output" / "runs"
    revision = root / "docs" / "revision"
    for d in (src, runs, revision):
        d.mkdir(parents=True, exist_ok=True)
    # advantage_frontier.py intentionally NOT created (retracted -> absent)
    (src / "partial_observability.py").write_text(
        "# matched emission: draw over n-1 non-true cells, then skip the true index\n"
        "# observed[flip_idx] = draw + (draw >= true_at_flip)\n",
        encoding="utf-8",
    )
    (src / "confound_controls.py").write_text(
        "# mobility + dominant-cell battery\n", encoding="utf-8"
    )
    (src / "sota_aif_candidate.py").write_text(
        "# precision weighting: b ** gamma renormalised, equivalent to softmax(gamma * log b)\n",
        encoding="utf-8",
    )
    (runs / "sota_program_gate.json").write_text(
        json.dumps({"gate_clears": False, "status": "sota_program_not_cleared"}),
        encoding="utf-8",
    )
    (runs / "aif_regime_search.json").write_text(
        json.dumps(
            {
                "regime_summaries": {
                    "symmetric_noise_045": {
                        "regime_id": "symmetric_noise_045",
                        "gate_clears": True,  # matches the ledger's uncorrected_significant below
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (runs / "power_analysis.json").write_text(
        json.dumps(
            {
                "large_clean_advantage_nats": -0.005,
                "aif_advantage_vanishes_at_scale_clean": True,
            }
        ),
        encoding="utf-8",
    )
    (runs / "partial_observability_analysis.json").write_text(
        json.dumps(
            {
                "advantage_positive_under_noise": True,
                "cross_realization_sign_stable_under_noise": True,
            }
        ),
        encoding="utf-8",
    )
    (runs / "multiplicity_ledger.json").write_text(
        json.dumps(
            {
                "family_size": 9,
                "holm": {"survivor_ids": []},
                "comparisons": [
                    {
                        "comparison_id": "aif_regime::symmetric_noise_045",
                        "uncorrected_significant": True,
                        "individually_clears": True,
                        "holm_survives": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (revision / "REVISION_PROGRAM.md").write_text(
        "long-memory in-sample. Multiscale backoff null.\n",
        encoding="utf-8",
    )


def test_registry_classes_are_valid():
    registry = build_confound_registry()
    valid = set(CONFOUND_CLASSES) | {"none"}
    for entry in registry:
        assert entry.confound_class in valid, entry.entry_id
    # exactly one surviving positive
    positives = [e for e in registry if e.confound_class == "none"]
    assert len(positives) == 1
    assert positives[0].entry_id == "partial_observability_positive"


def test_instrumentation_bug_class_is_mece_and_still_counted_as_a_confound():
    """Advisor gate (session 27, 2026-07-02): the new 'instrumentation-bug' class must not become
    a laundering bucket — (1) exactly one entry uses it (a negative control that no OTHER confound
    got silently relabeled into it), and (2) that entry is still counted in n_confounds (a
    'defect' does not stop being tallied just because it's an instrumentation bug rather than a
    model-level confound)."""
    registry = build_confound_registry()
    instrumentation_bug_entries = [
        e for e in registry if e.confound_class == "instrumentation-bug"
    ]
    assert [e.entry_id for e in instrumentation_bug_entries] == [
        "regime_qualified_clearance"
    ]
    # still a "confound" for counting purposes (class != "none"), not silently excluded
    confound_rows = [e for e in registry if e.confound_class != "none"]
    assert "regime_qualified_clearance" in {e.entry_id for e in confound_rows}


def test_all_hooks_pass_on_a_consistent_tree(tmp_path):
    """`individual_vs_population_confinement` re-derives its claim from a REAL external-simulator
    trace (config + parsed waypoint CSV), which this synthetic fixture tree deliberately does not
    fabricate (that would require genuine confined-mobility waypoint data, not a stub file) — so it
    is the one entry expected to read `unverified` here, exactly as it does on a host with no
    cached external-lane run (the external lane is optional; requires local JDK/Maven). Every OTHER
    entry must still verify: this is not a blanket relaxation, only the one documented,
    environment-dependent exception. See test_individual_vs_population_confinement_hook_fails_closed
    below for direct proof this hook still has teeth on missing-artifact inputs."""
    _passing_tree(tmp_path)
    payload = build_confound_taxonomy(tmp_path)
    unverified = [r for r in payload["entries"] if not r["verified"]]
    assert [r["entry_id"] for r in unverified] == [
        "individual_vs_population_confinement"
    ], unverified
    assert payload["n_unverified"] == 1
    assert payload["n_confounds"] == 7
    assert payload["n_confound_classes_present"] >= 5
    # cross-vendor catches are recorded (noise frontier, long-memory)
    assert payload["n_cross_vendor_caught"] >= 2
    assert payload["schema"] == "active-fractal-rabbit-confound-taxonomy-v1"
    # Forge M2: the verified tally is split so doc-greps are not read as live verification.
    assert (
        payload["n_verified_live_state"] + payload["n_verified_documented"]
        == payload["n_verified"]
    )
    assert (
        payload["n_verified_documented"] == 2
    )  # long-memory, multiscale (doc-grade)
    assert payload["n_verified_live_state"] == payload["n_verified"] - 2
    assert "verified against live state" in payload["verdict"]


def test_individual_vs_population_confinement_hook_fails_closed(tmp_path):
    """Direct proof-of-detection for the re-derivation hook added this session: it must report
    unverified (not a vacuous pass) when the real external-trace artifact it depends on is
    missing, at both failure points (config absent, trace file absent) — never fall back to
    reading an unrelated file's mere presence as evidence (the bug this hook replaced)."""
    from active_fractal_rabbit.confound_taxonomy import (
        _verify_individual_vs_population_confinement,
    )

    root_no_config = tmp_path / "no_config"
    root_no_config.mkdir()
    result = _verify_individual_vs_population_confinement(root_no_config)
    assert result["verified"] is False
    assert result["kind"] == "live_artifact"
    assert "config absent" in result["evidence"]

    root_no_trace = tmp_path / "no_trace"
    (root_no_trace / "configs").mkdir(parents=True)
    real_config = (
        Path(__file__).resolve().parents[1] / "configs" / "fractalrabbit_external.yaml"
    ).read_text(encoding="utf-8")
    # Same schema as the real project config (so load_config succeeds identically), pointed at
    # an output_dir under tmp_path where no trace CSV has been written.
    (root_no_trace / "configs" / "fractalrabbit_external.yaml").write_text(
        real_config.replace(
            "output_dir: output/runs/fractalrabbit_external",
            "output_dir: output/runs_no_trace",
        ),
        encoding="utf-8",
    )
    result = _verify_individual_vs_population_confinement(root_no_trace)
    assert result["verified"] is False
    assert result["kind"] == "live_artifact"
    assert "trace absent" in result["evidence"]


def test_noise_frontier_hook_fires_when_module_resurrected(tmp_path):
    _passing_tree(tmp_path)
    # resurrect the retracted module -> the absence hook must FIRE
    (tmp_path / "src" / "active_fractal_rabbit" / "advantage_frontier.py").write_text(
        "x = 1\n", encoding="utf-8"
    )
    payload = build_confound_taxonomy(tmp_path)
    row = next(r for r in payload["entries"] if r["entry_id"] == "noise_frontier")
    assert row["verified"] is False
    assert payload["n_unverified"] >= 1


def test_data_density_hook_fires_when_scale_advantage_positive(tmp_path):
    _passing_tree(tmp_path)
    # flip the at-scale clean advantage to a positive (non-vanishing) value -> hook must FIRE
    (tmp_path / "output" / "runs" / "power_analysis.json").write_text(
        json.dumps(
            {
                "large_clean_advantage_nats": 0.5,
                "aif_advantage_vanishes_at_scale_clean": False,
            }
        ),
        encoding="utf-8",
    )
    payload = build_confound_taxonomy(tmp_path)
    row = next(
        r
        for r in payload["entries"]
        if r["entry_id"] == "fair_frontier_scale_fragility"
    )
    assert row["verified"] is False


def test_regime_clearance_instrumentation_bug_hook_fires_on_mismatch(tmp_path):
    """The regime_qualified_clearance entry now documents a FIXED instrumentation bug (session 27,
    2026-07-02): the ledger's uncorrected_significant field must match an INDEPENDENT re-derivation
    of gate_clears from the raw upstream artifact (aif_regime_search.json / sota_program_gate.json),
    not merely the ledger's own internally-consistent fields (Forge cross-vendor HIGH-2: comparing
    the ledger against itself is true by construction once the ledger derives one field from the
    other, so this hook must re-derive from a genuinely independent source). It must FIRE the
    moment the ledger and the raw upstream artifact disagree, and must never vacuously pass on an
    absent ledger, an empty comparison list, or an upstream artifact that doesn't cover the id."""
    _passing_tree(tmp_path)
    # remove the ledger -> the fix cannot be adjudicated (unverified, not silently passed)
    (tmp_path / "output" / "runs" / "multiplicity_ledger.json").unlink()
    payload = build_confound_taxonomy(tmp_path)
    row = next(
        r for r in payload["entries"] if r["entry_id"] == "regime_qualified_clearance"
    )
    assert row["verified"] is False
    # the exact bug shape: ledger says uncorrected_significant=True, but the RAW upstream artifact
    # (aif_regime_search.json, still gate_clears=True from _passing_tree) — flip the upstream
    # artifact to gate_clears=False so it disagrees with the ledger -> the hook must FIRE.
    (tmp_path / "output" / "runs" / "aif_regime_search.json").write_text(
        json.dumps(
            {
                "regime_summaries": {
                    "symmetric_noise_045": {
                        "regime_id": "symmetric_noise_045",
                        "gate_clears": False,  # disagrees with the ledger below
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "output" / "runs" / "multiplicity_ledger.json").write_text(
        json.dumps(
            {
                "family_size": 9,
                "holm": {"survivor_ids": []},
                "comparisons": [
                    {
                        "comparison_id": "aif_regime::symmetric_noise_045",
                        "uncorrected_significant": True,
                        "individually_clears": True,
                        "holm_survives": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload2 = build_confound_taxonomy(tmp_path)
    row2 = next(
        r for r in payload2["entries"] if r["entry_id"] == "regime_qualified_clearance"
    )
    assert row2["verified"] is False
    # vacuous-pass guard: a ledger with NO comparisons must not verify the fix — there is nothing
    # to re-derive the invariant from.
    (tmp_path / "output" / "runs" / "multiplicity_ledger.json").write_text(
        json.dumps({"family_size": 0, "holm": {"survivor_ids": []}, "comparisons": []}),
        encoding="utf-8",
    )
    payload3 = build_confound_taxonomy(tmp_path)
    row3 = next(
        r for r in payload3["entries"] if r["entry_id"] == "regime_qualified_clearance"
    )
    assert row3["verified"] is False
    # vacuous-pass guard #2: a comparison whose id the upstream artifact does not cover must not
    # silently verify either — "checked > 0" requires at least one real independent re-derivation.
    (tmp_path / "output" / "runs" / "aif_regime_search.json").write_text(
        json.dumps({"regime_summaries": {}}), encoding="utf-8"
    )
    (tmp_path / "output" / "runs" / "multiplicity_ledger.json").write_text(
        json.dumps(
            {
                "family_size": 9,
                "holm": {"survivor_ids": []},
                "comparisons": [
                    {
                        "comparison_id": "aif_regime::symmetric_noise_045",
                        "uncorrected_significant": True,
                        "individually_clears": True,
                        "holm_survives": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload4 = build_confound_taxonomy(tmp_path)
    row4 = next(
        r for r in payload4["entries"] if r["entry_id"] == "regime_qualified_clearance"
    )
    assert row4["verified"] is False
    # positive case: ledger agrees with an independently re-derived upstream gate_clears ->
    # verified True (this is what _passing_tree itself sets up, confirmed separately by
    # test_all_hooks_pass_on_a_consistent_tree).
    (tmp_path / "output" / "runs" / "aif_regime_search.json").write_text(
        json.dumps(
            {
                "regime_summaries": {
                    "symmetric_noise_045": {
                        "regime_id": "symmetric_noise_045",
                        "gate_clears": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "output" / "runs" / "multiplicity_ledger.json").write_text(
        json.dumps(
            {
                "family_size": 9,
                "holm": {"survivor_ids": []},
                "comparisons": [
                    {
                        "comparison_id": "aif_regime::symmetric_noise_045",
                        "uncorrected_significant": False,
                        "individually_clears": False,
                        "holm_survives": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload5 = build_confound_taxonomy(tmp_path)
    row5 = next(
        r for r in payload5["entries"] if r["entry_id"] == "regime_qualified_clearance"
    )
    assert row5["verified"] is True


def test_run_and_write_emits_artifact(tmp_path):
    _passing_tree(tmp_path)
    path = run_and_write_confound_taxonomy(tmp_path)
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["verdict"]
    assert payload["n_entries"] == len(build_confound_registry())
