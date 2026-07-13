"""Semantic mini-brief audit for manuscript figures and cover art."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VISUAL_CLAIM_AUDIT_SCHEMA = "active-fractal-rabbit-visual-claim-audit-v1"


class VisualClaimAuditError(RuntimeError):
    """Raised when visual-claim audit violations are present."""


@dataclass(frozen=True)
class VisualExpectation:
    """Expected figure-level semantic contract."""

    story_job: str
    data_shape: str
    encoding: str
    expected_terms: tuple[str, ...]
    boundary_terms: tuple[str, ...]


FIGURE_EXPECTATIONS: dict[str, VisualExpectation] = {
    "evidence_atlas": VisualExpectation(
        "Orient the reader to the full evidence chain before the methods sections.",
        "Run metrics plus gate summaries from canonical fixture artifacts.",
        "Editorial atlas with sparse reports, categorical evidence, belief update, null gates, and safety outputs.",
        ("observations.csv", "discrete_features.csv", "minimization", "integrity"),
        (
            "surveillance readiness",
            "operational tasking",
            "empirical mobility validation",
        ),
    ),
    "waypoint_paths": VisualExpectation(
        "Show sparse report ordering and coverage gaps without implying continuous tracks.",
        "Observed report rows with traveller, day, x, and y fields.",
        "Per-traveller paths in report order with time-coded points.",
        ("observations.csv", "synthetic sporadic waypoint reports"),
        ("continuous-track reconstruction", "surveillance"),
    ),
    "spatial_bins": VisualExpectation(
        "Show the analysis-cell surface used by the model tensors.",
        "Discretized feature rows with cell assignments.",
        "Lattice view with occupied cells and report counts.",
        ("discrete_features.csv", "latent spatial cells"),
        ("geographic interpretation", "real-world map validity"),
    ),
    "spatial_resolution_frontier": VisualExpectation(
        "Make grid selection a measured frontier rather than a prose preference.",
        "Candidate grid rows with occupancy, loss, and no-location recovery metrics.",
        "Small-multiple frontier panels with verdict annotation.",
        ("spatial_resolution_frontier.json", "candidate square grids"),
        ("no-location recovery beyond the artifact frontier",),
    ),
    "reporting_bursts": VisualExpectation(
        "Expose reporting cadence as synthetic missingness structure.",
        "Positive inter-report intervals and burst categories.",
        "Interval trace and category counts.",
        ("discrete_features.csv", "reporting process"),
        ("real reporting cadence",),
    ),
    "speed_categories": VisualExpectation(
        "Expose the derived kinematic evidence channel and its cue boundary.",
        "Consecutive-report speed categories.",
        "Category bars plus derived speed trace.",
        ("discrete_features.csv", "speed evidence"),
        ("unobserved movement",),
    ),
    "transition_heatmap": VisualExpectation(
        "Audit transition-tensor sparsity and normalization.",
        "Cell-to-cell transition counts from discretized reports.",
        "Column-normalized heatmap with count context.",
        ("discrete_features.csv", "P(next cell | previous cell)"),
        ("external movement-model validation",),
    ),
    "likelihood_panels": VisualExpectation(
        "Audit observation likelihoods under the conditional-independence approximation.",
        "Modality likelihood tables for location, time, speed, and burst evidence.",
        "Small-multiple likelihood heatmaps.",
        ("conditional-independence approximation", "location, time, speed"),
        ("empirical sensor validation", "independence claims outside"),
    ),
    "posterior_entropy": VisualExpectation(
        "Show reference-filter predictive surprise and uncertainty over the trace.",
        "Inference trace rows with predictive loss and entropy.",
        "Time-series panels with uncertainty and loss annotations.",
        ("inference_trace.csv", "predictive loss"),
        ("real trajectory reconstruction",),
    ),
    "policy_proxy": VisualExpectation(
        "Expose proxy-control separability and preference terms.",
        "Policy trace rows with proxy score components.",
        "Score and entropy component panels.",
        ("policy_trace.csv", "proxy policy diagnostic"),
        ("operational control advice", "controller validation"),
    ),
    "candidate_scores": VisualExpectation(
        "Show closed-loop candidate ranking as a deterministic heuristic.",
        "Candidate-score JSON rows with objective, reporting ratio, and rank.",
        "Ranked bars with score labels.",
        ("candidate_scores.json", "declared next-run simulator candidates"),
        ("external simulator performance leadership",),
    ),
    "negative_controls": VisualExpectation(
        "Summarize destroyed-structure falsification checks.",
        "Negative-control summary rows with directional pass/fail probes.",
        "Pass/fail matrix with control labels.",
        ("negative_controls_summary.json", "4/4"),
        ("empirical mobility evidence",),
    ),
    "pymdp_agreement": VisualExpectation(
        "Separate matched-prior inference agreement from policy-induced divergence.",
        "pymdp and reference inference traces.",
        "Agreement and divergence panels with matched-prior annotation.",
        ("matched priors", "policy-induced prior divergence"),
        ("state-of-the-art performance claim",),
    ),
    "pymdp_diagnostics": VisualExpectation(
        "Expose pymdp interpretability surfaces rather than headline performance.",
        "pymdp trace rows with VFE, EFE, policy posterior, and runtime.",
        "Diagnostic time series and runtime summaries.",
        ("variational free energy", "expected free energy"),
        ("predictive-performance leadership",),
    ),
    "state_recovery": VisualExpectation(
        "Score synthetic-truth state recovery while flagging observed-location construction.",
        "Ground truth plus inference traces.",
        "Per-step recovery mass and correctness panels.",
        ("ground_truth.csv", "not empirical validity"),
        ("empirical validity",),
    ),
    "sparsity_recovery": VisualExpectation(
        "Show recovery degradation under deterministic report thinning.",
        "Sparsity analysis rows over achieved subsample fractions.",
        "Recovery and uncertainty curves by sparsity level.",
        ("sparsity_analysis.json", "subsamples"),
        ("population generalization",),
    ),
    "calibration": VisualExpectation(
        "Show null hierarchy and calibration floors for recovery claims.",
        "Calibration metrics, proper-score context, null floors, and per-bin counts.",
        "Shared-axis reliability panel plus Brier/null comparison with sparse-bin visibility.",
        ("calibration.json", "persistence"),
        ("by-construction location wins",),
    ),
    "anticipatory": VisualExpectation(
        "Show gap-filling verdict against the persistence oracle.",
        "Gap-filling analysis rows and planted-control checks.",
        "Skill bars and transition/cue panels.",
        ("anticipatory_analysis.json", "persistence"),
        ("real hidden paths",),
    ),
    "efe_decomposition": VisualExpectation(
        "Separate epistemic and pragmatic value in the active-sensing analysis.",
        "Active-sensing EFE component summary.",
        "Component bars or sweeps over preference sharpness.",
        ("epistemic", "pragmatic"),
        ("collection", "recommendation"),
    ),
    "anomaly_detection": VisualExpectation(
        "Show VFE surprise response to synthetic routine breaks.",
        "Anomaly analysis with injected perturbation scores.",
        "AUC and trace panels against novelty.",
        ("anomaly_analysis.json", "variational free energy"),
        ("deployment-ready anomaly detection",),
    ),
    "recoverability_frontier": VisualExpectation(
        "Quantify what modality minimization removes from recovery.",
        "Frontier rows over modality subsets.",
        "Recovery and occupancy-structure comparison panels.",
        ("frontier_analysis.json", "minimization"),
        ("real populations", "individual recoverability"),
    ),
    "return_structure": VisualExpectation(
        "Gate return features before promoting them as priors.",
        "Return-structure analysis with entropy and persistence headroom.",
        "Entropy-reduction bars with verdict.",
        ("return_structure_analysis.json", "entropy"),
        ("behavioral law",),
    ),
    "variance_bands": VisualExpectation(
        "Show synthetic-realization spread separately from population confidence.",
        "Variance band metrics across synthetic reruns.",
        "Median and min-max interval panels.",
        ("variance_bands.json", "synthetic-generator realizations"),
        ("population confidence intervals",),
    ),
    "upstream_sensitivity": VisualExpectation(
        "Show whether upstream simulator comparison is comparable or blocked.",
        "Upstream sensitivity status rows and attempted commits.",
        "Status panel with explicit blocked/comparable labeling.",
        ("upstream_sensitivity.json", "blocked"),
        ("silent robustness",),
    ),
    "fractal_dimension_sweep": VisualExpectation(
        "Treat fractal dimension as a generator knob and show sensitivity.",
        "Sweep rows over configured dimension values.",
        "Metric curves over dimension.",
        ("fractal_dimension_sweep.json", "dimension parameter"),
        ("real-mobility fractal dimension",),
    ),
    "circadian_phase_signal": VisualExpectation(
        "Gate phase-conditioned occupancy against a permutation null.",
        "Phase occupancy and gain metrics.",
        "Heatmap and gain bars.",
        ("circadian_phase_analysis.json", "permuted"),
        ("behavioral circadian claim",),
    ),
    "sequence_baselines": VisualExpectation(
        "Show the stronger prequential baseline ladder before performance language.",
        "Online sequence baseline loss rows.",
        "Ranked baseline comparisons.",
        ("sequence_baseline_analysis.json", "prequential"),
        ("pymdp predictive leadership",),
    ),
    "missingness_model": VisualExpectation(
        "Gate reporting-process structure against pooled and permuted nulls.",
        "Synthetic missingness model rows.",
        "Short-gap probabilities and log-loss gains.",
        ("missingness_model_analysis.json", "reporting cadence"),
        ("real nonresponse behavior",),
    ),
    "model_gate_matrix": VisualExpectation(
        "Show fixture/external readiness without stale external numbers.",
        "Lane-resolved gate matrix rows.",
        "Readiness matrix with unavailable states.",
        ("model_gate_matrix.json", "fixture and external lanes"),
        ("unavailable external lanes as failures or successes",),
    ),
    "m10_forward_gates": VisualExpectation(
        "Summarize promoted M10+ gates including nulls.",
        "Forward-gate summary and component artifacts.",
        "Gate status matrix.",
        ("forward_gates_summary.json", "honest nulls"),
        ("stale", "empirical simulator-validation"),
    ),
    "factorial_pymdp_modes": VisualExpectation(
        "Expose factor-specific posterior diagnostics in the factorial lane.",
        "Factorial pymdp posterior and label stability metrics.",
        "Per-factor entropy, confidence, and stability panels.",
        ("factorial_pymdp_analysis_fixture.json", "posterior entropy"),
        ("cognition", "intent", "mental state"),
    ),
    "latent_mode_recovery": VisualExpectation(
        "Report mean-field agreement and exact-marginal approximation cost.",
        "Factorial validation and recovery metrics.",
        "Agreement/gap and recovery-sharpness panels.",
        ("mean-field", "exact marginal"),
        ("empirical latent-mode validation",),
    ),
    "factorial_model_gate_matrix": VisualExpectation(
        "Compare factorial lane performance against stronger sequence baselines.",
        "Factorial model comparison rows.",
        "Model-loss matrix with gate status.",
        ("switching-HMM", "particle/state-space"),
        ("state-of-the-art predictive performance",),
    ),
    "efe_location_frontier": VisualExpectation(
        "Price location in epistemic value and check metadata recovery against a null.",
        "Location frontier rows, structure null, and positive control.",
        "Epistemic-value and localization-frontier panels.",
        ("efe_location_frontier.json", "epistemic"),
        ("collection recommendation",),
    ),
    "linkage_risk_frontier": VisualExpectation(
        "Show aggregate synthetic relink risk under minimization.",
        "Linkage frontier rows with structure-null bands and positive control.",
        "Risk bars with null intervals.",
        ("linkage_risk_frontier.json", "structure-null"),
        ("per-person linkage", "real-population"),
    ),
    "poisoning_robustness": VisualExpectation(
        "Show VFE integrity triage under crude synthetic corruptions.",
        "Poisoning robustness rows with VFE and Markov surprise.",
        "AUC/status panels against nulls.",
        ("poisoning_robustness.json", "Markov-surprise"),
        ("generalized robustness assurance",),
    ),
    "privacy_frontier": VisualExpectation(
        "Show synthetic unicity and co-travel minimization accounting.",
        "Privacy frontier rows over modality regimes.",
        "Risk metric panels by minimization regime.",
        ("privacy_frontier_analysis.json", "unicity"),
        ("real people", "association tasking"),
    ),
    "hardened_anomaly_detection": VisualExpectation(
        "Retest VFE surprise against a transition-aware Markov baseline.",
        "Hardened anomaly rows and operating-point metrics.",
        "AUC and operating-point comparison panels.",
        ("hardened_anomaly_analysis.json", "Markov-surprise"),
        ("deployment-ready anomaly monitoring",),
    ),
    "dataflow": VisualExpectation(
        "Trace the artifact contract from simulator boundary to manuscript variables.",
        "Source pipeline stages and generated artifact classes.",
        "Grouped artifact-flow diagram.",
        ("artifact dataflow", "manuscript variables"),
        ("trust in prose without artifacts",),
    ),
    "sota_leaderboard": VisualExpectation(
        "Rank the current FractalRabbit SOTA-program candidates by locked-split predictive loss with active inference directly included.",
        "SOTA-program leaderboard rows for active inference, soft filtering, sequence/state-space, neural/attention, Markov, persistence, and base-rate comparators.",
        "Horizontal loss bars on a shared loss axis with CI whiskers, active-inference rows colored distinctly, and the strongest-fair-baseline marker.",
        (
            "sota_program_leaderboard.csv",
            "precision-return AIF",
            "strongest fair baseline",
        ),
        ("global state-of-the-art performance claim",),
    ),
    "aif_regime_map": VisualExpectation(
        "Show where the separate regime-qualified AIF search has CI-separated local wins or bounded nulls.",
        "Regime-by-realization gate rows comparing the best AIF candidate with the strongest fair baseline.",
        "Signed fair-minus-AIF loss heatmap on a common scale with in-cell values, CI status text, and cleared-realization rings.",
        ("aif_regime_sota_gate.json", "regime-qualified synthetic search"),
        (
            "global SOTA",
            "external simulator SOTA",
            "empirical transfer",
            "operational readiness",
        ),
    ),
    "fair_inference": VisualExpectation(
        "Show how much latent-cell inference helps under a fair, transition-source-matched comparison.",
        "Predictor-ladder losses with CI plus the recursion-advantage-vs-noise curve.",
        "Horizontal loss bars with CI whiskers and a zero-crossing advantage curve over emission noise.",
        ("fair_inference.json", "soft marginalization"),
        ("state-of-the-art claim",),
    ),
    "fair_power_regime": VisualExpectation(
        "Show whether the fair AIF advantage survives a clustered bootstrap as travellers and noise vary.",
        "A traveller-count x observation-noise regime map of the clustered AIF advantage.",
        "A diverging heatmap with per-cell advantage values and solid or dashed rings that expose clustered-CI and seed-robust uncertainty status.",
        ("power_analysis.json", "clustered"),
        ("blanket state-of-the-art claim",),
    ),
    "partial_observability": VisualExpectation(
        "Isolate the genuine partial-observability regime where belief marginalization helps: marginalizing a belief versus committing to a point estimate under a known noisy emission.",
        "Soft-belief advantage in nats as a function of the observation-noise level epsilon, for soft-vs-hard and soft-vs-observed contrasts.",
        "A line plot over epsilon with a zero reference, the mean curve, a min-max envelope band across resamplings, and a fitted trend slope, growing monotonically from zero.",
        ("epsilon", "monotonically", "point estimate"),
        ("active-inference-specific superiority", "state-of-the-art claim"),
    ),
    "multiplicity_ledger": VisualExpectation(
        "Show how many of the benchmark's directional AIF-vs-fair comparisons survive multiplicity correction once the size of the comparison family is priced in.",
        "Per-comparison uncorrected and Holm-adjusted one-sided p-values for every comparison in the deterministically enumerated family.",
        "A log-scale dumbbell plot pairing each comparison's uncorrected p with its Holm-adjusted p against the alpha threshold line.",
        ("multiplicity_ledger.json", "Holm", "uncorrected"),
        ("state-of-the-art claim", "real-mobility claim"),
    ),
    "confound_taxonomy": VisualExpectation(
        "Map every apparent active-inference advantage this benchmark raised and dissolved onto the confound class that explains it, marking the single surviving positive.",
        "One marker per taxonomy entry positioned by confound class, with cross-vendor-caught and surviving-positive markers distinguished.",
        "A categorical scatter of entries over confound-class rows, with a starred surviving positive and diamond markers for cross-vendor catches.",
        ("confound_taxonomy.json", "confound", "cross-vendor"),
        ("state-of-the-art claim", "real-mobility claim"),
    ),
    "regime_mechanism_synthesis": VisualExpectation(
        "Show WHERE active inference wins or loses across the regime grid, HOW MUCH (magnitude and CI), and WHY (multiplicity and confound-battery standing), in one place.",
        "Per-regime AIF-vs-fair magnitude with clustered CI, Holm-survival status, and confound-battery/mobility standing, joined into one mechanism verdict per regime.",
        "A color-coded dumbbell/error-bar plot over regimes, colored by mechanism verdict, with per-row confound-battery and mobility annotations.",
        ("regime_mechanism_synthesis.json", "Holm", "confound"),
        ("state-of-the-art claim", "real-mobility claim"),
    ),
    "regime_switch_phase_diagnostic": VisualExpectation(
        "Show structurally WHERE within one regime's trajectory an AIF-vs-fair-baseline loss sits, not just its pooled magnitude.",
        "Per-movement-phase AIF advantage with a traveller-clustered CI, for the one regime whose generator has a named 3-segment movement-mode schedule.",
        "A bar-and-error-bar plot over the 3 phase segments in trajectory order, with a status box reporting whether the disadvantage is uniform across segments.",
        ("regime_switch_phase_diagnostic.json", "commute", "explore"),
        ("state-of-the-art claim", "real-mobility claim"),
    ),
    "privacy_utility_frontier": VisualExpectation(
        "Show what predictive utility survives each location-minimization regime, paired with the re-identification risk, so the governance tradeoff is visible.",
        "Per-regime surviving predictive utility (shuffle-corrected excess bits) against re-identification risk (event uniqueness), fine to coarse grids.",
        "A utility-versus-risk scatter with regimes connected fine-to-coarse and a withheld-location endpoint at the origin.",
        (
            "privacy_utility_frontier.json",
            "minimization",
            "re-identification",
            "minimum-collection",
        ),
        ("state-of-the-art claim", "real-mobility claim"),
    ),
    "privacy_utility_repeats": VisualExpectation(
        "Show repeated-draw privacy-utility stability only where source observations actually exist.",
        "Canonical fixture row plus explicit unavailable variance-realization rows.",
        "Status bars and utility-spread markers over completed and unavailable repeat rows.",
        (
            "privacy_utility_frontier_repeats.json",
            "unavailable",
            "repeated-draw",
            "stability",
        ),
        ("repeat-stability", "empirical privacy", "real-mobility"),
    ),
    "synthetic_task_utility_privacy": VisualExpectation(
        "Show whether any synthetic minimization regime clears utility and privacy thresholds together.",
        "Per-regime predictive utility, re-identification risk, co-travel, recoverability, and linkage-minimization status.",
        "Split utility (bits) and re-id risk (uniqueness) panels with joint release-utility gate labels.",
        (
            "synthetic_task_utility_privacy.json",
            "utility floor",
            "risk ceiling",
            "release-utility",
        ),
        ("operational usefulness", "privacy guarantees", "real-world utility"),
    ),
    "model_comparison_under_noise": VisualExpectation(
        "Show where active inference leads: rank every implemented predictor by fair loss under the noisy regime so the belief-marginalizing advantage is visible against the other methods.",
        "Per-method mean next-cell loss on the locked noisy-emission lane, grouped by how each method treats the noisy observation.",
        "A ranked horizontal bar chart of per-method loss with CI whiskers, coloured by method group, active inference marked at the top.",
        ("model_comparison_leaderboard.json", "noisy", "belief-marginalizing"),
        (
            "global state-of-the-art claim",
            "oracle-beating result",
            "real-mobility claim",
        ),
    ),
    "aif_deepening_candidates": VisualExpectation(
        "Separate scored active-inference candidates from audit-only surfaces under the locked predictive protocol.",
        "Rows for factorial tensor provenance, multiscale soft-belief backoff, and scored online adaptation.",
        "Candidate-status bars with protocol-entry and predictive-gate annotations.",
        (
            "aif_deepening_candidates.json",
            "Candidate.predict",
            "scored candidate",
            "predictive gate",
        ),
        ("predictive improvement", "SOTA", "operational claims"),
    ),
}


def audit_visual_claims(project_root: str | Path) -> dict[str, Any]:
    """Return a figure mini-brief audit over the generated registry and cover."""

    root = Path(project_root).resolve()
    registry_path = root / "output" / "figures" / "figure_registry.json"
    if not registry_path.is_file():
        raise VisualClaimAuditError("missing figure registry")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_schema = str(registry.get("schema", ""))
    entries = registry.get("entries", [])
    if not isinstance(entries, list) or not entries:
        raise VisualClaimAuditError("figure registry must contain entries")

    manuscript_blob = _manuscript_blob(root)
    violations: list[str] = []
    briefs: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for entry in entries:
        key = str(entry.get("key", ""))
        filename = str(entry.get("filename", ""))
        caption = str(entry.get("caption", ""))
        source = str(entry.get("source", ""))
        expectation = FIGURE_EXPECTATIONS.get(key)
        row_violations: list[str] = []
        if key in seen_keys:
            row_violations.append("duplicate key")
        seen_keys.add(key)
        if expectation is None:
            row_violations.append("missing visual expectation")
        if filename and filename not in manuscript_blob:
            row_violations.append("figure file is not referenced by manuscript source")
        has_support_contract = (
            "Supports " in caption and "; does not support " in caption
        )
        has_structured_contract = " Evidence: " in caption and " Boundary: " in caption
        if not (has_support_contract or has_structured_contract):
            row_violations.append("caption missing evidence/boundary contract")
        if source.startswith("output/") and not (root / source).exists():
            row_violations.append(f"source artifact missing: {source}")
        row_violations.extend(
            _registry_binding_violations(root, entry, registry_schema)
        )
        row_violations.extend(
            _source_conditioned_visual_violations(root, key, caption, source)
        )
        if expectation is not None:
            lower_caption = caption.lower()
            missing_expected = [
                term
                for term in expectation.expected_terms
                if term.lower() not in lower_caption
            ]
            missing_boundaries = [
                term
                for term in expectation.boundary_terms
                if term.lower() not in lower_caption
            ]
            if missing_expected:
                row_violations.append(
                    "caption missing expected term(s): " + ", ".join(missing_expected)
                )
            if missing_boundaries:
                row_violations.append(
                    "caption missing boundary term(s): " + ", ".join(missing_boundaries)
                )
            story_job = expectation.story_job
            data_shape = expectation.data_shape
            encoding = expectation.encoding
            reader_task = f"Use the visual to {expectation.story_job[0].lower()}{expectation.story_job[1:]}"
            claim_boundary = "; ".join(expectation.boundary_terms)
        else:
            story_job = data_shape = encoding = ""
            reader_task = claim_boundary = ""
        briefs.append(
            {
                "visual_layer": key,
                "filename": filename,
                "source": source,
                "story_job": story_job,
                "reader_task": reader_task,
                "data_shape": data_shape,
                "encoding_and_layout": encoding,
                "claim_boundary": claim_boundary,
                "interaction_or_fallback": "static manuscript PNG; claim must survive PDF export without hover",
                "accessibility": "caption and alt text carry source, claim, and non-claim boundary",
                "qa_check": "registry caption contract, source existence, manuscript call site, visual-quality audit, PDF render",
                "fresh_pass_status": "local RedTeam plus visualization-strategy mini-brief audit",
                "violations": row_violations,
            }
        )
        violations.extend(f"{key}: {item}" for item in row_violations)

    missing_expected_keys = sorted(set(FIGURE_EXPECTATIONS) - seen_keys)
    for key in missing_expected_keys:
        violations.append(f"{key}: expected figure key missing from registry")

    cover_row = _audit_cover(root)
    briefs.append(cover_row)
    violations.extend(f"cover: {item}" for item in cover_row["violations"])

    payload: dict[str, Any] = {
        "schema": VISUAL_CLAIM_AUDIT_SCHEMA,
        "status": "pass" if not violations else "fail",
        "figure_count": len(entries),
        "brief_count": len(briefs),
        "first_page_layers": ["cover", "evidence_atlas", "dataflow"],
        "violations": violations,
        "briefs": briefs,
    }
    return payload


def _registry_binding_violations(
    root: Path, entry: dict[str, Any], registry_schema: str
) -> list[str]:
    if registry_schema != "active-fractal-rabbit-figure-registry-v2":
        return []
    key = str(entry.get("key", ""))
    caption = str(entry.get("caption", ""))
    figure_path = root / "output" / "figures" / str(entry.get("filename", ""))
    source_path = root / str(entry.get("source", ""))
    image_sha = _sha256_file(figure_path) if figure_path.is_file() else None
    source_sha = _sha256_file(source_path) if source_path.is_file() else None
    violations: list[str] = []
    if entry.get("image_sha256") != image_sha:
        violations.append("image checksum differs from generated registry binding")
    if entry.get("source_sha256") != source_sha:
        violations.append("source checksum differs from generated registry binding")
    if image_sha is not None:
        digest = hashlib.sha256()
        digest.update(key.encode("utf-8"))
        digest.update(caption.encode("utf-8"))
        digest.update(image_sha.encode("ascii"))
        digest.update((source_sha or "non-file-source").encode("ascii"))
        if entry.get("binding_sha256") != digest.hexdigest():
            violations.append("figure/source/caption binding checksum mismatch")
    return violations


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_conditioned_visual_violations(
    root: Path, key: str, caption: str, source: str
) -> list[str]:
    if not source.startswith("output/"):
        return []
    source_path = root / source
    if not source_path.is_file():
        return []
    if source_path.suffix != ".json":
        return []
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"source artifact unreadable for conditioned audit: {exc}"]
    lower_caption = caption.lower()
    violations: list[str] = []
    if key == "regime_switch_phase_diagnostic":
        segments = payload.get("segments", [])
        segments = segments if isinstance(segments, list) else []
        negative = [
            str(row.get("segment", ""))
            for row in segments
            if isinstance(row, dict)
            and isinstance(row.get("clustered_ci_high"), int | float)
            and float(row["clustered_ci_high"]) < 0.0
        ]
        unresolved = [
            str(row.get("segment", ""))
            for row in segments
            if isinstance(row, dict)
            and isinstance(row.get("clustered_ci_low"), int | float)
            and isinstance(row.get("clustered_ci_high"), int | float)
            and float(row["clustered_ci_low"]) <= 0.0 <= float(row["clustered_ci_high"])
        ]
        if any(name and name.lower() not in lower_caption for name in negative + unresolved):
            violations.append("caption omits a source phase-segment status")
        if unresolved and (
            "all three" in lower_caption and "excluding zero" in lower_caption
        ):
            violations.append(
                "caption claims all phase intervals exclude zero although a source interval crosses zero"
            )
        if unresolved and "unresolved" not in lower_caption:
            violations.append("caption omits unresolved wording for a zero-crossing phase interval")
        uniform = payload.get("uniform_disadvantage")
        if uniform is False and "uniform disadvantage: false" not in lower_caption:
            violations.append("caption contradicts or omits the non-uniform source verdict")
        ratio = payload.get("worst_to_best_magnitude_ratio")
        if ratio is None and not any(
            term in lower_caption for term in ("ratio: n/a", "ratio unavailable")
        ):
            violations.append("caption implies a worst-to-best ratio although source ratio is null")
        return violations
    if key == "privacy_utility_repeats":
        if payload.get("can_claim_graceful_repeated_tradeoff") is False:
            stale_terms = (
                "stability claim clears",
                "repeat-stability wording is allowed",
                "graceful repeated tradeoff clears",
            )
            if any(term in lower_caption for term in stale_terms):
                violations.append(
                    "caption clears repeated privacy-utility stability although source gate is false"
                )
            if "blocked" not in lower_caption and "unavailable" not in lower_caption:
                violations.append(
                    "caption missing blocked or unavailable wording for failed repeated privacy-utility gate"
                )
        if (
            int(payload.get("unavailable_rows", 0) or 0) > 0
            and "unavailable" not in lower_caption
        ):
            violations.append(
                "caption missing unavailable-row wording for repeated privacy-utility artifact"
            )
        return violations
    if key == "synthetic_task_utility_privacy":
        gate = payload.get("any_release_utility_claim_allowed")
        if gate is False:
            if (
                "release utility allowed" in lower_caption
                or "release-utility allowed" in lower_caption
            ):
                violations.append(
                    "caption allows release utility although source joint utility/privacy gate is false"
                )
            if not any(
                term in lower_caption
                for term in ("denies", "blocked", "unless both thresholds clear")
            ):
                violations.append(
                    "caption missing blocked release-utility wording for failed joint utility/privacy gate"
                )
        return violations
    if key == "aif_deepening_candidates":
        gate = payload.get("predictive_gate_clears")
        if gate is False:
            stale_terms = (
                "predictive gate cleared",
                "predictive improvement clears",
                "supports predictive improvement",
                "unlocks predictive promotion",
            )
            if any(term in lower_caption for term in stale_terms):
                violations.append(
                    "caption clears AIF deepening predictive gate although source gate is false"
                )
            if "audit-only" not in lower_caption:
                violations.append(
                    "caption missing audit-only wording for non-predictive AIF deepening rows"
                )
        return violations
    if key != "privacy_utility_frontier":
        return []
    if payload.get("graceful_degradation_tradeoff") is True:
        stale_failure_terms = (
            "not monotone",
            "non-monotone",
            "fails the graceful-degradation gate",
            "not a graceful-degradation claim",
        )
        if any(term in lower_caption for term in stale_failure_terms):
            violations.append(
                "caption reports failed or non-monotone privacy-utility gate although source gate is true"
            )
        if "graceful" not in lower_caption or "monotone" not in lower_caption:
            violations.append(
                "caption missing source-cleared graceful-degradation wording"
            )
    elif payload.get("graceful_degradation_tradeoff") is False:
        if (
            "fall together" in lower_caption
            or "graceful-degradation governance tradeoff" in lower_caption
        ):
            violations.append(
                "caption claims graceful degradation although source gate is false"
            )
        if "not monotone" not in lower_caption and "non-monotone" not in lower_caption:
            violations.append(
                "caption missing non-monotone wording for failed graceful-degradation gate"
            )
    min_collection = payload.get("min_collection_operating_point", {})
    if isinstance(min_collection, dict) and min_collection.get("found") is False:
        if "no minimum-collection operating point" not in lower_caption:
            violations.append(
                "caption missing no minimum-collection operating-point null"
            )
    if isinstance(min_collection, dict) and min_collection.get("found") is True:
        if "no minimum-collection operating point" in lower_caption:
            violations.append(
                "caption reports no minimum-collection operating point although source found one"
            )
        if "minimum-collection operating point" not in lower_caption:
            violations.append(
                "caption missing source-found minimum-collection operating point"
            )
    return violations


def write_visual_claim_audit(
    project_root: str | Path,
    *,
    out_path: str | Path = "output/runs/visual_claim_audit.json",
    fail_on_violations: bool = True,
) -> Path:
    """Write visual mini-brief audit JSON and optionally fail on violations."""

    root = Path(project_root).resolve()
    payload = audit_visual_claims(root)
    output = root / out_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if fail_on_violations and payload["violations"]:
        raise VisualClaimAuditError("; ".join(payload["violations"]))
    return output


def _audit_cover(root: Path) -> dict[str, Any]:
    path = root / "output" / "cover" / "active_fractal_rabbit_cover.png"
    required_sources = (
        "output/runs/default/metrics.json",
        "output/runs/spatial_resolution_frontier.json",
        "output/runs/forward_gates_summary.json",
        "output/runs/efe_location_frontier.json",
        "output/runs/linkage_risk_frontier.json",
        "output/runs/poisoning_robustness.json",
    )
    violations: list[str] = []
    if not path.is_file():
        violations.append("cover image missing")
    elif path.stat().st_size < 1_000:
        violations.append("cover image too small to be a rendered artifact")
    for source in required_sources:
        if not (root / source).is_file():
            violations.append(f"cover source artifact missing: {source}")
    return {
        "visual_layer": "cover",
        "filename": "active_fractal_rabbit_cover.png",
        "source": ", ".join(required_sources),
        "story_job": "Present a data-bound evidence atlas before the manuscript text.",
        "reader_task": "Use the cover to identify which model family fits each synthetic information regime before reading the manuscript text.",
        "data_shape": "Default-run metrics, grid occupancy, gate summaries, and safety-boundary verdicts.",
        "encoding_and_layout": "Editorial cover with a synthetic waypoint lattice, three regime-qualified model cards, an evidence-boundary strip, and artifact metrics.",
        "claim_boundary": "decorative context only; manuscript text and registered captions carry source-bound claims",
        "interaction_or_fallback": "static PDF cover image generated from local artifacts",
        "accessibility": "cover image is decorative-contextual; manuscript abstract and captions carry the evidence text",
        "qa_check": "cover-art tests, source existence, nonblank image check, PDF render",
        "fresh_pass_status": "local RedTeam plus visualization-strategy mini-brief audit",
        "violations": violations,
    }


def _manuscript_blob(root: Path) -> str:
    parts = []
    for path in sorted((root / "manuscript").glob("*.md")):
        if path.name == "AGENTS.md":
            continue
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)
