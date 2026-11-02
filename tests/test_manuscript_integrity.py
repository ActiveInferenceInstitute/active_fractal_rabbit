from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest
import yaml

from active_fractal_rabbit.circadian_phase import run_and_write_circadian_phase
from active_fractal_rabbit.figures import generate_figures
from active_fractal_rabbit.forward_gates import run_and_write_forward_gates
from active_fractal_rabbit.fractal_dimension import (
    run_and_write_fractal_dimension_sweep,
)
from active_fractal_rabbit.hardened_anomaly import run_and_write_hardened_anomaly
from active_fractal_rabbit.manuscript_variables import compute_variables
from active_fractal_rabbit.model_gates import run_and_write_model_gates
from active_fractal_rabbit.model_spec import run_and_write_model_conformance
from active_fractal_rabbit.negative_controls import run_negative_controls
from active_fractal_rabbit.privacy_frontier import run_and_write_privacy_frontier
from active_fractal_rabbit.pymdp_benchmark import write_pymdp_benchmark_matrix
from active_fractal_rabbit.return_structure import run_and_write_return_structure
from active_fractal_rabbit.robustness import (
    run_and_write_upstream_sensitivity,
    write_variance_bands,
)
from active_fractal_rabbit.sparsity import run_sparsity_analysis
from active_fractal_rabbit.pipeline import run_pipeline_from_path
from active_fractal_rabbit.sweeps import run_replicated_sensitivity

ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
HARDCODED_REF_RE = re.compile(r"\b(Figure|Equation|Table)\s+\d+\b")
CITATION_RE = re.compile(r"@([A-Za-z][A-Za-z0-9_:-]*)")
BIB_KEY_RE = re.compile(r"@\w+\{([^,]+),")
FIGURE_PATH_RE = re.compile(r"\]\(\.\./output/figures/([^)]+\.png)\)")
XREF_RE = re.compile(r"\[@(fig|eq|tbl|sec):([A-Za-z0-9_-]+)\]")
LABEL_RE = re.compile(r"\{#(fig|eq|tbl|sec):([A-Za-z0-9_-]+)(?:\s[^}]*)?\}")
RAW_EQUATION_COUNTER_RE = re.compile(
    r"\\(?:begin\{equation\}|end\{equation\}|label\{|tag\{)"
)
DISPLAY_EQUATION_LABEL_RE = re.compile(r"\{#eq:[A-Za-z0-9_-]+(?:\s[^}]*)?\}")
# A genuine raw display-math delimiter is a lone backslash directly before `[`/`]`. Exclude the
# doubled-backslash LaTeX row-break-with-spacing token (`\\[4pt]`, legitimate inside a `cases`
# environment) via the negative lookbehind.
RAW_BRACKET_DISPLAY_MATH_RE = re.compile(r"(?<!\\)\\[\[\]]")

#: Structural constants allowed in manuscript prose: JDK 21, SHA-256, top-1, and
#: the 95% CI convention (percentile(2.5, 97.5) is a fixed code constant in
#: power_analysis.traveller_clustered_bootstrap_stats, independent of the
#: family-level MULTIPLICITY_ALPHA).
STRUCTURAL_NUMERALS = frozenset({"21", "256", "1", "95"})

#: Repository URLs and DOIs are publication identifiers, not run-dependent
#: numbers, so they are stripped before prose-numeral scans. The scoping stays
#: honest because test_publication_identifier_scoping_keeps_prose_drift_failing
#: proves other numerals still fail, and
#: test_abstract_publication_identifiers_match_configured_doi pins the
#: abstract's identifiers to manuscript/config.yaml.
PUBLICATION_IDENTIFIER_PATTERNS = (
    re.compile(r"https?://\S+"),
    re.compile(r"10\.\d{4,}/\S+"),
)
NUMERAL_RE = re.compile(r"\b\d+(?:\.\d+)?(?:e-?\d+)?\b")
ABSTRACT_FORBIDDEN_PATTERNS = {
    "citation": r"\[@",
    "figure reference": r"\bfig(?:ure)?[.:]",
    "manuscript token": r"\{\{[A-Z][A-Z0-9_]*\}\}",
    "code formatting": r"`",
    "numeric quantity": r"\b\d+(?:\.\d+)?(?:e-?\d+)?\b",
}
ABSTRACT_ENVIRONMENT_TERMS = ("java", "maven", "toolchain", "jdk")


def _strip_publication_identifiers(text: str) -> str:
    """Remove publication identifiers (URLs, DOIs) before prose-numeral scans."""

    for pattern in PUBLICATION_IDENTIFIER_PATTERNS:
        text = pattern.sub("", text)
    return text


def _abstract_offenders(body: str) -> list[str]:
    """Forbidden plain-text element names present in a lowercased abstract body."""

    scan_body = _strip_publication_identifiers(body)
    return [
        name
        for name, pattern in ABSTRACT_FORBIDDEN_PATTERNS.items()
        if re.search(pattern, scan_body)
    ]


def _numeral_scan(text: str) -> tuple[str, list[re.Match[str]]]:
    """Return (scan_text, offenders) for run-dependent numerals left in prose."""

    scan_text = _strip_publication_identifiers(text)
    offenders = [
        match
        for match in NUMERAL_RE.finditer(scan_text)
        if match.group(0) not in STRUCTURAL_NUMERALS
    ]
    return scan_text, offenders


def _prepare_default_outputs(root: Path) -> None:
    """Regenerate the default artifact surface inside ``root`` (a repo_sandbox).

    Must never be pointed at the production repo: several writers below are
    smoke-scale or fixture-valued (variance bands, fractal-dimension stub
    runner, upstream-sensitivity no-op) and would clobber canonical artifacts.
    Callers pass the ``repo_sandbox`` fixture, which also chdirs into ``root``
    so the relative-path writers land in the sandbox output tree.
    """
    result = run_pipeline_from_path("configs/default.yaml")
    run_replicated_sensitivity("configs/default.yaml")
    run_negative_controls("configs/default.yaml")
    run_sparsity_analysis("configs/default.yaml")
    missing_external = root / "output" / "runs" / "does_not_exist" / "observations.csv"
    run_and_write_return_structure(
        "configs/default.yaml",
        missing_external,
        "output/runs/return_structure_analysis.json",
    )
    run_and_write_privacy_frontier(
        "configs/default.yaml",
        missing_external,
        "output/runs/privacy_frontier_analysis.json",
    )
    run_and_write_hardened_anomaly(
        "configs/default.yaml",
        missing_external,
        "output/runs/hardened_anomaly_analysis.json",
    )
    run_and_write_circadian_phase(
        "configs/default.yaml",
        missing_external,
        "output/runs/circadian_phase_analysis.json",
    )
    write_variance_bands(
        "output/runs/variance_bands.json",
        [
            {
                "gap_filling_skill": 0.1,
                "blind_limit_skill": 0.0,
                "anomaly_random_auc": 0.7,
                "frontier_no_location_mass": 0.2,
                "structure_deep_latent_tv": 0.3,
                "mission_efe_minus_pragmatic": 0.4,
            },
            {
                "gap_filling_skill": 0.14,
                "blind_limit_skill": -0.02,
                "anomaly_random_auc": 0.74,
                "frontier_no_location_mass": 0.24,
                "structure_deep_latent_tv": 0.26,
                "mission_efe_minus_pragmatic": 0.36,
            },
        ],
    )
    run_and_write_upstream_sensitivity(
        "output/runs/upstream_sensitivity.json",
        current_commit="9933449c4f4fe1b26b6ac7bfdeeac76583085df5",
        candidate_limit=1,
        candidate_provider=lambda limit: ["candidate"][:limit],
        runner=lambda sha: None,
    )
    run_and_write_fractal_dimension_sweep(
        "configs/default.yaml",
        "output/runs/fractal_dimension_sweep.json",
        dimensions=[1.1, 1.33],
        runner=lambda dimension: {
            "n_observations": 2400,
            "n_latent_states": 16,
            "mean_predictive_loss": 2.0 + dimension,
            "mean_posterior_entropy": 0.5,
            "state_recovery_reference_mean_mass_on_true_cell": 0.8,
            "place_revisit_rate": 0.2 + dimension / 10.0,
            "place_mean_reports_per_place": 3.0,
        },
    )
    run_and_write_model_gates(
        project_root=root,
        fixture_config="configs/default.yaml",
        external_config="configs/fractalrabbit_external.yaml",
        external_ready=False,
    )
    run_and_write_model_conformance(
        "configs/default.yaml", "output/runs/model_conformance.json"
    )
    write_pymdp_benchmark_matrix(
        root, root / "output" / "runs" / "pymdp_benchmark_matrix.json"
    )
    run_and_write_forward_gates(
        project_root=root,
        fixture_config="configs/default.yaml",
        external_config="configs/fractalrabbit_external.yaml",
        external_ready=False,
    )
    generate_figures(result.output_dir)


def _manuscript_texts() -> list[str]:
    excluded = {"AGENTS.md", "README.md", "SYNTAX.md"}
    return [
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "manuscript").glob("*.md"))
        if path.name not in excluded
    ]


def _active_formalism_paths() -> list[Path]:
    excluded = {"AGENTS.md", "README.md", "SYNTAX.md"}
    return [
        *[
            path
            for path in sorted((ROOT / "manuscript").glob("*.md"))
            if path.name not in excluded
        ],
        *sorted((ROOT / "docs" / "methods").glob("*.md")),
    ]


def _unlabelled_display_equations(path: Path) -> list[str]:
    offenders: list[str] = []
    in_block = False
    start_line = 0
    has_label = False
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        marker_count = stripped.count("$$")
        if marker_count == 0:
            if in_block and DISPLAY_EQUATION_LABEL_RE.search(line):
                has_label = True
            continue
        if marker_count >= 2 and not in_block:
            if not DISPLAY_EQUATION_LABEL_RE.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{line_number}")
            continue
        if not in_block:
            in_block = True
            start_line = line_number
            has_label = bool(DISPLAY_EQUATION_LABEL_RE.search(line))
            continue
        has_label = has_label or bool(DISPLAY_EQUATION_LABEL_RE.search(line))
        if not has_label:
            offenders.append(f"{path.relative_to(ROOT)}:{start_line}-{line_number}")
        in_block = False
    if in_block:
        offenders.append(f"{path.relative_to(ROOT)}:{start_line}-EOF")
    return offenders


def test_manuscript_tokens_citations_and_figure_paths_resolve(repo_sandbox) -> None:
    _prepare_default_outputs(repo_sandbox)
    variables = compute_variables(repo_sandbox)
    manuscript = "\n".join(_manuscript_texts())
    missing_tokens = set(TOKEN_RE.findall(manuscript)) - set(variables)

    assert not missing_tokens
    assert not HARDCODED_REF_RE.search(manuscript)
    for figure_name in FIGURE_PATH_RE.findall(manuscript):
        assert (repo_sandbox / "output" / "figures" / figure_name).exists()

    bib_keys = set(
        BIB_KEY_RE.findall(
            (ROOT / "manuscript" / "references.bib").read_text(encoding="utf-8")
        )
    )
    citation_keys = {
        key
        for key in CITATION_RE.findall(manuscript)
        if not key.startswith(("fig:", "eq:", "tbl:", "sec:"))
    }
    assert citation_keys <= bib_keys
    required = {
        "barbosa2018human_mobility",
        "deng2025mirage",
        "smith2016software_citation",
        "gonzalez2008mobility",
        "song2010scaling",
        "song2010predictability",
        "nathan2008movement_ecology",
        "dacosta2020discrete_active_inference",
        "friston2010fep",
        "friston2022simpler",
        "shannon1948communication",
        "cover2006elements_information",
        "brier1950forecasts",
        "guo2017calibration",
        "dawid1984prequential",
        "kim2016burstiness_finite",
        "millidge2020efe",
        "sajid2021active_inference_design",
        "champion2024reframing_efe",
        "pimentel2014novelty_detection",
        "barabasi2005bursts",
        "sandve2013reproducible",
        "wilson2017good_enough",
        "demontjoye2013unique",
        "kondor2018trace_matching",
        "dwork2006differential_privacy",
        "kaelbling1998pomdp",
        "patterson2008state_space_movement",
        "patterson2017movement_models",
        "zucchini2016hmm_time_series",
        "doucet2001smc",
        "arulampalam2002particle_filters",
        "jordan1999variational_methods",
        "blei2017variational_inference",
        "luca2021deep_learning_mobility",
        "yan2017universal_mobility",
        "wu2024pmt",
        "gao2022ssdl_mobility",
        "shi2026sparse_trajectory_prediction",
        "han2025movegpt",
        "tang2024llama_mob",
        "fang2025mobert",
        "sweeney2026formal_equivalences",
        "maisto2025active_inference_tree_search",
        "nguyen2024raif",
    }
    assert required <= citation_keys
    assert "??" not in manuscript


def test_manuscript_cross_references_resolve_and_equations_are_autonumbered() -> None:
    manuscript = "\n".join(_manuscript_texts())
    labels = {f"{kind}:{name}" for kind, name in LABEL_RE.findall(manuscript)}
    references = {f"{kind}:{name}" for kind, name in XREF_RE.findall(manuscript)}

    assert references <= labels
    assert not RAW_EQUATION_COUNTER_RE.search(manuscript)

    figure_labels = {label for label in labels if label.startswith("fig:")}
    figure_refs = {ref for ref in references if ref.startswith("fig:")}
    assert figure_labels <= figure_refs


def test_active_formalisms_are_display_labelled_for_auto_numbering() -> None:
    offenders: list[str] = []
    bracket_offenders: list[str] = []
    raw_latex_offenders: list[str] = []
    for path in _active_formalism_paths():
        text = path.read_text(encoding="utf-8")
        offenders.extend(_unlabelled_display_equations(path))
        # A genuine raw display-math delimiter is a single backslash directly before `[`/`]`.
        # `\\[4pt]` (LaTeX row-break with spacing, e.g. inside a `cases` environment) is a
        # legitimate, differently-shaped token: two backslashes then a bracket. Excluding it
        # avoids flagging real labelled-equation cases blocks as raw bracket display math
        # (round-2 review, 2026-07-06: S03's piecewise speed formula tripped this false positive).
        if RAW_BRACKET_DISPLAY_MATH_RE.search(text):
            bracket_offenders.append(str(path.relative_to(ROOT)))
        if RAW_EQUATION_COUNTER_RE.search(text):
            raw_latex_offenders.append(str(path.relative_to(ROOT)))

    assert not offenders, (
        f"display equation blocks need {{#eq:...}} labels: {offenders}"
    )
    assert not bracket_offenders, (
        f"use labelled $$ blocks instead of raw bracket display math: {bracket_offenders}"
    )
    assert not raw_latex_offenders, (
        f"use pandoc-crossref labels, not raw LaTeX counters: {raw_latex_offenders}"
    )


def test_manuscript_and_docs_avoid_known_review_regressions() -> None:
    corpus_paths = [
        *sorted((ROOT / "manuscript").glob("*.md")),
        *sorted((ROOT / "docs").glob("*.md")),
        ROOT / "README.md",
    ]
    corpus = "\n".join(
        path.read_text(encoding="utf-8")
        for path in corpus_paths
        if path.name not in {"AGENTS.md", "SYNTAX.md"}
    )
    bib = (ROOT / "manuscript" / "references.bib").read_text(encoding="utf-8")

    assert "HMM-style" not in corpus
    assert "95% CI" not in corpus
    assert "expected-free-energy-inspired" not in corpus
    assert "combining ambiguity" not in corpus
    assert "combining projected predictive entropy with divergence" not in corpus
    assert "predictive-entropy plus preference-divergence proxy" not in corpus
    assert "no location-blind method can win" not in corpus
    assert "cannot match persistence's oracle access" not in corpus
    assert "near-unbeatable" not in corpus
    assert "B_t (t_i" not in corpus
    assert "preference cross-entropy proxy" in (
        ROOT / "manuscript" / "02b_formal_model.md"
    ).read_text(encoding="utf-8")
    assert "Identity-emission collapse" in (
        ROOT / "manuscript" / "02b_formal_model.md"
    ).read_text(encoding="utf-8")
    assert "set-valued signature is intentional" in (
        ROOT / "manuscript" / "S04_formal_diagnostics_and_pymdp.md"
    ).read_text(encoding="utf-8")
    assert "scripts/z_generate_manuscript_variables.py" in (
        ROOT / "manuscript" / "S02_reproducibility.md"
    ).read_text(encoding="utf-8")
    assert "/master/resources/" not in bib
    assert "/master/src/" not in bib
    stale_privacy_utility_phrases = {
        "falls in step",
        "minimization is a graceful-degradation tradeoff",
        "privacy-utility frontier reports a synthetic graceful-degradation tradeoff",
        "a graceful-degradation tradeoff rather than a cliff",
    }
    lower_corpus = corpus.lower()
    offenders = sorted(
        phrase for phrase in stale_privacy_utility_phrases if phrase in lower_corpus
    )
    assert not offenders, f"stale privacy-utility prose contradiction(s): {offenders}"
    overclaim_paths = [
        path
        for path in corpus_paths
        if path.relative_to(ROOT).as_posix() != "docs/scholarship_ledger.md"
    ]
    overclaim_corpus = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in overclaim_paths
        if path.name not in {"AGENTS.md", "SYNTAX.md"}
    )
    bayesian_mechanics_overclaims = {
        "real cognitive particle",
        "bayesian mechanics proves",
        "fep validates sparse-waypoint claims",
        "ness system",
        "living-system claim",
        "particular physics proves",
        "waypoint trace is a strange particle",
        "fep applies to any sparse-waypoint system",
        "stochastic thermodynamics validates",
        "waypoint fixture is autonomous life",
    }
    offenders = sorted(
        phrase for phrase in bayesian_mechanics_overclaims if phrase in overclaim_corpus
    )
    assert not offenders, f"Bayesian-mechanics overclaim phrase(s): {offenders}"
    s08 = (
        (ROOT / "manuscript" / "S08_results_safety_privacy_and_external.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "full-grid ladder is monotone" in s08
    assert "minimum-collection operating point" in s08
    assert "source-positive" in s08
    assert "threshold failure" in s08


def test_active_surfaces_do_not_reintroduce_legacy_method_language() -> None:
    paths = [
        ROOT / "README.md",
        *sorted(
            path
            for path in (ROOT / "docs").rglob("*.md")
            if "revision" not in path.parts
        ),
        *sorted(
            path
            for path in (ROOT / "manuscript").glob("*.md")
            if path.name
            not in {"AGENTS.md", "README.md", "SYNTAX.md", "references.bib"}
        ),
        *sorted((ROOT / "src" / "active_fractal_rabbit").glob("*.py")),
    ]
    forbidden = (
        "legacy",
        "former top-level",
        "older inference",
        "older matrix",
        "legacy sota",
        "fixture fallback",
        "external run (fixture fallback)",
        "obsolete",
        "deprecated",
    )
    offenders = [
        f"{path.relative_to(ROOT)}: {term}"
        for path in paths
        for term in forbidden
        if term in path.read_text(encoding="utf-8").lower()
    ]

    assert not offenders


def test_s01_execution_only_enumeration_matches_eval_contract_constant() -> None:
    """S01's disclosure sentence must enumerate every EXECUTION_ONLY_CLASSES entry, not a stale
    subset (round-2 review, 2026-07-06, found the manuscript listing 4 of 5 classes after a 5th
    was added to the code constant with no matching prose update) — pin the COUNT so the two can't
    silently re-diverge again the next time EXECUTION_ONLY_CLASSES grows."""
    from active_fractal_rabbit.eval_contract import EXECUTION_ONLY_CLASSES

    s01 = (ROOT / "manuscript" / "S01_validation_and_threat_model.md").read_text(
        encoding="utf-8"
    )
    # Distinctive, tense-robust fragments per class (manuscript prose paraphrases the code
    # constant's present-tense docstring-style wording into past-tense narrative prose).
    distinctive_fragments = (
        "own baseline",
        "reproduce on real data",
        "fabricated library call",
        "full re-scoring",
        "render-readiness",
    )
    assert len(distinctive_fragments) == len(EXECUTION_ONLY_CLASSES), (
        "EXECUTION_ONLY_CLASSES changed size — add/remove a matching distinctive_fragment here "
        "AND update the S01 enumeration sentence before this test can be trusted again"
    )
    missing = [frag for frag in distinctive_fragments if frag not in s01.lower()]
    assert not missing, f"S01 is missing execution-only class fragment(s): {missing}"


def test_manuscript_title_margin_and_new_section_references() -> None:
    config = yaml.safe_load(
        (ROOT / "manuscript" / "config.yaml").read_text(encoding="utf-8")
    )
    manuscript = "\n".join(_manuscript_texts())
    section_titles = {
        "01_introduction.md": "# Introduction: Sparse Waypoints Need Belief Filters and Claim Guards {#sec:introduction}",
        "02_methodology.md": "# Methods: Artifact-First POMDP Benchmarking on Synthetic Waypoints {#sec:methodology}",
        "02b_formal_model.md": "## Formal Model: Belief Filtering Under Bounded Policy Proxies {#sec:formal-model}",
        "03_results.md": "# Results: Belief Helps Only When Emissions Are Noisy {#sec:results}",
        "03a_results_partial_observability_and_model_comparison.md": "## Noisy Emissions Reward Belief Marginalization, Not Point Estimates {#sec:results-model-comparison-fixture}",
        "03b_results_null_hierarchy_and_multiplicity.md": "## Nulls, Calibration, and Multiplicity Keep Gains Bounded {#sec:results-null-hierarchy-program-extensions}",
        "04_discussion.md": "# Discussion: Belief Filtering Earns a Regime-Bound Role {#sec:discussion}",
        "05_limitations.md": "# Limitations: Synthetic Evidence Does Not Promote Real-Trace Claims {#sec:limitations}",
        "07_roadmap.md": "# Roadmap: Promotion Stays Blocked by Evidence Gates {#sec:roadmap}",
        "08_conclusion.md": "# Conclusion: A Synthetic Benchmark Can Clarify Without Promoting {#sec:conclusion}",
    }
    stale_titles = {
        "Introduction: A Synthetic Benchmark for Sparse Waypoint Inference",
        "Introduction: Synthetic Benchmarking, Variational Diagnostics, and Claim Boundaries",
        "Methods {#sec:methodology}",
        "Methods: The Artifact-Bound Active-Inference Loop",
        "Formal Model: POMDP Belief Filtering and Bounded Policy Proxy",
        "Generative Model and Partial-Observability Protocol",
        "Results {#sec:results}",
        "Results: The Positive Regime and Model Comparison Under Partial Observability",
        "Belief Marginalization Helps Under Noisy Partial Observability",
        "Nulls, Calibration, and Multiplicity {#sec:results-null-hierarchy-program-extensions}",
        "Results: The Null Hierarchy, Calibration, and Multiplicity Accounting",
        "Discussion: A Regime-Bound Belief-Filtering Result",
        "Discussion: Where Belief Filtering Earns Its Keep",
        "Limitations: Synthetic Evidence and Claim Boundaries",
        "Limitations: Synthetic-Only Evidence, Oracle Asymmetry",
        "Roadmap: Gates That Still Block Promotion",
        "Roadmap: Forward Gates After Current Evidence Boundaries",
        "Conclusion: What the Benchmark Establishes",
        "Conclusion: The Honest-Benchmark Discipline",
    }
    labels = {
        "sec:fractal-dimension-control",
        "sec:pymdp-performance-interpretability",
        "sec:results-sequence-reporting-gates",
        "sec:results-model-gate-matrix",
        "sec:results-forward-gates",
        "sec:results-location-risk-integrity",
        "sec:factorial-pymdp-latent-modes",
        "sec:discussion-bayesian-mechanics",
        "sec:roadmap-implemented-now",
        "sec:roadmap-sota-gates",
        "sec:conclusion-pymdp-now",
    }

    assert (
        config["paper"]["title"]
        == "Active FractalRabbit: A Synthetic Benchmark for Belief Filtering Under Sparse Waypoint Observations"
    )
    assert config["paper"]["date"] == ""
    assert (
        config["paper"]["cover"]["image"]
        == "output/cover/active_fractal_rabbit_cover.png"
    )
    assert config["authors"][0]["orcid"] == "0000-0001-6232-9096"
    assert config["publication"]["doi_status"] == "published"
    assert config["publication"]["doi"] == "10.5281/zenodo.21330636"
    assert config["publication"]["version_doi"] == "10.5281/zenodo.21330637"
    assert (
        config["metadata"]["geometry"]
        == "left=0.12in,right=0.12in,top=0.28in,bottom=0.28in"
    )
    assert config["metadata"]["documentclass"] == "extarticle"
    assert config["metadata"]["fontsize"] == "8pt"
    for filename, title in section_titles.items():
        assert (ROOT / "manuscript" / filename).read_text(
            encoding="utf-8"
        ).splitlines()[0] == title
    offenders = sorted(title for title in stale_titles if title in manuscript)
    assert offenders == []
    for label in labels:
        assert f"{{#{label}}}" in manuscript
        assert f"[@{label}]" in manuscript


def test_resolved_manuscript_has_no_unresolved_tokens_or_refs(repo_sandbox) -> None:
    # compute_variables + the resolved tree are the surface under test.
    # write_variables is deliberately NOT called here: its provenance side
    # effect requires the full 40-artifact surface (exercised by the sandboxed
    # figures/release tests and by the full study), not this render check.
    template_root = Path(
        os.environ.get("TEMPLATE_REPO_ROOT", ROOT.parents[2] / "template")
    )
    injection = (
        template_root / "infrastructure" / "rendering" / "manuscript_injection.py"
    )
    if not injection.is_file():
        pytest.skip(
            "manuscript render toolchain not present: missing "
            f"{injection}; set $TEMPLATE_REPO_ROOT or check out the sibling "
            "`template` repository (removal condition: render toolchain "
            "available; owner: render lane)"
        )
    _prepare_default_outputs(repo_sandbox)
    variables = compute_variables(repo_sandbox)
    if str(template_root) not in sys.path:
        sys.path.insert(0, str(template_root))
    from infrastructure.rendering.manuscript_injection import (
        write_resolved_manuscript_tree,
    )

    resolved = write_resolved_manuscript_tree(repo_sandbox, variables)
    rendered_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(Path(resolved).glob("*.md"))
        if path.name not in {"AGENTS.md", "README.md", "SYNTAX.md"}
    )
    assert not TOKEN_RE.search(rendered_text)
    assert "??" not in rendered_text


def test_abstract_is_problem_framed_not_environment_framed() -> None:
    """The abstract should lead with analytic purpose, not package/toolchain mechanics."""

    abstract = (ROOT / "manuscript" / "00_abstract.md").read_text(encoding="utf-8")
    body_lines = [
        line.strip()
        for line in abstract.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    body = "\n".join(body_lines).lower()
    assert len(body_lines) == 1, "abstract must be exactly one plain-text paragraph"
    offenders = _abstract_offenders(body)
    assert not offenders, (
        f"abstract contains forbidden plain-text element(s): {offenders}"
    )
    required_phrases = {
        "sparse waypoint",
        "privacy-sensitive",
        "open-source software",
        "national security agency",
        "synthetic fractalrabbit",
        "active inference",
        "partially observable markov decision process (pomdp)",
        "noisy partial-observability",
        "minimization",
        "integrity",
        "lowest-loss implemented predictor",
        "statistical tie",
        "point estimates suffice",
        "regime-specific synthetic model map",
        "separate evidence protocols",
    }
    missing = sorted(phrase for phrase in required_phrases if phrase not in body)
    assert not missing, f"abstract missing problem-framing phrase(s): {missing}"
    offenders = sorted(term for term in ABSTRACT_ENVIRONMENT_TERMS if term in body)
    assert not offenders, f"abstract drifted back into environment framing: {offenders}"
    assert "deterministic project-local synthetic fractalrabbit-format fixture" in body
    assert "separately retained lane" in body
    assert "independent simulator surface" in body
    assert (
        "synthetic fractalrabbit data generated by open-source software from the national security agency"
        not in body
    )


def test_conclusion_all_positive_claim_excludes_clean_zero_noise_level() -> None:
    conclusion = (ROOT / "manuscript" / "08_conclusion.md").read_text(
        encoding="utf-8"
    )
    assert "positive at every nonzero noise level" in conclusion
    assert "per-level resamplings positive" not in conclusion


def test_validation_and_reproducibility_are_supplemental() -> None:
    """Reproducibility and validation/threat-model material must live in S-prefixed
    supplemental sections (rendered after the main body, before references), not in the
    numbered main IMRaD flow. Anchors are preserved so cross-references still resolve."""
    md = ROOT / "manuscript"
    supplemental = {p.name for p in md.glob("S*.md")}
    assert "S01_validation_and_threat_model.md" in supplemental
    assert "S02_reproducibility.md" in supplemental
    assert "{#sec:validation}" in (md / "S01_validation_and_threat_model.md").read_text(
        encoding="utf-8"
    )
    assert "{#sec:reproducibility}" in (md / "S02_reproducibility.md").read_text(
        encoding="utf-8"
    )
    # no numbered main-body section may host the validation/reproducibility anchors (anti-drift)
    for section in md.glob("[0-9][0-9]_*.md"):
        body = section.read_text(encoding="utf-8")
        assert "{#sec:validation}" not in body, (
            f"{section.name} re-hosts the validation section"
        )
        assert "{#sec:reproducibility}" not in body, (
            f"{section.name} re-hosts the reproducibility section"
        )


def test_confound_taxonomy_is_supplemental_not_main_discussion() -> None:
    discussion = (ROOT / "manuscript" / "04_discussion.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "manuscript" / "07_roadmap.md").read_text(encoding="utf-8")
    supplement = (ROOT / "manuscript" / "S09_confound_taxonomy.md").read_text(
        encoding="utf-8"
    )
    assert (
        "The Confound Map: Why Naive Active-Inference Wins Dissolve" not in discussion
    )
    assert "confound_taxonomy.png" not in discussion
    assert "Confound Controls Decide Which AIF Models Enter" not in roadmap
    assert "{#sec:roadmap-confound-battery}" not in roadmap
    assert "[@sec:supplement-confound-taxonomy]" in discussion
    assert "{#sec:supplement-confound-taxonomy}" in supplement
    assert "confound_taxonomy.png" in supplement
    assert "#fig:confound-taxonomy" in supplement


def test_discussion_model_map_states_where_models_work_and_why() -> None:
    discussion = (ROOT / "manuscript" / "04_discussion.md").read_text(encoding="utf-8")
    assert "{#sec:discussion-model-map}" in discussion
    for regime in (
        "Clean or directly observed cell",
        "Noisy emission over a hidden cell",
        "Latent regime switching",
        "Sparse reporting gaps",
        "Location withheld, metadata retained",
    ):
        assert regime in discussion
    assert "there is no global winner" in discussion.lower()
    abstract = (ROOT / "manuscript" / "00_abstract.md").read_text(encoding="utf-8")
    introduction = (ROOT / "manuscript" / "01_introduction.md").read_text(encoding="utf-8")
    results = (ROOT / "manuscript" / "03_results.md").read_text(encoding="utf-8")
    conclusion = (ROOT / "manuscript" / "08_conclusion.md").read_text(encoding="utf-8")
    assert "lowest-loss implemented predictor" in abstract
    assert "statistical tie" in abstract
    assert "[@sec:discussion-model-map]" in introduction
    assert "adaptive baselines regain the lead" in results
    assert "commit when the state is effectively observed" in conclusion


def test_completed_adaptive_aif_result_is_not_hosted_in_forward_roadmap() -> None:
    discussion = (ROOT / "manuscript" / "04_discussion.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "manuscript" / "07_roadmap.md").read_text(encoding="utf-8")
    assert "{#sec:discussion-adaptive-aif}" in discussion
    assert "#fig:aif-deepening-candidates" in discussion
    assert "ADAPTIVE_AIF_MEAN_GAIN" in discussion
    assert "ADAPTIVE_AIF_MEAN_GAIN" not in roadmap
    assert "#fig:aif-deepening-candidates" not in roadmap
    assert "[@sec:discussion-adaptive-aif]" in roadmap


def test_no_hardcoded_run_dependent_numbers() -> None:
    """Every run-dependent number must arrive via a {{TOKEN}}; prose numerals are
    limited to structural constants (tool versions, algorithm names)."""

    excluded = {
        "AGENTS.md",
        "README.md",
        "SYNTAX.md",
        "99_references.md",
        "preamble.md",
    }
    offenders: list[str] = []
    for path in sorted((ROOT / "manuscript").glob("*.md")):
        if path.name in excluded:
            continue
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
        text = re.sub(r"\$[^$]*\$", "", text)
        text = re.sub(r"\{\{[A-Z][A-Z0-9_]*\}\}", "", text)
        text = re.sub(r"`[^`]*`", "", text)
        text = re.sub(r"\[@[^\]]*\]", "", text)
        text = re.sub(r"\{#[^}]*\}", "", text)
        scan_text, numeral_matches = _numeral_scan(text)
        for match in numeral_matches:
            start = max(0, match.start() - 60)
            offenders.append(
                f"{path.name}: '{match.group(0)}' "
                f"...{scan_text[start : match.end() + 30]!r}"
            )
    assert not offenders, "hardcoded run-dependent numbers found:\n" + "\n".join(
        offenders
    )


def test_publication_identifier_scoping_keeps_prose_drift_failing() -> None:
    """The DOI/URL exemption must not turn off numeric or boundary prose checks.

    Only the publication identifiers themselves are metadata. Any other
    run-dependent numeral in abstract/manuscript prose still fails both scans,
    so a drifted result number cannot hide behind a link or a DOI.
    """

    identifiers = (
        "archived at doi.org/10.5281/zenodo.21330636 and "
        "https://github.com/ActiveInferenceInstitute/active_fractal_rabbit"
    )
    assert _abstract_offenders(identifiers) == []
    assert _numeral_scan(identifiers)[1] == []

    drifted_number = "latent recovery reached 0.42 under the noisy regime"
    assert _abstract_offenders(drifted_number) == ["numeric quantity"]
    assert [match.group(0) for match in _numeral_scan(drifted_number)[1]] == ["0.42"]

    drifted_framing = "a java toolchain builds the synthetic fixture"
    assert sorted(
        term for term in ABSTRACT_ENVIRONMENT_TERMS if term in drifted_framing
    ) == ["java", "toolchain"]


def test_abstract_publication_identifiers_match_configured_doi() -> None:
    """The DOI/URL exemption is compensated by an explicit provenance pin.

    The numeric-prose scans no longer flag the abstract's DOI digits (they are
    publication metadata). Instead the abstract must cite exactly the concept
    DOI and repository URL recorded in ``manuscript/config.yaml``, so a drifted
    or typo'd identifier fails here.
    """

    abstract = (ROOT / "manuscript" / "00_abstract.md").read_text(encoding="utf-8")
    config = yaml.safe_load(
        (ROOT / "manuscript" / "config.yaml").read_text(encoding="utf-8")
    )
    publication = config["publication"]
    cited_dois = set(re.findall(r"10\.\d{4,}/[^\s)\]]+", abstract))
    assert cited_dois == {publication["doi"]}
    assert publication["repository_url"] in abstract


def test_manuscript_and_docs_do_not_make_positive_operational_claims() -> None:
    """Operational/tasking terms may appear only as explicit limits or nulls."""

    corpus_paths = [
        *sorted((ROOT / "manuscript").glob("*.md")),
        *sorted((ROOT / "docs").glob("*.md")),
        ROOT / "README.md",
    ]
    boundary_markers = {
        "not",
        "no ",
        "never",
        "without",
        "rather than",
        "does not",
        "do not",
        "should not",
        "null",
        "limitation",
        "boundary",
        "out of scope",
        "oversight",
        "minimization",
        "separate empirical",
        "requires",
        "require",
    }
    offenders: list[str] = []
    for path in corpus_paths:
        if path.name in {"AGENTS.md", "SYNTAX.md", "references.bib"}:
            continue
        text = path.read_text(encoding="utf-8")
        for sentence in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
            lower = sentence.lower()
            if "evasion" in lower and "no " not in lower:
                offenders.append(f"{path.relative_to(ROOT)}: {sentence.strip()}")
            if ("operational" in lower or "tasking" in lower) and not any(
                marker in lower for marker in boundary_markers
            ):
                offenders.append(f"{path.relative_to(ROOT)}: {sentence.strip()}")
    assert not offenders, (
        "positive operational/tasking/evasion wording found:\n" + "\n".join(offenders)
    )
