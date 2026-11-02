from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from active_fractal_rabbit.config import load_config
from active_fractal_rabbit.visual_claim_audit import audit_visual_claims
from active_fractal_rabbit.visual_quality import audit_visual_quality

ROOT = Path(__file__).resolve().parents[1]

PUBLIC_DOC_PATHS = (
    "README.md",
    "AGENTS.md",
    "data/AGENTS.md",
    "docs/architecture.md",
    "docs/config_schema.md",
    "docs/quickstart.md",
    "docs/validation.md",
    "docs/claim_ledger.md",
    "docs/ip_license_status.md",
    "docs/scholarship_ledger.md",
    "docs/methods/benchmark-protocol.md",
    "docs/methods/fair-frontier-sota.md",
    "docs/methods/statistics.md",
    "manuscript/README.md",
    "manuscript/S02_reproducibility.md",
    "manuscript/07_roadmap.md",
)

CLAIM_BOUNDARY_PATHS = PUBLIC_DOC_PATHS + (
    "docs/governance/privacy-and-minimization.md",
    "docs/revision/aif-model-designs.md",
    "docs/revision/pass-C-pymdp-push.md",
    "docs/revision/REVISION_PROGRAM.md",
    "docs/revision/pass-B-restructure-imrad.md",
    "manuscript/00_abstract.md",
    "manuscript/03b_results_null_hierarchy_and_multiplicity.md",
    "manuscript/04_discussion.md",
    "manuscript/08_conclusion.md",
    "manuscript/S01_validation_and_threat_model.md",
)

REQUIRED_SCHOLARSHIP_SOURCES = {
    "huygens1657_chance": "https://archive.org/details/ned-kbn-all-00003375-001",
    "jacob_bernoulli1713_ars_conjectandi": "https://archive.org/details/bub_gb_kz9nvk99EWoC",
    "demoivre1718_doctrine_chances": "https://archive.org/details/bim_eighteenth-century_the-doctrine-of-chances_moivre-abraham-de_1718",
    "daniel_bernoulli1738_mensura_sortis": "https://archive.org/details/SpecimenTheoriaeNovaeDeMensuraSortis",
    "bayes1763_doctrine_chances": "https://doi.org/10.1098/rstl.1763.0053",
    "laplace1774_probability_causes": "https://sites.mathdoc.fr/cgi-bin/oetoc?id=OE_LAPLACE__8",
    "berkeley1709_theory_vision": "https://archive.org/details/bim_eighteenth-century_an-essay-towards-a-new-t_1709",
    "locke1690_human_understanding": "https://www.gutenberg.org/ebooks/10615",
    "hume1748_human_understanding": "https://www.gutenberg.org/ebooks/9662",
    "kant1781_pure_reason": "https://www.gutenberg.org/ebooks/4280",
    "boyle1661_sceptical_chymist": "https://archive.org/details/scepticalchymist0000boyl_i6r3",
    "newton1729_principia_rules": "https://archive.org/details/bub_gb_6EqxPav3vIsC",
    "graunt1662_bills_mortality": "https://archive.org/details/2356015R.nlm.nih.gov",
    "halley1693_mortality": "https://doi.org/10.1098/rstl.1693.0007",
    "sinclair1791_statistical_account": "https://archive.org/details/b21365799_018",
    "blackstone1765_commentaries": "https://archive.org/details/bim_eighteenth-century_commentaries-on-the-laws_blackstone-william-sir_1765_1",
    "bentham1791_panopticon": "https://archive.org/details/10059224bsb",
    "uscensus1790_first_census": "https://www.census.gov/programs-surveys/decennial-census/decade.1790.html",
    "luca2021deep_learning_mobility": "https://doi.org/10.1145/3485125",
    "yan2017universal_mobility": "https://doi.org/10.1038/s41467-017-01892-8",
    "wu2024pmt": "https://arxiv.org/abs/2406.02578",
    "haydari2024mobilitygpt": "https://arxiv.org/abs/2402.03264",
    "gao2022ssdl_mobility": "https://arxiv.org/abs/2211.09625",
    "xu2025trajectory_llm_survey": "https://arxiv.org/abs/2506.03408",
    "shi2026sparse_trajectory_prediction": "https://doi.org/10.1109/TPAMI.2025.3626815",
    "long2024unimob": "https://arxiv.org/abs/2412.15294",
    "solatorio2023geoformer": "https://arxiv.org/abs/2311.05092",
    "han2025movegpt": "https://arxiv.org/abs/2505.18670",
    "tang2024llama_mob": "https://arxiv.org/abs/2410.23692",
    "qin2026mobllm": "https://doi.org/10.1016/j.trc.2026.105562",
    "liu2024nextlocllm": "https://arxiv.org/abs/2410.09129",
    "liu2026intentpoi": "https://arxiv.org/abs/2606.08122",
    "li2026trajgenagent": "https://arxiv.org/abs/2606.12657",
    "yuan2025movegcl": "https://arxiv.org/abs/2506.06694",
    "liu2026unified_mobility_generation": "https://arxiv.org/abs/2602.19694",
    "santos2026plausible_urban_sim": "https://arxiv.org/abs/2606.13835",
    "deng2025mirage": "https://doi.org/10.1145/3690624.3709180",
    "kapp2023synthetic_urban_mobility_review": "https://doi.org/10.1145/3610224",
    "kapp2024synthetic_mobility_utility": "https://doi.org/10.1145/3589132.3625661",
    "merhi2024rtct": "https://arxiv.org/abs/2407.16938",
    "yuan2026worldmove": "https://doi.org/10.1038/s41597-026-06555-2",
    "kaabachi2025medical_synthetic_metrics": "https://doi.org/10.1038/s41746-024-01359-3",
    "asano2025mobqa": "https://arxiv.org/abs/2508.11163",
    "ozaki2025privacy_preserving_daily_trajectories": "https://doi.org/10.1109/BigData66926.2025.11401071",
    "gonzalez2008mobility": "https://doi.org/10.1038/nature06958",
    "song2010predictability": "https://doi.org/10.1126/science.1177170",
    "ikanovic2016predictability": "https://arxiv.org/abs/1608.06419",
    "luca2023trajectory_overlap": "https://doi.org/10.1007/s10994-023-06386-x",
    "fang2025mobert": "https://doi.org/10.1145/3748636.3771316",
    "kaelbling1998pomdp": "https://doi.org/10.1016/S0004-3702(98)00023-X",
    "limanjaya2026sight": "https://doi.org/10.1145/3787973",
    "patterson2017movement_models": "https://doi.org/10.1007/s10182-017-0302-7",
    "heins2022pymdp": "https://doi.org/10.21105/joss.04098",
    "dacosta2020discrete_active_inference": "https://doi.org/10.1016/j.jmp.2020.102447",
    "sajid2021demystified": "https://doi.org/10.1162/neco_a_01357",
    "millidge2020efe": "https://doi.org/10.1162/neco_a_01354",
    "champion2024reframing_efe": "https://arxiv.org/abs/2402.14460",
    "sweeney2026formal_equivalences": "https://doi.org/10.3390/e28010001",
    "maisto2025active_inference_tree_search": "https://doi.org/10.1016/j.neucom.2024.129319",
    "nguyen2024raif": "https://arxiv.org/abs/2409.14216",
    "ramstead2023bayesian_mechanics": "https://doi.org/10.1098/rsfs.2022.0029",
    "dacosta2021bayesian_mechanics_stationary": "https://doi.org/10.1098/rspa.2021.0518",
    "friston2019particular_physics": "https://doi.org/10.48550/arXiv.1906.10184",
    "friston2023path_integrals_particular_kinds": "https://doi.org/10.1016/j.plrev.2023.08.016",
    "kirchhoff2018markov_blankets_life": "https://doi.org/10.1098/rsif.2017.0792",
    "parr2020markov_blankets_info_geometry": "https://doi.org/10.1098/rsta.2019.0159",
    "aguilera2022how_particular_fep": "https://doi.org/10.1016/j.plrev.2021.11.001",
    "gneiting2007scoring": "https://doi.org/10.1198/016214506000001437",
    "gneiting2014probabilistic_forecasting": "https://doi.org/10.1146/annurev-statistics-062713-085831",
    "diebold1995predictive_accuracy": "https://doi.org/10.1080/07350015.1995.10524599",
    "field2007clustered_bootstrap": "https://doi.org/10.1111/j.1467-9868.2007.00593.x",
    "cleveland1984graphical_perception": "https://doi.org/10.2307/2288400",
    "gelman2002tables_graphs": "https://doi.org/10.1198/000313002317572790",
    "munzner2014visualization_design": "https://doi.org/10.1201/b17511",
    "hullman2020uncertainty": "https://doi.org/10.1109/TVCG.2019.2934287",
    "spiegelhalter2011visualizing_uncertainty": "https://doi.org/10.1126/science.1191181",
    "demontjoye2013unique": "https://doi.org/10.1038/srep01376",
    "kondor2018trace_matching": "https://doi.org/10.1109/TBDATA.2018.2871693",
    "dwork2006differential_privacy": "https://doi.org/10.1007/11787006_1",
    "buchholz2024trajectory_privacy_sok": "https://doi.org/10.56553/popets-2024-0068",
    "bouras2026privacy_eval": "https://arxiv.org/abs/2605.15246",
    "cherigui2026synthetic_trajectory_dual": "https://arxiv.org/abs/2604.19653",
    "guepin2026diffghost": "https://arxiv.org/abs/2605.10647",
    "buchholz2025dp_trajectory_cost": "https://arxiv.org/abs/2506.09312",
    "yadav2024contextual_location_privacy": "https://doi.org/10.1145/3678717.3691211",
    "haggerty2000surveillant_assemblage": "https://doi.org/10.1080/00071310020015280",
    "heuer1999psychology_intelligence_analysis": "https://www.cia.gov/resources/csi/static/Pyschology-of-Intelligence-Analysis.pdf",
    "horowitz2024automation_bias": "https://doi.org/10.1093/isq/sqae020",
    "nist2023ai_rmf": "https://doi.org/10.6028/NIST.AI.100-1",
    "nist2024csf": "https://doi.org/10.6028/NIST.CSWP.29",
    "dod2021responsible_ai": "https://media.defense.gov/2021/May/27/2002730593/-1/-1/0/IMPLEMENTING-RESPONSIBLE-ARTIFICIAL-INTELLIGENCE-IN-THE-DEPARTMENT-OF-DEFENSE.PDF",
    "odni2022cai_report": "https://www.dni.gov/files/ODNI/documents/assessments/ODNI-Declassified-Report-on-CAI-January2022.pdf",
    "odni2024cai_framework": "https://www.dni.gov/files/ODNI/documents/CAI/Commercially-Available-Information-Framework-May2024.pdf",
    "ftc2024xmode_outlogic": "https://www.ftc.gov/news-events/news/press-releases/2024/04/ftc-finalizes-order-x-mode-successor-outlogic-prohibiting-it-sharing-or-selling-sensitive-location",
    "ftc2026gm_onstar": "https://www.ftc.gov/news-events/news/press-releases/2026/01/ftc-finalizes-order-settling-allegations-gm-onstar-collected-sold-geolocation-data-without-consumers",
    "sandve2013reproducible": "https://doi.org/10.1371/journal.pcbi.1003285",
    "wilson2017good_enough": "https://doi.org/10.1371/journal.pcbi.1005510",
    "smith2016software_citation": "https://doi.org/10.7717/peerj-cs.86",
}

ACTIVE_TASK_TODO_SNIPPETS = {
    "afr-23": "Real waypoint-data validation stays blocked",
    "afr-32": "Multi-realization external-lane fair ladder",
    "afr-33": "Multiscale or continuous latent state",
    "afr-34": "Regime-stressor lane",
    "afr-36": "Scholarship refresh remains source-backed and claim-bound",
    "afr-37": "Publish the challenge pack",
    "afr-38": "Stronger baselines re-scored on the locked split",
    "afr-39": "Learned hierarchical/factorial dynamics",
    "afr-40": "Synthetic SOTA progress cannot unlock",
    "afr-41": "Empirical transfer study (blocked)",
}

ACTIVE_TASK_TODO_BRIEFS = {
    "afr-23": "docs/todo/real-waypoint-data-validation-blocked.md",
    "afr-32": "docs/todo/external-realization-promotion-gate.md",
    "afr-33": "docs/todo/continuous-multiscale-particle-filter-dynamics.md",
    "afr-34": "docs/todo/larger-regime-structured-synthetic-stressors.md",
    "afr-36": "docs/todo/literature-and-leaderboard-baseline-map.md",
    "afr-37": "docs/todo/synthetic-challenge-suite-leakage-resistant-splits.md",
    "afr-38": "docs/todo/strong-baseline-harness-leaderboard.md",
    "afr-39": "docs/todo/learned-active-inference-candidate-models.md",
    "afr-40": "docs/todo/sota-promotion-claim-lock.md",
    "afr-41": "docs/todo/consented-empirical-benchmark-execution-blocked.md",
}


def test_default_fixture_signposts_match_default_config() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    default_fixture = Path(config.fractalrabbit.fixture_csv).name

    for rel_path in ("README.md", "AGENTS.md", "data/AGENTS.md"):
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        assert default_fixture in text
        assert "fractalrabbit_sample.csv` is the fixture for the default" not in text
        assert "Small checked CSV matching FractalRabbit's output contract" not in text

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "fractalrabbit_sample.csv` fixture remains the regression" in readme


def test_documented_figure_counts_match_registry_and_default_config() -> None:
    registry_path = ROOT / "output" / "figures" / "figure_registry.json"
    if not registry_path.is_file():
        pytest.skip(
            f"generated figure registry not present: missing {registry_path} "
            "(removal condition: run scripts/07_run_full_study.py --lane "
            "fixture and scripts/04_generate_figures.py --run "
            "output/runs/default; owner: fixture lane)"
        )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    config = load_config(ROOT / "configs" / "default.yaml")
    expected = len(registry["entries"])

    assert expected == len(config.figures.enabled)

    claims: list[tuple[str, int]] = []
    for rel_path in (
        "docs/architecture.md",
        "docs/config_schema.md",
        "docs/quickstart.md",
        "docs/validation.md",
    ):
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        for pattern in (
            r"registry contains (\d+) (?:manuscript )?figures",
            r"default figure list has (\d+) keys",
            r"default (\d+)-figure",
        ):
            claims.extend(
                (rel_path, int(match.group(1))) for match in re.finditer(pattern, text)
            )

    assert claims
    assert [(path, count) for path, count in claims if count != expected] == []


def test_documented_figure_keys_match_default_config() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    doc = (ROOT / "docs" / "config_schema.md").read_text(encoding="utf-8")
    section = doc.split("The default manuscript-facing figure keys are:", maxsplit=1)[1]
    section = section.split("## Lane-Suffixed", maxsplit=1)[0]
    documented = set(re.findall(r"`([a-z0-9_]+)`", section))

    assert documented == set(config.figures.enabled)


def test_public_docs_do_not_carry_stale_release_signposts() -> None:
    stale: list[tuple[str, str]] = []
    stale_patterns = {
        "old template path": r"docxology/template",
        "old 42-figure count": r"(?:default\s+)?42(?:-figure| figures| manuscript figures)",
        "old 39-figure count": r"(?:default\s+)?39(?:-figure| figures| manuscript figures)",
        "stale run-artifact path": r"`(?:\.\./)?output/(?:sota_|fair_|power_)[^`]+`",
        "stale oracle wording": r"near-unbeatable|unbeatable by construction|no location-blind method can win",
        "stale two-lane sota program": r"(?:two-lane|Two-lane) SOTA-program",
    }
    for rel_path in PUBLIC_DOC_PATHS:
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        for label, pattern in stale_patterns.items():
            if re.search(pattern, text):
                stale.append((rel_path, label))

    assert stale == []


def test_public_release_instructions_render_before_finalizing() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = (ROOT / "docs" / "quickstart.md").read_text(encoding="utf-8")
    status = (ROOT / "docs" / "ip_license_status.md").read_text(encoding="utf-8")
    supplement = (ROOT / "manuscript" / "S02_reproducibility.md").read_text(
        encoding="utf-8"
    )
    for text in (readme, quickstart, status, supplement):
        assert "scripts/29_finalize_publication_package.py" in text
    for text in (readme, quickstart, supplement):
        assert text.index("scripts/03_render_pdf.py") < text.index(
            "scripts/29_finalize_publication_package.py"
        )
        assert "scripts/29_finalize_publication_package.py --verify-only" in text
        assert "scripts/06_build_release_manifest.py" not in text
    for text in (readme, quickstart):
        assert "cd ../projects/working/active_fractal_rabbit" in text
        assert "cd ../working/active_fractal_rabbit" not in text


def test_forward_todo_tracks_open_taskboard_without_reintroducing_history() -> None:
    todo = (ROOT / "TODO.md").read_text(encoding="utf-8")
    taskboard = yaml.safe_load((ROOT / "tasks.yaml").read_text(encoding="utf-8"))
    open_tasks = {
        task["id"]: task
        for task in taskboard["tasks"]
        if task.get("status") in {"todo", "active", "in_progress", "blocked"}
    }

    assert set(ACTIVE_TASK_TODO_SNIPPETS) <= set(open_tasks)
    missing = [
        (task_id, snippet)
        for task_id, snippet in ACTIVE_TASK_TODO_SNIPPETS.items()
        if snippet not in todo
    ]
    assert missing == []
    assert "[x]" not in todo
    assert "## Implemented Gates" not in todo
    assert "This revision has delivered" not in todo
    assert "docs/revision/" in todo and "ISA.md" in todo


def test_open_taskboard_items_have_scoped_docs_todo_briefs() -> None:
    todo = (ROOT / "TODO.md").read_text(encoding="utf-8")
    taskboard = yaml.safe_load((ROOT / "tasks.yaml").read_text(encoding="utf-8"))
    open_tasks = {
        task["id"]: task
        for task in taskboard["tasks"]
        if task.get("status") in {"todo", "active", "in_progress", "blocked"}
    }

    assert set(ACTIVE_TASK_TODO_BRIEFS) == set(open_tasks)

    offenders: list[str] = []
    for task_id, rel_path in ACTIVE_TASK_TODO_BRIEFS.items():
        path = ROOT / rel_path
        if not path.exists():
            offenders.append(f"{task_id}: missing {rel_path}")
            continue

        brief = path.read_text(encoding="utf-8")
        required_phrases = (
            f"Source task: `{task_id}`",
            f"Status: {open_tasks[task_id]['status']}",
            "## Current Evidence",
            "## Blocked And Unblocked Conditions",
            "## Implementation Scope",
            "## Non-Goals",
            "## Acceptance Gates",
            "## Artifact And Provenance Surfaces",
            "## Claim Boundary",
        )
        missing = [phrase for phrase in required_phrases if phrase not in brief]
        if missing:
            offenders.append(f"{task_id}: {rel_path} missing {missing}")

        if rel_path not in todo:
            offenders.append(f"{task_id}: {rel_path} not linked from TODO.md")

    docs_index = (ROOT / "docs" / "todo" / "README.md").read_text(encoding="utf-8")
    for task_id, rel_path in ACTIVE_TASK_TODO_BRIEFS.items():
        if task_id not in docs_index or Path(rel_path).name not in docs_index:
            offenders.append(
                f"{task_id}: {rel_path} not indexed in docs/todo/README.md"
            )

    assert offenders == []


def test_unchecked_markdown_checklists_are_confined_to_active_or_historical_queues() -> (
    None
):
    allowed = {
        "TODO.md",
        "ISA.md",
    }
    offenders: list[str] = []
    for path in ROOT.rglob("*.md"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("docs/revision/"):
            continue
        if rel in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"(?m)^- \[ \] ", text):
            offenders.append(rel)

    assert offenders == []


def _bib_entries() -> dict[str, str]:
    text = (ROOT / "manuscript" / "references.bib").read_text(encoding="utf-8")
    return {
        match.group(1): match.group(2)
        for match in re.finditer(r"@\w+\{([^,]+),([\s\S]*?)(?=\n@|\Z)", text)
    }


def _scholarship_rows() -> dict[str, dict[str, str]]:
    text = (ROOT / "docs" / "scholarship_ledger.md").read_text(encoding="utf-8")
    rows: dict[str, dict[str, str]] = {}
    headers: list[str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        if cells[0] == "key":
            headers = cells
            continue
        if headers is None or len(cells) != len(headers):
            continue
        row = dict(zip(headers, cells, strict=True))
        rows[row["key"]] = row
    return rows


def _source_is_represented_in_bib(source: str, bib_body: str) -> bool:
    compact_body = re.sub(r"\s+", "", bib_body.lower())
    source_lower = source.lower()
    if "doi.org/" in source_lower:
        doi = source_lower.rsplit("doi.org/", maxsplit=1)[1]
        return doi in compact_body
    if "arxiv.org/abs/" in source_lower:
        arxiv_id = source_lower.rsplit("/", maxsplit=1)[1]
        return arxiv_id in compact_body
    return source_lower in compact_body


def _manuscript_citation_keys() -> set[str]:
    citation_keys: set[str] = set()
    for path in (ROOT / "manuscript").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\[@([^\]]+)\]", text):
            for raw_key in re.split(r"[;\s]+", match.group(1)):
                key = raw_key.strip().lstrip("@").strip(",")
                if key and not re.match(r"^(fig|tbl|eq|sec):", key):
                    citation_keys.add(key)
    return citation_keys


def test_scholarship_ledger_is_primary_source_backed_and_bibliography_bound() -> None:
    ledger = _scholarship_rows()
    bibliography = _bib_entries()

    assert set(REQUIRED_SCHOLARSHIP_SOURCES) <= set(ledger)
    assert set(REQUIRED_SCHOLARSHIP_SOURCES) <= set(bibliography)
    assert set(ledger) <= set(bibliography)
    assert _manuscript_citation_keys() == set(ledger)

    incomplete_rows: list[tuple[str, str]] = []
    mismatched_sources: list[tuple[str, str, str]] = []
    for key, row in ledger.items():
        for field in (
            "primary source",
            "manuscript surface",
            "claim supported",
            "claim constrained",
            "allowed wording",
            "forbidden overreach",
        ):
            if not row.get(field):
                incomplete_rows.append((key, field))
        if not _source_is_represented_in_bib(row["primary source"], bibliography[key]):
            mismatched_sources.append((key, "bibliography", row["primary source"]))

    for key, expected_source in REQUIRED_SCHOLARSHIP_SOURCES.items():
        row = ledger[key]
        if row["primary source"] != expected_source:
            mismatched_sources.append((key, row["primary source"], expected_source))

    assert incomplete_rows == []
    assert mismatched_sources == []


def test_scholarship_forbidden_overreach_phrases_stay_quarantined_to_ledger() -> None:
    ledger = _scholarship_rows()
    forbidden_phrases = {
        row["forbidden overreach"].lower()
        for row in ledger.values()
        if row.get("forbidden overreach")
    }
    checked_paths = [
        path
        for path in [*PUBLIC_DOC_PATHS, *CLAIM_BOUNDARY_PATHS]
        if path != "docs/scholarship_ledger.md"
    ]
    failures: list[tuple[str, str]] = []
    for rel_path in checked_paths:
        text = (ROOT / rel_path).read_text(encoding="utf-8").lower()
        for phrase in forbidden_phrases:
            if phrase in text:
                failures.append((rel_path, phrase))

    assert failures == []


def test_visual_audit_contract_shape_is_documented() -> None:
    registry_path = ROOT / "output" / "figures" / "figure_registry.json"
    if not registry_path.is_file():
        pytest.skip(
            f"generated figure registry not present: missing {registry_path} "
            "(removal condition: run scripts/07_run_full_study.py --lane "
            "fixture and scripts/04_generate_figures.py --run "
            "output/runs/default; owner: fixture lane)"
        )
    quality = audit_visual_quality(ROOT)
    visual_claims = audit_visual_claims(ROOT)
    docs_blob = "\n".join(
        (ROOT / rel_path).read_text(encoding="utf-8") for rel_path in PUBLIC_DOC_PATHS
    )

    summary = quality["quality_summary"]
    layout_critical = quality["layout_critical_figures"]
    assert summary["manual_review_surface"] == "output/figures/figure_contact_sheet.png"
    assert summary["manual_review_contact_sheet"]["exists"] is True
    assert summary["manual_review_contact_sheet"]["image_readable"] is True
    assert summary["manual_review_contact_sheet"]["current"] is True
    assert (
        layout_critical["manual_review_surface"]
        == "output/figures/figure_contact_sheet.png"
    )
    assert layout_critical["missing_keys"] == []
    assert layout_critical["figure_count"] == len(layout_critical["keys"])
    assert {
        "confound_taxonomy",
        "multiplicity_ledger",
        "privacy_utility_frontier",
        "linkage_risk_frontier",
    } <= set(layout_critical["keys"])
    for figure in layout_critical["figures"]:
        assert figure["path"].startswith("output/figures/")
        assert figure["source"].startswith("output/")
        assert figure["review_reason"]
    for field in (
        "shortest_figure",
        "narrowest_figure",
        "lowest_pixel_stddev_figure",
        "widest_aspect_figure",
        "tallest_aspect_figure",
    ):
        assert summary[field]["key"]
        assert summary[field]["path"].startswith("output/figures/")
    assert quality["min_aspect_ratio"] > 0.0
    assert quality["max_aspect_ratio"] >= quality["min_aspect_ratio"]

    for brief in visual_claims["briefs"]:
        assert brief["story_job"]
        assert brief["reader_task"]
        assert brief["data_shape"]
        assert brief["encoding_and_layout"]
        assert brief["claim_boundary"]

    assert "quality_summary" in docs_blob
    assert "layout_critical_figures" in docs_blob
    assert "reader task" in docs_blob
    assert "claim boundary" in docs_blob


def test_artifact_inventories_include_canonical_and_lane_qualified_outputs() -> None:
    inventory_paths = (
        "docs/config_schema.md",
        "docs/quickstart.md",
        "manuscript/S02_reproducibility.md",
    )
    for rel_path in inventory_paths:
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "`sequence_baseline_analysis.json`" in text
        assert "`missingness_model_analysis.json`" in text
        assert "sequence_baseline_analysis_" in text
        assert "missingness_model_analysis_" in text


def test_publication_metadata_and_ip_status_are_signposted() -> None:
    required_paths = (
        "LICENSE",
        "LICENSE-DOCS.md",
        "THIRD_PARTY_NOTICES.md",
        "CITATION.cff",
        "docs/ip_license_status.md",
    )
    for rel_path in required_paths:
        assert (ROOT / rel_path).is_file()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    status_doc = (ROOT / "docs" / "ip_license_status.md").read_text(encoding="utf-8")
    third_party = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    for rel_path in required_paths:
        assert rel_path in readme or rel_path == "docs/ip_license_status.md"
    assert "ip_license_status.md" in docs_index
    assert "public release" in status_doc.lower()
    assert "release manifest" in status_doc.lower()
    assert "10.5281/zenodo.21330636" in status_doc
    assert "Apache-2.0" in status_doc
    assert "MIT License" in third_party
    assert "9933449c4f4fe1b26b6ac7bfdeeac76583085df5" in third_party


def test_current_claim_boundaries_avoid_reviewed_overclaims() -> None:
    forbidden_patterns = {
        "realism laundering": r"realistic sparse-waypoint|realistic for sporadic location reports",
        "privacy guarantee overclaim": r"the protection is real|metadata-bundle recovery is real but tiny",
        "stale metadata verdict": r"metadata bundle has shifted from a null claim to a .*small positive",
        "universal oracle ceiling": r"absolute ceiling|no model can beat true previous cell",
        "stale leakage delta": r"0\.25 nats lower than any honest baseline",
        "closed-loop EFE conflation": r"closed-loop selector.{0,120}expected free energy",
        "stale cross-vendor pending": r"cross-vendor[^.\n]{0,100}pending|pending[^.\n]{0,100}cross-vendor|Forge[^.\n]{0,100}pending",
        "synthetic empirical laundering": r"core finding is empirical|verdict is empirical",
        "fair-frontier sota label": r"Fair-Frontier SOTA|fair-frontier SOTA verdict",
        "stale external execution pending": r"External-lane execution pending",
    }
    failures: list[tuple[str, str]] = []
    for rel_path in CLAIM_BOUNDARY_PATHS:
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        for label, pattern in forbidden_patterns.items():
            if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                failures.append((rel_path, label))

    assert failures == []


def test_revision_design_notes_do_not_launder_external_simulator_as_real_data() -> None:
    text = (ROOT / "docs" / "revision" / "aif-model-designs.md").read_text(
        encoding="utf-8"
    )
    forbidden_patterns = {
        "external data": r"\bexternal data\b",
        "true external data": r"\btrue external data\b",
        "real external data": r"\breal external data\b",
        "real data": r"\breal data\b",
        "real sparse data": r"\breal sparse data\b",
        "real reporting": r"\breal reporting\b",
        "real sparse reporting": r"\breal sparse[- ](?:mobility[- ])?reporting\b",
        "real sparse mobility": r"\breal sparse[- ]mobility\b",
        "not confirmed real": r"\bnot confirmed real\b",
    }

    failures = [
        label
        for label, pattern in forbidden_patterns.items()
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]

    assert failures == []


def test_public_docs_do_not_launder_uniqueness_or_privacy_proxy_claims() -> None:
    uniqueness_patterns = (
        r"unique(?:ly)? to active[- ]inference",
        r"active[- ]inference[- ]specific (?:advantage|mechanism|superiority)",
    )
    comparator_bound = (
        r"not unique|shared with|HMM|hidden Markov|particle|state[- ]space|"
        r"belief[- ]marginali[sz](?:e|ing)"
    )
    privacy_proxy_patterns = (
        r"privacy (?:is )?proven by (?:the )?(?:proxy|frontier|linkage)",
        r"(?:linkage|unicity|minimi[sz]ation|privacy[- ]utility)[^.]{0,80}"
        r"(?:proves|guarantees|certifies) privacy",
        r"synthetic risk prox(?:y|ies)[^.]{0,80}(?:prove|proves|guarantee|guarantees)",
    )
    failures: list[tuple[str, str, str]] = []
    checked_paths = [
        path
        for path in [*PUBLIC_DOC_PATHS, *CLAIM_BOUNDARY_PATHS]
        if path != "docs/scholarship_ledger.md"
    ]

    for rel_path in checked_paths:
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
            if sentence.strip()
        ]
        for sentence in sentences:
            for pattern in uniqueness_patterns:
                if re.search(pattern, sentence, flags=re.IGNORECASE) and not re.search(
                    comparator_bound, sentence, flags=re.IGNORECASE
                ):
                    failures.append(
                        (rel_path, "unbounded active-inference uniqueness", sentence)
                    )
            for pattern in privacy_proxy_patterns:
                if re.search(pattern, sentence, flags=re.IGNORECASE):
                    failures.append((rel_path, "privacy proxy laundering", sentence))

    assert failures == []


def test_roadmap_is_forward_only_not_a_delivered_work_recap() -> None:
    text = (ROOT / "manuscript" / "07_roadmap.md").read_text(encoding="utf-8")

    assert "This revision has delivered" not in text
    assert "## Implemented Gates" not in text
    assert (
        "Current Evidence Stops at Synthetic Artifacts {#sec:roadmap-implemented-now}"
        in text
    )


def test_fair_frontier_method_links_existing_output_artifacts() -> None:
    runs_dir = ROOT / "output" / "runs"
    if not runs_dir.is_dir():
        pytest.skip(
            f"generated run surface not present: missing {runs_dir} (removal "
            "condition: run scripts/07_run_full_study.py --lane fixture; "
            "owner: fixture lane)"
        )
    doc_path = ROOT / "docs" / "methods" / "fair-frontier-sota.md"
    text = doc_path.read_text(encoding="utf-8")
    artifact_paths = re.findall(r"`((?:\.\./\.\./)?output/runs/[^`]+)`", text)

    assert artifact_paths
    missing = []
    for rel_path in artifact_paths:
        path = (
            (doc_path.parent / rel_path).resolve()
            if rel_path.startswith("../")
            else ROOT / rel_path
        )
        if not path.exists():
            missing.append(rel_path)

    assert missing == []
