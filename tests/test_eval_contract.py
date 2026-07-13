"""The honest-benchmark evaluation contract: a claim-level linter.

Two paired requirements, both load-bearing: the project's OWN honest, regime-
scoped, null-bearing manuscript must pass every guard (zero false rejections,
including the honest NEGATIVE location-blind result), and seeded violations of
each guard must be caught (rejection recall).
"""

from __future__ import annotations

from pathlib import Path

from active_fractal_rabbit.eval_contract import (
    EXECUTION_ONLY_CLASSES,
    STATICALLY_CATCHABLE_CLASSES,
    check_active_inference_help_claims,
    check_cue_disclosure,
    check_deployment_surveillance_claims,
    check_null_regime,
    check_performance_claims,
    run_contract,
)

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"


def test_real_manuscript_passes_the_contract() -> None:
    """The project's own honest claims must all pass — no false rejections."""

    report = run_contract(MANUSCRIPT)
    assert report.passed, [f"{v.guard}: {v.sentence}" for v in report.violations]


def test_g1_catches_wrong_regime_skill_claim() -> None:
    """A positive 'beats persistence' claim citing a location-blind token is the
    session-6 trap and must be flagged."""

    bad = "The filter beats persistence with recovery skill {{BLIND_LIMIT_SCRP_SKILL}} even with no location reported."
    violations = check_null_regime(bad)
    assert any(
        v.guard == "G1-null-regime" and v.token == "BLIND_LIMIT_SCRP_SKILL"
        for v in violations
    )


def test_g1_allows_honest_limit_framing() -> None:
    """A location-blind token inside an explicit limit framing is honest."""

    ok = (
        "The location-blind skill against persistence is an information-asymmetry "
        "limit: flat {{BLIND_LIMIT_FLAT_SKILL}} and speed-conditioned "
        "{{BLIND_LIMIT_SCRP_SKILL}}."
    )
    assert check_null_regime(ok) == []


def test_g1_allows_gap_filling_regime_tokens() -> None:
    """A gap-filling token is not the wrong-regime location-blind error."""

    neutral = "The gap-filling run reports flat {{GAP_FILL_FLAT_SKILL}} and speed-conditioned {{GAP_FILL_SCRP_SKILL}}."
    assert check_null_regime(neutral) == []


def test_g1_is_not_spoofed_by_token_name_substring() -> None:
    """The limit-framing check must read the prose, not the token name (which
    contains the substring 'limit')."""

    spoof = "The filter clearly wins with recovery skill {{BLIND_LIMIT_SCRP_SKILL}}."
    violations = check_null_regime(spoof)
    assert any(v.token == "BLIND_LIMIT_SCRP_SKILL" for v in violations)


def test_g1_is_not_spoofed_by_vague_negative_word() -> None:
    """A stray word like 'negative' must not launder a positive wrong-regime
    claim against the location-blind oracle null."""

    spoof = "The result is not negative: the filter beats persistence with {{BLIND_LIMIT_SCRP_SKILL}}."
    violations = check_null_regime(spoof)

    assert any(
        v.guard == "G1-null-regime" and v.token == "BLIND_LIMIT_SCRP_SKILL"
        for v in violations
    )


def test_g1_not_split_by_abbreviation_in_limit_clause() -> None:
    """An abbreviation ('e.g.') inside a limit-framed sentence must not split the
    limit marker away from the location-blind token (a contract-robustness fix
    surfaced by the contract's own adversarial review)."""

    s = (
        "The location-blind lane is an information-asymmetry limit (e.g. flat skill "
        "{{BLIND_LIMIT_FLAT_SKILL}} and speed-conditioned {{BLIND_LIMIT_SCRP_SKILL}}), "
        "while the gap-filling regime is where inference legitimately wins."
    )
    assert check_null_regime(s) == []


def test_g2_catches_undisclosed_speed_cue() -> None:
    """A speed-cue recovery claim without the displacement-derivation disclosure fails."""

    undisclosed = "Speed-conditioned recovery reaches skill {{GAP_FILL_SCRP_SKILL}} by anticipating moves."
    assert any(
        v.guard == "G2-cue-disclosure" for v in check_cue_disclosure(undisclosed)
    )


def test_g2_passes_when_cue_disclosed() -> None:
    disclosed = (
        "Speed-conditioned recovery reaches skill {{GAP_FILL_SCRP_SKILL}}. "
        "Speed is derived from inter-report displacement, so it is a strong but disclosed cue."
    )
    assert check_cue_disclosure(disclosed) == []


def test_g3_blocks_unsupported_sota_or_performance_leading_claim() -> None:
    bad = "The pymdp agent is state-of-the-art for sparse waypoint prediction."

    violations = check_performance_claims(bad)

    assert any(v.guard == "G3-performance-claim" for v in violations)


def test_g3_allows_negated_future_or_gate_bound_sota_language() -> None:
    negated = "The current result is not a state-of-the-art performance claim."
    future = "Future state-of-the-art performance needs hidden Markov and particle baselines."
    gated = "The current state-of-the-art performance status is {{PYMDP_SOTA_STATUS}}."
    regime_gated = (
        "The regime-qualified synthetic state-of-the-art status is "
        "{{AIF_REGIME_SOTA_STATUS}}, bounded by {{AIF_REGIME_SOTA_BOUNDARY}}."
    )

    assert check_performance_claims(negated) == []
    assert check_performance_claims(future) == []
    assert check_performance_claims(gated) == []
    assert check_performance_claims(regime_gated) == []


def test_g3_incidental_substring_of_not_does_not_launder_a_real_claim() -> None:
    """Forge cross-vendor HIGH-3 (session 27, 2026-07-02): PERFORMANCE_LIMIT_MARKERS carries the
    same bare "not"/"no " substring exposure as G5 did — a word like "another" or "noted" could
    previously be misread as explicit negation/limit-framing, exempting an unqualified
    state-of-the-art claim from G3. Must still fire post-fix (word-boundary matching)."""

    bad = (
        "Reviewers noted that the pymdp agent is state-of-the-art for sparse "
        "waypoint prediction, a result achieved in another domain."
    )
    violations = check_performance_claims(bad)
    assert any(v.guard == "G3-performance-claim" for v in violations), bad


def test_g3_catches_leadership_paraphrases_not_only_verbatim_sota() -> None:
    """RedTeam MEDIUM: the verbatim allow-list was vacuous; leadership-class
    paraphrases must fire (negative control), and token presence cannot launder
    an assertive superiority verb."""
    for bad in (
        "Our active-inference model outperforms every baseline on the locked task.",
        "This is a new benchmark record for sparse waypoint prediction.",
        "The agent surpasses all sequence baselines.",
    ):
        assert any(
            v.guard == "G3-performance-claim" for v in check_performance_claims(bad)
        ), bad
    # a leadership verb bound only to a gate token (no limit framing) still fires
    laundered = "Our model outperforms all baselines, see {{SOTA_CLAIM_GATE_STATUS}}."
    assert any(
        v.guard == "G3-performance-claim" for v in check_performance_claims(laundered)
    )
    assert any(
        v.guard == "G3-performance-claim"
        for v in check_performance_claims(
            "The current state-of-the-art performance status is {{SOTA_CLAIM_GATE_STATUS}}: "
            "this work is state-of-the-art for sparse waypoint prediction."
        )
    )
    # but an honest, future-scoped leadership statement is allowed
    future = (
        "A future method would need to outperform every baseline before any such claim."
    )
    assert check_performance_claims(future) == []


def test_g3_blocks_global_sota_from_regime_qualified_result() -> None:
    bad = (
        "The regime-qualified synthetic SOTA result means active inference is "
        "state-of-the-art for sparse waypoint prediction."
    )

    assert any(v.guard == "G3-performance-claim" for v in check_performance_claims(bad))


def test_g4_blocks_unbounded_active_inference_help_claims() -> None:
    bad = "Active inference helps under partial observability."

    violations = check_active_inference_help_claims(bad)

    assert any(v.guard == "G4-active-inference-help" for v in violations)


def test_g4_catches_positive_advantage_survival_wording() -> None:
    bad = "The active-inference advantage survives in real mobility traces."

    violations = check_active_inference_help_claims(bad)

    assert any(v.guard == "G4-active-inference-help" for v in violations)


def test_g4_catches_broad_active_inference_benefit_paraphrases() -> None:
    bad = "The active-inference predictor earns a genuine predictive advantage."

    violations = check_active_inference_help_claims(bad)

    assert any(v.guard == "G4-active-inference-help" for v in violations)


def test_g4_blocks_active_inference_uniqueness_wording() -> None:
    for bad in (
        "The noisy-emission gain is unique to active inference.",
        "The noisy-emission gain is unique to active inference, unlike HMM or particle filters.",
        "Only active inference provides the noisy-emission gain.",
        "This is an active-inference-specific superiority result.",
    ):
        violations = check_active_inference_help_claims(bad)
        assert any(v.guard == "G4-active-inference-help" for v in violations), bad


def test_g4_allows_comparator_bounded_uniqueness_negation() -> None:
    ok = (
        "The noisy-emission gain is not unique to active inference; it is shared "
        "with HMM, particle, and state-space belief filters."
    )
    also_ok = (
        "The noisy-emission mechanism is not specific to active inference; it "
        "belongs to a broader hidden Markov and state-space belief-filter family."
    )

    assert check_active_inference_help_claims(ok) == []
    assert check_active_inference_help_claims(also_ok) == []


def test_g4_allows_token_bound_scoped_active_inference_help_claims() -> None:
    ok = (
        "Active inference helps only in the partial-observability sweep "
        "({{PARTIAL_OBS_MAX_NOISE_ADVANTAGE}} nats), a synthetic-latent software "
        "measurement under a known generative emission that does not beat the "
        "oracle-fed baseline and is not a real-mobility claim."
    )

    assert check_active_inference_help_claims(ok) == []


def test_g5_catches_flagrant_deployment_surveillance_claims() -> None:
    """RedTeam HIGH: the Standing Claim Locks name deployment/surveillance/
    operational-tasking/real-mobility-transfer language as the class the
    project most explicitly forbids, but before G5 there was zero
    machine-checkable coverage for it. Each of these probe sentences must fire."""

    for bad in (
        "This system is ready for operational deployment in national-security "
        "surveillance programs.",
        "The pipeline could enable real-time tasking of location-collection assets.",
        "These results pave the way toward fielding this system for actual "
        "mobility surveillance.",
    ):
        violations = check_deployment_surveillance_claims(bad)
        assert any(v.guard == "G5-deployment-surveillance-claim" for v in violations), (
            bad
        )


def test_g5_incidental_substring_of_not_does_not_launder_a_real_claim() -> None:
    """Forge cross-vendor HIGH-3 (session 27, 2026-07-02): the bare "not" negation marker
    previously matched as a raw substring, so words like "another", "annotation", and "noted"
    were misread as explicit negation, exempting a genuinely forbidden claim from G5. Reproduced
    empirically pre-fix; each of these must still fire post-fix (word-boundary matching)."""

    for bad in (
        "The pipeline enables real-time tasking of collection assets in another domain.",
        "This annotation rich system is ready for operational deployment in "
        "national-security surveillance programs.",
        "It should be noted that these results pave the way toward fielding this "
        "system for actual mobility surveillance.",
    ):
        violations = check_deployment_surveillance_claims(bad)
        assert any(v.guard == "G5-deployment-surveillance-claim" for v in violations), (
            bad
        )


def test_g5_allows_explicitly_negated_deployment_surveillance_claim() -> None:
    """A properly-negated sentence — the Standing Claim Locks' own phrasing --
    must not be flagged."""

    ok = (
        "Synthetic performance, even if improved, cannot unlock national-security "
        "deployment, surveillance-readiness, or operational tasking claims, and "
        "this benchmark remains blocked from any real mobility transfer."
    )
    assert check_deployment_surveillance_claims(ok) == []


def test_g5_real_manuscript_has_zero_deployment_surveillance_violations() -> None:
    """Negative-control-free-manuscript sanity check: the project's own honest,
    already-bounded prose in manuscript/*.md must not trip G5 (no false
    rejections), matching the specialist's confirmation that current prose is
    already clean of unnegated deployment/surveillance language."""

    corpus = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(MANUSCRIPT.glob("*.md"))
    )
    violations = check_deployment_surveillance_claims(corpus)
    assert violations == [], [v.sentence for v in violations]


def test_contract_reach_is_honestly_bounded() -> None:
    """The integrity test of the contract itself: it must state its own limited
    reach. A static linter cannot catch failures that only appear on execution,
    and the project's history proves most of its honesty failures were exactly
    those — so the contract names them as out of scope rather than overclaiming."""

    # The static contract does not catch MORE failure classes than it leaves to
    # the execution-level harness — the honest, non-overclaiming bound. G5
    # (deployment/surveillance claims) brought the two counts to parity; adding
    # a statically-catchable class is legitimate growth, not overclaiming, as
    # long as it never overtakes the execution-only side.
    assert len(STATICALLY_CATCHABLE_CLASSES) >= 1
    assert len(EXECUTION_ONLY_CLASSES) >= len(STATICALLY_CATCHABLE_CLASSES)
    # The five guards implemented (G1-G5) correspond to the five statically-catchable classes.
    assert len(STATICALLY_CATCHABLE_CLASSES) == 5
