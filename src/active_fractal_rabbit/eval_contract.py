"""Machine-checkable honest-benchmark evaluation contract (claim-level linter).

FractalRabbit was published by NSA Research so sparse-mobility-analytics
algorithms can be developed and compared on a controlled, reproducible,
ground-truth-bearing synthetic testbed. The program's core methodological hazard
is the one this project spent seven sessions learning to detect: a persuasive
number that does not reflect recoverable structure — a win claimed against the
wrong null, in the wrong regime, on a self-selected coverage subset, or resting
on an undisclosed cue.

This module lifts the project's negative-control philosophy one level, from the
pipeline to the *claim*. It is a static linter over the manuscript prose and its
token→artifact bindings, encoding the honesty rules as falsifiable guards:

- **G1 — null-regime guard.** The session-6 trap, made machine-checkable. A
  sentence asserting *positive* skill against the persistence null must cite a
  token bound to the gap-filling regime, never the location-blind regime (where
  persistence receives the true previous cell and the filter does not). A location-blind
  skill token may appear only inside an explicit *limit* framing.
- **G2 — cue-disclosure guard.** Any recovery/skill claim that leans on the
  speed cue requires the manuscript to disclose, somewhere, that speed is derived
  from inter-report displacement (a strong but disclosed kinematic cue), so no
  recovery claim rests on a smuggled feature.
- **G3 — performance-claim guard.** Positive state-of-the-art or
  performance-leading language must be negated, future-scoped, or bound to an
  explicit model-gate status token. The manuscript may discuss what would be
  needed to reach that bar; it may not imply that the current run has cleared it
  unless the gate artifact says so.
- **G4 — active-inference-help guard.** Broad "active inference helps" language
  must stay tied to the partial-observability evidence and carry the known-
  emission, oracle-boundary, and synthetic-only qualifiers in the same sentence.
  Wording that makes the mechanism unique to active inference must be explicitly
  negated or bounded against HMM, particle, state-space, or other belief filters.
- **G5 — deployment/surveillance-claim guard.** The Standing Claim Locks in
  `TODO.md` name this as the class the project most explicitly forbids: synthetic
  SOTA progress cannot unlock national-security deployment, surveillance-
  readiness, operational-tasking, or real-mobility-transfer language. A sentence
  containing deployment/surveillance/tasking/real-mobility wording must carry an
  explicit negation ("not", "cannot", "blocked", "does not support", "remains
  blocked", "never") in the same sentence.

Scope is deliberately honest: this is the CLAIM-LEVEL layer. It complements, and
does not replace, the EXECUTION-LEVEL harness (negative controls, full-trajectory
scoring tests, the binding gate, the no-hardcoded-numbers guard), which catches
the honesty failures that can only be seen by running the pipeline. The contract
ingests nothing but the project's own manuscript and artifacts — a pure
evaluation/oversight layer, never an instrument over real data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


def _any_marker_at_word_boundary(text: str, markers: tuple[str, ...]) -> bool:
    """Word-boundary marker match, not raw substring containment.

    Forge cross-vendor HIGH-3 (session 27, 2026-07-02): a bare ``"not"`` in a limit/negation
    marker list matched as a raw substring turns "another domain", "annotation", and "noted"
    into false negation hits, letting a genuinely forbidden claim slip through unflagged
    (reproduced empirically). ``"no "`` (trailing space) has the identical class of exposure,
    e.g. inside "casino ". Every negation/limit marker list in this module must route through
    this helper rather than ``marker in lowered``. Markers containing regex metacharacters are
    escaped; multi-word markers still require the phrase's own first/last characters to sit at
    a word boundary, so partial-word containment inside a longer token is excluded either way.
    """
    for marker in markers:
        pattern = r"\b" + re.escape(marker.strip()) + r"\b"
        if re.search(pattern, text):
            return True
    return False


# Skill tokens and the inference regime each is bound to. The naming contract in
# manuscript_variables.py is the source of truth: GAP_FILL_* tokens resolve to
# the gap_filling sibling of anticipatory_analysis.json; BLIND_LIMIT_* to the
# location_blind_limit sibling.
SKILL_TOKEN_REGIME: dict[str, str] = {
    "GAP_FILL_FLAT_SKILL": "gap_filling",
    "GAP_FILL_SCRP_SKILL": "gap_filling",
    "BLIND_LIMIT_FLAT_SKILL": "location_blind_limit",
    "BLIND_LIMIT_SCRP_SKILL": "location_blind_limit",
}

# Markers that frame a persistence-skill sentence as a LIMIT (honest negative)
# rather than a positive achievement. A location-blind skill token is only
# allowed in a sentence that carries one of these.
LIMIT_MARKERS = (
    "information-asymmetry",
    "oracle ceiling",
    "weaker information set",
    "almost entirely",
    "little beyond",
    "weakly negative",
    "is in fact",
    "not beaten",
    "is loose",
    "weak null",
    "stronger information set",
    "does not beat",
)

WRONG_REGIME_POSITIVE_MARKERS = (
    "beats",
    "outperforms",
    "wins",
    "improves",
    "exceeds",
    "surpasses",
    "reaches skill",
    "gains skill",
)

# Tokens whose claim leans on the (displacement-derived) speed cue.
SPEED_CUE_TOKENS = ("GAP_FILL_SCRP_SKILL",)

# Phrases that satisfy the speed-cue disclosure requirement anywhere in the corpus.
CUE_DISCLOSURE_MARKERS = (
    "derived from inter-report displacement",
    "inter-report displacement",
    "displacement-derived",
    "derived from the movement",
)

_TOKEN_RE = re.compile(r"\{\{([A-Z_]+)\}\}")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# Abbreviations whose internal periods must NOT end a sentence — otherwise a
# limit-framed clause ("information-asymmetry ... e.g. ... {{BLIND_LIMIT_*}}") splits
# and the limit marker is separated from the token, falsely tripping G1. (This
# robustness gap was surfaced by the contract's own adversarial review.)
_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "et al.",
    "vs.",
    "cf.",
    "Fig.",
    "Eq.",
    "etc.",
    "Dr.",
    "No.",
    "Inc.",
    "approx.",
)
_ABBREV_SENTINEL = "\x00"

# Honest reach of this static contract, applied to the project's own history.
# Of the project's documented honesty failures, only the wrong-regime class is
# statically catchable from prose + token bindings; the others were caught only
# by EXECUTING the analysis. The contract states this bound rather than claiming
# to encode the whole discipline — prototype-before-claim, applied to itself.
STATICALLY_CATCHABLE_CLASSES = (
    "wrong-regime null (positive skill claimed against an oracle null in the wrong regime)",
    "undisclosed cue (a recovery claim leaning on a cue not disclosed in the corpus)",
    "unsupported state-of-the-art or performance-leading wording without a cleared gate",
    "unbounded active-inference-help or uniqueness wording without partial-observability, scope, and comparator boundaries",
    "un-negated deployment, surveillance-readiness, operational-tasking, or real-mobility-transfer language",
)
EXECUTION_ONLY_CLASSES = (
    "a learned model that underperforms its own baseline (Dirichlet-pB, session 7)",
    "headline numbers that do not reproduce on real data (hierarchical/latent-mode, session 7)",
    "a fabricated library call (model-comparison candidate, session 7)",
    "a scoring artifact visible only on full re-scoring (own-gaps vs full-trajectory, sessions 6-7)",
    "external simulator and render-readiness failures that only appear when the full artifact pipeline runs",
)

PERFORMANCE_CLAIM_MARKERS = (
    "state-of-the-art",
    "state of the art",
    "performance-leading",
    "performance leading",
)

# Assertive leadership paraphrases. The verbatim allow-list above was vacuous: it
# only matched strings authors already self-censor. These are active assertions of
# superiority — mere gate-token *presence* cannot launder them; they require
# explicit limit-framing (negated / future-scoped).
PERFORMANCE_LEADERSHIP_MARKERS = (
    "outperform",
    "best-in-class",
    "new benchmark record",
    "benchmark record",
    "record-setting",
    "strongest result",
    "predictive leadership",
    "surpass",
    "beats every baseline",
    "beats all baselines",
)

ASSERTIVE_SOTA_MARKERS = (
    "is state-of-the-art",
    "is state of the art",
    "are state-of-the-art",
    "are state of the art",
    "this work is state-of-the-art",
    "this work is state of the art",
    "the model is state-of-the-art",
    "the model is state of the art",
    "state-of-the-art for",
    "state of the art for",
)

PERFORMANCE_LIMIT_MARKERS = (
    "not",
    "no ",
    "without",
    "unless",
    "until",
    "before",
    "before any",
    "required before",
    "what would",
    "would move",
    "needs",
    "future",
    "should be",
    "not_demonstrated",
)

PERFORMANCE_GATE_TOKENS = (
    "PYMDP_SOTA_STATUS",
    "MODEL_GATE_MATRIX_STATUS",
    "MODEL_GATE_SEQUENCE_EXTERNAL_GATE",
    "SOTA_CLAIM_GATE_STATUS",
    "AIF_REGIME_SOTA_STATUS",
)

ACTIVE_INFERENCE_HELP_MARKERS = (
    "active inference helps",
    "active-inference helps",
    "active-inference advantage",
    "active inference advantage",
    "active inference earns a mechanism-level advantage",
    "mechanism-level advantage exactly where",
)

ACTIVE_INFERENCE_SUBJECT_MARKERS = (
    "active inference",
    "active-inference",
)

ACTIVE_INFERENCE_POSITIVE_MARKERS = (
    "earns",
    "benefits",
    "survives",
)

ACTIVE_INFERENCE_NEGATIVE_MARKERS = (
    "no active-inference advantage",
    "no active inference advantage",
    "no robust, confound-controlled active-inference advantage",
)

ACTIVE_INFERENCE_UNIQUENESS_MARKERS = (
    "unique to active inference",
    "unique to active-inference",
    "uniquely active inference",
    "uniquely active-inference",
    "only active inference",
    "only active-inference",
    "active-inference-specific",
    "active inference-specific",
    "exclusive active-inference",
    "exclusive active inference",
    "proprietary to active inference",
    "proprietary to active-inference",
)

ACTIVE_INFERENCE_UNIQUENESS_NEGATION_MARKERS = (
    "not unique",
    "not conceptually unique",
    "not exclusive",
    "not proprietary",
    "not specific to active inference",
    "not specific to active-inference",
)

ACTIVE_INFERENCE_SHARED_MECHANISM_MARKERS = (
    "shared with",
    "also shared with",
    "also applies to",
    "one member of",
    "one uncertainty-aware member",
    "belongs to a broader",
)

ACTIVE_INFERENCE_COMPARATOR_MARKERS = (
    "hmm",
    "hidden markov",
    "particle",
    "state-space",
    "state space",
    "belief-marginalizing",
    "belief-marginalizing",
    "soft filter",
)

PARTIAL_OBSERVABILITY_TOKENS = (
    "PARTIAL_OBS_N_CELLS",
    "PARTIAL_OBS_CLEAN_ADVANTAGE",
    "PARTIAL_OBS_MAX_EPS",
    "PARTIAL_OBS_MAX_NOISE_ADVANTAGE",
    "PARTIAL_OBS_MAX_NOISE_SD",
    "PARTIAL_OBS_MAX_NOISE_MIN",
    "PARTIAL_OBS_N_REALIZATIONS",
    "PARTIAL_OBS_NOISE_SLOPE",
    "PARTIAL_OBS_NOISE_R2",
    "PARTIAL_OBS_ZERO_WHEN_CLEAN",
    "PARTIAL_OBS_MONOTONE",
    "PARTIAL_OBS_CROSS_DRAW_COUNT",
    "PARTIAL_OBS_CROSS_DRAW_MAX_SD",
    "PARTIAL_OBS_CROSS_DRAW_SIGN_STABLE",
)

KNOWN_EMISSION_MARKERS = (
    "known generative emission",
    "known symmetric emission",
    "known emission",
    "emission is known",
    "emission known",
)

ORACLE_BOUNDARY_MARKERS = (
    "does not beat the oracle",
    "not beat the oracle",
    "loses to the oracle",
    "still loses to the oracle",
    "never an oracle-beating",
    "not oracle-beating",
    "oracle-fed baseline",
)

SYNTHETIC_SCOPE_MARKERS = (
    "synthetic",
    "software measurement",
    "not empirical",
    "not a real-mobility",
    "not real-mobility",
    "pinned generator",
)

# G5 markers: deployment / surveillance / operational-tasking / real-mobility-
# transfer language the Standing Claim Locks (TODO.md) most explicitly forbid.
# Deliberately compound (e.g. "operational tasking", not bare "tasking") so the
# guard does not fire on innocuous uses of the individual words elsewhere in the
# corpus (e.g. "task the model", "surveillance-theory work").
DEPLOYMENT_SURVEILLANCE_MARKERS = (
    "operational deployment",
    "ready for deployment",
    "ready for operational deployment",
    "national-security deployment",
    "surveillance-ready",
    "surveillance readiness",
    "surveillance-readiness",
    "surveillance programs",
    "surveillance program",
    "operational tasking",
    "real-time tasking",
    "tasking of location-collection",
    "location-collection",
    "collection assets",
    "enable collection",
    "field this system",
    "fielding this system",
    "actual mobility",
    "real mobility",
    "real-mobility",
    "real-world mobility",
    "mobility surveillance",
)

# Explicit-negation markers. Anywhere-in-sentence, mirroring G3's is_limit_framed
# style: the guard is a claim-level linter over prose, not a clause-scoped parser.
DEPLOYMENT_SURVEILLANCE_LIMIT_MARKERS = (
    "not",
    "no ",
    "cannot",
    "blocked",
    "does not support",
    "remains blocked",
    "never",
    "none",
)


@dataclass(frozen=True)
class Violation:
    guard: str
    token: str
    reason: str
    sentence: str


def _sentences(text: str) -> list[str]:
    protected = text
    for abbrev in _ABBREVIATIONS:
        protected = protected.replace(abbrev, abbrev.replace(".", _ABBREV_SENTINEL))
    parts = _SENTENCE_RE.split(protected)
    return [p.replace(_ABBREV_SENTINEL, ".").strip() for p in parts if p.strip()]


def _prose_without_tokens(sentence: str) -> str:
    """Sentiment must be read from the PROSE, not the token names (a token like
    BLIND_LIMIT_SCRP_SKILL contains the substring 'limit' and would otherwise
    spoof the limit-framing check)."""

    return _TOKEN_RE.sub(" ", sentence).lower()


def _has_positive_wrong_regime_claim(sentence: str, token: str) -> bool:
    """Look for a positive skill verb in the local clause that introduces a token.

    The whole sentence can mention a valid gap-filling win while also reporting
    location-blind limit tokens. The wrong-regime guard should read the local
    clause around the location-blind token, not treat every later positive verb
    as attached to every token in the sentence.
    """

    lowered = sentence.lower()
    token_ref = "{{" + token.lower() + "}}"
    token_index = lowered.find(token_ref)
    if token_index < 0:  # pragma: no cover - token came from this sentence
        return False
    prefix = _TOKEN_RE.sub(" ", lowered[:token_index])
    clause_start = max(prefix.rfind(boundary) for boundary in (".", ";", "\n"))
    local_clause = prefix[clause_start + 1 :]
    return any(marker in local_clause for marker in WRONG_REGIME_POSITIVE_MARKERS)


def check_null_regime(text: str) -> list[Violation]:
    """G1: a positive persistence-skill claim must not cite a location-blind token."""

    violations: list[Violation] = []
    for sentence in _sentences(text):
        tokens = _TOKEN_RE.findall(sentence)
        skill_tokens = [t for t in tokens if t in SKILL_TOKEN_REGIME]
        if not skill_tokens:
            continue
        prose = _prose_without_tokens(sentence)
        is_limit_framed = any(marker in prose for marker in LIMIT_MARKERS)
        for token in skill_tokens:
            if SKILL_TOKEN_REGIME[token] == "location_blind_limit" and (
                not is_limit_framed or _has_positive_wrong_regime_claim(sentence, token)
            ):
                violations.append(
                    Violation(
                        guard="G1-null-regime",
                        token=token,
                        reason=(
                            "location-blind skill token presented as a positive claim; "
                            "persistence is an oracle null in the location-blind regime "
                            "and cannot be beaten there — cite a gap_filling token or add "
                            "an explicit limit framing"
                        ),
                        sentence=sentence.strip(),
                    )
                )
    return violations


def check_cue_disclosure(text: str) -> list[Violation]:
    """G2: a claim leaning on the speed cue requires speed-derivation disclosure."""

    uses_speed_cue = any(token in text for token in SPEED_CUE_TOKENS)
    if not uses_speed_cue:
        return []
    if any(marker in text.lower() for marker in CUE_DISCLOSURE_MARKERS):
        return []
    sentence = next(
        (s for s in _sentences(text) if any(t in s for t in SPEED_CUE_TOKENS)), ""
    )
    return [
        Violation(
            guard="G2-cue-disclosure",
            token=SPEED_CUE_TOKENS[0],
            reason=(
                "a recovery claim leans on the speed cue but the manuscript does not "
                "disclose that speed is derived from inter-report displacement"
            ),
            sentence=sentence.strip(),
        )
    ]


def check_performance_claims(text: str) -> list[Violation]:
    """G3: SOTA/performance-leading wording must be negated or gate-bound."""

    violations: list[Violation] = []
    for sentence in _sentences(text):
        lowered = sentence.lower()
        has_claim = any(marker in lowered for marker in PERFORMANCE_CLAIM_MARKERS)
        has_leadership = any(
            marker in lowered for marker in PERFORMANCE_LEADERSHIP_MARKERS
        )
        has_assertive_sota = any(marker in lowered for marker in ASSERTIVE_SOTA_MARKERS)
        if not (has_claim or has_leadership):
            continue
        tokens = set(_TOKEN_RE.findall(sentence))
        is_gate_bound = bool(tokens.intersection(PERFORMANCE_GATE_TOKENS))
        is_limit_framed = _any_marker_at_word_boundary(
            lowered, PERFORMANCE_LIMIT_MARKERS
        )
        # Status-style markers may be honestly gate-bound (reporting the gated status);
        # assertive leadership verbs may NOT — token presence cannot launder an
        # active superiority claim, so they require explicit limit-framing.
        if has_leadership or has_assertive_sota:
            exempt = is_limit_framed
        else:
            exempt = is_gate_bound or is_limit_framed
        if exempt:
            continue
        violations.append(
            Violation(
                guard="G3-performance-claim",
                token="",
                reason=(
                    "state-of-the-art / performance-leading wording (or a leadership-class "
                    "paraphrase) must be explicitly negated, future-scoped, or — for status "
                    "wording only — bound to a cleared model-gate status token"
                ),
                sentence=sentence.strip(),
            )
        )
    return violations


def check_deployment_surveillance_claims(text: str) -> list[Violation]:
    """G5: deployment/surveillance/operational-tasking/real-mobility-transfer
    wording must be explicitly negated — the Standing Claim Locks class."""

    violations: list[Violation] = []
    for sentence in _sentences(text):
        lowered = sentence.lower()
        has_claim = any(marker in lowered for marker in DEPLOYMENT_SURVEILLANCE_MARKERS)
        if not has_claim:
            continue
        is_limit_framed = _any_marker_at_word_boundary(
            lowered, DEPLOYMENT_SURVEILLANCE_LIMIT_MARKERS
        )
        if is_limit_framed:
            continue
        violations.append(
            Violation(
                guard="G5-deployment-surveillance-claim",
                token="",
                reason=(
                    "deployment / surveillance / operational-tasking / "
                    "real-mobility-transfer wording must be explicitly negated "
                    "(e.g. 'not', 'cannot', 'blocked', 'does not support', "
                    "'remains blocked', 'never') — synthetic SOTA progress cannot "
                    "unlock this language per the Standing Claim Locks"
                ),
                sentence=sentence.strip(),
            )
        )
    return violations


def _has_active_inference_help_claim(lowered: str) -> bool:
    if any(marker in lowered for marker in ACTIVE_INFERENCE_HELP_MARKERS):
        return True
    return any(
        marker in lowered for marker in ACTIVE_INFERENCE_SUBJECT_MARKERS
    ) and any(marker in lowered for marker in ACTIVE_INFERENCE_POSITIVE_MARKERS)


def _is_negative_active_inference_help_claim(lowered: str) -> bool:
    if any(marker in lowered for marker in ACTIVE_INFERENCE_NEGATIVE_MARKERS):
        return True
    return bool(
        re.search(
            r"\b(no|not|without)\b.{0,80}active[- ]inference.{0,80}"
            r"(advantage|help|helps|benefit|surviv|clear)",
            lowered,
        )
        or re.search(
            r"active[- ]inference.{0,80}\b(does not|doesn't|cannot|fails to)\b.{0,80}"
            r"(help|benefit|survive|clear)",
            lowered,
        )
    )


def _has_active_inference_uniqueness_claim(lowered: str) -> bool:
    return any(marker in lowered for marker in ACTIVE_INFERENCE_UNIQUENESS_MARKERS)


def _is_comparator_bounded_uniqueness_claim(lowered: str) -> bool:
    has_negated_uniqueness = any(
        marker in lowered for marker in ACTIVE_INFERENCE_UNIQUENESS_NEGATION_MARKERS
    )
    has_shared_mechanism = any(
        marker in lowered for marker in ACTIVE_INFERENCE_SHARED_MECHANISM_MARKERS
    )
    has_comparator = any(
        marker in lowered for marker in ACTIVE_INFERENCE_COMPARATOR_MARKERS
    )
    return has_negated_uniqueness or (has_shared_mechanism and has_comparator)


def check_active_inference_help_claims(text: str) -> list[Violation]:
    """G4: active-inference-help claims need evidence, scope, and comparator bounds."""

    violations: list[Violation] = []
    for sentence in _sentences(text):
        if sentence.lstrip().startswith("#"):
            continue
        lowered = sentence.lower()
        has_help_claim = _has_active_inference_help_claim(lowered)
        has_uniqueness_claim = _has_active_inference_uniqueness_claim(lowered)
        if not has_help_claim and not has_uniqueness_claim:
            continue
        if has_uniqueness_claim:
            if _is_comparator_bounded_uniqueness_claim(lowered):
                if not has_help_claim:
                    continue
            else:
                violations.append(
                    Violation(
                        guard="G4-active-inference-help",
                        token="",
                        reason=(
                            "active-inference uniqueness wording must be explicitly "
                            "negated or bounded against HMM, particle, state-space, or "
                            "other belief-filter comparators"
                        ),
                        sentence=sentence.strip(),
                    )
                )
                continue
        if _is_negative_active_inference_help_claim(lowered):
            continue
        tokens = set(_TOKEN_RE.findall(sentence))
        has_partial_token = bool(tokens.intersection(PARTIAL_OBSERVABILITY_TOKENS))
        has_known_emission = any(marker in lowered for marker in KNOWN_EMISSION_MARKERS)
        has_oracle_boundary = any(
            marker in lowered for marker in ORACLE_BOUNDARY_MARKERS
        )
        has_synthetic_scope = any(
            marker in lowered for marker in SYNTHETIC_SCOPE_MARKERS
        )
        if (
            has_partial_token
            and has_known_emission
            and has_oracle_boundary
            and has_synthetic_scope
        ):
            continue
        missing = []
        if not has_partial_token:
            missing.append("partial-observability token")
        if not has_known_emission:
            missing.append("known-emission qualifier")
        if not has_oracle_boundary:
            missing.append("oracle-boundary qualifier")
        if not has_synthetic_scope:
            missing.append("synthetic-only qualifier")
        violations.append(
            Violation(
                guard="G4-active-inference-help",
                token="",
                reason=(
                    "active-inference-help wording must be bound in the same sentence to "
                    + ", ".join(missing)
                ),
                sentence=sentence.strip(),
            )
        )
    return violations


@dataclass(frozen=True)
class ContractReport:
    violations: tuple[Violation, ...]

    @property
    def passed(self) -> bool:
        return not self.violations


def run_contract(manuscript_dir: str | Path) -> ContractReport:
    """Run all claim-level guards over a manuscript directory's markdown corpus."""

    corpus = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(Path(manuscript_dir).glob("*.md"))
    )
    violations: list[Violation] = []
    violations.extend(check_null_regime(corpus))
    violations.extend(check_cue_disclosure(corpus))
    violations.extend(check_performance_claims(corpus))
    violations.extend(check_active_inference_help_claims(corpus))
    violations.extend(check_deployment_surveillance_claims(corpus))
    return ContractReport(violations=tuple(violations))
