"""Phase 2K contextual reconstruction core (isolated, stdlib-only).

Phase 2K freezes the Phase 2J reviewed windows as inputs and generates
human-review artifacts:

  A  exact isolated raw Bronze
  B  exact target mechanically cleaned (target-only, context-free repairs)
  C  exact ordered enlarged context with the target clearly delimited
  D  separately stored contextual reconstruction + semantic polish passes

The module is intentionally isolated from the Phase 2J pipeline: it reads the
frozen Phase 2J artifacts and a read-only SQLite transcript DB, but never
imports or edits Phase 2J code/data, never runs Phase 2J scoring, and never
overwrites Bronze.  All provider interaction is injected through a ``chat``
callable so the focused test-suite stays offline; the CLI supplies
``core.llm.chat`` for live runs.

Pass 1 is text restoration only: its provider envelope contains
``clean_text``, explicit repairs/uncertainties, and provenance.  Entity,
pronoun, champion, and ability-ownership output is structurally rejected.
Those licensed resolutions appear only in the later reconstruction pass, and
semantic statements appear only in the still-later polish pass.  No final
strategy-ontology abstractions are emitted in any semantic output.

This module uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import subprocess
from typing import Any, Callable, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Versions and configuration
# ---------------------------------------------------------------------------

PIPELINE_VERSION = "phase2k-contextual-reconstruction-v7"
CONFIG_VERSION = "phase2k-config-v3"
INFERENCE_CONFIG_VERSION = "phase2k-inference-config-v2"
METADATA_ADAPTER_SCHEMA_VERSION = "phase2k-metadata-adapter-v1"
LEAGUE_VOCABULARY_SCHEMA_VERSION = "phase2k-league-lexical-vocabulary-v2"
METADATA_RELIABILITY_LEVELS = (
    "SUPPLIED_FACT",
    "HIGH",
    "MEDIUM",
    "LOW",
    "UNKNOWN",
)

FROZEN_INPUT_MANIFEST_SCHEMA_VERSION = "phase2k-frozen-input-manifest-v1"
RECORDS_SCHEMA_VERSION = "phase2k-reconstruction-records-v7"
HUMAN_PACKET_SCHEMA_VERSION = "phase2k-human-review-packet-v2"
HUMAN_MAPPING_SCHEMA_VERSION = "phase2k-human-review-mapping-v2"
BUILD_SUMMARY_SCHEMA_VERSION = "phase2k-build-summary-v5"
CONTEXT_SCHEMA_VERSION = "phase2k-context-v1"
PRESENTATION_SCHEMA_VERSION = "phase2k-review-presentation-v2"
HUMAN_SUMMARY_SCHEMA_VERSION = "phase2k-human-review-summary-v1"
TRANSFORMATION_SUMMARY_SCHEMA_VERSION = (
    "phase2k-transformation-audit-summary-v1"
)
COUNT_REPORT_SCHEMA_VERSION = "phase2k-count-report-v1"

MECHANICAL_PROMPT_VERSION = "phase2k-mechanical-cleanup-prompt-v4"
MECHANICAL_RESPONSE_SCHEMA_VERSION = "phase2k-mechanical-cleanup-response-v3"
MECHANICAL_CORRECTION_PROMPT_VERSION = (
    "phase2k-mechanical-cleanup-correction-prompt-v4"
)
SUFFICIENCY_PROMPT_VERSION = "phase2k-sufficiency-prompt-v2"
SUFFICIENCY_RESPONSE_SCHEMA_VERSION = "phase2k-sufficiency-response-v2"
SUFFICIENCY_CORRECTION_PROMPT_VERSION = (
    "phase2k-sufficiency-correction-prompt-v1"
)
SUFFICIENCY_NORMALIZED_SCHEMA_VERSION = "phase2k-sufficiency-normalized-v1"
RECONSTRUCTION_PROMPT_VERSION = "phase2k-reconstruction-prompt-v8"
RECONSTRUCTION_RESPONSE_SCHEMA_VERSION = "phase2k-reconstruction-response-v4"
RECONSTRUCTION_CORRECTION_PROMPT_VERSION = (
    "phase2k-reconstruction-correction-prompt-v7"
)
POLISH_PROMPT_VERSION = "phase2k-semantic-polish-prompt-v3"
POLISH_RESPONSE_SCHEMA_VERSION = "phase2k-semantic-polish-response-v2"
POLISH_CORRECTION_PROMPT_VERSION = (
    "phase2k-semantic-polish-correction-prompt-v3"
)

TEXT_RESTORATION_TASK_KIND = "TEXT_RESTORATION"
CONTEXTUAL_RECONSTRUCTION_TASK_KIND = "CONTEXTUAL_RECONSTRUCTION"
SEMANTIC_POLISH_TASK_KIND = "SEMANTIC_POLISH"

TRANSFORMATION_AUDIT_SCHEMA_VERSION = "phase2k-transformation-audit-packet-v2"
OPERATION_AUDIT_SCHEMA_VERSION = "phase2k-operation-audit-input-v1"
CLOSEOUT_STATUS_SCHEMA_VERSION = "phase2k-closeout-status-v2"

PHASE2J_MANIFEST_SCHEMA_VERSION = "phase2j-window-selection-manifest-v1"
PHASE2J_PACKET_SCHEMA_VERSION = "phase2j-endpoint-annotation-packet-v1"

RELEASE_GATE_AWAITING_REVIEW = "AWAITING_HUMAN_REVIEW"
RELEASE_GATE_REVIEWED = "REVIEWED"
RELEASE_GATE_LOCKED = "LOCKED"

TOKEN_FALLBACK_BOUND = 32
HARD_SEGMENT_CAP_PER_SIDE = 120
HARD_CHAR_CAP_PER_SIDE = 120_000
BOUNDED_LOCAL_EPISODE_SEGMENTS = 40

# Adaptive-loop stages.  ``radius`` labels follow the emitted context-radius
# entry names; the stage labels are the deterministic loop steps.
RADIUS_STAGES = (
    {"label": "r1", "radius": "r2", "previous": 2, "following": 2, "max": False},
    {"label": "r2", "radius": "r5", "previous": 5, "following": 5, "max": False},
    {"label": "r3", "radius": "r10", "previous": 10, "following": 10, "max": False},
    {
        "label": "r4_bounded_local_episode",
        "radius": "bounded_local_episode",
        "previous": BOUNDED_LOCAL_EPISODE_SEGMENTS,
        "following": BOUNDED_LOCAL_EPISODE_SEGMENTS,
        "max": True,
    },
)
RADIUS_ENTRY_LABELS = (
    "target_only",
    "r2",
    "r5",
    "r10",
    "bounded_local_episode",
)

SLOT_KEYS = (
    "principal_actors",
    "pronouns",
    "champion_identities",
    "ability_ownership",
    "core_action_event",
    "state_outcome",
    "condition",
    "consequence",
    "temporal_refs",
    "discourse_refs",
    "unresolved_asr",
)
REQUIRED_SLOT_KEYS = SLOT_KEYS  # SUFFICIENT requires every slot RESOLVED.

SLOT_STATUSES = (
    "RESOLVED",
    "UNKNOWN",
    "AMBIGUOUS",
    "MULTIPLE_CANDIDATES",
    "CONTEXT_INSUFFICIENT",
)
UNRESOLVED_SLOT_STATUSES = frozenset({"UNKNOWN", "CONTEXT_INSUFFICIENT"})

# The first five operation kinds are emitted blank in live builds; a human
# transformation auditor fills in decision/error-taxonomy fields later.
TRANSFORMATION_OPERATION_KINDS = (
    "MECHANICAL_REPAIR",
    "CONTEXTUAL_REPAIR",
    "ENTITY_BINDING",
    "PRONOUN_BINDING",
    "REFERENCE_BINDING",
    "ABILITY_BINDING",
    "POLISHED_STATEMENT",
)

COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION = (
    "phase2k-completed-transformation-audit-v2"
)

SUFFICIENCY_DECISIONS = (
    "SUFFICIENT",
    "NEED_MORE_PREVIOUS_CONTEXT",
    "NEED_MORE_FOLLOWING_CONTEXT",
    "NEED_BOTH",
    "MAX_CONTEXT_BUT_UNRESOLVED",
)
NEEDS_MORE_DECISIONS = frozenset({
    "NEED_MORE_PREVIOUS_CONTEXT",
    "NEED_MORE_FOLLOWING_CONTEXT",
    "NEED_BOTH",
})

CONFIDENCE_LEVELS = ("HIGH", "MEDIUM", "LOW")

MECHANICAL_REPAIR_TYPES = (
    "ASR_HOMOPHONE",
    "ASR_COLLATION",
    "PUNCTUATION",
    "CAPITALIZATION",
    "CONTRACTION_EXPANSION",
    "SPELLING",
    "WHITESPACE",
    "DOMAIN_LEXICAL",
    "DOMAIN_SPELLING",
)
# Ownership/entity/pronoun resolution is never a mechanical repair type.
FORBIDDEN_MECHANICAL_REPAIR_TYPES = frozenset({
    "ENTITY_RESOLUTION",
    "PRONOUN_RESOLUTION",
    "ABILITY_OWNERSHIP_RESOLUTION",
    "REFERENT_RESOLUTION",
    "COREFERENCE_RESOLUTION",
})

# Pass 1 uncertainties are explicit source spans; they never edit clean_text.
MECHANICAL_UNCERTAINTY_TYPES = (
    "ASR_ALTERNATIVES",
    "PUNCTUATION_UNCERTAIN",
    "DOMAIN_TOKEN_UNCERTAIN",
)

# Pass 1 uncertainty proposals are capped so the provider cannot flood the
# restoration envelope with manufactured alternatives for clear words.
MECHANICAL_UNCERTAINTY_CAP = 8

# Bounded provider correction policy: one initial attempt plus at most two
# correction attempts.  Validation stays strict; a failed response is never
# normalized as valid merely because it is close.
PROVIDER_MAX_CORRECTIONS = 2
# Stage-specific correction budgets: mechanical cleanup and semantic polish
# allow one initial attempt plus at most three corrections so a malformed
# intermediate correction response can be recovered; the sufficiency
# diagnostic keeps the global default.
MECHANICAL_MAX_CORRECTIONS = 3
POLISH_MAX_CORRECTIONS = 3
# Phase 2K v14 hardening: contextual reconstruction allows one initial
# attempt plus at most three corrections (four total calls) so strict
# array-only evidence_quotes and exact licensed-candidate corrections can
# be recovered.  Every other stage keeps its existing budget.
RECONSTRUCTION_MAX_CORRECTIONS = 3

# Structural keys/concepts that prove the provider tried semantic extraction
# during the mechanical Pass 1.  Generic prose words such as "event" are only
# rejected when they appear as a structural field, never inside repair text.
FORBIDDEN_SEMANTIC_EXTRACTION_KEYS = frozenset({
    "endpoints", "endpoint", "entities", "entity", "events", "event",
    "relations", "relation", "edges", "edge", "semantic_claims", "claims",
    "bindings", "binding", "resolved_entity", "champion_binding",
    "resolved_champion", "champion", "owner", "ability_owner", "abilities",
    "states", "outcomes", "semantic", "extraction",
})

# Contextual repairs are complete for every difference between the exact
# Bronze target and the clean source-faithful D transcript.  Closed types
# distinguish surface fixes from licensed semantic resolutions.
CONTEXTUAL_REPAIR_TYPES = (
    "PUNCTUATION",
    "CAPITALIZATION",
    "DUPLICATE",
    "FILLER",
    "DOMAIN_SPELLING",
    "CONTEXTUAL_ASR",
    "WHITESPACE",
    "ENTITY_RESOLUTION",
    "PRONOUN_RESOLUTION",
    "ABILITY_OWNERSHIP_RESOLUTION",
    "REFERENCE_RESOLUTION",
)
CONTEXTUAL_RESOLUTION_REPAIR_TYPES = frozenset({
    "ENTITY_RESOLUTION",
    "PRONOUN_RESOLUTION",
    "ABILITY_OWNERSHIP_RESOLUTION",
    "REFERENCE_RESOLUTION",
})
_CONTEXTUAL_REPAIR_BINDING_SLOTS = {
    "ENTITY_RESOLUTION": frozenset({"principal_actors", "champion_identities"}),
    "PRONOUN_RESOLUTION": frozenset({"pronouns"}),
    "ABILITY_OWNERSHIP_RESOLUTION": frozenset({"ability_ownership"}),
    "REFERENCE_RESOLUTION": frozenset({"discourse_refs", "temporal_refs"}),
}

REPAIR_CONFIDENCE_LEVELS = CONFIDENCE_LEVELS

BINDING_STATUSES = SLOT_STATUSES

UNSUPPORTED_REASON_TYPES = (
    "MODEL_INVENTION",
    "INFERRED_BEYOND_CONTEXT",
    "AMBIGUOUS_OWNERSHIP",
    "UNLICENSED_ENTITY",
    "METADATA_CONFLICT",
)

POLISH_STATEMENT_ATTESTATION_FIELDS = (
    "modality_preserved",
    "negation_preserved",
    "uncertainty_preserved",
)

# Closed semantic-support attestation for every polished statement.
POLISH_SUPPORT_MODES = (
    "UNCHANGED_EXACT",
    "EVIDENCE_PARAPHRASE",
    "RECONSTRUCTION_DERIVED",
)

# Closed transformation-audit error taxonomy required by the Notion metrics.
AUDIT_ERROR_TAXONOMY = (
    "ASR_REPAIR_CORRECT",
    "ASR_REPAIR_WRONG",
    "ASR_REPAIR_UNRESOLVED",
    "ENTITY_BIND_CORRECT",
    "ENTITY_BIND_WRONG",
    "ENTITY_BIND_UNRESOLVED",
    "ABILITY_OWNER_CORRECT",
    "ABILITY_OWNER_WRONG",
    "ABILITY_OWNER_UNRESOLVED",
    "PRONOUN_BIND_WRONG",
    "DISCOURSE_REFERENCE_UNRESOLVED",
    "CONTEXT_TOO_SHORT",
    "CONTEXT_EXPANDED_UNNECESSARILY",
    "UNCERTAINTY_ERASED",
    "NEGATION_CHANGED",
    "MODALITY_CHANGED",
    "CAUSALITY_INVENTED",
    "EVENT_INVENTED",
    "SOURCE_DETAIL_DROPPED",
    "OVERGENERALIZED",
    "OTHER",
)
AUDIT_OPERATION_DECISIONS = ("APPROVE", "REJECT", "AMBIGUOUS")

# Repair types counted as ASR work in the operation-level audit metrics.
ASR_AUDIT_REPAIR_TYPES = frozenset({
    "ASR_HOMOPHONE",
    "ASR_COLLATION",
    "CONTEXTUAL_ASR",
})

# Stages whose first failure is captured in the transformation audit.
WINDOW_GENERATION_STAGES = (
    "mechanical_cleanup",
    "adaptive_diagnostics",
    "reconstruction",
    "semantic_polish",
)

CLOSEOUT_STATUSES = (
    "WAITING_FOR_HUMAN_REVIEW",
    "WAITING_FOR_DOWNSTREAM",
    "CONTEXTUAL_POLISH_VALIDATED",
    "CONTEXT_ALONE_SUFFICIENT",
    "POLISH_UNSAFE_OVER_RECONSTRUCTING",
    "NO_MATERIAL_REPRESENTATION_GAIN",
    "INCONCLUSIVE",
)

FORBIDDEN_ABSTRACTION_TERMS = (
    "access",
    "continuity",
    "initiative",
    "role transfer",
    "role_transfer",
    "conversion",
    "tempo",
    "priority",
    "pressure",
)
_COMMON_CAPITALIZED_WORDS = frozenset({
    "The", "A", "An", "This", "That", "These", "Those", "It", "Its", "He",
    "His", "Him", "She", "Her", "They", "Them", "Their", "We", "Our", "You",
    "Your", "When", "If", "But", "And", "Or", "So", "Then", "After",
    "Before", "Because", "While", "Coach", "Coaching", "Now", "There",
    "Here", "One", "Once", "Again", "Actually", "Maybe", "Wait", "Look",
    "Ok", "Okay", "Yes", "No",
})
_FORBIDDEN_ABSTRACTION_PATTERNS = tuple(
    re.compile(r"(?<![A-Za-z])" + re.escape(term) + r"(?![A-Za-z])")
    for term in FORBIDDEN_ABSTRACTION_TERMS
)

# Pass 1 rationale guard.  Strategic ontology words are allowed in a
# mechanical lexical rationale (for example "'prio' means 'priority', a
# standard League term"), because the strategy-ontology ban belongs to the
# later semantic stages.  What remains forbidden in Pass 1 is a semantic
# endpoint/extraction list disguised as rationale: structural labels such as
# "entities:", "bindings:", "events:", "ability owner:", or extraction
# phrasing that enumerates semantic endpoints.  Generic prose words remain
# allowed; only list-style extraction markers are rejected.
SEMANTIC_ENDPOINT_LIST_LABELS = (
    "endpoints", "endpoint", "entities", "entity", "events", "event",
    "relations", "relation", "edges", "edge", "semantic claims", "claims",
    "bindings", "binding", "resolved entity", "resolved entities",
    "resolved champion", "resolved champions", "champion binding",
    "champion bindings", "ability owner", "ability owners", "owner",
    "semantic extraction", "extraction",
)
_SEMANTIC_ENDPOINT_LIST_PATTERNS = tuple(
    re.compile(
        r"(?<![A-Za-z])" + re.escape(label) + r"(?![A-Za-z])\s*:",
        re.IGNORECASE,
    )
    for label in SEMANTIC_ENDPOINT_LIST_LABELS
) + (
    re.compile(
        r"(?<![A-Za-z])(?:extract(?:ed|ing)?|semantic)\s+"
        r"(?:endpoints?|entities?|events?|relations?|edges?|claims?|"
        r"bindings?|extraction)",
        re.IGNORECASE,
    ),
)


def _contains_semantic_endpoint_list(text: str) -> bool:
    """True when rationale text enumerates semantic extraction endpoints.

    The check is intentionally narrow: a lexical definition such as
    "'prio' is a mishearing of 'prio' (priority), a standard League term"
    is allowed, while "entities: Lux, R; bindings: ..." or "extracted
    entities Lux and R" is rejected as a semantic endpoint list disguised
    as rationale.
    """
    return any(pattern.search(text) for pattern in _SEMANTIC_ENDPOINT_LIST_PATTERNS)

HUMAN_SCORE_FIELDS = (
    "coached_actor",
    "opponent_entity",
    "pronouns",
    "ability_ownership",
    "core_action",
    "condition",
    "consequence",
    "causality",
    "standalone_coaching_claim",
    "asr_repair_correctness",
    "entity_binding_correctness",
    "meaning_preservation",
    "unsupported_invention",
    "remaining_ambiguity",
)
HUMAN_SCORE_MIN = 0
HUMAN_SCORE_MAX = 5
NOT_APPLICABLE = "NOT_APPLICABLE"
NOT_APPLICABLE_SENTINELS = frozenset({NOT_APPLICABLE})

# Lower raw scores are better for these fields; every other field is
# higher-is-better.  Aggregate composites normalize lower-is-better fields so
# that a single "higher is better" 0-5 scale is used everywhere.
LOWER_IS_BETTER_SCORE_FIELDS = frozenset({
    "unsupported_invention",
    "remaining_ambiguity",
})

HUMAN_SCORE_RUBRIC = {
    "coached_actor": {
        "description": (
            "How clearly the presented text supports recovering who is being "
            "coached/advised.  Focus on content recoverability, not grammar."
        ),
        "direction": "higher_is_better",
        "not_applicable_allowed": True,
    },
    "opponent_entity": {
        "description": (
            "How clearly the presented text supports recovering the "
            "opponent or referenced entity.  Focus on content "
            "recoverability, not grammar."
        ),
        "direction": "higher_is_better",
        "not_applicable_allowed": True,
    },
    "pronouns": {
        "description": (
            "How clearly the presented text supports resolving pronouns to "
            "their referents.  Higher = referents recoverable."
        ),
        "direction": "higher_is_better",
        "not_applicable_allowed": True,
    },
    "ability_ownership": {
        "description": (
            "How clearly the presented text supports recovering who owns "
            "each ability/resource.  Higher = ownership recoverable."
        ),
        "direction": "higher_is_better",
        "not_applicable_allowed": True,
    },
    "core_action": {
        "description": (
            "How clearly the presented text supports recovering the core "
            "action/event.  Higher = action recoverable."
        ),
        "direction": "higher_is_better",
        "not_applicable_allowed": True,
    },
    "condition": {
        "description": (
            "How clearly the presented text supports recovering the "
            "condition/trigger of the action.  Higher = condition "
            "recoverable."
        ),
        "direction": "higher_is_better",
        "not_applicable_allowed": True,
    },
    "consequence": {
        "description": (
            "How clearly the presented text supports recovering the "
            "consequence/result.  Higher = consequence recoverable."
        ),
        "direction": "higher_is_better",
        "not_applicable_allowed": True,
    },
    "causality": {
        "description": (
            "How clearly the presented text supports recovering the causal "
            "link between action and outcome.  Higher = causality "
            "recoverable."
        ),
        "direction": "higher_is_better",
        "not_applicable_allowed": True,
    },
    "standalone_coaching_claim": {
        "description": (
            "Whether the presented text supports a standalone coaching "
            "claim/advice that can be evaluated on its own.  5 = fully "
            "standalone; 0 = not recoverable as coaching advice."
        ),
        "direction": "higher_is_better",
        "not_applicable_allowed": False,
    },
    "asr_repair_correctness": {
        "description": (
            "Whether ASR errors evident in the presented text were "
            "correctly repaired to the extent the presented text does so.  "
            "Score the correctness of any repairs you observe; NOT_APPLICABLE "
            "only when no ASR issue is evident in the presented text."
        ),
        "direction": "higher_is_better",
        "not_applicable_allowed": True,
    },
    "entity_binding_correctness": {
        "description": (
            "Whether entity references are correctly bound to their "
            "referents in the presented text.  Higher = bindings correct."
        ),
        "direction": "higher_is_better",
        "not_applicable_allowed": True,
    },
    "meaning_preservation": {
        "description": (
            "Whether the presented text preserves the source meaning "
            "without loss or distortion.  Higher = meaning preserved."
        ),
        "direction": "higher_is_better",
        "not_applicable_allowed": True,
    },
    "unsupported_invention": {
        "description": (
            "How much content beyond the source is invented in the "
            "presented text.  LOWER is better: 0 = nothing invented, 5 = "
            "heavily invented."
        ),
        "direction": "lower_is_better",
        "not_applicable_allowed": False,
    },
    "remaining_ambiguity": {
        "description": (
            "How much ambiguity remains unresolved in the presented text.  "
            "LOWER is better: 0 = no remaining ambiguity, 5 = essentially "
            "unusable ambiguity."
        ),
        "direction": "lower_is_better",
        "not_applicable_allowed": True,
    },
}

ENTITY_ABILITY_COMPLETENESS_FIELDS = (
    "coached_actor",
    "opponent_entity",
    "ability_ownership",
    "entity_binding_correctness",
)
SEMANTIC_RECOVERABILITY_FIELDS = (
    "pronouns",
    "core_action",
    "condition",
    "consequence",
    "causality",
    "meaning_preservation",
)

# Coaching-claim/recoverability fields that must be applicable (not
# NOT_APPLICABLE) for every reviewed D item; the pre-registered gate requires
# all 30 D cases to contribute.
GATE_REQUIRED_APPLICABLE_FIELDS = frozenset({
    "standalone_coaching_claim",
    *SEMANTIC_RECOVERABILITY_FIELDS,
})

# Pre-registered human-review gate (documented in docs/phase2k-*).  The gate
# is evaluated only after all reviews are complete and every D item has
# applicable coaching-claim/recoverability scores.
REVIEW_GATE_SPEC = {
    "schema_version": "phase2k-review-gate-v1",
    "d_semantic_recoverability_min": 4.0,
    "d_over_a_semantic_recoverability_gain_min": 0.75,
    "d_meaning_preservation_min": 4.5,
    "d_unsupported_invention_max": 0.5,
    "asr_repair_correctness_min": 4.0,
    "entity_binding_correctness_min": 4.0,
    "gate_correctness_fields": (
        "asr_repair_correctness",
        "entity_binding_correctness",
    ),
    "required_applicable_fields": sorted(GATE_REQUIRED_APPLICABLE_FIELDS),
    "note": (
        "D must clear every threshold; ASR-repair and entity-binding "
        "correctness means are evaluated over all applicable reviewed items."
    ),
}

# ---------------------------------------------------------------------------
# Hashing / serialization helpers
# ---------------------------------------------------------------------------


def canonical_sha256(value: object) -> str:
    """Canonical content hash consistent with repository conventions."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_path_locator(path: Path) -> str:
    """Deterministic path locator independent of relative/absolute spelling.

    Files inside the repository are recorded as repository-relative POSIX
    paths; external files (e.g. the archived transcript DB) are recorded as
    their canonical absolute path.  Both spellings of the same file therefore
    produce the same locator and the same frozen-manifest hash.
    """
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError:
        return str(resolved)
    return relative.as_posix()


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is not allowed: {value!r}")


def _reject_float(value: str) -> Any:
    raise ValueError(f"floating-point JSON value is not allowed: {value!r}")


def _unique_pairs(label: str) -> Callable[[list[tuple[str, Any]]], dict[str, Any]]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{label} JSON contains duplicate key {key!r}")
            result[key] = item
        return result
    return unique


def load_json_strict(path: Path, *, label: str) -> dict[str, Any]:
    """Load a JSON object with duplicate-key and non-finite rejection."""
    try:
        body = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_pairs(label),
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} JSON is unavailable or malformed: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError(f"{label} must be a JSON object")
    return body


def load_json_strict_text(text: str, *, label: str) -> dict[str, Any]:
    """Load a JSON object from text with duplicate-key rejection."""
    try:
        body = json.loads(
            text,
            object_pairs_hook=_unique_pairs(label),
            parse_constant=_reject_constant,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} JSON is malformed: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError(f"{label} must be a JSON object")
    return body


def _strip_fenced_code(raw: str) -> str:
    """Robustly strip ```json ... ``` or ``` ... ``` fences."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_provider_json(raw: str, *, label: str) -> Any:
    """Parse provider output fail-closed.

    Strips markdown fences, rejects duplicate keys, non-finite constants, and
    floats (every Phase 2K provider schema uses strings/ints/bools only).
    """
    text = _strip_fenced_code(raw)
    try:
        body = json.loads(
            text,
            object_pairs_hook=_unique_pairs(label),
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} provider JSON is malformed: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError(f"{label} provider JSON must be an object")
    return body


def _require_exact_keys(value: object, expected: Iterable[str], label: str) -> None:
    expected_set = frozenset(expected)
    if not isinstance(value, Mapping) or set(value) != expected_set:
        missing = sorted(expected_set - set(value)) if isinstance(value, Mapping) else []
        extra = sorted(set(value) - expected_set) if isinstance(value, Mapping) else []
        detail = f"missing={missing} extra={extra}" if isinstance(value, Mapping) else "not an object"
        raise ValueError(f"{label} key set is invalid: {detail}")


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _require_nonempty_string(value: object, label: str) -> str:
    text = _require_string(value, label)
    if not text.strip():
        raise ValueError(f"{label} must be non-empty")
    return text


def _require_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _require_enum(value: object, options: Iterable[str], label: str) -> str:
    text = _require_string(value, label)
    if text not in frozenset(options):
        raise ValueError(f"{label} has invalid value {text!r}")
    return text


def _require_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _serialize(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, value: object) -> None:
    _write_atomic(path, _serialize(value))


def _envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    inner = {key: item for key, item in value.items() if key != "content_sha256"}
    return {"content_sha256": canonical_sha256(inner), **inner}


def _safe_float(value: float) -> float:
    return round(float(value), 4)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return _safe_float(sum(values) / len(values))


# ---------------------------------------------------------------------------
# Bounded lexical vocabulary, metadata adapter, and build lineage
# ---------------------------------------------------------------------------

VOCABULARY_PATH = ROOT / "data" / "phase2k_support" / (
    "league_lexical_vocabulary_v2.json"
)

# Bounded guarded-syntax anchors for champion-shaped alias context.  Only
# exact local word anchors license a guarded alias hint; the anchor sets are
# deliberately small so ordinary fishing/geography/commodity usage never
# becomes a champion correction.
_GUARDED_COMPARATIVE_WORDS = frozenset({
    "stronger", "weaker", "better", "worse", "tankier", "faster", "slower",
    "squishier",
})
_GUARDED_MATCH_CASES = ("case_insensitive", "initial_capital")
_GUARDED_ANCHOR_DIRECTIONS = ("before", "after", "either")

# Phase 2K implementation files hashed into the build lineage.  This is the
# code that participates in producing the output; test fixtures are excluded.
PHASE2K_IMPLEMENTATION_FILES = (
    "pipeline/phase2k_contextual_reconstruction.py",
    "scripts/build_phase2k_reconstruction.py",
    "scripts/finalize_phase2k_human_review.py",
)


def load_lexical_vocabulary() -> dict[str, Any]:
    """Load the bounded League lexical hint snapshot and verify its schema."""
    vocabulary = load_json_strict(VOCABULARY_PATH, label="phase2k lexical vocabulary")
    if vocabulary.get("schema_version") != LEAGUE_VOCABULARY_SCHEMA_VERSION:
        raise ValueError("phase2k lexical vocabulary schema version is invalid")
    for key in ("purpose", "scope_note"):
        if not isinstance(vocabulary.get(key), str) or not vocabulary[key]:
            raise ValueError(f"phase2k lexical vocabulary {key} is invalid")
    for key in ("ability_keys", "summoner_spells", "basic_domain_tokens"):
        items = vocabulary.get(key)
        if not isinstance(items, list) or not items or any(
            not isinstance(item, str) or not item for item in items
        ):
            raise ValueError(f"phase2k lexical vocabulary {key} is invalid")
    rules = vocabulary.get("champion_alias_rules")
    if not isinstance(rules, Mapping):
        raise ValueError("phase2k lexical vocabulary champion_alias_rules is invalid")
    if not isinstance(rules.get("rules_schema_version"), str) or not (
        rules["rules_schema_version"]
    ):
        raise ValueError(
            "phase2k lexical vocabulary champion_alias_rules schema is invalid",
        )
    if not isinstance(rules.get("match"), str) or rules.get("match") != (
        "exact_word_boundary"
    ):
        raise ValueError(
            "phase2k lexical vocabulary alias match must be exact_word_boundary",
        )
    for category in ("direct", "guarded", "metadata_licensed", "never"):
        items = rules.get(category)
        if not isinstance(items, list):
            raise ValueError(
                f"phase2k lexical vocabulary champion_alias_rules {category} "
                "is invalid",
            )
        for index, rule in enumerate(items):
            if not isinstance(rule, Mapping):
                raise ValueError(
                    f"phase2k lexical vocabulary alias rule {category}[{index}] "
                    "is invalid",
                )
            alias = rule.get("alias")
            canonical = rule.get("canonical")
            if not isinstance(alias, str) or not re.fullmatch(
                r"[A-Za-z]+", alias,
            ):
                raise ValueError(
                    f"phase2k lexical vocabulary alias rule {category}[{index}] "
                    "alias must be a non-empty ASCII word",
                )
            if not isinstance(canonical, str) or not re.fullmatch(
                r"[A-Za-z ]+", canonical,
            ):
                raise ValueError(
                    f"phase2k lexical vocabulary alias rule {category}[{index}] "
                    "canonical must be a non-empty ASCII name",
                )
            match_case = rule.get("match_case", "case_insensitive")
            if match_case not in _GUARDED_MATCH_CASES:
                raise ValueError(
                    f"phase2k lexical vocabulary alias rule {category}[{index}] "
                    "match_case is invalid",
                )
            if category == "guarded":
                syntax = rule.get("syntax")
                if not isinstance(syntax, Mapping):
                    raise ValueError(
                        f"phase2k lexical vocabulary guarded rule {index} "
                        "requires syntax",
                    )
                anchors = syntax.get("anchors")
                if not isinstance(anchors, list) or not anchors:
                    raise ValueError(
                        f"phase2k lexical vocabulary guarded rule {index} "
                        "requires at least one syntax anchor",
                    )
                for anchor in anchors:
                    if not isinstance(anchor, Mapping):
                        raise ValueError(
                            f"phase2k lexical vocabulary guarded rule {index} "
                            "anchor is invalid",
                        )
                    if anchor.get("kind") == "comparative_than":
                        continue
                    tokens = anchor.get("tokens")
                    direction = anchor.get("direction")
                    max_gap = anchor.get("max_gap", 1)
                    if not isinstance(tokens, list) or not tokens or any(
                        not isinstance(token, str) or not token
                        for token in tokens
                    ):
                        raise ValueError(
                            f"phase2k lexical vocabulary guarded rule {index} "
                            "anchor tokens are invalid",
                        )
                    if direction not in _GUARDED_ANCHOR_DIRECTIONS:
                        raise ValueError(
                            f"phase2k lexical vocabulary guarded rule {index} "
                            "anchor direction is invalid",
                        )
                    if not isinstance(max_gap, int) or max_gap < 0:
                        raise ValueError(
                            f"phase2k lexical vocabulary guarded rule {index} "
                            "anchor max_gap is invalid",
                        )
            if category == "metadata_licensed":
                if not isinstance(rule.get("metadata_field"), str) or not (
                    rule["metadata_field"]
                ):
                    raise ValueError(
                        f"phase2k lexical vocabulary metadata-licensed rule "
                        f"{index} metadata_field is invalid",
                    )
                if not isinstance(rule.get("metadata_value"), str) or not (
                    rule["metadata_value"]
                ):
                    raise ValueError(
                        f"phase2k lexical vocabulary metadata-licensed rule "
                        f"{index} metadata_value is invalid",
                    )
    context_tokens = rules.get("champion_context_tokens")
    if not isinstance(context_tokens, list) or not context_tokens or any(
        not isinstance(token, str) or not token for token in context_tokens
    ):
        raise ValueError(
            "phase2k lexical vocabulary champion_context_tokens is invalid",
        )
    return vocabulary


def lexical_vocabulary_hash() -> str:
    """Canonical content hash of the loaded vocabulary snapshot."""
    return canonical_sha256(load_lexical_vocabulary())


def vocabulary_lineage() -> dict[str, Any]:
    """Vocabulary path/hash lineage bound by prompt/config/cache hashes."""
    vocabulary = load_lexical_vocabulary()
    return {
        "path": normalize_path_locator(VOCABULARY_PATH),
        "file_sha256": file_sha256(VOCABULARY_PATH),
        "content_sha256": canonical_sha256(vocabulary),
        "schema_version": vocabulary["schema_version"],
        "scope_note": vocabulary["scope_note"],
    }


def _alias_match_pattern(alias: str, match_case: str) -> re.Pattern[str]:
    """Compile one exact word-boundary alias matcher.

    ``case_insensitive`` matches the full alias in any case.  The documented
    ``initial_capital`` Kale guard requires an ASCII capital first letter and
    matches the remainder case-insensitively, so lowercase grocery "kale" is
    never an eligible hint while "Kale"/"KALE" remain unconditional.
    """
    if match_case == "initial_capital":
        return re.compile(
            r"(?<!\w)[" + re.escape(alias[0].upper()) + r"]"
            r"(?i:" + re.escape(alias[1:]) + r")(?!\w)",
        )
    return re.compile(
        r"(?<!\w)" + re.escape(alias) + r"(?!\w)",
        re.IGNORECASE,
    )


def _target_word_tokens(text: str) -> list[re.Match[str]]:
    """Deterministic word tokens (letters/digits/apostrophes) for guards."""
    return list(re.finditer(r"[A-Za-z0-9']+", text))


def _guarded_alias_syntax_hint(
    text: str,
    start: int,
    end: int,
    anchors: list[Mapping[str, Any]],
    champion_context: frozenset[str],
) -> str | None:
    """Return the local champion-shaped syntax that licenses one alias.

    Guards are deliberately narrow: verb/role/versus anchors must be near the
    alias token, and the comparative "stronger than X" pattern additionally
    requires a champion context token on the comparator side so ordinary
    "worse than rice" or "better than Sig" prose never becomes a hint.
    """
    tokens = _target_word_tokens(text)
    alias_index = None
    for index, token in enumerate(tokens):
        if token.start() <= start and end <= token.end():
            alias_index = index
            break
    if alias_index is None:
        return None
    for anchor in anchors:
        if anchor.get("kind") == "comparative_than":
            if alias_index < 2:
                continue
            if tokens[alias_index - 1].group().casefold() != "than":
                continue
            comparative = tokens[alias_index - 2].group().casefold()
            if comparative not in _GUARDED_COMPARATIVE_WORDS:
                continue
            lo = max(0, alias_index - 5)
            context = {
                tokens[index].group().casefold()
                for index in range(lo, alias_index - 2)
            }
            if context & champion_context:
                return f"comparative_than:{comparative}"
            continue
        direction = anchor.get("direction", "before")
        anchor_tokens = {
            token.casefold() for token in anchor.get("tokens", [])
        }
        max_gap = int(anchor.get("max_gap", 1))
        if direction in ("before", "either"):
            for offset in range(1, max_gap + 2):
                index = alias_index - offset
                if index < 0:
                    break
                token = tokens[index].group()
                if token.casefold() in anchor_tokens:
                    return f"anchor:{token}:before:gap{offset - 1}"
        if direction in ("after", "either"):
            for offset in range(1, max_gap + 2):
                index = alias_index + offset
                if index >= len(tokens):
                    break
                token = tokens[index].group()
                if token.casefold() in anchor_tokens:
                    return f"anchor:{token}:after:gap{offset - 1}"
    return None


def detect_champion_alias_hints(
    bronze_text: str,
    selected: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Deterministic eligible champion-alias spelling hints for one target.

    This is lexical Pass 1 hint detection only: it never mutates clean_text
    and never fabricates repairs.  Every returned hint is an exact
    word-boundary occurrence that the provider MUST represent as an explicit
    DOMAIN_SPELLING repair.  Guarded aliases require champion-shaped local
    syntax, metadata-licensed aliases require the supplied champion
    metadata, and never/uncertainty-only surfaces are never returned.
    """
    vocabulary = load_lexical_vocabulary()
    rules = vocabulary["champion_alias_rules"]
    base_offset = int(selected["upstream_start"])
    metadata = selected.get("metadata", {})
    champion_context = {
        token.casefold()
        for token in rules.get("champion_context_tokens", [])
    }
    occurrence_counters: dict[str, int] = {}
    hints: list[dict[str, Any]] = []

    def register(
        *,
        rule: Mapping[str, Any],
        match: re.Match[str],
        category: str,
        syntax_hint: str | None,
    ) -> None:
        start, end = match.span()
        alias = rule["alias"]
        key = alias.casefold()
        occurrence = occurrence_counters.get(key, 0)
        occurrence_counters[key] = occurrence + 1
        hints.append({
            "alias": alias,
            "canonical": rule["canonical"],
            "rule_category": category,
            "match_case": rule.get("match_case", "case_insensitive"),
            "syntax_hint": syntax_hint,
            "target_local_start": start,
            "target_local_end": end,
            "source_absolute_start": base_offset + start,
            "source_absolute_end": base_offset + end,
            "text": bronze_text[start:end],
            "occurrence_index": occurrence,
        })

    for rule in rules["direct"]:
        pattern = _alias_match_pattern(
            rule["alias"], rule.get("match_case", "case_insensitive"),
        )
        for match in pattern.finditer(bronze_text):
            register(
                rule=rule, match=match, category="direct",
                syntax_hint=None,
            )

    for rule in rules["guarded"]:
        pattern = _alias_match_pattern(
            rule["alias"], rule.get("match_case", "case_insensitive"),
        )
        anchors = rule["syntax"]["anchors"]
        for match in pattern.finditer(bronze_text):
            syntax_hint = _guarded_alias_syntax_hint(
                bronze_text,
                match.start(),
                match.end(),
                anchors,
                champion_context,
            )
            if syntax_hint is not None:
                register(
                    rule=rule, match=match, category="guarded",
                    syntax_hint=syntax_hint,
                )

    for rule in rules["metadata_licensed"]:
        metadata_entry = metadata.get(rule["metadata_field"])
        if isinstance(metadata_entry, Mapping):
            metadata_entry = metadata_entry.get("value")
        if not isinstance(metadata_entry, str) or metadata_entry.casefold() != (
            rule["metadata_value"].casefold()
        ):
            continue
        pattern = _alias_match_pattern(
            rule["alias"], rule.get("match_case", "case_insensitive"),
        )
        for match in pattern.finditer(bronze_text):
            register(
                rule=rule, match=match, category="metadata_licensed",
                syntax_hint=None,
            )

    hints.sort(key=lambda item: (
        item["target_local_start"], item["target_local_end"], item["canonical"],
    ))
    seen_spans: set[tuple[int, int]] = set()
    for position, hint in enumerate(hints, 1):
        span = (hint["target_local_start"], hint["target_local_end"])
        if span in seen_spans:
            raise ValueError(
                "phase2k lexical hint detection produced overlapping alias "
                f"hints at bronze{span}",
            )
        seen_spans.add(span)
        hint["hint_id"] = f"p2k:lex:hint:{position:04d}"
    return hints


def _validate_eligible_hint_repairs(
    repairs: list[Mapping[str, Any]],
    hints: list[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    """Require every eligible hint to be an explicit DOMAIN_SPELLING repair.

    Replacement must equal the exact canonical, the bound span must be the
    exact hint span, and any DOMAIN_SPELLING repair that is not licensed by
    an eligible hint (for example like->Pyke, then/when->Shen, ward->Bard,
    well->Rell, or Soie->Zoe) fails closed so Pass 1 can never invent
    champion-name corrections.
    """
    by_span: dict[tuple[int, int], Mapping[str, Any]] = {}
    for hint in hints:
        span = (hint["target_local_start"], hint["target_local_end"])
        if span in by_span:
            raise ValueError(f"{label} contains duplicate hint spans")
        by_span[span] = hint
    domain_repairs = [
        repair for repair in repairs
        if repair["repair_type"] == "DOMAIN_SPELLING"
    ]
    covered: set[tuple[int, int]] = set()
    for repair in domain_repairs:
        span = (repair["target_local_start"], repair["target_local_end"])
        hint = by_span.get(span)
        if hint is None:
            raise ValueError(
                f"{label} DOMAIN_SPELLING repair at bronze "
                f"[{repair['target_local_start']}:{repair['target_local_end']}] "
                f"{repair['original_text']!r}->{repair['replacement']!r} is "
                "not licensed by any eligible champion-spelling hint; remove "
                "it. Never map common/ambiguous words to champions "
                "(like/then/when/ward/well) and never auto-repair Soie to Zoe.",
            )
        if repair["original_text"] != hint["text"]:
            raise ValueError(
                f"{label} DOMAIN_SPELLING repair for hint {hint['hint_id']} "
                f"must quote the exact bronze slice {hint['text']!r}",
            )
        if repair["replacement"] != hint["canonical"]:
            raise ValueError(
                f"{label} eligible hint {hint['hint_id']} at bronze "
                f"[{hint['target_local_start']}:{hint['target_local_end']}] "
                f"({hint['text']!r}, rule {hint['rule_category']}) must be "
                f"repaired to the exact canonical {hint['canonical']!r}; got "
                f"{repair['replacement']!r}",
            )
        covered.add(span)
    missing = [
        hint for span, hint in by_span.items() if span not in covered
    ]
    if missing:
        details = "; ".join(
            f"{hint['hint_id']} bronze[{hint['target_local_start']}:"
            f"{hint['target_local_end']}]={hint['text']!r} "
            f"(canonical {hint['canonical']!r}, rule "
            f"{hint['rule_category']}, occurrence "
            f"{hint['occurrence_index']})"
            for hint in missing
        )
        raise ValueError(
            f"{label} must repair every eligible champion-spelling hint; "
            f"missing DOMAIN_SPELLING repair(s): {details}. Add one "
            "DOMAIN_SPELLING repair per hint quoting the exact listed "
            "surface as original_text and the exact canonical as replacement.",
        )


def build_metadata_adapter(selected: Mapping[str, Any]) -> dict[str, Any]:
    """Field-level supplied-metadata adapter entries for one selected window.

    Champion/role/video_title are supplied facts from the frozen Phase 2J
    manifest.  Missing or empty fields stay absent rather than being
    fabricated.  The adapter records the provenance hash of each value so a
    prompt/artifact can reproduce exactly which field/value pair was used.
    """
    record_hash = _require_nonempty_string(
        selected["canonical_record_sha256"], "selected canonical_record_sha256",
    )
    window_id = _require_nonempty_string(selected["window_id"], "selected window_id")
    adapter: dict[str, Any] = {}
    for field in ("champion", "role", "video_title"):
        value = selected["metadata"].get(field)
        if not isinstance(value, str) or not value:
            continue
        locator = (
            "phase2j-window-selection-manifest-v1.json"
            f"#selected.{window_id}.metadata.{field}"
        )
        # ``video_title`` is provenance/supplied-fact metadata, but it is
        # deliberately not an inference input.  ``inference_allowed`` is
        # therefore False for title and True only for champion/role.
        inference_allowed = field != "video_title"
        adapter[field] = {
            "field": field,
            "value": value,
            "source": f"phase2j_manifest:{locator}",
            "provenance_hash": canonical_sha256({
                "canonical_record_sha256": record_hash,
                "field": field,
                "value": value,
            }),
            "reliability": "SUPPLIED_FACT",
            "inference_allowed": inference_allowed,
        }
    return adapter


def supplied_facts(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Flat projection of adapter values, excluding provenance-only fields.

    ``video_title`` is provenance-only and is never passed to a provider as a
    model-visible fact.  Champion/role remain supplied facts that may help
    disambiguate local references, but no title-based matchup inference is
    performed anywhere.
    """
    facts: dict[str, Any] = {}
    for field, entry in metadata.items():
        if isinstance(entry, Mapping) and isinstance(entry.get("value"), str) and (
            entry.get("inference_allowed") is not False
        ):
            facts[field] = entry["value"]
    return facts


def _metadata_values(metadata: Mapping[str, Any]) -> set[str]:
    """String fact values from either flat metadata or adapter entries."""
    values: set[str] = set()
    for entry in metadata.values():
        if isinstance(entry, str):
            values.add(entry)
        elif isinstance(entry, Mapping) and isinstance(entry.get("value"), str):
            values.add(entry["value"])
    return values


def _validate_metadata_adapter(
    adapter: object,
    *,
    selected: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    """Validate a supplied metadata adapter without title-based inference."""
    expected = build_metadata_adapter(selected)
    if adapter != expected:
        raise ValueError(
            f"{label} metadata adapter must exactly match the supplied "
            "manifest fields and provenance hashes",
        )
    if not isinstance(adapter, Mapping):
        raise ValueError(f"{label} metadata adapter must be an object")
    for field, entry in adapter.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"{label} metadata adapter {field} is invalid")
        _require_exact_keys(
            entry,
            (
                "field", "value", "source", "provenance_hash",
                "reliability", "inference_allowed",
            ),
            f"{label} metadata adapter {field}",
        )
        _require_enum(
            entry["reliability"],
            METADATA_RELIABILITY_LEVELS,
            f"{label} metadata adapter {field} reliability",
        )
        if entry["inference_allowed"] is not (field != "video_title"):
            raise ValueError(
                f"{label} metadata adapter {field} inference_allowed must be "
                "false for provenance-only video_title and true otherwise",
            )
        if not isinstance(entry["inference_allowed"], bool):
            raise ValueError(
                f"{label} metadata adapter {field} inference_allowed must "
                "be a boolean",
            )
        if entry["provenance_hash"] != canonical_sha256({
            "canonical_record_sha256": selected["canonical_record_sha256"],
            "field": entry["field"],
            "value": entry["value"],
        }):
            raise ValueError(
                f"{label} metadata adapter {field} provenance hash is invalid",
            )
    return dict(adapter)


def repo_lineage() -> dict[str, Any]:
    """Repo HEAD commit and dirty flag; never claims clean when it is dirty."""
    commit: str | None = None
    dirty: bool | None = None
    git_note: str | None = None
    try:
        head = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = head.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("git rev-parse did not return a 40-char commit")
        status = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        commit = None
        dirty = None
        git_note = "git metadata unavailable; cleanliness is not asserted"
    result: dict[str, Any] = {
        "repo_commit": commit,
        "repo_dirty": dirty,
        "repo_clean": dirty is False,
    }
    if git_note is not None:
        result["note"] = git_note
    return result


def implementation_file_lineage() -> dict[str, Any]:
    """File hashes of the Phase 2K implementation files used for the build."""
    files: dict[str, Any] = {}
    for locator in PHASE2K_IMPLEMENTATION_FILES:
        path = ROOT / locator
        files[locator] = {
            "path": normalize_path_locator(path),
            "file_sha256": file_sha256(path),
        }
    return {"files": files}


def build_lineage() -> dict[str, Any]:
    """Runtime/repo/vocabulary lineage recorded in records + build summary."""
    return {
        "repo": repo_lineage(),
        "implementation": implementation_file_lineage(),
        "vocabulary": vocabulary_lineage(),
        "metadata_adapter_schema_version": METADATA_ADAPTER_SCHEMA_VERSION,
    }


# ---------------------------------------------------------------------------
# Phase 2J frozen inputs
# ---------------------------------------------------------------------------


def _validate_recomputed_content_hash(obj: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(obj.get("content_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", obj["content_sha256"],
    ):
        raise ValueError(f"{label} content_sha256 is missing or malformed")
    expected = canonical_sha256({
        key: value for key, value in obj.items() if key != "content_sha256"
    })
    if obj["content_sha256"] != expected:
        raise ValueError(f"{label} content_sha256 does not match canonical content")


# ---------------------------------------------------------------------------
# Inference configuration sealing
# ---------------------------------------------------------------------------

INFERENCE_CONFIG_REQUIRED_KEYS = (
    "provider",
    "model",
    "endpoint",
    "temperature",
    "max_tokens",
    "thinking",
    "purpose",
)
INFERENCE_CONFIG_THINKING_OPTIONS = ("enabled", "disabled")

NO_PROVIDER_INFERENCE_CONFIG = {
    "provider": "none",
    "model": None,
    "endpoint": None,
    "temperature": None,
    "max_tokens": None,
    "thinking": None,
    "purpose": "phase2k-no-provider",
}

# Compatibility default for internal helper calls that do not seal a live
# config (used only by focused helper tests).  Real builds always seal either
# the explicit no-provider snapshot or an explicit live snapshot.
UNSEALED_INFERENCE_CONFIG = {
    "provider": "unset",
    "model": None,
    "endpoint": None,
    "temperature": None,
    "max_tokens": None,
    "thinking": None,
    "purpose": "phase2k-unsealed-helper-compat",
}

_SECRET_FIELD_TOKENS = (
    "api_key",
    "apikey",
    "api-key",
    "secret",
    "authorization",
    "bearer",
    "password",
    "passwd",
    "token",
    "credential",
    "private_key",
)
_SECRET_VALUE_MARKERS = (
    "api_key=",
    "api-key=",
    "bearer ",
    "authorization:",
)


def _validate_secret_free(
    value: object,
    *,
    path: str,
    exempt_keys: frozenset[str] = frozenset(),
) -> None:
    """Reject secret-bearing field names/values in an inference snapshot."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key not in exempt_keys and any(
                marker in key_text for marker in _SECRET_FIELD_TOKENS
            ):
                raise ValueError(
                    "secret-bearing field is not allowed in inference config: "
                    f"{path}.{key}",
                )
            _validate_secret_free(
                item,
                path=f"{path}.{key}",
                exempt_keys=frozenset(),
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_secret_free(
                item,
                path=f"{path}[{index}]",
                exempt_keys=frozenset(),
            )
    elif isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in _SECRET_VALUE_MARKERS):
            raise ValueError(
                f"inference config value looks like a credential: {path}",
            )


def validate_inference_config(config: object, *, label: str) -> dict[str, Any]:
    """Validate a secret-free exact inference configuration snapshot.

    The snapshot records provider, model, provider endpoint when available,
    temperature, max_tokens, thinking mode, and purpose.  API keys and other
    credentials are never part of the snapshot and are rejected here.
    """
    if not isinstance(config, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    for key in INFERENCE_CONFIG_REQUIRED_KEYS:
        if key not in config:
            raise ValueError(f"{label} is missing required key {key!r}")
    provider = _require_nonempty_string(config["provider"], f"{label} provider")
    model = config["model"]
    if model is not None:
        model = _require_nonempty_string(model, f"{label} model")
    endpoint = config["endpoint"]
    if endpoint is not None:
        endpoint = _require_nonempty_string(endpoint, f"{label} endpoint")
    temperature = config["temperature"]
    if temperature is not None and (
        isinstance(temperature, bool) or not isinstance(temperature, (int, float))
    ):
        raise ValueError(f"{label} temperature must be a number or null")
    max_tokens = config["max_tokens"]
    if max_tokens is not None:
        max_tokens = _require_int(max_tokens, f"{label} max_tokens", minimum=1)
    thinking = config["thinking"]
    if thinking is not None:
        thinking = _require_enum(
            thinking,
            INFERENCE_CONFIG_THINKING_OPTIONS,
            f"{label} thinking",
        )
    purpose = _require_nonempty_string(config["purpose"], f"{label} purpose")
    normalized: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "endpoint": endpoint,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "purpose": purpose,
    }
    for key, value in config.items():
        if key not in normalized:
            normalized[key] = value
    _validate_secret_free(
        normalized,
        path=label,
        exempt_keys=frozenset(INFERENCE_CONFIG_REQUIRED_KEYS),
    )
    return normalized


def inference_config_hash(config: object) -> str:
    """Canonical hash of a sealed secret-free inference configuration."""
    return canonical_sha256(validate_inference_config(
        config,
        label="inference config",
    ))


def load_phase2j_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json_strict(path, label="phase2j window-selection-manifest")
    _require_exact_keys(
        manifest,
        (
            "content_sha256", "schema_version", "purpose", "release_gate",
            "selection_policy", "input_hashes", "legacy_source_exclusions",
            "selected", "partition_counts", "diversity_summary",
            "candidate_generator_version", "checkpoint",
        ),
        "phase2j window-selection-manifest",
    )
    if manifest["schema_version"] != PHASE2J_MANIFEST_SCHEMA_VERSION:
        raise ValueError("phase2j manifest schema version is not the frozen v1")
    if manifest["release_gate"] != "LOCKED":
        raise ValueError("phase2j manifest is not LOCKED")
    _validate_recomputed_content_hash(manifest, label="phase2j manifest")
    selected = _require_list(manifest["selected"], "phase2j manifest selected")
    if len(selected) != 30:
        raise ValueError("phase2j manifest must contain exactly 30 selected windows")
    seen_windows: set[str] = set()
    seen_groups: set[str] = set()
    for selected_item in selected:
        if not isinstance(selected_item, Mapping):
            raise ValueError("phase2j selected window must be an object")
        _require_exact_keys(
            selected_item,
            (
                "source_group_id", "window_id", "upstream_source_id",
                "upstream_start", "upstream_end", "source_text",
                "source_text_sha256", "upstream_content_sha256",
                "source_text_char_length", "metadata", "phenomena",
                "asr_punctuation_band", "partition",
                "candidate_generator_version", "candidate_count",
                "candidate_catalog_sha256", "canonical_record_sha256",
            ),
            "phase2j selected window",
        )
        window_id = _require_nonempty_string(
            selected_item["window_id"], "phase2j window_id",
        )
        group = _require_nonempty_string(
            selected_item["source_group_id"], "phase2j source_group_id",
        )
        source_id = _require_nonempty_string(
            selected_item["upstream_source_id"], "phase2j upstream_source_id",
        )
        if group != f"video:{source_id}":
            raise ValueError("phase2j source group must derive from the video ID")
        if window_id in seen_windows or group in seen_groups:
            raise ValueError("phase2j manifest contains duplicate window/group identity")
        seen_windows.add(window_id)
        seen_groups.add(group)
        start = _require_int(
            selected_item["upstream_start"], "phase2j upstream_start", minimum=0,
        )
        end = _require_int(
            selected_item["upstream_end"], "phase2j upstream_end", minimum=0,
        )
        if end <= start:
            raise ValueError("phase2j upstream offsets are invalid")
        text = _require_string(selected_item["source_text"], "phase2j source_text")
        if end - start != len(text):
            raise ValueError("phase2j Bronze offsets do not match source text length")
        if selected_item["source_text_sha256"] != text_sha256(text):
            raise ValueError("phase2j source_text_sha256 is invalid")
        if selected_item["source_text_char_length"] != len(text):
            raise ValueError("phase2j source_text_char_length is invalid")
        if not isinstance(selected_item["metadata"], Mapping) or set(
            selected_item["metadata"],
        ) != {"champion", "role", "video_title"}:
            raise ValueError("phase2j selected metadata is invalid")
        for key in ("champion", "role", "video_title"):
            if not isinstance(selected_item["metadata"][key], str):
                raise ValueError("phase2j metadata values must be strings")
        if not isinstance(selected_item["canonical_record_sha256"], str):
            raise ValueError("phase2j canonical_record_sha256 is invalid")
    return manifest


def load_phase2j_reviewed_packet(path: Path) -> dict[str, Any]:
    packet = load_json_strict(path, label="phase2j reviewed-endpoint-annotation-packet")
    _require_exact_keys(
        packet,
        (
            "content_sha256", "schema_version", "purpose", "annotation_version",
            "release_gate", "selection_manifest_sha256",
            "selection_manifest_schema_version", "candidate_generator_version",
            "candidate_catalog", "rules", "records",
        ),
        "phase2j reviewed-endpoint-annotation-packet",
    )
    if packet["schema_version"] != PHASE2J_PACKET_SCHEMA_VERSION:
        raise ValueError("phase2j packet schema version is not the frozen v1")
    if packet["release_gate"] != "LOCKED":
        raise ValueError("phase2j reviewed packet is not LOCKED")
    _validate_recomputed_content_hash(packet, label="phase2j reviewed packet")
    records = _require_list(packet["records"], "phase2j reviewed records")
    if len(records) != 30:
        raise ValueError("phase2j reviewed packet must contain exactly 30 records")
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("phase2j reviewed record must be an object")
        _require_exact_keys(
            record,
            (
                "record_index", "annotation_id", "source_group_id", "window_id",
                "upstream_source_id", "upstream_start", "upstream_end",
                "partition", "bronze_text", "bronze_text_sha256",
                "bronze_char_length", "tokens", "endpoints", "window_status",
                "pass_a", "pass_b", "ambiguity_controls", "exclusion_controls",
                "reviewer_notes",
            ),
            "phase2j reviewed record",
        )
        if record["window_status"] != "REVIEWED":
            raise ValueError("phase2j reviewed packet contains a non-REVIEWED record")
        bronze = _require_string(record["bronze_text"], "phase2j bronze_text")
        if record["bronze_text_sha256"] != text_sha256(bronze):
            raise ValueError("phase2j bronze_text_sha256 is invalid")
        if record["bronze_char_length"] != len(bronze):
            raise ValueError("phase2j bronze_char_length is invalid")
        start = _require_int(
            record["upstream_start"], "phase2j upstream_start", minimum=0,
        )
        end = _require_int(record["upstream_end"], "phase2j upstream_end", minimum=0)
        if end - start != len(bronze):
            raise ValueError("phase2j reviewed Bronze offsets are invalid")
        endpoints = _require_list(record["endpoints"], "phase2j endpoints")
        for endpoint in endpoints:
            if not isinstance(endpoint, Mapping) or endpoint.get("disposition") != "KEEP":
                raise ValueError("phase2j reviewed endpoint must be KEEP")
    return packet


def _canonical_hash_of_parsed(obj: Mapping[str, Any], *, label: str) -> str:
    return canonical_sha256({
        key: value for key, value in obj.items() if key != "content_sha256"
    })


def validate_phase2j_frozen_inputs(
    manifest_path: Path,
    packet_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate both frozen Phase 2J artifacts and their cross-bindings."""
    manifest = load_phase2j_manifest(manifest_path)
    packet = load_phase2j_reviewed_packet(packet_path)
    if packet["selection_manifest_sha256"] != manifest["content_sha256"]:
        raise ValueError(
            "phase2j reviewed packet is not bound to the frozen manifest",
        )
    manifest_windows = {item["window_id"]: item for item in manifest["selected"]}
    packet_windows = {record["window_id"]: record for record in packet["records"]}
    if set(manifest_windows) != set(packet_windows):
        raise ValueError("phase2j manifest/packet window IDs are not aligned")
    for window_id, selected in manifest_windows.items():
        record = packet_windows[window_id]
        if record["source_group_id"] != selected["source_group_id"]:
            raise ValueError("phase2j manifest/packet source groups are misaligned")
        if record["bronze_text"] != selected["source_text"]:
            raise ValueError("phase2j manifest/packet Bronze text is misaligned")
        if (record["upstream_start"], record["upstream_end"]) != (
            selected["upstream_start"], selected["upstream_end"],
        ):
            raise ValueError("phase2j manifest/packet offsets are misaligned")
        if record["partition"] != selected["partition"]:
            raise ValueError("phase2j manifest/packet partitions are misaligned")
    return manifest, packet


def frozen_input_hashes(
    manifest_path: Path,
    packet_path: Path,
    manifest: Mapping[str, Any],
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """File and canonical content hashes of the frozen Phase 2J inputs."""
    return {
        "manifest": {
            "path": normalize_path_locator(manifest_path),
            "file_sha256": file_sha256(manifest_path),
            "content_sha256": manifest["content_sha256"],
            "canonical_parsed_sha256": _canonical_hash_of_parsed(
                manifest, label="phase2j manifest",
            ),
            "schema_version": manifest["schema_version"],
        },
        "reviewed_packet": {
            "path": normalize_path_locator(packet_path),
            "file_sha256": file_sha256(packet_path),
            "content_sha256": packet["content_sha256"],
            "canonical_parsed_sha256": _canonical_hash_of_parsed(
                packet, label="phase2j reviewed packet",
            ),
            "schema_version": packet["schema_version"],
        },
    }


# ---------------------------------------------------------------------------
# Read-only SQLite transcript access
# ---------------------------------------------------------------------------


def open_transcript_db(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def validate_transcript_source(
    connection: sqlite3.Connection,
    *,
    source_id: str,
    game: str,
    expected_full_sha256: str,
) -> dict[str, Any]:
    """Validate one source row and return exact transcript + metadata."""
    row = connection.execute(
        "SELECT video_id, transcription, game, champion, role, video_title "
        "FROM videos WHERE video_id = ?",
        (source_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"source {source_id} is absent from the transcript DB")
    transcript = row["transcription"]
    if not isinstance(transcript, str):
        raise ValueError(f"source {source_id} transcription is not text")
    if row["game"] != game:
        raise ValueError(
            f"source {source_id} game {row['game']!r} != expected {game!r}",
        )
    full_hash = text_sha256(transcript)
    if full_hash != expected_full_sha256:
        raise ValueError(
            f"source {source_id} full transcript SHA does not match the frozen "
            "Phase 2J upstream hash",
        )
    return {
        "source_id": source_id,
        "transcript": transcript,
        "transcript_sha256": full_hash,
        "transcript_char_length": len(transcript),
        "game": row["game"],
        "champion": row["champion"],
        "role": row["role"],
        "video_title": row["video_title"],
    }


def validate_target_slice(
    transcript: str,
    *,
    target_start: int,
    target_end: int,
    bronze_text: str,
) -> None:
    if not 0 <= target_start < target_end <= len(transcript):
        raise ValueError("Phase 2K target offsets are outside the transcript")
    if transcript[target_start:target_end] != bronze_text:
        raise ValueError("Phase 2K target slice does not round-trip the Bronze text")


# ---------------------------------------------------------------------------
# Deterministic transcript segmentation and ordered context retrieval
# ---------------------------------------------------------------------------

_BOUNDARY_RE = re.compile(
    r"[.!?…;]+[ \t]*[ \t\r\n]+|\r\n|\r|\n",
)
_TOKEN_RE = re.compile(r"\S+")
# Dedicated lexical entity-token extraction.  This intentionally does NOT
# reuse ``_TOKEN_RE``: transcript token counting/context logic depends on the
# exact ``\S+`` behavior, while entity validation needs punctuation-stripped
# alphabetic word tokens and must never treat contractions as entities.
_ENTITY_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*")


def _entity_tokens(text: str) -> list[str]:
    """Lexical word tokens for named-entity validation.

    Yields alphabetic words with surrounding punctuation removed.  Ordinary
    contractions (``I'm``, ``It's``, ``you're``) are skipped, while proper
    names containing an apostrophe (for example ``K'Sante`` or ``Kai'Sa``)
    remain eligible for entity licensing.  Possessive ``'s`` is reduced to
    its base token unless the base is a contraction pronoun.  ``_TOKEN_RE``
    is untouched because transcript token counting and context segmentation
    use it.
    """
    tokens: list[str] = []
    for match in _ENTITY_TOKEN_RE.finditer(text):
        token = match.group()
        normalized = token.replace("\u2019", "'")
        if "'" in normalized:
            base, suffix = normalized.rsplit("'", 1)
            suffix_folded = suffix.casefold()
            base_folded = base.casefold()
            if suffix_folded in {"m", "re", "ve", "ll", "d", "t"}:
                continue
            if suffix_folded == "s":
                if base_folded in {
                    "it", "he", "she", "that", "what", "who", "where",
                    "when", "why", "how", "there", "here", "let",
                }:
                    continue
                token = base
        tokens.append(token)
    return tokens


def _surface_normalize(text: str) -> str:
    """Deterministic surface normalization for fallback quote binding.

    Ignores only capitalization, punctuation, Unicode-space variants, and
    whitespace runs.  Letter sequences stay exact, so ``W'd`` vs ``wed`` and
    ``pryo`` vs ``prio`` never normalized-match: no spelling/lexical change,
    edit distance, synonym, champion-name, or ASR substitution is ever
    permitted.
    """
    normalized: list[str] = []
    pending_space = False
    for char in text:
        if char.isspace():
            if normalized and not pending_space:
                normalized.append(" ")
                pending_space = True
            continue
        pending_space = False
        if char.isalnum():
            normalized.append(char.casefold())
    return "".join(normalized).strip()


def _surface_normalize_map(text: str) -> tuple[str, list[int]]:
    """Return ``(normalized_text, original_index_map)``.

    ``index_map[i]`` is the original character offset of normalized
    character ``i``, so a normalized substring maps back to one exact
    contiguous source span (internal punctuation/whitespace included,
    leading/trailing punctuation ignored).
    """
    normalized: list[str] = []
    index_map: list[int] = []
    pending_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if normalized and not pending_space:
                normalized.append(" ")
                index_map.append(index)
                pending_space = True
            continue
        pending_space = False
        if char.isalnum():
            normalized.append(char.casefold())
            index_map.append(index)
    return "".join(normalized), index_map


def _differs_only_in_unicode_whitespace(left: str, right: str) -> bool:
    """True when the two strings differ bytewise only in Unicode whitespace.

    The strings must be non-equal, the same length, and every differing
    character must be a Unicode whitespace character on both sides.  This
    is the narrow license for contextual WHITESPACE repairs: ordinary
    punctuation, case, spelling, and true no-ops never qualify.
    """
    if left == right or len(left) != len(right):
        return False
    return all(
        l_char == r_char or (l_char.isspace() and r_char.isspace())
        for l_char, r_char in zip(left, right)
    )


def _unique_whitespace_slice_suggestion(
    segments: list[Mapping[str, Any]],
    quote: str,
) -> str | None:
    """One bounded diagnostic suggestion sentence for an absent evidence
    quote, or ``None`` when the suggestion would be unsafe.

    A quote is eligible only through the already-explicit whitespace
    skeleton: every non-whitespace character of the quote must appear in
    the same order inside one exact contiguous source slice.  The
    suggestion is emitted only when the skeleton maps to exactly one
    distinct exact slice text across the supplied context segments; zero
    matches and multiple distinct slice texts (the provider's intended
    slice is ambiguous) yield no suggestion.  This is diagnostic guidance
    only: the malformed quote still fails validation and the provider must
    return the exact slice verbatim.
    """
    skeleton = [char for char in quote if not char.isspace()]
    if not skeleton:
        return None
    needle = "".join(skeleton)
    distinct: dict[str, tuple[int, int, int]] = {}
    for segment_index, segment in enumerate(segments):
        text = _require_string(segment["text"], "context segment text")
        stripped = [
            (index, char)
            for index, char in enumerate(text)
            if not char.isspace()
        ]
        stripped_text = "".join(char for _, char in stripped)
        position = 0
        while True:
            found = stripped_text.find(needle, position)
            if found == -1:
                break
            start = stripped[found][0]
            end = stripped[found + len(needle) - 1][0] + 1
            distinct.setdefault(text[start:end], (segment_index, start, end))
            position = found + len(needle)
    if len(distinct) != 1:
        return None
    slice_text, (segment_index, start, end) = next(iter(distinct.items()))
    segment = segments[segment_index]
    segment_start = _require_int(
        segment["source_absolute_start"], "segment source start", minimum=0,
    )
    segment_id = _require_string(segment["segment_id"], "context segment id")
    return (
        f"Suggested exact replacement evidence_quote: {slice_text!r} "
        f"(exact contiguous slice of segment {segment_id} at "
        f"source_absolute {segment_start + start}..{segment_start + end}). "
        "Quote it verbatim; repeat it once per intended occurrence if it "
        "occurs more than once."
    )


def _whitespace_skeleton_spans(text: str, quote: str) -> list[tuple[int, int]]:
    """Exact spans whose slice differs from ``quote`` only in Unicode
    whitespace.

    This is the explicit harness-side matcher for the contextual WHITESPACE
    special case only: the provider writes a regular-space form (for
    example ``[ __ ]``) that cannot reproduce the exact Bronze Unicode
    whitespace.  Every non-whitespace character — including punctuation and
    underscore masks — must match exactly in order, the mapped slice must
    differ bytewise only in Unicode whitespace, and a true no-op (an exact
    identical slice) never matches.  Generic unrepresented edits still fail
    closed; this matcher is never used for other repair types.
    """
    skeleton = [char for char in quote if not char.isspace()]
    if not skeleton:
        return []
    stripped = [
        (index, char)
        for index, char in enumerate(text)
        if not char.isspace()
    ]
    stripped_text = "".join(char for _, char in stripped)
    needle = "".join(skeleton)
    spans: list[tuple[int, int]] = []
    position = 0
    while True:
        found = stripped_text.find(needle, position)
        if found == -1:
            break
        start = stripped[found][0]
        end = stripped[found + len(needle) - 1][0] + 1
        if _differs_only_in_unicode_whitespace(text[start:end], quote):
            spans.append((start, end))
        position = found + len(needle)
    return spans


def _normalized_occurrence_spans(
    text: str,
    quote: str,
) -> list[tuple[int, int]]:
    """Non-overlapping surface-normalized matches as exact source spans.

    Returns zero spans when the normalized quote is empty (pure punctuation)
    or absent.  Each span starts at the first original character of the
    normalized match and ends just after its last original character, so
    ``text[start:end]`` is the exact contiguous source slice for the
    normalized surface.  Adjacent punctuation at either boundary is kept in
    the slice so trailing/leading punctuation the provider quoted remains
    part of the stored exact source span; whitespace never crosses a word
    boundary.
    """
    normalized_quote = _surface_normalize(quote)
    if not normalized_quote:
        # A quote that carries no alphanumeric surface (for example
        # "[<NBSP>__<NBSP>]" or "[ __ ]") cannot normalize-match through
        # alphanumerics.  When it still has a non-whitespace punctuation
        # skeleton, fall back to the explicit whitespace-skeleton matcher so
        # whitespace-only surface differences (NBSP vs regular space) do not
        # make an otherwise exact quote unbindable; pure-whitespace quotes
        # and quotes with no skeleton never match.
        return _whitespace_skeleton_spans(text, quote)
    normalized_text, index_map = _surface_normalize_map(text)
    spans: list[tuple[int, int]] = []
    position = 0
    while True:
        found = normalized_text.find(normalized_quote, position)
        if found == -1:
            break
        start = index_map[found]
        end = index_map[found + len(normalized_quote) - 1] + 1
        while (
            start > 0
            and not text[start - 1].isalnum()
            and not text[start - 1].isspace()
        ):
            start -= 1
        while (
            end < len(text)
            and not text[end].isalnum()
            and not text[end].isspace()
        ):
            end += 1
        spans.append((start, end))
        position = found + len(normalized_quote)
    return spans


def _surface_spans(
    text: str,
    quote: str,
    *,
    normalized: bool,
) -> list[tuple[int, int]]:
    """Exact or surface-normalized non-overlapping occurrence spans."""
    if normalized:
        return _normalized_occurrence_spans(text, quote)
    return [
        (start, start + len(quote))
        for start in _occurrences(text, quote)
    ]


def _surface_strategies(
    surfaces: list[str],
    *,
    sources: list[str],
    label: str,
    group_counts: dict[str, list[int]] | None = None,
    absent_guidance: str | None = None,
    absent_suggestion_fn: Callable[[str], str | None] | None = None,
    aggregate_absent: bool = False,
    anchored_exact: dict[str, tuple[int, int, int]] | None = None,
) -> dict[str, bool]:
    """Choose exact vs normalized binding per unique surface.

    Exact binding remains first choice: when a surface has exact
    occurrences and every semantic assertion group for the surface fits
    within the exact occurrence count, exact spans are used.  Otherwise a
    deterministic fallback may bind the surface through surface
    normalization (ignoring only capitalization, punctuation, Unicode-space
    variants, and whitespace runs), but only when every group fits within
    the normalized occurrence count and that count is non-zero.

    When ``group_counts`` is supplied, binding is grouped per semantic
    assertion: the k-th proposal inside one group binds to the k-th
    occurrence, while different groups may share the same occurrence
    (different slots/assertions may share a mention span).  Supplying fewer
    group proposals than total source occurrences is valid.  When
    ``group_counts`` is omitted, the legacy exact-count contract is kept
    (the supplied count must equal the occurrence count).

    Zero or multiple normalized matches fail closed.

    When ``absent_suggestion_fn`` is supplied, a genuinely absent surface
    (no exact and no normalized occurrence) may append one bounded
    diagnostic suggestion from that callback to the absence error.  When
    ``aggregate_absent`` is set, all unique absent surfaces are reported in
    one bounded error instead of failing on only the first.

    ``anchored_exact`` maps unique surfaces to one exact context occurrence
    that deterministically anchors an otherwise-rejected repair evidence
    quote to the repair's source-absolute span.  Anchored surfaces skip the
    all-context occurrence-count rule and bind exact; this is the narrow
    repair-evidence-only rescue and never applies to bindings, statements,
    or general evidence ambiguity rules.
    """

    def _absent(surface: str) -> None:
        message = (
            f"{label} quote {surface!r} is absent from the supplied "
            "source (exact and surface-normalized)"
        )
        if (
            absent_suggestion_fn is not None
            and sum(
                len(_occurrences(source, surface)) for source in sources
            )
            == 0
        ):
            suggestion = absent_suggestion_fn(surface)
            if suggestion:
                message += ". " + suggestion
        if absent_guidance:
            message += " " + absent_guidance
        raise ValueError(message)

    supplied: dict[str, int] = {}
    for surface in surfaces:
        supplied[surface] = supplied.get(surface, 0) + 1
    if aggregate_absent:
        absent_surfaces: list[str] = []
        for surface in supplied:
            normalized_total = sum(
                len(_normalized_occurrence_spans(source, surface))
                for source in sources
            )
            if normalized_total == 0:
                absent_surfaces.append(surface)
        if absent_surfaces:
            if len(absent_surfaces) == 1:
                _absent(absent_surfaces[0])
            quoted = ", ".join(repr(surface) for surface in absent_surfaces)
            raise ValueError(
                f"{label} quotes {quoted} are absent from the supplied "
                "source (exact and surface-normalized). Remove each of "
                "these entire bindings rather than repeating them: "
                "source-absent mentions carry no real mention and are "
                "never normalized."
            )
    strategies: dict[str, bool] = {}
    for surface, count in supplied.items():
        if anchored_exact is not None and surface in anchored_exact:
            strategies[surface] = False
            continue
        exact_total = sum(
            len(_occurrences(source, surface)) for source in sources
        )
        normalized_total = sum(
            len(_normalized_occurrence_spans(source, surface))
            for source in sources
        )
        if group_counts is not None:
            group_need = max(group_counts.get(surface, [count]))
            if group_need <= exact_total:
                strategies[surface] = False
                continue
            if normalized_total > 0 and group_need <= normalized_total:
                strategies[surface] = True
                continue
            if normalized_total == 0:
                _absent(surface)
            raise ValueError(
                f"{label} quote {surface!r} cannot be bound deterministically: "
                f"it occurs {exact_total} time(s) exactly and "
                f"{normalized_total} time(s) after surface normalization in "
                "the supplied source but a single semantic assertion group "
                f"supplied {group_need} time(s); repeat it once per intended "
                "occurrence, quote a longer unique span, or use a distinct "
                "slot/assertion to share the span",
            )
        if exact_total == count:
            strategies[surface] = False
            continue
        if normalized_total == count and normalized_total > 0:
            strategies[surface] = True
            continue
        if normalized_total == 0:
            _absent(surface)
        raise ValueError(
            f"{label} quote {surface!r} cannot be bound deterministically: "
            f"it occurs {exact_total} time(s) exactly and "
            f"{normalized_total} time(s) after surface normalization in "
            "the supplied source but was supplied "
            f"{count} time(s); repeat it once per intended occurrence or "
            "quote a longer unique span",
        )
    return strategies


def _segment_pieces(text: str) -> list[tuple[int, int]]:
    """Deterministic contiguous segment spans with bounded token fallback."""
    boundaries = [match.end() for match in _BOUNDARY_RE.finditer(text)]
    if not boundaries:
        spans = [(0, len(text))]
    else:
        spans = []
        previous = 0
        for boundary in boundaries:
            if boundary > previous:
                spans.append((previous, boundary))
                previous = boundary
        if previous < len(text):
            spans.append((previous, len(text)))
    if len(text) == 0:
        return []
    # Bounded token fallback: split any segment exceeding the token bound.
    result: list[tuple[int, int]] = []
    for start, end in spans:
        if start >= end:
            continue
        tokens = list(_TOKEN_RE.finditer(text, start, end))
        if len(tokens) <= TOKEN_FALLBACK_BOUND:
            result.append((start, end))
            continue
        piece_start = start
        for index, token in enumerate(tokens):
            if index > 0 and index % TOKEN_FALLBACK_BOUND == 0:
                result.append((piece_start, token.start()))
                piece_start = token.start()
        result.append((piece_start, end))
    return result


def build_segments(
    text: str,
    source_group_id: str,
) -> list[dict[str, Any]]:
    """Full ordered source segments with stable IDs and exact offsets."""
    segments: list[dict[str, Any]] = []
    for ordinal, (start, end) in enumerate(_segment_pieces(text), 1):
        segments.append({
            "segment_id": f"seg:{source_group_id}:{ordinal:05d}",
            "segment_ordinal": ordinal,
            "kind": "source",
            "is_partial": False,
            "source_absolute_start": start,
            "source_absolute_end": end,
            "text": text[start:end],
        })
    return segments


def _count_tokens(text: str) -> int:
    return sum(1 for _ in _TOKEN_RE.finditer(text))


def _stop_reason_for_side(
    *,
    requested: int,
    chosen: list[dict[str, Any]],
    cap_triggered: str | None,
) -> str:
    if cap_triggered is not None:
        return cap_triggered
    if requested > 0 and len(chosen) < requested:
        return "SOURCE_BOUNDARY"
    return "RADIUS_SATISFIED"


def _take_previous(
    text: str,
    segments: list[dict[str, Any]],
    first_target_index: int,
    *,
    requested: int,
    segment_cap: int,
    char_cap: int,
) -> tuple[list[dict[str, Any]], str]:
    if requested <= 0:
        return [], "RADIUS_SATISFIED"
    chosen: list[dict[str, Any]] = []
    chars = 0
    cap_triggered = None
    for index in range(first_target_index - 1, -1, -1):
        if len(chosen) >= segment_cap:
            cap_triggered = "HARD_SEGMENT_CAP"
            break
        segment = segments[index]
        if chars + len(segment["text"]) > char_cap:
            cap_triggered = "HARD_CHAR_CAP"
            break
        chosen.append(segment)
        chars += len(segment["text"])
        if len(chosen) >= requested:
            break
    previous_items = [
        dict(segment, kind="previous")
        for segment in reversed(chosen)
    ]
    reason = _stop_reason_for_side(
        requested=requested, chosen=previous_items, cap_triggered=cap_triggered,
    )
    return previous_items, reason


def _take_following(
    text: str,
    segments: list[dict[str, Any]],
    last_target_index: int,
    *,
    requested: int,
    segment_cap: int,
    char_cap: int,
) -> tuple[list[dict[str, Any]], str]:
    if requested <= 0:
        return [], "RADIUS_SATISFIED"
    chosen: list[dict[str, Any]] = []
    chars = 0
    cap_triggered = None
    for index in range(last_target_index + 1, len(segments)):
        if len(chosen) >= segment_cap:
            cap_triggered = "HARD_SEGMENT_CAP"
            break
        segment = segments[index]
        if chars + len(segment["text"]) > char_cap:
            cap_triggered = "HARD_CHAR_CAP"
            break
        chosen.append(segment)
        chars += len(segment["text"])
        if len(chosen) >= requested:
            break
    following_items = [
        dict(segment, kind="following")
        for segment in chosen
    ]
    reason = _stop_reason_for_side(
        requested=requested, chosen=following_items, cap_triggered=cap_triggered,
    )
    return following_items, reason


def retrieve_context(
    transcript: str,
    *,
    source_group_id: str,
    window_id: str,
    target_start: int,
    target_end: int,
    bronze_text: str,
    previous_segments: int,
    following_segments: int,
    radius_label: str,
    segments: list[dict[str, Any]] | None = None,
    segment_cap: int = HARD_SEGMENT_CAP_PER_SIDE,
    char_cap: int = HARD_CHAR_CAP_PER_SIDE,
) -> dict[str, Any]:
    """Build an ordered exact context record around the exact target span."""
    validate_target_slice(
        transcript,
        target_start=target_start,
        target_end=target_end,
        bronze_text=bronze_text,
    )
    previous_segments = _require_int(
        previous_segments, "previous_segments", minimum=0,
    )
    following_segments = _require_int(
        following_segments, "following_segments", minimum=0,
    )
    radius_label = _require_nonempty_string(radius_label, "radius_label")
    if segments is None:
        segments = build_segments(transcript, source_group_id)
    if not segments:
        raise ValueError("cannot retrieve context from an empty transcript")

    first_target_index: int | None = None
    last_target_index: int | None = None
    for index, segment in enumerate(segments):
        if segment["source_absolute_start"] < target_end and segment[
            "source_absolute_end"
        ] > target_start:
            if first_target_index is None:
                first_target_index = index
            last_target_index = index
    if first_target_index is None or last_target_index is None:
        raise ValueError("target span does not intersect any transcript segment")

    first_segment = segments[first_target_index]
    last_segment = segments[last_target_index]

    context_items: list[dict[str, Any]] = []
    previous_items, previous_reason = _take_previous(
        transcript,
        segments,
        first_target_index,
        requested=previous_segments,
        segment_cap=segment_cap,
        char_cap=char_cap,
    )
    context_items.extend(previous_items)

    enlarged = previous_segments > 0 or following_segments > 0
    if enlarged and target_start > first_segment["source_absolute_start"]:
        pre_item = {
            "segment_id": first_segment["segment_id"] + ":pre",
            "segment_ordinal": first_segment["segment_ordinal"],
            "kind": "target_context",
            "is_partial": True,
            "source_absolute_start": first_segment["source_absolute_start"],
            "source_absolute_end": target_start,
            "text": transcript[first_segment["source_absolute_start"]:target_start],
        }
        context_items.append(pre_item)

    target_item = {
        "segment_id": f"target:{window_id}",
        "segment_ordinal": None,
        "kind": "target",
        "is_partial": False,
        "source_absolute_start": target_start,
        "source_absolute_end": target_end,
        "text": bronze_text,
    }
    context_items.append(target_item)

    if enlarged and target_end < last_segment["source_absolute_end"]:
        post_item = {
            "segment_id": last_segment["segment_id"] + ":post",
            "segment_ordinal": last_segment["segment_ordinal"],
            "kind": "target_context",
            "is_partial": True,
            "source_absolute_start": target_end,
            "source_absolute_end": last_segment["source_absolute_end"],
            "text": transcript[target_end:last_segment["source_absolute_end"]],
        }
        context_items.append(post_item)

    following_items, following_reason = _take_following(
        transcript,
        segments,
        last_target_index,
        requested=following_segments,
        segment_cap=segment_cap,
        char_cap=char_cap,
    )
    context_items.extend(following_items)

    previous_chars = sum(len(item["text"]) for item in previous_items)
    following_chars = sum(len(item["text"]) for item in following_items)
    total_chars = sum(len(item["text"]) for item in context_items)
    stop_reasons = sorted({
        f"PREVIOUS_{previous_reason}",
        f"FOLLOWING_{following_reason}",
    })
    if previous_segments == 0 and following_segments == 0:
        stop_reasons = ["TARGET_ONLY"]
    context = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "context_id": f"p2k:ctx:{window_id}:{radius_label}",
        "window_id": window_id,
        "source_group_id": source_group_id,
        "radius": radius_label,
        "target": {
            "window_id": window_id,
            "source_absolute_start": target_start,
            "source_absolute_end": target_end,
            "text": bronze_text,
            "text_sha256": text_sha256(bronze_text),
            "char_length": len(bronze_text),
        },
        "requested": {
            "previous_segments": previous_segments,
            "following_segments": following_segments,
        },
        "actual": {
            "previous_segments": len(previous_items),
            "following_segments": len(following_items),
            "previous_chars": previous_chars,
            "following_chars": following_chars,
            "total_chars": total_chars,
            "previous_tokens": sum(
                _count_tokens(item["text"]) for item in previous_items
            ),
            "following_tokens": sum(
                _count_tokens(item["text"]) for item in following_items
            ),
            "total_tokens": sum(
                _count_tokens(item["text"]) for item in context_items
            ),
            "segment_count": len(context_items),
        },
        "segments": context_items,
        "previous_stop_reason": previous_reason,
        "following_stop_reason": following_reason,
        "stop_reason": "; ".join(stop_reasons),
        "source_boundaries": {
            "context_start": (
                context_items[0]["source_absolute_start"] if context_items else None
            ),
            "context_end": (
                context_items[-1]["source_absolute_end"] if context_items else None
            ),
        },
    }
    context["content_sha256"] = canonical_sha256({
        "segments": context_items,
        "target": context["target"],
    })
    return context


def validate_context(context: Mapping[str, Any], transcript: str) -> None:
    """Validate a context record against the exact source transcript."""
    _require_exact_keys(
        context,
        (
            "schema_version", "context_id", "window_id", "source_group_id",
            "radius", "target", "requested", "actual", "segments",
            "previous_stop_reason", "following_stop_reason", "stop_reason",
            "source_boundaries", "content_sha256",
        ),
        "phase2k context",
    )
    if context["schema_version"] != CONTEXT_SCHEMA_VERSION:
        raise ValueError("phase2k context schema version is invalid")
    target = context["target"]
    _require_exact_keys(
        target,
        ("window_id", "source_absolute_start", "source_absolute_end", "text",
         "text_sha256", "char_length"),
        "phase2k context target",
    )
    start = _require_int(
        target["source_absolute_start"], "context target start", minimum=0,
    )
    end = _require_int(target["source_absolute_end"], "context target end", minimum=0)
    if end <= start or end > len(transcript):
        raise ValueError("phase2k context target offsets are invalid")
    if transcript[start:end] != target["text"]:
        raise ValueError("phase2k context target text is not an exact source slice")
    if target["text_sha256"] != text_sha256(target["text"]):
        raise ValueError("phase2k context target hash is invalid")
    segments = _require_list(context["segments"], "phase2k context segments")
    if not segments:
        raise ValueError("phase2k context must contain the target segment")
    previous_end = None
    seen_target = False
    for segment in segments:
        _require_exact_keys(
            segment,
            (
                "segment_id", "segment_ordinal", "kind", "is_partial",
                "source_absolute_start", "source_absolute_end", "text",
            ),
            "phase2k context segment",
        )
        seg_start = _require_int(
            segment["source_absolute_start"], "context segment start", minimum=0,
        )
        seg_end = _require_int(
            segment["source_absolute_end"], "context segment end", minimum=0,
        )
        if seg_end <= seg_start or seg_end > len(transcript):
            raise ValueError("phase2k context segment offsets are invalid")
        if transcript[seg_start:seg_end] != segment["text"]:
            raise ValueError("phase2k context segment text is not an exact slice")
        if previous_end is not None and seg_start < previous_end:
            raise ValueError("phase2k context segments overlap or are unordered")
        previous_end = seg_end
        if segment["kind"] == "target":
            if seen_target:
                raise ValueError("phase2k context contains multiple target items")
            seen_target = True
            if (seg_start, seg_end) != (start, end):
                raise ValueError("phase2k context target item must be the exact span")
    if not seen_target:
        raise ValueError("phase2k context is missing the exact target item")
    if context["content_sha256"] != canonical_sha256({
        "segments": segments,
        "target": target,
    }):
        raise ValueError("phase2k context content_sha256 is invalid")


# ---------------------------------------------------------------------------
# Provider response cache
# ---------------------------------------------------------------------------


def _cache_key(
    *,
    prompt_hash: str,
    inference_config_hash: str,
    schema_version: str,
    prompt_version: str,
    attempt_index: int = 0,
    attempt_kind: str = "base",
) -> str:
    return canonical_sha256({
        "prompt_hash": prompt_hash,
        "inference_config_hash": inference_config_hash,
        "schema_version": schema_version,
        "prompt_version": prompt_version,
        "attempt_index": attempt_index,
        "attempt_kind": attempt_kind,
    })


def _cache_entry_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def cache_load(cache_dir: Path, key: str) -> str | None:
    path = _cache_entry_path(cache_dir, key)
    if not path.is_file():
        return None
    entry = load_json_strict(path, label="phase2k response cache entry")
    raw = entry.get("raw_response")
    if not isinstance(raw, str):
        raise ValueError("phase2k response cache entry has no raw response")
    return raw


def cache_store(
    cache_dir: Path,
    key: str,
    raw_response: str,
    *,
    prompt_hash: str,
    prompt_version: str,
    config_hash: str,
    inference_config_hash: str,
    inference_config: Mapping[str, Any],
    schema_version: str,
    attempt_index: int = 0,
    attempt_kind: str = "base",
) -> None:
    path = _cache_entry_path(cache_dir, key)
    if path.exists():
        return
    entry = {
        "key": key,
        "prompt_hash": prompt_hash,
        "prompt_version": prompt_version,
        "config_hash": config_hash,
        "inference_config_hash": inference_config_hash,
        "inference_config": dict(inference_config),
        "schema_version": schema_version,
        "attempt_index": attempt_index,
        "attempt_kind": attempt_kind,
        "raw_response": raw_response,
        "raw_response_sha256": hashlib.sha256(
            raw_response.encode("utf-8"),
        ).hexdigest(),
    }
    _write_json_atomic(path, entry)


# ---------------------------------------------------------------------------
# Provider call plumbing
# ---------------------------------------------------------------------------

ChatCallable = Callable[[str, str], str]


def _chat_with_retries(
    chat: ChatCallable,
    system: str,
    user: str,
    *,
    attempts: int = 3,
) -> str:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            raw = chat(system, user)
        except Exception as exc:  # bounded retry on transient provider failure
            last_error = exc
            continue
        if not isinstance(raw, str) or not raw.strip():
            last_error = ValueError("provider returned empty content")
            continue
        return raw.strip()
    raise RuntimeError(
        f"provider chat failed after {attempts} attempts: {last_error}",
    ) from last_error


class ProviderCorrectionExhausted(ValueError):
    """Strict provider validation exhausted all bounded correction attempts.

    The ordered attempt history (every raw response, its content-addressed
    linkage, per-attempt status, and the exact validator error) is preserved
    on ``.attempts`` so the build can write it into the per-window failure
    artifact without normalizing the failed output as valid.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: list[Mapping[str, Any]],
    ) -> None:
        super().__init__(message)
        self.attempts = list(attempts)


def _fetch_provider_raw(
    chat: ChatCallable,
    *,
    system: str,
    user: str,
    schema_version: str,
    prompt_version: str,
    config_hash: str,
    label: str,
    cache_dir: Path | None = None,
    inference_config: Mapping[str, Any] | None = None,
    lineage: Mapping[str, Any] | None = None,
    raw_response_dir: Path | None = None,
    attempt_index: int = 0,
    attempt_kind: str = "base",
) -> dict[str, Any]:
    """Fetch and content-address one raw provider response (with cache)."""
    sealed = validate_inference_config(
        inference_config if inference_config is not None else UNSEALED_INFERENCE_CONFIG,
        label="phase2k inference config",
    )
    sealed_hash = inference_config_hash(sealed)
    prompt_hash = canonical_sha256({
        "system": system,
        "user": user,
        "schema_version": schema_version,
        "prompt_version": prompt_version,
        "config_hash": config_hash,
        "inference_config_hash": sealed_hash,
        "lineage": canonical_sha256(lineage or {}),
    })
    key = _cache_key(
        prompt_hash=prompt_hash,
        inference_config_hash=sealed_hash,
        schema_version=schema_version,
        prompt_version=prompt_version,
        attempt_index=attempt_index,
        attempt_kind=attempt_kind,
    )
    cached = cache_load(cache_dir, key) if cache_dir is not None else None
    source = "cache" if cached is not None else "provider"
    if cached is None:
        raw = _chat_with_retries(chat, system, user)
        if cache_dir is not None:
            cache_store(
                cache_dir,
                key,
                raw,
                prompt_hash=prompt_hash,
                prompt_version=prompt_version,
                config_hash=config_hash,
                inference_config_hash=sealed_hash,
                inference_config=sealed,
                schema_version=schema_version,
                attempt_index=attempt_index,
                attempt_kind=attempt_kind,
            )
    else:
        raw = cached
    response_sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    raw_response_path: str | None = None
    if raw_response_dir is not None:
        raw_response_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_response_dir / f"{response_sha256}.txt"
        if not raw_file.exists():
            _write_atomic(raw_file, raw)
        raw_response_path = raw_file.name
    return {
        "source": source,
        "prompt_hash": prompt_hash,
        "cache_key": key,
        "config_hash": config_hash,
        "inference_config": sealed,
        "inference_config_hash": sealed_hash,
        "inference_config_version": INFERENCE_CONFIG_VERSION,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "attempt_index": attempt_index,
        "attempt_kind": attempt_kind,
        "raw_response": raw,
        "raw_response_sha256": response_sha256,
        "raw_response_path": raw_response_path,
    }


def call_provider(
    chat: ChatCallable,
    *,
    system: str,
    user: str,
    schema_version: str,
    prompt_version: str,
    config_hash: str,
    label: str,
    cache_dir: Path | None = None,
    inference_config: Mapping[str, Any] | None = None,
    lineage: Mapping[str, Any] | None = None,
    raw_response_dir: Path | None = None,
) -> dict[str, Any]:
    """Call an injected chat function and return a provider call record."""
    call = _fetch_provider_raw(
        chat,
        system=system,
        user=user,
        schema_version=schema_version,
        prompt_version=prompt_version,
        config_hash=config_hash,
        label=label,
        cache_dir=cache_dir,
        inference_config=inference_config,
        lineage=lineage,
        raw_response_dir=raw_response_dir,
    )
    raw = call.pop("raw_response")
    parsed = parse_provider_json(raw, label=label)
    return {
        **call,
        "parsed": parsed,
        "status": "OK",
        "error": None,
    }


def _provider_attempt_record(
    call: Mapping[str, Any],
    *,
    window_id: str,
    stage: str,
    status: str,
    error: str | None,
    response: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build one ordered provider attempt record with raw-response linkage."""
    return {
        "window_id": window_id,
        "stage": stage,
        "attempt_index": call["attempt_index"],
        "attempt_kind": call["attempt_kind"],
        "status": status,
        "error": error,
        "prompt_version": call["prompt_version"],
        "schema_version": call["schema_version"],
        "model_call": {
            "source": call["source"],
            "prompt_hash": call["prompt_hash"],
            "cache_key": call["cache_key"],
            "config_hash": call["config_hash"],
            "inference_config": call["inference_config"],
            "inference_config_hash": call["inference_config_hash"],
            "inference_config_version": call["inference_config_version"],
            "prompt_version": call["prompt_version"],
            "schema_version": call["schema_version"],
            "attempt_index": call["attempt_index"],
            "attempt_kind": call["attempt_kind"],
            "raw_response_sha256": call["raw_response_sha256"],
            "raw_response_path": call["raw_response_path"],
            "status": "OK",
            "error": None,
        },
        "response": response,
    }


def call_provider_with_corrections(
    chat: ChatCallable,
    *,
    system: str,
    user: str,
    build_correction_prompt: Callable[[str, Exception], tuple[str, str]],
    validator: Callable[[Any], Any],
    label: str,
    schema_version: str,
    prompt_version: str,
    correction_prompt_version: str,
    config_hash: str,
    window_id: str,
    stage: str,
    cache_dir: Path | None = None,
    inference_config: Mapping[str, Any] | None = None,
    lineage: Mapping[str, Any] | None = None,
    raw_response_dir: Path | None = None,
    max_corrections: int = PROVIDER_MAX_CORRECTIONS,
) -> dict[str, Any]:
    """Initial provider call plus bounded strict-validation corrections.

    Every raw attempt is fetched through the same content-addressed cache
    path; the cache key binds the exact prompt (including the verbatim prior
    raw response and validator error embedded in the correction prompt), the
    attempt index/kind, prompt version, schema, and inference config, so a
    correction can never silently reuse or overwrite a different attempt.
    Validation stays strict: on exhaustion the exact final failure and the
    ordered attempt history are preserved on the raised exception.
    """
    attempts: list[dict[str, Any]] = []
    prior_raw: str | None = None
    prior_raws: list[str] = []
    last_error: Exception | None = None
    for attempt_index in range(max_corrections + 1):
        if attempt_index == 0:
            attempt_system, attempt_user = system, user
            attempt_kind = "base"
            attempt_prompt_version = prompt_version
        else:
            if prior_raw is None or last_error is None:
                raise RuntimeError("correction attempt has no prior raw response")
            attempt_system, attempt_user = build_correction_prompt(
                prior_raw,
                last_error,
                attempt_index=attempt_index,
                prior_raws=prior_raws,
            )
            attempt_kind = f"correction:{attempt_index}"
            attempt_prompt_version = correction_prompt_version
        call = _fetch_provider_raw(
            chat,
            system=attempt_system,
            user=attempt_user,
            schema_version=schema_version,
            prompt_version=attempt_prompt_version,
            config_hash=config_hash,
            label=label,
            cache_dir=cache_dir,
            inference_config=inference_config,
            lineage=lineage,
            raw_response_dir=raw_response_dir,
            attempt_index=attempt_index,
            attempt_kind=attempt_kind,
        )
        raw = call.pop("raw_response")
        prior_raw = raw
        prior_raws.append(raw)
        try:
            parsed = parse_provider_json(raw, label=label)
            result = validator(parsed)
        except ValueError as exc:
            last_error = exc
            attempts.append(_provider_attempt_record(
                call,
                window_id=window_id,
                stage=stage,
                status="FAILED",
                error=f"{type(exc).__name__}: {exc}",
                response=None,
            ))
            continue
        attempts.append(_provider_attempt_record(
            call,
            window_id=window_id,
            stage=stage,
            status="OK",
            error=None,
            response={
                "status": "OK",
                "schema_version": schema_version,
                "raw_response_sha256": call["raw_response_sha256"],
                "parsed": result,
                "raw_compact": parsed,
            },
        ))
        return {
            "attempts": attempts,
            "result": result,
            "final_attempt": attempts[-1],
        }
    raise ProviderCorrectionExhausted(
        f"{label} failed strict validation after "
        f"{max_corrections + 1} attempts: "
        f"{type(last_error).__name__}: {last_error}",
        attempts=attempts,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _target_identity(selected: Mapping[str, Any]) -> dict[str, Any]:
    bronze_text = _require_string(selected["source_text"], "selected source_text")
    return {
        "window_id": selected["window_id"],
        "source_group_id": selected["source_group_id"],
        "canonical_record_sha256": selected["canonical_record_sha256"],
        "upstream_start": selected["upstream_start"],
        "upstream_end": selected["upstream_end"],
        "upstream_content_sha256": selected["upstream_content_sha256"],
        "bronze_text": bronze_text,
        "bronze_text_sha256": text_sha256(bronze_text),
    }


def build_mechanical_provenance(selected: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic Pass 1 provenance sealed by the harness.

    The provider never echoes this object; the normalizer derives it from
    the sealed frozen inputs and adds the model's short rationale.
    """
    return {
        "task_kind": TEXT_RESTORATION_TASK_KIND,
        "target": _target_identity(selected),
        "prompt_version": MECHANICAL_PROMPT_VERSION,
        "schema_version": MECHANICAL_RESPONSE_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "config_version": CONFIG_VERSION,
        "input_metadata": build_metadata_adapter(selected),
    }


def build_reconstruction_provenance(selected: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic reconstruction provenance base, sealed by the harness."""
    return {
        "task_kind": CONTEXTUAL_RECONSTRUCTION_TASK_KIND,
        "target": _target_identity(selected),
        "prompt_version": RECONSTRUCTION_PROMPT_VERSION,
        "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "config_version": CONFIG_VERSION,
        "input_metadata": build_metadata_adapter(selected),
    }


def _seal_reconstruction_provenance(
    selected: Mapping[str, Any],
    *,
    rationale: str,
) -> dict[str, Any]:
    """Seal full reconstruction provenance from frozen inputs + rationale."""
    return {
        **build_reconstruction_provenance(selected),
        "rationale": rationale,
    }


def build_mechanical_prompt(selected: Mapping[str, Any]) -> tuple[str, str]:
    metadata = build_metadata_adapter(selected)
    vocabulary = load_lexical_vocabulary()
    system = (
        "You perform a mechanical text-restoration pass on one exact ASR "
        "transcript target.  This is NOT semantic extraction: JSON is "
        "transport only.  Do not extract semantic endpoints, entities, "
        "events, states, outcomes, relations, edges, claims, bindings, "
        "resolved entities, champion bindings, ability owners, or strategic "
        "concepts in this response.  The cleanup is target-only and "
        "context-free: never use surrounding video context, never resolve "
        "entities, pronouns, or ability ownership, and never change the "
        "meaning or the referent of any word.  Only high-confidence, "
        "context-free mechanical fixes are allowed (punctuation, "
        "capitalization, uncontroversial spelling, collation, whitespace, "
        "and high-confidence domain lexical fixes licensed by the supplied "
        "metadata and the bounded lexical hints).  The lexical vocabulary is "
        "a hint list only: it carries no ownership information and must "
        "never be used to infer who owns an ability or who is being talked "
        "about.  A high-confidence context-free ASR/domain correction is "
        "allowed when the local phrase plus the game/domain metadata makes "
        "the intended form unambiguous without resolving any owner or "
        "identity: for example, in a League of Legends transcript, "
        "\"wed on the wave\" may be repaired to \"W'd on the wave\" (the "
        "ability key W with a reduced auxiliary) at HIGH confidence.  Keep "
        "the ownership of W unresolved and never expand who used it.  "
        "Ambiguous abbreviations such as \"HS\" (which could be \"his\", "
        "\"has\", or a champion tag) must remain unchanged when the intended "
        "form is not uniquely determined; record that ambiguity under "
        "\"uncertainties\" instead.  If a word looks like a possible ASR "
        "error whose intended form depends on who owns an ability or who is "
        "being talked about, do not repair it; leave it unchanged and list "
        "it as an uncertainty.\n"
        "The supplied lexical vocabulary may include champion alias rules: "
        "deterministic exact word-boundary spelling hints for champion "
        "names.  They are lexical spelling data only, never entity or "
        "binding output.  Only the exact eligible occurrences listed under "
        "\"lexical_hints\" may be repaired as champion-name spelling fixes, "
        "and EVERY listed hint MUST be repaired with repair_type "
        "DOMAIN_SPELLING: quote the exact listed \"surface_text\" verbatim "
        "as original_text and use the exact \"canonical\" value as "
        "replacement (the canonical case is fixed regardless of the surface "
        "case).  Never repair a champion-name spelling that is not listed "
        "in lexical_hints, never change an already-correct champion name, "
        "and never map common or ambiguous words to champions "
        "(like->Pyke, then/when->Shen, ward->Bard, well->Rell, Soie->Zoe "
        "and similar are always forbidden unless explicitly listed).  "
        "Guarded aliases (pike, rise, rice, Sig) are eligible only when the "
        "hint list includes that exact occurrence with a champion-shaped "
        "\"syntax_hint\"; ordinary usage such as fishing pike, grocery "
        "kale, ordinary rise/rice, or a signature \"sig\" is never listed "
        "and must stay untouched.  The lowercase word \"kale\" is not an "
        "eligible alias: only capital-initial Kale surfaces are listed, so "
        "never \"correct\" grocery kale.  darus is eligible only when "
        "listed, which requires Darius in the supplied champion metadata; "
        "never infer Darius from darus otherwise because Varus is a "
        "competing lexical possibility.  Soie is never an automatic Zoe "
        "repair: if genuinely uncertain, record it as a "
        "DOMAIN_TOKEN_UNCERTAIN uncertainty with Zoe among the "
        "alternatives; do not repair it.\n"
        "The harness seals every deterministic field itself.  Do NOT emit "
        "provenance, task_kind, target identity, hashes, metadata copies, "
        "offsets, span coordinates, IDs, or evidence fields: they are not "
        "part of this schema and any extra key fails validation.  You only "
        "supply the judgment fields below.\n"
        "Respond with a single JSON object with exactly these five keys: "
        '"schema_version", "clean_text", "repairs", "uncertainties", '
        '"rationale".  Do not add any sixth key.\n'
        '"clean_text" is the exact restored target text.  It must equal the '
        'deterministic application of your ordered "repairs" to the exact '
        "Bronze target; every difference between Bronze and clean_text must "
        "be represented by exactly one non-overlapping repair, and an "
        "unrepresented difference fails validation.\n"
        '"repairs" is a list of objects with exactly these keys: '
        '"original_text", "replacement", "repair_type", "confidence", '
        '"rationale".  "original_text" must be an exact quote from the '
        "Bronze target; \"replacement\" must be a non-empty different "
        "string (never an empty string).  Repair types are one of: "
        + ", ".join(MECHANICAL_REPAIR_TYPES) + ".  Confidence is one of "
        "HIGH, MEDIUM, LOW.  List repairs in left-to-right order.  When the "
        "same original_text occurs more than once in the target, either "
        "repair every occurrence you intend to change (each as its own "
        "ordered proposal) or quote a longer unique span; otherwise binding "
        "is ambiguous and fails.  Return an empty repairs list when no "
        "mechanical fix is safe.  Never emit ownership, pronoun, or entity "
        "resolution repair types.  Never emit a no-op proposal whose "
        "replacement equals original_text, never emit overlapping "
        "proposals, and every difference between Bronze and clean_text must "
        "be represented by exactly one ordered proposal.  For punctuation "
        "insertions or deletions, quote a full non-empty replacement span "
        "that includes the adjacent word, for example "
        '"raptor The" -> "raptor. The" for a period insertion or '
        '"hello." -> "hello" for a period deletion; never quote an empty '
        "span.  Champion-name spelling fixes must use repair_type "
        "DOMAIN_SPELLING and may only fix the exact occurrences listed in "
        "lexical_hints; a DOMAIN_SPELLING repair that is absent from the "
        "hint list fails validation.\n"
        '"uncertainties" is a list of at most '
        + str(MECHANICAL_UNCERTAINTY_CAP)
        + ' objects with exactly these keys: "surface_text", '
        '"uncertainty_type", "alternatives", "note".  "surface_text" must '
        "be an exact quote from the Bronze target.  Uncertainty types are "
        "one of: "
        + ", ".join(MECHANICAL_UNCERTAINTY_TYPES) + ".  Each alternative has "
        'exactly the keys "text" and "confidence" (a category HIGH, MEDIUM, '
        "or LOW, never a floating-point probability).  Only list genuine "
        "unresolved alternatives that could materially affect text "
        "restoration: do NOT manufacture alternatives for clear standard "
        "words, intentional speaker or turn markers such as \">>\", or "
        "already-correct domain tokens.  Empty is preferred when none.  "
        'Uncertainties never change "clean_text".\n'
        '"rationale" is one short paragraph summarizing the mechanical '
        "cleanup and any uncertainties.  Lexical definitions are allowed "
        "here: for example, explaining that 'pryo' is a common mishearing "
        "of 'prio' (priority), a standard League term, is a valid mechanical "
        "rationale.  Do NOT disguise a semantic extraction list as "
        "rationale: never enumerate endpoints, entities, events, relations, "
        "bindings, owners, or similar semantic objects under labels like "
        "\"entities:\", \"events:\", or \"bindings:\", and never write "
        "phrases such as \"extracted entities\" or \"semantic extraction\" "
        "in this response.  Never add binding or entity-resolution content "
        "to any field."
    )
    user = json.dumps(
        {
            "task": "mechanical_cleanup",
            "schema_version": MECHANICAL_RESPONSE_SCHEMA_VERSION,
            "target": _target_identity(selected),
            "metadata": metadata,
            "supplied_facts": supplied_facts(metadata),
            "metadata_adapter_schema_version": METADATA_ADAPTER_SCHEMA_VERSION,
            "lexical_hints": [
                {
                    "hint_id": hint["hint_id"],
                    "alias": hint["alias"],
                    "canonical": hint["canonical"],
                    "rule_category": hint["rule_category"],
                    "surface_text": hint["text"],
                    "occurrence_index": hint["occurrence_index"],
                    "syntax_hint": hint["syntax_hint"],
                }
                for hint in detect_champion_alias_hints(
                    selected["source_text"], selected,
                )
            ],
            "metadata_policy": {
                "champion_role_are_supplied_inference_facts": True,
                "video_title_is_provenance_only": True,
                "no_title_based_matchup_inference": True,
                "missing_fields_stay_absent": True,
            },
            "lexical_vocabulary": {
                "schema_version": vocabulary["schema_version"],
                "content_sha256": canonical_sha256(vocabulary),
                "ability_keys": vocabulary["ability_keys"],
                "summoner_spells": vocabulary["summoner_spells"],
                "basic_domain_tokens": vocabulary["basic_domain_tokens"],
                "champion_alias_rules": vocabulary["champion_alias_rules"],
                "scope_note": vocabulary["scope_note"],
            },
            "response_example": {
                "schema_version": MECHANICAL_RESPONSE_SCHEMA_VERSION,
                "clean_text": "He used W. Kayle top. HS stays unchanged.",
                "repairs": [{
                    "original_text": "he",
                    "replacement": "He",
                    "repair_type": "CAPITALIZATION",
                    "confidence": "HIGH",
                    "rationale": "sentence start",
                }, {
                    "original_text": "Kale",
                    "replacement": "Kayle",
                    "repair_type": "DOMAIN_SPELLING",
                    "confidence": "HIGH",
                    "rationale": "listed lexical hint for the champion "
                    "spelling Kale",
                }],
                "uncertainties": [{
                    "surface_text": "HS",
                    "uncertainty_type": "ASR_ALTERNATIVES",
                    "alternatives": [
                        {"text": "his", "confidence": "MEDIUM"},
                        {"text": "has", "confidence": "MEDIUM"},
                        {"text": "HS", "confidence": "LOW"},
                    ],
                    "note": "ambiguous abbreviation",
                }],
                "rationale": "One capitalization fix, one listed champion "
                "spelling fix, one ambiguity retained.",
            },
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return system, user


def build_mechanical_correction_prompt(
    selected: Mapping[str, Any],
    prior_raw: str,
    error: Exception,
) -> tuple[str, str]:
    """Build a strict mechanical correction prompt for one failed attempt.

    The correction prompt embeds the exact prior raw response and the exact
    validator error, and asks for a complete corrected JSON object only.  It
    is content-addressed by the harness exactly like the base prompt, so a
    correction attempt can never be confused with or silently overwrite any
    other attempt.
    """
    metadata = build_metadata_adapter(selected)
    system = (
        "You are correcting a mechanical text-restoration response that "
        "failed strict validation.  This is still Pass 1: text restoration "
        "only, never semantic extraction.  Return a COMPLETE corrected JSON "
        "object with exactly these five keys and nothing else: "
        '"schema_version", "clean_text", "repairs", "uncertainties", '
        '"rationale".\n'
        "This is a bounded correction pass: one initial attempt plus at "
        "most three corrections.  Fix EVERY validator-reported difference "
        "in this single response; do not assume another correction will "
        "follow.\n"
        'Rules that must hold exactly:\n'
        '1. "clean_text" must equal the deterministic application of your '
        'ordered "repairs" to the exact Bronze target; every difference '
        "between Bronze and clean_text must be represented by exactly one "
        "non-overlapping repair, and an unrepresented difference fails.\n"
        '2. Every "original_text" must be an exact quote from the Bronze '
        'target and "replacement" must be a non-empty DIFFERENT string.  '
        "Never emit a no-op repair whose replacement equals original_text.  "
        "Never emit overlapping repairs (one quoted span may not overlap "
        "another after binding).\n"
        "3. For punctuation insertions or deletions, quote a full "
        "non-empty replacement span including the adjacent word, for "
        'example "raptor The" -> "raptor. The" for a period insertion or '
        '"hello." -> "hello" for a period deletion.\n'
        '4. When the same "original_text" occurs more than once, either '
        "repair every occurrence you intend to change (each as its own "
        "ordered proposal) or quote a longer unique span; a single "
        "ambiguous proposal fails.\n"
        "5. Lexical definitions are allowed in rationale text (for example "
        "'prio' is a mishearing of 'prio' (priority), a standard League "
        "term), but never enumerate semantic endpoints, entities, events, "
        "relations, bindings, owners, or similar objects under labels like "
        '"entities:", "events:", "bindings:" or with extraction phrasing '
        'such as "extracted entities" or "semantic extraction".\n'
        "6. Never add extra keys, provenance, hashes, offsets, IDs, or "
        "evidence fields; any extra key fails validation.\n"
        "7. When clean_text does not match the deterministic application "
        "of your repairs, use the ordered non-equal diff feedback in "
        "validator_error (applied vs requested text with exact differing "
        "substrings and positions).  Return every reported difference as "
        "its own full non-empty replacement span with an exact "
        "original_text quote from Bronze (include the adjacent word for "
        "punctuation insertions/deletions).  Never drop, merge, or "
        "auto-create repairs, and never relax the exact Bronze-quote or "
        "non-empty replacement rules; return the COMPLETE corrected repair "
        "list that reproduces clean_text exactly.\n"
        "8. Every eligible champion-spelling hint listed in "
        "\"lexical_hints\" MUST be repaired with repair_type "
        "DOMAIN_SPELLING: quote the exact listed \"surface_text\" verbatim "
        "as original_text and the exact \"canonical\" value as replacement "
        "(one repair per listed occurrence).  Missing hint repairs, wrong "
        "replacements, or a DOMAIN_SPELLING repair that is not listed as a "
        "hint all fail validation.  Never map common or ambiguous words to "
        "champions (like/then/when/ward/well), never auto-repair Soie to "
        "Zoe, and never \"correct\" lowercase grocery kale; guarded aliases "
        "(pike, rise, rice, Sig) are eligible only when listed with a "
        "champion-shaped syntax_hint.\n"
        "Do not explain; output only the complete corrected JSON object."
    )
    user = json.dumps(
        {
            "task": "mechanical_cleanup_correction",
            "correction_prompt_version": MECHANICAL_CORRECTION_PROMPT_VERSION,
            "schema_version": MECHANICAL_RESPONSE_SCHEMA_VERSION,
            "target": _target_identity(selected),
            "metadata": metadata,
            "supplied_facts": supplied_facts(metadata),
            "metadata_adapter_schema_version": METADATA_ADAPTER_SCHEMA_VERSION,
            "lexical_hints": [
                {
                    "hint_id": hint["hint_id"],
                    "alias": hint["alias"],
                    "canonical": hint["canonical"],
                    "rule_category": hint["rule_category"],
                    "surface_text": hint["text"],
                    "occurrence_index": hint["occurrence_index"],
                    "syntax_hint": hint["syntax_hint"],
                }
                for hint in detect_champion_alias_hints(
                    selected["source_text"], selected,
                )
            ],
            "prior_raw_response": prior_raw,
            "validator_error": f"{type(error).__name__}: {error}",
            "response_schema": {
                "schema_version": MECHANICAL_RESPONSE_SCHEMA_VERSION,
                "clean_text": "exact restored target text equal to the "
                "deterministic application of repairs",
                "repairs": [{
                    "original_text": "exact quote from Bronze",
                    "replacement": "non-empty different string; include the "
                    "adjacent word for punctuation insertions/deletions",
                    "repair_type": "PUNCTUATION",
                    "confidence": "HIGH",
                    "rationale": "short mechanical rationale; lexical "
                    "definitions allowed, no semantic endpoint lists",
                }],
                "uncertainties": [{
                    "surface_text": "exact quote from Bronze",
                    "uncertainty_type": "ASR_ALTERNATIVES",
                    "alternatives": [{
                        "text": "candidate",
                        "confidence": "MEDIUM",
                    }],
                    "note": "short note",
                }],
                "rationale": "one short paragraph; never a semantic "
                "endpoint list",
            },
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return system, user


def _sufficiency_slot_template(status: str = "RESOLVED") -> dict[str, Any]:
    """Compact provider slot template for the exact sufficiency schema."""
    return {
        "status": status,
        "candidates": [{
            "candidate": "NONE",
            "confidence": "HIGH",
            "evidence_quotes": [],
        }],
        "confidence": "HIGH",
        "evidence_quotes": [],
    }


def _sufficiency_response_template() -> dict[str, Any]:
    """Complete exact response template used by the v2 sufficiency prompt."""
    return {
        "schema_version": SUFFICIENCY_RESPONSE_SCHEMA_VERSION,
        "decision": "SUFFICIENT",
        "slots": {
            slot_key: _sufficiency_slot_template()
            for slot_key in SLOT_KEYS
        },
        "metadata_conflicts": [],
        "rationale": "one short rationale paragraph",
    }


def build_sufficiency_prompt(
    selected: Mapping[str, Any],
    context: Mapping[str, Any],
    mechanical_cleaned_text: str,
    *,
    at_max_context: bool,
) -> tuple[str, str]:
    metadata = build_metadata_adapter(selected)
    system = (
        "You diagnose whether the given ordered transcript context is "
        "sufficient to resolve every slot of the delimited target.  This is "
        "the sufficiency diagnostic, which runs only after mechanical text "
        "restoration and context retrieval, so semantic analysis is allowed "
        "here.  You NEVER count or emit character offsets: the harness "
        "deterministically binds your exact evidence quotes to the supplied "
        "context segments.  Slots: "
        + ", ".join(SLOT_KEYS) + ".  Status is one of: "
        + ", ".join(SLOT_STATUSES) + ".  Use RESOLVED with candidates "
        '[{"candidate": "NONE", "confidence": "HIGH", "evidence_quotes": '
        "[]}] for slots that are genuinely not applicable.  "
        "Decision is exactly one of: "
        + ", ".join(SUFFICIENCY_DECISIONS) + ".  Return SUFFICIENT only when "
        "every slot is RESOLVED.  Never return MAX_CONTEXT_BUT_UNRESOLVED "
        "unless the supplied context is at maximum radius.  Record metadata "
        "conflicts explicitly instead of silently overriding them.\n"
        "Confidence is always the categorical HIGH, MEDIUM, or LOW; never "
        "use floating-point probabilities anywhere in the response.\n"
        "Every evidence_quotes entry must be an EXACT contiguous quote from "
        "one supplied context segment (segment text is provided in the "
        "context payload).  Do not paraphrase, do not approximate, do not "
        "include offsets or coordinates, and do not quote across two "
        "segments.  When the same quoted text occurs more than once in the "
        "supplied context, repeat the quote once per intended occurrence in "
        "left-to-right order, or quote a longer unique span; a single quote "
        "for a repeated surface fails deterministic binding.  Empty "
        "evidence_quotes is valid when no exact source quote exists.\n"
        "Respond with a single complete JSON object that exactly matches "
        "the supplied response_schema: keys \"schema_version\", "
        "\"decision\", \"slots\" (all "
        + str(len(SLOT_KEYS))
        + " exact slot keys), \"metadata_conflicts\", and \"rationale\".  "
        "Each slot has exactly \"status\", \"candidates\", \"confidence\", "
        "and \"evidence_quotes\".  Each candidate has exactly \"candidate\", "
        "\"confidence\", and \"evidence_quotes\".  Do not add extra keys "
        "such as slot_analysis, slot_analyses, slot_assessments, notes, "
        "decision_reasoning, decision_grounds, or decision_confidence; any "
        "extra or missing key fails validation."
    )
    user = json.dumps(
        {
            "task": "semantic_sufficiency",
            "schema_version": SUFFICIENCY_RESPONSE_SCHEMA_VERSION,
            "response_schema": _sufficiency_response_template(),
            "target": context["target"],
            "context": {
                "radius": context["radius"],
                "requested": context["requested"],
                "actual": context["actual"],
                "segments": context["segments"],
            },
            "mechanical_cleaned_text": mechanical_cleaned_text,
            "metadata": metadata,
            "supplied_facts": supplied_facts(metadata),
            "metadata_adapter_schema_version": METADATA_ADAPTER_SCHEMA_VERSION,
            "metadata_policy": {
                "champion_role_are_supplied_inference_facts": True,
                "video_title_is_provenance_only": True,
                "no_title_based_matchup_inference": True,
                "missing_fields_stay_absent": True,
            },
            "at_max_context": at_max_context,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return system, user


def build_sufficiency_correction_prompt(
    selected: Mapping[str, Any],
    context: Mapping[str, Any],
    mechanical_cleaned_text: str,
    *,
    at_max_context: bool,
    prior_raw: str,
    error: Exception,
) -> tuple[str, str]:
    """Build a strict sufficiency correction prompt for one failed attempt."""
    metadata = build_metadata_adapter(selected)
    system = (
        "You are correcting a semantic sufficiency diagnostic that failed "
        "strict validation.  Return a COMPLETE corrected JSON object that "
        "exactly matches the supplied response_schema (all "
        + str(len(SLOT_KEYS))
        + " slot keys) and nothing else.\n"
        "Rules that must hold exactly:\n"
        "1. Keys are exactly \"schema_version\", \"decision\", \"slots\", "
        "\"metadata_conflicts\", \"rationale\".  Never invent keys such as "
        "slot_analysis, slot_analyses, slot_assessments, notes, "
        "decision_reasoning, decision_grounds, or decision_confidence.\n"
        "2. Each slot has exactly \"status\", \"candidates\", "
        "\"confidence\", \"evidence_quotes\"; each candidate has exactly "
        "\"candidate\", \"confidence\", \"evidence_quotes\".\n"
        "3. Confidence is always the categorical HIGH, MEDIUM, or LOW; "
        "never emit floating-point numbers.\n"
        "4. Every evidence quote must be an exact contiguous quote from one "
        "supplied context segment, never a paraphrase and never across "
        "segments.  Repeat a quote once per occurrence when its surface "
        "appears more than once, or quote a longer unique span.\n"
        "5. Decision invariants: SUFFICIENT requires every slot RESOLVED; "
        "MAX_CONTEXT_BUT_UNRESOLVED is only valid when the supplied context "
        "is at maximum radius and at least one slot is unresolved; "
        "NEED_MORE_* decisions require at least one non-resolved slot.\n"
        "6. Do not explain; output only the complete corrected JSON object."
    )
    user = json.dumps(
        {
            "task": "semantic_sufficiency_correction",
            "correction_prompt_version": SUFFICIENCY_CORRECTION_PROMPT_VERSION,
            "schema_version": SUFFICIENCY_RESPONSE_SCHEMA_VERSION,
            "response_schema": _sufficiency_response_template(),
            "target": context["target"],
            "context": {
                "radius": context["radius"],
                "requested": context["requested"],
                "actual": context["actual"],
                "segments": context["segments"],
            },
            "mechanical_cleaned_text": mechanical_cleaned_text,
            "metadata": metadata,
            "supplied_facts": supplied_facts(metadata),
            "metadata_adapter_schema_version": METADATA_ADAPTER_SCHEMA_VERSION,
            "at_max_context": at_max_context,
            "prior_raw_response": prior_raw,
            "validator_error": f"{type(error).__name__}: {error}",
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return system, user


def _reconstruction_response_template() -> dict[str, Any]:
    """Exact full compact provider template; the model supplies judgment only."""
    return {
        "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
        "clean_target_transcript": (
            "complete deterministic application of the repair list to Bronze"
        ),
        "contextual_repairs": [
            {
                "original_text": "exact Bronze quote",
                "replacement": "non-empty replacement different from original_text",
                "repair_type": "PRONOUN_RESOLUTION",
                "confidence": "HIGH",
                "evidence_quotes": ["exact context quote"],
                "rationale": "short rationale",
            },
        ],
        "bindings": [
            {
                "slot": "pronouns",
                "mention_text": "exact Bronze quote",
                "resolved_candidate": "Viktor",
                "resolved_status": "RESOLVED",
                "confidence": "HIGH",
                "evidence_quotes": ["exact context quote"],
                "alternatives": [
                    {
                        "candidate": "Viktor",
                        "evidence_quotes": ["exact context quote"],
                        "note": "short note",
                    },
                ],
                "metadata_contributed": False,
                "rationale": "short rationale",
            },
        ],
        "unresolved_alternatives": [
            {
                "slot": "pronouns",
                "mention_text": "exact Bronze quote",
                "alternatives": [
                    {
                        "candidate": "Viktor",
                        "confidence": "LOW",
                        "evidence_quotes": ["exact context quote"],
                    },
                ],
                "evidence_quotes": ["exact context quote"],
                "note": "short note",
            },
        ],
        "rationale": "short rationale",
    }


def build_reconstruction_prompt(
    selected: Mapping[str, Any],
    context: Mapping[str, Any],
    mechanical_cleaned_text: str,
    diagnostic: Mapping[str, Any],
) -> tuple[str, str]:
    """Prompt for the compact reconstruction envelope (judgment fields only)."""
    metadata = build_metadata_adapter(selected)
    system = (
        "You produce the Phase 2K contextual reconstruction for one exact "
        "Bronze target.  The clean source-faithful transcript may resolve "
        "pronouns, entities, abilities, and contextual ASR while staying "
        "source-faithful.  D is not B and never masquerades as B or Bronze. "
        "You supply judgment fields only: the harness deterministically "
        "binds every quoted span, seals every ID/source offset, and seals "
        "full provenance from the frozen inputs plus your rationale.  Never "
        "emit IDs, offsets, coordinates, hashes, metadata copies, or "
        "provenance in your response.\n"
        "Exact response rules:\n"
        "1. Keys are exactly \"schema_version\", "
        "\"clean_target_transcript\", \"contextual_repairs\", \"bindings\", "
        "\"unresolved_alternatives\", \"rationale\".\n"
        "2. Every difference between Bronze and clean_target_transcript is "
        "represented by exactly one contextual repair; applying the "
        "complete non-overlapping repair list to Bronze must reproduce "
        "clean_target_transcript exactly.  If clean_target_transcript "
        "equals Bronze the repair list is empty.\n"
        "3. Each repair has exactly \"original_text\", \"replacement\", "
        "\"repair_type\", \"confidence\", \"evidence_quotes\", \"rationale\". "
        "original_text is an exact contiguous Bronze quote and replacement "
        "is non-empty and different, with exactly one narrow exception: "
        "repair_type FILLER may use an empty replacement to delete a "
        "non-lexical filler (for example \" MH\" -> \"\", quoting the "
        "filler plus its leading whitespace so the clean text has no "
        "double spaces).  Empty replacement is valid only for FILLER "
        "deletions: every FILLER deletion still requires a non-empty exact "
        "original_text and at least one exact evidence quote; no other "
        "repair type may use an empty replacement, and generic deletions "
        "are never allowed.  repair_type is one of: "
        + ", ".join(CONTEXTUAL_REPAIR_TYPES) + ".  confidence is HIGH, "
        "MEDIUM, or LOW.  evidence_quotes are exact contiguous quotes from "
        "the supplied context segments (never paraphrases, never across "
        "segments); every evidence quote must be a byte-exact contiguous "
        "slice - never add, remove, or substitute whitespace characters "
        "(including non-breaking spaces) inside a quote; repeat a repeated "
        "quote once per intended occurrence or quote a longer unique "
        "span.  Every repair requires at least "
        "one evidence quote.  For Unicode whitespace normalization (for "
        "example a non-breaking space that must become a regular space) "
        "use repair_type WHITESPACE: quote the exact Bronze whitespace "
        "character as original_text (a whitespace-only original_text such "
        "as a non-breaking space is valid) and put the replacement "
        "character; the harness deterministically binds repeated "
        "whitespace-only proposals left-to-right to the exact matching "
        "Bronze whitespace slices.  Never emit overlapping repairs: every "
        "difference must be one exact contiguous Bronze span, and when "
        "edits touch adjacent text, merge them into one span.\n"
        "4. Each binding has exactly \"slot\", \"mention_text\", "
        "\"resolved_candidate\", \"resolved_status\", \"confidence\", "
        "\"evidence_quotes\", \"alternatives\", \"metadata_contributed\", "
        "\"rationale\".  slot is one of the 11 diagnostic slots; "
        "mention_text is an exact Bronze quote; resolved_status must match "
        "the corresponding final diagnostic slot status (RESOLVED bindings "
        "must use a diagnostic/metadata-licensed candidate).  Each "
        "alternative has exactly \"candidate\", \"evidence_quotes\", "
        "\"note\".  Binding proposals are ordered left-to-right per semantic "
        "assertion (mention_text + slot + resolved_candidate + "
        "resolved_status): the k-th proposal inside one assertion binds to "
        "the k-th occurrence, and different slots/assertions may share the "
        "same mention span.  Supplying fewer proposals than total source "
        "occurrences is valid; never supply more proposals than "
        "occurrences for the same assertion.\n"
        "5. Entity, pronoun, ability-ownership, and reference resolutions "
        "are only allowed when a corresponding binding licenses the exact "
        "same mention with a RESOLVED candidate/status and evidence.  "
        "Unresolved mentions are never rewritten.\n"
        "6. Each binding's mention_text must be an exact contiguous Bronze "
        "quote.  Omit slots with no source mention entirely; never emit "
        "mention_text \"NONE\" or a placeholder binding for an absent "
        "mention, and never repeat a source-absent binding.  When a slot "
        "or candidate is not applicable, do not emit a binding for it.\n"
        "7. Each unresolved alternative has exactly \"slot\", "
        "\"mention_text\", \"alternatives\" (each with \"candidate\", "
        "\"confidence\", \"evidence_quotes\"), \"evidence_quotes\", "
        "\"note\".\n"
        "8. Do not emit paraphrases, statements, unsupported claims, or "
        "polished semantic text (semantic polish is a separate later "
        "pass).  Do not invent facts or new ontology abstractions.  "
        "Literal words that already appear in Bronze or in an exact "
        "evidence quote may remain, but never introduce an abstraction "
        "such as access, continuity, initiative, role transfer, "
        "conversion, tempo, priority, or pressure that is not licensed by "
        "exact source evidence.\n"
        "9. Respond with one JSON object that exactly matches the "
        "supplied response_schema and nothing else."
    )
    user = json.dumps(
        {
            "task": "reconstruction",
            "prompt_version": RECONSTRUCTION_PROMPT_VERSION,
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "response_schema": _reconstruction_response_template(),
            "target": _target_identity(selected),
            "context": {
                "radius": context["radius"],
                "actual": context["actual"],
                "segments": context["segments"],
            },
            "mechanical_cleaned_text": mechanical_cleaned_text,
            "metadata": metadata,
            "supplied_facts": supplied_facts(metadata),
            "metadata_adapter_schema_version": METADATA_ADAPTER_SCHEMA_VERSION,
            "metadata_policy": {
                "champion_role_are_supplied_inference_facts": True,
                "video_title_is_provenance_only": True,
                "no_title_based_matchup_inference": True,
                "missing_fields_stay_absent": True,
            },
            "diagnostics": {
                "decision": diagnostic["decision"],
                "slots": diagnostic["response"]["parsed"]["slots"],
                "metadata_conflicts": diagnostic["response"]["parsed"][
                    "metadata_conflicts"
                ],
            },
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return system, user


def build_reconstruction_correction_prompt(
    selected: Mapping[str, Any],
    context: Mapping[str, Any],
    mechanical_cleaned_text: str,
    diagnostic: Mapping[str, Any],
    *,
    prior_raw: str,
    error: Exception,
    attempt_index: int = 1,
    prior_raws: list[str] | None = None,
) -> tuple[str, str]:
    """Strict reconstruction correction prompt for one failed attempt."""
    prior_raws = list(prior_raws or [])
    earlier_raws = prior_raws[:-1] if prior_raws else []
    repeated = bool(earlier_raws) and any(
        item == prior_raw for item in earlier_raws
    )
    repeat_guidance = ""
    if repeated:
        repeat_guidance = (
            "  Your previous correction response is byte-identical to an "
            "earlier failed response.  You MUST return a materially "
            "different corrected JSON object: read validator_error "
            "carefully, fix every reported issue (merge overlapping edits "
            "into one exact Bronze span, remove redundant or source-absent "
            "repairs/bindings, quote exact Bronze spans), and never repeat "
            "the same invalid output.  Identical repeat responses are "
            "rejected."
        )
    system = (
        "You are correcting a Phase 2K contextual reconstruction that "
        "failed strict validation.  Return a COMPLETE corrected JSON "
        "object that exactly matches the supplied response_schema and "
        "nothing else.  Supply judgment fields only: no IDs, offsets, "
        "coordinates, hashes, metadata copies, or provenance.  Keep the "
        "rules exactly: complete non-overlapping repairs that reproduce "
        "clean_target_transcript, exact Bronze/context quotes (repeat "
        "repeated quotes once per intended occurrence), bindings licensed "
        "by the final diagnostics, unresolved mentions never rewritten, "
        "and no invented ontology abstractions.  "
        "Every evidence_quotes field at every nesting level (contextual "
        "repairs, bindings, binding alternatives, unresolved alternatives "
        "and their alternatives) must be a JSON array of exact quote "
        "strings - never a scalar string, object, or null - even when you "
        "supply only one quote.  "
        "Bindings: omit slots with no source mention; never emit "
        "mention_text \"NONE\"; every mention_text must quote Bronze "
        "exactly; when a binding is source-absent, remove the entire "
        "binding rather than repeating it; when a slot is not applicable, "
        "do not emit a binding for it.  When validator_error lists "
        "multiple source-absent binding quotes, remove EVERY listed "
        "binding entirely in this one correction rather than fixing one "
        "at a time.  "
        "RESOLVED bindings: resolved_candidate must copy one complete "
        "licensed diagnostic candidate or metadata value exactly, "
        "byte-for-byte, including case, descriptors, and parentheticals.  "
        "Never shorten, title-case, paraphrase, or strip parenthetical "
        "qualifiers.  When validator_error lists allowed candidates, copy "
        "the chosen full exact value from that list; if no exact value is "
        "licensed for the source mention, remove the binding when the "
        "mention is optional or source-absent, or preserve the "
        "contractually allowed uncertainty rather than inventing a nicer "
        "label.  "
        "Repairs: when validator_error reports overlapping contextual "
        "repairs, merge the overlapping edits into ONE exact Bronze span "
        "(one contiguous original_text/replacement pair covering every "
        "change) and return the complete non-overlapping repair list.  "
        "When validator_error reports a redundant or unrepresentable "
        "repair (for example Bronze and clean_target_transcript already "
        "match at that location), remove that entire repair.  When "
        "validator_error suggests an exact replacement evidence_quote "
        "(an exact whitespace-skeleton slice), use that suggested "
        "evidence_quote VERBATIM (repeat it once per intended occurrence "
        "if it occurs more than once) and never re-invent whitespace.  "
        "FILLER deletions are the ONLY contextual repairs that may use an "
        "empty replacement: quote the non-lexical filler plus its "
        "surrounding whitespace in original_text (for example "
        "\" MH\" -> \"\") so the clean text has no double spaces, and keep "
        "at least one exact evidence quote; never use an empty replacement "
        "for any other repair type and never replace a deletion with a "
        "no-op whitespace-preserving replacement.  "
        "When "
        "clean_target_transcript does not match the deterministic "
        "application of the contextual repair list, use the ordered "
        "non-equal diff feedback in validator_error (applied vs requested "
        "text with exact differing substrings, positions, and full-word "
        "diagnostic suggestions such as original_text=\"exhaust\" "
        "replacement=\"Exhaust\"): return every reported difference as its "
        "own full non-empty replacement span with an exact Bronze "
        "original_text quote (include the adjacent word for punctuation "
        "insertions/deletions), and never auto-create repairs.  Unicode "
        "whitespace differences use repair_type WHITESPACE with the exact "
        "Bronze whitespace character (including any non-breaking space) "
        "quoted byte-for-byte as original_text and the replacement "
        "character; quote the exact NBSP/Unicode whitespace character in "
        "evidence_quotes and never substitute a visually similar "
        "character.  The harness maps it to the exact Bronze slice."
        + repeat_guidance
        + "  Do not explain; output only the "
        "complete corrected JSON object."
    )
    correction_rules = [
        "Return the COMPLETE corrected JSON object exactly matching the "
        "response_schema; never omit fields or output explanations.",
        "Every evidence_quotes field at every nesting level (contextual "
        "repairs, bindings, binding alternatives, unresolved alternatives "
        "and their alternatives) must be a JSON array of exact quote "
        "strings, never a scalar string, object, or null, even for a "
        "single quote.",
        "RESOLVED bindings: resolved_candidate must copy one complete "
        "licensed diagnostic candidate or metadata value exactly "
        "(byte-for-byte), including case, descriptors, and parentheticals; "
        "never shorten, title-case, paraphrase, or strip parenthetical "
        "qualifiers.",
        "When validator_error lists allowed candidates, copy the chosen "
        "full exact value from that list; never invent a nicer label.",
        "If no exact value is licensed for the source mention, remove the "
        "binding when the mention is optional or source-absent, or "
        "preserve the contractually allowed uncertainty.",
        "Quote exact Bronze/context whitespace byte-for-byte, including "
        "NBSP/Unicode whitespace characters, and use repair_type "
        "WHITESPACE for whitespace normalization.",
        "Fix every validator_error issue in this single response; do not "
        "assume another correction will be offered.",
    ]
    user = json.dumps(
        {
            "task": "reconstruction_correction",
            "correction_prompt_version": RECONSTRUCTION_CORRECTION_PROMPT_VERSION,
            "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "response_schema": _reconstruction_response_template(),
            "correction_rules": correction_rules,
            "target": _target_identity(selected),
            "context": {
                "radius": context["radius"],
                "actual": context["actual"],
                "segments": context["segments"],
            },
            "mechanical_cleaned_text": mechanical_cleaned_text,
            "metadata": build_metadata_adapter(selected),
            "supplied_facts": supplied_facts(build_metadata_adapter(selected)),
            "diagnostics": {
                "decision": diagnostic["decision"],
                "slots": diagnostic["response"]["parsed"]["slots"],
                "metadata_conflicts": diagnostic["response"]["parsed"][
                    "metadata_conflicts"
                ],
            },
            "prior_raw_response": prior_raw,
            "validator_error": f"{type(error).__name__}: {error}",
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return system, user


# ---------------------------------------------------------------------------
# Evidence-span validation helpers
# ---------------------------------------------------------------------------


def _validate_evidence_span(
    span: object,
    *,
    transcript: str,
    context: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(span, Mapping):
        raise ValueError(f"{label} evidence span must be an object")
    _require_exact_keys(
        span,
        (
            "segment_id", "source_absolute_start", "source_absolute_end",
            "text",
        ),
        label,
    )
    start = _require_int(
        span["source_absolute_start"], f"{label} evidence start", minimum=0,
    )
    end = _require_int(
        span["source_absolute_end"], f"{label} evidence end", minimum=0,
    )
    bounds = context["source_boundaries"]
    if not (
        bounds["context_start"] is not None
        and bounds["context_end"] is not None
        and bounds["context_start"] <= start < end <= bounds["context_end"]
    ):
        raise ValueError(f"{label} evidence span is outside the context")
    if transcript[start:end] != span["text"]:
        raise ValueError(f"{label} evidence span is not an exact source slice")
    segment_ids = {item["segment_id"] for item in context["segments"]}
    if span["segment_id"] not in segment_ids:
        raise ValueError(f"{label} evidence segment ID is not in the context")
    return dict(span)


def _validate_bronze_span(
    span: object,
    *,
    bronze_text: str,
    base_offset: int,
    label: str,
) -> dict[str, Any]:
    if not isinstance(span, Mapping):
        raise ValueError(f"{label} span must be an object")
    _require_exact_keys(
        span,
        (
            "target_local_start", "target_local_end",
            "source_absolute_start", "source_absolute_end", "text",
        ),
        label,
    )
    local_start = _require_int(
        span["target_local_start"], f"{label} local start", minimum=0,
    )
    local_end = _require_int(
        span["target_local_end"], f"{label} local end", minimum=0,
    )
    if not 0 <= local_start < local_end <= len(bronze_text):
        raise ValueError(f"{label} local offsets are invalid")
    if bronze_text[local_start:local_end] != span["text"]:
        raise ValueError(f"{label} text is not an exact Bronze slice")
    if span["source_absolute_start"] != base_offset + local_start:
        raise ValueError(f"{label} source-absolute start is inconsistent")
    if span["source_absolute_end"] != base_offset + local_end:
        raise ValueError(f"{label} source-absolute end is inconsistent")
    return dict(span)


# ---------------------------------------------------------------------------
# Mechanical cleanup
# ---------------------------------------------------------------------------


def _validate_no_semantic_extraction(value: object, *, path: str) -> None:
    """Recursively reject structural semantic-extraction fields fail-closed.

    Generic words such as "event" are rejected only when they appear as a
    structural field (a mapping key), never when they occur in ordinary
    repair/uncertainty prose.  Strategic abstraction terms are rejected
    wherever they appear as standalone tokens.
    """
    if path.endswith(".input_metadata") or path == "input_metadata":
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key in FORBIDDEN_SEMANTIC_EXTRACTION_KEYS:
                raise ValueError(
                    "mechanical Pass 1 response contains a semantic "
                    f"extraction field {key!r} at {path}",
                )
            _validate_no_semantic_extraction(
                item, path=f"{path}.{key}",
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_no_semantic_extraction(item, path=f"{path}[{index}]")


def _seal_mechanical_provenance(
    selected: Mapping[str, Any],
    *,
    rationale: str,
) -> dict[str, Any]:
    """Deterministically seal the full Pass 1 provenance for one window.

    The provider is never asked to echo lineage.  Task kind, the exact
    target/source identity, prompt/schema/pipeline/config versions, and the
    field-level metadata adapter are all derived from the frozen Bronze
    inputs, so missing or wrong provider lineage can never fail or corrupt a
    window.  The model's one short rationale is the only judgment content.
    """
    provenance = build_mechanical_provenance(selected)
    provenance["rationale"] = rationale
    return provenance


def _validate_mechanical_provenance(
    provenance: object,
    *,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a sealed (normalized) mechanical provenance object."""
    if not isinstance(provenance, Mapping):
        raise ValueError("mechanical provenance must be an object")
    _require_exact_keys(
        provenance,
        (
            "task_kind", "target", "prompt_version", "schema_version",
            "pipeline_version", "config_version", "input_metadata",
            "rationale",
        ),
        "mechanical provenance",
    )
    if provenance["task_kind"] != TEXT_RESTORATION_TASK_KIND:
        raise ValueError("mechanical provenance task_kind must be TEXT_RESTORATION")
    if provenance["prompt_version"] != MECHANICAL_PROMPT_VERSION:
        raise ValueError("mechanical provenance prompt_version is invalid")
    if provenance["schema_version"] != MECHANICAL_RESPONSE_SCHEMA_VERSION:
        raise ValueError("mechanical provenance schema_version is invalid")
    if provenance["pipeline_version"] != PIPELINE_VERSION:
        raise ValueError("mechanical provenance pipeline_version is invalid")
    if provenance["config_version"] != CONFIG_VERSION:
        raise ValueError("mechanical provenance config_version is invalid")
    if provenance["input_metadata"] != build_metadata_adapter(selected):
        raise ValueError(
            "mechanical provenance input_metadata must exactly echo the "
            "supplied metadata adapter",
        )
    _validate_metadata_adapter(
        provenance["input_metadata"],
        selected=selected,
        label="mechanical provenance",
    )
    target = provenance["target"]
    expected_target = _target_identity(selected)
    if not isinstance(target, Mapping):
        raise ValueError("mechanical provenance target must be an object")
    _require_exact_keys(
        target,
        (
            "window_id", "source_group_id", "canonical_record_sha256",
            "upstream_start", "upstream_end", "upstream_content_sha256",
            "bronze_text", "bronze_text_sha256",
        ),
        "mechanical provenance target",
    )
    if target != expected_target:
        raise ValueError(
            "mechanical provenance target must exactly match the supplied "
            "window/source/hash identity",
        )
    rationale = _require_nonempty_string(
        provenance["rationale"], "mechanical provenance rationale",
    )
    if _contains_semantic_endpoint_list(rationale):
        raise ValueError(
            "mechanical provenance rationale enumerates a semantic "
            "endpoint/extraction list",
        )
    _validate_no_semantic_extraction(provenance, path="provenance")
    return dict(provenance)


def _occurrences(text: str, needle: str) -> list[int]:
    """Deterministic non-overlapping exact-substring occurrence starts."""
    if not needle:
        return []
    starts: list[int] = []
    position = 0
    while True:
        found = text.find(needle, position)
        if found == -1:
            break
        starts.append(found)
        position = found + len(needle)
    return starts


def _clean_application_diff_message(
    *,
    applied: str,
    requested: str,
    label: str,
    max_snippet: int = 120,
    max_changes: int = 5,
    max_text: int = 240,
) -> str:
    """Concise bounded diff feedback for clean_text/application mismatches.

    Reports the applied and requested texts (truncated) plus the ordered
    non-equal ``difflib`` opcodes with exact differing substrings and
    positions.  Each opcode is also expanded to the surrounding full word
    in the applied and requested texts with a concrete diagnostic
    suggestion such as ``original_text="exhaust" replacement="Exhaust"``
    (or the minimal full replaceable span including adjacent punctuation
    for insertions/deletions), so a correction prompt can act on every
    missing edit.  Suggestions are diagnostic only: the provider must still
    return an explicit repair and nothing is auto-created.  Sizes are
    bounded so errors/correction prompts cannot explode.
    """
    matcher = difflib.SequenceMatcher(a=applied, b=requested, autojunk=False)
    changes: list[str] = []
    for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        a_snippet = applied[a_start:a_end]
        b_snippet = requested[b_start:b_end]
        if len(a_snippet) > max_snippet:
            a_snippet = a_snippet[:max_snippet] + "..."
        if len(b_snippet) > max_snippet:
            b_snippet = b_snippet[:max_snippet] + "..."
        entry = (
            f"{tag} applied[{a_start}:{a_end}]={a_snippet!r} "
            f"requested[{b_start}:{b_end}]={b_snippet!r}"
        )
        a_low, a_high = _expand_to_word(applied, a_start, a_end)
        b_low, b_high = _expand_to_word(requested, b_start, b_end)
        a_word = applied[a_low:a_high]
        b_word = requested[b_low:b_high]
        if a_word and b_word and a_word != b_word:
            if len(a_word) > max_snippet:
                a_word = a_word[:max_snippet] + "..."
            if len(b_word) > max_snippet:
                b_word = b_word[:max_snippet] + "..."
            entry += (
                f" full_word applied[{a_low}:{a_high}]={a_word!r} "
                f"requested[{b_low}:{b_high}]={b_word!r} "
                f'suggest original_text="{a_word}" replacement="{b_word}"'
            )
        changes.append(entry)
        if len(changes) >= max_changes:
            changes.append("... further differences truncated")
            break

    def _bounded(text: str) -> str:
        if len(text) > max_text:
            return text[:max_text] + "..."
        return text

    return (
        f"{label}: applied={_bounded(applied)!r} "
        f"requested={_bounded(requested)!r} "
        f"ordered_non_equal_changes={changes!r}"
    )


def _expand_to_word(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand ``[start, end)`` to the smallest surrounding full word.

    Word characters are alphanumerics plus an internal apostrophe, so
    contractions stay intact.  Adjacent punctuation is included when it is
    part of the differing region or abuts the word, yielding the minimal
    full replaceable span for punctuation insertions/deletions.
    """
    low = min(start, len(text))
    high = min(end, len(text))
    while low > 0 and (text[low - 1].isalnum() or text[low - 1] == "'"):
        low -= 1
    while high < len(text) and (text[high].isalnum() or text[high] == "'"):
        high += 1
    return low, high


def _validate_repair_proposal(value: object, *, index: int) -> dict[str, Any]:
    """Validate one compact model repair proposal (judgment fields only)."""
    if not isinstance(value, Mapping):
        raise ValueError("mechanical repair proposal must be an object")
    _require_exact_keys(
        value,
        (
            "original_text", "replacement", "repair_type", "confidence",
            "rationale",
        ),
        f"mechanical repair proposal {index}",
    )
    original_text = _require_nonempty_string(
        value["original_text"], f"repair proposal {index} original_text",
    )
    replacement = _require_nonempty_string(
        value["replacement"], f"repair proposal {index} replacement",
    )
    if replacement == original_text:
        raise ValueError(
            f"repair proposal {index} replacement must differ from "
            "original_text",
        )
    repair_type = _require_enum(
        value["repair_type"],
        MECHANICAL_REPAIR_TYPES,
        f"repair proposal {index} repair_type",
    )
    if repair_type in FORBIDDEN_MECHANICAL_REPAIR_TYPES:
        raise ValueError(
            f"repair proposal {index} emits a forbidden mechanical repair type",
        )
    confidence = _require_enum(
        value["confidence"],
        REPAIR_CONFIDENCE_LEVELS,
        f"repair proposal {index} confidence",
    )
    rationale = _require_string(
        value["rationale"], f"repair proposal {index} rationale",
    )
    if _contains_semantic_endpoint_list(rationale):
        raise ValueError(
            f"repair proposal {index} rationale enumerates a semantic "
            "endpoint/extraction list",
        )
    return {
        "original_text": original_text,
        "replacement": replacement,
        "repair_type": repair_type,
        "confidence": confidence,
        "rationale": rationale,
    }


def _bind_repair_proposals(
    proposals: list[Mapping[str, Any]],
    *,
    bronze_text: str,
    selected: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Deterministically bind ordered repair proposals to exact Bronze spans.

    Coordinates, source-absolute offsets, evidence spans, IDs, and lineage
    are derived only from Bronze and the proposal text; the model's quoted
    original text is treated as untrusted proposal data.  Proposals bind in
    order: the k-th proposal quoting a given original text binds to the
    k-th occurrence of that exact text.  Unbindable or overlapping proposals
    fail closed rather than being silently re-bound or invented.
    """
    base_offset = int(selected["upstream_start"])
    bound: list[dict[str, Any]] = []
    occurrence_counters: dict[str, int] = {}
    for index, proposal in enumerate(proposals):
        original_text = proposal["original_text"]
        starts = _occurrences(bronze_text, original_text)
        occurrence = occurrence_counters.get(original_text, 0)
        if occurrence >= len(starts):
            raise ValueError(
                "mechanical repair proposal "
                f"{index} original_text cannot be bound deterministically "
                "to a unique Bronze slice",
            )
        occurrence_counters[original_text] = occurrence + 1
        local_start = starts[occurrence]
        local_end = local_start + len(original_text)
        bound.append({
            **proposal,
            "target_local_start": local_start,
            "target_local_end": local_end,
            "source_absolute_start": base_offset + local_start,
            "source_absolute_end": base_offset + local_end,
            "evidence_spans": [{
                "target_local_start": local_start,
                "target_local_end": local_end,
                "text": original_text,
            }],
            "proposal_index": index,
            "occurrence_index": occurrence,
        })
    bound.sort(key=lambda item: (
        item["target_local_start"], item["target_local_end"],
    ))
    for left, right in zip(bound, bound[1:]):
        if left["target_local_end"] > right["target_local_start"]:
            raise ValueError("mechanical repairs must not overlap")
    for position, repair in enumerate(bound, 1):
        repair["repair_id"] = f"p2k:mech:r{position:04d}"
    return bound


def _validate_uncertainty_proposal(value: object, *, index: int) -> dict[str, Any]:
    """Validate one compact model uncertainty proposal (judgment only)."""
    if not isinstance(value, Mapping):
        raise ValueError("mechanical uncertainty proposal must be an object")
    _require_exact_keys(
        value,
        ("surface_text", "uncertainty_type", "alternatives", "note"),
        f"mechanical uncertainty proposal {index}",
    )
    surface_text = _require_nonempty_string(
        value["surface_text"], f"uncertainty proposal {index} surface_text",
    )
    uncertainty_type = _require_enum(
        value["uncertainty_type"],
        MECHANICAL_UNCERTAINTY_TYPES,
        f"uncertainty proposal {index} uncertainty_type",
    )
    alternatives = _require_list(
        value["alternatives"], f"uncertainty proposal {index} alternatives",
    )
    if not alternatives:
        raise ValueError(
            f"uncertainty proposal {index} requires at least one alternative",
        )
    validated_alternatives = []
    for alternative in alternatives:
        if not isinstance(alternative, Mapping):
            raise ValueError("uncertainty alternative must be an object")
        _require_exact_keys(
            alternative, ("text", "confidence"), "uncertainty alternative",
        )
        validated_alternatives.append({
            "text": _require_nonempty_string(
                alternative["text"], "uncertainty alternative text",
            ),
            "confidence": _require_enum(
                alternative["confidence"],
                CONFIDENCE_LEVELS,
                "uncertainty alternative confidence",
            ),
        })
    note = _require_string(
        value["note"], f"uncertainty proposal {index} note",
    )
    return {
        "surface_text": surface_text,
        "uncertainty_type": uncertainty_type,
        "alternatives": validated_alternatives,
        "note": note,
    }


def _bind_uncertainty_proposals(
    proposals: list[Mapping[str, Any]],
    *,
    bronze_text: str,
    selected: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Deterministically bind ordered uncertainty proposals to Bronze spans.

    The same exact-quote/order rule as repair binding is used.  The cap is
    enforced here so a provider cannot flood the envelope with manufactured
    alternatives; an empty list is always valid.
    """
    if len(proposals) > MECHANICAL_UNCERTAINTY_CAP:
        raise ValueError(
            "mechanical uncertainty proposals exceed the cap of "
            f"{MECHANICAL_UNCERTAINTY_CAP}",
        )
    base_offset = int(selected["upstream_start"])
    bound: list[dict[str, Any]] = []
    occurrence_counters: dict[str, int] = {}
    for index, proposal in enumerate(proposals):
        surface_text = proposal["surface_text"]
        starts = _occurrences(bronze_text, surface_text)
        occurrence = occurrence_counters.get(surface_text, 0)
        if occurrence >= len(starts):
            raise ValueError(
                "mechanical uncertainty proposal "
                f"{index} surface_text cannot be bound deterministically "
                "to a unique Bronze slice",
            )
        occurrence_counters[surface_text] = occurrence + 1
        local_start = starts[occurrence]
        local_end = local_start + len(surface_text)
        bound.append({
            "target_local_start": local_start,
            "target_local_end": local_end,
            "source_absolute_start": base_offset + local_start,
            "source_absolute_end": base_offset + local_end,
            "text": surface_text,
            "uncertainty_type": proposal["uncertainty_type"],
            "alternatives": proposal["alternatives"],
            "evidence": [{
                "target_local_start": local_start,
                "target_local_end": local_end,
                "text": surface_text,
            }],
            "note": proposal["note"],
            "proposal_index": index,
            "occurrence_index": occurrence,
        })
    bound.sort(key=lambda item: (
        item["target_local_start"], item["target_local_end"],
    ))
    for position, uncertainty in enumerate(bound, 1):
        uncertainty["uncertainty_id"] = f"p2k:mech:u{position:04d}"
    return bound


def _validate_mechanical_uncertainties(
    value: object,
    *,
    bronze_text: str,
    selected: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate normalized (sealed) uncertainties in a stored B record."""
    items = _require_list(value, "mechanical uncertainties")
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    base_offset = int(selected["upstream_start"])
    for uncertainty in items:
        if not isinstance(uncertainty, Mapping):
            raise ValueError("mechanical uncertainty must be an object")
        _require_exact_keys(
            uncertainty,
            (
                "uncertainty_id", "target_local_start", "target_local_end",
                "source_absolute_start", "source_absolute_end", "text",
                "uncertainty_type", "alternatives", "evidence", "note",
                "proposal_index", "occurrence_index",
            ),
            "mechanical uncertainty",
        )
        uncertainty_id = _require_nonempty_string(
            uncertainty["uncertainty_id"], "uncertainty_id",
        )
        if uncertainty_id in seen_ids:
            raise ValueError("mechanical uncertainty IDs must be unique")
        seen_ids.add(uncertainty_id)
        uncertainty_type = _require_enum(
            uncertainty["uncertainty_type"],
            MECHANICAL_UNCERTAINTY_TYPES,
            "uncertainty_type",
        )
        local_start = _require_int(
            uncertainty["target_local_start"], "uncertainty local start",
            minimum=0,
        )
        local_end = _require_int(
            uncertainty["target_local_end"], "uncertainty local end", minimum=0,
        )
        if not 0 <= local_start < local_end <= len(bronze_text):
            raise ValueError("mechanical uncertainty local offsets are invalid")
        text = _require_string(uncertainty["text"], "uncertainty text")
        if bronze_text[local_start:local_end] != text:
            raise ValueError("mechanical uncertainty text is not an exact slice")
        if uncertainty["source_absolute_start"] != base_offset + local_start:
            raise ValueError(
                "mechanical uncertainty source-absolute start is inconsistent",
            )
        if uncertainty["source_absolute_end"] != base_offset + local_end:
            raise ValueError(
                "mechanical uncertainty source-absolute end is inconsistent",
            )
        alternatives = _require_list(
            uncertainty["alternatives"], "uncertainty alternatives",
        )
        if not alternatives:
            raise ValueError(
                "mechanical uncertainty requires at least one alternative",
            )
        validated_alternatives = []
        for alternative in alternatives:
            if not isinstance(alternative, Mapping):
                raise ValueError("uncertainty alternative must be an object")
            _require_exact_keys(
                alternative, ("text", "confidence"), "uncertainty alternative",
            )
            validated_alternatives.append({
                "text": _require_nonempty_string(
                    alternative["text"], "uncertainty alternative text",
                ),
                "confidence": _require_enum(
                    alternative["confidence"],
                    CONFIDENCE_LEVELS,
                    "uncertainty alternative confidence",
                ),
            })
        evidence = _require_list(uncertainty["evidence"], "uncertainty evidence")
        validated_evidence = []
        for span in evidence:
            if not isinstance(span, Mapping):
                raise ValueError("uncertainty evidence span must be an object")
            _require_exact_keys(
                span,
                ("target_local_start", "target_local_end", "text"),
                "uncertainty evidence span",
            )
            span_start = _require_int(
                span["target_local_start"], "uncertainty evidence start",
                minimum=0,
            )
            span_end = _require_int(
                span["target_local_end"], "uncertainty evidence end", minimum=0,
            )
            if not 0 <= span_start < span_end <= len(bronze_text):
                raise ValueError("uncertainty evidence offsets are invalid")
            if bronze_text[span_start:span_end] != span["text"]:
                raise ValueError(
                    "uncertainty evidence span is not an exact slice",
                )
            validated_evidence.append({
                "target_local_start": span_start,
                "target_local_end": span_end,
                "text": span["text"],
            })
        note = _require_string(uncertainty["note"], "uncertainty note")
        proposal_index = _require_int(
            uncertainty["proposal_index"], "uncertainty proposal_index",
            minimum=0,
        )
        occurrence_index = _require_int(
            uncertainty["occurrence_index"], "uncertainty occurrence_index",
            minimum=0,
        )
        validated.append({
            "uncertainty_id": uncertainty_id,
            "target_local_start": local_start,
            "target_local_end": local_end,
            "source_absolute_start": base_offset + local_start,
            "source_absolute_end": base_offset + local_end,
            "text": text,
            "uncertainty_type": uncertainty_type,
            "alternatives": validated_alternatives,
            "evidence": validated_evidence,
            "note": note,
            "proposal_index": proposal_index,
            "occurrence_index": occurrence_index,
        })
    validated.sort(key=lambda item: (
        item["target_local_start"], item["target_local_end"],
    ))
    return validated


def _normalize_mechanical_response(
    parsed: Mapping[str, Any],
    *,
    bronze_text: str,
    selected: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any],
    list[dict[str, Any]],
]:
    """Validate the compact provider envelope and deterministically seal it.

    Returns ``(repairs, uncertainties, provenance, raw_proposals, hints)``.
    Only the model's judgment fields are trusted: ``clean_text``, ordered
    repair proposals, uncertainty proposals, and a short rationale.  Every
    offset, evidence span, ID, and provenance field is recomputed from Bronze
    and the sealed frozen inputs.  The validated raw proposals are returned
    verbatim as the audit trail between the content-addressed raw response
    and the normalized mechanical record.  Eligible champion-spelling hints
    are recomputed deterministically and every hint must be represented by
    exactly one explicit DOMAIN_SPELLING repair.
    """
    _validate_no_semantic_extraction(parsed, path="response")
    _require_exact_keys(
        parsed,
        ("schema_version", "clean_text", "repairs", "uncertainties", "rationale"),
        "phase2k mechanical response",
    )
    if parsed["schema_version"] != MECHANICAL_RESPONSE_SCHEMA_VERSION:
        raise ValueError("phase2k mechanical response schema version is invalid")
    clean_text = _require_string(parsed["clean_text"], "mechanical clean_text")
    repair_proposals = [
        _validate_repair_proposal(item, index=index)
        for index, item in enumerate(_require_list(
            parsed["repairs"], "mechanical repairs",
        ))
    ]
    repairs = _bind_repair_proposals(
        repair_proposals,
        bronze_text=bronze_text,
        selected=selected,
    )
    hints = detect_champion_alias_hints(bronze_text, selected)
    _validate_eligible_hint_repairs(
        repairs, hints, label="phase2k mechanical response",
    )
    applied_clean = apply_mechanical_repairs(bronze_text, repairs)
    if applied_clean != clean_text:
        raise ValueError(
            "mechanical clean_text must equal the deterministic application "
            "of the repair proposals to the exact Bronze target; every "
            "difference must be represented by exactly one proposal. "
            + _clean_application_diff_message(
                applied=applied_clean,
                requested=clean_text,
                label="mechanical clean_text mismatch",
            ),
        )
    uncertainty_proposals = [
        _validate_uncertainty_proposal(item, index=index)
        for index, item in enumerate(_require_list(
            parsed["uncertainties"], "mechanical uncertainties",
        ))
    ]
    uncertainties = _bind_uncertainty_proposals(
        uncertainty_proposals,
        bronze_text=bronze_text,
        selected=selected,
    )
    rationale = _require_nonempty_string(
        parsed["rationale"], "mechanical rationale",
    )
    if _contains_semantic_endpoint_list(rationale):
        raise ValueError(
            "mechanical rationale enumerates a semantic endpoint/extraction "
            "list",
        )
    provenance = _seal_mechanical_provenance(selected, rationale=rationale)
    raw_proposals = {
        "clean_text": clean_text,
        "repairs": repair_proposals,
        "uncertainties": uncertainty_proposals,
        "rationale": rationale,
    }
    return repairs, uncertainties, provenance, raw_proposals, hints


def apply_mechanical_repairs(
    bronze_text: str,
    repairs: list[Mapping[str, Any]],
) -> str:
    """Deterministically apply non-overlapping repairs in ascending order."""
    ordered = sorted(
        repairs,
        key=lambda item: (item["target_local_start"], item["target_local_end"]),
    )
    pieces: list[str] = []
    cursor = 0
    for repair in ordered:
        start = _require_int(repair["target_local_start"], "repair start", minimum=0)
        end = _require_int(repair["target_local_end"], "repair end", minimum=0)
        if not cursor <= start < end <= len(bronze_text):
            raise ValueError("repair application violates span ordering")
        if bronze_text[start:end] != repair["original_text"]:
            raise ValueError("repair application failed the exact round-trip")
        pieces.append(bronze_text[cursor:start])
        pieces.append(_require_string(repair["replacement"], "repair replacement"))
        cursor = end
    pieces.append(bronze_text[cursor:])
    return "".join(pieces)


def run_mechanical_cleanup(
    selected: Mapping[str, Any],
    *,
    chat: ChatCallable,
    config_hash: str,
    cache_dir: Path | None = None,
    inference_config: Mapping[str, Any] | None = None,
    lineage: Mapping[str, Any] | None = None,
    raw_response_dir: Path | None = None,
) -> dict[str, Any]:
    """Run target-only mechanical text restoration and return a B record."""
    bronze_text = selected["source_text"]
    system, user = build_mechanical_prompt(selected)

    def validator(parsed: Mapping[str, Any]) -> Any:
        result = _normalize_mechanical_response(
            parsed,
            bronze_text=bronze_text,
            selected=selected,
        )
        (
            repairs, _uncertainties, _provenance, _raw_proposals, _hints,
        ) = result
        cleaned = apply_mechanical_repairs(bronze_text, repairs)
        if text_sha256(cleaned) == text_sha256(bronze_text) and repairs:
            raise ValueError("mechanical repairs produced no text change")
        return result

    outcome = call_provider_with_corrections(
        chat,
        system=system,
        user=user,
        build_correction_prompt=lambda prior_raw, error, **kwargs: (
            build_mechanical_correction_prompt(
                selected, prior_raw, error,
            )
        ),
        validator=validator,
        label="phase2k mechanical cleanup",
        schema_version=MECHANICAL_RESPONSE_SCHEMA_VERSION,
        prompt_version=MECHANICAL_PROMPT_VERSION,
        correction_prompt_version=MECHANICAL_CORRECTION_PROMPT_VERSION,
        config_hash=config_hash,
        window_id=selected["window_id"],
        stage="mechanical_cleanup",
        cache_dir=cache_dir,
        inference_config=inference_config,
        lineage=lineage,
        raw_response_dir=raw_response_dir,
        max_corrections=MECHANICAL_MAX_CORRECTIONS,
    )
    repairs, uncertainties, provenance, raw_proposals, hints = (
        outcome["result"]
    )
    cleaned = apply_mechanical_repairs(bronze_text, repairs)
    final_call = outcome["final_attempt"]["model_call"]
    return {
        "mechanical_cleaned_text": cleaned,
        "mechanical_cleaned_text_sha256": text_sha256(cleaned),
        "mechanical_cleaned_char_length": len(cleaned),
        "repairs": repairs,
        "repair_count": len(repairs),
        "lexical_hints": hints,
        "lexical_hint_count": len(hints),
        "uncertainties": uncertainties,
        "uncertainty_count": len(uncertainties),
        "provenance": provenance,
        "raw_proposals": raw_proposals,
        "model_call": {
            "source": final_call["source"],
            "prompt_hash": final_call["prompt_hash"],
            "cache_key": final_call["cache_key"],
            "config_hash": final_call["config_hash"],
            "inference_config": final_call["inference_config"],
            "inference_config_hash": final_call["inference_config_hash"],
            "inference_config_version": final_call["inference_config_version"],
            "prompt_version": final_call["prompt_version"],
            "schema_version": final_call["schema_version"],
            "attempt_index": final_call["attempt_index"],
            "attempt_kind": final_call["attempt_kind"],
            "raw_response_sha256": final_call["raw_response_sha256"],
            "raw_response_path": final_call["raw_response_path"],
            "status": final_call["status"],
            "error": final_call["error"],
        },
        "attempts": outcome["attempts"],
        "pipeline_version": PIPELINE_VERSION,
        "config_version": CONFIG_VERSION,
    }


# ---------------------------------------------------------------------------
# Semantic sufficiency diagnostics
# ---------------------------------------------------------------------------


def _validate_evidence_list(
    value: object,
    *,
    transcript: str,
    context: Mapping[str, Any],
    label: str,
) -> list[dict[str, Any]]:
    items = _require_list(value, label)
    validated = []
    for span in items:
        validated.append(_validate_evidence_span(
            span, transcript=transcript, context=context, label=label,
        ))
    return validated


def _validate_candidate(
    candidate: object,
    *,
    transcript: str,
    context: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise ValueError(f"{label} candidate must be an object")
    _require_exact_keys(
        candidate,
        ("candidate", "confidence", "evidence_spans"),
        f"{label} candidate",
    )
    _require_nonempty_string(candidate["candidate"], f"{label} candidate text")
    confidence = _require_enum(
        candidate["confidence"], CONFIDENCE_LEVELS, f"{label} candidate confidence",
    )
    evidence = _validate_evidence_list(
        candidate["evidence_spans"],
        transcript=transcript,
        context=context,
        label=f"{label} candidate evidence",
    )
    return {
        "candidate": candidate["candidate"],
        "confidence": confidence,
        "evidence_spans": evidence,
    }


def _validate_slot(
    slot: object,
    *,
    slot_key: str,
    transcript: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(slot, Mapping):
        raise ValueError(f"slot {slot_key} must be an object")
    _require_exact_keys(
        slot,
        ("status", "candidates", "confidence", "evidence_spans"),
        f"slot {slot_key}",
    )
    status = _require_enum(slot["status"], SLOT_STATUSES, f"slot {slot_key} status")
    confidence = _require_enum(
        slot["confidence"], CONFIDENCE_LEVELS, f"slot {slot_key} confidence",
    )
    candidates = _require_list(slot["candidates"], f"slot {slot_key} candidates")
    validated_candidates = []
    for candidate in candidates:
        validated_candidates.append(_validate_candidate(
            candidate,
            transcript=transcript,
            context=context,
            label=f"slot {slot_key} candidate",
        ))
    evidence = _validate_evidence_list(
        slot["evidence_spans"],
        transcript=transcript,
        context=context,
        label=f"slot {slot_key} evidence",
    )
    if status in ("UNKNOWN", "CONTEXT_INSUFFICIENT") and validated_candidates:
        raise ValueError(f"slot {slot_key} unresolved status cannot carry candidates")
    if status == "RESOLVED" and not validated_candidates:
        raise ValueError(f"slot {slot_key} RESOLVED requires at least one candidate")
    return {
        "status": status,
        "candidates": validated_candidates,
        "confidence": confidence,
        "evidence_spans": evidence,
    }


def _validate_metadata_conflicts(value: object, label: str) -> list[dict[str, Any]]:
    conflicts = _require_list(value, label)
    validated = []
    for conflict in conflicts:
        if not isinstance(conflict, Mapping):
            raise ValueError(f"{label} conflict must be an object")
        _require_exact_keys(
            conflict,
            ("field", "metadata_value", "context_evidence", "note"),
            f"{label} conflict",
        )
        _require_nonempty_string(conflict["field"], f"{label} conflict field")
        _require_nonempty_string(
            conflict["metadata_value"], f"{label} conflict metadata value",
        )
        _require_string(conflict["context_evidence"], f"{label} conflict evidence")
        _require_string(conflict["note"], f"{label} conflict note")
        validated.append(dict(conflict))
    return validated


def _bind_evidence_quotes(
    quotes: object,
    *,
    context: Mapping[str, Any],
    label: str,
    allow_normalized: bool = False,
    anchor_span: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Deterministically bind compact evidence quotes to exact source spans.

    The model never counts offsets.  Each quote must be an exact contiguous
    substring of one supplied context segment.  Occurrences are collected in
    left-to-right segment order and the k-th quote of a given surface binds
    to the k-th occurrence.  A quote is fail-closed when it is absent, when
    it spans a segment boundary, or when the number of supplied quotes for a
    repeated surface does not exactly equal its occurrence count (the
    intended occurrence cannot be resolved deterministically).

    When ``allow_normalized`` is set (compact reconstruction evidence
    quotes only), exact binding remains first choice and a deterministic
    surface-normalized fallback ignores only capitalization, punctuation,
    Unicode-space variants, and whitespace runs.  The final artifact always
    stores the exact source slice and offsets; the raw compact envelope
    preserves the provider's quote.

    When ``anchor_span`` is supplied (contextual repair evidence only), an
    exact quote that the all-context occurrence-count rule would otherwise
    reject is accepted when exactly one exact context occurrence contains
    the repair's source-absolute span; that exact context occurrence is
    stored.  Zero or multiple containing occurrences preserve the existing
    fail-closed ambiguity/absence behavior, and the anchor never applies to
    binding, statement, or general evidence ambiguity rules.
    """
    items = _require_list(quotes, f"{label} evidence_quotes")
    if not items:
        return []
    surfaces: list[str] = []
    for item in items:
        quote = _require_string(item, f"{label} evidence quote")
        if not quote:
            raise ValueError(f"{label} evidence quote must be non-empty")
        surfaces.append(quote)
    segment_texts = [
        _require_string(segment["text"], "context segment text")
        for segment in context["segments"]
    ]
    segment_starts = [
        _require_int(
            segment["source_absolute_start"], "segment source start", minimum=0,
        )
        for segment in context["segments"]
    ]
    exact_span_lists: dict[str, list[tuple[int, int, int]]] = {}
    for quote in surfaces:
        spans: list[tuple[int, int, int]] = []
        for segment_index, segment_text in enumerate(segment_texts):
            for start, end in _surface_spans(
                segment_text, quote, normalized=False,
            ):
                spans.append((segment_index, start, end))
        exact_span_lists[quote] = spans
    anchored: dict[str, tuple[int, int, int]] = {}
    if allow_normalized and anchor_span is not None:
        anchor_start, anchor_end = anchor_span
        supplied_counts: dict[str, int] = {}
        for quote in surfaces:
            supplied_counts[quote] = supplied_counts.get(quote, 0) + 1
        for quote, supplied_count in supplied_counts.items():
            exact_total = len(exact_span_lists[quote])
            normalized_total = sum(
                len(_normalized_occurrence_spans(text, quote))
                for text in segment_texts
            )
            if (
                supplied_count == exact_total
                or (normalized_total == supplied_count and normalized_total > 0)
            ):
                # The deterministic all-context rule already resolves this
                # surface; the anchor is a rescue for rejected quotes only.
                continue
            containing = [
                (segment_index, start, end)
                for segment_index, start, end in exact_span_lists[quote]
                if (
                    segment_starts[segment_index] + start <= anchor_start
                    and anchor_end <= segment_starts[segment_index] + end
                )
            ]
            if len(containing) == 1:
                anchored[quote] = containing[0]
    if allow_normalized:
        strategies = _surface_strategies(
            surfaces,
            sources=segment_texts,
            label=label,
            absent_guidance=(
                "Quote an exact contiguous context span from the supplied "
                "segments (repeat repeated quotes once per intended "
                "occurrence) or remove the proposal that depends on this "
                "quote."
            ),
            absent_suggestion_fn=(
                lambda quote: _unique_whitespace_slice_suggestion(
                    context["segments"], quote,
                )
            ),
            anchored_exact=anchored if anchored else None,
        )
    else:
        strategies = {quote: False for quote in surfaces}
        supplied: dict[str, int] = {}
        for quote in surfaces:
            supplied[quote] = supplied.get(quote, 0) + 1
        total_matches: dict[str, int] = {}
        for segment_text in segment_texts:
            for quote in supplied:
                total_matches[quote] = total_matches.get(quote, 0) + len(
                    _occurrences(segment_text, quote),
                )
        for quote, supplied_count in supplied.items():
            total = total_matches.get(quote, 0)
            if total == 0:
                raise ValueError(
                    f"{label} evidence quote {quote!r} is absent from the "
                    "supplied context",
                )
            if supplied_count != total:
                raise ValueError(
                    f"{label} evidence quote {quote!r} is ambiguous: it "
                    f"occurs {total} time(s) in the context but was "
                    f"supplied {supplied_count} time(s); repeat it once "
                    "per intended occurrence or quote a longer unique "
                    "span",
                )
    span_lists: dict[str, list[tuple[int, int, int]]] = {}
    for quote, normalized in strategies.items():
        spans: list[tuple[int, int, int]] = []
        for segment_index, segment_text in enumerate(segment_texts):
            for start, end in _surface_spans(
                segment_text, quote, normalized=normalized,
            ):
                spans.append((segment_index, start, end))
        span_lists[quote] = spans
    validated: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    for item in items:
        quote = _require_string(item, f"{label} evidence quote")
        anchored_span = anchored.get(quote)
        if anchored_span is not None:
            segment_index, start, end = anchored_span
        else:
            occurrence = counters.get(quote, 0)
            counters[quote] = occurrence + 1
            segment_index, start, end = span_lists[quote][occurrence]
        segment = context["segments"][segment_index]
        segment_start = _require_int(
            segment["source_absolute_start"], "segment source start",
            minimum=0,
        )
        validated.append({
            "segment_id": segment["segment_id"],
            "source_absolute_start": segment_start + start,
            "source_absolute_end": segment_start + end,
            "text": segment["text"][start:end],
        })
    return validated


def _validate_compact_candidate(
    candidate: object,
    *,
    label: str,
) -> dict[str, Any]:
    """Validate one compact provider slot candidate (quotes, not offsets)."""
    if not isinstance(candidate, Mapping):
        raise ValueError(f"{label} candidate must be an object")
    _require_exact_keys(
        candidate,
        ("candidate", "confidence", "evidence_quotes"),
        f"{label} candidate",
    )
    _require_nonempty_string(candidate["candidate"], f"{label} candidate text")
    confidence = _require_enum(
        candidate["confidence"], CONFIDENCE_LEVELS, f"{label} candidate confidence",
    )
    return {
        "candidate": candidate["candidate"],
        "confidence": confidence,
        "evidence_quotes": candidate["evidence_quotes"],
    }


def _validate_compact_slot(
    slot: object,
    *,
    slot_key: str,
) -> dict[str, Any]:
    """Validate one compact provider slot envelope (quotes, not offsets)."""
    if not isinstance(slot, Mapping):
        raise ValueError(f"slot {slot_key} must be an object")
    _require_exact_keys(
        slot,
        ("status", "candidates", "confidence", "evidence_quotes"),
        f"slot {slot_key}",
    )
    status = _require_enum(slot["status"], SLOT_STATUSES, f"slot {slot_key} status")
    confidence = _require_enum(
        slot["confidence"], CONFIDENCE_LEVELS, f"slot {slot_key} confidence",
    )
    candidates = _require_list(slot["candidates"], f"slot {slot_key} candidates")
    validated_candidates = [
        _validate_compact_candidate(
            candidate, label=f"slot {slot_key} candidate",
        )
        for candidate in candidates
    ]
    return {
        "status": status,
        "candidates": validated_candidates,
        "confidence": confidence,
        "evidence_quotes": slot["evidence_quotes"],
    }


def normalize_sufficiency_compact_response(
    parsed: Mapping[str, Any],
    *,
    transcript: str,
    context: Mapping[str, Any],
    at_max_context: bool,
) -> dict[str, Any]:
    """Validate the compact v2 provider envelope and seal exact source spans.

    The provider supplies categorical confidence and exact evidence quotes
    only; deterministic code binds every quote to an exact source span
    within the supplied context and then revalidates the normalized object
    with the strict conceptual schema used downstream.
    """
    _require_exact_keys(
        parsed,
        ("schema_version", "decision", "slots", "metadata_conflicts", "rationale"),
        "phase2k sufficiency response",
    )
    if parsed["schema_version"] != SUFFICIENCY_RESPONSE_SCHEMA_VERSION:
        raise ValueError("phase2k sufficiency response schema version is invalid")
    decision = _require_enum(
        parsed["decision"], SUFFICIENCY_DECISIONS, "sufficiency decision",
    )
    slots = parsed["slots"]
    if not isinstance(slots, Mapping) or set(slots) != set(SLOT_KEYS):
        raise ValueError("sufficiency slots key set is invalid")
    normalized_slots: dict[str, Any] = {}
    for slot_key in SLOT_KEYS:
        compact = _validate_compact_slot(slots[slot_key], slot_key=slot_key)
        validated_candidates = []
        for candidate in compact["candidates"]:
            evidence = _bind_evidence_quotes(
                candidate["evidence_quotes"],
                context=context,
                label=f"slot {slot_key} candidate",
            )
            validated_candidates.append({
                "candidate": candidate["candidate"],
                "confidence": candidate["confidence"],
                "evidence_spans": evidence,
            })
        slot_evidence = _bind_evidence_quotes(
            compact["evidence_quotes"],
            context=context,
            label=f"slot {slot_key}",
        )
        normalized_slots[slot_key] = {
            "status": compact["status"],
            "candidates": validated_candidates,
            "confidence": compact["confidence"],
            "evidence_spans": slot_evidence,
        }
    normalized = {
        "schema_version": SUFFICIENCY_NORMALIZED_SCHEMA_VERSION,
        "decision": decision,
        "slots": normalized_slots,
        "metadata_conflicts": parsed["metadata_conflicts"],
        "rationale": parsed["rationale"],
    }
    return validate_sufficiency_response(
        normalized,
        transcript=transcript,
        context=context,
        at_max_context=at_max_context,
    )


def validate_sufficiency_response(
    parsed: Mapping[str, Any],
    *,
    transcript: str,
    context: Mapping[str, Any],
    at_max_context: bool,
) -> dict[str, Any]:
    _require_exact_keys(
        parsed,
        ("schema_version", "decision", "slots", "metadata_conflicts", "rationale"),
        "phase2k sufficiency response",
    )
    if parsed["schema_version"] != SUFFICIENCY_NORMALIZED_SCHEMA_VERSION:
        raise ValueError("phase2k sufficiency normalized schema version is invalid")
    decision = _require_enum(
        parsed["decision"], SUFFICIENCY_DECISIONS, "sufficiency decision",
    )
    slots = parsed["slots"]
    if not isinstance(slots, Mapping) or set(slots) != set(SLOT_KEYS):
        raise ValueError("sufficiency slots key set is invalid")
    validated_slots: dict[str, Any] = {}
    for slot_key in SLOT_KEYS:
        validated_slots[slot_key] = _validate_slot(
            slots[slot_key],
            slot_key=slot_key,
            transcript=transcript,
            context=context,
        )
    conflicts = _validate_metadata_conflicts(
        parsed["metadata_conflicts"], "sufficiency metadata conflicts",
    )
    rationale = _require_string(parsed["rationale"], "sufficiency rationale")

    slot_statuses = {
        key: validated_slots[key]["status"] for key in SLOT_KEYS
    }
    any_unresolved = any(
        status in UNRESOLVED_SLOT_STATUSES for status in slot_statuses.values()
    )
    any_non_resolved = any(
        status != "RESOLVED" for status in slot_statuses.values()
    )
    if decision == "SUFFICIENT":
        if any_non_resolved:
            raise ValueError(
                "SUFFICIENT decision requires every slot to be RESOLVED",
            )
    elif decision == "MAX_CONTEXT_BUT_UNRESOLVED":
        if not at_max_context:
            raise ValueError("MAX_CONTEXT_BUT_UNRESOLVED requires max context")
        if not any_non_resolved:
            raise ValueError(
                "MAX_CONTEXT_BUT_UNRESOLVED requires at least one unresolved slot",
            )
    elif decision in NEEDS_MORE_DECISIONS:
        if at_max_context:
            raise ValueError(f"{decision} is not valid at max context")
        if not any_non_resolved:
            raise ValueError(
                f"{decision} requires at least one non-resolved slot",
            )
    return {
        "decision": decision,
        "slots": validated_slots,
        "metadata_conflicts": conflicts,
        "rationale": rationale,
    }


def run_sufficiency_diagnostic(
    selected: Mapping[str, Any],
    *,
    transcript: str,
    context: Mapping[str, Any],
    mechanical_cleaned_text: str,
    at_max_context: bool,
    stage_label: str,
    chat: ChatCallable,
    config_hash: str,
    cache_dir: Path | None = None,
    inference_config: Mapping[str, Any] | None = None,
    lineage: Mapping[str, Any] | None = None,
    raw_response_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one sufficiency diagnostic stage with bounded strict corrections.

    Returns ``{"attempts": [...], "final_attempt": {...}}``.  Every ordered
    attempt (including failed correction attempts) carries its own
    content-addressed raw response linkage, status, and exact error; the
    final attempt is the last successful one and keeps the normalized
    conceptual sufficiency response required downstream.
    """
    validate_context(context, transcript)
    system, user = build_sufficiency_prompt(
        selected,
        context,
        mechanical_cleaned_text,
        at_max_context=at_max_context,
    )

    def validator(parsed: Mapping[str, Any]) -> dict[str, Any]:
        return normalize_sufficiency_compact_response(
            parsed,
            transcript=transcript,
            context=context,
            at_max_context=at_max_context,
        )

    try:
        outcome = call_provider_with_corrections(
            chat,
            system=system,
            user=user,
            build_correction_prompt=lambda prior_raw, error, **kwargs: (
                build_sufficiency_correction_prompt(
                    selected,
                    context,
                    mechanical_cleaned_text,
                    at_max_context=at_max_context,
                    prior_raw=prior_raw,
                    error=error,
                )
            ),
            validator=validator,
            label="phase2k sufficiency diagnostic",
            schema_version=SUFFICIENCY_RESPONSE_SCHEMA_VERSION,
            prompt_version=SUFFICIENCY_PROMPT_VERSION,
            correction_prompt_version=SUFFICIENCY_CORRECTION_PROMPT_VERSION,
            config_hash=config_hash,
            window_id=selected["window_id"],
            stage=stage_label,
            cache_dir=cache_dir,
            inference_config=inference_config,
            lineage=lineage,
            raw_response_dir=raw_response_dir,
        )
    except ProviderCorrectionExhausted as exc:
        wrapped = [
            _sufficiency_attempt_record(
                record,
                selected=selected,
                stage_label=stage_label,
                context=context,
                at_max_context=at_max_context,
            )
            for record in exc.attempts
        ]
        raise ProviderCorrectionExhausted(str(exc), attempts=wrapped) from exc
    attempts = [
        _sufficiency_attempt_record(
            record,
            selected=selected,
            stage_label=stage_label,
            context=context,
            at_max_context=at_max_context,
        )
        for record in outcome["attempts"]
    ]
    return {"attempts": attempts, "final_attempt": attempts[-1]}


def _sufficiency_attempt_record(
    record: Mapping[str, Any],
    *,
    selected: Mapping[str, Any],
    stage_label: str,
    context: Mapping[str, Any],
    at_max_context: bool,
) -> dict[str, Any]:
    """Wrap one provider attempt into the full diagnostic attempt record."""
    attempt = {
        "attempt_id": (
            f"p2k:att:{selected['window_id']}:{stage_label}:"
            f"{record['attempt_index']}"
        ),
        "window_id": selected["window_id"],
        "stage": stage_label,
        "attempt_index": record["attempt_index"],
        "attempt_kind": record["attempt_kind"],
        "status": record["status"],
        "error": record["error"],
        "radius": context["radius"],
        "at_max_context": at_max_context,
        "context": context,
        "model_call": record["model_call"],
        "response": record["response"],
        "decision": (
            None
            if record["status"] != "OK"
            else record["response"]["parsed"]["decision"]
        ),
        "prompt_version": record["prompt_version"],
        "schema_version": record["schema_version"],
        "inference_config": record["model_call"]["inference_config"],
        "inference_config_hash": record["model_call"]["inference_config_hash"],
        "inference_config_version": record["model_call"][
            "inference_config_version"
        ],
        "pipeline_version": PIPELINE_VERSION,
        "config_version": CONFIG_VERSION,
    }
    attempt["content_sha256"] = canonical_sha256({
        key: value for key, value in attempt.items() if key != "content_sha256"
    })
    return attempt


def run_adaptive_diagnostics(
    selected: Mapping[str, Any],
    *,
    transcript: str,
    mechanical_cleaned_text: str,
    chat: ChatCallable,
    config_hash: str,
    cache_dir: Path | None = None,
    inference_config: Mapping[str, Any] | None = None,
    segments: list[dict[str, Any]] | None = None,
    lineage: Mapping[str, Any] | None = None,
    raw_response_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministic adaptive loop; returns (attempts, final attempt)."""
    attempts: list[dict[str, Any]] = []
    previous_radius: int | None = None
    following_radius: int | None = None
    for stage_index, stage in enumerate(RADIUS_STAGES):
        if stage_index == 0:
            previous_radius = stage["previous"]
            following_radius = stage["following"]
        elif previous_decision == "NEED_MORE_PREVIOUS_CONTEXT":
            previous_radius = stage["previous"]
        elif previous_decision == "NEED_MORE_FOLLOWING_CONTEXT":
            following_radius = stage["following"]
        elif previous_decision == "NEED_BOTH":
            previous_radius = stage["previous"]
            following_radius = stage["following"]
        else:  # SUFFICIENT would have stopped before advancing.
            raise ValueError("adaptive loop cannot advance after SUFFICIENT")
        context = retrieve_context(
            transcript,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=selected["upstream_start"],
            target_end=selected["upstream_end"],
            bronze_text=selected["source_text"],
            previous_segments=previous_radius,
            following_segments=following_radius,
            radius_label=stage["radius"],
            segments=segments,
        )
        try:
            stage_outcome = run_sufficiency_diagnostic(
                selected,
                transcript=transcript,
                context=context,
                mechanical_cleaned_text=mechanical_cleaned_text,
                at_max_context=stage["max"],
                stage_label=stage["label"],
                chat=chat,
                config_hash=config_hash,
                cache_dir=cache_dir,
                inference_config=inference_config,
                lineage=lineage,
                raw_response_dir=raw_response_dir,
            )
        except ProviderCorrectionExhausted as exc:
            attempts.extend(exc.attempts)
            raise
        attempts.extend(stage_outcome["attempts"])
        attempt = stage_outcome["final_attempt"]
        if attempt["decision"] == "SUFFICIENT":
            return attempts, attempt
        if stage["max"]:
            if attempt["decision"] != "MAX_CONTEXT_BUT_UNRESOLVED":
                raise ValueError(
                    "max-context diagnostic must return "
                    "MAX_CONTEXT_BUT_UNRESOLVED when not sufficient",
                )
            return attempts, attempt
        previous_decision = attempt["decision"]
        if previous_decision not in NEEDS_MORE_DECISIONS:
            raise ValueError(f"unexpected diagnostic decision {previous_decision!r}")
    raise RuntimeError("adaptive loop exhausted without a terminal decision")


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def _candidate_mention_resolution_pairs(
    candidate: str,
) -> list[tuple[str, str]]:
    """Parse diagnostic candidates into (mention-side, resolution-side) pairs.

    Supports explicit equations such as ``you = the player being coached
    (Lucian)``, compound equations such as ``she/her = enemy mid laner;
    you/your = Veigar player``, and annotated mentions such as
    ``I (Fizz player)``.  A candidate with no mention-side surface yields no
    pairs; the caller falls back to plain surface matching.
    """
    pairs: list[tuple[str, str]] = []
    for chunk in candidate.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        left: str | None = None
        right: str | None = None
        if "=" in chunk:
            left_raw, right_raw = chunk.split("=", 1)
            left = left_raw.strip()
            right = right_raw.strip()
        else:
            open_index = chunk.find("(")
            if open_index > 0 and chunk.endswith(")"):
                left = chunk[:open_index].strip()
                right = chunk[open_index + 1:-1].strip()
        if left and right:
            pairs.append((left, right))
    return pairs


def _mention_side_variants(mention_side: str) -> list[str]:
    """Deterministic mention-side surface variants for one side."""
    variants = [mention_side]
    if "/" in mention_side:
        variants.extend(
            part.strip() for part in mention_side.split("/") if part.strip()
        )
    return variants


def _mention_stem(mention: str) -> str | None:
    """Contracted mention stem: ``he's`` -> ``he`` (narrowly, for
    reference-resolution slots only)."""
    if "'" not in mention:
        return None
    stem = mention.split("'", 1)[0].strip()
    return stem or None


def _surface_license_left(
    candidate: str,
    mention: str,
) -> bool:
    """The candidate's mention-side surface licenses the binding mention."""
    if _surface_normalize(candidate) == _surface_normalize(mention):
        return True
    normalized_mention = _surface_normalize(mention)
    mention_stem = _mention_stem(mention)
    if mention_stem is not None and _surface_normalize(
        candidate,
    ) == _surface_normalize(mention_stem):
        return True
    for left, _right in _candidate_mention_resolution_pairs(candidate):
        for variant in _mention_side_variants(left):
            if _surface_normalize(variant) == normalized_mention:
                return True
            if (
                mention_stem is not None
                and _surface_normalize(variant) == _surface_normalize(mention_stem)
            ):
                return True
    return False


def _surface_license_right(
    candidate: str,
    resolved: str,
) -> bool:
    """The candidate's resolution-side surface licenses the resolved value."""
    if _surface_normalize(candidate) == _surface_normalize(resolved):
        return True
    normalized_resolved = _surface_normalize(resolved)
    return any(
        _surface_normalize(right) == normalized_resolved
        for _left, right in _candidate_mention_resolution_pairs(candidate)
    )


def _composed_reference_slots() -> frozenset[str]:
    """Reference-resolution slots eligible for composed candidate licensing.

    Pronouns, discourse references, and unresolved ASR participate when the
    binding mention itself is licensed by a candidate in its own diagnostic
    slot and the resolved candidate is licensed by an appropriate
    entity/principal-actor/other explicitly relevant diagnostic candidate or
    supplied metadata.  Ability ownership is intentionally excluded: the
    current data contract stores ownership as full-phrase candidates (for
    example ``Ignite is owned by the player (Lucian)``) rather than
    mention-licensed resolution candidates, so composed licensing would be
    meaningless there and direct candidate equality already applies.
    """
    return frozenset({"pronouns", "discourse_refs", "unresolved_asr"})


def _entity_resolution_slot_candidates(
    final_slots: Mapping[str, Any],
) -> list[str]:
    """Diagnostic candidates from entity/principal-actor slots."""
    entity_slots = ("champion_identities", "principal_actors")
    candidates: list[str] = []
    for slot in entity_slots:
        slot_value = final_slots.get(slot)
        if not isinstance(slot_value, Mapping):
            continue
        for item in slot_value.get("candidates", []):
            if isinstance(item, Mapping) and isinstance(
                item.get("candidate"), str,
            ):
                candidates.append(item["candidate"])
    return candidates


def _validate_binding(
    binding: object,
    *,
    index: int,
    bronze_text: str,
    base_offset: int,
    transcript: str,
    context: Mapping[str, Any],
    final_slots: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        raise ValueError(f"binding {index} must be an object")
    _require_exact_keys(
        binding,
        (
            "binding_id", "slot", "mention", "resolved_candidate",
            "resolved_status", "confidence", "evidence_spans", "alternatives",
            "metadata_contributed", "rationale",
        ),
        f"binding {index}",
    )
    binding_id = _require_nonempty_string(binding["binding_id"], f"binding {index} id")
    slot = _require_enum(binding["slot"], SLOT_KEYS, f"binding {index} slot")
    mention = _validate_bronze_span(
        binding["mention"],
        bronze_text=bronze_text,
        base_offset=base_offset,
        label=f"binding {index} mention",
    )
    resolved_status = _require_enum(
        binding["resolved_status"], BINDING_STATUSES, f"binding {index} status",
    )
    resolved_candidate = _require_string(
        binding["resolved_candidate"], f"binding {index} candidate",
    )
    confidence = _require_enum(
        binding["confidence"], CONFIDENCE_LEVELS, f"binding {index} confidence",
    )
    evidence = _validate_evidence_list(
        binding["evidence_spans"],
        transcript=transcript,
        context=context,
        label=f"binding {index} evidence",
    )
    alternatives = _require_list(
        binding["alternatives"], f"binding {index} alternatives",
    )
    validated_alternatives = []
    for alternative in alternatives:
        if not isinstance(alternative, Mapping):
            raise ValueError(f"binding {index} alternative must be an object")
        _require_exact_keys(
            alternative,
            ("candidate", "evidence_spans", "note"),
            f"binding {index} alternative",
        )
        validated_alternatives.append({
            "candidate": _require_nonempty_string(
                alternative["candidate"],
                f"binding {index} alternative candidate",
            ),
            "evidence_spans": _validate_evidence_list(
                alternative["evidence_spans"],
                transcript=transcript,
                context=context,
                label=f"binding {index} alternative evidence",
            ),
            "note": _require_string(
                alternative["note"], f"binding {index} alternative note",
            ),
        })
    metadata_contributed = _require_bool(
        binding["metadata_contributed"], f"binding {index} metadata flag",
    )
    rationale = _require_string(binding["rationale"], f"binding {index} rationale")

    slot_status = final_slots[slot]["status"]
    if resolved_status != slot_status:
        raise ValueError(
            f"binding {index} status {resolved_status} contradicts the final "
            f"diagnostic slot status {slot_status}",
        )
    candidates = [item["candidate"] for item in final_slots[slot]["candidates"]]
    if slot_status == "RESOLVED":
        metadata_values = _metadata_values(metadata)
        candidate_licensed = (
            resolved_candidate in candidates
            or resolved_candidate in metadata_values
            or any(
                _surface_normalize(item) == _surface_normalize(resolved_candidate)
                for item in candidates
            )
            or any(
                _surface_normalize(item) == _surface_normalize(resolved_candidate)
                for item in metadata_values
            )
        )
        if not candidate_licensed and slot in _composed_reference_slots():
            mention_licensed = any(
                _surface_license_left(item, mention["text"])
                for item in candidates
            )
            resolved_licensed = (
                resolved_candidate in metadata_values
                or any(
                    _surface_normalize(item)
                    == _surface_normalize(resolved_candidate)
                    for item in metadata_values
                )
                or any(
                    _surface_license_right(item, resolved_candidate)
                    for item in candidates
                )
                or any(
                    _surface_license_right(item, resolved_candidate)
                    for item in _entity_resolution_slot_candidates(final_slots)
                )
            )
            if mention_licensed and resolved_licensed:
                candidate_licensed = True
        if not candidate_licensed:
            raise ValueError(
                f"binding {index} resolved candidate is not licensed by the "
                "diagnostics or metadata: "
                f"slot={slot!r} mention={mention['text']!r} "
                f"candidate={resolved_candidate!r} "
                "allowed candidates="
                + ", ".join(repr(item) for item in candidates[:8])
                + (
                    f" and metadata values "
                    + ", ".join(repr(item) for item in sorted(metadata_values)[:8])
                    if metadata_values
                    else ""
                ),
            )
    else:
        if resolved_candidate != slot_status:
            raise ValueError(
                f"binding {index} unresolved candidate must equal its status",
            )
    return {
        "binding_id": binding_id,
        "slot": slot,
        "mention": mention,
        "resolved_candidate": resolved_candidate,
        "resolved_status": resolved_status,
        "confidence": confidence,
        "evidence_spans": evidence,
        "alternatives": validated_alternatives,
        "metadata_contributed": metadata_contributed,
        "rationale": rationale,
    }


def _validate_claim(
    claim: object,
    *,
    index: int,
    transcript: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(claim, Mapping):
        raise ValueError(f"claim {index} must be an object")
    _require_exact_keys(
        claim,
        (
            "claim_id", "slot", "claim", "condition", "consequence",
            "confidence", "evidence_spans",
        ),
        f"claim {index}",
    )
    _require_nonempty_string(claim["claim_id"], f"claim {index} id")
    _require_enum(claim["slot"], SLOT_KEYS, f"claim {index} slot")
    claim_text = _require_nonempty_string(claim["claim"], f"claim {index} text")
    condition = claim["condition"]
    consequence = claim["consequence"]
    if condition is not None:
        condition = _require_string(condition, f"claim {index} condition")
    if consequence is not None:
        consequence = _require_string(consequence, f"claim {index} consequence")
    confidence = _require_enum(
        claim["confidence"], CONFIDENCE_LEVELS, f"claim {index} confidence",
    )
    evidence = _validate_evidence_list(
        claim["evidence_spans"],
        transcript=transcript,
        context=context,
        label=f"claim {index} evidence",
    )
    return {
        "claim_id": claim["claim_id"],
        "slot": claim["slot"],
        "claim": claim_text,
        "condition": condition,
        "consequence": consequence,
        "confidence": confidence,
        "evidence_spans": evidence,
    }


def _validate_unsupported_claim(
    claim: object,
    *,
    index: int,
) -> dict[str, Any]:
    if not isinstance(claim, Mapping):
        raise ValueError(f"unsupported claim {index} must be an object")
    _require_exact_keys(
        claim,
        ("claim", "reason", "note"),
        f"unsupported claim {index}",
    )
    _require_nonempty_string(claim["claim"], f"unsupported claim {index} text")
    reason = _require_enum(
        claim["reason"], UNSUPPORTED_REASON_TYPES, f"unsupported claim {index} reason",
    )
    note = _require_string(claim["note"], f"unsupported claim {index} note")
    return {"claim": claim["claim"], "reason": reason, "note": note}


def _forbidden_abstraction_hits(text: str) -> list[str]:
    """Casefolded forbidden strategy terms present as standalone tokens."""
    lowered = text.casefold()
    return [
        term
        for term, pattern in zip(
            FORBIDDEN_ABSTRACTION_TERMS, _FORBIDDEN_ABSTRACTION_PATTERNS,
        )
        if pattern.search(lowered)
    ]


def _licensed_abstraction_terms(*, licensed_texts: Iterable[str]) -> set[str]:
    """Forbidden terms proven to be literal source language by exact text."""
    licensed_lowered = "\n".join(
        _require_string(text, "licensed abstraction text").casefold()
        for text in licensed_texts
    )
    return {
        term
        for term, pattern in zip(
            FORBIDDEN_ABSTRACTION_TERMS, _FORBIDDEN_ABSTRACTION_PATTERNS,
        )
        if pattern.search(licensed_lowered)
    }


def _validate_no_new_forbidden_abstractions(
    text: str,
    *,
    licensed_texts: Iterable[str],
    label: str,
) -> None:
    """Reject only newly introduced strategy abstractions.

    A forbidden term may remain when the exact word already appears in
    Bronze or in an exact quoted evidence span, because it is then literal
    source language rather than an invented final ontology abstraction.
    """
    licensed = _licensed_abstraction_terms(licensed_texts=licensed_texts)
    for term in _forbidden_abstraction_hits(text):
        if term not in licensed:
            raise ValueError(
                f"{label} introduces forbidden strategy abstraction "
                f"{term!r} not licensed by exact source evidence",
            )


def _licensed_entity_tokens(
    *,
    metadata: Mapping[str, Any],
    bindings: list[dict[str, Any]],
    licensed_texts: Iterable[str],
) -> set[str]:
    """Casefolded lexical tokens proven by Bronze/evidence/metadata/bindings.

    Exact Bronze and sealed evidence text license every word token they
    contain (case-insensitively); metadata values and resolved/alternative
    binding candidates license their word tokens too.  A candidate name
    therefore passes when its lexical token appears anywhere in this
    licensed set.
    """
    licensed: set[str] = set()
    for text in licensed_texts:
        licensed.update(token.casefold() for token in _entity_tokens(text))
    for value in _metadata_values(metadata):
        licensed.update(token.casefold() for token in _entity_tokens(value))
    for binding in bindings:
        licensed.update(
            token.casefold()
            for token in _entity_tokens(binding["resolved_candidate"])
        )
        for alternative in binding["alternatives"]:
            licensed.update(
                token.casefold()
                for token in _entity_tokens(alternative["candidate"])
            )
        for span in binding["evidence_spans"]:
            licensed.update(
                token.casefold() for token in _entity_tokens(span["text"])
            )
    return licensed


def _validate_entities_licensed(
    *,
    paraphrase_text: str,
    claims: list[dict[str, Any]],
    metadata: Mapping[str, Any],
    bindings: list[dict[str, Any]],
    licensed_texts: Iterable[str],
) -> None:
    """Reject only genuinely new capitalized named entities.

    A capitalized word is licensed case-insensitively when its lexical
    token appears in the exact Bronze/evidence texts or in
    metadata/resolved-binding evidence.  Contractions are never entity
    tokens, surrounding punctuation is stripped before checking, and
    ordinary sentence-initial capitalization of words already present in
    the licensed text is not treated as a new entity.  A genuinely new
    capitalized named entity absent from every license source still fails
    closed.
    """
    licensed = _licensed_entity_tokens(
        metadata=metadata,
        bindings=bindings,
        licensed_texts=licensed_texts,
    )
    for label, text in (
        ("paraphrase", paraphrase_text),
        *[(f"claim {item['claim_id']}", item["claim"]) for item in claims],
    ):
        for token in _entity_tokens(text):
            if not token[:1].isupper():
                continue
            if not re.search(r"[a-z]", token):
                continue
            if token in _COMMON_CAPITALIZED_WORDS:
                continue
            if token.casefold() not in licensed:
                raise ValueError(
                    f"resolved semantic output introduces unlicensed named "
                    f"entity {token!r} in {label}",
                )


def _validate_contextual_repairs(
    value: object,
    *,
    bronze_text: str,
    base_offset: int,
    transcript: str,
    context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate the complete, non-overlapping contextual repair list."""
    repairs = _require_list(value, "contextual_repairs")
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for repair in repairs:
        if not isinstance(repair, Mapping):
            raise ValueError("contextual repair must be an object")
        _require_exact_keys(
            repair,
            (
                "repair_id", "target_local_start", "target_local_end",
                "source_absolute_start", "source_absolute_end",
                "original_text", "replacement", "repair_type", "confidence",
                "evidence_spans", "rationale",
            ),
            "contextual repair",
        )
        repair_id = _require_nonempty_string(
            repair["repair_id"], "contextual repair id",
        )
        if repair_id in seen_ids:
            raise ValueError("contextual repair IDs must be unique")
        seen_ids.add(repair_id)
        repair_type = _require_enum(
            repair["repair_type"], CONTEXTUAL_REPAIR_TYPES, "contextual repair type",
        )
        confidence = _require_enum(
            repair["confidence"], REPAIR_CONFIDENCE_LEVELS,
            "contextual repair confidence",
        )
        local_start = _require_int(
            repair["target_local_start"], "contextual repair local start",
            minimum=0,
        )
        local_end = _require_int(
            repair["target_local_end"], "contextual repair local end",
            minimum=0,
        )
        if not 0 <= local_start < local_end <= len(bronze_text):
            raise ValueError("contextual repair local offsets are invalid")
        original = _require_string(
            repair["original_text"], "contextual repair original_text",
        )
        replacement = _require_string(
            repair["replacement"], "contextual repair replacement",
        )
        if bronze_text[local_start:local_end] != original:
            raise ValueError(
                "contextual repair original_text is not an exact Bronze slice",
            )
        if replacement == original:
            raise ValueError("contextual repair replacement must differ")
        if repair_type == "WHITESPACE" and not _differs_only_in_unicode_whitespace(
            original, replacement,
        ):
            raise ValueError(
                "contextual WHITESPACE repair must differ bytewise only "
                "in Unicode whitespace",
            )
        source_start = _require_int(
            repair["source_absolute_start"],
            "contextual repair source-absolute start",
            minimum=0,
        )
        source_end = _require_int(
            repair["source_absolute_end"],
            "contextual repair source-absolute end",
            minimum=0,
        )
        if source_start != base_offset + local_start:
            raise ValueError("contextual repair source-absolute start is inconsistent")
        if source_end != base_offset + local_end:
            raise ValueError("contextual repair source-absolute end is inconsistent")
        evidence = _validate_evidence_list(
            repair["evidence_spans"],
            transcript=transcript,
            context=context,
            label="contextual repair evidence",
        )
        if not evidence:
            raise ValueError(
                "contextual repair requires at least one exact-context "
                "evidence span",
            )
        rationale = _require_string(
            repair["rationale"], "contextual repair rationale",
        )
        validated.append({
            "repair_id": repair_id,
            "target_local_start": local_start,
            "target_local_end": local_end,
            "source_absolute_start": source_start,
            "source_absolute_end": source_end,
            "original_text": original,
            "replacement": replacement,
            "repair_type": repair_type,
            "confidence": confidence,
            "evidence_spans": evidence,
            "rationale": rationale,
        })
    validated.sort(key=lambda item: (
        item["target_local_start"], item["target_local_end"],
    ))
    for left_index in range(len(validated)):
        for right_index in range(left_index + 1, len(validated)):
            left = validated[left_index]
            right = validated[right_index]
            if left["target_local_end"] > right["target_local_start"]:
                raise ValueError("contextual repairs must not overlap")
    return validated


def _normalized_token_sequence(text: str) -> list[str]:
    """Deterministic surface-normalized word-token sequence for a phrase.

    Uses the exact surface normalization (case, punctuation, Unicode-space
    variants, and whitespace runs ignored) and splits into ordered tokens.
    This is the token-level comparison basis for the narrow exact-span
    composite entity license; it never permits lexical, spelling, or
    champion-name substitution.
    """
    return _surface_normalize(text).split()


def _exact_span_composite_license(
    repair: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> bool:
    """Narrow exact-span composite ENTITY_RESOLUTION license.

    Licenses a repair whose binding mention span exactly equals the repair
    span with exact Bronze text, whose resolved candidate appears as a
    complete normalized token sequence inside the replacement, and whose
    replacement differs from the original only by replacing one contiguous
    token sequence with that candidate while every surrounding normalized
    token remains identical in order.  This licenses exact-span forms such
    as ``this darus -> this Darius`` with candidate ``Darius`` while still
    rejecting broad rewrites, substring-inside-token candidate matches,
    wrong candidates, changed surrounding words, and non-exact mention
    spans.  The caller separately requires the binding to be RESOLVED in an
    allowed entity slot with evidence/metadata.
    """
    mention = binding["mention"]
    if not (
        mention["target_local_start"] == repair["target_local_start"]
        and mention["target_local_end"] == repair["target_local_end"]
        and mention["text"] == repair["original_text"]
    ):
        return False
    original_tokens = _normalized_token_sequence(repair["original_text"])
    replacement_tokens = _normalized_token_sequence(repair["replacement"])
    candidate_tokens = _normalized_token_sequence(
        binding["resolved_candidate"],
    )
    if not candidate_tokens:
        return False
    positions: list[int] = []
    for position in range(
        len(replacement_tokens) - len(candidate_tokens) + 1,
    ):
        if (
            replacement_tokens[
                position:position + len(candidate_tokens)
            ]
            == candidate_tokens
        ):
            positions.append(position)
    if not positions:
        return False
    for position in positions:
        if replacement_tokens[:position] != original_tokens[:position]:
            continue
        candidate_end = position + len(candidate_tokens)
        for replaced_length in range(1, len(original_tokens) - position + 1):
            if (
                replacement_tokens[candidate_end:]
                == original_tokens[position + replaced_length:]
            ):
                return True
    return False


def _validate_resolution_repairs_licensed(
    repairs: list[dict[str, Any]],
    bindings: list[dict[str, Any]],
) -> None:
    """Every semantic-resolution repair must be licensed by a RESOLVED binding
    over the exact same mention span; unresolved mentions are never
    rewritten.  For ENTITY_RESOLUTION only, a RESOLVED binding in an allowed
    entity slot may additionally license repeated identical repair
    occurrences when the binding mention source surface and the repair
    original_text are equal under strict surface normalization and the
    binding resolved candidate and the repair replacement are equal under
    strict surface normalization (one entity binding can license repeated
    canonicalization repairs).  A narrow exact-span composite license
    additionally applies when the binding mention span exactly equals the
    repair span with exact Bronze text, the resolved candidate appears as
    a complete normalized token sequence in the replacement, and the
    replacement differs from the original only by replacing one contiguous
    token sequence with that candidate while all surrounding normalized
    tokens remain identical in order.  Pronoun, reference, and
    ability-ownership repairs stay exact-span licensed because repeated
    pronouns/mentions can have different referents.
    """
    for repair in repairs:
        if repair["repair_type"] not in CONTEXTUAL_RESOLUTION_REPAIR_TYPES:
            continue
        allowed_slots = _CONTEXTUAL_REPAIR_BINDING_SLOTS[repair["repair_type"]]
        if repair["repair_type"] == "ENTITY_RESOLUTION":
            licensed = any(
                binding["resolved_status"] == "RESOLVED"
                and binding["slot"] in allowed_slots
                and binding["resolved_candidate"] != "NONE"
                and (binding["metadata_contributed"] or binding["evidence_spans"])
                and _surface_normalize(binding["mention"]["text"])
                == _surface_normalize(repair["original_text"])
                and _surface_normalize(binding["resolved_candidate"])
                == _surface_normalize(repair["replacement"])
                for binding in bindings
            )
            if not licensed:
                licensed = any(
                    binding["resolved_status"] == "RESOLVED"
                    and binding["slot"] in allowed_slots
                    and binding["resolved_candidate"] != "NONE"
                    and (
                        binding["metadata_contributed"]
                        or binding["evidence_spans"]
                    )
                    and _exact_span_composite_license(repair, binding)
                    for binding in bindings
                )
        else:
            matching = [
                binding
                for binding in bindings
                if binding["mention"]["target_local_start"]
                == repair["target_local_start"]
                and binding["mention"]["target_local_end"]
                == repair["target_local_end"]
                and binding["mention"]["text"] == repair["original_text"]
            ]
            if not matching:
                raise ValueError(
                    f"contextual {repair['repair_type']} repair "
                    f"{repair['repair_id']} has no binding over the same "
                    "mention span",
                )
            licensed = False
            for binding in matching:
                if binding["resolved_status"] != "RESOLVED":
                    continue
                if binding["slot"] not in allowed_slots:
                    continue
                candidate = binding["resolved_candidate"]
                if candidate == "NONE":
                    continue
                if binding["metadata_contributed"] or binding["evidence_spans"]:
                    if candidate.casefold() in repair["replacement"].casefold():
                        licensed = True
                        break
        if not licensed:
            raise ValueError(
                f"contextual {repair['repair_type']} repair {repair['repair_id']} "
                "is not licensed by a RESOLVED binding with candidate/status "
                "and evidence over the same mention span"
                + (
                    ": full-surface/repeated canonicalization requires the "
                    "binding mention and candidate to equal the repair "
                    "original and replacement under surface normalization; "
                    "exact-span composite requires the binding mention span "
                    "to exactly equal the repair span with exact Bronze "
                    "text, the resolved candidate to appear as a complete "
                    "normalized token sequence in the replacement, and the "
                    "replacement to differ from the original only by "
                    "replacing one contiguous token sequence with that "
                    "candidate"
                    if repair["repair_type"] == "ENTITY_RESOLUTION"
                    else ""
                ),
            )


def _validate_reconstruction_provenance(
    provenance: object,
    *,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ValueError("reconstruction provenance must be an object")
    _require_exact_keys(
        provenance,
        (
            "task_kind", "target", "prompt_version", "schema_version",
            "pipeline_version", "config_version", "input_metadata",
            "rationale",
        ),
        "reconstruction provenance",
    )
    if provenance["task_kind"] != CONTEXTUAL_RECONSTRUCTION_TASK_KIND:
        raise ValueError(
            "reconstruction provenance task_kind must be "
            "CONTEXTUAL_RECONSTRUCTION",
        )
    if provenance["prompt_version"] != RECONSTRUCTION_PROMPT_VERSION:
        raise ValueError("reconstruction provenance prompt_version is invalid")
    if provenance["schema_version"] != RECONSTRUCTION_RESPONSE_SCHEMA_VERSION:
        raise ValueError("reconstruction provenance schema_version is invalid")
    if provenance["pipeline_version"] != PIPELINE_VERSION:
        raise ValueError("reconstruction provenance pipeline_version is invalid")
    if provenance["config_version"] != CONFIG_VERSION:
        raise ValueError("reconstruction provenance config_version is invalid")
    if provenance["input_metadata"] != build_metadata_adapter(selected):
        raise ValueError(
            "reconstruction provenance input_metadata must exactly echo the "
            "supplied metadata adapter",
        )
    _validate_metadata_adapter(
        provenance["input_metadata"],
        selected=selected,
        label="reconstruction provenance",
    )
    target = provenance["target"]
    if not isinstance(target, Mapping):
        raise ValueError("reconstruction provenance target must be an object")
    _require_exact_keys(
        target,
        (
            "window_id", "source_group_id", "canonical_record_sha256",
            "upstream_start", "upstream_end", "upstream_content_sha256",
            "bronze_text", "bronze_text_sha256",
        ),
        "reconstruction provenance target",
    )
    if target != _target_identity(selected):
        raise ValueError(
            "reconstruction provenance target must exactly match the "
            "supplied window/source/hash identity",
        )
    rationale = _require_nonempty_string(
        provenance["rationale"], "reconstruction provenance rationale",
    )
    return dict(provenance)


def _validate_unresolved_alternatives(
    value: object,
    *,
    bronze_text: str,
    base_offset: int,
    transcript: str,
    context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    items = _require_list(value, "reconstruction unresolved_alternatives")
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items, 1):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"reconstruction unresolved alternative {index} must be an object",
            )
        _require_exact_keys(
            item,
            (
                "unresolved_id", "slot", "mention", "alternatives",
                "evidence_spans", "note",
            ),
            f"reconstruction unresolved alternative {index}",
        )
        unresolved_id = _require_nonempty_string(
            item["unresolved_id"], f"unresolved alternative {index} id",
        )
        if unresolved_id in seen_ids:
            raise ValueError("reconstruction unresolved alternative IDs must be unique")
        seen_ids.add(unresolved_id)
        slot = _require_enum(
            item["slot"], SLOT_KEYS, f"unresolved alternative {index} slot",
        )
        mention = _validate_bronze_span(
            item["mention"],
            bronze_text=bronze_text,
            base_offset=base_offset,
            label=f"unresolved alternative {index} mention",
        )
        alternatives = _require_list(
            item["alternatives"],
            f"unresolved alternative {index} alternatives",
        )
        if not alternatives:
            raise ValueError(
                "unresolved alternative requires at least one candidate",
            )
        validated_alternatives = []
        for alternative in alternatives:
            if not isinstance(alternative, Mapping):
                raise ValueError(
                    f"unresolved alternative {index} candidate must be an object",
                )
            _require_exact_keys(
                alternative,
                ("candidate", "confidence", "evidence_spans"),
                f"unresolved alternative {index} candidate",
            )
            validated_alternatives.append({
                "candidate": _require_nonempty_string(
                    alternative["candidate"],
                    f"unresolved alternative {index} candidate text",
                ),
                "confidence": _require_enum(
                    alternative["confidence"],
                    CONFIDENCE_LEVELS,
                    f"unresolved alternative {index} candidate confidence",
                ),
                "evidence_spans": _validate_evidence_list(
                    alternative["evidence_spans"],
                    transcript=transcript,
                    context=context,
                    label=f"unresolved alternative {index} candidate evidence",
                ),
            })
        evidence = _validate_evidence_list(
            item["evidence_spans"],
            transcript=transcript,
            context=context,
            label=f"unresolved alternative {index} evidence",
        )
        note = _require_string(
            item["note"], f"unresolved alternative {index} note",
        )
        validated.append({
            "unresolved_id": unresolved_id,
            "slot": slot,
            "mention": mention,
            "alternatives": validated_alternatives,
            "evidence_spans": evidence,
            "note": note,
        })
    return validated


def _validate_contextual_repair_proposal(
    value: object,
    *,
    index: int,
) -> dict[str, Any]:
    """Validate one compact contextual repair proposal (judgment only)."""
    if not isinstance(value, Mapping):
        raise ValueError("contextual repair proposal must be an object")
    _require_exact_keys(
        value,
        (
            "original_text", "replacement", "repair_type", "confidence",
            "evidence_quotes", "rationale",
        ),
        f"contextual repair proposal {index}",
    )
    repair_type = _require_enum(
        value["repair_type"],
        CONTEXTUAL_REPAIR_TYPES,
        f"contextual repair proposal {index} repair_type",
    )
    original_text = _require_string(
        value["original_text"],
        f"contextual repair proposal {index} original_text",
    )
    replacement = _require_string(
        value["replacement"],
        f"contextual repair proposal {index} replacement",
    )
    if not original_text:
        raise ValueError(
            f"contextual repair proposal {index} original_text must be non-empty",
        )
    if not replacement:
        if repair_type != "FILLER":
            raise ValueError(
                f"contextual repair proposal {index} replacement must be "
                "non-empty",
            )
    if not original_text.strip() and repair_type != "WHITESPACE":
        raise ValueError(
            f"contextual repair proposal {index} original_text must be "
            "non-empty (whitespace-only original_text is only valid for "
            "WHITESPACE repairs)",
        )
    if replacement == original_text and repair_type != "WHITESPACE":
        raise ValueError(
            f"contextual repair proposal {index} replacement must differ "
            "from original_text",
        )
    confidence = _require_enum(
        value["confidence"],
        REPAIR_CONFIDENCE_LEVELS,
        f"contextual repair proposal {index} confidence",
    )
    rationale = _require_string(
        value["rationale"], f"contextual repair proposal {index} rationale",
    )
    return {
        "original_text": original_text,
        "replacement": replacement,
        "repair_type": repair_type,
        "confidence": confidence,
        "evidence_quotes": value["evidence_quotes"],
        "rationale": rationale,
    }


def _whitespace_only_spans(text: str, length: int) -> list[tuple[int, int]]:
    """Exact contiguous spans whose slices are whitespace-only of ``length``."""
    if length <= 0:
        return []
    spans: list[tuple[int, int]] = []
    position = 0
    while True:
        found = -1
        for index in range(position, len(text) - length + 1):
            if text[index].isspace() and all(
                text[offset].isspace() for offset in range(index, index + length)
            ):
                found = index
                break
        if found == -1:
            break
        spans.append((found, found + length))
        position = found + length
    return spans


def _valid_repair_candidate(
    proposal: Mapping[str, Any],
    *,
    bronze_text: str,
    start: int,
    end: int,
) -> bool:
    """A candidate span is valid when the sealed replacement is a real change.

    WHITESPACE candidates must differ bytewise only in Unicode whitespace;
    every other candidate must differ from the exact Bronze slice so a true
    no-op (for example ``Eyeball`` -> ``eyeball`` when Bronze already holds
    ``eyeball``) is never silently accepted.
    """
    original = bronze_text[start:end]
    replacement = proposal["replacement"]
    if original == replacement:
        return False
    if proposal["repair_type"] == "WHITESPACE":
        return _differs_only_in_unicode_whitespace(original, replacement)
    return True


def _repair_candidate_spans(
    proposal: Mapping[str, Any],
    *,
    bronze_text: str,
) -> list[tuple[int, int]]:
    """Deterministic ordered candidate spans for one repair proposal.

    Candidate spans derive only from the three explicit rules: exact Bronze
    occurrences first, then strict surface-normalized occurrences, then the
    explicit contextual WHITESPACE matchers (Unicode-whitespace-only quotes
    bind left-to-right to whitespace-only Bronze slices of the same length;
    the existing whitespace-skeleton matcher handles regular-space forms
    that cannot reproduce exact Bronze Unicode whitespace).  True no-ops and
    non-whitespace differences are filtered out so they can never be
    selected.
    """
    original = proposal["original_text"]
    if (
        proposal["repair_type"] == "WHITESPACE"
        and original
        and all(char.isspace() for char in original)
    ):
        return [
            (start, end)
            for start, end in _whitespace_only_spans(bronze_text, len(original))
            if _valid_repair_candidate(
                proposal, bronze_text=bronze_text, start=start, end=end,
            )
        ]
    exact = [
        (start, start + len(original))
        for start in _occurrences(bronze_text, original)
    ]
    if exact:
        return [
            (start, end)
            for start, end in exact
            if _valid_repair_candidate(
                proposal, bronze_text=bronze_text, start=start, end=end,
            )
        ]
    surface = [
        (start, end)
        for start, end in _normalized_occurrence_spans(bronze_text, original)
        if _valid_repair_candidate(
            proposal, bronze_text=bronze_text, start=start, end=end,
        )
    ]
    if surface:
        return surface
    if proposal["repair_type"] == "WHITESPACE":
        return [
            (start, end)
            for start, end in _whitespace_skeleton_spans(bronze_text, original)
            if _valid_repair_candidate(
                proposal, bronze_text=bronze_text, start=start, end=end,
            )
        ]
    return []


def _repair_group_key(proposal: Mapping[str, Any]) -> tuple[str, str, str]:
    """Identical proposals bind to distinct occurrences in response order."""
    return (
        proposal["original_text"],
        proposal["replacement"],
        proposal["repair_type"],
    )


def _find_unique_repair_assignment(
    proposals: list[Mapping[str, Any]],
    candidate_lists: list[list[tuple[int, int]]],
    *,
    bronze_text: str,
    clean_text: str,
) -> list[tuple[int, tuple[int, int]]]:
    """Bounded deterministic global candidate assignment for repair spans.

    Searches for non-overlapping assignments of every proposal (each bound to
    one exact candidate span; repeated identical proposals bind left-to-right
    to distinct occurrences) whose ordered replacement application exactly
    equals the requested clean transcript.  Requires a unique valid
    assignment; ambiguous assignments and assignments with no valid solution
    fail closed.  No proposal may be silently dropped: providers must remove
    redundant proposals explicitly or merge overlapping edits into one exact
    Bronze span in a correction response.
    """
    group_positions: dict[tuple[str, str, str], list[int]] = {}
    for index, proposal in enumerate(proposals):
        group_positions.setdefault(_repair_group_key(proposal), []).append(index)

    def overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
        return left[0] < right[1] and right[0] < left[1]

    valid: list[list[tuple[int, tuple[int, int]]]] = []
    explored = 0
    EXPLORE_CAP = 200_000

    def apply_selection(selection: list[tuple[int, tuple[int, int]]]) -> str:
        ordered = sorted(selection, key=lambda item: item[1])
        pieces: list[str] = []
        cursor = 0
        for _index, (start, end) in ordered:
            pieces.append(bronze_text[cursor:start])
            pieces.append(proposals[_index]["replacement"])
            cursor = end
        pieces.append(bronze_text[cursor:])
        return "".join(pieces)

    def recurse(
        position: int,
        selected: list[tuple[int, tuple[int, int]]],
        used_spans: list[tuple[int, int]],
        last_in_group: dict[tuple[str, str, str], tuple[int, int] | None],
    ) -> None:
        nonlocal explored
        if len(valid) >= 2:
            return
        explored += 1
        if explored > EXPLORE_CAP:
            raise ValueError(
                "contextual repair candidate assignment exceeded the bounded "
                "deterministic search limit; fail closed",
            )
        if position == len(proposals):
            applied = apply_selection(selected)
            if applied == clean_text:
                valid.append(list(selected))
            return
        proposal = proposals[position]
        key = _repair_group_key(proposal)
        for span in candidate_lists[position]:
            if any(overlap(span, used) for used in used_spans):
                continue
            previous = last_in_group.get(key)
            if previous is not None and not (
                previous[0] < span[0] and previous[1] <= span[0]
            ):
                continue
            next_last = dict(last_in_group)
            next_last[key] = span
            recurse(
                position + 1,
                [*selected, (position, span)],
                [*used_spans, span],
                next_last,
            )

    recurse(0, [], [], {})
    if clean_text == bronze_text and proposals:
        raise ValueError(
            "contextual repair list must be empty when the clean transcript "
            "equals Bronze",
        )
    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        raise ValueError(
            "contextual repair candidate assignment is ambiguous: multiple "
            "non-overlapping span selections reproduce clean_target_transcript",
        )
    overlaps = [
        (left, right)
        for left_index in range(len(proposals))
        for right_index in range(left_index + 1, len(proposals))
        for left in candidate_lists[left_index]
        for right in candidate_lists[right_index]
        if overlap(left, right)
    ]
    if overlaps:
        raise ValueError(
            "contextual repairs must not overlap: merge overlapping edits "
            "into one exact Bronze span (include every change inside one "
            "contiguous original_text/replacement pair) and return the "
            "complete non-overlapping repair list; do not drop or duplicate "
            "repairs",
        )
    partial = _partial_repair_application_diff(
        proposals,
        candidate_lists,
        bronze_text=bronze_text,
        clean_text=clean_text,
    )
    raise ValueError(
        "no non-overlapping selection of the contextual repair spans "
        "reproduces clean_target_transcript. " + partial,
    )


def _partial_repair_application_diff(
    proposals: list[Mapping[str, Any]],
    candidate_lists: list[list[tuple[int, int]]],
    *,
    bronze_text: str,
    clean_text: str,
) -> str:
    """Best-effort deterministic partial assignment for actionable diff feedback."""
    selection: list[tuple[int, tuple[int, int]]] = []
    used: list[tuple[int, int]] = []
    last_in_group: dict[tuple[str, str, str], tuple[int, int] | None] = {}
    for index, (proposal, candidates) in enumerate(zip(proposals, candidate_lists)):
        key = _repair_group_key(proposal)
        for span in candidates:
            if any(
                span[0] < used[1] and used[0] < span[1]
                for used in used
            ):
                continue
            previous = last_in_group.get(key)
            if previous is not None and not (
                previous[0] < span[0] and previous[1] <= span[0]
            ):
                continue
            selection.append((index, span))
            used.append(span)
            last_in_group[key] = span
            break
    ordered = sorted(selection, key=lambda item: item[1])
    pieces: list[str] = []
    cursor = 0
    for index, (start, end) in ordered:
        pieces.append(bronze_text[cursor:start])
        pieces.append(proposals[index]["replacement"])
        cursor = end
    pieces.append(bronze_text[cursor:])
    applied = "".join(pieces)
    return _clean_application_diff_message(
        applied=applied,
        requested=clean_text,
        label="contextual repair list cannot reproduce the clean transcript",
    )


def _bind_contextual_repair_proposals(
    proposals: list[Mapping[str, Any]],
    *,
    bronze_text: str,
    selected: Mapping[str, Any],
    clean_text: str | None = None,
) -> list[dict[str, Any]]:
    """Deterministically bind ordered repair proposals to exact Bronze spans.

    Exact binding remains the rule.  A bounded deterministic global
    candidate assignment augments the old greedy first-occurrence binding:
    candidate spans derive only from exact Bronze occurrences, strict
    surface-normalized occurrences, and the explicit contextual WHITESPACE
    rules (including whitespace-only proposals such as NBSP -> regular
    space).  The unique non-overlapping selection whose ordered replacement
    application equals the requested clean transcript is sealed with exact
    Bronze slices and source offsets; ambiguous or invalid assignments fail
    closed.  True no-ops and non-whitespace differences are never accepted.
    """
    if clean_text is None:
        raise ValueError(
            "contextual repair binding requires the requested clean transcript",
        )
    base_offset = int(selected["upstream_start"])
    candidate_lists: list[list[tuple[int, int]]] = []
    for index, proposal in enumerate(proposals):
        candidates = _repair_candidate_spans(
            proposal, bronze_text=bronze_text,
        )
        if not candidates:
            raise ValueError(
                _unbindable_repair_error(
                    index + 1,
                    proposal,
                    bronze_text=bronze_text,
                    clean_text=clean_text,
                ),
            )
        candidate_lists.append(candidates)
    selected_assignment = _find_unique_repair_assignment(
        proposals,
        candidate_lists,
        bronze_text=bronze_text,
        clean_text=clean_text,
    )
    selected_spans = dict(selected_assignment)
    bound: list[dict[str, Any]] = []
    for index, proposal in enumerate(proposals):
        if index not in selected_spans:
            continue
        local_start, local_end = selected_spans[index]
        occurrence = candidate_lists[index].index((local_start, local_end))
        bound.append({
            **proposal,
            "target_local_start": local_start,
            "target_local_end": local_end,
            "source_absolute_start": base_offset + local_start,
            "source_absolute_end": base_offset + local_end,
            "proposal_index": index,
            "occurrence_index": occurrence,
        })
    bound.sort(key=lambda item: (
        item["target_local_start"], item["target_local_end"],
    ))
    for position, repair in enumerate(bound, 1):
        repair["repair_id"] = f"p2k:ctx:r{position:04d}"
    return bound


def _unbindable_repair_error(
    index: int,
    proposal: Mapping[str, Any],
    *,
    bronze_text: str,
    clean_text: str,
) -> str:
    """Actionable fail-closed error for a repair with no valid Bronze span."""
    original = proposal["original_text"]
    message = (
        f"contextual repair proposal {index} original_text {original!r} cannot "
        "be bound to any exact, surface-normalized, or explicit-whitespace "
        "Bronze span"
    )
    if (
        proposal["repair_type"] == "WHITESPACE"
        and proposal["replacement"] == original
    ):
        message += (
            " and would be a true no-op (the exact quote is already present "
            "in Bronze or the mapped slice changes nothing)"
        )
    elif original.casefold() in clean_text.casefold():
        message += (
            "; if Bronze and clean_target_transcript already match at this "
            "location, remove this entire repair rather than repeating it"
        )
    else:
        message += (
            "; either quote an exact contiguous Bronze span or remove this "
            "entire repair"
        )
    return message


def _validate_binding_proposal(
    value: object,
    *,
    index: int,
) -> dict[str, Any]:
    """Validate one compact binding proposal (judgment fields only)."""
    if not isinstance(value, Mapping):
        raise ValueError(f"binding proposal {index} must be an object")
    _require_exact_keys(
        value,
        (
            "slot", "mention_text", "resolved_candidate", "resolved_status",
            "confidence", "evidence_quotes", "alternatives",
            "metadata_contributed", "rationale",
        ),
        f"binding proposal {index}",
    )
    slot = _require_enum(
        value["slot"], SLOT_KEYS, f"binding proposal {index} slot",
    )
    mention_text = _require_nonempty_string(
        value["mention_text"], f"binding proposal {index} mention_text",
    )
    resolved_candidate = _require_nonempty_string(
        value["resolved_candidate"],
        f"binding proposal {index} resolved_candidate",
    )
    resolved_status = _require_enum(
        value["resolved_status"],
        BINDING_STATUSES,
        f"binding proposal {index} resolved_status",
    )
    confidence = _require_enum(
        value["confidence"],
        CONFIDENCE_LEVELS,
        f"binding proposal {index} confidence",
    )
    alternatives = _require_list(
        value["alternatives"], f"binding proposal {index} alternatives",
    )
    validated_alternatives = []
    for alternative in alternatives:
        if not isinstance(alternative, Mapping):
            raise ValueError(f"binding proposal {index} alternative must be an object")
        _require_exact_keys(
            alternative,
            ("candidate", "evidence_quotes", "note"),
            f"binding proposal {index} alternative",
        )
        validated_alternatives.append({
            "candidate": _require_nonempty_string(
                alternative["candidate"],
                f"binding proposal {index} alternative candidate",
            ),
            "evidence_quotes": alternative["evidence_quotes"],
            "note": _require_string(
                alternative["note"],
                f"binding proposal {index} alternative note",
            ),
        })
    metadata_contributed = _require_bool(
        value["metadata_contributed"],
        f"binding proposal {index} metadata_contributed",
    )
    rationale = _require_string(
        value["rationale"], f"binding proposal {index} rationale",
    )
    return {
        "slot": slot,
        "mention_text": mention_text,
        "resolved_candidate": resolved_candidate,
        "resolved_status": resolved_status,
        "confidence": confidence,
        "evidence_quotes": value["evidence_quotes"],
        "alternatives": validated_alternatives,
        "metadata_contributed": metadata_contributed,
        "rationale": rationale,
    }


def _validate_unresolved_alternative_proposal(
    value: object,
    *,
    index: int,
) -> dict[str, Any]:
    """Validate one compact unresolved-alternative proposal."""
    if not isinstance(value, Mapping):
        raise ValueError(
            f"unresolved alternative proposal {index} must be an object",
        )
    _require_exact_keys(
        value,
        ("slot", "mention_text", "alternatives", "evidence_quotes", "note"),
        f"unresolved alternative proposal {index}",
    )
    slot = _require_enum(
        value["slot"], SLOT_KEYS, f"unresolved alternative proposal {index} slot",
    )
    mention_text = _require_nonempty_string(
        value["mention_text"],
        f"unresolved alternative proposal {index} mention_text",
    )
    alternatives = _require_list(
        value["alternatives"],
        f"unresolved alternative proposal {index} alternatives",
    )
    if not alternatives:
        raise ValueError(
            f"unresolved alternative proposal {index} requires at least "
            "one candidate",
        )
    validated_alternatives = []
    for alternative in alternatives:
        if not isinstance(alternative, Mapping):
            raise ValueError(
                f"unresolved alternative proposal {index} candidate "
                "must be an object",
            )
        _require_exact_keys(
            alternative,
            ("candidate", "confidence", "evidence_quotes"),
            f"unresolved alternative proposal {index} candidate",
        )
        validated_alternatives.append({
            "candidate": _require_nonempty_string(
                alternative["candidate"],
                f"unresolved alternative proposal {index} candidate text",
            ),
            "confidence": _require_enum(
                alternative["confidence"],
                CONFIDENCE_LEVELS,
                f"unresolved alternative proposal {index} candidate confidence",
            ),
            "evidence_quotes": alternative["evidence_quotes"],
        })
    note = _require_string(
        value["note"], f"unresolved alternative proposal {index} note",
    )
    return {
        "slot": slot,
        "mention_text": mention_text,
        "alternatives": validated_alternatives,
        "evidence_quotes": value["evidence_quotes"],
        "note": note,
    }


def _bind_bronze_mentions(
    proposals: list[Mapping[str, Any]],
    *,
    bronze_text: str,
    base_offset: int,
    label: str,
    group_key: Any = None,
) -> list[dict[str, Any]]:
    """Deterministically bind ordered mention_text quotes to Bronze spans.

    Exact binding remains first choice.  When an exact surface is absent or
    its exact occurrence count cannot be matched, a deterministic
    surface-normalized fallback ignores only capitalization, punctuation,
    Unicode-space variants, and whitespace runs.  The fallback must be
    unique per occurrence: zero or multiple normalized matches fail closed,
    and lexical/spelling changes never match.  The final artifact stores the
    exact Bronze slice and exact offsets; ``raw_compact`` preserves the
    provider's mention_text.

    Proposals are grouped by semantic assertion: the k-th proposal inside
    one group binds to the k-th occurrence, while different groups (for
    example different slots/assertions) may share the same occurrence.
    Supplying fewer group proposals than the total source occurrence count
    is valid; supplying more proposals than occurrences inside one group
    fails closed.
    """
    if group_key is None:
        def group_key(proposal: Mapping[str, Any]) -> tuple[str, str, str, str]:
            return (
                proposal["mention_text"],
                proposal["slot"],
                proposal["resolved_candidate"],
                proposal["resolved_status"],
            )
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for proposal in proposals:
        groups.setdefault(group_key(proposal), []).append(proposal)
    group_counts: dict[str, list[int]] = {}
    for key, items in groups.items():
        group_counts.setdefault(key[0], []).append(len(items))
    surfaces = [proposal["mention_text"] for proposal in proposals]
    strategies = _surface_strategies(
        surfaces,
        sources=[bronze_text],
        label=label,
        group_counts=group_counts,
        absent_guidance=(
            "Remove this entire binding rather than repeating it: a "
            "source-absent mention carries no real mention and is never "
            "normalized."
            if label.startswith("binding")
            else (
                "Remove this entire unresolved alternative rather than "
                "repeating it, or quote an exact Bronze mention."
            )
        ),
        aggregate_absent=label.startswith("binding"),
    )
    bound: list[dict[str, Any]] = []
    counters: dict[Any, int] = {}
    for index, proposal in enumerate(proposals):
        mention_text = proposal["mention_text"]
        key = group_key(proposal)
        occurrence = counters.get(key, 0)
        counters[key] = occurrence + 1
        spans = _surface_spans(
            bronze_text,
            mention_text,
            normalized=strategies[mention_text],
        )
        if occurrence >= len(spans):
            raise ValueError(
                f"{label} quote {mention_text!r} cannot be bound "
                "deterministically: a single semantic assertion group "
                f"supplied {occurrence + 1} time(s) but only {len(spans)} "
                "matching source occurrence(s) exist",
            )
        local_start, local_end = spans[occurrence]
        bound.append({
            **proposal,
            "target_local_start": local_start,
            "target_local_end": local_end,
            "source_absolute_start": base_offset + local_start,
            "source_absolute_end": base_offset + local_end,
            "text": bronze_text[local_start:local_end],
            "proposal_index": index,
            "occurrence_index": occurrence,
        })
    return bound


def _unresolved_group_key(proposal: Mapping[str, Any]) -> tuple[str, str]:
    """Semantic grouping key for unresolved-alternative proposals."""
    return (proposal["mention_text"], proposal["slot"])


def _reconstruction_evidence_texts(
    repairs: list[dict[str, Any]],
    bindings: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
) -> list[str]:
    """All exact evidence quote texts for narrow abstraction licensing."""
    texts: list[str] = []
    for repair in repairs:
        texts.extend(span["text"] for span in repair["evidence_spans"])
    for binding in bindings:
        texts.extend(span["text"] for span in binding["evidence_spans"])
        for alternative in binding["alternatives"]:
            texts.extend(span["text"] for span in alternative["evidence_spans"])
    for item in unresolved:
        texts.extend(span["text"] for span in item["evidence_spans"])
        for alternative in item["alternatives"]:
            texts.extend(span["text"] for span in alternative["evidence_spans"])
    return texts


def _is_sentinel_binding_omitted(proposal: Mapping[str, Any]) -> bool:
    """Narrowly documented sentinel-omission rule for binding proposals.

    A proposal with explicit ``NONE`` as both mention and candidate is a
    "no binding" placeholder (the slot has no source mention), never a
    source mention.  A proposal whose ``resolved_candidate`` is ``NONE``
    carries no actual resolution claim (the provider is only echoing that a
    slot/candidate is not applicable), so it is also omitted from the
    normalized semantic bindings.  Both classes remain preserved verbatim in
    ``raw_compact`` for auditability.  Other source-absent mentions (for
    example ``your queue``) are NOT sentinels and still fail validation
    with explicit removal guidance.
    """
    mention = proposal.get("mention_text")
    candidate = proposal.get("resolved_candidate")
    if mention == "NONE" and candidate == "NONE":
        return True
    if candidate == "NONE":
        return True
    return False


def _is_context_only_binding_proposal(
    proposal: Mapping[str, Any],
    *,
    bronze_text: str,
    context: Mapping[str, Any],
) -> bool:
    """Narrow deterministic context-only binding-omission rule.

    A binding proposal whose ``mention_text`` has zero exact or
    surface-normalized matches in the target Bronze but at least one exact
    or surface-normalized match inside the supplied ordered context refers
    only to the surrounding context, never to a target-Bronze mention.
    Target bindings must refer to target Bronze mentions, so such a
    proposal is conservatively omitted from the normalized target bindings
    while its verbatim original remains in ``raw_compact`` and the omission
    is counted in ``omitted_binding_count``.

    The classifier never applies to a proposal with any target match: such
    proposals keep the normal deterministic binding path and still fail
    closed on count/ambiguity errors exactly as before (an ambiguous or
    multiply-normalized target mention is never reclassified as
    context-only).  Proposals absent from both the target and the supplied
    context are not context-only and keep failing with the existing
    remove-entire-binding guidance.
    """
    mention_text = proposal["mention_text"]
    target_exact_count = len(_occurrences(bronze_text, mention_text))
    target_normalized_count = len(
        _normalized_occurrence_spans(bronze_text, mention_text),
    )
    if target_exact_count or target_normalized_count:
        return False
    segment_texts = [
        _require_string(segment["text"], "context segment text")
        for segment in _require_list(
            context["segments"], "phase2k context segments",
        )
    ]
    context_exact_count = sum(
        len(_occurrences(segment_text, mention_text))
        for segment_text in segment_texts
    )
    context_normalized_count = sum(
        len(_normalized_occurrence_spans(segment_text, mention_text))
        for segment_text in segment_texts
    )
    return context_exact_count > 0 or context_normalized_count > 0


def _normalize_reconstruction_response(
    parsed: Mapping[str, Any],
    *,
    bronze_text: str,
    base_offset: int,
    transcript: str,
    context: Mapping[str, Any],
    final_diagnostic: Mapping[str, Any],
    metadata: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the compact v4 envelope, seal spans/IDs/provenance.

    The provider supplies only exact quotes and judgment fields.  Every
    Bronze mention/original and context evidence quote is bound by
    deterministic ordered-occurrence rules, IDs and source offsets are
    sealed by the harness, and the complete non-overlapping repair list
    must reproduce clean_target_transcript exactly.  The strict normalized
    validator is then applied downstream.
    """
    _require_exact_keys(
        parsed,
        (
            "schema_version", "clean_target_transcript", "contextual_repairs",
            "bindings", "unresolved_alternatives", "rationale",
        ),
        "phase2k reconstruction response",
    )
    if parsed["schema_version"] != RECONSTRUCTION_RESPONSE_SCHEMA_VERSION:
        raise ValueError("phase2k reconstruction response schema version is invalid")
    rationale = _require_nonempty_string(
        parsed["rationale"], "reconstruction rationale",
    )
    clean = _require_string(
        parsed["clean_target_transcript"], "clean_target_transcript",
    )
    repair_proposals = [
        _validate_contextual_repair_proposal(proposal, index=index)
        for index, proposal in enumerate(
            _require_list(
                parsed["contextual_repairs"], "contextual_repairs",
            ),
            1,
        )
    ]
    bound_repairs = _bind_contextual_repair_proposals(
        repair_proposals,
        bronze_text=bronze_text,
        selected=selected,
        clean_text=clean,
    )
    validated_repairs = []
    for repair in bound_repairs:
        evidence = _bind_evidence_quotes(
            repair["evidence_quotes"],
            context=context,
            label=(
                f"contextual repair proposal "
                f"{repair['proposal_index'] + 1}"
            ),
            allow_normalized=True,
            anchor_span=(
                repair["source_absolute_start"],
                repair["source_absolute_end"],
            ),
        )
        if not evidence:
            raise ValueError(
                "contextual repair requires at least one exact-context "
                "evidence quote",
            )
        validated_repairs.append({
            "repair_id": repair["repair_id"],
            "target_local_start": repair["target_local_start"],
            "target_local_end": repair["target_local_end"],
            "source_absolute_start": repair["source_absolute_start"],
            "source_absolute_end": repair["source_absolute_end"],
            "original_text": bronze_text[
                repair["target_local_start"]:repair["target_local_end"]
            ],
            "replacement": repair["replacement"],
            "repair_type": repair["repair_type"],
            "confidence": repair["confidence"],
            "evidence_spans": evidence,
            "rationale": repair["rationale"],
        })
    binding_proposals = [
        _validate_binding_proposal(proposal, index=index)
        for index, proposal in enumerate(
            _require_list(parsed["bindings"], "bindings"),
            1,
        )
    ]
    explicit_sentinels = [
        proposal
        for proposal in binding_proposals
        if proposal["mention_text"] == "NONE"
        and proposal["resolved_candidate"] == "NONE"
    ]
    real_binding_proposals = [
        proposal
        for proposal in binding_proposals
        if not (
            proposal["mention_text"] == "NONE"
            and proposal["resolved_candidate"] == "NONE"
        )
    ]
    context_only_bindings: list[dict[str, Any]] = []
    target_binding_proposals: list[dict[str, Any]] = []
    for proposal in real_binding_proposals:
        if _is_context_only_binding_proposal(
            proposal,
            bronze_text=bronze_text,
            context=context,
        ):
            context_only_bindings.append(proposal)
        else:
            target_binding_proposals.append(proposal)
    bound_bindings = _bind_bronze_mentions(
        target_binding_proposals,
        bronze_text=bronze_text,
        base_offset=base_offset,
        label="binding",
    )
    final_slots = final_diagnostic["response"]["parsed"]["slots"]
    validated_bindings = []
    omitted_binding_count = len(explicit_sentinels) + len(
        context_only_bindings,
    )
    kept_bindings = []
    for proposal in bound_bindings:
        if _is_sentinel_binding_omitted(proposal):
            omitted_binding_count += 1
            continue
        kept_bindings.append(proposal)
    for position, proposal in enumerate(kept_bindings, 1):
        evidence = _bind_evidence_quotes(
            proposal["evidence_quotes"],
            context=context,
            label=f"binding proposal {proposal['proposal_index']}",
            allow_normalized=True,
        )
        alternatives = []
        for alternative in proposal["alternatives"]:
            alternatives.append({
                "candidate": alternative["candidate"],
                "evidence_spans": _bind_evidence_quotes(
                    alternative["evidence_quotes"],
                    context=context,
                    label=(
                        f"binding proposal {proposal['proposal_index']} "
                        "alternative"
                    ),
                    allow_normalized=True,
                ),
                "note": alternative["note"],
            })
        normalized = {
            "binding_id": f"p2k:ctx:b{position:04d}",
            "slot": proposal["slot"],
            "mention": {
                "target_local_start": proposal["target_local_start"],
                "target_local_end": proposal["target_local_end"],
                "source_absolute_start": proposal["source_absolute_start"],
                "source_absolute_end": proposal["source_absolute_end"],
                "text": proposal["text"],
            },
            "resolved_candidate": proposal["resolved_candidate"],
            "resolved_status": proposal["resolved_status"],
            "confidence": proposal["confidence"],
            "evidence_spans": evidence,
            "alternatives": alternatives,
            "metadata_contributed": proposal["metadata_contributed"],
            "rationale": proposal["rationale"],
        }
        validated_bindings.append(_validate_binding(
            normalized,
            index=position,
            bronze_text=bronze_text,
            base_offset=base_offset,
            transcript=transcript,
            context=context,
            final_slots=final_slots,
            metadata=metadata,
        ))
    unresolved_proposals = [
        _validate_unresolved_alternative_proposal(proposal, index=index)
        for index, proposal in enumerate(
            _require_list(
                parsed["unresolved_alternatives"], "unresolved_alternatives",
            ),
            1,
        )
    ]
    bound_unresolved = _bind_bronze_mentions(
        unresolved_proposals,
        bronze_text=bronze_text,
        base_offset=base_offset,
        label="unresolved alternative",
        group_key=_unresolved_group_key,
    )
    normalized_unresolved = []
    for position, proposal in enumerate(bound_unresolved, 1):
        alternatives = []
        for alternative in proposal["alternatives"]:
            alternatives.append({
                "candidate": alternative["candidate"],
                "confidence": alternative["confidence"],
                "evidence_spans": _bind_evidence_quotes(
                    alternative["evidence_quotes"],
                    context=context,
                    label=(
                        f"unresolved alternative proposal "
                        f"{proposal['proposal_index']} candidate"
                    ),
                    allow_normalized=True,
                ),
            })
        normalized_unresolved.append({
            "unresolved_id": f"p2k:ctx:u{position:04d}",
            "slot": proposal["slot"],
            "mention": {
                "target_local_start": proposal["target_local_start"],
                "target_local_end": proposal["target_local_end"],
                "source_absolute_start": proposal["source_absolute_start"],
                "source_absolute_end": proposal["source_absolute_end"],
                "text": proposal["text"],
            },
            "alternatives": alternatives,
            "evidence_spans": _bind_evidence_quotes(
                proposal["evidence_quotes"],
                context=context,
                label=(
                    f"unresolved alternative proposal "
                    f"{proposal['proposal_index']}"
                ),
                allow_normalized=True,
            ),
            "note": proposal["note"],
        })
    provenance = _seal_reconstruction_provenance(
        selected,
        rationale=rationale,
    )
    normalized = {
        "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
        "clean_target_transcript": clean,
        "contextual_repairs": validated_repairs,
        "bindings": validated_bindings,
        "unresolved_alternatives": normalized_unresolved,
        "provenance": provenance,
    }
    validated = validate_reconstruction_response(
        normalized,
        bronze_text=bronze_text,
        base_offset=base_offset,
        transcript=transcript,
        context=context,
        final_diagnostic=final_diagnostic,
        metadata=metadata,
        selected=selected,
    )
    return {
        **validated,
        "rationale": rationale,
        "raw_compact": parsed,
        "omitted_binding_count": omitted_binding_count,
    }


def validate_reconstruction_response(
    parsed: Mapping[str, Any],
    *,
    bronze_text: str,
    base_offset: int,
    transcript: str,
    context: Mapping[str, Any],
    final_diagnostic: Mapping[str, Any],
    metadata: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    """Strict normalized reconstruction validator (exact sealed spans)."""
    _require_exact_keys(
        parsed,
        (
            "schema_version", "clean_target_transcript", "contextual_repairs",
            "bindings", "unresolved_alternatives", "provenance",
        ),
        "phase2k reconstruction response",
    )
    if parsed["schema_version"] != RECONSTRUCTION_RESPONSE_SCHEMA_VERSION:
        raise ValueError("phase2k reconstruction response schema version is invalid")
    clean = _require_string(
        parsed["clean_target_transcript"], "clean_target_transcript",
    )
    contextual_repairs = _validate_contextual_repairs(
        parsed["contextual_repairs"],
        bronze_text=bronze_text,
        base_offset=base_offset,
        transcript=transcript,
        context=context,
    )
    expected_clean = apply_mechanical_repairs(bronze_text, contextual_repairs)
    if clean != expected_clean:
        raise ValueError(
            "reconstruction clean transcript must equal the deterministic "
            "application of the complete contextual repair list to Bronze. "
            + _clean_application_diff_message(
                applied=expected_clean,
                requested=clean,
                label="reconstruction clean transcript mismatch",
            ),
        )
    if clean == bronze_text and contextual_repairs:
        raise ValueError(
            "contextual repair list must be empty when the clean transcript "
            "equals Bronze",
        )
    bindings = _require_list(parsed["bindings"], "bindings")
    final_slots = final_diagnostic["response"]["parsed"]["slots"]
    validated_bindings = [
        _validate_binding(
            binding,
            index=index,
            bronze_text=bronze_text,
            base_offset=base_offset,
            transcript=transcript,
            context=context,
            final_slots=final_slots,
            metadata=metadata,
        )
        for index, binding in enumerate(bindings, 1)
    ]
    _validate_resolution_repairs_licensed(
        contextual_repairs, validated_bindings,
    )
    unresolved = _validate_unresolved_alternatives(
        parsed["unresolved_alternatives"],
        bronze_text=bronze_text,
        base_offset=base_offset,
        transcript=transcript,
        context=context,
    )
    provenance = _validate_reconstruction_provenance(
        parsed["provenance"],
        selected=selected,
    )
    evidence_texts = _reconstruction_evidence_texts(
        contextual_repairs, validated_bindings, unresolved,
    )
    _validate_no_new_forbidden_abstractions(
        clean,
        licensed_texts=[bronze_text, *evidence_texts],
        label="reconstruction clean transcript",
    )
    for repair in contextual_repairs:
        _validate_no_new_forbidden_abstractions(
            repair["rationale"],
            licensed_texts=[
                bronze_text,
                *[span["text"] for span in repair["evidence_spans"]],
            ],
            label=f"reconstruction repair {repair['repair_id']} rationale",
        )
    _validate_no_new_forbidden_abstractions(
        provenance["rationale"],
        licensed_texts=[bronze_text, *evidence_texts],
        label="reconstruction provenance rationale",
    )
    _validate_entities_licensed(
        paraphrase_text=clean,
        claims=[],
        metadata=metadata,
        bindings=validated_bindings,
        licensed_texts=[bronze_text, *evidence_texts],
    )
    return {
        "clean_target_transcript": clean,
        "contextual_repairs": contextual_repairs,
        "bindings": validated_bindings,
        "unresolved_alternatives": unresolved,
        "provenance": provenance,
    }


def run_reconstruction(
    selected: Mapping[str, Any],
    *,
    transcript: str,
    context: Mapping[str, Any],
    mechanical_cleaned_text: str,
    final_diagnostic: Mapping[str, Any],
    chat: ChatCallable,
    config_hash: str,
    cache_dir: Path | None = None,
    inference_config: Mapping[str, Any] | None = None,
    lineage: Mapping[str, Any] | None = None,
    raw_response_dir: Path | None = None,
) -> dict[str, Any]:
    """Run compact reconstruction with bounded strict corrections."""
    validate_context(context, transcript)
    bronze_text = selected["source_text"]
    base_offset = selected["upstream_start"]
    metadata = build_metadata_adapter(selected)
    system, user = build_reconstruction_prompt(
        selected,
        context,
        mechanical_cleaned_text,
        final_diagnostic,
    )

    def validator(parsed: Mapping[str, Any]) -> dict[str, Any]:
        return _normalize_reconstruction_response(
            parsed,
            bronze_text=bronze_text,
            base_offset=base_offset,
            transcript=transcript,
            context=context,
            final_diagnostic=final_diagnostic,
            metadata=metadata,
            selected=selected,
        )

    outcome = call_provider_with_corrections(
        chat,
        system=system,
        user=user,
        build_correction_prompt=lambda prior_raw, error, **kwargs: (
            build_reconstruction_correction_prompt(
                selected,
                context,
                mechanical_cleaned_text,
                final_diagnostic,
                prior_raw=prior_raw,
                error=error,
                **kwargs,
            )
        ),
        validator=validator,
        label="phase2k reconstruction",
        schema_version=RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
        prompt_version=RECONSTRUCTION_PROMPT_VERSION,
        correction_prompt_version=RECONSTRUCTION_CORRECTION_PROMPT_VERSION,
        config_hash=config_hash,
        window_id=selected["window_id"],
        stage="reconstruction",
        cache_dir=cache_dir,
        inference_config=inference_config,
        lineage=lineage,
        raw_response_dir=raw_response_dir,
        max_corrections=RECONSTRUCTION_MAX_CORRECTIONS,
    )
    validated = outcome["result"]
    final_call = outcome["final_attempt"]["model_call"]
    bindings = validated["bindings"]
    unresolved_bindings = sum(
        1 for binding in bindings if binding["resolved_status"] != "RESOLVED"
    )
    resolution_repair_count = sum(
        1
        for repair in validated["contextual_repairs"]
        if repair["repair_type"] in CONTEXTUAL_RESOLUTION_REPAIR_TYPES
    )
    counts = {
        "binding_count": len(bindings),
        "unresolved_binding_count": unresolved_bindings,
        "contextual_repair_count": len(validated["contextual_repairs"]),
        "resolution_repair_count": resolution_repair_count,
        "unresolved_alternative_count": len(
            validated["unresolved_alternatives"],
        ),
        "metadata_conflict_count": len(
            final_diagnostic["response"]["parsed"]["metadata_conflicts"],
        ),
    }
    return {
        "generation_status": "GENERATED",
        "clean_target_transcript": validated["clean_target_transcript"],
        "clean_target_transcript_sha256": text_sha256(
            validated["clean_target_transcript"],
        ),
        "contextual_repairs": validated["contextual_repairs"],
        "bindings": bindings,
        "unresolved_alternatives": validated["unresolved_alternatives"],
        "provenance": validated["provenance"],
        "rationale": validated["rationale"],
        "raw_compact": validated["raw_compact"],
        "omitted_binding_count": validated["omitted_binding_count"],
        "failure": None,
        "counts": counts,
        "model_call": {
            "source": final_call["source"],
            "prompt_hash": final_call["prompt_hash"],
            "cache_key": final_call["cache_key"],
            "config_hash": final_call["config_hash"],
            "inference_config": final_call["inference_config"],
            "inference_config_hash": final_call["inference_config_hash"],
            "inference_config_version": final_call["inference_config_version"],
            "prompt_version": final_call["prompt_version"],
            "schema_version": final_call["schema_version"],
            "attempt_index": final_call["attempt_index"],
            "attempt_kind": final_call["attempt_kind"],
            "raw_response_sha256": final_call["raw_response_sha256"],
            "raw_response_path": final_call["raw_response_path"],
            "status": final_call["status"],
            "error": final_call["error"],
        },
        "attempts": outcome["attempts"],
        "pipeline_version": PIPELINE_VERSION,
        "config_version": CONFIG_VERSION,
    }


def _polish_response_template() -> dict[str, Any]:
    """Exact full compact provider template for semantic polish."""
    return {
        "schema_version": POLISH_RESPONSE_SCHEMA_VERSION,
        "statements": [
            {
                "text": "statement text",
                "modality_preserved": True,
                "negation_preserved": True,
                "uncertainty_preserved": True,
                "evidence_quotes": ["exact Bronze quote"],
                "reconstruction_operation_ids": [],
                "support_mode": "UNCHANGED_EXACT",
                "unchanged_source_quote": "exact Bronze quote or null",
            },
        ],
        "unsupported_claims": [
            {
                "claim": "unsupported claim text",
                "reason": "MODEL_INVENTION",
                "note": "short note",
            },
        ],
        "rationale": "short rationale",
    }


def build_polish_prompt(
    selected: Mapping[str, Any],
    context: Mapping[str, Any],
    mechanical_cleaned_text: str,
    reconstruction: Mapping[str, Any],
) -> tuple[str, str]:
    """Prompt for the separate, post-reconstruction semantic-polish pass."""
    metadata = build_metadata_adapter(selected)
    system = (
        "You run the Phase 2K semantic-polish pass, strictly after the "
        "contextual reconstruction pass has been validated and sealed.  "
        "Consume only the exact Bronze target, the mechanical cleaned text, "
        "the terminal ordered context, and the validated reconstruction.  "
        "Produce compiler-friendly resolved semantic statements only.  You "
        "supply judgment fields only: the harness deterministically binds "
        "every evidence/unchanged quote to an exact Bronze span and seals "
        "every statement ID.  Never emit statement IDs or Bronze offsets.\n"
        "Each statement has exactly: \"text\", \"modality_preserved\", "
        "\"negation_preserved\", \"uncertainty_preserved\", "
        "\"evidence_quotes\", \"reconstruction_operation_ids\", "
        "\"support_mode\", \"unchanged_source_quote\".  The preservation "
        "fields are booleans attesting modality/negation/uncertainty "
        "preservation.  evidence_quotes are exact contiguous Bronze quotes "
        "(never paraphrases); repeat a repeated quote once per intended "
        "occurrence or quote a longer unique span.  Every statement "
        "requires at least one evidence quote.\n"
        "support_mode is exactly one of:\n"
        "- UNCHANGED_EXACT: text exactly equals unchanged_source_quote "
        "(the exact Bronze slice).\n"
        "- RECONSTRUCTION_DERIVED: text is derived from at least one "
        "existing reconstruction operation id (repair/binding ids "
        "supplied in the reconstruction input).\n"
        "- EVIDENCE_PARAPHRASE: a source-grounded semantic paraphrase "
        "licensed by exact Bronze evidence; reconstruction operation ids "
        "are optional (may be empty) and are never required when no "
        "reconstruction repair/binding applies.\n"
        "Repaired text (text that differs from the exact Bronze source, for "
        "example after a capitalization or pronoun repair) must use "
        "RECONSTRUCTION_DERIVED: keep unchanged_source_quote null, "
        "reference the operation IDs that support the change, and quote "
        "the exact Bronze original text in evidence_quotes - never the "
        "repaired text.  Never relabel repaired text as UNCHANGED_EXACT; "
        "UNCHANGED_EXACT requires text byte-equal to its "
        "unchanged_source_quote.\n"
        "reconstruction_operation_ids must reference only existing ids "
        "from the supplied reconstruction; unknown ids fail.  "
        "unchanged_source_quote is an exact Bronze quote for "
        "UNCHANGED_EXACT and null otherwise.  Do not resolve any mention "
        "left unresolved by reconstruction.  Named entities and claims "
        "must remain licensed by Bronze/metadata/bindings; a human audit "
        "will judge preservation versus invention.  Literal source words "
        "that happen to match an ontology word (for example priority or "
        "pressure) are allowed when exact evidence proves they are source "
        "language, but never introduce a newly invented strategy "
        "abstraction.  Unsupported transformations go under "
        "\"unsupported_claims\" with a closed reason; never silently "
        "invent a clause.  Respond with one JSON object that exactly "
        "matches the supplied response_schema and nothing else."
    )
    user = json.dumps(
        {
            "task": "semantic_polish",
            "prompt_version": POLISH_PROMPT_VERSION,
            "schema_version": POLISH_RESPONSE_SCHEMA_VERSION,
            "response_schema": _polish_response_template(),
            "target": _target_identity(selected),
            "mechanical_cleaned_text": mechanical_cleaned_text,
            "context": {
                "radius": context["radius"],
                "actual": context["actual"],
                "segments": context["segments"],
            },
            "reconstruction": {
                "clean_target_transcript": reconstruction[
                    "clean_target_transcript"
                ],
                "contextual_repairs": reconstruction["contextual_repairs"],
                "bindings": reconstruction["bindings"],
                "unresolved_alternatives": reconstruction[
                    "unresolved_alternatives"
                ],
                "provenance": reconstruction["provenance"],
            },
            "metadata": metadata,
            "supplied_facts": supplied_facts(metadata),
            "metadata_adapter_schema_version": METADATA_ADAPTER_SCHEMA_VERSION,
            "metadata_policy": {
                "champion_role_are_supplied_inference_facts": True,
                "video_title_is_provenance_only": True,
                "no_title_based_matchup_inference": True,
                "missing_fields_stay_absent": True,
            },
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return system, user


def build_polish_correction_prompt(
    selected: Mapping[str, Any],
    context: Mapping[str, Any],
    mechanical_cleaned_text: str,
    reconstruction: Mapping[str, Any],
    *,
    prior_raw: str,
    error: Exception,
) -> tuple[str, str]:
    """Strict semantic-polish correction prompt for one failed attempt."""
    system = (
        "You are correcting a Phase 2K semantic-polish response that failed "
        "strict validation.  Return a COMPLETE corrected JSON object that "
        "exactly matches the supplied response_schema and nothing else.  "
        "Supply judgment fields only (no statement IDs or offsets), keep "
        "support_mode invariants exact, reference only existing "
        "reconstruction operation ids, quote exact Bronze evidence, and "
        "never invent ontology abstractions or resolve mentions left "
        "unresolved by reconstruction.\n"
        "This is a bounded correction pass: one initial attempt plus at "
        "most three corrections.  Fix EVERY validator-reported issue in "
        "this single response, including a malformed prior JSON: do not "
        "assume another correction will follow.\n"
        "Repaired text (text that differs from the exact Bronze source) "
        "must use support_mode RECONSTRUCTION_DERIVED with "
        "unchanged_source_quote null, reference the operation IDs that "
        "support the change, and quote the exact Bronze original text in "
        "evidence_quotes - never the repaired text.  When validator_error "
        "reports an UNCHANGED_EXACT text mismatch it includes the exact "
        "actual text and the exact unchanged_source_quote: either (a) "
        "safest - set statement text byte-exactly equal to that "
        "unchanged_source_quote (copy the exact quoted value verbatim), or "
        "(b) switch the statement to support_mode RECONSTRUCTION_DERIVED "
        "with unchanged_source_quote null and at least one valid "
        "reconstruction_operation_id that supports the change; never "
        "relabel repaired text as UNCHANGED_EXACT.  When validator_error "
        "suggests an exact replacement evidence_quote, use that suggestion "
        "verbatim.  UNCHANGED_EXACT requires text byte-equal to its "
        "unchanged_source_quote.  Do not explain; output only the complete "
        "corrected JSON object."
    )
    user = json.dumps(
        {
            "task": "semantic_polish_correction",
            "correction_prompt_version": POLISH_CORRECTION_PROMPT_VERSION,
            "schema_version": POLISH_RESPONSE_SCHEMA_VERSION,
            "response_schema": _polish_response_template(),
            "target": _target_identity(selected),
            "mechanical_cleaned_text": mechanical_cleaned_text,
            "context": {
                "radius": context["radius"],
                "actual": context["actual"],
                "segments": context["segments"],
            },
            "reconstruction": {
                "clean_target_transcript": reconstruction[
                    "clean_target_transcript"
                ],
                "contextual_repairs": reconstruction["contextual_repairs"],
                "bindings": reconstruction["bindings"],
                "unresolved_alternatives": reconstruction[
                    "unresolved_alternatives"
                ],
                "provenance": reconstruction["provenance"],
            },
            "metadata": build_metadata_adapter(selected),
            "supplied_facts": supplied_facts(build_metadata_adapter(selected)),
            "prior_raw_response": prior_raw,
            "validator_error": f"{type(error).__name__}: {error}",
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return system, user


def _reconstruction_operation_ids(reconstruction: Mapping[str, Any]) -> set[str]:
    return {
        repair["repair_id"] for repair in reconstruction["contextual_repairs"]
    } | {
        binding["binding_id"] for binding in reconstruction["bindings"]
    }


def _bind_bronze_quotes(
    quotes: object,
    *,
    bronze_text: str,
    base_offset: int,
    label: str,
) -> list[dict[str, Any]]:
    """Deterministically bind compact Bronze evidence quotes to exact spans.

    The model never counts offsets.  Each quote must be an exact substring
    of Bronze, and a repeated surface must be quoted once per intended
    occurrence (equal supplied and occurrence counts) so the intended
    occurrence is never guessed.
    """
    items = _require_list(quotes, f"{label} evidence_quotes")
    if not items:
        return []
    occurrence_counts: dict[str, int] = {}
    for item in items:
        quote = _require_nonempty_string(item, f"{label} evidence quote")
        occurrence_counts[quote] = occurrence_counts.get(quote, 0) + 1
    for quote, supplied in occurrence_counts.items():
        total = len(_occurrences(bronze_text, quote))
        if total == 0:
            message = f"{label} evidence quote {quote!r} is absent from Bronze"
            normalized_spans = _normalized_occurrence_spans(bronze_text, quote)
            distinct_slices = {
                bronze_text[start:end] for start, end in normalized_spans
            }
            if len(distinct_slices) == 1:
                suggested = next(iter(distinct_slices))
                message += (
                    f". Suggested exact replacement evidence_quote: "
                    f"{suggested!r} (exact Bronze slice); quote it verbatim. "
                    "This is guidance only: the quoted evidence still fails "
                    "validation"
                )
            raise ValueError(
                message,
            )
        if supplied != total:
            raise ValueError(
                f"{label} evidence quote {quote!r} is ambiguous: it occurs "
                f"{total} time(s) in Bronze but was supplied "
                f"{supplied} time(s); repeat it once per intended "
                "occurrence or quote a longer unique span",
            )
    validated: list[dict[str, Any]] = []
    counters: dict[str, int] = {}
    for item in items:
        quote = _require_nonempty_string(item, f"{label} evidence quote")
        occurrence = counters.get(quote, 0)
        counters[quote] = occurrence + 1
        local_start = _occurrences(bronze_text, quote)[occurrence]
        local_end = local_start + len(quote)
        validated.append({
            "target_local_start": local_start,
            "target_local_end": local_end,
            "source_absolute_start": base_offset + local_start,
            "source_absolute_end": base_offset + local_end,
            "text": quote,
        })
    return validated


def _bind_single_bronze_quote(
    quote: object,
    *,
    bronze_text: str,
    base_offset: int,
    label: str,
) -> dict[str, Any]:
    """Bind one unchanged-source quote to its unique exact Bronze span."""
    quote = _require_nonempty_string(quote, label)
    starts = _occurrences(bronze_text, quote)
    if len(starts) != 1:
        raise ValueError(
            f"{label} quote {quote!r} is absent or ambiguous in Bronze",
        )
    return {
        "target_local_start": starts[0],
        "target_local_end": starts[0] + len(quote),
        "source_absolute_start": base_offset + starts[0],
        "source_absolute_end": base_offset + starts[0] + len(quote),
        "text": quote,
    }


def _validate_polish_statement(
    statement: object,
    *,
    index: int,
    bronze_text: str,
    base_offset: int,
    metadata: Mapping[str, Any],
    bindings: list[dict[str, Any]],
    operation_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(statement, Mapping):
        raise ValueError(f"polish statement {index} must be an object")
    _require_exact_keys(
        statement,
        (
            "statement_id", "text", "modality_preserved",
            "negation_preserved", "uncertainty_preserved", "evidence_spans",
            "reconstruction_operation_ids", "support_mode",
            "unchanged_source_quote",
        ),
        f"polish statement {index}",
    )
    statement_id = _require_nonempty_string(
        statement["statement_id"], f"polish statement {index} id",
    )
    text = _require_nonempty_string(
        statement["text"], f"polish statement {index} text",
    )
    for field in POLISH_STATEMENT_ATTESTATION_FIELDS:
        _require_bool(
            statement[field], f"polish statement {index} {field}",
        )
    support_mode = _require_enum(
        statement["support_mode"],
        POLISH_SUPPORT_MODES,
        f"polish statement {index} support_mode",
    )
    evidence = _require_list(
        statement["evidence_spans"], f"polish statement {index} evidence_spans",
    )
    validated_evidence = [
        _validate_bronze_span(
            span,
            bronze_text=bronze_text,
            base_offset=base_offset,
            label=f"polish statement {index} evidence span",
        )
        for span in evidence
    ]
    if not validated_evidence:
        raise ValueError("polish statement requires at least one evidence span")
    operation_ids_list = _require_list(
        statement["reconstruction_operation_ids"],
        f"polish statement {index} reconstruction_operation_ids",
    )
    validated_operation_ids = []
    for operation_id in operation_ids_list:
        operation_id = _require_nonempty_string(
            operation_id, f"polish statement {index} operation id",
        )
        if operation_id not in operation_ids:
            raise ValueError(
                f"polish statement {index} references an unknown reconstruction "
                f"operation id {operation_id!r}",
            )
        validated_operation_ids.append(operation_id)
    unchanged_quote = statement["unchanged_source_quote"]
    validated_quote: dict[str, Any] | None = None
    if unchanged_quote is not None:
        validated_quote = _validate_bronze_span(
            unchanged_quote,
            bronze_text=bronze_text,
            base_offset=base_offset,
            label=f"polish statement {index} unchanged source quote",
        )
    if support_mode == "UNCHANGED_EXACT":
        if validated_quote is None:
            raise ValueError(
                f"polish statement {index} UNCHANGED_EXACT requires an "
                "unchanged_source_quote",
            )
        if text != validated_quote["text"]:
            raise ValueError(
                f"polish statement {index} UNCHANGED_EXACT text mismatch: "
                f"actual text is {json.dumps(text)} but "
                f"unchanged_source_quote is "
                f"{json.dumps(validated_quote['text'])}; UNCHANGED_EXACT "
                "text must exactly equal its unchanged_source_quote - "
                "either (a) safest: set text byte-exactly equal to the "
                f"unchanged_source_quote "
                f"{json.dumps(validated_quote['text'])} (quote it verbatim), "
                "or (b) use support_mode RECONSTRUCTION_DERIVED with "
                "unchanged_source_quote null and at least one valid "
                "reconstruction_operation_id supporting the change",
            )
    elif support_mode == "RECONSTRUCTION_DERIVED":
        if validated_quote is not None:
            raise ValueError(
                f"polish statement {index} RECONSTRUCTION_DERIVED must not "
                "carry an unchanged_source_quote",
            )
        if not validated_operation_ids:
            raise ValueError(
                f"polish statement {index} RECONSTRUCTION_DERIVED requires "
                "at least one reconstruction operation id",
            )
    else:  # EVIDENCE_PARAPHRASE
        if validated_quote is not None:
            raise ValueError(
                f"polish statement {index} EVIDENCE_PARAPHRASE must not "
                "carry an unchanged_source_quote",
            )
    _validate_no_new_forbidden_abstractions(
        text,
        licensed_texts=[
            bronze_text,
            *[span["text"] for span in validated_evidence],
        ],
        label=f"polish statement {index}",
    )
    _validate_entities_licensed(
        paraphrase_text=text,
        claims=[],
        metadata=metadata,
        bindings=bindings,
        licensed_texts=[
            bronze_text,
            *[span["text"] for span in validated_evidence],
        ],
    )
    return {
        "statement_id": statement_id,
        "text": text,
        "modality_preserved": statement["modality_preserved"],
        "negation_preserved": statement["negation_preserved"],
        "uncertainty_preserved": statement["uncertainty_preserved"],
        "evidence_spans": validated_evidence,
        "reconstruction_operation_ids": validated_operation_ids,
        "support_mode": support_mode,
        "unchanged_source_quote": validated_quote,
    }


def _normalize_polish_response(
    parsed: Mapping[str, Any],
    *,
    bronze_text: str,
    base_offset: int,
    reconstruction: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the compact polish envelope and seal spans/statement IDs."""
    _require_exact_keys(
        parsed,
        ("schema_version", "statements", "unsupported_claims", "rationale"),
        "phase2k polish response",
    )
    if parsed["schema_version"] != POLISH_RESPONSE_SCHEMA_VERSION:
        raise ValueError("phase2k polish response schema version is invalid")
    statements = _require_list(parsed["statements"], "polish statements")
    normalized_statements = []
    for index, statement in enumerate(statements, 1):
        if not isinstance(statement, Mapping):
            raise ValueError(f"polish statement {index} must be an object")
        _require_exact_keys(
            statement,
            (
                "text", "modality_preserved", "negation_preserved",
                "uncertainty_preserved", "evidence_quotes",
                "reconstruction_operation_ids", "support_mode",
                "unchanged_source_quote",
            ),
            f"polish statement {index}",
        )
        support_mode = _require_enum(
            statement["support_mode"],
            POLISH_SUPPORT_MODES,
            f"polish statement {index} support_mode",
        )
        unchanged_quote = statement["unchanged_source_quote"]
        bound_quote = (
            _bind_single_bronze_quote(
                unchanged_quote,
                bronze_text=bronze_text,
                base_offset=base_offset,
                label=f"polish statement {index} unchanged_source_quote",
            )
            if unchanged_quote is not None
            else None
        )
        normalized_statements.append({
            "statement_id": f"p2k:stmt:s{index:04d}",
            "text": statement["text"],
            "modality_preserved": statement["modality_preserved"],
            "negation_preserved": statement["negation_preserved"],
            "uncertainty_preserved": statement["uncertainty_preserved"],
            "evidence_spans": _bind_bronze_quotes(
                statement["evidence_quotes"],
                bronze_text=bronze_text,
                base_offset=base_offset,
                label=f"polish statement {index}",
            ),
            "reconstruction_operation_ids": statement[
                "reconstruction_operation_ids"
            ],
            "support_mode": support_mode,
            "unchanged_source_quote": bound_quote,
        })
    normalized = {
        "schema_version": POLISH_RESPONSE_SCHEMA_VERSION,
        "statements": normalized_statements,
        "unsupported_claims": parsed["unsupported_claims"],
        "rationale": parsed["rationale"],
    }
    validated = validate_polish_response(
        normalized,
        bronze_text=bronze_text,
        base_offset=base_offset,
        transcript="",
        context={"segments": []},
        reconstruction=reconstruction,
        metadata=metadata,
    )
    return {
        **validated,
        "raw_compact": parsed,
    }


def validate_polish_response(
    parsed: Mapping[str, Any],
    *,
    bronze_text: str,
    base_offset: int,
    transcript: str,
    context: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Strict normalized polish validator (exact sealed spans/IDs)."""
    del transcript, context  # reserved for invariant consistency downstream
    _require_exact_keys(
        parsed,
        ("schema_version", "statements", "unsupported_claims", "rationale"),
        "phase2k polish response",
    )
    if parsed["schema_version"] != POLISH_RESPONSE_SCHEMA_VERSION:
        raise ValueError("phase2k polish response schema version is invalid")
    operation_ids = _reconstruction_operation_ids(reconstruction)
    statements = _require_list(parsed["statements"], "polish statements")
    validated_statements: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, statement in enumerate(statements, 1):
        validated = _validate_polish_statement(
            statement,
            index=index,
            bronze_text=bronze_text,
            base_offset=base_offset,
            metadata=metadata,
            bindings=reconstruction["bindings"],
            operation_ids=operation_ids,
        )
        if validated["statement_id"] in seen_ids:
            raise ValueError("polish statement IDs must be unique")
        seen_ids.add(validated["statement_id"])
        validated_statements.append(validated)
    unsupported = _require_list(
        parsed["unsupported_claims"], "polish unsupported_claims",
    )
    validated_unsupported = [
        _validate_unsupported_claim(claim, index=index)
        for index, claim in enumerate(unsupported, 1)
    ]
    rationale = _require_string(parsed["rationale"], "polish rationale")
    evidence_texts = [
        span["text"]
        for statement in validated_statements
        for span in statement["evidence_spans"]
    ]
    _validate_no_new_forbidden_abstractions(
        rationale,
        licensed_texts=[bronze_text, *evidence_texts],
        label="polish rationale",
    )
    polished_text = "\n".join(statement["text"] for statement in validated_statements)
    return {
        "statements": validated_statements,
        "unsupported_claims": validated_unsupported,
        "polished_text": polished_text,
        "rationale": rationale,
    }


def run_polish(
    selected: Mapping[str, Any],
    *,
    transcript: str,
    context: Mapping[str, Any],
    mechanical_cleaned_text: str,
    reconstruction: Mapping[str, Any],
    chat: ChatCallable,
    config_hash: str,
    cache_dir: Path | None = None,
    inference_config: Mapping[str, Any] | None = None,
    lineage: Mapping[str, Any] | None = None,
    raw_response_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the compact semantic-polish pass with bounded corrections."""
    validate_context(context, transcript)
    bronze_text = selected["source_text"]
    base_offset = selected["upstream_start"]
    metadata = build_metadata_adapter(selected)
    system, user = build_polish_prompt(
        selected,
        context,
        mechanical_cleaned_text,
        reconstruction,
    )

    def validator(parsed: Mapping[str, Any]) -> dict[str, Any]:
        return _normalize_polish_response(
            parsed,
            bronze_text=bronze_text,
            base_offset=base_offset,
            reconstruction=reconstruction,
            metadata=metadata,
        )

    outcome = call_provider_with_corrections(
        chat,
        system=system,
        user=user,
        build_correction_prompt=lambda prior_raw, error, **kwargs: (
            build_polish_correction_prompt(
                selected,
                context,
                mechanical_cleaned_text,
                reconstruction,
                prior_raw=prior_raw,
                error=error,
            )
        ),
        validator=validator,
        label="phase2k semantic polish",
        schema_version=POLISH_RESPONSE_SCHEMA_VERSION,
        prompt_version=POLISH_PROMPT_VERSION,
        correction_prompt_version=POLISH_CORRECTION_PROMPT_VERSION,
        config_hash=config_hash,
        window_id=selected["window_id"],
        stage="semantic_polish",
        cache_dir=cache_dir,
        inference_config=inference_config,
        lineage=lineage,
        raw_response_dir=raw_response_dir,
        max_corrections=POLISH_MAX_CORRECTIONS,
    )
    validated = outcome["result"]
    final_call = outcome["final_attempt"]["model_call"]
    statements = validated["statements"]
    unsupported = validated["unsupported_claims"]
    counts = {
        "statement_count": len(statements),
        "unsupported_claim_count": len(unsupported),
        "unsupported_rate": _safe_float(
            len(unsupported) / max(1, len(statements) + len(unsupported)),
        ),
    }
    return {
        "generation_status": "GENERATED",
        "statements": statements,
        "unsupported_claims": unsupported,
        "polished_text": validated["polished_text"],
        "rationale": validated["rationale"],
        "raw_compact": validated["raw_compact"],
        "counts": counts,
        "failure": None,
        "model_call": {
            "source": final_call["source"],
            "prompt_hash": final_call["prompt_hash"],
            "cache_key": final_call["cache_key"],
            "config_hash": final_call["config_hash"],
            "inference_config": final_call["inference_config"],
            "inference_config_hash": final_call["inference_config_hash"],
            "inference_config_version": final_call["inference_config_version"],
            "prompt_version": final_call["prompt_version"],
            "schema_version": final_call["schema_version"],
            "attempt_index": final_call["attempt_index"],
            "attempt_kind": final_call["attempt_kind"],
            "raw_response_sha256": final_call["raw_response_sha256"],
            "raw_response_path": final_call["raw_response_path"],
            "status": final_call["status"],
            "error": final_call["error"],
        },
        "attempts": outcome["attempts"],
        "pipeline_version": PIPELINE_VERSION,
        "config_version": CONFIG_VERSION,
    }


# ---------------------------------------------------------------------------
# A/B/C/D record construction
# ---------------------------------------------------------------------------


def _target_block(selected: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "window_id": selected["window_id"],
        "source_group_id": selected["source_group_id"],
        "source_absolute_start": selected["upstream_start"],
        "source_absolute_end": selected["upstream_end"],
        "text": selected["source_text"],
        "text_sha256": text_sha256(selected["source_text"]),
        "char_length": len(selected["source_text"]),
    }


def build_record_a(selected: Mapping[str, Any]) -> dict[str, Any]:
    record = {
        "record_id": f"p2k:rec:{selected['window_id']}:A",
        "record_type": "A",
        "window_id": selected["window_id"],
        "target": _target_block(selected),
        "content": {
            "kind": "raw_bronze",
            "text": selected["source_text"],
            "text_sha256": text_sha256(selected["source_text"]),
            "char_length": len(selected["source_text"]),
        },
    }
    record["canonical_record_sha256"] = canonical_sha256(record)
    return record


def build_record_b(
    selected: Mapping[str, Any],
    mechanical: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if mechanical is None:
        hints = detect_champion_alias_hints(
            selected["source_text"], selected,
        )
        content = {
            "kind": "mechanical_clean",
            "generation_status": "NOT_GENERATED",
            "clean_text": selected["source_text"],
            "text": selected["source_text"],
            "text_sha256": text_sha256(selected["source_text"]),
            "char_length": len(selected["source_text"]),
            "repairs": [],
            "repair_count": 0,
            "lexical_hints": hints,
            "lexical_hint_count": len(hints),
            "uncertainties": [],
            "uncertainty_count": 0,
            "provenance": None,
            "raw_proposals": None,
            "model_call": None,
            "attempts": [],
            "pipeline_version": PIPELINE_VERSION,
            "config_version": CONFIG_VERSION,
        }
    else:
        content = {
            "kind": "mechanical_clean",
            "generation_status": "GENERATED",
            "clean_text": mechanical["mechanical_cleaned_text"],
            "text": mechanical["mechanical_cleaned_text"],
            "text_sha256": mechanical["mechanical_cleaned_text_sha256"],
            "char_length": mechanical["mechanical_cleaned_char_length"],
            "repairs": mechanical["repairs"],
            "repair_count": mechanical["repair_count"],
            "lexical_hints": mechanical["lexical_hints"],
            "lexical_hint_count": mechanical["lexical_hint_count"],
            "uncertainties": mechanical["uncertainties"],
            "uncertainty_count": mechanical["uncertainty_count"],
            "provenance": mechanical["provenance"],
            "raw_proposals": mechanical["raw_proposals"],
            "model_call": mechanical["model_call"],
            "attempts": mechanical["attempts"],
            "pipeline_version": mechanical["pipeline_version"],
            "config_version": mechanical["config_version"],
        }
    record = {
        "record_id": f"p2k:rec:{selected['window_id']}:B",
        "record_type": "B",
        "window_id": selected["window_id"],
        "target": _target_block(selected),
        "content": content,
    }
    record["canonical_record_sha256"] = canonical_sha256(record)
    return record


def build_record_c(
    selected: Mapping[str, Any],
    context: Mapping[str, Any],
    mechanical_text: str | None = None,
) -> dict[str, Any]:
    presentation_text = mechanical_text if mechanical_text is not None else (
        selected["source_text"]
    )
    record = {
        "record_id": f"p2k:rec:{selected['window_id']}:C",
        "record_type": "C",
        "window_id": selected["window_id"],
        "target": _target_block(selected),
        "content": {
            "kind": "enlarged_context",
            "context": context,
            "presentation_target": {
                "text": presentation_text,
                "text_sha256": text_sha256(presentation_text),
                "is_mechanical": mechanical_text is not None,
                "bronze_target_sha256": text_sha256(selected["source_text"]),
            },
        },
    }
    record["canonical_record_sha256"] = canonical_sha256(record)
    return record


def _reconstruction_subobject(
    reconstruction: Mapping[str, Any],
) -> dict[str, Any]:
    """Sealed D reconstruction subobject with audit attempts/raw compact."""
    return {
        "generation_status": reconstruction["generation_status"],
        "clean_target_transcript": reconstruction["clean_target_transcript"],
        "clean_target_transcript_sha256": reconstruction[
            "clean_target_transcript_sha256"
        ],
        "contextual_repairs": reconstruction["contextual_repairs"],
        "bindings": reconstruction["bindings"],
        "unresolved_alternatives": reconstruction["unresolved_alternatives"],
        "provenance": reconstruction["provenance"],
        "rationale": reconstruction["rationale"],
        "raw_compact": reconstruction["raw_compact"],
        "omitted_binding_count": reconstruction["omitted_binding_count"],
        "counts": reconstruction["counts"],
        "model_call": reconstruction["model_call"],
        "attempts": reconstruction["attempts"],
        "pipeline_version": reconstruction["pipeline_version"],
        "config_version": reconstruction["config_version"],
    }


def _polish_subobject(polish: Mapping[str, Any]) -> dict[str, Any]:
    """Sealed D semantic-polish subobject with audit attempts/raw compact."""
    return {
        "generation_status": polish["generation_status"],
        "statements": polish["statements"],
        "unsupported_claims": polish["unsupported_claims"],
        "polished_text": polish["polished_text"],
        "rationale": polish["rationale"],
        "raw_compact": polish["raw_compact"],
        "counts": polish["counts"],
        "model_call": polish["model_call"],
        "attempts": polish["attempts"],
        "pipeline_version": polish["pipeline_version"],
        "config_version": polish["config_version"],
    }


def build_record_d(
    selected: Mapping[str, Any],
    mechanical: Mapping[str, Any] | None,
    reconstruction: Mapping[str, Any] | None,
    polish: Mapping[str, Any] | None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A GENERATED D requires both passes; placeholders keep stage+history."""
    zero_counts = {
        "binding_count": 0,
        "unresolved_binding_count": 0,
        "contextual_repair_count": 0,
        "resolution_repair_count": 0,
        "unresolved_alternative_count": 0,
        "metadata_conflict_count": 0,
        "statement_count": 0,
        "unsupported_claim_count": 0,
        "unsupported_rate": 0.0,
    }
    if reconstruction is not None and polish is not None:
        content = {
            "kind": "reconstruction",
            "generation_status": reconstruction["generation_status"],
            "clean_target_transcript": reconstruction["clean_target_transcript"],
            "clean_target_transcript_sha256": reconstruction[
                "clean_target_transcript_sha256"
            ],
            "contextual_repairs": reconstruction["contextual_repairs"],
            "bindings": reconstruction["bindings"],
            "unresolved_alternatives": reconstruction["unresolved_alternatives"],
            "is_placeholder": False,
            "reconstruction": _reconstruction_subobject(reconstruction),
            "semantic_polish": _polish_subobject(polish),
            "model_calls": {
                "reconstruction": reconstruction["model_call"],
                "semantic_polish": polish["model_call"],
            },
            "failure": None,
            "counts": {
                **reconstruction["counts"],
                **polish["counts"],
            },
            "model_call": reconstruction["model_call"],
            "pipeline_version": reconstruction["pipeline_version"],
            "config_version": reconstruction["config_version"],
        }
    else:
        cleaned = (
            mechanical["mechanical_cleaned_text"]
            if mechanical is not None
            else selected["source_text"]
        )
        if failure is None:
            failure = {
                "reason": "NO_PROVIDER_MODE",
                "stage": None,
                "note": (
                    "No provider ran; reconstruction and semantic polish are "
                    "deferred to a live run."
                ),
            }
        attempts = list(failure.get("attempts") or [])
        normalized_failure = {
            "reason": _require_nonempty_string(
                failure.get("reason", "WINDOW_GENERATION_FAILED"),
                "D failure reason",
            ),
            "stage": failure.get("stage"),
            "note": _require_string(failure.get("note", ""), "D failure note"),
            "attempt_count": len(attempts),
            "attempts": attempts,
        }
        content = {
            "kind": "reconstruction",
            "generation_status": "NOT_GENERATED",
            "clean_target_transcript": cleaned,
            "clean_target_transcript_sha256": text_sha256(cleaned),
            "contextual_repairs": [],
            "bindings": [],
            "unresolved_alternatives": [],
            "is_placeholder": True,
            "placeholder_note": (
                "Reconstruction/polish unavailable; this placeholder carries "
                "the exact B/mechanical clean text, not a generated D."
            ),
            "reconstruction": (
                _reconstruction_subobject(reconstruction)
                if reconstruction is not None
                else None
            ),
            "semantic_polish": None,
            "model_calls": (
                {"reconstruction": reconstruction["model_call"]}
                if reconstruction is not None
                else {}
            ),
            "failure": normalized_failure,
            "counts": dict(zero_counts),
            "model_call": (
                reconstruction["model_call"]
                if reconstruction is not None
                else None
            ),
            "pipeline_version": PIPELINE_VERSION,
            "config_version": CONFIG_VERSION,
        }
    record = {
        "record_id": f"p2k:rec:{selected['window_id']}:D",
        "record_type": "D",
        "window_id": selected["window_id"],
        "target": _target_block(selected),
        "content": content,
    }
    record["canonical_record_sha256"] = canonical_sha256(record)
    return record


def build_radius_entries(
    selected: Mapping[str, Any],
    transcript: str,
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entries = []
    radius_configs = (
        ("target_only", 0, 0, None),
        ("r2", 2, 2, "r1"),
        ("r5", 5, 5, "r2"),
        ("r10", 10, 10, "r3"),
        ("bounded_local_episode", 40, 40, "r4_bounded_local_episode"),
    )
    for radius_label, previous, following, stage_label in radius_configs:
        context = retrieve_context(
            transcript,
            source_group_id=selected["source_group_id"],
            window_id=selected["window_id"],
            target_start=selected["upstream_start"],
            target_end=selected["upstream_end"],
            bronze_text=selected["source_text"],
            previous_segments=previous,
            following_segments=following,
            radius_label=radius_label,
            segments=segments,
        )
        entry = {
            "entry_id": f"p2k:radius:{selected['window_id']}:{radius_label}",
            "window_id": selected["window_id"],
            "radius": radius_label,
            "stage_label": stage_label,
            "requested": context["requested"],
            "context": context,
        }
        entry["canonical_entry_sha256"] = canonical_sha256(entry)
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Human review packet and blinding
# ---------------------------------------------------------------------------


def _new_blinded_labels(
    rng: random.Random,
    count: int,
    used: set[str],
) -> list[str]:
    """Fully opaque labels; the prefix never encodes a condition/radius."""
    labels: list[str] = []
    while len(labels) < count:
        label = f"BLIND-{rng.getrandbits(32):08x}"
        if label not in used:
            used.add(label)
            labels.append(label)
    return labels


def _render_context_presentation(
    context: Mapping[str, Any],
    mechanical_target_text: str | None,
) -> str:
    """Ordered exact context with the target clearly delimited.

    The surrounding source context stays exact; the delimited target is
    replaced with B.clean_text when a mechanical pass produced one.
    """
    lines = []
    for segment in context["segments"]:
        if segment["kind"] == "target":
            target_text = (
                mechanical_target_text
                if mechanical_target_text is not None
                else segment["text"]
            )
            lines.append("⟪TARGET⟫" + target_text + "⟪/TARGET⟫")
        else:
            lines.append(segment["text"])
    return "\n".join(lines)


def _presentation_for_item(
    code: str,
    record: Mapping[str, Any] | None,
    radius_entry: Mapping[str, Any] | None,
    mechanical_target_text: str | None,
) -> dict[str, Any]:
    """Neutral reviewer-facing presentation for one blinded item.

    The object uses the same generic shape for every condition; section ids
    are neutral and the actual presentation text is embedded verbatim so the
    reviewer never needs provenance to read or score it.
    """
    sections: list[dict[str, Any]] = []
    if code in ("A", "B"):
        sections.append({
            "id": "primary",
            "text": record["content"]["text"],
        })
    elif code == "C":
        context = (
            record["content"]["context"]
            if record is not None
            else radius_entry["context"]
        )
        sections.append({
            "id": "primary",
            "text": _render_context_presentation(context, mechanical_target_text),
        })
    elif code == "D":
        content = record["content"]
        sections.append({
            "id": "primary",
            "text": content["clean_target_transcript"],
        })
        polish = content["semantic_polish"]
        if polish is not None and isinstance(polish, Mapping):
            sections.append({
                "id": "supplement",
                "text": polish["polished_text"],
            })
    else:
        raise ValueError(f"unknown presentation condition {code!r}")
    target_text = (
        record["target"]["text"]
        if record is not None
        else radius_entry["context"]["target"]["text"]
    )
    return {
        "schema_version": PRESENTATION_SCHEMA_VERSION,
        "target_sha256": text_sha256(target_text),
        "displayed_target_sha256": text_sha256(
            mechanical_target_text
            if code == "C" and mechanical_target_text is not None
            else target_text
        ),
        "sections": sections,
    }


def build_human_review_packet(
    records: list[dict[str, Any]],
    radius_entries: list[dict[str, Any]],
    *,
    include_d: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build blinded human review items plus the separately retained mapping.

    Official packet items are strictly blind: no condition code, record type,
    radius label, stage name, or unblinded record id appears anywhere.  Each
    item embeds the actual reviewer-facing presentation text; exact
    provenance lives only in the separate mapping artifact.  Presentation
    order is shuffled deterministically per window and labels never encode
    the condition/radius.
    """
    rng = random.Random("phase2k-hr-blinding-20260819")
    items: list[dict[str, Any]] = []
    mapping: dict[str, Any] = {}
    used_labels: set[str] = set()
    per_window_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        per_window_records.setdefault(record["window_id"], []).append(record)
    mechanical_by_window: dict[str, str | None] = {}
    for record in records:
        if record["record_type"] != "B":
            continue
        mechanical_by_window[record["window_id"]] = (
            record["content"]["clean_text"]
            if record["content"].get("generation_status") == "GENERATED"
            else None
        )
    radius_by_window: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in radius_entries:
        radius_by_window.setdefault(entry["window_id"], {})[entry["radius"]] = entry

    for window_id in sorted(per_window_records):
        window_records = per_window_records[window_id]
        condition_records = {record["record_type"]: record for record in window_records}
        available: list[tuple[str, dict[str, Any] | None, str | None]] = []
        for code in ("A", "B", "C", "D"):
            if code == "D" and not include_d:
                continue
            record = condition_records.get(code)
            if code == "D" and record is not None and record["content"].get(
                "generation_status",
            ) != "GENERATED":
                continue
            available.append((code, record, None))
        for radius in RADIUS_ENTRY_LABELS:
            entry = radius_by_window.get(window_id, {}).get(radius)
            if entry is not None:
                available.append(("C", None, radius))
        labels = _new_blinded_labels(rng, len(available), used_labels)
        pool = list(available)
        rng.shuffle(pool)
        mechanical_target_text = mechanical_by_window.get(window_id)
        for (code, record, radius), label in zip(pool, labels):
            if record is not None:
                content_sha256 = record["canonical_record_sha256"]
                mapping_source = {
                    "record_id": record["record_id"],
                    "record_sha256": record["canonical_record_sha256"],
                    "entry_id": None,
                    "entry_sha256": None,
                    "target_text_sha256": record["target"]["text_sha256"],
                    "target_source_absolute_start": record["target"][
                        "source_absolute_start"
                    ],
                    "target_source_absolute_end": record["target"][
                        "source_absolute_end"
                    ],
                }
                record_for_presentation = record
                radius_entry_for_presentation = None
            else:
                entry = radius_by_window[window_id][radius]
                content_sha256 = entry["canonical_entry_sha256"]
                mapping_source = {
                    "record_id": None,
                    "record_sha256": None,
                    "entry_id": entry["entry_id"],
                    "entry_sha256": entry["canonical_entry_sha256"],
                    "target_text_sha256": entry["context"]["target"][
                        "text_sha256"
                    ],
                    "target_source_absolute_start": entry["context"]["target"][
                        "source_absolute_start"
                    ],
                    "target_source_absolute_end": entry["context"]["target"][
                        "source_absolute_end"
                    ],
                }
                record_for_presentation = None
                radius_entry_for_presentation = entry
            presentation = _presentation_for_item(
                code,
                record_for_presentation,
                radius_entry_for_presentation,
                mechanical_target_text,
            )
            item = {
                "review_item_id": f"p2k:hr:{window_id}:{label}",
                "window_id": window_id,
                "blinded_label": label,
                "presentation": presentation,
                "content_sha256": content_sha256,
                "scores": {field: None for field in HUMAN_SCORE_FIELDS},
                "reviewer": None,
                "completed_at": None,
                "notes": [],
            }
            items.append(item)
            mapping[label] = {
                "window_id": window_id,
                "condition_code": code,
                "radius": radius,
                "record_type": ("RADIUS_ENTRY" if record is None else code),
                "presentation_sha256": canonical_sha256(presentation),
                "presentation_target_sha256": presentation[
                    "displayed_target_sha256"
                ],
                **mapping_source,
            }
    packet = _envelope({
        "schema_version": HUMAN_PACKET_SCHEMA_VERSION,
        "purpose": (
            "Blinded Phase 2K human review packet.  Condition labels are "
            "randomized and no condition code, record type, radius label, "
            "stage name, or unblinded record identity appears in this "
            "packet.  Every item embeds the reviewer-facing presentation "
            "with blank scoring fields; the label-to-condition mapping is "
            "retained in a separate file and is bound here only by hash."
        ),
        "release_gate": RELEASE_GATE_AWAITING_REVIEW,
        "blinding": {
            "method": "seeded_random_condition_labels",
            "seed": "phase2k-hr-blinding-20260819",
            "mapping_file": "phase2k-human-review-mapping-v2.json",
        },
        "review_items": items,
        "scoring_fields": list(HUMAN_SCORE_FIELDS),
        "score_range": {"min": HUMAN_SCORE_MIN, "max": HUMAN_SCORE_MAX},
        "rubric": dict(HUMAN_SCORE_RUBRIC),
    })
    mapping_obj = _envelope({
        "schema_version": HUMAN_MAPPING_SCHEMA_VERSION,
        "purpose": (
            "Separately retained Phase 2K human-review label mapping.  This "
            "artifact carries the exact condition/radius provenance and "
            "record/entry identity that the official packet must not expose."
        ),
        "labels": dict(sorted(mapping.items())),
    })
    packet["blinding"]["mapping_sha256"] = mapping_obj["content_sha256"]
    packet["content_sha256"] = canonical_sha256({
        key: value for key, value in packet.items() if key != "content_sha256"
    })
    return packet, mapping_obj


FORBIDDEN_PACKET_KEYS = frozenset({
    "condition_code", "record_type", "radius", "radius_label", "stage",
    "stage_label", "record_id", "entry_id", "provenance", "kind",
    "semantic_stage", "generation_status", "clean_target_transcript",
    "resolved_semantic_paraphrase", "paraphrase_text", "mechanical_clean",
    "window_condition", "raw_bronze", "enlarged_context",
})
FORBIDDEN_PACKET_VALUES = frozenset({
    "A", "B", "C", "D",
    "target_only", "r2", "r5", "r10", "bounded_local_episode",
    "r1", "r3", "r4_bounded_local_episode",
    "raw_bronze", "mechanical_clean", "enlarged_context", "reconstruction",
    "NOT_GENERATED", "GENERATED",
})
_FORBIDDEN_RECORD_ID_PATTERN = re.compile(r"p2k:(rec|radius):")
_PACKET_FREE_TEXT_KEYS = frozenset({"text", "notes"})


def _scan_packet_forbidden_leaks(value: object, *, path: str) -> None:
    """Reject any structural field/value that reveals the condition/radius.

    Free-text presentation sections and reviewer notes are not scanned
    because their content is reviewer-facing text, not provenance.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_PACKET_KEYS:
                raise ValueError(
                    f"human review packet leaks forbidden key {key!r} at {path}",
                )
            if key in _PACKET_FREE_TEXT_KEYS:
                continue
            _scan_packet_forbidden_leaks(
                item, path=f"{path}.{key}",
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_packet_forbidden_leaks(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if value in FORBIDDEN_PACKET_VALUES:
            raise ValueError(
                f"human review packet leaks forbidden value {value!r} at {path}",
            )
        if _FORBIDDEN_RECORD_ID_PATTERN.search(value):
            raise ValueError(
                "human review packet leaks an unblinded record identity "
                f"at {path}",
            )


def validate_human_review_packet(
    packet: Mapping[str, Any],
    *,
    require_blank: bool,
) -> None:
    _require_exact_keys(
        packet,
        (
            "schema_version", "purpose", "release_gate", "blinding",
            "review_items", "scoring_fields", "score_range", "rubric",
            "content_sha256",
        ),
        "phase2k human review packet",
    )
    if packet["schema_version"] != HUMAN_PACKET_SCHEMA_VERSION:
        raise ValueError("human review packet schema version is invalid")
    if tuple(packet["scoring_fields"]) != HUMAN_SCORE_FIELDS:
        raise ValueError("human review packet scoring fields are invalid")
    score_range = packet["score_range"]
    if score_range != {"min": HUMAN_SCORE_MIN, "max": HUMAN_SCORE_MAX}:
        raise ValueError("human review packet score range is invalid")
    rubric = packet["rubric"]
    if not isinstance(rubric, Mapping) or set(rubric) != set(HUMAN_SCORE_FIELDS):
        raise ValueError("human review packet rubric is incomplete")
    for field, entry in rubric.items():
        if not isinstance(entry, Mapping) or set(entry) != {
            "description", "direction", "not_applicable_allowed",
        }:
            raise ValueError(f"human review rubric entry {field} is invalid")
        if not isinstance(entry["description"], str) or not entry["description"]:
            raise ValueError(f"human review rubric {field} has no description")
        if entry["direction"] not in {"higher_is_better", "lower_is_better"}:
            raise ValueError(f"human review rubric {field} direction is invalid")
        if not isinstance(entry["not_applicable_allowed"], bool):
            raise ValueError(f"human review rubric {field} N/A flag is invalid")
    _scan_packet_forbidden_leaks(packet, path="packet")
    items = _require_list(packet["review_items"], "human review items")
    seen_ids: set[str] = set()
    for item in items:
        _require_exact_keys(
            item,
            (
                "review_item_id", "window_id", "blinded_label",
                "presentation", "content_sha256", "scores", "reviewer",
                "completed_at", "notes",
            ),
            "human review item",
        )
        _require_nonempty_string(item["review_item_id"], "review item id")
        if item["review_item_id"] in seen_ids:
            raise ValueError("human review item IDs must be unique")
        seen_ids.add(item["review_item_id"])
        _require_nonempty_string(item["window_id"], "review item window_id")
        _require_nonempty_string(item["blinded_label"], "review item label")
        if "rec:" in item["review_item_id"] or "radius:" in item[
            "review_item_id"
        ]:
            raise ValueError("review item ID encodes unblinded identity")
        presentation = item["presentation"]
        if not isinstance(presentation, Mapping):
            raise ValueError("review item presentation must be an object")
        allowed_presentation_keys = (
            frozenset({"schema_version", "target_sha256", "sections"}),
            frozenset({
                "schema_version", "target_sha256", "displayed_target_sha256",
                "sections",
            }),
        )
        if frozenset(presentation) not in allowed_presentation_keys:
            raise ValueError("review item presentation keys are invalid")
        if presentation["schema_version"] != PRESENTATION_SCHEMA_VERSION:
            raise ValueError("review item presentation schema version is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", presentation["target_sha256"]):
            raise ValueError("review item presentation target hash is invalid")
        if "displayed_target_sha256" in presentation and not re.fullmatch(
            r"[0-9a-f]{64}", presentation["displayed_target_sha256"],
        ):
            raise ValueError(
                "review item presentation displayed target hash is invalid",
            )
        sections = _require_list(
            presentation["sections"], "review item presentation sections",
        )
        if not sections:
            raise ValueError("review item presentation has no content")
        seen_section_ids: set[str] = set()
        for section in sections:
            if not isinstance(section, Mapping):
                raise ValueError(
                    "review item presentation section must be an object",
                )
            _require_exact_keys(
                section, ("id", "text"), "review item presentation section",
            )
            section_id = _require_nonempty_string(
                section["id"], "presentation section id",
            )
            if section_id not in {"primary", "supplement"}:
                raise ValueError("presentation section id is not neutral")
            if section_id in seen_section_ids:
                raise ValueError("presentation section ids must be unique")
            seen_section_ids.add(section_id)
            _require_nonempty_string(section["text"], "presentation section text")
        if "primary" not in seen_section_ids:
            raise ValueError("review item presentation requires primary content")
        if not re.fullmatch(r"[0-9a-f]{64}", item["content_sha256"]):
            raise ValueError("review item content_sha256 is invalid")
        scores = item["scores"]
        if not isinstance(scores, Mapping) or set(scores) != set(HUMAN_SCORE_FIELDS):
            raise ValueError("human review item scores are invalid")
        for field, value in scores.items():
            if require_blank and value is not None:
                raise ValueError(
                    f"official human review packet must remain blank at "
                    f"{item['review_item_id']}.{field}",
                )
            if not require_blank:
                if isinstance(value, bool) or not isinstance(value, int) \
                        or not HUMAN_SCORE_MIN <= value <= HUMAN_SCORE_MAX:
                    if value not in NOT_APPLICABLE_SENTINELS:
                        raise ValueError(
                            f"human review score "
                            f"{item['review_item_id']}.{field} is invalid",
                        )
        if not require_blank:
            if not isinstance(item["reviewer"], str) or not item["reviewer"]:
                raise ValueError("completed human review requires a reviewer")
            if not isinstance(item["completed_at"], str) or not item["completed_at"]:
                raise ValueError("completed human review requires completed_at")
        if item["reviewer"] is not None and require_blank:
            raise ValueError("blank human review packet cannot be signed")
    expected_hash = canonical_sha256({
        key: value for key, value in packet.items() if key != "content_sha256"
    })
    if packet["content_sha256"] != expected_hash:
        raise ValueError("human review packet content_sha256 is invalid")


def import_completed_human_reviews(
    packet: Mapping[str, Any],
    reviews: Mapping[str, Any],
    *,
    reviewer: str,
    completed_at: str,
) -> dict[str, Any]:
    """Return a finalized packet with human scores; refuses incomplete input."""
    validate_human_review_packet(packet, require_blank=True)
    items = packet["review_items"]
    review_ids = set(reviews)
    item_ids = {item["review_item_id"] for item in items}
    if review_ids != item_ids:
        missing = sorted(item_ids - review_ids)
        extra = sorted(review_ids - item_ids)
        raise ValueError(
            "completed human reviews must cover every review item; "
            f"missing={missing} extra={extra}",
        )
    reviewer = _require_nonempty_string(reviewer, "reviewer")
    completed_at = _require_nonempty_string(completed_at, "completed_at")
    finalized_items = []
    for item in items:
        review = reviews[item["review_item_id"]]
        if not isinstance(review, Mapping):
            raise ValueError("human review entry must be an object")
        _require_exact_keys(
            review,
            ("scores", "reviewer", "completed_at", "notes"),
            "human review entry",
        )
        scores = review["scores"]
        if not isinstance(scores, Mapping) or set(scores) != set(HUMAN_SCORE_FIELDS):
            raise ValueError("human review scores are incomplete")
        validated_scores: dict[str, Any] = {}
        for field, value in scores.items():
            if isinstance(value, bool) or not isinstance(value, int) \
                    or not HUMAN_SCORE_MIN <= value <= HUMAN_SCORE_MAX:
                if value not in NOT_APPLICABLE_SENTINELS:
                    raise ValueError(f"human review score {field} is invalid")
            validated_scores[field] = value
        notes = _require_list(review["notes"], "human review notes")
        if any(not isinstance(note, str) for note in notes):
            raise ValueError("human review notes must be strings")
        finalized_items.append({
            **item,
            "scores": validated_scores,
            "reviewer": _require_nonempty_string(
                review["reviewer"], "review entry reviewer",
            ),
            "completed_at": _require_nonempty_string(
                review["completed_at"], "review entry completed_at",
            ),
            "notes": list(notes),
        })
    finalized = _envelope({
        "schema_version": packet["schema_version"],
        "purpose": packet["purpose"],
        "release_gate": RELEASE_GATE_REVIEWED,
        "blinding": dict(packet["blinding"]),
        "review_items": finalized_items,
        "scoring_fields": list(packet["scoring_fields"]),
        "score_range": dict(packet["score_range"]),
        "rubric": dict(packet["rubric"]),
    })
    finalized["blinding"]["imported_by"] = reviewer
    finalized["blinding"]["imported_at"] = completed_at
    finalized["content_sha256"] = canonical_sha256({
        key: value for key, value in finalized.items() if key != "content_sha256"
    })
    validate_human_review_packet(finalized, require_blank=False)
    return finalized


def _item_scores(item: Mapping[str, Any]) -> dict[str, Any]:
    return {field: item["scores"][field] for field in HUMAN_SCORE_FIELDS}


def _applicable_values(scores: Mapping[str, Any], field: str) -> list[float]:
    value = scores[field]
    if value is None or value in NOT_APPLICABLE_SENTINELS:
        return []
    return [float(value)]


def _normalized_values(scores: Mapping[str, Any], field: str) -> list[float]:
    values = _applicable_values(scores, field)
    if field in LOWER_IS_BETTER_SCORE_FIELDS:
        return [HUMAN_SCORE_MAX - value for value in values]
    return values


def _human_metric(
    scores: Mapping[str, Any],
    fields: Iterable[str],
) -> float | None:
    """Composite over the listed fields, normalized to higher-is-better."""
    values: list[float] = []
    for field in fields:
        values.extend(_normalized_values(scores, field))
    return _mean(values)


def _field_mean(
    score_list: list[Mapping[str, Any]],
    field: str,
) -> float | None:
    values: list[float] = []
    for scores in score_list:
        values.extend(_normalized_values(scores, field))
    return _mean(values)


def _raw_field_mean(
    score_list: list[Mapping[str, Any]],
    field: str,
) -> float | None:
    values: list[float] = []
    for scores in score_list:
        values.extend(_applicable_values(scores, field))
    return _mean(values)


def _applicable_counts(
    score_list: list[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    return {
        field: {
            "applicable": sum(
                1 for scores in score_list if _applicable_values(scores, field)
            ),
            "not_applicable": sum(
                1
                for scores in score_list
                if scores[field] in NOT_APPLICABLE_SENTINELS
            ),
        }
        for field in HUMAN_SCORE_FIELDS
    }


def _percent_sufficient_by_radius(
    diagnostic_summaries: list[Mapping[str, Any]],
    radius: str,
) -> float:
    """Cumulative percent of windows sufficient at or before this radius."""
    stage_index = None
    for index, stage in enumerate(RADIUS_STAGES):
        if stage["radius"] == radius:
            stage_index = index
            break
    if stage_index is None:
        raise ValueError(f"unknown radius {radius!r}")
    bounded = list(RADIUS_STAGES)
    cumulative_labels = [stage["label"] for stage in bounded[: stage_index + 1]]
    sufficient = 0
    for summary in diagnostic_summaries:
        if summary.get("final_decision") != "SUFFICIENT":
            continue
        if summary.get("stopping_stage") in cumulative_labels:
            sufficient += 1
    return _safe_float(sufficient / max(1, len(diagnostic_summaries)))


def _unblind_items(
    finalized_packet: Mapping[str, Any],
    mapping: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Attach condition/radius provenance from the separate mapping."""
    if mapping.get("content_sha256") != finalized_packet.get("blinding", {}).get(
        "mapping_sha256",
    ):
        raise ValueError("human review mapping is not bound to the packet")
    labels = mapping["labels"]
    if not isinstance(labels, Mapping):
        raise ValueError("human review mapping labels are invalid")
    unblinded: list[dict[str, Any]] = []
    for item in finalized_packet["review_items"]:
        label = item["blinded_label"]
        entry = labels.get(label)
        if not isinstance(entry, Mapping):
            raise ValueError(f"human review label {label!r} is missing from the mapping")
        required = (
            "window_id", "condition_code", "radius", "record_type",
            "record_id", "record_sha256", "entry_id", "entry_sha256",
            "target_text_sha256", "presentation_sha256",
        )
        for key in required:
            if key not in entry:
                raise ValueError(f"human review mapping entry {label!r} is incomplete")
        if entry["window_id"] != item["window_id"]:
            raise ValueError("human review mapping window identity is inconsistent")
        unblinded.append({**item, **{
            "condition_code": entry["condition_code"],
            "radius": entry["radius"],
            "record_type": entry["record_type"],
        }})
    return unblinded


def _group_condition_metrics(
    unblinded: list[Mapping[str, Any]],
    *,
    diagnostic_summaries: list[Mapping[str, Any]],
    radius_sizes: dict[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    by_condition: dict[str, list[Mapping[str, Any]]] = {
        code: [] for code in ("A", "B", "C", "D")
    }
    for item in unblinded:
        by_condition[item["condition_code"]].append(item)
    condition_metrics: dict[str, Any] = {}
    for code, code_items in by_condition.items():
        if not code_items:
            condition_metrics[code] = {"item_count": 0}
            continue
        score_list = [_item_scores(item) for item in code_items]
        condition_metrics[code] = {
            "item_count": len(code_items),
            "entity_ability_completeness": _mean([
                metric
                for scores in score_list
                for metric in [_human_metric(
                    scores, ENTITY_ABILITY_COMPLETENESS_FIELDS,
                )]
                if metric is not None
            ]),
            "semantic_recoverability": _mean([
                metric
                for scores in score_list
                for metric in [_human_metric(
                    scores, SEMANTIC_RECOVERABILITY_FIELDS,
                )]
                if metric is not None
            ]),
            "meaning_preservation": _field_mean(score_list, "meaning_preservation"),
            "unsupported_invention": _raw_field_mean(
                score_list, "unsupported_invention",
            ),
            "remaining_ambiguity": _raw_field_mean(score_list, "remaining_ambiguity"),
            "asr_repair_correctness": _field_mean(
                score_list, "asr_repair_correctness",
            ),
            "entity_binding_correctness": _field_mean(
                score_list, "entity_binding_correctness",
            ),
            "standalone_coaching_claim": _field_mean(
                score_list, "standalone_coaching_claim",
            ),
            "applicable_counts": _applicable_counts(score_list),
            "average_context_chars": _mean([
                float(item["radius_size"]["total_chars"])
                for item in code_items
                if item.get("radius_size")
            ]),
            "average_context_tokens": _mean([
                float(item["radius_size"]["total_tokens"])
                for item in code_items
                if item.get("radius_size")
            ]),
        }
    return condition_metrics


def _group_radius_metrics(
    unblinded: list[Mapping[str, Any]],
    *,
    diagnostic_summaries: list[Mapping[str, Any]],
    radius_sizes: dict[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    by_radius: dict[str, list[Mapping[str, Any]]] = {
        label: [] for label in RADIUS_ENTRY_LABELS
    }
    for item in unblinded:
        if item["condition_code"] == "C" and item["radius"] in by_radius:
            by_radius[item["radius"]].append(item)
    radius_metrics: dict[str, Any] = {}
    for label, radius_items in by_radius.items():
        if not radius_items:
            radius_metrics[label] = {
                "item_count": 0,
                "percent_sufficient": None,
            }
            continue
        score_list = [_item_scores(item) for item in radius_items]
        sizes = radius_sizes.get(label, [])
        radius_metrics[label] = {
            "item_count": len(radius_items),
            "entity_ability_completeness": _mean([
                metric
                for scores in score_list
                for metric in [_human_metric(
                    scores, ENTITY_ABILITY_COMPLETENESS_FIELDS,
                )]
                if metric is not None
            ]),
            "semantic_recoverability": _mean([
                metric
                for scores in score_list
                for metric in [_human_metric(
                    scores, SEMANTIC_RECOVERABILITY_FIELDS,
                )]
                if metric is not None
            ]),
            "meaning_preservation": _field_mean(score_list, "meaning_preservation"),
            "unsupported_invention": _raw_field_mean(
                score_list, "unsupported_invention",
            ),
            "remaining_ambiguity": _raw_field_mean(score_list, "remaining_ambiguity"),
            "asr_repair_correctness": _field_mean(
                score_list, "asr_repair_correctness",
            ),
            "entity_binding_correctness": _field_mean(
                score_list, "entity_binding_correctness",
            ),
            "standalone_coaching_claim": _field_mean(
                score_list, "standalone_coaching_claim",
            ),
            "applicable_counts": _applicable_counts(score_list),
            "average_context_chars": _mean([
                float(size["total_chars"]) for size in sizes
            ]),
            "average_context_tokens": _mean([
                float(size["total_tokens"]) for size in sizes
            ]),
            "percent_sufficient": (
                None
                if label == "target_only"
                else _percent_sufficient_by_radius(diagnostic_summaries, label)
            ),
        }
    return radius_metrics


def evaluate_review_gate(
    finalized_packet: Mapping[str, Any],
    *,
    mapping: Mapping[str, Any],
    records_file: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic pre-registered review gate, evaluated only post-review."""
    validate_human_review_packet(finalized_packet, require_blank=False)
    unblinded = _unblind_items(finalized_packet, mapping)
    by_condition: dict[str, list[Mapping[str, Any]]] = {
        code: [] for code in ("A", "B", "C", "D")
    }
    for item in unblinded:
        by_condition[item["condition_code"]].append(item)
    d_score_list = [_item_scores(item) for item in by_condition["D"]]
    a_score_list = [_item_scores(item) for item in by_condition["A"]]
    all_score_list = [_item_scores(item) for item in unblinded]
    reasons: list[dict[str, Any]] = []

    d_recoverability = _mean([
        metric
        for scores in d_score_list
        for metric in [_human_metric(scores, SEMANTIC_RECOVERABILITY_FIELDS)]
        if metric is not None
    ])
    a_recoverability = _mean([
        metric
        for scores in a_score_list
        for metric in [_human_metric(scores, SEMANTIC_RECOVERABILITY_FIELDS)]
        if metric is not None
    ])
    d_meaning = _field_mean(d_score_list, "meaning_preservation")
    d_unsupported = _raw_field_mean(d_score_list, "unsupported_invention")
    asr_correctness = _field_mean(all_score_list, "asr_repair_correctness")
    entity_correctness = _field_mean(all_score_list, "entity_binding_correctness")

    def _check(
        label: str,
        passed: bool,
        *,
        value: float | None,
        threshold: float,
        detail: str,
    ) -> None:
        reasons.append({
            "criterion": label,
            "passed": bool(passed),
            "value": value,
            "threshold": threshold,
            "detail": detail,
        })

    missing_d = len(d_score_list) == 0
    if missing_d:
        reasons.append({
            "criterion": "d_items_available",
            "passed": False,
            "value": None,
            "threshold": 1,
            "detail": "no reviewed D items are available",
        })
    else:
        _check(
            "d_semantic_recoverability",
            d_recoverability is not None
            and d_recoverability >= REVIEW_GATE_SPEC["d_semantic_recoverability_min"],
            value=d_recoverability,
            threshold=REVIEW_GATE_SPEC["d_semantic_recoverability_min"],
            detail="D mean semantic-recoverability composite",
        )
        _check(
            "d_over_a_semantic_recoverability_gain",
            d_recoverability is not None and a_recoverability is not None
            and d_recoverability - a_recoverability
            >= REVIEW_GATE_SPEC["d_over_a_semantic_recoverability_gain_min"],
            value=(
                None
                if d_recoverability is None or a_recoverability is None
                else _safe_float(d_recoverability - a_recoverability)
            ),
            threshold=REVIEW_GATE_SPEC["d_over_a_semantic_recoverability_gain_min"],
            detail="D minus A semantic-recoverability composite",
        )
        _check(
            "d_meaning_preservation",
            d_meaning is not None
            and d_meaning >= REVIEW_GATE_SPEC["d_meaning_preservation_min"],
            value=d_meaning,
            threshold=REVIEW_GATE_SPEC["d_meaning_preservation_min"],
            detail="D mean meaning preservation",
        )
        _check(
            "d_unsupported_invention",
            d_unsupported is not None
            and d_unsupported <= REVIEW_GATE_SPEC["d_unsupported_invention_max"],
            value=d_unsupported,
            threshold=REVIEW_GATE_SPEC["d_unsupported_invention_max"],
            detail="D mean unsupported invention (lower is better)",
        )
        for field in sorted(GATE_REQUIRED_APPLICABLE_FIELDS):
            not_applicable = [
                item["review_item_id"]
                for item in by_condition["D"]
                if item["scores"][field] in NOT_APPLICABLE_SENTINELS
            ]
            if not_applicable:
                reasons.append({
                    "criterion": f"d_{field}_applicable",
                    "passed": False,
                    "value": len(d_score_list) - len(not_applicable),
                    "threshold": len(d_score_list),
                    "detail": (
                        "D coaching-claim/recoverability fields must all be "
                        f"applicable; NOT_APPLICABLE at {not_applicable}"
                    ),
                })
    asr_applicable = sum(
        1 for scores in all_score_list if _applicable_values(
            scores, "asr_repair_correctness",
        )
    )
    entity_applicable = sum(
        1 for scores in all_score_list if _applicable_values(
            scores, "entity_binding_correctness",
        )
    )
    if asr_applicable == 0:
        reasons.append({
            "criterion": "asr_repair_correctness_applicable",
            "passed": False,
            "value": 0,
            "threshold": 1,
            "detail": "at least one applicable ASR-repair correctness score required",
        })
    else:
        _check(
            "asr_repair_correctness",
            asr_correctness is not None
            and asr_correctness >= REVIEW_GATE_SPEC["asr_repair_correctness_min"],
            value=asr_correctness,
            threshold=REVIEW_GATE_SPEC["asr_repair_correctness_min"],
            detail="mean ASR-repair correctness over applicable reviewed items",
        )
    if entity_applicable == 0:
        reasons.append({
            "criterion": "entity_binding_correctness_applicable",
            "passed": False,
            "value": 0,
            "threshold": 1,
            "detail": "at least one applicable entity-binding correctness score required",
        })
    else:
        _check(
            "entity_binding_correctness",
            entity_correctness is not None
            and entity_correctness >= REVIEW_GATE_SPEC["entity_binding_correctness_min"],
            value=entity_correctness,
            threshold=REVIEW_GATE_SPEC["entity_binding_correctness_min"],
            detail="mean entity-binding correctness over applicable reviewed items",
        )
    passed = all(reason["passed"] for reason in reasons) and bool(reasons)
    return {
        "schema_version": REVIEW_GATE_SPEC["schema_version"],
        "status": "PASSED" if passed else "FAILED",
        "evaluated": True,
        "thresholds": {
            "d_semantic_recoverability_min": REVIEW_GATE_SPEC[
                "d_semantic_recoverability_min"
            ],
            "d_over_a_semantic_recoverability_gain_min": REVIEW_GATE_SPEC[
                "d_over_a_semantic_recoverability_gain_min"
            ],
            "d_meaning_preservation_min": REVIEW_GATE_SPEC[
                "d_meaning_preservation_min"
            ],
            "d_unsupported_invention_max": REVIEW_GATE_SPEC[
                "d_unsupported_invention_max"
            ],
            "asr_repair_correctness_min": REVIEW_GATE_SPEC[
                "asr_repair_correctness_min"
            ],
            "entity_binding_correctness_min": REVIEW_GATE_SPEC[
                "entity_binding_correctness_min"
            ],
        },
        "metrics": {
            "d_semantic_recoverability": d_recoverability,
            "a_semantic_recoverability": a_recoverability,
            "d_over_a_semantic_recoverability_gain": (
                None
                if d_recoverability is None or a_recoverability is None
                else _safe_float(d_recoverability - a_recoverability)
            ),
            "d_meaning_preservation": d_meaning,
            "d_unsupported_invention": d_unsupported,
            "asr_repair_correctness": asr_correctness,
            "entity_binding_correctness": entity_correctness,
            "asr_repair_correctness_applicable": asr_applicable,
            "entity_binding_correctness_applicable": entity_applicable,
        },
        "reasons": reasons,
    }


def summarize_human_reviews(
    finalized_packet: Mapping[str, Any],
    *,
    mapping: Mapping[str, Any],
    records_file: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summary metrics by condition and radius from completed human reviews.

    The finalized packet remains blind; condition/radius provenance is taken
    exclusively from the separate mapping artifact.
    """
    validate_human_review_packet(finalized_packet, require_blank=False)
    unblinded = _unblind_items(finalized_packet, mapping)
    diagnostic_summaries: list[Mapping[str, Any]] = []
    if records_file is not None:
        diagnostic_summaries = records_file.get("diagnostic_summaries", [])
    radius_sizes: dict[str, list[Mapping[str, Any]]] = {}
    if records_file is not None:
        for entry in records_file.get("context_radius_entries", []):
            radius_sizes.setdefault(entry["radius"], []).append(
                entry["context"]["actual"],
            )
    condition_metrics = _group_condition_metrics(
        unblinded,
        diagnostic_summaries=diagnostic_summaries,
        radius_sizes=radius_sizes,
    )
    radius_metrics = _group_radius_metrics(
        unblinded,
        diagnostic_summaries=diagnostic_summaries,
        radius_sizes=radius_sizes,
    )
    return {
        "schema_version": HUMAN_SUMMARY_SCHEMA_VERSION,
        "by_condition": condition_metrics,
        "by_radius": radius_metrics,
        "overall": {
            "item_count": len(finalized_packet["review_items"]),
            "window_count": len({
                item["window_id"] for item in finalized_packet["review_items"]
            }),
        },
        "review_gate": evaluate_review_gate(
            finalized_packet,
            mapping=mapping,
            records_file=records_file,
        ),
    }


def build_count_report_skeleton() -> dict[str, Any]:
    """Exact Phase 2K count-report skeleton with no invented values."""
    return {
        "schema_version": COUNT_REPORT_SCHEMA_VERSION,
        "review_items": None,
        "windows": None,
        "asr": {
            "proposed": None,
            "approved": None,
            "approval_rate": None,
            "by_repair_type": None,
            "by_confidence": None,
        },
        "entity": {
            "proposed_resolvable": None,
            "approved": None,
            "precision": None,
            "required_resolvable": None,
            "required_resolvable_approved": None,
            "required_resolvable_recall": None,
        },
        "ability_ownership": {
            "proposed": None,
            "approved": None,
            "accuracy": None,
        },
        "unsupported": {
            "count": None,
            "statement_count": None,
            "rate": None,
        },
        "polish_preservation": {
            "modality_preserved_rate": None,
            "negation_preserved_rate": None,
            "uncertainty_preserved_rate": None,
            "approved_statements": None,
        },
        "first_failures": {
            "by_stage": None,
            "reconstruction_count": None,
        },
    }


def build_closeout_status(
    *,
    human_review_complete: bool,
    downstream_comparison_complete: bool,
    closeout_decision: str | None = None,
    count_report: Mapping[str, Any] | None = None,
    downstream_comparison: Mapping[str, Any] | None = None,
    human_review_gate_passed: bool | None = None,
) -> dict[str, Any]:
    """Deterministic Phase 2K closeout gate; no invented downstream values.

    A final closeout requires the human review gate to have passed and a
    validated v2 downstream comparison whose decision matches
    ``closeout_decision``.  The validated comparison is embedded as
    ``downstream_comparison`` so the closeout artifact carries the exact
    measured evidence, metrics, deltas, and diagnosis instead of discarding
    them.
    """
    _require_bool(human_review_complete, "human_review_complete")
    _require_bool(
        downstream_comparison_complete, "downstream_comparison_complete",
    )
    if human_review_gate_passed is not None:
        _require_bool(
            human_review_gate_passed, "human_review_gate_passed",
        )
    if human_review_gate_passed is False:
        if (
            downstream_comparison_complete
            or downstream_comparison is not None
            or closeout_decision is not None
        ):
            raise ValueError(
                "a failed human review gate cannot close Phase 2K",
            )
        return {
            "schema_version": CLOSEOUT_STATUS_SCHEMA_VERSION,
            "status": "WAITING_FOR_HUMAN_REVIEW",
            "inputs_complete": False,
            "count_report": dict(count_report or build_count_report_skeleton()),
            "downstream_comparison": None,
        }
    if not human_review_complete:
        return {
            "schema_version": CLOSEOUT_STATUS_SCHEMA_VERSION,
            "status": "WAITING_FOR_HUMAN_REVIEW",
            "inputs_complete": False,
            "count_report": dict(count_report or build_count_report_skeleton()),
            "downstream_comparison": None,
        }
    if not downstream_comparison_complete:
        if downstream_comparison is not None:
            raise ValueError(
                "downstream_comparison supplied while not complete",
            )
        return {
            "schema_version": CLOSEOUT_STATUS_SCHEMA_VERSION,
            "status": "WAITING_FOR_DOWNSTREAM",
            "inputs_complete": False,
            "count_report": dict(count_report or build_count_report_skeleton()),
            "downstream_comparison": None,
        }
    if not isinstance(downstream_comparison, Mapping):
        raise ValueError(
            "downstream_comparison is required when the comparison is complete",
        )
    decision = _require_enum(
        closeout_decision, FINAL_CLOSEOUT_STATUSES, "closeout decision",
    )
    if downstream_comparison.get("decision") != decision:
        raise ValueError(
            "closeout decision must match the downstream comparison decision",
        )
    return {
        "schema_version": CLOSEOUT_STATUS_SCHEMA_VERSION,
        "status": decision,
        "inputs_complete": True,
        "count_report": dict(count_report or build_count_report_skeleton()),
        "downstream_comparison": dict(downstream_comparison),
    }
# Output directory build / validation
# ---------------------------------------------------------------------------

OUTPUT_FILENAMES = {
    "frozen_input_manifest": "phase2k-frozen-input-manifest-v1.json",
    "records": "phase2k-reconstruction-records-v7.json",
    "human_packet": "phase2k-human-review-packet-v2.json",
    "human_mapping": "phase2k-human-review-mapping-v2.json",
    "build_summary": "phase2k-build-summary-v5.json",
    "transformation_audit": "phase2k-transformation-audit-packet-v2.json",
    "finalized_packet": "phase2k-human-review-packet-v2-finalized.json",
    "human_summary": "phase2k-human-review-summary-v1.json",
    "finalized_transformation_audit": (
        "phase2k-transformation-audit-packet-v2-finalized.json"
    ),
    "transformation_summary": (
        "phase2k-transformation-audit-summary-v1.json"
    ),
    "closeout_status": "phase2k-closeout-status-v2.json",
}


def _config_hash() -> str:
    return canonical_sha256({
        "config_version": CONFIG_VERSION,
        "token_fallback_bound": TOKEN_FALLBACK_BOUND,
        "hard_segment_cap_per_side": HARD_SEGMENT_CAP_PER_SIDE,
        "hard_char_cap_per_side": HARD_CHAR_CAP_PER_SIDE,
        "bounded_local_episode_segments": BOUNDED_LOCAL_EPISODE_SEGMENTS,
        "vocabulary_hash": lexical_vocabulary_hash(),
        "vocabulary_schema_version": LEAGUE_VOCABULARY_SCHEMA_VERSION,
        "metadata_adapter_schema_version": METADATA_ADAPTER_SCHEMA_VERSION,
        "mechanical_prompt_version": MECHANICAL_PROMPT_VERSION,
        "mechanical_correction_prompt_version": MECHANICAL_CORRECTION_PROMPT_VERSION,
        "radius_stages": [
            {
                "label": stage["label"],
                "radius": stage["radius"],
                "previous": stage["previous"],
                "following": stage["following"],
                "max": stage["max"],
            }
            for stage in RADIUS_STAGES
        ],
    })


def build_frozen_input_manifest(
    *,
    manifest_path: Path,
    packet_path: Path,
    doc_path: Path | None,
    db_path: Path,
    manifest: Mapping[str, Any],
    packet: Mapping[str, Any],
    transcripts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    input_hashes = frozen_input_hashes(
        manifest_path, packet_path, manifest, packet,
    )
    if doc_path is not None:
        input_hashes["phase2j_replication_doc"] = {
            "path": normalize_path_locator(doc_path),
            "file_sha256": file_sha256(doc_path),
        }
    windows = []
    for selected in manifest["selected"]:
        window_id = selected["window_id"]
        transcript_info = transcripts[selected["upstream_source_id"]]
        windows.append({
            "window_id": window_id,
            "source_group_id": selected["source_group_id"],
            "upstream_source_id": selected["upstream_source_id"],
            "upstream_start": selected["upstream_start"],
            "upstream_end": selected["upstream_end"],
            "target_text": selected["source_text"],
            "target_text_sha256": selected["source_text_sha256"],
            "target_char_length": selected["source_text_char_length"],
            "full_transcript_sha256": transcript_info["transcript_sha256"],
            "full_transcript_char_length": transcript_info["transcript_char_length"],
            "metadata": dict(selected["metadata"]),
            "partition": selected["partition"],
            "asr_punctuation_band": selected["asr_punctuation_band"],
            "reviewed_packet_annotation_id": next(
                record["annotation_id"]
                for record in packet["records"]
                if record["window_id"] == window_id
            ),
        })
    return _envelope({
        "schema_version": FROZEN_INPUT_MANIFEST_SCHEMA_VERSION,
        "purpose": (
            "Frozen Phase 2K inputs: Phase 2J file/canonical hashes, the "
            "read-only transcript DB hash, and all 30 exact Phase 2J targets."
        ),
        "pipeline_version": PIPELINE_VERSION,
        "config_version": CONFIG_VERSION,
        "config_hash": _config_hash(),
        "phase2j_inputs": input_hashes,
        "transcript_db": {
            "path": normalize_path_locator(db_path),
            "file_sha256": file_sha256(db_path),
        },
        "window_count": len(windows),
        "windows": windows,
        "game": "lol",
    })


def _window_attempts_dir(output_dir: Path, window_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", window_id)
    return output_dir / "attempts" / safe


def _write_attempts(
    output_dir: Path,
    attempts: list[Mapping[str, Any]],
    window_id: str,
) -> list[str]:
    directory = _window_attempts_dir(output_dir, window_id)
    paths = []
    for attempt in attempts:
        attempt_index = attempt.get("attempt_index", 0)
        if attempt_index == 0:
            relative = f"{attempt['stage']}.json"
        else:
            relative = f"{attempt['stage']}.attempt-{attempt_index}.json"
        path = directory / relative
        _write_json_atomic(path, dict(attempt))
        paths.append(str(path.relative_to(output_dir)))
    return paths


def build_transformation_audit(
    records: list[dict[str, Any]],
    window_failures: Mapping[str, Mapping[str, Any]],
    *,
    records_sha256: str,
) -> dict[str, Any]:
    """Operation-level, downstream-result-blind transformation audit packet.

    This is built only for live outputs.  It exposes operation type, exact
    evidence, and blank human decision slots; it never embeds downstream
    extractor results.
    """
    windows: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        windows.setdefault(record["window_id"], {})[record["record_type"]] = record
    window_audits: list[dict[str, Any]] = []
    operation_map: dict[str, dict[str, Any]] = {}
    operation_ordinal = 0
    for window_id in sorted(windows):
        by_type = windows[window_id]
        bronze_target = by_type["A"]["target"]
        b_content = by_type["B"]["content"]
        d_content = by_type["D"]["content"]
        operations: dict[str, list[dict[str, Any]]] = {
            "mechanical_repairs": [],
            "contextual_repairs": [],
            "entity_bindings": [],
            "pronoun_bindings": [],
            "reference_bindings": [],
            "ability_bindings": [],
            "polished_statements": [],
        }

        def register_operation(
            category: str,
            operation: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal operation_ordinal
            operations[category].append(operation)
            operation_map[operation["operation_id"]] = {
                "operation_id": operation["operation_id"],
                "window_id": window_id,
                "category": category,
                "operation_kind": operation["operation_kind"],
                "ordinal": operation_ordinal,
            }
            operation_ordinal += 1
            return operation

        for repair in b_content.get("repairs", []):
            register_operation(
                "mechanical_repairs",
                {
                    "operation_id": f"{window_id}::mech::{repair['repair_id']}",
                    "operation_kind": "MECHANICAL_REPAIR",
                    "repair_type": repair["repair_type"],
                    "confidence": repair["confidence"],
                    "original_text": repair["original_text"],
                    "replacement": repair["replacement"],
                    "evidence_spans": repair["evidence_spans"],
                    "decision": None,
                    "corrected_replacement": None,
                    "error_taxonomy": None,
                },
            )
        reconstruction = d_content.get("reconstruction")
        polish = d_content.get("semantic_polish")
        if isinstance(reconstruction, Mapping):
            for repair in reconstruction.get("contextual_repairs", []):
                register_operation(
                    "contextual_repairs",
                    {
                        "operation_id": (
                            f"{window_id}::ctx::{repair['repair_id']}"
                        ),
                        "operation_kind": "CONTEXTUAL_REPAIR",
                        "repair_type": repair["repair_type"],
                        "confidence": repair["confidence"],
                        "original_text": repair["original_text"],
                        "replacement": repair["replacement"],
                        "evidence_spans": repair["evidence_spans"],
                        "decision": None,
                        "corrected_replacement": None,
                        "error_taxonomy": None,
                    },
                )
            for binding in reconstruction.get("bindings", []):
                if binding["slot"] == "ability_ownership":
                    operation_kind = "ABILITY_BINDING"
                    category = "ability_bindings"
                elif binding["slot"] == "pronouns":
                    operation_kind = "PRONOUN_BINDING"
                    category = "pronoun_bindings"
                elif binding["slot"] in {"temporal_refs", "discourse_refs"}:
                    operation_kind = "REFERENCE_BINDING"
                    category = "reference_bindings"
                else:
                    operation_kind = "ENTITY_BINDING"
                    category = "entity_bindings"
                entry = {
                    "operation_id": (
                        f"{window_id}::bind::{binding['binding_id']}"
                    ),
                    "operation_kind": operation_kind,
                    "binding_id": binding["binding_id"],
                    "slot": binding["slot"],
                    "mention": binding["mention"],
                    "resolved_candidate": binding["resolved_candidate"],
                    "resolved_status": binding["resolved_status"],
                    "evidence_spans": binding["evidence_spans"],
                    "human_resolvable_required": (
                        binding["resolved_status"] == "RESOLVED"
                    ),
                    "decision": None,
                    "error_taxonomy": None,
                }
                register_operation(category, entry)
        if isinstance(polish, Mapping):
            for statement in polish.get("statements", []):
                register_operation(
                    "polished_statements",
                    {
                        "operation_id": (
                            f"{window_id}::stmt::{statement['statement_id']}"
                        ),
                        "operation_kind": "POLISHED_STATEMENT",
                        "statement_id": statement["statement_id"],
                        "text": statement["text"],
                        "evidence_spans": statement["evidence_spans"],
                        "reconstruction_operation_ids": statement[
                            "reconstruction_operation_ids"
                        ],
                        "support_mode": statement["support_mode"],
                        "unchanged_source_quote": statement[
                            "unchanged_source_quote"
                        ],
                        "decision": None,
                        "supported": None,
                        "uncertainty_preserved": None,
                        "negation_preserved": None,
                        "modality_preserved": None,
                        "causality_invented": None,
                        "source_detail_dropped": None,
                        "error_taxonomy": None,
                    },
                )
        failure = window_failures.get(window_id)
        reconstruction_failure = (
            failure
            if failure is not None and failure.get("stage") == "reconstruction"
            else None
        )
        failure_block = None
        if failure is not None:
            failure_block = {
                "stage": failure["stage"],
                "prompt_version": failure["prompt_version"],
                "response_schema_version": failure[
                    "response_schema_version"
                ],
                "error": failure["error"],
                "error_taxonomy": None,
            }
        reconstruction_failure_block = None
        if reconstruction_failure is not None:
            reconstruction_failure_block = {
                "stage": reconstruction_failure["stage"],
                "prompt_version": reconstruction_failure["prompt_version"],
                "response_schema_version": reconstruction_failure[
                    "response_schema_version"
                ],
                "error": reconstruction_failure["error"],
                "error_taxonomy": None,
            }
        window_audits.append({
            "window_id": window_id,
            "bronze_target": {
                "text": bronze_target["text"],
                "text_sha256": bronze_target["text_sha256"],
                "source_absolute_start": bronze_target["source_absolute_start"],
                "source_absolute_end": bronze_target["source_absolute_end"],
            },
            "operations": operations,
            "first_failure": failure_block,
            "first_reconstruction_failure": reconstruction_failure_block,
        })
    return _envelope({
        "schema_version": TRANSFORMATION_AUDIT_SCHEMA_VERSION,
        "purpose": (
            "Downstream-result-blind Phase 2K transformation audit for live "
            "outputs.  It exposes operation type and exact evidence for "
            "correctness review but never downstream extractor results, and "
            "carries blank human decision slots at operation level."
        ),
        "release_gate": RELEASE_GATE_AWAITING_REVIEW,
        "binding": {"records_sha256": records_sha256},
        "error_taxonomy": list(AUDIT_ERROR_TAXONOMY),
        "decisions": list(AUDIT_OPERATION_DECISIONS),
        "operation_kinds": list(TRANSFORMATION_OPERATION_KINDS),
        "operation_map": dict(sorted(operation_map.items())),
        "window_audits": window_audits,
    })


def _audit_operation_categories() -> tuple[str, ...]:
    return (
        "mechanical_repairs",
        "contextual_repairs",
        "entity_bindings",
        "pronoun_bindings",
        "reference_bindings",
        "ability_bindings",
        "polished_statements",
    )


def _audit_operation_kind_for_category(category: str) -> str:
    kinds = {
        "mechanical_repairs": "MECHANICAL_REPAIR",
        "contextual_repairs": "CONTEXTUAL_REPAIR",
        "entity_bindings": "ENTITY_BINDING",
        "pronoun_bindings": "PRONOUN_BINDING",
        "reference_bindings": "REFERENCE_BINDING",
        "ability_bindings": "ABILITY_BINDING",
        "polished_statements": "POLISHED_STATEMENT",
    }
    if category not in kinds:
        raise ValueError(f"unknown transformation audit category {category!r}")
    return kinds[category]


def _validate_transformation_audit_operation(
    operation: object,
    *,
    category: str,
    label: str,
    require_completed: bool,
    bronze_target: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one transformation-audit operation.

    Binding mentions are the canonical full five-field Bronze span object
    (``target_local_start``/``target_local_end``,
    ``source_absolute_start``/``source_absolute_end``, ``text``) and are
    validated against the window Bronze target exactly like reconstruction
    bindings.
    """
    if not isinstance(operation, Mapping):
        raise ValueError(f"{label} must be an object")
    kind = _audit_operation_kind_for_category(category)
    common = ("operation_id", "operation_kind")
    repair_keys = (
        "repair_type", "confidence", "original_text", "replacement",
        "evidence_spans", "decision", "corrected_replacement",
        "error_taxonomy",
    )
    binding_keys = (
        "binding_id", "slot", "mention", "resolved_candidate",
        "resolved_status", "evidence_spans", "human_resolvable_required",
        "decision", "error_taxonomy",
    )
    statement_keys = (
        "statement_id", "text", "evidence_spans",
        "reconstruction_operation_ids", "support_mode",
        "unchanged_source_quote", "decision", "supported",
        "uncertainty_preserved", "negation_preserved", "modality_preserved",
        "causality_invented", "source_detail_dropped", "error_taxonomy",
    )
    if category in {"mechanical_repairs", "contextual_repairs"}:
        _require_exact_keys(operation, common + repair_keys, label)
        _require_enum(
            operation["repair_type"],
            (
                MECHANICAL_REPAIR_TYPES
                if category == "mechanical_repairs"
                else CONTEXTUAL_REPAIR_TYPES
            ),
            f"{label} repair_type",
        )
        _require_enum(
            operation["confidence"],
            CONFIDENCE_LEVELS,
            f"{label} confidence",
        )
        _require_string(operation["original_text"], f"{label} original_text")
        _require_string(operation["replacement"], f"{label} replacement")
        _require_list(operation["evidence_spans"], f"{label} evidence_spans")
    elif category.endswith("bindings"):
        _require_exact_keys(operation, common + binding_keys, label)
        _require_enum(operation["slot"], SLOT_KEYS, f"{label} slot")
        _validate_bronze_span(
            operation["mention"],
            bronze_text=bronze_target["text"],
            base_offset=bronze_target["source_absolute_start"],
            label=f"{label} mention",
        )
        _require_enum(
            operation["resolved_status"],
            BINDING_STATUSES,
            f"{label} resolved_status",
        )
        _require_string(
            operation["resolved_candidate"], f"{label} resolved_candidate",
        )
        if not isinstance(
            operation["human_resolvable_required"], bool,
        ):
            raise ValueError(
                f"{label} human_resolvable_required must be a boolean",
            )
    else:
        _require_exact_keys(operation, common + statement_keys, label)
        _require_nonempty_string(operation["statement_id"], f"{label} statement_id")
        _require_nonempty_string(operation["text"], f"{label} text")
        _require_list(operation["evidence_spans"], f"{label} evidence_spans")
        _require_list(
            operation["reconstruction_operation_ids"],
            f"{label} reconstruction_operation_ids",
        )
        _require_enum(
            operation["support_mode"],
            POLISH_SUPPORT_MODES,
            f"{label} support_mode",
        )
        if operation["unchanged_source_quote"] is not None:
            _require_exact_keys(
                operation["unchanged_source_quote"],
                (
                    "target_local_start", "target_local_end",
                    "source_absolute_start", "source_absolute_end", "text",
                ),
                f"{label} unchanged_source_quote",
            )

    if operation["operation_kind"] != kind:
        raise ValueError(
            f"{label} operation_kind {operation['operation_kind']!r} must "
            f"be {kind!r}",
        )
    operation_id = _require_nonempty_string(
        operation["operation_id"], f"{label} operation_id",
    )
    decision = operation["decision"]
    if require_completed:
        decision = _require_enum(
            decision, AUDIT_OPERATION_DECISIONS, f"{label} decision",
        )
    elif decision is not None:
        raise ValueError(f"{label} blank audit decision must be null")

    if category in {"mechanical_repairs", "contextual_repairs"}:
        corrected = operation["corrected_replacement"]
        if require_completed:
            if corrected is not None:
                _require_string(corrected, f"{label} corrected_replacement")
        elif corrected is not None:
            raise ValueError(
                f"{label} blank audit corrected_replacement must be null",
            )
    if category.endswith("bindings") and require_completed:
        pass
    if category == "polished_statements" and require_completed:
        _require_bool(operation["supported"], f"{label} supported")
        for field in POLISH_STATEMENT_ATTESTATION_FIELDS:
            _require_bool(operation[field], f"{label} {field}")
        _require_bool(
            operation["causality_invented"], f"{label} causality_invented",
        )
        _require_bool(
            operation["source_detail_dropped"],
            f"{label} source_detail_dropped",
        )
    elif not require_completed:
        null_checks: tuple[str, ...] = ()
        if category == "polished_statements":
            null_checks = (
                "supported", "uncertainty_preserved", "negation_preserved",
                "modality_preserved", "causality_invented",
                "source_detail_dropped",
            )
        for key in null_checks:
            if operation[key] is not None:
                raise ValueError(f"{label} blank audit {key} must be null")
    taxonomy = operation["error_taxonomy"]
    if require_completed:
        if taxonomy is not None:
            _require_enum(taxonomy, AUDIT_ERROR_TAXONOMY, f"{label} error_taxonomy")
        if decision == "REJECT" and taxonomy is None:
            raise ValueError(f"{label} REJECT requires an error taxonomy value")
    elif taxonomy is not None:
        raise ValueError(f"{label} blank audit error_taxonomy must be null")
    return dict(operation)


def validate_transformation_audit_packet(
    audit: Mapping[str, Any],
    *,
    records_obj: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly validate the blank, records-bound transformation audit."""
    _validate_recomputed_content_hash(
        audit, label="phase2k transformation audit",
    )
    _require_exact_keys(
        audit,
        (
            "content_sha256", "schema_version", "purpose", "release_gate",
            "binding", "error_taxonomy", "decisions", "operation_kinds",
            "operation_map", "window_audits",
        ),
        "phase2k transformation audit",
    )
    if audit["schema_version"] != TRANSFORMATION_AUDIT_SCHEMA_VERSION:
        raise ValueError("phase2k transformation audit schema version is invalid")
    if audit["release_gate"] != RELEASE_GATE_AWAITING_REVIEW:
        raise ValueError("blank transformation audit must await human review")
    binding = audit["binding"]
    if not isinstance(binding, Mapping) or set(binding) != {"records_sha256"}:
        raise ValueError("phase2k transformation audit binding is invalid")
    if binding["records_sha256"] != records_obj["content_sha256"]:
        raise ValueError("phase2k transformation audit is not bound to records")
    if tuple(audit["error_taxonomy"]) != AUDIT_ERROR_TAXONOMY:
        raise ValueError("phase2k transformation audit error taxonomy is invalid")
    if tuple(audit["decisions"]) != AUDIT_OPERATION_DECISIONS:
        raise ValueError("phase2k transformation audit decisions are invalid")
    if tuple(audit["operation_kinds"]) != TRANSFORMATION_OPERATION_KINDS:
        raise ValueError("phase2k transformation audit operation kinds are invalid")
    windows = sorted({
        record["window_id"] for record in records_obj["records"]
    })
    window_audits = _require_list(
        audit["window_audits"], "phase2k transformation audit windows",
    )
    if any(not isinstance(item, Mapping) for item in window_audits):
        raise ValueError(
            "phase2k transformation audit windows must be objects",
        )
    if [item["window_id"] for item in window_audits] != windows:
        raise ValueError(
            "phase2k transformation audit window IDs must exactly match "
            "records windows in sorted order",
        )
    operation_map = audit["operation_map"]
    if not isinstance(operation_map, Mapping):
        raise ValueError("phase2k transformation audit operation_map is invalid")
    records_by_window = {
        record["window_id"]: {
            item["record_type"]: item
            for item in records_obj["records"]
            if item["window_id"] == record["window_id"]
        }
        for record in records_obj["records"]
    }
    seen_operation_ids: set[str] = set()
    for window_audit in window_audits:
        _require_exact_keys(
            window_audit,
            (
                "window_id", "bronze_target", "operations", "first_failure",
                "first_reconstruction_failure",
            ),
            "phase2k transformation audit window",
        )
        window_id = _require_nonempty_string(
            window_audit["window_id"], "transformation audit window_id",
        )
        window_records = records_by_window[window_id]
        if window_audit["bronze_target"] != {
            key: window_records["A"]["target"][key]
            for key in ("text", "text_sha256", "source_absolute_start",
                        "source_absolute_end")
        }:
            raise ValueError(
                "phase2k transformation audit Bronze target is inconsistent",
            )
        operations = window_audit["operations"]
        if not isinstance(operations, Mapping) or set(operations) != set(
            _audit_operation_categories()
        ):
            raise ValueError(
                "phase2k transformation audit operation categories are invalid",
            )
        for category in _audit_operation_categories():
            for index, operation in enumerate(operations[category], 1):
                validated = _validate_transformation_audit_operation(
                    operation,
                    category=category,
                    label=(
                        f"phase2k transformation audit {window_id} "
                        f"{category}[{index}]"
                    ),
                    require_completed=False,
                    bronze_target=window_records["A"]["target"],
                )
                operation_id = validated["operation_id"]
                if operation_id in seen_operation_ids:
                    raise ValueError(
                        "phase2k transformation audit operation IDs must be "
                        "unique",
                    )
                seen_operation_ids.add(operation_id)
                mapped = operation_map.get(operation_id)
                if not isinstance(mapped, Mapping) or set(mapped) != {
                    "operation_id", "window_id", "category", "operation_kind",
                    "ordinal",
                }:
                    raise ValueError(
                        "phase2k transformation audit operation_map entry is "
                        "invalid",
                    )
                if mapped["operation_id"] != operation_id or (
                    mapped["window_id"] != window_id
                ) or mapped["category"] != category or (
                    mapped["operation_kind"] != validated["operation_kind"]
                ):
                    raise ValueError(
                        "phase2k transformation audit operation_map is "
                        "inconsistent",
                    )
                _require_int(
                    mapped["ordinal"],
                    "phase2k transformation audit operation ordinal",
                    minimum=0,
                )
        for key in ("first_failure", "first_reconstruction_failure"):
            failure = window_audit[key]
            if failure is not None:
                _require_exact_keys(
                    failure,
                    (
                        "stage", "prompt_version", "response_schema_version",
                        "error", "error_taxonomy",
                    ),
                    f"phase2k transformation audit {key}",
                )
                _require_enum(
                    failure["stage"], WINDOW_GENERATION_STAGES, f"{key} stage",
                )
                if failure["error_taxonomy"] is not None:
                    raise ValueError(f"{key} blank error taxonomy must be null")
        if (
            window_audit["first_reconstruction_failure"] is not None
            and window_audit["first_reconstruction_failure"]["stage"]
            != "reconstruction"
        ):
            raise ValueError(
                "first_reconstruction_failure must have reconstruction stage",
            )
    if set(operation_map) != seen_operation_ids:
        raise ValueError(
            "phase2k transformation audit operation_map does not match "
            "window operations",
        )
    ordinals = sorted(item["ordinal"] for item in operation_map.values())
    if ordinals != list(range(len(ordinals))):
        raise ValueError(
            "phase2k transformation audit operation ordinals are not dense "
            "and ordered",
        )
    return dict(audit)


def validate_completed_transformation_audits(
    template: Mapping[str, Any],
    completed: Mapping[str, Any],
    *,
    records_obj: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate completed operation decisions against the blank template."""
    validate_transformation_audit_packet(template, records_obj=records_obj)
    _validate_recomputed_content_hash(
        completed, label="completed transformation audit",
    )
    if not isinstance(completed, Mapping) or set(completed) != set(template):
        raise ValueError("completed transformation audit key set is invalid")
    if completed["schema_version"] != COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION:
        raise ValueError("completed transformation audit schema version is invalid")
    if completed["release_gate"] != RELEASE_GATE_REVIEWED:
        raise ValueError("completed transformation audit must be REVIEWED")
    if completed["binding"] != template["binding"]:
        raise ValueError("completed transformation audit binding must match")
    if tuple(completed["error_taxonomy"]) != AUDIT_ERROR_TAXONOMY:
        raise ValueError("completed transformation audit taxonomy is invalid")
    if tuple(completed["decisions"]) != AUDIT_OPERATION_DECISIONS:
        raise ValueError("completed transformation audit decisions are invalid")
    if tuple(completed["operation_kinds"]) != TRANSFORMATION_OPERATION_KINDS:
        raise ValueError("completed transformation audit kinds are invalid")
    if completed["operation_map"] != template["operation_map"]:
        raise ValueError("completed transformation audit operation map must match")
    if any(not isinstance(item, Mapping) for item in completed["window_audits"]):
        raise ValueError(
            "completed transformation audit windows must be objects",
        )
    template_windows = {item["window_id"]: item for item in template["window_audits"]}
    completed_windows = {item["window_id"]: item for item in completed["window_audits"]}
    if set(completed_windows) != set(template_windows):
        raise ValueError("completed transformation audit windows are incomplete")
    for window_id, template_window in template_windows.items():
        completed_window = completed_windows[window_id]
        if set(completed_window) != set(template_window):
            raise ValueError(
                "completed transformation audit window key set is invalid",
            )
        if completed_window["bronze_target"] != template_window["bronze_target"]:
            raise ValueError(
                "completed transformation audit Bronze target must match",
            )
        if (
            completed_window["first_failure"] != template_window["first_failure"]
            or completed_window["first_reconstruction_failure"]
            != template_window["first_reconstruction_failure"]
        ):
            raise ValueError(
                "completed transformation audit failure records must match",
            )
        for category in _audit_operation_categories():
            template_operations = template_window["operations"][category]
            completed_operations = completed_window["operations"][category]
            if len(completed_operations) != len(template_operations):
                raise ValueError(
                    "completed transformation audit operation count is "
                    f"inconsistent for {window_id}.{category}",
                )
            for index, template_operation in enumerate(template_operations, 1):
                validated = _validate_transformation_audit_operation(
                    completed_operations[index - 1],
                    category=category,
                    label=(
                        f"completed transformation audit {window_id} "
                        f"{category}[{index}]"
                    ),
                    require_completed=True,
                    bronze_target=template_window["bronze_target"],
                )
                if validated["operation_id"] != template_operation["operation_id"]:
                    raise ValueError(
                        "completed transformation audit operation order/ID "
                        "does not match the template",
                    )
                if category in {"mechanical_repairs", "contextual_repairs"}:
                    for key in (
                        "repair_type", "confidence", "original_text",
                        "replacement", "evidence_spans",
                    ):
                        if validated[key] != template_operation[key]:
                            raise ValueError(
                                "completed transformation audit repair "
                                f"invariant {key} must match the template",
                            )
                if category.endswith("bindings") and (
                    validated["human_resolvable_required"]
                    != template_operation["human_resolvable_required"]
                ):
                    raise ValueError(
                        "completed transformation audit resolvable flag must "
                        "match the template",
                    )
                if category.endswith("bindings"):
                    for key in (
                        "binding_id", "slot", "mention", "resolved_candidate",
                        "resolved_status", "evidence_spans",
                    ):
                        if validated[key] != template_operation[key]:
                            raise ValueError(
                                "completed transformation audit binding "
                                f"invariant {key} must match the template",
                            )
                if category == "polished_statements":
                    for key in (
                        "statement_id", "text", "evidence_spans",
                        "reconstruction_operation_ids", "support_mode",
                        "unchanged_source_quote",
                    ):
                        if validated[key] != template_operation[key]:
                            raise ValueError(
                                "completed transformation audit statement "
                                f"invariant {key} must match the template",
                            )
    return dict(completed)


def summarize_transformation_audits(
    completed: Mapping[str, Any],
    *,
    records_obj: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministic transformation-audit metrics (never fabricated)."""
    asr_by_type: dict[str, dict[str, int]] = {}
    asr_by_confidence: dict[str, dict[str, int]] = {}
    asr_proposed = 0
    asr_approved = 0
    entity_proposed = 0
    entity_approved = 0
    entity_required = 0
    entity_required_approved = 0
    ability_proposed = 0
    ability_approved = 0
    statement_approved = 0
    preservation: dict[str, int] = {
        field: 0 for field in POLISH_STATEMENT_ATTESTATION_FIELDS
    }
    first_failure_counts: dict[str, int] = {}
    reconstruction_failure_count = 0
    operation_decision_counts: dict[str, int] = {
        decision: 0 for decision in AUDIT_OPERATION_DECISIONS
    }
    corrected_replacement_count = 0

    for window_audit in completed["window_audits"]:
        operations = window_audit["operations"]
        for category in _audit_operation_categories():
            for operation in operations[category]:
                operation_decision_counts[operation["decision"]] += 1
                if category in {"mechanical_repairs", "contextual_repairs"}:
                    if operation["repair_type"] in ASR_AUDIT_REPAIR_TYPES:
                        asr_proposed += 1
                        if operation["decision"] == "APPROVE":
                            asr_approved += 1
                        asr_by_type.setdefault(
                            operation["repair_type"], {"proposed": 0, "approved": 0},
                        )["proposed"] += 1
                        if operation["decision"] == "APPROVE":
                            asr_by_type[operation["repair_type"]]["approved"] += 1
                        asr_by_confidence.setdefault(
                            operation["confidence"], {"proposed": 0, "approved": 0},
                        )["proposed"] += 1
                        if operation["decision"] == "APPROVE":
                            asr_by_confidence[operation["confidence"]]["approved"] += 1
                    if operation["decision"] == "REJECT" and (
                        operation["corrected_replacement"] is not None
                    ):
                        corrected_replacement_count += 1
                elif category == "entity_bindings":
                    if operation["human_resolvable_required"]:
                        entity_proposed += 1
                        entity_required += 1
                        if operation["decision"] == "APPROVE":
                            entity_approved += 1
                            entity_required_approved += 1
                elif category == "ability_bindings":
                    if operation["human_resolvable_required"]:
                        ability_proposed += 1
                        if operation["decision"] == "APPROVE":
                            ability_approved += 1
                elif category == "polished_statements":
                    if operation["decision"] == "APPROVE":
                        statement_approved += 1
                        for field in POLISH_STATEMENT_ATTESTATION_FIELDS:
                            if operation[field] is True:
                                preservation[field] += 1
        failure = window_audit["first_failure"]
        if failure is not None:
            first_failure_counts[failure["stage"]] = (
                first_failure_counts.get(failure["stage"], 0) + 1
            )
        if window_audit["first_reconstruction_failure"] is not None:
            reconstruction_failure_count += 1

    records = records_obj.get("records", [])
    unsupported_count = 0
    statement_count = 0
    for record in records:
        if record.get("record_type") != "D":
            continue
        content = record.get("content", {})
        if content.get("generation_status") != "GENERATED":
            continue
        counts = content.get("counts", {})
        unsupported_count += int(counts.get("unsupported_claim_count", 0))
        statement_count += int(counts.get("statement_count", 0))
        statement_count += int(counts.get("unsupported_claim_count", 0))
    unsupported_rate = _safe_float(
        unsupported_count / max(1, statement_count),
    ) if statement_count else 0.0
    precision = _safe_float(
        entity_approved / max(1, entity_proposed),
    )
    recall = _safe_float(
        entity_required_approved / max(1, entity_required),
    )
    ability_accuracy = _safe_float(
        ability_approved / max(1, ability_proposed),
    )
    preservation_rates = {
        field: _safe_float(preservation[field] / max(1, statement_approved))
        for field in POLISH_STATEMENT_ATTESTATION_FIELDS
    }
    return {
        "schema_version": TRANSFORMATION_SUMMARY_SCHEMA_VERSION,
        "asr": {
            "proposed": asr_proposed,
            "approved": asr_approved,
            "approval_rate": _safe_float(
                asr_approved / max(1, asr_proposed),
            ),
            "by_repair_type": dict(sorted(asr_by_type.items())),
            "by_confidence": dict(sorted(asr_by_confidence.items())),
        },
        "entity": {
            "proposed_resolvable": entity_proposed,
            "approved": entity_approved,
            "precision": precision,
            "required_resolvable": entity_required,
            "required_resolvable_approved": entity_required_approved,
            "required_resolvable_recall": recall,
        },
        "ability_ownership": {
            "proposed": ability_proposed,
            "approved": ability_approved,
            "accuracy": ability_accuracy,
        },
        "unsupported": {
            "count": unsupported_count,
            "statement_count": statement_count,
            "rate": unsupported_rate,
        },
        "polish_preservation": {
            **preservation_rates,
            "approved_statements": statement_approved,
        },
        "first_failures": {
            "by_stage": dict(sorted(first_failure_counts.items())),
            "reconstruction_count": reconstruction_failure_count,
        },
        "operation_decisions": operation_decision_counts,
        "corrected_replacement_count": corrected_replacement_count,
        "window_count": len(completed["window_audits"]),
    }


def build_phase2k_outputs(
    *,
    manifest_path: Path,
    packet_path: Path,
    db_path: Path,
    doc_path: Path | None,
    output_dir: Path,
    mode: str,
    chat: ChatCallable | None = None,
    cache_dir: Path | None = None,
    inference_config: Mapping[str, Any] | None = None,
    fail_if_exists: bool = True,
) -> dict[str, Any]:
    """Deterministic Phase 2K build; writes only into a fresh output dir."""
    if mode not in ("no_provider", "live"):
        raise ValueError("mode must be no_provider or live")
    if mode == "live" and chat is None:
        raise ValueError("live mode requires an injected chat function")
    if mode == "live" and inference_config is None:
        raise ValueError(
            "live mode requires a sealed secret-free inference config snapshot",
        )
    if mode == "no_provider":
        if inference_config is not None and inference_config_hash(
            inference_config,
        ) != inference_config_hash(NO_PROVIDER_INFERENCE_CONFIG):
            raise ValueError(
                "no-provider mode must use the explicit no-provider "
                "inference config snapshot",
            )
        inference_config = NO_PROVIDER_INFERENCE_CONFIG
    sealed_inference_config = validate_inference_config(
        inference_config,
        label="phase2k inference config",
    )
    sealed_inference_config_hash = inference_config_hash(sealed_inference_config)
    if fail_if_exists and output_dir.exists():
        raise ValueError(
            f"phase2k output directory already exists; refusing to overwrite: "
            f"{output_dir}",
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest, packet = validate_phase2j_frozen_inputs(manifest_path, packet_path)
    connection = open_transcript_db(db_path)
    transcripts: dict[str, dict[str, Any]] = {}
    segments_by_source: dict[str, list[dict[str, Any]]] = {}
    try:
        for selected in manifest["selected"]:
            source_id = selected["upstream_source_id"]
            transcript_info = validate_transcript_source(
                connection,
                source_id=source_id,
                game="lol",
                expected_full_sha256=selected["upstream_content_sha256"],
            )
            validate_target_slice(
                transcript_info["transcript"],
                target_start=selected["upstream_start"],
                target_end=selected["upstream_end"],
                bronze_text=selected["source_text"],
            )
            transcripts[source_id] = transcript_info
            segments_by_source[source_id] = build_segments(
                transcript_info["transcript"],
                selected["source_group_id"],
            )
    finally:
        connection.close()

    frozen_manifest = build_frozen_input_manifest(
        manifest_path=manifest_path,
        packet_path=packet_path,
        doc_path=doc_path,
        db_path=db_path,
        manifest=manifest,
        packet=packet,
        transcripts=transcripts,
    )
    config_hash = _config_hash()
    lineage = build_lineage()
    raw_response_dir = output_dir / "raw_responses" if mode == "live" else None
    records: list[dict[str, Any]] = []
    radius_entries: list[dict[str, Any]] = []
    diagnostic_summaries: list[dict[str, Any]] = []
    attempt_file_paths: list[str] = []
    window_failures: dict[str, dict[str, Any]] = {}

    if mode == "live":
        assert chat is not None
        for selected in manifest["selected"]:
            stage_label: str | None = None
            prompt_version: str | None = None
            response_schema_version: str | None = None
            transcript = transcripts[selected["upstream_source_id"]]["transcript"]
            segments = segments_by_source[selected["upstream_source_id"]]
            mechanical: Mapping[str, Any] | None = None
            final_context: Mapping[str, Any] | None = None
            reconstruction: Mapping[str, Any] | None = None
            polish: Mapping[str, Any] | None = None
            attempts: list[dict[str, Any]] = []
            try:
                stage_label = "mechanical_cleanup"
                prompt_version = MECHANICAL_PROMPT_VERSION
                response_schema_version = MECHANICAL_RESPONSE_SCHEMA_VERSION
                mechanical = run_mechanical_cleanup(
                    selected,
                    chat=chat,
                    config_hash=config_hash,
                    cache_dir=cache_dir,
                    inference_config=sealed_inference_config,
                    lineage=lineage,
                    raw_response_dir=raw_response_dir,
                )
                stage_label = "adaptive_diagnostics"
                prompt_version = SUFFICIENCY_PROMPT_VERSION
                response_schema_version = SUFFICIENCY_RESPONSE_SCHEMA_VERSION
                attempts, final_attempt = run_adaptive_diagnostics(
                    selected,
                    transcript=transcript,
                    mechanical_cleaned_text=mechanical["mechanical_cleaned_text"],
                    chat=chat,
                    config_hash=config_hash,
                    cache_dir=cache_dir,
                    segments=segments,
                    inference_config=sealed_inference_config,
                    lineage=lineage,
                    raw_response_dir=raw_response_dir,
                )
                attempt_file_paths.extend(_write_attempts(
                    output_dir, attempts, selected["window_id"],
                ))
                final_context = final_attempt["context"]
                stage_label = "reconstruction"
                prompt_version = RECONSTRUCTION_PROMPT_VERSION
                response_schema_version = RECONSTRUCTION_RESPONSE_SCHEMA_VERSION
                reconstruction = run_reconstruction(
                    selected,
                    transcript=transcript,
                    context=final_context,
                    mechanical_cleaned_text=mechanical["mechanical_cleaned_text"],
                    final_diagnostic=final_attempt,
                    chat=chat,
                    config_hash=config_hash,
                    cache_dir=cache_dir,
                    inference_config=sealed_inference_config,
                    lineage=lineage,
                    raw_response_dir=raw_response_dir,
                )
                stage_label = "semantic_polish"
                prompt_version = POLISH_PROMPT_VERSION
                response_schema_version = POLISH_RESPONSE_SCHEMA_VERSION
                polish = run_polish(
                    selected,
                    transcript=transcript,
                    context=final_context,
                    mechanical_cleaned_text=mechanical["mechanical_cleaned_text"],
                    reconstruction=reconstruction,
                    chat=chat,
                    config_hash=config_hash,
                    cache_dir=cache_dir,
                    inference_config=sealed_inference_config,
                    lineage=lineage,
                    raw_response_dir=raw_response_dir,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                attempt_history = (
                    exc.attempts
                    if isinstance(exc, ProviderCorrectionExhausted)
                    and getattr(exc, "attempts", None)
                    else []
                )
                failure = {
                    "window_id": selected["window_id"],
                    "status": "FAILED",
                    "stage": stage_label,
                    "prompt_version": prompt_version,
                    "response_schema_version": response_schema_version,
                    "error": error,
                    "attempt_count": len(attempt_history),
                    "attempts": attempt_history,
                    "pipeline_version": PIPELINE_VERSION,
                    "config_version": CONFIG_VERSION,
                    "inference_config": sealed_inference_config,
                    "inference_config_hash": sealed_inference_config_hash,
                    "inference_config_version": INFERENCE_CONFIG_VERSION,
                }
                _write_json_atomic(
                    _window_attempts_dir(output_dir, selected["window_id"])
                    / "failure.json",
                    failure,
                )
                window_failures[selected["window_id"]] = {
                    "status": "FAILED",
                    "stage": stage_label,
                    "prompt_version": prompt_version,
                    "response_schema_version": response_schema_version,
                    "error": error,
                    "attempt_count": len(attempt_history),
                }
                if final_context is None:
                    final_context = retrieve_context(
                        transcript,
                        source_group_id=selected["source_group_id"],
                        window_id=selected["window_id"],
                        target_start=selected["upstream_start"],
                        target_end=selected["upstream_end"],
                        bronze_text=selected["source_text"],
                        previous_segments=10,
                        following_segments=10,
                        radius_label="r10",
                        segments=segments,
                    )
                diagnostic_summaries.append({
                    "window_id": selected["window_id"],
                    "stages": [
                        {
                            "stage": attempt["stage"],
                            "radius": attempt["radius"],
                            "decision": attempt["decision"],
                        }
                        for attempt in attempts
                    ],
                    "stopping_stage": None,
                    "final_decision": None,
                    "at_max_context": False,
                    "attempt_count": len(attempts),
                    "failed_stage": stage_label,
                })
                records.append(build_record_a(selected))
                records.append(build_record_b(selected, mechanical))
                records.append(build_record_c(
                    selected,
                    final_context,
                    mechanical["mechanical_cleaned_text"]
                    if mechanical is not None else None,
                ))
                records.append(build_record_d(
                    selected,
                    mechanical,
                    reconstruction,
                    None,
                    failure={
                        "reason": "WINDOW_GENERATION_FAILED",
                        "stage": stage_label,
                        "note": error,
                        "attempts": attempt_history,
                    },
                ))
                radius_entries.extend(build_radius_entries(
                    selected, transcript, segments,
                ))
                continue
            diagnostic_summaries.append({
                "window_id": selected["window_id"],
                "stages": [
                    {
                        "stage": attempt["stage"],
                        "radius": attempt["radius"],
                        "decision": attempt["decision"],
                    }
                    for attempt in attempts
                ],
                "stopping_stage": final_attempt["stage"],
                "final_decision": final_attempt["decision"],
                "at_max_context": final_attempt["at_max_context"],
                "attempt_count": len(attempts),
            })
            records.append(build_record_a(selected))
            records.append(build_record_b(selected, mechanical))
            records.append(build_record_c(
                selected,
                final_context,
                mechanical["mechanical_cleaned_text"],
            ))
            records.append(build_record_d(
                selected, mechanical, reconstruction, polish,
            ))
            radius_entries.extend(build_radius_entries(
                selected, transcript, segments,
            ))
    else:
        for selected in manifest["selected"]:
            transcript = transcripts[selected["upstream_source_id"]]["transcript"]
            segments = segments_by_source[selected["upstream_source_id"]]
            default_context = retrieve_context(
                transcript,
                source_group_id=selected["source_group_id"],
                window_id=selected["window_id"],
                target_start=selected["upstream_start"],
                target_end=selected["upstream_end"],
                bronze_text=selected["source_text"],
                previous_segments=10,
                following_segments=10,
                radius_label="r10",
                segments=segments,
            )
            records.append(build_record_a(selected))
            records.append(build_record_b(selected, None))
            records.append(build_record_c(selected, default_context))
            records.append(build_record_d(selected, None, None, None))
            radius_entries.extend(build_radius_entries(
                selected, transcript, segments,
            ))
            diagnostic_summaries.append({
                "window_id": selected["window_id"],
                "stages": [],
                "stopping_stage": None,
                "final_decision": None,
                "at_max_context": False,
                "attempt_count": 0,
            })

    records.sort(key=lambda record: (record["window_id"], record["record_type"]))
    radius_entries.sort(key=lambda entry: (
        entry["window_id"], RADIUS_ENTRY_LABELS.index(entry["radius"]),
    ))
    records_obj = _envelope({
        "schema_version": RECORDS_SCHEMA_VERSION,
        "purpose": (
            "Phase 2K A/B/C/D human-review records plus exact context-radius "
            "entries for every frozen Phase 2J target."
        ),
        "release_gate": RELEASE_GATE_AWAITING_REVIEW,
        "pipeline_version": PIPELINE_VERSION,
        "config_version": CONFIG_VERSION,
        "config_hash": config_hash,
        "inference_config_version": INFERENCE_CONFIG_VERSION,
        "inference_config": sealed_inference_config,
        "inference_config_hash": sealed_inference_config_hash,
        "mode": mode,
        "lineage": lineage,
        "metadata_adapter_schema_version": METADATA_ADAPTER_SCHEMA_VERSION,
        "vocabulary_hash": lexical_vocabulary_hash(),
        "vocabulary_schema_version": LEAGUE_VOCABULARY_SCHEMA_VERSION,
        "frozen_input_manifest_sha256": frozen_manifest["content_sha256"],
        "records": records,
        "context_radius_entries": radius_entries,
        "diagnostic_summaries": diagnostic_summaries,
        "attempt_files": sorted(attempt_file_paths),
    })

    include_d = mode == "live"
    human_packet, human_mapping = build_human_review_packet(
        records,
        radius_entries,
        include_d=include_d,
    )
    transformation_audit: dict[str, Any] | None = None
    if mode == "live":
        transformation_audit = build_transformation_audit(
            records,
            window_failures,
            records_sha256=records_obj["content_sha256"],
        )
    build_summary = _envelope({
        "schema_version": BUILD_SUMMARY_SCHEMA_VERSION,
        "purpose": (
            "Phase 2K build summary; deterministic pipeline config and sealed "
            "secret-free inference config."
        ),
        "mode": mode,
        "pipeline_version": PIPELINE_VERSION,
        "config_version": CONFIG_VERSION,
        "config_hash": config_hash,
        "inference_config_version": INFERENCE_CONFIG_VERSION,
        "inference_config": sealed_inference_config,
        "inference_config_hash": sealed_inference_config_hash,
        "lineage": lineage,
        "metadata_adapter_schema_version": METADATA_ADAPTER_SCHEMA_VERSION,
        "vocabulary_hash": lexical_vocabulary_hash(),
        "vocabulary_schema_version": LEAGUE_VOCABULARY_SCHEMA_VERSION,
        "prompt_versions": {
            "mechanical_cleanup": MECHANICAL_PROMPT_VERSION,
            "sufficiency": SUFFICIENCY_PROMPT_VERSION,
            "reconstruction": RECONSTRUCTION_PROMPT_VERSION,
            "semantic_polish": POLISH_PROMPT_VERSION,
        },
        "correction_prompt_versions": {
            "mechanical_cleanup": MECHANICAL_CORRECTION_PROMPT_VERSION,
            "sufficiency": SUFFICIENCY_CORRECTION_PROMPT_VERSION,
            "reconstruction": RECONSTRUCTION_CORRECTION_PROMPT_VERSION,
            "semantic_polish": POLISH_CORRECTION_PROMPT_VERSION,
        },
        "response_schema_versions": {
            "mechanical_cleanup": MECHANICAL_RESPONSE_SCHEMA_VERSION,
            "sufficiency": SUFFICIENCY_RESPONSE_SCHEMA_VERSION,
            "reconstruction": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
            "semantic_polish": POLISH_RESPONSE_SCHEMA_VERSION,
        },
        "frozen_input_manifest_sha256": frozen_manifest["content_sha256"],
        "records_sha256": records_obj["content_sha256"],
        "human_packet_sha256": human_packet["content_sha256"],
        "human_mapping_sha256": human_mapping["content_sha256"],
        "transformation_audit_sha256": (
            transformation_audit["content_sha256"]
            if transformation_audit is not None
            else None
        ),
        "window_failure_count": len(window_failures),
        "raw_response_count": (
            len(list((output_dir / "raw_responses").glob("*.txt")))
            if mode == "live"
            else 0
        ),
        "transformation_audit_status": (
            "AWAITING_HUMAN_REVIEW" if mode == "live" else "NOT_APPLICABLE"
        ),
        "closeout_status": "WAITING_FOR_HUMAN_REVIEW",
        "review_gate": {
            "status": "AWAITING_HUMAN_REVIEW",
            "spec": dict(REVIEW_GATE_SPEC),
            "note": (
                "The deterministic gate is evaluated only after complete "
                "human reviews are imported."
            ),
        },
        "window_count": len(manifest["selected"]),
        "cache": {
            "enabled": cache_dir is not None,
            "dir": str(cache_dir) if cache_dir is not None else None,
        },
    })

    paths = {
        "output_dir": output_dir,
        "frozen_input_manifest": output_dir / OUTPUT_FILENAMES["frozen_input_manifest"],
        "records": output_dir / OUTPUT_FILENAMES["records"],
        "human_packet": output_dir / OUTPUT_FILENAMES["human_packet"],
        "human_mapping": output_dir / OUTPUT_FILENAMES["human_mapping"],
        "build_summary": output_dir / OUTPUT_FILENAMES["build_summary"],
        "transformation_audit": (
            output_dir / OUTPUT_FILENAMES["transformation_audit"]
            if mode == "live"
            else None
        ),
    }
    _write_json_atomic(paths["frozen_input_manifest"], frozen_manifest)
    _write_json_atomic(paths["records"], records_obj)
    _write_json_atomic(paths["human_packet"], human_packet)
    _write_json_atomic(paths["human_mapping"], human_mapping)
    _write_json_atomic(paths["build_summary"], build_summary)
    if transformation_audit is not None:
        _write_json_atomic(paths["transformation_audit"], transformation_audit)
    return {
        "output_dir": output_dir,
        "paths": paths,
        "summary": build_summary,
        "frozen_manifest_sha256": frozen_manifest["content_sha256"],
        "records_sha256": records_obj["content_sha256"],
        "human_packet_sha256": human_packet["content_sha256"],
        "human_mapping_sha256": human_mapping["content_sha256"],
        "transformation_audit_sha256": (
            transformation_audit["content_sha256"]
            if transformation_audit is not None
            else None
        ),
        "inference_config_hash": sealed_inference_config_hash,
        "mode": mode,
        "window_count": len(manifest["selected"]),
        "window_failure_count": len(window_failures),
        "raw_response_count": (
            len(list((output_dir / "raw_responses").glob("*.txt")))
            if mode == "live"
            else 0
        ),
    }


def _validate_model_call_inference_binding(
    model_call: object,
    *,
    expected_hash: str,
    expected_config: Mapping[str, Any],
    label: str,
) -> None:
    """Verify a model-call record agrees with the sealed top-level config."""
    if model_call is None:
        return
    if not isinstance(model_call, Mapping):
        raise ValueError(f"{label} model_call must be an object or null")
    if model_call.get("inference_config_hash") != expected_hash:
        raise ValueError(
            f"{label} inference config hash does not match the sealed "
            "top-level inference config",
        )
    recorded = model_call.get("inference_config")
    if recorded != expected_config:
        raise ValueError(
            f"{label} inference config snapshot does not match the sealed "
            "top-level inference config",
        )
    if inference_config_hash(recorded) != expected_hash:
        raise ValueError(f"{label} inference config hash is invalid")


def _collect_model_calls(value: object, *, path: str) -> list[tuple[dict[str, Any], str]]:
    """Recursively collect model-call records from validated artifacts."""
    calls: list[tuple[dict[str, Any], str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "model_call":
                if item is not None:
                    if not isinstance(item, Mapping):
                        raise ValueError(f"{path}.model_call must be an object")
                    calls.append((dict(item), f"{path}.model_call"))
                continue
            if key == "model_calls" and isinstance(item, Mapping):
                for sub_key, sub_item in item.items():
                    if not isinstance(sub_item, Mapping):
                        raise ValueError(
                            f"{path}.model_calls.{sub_key} must be an object",
                        )
                    calls.append((dict(sub_item), f"{path}.model_calls.{sub_key}"))
                continue
            calls.extend(_collect_model_calls(item, path=f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            calls.extend(_collect_model_calls(item, path=f"{path}[{index}]"))
    return calls


def _validate_raw_response_reference(
    model_call: Mapping[str, Any],
    *,
    output_dir: Path,
    label: str,
    referenced_raw_files: set[str],
) -> None:
    """Verify the content-addressed raw response file exists and hashes."""
    raw_hash = model_call.get("raw_response_sha256")
    raw_path_name = model_call.get("raw_response_path")
    if raw_hash is None or raw_path_name is None:
        raise ValueError(f"{label} is missing raw response linkage")
    if not isinstance(raw_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", raw_hash):
        raise ValueError(f"{label} raw response hash is malformed")
    if not isinstance(raw_path_name, str) or Path(raw_path_name).name != raw_path_name:
        raise ValueError(f"{label} raw response path is not a safe basename")
    if raw_path_name != f"{raw_hash}.txt":
        raise ValueError(f"{label} raw response filename does not match its hash")
    raw_file = output_dir / "raw_responses" / raw_path_name
    if not raw_file.is_file():
        raise ValueError(f"{label} raw response file is missing: {raw_file}")
    if file_sha256(raw_file) != raw_hash:
        raise ValueError(f"{label} raw response file is tampered")
    referenced_raw_files.add(raw_path_name)


def _validate_attempt_history(
    value: object,
    *,
    output_dir: Path,
    expected_hash: str,
    expected_config: Mapping[str, Any],
    label: str,
    referenced_raw_files: set[str],
) -> None:
    """Validate an ordered attempt history and every retry raw response."""
    attempts = _require_list(value, f"{label} attempts")
    for position, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            raise ValueError(f"{label} attempt {position} must be an object")
        attempt_index = attempt.get("attempt_index")
        if not isinstance(attempt_index, int) or attempt_index < 0:
            raise ValueError(
                f"{label} attempt {position} has an invalid attempt_index",
            )
        if attempt_index != position:
            raise ValueError(
                f"{label} attempt history is not ordered: attempt {position} "
                f"has attempt_index {attempt_index}",
            )
        attempt_label = f"{label} attempt {position}"
        model_call = attempt.get("model_call")
        if not isinstance(model_call, Mapping):
            raise ValueError(f"{attempt_label} is missing its model call")
        _validate_model_call_inference_binding(
            model_call,
            expected_hash=expected_hash,
            expected_config=expected_config,
            label=attempt_label,
        )
        _validate_raw_response_reference(
            model_call,
            output_dir=output_dir,
            label=attempt_label,
            referenced_raw_files=referenced_raw_files,
        )


def _validate_reconstruction_raw_compact(
    raw_compact: object,
    *,
    reconstruction: Mapping[str, Any],
    label: str,
) -> None:
    """Verify the retained raw compact reconstruction proposals are intact."""
    if not isinstance(raw_compact, Mapping):
        raise ValueError(f"{label} raw_compact must be an object")
    _require_exact_keys(
        raw_compact,
        (
            "schema_version", "clean_target_transcript", "contextual_repairs",
            "bindings", "unresolved_alternatives", "rationale",
        ),
        f"{label} raw_compact",
    )
    if raw_compact["schema_version"] != RECONSTRUCTION_RESPONSE_SCHEMA_VERSION:
        raise ValueError(f"{label} raw_compact schema version is invalid")
    if raw_compact["clean_target_transcript"] != reconstruction.get(
        "clean_target_transcript",
    ):
        raise ValueError(f"{label} raw_compact clean transcript is inconsistent")
    if raw_compact["rationale"] != reconstruction.get("rationale"):
        raise ValueError(f"{label} raw_compact rationale is inconsistent")
    for key in ("contextual_repairs", "bindings", "unresolved_alternatives"):
        if not isinstance(raw_compact.get(key), list) or len(
            raw_compact[key],
        ) != _raw_compact_expected_count(reconstruction, key):
            raise ValueError(
                f"{label} raw_compact {key} count is inconsistent",
            )


def _raw_compact_expected_count(
    reconstruction: Mapping[str, Any],
    key: str,
) -> int:
    """Expected raw_compact list count for one reconstruction key.

    Sentinel binding proposals (explicit NONE placeholders and NONE-candidate
    proposals carrying no resolution claim) and context-only proposals (an
    exact/surface-normalized context mention with no target-Bronze match) are
    preserved verbatim in ``raw_compact`` but omitted from the normalized
    semantic bindings, so the raw count equals the normalized count plus the
    documented omission count.
    """
    normalized = reconstruction.get(key, [])
    if not isinstance(normalized, list):
        return 0
    if key != "bindings":
        return len(normalized)
    return len(normalized) + int(
        reconstruction.get("omitted_binding_count", 0) or 0,
    )


def _validate_polish_raw_compact(
    raw_compact: object,
    *,
    polish: Mapping[str, Any],
    label: str,
) -> None:
    """Verify the retained raw compact polish proposals are intact."""
    if not isinstance(raw_compact, Mapping):
        raise ValueError(f"{label} raw_compact must be an object")
    _require_exact_keys(
        raw_compact,
        ("schema_version", "statements", "unsupported_claims", "rationale"),
        f"{label} raw_compact",
    )
    if raw_compact["schema_version"] != POLISH_RESPONSE_SCHEMA_VERSION:
        raise ValueError(f"{label} raw_compact schema version is invalid")
    if raw_compact["rationale"] != polish.get("rationale"):
        raise ValueError(f"{label} raw_compact rationale is inconsistent")
    if not isinstance(raw_compact.get("statements"), list) or len(
        raw_compact["statements"],
    ) != len(polish.get("statements", [])):
        raise ValueError(f"{label} raw_compact statement count is inconsistent")
    if not isinstance(raw_compact.get("unsupported_claims"), list) or len(
        raw_compact["unsupported_claims"],
    ) != len(polish.get("unsupported_claims", [])):
        raise ValueError(
            f"{label} raw_compact unsupported claim count is inconsistent",
        )


def _validate_raw_response_directory(
    *,
    output_dir: Path,
    mode: str,
    referenced_raw_files: set[str],
) -> None:
    raw_dir = output_dir / "raw_responses"
    if mode != "live":
        if raw_dir.exists():
            files = sorted(path.name for path in raw_dir.glob("*.txt"))
            if files:
                raise ValueError(
                    "no-provider output must not contain raw response files",
                )
        return
    if not raw_dir.is_dir():
        if referenced_raw_files:
            raise ValueError(
                "live phase2k output is missing the raw response directory "
                "for referenced model calls",
            )
        return
    on_disk = {path.name for path in raw_dir.glob("*.txt")}
    missing_reference = sorted(referenced_raw_files - on_disk)
    orphan = sorted(on_disk - referenced_raw_files)
    if missing_reference:
        raise ValueError(
            "live phase2k output has referenced raw responses that are "
            f"missing: {missing_reference}",
        )
    if orphan:
        raise ValueError(
            "live phase2k output has unreferenced orphan raw responses: "
            f"{orphan}",
        )


def _validate_optional_finalized_artifacts(
    *,
    output_dir: Path,
    mode: str,
    human_mapping: Mapping[str, Any],
    records_obj: Mapping[str, Any],
    transformation_audit: Mapping[str, Any] | None,
) -> None:
    """Validate finalized artifacts when present, fail-closed on partials."""
    finalized_path = output_dir / OUTPUT_FILENAMES["finalized_packet"]
    summary_path = output_dir / OUTPUT_FILENAMES["human_summary"]
    audit_path = output_dir / OUTPUT_FILENAMES["finalized_transformation_audit"]
    audit_summary_path = output_dir / OUTPUT_FILENAMES["transformation_summary"]
    closeout_path = output_dir / OUTPUT_FILENAMES["closeout_status"]

    has_review = finalized_path.exists() or summary_path.exists()
    if has_review and not (finalized_path.exists() and summary_path.exists()):
        raise ValueError("phase2k finalized human-review artifacts are partial")
    finalized: dict[str, Any] | None = None
    recorded_summary: dict[str, Any] | None = None
    if has_review:
        finalized = load_json_strict(
            finalized_path, label="phase2k finalized human packet",
        )
        validate_human_review_packet(finalized, require_blank=False)
        if finalized.get("blinding", {}).get("mapping_sha256") != (
            human_mapping.get("content_sha256")
        ):
            raise ValueError(
                "phase2k finalized packet is not bound to the human mapping",
            )
        recorded_summary = load_json_strict(
            summary_path, label="phase2k human review summary",
        )
        if recorded_summary.get("schema_version") != HUMAN_SUMMARY_SCHEMA_VERSION:
            raise ValueError("phase2k human review summary schema version is invalid")
        recomputed = summarize_human_reviews(
            finalized,
            mapping=human_mapping,
            records_file=records_obj,
        )
        if recorded_summary != recomputed:
            raise ValueError(
                "phase2k human review summary does not match its inputs",
            )

    has_audit = audit_path.exists() or audit_summary_path.exists()
    if has_audit and not (audit_path.exists() and audit_summary_path.exists()):
        raise ValueError("phase2k finalized transformation-audit artifacts are partial")
    completed: dict[str, Any] | None = None
    if has_audit:
        if mode != "live" or transformation_audit is None:
            raise ValueError(
                "phase2k finalized transformation audit requires a live build",
            )
        completed = load_json_strict(
            audit_path, label="phase2k finalized transformation audit",
        )
        validate_completed_transformation_audits(
            transformation_audit,
            completed,
            records_obj=records_obj,
        )
        recorded_metrics = load_json_strict(
            audit_summary_path, label="phase2k transformation summary",
        )
        if recorded_metrics.get("schema_version") != (
            TRANSFORMATION_SUMMARY_SCHEMA_VERSION
        ):
            raise ValueError(
                "phase2k transformation summary schema version is invalid",
            )
        recomputed_metrics = summarize_transformation_audits(
            completed, records_obj=records_obj,
        )
        if recorded_metrics != recomputed_metrics:
            raise ValueError(
                "phase2k transformation summary does not match its inputs",
            )

    if closeout_path.exists():
        closeout = load_json_strict(
            closeout_path, label="phase2k closeout status",
        )
        _require_exact_keys(
            closeout,
            (
                "schema_version",
                "status",
                "inputs_complete",
                "count_report",
                "downstream_comparison",
            ),
            "phase2k closeout status",
        )
        if closeout["schema_version"] != CLOSEOUT_STATUS_SCHEMA_VERSION:
            raise ValueError("phase2k closeout schema version is invalid")
        _require_enum(closeout["status"], CLOSEOUT_STATUSES, "closeout status")
        _require_bool(closeout["inputs_complete"], "closeout inputs_complete")
        if not closeout["inputs_complete"] and closeout["status"] not in (
            "WAITING_FOR_HUMAN_REVIEW", "WAITING_FOR_DOWNSTREAM",
        ):
            raise ValueError("incomplete closeout must have a waiting status")
        if closeout["inputs_complete"] and closeout["status"] not in (
            FINAL_CLOSEOUT_STATUSES
        ):
            raise ValueError("complete closeout must use a final status")
        if closeout["inputs_complete"]:
            embedded = closeout.get("downstream_comparison")
            if not isinstance(embedded, Mapping):
                raise ValueError(
                    "phase2k complete closeout must embed the validated "
                    "downstream comparison",
                )
            if finalized is None or recorded_summary is None:
                raise ValueError(
                    "phase2k complete closeout requires finalized "
                    "human-review artifacts",
                )
            validate_downstream_comparison(
                embedded,
                label="phase2k closeout downstream comparison",
                records_obj=records_obj,
                finalized_packet=finalized,
                human_summary=recorded_summary,
                completed_audit=completed if mode == "live" else None,
            )
            if embedded["decision"] != closeout["status"]:
                raise ValueError(
                    "phase2k closeout decision must match the downstream "
                    "comparison decision",
                )
        elif closeout.get("downstream_comparison") is not None:
            raise ValueError(
                "phase2k waiting closeout must not embed a downstream comparison",
            )
        report = closeout["count_report"]
        expected_report = build_count_report_skeleton()
        if not isinstance(report, Mapping) or set(report) != set(expected_report):
            raise ValueError("phase2k closeout count report keys are invalid")
        if report.get("schema_version") != COUNT_REPORT_SCHEMA_VERSION:
            raise ValueError("phase2k closeout count report schema is invalid")
        for key, expected_section in expected_report.items():
            if isinstance(expected_section, Mapping):
                if not isinstance(report.get(key), Mapping) or set(
                    report[key],
                ) != set(expected_section):
                    raise ValueError(
                        "phase2k closeout count report section keys are invalid",
                    )


def validate_output_directory(
    *,
    output_dir: Path,
    manifest_path: Path,
    packet_path: Path,
    db_path: Path,
) -> dict[str, Any]:
    """Deterministic no-provider validation of an existing Phase 2K output."""
    required = [
        output_dir / OUTPUT_FILENAMES["frozen_input_manifest"],
        output_dir / OUTPUT_FILENAMES["records"],
        output_dir / OUTPUT_FILENAMES["human_packet"],
        output_dir / OUTPUT_FILENAMES["human_mapping"],
        output_dir / OUTPUT_FILENAMES["build_summary"],
    ]
    if (output_dir / OUTPUT_FILENAMES["records"]).exists():
        peek_mode: str | None = None
        try:
            peek_mode = load_json_strict(
                output_dir / OUTPUT_FILENAMES["records"],
                label="phase2k records mode probe",
            ).get("mode")
        except ValueError:
            peek_mode = None
        if peek_mode == "live":
            required.append(
                output_dir / OUTPUT_FILENAMES["transformation_audit"],
            )
    for path in required:
        if not path.is_file():
            raise ValueError(f"phase2k output set is incomplete: {path}")
    manifest, packet = validate_phase2j_frozen_inputs(manifest_path, packet_path)
    frozen = load_json_strict(
        output_dir / OUTPUT_FILENAMES["frozen_input_manifest"],
        label="phase2k frozen input manifest",
    )
    records_obj = load_json_strict(
        output_dir / OUTPUT_FILENAMES["records"],
        label="phase2k records",
    )
    human_packet = load_json_strict(
        output_dir / OUTPUT_FILENAMES["human_packet"],
        label="phase2k human review packet",
    )
    human_mapping = load_json_strict(
        output_dir / OUTPUT_FILENAMES["human_mapping"],
        label="phase2k human review mapping",
    )
    build_summary = load_json_strict(
        output_dir / OUTPUT_FILENAMES["build_summary"],
        label="phase2k build summary",
    )
    transformation_audit: dict[str, Any] | None = None
    if (output_dir / OUTPUT_FILENAMES["transformation_audit"]).is_file():
        transformation_audit = load_json_strict(
            output_dir / OUTPUT_FILENAMES["transformation_audit"],
            label="phase2k transformation audit",
        )
    _validate_recomputed_content_hash(frozen, label="phase2k frozen input manifest")
    _validate_recomputed_content_hash(records_obj, label="phase2k records")
    _validate_recomputed_content_hash(human_packet, label="phase2k human packet")
    _validate_recomputed_content_hash(human_mapping, label="phase2k human mapping")
    _validate_recomputed_content_hash(build_summary, label="phase2k build summary")
    if transformation_audit is not None:
        _validate_recomputed_content_hash(
            transformation_audit, label="phase2k transformation audit",
        )
    validate_human_review_packet(human_packet, require_blank=True)

    mode = records_obj.get("mode")
    if mode not in ("no_provider", "live"):
        raise ValueError("phase2k records mode is invalid")
    if mode == "live":
        if transformation_audit is None:
            raise ValueError(
                "live phase2k output is missing the transformation audit",
            )
        validate_transformation_audit_packet(
            transformation_audit, records_obj=records_obj,
        )
        if build_summary.get("transformation_audit_sha256") != (
            transformation_audit["content_sha256"]
        ):
            raise ValueError(
                "build summary is not bound to the transformation audit",
            )
        if build_summary.get("transformation_audit_status") != (
            "AWAITING_HUMAN_REVIEW"
        ):
            raise ValueError(
                "build summary transformation audit status is invalid",
            )
        if build_summary.get("closeout_status") != "WAITING_FOR_HUMAN_REVIEW":
            raise ValueError("build summary closeout status is invalid")
    elif transformation_audit is not None:
        raise ValueError(
            "no-provider phase2k output must not contain a transformation audit",
        )
    else:
        if build_summary.get("transformation_audit_sha256") is not None:
            raise ValueError(
                "no-provider build summary must not reference an audit",
            )
    referenced_raw_files: set[str] = set()
    top_inference_config = records_obj.get("inference_config")
    top_inference_hash = records_obj.get("inference_config_hash")
    if not isinstance(top_inference_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", top_inference_hash,
    ):
        raise ValueError(
            "phase2k records inference config hash is missing or malformed",
        )
    if top_inference_config is None:
        raise ValueError("phase2k records inference config snapshot is missing")
    if inference_config_hash(top_inference_config) != top_inference_hash:
        raise ValueError(
            "phase2k records inference config hash does not match its snapshot",
        )
    if records_obj.get("inference_config_version") != INFERENCE_CONFIG_VERSION:
        raise ValueError("phase2k records inference config version is invalid")
    if build_summary.get("inference_config_hash") != top_inference_hash:
        raise ValueError(
            "build summary inference config hash does not match records",
        )
    if build_summary.get("inference_config") != top_inference_config:
        raise ValueError(
            "build summary inference config snapshot does not match records",
        )
    if build_summary.get("inference_config_version") != INFERENCE_CONFIG_VERSION:
        raise ValueError("build summary inference config version is invalid")
    if build_summary.get("config_hash") != records_obj.get("config_hash"):
        raise ValueError(
            "build summary pipeline config hash does not match records",
        )
    if build_summary.get("lineage") != records_obj.get("lineage"):
        raise ValueError(
            "build summary lineage does not match records lineage",
        )
    if records_obj.get("metadata_adapter_schema_version") != (
        METADATA_ADAPTER_SCHEMA_VERSION
    ):
        raise ValueError("phase2k records metadata adapter schema version is invalid")
    if records_obj.get("vocabulary_schema_version") != (
        LEAGUE_VOCABULARY_SCHEMA_VERSION
    ):
        raise ValueError("phase2k records vocabulary schema version is invalid")
    if build_summary.get("metadata_adapter_schema_version") != (
        METADATA_ADAPTER_SCHEMA_VERSION
    ):
        raise ValueError(
            "build summary metadata adapter schema version is invalid",
        )
    if build_summary.get("vocabulary_schema_version") != (
        LEAGUE_VOCABULARY_SCHEMA_VERSION
    ):
        raise ValueError("build summary vocabulary schema version is invalid")
    if records_obj.get("vocabulary_hash") != lexical_vocabulary_hash():
        raise ValueError(
            "phase2k records vocabulary hash does not match the current "
            "lexical vocabulary snapshot",
        )
    if build_summary.get("vocabulary_hash") != lexical_vocabulary_hash():
        raise ValueError(
            "build summary vocabulary hash does not match the current "
            "lexical vocabulary snapshot",
        )
    lineage = records_obj.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("phase2k records lineage is missing")
    recorded_vocab = lineage.get("vocabulary")
    current_vocab = vocabulary_lineage()
    if not isinstance(recorded_vocab, Mapping):
        raise ValueError("phase2k records vocabulary lineage is missing")
    for key in ("path", "file_sha256", "content_sha256", "schema_version"):
        if recorded_vocab.get(key) != current_vocab[key]:
            raise ValueError(
                f"phase2k records vocabulary lineage {key} is inconsistent",
            )
    repo = lineage.get("repo")
    if not isinstance(repo, Mapping) or "repo_dirty" not in repo:
        raise ValueError("phase2k records repo lineage is invalid")
    if mode == "no_provider":
        if top_inference_config != NO_PROVIDER_INFERENCE_CONFIG:
            raise ValueError(
                "no-provider output must seal the explicit no-provider "
                "inference config snapshot",
            )
        if top_inference_hash != inference_config_hash(
            NO_PROVIDER_INFERENCE_CONFIG,
        ):
            raise ValueError("no-provider inference config hash is invalid")

    recorded_manifest_path = frozen["phase2j_inputs"]["manifest"]["path"]
    recorded_packet_path = frozen["phase2j_inputs"]["reviewed_packet"]["path"]
    recorded_db_path = frozen["transcript_db"]["path"]
    if recorded_manifest_path != normalize_path_locator(manifest_path):
        raise ValueError(
            "frozen manifest input locator is inconsistent with the supplied "
            f"manifest path: recorded={recorded_manifest_path!r} "
            f"supplied={normalize_path_locator(manifest_path)!r}",
        )
    if recorded_packet_path != normalize_path_locator(packet_path):
        raise ValueError(
            "frozen manifest input locator is inconsistent with the supplied "
            f"packet path: recorded={recorded_packet_path!r} "
            f"supplied={normalize_path_locator(packet_path)!r}",
        )
    if recorded_db_path != normalize_path_locator(db_path):
        raise ValueError(
            "frozen manifest transcript-DB locator is inconsistent with the "
            f"supplied DB path: recorded={recorded_db_path!r} "
            f"supplied={normalize_path_locator(db_path)!r}",
        )

    connection = open_transcript_db(db_path)
    try:
        by_window: dict[str, dict[str, Any]] = {}
        for selected in manifest["selected"]:
            window_id = selected["window_id"]
            transcript_info = validate_transcript_source(
                connection,
                source_id=selected["upstream_source_id"],
                game="lol",
                expected_full_sha256=selected["upstream_content_sha256"],
            )
            transcript = transcript_info["transcript"]
            validate_target_slice(
                transcript,
                target_start=selected["upstream_start"],
                target_end=selected["upstream_end"],
                bronze_text=selected["source_text"],
            )
            by_window[window_id] = {
                "selected": selected,
                "transcript": transcript,
            }
            expected_target = {
                "window_id": window_id,
                "source_absolute_start": selected["upstream_start"],
                "source_absolute_end": selected["upstream_end"],
                "text": selected["source_text"],
            }
            window_records = [
                record
                for record in records_obj["records"]
                if record["window_id"] == window_id
            ]
            record_types = {record["record_type"] for record in window_records}
            if not {"A", "B", "C"}.issubset(record_types) \
                    or not record_types.issubset({"A", "B", "C", "D"}):
                raise ValueError(
                    f"phase2k records are incomplete for window {window_id}",
                )
            by_type = {record["record_type"]: record for record in window_records}
            for record in window_records:
                content = record.get("content")
                if not isinstance(content, Mapping):
                    continue
                if content.get("model_call") is not None:
                    _validate_model_call_inference_binding(
                        content["model_call"],
                        expected_hash=top_inference_hash,
                        expected_config=top_inference_config,
                        label=(
                            f"phase2k {record['record_type']} record for "
                            f"{window_id}"
                        ),
                    )
                    _validate_raw_response_reference(
                        content["model_call"],
                        output_dir=output_dir,
                        label=(
                            f"phase2k {record['record_type']} record for "
                            f"{window_id}"
                        ),
                        referenced_raw_files=referenced_raw_files,
                    )
                reconstruction_subobject = content.get("reconstruction")
                if isinstance(reconstruction_subobject, Mapping) and (
                    reconstruction_subobject.get("model_call") is not None
                ):
                    _validate_model_call_inference_binding(
                        reconstruction_subobject["model_call"],
                        expected_hash=top_inference_hash,
                        expected_config=top_inference_config,
                        label=(
                            f"phase2k D reconstruction for {window_id}"
                        ),
                    )
                    _validate_raw_response_reference(
                        reconstruction_subobject["model_call"],
                        output_dir=output_dir,
                        label=f"phase2k D reconstruction for {window_id}",
                        referenced_raw_files=referenced_raw_files,
                    )
                polish_subobject = content.get("semantic_polish")
                if isinstance(polish_subobject, Mapping) and (
                    polish_subobject.get("model_call") is not None
                ):
                    _validate_model_call_inference_binding(
                        polish_subobject["model_call"],
                        expected_hash=top_inference_hash,
                        expected_config=top_inference_config,
                        label=f"phase2k D semantic polish for {window_id}",
                    )
                    _validate_raw_response_reference(
                        polish_subobject["model_call"],
                        output_dir=output_dir,
                        label=f"phase2k D semantic polish for {window_id}",
                        referenced_raw_files=referenced_raw_files,
                    )
            for record_type, record in by_type.items():
                target = record["target"]
                for key, value in expected_target.items():
                    if target.get(key) != value:
                        raise ValueError(
                            f"phase2k {record_type} target identity is not "
                            "invariant for " + window_id,
                        )
                if target["text_sha256"] != text_sha256(target["text"]):
                    raise ValueError(f"phase2k {record_type} target hash is invalid")
                if record["canonical_record_sha256"] != canonical_sha256({
                    key: value for key, value in record.items()
                    if key != "canonical_record_sha256"
                }):
                    raise ValueError(
                        f"phase2k {record_type} canonical hash is invalid",
                    )
            if by_type["A"]["content"]["text"] != selected["source_text"]:
                raise ValueError("phase2k A record must be the exact raw Bronze")
            b_content = by_type["B"]["content"]
            cleaned = b_content["text"]
            expected_cleaned = apply_mechanical_repairs(
                selected["source_text"],
                b_content["repairs"],
            )
            if cleaned != expected_cleaned:
                raise ValueError("phase2k B record is not the deterministic edit application")
            if b_content.get("clean_text") != cleaned:
                raise ValueError("phase2k B clean_text must equal the applied text")
            if b_content.get("repair_count") != len(b_content["repairs"]):
                raise ValueError("phase2k B repair_count is inconsistent")
            if b_content.get("uncertainty_count") != len(
                b_content.get("uncertainties", []),
            ):
                raise ValueError("phase2k B uncertainty_count is inconsistent")
            expected_hints = detect_champion_alias_hints(
                selected["source_text"], selected,
            )
            if b_content.get("lexical_hints") != expected_hints:
                raise ValueError(
                    "phase2k B lexical_hints are inconsistent with the "
                    "frozen target",
                )
            if b_content.get("lexical_hint_count") != len(expected_hints):
                raise ValueError(
                    "phase2k B lexical_hint_count is inconsistent",
                )
            raw_proposals = b_content.get("raw_proposals")
            if b_content.get("generation_status") == "GENERATED":
                _validate_eligible_hint_repairs(
                    b_content["repairs"],
                    expected_hints,
                    label="phase2k B",
                )
                _validate_mechanical_uncertainties(
                    b_content["uncertainties"],
                    bronze_text=selected["source_text"],
                    selected=selected,
                )
                validated_provenance = _validate_mechanical_provenance(
                    b_content["provenance"],
                    selected=selected,
                )
                _validate_metadata_adapter(
                    validated_provenance["input_metadata"],
                    selected=selected,
                    label="phase2k B provenance",
                )
                if b_content.get("model_call") is None:
                    raise ValueError("phase2k GENERATED B record is missing its model call")
                attempts = b_content.get("attempts")
                if not isinstance(attempts, list) or not attempts:
                    raise ValueError(
                        "phase2k GENERATED B record is missing its ordered "
                        "attempt history",
                    )
                _validate_attempt_history(
                    attempts,
                    output_dir=output_dir,
                    expected_hash=top_inference_hash,
                    expected_config=top_inference_config,
                    label=f"phase2k B record for {window_id}",
                    referenced_raw_files=referenced_raw_files,
                )
                if attempts[-1].get("status") != "OK":
                    raise ValueError(
                        "phase2k GENERATED B record final attempt is not OK",
                    )
                if attempts[-1].get("model_call") != b_content.get("model_call"):
                    raise ValueError(
                        "phase2k B record model_call must be the final "
                        "successful attempt",
                    )
                if not isinstance(raw_proposals, Mapping):
                    raise ValueError(
                        "phase2k GENERATED B record is missing its raw "
                        "proposal audit trail",
                    )
                if raw_proposals.get("clean_text") != cleaned:
                    raise ValueError(
                        "phase2k B raw proposal clean_text is inconsistent",
                    )
                if len(raw_proposals.get("repairs", [])) != b_content.get(
                    "repair_count",
                ):
                    raise ValueError(
                        "phase2k B raw proposal repair count is inconsistent",
                    )
                if len(raw_proposals.get("uncertainties", [])) != b_content.get(
                    "uncertainty_count",
                ):
                    raise ValueError(
                        "phase2k B raw proposal uncertainty count is "
                        "inconsistent",
                    )
            else:
                if b_content.get("provenance") is not None:
                    raise ValueError("phase2k NOT_GENERATED B must not carry provenance")
                if b_content.get("uncertainties"):
                    raise ValueError(
                        "phase2k NOT_GENERATED B must not carry uncertainties",
                    )
                if raw_proposals is not None:
                    raise ValueError(
                        "phase2k NOT_GENERATED B must not carry raw proposals",
                    )
                if b_content.get("attempts"):
                    raise ValueError(
                        "phase2k NOT_GENERATED B must not carry attempts",
                    )
            validate_context(by_type["C"]["content"]["context"], transcript)
            c_content = by_type["C"]["content"]
            presentation_target = c_content.get("presentation_target")
            if not isinstance(presentation_target, Mapping):
                raise ValueError("phase2k C record is missing its presentation target")
            expected_c_text = (
                b_content["clean_text"]
                if b_content.get("generation_status") == "GENERATED"
                else selected["source_text"]
            )
            if presentation_target.get("text") != expected_c_text:
                raise ValueError(
                    "phase2k C presentation target must be the mechanical "
                    "target where B exists",
                )
            if presentation_target.get("text_sha256") != text_sha256(
                expected_c_text,
            ):
                raise ValueError("phase2k C presentation target hash is invalid")
            if presentation_target.get("bronze_target_sha256") != text_sha256(
                selected["source_text"],
            ):
                raise ValueError(
                    "phase2k C record must retain the exact Bronze target hash",
                )
            d_content = by_type["D"]["content"]
            if d_content["generation_status"] == "NOT_GENERATED":
                if d_content["contextual_repairs"]:
                    raise ValueError(
                        "phase2k NOT_GENERATED D must not carry contextual repairs",
                    )
                if d_content.get("bindings") or d_content.get(
                    "unresolved_alternatives",
                ):
                    raise ValueError(
                        "phase2k NOT_GENERATED D must not carry bindings",
                    )
                if not d_content.get("is_placeholder"):
                    raise ValueError(
                        "phase2k NOT_GENERATED D must be marked as a placeholder",
                    )
                failure = d_content.get("failure")
                if not isinstance(failure, Mapping):
                    raise ValueError(
                        "phase2k NOT_GENERATED D is missing its failure record",
                    )
                _require_exact_keys(
                    failure,
                    ("reason", "stage", "note", "attempt_count", "attempts"),
                    f"phase2k NOT_GENERATED D failure for {window_id}",
                )
                if failure["stage"] is None:
                    if failure.get("attempts") or failure.get("attempt_count"):
                        raise ValueError(
                            "phase2k no-provider D failure must not carry "
                            "provider attempts",
                        )
                else:
                    _require_enum(
                        failure["stage"],
                        WINDOW_GENERATION_STAGES,
                        f"phase2k D failure stage for {window_id}",
                    )
                    failure_attempts = failure.get("attempts")
                    if not isinstance(failure_attempts, list):
                        raise ValueError(
                            "phase2k D failure attempts must be a list",
                        )
                    if failure_attempts:
                        _validate_attempt_history(
                            failure_attempts,
                            output_dir=output_dir,
                            expected_hash=top_inference_hash,
                            expected_config=top_inference_config,
                            label=f"phase2k D failure for {window_id}",
                            referenced_raw_files=referenced_raw_files,
                        )
                    if failure.get("attempt_count") != len(failure_attempts):
                        raise ValueError(
                            "phase2k D failure attempt_count is inconsistent",
                        )
                reconstruction_subobject = d_content.get("reconstruction")
                if d_content.get("semantic_polish") is not None:
                    raise ValueError(
                        "phase2k NOT_GENERATED D must not carry a polish "
                        "subobject",
                    )
                if reconstruction_subobject is not None:
                    if failure["stage"] != "semantic_polish":
                        raise ValueError(
                            "phase2k partial D may retain reconstruction only "
                            "when semantic polish failed",
                        )
                    if reconstruction_subobject.get("generation_status") != (
                        "GENERATED"
                    ):
                        raise ValueError(
                            "phase2k partial D reconstruction subobject is "
                            "not generated",
                        )
                    sub_model_call = reconstruction_subobject.get("model_call")
                    if not isinstance(sub_model_call, Mapping):
                        raise ValueError(
                            "phase2k partial D reconstruction is missing its "
                            "model call",
                        )
                    _validate_model_call_inference_binding(
                        sub_model_call,
                        expected_hash=top_inference_hash,
                        expected_config=top_inference_config,
                        label=f"phase2k partial D reconstruction for {window_id}",
                    )
                    _validate_raw_response_reference(
                        sub_model_call,
                        output_dir=output_dir,
                        label=f"phase2k partial D reconstruction for {window_id}",
                        referenced_raw_files=referenced_raw_files,
                    )
                    sub_attempts = reconstruction_subobject.get("attempts")
                    if not isinstance(sub_attempts, list) or not sub_attempts:
                        raise ValueError(
                            "phase2k partial D reconstruction is missing its "
                            "attempt history",
                        )
                    _validate_attempt_history(
                        sub_attempts,
                        output_dir=output_dir,
                        expected_hash=top_inference_hash,
                        expected_config=top_inference_config,
                        label=f"phase2k partial D reconstruction for {window_id}",
                        referenced_raw_files=referenced_raw_files,
                    )
                    if sub_attempts[-1].get("model_call") != sub_model_call:
                        raise ValueError(
                            "phase2k partial D reconstruction model_call must "
                            "be its final attempt",
                        )
                    _validate_reconstruction_raw_compact(
                        reconstruction_subobject.get("raw_compact"),
                        reconstruction=reconstruction_subobject,
                        label=f"phase2k partial D reconstruction for {window_id}",
                    )
                    if d_content.get("model_calls") != {
                        "reconstruction": sub_model_call,
                    }:
                        raise ValueError(
                            "phase2k partial D model_calls is inconsistent",
                        )
                    if d_content.get("model_call") != sub_model_call:
                        raise ValueError(
                            "phase2k partial D model_call must reference the "
                            "sealed reconstruction call",
                        )
                else:
                    if d_content.get("model_calls") != {}:
                        raise ValueError(
                            "phase2k bare placeholder D model_calls must be empty",
                        )
                    if d_content.get("model_call") is not None:
                        raise ValueError(
                            "phase2k bare placeholder D must not carry a "
                            "top-level model call",
                        )
            else:
                _validate_generated_record_d(
                    by_type["D"],
                    context_record=by_type["C"],
                    transcript=transcript,
                    selected=selected,
                    attempts_dir=_window_attempts_dir(output_dir, window_id),
                )
                reconstruction_subobject = d_content["reconstruction"]
                polish_subobject = d_content["semantic_polish"]
                for label, subobject in (
                    ("reconstruction", reconstruction_subobject),
                    ("semantic_polish", polish_subobject),
                ):
                    sub_attempts = subobject.get("attempts")
                    if not isinstance(sub_attempts, list) or not sub_attempts:
                        raise ValueError(
                            f"phase2k generated D {label} for {window_id} is "
                            "missing its attempt history",
                        )
                    _validate_attempt_history(
                        sub_attempts,
                        output_dir=output_dir,
                        expected_hash=top_inference_hash,
                        expected_config=top_inference_config,
                        label=f"phase2k D {label} for {window_id}",
                        referenced_raw_files=referenced_raw_files,
                    )
                    if sub_attempts[-1].get("model_call") != subobject.get(
                        "model_call",
                    ):
                        raise ValueError(
                            f"phase2k D {label} model_call must be its final "
                            "successful attempt",
                        )
                    if label == "reconstruction":
                        _validate_reconstruction_raw_compact(
                            subobject.get("raw_compact"),
                            reconstruction=subobject,
                            label=f"phase2k D reconstruction for {window_id}",
                        )
                    else:
                        _validate_polish_raw_compact(
                            subobject.get("raw_compact"),
                            polish=subobject,
                            label=f"phase2k D semantic polish for {window_id}",
                        )
            for entry in records_obj["context_radius_entries"]:
                if entry["window_id"] == window_id:
                    validate_context(entry["context"], transcript)
                    if entry["canonical_entry_sha256"] != canonical_sha256({
                        key: value for key, value in entry.items()
                        if key != "canonical_entry_sha256"
                    }):
                        raise ValueError("phase2k radius entry hash is invalid")
    finally:
        connection.close()
    if mode == "live":
        attempts_root = output_dir / "attempts"
        if not attempts_root.is_dir():
            raise ValueError(
                "live phase2k output is missing the attempts directory",
            )
        for path in sorted(attempts_root.rglob("*.json")):
            attempt = load_json_strict(path, label="phase2k attempt artifact")
            if attempt.get("status") == "FAILED":
                if attempt.get("inference_config_hash") != top_inference_hash:
                    raise ValueError(
                        "phase2k failure artifact inference config hash does "
                        "not match the sealed top-level inference config",
                    )
                if attempt.get("inference_config") != top_inference_config:
                    raise ValueError(
                        "phase2k failure artifact inference config snapshot "
                        "does not match the sealed top-level inference config",
                    )
                if inference_config_hash(
                    attempt.get("inference_config"),
                ) != top_inference_hash:
                    raise ValueError(
                        "phase2k failure artifact inference config hash is "
                        "invalid",
                    )
                if "attempts" in attempt:
                    _validate_attempt_history(
                        attempt["attempts"],
                        output_dir=output_dir,
                        expected_hash=top_inference_hash,
                        expected_config=top_inference_config,
                        label=f"phase2k failure artifact {path.name}",
                        referenced_raw_files=referenced_raw_files,
                    )
                elif attempt.get("model_call") is not None:
                    _validate_model_call_inference_binding(
                        attempt["model_call"],
                        expected_hash=top_inference_hash,
                        expected_config=top_inference_config,
                        label=f"phase2k failure artifact {path.name}",
                    )
                    _validate_raw_response_reference(
                        attempt["model_call"],
                        output_dir=output_dir,
                        label=f"phase2k failure artifact {path.name}",
                        referenced_raw_files=referenced_raw_files,
                    )
            else:
                _validate_model_call_inference_binding(
                    attempt.get("model_call"),
                    expected_hash=top_inference_hash,
                    expected_config=top_inference_config,
                    label=f"phase2k attempt artifact {path.name}",
                )
                _validate_raw_response_reference(
                    attempt["model_call"],
                    output_dir=output_dir,
                    label=f"phase2k attempt artifact {path.name}",
                    referenced_raw_files=referenced_raw_files,
                )
    _validate_raw_response_directory(
        output_dir=output_dir,
        mode=mode,
        referenced_raw_files=referenced_raw_files,
    )
    expected_raw_count = (
        len(list((output_dir / "raw_responses").glob("*.txt")))
        if mode == "live"
        else 0
    )
    if build_summary.get("raw_response_count") != expected_raw_count:
        raise ValueError("build summary raw response count is inconsistent")
    mapping_labels = human_mapping["labels"]
    item_ids = {item["review_item_id"] for item in human_packet["review_items"]}
    if human_mapping.get("content_sha256") != human_packet.get("blinding", {}).get(
        "mapping_sha256",
    ):
        raise ValueError("human review mapping is not bound to the packet")
    for item in human_packet["review_items"]:
        label = item["blinded_label"]
        mapped = mapping_labels.get(label)
        if mapped is None:
            raise ValueError("human review label is missing from the mapping")
        if mapped["window_id"] != item["window_id"]:
            raise ValueError("human review mapping is inconsistent with the packet")
        if mapped["target_text_sha256"] != item["presentation"]["target_sha256"]:
            raise ValueError(
                "human review mapping target identity is inconsistent with "
                "the packet presentation",
            )
        if mapped["presentation_sha256"] != canonical_sha256(
            item["presentation"],
        ):
            raise ValueError(
                "human review mapping presentation hash is inconsistent with "
                "the packet item",
            )
        if not (
            mapped.get("record_sha256") == item["content_sha256"]
            or mapped.get("entry_sha256") == item["content_sha256"]
        ):
            raise ValueError(
                "human review mapping record/entry hash is inconsistent with "
                "the packet item",
            )
    if len(item_ids) != len(human_packet["review_items"]):
        raise ValueError("human review item IDs are not unique")
    if build_summary["records_sha256"] != records_obj["content_sha256"]:
        raise ValueError("build summary is not bound to the records file")
    _validate_optional_finalized_artifacts(
        output_dir=output_dir,
        mode=mode,
        human_mapping=human_mapping,
        records_obj=records_obj,
        transformation_audit=transformation_audit,
    )
    return {
        "output_dir": output_dir,
        "frozen_input_manifest_sha256": frozen["content_sha256"],
        "records_sha256": records_obj["content_sha256"],
        "human_packet_sha256": human_packet["content_sha256"],
        "human_mapping_sha256": human_mapping["content_sha256"],
        "inference_config_hash": top_inference_hash,
        "window_count": len(manifest["selected"]),
        "mode": records_obj["mode"],
        "window_failure_count": build_summary.get("window_failure_count"),
        "raw_response_count": expected_raw_count,
    }


def _validate_generated_record_d(
    record: Mapping[str, Any],
    *,
    context_record: Mapping[str, Any],
    transcript: str,
    selected: Mapping[str, Any],
    attempts_dir: Path,
) -> None:
    """Deep-validate a GENERATED D record and both sealed subobjects."""
    content = record["content"]
    for forbidden in (
        "resolved_semantic_paraphrase",
        "paraphrase_text",
        "semantic_claims",
    ):
        if forbidden in content:
            raise ValueError(
                "GENERATED phase2k D record must not contain the retired "
                f"combined semantic field {forbidden!r}",
            )
    reconstruction = content.get("reconstruction")
    polish = content.get("semantic_polish")
    if not isinstance(reconstruction, Mapping) or not isinstance(polish, Mapping):
        raise ValueError("GENERATED phase2k D record is missing a pass subobject")
    if content.get("generation_status") != reconstruction.get("generation_status"):
        raise ValueError(
            "phase2k D generation status must match the reconstruction pass",
        )
    if reconstruction.get("generation_status") != "GENERATED":
        raise ValueError("phase2k D reconstruction subobject is not generated")
    if polish.get("generation_status") != "GENERATED":
        raise ValueError("phase2k D polish subobject is not generated")
    if content.get("clean_target_transcript") != reconstruction.get(
        "clean_target_transcript"
    ):
        raise ValueError("phase2k D clean transcript must match its reconstruction")
    for key in ("contextual_repairs", "bindings", "unresolved_alternatives"):
        if content.get(key) != reconstruction.get(key):
            raise ValueError(
                f"phase2k D {key} must match its reconstruction subobject",
            )
    expected_model_calls = {
        "reconstruction": reconstruction.get("model_call"),
        "semantic_polish": polish.get("model_call"),
    }
    if content.get("model_calls") != expected_model_calls:
        raise ValueError("phase2k D model_calls must carry both separate calls")
    if content.get("model_call") != reconstruction.get("model_call"):
        raise ValueError(
            "phase2k D top-level model_call must reference the sealed "
            "reconstruction call",
        )
    attempt_files = sorted(attempts_dir.glob("*.json"))
    final_attempt = None
    for path in attempt_files:
        if path.name == "failure.json":
            continue
        attempt = load_json_strict(path, label="phase2k diagnostic attempt")
        if attempt.get("decision") in ("SUFFICIENT", "MAX_CONTEXT_BUT_UNRESOLVED"):
            final_attempt = attempt
    if final_attempt is None:
        raise ValueError(
            "GENERATED phase2k D record has no terminal diagnostic attempt",
        )
    parsed_reconstruction = {
        "schema_version": RECONSTRUCTION_RESPONSE_SCHEMA_VERSION,
        "clean_target_transcript": reconstruction["clean_target_transcript"],
        "contextual_repairs": reconstruction["contextual_repairs"],
        "bindings": reconstruction["bindings"],
        "unresolved_alternatives": reconstruction["unresolved_alternatives"],
        "provenance": reconstruction["provenance"],
    }
    validated_reconstruction = validate_reconstruction_response(
        parsed_reconstruction,
        bronze_text=selected["source_text"],
        base_offset=selected["upstream_start"],
        transcript=transcript,
        context=context_record["content"]["context"],
        final_diagnostic=final_attempt,
        metadata=build_metadata_adapter(selected),
        selected=selected,
    )
    if reconstruction.get("rationale") != validated_reconstruction["provenance"][
        "rationale"
    ]:
        raise ValueError(
            "phase2k D reconstruction rationale must match its sealed "
            "provenance",
        )
    if reconstruction["counts"].get("metadata_conflict_count") != len(
        final_attempt["response"]["parsed"]["metadata_conflicts"]
    ):
        raise ValueError(
            "phase2k D metadata conflicts must match the final diagnostic",
        )
    parsed_polish = {
        "schema_version": POLISH_RESPONSE_SCHEMA_VERSION,
        "statements": polish["statements"],
        "unsupported_claims": polish["unsupported_claims"],
        "rationale": polish["rationale"],
    }
    validate_polish_response(
        parsed_polish,
        bronze_text=selected["source_text"],
        base_offset=selected["upstream_start"],
        transcript=transcript,
        context=context_record["content"]["context"],
        reconstruction=validated_reconstruction,
        metadata=build_metadata_adapter(selected),
    )
    counts = content["counts"]
    expected_counts = {
        **reconstruction["counts"],
        **polish["counts"],
    }
    if counts != expected_counts:
        raise ValueError(
            "phase2k D counts must equal the reconstruction + polish counts",
        )
    if counts["contextual_repair_count"] != len(content["contextual_repairs"]):
        raise ValueError("phase2k D contextual repair count is inconsistent")
    if counts["resolution_repair_count"] != sum(
        1
        for repair in content["contextual_repairs"]
        if repair["repair_type"] in CONTEXTUAL_RESOLUTION_REPAIR_TYPES
    ):
        raise ValueError("phase2k D resolution repair count is inconsistent")
    if counts["binding_count"] != len(reconstruction["bindings"]):
        raise ValueError("phase2k D binding count is inconsistent")
    if counts["unresolved_alternative_count"] != len(
        reconstruction["unresolved_alternatives"],
    ):
        raise ValueError("phase2k D unresolved alternative count is inconsistent")
    if counts["statement_count"] != len(polish["statements"]):
        raise ValueError("phase2k D statement count is inconsistent")
    if counts["unsupported_claim_count"] != len(polish["unsupported_claims"]):
        raise ValueError("phase2k D unsupported claim count is inconsistent")


# Re-exported Phase 2K downstream-comparison v2 contract.  The focused
# implementation lives in pipeline/phase2k_downstream_comparison.py (which is
# import-free of this module) so these names remain available to existing
# scripts/tests without a circular import.
from pipeline.phase2k_downstream_comparison import (  # noqa: E402
    DOWNSTREAM_COMPARISON_SCHEMA_VERSION,
    FINAL_CLOSEOUT_STATUSES,
    validate_downstream_comparison,
)
