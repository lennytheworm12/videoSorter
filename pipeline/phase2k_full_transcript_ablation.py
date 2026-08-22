"""Phase 2K full-transcript context ablation (isolated, stdlib-only).

This module prepares, validates, executes, imports, reviews, and evaluates
the controlled Phase 2K architecture experiment:

  A  0x Alpha receives only the exact Bronze target passage plus useful
     reliable metadata plus the League vocabulary plus the byte-identical
     extraction instructions.  No surrounding transcript.
  B  0x Alpha receives the exact same Bronze target clearly marked inside
     the FULL ordered transcript for its source session, plus the exact
     same useful metadata, the exact same League vocabulary, and the exact
     same byte-identical extraction instructions.

Both conditions extract only source-level relational meaning: actors,
reference bindings, abilities/resources, events/actions, states,
conditions, recommended advice, consequences/outcomes, explicit
relationships over a bounded relation vocabulary, uncertainty/unresolved
references, and supporting source spans.  All citation grounding is exact
source grounding: every quote must byte-for-byte equal the supplied source
slice at its integer ``[char_start, char_end)`` range.  Condition A offsets
are into the supplied Bronze target; condition B offsets are into the
supplied full transcript.  Offsets are computed mechanically from exact
quotes plus zero-based occurrence indexes; the model never supplies
character offsets.

No Mechanical Clean, contextual reconstruction, adaptive context expansion,
semantic polish, or strategic abstraction exists anywhere in this pipeline.
The module reads the frozen Phase 2J window-selection manifest and the
read-only SQLite transcript DB, never edits Bronze, and never imports or
runs Phase 2K reconstruction code.  Only the Python standard library is
used here; live model calls are executed by the companion script through
the OpenCode CLI.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping

from pipeline.phase2j_context_ablation import (
    DIFFICULTY_WEIGHTS,
    SELECTION_COUNT,
    _envelope,
    _exact_quote_occurrences,
    _require_enum,
    _require_exact_keys,
    _require_int,
    _require_list,
    _require_nonempty_string,
    _validate_recomputed_content_hash,
    canonical_sha256,
    champion_abilities_for_transcript,
    file_sha256,
    load_json_strict,
    load_lexical_vocabulary,
    load_phase2j_manifest,
    normalize_path_locator,
    open_transcript_db,
    select_cases,
    text_sha256,
)


ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Versions and configuration
# ---------------------------------------------------------------------------

PIPELINE_VERSION = "phase2k-full-transcript-ablation-v1"

SELECTION_SCHEMA_VERSION = "phase2k-full-transcript-ablation-selection-v1"
SELECTION_POLICY_SCHEMA_VERSION = (
    "phase2k-full-transcript-ablation-selection-policy-v1"
)
INSTRUCTIONS_SCHEMA_VERSION = (
    "phase2k-full-transcript-ablation-extraction-instructions-v1"
)
VOCABULARY_SCHEMA_VERSION = (
    "phase2k-full-transcript-ablation-vocabulary-v1"
)
PAYLOAD_SCHEMA_VERSION = "phase2k-full-transcript-ablation-condition-payload-v1"
PAYLOADS_SCHEMA_VERSION = "phase2k-full-transcript-ablation-condition-payloads-v1"
INTERMEDIATE_SCHEMA_VERSION = (
    "phase2k-full-transcript-ablation-intermediate-response-v1"
)
OUTPUT_SCHEMA_VERSION = "phase2k-full-transcript-ablation-extraction-output-v1"
OUTPUTS_SCHEMA_VERSION = "phase2k-full-transcript-ablation-extraction-outputs-v1"
RUN_SCHEMA_VERSION = "phase2k-full-transcript-ablation-opencode-run-v1"
REVIEW_PACKET_SCHEMA_VERSION = (
    "phase2k-full-transcript-ablation-review-packet-v1"
)
COMPLETED_REVIEWS_SCHEMA_VERSION = (
    "phase2k-full-transcript-ablation-completed-reviews-v1"
)
EVALUATION_SUMMARY_SCHEMA_VERSION = (
    "phase2k-full-transcript-ablation-evaluation-summary-v1"
)
BUILD_SUMMARY_SCHEMA_VERSION = (
    "phase2k-full-transcript-ablation-build-summary-v1"
)

PHASE2J_MANIFEST_SCHEMA_VERSION = "phase2j-window-selection-manifest-v1"
LEAGUE_VOCABULARY_SCHEMA_VERSION = "phase2k-league-lexical-vocabulary-v2"

DEFAULT_MANIFEST_PATH = ROOT / "data/phase2j/window-selection-manifest-v1.json"
DEFAULT_DB_PATH = Path(
    "/home/bphan944/PersonalProjects/videoSorter-homework-archive/videos.db",
)
DEFAULT_VOCABULARY_PATH = (
    ROOT / "data/phase2k_support/league_lexical_vocabulary_v2.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/phase2k_context_ablation"

RELEASE_GATE_LOCKED = "LOCKED"

CONDITION_CODES = ("A", "B")
CASE_ID_PATTERN_PREFIX = "p2k:case:"

# Source-level semantic fields required by the Phase 2K contract.
SEMANTIC_FIELDS = (
    "actors_entities",
    "reference_bindings",
    "abilities_resources",
    "events_actions",
    "states",
    "conditions",
    "recommended_advice",
    "consequences_outcomes",
    "explicit_relationships",
    "uncertainty_unresolved",
    "supporting_source_spans",
)

RELATION_FIELD = "explicit_relationships"
SUPPORT_FIELD = "supporting_source_spans"

# Bounded source-semantic relation vocabulary (no strategic abstractions).
RELATION_TYPES = (
    "ACTOR",
    "TARGET",
    "USES",
    "AFFECTS",
    "CAUSES",
    "ENABLES",
    "PREVENTS",
    "REQUIRES",
    "CONDITION",
    "RESULT",
    "BEFORE",
    "AFTER",
    "UNTIL",
    "NEGATES",
    "REFERS_TO",
)

RESOLUTION_STATUSES = (
    "literal_explicit",
    "context_resolved",
    "vocabulary_supported",
    "unresolved",
)

CORRECTNESS_VALUES = ("CORRECT", "PARTIAL", "INCORRECT", "ABSENT_CORRECTLY")
UNSUPPORTED_INFERENCE_VALUES = ("NONE", "MINOR", "MAJOR")
SOURCE_GROUNDING_VALUES = (
    "GROUNDED",
    "PARTIAL",
    "UNGROUNDED",
    "NOT_APPLICABLE",
)

# Fields whose recovery directly tests the central hypothesis.
HYPOTHESIS_FOCUS_FIELDS = (
    "actors_entities",
    "abilities_resources",
    "events_actions",
    "conditions",
    "explicit_relationships",
)

METADATA_BASE_FIELDS = ("video_title", "champion", "role", "game")
METADATA_OPTIONAL_FIELDS = ("rank", "description")


class ArtifactError(ValueError):
    """Raised when a frozen or derived Phase 2K artifact fails validation."""


def load_phase2j_manifest_frozen(path: Path) -> dict[str, Any]:
    """Load and fully validate the frozen Phase 2J window-selection manifest."""
    return load_phase2j_manifest(path)


# ---------------------------------------------------------------------------
# Metadata policy: optional reliable evidence, never fabricated
# ---------------------------------------------------------------------------


def fetch_video_row_with_description(
    connection: sqlite3.Connection,
    *,
    source_id: str,
    expected_full_sha256: str,
) -> dict[str, Any]:
    """Validate one source row and return transcript + metadata + description."""
    row = connection.execute(
        "SELECT video_id, video_url, video_title, description, role, "
        "champion, rank, game, transcription FROM videos "
        "WHERE video_id = ?",
        (source_id,),
    ).fetchone()
    if row is None:
        raise ArtifactError(
            f"source {source_id} is absent from the transcript DB",
        )
    transcript = row["transcription"]
    if not isinstance(transcript, str):
        raise ArtifactError(f"source {source_id} transcription is not text")
    full_hash = text_sha256(transcript)
    if full_hash != expected_full_sha256:
        raise ArtifactError(
            f"source {source_id} full transcript SHA does not match the "
            "frozen Phase 2J upstream hash",
        )
    return {
        "source_id": source_id,
        "video_id": source_id,
        "video_url": row["video_url"],
        "video_title": row["video_title"],
        "description": row["description"],
        "role": row["role"],
        "champion": row["champion"],
        "rank": row["rank"],
        "game": row["game"],
        "transcript": transcript,
        "transcript_sha256": full_hash,
        "transcript_char_length": len(transcript),
    }


def fetch_source_rows(
    connection: sqlite3.Connection,
    selected: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in selected:
        source_id = item["upstream_source_id"]
        rows[source_id] = fetch_video_row_with_description(
            connection,
            source_id=source_id,
            expected_full_sha256=item["upstream_content_sha256"],
        )
    return rows


def build_case_metadata(row: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Build useful reliable metadata; absent optional fields are omitted.

    Returns the metadata object plus the exact list of supplied field names.
    Nothing is fabricated: a missing rank or description simply stays absent.
    """
    metadata: dict[str, Any] = {}
    supplied: list[str] = []
    for field in METADATA_BASE_FIELDS:
        value = row[field]
        if isinstance(value, str) and value.strip():
            metadata[field] = value
            supplied.append(field)
        else:
            metadata[field] = None
    for field in METADATA_OPTIONAL_FIELDS:
        value = row[field]
        if isinstance(value, str) and value.strip():
            metadata[field] = value
            supplied.append(field)
    return metadata, supplied


def bronze_text_for_offsets(transcript: str, start: int, end: int) -> str:
    return transcript[start:end]


# ---------------------------------------------------------------------------
# Case selection (frozen Phase 2J difficulty policy, p2k identifiers)
# ---------------------------------------------------------------------------


def select_phase2k_cases(
    manifest: Mapping[str, Any],
    *,
    source_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Deterministically freeze the 10 difficult Phase 2J targets.

    Selection order is exactly the preregistered Phase 2J difficulty-weighted
    top-10 (descending difficulty score, descending phenomenon count, original
    manifest order).  Only frozen manifest phenomenon tags are used as
    signals.  Identifiers are remapped to ``p2k:case:NNNN`` while retaining
    the frozen Phase 2J case id for traceability.
    """
    base_cases = select_cases(manifest)
    cases: list[dict[str, Any]] = []
    for base in base_cases:
        source_id = base["upstream_source_id"]
        row = source_rows[source_id]
        rank = base["selection_rank"]
        case = dict(base)
        case["case_id"] = f"{CASE_ID_PATTERN_PREFIX}{rank:04d}"
        case["phase2j_case_id"] = base["case_id"]
        case["full_transcript_char_length"] = row["transcript_char_length"]
        case["game"] = row["game"]
        cases.append(case)
    return cases


def expected_top10_window_ids(manifest: Mapping[str, Any]) -> list[str]:
    """Independently recompute the canonical top-10 window ids."""
    rows = []
    for index, selected in enumerate(manifest["selected"]):
        phenomena = list(selected["phenomena"])
        present = [tag for tag in phenomena if tag in DIFFICULTY_WEIGHTS]
        rows.append((
            index,
            sum(DIFFICULTY_WEIGHTS[tag] for tag in present),
            len(phenomena),
            selected["window_id"],
        ))
    rows.sort(key=lambda row: (-row[1], -row[2], row[0]))
    return [row[3] for row in rows[:SELECTION_COUNT]]


def build_selection_artifact(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    db_path: Path,
    cases: list[Mapping[str, Any]],
) -> dict[str, Any]:
    policy = {
        "schema_version": SELECTION_POLICY_SCHEMA_VERSION,
        "count": SELECTION_COUNT,
        "difficulty_weights": dict(DIFFICULTY_WEIGHTS),
        "scoring": (
            "difficulty score = sum of the preregistered weights of the "
            "frozen Phase 2J manifest phenomenon tags present in the window"
        ),
        "sort": (
            "descending difficulty score, then descending total phenomenon "
            "count, then original manifest selected order"
        ),
        "tie_break": "original manifest selected order",
        "selection_signals": [
            "frozen phase2j-window-selection-manifest-v1 phenomenon tags only",
        ],
        "excluded_signals": [
            "phase2k reconstruction results",
            "model predictions",
            "human semantic outputs",
            "endpoint counts",
            "partition",
            "gold labels",
        ],
        "identifier_policy": (
            "case ids are p2k-prefixed remappings of the identical frozen "
            "Phase 2J difficulty-weighted top-10 selection order; the frozen "
            "phase2j case id is retained for traceability"
        ),
    }
    artifact = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "purpose": (
            "Frozen Phase 2K full-transcript ablation 10-target manifest."
        ),
        "release_gate": RELEASE_GATE_LOCKED,
        "pipeline_version": PIPELINE_VERSION,
        "selection_policy": policy,
        "input_hashes": {
            "manifest": {
                "path": normalize_path_locator(manifest_path),
                "file_sha256": file_sha256(manifest_path),
                "content_sha256": manifest["content_sha256"],
                "schema_version": manifest["schema_version"],
            },
            "transcript_db": {
                "path": normalize_path_locator(db_path),
                "file_sha256": file_sha256(db_path),
            },
        },
        "cases": [dict(case) for case in cases],
    }
    return _envelope(artifact)


def validate_selection_artifact(
    artifact: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    db_path: Path,
    connection: sqlite3.Connection,
) -> None:
    _require_exact_keys(
        artifact,
        (
            "schema_version", "purpose", "release_gate", "pipeline_version",
            "selection_policy", "input_hashes", "cases", "content_sha256",
        ),
        "phase2k selection artifact",
    )
    if artifact["schema_version"] != SELECTION_SCHEMA_VERSION:
        raise ArtifactError("selection artifact schema version is invalid")
    if artifact["release_gate"] != RELEASE_GATE_LOCKED:
        raise ArtifactError("selection artifact release gate is not LOCKED")
    if artifact["pipeline_version"] != PIPELINE_VERSION:
        raise ArtifactError("selection artifact pipeline version is invalid")
    _validate_recomputed_content_hash(artifact, label="selection artifact")
    cases = artifact["cases"]
    if not isinstance(cases, list) or len(cases) != SELECTION_COUNT:
        raise ArtifactError("selection artifact must contain exactly 10 cases")
    expected_window_ids = expected_top10_window_ids(manifest)
    observed_window_ids = [case["window_id"] for case in cases]
    if observed_window_ids != expected_window_ids:
        raise ArtifactError(
            "selection artifact window order is not the canonical top-10",
        )
    hashes = artifact["input_hashes"]
    if hashes["manifest"]["content_sha256"] != manifest["content_sha256"]:
        raise ArtifactError("selection artifact manifest hash is invalid")
    if hashes["manifest"]["file_sha256"] != file_sha256(manifest_path):
        raise ArtifactError("selection artifact manifest file hash is invalid")
    if hashes["transcript_db"]["file_sha256"] != file_sha256(db_path):
        raise ArtifactError("selection artifact transcript DB hash is invalid")
    for rank, case in enumerate(cases, 1):
        _require_exact_keys(
            case,
            (
                "selection_rank", "case_id", "phase2j_case_id",
                "manifest_index", "window_id", "source_group_id",
                "upstream_source_id", "video_url", "partition", "metadata",
                "phenomena", "difficulty_score", "contributing_tags",
                "phenomenon_count", "contributing_tag_count",
                "bronze_char_length", "upstream_start", "upstream_end",
                "bronze_text_sha256", "source_text_sha256",
                "upstream_content_sha256", "canonical_record_sha256",
                "candidate_catalog_sha256", "full_transcript_sha256",
                "full_transcript_char_length", "game",
            ),
            "selection case",
        )
        if case["selection_rank"] != rank:
            raise ArtifactError("selection case rank ordering is invalid")
        if case["case_id"] != f"{CASE_ID_PATTERN_PREFIX}{rank:04d}":
            raise ArtifactError("selection case id is invalid")
        row = connection.execute(
            "SELECT transcription, video_title, champion, role FROM videos "
            "WHERE video_id = ?",
            (case["upstream_source_id"],),
        ).fetchone()
        if row is None:
            raise ArtifactError("selection case source is absent from the DB")
        transcript = row["transcription"]
        if text_sha256(transcript) != case["upstream_content_sha256"]:
            raise ArtifactError(
                "selection case transcript hash does not match the frozen "
                "manifest upstream hash",
            )
        bronze = bronze_text_for_offsets(
            transcript, case["upstream_start"], case["upstream_end"],
        )
        if text_sha256(bronze) != case["bronze_text_sha256"]:
            raise ArtifactError("selection case bronze hash is invalid")
        if len(bronze) != case["bronze_char_length"]:
            raise ArtifactError("selection case bronze length is invalid")
        if row["video_title"] != case["metadata"]["video_title"]:
            raise ArtifactError("selection case metadata title mismatch")
        if row["champion"] != case["metadata"]["champion"]:
            raise ArtifactError("selection case metadata champion mismatch")
        if row["role"] != case["metadata"]["role"]:
            raise ArtifactError("selection case metadata role mismatch")
        if len(transcript) != case["full_transcript_char_length"]:
            raise ArtifactError(
                "selection case full transcript length is invalid",
            )


# ---------------------------------------------------------------------------
# Extraction instructions (byte-identical across both conditions)
# ---------------------------------------------------------------------------

FIELD_GUIDANCE: dict[str, str] = {
    "actors_entities": (
        "Champions, players, and other entities the target passage is about. "
        "Preserve literal surface names when the source wording is used; mark "
        "context-resolved bindings explicitly."
    ),
    "reference_bindings": (
        "Every pronoun or reference in the target passage and what it binds "
        "to. If a reference cannot be resolved from the supplied material, "
        "record it with resolution_status 'unresolved' instead of guessing."
    ),
    "abilities_resources": (
        "Ability, summoner spell, item, and resource references, including "
        "whose ability each one is when the source supports ownership."
    ),
    "events_actions": (
        "Concrete things that happened or were done in the game, as stated "
        "or entailed by the source."
    ),
    "states": (
        "Game states that hold during the target passage: health, mana, "
        "cooldowns, position, wave states, objective states, and similar."
    ),
    "conditions": (
        "Explicit or clearly implied conditions under which events, advice, "
        "or consequences hold ('if', 'when', 'until', 'unless' content)."
    ),
    "recommended_advice": (
        "Advice or recommended actions the coach actually gives in the "
        "target passage. Do not invent plausible League advice the coach "
        "did not state."
    ),
    "consequences_outcomes": (
        "Stated or clearly implied consequences and outcomes of events, "
        "states, or (non-)compliance with advice."
    ),
    "explicit_relationships": (
        "Typed relationships between extracted items using ONLY the supplied "
        "relation_types vocabulary. Each relationship must be supported by "
        "the source."
    ),
    "uncertainty_unresolved": (
        "Spans or bindings the speaker or you are uncertain about, ASR "
        "corruptions that cannot be recovered, and references that remain "
        "unresolved. Preserving uncertainty is required; resolving it by "
        "guessing is forbidden."
    ),
    "supporting_source_spans": (
        "The minimal source spans that jointly support the extraction for "
        "this target passage."
    ),
}

_INSTRUCTION_RULES = (
    "Use only information supported by the supplied source material and "
    "metadata.",
    "The target passage is marked explicitly; extract meaning FOR THE TARGET "
    "PASSAGE.",
    "Resolve pronouns, champion references, and ability ownership only when "
    "the supplied material supports the resolution.",
    "Do not invent strategic conclusions that are merely plausible League "
    "advice.",
    "Do not normalize results into higher-level strategic concepts such as "
    "access, continuity, initiative, beatdown, or conversion unless the "
    "coach explicitly states that concept.",
    "Do not repair ASR corruption beyond what the supplied vocabulary and "
    "context support; record unrecoverable spans as unresolved.",
    "Preserve uncertainty: use resolution_status 'unresolved' whenever a "
    "binding or span cannot be supported.",
)

_GROUNDING_RULES = (
    "Every extracted item must cite at least one source reference.",
    "A source reference is an exact contiguous 'quote' copied byte-for-byte "
    "from the condition's supplied source text plus a zero-based "
    "'occurrence_index' counted among all exact non-overlapping substring "
    "matches of that quote in the condition source.",
    "Never supply character offsets; deterministic tooling resolves your "
    "quotes to ranges.",
    "Condition A source text is the supplied Bronze target passage. "
    "Condition B source text is the supplied full transcript.",
    "For explicit_relationships items, relation_type MUST come from the "
    "supplied relation_types list.",
    "For supporting_source_spans items, cite the exact quotes that cover "
    "the support.",
)


def build_extraction_instructions() -> dict[str, Any]:
    response_contract = {
        "shape": (
            "Return ONLY one minified JSON object with exactly these keys: "
            "schema_version, case_id, condition, payload_sha256, "
            "instructions_sha256, fields."
        ),
        "header_rules": [
            "schema_version must be exactly '" + INTERMEDIATE_SCHEMA_VERSION + "'.",
            "case_id and condition must copy the payload values exactly.",
            "payload_sha256 and instructions_sha256 must copy the payload "
            "hashes exactly.",
        ],
        "fields_rules": [
            "fields must contain exactly these keys: "
            + ", ".join(SEMANTIC_FIELDS) + ".",
            "Each field value is a possibly-empty array of item objects.",
            "Every item object has exactly the keys: extraction_text "
            "(non-empty string, max 2000 chars), resolution_status (one of "
            + ", ".join(RESOLUTION_STATUSES) + "), source_references "
            "(non-empty array).",
            "Every source_reference object has exactly the keys: quote "
            "(non-empty exact substring of the condition source), "
            "occurrence_index (integer >= 0).",
            "Items in the explicit_relationships field additionally have "
            "exactly one more key: relation_type (one of "
            + ", ".join(RELATION_TYPES) + ").",
        ],
    }
    instructions = {
        "schema_version": INSTRUCTIONS_SCHEMA_VERSION,
        "role": (
            "You are extracting source-grounded semantic information from a "
            "League of Legends coaching transcript."
        ),
        "task_rules": list(_INSTRUCTION_RULES),
        "fields": dict(FIELD_GUIDANCE),
        "relation_types": list(RELATION_TYPES),
        "resolution_statuses": list(RESOLUTION_STATUSES),
        "grounding_rules": list(_GROUNDING_RULES),
        "response_contract": response_contract,
        "forbidden_processing": [
            "mechanical_clean",
            "contextual_reconstruction",
            "adaptive_context_expansion",
            "semantic_polish",
            "strategic_abstraction",
        ],
    }
    return _envelope(instructions)


def validate_extraction_instructions(instructions: Mapping[str, Any]) -> None:
    _require_exact_keys(
        instructions,
        (
            "schema_version", "role", "task_rules", "fields",
            "relation_types", "resolution_statuses", "grounding_rules",
            "response_contract", "forbidden_processing", "content_sha256",
        ),
        "phase2k extraction instructions",
    )
    if instructions["schema_version"] != INSTRUCTIONS_SCHEMA_VERSION:
        raise ArtifactError("extraction instructions schema version is invalid")
    _validate_recomputed_content_hash(
        instructions, label="extraction instructions",
    )
    for rule in _INSTRUCTION_RULES:
        if rule not in instructions["task_rules"]:
            raise ArtifactError("extraction instructions task rules changed")
    if list(instructions["relation_types"]) != list(RELATION_TYPES):
        raise ArtifactError("extraction instructions relation types changed")
    if list(instructions["resolution_statuses"]) != list(RESOLUTION_STATUSES):
        raise ArtifactError("extraction instructions statuses changed")
    if set(instructions["fields"]) != set(SEMANTIC_FIELDS):
        raise ArtifactError("extraction instructions fields changed")
    for rule in _GROUNDING_RULES:
        if rule not in instructions["grounding_rules"]:
            raise ArtifactError("extraction instructions grounding changed")
    for name in (
        "mechanical_clean",
        "contextual_reconstruction",
        "adaptive_context_expansion",
        "semantic_polish",
        "strategic_abstraction",
    ):
        if name not in instructions["forbidden_processing"]:
            raise ArtifactError(
                "extraction instructions forbidden processing changed",
            )


def build_instructions_artifact() -> dict[str, Any]:
    return build_extraction_instructions()


def validate_instructions_artifact(artifact: Mapping[str, Any]) -> None:
    validate_extraction_instructions(artifact)


# ---------------------------------------------------------------------------
# Case vocabulary (lexical v2 + DB champion abilities)
# ---------------------------------------------------------------------------


def build_case_vocabulary(
    *,
    case_id: str,
    lexical_vocabulary: Mapping[str, Any],
    champion_data: Mapping[str, Any],
) -> dict[str, Any]:
    vocabulary = {
        "schema_version": VOCABULARY_SCHEMA_VERSION,
        "case_id": case_id,
        "lexical_vocabulary": dict(lexical_vocabulary),
        "lexical_vocabulary_sha256": canonical_sha256(lexical_vocabulary),
        "champions": champion_data["champions"],
        "champion_abilities": champion_data["champion_abilities"],
        "selected_champion_count": champion_data["selected_champion_count"],
        "ability_row_count": champion_data["ability_row_count"],
    }
    return _envelope(vocabulary)


def validate_case_vocabulary(
    vocabulary: Mapping[str, Any],
    *,
    case_id: str,
    lexical_vocabulary: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        vocabulary,
        (
            "schema_version", "case_id", "lexical_vocabulary",
            "lexical_vocabulary_sha256", "champions", "champion_abilities",
            "selected_champion_count", "ability_row_count", "content_sha256",
        ),
        "phase2k case vocabulary",
    )
    if vocabulary["schema_version"] != VOCABULARY_SCHEMA_VERSION:
        raise ArtifactError("case vocabulary schema version is invalid")
    if vocabulary["case_id"] != case_id:
        raise ArtifactError("case vocabulary case_id is invalid")
    _validate_recomputed_content_hash(vocabulary, label="case vocabulary")
    if canonical_sha256(vocabulary["lexical_vocabulary"]) != (
        vocabulary["lexical_vocabulary_sha256"]
    ):
        raise ArtifactError("case vocabulary lexical hash is invalid")
    if canonical_sha256(lexical_vocabulary) != (
        vocabulary["lexical_vocabulary_sha256"]
    ):
        raise ArtifactError("case vocabulary lexical vocabulary mismatch")
    for champion in _require_list(vocabulary["champions"], "vocabulary champions"):
        _require_exact_keys(
            champion,
            ("champion", "selection_reasons", "provenance"),
            "vocabulary champion",
        )
    for ability in _require_list(
        vocabulary["champion_abilities"], "vocabulary abilities",
    ):
        _require_exact_keys(
            ability,
            (
                "champion", "ability_slot", "name", "description",
                "cooldown", "range", "cost", "properties", "provenance",
            ),
            "vocabulary ability",
        )


# ---------------------------------------------------------------------------
# Condition payloads
# ---------------------------------------------------------------------------

_A_PAYLOAD_KEYS = (
    "schema_version", "condition", "case_id", "selection_rank",
    "target", "metadata", "metadata_fields_supplied", "vocabulary",
    "vocabulary_sha256", "instructions", "instructions_sha256",
    "content_sha256",
)

_B_PAYLOAD_KEYS = (
    "schema_version", "condition", "case_id", "selection_rank",
    "target", "transcript", "target_char_start", "target_char_end",
    "metadata", "metadata_fields_supplied", "vocabulary",
    "vocabulary_sha256", "instructions", "instructions_sha256",
    "content_sha256",
)

_FORBIDDEN_PAYLOAD_KEYS = frozenset({
    # Identity/provenance must stay at the outer artifact level.  Nested
    # vocabulary selection provenance is allowed evidence.
    "video_id", "video_url", "source_id", "source_group_id", "window_id",
    "upstream_start", "upstream_end",
    # Gold/review/scoring leakage.
    "gold", "labels", "predictions", "human_review", "evaluation",
    "materiality", "endpoints", "review_items",
    # Strategic/generated layers are forbidden in every condition.
    "archetypes", "fingerprints", "strategic", "insights",
    "compiled_principles",
})

_FORBIDDEN_A_KEYS = _FORBIDDEN_PAYLOAD_KEYS | {
    # Condition A must never see surrounding discourse.
    "transcript", "target_char_start", "target_char_end",
    "surrounding_context", "context_window",
}


def _scan_forbidden_payload_keys(
    value: object,
    *,
    forbidden: frozenset[str],
    path: str = "payload",
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in forbidden:
                raise ArtifactError(
                    f"payload leaks forbidden key {key!r} at {path}",
                )
            _scan_forbidden_payload_keys(
                item, forbidden=forbidden, path=f"{path}.{key}",
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_forbidden_payload_keys(
                item, forbidden=forbidden, path=f"{path}[{index}]",
            )


def build_condition_payloads(
    *,
    cases: list[Mapping[str, Any]],
    source_rows: Mapping[str, Mapping[str, Any]],
    vocabulary_by_case: Mapping[str, Mapping[str, Any]],
    instructions: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build model-visible A/B payloads plus outer provenance bindings."""
    instructions_sha256 = canonical_sha256(instructions)
    payload_cases: list[dict[str, Any]] = []
    provenance_by_case: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = case["case_id"]
        source_id = case["upstream_source_id"]
        row = source_rows[source_id]
        transcript = row["transcript"]
        start = case["upstream_start"]
        end = case["upstream_end"]
        bronze_text = bronze_text_for_offsets(transcript, start, end)
        metadata, supplied = build_case_metadata(row)
        common_target = {
            "bronze_text": bronze_text,
            "bronze_text_sha256": text_sha256(bronze_text),
            "bronze_char_length": len(bronze_text),
        }
        shared = {
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "case_id": case_id,
            "selection_rank": case["selection_rank"],
            "target": dict(common_target),
            "metadata": dict(metadata),
            "metadata_fields_supplied": list(supplied),
            "vocabulary": dict(vocabulary_by_case[case_id]),
            "vocabulary_sha256": vocabulary_by_case[case_id]["content_sha256"],
            "instructions": dict(instructions),
            "instructions_sha256": instructions_sha256,
        }
        payload_a = _envelope({**shared, "condition": "A"})
        payload_b = _envelope({
            **shared,
            "condition": "B",
            "transcript": transcript,
            "target_char_start": start,
            "target_char_end": end,
        })
        payload_cases.append({
            "case_id": case_id,
            "selection_rank": case["selection_rank"],
            "A": payload_a,
            "B": payload_b,
        })
        provenance_by_case[case_id] = {
            "video_id": source_id,
            "video_url": row["video_url"],
            "source_group_id": case["source_group_id"],
            "window_id": case["window_id"],
            "phase2j_case_id": case["phase2j_case_id"],
            "full_transcript_sha256": row["transcript_sha256"],
            "full_transcript_char_length": row["transcript_char_length"],
            "target_char_start": start,
            "target_char_end": end,
            "vocabulary_sha256": vocabulary_by_case[case_id]["content_sha256"],
            "metadata_fields_supplied": list(supplied),
        }
    return payload_cases, provenance_by_case


def build_payloads_artifact(
    *,
    selection: Mapping[str, Any],
    instructions: Mapping[str, Any],
    payload_cases: list[Mapping[str, Any]],
    provenance_by_case: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return _envelope({
        "schema_version": PAYLOADS_SCHEMA_VERSION,
        "purpose": (
            "Frozen Phase 2K full-transcript ablation condition payloads: "
            "A = isolated Bronze target plus useful reliable metadata plus "
            "League vocabulary plus extraction instructions; B = the exact "
            "same Bronze target clearly marked inside the FULL ordered "
            "transcript plus the exact same metadata, vocabulary, and "
            "instructions.  Only discourse context differs.  Source identity "
            "provenance is retained at this outer level, not inside the "
            "model-visible payloads."
        ),
        "release_gate": RELEASE_GATE_LOCKED,
        "pipeline_version": PIPELINE_VERSION,
        "selection_sha256": selection["content_sha256"],
        "instructions_sha256": canonical_sha256(instructions),
        "provenance_by_case": {
            str(case_id): dict(entry)
            for case_id, entry in sorted(provenance_by_case.items())
        },
        "cases": [dict(case) for case in payload_cases],
    })


def _payload_source_text(payload: Mapping[str, Any]) -> str:
    if payload["condition"] == "A":
        return payload["target"]["bronze_text"]
    return payload["transcript"]


def _validate_metadata(metadata: Mapping[str, Any], *, label: str) -> None:
    allowed = set(METADATA_BASE_FIELDS) | set(METADATA_OPTIONAL_FIELDS)
    if not set(metadata) <= allowed:
        raise ArtifactError(f"{label} metadata contains unknown fields")
    if not set(METADATA_BASE_FIELDS) <= set(metadata):
        raise ArtifactError(f"{label} metadata is missing required fields")
    for key, value in metadata.items():
        if value is not None and not isinstance(value, str):
            raise ArtifactError(f"{label} metadata {key} must be a string or null")
    for key in METADATA_BASE_FIELDS:
        if not isinstance(metadata[key], str) or not metadata[key].strip():
            raise ArtifactError(f"{label} metadata {key} must be supplied")


def _validate_condition_payload(
    payload: Mapping[str, Any],
    *,
    expected_case_id: str,
    expected_condition: str,
    instructions: Mapping[str, Any],
    lexical_vocabulary: Mapping[str, Any],
) -> None:
    keys = _B_PAYLOAD_KEYS if expected_condition == "B" else _A_PAYLOAD_KEYS
    label = f"phase2k condition {expected_condition} payload"
    _require_exact_keys(payload, keys, label)
    if payload["schema_version"] != PAYLOAD_SCHEMA_VERSION:
        raise ArtifactError("condition payload schema version is invalid")
    _validate_recomputed_content_hash(payload, label=label)
    if payload["condition"] != expected_condition:
        raise ArtifactError("condition payload condition is invalid")
    if payload["case_id"] != expected_case_id:
        raise ArtifactError("condition payload case_id is invalid")
    if canonical_sha256(payload["instructions"]) != payload["instructions_sha256"]:
        raise ArtifactError("condition payload instructions hash is invalid")
    if canonical_sha256(instructions) != payload["instructions_sha256"]:
        raise ArtifactError(
            "condition payload instructions do not match the canonical object",
        )
    validate_extraction_instructions(payload["instructions"])
    target = payload["target"]
    if not isinstance(target, Mapping):
        raise ArtifactError("condition payload target must be an object")
    _require_exact_keys(
        target,
        ("bronze_text", "bronze_text_sha256", "bronze_char_length"),
        "condition payload target",
    )
    bronze_text = _require_nonempty_string(target["bronze_text"], "payload bronze_text")
    if target["bronze_text_sha256"] != text_sha256(bronze_text):
        raise ArtifactError("condition payload bronze_text_sha256 is invalid")
    if target["bronze_char_length"] != len(bronze_text):
        raise ArtifactError("condition payload bronze_char_length is invalid")
    _validate_metadata(payload["metadata"], label=label)
    supplied = payload["metadata_fields_supplied"]
    if not isinstance(supplied, list) or any(not isinstance(s, str) for s in supplied):
        raise ArtifactError("condition payload metadata_fields_supplied is invalid")
    if sorted(set(supplied)) != sorted([
        key for key, value in payload["metadata"].items()
        if isinstance(value, str) and value.strip()
    ]):
        raise ArtifactError(
            "condition payload metadata_fields_supplied does not match "
            "the supplied metadata",
        )
    if payload["vocabulary_sha256"] != payload["vocabulary"]["content_sha256"]:
        raise ArtifactError("condition payload vocabulary hash is invalid")
    validate_case_vocabulary(
        payload["vocabulary"],
        case_id=expected_case_id,
        lexical_vocabulary=lexical_vocabulary,
    )
    if expected_condition == "B":
        transcript = _require_nonempty_string(
            payload["transcript"], "condition B transcript",
        )
        start = _require_int(
            payload["target_char_start"], "condition B target_char_start",
            minimum=0,
        )
        end = _require_int(
            payload["target_char_end"], "condition B target_char_end",
            minimum=1,
        )
        if not 0 <= start < end <= len(transcript):
            raise ArtifactError("condition B target character offsets are invalid")
        if transcript[start:end] != bronze_text:
            raise ArtifactError(
                "condition B target offsets do not slice the supplied "
                "transcript to the Bronze text",
            )
        forbidden = _FORBIDDEN_PAYLOAD_KEYS
    else:
        forbidden = _FORBIDDEN_A_KEYS
    _scan_forbidden_payload_keys(payload, forbidden=forbidden)


def validate_payload_pair_isolation(pair: Mapping[str, Any]) -> None:
    """A and B may differ ONLY in discourse context fields."""
    payload_a = pair["A"]
    payload_b = pair["B"]
    shared_keys = (
        "schema_version", "case_id", "selection_rank", "target",
        "metadata", "metadata_fields_supplied", "vocabulary",
        "instructions",
    )
    for key in shared_keys:
        if canonical_sha256(payload_a[key]) != canonical_sha256(payload_b[key]):
            raise ArtifactError(
                f"A/B payloads differ in non-context key {key!r}",
            )
    if payload_a["content_sha256"] == payload_b["content_sha256"]:
        raise ArtifactError(
            "A/B payload content hashes collide; contexts are not distinct",
        )


def validate_payloads_artifact(
    artifact: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    instructions: Mapping[str, Any],
    lexical_vocabulary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    connection: sqlite3.Connection,
) -> None:
    _require_exact_keys(
        artifact,
        (
            "schema_version", "purpose", "release_gate", "pipeline_version",
            "selection_sha256", "instructions_sha256", "provenance_by_case",
            "cases", "content_sha256",
        ),
        "phase2k payloads artifact",
    )
    if artifact["schema_version"] != PAYLOADS_SCHEMA_VERSION:
        raise ArtifactError("payloads artifact schema version is invalid")
    if artifact["release_gate"] != RELEASE_GATE_LOCKED:
        raise ArtifactError("payloads artifact release gate is not LOCKED")
    if artifact["pipeline_version"] != PIPELINE_VERSION:
        raise ArtifactError("payloads artifact pipeline version is invalid")
    _validate_recomputed_content_hash(artifact, label="payloads artifact")
    if artifact["selection_sha256"] != selection["content_sha256"]:
        raise ArtifactError("payloads artifact selection hash is invalid")
    if artifact["instructions_sha256"] != canonical_sha256(instructions):
        raise ArtifactError("payloads artifact instructions hash is invalid")
    cases = artifact["cases"]
    if not isinstance(cases, list) or len(cases) != SELECTION_COUNT:
        raise ArtifactError("payloads artifact must contain exactly 10 cases")
    selection_by_id = {case["case_id"]: case for case in selection["cases"]}
    for pair in cases:
        case_id = pair["case_id"]
        if case_id not in selection_by_id:
            raise ArtifactError("payloads artifact case id is not selected")
        case = selection_by_id[case_id]
        for condition in CONDITION_CODES:
            payload = pair[condition]
            _validate_condition_payload(
                payload,
                expected_case_id=case_id,
                expected_condition=condition,
                instructions=instructions,
                lexical_vocabulary=lexical_vocabulary,
            )
        validate_payload_pair_isolation(pair)
        # Rebuild the exact payloads deterministically and compare.
        rebuilt_pairs, rebuilt_provenance = build_condition_payloads(
            cases=[case],
            source_rows={
                case["upstream_source_id"]: fetch_video_row_with_description(
                    connection,
                    source_id=case["upstream_source_id"],
                    expected_full_sha256=case["upstream_content_sha256"],
                ),
            },
            vocabulary_by_case={case_id: pair["A"]["vocabulary"]},
            instructions=instructions,
        )
        if canonical_sha256(rebuilt_pairs[0]) != canonical_sha256(pair):
            raise ArtifactError(
                f"payload pair for {case_id} is not canonically reproducible",
            )
        recorded = artifact["provenance_by_case"][case_id]
        if canonical_sha256(recorded) != canonical_sha256(
            rebuilt_provenance[case_id],
        ):
            raise ArtifactError(
                f"provenance for {case_id} is not canonically reproducible",
            )


# ---------------------------------------------------------------------------
# Intermediate response schema / validation / import
# ---------------------------------------------------------------------------


def build_intermediate_schema() -> dict[str, Any]:
    reference_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["quote", "occurrence_index"],
        "properties": {
            "quote": {"type": "string", "minLength": 1},
            "occurrence_index": {"type": "integer", "minimum": 0},
        },
    }
    item_required = [
        "extraction_text", "resolution_status", "source_references",
    ]
    item_properties = {
        "extraction_text": {
            "type": "string", "minLength": 1, "maxLength": 2000,
        },
        "resolution_status": {
            "type": "string", "enum": list(RESOLUTION_STATUSES),
        },
        "source_references": {
            "type": "array",
            "minItems": 1,
            "items": reference_schema,
        },
    }
    relation_item_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": item_required + ["relation_type"],
        "properties": {
            **item_properties,
            "relation_type": {
                "type": "string", "enum": list(RELATION_TYPES),
            },
        },
    }
    plain_item_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": item_required,
        "properties": dict(item_properties),
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Phase 2K full-transcript ablation intermediate response",
        "schema_version": INTERMEDIATE_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "case_id", "condition", "payload_sha256",
            "instructions_sha256", "fields",
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": INTERMEDIATE_SCHEMA_VERSION,
            },
            "case_id": {"type": "string", "pattern": r"^p2k:case:[0-9]{4}$"},
            "condition": {"type": "string", "enum": list(CONDITION_CODES)},
            "payload_sha256": {
                "type": "string", "pattern": r"^[0-9a-f]{64}$",
            },
            "instructions_sha256": {
                "type": "string", "pattern": r"^[0-9a-f]{64}$",
            },
            "fields": {
                "type": "object",
                "additionalProperties": False,
                "required": list(SEMANTIC_FIELDS),
                "properties": {
                    RELATION_FIELD: {
                        "type": "array",
                        "items": relation_item_schema,
                    },
                    **{
                        field: {"type": "array", "items": plain_item_schema}
                        for field in SEMANTIC_FIELDS
                        if field != RELATION_FIELD
                    },
                },
            },
        },
    }


def validate_intermediate_schema(schema: Mapping[str, Any]) -> None:
    if dict(schema) != build_intermediate_schema():
        raise ArtifactError("intermediate schema is not the canonical schema")


def validate_intermediate_response(
    response: Mapping[str, Any],
    *,
    case_id: str,
    condition: str,
    payload: Mapping[str, Any],
) -> None:
    """Strictly validate a model-visible intermediate response."""
    _require_exact_keys(
        response,
        (
            "schema_version", "case_id", "condition", "payload_sha256",
            "instructions_sha256", "fields",
        ),
        "phase2k intermediate response",
    )
    if response["schema_version"] != INTERMEDIATE_SCHEMA_VERSION:
        raise ArtifactError("intermediate response schema version is invalid")
    if response["case_id"] != case_id:
        raise ArtifactError("intermediate response case_id is invalid")
    if response["condition"] != condition:
        raise ArtifactError("intermediate response condition is invalid")
    if response["payload_sha256"] != payload["content_sha256"]:
        raise ArtifactError(
            "intermediate response is not bound to the condition payload",
        )
    if response["instructions_sha256"] != payload["instructions_sha256"]:
        raise ArtifactError("intermediate response instructions hash is invalid")
    fields = response["fields"]
    if not isinstance(fields, Mapping) or set(fields) != set(SEMANTIC_FIELDS):
        raise ArtifactError("intermediate response fields are invalid")
    for field in SEMANTIC_FIELDS:
        items = _require_list(fields[field], f"intermediate field {field}")
        for item in items:
            if not isinstance(item, Mapping):
                raise ArtifactError("intermediate item must be an object")
            expected = ["extraction_text", "resolution_status",
                        "source_references"]
            if field == RELATION_FIELD:
                expected = expected + ["relation_type"]
            _require_exact_keys(item, expected, f"intermediate {field} item")
            extraction_text = _require_nonempty_string(
                item["extraction_text"], "intermediate extraction_text",
            )
            if len(extraction_text) > 2000:
                raise ArtifactError(
                    "intermediate extraction_text exceeds the 2000-char bound",
                )
            _require_enum(
                item["resolution_status"], RESOLUTION_STATUSES,
                "intermediate resolution_status",
            )
            if field == RELATION_FIELD:
                _require_enum(
                    item["relation_type"], RELATION_TYPES,
                    "intermediate relation_type",
                )
            references = _require_list(
                item["source_references"],
                "intermediate source_references",
            )
            if not references:
                raise ArtifactError(
                    "intermediate item must cite at least one source reference",
                )
            for reference in references:
                if not isinstance(reference, Mapping):
                    raise ArtifactError(
                        "intermediate source_reference must be an object",
                    )
                _require_exact_keys(
                    reference,
                    ("quote", "occurrence_index"),
                    "intermediate source_reference",
                )
                _require_nonempty_string(
                    reference["quote"], "intermediate quote",
                )
                _require_int(
                    reference["occurrence_index"],
                    "intermediate occurrence_index",
                    minimum=0,
                )


def import_intermediate_response(
    response: Mapping[str, Any],
    *,
    case_id: str,
    condition: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministically resolve quotes to ranges and derive item IDs."""
    validate_intermediate_response(
        response, case_id=case_id, condition=condition, payload=payload,
    )
    source = _payload_source_text(payload)
    fields: dict[str, list[dict[str, Any]]] = {}
    for field in SEMANTIC_FIELDS:
        imported_items: list[dict[str, Any]] = []
        for item_index, item in enumerate(response["fields"][field], 1):
            references: list[dict[str, Any]] = []
            for reference in item["source_references"]:
                quote = reference["quote"]
                occurrence_index = reference["occurrence_index"]
                matches = _exact_quote_occurrences(source, quote)
                if occurrence_index >= len(matches):
                    raise ArtifactError(
                        f"quote {quote!r} has only {len(matches)} exact "
                        f"occurrence(s); occurrence_index "
                        f"{occurrence_index} is out of range",
                    )
                char_start = matches[occurrence_index]
                references.append({
                    "quote": quote,
                    "source_range": {
                        "char_start": char_start,
                        "char_end": char_start + len(quote),
                    },
                })
            imported_item: dict[str, Any] = {
                "item_id": f"{case_id}:{condition}:{field}:{item_index:04d}",
                "extraction_text": item["extraction_text"],
                "resolution_status": item["resolution_status"],
                "source_references": references,
            }
            if field == RELATION_FIELD:
                imported_item["relation_type"] = item["relation_type"]
            if field == SUPPORT_FIELD:
                imported_item["source_range"] = {
                    "char_start": min(
                        reference["source_range"]["char_start"]
                        for reference in references
                    ),
                    "char_end": max(
                        reference["source_range"]["char_end"]
                        for reference in references
                    ),
                }
            imported_items.append(imported_item)
        fields[field] = imported_items
    output = _envelope({
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "case_id": case_id,
        "condition": condition,
        "payload_sha256": payload["content_sha256"],
        "instructions_sha256": payload["instructions_sha256"],
        "fields": fields,
    })
    validate_extraction_output(
        output, case_id=case_id, condition=condition, payload=payload,
    )
    return output


def _validate_resolved_reference(
    reference: Mapping[str, Any],
    *,
    source: str,
    label: str,
) -> None:
    _require_exact_keys(
        reference,
        ("quote", "source_range"),
        f"{label} source_reference",
    )
    quote = _require_nonempty_string(reference["quote"], f"{label} quote")
    source_range = reference["source_range"]
    _require_exact_keys(
        source_range, ("char_start", "char_end"), f"{label} source_range",
    )
    start = _require_int(
        source_range["char_start"], f"{label} char_start", minimum=0,
    )
    end = _require_int(
        source_range["char_end"], f"{label} char_end", minimum=1,
    )
    if end <= start or end > len(source):
        raise ArtifactError(f"{label} source range is out of bounds")
    if source[start:end] != quote:
        raise ArtifactError(
            f"{label} quote is not byte-exact at its resolved range",
        )


def validate_extraction_output(
    output: Mapping[str, Any],
    *,
    case_id: str,
    condition: str,
    payload: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        output,
        (
            "schema_version", "case_id", "condition", "payload_sha256",
            "instructions_sha256", "fields", "content_sha256",
        ),
        "phase2k extraction output",
    )
    if output["schema_version"] != OUTPUT_SCHEMA_VERSION:
        raise ArtifactError("extraction output schema version is invalid")
    _validate_recomputed_content_hash(output, label="extraction output")
    if output["case_id"] != case_id or output["condition"] != condition:
        raise ArtifactError("extraction output identity is invalid")
    if output["payload_sha256"] != payload["content_sha256"]:
        raise ArtifactError("extraction output payload binding is invalid")
    if output["instructions_sha256"] != payload["instructions_sha256"]:
        raise ArtifactError("extraction output instructions binding is invalid")
    fields = output["fields"]
    if not isinstance(fields, Mapping) or set(fields) != set(SEMANTIC_FIELDS):
        raise ArtifactError("extraction output fields are invalid")
    source = _payload_source_text(payload)
    seen_ids: set[str] = set()
    for field in SEMANTIC_FIELDS:
        items = _require_list(fields[field], f"output field {field}")
        for index, item in enumerate(items, 1):
            expected = ["item_id", "extraction_text", "resolution_status",
                        "source_references"]
            if field == RELATION_FIELD:
                expected.append("relation_type")
            if field == SUPPORT_FIELD:
                expected.append("source_range")
            _require_exact_keys(item, expected, f"output {field} item")
            item_id = item["item_id"]
            if item_id in seen_ids:
                raise ArtifactError("duplicate output item_id")
            seen_ids.add(item_id)
            expected_id = f"{case_id}:{condition}:{field}:{index:04d}"
            if item_id != expected_id:
                raise ArtifactError("output item_id ordering is invalid")
            _require_nonempty_string(
                item["extraction_text"], "output extraction_text",
            )
            if len(item["extraction_text"]) > 2000:
                raise ArtifactError("output extraction_text exceeds bound")
            _require_enum(
                item["resolution_status"], RESOLUTION_STATUSES,
                "output resolution_status",
            )
            if field == RELATION_FIELD:
                _require_enum(
                    item["relation_type"], RELATION_TYPES,
                    "output relation_type",
                )
            references = _require_list(
                item["source_references"], "output source_references",
            )
            if not references:
                raise ArtifactError(
                    "output item must cite at least one source reference",
                )
            for reference in references:
                _validate_resolved_reference(
                    reference, source=source, label=f"output {field}",
                )
            if field == SUPPORT_FIELD:
                starts = [
                    reference["source_range"]["char_start"]
                    for reference in references
                ]
                ends = [
                    reference["source_range"]["char_end"]
                    for reference in references
                ]
                if (
                    item["source_range"]["char_start"] != min(starts)
                    or item["source_range"]["char_end"] != max(ends)
                ):
                    raise ArtifactError(
                        "output supporting span range is not the minimal "
                        "bounding range",
                    )


def build_outputs_bundle(
    *,
    payloads: Mapping[str, Any],
    outputs_by_call: Mapping[tuple[str, str], Mapping[str, Any]],
    by_call_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cases = []
    for pair in payloads["cases"]:
        case_id = pair["case_id"]
        entry: dict[str, Any] = {"case_id": case_id}
        for condition in CONDITION_CODES:
            output = outputs_by_call.get((case_id, condition))
            if output is None:
                raise ArtifactError(
                    f"missing validated output for {case_id}/{condition}",
                )
            validate_extraction_output(
                output,
                case_id=case_id,
                condition=condition,
                payload=pair[condition],
            )
            entry[condition] = dict(output)
        cases.append(entry)
    bundle: dict[str, Any] = {
        "schema_version": OUTPUTS_SCHEMA_VERSION,
        "purpose": (
            "Validated Phase 2K full-transcript ablation extraction outputs."
        ),
        "release_gate": RELEASE_GATE_LOCKED,
        "payloads_sha256": payloads["content_sha256"],
        "instructions_sha256": payloads["instructions_sha256"],
        "cases": cases,
    }
    if by_call_evidence is not None:
        bundle["by_call_evidence"] = dict(sorted(by_call_evidence.items()))
    return _envelope(bundle)


def validate_outputs_bundle(
    bundle: Mapping[str, Any],
    *,
    payloads: Mapping[str, Any],
) -> None:
    allowed_key_sets = (
        (
            "schema_version", "purpose", "release_gate", "payloads_sha256",
            "instructions_sha256", "cases", "content_sha256",
        ),
        (
            "schema_version", "purpose", "release_gate", "payloads_sha256",
            "instructions_sha256", "cases", "by_call_evidence",
            "content_sha256",
        ),
    )
    if tuple(sorted(bundle)) not in (
        tuple(sorted(keys)) for keys in allowed_key_sets
    ):
        raise ArtifactError("outputs bundle key set is invalid")
    if bundle["schema_version"] != OUTPUTS_SCHEMA_VERSION:
        raise ArtifactError("outputs bundle schema version is invalid")
    _validate_recomputed_content_hash(bundle, label="outputs bundle")
    if bundle["payloads_sha256"] != payloads["content_sha256"]:
        raise ArtifactError("outputs bundle payload binding is invalid")
    if bundle["instructions_sha256"] != payloads["instructions_sha256"]:
        raise ArtifactError("outputs bundle instructions binding is invalid")
    pairs_by_id = {pair["case_id"]: pair for pair in payloads["cases"]}
    if len(bundle["cases"]) != len(pairs_by_id):
        raise ArtifactError("outputs bundle case count is invalid")
    for entry in bundle["cases"]:
        case_id = entry["case_id"]
        pair = pairs_by_id.get(case_id)
        if pair is None:
            raise ArtifactError("outputs bundle case id is not in payloads")
        for condition in CONDITION_CODES:
            validate_extraction_output(
                entry[condition],
                case_id=case_id,
                condition=condition,
                payload=pair[condition],
            )


# ---------------------------------------------------------------------------
# Human review packet, completed reviews, and evaluation
# ---------------------------------------------------------------------------

REVIEW_FIELDS = SEMANTIC_FIELDS


def _format_output_for_review(
    output: Mapping[str, Any],
    *,
    source: str,
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for field in SEMANTIC_FIELDS:
        items = []
        for item in output["fields"][field]:
            citations = []
            for reference in item["source_references"]:
                citations.append({
                    "quote": reference["quote"],
                    "char_start": reference["source_range"]["char_start"],
                    "char_end": reference["source_range"]["char_end"],
                    "verified_byte_exact": (
                        source[
                            reference["source_range"]["char_start"]:
                            reference["source_range"]["char_end"]
                        ] == reference["quote"]
                    ),
                })
            entry = {
                "extraction_text": item["extraction_text"],
                "resolution_status": item["resolution_status"],
                "citations": citations,
            }
            if "relation_type" in item:
                entry["relation_type"] = item["relation_type"]
            if "source_range" in item:
                entry["span"] = dict(item["source_range"])
            items.append(entry)
        rendered.append({"field": field, "items": items})
    return rendered


def build_review_packet(
    *,
    payloads: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    validate_outputs_bundle(outputs, payloads=payloads)
    provenance = payloads["provenance_by_case"]
    cases = []
    if len(payloads["cases"]) != len(outputs["cases"]):
        raise ArtifactError("outputs bundle case count does not match payloads")
    for pair, output_entry in zip(payloads["cases"], outputs["cases"]):
        if pair["case_id"] != output_entry["case_id"]:
            raise ArtifactError(
                "payload/output case alignment is invalid",
            )
        case_id = pair["case_id"]
        record = provenance[case_id]
        transcript = pair["B"]["transcript"]
        bronze = pair["A"]["target"]["bronze_text"]
        start = record["target_char_start"]
        end = record["target_char_end"]
        pre = transcript[max(0, start - 120):start]
        post = transcript[end:min(len(transcript), end + 120)]
        cases.append({
            "case_id": case_id,
            "selection_rank": pair["selection_rank"],
            "metadata": pair["A"]["metadata"],
            "metadata_fields_supplied": pair["A"]["metadata_fields_supplied"],
            "window_id": record["window_id"],
            "video_url": record["video_url"],
            "target": {
                "bronze_text": bronze,
                "bronze_text_sha256": pair["A"]["target"]["bronze_text_sha256"],
                "char_start": start,
                "char_end": end,
                "context_before": pre,
                "context_after": post,
            },
            "condition_A_isolated_bronze": {
                "source_kind": "isolated_bronze_target_only",
                "structured_extraction": _format_output_for_review(
                    output_entry["A"], source=bronze,
                ),
                "raw_response_binding": {
                    "payload_sha256": output_entry["A"]["payload_sha256"],
                },
            },
            "condition_B_full_transcript": {
                "source_kind": "full_ordered_transcript_with_marked_target",
                "structured_extraction": _format_output_for_review(
                    output_entry["B"], source=transcript,
                ),
                "raw_response_binding": {
                    "payload_sha256": output_entry["B"]["payload_sha256"],
                },
            },
        })
    packet = {
        "schema_version": REVIEW_PACKET_SCHEMA_VERSION,
        "purpose": (
            "Human-reviewable A/B comparison per target. Score semantic "
            "recovery only; do not credit prose fluency."
        ),
        "release_gate": RELEASE_GATE_LOCKED,
        "payloads_sha256": payloads["content_sha256"],
        "outputs_sha256": outputs["content_sha256"],
        "scoring_scales": {
            "correctness": list(CORRECTNESS_VALUES),
            "unsupported_inference": list(UNSUPPORTED_INFERENCE_VALUES),
            "source_grounding": list(SOURCE_GROUNDING_VALUES),
        },
        "strict_success_definition": (
            "correctness in {CORRECT, ABSENT_CORRECTLY} AND "
            "unsupported_inference = NONE AND source_grounding in "
            "{GROUNDED, NOT_APPLICABLE}"
        ),
        "review_fields": list(REVIEW_FIELDS),
        "hypothesis_focus_fields": list(HYPOTHESIS_FOCUS_FIELDS),
        "cases": cases,
    }
    return _envelope(packet)


def validate_review_packet(packet: Mapping[str, Any]) -> None:
    _require_exact_keys(
        packet,
        (
            "schema_version", "purpose", "release_gate", "payloads_sha256",
            "outputs_sha256", "scoring_scales", "strict_success_definition",
            "review_fields", "hypothesis_focus_fields", "cases",
            "content_sha256",
        ),
        "phase2k review packet",
    )
    if packet["schema_version"] != REVIEW_PACKET_SCHEMA_VERSION:
        raise ArtifactError("review packet schema version is invalid")
    _validate_recomputed_content_hash(packet, label="review packet")
    if packet["review_fields"] != list(REVIEW_FIELDS):
        raise ArtifactError("review packet fields changed")
    for case in packet["cases"]:
        for section in (
            "condition_A_isolated_bronze", "condition_B_full_transcript",
        ):
            rendered = case[section]["structured_extraction"]
            if [entry["field"] for entry in rendered] != list(SEMANTIC_FIELDS):
                raise ArtifactError("review packet extraction rendering invalid")


def validate_completed_reviews(
    completed: Mapping[str, Any],
    *,
    review_packet: Mapping[str, Any],
) -> None:
    if completed.get("schema_version") != COMPLETED_REVIEWS_SCHEMA_VERSION:
        raise ArtifactError("completed reviews schema version is invalid")
    kind = completed.get("reviewer_kind")
    if kind == "human":
        _require_exact_keys(
            completed,
            (
                "schema_version", "reviewer_kind", "reviewer_identity",
                "completed_at", "reviews", "human_review_attested",
                "attestation_statement", "content_sha256",
            ),
            "phase2k completed reviews",
        )
        if completed["human_review_attested"] is not True:
            raise ArtifactError("human reviews must attest personal review")
        statement = completed["attestation_statement"]
    elif kind == "agent":
        _require_exact_keys(
            completed,
            (
                "schema_version", "reviewer_kind", "reviewer_identity",
                "completed_at", "reviews", "agent_scoping_statement",
                "content_sha256",
            ),
            "phase2k completed reviews",
        )
        statement = completed["agent_scoping_statement"]
    else:
        raise ArtifactError(
            "completed reviews reviewer_kind must be 'human' or 'agent'",
        )
    _require_nonempty_string(
        completed["reviewer_identity"], "completed reviews reviewer_identity",
    )
    _require_nonempty_string(
        completed["completed_at"], "completed reviews completed_at",
    )
    if not (isinstance(statement, str) and statement.strip()):
        raise ArtifactError("completed reviews require a scoping/attestation statement")
    _require_nonempty_string(
        completed["reviewer_identity"], "completed reviews reviewer_identity",
    )
    _require_nonempty_string(
        completed["completed_at"], "completed reviews completed_at",
    )
    _validate_recomputed_content_hash(completed, label="completed reviews")
    reviews = completed["reviews"]
    expected_keys = {
        f"{case['case_id']}:{section}:{field}"
        for case in review_packet["cases"]
        for section in ("A", "B")
        for field in REVIEW_FIELDS
    }
    if set(reviews) != expected_keys:
        raise ArtifactError(
            "completed reviews must score exactly all case/condition/field items",
        )
    for key, entry in reviews.items():
        _require_exact_keys(
            entry,
            ("correctness", "unsupported_inference", "source_grounding",
             "notes"),
            f"review {key}",
        )
        _require_enum(entry["correctness"], CORRECTNESS_VALUES, "correctness")
        _require_enum(
            entry["unsupported_inference"],
            UNSUPPORTED_INFERENCE_VALUES,
            "unsupported_inference",
        )
        _require_enum(
            entry["source_grounding"], SOURCE_GROUNDING_VALUES,
            "source_grounding",
        )
        if not isinstance(entry["notes"], list):
            raise ArtifactError(f"review {key} notes must be a list")


def _is_strict_success(scores: Mapping[str, Any]) -> bool:
    return (
        scores["correctness"] in ("CORRECT", "ABSENT_CORRECTLY")
        and scores["unsupported_inference"] == "NONE"
        and scores["source_grounding"] in ("GROUNDED", "NOT_APPLICABLE")
    )


def compute_evaluation_summary(
    *,
    review_packet: Mapping[str, Any],
    completed_reviews: Mapping[str, Any],
) -> dict[str, Any]:
    validate_review_packet(review_packet)
    validate_completed_reviews(completed_reviews, review_packet=review_packet)
    reviews = completed_reviews["reviews"]

    def strict(case_id: str, condition: str, field: str) -> bool:
        return _is_strict_success(reviews[f"{case_id}:{condition}:{field}"])

    def unsupported(case_id: str, condition: str, field: str) -> str:
        return reviews[f"{case_id}:{condition}:{field}"]["unsupported_inference"]

    per_field: dict[str, dict[str, int]] = {}
    for field in REVIEW_FIELDS:
        stats = {"A_strict_successes": 0, "B_strict_successes": 0}
        for case in review_packet["cases"]:
            case_id = case["case_id"]
            if strict(case_id, "A", field):
                stats["A_strict_successes"] += 1
            if strict(case_id, "B", field):
                stats["B_strict_successes"] += 1
        total = len(review_packet["cases"])
        stats["A_rate"] = round(stats["A_strict_successes"] / total, 4)
        stats["B_rate"] = round(stats["B_strict_successes"] / total, 4)
        stats["net_B_minus_A"] = (
            stats["B_strict_successes"] - stats["A_strict_successes"]
        )
        per_field[field] = stats

    unsupported_counts = {
        condition: {
            "NONE": 0, "MINOR": 0, "MAJOR": 0,
        }
        for condition in CONDITION_CODES
    }
    unresolved_counts = {condition: 0 for condition in CONDITION_CODES}
    grounding_failures = {condition: 0 for condition in CONDITION_CODES}
    for case in review_packet["cases"]:
        case_id = case["case_id"]
        for condition in CONDITION_CODES:
            for field in REVIEW_FIELDS:
                entry = reviews[f"{case_id}:{condition}:{field}"]
                unsupported_counts[condition][entry["unsupported_inference"]] += 1
                if entry["source_grounding"] in ("PARTIAL", "UNGROUNDED"):
                    grounding_failures[condition] += 1

    per_target = []
    b_wins = a_wins = ties = 0
    focus_b_wins = {field: 0 for field in HYPOTHESIS_FOCUS_FIELDS}
    focus_a_wins = {field: 0 for field in HYPOTHESIS_FOCUS_FIELDS}
    for case in review_packet["cases"]:
        case_id = case["case_id"]
        a_count = sum(strict(case_id, "A", f) for f in REVIEW_FIELDS)
        b_count = sum(strict(case_id, "B", f) for f in REVIEW_FIELDS)
        if b_count > a_count:
            verdict = "B_STRICTLY_WINS"
            b_wins += 1
        elif a_count > b_count:
            verdict = "A_STRICTLY_WINS"
            a_wins += 1
        else:
            verdict = "TIE"
            ties += 1
        focus_results = {}
        for field in HYPOTHESIS_FOCUS_FIELDS:
            if strict(case_id, "B", field) and not strict(case_id, "A", field):
                focus_results[field] = "B_ONLY_SUCCESS"
                focus_b_wins[field] += 1
            elif strict(case_id, "A", field) and not strict(case_id, "B", field):
                focus_results[field] = "A_ONLY_SUCCESS"
                focus_a_wins[field] += 1
            elif strict(case_id, "A", field) and strict(case_id, "B", field):
                focus_results[field] = "BOTH_SUCCESS"
            else:
                focus_results[field] = "BOTH_FAIL"
        per_target.append({
            "case_id": case_id,
            "A_strict_successes": a_count,
            "B_strict_successes": b_count,
            "verdict": verdict,
            "focus_field_results": focus_results,
        })

    total_paired = len(review_packet["cases"]) * len(REVIEW_FIELDS)
    net = sum(stats["net_B_minus_A"] for stats in per_field.values())
    major_increase = (
        unsupported_counts["B"]["MAJOR"] > unsupported_counts["A"]["MAJOR"]
    )

    # Preregistered materiality gate (recorded before any model outputs).
    materiality_criteria = {
        "net_strict_success_margin_at_least_15_percent": (
            net * 100 >= 15 * total_paired
        ),
        "b_strictly_wins_at_least_4_targets": b_wins >= 4,
        "a_strictly_wins_at_most_2_targets": a_wins <= 2,
        "no_major_unsupported_increase_in_b": not major_increase,
        "focus_improvement_in_at_least_3_of_5_focus_fields": (
            sum(
                1 for field in HYPOTHESIS_FOCUS_FIELDS
                if per_field[field]["net_B_minus_A"] > 0
            ) >= 3
        ),
    }
    if all(materiality_criteria.values()):
        decision = "ISOLATED_BRONZE_WAS_THE_WRONG_SEMANTIC_UNIT"
    elif net > 0 or b_wins > a_wins:
        decision = "GLOBAL_CONTEXT_HELPS_BUT_DOES_NOT_SOLVE_SOURCE_RECOVERY"
    else:
        decision = "FULL_TRANSCRIPT_CONTEXT_DOES_NOT_EXPLAIN_THE_FAILURE"

    summary = {
        "schema_version": EVALUATION_SUMMARY_SCHEMA_VERSION,
        "purpose": (
            "Deterministic aggregate evaluation of the Phase 2K "
            "full-transcript ablation."
        ),
        "review_packet_sha256": review_packet["content_sha256"],
        "completed_reviews_sha256": completed_reviews["content_sha256"],
        "reviewer_identity": completed_reviews["reviewer_identity"],
        "total_paired_field_judgments": total_paired,
        "per_field": per_field,
        "hypothesis_focus_net": {
            field: per_field[field]["net_B_minus_A"]
            for field in HYPOTHESIS_FOCUS_FIELDS
        },
        "per_target": per_target,
        "target_verdict_counts": {
            "B_STRICTLY_WINS": b_wins,
            "A_STRICTLY_WINS": a_wins,
            "TIE": ties,
        },
        "unsupported_inference_counts": unsupported_counts,
        "grounding_failures": grounding_failures,
        "materiality_criteria": materiality_criteria,
        "decision_gate": decision,
    }
    return _envelope(summary)


def validate_evaluation_summary(
    summary: Mapping[str, Any],
    *,
    review_packet: Mapping[str, Any],
    completed_reviews: Mapping[str, Any],
) -> None:
    recomputed = compute_evaluation_summary(
        review_packet=review_packet,
        completed_reviews=completed_reviews,
    )
    if canonical_sha256(summary) != canonical_sha256(recomputed):
        raise ArtifactError(
            "evaluation summary is not canonically reproducible",
        )


# ---------------------------------------------------------------------------
# Build summary
# ---------------------------------------------------------------------------


def build_build_summary(
    *,
    output_dir: Path,
    selection: Mapping[str, Any],
    instructions: Mapping[str, Any],
    payloads: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    artifacts = {
        "selection": {
            "path": str((Path(output_dir)).joinpath(
                "phase2k-context-ablation-selection-v1.json",
            )),
            "content_sha256": selection["content_sha256"],
            "schema_version": selection["schema_version"],
        },
        "instructions": {
            "path": str(Path(output_dir).joinpath(
                "phase2k-context-ablation-extraction-instructions-v1.json",
            )),
            "content_sha256": instructions["content_sha256"],
            "schema_version": instructions["schema_version"],
        },
        "payloads": {
            "path": str(Path(output_dir).joinpath(
                "phase2k-context-ablation-condition-payloads-v1.json",
            )),
            "content_sha256": payloads["content_sha256"],
            "schema_version": payloads["schema_version"],
        },
    }
    summary = {
        "schema_version": BUILD_SUMMARY_SCHEMA_VERSION,
        "purpose": (
            "Phase 2K full-transcript ablation build summary (no model calls)."
        ),
        "mode": mode,
        "pipeline_version": PIPELINE_VERSION,
        "artifacts": artifacts,
        "selected_case_ids": [
            case["case_id"] for case in payloads["cases"]
        ],
    }
    return _envelope(summary)
