"""Phase 2J source-grounded semantic-extraction ablation (isolated, stdlib-only).

This module prepares and validates the controlled 10-example Phase 2J
context ablation:

  A  GPT-5.6 Sol receives only the isolated Bronze target plus the exact
     byte-identical extraction instructions.  No full transcript, no
     metadata, no vocabulary, no source/video identity, and no surrounding
     text.
  B  GPT-5.6 Sol receives the exact same Bronze target, the full archived
     transcript with the target's character offsets, useful ordinary
     metadata only, and the League champion/ability vocabulary, plus the
     exact same byte-identical extraction instructions.

Both conditions directly extract actors, ability/resource references,
event, condition, advice/action, consequence, uncertainty, and supporting
source ranges.  All citation grounding is exact source grounding: every
quote must byte-for-byte equal the supplied source slice at its integer
``[char_start, char_end)`` range.  Condition A offsets are into the
supplied Bronze target; condition B offsets are into the supplied full
transcript.  There is no mechanical cleaning, contextual rewriting,
semantic polish, or strategic abstraction anywhere in the pipeline.

The module is intentionally isolated: it reads the frozen Phase 2J
artifacts and a read-only SQLite transcript DB, but never imports or edits
Phase 2K code/data, never runs Phase 2J scoring, and never touches Phase 2K
implementation or artifacts.  It uses only the Python standard library and
makes no model calls.

Selection is frozen from the frozen Phase 2J window-selection manifest tags
only (never from Phase 2K results, model predictions, human semantic
outputs, endpoint counts, or partition).  Transcript text is read directly
from the archived read-only DB; no timestamps or captions are required or
invented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
from typing import Any, Callable, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Versions and configuration
# ---------------------------------------------------------------------------

PIPELINE_VERSION = "phase2j-context-ablation-v2"

SELECTION_SCHEMA_VERSION = "phase2j-context-ablation-selection-v1"
SELECTION_POLICY_SCHEMA_VERSION = "phase2j-context-ablation-selection-policy-v1"
INSTRUCTIONS_SCHEMA_VERSION = "phase2j-context-ablation-extraction-instructions-v2"
PAYLOAD_SCHEMA_VERSION = "phase2j-context-ablation-condition-payload-v2"
PAYLOADS_SCHEMA_VERSION = "phase2j-context-ablation-condition-payloads-v2"
VOCABULARY_SCHEMA_VERSION = "phase2j-context-ablation-vocabulary-v1"
OUTPUT_SCHEMA_VERSION = "phase2j-context-ablation-extraction-output-v2"
OUTPUTS_SCHEMA_VERSION = "phase2j-context-ablation-extraction-outputs-v2"
REVIEW_PRESENTATION_SCHEMA_VERSION = (
    "phase2j-context-ablation-review-presentation-v2"
)
HUMAN_PACKET_SCHEMA_VERSION = "phase2j-context-ablation-human-review-packet-v2"
HUMAN_MAPPING_SCHEMA_VERSION = "phase2j-context-ablation-human-review-mapping-v2"
COMPLETED_REVIEWS_SCHEMA_VERSION = "phase2j-context-ablation-completed-reviews-v2"
MATERIALITY_POLICY_SCHEMA_VERSION = (
    "phase2j-context-ablation-materiality-policy-v2"
)
MATERIALITY_SUMMARY_SCHEMA_VERSION = (
    "phase2j-context-ablation-materiality-summary-v2"
)
DEEPSEEK_RUN_SCHEMA_VERSION = "phase2j-context-ablation-deepseek-run-v2"
DEEPSEEK_IMPORT_SCHEMA_VERSION = "phase2j-context-ablation-deepseek-import-v2"
SOL_INTERMEDIATE_SCHEMA_VERSION = "phase2j-context-ablation-sol-intermediate-v2"
BUILD_SUMMARY_SCHEMA_VERSION = "phase2j-context-ablation-build-summary-v2"

PHASE2J_MANIFEST_SCHEMA_VERSION = "phase2j-window-selection-manifest-v1"
PHASE2J_PACKET_SCHEMA_VERSION = "phase2j-endpoint-annotation-packet-v1"
LEAGUE_VOCABULARY_SCHEMA_VERSION = "phase2k-league-lexical-vocabulary-v2"

RELEASE_GATE_LOCKED = "LOCKED"
RELEASE_GATE_AWAITING_REVIEW = "AWAITING_HUMAN_REVIEW"
RELEASE_GATE_REVIEWED = "REVIEWED"

BUILD_MODE_READY_FOR_SOL = "ready_for_sol"
BUILD_MODE_REVIEW_PACKET = "review_packet"

CONDITION_CODES = ("A", "B")

SEMANTIC_FIELDS = (
    "actors",
    "ability_resource_references",
    "event",
    "condition",
    "advice_action",
    "consequence",
    "uncertainty",
    "supporting_source_ranges",
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

REVIEW_FIELDS = (
    "correctness",
    "unsupported_inference",
    "source_grounding",
    "notes",
)

STRICT_SUCCESS = {
    "correctness": ["CORRECT", "ABSENT_CORRECTLY"],
    "unsupported_inference": ["NONE"],
    "source_grounding": ["GROUNDED", "NOT_APPLICABLE"],
}

# Preregistered materiality decision (frozen before any model outputs).
MATERIALITY_POLICY = {
    "schema_version": MATERIALITY_POLICY_SCHEMA_VERSION,
    "strict_success": {
        "correctness": ["CORRECT", "ABSENT_CORRECTLY"],
        "unsupported_inference": ["NONE"],
        "source_grounding": ["GROUNDED", "NOT_APPLICABLE"],
    },
    "paired_field_judgments": 10 * len(SEMANTIC_FIELDS),
    "material_conditions": [
        "B strict-success fields - A strict-success fields >= 12",
        "B strictly wins >= 4 cases by per-case strict-success count",
        "A strictly wins <= 1 case",
        "B MAJOR unsupported-inference judgments <= A MAJOR "
        "unsupported-inference judgments",
    ],
    "thresholds": {
        "field_gain_min": 12,
        "case_wins_min": 4,
        "a_case_wins_max": 1,
    },
}

# Exact preregistered difficulty weight table (frozen manifest tags only).
DIFFICULTY_WEIGHTS: dict[str, int] = {
    "punctuation_poor": 3,
    "omitted_actor": 3,
    "pronoun": 2,
    "multiple_abilities": 2,
    "multiple_champions": 2,
    "nested_condition": 2,
    "cause_chain": 2,
    "uncertainty": 2,
    "contradiction": 1,
    "implicit_cause": 1,
    "explicit_cause": 1,
    "conditional": 1,
    "temporal": 1,
    "contrast": 1,
    "advice_explanation": 1,
    "resource_exchange": 1,
    "wave_reasoning": 1,
    "multi_sentence": 1,
}

SELECTION_COUNT = 10

# Private deterministic blinding constants.  These are intentionally not
# exposed in any reviewer-visible artifact or in the docs given to the
# reviewer; the sealed mapping is the only authority for label->condition.
_REVIEW_LABEL_SEED = "phase2j-context-ablation-label-20260820"
_REVIEW_ORDER_SEED = "phase2j-context-ablation-order-20260820"

DEFAULT_MANIFEST_PATH = ROOT / "data/phase2j/window-selection-manifest-v1.json"
DEFAULT_PACKET_PATH = ROOT / "data/phase2j/reviewed-endpoint-annotation-packet-v1.json"
DEFAULT_DB_PATH = Path(
    "/home/bphan944/PersonalProjects/videoSorter-homework-archive/videos.db",
)
DEFAULT_VOCABULARY_PATH = ROOT / "data/phase2k_support/league_lexical_vocabulary_v2.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/phase2j_context_ablation"

OUTPUT_FILENAMES = {
    "selection": "phase2j-context-ablation-selection-v1.json",
    "instructions": "phase2j-context-ablation-extraction-instructions-v2.json",
    "payloads": "phase2j-context-ablation-condition-payloads-v2.json",
    "outputs": "phase2j-context-ablation-extraction-outputs-v2.json",
    "human_packet": "phase2j-context-ablation-human-review-packet-v2.json",
    "human_mapping": "phase2j-context-ablation-human-review-mapping-v2.json",
    "completed_reviews": "phase2j-context-ablation-completed-reviews-v2.json",
    "finalized_packet": (
        "phase2j-context-ablation-human-review-packet-v2-finalized.json"
    ),
    "materiality_summary": "phase2j-context-ablation-materiality-summary-v2.json",
    "deepseek_run": "phase2j-context-ablation-deepseek-run-v2.json",
    "deepseek_import": "phase2j-context-ablation-deepseek-import-v2.json",
    "build_summary": "phase2j-context-ablation-build-summary-v2.json",
}

# ---------------------------------------------------------------------------
# Hashing / serialization helpers (consistent with repository conventions)
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
    """Deterministic path locator independent of relative/absolute spelling."""
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
    if not text:
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


def _validate_recomputed_content_hash(
    obj: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(obj.get("content_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", obj["content_sha256"],
    ):
        raise ValueError(f"{label} content_sha256 is missing or malformed")
    expected = canonical_sha256({
        key: value for key, value in obj.items() if key != "content_sha256"
    })
    if obj["content_sha256"] != expected:
        raise ValueError(f"{label} content_sha256 does not match canonical content")


def _canonical_hash_of_parsed(obj: Mapping[str, Any], *, label: str) -> str:
    return canonical_sha256({
        key: value for key, value in obj.items() if key != "content_sha256"
    })


# ---------------------------------------------------------------------------
# Frozen Phase 2J input loaders (isolated, do not import Phase 2K modules)
# ---------------------------------------------------------------------------


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
    if manifest["release_gate"] != RELEASE_GATE_LOCKED:
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
        metadata = selected_item["metadata"]
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "champion", "role", "video_title",
        }:
            raise ValueError("phase2j selected metadata is invalid")
        for key in ("champion", "role", "video_title"):
            if not isinstance(metadata[key], str):
                raise ValueError("phase2j metadata values must be strings")
        phenomena = _require_list(selected_item["phenomena"], "phase2j phenomena")
        if any(not isinstance(item, str) for item in phenomena):
            raise ValueError("phase2j phenomena must be strings")
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
    if packet["release_gate"] != RELEASE_GATE_LOCKED:
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
        if any(
            not isinstance(endpoint, Mapping) or endpoint.get("disposition") != "KEEP"
            for endpoint in endpoints
        ):
            raise ValueError("phase2j reviewed endpoint must be KEEP")
    return packet


def validate_phase2j_frozen_inputs(
    manifest_path: Path,
    packet_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate both frozen Phase 2J artifacts and their cross-bindings."""
    manifest = load_phase2j_manifest(manifest_path)
    packet = load_phase2j_reviewed_packet(packet_path)
    if packet["selection_manifest_sha256"] != manifest["content_sha256"]:
        raise ValueError("phase2j reviewed packet is not bound to the frozen manifest")
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
    *,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """File and canonical content hashes of the frozen Phase 2J inputs."""
    result = {
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
    if db_path is not None:
        result["transcript_db"] = {
            "path": normalize_path_locator(db_path),
            "file_sha256": file_sha256(db_path),
        }
    return result


# ---------------------------------------------------------------------------
# Deterministic case selection (frozen manifest tags only)
# ---------------------------------------------------------------------------


def _difficulty_score(phenomena: Iterable[str]) -> tuple[int, list[str]]:
    present = [tag for tag in phenomena if tag in DIFFICULTY_WEIGHTS]
    return sum(DIFFICULTY_WEIGHTS[tag] for tag in present), present


def select_cases(
    manifest: Mapping[str, Any],
    *,
    transcript_rows: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Freeze the 10 controlled cases from frozen manifest tags only.

    Sort is descending difficulty score, then descending total phenomenon
    count, then original manifest selected order.  No Phase 2K results,
    model predictions, human semantic outputs, endpoint counts, or partition
    values are used as selection signals.
    """
    rows: list[tuple[int, int, int, int, list[str], list[str], dict[str, Any]]] = []
    for index, selected in enumerate(manifest["selected"]):
        phenomena = list(selected["phenomena"])
        score, contributing = _difficulty_score(phenomena)
        rows.append((index, score, len(phenomena), len(contributing), phenomena,
                     contributing, selected))
    rows.sort(key=lambda row: (-row[1], -row[2], row[0]))
    cases: list[dict[str, Any]] = []
    for rank, (index, score, phenomenon_count, tag_count, phenomena,
               contributing, selected) in enumerate(rows[:SELECTION_COUNT], 1):
        source_id = selected["upstream_source_id"]
        video_url = None
        game = None
        transcript_len = None
        if transcript_rows is not None:
            row = transcript_rows.get(source_id)
            if row is None:
                raise ValueError(
                    f"source {source_id} is absent from the transcript rows",
                )
            video_url = row.get("video_url")
            game = row.get("game")
            transcript_len = row.get("transcript_char_length")
        case = {
            "selection_rank": rank,
            "case_id": f"p2ja:case:{rank:04d}",
            "manifest_index": index,
            "window_id": selected["window_id"],
            "source_group_id": selected["source_group_id"],
            "upstream_source_id": source_id,
            "video_url": video_url,
            "partition": selected["partition"],
            "metadata": dict(selected["metadata"]),
            "phenomena": phenomena,
            "difficulty_score": score,
            "contributing_tags": contributing,
            "phenomenon_count": phenomenon_count,
            "contributing_tag_count": tag_count,
            "bronze_char_length": len(selected["source_text"]),
            "upstream_start": selected["upstream_start"],
            "upstream_end": selected["upstream_end"],
            "bronze_text_sha256": selected["source_text_sha256"],
            "source_text_sha256": selected["source_text_sha256"],
            "upstream_content_sha256": selected["upstream_content_sha256"],
            "canonical_record_sha256": selected["canonical_record_sha256"],
            "candidate_catalog_sha256": selected["candidate_catalog_sha256"],
            "full_transcript_sha256": selected["upstream_content_sha256"],
            "full_transcript_char_length": transcript_len,
            "game": game,
        }
        cases.append(case)
    return cases


def build_selection_artifact(
    *,
    manifest_path: Path,
    packet_path: Path,
    manifest: Mapping[str, Any],
    packet: Mapping[str, Any],
    cases: list[Mapping[str, Any]],
    db_path: Path,
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
            "phase2k results",
            "model predictions",
            "human semantic outputs",
            "endpoint counts",
            "partition",
            "gold labels",
        ],
    }
    artifact = _envelope({
        "schema_version": SELECTION_SCHEMA_VERSION,
        "purpose": (
            "Frozen Phase 2J full-context ablation: exactly 10 controlled "
            "cases selected from the frozen Phase 2J window-selection "
            "manifest tags only.  Not gold and not labels."
        ),
        "release_gate": RELEASE_GATE_LOCKED,
        "selection_policy": policy,
        "input_hashes": frozen_input_hashes(
            manifest_path, packet_path, manifest, packet, db_path=db_path,
        ),
        "cases": [dict(case) for case in cases],
    })
    return artifact


def validate_selection_artifact(
    artifact: Mapping[str, Any],
    *,
    manifest_path: Path,
    packet_path: Path,
    manifest: Mapping[str, Any],
    packet: Mapping[str, Any],
    db_path: Path,
) -> None:
    """Validate selection independently from the frozen manifest + DB.

    The canonical cases are recomputed from the frozen manifest tags and the
    read-only DB (never from the artifact's own cases), so a self-rehashed
    tampered case cannot pass.
    """
    _require_exact_keys(
        artifact,
        (
            "schema_version", "purpose", "release_gate", "selection_policy",
            "input_hashes", "cases", "content_sha256",
        ),
        "phase2j context-ablation selection",
    )
    if artifact["schema_version"] != SELECTION_SCHEMA_VERSION:
        raise ValueError("selection artifact schema version is invalid")
    if artifact["release_gate"] != RELEASE_GATE_LOCKED:
        raise ValueError("selection artifact is not LOCKED")
    _validate_recomputed_content_hash(artifact, label="selection artifact")
    connection = open_transcript_db(db_path)
    try:
        rows = fetch_transcript_rows(connection, manifest["selected"])
    finally:
        connection.close()
    cases = select_cases(manifest, transcript_rows=rows)
    for case in cases:
        validate_manifest_db_alignment(case, rows[case["upstream_source_id"]])
    expected = build_selection_artifact(
        manifest_path=manifest_path,
        packet_path=packet_path,
        manifest=manifest,
        packet=packet,
        cases=cases,
        db_path=db_path,
    )
    if dict(artifact) != dict(expected):
        raise ValueError("selection artifact does not match the canonical recomputation")
    if artifact["content_sha256"] != expected["content_sha256"]:
        raise ValueError("selection artifact content does not match canonical build")
    cases = _require_list(artifact["cases"], "selection cases")
    if len(cases) != SELECTION_COUNT:
        raise ValueError("selection artifact must contain exactly 10 cases")
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError("selection case must be an object")
        _require_exact_keys(
            case,
            (
                "selection_rank", "case_id", "manifest_index", "window_id",
                "source_group_id", "upstream_source_id", "video_url",
                "partition", "metadata", "phenomena", "difficulty_score",
                "contributing_tags", "phenomenon_count",
                "contributing_tag_count", "bronze_char_length",
                "upstream_start", "upstream_end", "bronze_text_sha256",
                "source_text_sha256", "upstream_content_sha256",
                "canonical_record_sha256", "candidate_catalog_sha256",
                "full_transcript_sha256", "full_transcript_char_length",
                "game",
            ),
            "selection case",
        )
        if case["selection_rank"] != index + 1:
            raise ValueError("selection case ranks are not sequential")


# ---------------------------------------------------------------------------
# Read-only SQLite transcript access
# ---------------------------------------------------------------------------


def open_transcript_db(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def fetch_video_row(
    connection: sqlite3.Connection,
    *,
    source_id: str,
    expected_full_sha256: str,
) -> dict[str, Any]:
    """Validate one source row and return exact transcript + metadata."""
    row = connection.execute(
        "SELECT video_id, video_url, video_title, role, champion, rank, game, "
        "transcription FROM videos WHERE video_id = ?",
        (source_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"source {source_id} is absent from the transcript DB")
    transcript = row["transcription"]
    if not isinstance(transcript, str):
        raise ValueError(f"source {source_id} transcription is not text")
    full_hash = text_sha256(transcript)
    if full_hash != expected_full_sha256:
        raise ValueError(
            f"source {source_id} full transcript SHA does not match the frozen "
            "Phase 2J upstream hash",
        )
    return {
        "source_id": source_id,
        "video_id": source_id,
        "video_url": row["video_url"],
        "video_title": row["video_title"],
        "role": row["role"],
        "champion": row["champion"],
        "rank": row["rank"],
        "game": row["game"],
        "transcript": transcript,
        "transcript_sha256": full_hash,
        "transcript_char_length": len(transcript),
    }


def fetch_transcript_rows(
    connection: sqlite3.Connection,
    selected: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in selected:
        source_id = item["upstream_source_id"]
        row = fetch_video_row(
            connection,
            source_id=source_id,
            expected_full_sha256=item["upstream_content_sha256"],
        )
        rows[source_id] = row
    return rows


def validate_manifest_db_alignment(
    case: Mapping[str, Any],
    row: Mapping[str, Any],
) -> None:
    """Manifest metadata must match the authoritative DB row exactly."""
    metadata = case["metadata"]
    if row["video_title"] != metadata["video_title"]:
        raise ValueError(
            f"source {case['upstream_source_id']} DB video_title does not match "
            "the frozen manifest",
        )
    if row["champion"] != metadata["champion"]:
        raise ValueError(
            f"source {case['upstream_source_id']} DB champion does not match "
            "the frozen manifest",
        )
    if row["role"] != metadata["role"]:
        raise ValueError(
            f"source {case['upstream_source_id']} DB role does not match "
            "the frozen manifest",
        )
    transcript = row["transcript"]
    start = case["upstream_start"]
    end = case["upstream_end"]
    if transcript[start:end] != _bronze_text_for_case(case, transcript):
        raise ValueError("target slice does not round-trip the Bronze text")


def _bronze_text_for_case(case: Mapping[str, Any], transcript: str) -> str:
    # Convenience accessor; the manifest already validated offsets/lengths.
    return transcript[case["upstream_start"]:case["upstream_end"]]


# ---------------------------------------------------------------------------
# B vocabulary: lexical vocabulary v2 + DB champion_abilities rows
# ---------------------------------------------------------------------------


def load_lexical_vocabulary(path: Path) -> dict[str, Any]:
    vocabulary = load_json_strict(path, label="phase2k lexical vocabulary")
    if vocabulary.get("schema_version") != LEAGUE_VOCABULARY_SCHEMA_VERSION:
        raise ValueError("phase2k lexical vocabulary schema version is invalid")
    for key in ("ability_keys", "summoner_spells", "basic_domain_tokens"):
        items = vocabulary.get(key)
        if not isinstance(items, list) or any(
            not isinstance(item, str) for item in items
        ):
            raise ValueError(f"phase2k lexical vocabulary {key} is invalid")
    if not isinstance(vocabulary.get("champion_alias_rules"), dict):
        raise ValueError("phase2k lexical vocabulary champion_alias_rules is invalid")
    return vocabulary


def _champion_word_pattern(champion: str) -> re.Pattern[str]:
    return re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(champion) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def champion_abilities_for_transcript(
    connection: sqlite3.Connection,
    *,
    metadata_champion: str,
    transcript: str,
    video_id: str,
) -> dict[str, Any]:
    """Build the B vocabulary for one case from the DB, preserving provenance.

    Champions are the metadata champion plus any canonical champion names
    literally present in the full transcript (exact word-boundary match).
    Only champion_abilities rows are included: no archetypes, fingerprints,
    strategic relations, labels, or Phase 2K generated bindings.
    """
    names = [
        row[0] for row in connection.execute(
            "SELECT DISTINCT champion FROM champion_abilities ORDER BY champion",
        ).fetchall()
    ]
    named: set[str] = set()
    if metadata_champion in names:
        named.add(metadata_champion)
    for champion in names:
        if _champion_word_pattern(champion).search(transcript):
            named.add(champion)
    selected_champions = sorted(named)
    champions: list[dict[str, Any]] = []
    abilities: list[dict[str, Any]] = []
    for champion in selected_champions:
        reasons = []
        if champion == metadata_champion:
            reasons.append("metadata_named")
        if _champion_word_pattern(champion).search(transcript):
            reasons.append("transcript_literal")
        reasons = sorted(set(reasons))
        provenance = {
            "source": "champion_abilities",
            "selection_reasons": reasons,
        }
        champions.append({
            "champion": champion,
            "selection_reasons": reasons,
            "provenance": dict(provenance),
        })
        rows = connection.execute(
            "SELECT champion, ability_slot, name, description, cooldown, "
            "range, cost, properties FROM champion_abilities "
            "WHERE champion = ? ORDER BY ability_slot",
            (champion,),
        ).fetchall()
        for row in rows:
            abilities.append({
                "champion": row["champion"],
                "ability_slot": row["ability_slot"],
                "name": row["name"],
                "description": row["description"],
                "cooldown": row["cooldown"],
                "range": row["range"],
                "cost": row["cost"],
                "properties": row["properties"],
                "provenance": dict(provenance),
            })
    return {
        "champions": champions,
        "champion_abilities": abilities,
        "selected_champion_count": len(selected_champions),
        "ability_row_count": len(abilities),
    }


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
        "phase2j case vocabulary",
    )
    if vocabulary["schema_version"] != VOCABULARY_SCHEMA_VERSION:
        raise ValueError("case vocabulary schema version is invalid")
    if vocabulary["case_id"] != case_id:
        raise ValueError("case vocabulary case_id is invalid")
    _validate_recomputed_content_hash(vocabulary, label="case vocabulary")
    if canonical_sha256(vocabulary["lexical_vocabulary"]) != (
        vocabulary["lexical_vocabulary_sha256"]
    ):
        raise ValueError("case vocabulary lexical hash is invalid")
    if canonical_sha256(lexical_vocabulary) != (
        vocabulary["lexical_vocabulary_sha256"]
    ):
        raise ValueError("case vocabulary lexical vocabulary does not match v2")
    for champion in _require_list(vocabulary["champions"], "vocabulary champions"):
        if not isinstance(champion, Mapping):
            raise ValueError("vocabulary champion must be an object")
        _require_exact_keys(
            champion,
            ("champion", "selection_reasons", "provenance"),
            "vocabulary champion",
        )
    for ability in _require_list(
        vocabulary["champion_abilities"], "vocabulary abilities",
    ):
        if not isinstance(ability, Mapping):
            raise ValueError("vocabulary ability must be an object")
        _require_exact_keys(
            ability,
            (
                "champion", "ability_slot", "name", "description",
                "cooldown", "range", "cost", "properties", "provenance",
            ),
            "vocabulary ability",
        )


# ---------------------------------------------------------------------------
# Shared extraction instructions (byte-identical for both conditions)
# ---------------------------------------------------------------------------


def build_extraction_instructions() -> dict[str, Any]:
    """Canonical byte-identical extraction instructions for both conditions."""
    return {
        "schema_version": INSTRUCTIONS_SCHEMA_VERSION,
        "version": INSTRUCTIONS_SCHEMA_VERSION,
        "purpose": (
            "Direct source-grounded semantic extraction of the isolated "
            "Bronze target.  Byte-identical for condition A and condition B.  "
            "No mechanical cleaning, contextual rewriting, semantic polish, "
            "or strategic abstraction is permitted."
        ),
        "task": (
            "Extract every item for each of the eight semantic fields from "
            "the supplied target.  Quote exact contiguous source slices and "
            "cite each quote's zero-based occurrence index.  Do not guess, "
            "do not invent actors, abilities, resources, events, conditions, "
            "advice/actions, consequences, or uncertainties that are not "
            "present in the supplied source."
        ),
        "direct_extraction_only": True,
        "no_mechanical_clean": True,
        "no_contextual_rewriting": True,
        "no_semantic_polish": True,
        "no_strategic_abstraction": True,
        "fields": {
            "actors": (
                "Entities that perform or undergo the described action, "
                "including explicit and context-resolved actors."
            ),
            "ability_resource_references": (
                "Explicit ability keys/names and resource references "
                "(mana, cooldowns, summoner spells, items, waves) that are "
                "mentioned in the target."
            ),
            "event": (
                "The described game event(s), including mechanical and "
                "strategic occurrences."
            ),
            "condition": (
                "Any stated condition, requirement, or contingency for the "
                "described play."
            ),
            "advice_action": (
                "Explicit advice and recommended actions addressed to the "
                "player."
            ),
            "consequence": (
                "Stated results or follow-on effects of an action or event."
            ),
            "uncertainty": (
                "Explicit uncertainty, hedging, or unresolved information in "
                "the target."
            ),
            "supporting_source_ranges": (
                "Exact supplied source character ranges that support the "
                "extracted semantic items."
            ),
        },
        "resolution_statuses": list(RESOLUTION_STATUSES),
        "resolution_rule": (
            "literal_explicit: directly stated; context_resolved: resolved "
            "only from the supplied context; vocabulary_supported: resolved "
            "using the supplied vocabulary; unresolved: present but not "
            "resolvable."
        ),
        "reference_rule": (
            "Every extracted item must cite at least one exact source "
            "reference carrying an exact contiguous quote and a zero-based "
            "occurrence_index.  The quote must occur verbatim in the "
            "supplied source; a deterministic importer resolves "
            "occurrence_index against all exact substring matches and "
            "computes the byte-exact [char_start, char_end) range.  Do not "
            "count, estimate, or return character offsets.  Condition A "
            "quotes are resolved against the supplied Bronze target; "
            "condition B quotes are resolved against the supplied full "
            "transcript."
        ),
        "source_range_rule": (
            "supporting_source_ranges items cite exact contiguous quotes "
            "with zero-based occurrence_index values on their source "
            "references; "
            "the importer derives the item source_range as the minimal "
            "bounding [char_start, char_end) range of the resolved "
            "references."
        ),
        "output_rules": [
            "Empty field lists are allowed when nothing is present.",
            "unresolved and uncertainty items are allowed.",
            "Unsupported guessing is forbidden.",
            "Every item must carry extraction_text, resolution_status, and "
            "source_references; every source reference must carry an exact "
            "contiguous quote and a zero-based occurrence_index.",
        ],
    }


def validate_extraction_instructions(instructions: Mapping[str, Any]) -> None:
    _require_exact_keys(
        instructions,
        (
            "schema_version", "version", "purpose", "task",
            "direct_extraction_only", "no_mechanical_clean",
            "no_contextual_rewriting", "no_semantic_polish",
            "no_strategic_abstraction", "fields", "resolution_statuses",
            "resolution_rule", "reference_rule", "source_range_rule",
            "output_rules",
        ),
        "phase2j extraction instructions",
    )
    if instructions["schema_version"] != INSTRUCTIONS_SCHEMA_VERSION:
        raise ValueError("extraction instructions schema version is invalid")
    if not isinstance(instructions["fields"], Mapping) or set(
        instructions["fields"],
    ) != set(SEMANTIC_FIELDS):
        raise ValueError("extraction instructions fields are invalid")
    if tuple(instructions["resolution_statuses"]) != RESOLUTION_STATUSES:
        raise ValueError("extraction instructions resolution statuses are invalid")


def build_instructions_artifact() -> dict[str, Any]:
    return _envelope(dict(build_extraction_instructions()))


def validate_instructions_artifact(artifact: Mapping[str, Any]) -> None:
    _validate_recomputed_content_hash(artifact, label="instructions artifact")
    validate_extraction_instructions({
        key: value for key, value in artifact.items()
        if key != "content_sha256"
    })


# ---------------------------------------------------------------------------
# Condition payloads (A: isolated Bronze + instructions only)
# ---------------------------------------------------------------------------


def build_condition_payloads(
    *,
    cases: list[Mapping[str, Any]],
    transcript_rows: Mapping[str, Mapping[str, Any]],
    vocabulary_by_case: Mapping[str, Mapping[str, Any]],
    instructions: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build the model-visible A/B payloads plus outer provenance bindings."""
    instructions_sha256 = canonical_sha256(instructions)
    payload_cases: list[dict[str, Any]] = []
    provenance_by_case: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = case["case_id"]
        source_id = case["upstream_source_id"]
        row = transcript_rows[source_id]
        transcript = row["transcript"]
        bronze_start = case["upstream_start"]
        bronze_end = case["upstream_end"]
        bronze_text = transcript[bronze_start:bronze_end]
        common_target = {
            "bronze_text": bronze_text,
            "bronze_text_sha256": text_sha256(bronze_text),
            "bronze_char_length": len(bronze_text),
        }
        payload_a = _envelope({
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "condition": "A",
            "case_id": case_id,
            "selection_rank": case["selection_rank"],
            "target": dict(common_target),
            "instructions": dict(instructions),
            "instructions_sha256": instructions_sha256,
        })
        payload_b = _envelope({
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "condition": "B",
            "case_id": case_id,
            "selection_rank": case["selection_rank"],
            "target": dict(common_target),
            "transcript": transcript,
            "target_char_start": bronze_start,
            "target_char_end": bronze_end,
            "metadata": {
                "video_title": row["video_title"],
                "champion": row["champion"],
                "role": row["role"],
                "rank": row["rank"],
                "game": row["game"],
            },
            "vocabulary": dict(vocabulary_by_case[case_id]),
            "vocabulary_sha256": vocabulary_by_case[case_id]["content_sha256"],
            "instructions": dict(instructions),
            "instructions_sha256": instructions_sha256,
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
            "full_transcript_sha256": row["transcript_sha256"],
            "full_transcript_char_length": row["transcript_char_length"],
            "target_char_start": bronze_start,
            "target_char_end": bronze_end,
            "vocabulary_sha256": vocabulary_by_case[case_id]["content_sha256"],
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
            "Frozen Phase 2J context-ablation condition payloads: A = "
            "isolated Bronze plus extraction instructions; B = exact same "
            "Bronze target plus full archived transcript with target "
            "character offsets, useful ordinary metadata, and League "
            "champion/ability vocabulary.  Both conditions bind the same "
            "extraction instructions hash.  Source identity provenance is "
            "retained at this outer artifact level, not inside the "
            "model-visible payloads."
        ),
        "release_gate": RELEASE_GATE_LOCKED,
        "selection_manifest_sha256": selection["content_sha256"],
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


def _validate_source_range(
    value: Mapping[str, Any],
    *,
    source: str,
    label: str,
) -> None:
    _require_exact_keys(
        value, ("char_start", "char_end"), f"{label} source_range",
    )
    start = _require_int(
        value["char_start"], f"{label} source_range char_start", minimum=0,
    )
    end = _require_int(
        value["char_end"], f"{label} source_range char_end", minimum=1,
    )
    if end <= start:
        raise ValueError(f"{label} source_range must have char_end > char_start")
    if end > len(source):
        raise ValueError(f"{label} source_range exceeds the supplied source length")


def _validate_condition_payload(
    payload: Mapping[str, Any],
    *,
    expected_case_id: str,
    expected_condition: str,
    instructions: Mapping[str, Any],
    lexical_vocabulary: Mapping[str, Any],
) -> None:
    if expected_condition == "B":
        _require_exact_keys(
            payload,
            (
                "schema_version", "condition", "case_id", "selection_rank",
                "target", "transcript", "target_char_start",
                "target_char_end", "metadata", "vocabulary",
                "vocabulary_sha256", "instructions", "instructions_sha256",
                "content_sha256",
            ),
            "phase2j condition B payload",
        )
    else:
        _require_exact_keys(
            payload,
            (
                "schema_version", "condition", "case_id", "selection_rank",
                "target", "instructions", "instructions_sha256",
                "content_sha256",
            ),
            "phase2j condition A payload",
        )
    if payload["schema_version"] != PAYLOAD_SCHEMA_VERSION:
        raise ValueError("condition payload schema version is invalid")
    _validate_recomputed_content_hash(payload, label="condition payload")
    if payload["condition"] != expected_condition:
        raise ValueError("condition payload condition is invalid")
    if payload["case_id"] != expected_case_id:
        raise ValueError("condition payload case_id is invalid")
    if canonical_sha256(payload["instructions"]) != payload["instructions_sha256"]:
        raise ValueError("condition payload instructions hash is invalid")
    if canonical_sha256(instructions) != payload["instructions_sha256"]:
        raise ValueError(
            "condition payload instructions do not match the canonical object",
        )
    target = payload["target"]
    if not isinstance(target, Mapping):
        raise ValueError("condition payload target must be an object")
    _require_exact_keys(
        target,
        ("bronze_text", "bronze_text_sha256", "bronze_char_length"),
        "condition payload target",
    )
    bronze_text = _require_nonempty_string(target["bronze_text"], "payload bronze_text")
    if target["bronze_text_sha256"] != text_sha256(bronze_text):
        raise ValueError("condition payload bronze_text_sha256 is invalid")
    if target["bronze_char_length"] != len(bronze_text):
        raise ValueError("condition payload bronze_char_length is invalid")
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
            raise ValueError("condition B target character offsets are invalid")
        if transcript[start:end] != bronze_text:
            raise ValueError(
                "condition B target offsets do not slice the supplied "
                "transcript to the Bronze text",
            )
        metadata = payload["metadata"]
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "video_title", "champion", "role", "rank", "game",
        }:
            raise ValueError("condition B payload metadata is invalid")
        for key in ("video_title", "champion", "role", "game"):
            if not isinstance(metadata[key], str):
                raise ValueError(
                    f"condition B payload metadata {key} must be a string",
                )
        if metadata["rank"] is not None and not isinstance(metadata["rank"], str):
            raise ValueError("condition B payload metadata rank must be a string or null")
        if payload["vocabulary_sha256"] != payload["vocabulary"]["content_sha256"]:
            raise ValueError("condition B payload vocabulary hash is invalid")
        validate_case_vocabulary(
            payload["vocabulary"],
            case_id=expected_case_id,
            lexical_vocabulary=lexical_vocabulary,
        )
    else:
        _scan_forbidden_payload_keys(payload, condition="A")


_A_FORBIDDEN_PAYLOAD_KEYS = frozenset({
    "transcript", "metadata", "vocabulary", "champion_abilities",
    "caption_segments", "captions", "timestamps", "timestamp",
    "video_id", "video_url", "source_id", "source_group_id", "window_id",
    "source_coordinates", "provenance", "full_transcript_sha256",
    "full_transcript_char_length", "target_char_start", "target_char_end",
    "phenomena", "partition", "endpoints", "gold", "labels", "predictions",
    "human_review", "materiality", "archetypes", "fingerprints", "strategic",
    "insights", "phase2k",
})

_B_FORBIDDEN_PAYLOAD_KEYS = frozenset({
    "caption_segments", "captions", "timestamps", "timestamp",
    "video_id", "video_url", "source_id", "source_group_id", "window_id",
    "source_coordinates", "full_transcript_sha256",
    "full_transcript_char_length", "target_char_start", "target_char_end",
    "phenomena", "partition", "endpoints", "gold", "labels", "predictions",
    "human_review", "materiality", "archetypes", "fingerprints", "strategic",
    "insights", "phase2k",
})


def _scan_forbidden_payload_keys(
    value: object,
    *,
    condition: str,
    path: str = "payload",
) -> None:
    """Reject structural boundary leaks in condition payloads."""
    forbidden = (
        _A_FORBIDDEN_PAYLOAD_KEYS
        if condition == "A"
        else _B_FORBIDDEN_PAYLOAD_KEYS
    )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in forbidden:
                raise ValueError(
                    f"condition {condition} payload leaks forbidden key "
                    f"{key!r} at {path}",
                )
            _scan_forbidden_payload_keys(
                item, condition=condition, path=f"{path}.{key}",
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_forbidden_payload_keys(
                item, condition=condition, path=f"{path}[{index}]",
            )


def validate_payloads_artifact(
    artifact: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    instructions: Mapping[str, Any],
    lexical_vocabulary: Mapping[str, Any],
    manifest_path: Path,
    packet_path: Path,
    manifest: Mapping[str, Any],
    packet: Mapping[str, Any],
    db_path: Path,
) -> None:
    """Fail-closed canonical validation of the payloads artifact.

    Rebuilds the exact canonical payloads from the frozen manifest, the
    read-only DB, the shared instructions, and the lexical vocabulary, then
    requires exact equality.  Self-rehashed tampering therefore cannot pass.
    """
    _require_exact_keys(
        artifact,
        (
            "schema_version", "purpose", "release_gate",
            "selection_manifest_sha256", "instructions_sha256",
            "provenance_by_case", "cases", "content_sha256",
        ),
        "phase2j condition payloads",
    )
    if artifact["schema_version"] != PAYLOADS_SCHEMA_VERSION:
        raise ValueError("condition payloads schema version is invalid")
    if artifact["release_gate"] != RELEASE_GATE_LOCKED:
        raise ValueError("condition payloads artifact is not LOCKED")
    _validate_recomputed_content_hash(artifact, label="condition payloads")
    if artifact["selection_manifest_sha256"] != selection["content_sha256"]:
        raise ValueError("condition payloads are not bound to the selection")
    if artifact["instructions_sha256"] != canonical_sha256(instructions):
        raise ValueError("condition payloads instructions hash is invalid")
    cases = _require_list(artifact["cases"], "payload cases")
    if len(cases) != SELECTION_COUNT:
        raise ValueError("condition payloads must contain exactly 10 cases")
    selection_cases = selection["cases"]
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError("payload case must be an object")
        _require_exact_keys(
            case, ("case_id", "selection_rank", "A", "B"), "payload case",
        )
        if case["selection_rank"] != index + 1:
            raise ValueError("payload case ranks are not sequential")
        if case["case_id"] != selection_cases[index]["case_id"]:
            raise ValueError(
                "payload case IDs do not match the canonical selection order",
            )
        _validate_condition_payload(
            case["A"],
            expected_case_id=case["case_id"],
            expected_condition="A",
            instructions=instructions,
            lexical_vocabulary=lexical_vocabulary,
        )
        _validate_condition_payload(
            case["B"],
            expected_case_id=case["case_id"],
            expected_condition="B",
            instructions=instructions,
            lexical_vocabulary=lexical_vocabulary,
        )
        if case["A"]["target"] != case["B"]["target"]:
            raise ValueError(
                "condition A and B payload targets are not byte-identical",
            )
    provenance = artifact["provenance_by_case"]
    if not isinstance(provenance, Mapping):
        raise ValueError("condition payloads provenance_by_case must be an object")
    if set(provenance) != {case["case_id"] for case in cases}:
        raise ValueError("condition payloads provenance does not cover every case")
    for case_id, entry in provenance.items():
        if not isinstance(entry, Mapping):
            raise ValueError("condition payload provenance entry must be an object")
        _require_exact_keys(
            entry,
            (
                "video_id", "video_url", "source_group_id", "window_id",
                "full_transcript_sha256", "full_transcript_char_length",
                "target_char_start", "target_char_end", "vocabulary_sha256",
            ),
            "condition payload provenance",
        )
    # Canonical rebuild from the frozen manifest + read-only DB + vocabulary.
    connection = open_transcript_db(db_path)
    try:
        rows = fetch_transcript_rows(connection, manifest["selected"])
    finally:
        connection.close()
    canonical_cases = select_cases(manifest, transcript_rows=rows)
    connection = open_transcript_db(db_path)
    try:
        vocabulary_by_case: dict[str, dict[str, Any]] = {}
        for case in canonical_cases:
            source_id = case["upstream_source_id"]
            validate_manifest_db_alignment(case, rows[source_id])
            champion_data = champion_abilities_for_transcript(
                connection,
                metadata_champion=case["metadata"]["champion"],
                transcript=rows[source_id]["transcript"],
                video_id=source_id,
            )
            vocabulary_by_case[case["case_id"]] = build_case_vocabulary(
                case_id=case["case_id"],
                lexical_vocabulary=lexical_vocabulary,
                champion_data=champion_data,
            )
    finally:
        connection.close()
    canonical_payload_cases, canonical_provenance = build_condition_payloads(
        cases=canonical_cases,
        transcript_rows=rows,
        vocabulary_by_case=vocabulary_by_case,
        instructions=instructions,
    )
    expected = build_payloads_artifact(
        selection=selection,
        instructions=instructions,
        payload_cases=canonical_payload_cases,
        provenance_by_case=canonical_provenance,
    )
    if dict(artifact) != dict(expected):
        raise ValueError(
            "condition payloads do not match the canonical build from the "
            "frozen manifest, DB, instructions, and vocabulary",
        )


# ---------------------------------------------------------------------------
# Strict extraction-output schema and validators
# ---------------------------------------------------------------------------


OUTPUT_ITEM_ID_PATTERN = re.compile(
    r"^p2ja:case:[0-9]{4}:([AB]):"
    r"(actors|ability_resource_references|event|condition|advice_action|"
    r"consequence|uncertainty|supporting_source_ranges):[0-9]{4}$",
)

FORBIDDEN_OUTPUT_KEYS = frozenset({
    "metadata", "vocabulary", "champion_abilities", "full_transcript",
    "transcript", "caption_segments", "captions", "timestamps", "timestamp",
    "phenomena", "partition", "endpoints", "gold", "labels", "predictions",
    "human_review", "materiality", "archetypes", "fingerprints", "strategic",
    "insights", "phase2k", "instructions", "prompt", "system_prompt",
    "raw_response", "payload", "video_id", "video_url", "source_id",
    "source_group_id", "window_id", "provenance",
})


def _scan_forbidden_output_keys(value: object, *, path: str = "output") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_OUTPUT_KEYS:
                raise ValueError(
                    f"extraction output leaks forbidden key {key!r} at {path}",
                )
            _scan_forbidden_output_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_forbidden_output_keys(item, path=f"{path}[{index}]")


def _validate_output_item(
    item: Mapping[str, Any],
    *,
    field: str,
    case_id: str,
    condition: str,
    payload: Mapping[str, Any],
    item_index: int,
) -> None:
    expected_keys = {
        "item_id", "extraction_text", "resolution_status", "source_references",
    }
    if field == "supporting_source_ranges":
        expected_keys.add("source_range")
    _require_exact_keys(item, expected_keys, "extraction output item")
    item_id = _require_nonempty_string(item["item_id"], "output item_id")
    match = OUTPUT_ITEM_ID_PATTERN.fullmatch(item_id)
    if match is None:
        raise ValueError(f"output item_id {item_id!r} is malformed")
    if match.group(1) != condition or match.group(2) != field:
        raise ValueError(
            f"output item_id {item_id!r} does not match case/condition/field",
        )
    expected_id = f"{case_id}:{condition}:{field}:{item_index:04d}"
    if item_id != expected_id:
        raise ValueError(
            f"output item_id {item_id!r} does not match sequential position "
            f"{expected_id!r}",
        )
    extraction_text = _require_nonempty_string(
        item["extraction_text"], "output extraction_text",
    )
    _require_enum(
        item["resolution_status"], RESOLUTION_STATUSES,
        "output resolution_status",
    )
    source = _payload_source_text(payload)
    references = _require_list(item["source_references"], "output source_references")
    if not references:
        raise ValueError("output item must cite at least one source reference")
    for reference in references:
        if not isinstance(reference, Mapping):
            raise ValueError("output source_reference must be an object")
        _require_exact_keys(
            reference, ("quote", "source_range"), "output source_reference",
        )
        quote = _require_nonempty_string(reference["quote"], "output quote")
        _validate_source_range(
            reference["source_range"],
            source=source,
            label="output source_reference",
        )
        char_start = reference["source_range"]["char_start"]
        char_end = reference["source_range"]["char_end"]
        if source[char_start:char_end] != quote:
            raise ValueError(
                "output quote is not byte-for-byte equal to the supplied "
                "source slice at its source_range",
            )
    if field == "supporting_source_ranges":
        item_range = item["source_range"]
        if not isinstance(item_range, Mapping):
            raise ValueError("output source_range must be an object")
        _validate_source_range(
            item_range,
            source=source,
            label="output supporting source_range",
        )
        referenced_starts = [
            reference["source_range"]["char_start"] for reference in references
        ]
        referenced_ends = [
            reference["source_range"]["char_end"] for reference in references
        ]
        if item_range["char_start"] < min(referenced_starts) or \
                item_range["char_end"] > max(referenced_ends):
            raise ValueError(
                "output supporting source_range is outside the cited "
                "references' union",
            )
    # Keep the concise extraction text non-empty and bounded sanity check.
    if len(extraction_text) > 2000:
        raise ValueError("output extraction_text exceeds the 2000-char bound")


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
        "phase2j extraction output",
    )
    if output["schema_version"] != OUTPUT_SCHEMA_VERSION:
        raise ValueError("extraction output schema version is invalid")
    _validate_recomputed_content_hash(output, label="extraction output")
    if output["case_id"] != case_id:
        raise ValueError("extraction output case_id is invalid")
    if output["condition"] != condition:
        raise ValueError("extraction output condition is invalid")
    if output["payload_sha256"] != payload["content_sha256"]:
        raise ValueError("extraction output is not bound to the condition payload")
    if output["instructions_sha256"] != payload["instructions_sha256"]:
        raise ValueError("extraction output instructions hash is invalid")
    fields = output["fields"]
    if not isinstance(fields, Mapping) or set(fields) != set(SEMANTIC_FIELDS):
        raise ValueError("extraction output fields are invalid")
    _scan_forbidden_output_keys(output)
    seen_ids: set[str] = set()
    for field in SEMANTIC_FIELDS:
        items = _require_list(fields[field], f"output field {field}")
        for item_index, item in enumerate(items, 1):
            if not isinstance(item, Mapping):
                raise ValueError("extraction output item must be an object")
            _validate_output_item(
                item,
                field=field,
                case_id=case_id,
                condition=condition,
                payload=payload,
                item_index=item_index,
            )
            item_id = item["item_id"]
            if item_id in seen_ids:
                raise ValueError(f"duplicate output item_id {item_id!r}")
            seen_ids.add(item_id)


def validate_extraction_outputs_bundle(
    bundle: Mapping[str, Any],
    *,
    payloads_artifact: Mapping[str, Any],
    instructions: Mapping[str, Any],
) -> None:
    """Validate the full 10-case A/B output bundle in frozen case order."""
    _require_exact_keys(
        bundle,
        (
            "schema_version", "purpose", "release_gate",
            "payloads_sha256", "instructions_sha256", "cases",
            "content_sha256",
        ),
        "phase2j extraction outputs",
    )
    if bundle["schema_version"] != OUTPUTS_SCHEMA_VERSION:
        raise ValueError("extraction outputs schema version is invalid")
    if bundle["release_gate"] != RELEASE_GATE_LOCKED:
        raise ValueError("extraction outputs are not LOCKED")
    _validate_recomputed_content_hash(bundle, label="extraction outputs")
    if bundle["payloads_sha256"] != payloads_artifact["content_sha256"]:
        raise ValueError("extraction outputs are not bound to the payloads")
    if bundle["instructions_sha256"] != canonical_sha256(instructions):
        raise ValueError("extraction outputs instructions hash is invalid")
    cases = _require_list(bundle["cases"], "extraction output cases")
    if len(cases) != SELECTION_COUNT:
        raise ValueError("extraction outputs must contain exactly 10 cases")
    payload_cases = payloads_artifact["cases"]
    expected_case_ids = [case["case_id"] for case in payload_cases]
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError("extraction output case must be an object")
        _require_exact_keys(
            case, ("case_id", "A", "B"), "extraction output case",
        )
        if case["case_id"] != expected_case_ids[index]:
            raise ValueError(
                "extraction output cases are out of order or have wrong IDs; "
                f"expected {expected_case_ids[index]!r}",
            )
        payload_case = payload_cases[index]
        for condition in CONDITION_CODES:
            validate_extraction_output(
                case[condition],
                case_id=case["case_id"],
                condition=condition,
                payload=payload_case[condition],
            )


def build_outputs_bundle(
    *,
    payloads_artifact: Mapping[str, Any],
    instructions: Mapping[str, Any],
    outputs_by_case: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    cases = []
    for payload_case in payloads_artifact["cases"]:
        case_id = payload_case["case_id"]
        entry = outputs_by_case.get(case_id)
        if entry is None:
            raise ValueError(f"extraction outputs missing case {case_id}")
        cases.append({
            "case_id": case_id,
            "A": entry["A"],
            "B": entry["B"],
        })
    bundle = _envelope({
        "schema_version": OUTPUTS_SCHEMA_VERSION,
        "purpose": (
            "Validated Phase 2J context-ablation extraction outputs, one "
            "validated A and B output per frozen case."
        ),
        "release_gate": RELEASE_GATE_LOCKED,
        "payloads_sha256": payloads_artifact["content_sha256"],
        "instructions_sha256": canonical_sha256(instructions),
        "cases": cases,
    })
    validate_extraction_outputs_bundle(
        bundle,
        payloads_artifact=payloads_artifact,
        instructions=instructions,
    )
    return bundle


# ---------------------------------------------------------------------------
# Intermediate Sol response schema and deterministic importer
# ---------------------------------------------------------------------------


def build_sol_intermediate_schema() -> dict[str, Any]:
    """Canonical JSON Schema for the model-visible intermediate response.

    The model returns exact contiguous quotes plus zero-based occurrence
    indexes only; the deterministic importer resolves byte-exact
    ``[char_start, char_end)`` ranges and derives item IDs and supporting
    source ranges.  Every object is closed (``additionalProperties=false``)
    and empty field lists are allowed.
    """
    reference_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["quote", "occurrence_index"],
        "properties": {
            "quote": {"type": "string", "minLength": 1},
            "occurrence_index": {"type": "integer", "minimum": 0},
        },
    }
    item_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "extraction_text", "resolution_status", "source_references",
        ],
        "properties": {
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
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Phase 2J context-ablation Sol intermediate response",
        "schema_version": SOL_INTERMEDIATE_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "case_id", "condition", "payload_sha256",
            "instructions_sha256", "fields",
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": SOL_INTERMEDIATE_SCHEMA_VERSION,
            },
            "case_id": {
                "type": "string",
                "pattern": r"^p2ja:case:[0-9]{4}$",
            },
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
                    field: {
                        "type": "array",
                        "items": item_schema,
                    }
                    for field in SEMANTIC_FIELDS
                },
            },
        },
    }


def validate_sol_intermediate_schema(schema: Mapping[str, Any]) -> None:
    if dict(schema) != build_sol_intermediate_schema():
        raise ValueError("sol intermediate schema is not the canonical schema")


def validate_sol_intermediate_response(
    response: Mapping[str, Any],
    *,
    case_id: str,
    condition: str,
    payload: Mapping[str, Any],
) -> None:
    """Strictly validate a model-visible intermediate Sol response."""
    _require_exact_keys(
        response,
        (
            "schema_version", "case_id", "condition", "payload_sha256",
            "instructions_sha256", "fields",
        ),
        "phase2j sol intermediate response",
    )
    if response["schema_version"] != SOL_INTERMEDIATE_SCHEMA_VERSION:
        raise ValueError("sol intermediate response schema version is invalid")
    if response["case_id"] != case_id:
        raise ValueError("sol intermediate response case_id is invalid")
    if response["condition"] != condition:
        raise ValueError("sol intermediate response condition is invalid")
    if response["payload_sha256"] != payload["content_sha256"]:
        raise ValueError(
            "sol intermediate response is not bound to the condition payload",
        )
    if response["instructions_sha256"] != payload["instructions_sha256"]:
        raise ValueError("sol intermediate response instructions hash is invalid")
    fields = response["fields"]
    if not isinstance(fields, Mapping) or set(fields) != set(SEMANTIC_FIELDS):
        raise ValueError("sol intermediate response fields are invalid")
    _scan_forbidden_output_keys(response)
    for field in SEMANTIC_FIELDS:
        items = _require_list(fields[field], f"sol intermediate field {field}")
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError("sol intermediate item must be an object")
            _require_exact_keys(
                item,
                ("extraction_text", "resolution_status", "source_references"),
                "sol intermediate item",
            )
            extraction_text = _require_nonempty_string(
                item["extraction_text"], "sol intermediate extraction_text",
            )
            if len(extraction_text) > 2000:
                raise ValueError(
                    "sol intermediate extraction_text exceeds the 2000-char bound",
                )
            _require_enum(
                item["resolution_status"], RESOLUTION_STATUSES,
                "sol intermediate resolution_status",
            )
            references = _require_list(
                item["source_references"], "sol intermediate source_references",
            )
            if not references:
                raise ValueError(
                    "sol intermediate item must cite at least one source reference",
                )
            for reference in references:
                if not isinstance(reference, Mapping):
                    raise ValueError("sol intermediate source_reference must be an object")
                _require_exact_keys(
                    reference,
                    ("quote", "occurrence_index"),
                    "sol intermediate source_reference",
                )
                _require_nonempty_string(
                    reference["quote"], "sol intermediate quote",
                )
                _require_int(
                    reference["occurrence_index"],
                    "sol intermediate occurrence_index",
                    minimum=0,
                )


def _exact_quote_occurrences(source: str, quote: str) -> list[int]:
    """Start offsets of all exact non-overlapping substring matches."""
    matches: list[int] = []
    start = 0
    while True:
        index = source.find(quote, start)
        if index < 0:
            break
        matches.append(index)
        start = index + len(quote)
    return matches


def import_sol_intermediate_response(
    response: Mapping[str, Any],
    *,
    case_id: str,
    condition: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministically import an intermediate Sol response into an output.

    Resolves every exact quote against the condition source, derives
    byte-exact ``[char_start, char_end)`` ranges, assigns sequential item
    IDs, derives ``supporting_source_ranges`` item ranges as the minimal
    bounding range of the resolved references, and validates the final
    output.  Extraction text, resolution statuses, quotes, field
    membership, item order, and item counts are never altered.
    """
    validate_sol_intermediate_response(
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
                    raise ValueError(
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
            if field == "supporting_source_ranges":
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


# ---------------------------------------------------------------------------
# Blinded human review packet and sealed mapping
# ---------------------------------------------------------------------------


def _new_blinded_labels(
    rng: random.Random,
    count: int,
    used: set[str],
) -> list[str]:
    labels: list[str] = []
    while len(labels) < count:
        label = f"BLIND-{rng.getrandbits(32):08x}"
        if label not in used:
            used.add(label)
            labels.append(label)
    return labels


def _build_source_evidence(
    transcript_rows: Mapping[str, Mapping[str, Any]],
    payloads_artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for case in payloads_artifact["cases"]:
        case_id = case["case_id"]
        provenance = payloads_artifact["provenance_by_case"][case_id]
        row = transcript_rows[provenance["video_id"]]
        bronze = row["transcript"][
            provenance["target_char_start"]:provenance["target_char_end"]
        ]
        evidence.append({
            "case_id": case_id,
            "source_id": provenance["video_id"],
            "transcript": row["transcript"],
            "transcript_sha256": row["transcript_sha256"],
            "target_text": bronze,
            "target_text_sha256": text_sha256(bronze),
            "target_char_start": provenance["target_char_start"],
            "target_char_end": provenance["target_char_end"],
        })
    return evidence


def _source_evidence_index_for_case(
    packet: Mapping[str, Any],
    case_id: str,
) -> int:
    evidence = packet["source_evidence"]
    for index, entry in enumerate(evidence):
        if entry["case_id"] == case_id:
            return index
    raise ValueError(f"source evidence is missing case {case_id}")


def _review_presentation(
    *,
    case_id: str,
    field: str,
    output: Mapping[str, Any],
    payload: Mapping[str, Any],
    source_evidence_index: int,
) -> dict[str, Any]:
    # Condition-neutral presentation: both A and B point to the identical
    # shared full-transcript source evidence for the case.
    items = []
    for index, item in enumerate(output["fields"][field], 1):
        presentation_item = {
            "item_index": index,
            "extraction_text": item["extraction_text"],
            "resolution_status": item["resolution_status"],
            "source_references": item["source_references"],
        }
        if field == "supporting_source_ranges":
            presentation_item["source_range"] = item["source_range"]
        items.append(presentation_item)
    return {
        "schema_version": REVIEW_PRESENTATION_SCHEMA_VERSION,
        "case_id": case_id,
        "field": field,
        "target_text": payload["target"]["bronze_text"],
        "target_text_sha256": payload["target"]["bronze_text_sha256"],
        "source_evidence_index": source_evidence_index,
        "extraction_items": items,
    }


def build_human_review_packet(
    *,
    outputs_artifact: Mapping[str, Any],
    payloads_artifact: Mapping[str, Any],
    transcript_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the blinded review packet plus the separately sealed mapping."""
    source_evidence = _build_source_evidence(
        transcript_rows, payloads_artifact,
    )
    rng = random.Random(_REVIEW_LABEL_SEED)
    items: list[dict[str, Any]] = []
    mapping_entries: dict[str, Any] = {}
    used_labels: set[str] = set()
    payload_cases = payloads_artifact["cases"]
    output_cases = outputs_artifact["cases"]
    for output_case, payload_case in zip(output_cases, payload_cases):
        case_id = output_case["case_id"]
        if payload_case["case_id"] != case_id:
            raise ValueError("outputs/payloads case order is misaligned")
        for condition in CONDITION_CODES:
            output = output_case[condition]
            payload = payload_case[condition]
            labels = _new_blinded_labels(rng, len(SEMANTIC_FIELDS), used_labels)
            for field, label in zip(SEMANTIC_FIELDS, labels):
                presentation = _review_presentation(
                    case_id=case_id,
                    field=field,
                    output=output,
                    payload=payload,
                    source_evidence_index=_source_evidence_index_for_case(
                        {"source_evidence": source_evidence}, case_id,
                    ),
                )
                review_item_id = f"p2ja:hr:{case_id}:{label}:{field}"
                items.append({
                    "review_item_id": review_item_id,
                    "case_id": case_id,
                    "blinded_label": label,
                    "field": field,
                    "presentation": presentation,
                    "output_sha256": output["content_sha256"],
                    "scores": {
                        "correctness": None,
                        "unsupported_inference": None,
                        "source_grounding": None,
                        "notes": [],
                    },
                    "reviewer": None,
                    "completed_at": None,
                })
                mapping_entries[review_item_id] = {
                    "case_id": case_id,
                    "condition_code": condition,
                    "blinded_label": label,
                    "field": field,
                    "output_sha256": output["content_sha256"],
                    "presentation_sha256": canonical_sha256(presentation),
                    "target_text_sha256": presentation["target_text_sha256"],
                }
    # Deterministically shuffle the review-item order after construction so
    # the condition ordering is not revealed.  The shuffle seed is a private
    # code constant and is never exposed in the reviewer-visible packet.
    random.Random(_REVIEW_ORDER_SEED).shuffle(items)
    mapping_obj = _envelope({
        "schema_version": HUMAN_MAPPING_SCHEMA_VERSION,
        "purpose": (
            "Separately retained Phase 2J context-ablation human-review label "
            "mapping.  This artifact carries the exact condition provenance "
            "that the official blinded packet must not expose."
        ),
        "entries": dict(sorted(mapping_entries.items())),
    })
    packet = _envelope({
        "schema_version": HUMAN_PACKET_SCHEMA_VERSION,
        "purpose": (
            "Blinded Phase 2J context-ablation human review packet.  "
            "Condition labels are randomized and no condition code or "
            "blinding seed appears in this packet; the label-to-condition "
            "mapping is retained in a separate sealed artifact bound here "
            "only by hash.  One shared full-transcript source evidence entry "
            "per case lets the reviewer check context-resolved outputs and "
            "every cited quote.  Per example/condition/field the reviewer "
            "scores correctness, unsupported inference, source grounding, "
            "and notes."
        ),
        "release_gate": RELEASE_GATE_AWAITING_REVIEW,
        "blinding": {
            "method": "deterministic_private_condition_labels",
            "review_item_order": "deterministic_shuffle_after_construction",
            "seed_visible_to_reviewer": False,
            "mapping_file": OUTPUT_FILENAMES["human_mapping"],
            "mapping_sha256": mapping_obj["content_sha256"],
        },
        "source_evidence": source_evidence,
        "review_items": items,
        "review_fields": list(REVIEW_FIELDS),
        "semantic_fields": list(SEMANTIC_FIELDS),
        "value_options": {
            "correctness": list(CORRECTNESS_VALUES),
            "unsupported_inference": list(UNSUPPORTED_INFERENCE_VALUES),
            "source_grounding": list(SOURCE_GROUNDING_VALUES),
        },
        "rubric": {
            "correctness": (
                "CORRECT: extraction item is fully correct; PARTIAL: partly "
                "correct; INCORRECT: wrong; ABSENT_CORRECTLY: correctly absent."
            ),
            "unsupported_inference": (
                "NONE: no unsupported inference; MINOR: minor unsupported "
                "inference; MAJOR: major unsupported inference."
            ),
            "source_grounding": (
                "GROUNDED: citations exactly ground the item; PARTIAL: partly "
                "grounded; UNGROUNDED: not grounded; NOT_APPLICABLE: no "
                "source grounding is applicable."
            ),
        },
    })
    packet["content_sha256"] = canonical_sha256({
        key: value for key, value in packet.items() if key != "content_sha256"
    })
    return packet, mapping_obj


FORBIDDEN_PACKET_KEYS = frozenset({
    "condition_code", "condition", "record_type", "provenance", "kind",
    "payload", "payloads", "materiality", "metadata", "vocabulary",
    "full_transcript", "caption_segments", "captions", "timestamps",
    "timestamp", "phenomena", "partition", "endpoints", "gold", "labels",
    "predictions", "phase2k", "summary", "seed", "video_url",
})

_FORBIDDEN_PACKET_VALUE_PATTERN = re.compile(r"^(A|B)$")
_PACKET_FREE_TEXT_KEYS = frozenset({
    "text", "notes", "quote", "extraction_text", "target_text", "transcript",
})


def _scan_packet_forbidden_leaks(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_PACKET_KEYS:
                raise ValueError(
                    f"human review packet leaks forbidden key {key!r} at {path}",
                )
            if key in _PACKET_FREE_TEXT_KEYS:
                continue
            _scan_packet_forbidden_leaks(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_packet_forbidden_leaks(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if _FORBIDDEN_PACKET_VALUE_PATTERN.fullmatch(value):
            raise ValueError(
                f"human review packet leaks forbidden value {value!r} at {path}",
            )


def _scan_packet_for_seed_key(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "seed":
                raise ValueError(
                    f"human review packet exposes a blinding seed at {path}.seed",
                )
            _scan_packet_for_seed_key(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_packet_for_seed_key(item, path=f"{path}[{index}]")


def _validate_source_evidence(
    evidence: list[dict[str, Any]],
) -> None:
    if len(evidence) != SELECTION_COUNT:
        raise ValueError("source evidence must contain exactly 10 entries")
    seen_cases: set[str] = set()
    for index, entry in enumerate(evidence):
        if not isinstance(entry, Mapping):
            raise ValueError("source evidence entry must be an object")
        _require_exact_keys(
            entry,
            (
                "case_id", "source_id", "transcript", "transcript_sha256",
                "target_text", "target_text_sha256", "target_char_start",
                "target_char_end",
            ),
            "source evidence entry",
        )
        case_id = _require_nonempty_string(entry["case_id"], "source evidence case_id")
        if case_id in seen_cases:
            raise ValueError("source evidence case IDs must be unique")
        seen_cases.add(case_id)
        if case_id != f"p2ja:case:{index + 1:04d}":
            raise ValueError("source evidence case order is invalid")
        source_id = _require_nonempty_string(
            entry["source_id"], "source evidence source_id",
        )
        transcript = _require_nonempty_string(
            entry["transcript"], "source evidence transcript",
        )
        if entry["transcript_sha256"] != text_sha256(transcript):
            raise ValueError("source evidence transcript hash is invalid")
        start = _require_int(
            entry["target_char_start"], "source evidence target_char_start",
            minimum=0,
        )
        end = _require_int(
            entry["target_char_end"], "source evidence target_char_end",
            minimum=1,
        )
        if not 0 <= start < end <= len(transcript):
            raise ValueError("source evidence target offsets are invalid")
        target_text = _require_nonempty_string(
            entry["target_text"], "source evidence target_text",
        )
        if transcript[start:end] != target_text:
            raise ValueError(
                "source evidence target offsets do not slice the transcript "
                "to the target text",
            )
        if entry["target_text_sha256"] != text_sha256(target_text):
            raise ValueError("source evidence target hash is invalid")


def validate_human_review_packet(
    packet: Mapping[str, Any],
    *,
    require_blank: bool,
) -> None:
    common_keys = (
        "schema_version", "purpose", "release_gate", "blinding",
        "source_evidence", "review_items", "review_fields",
        "semantic_fields", "value_options", "rubric", "content_sha256",
    )
    if require_blank:
        _require_exact_keys(
            packet, common_keys, "phase2j human review packet",
        )
    else:
        _require_exact_keys(
            packet,
            common_keys + ("review_attestation",),
            "phase2j human review packet",
        )
    if packet["schema_version"] != HUMAN_PACKET_SCHEMA_VERSION:
        raise ValueError("human review packet schema version is invalid")
    if tuple(packet["review_fields"]) != REVIEW_FIELDS:
        raise ValueError("human review packet review fields are invalid")
    if tuple(packet["semantic_fields"]) != SEMANTIC_FIELDS:
        raise ValueError("human review packet semantic fields are invalid")
    _validate_recomputed_content_hash(packet, label="human review packet")
    if require_blank:
        if packet["release_gate"] != RELEASE_GATE_AWAITING_REVIEW:
            raise ValueError("blank human review packet release gate is invalid")
    else:
        if packet["release_gate"] != RELEASE_GATE_REVIEWED:
            raise ValueError("finalized human review packet release gate is invalid")
        attestation = packet["review_attestation"]
        if not isinstance(attestation, Mapping):
            raise ValueError("review_attestation must be an object")
        _require_exact_keys(
            attestation,
            (
                "reviewer_kind", "human_review_attested",
                "attestation_statement", "reviewer", "completed_at",
            ),
            "review attestation",
        )
        if attestation["reviewer_kind"] != "human":
            raise ValueError("review attestation reviewer_kind must be human")
        if attestation["human_review_attested"] is not True:
            raise ValueError("review attestation human_review_attested must be true")
        _require_nonempty_string(
            attestation["attestation_statement"], "review attestation statement",
        )
    blinding = packet["blinding"]
    if not isinstance(blinding, Mapping):
        raise ValueError("human review packet blinding must be an object")
    _scan_packet_for_seed_key(packet, path="packet")
    _scan_packet_forbidden_leaks(packet, path="packet")
    _validate_source_evidence(_require_list(
        packet["source_evidence"], "source evidence",
    ))
    items = _require_list(packet["review_items"], "human review items")
    if len(items) != SELECTION_COUNT * len(CONDITION_CODES) * len(SEMANTIC_FIELDS):
        raise ValueError("human review packet item count is invalid")
    seen_ids: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("human review item must be an object")
        _require_exact_keys(
            item,
            (
                "review_item_id", "case_id", "blinded_label", "field",
                "presentation", "output_sha256", "scores", "reviewer",
                "completed_at",
            ),
            "human review item",
        )
        review_item_id = _require_nonempty_string(
            item["review_item_id"], "review item id",
        )
        if review_item_id in seen_ids:
            raise ValueError("human review item IDs must be unique")
        seen_ids.add(review_item_id)
        if item["case_id"] not in {f"p2ja:case:{rank:04d}" for rank in range(1, 11)}:
            raise ValueError("human review item case_id is invalid")
        _require_enum(item["field"], SEMANTIC_FIELDS, "human review item field")
        scores = item["scores"]
        if not isinstance(scores, Mapping) or set(scores) != set(REVIEW_FIELDS):
            raise ValueError("human review item scores are incomplete")
        if require_blank:
            if scores["correctness"] is not None or \
                    scores["unsupported_inference"] is not None or \
                    scores["source_grounding"] is not None:
                raise ValueError("blank human review packet contains filled scores")
        else:
            _require_enum(
                scores["correctness"], CORRECTNESS_VALUES,
                "review correctness",
            )
            _require_enum(
                scores["unsupported_inference"],
                UNSUPPORTED_INFERENCE_VALUES,
                "review unsupported_inference",
            )
            _require_enum(
                scores["source_grounding"],
                SOURCE_GROUNDING_VALUES,
                "review source_grounding",
            )
            _require_list(scores["notes"], "review notes")
        if not isinstance(scores["notes"], list) or any(
            not isinstance(note, str) for note in scores["notes"]
        ):
            raise ValueError("review notes must be a list of strings")
        presentation = item["presentation"]
        if not isinstance(presentation, Mapping):
            raise ValueError("review presentation must be an object")
        _require_exact_keys(
            presentation,
            (
                "schema_version", "case_id", "field", "target_text",
                "target_text_sha256", "source_evidence_index",
                "extraction_items",
            ),
            "review presentation",
        )
        if presentation["schema_version"] != REVIEW_PRESENTATION_SCHEMA_VERSION:
            raise ValueError("review presentation schema version is invalid")
        if presentation["case_id"] != item["case_id"]:
            raise ValueError("review presentation case_id is inconsistent")
        if presentation["field"] != item["field"]:
            raise ValueError("review presentation field is inconsistent")
        if not re.fullmatch(r"[0-9a-f]{64}", presentation["target_text_sha256"]):
            raise ValueError("review presentation target hash is invalid")
        if text_sha256(presentation["target_text"]) != (
            presentation["target_text_sha256"]
        ):
            raise ValueError("review presentation target text hash is invalid")
        index = _require_int(
            presentation["source_evidence_index"],
            "review presentation source_evidence_index",
            minimum=0,
        )
        if index >= len(packet["source_evidence"]):
            raise ValueError("review presentation source evidence index is invalid")
        if packet["source_evidence"][index]["case_id"] != item["case_id"]:
            raise ValueError(
                "review presentation points to the wrong source evidence case",
            )


def validate_human_review_mapping(
    mapping: Mapping[str, Any],
    *,
    packet: Mapping[str, Any],
    outputs_artifact: Mapping[str, Any] | None = None,
    payloads_artifact: Mapping[str, Any] | None = None,
) -> None:
    _require_exact_keys(
        mapping,
        ("schema_version", "purpose", "entries", "content_sha256"),
        "phase2j human review mapping",
    )
    if mapping["schema_version"] != HUMAN_MAPPING_SCHEMA_VERSION:
        raise ValueError("human review mapping schema version is invalid")
    _validate_recomputed_content_hash(mapping, label="human review mapping")
    if mapping["content_sha256"] != packet["blinding"]["mapping_sha256"]:
        raise ValueError("human review mapping is not bound to the packet")
    entries = mapping["entries"]
    if not isinstance(entries, Mapping):
        raise ValueError("human review mapping entries must be an object")
    item_ids = {item["review_item_id"] for item in packet["review_items"]}
    if set(entries) != item_ids:
        raise ValueError("human review mapping does not cover every review item")
    item_by_id = {item["review_item_id"]: item for item in packet["review_items"]}
    per_case_field: dict[tuple[str, str], dict[str, str]] = {}
    for review_item_id, entry in entries.items():
        if not isinstance(entry, Mapping):
            raise ValueError("human review mapping entry must be an object")
        _require_exact_keys(
            entry,
            (
                "case_id", "condition_code", "blinded_label", "field",
                "output_sha256", "presentation_sha256",
                "target_text_sha256",
            ),
            "human review mapping entry",
        )
        _require_enum(entry["condition_code"], CONDITION_CODES, "mapping condition")
        item = item_by_id[review_item_id]
        if entry["case_id"] != item["case_id"]:
            raise ValueError("human review mapping entry case_id is inconsistent")
        if entry["blinded_label"] != item["blinded_label"]:
            raise ValueError("human review mapping entry blinded_label is inconsistent")
        if entry["field"] != item["field"]:
            raise ValueError("human review mapping entry field is inconsistent")
        if entry["output_sha256"] != item["output_sha256"]:
            raise ValueError("human review mapping entry output hash is inconsistent")
        if entry["presentation_sha256"] != canonical_sha256(item["presentation"]):
            raise ValueError(
                "human review mapping entry presentation hash is inconsistent",
            )
        if entry["target_text_sha256"] != item["presentation"]["target_text_sha256"]:
            raise ValueError("human review mapping entry target hash is inconsistent")
        pair_key = (entry["case_id"], entry["field"])
        seen_conditions = per_case_field.setdefault(pair_key, {})
        if entry["condition_code"] in seen_conditions:
            raise ValueError(
                "human review mapping contains duplicate condition per "
                "case/field",
            )
        seen_conditions[entry["condition_code"]] = review_item_id
    for (case_id, field), conditions in sorted(per_case_field.items()):
        if set(conditions) != set(CONDITION_CODES):
            raise ValueError(
                f"human review mapping must contain exactly one A and one B "
                f"for case {case_id} field {field}",
            )
    if outputs_artifact is not None and payloads_artifact is not None:
        outputs_by_case = {
            case["case_id"]: case for case in outputs_artifact["cases"]
        }
        payloads_by_case = {
            case["case_id"]: case for case in payloads_artifact["cases"]
        }
        for review_item_id, entry in entries.items():
            case_id = entry["case_id"]
            condition = entry["condition_code"]
            output_case = outputs_by_case.get(case_id)
            payload_case = payloads_by_case.get(case_id)
            if output_case is None or payload_case is None:
                raise ValueError(
                    f"human review mapping case {case_id} is missing from "
                    "outputs/payloads",
                )
            output = output_case[condition]
            payload = payload_case[condition]
            if entry["output_sha256"] != output["content_sha256"]:
                raise ValueError(
                    "human review mapping output hash does not match the "
                    "condition output",
                )
            source_evidence_index = _source_evidence_index_for_case(
                packet, case_id,
            )
            expected_presentation = _review_presentation(
                case_id=case_id,
                field=entry["field"],
                output=output,
                payload=payload,
                source_evidence_index=source_evidence_index,
            )
            if entry["presentation_sha256"] != canonical_sha256(
                expected_presentation,
            ):
                raise ValueError(
                    "human review mapping presentation does not match the "
                    "canonical condition presentation",
                )
            item = item_by_id[review_item_id]
            if item["presentation"] != expected_presentation:
                raise ValueError(
                    "human review packet presentation does not match the "
                    "canonical condition presentation",
                )


def _unblind_items(
    finalized_packet: Mapping[str, Any],
    mapping: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entries = mapping["entries"]
    unblinded = []
    for item in finalized_packet["review_items"]:
        review_item_id = item["review_item_id"]
        entry = entries.get(review_item_id)
        if not isinstance(entry, Mapping):
            raise ValueError(f"review item {review_item_id!r} is missing from the mapping")
        unblinded.append({
            **dict(item),
            "condition_code": entry["condition_code"],
        })
    return unblinded


# ---------------------------------------------------------------------------
# Completed reviews, finalization, and materiality decision
# ---------------------------------------------------------------------------


def _is_strict_success(scores: Mapping[str, Any]) -> bool:
    return (
        scores["correctness"] in set(STRICT_SUCCESS["correctness"])
        and scores["unsupported_inference"]
        in set(STRICT_SUCCESS["unsupported_inference"])
        and scores["source_grounding"]
        in set(STRICT_SUCCESS["source_grounding"])
    )


def import_completed_reviews(
    packet: Mapping[str, Any],
    completed: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a finalized packet with human scores; refuses incomplete input."""
    validate_human_review_packet(packet, require_blank=True)
    _require_exact_keys(
        completed,
        (
            "schema_version", "reviewer_kind", "human_review_attested",
            "attestation_statement", "reviewer", "completed_at", "reviews",
            "content_sha256",
        ),
        "completed reviews",
    )
    if completed["schema_version"] != COMPLETED_REVIEWS_SCHEMA_VERSION:
        raise ValueError("completed reviews schema version is invalid")
    _validate_recomputed_content_hash(completed, label="completed reviews")
    if completed["reviewer_kind"] != "human":
        raise ValueError("completed reviews reviewer_kind must be exactly human")
    if completed["human_review_attested"] is not True:
        raise ValueError("completed reviews human_review_attested must be exactly true")
    _require_nonempty_string(
        completed["attestation_statement"], "completed reviews attestation statement",
    )
    reviews = completed["reviews"]
    if not isinstance(reviews, Mapping):
        raise ValueError("completed reviews must map item IDs to judgments")
    item_ids = {item["review_item_id"] for item in packet["review_items"]}
    if set(reviews) != item_ids:
        missing = sorted(item_ids - set(reviews))
        extra = sorted(set(reviews) - item_ids)
        raise ValueError(
            "completed reviews must cover every review item; "
            f"missing={missing} extra={extra}",
        )
    reviewer = _require_nonempty_string(completed["reviewer"], "reviewer")
    completed_at = _require_nonempty_string(completed["completed_at"], "completed_at")
    finalized_items = []
    for item in packet["review_items"]:
        review = reviews[item["review_item_id"]]
        if not isinstance(review, Mapping):
            raise ValueError("completed review entry must be an object")
        _require_exact_keys(
            review,
            (
                "correctness", "unsupported_inference",
                "source_grounding", "notes",
            ),
            "completed review entry",
        )
        notes = _require_list(review["notes"], "review notes")
        if any(not isinstance(note, str) for note in notes):
            raise ValueError("review notes must be strings")
        scores = {
            "correctness": _require_enum(
                review["correctness"], CORRECTNESS_VALUES, "review correctness",
            ),
            "unsupported_inference": _require_enum(
                review["unsupported_inference"],
                UNSUPPORTED_INFERENCE_VALUES,
                "review unsupported_inference",
            ),
            "source_grounding": _require_enum(
                review["source_grounding"],
                SOURCE_GROUNDING_VALUES,
                "review source_grounding",
            ),
            "notes": list(notes),
        }
        finalized_items.append({
            **dict(item),
            "scores": scores,
            "reviewer": reviewer,
            "completed_at": completed_at,
        })
    finalized = _envelope({
        "schema_version": packet["schema_version"],
        "purpose": packet["purpose"],
        "release_gate": RELEASE_GATE_REVIEWED,
        "blinding": dict(packet["blinding"]),
        "source_evidence": packet["source_evidence"],
        "review_items": finalized_items,
        "review_fields": list(packet["review_fields"]),
        "semantic_fields": list(packet["semantic_fields"]),
        "value_options": dict(packet["value_options"]),
        "rubric": dict(packet["rubric"]),
        "review_attestation": {
            "reviewer_kind": completed["reviewer_kind"],
            "human_review_attested": completed["human_review_attested"],
            "attestation_statement": completed["attestation_statement"],
            "reviewer": reviewer,
            "completed_at": completed_at,
        },
    })
    finalized["blinding"]["imported_by"] = reviewer
    finalized["blinding"]["imported_at"] = completed_at
    finalized["content_sha256"] = canonical_sha256({
        key: value for key, value in finalized.items() if key != "content_sha256"
    })
    validate_human_review_packet(finalized, require_blank=False)
    return finalized


def compute_materiality(
    finalized_packet: Mapping[str, Any],
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministic preregistered materiality decision over 80 paired fields."""
    validate_human_review_packet(finalized_packet, require_blank=False)
    validate_human_review_mapping(mapping, packet=finalized_packet)
    unblinded = _unblind_items(finalized_packet, mapping)
    per_case: dict[str, dict[str, int]] = {}
    per_field: dict[str, dict[str, int]] = {}
    a_success_total = 0
    b_success_total = 0
    a_major_unsupported = 0
    b_major_unsupported = 0
    for item in unblinded:
        case_id = item["case_id"]
        field = item["field"]
        condition = item["condition_code"]
        scores = item["scores"]
        success = _is_strict_success(scores)
        case_entry = per_case.setdefault(case_id, {"A": 0, "B": 0})
        field_entry = per_field.setdefault(
            field, {"A_success": 0, "B_success": 0},
        )
        case_entry[condition] += 1 if success else 0
        if success:
            field_entry[f"{condition}_success"] += 1
            if condition == "A":
                a_success_total += 1
            else:
                b_success_total += 1
        if scores["unsupported_inference"] == "MAJOR":
            if condition == "A":
                a_major_unsupported += 1
            else:
                b_major_unsupported += 1
    case_wins = {"A": 0, "B": 0, "ties": 0}
    per_case_summary: dict[str, Any] = {}
    for case_id in sorted(per_case):
        a = per_case[case_id]["A"]
        b = per_case[case_id]["B"]
        if b > a:
            winner = "B"
        elif a > b:
            winner = "A"
        else:
            winner = "TIE"
        case_wins[winner if winner != "TIE" else "ties"] += 1
        per_case_summary[case_id] = {
            "A_success": a,
            "B_success": b,
            "winner": winner,
        }
    per_field_summary = {
        field: {
            "A_success": entry["A_success"],
            "B_success": entry["B_success"],
            "delta": entry["B_success"] - entry["A_success"],
        }
        for field, entry in sorted(per_field.items())
    }
    field_delta = b_success_total - a_success_total
    thresholds = MATERIALITY_POLICY["thresholds"]
    passed = (
        field_delta >= thresholds["field_gain_min"]
        and case_wins["B"] >= thresholds["case_wins_min"]
        and case_wins["A"] <= thresholds["a_case_wins_max"]
        and b_major_unsupported <= a_major_unsupported
    )
    return {
        "schema_version": MATERIALITY_POLICY["schema_version"],
        "decision": "MATERIAL" if passed else "NOT_MATERIAL",
        "thresholds": dict(thresholds),
        "strict_success": {
            "A": a_success_total,
            "B": b_success_total,
            "delta": field_delta,
        },
        "major_unsupported_inference": {
            "A": a_major_unsupported,
            "B": b_major_unsupported,
            "delta": b_major_unsupported - a_major_unsupported,
            "rule": "B must have no increase vs A",
        },
        "case_wins": case_wins,
        "per_field": per_field_summary,
        "per_case": per_case_summary,
        "paired_field_judgments": len(unblinded),
    }


def build_materiality_summary(
    *,
    selection: Mapping[str, Any],
    instructions: Mapping[str, Any],
    payloads: Mapping[str, Any],
    outputs: Mapping[str, Any],
    packet: Mapping[str, Any],
    mapping: Mapping[str, Any],
    finalized_packet: Mapping[str, Any],
    completed: Mapping[str, Any],
    materiality: Mapping[str, Any],
    frozen_at: str,
) -> dict[str, Any]:
    return _envelope({
        "schema_version": MATERIALITY_SUMMARY_SCHEMA_VERSION,
        "purpose": (
            "Frozen Sol comparison summary for the Phase 2J context ablation. "
            "DeepSeek B remains gate-locked until this summary is frozen with "
            "decision MATERIAL."
        ),
        "release_gate": RELEASE_GATE_LOCKED,
        "decision": materiality["decision"],
        "frozen_at": frozen_at,
        "preregistered_policy": dict(MATERIALITY_POLICY),
        "input_hashes": {
            "selection": selection["content_sha256"],
            "instructions": instructions["content_sha256"],
            "payloads": payloads["content_sha256"],
            "outputs": outputs["content_sha256"],
            "blank_packet": packet["content_sha256"],
            "mapping": mapping["content_sha256"],
            "finalized_packet": finalized_packet["content_sha256"],
            "completed_reviews": completed["content_sha256"],
        },
        "materiality": materiality,
    })


def validate_materiality_summary(
    summary: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    instructions: Mapping[str, Any],
    payloads: Mapping[str, Any],
    outputs: Mapping[str, Any],
    packet: Mapping[str, Any],
    mapping: Mapping[str, Any],
    finalized_packet: Mapping[str, Any],
    completed: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        summary,
        (
            "schema_version", "purpose", "release_gate", "decision",
            "frozen_at", "preregistered_policy", "input_hashes",
            "materiality", "content_sha256",
        ),
        "phase2j materiality summary",
    )
    if summary["schema_version"] != MATERIALITY_SUMMARY_SCHEMA_VERSION:
        raise ValueError("materiality summary schema version is invalid")
    if summary["release_gate"] != RELEASE_GATE_LOCKED:
        raise ValueError("materiality summary is not frozen/LOCKED")
    _validate_recomputed_content_hash(summary, label="materiality summary")
    _require_enum(summary["decision"], ("MATERIAL", "NOT_MATERIAL"), "materiality decision")
    expected_hashes = {
        "selection": selection["content_sha256"],
        "instructions": instructions["content_sha256"],
        "payloads": payloads["content_sha256"],
        "outputs": outputs["content_sha256"],
        "blank_packet": packet["content_sha256"],
        "mapping": mapping["content_sha256"],
        "finalized_packet": finalized_packet["content_sha256"],
        "completed_reviews": completed["content_sha256"],
    }
    if summary["input_hashes"] != expected_hashes:
        raise ValueError("materiality summary input hashes are invalid")
    recomputed = compute_materiality(finalized_packet, mapping)
    if summary["materiality"] != recomputed:
        raise ValueError("materiality summary does not match recomputed decision")
    if summary["decision"] != recomputed["decision"]:
        raise ValueError("materiality summary decision is inconsistent")


# ---------------------------------------------------------------------------
# DeepSeek B gate
# ---------------------------------------------------------------------------


def require_frozen_material_summary(summary: Mapping[str, Any]) -> str:
    """DeepSeek B stays gate-locked until a frozen MATERIAL Sol summary."""
    _require_exact_keys(
        summary,
        (
            "schema_version", "purpose", "release_gate", "decision",
            "frozen_at", "preregistered_policy", "input_hashes",
            "materiality", "content_sha256",
        ),
        "phase2j materiality summary",
    )
    if summary["schema_version"] != MATERIALITY_SUMMARY_SCHEMA_VERSION:
        raise ValueError("materiality summary schema version is invalid")
    if summary["release_gate"] != RELEASE_GATE_LOCKED:
        raise ValueError(
            "DeepSeek B is gate-locked: materiality summary is not frozen",
        )
    _validate_recomputed_content_hash(summary, label="materiality summary")
    if summary["decision"] != "MATERIAL":
        raise ValueError(
            "DeepSeek B is gate-locked: frozen Sol summary is NOT_MATERIAL",
        )
    return summary["content_sha256"]


def build_deepseek_run_packet(
    *,
    summary: Mapping[str, Any],
    payloads_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    summary_hash = require_frozen_material_summary(summary)
    cases = [
        {
            "case_id": case["case_id"],
            "selection_rank": case["selection_rank"],
            "payload_sha256": case["B"]["content_sha256"],
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
        }
        for case in payloads_artifact["cases"]
    ]
    return _envelope({
        "schema_version": DEEPSEEK_RUN_SCHEMA_VERSION,
        "purpose": (
            "Gate-locked DeepSeek B run packet.  Emitted only after a frozen "
            "MATERIAL Sol comparison summary; consumes the B condition "
            "payload for every frozen case."
        ),
        "release_gate": RELEASE_GATE_LOCKED,
        "materiality_summary_sha256": summary_hash,
        "materiality_decision": "MATERIAL",
        "condition": "B",
        "cases": cases,
    })


def validate_deepseek_run_packet(
    run_packet: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
    payloads_artifact: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        run_packet,
        (
            "schema_version", "purpose", "release_gate",
            "materiality_summary_sha256", "materiality_decision",
            "condition", "cases", "content_sha256",
        ),
        "phase2j deepseek run packet",
    )
    if run_packet["schema_version"] != DEEPSEEK_RUN_SCHEMA_VERSION:
        raise ValueError("deepseek run packet schema version is invalid")
    if run_packet["release_gate"] != RELEASE_GATE_LOCKED:
        raise ValueError("deepseek run packet is not LOCKED")
    _validate_recomputed_content_hash(run_packet, label="deepseek run packet")
    summary_hash = require_frozen_material_summary(summary)
    if run_packet["materiality_summary_sha256"] != summary_hash:
        raise ValueError("deepseek run packet is not bound to the frozen summary")
    if run_packet["materiality_decision"] != "MATERIAL":
        raise ValueError("deepseek run packet decision is invalid")
    if run_packet["condition"] != "B":
        raise ValueError("deepseek run packet condition must be B")
    cases = _require_list(run_packet["cases"], "deepseek run cases")
    if len(cases) != SELECTION_COUNT:
        raise ValueError("deepseek run packet must contain exactly 10 cases")
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError("deepseek run case must be an object")
        _require_exact_keys(
            case,
            (
                "case_id", "selection_rank", "payload_sha256",
                "output_schema_version",
            ),
            "deepseek run case",
        )
        payload_case = payloads_artifact["cases"][index]
        if case["case_id"] != payload_case["case_id"]:
            raise ValueError("deepseek run case order is misaligned")
        if case["selection_rank"] != payload_case["selection_rank"]:
            raise ValueError("deepseek run case rank is misaligned")
        if case["payload_sha256"] != payload_case["B"]["content_sha256"]:
            raise ValueError("deepseek run payload hash is invalid")
        if case["output_schema_version"] != OUTPUT_SCHEMA_VERSION:
            raise ValueError("deepseek run output schema version is invalid")


def import_deepseek_run_outputs(
    *,
    summary: Mapping[str, Any],
    run_packet: Mapping[str, Any],
    outputs_by_case: Mapping[str, Mapping[str, Any]],
    payloads_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Import validated DeepSeek B outputs; gate-locked until MATERIAL."""
    validate_deepseek_run_packet(
        run_packet, summary=summary, payloads_artifact=payloads_artifact,
    )
    run_case_ids = [case["case_id"] for case in run_packet["cases"]]
    missing = sorted(set(run_case_ids) - set(outputs_by_case))
    extra = sorted(set(outputs_by_case) - set(run_case_ids))
    if missing or extra:
        raise ValueError(
            "deepseek outputs must contain exactly the run packet cases; "
            f"missing={missing} extra={extra}",
        )
    cases = []
    for index, run_case in enumerate(run_packet["cases"]):
        case_id = run_case["case_id"]
        output = outputs_by_case.get(case_id)
        if output is None:
            raise ValueError(f"deepseek outputs missing case {case_id}")
        payload_case = payloads_artifact["cases"][index]
        validate_extraction_output(
            output,
            case_id=case_id,
            condition="B",
            payload=payload_case["B"],
        )
        cases.append({"case_id": case_id, "condition": "B", "output": dict(output)})
    import_artifact = _envelope({
        "schema_version": DEEPSEEK_IMPORT_SCHEMA_VERSION,
        "purpose": (
            "Validated imported DeepSeek B outputs, gate-locked behind a "
            "frozen MATERIAL Sol comparison summary."
        ),
        "release_gate": RELEASE_GATE_LOCKED,
        "materiality_summary_sha256": run_packet["materiality_summary_sha256"],
        "run_packet_sha256": run_packet["content_sha256"],
        "cases": cases,
    })
    return import_artifact


def validate_deepseek_import_artifact(
    artifact: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
    run_packet: Mapping[str, Any],
    payloads_artifact: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        artifact,
        (
            "schema_version", "purpose", "release_gate",
            "materiality_summary_sha256", "run_packet_sha256", "cases",
            "content_sha256",
        ),
        "phase2j deepseek import",
    )
    if artifact["schema_version"] != DEEPSEEK_IMPORT_SCHEMA_VERSION:
        raise ValueError("deepseek import schema version is invalid")
    if artifact["release_gate"] != RELEASE_GATE_LOCKED:
        raise ValueError("deepseek import artifact is not LOCKED")
    _validate_recomputed_content_hash(artifact, label="deepseek import artifact")
    require_frozen_material_summary(summary)
    if artifact["materiality_summary_sha256"] != run_packet["materiality_summary_sha256"]:
        raise ValueError("deepseek import summary binding is invalid")
    if artifact["run_packet_sha256"] != run_packet["content_sha256"]:
        raise ValueError("deepseek import run-packet binding is invalid")
    cases = _require_list(artifact["cases"], "deepseek import cases")
    run_case_ids = [case["case_id"] for case in run_packet["cases"]]
    if [case["case_id"] for case in cases] != run_case_ids:
        raise ValueError(
            "deepseek import case identities/order do not match the run packet",
        )
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError("deepseek import case must be an object")
        _require_exact_keys(
            case, ("case_id", "condition", "output"), "deepseek import case",
        )
        if case["case_id"] != run_case_ids[index]:
            raise ValueError("deepseek import case order is misaligned")
        if case["condition"] != "B":
            raise ValueError("deepseek import condition must be B")
        validate_extraction_output(
            case["output"],
            case_id=case["case_id"],
            condition="B",
            payload=payloads_artifact["cases"][index]["B"],
        )


# ---------------------------------------------------------------------------
# Build summary validation
# ---------------------------------------------------------------------------


def validate_build_summary(
    summary: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    manifest_path: Path,
    packet_path: Path,
    manifest: Mapping[str, Any],
    packet: Mapping[str, Any],
    db_path: Path,
    cases: list[Mapping[str, Any]],
) -> None:
    _require_exact_keys(
        summary,
        (
            "schema_version", "purpose", "mode", "input_hashes", "artifacts",
            "selected_case_ids", "selected_window_ids", "content_sha256",
        ),
        "phase2j build summary",
    )
    if summary["schema_version"] != BUILD_SUMMARY_SCHEMA_VERSION:
        raise ValueError("build summary schema version is invalid")
    _validate_recomputed_content_hash(summary, label="build summary")
    _require_enum(
        summary["mode"], (BUILD_MODE_READY_FOR_SOL, BUILD_MODE_REVIEW_PACKET),
        "build summary mode",
    )
    review_artifacts_present = (
        "outputs" in artifacts
        and "human_packet" in artifacts
        and "human_mapping" in artifacts
    )
    if summary["mode"] == BUILD_MODE_READY_FOR_SOL:
        if review_artifacts_present:
            raise ValueError(
                "build summary mode ready_for_sol is inconsistent with "
                "review artifacts present",
            )
    else:
        if not review_artifacts_present:
            raise ValueError(
                "build summary mode review_packet is inconsistent with "
                "missing review artifacts",
            )
    expected_input_hashes = frozen_input_hashes(
        manifest_path, packet_path, manifest, packet, db_path=db_path,
    )
    if summary["input_hashes"] != expected_input_hashes:
        raise ValueError("build summary input hashes are invalid")
    expected_artifact_keys = set(artifacts) - {"build_summary"}
    summary_artifact_keys = set(summary["artifacts"])
    if not summary_artifact_keys <= expected_artifact_keys:
        raise ValueError(
            "build summary artifact coverage is invalid; "
            f"extra={sorted(summary_artifact_keys - expected_artifact_keys)}",
        )
    for key, entry in summary["artifacts"].items():
        if not isinstance(entry, Mapping):
            raise ValueError("build summary artifact entry must be an object")
        _require_exact_keys(
            entry,
            ("path", "file_sha256", "content_sha256", "schema_version"),
            "build summary artifact entry",
        )
        artifact = artifacts[key]
        path = ROOT / entry["path"]
        if entry["content_sha256"] != artifact["content_sha256"]:
            raise ValueError(f"build summary artifact {key} content hash is invalid")
        if entry["schema_version"] != artifact["schema_version"]:
            raise ValueError(f"build summary artifact {key} schema version is invalid")
        if entry["file_sha256"] != file_sha256(path):
            raise ValueError(f"build summary artifact {key} file hash is invalid")
    expected_case_ids = [case["case_id"] for case in cases]
    expected_window_ids = [case["window_id"] for case in cases]
    if summary["selected_case_ids"] != expected_case_ids:
        raise ValueError("build summary selected case IDs are invalid")
    if summary["selected_window_ids"] != expected_window_ids:
        raise ValueError("build summary selected window IDs are invalid")


# ---------------------------------------------------------------------------
# Build orchestration and output-directory validation
# ---------------------------------------------------------------------------


def _load_all_artifacts(output_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, filename in OUTPUT_FILENAMES.items():
        path = output_dir / filename
        if path.is_file():
            result[key] = load_json_strict(path, label=f"phase2j {key} artifact")
    return result


def build_phase2j_context_ablation_outputs(
    *,
    manifest_path: Path,
    packet_path: Path,
    db_path: Path,
    output_dir: Path,
    outputs_path: Path | None = None,
    vocabulary_path: Path | None = None,
) -> dict[str, Any]:
    """Deterministic no-model build of the Phase 2J context-ablation outputs.

    Selection, extraction instructions, and the A/B condition payloads are
    always frozen from the frozen manifest, the read-only DB transcript, and
    the lexical vocabulary.  When validated extraction outputs are supplied,
    the blinded human review packet and sealed mapping are additionally
    generated.
    """
    manifest, packet = validate_phase2j_frozen_inputs(manifest_path, packet_path)
    connection = open_transcript_db(db_path)
    try:
        rows = fetch_transcript_rows(connection, manifest["selected"])
    finally:
        connection.close()
    cases = select_cases(manifest, transcript_rows=rows)
    for case in cases:
        validate_manifest_db_alignment(case, rows[case["upstream_source_id"]])
    selection = build_selection_artifact(
        manifest_path=manifest_path,
        packet_path=packet_path,
        manifest=manifest,
        packet=packet,
        cases=cases,
        db_path=db_path,
    )
    vocabulary_path = vocabulary_path or DEFAULT_VOCABULARY_PATH
    lexical_vocabulary = load_lexical_vocabulary(vocabulary_path)
    instructions = build_extraction_instructions()
    instructions_artifact = build_instructions_artifact()
    connection = open_transcript_db(db_path)
    try:
        vocabulary_by_case: dict[str, dict[str, Any]] = {}
        for case in cases:
            source_id = case["upstream_source_id"]
            champion_data = champion_abilities_for_transcript(
                connection,
                metadata_champion=case["metadata"]["champion"],
                transcript=rows[source_id]["transcript"],
                video_id=source_id,
            )
            vocabulary_by_case[case["case_id"]] = build_case_vocabulary(
                case_id=case["case_id"],
                lexical_vocabulary=lexical_vocabulary,
                champion_data=champion_data,
            )
    finally:
        connection.close()
    payload_cases, provenance_by_case = build_condition_payloads(
        cases=cases,
        transcript_rows=rows,
        vocabulary_by_case=vocabulary_by_case,
        instructions=instructions,
    )
    payloads = build_payloads_artifact(
        selection=selection,
        instructions=instructions,
        payload_cases=payload_cases,
        provenance_by_case=provenance_by_case,
    )
    validate_payloads_artifact(
        payloads,
        selection=selection,
        instructions=instructions,
        lexical_vocabulary=lexical_vocabulary,
        manifest_path=manifest_path,
        packet_path=packet_path,
        manifest=manifest,
        packet=packet,
        db_path=db_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / OUTPUT_FILENAMES["selection"]
    instructions_path = output_dir / OUTPUT_FILENAMES["instructions"]
    payloads_path = output_dir / OUTPUT_FILENAMES["payloads"]
    _write_json_atomic(selection_path, selection)
    _write_json_atomic(instructions_path, instructions_artifact)
    _write_json_atomic(payloads_path, payloads)
    artifacts: dict[str, dict[str, Any]] = {
        "selection": selection,
        "instructions": instructions_artifact,
        "payloads": payloads,
    }
    mode = BUILD_MODE_READY_FOR_SOL

    if outputs_path is not None:
        outputs_bundle = load_json_strict(
            outputs_path, label="phase2j extraction outputs",
        )
        validate_extraction_outputs_bundle(
            outputs_bundle,
            payloads_artifact=payloads,
            instructions=instructions,
        )
        outputs_path_out = output_dir / OUTPUT_FILENAMES["outputs"]
        _write_json_atomic(outputs_path_out, outputs_bundle)
        human_packet, mapping = build_human_review_packet(
            outputs_artifact=outputs_bundle,
            payloads_artifact=payloads,
            transcript_rows=rows,
        )
        validate_human_review_packet(human_packet, require_blank=True)
        validate_human_review_mapping(
            mapping,
            packet=human_packet,
            outputs_artifact=outputs_bundle,
            payloads_artifact=payloads,
        )
        packet_path_out = output_dir / OUTPUT_FILENAMES["human_packet"]
        mapping_path_out = output_dir / OUTPUT_FILENAMES["human_mapping"]
        _write_json_atomic(packet_path_out, human_packet)
        _write_json_atomic(mapping_path_out, mapping)
        artifacts.update({
            "outputs": outputs_bundle,
            "human_packet": human_packet,
            "human_mapping": mapping,
        })
        mode = BUILD_MODE_REVIEW_PACKET

    artifact_entries = {}
    for key, artifact in artifacts.items():
        artifact_entries[key] = {
            "path": normalize_path_locator(output_dir / OUTPUT_FILENAMES[key]),
            "file_sha256": file_sha256(output_dir / OUTPUT_FILENAMES[key]),
            "content_sha256": artifact["content_sha256"],
            "schema_version": artifact["schema_version"],
        }
    build_summary = _envelope({
        "schema_version": BUILD_SUMMARY_SCHEMA_VERSION,
        "purpose": "Phase 2J context-ablation build summary (no model calls).",
        "mode": mode,
        "input_hashes": frozen_input_hashes(
            manifest_path, packet_path, manifest, packet, db_path=db_path,
        ),
        "artifacts": artifact_entries,
        "selected_case_ids": [case["case_id"] for case in cases],
        "selected_window_ids": [case["window_id"] for case in cases],
    })
    build_summary_path = output_dir / OUTPUT_FILENAMES["build_summary"]
    _write_json_atomic(build_summary_path, build_summary)
    artifacts["build_summary"] = build_summary
    return {
        "mode": mode,
        "output_dir": output_dir,
        "selection_sha256": selection["content_sha256"],
        "instructions_sha256": instructions_artifact["content_sha256"],
        "payloads_sha256": payloads["content_sha256"],
        "outputs_sha256": (
            artifacts["outputs"]["content_sha256"]
            if "outputs" in artifacts else None
        ),
        "human_packet_sha256": (
            artifacts["human_packet"]["content_sha256"]
            if "human_packet" in artifacts else None
        ),
        "human_mapping_sha256": (
            artifacts["human_mapping"]["content_sha256"]
            if "human_mapping" in artifacts else None
        ),
        "build_summary_sha256": build_summary["content_sha256"],
        "selected_case_ids": [case["case_id"] for case in cases],
        "selected_window_ids": [case["window_id"] for case in cases],
    }


def finalize_materiality_outputs(
    *,
    output_dir: Path,
    reviews_path: Path,
    frozen_at: str,
) -> dict[str, Any]:
    """Import completed reviews and freeze the Sol comparison summary."""
    required = (
        "selection", "instructions", "payloads", "outputs",
        "human_packet", "human_mapping",
    )
    artifacts = _load_all_artifacts(output_dir)
    for key in required:
        if key not in artifacts:
            raise ValueError(f"cannot finalize without artifact {key!r}")
    packet = artifacts["human_packet"]
    mapping = artifacts["human_mapping"]
    validate_human_review_packet(packet, require_blank=True)
    validate_human_review_mapping(
        mapping,
        packet=packet,
        outputs_artifact=artifacts["outputs"],
        payloads_artifact=artifacts["payloads"],
    )
    completed = load_json_strict(reviews_path, label="completed reviews")
    finalized_packet = import_completed_reviews(packet, completed)
    materiality = compute_materiality(finalized_packet, mapping)
    summary = build_materiality_summary(
        selection=artifacts["selection"],
        instructions=artifacts["instructions"],
        payloads=artifacts["payloads"],
        outputs=artifacts["outputs"],
        packet=packet,
        mapping=mapping,
        finalized_packet=finalized_packet,
        completed=completed,
        materiality=materiality,
        frozen_at=frozen_at,
    )
    validate_materiality_summary(
        summary,
        selection=artifacts["selection"],
        instructions=artifacts["instructions"],
        payloads=artifacts["payloads"],
        outputs=artifacts["outputs"],
        packet=packet,
        mapping=mapping,
        finalized_packet=finalized_packet,
        completed=completed,
    )
    finalized_path = output_dir / OUTPUT_FILENAMES["finalized_packet"]
    summary_path = output_dir / OUTPUT_FILENAMES["materiality_summary"]
    completed_path = output_dir / OUTPUT_FILENAMES["completed_reviews"]
    _write_json_atomic(completed_path, completed)
    _write_json_atomic(finalized_path, finalized_packet)
    _write_json_atomic(summary_path, summary)
    return {
        "decision": summary["decision"],
        "materiality": summary["materiality"],
        "finalized_packet_sha256": finalized_packet["content_sha256"],
        "materiality_summary_sha256": summary["content_sha256"],
        "output_dir": output_dir,
    }


def validate_output_directory(
    *,
    output_dir: Path,
    manifest_path: Path,
    packet_path: Path,
    db_path: Path,
    vocabulary_path: Path | None = None,
) -> dict[str, Any]:
    """Deterministically revalidate an existing Phase 2J output directory."""
    manifest, packet = validate_phase2j_frozen_inputs(manifest_path, packet_path)
    connection = open_transcript_db(db_path)
    try:
        rows = fetch_transcript_rows(connection, manifest["selected"])
    finally:
        connection.close()
    artifacts = _load_all_artifacts(output_dir)
    selection = artifacts.get("selection")
    if selection is None:
        raise ValueError("output directory is missing the selection artifact")
    if "instructions" not in artifacts or "payloads" not in artifacts:
        raise ValueError(
            "output directory is missing the instructions/payloads artifacts",
        )
    if "build_summary" not in artifacts:
        raise ValueError("output directory is missing the build summary artifact")
    cases = select_cases(manifest, transcript_rows=rows)
    validate_selection_artifact(
        selection,
        manifest_path=manifest_path,
        packet_path=packet_path,
        manifest=manifest,
        packet=packet,
        db_path=db_path,
    )
    # Fail closed on partial artifact combinations.
    if "outputs" in artifacts and "payloads" not in artifacts:
        raise ValueError("outputs present without payloads/instructions")
    if ("human_packet" in artifacts) != ("human_mapping" in artifacts):
        raise ValueError(
            "output directory must contain both the review packet and mapping",
        )
    if "human_packet" in artifacts and "outputs" not in artifacts:
        raise ValueError("review packet present without validated outputs")
    finalized_required = {
        "outputs", "human_packet", "human_mapping", "completed_reviews",
        "materiality_summary",
    }
    if "finalized_packet" in artifacts:
        missing = sorted(finalized_required - set(artifacts))
        if missing:
            raise ValueError(
                "finalized packet present but required artifacts missing: "
                f"{missing}",
            )
    if "completed_reviews" in artifacts:
        missing = sorted(
            {"finalized_packet", "materiality_summary"} - set(artifacts),
        )
        if missing:
            raise ValueError(
                "completed reviews present but required artifacts missing: "
                f"{missing}",
            )
    if "materiality_summary" in artifacts:
        missing = sorted(
            {
                "outputs", "human_packet", "human_mapping",
                "finalized_packet", "completed_reviews",
            } - set(artifacts),
        )
        if missing:
            raise ValueError(
                "materiality summary present but required artifacts missing: "
                f"{missing}",
            )
    if "deepseek_run" in artifacts:
        if "materiality_summary" not in artifacts or "payloads" not in artifacts:
            raise ValueError(
                "deepseek run packet present without a frozen materiality "
                "summary and payloads",
            )
    if "deepseek_import" in artifacts:
        if "deepseek_run" not in artifacts:
            raise ValueError(
                "deepseek import present without the deepseek run packet",
            )

    summary_result: dict[str, Any] = {
        "selection_sha256": selection["content_sha256"],
    }
    if "instructions" in artifacts and "payloads" in artifacts:
        validate_instructions_artifact(artifacts["instructions"])
        instructions = {
            key: value for key, value in artifacts["instructions"].items()
            if key != "content_sha256"
        }
        lexical_vocabulary = load_lexical_vocabulary(
            vocabulary_path or DEFAULT_VOCABULARY_PATH,
        )
        validate_payloads_artifact(
            artifacts["payloads"],
            selection=selection,
            instructions=instructions,
            lexical_vocabulary=lexical_vocabulary,
            manifest_path=manifest_path,
            packet_path=packet_path,
            manifest=manifest,
            packet=packet,
            db_path=db_path,
        )
        summary_result["instructions_sha256"] = (
            artifacts["instructions"]["content_sha256"]
        )
        summary_result["payloads_sha256"] = artifacts["payloads"]["content_sha256"]
        if "outputs" in artifacts:
            validate_extraction_outputs_bundle(
                artifacts["outputs"],
                payloads_artifact=artifacts["payloads"],
                instructions=instructions,
            )
            summary_result["outputs_sha256"] = artifacts["outputs"]["content_sha256"]
        if "human_packet" in artifacts and "human_mapping" in artifacts:
            validate_human_review_packet(artifacts["human_packet"], require_blank=True)
            validate_human_review_mapping(
                artifacts["human_mapping"],
                packet=artifacts["human_packet"],
                outputs_artifact=artifacts["outputs"],
                payloads_artifact=artifacts["payloads"],
            )
            summary_result["human_packet_sha256"] = (
                artifacts["human_packet"]["content_sha256"]
            )
            summary_result["human_mapping_sha256"] = (
                artifacts["human_mapping"]["content_sha256"]
            )
        if "finalized_packet" in artifacts:
            validate_human_review_packet(
                artifacts["finalized_packet"], require_blank=False,
            )
            summary_result["finalized_packet_sha256"] = (
                artifacts["finalized_packet"]["content_sha256"]
            )
        if "completed_reviews" in artifacts:
            summary_result["completed_reviews_sha256"] = (
                artifacts["completed_reviews"]["content_sha256"]
            )
        if "materiality_summary" in artifacts:
            validate_materiality_summary(
                artifacts["materiality_summary"],
                selection=selection,
                instructions=artifacts["instructions"],
                payloads=artifacts["payloads"],
                outputs=artifacts["outputs"],
                packet=artifacts["human_packet"],
                mapping=artifacts["human_mapping"],
                finalized_packet=artifacts["finalized_packet"],
                completed=artifacts["completed_reviews"],
            )
            summary_result["materiality_summary_sha256"] = (
                artifacts["materiality_summary"]["content_sha256"]
            )
        if "deepseek_run" in artifacts:
            validate_deepseek_run_packet(
                artifacts["deepseek_run"],
                summary=artifacts["materiality_summary"],
                payloads_artifact=artifacts["payloads"],
            )
            summary_result["deepseek_run_sha256"] = (
                artifacts["deepseek_run"]["content_sha256"]
            )
        if "deepseek_import" in artifacts:
            validate_deepseek_import_artifact(
                artifacts["deepseek_import"],
                summary=artifacts["materiality_summary"],
                run_packet=artifacts["deepseek_run"],
                payloads_artifact=artifacts["payloads"],
            )
            summary_result["deepseek_import_sha256"] = (
                artifacts["deepseek_import"]["content_sha256"]
            )
    if "build_summary" in artifacts:
        validate_build_summary(
            artifacts["build_summary"],
            artifacts=artifacts,
            manifest_path=manifest_path,
            packet_path=packet_path,
            manifest=manifest,
            packet=packet,
            db_path=db_path,
            cases=cases,
        )
        summary_result["build_summary_sha256"] = (
            artifacts["build_summary"]["content_sha256"]
        )
    return {
        "output_dir": output_dir,
        "valid": True,
        **summary_result,
    }


def _build_summary_json(result: dict[str, Any]) -> str:
    return json.dumps({
        "pipeline_version": PIPELINE_VERSION,
        "mode": result["mode"],
        "output_dir": str(result["output_dir"]),
        "selection_sha256": result["selection_sha256"],
        "instructions_sha256": result["instructions_sha256"],
        "payloads_sha256": result["payloads_sha256"],
        "outputs_sha256": result.get("outputs_sha256"),
        "human_packet_sha256": result.get("human_packet_sha256"),
        "human_mapping_sha256": result.get("human_mapping_sha256"),
        "build_summary_sha256": result["build_summary_sha256"],
        "selected_case_ids": result["selected_case_ids"],
        "selected_window_ids": result["selected_window_ids"],
    }, sort_keys=True, indent=2)
