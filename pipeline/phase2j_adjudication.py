"""Phase 2J post-Pass-A human-vs-Sol adjudication packet builder.

After the human completes Pass A on the locked Phase 2J annotation packet, the
sealed independent Sol navigation/audit review may be revealed for explicit
human adjudication.  This module builds a deterministic, sanitized
adjudication packet that contains:

* the exact locked Bronze windows (identity/order/text/tokens);
* the sanitized human Pass A endpoints and Sol proposed endpoints;
* connected components of inclusive overlapping token intervals across the two
  sides, classified as EXACT_AGREEMENT, TYPE_DISAGREEMENT,
  BOUNDARY_DISAGREEMENT, SOL_ONLY, or HUMAN_ONLY;
* input hashes, schema versions, and canonical content hashing;

and nothing else: no model scores, predictions, ranks, candidate data, reviewer
identity, or packet-internal fields (partition, upstream coordinates, pass
records, rules).  Sol proposals are a second opinion and are never auto-promoted
to gold; the adjudication result remains REVIEW MATERIAL until a separately
validated canonical import/finalizer runs.

All token intervals use inclusive ``token_start``/``token_end`` bounds, so two
intervals overlap when ``left.token_start <= right.token_end`` and
``right.token_start <= left.token_end``.  Adjacent whole-token spans do not
overlap.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from pipeline.phase2j_annotation_packet import (
    ANNOTATION_VERSION,
    PACKET_SCHEMA_VERSION,
    load_annotation_packet,
)
from pipeline.phase2j_source_selection import canonical_sha256, file_sha256
from pipeline.semantic_ir import NodeType


ADJUDICATION_PACKET_SCHEMA_VERSION = "phase2j-adjudication-packet-v1"
ADJUDICATION_VERSION = "phase2j-adjudication-v1"
HUMAN_SESSION_SCHEMA_VERSION = "phase2j-review-session-v1"
SOL_REVIEW_SCHEMA_VERSION = "phase2j-sol-parallel-review-v1"
VISIBILITY_GATE = "SOL_VISIBLE_FOR_ADJUDICATION"
SOL_NON_GOLD_PROVENANCE = "SOL_PARALLEL_NON_GOLD"

COMPONENT_CLASSES = (
    "EXACT_AGREEMENT",
    "TYPE_DISAGREEMENT",
    "BOUNDARY_DISAGREEMENT",
    "SOL_ONLY",
    "HUMAN_ONLY",
)

ENDPOINT_TYPES = frozenset(item.value for item in NodeType) | {"UNDETERMINED"}
SOL_AMBIGUITY_STATES = frozenset({
    "NONE", "UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE", "MULTIPLE_CANDIDATES",
})
WINDOW_OUTCOMES = ("CLEAN", "AMBIGUOUS", "EXCLUDED")
WINDOW_STATUSES = ("UNREVIEWED", "IN_REVIEW", "AMBIGUOUS", "EXCLUDED")
SOL_VISIBILITY_GATES = frozenset({
    "SEALED_UNTIL_HUMAN_PASS_A_COMPLETE",
    "SOL_VISIBLE_FOR_ADJUDICATION",
})

_SHA256 = re.compile(r"[0-9a-f]{64}")

# Keys that must never appear anywhere in the generated adjudication packet:
# scorer/model material, candidate material, packet-internal machine fields,
# and reviewer identity/PII.  Exact case-insensitive key matches and floating
# point values are rejected recursively.
OUTPUT_FORBIDDEN_KEYS = frozenset({
    "score", "scores", "probability", "probabilities", "confidence",
    "rank", "ranks", "ranked", "ranking", "rankings",
    "prediction", "predictions", "predicted", "predicted_label",
    "predicted_labels", "label", "labels", "gold_label", "gold_labels",
    "syntax_importance", "syntax_importances", "feature_importance",
    "feature_importances", "importance", "importances", "error_taxonomy",
    "model_suggestion", "model_suggestions", "suggestion", "suggestions",
    "model", "model_id", "model_name", "model_score", "model_data",
    "reviewer_model", "reasoning_effort", "logits", "proba",
    "candidate", "candidates", "candidate_catalog", "candidate_generator_version",
    "catalog", "catalog_sha256", "partition", "champion", "role",
    "video_title", "video_title_url", "annotation_id", "upstream_source_id",
    "upstream_start", "upstream_end", "pass_a", "pass_b", "release_gate",
    "rules", "selection_manifest_sha256",
    "selection_manifest_schema_version", "reviewer_name", "reviewed_at",
    "completed_at", "exported_at",
    "ambiguity_controls", "exclusion_controls", "reviewer_notes",
    "proposal", "proposals", "sol_notes",
})

# Keys that must never appear inside the human review session input.  The
# session legitimately carries reviewer_name and timestamps, so those are not
# banned here; the builder strips them from the output instead.
HUMAN_SESSION_FORBIDDEN_KEYS = frozenset(
    set(OUTPUT_FORBIDDEN_KEYS) | {
        "model", "model_data", "partition", "candidate", "candidates",
        "candidate_catalog", "candidate_generator_version", "catalog",
        "catalog_sha256", "champion", "role", "video_title", "video_title_url",
        "annotation_id", "upstream_source_id", "upstream_start", "upstream_end",
        "ambiguity_controls", "exclusion_controls", "pass_a", "pass_b",
        "release_gate", "purpose", "rules", "content_sha256",
        "selection_manifest_sha256", "selection_manifest_schema_version",
        "proposal", "proposals", "sol",
    }
) - {
    "reviewer_name", "completed_at", "exported_at",
}

_EXPECTED_PACKET_KEYS = frozenset({
    "content_sha256", "schema_version", "adjudication_version",
    "annotation_version", "packet_schema_version", "packet_sha256",
    "human_session_schema_version", "human_session_sha256",
    "sol_review_schema_version", "sol_review_sha256", "visibility_gate",
    "purpose", "totals", "records",
})
_EXPECTED_TOTALS_KEYS = frozenset({
    "windows", "components", "exact_agreements", "type_disagreements",
    "boundary_disagreements", "sol_only", "human_only",
    "human_endpoints", "sol_endpoints",
})
_EXPECTED_RECORD_KEYS = frozenset({
    "record_index", "window_id", "source_group_id", "bronze_text",
    "bronze_text_sha256", "bronze_char_length", "tokens", "human_outcome",
    "human_endpoints", "sol_endpoints", "components",
})
_EXPECTED_HUMAN_ENDPOINT_KEYS = frozenset({
    "endpoint_id", "exact_bronze_text", "char_start", "char_end",
    "token_start", "token_end", "node_type",
})
_EXPECTED_SOL_ENDPOINT_KEYS = frozenset({
    "endpoint_id", "exact_bronze_text", "char_start", "char_end",
    "token_start", "token_end", "node_type", "sol_ambiguity_state",
    "sol_rationale",
})
_EXPECTED_COMPONENT_KEYS = frozenset({
    "component_id", "classification", "human_endpoint_ids", "sol_endpoint_ids",
})
_EXPECTED_HUMAN_SESSION_KEYS = frozenset({
    "schema_version", "annotation_version", "packet_schema_version",
    "packet_sha256", "exported_at", "records",
})
_EXPECTED_HUMAN_RECORD_KEYS = frozenset({
    "record_index", "window_id", "source_group_id", "bronze_text",
    "bronze_text_sha256", "bronze_char_length", "tokens", "endpoints",
    "window_status", "outcome", "note", "reviewer_name", "completed_at",
    "pass_a_complete",
})
_EXPECTED_HUMAN_ENDPOINT_INPUT_KEYS = frozenset({
    "endpoint_id", "exact_bronze_text", "char_start", "char_end",
    "token_start", "token_end", "node_type", "ambiguity_state",
    "disposition", "pass_provenance", "human_accepted", "created_sequence",
})
_EXPECTED_SOL_REVIEW_KEYS = frozenset({
    "schema_version", "content_sha256", "blank_packet_sha256",
    "selection_manifest_sha256", "purpose", "reasoning_effort",
    "reviewer_model", "visibility_gate", "records",
})
_EXPECTED_SOL_RECORD_KEYS = frozenset({
    "record_index", "window_id", "bronze_text_sha256",
    "window_ambiguity_notes", "omission_audit_notes", "proposed_endpoints",
})
_EXPECTED_SOL_ENDPOINT_INPUT_KEYS = frozenset({
    "ambiguity_state", "concise_rationale", "exact_bronze_text", "node_type",
    "pass_provenance", "token_end", "token_start",
})


def _load_json_strict(path: Path, *, label: str) -> dict[str, Any]:
    """Strict JSON object load with duplicate-key rejection."""
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} JSON contains duplicate keys")
            value[key] = item
        return value

    try:
        body = json.loads(
            Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
        )
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} JSON is unavailable or malformed") from exc
    if not isinstance(body, dict):
        raise ValueError(f"{label} must be a JSON object")
    return body


def _validate_forbidden_content(
    value: object,
    forbidden_keys: frozenset[str],
    *,
    path: tuple[str, ...] = (),
) -> None:
    """Recursively reject forbidden scorer/model/candidate/PII keys and floats."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("phase2j adjudication keys must be strings")
            if key.casefold() in forbidden_keys:
                raise ValueError(
                    "phase2j adjudication content contains forbidden key "
                    + repr(key) + " at " + ".".join(path + (key,)),
                )
            _validate_forbidden_content(item, forbidden_keys, path=path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_forbidden_content(
                item, forbidden_keys, path=path + (f"[{index}]",),
            )
    elif isinstance(value, float):
        raise ValueError(
            "phase2j adjudication content contains a floating-point value",
        )


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"phase2j {label} must be a string")
    return value


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"phase2j {label} must be an integer")
    return value


def _token_interval_overlaps(
    left: tuple[int, int], right: tuple[int, int],
) -> bool:
    """Inclusive token-interval overlap used for cross-side components."""
    return left[0] <= right[1] and right[0] <= left[1]


def _char_interval_overlaps(
    left: tuple[int, int], right: tuple[int, int],
) -> bool:
    """Half-open character-interval overlap used for internal overlap checks."""
    return left[0] < right[1] and right[0] < left[1]


def _validate_tokens_match(
    bronze_text: str,
    tokens: object,
    *,
    window_id: str,
) -> list[dict[str, Any]]:
    if not isinstance(tokens, list):
        raise ValueError(f"phase2j token table for window {window_id} must be a list")
    previous_end = 0
    validated: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        if not isinstance(token, Mapping) or set(token) != {
            "token_index", "start", "end", "text",
        }:
            raise ValueError(f"phase2j token record {index} in {window_id} is invalid")
        if token["token_index"] != index:
            raise ValueError(f"phase2j token indices must be sequential in {window_id}")
        start, end = token["start"], token["end"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)) \
                or not 0 <= start < end <= len(bronze_text):
            raise ValueError(f"phase2j token offsets are invalid in {window_id}")
        if not isinstance(token["text"], str) \
                or bronze_text[start:end] != token["text"]:
            raise ValueError(
                f"phase2j token text is not an exact source slice in {window_id}",
            )
        if index > 0 and start <= previous_end:
            raise ValueError(
                f"phase2j tokens must be ordered and non-overlapping in {window_id}",
            )
        if bronze_text[previous_end:start].strip():
            raise ValueError(
                f"phase2j token table discarded source text in {window_id}",
            )
        previous_end = end
        validated.append(dict(token))
    if validated and bronze_text[previous_end:].strip():
        raise ValueError(
            f"phase2j token table discarded trailing source text in {window_id}",
        )
    return validated


def _validate_span_endpoint(
    endpoint: Mapping[str, Any],
    *,
    bronze_text: str,
    tokens: list[dict[str, Any]],
    window_id: str,
    label: str,
) -> tuple[int, int]:
    """Validate token/char bounds and exact Bronze slices; return (cs, ce)."""
    char_start = _require_int(endpoint["char_start"], f"{label} char_start")
    char_end = _require_int(endpoint["char_end"], f"{label} char_end")
    token_start = _require_int(endpoint["token_start"], f"{label} token_start")
    token_end = _require_int(endpoint["token_end"], f"{label} token_end")
    if not 0 <= char_start < char_end <= len(bronze_text):
        raise ValueError(f"phase2j {label} character offsets are invalid in {window_id}")
    if not 0 <= token_start <= token_end < len(tokens):
        raise ValueError(f"phase2j {label} token offsets are invalid in {window_id}")
    if tokens[token_start]["start"] != char_start \
            or tokens[token_end]["end"] != char_end:
        raise ValueError(
            f"phase2j {label} token boundaries must match character spans in {window_id}",
        )
    exact = _require_string(endpoint["exact_bronze_text"], f"{label} exact text")
    if bronze_text[char_start:char_end] != exact:
        raise ValueError(
            f"phase2j {label} text is not an exact Bronze slice in {window_id}",
        )
    return char_start, char_end


def _validate_no_internal_overlap(
    endpoints: list[Mapping[str, Any]],
    *,
    window_id: str,
    label: str,
) -> None:
    seen_ids: set[str] = set()
    spans: list[tuple[int, int, str]] = []
    for endpoint in endpoints:
        endpoint_id = _require_string(endpoint["endpoint_id"], f"{label} endpoint id")
        if endpoint_id in seen_ids:
            raise ValueError(
                f"phase2j duplicate {label} endpoint id {endpoint_id} in {window_id}",
            )
        seen_ids.add(endpoint_id)
        spans.append((
            _require_int(endpoint["char_start"], f"{label} char_start"),
            _require_int(endpoint["char_end"], f"{label} char_end"),
            endpoint_id,
        ))
    for left_index in range(len(spans)):
        for right_index in range(left_index + 1, len(spans)):
            left = spans[left_index]
            right = spans[right_index]
            if _char_interval_overlaps((left[0], left[1]), (right[0], right[1])):
                raise ValueError(
                    f"phase2j duplicate/overlapping {label} spans are rejected in "
                    f"{window_id}: {left[2]} vs {right[2]}",
                )


def validate_human_session(
    session: Mapping[str, Any],
    packet: Mapping[str, Any],
) -> None:
    """Strict validation of the human Pass A review session against the packet."""
    if set(session) != _EXPECTED_HUMAN_SESSION_KEYS:
        raise ValueError("phase2j human review session envelope is invalid")
    if session["schema_version"] != HUMAN_SESSION_SCHEMA_VERSION:
        raise ValueError("phase2j human review session version is unsupported")
    if session["annotation_version"] != ANNOTATION_VERSION:
        raise ValueError("phase2j human annotation version is unsupported")
    if session["packet_schema_version"] != PACKET_SCHEMA_VERSION:
        raise ValueError("phase2j human packet schema version is unsupported")
    if session["packet_sha256"] != packet["content_sha256"]:
        raise ValueError("phase2j human session is not bound to the locked packet")
    exported_at = session["exported_at"]
    if exported_at is not None and not isinstance(exported_at, str):
        raise ValueError("phase2j human session exported_at must be null or a string")
    _validate_forbidden_content(session, HUMAN_SESSION_FORBIDDEN_KEYS)
    records = session["records"]
    if not isinstance(records, list) or len(records) != len(packet["records"]):
        raise ValueError(
            "phase2j human session must contain exactly "
            f"{len(packet['records'])} windows",
        )
    packet_by_window = {record["window_id"]: record for record in packet["records"]}
    for index, (human, locked) in enumerate(zip(records, packet["records"]), 1):
        if not isinstance(human, Mapping) or set(human) != _EXPECTED_HUMAN_RECORD_KEYS:
            raise ValueError(f"phase2j human record {index} is invalid")
        if human["record_index"] != index:
            raise ValueError(f"phase2j human record_index must be {index}")
        for field in (
            "window_id", "source_group_id", "bronze_text", "bronze_text_sha256",
            "bronze_char_length",
        ):
            if human[field] != locked[field]:
                raise ValueError(
                    f"phase2j human record {index} {field} contradicts the locked packet",
                )
        tokens = _validate_tokens_match(
            human["bronze_text"], human["tokens"], window_id=human["window_id"],
        )
        if tokens != locked["tokens"]:
            raise ValueError(
                f"phase2j human record {index} token table contradicts the locked packet",
            )
        outcome = human["outcome"]
        window_status = human["window_status"]
        if outcome not in WINDOW_OUTCOMES:
            raise ValueError(f"phase2j human record {index} outcome is invalid")
        if window_status not in WINDOW_STATUSES:
            raise ValueError(f"phase2j human record {index} window_status is invalid")
        note = _require_string(human["note"], f"human record {index} note")
        reviewer_name = _require_string(
            human["reviewer_name"], f"human record {index} reviewer_name",
        )
        completed_at = human["completed_at"]
        if completed_at is not None and not isinstance(completed_at, str):
            raise ValueError(f"phase2j human record {index} completed_at is invalid")
        pass_a_complete = human["pass_a_complete"]
        if not isinstance(pass_a_complete, bool):
            raise ValueError(f"phase2j human record {index} pass_a_complete is invalid")
        if pass_a_complete and (not reviewer_name.strip() or not completed_at):
            raise ValueError(
                f"phase2j human record {index} Pass A completion requires identity",
            )
        if window_status == "UNREVIEWED":
            if outcome != "CLEAN" or human["endpoints"] \
                    or note or reviewer_name or completed_at or pass_a_complete:
                raise ValueError(
                    f"phase2j human record {index} UNREVIEWED window must remain blank",
                )
        elif window_status == "IN_REVIEW":
            if outcome != "CLEAN":
                raise ValueError(
                    f"phase2j human record {index} IN_REVIEW window must be CLEAN",
                )
        elif window_status == "AMBIGUOUS":
            if outcome != "AMBIGUOUS" or not note.strip():
                raise ValueError(
                    f"phase2j human record {index} AMBIGUOUS window requires a note",
                )
        elif window_status == "EXCLUDED":
            if outcome != "EXCLUDED" or not note.strip():
                raise ValueError(
                    f"phase2j human record {index} EXCLUDED window requires a note",
                )
            if human["endpoints"]:
                raise ValueError(
                    f"phase2j human record {index} EXCLUDED window cannot have endpoints",
                )
        endpoints = human["endpoints"]
        if not isinstance(endpoints, list):
            raise ValueError(f"phase2j human record {index} endpoints must be a list")
        expected_prefix = f"p2j:review:{human['window_id']}:ep:"
        for endpoint_index, endpoint in enumerate(endpoints, 1):
            if not isinstance(endpoint, Mapping) \
                    or set(endpoint) != _EXPECTED_HUMAN_ENDPOINT_INPUT_KEYS:
                raise ValueError(
                    f"phase2j human endpoint {endpoint_index} in record {index} is invalid",
                )
            endpoint_id = _require_string(
                endpoint["endpoint_id"], f"human endpoint {endpoint_index} id",
            )
            if not endpoint_id.startswith(expected_prefix) \
                    or not re.fullmatch(r"[0-9]{4}", endpoint_id[len(expected_prefix):]):
                raise ValueError(
                    f"phase2j human endpoint id is not bound to its window in {human['window_id']}",
                )
            _validate_span_endpoint(
                endpoint,
                bronze_text=human["bronze_text"],
                tokens=tokens,
                window_id=human["window_id"],
                label=f"human endpoint {endpoint_index}",
            )
            node_type = endpoint["node_type"]
            if node_type not in ENDPOINT_TYPES:
                raise ValueError(
                    f"phase2j human endpoint {endpoint_index} node_type is invalid",
                )
            if endpoint["ambiguity_state"] != "NONE" \
                    or endpoint["disposition"] != "KEEP" \
                    or endpoint["pass_provenance"] != "PASS_A" \
                    or endpoint["human_accepted"] is not True:
                raise ValueError(
                    f"phase2j human endpoint {endpoint_index} must be a KEEP Pass A endpoint",
                )
            _require_int(
                endpoint["created_sequence"], f"human endpoint {endpoint_index} sequence",
            )
        _validate_no_internal_overlap(
            endpoints, window_id=human["window_id"], label="human",
        )


def validate_sol_review(
    review: Mapping[str, Any],
    packet: Mapping[str, Any],
) -> None:
    """Strict validation of the sealed Sol parallel review against the packet."""
    if set(review) != _EXPECTED_SOL_REVIEW_KEYS:
        raise ValueError("phase2j Sol review envelope is invalid")
    if review["schema_version"] != SOL_REVIEW_SCHEMA_VERSION:
        raise ValueError("phase2j Sol review schema version is unsupported")
    if not isinstance(review["reviewer_model"], str) or not review["reviewer_model"]:
        raise ValueError("phase2j Sol reviewer_model is invalid")
    if not isinstance(review["reasoning_effort"], str) or not review["reasoning_effort"]:
        raise ValueError("phase2j Sol reasoning_effort is invalid")
    if not isinstance(review["purpose"], str) \
            or "NOT GOLD" not in review["purpose"].upper():
        raise ValueError("phase2j Sol review must declare itself NOT GOLD")
    if review["visibility_gate"] not in SOL_VISIBILITY_GATES:
        raise ValueError("phase2j Sol visibility gate is invalid")
    inner = {key: value for key, value in review.items() if key != "content_sha256"}
    if review["content_sha256"] != canonical_sha256(inner):
        raise ValueError("phase2j Sol review content hash is invalid")
    if review["blank_packet_sha256"] != packet["content_sha256"]:
        raise ValueError("phase2j Sol review is not bound to the locked packet")
    if review["selection_manifest_sha256"] != packet["selection_manifest_sha256"]:
        raise ValueError("phase2j Sol review is not bound to the selection manifest")
    records = review["records"]
    if not isinstance(records, list) or len(records) != len(packet["records"]):
        raise ValueError(
            "phase2j Sol review must contain exactly "
            f"{len(packet['records'])} windows",
        )
    packet_by_window = {record["window_id"]: record for record in packet["records"]}
    for index, (sol_record, locked) in enumerate(zip(records, packet["records"]), 1):
        if not isinstance(sol_record, Mapping) or set(sol_record) != _EXPECTED_SOL_RECORD_KEYS:
            raise ValueError(f"phase2j Sol record {index} is invalid")
        if sol_record["record_index"] != index:
            raise ValueError(f"phase2j Sol record_index must be {index}")
        if sol_record["window_id"] != locked["window_id"]:
            raise ValueError(
                f"phase2j Sol record {index} window_id contradicts the locked packet",
            )
        if sol_record["bronze_text_sha256"] != locked["bronze_text_sha256"]:
            raise ValueError(
                f"phase2j Sol record {index} bronze hash contradicts the locked packet",
            )
        for field in ("window_ambiguity_notes", "omission_audit_notes"):
            value = sol_record[field]
            if not isinstance(value, list) \
                    or any(not isinstance(item, str) for item in value):
                raise ValueError(f"phase2j Sol record {index} {field} must be strings")
        proposed = sol_record["proposed_endpoints"]
        if not isinstance(proposed, list):
            raise ValueError(f"phase2j Sol record {index} proposed_endpoints must be a list")
        tokens = locked["tokens"]
        for endpoint_index, endpoint in enumerate(proposed, 1):
            if not isinstance(endpoint, Mapping) \
                    or set(endpoint) != _EXPECTED_SOL_ENDPOINT_INPUT_KEYS:
                raise ValueError(
                    f"phase2j Sol endpoint {endpoint_index} in record {index} is invalid",
                )
            token_start = _require_int(
                endpoint["token_start"], f"Sol endpoint {endpoint_index} token_start",
            )
            token_end = _require_int(
                endpoint["token_end"], f"Sol endpoint {endpoint_index} token_end",
            )
            if not 0 <= token_start <= token_end < len(tokens):
                raise ValueError(
                    f"phase2j Sol endpoint {endpoint_index} token range is invalid "
                    f"in {sol_record['window_id']}",
                )
            char_start = tokens[token_start]["start"]
            char_end = tokens[token_end]["end"]
            exact = _require_string(
                endpoint["exact_bronze_text"], f"Sol endpoint {endpoint_index} text",
            )
            if locked["bronze_text"][char_start:char_end] != exact:
                raise ValueError(
                    f"phase2j Sol endpoint {endpoint_index} is not an exact Bronze "
                    f"slice in {sol_record['window_id']}",
                )
            node_type = endpoint["node_type"]
            if node_type is not None and node_type not in ENDPOINT_TYPES:
                raise ValueError(
                    f"phase2j Sol endpoint {endpoint_index} node_type is invalid",
                )
            if endpoint["pass_provenance"] != SOL_NON_GOLD_PROVENANCE:
                raise ValueError(
                    f"phase2j Sol endpoint {endpoint_index} provenance is invalid",
                )
            if endpoint["ambiguity_state"] not in SOL_AMBIGUITY_STATES:
                raise ValueError(
                    f"phase2j Sol endpoint {endpoint_index} ambiguity_state is invalid",
                )
            _require_string(
                endpoint["concise_rationale"], f"Sol endpoint {endpoint_index} rationale",
            )
        _validate_no_internal_overlap(
            [
                {
                    **endpoint,
                    "endpoint_id": f"p2j:sol:{sol_record['window_id']}:ep:"
                    f"{str(endpoint_index).zfill(4)}",
                    "char_start": tokens[endpoint["token_start"]]["start"],
                    "char_end": tokens[endpoint["token_end"]]["end"],
                }
                for endpoint_index, endpoint in enumerate(proposed, 1)
            ],
            window_id=sol_record["window_id"],
            label="Sol",
        )


def _sanitize_human_endpoints(
    record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "endpoint_id": endpoint["endpoint_id"],
            "exact_bronze_text": endpoint["exact_bronze_text"],
            "char_start": endpoint["char_start"],
            "char_end": endpoint["char_end"],
            "token_start": endpoint["token_start"],
            "token_end": endpoint["token_end"],
            "node_type": endpoint["node_type"],
        }
        for endpoint in record["endpoints"]
    ]


def _sanitize_sol_endpoints(
    sol_record: Mapping[str, Any],
    locked: Mapping[str, Any],
) -> list[dict[str, Any]]:
    tokens = locked["tokens"]
    output = []
    for index, endpoint in enumerate(sol_record["proposed_endpoints"], 1):
        char_start = tokens[endpoint["token_start"]]["start"]
        char_end = tokens[endpoint["token_end"]]["end"]
        output.append({
            "endpoint_id": f"p2j:sol:{sol_record['window_id']}:ep:{str(index).zfill(4)}",
            "exact_bronze_text": endpoint["exact_bronze_text"],
            "char_start": char_start,
            "char_end": char_end,
            "token_start": endpoint["token_start"],
            "token_end": endpoint["token_end"],
            "node_type": endpoint["node_type"],
            "sol_ambiguity_state": endpoint["ambiguity_state"],
            "sol_rationale": endpoint["concise_rationale"],
        })
    return output


def _classify_component(
    human: list[Mapping[str, Any]],
    sol: list[Mapping[str, Any]],
) -> str:
    if human and sol:
        if len(human) == 1 and len(sol) == 1:
            human_span = (
                human[0]["token_start"], human[0]["token_end"],
                human[0]["node_type"],
            )
            sol_span = (
                sol[0]["token_start"], sol[0]["token_end"],
                sol[0]["node_type"],
            )
            if human_span == sol_span:
                return "EXACT_AGREEMENT"
            if (human[0]["token_start"], human[0]["token_end"]) == (
                sol[0]["token_start"], sol[0]["token_end"],
            ):
                return "TYPE_DISAGREEMENT"
        return "BOUNDARY_DISAGREEMENT"
    if sol:
        return "SOL_ONLY"
    return "HUMAN_ONLY"


def build_components(
    window_id: str,
    human_endpoints: list[Mapping[str, Any]],
    sol_endpoints: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Connected components of inclusive overlapping token intervals."""
    nodes: list[dict[str, str | Mapping[str, Any]]] = []
    for endpoint in human_endpoints:
        nodes.append({"side": "H", "endpoint": endpoint})
    for endpoint in sol_endpoints:
        nodes.append({"side": "S", "endpoint": endpoint})
    parent = list(range(len(nodes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left_index in range(len(nodes)):
        left = nodes[left_index]
        left_span = (
            left["endpoint"]["token_start"], left["endpoint"]["token_end"],
        )
        for right_index in range(left_index + 1, len(nodes)):
            right = nodes[right_index]
            right_span = (
                right["endpoint"]["token_start"], right["endpoint"]["token_end"],
            )
            if _token_interval_overlaps(left_span, right_span):
                union(left_index, right_index)

    grouped: dict[int, list[int]] = {}
    for index in range(len(nodes)):
        grouped.setdefault(find(index), []).append(index)

    # Deterministic Bronze source-position ordering: components are numbered
    # by the earliest covered token, then the furthest covered token, then
    # the stable side/id identity of the earliest member endpoint.  Sorting
    # by union-find member indexes instead would order by annotation side
    # (all Human nodes precede all Sol nodes), which is not a stable Bronze
    # source order.  Connected-component membership is unchanged.
    ordered: list[
        tuple[
            tuple[int, int, tuple[str, str]],
            list[Mapping[str, Any]],
            list[Mapping[str, Any]],
        ]
    ] = []
    for indexes in grouped.values():
        member_endpoints = [nodes[index] for index in indexes]
        human = [
            item["endpoint"] for item in member_endpoints if item["side"] == "H"
        ]
        sol = [
            item["endpoint"] for item in member_endpoints if item["side"] == "S"
        ]
        source_key = (
            min(item["endpoint"]["token_start"] for item in member_endpoints),
            max(item["endpoint"]["token_end"] for item in member_endpoints),
            min(
                (item["side"], item["endpoint"]["endpoint_id"])
                for item in member_endpoints
            ),
        )
        ordered.append((source_key, human, sol))

    components: list[dict[str, Any]] = []
    for sequence, (_, human, sol) in enumerate(
        sorted(ordered, key=lambda item: item[0]), 1,
    ):
        classification = _classify_component(human, sol)
        components.append({
            "component_id": f"p2j:adjudicate:{window_id}:c:{str(sequence).zfill(4)}",
            "classification": classification,
            "human_endpoint_ids": [item["endpoint_id"] for item in human],
            "sol_endpoint_ids": [item["endpoint_id"] for item in sol],
        })
    return components


def _build_record(
    locked: Mapping[str, Any],
    human: Mapping[str, Any],
    sol: Mapping[str, Any],
) -> dict[str, Any]:
    human_endpoints = _sanitize_human_endpoints(human)
    sol_endpoints = _sanitize_sol_endpoints(sol, locked)
    components = build_components(
        locked["window_id"], human_endpoints, sol_endpoints,
    )
    return {
        "record_index": locked["record_index"],
        "window_id": locked["window_id"],
        "source_group_id": locked["source_group_id"],
        "bronze_text": locked["bronze_text"],
        "bronze_text_sha256": locked["bronze_text_sha256"],
        "bronze_char_length": locked["bronze_char_length"],
        "tokens": locked["tokens"],
        "human_outcome": human["outcome"],
        "human_endpoints": human_endpoints,
        "sol_endpoints": sol_endpoints,
        "components": components,
    }


def build_adjudication_packet(
    packet: Mapping[str, Any],
    human_session: Mapping[str, Any],
    sol_review: Mapping[str, Any],
    *,
    human_session_path: Path,
    sol_review_path: Path,
) -> dict[str, Any]:
    """Build the deterministic sanitized adjudication packet."""
    validate_human_session(human_session, packet)
    validate_sol_review(sol_review, packet)
    records = [
        _build_record(locked, human, sol)
        for locked, human, sol in zip(
            packet["records"], human_session["records"], sol_review["records"],
        )
    ]
    totals = _compute_totals(records)
    inner = {
        "schema_version": ADJUDICATION_PACKET_SCHEMA_VERSION,
        "adjudication_version": ADJUDICATION_VERSION,
        "annotation_version": ANNOTATION_VERSION,
        "packet_schema_version": PACKET_SCHEMA_VERSION,
        "packet_sha256": packet["content_sha256"],
        "human_session_schema_version": HUMAN_SESSION_SCHEMA_VERSION,
        "human_session_sha256": file_sha256(human_session_path),
        "sol_review_schema_version": SOL_REVIEW_SCHEMA_VERSION,
        "sol_review_sha256": file_sha256(sol_review_path),
        "visibility_gate": VISIBILITY_GATE,
        "purpose": (
            "Post-Pass-A human-vs-Sol adjudication deck for the 30 locked Phase "
            "2J Bronze windows. Sol proposals are a second opinion "
            "(navigation/audit only) and are never gold; the resolved output "
            "remains REVIEW MATERIAL until a separately validated canonical "
            "import/finalizer runs."
        ),
        "totals": totals,
        "records": records,
    }
    packet_built = {
        "content_sha256": canonical_sha256(inner), **inner,
    }
    validate_adjudication_packet(packet_built)
    return packet_built


def _compute_totals(records: list[Mapping[str, Any]]) -> dict[str, int]:
    totals = {
        "windows": len(records),
        "components": 0,
        "exact_agreements": 0,
        "type_disagreements": 0,
        "boundary_disagreements": 0,
        "sol_only": 0,
        "human_only": 0,
        "human_endpoints": 0,
        "sol_endpoints": 0,
    }
    for record in records:
        totals["components"] += len(record["components"])
        totals["human_endpoints"] += len(record["human_endpoints"])
        totals["sol_endpoints"] += len(record["sol_endpoints"])
        for component in record["components"]:
            totals[{
                "EXACT_AGREEMENT": "exact_agreements",
                "TYPE_DISAGREEMENT": "type_disagreements",
                "BOUNDARY_DISAGREEMENT": "boundary_disagreements",
                "SOL_ONLY": "sol_only",
                "HUMAN_ONLY": "human_only",
            }[component["classification"]]] += 1
    return totals


def _validate_components(
    record: Mapping[str, Any],
) -> None:
    human_by_id = {item["endpoint_id"]: item for item in record["human_endpoints"]}
    sol_by_id = {item["endpoint_id"]: item for item in record["sol_endpoints"]}
    components = record["components"]
    if not isinstance(components, list):
        raise ValueError(f"phase2j record {record['record_index']} components must be a list")
    seen_component_ids: set[str] = set()
    covered_human: set[str] = set()
    covered_sol: set[str] = set()
    for sequence, component in enumerate(components, 1):
        if not isinstance(component, Mapping) or set(component) != _EXPECTED_COMPONENT_KEYS:
            raise ValueError(f"phase2j component {sequence} in record {record['record_index']} is invalid")
        component_id = component["component_id"]
        expected_id = (
            f"p2j:adjudicate:{record['window_id']}:c:{str(sequence).zfill(4)}"
        )
        if component_id != expected_id or component_id in seen_component_ids:
            raise ValueError(f"phase2j component ids must be unique and sequential in {record['window_id']}")
        seen_component_ids.add(component_id)
        classification = component["classification"]
        if classification not in COMPONENT_CLASSES:
            raise ValueError(f"phase2j component {component_id} classification is invalid")
        human_ids = component["human_endpoint_ids"]
        sol_ids = component["sol_endpoint_ids"]
        if not isinstance(human_ids, list) or any(
            item not in human_by_id for item in human_ids
        ):
            raise ValueError(f"phase2j component {component_id} human references are invalid")
        if not isinstance(sol_ids, list) or any(
            item not in sol_by_id for item in sol_ids
        ):
            raise ValueError(f"phase2j component {component_id} Sol references are invalid")
        if len(set(human_ids)) != len(human_ids) or len(set(sol_ids)) != len(sol_ids):
            raise ValueError(f"phase2j component {component_id} references are duplicated")
        overlap = set(human_ids) & covered_human
        if overlap:
            raise ValueError(
                f"phase2j human endpoint {sorted(overlap)[0]} appears in multiple components",
            )
        overlap = set(sol_ids) & covered_sol
        if overlap:
            raise ValueError(
                f"phase2j Sol endpoint {sorted(overlap)[0]} appears in multiple components",
            )
        covered_human.update(human_ids)
        covered_sol.update(sol_ids)
        human = [human_by_id[item] for item in human_ids]
        sol = [sol_by_id[item] for item in sol_ids]
        expected = _classify_component(human, sol)
        if expected != classification:
            raise ValueError(
                f"phase2j component {component_id} classification is inconsistent "
                f"(expected {expected})",
            )
        if classification in {"EXACT_AGREEMENT", "TYPE_DISAGREEMENT"} \
                and (len(human) != 1 or len(sol) != 1):
            raise ValueError(
                f"phase2j component {component_id} must contain exactly one span per side",
            )
    if set(covered_human) != set(human_by_id) or set(covered_sol) != set(sol_by_id):
        raise ValueError(
            f"phase2j components must cover every endpoint in {record['window_id']}",
        )


def validate_adjudication_packet(packet: Mapping[str, Any]) -> None:
    """Validate the adjudication packet envelope, hashes, records, and totals."""
    if not isinstance(packet, Mapping) or set(packet) != _EXPECTED_PACKET_KEYS:
        raise ValueError("phase2j adjudication packet envelope is invalid")
    if packet["schema_version"] != ADJUDICATION_PACKET_SCHEMA_VERSION:
        raise ValueError("phase2j adjudication packet version is unsupported")
    if packet["adjudication_version"] != ADJUDICATION_VERSION:
        raise ValueError("phase2j adjudication version is unsupported")
    if packet["annotation_version"] != ANNOTATION_VERSION:
        raise ValueError("phase2j annotation version is unsupported")
    if packet["packet_schema_version"] != PACKET_SCHEMA_VERSION:
        raise ValueError("phase2j packet schema version is unsupported")
    if not _SHA256.fullmatch(packet["packet_sha256"]):
        raise ValueError("phase2j adjudication packet binding hash is invalid")
    if packet["human_session_schema_version"] != HUMAN_SESSION_SCHEMA_VERSION:
        raise ValueError("phase2j human session schema version is unsupported")
    if packet["sol_review_schema_version"] != SOL_REVIEW_SCHEMA_VERSION:
        raise ValueError("phase2j Sol review schema version is unsupported")
    if not _SHA256.fullmatch(packet["human_session_sha256"]) \
            or not _SHA256.fullmatch(packet["sol_review_sha256"]):
        raise ValueError("phase2j adjudication input hashes are invalid")
    if packet["visibility_gate"] != VISIBILITY_GATE:
        raise ValueError("phase2j adjudication visibility gate is invalid")
    if not isinstance(packet["purpose"], str) or not packet["purpose"]:
        raise ValueError("phase2j adjudication purpose is invalid")
    inner = {key: value for key, value in packet.items() if key != "content_sha256"}
    if packet["content_sha256"] != canonical_sha256(inner):
        raise ValueError("phase2j adjudication packet content hash is invalid")
    _validate_forbidden_content(packet, OUTPUT_FORBIDDEN_KEYS)
    totals = packet["totals"]
    if not isinstance(totals, Mapping) or set(totals) != _EXPECTED_TOTALS_KEYS:
        raise ValueError("phase2j adjudication totals are invalid")
    records = packet["records"]
    if not isinstance(records, list) or len(records) != 30:
        raise ValueError("phase2j adjudication packet must contain exactly 30 records")
    seen_window_ids: set[str] = set()
    for index, record in enumerate(records, 1):
        if not isinstance(record, Mapping) or set(record) != _EXPECTED_RECORD_KEYS:
            raise ValueError(f"phase2j adjudication record {index} is invalid")
        if record["record_index"] != index:
            raise ValueError(f"phase2j adjudication record_index must be {index}")
        window_id = record["window_id"]
        if not isinstance(window_id, str) or window_id in seen_window_ids:
            raise ValueError(f"phase2j adjudication window identity is invalid in record {index}")
        seen_window_ids.add(window_id)
        if not isinstance(record["source_group_id"], str) or not record["source_group_id"]:
            raise ValueError(f"phase2j adjudication record {index} source_group_id is invalid")
        bronze_text = record["bronze_text"]
        if not isinstance(bronze_text, str) or not bronze_text.strip():
            raise ValueError(f"phase2j adjudication record {index} bronze_text is invalid")
        if record["bronze_text_sha256"] != hashlib.sha256(
            bronze_text.encode("utf-8"),
        ).hexdigest():
            raise ValueError(f"phase2j adjudication record {index} bronze hash is invalid")
        if isinstance(record["bronze_char_length"], bool) \
                or record["bronze_char_length"] != len(bronze_text):
            raise ValueError(f"phase2j adjudication record {index} bronze length is invalid")
        tokens = _validate_tokens_match(
            bronze_text, record["tokens"], window_id=window_id,
        )
        if record["human_outcome"] not in WINDOW_OUTCOMES:
            raise ValueError(f"phase2j adjudication record {index} human_outcome is invalid")
        human_endpoints = record["human_endpoints"]
        sol_endpoints = record["sol_endpoints"]
        if not isinstance(human_endpoints, list) or not isinstance(sol_endpoints, list):
            raise ValueError(f"phase2j adjudication record {index} endpoint lists are invalid")
        for endpoint_index, endpoint in enumerate(human_endpoints, 1):
            if not isinstance(endpoint, Mapping) \
                    or set(endpoint) != _EXPECTED_HUMAN_ENDPOINT_KEYS:
                raise ValueError(
                    f"phase2j adjudication human endpoint {endpoint_index} in record {index} is invalid",
                )
            if not isinstance(endpoint["endpoint_id"], str) \
                    or not endpoint["endpoint_id"].startswith(
                        f"p2j:review:{window_id}:ep:",
                    ):
                raise ValueError(
                    f"phase2j adjudication human endpoint id is invalid in {window_id}",
                )
            _validate_span_endpoint(
                endpoint,
                bronze_text=bronze_text,
                tokens=tokens,
                window_id=window_id,
                label=f"adjudication human endpoint {endpoint_index}",
            )
            if endpoint["node_type"] not in ENDPOINT_TYPES:
                raise ValueError(
                    f"phase2j adjudication human endpoint {endpoint_index} node_type is invalid",
                )
        for endpoint_index, endpoint in enumerate(sol_endpoints, 1):
            if not isinstance(endpoint, Mapping) \
                    or set(endpoint) != _EXPECTED_SOL_ENDPOINT_KEYS:
                raise ValueError(
                    f"phase2j adjudication Sol endpoint {endpoint_index} in record {index} is invalid",
                )
            if not isinstance(endpoint["endpoint_id"], str) \
                    or not endpoint["endpoint_id"].startswith(
                        f"p2j:sol:{window_id}:ep:",
                    ):
                raise ValueError(
                    f"phase2j adjudication Sol endpoint id is invalid in {window_id}",
                )
            _validate_span_endpoint(
                endpoint,
                bronze_text=bronze_text,
                tokens=tokens,
                window_id=window_id,
                label=f"adjudication Sol endpoint {endpoint_index}",
            )
            if endpoint["node_type"] is not None \
                    and endpoint["node_type"] not in ENDPOINT_TYPES:
                raise ValueError(
                    f"phase2j adjudication Sol endpoint {endpoint_index} node_type is invalid",
                )
            if endpoint["sol_ambiguity_state"] not in SOL_AMBIGUITY_STATES:
                raise ValueError(
                    f"phase2j adjudication Sol endpoint {endpoint_index} ambiguity state is invalid",
                )
            _require_string(
                endpoint["sol_rationale"],
                f"adjudication Sol endpoint {endpoint_index} rationale",
            )
        _validate_no_internal_overlap(
            human_endpoints, window_id=window_id, label="human",
        )
        _validate_no_internal_overlap(
            sol_endpoints, window_id=window_id, label="Sol",
        )
        _validate_components(record)
    computed = _compute_totals(records)
    if totals != computed:
        raise ValueError("phase2j adjudication totals are inconsistent")
    if totals["windows"] != 30:
        raise ValueError("phase2j adjudication totals must report 30 windows")


def load_adjudication_packet(path: Path) -> dict[str, Any]:
    """Strict canonical load of the adjudication packet."""
    return _load_json_strict(path, label="phase2j adjudication packet")


__all__ = [
    "ADJUDICATION_PACKET_SCHEMA_VERSION", "ADJUDICATION_VERSION",
    "COMPONENT_CLASSES", "ENDPOINT_TYPES", "HUMAN_SESSION_SCHEMA_VERSION",
    "OUTPUT_FORBIDDEN_KEYS", "SOL_REVIEW_SCHEMA_VERSION",
    "VISIBILITY_GATE", "build_adjudication_packet", "build_components",
    "load_adjudication_packet", "validate_adjudication_packet",
    "validate_human_session", "validate_sol_review",
]
