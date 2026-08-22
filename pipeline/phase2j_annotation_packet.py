"""Phase 2J scorer-blind two-pass endpoint annotation packet.

The annotation-facing packet contains exactly one record per selected window,
the exact Bronze text, a deterministic whitespace-token table, and a blank
endpoint list.  Pass A is endpoint discovery; Pass B is a later blinded
boundary/omission/role/duplicate/ambiguity audit.  Pass B cannot complete
before Pass A validates, and a packet is gold-eligible only when both passes
complete with no unresolved ambiguity, exclusion, or adjudication.

The packet deliberately contains no probabilities, scores, ranks, selected or
predicted labels, syntax features/importances, model error taxonomy, or model
suggestions (enforced recursively).  The frozen candidate catalog is bound by
generator version, count, and hash only; candidate suggestions are never
exposed in Pass A.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from pipeline.semantic_ir import AmbiguityState, NodeType
from pipeline.semantic_mentions import MENTION_CATALOG_VERSION
from pipeline.phase2j_source_selection import (
    PARTITION_SIZES,
    SELECTION_SCHEMA_VERSION,
    canonical_sha256,
    validate_selection_manifest,
)


PACKET_SCHEMA_VERSION = "phase2j-endpoint-annotation-packet-v1"
ANNOTATION_VERSION = "phase2j-endpoint-annotation-v1"
RELEASE_GATE = "LOCKED"
WINDOW_STATUSES = (
    "UNREVIEWED", "IN_REVIEW", "REVIEWED", "AMBIGUOUS", "EXCLUDED",
    "ADJUDICATION_REQUIRED",
)
PASS_A_STATUSES = ("PENDING", "IN_PROGRESS", "COMPLETE")
PASS_B_STATUSES = ("LOCKED_AWAITING_PASS_A", "PENDING", "IN_PROGRESS", "COMPLETE")
ENDPOINT_DISPOSITIONS = ("KEEP", "AMBIGUOUS", "EXCLUDED", "ADJUDICATION_REQUIRED")
PASS_PROVENANCES = ("PASS_A", "PASS_B")
NODE_TYPES = frozenset(item.value for item in NodeType)
AMBIGUITY_STATES = frozenset(item.value for item in AmbiguityState)
AUDIT_CHECKS = ("boundaries", "omissions", "roles", "duplicates", "ambiguity")

# Recursive forbidden-key validation for annotation-facing content.  Exact
# case-insensitive key matches; floating point values are also rejected because
# the blank packet contains only strings, integers, booleans, and nulls.
FORBIDDEN_KEYS = frozenset({
    "score", "scores", "probability", "probabilities", "confidence",
    "rank", "ranks", "ranked", "ranking", "rankings",
    "prediction", "predictions", "predicted", "predicted_label",
    "predicted_labels", "label", "labels", "gold_label", "gold_labels",
    "syntax_importance", "syntax_importances", "feature_importance",
    "feature_importances", "importance", "importances", "error_taxonomy",
    "model_suggestion", "model_suggestions", "suggestion", "suggestions",
    "model_id", "model_name", "model_score", "logits", "proba",
})

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"\S+")
_EXPECTED_PACKET_KEYS = frozenset({
    "content_sha256", "schema_version", "purpose", "annotation_version",
    "release_gate", "selection_manifest_sha256",
    "selection_manifest_schema_version", "candidate_generator_version",
    "candidate_catalog", "rules", "records",
})
_EXPECTED_RULES_KEYS = frozenset({
    "window_statuses", "endpoint_dispositions", "pass_a", "pass_b",
    "in_review_rule", "pass_b_requires_pass_a", "gold_eligibility_rule",
    "overlap_rule", "non_keep_rule", "reviewer_instructions",
})
_EXPECTED_RECORD_KEYS = frozenset({
    "record_index", "annotation_id", "source_group_id", "window_id",
    "upstream_source_id", "upstream_start", "upstream_end", "partition",
    "bronze_text", "bronze_text_sha256", "bronze_char_length", "tokens",
    "endpoints", "window_status", "pass_a", "pass_b", "ambiguity_controls",
    "exclusion_controls", "reviewer_notes",
})
_EXPECTED_PASS_A_KEYS = frozenset({
    "status", "reviewer", "completed_at", "notes", "endpoint_count",
})
_EXPECTED_PASS_B_KEYS = frozenset({
    "status", "reviewer", "completed_at", "notes", "audit_checks",
})
_EXPECTED_ENDPOINT_KEYS = frozenset({
    "endpoint_id", "bronze_text", "char_start", "char_end", "token_start",
    "token_end", "node_type", "ambiguity_state", "disposition",
    "adjudication_requested", "notes", "pass_provenance",
})


def token_table(text: str) -> list[dict[str, Any]]:
    """Deterministic whitespace-token table with index/start/end/text."""
    return [
        {
            "token_index": index,
            "start": match.start(),
            "end": match.end(),
            "text": match.group(),
        }
        for index, match in enumerate(_TOKEN.finditer(text))
    ]


def _validate_token_table(text: str, tokens: object) -> list[dict[str, Any]]:
    if not isinstance(tokens, list):
        raise ValueError("phase2j token table must be a list")
    previous_end = 0
    seen_first = False
    seen_last = False
    validated: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        if not isinstance(token, Mapping) or set(token) != {
            "token_index", "start", "end", "text",
        }:
            raise ValueError("phase2j token record is invalid")
        if token["token_index"] != index:
            raise ValueError("phase2j token indices must be sequential")
        start, end = token["start"], token["end"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)) \
                or not 0 <= start < end <= len(text):
            raise ValueError("phase2j token offsets are invalid")
        if not isinstance(token["text"], str) or text[start:end] != token["text"]:
            raise ValueError("phase2j token text is not an exact source slice")
        if index > 0 and start <= previous_end:
            raise ValueError("phase2j tokens must be ordered and non-overlapping")
        if text[previous_end:start].strip():
            raise ValueError("phase2j token table discarded non-whitespace source text")
        previous_end = end
        if index == 0:
            seen_first = start == 0 or not text[:start].strip()
        validated.append(dict(token))
    if tokens:
        seen_last = previous_end == len(text) or not text[previous_end:].strip()
    if not tokens and text.strip():
        raise ValueError("phase2j token table must cover nonempty bronze text")
    if text.strip() and not (seen_first and seen_last):
        raise ValueError("phase2j token table must cover the full bronze text")
    return validated


def _validate_forbidden_content(value: object, *, path: tuple[str, ...] = ()) -> None:
    """Recursively reject forbidden scorer/model fields and floating scores."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("phase2j annotation keys must be strings")
            if key.casefold() in FORBIDDEN_KEYS:
                raise ValueError(
                    "phase2j annotation-facing content contains forbidden key "
                    + repr(key) + " at " + ".".join(path + (key,)),
                )
            _validate_forbidden_content(item, path=path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_forbidden_content(item, path=path + (f"[{index}]",))
    elif isinstance(value, float):
        raise ValueError(
            "phase2j annotation-facing content contains a floating-point value",
        )


def _blank_pass_a() -> dict[str, Any]:
    return {
        "status": "PENDING",
        "reviewer": None,
        "completed_at": None,
        "notes": [],
        "endpoint_count": 0,
    }


def _blank_pass_b() -> dict[str, Any]:
    return {
        "status": "LOCKED_AWAITING_PASS_A",
        "reviewer": None,
        "completed_at": None,
        "notes": [],
        "audit_checks": {key: False for key in AUDIT_CHECKS},
    }


def _rules() -> dict[str, Any]:
    return {
        "window_statuses": list(WINDOW_STATUSES),
        "endpoint_dispositions": list(ENDPOINT_DISPOSITIONS),
        "pass_a": (
            "PASS_A endpoint discovery: mark every existing source endpoint; "
            "the window moves from UNREVIEWED to IN_REVIEW while pass_a runs "
            "and remains IN_REVIEW after pass_a completes until pass_b starts."
        ),
        "pass_b": (
            "PASS_B later blinded audit: boundary/omission/role/duplicate/"
            "ambiguity checks; may flag or mark endpoints.  A window remains "
            "IN_REVIEW during pass_b and reaches a final status only when "
            "pass_b completes."
        ),
        "in_review_rule": (
            "IN_REVIEW is the intermediate clean two-pass status: valid with "
            "pass_a IN_PROGRESS and pass_b LOCKED_AWAITING_PASS_A, or with "
            "pass_a COMPLETE and pass_b PENDING or IN_PROGRESS.  It holds only "
            "KEEP dispositions, never ambiguity/exclusion/adjudication flags, "
            "and is never gold-eligible."
        ),
        "pass_b_requires_pass_a": True,
        "gold_eligibility_rule": (
            "A record is gold-eligible only when pass_a.status == COMPLETE, "
            "pass_b.status == COMPLETE, window_status == REVIEWED, every "
            "endpoint disposition == KEEP, no ambiguity/exclusion/adjudication "
            "flags, and every Pass B audit check is true."
        ),
        "overlap_rule": (
            "Duplicate or overlapping endpoint annotations are rejected unless "
            "every overlapping pair is explicitly marked "
            "adjudication_requested=true; such a window must carry "
            "window_status == ADJUDICATION_REQUIRED and is never gold-eligible."
        ),
        "non_keep_rule": (
            "AMBIGUOUS, EXCLUDED, and ADJUDICATION_REQUIRED endpoint "
            "dispositions never silently become KEEP; they block gold "
            "eligibility and require the matching window status."
        ),
        "reviewer_instructions": (
            "Do not use model suggestions, scores, ranks, probabilities, "
            "labels, syntax features, or error taxonomy.  Partition is a "
            "machine-integrity field; do not treat FROZEN_REPLICATION records "
            "differently from EXPANDED_DEV records during review."
        ),
    }


def build_annotation_packet(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Build the blank two-pass annotation packet from a validated manifest."""
    validate_selection_manifest(manifest)
    records = []
    for index, selected in enumerate(manifest["selected"], 1):
        bronze_text = selected["source_text"]
        tokens = token_table(bronze_text)
        records.append({
            "record_index": index,
            "annotation_id": f"p2j:{selected['window_id']}",
            "source_group_id": selected["source_group_id"],
            "window_id": selected["window_id"],
            "upstream_source_id": selected["upstream_source_id"],
            "upstream_start": selected["upstream_start"],
            "upstream_end": selected["upstream_end"],
            "partition": selected["partition"],
            "bronze_text": bronze_text,
            "bronze_text_sha256": hashlib.sha256(
                bronze_text.encode("utf-8"),
            ).hexdigest(),
            "bronze_char_length": len(bronze_text),
            "tokens": tokens,
            "endpoints": [],
            "window_status": "UNREVIEWED",
            "pass_a": _blank_pass_a(),
            "pass_b": _blank_pass_b(),
            "ambiguity_controls": {"flagged": False, "notes": []},
            "exclusion_controls": {"flagged": False, "notes": []},
            "reviewer_notes": [],
        })
    candidate_catalog = {
        "count": int(manifest["diversity_summary"]["candidate_count"]),
        "per_window": {
            selected["window_id"]: {
                "count": selected["candidate_count"],
                "catalog_sha256": selected["candidate_catalog_sha256"],
            }
            for selected in manifest["selected"]
        },
    }
    inner = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "purpose": (
            "Blank scorer-blind two-pass endpoint annotation packet for the "
            "Phase 2J pre-annotation checkpoint; no valid new gold exists yet."
        ),
        "annotation_version": ANNOTATION_VERSION,
        "release_gate": RELEASE_GATE,
        "selection_manifest_sha256": manifest["content_sha256"],
        "selection_manifest_schema_version": SELECTION_SCHEMA_VERSION,
        "candidate_generator_version": MENTION_CATALOG_VERSION,
        "candidate_catalog": candidate_catalog,
        "rules": _rules(),
        "records": records,
    }
    packet = {"content_sha256": canonical_sha256(inner), **inner}
    validate_annotation_packet(packet, manifest=manifest)
    return packet


def _validate_pass_a(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _EXPECTED_PASS_A_KEYS:
        raise ValueError("phase2j pass_a review record is invalid")
    status = value["status"]
    if status not in PASS_A_STATUSES:
        raise ValueError("phase2j pass_a status is invalid")
    reviewer = value["reviewer"]
    completed_at = value["completed_at"]
    if status == "COMPLETE":
        if not isinstance(reviewer, str) or not reviewer \
                or not isinstance(completed_at, str) or not completed_at:
            raise ValueError("phase2j completed pass_a requires reviewer and timestamp")
    elif reviewer is not None or completed_at is not None:
        raise ValueError("phase2j pending/in-progress pass_a cannot be signed")
    if not isinstance(value["notes"], list) \
            or any(not isinstance(item, str) for item in value["notes"]):
        raise ValueError("phase2j pass_a notes must be strings")
    if isinstance(value["endpoint_count"], bool) or not isinstance(value["endpoint_count"], int) \
            or value["endpoint_count"] < 0:
        raise ValueError("phase2j pass_a endpoint count is invalid")


def _validate_pass_b(value: object, *, pass_a_status: str) -> None:
    if not isinstance(value, Mapping) or set(value) != _EXPECTED_PASS_B_KEYS:
        raise ValueError("phase2j pass_b review record is invalid")
    status = value["status"]
    if status not in PASS_B_STATUSES:
        raise ValueError("phase2j pass_b status is invalid")
    if pass_a_status != "COMPLETE":
        if status != "LOCKED_AWAITING_PASS_A":
            raise ValueError("phase2j pass_b cannot start before pass_a is complete")
    elif status == "LOCKED_AWAITING_PASS_A":
        raise ValueError("phase2j pass_b must unlock after pass_a is complete")
    reviewer = value["reviewer"]
    completed_at = value["completed_at"]
    if status == "COMPLETE":
        if not isinstance(reviewer, str) or not reviewer \
                or not isinstance(completed_at, str) or not completed_at:
            raise ValueError("phase2j completed pass_b requires reviewer and timestamp")
    elif reviewer is not None or completed_at is not None:
        raise ValueError("phase2j non-complete pass_b cannot be signed")
    if not isinstance(value["notes"], list) \
            or any(not isinstance(item, str) for item in value["notes"]):
        raise ValueError("phase2j pass_b notes must be strings")
    audit_checks = value["audit_checks"]
    if not isinstance(audit_checks, Mapping) or set(audit_checks) != set(AUDIT_CHECKS) \
            or any(not isinstance(item, bool) for item in audit_checks.values()):
        raise ValueError("phase2j pass_b audit checks are invalid")
    if status == "COMPLETE" and not all(audit_checks.values()):
        raise ValueError("phase2j completed pass_b requires every audit check")


def _validate_endpoint(
    endpoint: object,
    *,
    window_text: str,
    tokens: list[dict[str, Any]],
    window_id: str,
    index: int,
    pass_a_status: str,
    pass_b_status: str,
) -> None:
    if not isinstance(endpoint, Mapping) or set(endpoint) != _EXPECTED_ENDPOINT_KEYS:
        raise ValueError("phase2j endpoint entry is invalid")
    endpoint_id = endpoint["endpoint_id"]
    expected_prefix = f"p2j:{window_id}:ep:"
    if not isinstance(endpoint_id, str) or not endpoint_id.startswith(expected_prefix):
        raise ValueError("phase2j endpoint ID is not bound to its window")
    suffix = endpoint_id[len(expected_prefix):]
    if not re.fullmatch(r"[0-9]{4}", suffix) or int(suffix) != index:
        raise ValueError("phase2j endpoint IDs must be sequential per window")
    char_start, char_end = endpoint["char_start"], endpoint["char_end"]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (char_start, char_end)) \
            or not 0 <= char_start < char_end <= len(window_text):
        raise ValueError("phase2j endpoint character offsets are invalid")
    token_start, token_end = endpoint["token_start"], endpoint["token_end"]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (token_start, token_end)) \
            or not 0 <= token_start <= token_end < len(tokens):
        raise ValueError("phase2j endpoint token offsets are invalid")
    if tokens[token_start]["start"] != char_start or tokens[token_end]["end"] != char_end:
        raise ValueError("phase2j endpoint token boundaries must match exact character spans")
    bronze_text = endpoint["bronze_text"]
    if not isinstance(bronze_text, str) or window_text[char_start:char_end] != bronze_text:
        raise ValueError("phase2j endpoint text is not an exact Bronze slice")
    node_type = endpoint["node_type"]
    if node_type is not None and node_type not in NODE_TYPES:
        raise ValueError("phase2j endpoint node type is invalid")
    ambiguity_state = endpoint["ambiguity_state"]
    if ambiguity_state not in AMBIGUITY_STATES:
        raise ValueError("phase2j endpoint ambiguity state is invalid")
    disposition = endpoint["disposition"]
    if disposition not in ENDPOINT_DISPOSITIONS:
        raise ValueError("phase2j endpoint disposition is invalid")
    adjudication = endpoint["adjudication_requested"]
    if not isinstance(adjudication, bool):
        raise ValueError("phase2j endpoint adjudication flag must be boolean")
    if disposition == "KEEP" and (ambiguity_state != "NONE" or adjudication):
        raise ValueError("phase2j KEEP endpoint cannot carry ambiguity or adjudication")
    if disposition == "ADJUDICATION_REQUIRED" and not adjudication:
        raise ValueError("phase2j adjudication endpoint must request adjudication")
    if not isinstance(endpoint["notes"], str):
        raise ValueError("phase2j endpoint notes must be a string")
    pass_provenance = endpoint["pass_provenance"]
    if pass_provenance not in PASS_PROVENANCES:
        raise ValueError("phase2j endpoint pass provenance is invalid")
    if pass_provenance == "PASS_B":
        if pass_a_status != "COMPLETE" or pass_b_status not in {"IN_PROGRESS", "COMPLETE"}:
            raise ValueError("phase2j PASS_B endpoint requires an active pass_b")


def _validate_overlaps(endpoints: list[dict[str, Any]], window_id: str) -> None:
    spans = [(item["char_start"], item["char_end"], item["endpoint_id"]) for item in endpoints]
    for left_index in range(len(spans)):
        left_start, left_end, left_id = spans[left_index]
        for right_index in range(left_index + 1, len(spans)):
            right_start, right_end, right_id = spans[right_index]
            overlaps = left_start < right_end and right_start < left_end
            duplicate = (left_start, left_end) == (right_start, right_end)
            if not overlaps and not duplicate:
                continue
            left = endpoints[left_index]
            right = endpoints[right_index]
            if not (left["adjudication_requested"] and right["adjudication_requested"]):
                raise ValueError(
                    "phase2j duplicate/overlapping endpoint annotations are "
                    f"rejected in window {window_id}: {left_id} vs {right_id}",
                )


def _validate_record(
    record: object, *,
    record_index: int,
    seen_annotation_ids: set[str],
    seen_window_ids: set[str],
    seen_groups: set[str],
) -> None:
    if not isinstance(record, Mapping) or set(record) != _EXPECTED_RECORD_KEYS:
        raise ValueError("phase2j annotation record is invalid")
    if record["record_index"] != record_index:
        raise ValueError("phase2j record indices must be sequential")
    annotation_id = record["annotation_id"]
    window_id = record["window_id"]
    group = record["source_group_id"]
    upstream = record["upstream_source_id"]
    if not all(isinstance(value, str) and value for value in (
        annotation_id, window_id, group, upstream,
    )):
        raise ValueError("phase2j annotation record identity is invalid")
    if annotation_id in seen_annotation_ids or window_id in seen_window_ids \
            or group in seen_groups:
        raise ValueError("phase2j annotation records contain duplicate identity")
    seen_annotation_ids.add(annotation_id)
    seen_window_ids.add(window_id)
    seen_groups.add(group)
    if annotation_id != f"p2j:{window_id}":
        raise ValueError("phase2j annotation ID must bind the window ID")
    if group != f"video:{upstream}":
        raise ValueError("phase2j annotation source group must derive from the video ID")
    start, end = record["upstream_start"], record["upstream_end"]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)) \
            or start < 0 or end <= start:
        raise ValueError("phase2j annotation upstream offsets are invalid")
    partition = record["partition"]
    if partition not in PARTITION_SIZES:
        raise ValueError("phase2j annotation partition is invalid")
    bronze_text = record["bronze_text"]
    if not isinstance(bronze_text, str) or not bronze_text.strip() \
            or end - start != len(bronze_text):
        raise ValueError("phase2j annotation Bronze text/offsets are invalid")
    if record["bronze_text_sha256"] != hashlib.sha256(bronze_text.encode("utf-8")).hexdigest():
        raise ValueError("phase2j annotation Bronze text hash is invalid")
    if isinstance(record["bronze_char_length"], bool) \
            or record["bronze_char_length"] != len(bronze_text):
        raise ValueError("phase2j annotation Bronze character length is invalid")
    tokens = _validate_token_table(bronze_text, record["tokens"])
    status = record["window_status"]
    if status not in WINDOW_STATUSES:
        raise ValueError("phase2j window status is invalid")
    _validate_pass_a(record["pass_a"])
    _validate_pass_b(record["pass_b"], pass_a_status=record["pass_a"]["status"])
    endpoints = record["endpoints"]
    if not isinstance(endpoints, list):
        raise ValueError("phase2j endpoints must be a list")
    for index, endpoint in enumerate(endpoints, 1):
        _validate_endpoint(
            endpoint,
            window_text=bronze_text,
            tokens=tokens,
            window_id=window_id,
            index=index,
            pass_a_status=record["pass_a"]["status"],
            pass_b_status=record["pass_b"]["status"],
        )
    ordered = sorted(endpoints, key=lambda item: (
        item["char_start"], item["char_end"], item["endpoint_id"],
    ))
    if ordered != endpoints:
        raise ValueError("phase2j endpoints must be in deterministic sorted order")
    if record["pass_a"]["endpoint_count"] != len(endpoints):
        raise ValueError("phase2j pass_a endpoint count must match the endpoint list")
    _validate_overlaps(endpoints, window_id)
    ambiguity = record["ambiguity_controls"]
    exclusion = record["exclusion_controls"]
    if not isinstance(ambiguity, Mapping) or set(ambiguity) != {"flagged", "notes"} \
            or not isinstance(ambiguity["flagged"], bool) \
            or not isinstance(ambiguity["notes"], list) \
            or any(not isinstance(item, str) for item in ambiguity["notes"]):
        raise ValueError("phase2j ambiguity controls are invalid")
    if not isinstance(exclusion, Mapping) or set(exclusion) != {"flagged", "notes"} \
            or not isinstance(exclusion["flagged"], bool) \
            or not isinstance(exclusion["notes"], list) \
            or any(not isinstance(item, str) for item in exclusion["notes"]):
        raise ValueError("phase2j exclusion controls are invalid")
    if not isinstance(record["reviewer_notes"], list) \
            or any(not isinstance(item, str) for item in record["reviewer_notes"]):
        raise ValueError("phase2j reviewer notes must be strings")

    pass_a_status = record["pass_a"]["status"]
    pass_b_status = record["pass_b"]["status"]
    pass_a_complete = pass_a_status == "COMPLETE"
    pass_b_complete = pass_b_status == "COMPLETE"
    dispositions = {item["disposition"] for item in endpoints}
    any_adjudication = any(item["adjudication_requested"] for item in endpoints)
    if status == "UNREVIEWED":
        if record["pass_a"]["status"] != "PENDING" or endpoints \
                or ambiguity["flagged"] or exclusion["flagged"]:
            raise ValueError("phase2j UNREVIEWED window must remain blank")
    elif status == "IN_REVIEW":
        if pass_a_status not in {"IN_PROGRESS", "COMPLETE"}:
            raise ValueError(
                "phase2j IN_REVIEW window requires an active or complete pass_a",
            )
        if pass_b_status == "COMPLETE":
            raise ValueError("phase2j IN_REVIEW window cannot have a complete pass_b")
        if dispositions - {"KEEP"} or any_adjudication \
                or ambiguity["flagged"] or exclusion["flagged"]:
            raise ValueError(
                "phase2j IN_REVIEW window must be clean KEEP-only",
            )
    elif not pass_a_complete:
        raise ValueError("phase2j reviewed states require complete pass_a")
    if status == "REVIEWED":
        if not pass_b_complete or dispositions != {"KEEP"} or any_adjudication \
                or ambiguity["flagged"] or exclusion["flagged"]:
            raise ValueError("phase2j REVIEWED window requires clean two-pass completion")
    if status == "AMBIGUOUS":
        if not ambiguity["flagged"] or pass_b_complete:
            raise ValueError("phase2j AMBIGUOUS window requires flag and open pass_b")
        if not any(item["disposition"] == "AMBIGUOUS" for item in endpoints) \
                and not ambiguity["notes"]:
            raise ValueError("phase2j AMBIGUOUS window requires an ambiguity control")
    if status == "ADJUDICATION_REQUIRED":
        if pass_b_complete or not (any_adjudication or "ADJUDICATION_REQUIRED" in dispositions):
            raise ValueError(
                "phase2j ADJUDICATION_REQUIRED window requires open adjudication entries",
            )
    elif any_adjudication or "ADJUDICATION_REQUIRED" in dispositions:
        raise ValueError("phase2j adjudication entries require ADJUDICATION_REQUIRED window")
    if status == "EXCLUDED":
        if not exclusion["flagged"] or endpoints:
            raise ValueError("phase2j EXCLUDED window must be empty and flagged")
    for disposition in dispositions:
        if disposition == "AMBIGUOUS" and status != "AMBIGUOUS":
            raise ValueError("phase2j AMBIGUOUS endpoint requires AMBIGUOUS window")
        if disposition == "EXCLUDED" and status != "EXCLUDED":
            raise ValueError("phase2j EXCLUDED endpoint requires EXCLUDED window")
        if disposition == "ADJUDICATION_REQUIRED" and status != "ADJUDICATION_REQUIRED":
            raise ValueError(
                "phase2j ADJUDICATION_REQUIRED endpoint requires ADJUDICATION_REQUIRED window",
            )


def is_window_gold_eligible(record: Mapping[str, Any]) -> bool:
    """Two-pass gold eligibility for one annotation record."""
    if record["window_status"] != "REVIEWED":
        return False
    if record["pass_a"]["status"] != "COMPLETE" or record["pass_b"]["status"] != "COMPLETE":
        return False
    if record["ambiguity_controls"]["flagged"] or record["exclusion_controls"]["flagged"]:
        return False
    if not record["pass_b"]["audit_checks"] or not all(record["pass_b"]["audit_checks"].values()):
        return False
    if not record["endpoints"]:
        return False
    return all(
        item["disposition"] == "KEEP" and not item["adjudication_requested"]
        for item in record["endpoints"]
    )


def is_packet_gold_eligible(packet: Mapping[str, Any]) -> bool:
    return all(is_window_gold_eligible(record) for record in packet["records"])


def validate_annotation_packet(
    packet: Mapping[str, Any], *, manifest: Mapping[str, Any] | None = None,
) -> None:
    """Validate the packet envelope, records, rules, and cross-bindings."""
    if not isinstance(packet, Mapping) or set(packet) != _EXPECTED_PACKET_KEYS:
        raise ValueError("phase2j annotation packet envelope is invalid")
    if packet["schema_version"] != PACKET_SCHEMA_VERSION:
        raise ValueError("phase2j annotation packet version is unsupported")
    if packet["annotation_version"] != ANNOTATION_VERSION:
        raise ValueError("phase2j annotation version is unsupported")
    if packet["release_gate"] != RELEASE_GATE:
        raise ValueError("phase2j release gate must remain LOCKED")
    if not isinstance(packet["purpose"], str) or not packet["purpose"]:
        raise ValueError("phase2j packet purpose is invalid")
    inner = {key: item for key, item in packet.items() if key != "content_sha256"}
    if packet["content_sha256"] != canonical_sha256(inner):
        raise ValueError("phase2j annotation packet content hash is invalid")
    _validate_forbidden_content(packet)
    if not _SHA256.fullmatch(packet["selection_manifest_sha256"]):
        raise ValueError("phase2j selection manifest hash is invalid")
    if packet["selection_manifest_schema_version"] != SELECTION_SCHEMA_VERSION:
        raise ValueError("phase2j selection manifest schema version is invalid")
    if packet["candidate_generator_version"] != MENTION_CATALOG_VERSION:
        raise ValueError("phase2j candidate generator version is unsupported")
    rules = packet["rules"]
    if not isinstance(rules, Mapping) or set(rules) != _EXPECTED_RULES_KEYS:
        raise ValueError("phase2j packet rules are invalid")
    if tuple(rules["window_statuses"]) != WINDOW_STATUSES \
            or tuple(rules["endpoint_dispositions"]) != ENDPOINT_DISPOSITIONS:
        raise ValueError("phase2j packet status/disposition rules are invalid")
    if rules["pass_b_requires_pass_a"] is not True:
        raise ValueError("phase2j pass_b must require pass_a")
    for key in (
        "pass_a", "pass_b", "in_review_rule", "gold_eligibility_rule",
        "overlap_rule", "non_keep_rule", "reviewer_instructions",
    ):
        if not isinstance(rules[key], str) or not rules[key]:
            raise ValueError("phase2j packet rule text is invalid")
    records = packet["records"]
    if not isinstance(records, list) or len(records) != 30:
        raise ValueError("phase2j packet must contain exactly 30 annotation records")
    seen_annotation_ids: set[str] = set()
    seen_window_ids: set[str] = set()
    seen_groups: set[str] = set()
    for index, record in enumerate(records, 1):
        _validate_record(
            record, record_index=index, seen_annotation_ids=seen_annotation_ids,
            seen_window_ids=seen_window_ids, seen_groups=seen_groups,
        )
    catalog = packet["candidate_catalog"]
    if not isinstance(catalog, Mapping) or set(catalog) != {"count", "per_window"}:
        raise ValueError("phase2j candidate catalog binding is invalid")
    if isinstance(catalog["count"], bool) or not isinstance(catalog["count"], int) \
            or catalog["count"] <= 0:
        raise ValueError("phase2j candidate catalog count is invalid")
    per_window = catalog["per_window"]
    if not isinstance(per_window, Mapping) or set(per_window) != set(seen_window_ids):
        raise ValueError("phase2j candidate catalog must cover every selected window")
    for window_id, binding in per_window.items():
        if not isinstance(binding, Mapping) or set(binding) != {"count", "catalog_sha256"}:
            raise ValueError("phase2j per-window catalog binding is invalid")
        if isinstance(binding["count"], bool) or not isinstance(binding["count"], int) \
                or binding["count"] <= 0 or not _SHA256.fullmatch(binding["catalog_sha256"]):
            raise ValueError("phase2j per-window catalog binding values are invalid")
    if sum(binding["count"] for binding in per_window.values()) != catalog["count"]:
        raise ValueError("phase2j candidate catalog totals are inconsistent")
    if manifest is not None:
        validate_selection_manifest(manifest)
        if packet["selection_manifest_sha256"] != manifest["content_sha256"]:
            raise ValueError("phase2j packet is not bound to the selection manifest")
        manifest_window_ids = [item["window_id"] for item in manifest["selected"]]
        packet_window_ids = [record["window_id"] for record in records]
        if packet_window_ids != manifest_window_ids:
            raise ValueError("phase2j packet record order must match the selection manifest")
        for packet_record, manifest_record in zip(records, manifest["selected"]):
            for field in (
                "source_group_id", "window_id", "upstream_source_id",
                "upstream_start", "upstream_end", "partition",
            ):
                if packet_record[field] != manifest_record[field]:
                    raise ValueError(
                        "phase2j annotation record contradicts the selection "
                        f"manifest for {packet_record['window_id']}",
                    )
            if packet_record["bronze_text"] != manifest_record["source_text"] \
                    or packet_record["bronze_text_sha256"] != manifest_record["source_text_sha256"] \
                    or packet_record["bronze_char_length"] != manifest_record["source_text_char_length"]:
                raise ValueError(
                    "phase2j annotation Bronze text contradicts the selection manifest",
                )
        manifest_by_window = {item["window_id"]: item for item in manifest["selected"]}
        for window_id, binding in per_window.items():
            expected = manifest_by_window.get(window_id)
            if expected is None or binding["count"] != expected["candidate_count"] \
                    or binding["catalog_sha256"] != expected["candidate_catalog_sha256"]:
                raise ValueError("phase2j packet catalog binding contradicts the manifest")


def load_annotation_packet(
    path: Path, *, manifest: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Strict canonical load with duplicate-key rejection and validation."""
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("phase2j annotation packet JSON contains duplicate keys")
            value[key] = item
        return value

    try:
        body = json.loads(
            Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
        )
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("phase2j annotation packet JSON is unavailable or malformed") from exc
    validate_annotation_packet(body, manifest=manifest)
    return body


__all__ = [
    "AMBIGUITY_STATES", "ANNOTATION_VERSION", "AUDIT_CHECKS",
    "ENDPOINT_DISPOSITIONS", "FORBIDDEN_KEYS", "NODE_TYPES",
    "PACKET_SCHEMA_VERSION", "PASS_A_STATUSES", "PASS_B_STATUSES",
    "PASS_PROVENANCES", "RELEASE_GATE", "WINDOW_STATUSES",
    "build_annotation_packet", "is_packet_gold_eligible",
    "is_window_gold_eligible", "load_annotation_packet", "token_table",
    "validate_annotation_packet",
]
