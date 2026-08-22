"""Phase 2J post-adjudication canonical import gate.

After the human completes the explicit Pass B audit attestation and exports a
``phase2j-adjudication-export-v2`` REVIEW MATERIAL file, this module turns the
locked blank annotation packet into a separate reviewed canonical packet.  It
never trusts the browser output alone: every input is loaded with
duplicate-key rejection, the human Pass A session is re-validated against the
locked packet and its file SHA-256 is cross-bound to the adjudication packet,
the adjudication packet is re-validated, and the export is checked field for
field against the adjudication packet (component identity/order, decision and
``resolved_by`` semantics, derived resolved endpoints, Bronze slices, audit
attestation, and identity/time fields).

The reviewed packet is built from the locked blank packet, never from export
text: Bronze text, token tables, source identity, partition, candidate-catalog
binding, rules, and ``release_gate=LOCKED`` are preserved exactly.  Sol stays
NON_GOLD; a Sol endpoint enters the reviewed packet only when the export
explicitly chose ``KEEP_SOL_SET`` or ``CUSTOM``, and provenance remains visible
in endpoint notes and pass provenance.  No scores, predictions, candidates,
model data, syntax data, or floats enter reviewed artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

from pipeline.phase2j_adjudication import (
    ADJUDICATION_VERSION,
    ENDPOINT_TYPES,
    OUTPUT_FORBIDDEN_KEYS,
    load_adjudication_packet,
    validate_adjudication_packet,
    validate_human_session,
)
from pipeline.phase2j_annotation_packet import (
    AUDIT_CHECKS,
    PACKET_SCHEMA_VERSION,
    is_window_gold_eligible,
    load_annotation_packet,
    validate_annotation_packet,
)
from pipeline.phase2j_source_selection import (
    canonical_sha256,
    file_sha256,
    load_selection_manifest,
)


ADJUDICATION_EXPORT_SCHEMA_VERSION = "phase2j-adjudication-export-v2"
REVIEW_MATERIAL_STATUS = "REVIEW_MATERIAL"
REVIEWED_PACKET_FILENAME = "reviewed-endpoint-annotation-packet-v1.json"

EXPORT_OUTCOMES = ("CLEAN", "AMBIGUOUS", "EXCLUDED")
DECISION_KINDS = ("KEEP_HUMAN_SET", "KEEP_SOL_SET", "DROP", "CUSTOM")
RESOLVED_BY_VALUES = (
    "PRE_RESOLVED", "HUMAN_SET", "SOL_SET", "DROP", "CUSTOM",
    "WINDOW_AMBIGUOUS", "WINDOW_EXCLUDED",
)
PROVENANCE_SOURCES = ("HUMAN", "SOL", "SHARED", "CUSTOM")

# Keys that may appear nowhere in an adjudication export.  The export
# legitimately carries reviewer identity and the export timestamp, so those
# two sanctioned fields are removed from the recursive forbidden set.
EXPORT_FORBIDDEN_KEYS = frozenset(
    key for key in OUTPUT_FORBIDDEN_KEYS
    if key not in {"reviewer_name", "exported_at"}
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_EXPECTED_EXPORT_KEYS = frozenset({
    "schema_version", "adjudication_version", "packet_schema_version",
    "adjudication_packet_sha256", "packet_sha256", "human_session_sha256",
    "sol_review_sha256", "status_label", "reviewer_name", "exported_at",
    "audit_checks", "records",
})
_EXPECTED_EXPORT_RECORD_KEYS = frozenset({
    "record_index", "window_id", "outcome", "note", "components",
    "resolved_endpoints",
})
_EXPECTED_EXPORT_COMPONENT_KEYS = frozenset({
    "component_id", "classification", "decision", "resolved_by",
})
_EXPECTED_DECISION_KEYS = frozenset({"kind"})
_EXPECTED_CUSTOM_DECISION_KEYS = frozenset({
    "kind", "token_start", "token_end", "node_type",
})
_EXPECTED_RESOLVED_ENDPOINT_KEYS = frozenset({
    "endpoint_id", "component_id", "exact_bronze_text", "char_start",
    "char_end", "token_start", "token_end", "node_type", "provenance_source",
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
    *,
    path: tuple[str, ...] = (),
) -> None:
    """Recursively reject forbidden scorer/model/candidate/PII keys and floats."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("phase2j adjudication export keys must be strings")
            if key.casefold() in EXPORT_FORBIDDEN_KEYS:
                raise ValueError(
                    "phase2j adjudication export contains forbidden key "
                    + repr(key) + " at " + ".".join(path + (key,)),
                )
            _validate_forbidden_content(item, path=path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_forbidden_content(item, path=path + (f"[{index}]",))
    elif isinstance(value, float):
        raise ValueError(
            "phase2j adjudication export contains a floating-point value",
        )


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"phase2j {label} must be a string")
    return value


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"phase2j {label} must be an integer")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"phase2j {label} must be a boolean")
    return value


def _token_interval_overlaps(
    left: tuple[int, int], right: tuple[int, int],
) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _require_sha256(value: object, label: str) -> str:
    text = _require_string(value, label)
    if not _SHA256.fullmatch(text):
        raise ValueError(f"phase2j {label} must be a 64-character hex string")
    return text


def _validate_decision(
    decision: object,
    *,
    component_id: str,
    classification: str,
    resolved_by: str,
    outcome: str,
    window_id: str,
    adjudication_record: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate one export component decision; None means explicit non-decision."""
    if decision is None:
        if outcome == "EXCLUDED":
            if resolved_by != "WINDOW_EXCLUDED":
                raise ValueError(
                    f"phase2j export component {component_id} must be WINDOW_EXCLUDED",
                )
            return None
        if outcome == "AMBIGUOUS":
            if resolved_by != "WINDOW_AMBIGUOUS":
                raise ValueError(
                    f"phase2j export component {component_id} must be WINDOW_AMBIGUOUS",
                )
            return None
        raise ValueError(
            f"phase2j export component {component_id} is unresolved in a CLEAN window",
        )
    if not isinstance(decision, Mapping):
        raise ValueError(f"phase2j export component {component_id} decision is invalid")
    kind = decision.get("kind")
    if kind not in DECISION_KINDS:
        raise ValueError(f"phase2j export component {component_id} decision kind is invalid")
    expected_keys = (
        _EXPECTED_CUSTOM_DECISION_KEYS
        if kind == "CUSTOM" else _EXPECTED_DECISION_KEYS
    )
    if set(decision) != expected_keys:
        raise ValueError(f"phase2j export component {component_id} decision keys are invalid")
    if kind == "CUSTOM":
        tokens = adjudication_record["tokens"]
        token_start = _require_int(
            decision["token_start"], f"component {component_id} token_start",
        )
        token_end = _require_int(
            decision["token_end"], f"component {component_id} token_end",
        )
        node_type = decision["node_type"]
        if node_type not in ENDPOINT_TYPES:
            raise ValueError(f"phase2j export component {component_id} custom node_type is invalid")
        if not 0 <= token_start <= token_end < len(tokens):
            raise ValueError(
                f"phase2j export component {component_id} custom span is out of bounds",
            )
    allowed = _allowed_decisions(classification)
    if kind not in allowed:
        raise ValueError(
            f"phase2j export component {component_id} decision {kind} is not allowed "
            f"for {classification}",
        )
    expected_resolved_by = _resolved_by_for(classification, kind)
    if resolved_by != expected_resolved_by:
        raise ValueError(
            f"phase2j export component {component_id} resolved_by must be "
            f"{expected_resolved_by} for {kind}",
        )
    return dict(decision)


def _allowed_decisions(classification: str) -> frozenset[str]:
    if classification == "EXACT_AGREEMENT":
        return frozenset({"KEEP_HUMAN_SET", "DROP", "CUSTOM"})
    if classification in {"TYPE_DISAGREEMENT", "BOUNDARY_DISAGREEMENT"}:
        return frozenset({"KEEP_HUMAN_SET", "KEEP_SOL_SET", "DROP", "CUSTOM"})
    if classification == "SOL_ONLY":
        return frozenset({"KEEP_SOL_SET", "DROP", "CUSTOM"})
    if classification == "HUMAN_ONLY":
        return frozenset({"KEEP_HUMAN_SET", "DROP", "CUSTOM"})
    raise ValueError(f"phase2j export component classification {classification} is invalid")


def _resolved_by_for(classification: str, kind: str) -> str:
    if kind == "KEEP_HUMAN_SET":
        return "PRE_RESOLVED" if classification == "EXACT_AGREEMENT" else "HUMAN_SET"
    if kind == "KEEP_SOL_SET":
        return "SOL_SET"
    return kind  # DROP -> DROP, CUSTOM -> CUSTOM


def derive_resolved_endpoints(
    adjudication_record: Mapping[str, Any],
    export_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Recompute the export's derived resolved endpoint set for one window.

    Mirrors the browser derivation exactly: EXCLUDED clears endpoints,
    AMBIGUOUS keeps only decided components, CLEAN requires every component,
    provenance is HUMAN/SHARED/SOL/CUSTOM, candidates are sorted by Bronze
    position then component id, overlapping spans are rejected, and endpoint
    ids are deterministic and sequential.
    """
    outcome = export_record["outcome"]
    if outcome == "EXCLUDED":
        return []
    window_id = adjudication_record["window_id"]
    bronze_text = adjudication_record["bronze_text"]
    tokens = adjudication_record["tokens"]
    human_by_id = {
        endpoint["endpoint_id"]: endpoint
        for endpoint in adjudication_record["human_endpoints"]
    }
    sol_by_id = {
        endpoint["endpoint_id"]: endpoint
        for endpoint in adjudication_record["sol_endpoints"]
    }
    components_by_id = {
        component["component_id"]: component
        for component in adjudication_record["components"]
    }
    candidates: list[dict[str, Any]] = []
    for entry in export_record["components"]:
        component_id = entry["component_id"]
        component = components_by_id[component_id]
        classification = entry["classification"]
        decision = entry["decision"]
        if outcome == "AMBIGUOUS" and decision is None:
            continue
        if decision is None:
            raise ValueError(f"phase2j export component {component_id} is unresolved")
        kind = decision["kind"]
        if kind == "DROP":
            continue
        if kind == "KEEP_HUMAN_SET":
            provenance = "SHARED" if classification == "EXACT_AGREEMENT" else "HUMAN"
            for endpoint_id in component["human_endpoint_ids"]:
                endpoint = human_by_id[endpoint_id]
                candidates.append({
                    "endpoint_id": endpoint["endpoint_id"],
                    "component_id": component_id,
                    "exact_bronze_text": endpoint["exact_bronze_text"],
                    "char_start": endpoint["char_start"],
                    "char_end": endpoint["char_end"],
                    "token_start": endpoint["token_start"],
                    "token_end": endpoint["token_end"],
                    "node_type": endpoint["node_type"],
                    "provenance_source": provenance,
                })
        elif kind == "KEEP_SOL_SET":
            for endpoint_id in component["sol_endpoint_ids"]:
                endpoint = sol_by_id[endpoint_id]
                if endpoint["node_type"] is None:
                    raise ValueError(
                        f"phase2j export component {component_id} keeps a Sol endpoint "
                        "with no type; choose a type first",
                    )
                candidates.append({
                    "endpoint_id": endpoint["endpoint_id"],
                    "component_id": component_id,
                    "exact_bronze_text": endpoint["exact_bronze_text"],
                    "char_start": endpoint["char_start"],
                    "char_end": endpoint["char_end"],
                    "token_start": endpoint["token_start"],
                    "token_end": endpoint["token_end"],
                    "node_type": endpoint["node_type"],
                    "provenance_source": "SOL",
                })
        elif kind == "CUSTOM":
            token_start = decision["token_start"]
            token_end = decision["token_end"]
            char_start = tokens[token_start]["start"]
            char_end = tokens[token_end]["end"]
            candidates.append({
                "endpoint_id": f"p2j:adjudicate:{window_id}:ep:custom",
                "component_id": component_id,
                "exact_bronze_text": bronze_text[char_start:char_end],
                "char_start": char_start,
                "char_end": char_end,
                "token_start": token_start,
                "token_end": token_end,
                "node_type": decision["node_type"],
                "provenance_source": "CUSTOM",
            })
    candidates.sort(key=lambda item: (
        item["char_start"], item["char_end"], item["component_id"],
    ))
    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            left = candidates[left_index]
            right = candidates[right_index]
            if _token_interval_overlaps(
                (left["char_start"], left["char_end"]),
                (right["char_start"], right["char_end"]),
            ):
                raise ValueError(
                    "phase2j export resolved endpoints overlap in "
                    f"{window_id}: {left['component_id']} vs {right['component_id']}",
                )
    return [
        {
            **candidate,
            "endpoint_id": (
                f"p2j:adjudicate:{window_id}:ep:{str(index).zfill(4)}"
            ),
        }
        for index, candidate in enumerate(candidates, 1)
    ]


def _validate_resolved_endpoint(
    endpoint: Mapping[str, Any],
    *,
    index: int,
    adjudication_record: Mapping[str, Any],
) -> None:
    if not isinstance(endpoint, Mapping) \
            or set(endpoint) != _EXPECTED_RESOLVED_ENDPOINT_KEYS:
        raise ValueError(
            f"phase2j export resolved endpoint {index} in "
            f"{adjudication_record['window_id']} is invalid",
        )
    window_id = adjudication_record["window_id"]
    expected_id = f"p2j:adjudicate:{window_id}:ep:{str(index).zfill(4)}"
    if endpoint["endpoint_id"] != expected_id:
        raise ValueError(
            f"phase2j export resolved endpoint ids must be sequential in {window_id}",
        )
    component_id = _require_string(
        endpoint["component_id"], f"resolved endpoint {index} component_id",
    )
    if not any(
        component["component_id"] == component_id
        for component in adjudication_record["components"]
    ):
        raise ValueError(
            f"phase2j export resolved endpoint {index} component_id does not "
            f"belong to {window_id}",
        )
    if endpoint["provenance_source"] not in PROVENANCE_SOURCES:
        raise ValueError(
            f"phase2j export resolved endpoint {index} provenance_source is invalid",
        )
    node_type = endpoint["node_type"]
    if node_type not in ENDPOINT_TYPES:
        raise ValueError(
            f"phase2j export resolved endpoint {index} node_type is invalid",
        )
    bronze_text = adjudication_record["bronze_text"]
    tokens = adjudication_record["tokens"]
    char_start = _require_int(endpoint["char_start"], f"resolved endpoint {index} char_start")
    char_end = _require_int(endpoint["char_end"], f"resolved endpoint {index} char_end")
    token_start = _require_int(endpoint["token_start"], f"resolved endpoint {index} token_start")
    token_end = _require_int(endpoint["token_end"], f"resolved endpoint {index} token_end")
    if not 0 <= char_start < char_end <= len(bronze_text) \
            or not 0 <= token_start <= token_end < len(tokens) \
            or tokens[token_start]["start"] != char_start \
            or tokens[token_end]["end"] != char_end:
        raise ValueError(
            f"phase2j export resolved endpoint {index} bounds are inconsistent in {window_id}",
        )
    exact = _require_string(
        endpoint["exact_bronze_text"], f"resolved endpoint {index} exact text",
    )
    if bronze_text[char_start:char_end] != exact:
        raise ValueError(
            f"phase2j export resolved endpoint {index} text is not an exact Bronze "
            f"slice in {window_id}",
        )


def validate_adjudication_export(
    export: Mapping[str, Any],
    adjudication_packet: Mapping[str, Any],
) -> None:
    """Strict validation of an adjudication export v2 against the packet."""
    if not isinstance(export, Mapping) or set(export) != _EXPECTED_EXPORT_KEYS:
        raise ValueError("phase2j adjudication export envelope is invalid")
    if export["schema_version"] != ADJUDICATION_EXPORT_SCHEMA_VERSION:
        raise ValueError("phase2j adjudication export version is unsupported")
    if export["adjudication_version"] != ADJUDICATION_VERSION:
        raise ValueError("phase2j adjudication export version is unsupported")
    if export["packet_schema_version"] != PACKET_SCHEMA_VERSION:
        raise ValueError("phase2j adjudication export packet schema is unsupported")
    if _require_sha256(
        export["adjudication_packet_sha256"], "export adjudication_packet_sha256",
    ) != adjudication_packet["content_sha256"]:
        raise ValueError(
            "phase2j adjudication export is not bound to the adjudication packet",
        )
    if _require_sha256(
        export["packet_sha256"], "export packet_sha256",
    ) != adjudication_packet["packet_sha256"]:
        raise ValueError("phase2j adjudication export packet hash is inconsistent")
    if _require_sha256(
        export["human_session_sha256"], "export human_session_sha256",
    ) != adjudication_packet["human_session_sha256"]:
        raise ValueError(
            "phase2j adjudication export human session hash is inconsistent",
        )
    if _require_sha256(
        export["sol_review_sha256"], "export sol_review_sha256",
    ) != adjudication_packet["sol_review_sha256"]:
        raise ValueError("phase2j adjudication export Sol review hash is inconsistent")
    if export["status_label"] != REVIEW_MATERIAL_STATUS:
        raise ValueError("phase2j adjudication export status_label must be REVIEW_MATERIAL")
    reviewer = _require_string(export["reviewer_name"], "export reviewer_name")
    if not reviewer.strip():
        raise ValueError("phase2j adjudication export reviewer_name is required")
    exported_at = _require_string(export["exported_at"], "export exported_at")
    if not exported_at.strip():
        raise ValueError("phase2j adjudication export exported_at is required")
    audit_checks = export["audit_checks"]
    if not isinstance(audit_checks, Mapping) or set(audit_checks) != set(AUDIT_CHECKS):
        raise ValueError(
            "phase2j adjudication export audit_checks must contain exactly the "
            "five Pass B checks",
        )
    if any(not isinstance(item, bool) for item in audit_checks.values()):
        raise ValueError(
            "phase2j adjudication export audit_checks values must be booleans",
        )
    if not all(audit_checks.values()):
        raise ValueError(
            "phase2j adjudication export audit_checks requires every check to be true",
        )
    _validate_forbidden_content(export)
    records = export["records"]
    if not isinstance(records, list) or len(records) != len(adjudication_packet["records"]):
        raise ValueError(
            "phase2j adjudication export must contain exactly "
            f"{len(adjudication_packet['records'])} windows",
        )
    for index, (export_record, adjudication_record) in enumerate(
        zip(records, adjudication_packet["records"]), 1,
    ):
        _validate_export_record(
            export_record, adjudication_record, index=index,
        )


def _validate_export_record(
    export_record: Mapping[str, Any],
    adjudication_record: Mapping[str, Any],
    *,
    index: int,
) -> None:
    if not isinstance(export_record, Mapping) \
            or set(export_record) != _EXPECTED_EXPORT_RECORD_KEYS:
        raise ValueError(f"phase2j adjudication export record {index} is invalid")
    if export_record["record_index"] != index:
        raise ValueError(f"phase2j adjudication export record_index must be {index}")
    if export_record["window_id"] != adjudication_record["window_id"]:
        raise ValueError(
            f"phase2j adjudication export record {index} window_id does not match "
            "the adjudication packet",
        )
    outcome = export_record["outcome"]
    if outcome not in EXPORT_OUTCOMES:
        raise ValueError(f"phase2j adjudication export record {index} outcome is invalid")
    note = _require_string(export_record["note"], f"export record {index} note")
    if outcome != "CLEAN" and not note.strip():
        raise ValueError(
            f"phase2j adjudication export record {index} requires a note for {outcome}",
        )
    window_id = adjudication_record["window_id"]
    components = export_record["components"]
    adjudication_components = adjudication_record["components"]
    if not isinstance(components, list) or len(components) != len(adjudication_components):
        raise ValueError(
            f"phase2j adjudication export record {index} components must cover "
            f"every adjudication component in {window_id}",
        )
    normalized_components: list[dict[str, Any]] = []
    for component_index, (entry, adjudication_component) in enumerate(
        zip(components, adjudication_components), 1,
    ):
        if not isinstance(entry, Mapping) \
                or set(entry) != _EXPECTED_EXPORT_COMPONENT_KEYS:
            raise ValueError(
                f"phase2j export component {component_index} in record {index} is invalid",
            )
        if entry["component_id"] != adjudication_component["component_id"] \
                or entry["classification"] != adjudication_component["classification"]:
            raise ValueError(
                f"phase2j export component {component_index} in record {index} "
                "does not match the adjudication packet",
            )
        resolved_by = entry["resolved_by"]
        if resolved_by not in RESOLVED_BY_VALUES:
            raise ValueError(
                f"phase2j export component {entry['component_id']} resolved_by is invalid",
            )
        decision = _validate_decision(
            entry["decision"],
            component_id=entry["component_id"],
            classification=entry["classification"],
            resolved_by=resolved_by,
            outcome=outcome,
            window_id=window_id,
            adjudication_record=adjudication_record,
        )
        normalized_components.append({
            **entry,
            "decision": decision,
        })
    normalized_record = {
        **export_record,
        "components": normalized_components,
    }
    derived = derive_resolved_endpoints(adjudication_record, normalized_record)
    if outcome == "CLEAN" and not derived:
        raise ValueError(
            f"phase2j adjudication export record {index} CLEAN window has zero endpoints",
        )
    raw_endpoints = export_record["resolved_endpoints"]
    if not isinstance(raw_endpoints, list) or len(raw_endpoints) != len(derived):
        raise ValueError(
            f"phase2j adjudication export record {index} resolved_endpoints do not "
            f"match the derived endpoint set in {window_id}",
        )
    for endpoint_index, endpoint in enumerate(raw_endpoints, 1):
        _validate_resolved_endpoint(
            endpoint, index=endpoint_index, adjudication_record=adjudication_record,
        )
    if list(raw_endpoints) != derived:
        raise ValueError(
            f"phase2j adjudication export record {index} resolved_endpoints do not "
            f"match the derived endpoint set in {window_id}",
        )


def _reviewed_endpoints(
    resolved: list[Mapping[str, Any]],
    *,
    window_id: str,
) -> list[dict[str, Any]]:
    """Map validated export endpoints to canonical KEEP endpoints."""
    ordered = sorted(
        resolved,
        key=lambda item: (
            item["char_start"], item["char_end"], item["component_id"],
            item["provenance_source"],
        ),
    )
    endpoints: list[dict[str, Any]] = []
    for index, endpoint in enumerate(ordered, 1):
        provenance_source = endpoint["provenance_source"]
        node_type = endpoint["node_type"]
        if node_type == "UNDETERMINED":
            node_type = None
        endpoints.append({
            "endpoint_id": f"p2j:{window_id}:ep:{str(index).zfill(4)}",
            "bronze_text": endpoint["exact_bronze_text"],
            "char_start": endpoint["char_start"],
            "char_end": endpoint["char_end"],
            "token_start": endpoint["token_start"],
            "token_end": endpoint["token_end"],
            "node_type": node_type,
            "ambiguity_state": "NONE",
            "disposition": "KEEP",
            "adjudication_requested": False,
            "notes": (
                f"adjudicated {endpoint['component_id']}; "
                f"source {provenance_source}"
            ),
            "pass_provenance": (
                "PASS_A" if provenance_source in {"HUMAN", "SHARED"} else "PASS_B"
            ),
        })
    return endpoints


def _reviewed_record(
    blank_record: Mapping[str, Any],
    adjudication_record: Mapping[str, Any],
    human_record: Mapping[str, Any],
    export_record: Mapping[str, Any],
    *,
    export_reviewer: str,
    export_exported_at: str,
    export_sha256: str,
    export_audit_checks: Mapping[str, bool],
) -> dict[str, Any]:
    outcome = export_record["outcome"]
    resolved = derive_resolved_endpoints(adjudication_record, export_record)
    endpoints = _reviewed_endpoints(resolved, window_id=blank_record["window_id"])
    if outcome == "CLEAN":
        window_status = "REVIEWED"
    elif outcome == "AMBIGUOUS":
        window_status = "AMBIGUOUS"
    else:
        window_status = "EXCLUDED"
    human_count = len(adjudication_record["human_endpoints"])
    pass_a = {
        "status": "COMPLETE",
        "reviewer": human_record["reviewer_name"],
        "completed_at": human_record["completed_at"],
        "notes": [
            "Reviewed reconciliation: Pass A recorded "
            f"{human_count} endpoint(s); reviewed canonical list contains "
            f"{len(endpoints)} endpoint(s).",
        ],
        "endpoint_count": len(endpoints),
    }
    audit_checks = dict(export_audit_checks)
    window_note = export_record["note"]
    if outcome == "AMBIGUOUS":
        pass_b = {
            "status": "IN_PROGRESS",
            "reviewer": None,
            "completed_at": None,
            "notes": [
                f"Adjudication export {export_sha256[:12]} by {export_reviewer} "
                f"at {export_exported_at}; Pass B audit attestation true.",
                f"Window remains ambiguous: {window_note}",
            ],
            "audit_checks": audit_checks,
        }
        ambiguity_controls = {"flagged": True, "notes": [window_note]}
        exclusion_controls = {"flagged": False, "notes": []}
    elif outcome == "EXCLUDED":
        pass_b = {
            "status": "COMPLETE",
            "reviewer": export_reviewer,
            "completed_at": export_exported_at,
            "notes": [
                f"Adjudication export {export_sha256[:12]} by {export_reviewer} "
                f"at {export_exported_at}; Pass B audit attestation true.",
            ],
            "audit_checks": audit_checks,
        }
        ambiguity_controls = {"flagged": False, "notes": []}
        exclusion_controls = {"flagged": True, "notes": [window_note]}
    else:
        pass_b = {
            "status": "COMPLETE",
            "reviewer": export_reviewer,
            "completed_at": export_exported_at,
            "notes": [
                f"Adjudication export {export_sha256[:12]} by {export_reviewer} "
                f"at {export_exported_at}; Pass B source-grounded audit "
                "attestation complete.",
            ],
            "audit_checks": audit_checks,
        }
        ambiguity_controls = {"flagged": False, "notes": []}
        exclusion_controls = {"flagged": False, "notes": []}
    return {
        **blank_record,
        "window_status": window_status,
        "endpoints": endpoints,
        "pass_a": pass_a,
        "pass_b": pass_b,
        "ambiguity_controls": ambiguity_controls,
        "exclusion_controls": exclusion_controls,
        "reviewer_notes": [],
    }


def build_reviewed_packet(
    *,
    blank_packet_path: Path,
    manifest_path: Path,
    human_session_path: Path,
    adjudication_packet_path: Path,
    export_path: Path,
) -> dict[str, Any]:
    """Validate every input and build the reviewed canonical annotation packet."""
    manifest = load_selection_manifest(manifest_path)
    blank = load_annotation_packet(blank_packet_path, manifest=manifest)
    adjudication = load_adjudication_packet(adjudication_packet_path)
    validate_adjudication_packet(adjudication)
    if adjudication["packet_sha256"] != blank["content_sha256"]:
        raise ValueError(
            "phase2j adjudication packet is not bound to the locked blank packet",
        )
    human = _load_json_strict(
        human_session_path, label="phase2j human Pass A session",
    )
    validate_human_session(human, blank)
    if file_sha256(human_session_path) != adjudication["human_session_sha256"]:
        raise ValueError(
            "phase2j human session file hash does not match the adjudication packet",
        )
    for index, (human_record, adjudication_record) in enumerate(
        zip(human["records"], adjudication["records"]), 1,
    ):
        if human_record["pass_a_complete"] is not True:
            raise ValueError(
                f"phase2j human record {index} did not complete Pass A",
            )
        if not human_record["reviewer_name"].strip() \
                or not human_record["completed_at"]:
            raise ValueError(
                f"phase2j human record {index} Pass A completion lacks identity",
            )
        if human_record["outcome"] != adjudication_record["human_outcome"]:
            raise ValueError(
                f"phase2j human record {index} outcome contradicts the "
                "adjudication packet",
            )
    export = _load_json_strict(
        export_path, label="phase2j adjudication export",
    )
    validate_adjudication_export(export, adjudication)
    export_sha256 = file_sha256(export_path)
    records = [
        _reviewed_record(
            blank_record,
            adjudication_record,
            human_record,
            export_record,
            export_reviewer=export["reviewer_name"],
            export_exported_at=export["exported_at"],
            export_sha256=export_sha256,
            export_audit_checks=export["audit_checks"],
        )
        for blank_record, adjudication_record, human_record, export_record in zip(
            blank["records"], adjudication["records"],
            human["records"], export["records"],
        )
    ]
    inner = {
        key: value for key, value in blank.items() if key != "content_sha256"
    }
    inner["purpose"] = (
        "Reviewed scorer-blind two-pass Phase 2J endpoint annotation packet: "
        "human Pass A endpoint discovery plus the adjudicated Pass B boundary/"
        "omission/role/duplicate/ambiguity audit. Sol was a second opinion and "
        "is never gold; no model scores, predictions, syntax data, or candidate "
        "rows are stored."
    )
    inner["records"] = records
    reviewed = {"content_sha256": canonical_sha256(inner), **inner}
    validate_annotation_packet(reviewed, manifest=manifest)
    return reviewed


def serialize_reviewed_packet(packet: Mapping[str, Any]) -> str:
    """Deterministic canonical pretty JSON with a trailing newline."""
    return json.dumps(
        packet, sort_keys=True, indent=2, ensure_ascii=False,
    ) + "\n"


def load_reviewed_packet(
    path: Path, *, manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Strict canonical load of an existing reviewed packet."""
    return load_annotation_packet(path, manifest=manifest)


def summarize_reviewed_packet(
    packet: Mapping[str, Any],
    *,
    blank_packet_sha256: str,
    manifest_sha256: str,
    human_session_sha256: str,
    adjudication_packet_sha256: str,
    sol_review_sha256: str,
    export_sha256: str,
) -> dict[str, Any]:
    """Deterministic CLI summary of the reviewed packet and its inputs."""
    window_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    pass_a_counts: dict[str, int] = {}
    pass_b_counts: dict[str, int] = {}
    endpoint_count = 0
    reviewed_windows = 0
    gold_eligible_windows = 0
    gold_eligible_endpoints = 0
    for record in packet["records"]:
        status = record["window_status"]
        window_counts[status] = window_counts.get(status, 0) + 1
        outcome = {
            "REVIEWED": "CLEAN",
            "AMBIGUOUS": "AMBIGUOUS",
            "EXCLUDED": "EXCLUDED",
        }.get(status)
        if outcome:
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        pass_b_status = record["pass_b"]["status"]
        pass_b_counts[pass_b_status] = pass_b_counts.get(pass_b_status, 0) + 1
        pass_a_status = record["pass_a"]["status"]
        pass_a_counts[pass_a_status] = pass_a_counts.get(pass_a_status, 0) + 1
        if status == "REVIEWED":
            reviewed_windows += 1
            if is_window_gold_eligible(record):
                gold_eligible_windows += 1
                gold_eligible_endpoints += len(record["endpoints"])
        endpoint_count += len(record["endpoints"])
    return {
        "reviewed_packet_sha256": packet["content_sha256"],
        "blank_packet_sha256": blank_packet_sha256,
        "manifest_sha256": manifest_sha256,
        "human_session_sha256": human_session_sha256,
        "adjudication_packet_sha256": adjudication_packet_sha256,
        "sol_review_sha256": sol_review_sha256,
        "export_sha256": export_sha256,
        "release_gate": packet["release_gate"],
        "window_statuses": dict(sorted(window_counts.items())),
        "outcomes": {
            key: outcome_counts.get(key, 0)
            for key in EXPORT_OUTCOMES
            if outcome_counts.get(key, 0)
        },
        "pass_a_statuses": dict(sorted(pass_a_counts.items())),
        "pass_b_statuses": dict(sorted(pass_b_counts.items())),
        "endpoint_count": endpoint_count,
        "reviewed_windows": reviewed_windows,
        "gold_eligible_windows": gold_eligible_windows,
        "gold_eligible_endpoints": gold_eligible_endpoints,
        "ambiguous_windows": window_counts.get("AMBIGUOUS", 0),
        "excluded_windows": window_counts.get("EXCLUDED", 0),
    }


__all__ = [
    "ADJUDICATION_EXPORT_SCHEMA_VERSION",
    "DECISION_KINDS",
    "EXPORT_FORBIDDEN_KEYS",
    "REVIEW_MATERIAL_STATUS",
    "REVIEWED_PACKET_FILENAME",
    "build_reviewed_packet",
    "derive_resolved_endpoints",
    "load_reviewed_packet",
    "serialize_reviewed_packet",
    "summarize_reviewed_packet",
    "validate_adjudication_export",
]
