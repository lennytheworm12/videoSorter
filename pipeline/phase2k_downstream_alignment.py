"""Phase 2K downstream semantic-target alignment contract (v1).

This module prepares the scorer-blind, post-human-review target alignment
that must exist before the paired Phase 2F generative and Phase 2H
discriminative reruns.  It is contract/tooling only: it never runs
providers, never runs Phase 2F/2H scoring, never fabricates human
decisions, and never performs semantic extraction.

The alignment packet carries one ordered item per Phase 2J KEEP endpoint
(exactly 311 items across the 30 reviewed windows).  Raw Bronze target
identity is preserved exactly, except for the single versioned boundary
rule: the exact 48 candidate-coverage-identified missing endpoints
(``MIXED_BOUNDARY_MISMATCH`` / ``CANDIDATE_GENERATION_MISS``, each ending in
one terminal ``.`` or ``,`` with exactly one overlap candidate at
start/end-1/text-without-terminal-punctuation) use an evaluation span with
that one terminal character dropped.  The other 263 covered endpoints keep
the exact reviewed span.  No Phase 2J artifact is mutated and all 311
endpoint identities remain unchanged.

The builder fails closed unless the Phase 2K output directory is a live
build whose finalized human review artifacts recompute to a PASSED review
gate and whose completed transformation audit validates against the blank
audit and records.  No-provider mode, placeholder/not-generated D records,
missing semantic polish, stale/invalid records, and missing/invalid audits
are all rejected.  No downstream architecture predictions/results may be
inputs to or present in this packet.

The module imports strict/canonical helpers from the Phase 2K core and the
Phase 2J coverage validator, but the core does not import this module, so
there is no circular import.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from pipeline.phase2j_candidate_coverage import load_candidate_coverage
from pipeline.phase2k_contextual_reconstruction import (
    COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION,
    HUMAN_MAPPING_SCHEMA_VERSION,
    HUMAN_PACKET_SCHEMA_VERSION,
    HUMAN_SUMMARY_SCHEMA_VERSION,
    OUTPUT_FILENAMES,
    RELEASE_GATE_AWAITING_REVIEW,
    RELEASE_GATE_REVIEWED,
    ROOT,
    RECORDS_SCHEMA_VERSION,
    TRANSFORMATION_AUDIT_SCHEMA_VERSION,
    canonical_sha256,
    file_sha256,
    load_json_strict,
    load_phase2j_reviewed_packet,
    summarize_human_reviews,
    text_sha256,
    validate_completed_transformation_audits,
    validate_human_review_packet,
    validate_output_directory,
)


ALIGNMENT_PACKET_SCHEMA_VERSION = "phase2k-downstream-alignment-packet-v1"
ALIGNMENT_SUMMARY_SCHEMA_VERSION = "phase2k-downstream-alignment-summary-v1"
BOUNDARY_RULE_VERSION = "phase2k-target-boundary-rule-v1-phase2j-terminal-punctuation"

TARGET_COUNT = 311
TARGET_WINDOW_COUNT = 30
TARGET_RECORD_COUNT = 120
UNCHANGED_ENDPOINT_COUNT = 263
CORRECTED_ENDPOINT_COUNT = 48
MISSING_PERIOD_COUNT = 28
MISSING_COMMA_COUNT = 20

ALIGNMENT_DECISION_STATES = ("ALIGNED", "ABSENT", "AMBIGUOUS", "MULTIPLE_CANDIDATES")
CORRECTION_STATUSES = ("UNCHANGED", "TERMINAL_PUNCTUATION_DROPPED")
TERMINAL_PUNCTUATION = (".", ",")

_HEX64 = re.compile(r"[0-9a-f]{64}")
_NODE_TYPES = frozenset({
    "ENTITY", "ABILITY_OR_RESOURCE", "EVENT", "ACTION", "STATE", "OUTCOME",
    "QUANTITY", "TIME", "LOCATION_OR_SPACE",
})

_TOP_LEVEL_KEYS = (
    "schema_version",
    "content_sha256",
    "purpose",
    "release_gate",
    "dataset_binding",
    "boundary_rule",
    "items",
)
_DATASET_BINDING_KEYS = (
    "phase2k_records_sha256",
    "phase2j_reviewed_packet_sha256",
    "phase2j_coverage_sha256",
    "finalized_human_packet_sha256",
    "human_summary_sha256",
    "completed_transformation_audit_sha256",
    "window_ids_sha256",
    "window_count",
    "target_count",
    "human_review_gate_status",
)
_BOUNDARY_RULE_KEYS = (
    "rule_version",
    "unchanged_count",
    "corrected_count",
    "dropped_terminal_period_count",
    "dropped_terminal_comma_count",
    "behavior",
)
_ITEM_KEYS = (
    "alignment_id",
    "window_id",
    "endpoint_id",
    "node_type",
    "bronze_target",
    "representation",
    "decision",
)
_BRONZE_TARGET_KEYS = (
    "original_start",
    "original_end",
    "original_text",
    "source_absolute_start",
    "source_absolute_end",
    "evaluation_start",
    "evaluation_end",
    "evaluation_text",
    "correction_status",
    "dropped_text",
)
_REPRESENTATION_KEYS = (
    "clean_target_transcript",
    "clean_target_transcript_sha256",
    "polished_text",
    "polished_text_sha256",
)
_DECISION_KEYS = ("state", "polished_spans", "reviewer", "completed_at", "notes")
_SPAN_KEYS = ("start", "end", "text")

# Downstream/model-result fields that must never appear in this packet.  The
# alignment is scorer/model blind by contract.
FORBIDDEN_ALIGNMENT_KEYS = frozenset({
    "model_predictions", "model_scoring", "predictions", "prediction",
    "predicted", "predicted_label", "predicted_labels", "score", "scores",
    "probability", "probabilities", "rank", "ranks", "ranking", "rankings",
    "threshold", "thresholds", "scorer", "scoring", "architecture",
    "architectures", "semantic_ir", "semantic_claims", "semantic_extraction",
    "entities", "relations", "claims", "extracted", "extractor",
    "generative", "discriminative", "model_result", "model_results",
})
FORBIDDEN_ALIGNMENT_VALUES = frozenset({
    "PHASE2F", "PHASE2H", "GENERATIVE", "DISCRIMINATIVE", "SCORED", "PREDICTED",
})


def _require_exact_keys(value: object, expected: Iterable[str], label: str) -> None:
    expected_set = frozenset(expected)
    if not isinstance(value, Mapping) or set(value) != expected_set:
        missing = (
            sorted(expected_set - set(value)) if isinstance(value, Mapping) else []
        )
        extra = sorted(set(value) - expected_set) if isinstance(value, Mapping) else []
        detail = (
            f"missing={missing} extra={extra}"
            if isinstance(value, Mapping)
            else "not an object"
        )
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


def _require_hex64(value: object, label: str) -> str:
    text = _require_string(value, label)
    if _HEX64.fullmatch(text) is None:
        raise ValueError(f"{label} must be 64 lowercase hex characters")
    return text


def _safe_float(value: float) -> float:
    return round(float(value), 4)


def _validate_recomputed_content_hash(obj: Mapping[str, Any], *, label: str) -> None:
    _require_hex64(obj.get("content_sha256"), f"{label} content_sha256")
    expected = canonical_sha256({
        key: value for key, value in obj.items() if key != "content_sha256"
    })
    if obj["content_sha256"] != expected:
        raise ValueError(f"{label} content_sha256 does not match canonical content")


def _scan_forbidden_leaks(value: object, *, path: str) -> None:
    """Reject any key/value that would make the packet scorer/model aware."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_ALIGNMENT_KEYS:
                raise ValueError(
                    f"alignment packet leaks forbidden key {key!r} at {path}",
                )
            _scan_forbidden_leaks(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_forbidden_leaks(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if value in FORBIDDEN_ALIGNMENT_VALUES:
            raise ValueError(
                f"alignment packet leaks forbidden value {value!r} at {path}",
            )


# ---------------------------------------------------------------------------
# Boundary manifest (versioned Phase 2J terminal-punctuation rule)
# ---------------------------------------------------------------------------


def build_boundary_manifest(
    packet: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Classify all 311 KEEP endpoints against the frozen coverage artifact.

    ``packet`` must be the validated Phase 2J reviewed packet and ``coverage``
    the validated candidate-coverage artifact.  Returns a mapping keyed by
    endpoint ID with exact evaluation spans.  This helper never mutates either
    artifact and fails closed on any deviation from the 263/48 rule.
    """
    covered_by_id = {
        record["endpoint_id"]: record
        for record in coverage["covered_endpoints"]
    }
    missing_by_id = {
        record["endpoint_id"]: record
        for record in coverage["missing_endpoints"]
    }
    endpoints: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for record in packet["records"]:
        for endpoint in record["endpoints"]:
            if endpoint.get("disposition") == "KEEP":
                endpoints.append((record, endpoint))
    if len(endpoints) != TARGET_COUNT:
        raise ValueError(
            f"Phase 2J reviewed packet must contain exactly {TARGET_COUNT} "
            f"KEEP endpoints; found {len(endpoints)}",
        )
    if len(covered_by_id) != UNCHANGED_ENDPOINT_COUNT or len(
        missing_by_id,
    ) != CORRECTED_ENDPOINT_COUNT:
        raise ValueError(
            "Phase 2J coverage counts are invalid; expected "
            f"{UNCHANGED_ENDPOINT_COUNT} covered and "
            f"{CORRECTED_ENDPOINT_COUNT} missing",
        )
    packet_ids = {endpoint["endpoint_id"] for _, endpoint in endpoints}
    if covered_by_id.keys() & missing_by_id.keys():
        raise ValueError("Phase 2J coverage endpoint IDs must not overlap")
    if set(covered_by_id) | set(missing_by_id) != packet_ids:
        raise ValueError(
            "Phase 2J coverage endpoint IDs must exactly match the 311 "
            "reviewed KEEP endpoint IDs",
        )
    manifest: dict[str, dict[str, Any]] = {}
    for record, endpoint in endpoints:
        endpoint_id = endpoint["endpoint_id"]
        original_start = _require_int(
            endpoint["char_start"], "endpoint char_start", minimum=0,
        )
        original_end = _require_int(
            endpoint["char_end"], "endpoint char_end", minimum=0,
        )
        original_text = _require_string(
            endpoint["bronze_text"], "endpoint bronze_text",
        )
        if not original_start < original_end <= len(record["bronze_text"]):
            raise ValueError(f"endpoint {endpoint_id} span is out of bounds")
        if record["bronze_text"][original_start:original_end] != original_text:
            raise ValueError(
                f"endpoint {endpoint_id} bronze text is not the exact slice",
            )
        source_absolute_start = _require_int(
            record["upstream_start"], "record upstream_start", minimum=0,
        )
        absolute_start = source_absolute_start + original_start
        absolute_end = source_absolute_start + original_end
        covered = covered_by_id.get(endpoint_id)
        missing = missing_by_id.get(endpoint_id)
        if covered is not None and missing is not None:
            raise ValueError(f"endpoint {endpoint_id} is duplicated in coverage")
        if covered is not None:
            _validate_coverage_identity(
                covered,
                record=record,
                endpoint=endpoint,
                label="covered endpoint",
            )
            manifest[endpoint_id] = {
                "correction_status": "UNCHANGED",
                "original_start": original_start,
                "original_end": original_end,
                "original_text": original_text,
                "evaluation_start": original_start,
                "evaluation_end": original_end,
                "evaluation_text": original_text,
                "dropped_text": None,
                "source_absolute_start": absolute_start,
                "source_absolute_end": absolute_end,
            }
            continue
        if missing is None:
            raise ValueError(
                f"endpoint {endpoint_id} is absent from the coverage artifact",
            )
        _validate_coverage_identity(
            missing,
            record=record,
            endpoint=endpoint,
            label="missing endpoint",
        )
        if missing.get("failure_category") != "MIXED_BOUNDARY_MISMATCH" or (
            missing.get("error_code") != "CANDIDATE_GENERATION_MISS"
        ):
            raise ValueError(
                f"missing endpoint {endpoint_id} has unexpected coverage "
                "classification",
            )
        if not original_text or original_text[-1] not in TERMINAL_PUNCTUATION:
            raise ValueError(
                f"missing endpoint {endpoint_id} must end in one terminal "
                "'.' or ','",
            )
        expected_candidate = original_text[:-1]
        overlap_matches = [
            overlap
            for overlap in missing.get("overlaps", [])
            if overlap.get("start") == original_start
            and overlap.get("end") == original_end - 1
            and overlap.get("text") == expected_candidate
        ]
        if len(overlap_matches) != 1:
            raise ValueError(
                f"missing endpoint {endpoint_id} must have exactly one "
                "overlap candidate at start/end-1/text without terminal "
                "punctuation",
            )
        manifest[endpoint_id] = {
            "correction_status": "TERMINAL_PUNCTUATION_DROPPED",
            "original_start": original_start,
            "original_end": original_end,
            "original_text": original_text,
            "evaluation_start": original_start,
            "evaluation_end": original_end - 1,
            "evaluation_text": expected_candidate,
            "dropped_text": original_text[-1],
            "source_absolute_start": absolute_start,
            "source_absolute_end": absolute_end,
        }
    unchanged = sum(
        1 for entry in manifest.values()
        if entry["correction_status"] == "UNCHANGED"
    )
    corrected = sum(
        1 for entry in manifest.values()
        if entry["correction_status"] == "TERMINAL_PUNCTUATION_DROPPED"
    )
    periods = sum(
        1 for entry in manifest.values() if entry["dropped_text"] == "."
    )
    commas = sum(
        1 for entry in manifest.values() if entry["dropped_text"] == ","
    )
    if (
        unchanged != UNCHANGED_ENDPOINT_COUNT
        or corrected != CORRECTED_ENDPOINT_COUNT
        or periods != MISSING_PERIOD_COUNT
        or commas != MISSING_COMMA_COUNT
    ):
        raise ValueError(
            "Phase 2J boundary manifest counts are invalid; expected "
            f"{UNCHANGED_ENDPOINT_COUNT} unchanged / {CORRECTED_ENDPOINT_COUNT} "
            f"corrected with {MISSING_PERIOD_COUNT} periods and "
            f"{MISSING_COMMA_COUNT} commas",
        )
    return manifest


def _validate_coverage_identity(
    coverage_record: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    label: str,
) -> None:
    endpoint_id = endpoint["endpoint_id"]
    if (
        coverage_record.get("endpoint_id") != endpoint_id
        or coverage_record.get("window_id") != record["window_id"]
        or coverage_record.get("source_group_id") != record["source_group_id"]
        or coverage_record.get("char_start") != endpoint["char_start"]
        or coverage_record.get("char_end") != endpoint["char_end"]
        or coverage_record.get("bronze_text") != endpoint["bronze_text"]
        or coverage_record.get("node_type") != endpoint["node_type"]
    ):
        raise ValueError(
            f"{label} {endpoint_id} identity is inconsistent with the "
            "reviewed packet",
        )
    expected_absolute_start = record["upstream_start"] + endpoint["char_start"]
    expected_absolute_end = record["upstream_start"] + endpoint["char_end"]
    if (
        coverage_record.get("absolute_start") != expected_absolute_start
        or coverage_record.get("absolute_end") != expected_absolute_end
    ):
        raise ValueError(
            f"{label} {endpoint_id} absolute offsets are inconsistent",
        )


# ---------------------------------------------------------------------------
# Packet construction
# ---------------------------------------------------------------------------


def build_dataset_binding(
    *,
    records_obj: Mapping[str, Any],
    reviewed_packet: Mapping[str, Any],
    coverage: Mapping[str, Any],
    finalized_packet: Mapping[str, Any],
    human_summary: Mapping[str, Any],
    completed_audit: Mapping[str, Any],
    window_ids: list[str],
) -> dict[str, Any]:
    """Exact cryptographic dataset binding for the alignment packet."""
    return {
        "phase2k_records_sha256": _require_hex64(
            records_obj.get("content_sha256"),
            "phase2k records content hash",
        ),
        "phase2j_reviewed_packet_sha256": _require_hex64(
            reviewed_packet.get("content_sha256"),
            "phase2j reviewed packet content hash",
        ),
        "phase2j_coverage_sha256": _require_hex64(
            coverage.get("content_sha256"),
            "phase2j coverage content hash",
        ),
        "finalized_human_packet_sha256": _require_hex64(
            finalized_packet.get("content_sha256"),
            "finalized human packet content hash",
        ),
        "human_summary_sha256": canonical_sha256(human_summary),
        "completed_transformation_audit_sha256": _require_hex64(
            completed_audit.get("content_sha256"),
            "completed transformation audit content hash",
        ),
        "window_ids_sha256": canonical_sha256(window_ids),
        "window_count": len(window_ids),
        "target_count": TARGET_COUNT,
        "human_review_gate_status": "PASSED",
    }


def build_boundary_rule() -> dict[str, Any]:
    """Versioned boundary-rule declaration with the exact 263/48 counts."""
    return {
        "rule_version": BOUNDARY_RULE_VERSION,
        "unchanged_count": UNCHANGED_ENDPOINT_COUNT,
        "corrected_count": CORRECTED_ENDPOINT_COUNT,
        "dropped_terminal_period_count": MISSING_PERIOD_COUNT,
        "dropped_terminal_comma_count": MISSING_COMMA_COUNT,
        "behavior": (
            "For the exact 48 Phase 2J candidate-coverage-identified missing "
            "endpoints only (MIXED_BOUNDARY_MISMATCH / "
            "CANDIDATE_GENERATION_MISS, each with exactly one overlap "
            "candidate at the same start/end-1/text without terminal "
            "punctuation), the raw evaluation span is the reviewed span with "
            "exactly one terminal '.' or ',' dropped: start unchanged, end = "
            "original end - 1, text = original text without the terminal "
            "punctuation.  The other 263 covered endpoints preserve the exact "
            "reviewed span.  No Phase 2J reviewed packet or coverage artifact "
            "is mutated and all 311 endpoint identities are preserved."
        ),
    }


def build_alignment_items(
    *,
    records_obj: Mapping[str, Any],
    reviewed_packet: Mapping[str, Any],
    boundary_manifest: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build one ordered blank item per KEEP endpoint.

    Order is stable: the Phase 2K records file's window order (sorted window
    IDs), then each endpoint's position in the reviewed packet record.
    """
    window_ids = sorted({record["window_id"] for record in records_obj["records"]})
    packet_by_window = {
        record["window_id"]: record for record in reviewed_packet["records"]
    }
    d_by_window: dict[str, Mapping[str, Any]] = {}
    for record in records_obj["records"]:
        if record["record_type"] == "D":
            d_by_window[record["window_id"]] = record
    items: list[dict[str, Any]] = []
    for window_id in window_ids:
        packet_record = packet_by_window[window_id]
        d_record = d_by_window[window_id]
        d_content = d_record["content"]
        semantic_polish = d_content["semantic_polish"]
        clean_text = d_content["clean_target_transcript"]
        polished_text = semantic_polish["polished_text"]
        for endpoint in packet_record["endpoints"]:
            if endpoint.get("disposition") != "KEEP":
                continue
            endpoint_id = endpoint["endpoint_id"]
            boundary = boundary_manifest[endpoint_id]
            items.append({
                "alignment_id": f"p2k:align:{endpoint_id}",
                "window_id": window_id,
                "endpoint_id": endpoint_id,
                "node_type": endpoint["node_type"],
                "bronze_target": {
                    "original_start": boundary["original_start"],
                    "original_end": boundary["original_end"],
                    "original_text": boundary["original_text"],
                    "source_absolute_start": boundary["source_absolute_start"],
                    "source_absolute_end": boundary["source_absolute_end"],
                    "evaluation_start": boundary["evaluation_start"],
                    "evaluation_end": boundary["evaluation_end"],
                    "evaluation_text": boundary["evaluation_text"],
                    "correction_status": boundary["correction_status"],
                    "dropped_text": boundary["dropped_text"],
                },
                "representation": {
                    "clean_target_transcript": clean_text,
                    "clean_target_transcript_sha256": text_sha256(clean_text),
                    "polished_text": polished_text,
                    "polished_text_sha256": text_sha256(polished_text),
                },
                "decision": {
                    "state": None,
                    "polished_spans": [],
                    "reviewer": None,
                    "completed_at": None,
                    "notes": [],
                },
            })
    return items


def build_downstream_alignment_packet(
    *,
    phase2k_dir: Path,
    reviewed_packet_path: Path,
    coverage_path: Path,
) -> dict[str, Any]:
    """Build the blank alignment packet after all live-review gates pass.

    Fails closed on no-provider/placeholder/not-generated D records, missing
    semantic polish, missing/stale human-review artifacts, a non-PASSED
    recomputed review gate, and missing/invalid completed audits.
    """
    inputs = load_alignment_inputs(
        phase2k_dir=phase2k_dir,
        reviewed_packet_path=reviewed_packet_path,
        coverage_path=coverage_path,
    )
    boundary_manifest = build_boundary_manifest(
        inputs["reviewed_packet"],
        inputs["coverage"],
    )
    items = build_alignment_items(
        records_obj=inputs["records_obj"],
        reviewed_packet=inputs["reviewed_packet"],
        boundary_manifest=boundary_manifest,
    )
    dataset_binding = build_dataset_binding(
        records_obj=inputs["records_obj"],
        reviewed_packet=inputs["reviewed_packet"],
        coverage=inputs["coverage"],
        finalized_packet=inputs["finalized_packet"],
        human_summary=inputs["human_summary"],
        completed_audit=inputs["completed_audit"],
        window_ids=inputs["window_ids"],
    )
    packet = {
        "schema_version": ALIGNMENT_PACKET_SCHEMA_VERSION,
        "purpose": (
            "Scorer/model-blind Phase 2K downstream semantic-target "
            "alignment packet.  One ordered item per Phase 2J KEEP endpoint "
            "(311 total across 30 reviewed windows) with exact raw Bronze "
            "target identity, the sealed Phase 2K D clean/polished "
            "representation, and blank human alignment decisions.  Built "
            "only after the finalized human review gate is PASSED and the "
            "completed transformation audit is validated; carries no "
            "downstream predictions, model results, scores, or semantic "
            "extraction."
        ),
        "release_gate": RELEASE_GATE_AWAITING_REVIEW,
        "dataset_binding": dataset_binding,
        "boundary_rule": build_boundary_rule(),
        "items": items,
    }
    packet = {
        "content_sha256": canonical_sha256({
            key: value for key, value in packet.items() if key != "content_sha256"
        }),
        **packet,
    }
    validate_downstream_alignment_packet(
        packet,
        require_blank=True,
        bindings=inputs,
    )
    return packet


# ---------------------------------------------------------------------------
# Phase 2K output gate validation
# ---------------------------------------------------------------------------


def _resolve_frozen_input_path(locator: object, *, label: str) -> Path:
    """Resolve a frozen-input path locator to a filesystem path.

    Repository-relative locators resolve from the repository root (the same
    convention ``normalize_path_locator`` uses to record them); absolute
    locators such as an archived transcript DB remain absolute.
    """
    if not isinstance(locator, str) or not locator.strip():
        raise ValueError(f"alignment input locator is missing: {label}")
    path = Path(locator)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_alignment_inputs(
    *,
    phase2k_dir: Path,
    reviewed_packet_path: Path,
    coverage_path: Path,
) -> dict[str, Any]:
    """Validate every Phase 2K/2J input the alignment packet binds."""
    if not isinstance(phase2k_dir, Path):
        phase2k_dir = Path(phase2k_dir)
    records_path = phase2k_dir / OUTPUT_FILENAMES["records"]
    frozen_manifest_path = phase2k_dir / OUTPUT_FILENAMES["frozen_input_manifest"]
    blank_packet_path = phase2k_dir / OUTPUT_FILENAMES["human_packet"]
    mapping_path = phase2k_dir / OUTPUT_FILENAMES["human_mapping"]
    finalized_path = phase2k_dir / OUTPUT_FILENAMES["finalized_packet"]
    summary_path = phase2k_dir / OUTPUT_FILENAMES["human_summary"]
    build_summary_path = phase2k_dir / OUTPUT_FILENAMES["build_summary"]
    audit_path = phase2k_dir / OUTPUT_FILENAMES["transformation_audit"]
    completed_audit_path = phase2k_dir / OUTPUT_FILENAMES[
        "finalized_transformation_audit"
    ]
    for label, path in (
        ("phase2k records", records_path),
        ("phase2k frozen input manifest", frozen_manifest_path),
        ("blank human review packet", blank_packet_path),
        ("human review mapping", mapping_path),
        ("finalized human review packet", finalized_path),
        ("human review summary", summary_path),
        ("phase2k build summary", build_summary_path),
        ("blank transformation audit", audit_path),
        ("completed transformation audit", completed_audit_path),
        ("Phase 2J reviewed packet", reviewed_packet_path),
        ("Phase 2J candidate coverage", coverage_path),
    ):
        if not Path(path).is_file():
            raise ValueError(f"alignment input is missing: {label}: {path}")

    records_obj = load_json_strict(records_path, label="phase2k records")
    if records_obj.get("schema_version") != RECORDS_SCHEMA_VERSION:
        raise ValueError("phase2k records schema version is invalid")
    if records_obj.get("release_gate") != RELEASE_GATE_AWAITING_REVIEW:
        raise ValueError("phase2k records release gate is invalid")
    _validate_recomputed_content_hash(records_obj, label="phase2k records")
    mode = records_obj.get("mode")
    if mode != "live":
        raise ValueError(
            "Phase 2K downstream alignment requires live Phase 2K records; "
            "no-provider/placeholder mode is rejected",
        )
    _validate_records_structure(records_obj)

    blank_packet = load_json_strict(
        blank_packet_path, label="phase2k blank human review packet",
    )
    mapping = load_json_strict(
        mapping_path, label="phase2k human review mapping",
    )
    finalized_packet = load_json_strict(
        finalized_path, label="phase2k finalized human review packet",
    )
    stored_summary = load_json_strict(
        summary_path, label="phase2k human review summary",
    )
    validate_human_review_packet(blank_packet, require_blank=True)
    validate_human_review_packet(finalized_packet, require_blank=False)
    if mapping.get("schema_version") != HUMAN_MAPPING_SCHEMA_VERSION:
        raise ValueError("phase2k human review mapping schema version is invalid")
    _validate_recomputed_content_hash(mapping, label="phase2k human review mapping")
    for packet_obj, label in (
        (blank_packet, "blank human review packet"),
        (finalized_packet, "finalized human review packet"),
    ):
        if packet_obj.get("blinding", {}).get("mapping_sha256") != mapping.get(
            "content_sha256",
        ):
            raise ValueError(
                f"phase2k {label} is not bound to the human review mapping",
            )
    if stored_summary.get("schema_version") != HUMAN_SUMMARY_SCHEMA_VERSION:
        raise ValueError("phase2k human review summary schema version is invalid")
    recomputed_summary = summarize_human_reviews(
        finalized_packet,
        mapping=mapping,
        records_file=records_obj,
    )
    if stored_summary != recomputed_summary:
        raise ValueError(
            "phase2k human review summary does not match its finalized inputs",
        )
    if recomputed_summary.get("review_gate", {}).get("status") != "PASSED":
        raise ValueError(
            "Phase 2K downstream alignment requires the recomputed human "
            "review gate to be PASSED",
        )

    audit_template = load_json_strict(
        audit_path, label="phase2k blank transformation audit",
    )
    completed_audit = load_json_strict(
        completed_audit_path, label="phase2k completed transformation audit",
    )
    if audit_template.get("schema_version") != TRANSFORMATION_AUDIT_SCHEMA_VERSION:
        raise ValueError("phase2k blank transformation audit schema version is invalid")
    if completed_audit.get("schema_version") != (
        COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION
    ):
        raise ValueError(
            "phase2k completed transformation audit schema version is invalid",
        )
    completed_audit = validate_completed_transformation_audits(
        audit_template,
        completed_audit,
        records_obj=records_obj,
    )

    reviewed_packet = load_phase2j_reviewed_packet(reviewed_packet_path)
    coverage = load_candidate_coverage(coverage_path)
    if coverage["reviewed_packet"]["content_sha256"] != reviewed_packet[
        "content_sha256"
    ]:
        raise ValueError(
            "phase2j candidate coverage is not bound to the reviewed packet "
            "content hash",
        )
    if coverage["reviewed_packet"]["file_sha256"] != file_sha256(
        reviewed_packet_path,
    ):
        raise ValueError(
            "phase2j candidate coverage is not bound to the reviewed packet "
            "file hash",
        )
    window_ids = sorted({record["window_id"] for record in records_obj["records"]})
    packet_windows = sorted({
        record["window_id"] for record in reviewed_packet["records"]
    })
    if len(window_ids) != TARGET_WINDOW_COUNT or window_ids != packet_windows:
        raise ValueError(
            "Phase 2K records and Phase 2J reviewed packet window IDs must "
            "align across all 30 windows",
        )
    frozen_manifest = load_json_strict(
        frozen_manifest_path, label="phase2k frozen input manifest",
    )
    phase2j_inputs = frozen_manifest.get("phase2j_inputs")
    if not isinstance(phase2j_inputs, Mapping) or not isinstance(
        phase2j_inputs.get("manifest"), Mapping,
    ):
        raise ValueError(
            "phase2k frozen input manifest is missing its Phase 2J input "
            "records",
        )
    transcript_db = frozen_manifest.get("transcript_db")
    if not isinstance(transcript_db, Mapping):
        raise ValueError(
            "phase2k frozen input manifest is missing its transcript DB "
            "record",
        )
    manifest_path = _resolve_frozen_input_path(
        phase2j_inputs["manifest"].get("path"),
        label="Phase 2J window-selection manifest",
    )
    db_path = _resolve_frozen_input_path(
        transcript_db.get("path"),
        label="Phase 2K transcript DB",
    )
    for label, path in (
        ("Phase 2J window-selection manifest", manifest_path),
        ("Phase 2K transcript DB", db_path),
    ):
        if not Path(path).is_file():
            raise ValueError(f"alignment input is missing: {label}: {path}")
    # Deep current-contract validation of the live output: current
    # pipeline/prompt/config schema versions, sealed reconstruction/polish
    # validation, diagnostic attempts, raw response files, provider lineage,
    # and the frozen Phase 2J manifest/transcript DB locators.  A
    # self-consistent stale output must fail here, before it can be accepted.
    validate_output_directory(
        output_dir=phase2k_dir,
        manifest_path=manifest_path,
        packet_path=reviewed_packet_path,
        db_path=db_path,
    )
    return {
        "phase2k_dir": phase2k_dir,
        "records_obj": records_obj,
        "blank_packet": blank_packet,
        "human_mapping": mapping,
        "finalized_packet": finalized_packet,
        "human_summary": recomputed_summary,
        "completed_audit": completed_audit,
        "reviewed_packet": reviewed_packet,
        "coverage": coverage,
        "window_ids": window_ids,
    }


def _validate_records_structure(records_obj: Mapping[str, Any]) -> None:
    """Structural Phase 2K record gates used by the alignment builder."""
    records = _require_list(records_obj.get("records"), "phase2k records list")
    if len(records) != TARGET_RECORD_COUNT:
        raise ValueError(
            f"phase2k records must contain {TARGET_RECORD_COUNT} A/B/C/D "
            f"records; found {len(records)}",
        )
    by_window: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("phase2k records entries must be objects")
        window_id = _require_nonempty_string(record.get("window_id"), "record window_id")
        record_type = _require_enum(
            record.get("record_type"), ("A", "B", "C", "D"), "record type",
        )
        by_window.setdefault(window_id, {})[record_type] = record
    if len(by_window) != TARGET_WINDOW_COUNT:
        raise ValueError(
            f"phase2k records must cover exactly {TARGET_WINDOW_COUNT} windows",
        )
    for window_id, by_type in by_window.items():
        if set(by_type) != {"A", "B", "C", "D"}:
            raise ValueError(
                f"phase2k records are incomplete for window {window_id}",
            )
        base_target = by_type["A"].get("target")
        if not isinstance(base_target, Mapping):
            raise ValueError(f"phase2k A target is missing for {window_id}")
        for record_type in ("B", "C", "D"):
            if by_type[record_type].get("target") != base_target:
                raise ValueError(
                    f"phase2k {record_type} target identity is not invariant "
                    f"for {window_id}",
                )
        if by_type["A"].get("content", {}).get("kind") != "raw_bronze":
            raise ValueError(f"phase2k A record is not raw Bronze for {window_id}")
    for record in records:
        if record.get("canonical_record_sha256") != canonical_sha256({
            key: value for key, value in record.items()
            if key != "canonical_record_sha256"
        }):
            raise ValueError(
                f"phase2k record canonical hash is invalid for "
                f"{record.get('record_id')}",
            )
        d_content = record.get("content", {})
        if record["record_type"] == "D":
            _validate_d_record_gates(d_content, record["window_id"])


def _validate_d_record_gates(
    d_content: Mapping[str, Any],
    window_id: str,
) -> None:
    """Reject placeholders, not-generated D, and missing semantic polish."""
    if not isinstance(d_content, Mapping):
        raise ValueError(f"phase2k D content is missing for {window_id}")
    if d_content.get("generation_status") != "GENERATED":
        raise ValueError(
            f"phase2k D record for {window_id} is not GENERATED; "
            "downstream alignment requires generated D records",
        )
    if d_content.get("is_placeholder") is not False:
        raise ValueError(
            f"phase2k D record for {window_id} is a placeholder; "
            "downstream alignment rejects placeholders",
        )
    reconstruction = d_content.get("reconstruction")
    semantic_polish = d_content.get("semantic_polish")
    if not isinstance(reconstruction, Mapping) or not isinstance(
        semantic_polish, Mapping,
    ):
        raise ValueError(
            f"phase2k D record for {window_id} is missing a sealed "
            "reconstruction/semantic-polish subobject",
        )
    if reconstruction.get("generation_status") != "GENERATED" or (
        semantic_polish.get("generation_status") != "GENERATED"
    ):
        raise ValueError(
            f"phase2k D subobjects for {window_id} are not both GENERATED",
        )
    clean_text = d_content.get("clean_target_transcript")
    polished_text = semantic_polish.get("polished_text")
    if not isinstance(clean_text, str) or not isinstance(polished_text, str):
        raise ValueError(
            f"phase2k D record for {window_id} has invalid clean/polished text",
        )
    if d_content.get("clean_target_transcript_sha256") != text_sha256(clean_text):
        raise ValueError(
            f"phase2k D clean transcript hash is invalid for {window_id}",
        )


# ---------------------------------------------------------------------------
# Packet validation
# ---------------------------------------------------------------------------


def _validate_bronze_target(
    bronze_target: object,
    *,
    label: str,
) -> dict[str, Any]:
    _require_exact_keys(bronze_target, _BRONZE_TARGET_KEYS, label)
    original_start = _require_int(
        bronze_target["original_start"], f"{label} original_start", minimum=0,
    )
    original_end = _require_int(
        bronze_target["original_end"], f"{label} original_end", minimum=0,
    )
    original_text = _require_string(
        bronze_target["original_text"], f"{label} original_text",
    )
    source_absolute_start = _require_int(
        bronze_target["source_absolute_start"],
        f"{label} source_absolute_start",
        minimum=0,
    )
    source_absolute_end = _require_int(
        bronze_target["source_absolute_end"],
        f"{label} source_absolute_end",
        minimum=0,
    )
    evaluation_start = _require_int(
        bronze_target["evaluation_start"], f"{label} evaluation_start", minimum=0,
    )
    evaluation_end = _require_int(
        bronze_target["evaluation_end"], f"{label} evaluation_end", minimum=0,
    )
    evaluation_text = _require_string(
        bronze_target["evaluation_text"], f"{label} evaluation_text",
    )
    if not original_start < original_end:
        raise ValueError(f"{label} original span is invalid")
    if source_absolute_end - source_absolute_start != (
        original_end - original_start
    ):
        raise ValueError(f"{label} source absolute span is inconsistent")
    status = _require_enum(
        bronze_target["correction_status"], CORRECTION_STATUSES,
        f"{label} correction_status",
    )
    dropped_text = bronze_target["dropped_text"]
    if status == "UNCHANGED":
        if dropped_text is not None:
            raise ValueError(f"{label} UNCHANGED target must have null dropped_text")
        if (
            evaluation_start != original_start
            or evaluation_end != original_end
            or evaluation_text != original_text
        ):
            raise ValueError(f"{label} UNCHANGED evaluation span must equal the original")
    else:
        if dropped_text not in TERMINAL_PUNCTUATION:
            raise ValueError(f"{label} dropped_text must be '.' or ','")
        if (
            evaluation_start != original_start
            or evaluation_end != original_end - 1
            or evaluation_text != original_text[:-1]
        ):
            raise ValueError(
                f"{label} corrected evaluation span must drop exactly one "
                "terminal punctuation character",
            )
    return {
        "original_start": original_start,
        "original_end": original_end,
        "original_text": original_text,
        "source_absolute_start": source_absolute_start,
        "source_absolute_end": source_absolute_end,
        "evaluation_start": evaluation_start,
        "evaluation_end": evaluation_end,
        "evaluation_text": evaluation_text,
        "correction_status": status,
        "dropped_text": dropped_text,
    }


def _validate_polished_span(
    span: object,
    *,
    polished_text: str,
    label: str,
) -> dict[str, Any]:
    _require_exact_keys(span, _SPAN_KEYS, label)
    start = _require_int(span["start"], f"{label} start", minimum=0)
    end = _require_int(span["end"], f"{label} end", minimum=0)
    text = _require_string(span["text"], f"{label} text")
    if not start < end <= len(polished_text):
        raise ValueError(
            f"{label} span is out of bounds for the polished text",
        )
    if polished_text[start:end] != text:
        raise ValueError(f"{label} span is not the exact half-open slice")
    return {"start": start, "end": end, "text": text}


def _validate_decision(
    decision: object,
    *,
    polished_text: str,
    require_blank: bool,
    label: str,
) -> dict[str, Any]:
    _require_exact_keys(decision, _DECISION_KEYS, label)
    if require_blank:
        if decision["state"] is not None:
            raise ValueError(f"blank {label} must have a null state")
        spans = _require_list(decision["polished_spans"], f"{label} spans")
        if spans:
            raise ValueError(f"blank {label} must have empty polished spans")
        if decision["reviewer"] is not None or decision["completed_at"] is not None:
            raise ValueError(f"blank {label} must have null reviewer/completed_at")
        notes = _require_list(decision["notes"], f"{label} notes")
        if notes:
            raise ValueError(f"blank {label} must have empty notes")
        return {
            "state": None,
            "polished_spans": [],
            "reviewer": None,
            "completed_at": None,
            "notes": [],
        }
    state = _require_enum(decision["state"], ALIGNMENT_DECISION_STATES, f"{label} state")
    reviewer = _require_nonempty_string(decision["reviewer"], f"{label} reviewer")
    completed_at = _require_nonempty_string(
        decision["completed_at"], f"{label} completed_at",
    )
    notes = _require_list(decision["notes"], f"{label} notes")
    if any(not isinstance(note, str) for note in notes):
        raise ValueError(f"{label} notes must be strings")
    spans = _require_list(decision["polished_spans"], f"{label} polished spans")
    validated_spans = [
        _validate_polished_span(
            span,
            polished_text=polished_text,
            label=f"{label} polished_spans[{index}]",
        )
        for index, span in enumerate(spans)
    ]
    pairs = [(span["start"], span["end"]) for span in validated_spans]
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{label} polished spans must be unique")
    if pairs != sorted(pairs):
        raise ValueError(
            f"{label} polished spans must be deterministically sorted by "
            "(start, end)",
        )
    if state == "ALIGNED" and not validated_spans:
        raise ValueError(f"{label} ALIGNED requires at least one polished span")
    if state == "ABSENT" and validated_spans:
        raise ValueError(f"{label} ABSENT requires zero polished spans")
    if state == "MULTIPLE_CANDIDATES" and len(validated_spans) < 2:
        raise ValueError(
            f"{label} MULTIPLE_CANDIDATES requires at least two polished spans",
        )
    return {
        "state": state,
        "polished_spans": validated_spans,
        "reviewer": reviewer,
        "completed_at": completed_at,
        "notes": list(notes),
    }


def _validate_boundary_rule_from_items(items: list[Mapping[str, Any]]) -> None:
    unchanged = 0
    corrected = 0
    periods = 0
    commas = 0
    for item in items:
        status = item["bronze_target"]["correction_status"]
        if status == "UNCHANGED":
            unchanged += 1
        elif status == "TERMINAL_PUNCTUATION_DROPPED":
            corrected += 1
            dropped = item["bronze_target"]["dropped_text"]
            if dropped == ".":
                periods += 1
            elif dropped == ",":
                commas += 1
    if (
        unchanged != UNCHANGED_ENDPOINT_COUNT
        or corrected != CORRECTED_ENDPOINT_COUNT
        or periods != MISSING_PERIOD_COUNT
        or commas != MISSING_COMMA_COUNT
    ):
        raise ValueError(
            "alignment item boundary counts are invalid; expected "
            f"{UNCHANGED_ENDPOINT_COUNT} unchanged / {CORRECTED_ENDPOINT_COUNT} "
            f"corrected with {MISSING_PERIOD_COUNT} periods and "
            f"{MISSING_COMMA_COUNT} commas",
        )


def _validate_cross_item_span_uniqueness(items: list[Mapping[str, Any]]) -> None:
    """One selected polished span must never count as two endpoint targets."""
    assigned: dict[tuple[str, int, int], str] = {}
    for item in items:
        state = item["decision"]["state"]
        if state not in ("ALIGNED", "MULTIPLE_CANDIDATES"):
            continue
        for span in item["decision"]["polished_spans"]:
            key = (item["window_id"], span["start"], span["end"])
            previous = assigned.get(key)
            if previous is not None and previous != item["alignment_id"]:
                raise ValueError(
                    f"cross-target duplicate polished span {key[1:]}:{key[0]} "
                    f"is assigned to both {previous} and {item['alignment_id']}",
                )
            assigned[key] = item["alignment_id"]


def validate_downstream_alignment_packet(
    packet: object,
    *,
    require_blank: bool,
    bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strict, canonical, fail-closed alignment packet validation.

    With ``bindings`` (the dict returned by ``load_alignment_inputs``) every
    dataset/hash/item source identity is cross-checked; without bindings the
    packet is validated self-consistently (used by the finalizer).
    """
    if not isinstance(packet, Mapping):
        raise ValueError("phase2k downstream alignment packet must be an object")
    _require_exact_keys(packet, _TOP_LEVEL_KEYS, "phase2k downstream alignment packet")
    if packet["schema_version"] != ALIGNMENT_PACKET_SCHEMA_VERSION:
        raise ValueError("phase2k downstream alignment packet schema version is invalid")
    _validate_recomputed_content_hash(
        packet, label="phase2k downstream alignment packet",
    )
    purpose = _require_nonempty_string(
        packet["purpose"], "alignment packet purpose",
    )
    expected_gate = (
        RELEASE_GATE_AWAITING_REVIEW if require_blank else RELEASE_GATE_REVIEWED
    )
    if packet["release_gate"] != expected_gate:
        raise ValueError("alignment packet release gate is invalid")
    _scan_forbidden_leaks(packet, path="packet")

    dataset_binding = packet["dataset_binding"]
    _require_exact_keys(
        dataset_binding, _DATASET_BINDING_KEYS, "alignment packet dataset_binding",
    )
    for key in (
        "phase2k_records_sha256",
        "phase2j_reviewed_packet_sha256",
        "phase2j_coverage_sha256",
        "finalized_human_packet_sha256",
        "human_summary_sha256",
        "completed_transformation_audit_sha256",
        "window_ids_sha256",
    ):
        _require_hex64(dataset_binding[key], f"alignment packet dataset_binding.{key}")
    window_count = _require_int(
        dataset_binding["window_count"],
        "alignment packet dataset_binding.window_count",
        minimum=1,
    )
    if window_count != TARGET_WINDOW_COUNT:
        raise ValueError("alignment packet dataset_binding.window_count must be 30")
    target_count = _require_int(
        dataset_binding["target_count"],
        "alignment packet dataset_binding.target_count",
        minimum=1,
    )
    if target_count != TARGET_COUNT:
        raise ValueError("alignment packet dataset_binding.target_count must be 311")
    if dataset_binding["human_review_gate_status"] != "PASSED":
        raise ValueError("alignment packet dataset_binding gate status must be PASSED")

    boundary_rule = packet["boundary_rule"]
    _require_exact_keys(boundary_rule, _BOUNDARY_RULE_KEYS, "alignment packet boundary_rule")
    if boundary_rule["rule_version"] != BOUNDARY_RULE_VERSION:
        raise ValueError("alignment packet boundary rule version is invalid")
    for key, expected in (
        ("unchanged_count", UNCHANGED_ENDPOINT_COUNT),
        ("corrected_count", CORRECTED_ENDPOINT_COUNT),
        ("dropped_terminal_period_count", MISSING_PERIOD_COUNT),
        ("dropped_terminal_comma_count", MISSING_COMMA_COUNT),
    ):
        if _require_int(
            boundary_rule[key], f"alignment packet boundary_rule.{key}",
        ) != expected:
            raise ValueError(f"alignment packet boundary_rule.{key} is invalid")
    _require_nonempty_string(boundary_rule["behavior"], "alignment packet boundary behavior")

    items = _require_list(packet["items"], "alignment packet items")
    if len(items) != TARGET_COUNT:
        raise ValueError(
            f"alignment packet must contain exactly {TARGET_COUNT} items",
        )
    seen_alignment_ids: set[str] = set()
    validated_items: list[dict[str, Any]] = []
    packet_by_window: dict[str, Mapping[str, Any]] | None = None
    d_by_window: dict[str, Mapping[str, Any]] | None = None
    endpoint_by_id: dict[str, Mapping[str, Any]] | None = None
    if bindings is not None:
        packet_by_window = {
            record["window_id"]: record
            for record in bindings["reviewed_packet"]["records"]
        }
        endpoint_by_id = {}
        for record in bindings["reviewed_packet"]["records"]:
            for endpoint in record["endpoints"]:
                endpoint_by_id[endpoint["endpoint_id"]] = endpoint
        d_by_window = {}
        for record in bindings["records_obj"]["records"]:
            if record["record_type"] == "D":
                d_by_window[record["window_id"]] = record
        expected_binding = build_dataset_binding(
            records_obj=bindings["records_obj"],
            reviewed_packet=bindings["reviewed_packet"],
            coverage=bindings["coverage"],
            finalized_packet=bindings["finalized_packet"],
            human_summary=bindings["human_summary"],
            completed_audit=bindings["completed_audit"],
            window_ids=bindings["window_ids"],
        )
        if dataset_binding != expected_binding:
            raise ValueError("alignment packet dataset_binding does not match its inputs")

    for index, item in enumerate(items):
        label = f"alignment packet items[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} must be an object")
        _require_exact_keys(item, _ITEM_KEYS, label)
        alignment_id = _require_nonempty_string(
            item["alignment_id"], f"{label} alignment_id",
        )
        if not alignment_id.startswith("p2k:align:"):
            raise ValueError(f"{label} alignment_id prefix is invalid")
        if alignment_id in seen_alignment_ids:
            raise ValueError("alignment item IDs must be unique")
        seen_alignment_ids.add(alignment_id)
        window_id = _require_nonempty_string(
            item["window_id"], f"{label} window_id",
        )
        endpoint_id = _require_nonempty_string(
            item["endpoint_id"], f"{label} endpoint_id",
        )
        if alignment_id != f"p2k:align:{endpoint_id}":
            raise ValueError(f"{label} alignment_id must derive from endpoint_id")
        node_type = item["node_type"]
        if node_type is not None and node_type not in _NODE_TYPES:
            raise ValueError(f"{label} node_type is invalid")
        if bindings is not None:
            endpoint = endpoint_by_id.get(endpoint_id)
            packet_record = packet_by_window.get(window_id)
            if endpoint is None or packet_record is None:
                raise ValueError(f"{label} endpoint identity is not in the reviewed packet")
            if endpoint.get("node_type") != node_type:
                raise ValueError(
                    f"{label} node_type must be inherited from the bound "
                    "reviewed packet",
                )
            if endpoint.get("disposition") != "KEEP":
                raise ValueError(f"{label} endpoint is not KEEP")
            d_record = d_by_window.get(window_id)
            if d_record is None:
                raise ValueError(f"{label} window has no sealed D record")
            d_content = d_record["content"]
            semantic_polish = d_content["semantic_polish"]
        bronze_target = _validate_bronze_target(
            item["bronze_target"], label=f"{label} bronze_target",
        )
        if bindings is not None:
            expected_original = {
                "original_start": endpoint["char_start"],
                "original_end": endpoint["char_end"],
                "original_text": endpoint["bronze_text"],
                "source_absolute_start": (
                    packet_record["upstream_start"] + endpoint["char_start"]
                ),
                "source_absolute_end": (
                    packet_record["upstream_start"] + endpoint["char_end"]
                ),
            }
            for key, expected in expected_original.items():
                if bronze_target[key] != expected:
                    raise ValueError(
                        f"{label} bronze_target.{key} does not match the "
                        "reviewed packet",
                    )
        representation = item["representation"]
        if not isinstance(representation, Mapping):
            raise ValueError(f"{label} representation must be an object")
        _require_exact_keys(representation, _REPRESENTATION_KEYS, f"{label} representation")
        clean_text = _require_string(
            representation["clean_target_transcript"],
            f"{label} representation clean_target_transcript",
        )
        polished_text = _require_string(
            representation["polished_text"],
            f"{label} representation polished_text",
        )
        if representation["clean_target_transcript_sha256"] != text_sha256(clean_text):
            raise ValueError(f"{label} representation clean hash is invalid")
        if representation["polished_text_sha256"] != text_sha256(polished_text):
            raise ValueError(f"{label} representation polished hash is invalid")
        if bindings is not None:
            if clean_text != d_content["clean_target_transcript"]:
                raise ValueError(
                    f"{label} clean text must equal the sealed D "
                    "clean_target_transcript",
                )
            if polished_text != semantic_polish["polished_text"]:
                raise ValueError(
                    f"{label} polished text must equal the sealed D "
                    "semantic_polish.polished_text",
                )
            if representation["clean_target_transcript_sha256"] != d_content.get(
                "clean_target_transcript_sha256",
            ):
                raise ValueError(f"{label} clean hash must match the sealed D record")
        decision = _validate_decision(
            item["decision"],
            polished_text=polished_text,
            require_blank=require_blank,
            label=f"{label} decision",
        )
        validated_items.append({
            **item,
            "bronze_target": bronze_target,
            "decision": decision,
        })
    if len(seen_alignment_ids) != len(items):
        raise ValueError("alignment item IDs are not unique")
    _validate_boundary_rule_from_items(validated_items)
    if not require_blank:
        _validate_cross_item_span_uniqueness(validated_items)
    return {
        "schema_version": ALIGNMENT_PACKET_SCHEMA_VERSION,
        "content_sha256": packet["content_sha256"],
        "purpose": purpose,
        "release_gate": packet["release_gate"],
        "dataset_binding": dict(dataset_binding),
        "boundary_rule": dict(boundary_rule),
        "items": validated_items,
    }


# ---------------------------------------------------------------------------
# Finalization and summary
# ---------------------------------------------------------------------------


def finalize_downstream_alignment_packet(
    packet: Mapping[str, Any],
    decisions: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply a compact alignment decision map to a blank packet.

    ``decisions`` must map every ``alignment_id`` to an object with exact keys
    ``state``/``polished_spans``/``reviewer``/``completed_at``/``notes``.
    Source/display content is preserved exactly; no decision is inferred.
    With ``bindings`` (the dict returned by ``load_alignment_inputs``) both
    the blank packet and the finalized packet are cross-checked against the
    current Phase 2K/2J sources, so a canonical-but-forged packet is
    rejected.
    """
    validate_downstream_alignment_packet(
        packet, require_blank=True, bindings=bindings,
    )
    item_ids = {item["alignment_id"] for item in packet["items"]}
    if set(decisions) != item_ids:
        missing = sorted(item_ids - set(decisions))
        extra = sorted(set(decisions) - item_ids)
        raise ValueError(
            "alignment decisions must cover every alignment ID exactly; "
            f"missing={missing} extra={extra}",
        )
    finalized_items: list[dict[str, Any]] = []
    for item in packet["items"]:
        decision = decisions[item["alignment_id"]]
        if not isinstance(decision, Mapping):
            raise ValueError("alignment decision entry must be an object")
        _require_exact_keys(
            decision,
            _DECISION_KEYS,
            f"alignment decision {item['alignment_id']}",
        )
        state = _require_enum(
            decision["state"],
            ALIGNMENT_DECISION_STATES,
            f"alignment decision {item['alignment_id']} state",
        )
        reviewer = _require_nonempty_string(
            decision["reviewer"],
            f"alignment decision {item['alignment_id']} reviewer",
        )
        completed_at = _require_nonempty_string(
            decision["completed_at"],
            f"alignment decision {item['alignment_id']} completed_at",
        )
        notes = _require_list(
            decision["notes"],
            f"alignment decision {item['alignment_id']} notes",
        )
        if any(not isinstance(note, str) for note in notes):
            raise ValueError(
                f"alignment decision {item['alignment_id']} notes must be strings",
            )
        raw_spans = _require_list(
            decision["polished_spans"],
            f"alignment decision {item['alignment_id']} polished_spans",
        )
        polished_text = item["representation"]["polished_text"]
        validated_spans = [
            _validate_polished_span(
                span,
                polished_text=polished_text,
                label=(
                    f"alignment decision {item['alignment_id']} "
                    f"polished_spans[{index}]"
                ),
            )
            for index, span in enumerate(raw_spans)
        ]
        validated_spans.sort(key=lambda span: (span["start"], span["end"]))
        pairs = [(span["start"], span["end"]) for span in validated_spans]
        if len(set(pairs)) != len(pairs):
            raise ValueError(
                f"alignment decision {item['alignment_id']} spans must be unique",
            )
        if state == "ALIGNED" and not validated_spans:
            raise ValueError(
                f"alignment decision {item['alignment_id']} ALIGNED requires "
                "at least one span",
            )
        if state == "ABSENT" and validated_spans:
            raise ValueError(
                f"alignment decision {item['alignment_id']} ABSENT requires "
                "zero spans",
            )
        if state == "MULTIPLE_CANDIDATES" and len(validated_spans) < 2:
            raise ValueError(
                f"alignment decision {item['alignment_id']} "
                "MULTIPLE_CANDIDATES requires at least two spans",
            )
        finalized_items.append({
            **item,
            "decision": {
                "state": state,
                "polished_spans": validated_spans,
                "reviewer": reviewer,
                "completed_at": completed_at,
                "notes": list(notes),
            },
        })
    finalized = {
        "schema_version": ALIGNMENT_PACKET_SCHEMA_VERSION,
        "purpose": packet["purpose"],
        "release_gate": RELEASE_GATE_REVIEWED,
        "dataset_binding": dict(packet["dataset_binding"]),
        "boundary_rule": dict(packet["boundary_rule"]),
        "items": finalized_items,
    }
    finalized = {
        "content_sha256": canonical_sha256({
            key: value for key, value in finalized.items() if key != "content_sha256"
        }),
        **finalized,
    }
    validate_downstream_alignment_packet(
        finalized, require_blank=False, bindings=bindings,
    )
    return finalized


def build_alignment_summary(finalized: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic summary over the finalized alignment packet."""
    validate_downstream_alignment_packet(finalized, require_blank=False)
    items = finalized["items"]
    total = len(items)

    def _rate(count: int) -> float:
        return _safe_float(count / total) if total else 0.0

    by_state = {
        state: {"count": 0, "rate": 0.0}
        for state in ALIGNMENT_DECISION_STATES
    }
    by_node_type: dict[str, dict[str, Any]] = {}
    by_window: dict[str, dict[str, Any]] = {}
    boundary = {
        "unchanged_count": 0,
        "corrected_count": 0,
        "dropped_terminal_period_count": 0,
        "dropped_terminal_comma_count": 0,
    }
    unresolved_ids: list[str] = []
    for item in items:
        state = item["decision"]["state"]
        by_state[state]["count"] += 1
        node_key = item["node_type"] if item["node_type"] is not None else "null"
        node_entry = by_node_type.setdefault(
            node_key, {"count": 0, "rate": 0.0},
        )
        node_entry["count"] += 1
        window_entry = by_window.setdefault(item["window_id"], {
            "total": 0,
            "rate": 0.0,
            "by_state": {
                state_name: {"count": 0, "rate": 0.0}
                for state_name in ALIGNMENT_DECISION_STATES
            },
        })
        window_entry["total"] += 1
        window_entry["by_state"][state]["count"] += 1
        if state in ("ABSENT", "AMBIGUOUS"):
            unresolved_ids.append(item["alignment_id"])
        status = item["bronze_target"]["correction_status"]
        if status == "UNCHANGED":
            boundary["unchanged_count"] += 1
        else:
            boundary["corrected_count"] += 1
            dropped = item["bronze_target"]["dropped_text"]
            if dropped == ".":
                boundary["dropped_terminal_period_count"] += 1
            elif dropped == ",":
                boundary["dropped_terminal_comma_count"] += 1
    for state, entry in by_state.items():
        entry["rate"] = _rate(entry["count"])
    for node_key, entry in by_node_type.items():
        entry["rate"] = _rate(entry["count"])
    for window_entry in by_window.values():
        window_entry["rate"] = _rate(window_entry["total"])
        for state, entry in window_entry["by_state"].items():
            entry["rate"] = _safe_float(
                entry["count"] / window_entry["total"],
            ) if window_entry["total"] else 0.0
    return {
        "schema_version": ALIGNMENT_SUMMARY_SCHEMA_VERSION,
        "alignment_packet_sha256": finalized["content_sha256"],
        "total": total,
        "by_state": by_state,
        "by_node_type": dict(sorted(by_node_type.items())),
        "by_window": {
            window_id: by_window[window_id]
            for window_id in sorted(by_window)
        },
        "boundary_corrections": boundary,
        "unresolved_targets": {
            "count": len(unresolved_ids),
            "alignment_ids": unresolved_ids,
        },
    }


def validate_alignment_summary(
    summary: object,
    *,
    finalized: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail-closed validation of the deterministic alignment summary."""
    if not isinstance(summary, Mapping):
        raise ValueError("phase2k downstream alignment summary must be an object")
    _require_exact_keys(
        summary,
        (
            "schema_version",
            "alignment_packet_sha256",
            "total",
            "by_state",
            "by_node_type",
            "by_window",
            "boundary_corrections",
            "unresolved_targets",
        ),
        "phase2k downstream alignment summary",
    )
    if summary["schema_version"] != ALIGNMENT_SUMMARY_SCHEMA_VERSION:
        raise ValueError("phase2k downstream alignment summary schema version is invalid")
    expected = build_alignment_summary(finalized)
    if summary != expected:
        raise ValueError("phase2k downstream alignment summary does not match the packet")
    return dict(summary)
