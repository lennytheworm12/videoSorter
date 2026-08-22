"""Phase 2J frozen candidate-coverage gate (scorer-blind discovery coverage).

This module maps the newly imported, scorer-blind reviewed Phase 2J gold
endpoints to the EXACT frozen mention-candidate generator and emits a
self-verifying coverage artifact.  It is discovery coverage only: no model is
scored, no syntax is parsed, nothing is tuned, and candidate generation is
never modified or wrapped.

Every input is loaded with duplicate-key rejection and strict validation:
the locked selection manifest and the reviewed annotation packet must agree on
the exact 30-window identity/order/partition/source-group/Bronze bindings, the
packet must be fully gold eligible (all 30 windows REVIEWED/CLEAN), and every
per-window candidate catalog is regenerated with the frozen generator and the
exact Bronze source/window contract and verified against BOTH the manifest and
the packet catalog bindings (generator version, candidate count, canonical
catalog hash).

Gold-to-candidate matching is exact local Bronze character span
``(char_start, char_end)``; each covered endpoint maps to exactly one candidate
row/span.  Remaining candidates are not manually labeled here.  Missing
endpoints carry deterministic failure categories plus every overlapping
candidate's ID, span, and text.  The FROZEN_REPLICATION partition is reported
only as coverage metadata and never as model performance; the artifact
explicitly declares that model scoring, predictions, and thresholds are
absent.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from pipeline.phase2j_annotation_packet import (
    NODE_TYPES,
    is_packet_gold_eligible,
    is_window_gold_eligible,
    load_annotation_packet,
    validate_annotation_packet,
)
from pipeline.phase2j_source_selection import (
    PARTITIONS,
    PARTITION_SIZES,
    TARGET_WINDOW_COUNT,
    candidate_catalog_binding,
    canonical_sha256,
    file_sha256,
    load_selection_manifest,
    validate_selection_manifest,
)
from pipeline.semantic_mentions import (
    MENTION_CATALOG_VERSION,
    MentionCandidate,
    generate_mention_candidates,
)
from pipeline.semantic_source import BronzeSource, window_from_exact_span


COVERAGE_SCHEMA_VERSION = "phase2j-candidate-coverage-v1"
COVERAGE_FILENAME = "candidate-coverage-v1.json"
RELEASE_GATE = "LOCKED"
CHECKPOINT = "CANDIDATE_COVERAGE_GATE"
ERROR_CODE = "CANDIDATE_GENERATION_MISS"

MISSING_NO_OVERLAPPING_CANDIDATE = "NO_OVERLAPPING_CANDIDATE"
MISSING_LONGER_SPAN_ONLY = "LONGER_SPAN_ONLY"
MISSING_SHORTER_FRAGMENT_ONLY = "SHORTER_FRAGMENT_ONLY"
MISSING_PARTIAL_OVERLAP_ONLY = "PARTIAL_OVERLAP_ONLY"
MISSING_MIXED_BOUNDARY_MISMATCH = "MIXED_BOUNDARY_MISMATCH"
MISSING_CATEGORIES = (
    MISSING_NO_OVERLAPPING_CANDIDATE,
    MISSING_LONGER_SPAN_ONLY,
    MISSING_SHORTER_FRAGMENT_ONLY,
    MISSING_PARTIAL_OVERLAP_ONLY,
    MISSING_MIXED_BOUNDARY_MISMATCH,
)

# Recursive forbidden-key validation for the machine-facing coverage artifact.
# The dedicated ``scoring_absence`` subtree is the one sanctioned place where
# the artifact may name scorer/model material, and it is exempted from the scan.
FORBIDDEN_KEYS = frozenset({
    "score", "scores", "probability", "probabilities", "confidence",
    "rank", "ranks", "ranked", "ranking", "rankings",
    "prediction", "predictions", "predicted", "predicted_label",
    "predicted_labels", "label", "labels", "gold_label", "gold_labels",
    "syntax_importance", "syntax_importances", "feature_importance",
    "feature_importances", "importance", "importances", "error_taxonomy",
    "model_suggestion", "model_suggestions", "suggestion", "suggestions",
    "model_id", "model_name", "model_score", "logits", "proba",
    "threshold", "thresholds",
})

_SHA256 = re.compile(r"[0-9a-f]{64}")
_EXPECTED_ENVELOPE_KEYS = frozenset({
    "content_sha256", "schema_version", "purpose", "release_gate",
    "checkpoint", "candidate_generator_version", "selection_manifest",
    "reviewed_packet", "scoring_absence", "coverage",
    "covered_endpoints", "missing_endpoints",
})
_EXPECTED_INPUT_BINDING_KEYS = frozenset({"file_sha256", "content_sha256"})
_EXPECTED_SCORING_ABSENCE_KEYS = frozenset({
    "model_scoring", "model_predictions", "thresholds", "statement",
})
_EXPECTED_COVERAGE_KEYS = frozenset({
    "aggregate", "total_candidates", "per_partition", "per_source_group",
    "per_node_type", "per_role", "per_window",
    "node_type_key_rule", "role_key_rule",
})
_EXPECTED_METRIC_KEYS = frozenset({"hit_count", "denominator", "rate"})
_EXPECTED_WINDOW_METRIC_KEYS = frozenset({
    "candidate_count", "gold_count", "hit_count", "rate",
})
_EXPECTED_COVERED_ENDPOINT_KEYS = frozenset({
    "endpoint_id", "window_id", "source_group_id", "partition", "role",
    "node_type", "char_start", "char_end", "absolute_start", "absolute_end",
    "bronze_text", "candidate_id", "candidate_alias", "candidate_window_id",
    "candidate_segment_ids", "candidate_catalog_sha256",
    "candidate_generator_version",
})
_EXPECTED_MISSING_ENDPOINT_KEYS = frozenset({
    "endpoint_id", "window_id", "source_group_id", "partition", "role",
    "node_type", "char_start", "char_end", "absolute_start", "absolute_end",
    "bronze_text", "error_code", "failure_category", "overlap_count",
    "overlaps",
})
_EXPECTED_OVERLAP_KEYS = frozenset({
    "candidate_id", "candidate_alias", "start", "end", "absolute_start",
    "absolute_end", "text",
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


def _validate_forbidden_content(value: object, *, path: tuple[str, ...] = ()) -> None:
    """Recursively reject scorer/model fields outside the absence declaration."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("phase2j candidate coverage keys must be strings")
            if key == "scoring_absence" and path == ():
                # The dedicated absence declaration is the one sanctioned place
                # where scorer/model material may be named.
                continue
            if key.casefold() in FORBIDDEN_KEYS:
                raise ValueError(
                    "phase2j candidate coverage contains forbidden key "
                    + repr(key) + " at " + ".".join(path + (key,)),
                )
            _validate_forbidden_content(item, path=path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_forbidden_content(item, path=path + (f"[{index}]",))


def classify_missing_span(
    gold_start: int,
    gold_end: int,
    candidate_spans: Iterable[tuple[int, int]],
) -> tuple[str | None, tuple[tuple[int, int], ...]]:
    """Classify one missing gold span against the regenerated candidate spans.

    Returns ``(category, overlapping_spans)``.  ``category`` is ``None`` when an
    exact candidate span exists (the endpoint is covered, not missing).  The
    overlap relationships are deterministic:

    * NO_OVERLAPPING_CANDIDATE - no candidate overlaps the gold span;
    * LONGER_SPAN_ONLY - every overlap strictly contains the gold span;
    * SHORTER_FRAGMENT_ONLY - every overlap lies strictly inside the gold span;
    * PARTIAL_OVERLAP_ONLY - every overlap crosses exactly one gold boundary
      without containment;
    * MIXED_BOUNDARY_MISMATCH - any other or mixed overlap relationship.
    """
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in (gold_start, gold_end)) or not 0 <= gold_start < gold_end:
        raise ValueError("phase2j gold span offsets are invalid")
    overlaps = tuple(sorted({
        (start, end)
        for start, end in candidate_spans
        if isinstance(start, int) and isinstance(end, int)
        and not isinstance(start, bool) and not isinstance(end, bool)
        and 0 <= start < end and start < gold_end and gold_start < end
    }))
    if not overlaps:
        return MISSING_NO_OVERLAPPING_CANDIDATE, overlaps
    relations: set[str] = set()
    for start, end in overlaps:
        if start == gold_start and end == gold_end:
            return None, overlaps
        if start <= gold_start and end >= gold_end:
            relations.add("LONGER")
        elif start >= gold_start and end <= gold_end:
            relations.add("SHORTER")
        elif start < gold_start < end < gold_end \
                or gold_start < start < gold_end < end:
            relations.add("PARTIAL")
        else:
            relations.add("MIXED")
    if relations == {"LONGER"}:
        return MISSING_LONGER_SPAN_ONLY, overlaps
    if relations == {"SHORTER"}:
        return MISSING_SHORTER_FRAGMENT_ONLY, overlaps
    if relations == {"PARTIAL"}:
        return MISSING_PARTIAL_OVERLAP_ONLY, overlaps
    return MISSING_MIXED_BOUNDARY_MISMATCH, overlaps


def _catalog(
    manifest_record: Mapping[str, Any],
) -> tuple[tuple[MentionCandidate, ...], list[dict[str, Any]], dict[str, Any]]:
    """Regenerate the frozen candidate catalog for one exact Bronze window.

    The catalog is generated with the frozen generator and the exact Bronze
    source/window contract used by ``candidate_catalog_binding`` and verified
    against that canonical binding.  Candidate rows are never scored or
    exposed to reviewers.
    """
    binding = candidate_catalog_binding(manifest_record)
    if binding["candidate_generator_version"] != MENTION_CATALOG_VERSION:
        raise ValueError("phase2j candidate generator version is unsupported")
    source_id = f"transcript:{manifest_record['upstream_source_id']}"
    text = manifest_record["source_text"]
    source = BronzeSource(source_id, text)
    window = window_from_exact_span(source, 0, len(text))
    candidates = generate_mention_candidates(window)
    aliases = tuple(f"C{index:04d}" for index in range(1, len(candidates) + 1))
    upstream_start = manifest_record["upstream_start"]
    catalog_records = [
        {
            "alias": alias,
            "candidate_id": item.candidate_id,
            "window_id": item.window_id,
            "start": item.start,
            "end": item.end,
            "absolute_start": upstream_start + item.start,
            "absolute_end": upstream_start + item.end,
            "text": item.source_text,
            "segment_ids": list(item.segment_ids),
        }
        for alias, item in zip(aliases, candidates)
    ]
    seen_spans: set[tuple[int, int]] = set()
    for candidate in candidates:
        span = (candidate.start, candidate.end)
        if span in seen_spans:
            raise ValueError(
                "phase2j regenerated candidate catalog contains duplicate "
                f"span {span} for {manifest_record['window_id']}",
            )
        seen_spans.add(span)
    rebuilt = {
        "candidate_generator_version": MENTION_CATALOG_VERSION,
        "candidate_count": len(catalog_records),
        "candidate_catalog_sha256": canonical_sha256(catalog_records),
    }
    if rebuilt != binding:
        raise ValueError(
            "phase2j regenerated catalog contradicts candidate_catalog_binding",
        )
    return candidates, catalog_records, rebuilt


def _covered_record(
    endpoint: Mapping[str, Any],
    *,
    candidate: MentionCandidate,
    catalog_record: Mapping[str, Any],
    manifest_record: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    upstream_start = manifest_record["upstream_start"]
    return {
        "endpoint_id": endpoint["endpoint_id"],
        "window_id": manifest_record["window_id"],
        "source_group_id": manifest_record["source_group_id"],
        "partition": manifest_record["partition"],
        "role": manifest_record["metadata"]["role"],
        "node_type": endpoint["node_type"],
        "char_start": endpoint["char_start"],
        "char_end": endpoint["char_end"],
        "absolute_start": upstream_start + endpoint["char_start"],
        "absolute_end": upstream_start + endpoint["char_end"],
        "bronze_text": endpoint["bronze_text"],
        "candidate_id": candidate.candidate_id,
        "candidate_alias": catalog_record["alias"],
        "candidate_window_id": candidate.window_id,
        "candidate_segment_ids": list(candidate.segment_ids),
        "candidate_catalog_sha256": binding["candidate_catalog_sha256"],
        "candidate_generator_version": MENTION_CATALOG_VERSION,
    }


def _missing_record(
    endpoint: Mapping[str, Any],
    *,
    overlapping: list[tuple[MentionCandidate, Mapping[str, Any]]],
    category: str,
    manifest_record: Mapping[str, Any],
) -> dict[str, Any]:
    upstream_start = manifest_record["upstream_start"]
    return {
        "endpoint_id": endpoint["endpoint_id"],
        "window_id": manifest_record["window_id"],
        "source_group_id": manifest_record["source_group_id"],
        "partition": manifest_record["partition"],
        "role": manifest_record["metadata"]["role"],
        "node_type": endpoint["node_type"],
        "char_start": endpoint["char_start"],
        "char_end": endpoint["char_end"],
        "absolute_start": upstream_start + endpoint["char_start"],
        "absolute_end": upstream_start + endpoint["char_end"],
        "bronze_text": endpoint["bronze_text"],
        "error_code": ERROR_CODE,
        "failure_category": category,
        "overlap_count": len(overlapping),
        "overlaps": [
            {
                "candidate_id": candidate.candidate_id,
                "candidate_alias": catalog_record["alias"],
                "start": candidate.start,
                "end": candidate.end,
                "absolute_start": upstream_start + candidate.start,
                "absolute_end": upstream_start + candidate.end,
                "text": candidate.source_text,
            }
            for candidate, catalog_record in overlapping
        ],
    }


def _match_window(
    manifest_record: Mapping[str, Any],
    packet_record: Mapping[str, Any],
    candidates: tuple[MentionCandidate, ...],
    catalog_records: list[dict[str, Any]],
    binding: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exact local Bronze span matching for one reviewed window."""
    by_span: dict[tuple[int, int], tuple[MentionCandidate, Mapping[str, Any]]] = {}
    for candidate, catalog_record in zip(candidates, catalog_records):
        span = (candidate.start, candidate.end)
        if span in by_span:
            raise ValueError(
                "phase2j regenerated candidate catalog contains duplicate "
                f"span {span} for {manifest_record['window_id']}",
            )
        by_span[span] = (candidate, catalog_record)
    covered: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for endpoint in packet_record["endpoints"]:
        char_start = endpoint["char_start"]
        char_end = endpoint["char_end"]
        match = by_span.get((char_start, char_end))
        if match is not None:
            candidate, catalog_record = match
            covered.append(_covered_record(
                endpoint,
                candidate=candidate,
                catalog_record=catalog_record,
                manifest_record=manifest_record,
                binding=binding,
            ))
            continue
        overlapping = sorted(
            (
                (candidate, catalog_record)
                for candidate, catalog_record in zip(candidates, catalog_records)
                if candidate.start < char_end and char_start < candidate.end
            ),
            key=lambda item: (item[0].start, item[0].end, item[0].candidate_id),
        )
        category, _ = classify_missing_span(
            char_start,
            char_end,
            ((candidate.start, candidate.end) for candidate, _ in overlapping),
        )
        if category is None:
            raise ValueError(
                "phase2j missing endpoint span is present in the regenerated "
                f"catalog for {manifest_record['window_id']}",
            )
        missing.append(_missing_record(
            endpoint,
            overlapping=overlapping,
            category=category,
            manifest_record=manifest_record,
        ))
    return covered, missing


def _metric(hit_count: int, denominator: int) -> dict[str, Any]:
    return {
        "hit_count": int(hit_count),
        "denominator": int(denominator),
        "rate": hit_count / denominator if denominator else 0.0,
    }


def build_candidate_coverage(
    *, manifest_path: Path, packet_path: Path,
) -> dict[str, Any]:
    """Strictly load both inputs and build the deterministic coverage artifact."""
    manifest = load_selection_manifest(manifest_path)
    packet = load_annotation_packet(packet_path, manifest=manifest)
    if len(packet["records"]) != TARGET_WINDOW_COUNT:
        raise ValueError("phase2j reviewed packet must contain exactly 30 windows")
    if not is_packet_gold_eligible(packet):
        raise ValueError("phase2j reviewed packet is not fully gold eligible")
    for record in packet["records"]:
        if record["window_status"] != "REVIEWED" or not is_window_gold_eligible(record):
            raise ValueError(
                f"phase2j reviewed window {record['window_id']} is not gold eligible",
            )
    manifest_by_window = {item["window_id"]: item for item in manifest["selected"]}
    packet_by_window = {record["window_id"]: record for record in packet["records"]}
    per_window: dict[str, Any] = {}
    partition_hit = {key: 0 for key in PARTITIONS}
    partition_denominator = {key: 0 for key in PARTITIONS}
    source_group_hit: dict[str, int] = {}
    source_group_denominator: dict[str, int] = {}
    node_type_hit: dict[str, int] = {}
    node_type_denominator: dict[str, int] = {}
    role_hit: dict[str, int] = {}
    role_denominator: dict[str, int] = {}
    covered_endpoints: list[dict[str, Any]] = []
    missing_endpoints: list[dict[str, Any]] = []
    total_candidates = 0
    for manifest_record in manifest["selected"]:
        window_id = manifest_record["window_id"]
        packet_record = packet_by_window[window_id]
        candidates, catalog_records, binding = _catalog(manifest_record)
        manifest_binding = manifest_by_window[window_id]
        if (manifest_binding["candidate_generator_version"]
                != binding["candidate_generator_version"]
                or manifest_binding["candidate_count"] != binding["candidate_count"]
                or manifest_binding["candidate_catalog_sha256"]
                != binding["candidate_catalog_sha256"]):
            raise ValueError(
                "phase2j regenerated catalog contradicts selection manifest "
                f"binding for {window_id}",
            )
        packet_binding = packet["candidate_catalog"]["per_window"][window_id]
        if (packet_binding["count"] != binding["candidate_count"]
                or packet_binding["catalog_sha256"]
                != binding["candidate_catalog_sha256"]):
            raise ValueError(
                "phase2j regenerated catalog contradicts reviewed packet "
                f"binding for {window_id}",
            )
        total_candidates += binding["candidate_count"]
        covered, missing = _match_window(
            manifest_record, packet_record, candidates, catalog_records, binding,
        )
        gold_count = len(packet_record["endpoints"])
        hit_count = len(covered)
        per_window[window_id] = {
            "candidate_count": binding["candidate_count"],
            "gold_count": gold_count,
            "hit_count": hit_count,
            "rate": hit_count / gold_count if gold_count else 0.0,
        }
        partition = manifest_record["partition"]
        partition_hit[partition] += hit_count
        partition_denominator[partition] += gold_count
        source_group = manifest_record["source_group_id"]
        source_group_hit[source_group] = source_group_hit.get(source_group, 0) + hit_count
        source_group_denominator[source_group] = (
            source_group_denominator.get(source_group, 0) + gold_count
        )
        covered_endpoints.extend(covered)
        missing_endpoints.extend(missing)
    for record in covered_endpoints:
        node_key = record["node_type"] if record["node_type"] is not None else "null"
        role_key = record["role"] if record["role"] else "none"
        node_type_hit[node_key] = node_type_hit.get(node_key, 0) + 1
        node_type_denominator[node_key] = node_type_denominator.get(node_key, 0) + 1
        role_hit[role_key] = role_hit.get(role_key, 0) + 1
        role_denominator[role_key] = role_denominator.get(role_key, 0) + 1
    for record in missing_endpoints:
        node_key = record["node_type"] if record["node_type"] is not None else "null"
        role_key = record["role"] if record["role"] else "none"
        node_type_denominator[node_key] = node_type_denominator.get(node_key, 0) + 1
        role_denominator[role_key] = role_denominator.get(role_key, 0) + 1
    aggregate_hit = len(covered_endpoints)
    aggregate_denominator = len(covered_endpoints) + len(missing_endpoints)
    if total_candidates != manifest["diversity_summary"]["candidate_count"]:
        raise ValueError(
            "phase2j regenerated catalog totals contradict the selection manifest",
        )
    if total_candidates != packet["candidate_catalog"]["count"]:
        raise ValueError(
            "phase2j regenerated catalog totals contradict the reviewed packet",
        )
    coverage = {
        "aggregate": _metric(aggregate_hit, aggregate_denominator),
        "total_candidates": int(total_candidates),
        "per_partition": {
            key: _metric(partition_hit[key], partition_denominator[key])
            for key in PARTITIONS
        },
        "per_source_group": {
            key: _metric(source_group_hit.get(key, 0), source_group_denominator[key])
            for key in sorted(source_group_denominator)
        },
        "per_node_type": {
            key: _metric(node_type_hit.get(key, 0), node_type_denominator[key])
            for key in sorted(node_type_denominator)
        },
        "per_role": {
            key: _metric(role_hit.get(key, 0), role_denominator[key])
            for key in sorted(role_denominator)
        },
        "per_window": per_window,
        "node_type_key_rule": (
            "node_type values are used as keys; the None node_type is keyed "
            "as 'null'."
        ),
        "role_key_rule": (
            "manifest metadata role values are used as keys; an empty role is "
            "keyed as 'none'."
        ),
    }
    inner = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "purpose": (
            "Phase 2J frozen candidate-coverage gate: exact local Bronze "
            "character span matching of reviewed gold endpoints against the "
            "frozen phase2f mention-candidate generator. Discovery coverage "
            "only; no model scoring, predictions, thresholds, ranks, labels, "
            "syntax features, or error taxonomy. The FROZEN_REPLICATION "
            "partition is reported only as coverage metadata and never as "
            "model performance."
        ),
        "release_gate": RELEASE_GATE,
        "checkpoint": CHECKPOINT,
        "candidate_generator_version": MENTION_CATALOG_VERSION,
        "selection_manifest": {
            "file_sha256": file_sha256(manifest_path),
            "content_sha256": manifest["content_sha256"],
        },
        "reviewed_packet": {
            "file_sha256": file_sha256(packet_path),
            "content_sha256": packet["content_sha256"],
        },
        "scoring_absence": {
            "model_scoring": "ABSENT",
            "model_predictions": "ABSENT",
            "thresholds": "ABSENT",
            "statement": (
                "Discovery coverage only. This artifact computes exact "
                "Bronze-span mention-candidate coverage for reviewed gold "
                "endpoints. It contains no model scoring, predictions, "
                "thresholds, ranks, labels, syntax features, or error "
                "taxonomy."
            ),
        },
        "coverage": coverage,
        "covered_endpoints": covered_endpoints,
        "missing_endpoints": missing_endpoints,
    }
    artifact = {"content_sha256": canonical_sha256(inner), **inner}
    validate_candidate_coverage(artifact, manifest=manifest, packet=packet)
    return artifact


def _validate_envelope(artifact: Mapping[str, Any]) -> None:
    if not isinstance(artifact, Mapping) or set(artifact) != _EXPECTED_ENVELOPE_KEYS:
        raise ValueError("phase2j candidate coverage envelope is invalid")
    if artifact["schema_version"] != COVERAGE_SCHEMA_VERSION:
        raise ValueError("phase2j candidate coverage version is unsupported")
    if artifact["release_gate"] != RELEASE_GATE:
        raise ValueError("phase2j candidate coverage release gate must remain LOCKED")
    if artifact["checkpoint"] != CHECKPOINT:
        raise ValueError("phase2j candidate coverage checkpoint is invalid")
    if artifact["candidate_generator_version"] != MENTION_CATALOG_VERSION:
        raise ValueError("phase2j candidate generator version is unsupported")
    if not isinstance(artifact["purpose"], str) or not artifact["purpose"]:
        raise ValueError("phase2j candidate coverage purpose is invalid")
    inner = {key: value for key, value in artifact.items() if key != "content_sha256"}
    if artifact["content_sha256"] != canonical_sha256(inner):
        raise ValueError("phase2j candidate coverage content hash is invalid")
    for label, binding in (
        ("selection manifest", artifact["selection_manifest"]),
        ("reviewed packet", artifact["reviewed_packet"]),
    ):
        if not isinstance(binding, Mapping) or set(binding) != _EXPECTED_INPUT_BINDING_KEYS:
            raise ValueError(f"phase2j {label} binding is invalid")
        if not _SHA256.fullmatch(binding["file_sha256"]) \
                or not _SHA256.fullmatch(binding["content_sha256"]):
            raise ValueError(f"phase2j {label} binding hashes are invalid")
    absence = artifact["scoring_absence"]
    if not isinstance(absence, Mapping) or set(absence) != _EXPECTED_SCORING_ABSENCE_KEYS:
        raise ValueError("phase2j scoring absence declaration is invalid")
    for key in ("model_scoring", "model_predictions", "thresholds"):
        if absence[key] != "ABSENT":
            raise ValueError(
                "phase2j candidate coverage must declare model scoring, "
                "predictions, and thresholds absent",
            )
    if not isinstance(absence["statement"], str) or not absence["statement"]:
        raise ValueError("phase2j scoring absence statement is invalid")
    _validate_forbidden_content(artifact)


def _validate_metric(value: object, *, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != _EXPECTED_METRIC_KEYS:
        raise ValueError(f"phase2j {label} metric is invalid")
    hit_count = value["hit_count"]
    denominator = value["denominator"]
    rate = value["rate"]
    if any(isinstance(item, bool) or not isinstance(item, int)
           for item in (hit_count, denominator)) \
            or hit_count < 0 or denominator < 0 or hit_count > denominator:
        raise ValueError(f"phase2j {label} metric counts are invalid")
    expected_rate = hit_count / denominator if denominator else 0.0
    if not isinstance(rate, float) or rate != expected_rate:
        raise ValueError(f"phase2j {label} metric rate is invalid")


def _validate_endpoint_identity(record: Mapping[str, Any]) -> None:
    if not isinstance(record["endpoint_id"], str) or not record["endpoint_id"] \
            or not isinstance(record["window_id"], str) or not record["window_id"] \
            or not isinstance(record["source_group_id"], str) \
            or not record["source_group_id"] \
            or record["partition"] not in PARTITION_SIZES \
            or not isinstance(record["role"], str):
        raise ValueError("phase2j coverage endpoint identity is invalid")
    char_start = record["char_start"]
    char_end = record["char_end"]
    absolute_start = record["absolute_start"]
    absolute_end = record["absolute_end"]
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in (char_start, char_end, absolute_start, absolute_end)) \
            or not 0 <= char_start < char_end \
            or absolute_end - absolute_start != char_end - char_start:
        raise ValueError("phase2j coverage endpoint offsets are invalid")
    if not isinstance(record["bronze_text"], str) or not record["bronze_text"]:
        raise ValueError("phase2j coverage endpoint text is invalid")
    if record["node_type"] is not None and record["node_type"] not in NODE_TYPES:
        raise ValueError("phase2j coverage endpoint node type is invalid")


def _validate_covered_record(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping) or set(record) != _EXPECTED_COVERED_ENDPOINT_KEYS:
        raise ValueError("phase2j covered endpoint record is invalid")
    _validate_endpoint_identity(record)
    if not isinstance(record["candidate_id"], str) or not record["candidate_id"] \
            or not isinstance(record["candidate_alias"], str) \
            or not record["candidate_alias"] \
            or not isinstance(record["candidate_window_id"], str) \
            or not record["candidate_window_id"] \
            or not isinstance(record["candidate_segment_ids"], list) \
            or any(not isinstance(item, str) for item in record["candidate_segment_ids"]) \
            or not _SHA256.fullmatch(record["candidate_catalog_sha256"]) \
            or record["candidate_generator_version"] != MENTION_CATALOG_VERSION:
        raise ValueError("phase2j covered endpoint provenance is invalid")


def _validate_missing_record(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping) or set(record) != _EXPECTED_MISSING_ENDPOINT_KEYS:
        raise ValueError("phase2j missing endpoint record is invalid")
    _validate_endpoint_identity(record)
    if record["error_code"] != ERROR_CODE \
            or record["failure_category"] not in MISSING_CATEGORIES:
        raise ValueError("phase2j missing endpoint classification is invalid")
    overlaps = record["overlaps"]
    if not isinstance(overlaps, list) \
            or isinstance(record["overlap_count"], bool) \
            or not isinstance(record["overlap_count"], int) \
            or record["overlap_count"] != len(overlaps):
        raise ValueError("phase2j missing endpoint overlap diagnostics are invalid")
    for overlap in overlaps:
        if not isinstance(overlap, Mapping) or set(overlap) != _EXPECTED_OVERLAP_KEYS:
            raise ValueError("phase2j missing endpoint overlap record is invalid")
        if not isinstance(overlap["candidate_id"], str) or not overlap["candidate_id"] \
                or not isinstance(overlap["candidate_alias"], str) \
                or not overlap["candidate_alias"] \
                or not isinstance(overlap["text"], str) or not overlap["text"]:
            raise ValueError("phase2j missing endpoint overlap values are invalid")
        start = overlap["start"]
        end = overlap["end"]
        absolute_start = overlap["absolute_start"]
        absolute_end = overlap["absolute_end"]
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (start, end, absolute_start, absolute_end)) \
                or not 0 <= start < end \
                or absolute_end - absolute_start != end - start:
            raise ValueError("phase2j missing endpoint overlap offsets are invalid")


def _validate_coverage(artifact: Mapping[str, Any]) -> None:
    coverage = artifact["coverage"]
    if not isinstance(coverage, Mapping) or set(coverage) != _EXPECTED_COVERAGE_KEYS:
        raise ValueError("phase2j candidate coverage section is invalid")
    covered = artifact["covered_endpoints"]
    missing = artifact["missing_endpoints"]
    if not isinstance(covered, list) or not isinstance(missing, list):
        raise ValueError("phase2j candidate coverage endpoint lists are invalid")
    for record in covered:
        _validate_covered_record(record)
    for record in missing:
        _validate_missing_record(record)
    endpoint_ids: set[str] = set()
    for record in covered + missing:
        endpoint_id = record["endpoint_id"]
        if endpoint_id in endpoint_ids:
            raise ValueError(
                f"phase2j coverage endpoint {endpoint_id} appears more than once",
            )
        endpoint_ids.add(endpoint_id)
    _validate_metric(coverage["aggregate"], label="aggregate")
    aggregate = coverage["aggregate"]
    if aggregate["hit_count"] != len(covered) \
            or aggregate["denominator"] != len(covered) + len(missing):
        raise ValueError(
            "phase2j aggregate coverage does not match the endpoint lists",
        )
    total_candidates = coverage["total_candidates"]
    if isinstance(total_candidates, bool) or not isinstance(total_candidates, int) \
            or total_candidates <= 0:
        raise ValueError("phase2j total candidate count is invalid")
    per_window = coverage["per_window"]
    if not isinstance(per_window, Mapping):
        raise ValueError("phase2j per-window coverage is invalid")
    window_hit: dict[str, int] = {}
    window_gold: dict[str, int] = {}
    partition_hit = {key: 0 for key in PARTITIONS}
    partition_denominator = {key: 0 for key in PARTITIONS}
    source_group_hit: dict[str, int] = {}
    source_group_denominator: dict[str, int] = {}
    node_type_hit: dict[str, int] = {}
    node_type_denominator: dict[str, int] = {}
    role_hit: dict[str, int] = {}
    role_denominator: dict[str, int] = {}
    for record in covered:
        window_id = record["window_id"]
        window_hit[window_id] = window_hit.get(window_id, 0) + 1
        window_gold[window_id] = window_gold.get(window_id, 0) + 1
        partition_hit[record["partition"]] += 1
        partition_denominator[record["partition"]] += 1
        source_group_hit[record["source_group_id"]] = (
            source_group_hit.get(record["source_group_id"], 0) + 1
        )
        source_group_denominator[record["source_group_id"]] = (
            source_group_denominator.get(record["source_group_id"], 0) + 1
        )
        node_key = record["node_type"] if record["node_type"] is not None else "null"
        role_key = record["role"] if record["role"] else "none"
        node_type_hit[node_key] = node_type_hit.get(node_key, 0) + 1
        node_type_denominator[node_key] = node_type_denominator.get(node_key, 0) + 1
        role_hit[role_key] = role_hit.get(role_key, 0) + 1
        role_denominator[role_key] = role_denominator.get(role_key, 0) + 1
    for record in missing:
        window_id = record["window_id"]
        window_gold[window_id] = window_gold.get(window_id, 0) + 1
        partition_denominator[record["partition"]] += 1
        source_group_denominator[record["source_group_id"]] = (
            source_group_denominator.get(record["source_group_id"], 0) + 1
        )
        node_key = record["node_type"] if record["node_type"] is not None else "null"
        role_key = record["role"] if record["role"] else "none"
        node_type_denominator[node_key] = node_type_denominator.get(node_key, 0) + 1
        role_denominator[role_key] = role_denominator.get(role_key, 0) + 1
    candidate_total = 0
    for window_id, metric in per_window.items():
        if not isinstance(metric, Mapping) \
                or set(metric) != _EXPECTED_WINDOW_METRIC_KEYS:
            raise ValueError("phase2j per-window metric is invalid")
        candidate_count = metric["candidate_count"]
        gold_count = metric["gold_count"]
        hit_count = metric["hit_count"]
        rate = metric["rate"]
        if any(isinstance(item, bool) or not isinstance(item, int)
               for item in (candidate_count, gold_count, hit_count)) \
                or candidate_count <= 0 or gold_count <= 0 \
                or hit_count < 0 or hit_count > gold_count:
            raise ValueError(f"phase2j per-window metric for {window_id} is invalid")
        if not isinstance(rate, float) or rate != (hit_count / gold_count):
            raise ValueError(f"phase2j per-window rate for {window_id} is invalid")
        if window_hit.get(window_id, 0) != hit_count \
                or window_gold.get(window_id, 0) != gold_count:
            raise ValueError(
                f"phase2j per-window metric contradicts endpoint lists for {window_id}",
            )
        candidate_total += candidate_count
    if set(window_gold) != set(per_window):
        raise ValueError("phase2j per-window coverage must cover every reviewed window")
    if candidate_total != total_candidates:
        raise ValueError("phase2j per-window candidate counts do not sum to the total")
    expected_partition = {
        key: _metric(partition_hit[key], partition_denominator[key])
        for key in PARTITIONS
    }
    if coverage["per_partition"] != expected_partition:
        raise ValueError("phase2j per-partition metrics are inconsistent")
    expected_groups = {
        key: _metric(source_group_hit.get(key, 0), source_group_denominator[key])
        for key in sorted(source_group_denominator)
    }
    if coverage["per_source_group"] != expected_groups:
        raise ValueError("phase2j per-source-group metrics are inconsistent")
    expected_node_types = {
        key: _metric(node_type_hit.get(key, 0), node_type_denominator[key])
        for key in sorted(node_type_denominator)
    }
    if coverage["per_node_type"] != expected_node_types:
        raise ValueError("phase2j per-node-type metrics are inconsistent")
    expected_roles = {
        key: _metric(role_hit.get(key, 0), role_denominator[key])
        for key in sorted(role_denominator)
    }
    if coverage["per_role"] != expected_roles:
        raise ValueError("phase2j per-role metrics are inconsistent")
    if not isinstance(coverage["node_type_key_rule"], str) \
            or not coverage["node_type_key_rule"] \
            or not isinstance(coverage["role_key_rule"], str) \
            or not coverage["role_key_rule"]:
        raise ValueError("phase2j coverage key rules are invalid")


def validate_candidate_coverage(
    artifact: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    packet: Mapping[str, Any] | None = None,
    manifest_path: Path | None = None,
    packet_path: Path | None = None,
) -> None:
    """Strictly validate the coverage artifact and its input bindings.

    Without inputs this checks the envelope, canonical content hash, absence
    declaration, metrics, and endpoint/provenance structure.  With ``manifest``
    and ``packet`` it additionally requires full gold eligibility and exact
    input bindings.  With both input paths it recomputes file hashes and
    rejects any mismatch against deterministic regeneration.
    """
    _validate_envelope(artifact)
    _validate_coverage(artifact)
    if (manifest is None) != (packet is None):
        raise ValueError("phase2j manifest and packet must be supplied together")
    if manifest is None:
        return
    validate_selection_manifest(manifest)
    validate_annotation_packet(packet, manifest=manifest)
    if len(packet["records"]) != TARGET_WINDOW_COUNT \
            or not is_packet_gold_eligible(packet):
        raise ValueError("phase2j reviewed packet is not fully gold eligible")
    for record in packet["records"]:
        if record["window_status"] != "REVIEWED" or not is_window_gold_eligible(record):
            raise ValueError(
                f"phase2j reviewed window {record['window_id']} is not gold eligible",
            )
    if artifact["selection_manifest"]["content_sha256"] != manifest["content_sha256"]:
        raise ValueError("phase2j coverage selection manifest binding is invalid")
    if artifact["reviewed_packet"]["content_sha256"] != packet["content_sha256"]:
        raise ValueError("phase2j coverage reviewed packet binding is invalid")
    if (manifest_path is None) != (packet_path is None):
        raise ValueError(
            "phase2j full coverage validation requires both input paths",
        )
    if manifest_path is None:
        return
    if artifact["selection_manifest"]["file_sha256"] != file_sha256(manifest_path):
        raise ValueError("phase2j coverage selection manifest file hash is invalid")
    if artifact["reviewed_packet"]["file_sha256"] != file_sha256(packet_path):
        raise ValueError("phase2j coverage reviewed packet file hash is invalid")
    fresh = build_candidate_coverage(
        manifest_path=manifest_path, packet_path=packet_path,
    )
    fresh_inner = {
        key: value for key, value in fresh.items() if key != "content_sha256"
    }
    inner = {key: value for key, value in artifact.items() if key != "content_sha256"}
    if fresh_inner != inner:
        raise ValueError(
            "phase2j candidate coverage does not match deterministic regeneration",
        )


def load_candidate_coverage(
    path: Path, *, manifest_path: Path | None = None,
    packet_path: Path | None = None,
) -> dict[str, Any]:
    """Strict canonical load with duplicate-key rejection and validation."""
    body = _load_json_strict(path, label="phase2j candidate coverage")
    if manifest_path is not None or packet_path is not None:
        if manifest_path is None or packet_path is None:
            raise ValueError(
                "phase2j candidate coverage validation requires both input paths",
            )
        manifest = load_selection_manifest(manifest_path)
        packet = load_annotation_packet(packet_path, manifest=manifest)
        validate_candidate_coverage(
            body, manifest=manifest, packet=packet,
            manifest_path=manifest_path, packet_path=packet_path,
        )
    else:
        validate_candidate_coverage(body)
    return body


def serialize_candidate_coverage(artifact: Mapping[str, Any]) -> str:
    """Deterministic canonical pretty JSON with a trailing newline."""
    return json.dumps(
        artifact, sort_keys=True, indent=2, ensure_ascii=False,
    ) + "\n"


def summarize_candidate_coverage(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic CLI summary of the coverage artifact and its inputs."""
    coverage = artifact["coverage"]
    return {
        "schema_version": artifact["schema_version"],
        "candidate_generator_version": artifact["candidate_generator_version"],
        "selection_manifest_sha256": artifact["selection_manifest"]["content_sha256"],
        "reviewed_packet_sha256": artifact["reviewed_packet"]["content_sha256"],
        "coverage_content_sha256": artifact["content_sha256"],
        "aggregate": coverage["aggregate"],
        "total_candidates": coverage["total_candidates"],
        "per_partition": coverage["per_partition"],
        "window_count": len(coverage["per_window"]),
        "covered_endpoint_count": len(artifact["covered_endpoints"]),
        "missing_endpoint_count": len(artifact["missing_endpoints"]),
    }


__all__ = [
    "CHECKPOINT", "COVERAGE_FILENAME", "COVERAGE_SCHEMA_VERSION", "ERROR_CODE",
    "FORBIDDEN_KEYS", "MISSING_CATEGORIES",
    "MISSING_LONGER_SPAN_ONLY", "MISSING_MIXED_BOUNDARY_MISMATCH",
    "MISSING_NO_OVERLAPPING_CANDIDATE", "MISSING_PARTIAL_OVERLAP_ONLY",
    "MISSING_SHORTER_FRAGMENT_ONLY", "RELEASE_GATE",
    "build_candidate_coverage", "classify_missing_span",
    "load_candidate_coverage", "serialize_candidate_coverage",
    "summarize_candidate_coverage", "validate_candidate_coverage",
]
