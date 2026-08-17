"""Phase 2G controlled three-condition endpoint-recovery ablation.

The experiment reuses ``pipeline.semantic_mentions.generate_mention_candidates``
as the complete deterministic candidate universe for each locked Phase 2F
legacy case, assigns compact immutable aliases ``C0001``... in sorted catalog
order, and asks the model one case-level question per condition:

* ``RAW_BRONZE`` -- the immutable exact bronze window text;
* ``MECHANICAL_SILVER`` -- the reversible linguistic cleanup from
  ``pipeline.phase2g_silver``;
* ``RESOLVED_SILVER`` -- mechanical plus high-confidence linguistic reference
  resolution.

Every existing reviewed mention question becomes a separate endpoint task with
an opaque task ID inside the one case-level request.  Candidate aliases, exact
Bronze offsets, and authoritative Bronze text are identical across all three
conditions; the model never supplies source text or offsets as authority.
Each existing benchmark reference gets one separate status judgment
(gold ``INSUFFICIENT_EVIDENCE`` maps to ``UNKNOWN`` for this interface).

The parser accepts strict JSON or exactly one complete JSON Markdown fence,
rejects duplicate JSON keys, missing/unknown keys at every level, wrong task-ID
sets, empty role arrays, duplicate aliases within a role, wrong
types/statuses/roles, and status/selection inconsistencies, and retains
invented candidate IDs as diagnostic evidence.  Deterministic candidate
resolution owns every Bronze value.  Endpoint status ``NONE`` means the mention
is unambiguous and requires at least one selected candidate; ``UNKNOWN`` and
``AMBIGUOUS`` require no endpoint candidates.  Reference ``NONE`` is
unambiguous and may carry targets; reference ``UNKNOWN``/``AMBIGUOUS`` must
have no targets.

The model-facing catalog is rendered compactly as an immutable alias ->
authoritative exact Bronze text mapping (offsets omitted from the prompt to
keep it bounded); the retained input representation and every per-case
condition artifact always include the full catalog with the Phase 2F candidate
IDs, window-local and upstream-Bronze absolute offsets, exact Bronze text, and
segment provenance, plus the expected endpoint/status task definitions.

Metrics use explicit hit/denominator/rate records; endpoint precision is
defined over task-scoped endpoint candidate assignments, role accuracy is
task-level over recalled endpoints, and the unsupported/invented rate counts
endpoint assignments plus reference-target selections.  Every failed endpoint
receives exactly one documented first-failure classification, and a promotion
gate may pass if at least one condition satisfies every threshold.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Mapping

from pipeline.phase2g_silver import (
    CONDITIONS,
    MECHANICAL_SILVER,
    RAW_BRONZE,
    RESOLVED_SILVER,
    BENCHMARK_CONTENT_SHA256 as SILVER_BENCHMARK_SHA256,
    SILVER_FIXTURE_CONTENT_SHA256,
    canonical_sha256,
    condition_text,
    load_silver_fixture,
    silver_input_record,
    validate_fixture_against_benchmark,
)
from pipeline.semantic_mentions import generate_mention_candidates
from pipeline.semantic_source import BronzeSource, window_from_exact_span


BENCHMARK_CONTENT_SHA256 = "a17674b6e2c491f0d7a1600dde0cfb8cc533d1d17db8633d8d94b2de9a57c1dd"
RUN_VERSION = "phase2g-endpoint-recovery-v1"
GATE_VERSION = "phase2g-promotion-gate-v1"

REFERENCE_MODEL = "deepseek-v4-pro"
REFERENCE_THINKING = "disabled"
REFERENCE_ENDPOINT = "https://api.deepseek.com"
REFERENCE_TEMPERATURE = 0.0
REFERENCE_MAX_TOKENS = 4096

NODE_TYPES = frozenset({
    "ENTITY", "ABILITY_OR_RESOURCE", "EVENT", "ACTION", "STATE", "OUTCOME",
    "QUANTITY", "TIME", "LOCATION_OR_SPACE",
})
REFERENCE_STATUSES = frozenset({"NONE", "UNKNOWN", "AMBIGUOUS"})

FAILURE_CODES = frozenset({
    "CANDIDATE_MISSING", "WRONG_CANDIDATE_SELECTED", "RIGHT_CANDIDATE_WRONG_ROLE",
    "REFERENCE_UNRESOLVED", "SOURCE_AMBIGUOUS", "MODEL_ABSTAINED",
    "MODEL_INVENTED", "PARSER_FAILURE", "OTHER",
})

# Documented first-failure precedence: the first applicable code in this list
# is the endpoint's single classification.
FAILURE_PRECEDENCE = (
    "PARSER_FAILURE",
    "CANDIDATE_MISSING",
    "RIGHT_CANDIDATE_WRONG_ROLE",
    "MODEL_INVENTED",
    "MODEL_ABSTAINED",
    "SOURCE_AMBIGUOUS",
    "REFERENCE_UNRESOLVED",
    "WRONG_CANDIDATE_SELECTED",
    "OTHER",
)

GATE_THRESHOLDS = {
    "candidate_coverage": 1.0,
    "endpoint_recall": 0.90,
    "endpoint_precision": 0.90,
    "role_accuracy": 0.85,
    "unsupported_or_invented_rate": 0.05,
    "source_alignment_violations": 0,
}

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class Phase2GParseError(ValueError):
    """The model response violated the strict Phase 2G JSON contract."""


class Phase2GCoverageError(ValueError):
    """Deterministic candidate coverage validation failed."""


def _metric(hit: int, denominator: int) -> dict[str, Any]:
    return {
        "hit_count": int(hit),
        "denominator": int(denominator),
        "rate": hit / denominator if denominator else None,
    }


def _sum_metrics(items: list[Mapping[str, Any]]) -> dict[str, Any]:
    hits = sum(item["hit_count"] for item in items)
    denom = sum(item["denominator"] for item in items)
    return _metric(hits, denom)


def load_benchmark(path: str | Path) -> Mapping[str, Any]:
    """Load the locked Phase 2F legacy benchmark and verify its content lock."""
    path = Path(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, Mapping):
        raise ValueError("benchmark must be a JSON object")
    if body.get("content_sha256") != BENCHMARK_CONTENT_SHA256:
        raise ValueError("benchmark content does not match the preregistered lock")
    if body.get("split") != "LEGACY_FAILURE" or not isinstance(body.get("cases"), list) \
            or len(body["cases"]) != 5:
        raise ValueError("benchmark is not the five-case LEGACY_FAILURE split")
    return body


def build_case_experiment(case: Mapping[str, Any]) -> dict[str, Any]:
    """Build the deterministic per-case experiment (catalog, aliases, tasks)."""
    source = BronzeSource(case["source_id"], case["source_text"])
    window = window_from_exact_span(source, 0, len(source.text))
    catalog = tuple(generate_mention_candidates(window))
    aliases = tuple(f"C{index:04d}" for index in range(1, len(catalog) + 1))
    upstream_start = case["upstream_start"]
    catalog_records = []
    for alias, item in zip(aliases, catalog):
        catalog_records.append({
            "alias": alias,
            "candidate_id": item.candidate_id,
            "window_id": item.window_id,
            "start": item.start,
            "end": item.end,
            "absolute_start": upstream_start + item.start,
            "absolute_end": upstream_start + item.end,
            "text": item.source_text,
            "segment_ids": list(item.segment_ids),
        })
    validate_catalog_records(catalog_records, case["source_text"], upstream_start)
    alias_to_record = {record["alias"]: record for record in catalog_records}
    catalog_spans = {(record["start"], record["end"]) for record in catalog_records}

    endpoint_tasks = []
    status_tasks = []
    for question in case["questions"]:
        required = question.get("requires")
        if not isinstance(required, list) or len(required) != 1:
            continue
        requirement = required[0]
        if requirement.startswith("mention:"):
            mention_id = requirement[len("mention:"):]
            mention = next(item for item in case["mentions"] if item["id"] == mention_id)
            endpoint_tasks.append({
                "task_id": f"ep-{len(endpoint_tasks) + 1:02d}",
                "question_id": question["id"],
                "prompt": question["prompt"],
                "gold_mention_id": mention_id,
                "gold_spans": [tuple(span) for span in mention["acceptable_spans"]],
                "gold_node_types": tuple(mention["node_types"]),
            })
        elif requirement.startswith("reference:"):
            reference_id = requirement[len("reference:"):]
            reference = next(item for item in case["references"] if item["id"] == reference_id)
            status_tasks.append({
                "task_id": f"st-{len(status_tasks) + 1:02d}",
                "question_id": question["id"],
                "prompt": question["prompt"],
                "gold_reference_id": reference_id,
                "gold_status": "UNKNOWN",
                "gold_source": reference["source"],
                "gold_target_spans": [
                    list(span) for span in reference.get("targets", [])
                    if isinstance(span, (list, tuple)) and len(span) == 2
                ],
            })

    missing = [
        task["gold_mention_id"]
        for task in endpoint_tasks
        if not any(tuple(span) in catalog_spans for span in task["gold_spans"])
    ]
    if missing:
        raise Phase2GCoverageError(
            f"deterministic mention catalog misses gold endpoints: {missing}",
        )

    return {
        "case_id": case["id"],
        "source_id": case["source_id"],
        "bronze_text": case["source_text"],
        "bronze_text_sha256": hashlib.sha256(case["source_text"].encode()).hexdigest(),
        "upstream_start": upstream_start,
        "catalog": catalog_records,
        "catalog_sha256": canonical_sha256(catalog_records),
        "alias_to_record": alias_to_record,
        "endpoint_tasks": endpoint_tasks,
        "status_tasks": status_tasks,
        "expected_endpoint_count": len(endpoint_tasks),
        "expected_status_count": len(status_tasks),
    }


def validate_catalog_records(
    records: list[Mapping[str, Any]], bronze_text: str, upstream_start: int,
) -> None:
    """Validate every catalog field against the authoritative Bronze window."""
    if not isinstance(records, list) or not records:
        raise Phase2GCoverageError("candidate catalog must be a nonempty list")
    aliases: list[str] = []
    candidate_ids: list[str] = []
    spans: list[tuple[int, int]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise Phase2GCoverageError("candidate record must be an object")
        alias = record.get("alias")
        candidate_id = record.get("candidate_id")
        window_id = record.get("window_id")
        start, end = record.get("start"), record.get("end")
        absolute_start, absolute_end = (
            record.get("absolute_start"), record.get("absolute_end"),
        )
        text = record.get("text")
        segment_ids = record.get("segment_ids")
        if not isinstance(alias, str) or not alias:
            raise Phase2GCoverageError("candidate alias must be a nonempty string")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise Phase2GCoverageError("candidate id must be a nonempty string")
        if not isinstance(window_id, str) or not window_id:
            raise Phase2GCoverageError("candidate window id must be a nonempty string")
        if not candidate_id.startswith(window_id + ":m"):
            raise Phase2GCoverageError(
                f"candidate id {candidate_id!r} is not bound to its window",
            )
        if (
            isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)
            or not 0 <= start < end <= len(bronze_text)
        ):
            raise Phase2GCoverageError("candidate local offsets are invalid")
        if (
            isinstance(absolute_start, bool) or isinstance(absolute_end, bool)
            or not isinstance(absolute_start, int) or not isinstance(absolute_end, int)
            or absolute_start != upstream_start + start
            or absolute_end != upstream_start + end
        ):
            raise Phase2GCoverageError(
                "candidate absolute offsets do not match upstream bronze",
            )
        if not isinstance(text, str) or bronze_text[start:end] != text:
            raise Phase2GCoverageError("candidate text is not the exact bronze slice")
        if (
            not isinstance(segment_ids, list) or not segment_ids
            or any(not isinstance(item, str) or not item for item in segment_ids)
        ):
            raise Phase2GCoverageError(
                "candidate segment provenance must be a nonempty id list",
            )
        aliases.append(alias)
        candidate_ids.append(candidate_id)
        spans.append((start, end))
    if len(set(aliases)) != len(aliases):
        raise Phase2GCoverageError("candidate aliases must be unique")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise Phase2GCoverageError("candidate ids must be unique")
    if len(set(spans)) != len(spans):
        raise Phase2GCoverageError("candidate spans must be unique")


def validate_experiment_coverage(experiments: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Validate 33/33 reviewed exact endpoint coverage before any provider call."""
    total = 0
    covered = 0
    per_case = {}
    for case_id, experiment in experiments.items():
        case_total = experiment["expected_endpoint_count"]
        case_covered = sum(
            any(tuple(span) in {
                (record["start"], record["end"]) for record in experiment["catalog"]
            } for span in task["gold_spans"])
            for task in experiment["endpoint_tasks"]
        )
        total += case_total
        covered += case_covered
        per_case[case_id] = _metric(case_covered, case_total)
    if covered != 33 or total != 33:
        raise Phase2GCoverageError(
            f"reviewed exact endpoint coverage is {covered}/{total}, expected 33/33",
        )
    return {
        "candidate_coverage": _metric(covered, total),
        "per_case": per_case,
    }


def _catalog_prompt(catalog: list[Mapping[str, Any]]) -> str:
    # Offsets disambiguate repeated literal strings (for example the several
    # distinct ``you`` spans in one window) without asking the model to
    # reproduce either text or offsets.  Compact tuples keep the complete
    # candidate universe inside the reference-model context.
    values = [
        [record["alias"], record["start"], record["end"], record["text"]]
        for record in catalog
    ]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


ENDPOINT_RECOVERY_SYSTEM = (
    "You recover exact low-level source-semantic mentions from a transcript. "
    "Return strict JSON only: one object, no prose, and no Markdown unless the "
    "entire response is exactly one JSON fenced block. Use only the supplied "
    "candidate aliases (C####), the supplied roles, and the supplied statuses. "
    "Never invent aliases, source text, offsets, strategic concepts, causal "
    "edges, or graphs. Each CANDIDATE CATALOG row is [alias,start,end,text] "
    "and maps an alias to its exact authoritative Bronze span; text is bronze[start:end] for the exact "
    "half-open character span retained in the artifact. Roles are exactly: "
    "ENTITY, ABILITY_OR_RESOURCE, EVENT, "
    "ACTION, STATE, OUTCOME, QUANTITY, TIME, LOCATION_OR_SPACE. Statuses are "
    "exactly: NONE, UNKNOWN, AMBIGUOUS. Select the smallest complete source "
    "spans; do not use a long clause as a proxy for meaning that begins "
    "elsewhere. Multiple candidates per role are allowed. Preserve grammatical "
    "pronouns as mentions; do not rewrite the source. If a task is uncertain, "
    "say so with its status rather than guessing. Endpoint status NONE means "
    "the mention is unambiguous and requires at least one selected candidate; "
    "UNKNOWN and AMBIGUOUS require no endpoint candidates. Reference status "
    "NONE is unambiguous and may carry target candidates; UNKNOWN and "
    "AMBIGUOUS must have no targets."
)


def build_request(
    experiment: Mapping[str, Any],
    text: str,
    *,
    condition: str,
    model: str = REFERENCE_MODEL,
    thinking: str = REFERENCE_THINKING,
    max_tokens: int = REFERENCE_MAX_TOKENS,
) -> dict[str, Any]:
    """Build the one case-level request for a condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition!r}")
    if model != REFERENCE_MODEL:
        raise ValueError("Phase 2G uses the preregistered reference model only")
    if thinking != REFERENCE_THINKING:
        raise ValueError("Phase 2G disables thinking for the reference run")
    lines = [
        f"CONDITION: {condition}",
        "",
        "CONDITION TEXT:",
        text,
        "",
        "AUTHORITATIVE BRONZE TEXT (the catalog text is exactly bronze[start:end] "
        "for the retained local and absolute offsets; keep it fixed):",
        experiment["bronze_text"],
        "",
        "CANDIDATE CATALOG (complete rows [alias,start,end,exact authoritative bronze text]):",
        _catalog_prompt(experiment["catalog"]),
        "",
        f"ENDPOINT TASKS ({len(experiment['endpoint_tasks'])} tasks):",
    ]
    for task in experiment["endpoint_tasks"]:
        lines.append(f"TASK {task['task_id']}: {task['prompt']}")
    lines.append("")
    lines.append(f"REFERENCE STATUS TASKS ({len(experiment['status_tasks'])} tasks):")
    for task in experiment["status_tasks"]:
        lines.append(f"TASK {task['task_id']}: {task['prompt']}")
    lines.extend([
        "",
        "Return exactly one JSON object of this shape:",
        '{"endpoint_selections": {'
        '"<task_id>": {"roles": {"<ROLE>": ["C####", ...], ...}, "status": "NONE|UNKNOWN|AMBIGUOUS"}'
        ", ...}, "
        '"reference_statuses": {'
        '"<task_id>": {"status": "NONE|UNKNOWN|AMBIGUOUS", "targets": ["C####", ...]}'
        ", ...}}",
        "Every endpoint task_id and reference task_id must appear. For an endpoint task, "
        "roles with no candidates may be omitted; status NONE means the mention exists and "
        "was selected, and NONE is valid only with at least one selected candidate. "
        "UNKNOWN means the mention cannot be determined; AMBIGUOUS means the source is "
        "ambiguous; UNKNOWN and AMBIGUOUS must have no endpoint candidates. Reference "
        "status NONE is unambiguous and may carry target candidates; UNKNOWN and "
        "AMBIGUOUS must have no target IDs.",
    ])
    user = "\n".join(lines)
    request_payload = {
        "system": ENDPOINT_RECOVERY_SYSTEM,
        "user": user,
        "model": model,
        "temperature": REFERENCE_TEMPERATURE,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "provider_endpoint": REFERENCE_ENDPOINT,
    }
    request_json = json.dumps(
        request_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return {
        "condition": condition,
        "system": ENDPOINT_RECOVERY_SYSTEM,
        "user": user,
        "request_json": request_json,
        "request_sha256": hashlib.sha256(request_json.encode("utf-8")).hexdigest(),
        "model": model,
        "thinking": thinking,
        "temperature": REFERENCE_TEMPERATURE,
        "max_tokens": max_tokens,
        "provider_endpoint": REFERENCE_ENDPOINT,
    }


def _unique_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """JSON object hook that rejects duplicate keys at any nesting level."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Phase2GParseError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_model_response(raw: str, experiment: Mapping[str, Any]) -> Mapping[str, Any]:
    """Strictly parse a Phase 2G model response.

    Accepts strict JSON or exactly one complete JSON Markdown fence.  Rejects
    duplicate JSON keys at any level, missing/unknown root and entry keys,
    missing/unknown endpoint and reference task IDs, empty role arrays,
    duplicate aliases within one role, unknown roles/statuses, wrong types, and
    status/selection inconsistencies (endpoint NONE without candidates,
    endpoint UNKNOWN/AMBIGUOUS with candidates, reference UNKNOWN/AMBIGUOUS
    with targets).  Invented candidate aliases are *not* rejected here; they
    are retained as diagnostic evidence by resolution.
    """
    if not isinstance(raw, str):
        raise Phase2GParseError("model response must be text")
    text = raw.strip()
    if not text:
        raise Phase2GParseError("empty model response")
    if text.startswith("```"):
        match = _FENCE.fullmatch(text)
        if match is None:
            raise Phase2GParseError("expected exactly one complete JSON markdown fence")
        body = match.group(1)
    else:
        if "```" in text:
            raise Phase2GParseError("markdown fence inside non-fenced response")
        body = text
    try:
        parsed = json.loads(body, object_pairs_hook=_unique_object_pairs)
    except json.JSONDecodeError as exc:
        raise Phase2GParseError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, Mapping):
        raise Phase2GParseError("response root must be a JSON object")
    _require_exact_keys(parsed, {"endpoint_selections", "reference_statuses"}, "root")

    endpoint_tasks = {task["task_id"]: task for task in experiment["endpoint_tasks"]}
    status_tasks = {task["task_id"]: task for task in experiment["status_tasks"]}
    endpoint_selections = parsed["endpoint_selections"]
    reference_statuses = parsed["reference_statuses"]
    if not isinstance(endpoint_selections, Mapping) or not isinstance(reference_statuses, Mapping):
        raise Phase2GParseError("endpoint_selections/reference_statuses must be objects")
    _require_exact_keys(endpoint_selections, set(endpoint_tasks), "endpoint_selections")
    _require_exact_keys(reference_statuses, set(status_tasks), "reference_statuses")
    for task_id, entry in endpoint_selections.items():
        if task_id not in endpoint_tasks:
            raise Phase2GParseError(f"unknown endpoint task id {task_id!r}")
        if not isinstance(entry, Mapping):
            raise Phase2GParseError(f"endpoint task {task_id} must be an object")
        _require_exact_keys(entry, {"roles", "status"}, f"endpoint task {task_id}")
        if entry["status"] not in REFERENCE_STATUSES:
            raise Phase2GParseError(
                f"endpoint task {task_id} has unknown status {entry['status']!r}",
            )
        roles = entry["roles"]
        if not isinstance(roles, Mapping):
            raise Phase2GParseError(f"endpoint task {task_id} roles must be an object")
        total_selected = 0
        for role, aliases in roles.items():
            if role not in NODE_TYPES:
                raise Phase2GParseError(
                    f"endpoint task {task_id} has unknown role {role!r}",
                )
            if not isinstance(aliases, list) or not aliases:
                raise Phase2GParseError(
                    f"endpoint task {task_id} role {role} must map to a nonempty list of alias strings",
                )
            if any(not isinstance(alias, str) or not alias for alias in aliases):
                raise Phase2GParseError(
                    f"endpoint task {task_id} role {role} must contain nonempty alias strings",
                )
            if len(set(aliases)) != len(aliases):
                raise Phase2GParseError(
                    f"endpoint task {task_id} role {role} contains duplicate aliases",
                )
            total_selected += len(aliases)
        if entry["status"] == "NONE" and total_selected == 0:
            raise Phase2GParseError(
                f"endpoint task {task_id} status NONE requires at least one selected candidate",
            )
        if entry["status"] in ("UNKNOWN", "AMBIGUOUS") and total_selected:
            raise Phase2GParseError(
                f"endpoint task {task_id} status {entry['status']} requires no endpoint candidates",
            )
    for task_id, entry in reference_statuses.items():
        if task_id not in status_tasks:
            raise Phase2GParseError(f"unknown reference status task id {task_id!r}")
        if not isinstance(entry, Mapping):
            raise Phase2GParseError(f"reference task {task_id} must be an object")
        _require_exact_keys(entry, {"status", "targets"}, f"reference task {task_id}")
        if entry["status"] not in REFERENCE_STATUSES:
            raise Phase2GParseError(
                f"reference task {task_id} has unknown status {entry['status']!r}",
            )
        targets = entry["targets"]
        if not isinstance(targets, list) or any(
            not isinstance(target, str) or not target for target in targets
        ):
            raise Phase2GParseError(
                f"reference task {task_id} targets must be a list of alias strings",
            )
        if entry["status"] in ("UNKNOWN", "AMBIGUOUS") and targets:
            raise Phase2GParseError(
                f"reference task {task_id} status {entry['status']} requires no target IDs",
            )
    return parsed


def _require_exact_keys(
    value: Mapping[str, Any], allowed: set[str], label: str,
) -> None:
    allowed = set(allowed)
    unknown = set(value) - allowed
    missing = allowed - set(value)
    if not unknown and not missing:
        return
    parts = []
    if missing:
        parts.append(f"missing keys {sorted(missing)}")
    if unknown:
        parts.append(f"unknown keys {sorted(unknown)}")
    raise Phase2GParseError(
        f"{label} keys must be exactly {sorted(allowed)} ({'; '.join(parts)})",
    )


def resolve_parsed_payload(
    experiment: Mapping[str, Any], parsed: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Resolve parsed selections to deterministic Bronze values.

    Invented/unknown candidate IDs are retained as diagnostic evidence with
    ``known=False`` and no source authority.
    """
    alias_to_record = experiment["alias_to_record"]
    endpoint_resolutions = {}
    for task in experiment["endpoint_tasks"]:
        task_id = task["task_id"]
        entry = (parsed or {}).get("endpoint_selections", {}).get(task_id)
        assignments = []
        if entry is not None:
            for role, aliases in entry["roles"].items():
                for alias in aliases:
                    record = alias_to_record.get(alias)
                    assignments.append({
                        "alias": alias,
                        "role": role,
                        "known": record is not None,
                        "start": record["start"] if record else None,
                        "end": record["end"] if record else None,
                        "text": record["text"] if record else None,
                    })
        endpoint_resolutions[task_id] = {
            "status": entry["status"] if entry is not None else None,
            "assignments": assignments,
        }
    status_resolutions = {}
    for task in experiment["status_tasks"]:
        task_id = task["task_id"]
        entry = (parsed or {}).get("reference_statuses", {}).get(task_id)
        status_resolutions[task_id] = {
            "status": entry["status"] if entry is not None else None,
            "targets": [
                {
                    "alias": alias,
                    "known": alias in alias_to_record,
                    "start": alias_to_record[alias]["start"] if alias in alias_to_record else None,
                    "end": alias_to_record[alias]["end"] if alias in alias_to_record else None,
                    "text": alias_to_record[alias]["text"] if alias in alias_to_record else None,
                }
                for alias in (entry["targets"] if entry is not None else [])
            ],
        }
    return {
        "endpoints": endpoint_resolutions,
        "statuses": status_resolutions,
    }


def classify_endpoint(
    task: Mapping[str, Any],
    resolution: Mapping[str, Any],
    *,
    parser_failed: bool,
    gold_in_catalog: bool,
) -> tuple[bool, str | None, str]:
    """Return (full_correct, classification_code, detail).

    Full correctness requires: the endpoint is recalled (at least one exact-span
    assignment), every selected alias is known, no extra known wrong-span
    candidate is selected, and every correct-span assignment uses a gold-allowed
    role.  Every not-fully-correct endpoint receives exactly one classification
    using the documented precedence: parser failure, missing gold, wrong role on
    a right candidate, invented IDs, abstention, source ambiguity, unresolved
    reference, then wrong/extra candidate selection.
    """
    gold_spans = set(task["gold_spans"])
    gold_types = set(task["gold_node_types"])
    assignments = resolution["assignments"]
    correct = [a for a in assignments if a["known"] and (a["start"], a["end"]) in gold_spans]
    invented = [a for a in assignments if not a["known"]]
    wrong_known = [
        a for a in assignments
        if a["known"] and (a["start"], a["end"]) not in gold_spans
    ]
    recalled = bool(correct)
    role_correct = bool(correct) and all(a["role"] in gold_types for a in correct)
    status = resolution.get("status")
    full_correct = recalled and role_correct and not invented and not wrong_known
    if full_correct:
        return True, None, ""
    if parser_failed:
        return False, "PARSER_FAILURE", "case response failed strict parsing"
    if not gold_in_catalog:
        return False, "CANDIDATE_MISSING", "gold endpoint absent from deterministic catalog"
    if recalled and not role_correct:
        return False, "RIGHT_CANDIDATE_WRONG_ROLE", (
            "gold-span candidate selected without a gold-allowed node type",
        )
    if invented:
        return False, "MODEL_INVENTED", (
            "model selected unknown candidate IDs: "
            + ",".join(sorted({a["alias"] for a in invented}))
        )
    if recalled and wrong_known:
        return False, "WRONG_CANDIDATE_SELECTED", (
            "right span selected together with wrong-span known candidates: "
            + ",".join(sorted({
                a["alias"] for a in wrong_known
            }))
        )
    if not assignments and status in (None, "NONE"):
        return False, "MODEL_ABSTAINED", "no candidate assignments for the task"
    if not recalled and status == "AMBIGUOUS":
        return False, "SOURCE_AMBIGUOUS", "model reported an ambiguous source"
    if not recalled and status == "UNKNOWN":
        return False, "REFERENCE_UNRESOLVED", "model could not resolve the mention"
    if not recalled:
        return False, "WRONG_CANDIDATE_SELECTED", (
            "selected candidates do not cover the gold endpoint"
        )
    return False, "OTHER", "unclassified endpoint failure"


def evaluate_case(
    experiment: Mapping[str, Any],
    resolution: Mapping[str, Any],
    *,
    parser_failed: bool,
    provider_failure: str | None = None,
) -> dict[str, Any]:
    """Compute the per-case metrics, failures, and expected-vs-selected report.

    Candidate coverage is recomputed from the actual catalog spans of this
    case.  Endpoint recall is task-level exact-span hits regardless of role;
    endpoint precision is exact-span assignments over all endpoint candidate
    assignments; role accuracy is task-level over recalled endpoints (every
    correct-span assignment must use a gold-allowed role).  Unsupported and
    invented IDs are counted across endpoint assignments and reference-target
    selections; source alignment validates every known selected endpoint and
    reference candidate against the exact Bronze text.
    """
    catalog_spans = {
        (record["start"], record["end"]) for record in experiment["catalog"]
    }
    endpoint_details = []
    first_failures = []
    failure_counts = {code: 0 for code in FAILURE_CODES}
    recalled_tasks = 0
    role_correct_tasks = 0
    all_assignments = 0
    correct_assignments = 0
    known_assignments = 0
    unsupported = 0
    invented = 0
    alignment_violations = 0
    endpoint_tasks = experiment["endpoint_tasks"]
    covered_endpoints = sum(
        1 for task in endpoint_tasks
        if any(tuple(span) in catalog_spans for span in task["gold_spans"])
    )
    for task in endpoint_tasks:
        task_resolution = resolution["endpoints"][task["task_id"]]
        gold_in_catalog = any(
            tuple(span) in catalog_spans for span in task["gold_spans"]
        )
        if provider_failure is not None:
            full_correct, code, detail = (
                False, "OTHER", f"PROVIDER_FAILURE: {provider_failure}",
            )
        else:
            full_correct, code, detail = classify_endpoint(
                task, task_resolution,
                parser_failed=parser_failed, gold_in_catalog=gold_in_catalog,
            )
        gold_spans = set(task["gold_spans"])
        gold_types = set(task["gold_node_types"])
        assignments = task_resolution["assignments"]
        task_correct = [a for a in assignments if a["known"] and (a["start"], a["end"]) in gold_spans]
        task_unsupported = [
            a for a in assignments if a["known"] and (a["start"], a["end"]) not in gold_spans
        ]
        task_invented = [a for a in assignments if not a["known"]]
        all_assignments += len(assignments)
        correct_assignments += len(task_correct)
        known_assignments += len(task_correct) + len(task_unsupported)
        unsupported += len(task_unsupported)
        invented += len(task_invented)
        if task_correct:
            recalled_tasks += 1
            if all(a["role"] in gold_types for a in task_correct):
                role_correct_tasks += 1
        # Source alignment validates every known selected endpoint candidate
        # against the authoritative Bronze text, not only correct ones.
        for a in assignments:
            if not a["known"]:
                continue
            bronze_slice = experiment["bronze_text"][a["start"]:a["end"]]
            if bronze_slice != a["text"]:
                alignment_violations += 1
        endpoint_details.append({
            "task_id": task["task_id"],
            "question_id": task["question_id"],
            "question_prompt": task["prompt"],
            "gold_mention_id": task["gold_mention_id"],
            "gold_spans": [list(span) for span in task["gold_spans"]],
            "gold_node_types": list(task["gold_node_types"]),
            "selected": assignments,
            "status": task_resolution["status"],
            "recalled": bool(task_correct),
            "role_correct": bool(task_correct) and all(
                a["role"] in gold_types for a in task_correct
            ),
            "correct": full_correct,
            "classification": code,
        })
        if code is not None:
            failure_counts[code] += 1
            first_failures.append({
                "task_id": task["task_id"],
                "question_id": task["question_id"],
                "gold_mention_id": task["gold_mention_id"],
                "code": code,
                "detail": detail,
            })

    status_details = []
    correct_statuses = 0
    status_target_count = 0
    known_targets = 0
    for task in experiment["status_tasks"]:
        task_resolution = resolution["statuses"][task["task_id"]]
        targets = task_resolution["targets"]
        status_target_count += len(targets)
        gold_target_spans = {
            tuple(span) for span in task.get("gold_target_spans", [])
        }
        for target in targets:
            if not target["known"]:
                invented += 1
                continue
            known_targets += 1
            if (target["start"], target["end"]) not in gold_target_spans:
                unsupported += 1
            if experiment["bronze_text"][target["start"]:target["end"]] != target["text"]:
                alignment_violations += 1
        correct = not parser_failed and task_resolution["status"] == task["gold_status"]
        reason = None
        if not correct:
            if parser_failed:
                reason = "parser failure"
            elif task_resolution["status"] != task["gold_status"]:
                reason = (
                    f"selected {task_resolution['status']} "
                    f"!= gold {task['gold_status']}"
                )
            elif task["gold_status"] == "UNKNOWN" and targets:
                reason = "UNKNOWN gold must not carry target IDs"
        elif task["gold_status"] == "UNKNOWN" and targets:
            correct = False
            reason = "UNKNOWN gold must not carry target IDs"
        if correct:
            correct_statuses += 1
        status_details.append({
            "task_id": task["task_id"],
            "question_id": task["question_id"],
            "question_prompt": task["prompt"],
            "gold_reference_id": task["gold_reference_id"],
            "gold_status": task["gold_status"],
            "selected_status": task_resolution["status"],
            "targets": targets,
            "correct": correct,
            "reason": reason,
        })

    status_denominator = len(experiment["status_tasks"])
    overall_selections = all_assignments + status_target_count
    alignment_denominator = known_assignments + known_targets
    return {
        "case_id": experiment["case_id"],
        "expected_endpoint_count": len(endpoint_tasks),
        "expected_status_count": status_denominator,
        "parseable": _metric(0 if parser_failed else 1, 1),
        "candidate_coverage": _metric(covered_endpoints, len(endpoint_tasks)),
        "endpoint_recall": _metric(recalled_tasks, len(endpoint_tasks)),
        "endpoint_precision": _metric(correct_assignments, all_assignments),
        "role_accuracy": _metric(role_correct_tasks, recalled_tasks),
        "status_accuracy": _metric(correct_statuses, status_denominator),
        "unsupported_selections": unsupported,
        "invented_selections": invented,
        "overall_selections": overall_selections,
        "unsupported_or_invented_rate": (
            (unsupported + invented) / overall_selections if overall_selections else None
        ),
        "source_alignment_violations": _metric(
            alignment_violations, alignment_denominator,
        ),
        "failure_counts": failure_counts,
        "first_failures": first_failures,
        "endpoint_details": endpoint_details,
        "status_details": status_details,
    }


def condition_aggregate(condition: str, case_results: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case metrics into the per-condition report."""
    endpoint_selection_count = sum(
        item["endpoint_precision"]["denominator"] for item in case_results
    )
    overall_selections = sum(item["overall_selections"] for item in case_results)
    unsupported = sum(item["unsupported_selections"] for item in case_results)
    invented = sum(item["invented_selections"] for item in case_results)
    gate = promotion_gate({
        "candidate_coverage": _sum_metrics([item["candidate_coverage"] for item in case_results]),
        "endpoint_recall": _sum_metrics([item["endpoint_recall"] for item in case_results]),
        "endpoint_precision": _sum_metrics([item["endpoint_precision"] for item in case_results]),
        "role_accuracy": _sum_metrics([item["role_accuracy"] for item in case_results]),
        "status_accuracy": _sum_metrics([item["status_accuracy"] for item in case_results]),
        "parseability": _sum_metrics([item["parseable"] for item in case_results]),
        "source_alignment_violations": _sum_metrics(
            [item["source_alignment_violations"] for item in case_results],
        ),
        "unsupported_or_invented_rate": (
            (unsupported + invented) / overall_selections if overall_selections else None
        ),
    })
    failure_counts = {code: 0 for code in FAILURE_CODES}
    first_failures = []
    for item in case_results:
        for code, count in item["failure_counts"].items():
            failure_counts[code] += count
        first_failures.extend(item["first_failures"])
    return {
        "condition": condition,
        "candidate_coverage": _sum_metrics([item["candidate_coverage"] for item in case_results]),
        "endpoint_recall": _sum_metrics([item["endpoint_recall"] for item in case_results]),
        "endpoint_precision": _sum_metrics([item["endpoint_precision"] for item in case_results]),
        "role_accuracy": _sum_metrics([item["role_accuracy"] for item in case_results]),
        "status_accuracy": _sum_metrics([item["status_accuracy"] for item in case_results]),
        "parseability": _sum_metrics([item["parseable"] for item in case_results]),
        "unsupported_selections": unsupported,
        "invented_selections": invented,
        "endpoint_selection_count": endpoint_selection_count,
        "overall_selections": overall_selections,
        "unsupported_or_invented_rate": (
            (unsupported + invented) / overall_selections if overall_selections else None
        ),
        "source_alignment_violations": _sum_metrics(
            [item["source_alignment_violations"] for item in case_results],
        ),
        "failure_counts": failure_counts,
        "first_failures": first_failures,
        "gate": gate,
    }


def promotion_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the preregistered per-condition promotion thresholds."""
    checks = {
        "candidate_coverage_100": metrics["candidate_coverage"]["rate"] == 1.0,
        "endpoint_recall_gte_0.90": metrics["endpoint_recall"]["rate"] is not None
        and metrics["endpoint_recall"]["rate"] >= GATE_THRESHOLDS["endpoint_recall"],
        "endpoint_precision_gte_0.90": metrics["endpoint_precision"]["rate"] is not None
        and metrics["endpoint_precision"]["rate"] >= GATE_THRESHOLDS["endpoint_precision"],
        "role_accuracy_gte_0.85": metrics["role_accuracy"]["rate"] is not None
        and metrics["role_accuracy"]["rate"] >= GATE_THRESHOLDS["role_accuracy"],
        "unsupported_or_invented_lte_0.05": (
            metrics["unsupported_or_invented_rate"] is not None
            and metrics["unsupported_or_invented_rate"] <= GATE_THRESHOLDS[
                "unsupported_or_invented_rate"
            ]
        ),
        "source_alignment_violations_zero": (
            metrics["source_alignment_violations"]["hit_count"]
            == GATE_THRESHOLDS["source_alignment_violations"]
        ),
    }
    return {
        "gate_version": GATE_VERSION,
        "thresholds": dict(GATE_THRESHOLDS),
        "checks": checks,
        "passed": all(checks.values()),
    }


def assemble_case_record(
    experiment: Mapping[str, Any],
    case: Mapping[str, Any],
    condition: str,
    fixture: Mapping[str, Any],
    *,
    text: str,
    request: Mapping[str, Any],
    raw_response: str | None,
    provider_failure: str | None,
    parse_error: str | None,
    parsed: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the retained per-case condition artifact record."""
    parser_failed = provider_failure is not None or parse_error is not None
    resolution = resolve_parsed_payload(experiment, parsed)
    evaluation = evaluate_case(
        experiment, resolution,
        parser_failed=parse_error is not None,
        provider_failure=provider_failure,
    )
    return {
        "case_id": experiment["case_id"],
        "condition": condition,
        "input": silver_input_record(case, condition, fixture),
        "catalog": experiment["catalog"],
        "expected_endpoint_tasks": experiment["endpoint_tasks"],
        "expected_status_tasks": experiment["status_tasks"],
        "input_hashes": {
            "bronze_text_sha256": experiment["bronze_text_sha256"],
            "catalog_sha256": experiment["catalog_sha256"],
            "condition_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "request_sha256": request["request_sha256"],
        },
        "request": request,
        "raw_response": raw_response,
        "raw_response_sha256": (
            hashlib.sha256(str(raw_response).encode("utf-8")).hexdigest()
            if raw_response is not None else None
        ),
        "provider_failure": provider_failure,
        "parse_error": parse_error,
        "parsed": parsed,
        "resolutions": resolution,
        "metrics": evaluation,
    }


def run_case_condition(
    experiment: Mapping[str, Any],
    case: Mapping[str, Any],
    condition: str,
    fixture: Mapping[str, Any],
    chat: Callable[..., str],
) -> dict[str, Any]:
    """Run one case/condition provider call and evaluate the result."""
    text = condition_text(case, condition, fixture)
    request = build_request(experiment, text, condition=condition)
    raw_response = None
    provider_failure = None
    try:
        raw_response = chat(
            system=request["system"],
            user=request["user"],
            temperature=REFERENCE_TEMPERATURE,
            max_tokens=request["max_tokens"],
            model=request["model"],
            thinking=request["thinking"],
        )
    except Exception as exc:
        provider_failure = f"{type(exc).__name__}: {exc}"
    parsed = None
    parse_error = None
    if provider_failure is None:
        try:
            parsed = parse_model_response(raw_response, experiment)
        except Phase2GParseError as exc:
            parse_error = str(exc)
    return assemble_case_record(
        experiment, case, condition, fixture,
        text=text,
        request=request,
        raw_response=raw_response,
        provider_failure=provider_failure,
        parse_error=parse_error,
        parsed=parsed,
    )


def run_experiment(
    benchmark: Mapping[str, Any],
    fixture: Mapping[str, Any],
    chat: Callable[..., str],
) -> dict[str, Any]:
    """Run all 15 case/condition provider calls (one per case per condition)."""
    validate_fixture_against_benchmark(benchmark, fixture)
    experiments = {
        case["id"]: build_case_experiment(case) for case in benchmark["cases"]
    }
    validate_experiment_coverage(experiments)
    condition_results = {}
    for condition in CONDITIONS:
        case_results = []
        for case in benchmark["cases"]:
            case_results.append(
                run_case_condition(
                    experiments[case["id"]], case, condition, fixture, chat,
                ),
            )
        condition_results[condition] = {
            "metrics": condition_aggregate(
                condition, [item["metrics"] for item in case_results],
            ),
            "cases": {item["case_id"]: item for item in case_results},
        }
    satisfied = [
        condition for condition, result in condition_results.items()
        if result["metrics"]["gate"]["passed"]
    ]
    return {
        "conditions": condition_results,
        "promotion_gate": {
            "gate_version": GATE_VERSION,
            "passed": bool(satisfied),
            "satisfied_conditions": satisfied,
        },
    }


def _definition() -> dict[str, Any]:
    return {
        "run_version": RUN_VERSION,
        "conditions": list(CONDITIONS),
        "node_types": sorted(NODE_TYPES),
        "reference_statuses": sorted(REFERENCE_STATUSES),
        "failure_codes": sorted(FAILURE_CODES),
        "failure_precedence": list(FAILURE_PRECEDENCE),
        "gate_thresholds": dict(GATE_THRESHOLDS),
        "model": REFERENCE_MODEL,
        "thinking": REFERENCE_THINKING,
        "temperature": REFERENCE_TEMPERATURE,
        "max_tokens": REFERENCE_MAX_TOKENS,
        "endpoint": REFERENCE_ENDPOINT,
        "catalog_rendering": (
            "complete alias -> authoritative bronze text mapping in every "
            "request; full phase-2f candidate ids, window-local and upstream "
            "absolute offsets, bronze text, and segment provenance retained in "
            "the input representation and every per-case condition artifact "
            "together with the expected endpoint/status task definitions"
        ),
    }


def _git_state(repo: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        text=True, capture_output=True,
    ).stdout.strip())
    return commit, dirty


def build_aggregate(
    benchmark_path: Path,
    fixture_path: Path,
    result: Mapping[str, Any],
    *,
    repo: Path,
    provider: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    benchmark_file_sha256 = hashlib.sha256(
        benchmark_path.read_bytes(),
    ).hexdigest()
    fixture_file_sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    definition = _definition()
    commit, dirty = _git_state(repo)
    inner = {
        "run_version": RUN_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": commit,
        "repository_dirty": dirty,
        "provider": provider,
        "definition": definition,
        "definition_sha256": canonical_sha256(definition),
        "input_hashes": {
            "benchmark_content_sha256": BENCHMARK_CONTENT_SHA256,
            "benchmark_file_sha256": benchmark_file_sha256,
            "silver_fixture_content_sha256": SILVER_FIXTURE_CONTENT_SHA256,
            "silver_fixture_file_sha256": fixture_file_sha256,
        },
        "benchmark_content_sha256": BENCHMARK_CONTENT_SHA256,
        "silver_fixture_content_sha256": SILVER_FIXTURE_CONTENT_SHA256,
        "promotion_gate": result["promotion_gate"],
        "conditions": result["conditions"],
    }
    return {"content_sha256": canonical_sha256(inner), **inner}


def publish_artifact(
    output: Path,
    aggregate: Mapping[str, Any],
) -> Path:
    """Atomically publish the immutable artifact outside the repository."""
    output = Path(output)
    if output.exists():
        raise ValueError("output directory already exists; artifacts are immutable")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=parent))
    files = []
    try:
        aggregate_path = temporary / "phase2g-endpoint-recovery.json"
        aggregate_path.write_text(
            json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        files.append(aggregate_path)
        condition_dir = temporary / "conditions"
        for condition, record in aggregate["conditions"].items():
            for case_id, case_record in record["cases"].items():
                case_path = condition_dir / condition / f"{case_id}.json"
                case_path.parent.mkdir(parents=True, exist_ok=True)
                case_path.write_text(
                    json.dumps(case_record, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                files.append(case_path)
        manifest = {
            "files": [
                {
                    "path": str(path.relative_to(temporary)),
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(files, key=lambda item: str(item.relative_to(temporary)))
            ],
        }
        manifest_path = temporary / "MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        files.append(manifest_path)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def _load_aggregate(directory: Path) -> Mapping[str, Any]:
    return json.loads(
        (Path(directory) / "phase2g-endpoint-recovery.json").read_text(encoding="utf-8"),
    )


def compare_artifacts(left: Path, right: Path) -> list[str]:
    """Compare two clean reruns of deterministic inputs and score/failure
    distributions while allowing timestamps and raw-output hashes to differ."""
    left_body = _load_aggregate(left)
    right_body = _load_aggregate(right)
    differences: list[str] = []
    if left_body.get("run_version") != right_body.get("run_version"):
        differences.append("run_version differs")
    if left_body.get("definition_sha256") != right_body.get("definition_sha256"):
        differences.append("definition_sha256 differs")
    if left_body.get("input_hashes") != right_body.get("input_hashes"):
        differences.append("input_hashes differ")
    if left_body.get("promotion_gate") != right_body.get("promotion_gate"):
        differences.append("promotion_gate differs")
    left_conditions = left_body.get("conditions", {})
    right_conditions = right_body.get("conditions", {})
    if set(left_conditions) != set(right_conditions):
        return differences + ["condition sets differ"]
    for condition in CONDITIONS:
        left_record = left_conditions[condition]
        right_record = right_conditions[condition]
        if left_record["metrics"] != right_record["metrics"]:
            differences.append(f"{condition}: metrics differ")
        left_cases = left_record.get("cases", {})
        right_cases = right_record.get("cases", {})
        if set(left_cases) != set(right_cases):
            differences.append(f"{condition}: case sets differ")
            continue
        for case_id in left_cases:
            left_case = left_cases[case_id]
            right_case = right_cases[case_id]
            for key in ("input_hashes", "metrics"):
                if left_case.get(key) != right_case.get(key):
                    differences.append(f"{condition}/{case_id}: {key} differs")
    return differences
