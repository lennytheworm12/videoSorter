"""Constrained, catalog-complete source-local coreference for Phase 2F."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Any, Callable, Mapping

from pipeline.semantic_ir import (
    AmbiguityState, COMPILER_VERSION, EdgeType, ModelDecisionProvenance,
    SemanticEdge, SemanticNode, SemanticQualifiers, SourceSpan, content_sha256,
)
from pipeline.semantic_source import SemanticSourceWindow


COREFERENCE_VERSION = "phase2f-coreference-v3-reference-groups"
COREFERENCE_PROMPT_VERSION = "phase2f-coreference-classifier-v2"
COREFERENCE_SYSTEM = (
    "Return strict JSON only. Resolve a source reference only when the supplied context supports it. "
    "Use supplied node IDs only. Ambiguity or abstention is preferable to guessing."
)
COREFERENCE_STATUSES = frozenset({
    "RESOLVED", "NONE", "UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE",
})
_REFERENCE_FORMS = frozenset({
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their",
    "theirs", "this", "that", "these", "those",
    "you all", "you guys", "you two",
    "myself", "yourself", "yourselves", "ourselves", "himself", "herself", "itself",
    "themselves", "oneself", "each other", "one another", "former", "latter",
    "the former", "the latter",
})
_REFERENCE_PREFIXES = frozenset({
    "this", "that", "these", "those",
})


@dataclass(frozen=True)
class CoreferenceCandidateSet:
    candidate_set_id: str
    window_id: str
    source_node_id: str
    target_node_ids: tuple[str, ...]
    evidence_span: SourceSpan
    max_segment_distance: int = 2
    version: str = COREFERENCE_VERSION

    def validate(self, window: SemanticSourceWindow, nodes: tuple[SemanticNode, ...]) -> None:
        window.validate()
        _validate_nodes(window, nodes)
        if self.version != COREFERENCE_VERSION or self.window_id != window.window_id:
            raise ValueError("coreference candidate identity/version is invalid")
        if isinstance(self.max_segment_distance, bool) or not isinstance(
            self.max_segment_distance, int,
        ) or self.max_segment_distance < 0:
            raise ValueError("coreference segment distance must be a non-negative integer")
        expected = generate_coreference_candidate_sets(
            window, nodes, max_segment_distance=self.max_segment_distance,
        )
        match = next((item for item in expected if item.source_node_id == self.source_node_id), None)
        if match != self:
            raise ValueError("coreference candidate set is not the complete deterministic source set")


@dataclass(frozen=True)
class CoreferenceDecision:
    candidate_set_id: str
    status: str
    target_node_id: str | None
    candidate_node_ids: tuple[str, ...]
    confidence: float
    raw_output: str
    parsed_output: Mapping[str, Any] | None
    request_json: str
    model_id: str
    configuration_sha256: str
    failure: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_set_id, str) or not self.candidate_set_id:
            raise ValueError("coreference decision candidate-set ID is invalid")
        if self.status not in COREFERENCE_STATUSES:
            raise ValueError("coreference decision status is invalid")
        if self.target_node_id is not None and (
            not isinstance(self.target_node_id, str) or not self.target_node_id
        ):
            raise ValueError("coreference target node ID is invalid")
        if not isinstance(self.candidate_node_ids, tuple) or any(
            not isinstance(item, str) or not item for item in self.candidate_node_ids
        ) or len(set(self.candidate_node_ids)) != len(self.candidate_node_ids):
            raise ValueError("coreference decision candidate IDs must be a unique immutable tuple")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)) \
                or not 0 <= self.confidence <= 1:
            raise ValueError("coreference decision confidence must be between zero and one")
        if not isinstance(self.raw_output, str) or not isinstance(self.request_json, str):
            raise ValueError("coreference raw/request output must be retained strings")
        if self.parsed_output is not None and not isinstance(self.parsed_output, Mapping):
            raise ValueError("coreference parsed output must be a mapping or null")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("coreference decision model ID is invalid")
        if re.fullmatch(r"[0-9a-f]{64}", self.configuration_sha256) is None:
            raise ValueError("coreference decision configuration hash is invalid")
        if self.failure is not None and (not isinstance(self.failure, str) or not self.failure):
            raise ValueError("coreference failure must be null or a non-empty string")


@dataclass(frozen=True)
class CoreferenceApplication:
    nodes: tuple[SemanticNode, ...]
    edges: tuple[SemanticEdge, ...]
    status: str


@dataclass(frozen=True)
class CoreferenceCatalogResult:
    status: str
    candidate_sets: tuple[CoreferenceCandidateSet, ...]
    decisions: tuple[CoreferenceDecision, ...]
    nodes: tuple[SemanticNode, ...]
    edges: tuple[SemanticEdge, ...]
    max_segment_distance: int
    failures: tuple[str, ...] = ()
    abstentions: tuple[str, ...] = ()


def generate_coreference_candidate_sets(
    window: SemanticSourceWindow,
    nodes: tuple[SemanticNode, ...],
    *,
    max_segment_distance: int = 2,
) -> tuple[CoreferenceCandidateSet, ...]:
    window.validate()
    _validate_nodes(window, nodes)
    if isinstance(max_segment_distance, bool) or not isinstance(
        max_segment_distance, int,
    ) or max_segment_distance < 0:
        raise ValueError("coreference segment distance must be non-negative")
    memberships = {node.node_id: _segment_indexes(window, node.source_span) for node in nodes}
    values = []
    for source in nodes:
        if not _is_reference_expression(source.source_span.text):
            continue
        targets = tuple(sorted((
            target for target in nodes
            if target.node_id != source.node_id
            and not _overlaps(source.source_span, target.source_span)
            and min(
                abs(a - b) for a in memberships[source.node_id] for b in memberships[target.node_id]
            ) <= max_segment_distance
        ), key=lambda item: item.node_id))
        relevant_segments = tuple(
            segment for index, segment in enumerate(window.segments)
            if min(abs(index - source_index) for source_index in memberships[source.node_id])
            <= max_segment_distance
        )
        start = min(segment.start for segment in relevant_segments)
        end = max(segment.end for segment in relevant_segments)
        evidence = SourceSpan(
            window.source_id, window.window_id, start, end, window.text[start:end],
            window.source_start + start, window.source_start + end, window.speaker,
            window.start_ms, window.end_ms,
        )
        target_ids = tuple(item.node_id for item in targets)
        values.append(CoreferenceCandidateSet(
            _candidate_set_id(window.window_id, source.node_id, target_ids, max_segment_distance),
            window.window_id, source.node_id, target_ids, evidence, max_segment_distance,
        ))
    return tuple(sorted(values, key=lambda item: item.candidate_set_id))


def coreference_prompt(
    window: SemanticSourceWindow,
    candidate_set: CoreferenceCandidateSet,
    node_by_id: Mapping[str, SemanticNode],
) -> str:
    source = node_by_id[candidate_set.source_node_id]
    reference = {
        "node_id": source.node_id, "node_type": source.node_type.value,
        "source_text": source.source_span.text, "start": source.source_span.local_start,
        "end": source.source_span.local_end,
    }
    targets = [{
        "node_id": node_id, "node_type": node_by_id[node_id].node_type.value,
        "source_text": node_by_id[node_id].source_span.text,
        "start": node_by_id[node_id].source_span.local_start,
        "end": node_by_id[node_id].source_span.local_end,
    } for node_id in candidate_set.target_node_ids]
    return (
        f"SOURCE WINDOW:\n{window.text}\n"
        f"LOCAL EVIDENCE [{candidate_set.evidence_span.local_start},"
        f"{candidate_set.evidence_span.local_end}):\n{candidate_set.evidence_span.text}\n"
        f"REFERENCE: {json.dumps(reference, ensure_ascii=False, separators=(',', ':'))}\n"
        f"POSSIBLE TARGETS: {json.dumps(targets, ensure_ascii=False, separators=(',', ':'))}\n"
        "If no target is supplied, RESOLVED and AMBIGUOUS are impossible; preserve UNKNOWN or "
        "INSUFFICIENT_EVIDENCE rather than guessing. Return exactly "
        '{"status":"RESOLVED|NONE|UNKNOWN|AMBIGUOUS|INSUFFICIENT_EVIDENCE",'
        '"target_node_id":"<one supplied ID or null>","candidate_node_ids":[],"confidence":0.0}.'
    )


def classify_coreference(
    window: SemanticSourceWindow,
    nodes: tuple[SemanticNode, ...],
    candidate_set: CoreferenceCandidateSet,
    chat: Callable[..., str],
    *,
    model: str,
    configuration: Mapping[str, Any],
    max_tokens: int = 256,
    thinking: str | None = None,
) -> CoreferenceDecision:
    candidate_set.validate(window, nodes)
    if not isinstance(model, str) or not model.strip() or not isinstance(configuration, Mapping):
        raise ValueError("coreference model/configuration is invalid")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("coreference max_tokens must be positive")
    if thinking is not None and (not isinstance(thinking, str) or not thinking.strip()):
        raise ValueError("coreference thinking mode is invalid")
    node_by_id = {node.node_id: node for node in nodes}
    prompt = coreference_prompt(window, candidate_set, node_by_id)
    effective = {
        "caller_configuration": dict(configuration), "temperature": 0.0, "max_tokens": max_tokens,
        "model": model, "thinking": thinking, "prompt_version": COREFERENCE_PROMPT_VERSION,
        "candidate_version": candidate_set.version,
        "max_segment_distance": candidate_set.max_segment_distance,
    }
    request = {"system": COREFERENCE_SYSTEM, "user": prompt, **effective}
    request_json = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    config_hash = content_sha256(effective)
    try:
        raw = chat(
            system=COREFERENCE_SYSTEM, user=prompt, temperature=0.0, max_tokens=max_tokens,
            model=model, thinking=thinking,
        )
    except Exception as exc:
        return CoreferenceDecision(
            candidate_set.candidate_set_id, "INSUFFICIENT_EVIDENCE", None, (), 0.0,
            "", None, request_json, model, config_hash,
            f"CoreferenceProviderError:{type(exc).__name__}",
        )
    try:
        status, target_id, candidate_ids, confidence, body = parse_coreference_decision(
            raw, candidate_set,
        )
    except Exception as exc:
        return CoreferenceDecision(
            candidate_set.candidate_set_id, "INSUFFICIENT_EVIDENCE", None, (), 0.0,
            raw if isinstance(raw, str) else repr(raw), None, request_json, model, config_hash,
            type(exc).__name__,
        )
    return CoreferenceDecision(
        candidate_set.candidate_set_id, status, target_id, candidate_ids, confidence,
        raw, body, request_json, model, config_hash,
    )


def classify_coreference_catalog(
    window: SemanticSourceWindow,
    nodes: tuple[SemanticNode, ...],
    candidate_sets: tuple[CoreferenceCandidateSet, ...],
    chat: Callable[..., str],
    *,
    model: str,
    configuration: Mapping[str, Any],
    max_tokens: int = 256,
    thinking: str | None = None,
    max_segment_distance: int = 2,
) -> CoreferenceCatalogResult:
    expected = generate_coreference_candidate_sets(
        window, nodes, max_segment_distance=max_segment_distance,
    )
    if not isinstance(candidate_sets, tuple) or candidate_sets != expected:
        raise ValueError("coreference catalog must be the complete deterministic catalog")
    decisions = tuple(classify_coreference(
        window, nodes, candidate_set, chat, model=model, configuration=configuration,
        max_tokens=max_tokens, thinking=thinking,
    ) for candidate_set in candidate_sets)
    return assemble_coreference_catalog(
        window, nodes, candidate_sets, decisions,
        max_segment_distance=max_segment_distance,
    )


def parse_coreference_decision(
    raw: str, candidate_set: CoreferenceCandidateSet,
) -> tuple[str, str | None, tuple[str, ...], float, Mapping[str, Any]]:
    body = _strict_object(raw)
    if set(body) != {"status", "target_node_id", "candidate_node_ids", "confidence"}:
        raise ValueError("coreference decision shape is invalid")
    status, target_id = body.get("status"), body.get("target_node_id")
    candidate_ids, confidence = body.get("candidate_node_ids"), body.get("confidence")
    if status not in COREFERENCE_STATUSES:
        raise ValueError("coreference status is invalid")
    if not isinstance(candidate_ids, list) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("coreference candidate IDs must be a unique list")
    if any(item not in candidate_set.target_node_ids for item in candidate_ids):
        raise ValueError("coreference candidate is outside the deterministic set")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("coreference confidence must be between zero and one")
    if status == "RESOLVED":
        if target_id not in candidate_set.target_node_ids or candidate_ids or confidence < 0.5:
            raise ValueError("resolved reference requires one supported target above threshold")
    elif status == "AMBIGUOUS":
        if target_id is not None or len(candidate_ids) < 2:
            raise ValueError("ambiguous reference requires at least two supplied candidates")
    elif target_id is not None or candidate_ids:
        raise ValueError("non-resolved reference cannot smuggle targets")
    return str(status), target_id, tuple(candidate_ids), float(confidence), body


def apply_coreference_decision(
    window: SemanticSourceWindow,
    nodes: tuple[SemanticNode, ...],
    candidate_set: CoreferenceCandidateSet,
    decision: CoreferenceDecision,
) -> CoreferenceApplication:
    """Apply one validated decision; final compiler assembly uses the catalog API."""
    updated, edges = _apply_decision(window, nodes, candidate_set, decision)
    return CoreferenceApplication(updated, edges, decision.status)


def assemble_coreference_catalog(
    window: SemanticSourceWindow,
    nodes: tuple[SemanticNode, ...],
    candidate_sets: tuple[CoreferenceCandidateSet, ...],
    decisions: tuple[CoreferenceDecision, ...],
    *,
    max_segment_distance: int = 2,
) -> CoreferenceCatalogResult:
    expected = generate_coreference_candidate_sets(
        window, nodes, max_segment_distance=max_segment_distance,
    )
    if not isinstance(candidate_sets, tuple) or candidate_sets != expected:
        raise ValueError("coreference catalog is incomplete or non-deterministic")
    if not isinstance(decisions, tuple) or tuple(
        item.candidate_set_id for item in decisions
    ) != tuple(item.candidate_set_id for item in candidate_sets):
        raise ValueError("coreference decisions must exactly cover the complete catalog")
    current_nodes = nodes
    edges = []
    failures = []
    abstentions = []
    completed = 0
    for candidate_set, decision in zip(candidate_sets, decisions):
        current_nodes, produced = _apply_decision(
            window, current_nodes, candidate_set, decision,
        )
        edges.extend(produced)
        if decision.failure:
            failures.append(f"reference:{candidate_set.candidate_set_id}:{decision.failure}")
        elif decision.status in {"UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"}:
            abstentions.append(f"reference:{candidate_set.candidate_set_id}:{decision.status}")
        else:
            completed += 1
    if failures or abstentions:
        status = "PARTIAL" if completed else "INSUFFICIENT_EVIDENCE"
    else:
        status = "OK" if decisions else "NONE"
    return CoreferenceCatalogResult(
        status, candidate_sets, decisions, current_nodes, tuple(edges), max_segment_distance,
        tuple(failures), tuple(abstentions),
    )


def _apply_decision(
    window: SemanticSourceWindow,
    nodes: tuple[SemanticNode, ...],
    candidate_set: CoreferenceCandidateSet,
    decision: CoreferenceDecision,
) -> tuple[tuple[SemanticNode, ...], tuple[SemanticEdge, ...]]:
    candidate_set.validate(window, nodes)
    if decision.candidate_set_id != candidate_set.candidate_set_id:
        raise ValueError("coreference decision belongs to another candidate set")
    node_by_id = {node.node_id: node for node in nodes}
    request = _validate_request(window, candidate_set, node_by_id, decision)
    source = node_by_id[candidate_set.source_node_id]
    if decision.failure:
        _validate_failure(decision, candidate_set)
        status, target_id, candidate_ids, confidence = (
            "INSUFFICIENT_EVIDENCE", None, (), 0.0,
        )
    else:
        status, target_id, candidate_ids, confidence, body = parse_coreference_decision(
            decision.raw_output, candidate_set,
        )
        if (status, target_id, candidate_ids, confidence, body) != (
            decision.status, decision.target_node_id, decision.candidate_node_ids,
            decision.confidence, decision.parsed_output,
        ):
            raise ValueError("coreference raw output contradicts its retained decision")
    decision_output = {
        "raw_output": decision.raw_output, "status": decision.status, "failure": decision.failure,
    }
    provenance = ModelDecisionProvenance(
        candidate_set.candidate_set_id, decision.model_id, COREFERENCE_PROMPT_VERSION,
        decision.configuration_sha256, content_sha256(request), content_sha256(decision_output),
        (candidate_set.candidate_set_id,) + candidate_set.target_node_ids,
    )
    edges: tuple[SemanticEdge, ...] = ()
    if status == "RESOLVED":
        target = node_by_id[target_id]
        updated = replace(
            source, ambiguity=AmbiguityState.NONE,
            referent_candidates=(target.source_span,),
            additional_provenance=source.additional_provenance + (provenance,),
            referent_candidate_node_ids=(target.node_id,),
        )
        edges = (SemanticEdge(
            EdgeType.REFERS_TO, updated.node_id, target.node_id, (candidate_set.evidence_span,),
            provenance, SemanticQualifiers(), AmbiguityState.NONE, confidence, COMPILER_VERSION,
        ),)
    elif status == "AMBIGUOUS":
        ordered_targets = tuple(node_by_id[item] for item in candidate_ids)
        updated = replace(
            source, ambiguity=AmbiguityState.MULTIPLE_CANDIDATES,
            referent_candidates=tuple(item.source_span for item in ordered_targets),
            additional_provenance=source.additional_provenance + (provenance,),
            referent_candidate_node_ids=tuple(item.node_id for item in ordered_targets),
        )
    else:
        ambiguity = {
            "NONE": AmbiguityState.NONE,
            "UNKNOWN": AmbiguityState.UNKNOWN,
            "INSUFFICIENT_EVIDENCE": AmbiguityState.INSUFFICIENT_EVIDENCE,
        }[status]
        updated = replace(
            source, ambiguity=ambiguity, referent_candidates=(),
            additional_provenance=source.additional_provenance + (provenance,),
            referent_candidate_node_ids=(),
        )
    updated_nodes = tuple(updated if node.node_id == source.node_id else node for node in nodes)
    return updated_nodes, edges


def _validate_request(
    window: SemanticSourceWindow,
    candidate_set: CoreferenceCandidateSet,
    node_by_id: Mapping[str, SemanticNode],
    decision: CoreferenceDecision,
) -> Mapping[str, Any]:
    request = _strict_object(decision.request_json)
    expected_keys = {
        "system", "user", "caller_configuration", "temperature", "max_tokens", "model",
        "thinking", "prompt_version", "candidate_version", "max_segment_distance",
    }
    if set(request) != expected_keys or not isinstance(request.get("caller_configuration"), Mapping):
        raise ValueError("coreference request artifact has an invalid shape")
    temperature, max_tokens, thinking = (
        request.get("temperature"), request.get("max_tokens"), request.get("thinking"),
    )
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature != 0.0:
        raise ValueError("coreference request temperature is invalid")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("coreference request max_tokens is invalid")
    if thinking is not None and (not isinstance(thinking, str) or not thinking.strip()):
        raise ValueError("coreference request thinking mode is invalid")
    if (
        request.get("system") != COREFERENCE_SYSTEM
        or request.get("user") != coreference_prompt(window, candidate_set, node_by_id)
        or request.get("prompt_version") != COREFERENCE_PROMPT_VERSION
        or request.get("candidate_version") != candidate_set.version
        or request.get("max_segment_distance") != candidate_set.max_segment_distance
        or request.get("model") != decision.model_id
    ):
        raise ValueError("coreference request constants are invalid")
    effective = {key: request[key] for key in (
        "caller_configuration", "temperature", "max_tokens", "model", "thinking",
        "prompt_version", "candidate_version", "max_segment_distance",
    )}
    if decision.configuration_sha256 != content_sha256(effective):
        raise ValueError("coreference request provenance is invalid")
    return request


def _validate_failure(
    decision: CoreferenceDecision, candidate_set: CoreferenceCandidateSet,
) -> None:
    if (
        decision.status != "INSUFFICIENT_EVIDENCE" or decision.target_node_id is not None
        or decision.candidate_node_ids or decision.confidence != 0.0
        or decision.parsed_output is not None
    ):
        raise ValueError("failed coreference decision cannot retain accepted semantics")
    if decision.failure.startswith("CoreferenceProviderError:"):
        if decision.raw_output or re.fullmatch(
            r"CoreferenceProviderError:[A-Za-z_][A-Za-z0-9_]*", decision.failure,
        ) is None:
            raise ValueError("coreference provider failure evidence is invalid")
        return
    try:
        parse_coreference_decision(decision.raw_output, candidate_set)
    except Exception as exc:
        if decision.failure != type(exc).__name__:
            raise ValueError("coreference parse failure taxonomy is invalid") from exc
        return
    raise ValueError("claimed coreference parse failure output is actually valid")


def _validate_nodes(window: SemanticSourceWindow, nodes: tuple[SemanticNode, ...]) -> None:
    if not isinstance(nodes, tuple) or any(not isinstance(node, SemanticNode) for node in nodes):
        raise ValueError("coreference nodes must be an immutable semantic-node tuple")
    if len({node.node_id for node in nodes}) != len(nodes):
        raise ValueError("coreference nodes must have unique IDs")
    for node in nodes:
        node.source_span.validate_against(
            window.source_id, window.window_id, window.text,
            window_source_start=window.source_start, speaker=window.speaker,
            start_timestamp=window.start_ms, end_timestamp=window.end_ms,
        )


def _is_reference_expression(text: str) -> bool:
    words = text.strip().casefold().replace("’", "'").split()
    normalized = " ".join(words)
    return normalized in _REFERENCE_FORMS or (
        1 < len(words) <= 5 and words[0] in _REFERENCE_PREFIXES
    )


def _overlaps(left: SourceSpan, right: SourceSpan) -> bool:
    return left.local_start < right.local_end and right.local_start < left.local_end


def _segment_indexes(window: SemanticSourceWindow, span: SourceSpan) -> tuple[int, ...]:
    indexes = tuple(index for index, segment in enumerate(window.segments)
                    if segment.start < span.local_end and span.local_start < segment.end)
    if not indexes:
        raise ValueError("coreference node lies outside Pass 0 segments")
    return indexes


def _candidate_set_id(
    window_id: str, source_id: str, target_ids: tuple[str, ...], distance: int,
) -> str:
    payload = json.dumps(
        [COREFERENCE_VERSION, window_id, source_id, list(target_ids), distance],
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{window_id}:r{hashlib.sha256(payload).hexdigest()[:20]}"


def _strict_object(raw: str) -> Mapping[str, Any]:
    if not isinstance(raw, str):
        raise ValueError("coreference output must be a string")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON contains duplicate keys")
            result[key] = value
        return result

    try:
        body = json.loads(raw, object_pairs_hook=unique)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("coreference classifier returned malformed JSON") from exc
    if not isinstance(body, Mapping):
        raise ValueError("coreference classifier must return an object")
    return body


__all__ = [
    "COREFERENCE_VERSION", "COREFERENCE_PROMPT_VERSION", "CoreferenceCandidateSet",
    "CoreferenceDecision", "CoreferenceApplication", "CoreferenceCatalogResult",
    "generate_coreference_candidate_sets", "coreference_prompt", "classify_coreference",
    "classify_coreference_catalog", "parse_coreference_decision",
    "apply_coreference_decision", "assemble_coreference_catalog",
]
