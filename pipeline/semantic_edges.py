"""Bounded candidate-pair generation and general semantic edge classification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Callable, Iterable, Mapping

from pipeline.semantic_ir import (
    AmbiguityState,
    COMPILER_VERSION,
    EdgeType,
    ModelDecisionProvenance,
    NodeType,
    SemanticEdge,
    SemanticNode,
    SemanticQualifiers,
    SemanticGraph,
    SourceSpan,
    content_sha256,
    edge_type_supports,
)
from pipeline.semantic_source import SemanticSourceWindow


EDGE_PAIR_VERSION = "phase2f-edge-pairs-v2-schema-v4-signatures"
EDGE_CLASSIFIER_PROMPT_VERSION = "phase2f-edge-classifier-v1"
EDGE_CLASSIFIER_SYSTEM = (
    "Return strict JSON only. Classify a general source-semantic relation between supplied mentions. "
    "Do not generate propositions, strategic concepts, endpoints, or source offsets."
)
CLASSIFICATION_STATUSES = frozenset({"SUPPORTED", "NO_RELATION", "UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"})
SUPPORTED_MIN_CONFIDENCE = 0.5


@dataclass(frozen=True)
class CandidateEdgePair:
    pair_id: str
    window_id: str
    source_node_id: str
    target_node_id: str
    allowed_edge_types: tuple[EdgeType, ...]
    evidence_span: SourceSpan
    character_distance: int
    segment_distance: int
    max_character_distance: int
    max_segment_distance: int
    version: str = EDGE_PAIR_VERSION

    def validate(self, window: SemanticSourceWindow, node_by_id: Mapping[str, SemanticNode]) -> None:
        window.validate()
        if not isinstance(node_by_id, Mapping) or any(
            not isinstance(key, str) or not isinstance(node, SemanticNode)
            for key, node in node_by_id.items()
        ):
            raise ValueError("candidate edge node mapping is malformed")
        if self.window_id != window.window_id:
            raise ValueError("candidate edge pair belongs to a different window")
        if self.version != EDGE_PAIR_VERSION:
            raise ValueError("candidate edge pair version is unsupported")
        if self.source_node_id == self.target_node_id:
            raise ValueError("candidate edge pair cannot be self-referential")
        if self.source_node_id not in node_by_id or self.target_node_id not in node_by_id:
            raise ValueError("candidate edge pair endpoint is missing")
        if any(key != node.node_id for key, node in node_by_id.items()):
            raise ValueError("candidate edge node mapping key contradicts node identity")
        source, target = node_by_id[self.source_node_id], node_by_id[self.target_node_id]
        for node in (source, target):
            node.source_span.validate_against(
                window.source_id, window.window_id, window.text, window_source_start=window.source_start,
                speaker=window.speaker, start_timestamp=window.start_ms, end_timestamp=window.end_ms,
            )
        if not self.allowed_edge_types or len(set(self.allowed_edge_types)) != len(self.allowed_edge_types):
            raise ValueError("candidate edge pair needs unique allowed relations")
        if self.allowed_edge_types != _compatible_edges(source, target):
            raise ValueError("candidate edge pair relations do not match structural signatures")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.character_distance, self.segment_distance,
                self.max_character_distance, self.max_segment_distance,
            )
        ):
            raise ValueError("candidate edge pair distances must be non-negative integers")
        if self.character_distance != _span_distance(source.source_span, target.source_span):
            raise ValueError("candidate edge pair character distance is invalid")
        if self.segment_distance != _segment_distance(window, source.source_span, target.source_span):
            raise ValueError("candidate edge pair segment distance is invalid")
        if self.character_distance > self.max_character_distance or self.segment_distance > self.max_segment_distance:
            raise ValueError("candidate edge pair violates its retained pruning configuration")
        self.evidence_span.validate_against(
            window.source_id, window.window_id, window.text, window_source_start=window.source_start,
            speaker=window.speaker, start_timestamp=window.start_ms, end_timestamp=window.end_ms,
        )
        expected_evidence = _evidence_for_pair(window, source.source_span, target.source_span)
        if self.evidence_span != expected_evidence:
            raise ValueError("candidate edge pair evidence is not the deterministic discourse envelope")
        expected = _pair_id(
            window.window_id, self.source_node_id, self.target_node_id,
            self.max_character_distance, self.max_segment_distance,
        )
        if self.pair_id != expected:
            raise ValueError("candidate edge pair ID is not stable")


@dataclass(frozen=True)
class EdgeClassificationResult:
    pair_id: str
    status: str
    edges: tuple[SemanticEdge, ...]
    raw_output: str
    parsed_output: Mapping[str, Any] | None
    failure: str | None = None
    latency_ms: int | None = None
    request_json: str = ""
    model_id: str | None = None
    configuration_sha256: str | None = None

    @property
    def edge(self) -> SemanticEdge | None:
        return self.edges[0] if len(self.edges) == 1 else None


@dataclass(frozen=True)
class EdgeCatalogClassificationResult:
    status: str
    pairs: tuple[CandidateEdgePair, ...]
    results: tuple[EdgeClassificationResult, ...]
    edges: tuple[SemanticEdge, ...]
    max_character_distance: int = 600
    max_segment_distance: int = 2
    failures: tuple[str, ...] = ()
    abstentions: tuple[str, ...] = ()


class EdgeProviderError(Exception):
    """Provider failed before a pair received raw classification output."""


def generate_candidate_edge_pairs(
    window: SemanticSourceWindow,
    nodes: tuple[SemanticNode, ...],
    *,
    max_character_distance: int = 600,
    max_segment_distance: int = 2,
) -> tuple[CandidateEdgePair, ...]:
    """Enumerate compatible ordered pairs without asking a model to invent endpoints."""
    window.validate()
    if not isinstance(nodes, tuple) or any(not isinstance(node, SemanticNode) for node in nodes):
        raise ValueError("semantic nodes must be an immutable tuple")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (max_character_distance, max_segment_distance)
    ):
        raise ValueError("pair pruning distances must be non-negative")
    node_by_id = {node.node_id: node for node in nodes}
    if len(node_by_id) != len(nodes):
        raise ValueError("semantic nodes must have unique IDs before pair generation")
    for node in nodes:
        node.source_span.validate_against(
            window.source_id, window.window_id, window.text, window_source_start=window.source_start,
            speaker=window.speaker, start_timestamp=window.start_ms, end_timestamp=window.end_ms,
        )
    pairs = []
    for source in nodes:
        for target in nodes:
            if source.node_id == target.node_id:
                continue
            allowed = _compatible_edges(source, target)
            if not allowed:
                continue
            character_distance = _span_distance(source.source_span, target.source_span)
            segment_distance = _segment_distance(window, source.source_span, target.source_span)
            if character_distance > max_character_distance or segment_distance > max_segment_distance:
                continue
            evidence = _evidence_for_pair(window, source.source_span, target.source_span)
            pair = CandidateEdgePair(
                pair_id=_pair_id(
                    window.window_id, source.node_id, target.node_id,
                    max_character_distance, max_segment_distance,
                ),
                window_id=window.window_id,
                source_node_id=source.node_id,
                target_node_id=target.node_id,
                allowed_edge_types=allowed,
                evidence_span=evidence,
                character_distance=character_distance,
                segment_distance=segment_distance,
                max_character_distance=max_character_distance,
                max_segment_distance=max_segment_distance,
            )
            pair.validate(window, node_by_id)
            pairs.append(pair)
    return tuple(sorted(pairs, key=lambda item: item.pair_id))


def edge_classification_prompt(pair: CandidateEdgePair, node_by_id: Mapping[str, SemanticNode]) -> str:
    source = node_by_id[pair.source_node_id]
    target = node_by_id[pair.target_node_id]
    allowed = [item.value for item in pair.allowed_edge_types]
    return (
        f"EVIDENCE:\n{pair.evidence_span.text}\n\n"
        f"A {source.node_type.value}: {source.source_span.text}\n"
        f"B {target.node_type.value}: {target.source_span.text}\n"
        f"Allowed directed A->B relations: {json.dumps(allowed)}\n"
        "Select every directed relation explicitly supported by this evidence. Return exactly "
        '{"status":"SUPPORTED|NO_RELATION|UNKNOWN|AMBIGUOUS|INSUFFICIENT_EVIDENCE",'
        '"edge_types":["<zero or more allowed>"],"confidence":0.0,'
        '"ambiguity":"NONE|UNKNOWN|AMBIGUOUS|MULTIPLE_CANDIDATES|INSUFFICIENT_EVIDENCE"}.'
    )


def classify_edge_pair(
    pair: CandidateEdgePair,
    window: SemanticSourceWindow,
    node_by_id: Mapping[str, SemanticNode],
    chat: Callable[..., str],
    *,
    model: str,
    configuration: Mapping[str, Any],
    max_tokens: int = 256,
    thinking: str | None = None,
) -> EdgeClassificationResult:
    """Run one constrained directed-pair decision with retained raw evidence."""
    pair.validate(window, node_by_id)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("edge classifier model must be nonempty")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("edge classifier max_tokens must be a positive integer")
    if thinking is not None and (not isinstance(thinking, str) or not thinking.strip()):
        raise ValueError("edge classifier thinking must be a nonempty string when present")
    if not isinstance(configuration, Mapping):
        raise ValueError("edge classifier configuration must be a mapping")
    prompt = edge_classification_prompt(pair, node_by_id)
    effective_configuration = {
        "caller_configuration": dict(configuration), "temperature": 0.0, "max_tokens": max_tokens,
        "model": model, "thinking": thinking, "prompt_version": EDGE_CLASSIFIER_PROMPT_VERSION,
        "pair_version": pair.version, "max_character_distance": pair.max_character_distance,
        "max_segment_distance": pair.max_segment_distance,
    }
    request = {"system": EDGE_CLASSIFIER_SYSTEM, "user": prompt, **effective_configuration}
    request_json = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    config_hash = content_sha256(effective_configuration)
    try:
        raw = chat(
            system=EDGE_CLASSIFIER_SYSTEM,
            user=prompt,
            temperature=0.0,
            max_tokens=max_tokens,
            model=model,
            thinking=thinking,
        )
    except Exception as exc:
        return EdgeClassificationResult(
            pair.pair_id, "INSUFFICIENT_EVIDENCE", (), "", None,
            f"{EdgeProviderError.__name__}:{type(exc).__name__}", None,
            request_json, model, config_hash,
        )
    try:
        status, edge_types, confidence, ambiguity, body = parse_edge_classification(raw, pair)
    except Exception as exc:
        retained_raw = raw if isinstance(raw, str) else repr(raw)
        return EdgeClassificationResult(
            pair.pair_id, "INSUFFICIENT_EVIDENCE", (), retained_raw, None, type(exc).__name__, None,
            request_json, model, config_hash,
        )
    if status != "SUPPORTED":
        return EdgeClassificationResult(
            pair.pair_id, status, (), raw, body, None, None, request_json, model, config_hash,
        )
    provenance = ModelDecisionProvenance(
        decision_id=pair.pair_id, model_id=model, prompt_version=EDGE_CLASSIFIER_PROMPT_VERSION,
        configuration_sha256=config_hash, input_sha256=content_sha256(request),
        output_sha256=content_sha256(raw),
        candidate_ids=(pair.pair_id, pair.source_node_id, pair.target_node_id),
    )
    edges = tuple(SemanticEdge(
        edge_type=edge_type, source_node_id=pair.source_node_id,
        target_node_id=pair.target_node_id, evidence=(pair.evidence_span,),
        provenance=provenance, qualifiers=SemanticQualifiers(), ambiguity=ambiguity,
        confidence=confidence, compiler_version=COMPILER_VERSION,
    ) for edge_type in edge_types)
    return EdgeClassificationResult(
        pair.pair_id, status, edges, raw, body, None, None, request_json, model, config_hash,
    )


def parse_edge_classification(
    raw: str,
    pair: CandidateEdgePair,
) -> tuple[str, tuple[EdgeType, ...], float, AmbiguityState, Mapping[str, Any]]:
    body = _strict_object(raw)
    if set(body) != {"status", "edge_types", "confidence", "ambiguity"}:
        raise ValueError("edge classification has an invalid shape")
    status = body.get("status")
    if status not in CLASSIFICATION_STATUSES:
        raise ValueError("edge classification status is invalid")
    edge_values = body.get("edge_types")
    confidence = body.get("confidence")
    ambiguity_value = body.get("ambiguity")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("edge classification confidence must be between zero and one")
    try:
        ambiguity = AmbiguityState(ambiguity_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("edge classification ambiguity is invalid") from exc
    if not isinstance(edge_values, list):
        raise ValueError("edge classification types must be a list")
    edge_types = []
    for edge_value in edge_values:
        try:
            edge_type = EdgeType(edge_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("edge classification type is invalid") from exc
        if edge_type not in pair.allowed_edge_types:
            raise ValueError("edge classification selected a relation outside the pair catalog")
        edge_types.append(edge_type)
    if len(set(edge_types)) != len(edge_types):
        raise ValueError("edge classification types must be unique")
    edge_set = set(edge_types)
    incompatible = (
        {EdgeType.TEMPORAL_BEFORE, EdgeType.TEMPORAL_AFTER},
        {EdgeType.ENABLES, EdgeType.PREVENTS},
        {EdgeType.CAUSES, EdgeType.PREVENTS},
    )
    if any(group <= edge_set for group in incompatible):
        raise ValueError("edge classification contains mutually incompatible relations")
    if status == "SUPPORTED":
        if not edge_types or ambiguity is not AmbiguityState.NONE or confidence < SUPPORTED_MIN_CONFIDENCE:
            raise ValueError("supported edges require unambiguous allowed relations above the acceptance threshold")
    elif edge_types:
        raise ValueError("non-supported classification cannot smuggle an edge")
    expected_ambiguity = {
        "NO_RELATION": {AmbiguityState.NONE},
        "UNKNOWN": {AmbiguityState.UNKNOWN},
        "AMBIGUOUS": {AmbiguityState.AMBIGUOUS, AmbiguityState.MULTIPLE_CANDIDATES},
        "INSUFFICIENT_EVIDENCE": {AmbiguityState.INSUFFICIENT_EVIDENCE},
    }
    if status != "SUPPORTED" and ambiguity not in expected_ambiguity[str(status)]:
        raise ValueError("edge classification status contradicts its ambiguity state")
    return str(status), tuple(sorted(edge_types, key=lambda item: item.value)), float(confidence), ambiguity, body


def classify_edge_catalog(
    window: SemanticSourceWindow,
    nodes: tuple[SemanticNode, ...],
    pairs: tuple[CandidateEdgePair, ...],
    chat: Callable[..., str],
    *,
    model: str,
    configuration: Mapping[str, Any],
    max_tokens: int = 256,
    thinking: str | None = None,
    max_character_distance: int = 600,
    max_segment_distance: int = 2,
) -> EdgeCatalogClassificationResult:
    """Classify every retained directed pair; no-relation and abstentions remain auditable."""
    node_by_id = {node.node_id: node for node in nodes}
    if not isinstance(pairs, tuple) or len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("edge catalog must be an immutable tuple with unique pair IDs")
    expected_pairs = generate_candidate_edge_pairs(
        window, nodes, max_character_distance=max_character_distance,
        max_segment_distance=max_segment_distance,
    )
    if pairs != expected_pairs:
        raise ValueError("edge catalog must equal the complete deterministic pair catalog")
    for pair in pairs:
        pair.validate(window, node_by_id)
    results = tuple(
        classify_edge_pair(
            pair, window, node_by_id, chat, model=model, configuration=configuration,
            max_tokens=max_tokens, thinking=thinking,
        )
        for pair in pairs
    )
    return assemble_edge_catalog_classification(
        window, nodes, pairs, results,
        max_character_distance=max_character_distance,
        max_segment_distance=max_segment_distance,
    )


def assemble_edge_catalog_classification(
    window: SemanticSourceWindow,
    nodes: tuple[SemanticNode, ...],
    pairs: tuple[CandidateEdgePair, ...],
    results: tuple[EdgeClassificationResult, ...],
    *,
    max_character_distance: int = 600,
    max_segment_distance: int = 2,
) -> EdgeCatalogClassificationResult:
    """Assemble already-retained pair decisions into a complete catalog run."""
    if not isinstance(results, tuple):
        raise ValueError("edge results must be an immutable tuple")
    edges = tuple(edge for result in results for edge in result.edges)
    failures = tuple(
        f"pair:{result.pair_id}:{result.failure}" for result in results if result.failure
    )
    abstentions = tuple(
        f"pair:{result.pair_id}:{result.status}" for result in results
        if not result.failure and result.status in {"UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"}
    )
    completed = sum(not result.failure and result.status in {"SUPPORTED", "NO_RELATION"} for result in results)
    if failures or abstentions:
        status = "PARTIAL" if completed else "INSUFFICIENT_EVIDENCE"
    else:
        status = "OK" if results else "NONE"
    run = EdgeCatalogClassificationResult(
        status, pairs, results, edges, max_character_distance, max_segment_distance,
        failures, abstentions,
    )
    _validate_edge_catalog_classification(window, nodes, run)
    return run


def assemble_semantic_graph(
    window: SemanticSourceWindow,
    nodes: tuple[SemanticNode, ...],
    classification: EdgeCatalogClassificationResult,
) -> SemanticGraph:
    """Assemble only edges reproducible from their retained pair decisions."""
    _validate_edge_catalog_classification(window, nodes, classification)
    return SemanticGraph.from_source_window(window, nodes, classification.edges)


def validate_edge_catalog_classification(
    window: SemanticSourceWindow,
    nodes: tuple[SemanticNode, ...],
    classification: EdgeCatalogClassificationResult,
) -> None:
    """Reconstructively validate a complete classification without assembling it.

    The compiler uses this when its dedicated coreference pass owns accepted
    ``REFERS_TO`` edges while the general edge pass remains retained as an
    independently auditable corroborating decision.
    """
    _validate_edge_catalog_classification(window, nodes, classification)


def _validate_edge_catalog_classification(
    window: SemanticSourceWindow,
    nodes: tuple[SemanticNode, ...],
    run: EdgeCatalogClassificationResult,
) -> None:
    node_by_id = {node.node_id: node for node in nodes}
    if not isinstance(run.pairs, tuple) or not isinstance(run.results, tuple) or not isinstance(run.edges, tuple):
        raise ValueError("edge classification catalogs must be immutable tuples")
    expected_pairs = generate_candidate_edge_pairs(
        window, nodes, max_character_distance=run.max_character_distance,
        max_segment_distance=run.max_segment_distance,
    )
    if run.pairs != expected_pairs:
        raise ValueError("retained edge pairs are not the complete deterministic catalog")
    if tuple(result.pair_id for result in run.results) != tuple(pair.pair_id for pair in run.pairs):
        raise ValueError("edge results do not exactly cover the retained pair catalog")
    if len({pair.pair_id for pair in run.pairs}) != len(run.pairs):
        raise ValueError("edge pair catalog IDs must be unique")
    pair_by_id = {}
    expected_edges = []
    expected_failures = []
    expected_abstentions = []
    completed = 0
    for pair, result in zip(run.pairs, run.results):
        pair_by_id[pair.pair_id] = pair
        reconstructed, failure, abstention, is_completed = _validate_edge_result(
            pair, window, node_by_id, result,
        )
        expected_edges.extend(reconstructed)
        if failure is not None:
            expected_failures.append(failure)
        if abstention is not None:
            expected_abstentions.append(abstention)
        completed += int(is_completed)
    if tuple(expected_edges) != run.edges:
        raise ValueError("aggregate semantic edges contradict pair decisions")
    if tuple(expected_failures) != run.failures or tuple(expected_abstentions) != run.abstentions:
        raise ValueError("aggregate edge failure/abstention evidence is inconsistent")
    if expected_failures or expected_abstentions:
        expected_status = "PARTIAL" if completed else "INSUFFICIENT_EVIDENCE"
    else:
        expected_status = "OK" if run.results else "NONE"
    if run.status != expected_status:
        raise ValueError("aggregate edge status contradicts pair decisions")


def validate_edge_classification_result(
    pair: CandidateEdgePair,
    window: SemanticSourceWindow,
    node_by_id: Mapping[str, SemanticNode],
    result: EdgeClassificationResult,
) -> None:
    """Reconstruct one retained pair decision without requiring a full catalog."""
    _validate_edge_result(pair, window, node_by_id, result)


def _validate_edge_result(
    pair: CandidateEdgePair,
    window: SemanticSourceWindow,
    node_by_id: Mapping[str, SemanticNode],
    result: EdgeClassificationResult,
) -> tuple[tuple[SemanticEdge, ...], str | None, str | None, bool]:
    pair.validate(window, node_by_id)
    if result.latency_ms is not None and (
        isinstance(result.latency_ms, bool)
        or not isinstance(result.latency_ms, int)
        or result.latency_ms < 0
    ):
        raise ValueError("edge result latency must be a non-negative integer or null")
    if result.pair_id != pair.pair_id:
        raise ValueError("edge result belongs to another pair")
    request = _strict_request(result.request_json)
    if request.get("system") != EDGE_CLASSIFIER_SYSTEM or request.get("user") != edge_classification_prompt(pair, node_by_id):
        raise ValueError("edge request contradicts its retained pair prompt")
    effective = {key: request[key] for key in (
        "caller_configuration", "temperature", "max_tokens", "model", "thinking",
        "prompt_version", "pair_version", "max_character_distance", "max_segment_distance",
    )}
    if (
        isinstance(request.get("temperature"), bool)
        or not isinstance(request.get("temperature"), (int, float))
        or request.get("temperature") != 0.0
        or request.get("prompt_version") != EDGE_CLASSIFIER_PROMPT_VERSION
        or request.get("pair_version") != pair.version
        or request.get("max_character_distance") != pair.max_character_distance
        or request.get("max_segment_distance") != pair.max_segment_distance
        or isinstance(request.get("max_tokens"), bool)
        or not isinstance(request.get("max_tokens"), int)
        or request["max_tokens"] <= 0
        or not isinstance(request.get("model"), str) or not request["model"].strip()
        or (request.get("thinking") is not None and (
            not isinstance(request["thinking"], str) or not request["thinking"].strip()
        ))
    ):
        raise ValueError("edge request constants/configuration are invalid")
    if result.model_id != request.get("model") or result.configuration_sha256 != content_sha256(effective):
        raise ValueError("edge effective request configuration is invalid")
    if result.failure:
        if result.status != "INSUFFICIENT_EVIDENCE" or result.edges or result.parsed_output is not None:
            raise ValueError("failed edge decision cannot retain accepted relations")
        if result.failure.startswith(f"{EdgeProviderError.__name__}:"):
            if result.raw_output != "" or not re.fullmatch(r"EdgeProviderError:[A-Za-z_][A-Za-z0-9_]*", result.failure):
                raise ValueError("provider failure evidence is inconsistent")
        else:
            if not isinstance(result.raw_output, str) or not result.raw_output:
                raise ValueError("model parse failure must retain nonempty raw output")
            try:
                parse_edge_classification(result.raw_output, pair)
            except Exception as exc:
                if result.failure != type(exc).__name__:
                    raise ValueError("model parse failure taxonomy is inconsistent") from exc
            else:
                raise ValueError("claimed model parse failure reparses successfully")
        return (), f"pair:{result.pair_id}:{result.failure}", None, False
    status, edge_types, confidence, ambiguity, body = parse_edge_classification(result.raw_output, pair)
    if (status, body) != (result.status, result.parsed_output):
        raise ValueError("edge raw output contradicts its parsed decision")
    provenance = ModelDecisionProvenance(
        pair.pair_id, result.model_id, EDGE_CLASSIFIER_PROMPT_VERSION,
        result.configuration_sha256, content_sha256(request), content_sha256(result.raw_output),
        (pair.pair_id, pair.source_node_id, pair.target_node_id),
    )
    reconstructed = tuple(SemanticEdge(
        edge_type, pair.source_node_id, pair.target_node_id, (pair.evidence_span,), provenance,
        SemanticQualifiers(), ambiguity, confidence, COMPILER_VERSION,
    ) for edge_type in edge_types)
    if reconstructed != result.edges:
        raise ValueError("edge objects contradict their retained raw decision")
    abstention = (
        f"pair:{result.pair_id}:{status}"
        if status in {"UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"} else None
    )
    return reconstructed, None, abstention, status in {"SUPPORTED", "NO_RELATION"}


def _strict_request(payload: str) -> Mapping[str, Any]:
    body = _strict_object(payload)
    expected = {
        "system", "user", "caller_configuration", "temperature", "max_tokens", "model", "thinking",
        "prompt_version", "pair_version", "max_character_distance", "max_segment_distance",
    }
    if set(body) != expected or not isinstance(body.get("caller_configuration"), Mapping):
        raise ValueError("edge request artifact has an invalid shape")
    return body


def candidate_pair_coverage(
    pairs: tuple[CandidateEdgePair, ...],
    reviewed: Iterable[tuple[str, str, EdgeType]],
    *,
    window: SemanticSourceWindow,
    nodes: tuple[SemanticNode, ...],
) -> dict[str, int | float]:
    """Measure whether each reviewed directed edge was offered before classification."""
    if not isinstance(pairs, tuple) or not isinstance(nodes, tuple):
        raise ValueError("pair coverage requires immutable pair and node catalogs")
    node_by_id = {node.node_id: node for node in nodes}
    if len(node_by_id) != len(nodes):
        raise ValueError("pair coverage requires unique semantic nodes")
    for pair in pairs:
        if not isinstance(pair, CandidateEdgePair):
            raise ValueError("pair coverage catalog contains an invalid value")
        pair.validate(window, node_by_id)
    offered = {
        (pair.source_node_id, pair.target_node_id, edge_type)
        for pair in pairs for edge_type in pair.allowed_edge_types
    }
    labels = tuple(reviewed)
    if len(set(labels)) != len(labels):
        raise ValueError("reviewed edge labels must be unique")
    if any(
        not isinstance(source_id, str) or source_id not in node_by_id
        or not isinstance(target_id, str) or target_id not in node_by_id
        or not isinstance(edge_type, EdgeType)
        for source_id, target_id, edge_type in labels
    ):
        raise ValueError("reviewed edge label is invalid")
    hits = sum(label in offered for label in labels)
    return {"hit_count": hits, "denominator": len(labels), "recall": hits / len(labels) if labels else 0.0}


def _compatible_edges(source: SemanticNode, target: SemanticNode) -> tuple[EdgeType, ...]:
    left, right = source.node_type.value, target.node_type.value
    values: set[EdgeType] = {EdgeType.CONTRASTS_WITH, EdgeType.MODIFIES}
    occurrences = {"EVENT", "ACTION", "STATE", "OUTCOME"}
    if left == "ENTITY" and right in {"EVENT", "ACTION", "STATE", "OUTCOME"}:
        values.update({EdgeType.ACTOR, EdgeType.EXPERIENCER})
    if left in {"EVENT", "ACTION"} and right in {"ENTITY", "ABILITY_OR_RESOURCE", "LOCATION_OR_SPACE"}:
        values.update({EdgeType.TARGET, EdgeType.OBJECT})
        if right == "ABILITY_OR_RESOURCE":
            values.add(EdgeType.REQUIRES)
    if left == "ABILITY_OR_RESOURCE" and right in {"EVENT", "ACTION", "STATE"}:
        values.add(EdgeType.OBJECT)
    if left in occurrences and right in occurrences:
        values.update({
            EdgeType.CAUSES, EdgeType.ENABLES, EdgeType.PREVENTS, EdgeType.REQUIRES,
            EdgeType.PURPOSE, EdgeType.RESULT, EdgeType.TEMPORAL_BEFORE,
            EdgeType.TEMPORAL_AFTER, EdgeType.TEMPORAL_UNTIL, EdgeType.TERMINATES,
        })
    if left in occurrences | {"TIME"} and right in occurrences | {"TIME"}:
        values.update({EdgeType.TEMPORAL_BEFORE, EdgeType.TEMPORAL_AFTER, EdgeType.TEMPORAL_UNTIL})
    if left in set(item.value for item in NodeType) - {"ENTITY"} and right in occurrences:
        values.add(EdgeType.CONDITION)
    if left == "TIME" and right in occurrences:
        values.add(EdgeType.TERMINATES)
    if left in {"TIME", "QUANTITY", "STATE", "OUTCOME"}:
        values.update({EdgeType.MODIFIES, EdgeType.NEGATES})
    if (
        source.referent_candidates
        and target.node_id in source.referent_candidate_node_ids
        and target.source_span in source.referent_candidates
    ):
        values.add(EdgeType.REFERS_TO)
    return tuple(sorted(
        (item for item in values if edge_type_supports(item, source.node_type, target.node_type)),
        key=lambda item: item.value,
    ))


def _is_reference_expression(text: str) -> bool:
    return text.strip().casefold().replace("’", "'") in {
        "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours",
        "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their",
        "theirs", "this", "that", "these", "those", "this wave", "that action", "this event",
    }


def _span_distance(left: SourceSpan, right: SourceSpan) -> int:
    if left.local_end < right.local_start:
        return right.local_start - left.local_end
    if right.local_end < left.local_start:
        return left.local_start - right.local_end
    return 0


def _segment_distance(window: SemanticSourceWindow, left: SourceSpan, right: SourceSpan) -> int:
    def indexes(span: SourceSpan) -> tuple[int, ...]:
        return tuple(index for index, segment in enumerate(window.segments)
                     if segment.start < span.local_end and span.local_start < segment.end)
    left_indexes, right_indexes = indexes(left), indexes(right)
    if not left_indexes or not right_indexes:
        raise ValueError("semantic node lies outside deterministic Pass 0 segments")
    return min(abs(a - b) for a in left_indexes for b in right_indexes)


def _pair_id(
    window_id: str, source_node_id: str, target_node_id: str,
    max_character_distance: int, max_segment_distance: int,
) -> str:
    raw = (
        f"{EDGE_PAIR_VERSION}:{window_id}:{source_node_id}:{target_node_id}:"
        f"{max_character_distance}:{max_segment_distance}"
    ).encode("utf-8")
    return f"{window_id}:p{hashlib.sha256(raw).hexdigest()[:20]}"


def _evidence_for_pair(
    window: SemanticSourceWindow, left: SourceSpan, right: SourceSpan,
) -> SourceSpan:
    relevant = tuple(
        segment for segment in window.segments
        if (segment.start < left.local_end and left.local_start < segment.end)
        or (segment.start < right.local_end and right.local_start < segment.end)
    )
    if not relevant:
        raise ValueError("candidate edge endpoints have no Pass 0 discourse evidence")
    start, end = min(item.start for item in relevant), max(item.end for item in relevant)
    return SourceSpan(
        window.source_id, window.window_id, start, end, window.text[start:end],
        window.source_start + start, window.source_start + end, window.speaker,
        window.start_ms, window.end_ms,
    )


def _strict_object(raw: str) -> Mapping[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON contains duplicate keys")
            result[key] = value
        return result
    if not isinstance(raw, str):
        raise ValueError("edge classification output must be a string")
    try:
        body = json.loads(raw, object_pairs_hook=unique)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("edge classification returned malformed JSON") from exc
    if not isinstance(body, Mapping):
        raise ValueError("edge classification must be a JSON object")
    return body
