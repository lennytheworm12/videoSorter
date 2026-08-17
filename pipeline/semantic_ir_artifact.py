"""Deterministic, proof-carrying artifacts for Phase 2F compiler runs."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Mapping

from pipeline.semantic_compiler import (
    CompilerFailure, NodeQualifierRun, SemanticCompileRun, SemanticCompilerConfig,
)
from pipeline.semantic_coreference import (
    CoreferenceCandidateSet, CoreferenceCatalogResult, CoreferenceDecision,
)
from pipeline.semantic_edges import (
    CandidateEdgePair, EdgeCatalogClassificationResult, EdgeClassificationResult,
)
from pipeline.semantic_ir import (
    SCHEMA_VERSION, EdgeType, ModelDecisionProvenance, QualifierKind,
    SemanticEdge, SemanticGraph, SemanticNode, SemanticQualifiers, SourceSpan,
)
from pipeline.semantic_mentions import (
    MentionCandidate, MentionCatalogSelectionResult, MentionSelection,
    MentionSelectionResult,
)
from pipeline.semantic_qualifiers import (
    QualifierCandidate, QualifierFieldSelection, QualifierSelectionResult,
)
from pipeline.semantic_source import PASS0_VERSION, SemanticSourceWindow, SourceSegment


RUN_ARTIFACT_VERSION = "phase2f-semantic-run-artifact-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class SemanticRunArtifact:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        validate_semantic_run_artifact(self.payload)

    @property
    def content_sha256(self) -> str:
        return str(self.payload["content_sha256"])

    def to_json(self) -> str:
        """Return the one canonical byte representation covered by file_sha256."""
        return json.dumps(
            self.payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )

    @property
    def file_sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, value: str) -> "SemanticRunArtifact":
        if not isinstance(value, str):
            raise ValueError("semantic run artifact JSON must be a string")

        def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("semantic run artifact JSON contains duplicate keys")
                result[key] = item
            return result

        try:
            body = json.loads(value, object_pairs_hook=unique)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("semantic run artifact JSON is malformed") from exc
        if not isinstance(body, Mapping):
            raise ValueError("semantic run artifact must be a JSON object")
        return cls(body)


def build_semantic_run_artifact(
    run: SemanticCompileRun,
    *,
    git_commit: str,
    repository_dirty: bool,
    created_at: str,
    input_hashes: Mapping[str, str] | None = None,
    evaluation: Mapping[str, Any] | None = None,
) -> SemanticRunArtifact:
    if not isinstance(run, SemanticCompileRun):
        raise ValueError("semantic artifact requires a typed compiler run")
    run.validate()
    required_inputs = {
        "bronze_source_content_sha256": run.window.source_content_sha256,
        "bronze_source_provenance_sha256": run.window.source_provenance_sha256,
        "source_window_sha256": _canonical_sha256(_window_to_dict(run.window)),
    }
    for key, value in (input_hashes or {}).items():
        if key in required_inputs:
            raise ValueError("caller input hash collides with a required source hash")
        required_inputs[key] = value
    payload = {
        "artifact_version": RUN_ARTIFACT_VERSION,
        "semantic_ir_schema_version": SCHEMA_VERSION,
        "pass0_version": PASS0_VERSION,
        "git_commit": git_commit,
        "repository_dirty": repository_dirty,
        "created_at": created_at,
        "input_hashes": dict(sorted(required_inputs.items())),
        "run": _run_to_dict(run),
        "evaluation": _jsonable(dict(evaluation)) if evaluation is not None else None,
    }
    final = {"content_sha256": _canonical_sha256(payload), **payload}
    return SemanticRunArtifact(final)


def validate_semantic_run_artifact(value: Mapping[str, Any]) -> None:
    expected = {
        "content_sha256", "artifact_version", "semantic_ir_schema_version", "pass0_version",
        "git_commit", "repository_dirty", "created_at", "input_hashes", "run", "evaluation",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("semantic run artifact has an invalid envelope")
    if value["artifact_version"] != RUN_ARTIFACT_VERSION:
        raise ValueError("semantic run artifact version is unsupported")
    if value["semantic_ir_schema_version"] != SCHEMA_VERSION or value["pass0_version"] != PASS0_VERSION:
        raise ValueError("semantic run artifact compiler schema/version is unsupported")
    if not isinstance(value["git_commit"], str) or _COMMIT.fullmatch(value["git_commit"]) is None:
        raise ValueError("semantic run artifact git commit must be a full lowercase revision")
    if not isinstance(value["repository_dirty"], bool):
        raise ValueError("semantic run artifact dirty state must be boolean")
    if not isinstance(value["created_at"], str):
        raise ValueError("semantic run artifact timestamp must be a string")
    try:
        parsed_time = datetime.fromisoformat(value["created_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("semantic run artifact timestamp must be ISO-8601") from exc
    if parsed_time.tzinfo is None:
        raise ValueError("semantic run artifact timestamp must include a timezone")
    input_hashes = value["input_hashes"]
    if not isinstance(input_hashes, Mapping) or set(input_hashes) < {
        "bronze_source_content_sha256", "bronze_source_provenance_sha256", "source_window_sha256",
    } or any(
        not isinstance(key, str) or not key or not isinstance(item, str) or _SHA256.fullmatch(item) is None
        for key, item in input_hashes.items()
    ):
        raise ValueError("semantic run artifact input hashes are invalid")
    inner = {key: item for key, item in value.items() if key != "content_sha256"}
    if value["content_sha256"] != _canonical_sha256(inner):
        raise ValueError("semantic run artifact content hash does not match its content")
    if value["evaluation"] is not None and not isinstance(value["evaluation"], Mapping):
        raise ValueError("semantic run artifact evaluation must be an object or null")
    run = value["run"]
    typed_run = _run_from_dict(run)
    typed_run.validate()
    if _run_to_dict(typed_run) != run:
        raise ValueError("semantic run artifact is not in canonical typed form")
    if run["status"] != typed_run.status:
        raise ValueError("semantic run artifact status contradicts the reconstructed run")
    required_expected = {
        "bronze_source_content_sha256": typed_run.window.source_content_sha256,
        "bronze_source_provenance_sha256": typed_run.window.source_provenance_sha256,
        "source_window_sha256": _canonical_sha256(_window_to_dict(typed_run.window)),
    }
    if any(input_hashes.get(key) != expected for key, expected in required_expected.items()):
        raise ValueError("semantic run artifact source input hashes contradict its window")


def _run_to_dict(run: SemanticCompileRun) -> dict[str, Any]:
    return {
        "status": run.status,
        "window": _window_to_dict(run.window),
        "config": _jsonable(run.config),
        "entity_aliases": list(run.entity_aliases),
        "ability_aliases": list(run.ability_aliases),
        "mention_catalog": _jsonable(run.mention_catalog),
        "mention_selection": _jsonable(run.mention_selection),
        "mention_nodes": [item.to_dict() for item in run.mention_nodes],
        "qualifier_catalog": _jsonable(run.qualifier_catalog),
        "qualifier_runs": _jsonable(run.qualifier_runs),
        "qualified_nodes": [item.to_dict() for item in run.qualified_nodes],
        "coreference_candidate_sets": _jsonable(run.coreference_candidate_sets),
        "coreference_decisions": _jsonable(run.coreference_decisions),
        "coreference": _jsonable(run.coreference),
        "edge_pairs": _jsonable(run.edge_pairs),
        "edge_results": _jsonable(run.edge_results),
        "edge_classification": _jsonable(run.edge_classification),
        "merged_edges": [item.to_dict() for item in run.merged_edges],
        "graph": run.graph.to_artifact() if run.graph is not None else None,
        "failures": _jsonable(run.failures),
        "integrity_sha256": run.integrity_sha256,
        "version": run.version,
    }


def _window_to_dict(window: SemanticSourceWindow) -> dict[str, Any]:
    window.validate()
    return _jsonable(window)


def _window_from_dict(value: Mapping[str, Any]) -> SemanticSourceWindow:
    expected = {
        "window_id", "source_id", "source_kind", "source_start", "source_end", "text",
        "source_content_sha256", "source_provenance_sha256", "source_context_sha256",
        "speaker", "start_ms", "end_ms", "metadata", "segments", "version",
    }
    if set(value) != expected or not isinstance(value["metadata"], list) or not isinstance(value["segments"], list):
        raise ValueError("semantic run source window artifact is invalid")
    segments = []
    for raw in value["segments"]:
        if not isinstance(raw, Mapping) or set(raw) != {
            "segment_id", "window_id", "kind", "start", "end", "absolute_start", "absolute_end",
            "source_text", "version",
        }:
            raise ValueError("semantic run source segment artifact is invalid")
        segments.append(SourceSegment(**raw))
    metadata = tuple(tuple(item) for item in value["metadata"])
    window = SemanticSourceWindow(
        value["window_id"], value["source_id"], value["source_kind"], value["source_start"],
        value["source_end"], value["text"], value["source_content_sha256"],
        value["source_provenance_sha256"], value["source_context_sha256"], value["speaker"],
        value["start_ms"], value["end_ms"], metadata, tuple(segments), value["version"],
    )
    window.validate()
    return window


def _validate_run_payload(value: object) -> None:
    _run_from_dict(value)


def _run_from_dict(value: object) -> SemanticCompileRun:
    expected = {
        "status", "window", "config", "entity_aliases", "ability_aliases",
        "mention_catalog", "mention_selection", "mention_nodes",
        "qualifier_catalog", "qualifier_runs", "qualified_nodes", "coreference",
        "coreference_candidate_sets", "coreference_decisions", "edge_pairs", "edge_results",
        "edge_classification", "merged_edges", "graph", "failures",
        "integrity_sha256", "version",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("semantic run artifact payload has an invalid shape")
    if value["status"] not in {"OK", "PARTIAL", "NONE", "FAILURE"}:
        raise ValueError("semantic run artifact status is invalid")
    window_raw = value["window"]
    if not isinstance(window_raw, Mapping):
        raise ValueError("semantic run artifact lacks a source window")
    window = _window_from_dict(window_raw)
    for collection in (
        "entity_aliases", "ability_aliases", "mention_catalog", "mention_nodes",
        "qualifier_catalog", "qualifier_runs",
        "qualified_nodes", "coreference_candidate_sets", "coreference_decisions",
        "edge_pairs", "edge_results", "merged_edges", "failures",
    ):
        if not isinstance(value[collection], list):
            raise ValueError(f"semantic run artifact {collection} must be a list")
    graph_raw = value["graph"]
    if graph_raw is not None:
        if not isinstance(graph_raw, Mapping):
            raise ValueError("semantic run graph artifact must be an object or null")
        graph = SemanticGraph.from_artifact(graph_raw)
        graph.validate_against_source_window(window)
    if value["status"] == "FAILURE" and graph_raw is not None:
        raise ValueError("failed semantic run cannot retain an accepted graph")
    if value["status"] != "FAILURE" and graph_raw is None:
        raise ValueError("nonfailed semantic run must retain its graph")
    config = _config_from_dict(_mapping(value["config"], "compiler config"))
    mention_catalog = tuple(
        _mention_candidate(item) for item in _list(value["mention_catalog"], "mention catalog")
    )
    mention_selection = (
        None if value["mention_selection"] is None
        else _mention_catalog_result(_mapping(value["mention_selection"], "mention selection"))
    )
    mention_nodes = tuple(
        SemanticNode.from_dict(_mapping(item, "mention node"))
        for item in _list(value["mention_nodes"], "mention nodes")
    )
    qualifier_catalog = tuple(
        _qualifier_candidate(item)
        for item in _list(value["qualifier_catalog"], "qualifier catalog")
    )
    qualifier_runs = tuple(
        _qualifier_run(item) for item in _list(value["qualifier_runs"], "qualifier runs")
    )
    qualified_nodes = tuple(
        SemanticNode.from_dict(_mapping(item, "qualified node"))
        for item in _list(value["qualified_nodes"], "qualified nodes")
    )
    coreference_sets = tuple(
        _coreference_set(item)
        for item in _list(value["coreference_candidate_sets"], "coreference candidate sets")
    )
    coreference_decisions = tuple(
        _coreference_decision(item)
        for item in _list(value["coreference_decisions"], "coreference decisions")
    )
    coreference = (
        None if value["coreference"] is None
        else _coreference_result(_mapping(value["coreference"], "coreference result"))
    )
    edge_pairs = tuple(
        _edge_pair(item) for item in _list(value["edge_pairs"], "edge pairs")
    )
    edge_results = tuple(
        _edge_result(item) for item in _list(value["edge_results"], "edge results")
    )
    edge_classification = (
        None if value["edge_classification"] is None
        else _edge_catalog_result(_mapping(value["edge_classification"], "edge classification"))
    )
    merged_edges = tuple(
        SemanticEdge.from_dict(_mapping(item, "merged edge"))
        for item in _list(value["merged_edges"], "merged edges")
    )
    failures = tuple(
        CompilerFailure(**_exact_mapping(item, {"stage", "code", "item_id", "detail"}, "compiler failure"))
        for item in _list(value["failures"], "compiler failures")
    )
    run = SemanticCompileRun(
        window=window, config=config,
        entity_aliases=tuple(_string_list(value["entity_aliases"], "entity aliases")),
        ability_aliases=tuple(_string_list(value["ability_aliases"], "ability aliases")),
        mention_catalog=mention_catalog, mention_selection=mention_selection,
        mention_nodes=mention_nodes, qualifier_catalog=qualifier_catalog,
        qualifier_runs=qualifier_runs, qualified_nodes=qualified_nodes,
        coreference_candidate_sets=coreference_sets,
        coreference_decisions=coreference_decisions, coreference=coreference,
        edge_pairs=edge_pairs, edge_results=edge_results,
        edge_classification=edge_classification, merged_edges=merged_edges,
        graph=(None if graph_raw is None else SemanticGraph.from_artifact(graph_raw)),
        failures=failures, integrity_sha256=value["integrity_sha256"],
        version=value["version"],
    )
    run.validate()
    if value["status"] != run.status:
        raise ValueError("serialized compiler status contradicts its typed run")
    return run


def _config_from_dict(value: Mapping[str, Any]) -> SemanticCompilerConfig:
    expected = {
        "model", "provider_configuration_json", "thinking", "mention_partition_size",
        "mention_max_tokens", "qualifier_max_tokens", "coreference_max_tokens",
        "edge_max_tokens", "coreference_max_segment_distance",
        "edge_max_character_distance", "edge_max_segment_distance", "version",
    }
    return SemanticCompilerConfig(**_exact_mapping(value, expected, "compiler config"))


def _mention_candidate(value: object) -> MentionCandidate:
    raw = _exact_mapping(value, {
        "candidate_id", "window_id", "start", "end", "absolute_start", "absolute_end",
        "source_text", "type_hints", "segment_ids", "version",
    }, "mention candidate")
    raw["type_hints"] = tuple(_string_list(raw["type_hints"], "mention type hints"))
    raw["segment_ids"] = tuple(_string_list(raw["segment_ids"], "mention segment IDs"))
    return MentionCandidate(**raw)


def _mention_selection(value: object) -> MentionSelection:
    return MentionSelection(**_exact_mapping(
        value, {"candidate_id", "node_type", "confidence", "ambiguity"}, "mention selection",
    ))


def _mention_partition_result(value: object) -> MentionSelectionResult:
    raw = _exact_mapping(value, {
        "status", "mentions", "raw_output", "parsed_output", "failure", "candidate_ids",
        "prompt", "model_id", "configuration_sha256", "request_json",
    }, "mention partition result")
    raw["mentions"] = tuple(_mention_selection(item) for item in _list(raw["mentions"], "mention selections"))
    raw["candidate_ids"] = tuple(_string_list(raw["candidate_ids"], "mention candidate IDs"))
    return MentionSelectionResult(**raw)


def _mention_catalog_result(value: Mapping[str, Any]) -> MentionCatalogSelectionResult:
    raw = _exact_mapping(value, {
        "status", "catalog", "partition_results", "mentions", "failures", "abstentions",
    }, "mention catalog result")
    raw["catalog"] = tuple(_mention_candidate(item) for item in _list(raw["catalog"], "mention catalog"))
    raw["partition_results"] = tuple(
        _mention_partition_result(item) for item in _list(raw["partition_results"], "mention partitions")
    )
    raw["mentions"] = tuple(_mention_selection(item) for item in _list(raw["mentions"], "mentions"))
    raw["failures"] = tuple(_string_list(raw["failures"], "mention failures"))
    raw["abstentions"] = tuple(_string_list(raw["abstentions"], "mention abstentions"))
    return MentionCatalogSelectionResult(**raw)


def _qualifier_candidate(value: object) -> QualifierCandidate:
    raw = _exact_mapping(value, {
        "candidate_id", "window_id", "kind", "start", "end", "absolute_start",
        "absolute_end", "source_text", "version",
    }, "qualifier candidate")
    raw["kind"] = QualifierKind(raw["kind"])
    return QualifierCandidate(**raw)


def _qualifier_field(value: object) -> QualifierFieldSelection:
    raw = _exact_mapping(value, {
        "status", "value", "cue_ids", "confidence", "candidate_values",
    }, "qualifier field")
    raw["cue_ids"] = tuple(_string_list(raw["cue_ids"], "qualifier cue IDs"))
    raw["candidate_values"] = tuple(_string_list(raw["candidate_values"], "qualifier candidate values"))
    return QualifierFieldSelection(**raw)


def _qualifier_result(value: object) -> QualifierSelectionResult:
    raw = _exact_mapping(value, {
        "node_id", "status", "fields", "raw_output", "parsed_output", "candidate_ids",
        "request_json", "model_id", "configuration_sha256", "failure",
    }, "qualifier result")
    fields_value = _list(raw["fields"], "qualifier fields")
    parsed_fields = []
    for item in fields_value:
        if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
            raise ValueError("qualifier result field entry is invalid")
        parsed_fields.append((item[0], _qualifier_field(item[1])))
    raw["fields"] = tuple(parsed_fields)
    raw["candidate_ids"] = tuple(_string_list(raw["candidate_ids"], "qualifier candidate IDs"))
    return QualifierSelectionResult(**raw)


def _qualifier_run(value: object) -> NodeQualifierRun:
    raw = _exact_mapping(value, {
        "node_id", "candidates", "result", "output_node", "application_failure",
    }, "qualifier run")
    return NodeQualifierRun(
        raw["node_id"],
        tuple(_qualifier_candidate(item) for item in _list(raw["candidates"], "qualifier candidates")),
        _qualifier_result(raw["result"]),
        None if raw["output_node"] is None else SemanticNode.from_dict(_mapping(raw["output_node"], "qualifier output node")),
        raw["application_failure"],
    )


def _coreference_set(value: object) -> CoreferenceCandidateSet:
    raw = _exact_mapping(value, {
        "candidate_set_id", "window_id", "source_node_id", "target_node_ids",
        "evidence_span", "max_segment_distance", "version",
    }, "coreference candidate set")
    raw["target_node_ids"] = tuple(_string_list(raw["target_node_ids"], "coreference targets"))
    raw["evidence_span"] = SourceSpan.from_dict(_mapping(raw["evidence_span"], "coreference evidence"))
    return CoreferenceCandidateSet(**raw)


def _coreference_decision(value: object) -> CoreferenceDecision:
    raw = _exact_mapping(value, {
        "candidate_set_id", "status", "target_node_id", "candidate_node_ids", "confidence",
        "raw_output", "parsed_output", "request_json", "model_id", "configuration_sha256", "failure",
    }, "coreference decision")
    raw["candidate_node_ids"] = tuple(_string_list(raw["candidate_node_ids"], "coreference candidates"))
    return CoreferenceDecision(**raw)


def _coreference_result(value: Mapping[str, Any]) -> CoreferenceCatalogResult:
    raw = _exact_mapping(value, {
        "status", "candidate_sets", "decisions", "nodes", "edges",
        "max_segment_distance", "failures", "abstentions",
    }, "coreference result")
    raw["candidate_sets"] = tuple(_coreference_set(item) for item in _list(raw["candidate_sets"], "coreference sets"))
    raw["decisions"] = tuple(_coreference_decision(item) for item in _list(raw["decisions"], "coreference decisions"))
    raw["nodes"] = tuple(SemanticNode.from_dict(_mapping(item, "coreference node")) for item in _list(raw["nodes"], "coreference nodes"))
    raw["edges"] = tuple(SemanticEdge.from_dict(_mapping(item, "coreference edge")) for item in _list(raw["edges"], "coreference edges"))
    raw["failures"] = tuple(_string_list(raw["failures"], "coreference failures"))
    raw["abstentions"] = tuple(_string_list(raw["abstentions"], "coreference abstentions"))
    return CoreferenceCatalogResult(**raw)


def _edge_pair(value: object) -> CandidateEdgePair:
    raw = _exact_mapping(value, {
        "pair_id", "window_id", "source_node_id", "target_node_id", "allowed_edge_types",
        "evidence_span", "character_distance", "segment_distance", "max_character_distance",
        "max_segment_distance", "version",
    }, "edge pair")
    raw["allowed_edge_types"] = tuple(EdgeType(item) for item in _list(raw["allowed_edge_types"], "allowed edge types"))
    raw["evidence_span"] = SourceSpan.from_dict(_mapping(raw["evidence_span"], "edge-pair evidence"))
    return CandidateEdgePair(**raw)


def _edge_result(value: object) -> EdgeClassificationResult:
    raw = _exact_mapping(value, {
        "pair_id", "status", "edges", "raw_output", "parsed_output", "failure",
        "latency_ms", "request_json", "model_id", "configuration_sha256",
    }, "edge result")
    raw["edges"] = tuple(SemanticEdge.from_dict(_mapping(item, "classified edge")) for item in _list(raw["edges"], "classified edges"))
    return EdgeClassificationResult(**raw)


def _edge_catalog_result(value: Mapping[str, Any]) -> EdgeCatalogClassificationResult:
    raw = _exact_mapping(value, {
        "status", "pairs", "results", "edges", "max_character_distance",
        "max_segment_distance", "failures", "abstentions",
    }, "edge catalog result")
    raw["pairs"] = tuple(_edge_pair(item) for item in _list(raw["pairs"], "edge pairs"))
    raw["results"] = tuple(_edge_result(item) for item in _list(raw["results"], "edge results"))
    raw["edges"] = tuple(SemanticEdge.from_dict(_mapping(item, "aggregate edge")) for item in _list(raw["edges"], "aggregate edges"))
    raw["failures"] = tuple(_string_list(raw["failures"], "edge failures"))
    raw["abstentions"] = tuple(_string_list(raw["abstentions"], "edge abstentions"))
    return EdgeCatalogClassificationResult(**raw)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"semantic run artifact {label} must be an object")
    return value


def _exact_mapping(
    value: object, keys: set[str], label: str,
) -> dict[str, Any]:
    raw = _mapping(value, label)
    if set(raw) != keys:
        raise ValueError(f"semantic run artifact {label} has an invalid shape")
    return dict(raw)


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"semantic run artifact {label} must be a list")
    return value


def _string_list(value: object, label: str) -> list[str]:
    items = _list(value, label)
    if any(not isinstance(item, str) for item in items):
        raise ValueError(f"semantic run artifact {label} must contain strings")
    return items


def _jsonable(value: Any) -> Any:
    if isinstance(value, SemanticGraph):
        return value.to_artifact()
    if isinstance(value, SemanticNode):
        return value.to_dict()
    if isinstance(value, SemanticEdge):
        return value.to_dict()
    if isinstance(value, (SourceSpan, SemanticQualifiers, ModelDecisionProvenance)):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("semantic run artifact cannot serialize non-finite floats")
        return value
    raise ValueError(f"semantic run artifact cannot serialize {type(value).__name__}")


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "RUN_ARTIFACT_VERSION", "SemanticRunArtifact", "build_semantic_run_artifact",
    "validate_semantic_run_artifact",
]
