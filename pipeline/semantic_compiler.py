"""End-to-end bronze-window to source-semantic IR orchestration.

This module composes the independently testable Pass 0/Pass 1 boundaries. It
does not canonicalize claims, normalize domain concepts, or persist strategic
relations. Every intermediate model decision remains in ``SemanticCompileRun``
and successful runs can be reconstructed from their retained evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import re
from typing import Any, Callable, Iterable, Mapping

from pipeline.semantic_coreference import (
    CoreferenceCandidateSet, CoreferenceCatalogResult, CoreferenceDecision,
    apply_coreference_decision, assemble_coreference_catalog, classify_coreference,
    generate_coreference_candidate_sets,
)
from pipeline.semantic_edges import (
    CandidateEdgePair, EdgeCatalogClassificationResult, EdgeClassificationResult,
    assemble_edge_catalog_classification, classify_edge_pair,
    generate_candidate_edge_pairs, validate_edge_catalog_classification,
    validate_edge_classification_result,
)
from pipeline.semantic_ir import EdgeType, SemanticEdge, SemanticGraph, SemanticNode
from pipeline.semantic_mentions import (
    MENTION_SELECTION_PROMPT_VERSION,
    MENTION_SELECTION_PROMPT_VERSION_LEGACY,
    MentionCandidate, MentionCatalogSelectionResult, assemble_semantic_nodes,
    generate_mention_candidates, partition_candidate_catalog, select_mention_catalog,
)
from pipeline.semantic_qualifiers import (
    QualifierCandidate, QualifierSelectionResult, apply_node_qualifiers,
    classify_node_qualifiers, generate_qualifier_candidates,
    qualifier_candidates_for_node,
    validate_qualifier_selection_result,
)
from pipeline.semantic_source import SemanticSourceWindow


COMPILER_ORCHESTRATION_VERSION_LEGACY = "phase2f-semantic-compiler-orchestration-v2"
COMPILER_ORCHESTRATION_VERSION = "phase2f-semantic-compiler-orchestration-v3-focal-mentions"
_SUPPORTED_COMPILER_ORCHESTRATION_VERSIONS = frozenset({
    COMPILER_ORCHESTRATION_VERSION_LEGACY, COMPILER_ORCHESTRATION_VERSION,
})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FAILURE_CODES_BY_STAGE = {
    "mention_catalog": {"ASSEMBLY_FAILURE"},
    "mentions": {
        "PROVIDER_FAILURE", "MODEL_PARSE_FAILURE", "MENTION_TYPE_ERROR",
        "UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE", "ASSEMBLY_FAILURE",
    },
    "mention_assembly": {"ASSEMBLY_FAILURE", "MENTION_TYPE_ERROR"},
    "qualifier_catalog": {"ASSEMBLY_FAILURE"},
    "qualifiers": {
        "PROVIDER_FAILURE", "MODEL_PARSE_FAILURE", "UNKNOWN", "AMBIGUOUS",
        "INSUFFICIENT_EVIDENCE", "ASSEMBLY_FAILURE",
    },
    "coreference_catalog": {"ASSEMBLY_FAILURE"},
    "coreference": {
        "PROVIDER_FAILURE", "MODEL_PARSE_FAILURE", "REFERENCE_RESOLUTION_ERROR",
        "UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE", "ASSEMBLY_FAILURE",
    },
    "edge_catalog": {"ASSEMBLY_FAILURE"},
    "edges": {
        "PROVIDER_FAILURE", "MODEL_PARSE_FAILURE", "UNSUPPORTED_EDGE",
        "UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE", "ASSEMBLY_FAILURE",
    },
    "assembly": {
        "ASSEMBLY_FAILURE", "CONDITION_LOSS", "TEMPORAL_LOSS",
        "REFERENCE_RESOLUTION_ERROR",
    },
}


@dataclass(frozen=True)
class SemanticCompilerConfig:
    model: str
    provider_configuration_json: str
    thinking: str | None = None
    mention_partition_size: int = 180
    mention_max_tokens: int = 2048
    qualifier_max_tokens: int = 512
    coreference_max_tokens: int = 256
    edge_max_tokens: int = 256
    coreference_max_segment_distance: int = 2
    edge_max_character_distance: int = 600
    edge_max_segment_distance: int = 2
    version: str = COMPILER_ORCHESTRATION_VERSION

    def __post_init__(self) -> None:
        if self.version not in _SUPPORTED_COMPILER_ORCHESTRATION_VERSIONS:
            raise ValueError("semantic compiler configuration version is unsupported")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("semantic compiler model must be non-empty")
        configuration = _strict_json_object(
            self.provider_configuration_json, "provider configuration",
        )
        if _canonical_json(configuration) != self.provider_configuration_json:
            raise ValueError("provider configuration must use canonical JSON")
        if self.thinking is not None and (
            not isinstance(self.thinking, str) or not self.thinking.strip()
        ):
            raise ValueError("semantic compiler thinking mode is invalid")
        for value, label in (
            (self.mention_partition_size, "mention_partition_size"),
            (self.mention_max_tokens, "mention_max_tokens"),
            (self.qualifier_max_tokens, "qualifier_max_tokens"),
            (self.coreference_max_tokens, "coreference_max_tokens"),
            (self.edge_max_tokens, "edge_max_tokens"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        for value, label in (
            (self.coreference_max_segment_distance, "coreference_max_segment_distance"),
            (self.edge_max_character_distance, "edge_max_character_distance"),
            (self.edge_max_segment_distance, "edge_max_segment_distance"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")

    @classmethod
    def create(
        cls, model: str, *, provider_configuration: Mapping[str, Any], **kwargs: Any,
    ) -> "SemanticCompilerConfig":
        if not isinstance(provider_configuration, Mapping):
            raise ValueError("provider configuration must be a mapping")
        try:
            encoded = _canonical_json(provider_configuration)
        except (TypeError, ValueError) as exc:
            raise ValueError("provider configuration must be finite JSON data") from exc
        return cls(model, encoded, **kwargs)

    def provider_mapping(self) -> dict[str, Any]:
        # A fresh decode prevents caller or provider mutation from changing the
        # retained compiler configuration.
        return dict(_strict_json_object(
            self.provider_configuration_json, "provider configuration",
        ))


@dataclass(frozen=True)
class NodeQualifierRun:
    node_id: str
    candidates: tuple[QualifierCandidate, ...]
    result: QualifierSelectionResult
    output_node: SemanticNode | None
    application_failure: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("qualifier run node ID is invalid")
        if not isinstance(self.candidates, tuple) or any(
            not isinstance(item, QualifierCandidate) for item in self.candidates
        ):
            raise ValueError("qualifier run candidates must be an immutable typed tuple")
        if not isinstance(self.result, QualifierSelectionResult):
            raise ValueError("qualifier run result is invalid")
        if self.output_node is not None and not isinstance(self.output_node, SemanticNode):
            raise ValueError("qualifier run output is invalid")
        if self.application_failure is not None and (
            not isinstance(self.application_failure, str) or not self.application_failure
        ):
            raise ValueError("qualifier application failure is invalid")
        if (self.output_node is None) == (self.application_failure is None):
            raise ValueError("qualifier run needs exactly one output or application failure")


@dataclass(frozen=True)
class CompilerFailure:
    stage: str
    code: str
    item_id: str | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("compiler failure stage is invalid")
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("compiler failure code is invalid")
        if self.stage not in _FAILURE_CODES_BY_STAGE or self.code not in _FAILURE_CODES_BY_STAGE[self.stage]:
            raise ValueError("compiler failure stage/code taxonomy is invalid")
        if self.item_id is not None and (
            not isinstance(self.item_id, str) or not self.item_id
        ):
            raise ValueError("compiler failure item ID is invalid")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("compiler failure detail is invalid")


@dataclass(frozen=True)
class SemanticCompileRun:
    window: SemanticSourceWindow
    config: SemanticCompilerConfig
    entity_aliases: tuple[str, ...]
    ability_aliases: tuple[str, ...]
    mention_catalog: tuple[MentionCandidate, ...]
    mention_selection: MentionCatalogSelectionResult | None
    mention_nodes: tuple[SemanticNode, ...]
    qualifier_catalog: tuple[QualifierCandidate, ...]
    qualifier_runs: tuple[NodeQualifierRun, ...]
    qualified_nodes: tuple[SemanticNode, ...]
    coreference_candidate_sets: tuple[CoreferenceCandidateSet, ...]
    coreference_decisions: tuple[CoreferenceDecision, ...]
    coreference: CoreferenceCatalogResult | None
    edge_pairs: tuple[CandidateEdgePair, ...]
    edge_results: tuple[EdgeClassificationResult, ...]
    edge_classification: EdgeCatalogClassificationResult | None
    merged_edges: tuple[SemanticEdge, ...]
    graph: SemanticGraph | None
    failures: tuple[CompilerFailure, ...]
    integrity_sha256: str
    version: str = COMPILER_ORCHESTRATION_VERSION

    @property
    def status(self) -> str:
        if self.graph is None:
            return "FAILURE"
        if self.failures:
            return "PARTIAL"
        if not self.graph.nodes:
            return "NONE"
        return "OK"

    def validate(self) -> None:
        """Reconstruct every completed boundary and verify the sealed run."""
        self.window.validate()
        self.config.__post_init__()
        if self.version not in _SUPPORTED_COMPILER_ORCHESTRATION_VERSIONS:
            raise ValueError("semantic compiler run version is unsupported")
        if self.version != self.config.version:
            raise ValueError("semantic compiler run/config versions disagree")
        for aliases, label in (
            (self.entity_aliases, "entity"), (self.ability_aliases, "ability"),
        ):
            if aliases != _normalize_aliases(aliases, label):
                raise ValueError(f"retained {label} aliases are not canonical")
        stage_failure = _terminal_stage_failure(self.failures)
        if stage_failure == "mention_catalog":
            _validate_terminal_suffix(self, stage_failure)
            self._validate_integrity()
            return
        expected_mentions = generate_mention_candidates(
            self.window, entity_aliases=self.entity_aliases,
            ability_aliases=self.ability_aliases,
        )
        if self.mention_catalog != expected_mentions:
            raise ValueError("retained mention catalog is not deterministic")

        if self.mention_selection is None:
            if stage_failure != "mentions":
                raise ValueError("missing mention selection lacks a terminal stage failure")
            _validate_terminal_suffix(self, stage_failure)
            self._validate_integrity()
            return
        if self.version == COMPILER_ORCHESTRATION_VERSION:
            expected_partitions = partition_candidate_catalog(
                self.mention_catalog, max_candidates=self.config.mention_partition_size,
            )
            retained_partitions = tuple(
                tuple(result.candidate_ids)
                for result in self.mention_selection.partition_results
            )
            if retained_partitions != tuple(
                tuple(item.candidate_id for item in partition)
                for partition in expected_partitions
            ):
                raise ValueError("retained focal mention partitions are not deterministic")
        mention_nodes = assemble_semantic_nodes(self.window, self.mention_selection)
        expected_mention_prompt_version = (
            MENTION_SELECTION_PROMPT_VERSION_LEGACY
            if self.version == COMPILER_ORCHESTRATION_VERSION_LEGACY
            else MENTION_SELECTION_PROMPT_VERSION
        )
        for result in self.mention_selection.partition_results:
            request = _strict_json_object(result.request_json, "mention request")
            if request.get("prompt_version") != expected_mention_prompt_version:
                raise ValueError("compiler orchestration and mention prompt versions disagree")
        _validate_stage_requests(self, "mentions")
        if stage_failure == "mention_assembly":
            _validate_terminal_suffix(self, stage_failure)
            self._validate_integrity()
            return
        if mention_nodes != self.mention_nodes:
            raise ValueError("retained mention nodes contradict mention decisions")

        if stage_failure == "qualifier_catalog":
            _validate_terminal_suffix(self, stage_failure)
            self._validate_integrity()
            return
        expected_qualifiers = generate_qualifier_candidates(self.window)
        if self.qualifier_catalog != expected_qualifiers:
            raise ValueError("retained qualifier catalog is not deterministic")
        reconstructed_qualified = []
        for index, run in enumerate(self.qualifier_runs):
            if index >= len(self.mention_nodes):
                raise ValueError("qualifier runs exceed retained mention nodes")
            node = self.mention_nodes[index]
            candidates = qualifier_candidates_for_node(
                self.window, node, self.qualifier_catalog,
            )
            if run.node_id != node.node_id or run.candidates != candidates:
                raise ValueError("qualifier run order/catalog contradicts mention nodes")
            if run.application_failure is not None:
                if stage_failure != "qualifiers" or index != len(self.qualifier_runs) - 1:
                    raise ValueError("qualifier application failure is not the terminal attempt")
                validate_qualifier_selection_result(
                    self.window, node, candidates, run.result,
                )
                continue
            output = apply_node_qualifiers(
                self.window, node, candidates, run.result,
            )
            if output != run.output_node:
                raise ValueError("qualifier output contradicts its retained decision")
            reconstructed_qualified.append(output)
        if tuple(reconstructed_qualified) != self.qualified_nodes:
            raise ValueError("aggregate qualified nodes contradict qualifier runs")
        _validate_stage_requests(self, "qualifiers")
        if len(self.qualifier_runs) != len(self.mention_nodes) or any(
            item.application_failure is not None for item in self.qualifier_runs
        ):
            if stage_failure != "qualifiers":
                raise ValueError("incomplete qualifier catalog lacks a terminal failure")
            _validate_terminal_suffix(self, stage_failure)
            self._validate_integrity()
            return

        if stage_failure == "coreference_catalog":
            _validate_terminal_suffix(self, stage_failure)
            self._validate_integrity()
            return
        expected_sets = generate_coreference_candidate_sets(
            self.window, self.qualified_nodes,
            max_segment_distance=self.config.coreference_max_segment_distance,
        )
        if self.coreference_candidate_sets != expected_sets:
            raise ValueError("retained coreference candidate catalog is not deterministic")
        if self.coreference is None:
            if stage_failure != "coreference":
                raise ValueError("missing coreference run lacks a terminal failure")
            if len(self.coreference_decisions) > len(self.coreference_candidate_sets) or tuple(
                decision.candidate_set_id for decision in self.coreference_decisions
            ) != tuple(
                item.candidate_set_id
                for item in self.coreference_candidate_sets[:len(self.coreference_decisions)]
            ):
                raise ValueError("partial coreference decisions are not an ordered catalog prefix")
            current = self.qualified_nodes
            for candidate_set, decision in zip(
                self.coreference_candidate_sets, self.coreference_decisions,
            ):
                current = apply_coreference_decision(
                    self.window, current, candidate_set, decision,
                ).nodes
            _validate_stage_requests(self, "coreference")
            _validate_terminal_suffix(self, stage_failure)
            self._validate_integrity()
            return
        reconstructed_coref = assemble_coreference_catalog(
            self.window, self.qualified_nodes, self.coreference_candidate_sets,
            self.coreference_decisions,
            max_segment_distance=self.config.coreference_max_segment_distance,
        )
        if reconstructed_coref != self.coreference:
            raise ValueError("retained coreference run is not reconstructible")
        _validate_stage_requests(self, "coreference")

        if stage_failure == "edge_catalog":
            _validate_terminal_suffix(self, stage_failure)
            self._validate_integrity()
            return
        expected_pairs = generate_candidate_edge_pairs(
            self.window, self.coreference.nodes,
            max_character_distance=self.config.edge_max_character_distance,
            max_segment_distance=self.config.edge_max_segment_distance,
        )
        if self.edge_pairs != expected_pairs:
            raise ValueError("retained edge-pair catalog is not deterministic")
        if self.edge_classification is None:
            if stage_failure != "edges":
                raise ValueError("missing edge run lacks a terminal failure")
            node_by_id = {node.node_id: node for node in self.coreference.nodes}
            if len(self.edge_results) > len(self.edge_pairs) or tuple(
                result.pair_id for result in self.edge_results
            ) != tuple(pair.pair_id for pair in self.edge_pairs[:len(self.edge_results)]):
                raise ValueError("partial edge results are not an ordered pair-catalog prefix")
            for pair, result in zip(self.edge_pairs, self.edge_results):
                validate_edge_classification_result(pair, self.window, node_by_id, result)
            _validate_stage_requests(self, "edges")
            _validate_terminal_suffix(self, stage_failure)
            self._validate_integrity()
            return
        # Public assembly reconstructively validates the complete edge-pair
        # catalog and every raw pair decision.
        validate_edge_catalog_classification(
            self.window, self.coreference.nodes, self.edge_classification,
        )
        if (
            self.edge_classification.pairs != self.edge_pairs
            or self.edge_classification.results != self.edge_results
        ):
            raise ValueError("edge aggregate contradicts retained pair attempts")
        _validate_stage_requests(self, "edges")
        reference_failures = _reference_disagreements(
            self.coreference, self.edge_classification,
        )
        expected_edges = _merge_edges(
            self.coreference.edges, self.edge_classification.edges,
        )
        if expected_edges != self.merged_edges:
            raise ValueError("merged edges contradict retained decisions")
        expected_failures = _collect_failures(
            self.mention_selection, self.qualifier_runs, self.coreference,
            self.edge_classification,
        ) + reference_failures
        assembly_failures = tuple(
            item for item in self.failures if item.stage == "assembly"
        )
        if tuple(self.failures) != expected_failures + assembly_failures:
            raise ValueError("compiler failures contradict retained stage evidence")
        if self.graph is not None:
            expected_graph = SemanticGraph.from_source_window(
                self.window, self.coreference.nodes, self.merged_edges,
            )
            if expected_graph != self.graph or assembly_failures:
                raise ValueError("retained graph contradicts its proof-carrying inputs")
        elif not assembly_failures:
            raise ValueError("missing graph lacks an assembly failure")
        self._validate_integrity()

    def _validate_integrity(self) -> None:
        if _SHA256.fullmatch(self.integrity_sha256) is None:
            raise ValueError("compiler run integrity hash is invalid")
        if self.integrity_sha256 != _run_integrity_sha256(self):
            raise ValueError("compiler run content changed after sealing")


def compile_source_semantic_ir(
    window: SemanticSourceWindow,
    chat: Callable[..., str],
    *,
    config: SemanticCompilerConfig,
    entity_aliases: Iterable[str] = (),
    ability_aliases: Iterable[str] = (),
) -> SemanticCompileRun:
    """Compile one exact Pass 0 window without crossing into Pass 2 semantics.

    Malformed caller inputs fail before a model call. Unexpected deterministic
    stage failures return a sealed partial run with every completed catalog.
    """
    if not isinstance(window, SemanticSourceWindow):
        raise ValueError("semantic compiler requires a Pass 0 source window")
    window.validate()
    if not isinstance(config, SemanticCompilerConfig):
        raise ValueError("semantic compiler requires typed configuration")
    config.__post_init__()
    if config.version != COMPILER_ORCHESTRATION_VERSION:
        raise ValueError("legacy semantic compiler configuration is deserialization-only")
    if not callable(chat):
        raise ValueError("semantic compiler chat provider must be callable")
    entities = _normalize_aliases(entity_aliases, "entity")
    abilities = _normalize_aliases(ability_aliases, "ability")
    provider_configuration = config.provider_mapping()

    mention_catalog: tuple[MentionCandidate, ...] = ()
    mention_selection: MentionCatalogSelectionResult | None = None
    mention_nodes: tuple[SemanticNode, ...] = ()
    qualifier_catalog: tuple[QualifierCandidate, ...] = ()
    qualifier_runs: tuple[NodeQualifierRun, ...] = ()
    qualified_nodes: tuple[SemanticNode, ...] = ()
    coreference_sets: tuple[CoreferenceCandidateSet, ...] = ()
    coreference_decisions: tuple[CoreferenceDecision, ...] = ()
    coreference: CoreferenceCatalogResult | None = None
    edge_pairs: tuple[CandidateEdgePair, ...] = ()
    edge_results: tuple[EdgeClassificationResult, ...] = ()
    edge_classification: EdgeCatalogClassificationResult | None = None
    merged_edges: tuple[SemanticEdge, ...] = ()
    failures: tuple[CompilerFailure, ...] = ()

    try:
        mention_catalog = generate_mention_candidates(
            window, entity_aliases=entities, ability_aliases=abilities,
        )
    except Exception as exc:
        return _failed_run(
            window, config, entities, abilities, mention_catalog, mention_selection,
            mention_nodes, qualifier_catalog, qualifier_runs, qualified_nodes,
            coreference_sets, coreference_decisions, coreference,
            edge_pairs, edge_results, edge_classification, merged_edges,
            CompilerFailure("mention_catalog", "ASSEMBLY_FAILURE", None, _detail(exc)),
        )
    try:
        mention_selection = select_mention_catalog(
            window, mention_catalog, chat, model=config.model,
            configuration=provider_configuration,
            max_candidates=config.mention_partition_size,
            max_tokens=config.mention_max_tokens, thinking=config.thinking,
        )
    except Exception as exc:
        return _failed_run(
            window, config, entities, abilities, mention_catalog, mention_selection,
            mention_nodes, qualifier_catalog, qualifier_runs, qualified_nodes,
            coreference_sets, coreference_decisions, coreference,
            edge_pairs, edge_results, edge_classification, merged_edges,
            CompilerFailure("mentions", _exception_failure_code("mentions", exc), None, _detail(exc)),
        )
    try:
        mention_nodes = assemble_semantic_nodes(window, mention_selection)
    except Exception as exc:
        return _failed_run(
            window, config, entities, abilities, mention_catalog, mention_selection,
            mention_nodes, qualifier_catalog, qualifier_runs, qualified_nodes,
            coreference_sets, coreference_decisions, coreference,
            edge_pairs, edge_results, edge_classification, merged_edges,
            CompilerFailure("mention_assembly", _exception_failure_code("mention_assembly", exc), None, _detail(exc)),
        )

    try:
        qualifier_catalog = generate_qualifier_candidates(window)
    except Exception as exc:
        return _failed_run(
            window, config, entities, abilities, mention_catalog, mention_selection,
            mention_nodes, qualifier_catalog, qualifier_runs, qualified_nodes,
            coreference_sets, coreference_decisions, coreference,
            edge_pairs, edge_results, edge_classification, merged_edges,
            CompilerFailure("qualifier_catalog", "ASSEMBLY_FAILURE", None, _detail(exc)),
        )
    completed: list[NodeQualifierRun] = []
    outputs: list[SemanticNode] = []
    try:
        for node in mention_nodes:
            candidates = qualifier_candidates_for_node(window, node, qualifier_catalog)
            result = classify_node_qualifiers(
                window, node, candidates, chat, model=config.model,
                configuration=provider_configuration,
                max_tokens=config.qualifier_max_tokens, thinking=config.thinking,
            )
            try:
                output = apply_node_qualifiers(window, node, candidates, result)
            except Exception as exc:
                completed.append(NodeQualifierRun(
                    node.node_id, candidates, result, None, _detail(exc),
                ))
                raise
            completed.append(NodeQualifierRun(node.node_id, candidates, result, output))
            outputs.append(output)
        qualifier_runs = tuple(completed)
        qualified_nodes = tuple(outputs)
    except Exception as exc:
        qualifier_runs = tuple(completed)
        qualified_nodes = tuple(outputs)
        return _failed_run(
            window, config, entities, abilities, mention_catalog, mention_selection,
            mention_nodes, qualifier_catalog, qualifier_runs, qualified_nodes,
            coreference_sets, coreference_decisions, coreference,
            edge_pairs, edge_results, edge_classification, merged_edges,
            CompilerFailure("qualifiers", _exception_failure_code("qualifiers", exc), None, _detail(exc)),
        )

    try:
        coreference_sets = generate_coreference_candidate_sets(
            window, qualified_nodes,
            max_segment_distance=config.coreference_max_segment_distance,
        )
    except Exception as exc:
        return _failed_run(
            window, config, entities, abilities, mention_catalog, mention_selection,
            mention_nodes, qualifier_catalog, qualifier_runs, qualified_nodes,
            coreference_sets, coreference_decisions, coreference,
            edge_pairs, edge_results, edge_classification, merged_edges,
            CompilerFailure("coreference_catalog", "ASSEMBLY_FAILURE", None, _detail(exc)),
        )
    completed_coreference: list[CoreferenceDecision] = []
    try:
        for candidate_set in coreference_sets:
            completed_coreference.append(classify_coreference(
                window, qualified_nodes, candidate_set, chat, model=config.model,
                configuration=provider_configuration,
                max_tokens=config.coreference_max_tokens, thinking=config.thinking,
            ))
        coreference_decisions = tuple(completed_coreference)
        coreference = assemble_coreference_catalog(
            window, qualified_nodes, coreference_sets, coreference_decisions,
            max_segment_distance=config.coreference_max_segment_distance,
        )
    except Exception as exc:
        coreference_decisions = tuple(completed_coreference)
        return _failed_run(
            window, config, entities, abilities, mention_catalog, mention_selection,
            mention_nodes, qualifier_catalog, qualifier_runs, qualified_nodes,
            coreference_sets, coreference_decisions, coreference,
            edge_pairs, edge_results, edge_classification, merged_edges,
            CompilerFailure("coreference", _exception_failure_code("coreference", exc), None, _detail(exc)),
        )

    try:
        edge_pairs = generate_candidate_edge_pairs(
            window, coreference.nodes,
            max_character_distance=config.edge_max_character_distance,
            max_segment_distance=config.edge_max_segment_distance,
        )
    except Exception as exc:
        return _failed_run(
            window, config, entities, abilities, mention_catalog, mention_selection,
            mention_nodes, qualifier_catalog, qualifier_runs, qualified_nodes,
            coreference_sets, coreference_decisions, coreference,
            edge_pairs, edge_results, edge_classification, merged_edges,
            CompilerFailure("edge_catalog", "ASSEMBLY_FAILURE", None, _detail(exc)),
        )
    completed_edges: list[EdgeClassificationResult] = []
    try:
        node_by_id = {node.node_id: node for node in coreference.nodes}
        for pair in edge_pairs:
            completed_edges.append(classify_edge_pair(
                pair, window, node_by_id, chat, model=config.model,
                configuration=provider_configuration, max_tokens=config.edge_max_tokens,
                thinking=config.thinking,
            ))
        edge_results = tuple(completed_edges)
        edge_classification = assemble_edge_catalog_classification(
            window, coreference.nodes, edge_pairs, edge_results,
            max_character_distance=config.edge_max_character_distance,
            max_segment_distance=config.edge_max_segment_distance,
        )
    except Exception as exc:
        edge_results = tuple(completed_edges)
        return _failed_run(
            window, config, entities, abilities, mention_catalog, mention_selection,
            mention_nodes, qualifier_catalog, qualifier_runs, qualified_nodes,
            coreference_sets, coreference_decisions, coreference,
            edge_pairs, edge_results, edge_classification, merged_edges,
            CompilerFailure("edges", _exception_failure_code("edges", exc), None, _detail(exc)),
        )

    merged_edges = _merge_edges(coreference.edges, edge_classification.edges)
    failures = _collect_failures(
        mention_selection, qualifier_runs, coreference, edge_classification,
    ) + _reference_disagreements(coreference, edge_classification)
    try:
        graph = SemanticGraph.from_source_window(window, coreference.nodes, merged_edges)
    except Exception as exc:
        graph = None
        failures += (CompilerFailure(
            "assembly", _assembly_failure_code(exc), None, _detail(exc),
        ),)
    run = SemanticCompileRun(
        window=window, config=config, entity_aliases=entities, ability_aliases=abilities,
        mention_catalog=mention_catalog, mention_selection=mention_selection,
        mention_nodes=mention_nodes, qualifier_catalog=qualifier_catalog,
        qualifier_runs=qualifier_runs, qualified_nodes=qualified_nodes,
        coreference_candidate_sets=coreference_sets,
        coreference_decisions=coreference_decisions, coreference=coreference,
        edge_pairs=edge_pairs, edge_results=edge_results,
        edge_classification=edge_classification, merged_edges=merged_edges,
        graph=graph, failures=failures, integrity_sha256="",
    )
    return _seal_run(run)


def _failed_run(
    window: SemanticSourceWindow,
    config: SemanticCompilerConfig,
    entities: tuple[str, ...],
    abilities: tuple[str, ...],
    mention_catalog: tuple[MentionCandidate, ...],
    mention_selection: MentionCatalogSelectionResult | None,
    mention_nodes: tuple[SemanticNode, ...],
    qualifier_catalog: tuple[QualifierCandidate, ...],
    qualifier_runs: tuple[NodeQualifierRun, ...],
    qualified_nodes: tuple[SemanticNode, ...],
    coreference_sets: tuple[CoreferenceCandidateSet, ...],
    coreference_decisions: tuple[CoreferenceDecision, ...],
    coreference: CoreferenceCatalogResult | None,
    edge_pairs: tuple[CandidateEdgePair, ...],
    edge_results: tuple[EdgeClassificationResult, ...],
    edge_classification: EdgeCatalogClassificationResult | None,
    merged_edges: tuple[SemanticEdge, ...],
    failure: CompilerFailure,
) -> SemanticCompileRun:
    prior = _collect_available_failures(
        mention_selection, qualifier_runs, coreference_sets,
        coreference_decisions, coreference, edge_pairs, edge_results,
        edge_classification,
    )
    run = SemanticCompileRun(
        window=window, config=config, entity_aliases=entities, ability_aliases=abilities,
        mention_catalog=mention_catalog, mention_selection=mention_selection,
        mention_nodes=mention_nodes, qualifier_catalog=qualifier_catalog,
        qualifier_runs=qualifier_runs, qualified_nodes=qualified_nodes,
        coreference_candidate_sets=coreference_sets,
        coreference_decisions=coreference_decisions, coreference=coreference,
        edge_pairs=edge_pairs, edge_results=edge_results,
        edge_classification=edge_classification, merged_edges=merged_edges,
        graph=None, failures=prior + (failure,), integrity_sha256="",
    )
    return _seal_run(run)


def _seal_run(run: SemanticCompileRun) -> SemanticCompileRun:
    sealed = replace(run, integrity_sha256=_run_integrity_sha256(run))
    sealed.validate()
    return sealed


def _merge_edges(
    coreference_edges: tuple[SemanticEdge, ...],
    classified_edges: tuple[SemanticEdge, ...],
) -> tuple[SemanticEdge, ...]:
    """Dedicated coreference owns REFERS_TO; other pair decisions remain intact."""
    selected: dict[tuple[str, str, str], SemanticEdge] = {}
    for edge in coreference_edges:
        key = (edge.edge_type.value, edge.source_node_id, edge.target_node_id)
        selected[key] = edge
    for edge in classified_edges:
        if edge.edge_type is EdgeType.REFERS_TO:
            continue
        key = (edge.edge_type.value, edge.source_node_id, edge.target_node_id)
        selected.setdefault(key, edge)
    return tuple(sorted(selected.values(), key=lambda item: item.edge_id))


def _reference_disagreements(
    coreference: CoreferenceCatalogResult,
    edges: EdgeCatalogClassificationResult,
) -> tuple[CompilerFailure, ...]:
    results = {item.pair_id: item for item in edges.results}
    failures = []
    for reference_edge in coreference.edges:
        matching_pair = next((
            pair for pair in edges.pairs
            if pair.source_node_id == reference_edge.source_node_id
            and pair.target_node_id == reference_edge.target_node_id
            and EdgeType.REFERS_TO in pair.allowed_edge_types
        ), None)
        if matching_pair is None:
            failures.append(CompilerFailure(
                "coreference", "REFERENCE_RESOLUTION_ERROR", reference_edge.edge_id,
                "resolved reference was absent from the deterministic edge-pair catalog",
            ))
            continue
        result = results[matching_pair.pair_id]
        if result.failure or result.status in {
            "UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE",
        }:
            # Missing corroboration is already represented by the provider,
            # parse, or abstention status. It is not contradictory evidence.
            continue
        agrees = any(
            edge.edge_type is EdgeType.REFERS_TO
            and edge.source_node_id == reference_edge.source_node_id
            and edge.target_node_id == reference_edge.target_node_id
            for edge in result.edges
        )
        if result.status in {"SUPPORTED", "NO_RELATION"} and not agrees:
            failures.append(CompilerFailure(
                "coreference", "REFERENCE_RESOLUTION_ERROR", matching_pair.pair_id,
                "dedicated coreference resolution was not corroborated by pair classification",
            ))
    for edge in edges.edges:
        if edge.edge_type is EdgeType.REFERS_TO and not any(
            item.source_node_id == edge.source_node_id
            and item.target_node_id == edge.target_node_id
            for item in coreference.edges
        ):
            failures.append(CompilerFailure(
                "coreference", "REFERENCE_RESOLUTION_ERROR", edge.edge_id,
                "pair classifier asserted REFERS_TO without dedicated coreference support",
            ))
    return tuple(failures)


def _collect_failures(
    mentions: MentionCatalogSelectionResult,
    qualifiers: tuple[NodeQualifierRun, ...],
    coreference: CoreferenceCatalogResult,
    edges: EdgeCatalogClassificationResult,
) -> tuple[CompilerFailure, ...]:
    failures = []
    for index, result in enumerate(mentions.partition_results, 1):
        if result.failure:
            failures.append(CompilerFailure(
                "mentions", _mention_failure_code(result),
                f"partition:{index}", result.failure,
            ))
        elif result.status in {"UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"}:
            failures.append(CompilerFailure(
                "mentions", result.status, f"partition:{index}", result.status,
            ))
    for run in qualifiers:
        if run.result.failure:
            failures.append(CompilerFailure(
                "qualifiers", _boundary_failure_code("qualifiers", run.result.failure),
                run.node_id, run.result.failure,
            ))
        elif run.result.status in {"UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"}:
            failures.append(CompilerFailure(
                "qualifiers", run.result.status, run.node_id, run.result.status,
            ))
    for candidate_set, decision in zip(coreference.candidate_sets, coreference.decisions):
        if decision.failure:
            failures.append(CompilerFailure(
                "coreference", _coreference_failure_code(decision, candidate_set),
                decision.candidate_set_id, decision.failure,
            ))
        elif decision.status in {"UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"}:
            failures.append(CompilerFailure(
                "coreference", decision.status, decision.candidate_set_id, decision.status,
            ))
    for pair, result in zip(edges.pairs, edges.results):
        if result.failure:
            failures.append(CompilerFailure(
                "edges", _edge_failure_code(result, pair),
                result.pair_id, result.failure,
            ))
        elif result.status in {"UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"}:
            failures.append(CompilerFailure(
                "edges", result.status, result.pair_id, result.status,
            ))
    return tuple(failures)


def _collect_available_failures(
    mentions: MentionCatalogSelectionResult | None,
    qualifiers: tuple[NodeQualifierRun, ...],
    coreference_sets: tuple[CoreferenceCandidateSet, ...],
    coreference_decisions: tuple[CoreferenceDecision, ...],
    coreference: CoreferenceCatalogResult | None,
    edge_pairs: tuple[CandidateEdgePair, ...],
    edge_results: tuple[EdgeClassificationResult, ...],
    edges: EdgeCatalogClassificationResult | None,
) -> tuple[CompilerFailure, ...]:
    """Retain failure evidence from every boundary completed before a stop."""
    failures = []
    if mentions is not None:
        for index, result in enumerate(mentions.partition_results, 1):
            if result.failure:
                failures.append(CompilerFailure(
                    "mentions", _mention_failure_code(result),
                    f"partition:{index}", result.failure,
                ))
            elif result.status in {"UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"}:
                failures.append(CompilerFailure(
                    "mentions", result.status, f"partition:{index}", result.status,
                ))
    for run in qualifiers:
        if run.result.failure:
            failures.append(CompilerFailure(
                "qualifiers", _boundary_failure_code("qualifiers", run.result.failure),
                run.node_id, run.result.failure,
            ))
        elif run.result.status in {"UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"}:
            failures.append(CompilerFailure(
                "qualifiers", run.result.status, run.node_id, run.result.status,
            ))
    retained_sets = coreference.candidate_sets if coreference is not None else coreference_sets
    retained_decisions = coreference.decisions if coreference is not None else coreference_decisions
    for candidate_set, decision in zip(retained_sets, retained_decisions):
            if decision.failure:
                failures.append(CompilerFailure(
                    "coreference", _coreference_failure_code(decision, candidate_set),
                    decision.candidate_set_id, decision.failure,
                ))
            elif decision.status in {"UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"}:
                failures.append(CompilerFailure(
                    "coreference", decision.status, decision.candidate_set_id, decision.status,
                ))
    retained_pairs = edges.pairs if edges is not None else edge_pairs
    retained_results = edges.results if edges is not None else edge_results
    for pair, result in zip(retained_pairs, retained_results):
            if result.failure:
                failures.append(CompilerFailure(
                    "edges", _edge_failure_code(result, pair), result.pair_id, result.failure,
                ))
            elif result.status in {"UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"}:
                failures.append(CompilerFailure(
                    "edges", result.status, result.pair_id, result.status,
                ))
    return tuple(failures)


def _boundary_failure_code(stage: str, failure: str) -> str:
    prefixes = {
        "mentions": "MentionProviderError:",
        "qualifiers": "QualifierProviderError:",
        "coreference": "CoreferenceProviderError:",
        "edges": "EdgeProviderError:",
    }
    if failure.startswith(prefixes[stage]):
        return "PROVIDER_FAILURE"
    return "MODEL_PARSE_FAILURE"


def _mention_failure_code(result: Any) -> str:
    if result.failure and result.failure.startswith("MentionProviderError:"):
        return "PROVIDER_FAILURE"
    try:
        body = json.loads(result.raw_output)
    except Exception:
        return "MODEL_PARSE_FAILURE"
    if isinstance(body, Mapping) and isinstance(body.get("mentions"), list):
        allowed = {
            "ENTITY", "ABILITY_OR_RESOURCE", "EVENT", "ACTION", "STATE", "OUTCOME",
            "QUANTITY", "TIME", "LOCATION_OR_SPACE",
        }
        if any(
            isinstance(item, Mapping) and item.get("node_type") not in allowed
            for item in body["mentions"]
        ):
            return "MENTION_TYPE_ERROR"
    return "MODEL_PARSE_FAILURE"


def _edge_failure_code(result: Any, pair: CandidateEdgePair) -> str:
    if result.failure and result.failure.startswith("EdgeProviderError:"):
        return "PROVIDER_FAILURE"
    try:
        body = json.loads(result.raw_output)
    except Exception:
        return "MODEL_PARSE_FAILURE"
    allowed = {item.value for item in pair.allowed_edge_types}
    if isinstance(body, Mapping) and isinstance(body.get("edge_types"), list) and any(
        not isinstance(item, str) or item not in allowed for item in body["edge_types"]
    ):
        return "UNSUPPORTED_EDGE"
    return "MODEL_PARSE_FAILURE"


def _coreference_failure_code(
    decision: CoreferenceDecision, candidate_set: CoreferenceCandidateSet,
) -> str:
    if decision.failure and decision.failure.startswith("CoreferenceProviderError:"):
        return "PROVIDER_FAILURE"
    try:
        body = json.loads(decision.raw_output)
    except Exception:
        return "MODEL_PARSE_FAILURE"
    allowed = set(candidate_set.target_node_ids)
    if isinstance(body, Mapping):
        target = body.get("target_node_id")
        candidates = body.get("candidate_node_ids")
        if (target is not None and target not in allowed) or (
            isinstance(candidates, list) and any(item not in allowed for item in candidates)
        ):
            return "REFERENCE_RESOLUTION_ERROR"
    return "MODEL_PARSE_FAILURE"


def _assembly_failure_code(exc: Exception) -> str:
    detail = str(exc).casefold()
    if "conditional" in detail and "condition" in detail:
        return "CONDITION_LOSS"
    if "temporal" in detail:
        return "TEMPORAL_LOSS"
    if "refer" in detail or "coreference" in detail:
        return "REFERENCE_RESOLUTION_ERROR"
    return "ASSEMBLY_FAILURE"


def _exception_failure_code(stage: str, exc: Exception) -> str:
    if stage == "mention_assembly" and "node type" in str(exc).casefold():
        return "MENTION_TYPE_ERROR"
    if stage == "edges" and "edge" in str(exc).casefold() and "outside" in str(exc).casefold():
        return "UNSUPPORTED_EDGE"
    return "ASSEMBLY_FAILURE"


def _terminal_stage_failure(failures: tuple[CompilerFailure, ...]) -> str | None:
    if failures and failures[-1].stage in {
        "mention_catalog", "mentions", "mention_assembly", "qualifier_catalog",
        "qualifiers", "coreference_catalog", "coreference", "edge_catalog", "edges",
    }:
        return failures[-1].stage
    return None


def _validate_terminal_suffix(run: SemanticCompileRun, stage: str) -> None:
    """Prove that no unvalidated state appears after a terminal stage failure."""
    if run.graph is not None or run.merged_edges:
        raise ValueError("terminal compiler failure cannot retain an accepted graph/edges")
    downstream: dict[str, tuple[object, ...]] = {
        "mention_catalog": (
            run.mention_catalog, run.mention_selection, run.mention_nodes,
            run.qualifier_catalog, run.qualifier_runs, run.qualified_nodes,
            run.coreference_candidate_sets, run.coreference_decisions, run.coreference,
            run.edge_pairs, run.edge_results, run.edge_classification,
        ),
        "mentions": (
            run.mention_selection, run.mention_nodes, run.qualifier_catalog,
            run.qualifier_runs, run.qualified_nodes, run.coreference_candidate_sets,
            run.coreference_decisions, run.coreference, run.edge_pairs, run.edge_results,
            run.edge_classification,
        ),
        "mention_assembly": (
            run.mention_nodes, run.qualifier_catalog, run.qualifier_runs,
            run.qualified_nodes, run.coreference_candidate_sets,
            run.coreference_decisions, run.coreference, run.edge_pairs,
            run.edge_results, run.edge_classification,
        ),
        "qualifier_catalog": (
            run.qualifier_catalog, run.qualifier_runs, run.qualified_nodes,
            run.coreference_candidate_sets, run.coreference_decisions,
            run.coreference, run.edge_pairs, run.edge_results, run.edge_classification,
        ),
        "qualifiers": (
            run.coreference_candidate_sets, run.coreference_decisions,
            run.coreference, run.edge_pairs, run.edge_results, run.edge_classification,
        ),
        "coreference_catalog": (
            run.coreference_candidate_sets, run.coreference_decisions,
            run.coreference, run.edge_pairs, run.edge_results, run.edge_classification,
        ),
        "coreference": (
            run.coreference, run.edge_pairs, run.edge_results, run.edge_classification,
        ),
        "edge_catalog": (run.edge_pairs, run.edge_results, run.edge_classification),
        "edges": (run.edge_classification,),
    }
    if stage not in downstream:
        raise ValueError("terminal compiler stage is invalid")
    if any(item not in (None, ()) for item in downstream[stage]):
        raise ValueError("terminal compiler failure retains downstream state")
    expected_prior = _collect_available_failures(
        run.mention_selection, run.qualifier_runs,
        run.coreference_candidate_sets, run.coreference_decisions, run.coreference,
        run.edge_pairs, run.edge_results, run.edge_classification,
    )
    if run.failures[:-1] != expected_prior:
        raise ValueError("terminal compiler failure omits or invents prefix failures")
    terminal = run.failures[-1]
    if stage == "qualifiers" and run.qualifier_runs \
            and run.qualifier_runs[-1].application_failure is not None:
        if (
            terminal.code != "ASSEMBLY_FAILURE" or terminal.item_id is not None
            or terminal.detail != run.qualifier_runs[-1].application_failure
        ):
            raise ValueError("qualifier terminal failure contradicts retained application failure")


def _validate_stage_requests(run: SemanticCompileRun, stage: str) -> None:
    expected_configuration = run.config.provider_mapping()
    if stage == "mentions":
        results = run.mention_selection.partition_results  # type: ignore[union-attr]
        expected_tokens = run.config.mention_max_tokens
    elif stage == "qualifiers":
        results = tuple(item.result for item in run.qualifier_runs)
        expected_tokens = run.config.qualifier_max_tokens
    elif stage == "coreference":
        results = run.coreference_decisions
        expected_tokens = run.config.coreference_max_tokens
    elif stage == "edges":
        results = run.edge_results
        expected_tokens = run.config.edge_max_tokens
    else:  # pragma: no cover - internal invariant
        raise AssertionError(stage)
    for result in results:
        request = _strict_json_object(result.request_json, stage + " request")
        if (
            request.get("caller_configuration") != expected_configuration
            or request.get("model") != run.config.model
            or request.get("thinking") != run.config.thinking
            or request.get("max_tokens") != expected_tokens
        ):
            raise ValueError(f"{stage} request contradicts retained compiler configuration")


def _normalize_aliases(values: Iterable[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} aliases must be an iterable of strings")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{label} aliases must be iterable") from exc
    if any(not isinstance(item, str) or not item.strip() or item != item.strip() for item in items):
        raise ValueError(f"{label} aliases must be non-empty trimmed strings")
    return tuple(sorted(set(items), key=lambda item: (item.casefold(), item)))


def _run_integrity_sha256(run: SemanticCompileRun) -> str:
    value = {
        field.name: _canonical_value(getattr(run, field.name))
        for field in fields(run) if field.name != "integrity_sha256"
    }
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _canonical_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"compiler run contains non-JSON value {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _strict_json_object(raw: str, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, str):
        raise ValueError(f"{label} must be a JSON string")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is malformed") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _detail(exc: Exception) -> str:
    return type(exc).__name__ + ":" + str(exc)


__all__ = [
    "COMPILER_ORCHESTRATION_VERSION", "COMPILER_ORCHESTRATION_VERSION_LEGACY",
    "SemanticCompilerConfig",
    "NodeQualifierRun", "CompilerFailure", "SemanticCompileRun",
    "compile_source_semantic_ir",
]
