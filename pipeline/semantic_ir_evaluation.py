"""Reviewed semantic-preservation evaluation for Phase 2F source IR."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from pipeline.semantic_compiler import SemanticCompileRun
from pipeline.semantic_ir import (
    AmbiguityState, ComparativeDegree, Conditionality, EdgeType, Modality,
    NodeType, Polarity, QualifierKind, Restriction, TemporalScope, Uncertainty,
)
from pipeline.semantic_ir_pool import validate_semantic_window_pool


BENCHMARK_SCHEMA_VERSION = "phase2f-semantic-benchmark-v1"
BENCHMARK_SPLITS = frozenset({"DEV", "FROZEN_EVAL", "LEGACY_FAILURE"})
QUESTION_DIMENSIONS = frozenset({
    "entity_recovery", "ability_resource_recovery", "event_recovery",
    "action_recovery", "state_outcome_recovery", "actor_target_roles",
    "condition_recovery", "negation", "modality", "causal_edges",
    "temporal_edges", "coreference", "uncertainty", "quantity",
    "location_or_space", "contrast", "semantic_completeness",
    "comparison",
})
REQUIRED_BENCHMARK_DIMENSIONS = frozenset({
    "entity_recovery", "event_recovery", "action_recovery",
    "condition_recovery", "causal_edges",
})
_QUALIFIER_FIELDS = {
    "polarity": QualifierKind.POLARITY,
    "modality": QualifierKind.MODALITY,
    "temporal_scope": QualifierKind.TEMPORAL_SCOPE,
    "conditionality": QualifierKind.CONDITIONALITY,
    "comparative_degree": QualifierKind.COMPARATIVE_DEGREE,
    "uncertainty": QualifierKind.UNCERTAINTY,
    "restriction": QualifierKind.RESTRICTION,
}
_QUALIFIER_ENUMS = {
    "polarity": Polarity,
    "modality": Modality,
    "temporal_scope": TemporalScope,
    "conditionality": Conditionality,
    "comparative_degree": ComparativeDegree,
    "uncertainty": Uncertainty,
    "restriction": Restriction,
}
_CAUSAL = {
    EdgeType.CAUSES, EdgeType.ENABLES, EdgeType.PREVENTS, EdgeType.REQUIRES,
    EdgeType.PURPOSE, EdgeType.RESULT,
}
_TEMPORAL = {
    EdgeType.TEMPORAL_BEFORE, EdgeType.TEMPORAL_AFTER,
    EdgeType.TEMPORAL_UNTIL, EdgeType.TERMINATES,
}


@dataclass(frozen=True)
class GoldMention:
    mention_id: str
    node_types: tuple[NodeType, ...]
    acceptable_spans: tuple[tuple[int, int], ...]
    critical: bool = False


@dataclass(frozen=True)
class GoldEdge:
    edge_id: str
    source_mention_id: str
    target_mention_id: str
    edge_types: tuple[EdgeType, ...]
    critical: bool = False


@dataclass(frozen=True)
class GoldQualifier:
    qualifier_id: str
    mention_id: str
    field: str
    value: str
    cue_spans: tuple[tuple[int, int], ...]
    critical: bool = False


@dataclass(frozen=True)
class GoldReference:
    reference_id: str
    source_mention_id: str
    status: str
    target_mention_ids: tuple[str, ...]
    critical: bool = False


@dataclass(frozen=True)
class SemanticQuestion:
    question_id: str
    prompt: str
    dimension: str
    requires: tuple[str, ...]
    critical: bool = False


@dataclass(frozen=True)
class SemanticBenchmarkCase:
    case_id: str
    split: str
    source_id: str
    source_kind: str
    source_text: str
    upstream_source_id: str
    upstream_start: int
    upstream_end: int
    phenomena: tuple[str, ...]
    exhaustive: bool
    mentions: tuple[GoldMention, ...]
    edges: tuple[GoldEdge, ...]
    qualifiers: tuple[GoldQualifier, ...]
    references: tuple[GoldReference, ...]
    questions: tuple[SemanticQuestion, ...]

    @property
    def source_fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(
            [self.upstream_source_id, self.upstream_start, self.upstream_end, self.source_text],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SemanticBenchmark:
    split: str
    purpose: str
    pool_manifest_sha256: str
    cases: tuple[SemanticBenchmarkCase, ...]
    content_sha256: str


def load_semantic_benchmark(
    path: Path,
    *,
    expected_split: str,
    expected_content_sha256: str,
    expected_pool_manifest_sha256: str,
    pool_manifest: Mapping[str, Any] | None = None,
    prohibited_upstream_sources: Iterable[str] = (),
) -> SemanticBenchmark:
    if expected_split not in BENCHMARK_SPLITS:
        raise ValueError("semantic benchmark expected split is invalid")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError("semantic benchmark fixture is unavailable") from exc
    body = _strict_json_object(raw)
    expected = {
        "schema_version", "content_sha256", "split", "purpose",
        "pool_manifest_sha256", "cases",
    }
    if set(body) != expected or body.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("semantic benchmark fixture envelope is invalid")
    if body.get("split") != expected_split:
        raise ValueError("semantic benchmark fixture split is not the requested split")
    inner = {key: value for key, value in body.items() if key != "content_sha256"}
    if body.get("content_sha256") != _canonical_sha256(inner):
        raise ValueError("semantic benchmark fixture content hash is invalid")
    if not _is_sha256(expected_content_sha256) or body["content_sha256"] != expected_content_sha256:
        raise ValueError("semantic benchmark fixture does not match its trusted content lock")
    if not isinstance(body.get("purpose"), str) or not body["purpose"].strip():
        raise ValueError("semantic benchmark purpose must be non-empty")
    if not _is_sha256(body.get("pool_manifest_sha256")):
        raise ValueError("semantic benchmark pool-manifest hash is invalid")
    if not _is_sha256(expected_pool_manifest_sha256) or body["pool_manifest_sha256"] != expected_pool_manifest_sha256:
        raise ValueError("semantic benchmark does not match its trusted pool-manifest lock")
    raw_cases = body.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("semantic benchmark fixture must contain reviewed cases")
    cases = tuple(_case_from_dict(item, expected_split) for item in raw_cases)
    if len({item.case_id for item in cases}) != len(cases):
        raise ValueError("semantic benchmark case IDs must be unique")
    if len({item.source_fingerprint for item in cases}) != len(cases):
        raise ValueError("semantic benchmark contains duplicate source windows")
    _reject_internal_overlap(cases)
    if expected_split in {"DEV", "FROZEN_EVAL"}:
        dimensions = {question.dimension for case in cases for question in case.questions}
        missing_dimensions = REQUIRED_BENCHMARK_DIMENSIONS - dimensions
        if missing_dimensions:
            raise ValueError(
                "semantic benchmark omits required dimensions: "
                + ", ".join(sorted(missing_dimensions))
            )
        critical_dimensions = {
            question.dimension for case in cases for question in case.questions
            if question.critical
        }
        missing_critical = REQUIRED_BENCHMARK_DIMENSIONS - critical_dimensions
        if missing_critical:
            raise ValueError(
                "semantic benchmark omits required critical dimensions: "
                + ", ".join(sorted(missing_critical)),
            )
    prohibited = set(prohibited_upstream_sources)
    overlap = sorted({item.upstream_source_id for item in cases} & prohibited)
    if overlap:
        raise ValueError("semantic benchmark overlaps a prohibited held-out source: " + ", ".join(overlap))
    benchmark = SemanticBenchmark(
        expected_split, body["purpose"], body["pool_manifest_sha256"], cases,
        body["content_sha256"],
    )
    _validate_pool_membership(benchmark, pool_manifest)
    return benchmark


def verify_benchmark_isolation(
    left: SemanticBenchmark, right: SemanticBenchmark,
) -> None:
    if left.split == right.split:
        raise ValueError("benchmark isolation requires distinct splits")
    case_overlap = {item.case_id for item in left.cases} & {item.case_id for item in right.cases}
    if case_overlap:
        raise ValueError("semantic benchmark case IDs overlap across splits")
    if (
        {left.split, right.split} == {"DEV", "FROZEN_EVAL"}
        and left.pool_manifest_sha256 != right.pool_manifest_sha256
    ):
        raise ValueError("DEV and FROZEN_EVAL must be selected from the same locked pool")
    for first in left.cases:
        for second in right.cases:
            if first.upstream_source_id != second.upstream_source_id:
                continue
            if first.upstream_start < second.upstream_end and second.upstream_start < first.upstream_end:
                raise ValueError("semantic benchmark source spans overlap across splits")


def _validate_pool_membership(
    benchmark: SemanticBenchmark,
    pool_manifest: Mapping[str, Any] | None,
) -> None:
    if benchmark.split in {"DEV", "FROZEN_EVAL"} and pool_manifest is None:
        raise ValueError("DEV/FROZEN semantic benchmarks require their locked pool manifest")
    if pool_manifest is None:
        return
    validate_semantic_window_pool(pool_manifest)
    if pool_manifest["content_sha256"] != benchmark.pool_manifest_sha256:
        raise ValueError("semantic benchmark pool manifest content does not match its lock")
    pool_windows = {
        (
            item["source_id"], item["source_kind"], item["upstream_source_id"],
            item["upstream_start"], item["upstream_end"], item["source_text"],
        )
        for item in pool_manifest["windows"]
    }
    missing = [
        case.case_id for case in benchmark.cases
        if (
            case.source_id, case.source_kind, case.upstream_source_id,
            case.upstream_start, case.upstream_end, case.source_text,
        ) not in pool_windows
    ]
    if missing:
        raise ValueError(
            "semantic benchmark contains windows outside its locked pool manifest: "
            + ", ".join(missing),
        )


def _validate_case_object(case: SemanticBenchmarkCase) -> None:
    if not isinstance(case, SemanticBenchmarkCase):
        raise ValueError("semantic evaluation requires a typed reviewed case")
    reconstructed = _case_from_dict(_case_to_dict(case), case.split)
    if reconstructed != case:
        raise ValueError("semantic benchmark case is not in canonical reviewed form")


def _validate_benchmark_object(
    benchmark: SemanticBenchmark,
    *,
    expected_content_sha256: str,
    pool_manifest: Mapping[str, Any] | None,
) -> None:
    if not isinstance(benchmark, SemanticBenchmark):
        raise ValueError("semantic evaluation requires a typed benchmark")
    if benchmark.split not in BENCHMARK_SPLITS \
            or not isinstance(benchmark.purpose, str) or not benchmark.purpose.strip() \
            or not _is_sha256(benchmark.pool_manifest_sha256) \
            or not _is_sha256(benchmark.content_sha256):
        raise ValueError("semantic benchmark typed envelope is invalid")
    if not _is_sha256(expected_content_sha256) \
            or benchmark.content_sha256 != expected_content_sha256:
        raise ValueError("semantic benchmark does not match its trusted content lock")
    if not benchmark.cases or any(case.split != benchmark.split for case in benchmark.cases):
        raise ValueError("semantic benchmark typed cases/split are invalid")
    for case in benchmark.cases:
        _validate_case_object(case)
    if len({case.case_id for case in benchmark.cases}) != len(benchmark.cases):
        raise ValueError("semantic benchmark case IDs must be unique")
    if len({case.source_fingerprint for case in benchmark.cases}) != len(benchmark.cases):
        raise ValueError("semantic benchmark contains duplicate source windows")
    _reject_internal_overlap(benchmark.cases)
    if benchmark.split in {"DEV", "FROZEN_EVAL"}:
        dimensions = {
            question.dimension for case in benchmark.cases for question in case.questions
        }
        missing = REQUIRED_BENCHMARK_DIMENSIONS - dimensions
        if missing:
            raise ValueError(
                "semantic benchmark omits required dimensions: "
                + ", ".join(sorted(missing)),
            )
        critical_dimensions = {
            question.dimension for case in benchmark.cases for question in case.questions
            if question.critical
        }
        missing_critical = REQUIRED_BENCHMARK_DIMENSIONS - critical_dimensions
        if missing_critical:
            raise ValueError(
                "semantic benchmark omits required critical dimensions: "
                + ", ".join(sorted(missing_critical)),
            )
    inner = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "split": benchmark.split,
        "purpose": benchmark.purpose,
        "pool_manifest_sha256": benchmark.pool_manifest_sha256,
        "cases": [_case_to_dict(case) for case in benchmark.cases],
    }
    if _canonical_sha256(inner) != benchmark.content_sha256:
        raise ValueError("semantic benchmark typed content hash is invalid")
    _validate_pool_membership(benchmark, pool_manifest)


def _case_to_dict(case: SemanticBenchmarkCase) -> dict[str, Any]:
    return {
        "id": case.case_id, "split": case.split, "source_id": case.source_id,
        "source_kind": case.source_kind, "source_text": case.source_text,
        "upstream_source_id": case.upstream_source_id,
        "upstream_start": case.upstream_start, "upstream_end": case.upstream_end,
        "phenomena": list(case.phenomena), "exhaustive": case.exhaustive,
        "mentions": [{
            "id": item.mention_id,
            "node_types": [node_type.value for node_type in item.node_types],
            "acceptable_spans": [list(span) for span in item.acceptable_spans],
            "critical": item.critical,
        } for item in case.mentions],
        "edges": [{
            "id": item.edge_id, "source": item.source_mention_id,
            "target": item.target_mention_id,
            "edge_types": [edge_type.value for edge_type in item.edge_types],
            "critical": item.critical,
        } for item in case.edges],
        "qualifiers": [{
            "id": item.qualifier_id, "mention": item.mention_id,
            "field": item.field, "value": item.value,
            "cue_spans": [list(span) for span in item.cue_spans],
            "critical": item.critical,
        } for item in case.qualifiers],
        "references": [{
            "id": item.reference_id, "source": item.source_mention_id,
            "status": item.status, "targets": list(item.target_mention_ids),
            "critical": item.critical,
        } for item in case.references],
        "questions": [{
            "id": item.question_id, "prompt": item.prompt,
            "dimension": item.dimension, "requires": list(item.requires),
            "critical": item.critical,
        } for item in case.questions],
    }


def evaluate_semantic_case(
    case: SemanticBenchmarkCase, run: SemanticCompileRun,
) -> dict[str, Any]:
    _validate_case_object(case)
    run.validate()
    if (
        run.window.source_id != case.source_id
        or run.window.text != case.source_text
        or run.window.source_start != case.upstream_start
        or run.window.source_end != case.upstream_end
    ):
        return _source_window_loss(case, run)
    nodes = _latest_nodes(run)
    edges = run.merged_edges
    edge_pairs = run.edge_classification.pairs if run.edge_classification is not None else run.edge_pairs
    coreference_sets = (
        run.coreference.candidate_sets if run.coreference is not None
        else run.coreference_candidate_sets
    )
    candidate_spans = {(item.start, item.end) for item in run.mention_catalog}
    predicted_by_span: dict[tuple[int, int], list[Any]] = {}
    for node in nodes:
        predicted_by_span.setdefault(
            (node.source_span.local_start, node.source_span.local_end), [],
        ).append(node)

    mention_matches: dict[str, Any] = {}
    used_node_ids = set()
    facts = set()
    failures = []
    coverage_hits = 0
    selection_hits = 0
    type_hits = 0
    family_counts: dict[str, list[int]] = {}
    for gold in case.mentions:
        offered = any(span in candidate_spans for span in gold.acceptable_spans)
        coverage_hits += int(offered)
        candidates = [
            node for span in gold.acceptable_spans for node in predicted_by_span.get(span, ())
            if node.node_id not in used_node_ids
        ]
        typed = next((node for node in candidates if node.node_type in gold.node_types), None)
        family_key = "/".join(sorted(item.value for item in gold.node_types))
        family_metric = family_counts.setdefault(family_key, [0, 0])
        family_metric[1] += 1
        selection_hits += int(bool(candidates))
        if typed is not None:
            mention_matches[gold.mention_id] = typed
            used_node_ids.add(typed.node_id)
            facts.add("mention:" + gold.mention_id)
            type_hits += 1
            family_metric[0] += 1
        elif candidates:
            failures.append(_failure(
                "MENTION_TYPE_ERROR", "mention:" + gold.mention_id, gold.critical,
            ))
        elif not offered:
            failures.append(_failure(
                "MENTION_CANDIDATE_MISSING", "mention:" + gold.mention_id, gold.critical,
            ))
        else:
            failures.append(_failure(
                "MENTION_SELECTION_MISS", "mention:" + gold.mention_id, gold.critical,
            ))

    pair_offered = 0
    pair_reached = 0
    edge_hits = 0
    matched_edge_ids = set()
    for gold in case.edges:
        source = mention_matches.get(gold.source_mention_id)
        target = mention_matches.get(gold.target_mention_id)
        if source is None or target is None:
            failures.append(_failure(
                "ASSEMBLY_FAILURE", "edge:" + gold.edge_id, gold.critical,
            ))
            continue
        pair_reached += 1
        offered = any(
            pair.source_node_id == source.node_id and pair.target_node_id == target.node_id
            and any(edge_type in pair.allowed_edge_types for edge_type in gold.edge_types)
            for pair in edge_pairs
        )
        pair_offered += int(offered)
        predicted = next((
            edge for edge in edges
            if edge.source_node_id == source.node_id and edge.target_node_id == target.node_id
            and edge.edge_type in gold.edge_types
            and edge.edge_id not in matched_edge_ids
        ), None)
        if predicted is not None:
            facts.add("edge:" + gold.edge_id)
            matched_edge_ids.add(predicted.edge_id)
            edge_hits += 1
        elif not offered:
            failures.append(_failure(
                "EDGE_PAIR_NOT_ENUMERATED", "edge:" + gold.edge_id, gold.critical,
            ))
        elif any(
            edge.source_node_id == target.node_id and edge.target_node_id == source.node_id
            and edge.edge_type in gold.edge_types for edge in edges
        ):
            failures.append(_failure(
                "EDGE_DIRECTION_ERROR", "edge:" + gold.edge_id, gold.critical,
            ))
        else:
            failures.append(_failure(
                _edge_failure_code(gold), "edge:" + gold.edge_id, gold.critical,
            ))

    qualifier_hits = 0
    qualifier_candidate_hits = 0
    for gold in case.qualifiers:
        kind = _QUALIFIER_FIELDS[gold.field]
        offered = all(any(
            item.kind is kind and (item.start, item.end) == span
            for item in run.qualifier_catalog
        ) for span in gold.cue_spans)
        qualifier_candidate_hits += int(offered)
        node = mention_matches.get(gold.mention_id)
        recovered = False
        if node is not None:
            actual = getattr(node.qualifiers, gold.field)
            actual_value = actual.value if hasattr(actual, "value") else str(actual)
            cue_spans = {
                (cue.span.local_start, cue.span.local_end)
                for cue in node.qualifiers.cues if cue.kind is kind
            }
            recovered = actual_value == gold.value and set(gold.cue_spans) <= cue_spans
        if recovered:
            facts.add("qualifier:" + gold.qualifier_id)
            qualifier_hits += 1
        else:
            failures.append(_failure(
                (
                    _qualifier_failure_code(gold)
                    if offered else "QUALIFIER_CANDIDATE_MISSING"
                ),
                "qualifier:" + gold.qualifier_id, gold.critical,
            ))

    reference_hits = 0
    reference_candidate_hits = 0
    matched_reference_edge_ids = set()
    for gold in case.references:
        source = mention_matches.get(gold.source_mention_id)
        targets = tuple(mention_matches.get(item) for item in gold.target_mention_ids)
        if source is None or any(item is None for item in targets):
            failures.append(_failure(
                "ASSEMBLY_FAILURE", "reference:" + gold.reference_id, gold.critical,
            ))
            continue
        target_ids = tuple(item.node_id for item in targets)
        candidate_set = next((
            item for item in coreference_sets
            if item.source_node_id == source.node_id
        ), None)
        offered = candidate_set is not None and set(target_ids) <= set(candidate_set.target_node_ids)
        if not target_ids and candidate_set is not None:
            offered = True
        reference_candidate_hits += int(offered)
        decision = next((
            item for item in run.coreference_decisions
            if candidate_set is not None and item.candidate_set_id == candidate_set.candidate_set_id
        ), None)
        recovered = _reference_recovered(
            gold.status, source, target_ids, edges, decision,
        )
        if recovered:
            facts.add("reference:" + gold.reference_id)
            reference_hits += 1
            matched_reference_edge_ids.update(
                edge.edge_id for edge in edges
                if edge.edge_type is EdgeType.REFERS_TO and edge.source_node_id == source.node_id
            )
        else:
            failures.append(_failure(
                "REFERENCE_RESOLUTION_ERROR", "reference:" + gold.reference_id, gold.critical,
            ))

    question_hits = sum(all(requirement in facts for requirement in item.requires) for item in case.questions)
    question_results = [{
        "question_id": item.question_id, "prompt": item.prompt, "dimension": item.dimension,
        "critical": item.critical,
        "answerable_from_bronze": True,
        "answerable_from_ir": all(requirement in facts for requirement in item.requires),
        "missing_requirements": [value for value in item.requires if value not in facts],
    } for item in case.questions]

    unscored_nodes = [node.node_id for node in nodes if node.node_id not in used_node_ids]
    matched_all_edges = matched_edge_ids | matched_reference_edge_ids
    unscored_edges = [edge.edge_id for edge in edges if edge.edge_id not in matched_all_edges]
    unsupported_nodes = unscored_nodes if case.exhaustive else []
    unsupported_edges = unscored_edges if case.exhaustive else []
    if case.exhaustive:
        failures.extend(
            _failure("UNSUPPORTED_NODE", "node:" + node_id, False) for node_id in unsupported_nodes
        )
        failures.extend(
            _failure("UNSUPPORTED_EDGE", "edge:" + edge_id, False) for edge_id in unsupported_edges
        )
    for compiler_failure in run.failures:
        failures.append({
            "code": compiler_failure.code, "fact_id": compiler_failure.item_id,
            "critical": False, "stage": compiler_failure.stage,
            "detail": compiler_failure.detail,
        })

    dimension_counts: dict[str, list[int]] = {}
    critical_dimension_counts: dict[str, list[int]] = {}
    for question, result in zip(case.questions, question_results):
        counts = dimension_counts.setdefault(question.dimension, [0, 0])
        counts[1] += 1
        counts[0] += int(result["answerable_from_ir"])
        if question.critical:
            critical_counts = critical_dimension_counts.setdefault(question.dimension, [0, 0])
            critical_counts[1] += 1
            critical_counts[0] += int(result["answerable_from_ir"])
    return {
        "case_id": case.case_id,
        "split": case.split,
        "status": run.status,
        "source_span_validity": _source_spans_valid(run, nodes, edges),
        "edge_provenance_traceability": _edge_provenance_valid(run, edges),
        "mention_candidate_coverage": _metric(coverage_hits, len(case.mentions)),
        "mention_selection_recall": _metric(selection_hits, len(case.mentions)),
        "mention_type_recall": _metric(type_hits, len(case.mentions)),
        "mention_families": {
            key: _metric(hit, denominator)
            for key, (hit, denominator) in sorted(family_counts.items())
        },
        "edge_pair_coverage": _metric(pair_offered, len(case.edges)),
        "edge_pair_coverage_when_endpoints_recovered": _metric(pair_offered, pair_reached),
        "edge_recall": _metric(edge_hits, len(case.edges)),
        "qualifier_candidate_coverage": _metric(qualifier_candidate_hits, len(case.qualifiers)),
        "qualifier_recall": _metric(qualifier_hits, len(case.qualifiers)),
        "reference_candidate_coverage": _metric(reference_candidate_hits, len(case.references)),
        "reference_recall": _metric(reference_hits, len(case.references)),
        "semantic_completeness": _metric(len(facts), _fact_denominator(case)),
        "semantic_checksum": _metric(question_hits, len(case.questions)),
        "dimensions": {
            key: _metric(hit, denominator)
            for key, (hit, denominator) in sorted(dimension_counts.items())
        },
        "critical_dimensions": {
            key: _metric(hit, denominator)
            for key, (hit, denominator) in sorted(critical_dimension_counts.items())
        },
        "unsupported_nodes": unsupported_nodes,
        "unsupported_edges": unsupported_edges,
        "unscored_nodes": unscored_nodes if not case.exhaustive else [],
        "unscored_edges": unscored_edges if not case.exhaustive else [],
        "unsupported_node_rate": _metric(len(unsupported_nodes), len(nodes) if case.exhaustive else 0),
        "unsupported_edge_rate": _metric(len(unsupported_edges), len(edges) if case.exhaustive else 0),
        "questions": question_results,
        "recovered_facts": sorted(facts),
        "failures": failures,
        "first_loss": _first_loss(failures),
    }


def evaluate_semantic_benchmark(
    benchmark: SemanticBenchmark,
    runs: Mapping[str, SemanticCompileRun],
    *,
    expected_content_sha256: str,
    pool_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_benchmark_object(
        benchmark, expected_content_sha256=expected_content_sha256,
        pool_manifest=pool_manifest,
    )
    if set(runs) != {item.case_id for item in benchmark.cases}:
        raise ValueError("semantic benchmark runs must exactly cover every reviewed case")
    cases = [evaluate_semantic_case(item, runs[item.case_id]) for item in benchmark.cases]
    question_hit = sum(item["semantic_checksum"]["hit_count"] for item in cases)
    question_denominator = sum(item["semantic_checksum"]["denominator"] for item in cases)
    unsupported_nodes = sum(len(item["unsupported_nodes"]) for item in cases)
    exhaustive_case_ids = {item.case_id for item in benchmark.cases if item.exhaustive}
    produced_nodes = sum(
        len(_latest_nodes(runs[item.case_id])) for item in benchmark.cases
        if item.case_id in exhaustive_case_ids
    )
    unsupported_edges = sum(len(item["unsupported_edges"]) for item in cases)
    produced_edges = sum(
        len(runs[item.case_id].merged_edges) for item in benchmark.cases
        if item.case_id in exhaustive_case_ids
    )
    dimensions: dict[str, list[int]] = {}
    critical_dimensions: dict[str, list[int]] = {}
    for item in cases:
        for key, metric in item["dimensions"].items():
            counts = dimensions.setdefault(key, [0, 0])
            counts[0] += metric["hit_count"]
            counts[1] += metric["denominator"]
        for key, metric in item["critical_dimensions"].items():
            counts = critical_dimensions.setdefault(key, [0, 0])
            counts[0] += metric["hit_count"]
            counts[1] += metric["denominator"]
    failure_counts: dict[str, int] = {}
    for item in cases:
        for failure in item["failures"]:
            failure_counts[failure["code"]] = failure_counts.get(failure["code"], 0) + 1
    aggregate_metrics = {}
    for name in (
        "mention_candidate_coverage", "mention_selection_recall", "mention_type_recall",
        "edge_pair_coverage", "edge_pair_coverage_when_endpoints_recovered", "edge_recall",
        "qualifier_candidate_coverage",
        "qualifier_recall", "reference_candidate_coverage", "reference_recall",
        "semantic_completeness", "semantic_checksum",
    ):
        hit = sum(item[name]["hit_count"] for item in cases)
        denominator = sum(item[name]["denominator"] for item in cases)
        aggregate_metrics[name] = _metric(hit, denominator)
    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_content_sha256": benchmark.content_sha256,
        "split": benchmark.split,
        "case_count": len(cases),
        **aggregate_metrics,
        "source_span_validity": _metric(
            sum(item["source_span_validity"] for item in cases), len(cases),
        ),
        "edge_provenance_traceability": _metric(
            sum(item["edge_provenance_traceability"] for item in cases), len(cases),
        ),
        "dimensions": {
            key: _metric(hit, denominator)
            for key, (hit, denominator) in sorted(dimensions.items())
        },
        "critical_dimensions": {
            key: _metric(hit, denominator)
            for key, (hit, denominator) in sorted(critical_dimensions.items())
        },
        "mention_families": _aggregate_nested_metrics(cases, "mention_families"),
        "unsupported_node_rate": _metric(unsupported_nodes, produced_nodes),
        "unsupported_edge_rate": _metric(unsupported_edges, produced_edges),
        "exhaustive_case_count": len(exhaustive_case_ids),
        "failure_counts": dict(sorted(failure_counts.items())),
        "cases": cases,
    }


def _case_from_dict(value: object, expected_split: str) -> SemanticBenchmarkCase:
    expected = {
        "id", "split", "source_id", "source_kind", "source_text", "upstream_source_id",
        "upstream_start", "upstream_end", "phenomena", "exhaustive", "mentions", "edges",
        "qualifiers", "references", "questions",
    }
    if not isinstance(value, Mapping) or set(value) != expected or value.get("split") != expected_split:
        raise ValueError("semantic benchmark case shape/split is invalid")
    for key in ("id", "source_id", "source_kind", "source_text", "upstream_source_id"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError("semantic benchmark case text identity is invalid")
    start, end = value["upstream_start"], value["upstream_end"]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (start, end)) \
            or start < 0 or end <= start or end - start != len(value["source_text"]):
        raise ValueError("semantic benchmark upstream offsets are invalid")
    if value["source_id"] != value["source_kind"] + ":" + value["upstream_source_id"]:
        raise ValueError("semantic benchmark source ID is not bound to its upstream source")
    if not isinstance(value["phenomena"], list) or not value["phenomena"] or any(
        not isinstance(item, str) or not item for item in value["phenomena"]
    ) or not isinstance(value["exhaustive"], bool):
        raise ValueError("semantic benchmark phenomena/exhaustive metadata is invalid")
    mentions = tuple(_mention_from_dict(item, value["source_text"]) for item in _list(value["mentions"], "mentions"))
    if not mentions or len({item.mention_id for item in mentions}) != len(mentions):
        raise ValueError("semantic benchmark mentions must be non-empty and unique")
    if len({
        (frozenset(item.node_types), frozenset(item.acceptable_spans)) for item in mentions
    }) != len(mentions):
        raise ValueError("semantic benchmark contains duplicate semantic mentions")
    mention_ids = {item.mention_id for item in mentions}
    edges = tuple(_edge_from_dict(item, mention_ids) for item in _list(value["edges"], "edges"))
    qualifiers = tuple(_qualifier_from_dict(item, mention_ids, value["source_text"]) for item in _list(value["qualifiers"], "qualifiers"))
    references = tuple(_reference_from_dict(item, mention_ids) for item in _list(value["references"], "references"))
    all_fact_ids = (
        {"mention:" + item.mention_id for item in mentions}
        | {"edge:" + item.edge_id for item in edges}
        | {"qualifier:" + item.qualifier_id for item in qualifiers}
        | {"reference:" + item.reference_id for item in references}
    )
    questions = tuple(_question_from_dict(item, all_fact_ids) for item in _list(value["questions"], "questions"))
    ids = [item.edge_id for item in edges] + [item.qualifier_id for item in qualifiers] + [item.reference_id for item in references]
    if len(ids) != len(set(ids)) or not questions or len({item.question_id for item in questions}) != len(questions):
        raise ValueError("semantic benchmark fact/question IDs must be unique and questions non-empty")
    if len({
        (item.source_mention_id, item.target_mention_id, frozenset(item.edge_types))
        for item in edges
    }) != len(edges):
        raise ValueError("semantic benchmark contains duplicate semantic edges")
    for index, first in enumerate(edges):
        for second in edges[index + 1:]:
            if (
                first.source_mention_id == second.source_mention_id
                and first.target_mention_id == second.target_mention_id
                and set(first.edge_types) & set(second.edge_types)
            ):
                raise ValueError("semantic benchmark edge alternatives overlap")
    if len({(item.mention_id, item.field) for item in qualifiers}) != len(qualifiers):
        raise ValueError("semantic benchmark contains duplicate semantic qualifiers")
    if len({item.source_mention_id for item in references}) != len(references):
        raise ValueError("semantic benchmark contains duplicate reference judgments")
    covered_facts = tuple(requirement for item in questions for requirement in item.requires)
    if set(covered_facts) != all_fact_ids or len(covered_facts) != len(all_fact_ids) \
            or any(len(item.requires) != 1 for item in questions):
        raise ValueError(
            "semantic checksum must map each reviewed fact to exactly one question",
        )
    compatible_dimensions = _compatible_fact_dimensions(
        mentions, edges, qualifiers, references,
    )
    for question in questions:
        fact_id = question.requires[0]
        allowed = compatible_dimensions[fact_id] | {"semantic_completeness"}
        if question.dimension not in allowed:
            raise ValueError(
                "semantic checksum question dimension is incompatible with its reviewed fact",
            )
    critical_facts = (
        {"mention:" + item.mention_id for item in mentions if item.critical}
        | {"edge:" + item.edge_id for item in edges if item.critical}
        | {"qualifier:" + item.qualifier_id for item in qualifiers if item.critical}
        | {"reference:" + item.reference_id for item in references if item.critical}
    )
    critical_covered = {
        requirement for item in questions if item.critical for requirement in item.requires
    }
    if not critical_facts <= critical_covered:
        raise ValueError("critical reviewed facts require critical checksum questions")
    return SemanticBenchmarkCase(
        value["id"], expected_split, value["source_id"], value["source_kind"],
        value["source_text"], value["upstream_source_id"], start, end,
        tuple(value["phenomena"]), value["exhaustive"], mentions, edges,
        qualifiers, references, questions,
    )


def _compatible_fact_dimensions(
    mentions: tuple[GoldMention, ...],
    edges: tuple[GoldEdge, ...],
    qualifiers: tuple[GoldQualifier, ...],
    references: tuple[GoldReference, ...],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    mention_dimensions = {
        NodeType.ENTITY: {"entity_recovery"},
        NodeType.ABILITY_OR_RESOURCE: {"ability_resource_recovery"},
        NodeType.EVENT: {"event_recovery"},
        NodeType.ACTION: {"action_recovery"},
        NodeType.STATE: {"state_outcome_recovery"},
        NodeType.OUTCOME: {"state_outcome_recovery"},
        NodeType.QUANTITY: {"quantity"},
        NodeType.TIME: {"temporal_edges", "condition_recovery"},
        NodeType.LOCATION_OR_SPACE: {"location_or_space"},
    }
    for mention in mentions:
        result["mention:" + mention.mention_id] = set().union(
            *(mention_dimensions[item] for item in mention.node_types),
        )
    for edge in edges:
        dimensions = set()
        types = set(edge.edge_types)
        if types & {EdgeType.ACTOR, EdgeType.TARGET, EdgeType.OBJECT, EdgeType.EXPERIENCER}:
            dimensions.add("actor_target_roles")
        if EdgeType.CONDITION in types:
            dimensions.add("condition_recovery")
        if types & _CAUSAL:
            dimensions.add("causal_edges")
        if types & _TEMPORAL:
            dimensions.add("temporal_edges")
        if EdgeType.REFERS_TO in types:
            dimensions.add("coreference")
        if EdgeType.NEGATES in types:
            dimensions.add("negation")
        if EdgeType.CONTRASTS_WITH in types:
            dimensions.add("contrast")
        if EdgeType.MODIFIES in types:
            dimensions.add("semantic_completeness")
        result["edge:" + edge.edge_id] = dimensions
    qualifier_dimensions = {
        "polarity": {"negation"}, "modality": {"modality"},
        "temporal_scope": {"temporal_edges"},
        "conditionality": {"condition_recovery"},
        "comparative_degree": {"comparison"}, "uncertainty": {"uncertainty"},
        "restriction": {"semantic_completeness"},
    }
    for qualifier in qualifiers:
        result["qualifier:" + qualifier.qualifier_id] = set(
            qualifier_dimensions[qualifier.field],
        )
    for reference in references:
        result["reference:" + reference.reference_id] = {"coreference"}
    return result


def _mention_from_dict(value: object, text: str) -> GoldMention:
    if not isinstance(value, Mapping) or set(value) != {"id", "node_types", "acceptable_spans", "critical"}:
        raise ValueError("semantic benchmark mention is invalid")
    try:
        node_types = tuple(NodeType(item) for item in _list(value["node_types"], "node_types"))
    except ValueError as exc:
        raise ValueError("semantic benchmark mention type is invalid") from exc
    spans = tuple(_span(item, text) for item in _list(value["acceptable_spans"], "acceptable_spans"))
    if not isinstance(value["id"], str) or not value["id"] or not node_types or not spans \
            or not isinstance(value["critical"], bool):
        raise ValueError("semantic benchmark mention fields are invalid")
    if len(set(node_types)) != len(node_types) or len(set(spans)) != len(spans):
        raise ValueError("semantic benchmark mention alternatives must be unique")
    if node_types != tuple(sorted(node_types, key=lambda item: item.value)) \
            or spans != tuple(sorted(spans)):
        raise ValueError("semantic benchmark mention alternatives must be canonical")
    return GoldMention(value["id"], node_types, spans, value["critical"])


def _edge_from_dict(value: object, mention_ids: set[str]) -> GoldEdge:
    if not isinstance(value, Mapping) or set(value) != {"id", "source", "target", "edge_types", "critical"}:
        raise ValueError("semantic benchmark edge is invalid")
    try:
        types = tuple(EdgeType(item) for item in _list(value["edge_types"], "edge_types"))
    except ValueError as exc:
        raise ValueError("semantic benchmark edge type is invalid") from exc
    if value["source"] not in mention_ids or value["target"] not in mention_ids or not types \
            or value["source"] == value["target"] or not isinstance(value["critical"], bool):
        raise ValueError("semantic benchmark edge endpoints are invalid")
    if len(set(types)) != len(types):
        raise ValueError("semantic benchmark edge alternatives must be unique")
    if types != tuple(sorted(types, key=lambda item: item.value)):
        raise ValueError("semantic benchmark edge alternatives must be canonical")
    return GoldEdge(value["id"], value["source"], value["target"], types, value["critical"])


def _qualifier_from_dict(value: object, mention_ids: set[str], text: str) -> GoldQualifier:
    if not isinstance(value, Mapping) or set(value) != {"id", "mention", "field", "value", "cue_spans", "critical"}:
        raise ValueError("semantic benchmark qualifier is invalid")
    if value["mention"] not in mention_ids or value["field"] not in _QUALIFIER_FIELDS \
            or not isinstance(value["value"], str) or not value["value"] \
            or not isinstance(value["critical"], bool):
        raise ValueError("semantic benchmark qualifier fields are invalid")
    try:
        parsed_value = _QUALIFIER_ENUMS[value["field"]](value["value"])
    except ValueError as exc:
        raise ValueError("semantic benchmark qualifier value is invalid") from exc
    if parsed_value.value == "UNKNOWN":
        raise ValueError("reviewed qualifier cannot use UNKNOWN as an asserted value")
    spans = tuple(_span(item, text) for item in _list(value["cue_spans"], "cue_spans"))
    if not spans:
        raise ValueError("semantic benchmark qualifier requires exact cue spans")
    if len(set(spans)) != len(spans) or spans != tuple(sorted(spans)):
        raise ValueError("semantic benchmark qualifier cue spans must be canonical")
    return GoldQualifier(value["id"], value["mention"], value["field"], value["value"], spans, value["critical"])


def _reference_from_dict(value: object, mention_ids: set[str]) -> GoldReference:
    if not isinstance(value, Mapping) or set(value) != {"id", "source", "status", "targets", "critical"}:
        raise ValueError("semantic benchmark reference is invalid")
    targets = tuple(_list(value["targets"], "reference targets"))
    if value["source"] not in mention_ids or any(item not in mention_ids for item in targets) \
            or value["status"] not in {"RESOLVED", "AMBIGUOUS", "UNKNOWN", "INSUFFICIENT_EVIDENCE", "NONE"} \
            or not isinstance(value["critical"], bool):
        raise ValueError("semantic benchmark reference fields are invalid")
    if len(set(targets)) != len(targets):
        raise ValueError("semantic benchmark reference targets must be unique")
    if targets != tuple(sorted(targets)):
        raise ValueError("semantic benchmark reference targets must be canonical")
    if value["status"] == "RESOLVED" and len(targets) != 1:
        raise ValueError("resolved reviewed reference requires exactly one target")
    if value["status"] == "AMBIGUOUS" and len(targets) < 2:
        raise ValueError("ambiguous reviewed reference requires multiple targets")
    if value["status"] not in {"RESOLVED", "AMBIGUOUS"} and targets:
        raise ValueError("unresolved reviewed reference cannot retain targets")
    return GoldReference(value["id"], value["source"], value["status"], targets, value["critical"])


def _question_from_dict(value: object, fact_ids: set[str]) -> SemanticQuestion:
    if not isinstance(value, Mapping) or set(value) != {"id", "prompt", "dimension", "requires", "critical"}:
        raise ValueError("semantic benchmark question is invalid")
    requires = tuple(_list(value["requires"], "question requirements"))
    if any(item not in fact_ids for item in requires) or not requires:
        raise ValueError("semantic benchmark question references an unknown/empty fact set")
    if len(set(requires)) != len(requires):
        raise ValueError("semantic benchmark question requirements must be unique")
    if any(not isinstance(value[key], str) or not value[key] for key in ("id", "prompt", "dimension")) \
            or not isinstance(value["critical"], bool):
        raise ValueError("semantic benchmark question fields are invalid")
    if value["dimension"] not in QUESTION_DIMENSIONS:
        raise ValueError("semantic benchmark question dimension is invalid")
    return SemanticQuestion(value["id"], value["prompt"], value["dimension"], requires, value["critical"])


def _reference_recovered(
    status: str, source: Any, target_ids: tuple[str, ...], edges: tuple[Any, ...],
    decision: Any,
) -> bool:
    if decision is None or decision.failure:
        return False
    if status == "RESOLVED":
        return (
            source.ambiguity is AmbiguityState.NONE
            and source.referent_candidate_node_ids == target_ids
            and sum(
                edge.edge_type is EdgeType.REFERS_TO
                and edge.source_node_id == source.node_id
                and edge.target_node_id == target_ids[0]
                for edge in edges
            ) == 1
        )
    if status == "AMBIGUOUS":
        return source.ambiguity is AmbiguityState.MULTIPLE_CANDIDATES \
            and set(source.referent_candidate_node_ids) == set(target_ids) \
            and not any(edge.edge_type is EdgeType.REFERS_TO and edge.source_node_id == source.node_id for edge in edges)
    expected = {
        "NONE": AmbiguityState.NONE,
        "UNKNOWN": AmbiguityState.UNKNOWN,
        "INSUFFICIENT_EVIDENCE": AmbiguityState.INSUFFICIENT_EVIDENCE,
    }[status]
    return source.ambiguity is expected and not source.referent_candidates


def _source_window_loss(case: SemanticBenchmarkCase, run: SemanticCompileRun) -> dict[str, Any]:
    denominator = len(case.questions)
    failures = [_failure("SOURCE_WINDOW_LOSS", "source:" + case.case_id, True)]
    dimensions: dict[str, list[int]] = {}
    critical_dimensions: dict[str, list[int]] = {}
    for question in case.questions:
        dimensions.setdefault(question.dimension, [0, 0])[1] += 1
        if question.critical:
            critical_dimensions.setdefault(question.dimension, [0, 0])[1] += 1
    mention_families: dict[str, list[int]] = {}
    for mention in case.mentions:
        key = "/".join(sorted(item.value for item in mention.node_types))
        mention_families.setdefault(key, [0, 0])[1] += 1
    return {
        "case_id": case.case_id, "split": case.split, "status": "FAILURE",
        "source_span_validity": False, "edge_provenance_traceability": False,
        "mention_candidate_coverage": _metric(0, len(case.mentions)),
        "mention_selection_recall": _metric(0, len(case.mentions)),
        "mention_type_recall": _metric(0, len(case.mentions)),
        "mention_families": {
            key: _metric(0, counts[1]) for key, counts in sorted(mention_families.items())
        },
        "edge_pair_coverage": _metric(0, len(case.edges)),
        "edge_pair_coverage_when_endpoints_recovered": _metric(0, 0),
        "edge_recall": _metric(0, len(case.edges)),
        "qualifier_candidate_coverage": _metric(0, len(case.qualifiers)),
        "qualifier_recall": _metric(0, len(case.qualifiers)),
        "reference_candidate_coverage": _metric(0, len(case.references)),
        "reference_recall": _metric(0, len(case.references)),
        "semantic_completeness": _metric(0, _fact_denominator(case)),
        "semantic_checksum": _metric(0, denominator),
        "dimensions": {
            key: _metric(0, counts[1]) for key, counts in sorted(dimensions.items())
        },
        "critical_dimensions": {
            key: _metric(0, counts[1])
            for key, counts in sorted(critical_dimensions.items())
        },
        "unsupported_nodes": [], "unsupported_edges": [],
        "unscored_nodes": [], "unscored_edges": [],
        "unsupported_node_rate": _metric(0, 0),
        "unsupported_edge_rate": _metric(0, 0),
        "questions": [], "recovered_facts": [], "failures": failures,
        "first_loss": "SOURCE_WINDOW_LOSS",
    }


def _qualifier_failure_code(gold: GoldQualifier) -> str:
    return {
        "polarity": "NEGATION_LOSS" if gold.value == "NEGATIVE" else "QUALIFIER_LOSS",
        "conditionality": "CONDITION_LOSS",
        "temporal_scope": "TEMPORAL_LOSS",
        "modality": "MODALITY_LOSS",
    }.get(gold.field, "QUALIFIER_LOSS")


def _edge_failure_code(gold: GoldEdge) -> str:
    types = set(gold.edge_types)
    if EdgeType.CONDITION in types:
        return "CONDITION_LOSS"
    if types & _TEMPORAL:
        return "TEMPORAL_LOSS"
    if types & _CAUSAL:
        return "CAUSAL_EDGE_LOSS"
    return "EDGE_CLASSIFICATION_MISS"


def _source_spans_valid(
    run: SemanticCompileRun, nodes: tuple[Any, ...], edges: tuple[Any, ...],
) -> bool:
    try:
        for node in nodes:
            node.source_span.validate_against(
                run.window.source_id, run.window.window_id, run.window.text,
                window_source_start=run.window.source_start, speaker=run.window.speaker,
                start_timestamp=run.window.start_ms, end_timestamp=run.window.end_ms,
            )
        for edge in edges:
            for span in edge.evidence:
                span.validate_against(
                    run.window.source_id, run.window.window_id, run.window.text,
                    window_source_start=run.window.source_start, speaker=run.window.speaker,
                    start_timestamp=run.window.start_ms, end_timestamp=run.window.end_ms,
                )
    except ValueError:
        return False
    return True


def _edge_provenance_valid(run: SemanticCompileRun, edges: tuple[Any, ...]) -> bool:
    node_ids = {
        node.node_id for node in (
            _latest_nodes(run)
        )
    }
    return all(
        edge.source_node_id in node_ids and edge.target_node_id in node_ids
        and edge.evidence and edge.provenance is not None
        for edge in edges
    )


def _latest_nodes(run: SemanticCompileRun) -> tuple[Any, ...]:
    """Overlay each completed pass without discarding an unfinished suffix."""
    by_id = {node.node_id: node for node in run.mention_nodes}
    for node in run.qualified_nodes:
        by_id[node.node_id] = node
    if run.coreference is not None:
        for node in run.coreference.nodes:
            by_id[node.node_id] = node
    return tuple(sorted(by_id.values(), key=lambda item: item.node_id))


def _first_loss(failures: list[dict[str, Any]]) -> str | None:
    if not failures:
        return None
    return min(
        enumerate(failures), key=lambda item: (_failure_chronology(item[1]), item[0]),
    )[1]["code"]


def _failure_chronology(failure: Mapping[str, Any]) -> tuple[int, int]:
    code = failure.get("code")
    stage = failure.get("stage")
    fact_id = failure.get("fact_id")
    if code == "SOURCE_WINDOW_LOSS":
        return 0, 0
    stage_order = {
        "mention_catalog": 10, "mentions": 11, "mention_assembly": 12,
        "qualifier_catalog": 20, "qualifiers": 21,
        "coreference_catalog": 30, "coreference": 31,
        "edge_catalog": 40, "edges": 41, "assembly": 50,
    }
    if stage in stage_order:
        within = {
            "ASSEMBLY_FAILURE": 0, "PROVIDER_FAILURE": 1,
            "MODEL_PARSE_FAILURE": 2, "UNKNOWN": 3, "AMBIGUOUS": 3,
            "INSUFFICIENT_EVIDENCE": 3,
        }.get(code, 4)
        return stage_order[stage], within
    if isinstance(fact_id, str):
        if fact_id.startswith("mention:") or fact_id.startswith("node:"):
            return (10 if code == "MENTION_CANDIDATE_MISSING" else 11), 4
        if fact_id.startswith("qualifier:"):
            return (20, 0) if code == "QUALIFIER_CANDIDATE_MISSING" else (21, 4)
        if fact_id.startswith("reference:"):
            return 31, 4
        if fact_id.startswith("edge:"):
            return (40 if code == "EDGE_PAIR_NOT_ENUMERATED" else 41), 4
    semantic_order = {
        "MENTION_CANDIDATE_MISSING": (10, 0),
        "MENTION_SELECTION_MISS": (11, 4),
        "MENTION_TYPE_ERROR": (11, 5),
        "UNSUPPORTED_NODE": (12, 5),
        "QUALIFIER_CANDIDATE_MISSING": (20, 0),
        "QUALIFIER_LOSS": (21, 4), "NEGATION_LOSS": (21, 4),
        "MODALITY_LOSS": (21, 4),
        "REFERENCE_RESOLUTION_ERROR": (31, 4),
        "EDGE_PAIR_NOT_ENUMERATED": (40, 0),
        "EDGE_CLASSIFICATION_MISS": (41, 4), "CAUSAL_EDGE_LOSS": (41, 4),
        "CONDITION_LOSS": (41, 4), "TEMPORAL_LOSS": (41, 4),
        "EDGE_DIRECTION_ERROR": (41, 5), "UNSUPPORTED_EDGE": (41, 6),
        "ASSEMBLY_FAILURE": (50, 0),
        "PROVIDER_FAILURE": (60, 1), "MODEL_PARSE_FAILURE": (60, 2),
    }
    return semantic_order.get(code, (55, 0))


def _reject_internal_overlap(cases: tuple[SemanticBenchmarkCase, ...]) -> None:
    for index, first in enumerate(cases):
        for second in cases[index + 1:]:
            if first.upstream_source_id != second.upstream_source_id:
                continue
            if first.upstream_start < second.upstream_end and second.upstream_start < first.upstream_end:
                raise ValueError("semantic benchmark source spans overlap within its split")


def _fact_denominator(case: SemanticBenchmarkCase) -> int:
    return len(case.mentions) + len(case.edges) + len(case.qualifiers) + len(case.references)


def _failure(code: str, fact_id: str, critical: bool) -> dict[str, Any]:
    return {"code": code, "fact_id": fact_id, "critical": critical, "stage": None, "detail": code}


def _metric(hit: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "hit_count": hit, "denominator": denominator,
        "rate": hit / denominator if denominator else None,
    }


def _aggregate_nested_metrics(
    cases: list[dict[str, Any]], field: str,
) -> dict[str, dict[str, int | float | None]]:
    totals: dict[str, list[int]] = {}
    for case in cases:
        for key, metric in case[field].items():
            counts = totals.setdefault(key, [0, 0])
            counts[0] += metric["hit_count"]
            counts[1] += metric["denominator"]
    return {
        key: _metric(hit, denominator)
        for key, (hit, denominator) in sorted(totals.items())
    }


def _span(value: object, text: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("semantic benchmark span must be a [start,end] pair")
    start, end = value
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (start, end)) \
            or not 0 <= start < end <= len(text):
        raise ValueError("semantic benchmark span offsets are invalid")
    return start, end


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"semantic benchmark {label} must be a list")
    return value


def _strict_json_object(raw: str) -> Mapping[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("semantic benchmark JSON contains duplicate keys")
            result[key] = value
        return result
    try:
        body = json.loads(raw, object_pairs_hook=unique)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("semantic benchmark JSON is malformed") from exc
    if not isinstance(body, Mapping):
        raise ValueError("semantic benchmark must be a JSON object")
    return body


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(item in "0123456789abcdef" for item in value)


__all__ = [
    "BENCHMARK_SCHEMA_VERSION", "BENCHMARK_SPLITS", "QUESTION_DIMENSIONS",
    "REQUIRED_BENCHMARK_DIMENSIONS", "GoldMention", "GoldEdge",
    "GoldQualifier", "GoldReference", "SemanticQuestion", "SemanticBenchmarkCase",
    "SemanticBenchmark", "load_semantic_benchmark", "verify_benchmark_isolation",
    "evaluate_semantic_case", "evaluate_semantic_benchmark",
]
