"""Deterministic Phase 2D metrics that keep pipeline failures separate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pipeline.candidate_generation import CandidateSet
from pipeline.constrained_mapper import MappingSelection


@dataclass(frozen=True)
class CanonicalReference:
    subject_id: str
    relation_id: str
    object_id: str
    condition_required: bool = False
    condition_source: str | None = None


def candidate_coverage(candidates: CandidateSet, reference: CanonicalReference) -> dict[str, bool]:
    """Report slot coverage independently from model selection."""
    slots = {
        "subject": reference.subject_id in {item.id for item in candidates.subject},
        "predicate": reference.relation_id in {item.id for item in candidates.relation},
        "object": reference.object_id in {item.id for item in candidates.object},
        "condition": _has_reference_condition(candidates, reference),
    }
    slots["full_triple"] = all(slots.values())
    return slots


def mapper_result(
    selection: MappingSelection, reference: CanonicalReference, candidates: CandidateSet | None = None,
) -> dict[str, bool]:
    """Score an ID-only selection; call only after recording coverage."""
    mapped = selection.status == "mapped"
    condition = _selected_condition_matches(selection, reference, candidates)
    return {
        "mapped": mapped,
        "subject": mapped and selection.subject_id == reference.subject_id,
        "predicate": mapped and selection.relation_id == reference.relation_id,
        "object": mapped and selection.object_id == reference.object_id,
        "condition": mapped and condition,
        "full_triple": mapped and selection.subject_id == reference.subject_id and selection.relation_id == reference.relation_id and selection.object_id == reference.object_id and condition,
    }


def summarize_cases(cases: Iterable[tuple[dict[str, bool], dict[str, bool] | None]]) -> dict[str, float | int]:
    """Summarize candidate coverage and mapper accuracy given availability."""
    values = list(cases)
    total = len(values)
    coverage = [item[0] for item in values]
    mapped = [(cover, result) for cover, result in values]
    covered = [(cover, result) for cover, result in mapped if cover["full_triple"]]
    return {
        "case_count": total,
        "subject_candidate_recall": _rate(coverage, "subject"),
        "predicate_candidate_recall": _rate(coverage, "predicate"),
        "object_candidate_recall": _rate(coverage, "object"),
        "condition_candidate_recall": _rate(coverage, "condition"),
        "full_triple_candidate_coverage": _rate(coverage, "full_triple"),
        "end_to_end_mapping_success_rate": _rate([result or {} for _, result in mapped], "full_triple"),
        "mapper_accuracy_given_candidate_coverage": _rate([result or {} for _, result in covered], "full_triple"),
    }


def primary_failure(
    coverage: dict[str, bool], selection: MappingSelection | None, reference: CanonicalReference | None = None,
    mapper_failure: str | None = None, candidates: CandidateSet | None = None,
) -> str | None:
    """Assign the Phase 2D primary miss category in causal pipeline order."""
    for slot, label in (("subject", "subject_candidate_miss"), ("predicate", "predicate_candidate_miss"), ("object", "object_candidate_miss"), ("condition", "condition_candidate_miss")):
        if not coverage[slot]:
            return label
    if mapper_failure not in {None, "structured_output_failure", "parsing_failure", "provider_failure", "timeout", "invalid_selection"}:
        raise ValueError("unknown mapper failure category")
    if mapper_failure:
        return f"other:{mapper_failure}"
    if selection is None or selection.status != "mapped":
        return "mapper_misselection"
    if reference is not None and not mapper_result(selection, reference, candidates)["full_triple"]:
        return "mapper_misselection"
    return None


def _rate(items: Iterable[dict[str, bool]], key: str) -> float:
    values = list(items)
    return sum(bool(item.get(key)) for item in values) / len(values) if values else 0.0


def _has_reference_condition(candidates: CandidateSet, reference: CanonicalReference) -> bool:
    if not reference.condition_required:
        return True
    if reference.condition_source is None:
        return bool(candidates.condition)
    target = _condition_key(reference.condition_source)
    return any(_condition_key(item.source_text) == target for item in candidates.condition)


def _selected_condition_matches(
    selection: MappingSelection, reference: CanonicalReference, candidates: CandidateSet | None,
) -> bool:
    if not reference.condition_required:
        return True
    if (
        selection.condition_index is None
        or candidates is None
        or selection.condition_index < 0
        or selection.condition_index >= len(candidates.condition)
    ):
        return False
    if reference.condition_source is None:
        return True
    return _condition_key(candidates.condition[selection.condition_index].source_text) == _condition_key(reference.condition_source)


def _condition_key(value: str) -> str:
    return " ".join(value.lower().split())
