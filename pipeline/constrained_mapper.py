"""Phase 2D ID-only ontology selection over system-generated candidates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Literal, Mapping

from pipeline.candidate_generation import CandidateSet, CanonicalCandidate
from pipeline.relation_extract import GroundedProposition


MappingStatus = Literal["mapped", "unmapped", "no_relation"]
_STATUSES = frozenset(("mapped", "unmapped", "no_relation"))
_RESPONSE_KEYS = frozenset(("mapping_status", "subject_id", "relation_id", "object_id", "condition_index", "confidence"))

MAPPER_SYSTEM = """Return JSON only. Select IDs solely from the supplied
candidate lists. Do not invent canonical strings, entities, concepts, relation
types, or conditions. Return unmapped if the proposition is causal but no
candidate combination is justified. Return no_relation if it contains no
causal relation."""


@dataclass(frozen=True)
class MappingSelection:
    status: MappingStatus
    subject_id: str | None = None
    relation_id: str | None = None
    object_id: str | None = None
    condition_index: int | None = None
    confidence: float | None = None


def mapper_prompt(proposition: GroundedProposition, candidates: CandidateSet) -> str:
    """Render candidate IDs, never a free-form canonical output contract."""
    return (
        "GROUNDED PROPOSITION:\n" + json.dumps({
            "subject_source": proposition.subject_source,
            "predicate_source": proposition.predicate_source,
            "effect_source": proposition.effect_source,
            "condition_source": proposition.condition_source,
        })
        + "\nSUBJECT CANDIDATES: " + _candidate_json(candidates.subject)
        + "\nRELATION CANDIDATES: " + _candidate_json(candidates.relation)
        + "\nOBJECT CANDIDATES: " + _candidate_json(candidates.object)
        + "\nCONDITION CANDIDATES: " + json.dumps([
            {"index": index, "source_text": item.source_text, "event": item.event, "derived_state": item.derived_state}
            for index, item in enumerate(candidates.condition)
        ])
        + "\nReturn exactly {\"mapping_status\":\"mapped|unmapped|no_relation\",\"subject_id\":null,\"relation_id\":null,\"object_id\":null,\"condition_index\":null,\"confidence\":null}. For mapped, all three IDs must be supplied from their matching lists and confidence may be 0.0 through 1.0. For unmapped or no_relation, all ID fields, condition_index, and confidence must be null."
    )


def select_mapping(
    proposition: GroundedProposition, candidates: CandidateSet, chat: Callable[..., str], *, model: str | None = None,
    max_tokens: int = 256,
) -> MappingSelection:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    raw = chat(system=MAPPER_SYSTEM, user=mapper_prompt(proposition, candidates), temperature=0.0, max_tokens=max_tokens, model=model)
    return parse_mapping_selection(raw, candidates)


def parse_mapping_selection(raw: str, candidates: CandidateSet) -> MappingSelection:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("mapper returned malformed JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("mapper response must be an object")
    if set(value) != _RESPONSE_KEYS:
        raise ValueError("mapper response has unknown or missing fields")
    status = value.get("mapping_status")
    if status not in _STATUSES:
        raise ValueError("mapper has invalid mapping_status")
    subject, relation, obj, condition = value.get("subject_id"), value.get("relation_id"), value.get("object_id"), value.get("condition_index")
    if status != "mapped":
        if any(item is not None for item in (subject, relation, obj, condition, value.get("confidence"))):
            raise ValueError("unmapped and no_relation responses must not select candidates")
        return MappingSelection(status)
    if not all(isinstance(item, str) for item in (subject, relation, obj)):
        raise ValueError("mapped response requires string subject, relation, and object IDs")
    _validate_id(subject, candidates.subject, "subject")
    _validate_id(relation, candidates.relation, "relation")
    _validate_id(obj, candidates.object, "object")
    if condition is not None and (not isinstance(condition, int) or isinstance(condition, bool) or condition < 0 or condition >= len(candidates.condition)):
        raise ValueError("mapper selected invalid condition index")
    confidence = value.get("confidence")
    if confidence is not None and (not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0.0 <= float(confidence) <= 1.0):
        raise ValueError("mapper confidence must be between 0 and 1")
    return MappingSelection("mapped", subject, relation, obj, condition, float(confidence) if confidence is not None else None)


def _candidate_json(candidates: tuple[CanonicalCandidate, ...]) -> str:
    return json.dumps([{"id": item.id, "score": item.score, "reason": item.reason} for item in candidates])


def _validate_id(value: str, candidates: tuple[CanonicalCandidate, ...], label: str) -> None:
    if value not in {item.id for item in candidates}:
        raise ValueError(f"mapper selected invalid {label} ID")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"mapper response has duplicate field: {key}")
        result[key] = value
    return result
