"""Grounded qualifier-cue candidates and constrained node qualification."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Any, Callable, Iterable, Mapping

from pipeline.semantic_ir import (
    ComparativeDegree, Conditionality, ModelDecisionProvenance, Modality, Polarity, Restriction,
    QualifierAmbiguity, QualifierAmbiguityState, QualifierCue, QualifierKind,
    SemanticNode, SemanticQualifiers, SourceSpan, TemporalScope, Uncertainty,
    content_sha256,
)
from pipeline.semantic_source import SemanticSourceWindow


QUALIFIER_CATALOG_VERSION = "phase2f-qualifier-catalog-v2"
QUALIFIER_PROMPT_VERSION = "phase2f-qualifier-classifier-v2"
QUALIFIER_SYSTEM = (
    "Return strict JSON only. Preserve explicit source qualifiers for one mention. "
    "Use only supplied cue IDs and closed values. Do not infer strategic concepts or source offsets."
)
QUALIFIER_STATUSES = frozenset({"OK", "NONE", "UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"})
QUALIFIER_FIELD_STATUSES = frozenset({
    "ASSERTED", "NONE", "UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE",
})

_FIELD_TYPES = {
    "polarity": (QualifierKind.POLARITY, Polarity),
    "modality": (QualifierKind.MODALITY, Modality),
    "temporal_scope": (QualifierKind.TEMPORAL_SCOPE, TemporalScope),
    "conditionality": (QualifierKind.CONDITIONALITY, Conditionality),
    "comparative_degree": (QualifierKind.COMPARATIVE_DEGREE, ComparativeDegree),
    "uncertainty": (QualifierKind.UNCERTAINTY, Uncertainty),
    "restriction": (QualifierKind.RESTRICTION, Restriction),
}
_CUES: tuple[tuple[str, tuple[QualifierKind, ...]], ...] = (
    (r"\b(?:only\s+if|as\s+long\s+as|provided(?:\s+that)?|assuming(?:\s+that)?|in\s+case)\b",
     (QualifierKind.CONDITIONALITY,)),
    (r"\bnot\s+only\b|(?<!not )\bonly\b(?!\s+if\b)", (QualifierKind.RESTRICTION,)),
    (r"\b(?:can(?:not|['’]t|t)|couldn(?:['’]t|t)|won(?:['’]t|t)|wouldn(?:['’]t|t)|shouldn(?:['’]t|t)|mustn(?:['’]t|t)|needn(?:['’]t|t)|unable\s+to|not\s+able\s+to)\b",
     (QualifierKind.POLARITY, QualifierKind.MODALITY)),
    (r"\b(?:(?:did|does|do|is|are|was|were|has|have|had)n(?:['’]t|t)|never|not|no\s+longer|no|without)\b",
     (QualifierKind.POLARITY,)),
    (r"\b(?:can|could|may|might|must|should|would|will|shall|ought\s+to|able\s+to|has\s+to|have\s+to|had\s+to|needs?\s+to)\b",
     (QualifierKind.MODALITY,)),
    (r"\b(?:may|might|could)\b", (QualifierKind.UNCERTAINTY,)),
    (r"\b(?:if|unless|when|whenever|once|while)\b", (QualifierKind.CONDITIONALITY,)),
    (r"\b(?:when|whenever|once|while|before|after|until|during|since|by|first|then|later|earlier|previously|already|yet|still|again|currently|now|always|never|usually|sometimes|often|rarely|no\s+longer|will|won(?:['’]t|t))\b",
     (QualifierKind.TEMPORAL_SCOPE,)),
    (r"\b(?:same|equal|more|less|greater|fewer|better|worse|most|least|higher|lower|longer|shorter|faster|slower|closer|farther|best|worst|highest|lowest|longest|shortest|fastest|slowest|closest|farthest|than)\b",
     (QualifierKind.COMPARATIVE_DEGREE,)),
    (r"\b(?:probably|likely)\b", (QualifierKind.MODALITY,)),
    (r"\b(?:maybe|perhaps|probably|likely|unlikely|usually|sometimes|often|rarely|roughly|approximately|about)\b",
     (QualifierKind.UNCERTAINTY,)),
)


@dataclass(frozen=True)
class QualifierCandidate:
    candidate_id: str
    window_id: str
    kind: QualifierKind
    start: int
    end: int
    absolute_start: int
    absolute_end: int
    source_text: str
    version: str = QUALIFIER_CATALOG_VERSION

    def validate(self, window: SemanticSourceWindow) -> None:
        window.validate()
        if self.version != QUALIFIER_CATALOG_VERSION or self.window_id != window.window_id:
            raise ValueError("qualifier candidate identity/version is invalid")
        if not isinstance(self.kind, QualifierKind):
            raise ValueError("qualifier candidate kind is invalid")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (
            self.start, self.end, self.absolute_start, self.absolute_end,
        )):
            raise ValueError("qualifier candidate offsets must be integers")
        if not 0 <= self.start < self.end <= len(window.text):
            raise ValueError("qualifier candidate offsets are invalid")
        if (self.absolute_start, self.absolute_end) != (
            window.source_start + self.start, window.source_start + self.end,
        ):
            raise ValueError("qualifier candidate absolute offsets are invalid")
        if window.text[self.start:self.end] != self.source_text:
            raise ValueError("qualifier candidate is not an exact source slice")
        if self.candidate_id != _candidate_id(window.window_id, self.kind, self.start, self.end, self.source_text):
            raise ValueError("qualifier candidate ID is not stable")


@dataclass(frozen=True)
class QualifierFieldSelection:
    status: str
    value: str | None
    cue_ids: tuple[str, ...]
    confidence: float
    candidate_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in QUALIFIER_FIELD_STATUSES:
            raise ValueError("qualifier field status is invalid")
        if self.value is not None and not isinstance(self.value, str):
            raise ValueError("qualifier field value must be a string or null")
        if not isinstance(self.cue_ids, tuple) or any(
            not isinstance(item, str) or not item for item in self.cue_ids
        ) or len(set(self.cue_ids)) != len(self.cue_ids):
            raise ValueError("qualifier field cue IDs must be a unique immutable tuple")
        if not isinstance(self.candidate_values, tuple) or any(
            not isinstance(item, str) or not item for item in self.candidate_values
        ) or len(set(self.candidate_values)) != len(self.candidate_values):
            raise ValueError("qualifier field candidate values must be a unique immutable tuple")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)) \
                or not 0 <= self.confidence <= 1:
            raise ValueError("qualifier field confidence must be between zero and one")


@dataclass(frozen=True)
class QualifierSelectionResult:
    node_id: str
    status: str
    fields: tuple[tuple[str, QualifierFieldSelection], ...]
    raw_output: str
    parsed_output: Mapping[str, Any] | None
    candidate_ids: tuple[str, ...]
    request_json: str
    model_id: str
    configuration_sha256: str
    failure: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("qualifier result node ID is invalid")
        if self.status not in QUALIFIER_STATUSES:
            raise ValueError("qualifier result status is invalid")
        if not isinstance(self.fields, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2
            or not isinstance(item[0], str) or not isinstance(item[1], QualifierFieldSelection)
            for item in self.fields
        ):
            raise ValueError("qualifier result fields must be an immutable typed tuple")
        if self.parsed_output is not None and not isinstance(self.parsed_output, Mapping):
            raise ValueError("qualifier parsed output must be a mapping or null")
        if not isinstance(self.raw_output, str) or not isinstance(self.request_json, str):
            raise ValueError("qualifier result raw/request output must be retained strings")
        if not isinstance(self.candidate_ids, tuple) or any(
            not isinstance(item, str) or not item for item in self.candidate_ids
        ) or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("qualifier result candidate IDs must be a unique immutable tuple")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("qualifier result model ID is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", self.configuration_sha256):
            raise ValueError("qualifier result configuration hash is invalid")
        if self.failure is not None and (not isinstance(self.failure, str) or not self.failure):
            raise ValueError("qualifier result failure must be null or a non-empty string")


def generate_qualifier_candidates(window: SemanticSourceWindow) -> tuple[QualifierCandidate, ...]:
    window.validate()
    found: dict[tuple[int, int, QualifierKind], str] = {}
    for pattern, kinds in _CUES:
        for match in re.finditer(pattern, window.text, re.IGNORECASE):
            for kind in kinds:
                found[(match.start(), match.end(), kind)] = match.group()
    candidates = tuple(QualifierCandidate(
        _candidate_id(window.window_id, kind, start, end, text), window.window_id,
        kind, start, end, window.source_start + start, window.source_start + end, text,
    ) for (start, end, kind), text in sorted(
        found.items(), key=lambda item: (item[0][0], item[0][1], item[0][2].value),
    ))
    for candidate in candidates:
        candidate.validate(window)
    return candidates


def qualifier_candidates_for_node(
    window: SemanticSourceWindow,
    node: SemanticNode,
    catalog: tuple[QualifierCandidate, ...],
) -> tuple[QualifierCandidate, ...]:
    _validate_node(window, node)
    _require_complete_window_catalog(window, catalog)
    segment_indexes = {
        index for index, segment in enumerate(window.segments)
        if segment.start < node.source_span.local_end and node.source_span.local_start < segment.end
    }
    if not segment_indexes:
        raise ValueError("semantic node is outside Pass 0 segments")
    selected = []
    for candidate in catalog:
        cue_indexes = {
            index for index, segment in enumerate(window.segments)
            if segment.start < candidate.end and candidate.start < segment.end
        }
        distance = min(abs(a - b) for a in segment_indexes for b in cue_indexes)
        character_distance = max(
            node.source_span.local_start - candidate.end,
            candidate.start - node.source_span.local_end,
            0,
        )
        if distance <= 1 and character_distance <= 160:
            selected.append(candidate)
    return tuple(selected)


def qualifier_prompt(
    window: SemanticSourceWindow,
    node: SemanticNode,
    candidates: tuple[QualifierCandidate, ...],
) -> str:
    values = [{
        "id": item.candidate_id, "text": item.source_text, "kind": item.kind.value,
        "start": item.start, "end": item.end,
        "compatible_values": sorted(_compatible_values(item)),
    } for item in candidates]
    allowed = {
        field: [item.value for item in enum_type if item.value != "UNKNOWN"]
        for field, (_, enum_type) in _FIELD_TYPES.items()
    }
    empty = {
        field: {
            "status": "NONE", "value": None, "cue_ids": [],
            "candidate_values": [], "confidence": 0.0,
        }
        for field in _FIELD_TYPES
    }
    return (
        f"SOURCE WINDOW:\n{window.text}\n"
        f"MENTION {node.node_type.value} [{node.source_span.local_start},{node.source_span.local_end}): "
        f"{node.source_span.text}\n"
        f"CONTEXT CUES: {json.dumps(values, ensure_ascii=False, separators=(',', ':'))}\n"
        f"ALLOWED VALUES: {json.dumps(allowed, separators=(',', ':'))}\n"
        "Classify only qualifiers that apply to this mention. Each field status is "
        "ASSERTED|NONE|UNKNOWN|AMBIGUOUS|INSUFFICIENT_EVIDENCE. For AMBIGUOUS retain at least "
        "two allowed candidate_values and exact cue IDs. ASSERTED requires confidence >= 0.5. "
        "If any field is ASSERTED the aggregate status is OK; otherwise precedence is "
        "AMBIGUOUS, then UNKNOWN, then INSUFFICIENT_EVIDENCE, then NONE. UNKNOWN is preferable "
        "to invention. "
        "A conditional or temporal qualifier only annotates this mention; it never replaces the "
        "antecedent/anchor mention and CONDITION/TEMPORAL graph edge. Quantities and durations are "
        "represented by source nodes, not generated here. "
        "Return exactly " + json.dumps({"status": "OK|NONE|UNKNOWN|AMBIGUOUS|INSUFFICIENT_EVIDENCE",
                                         "qualifiers": empty}, separators=(",", ":"))
    )


def classify_node_qualifiers(
    window: SemanticSourceWindow,
    node: SemanticNode,
    candidates: tuple[QualifierCandidate, ...],
    chat: Callable[..., str],
    *,
    model: str,
    configuration: Mapping[str, Any],
    max_tokens: int = 512,
    thinking: str | None = None,
) -> QualifierSelectionResult:
    _validate_node_catalog(window, node, candidates)
    if not isinstance(model, str) or not model.strip() or not isinstance(configuration, Mapping):
        raise ValueError("qualifier classifier model/configuration is invalid")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("qualifier classifier max_tokens must be positive")
    prompt = qualifier_prompt(window, node, candidates)
    effective = {
        "caller_configuration": dict(configuration), "temperature": 0.0, "max_tokens": max_tokens,
        "model": model, "thinking": thinking, "prompt_version": QUALIFIER_PROMPT_VERSION,
        "catalog_version": QUALIFIER_CATALOG_VERSION,
    }
    request = {"system": QUALIFIER_SYSTEM, "user": prompt, **effective}
    request_json = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    config_hash = content_sha256(effective)
    candidate_ids = tuple(item.candidate_id for item in candidates)
    try:
        raw = chat(
            system=QUALIFIER_SYSTEM, user=prompt, temperature=0.0, max_tokens=max_tokens,
            model=model, thinking=thinking,
        )
    except Exception as exc:
        return QualifierSelectionResult(
            node.node_id, "INSUFFICIENT_EVIDENCE", (), "", None, candidate_ids,
            request_json, model, config_hash, f"QualifierProviderError:{type(exc).__name__}",
        )
    try:
        status, fields, body = parse_qualifier_selection(raw, candidates)
    except Exception as exc:
        retained_raw = raw if isinstance(raw, str) else repr(raw)
        return QualifierSelectionResult(
            node.node_id, "INSUFFICIENT_EVIDENCE", (), retained_raw, None, candidate_ids,
            request_json, model, config_hash, type(exc).__name__,
        )
    return QualifierSelectionResult(
        node.node_id, status, fields, raw, body, candidate_ids, request_json, model, config_hash,
    )


def parse_qualifier_selection(
    raw: str, candidates: tuple[QualifierCandidate, ...],
) -> tuple[str, tuple[tuple[str, QualifierFieldSelection], ...], Mapping[str, Any]]:
    body = _strict_object(raw)
    if set(body) != {"status", "qualifiers"} or body.get("status") not in QUALIFIER_STATUSES:
        raise ValueError("qualifier selection envelope is invalid")
    raw_fields = body.get("qualifiers")
    if not isinstance(raw_fields, Mapping) or set(raw_fields) != set(_FIELD_TYPES):
        raise ValueError("qualifier selection fields are invalid")
    by_id = {item.candidate_id: item for item in candidates}
    fields = []
    any_asserted = False
    for field, (kind, enum_type) in _FIELD_TYPES.items():
        item = raw_fields[field]
        if not isinstance(item, Mapping) or set(item) != {
            "status", "value", "cue_ids", "candidate_values", "confidence",
        }:
            raise ValueError("qualifier field shape is invalid")
        field_status = item.get("status")
        value, cue_ids = item.get("value"), item.get("cue_ids")
        candidate_values, confidence = item.get("candidate_values"), item.get("confidence")
        if field_status not in QUALIFIER_FIELD_STATUSES:
            raise ValueError("qualifier field status is invalid")
        if not isinstance(cue_ids, list) or len(set(cue_ids)) != len(cue_ids):
            raise ValueError("qualifier cue IDs must be a unique list")
        if not isinstance(candidate_values, list) or len(set(candidate_values)) != len(candidate_values):
            raise ValueError("qualifier candidate values must be a unique list")
        if any(cue_id not in by_id or by_id[cue_id].kind is not kind for cue_id in cue_ids):
            raise ValueError("qualifier cue ID has the wrong kind or is unknown")
        parsed_candidates = []
        for candidate_value in candidate_values:
            try:
                parsed_candidate = enum_type(candidate_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("qualifier candidate value is outside the closed vocabulary") from exc
            if parsed_candidate.value == "UNKNOWN":
                raise ValueError("UNKNOWN is a field status, not a candidate qualifier value")
            parsed_candidates.append(parsed_candidate.value)
        if field_status == "ASSERTED":
            try:
                parsed_value = enum_type(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("qualifier value is outside the closed vocabulary") from exc
            if parsed_value.value == "UNKNOWN" or not cue_ids or candidate_values:
                raise ValueError("asserted qualifier requires one closed value and exact cues")
            if confidence is None or isinstance(confidence, bool) or not isinstance(confidence, (int, float)) \
                    or not 0.5 <= confidence <= 1:
                raise ValueError("asserted qualifier confidence is below the acceptance threshold")
            if any(parsed_value.value not in _compatible_values(by_id[cue_id]) for cue_id in cue_ids):
                raise ValueError("qualifier value contradicts its retained source cue")
            any_asserted = True
        elif field_status == "AMBIGUOUS":
            if value is not None or len(parsed_candidates) < 2 or not cue_ids:
                raise ValueError("ambiguous qualifier requires cues and at least two candidate values")
            if any(
                candidate_value not in _compatible_values(by_id[cue_id])
                for cue_id in cue_ids for candidate_value in parsed_candidates
            ):
                raise ValueError("ambiguous qualifier candidates contradict their retained cue")
        else:
            if value is not None or candidate_values:
                raise ValueError("unasserted qualifier cannot smuggle a value")
            if field_status == "NONE" and cue_ids:
                raise ValueError("NONE qualifier cannot retain cue evidence")
        selection = QualifierFieldSelection(
            str(field_status), value, tuple(cue_ids), confidence, tuple(parsed_candidates),
        )
        fields.append((field, selection))
    status = str(body["status"])
    expected_status = _aggregate_field_status(tuple(value for _, value in fields))
    if status != expected_status or (status == "OK") != any_asserted:
        raise ValueError("qualifier aggregate status contradicts its field decisions")
    return status, tuple(fields), body


def apply_node_qualifiers(
    window: SemanticSourceWindow,
    node: SemanticNode,
    candidates: tuple[QualifierCandidate, ...],
    result: QualifierSelectionResult,
) -> SemanticNode:
    request = validate_qualifier_selection_result(window, node, candidates, result)
    if result.failure:
        return _append_decision_provenance(node, result, request)
    status, fields, body = parse_qualifier_selection(result.raw_output, candidates)
    # ``validate_qualifier_selection_result`` proved this equality; retaining
    # the values here keeps construction explicit and type-directed.
    assert status == result.status and fields == result.fields and body == result.parsed_output
    by_id = {item.candidate_id: item for item in candidates}
    values = {
        field: selection.value if selection.status == "ASSERTED" else None
        for field, selection in fields
    }
    cues = tuple(QualifierCue(
        _FIELD_TYPES[field][0], SourceSpan(
            window.source_id, window.window_id, by_id[cue_id].start, by_id[cue_id].end,
            by_id[cue_id].source_text, by_id[cue_id].absolute_start, by_id[cue_id].absolute_end,
            window.speaker, window.start_ms, window.end_ms,
        ),
    ) for field, selection in fields if selection.status == "ASSERTED" for cue_id in selection.cue_ids)
    ambiguities = tuple(QualifierAmbiguity(
        _FIELD_TYPES[field][0], QualifierAmbiguityState(selection.status),
        tuple(QualifierCue(
            _FIELD_TYPES[field][0], SourceSpan(
                window.source_id, window.window_id, by_id[cue_id].start, by_id[cue_id].end,
                by_id[cue_id].source_text, by_id[cue_id].absolute_start, by_id[cue_id].absolute_end,
                window.speaker, window.start_ms, window.end_ms,
            ),
        ) for cue_id in selection.cue_ids),
        selection.candidate_values, selection.confidence,
    ) for field, selection in fields if selection.status in {
        "UNKNOWN", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE",
    })
    qualifiers = SemanticQualifiers(
        polarity=Polarity(values["polarity"] or Polarity.UNKNOWN.value),
        negated=values["polarity"] == Polarity.NEGATIVE.value,
        modality=Modality(values["modality"] or Modality.UNKNOWN.value),
        temporal_scope=TemporalScope(values["temporal_scope"] or TemporalScope.UNKNOWN.value),
        conditionality=Conditionality(values["conditionality"] or Conditionality.UNKNOWN.value),
        comparative_degree=ComparativeDegree(values["comparative_degree"] or ComparativeDegree.UNKNOWN.value),
        uncertainty=Uncertainty(values["uncertainty"] or Uncertainty.UNKNOWN.value),
        restriction=Restriction(values["restriction"] or Restriction.UNKNOWN.value),
        cues=cues,
        ambiguities=ambiguities,
    )
    updated = replace(node, qualifiers=qualifiers)
    return _append_decision_provenance(updated, result, request)


def validate_qualifier_selection_result(
    window: SemanticSourceWindow,
    node: SemanticNode,
    candidates: tuple[QualifierCandidate, ...],
    result: QualifierSelectionResult,
) -> Mapping[str, Any]:
    """Validate one retained model decision without applying its semantics."""
    if result.node_id != node.node_id:
        raise ValueError("qualifier result belongs to a different semantic node")
    _validate_node_catalog(window, node, candidates)
    if result.candidate_ids != tuple(item.candidate_id for item in candidates):
        raise ValueError("qualifier result does not exactly cover its retained cue catalog")
    request = _validate_result_request(window, node, candidates, result)
    if result.failure:
        _validate_failure(result, candidates)
        return request
    status, fields, body = parse_qualifier_selection(result.raw_output, candidates)
    if status != result.status or fields != result.fields or body != result.parsed_output:
        raise ValueError("qualifier raw output contradicts its retained decision")
    return request


def _validate_node(window: SemanticSourceWindow, node: SemanticNode) -> None:
    if not isinstance(node, SemanticNode):
        raise ValueError("qualifier target must be a semantic node")
    node.source_span.validate_against(
        window.source_id, window.window_id, window.text, window_source_start=window.source_start,
        speaker=window.speaker, start_timestamp=window.start_ms, end_timestamp=window.end_ms,
    )


def _require_complete_window_catalog(
    window: SemanticSourceWindow, catalog: tuple[QualifierCandidate, ...],
) -> None:
    if not isinstance(catalog, tuple):
        raise ValueError("qualifier candidate catalog must be an immutable tuple")
    expected = generate_qualifier_candidates(window)
    if catalog != expected:
        raise ValueError("qualifier catalog is not the complete deterministic window catalog")


def _validate_node_catalog(
    window: SemanticSourceWindow,
    node: SemanticNode,
    candidates: tuple[QualifierCandidate, ...],
) -> None:
    _validate_node(window, node)
    expected = qualifier_candidates_for_node(window, node, generate_qualifier_candidates(window))
    if candidates != expected:
        raise ValueError("qualifier catalog is not the complete deterministic node-local catalog")


def _aggregate_field_status(fields: tuple[QualifierFieldSelection, ...]) -> str:
    statuses = {item.status for item in fields}
    if "ASSERTED" in statuses:
        return "OK"
    if "AMBIGUOUS" in statuses:
        return "AMBIGUOUS"
    if "UNKNOWN" in statuses:
        return "UNKNOWN"
    if "INSUFFICIENT_EVIDENCE" in statuses:
        return "INSUFFICIENT_EVIDENCE"
    return "NONE"


def _compatible_values(candidate: QualifierCandidate) -> frozenset[str]:
    text = candidate.source_text.casefold().replace("’", "'")
    if candidate.kind is QualifierKind.POLARITY:
        return frozenset({Polarity.NEGATIVE.value})
    if candidate.kind is QualifierKind.CONDITIONALITY:
        return frozenset({
            Conditionality.CONDITIONAL.value,
            Conditionality.HYPOTHETICAL.value,
            Conditionality.COUNTERFACTUAL.value,
        })
    if candidate.kind is QualifierKind.RESTRICTION:
        return frozenset({
            Restriction.ADDITIVE.value if text == "not only" else Restriction.EXCLUSIVE.value,
        })
    if candidate.kind is QualifierKind.MODALITY:
        if re.search(r"\b(?:must|has to|have to|had to|needs? to|mustn't|needn't)\b", text):
            return frozenset({Modality.NECESSARY.value, Modality.OBLIGATORY.value})
        if re.search(r"\b(?:should|shouldn't|ought to)\b", text):
            return frozenset({Modality.OBLIGATORY.value})
        if re.search(r"\b(?:would|wouldn't)\b", text):
            return frozenset({Modality.POSSIBLE.value, Modality.COUNTERFACTUAL.value})
        if re.search(r"\b(?:probably|likely)\b", text):
            return frozenset({Modality.PROBABLE.value})
        if re.search(r"\b(?:will|won't|wont|shall)\b", text):
            return frozenset({Modality.ASSERTED.value})
        return frozenset({Modality.POSSIBLE.value})
    if candidate.kind is QualifierKind.TEMPORAL_SCOPE:
        if text in {"always", "never", "usually", "sometimes", "often", "rarely"}:
            return frozenset({TemporalScope.HABITUAL.value})
        if text in {"currently", "now"}:
            return frozenset({TemporalScope.PRESENT.value})
        if text in {"later", "will", "won't", "wont"}:
            return frozenset({TemporalScope.FUTURE.value})
        if text in {"earlier", "previously", "already"}:
            return frozenset({TemporalScope.PAST.value})
        if text in {"still", "since", "while"}:
            return frozenset({TemporalScope.ONGOING.value, TemporalScope.BOUNDED.value})
        return frozenset({TemporalScope.BOUNDED.value})
    if candidate.kind is QualifierKind.COMPARATIVE_DEGREE:
        if text in {"same", "equal"}:
            return frozenset({ComparativeDegree.EQUAL.value})
        if text in {"most", "best", "highest", "longest", "fastest", "closest"}:
            return frozenset({ComparativeDegree.MAXIMUM.value})
        if text in {"least", "worst", "lowest", "shortest", "slowest"}:
            return frozenset({ComparativeDegree.MINIMUM.value})
        if text == "farthest":
            return frozenset({ComparativeDegree.MAXIMUM.value})
        if text in {"less", "fewer", "worse", "lower", "shorter", "slower"}:
            return frozenset({ComparativeDegree.LESS.value})
        if text == "than":
            return frozenset({ComparativeDegree.GREATER.value, ComparativeDegree.LESS.value})
        return frozenset({ComparativeDegree.GREATER.value})
    if candidate.kind is QualifierKind.UNCERTAINTY:
        if text in {"probably", "likely", "usually"}:
            return frozenset({Uncertainty.LIKELY.value})
        if text in {"maybe", "perhaps", "may", "might", "could"}:
            return frozenset({Uncertainty.POSSIBLE.value})
        return frozenset({Uncertainty.UNCERTAIN.value})
    raise ValueError("unsupported qualifier cue kind")


def _validate_result_request(
    window: SemanticSourceWindow,
    node: SemanticNode,
    candidates: tuple[QualifierCandidate, ...],
    result: QualifierSelectionResult,
) -> Mapping[str, Any]:
    request = _strict_object(result.request_json)
    expected_keys = {
        "system", "user", "caller_configuration", "temperature", "max_tokens", "model",
        "thinking", "prompt_version", "catalog_version",
    }
    if set(request) != expected_keys or not isinstance(request.get("caller_configuration"), Mapping):
        raise ValueError("qualifier request artifact has an invalid shape")
    temperature = request.get("temperature")
    max_tokens = request.get("max_tokens")
    thinking = request.get("thinking")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature != 0.0:
        raise ValueError("qualifier request constants are invalid")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("qualifier request max_tokens is invalid")
    if thinking is not None and (not isinstance(thinking, str) or not thinking.strip()):
        raise ValueError("qualifier request thinking mode is invalid")
    if (
        request.get("system") != QUALIFIER_SYSTEM
        or request.get("user") != qualifier_prompt(window, node, candidates)
        or request.get("prompt_version") != QUALIFIER_PROMPT_VERSION
        or request.get("catalog_version") != QUALIFIER_CATALOG_VERSION
        or request.get("model") != result.model_id
    ):
        raise ValueError("qualifier request constants are invalid")
    effective = {key: request[key] for key in (
        "caller_configuration", "temperature", "max_tokens", "model", "thinking",
        "prompt_version", "catalog_version",
    )}
    if result.configuration_sha256 != content_sha256(effective):
        raise ValueError("qualifier request provenance is invalid")
    return request


def _validate_failure(
    result: QualifierSelectionResult, candidates: tuple[QualifierCandidate, ...],
) -> None:
    if result.status != "INSUFFICIENT_EVIDENCE" or result.fields or result.parsed_output is not None:
        raise ValueError("failed qualifier decision cannot retain accepted semantics")
    if result.failure.startswith("QualifierProviderError:"):
        if result.raw_output or re.fullmatch(r"QualifierProviderError:[A-Za-z_][A-Za-z0-9_]*", result.failure) is None:
            raise ValueError("qualifier provider failure evidence is invalid")
        return
    try:
        parse_qualifier_selection(result.raw_output, candidates)
    except Exception as exc:
        if result.failure != type(exc).__name__:
            raise ValueError("qualifier parse failure taxonomy is invalid") from exc
        return
    raise ValueError("qualifier parse failure output is actually valid")


def _append_decision_provenance(
    node: SemanticNode,
    result: QualifierSelectionResult,
    request: Mapping[str, Any],
) -> SemanticNode:
    decision_output = {
        "raw_output": result.raw_output,
        "status": result.status,
        "failure": result.failure,
    }
    raw_hash = content_sha256(decision_output)
    decision_id = (
        f"{node.node_id}:qualifiers:{result.configuration_sha256[:12]}:{raw_hash[:12]}:"
        f"{result.status.lower()}"
    )
    provenance = ModelDecisionProvenance(
        decision_id, result.model_id, QUALIFIER_PROMPT_VERSION,
        result.configuration_sha256, content_sha256(request), raw_hash, result.candidate_ids,
    )
    existing = {item.decision_id: item for item in node.additional_provenance}
    prior_qualifier_decisions = {
        item.decision_id: item for item in node.additional_provenance
        if ":qualifiers:" in item.decision_id
    }
    if prior_qualifier_decisions and decision_id not in prior_qualifier_decisions:
        raise ValueError("a semantic node may have only one qualifier decision")
    if decision_id in existing:
        if existing[decision_id] != provenance:
            raise ValueError("qualifier decision ID collides with different provenance")
        return node
    return replace(
        node, additional_provenance=node.additional_provenance + (provenance,),
    )


def qualifier_candidate_coverage(
    window: SemanticSourceWindow,
    candidates: tuple[QualifierCandidate, ...],
    reviewed: Iterable[tuple[int, int, str]],
) -> dict[str, dict[str, int | float]]:
    window.validate()
    _require_complete_window_catalog(window, candidates)
    labels = tuple(reviewed)
    buckets = {kind.value.lower(): [0, 0] for kind in QualifierKind}
    buckets["negation"] = [0, 0]
    offered = {(item.start, item.end, item.kind.value) for item in candidates}
    for start, end, label in labels:
        kind_label = QualifierKind.POLARITY.value if label == "NEGATION" else label
        try:
            kind = QualifierKind(kind_label)
        except ValueError as exc:
            raise ValueError("reviewed qualifier kind is invalid") from exc
        if not 0 <= start < end <= len(window.text):
            raise ValueError("reviewed qualifier span is invalid")
        bucket = "negation" if label == "NEGATION" else kind.value.lower()
        buckets[bucket][1] += 1
        if (start, end, kind.value) in offered:
            buckets[bucket][0] += 1
    return {
        key: {"hit_count": hit, "denominator": total, "recall": hit / total if total else 0.0}
        for key, (hit, total) in sorted(buckets.items())
    }


def _candidate_id(window_id: str, kind: QualifierKind, start: int, end: int, text: str) -> str:
    payload = json.dumps(
        [QUALIFIER_CATALOG_VERSION, window_id, kind.value, start, end, text],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return f"{window_id}:q{hashlib.sha256(payload).hexdigest()[:20]}"


def _strict_object(raw: str) -> Mapping[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON contains duplicate keys")
            result[key] = value
        return result
    if not isinstance(raw, str):
        raise ValueError("qualifier classifier output must be a string")
    try:
        body = json.loads(raw, object_pairs_hook=unique)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("qualifier classifier returned malformed JSON") from exc
    if not isinstance(body, Mapping):
        raise ValueError("qualifier classifier must return a JSON object")
    return body
