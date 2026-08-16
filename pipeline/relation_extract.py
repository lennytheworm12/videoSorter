"""Phase 2 compiler from source-grounded insights to strategic relations.

The LLM is only allowed to propose constrained candidates. This module validates
and canonicalizes them before they become Phase 1 StrategicRelation objects.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Mapping

from core.ontology import ENTITY_TYPES, ONTOLOGY_VERSION, RELATION_TYPES, STRATEGIC_CONCEPTS
from core.champions import champions_mentioned
from core.relation_normalization import (
    canonical_condition,
    canonical_entity,
    canonical_relation_type,
    concept_is_mentioned,
    normalized_key,
)
from core.strategic_types import (
    AUTOMATED_RELATION_DATA_VERSION,
    ConditionEvent,
    EvidenceRef,
    RelationAlignment,
    StrategicRelation,
    StrategicValidationError,
    relation_types_conflict,
)


EXTRACTION_PROMPT_VERSION = "strategic-relation-extraction-v1-source-aligned"
DEFAULT_ACCEPTANCE_THRESHOLD = 0.60


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


DEFAULT_MAX_OUTPUT_TOKENS = _positive_env_int("RELATION_EXTRACTION_MAX_TOKENS", 4096)
DEEPSEEK_THINKING_MODE = os.environ.get("RELATION_EXTRACTION_DEEPSEEK_THINKING", "disabled")
if DEEPSEEK_THINKING_MODE not in {"enabled", "disabled"}:
    raise RuntimeError("RELATION_EXTRACTION_DEEPSEEK_THINKING must be enabled or disabled")


def _model_env(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    if not value:
        raise RuntimeError(f"{name} must not be blank")
    return value


RELATION_FLASH_MODEL = _model_env("DEEPSEEK_RELATION_FLASH_MODEL", "deepseek-v4-flash")
RELATION_PRO_MODEL = _model_env("DEEPSEEK_RELATION_PRO_MODEL", "deepseek-v4-pro")

RELATION_EXTRACTION_SYSTEM = """You are a constrained strategic relation compiler.
Return JSON only. Derive only relations supported by the supplied evidence.
Do not use League knowledge not stated or directly implied by the evidence.
Keep conditions such as availability, misses, timing, targets, and exceptions.
Copy condition wording and temporal order from the source evidence. Do not
replace source ability names/slots with aliases inside a condition and do not
invent a condition that is not stated in the cited evidence.
Use only supplied entity types, strategic concepts, and relation types. Return
an empty relations list when no supported causal relation exists. Every relation
must cite one or more supplied evidence IDs. Do not turn generic advice or
co-occurrence into causation. Never create principles or matchup answers."""

RELATION_EXTRACTION_USER = """Ontology version: {ontology_version}
Allowed relation types: {relation_types}
Allowed entity types: {entity_types}
Strategic concepts: {concepts}
Recognized ability aliases: {ability_aliases}
Recognized non-concept entity aliases: {entity_aliases}

SOURCE EVIDENCE (the only factual basis):
{evidence}

For each canonical field, include the exact source phrase and the evidence ID
where that phrase occurs. The source phrase must be copied from the supplied
evidence, not paraphrased. Canonical terms do not need to appear literally,
but their source phrase must support the mapping. `subject_type` and
`object_type` must be exactly an allowed entity type. Do not emit action,
attribute, opportunity, environment, or any other free-form node type. Use an
allowed canonical ability, champion, or strategic concept only when source
evidence supports it. A non-core allowed entity type is permitted only when its
exact value appears in Recognized non-concept entity aliases. Otherwise return
zero relations. Return exactly this JSON
shape:
{{"relations":[{{"subject":"...","subject_type":"...","relation_type":"...","object":"...","object_type":"...","condition":null,"condition_event":null,"effect":null,"concepts":["..."],"provenance_type":"source_claim|coach_supported_inference","evidence_ids":["..."],"extraction_confidence":0.0,"patch_sensitivity":"very_low|low|medium|high","grounding":{{"subject":{{"source_text":"...","evidence_id":"..."}},"predicate":{{"source_text":"...","evidence_id":"..."}},"object":{{"source_text":"...","evidence_id":"..."}},"condition":null}}}}]}}
"""


@dataclass(frozen=True)
class EvidenceItem:
    insight_id: str
    source_id: str
    text: str
    source_type: str = "insight"
    source_score: float | None = None
    cluster_score: float | None = None
    confidence: float | None = None
    patch_sensitivity: str = "low"

    def validate(self) -> None:
        if not self.insight_id or not self.source_id or not self.text.strip() or not self.source_type.strip():
            raise ValueError("evidence requires insight_id, source_id, and text")

    @property
    def evidence_quality(self) -> float:
        value = self.confidence if self.confidence is not None else self.source_score
        return min(1.0, max(0.0, float(value if value is not None else 0.5)))


@dataclass(frozen=True)
class ExtractionPacket:
    evidence: tuple[EvidenceItem, ...]
    ability_aliases: Mapping[str, str] = field(default_factory=dict)
    entity_aliases: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    ontology_version: str = ONTOLOGY_VERSION
    prompt_version: str = EXTRACTION_PROMPT_VERSION

    def validate(self) -> None:
        if self.ontology_version != ONTOLOGY_VERSION:
            raise ValueError("stale ontology version in extraction packet")
        if not self.evidence:
            raise ValueError("extraction packet requires evidence")
        ids = [item.insight_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("extraction packet has duplicate evidence IDs")
        for item in self.evidence:
            item.validate()

    def prompt(self) -> str:
        self.validate()
        evidence = "\n".join(
            f"[evidence_id={item.insight_id}; source_id={item.source_id}; quality={item.evidence_quality:.2f}] {item.text}"
            for item in self.evidence
        )
        aliases = ", ".join(f"{key} -> {value}" for key, value in sorted(self.ability_aliases.items())) or "(none)"
        entity_aliases = "; ".join(
            f"{entity_type}: " + ", ".join(f"{key} -> {value}" for key, value in sorted(aliases.items()))
            for entity_type, aliases in sorted(self.entity_aliases.items())
        ) or "(none)"
        return RELATION_EXTRACTION_USER.format(
            ontology_version=self.ontology_version,
            relation_types=", ".join(sorted(RELATION_TYPES)),
            entity_types=", ".join(sorted(ENTITY_TYPES)),
            concepts=", ".join(sorted(STRATEGIC_CONCEPTS)),
            ability_aliases=aliases,
            entity_aliases=entity_aliases,
            evidence=evidence,
        )


@dataclass(frozen=True)
class ExtractionDecision:
    raw: Mapping[str, Any]
    relation: StrategicRelation | None
    status: str
    warnings: tuple[str, ...] = ()
    confidence_components: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionTrace:
    raw_response: str | None
    candidates: tuple[Mapping[str, Any], ...] = ()
    decisions: tuple[ExtractionDecision, ...] = ()
    failure_stage: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    latency_ms: int = 0
    exception: Exception | None = field(default=None, repr=False, compare=False)


def packet_from_insight_ids(db_path: str, insight_ids: tuple[str, ...] | list[str]) -> ExtractionPacket:
    """Build a source-grounded extraction packet from explicit local insight IDs."""
    ids = tuple(str(item).strip() for item in insight_ids if str(item).strip())
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("provide one or more unique insight IDs")
    placeholders = ",".join("?" for _ in ids)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT i.id, i.video_id, i.text, i.source_score, i.cluster_score,
                   i.confidence, v.champion, v.subject
            FROM insights AS i
            JOIN videos AS v ON v.video_id = i.video_id
            WHERE i.id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        by_id = {str(row["id"]): row for row in rows}
        if set(by_id) != set(ids):
            missing = ", ".join(sorted(set(ids) - set(by_id)))
            raise ValueError(f"unknown insight IDs: {missing}")
        evidence = tuple(
            EvidenceItem(
                insight_id=insight_id,
                source_id=by_id[insight_id]["video_id"],
                text=by_id[insight_id]["text"],
                source_score=by_id[insight_id]["source_score"],
                cluster_score=by_id[insight_id]["cluster_score"],
                confidence=by_id[insight_id]["confidence"],
            )
            for insight_id in ids
        )
        champions = _packet_champions(rows)
        aliases = _ability_aliases(conn, champions, "\n".join(item.text for item in evidence))
    return ExtractionPacket(evidence=evidence, ability_aliases=aliases)


def _packet_champions(rows: list[sqlite3.Row]) -> tuple[str, ...]:
    names = []
    for row in rows:
        for value in (row["champion"], row["subject"], row["text"]):
            names.extend(champions_mentioned(value or ""))
    return tuple(dict.fromkeys(names))


def _ability_aliases(conn: sqlite3.Connection, champions: tuple[str, ...], evidence_text: str) -> dict[str, str]:
    """Expose only ability names that the selected evidence explicitly names."""
    if not champions:
        return {}
    placeholders = ",".join("?" for _ in champions)
    rows = conn.execute(
        f"SELECT champion, ability_slot, name FROM champion_abilities WHERE champion IN ({placeholders})",
        champions,
    ).fetchall()
    aliases: dict[str, str] = {}
    for row in rows:
        canonical = f"{row['champion']} {row['ability_slot']}"
        names = [canonical]
        if row["name"]:
            names.extend(part.strip() for part in row["name"].split("/") if part.strip())
            names.append(f"{row['champion']} {row['name']}")
        for name in names:
            if _text_mentions_alias(evidence_text, name):
                aliases[name] = canonical
        # A bare slot ("her E", "their W") is safe only with one grounded
        # champion; otherwise it would fabricate an ambiguous ability binding.
        if len(champions) == 1 and _text_mentions_alias(evidence_text, row["ability_slot"]):
            aliases[row["ability_slot"]] = canonical
    return aliases


def _text_mentions_alias(text: str, alias: str) -> bool:
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", text, re.IGNORECASE))


def parse_model_response(raw: str) -> list[Mapping[str, Any]]:
    """Parse one strict JSON response; malformed or partial responses are rejected."""
    if isinstance(raw, str):
        fenced = re.fullmatch(r"\s*```(?:json)?\s*\n?(.*?)\n?```\s*", raw, re.DOTALL | re.IGNORECASE)
        if fenced:
            raw = fenced.group(1)
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("relation extractor returned malformed JSON") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("relations"), list):
        raise ValueError("relation extractor response requires a relations list")
    if not all(isinstance(item, dict) for item in parsed["relations"]):
        raise ValueError("relation extractor relations must be objects")
    return parsed["relations"]


def compile_candidates(packet: ExtractionPacket, candidates: list[Mapping[str, Any]]) -> tuple[ExtractionDecision, ...]:
    """Validate untrusted candidates without performing an LLM call."""
    packet.validate()
    evidence_by_id = {item.insight_id: item for item in packet.evidence}
    decisions = tuple(
        _compile_candidate(
            candidate,
            evidence_by_id,
            packet.ability_aliases,
            packet.entity_aliases,
            packet.ontology_version,
        )
        for candidate in candidates
    )
    return _flag_structural_contradictions(decisions)


def extract_relations(
    packet: ExtractionPacket,
    chat: Callable[..., str],
    *,
    acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
    model: str | None = None,
) -> tuple[ExtractionDecision, ...]:
    """Call the configured cheap-model adapter, then compile its response safely."""
    trace = extract_relation_trace(
        packet, chat, acceptance_threshold=acceptance_threshold, model=model
    )
    if trace.exception:
        raise trace.exception
    return trace.decisions


def extract_relation_trace(
    packet: ExtractionPacket,
    chat: Callable[..., str],
    *,
    acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
    model: str | None = None,
) -> ExtractionTrace:
    """Run extraction without hiding raw output or the stage of a failure."""
    if not 0.0 <= acceptance_threshold <= 1.0:
        raise ValueError("acceptance_threshold must be between 0 and 1")
    started = time.perf_counter()
    try:
        raw = chat(
            system=RELATION_EXTRACTION_SYSTEM,
            user=packet.prompt(),
            temperature=0.0,
            max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            thinking=DEEPSEEK_THINKING_MODE,
            model=model,
        )
    except Exception as exc:
        return _failed_trace("model_call", None, exc, started)
    try:
        candidates = tuple(parse_model_response(raw))
    except Exception as exc:
        return _failed_trace("parsing", raw, exc, started)
    try:
        decisions = tuple(_apply_threshold(decision, acceptance_threshold) for decision in compile_candidates(packet, list(candidates)))
    except Exception as exc:
        return _failed_trace("validation", raw, exc, started, candidates)
    return ExtractionTrace(raw, candidates, decisions, latency_ms=_elapsed_ms(started))


def _failed_trace(
    stage: str,
    raw: str | None,
    exc: Exception,
    started: float,
    candidates: tuple[Mapping[str, Any], ...] = (),
) -> ExtractionTrace:
    return ExtractionTrace(
        raw, candidates, failure_stage=stage, failure_type=type(exc).__name__,
        failure_message=str(exc), latency_ms=_elapsed_ms(started), exception=exc,
    )


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _compile_candidate(
    candidate: Mapping[str, Any],
    evidence_by_id: Mapping[str, EvidenceItem],
    ability_aliases: Mapping[str, str],
    entity_aliases: Mapping[str, Mapping[str, str]],
    ontology_version: str,
) -> ExtractionDecision:
    evidence_ids = candidate.get("evidence_ids")
    if not isinstance(evidence_ids, list) or not evidence_ids:
        return ExtractionDecision(candidate, None, "rejected", ("missing evidence_ids",))
    if not all(isinstance(item, str) and item in evidence_by_id for item in evidence_ids):
        return ExtractionDecision(candidate, None, "rejected", ("unknown evidence_id",))
    if len(evidence_ids) != len(set(evidence_ids)):
        return ExtractionDecision(candidate, None, "rejected", ("duplicate evidence_id",))
    subject_type = str(candidate.get("subject_type") or "")
    object_type = str(candidate.get("object_type") or "")
    subject = canonical_entity(
        subject_type,
        str(candidate.get("subject") or ""),
        ability_aliases=ability_aliases,
        entity_aliases=entity_aliases.get(subject_type, {}),
    )
    obj = canonical_entity(
        object_type,
        str(candidate.get("object") or ""),
        ability_aliases=ability_aliases,
        entity_aliases=entity_aliases.get(object_type, {}),
    )
    relation_type = canonical_relation_type(candidate.get("relation_type"))
    if subject is None or obj is None or relation_type is None:
        return ExtractionDecision(candidate, None, "rejected", ("unknown entity, concept, or relation type",))
    if subject.entity_type == "concept" and not _concept_is_supported_by_evidence(subject.key, evidence_ids, evidence_by_id):
        return ExtractionDecision(candidate, None, "rejected", ("unsupported strategic concept",))
    if obj.entity_type == "concept" and not _concept_is_supported_by_evidence(obj.key, evidence_ids, evidence_by_id):
        return ExtractionDecision(candidate, None, "rejected", ("unsupported strategic concept",))
    if subject.entity_type == obj.entity_type == "concept" and subject.key == obj.key:
        return ExtractionDecision(candidate, None, "rejected", ("unexpected self-loop relation",))
    provenance = candidate.get("provenance_type")
    if provenance not in {"source_claim", "coach_supported_inference"}:
        return ExtractionDecision(candidate, None, "rejected", ("unsupported automated provenance type",))
    concepts = _canonical_concepts(candidate.get("concepts"), subject, obj)
    if concepts is None:
        return ExtractionDecision(candidate, None, "rejected", ("unknown strategic concept",))
    unsupported_concepts = tuple(
        concept for concept in concepts
        if not _concept_is_supported_by_evidence(concept, evidence_ids, evidence_by_id)
    )
    concepts = tuple(concept for concept in concepts if concept not in unsupported_concepts)
    if candidate.get("condition") is not None and not isinstance(candidate.get("condition"), str):
        return ExtractionDecision(candidate, None, "rejected", ("condition must be a string or null",))
    if candidate.get("effect") is not None and not isinstance(candidate.get("effect"), str):
        return ExtractionDecision(candidate, None, "rejected", ("effect must be a string or null",))
    condition = canonical_condition(candidate.get("condition"))
    effect = canonical_condition(candidate.get("effect"))
    if condition and not _condition_is_supported(condition, evidence_ids, evidence_by_id):
        return ExtractionDecision(candidate, None, "rejected", ("condition is not supported by evidence",))
    condition_event, condition_event_error = _compile_condition_event(
        candidate.get("condition_event"), condition, evidence_by_id, evidence_ids, ability_aliases
    )
    if condition_event_error:
        return ExtractionDecision(candidate, None, "rejected", (condition_event_error,))
    alignments, alignment_error = _compile_source_alignments(
        candidate,
        evidence_by_id,
        evidence_ids,
        ability_aliases,
        entity_aliases,
        subject,
        relation_type,
        obj,
        condition,
    )
    if alignment_error:
        return ExtractionDecision(candidate, None, "rejected", (alignment_error,))
    patch = candidate.get("patch_sensitivity", "low")
    if patch not in {"very_low", "low", "medium", "high"}:
        return ExtractionDecision(candidate, None, "rejected", ("unknown patch sensitivity",))
    try:
        extraction_confidence = float(candidate.get("extraction_confidence"))
    except (TypeError, ValueError):
        return ExtractionDecision(candidate, None, "rejected", ("invalid extraction confidence",))
    if not 0.0 <= extraction_confidence <= 1.0:
        return ExtractionDecision(candidate, None, "rejected", ("invalid extraction confidence",))
    source_quality = sum(evidence_by_id[item].evidence_quality for item in evidence_ids) / len(evidence_ids)
    canonicalization = min(subject.certainty, obj.certainty)
    confidence = round(0.55 * extraction_confidence + 0.35 * source_quality + 0.10 * canonicalization, 4)
    refs = tuple(
        EvidenceRef(evidence_by_id[item].source_type, evidence_by_id[item].source_id, item, evidence_by_id[item].text)
        for item in evidence_ids
    )
    evidence_patch = max((evidence_by_id[item].patch_sensitivity for item in evidence_ids), key=_patch_rank)
    patch = max((patch, evidence_patch), key=_patch_rank)
    relation = StrategicRelation(
        id=_stable_relation_id(
            subject.entity_type, subject.key, relation_type, obj.entity_type, obj.key,
            condition, effect,
            json.dumps(asdict(condition_event), sort_keys=True) if condition_event else None,
        ),
        subject_type=subject.entity_type,
        subject_key=subject.key,
        relation_type=relation_type,
        object_type=obj.entity_type,
        object_key=obj.key,
        confidence=confidence,
        provenance_type=provenance,
        evidence_refs=refs,
        condition=condition,
        condition_event=condition_event,
        effect=effect,
        concepts=concepts,
        patch_sensitivity=patch,
        data_version=AUTOMATED_RELATION_DATA_VERSION,
        ontology_version=ontology_version,
        alignments=alignments,
    )
    relation.validate()
    warnings = (
        ("removed unsupported strategic concepts: " + ", ".join(unsupported_concepts),)
        if unsupported_concepts
        else ()
    )
    return ExtractionDecision(candidate, relation, "accepted", warnings, {"extraction": extraction_confidence, "evidence": source_quality, "canonicalization": canonicalization})


def _compile_condition_event(
    raw: Any,
    condition: str | None,
    evidence_by_id: Mapping[str, EvidenceItem],
    evidence_ids: list[str],
    ability_aliases: Mapping[str, str],
) -> tuple[ConditionEvent | None, str | None]:
    if raw is None:
        return None, None
    if condition is None or not isinstance(raw, Mapping):
        return None, "condition event requires a condition object"
    source_text, evidence_id = raw.get("source_text"), raw.get("evidence_id")
    if not isinstance(source_text, str) or evidence_id not in evidence_ids:
        return None, "invalid condition event source"
    evidence = evidence_by_id[evidence_id]
    if not _source_phrase_is_present(source_text, evidence.text):
        return None, "condition event source phrase is not present in evidence"
    event = raw.get("event")
    temporal_operator = raw.get("temporal_operator")
    entity = raw.get("entity")
    canonical_entity_key = None
    if entity is not None:
        if not isinstance(entity, str):
            return None, "invalid condition event entity"
        canonical = canonical_entity("ability", entity, ability_aliases=ability_aliases)
        if canonical is None:
            return None, "unknown condition event entity"
        canonical_entity_key = canonical.key
        if not _source_phrase_is_present(entity, source_text):
            return None, "condition event entity is not present in source phrase"
    derived_state = raw.get("derived_state")
    try:
        event_detail = ConditionEvent(
            source_text=source_text.strip(), evidence_id=evidence_id,
            entity=canonical_entity_key, event=str(event),
            derived_state=str(derived_state) if derived_state is not None else None,
            temporal_operator=str(temporal_operator),
        )
        event_detail.validate()
    except (StrategicValidationError, TypeError, ValueError):
        return None, "invalid condition event"
    if event_detail.event == "missed" and "miss" not in source_text.lower():
        return None, "condition event is not supported by source phrase"
    if event_detail.temporal_operator not in source_text.lower().split():
        return None, "condition event temporal operator is not present in source phrase"
    return event_detail, None


def _compile_source_alignments(
    candidate: Mapping[str, Any],
    evidence_by_id: Mapping[str, EvidenceItem],
    evidence_ids: list[str],
    ability_aliases: Mapping[str, str],
    entity_aliases: Mapping[str, Mapping[str, str]],
    subject: Any,
    relation_type: str,
    obj: Any,
    condition: str | None,
) -> tuple[tuple[RelationAlignment, ...], str | None]:
    """Verify that each canonical field retains a literal evidence anchor."""
    grounding = candidate.get("grounding")
    if not isinstance(grounding, Mapping):
        return (), "missing source grounding"
    targets = {
        "subject": subject.key,
        "predicate": relation_type,
        "object": obj.key,
    }
    if condition is not None:
        targets["condition"] = condition
    alignments: list[RelationAlignment] = []
    for field, target in targets.items():
        raw = grounding.get(field)
        if not isinstance(raw, Mapping):
            return (), f"missing {field} source grounding"
        source_text = raw.get("source_text")
        evidence_id = raw.get("evidence_id")
        if not isinstance(source_text, str) or not source_text.strip() or evidence_id not in evidence_ids:
            return (), f"invalid {field} source grounding"
        evidence = evidence_by_id[evidence_id]
        source_text = source_text.strip()
        if not _source_phrase_is_present(source_text, evidence.text):
            return (), f"{field} source phrase is not present in evidence"
        if field == "predicate" and _source_predicate_is_negated(source_text, evidence.text):
            return (), "negated predicate source phrase"
        if field == "condition":
            if not _condition_source_supports(source_text, target):
                return (), "condition source phrase does not support canonical condition"
            if not _condition_is_supported(target, [evidence_id], evidence_by_id):
                return (), "condition source phrase does not support canonical condition"
            canonical, mapping_type = target, "condition_source_phrase"
        else:
            canonical, mapping_type = _canonical_from_source(
                field, source_text, subject, obj, ability_aliases, entity_aliases
            )
        if canonical != target:
            return (), f"{field} source phrase does not support canonical field"
        alignments.append(
            RelationAlignment(
                field=field,
                source_text=source_text,
                evidence_id=evidence_id,
                canonical_value=target,
                mapping_type=mapping_type,
                mapping_confidence=1.0,
            )
        )
    if condition is None and grounding.get("condition") is not None:
        return (), "unexpected condition source grounding"
    return tuple(alignments), None


def _source_phrase_is_present(source_text: str, evidence_text: str) -> bool:
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(source_text) + r"(?![a-z0-9])", evidence_text, re.IGNORECASE))


def _source_predicate_is_negated(source_text: str, evidence_text: str) -> bool:
    """Reject local negation rather than reversing a relation from a bare verb."""
    pattern = re.compile(r"(?<![a-z0-9])" + re.escape(source_text) + r"(?![a-z0-9])", re.IGNORECASE)
    for match in pattern.finditer(evidence_text):
        before = re.findall(r"[a-z0-9]+", evidence_text[max(0, match.start() - 48):match.start()].lower())
        after = re.findall(r"[a-z0-9]+", evidence_text[match.end():match.end() + 24].lower())
        if any(term in _NEGATION_TERMS for term in before[-4:]):
            return True
        if any(term in {"no", "nothing", "nobody", "none"} for term in after[:2]):
            return True
    return False


def _canonical_from_source(
    field: str,
    source_text: str,
    subject: Any,
    obj: Any,
    ability_aliases: Mapping[str, str],
    entity_aliases: Mapping[str, Mapping[str, str]],
) -> tuple[str | None, str]:
    if field == "subject":
        source = canonical_entity(
            subject.entity_type,
            source_text,
            ability_aliases=ability_aliases,
            entity_aliases=entity_aliases.get(subject.entity_type, {}),
        )
        if source is None:
            return None, "entity_alias"
        return source.key, "exact" if normalized_key(source_text) == normalized_key(source.key) else "entity_alias"
    if field == "predicate":
        canonical = canonical_relation_type(source_text)
        return canonical, "exact" if normalized_key(source_text) == normalized_key(canonical or "") else "relation_alias"
    if field == "condition":
        return canonical_condition(source_text), "condition_exact"
    source = canonical_entity(
        obj.entity_type,
        source_text,
        ability_aliases=ability_aliases,
        entity_aliases=entity_aliases.get(obj.entity_type, {}),
    )
    if source is None:
        return None, "ontology_abstraction"
    mapping_type = "exact" if normalized_key(source_text) == normalized_key(source.key) else "ontology_abstraction"
    return source.key, mapping_type


def _canonical_concepts(raw: Any, subject: Any, obj: Any) -> tuple[str, ...] | None:
    from core.relation_normalization import canonical_concept

    if not isinstance(raw, list):
        return None
    values = [canonical_concept(value) for value in raw if isinstance(value, str)]
    if len(values) != len(raw) or any(value is None for value in values):
        return None
    if subject.entity_type == "concept":
        values.append(subject.key)
    if obj.entity_type == "concept":
        values.append(obj.key)
    return tuple(dict.fromkeys(values))


def _concept_is_supported_by_evidence(
    concept: str,
    evidence_ids: list[str],
    evidence_by_id: Mapping[str, EvidenceItem],
) -> bool:
    return any(concept_is_mentioned(evidence_by_id[evidence_id].text, concept) for evidence_id in evidence_ids)


_CONDITION_STOPWORDS = frozenset({"a", "an", "and", "are", "as", "at", "before", "after", "during", "for", "if", "in", "is", "of", "on", "or", "the", "then", "to", "when", "while", "with", "you", "your"})
_NEGATION_TERMS = frozenset({"not", "never", "nothing", "cannot", "cant", "without"})


def _condition_is_supported(
    condition: str, evidence_ids: list[str], evidence_by_id: Mapping[str, EvidenceItem],
) -> bool:
    """Reject qualifiers containing facts absent from the cited source evidence."""
    terms = _condition_terms(condition)
    if not terms:
        return False
    for item in evidence_ids:
        source_terms = _condition_terms(evidence_by_id[item].text)
        if _is_subsequence(terms, source_terms):
            source_has_negation = any(term in _NEGATION_TERMS for term in source_terms)
            condition_has_negation = any(term in _NEGATION_TERMS for term in terms)
            if source_has_negation != condition_has_negation:
                continue
            return True
    return False


def _condition_source_supports(source_phrase: str, canonical_condition: str) -> bool:
    """Ensure a condition alignment names a qualifier, not arbitrary nearby text."""
    source_terms = _condition_terms(source_phrase)
    condition_terms = _condition_terms(canonical_condition)
    if not source_terms or not _is_subsequence(source_terms, condition_terms):
        return False
    return any(term in _NEGATION_TERMS for term in source_terms) == any(
        term in _NEGATION_TERMS for term in condition_terms
    )


def _condition_terms(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9]+", text.lower()) if term not in _CONDITION_STOPWORDS]


def _is_subsequence(needles: list[str], haystack: list[str]) -> bool:
    position = 0
    for term in haystack:
        if position < len(needles) and term == needles[position]:
            position += 1
    return position == len(needles)


def _stable_relation_id(*parts: str | None) -> str:
    canonical = "|".join(part or "" for part in parts)
    return "auto-rel-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _patch_rank(value: str) -> int:
    return {"very_low": 0, "low": 1, "medium": 2, "high": 3}[value]


def _apply_threshold(decision: ExtractionDecision, threshold: float) -> ExtractionDecision:
    if decision.relation is None or decision.relation.confidence >= threshold:
        return decision
    return ExtractionDecision(
        decision.raw,
        decision.relation,
        "review",
        decision.warnings + (f"confidence below threshold {threshold:.2f}",),
        decision.confidence_components,
    )


def _flag_structural_contradictions(decisions: tuple[ExtractionDecision, ...]) -> tuple[ExtractionDecision, ...]:
    """Quarantine same-condition opposite edges instead of hiding either claim."""
    conflicts: set[int] = set()
    accepted = [(index, decision.relation) for index, decision in enumerate(decisions) if decision.relation]
    for index, relation in accepted:
        for other_index, other in accepted:
            if index >= other_index or not _same_relation_target(relation, other):
                continue
            if relation_types_conflict(relation.relation_type, other.relation_type):
                conflicts.update((index, other_index))
    return tuple(
        replace(
            decision,
            status="review",
            warnings=decision.warnings + ("potential same-condition contradictory relation",),
        )
        if index in conflicts and decision.status == "accepted"
        else decision
        for index, decision in enumerate(decisions)
    )


def _same_relation_target(left: StrategicRelation, right: StrategicRelation) -> bool:
    return (
        left.subject_type == right.subject_type
        and left.subject_key == right.subject_key
        and left.object_type == right.object_type
        and left.object_key == right.object_key
        and left.condition == right.condition
    )
