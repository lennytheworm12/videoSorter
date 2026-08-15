"""Domain types for derived strategic knowledge.

These objects represent derived knowledge, not raw coaching evidence. Every
stored relation or fingerprint must retain evidence references so callers can
keep source-grounded evidence separate from strategic inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any

from core.ontology import (
    ENTITY_TYPES,
    ONTOLOGY_VERSION,
    PATCH_SENSITIVITY,
    PROVENANCE_TYPES,
    RELATION_TYPES,
    STRATEGIC_CONCEPTS,
)


CURRENT_STRATEGIC_DATA_VERSION = "strategic-fixtures-v0"


class StrategicValidationError(ValueError):
    """Raised when strategic fixture or domain data violates invariants."""


@dataclass(frozen=True)
class EvidenceRef:
    source_type: str
    source_id: str
    insight_id: str | None = None
    quote: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceRef":
        ref = cls(
            source_type=str(data.get("source_type") or "").strip(),
            source_id=str(data.get("source_id") or "").strip(),
            insight_id=(
                str(data["insight_id"]).strip()
                if data.get("insight_id") is not None
                else None
            ),
            quote=str(data["quote"]).strip() if data.get("quote") else None,
        )
        ref.validate()
        return ref

    def stable_key(self) -> tuple[str, str, str | None]:
        return (self.source_type, self.source_id, self.insight_id)

    def validate(self) -> None:
        if not self.source_type:
            raise StrategicValidationError("evidence source_type is required")
        if not self.source_id:
            raise StrategicValidationError("evidence source_id is required")


@dataclass(frozen=True)
class StrategicRelation:
    id: str
    subject_type: str
    subject_key: str
    relation_type: str
    object_type: str
    object_key: str
    confidence: float
    provenance_type: str
    evidence_refs: tuple[EvidenceRef, ...] = field(default_factory=tuple)
    condition: str | None = None
    effect: str | None = None
    concepts: tuple[str, ...] = field(default_factory=tuple)
    patch_sensitivity: str = "very_low"
    data_version: str = CURRENT_STRATEGIC_DATA_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategicRelation":
        relation = cls(
            id=str(data.get("id") or "").strip(),
            subject_type=str(data.get("subject_type") or "").strip(),
            subject_key=str(data.get("subject_key") or "").strip(),
            relation_type=str(data.get("relation_type") or "").strip(),
            object_type=str(data.get("object_type") or "").strip(),
            object_key=str(data.get("object_key") or "").strip(),
            confidence=_parse_confidence(data.get("confidence", 0.0), "relation"),
            provenance_type=str(data.get("provenance_type") or "").strip(),
            evidence_refs=tuple(
                EvidenceRef.from_dict(ref) for ref in data.get("evidence_refs", [])
            ),
            condition=str(data["condition"]).strip() if data.get("condition") else None,
            effect=str(data["effect"]).strip() if data.get("effect") else None,
            concepts=tuple(str(c).strip() for c in data.get("concepts", [])),
            patch_sensitivity=str(data.get("patch_sensitivity") or "very_low").strip(),
            data_version=str(
                data.get("data_version") or CURRENT_STRATEGIC_DATA_VERSION
            ).strip(),
        )
        relation.validate()
        return relation

    def stable_key(self) -> tuple[str, str, str, str, str, str | None, str | None]:
        return (
            self.subject_type,
            _normalize_key(self.subject_key),
            self.relation_type,
            self.object_type,
            _normalize_key(self.object_key),
            self.condition,
            self.effect,
        )

    def validate(self) -> None:
        if not self.id:
            raise StrategicValidationError("relation id is required")
        if self.subject_type not in ENTITY_TYPES:
            raise StrategicValidationError(f"unknown subject_type: {self.subject_type}")
        if self.object_type not in ENTITY_TYPES:
            raise StrategicValidationError(f"unknown object_type: {self.object_type}")
        if not self.subject_key:
            raise StrategicValidationError("relation subject_key is required")
        if not self.object_key:
            raise StrategicValidationError("relation object_key is required")
        if self.relation_type not in RELATION_TYPES:
            raise StrategicValidationError(f"unknown relation_type: {self.relation_type}")
        if self.provenance_type not in PROVENANCE_TYPES:
            raise StrategicValidationError(
                f"unknown provenance_type: {self.provenance_type}"
            )
        if self.patch_sensitivity not in PATCH_SENSITIVITY:
            raise StrategicValidationError(
                f"unknown patch_sensitivity: {self.patch_sensitivity}"
            )
        if self.data_version != CURRENT_STRATEGIC_DATA_VERSION:
            raise StrategicValidationError(
                f"unsupported strategic data version: {self.data_version}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise StrategicValidationError("relation confidence must be between 0 and 1")
        if (
            self.provenance_type != "speculative_hypothesis"
            and not self.evidence_refs
        ):
            raise StrategicValidationError(
                f"relation {self.id} must include evidence_refs"
            )
        seen_refs: set[tuple[str, str, str | None]] = set()
        for ref in self.evidence_refs:
            ref.validate()
            key = ref.stable_key()
            if key in seen_refs:
                raise StrategicValidationError(
                    f"relation {self.id} has duplicate evidence ref {key}"
                )
            seen_refs.add(key)
        for concept in self.concepts:
            if concept not in STRATEGIC_CONCEPTS:
                raise StrategicValidationError(f"unknown concept: {concept}")
        if self.subject_type == "concept" and self.subject_key not in STRATEGIC_CONCEPTS:
            raise StrategicValidationError(f"unknown subject concept: {self.subject_key}")
        if self.object_type == "concept" and self.object_key not in STRATEGIC_CONCEPTS:
            raise StrategicValidationError(f"unknown object concept: {self.object_key}")


@dataclass(frozen=True)
class ChampionFingerprint:
    champion: str
    preferred_states: tuple[str, ...] = field(default_factory=tuple)
    avoided_states: tuple[str, ...] = field(default_factory=tuple)
    persistent_advantages: tuple[str, ...] = field(default_factory=tuple)
    conditional_advantages: tuple[str, ...] = field(default_factory=tuple)
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    access_tools: tuple[str, ...] = field(default_factory=tuple)
    access_denial_tools: tuple[str, ...] = field(default_factory=tuple)
    continuity_requirements: tuple[str, ...] = field(default_factory=tuple)
    conversion_patterns: tuple[str, ...] = field(default_factory=tuple)
    role_flip_events: tuple[str, ...] = field(default_factory=tuple)
    failure_modes: tuple[str, ...] = field(default_factory=tuple)
    evidence_refs: tuple[EvidenceRef, ...] = field(default_factory=tuple)
    confidence: float = 0.0
    provenance_type: str = "manual_fixture"
    patch_sensitivity: str = "low"
    data_version: str = CURRENT_STRATEGIC_DATA_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChampionFingerprint":
        fingerprint = cls(
            champion=str(data.get("champion") or "").strip(),
            preferred_states=_tuple_field(data, "preferred_states"),
            avoided_states=_tuple_field(data, "avoided_states"),
            persistent_advantages=_tuple_field(data, "persistent_advantages"),
            conditional_advantages=_tuple_field(data, "conditional_advantages"),
            dependencies=_tuple_field(data, "dependencies"),
            access_tools=_tuple_field(data, "access_tools"),
            access_denial_tools=_tuple_field(data, "access_denial_tools"),
            continuity_requirements=_tuple_field(data, "continuity_requirements"),
            conversion_patterns=_tuple_field(data, "conversion_patterns"),
            role_flip_events=_tuple_field(data, "role_flip_events"),
            failure_modes=_tuple_field(data, "failure_modes"),
            evidence_refs=tuple(
                EvidenceRef.from_dict(ref) for ref in data.get("evidence_refs", [])
            ),
            confidence=_parse_confidence(data.get("confidence", 0.0), "fingerprint"),
            provenance_type=str(data.get("provenance_type") or "manual_fixture").strip(),
            patch_sensitivity=str(data.get("patch_sensitivity") or "low").strip(),
            data_version=str(
                data.get("data_version") or CURRENT_STRATEGIC_DATA_VERSION
            ).strip(),
        )
        fingerprint.validate()
        return fingerprint

    def validate(self) -> None:
        if not self.champion:
            raise StrategicValidationError("fingerprint champion is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise StrategicValidationError("fingerprint confidence must be between 0 and 1")
        if self.provenance_type not in PROVENANCE_TYPES:
            raise StrategicValidationError(
                f"unknown fingerprint provenance_type: {self.provenance_type}"
            )
        if self.patch_sensitivity not in PATCH_SENSITIVITY:
            raise StrategicValidationError(
                f"unknown fingerprint patch_sensitivity: {self.patch_sensitivity}"
            )
        if self.data_version != CURRENT_STRATEGIC_DATA_VERSION:
            raise StrategicValidationError(
                f"unsupported strategic data version: {self.data_version}"
            )
        for concept in self.dependencies:
            if concept not in STRATEGIC_CONCEPTS:
                raise StrategicValidationError(
                    f"unknown fingerprint dependency concept: {concept}"
                )
        if not self.evidence_refs:
            raise StrategicValidationError(
                f"fingerprint {self.champion} must include evidence_refs"
            )
        seen_refs: set[tuple[str, str, str | None]] = set()
        for ref in self.evidence_refs:
            ref.validate()
            key = ref.stable_key()
            if key in seen_refs:
                raise StrategicValidationError(
                    f"fingerprint {self.champion} has duplicate evidence ref {key}"
                )
            seen_refs.add(key)


@dataclass(frozen=True)
class CompiledPrinciple:
    id: str
    title: str
    summary: str
    concepts: tuple[str, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    confidence: float
    provenance_type: str = "manual_fixture"
    scope: str = "global"
    patch_sensitivity: str = "very_low"
    data_version: str = CURRENT_STRATEGIC_DATA_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompiledPrinciple":
        principle = cls(
            id=str(data.get("id") or "").strip(),
            title=str(data.get("title") or "").strip(),
            summary=str(data.get("summary") or "").strip(),
            concepts=tuple(str(c).strip() for c in data.get("concepts", [])),
            evidence_refs=tuple(
                EvidenceRef.from_dict(ref) for ref in data.get("evidence_refs", [])
            ),
            confidence=_parse_confidence(data.get("confidence", 0.0), "principle"),
            provenance_type=str(data.get("provenance_type") or "manual_fixture").strip(),
            scope=str(data.get("scope") or "global").strip(),
            patch_sensitivity=str(data.get("patch_sensitivity") or "very_low").strip(),
            data_version=str(
                data.get("data_version") or CURRENT_STRATEGIC_DATA_VERSION
            ).strip(),
        )
        principle.validate()
        return principle

    def validate(self) -> None:
        if not self.id:
            raise StrategicValidationError("principle id is required")
        if not self.title:
            raise StrategicValidationError("principle title is required")
        if not self.summary:
            raise StrategicValidationError("principle summary is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise StrategicValidationError("principle confidence must be between 0 and 1")
        if self.provenance_type not in PROVENANCE_TYPES:
            raise StrategicValidationError(
                f"unknown principle provenance_type: {self.provenance_type}"
            )
        if self.patch_sensitivity not in PATCH_SENSITIVITY:
            raise StrategicValidationError(
                f"unknown principle patch_sensitivity: {self.patch_sensitivity}"
            )
        if self.data_version != CURRENT_STRATEGIC_DATA_VERSION:
            raise StrategicValidationError(
                f"unsupported strategic data version: {self.data_version}"
            )
        if not self.evidence_refs:
            raise StrategicValidationError(f"principle {self.id} must include evidence_refs")
        seen_refs: set[tuple[str, str, str | None]] = set()
        for ref in self.evidence_refs:
            ref.validate()
            key = ref.stable_key()
            if key in seen_refs:
                raise StrategicValidationError(
                    f"principle {self.id} has duplicate evidence ref {key}"
                )
            seen_refs.add(key)
        for concept in self.concepts:
            if concept not in STRATEGIC_CONCEPTS:
                raise StrategicValidationError(f"unknown concept: {concept}")


@dataclass(frozen=True)
class StrategicFixture:
    ontology_version: str
    data_version: str
    fingerprints: tuple[ChampionFingerprint, ...]
    relations: tuple[StrategicRelation, ...]
    principles: tuple[CompiledPrinciple, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategicFixture":
        fixture = cls(
            ontology_version=str(data.get("ontology_version") or "").strip(),
            data_version=str(data.get("data_version") or "").strip(),
            fingerprints=tuple(
                ChampionFingerprint.from_dict(item)
                for item in data.get("fingerprints", [])
            ),
            relations=dedupe_relations(
                StrategicRelation.from_dict(item)
                for item in data.get("relations", [])
            ),
            principles=tuple(
                CompiledPrinciple.from_dict(item) for item in data.get("principles", [])
            ),
        )
        fixture.validate()
        return fixture

    def validate(self) -> None:
        if self.ontology_version != ONTOLOGY_VERSION:
            raise StrategicValidationError(
                f"unsupported ontology version: {self.ontology_version}"
            )
        if self.data_version != CURRENT_STRATEGIC_DATA_VERSION:
            raise StrategicValidationError(
                f"unsupported strategic data version: {self.data_version}"
            )
        for fingerprint in self.fingerprints:
            fingerprint.validate()
        for relation in self.relations:
            relation.validate()
        for principle in self.principles:
            principle.validate()
        champion_names = [fp.champion.lower() for fp in self.fingerprints]
        if len(champion_names) != len(set(champion_names)):
            raise StrategicValidationError("duplicate champion fingerprint")
        relation_ids = [relation.id for relation in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise StrategicValidationError("duplicate relation id")
        principle_ids = [principle.id for principle in self.principles]
        if len(principle_ids) != len(set(principle_ids)):
            raise StrategicValidationError("duplicate principle id")


def dedupe_relations(relations: Any) -> tuple[StrategicRelation, ...]:
    deduped: list[StrategicRelation] = []
    indexes: dict[tuple[str, str, str, str, str, str | None, str | None], int] = {}
    for relation in relations:
        key = relation.stable_key()
        if key in indexes:
            existing_index = indexes[key]
            existing = deduped[existing_index]
            refs = list(existing.evidence_refs)
            seen_refs = {ref.stable_key() for ref in refs}
            for ref in relation.evidence_refs:
                if ref.stable_key() not in seen_refs:
                    refs.append(ref)
                    seen_refs.add(ref.stable_key())
            deduped[existing_index] = replace(
                existing,
                confidence=max(existing.confidence, relation.confidence),
                evidence_refs=tuple(refs),
            )
            continue
        indexes[key] = len(deduped)
        deduped.append(relation)
    return tuple(deduped)


def load_strategic_fixture(path: str | Path) -> StrategicFixture:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return StrategicFixture.from_dict(data)


def _tuple_field(data: dict[str, Any], key: str) -> tuple[str, ...]:
    values = data.get(key, [])
    if values is None:
        return ()
    if not isinstance(values, list):
        raise StrategicValidationError(f"{key} must be a list")
    cleaned = tuple(str(value).strip() for value in values if str(value).strip())
    if len(cleaned) != len(set(cleaned)):
        raise StrategicValidationError(f"{key} contains duplicate values")
    return cleaned


def _normalize_key(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _parse_confidence(value: Any, owner: str) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise StrategicValidationError(f"{owner} confidence must be numeric") from exc
    if not math.isfinite(confidence):
        raise StrategicValidationError(f"{owner} confidence must be finite")
    return confidence
