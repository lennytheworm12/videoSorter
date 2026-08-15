"""Deterministic canonicalization for Phase 2 strategic relation extraction.

This module owns aliases at the boundary between untrusted model output and the
Phase 1 strategic ontology. It deliberately does not expand the ontology.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from core.champions import canonical_champion_name
from core.ontology import ENTITY_TYPES, RELATION_TYPES, STRATEGIC_CONCEPTS


_CONCEPT_ALIASES = {
    "continued contact": "continuity",
    "staying attached": "continuity",
    "continued attachment": "continuity",
    "forward access": "access",
    "enemy access": "access",
    "space": "territory",
    "forward space": "territory",
    "cooldown pressure": "intermittent_pressure",
    "persistent low cost pressure": "persistent_pressure",
}

_RELATION_ALIASES = {
    "create": "creates",
    "provide": "creates",
    "provides": "creates",
    "break": "denies",
    "breaks": "denies",
    "increase cost of": "increases_cost_of",
    "increases cost of": "increases_cost_of",
    "reduce cost of": "reduces_cost_of",
    "reduces cost of": "reduces_cost_of",
}


def normalized_key(value: str) -> str:
    """Normalize a lookup key without changing the user-visible canonical form."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def canonical_concept(value: str) -> str | None:
    """Resolve a model-supplied concept only to ontology v0 vocabulary."""
    if not isinstance(value, str):
        return None
    raw = value.strip().lower().replace("-", "_").replace(" ", "_")
    if raw in STRATEGIC_CONCEPTS:
        return raw
    alias = _CONCEPT_ALIASES.get(normalized_key(value))
    return alias if alias in STRATEGIC_CONCEPTS else None


def concept_is_mentioned(text: str, concept: str) -> bool:
    """Check source text against a canonical concept and its deterministic aliases."""
    phrases = [concept.replace("_", " ")]
    phrases.extend(alias for alias, canonical in _CONCEPT_ALIASES.items() if canonical == concept)
    return any(
        re.search(r"\b" + re.escape(phrase) + r"\w*\b", text, re.IGNORECASE)
        for phrase in phrases
    )


def canonical_relation_type(value: str) -> str | None:
    """Resolve a relation verb only to the existing constrained vocabulary."""
    if not isinstance(value, str):
        return None
    raw = value.strip().lower().replace("-", "_").replace(" ", "_")
    if raw in RELATION_TYPES:
        return raw
    alias = _RELATION_ALIASES.get(normalized_key(value))
    return alias if alias in RELATION_TYPES else None


def canonical_entity_type(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("strategic_concept", "concept")
    return normalized if normalized in ENTITY_TYPES else None


@dataclass(frozen=True)
class CanonicalEntity:
    entity_type: str
    key: str
    certainty: float


def canonical_entity(
    entity_type: str,
    value: str,
    *,
    ability_aliases: Mapping[str, str] | None = None,
    entity_aliases: Mapping[str, str] | None = None,
) -> CanonicalEntity | None:
    """Canonicalize an entity without manufacturing unrecognized graph nodes."""
    resolved_type = canonical_entity_type(entity_type)
    if resolved_type is None or not isinstance(value, str) or not value.strip():
        return None
    if resolved_type == "concept":
        concept = canonical_concept(value)
        return CanonicalEntity("concept", concept, 1.0) if concept else None
    if resolved_type == "champion":
        champion = canonical_champion_name(value)
        return CanonicalEntity("champion", champion, 1.0) if champion else None
    if resolved_type == "ability":
        aliases = {normalized_key(k): v for k, v in (ability_aliases or {}).items()}
        canonical = aliases.get(normalized_key(value))
        if canonical is None:
            canonical = aliases.get(normalized_key(re.sub(r"\s*\([^)]*\)", "", value)))
        if canonical:
            return CanonicalEntity("ability", canonical, 1.0)
        return None
    aliases = {normalized_key(k): v for k, v in (entity_aliases or {}).items()}
    canonical = aliases.get(normalized_key(value))
    return CanonicalEntity(resolved_type, canonical, 1.0) if canonical else None


def canonical_condition(value: str | None) -> str | None:
    """Keep qualifiers as text while normalizing whitespace for dedupe identity."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.strip().split())
    return cleaned or None
