"""Deterministic Phase 2D candidate generation before constrained mapping."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from core.champions import canonical_champion_name, champions_mentioned
from core.ontology import RELATION_TYPES, STRATEGIC_CONCEPTS
from core.relation_normalization import (
    RELATION_SOURCE_ALIASES,
    SEMANTIC_CONCEPT_ALIASES,
    canonical_entity,
    canonical_relation_type,
    normalized_key,
)
from pipeline.relation_extract import GroundedProposition


@dataclass(frozen=True)
class CanonicalCandidate:
    id: str
    score: float
    reason: str
    entity_type: str | None = None


@dataclass(frozen=True)
class ConditionCandidate:
    source_text: str
    event: str | None = None
    derived_state: str | None = None


@dataclass(frozen=True)
class CandidateSet:
    proposition_signature: tuple[str, str, str, str | None, tuple[str, ...]]
    ability_aliases: tuple[tuple[str, str], ...]
    top_k_concepts: int
    subject: tuple[CanonicalCandidate, ...]
    relation: tuple[CanonicalCandidate, ...]
    object: tuple[CanonicalCandidate, ...]
    condition: tuple[ConditionCandidate, ...]


def generate_candidates(
    proposition: GroundedProposition,
    *,
    ability_aliases: Mapping[str, str] | None = None,
    top_k_concepts: int = 5,
) -> CandidateSet:
    """Generate legal canonical choices without mapping or persistence."""
    if top_k_concepts <= 0:
        raise ValueError("top_k_concepts must be positive")
    aliases = tuple(sorted((str(key), str(value)) for key, value in (ability_aliases or {}).items()))
    return CandidateSet(
        proposition_signature=_proposition_signature(proposition),
        ability_aliases=aliases,
        top_k_concepts=top_k_concepts,
        subject=_entity_candidates(proposition.subject_source, dict(aliases)),
        relation=_relation_candidates(proposition.predicate_source),
        object=_concept_candidates(proposition.effect_source, top_k_concepts),
        condition=_condition_candidates(proposition.condition_source, dict(aliases)),
    )


def _entity_candidates(text: str, ability_aliases: Mapping[str, str]) -> tuple[CanonicalCandidate, ...]:
    results: dict[str, CanonicalCandidate] = {}
    ability = canonical_entity("ability", text, ability_aliases=ability_aliases)
    if ability:
        results[f"ability:{ability.key}"] = CanonicalCandidate(f"ability:{ability.key}", 1.0, "ability_alias", "ability")
    for alias, canonical in ability_aliases.items():
        if _alias_in_phrase(alias, text):
            results.setdefault(f"ability:{canonical}", CanonicalCandidate(f"ability:{canonical}", .95, "ability_alias_in_phrase", "ability"))
    for champion in champions_mentioned(text):
        canonical = canonical_champion_name(champion)
        if canonical:
            results[f"champion:{canonical}"] = CanonicalCandidate(f"champion:{canonical}", 1.0, "champion_mention", "champion")
    return tuple(sorted(results.values(), key=lambda item: (-item.score, item.id)))


def _relation_candidates(text: str) -> tuple[CanonicalCandidate, ...]:
    value = canonical_relation_type(text)
    results: dict[str, CanonicalCandidate] = {}
    if value:
        results[value] = CanonicalCandidate(value, 1.0, "relation_alias")
    normalized = normalized_key(text)
    # Keep only nearby legal alternatives, never create a verb outside v0.
    if _has_positive_cue(normalized, ("prevent", "stop", "deny", "block")):
        for relation, score in (("denies", .9), ("reduces", .5), ("increases_cost_of", .35)):
            results.setdefault(relation, CanonicalCandidate(relation, score, "negative_effect_cue"))
    if _has_positive_cue(normalized, ("enable",)):
        for relation, score in (("enables", .9), ("creates", .5), ("expands", .35)):
            results.setdefault(relation, CanonicalCandidate(relation, score, "enabling_effect_cue"))
    return tuple(sorted((item for item in results.values() if item.id in RELATION_TYPES), key=lambda item: (-item.score, item.id)))


def _concept_candidates(text: str, top_k: int) -> tuple[CanonicalCandidate, ...]:
    normalized = normalized_key(text)
    results: dict[str, CanonicalCandidate] = {}
    for alias, concept in SEMANTIC_CONCEPT_ALIASES.items():
        if normalized_key(alias) in normalized:
            results[concept] = CanonicalCandidate(concept, 1.0, f"semantic_alias:{alias}", "concept")
    tokens = set(_tokens(text))
    for key, concept in STRATEGIC_CONCEPTS.items():
        concept_tokens = set(_tokens(key.replace("_", " ") + " " + concept.description))
        overlap = tokens & concept_tokens
        if overlap:
            score = round(len(overlap) / max(1, len(tokens)), 4)
            previous = results.get(key)
            if previous is None or score > previous.score:
                results[key] = CanonicalCandidate(key, score, "ontology_description_overlap", "concept")
    return tuple(sorted(results.values(), key=lambda item: (-item.score, item.id))[:top_k])


def _condition_candidates(text: str | None, ability_aliases: Mapping[str, str]) -> tuple[ConditionCandidate, ...]:
    if not text:
        return ()
    lowered = normalized_key(text)
    if "miss" in lowered:
        abilities = {ability for alias, ability in ability_aliases.items() if _alias_in_phrase(alias, text)}
        if len(abilities) == 1:
            return (ConditionCandidate(text, "missed", "temporarily_unavailable"),)
    return (ConditionCandidate(text),)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in re.findall(r"[a-z0-9']+", text.lower()) if len(token) > 2)


def _proposition_signature(proposition: GroundedProposition) -> tuple[str, str, str, str | None, tuple[str, ...]]:
    return (
        proposition.subject_source,
        proposition.predicate_source,
        proposition.effect_source,
        proposition.condition_source,
        proposition.evidence_ids,
    )


def _alias_in_phrase(alias: str, text: str) -> bool:
    """Match an alias as complete normalized tokens, never a substring."""
    alias_tokens = normalized_key(alias).split()
    text_tokens = normalized_key(text).split()
    width = len(alias_tokens)
    return bool(width and any(text_tokens[index:index + width] == alias_tokens for index in range(len(text_tokens) - width + 1)))


def _has_positive_cue(text: str, cues: tuple[str, ...]) -> bool:
    tokens = normalized_key(text).split()
    if _contains_negation(tokens):
        return False
    for index, token in enumerate(tokens):
        if not any(token.startswith(cue) for cue in cues):
            continue
        prefix = tokens[max(0, index - 2):index]
        if any(value in {"not", "no", "never", "nothing"} for value in prefix):
            continue
        return True
    return False


def _contains_negation(tokens: list[str]) -> bool:
    """Suppress directional candidates when predicate polarity is negative."""
    negative = {"not", "no", "never", "nothing", "cannot", "cant", "doesn", "didn", "don", "isn", "wasn", "weren", "hasn", "hadn", "won", "wouldn", "couldn", "shouldn", "unable", "fail", "fails", "failed", "failing", "without"}
    return any(token in negative for token in tokens)
