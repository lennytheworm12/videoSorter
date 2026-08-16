"""Optional local retrieval of derived strategic context for Phase 1."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from core.db_paths import all_content_db_paths
from core.ontology import ONTOLOGY_VERSION, STRATEGIC_CONCEPTS
from core.strategic_types import AUTOMATED_RELATION_DATA_VERSION, CURRENT_STRATEGIC_DATA_VERSION


@dataclass(frozen=True)
class StrategicContext:
    fingerprints: tuple[dict, ...] = ()
    relations: tuple[dict, ...] = ()
    principles: tuple[dict, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.fingerprints or self.relations or self.principles)


def format_strategic_context(context: StrategicContext) -> str:
    """Render derived knowledge separately from retrieved source evidence."""
    if context.is_empty:
        return ""

    blocks: list[str] = []
    if context.fingerprints:
        lines = ["Fingerprints:"]
        for fingerprint in context.fingerprints:
            details = []
            for field, label in (
                ("persistent_advantages", "persistent advantages"),
                ("conditional_advantages", "conditional advantages"),
                ("dependencies", "dependencies"),
                ("access_tools", "access tools"),
                ("continuity_requirements", "continuity requirements"),
                ("failure_modes", "failure modes"),
            ):
                values = fingerprint.get(field) or ()
                if values:
                    details.append(f"{label}: {', '.join(values)}")
            lines.append(
                f"- {fingerprint['champion']} (confidence {fingerprint['confidence']:.2f}; "
                f"evidence: {_format_evidence_refs(fingerprint['evidence_refs'])}): "
                + "; ".join(details)
            )
        blocks.append("\n".join(lines))

    if context.relations:
        lines = ["Causal relations:"]
        for relation in context.relations:
            condition = relation.get("condition") or "always"
            event = relation.get("condition_event")
            effect = relation.get("effect") or "no explicit effect recorded"
            event_detail = (
                f"; source event: {event['source_text']} -> {event['derived_state'] or event['event']}"
                if event
                else ""
            )
            lines.append(
                f"- {relation['subject_key']} {relation['relation_type']} {relation['object_key']} "
                f"when {condition}{event_detail}; effect: {effect} "
                f"(concepts: {', '.join(relation['concepts'])}; confidence {relation['confidence']:.2f}; "
                f"evidence: {_format_evidence_refs(relation['evidence_refs'])})"
            )
        blocks.append("\n".join(lines))

    if context.principles:
        lines = ["Compiled principles:"]
        for principle in context.principles:
            lines.append(
                f"- {principle['title']}: {principle['summary']} "
                f"(concepts: {', '.join(principle['concepts'])}; confidence {principle['confidence']:.2f}; "
                f"evidence: {_format_evidence_refs(principle['evidence_refs'])})"
            )
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def build_strategic_context(
    question: str,
    entities: tuple[str, ...] = (),
    *,
    db_paths: tuple[str, ...] | None = None,
    min_confidence: float = 0.7,
    max_hops: int = 1,
    max_relations: int = 12,
) -> StrategicContext:
    """Return a small current-version neighborhood, or empty context on no data."""
    if not isinstance(question, str) or not isinstance(entities, tuple):
        raise ValueError("question must be text and entities must be a tuple")
    if any(not isinstance(entity, str) for entity in entities):
        raise ValueError("entities must contain only text")
    if (
        isinstance(min_confidence, bool)
        or not isinstance(min_confidence, (int, float))
        or not 0 <= min_confidence <= 1
        or isinstance(max_hops, bool)
        or not isinstance(max_hops, int)
        or max_hops < 0
        or isinstance(max_relations, bool)
        or not isinstance(max_relations, int)
        or max_relations < 0
    ):
        raise ValueError("invalid strategic retrieval bounds")
    paths = tuple(all_content_db_paths()) if db_paths is None else db_paths
    fingerprints: list[dict] = []
    relations: list[dict] = []
    principles: list[dict] = []
    seen_fingerprints: set[str] = set()
    seen_relations: set[str] = set()
    seen_principles: set[str] = set()
    normalized_entities = {_normalize(value) for value in entities if value.strip()}
    for path in paths:
        if not Path(path).exists():
            continue
        try:
            local_seen_fingerprints = set(seen_fingerprints)
            local_seen_relations = set(seen_relations)
            local_seen_principles = set(seen_principles)
            local_fingerprints: list[dict] = []
            local_relations: list[dict] = []
            local_principles: list[dict] = []
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                conn.row_factory = sqlite3.Row
                if not _has_tables(conn):
                    continue
                local_entities = normalized_entities | _question_fingerprint_entities(
                    conn,
                    question,
                    min_confidence,
                )
                local_fingerprints = _fingerprints(conn, local_entities, local_seen_fingerprints, min_confidence)
                local_relations = _relations(
                    conn, question, local_entities, local_seen_relations,
                    min_confidence, max_hops, max_relations - len(relations),
                )
                concepts = {concept for relation in local_relations for concept in relation["concepts"]}
                local_principles = _principles(conn, concepts, local_seen_principles, min_confidence)
            fingerprints.extend(local_fingerprints)
            relations.extend(local_relations)
            principles.extend(local_principles)
            seen_fingerprints, seen_relations, seen_principles = local_seen_fingerprints, local_seen_relations, local_seen_principles
        except (sqlite3.Error, json.JSONDecodeError, TypeError):
            continue
    return StrategicContext(tuple(fingerprints), tuple(relations), tuple(principles))


def _has_tables(conn: sqlite3.Connection) -> bool:
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return {"strategic_relations", "strategic_relation_evidence", "champion_fingerprints", "champion_fingerprint_evidence", "compiled_principles", "compiled_principle_evidence"}.issubset(names)


def _fingerprints(conn, entities, seen, minimum):
    rows = conn.execute("SELECT * FROM champion_fingerprints WHERE data_version=? AND confidence>=? ORDER BY confidence DESC, champion", (CURRENT_STRATEGIC_DATA_VERSION, minimum))
    results = []
    for row in rows:
        if _normalize(row["champion"]) not in entities or _normalize(row["champion"]) in seen:
            continue
        result = _decode_json_fields(dict(row), ("preferred_states", "avoided_states", "persistent_advantages", "conditional_advantages", "dependencies", "access_tools", "access_denial_tools", "continuity_requirements", "conversion_patterns", "role_flip_events", "failure_modes"))
        result["evidence_refs"] = _evidence(conn, "champion_fingerprint_evidence", (row["champion"], row["data_version"]))
        if result["evidence_refs"]:
            seen.add(_normalize(row["champion"]))
            results.append(result)
    return results


def _question_fingerprint_entities(conn, question, minimum):
    """Find fixture-backed champions named in the question without RAG coverage."""
    normalized_question = _normalize(question)
    entities = set()
    for row in conn.execute(
        "SELECT champion FROM champion_fingerprints WHERE data_version=? AND confidence>=?",
        (CURRENT_STRATEGIC_DATA_VERSION, minimum),
    ):
        champion = _normalize(row["champion"])
        if re.search(r"\b" + re.escape(champion) + r"\b", normalized_question):
            entities.add(champion)
    return entities


def _relations(conn, question, entities, seen, minimum, max_hops, limit):
    rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT * FROM strategic_relations
            WHERE data_version IN (?, ?) AND ontology_version = ? AND confidence >= ?
            ORDER BY confidence DESC, id
            """,
            (CURRENT_STRATEGIC_DATA_VERSION, AUTOMATED_RELATION_DATA_VERSION, ONTOLOGY_VERSION, minimum),
        )
    ]
    frontier = set(entities) | {_normalize(c) for c in _question_concepts(question)}
    result = []
    for _ in range(max_hops + 1):
        next_frontier = set()
        for row in rows:
            if row["id"] in seen or len(result) >= limit:
                continue
            subject, object_ = _normalize(row["subject_key"]), _normalize(row["object_key"])
            if subject not in frontier and object_ not in frontier:
                continue
            concepts = json.loads(row["concepts"])
            condition, event, effect = json.loads(row["condition_json"]), json.loads(row.get("condition_event_json", "null")), json.loads(row["effect_json"])
            if not isinstance(concepts, list) or not all(isinstance(item, str) for item in concepts):
                raise TypeError("relation concepts must be a JSON string list")
            if not isinstance(condition, str) or not isinstance(effect, str):
                raise TypeError("relation condition and effect must be JSON strings")
            row["concepts"] = tuple(concepts)
            if event is not None and not isinstance(event, dict):
                raise TypeError("relation condition event must be a JSON object or null")
            row["condition"], row["condition_event"], row["effect"] = condition, event, effect
            row["evidence_refs"] = _evidence(conn, "strategic_relation_evidence", (row["id"],))
            if not row["evidence_refs"]: continue
            result.append(row); seen.add(row["id"]); next_frontier.update((subject, object_)); next_frontier.update(_normalize(c) for c in row["concepts"])
        frontier = next_frontier
        if not frontier or len(result) >= limit:
            break
    return result


def _principles(conn, concepts, seen, minimum):
    if not concepts:
        return []
    results = []
    for row in conn.execute("SELECT * FROM compiled_principles WHERE data_version=? AND confidence>=? ORDER BY confidence DESC, id", (CURRENT_STRATEGIC_DATA_VERSION, minimum)):
        values = _decode_json_fields(dict(row), ("concepts",))["concepts"]
        if concepts and not concepts.intersection(values):
            continue
        if row["id"] in seen:
            continue
        result = dict(row); result["concepts"] = values
        result["evidence_refs"] = _evidence(conn, "compiled_principle_evidence", (row["id"],))
        if result["evidence_refs"]:
            seen.add(row["id"])
            results.append(result)
    return results


def _question_concepts(question):
    normalized = _normalize(question).replace("_", " ")
    return tuple(
        name for name in STRATEGIC_CONCEPTS
        if name.replace("_", " ") in normalized
    )


def _evidence(conn, table, owner_values):
    owner_columns = {
        "strategic_relation_evidence": ("relation_id",),
        "champion_fingerprint_evidence": ("champion", "data_version"),
        "compiled_principle_evidence": ("principle_id",),
    }[table]
    where = " AND ".join(f"{column} = ?" for column in owner_columns)
    return tuple(dict(row) for row in conn.execute(f"SELECT source_type, source_id, insight_id, quote FROM {table} WHERE {where}", owner_values))


def _format_evidence_refs(refs: tuple[dict, ...]) -> str:
    return ", ".join(
        f"{ref['source_type']}:{ref['source_id']}"
        + (f"#{ref['insight_id']}" if ref.get("insight_id") else "")
        for ref in refs
    )


def _decode_json_fields(row, fields):
    for field in fields:
        value = json.loads(row[field])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise TypeError(f"{field} must be a JSON string list")
        row[field] = tuple(value)
    return row


def _normalize(value: str) -> str:
    normalized = re.sub(r"(?<=[a-z0-9])['’]s\b", "", value.lower())
    normalized = re.sub(r"['’]", "", normalized)
    normalized = re.sub(r"[_-]+", " ", normalized)
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", normalized).split())


def _seen(values: set[str], key: str) -> bool:
    if key in values:
        return True
    values.add(key)
    return False
