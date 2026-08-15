"""SQLite database setup and insert/query helpers for videoSorter."""

import os
import re
import sqlite3
import pathlib
from dataclasses import replace
from typing import Optional

from core.ontology import ONTOLOGY_VERSION, STRATEGIC_CONCEPTS
from core.strategic_types import (
    AUTOMATED_RELATION_DATA_VERSION,
    EvidenceRef,
    StrategicFixture,
    dedupe_relations,
)

# Override with DB_PATH env var to target a different database file.
# Secondary knowledge ingestion defaults to knowledge.db so broader scraped
# content stays separate from the primary Discord/coaching dataset in videos.db.
DB_PATH = pathlib.Path(os.environ.get("DB_PATH", "videos.db"))


_AOE2_CONTROLS_SETTINGS_RE = re.compile(
    r"\b("
    r"hotkey|hotkeys|keybind|keybinds|control group|control groups|grouping|"
    r"ui|interface|camera|settings|shortcut|shortcuts|"
    r"select all|go to tc|idle villager|"
    r"shift queue|shift-queue|waypoint|waypoints|minimap"
    r")\b",
    re.IGNORECASE,
)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row  # lets you access columns by name
    return conn


def _migrate_aoe2_insight_types(conn: sqlite3.Connection) -> None:
    """Rewrite old AoE2 insight keys into clearer canonical names."""
    rows = conn.execute(
        """
        SELECT i.id, i.text
        FROM insights AS i
        JOIN videos AS v ON v.video_id = i.video_id
        WHERE i.insight_type = 'game_mechanics'
          AND v.game = 'aoe2'
        """
    ).fetchall()
    migrate_ids = [
        row["id"]
        for row in rows
        if _AOE2_CONTROLS_SETTINGS_RE.search((row["text"] or "").lower())
    ]
    if not migrate_ids:
        return
    conn.executemany(
        "UPDATE insights SET insight_type = 'controls_settings' WHERE id = ?",
        [(insight_id,) for insight_id in migrate_ids],
    )


def _init_strategic_tables(conn: sqlite3.Connection) -> None:
    """Create derived strategic-knowledge tables without modifying raw evidence."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS strategic_concepts (
            canonical_name     TEXT NOT NULL,
            ontology_version   TEXT NOT NULL,
            concept_type       TEXT NOT NULL,
            description        TEXT NOT NULL,
            scope              TEXT NOT NULL,
            patch_sensitivity  TEXT NOT NULL,
            PRIMARY KEY (canonical_name, ontology_version)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS strategic_relations (
            id                TEXT PRIMARY KEY,
            subject_type      TEXT NOT NULL,
            subject_key       TEXT NOT NULL,
            relation_type     TEXT NOT NULL,
            object_type       TEXT NOT NULL,
            object_key        TEXT NOT NULL,
            condition_json    TEXT NOT NULL DEFAULT '\"\"',
            effect_json       TEXT NOT NULL DEFAULT '\"\"',
            concepts          TEXT NOT NULL DEFAULT '[]',
            confidence        REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            provenance_type   TEXT NOT NULL,
            patch_sensitivity TEXT NOT NULL,
            data_version      TEXT NOT NULL,
            ontology_version  TEXT NOT NULL DEFAULT 'strategic-ontology-v0',
            created_at        TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at        TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE (
                data_version,
                ontology_version,
                subject_type,
                subject_key,
                relation_type,
                object_type,
                object_key,
                condition_json,
                effect_json
            )
        )
        """
    )
    _migrate_strategic_json_columns(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS strategic_relation_evidence (
            relation_id  TEXT NOT NULL REFERENCES strategic_relations(id) ON DELETE CASCADE,
            source_type  TEXT NOT NULL,
            source_id    TEXT NOT NULL,
            insight_id   TEXT NOT NULL DEFAULT '',
            quote        TEXT,
            PRIMARY KEY (relation_id, source_type, source_id, insight_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS champion_fingerprints (
            champion             TEXT NOT NULL,
            data_version         TEXT NOT NULL,
            preferred_states    TEXT NOT NULL DEFAULT '[]',
            avoided_states      TEXT NOT NULL DEFAULT '[]',
            persistent_advantages TEXT NOT NULL DEFAULT '[]',
            conditional_advantages TEXT NOT NULL DEFAULT '[]',
            dependencies        TEXT NOT NULL DEFAULT '[]',
            access_tools        TEXT NOT NULL DEFAULT '[]',
            access_denial_tools TEXT NOT NULL DEFAULT '[]',
            continuity_requirements TEXT NOT NULL DEFAULT '[]',
            conversion_patterns TEXT NOT NULL DEFAULT '[]',
            role_flip_events    TEXT NOT NULL DEFAULT '[]',
            failure_modes       TEXT NOT NULL DEFAULT '[]',
            confidence            REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            provenance_type       TEXT NOT NULL,
            patch_sensitivity     TEXT NOT NULL,
            created_at            TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at            TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (champion, data_version)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS champion_fingerprint_evidence (
            champion      TEXT NOT NULL,
            data_version  TEXT NOT NULL,
            source_type   TEXT NOT NULL,
            source_id     TEXT NOT NULL,
            insight_id    TEXT NOT NULL DEFAULT '',
            quote         TEXT,
            PRIMARY KEY (champion, data_version, source_type, source_id, insight_id),
            FOREIGN KEY (champion, data_version)
                REFERENCES champion_fingerprints(champion, data_version)
                ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compiled_principles (
            id                TEXT PRIMARY KEY,
            title             TEXT NOT NULL,
            summary           TEXT NOT NULL,
            concepts          TEXT NOT NULL DEFAULT '[]',
            confidence        REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            provenance_type   TEXT NOT NULL,
            scope             TEXT NOT NULL,
            patch_sensitivity TEXT NOT NULL,
            data_version      TEXT NOT NULL,
            created_at        TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compiled_principle_evidence (
            principle_id TEXT NOT NULL REFERENCES compiled_principles(id) ON DELETE CASCADE,
            source_type  TEXT NOT NULL,
            source_id    TEXT NOT NULL,
            insight_id   TEXT NOT NULL DEFAULT '',
            quote        TEXT,
            PRIMARY KEY (principle_id, source_type, source_id, insight_id)
        )
        """
    )
    conn.execute("DROP INDEX IF EXISTS strategic_relations_subject_idx")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS strategic_relations_subject_idx "
        "ON strategic_relations (data_version, ontology_version, subject_type, subject_key)"
    )
    conn.execute("DROP INDEX IF EXISTS strategic_relations_object_idx")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS strategic_relations_object_idx "
        "ON strategic_relations (data_version, ontology_version, object_type, object_key)"
    )


def _migrate_strategic_json_columns(conn: sqlite3.Connection) -> None:
    """Preserve data written by the first local Milestone 2 schema shape."""
    migrations = {
        "strategic_relations": {"concepts_json": "concepts"},
        "champion_fingerprints": {
            "preferred_states_json": "preferred_states",
            "avoided_states_json": "avoided_states",
            "persistent_advantages_json": "persistent_advantages",
            "conditional_advantages_json": "conditional_advantages",
            "dependencies_json": "dependencies",
            "access_tools_json": "access_tools",
            "access_denial_tools_json": "access_denial_tools",
            "continuity_requirements_json": "continuity_requirements",
            "conversion_patterns_json": "conversion_patterns",
            "role_flip_events_json": "role_flip_events",
            "failure_modes_json": "failure_modes",
        },
        "compiled_principles": {"concepts_json": "concepts"},
    }
    for table, columns in migrations.items():
        existing = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for old_name, new_name in columns.items():
            if old_name not in existing or new_name in existing:
                continue
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {new_name} TEXT NOT NULL DEFAULT '[]'"
            )
            conn.execute(f"UPDATE {table} SET {new_name} = {old_name}")
            existing.add(new_name)
        if table == "strategic_relations" and "ontology_version" not in existing:
            conn.execute(
                "ALTER TABLE strategic_relations ADD COLUMN ontology_version "
                "TEXT NOT NULL DEFAULT 'strategic-ontology-v0'"
            )
            existing.add("ontology_version")
    _rebuild_legacy_relation_identity(conn)


def _rebuild_legacy_relation_identity(conn: sqlite3.Connection) -> None:
    """Upgrade the old SQLite relation unique key without losing provenance rows."""
    unique_indexes = conn.execute("PRAGMA index_list(strategic_relations)").fetchall()
    for index in unique_indexes:
        if not index["unique"]:
            continue
        columns = [
            row["name"]
            for row in conn.execute(f"PRAGMA index_info({index['name']})").fetchall()
        ]
        if "data_version" in columns and "subject_key" in columns:
            if "ontology_version" in columns:
                return
            break
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute(
            """
            CREATE TABLE strategic_relations_v2 (
                id TEXT PRIMARY KEY,
                subject_type TEXT NOT NULL,
                subject_key TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                object_type TEXT NOT NULL,
                object_key TEXT NOT NULL,
                condition_json TEXT NOT NULL DEFAULT '\"\"',
                effect_json TEXT NOT NULL DEFAULT '\"\"',
                concepts TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                provenance_type TEXT NOT NULL,
                patch_sensitivity TEXT NOT NULL,
                data_version TEXT NOT NULL,
                ontology_version TEXT NOT NULL DEFAULT 'strategic-ontology-v0',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE (data_version, ontology_version, subject_type, subject_key,
                    relation_type, object_type, object_key, condition_json, effect_json)
            )
            """
        )
        columns = (
            "id, subject_type, subject_key, relation_type, object_type, object_key, "
            "condition_json, effect_json, concepts, confidence, provenance_type, "
            "patch_sensitivity, data_version, ontology_version, created_at, updated_at"
        )
        available = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(strategic_relations)").fetchall()
        }
        select_columns = [
            column if column in available else "datetime('now')"
            for column in columns.split(", ")
        ]
        conn.execute(
            f"INSERT INTO strategic_relations_v2 ({columns}) "
            f"SELECT {', '.join(select_columns)} FROM strategic_relations"
        )
        conn.execute("DROP TABLE strategic_relations")
        conn.execute("ALTER TABLE strategic_relations_v2 RENAME TO strategic_relations")
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def _json_list(values: tuple[str, ...]) -> str:
    """Serialize immutable domain lists in a deterministic SQLite-compatible form."""
    import json

    return json.dumps(values, separators=(",", ":"))


def _json_scalar(value: str | None) -> str:
    import json

    return json.dumps(value or "", separators=(",", ":"))


def _normalize_strategic_key(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _persist_evidence_refs(
    conn: sqlite3.Connection,
    table: str,
    owner_values: tuple[str, ...],
    evidence_refs: tuple,
) -> None:
    columns = {
        "strategic_relation_evidence": "relation_id",
        "champion_fingerprint_evidence": "champion, data_version",
        "compiled_principle_evidence": "principle_id",
    }[table]
    owner_placeholders = ", ".join("?" for _ in owner_values)
    conn.execute(
        f"DELETE FROM {table} WHERE "
        + " AND ".join(f"{column.strip()} = ?" for column in columns.split(",")),
        owner_values,
    )
    conn.executemany(
        f"""
        INSERT INTO {table} ({columns}, source_type, source_id, insight_id, quote)
        VALUES ({owner_placeholders}, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        [
            (*owner_values, ref.source_type, ref.source_id, ref.insight_id or "", ref.quote)
            for ref in evidence_refs
        ],
    )


def _merge_evidence_refs(existing_refs: tuple, incoming_refs: tuple) -> tuple:
    merged = list(existing_refs)
    keys = {ref.stable_key() for ref in merged}
    for ref in incoming_refs:
        if ref.stable_key() not in keys:
            merged.append(ref)
            keys.add(ref.stable_key())
    return tuple(merged)


def persist_strategic_fixture(fixture: StrategicFixture) -> None:
    """Persist manual fixture data; automated batches use accepted decisions only."""
    if fixture.data_version == AUTOMATED_RELATION_DATA_VERSION:
        raise ValueError("automated relations must be persisted from accepted decisions")
    _persist_strategic_fixture(fixture)


def _persist_strategic_fixture(fixture: StrategicFixture) -> None:
    """Persist validated derived knowledge while keeping it separate from insights."""
    fixture.validate()
    with get_connection() as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        _init_strategic_tables(conn)
        conn.executemany(
            """
            INSERT INTO strategic_concepts (
                canonical_name, ontology_version, concept_type, description, scope, patch_sensitivity
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical_name, ontology_version) DO UPDATE SET
                concept_type = excluded.concept_type,
                description = excluded.description,
                scope = excluded.scope,
                patch_sensitivity = excluded.patch_sensitivity
            """,
            [
                (
                    concept.canonical_name,
                    ONTOLOGY_VERSION,
                    concept.concept_type,
                    concept.description,
                    concept.scope,
                    concept.patch_sensitivity,
                )
                for concept in STRATEGIC_CONCEPTS.values()
            ],
        )
        for relation in dedupe_relations(fixture.relations):
            stable_candidates = conn.execute(
                """
                SELECT id, confidence, subject_key, object_key FROM strategic_relations
                WHERE data_version = ? AND ontology_version = ? AND subject_type = ?
                  AND relation_type = ? AND object_type = ?
                  AND condition_json = ? AND effect_json = ?
                """,
                (
                    relation.data_version,
                    relation.ontology_version,
                    relation.subject_type,
                    relation.relation_type,
                    relation.object_type,
                    _json_scalar(relation.condition),
                    _json_scalar(relation.effect),
                ),
            ).fetchall()
            stable_match = next(
                (
                    candidate
                    for candidate in stable_candidates
                    if _normalize_strategic_key(candidate["subject_key"])
                    == _normalize_strategic_key(relation.subject_key)
                    and _normalize_strategic_key(candidate["object_key"])
                    == _normalize_strategic_key(relation.object_key)
                ),
                None,
            )
            relation_to_persist = relation
            if stable_match is not None and stable_match["id"] != relation.id:
                stored_refs = tuple(
                    EvidenceRef(
                        source_type=row["source_type"],
                        source_id=row["source_id"],
                        insight_id=row["insight_id"] or None,
                        quote=row["quote"],
                    )
                    for row in conn.execute(
                        """
                        SELECT source_type, source_id, insight_id, quote
                        FROM strategic_relation_evidence WHERE relation_id = ?
                        """,
                        (stable_match["id"],),
                    )
                )
                relation_to_persist = replace(
                    relation,
                    id=stable_match["id"],
                    subject_key=stable_match["subject_key"],
                    object_key=stable_match["object_key"],
                    confidence=max(relation.confidence, stable_match["confidence"]),
                    evidence_refs=_merge_evidence_refs(stored_refs, relation.evidence_refs),
                )
            conn.execute(
                """
                INSERT INTO strategic_relations (
                    id, subject_type, subject_key, relation_type, object_type, object_key,
                    condition_json, effect_json, concepts, confidence, provenance_type,
                    patch_sensitivity, data_version, ontology_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    subject_type = excluded.subject_type,
                    subject_key = excluded.subject_key,
                    relation_type = excluded.relation_type,
                    object_type = excluded.object_type,
                    object_key = excluded.object_key,
                    condition_json = excluded.condition_json,
                    effect_json = excluded.effect_json,
                    concepts = excluded.concepts,
                    confidence = excluded.confidence,
                    provenance_type = excluded.provenance_type,
                    patch_sensitivity = excluded.patch_sensitivity,
                    data_version = excluded.data_version,
                    ontology_version = excluded.ontology_version,
                    updated_at = datetime('now')
                """,
                (
                    relation_to_persist.id,
                    relation_to_persist.subject_type,
                    relation_to_persist.subject_key,
                    relation_to_persist.relation_type,
                    relation_to_persist.object_type,
                    relation_to_persist.object_key,
                    _json_scalar(relation_to_persist.condition),
                    _json_scalar(relation_to_persist.effect),
                    _json_list(relation_to_persist.concepts),
                    relation_to_persist.confidence,
                    relation_to_persist.provenance_type,
                    relation_to_persist.patch_sensitivity,
                    relation_to_persist.data_version,
                    relation_to_persist.ontology_version,
                ),
            )
            _persist_evidence_refs(
                conn,
                "strategic_relation_evidence",
                (relation_to_persist.id,),
                relation_to_persist.evidence_refs,
            )
        for fingerprint in fixture.fingerprints:
            conn.execute(
                """
                INSERT INTO champion_fingerprints (
                    champion, data_version, preferred_states, avoided_states,
                    persistent_advantages, conditional_advantages, dependencies,
                    access_tools, access_denial_tools, continuity_requirements,
                    conversion_patterns, role_flip_events, failure_modes,
                    confidence, provenance_type, patch_sensitivity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(champion, data_version) DO UPDATE SET
                    preferred_states = excluded.preferred_states,
                    avoided_states = excluded.avoided_states,
                    persistent_advantages = excluded.persistent_advantages,
                    conditional_advantages = excluded.conditional_advantages,
                    dependencies = excluded.dependencies,
                    access_tools = excluded.access_tools,
                    access_denial_tools = excluded.access_denial_tools,
                    continuity_requirements = excluded.continuity_requirements,
                    conversion_patterns = excluded.conversion_patterns,
                    role_flip_events = excluded.role_flip_events,
                    failure_modes = excluded.failure_modes,
                    confidence = excluded.confidence,
                    provenance_type = excluded.provenance_type,
                    patch_sensitivity = excluded.patch_sensitivity,
                    updated_at = datetime('now')
                """,
                (
                    fingerprint.champion,
                    fingerprint.data_version,
                    _json_list(fingerprint.preferred_states),
                    _json_list(fingerprint.avoided_states),
                    _json_list(fingerprint.persistent_advantages),
                    _json_list(fingerprint.conditional_advantages),
                    _json_list(fingerprint.dependencies),
                    _json_list(fingerprint.access_tools),
                    _json_list(fingerprint.access_denial_tools),
                    _json_list(fingerprint.continuity_requirements),
                    _json_list(fingerprint.conversion_patterns),
                    _json_list(fingerprint.role_flip_events),
                    _json_list(fingerprint.failure_modes),
                    fingerprint.confidence,
                    fingerprint.provenance_type,
                    fingerprint.patch_sensitivity,
                ),
            )
            _persist_evidence_refs(
                conn,
                "champion_fingerprint_evidence",
                (fingerprint.champion, fingerprint.data_version),
                fingerprint.evidence_refs,
            )
        for principle in fixture.principles:
            conn.execute(
                """
                INSERT INTO compiled_principles (
                    id, title, summary, concepts, confidence, provenance_type,
                    scope, patch_sensitivity, data_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    summary = excluded.summary,
                    concepts = excluded.concepts,
                    confidence = excluded.confidence,
                    provenance_type = excluded.provenance_type,
                    scope = excluded.scope,
                    patch_sensitivity = excluded.patch_sensitivity,
                    data_version = excluded.data_version,
                    updated_at = datetime('now')
                """,
                (
                    principle.id,
                    principle.title,
                    principle.summary,
                    _json_list(principle.concepts),
                    principle.confidence,
                    principle.provenance_type,
                    principle.scope,
                    principle.patch_sensitivity,
                    principle.data_version,
                ),
            )
            _persist_evidence_refs(
                conn,
                "compiled_principle_evidence",
                (principle.id,),
                principle.evidence_refs,
            )
        conn.commit()


def persist_strategic_relations(decisions: tuple | list) -> None:
    """Persist only accepted Phase 2 compiler decisions, never review/rejected output."""
    relations = []
    for decision in decisions:
        if getattr(decision, "status", None) != "accepted" or getattr(decision, "relation", None) is None:
            raise ValueError("automated persistence requires accepted compiler decisions")
        relation = decision.relation
        if relation.data_version != AUTOMATED_RELATION_DATA_VERSION:
            raise ValueError("automated persistence requires automated relation data")
        relations.append(relation)
    fixture = StrategicFixture(
        ontology_version=ONTOLOGY_VERSION,
        data_version=AUTOMATED_RELATION_DATA_VERSION,
        fingerprints=(),
        relations=dedupe_relations(relations),
        principles=(),
    )
    _persist_strategic_fixture(fixture)


def init_db() -> None:
    """Create tables if they don't exist yet."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id          TEXT PRIMARY KEY,
                video_url         TEXT NOT NULL,
                video_title       TEXT,
                description       TEXT,
                game              TEXT DEFAULT 'lol',
                role              TEXT NOT NULL,
                subject           TEXT,
                champion          TEXT,
                rank              TEXT,
                website_rating    REAL,
                message_timestamp TEXT,
                status            TEXT DEFAULT 'pending',
                transcription     TEXT,
                created_at        TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS insights (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id      TEXT NOT NULL REFERENCES videos(video_id),
                insight_type  TEXT NOT NULL,
                text          TEXT NOT NULL,
                subject       TEXT,
                subject_type  TEXT,
                source_score  REAL DEFAULT NULL,
                cluster_score REAL DEFAULT NULL,
                confidence    REAL DEFAULT NULL,
                created_at    TEXT DEFAULT (datetime('now'))
            )
        """)
        # Add columns to existing DBs that predate this schema
        for col, typedef in [
            ("subject",           "TEXT DEFAULT NULL"),
            ("subject_type",      "TEXT DEFAULT NULL"),
            ("source_score",      "REAL DEFAULT NULL"),
            ("cluster_score",     "REAL DEFAULT NULL"),
            ("confidence",        "REAL DEFAULT NULL"),
            ("repetition_count",  "INTEGER DEFAULT 1"),
            ("is_duplicate",      "INTEGER DEFAULT 0"),
        ]:
            try:
                conn.execute(f"ALTER TABLE insights ADD COLUMN {col} {typedef}")
            except Exception:
                pass
        for col, typedef in [
            ("source", "TEXT DEFAULT 'discord'"),
            ("game", "TEXT DEFAULT 'lol'"),
            ("subject", "TEXT DEFAULT NULL"),
            ("website_rating", "REAL DEFAULT NULL"),
        ]:
            try:
                conn.execute(f"ALTER TABLE videos ADD COLUMN {col} {typedef}")
            except Exception:
                pass  # column already exists
        try:
            conn.execute(
                "UPDATE videos SET game = 'lol' WHERE game IS NULL OR TRIM(game) = ''"
            )
        except Exception:
            pass
        try:
            conn.execute(
                """
                UPDATE videos
                SET subject = champion
                WHERE (subject IS NULL OR TRIM(subject) = '')
                  AND champion IS NOT NULL
                  AND TRIM(champion) != ''
                """
            )
        except Exception:
            pass
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_descriptions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                role            TEXT NOT NULL,
                description     TEXT NOT NULL,
                message_timestamp TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS champion_archetypes (
                champion   TEXT NOT NULL,
                role       TEXT NOT NULL,
                archetype  TEXT NOT NULL,
                source     TEXT DEFAULT 'empirical',
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (champion, role)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS champion_abilities (
                champion      TEXT NOT NULL,
                ability_slot  TEXT NOT NULL,
                name          TEXT,
                description   TEXT,
                cooldown      TEXT,
                range         TEXT,
                cost          TEXT,
                properties    TEXT,
                PRIMARY KEY (champion, ability_slot)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS champion_stats (
                champion       TEXT PRIMARY KEY,
                hp             REAL, hp_level      REAL,
                armor          REAL, armor_level   REAL,
                mr             REAL, mr_level      REAL,
                attack_range   REAL,
                attack_damage  REAL, ad_level      REAL,
                attack_speed   REAL, as_level      REAL,
                movespeed      REAL,
                scraped_at     TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS champion_stat_notes (
                champion         TEXT NOT NULL,
                stat_key         TEXT NOT NULL,
                note             TEXT NOT NULL,
                z_score          REAL NOT NULL,
                comparison_group TEXT NOT NULL,
                PRIMARY KEY (champion, stat_key)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_queries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                question        TEXT NOT NULL,
                expected_answer TEXT,
                notes           TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_ratings (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id              INTEGER REFERENCES eval_queries(id),
                question              TEXT NOT NULL,
                intent_type           TEXT,
                champion_a            TEXT,
                champion_b            TEXT,
                answer_good           INTEGER NOT NULL,  -- 1=good, 0=bad
                confidence_aligned    INTEGER NOT NULL,  -- 1=aligned, 0=misaligned
                retrieved_insight_ids TEXT,
                generated_answer      TEXT,
                retrieval_method      TEXT DEFAULT 'rrf',
                shown_order           INTEGER,
                rated_at              TEXT DEFAULT (datetime('now'))
            )
        """)
        _init_strategic_tables(conn)
        try:
            _migrate_aoe2_insight_types(conn)
        except Exception:
            pass
        conn.commit()


def insert_video(
    video_id: str,
    video_url: str,
    role: str,
    message_timestamp: str,
    video_title: Optional[str] = None,
    description: Optional[str] = None,
    game: str = "lol",
    subject: Optional[str] = None,
    champion: Optional[str] = None,
    rank: Optional[str] = None,
    website_rating: float | None = None,
    source: str = "discord",
) -> None:
    """Insert a video row, ignoring duplicates (same video_id)."""
    if subject is None:
        subject = champion
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO videos
                (video_id, video_url, video_title, description, game, role, subject, champion, rank, website_rating, message_timestamp, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_id,
                video_url,
                video_title,
                description,
                game,
                role,
                subject,
                champion,
                rank,
                website_rating,
                message_timestamp,
                source,
            ),
        )
        conn.commit()


def insert_pending_description(role: str, description: str, message_timestamp: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO pending_descriptions (role, description, message_timestamp) VALUES (?, ?, ?)",
            (role, description, message_timestamp),
        )
        conn.commit()


def set_status(video_id: str, status: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE videos SET status = ? WHERE video_id = ?", (status, video_id))
        conn.commit()


def set_transcription(video_id: str, transcription: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE videos SET transcription = ?, status = 'transcribed' WHERE video_id = ?",
            (transcription, video_id),
        )
        conn.commit()


def insert_insight(
    video_id: str,
    insight_type: str,
    text: str,
    source_score: float | None = None,
    repetition_count: int = 1,
    subject: Optional[str] = None,
    subject_type: Optional[str] = None,
) -> int:
    """Insert an insight and return its row id."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO insights (
                video_id, insight_type, text, subject, subject_type, source_score, repetition_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (video_id, insight_type, text, subject, subject_type, source_score, repetition_count),
        )
        conn.commit()
        return cur.lastrowid


def update_cluster_scores(scores: list[tuple[float, float, int]]) -> None:
    """
    Bulk-update cluster_score and confidence for a list of insights.
    scores: list of (cluster_score, confidence, insight_id)
    """
    with get_connection() as conn:
        conn.executemany(
            "UPDATE insights SET cluster_score = ?, confidence = ? WHERE id = ?",
            scores,
        )
        conn.commit()


def get_all_insights_with_embeddings() -> list:
    """Return all insights that have an embedding stored."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT id, video_id, insight_type, text, subject, subject_type, embedding, source_score
            FROM insights
            WHERE embedding IS NOT NULL
            """
        ).fetchall()


def get_videos_by_status(status: str) -> list:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM videos WHERE status = ?", (status,)
        ).fetchall()


def try_fill_descriptions() -> None:
    """
    For any video with no description, look for a pending_description
    in the same role within 2 hours of the video's message_timestamp.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT video_id, role, message_timestamp FROM videos WHERE description IS NULL OR description = ''"
        ).fetchall()

        for row in rows:
            match = conn.execute(
                """
                SELECT id, description FROM pending_descriptions
                WHERE role = ?
                  AND ABS(
                      strftime('%s', message_timestamp) - strftime('%s', ?)
                  ) <= 7200
                ORDER BY ABS(
                    strftime('%s', message_timestamp) - strftime('%s', ?)
                )
                LIMIT 1
                """,
                (row["role"], row["message_timestamp"], row["message_timestamp"]),
            ).fetchone()

            if match:
                conn.execute(
                    "UPDATE videos SET description = ? WHERE video_id = ?",
                    (match["description"], row["video_id"]),
                )
                conn.execute(
                    "DELETE FROM pending_descriptions WHERE id = ?",
                    (match["id"],),
                )

        conn.commit()
