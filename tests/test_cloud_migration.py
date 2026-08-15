import pathlib
import sqlite3
import tempfile
import unittest

import numpy as np

from cloud.migrate_supabase import (
    STRATEGIC_TABLE_COLUMNS,
    VECTOR_DIMENSION,
    embedding_blob_to_vector,
    iter_insight_payloads,
    iter_strategic_payloads,
    iter_video_payloads,
    source_db_name,
)
import core.database as db
from core.strategic_types import load_strategic_fixture


class CloudMigrationTests(unittest.TestCase):
    def test_embedding_blob_to_vector_converts_float32_blob(self) -> None:
        vector = np.arange(VECTOR_DIMENSION, dtype=np.float32)
        converted = embedding_blob_to_vector(vector.tobytes())

        self.assertEqual(len(converted or []), VECTOR_DIMENSION)
        self.assertEqual(converted[:3], [0.0, 1.0, 2.0])

    def test_embedding_blob_to_vector_rejects_wrong_dimension(self) -> None:
        with self.assertRaises(ValueError):
            embedding_blob_to_vector(np.arange(3, dtype=np.float32).tobytes())

    def test_iter_payloads_include_source_db_and_embedding_vector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = pathlib.Path(tmp) / "knowledge.db"
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE videos (
                    video_id TEXT PRIMARY KEY,
                    video_url TEXT NOT NULL,
                    video_title TEXT,
                    description TEXT,
                    game TEXT,
                    role TEXT,
                    subject TEXT,
                    champion TEXT,
                    rank TEXT,
                    website_rating REAL,
                    message_timestamp TEXT,
                    status TEXT,
                    transcription TEXT,
                    source TEXT,
                    created_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    subject TEXT,
                    subject_type TEXT,
                    source_score REAL,
                    cluster_score REAL,
                    confidence REAL,
                    repetition_count INTEGER,
                    is_duplicate INTEGER,
                    embedding BLOB,
                    created_at TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO videos (
                    video_id, video_url, video_title, game, role, subject, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("v1", "https://example.com", "Example", "aoe2", "general", "Khmer", "aoe2_pdf"),
            )
            conn.execute(
                """
                INSERT INTO insights (
                    video_id, insight_type, text, subject, subject_type, embedding
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "v1",
                    "civilization_identity",
                    "Khmer skip buildings.",
                    "Khmer",
                    "civilization",
                    np.zeros(VECTOR_DIMENSION, dtype=np.float32).tobytes(),
                ),
            )
            conn.commit()
            conn.close()

            videos = list(iter_video_payloads(db_path))
            insights = list(iter_insight_payloads(db_path))

        self.assertEqual(source_db_name(db_path), "knowledge")
        self.assertEqual(videos[0]["source_db"], "knowledge")
        self.assertEqual(videos[0]["source"], "aoe2_pdf")
        self.assertEqual(insights[0]["source_db"], "knowledge")
        self.assertEqual(insights[0]["local_id"], 1)
        self.assertEqual(len(insights[0]["embedding"]), VECTOR_DIMENSION)

    def test_iter_strategic_payloads_matches_local_persistence_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous_path = db.DB_PATH
            db_path = pathlib.Path(tmp) / "strategic.db"
            try:
                db.DB_PATH = db_path
                db.init_db()
                db.persist_strategic_fixture(
                    load_strategic_fixture("data/strategic_fixtures_v0.json")
                )

                relations = list(iter_strategic_payloads(db_path, "strategic_relations"))
                fingerprints = list(
                    iter_strategic_payloads(db_path, "champion_fingerprints")
                )
                relation_evidence = list(
                    iter_strategic_payloads(db_path, "strategic_relation_evidence")
                )
            finally:
                db.DB_PATH = previous_path

        self.assertEqual(set(relations[0]), set(STRATEGIC_TABLE_COLUMNS["strategic_relations"]))
        self.assertIn("condition_json", relations[0])
        self.assertIn("preferred_states", fingerprints[0])
        self.assertTrue(relation_evidence)

    def test_iter_strategic_payloads_handles_legacy_database_without_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = pathlib.Path(tmp) / "legacy.db"
            sqlite3.connect(db_path).close()

            rows = list(iter_strategic_payloads(db_path, "strategic_relations"))

        self.assertEqual(rows, [])

    def test_iter_strategic_payloads_defaults_legacy_relation_ontology_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = pathlib.Path(tmp) / "legacy.db"
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE strategic_relations (
                    id TEXT PRIMARY KEY, subject_type TEXT, subject_key TEXT,
                    relation_type TEXT, object_type TEXT, object_key TEXT,
                    condition_json TEXT, effect_json TEXT, concepts TEXT,
                    confidence REAL, provenance_type TEXT, patch_sensitivity TEXT,
                    data_version TEXT, created_at TEXT, updated_at TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO strategic_relations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("legacy", "ability", "Thresh E", "denies", "concept", "continuity", '""', '""', "[]", 0.8, "manual_fixture", "low", "strategic-fixtures-v0", None, None),
            )
            conn.commit()
            conn.close()

            rows = list(iter_strategic_payloads(db_path, "strategic_relations"))

        self.assertEqual(rows[0]["ontology_version"], "strategic-ontology-v0")

    def test_iter_strategic_payloads_rejects_unknown_table(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown strategic table"):
            list(iter_strategic_payloads("missing.db", "not_a_strategic_table"))

if __name__ == "__main__":
    unittest.main()
