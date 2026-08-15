import json
from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import unittest

import core.database as db
from core.ontology import ONTOLOGY_VERSION, STRATEGIC_CONCEPTS
from core.strategic_types import (
    EvidenceRef,
    StrategicValidationError,
    load_strategic_fixture,
)


FIXTURE_PATH = Path("data/strategic_fixtures_v0.json")
STRATEGIC_TABLES = {
    "strategic_concepts",
    "strategic_relations",
    "strategic_relation_evidence",
    "champion_fingerprints",
    "champion_fingerprint_evidence",
    "compiled_principles",
    "compiled_principle_evidence",
}


class StrategicPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_db_path = db.DB_PATH
        db.DB_PATH = Path(self._tmpdir.name) / "strategic.db"

    def tearDown(self) -> None:
        db.DB_PATH = self._old_db_path
        self._tmpdir.cleanup()

    def test_init_db_creates_strategic_schema_idempotently_without_touching_evidence(self) -> None:
        db.init_db()
        db.init_db()

        with db.get_connection() as conn:
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            insight_count = conn.execute("SELECT COUNT(*) FROM insights").fetchone()[0]

        self.assertTrue(STRATEGIC_TABLES.issubset(tables))
        self.assertEqual(insight_count, 0)

    def test_fixture_persists_derived_records_and_all_provenance(self) -> None:
        db.init_db()
        fixture = load_strategic_fixture(FIXTURE_PATH)

        db.persist_strategic_fixture(fixture)

        with db.get_connection() as conn:
            concept_count = conn.execute(
                "SELECT COUNT(*) FROM strategic_concepts WHERE ontology_version = ?",
                (ONTOLOGY_VERSION,),
            ).fetchone()[0]
            relation_count = conn.execute(
                "SELECT COUNT(*) FROM strategic_relations"
            ).fetchone()[0]
            fingerprint_count = conn.execute(
                "SELECT COUNT(*) FROM champion_fingerprints"
            ).fetchone()[0]
            principle_count = conn.execute(
                "SELECT COUNT(*) FROM compiled_principles"
            ).fetchone()[0]
            relation_evidence_count = conn.execute(
                "SELECT COUNT(*) FROM strategic_relation_evidence"
            ).fetchone()[0]
            fingerprint_evidence_count = conn.execute(
                "SELECT COUNT(*) FROM champion_fingerprint_evidence"
            ).fetchone()[0]
            principle_evidence_count = conn.execute(
                "SELECT COUNT(*) FROM compiled_principle_evidence"
            ).fetchone()[0]
            stored_relation = conn.execute(
                "SELECT condition_json, effect_json, concepts FROM strategic_relations LIMIT 1"
            ).fetchone()
            insight_count = conn.execute("SELECT COUNT(*) FROM insights").fetchone()[0]

        self.assertEqual(concept_count, len(STRATEGIC_CONCEPTS))
        self.assertEqual(relation_count, len(fixture.relations))
        self.assertEqual(fingerprint_count, len(fixture.fingerprints))
        self.assertEqual(principle_count, len(fixture.principles))
        self.assertEqual(
            relation_evidence_count,
            sum(len(relation.evidence_refs) for relation in fixture.relations),
        )
        self.assertEqual(
            fingerprint_evidence_count,
            sum(len(fingerprint.evidence_refs) for fingerprint in fixture.fingerprints),
        )
        self.assertEqual(
            principle_evidence_count,
            sum(len(principle.evidence_refs) for principle in fixture.principles),
        )
        self.assertIsInstance(json.loads(stored_relation["condition_json"]), str)
        self.assertIsInstance(json.loads(stored_relation["effect_json"]), str)
        self.assertIsInstance(json.loads(stored_relation["concepts"]), list)
        self.assertEqual(insight_count, 0)

    def test_repeated_fixture_write_is_idempotent(self) -> None:
        db.init_db()
        fixture = load_strategic_fixture(FIXTURE_PATH)

        db.persist_strategic_fixture(fixture)
        db.persist_strategic_fixture(fixture)

        with db.get_connection() as conn:
            relation_count = conn.execute(
                "SELECT COUNT(*) FROM strategic_relations"
            ).fetchone()[0]
            evidence_count = conn.execute(
                "SELECT COUNT(*) FROM strategic_relation_evidence"
            ).fetchone()[0]

        self.assertEqual(relation_count, len(fixture.relations))
        self.assertEqual(
            evidence_count,
            sum(len(relation.evidence_refs) for relation in fixture.relations),
        )

    def test_duplicate_relation_support_is_merged_before_persistence(self) -> None:
        db.init_db()
        fixture = load_strategic_fixture(FIXTURE_PATH)
        original = fixture.relations[0]
        duplicate = replace(
            original,
            id="same-relation-with-extra-support",
            evidence_refs=(
                EvidenceRef(
                    source_type="manual_analysis",
                    source_id="extra-proof",
                    insight_id="extra-insight",
                ),
            ),
        )
        merged_fixture = replace(
            fixture,
            relations=(original, duplicate, *fixture.relations[1:]),
        )

        db.persist_strategic_fixture(merged_fixture)

        with db.get_connection() as conn:
            evidence_count = conn.execute(
                "SELECT COUNT(*) FROM strategic_relation_evidence WHERE relation_id = ?",
                (original.id,),
            ).fetchone()[0]

        self.assertEqual(evidence_count, 2)

    def test_relation_upsert_replaces_stale_provenance_for_the_same_relation_id(self) -> None:
        db.init_db()
        fixture = load_strategic_fixture(FIXTURE_PATH)
        db.persist_strategic_fixture(fixture)
        original = fixture.relations[0]
        changed = replace(
            original,
            effect="updated strategic effect",
            evidence_refs=(
                EvidenceRef(
                    source_type="manual_analysis",
                    source_id="replacement-proof",
                    insight_id="replacement-insight",
                ),
            ),
        )

        db.persist_strategic_fixture(
            replace(fixture, relations=(changed, *fixture.relations[1:]))
        )

        with db.get_connection() as conn:
            evidence_rows = conn.execute(
                """
                SELECT source_id FROM strategic_relation_evidence
                WHERE relation_id = ?
                """,
                (original.id,),
            ).fetchall()

        self.assertEqual([row["source_id"] for row in evidence_rows], ["replacement-proof"])

    def test_conditioned_contradictions_persist_as_distinct_relations(self) -> None:
        db.init_db()
        fixture = load_strategic_fixture(FIXTURE_PATH)
        original = fixture.relations[0]
        contradiction = replace(
            original,
            id="same-edge-different-condition",
            condition="when the enemy cannot contest the lane",
            effect="the opposite strategic consequence applies",
        )

        db.persist_strategic_fixture(
            replace(fixture, relations=(original, contradiction, *fixture.relations[1:]))
        )

        with db.get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM strategic_relations WHERE subject_key = ?",
                (original.subject_key,),
            ).fetchone()[0]

        self.assertGreaterEqual(count, 2)

    def test_persistence_revalidates_directly_constructed_stale_relation(self) -> None:
        db.init_db()
        fixture = load_strategic_fixture(FIXTURE_PATH)
        stale_relation = replace(fixture.relations[0], data_version="strategic-fixtures-v99")

        with self.assertRaisesRegex(StrategicValidationError, "unsupported strategic data version"):
            db.persist_strategic_fixture(
                replace(fixture, relations=(stale_relation, *fixture.relations[1:]))
            )

    def test_persistence_revalidates_directly_constructed_invalid_evidence_ref(self) -> None:
        db.init_db()
        fixture = load_strategic_fixture(FIXTURE_PATH)
        invalid_relation = replace(
            fixture.relations[0],
            evidence_refs=(EvidenceRef(source_type="", source_id=""),),
        )

        with self.assertRaisesRegex(StrategicValidationError, "evidence source_type"):
            db.persist_strategic_fixture(
                replace(fixture, relations=(invalid_relation, *fixture.relations[1:]))
            )

    def test_stable_duplicate_across_separate_writes_merges_evidence(self) -> None:
        db.init_db()
        fixture = load_strategic_fixture(FIXTURE_PATH)
        original = fixture.relations[0]
        db.persist_strategic_fixture(fixture)
        duplicate = replace(
            original,
            id="stable-duplicate-second-write",
            evidence_refs=(
                EvidenceRef(
                    source_type="manual_analysis",
                    source_id="second-write-proof",
                    insight_id="second-write-insight",
                ),
            ),
        )

        db.persist_strategic_fixture(
            replace(fixture, relations=(duplicate, *fixture.relations[1:]))
        )

        with db.get_connection() as conn:
            relation_count = conn.execute(
                "SELECT COUNT(*) FROM strategic_relations WHERE subject_key = ? AND relation_type = ?",
                (original.subject_key, original.relation_type),
            ).fetchone()[0]
            evidence_count = conn.execute(
                "SELECT COUNT(*) FROM strategic_relation_evidence WHERE relation_id = ?",
                (original.id,),
            ).fetchone()[0]

        self.assertEqual(relation_count, 1)
        self.assertEqual(evidence_count, 2)

    def test_stable_duplicate_normalizes_entity_key_case_and_whitespace(self) -> None:
        db.init_db()
        fixture = load_strategic_fixture(FIXTURE_PATH)
        original = fixture.relations[0]
        db.persist_strategic_fixture(fixture)
        duplicate = replace(
            original,
            id="normalized-stable-duplicate",
            subject_key=f"  {original.subject_key.lower()}  ",
            evidence_refs=(
                EvidenceRef(
                    source_type="manual_analysis",
                    source_id="normalized-proof",
                    insight_id="normalized-insight",
                ),
            ),
        )

        db.persist_strategic_fixture(
            replace(fixture, relations=(duplicate, *fixture.relations[1:]))
        )

        with db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id FROM strategic_relations
                WHERE subject_key = ? AND relation_type = ? AND object_key = ?
                """,
                (original.subject_key, original.relation_type, original.object_key),
            ).fetchall()
            evidence_count = conn.execute(
                "SELECT COUNT(*) FROM strategic_relation_evidence WHERE relation_id = ?",
                (original.id,),
            ).fetchone()[0]

        self.assertEqual(len(rows), 1)
        self.assertEqual(evidence_count, 2)

    def test_init_db_migrates_early_strategic_json_column_names(self) -> None:
        with db.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE strategic_relations (
                    id TEXT PRIMARY KEY,
                    subject_type TEXT NOT NULL,
                    subject_key TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    object_key TEXT NOT NULL,
                    condition_json TEXT NOT NULL DEFAULT '\"\"',
                    effect_json TEXT NOT NULL DEFAULT '\"\"',
                    concepts_json TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL,
                    provenance_type TEXT NOT NULL,
                    patch_sensitivity TEXT NOT NULL,
                    data_version TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO strategic_relations (
                    id, subject_type, subject_key, relation_type, object_type, object_key,
                    concepts_json, confidence, provenance_type, patch_sensitivity, data_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "early-row",
                    "champion",
                    "Caitlyn",
                    "creates",
                    "concept",
                    "access",
                    '["access"]',
                    0.7,
                    "manual_fixture",
                    "low",
                    "strategic-fixtures-v0",
                ),
            )
            conn.commit()

        db.init_db()

        with db.get_connection() as conn:
            migrated = conn.execute(
                "SELECT concepts FROM strategic_relations WHERE id = 'early-row'"
            ).fetchone()[0]

        self.assertEqual(migrated, '["access"]')

    def test_orphan_relation_provenance_is_rejected(self) -> None:
        db.init_db()

        with db.get_connection() as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO strategic_relation_evidence (
                        relation_id, source_type, source_id, insight_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    ("missing-relation", "manual_analysis", "missing", "missing"),
                )

    def test_sqlite_confidence_constraint_rejects_malformed_relation(self) -> None:
        db.init_db()

        with db.get_connection() as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO strategic_relations (
                        id, subject_type, subject_key, relation_type, object_type,
                        object_key, confidence, provenance_type, patch_sensitivity, data_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "bad-confidence",
                        "champion",
                        "Caitlyn",
                        "creates",
                        "concept",
                        "access",
                        1.1,
                        "manual_fixture",
                        "low",
                        "strategic-fixtures-v0",
                    ),
                )

    def test_hosted_schema_declares_equivalent_derived_tables_and_rls(self) -> None:
        schema = Path("supabase/schema.sql").read_text(encoding="utf-8")

        for table in STRATEGIC_TABLES:
            self.assertIn(f"public.{table}", schema)
            self.assertIn(f"alter table public.{table} enable row level security", schema)
        self.assertIn("strategic_relations_stable_key_idx", schema)
        self.assertIn("references public.strategic_relations(id) on delete cascade", schema)
        self.assertIn("condition_json jsonb not null default '\"\"'::jsonb", schema)


if __name__ == "__main__":
    unittest.main()
