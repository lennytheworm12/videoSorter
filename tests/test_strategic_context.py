from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import core.database as db
from core.strategic_types import load_strategic_fixture
from pipeline.relation_extract import EvidenceItem, ExtractionPacket, compile_candidates
from retrieval.strategic_context import _question_concepts, build_strategic_context


class StrategicContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_path = db.DB_PATH
        self.path = Path(self.tmp.name) / "strategic.db"
        db.DB_PATH = self.path
        db.init_db()
        db.persist_strategic_fixture(load_strategic_fixture("data/strategic_fixtures_v0.json"))

    def tearDown(self):
        db.DB_PATH = self.old_path
        self.tmp.cleanup()

    def test_empty_or_unknown_context_is_a_noop(self):
        context = build_strategic_context("unrelated question", ("Unknown",), db_paths=(str(self.path),))
        self.assertTrue(context.is_empty)
        self.assertTrue(build_strategic_context("x", ("Caitlyn",), db_paths=(str(self.path.with_name("missing.db")),)).is_empty)

    def test_known_entity_returns_provenanced_fingerprint_and_relations(self):
        context = build_strategic_context("Caitlyn access and conversion", ("Caitlyn",), db_paths=(str(self.path),))
        self.assertTrue(context.fingerprints)
        self.assertTrue(context.relations)
        self.assertTrue(all(row["evidence_refs"] for row in context.fingerprints + context.relations))
        self.assertTrue(any("access" in row["concepts"] for row in context.relations))

    def test_question_named_fixture_champion_does_not_require_external_entity_lookup(self):
        context = build_strategic_context(
            "Why does Caitlyn have persistent control against a mage?",
            db_paths=(str(self.path),),
        )

        self.assertTrue(any(row["champion"] == "Caitlyn" for row in context.fingerprints))
        self.assertTrue(any("Caitlyn" in (row["subject_key"], row["object_key"]) for row in context.relations))

    def test_question_entity_matching_accepts_punctuation_free_champion_aliases(self):
        context = build_strategic_context(
            "When is Kaisa's access conditional?",
            db_paths=(str(self.path),),
        )

        self.assertTrue(any(row["champion"] == "Kai'Sa" for row in context.fingerprints))
        self.assertTrue(any("access" in row["concepts"] for row in context.relations))

    def test_question_concepts_accept_underscore_and_hyphen_delimiters(self):
        concepts = set(_question_concepts("persistent_pressure and access-vs-continuity"))

        self.assertTrue({"persistent_pressure", "access", "continuity"}.issubset(concepts))

    def test_four_champion_lane_question_returns_all_represented_fingerprints(self):
        context = build_strategic_context(
            "How do Yunara and Thresh play into Tristana and Yuumi through access versus continuity?",
            db_paths=(str(self.path),),
        )

        self.assertEqual(
            {row["champion"] for row in context.fingerprints},
            {"Yunara", "Thresh", "Tristana"},
        )
        concepts = {concept for relation in context.relations for concept in relation["concepts"]}
        self.assertTrue({"access", "continuity"}.issubset(concepts))

    def test_confidence_and_version_filtering_are_applied(self):
        with db.get_connection() as conn:
            conn.execute("UPDATE strategic_relations SET confidence = 0.1 WHERE id = (SELECT id FROM strategic_relations LIMIT 1)")
            conn.execute("DELETE FROM champion_fingerprint_evidence WHERE champion = 'Caitlyn'")
            conn.execute("UPDATE champion_fingerprints SET data_version = 'stale' WHERE champion = 'Caitlyn'")
            conn.commit()
        context = build_strategic_context("Caitlyn access", ("Caitlyn",), db_paths=(str(self.path),), min_confidence=0.7)
        self.assertFalse(context.fingerprints)
        self.assertTrue(all(row["confidence"] >= 0.7 for row in context.relations))

    def test_relation_retrieval_filters_stale_ontology_versions(self):
        with db.get_connection() as conn:
            conn.execute("UPDATE strategic_relations SET ontology_version = 'stale' WHERE id = (SELECT id FROM strategic_relations LIMIT 1)")
            conn.commit()
        context = build_strategic_context("Caitlyn access", ("Caitlyn",), db_paths=(str(self.path),))
        self.assertTrue(all(row["ontology_version"] != "stale" for row in context.relations))

    def test_hop_and_relation_limits_bound_cycles(self):
        base = build_strategic_context("Yunara access", ("Yunara",), db_paths=(str(self.path),), max_hops=0, max_relations=1)
        expanded = build_strategic_context("Yunara access", ("Yunara",), db_paths=(str(self.path),), max_hops=2, max_relations=4)
        self.assertLessEqual(len(base.relations), 1)
        self.assertLessEqual(len(expanded.relations), 4)
        self.assertEqual(len({row["id"] for row in expanded.relations}), len(expanded.relations))

    def test_invalid_bounds_fail_deterministically(self):
        with self.assertRaises(ValueError):
            build_strategic_context("x", db_paths=(str(self.path),), max_hops=-1)
        with self.assertRaises(ValueError):
            build_strategic_context(None, db_paths=(str(self.path),))
        with self.assertRaises(ValueError):
            build_strategic_context("x", (1,), db_paths=(str(self.path),))
        for keyword, value in (("min_confidence", "0.7"), ("max_hops", 1.5), ("max_relations", 1.5)):
            with self.subTest(keyword=keyword):
                with self.assertRaises(ValueError):
                    build_strategic_context("x", db_paths=(str(self.path),), **{keyword: value})

    def test_zero_relation_limit_keeps_fingerprints_and_empty_paths_stay_empty(self):
        limited = build_strategic_context(
            "Caitlyn access", ("Caitlyn",), db_paths=(str(self.path),), max_relations=0
        )
        self.assertTrue(limited.fingerprints)
        self.assertFalse(limited.relations)
        self.assertTrue(build_strategic_context("Caitlyn access", ("Caitlyn",), db_paths=()).is_empty)

    def test_partial_or_malformed_storage_falls_back_to_empty_context(self):
        partial_path = Path(self.tmp.name) / "partial.db"
        with db.get_connection() as conn:
            conn.execute("UPDATE strategic_relations SET concepts = '\"access\"' WHERE id = (SELECT id FROM strategic_relations LIMIT 1)")
            conn.commit()
        malformed = build_strategic_context("Caitlyn access", ("Caitlyn",), db_paths=(str(self.path),))
        partial_path.touch()
        partial = build_strategic_context("Caitlyn access", ("Caitlyn",), db_paths=(str(partial_path),))
        self.assertTrue(malformed.is_empty)
        self.assertTrue(partial.is_empty)

    def test_malformed_database_is_atomic_when_followed_by_healthy_database(self):
        healthy_path = Path(self.tmp.name) / "healthy.db"
        previous_path = db.DB_PATH
        try:
            db.DB_PATH = healthy_path
            db.init_db()
            db.persist_strategic_fixture(load_strategic_fixture("data/strategic_fixtures_v0.json"))
        finally:
            db.DB_PATH = previous_path
        with db.get_connection() as conn:
            conn.execute("UPDATE compiled_principles SET concepts = '{\"access\": true}' WHERE id = (SELECT id FROM compiled_principles LIMIT 1)")
            conn.commit()

        healthy = build_strategic_context("Caitlyn access", ("Caitlyn",), db_paths=(str(healthy_path),))
        combined = build_strategic_context(
            "Caitlyn access", ("Caitlyn",), db_paths=(str(self.path), str(healthy_path))
        )
        self.assertEqual(combined, healthy)

    def test_relation_cap_is_global_across_databases(self):
        second_path = Path(self.tmp.name) / "second.db"
        previous_path = db.DB_PATH
        try:
            db.DB_PATH = second_path
            db.init_db()
            db.persist_strategic_fixture(load_strategic_fixture("data/strategic_fixtures_v0.json"))
        finally:
            db.DB_PATH = previous_path
        context = build_strategic_context(
            "Caitlyn access and conversion",
            ("Caitlyn",),
            db_paths=(str(self.path), str(second_path)),
            max_relations=1,
        )
        self.assertLessEqual(len(context.relations), 1)

    def test_accepted_automated_relation_uses_existing_context_retrieval(self):
        packet = ExtractionPacket(
            evidence=(EvidenceItem("source-1", "video-1", "Thresh Flay denies continued contact while available.", confidence=0.9),),
            ability_aliases={"Flay": "Thresh E"},
        )
        candidate = {
            "subject": "Flay", "subject_type": "ability", "relation_type": "denies",
            "object": "continued contact", "object_type": "concept",
            "condition": "while available", "effect": None,
            "concepts": ["continuity"], "provenance_type": "source_claim",
            "evidence_ids": ["source-1"], "extraction_confidence": 0.95,
            "patch_sensitivity": "low",
            "grounding": {
                "subject": {"source_text": "Flay", "evidence_id": "source-1"},
                "predicate": {"source_text": "denies", "evidence_id": "source-1"},
                "object": {"source_text": "continued contact", "evidence_id": "source-1"},
                "condition": {"source_text": "while available", "evidence_id": "source-1"},
            },
        }
        decision = compile_candidates(packet, [candidate])[0]
        db.persist_strategic_relations((decision,))

        context = build_strategic_context("How does Thresh deny continuity?", ("Thresh",), db_paths=(str(self.path),))

        self.assertTrue(any(row["id"] == decision.relation.id for row in context.relations))


if __name__ == "__main__":
    unittest.main()
