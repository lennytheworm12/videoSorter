import json
from pathlib import Path
import tempfile
import unittest

import core.database as db
from core.strategic_types import EvidenceRef, StrategicRelation
from scripts.inspect_relations import _decode_relation, load_relations, load_review_decisions, render_relations


class RelationInspectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_path = db.DB_PATH
        self.path = Path(self.tmp.name) / "strategic.db"
        db.DB_PATH = self.path
        db.init_db()
        relation = StrategicRelation(
            id="automatic-thresh-e", subject_type="ability", subject_key="Thresh E",
            relation_type="denies", object_type="concept", object_key="continuity",
            condition="while available", effect=None, concepts=("continuity",),
            confidence=0.8, provenance_type="source_claim", patch_sensitivity="low",
            data_version="strategic-relations-v0",
            evidence_refs=(EvidenceRef("insight", "video-1", "42"),),
        )
        db.persist_strategic_relations((type("Decision", (), {"status": "accepted", "relation": relation})(),))

    def tearDown(self):
        db.DB_PATH = self.old_path
        self.tmp.cleanup()

    def test_filters_and_renders_provenance(self):
        relations = load_relations(self.path, champion="Thresh", evidence_id="42")
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0]["evidence_refs"][0]["source_id"], "video-1")
        self.assertIn("insight:video-1 insight=42", render_relations(relations))
        self.assertFalse(load_relations(self.path, concept="access"))

    def test_review_file_keeps_rejected_and_review_decisions_visible(self):
        path = Path(self.tmp.name) / "dry-run.json"
        path.write_text(json.dumps({"decisions": [{"status": "accepted"}, {"status": "review"}, {"status": "rejected"}]}), encoding="utf-8")
        self.assertEqual([item["status"] for item in load_review_decisions(path)], ["review", "rejected"])

    def test_legacy_relation_without_alignment_json_decodes_as_empty(self):
        decoded = _decode_relation({"condition_json": '\"\"', "effect_json": '\"\"', "concepts": "[]"})
        self.assertEqual(decoded["alignment"], [])


if __name__ == "__main__":
    unittest.main()
