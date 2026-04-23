import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

import core.database as db
from core.database import init_db, insert_insight, insert_video
import pipeline.aoe2_crossref as aoe2_crossref


class Aoe2CrossrefTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db.DB_PATH = pathlib.Path(self._tmpdir.name) / "aoe2_crossref.db"
        init_db()
        with db.get_connection() as conn:
            conn.execute("ALTER TABLE insights ADD COLUMN embedding BLOB")
            conn.commit()
        aoe2_crossref._init_tables()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _set_embedding(self, insight_id: int, values: list[float]) -> None:
        with db.get_connection() as conn:
            conn.execute(
                "UPDATE insights SET embedding = ?, confidence = ?, source_score = ? WHERE id = ?",
                (np.array(values, dtype=np.float32).tobytes(), 0.8, 0.7, insight_id),
            )
            conn.commit()

    def test_build_civilization_vectors_uses_only_civ_specific_aoe2_rows(self) -> None:
        insert_video(
            video_id="franks_1",
            video_url="https://example.com/franks",
            video_title="Franks guide",
            description="",
            game="aoe2",
            role="general",
            subject="Franks",
            champion=None,
            message_timestamp="2026-04-22T00:00:00+00:00",
            source="aoe2_video",
        )
        insert_video(
            video_id="general_1",
            video_url="https://example.com/general",
            video_title="General guide",
            description="",
            game="aoe2",
            role="general",
            subject=None,
            champion=None,
            message_timestamp="2026-04-22T00:00:00+00:00",
            source="aoe2_video",
        )
        insert_video(
            video_id="lol_1",
            video_url="https://example.com/lol",
            video_title="LoL guide",
            description="",
            game="lol",
            role="mid",
            subject="Ahri",
            champion="Ahri",
            message_timestamp="2026-04-22T00:00:00+00:00",
            source="youtube_guide",
        )

        franks_id = insert_insight(
            "franks_1",
            "civilization_identity",
            "Franks want to leverage cavalry momentum and strong eco timings.",
            subject="Franks",
            subject_type="civ",
        )
        general_id = insert_insight(
            "general_1",
            "principles",
            "Keep villager production constant.",
            subject=None,
            subject_type="general",
        )
        lol_id = insert_insight(
            "lol_1",
            "champion_identity",
            "Ahri wants short trades.",
            subject="Ahri",
            subject_type="champion",
        )

        self._set_embedding(franks_id, [1.0, 0.0, 0.0])
        self._set_embedding(general_id, [0.0, 1.0, 0.0])
        self._set_embedding(lol_id, [0.0, 0.0, 1.0])

        vectors = aoe2_crossref.build_civilization_vectors()

        self.assertEqual(list(vectors.keys()), ["Franks"])
        self.assertAlmostEqual(float(np.linalg.norm(vectors["Franks"])), 1.0, places=4)

    def test_get_applicable_insights_prefers_transferable_similar_tagged_rows(self) -> None:
        insert_video(
            video_id="franks_video",
            video_url="https://example.com/franks",
            video_title="Franks guide",
            description="",
            game="aoe2",
            role="general",
            subject="Franks",
            champion=None,
            message_timestamp="2026-04-22T00:00:00+00:00",
            source="aoe2_video",
        )
        insert_video(
            video_id="huns_video",
            video_url="https://example.com/huns",
            video_title="Huns guide",
            description="",
            game="aoe2",
            role="general",
            subject="Huns",
            champion=None,
            message_timestamp="2026-04-22T00:00:00+00:00",
            source="aoe2_video",
        )
        insert_video(
            video_id="teutons_video",
            video_url="https://example.com/teutons",
            video_title="Teutons guide",
            description="",
            game="aoe2",
            role="general",
            subject="Teutons",
            champion=None,
            message_timestamp="2026-04-22T00:00:00+00:00",
            source="aoe2_video",
        )

        franks_id = insert_insight(
            "franks_video",
            "civilization_identity",
            "Franks want cavalry openings into knight pressure.",
            subject="Franks",
            subject_type="civ",
        )
        huns_id = insert_insight(
            "huns_video",
            "matchup_advice",
            "Against archer openings, lean on early cavalry pressure before overcommitting to a boom.",
            subject="Huns",
            subject_type="civ",
        )
        teutons_id = insert_insight(
            "teutons_video",
            "matchup_advice",
            "Use infantry and monk support to stabilize slow openings.",
            subject="Teutons",
            subject_type="civ",
        )

        self._set_embedding(franks_id, [1.0, 0.0, 0.0])
        self._set_embedding(huns_id, [0.95, 0.05, 0.0])
        self._set_embedding(teutons_id, [0.2, 0.9, 0.0])

        aoe2_crossref.build_civilization_vectors()
        with db.get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO aoe2_crossref_insights (insight_id, subject, scope, situation_tags)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (huns_id, "Huns", "transferable", json.dumps(["feudal_pressure", "cavalry"])),
                    (teutons_id, "Teutons", "transferable", json.dumps(["defense", "infantry"])),
                    (franks_id, "Franks", "transferable", json.dumps(["cavalry"])),
                ],
            )
            conn.commit()

        hits = aoe2_crossref.get_applicable_insights(
            "Franks",
            preferred_types=["matchup_advice"],
            situation_tags=["feudal_pressure", "cavalry"],
            top_k=5,
        )

        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0]["source_subject"], "Huns")
        self.assertEqual(hits[0]["retrieval_layer"], "aoe2_crossref")
        self.assertNotIn("Franks", [hit["source_subject"] for hit in hits])

    def test_label_applicable_insights_skips_previously_labeled_rows(self) -> None:
        insert_video(
            video_id="magyars_video",
            video_url="https://example.com/magyars",
            video_title="Magyars guide",
            description="",
            game="aoe2",
            role="general",
            subject="Magyars",
            champion=None,
            message_timestamp="2026-04-22T00:00:00+00:00",
            source="aoe2_video",
        )
        first_id = insert_insight(
            "magyars_video",
            "civilization_identity",
            "Magyars want to convert early military tempo into map pressure.",
            subject="Magyars",
            subject_type="civ",
        )
        second_id = insert_insight(
            "magyars_video",
            "matchup_advice",
            "Use cavalry mobility to punish exposed archer numbers.",
            subject="Magyars",
            subject_type="civ",
        )

        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO aoe2_crossref_insights (insight_id, subject, scope, situation_tags)
                VALUES (?, ?, ?, ?)
                """,
                (first_id, "Magyars", "specific", json.dumps(["cavalry"])),
            )
            conn.commit()

        llm_responses = [
            json.dumps({"scope": "transferable", "situation_tags": ["cavalry", "feudal_pressure"]}),
        ]
        with mock.patch("pipeline.aoe2_crossref.llm_chat", side_effect=llm_responses) as mocked_llm:
            aoe2_crossref.label_applicable_insights(dry_run=False)

        self.assertEqual(mocked_llm.call_count, 1)
        with db.get_connection() as conn:
            rows = conn.execute(
                "SELECT insight_id, scope, situation_tags FROM aoe2_crossref_insights ORDER BY insight_id"
            ).fetchall()

        self.assertEqual(len(rows), 2)
        by_id = {row["insight_id"]: row for row in rows}
        self.assertEqual(by_id[first_id]["scope"], "specific")
        self.assertEqual(by_id[second_id]["scope"], "transferable")
        self.assertEqual(json.loads(by_id[second_id]["situation_tags"]), ["cavalry", "feudal_pressure"])


if __name__ == "__main__":
    unittest.main()
