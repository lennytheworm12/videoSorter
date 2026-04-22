import pathlib
import tempfile
import unittest

import numpy as np

import core.database as db
from core.database import init_db, insert_insight, insert_video
import pipeline.embed as embed
from scrape.aoe2_import import import_rows


class Aoe2PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        db.DB_PATH = pathlib.Path(self._tmpdir.name) / "aoe2_test.db"
        embed._ALL_DBS = [str(db.DB_PATH)]
        init_db()
        with db.get_connection() as conn:
            conn.execute("ALTER TABLE insights ADD COLUMN embedding BLOB")
            conn.commit()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _set_embedding(self, insight_id: int, values: list[float]) -> None:
        with db.get_connection() as conn:
            conn.execute(
                "UPDATE insights SET embedding = ? WHERE id = ?",
                (np.array(values, dtype=np.float32).tobytes(), insight_id),
            )
            conn.commit()

    def test_aoe2_subject_filter_uses_insight_subject_not_video_subject(self) -> None:
        insert_video(
            video_id="franks_video",
            video_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
            video_title="Franks guide",
            description="",
            game="aoe2",
            role="general",
            subject="Franks",
            champion=None,
            message_timestamp="2026-04-21T00:00:00+00:00",
            source="aoe2_video",
        )

        civ_id = insert_insight(
            "franks_video",
            "civilization_identity",
            "Use the Franks farm and cavalry bonuses to hit stronger scout into knight timings.",
            subject="Franks",
            subject_type="civ",
        )
        general_id = insert_insight(
            "franks_video",
            "economy_macro",
            "Keep town center uptime clean before adding extra production.",
            subject=None,
            subject_type="general",
        )
        self._set_embedding(civ_id, [1.0, 0.0, 0.0])
        self._set_embedding(general_id, [0.0, 1.0, 0.0])

        _, civ_texts, civ_meta, _ = embed.load_all_vectors(game="aoe2", subject="Franks")
        self.assertEqual(civ_texts, [
            "Use the Franks farm and cavalry bonuses to hit stronger scout into knight timings."
        ])
        self.assertEqual(civ_meta[0]["subject"], "Franks")
        self.assertEqual(civ_meta[0]["subject_type"], "civ")

        _, general_texts, general_meta, _ = embed.load_all_vectors(game="aoe2", subject=None)
        self.assertEqual(general_texts, [
            "Keep town center uptime clean before adding extra production."
        ])
        self.assertIsNone(general_meta[0]["subject"])
        self.assertEqual(general_meta[0]["subject_type"], "general")

    def test_manual_import_marks_text_rows_transcribed(self) -> None:
        inserted, transcribed = import_rows([
            {
                "title": "Hera Franks guide",
                "video_url": "https://www.youtube.com/watch?v=bbbbbbbbbbb",
                "source": "aoe2_video",
                "civ": "Franks",
            },
            {
                "title": "Franks civilization page",
                "source": "aoe2_wiki",
                "civ": "frank",
                "content": "Franks have strong cavalry bonuses and faster berry gathering.",
            },
        ])
        self.assertEqual(inserted, 2)
        self.assertEqual(transcribed, 1)

        with db.get_connection() as conn:
            rows = conn.execute(
                "SELECT video_id, source, subject, status, transcription FROM videos ORDER BY video_id"
            ).fetchall()

        by_source = {row["source"]: row for row in rows}
        self.assertEqual(by_source["aoe2_video"]["subject"], "Franks")
        self.assertEqual(by_source["aoe2_video"]["status"], "pending")
        self.assertIsNone(by_source["aoe2_video"]["transcription"])

        self.assertEqual(by_source["aoe2_wiki"]["subject"], "Franks")
        self.assertEqual(by_source["aoe2_wiki"]["status"], "transcribed")
        self.assertIn("strong cavalry bonuses", by_source["aoe2_wiki"]["transcription"])


if __name__ == "__main__":
    unittest.main()
