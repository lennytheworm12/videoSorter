import pathlib
import tempfile
import unittest

import numpy as np

import core.database as db
from core.database import get_connection, init_db, insert_insight, insert_video
import pipeline.consolidate as consolidate


class ConsolidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_all_dbs = None if consolidate._ALL_DBS is None else list(consolidate._ALL_DBS)

    def tearDown(self) -> None:
        consolidate._ALL_DBS = self._old_all_dbs
        self._tmpdir.cleanup()

    def _seed_db(self, path: pathlib.Path) -> None:
        db.DB_PATH = path
        init_db()
        with get_connection() as conn:
            conn.execute("ALTER TABLE insights ADD COLUMN embedding BLOB")
            conn.commit()

        insert_video(
            video_id=f"{path.stem}_video",
            video_url=f"https://example.com/{path.stem}",
            video_title=path.stem,
            description="",
            game="aoe2",
            role="general",
            subject=None,
            champion=None,
            message_timestamp="2026-04-22T00:00:00+00:00",
            source="aoe2_video",
        )
        keep_id = insert_insight(
            f"{path.stem}_video",
            "principles",
            "Keep producing villagers.",
            subject=None,
            subject_type="general",
        )
        dupe_id = insert_insight(
            f"{path.stem}_video",
            "principles",
            "Keep producing villagers.",
            subject=None,
            subject_type="general",
        )
        blob = np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes()
        with get_connection() as conn:
            conn.execute(
                "UPDATE insights SET embedding = ?, source_score = ? WHERE id = ?",
                (blob, 0.8, keep_id),
            )
            conn.execute(
                "UPDATE insights SET embedding = ?, source_score = ? WHERE id = ?",
                (blob, 0.4, dupe_id),
            )
            conn.commit()

    def test_consolidate_all_databases_processes_each_db(self) -> None:
        videos_db = pathlib.Path(self._tmpdir.name) / "videos.db"
        knowledge_db = pathlib.Path(self._tmpdir.name) / "knowledge.db"
        for path in (videos_db, knowledge_db):
            self._seed_db(path)

        consolidate._ALL_DBS = [str(videos_db), str(knowledge_db)]
        consolidate.consolidate_all_databases()

        for path in (videos_db, knowledge_db):
            db.DB_PATH = path
            with get_connection() as conn:
                rows = conn.execute(
                    "SELECT repetition_count, is_duplicate FROM insights ORDER BY id"
                ).fetchall()
            self.assertEqual(rows[0]["repetition_count"], 2)
            self.assertEqual(rows[0]["is_duplicate"], 0)
            self.assertEqual(rows[1]["is_duplicate"], 1)


if __name__ == "__main__":
    unittest.main()
