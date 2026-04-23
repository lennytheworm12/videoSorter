import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

import core.database as db
from core.database import get_connection, init_db, insert_insight, insert_video
import pipeline.embed as embed
import pipeline.score_clusters as score_clusters
import retrieval.query as retrieval_query


class ScoreClustersTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = pathlib.Path(self._tmpdir.name) / "score_test.db"
        db.DB_PATH = self._db_path
        init_db()
        with get_connection() as conn:
            conn.execute("ALTER TABLE insights ADD COLUMN embedding BLOB")
            conn.commit()
        self._old_embed_dbs = None if embed._ALL_DBS is None else list(embed._ALL_DBS)

    def tearDown(self) -> None:
        embed._ALL_DBS = self._old_embed_dbs
        self._tmpdir.cleanup()

    def _set_embedding(self, insight_id: int, values: list[float], source_score: float = 0.5) -> None:
        with get_connection() as conn:
            conn.execute(
                "UPDATE insights SET embedding = ?, source_score = ? WHERE id = ?",
                (np.array(values, dtype=np.float32).tobytes(), source_score, insight_id),
            )
            conn.commit()

    def test_compute_cluster_scores_ignores_duplicates_and_rewards_cross_video_recurrence(self) -> None:
        for video_id in ("v1", "v2", "v3"):
            insert_video(
                video_id=video_id,
                video_url=f"https://example.com/{video_id}",
                video_title=video_id,
                description="",
                game="aoe2",
                role="general",
                subject=None,
                champion=None,
                message_timestamp="2026-04-22T00:00:00+00:00",
                source="aoe2_video",
            )

        rep_id = insert_insight("v1", "economy_macro", "Keep your economy balanced.", subject=None, subject_type="general")
        dupe_id = insert_insight("v1", "economy_macro", "Keep your economy balanced.", subject=None, subject_type="general")
        peer_id = insert_insight("v2", "economy_macro", "Keep your economy balanced.", subject=None, subject_type="general")
        iso_id = insert_insight("v3", "scouting", "Scout earlier to identify pressure.", subject=None, subject_type="general")

        self._set_embedding(rep_id, [1.0, 0.0, 0.0], source_score=0.5)
        self._set_embedding(dupe_id, [1.0, 0.0, 0.0], source_score=0.4)
        self._set_embedding(peer_id, [1.0, 0.0, 0.0], source_score=0.5)
        self._set_embedding(iso_id, [0.0, 1.0, 0.0], source_score=0.5)

        with get_connection() as conn:
            conn.execute("UPDATE insights SET is_duplicate = 1 WHERE id = ?", (dupe_id,))
            conn.commit()

        score_clusters.compute_cluster_scores()

        with get_connection() as conn:
            rows = conn.execute(
                "SELECT id, cluster_score, confidence FROM insights WHERE id IN (?, ?, ?, ?) ORDER BY id",
                (rep_id, dupe_id, peer_id, iso_id),
            ).fetchall()

        by_id = {row["id"]: row for row in rows}
        self.assertGreater(by_id[rep_id]["cluster_score"], by_id[iso_id]["cluster_score"])
        self.assertGreater(by_id[rep_id]["confidence"], by_id[iso_id]["confidence"])
        self.assertIsNone(by_id[dupe_id]["cluster_score"])
        self.assertIsNone(by_id[dupe_id]["confidence"])

    def test_compute_cluster_scores_does_not_mix_games_within_same_db(self) -> None:
        insert_video(
            video_id="lol_video",
            video_url="https://example.com/lol_video",
            video_title="LoL",
            description="",
            game="lol",
            role="mid",
            subject="Ahri",
            champion="Ahri",
            message_timestamp="2026-04-22T00:00:00+00:00",
            source="youtube_guide",
        )
        insert_video(
            video_id="aoe_video",
            video_url="https://example.com/aoe_video",
            video_title="AoE2",
            description="",
            game="aoe2",
            role="general",
            subject=None,
            champion=None,
            message_timestamp="2026-04-22T00:00:00+00:00",
            source="aoe2_video",
        )

        lol_id = insert_insight(
            "lol_video",
            "principles",
            "Take short trades when you have priority.",
            subject="Ahri",
            subject_type="champion",
        )
        aoe_id = insert_insight(
            "aoe_video",
            "principles",
            "Take short trades when you have priority.",
            subject=None,
            subject_type="general",
        )

        shared_vec = [1.0, 0.0, 0.0]
        self._set_embedding(lol_id, shared_vec, source_score=0.5)
        self._set_embedding(aoe_id, shared_vec, source_score=0.5)

        score_clusters.compute_cluster_scores()

        with get_connection() as conn:
            rows = conn.execute(
                "SELECT id, cluster_score, confidence FROM insights WHERE id IN (?, ?) ORDER BY id",
                (lol_id, aoe_id),
            ).fetchall()

        for row in rows:
            self.assertEqual(row["cluster_score"], 0.0)
            self.assertEqual(row["confidence"], 0.3)

    def test_score_all_databases_updates_each_db_independently(self) -> None:
        videos_db = pathlib.Path(self._tmpdir.name) / "videos.db"
        guides_db = pathlib.Path(self._tmpdir.name) / "knowledge.db"

        for target_db in (videos_db, guides_db):
            db.DB_PATH = target_db
            init_db()
            with get_connection() as conn:
                conn.execute("ALTER TABLE insights ADD COLUMN embedding BLOB")
                conn.commit()
            for video_id in ("a", "b"):
                insert_video(
                    video_id=f"{target_db.stem}_{video_id}",
                    video_url=f"https://example.com/{target_db.stem}_{video_id}",
                    video_title=video_id,
                    description="",
                    game="aoe2",
                    role="general",
                    subject=None,
                    champion=None,
                    message_timestamp="2026-04-22T00:00:00+00:00",
                    source="aoe2_video",
                )
            i1 = insert_insight(f"{target_db.stem}_a", "principles", "Protect your economy.", subject=None, subject_type="general")
            i2 = insert_insight(f"{target_db.stem}_b", "principles", "Scout the map.", subject=None, subject_type="general")
            with get_connection() as conn:
                conn.execute(
                    "UPDATE insights SET embedding = ?, source_score = ? WHERE id = ?",
                    (np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes(), 0.5, i1),
                )
                conn.execute(
                    "UPDATE insights SET embedding = ?, source_score = ? WHERE id = ?",
                    (np.array([0.0, 1.0, 0.0], dtype=np.float32).tobytes(), 0.5, i2),
                )
                conn.commit()

        with mock.patch.object(score_clusters, "_ALL_DBS", [str(videos_db), str(guides_db)]):
            score_clusters.score_all_databases()

        for target_db in (videos_db, guides_db):
            db.DB_PATH = target_db
            with get_connection() as conn:
                values = conn.execute(
                    "SELECT cluster_score, confidence FROM insights ORDER BY id"
                ).fetchall()
            self.assertTrue(all(row["cluster_score"] == 0.0 for row in values))
            self.assertTrue(all(row["confidence"] == 0.3 for row in values))

    def test_retrieve_prefers_higher_confidence_when_similarity_matches(self) -> None:
        embed._ALL_DBS = [str(self._db_path)]

        for video_id in ("high", "low"):
            insert_video(
                video_id=video_id,
                video_url=f"https://example.com/{video_id}",
                video_title=video_id,
                description="",
                game="aoe2",
                role="general",
                subject=None,
                champion=None,
                message_timestamp="2026-04-22T00:00:00+00:00",
                source="aoe2_video",
            )

        high_id = insert_insight("high", "economy_macro", "Economy balance matters.", subject=None, subject_type="general")
        low_id = insert_insight("low", "economy_macro", "Economy balance matters.", subject=None, subject_type="general")

        with get_connection() as conn:
            blob = np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes()
            conn.execute(
                "UPDATE insights SET embedding = ?, source_score = ?, confidence = ? WHERE id = ?",
                (blob, 0.5, 0.9, high_id),
            )
            conn.execute(
                "UPDATE insights SET embedding = ?, source_score = ?, confidence = ? WHERE id = ?",
                (blob, 0.5, 0.3, low_id),
            )
            conn.commit()

        class FakeModel:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def encode(self, text, convert_to_numpy=True, normalize_embeddings=True):
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)

        with mock.patch.object(retrieval_query, "SentenceTransformer", FakeModel):
            results = retrieval_query.retrieve(
                "economy balance matters",
                game="aoe2",
                subject=None,
                top_k=2,
            )

        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0]["confidence"], results[1]["confidence"])


if __name__ == "__main__":
    unittest.main()
