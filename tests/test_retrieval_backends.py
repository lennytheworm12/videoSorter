import os
import unittest
from unittest import mock

import numpy as np

import retrieval.query as retrieval_query


class RetrievalBackendTests(unittest.TestCase):
    def test_supabase_retrieve_uses_remote_embeddings_when_configured(self) -> None:
        fake_row = {
            "text": "Use your lead to pressure objectives.",
            "insight_type": "macro_advice",
            "role": "top",
            "subject": "Aatrox",
            "subject_type": "champion",
            "champion": "Aatrox",
            "game": "lol",
            "rank": None,
            "website_rating": None,
            "source": "youtube_guide",
            "source_score": 0.6,
            "confidence": 0.7,
            "score": 0.8,
        }
        fake_client = mock.Mock()
        fake_client.feature_extraction.return_value = [0.0, 1.0, 0.0]

        with mock.patch.dict(
            os.environ,
            {
                "VECTOR_BACKEND": "supabase",
                "EMBEDDING_BACKEND": "hf_remote",
                "HF_TOKEN": "hf_x",
            },
            clear=False,
        ), mock.patch.object(retrieval_query.vector_store, "enabled", return_value=True), mock.patch.object(
            retrieval_query, "_get_hf_embed_client", return_value=fake_client
        ), mock.patch.object(
            retrieval_query.vector_store, "search_insights", return_value=[fake_row]
        ):
            results = retrieval_query.retrieve("How do I snowball top?", champion="Aatrox", game="lol", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["retrieval_layer"], "supabase")
        self.assertEqual(retrieval_query.current_retrieval_mode(), "semantic-hf-remote")

    def test_supabase_retrieve_falls_back_to_lexical_when_remote_embedding_fails(self) -> None:
        lexical_row = {
            "text": "Play around your wave and punish cooldowns.",
            "insight_type": "laning_tips",
            "role": "top",
            "subject": "Aatrox",
            "subject_type": "champion",
            "champion": "Aatrox",
            "game": "lol",
            "rank": None,
            "website_rating": None,
            "source": "youtube_guide",
            "source_score": 0.55,
            "confidence": 0.65,
            "lexical_score": 0.9,
        }

        with mock.patch.dict(
            os.environ,
            {
                "VECTOR_BACKEND": "supabase",
                "EMBEDDING_BACKEND": "hf_remote",
                "HF_TOKEN": "hf_x",
            },
            clear=False,
        ), mock.patch.object(retrieval_query.vector_store, "enabled", return_value=True), mock.patch.object(
            retrieval_query, "_get_hf_embed_client", side_effect=RuntimeError("HF down")
        ), mock.patch.object(
            retrieval_query.vector_store, "search_keyword_candidates", return_value=[lexical_row]
        ):
            results = retrieval_query.retrieve("How do I lane as Aatrox?", champion="Aatrox", game="lol", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["retrieval_layer"], "supabase-lexical")
        self.assertEqual(retrieval_query.current_retrieval_mode(), "bm25-fallback")

    def test_normalize_query_vector_reduces_matrix_to_unit_vector(self) -> None:
        vector = retrieval_query._normalize_query_vector(np.array([[3.0, 4.0]], dtype=np.float32))
        self.assertEqual(vector.shape, (2,))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_lol_overview_answer_bypasses_retrieval(self) -> None:
        with mock.patch.object(retrieval_query, "retrieve") as retrieve:
            answer = retrieval_query.answer("what is League of Legends", game="lol", show_sources=True)

        retrieve.assert_not_called()
        self.assertIn("five-versus-five multiplayer strategy game", answer)
        self.assertNotIn("Sources:", answer)

    def test_sources_block_contains_rows_only_for_wrapping_headers(self) -> None:
        block = retrieval_query._sources_block(
            [
                {
                    "score": 0.42,
                    "confidence": 0.77,
                    "source_weight": 1.6,
                    "retrieval_layer": "supabase",
                    "source": "discord",
                    "insight_type": "principles",
                    "text": "Identify your win condition before forcing a fight.",
                }
            ]
        )

        self.assertTrue(block.startswith("[0.42 | conf 0.77"))
        self.assertIn("(principles)", block)
        self.assertNotIn("Sources:", block)


if __name__ == "__main__":
    unittest.main()
