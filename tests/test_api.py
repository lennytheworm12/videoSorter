import os
import unittest
from unittest import mock

from fastapi import HTTPException
from starlette.requests import Request

import api.main as api_main
from api.main import (
    _acquire_query_slot,
    _QUERY_COUNT_BY_DAY_AND_IP,
    _backend_label,
    _backend_quality,
    _daily_query_limit,
    _enforce_daily_query_limit,
    _retrieval_mode,
    _split_answer_sources,
    _validate_runtime_config,
    health,
    query,
    QueryRequest,
)


class ApiTests(unittest.TestCase):
    def test_split_answer_sources_returns_body_and_source_lines(self) -> None:
        body, sources = _split_answer_sources(
            "Answer text\n\n---\nSources by section:\n  Core:\n    [0.1] row"
        )

        self.assertEqual(body, "Answer text")
        self.assertEqual(sources[0], "Sources by section:")
        self.assertIn("[0.1] row", sources[-1])

    def test_split_answer_sources_handles_answer_without_sources(self) -> None:
        body, sources = _split_answer_sources("Answer only")

        self.assertEqual(body, "Answer only")
        self.assertEqual(sources, [])

    def test_split_answer_sources_handles_single_newline_sources_marker(self) -> None:
        body, sources = _split_answer_sources(
            "Answer text\n---\nSources for Illaoi:\n  [0.1] row"
        )

        self.assertEqual(body, "Answer text")
        self.assertEqual(sources[0], "Sources for Illaoi:")
        self.assertEqual(sources[-1], "  [0.1] row")

    def test_validate_runtime_config_requires_supabase_auth_vars_when_auth_enabled(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "REQUIRE_AUTH": "true",
                "VECTOR_BACKEND": "sqlite",
                "GOOGLE_API_KEY": "x",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "SUPABASE_URL"):
                _validate_runtime_config()

    def test_validate_runtime_config_requires_supabase_db_for_vector_backend(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "REQUIRE_AUTH": "false",
                "VECTOR_BACKEND": "supabase",
                "GOOGLE_API_KEY": "x",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "SUPABASE_DATABASE_URL"):
                _validate_runtime_config()

    def test_validate_runtime_config_requires_google_api_key(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "REQUIRE_AUTH": "false",
                "VECTOR_BACKEND": "sqlite",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "GOOGLE_API_KEY"):
                _validate_runtime_config()

    def test_validate_runtime_config_requires_hf_token_for_remote_embeddings(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "REQUIRE_AUTH": "false",
                "VECTOR_BACKEND": "supabase",
                "EMBEDDING_BACKEND": "hf_remote",
                "SUPABASE_DATABASE_URL": "postgresql://x",
                "GOOGLE_API_KEY": "x",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                _validate_runtime_config()

    def test_validate_runtime_config_does_not_require_hf_token_for_bm25_only(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "REQUIRE_AUTH": "false",
                "VECTOR_BACKEND": "supabase",
                "EMBEDDING_BACKEND": "bm25_only",
                "SUPABASE_DATABASE_URL": "postgresql://x",
                "GOOGLE_API_KEY": "x",
            },
            clear=True,
        ):
            _validate_runtime_config()

    def test_daily_query_limit_defaults_to_100(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_daily_query_limit(), 100)

    def test_backend_metadata_helpers_use_env_overrides(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "BACKEND_LABEL": "Home backend",
                "BACKEND_QUALITY": "strong",
                "RETRIEVAL_MODE": "semantic-remote",
            },
            clear=True,
        ):
            self.assertEqual(_backend_label(), "Home backend")
            self.assertEqual(_backend_quality(), "strong")
            self.assertEqual(_retrieval_mode(), "semantic-remote")

    def test_health_includes_backend_metadata(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "BACKEND_LABEL": "Render fallback",
                "BACKEND_QUALITY": "fallback",
                "VECTOR_BACKEND": "supabase",
                "EMBEDDING_BACKEND": "bm25_only",
                "REQUIRE_AUTH": "false",
                "DAILY_QUERY_LIMIT": "100",
            },
            clear=True,
        ):
            payload = health()

        self.assertEqual(payload["backend_label"], "Render fallback")
        self.assertEqual(payload["backend_quality"], "fallback")
        self.assertEqual(payload["retrieval_mode"], "bm25-fallback")
        self.assertFalse(payload["semantic_enabled"])
        self.assertEqual(payload["embedding_backend"], "bm25_only")
        self.assertIn("active_queries", payload)

    def test_query_includes_backend_metadata(self) -> None:
        scope = {
            "type": "http",
            "method": "POST",
            "headers": [],
            "client": ("203.0.113.10", 1234),
        }
        request = Request(scope)
        _QUERY_COUNT_BY_DAY_AND_IP.clear()
        with mock.patch.dict(
            os.environ,
            {
                "BACKEND_LABEL": "Home backend",
                "BACKEND_QUALITY": "strong",
                "RETRIEVAL_MODE": "semantic-remote",
                "DAILY_QUERY_LIMIT": "100",
            },
            clear=False,
        ), mock.patch("api.main.normalize", return_value={"normalized": "Aatrox into Darius", "role": "top", "reasoning": "normalized"}), mock.patch(
            "api.main.rag_answer",
            return_value="Answer text\n\n---\nSources:\n- one",
        ), mock.patch(
            "api.main.current_retrieval_mode",
            return_value="bm25-fallback",
        ):
            response = query(QueryRequest(question="raw question"), request, {"id": "test"})

        self.assertEqual(response.answer, "Answer text")
        self.assertEqual(response.metadata["backend_label"], "Home backend")
        self.assertEqual(response.metadata["backend_quality"], "strong")
        self.assertEqual(response.metadata["retrieval_mode"], "bm25-fallback")
        self.assertFalse(response.metadata["semantic_enabled"])

    def test_enforce_daily_query_limit_blocks_after_limit(self) -> None:
        scope = {
            "type": "http",
            "method": "POST",
            "headers": [],
            "client": ("203.0.113.10", 1234),
        }
        request = Request(scope)
        _QUERY_COUNT_BY_DAY_AND_IP.clear()
        with mock.patch.dict(os.environ, {"DAILY_QUERY_LIMIT": "2"}, clear=False):
            _enforce_daily_query_limit(request)
            _enforce_daily_query_limit(request)
            with self.assertRaises(HTTPException) as ctx:
                _enforce_daily_query_limit(request)

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIn("2 per day", str(ctx.exception.detail))

    def test_acquire_query_slot_rejects_when_active_and_queue_are_full(self) -> None:
        old_active = api_main._ACTIVE_QUERY_COUNT
        old_waiting = api_main._WAITING_QUERY_COUNT
        api_main._ACTIVE_QUERY_COUNT = 1
        api_main._WAITING_QUERY_COUNT = 0
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "MAX_ACTIVE_QUERIES": "1",
                    "MAX_QUEUED_QUERIES": "0",
                },
                clear=False,
            ):
                with self.assertRaises(HTTPException) as ctx:
                    with _acquire_query_slot():
                        pass
        finally:
            api_main._ACTIVE_QUERY_COUNT = old_active
            api_main._WAITING_QUERY_COUNT = old_waiting

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("saturated", str(ctx.exception.detail))


if __name__ == "__main__":
    unittest.main()
