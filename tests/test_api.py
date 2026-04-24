import os
import unittest
from unittest import mock

from fastapi import HTTPException
from starlette.requests import Request

from api.main import (
    _QUERY_COUNT_BY_DAY_AND_IP,
    _daily_query_limit,
    _enforce_daily_query_limit,
    _split_answer_sources,
    _validate_runtime_config,
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

    def test_daily_query_limit_defaults_to_100(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_daily_query_limit(), 100)

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


if __name__ == "__main__":
    unittest.main()
