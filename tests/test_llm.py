import json
import importlib
import os
import unittest
from unittest import mock

import core.llm as llm
from core.llm import _deepseek_generate


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")


class DeepSeekGenerateTests(unittest.TestCase):
    def test_builds_openai_compatible_chat_request(self):
        response = _Response({"choices": [{"message": {"content": "  answer  "}}]})
        with mock.patch("core.llm.urlrequest.urlopen", return_value=response) as urlopen:
            answer = _deepseek_generate(
                "secret",
                "https://api.deepseek.com/",
                "deepseek-v4-flash",
                "system text",
                "user text",
                0.2,
                64,
                12,
            )

        self.assertEqual(answer, "answer")
        req = urlopen.call_args.args[0]
        self.assertEqual(req.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(req.get_header("Authorization"), "Bearer secret")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 12)
        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertEqual(payload["messages"], [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "user text"},
        ])
        self.assertFalse(payload["stream"])

    def test_rejects_missing_or_empty_content(self):
        for payload in ({}, {"choices": []}, {"choices": [{"message": {"content": " "}}]}, b"not-json"):
            with self.subTest(payload=payload), mock.patch(
                "core.llm.urlrequest.urlopen",
                return_value=_Response(payload),
            ):
                with self.assertRaisesRegex(RuntimeError, "DeepSeek"):
                    _deepseek_generate(
                        "secret",
                        "https://api.deepseek.com",
                        "deepseek-v4-flash",
                        "system",
                        "user",
                        0.1,
                        None,
                        12,
                    )

    def test_sends_explicit_thinking_mode_when_requested(self):
        response = _Response({"choices": [{"message": {"content": "answer"}}]})
        with mock.patch("core.llm.urlrequest.urlopen", return_value=response) as urlopen:
            _deepseek_generate("secret", "https://api.deepseek.com", "deepseek-v4-flash", "system", "user", 0.2, 64, 12, "disabled")

        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_rejects_invalid_thinking_mode(self):
        with self.assertRaisesRegex(ValueError, "thinking"):
            _deepseek_generate("secret", "https://api.deepseek.com", "deepseek-v4-flash", "system", "user", 0.2, 64, 12, "maybe")


class ProviderSelectionTests(unittest.TestCase):
    def tearDown(self):
        importlib.reload(llm)

    def test_deepseek_key_selects_deepseek_without_overriding_explicit_model(self):
        with mock.patch.dict(
            os.environ,
            {
                "GOOGLE_API_KEY": "",
                "GOOGLE_API_KEY_TWO": "",
                "GOOGLE_CLOUD_API_KEY": "",
                "LLM_MODEL": "",
                "LLM_PROVIDER": "",
                "DEEPSEEK_API_KEY": "test-key",
                "DEEPSEEK_MODEL": "deepseek-v4-flash",
            },
            clear=True,
        ):
            configured = importlib.reload(llm)

        self.assertEqual(configured.BACKEND, "deepseek")
        self.assertEqual(configured.MODEL, "deepseek-v4-flash")

    def test_explicit_deepseek_overrides_google_auto_priority(self):
        with mock.patch.dict(
            os.environ,
            {
                "GOOGLE_API_KEY": "google-key",
                "DEEPSEEK_API_KEY": "deepseek-key",
                "DEEPSEEK_MODEL": "deepseek-v4-flash",
                "LLM_MODEL": "",
                "LLM_PROVIDER": "deepseek",
            },
            clear=True,
        ):
            configured = importlib.reload(llm)

        self.assertEqual(configured.BACKEND, "deepseek")
        self.assertEqual(configured.MODEL, "deepseek-v4-flash")

    def test_explicit_provider_requires_its_credential(self):
        with mock.patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "",
                "LLM_MODEL": "",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires DEEPSEEK_API_KEY"):
                importlib.reload(llm)

    def test_no_cloud_keys_falls_back_to_ollama(self):
        with mock.patch.dict(
            os.environ,
            {
                "GOOGLE_API_KEY": "",
                "GOOGLE_API_KEY_TWO": "",
                "GOOGLE_CLOUD_API_KEY": "",
                "DEEPSEEK_API_KEY": "",
                "LLM_MODEL": "",
                "LLM_PROVIDER": "",
            },
            clear=True,
        ):
            configured = importlib.reload(llm)

        self.assertEqual(configured.BACKEND, "ollama")


if __name__ == "__main__":
    unittest.main()
