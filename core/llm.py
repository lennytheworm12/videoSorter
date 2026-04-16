"""
LLM abstraction layer.

Uses Google Gemini when GOOGLE_API_KEY is set, falls back to Ollama for local dev.

Environment variables:
    GOOGLE_API_KEY  — enables Gemini (set this in .env)
    LLM_MODEL       — model to use
                      Gemini default : gemini-2.5-flash-preview-04-17
                      Ollama default : gemma4:e2b
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

_GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
_LLM_MODEL = os.environ.get("LLM_MODEL")

# Suppress noisy SDK warnings
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# ── backend selection ──────────────────────────────────────────────────────────

if _GOOGLE_API_KEY:
    from google import genai as _genai
    from google.genai import types as _gtypes

    _DEFAULT_MODEL = _LLM_MODEL or "gemini-2.5-flash"
    _client = _genai.Client(api_key=_GOOGLE_API_KEY)

    def chat(
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> str:
        response = _client.models.generate_content(
            model=model or _DEFAULT_MODEL,
            contents=user,
            config=_gtypes.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text.strip()

    BACKEND = "gemini"
    MODEL = _DEFAULT_MODEL

else:
    import ollama as _ollama

    _DEFAULT_MODEL = _LLM_MODEL or "gemma4:e2b"

    def chat(
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> str:
        response = _ollama.chat(
            model=model or _DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={
                "temperature": temperature,
                "num_ctx": 16384,
                "num_predict": max_tokens,
            },
        )
        msg = response["message"] if isinstance(response, dict) else response.message
        return (msg["content"] if isinstance(msg, dict) else msg.content).strip()

    BACKEND = "ollama"
    MODEL = _DEFAULT_MODEL
