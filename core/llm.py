"""
LLM abstraction layer.

Uses Google Gemini when GOOGLE_API_KEY is set, falls back to Ollama for local dev.
A second Gemini key (GOOGLE_CLOUD_API_KEY) is used as an automatic fallback when
the primary key hits its daily quota (429 RESOURCE_EXHAUSTED).

Environment variables:
    GOOGLE_API_KEY        — primary Gemini key (free tier or paid)
    GOOGLE_CLOUD_API_KEY  — fallback Gemini key (Google Cloud project key)
    LLM_MODEL             — model to use
                            Gemini default : gemini-2.0-flash
                            Ollama default : gemma4:e2b
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

_GOOGLE_API_KEY       = os.environ.get("GOOGLE_API_KEY")
_GOOGLE_CLOUD_API_KEY = os.environ.get("GOOGLE_CLOUD_API_KEY")
_LLM_MODEL            = os.environ.get("LLM_MODEL")

# Suppress noisy SDK warnings
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# ── backend selection ──────────────────────────────────────────────────────────

if _GOOGLE_API_KEY:
    from google import genai as _genai
    from google.genai import types as _gtypes

    _DEFAULT_MODEL = _LLM_MODEL or "gemini-2.0-flash"

    _client = _genai.Client(api_key=_GOOGLE_API_KEY)
    _fallback_client = _genai.Client(api_key=_GOOGLE_CLOUD_API_KEY) if _GOOGLE_CLOUD_API_KEY else None

    # Query the model's actual output token limit at startup
    _model_info = _client.models.get(model=_DEFAULT_MODEL)
    _MAX_OUTPUT_TOKENS = _model_info.output_token_limit
    _fallback_label = "cloud key" if _fallback_client else "none"
    print(f"[llm] Gemini backend: {_DEFAULT_MODEL} (max output tokens: {_MAX_OUTPUT_TOKENS:,}, fallback: {_fallback_label})")


    def _generate(client, model: str, system: str, user: str, temperature: float, max_tokens: int | None) -> str:
        response = client.models.generate_content(
            model=model,
            contents=user,
            config=_gtypes.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens or _MAX_OUTPUT_TOKENS,
                thinking_config=_gtypes.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()


    def chat(
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> str:
        _model = model or _DEFAULT_MODEL
        try:
            return _generate(_client, _model, system, user, temperature, max_tokens)
        except Exception as exc:
            # 429 RESOURCE_EXHAUSTED — daily quota on primary key exhausted
            exc_str = str(exc)
            if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                if _fallback_client:
                    print(f"\n[llm] Primary key quota exhausted — switching to Google Cloud fallback key")
                    return _generate(_fallback_client, _model, system, user, temperature, max_tokens)
                else:
                    raise RuntimeError(
                        "Primary Gemini key hit daily quota and no GOOGLE_CLOUD_API_KEY fallback is set.\n"
                        "Add GOOGLE_CLOUD_API_KEY to your .env to enable automatic failover."
                    ) from exc
            raise

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
