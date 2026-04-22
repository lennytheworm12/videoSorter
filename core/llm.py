"""
LLM abstraction layer.

Uses Google Gemini when GOOGLE_API_KEY is set, falls back to Ollama for local dev.
A second Gemini key (GOOGLE_CLOUD_API_KEY) is used as an automatic fallback when
the primary key hits its daily quota (429 RESOURCE_EXHAUSTED).

Environment variables:
    GOOGLE_API_KEY        — primary Gemini key (free tier or paid)
    GOOGLE_CLOUD_API_KEY  — fallback Gemini key (Google Cloud project key)
    LLM_MODEL             — model to use
                            Gemini default : gemini-3.1-flash-lite-preview
                            Ollama default : gemma4:e2b
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

_GOOGLE_API_KEY       = os.environ.get("GOOGLE_API_KEY")
_GOOGLE_CLOUD_API_KEY = os.environ.get("GOOGLE_API_KEY_TWO") or os.environ.get("GOOGLE_CLOUD_API_KEY")
_LLM_MODEL            = os.environ.get("LLM_MODEL")

# Suppress noisy SDK warnings
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# ── backend selection ──────────────────────────────────────────────────────────

if _GOOGLE_API_KEY:
    from google import genai as _genai
    from google.genai import types as _gtypes

    _DEFAULT_MODEL = _LLM_MODEL or "gemini-3.1-flash-lite-preview"
    _client = _genai.Client(api_key=_GOOGLE_API_KEY)
    _fallback_client = _genai.Client(api_key=_GOOGLE_CLOUD_API_KEY) if _GOOGLE_CLOUD_API_KEY else None

    _MAX_OUTPUT_TOKENS = 65536
    _fallback_label = "GOOGLE_API_KEY_TWO" if _fallback_client else "none"
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
        import time as _time
        _model = model or _DEFAULT_MODEL
        _503_delays = [10, 30, 60]  # retry on transient server overload

        for attempt in range(len(_503_delays) + 1):
            try:
                return _generate(_client, _model, system, user, temperature, max_tokens)
            except Exception as exc:
                exc_str = str(exc)

                # 503 UNAVAILABLE — Gemini overloaded, retry with backoff
                if ("503" in exc_str or "UNAVAILABLE" in exc_str) and attempt < len(_503_delays):
                    wait = _503_delays[attempt]
                    print(f"\n[llm] Gemini 503 — retrying in {wait}s (attempt {attempt + 1})…", flush=True)
                    _time.sleep(wait)
                    continue

                # 429 RESOURCE_EXHAUSTED — rate limit or daily quota
                if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                    import re as _re
                    _is_daily = "PerDay" in exc_str or "per day" in exc_str.lower() or "free_tier_requests" in exc_str
                    if _is_daily:
                        if _fallback_client:
                            print(f"\n[llm] Primary key daily quota exhausted — switching to GOOGLE_API_KEY_TWO", flush=True)
                            try:
                                return _generate(_fallback_client, _model, system, user, temperature, max_tokens)
                            except Exception as fallback_exc:
                                raise RuntimeError(
                                    "Both Gemini keys hit their daily quota (500 req/day each).\n"
                                    "Quota resets at midnight Pacific. Try again tomorrow."
                                ) from fallback_exc
                        raise RuntimeError(
                            "Primary Gemini key hit daily quota and no fallback key is set.\n"
                            "Add GOOGLE_API_KEY_TWO to your .env to enable automatic failover."
                        ) from exc
                    # Per-minute rate limit: sleep and retry
                    if "retryDelay" in exc_str or ("Per" in exc_str and "Minute" in exc_str):
                        delay_match = _re.search(r"retryDelay.*?(\d+)s", exc_str)
                        wait = int(delay_match.group(1)) + 2 if delay_match else 60
                        code_match = _re.search(r"(\d{3})[^\n]*?([A-Z_]{3,})", exc_str)
                        code_hint = f" [{code_match.group(0)[:60]}]" if code_match else ""
                        print(f"\n[llm] Rate limited{code_hint} — sleeping {wait}s…", flush=True)
                        _time.sleep(wait)
                        continue
                    # Unknown 429 — try fallback if available, else raise
                    if _fallback_client:
                        print(f"\n[llm] Primary key quota exhausted — switching to GOOGLE_API_KEY_TWO", flush=True)
                        try:
                            return _generate(_fallback_client, _model, system, user, temperature, max_tokens)
                        except Exception as fallback_exc:
                            raise RuntimeError(
                                "Both Gemini keys are quota-exhausted."
                            ) from fallback_exc
                    raise RuntimeError(
                        "Primary Gemini key hit daily quota and no fallback key is set.\n"
                        "Add GOOGLE_API_KEY_TWO to your .env to enable automatic failover."
                    ) from exc

                raise

        raise RuntimeError("Gemini 503 persisted after all retries")

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
