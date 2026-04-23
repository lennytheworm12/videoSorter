"""
Runs local LLM analysis on transcribed videos to extract structured LoL insights.

Requires Ollama running locally:
    ollama serve
    ollama pull llama3.1:8b

Run after transcribing:
    python analyze.py

Safe to re-run — already-analyzed videos are skipped.
"""

import json
import os
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from core.database import get_videos_by_status, set_status, insert_insight, get_connection
from core.champions import correct_names, champion_names_for_prompt
from core.llm import chat as llm_chat, BACKEND, MODEL as LLM_MODEL
from prompts.lol import LOL_COACHING_EXTRACTION_PROMPT, LOL_COACHING_SYSTEM_PROMPT
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Window size (words) and stride for sliding-window source grounding
_WINDOW_WORDS = 40
_WINDOW_STRIDE = 20

# ~4 chars per token. Keep chunks at 20K chars (~5K tokens) so the full prompt
# (system + champion list + extraction prompt + chunk) stays within Gemma's
# 16K context window and leaves ~6K tokens for the JSON output.
CHUNK_CHARS = 10_000

_embed_model: SentenceTransformer | None = None
_EMBED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=_EMBED_DEVICE)
    return _embed_model


def _embed_chunk_windows(chunk: str) -> np.ndarray:
    """
    Split a transcript chunk into overlapping word windows and embed them.
    Returns a (n_windows, 384) normalised float32 matrix.
    Used once per chunk so all insights in that chunk can be scored cheaply.
    """
    words = chunk.split()
    windows = [
        " ".join(words[i : i + _WINDOW_WORDS])
        for i in range(0, max(1, len(words) - _WINDOW_WORDS + 1), _WINDOW_STRIDE)
    ]
    if not windows:
        windows = [chunk[:500]]
    model = _get_embed_model()
    return model.encode(windows, convert_to_numpy=True, normalize_embeddings=True)


def score_source_grounding(insight_text: str, window_matrix: np.ndarray) -> float:
    """
    Return the max cosine similarity between an insight and any window of its
    source transcript chunk.  Pre-normalised vectors → dot product = cosine sim.

    Score interpretation:
      > 0.50  likely grounded in the transcript
      0.30-0.50  uncertain
      < 0.30  likely hallucinated from model priors
    """
    model = _get_embed_model()
    vec = model.encode(insight_text, convert_to_numpy=True, normalize_embeddings=True)
    return float((window_matrix @ vec).max())


def chunk_transcript(transcript: str) -> list[str]:
    """
    Gemini: returns the full transcript as a single chunk — 1M token context
    means even the longest coaching session fits in one call.

    Ollama fallback: splits into CHUNK_CHARS-sized chunks at sentence
    boundaries (. ! ?) with a 200-char lookahead, falling back to word
    boundaries for unpunctuated auto-captions.
    """
    if BACKEND == "gemini":
        return [transcript]

    SENTENCE_ENDINGS = {'.', '!', '?'}
    LOOKAHEAD = 200

    chunks = []
    start = 0
    n = len(transcript)

    while start < n:
        end = min(start + CHUNK_CHARS, n)

        if end < n:
            window_start = max(start, end - LOOKAHEAD)
            split_at = None
            for i in range(end - 1, window_start - 1, -1):
                if transcript[i] in SENTENCE_ENDINGS:
                    split_at = i + 1
                    break

            if split_at is None:
                space = transcript.rfind(' ', window_start, end)
                split_at = space + 1 if space != -1 else end

            end = split_at

        chunks.append(transcript[start:end].strip())
        start = end

    return [c for c in chunks if c]


def _call_ollama(chunk: str, role: str, champion: str | None, description: str | None, model: str) -> dict[str, list[str]]:
    """Single LLM call for one chunk. Returns parsed dict or raises ValueError on bad JSON."""
    prompt = LOL_COACHING_EXTRACTION_PROMPT.format(
        role=role,
        champion=champion or "unknown",
        description=description or "no description",
        transcript_chunk=chunk,
    )

    # Append full champion list to system prompt so the model knows every name
    system = LOL_COACHING_SYSTEM_PROMPT + "\n\nFULL CHAMPION LIST (use exact spelling from this list):\n" + champion_names_for_prompt()

    t0 = time.time()
    raw = llm_chat(system=system, user=prompt, temperature=0.1, max_tokens=4096)
    print(f"      [timing] {BACKEND}: {time.time() - t0:.1f}s | {len(chunk):,} chars in", end=" ")

    # Strip markdown code fences if the model wraps output despite instructions
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse failed: {raw[:200]}") from exc


def _merge_results(a: dict, b: dict) -> dict:
    """Merge two insight dicts by extending each category's list."""
    merged = dict(a)
    for key, items in b.items():
        if key in merged:
            merged[key] = merged[key] + (items if isinstance(items, list) else [])
        else:
            merged[key] = items if isinstance(items, list) else []
    return merged


def extract_insights_from_chunk(
    chunk: str,
    role: str,
    champion: str | None,
    description: str | None,
    _depth: int = 0,
) -> dict[str, list[str]]:
    """
    Extract insights from a transcript chunk.

    If the model's JSON output is truncated (parse failure), the chunk is split
    in half and each half is retried independently — up to 2 levels deep (quartering
    the original chunk). This handles dense chunks without shrinking all chunks globally.
    """
    MAX_DEPTH = 2
    MIN_CHARS = 2_000  # don't split below this; just skip

    try:
        result = _call_ollama(chunk, role, champion, description, model=None)
    except ValueError as exc:
        if _depth >= MAX_DEPTH or len(chunk) < MIN_CHARS:
            print(f"    [warn] {exc} (chunk too small to split further, skipping)")
            return {}

        mid = len(chunk) // 2
        # Split at nearest word boundary
        split_at = chunk.rfind(" ", mid - 200, mid + 200)
        if split_at == -1:
            split_at = mid

        half_a, half_b = chunk[:split_at], chunk[split_at:]
        print(f"    [retry] splitting chunk in half ({len(half_a):,} + {len(half_b):,} chars, depth={_depth + 1})")

        result_a = extract_insights_from_chunk(half_a, role, champion, description, _depth + 1)
        result_b = extract_insights_from_chunk(half_b, role, champion, description, _depth + 1)
        result = _merge_results(result_a, result_b)

    # Normalise each item to (text, emphasis) tuple.
    # Handles both old string format and new {text, emphasis} object format.
    for key, items in result.items():
        if not isinstance(items, list):
            continue
        normalised = []
        for item in items:
            if isinstance(item, str) and item.strip():
                normalised.append((correct_names(item.strip()), 1))
            elif isinstance(item, dict) and item.get("text", "").strip():
                text = correct_names(item["text"].strip())
                emphasis = int(item.get("emphasis", 1))
                normalised.append((text, emphasis))
        result[key] = normalised
    return result


def already_analyzed(video_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM insights WHERE video_id = ? LIMIT 1", (video_id,)
        ).fetchone()
    return row is not None


def run() -> None:
    videos = get_videos_by_status("transcribed")
    print(f"Videos to analyze: {len(videos)}")

    for video in videos:
        video_id = video["video_id"]
        role = video["role"]
        champion = video["champion"]
        description = video["description"]
        transcript = video["transcription"]

        if already_analyzed(video_id):
            print(f"[skip] {video_id} already analyzed")
            set_status(video_id, "analyzed")
            continue

        print(f"\n[{role}] {video_id} | champion={champion or '?'}")
        chunks = chunk_transcript(transcript)
        print(f"  {len(transcript.split())} words → {len(chunks)} chunk(s)")

        # Aggregate insights across all chunks
        aggregated: dict[str, list[str]] = {}

        for i, chunk in enumerate(chunks):
            print(f"  Analyzing chunk {i + 1}/{len(chunks)}…")
            try:
                result = extract_insights_from_chunk(chunk, role, champion, description, OLLAMA_MODEL)
            except Exception as e:
                print(f"  [error] chunk {i + 1}: {e}")
                continue

            # Pre-embed this chunk's windows once; reuse for all insights
            window_matrix = _embed_chunk_windows(chunk)

            for insight_type, items in result.items():
                if insight_type not in aggregated:
                    aggregated[insight_type] = []
                for item in items:
                    if isinstance(item, str):
                        score = score_source_grounding(item, window_matrix)
                        aggregated[insight_type].append((item, score))

        # Persist to DB
        total = 0
        flagged = 0
        for insight_type, items in aggregated.items():
            for item, source_score in items:
                if item.strip():
                    insert_insight(video_id, insight_type, item.strip(), source_score)
                    total += 1
                    if source_score < 0.30:
                        flagged += 1

        set_status(video_id, "analyzed")
        print(f"  Saved {total} insights ({flagged} low source-score, possible hallucinations)")


if __name__ == "__main__":
    run()
