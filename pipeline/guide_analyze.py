"""
LLM analysis for guide-style sources in knowledge.db.

Supported sources:
  - youtube_guide   : transcribed League educational videos
  - mobafire_guide  : cleaned League written guide text saved directly into transcription
  - aoe2_video      : transcribed Age of Empires II educational videos
  - aoe2_coaching   : transcribed Age of Empires II coaching / review videos
  - aoe2_wiki       : imported Age of Empires II reference/wiki text

Usage:
    uv run python -m pipeline.guide_analyze               # all guide sources
    uv run python -m pipeline.guide_analyze --champion Aatrox
    uv run python -m pipeline.guide_analyze --source mobafire_guide
    uv run python -m pipeline.guide_analyze --dry-run     # print insights, don't save
"""

import json
import time
import argparse
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from core.db_paths import activate_knowledge_db

activate_knowledge_db()

from core.database import get_connection, set_status, insert_insight, init_db
from core.champions import correct_names
from core.game_registry import (
    DEFAULT_GAME,
    SUPPORTED_GAMES,
    analysis_spec,
    canonical_aoe2_civilization,
    normalize_game,
)
from core.llm import chat as llm_chat, BACKEND

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_WINDOW_WORDS  = 40
_WINDOW_STRIDE = 20

GUIDE_SOURCES = ("youtube_guide", "mobafire_guide", "aoe2_video", "aoe2_coaching", "aoe2_wiki")

# Match the main League analyzer's smart context breaking for long transcripts.
CHUNK_CHARS = 10_000

_embed_model: SentenceTransformer | None = None
_EMBED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=_EMBED_DEVICE)
    return _embed_model


def _embed_chunk_windows(chunk: str) -> np.ndarray:
    words = chunk.split()
    windows = [
        " ".join(words[i: i + _WINDOW_WORDS])
        for i in range(0, max(1, len(words) - _WINDOW_WORDS + 1), _WINDOW_STRIDE)
    ]
    if not windows:
        windows = [chunk[:500]]
    model = _get_embed_model()
    return model.encode(windows, convert_to_numpy=True, normalize_embeddings=True)


def score_source_grounding(insight_text: str, window_matrix: np.ndarray) -> float:
    model = _get_embed_model()
    vec = model.encode(insight_text, convert_to_numpy=True, normalize_embeddings=True)
    return float((window_matrix @ vec).max())


def chunk_transcript(transcript: str) -> list[str]:
    """
    Match the main League analyzer's context-aware transcript chunking.

    Gemini takes the full transcript in one call. Other backends split near
    sentence boundaries with a short lookback window and fall back to word
    boundaries for messy auto-captions.
    """
    if BACKEND == "gemini":
        return [transcript]

    sentence_endings = {".", "!", "?"}
    lookahead = 200
    chunks: list[str] = []
    start = 0
    n = len(transcript)

    while start < n:
        end = min(start + CHUNK_CHARS, n)

        if end < n:
            window_start = max(start, end - lookahead)
            split_at = None
            for i in range(end - 1, window_start - 1, -1):
                if transcript[i] in sentence_endings:
                    split_at = i + 1
                    break

            if split_at is None:
                space = transcript.rfind(" ", window_start, end)
                split_at = space + 1 if space != -1 else end

            end = split_at

        chunks.append(transcript[start:end].strip())
        start = end

    return [chunk for chunk in chunks if chunk] or [transcript]


def _call_llm(
    chunk: str,
    subject: str,
    role: str,
    title: str,
    chunk_label: str,
    source: str,
    game: str,
) -> dict:
    # Escape curly braces in user content before .format() so guide text with
    # {Q}/{W} ability notation or JSON examples doesn't get parsed as format fields.
    spec = analysis_spec(game, source)
    safe_chunk = chunk.replace("{", "{{").replace("}", "}}")
    prompt = spec.extraction_prompt.format(
        subject=subject or "unknown",
        role=role or "unknown",
        title=title or "",
        chunk_label=chunk_label,
        transcript_chunk=safe_chunk,
    )
    system = spec.system_prompt
    if spec.reference_block:
        system += "\n\nREFERENCE LIST:\n" + spec.reference_block

    t0 = time.time()
    raw = llm_chat(system=system, user=prompt, temperature=0.1, max_tokens=4096)
    print(f"      [{BACKEND}] {time.time() - t0:.1f}s | {len(chunk.split()):,} words in", end=" ")

    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(0))
            except Exception:
                result = {}
        else:
            result = {}

    normalised_result = {key: result.get(key, []) for key in spec.insight_types}

    # Normalise items to structured tuples so downstream storage can tag insights.
    for key, items in normalised_result.items():
        if not isinstance(items, list):
            normalised_result[key] = []
            continue
        normalised = []
        for item in items:
            if isinstance(item, str) and item.strip():
                text = correct_names(item.strip()) if game == "lol" else item.strip()
                normalised.append((text, 1, None, None))
            elif isinstance(item, dict) and item.get("text", "").strip():
                text = item["text"].strip()
                if game == "lol":
                    text = correct_names(text)
                emphasis = int(item.get("emphasis", 1))
                insight_subject = item.get("subject")
                insight_subject_type = item.get("subject_type")
                if game == "aoe2":
                    if insight_subject_type not in {"civ", "general"}:
                        insight_subject_type = "general" if not insight_subject else "civ"
                    if insight_subject_type == "civ":
                        insight_subject = canonical_aoe2_civilization(str(insight_subject or "").strip())
                        if not insight_subject:
                            insight_subject_type = "general"
                    else:
                        insight_subject = None
                normalised.append((text, emphasis, insight_subject, insight_subject_type))
        normalised_result[key] = normalised

    return normalised_result


def _already_analyzed(video_id: str) -> bool:
    with get_connection() as conn:
        return conn.execute(
            "SELECT 1 FROM insights WHERE video_id = ? LIMIT 1", (video_id,)
        ).fetchone() is not None


def _get_pending_guide_videos(
    subject: str | None = None,
    source: str | None = None,
    game: str | None = None,
) -> list:
    with get_connection() as conn:
        source_clause = "source IN ('youtube_guide', 'mobafire_guide', 'aoe2_video', 'aoe2_coaching', 'aoe2_wiki')"
        params: list = []
        if source:
            source_clause = "source = ?"
            params.append(source)
        game_clause = ""
        if game:
            game_clause = " AND game = ?"
            params.append(normalize_game(game))
        if subject:
            return conn.execute(
                """
                SELECT * FROM videos
                WHERE status = 'transcribed'
                  AND """ + source_clause + """
                  """ + game_clause + """
                  AND LOWER(COALESCE(subject, champion)) = LOWER(?)
                ORDER BY COALESCE(subject, champion), video_id
                """,
                params + [subject]
            ).fetchall()
        return conn.execute(
            """
            SELECT * FROM videos
            WHERE status = 'transcribed'
              AND """ + source_clause + """
              """ + game_clause + """
            ORDER BY game, COALESCE(subject, champion), video_id
            """,
            params,
        ).fetchall()


def analyze_video(video: dict, dry_run: bool = False) -> int:
    """Analyze one guide video. Returns number of insights saved (or found in dry_run)."""
    video_id = video["video_id"]
    game      = normalize_game(video["game"] or DEFAULT_GAME)
    subject   = video["subject"] or video["champion"] or "unknown"
    role      = video["role"] or "unknown"
    title     = video["video_title"] or ""
    source    = video["source"] or "youtube_guide"
    transcript = video["transcription"] or ""

    chunks = chunk_transcript(transcript)
    total_words = len(transcript.split())
    print(f"  {total_words:,} words → {len(chunks)} chunk(s)")

    aggregated: dict[str, list] = {}

    for i, chunk in enumerate(chunks):
        chunk_label = f"chunk {i + 1} of {len(chunks)}"
        print(f"  Analyzing {chunk_label}…", end=" ", flush=True)
        result = _call_llm(chunk, subject, role, title, chunk_label, source, game)

        window_matrix = _embed_chunk_windows(chunk)
        n_this_chunk = 0
        for insight_type, items in result.items():
            aggregated.setdefault(insight_type, [])
            for text, emphasis, insight_subject, insight_subject_type in items:
                score = score_source_grounding(text, window_matrix)
                aggregated[insight_type].append(
                    (text, emphasis, score, insight_subject, insight_subject_type)
                )
                n_this_chunk += 1
        print(f"→ {n_this_chunk} insights")

    total = 0
    for insight_type, items in aggregated.items():
        for text, emphasis, source_score, insight_subject, insight_subject_type in items:
            if not text.strip():
                continue
            if dry_run:
                subject_suffix = ""
                if insight_subject_type:
                    subject_suffix = f" | {insight_subject_type}:{insight_subject or 'general'}"
                print(f"    [{insight_type}{subject_suffix}] {text}")
            else:
                insert_insight(
                    video_id,
                    insight_type,
                    text.strip(),
                    source_score,
                    repetition_count=emphasis,
                    subject=insight_subject,
                    subject_type=insight_subject_type,
                )
            total += 1

    if not dry_run:
        set_status(video_id, "analyzed")

    return total


def run(
    subject: str | None = None,
    dry_run: bool = False,
    source: str | None = None,
    game: str | None = None,
) -> None:
    videos = _get_pending_guide_videos(subject, source=source, game=game)
    print(f"Guide entries to analyze: {len(videos)}")

    for video in videos:
        video_id = video["video_id"]
        subject_name = video["subject"] or video["champion"] or "general"
        game_name = normalize_game(video["game"] or DEFAULT_GAME)

        if _already_analyzed(video_id):
            print(f"[skip] {video_id} already analyzed")
            set_status(video_id, "analyzed")
            continue

        print(
            f"\n[{game_name}] [{video['source']}] "
            f"[{subject_name}] {video_id} | {(video['video_title'] or '')[:60]}"
        )
        try:
            n = analyze_video(dict(video), dry_run=dry_run)
            print(f"  → {n} insights {'(dry-run, not saved)' if dry_run else 'saved'}")
        except Exception as e:
            print(f"  [error] {e}")
            if not dry_run:
                set_status(video_id, "error")


def _reset_for_reanalysis(subject: str | None, source: str | None, game: str | None) -> None:
    """Delete existing insights and reset status to transcribed so run() picks them up."""
    with get_connection() as conn:
        source_clause = "source IN ('youtube_guide', 'mobafire_guide', 'aoe2_video', 'aoe2_coaching', 'aoe2_wiki')"
        params: list = []
        if source:
            source_clause = "source = ?"
            params.append(source)
        game_clause = ""
        if game:
            game_clause = " AND game = ?"
            params.append(normalize_game(game))
        if subject:
            video_ids = [r["video_id"] for r in conn.execute(
                "SELECT video_id FROM videos WHERE " + source_clause + game_clause + " AND LOWER(COALESCE(subject, champion)) = LOWER(?)",
                params + [subject]
            ).fetchall()]
        else:
            video_ids = [r["video_id"] for r in conn.execute(
                "SELECT video_id FROM videos WHERE " + source_clause + game_clause,
                params
            ).fetchall()]

        if not video_ids:
            return

        placeholders = ",".join("?" * len(video_ids))
        n_insights = conn.execute(
            f"DELETE FROM insights WHERE video_id IN ({placeholders})", video_ids
        ).rowcount
        conn.execute(
            f"UPDATE videos SET status='transcribed' WHERE video_id IN ({placeholders})",
            video_ids
        )
        conn.commit()
    label = subject or "all subjects"
    print(f"Reset {len(video_ids)} video(s) for {label} ({n_insights} insights deleted)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze guide-style entries in knowledge.db")
    parser.add_argument("--game", choices=sorted(SUPPORTED_GAMES), default=DEFAULT_GAME,
                        help="Game namespace to analyze")
    parser.add_argument("--subject", help="Only analyze videos for this subject (champion, civ, strategy)")
    parser.add_argument("--champion", help="Compatibility alias for --subject")
    parser.add_argument("--source", choices=list(GUIDE_SOURCES),
                        help="Limit to one guide source")
    parser.add_argument("--dry-run", action="store_true", help="Print insights, don't save")
    parser.add_argument("--reanalyze", action="store_true",
                        help="Delete existing insights and re-run (use with --champion or alone for all)")
    args = parser.parse_args()

    init_db()
    subject = args.subject or args.champion
    if args.reanalyze:
        _reset_for_reanalysis(subject, args.source, args.game)
    run(subject=subject, dry_run=args.dry_run, source=args.source, game=args.game)


if __name__ == "__main__":
    main()
