"""
Transcribe YouTube-backed guide rows in knowledge.db.

This wrapper targets only transcribable guide sources:
  - youtube_guide
  - aoe2_video
  - aoe2_coaching

Usage:
    uv run python -m pipeline.guide_transcribe
    uv run python -m pipeline.guide_transcribe --game aoe2
    uv run python -m pipeline.guide_transcribe --source aoe2_video
    uv run python -m pipeline.guide_transcribe --reanalyze --source aoe2_video
"""

from __future__ import annotations

import argparse
from core.db_paths import activate_knowledge_db

activate_knowledge_db()

from core.database import get_connection
from pipeline.transcribe import run as transcribe_run

TRANSCRIBABLE_GUIDE_SOURCES = ("youtube_guide", "aoe2_video", "aoe2_coaching")
GAME_SOURCE_MAP = {
    "lol": ("youtube_guide",),
    "aoe2": ("aoe2_video", "aoe2_coaching"),
}


def _selected_sources(source: str | None = None, game: str | None = None) -> list[str]:
    if source:
        return [source]
    if game:
        return list(GAME_SOURCE_MAP[game])
    return list(TRANSCRIBABLE_GUIDE_SOURCES)


def _reset_for_retranscribe(source: str | None = None, game: str | None = None) -> tuple[int, int]:
    targets = _selected_sources(source=source, game=game)
    placeholders = ",".join("?" * len(targets))

    with get_connection() as conn:
        video_ids = [
            row["video_id"]
            for row in conn.execute(
                f"SELECT video_id FROM videos WHERE source IN ({placeholders})",
                targets,
            ).fetchall()
        ]
        if not video_ids:
            return 0, 0

        insight_placeholders = ",".join("?" * len(video_ids))
        deleted_insights = conn.execute(
            f"DELETE FROM insights WHERE video_id IN ({insight_placeholders})",
            video_ids,
        ).rowcount
        conn.execute(
            f"""
            UPDATE videos
            SET status = 'pending',
                transcription = NULL
            WHERE source IN ({placeholders})
            """,
            targets,
        )
        conn.commit()

    return len(video_ids), deleted_insights


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe guide videos in knowledge.db")
    parser.add_argument(
        "--game",
        choices=sorted(GAME_SOURCE_MAP),
        help="Limit transcription to one game's guide-video sources",
    )
    parser.add_argument(
        "--source",
        choices=list(TRANSCRIBABLE_GUIDE_SOURCES),
        help="Limit transcription to one transcribable guide source",
    )
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="Reset selected rows to pending, clear stored transcripts, and delete old insights first",
    )
    args = parser.parse_args()

    if args.game and args.source:
        valid_sources = set(GAME_SOURCE_MAP[args.game])
        if args.source not in valid_sources:
            parser.error(f"--source {args.source} does not belong to --game {args.game}")

    if args.reanalyze:
        reset_count, deleted_insights = _reset_for_retranscribe(args.source, game=args.game)
        scope = args.source or args.game or "all transcribable guide sources"
        print(
            f"Reset {reset_count} video(s) for {scope} "
            f"({deleted_insights} insight row(s) deleted)"
        )

    selected_sources = _selected_sources(source=args.source, game=args.game)
    transcribe_run(sources=selected_sources, game=args.game)


if __name__ == "__main__":
    main()
