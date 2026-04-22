"""
Manually import Age of Empires II source rows from CSV or JSON.

This is the v1 ingestion path for AoE2. It supports:
  - video rows that still need transcription (`source=aoe2_video`)
  - wiki/reference rows that already include raw text (`source=aoe2_wiki`)

Examples:
    uv run python -m scrape.aoe2_import --input data/aoe2_videos.csv
    uv run python -m scrape.aoe2_import --input data/aoe2_sources.json --source aoe2_wiki
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import re
from datetime import datetime, timezone

os.environ.setdefault("DB_PATH", "guide_test.db")

from core.database import get_connection, init_db, insert_video, set_transcription
from core.game_registry import canonical_aoe2_civilization

YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})")


def _read_rows(path: pathlib.Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of objects")
        return [dict(row) for row in data]

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    raise ValueError(f"Unsupported input format: {path.suffix}")


def _pick(row: dict, *keys: str, default: str | None = None) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
        if value not in ("", None):
            return value
    return default


def _derive_video_id(row: dict, source: str) -> str:
    video_url = _pick(row, "video_url", "url")
    if video_url:
        match = YOUTUBE_ID_RE.search(video_url)
        if match:
            return match.group(1)

    provided = _pick(row, "video_id", "id", "source_id")
    if provided:
        return provided

    seed = "|".join([
        source,
        _pick(row, "video_title", "title", default="") or "",
        video_url or "",
        _pick(row, "subject", "civ", default="") or "",
    ])
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"{source}_{digest}"


def _canonical_subject(row: dict) -> str | None:
    raw = _pick(row, "subject", "civ", "civilization")
    return canonical_aoe2_civilization(raw) or raw


def _video_url(row: dict, source: str, video_id: str) -> str:
    url = _pick(row, "video_url", "url")
    if url:
        return url
    return f"aoe2://{source}/{video_id}"


def _role_or_context(row: dict) -> str:
    return _pick(row, "role", "context", default="general") or "general"


def _transcription_text(row: dict) -> str | None:
    return _pick(row, "transcription", "text", "content", "body")


def import_rows(rows: list[dict], default_source: str | None = None) -> tuple[int, int]:
    with get_connection() as conn:
        existing_ids = {
            row["video_id"] for row in conn.execute("SELECT video_id FROM videos").fetchall()
        }

    inserted = 0
    transcribed = 0

    for row in rows:
        source = _pick(row, "source", default=default_source or "aoe2_video") or "aoe2_video"
        video_id = _derive_video_id(row, source)
        transcript = _transcription_text(row)
        is_new = video_id not in existing_ids

        insert_video(
            video_id=video_id,
            video_url=_video_url(row, source, video_id),
            video_title=_pick(row, "video_title", "title"),
            description=_pick(row, "description", "notes"),
            game="aoe2",
            role=_role_or_context(row),
            subject=_canonical_subject(row),
            champion=None,
            rank=None,
            website_rating=None,
            message_timestamp=_pick(
                row,
                "message_timestamp",
                "created_at",
                default=datetime.now(timezone.utc).isoformat(),
            ),
            source=source,
        )
        if is_new:
            inserted += 1
            existing_ids.add(video_id)

        if transcript:
            set_transcription(video_id, transcript)
            transcribed += 1

    return inserted, transcribed


def main() -> None:
    parser = argparse.ArgumentParser(description="Import AoE2 videos/wiki rows into guide_test.db")
    parser.add_argument("--input", required=True, help="CSV or JSON file with AoE2 source rows")
    parser.add_argument("--source", choices=["aoe2_video", "aoe2_wiki"],
                        help="Default source label when rows do not provide one")
    args = parser.parse_args()

    init_db()
    input_path = pathlib.Path(args.input)
    rows = _read_rows(input_path)
    inserted, transcribed = import_rows(rows, default_source=args.source)
    print(
        f"Imported {inserted} AoE2 row(s) from {input_path} "
        f"({transcribed} marked transcribed, {inserted - transcribed} left pending)"
    )


if __name__ == "__main__":
    main()
