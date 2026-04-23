"""
Import Age of Empires II videos from a plain text file of YouTube URLs.

Each non-empty line should contain one of:
  - a single YouTube video URL
  - a YouTube playlist URL
  - video|<YouTube URL> to force standard educational-video analysis
  - coaching|<YouTube URL> to force coaching / replay-review analysis

Blank lines and lines beginning with '#' are ignored.

Examples:
    uv run python -m scrape.aoe2_video_import
    uv run python -m scrape.aoe2_video_import --input data/aoe2_video_urls.txt
    uv run python -m scrape.aoe2_video_import --review
"""

from __future__ import annotations

import argparse
import pathlib
import re
from datetime import datetime, timezone

import yt_dlp

from core.db_paths import activate_knowledge_db

activate_knowledge_db()

from core.database import get_connection, init_db, insert_video

DEFAULT_INPUT = pathlib.Path("data/aoe2_video_urls.txt")
DEFAULT_SOURCE = "aoe2_video"
COACHING_SOURCE = "aoe2_coaching"
AOE2_VIDEO_SOURCES = {DEFAULT_SOURCE, COACHING_SOURCE}
SOURCE_OVERRIDES = {
    "video": DEFAULT_SOURCE,
    "coaching": COACHING_SOURCE,
}

COACHING_PATTERNS = (
    ("coaching", re.compile(r"\bcoach(?:ing|ed)?\b", re.IGNORECASE)),
    ("student", re.compile(r"\bstudent\b", re.IGNORECASE)),
    ("review", re.compile(r"\b(?:vod|replay|gameplay)?\s*review\b", re.IGNORECASE)),
    ("analysis", re.compile(r"\banalys(?:e|is|ed|ing)\b", re.IGNORECASE)),
    ("mistakes", re.compile(r"\bmistakes?\b", re.IGNORECASE)),
    ("fixing", re.compile(r"\bfix(?:ing|ed)?\b", re.IGNORECASE)),
    ("live coaching", re.compile(r"\blive coaching\b", re.IGNORECASE)),
)

VIDEO_PATTERNS = (
    ("guide", re.compile(r"\bguide\b", re.IGNORECASE)),
    ("build order", re.compile(r"\bbuild order\b", re.IGNORECASE)),
    ("how to", re.compile(r"\bhow to\b", re.IGNORECASE)),
    ("tips", re.compile(r"\btips?\b", re.IGNORECASE)),
    ("strategy", re.compile(r"\bstrateg(?:y|ies|ic)\b", re.IGNORECASE)),
    ("opening", re.compile(r"\bopening\b", re.IGNORECASE)),
    ("beginner", re.compile(r"\bbeginner(?:s)?\b", re.IGNORECASE)),
    ("fundamentals", re.compile(r"\bfundamentals?\b", re.IGNORECASE)),
)


def _parse_input_line(raw_line: str) -> dict | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None

    prefix, sep, remainder = line.partition("|")
    forced_source = SOURCE_OVERRIDES.get(prefix.strip().lower()) if sep else None
    url = remainder.strip() if forced_source else line
    if not url:
        return None

    return {
        "url": url,
        "forced_source": forced_source,
    }


def _read_url_lines(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_input_line(raw_line)
        if parsed:
            rows.append(parsed)
    return rows


def _canonical_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _expand_url(url: str) -> list[dict]:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if not info:
        return []

    if info.get("_type") == "playlist" or info.get("entries"):
        playlist_title = info.get("title") or ""
        playlist_channel = info.get("channel") or info.get("uploader") or ""
        rows: list[dict] = []
        for entry in info.get("entries") or []:
            if not entry:
                continue
            video_id = entry.get("id")
            if not video_id:
                continue
            rows.append({
                "video_id": video_id,
                "video_url": (
                    entry.get("url")
                    if str(entry.get("url", "")).startswith("http")
                    else _canonical_watch_url(video_id)
                ),
                "video_title": entry.get("title") or video_id,
                "description": f"Imported from playlist: {playlist_title}" if playlist_title else None,
                "playlist_title": playlist_title or None,
                "channel": entry.get("channel") or entry.get("uploader") or playlist_channel or None,
                "uploader": entry.get("uploader") or playlist_channel or None,
                "source_url": url,
            })
        return rows

    video_id = info.get("id")
    if not video_id:
        return []

    return [{
        "video_id": video_id,
        "video_url": info.get("webpage_url") or _canonical_watch_url(video_id),
        "video_title": info.get("title") or video_id,
        "description": None,
        "playlist_title": None,
        "channel": info.get("channel") or info.get("uploader") or None,
        "uploader": info.get("uploader") or info.get("channel") or None,
        "source_url": url,
    }]


def _matching_labels(text: str, patterns: tuple[tuple[str, re.Pattern[str]], ...]) -> list[str]:
    if not text:
        return []
    matches: list[str] = []
    for label, pattern in patterns:
        if pattern.search(text):
            matches.append(label)
    return matches


def _classify_row(row: dict, forced_source: str | None = None) -> dict:
    enriched = dict(row)
    if forced_source:
        enriched["source"] = forced_source
        enriched["classification"] = "forced"
        enriched["uncertain"] = False
        enriched["reason"] = "explicit override"
        return enriched

    metadata_blob = " | ".join(
        str(value)
        for value in (
            row.get("video_title"),
            row.get("playlist_title"),
            row.get("channel"),
            row.get("uploader"),
            row.get("description"),
        )
        if value
    )
    coaching_hits = _matching_labels(metadata_blob, COACHING_PATTERNS)
    video_hits = _matching_labels(metadata_blob, VIDEO_PATTERNS)

    if coaching_hits and not video_hits:
        source = COACHING_SOURCE
        classification = "heuristic"
        uncertain = False
        reason = f"coaching cues: {', '.join(coaching_hits)}"
    elif video_hits and not coaching_hits:
        source = DEFAULT_SOURCE
        classification = "heuristic"
        uncertain = False
        reason = f"video cues: {', '.join(video_hits)}"
    elif len(coaching_hits) >= len(video_hits) + 2 and coaching_hits:
        source = COACHING_SOURCE
        classification = "heuristic"
        uncertain = False
        reason = f"coaching-leaning mixed cues: {', '.join(coaching_hits)}"
    elif len(video_hits) >= len(coaching_hits) + 2 and video_hits:
        source = DEFAULT_SOURCE
        classification = "heuristic"
        uncertain = False
        reason = f"video-leaning mixed cues: {', '.join(video_hits)}"
    else:
        source = DEFAULT_SOURCE
        classification = "uncertain"
        uncertain = True
        reason_parts = []
        if coaching_hits:
            reason_parts.append(f"coaching={', '.join(coaching_hits)}")
        if video_hits:
            reason_parts.append(f"video={', '.join(video_hits)}")
        reason = "; ".join(reason_parts) if reason_parts else "no strong metadata cues"

    enriched["source"] = source
    enriched["classification"] = classification
    enriched["uncertain"] = uncertain
    enriched["reason"] = reason
    return enriched


def _expand_and_classify(entries: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for entry in entries:
        expanded = _expand_url(entry["url"])
        for row in expanded:
            rows.append(_classify_row(row, forced_source=entry.get("forced_source")))
    return rows


def _print_review(rows: list[dict]) -> None:
    grouped = {
        COACHING_SOURCE: [],
        DEFAULT_SOURCE: [],
        "uncertain": [],
    }
    for row in rows:
        key = "uncertain" if row.get("uncertain") else row["source"]
        grouped.setdefault(key, []).append(row)

    print("Classification preview:")
    print(f"  coaching:  {len(grouped.get(COACHING_SOURCE, []))}")
    print(f"  video:     {len(grouped.get(DEFAULT_SOURCE, []))}")
    print(f"  uncertain: {len(grouped.get('uncertain', []))}")

    for key in ("uncertain", COACHING_SOURCE, DEFAULT_SOURCE):
        items = grouped.get(key) or []
        if not items:
            continue
        print(f"\n[{key}]")
        for row in items:
            print(
                f"  - {row['video_id']} | {row['video_title']} "
                f"| {row['source']} | {row['reason']}"
            )


def _update_existing_row(row: dict, message_timestamp: str) -> bool:
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT source FROM videos WHERE video_id = ?",
            (row["video_id"],),
        ).fetchone()
        if not existing or existing["source"] not in AOE2_VIDEO_SOURCES:
            return False

        existing_source = existing["source"]
        incoming_source = row["source"]
        source_is_forced = row.get("classification") == "forced"
        next_source = incoming_source if (source_is_forced or existing_source == incoming_source) else existing_source

        conn.execute(
            """
            UPDATE videos
            SET video_url = ?, video_title = ?, description = ?, game = 'aoe2',
                role = 'general', subject = NULL, champion = NULL,
                message_timestamp = ?, source = ?
            WHERE video_id = ?
            """,
            (
                row["video_url"],
                row["video_title"],
                row.get("description"),
                message_timestamp,
                next_source,
                row["video_id"],
            ),
        )
        conn.commit()
        return existing_source != next_source


def import_urls(entries: list[dict], review: bool = False) -> dict[str, int]:
    expanded_rows: list[dict] = []
    expansion_errors = 0
    for entry in entries:
        try:
            expanded_rows.extend(_expand_and_classify([entry]))
        except Exception as exc:
            print(f"[skip] {entry['url']} ({exc})")
            expansion_errors += 1

    if review:
        _print_review(expanded_rows)
        return {
            "imported": 0,
            "expanded": len(expanded_rows),
            "reviewed": len(expanded_rows),
            "coaching": sum(1 for row in expanded_rows if row["source"] == COACHING_SOURCE and not row["uncertain"]),
            "video": sum(1 for row in expanded_rows if row["source"] == DEFAULT_SOURCE and not row["uncertain"]),
            "uncertain": sum(1 for row in expanded_rows if row["uncertain"]),
            "reclassified": 0,
            "errors": expansion_errors,
        }

    with get_connection() as conn:
        existing_ids = {
            row["video_id"] for row in conn.execute("SELECT video_id FROM videos").fetchall()
        }

    imported = 0
    reclassified = 0
    message_timestamp = datetime.now(timezone.utc).isoformat()

    if expanded_rows:
        _print_review(expanded_rows)

    for row in expanded_rows:
        if row["video_id"] in existing_ids:
            if _update_existing_row(row, message_timestamp):
                reclassified += 1
            continue

        insert_video(
            video_id=row["video_id"],
            video_url=row["video_url"],
            video_title=row["video_title"],
            description=row.get("description"),
            game="aoe2",
            role="general",
            subject=None,
            champion=None,
            rank=None,
            website_rating=None,
            message_timestamp=message_timestamp,
            source=row["source"],
        )
        imported += 1
        existing_ids.add(row["video_id"])

    return {
        "imported": imported,
        "expanded": len(expanded_rows),
        "reviewed": 0,
        "coaching": sum(1 for row in expanded_rows if row["source"] == COACHING_SOURCE and not row["uncertain"]),
        "video": sum(1 for row in expanded_rows if row["source"] == DEFAULT_SOURCE and not row["uncertain"]),
        "uncertain": sum(1 for row in expanded_rows if row["uncertain"]),
        "reclassified": reclassified,
        "errors": expansion_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import AoE2 YouTube videos/playlists from a txt file")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Text file with one video or playlist URL per line")
    parser.add_argument("--review", action="store_true", help="Preview per-video classification and exit without inserting")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --review")
    args = parser.parse_args()

    init_db()
    input_path = pathlib.Path(args.input)
    urls = _read_url_lines(input_path)
    summary = import_urls(urls, review=args.review or args.dry_run)
    if args.review or args.dry_run:
        print(
            f"\nReviewed {summary['reviewed']} AoE2 video row(s) from {input_path} "
            f"({summary['errors']} expansion error(s))"
        )
        return

    print(
        f"Imported {summary['imported']} new AoE2 video row(s) from {input_path} "
        f"({summary['expanded']} expanded entries scanned, "
        f"{summary['reclassified']} existing row(s) reclassified, "
        f"{summary['uncertain']} uncertain defaulted to {DEFAULT_SOURCE})"
    )


if __name__ == "__main__":
    main()
