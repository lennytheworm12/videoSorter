"""
Search YouTube for champion guide videos and insert them as pending rows.

Uses yt-dlp for search (already installed) — no browser automation needed.
Searches "league of legends {champion} guide", filters by title keywords and
duration > 5 minutes, and inserts up to --limit videos per champion.

Chunking note: transcripts are stored as-is. pipeline/guide_analyze.py splits
them into 1-hour windows at analysis time so long guides don't dilute context.

Usage:
    uv run python -m scrape.youtube_scrape --champion Aatrox   # single champion
    uv run python -m scrape.youtube_scrape                      # all A→Z
    uv run python -m scrape.youtube_scrape --limit 10           # videos per champion
    uv run python -m scrape.youtube_scrape --status             # show DB counts
"""

import os
import re
import time
import random
import argparse
import yt_dlp

# Write to isolated test DB until guide prompts are validated
os.environ.setdefault("DB_PATH", "guide_test.db")

from core.database import get_connection, insert_video, init_db
from core.champions import load_champion_names

GUIDE_KEYWORDS = {
    "guide", "how to play", "tips", "tutorial", "educational",
    "breakdown", "beginners", "beginner", "learn", "master",
    "climb", "ranked", "build", "runes",
}

ROLE_PATTERNS: list[tuple[str, list[str]]] = [
    ("top",     ["top lane", "top laner", "toplane", " top ", "toplaner"]),
    ("jungle",  ["jungle", "jungler", " jg ", "jg guide"]),
    ("mid",     ["mid lane", "mid laner", "midlane", " mid ", "midlaner"]),
    ("adc",     ["adc", "bot lane", "bot laner", "marksman", "carry"]),
    ("support", ["support", "supp ", "sup guide"]),
]

MIN_DURATION_S = 5 * 60    # 5 minutes
MAX_DURATION_S = 6 * 3600  # 6 hours
SEARCH_FETCH = 30         # fetch more candidates so sort-by-duration has a good pool


def _detect_role_from_title(title: str) -> str | None:
    t = title.lower()
    for role, patterns in ROLE_PATTERNS:
        if any(p in t for p in patterns):
            return role
    return None


def _champion_primary_role(champion: str) -> str:
    """Return the champion's most-used role from video data, fallback archetype."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT role, COUNT(*) AS cnt
            FROM videos
            WHERE champion = ? AND role NOT IN ('ability_enrichment')
            GROUP BY role
            ORDER BY cnt DESC
            LIMIT 1
            """,
            (champion,)
        ).fetchone()
        if row:
            return row["role"]
        row = conn.execute(
            "SELECT role FROM champion_archetypes WHERE champion = ? LIMIT 1",
            (champion,)
        ).fetchone()
        if row:
            return row["role"]
    return "unknown"


def _has_guide_keyword(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in GUIDE_KEYWORDS)


def _get_existing_ids() -> set[str]:
    with get_connection() as conn:
        rows = conn.execute("SELECT video_id FROM videos").fetchall()
    return {r["video_id"] for r in rows}


def search_youtube(query: str, n: int = SEARCH_FETCH) -> list[dict]:
    """
    Run a YouTube search via yt-dlp and return basic metadata dicts.
    Each dict: {video_id, title, duration, url}
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }
    results = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
        if not info or "entries" not in info:
            return []
        for entry in (info["entries"] or []):
            if not entry or not entry.get("id"):
                continue
            results.append({
                "video_id": entry["id"],
                "title":    entry.get("title", ""),
                "duration": entry.get("duration") or 0,
                "url":      f"https://www.youtube.com/watch?v={entry['id']}",
            })
    return results


def filter_results(
    results: list[dict],
    existing_ids: set[str],
    limit: int,
) -> list[dict]:
    """Filter by duration and keywords, then sort longest-first."""
    kept = []
    for v in results:
        if v["video_id"] in existing_ids:
            continue
        if v["duration"] and v["duration"] < MIN_DURATION_S:
            continue
        if v["duration"] and v["duration"] > MAX_DURATION_S:
            continue
        if not _has_guide_keyword(v["title"]):
            continue
        kept.append(v)
    # Longest videos first — educational deep-dives over short clips
    kept.sort(key=lambda v: v["duration"] or 0, reverse=True)
    return kept[:limit]


def _champion_video_count(champion: str) -> int:
    with get_connection() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM videos WHERE champion = ? AND source = 'youtube_guide'",
            (champion,)
        ).fetchone()[0]


def scrape_champion(champion: str, limit: int = 10, dry_run: bool = False) -> int:
    """
    Search, filter, and insert guide videos for one champion.
    Returns the number of new rows inserted (or would-be inserted in dry_run).
    """
    existing = _get_existing_ids()
    primary_role = _champion_primary_role(champion)
    role_term = primary_role if primary_role != "unknown" else ""
    query = f"league of legends {champion} {role_term} guide".strip()
    results = search_youtube(query)
    filtered = filter_results(results, existing, limit)

    inserted = 0
    for v in filtered:
        role = _detect_role_from_title(v["title"]) or primary_role
        dur_str = f"{v['duration']//60}m" if v["duration"] else "?m"
        print(f"  [{'dry' if dry_run else 'save'}] {v['video_id']} | {dur_str} | {v['title'][:70]}")
        if not dry_run:
            insert_video(
                video_id=v["video_id"],
                video_url=v["url"],
                role=role,
                message_timestamp="",
                video_title=v["title"],
                description=v["title"],
                champion=champion,
                source="youtube_guide",
            )
        inserted += 1

    return inserted


def print_status() -> None:
    with get_connection() as conn:
        n_videos = conn.execute(
            "SELECT COUNT(*) FROM videos WHERE source = 'youtube_guide'"
        ).fetchone()[0]
        n_champs = conn.execute(
            "SELECT COUNT(DISTINCT champion) FROM videos WHERE source = 'youtube_guide'"
        ).fetchone()[0]
        n_pending = conn.execute(
            "SELECT COUNT(*) FROM videos WHERE source = 'youtube_guide' AND status = 'pending'"
        ).fetchone()[0]
    print(f"\nyoutube_guide: {n_videos} videos across {n_champs} champions ({n_pending} pending transcription)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape YouTube guide videos per champion")
    parser.add_argument("--champion", help="Single champion to scrape")
    parser.add_argument("--limit", type=int, default=5, help="Max videos per champion (default 5)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without inserting")
    parser.add_argument("--status", action="store_true", help="Show DB counts and exit")
    parser.add_argument("--delay-min", type=float, default=15.0,
                        help="Min seconds to sleep between champions (default 15)")
    parser.add_argument("--delay-max", type=float, default=30.0,
                        help="Max seconds to sleep between champions (default 30)")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip champions that already have >= limit videos in the DB")
    args = parser.parse_args()

    init_db()

    if args.status:
        print_status()
        return

    if args.champion:
        champions = [args.champion]
    else:
        champions = sorted(load_champion_names())

    skipped = total = 0
    for i, champion in enumerate(champions, 1):
        if args.skip_done and not args.dry_run:
            existing_count = _champion_video_count(champion)
            if existing_count >= args.limit:
                print(f"\n[{i}/{len(champions)}] {champion} — skip ({existing_count} already)")
                skipped += 1
                continue

        print(f"\n[{i}/{len(champions)}] {champion}")
        n = scrape_champion(champion, limit=args.limit, dry_run=args.dry_run)
        total += n
        if not args.dry_run and i < len(champions):
            delay = random.uniform(args.delay_min, args.delay_max)
            print(f"  Waiting {delay:.1f}s…")
            time.sleep(delay)

    action = "would insert" if args.dry_run else "inserted"
    print(f"\nDone — {action} {total} videos across {len(champions)} champion(s). ({skipped} skipped)")
    if not args.dry_run:
        print_status()


if __name__ == "__main__":
    main()
