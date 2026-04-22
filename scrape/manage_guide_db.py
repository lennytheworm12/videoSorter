"""
Review and remove videos from guide_test.db.

Usage:
    uv run python -m scrape.manage_guide_db --list              # all videos grouped by champion
    uv run python -m scrape.manage_guide_db --list Aatrox       # one champion
    uv run python -m scrape.manage_guide_db --remove VIDEO_ID   # delete a video + its insights
    uv run python -m scrape.manage_guide_db --interactive       # step through each champion
"""

import os, argparse
os.environ.setdefault("DB_PATH", "guide_test.db")

from core.database import get_connection, init_db


def list_videos(champion: str | None = None) -> None:
    with get_connection() as conn:
        if champion:
            rows = conn.execute(
                """
                SELECT v.video_id, v.champion, v.role, v.status, v.video_title,
                       COUNT(i.id) AS n_insights
                FROM videos v
                LEFT JOIN insights i ON i.video_id = v.video_id
                WHERE v.source IN ('youtube_guide', 'mobafire_guide') AND v.champion = ?
                GROUP BY v.video_id
                ORDER BY v.champion, v.video_title
                """,
                (champion,)
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT v.video_id, v.champion, v.role, v.status, v.source, v.video_title,
                       COUNT(i.id) AS n_insights
                FROM videos v
                LEFT JOIN insights i ON i.video_id = v.video_id
                WHERE v.source IN ('youtube_guide', 'mobafire_guide')
                GROUP BY v.video_id
                ORDER BY v.champion, v.video_title
                """
            ).fetchall()

    current = None
    for r in rows:
        if r["champion"] != current:
            current = r["champion"]
            print(f"\n── {current} ──")
        insights = f" [{r['n_insights']} insights]" if r["n_insights"] else ""
        src = r["source"] if "source" in r.keys() else "youtube_guide"
        print(f"  {r['video_id']}  [{src} | {r['status']}]{insights}  {r['video_title'] or '(no title)'}")

    print(f"\nTotal: {len(rows)} videos")


def remove_video(video_id: str) -> None:
    with get_connection() as conn:
        n_insights = conn.execute(
            "DELETE FROM insights WHERE video_id = ?", (video_id,)
        ).rowcount
        n_videos = conn.execute(
            "DELETE FROM videos WHERE video_id = ?", (video_id,)
        ).rowcount
        conn.commit()
    if n_videos:
        print(f"Removed {video_id} ({n_insights} insights deleted)")
    else:
        print(f"Not found: {video_id}")


_BACK = object()   # sentinel: go back one champion
_SKIP = object()   # sentinel: skip to next champion
_QUIT = object()   # sentinel: quit


def review_champion(champion: str, index: int, total: int):
    """
    Step through each video for a champion one at a time.
    Returns _BACK, _SKIP, or _QUIT based on user input; None on normal completion.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT v.video_id, v.role, v.status, v.video_title, v.video_url,
                   COUNT(i.id) AS n_insights
            FROM videos v
            LEFT JOIN insights i ON i.video_id = v.video_id
            WHERE v.source IN ('youtube_guide', 'mobafire_guide') AND v.champion = ?
            GROUP BY v.video_id
            ORDER BY v.video_title
            """,
            (champion,)
        ).fetchall()

    print(f"\n{'='*60}")
    print(f"  [{index}/{total}] {champion}  ({len(rows)} video{'s' if len(rows) != 1 else ''})")
    print(f"  Controls per video: y=remove  n/Enter=keep  s=skip champion  b=back  q=quit")
    print(f"{'='*60}")

    if not rows:
        print("  (no videos)")
        return None

    removed = 0
    i = 0
    while i < len(rows):
        r = rows[i]
        insights = f" [{r['n_insights']} insights]" if r["n_insights"] else ""
        url = r["video_url"] or ""
        print(f"\n  ({i+1}/{len(rows)}) [{r['status']}]{insights}")
        print(f"  {r['video_title'] or '(no title)'}")
        print(f"  {url}")
        print(f"  ID: {r['video_id']}")
        ans = input("  > ").strip().lower()

        if ans == "q":
            return _QUIT
        if ans == "b":
            if i > 0:
                # undo last removal isn't possible, just go back in video list
                i -= 1
            else:
                return _BACK
            continue
        if ans == "s":
            return _SKIP
        if ans == "y":
            remove_video(r["video_id"])
            removed += 1
        i += 1

    print(f"  → {len(rows) - removed} kept, {removed} removed")
    return None


def interactive(start_champion: str | None = None) -> None:
    with get_connection() as conn:
        champions = [r[0] for r in conn.execute(
            "SELECT DISTINCT champion FROM videos WHERE source IN ('youtube_guide', 'mobafire_guide') ORDER BY champion"
        ).fetchall()]

    if not champions:
        print("No champions in guide_test.db.")
        return

    idx = 0
    if start_champion:
        names = [c.lower() for c in champions]
        if start_champion.lower() in names:
            idx = names.index(start_champion.lower())
        else:
            print(f"Champion '{start_champion}' not found — starting from beginning.")

    total = len(champions)
    while 0 <= idx < total:
        result = review_champion(champions[idx], idx + 1, total)
        if result is _QUIT:
            print("\nStopped.")
            return
        if result is _BACK:
            idx = max(0, idx - 1)
        else:
            idx += 1

    print("\nAll champions reviewed.")


def bulk_remove_by_keyword(keyword: str) -> None:
    """Remove all guide videos whose title contains keyword (case-insensitive)."""
    kw = keyword.lower()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT video_id, video_title FROM videos WHERE source IN ('youtube_guide', 'mobafire_guide')",
        ).fetchall()
        matched = [r for r in rows if kw in (r["video_title"] or "").lower()]
        if not matched:
            print("No matches.")
            return
        print(f"Matches ({len(matched)}):")
        for r in matched:
            print(f"  {r['video_id']}  {r['video_title']}")
        ans = input(f"\nRemove all {len(matched)}? [y/N] ").strip().lower()
        if ans == "y":
            for r in matched:
                remove_video(r["video_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage guide_test.db videos")
    parser.add_argument("--list", nargs="?", const="ALL", metavar="CHAMPION",
                        help="List videos (optionally for one champion)")
    parser.add_argument("--remove", metavar="VIDEO_ID", help="Remove a video and its insights")
    parser.add_argument("--interactive", action="store_true",
                        help="Step through each champion for review")
    parser.add_argument("--start", metavar="CHAMPION",
                        help="Champion to start --interactive from (resume point)")
    parser.add_argument("--review", metavar="CHAMPION",
                        help="Review videos for one champion one-at-a-time")
    parser.add_argument("--bulk-remove", metavar="KEYWORD",
                        help="Remove all guide videos whose title contains KEYWORD")
    args = parser.parse_args()

    init_db()

    if args.list is not None:
        champ = None if args.list == "ALL" else args.list
        list_videos(champ)
    elif args.remove:
        remove_video(args.remove)
    elif args.review:
        review_champion(args.review)
    elif args.bulk_remove:
        bulk_remove_by_keyword(args.bulk_remove)
    elif args.interactive:
        interactive(start_champion=args.start)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
