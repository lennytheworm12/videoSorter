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
                WHERE v.source = 'youtube_guide' AND v.champion = ?
                GROUP BY v.video_id
                ORDER BY v.champion, v.video_title
                """,
                (champion,)
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT v.video_id, v.champion, v.role, v.status, v.video_title,
                       COUNT(i.id) AS n_insights
                FROM videos v
                LEFT JOIN insights i ON i.video_id = v.video_id
                WHERE v.source = 'youtube_guide'
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
        print(f"  {r['video_id']}  [{r['status']}]{insights}  {r['video_title'] or '(no title)'}")

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


def interactive() -> None:
    with get_connection() as conn:
        champions = [r[0] for r in conn.execute(
            "SELECT DISTINCT champion FROM videos WHERE source='youtube_guide' ORDER BY champion"
        ).fetchall()]

    for champion in champions:
        print(f"\n{'='*60}")
        list_videos(champion)
        ids_to_remove = input("\nEnter video IDs to remove (space-separated), or Enter to skip: ").strip()
        for vid_id in ids_to_remove.split():
            remove_video(vid_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage guide_test.db videos")
    parser.add_argument("--list", nargs="?", const="ALL", metavar="CHAMPION",
                        help="List videos (optionally for one champion)")
    parser.add_argument("--remove", metavar="VIDEO_ID", help="Remove a video and its insights")
    parser.add_argument("--interactive", action="store_true",
                        help="Step through each champion for review")
    args = parser.parse_args()

    init_db()

    if args.list is not None:
        champ = None if args.list == "ALL" else args.list
        list_videos(champ)
    elif args.remove:
        remove_video(args.remove)
    elif args.interactive:
        interactive()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
