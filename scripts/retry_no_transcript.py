"""
Retry transcription for videos marked no_transcript.

Some may have failed due to proxy errors or rate limits rather than
genuinely having no captions. Resets them to pending and retries.

Usage:
    uv run python -m scripts.retry_no_transcript            # all roles
    uv run python -m scripts.retry_no_transcript --role top # one role
    uv run python -m scripts.retry_no_transcript --dry-run  # show counts without retrying
"""

import argparse
import time
from core.database import get_connection, set_status, set_transcription
from pipeline.transcribe import fetch_via_transcript_api, fetch_via_yt_dlp, INTER_VIDEO_DELAY

ROLES = ["top", "jungle", "mid", "adc", "support"]


def get_no_transcript_videos(roles: list[str]) -> list:
    placeholders = ",".join("?" * len(roles))
    with get_connection() as conn:
        return conn.execute(
            f"SELECT * FROM videos WHERE status = 'no_transcript' AND role IN ({placeholders})",
            roles,
        ).fetchall()


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry no_transcript videos")
    parser.add_argument("--role", choices=ROLES, nargs="+", metavar="ROLE")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without retrying")
    args = parser.parse_args()

    roles = args.role or ROLES
    videos = get_no_transcript_videos(roles)

    if not videos:
        print("No no_transcript videos found.")
        return

    print(f"Found {len(videos)} no_transcript video(s):\n")
    by_role: dict[str, list] = {}
    for v in videos:
        by_role.setdefault(v["role"], []).append(v)
    for role, vids in by_role.items():
        print(f"  {role}: {len(vids)}")

    if args.dry_run:
        return

    print()
    ok = failed = 0
    for i, video in enumerate(videos, 1):
        vid_id = video["video_id"]
        desc = (video["description"] or "(no desc)")[:55]
        print(f"[{i:>3}/{len(videos)}] {vid_id}  {desc}")

        transcript = fetch_via_transcript_api(vid_id)
        if not transcript:
            transcript = fetch_via_yt_dlp(vid_id, video["video_url"])

        if transcript:
            set_transcription(vid_id, transcript)
            words = len(transcript.split())
            print(f"  ✓ {words:,} words")
            ok += 1
            time.sleep(INTER_VIDEO_DELAY)
        else:
            print(f"  ✗ still no transcript")
            failed += 1

    print(f"\nDone — {ok} recovered, {failed} still unavailable.")


if __name__ == "__main__":
    main()
