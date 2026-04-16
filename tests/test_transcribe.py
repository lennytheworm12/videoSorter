"""
Quick test of the YouTube transcript API on a handful of video IDs.

Usage:
    # Test with specific video IDs
    python test_transcribe.py dQw4w9WgXcQ abc123xyz11

    # Test with full YouTube URLs
    python test_transcribe.py https://www.youtube.com/watch?v=dQw4w9WgXcQ

    # Test videos already in the DB (picks first 3 per role)
    python test_transcribe.py --from-db
"""

import sys
import re
import pathlib
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable


YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")

PREVIEW_WORDS = 100  # how many words of transcript to print as preview


def extract_id(raw: str) -> str:
    """Accept either an 11-char ID or a full YouTube URL."""
    if len(raw) == 11 and re.match(r"^[A-Za-z0-9_-]{11}$", raw):
        return raw
    match = YOUTUBE_ID_RE.search(raw)
    if match:
        return match.group(1)
    raise ValueError(f"Could not parse video ID from: {raw!r}")


def test_video(video_id: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"Video ID : {video_id}")
    print(f"URL      : https://www.youtube.com/watch?v={video_id}")

    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        # Show what caption tracks are available
        print("Captions available:")
        for t in transcript_list:
            kind = "auto-generated" if t.is_generated else "manual"
            print(f"  [{kind}] {t.language} ({t.language_code})")

        # Fetch the best English track
        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
            source = "manual"
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(["en"])
            source = "auto-generated"

        segments = transcript.fetch()
        full_text = " ".join(seg.text.strip() for seg in segments)
        words = full_text.split()

        print(f"\nSource   : {source}")
        print(f"Segments : {len(segments)}")
        print(f"Words    : {len(words):,}")
        print(f"\nPreview ({PREVIEW_WORDS} words):")
        print("  " + " ".join(words[:PREVIEW_WORDS]) + ("…" if len(words) > PREVIEW_WORDS else ""))

    except TranscriptsDisabled:
        print("RESULT   : FAIL — transcripts are disabled for this video")
    except NoTranscriptFound:
        print("RESULT   : FAIL — no English transcript found")
    except VideoUnavailable:
        print("RESULT   : FAIL — video is unavailable or private")
    except Exception as e:
        print(f"RESULT   : ERROR — {e}")


def test_from_db(limit_per_role: int = 3) -> None:
    try:
        from database import get_connection
    except ImportError:
        print("database.py not found")
        return

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT video_id, role, video_url
            FROM videos
            WHERE status = 'pending'
            GROUP BY role
            ORDER BY role, rowid
        """).fetchall()

    if not rows:
        print("No pending videos in DB yet. Run main.py first to scrape Discord threads.")
        return

    # Take up to limit_per_role per role
    seen: dict[str, int] = {}
    selected = []
    for row in rows:
        role = row["role"]
        if seen.get(role, 0) < limit_per_role:
            selected.append(row)
            seen[role] = seen.get(role, 0) + 1

    print(f"Testing {len(selected)} videos from DB ({limit_per_role} per role max)…")
    for row in selected:
        print(f"\n  Role: {row['role']}")
        test_video(row["video_id"])


def main() -> None:
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(0)

    if args == ["--from-db"]:
        test_from_db()
        return

    video_ids = []
    for arg in args:
        try:
            video_ids.append(extract_id(arg))
        except ValueError as e:
            print(f"Skipping invalid argument: {e}")

    if not video_ids:
        print("No valid video IDs provided.")
        sys.exit(1)

    print(f"Testing {len(video_ids)} video(s)…")
    for vid in video_ids:
        test_video(vid)

    print(f"\n{'─' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
