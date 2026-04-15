"""
Fetches transcripts for each video using YouTube's auto-generated captions.
Falls back to yt-dlp subtitle download if the transcript API fails.

Run after scraping:
    python transcribe.py

Safe to re-run — already-transcribed videos are skipped.
"""

import os
import pathlib
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from database import get_videos_by_status, set_status, set_transcription

SUBTITLE_DIR = pathlib.Path("subtitles")
SUBTITLE_DIR.mkdir(exist_ok=True)


def fetch_via_transcript_api(video_id: str) -> str | None:
    """
    Pull auto-generated or manual captions from YouTube.
    Returns joined transcript string, or None if unavailable.
    """
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(["en"])

        segments = transcript.fetch()
        # Join all segments; each segment has 'text', 'start', 'duration'
        text = " ".join(seg.text.strip() for seg in segments)
        return text

    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        print(f"    [transcript api] unavailable: {e}")
        return None
    except Exception as e:
        print(f"    [transcript api] unexpected error: {e}")
        return None


def fetch_via_yt_dlp(video_id: str, video_url: str) -> str | None:
    """
    Fallback: download auto-subtitle .vtt file with yt-dlp and parse it.
    """
    out_template = str(SUBTITLE_DIR / f"{video_id}.%(ext)s")

    ydl_opts = {
        "skip_download": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "vtt",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        print(f"    [yt-dlp] download error: {e}")
        return None

    vtt_path = SUBTITLE_DIR / f"{video_id}.en.vtt"
    if not vtt_path.exists():
        print("    [yt-dlp] no subtitle file found after download")
        return None

    return parse_vtt(vtt_path)


def parse_vtt(path: pathlib.Path) -> str:
    """Strip VTT timestamps and tags, return plain text."""
    import re

    lines = path.read_text(encoding="utf-8").splitlines()
    text_lines = []
    for line in lines:
        # Skip header, timestamp lines, and empty lines
        if line.startswith("WEBVTT") or "-->" in line or not line.strip():
            continue
        # Strip inline tags like <c>, <00:00:01.000>
        clean = re.sub(r"<[^>]+>", "", line).strip()
        if clean:
            text_lines.append(clean)

    # Deduplicate consecutive duplicate lines (VTT often repeats lines)
    deduped = []
    for line in text_lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)

    return " ".join(deduped)


def run() -> None:
    videos = get_videos_by_status("pending")
    print(f"Videos to transcribe: {len(videos)}")

    ok = failed = 0

    for video in videos:
        video_id = video["video_id"]
        video_url = video["video_url"]
        role = video["role"]
        print(f"\n[{role}] {video_id}")

        # Try YouTube transcript API first (fast, no download)
        print("  Trying YouTube transcript API…")
        transcript = fetch_via_transcript_api(video_id)

        # Fallback to yt-dlp subtitle download
        if not transcript:
            print("  Falling back to yt-dlp subtitle download…")
            transcript = fetch_via_yt_dlp(video_id, video_url)

        if transcript:
            set_transcription(video_id, transcript)
            word_count = len(transcript.split())
            print(f"  Saved — {word_count:,} words")
            ok += 1
        else:
            set_status(video_id, "no_transcript")
            print("  No transcript available, skipping.")
            failed += 1

    print(f"\nDone. {ok} transcribed, {failed} skipped (no captions).")


if __name__ == "__main__":
    run()
