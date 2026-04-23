"""
Fetches transcripts for each video using YouTube's auto-generated captions.
Falls back to yt-dlp subtitle download if the transcript API fails.

Run after scraping:
    python transcribe.py

Safe to re-run — already-transcribed videos are skipped.
"""

import os
import time
import random
import pathlib
import yt_dlp
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from core.database import get_videos_by_status, set_status, set_transcription
from core.youtube_network import YouTubeNetworkPolicy

load_dotenv()

SUBTITLE_DIR = pathlib.Path("subtitles")
SUBTITLE_DIR.mkdir(exist_ok=True)
from youtube_transcript_api.proxies import GenericProxyConfig

NETWORK_POLICY = YouTubeNetworkPolicy()
print(NETWORK_POLICY.describe())

# Random delay range between transcript fetches to avoid rate limiting.
# Override with TRANSCRIPT_DELAY_MIN / TRANSCRIPT_DELAY_MAX env vars.
INTER_VIDEO_DELAY_MIN = int(os.environ.get("TRANSCRIPT_DELAY_MIN", "40"))
INTER_VIDEO_DELAY_MAX = int(os.environ.get("TRANSCRIPT_DELAY_MAX", "60"))
INTER_VIDEO_DELAY = INTER_VIDEO_DELAY_MIN

def fetch_via_transcript_api(video_id: str) -> str | None:
    """
    Pull auto-generated or manual captions from YouTube using shared network policy.
    """
    def _operation(proxy_url: str | None) -> str:
        route = "proxy" if proxy_url else "local IP"
        proxy_config = (
            GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
            if proxy_url else None
        )
        api = YouTubeTranscriptApi(proxy_config=proxy_config)
        transcript_list = api.list(video_id)

        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
            caption_type = "manual"
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(["en"])
            caption_type = "auto-generated"

        segments = transcript.fetch()
        text = " ".join(seg.text.strip() for seg in segments)
        print(
            f"    [transcript api] fetched {caption_type} English captions via {route}",
            flush=True,
        )
        return text

    try:
        return NETWORK_POLICY.run("transcript api", _operation)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as exc:
        print(f"    [transcript api] unavailable: {exc}")
        return None
    except Exception as exc:
        print(f"    [transcript api] unexpected error: {exc}")
        return None


def fetch_via_yt_dlp(video_id: str, video_url: str) -> str | None:
    """
    Fallback: download auto-subtitle .vtt file with yt-dlp and parse it.
    Uses the same local-first network policy as the transcript API path.
    """
    out_template = str(SUBTITLE_DIR / f"{video_id}.%(ext)s")

    base_opts = {
        "skip_download": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "vtt",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "cookiefile": "cookies.txt",  # Netscape-format cookies exported from browser
    }

    def _operation(proxy_url: str | None) -> None:
        route = "proxy" if proxy_url else "local IP"
        ydl_opts = dict(base_opts)
        if proxy_url:
            ydl_opts["proxy"] = proxy_url
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        print(f"    [yt-dlp] downloaded subtitle track via {route}", flush=True)

    try:
        NETWORK_POLICY.run("yt-dlp", _operation)
    except Exception as exc:
        print(f"    [yt-dlp] download error: {exc}")
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


def run(
    sources: list[str] | tuple[str, ...] | None = None,
    game: str | None = None,
) -> None:
    videos = get_videos_by_status("pending")
    if sources:
        allowed = set(sources)
        videos = [video for video in videos if (video["source"] or "discord") in allowed]
    if game:
        videos = [video for video in videos if (video["game"] or "lol") == game]
    print(f"Videos to transcribe: {len(videos)}")

    ok = failed = 0

    for video in videos:
        video_id = video["video_id"]
        video_url = video["video_url"]
        role = video["role"]

        # Skip synthetic rows (ability enrichment, etc.) that have no real URL
        if not video_url or role == "ability_enrichment":
            continue

        title = video["video_title"] or video_id
        print(f"\n[{role}] {title[:80]}")

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

        delay = random.uniform(INTER_VIDEO_DELAY_MIN, INTER_VIDEO_DELAY_MAX)
        print(f"  Waiting {delay:.1f}s…")
        time.sleep(delay)

    print(f"\nDone. {ok} transcribed, {failed} skipped (no captions).")


if __name__ == "__main__":
    run()
