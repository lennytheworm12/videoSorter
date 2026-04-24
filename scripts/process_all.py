"""
Full pipeline: transcribe + analyze every scraped video, one at a time.

Progress is printed after each video. Safe to interrupt and resume at any
time — the video status in the DB (pending → transcribed → analyzed) is the
checkpoint. Re-running picks up exactly where it left off with no
reprocessing.

Usage:
    uv run python -m scripts.process_all            # all roles
    uv run python -m scripts.process_all --role mid # one role only
    uv run python -m scripts.process_all --status   # print counts and exit
"""

import argparse
import logging
import random
import sys
import time

# Suppress noisy model-load warnings
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

from core.database import (
    get_connection, get_videos_by_status,
    set_status, set_transcription, insert_insight,
)
from pipeline.transcribe import fetch_via_transcript_api, fetch_via_yt_dlp, INTER_VIDEO_DELAY
from pipeline.analyze import (
    extract_insights_from_chunk, chunk_transcript,
    _embed_chunk_windows, score_source_grounding,
    already_analyzed,
)
from core.champions import correct_names, champion_names_for_prompt

ROLES = ["top", "jungle", "mid", "adc", "support"]


# ── helpers ───────────────────────────────────────────────────────────────────

def counts_by_role() -> dict[str, dict[str, int]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, status, COUNT(*) as n FROM videos GROUP BY role, status"
        ).fetchall()
    result: dict[str, dict[str, int]] = {}
    for r in rows:
        result.setdefault(r["role"], {})[r["status"]] = r["n"]
    return result


def print_status() -> None:
    data = counts_by_role()
    print(f"\n{'Role':<10} {'pending':>8} {'transcribed':>12} {'analyzed':>9} {'no_transcript':>14}")
    print("-" * 58)
    for role in ROLES:
        d = data.get(role, {})
        print(
            f"{role:<10}"
            f"{d.get('pending', 0):>8}"
            f"{d.get('transcribed', 0):>12}"
            f"{d.get('analyzed', 0):>9}"
            f"{d.get('no_transcript', 0):>14}"
        )
    print()


# ── per-video pipeline ────────────────────────────────────────────────────────

def transcribe_video(video: dict) -> bool:
    """Transcribe one video. Returns True if transcript was obtained."""
    video_id = video["video_id"]
    video_url = video["video_url"]

    transcript = fetch_via_transcript_api(video_id)
    if not transcript:
        transcript = fetch_via_yt_dlp(video_id, video_url)

    if transcript:
        set_transcription(video_id, transcript)
        delay = random.uniform(INTER_VIDEO_DELAY, INTER_VIDEO_DELAY + 10)
        time.sleep(delay)
        return True
    else:
        set_status(video_id, "no_transcript")
        return False


def analyze_video(video: dict) -> tuple[int, int]:
    """
    Analyze one transcribed video. Returns (total_insights, flagged_count).
    Flagged = source_score < 0.30 (likely hallucinated).
    """
    video_id   = video["video_id"]
    role       = video["role"]
    champion   = video["champion"]
    description = video["description"]
    transcript = video["transcription"]

    if already_analyzed(video_id):
        set_status(video_id, "analyzed")
        return 0, 0

    chunks = chunk_transcript(transcript)
    aggregated: dict[str, list[tuple[str, float]]] = {}

    for i, chunk in enumerate(chunks):
        print(f"    chunk {i + 1}/{len(chunks)}…", flush=True)
        try:
            result = extract_insights_from_chunk(
                chunk, role, champion, description
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        t_embed = time.time()
        window_matrix = _embed_chunk_windows(chunk)
        chunk_total = 0
        for insight_type, items in result.items():
            for text, emphasis in items:
                score = score_source_grounding(text, window_matrix)
                aggregated.setdefault(insight_type, []).append((text, score, emphasis))
                chunk_total += 1
        print(f"embed: {time.time() - t_embed:.2f}s → {chunk_total} insights")

    total = flagged = 0
    for insight_type, items in aggregated.items():
        for text, source_score, emphasis in items:
            insert_insight(video_id, insight_type, text, source_score, repetition_count=emphasis)
            total += 1
            if source_score < 0.30:
                flagged += 1

    set_status(video_id, "analyzed")
    return total, flagged


# ── main loop ─────────────────────────────────────────────────────────────────

def process_role(role: str, videos: list) -> None:
    pending     = [v for v in videos if v["status"] == "pending"]
    transcribed = [v for v in videos if v["status"] == "transcribed"]
    analyzed    = [v for v in videos if v["status"] == "analyzed"]
    total       = len(videos)
    done        = len(analyzed)

    print(f"\n{'═' * 60}")
    print(f"  {role.upper()}  —  {total} videos  "
          f"({done} analyzed, {len(transcribed)} transcribed, {len(pending)} pending)")
    print(f"{'═' * 60}")

    # Work through pending (needs transcribe + analyze) then transcribed (needs analyze)
    work_queue = pending + transcribed
    for idx, video in enumerate(work_queue, start=done + 1):
        vid_id = video["video_id"]
        desc   = (video["description"] or "(no desc)")[:55]

        print(f"\n[{idx:>3}/{total}] {vid_id}  {desc}")

        # Step 1 — transcribe if still pending
        if video["status"] == "pending":
            print("  → transcribing…", end=" ", flush=True)
            ok = transcribe_video(video)
            if not ok:
                print("no transcript available, skipping")
                continue

            # Reload to get transcription text
            with get_connection() as conn:
                video = conn.execute(
                    "SELECT * FROM videos WHERE video_id = ?", (vid_id,)
                ).fetchone()
            words = len((video["transcription"] or "").split())
            print(f"{words:,} words")

        # Step 2 — analyze
        print("  → analyzing…")
        n_insights, n_flagged = analyze_video(video)
        flag_note = f", {n_flagged} low-confidence" if n_flagged else ""
        print(f"  ✓ {n_insights} insights saved{flag_note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe + analyze all videos")
    parser.add_argument("--role", choices=ROLES, nargs="+", metavar="ROLE",
                        help="One or more roles to process (e.g. --role support top)")
    parser.add_argument("--video", metavar="VIDEO_ID", help="Process a single video by ID (resets it first if already analyzed)")
    parser.add_argument("--status", action="store_true", help="Print counts and exit")
    parser.add_argument(
        "--reanalyze",
        action="store_true",
        help="Re-analyze already-analyzed videos (deletes old insights, keeps transcripts)",
    )
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    # ── single-video mode ──────────────────────────────────────────────────────
    if args.video:
        vid_id = args.video
        with get_connection() as conn:
            video = conn.execute(
                "SELECT * FROM videos WHERE video_id = ?", (vid_id,)
            ).fetchone()
        if not video:
            print(f"Video not found: {vid_id}")
            return
        # Reset so it gets reprocessed regardless of current status
        with get_connection() as conn:
            conn.execute("DELETE FROM insights WHERE video_id = ?", (vid_id,))
            conn.execute(
                "UPDATE videos SET status = 'transcribed' WHERE video_id = ? AND status = 'analyzed'",
                (vid_id,),
            )
            conn.commit()
        # Reload after potential status change
        with get_connection() as conn:
            video = conn.execute(
                "SELECT * FROM videos WHERE video_id = ?", (vid_id,)
            ).fetchone()
        print(f"\nSingle-video mode: {vid_id}  ({video['role']} | {video['champion'] or '?'})")
        print(f"Status: {video['status']}\n")
        if video["status"] == "pending":
            print("  → transcribing…", end=" ", flush=True)
            ok = transcribe_video(video)
            if not ok:
                print("no transcript available")
                return
            with get_connection() as conn:
                video = conn.execute(
                    "SELECT * FROM videos WHERE video_id = ?", (vid_id,)
                ).fetchone()
            print(f"{len((video['transcription'] or '').split()):,} words")
        print("  → analyzing…")
        n_insights, n_flagged = analyze_video(video)
        flag_note = f", {n_flagged} low-confidence" if n_flagged else ""
        print(f"  ✓ {n_insights} insights saved{flag_note}")
        return

    roles_to_run = args.role if args.role else ROLES

    if args.reanalyze:
        with get_connection() as conn:
            placeholders = ",".join("?" * len(roles_to_run))
            rows = conn.execute(
                f"SELECT video_id FROM videos WHERE role IN ({placeholders}) AND status = 'analyzed'",
                roles_to_run,
            ).fetchall()
            ids = [r["video_id"] for r in rows]
        if not ids:
            print("No analyzed videos to reset.")
        else:
            print(f"Resetting {len(ids)} analyzed videos (keeping transcripts)…")
            with get_connection() as conn:
                conn.executemany(
                    "DELETE FROM insights WHERE video_id = ?", [(i,) for i in ids]
                )
                conn.executemany(
                    "UPDATE videos SET status = 'transcribed' WHERE video_id = ?",
                    [(i,) for i in ids],
                )
                conn.commit()
            print(f"Done — {len(ids)} videos reset to 'transcribed', insights cleared.\n")

    # Load all relevant videos ordered by role then timestamp
    with get_connection() as conn:
        placeholders = ",".join("?" * len(roles_to_run))
        all_videos = conn.execute(
            f"SELECT * FROM videos WHERE role IN ({placeholders})"
            f"  AND status NOT IN ('no_transcript')"
            f"  ORDER BY role, message_timestamp",
            roles_to_run,
        ).fetchall()

    by_role: dict[str, list] = {r: [] for r in roles_to_run}
    for v in all_videos:
        by_role[v["role"]].append(v)

    print_status()

    for role in roles_to_run:
        videos = by_role[role]
        if not videos:
            print(f"\n[{role}] nothing to process")
            continue
        process_role(role, videos)

    print("\n" + "═" * 60)
    print("All done. Run embed.py then score_clusters.py next.")
    print_status()


if __name__ == "__main__":
    main()
