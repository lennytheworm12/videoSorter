"""
videoSorter pipeline — scrape Discord threads, transcribe, extract insights.

Usage:
    python main.py                  # run all stages
    python main.py --scrape         # scrape Discord only
    python main.py --transcribe     # fetch transcripts only
    python main.py --analyze        # run LLM analysis only
"""

import sys
import os
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

from gather_messages import go_through_channel
from database import init_db, try_fill_descriptions

load_dotenv()

CHANNEL_KEYS = ["TOP_URL", "JUNGLE_URL", "MID_URL", "ADC_URL", "SUPPORT_URL"]


def stage_scrape() -> None:
    print("\n=== STAGE 1: Scraping Discord threads ===")

    role_url_map: dict[str, str] = {
        key.removesuffix("_URL").lower(): os.environ[key]
        for key in CHANNEL_KEYS
        if os.environ.get(key)
    }
    print(f"Channels: {list(role_url_map.keys())}")

    headless = os.environ.get("HEADLESS", "false").lower() == "true"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(storage_state="auth_state.json")

        for role, url in role_url_map.items():
            print(f"\n--- [{role.upper()}] ---")
            go_through_channel(context, url, role)

        context.close()
        browser.close()

    print("\nMatching pending descriptions to videos…")
    try_fill_descriptions()
    print("Scrape stage complete.")


def stage_transcribe() -> None:
    print("\n=== STAGE 2: Fetching transcripts ===")
    from transcribe import run as transcribe_run
    transcribe_run()


def stage_analyze() -> None:
    print("\n=== STAGE 3: Extracting insights with LLM ===")
    from analyze import run as analyze_run
    analyze_run()


def main() -> None:
    init_db()

    args = set(sys.argv[1:])
    run_all = not args

    if run_all or "--scrape" in args:
        stage_scrape()

    if run_all or "--transcribe" in args:
        stage_transcribe()

    if run_all or "--analyze" in args:
        stage_analyze()

    print("\nAll requested stages complete.")


if __name__ == "__main__":
    main()
