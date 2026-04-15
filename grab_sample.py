"""One-off script to grab a few YouTube links from the mid lane thread for testing."""

import re
import os
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_RE = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com/watch\?(?:[^&\s]*&)*v=|youtu\.be/)([A-Za-z0-9_-]{11})"
)

def grab_sample(url: str, max_videos: int = 5) -> list[dict]:
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state="auth_state.json")
        page = context.new_page()

        print(f"Navigating to mid lane thread…")
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(3000)

        # Scroll up a few times to load some messages
        print("Loading messages…")
        for i in range(8):
            page.evaluate("""
                const el = document.querySelector("ol[data-list-id='chat-messages']");
                if (el) el.scrollTop = 0;
            """)
            page.wait_for_timeout(1500)

        messages = page.query_selector_all("li[id^='chat-messages-']")
        print(f"Found {len(messages)} message elements, scanning for YouTube links…")

        for msg_el in messages:
            if len(results) >= max_videos:
                break

            content_el = msg_el.query_selector("[class*='messageContent']")
            if not content_el:
                continue
            text = content_el.inner_text().strip()

            match = YOUTUBE_RE.search(text)
            if match:
                video_id = match.group(1)
                video_url = match.group(0)
                description = text.replace(video_url, "").strip()
                results.append({
                    "video_id": video_id,
                    "video_url": video_url,
                    "description": description or None,
                })
                print(f"  Found: {video_id} | {description[:60] if description else '(no desc)'}")

        context.close()
        browser.close()

    return results


if __name__ == "__main__":
    mid_url = os.environ.get("MID_URL")
    if not mid_url:
        print("MID_URL not set in .env")
        exit(1)

    videos = grab_sample(mid_url, max_videos=5)

    if not videos:
        print("No YouTube links found.")
        exit(1)

    print(f"\nGrabbed {len(videos)} videos. Running transcript test…\n")

    # Run the transcript test inline
    from test_transcribe import test_video
    for v in videos:
        test_video(v["video_id"])

    print("\nDone.")
