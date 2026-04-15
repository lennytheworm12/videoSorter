"""Scrolls through a Discord thread channel and parses YouTube links from messages."""

import re
import time
from playwright.sync_api import BrowserContext
from database import insert_video, insert_pending_description

# Matches the full YouTube URL including all query params (stops at whitespace)
# Group 1 = full URL, used for stripping from description text
YOUTUBE_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/\S+"
)

# Extracts the 11-char video ID from any YouTube URL form
_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def extract_video_id(url: str) -> str | None:
    match = _VIDEO_ID_RE.search(url)
    return match.group(1) if match else None


def clean_description(text: str) -> str | None:
    """
    Normalise whitespace/newlines and strip stray URL query fragments
    (e.g. '&feature=youtu.be' left on its own line after the URL is removed).
    Returns None if effectively empty.
    """
    lines = text.splitlines()
    filtered = [
        line for line in lines
        if not re.match(r"^\s*[&?][A-Za-z0-9_=%&.]+\s*$", line)
    ]
    cleaned = re.sub(r"\s+", " ", " ".join(filtered)).strip()
    return cleaned if cleaned else None


def parse_message(text: str) -> tuple[str | None, str | None]:
    """
    Return (youtube_url, description) from a raw Discord message.
    The full URL (including query params) is stripped before building the description.
    """
    match = YOUTUBE_URL_RE.search(text)
    if not match:
        return None, None

    full_url = match.group(0)
    before = text[: match.start()]
    after = text[match.end() :]
    description = clean_description(before + " " + after)
    return full_url, description


def go_through_channel(context: BrowserContext, channel_url: str, role: str) -> None:
    """
    Navigate to a Discord thread, scroll to load all messages,
    parse YouTube links, and store them in the DB.
    """
    page = context.new_page()
    page.goto(channel_url, wait_until="networkidle")
    page.wait_for_timeout(2000)

    message_list_selector = "ol[data-list-id='chat-messages']"
    page.wait_for_selector(message_list_selector, timeout=15000)

    print(f"[{role}] Scrolling to load all messages…")
    _scroll_to_top(page)

    messages = page.query_selector_all("li[id^='chat-messages-']")
    print(f"[{role}] Parsing {len(messages)} message elements")

    saved = pending = 0

    for msg_el in messages:
        time_el = msg_el.query_selector("time")
        timestamp = time_el.get_attribute("datetime") if time_el else None

        content_el = msg_el.query_selector("[class*='messageContent']")
        if not content_el:
            continue
        text = content_el.inner_text().strip()
        if not text:
            continue

        youtube_url, description = parse_message(text)

        if youtube_url:
            video_id = extract_video_id(youtube_url)
            if video_id:
                insert_video(
                    video_id=video_id,
                    video_url=youtube_url,
                    role=role,
                    message_timestamp=timestamp or "",
                    video_title=description,
                    description=description,
                )
                saved += 1
                print(f"  [saved] {video_id} | {description or '(no desc)'}")
        else:
            if timestamp:
                insert_pending_description(role=role, description=text, message_timestamp=timestamp)
                pending += 1

    print(f"[{role}] Done — {saved} videos saved, {pending} pending descriptions")
    page.close()


def _scroll_to_top(page, max_attempts: int = 150) -> None:
    """
    Scroll the Discord message list upward until no new messages load.
    Uses page key HOME as a fallback if scrollTop doesn't trigger loads.
    """
    prev_count = 0
    stale_rounds = 0

    for attempt in range(max_attempts):
        # Primary: set scrollTop to 0 on the message list
        page.evaluate("""
            const el = document.querySelector("ol[data-list-id='chat-messages']");
            if (el) el.scrollTop = 0;
        """)
        time.sleep(1.5)

        current_count = page.eval_on_selector_all(
            "li[id^='chat-messages-']", "els => els.length"
        )

        if current_count == prev_count:
            stale_rounds += 1
            if stale_rounds >= 4:
                print(f"  → Reached top after {attempt + 1} scroll attempts ({current_count} messages loaded)")
                break
        else:
            stale_rounds = 0
            if attempt % 10 == 0 and attempt > 0:
                print(f"  → {current_count} messages loaded…")

        prev_count = current_count
