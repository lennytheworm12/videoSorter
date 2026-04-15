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
    page.wait_for_timeout(4000)  # give Discord extra time to hydrate

    message_list_selector = "ol[data-list-id='chat-messages']"
    page.wait_for_selector(message_list_selector, timeout=15000)
    page.wait_for_timeout(2000)  # wait for initial messages to render

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


def _scroll_to_top(page, max_attempts: int = 400) -> None:
    """
    Scroll the Discord message list upward until all messages are loaded.

    Scrolls incrementally by -1500px per step rather than jumping to 0 —
    incremental scrolling reliably triggers Discord's lazy-load whereas
    setting scrollTop=0 directly can bypass the IntersectionObserver.

    Stops when:
      - scrollTop reaches 0 AND message count is stable (truly at top), OR
      - A Discord "beginning of channel" marker is visible, OR
      - Message count has been stale for 8+ rounds (safety exit)
    """
    prev_count = 0
    stale_rounds = 0

    for attempt in range(max_attempts):
        # Scroll up by a chunk on the actual scrollable container.
        # Discord's ol[data-list-id] has no overflow — we need its closest
        # scrollable ancestor (div[class*="scroller"]).
        at_top = page.evaluate("""
            () => {
                const ol = document.querySelector("ol[data-list-id='chat-messages']");
                if (!ol) return true;
                // Walk up to find the scrollable parent
                let el = ol.parentElement;
                while (el && el !== document.body) {
                    if (el.scrollHeight > el.clientHeight) break;
                    el = el.parentElement;
                }
                if (!el || el === document.body) return true;
                el.scrollTop = Math.max(0, el.scrollTop - 1500);
                return el.scrollTop === 0;
            }
        """)

        # Wait for Discord to fetch and render older messages
        time.sleep(2.0)

        current_count = page.eval_on_selector_all(
            "li[id^='chat-messages-']", "els => els.length"
        )

        reached_beginning = page.evaluate("""
            () => !!document.querySelector(
                '[class*="channelBeginning"], [class*="firstMessage"], [class*="beginning-"], [class*="welcomeCta"]'
            )
        """)

        if reached_beginning:
            print(f"  → Reached beginning of channel ({current_count} messages loaded)")
            break

        if current_count == prev_count:
            stale_rounds += 1
            # Only stop if we're actually at the top scroll position
            if at_top and stale_rounds >= 3:
                print(f"  → At scroll top, no new messages ({current_count} loaded)")
                break
            # Safety exit if totally stuck regardless of position
            if stale_rounds >= 8:
                print(f"  → No new messages after {stale_rounds} attempts ({current_count} loaded)")
                break
        else:
            stale_rounds = 0
            if current_count % 50 == 0 or attempt % 20 == 0:
                print(f"  → {current_count} messages loaded…")

        prev_count = current_count
