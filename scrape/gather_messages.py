"""Scrolls through a Discord thread channel and parses YouTube links from messages."""

import re
import time
from playwright.sync_api import BrowserContext, Page
from core.database import insert_video, insert_pending_description
from core.champions import extract_champion_from_title

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

    Messages are parsed incrementally during scrolling so that any messages
    Discord de-renders from the DOM (virtual scroll) are still captured.
    """
    page = context.new_page()
    page.goto(channel_url, wait_until="networkidle")
    page.wait_for_timeout(4000)  # give Discord extra time to hydrate

    message_list_selector = "ol[data-list-id='chat-messages']"
    page.wait_for_selector(message_list_selector, timeout=15000)
    page.wait_for_timeout(2000)  # wait for initial messages to render

    seen_msg_ids: set[str] = set()
    saved = pending = 0

    def flush_visible(label: str = "") -> None:
        """Parse and persist all currently visible messages not yet seen."""
        nonlocal saved, pending
        messages = page.query_selector_all("li[id^='chat-messages-']")
        for msg_el in messages:
            msg_id = msg_el.get_attribute("id") or ""
            if msg_id in seen_msg_ids:
                continue
            seen_msg_ids.add(msg_id)

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
                    champion = extract_champion_from_title(description)
                    insert_video(
                        video_id=video_id,
                        video_url=youtube_url,
                        role=role,
                        message_timestamp=timestamp or "",
                        video_title=description,
                        description=description,
                        champion=champion,
                    )
                    saved += 1
                    print(f"  [saved] {video_id} | {description or '(no desc)'}")
            else:
                if timestamp:
                    insert_pending_description(
                        role=role, description=text, message_timestamp=timestamp
                    )
                    pending += 1

    print(f"[{role}] Scrolling to load all messages…")
    _load_all_messages(page, flush_visible)

    # Final flush for anything visible after scrolling completes
    flush_visible("final")

    print(f"[{role}] Done — {saved} videos saved, {pending} pending descriptions")
    page.close()


def _scroller_state(page: Page) -> dict:
    """Return scrollTop and scrollHeight of the main chat scroller."""
    return page.evaluate("""
        () => {
            const el = document.querySelector('[class*="scroller__36d07"]')
                     || document.querySelector('[class*="scroller"][class*="auto"]');
            return el
                ? { scrollTop: el.scrollTop, scrollHeight: el.scrollHeight,
                    clientHeight: el.clientHeight }
                : { scrollTop: 0, scrollHeight: 0, clientHeight: 0 };
        }
    """)


def _load_all_messages(page: Page, flush_fn, max_attempts: int = 400) -> None:
    """
    Load every message in a Discord thread channel using a two-phase strategy,
    flushing visible messages to the DB after each scroll step.

    Phase 1 — scroll DOWN: some channels start at the oldest message (scrollTop=0)
    and load newer ones as you advance toward the bottom.

    Phase 2 — scroll UP: other channels start at the newest message (bottom) and
    load older ones as you scroll up. Also catches any remaining old messages for
    channels that were already fully loaded by Phase 1.

    Termination in each phase is driven by scrollTop position (are we actually at
    the edge?) rather than just message-count stability, so virtual-DOM channels
    (where Discord de-renders messages to save memory) don't stop early.
    """
    try:
        page.click("ol[data-list-id='chat-messages']", timeout=3000)
    except Exception:
        pass

    half = max_attempts // 2

    def _scroll_down_step() -> None:
        page.evaluate("""
            () => {
                document.querySelectorAll('[class*="scroller"]').forEach(el => {
                    if (el.scrollHeight > el.clientHeight)
                        el.scrollTop = el.scrollHeight;
                });
            }
        """)
        page.keyboard.press("End")

    def _scroll_up_step() -> None:
        page.evaluate("""
            () => {
                document.querySelectorAll('[class*="scroller"]').forEach(el => {
                    if (el.scrollHeight > el.clientHeight)
                        el.scrollTop = Math.max(0, el.scrollTop - 2000);
                });
            }
        """)
        page.keyboard.press("PageUp")

    # ── Phase 1: scroll to bottom ──────────────────────────────────────────
    prev_count = 0
    stale_rounds = 0
    for attempt in range(half):
        _scroll_down_step()
        time.sleep(2.5)
        flush_fn()

        state = _scroller_state(page)
        at_bottom = state["scrollHeight"] - state["clientHeight"] - state["scrollTop"] < 20

        current_count = page.eval_on_selector_all(
            "li[id^='chat-messages-']", "els => els.length"
        )
        if current_count == prev_count:
            stale_rounds += 1
            if stale_rounds >= 4 and at_bottom:
                print(f"  → Phase 1 stable at bottom ({current_count} visible)")
                break
        else:
            stale_rounds = 0
            if attempt % 5 == 0 and attempt > 0:
                print(f"  → {current_count} messages visible (↓)…")
        prev_count = current_count

    # ── Phase 2: scroll to top ─────────────────────────────────────────────
    at_top_rounds = 0
    prev_count = 0
    stale_rounds = 0
    for attempt in range(half):
        _scroll_up_step()
        time.sleep(2.5)
        flush_fn()

        reached_beginning = page.evaluate(
            "() => !!document.querySelector('[class*=\"channelBeginning\"]')"
        )
        if reached_beginning:
            current_count = page.eval_on_selector_all(
                "li[id^='chat-messages-']", "els => els.length"
            )
            print(f"  → Reached beginning ({current_count} visible)")
            break

        state = _scroller_state(page)
        at_top = state["scrollTop"] <= 10

        if at_top:
            at_top_rounds += 1
            if at_top_rounds >= 3:
                current_count = page.eval_on_selector_all(
                    "li[id^='chat-messages-']", "els => els.length"
                )
                print(f"  → scrollTop=0 for {at_top_rounds} rounds, stopping ({current_count} visible)")
                break
        else:
            at_top_rounds = 0

        current_count = page.eval_on_selector_all(
            "li[id^='chat-messages-']", "els => els.length"
        )
        if current_count != prev_count:
            stale_rounds = 0
            if attempt % 5 == 0 and attempt > 0:
                print(f"  → {current_count} messages visible (↑)…")
        else:
            stale_rounds += 1
        prev_count = current_count
