"""Debug what Discord actually renders in the channel."""
import os, time
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
load_dotenv()

url = os.environ["TOP_URL"]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(storage_state="auth_state.json")
    page = context.new_page()
    page.goto(url, wait_until="networkidle")
    page.wait_for_timeout(3000)

    # Dump all possible container selectors and counts
    selectors = [
        "ol[data-list-id='chat-messages']",
        "li[id^='chat-messages-']",
        "[class*='messageContent']",
        "[class*='forumPost']",
        "[class*='threadCard']",
        "[class*='card-']",
        "article",
    ]
    print("=== Element counts on page ===")
    for sel in selectors:
        count = page.eval_on_selector_all(sel, "els => els.length")
        print(f"  {sel}: {count}")

    print("\n=== Page title ===")
    print(page.title())

    print("\n=== scrollHeight and scrollTop of message list ===")
    info = page.evaluate("""
        () => {
            const el = document.querySelector("ol[data-list-id='chat-messages']");
            if (!el) return 'NOT FOUND';
            return {
                scrollTop: el.scrollTop,
                scrollHeight: el.scrollHeight,
                clientHeight: el.clientHeight
            };
        }
    """)
    print(info)

    print("\n=== First 3 message li IDs ===")
    ids = page.evaluate("""
        () => Array.from(document.querySelectorAll("li[id^='chat-messages-']"))
                   .slice(0, 3).map(el => el.id)
    """)
    print(ids)

    # Find ALL scrollable elements on the page
    print("\n=== All scrollable elements ===")
    scrollables = page.evaluate("""
        () => {
            const results = [];
            document.querySelectorAll('*').forEach(el => {
                if (el.scrollHeight > el.clientHeight + 10) {
                    results.push({
                        tag: el.tagName,
                        id: el.id || '',
                        classList: Array.from(el.classList).slice(0, 3).join(' '),
                        scrollHeight: el.scrollHeight,
                        clientHeight: el.clientHeight,
                        scrollTop: el.scrollTop,
                    });
                }
            });
            return results.slice(0, 10);
        }
    """)
    for s in scrollables:
        print(f"  <{s['tag']}> id={s['id']!r} class={s['classList']!r} "
              f"scrollH={s['scrollHeight']} clientH={s['clientHeight']} scrollTop={s['scrollTop']}")

    context.close()
    browser.close()
