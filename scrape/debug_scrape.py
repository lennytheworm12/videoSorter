"""Debug what Discord actually renders in the channel."""
import os, time
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
load_dotenv()

url = os.environ["MID_URL"]

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

    beginning_initially = page.evaluate("() => !!document.querySelector('[class*=\"channelBeginning\"]')")
    print(f"\n=== channelBeginning visible initially: {beginning_initially} ===")

    print("\n=== Repeatedly scroll to BOTTOM until stable ===")
    prev = 0
    for i in range(10):
        page.evaluate("""
            () => {
                document.querySelectorAll('[class*="scroller"]').forEach(el => {
                    if (el.scrollHeight > el.clientHeight) {
                        el.scrollTop = el.scrollHeight;
                    }
                });
            }
        """)
        page.wait_for_timeout(2500)
        count = page.eval_on_selector_all("li[id^='chat-messages-']", "els => els.length")
        beginning = page.evaluate("() => !!document.querySelector('[class*=\"channelBeginning\"]')")
        scroller_st = page.evaluate("""
            () => {
                const el = document.querySelector('[class*="scroller__36d07"]');
                if (!el) return {};
                return { scrollTop: el.scrollTop, scrollHeight: el.scrollHeight };
            }
        """)
        print(f"  iter {i+1}: {count} messages | scrollTop={scroller_st.get('scrollTop')} scrollH={scroller_st.get('scrollHeight')} | beginning={beginning}")
        if beginning:
            print("  → channelBeginning found!")
            break
        if count == prev and i >= 2:
            print(f"  → Stale at {count}")
            break
        prev = count

    print("\n=== Parent chain of the ol (first 8 ancestors) ===")
    chain = page.evaluate("""
        () => {
            const ol = document.querySelector("ol[data-list-id='chat-messages']");
            if (!ol) return [];
            const results = [];
            let el = ol.parentElement;
            for (let i = 0; i < 8 && el; i++) {
                const style = window.getComputedStyle(el);
                results.push({
                    tag: el.tagName,
                    id: el.id || '',
                    classList: Array.from(el.classList).slice(0, 4).join(' '),
                    overflow: style.overflow,
                    overflowY: style.overflowY,
                    scrollHeight: el.scrollHeight,
                    clientHeight: el.clientHeight,
                    scrollTop: el.scrollTop,
                });
                el = el.parentElement;
            }
            return results;
        }
    """)
    for i, el in enumerate(chain):
        print(f"  [{i}] <{el['tag']}> id={el['id']!r} class={el['classList']!r} "
              f"overflow={el['overflow']!r}/{el['overflowY']!r} "
              f"scrollH={el['scrollHeight']} clientH={el['clientHeight']} scrollTop={el['scrollTop']}")

    print("\n=== All elements with overflow auto/scroll ===")
    overflow_els = page.evaluate("""
        () => {
            const results = [];
            document.querySelectorAll('*').forEach(el => {
                const style = window.getComputedStyle(el);
                const oy = style.overflowY;
                if (oy === 'auto' || oy === 'scroll') {
                    results.push({
                        tag: el.tagName,
                        id: el.id || '',
                        classList: Array.from(el.classList).slice(0, 4).join(' '),
                        overflowY: oy,
                        scrollHeight: el.scrollHeight,
                        clientHeight: el.clientHeight,
                        scrollTop: el.scrollTop,
                    });
                }
            });
            return results;
        }
    """)
    for el in overflow_els:
        print(f"  <{el['tag']}> id={el['id']!r} class={el['classList']!r} "
              f"overflowY={el['overflowY']!r} "
              f"scrollH={el['scrollHeight']} clientH={el['clientHeight']} scrollTop={el['scrollTop']}")

    context.close()
    browser.close()
