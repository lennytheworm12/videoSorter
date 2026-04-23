"""
Scrape a portal-rooted Age of Empires II wiki reference set into knowledge.db.

This collects a portal-core set of pages from the AoE wiki:
  - the portal page itself
  - core AoE2 reference pages linked from the portal
  - direct civilization pages linked from the portal
  - age pages linked from core AoE2 article pages

Usage:
    uv run python -m scrape.aoe2_wiki_scrape
    uv run python -m scrape.aoe2_wiki_scrape --limit 25
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
from collections import OrderedDict

from bs4 import BeautifulSoup
import requests

from core.db_paths import activate_knowledge_db

activate_knowledge_db()

from core.game_registry import canonical_aoe2_civilization
from scrape.aoe2_import import import_rows

WIKI_BASE = "https://ageofempires.fandom.com"
PORTAL_URL = f"{WIKI_BASE}/wiki/Age_of_Empires_II:Portal"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": WIKI_BASE,
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

CORE_PAGE_TITLES = {
    "Age of Empires II",
    "Age of Empires II: The Age of Kings",
    "Age of Empires II: Definitive Edition",
    "Civilizations",
    "Tech tree",
    "Units",
    "Buildings",
    "Technologies",
}
HELP_PAGE_TITLES = {
    "Wiki Help desk",
    "Policies and guidelines",
    "Community",
    "Community portal",
}
SUPPLEMENTAL_TITLES = {
    "Dark Age",
    "Feudal Age",
    "Castle Age",
    "Imperial Age",
    "Hotkey",
    "Hotkeys",
    "Control group",
    "Control groups",
}
SKIP_PAGE_TITLES = {
    "Sandbox",
}
DROP_SELECTORS = (
    "script",
    "style",
    "noscript",
    ".mw-editsection",
    ".reference",
    ".reflist",
    ".navbox",
    ".toc",
    ".thumb",
    ".gallery",
    ".wds-global-footer",
    ".page-footer",
    ".license-description",
    ".portable-infobox",
    "figure",
)
NOISE_PATTERNS = [
    r"^\[?\s*full article\.*\s*\]?$",
    r"^sign in to save.*$",
    r"^view source$",
    r"^history$",
    r"^talk \(\d+\)$",
    r"^image:.*$",
    r"^english$",
    r"^español$",
    r"^anyone can edit.*$",
    r"^please log in and create a username.*$",
    r"^creating a user name.*$",
]


def _fetch_html(url: str, timeout: int = 20) -> str:
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        response.raise_for_status()
        response.encoding = response.encoding or "utf-8"
        return response.text
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            fallback_html = _fetch_via_mediawiki_parse(url, timeout=timeout)
            if fallback_html:
                return fallback_html
        raise


def _mediawiki_page_name(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    if "/wiki/" not in parsed.path:
        return None
    page_name = parsed.path.split("/wiki/", 1)[-1]
    page_name = urllib.parse.unquote(page_name)
    return page_name or None


def _fetch_via_mediawiki_parse(url: str, timeout: int = 20) -> str | None:
    page_name = _mediawiki_page_name(url)
    if not page_name:
        return None

    api_url = f"{WIKI_BASE}/api.php"
    params = {
        "action": "parse",
        "page": page_name,
        "prop": "text|displaytitle",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
    }
    response = requests.get(api_url, headers=DEFAULT_HEADERS, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    parsed = payload.get("parse") or {}
    html = parsed.get("text")
    if not html:
        return None
    display_title = parsed.get("displaytitle") or page_name.replace("_", " ")
    return (
        "<html><body>"
        f'<h1 id="firstHeading">{display_title}</h1>'
        f'<div class="mw-parser-output">{html}</div>'
        "</body></html>"
    )


def _article_root(soup: BeautifulSoup) -> BeautifulSoup | None:
    for selector in (
        ".mw-parser-output",
        "[itemprop='articleBody']",
        ".page-content__content",
        "#mw-content-text",
    ):
        root = soup.select_one(selector)
        if root:
            return root
    return None


def _heading_title(soup: BeautifulSoup) -> str:
    heading = soup.select_one("#firstHeading")
    if heading:
        return heading.get_text(" ", strip=True)
    if soup.title:
        return soup.title.get_text(" ", strip=True).split("|", 1)[0].strip()
    return "AoE2 Wiki Page"


def _clean_text_line(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_internal_wiki_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc == urllib.parse.urlparse(WIKI_BASE).netloc and parsed.path.startswith("/wiki/")


def _absolute_url(href: str) -> str:
    return urllib.parse.urljoin(WIKI_BASE, href)


def _candidate_title(text: str, url: str) -> str:
    text = _clean_text_line(text)
    if text:
        return text
    slug = urllib.parse.unquote(url.split("/wiki/", 1)[-1])
    return slug.replace("_", " ")


def _include_portal_link(title: str) -> bool:
    if title in SKIP_PAGE_TITLES:
        return False
    if canonical_aoe2_civilization(title):
        return True
    if title in CORE_PAGE_TITLES or title in HELP_PAGE_TITLES or title in SUPPLEMENTAL_TITLES:
        return True
    return False


def discover_portal_pages(portal_html: str, portal_url: str = PORTAL_URL) -> list[dict]:
    soup = BeautifulSoup(portal_html, "html.parser")
    root = _article_root(soup)
    if not root:
        return []

    discovered: "OrderedDict[str, dict]" = OrderedDict()
    discovered[portal_url] = {
        "title": "Age of Empires II:Portal",
        "url": portal_url,
        "subject": None,
    }

    for anchor in root.find_all("a", href=True):
        url = _absolute_url(anchor["href"])
        if not _is_internal_wiki_url(url):
            continue
        title = _candidate_title(anchor.get_text(" ", strip=True), url)
        if not _include_portal_link(title):
            continue
        discovered.setdefault(url, {
            "title": title,
            "url": url,
            "subject": canonical_aoe2_civilization(title),
        })

    supplemental_sources = [
        page["url"] for page in discovered.values()
        if page["title"] in {
            "Age of Empires II",
            "Age of Empires II: The Age of Kings",
            "Age of Empires II: Definitive Edition",
        }
    ]
    for source_url in supplemental_sources:
        try:
            html = _fetch_html(source_url)
        except Exception:
            continue
        source_soup = BeautifulSoup(html, "html.parser")
        source_root = _article_root(source_soup)
        if not source_root:
            continue
        for anchor in source_root.find_all("a", href=True):
            url = _absolute_url(anchor["href"])
            if not _is_internal_wiki_url(url):
                continue
            title = _candidate_title(anchor.get_text(" ", strip=True), url)
            if title not in SUPPLEMENTAL_TITLES:
                continue
            discovered.setdefault(url, {
                "title": title,
                "url": url,
                "subject": canonical_aoe2_civilization(title),
            })

    return list(discovered.values())


def _extract_infobox_lines(soup: BeautifulSoup) -> list[str]:
    lines: list[str] = []
    for node in soup.select(".portable-infobox .pi-data"):
        label = _clean_text_line(node.select_one(".pi-data-label").get_text(" ", strip=True)) if node.select_one(".pi-data-label") else ""
        value = _clean_text_line(node.select_one(".pi-data-value").get_text(" ", strip=True)) if node.select_one(".pi-data-value") else ""
        if label and value:
            lines.append(f"{label}: {value}")
    return lines


def extract_page_text(page_html: str) -> tuple[str, str]:
    soup = BeautifulSoup(page_html, "html.parser")
    title = _heading_title(soup)
    root = _article_root(soup)
    if not root:
        return title, title

    infobox_lines = _extract_infobox_lines(soup)

    clone = BeautifulSoup(str(root), "html.parser")
    for selector in DROP_SELECTORS:
        for node in clone.select(selector):
            node.decompose()

    lines: list[str] = [title]
    if infobox_lines:
        lines.append("## Reference")
        lines.extend(infobox_lines)

    for node in clone.find_all(["h2", "h3", "h4", "p", "li"]):
        text = _clean_text_line(node.get_text(" ", strip=True))
        if not text:
            continue
        if any(re.fullmatch(pattern, text, flags=re.IGNORECASE) for pattern in NOISE_PATTERNS):
            continue
        if node.name.startswith("h"):
            text = text.strip("[]")
            if not text or text.lower() == title.lower():
                continue
            lines.append(f"## {text}")
            continue
        if len(text) < 3:
            continue
        if node.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    cleaned: list[str] = []
    previous = ""
    for line in lines:
        if line == previous:
            continue
        cleaned.append(line)
        previous = line

    text = "\n".join(cleaned).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def scrape_pages(pages: list[dict], limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    for page in pages[:limit]:
        html = _fetch_html(page["url"])
        title, content = extract_page_text(html)
        rows.append({
            "source": "aoe2_wiki",
            "source_id": "aoe2_wiki_" + re.sub(r"[^a-z0-9]+", "_", page["url"].lower()).strip("_"),
            "url": page["url"],
            "title": title or page["title"],
            "content": content,
            "subject": page["subject"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape a portal-core AoE2 wiki reference set")
    parser.add_argument("--portal-url", default=PORTAL_URL, help="Portal URL to seed the scrape from")
    parser.add_argument("--limit", type=int, help="Optional cap on imported pages after discovery")
    args = parser.parse_args()

    portal_html = _fetch_html(args.portal_url)
    pages = discover_portal_pages(portal_html, args.portal_url)
    rows = scrape_pages(pages, limit=args.limit)
    inserted, transcribed = import_rows(rows, default_source="aoe2_wiki")
    print(
        f"Discovered {len(pages)} portal-core page(s); "
        f"imported {inserted} new row(s) ({transcribed} transcribed) into knowledge.db"
    )


if __name__ == "__main__":
    main()
