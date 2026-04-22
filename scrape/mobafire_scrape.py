"""
Scrape top-rated Diamond+ MOBAFire written guides and store them in guide_test.db.

This treats written guides as a guide source parallel to youtube_guide, but skips
transcription entirely by saving the cleaned guide text directly into the
`transcription` column and marking rows as `transcribed`.

The scraper assumes the champion browse page is sorted by top-rated guides and
selects up to --limit qualifying written guides per champion.

Usage:
    uv run python -m scrape.mobafire_scrape --champion Aatrox
    uv run python -m scrape.mobafire_scrape --limit 3
    uv run python -m scrape.mobafire_scrape --status
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable

from bs4 import BeautifulSoup

os.environ.setdefault("DB_PATH", "guide_test.db")

from core.champions import load_champion_names
from core.database import get_connection, init_db, insert_video, set_transcription

MOBAFIRE_BASE = "https://www.mobafire.com"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)
GUIDE_SOURCES = ("youtube_guide", "mobafire_guide")
HIGH_ELO_TIERS = ("challenger", "grandmaster", "master", "diamond")
RANK_PATTERNS: list[tuple[str, str]] = [
    ("Challenger", r"\b(?:challenger|chall)\b"),
    ("Grandmaster", r"\b(?:grandmaster|grand master|gm)\b"),
    ("Master", r"\bmaster\b"),
    ("Diamond", r"\b(?:diamond|dia|d[1-4])\b"),
    ("Emerald", r"\b(?:emerald|emer|e[1-4])\b"),
    ("Platinum", r"\b(?:platinum|plat|p[1-4])\b"),
    ("Gold", r"\b(?:gold|g[1-4])\b"),
    ("Silver", r"\b(?:silver|s[1-4])\b"),
    ("Bronze", r"\b(?:bronze|b[1-4])\b"),
    ("Iron", r"\b(?:iron|i[1-4])\b"),
    ("Unranked", r"\bunranked\b"),
]
GUIDE_HREF_RE = re.compile(r"^/league-of-legends/build/[^?#]+-\d+$")
ROLE_PATTERNS: list[tuple[str, list[str]]] = [
    ("top", [r"\btop\b", r"\btop lane\b", r"\btoplane\b", r"\btoplaner\b"]),
    ("jungle", [r"\bjungle\b", r"\bjungler\b", r"\bjg\b"]),
    ("mid", [r"\bmid\b", r"\bmid lane\b", r"\bmidlane\b", r"\bmidlaner\b", r"\bmiddle\b"]),
    ("adc", [r"\badc\b", r"\bbot lane\b", r"\bbotlaner\b", r"\bmarksman\b", r"\bbottom\b", r"\bcarry\b"]),
    ("support", [r"\bsupport\b", r"\bsupp\b", r"\bsup\b", r"\benchanter\b"]),
]
NOISE_PATTERNS = [
    r"League of Legends Champions:.*",
    r"Vote Vote.*",
    r"Guide Discussion.*",
    r"More .* Guides.*",
    r"Cast Your Vote Today!.*",
    r"Find the best .* builds.*",
]
CHAPTER_SKIP_MARKERS = {
    "table of contents",
    "best items",
    "items",
    "itemization",
    "runes",
    "recommended items",
    "choose champion build",
    "dashable walls",
    "dash map",
}
INLINE_SKIP_PATTERNS = [
    r"^vote$",
    r"^comment$",
    r"^follow$",
    r"^guide discussion.*",
    r"^updated on$",
    r"^build guide by$",
    r"^more .* guides$",
    r"^you must be logged in.*",
    r"^please verify that you are not a bot.*",
    r"^did this guide help you.*",
    r"^commenting is required to vote.*",
    r"^thank you!$",
    r"^back to table$",
    r"^⮜\s*back to table\s*⮞$",
    r"^show all$",
    r"^extreme threats$",
    r"^ideal synergies$",
]
ITEM_LINE_PATTERNS = [
    r"^my 1st item choice:.*",
    r"^1st item$",
    r"^buildable legendaries$",
    r"^example builds?$",
    r"^boots?( are important!)?$",
    r"^starters?$",
    r"^recommended items$",
    r"^if ad:.*",
    r"^if enemy has.*$",
    r"^hard ap.*$",
]


@dataclass
class GuideCandidate:
    url: str
    title_hint: str
    summary_text: str
    website_rating: float | None = None


@dataclass
class ParsedGuide:
    video_id: str
    url: str
    title: str
    champion: str
    role: str
    author_rank: str
    website_rating: float | None
    guide_text: str
    description: str


_RANK_WEIGHTS = {
    None: 1.0,
    "Diamond": 1.0,
    "Master": 1.4,
    "Grandmaster": 1.8,
    "Challenger": 2.0,
    "Emerald": 0.7,
}


def _slugify_champion(name: str) -> str:
    normalized = name.lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[.'’]", "", normalized)
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return slug.replace("-and-", "-")


def _compact_slugify_champion(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _fetch_html(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _champion_index_urls(champion: str) -> list[str]:
    """
    Generate several plausible MOBAFire guide listing URLs.

    The older `/<slug>-guide` route works for many champions but breaks on some
    names with punctuation, while the newer `/champion/<slug>` route appears on
    champion browse pages and supports role-filtered listings.
    """
    slug = _slugify_champion(champion)
    compact_slug = _compact_slugify_champion(champion)
    valid_roles = _champion_valid_roles(champion)

    role_slugs = {
        "top": "top",
        "jungle": "jungle",
        "mid": "middle",
        "adc": "adc",
        "support": "support",
    }

    urls: list[str] = []

    def add(url: str) -> None:
        if url not in urls:
            urls.append(url)

    for candidate_slug in (slug, compact_slug):
        if not candidate_slug:
            continue
        add(f"{MOBAFIRE_BASE}/league-of-legends/{candidate_slug}-guide")
        add(f"{MOBAFIRE_BASE}/league-of-legends/champion/{candidate_slug}")
        add(f"{MOBAFIRE_BASE}/league-of-legends/champion/{candidate_slug}?depth=all&author=all")

        for role in valid_roles:
            role_slug = role_slugs.get(role)
            if role_slug:
                add(
                    f"{MOBAFIRE_BASE}/league-of-legends/champion/"
                    f"{candidate_slug}/{role_slug}?depth=all&author=all"
                )

    return urls


def _detect_role(text: str, champion: str) -> str:
    low = text.lower()
    for role, patterns in ROLE_PATTERNS:
        if any(re.search(p, low) for p in patterns):
            return role
    return "unknown"


def _detect_roles(text: str) -> list[str]:
    low = text.lower()
    found: list[str] = []
    for role, patterns in ROLE_PATTERNS:
        if any(re.search(p, low) for p in patterns):
            found.append(role)
    return found


def _pick_role(champion: str, candidate: GuideCandidate, title: str, header_text: str) -> str | None:
    """
    Infer the guide role without letting noisy page chrome override the listing/title.

    MOBAFire guide headers can include unrelated role words from sidebars, matchup
    widgets, or cross-links. Treat the listing summary and title as the only
    high-confidence role signals, then fall back to the champion's known roles.
    """
    valid_roles = _champion_valid_roles(champion)
    primary_role = _champion_primary_role(champion)
    constrained = bool(valid_roles and "unknown" not in valid_roles)

    summary_roles = _detect_roles(candidate.summary_text)
    title_roles = _detect_roles(title)
    header_roles = _detect_roles(header_text)

    if constrained:
        overlap = [role for role in summary_roles if role in valid_roles]
        if overlap:
            return overlap[0]

        overlap = [role for role in title_roles if role in valid_roles]
        if overlap:
            return overlap[0]

        if not summary_roles and not title_roles:
            if primary_role != "unknown":
                return primary_role
            overlap = [role for role in header_roles if role in valid_roles]
            if overlap:
                return overlap[0]
            return None

        return None

    combined_roles = summary_roles or title_roles or header_roles
    if combined_roles:
        return combined_roles[0]
    if primary_role != "unknown":
        return primary_role
    return "unknown"


def _champion_valid_roles(champion: str) -> set[str]:
    """
    Return the champion's primary + secondary roles from the main champion/video DB.
    This is used to reject high-ranked off-role guides that would pollute retrieval.
    """
    roles: list[str] = []
    try:
        main_conn = sqlite3.connect("videos.db")
        main_conn.row_factory = sqlite3.Row
        db_rows = main_conn.execute(
            """
            SELECT role, COUNT(*) AS cnt
            FROM videos
            WHERE champion = ? AND role NOT IN ('ability_enrichment', 'unknown')
            GROUP BY role
            ORDER BY cnt DESC
            LIMIT 2
            """,
            (champion,),
        ).fetchall()
        if db_rows:
            roles = [r["role"] for r in db_rows]
        else:
            arch_rows = main_conn.execute(
                """
                SELECT role
                FROM champion_archetypes
                WHERE champion = ?
                LIMIT 2
                """,
                (champion,),
            ).fetchall()
            roles = [r["role"] for r in arch_rows]
        main_conn.close()
    except Exception:
        pass

    return set(roles) if roles else {"unknown"}


def _champion_primary_role(champion: str) -> str:
    try:
        main_conn = sqlite3.connect("videos.db")
        main_conn.row_factory = sqlite3.Row
        row = main_conn.execute(
            """
            SELECT role, COUNT(*) AS cnt
            FROM videos
            WHERE champion = ? AND role NOT IN ('ability_enrichment', 'unknown')
            GROUP BY role
            ORDER BY cnt DESC
            LIMIT 1
            """,
            (champion,),
        ).fetchone()
        if row:
            main_conn.close()
            return row["role"]

        row = main_conn.execute(
            "SELECT role FROM champion_archetypes WHERE champion = ? LIMIT 1",
            (champion,),
        ).fetchone()
        main_conn.close()
        if row:
            return row["role"]
    except Exception:
        pass
    return "unknown"


def _mobafire_video_id(url: str) -> str:
    m = re.search(r"-(\d+)$", url.rstrip("/"))
    if m:
        return f"mobafire_{m.group(1)}"
    return "mobafire_" + re.sub(r"[^a-z0-9]+", "_", url.lower()).strip("_")


def _existing_mobafire_count(champion: str) -> int:
    with get_connection() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM videos WHERE champion = ? AND source = 'mobafire_guide'",
            (champion,),
        ).fetchone()[0]


def _extract_rank(text: str) -> str | None:
    low = text.lower()
    for canonical, pattern in RANK_PATTERNS:
        if re.search(pattern, low):
            return canonical
    return None


def _extract_author_rank(candidate: GuideCandidate, title: str, header_text: str) -> str | None:
    """
    Prefer structured listing metadata for rank extraction.

    Guide titles often contain bait phrases like "Rank 1" or "Challenger guide"
    that describe the content, not necessarily the displayed author rank. Use the
    listing summary first, then fall back to header/banner text with the title
    removed so title words cannot spoof the rank.
    """
    rank = _extract_rank(candidate.summary_text)
    if rank:
        return rank

    sanitized_header = header_text
    if title:
        sanitized_header = sanitized_header.replace(title, " ")
    sanitized_header = re.sub(r"\s{2,}", " ", sanitized_header).strip()
    return _extract_rank(sanitized_header)


def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    lines = [ln.strip() for ln in text.splitlines()]
    kept: list[str] = []
    for line in lines:
        if not line:
            continue
        if len(line) < 2:
            continue
        if any(re.fullmatch(pat, line, flags=re.IGNORECASE) for pat in NOISE_PATTERNS):
            continue
        if any(re.fullmatch(pat, line, flags=re.IGNORECASE) for pat in INLINE_SKIP_PATTERNS):
            continue
        kept.append(line)

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _clean_title(raw_title: str) -> str:
    title = raw_title.strip()
    title = re.sub(r"\s*::\s*League of Legends Strategy Builds\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^\s*[A-Za-z' .&-]+\s+Build Guide\s*:\s*", "", title)
    return re.sub(r"\s{2,}", " ", title).strip()


def _normalise_pipe_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s*\|\s*", "\n", text)
    text = re.sub(r"[━─]{3,}", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _clean_text(text)


def _chapter_heading(node: BeautifulSoup) -> str:
    heading = ""
    for tag in ("h1", "h2", "h3", "h4", "header", "strong", "b"):
        found = node.find(tag)
        if found:
            heading = found.get_text(" ", strip=True)
            if heading:
                break
    if not heading:
        text = node.get_text("\n", strip=True)
        heading = text.split("\n", 1)[0].strip() if text else ""
    heading = heading.strip("⟡⚔️☯ ")
    return heading


def _should_skip_chapter(heading: str) -> bool:
    low = heading.lower().strip()
    return any(marker in low for marker in CHAPTER_SKIP_MARKERS)


def _filter_matchup_lines(lines: Iterable[str]) -> list[str]:
    kept: list[str] = []
    champion_names = {c.lower() for c in load_champion_names()}
    threat_labels = {"extreme", "major", "even", "minor", "tiny", "ideal", "strong", "ok", "low", "none"}

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if any(re.fullmatch(pat, line, flags=re.IGNORECASE) for pat in ITEM_LINE_PATTERNS):
            continue
        if low in threat_labels:
            continue
        if low in champion_names:
            continue
        if len(line.split()) <= 3 and line.lower() not in {"q", "w", "e", "r", "passive"}:
            continue
        kept.append(line)
    return kept


def _extract_threat_synergy_sections(soup: BeautifulSoup) -> list[str]:
    sections: list[str] = []
    champion_names = {c.lower() for c in load_champion_names()}
    rating_labels = {"extreme", "major", "even", "minor", "tiny", "ideal", "strong", "ok", "low", "none"}

    for selector, label in [
        (".view-guide__tS__bot__left", "Threats & Matchups"),
        (".view-guide__tS__bot__right", "Synergies"),
    ]:
        node = soup.select_one(selector)
        if not node:
            continue
        text = _normalise_pipe_text(node.get_text(" | ", strip=True))
        raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        entries: list[str] = []
        current_champion: str | None = None
        current_rating: str | None = None
        notes: list[str] = []

        def flush() -> None:
            nonlocal current_champion, current_rating, notes
            if not current_champion or not notes:
                current_champion = None
                current_rating = None
                notes = []
                return
            note_text = " ".join(notes).strip()
            prefix = f"{current_champion}"
            if current_rating:
                prefix += f" ({current_rating})"
            entries.append(f"{prefix}: {note_text}")
            current_champion = None
            current_rating = None
            notes = []

        for line in raw_lines:
            low = line.lower()
            if any(re.fullmatch(pat, line, flags=re.IGNORECASE) for pat in ITEM_LINE_PATTERNS):
                continue
            if low in champion_names:
                flush()
                current_champion = line
                continue
            if low in rating_labels:
                current_rating = line
                continue
            if any(re.fullmatch(pat, line, flags=re.IGNORECASE) for pat in INLINE_SKIP_PATTERNS):
                continue
            if len(line.split()) <= 2:
                continue
            notes.append(line)
        flush()

        lines = _filter_matchup_lines(entries)
        if len(lines) < 2:
            continue
        section = f"{label}\n" + "\n".join(lines).strip()
        if len(section.split()) >= 30:
            sections.append(section)

    return sections


def _extract_chapter_sections(soup: BeautifulSoup) -> list[str]:
    chapters_root = soup.select_one(".view-guide__chapters")
    if not chapters_root:
        return []

    sections: list[str] = []
    for chapter in chapters_root.select(".view-guide__chapter"):
        heading = _chapter_heading(chapter)
        if heading and _should_skip_chapter(heading):
            continue

        text = _normalise_pipe_text(chapter.get_text(" | ", strip=True))
        if not text:
            continue

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if heading.lower().startswith("threats") or heading.lower().startswith("synergies"):
            lines = _filter_matchup_lines(lines)
        else:
            lines = [
                ln for ln in lines
                if not any(re.fullmatch(pat, ln, flags=re.IGNORECASE) for pat in ITEM_LINE_PATTERNS)
            ]

        if len(lines) < 3:
            continue

        section = "\n".join(lines).strip()
        if len(section.split()) < 40:
            continue
        sections.append(section)

    return sections


def _extract_main_guide_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "svg", "img", "form", "button"]):
        tag.decompose()

    chapter_sections = _extract_chapter_sections(soup)
    threat_sections = _extract_threat_synergy_sections(soup)
    combined_sections = chapter_sections + threat_sections
    if combined_sections:
        return "\n\n".join(combined_sections)

    return _clean_text(soup.get_text("\n", strip=True))


def _looks_like_low_signal_guide(text: str) -> bool:
    """
    Reject guides that are mostly sparse labels/build fragments rather than actual
    explanatory prose about how the champion is played.
    """
    words = text.split()
    if len(words) < 550:
        return True

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    long_lines = [ln for ln in lines if len(ln.split()) >= 8]
    sentence_like = [
        ln for ln in long_lines
        if any(ch in ln for ch in ".!?—:") or len(ln.split()) >= 14
    ]

    if len(long_lines) < 18:
        return True
    if len(sentence_like) < 10:
        return True

    strategic_markers = [
        "laning", "gameplan", "trade", "combo", "teamfight", "matchup",
        "vision", "split push", "roam", "power spike", "cooldown",
    ]
    low = text.lower()
    marker_hits = sum(1 for marker in strategic_markers if marker in low)
    return marker_hits < 3


def rank_weight(rank: str | None) -> float:
    if rank is None:
        return _RANK_WEIGHTS[None]
    return _RANK_WEIGHTS.get(rank.title(), 0.0)


def _extract_site_rating(text: str) -> float | None:
    m = re.search(r"\b((?:10|[0-9](?:\.[0-9])?))\s+\d+\s+Votes\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _parse_index_candidates(html: str) -> list[GuideCandidate]:
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    candidates: list[GuideCandidate] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not GUIDE_HREF_RE.match(href):
            continue
        url = urllib.parse.urljoin(MOBAFIRE_BASE, href)
        if url in seen:
            continue

        container = a
        summary_text = ""
        for _ in range(4):
            container = container.parent
            if container is None:
                break
            text = container.get_text(" ", strip=True)
            if 80 <= len(text) <= 1500:
                summary_text = text
                break
        if not summary_text:
            summary_text = a.get_text(" ", strip=True)

        seen.add(url)
        candidates.append(
            GuideCandidate(
                url=url,
                title_hint=a.get_text(" ", strip=True),
                summary_text=summary_text,
                website_rating=_extract_site_rating(summary_text),
            )
        )
    return candidates


def _parse_guide_page(champion: str, candidate: GuideCandidate) -> ParsedGuide | None:
    html = _fetch_html(candidate.url)
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    heading = soup.select_one(".view-guide__header h1, .view-guide__header .h1, h1")
    if heading:
        title = heading.get_text(" ", strip=True)
    if not title and soup.title and soup.title.string:
        title = _clean_title(soup.title.string)
    if not title:
        title = candidate.title_hint

    header_node = soup.select_one(".view-guide__header")
    header_text = header_node.get_text(" ", strip=True) if header_node else ""
    full_text = soup.get_text(" ", strip=True)
    rank = _extract_author_rank(candidate, title, header_text)
    if rank_weight(rank) <= 0.0:
        return None
    rating = candidate.website_rating
    if rating is None:
        rating = _extract_site_rating(full_text)
    if rank != "Challenger" and (rating is None or rating < 8.5):
        return None

    guide_text = _extract_main_guide_text(soup)
    if _looks_like_low_signal_guide(guide_text):
        return None

    role = _pick_role(champion, candidate, title, header_text)
    if role is None:
        return None

    desc_bits = [rank or "NoRank", "MOBAFire", title]
    if rating is not None:
        desc_bits.insert(0, f"Rating {rating:.1f}")
    return ParsedGuide(
        video_id=_mobafire_video_id(candidate.url),
        url=candidate.url,
        title=title,
        champion=champion,
        role=role,
        author_rank=rank,
        website_rating=rating,
        guide_text=guide_text,
        description=" | ".join(bit for bit in desc_bits if bit),
    )


def scrape_champion(
    champion: str,
    limit: int = 3,
    dry_run: bool = False,
    index_url_override: str | None = None,
) -> int:
    already_have = _existing_mobafire_count(champion)
    needed = max(0, limit - already_have)
    if needed == 0:
        print(f"  [skip] already have {already_have} mobafire guides (limit {limit})")
        return 0

    candidates_by_url: dict[str, GuideCandidate] = {}
    last_http_error: urllib.error.HTTPError | None = None
    last_error: Exception | None = None

    index_urls = [index_url_override] if index_url_override else _champion_index_urls(champion)
    for index_url in index_urls:
        try:
            html = _fetch_html(index_url)
            parsed_candidates = _parse_index_candidates(html)
            for candidate in parsed_candidates:
                existing = candidates_by_url.get(candidate.url)
                if existing is None:
                    candidates_by_url[candidate.url] = candidate
                    continue

                existing_rating = existing.website_rating or 0.0
                candidate_rating = candidate.website_rating or 0.0
                if candidate_rating > existing_rating:
                    candidates_by_url[candidate.url] = candidate
                    continue
                if len(candidate.summary_text) > len(existing.summary_text):
                    candidates_by_url[candidate.url] = candidate
        except urllib.error.HTTPError as exc:
            last_http_error = exc
            continue
        except Exception as exc:
            last_error = exc
            continue

    candidates = list(candidates_by_url.values())
    if not candidates:
        if last_http_error is not None:
            raise last_http_error
        if last_error is not None:
            raise last_error
        return 0

    accepted: list[ParsedGuide] = []
    seen_ids: set[str] = set()

    for candidate in candidates:
        try:
            parsed = _parse_guide_page(champion, candidate)
        except urllib.error.HTTPError as exc:
            print(f"  [skip] {candidate.url} HTTP {exc.code}")
            continue
        except Exception as exc:
            print(f"  [skip] {candidate.url} parse failed: {exc}")
            continue

        if not parsed:
            continue
        time.sleep(random.uniform(1.0, 2.0))
        if parsed.video_id in seen_ids:
            continue
        seen_ids.add(parsed.video_id)
        accepted.append(parsed)

    accepted.sort(
        key=lambda g: (
            float(g.website_rating or 0.0),
            rank_weight(g.author_rank),
            len(g.guide_text.split()),
        ),
        reverse=True,
    )

    selected = accepted[:needed]
    saved = 0
    for parsed in selected:
        words = len(parsed.guide_text.split())
        rank_label = parsed.author_rank or "NoRank"
        print(
            f"  [{'dry' if dry_run else 'save'}] {parsed.video_id} | "
            f"{rank_label:<11} | {parsed.role:<7} | {words:>5}w | {parsed.title[:70]}"
        )
        if not dry_run:
            insert_video(
                video_id=parsed.video_id,
                video_url=parsed.url,
                role=parsed.role,
                message_timestamp="",
                video_title=parsed.title,
                description=parsed.description,
                champion=parsed.champion,
                rank=parsed.author_rank,
                website_rating=parsed.website_rating,
                source="mobafire_guide",
            )
            set_transcription(parsed.video_id, parsed.guide_text)
        saved += 1

    return saved


def print_status() -> None:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT champion, COUNT(*) AS n
            FROM videos
            WHERE source = 'mobafire_guide'
            GROUP BY champion
            ORDER BY champion
            """
        ).fetchall()
    total = sum(r["n"] for r in rows)
    print(f"\nmobafire_guide: {total} guides across {len(rows)} champions\n")
    for row in rows[:20]:
        print(f"  {row['champion']:<18} {row['n']}")
    if len(rows) > 20:
        print("  ...")


def _reset_mobafire_guides(champion: str | None = None) -> None:
    """
    Delete existing mobafire_guide rows and linked insights so a fresh scrape can
    repopulate them under the current filtering policy.
    """
    with get_connection() as conn:
        if champion:
            rows = conn.execute(
                "SELECT video_id FROM videos WHERE source = 'mobafire_guide' AND champion = ?",
                (champion,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT video_id FROM videos WHERE source = 'mobafire_guide'"
            ).fetchall()

        video_ids = [r["video_id"] for r in rows]
        if not video_ids:
            label = champion or "all champions"
            print(f"No existing mobafire_guide rows to reset for {label}.")
            return

        placeholders = ",".join("?" for _ in video_ids)
        deleted_insights = conn.execute(
            f"DELETE FROM insights WHERE video_id IN ({placeholders})",
            video_ids,
        ).rowcount

        if champion:
            deleted_videos = conn.execute(
                "DELETE FROM videos WHERE source = 'mobafire_guide' AND champion = ?",
                (champion,),
            ).rowcount
        else:
            deleted_videos = conn.execute(
                "DELETE FROM videos WHERE source = 'mobafire_guide'"
            ).rowcount
        conn.commit()

    label = champion or "all champions"
    print(f"Reset mobafire_guide rows for {label}: {deleted_videos} videos, {deleted_insights} insights deleted")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape top MOBAFire guides into guide_test.db")
    parser.add_argument("--champion", help="Only scrape one champion")
    parser.add_argument("--index-url", help="Override the MOBAFire listing URL for --champion")
    parser.add_argument("--limit", type=int, default=3, help="Max qualifying guides per champion")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be saved")
    parser.add_argument("--status", action="store_true", help="Show DB counts and exit")
    parser.add_argument("--reanalyze", action="store_true",
                        help="Delete existing mobafire_guide rows first, then scrape fresh")
    args = parser.parse_args()

    init_db()
    if args.status:
        print_status()
        return
    if args.reanalyze:
        _reset_mobafire_guides(args.champion)

    champions = [args.champion] if args.champion else load_champion_names()
    total = 0
    for idx, champion in enumerate(champions, start=1):
        print(f"\n[{idx}/{len(champions)}] {champion}")
        try:
            total += scrape_champion(
                champion,
                limit=args.limit,
                dry_run=args.dry_run,
                index_url_override=args.index_url if args.champion == champion else None,
            )
        except Exception as exc:
            print(f"  [error] {champion}: {exc}")
    print(f"\nDone — {'would save' if args.dry_run else 'saved'} {total} MOBAFire guides.")


if __name__ == "__main__":
    main()
