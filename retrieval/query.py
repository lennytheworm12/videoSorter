"""
PHASE 5: Query the knowledge base using RRF (semantic + BM25) + RAG.

Retrieval uses Reciprocal Rank Fusion over two ranked lists:
  1. Semantic search  — cosine similarity on sentence-transformer embeddings
  2. BM25 keyword     — exact term overlap, good for champion/item names

RRF formula: score(d) = Σ 1 / (k + rank_i(d))  where k=60

Intent detection (no LLM):
  - Matchup : "kaisa into yunara", "kaisa vs yunara", "kaisa against yunara"
  - Synergy : "kaisa with thresh", "jinx alongside lulu", "jinx and lulu bot"
  - General : everything else

Usage:
    python -m retrieval.query "how do I play Cassiopeia against poke mages?"
    python -m retrieval.query "kaisa into yunara"
    python -m retrieval.query "jinx with lulu bot lane"
    python -m retrieval.query "what does the coach say about wave management?" --role mid
    python -m retrieval.query "when should I take teleport vs ignite?" --type principles
"""

import re
import math
import json
import html
import logging
import argparse
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from pipeline.embed import load_all_vectors
from pipeline.champion_crossref import get_archetype_insights
from pipeline.aoe2_crossref import get_applicable_insights
from core.llm import chat as llm_chat
from core.db_paths import all_content_db_paths
from core.game_registry import AOE2_CIVILIZATIONS, DEFAULT_GAME, canonical_aoe2_civilization, game_label, normalize_game

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 35
RRF_K = 60
RANK_WEIGHTS = {
    "": 1.0,
    "diamond": 1.0,
    "emerald": 0.7,
    "master": 1.4,
    "grandmaster": 1.8,
    "challenger": 2.0,
}

_ABILITY_SLOT_ORDER = {"P": 0, "Q": 1, "W": 2, "E": 3, "R": 4}
_ABILITY_TAG_PRIORITY = [
    "suppression", "stasis", "sleep", "polymorph", "airborne", "stun",
    "charm", "taunt", "fear", "grounded", "root", "silence", "dash",
    "blink", "untargetable", "invulnerable", "spell_shield", "shield",
    "heal", "slow", "execute", "true_damage",
]


def _source_weight(meta: dict) -> float:
    source = meta.get("source") or "discord"
    if source == "discord":
        return 1.6
    if source == "aoe2_crossref":
        return 0.9
    if source == "aoe2_wiki":
        return 1.1
    if source == "aoe2_pdf":
        return 2.0
    if source in {"aoe2_video", "aoe2_coaching"}:
        return 1.0
    if source == "mobafire_guide":
        rank_weight = RANK_WEIGHTS.get(str(meta.get("rank") or "").lower(), 1.0)
        rating = meta.get("website_rating")
        rating_bonus = 1.0
        if rating is not None:
            try:
                rating_bonus += min(max((float(rating) - 8.5) * 0.05, 0.0), 0.15)
            except (TypeError, ValueError):
                pass
        return rank_weight * rating_bonus
    return 1.0


def _source_label(meta: dict) -> str:
    source = meta.get("source") or "discord"
    if source == "mobafire_guide":
        parts = ["mobafire"]
        if meta.get("rank"):
            parts.append(str(meta["rank"]))
        if meta.get("website_rating") is not None:
            try:
                parts.append(f"rating {float(meta['website_rating']):.1f}")
            except (TypeError, ValueError):
                pass
        return " | ".join(parts)
    if source == "aoe2_wiki":
        return "wiki"
    if source == "aoe2_pdf":
        return "pdf guide"
    if source == "aoe2_coaching":
        return "coaching"
    if source == "aoe2_video":
        return "youtube"
    if source == "aoe2_crossref":
        subject = meta.get("source_subject") or meta.get("subject")
        return f"crossref{f' | {subject}' if subject else ''}"
    if source == "youtube_guide":
        return "youtube"
    return source


def _strip_html(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _shorten_ability_description(description: str, limit: int = 150) -> str:
    clean = _strip_html(description)
    if not clean:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", clean)
    summary = parts[0].strip() if parts and parts[0].strip() else clean
    summary = re.sub(rf"^{re.escape(summary.split(':', 1)[0])}:\s*", "", summary) if ":" in summary[:30] else summary
    if len(summary) <= limit:
        return summary
    clipped = summary[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return clipped + "…"


def _plain_english_ability_summary(champion: str, description: str, limit: int = 150) -> str:
    summary = _shorten_ability_description(description, limit=400)
    if not summary:
        return ""

    summary = re.sub(rf"^{re.escape(champion)}\s+", "", summary)

    replacements = [
        (r"\ban enemy it encounters\b", "the first enemy hit"),
        (r"\bblows a kiss\b", "throws a kiss skillshot forward"),
        (r"\bsends out and pulls back\b", "throws out and pulls back"),
        (r"\bgains a brief burst of Move Speed\b", "gets a brief move speed burst"),
        (r"\binstantly stopping movement abilities\b", "stopping dashes"),
        (r"\bwalk harmlessly towards her\b", f"walk toward {champion}"),
        (r"\bwalk harmlessly towards him\b", f"walk toward {champion}"),
        (r"\bwalk harmlessly towards them\b", f"walk toward {champion}"),
        (r"\bnearby enemies\b", "nearby targets"),
    ]
    for pattern, replacement in replacements:
        summary = re.sub(pattern, replacement, summary, flags=re.IGNORECASE)

    summary = summary.replace("Move Speed", "move speed")
    summary = re.sub(r"\s+", " ", summary).strip(" ,;:")
    if summary:
        summary = summary[0].upper() + summary[1:]

    if len(summary) <= limit:
        return summary
    clipped = summary[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return clipped + "…"


def _important_ability_tags(raw_properties: str | None, limit: int = 3) -> list[str]:
    if not raw_properties:
        return []
    try:
        tags = json.loads(raw_properties)
    except Exception:
        return []
    if not isinstance(tags, list):
        return []
    ordered: list[str] = []
    seen: set[str] = set()
    for tag in _ABILITY_TAG_PRIORITY:
        if tag in tags and tag not in seen:
            ordered.append(tag)
            seen.add(tag)
    for tag in tags:
        if isinstance(tag, str) and tag not in seen:
            ordered.append(tag)
            seen.add(tag)
    return ordered[:limit]


def _ability_rows(champion: str) -> list[sqlite3.Row]:
    """
    Fetch champion ability rows from the canonical Data Dragon-backed table.
    videos.db is the source of truth for champion kit reference data.
    """
    rows: list[sqlite3.Row] = []
    try:
        with sqlite3.connect("videos.db") as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT champion, ability_slot, name, description, properties
                FROM champion_abilities
                WHERE LOWER(champion) = LOWER(?)
                """,
                (champion,),
            ).fetchall()
    except Exception:
        rows = []
    return sorted(rows, key=lambda r: _ABILITY_SLOT_ORDER.get(r["ability_slot"], 99))


def _format_ability_reference(champion: str) -> str:
    rows = _ability_rows(champion)
    if not rows:
        return ""
    lines = [f"### {champion} Abilities"]
    for row in rows:
        slot = row["ability_slot"]
        name = row["name"] or slot
        desc = _plain_english_ability_summary(champion, row["description"] or "")
        tags = _important_ability_tags(row["properties"])
        tag_text = f" ({', '.join(tags)})" if tags else ""
        if desc:
            lines.append(f"- `{slot}` {name}{tag_text}: {desc}")
        else:
            lines.append(f"- `{slot}` {name}{tag_text}")
    return "\n".join(lines)


def _ability_reference_block(champions: list[str]) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for champion in champions:
        if champion and champion not in seen:
            ordered.append(champion)
            seen.add(champion)
    if not ordered:
        return ""

    sections = [_format_ability_reference(champion) for champion in ordered]
    sections = [section for section in sections if section]
    if not sections:
        return ""
    return "**Champion Kits**\n" + "\n\n".join(sections)


# ── Prompts ───────────────────────────────────────────────────────────────────

GENERAL_SYSTEM = """
You are a League of Legends coaching assistant. You have access to insights
extracted from a real coach's video library. Answer using ONLY the provided
coaching insights — do not add advice not grounded in the retrieved insights.

When the question is about how to play a specific champion, structure your answer as:

### Champion Identity
- What kind of champion this is — win condition, role in fights, key strengths and weaknesses
- Item spike dependencies or power curves if present in the insights

### Key Mechanics & Ability Interactions
- Passive behavior, ability windows, resource management (energy/mana), and auto-attack weaving cues
- When to use or hold specific abilities (e.g. shield to absorb a key enemy spell, dash timing)
- CC windows and follow-up opportunities
- Omit this section if the insights contain no mechanic-level detail

### Laning / Early Game
- Specific laning mechanics, trade patterns, level spikes, what to do and avoid early

### Wave Control
- How this champion interacts with the wave — push, freeze, slow-push, or crash
- Any ability or stat properties that make them strong or weak at wave management
- Omit this section if the insights contain nothing relevant to wave control

### Teamfight & Positioning
- Where to position, when to engage, target priority
- How to use mobility or defensives to survive and reach carries
- CC threats to respect that can shut down their pattern
- Omit this section if the insights contain no teamfight-specific detail

### Macro
- Mid/late game decision-making, roaming, split push vs. grouping

### Synergies
- Champions or compositions this champion works well with, and why
- Omit this section if the insights contain no synergy information

### General Principles
- Only include here: mindset, practice habits, or universal tips that are not champion-specific
- Omit this section entirely if nothing fits

For non-champion-specific questions, skip the headers and answer directly and concisely.

Rules:
- Champion-specific insights (champion_identity, champion_mechanics, laning_tips, matchup_advice) take priority
- For exact matchup questions, champion_matchups insights are the highest-fidelity source and should be prioritised over broader matchup_advice when present
- Never bury champion mechanics inside a generic principles section
- Only include a section if the insights actually support it — do not pad with generic advice
- If insights contradict each other, acknowledge both perspectives
- Be concise — players want clear, direct coaching advice
""".strip()

GENERAL_USER = """
Player question: {question}

Relevant coaching insights:
{insights}

Answer using only the insights above.
""".strip()

MATCHUP_SYSTEM = """
You are a League of Legends coaching assistant. You have insights about two
champions extracted from a real coach's video library.

Explain how {champion_a} should play against {champion_b} using ONLY the
provided insights. Synthesize the interaction — if A wants short trades and B
wants extended fights, conclude that A should take short trades. Do not invent
advice not grounded in the insights.

CRITICAL — ability attribution: The insights are grouped by champion. Any
[ability_windows] insight listed under "{champion_a}" describes {champion_a}'s
own abilities. Any [ability_windows] insight listed under "{champion_b}"
describes {champion_b}'s own abilities. Never attribute an ability mechanic to
the wrong champion. If an ability_windows insight says "your Q does X", that
"your" refers to the champion whose section it appears in.

Structure your answer:

**Laning Phase**
- {champion_a}'s win condition in this matchup
- How {champion_a}'s strengths exploit or survive {champion_b}'s gameplan
- Specific trading, wave, and kill setup tips

**Post-Lane**
- How the matchup dynamic shifts after laning
- Win condition execution

Only include a section if the insights support it.
""".strip()

MATCHUP_USER = """
Player question: {question}

=== {champion_a} insights === {note_a}
{insights_a}

=== {champion_b} insights === {note_b}
{insights_b}

Using the insights above, explain how {champion_a} should play against {champion_b}.
If a side only has archetype data, reason from the shared playstyle — do not refuse to answer.
If stat context is provided below, factor it into your answer (e.g. range mismatch changes
safe farming distance; low base HP/armor increases early burst windows).
""".strip()

SYNERGY_SYSTEM = """
You are a League of Legends coaching assistant. You have insights about two
champions extracted from a real coach's video library.

Explain how {champion_a} and {champion_b} work together using ONLY the
provided insights. Synthesize their kits — if A sets up engage and B has
high follow-up damage, describe the combo. Do not invent advice not grounded
in the insights.

Structure your answer:

**Win Condition**
- How {champion_a} and {champion_b}'s strengths combine

**Execution**
- Specific combo, timing, and positioning tips
- What each player needs to do to enable the other

Only include a section if the insights support it.
""".strip()

SYNERGY_USER = """
Player question: {question}

Insights about {champion_a}: {note_a}
{insights_a}

Insights about {champion_b}: {note_b}
{insights_b}

Using the insights above, explain how {champion_a} and {champion_b} work together.
If a side only has archetype data, reason from the shared playstyle — do not refuse to answer.
""".strip()

GENERIC_SYSTEM = """
You are a {game_name} coaching assistant. You have access to insights extracted
from a curated knowledge base of educational content. Answer using ONLY the
provided insights.

When the insights support it, organise the answer into concise sections such as:
- Core Identity / Gameplan
- Opening / Early Game
- Economy / Macro
- Micro / Execution
- Scouting / Adaptation
- Composition / Fights
- Map Control / Win Condition

Only include sections that are actually supported by the retrieved insights.
Do not invent game-specific advice that is not grounded in the retrieved text.
""".strip()

GENERIC_USER = """
Player question: {question}

Relevant insights:
{insights}

Answer using only the insights above.
""".strip()

AOE2_MATCHUP_SYSTEM = """
You are an Age of Empires II coaching assistant. You have insights about two
civilizations plus shared general AoE2 strategy context.

Explain how {subject_a} should play against {subject_b} using ONLY the provided
insights. Synthesize the interaction instead of listing both civilizations in
isolation. Do not invent advice not grounded in the retrieved text.

Structure the answer when supported:
- Opening / Early Game
- Midgame Transitions
- Key Interactions / Unit Choices
- Win Condition

Only include sections supported by the insights.
""".strip()

AOE2_MATCHUP_USER = """
Player question: {question}

=== {subject_a} insights ===
{insights_a}

=== {subject_b} insights ===
{insights_b}

=== Shared AoE2 context ===
{general_insights}

Using only the insights above, explain how {subject_a} should play against {subject_b}.
""".strip()


# ── Intent detection ──────────────────────────────────────────────────────────

_CHAMPION_LOOKUP: dict[str, str] | None = None  # {normalized_lowercase: canonical}
_AOE2_LOOKUP: dict[str, str] | None = None

# Common shorthands and nicknames players actually type
_CHAMPION_ALIASES: dict[str, str] = {
    "cassio": "Cassiopeia", "cass": "Cassiopeia",
    "mf": "Miss Fortune",
    "tf": "Twisted Fate",
    "yi": "Master Yi",
    "j4": "Jarvan IV", "jarvan": "Jarvan IV",
    "gp": "Gangplank",
    "naut": "Nautilus",
    "morg": "Morgana",
    "asol": "Aurelion Sol",
    "wb": "Wukong",
    "sej": "Sejuani",
    "kog": "Kog'Maw", "kogmaw": "Kog'Maw",
    "xin": "Xin Zhao",
    "vlad": "Vladimir",
    "mundo": "Dr. Mundo",
    "malph": "Malphite",
    "fiddle": "Fiddlesticks", "fiddlesticks": "Fiddlesticks",
    "kass": "Kassadin",
    "kata": "Katarina",
    "liss": "Lissandra",
    "ori": "Orianna",
    "panth": "Pantheon",
    "rengar": "Rengar",
    "trist": "Tristana",
    "zyra": "Zyra",
    "heimer": "Heimerdinger",
    "hwei": "Hwei",
    "khazix": "Kha'Zix", "kha": "Kha'Zix",
    "chogath": "Cho'Gath", "cho": "Cho'Gath",
    "veig": "Veigar",
    "ww": "Warwick",
    "twitch": "Twitch",
    "rek": "Rek'Sai", "reksai": "Rek'Sai",
    "trynd": "Tryndamere",
    "mordekaiser": "Mordekaiser", "morde": "Mordekaiser",
    "tahm": "Tahm Kench",
    "aurelion": "Aurelion Sol",
    "ambessa": "Ambessa",
}

def _normalize(name: str) -> str:
    """Strip apostrophes and special chars so 'kaisa' matches \"Kai'Sa\"."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()

def _get_champion_lookup() -> dict[str, str]:
    """
    Build a lookup from normalized lowercase name → canonical name.
    Includes both the raw lowercase key AND a normalized key (no apostrophes etc.)
    so players can type "kaisa" and match "Kai'Sa".
    Pulls from champion_archetypes (broader coverage) and videos DB.
    """
    global _CHAMPION_LOOKUP
    if _CHAMPION_LOOKUP is None:
        from core.database import get_connection
        with get_connection() as conn:
            arch = conn.execute(
                "SELECT DISTINCT champion FROM champion_archetypes"
            ).fetchall()
            vids = conn.execute(
                "SELECT DISTINCT champion FROM videos WHERE champion IS NOT NULL AND champion != ''"
            ).fetchall()
        names = {r["champion"] for r in arch} | {r["champion"] for r in vids}
        lookup: dict[str, str] = {}
        for name in names:
            lookup[name.lower()] = name          # exact lowercase
            lookup[_normalize(name)] = name      # normalized (strips apostrophes etc.)
        for alias, canonical in _CHAMPION_ALIASES.items():
            lookup[alias] = canonical
        _CHAMPION_LOOKUP = lookup
    return _CHAMPION_LOOKUP


def _get_aoe2_lookup() -> dict[str, str]:
    global _AOE2_LOOKUP
    if _AOE2_LOOKUP is None:
        lookup: dict[str, str] = {}
        for name in AOE2_CIVILIZATIONS:
            lookup[name.lower()] = name
            lookup[_normalize(name)] = name
        _AOE2_LOOKUP = lookup
    return _AOE2_LOOKUP


def detect_aoe2_intent(question: str) -> dict:
    """
    Detect civilization-vs-civilization intent for AoE2 questions.

    Returns:
        {"type": "matchup", "a": "Franks", "b": "Hindustanis"}
        {"type": "general"}
    """
    q = question.lower()
    q_norm = _normalize(q)
    lookup = _get_aoe2_lookup()

    found_with_pos: list[tuple[int, str]] = []
    seen_spans: list[tuple[int, int]] = []

    for name_key in sorted(lookup, key=len, reverse=True):
        for search_q in (q, q_norm):
            m = re.search(r"\b" + re.escape(name_key) + r"\b", search_q)
            if m:
                start, end = m.start(), m.end()
                if any(s <= start < e or s < end <= e for s, e in seen_spans):
                    break
                canonical = lookup[name_key]
                if not any(c == canonical for _, c in found_with_pos):
                    found_with_pos.append((start, canonical))
                seen_spans.append((start, end))
                break

    found_with_pos.sort(key=lambda item: item[0])
    found = [civilization for _, civilization in found_with_pos]
    if len(found) < 2:
        return {"type": "general"}

    a, b = found[0], found[1]
    if re.search(r"\b(into|vs\.?|versus|against)\b", q):
        return {"type": "matchup", "a": a, "b": b}
    return {"type": "matchup", "a": a, "b": b}


def detect_intent(question: str) -> dict:
    """
    Detect matchup (X into/vs/against Y) or synergy (X with/alongside Y) intent.

    Returns one of:
        {"type": "matchup", "a": "Kai'Sa", "b": "Yunara"}
        {"type": "synergy", "a": "Jinx",   "b": "Lulu"}
        {"type": "general"}
    """
    q = question.lower()
    q_norm = _normalize(q)
    lookup = _get_champion_lookup()

    # Find all champion names mentioned — track position so we preserve question order
    # (longest key first to avoid "nunu" matching before "nunu & willump")
    found_with_pos: list[tuple[int, str]] = []  # (start_pos, canonical)
    seen_spans: list[tuple[int, int]] = []

    for name_key in sorted(lookup, key=len, reverse=True):
        for search_q in (q, q_norm):
            m = re.search(r'\b' + re.escape(name_key) + r'\b', search_q)
            if m:
                start, end = m.start(), m.end()
                if any(s <= start < e or s < end <= e for s, e in seen_spans):
                    break
                canonical = lookup[name_key]
                if not any(c == canonical for _, c in found_with_pos):
                    found_with_pos.append((start, canonical))
                seen_spans.append((start, end))
                break

    # Sort by position in question so "kaisa into yunara" → a=Kai'Sa, b=Yunara
    found_with_pos.sort(key=lambda x: x[0])
    found = [c for _, c in found_with_pos]

    if len(found) < 2:
        return {"type": "general"}

    a, b = found[0], found[1]

    if re.search(r'\b(into|vs\.?|versus|against)\b', q):
        return {"type": "matchup", "a": a, "b": b}
    if re.search(r'\b(with|alongside|pairing|combo)\b', q):
        return {"type": "synergy", "a": a, "b": b}

    # Two champions mentioned with "and" in a bot-lane context → synergy
    if re.search(r'\band\b', q) and re.search(r'\b(bot|lane|support|adc|duo)\b', q):
        return {"type": "synergy", "a": a, "b": b}

    # Two champions, no clear connector — default to matchup
    return {"type": "matchup", "a": a, "b": b}


# ── BM25 ──────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _bm25_scores(query_tokens: list[str], corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    n = len(corpus)
    if n == 0:
        return np.array([], dtype=np.float32)
    doc_lens = np.array([len(d) for d in corpus], dtype=np.float32)
    avgdl = doc_lens.mean() if doc_lens.mean() > 0 else 1.0
    df = {t: sum(1 for d in corpus if t in set(d)) for t in set(query_tokens)}
    scores = np.zeros(n, dtype=np.float32)
    for term in query_tokens:
        idf = math.log((n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
        for i, doc in enumerate(corpus):
            tf = doc.count(term)
            if tf == 0:
                continue
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_lens[i] / avgdl)
            scores[i] += idf * numerator / denominator
    return scores


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _rrf_fuse(ranked_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused, key=lambda i: fused[i], reverse=True)


# ── Core retrieval ────────────────────────────────────────────────────────────

def retrieve(
    question: str,
    role: str | None = None,
    champion: str | None = None,
    subject: str | None = None,
    insight_type: str | None = None,
    preferred_types: list[str] | None = None,
    situation_tags: list[str] | None = None,
    game: str = DEFAULT_GAME,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Return top_k most relevant insights via RRF (semantic + BM25).

    Layer 1: direct video insights (filtered by champion/role/type if given).
    Layer 2: generalizable archetype insights blended into remaining slots.
    """
    game = normalize_game(game)
    if subject is None:
        subject = champion
    ids, texts, metadata, matrix = load_all_vectors(
        game=game,
        role=role,
        subject=subject,
        champion=champion,
        insight_type=insight_type,
    )
    if not ids:
        if game == "aoe2" and subject:
            return get_applicable_insights(
                subject,
                preferred_types=preferred_types,
                situation_tags=situation_tags,
                top_k=top_k,
            )
        return []

    n = len(ids)
    fetch_k = min(n, max(top_k * 3, 50))

    model = SentenceTransformer(MODEL_NAME)
    query_vec = model.encode(question, convert_to_numpy=True, normalize_embeddings=True)
    cosine_scores = matrix @ query_vec
    semantic_ranked = np.argsort(cosine_scores)[::-1][:fetch_k].tolist()

    corpus_tokens = [_tokenize(t) for t in texts]
    query_tokens = _tokenize(question)
    bm25 = _bm25_scores(query_tokens, corpus_tokens)
    bm25_ranked = np.argsort(bm25)[::-1][:fetch_k].tolist()

    fused_indices = _rrf_fuse([semantic_ranked, bm25_ranked])
    candidate_count = min(len(fused_indices), max(top_k * 4, 60))
    fused_indices = fused_indices[:candidate_count]

    confidences = np.array([m.get("confidence") or 0.5 for m in metadata])
    source_scores = np.array([m.get("source_score") or 0.5 for m in metadata])
    # Discord coaching sessions carry higher trust — real coach/student interactions
    # are more signal-dense than scraped YouTube guides, so boost them at ranking time.
    source_weights = np.array([_source_weight(m) for m in metadata])
    combined = (0.6 * confidences + 0.4 * source_scores)
    sim_scores = matrix @ query_vec
    preferred_map = {
        insight: max(0.0, 0.12 - (index * 0.02))
        for index, insight in enumerate(preferred_types or [])
    }

    fused_indices.sort(
        key=lambda i: (
            0.5 * float(sim_scores[i])
            + 0.5 * float(combined[i])
            + preferred_map.get(metadata[i]["insight_type"], 0.0)
        ) * float(source_weights[i]),
        reverse=True,
    )

    results = []
    for i in fused_indices[:top_k]:
        results.append({
            "text": texts[i],
            "insight_type": metadata[i]["insight_type"],
            "role": metadata[i]["role"],
            "subject": metadata[i].get("subject"),
            "subject_type": metadata[i].get("subject_type"),
            "champion": metadata[i]["champion"],
            "game": metadata[i].get("game", game),
            "rank": metadata[i].get("rank"),
            "website_rating": metadata[i].get("website_rating"),
            "source": metadata[i].get("source", "discord"),
            "source_weight": round(float(source_weights[i]), 4),
            "score": round(float(cosine_scores[i]), 4),
            "confidence": round(float(confidences[i]), 4),
            "retrieval_layer": "direct",
        })

    if game == "aoe2" and subject:
        _blend_aoe2_applicable_insights(
            results,
            subject,
            preferred_types=preferred_types or [],
            situation_tags=situation_tags or [],
            top_k=top_k,
        )

    if game == "lol" and champion:
        _blend_archetype_insights(results, champion, top_k)

    return results


def _aoe2_query_profile(question: str) -> dict[str, object]:
    q = question.lower()
    preferred: list[str] = []
    situation_tags: list[str] = []
    notes: list[str] = []
    is_detail = bool(
        re.search(r"\b(detail|detailed|in[- ]?depth|step by step|full guide|explain more)\b", q)
    )
    is_civ_overview = bool(
        re.search(r"\b(how (?:should|do) i play|how to play|playstyle|guide|gameplan)\b", q)
    )

    def add(*insight_types: str) -> None:
        for insight_type in insight_types:
            if insight_type not in preferred:
                preferred.append(insight_type)

    def add_tag(*tags: str) -> None:
        for tag in tags:
            if tag not in situation_tags:
                situation_tags.append(tag)

    age_patterns = [
        ("dark_age", r"\bdark age\b"),
        ("feudal_age", r"\bfeudal\b"),
        ("castle_age", r"\bcastle age\b|\bcastle\b"),
        ("imperial_age", r"\bimperial age\b|\bimperial\b|\bimp\b"),
    ]
    for age_bucket, pattern in age_patterns:
        if re.search(pattern, q):
            add(age_bucket)
            if age_bucket == "dark_age":
                add_tag("dark_age")
            elif age_bucket == "castle_age":
                add_tag("castle_timing")
            elif age_bucket == "imperial_age":
                add_tag("imperial_transition")
            break

    if re.search(r"\b(hotkey|hotkeys|control group|control groups|grouping|ui|interface|shortcut|shortcuts|camera|keybind|keybinds|settings)\b", q):
        add("controls_settings", "micro")
        notes.append(
            "If the question is about hotkeys, control groups, or UI setup, lead with controls/settings advice before broader strategy."
        )

    if re.search(r"\b(wildlife|boar|boars|wolf|wolves|armor class|armour class|bonus damage|attack bonus|conversion|convert|monk conversion|elevation|hill bonus|high ground|projectile|projectiles|ballistics|minimum range|line of sight|pathing|collision|building armor|building armour|pierce armor|pierce armour)\b", q):
        add("game_mechanics", "unit_compositions")
        if re.search(r"\b(convert|conversion|monk conversion)\b", q):
            add_tag("monks")
        notes.append(
            "If the question is about AoE2 interaction rules, explain the underlying game mechanics before giving composition or strategic advice."
        )

    if re.search(r"\b(micro|split micro|focus fire|formation|formations|stutter|kiting|dodge|quickwall|army control)\b", q):
        add("micro", "unit_compositions")
        notes.append(
            "When the question is about execution, include concrete unit-control details and avoid drifting into vague macro filler."
        )

    if re.search(r"\b(defend|defense|defence|hold|survive|stabilize|stabilise|under attack|against pressure|vs pressure|stop a rush)\b", q):
        add("scouting", "economy_macro", "micro", "map_control", "matchup_advice")
        add_tag("defense")
        notes.append(
            "Frame the answer around scouting the threat, stabilizing efficiently, and transitioning back to a healthy economy."
        )
    elif re.search(r"\b(attack|aggression|aggressive|pressure|push|timing attack|allin|all-in|raid|raiding)\b", q):
        add("build_orders", "feudal_age", "castle_age", "unit_compositions", "map_control", "micro")
        add_tag("feudal_pressure")
        if re.search(r"\b(raid|raiding)\b", q):
            add_tag("raiding")
        notes.append(
            "Frame the answer around setup, timing, execution, and what to do if the initial attack does not end the game."
        )

    if re.search(r"\b(boom|booming|eco|economy|villager|villagers|farm|farms|town center|tc uptime)\b", q):
        add_tag("boom", "economy")

    if re.search(r"\b(scout|scouting)\b", q):
        add_tag("scouting")

    if re.search(r"\b(relic|relics|hill|hills|map control|control the map)\b", q):
        add_tag("map_control")

    if re.search(r"\b(cavalry|cav|knight|scout cavalry|paladin|camel)\b", q):
        add_tag("cavalry")
    if re.search(r"\b(archer|archers|xbow|crossbow|cavalry archer|cav archer|skirm|skirmisher)\b", q):
        add_tag("archers")
    if re.search(r"\b(infantry|militia|maa|man at arms|longsword|champion line|spearman|pikeman|halberdier)\b", q):
        add_tag("infantry")
    if re.search(r"\b(siege|mangonel|onager|scorpion|ram|trebuchet|bbc|bombard cannon)\b", q):
        add_tag("siege")
    if re.search(r"\b(monk|monks|conversion|convert)\b", q):
        add_tag("monks")
    if re.search(r"\b(switch|transition|tech switch)\b", q):
        add_tag("tech_switch")

    if is_civ_overview:
        add(
            "civilization_identity",
            "build_orders",
            "dark_age",
            "feudal_age",
            "castle_age",
            "imperial_age",
            "economy_macro",
            "unit_compositions",
            "map_control",
        )
        add_tag("dark_age", "feudal_pressure", "castle_timing", "imperial_transition")
        notes.append(
            "For a civilization overview, include a practical opening plan and describe how the game should progress through each age."
        )

    if is_detail:
        notes.append(
            "The player asked for detail, so give a step-by-step answer with more concrete checkpoints and common mistakes."
        )

    if not preferred:
        add("principles", "economy_macro")

    guidance = ""
    if notes:
        guidance = "\n\nAoE2 answer guidance:\n- " + "\n- ".join(notes)
    if is_civ_overview:
        guidance += (
            "\n\nFor civilization overview questions, structure the answer with these sections when supported by the retrieved insights: "
            "Core Identity, Opening / First Minutes, Dark Age, Feudal Age, Castle Age, Imperial / Win Condition, Common Mistakes."
        )

    return {
        "preferred_types": preferred,
        "situation_tags": situation_tags,
        "guidance": guidance,
        "detail": is_detail,
        "civ_overview": is_civ_overview,
    }


def _merge_ranked_results(*result_sets: list[dict], limit: int) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for result_set in result_sets:
        for row in result_set:
            key = (row.get("text") or "", row.get("insight_type") or "")
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    merged.sort(
        key=lambda r: (
            float(r.get("score") or 0.0) * 0.5
            + float(r.get("confidence") or 0.0) * 0.5
        ) * float(r.get("source_weight") or 1.0),
        reverse=True,
    )
    return merged[:limit]


def _blend_archetype_insights(results: list[dict], champion: str, top_k: int) -> None:
    archetype_hits = get_archetype_insights(champion, top_k=top_k)
    if not archetype_hits:
        return
    existing_texts = {r["text"] for r in results}
    additions = [
        {
            "text": h["text"],
            "insight_type": h["insight_type"],
            "role": None,
            "champion": h["source_champion"],
            "rank": None,
            "website_rating": None,
            "source": "archetype",
            "source_weight": 1.0,
            "score": round(h["similarity"], 4),
            "confidence": round(h["confidence"], 4),
            "retrieval_layer": "archetype",
        }
        for h in archetype_hits
        if h["text"] not in existing_texts
    ]
    slots = max(0, top_k - len(results))
    if slots > 0:
        results.extend(additions[:slots])


def _blend_aoe2_applicable_insights(
    results: list[dict],
    subject: str,
    preferred_types: list[str],
    situation_tags: list[str],
    top_k: int,
) -> None:
    hits = get_applicable_insights(
        subject,
        preferred_types=preferred_types,
        situation_tags=situation_tags,
        top_k=top_k,
    )
    if not hits:
        return
    existing_texts = {row["text"] for row in results}
    additions = [hit for hit in hits if hit["text"] not in existing_texts]
    slots = max(0, top_k - len(results))
    if slots > 0:
        results.extend(additions[:slots])


# ── Duo retrieval (matchup / synergy) ─────────────────────────────────────────

def _fetch_ability_windows(champion: str, limit: int = 5) -> list[dict]:
    """
    Pull pre-generated ability_windows insights from DB for a champion.
    These are always included in matchup/synergy context regardless of embedding.
    """
    from core.database import get_connection
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT i.text FROM insights i
            JOIN videos v ON i.video_id = v.video_id
            WHERE v.champion = ? AND i.insight_type = 'ability_windows'
            ORDER BY i.id
            LIMIT ?
            """,
            (champion, limit)
        ).fetchall()
    return [
        {
            "text": r["text"],
            "insight_type": "ability_windows",
            "role": None,
            "champion": champion,
            "rank": None,
            "website_rating": None,
            "source": "ability_enrichment",
            "source_weight": 1.0,
            "score": 1.0,
            "confidence": 0.8,
            "retrieval_layer": "ability_enrichment",
        }
        for r in rows
    ]


def _fetch_specific_matchup_notes(
    champion_a: str,
    champion_b: str,
    role: str | None = None,
    game: str = DEFAULT_GAME,
    limit: int = 8,
) -> list[dict]:
    """
    Pull explicit champion-vs-champion notes first from the dedicated
    champion_matchups bucket, then from broader matchup_advice when the text
    explicitly names champion_b. This is meant for X-into-Y queries where exact
    written-guide matchup notes are higher fidelity than general champion data.
    """
    import pathlib
    import core.database as _db

    enemy_patterns = {
        champion_b.lower(),
        _normalize(champion_b),
    }

    hits: list[dict] = []
    seen_texts: set[str] = set()
    for db_path in all_content_db_paths():
        if not pathlib.Path(db_path).exists():
            continue
        _db.DB_PATH = pathlib.Path(db_path)
        with _db.get_connection() as conn:
            query = """
            SELECT i.text, i.insight_type, i.confidence, i.source_score,
                   v.role, COALESCE(v.subject, v.champion) AS subject, v.champion, v.game,
                   v.rank, v.website_rating, COALESCE(v.source, 'discord') AS source
            FROM insights i
                JOIN videos v ON i.video_id = v.video_id
                WHERE v.champion = ?
                  AND v.game = ?
                  AND i.insight_type IN ('champion_matchups', 'matchup_advice')
            """
            params: list = [champion_a, normalize_game(game)]
            if role:
                query += " AND v.role = ?"
                params.append(role)
            query += " ORDER BY CASE WHEN i.insight_type = 'champion_matchups' THEN 0 ELSE 1 END, i.id"
            rows = conn.execute(query, params).fetchall()

        for row in rows:
            text = row["text"]
            text_norm = _normalize(text)
            text_low = text.lower()
            if not any(
                re.search(r"\b" + re.escape(name) + r"\b", text_low) or
                re.search(r"\b" + re.escape(name) + r"\b", text_norm)
                for name in enemy_patterns
            ):
                continue
            if text in seen_texts:
                continue
            seen_texts.add(text)
            hits.append({
                "text": text,
                "insight_type": row["insight_type"],
                "role": row["role"],
                "subject": row["subject"],
                "champion": row["champion"],
                "game": row["game"],
                "rank": row["rank"],
                "website_rating": row["website_rating"],
                "source": row["source"],
                "score": 1.0 if row["insight_type"] == "champion_matchups" else 0.9,
                "confidence": round(float(row["confidence"] or row["source_score"] or 0.75), 4),
                "source_weight": round(float(_source_weight(dict(row))), 4),
                "retrieval_layer": "exact_matchup",
            })
    hits.sort(
        key=lambda r: (
            1 if r["insight_type"] == "champion_matchups" else 0,
            float(r.get("source_weight") or 1.0),
            float(r.get("confidence") or 0.0),
            float(r.get("website_rating") or 0.0),
        ),
        reverse=True,
    )
    return hits[:limit]


def _fetch_specific_aoe2_matchup_notes(
    subject_a: str,
    subject_b: str,
    limit: int = 8,
) -> list[dict]:
    import pathlib
    import core.database as _db

    subject_a = canonical_aoe2_civilization(subject_a) or subject_a
    subject_b = canonical_aoe2_civilization(subject_b) or subject_b
    enemy_patterns = {
        subject_b.lower(),
        _normalize(subject_b),
    }

    hits: list[dict] = []
    seen_texts: set[str] = set()
    for db_path in all_content_db_paths():
        if not pathlib.Path(db_path).exists():
            continue
        _db.DB_PATH = pathlib.Path(db_path)
        with _db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT i.text, i.insight_type, i.confidence, i.source_score,
                       i.subject, i.subject_type, v.game, COALESCE(v.source, 'discord') AS source
                FROM insights i
                JOIN videos v ON i.video_id = v.video_id
                WHERE v.game = 'aoe2'
                  AND LOWER(COALESCE(i.subject, v.subject)) = LOWER(?)
                  AND i.insight_type = 'matchup_advice'
                ORDER BY i.id
                """,
                (subject_a,),
            ).fetchall()

        for row in rows:
            text = row["text"]
            text_norm = _normalize(text)
            text_low = text.lower()
            if not any(
                re.search(r"\b" + re.escape(name) + r"\b", text_low)
                or re.search(r"\b" + re.escape(name) + r"\b", text_norm)
                for name in enemy_patterns
            ):
                continue
            if text in seen_texts:
                continue
            seen_texts.add(text)
            meta = dict(row)
            meta["source_subject"] = subject_a
            hits.append({
                "text": text,
                "insight_type": "matchup_advice",
                "role": None,
                "subject": row["subject"],
                "subject_type": row["subject_type"],
                "champion": None,
                "game": "aoe2",
                "rank": None,
                "website_rating": None,
                "source": row["source"],
                "score": 1.0,
                "confidence": round(float(row["confidence"] or row["source_score"] or 0.75), 4),
                "source_weight": round(float(_source_weight(meta)), 4),
                "retrieval_layer": "exact_matchup",
            })

    hits.sort(
        key=lambda item: (
            float(item.get("source_weight") or 1.0),
            float(item.get("confidence") or 0.0),
        ),
        reverse=True,
    )
    return hits[:limit]


def retrieve_duo(
    question: str,
    champion_a: str,
    champion_b: str,
    role: str | None = None,
    game: str = DEFAULT_GAME,
    top_k: int = TOP_K,
) -> tuple[list[dict], list[dict]]:
    """
    Retrieve insights for two champions independently.
    champion_a gets role filter (the player's champion); champion_b is unfiltered.

    If a champion has no direct video coverage, falls back to archetype retrieval
    so the LLM gets same-archetype insights rather than nothing (which causes
    hallucination from the opponent's data).

    Ability_windows are always appended from DB (no embedding required) since
    they are pre-generated by pipeline/ability_enrich.py.

    Returns (insights_a, insights_b).
    """
    game = normalize_game(game)
    per_side = max(top_k // 2, 6)

    matchup_notes_a: list[dict] = []
    matchup_notes_b: list[dict] = []
    if game == "lol":
        matchup_notes_a = _fetch_specific_matchup_notes(
            champion_a, champion_b, role=role, game=game, limit=max(4, per_side // 2)
        )
        matchup_notes_b = _fetch_specific_matchup_notes(
            champion_b, champion_a, game=game, limit=max(4, per_side // 2)
        )

    insights_a = matchup_notes_a + retrieve(question, champion=champion_a, role=role, game=game, top_k=per_side)
    if not insights_a:
        insights_a = _archetype_fallback(champion_a, per_side) if game == "lol" else []

    insights_b = matchup_notes_b + retrieve(question, champion=champion_b, game=game, top_k=per_side)
    if not insights_b:
        insights_b = _archetype_fallback(champion_b, per_side) if game == "lol" else []

    # Keep exact matchup notes first, then dedupe by text.
    deduped_a: list[dict] = []
    seen_a: set[str] = set()
    for row in insights_a:
        if row["text"] in seen_a:
            continue
        seen_a.add(row["text"])
        deduped_a.append(row)
    insights_a = deduped_a[: max(per_side + len(matchup_notes_a), per_side)]

    deduped_b: list[dict] = []
    seen_b: set[str] = set()
    for row in insights_b:
        if row["text"] in seen_b:
            continue
        seen_b.add(row["text"])
        deduped_b.append(row)
    insights_b = deduped_b[: max(per_side + len(matchup_notes_b), per_side)]

    # Append ability_windows for both sides (these capture CC/mobility interactions
    # that are critical for matchup context — not retrieved by embedding search)
    windows_a = _fetch_ability_windows(champion_a)
    windows_b = _fetch_ability_windows(champion_b)
    existing_a = {r["text"] for r in insights_a}
    existing_b = {r["text"] for r in insights_b}
    insights_a.extend(w for w in windows_a if w["text"] not in existing_a)
    insights_b.extend(w for w in windows_b if w["text"] not in existing_b)

    return insights_a, insights_b


def _retrieve_aoe2_duo(
    question: str,
    subject_a: str,
    subject_b: str,
    top_k: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    per_side = max(top_k // 2, 6)
    profile = _aoe2_query_profile(question)

    matchup_notes_a = _fetch_specific_aoe2_matchup_notes(subject_a, subject_b, limit=max(4, per_side // 2))
    matchup_notes_b = _fetch_specific_aoe2_matchup_notes(subject_b, subject_a, limit=max(4, per_side // 2))

    insights_a = matchup_notes_a + retrieve(
        question,
        subject=subject_a,
        preferred_types=profile["preferred_types"],
        situation_tags=profile["situation_tags"],
        game="aoe2",
        top_k=per_side,
    )
    _blend_aoe2_applicable_insights(
        insights_a,
        subject_a,
        preferred_types=profile["preferred_types"],
        situation_tags=profile["situation_tags"],
        top_k=max(per_side + 4, top_k),
    )

    insights_b = matchup_notes_b + retrieve(
        question,
        subject=subject_b,
        preferred_types=profile["preferred_types"],
        situation_tags=profile["situation_tags"],
        game="aoe2",
        top_k=per_side,
    )
    _blend_aoe2_applicable_insights(
        insights_b,
        subject_b,
        preferred_types=profile["preferred_types"],
        situation_tags=profile["situation_tags"],
        top_k=max(per_side + 4, top_k),
    )

    general_insights = retrieve(
        question,
        subject=None,
        preferred_types=profile["preferred_types"],
        situation_tags=profile["situation_tags"],
        game="aoe2",
        top_k=max(4, top_k // 3),
    )

    deduped_a: list[dict] = []
    seen_a: set[str] = set()
    for row in insights_a:
        if row["text"] in seen_a:
            continue
        seen_a.add(row["text"])
        deduped_a.append(row)

    deduped_b: list[dict] = []
    seen_b: set[str] = set()
    for row in insights_b:
        if row["text"] in seen_b:
            continue
        seen_b.add(row["text"])
        deduped_b.append(row)

    return deduped_a[:per_side], deduped_b[:per_side], general_insights


def _fetch_stat_notes(champion: str, top_n: int = 3) -> list[str]:
    """Pull the most extreme pre-computed stat anomaly notes for a champion."""
    from core.database import get_connection
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT note FROM champion_stat_notes
            WHERE champion = ?
            ORDER BY ABS(z_score) DESC
            LIMIT ?
            """,
            (champion, top_n)
        ).fetchall()
    return [r["note"] for r in rows]


def _range_matchup_note(champion_a: str, champion_b: str) -> str | None:
    """
    Compare attack ranges of two champions and return a situational note
    only when the gap is significant (≥100 units). This is computed at query
    time since low range is only meaningful relative to the opponent.
    """
    from core.database import get_connection
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT champion, attack_range FROM champion_stats WHERE champion IN (?, ?)",
            (champion_a, champion_b)
        ).fetchall()

    ranges = {r["champion"]: r["attack_range"] for r in rows if r["attack_range"]}
    ra = ranges.get(champion_a)
    rb = ranges.get(champion_b)
    if ra is None or rb is None:
        return None

    diff = rb - ra  # positive = B has more range
    if abs(diff) < 100:
        return None  # not significant enough to call out

    longer = champion_b if diff > 0 else champion_a
    shorter = champion_a if diff > 0 else champion_b
    gap = abs(int(diff))
    longer_range = int(rb if diff > 0 else ra)
    shorter_range = int(ra if diff > 0 else rb)

    return (
        f"Range mismatch: {longer} ({longer_range}) has {gap} more range than "
        f"{shorter} ({shorter_range}) — {shorter} must play closer to trade, "
        f"giving {longer} the poke/auto advantage at all times."
    )


def _archetype_fallback(champion: str, top_k: int) -> list[dict]:
    """
    When a champion has no direct video coverage, pull generalizable insights
    from same-archetype champions as a labeled substitute.
    """
    hits = get_archetype_insights(champion, top_k=top_k)
    if not hits:
        return []
    return [
        {
            "text": h["text"],
            "insight_type": h["insight_type"],
            "role": None,
            "champion": h["source_champion"],
            "rank": None,
            "website_rating": None,
            "source": "archetype",
            "source_weight": 1.0,
            "score": round(h["similarity"], 4),
            "confidence": round(h["confidence"], 4),
            "retrieval_layer": "archetype",
        }
        for h in hits
    ]


def _coverage_note(insights: list[dict], champion: str) -> str:
    """Return a note for the LLM explaining data source fidelity."""
    if not insights:
        return f"(No data available for {champion}.)"
    if all(r.get("retrieval_layer") == "archetype" for r in insights):
        archetypes = {r.get("champion") for r in insights if r.get("champion")}
        return (
            f"(No direct video coverage for {champion}. "
            f"Insights below are from same-archetype champions: {', '.join(sorted(archetypes))}. "
            f"Reason from their shared playstyle to advise {champion}.)"
        )
    return ""


def _format_insights(insights: list[dict]) -> str:
    if not insights:
        return "  (no direct video coverage — no insights available)"
    lines = []
    for r in insights:
        layer_tag = {
            "archetype": " [archetype]",
            "aoe2_crossref": " [crossref]",
            "ability_enrichment": " [ability]",
            "exact_matchup": " [exact-matchup]",
        }.get(r.get("retrieval_layer", ""), "")
        source_tag = f" | {_source_label(r)}" if r.get("source") else ""
        # Explicitly label the champion on ability_windows rows so the LLM
        # cannot mistake which champion's ability is being described.
        champ_tag = ""
        if r.get("insight_type") == "ability_windows" and r.get("champion"):
            champ_tag = f" | {r['champion']}'s ability"
        lines.append(f"  - [{r['insight_type']}{layer_tag}{champ_tag}{source_tag}] {r['text']}")
    return "\n".join(lines)


# ── Champion overview multi-query expansion ───────────────────────────────────

_HOW_TO_PLAY_RE = re.compile(
    r'\b(how (do i|to|should i) play|guide (for|on|to)|tips (for|on)|'
    r'how (do i|to) (get good|improve|learn|master)|'
    r'what (should|do) i do (as|playing|on)|general (tips|guide|gameplan))\b',
    re.IGNORECASE,
)


def _is_how_to_play(question: str) -> bool:
    return bool(_HOW_TO_PLAY_RE.search(question))


def _multi_retrieve_champion(champion: str, role: str | None, top_k: int) -> list[dict]:
    """
    For broad 'how do I play X' questions, run targeted sub-queries per game phase
    and merge results so all sections of the response have coverage.
    """
    sub_queries = [
        (f"What is {champion}'s champion identity, win condition, and strategic gameplan?",
         ["champion_identity", "principles"]),
        (f"How do I play {champion} in lane early game, trade patterns and level spikes?",
         ["laning_tips", "champion_mechanics"]),
        (f"What direct champion-specific matchup notes exist for {champion} against named enemy champions?",
         ["champion_matchups", "matchup_advice"]),
        (f"What are {champion}'s key ability mechanics, passive interactions, and ability windows?",
         ["ability_windows", "champion_mechanics"]),
        (f"How do I play {champion} in teamfights and mid/late game macro decisions?",
         ["teamfight_tips", "macro_advice"]),
        (f"What items should I build on {champion} and what are their synergies with teammates?",
         ["itemization", "champion_identity"]),
    ]

    per_sub = max(top_k // len(sub_queries), 6)
    seen_texts: set[str] = set()
    merged: list[dict] = []

    for query_text, _ in sub_queries:
        hits = retrieve(query_text, role=role, champion=champion, top_k=per_sub)
        for h in hits:
            if h["text"] not in seen_texts:
                seen_texts.add(h["text"])
                merged.append(h)

    # Re-rank merged pool by score * confidence * source weight
    merged.sort(
        key=lambda r: (r["score"] * 0.5 + r["confidence"] * 0.5)
                      * _source_weight(r),
        reverse=True,
    )
    return merged[:top_k]


# ── Answer ────────────────────────────────────────────────────────────────────

def answer(
    question: str,
    role: str | None = None,
    champion: str | None = None,
    subject: str | None = None,
    insight_type: str | None = None,
    game: str = DEFAULT_GAME,
    top_k: int = TOP_K,
    show_sources: bool = True,
) -> str:
    """
    Full RAG pipeline with automatic intent routing:
      - Matchup (X into/vs Y) → retrieve both sides, synthesize interaction
      - Synergy (X with/and Y) → retrieve both sides, synthesize combo
      - General               → single retrieve + answer
    """
    game = normalize_game(game)
    if subject is None:
        subject = champion

    if game != "lol":
        if game == "aoe2" and not subject:
            intent = detect_aoe2_intent(question)
            if intent["type"] == "matchup":
                return _answer_aoe2_duo(question, intent, top_k=top_k, show_sources=show_sources)
        aoe2_profile = _aoe2_query_profile(question) if game == "aoe2" else {
            "preferred_types": None,
            "situation_tags": None,
            "guidance": "",
            "detail": False,
            "civ_overview": False,
        }
        effective_top_k = max(top_k, 24) if aoe2_profile.get("detail") else top_k
        insights = retrieve(
            question,
            role=role,
            subject=subject,
            insight_type=insight_type,
            preferred_types=aoe2_profile["preferred_types"],
            situation_tags=aoe2_profile["situation_tags"],
            game=game,
            top_k=effective_top_k,
        )
        if subject:
            general_top_k = max(8, effective_top_k // 2) if aoe2_profile.get("civ_overview") else max(4, effective_top_k // 3)
            general_hits = retrieve(
                question,
                role=role,
                subject=None,
                insight_type=insight_type,
                preferred_types=aoe2_profile["preferred_types"],
                situation_tags=aoe2_profile["situation_tags"],
                game=game,
                top_k=general_top_k,
            )
            insights = _merge_ranked_results(insights, general_hits, limit=effective_top_k)
        if not insights:
            return "No relevant insights found. Make sure embed.py has been run."

        formatted = "\n".join(
            f"{i + 1}. [{r['insight_type']} | {r.get('role') or 'n/a'}"
            + (f" | {r.get('subject')}" if r.get("subject") else "")
            + (f" | {r.get('subject_type')}" if r.get("subject_type") else "")
            + (f" | {_source_label(r)}" if r.get("source") else "")
            + f"] {r['text']}"
            for i, r in enumerate(insights)
        )

        generated = llm_chat(
            system=GENERIC_SYSTEM.format(game_name=game_label(game)) + str(aoe2_profile["guidance"] or ""),
            user=GENERIC_USER.format(question=question, insights=formatted),
            temperature=0.2,
        )
        if show_sources:
            generated += _sources_block(insights)
        return generated

    # Only auto-detect intent when the user hasn't manually specified a champion
    if not champion:
        intent = detect_intent(question)
    else:
        intent = {"type": "general"}

    if intent["type"] in ("matchup", "synergy"):
        return _answer_duo(question, intent, role=role, game=game, top_k=top_k, show_sources=show_sources)

    # ── General / single-champion path ────────────────────────────────────────
    detected_champ_early = champion
    if not detected_champ_early:
        q_norm = _normalize(question)
        lookup = _get_champion_lookup()
        for name_key in sorted(lookup, key=len, reverse=True):
            if re.search(r'\b' + re.escape(name_key) + r'\b', q_norm):
                detected_champ_early = lookup[name_key]
                break

    if detected_champ_early and _is_how_to_play(question):
        insights = _multi_retrieve_champion(detected_champ_early, role=role, top_k=top_k)
    else:
        insights = retrieve(question, role=role, champion=champion,
                            insight_type=insight_type, top_k=top_k)
    if not insights:
        return "No relevant insights found. Make sure embed.py has been run."

    formatted = "\n".join(
        f"{i + 1}. [{r['insight_type']} | {r['role']}"
        + (f" | {r['champion']}" if r['champion'] else "")
        + (f" | {_source_label(r)}" if r.get("source") else "")
        + f"] {r['text']}"
        for i, r in enumerate(insights)
    )

    detected_champ = detected_champ_early

    stat_block = ""
    if detected_champ:
        stat_ctx = _stat_context_block(detected_champ)
        if stat_ctx:
            stat_block = f"\n\nStat context (factor this into your answer if relevant):\n{stat_ctx}"

    generated = llm_chat(
        system=GENERAL_SYSTEM,
        user=GENERAL_USER.format(question=question, insights=formatted) + stat_block,
        temperature=0.2,
    )

    ability_block = _ability_reference_block([detected_champ]) if detected_champ else ""
    if ability_block:
        generated = ability_block + "\n\n" + generated

    if show_sources:
        generated += _sources_block(insights)
    return generated


def _stat_context_block(champion_a: str, champion_b: str | None = None) -> str:
    """
    Build a stat context block for the prompt. Includes:
    - Pre-computed anomaly notes for each champion (HP, armor, scaling edges)
    - Query-time range matchup note (only when gap ≥ 100 units)
    """
    lines = []

    notes_a = _fetch_stat_notes(champion_a)
    if notes_a:
        lines.append(f"Stat context for {champion_a}:")
        for n in notes_a:
            lines.append(f"  • {n}")

    if champion_b:
        notes_b = _fetch_stat_notes(champion_b)
        if notes_b:
            lines.append(f"Stat context for {champion_b}:")
            for n in notes_b:
                lines.append(f"  • {n}")

        range_note = _range_matchup_note(champion_a, champion_b)
        if range_note:
            lines.append(f"  • {range_note}")

    return "\n".join(lines)


def _answer_duo(
    question: str,
    intent: dict,
    role: str | None,
    game: str,
    top_k: int,
    show_sources: bool,
) -> str:
    a, b = intent["a"], intent["b"]
    mode = intent["type"]

    insights_a, insights_b = retrieve_duo(question, a, b, role=role, game=game, top_k=top_k)

    if not insights_a and not insights_b:
        return f"No insights found for {a} or {b}. Make sure embed.py has been run."

    fmt_a = _format_insights(insights_a)
    fmt_b = _format_insights(insights_b)

    # Note archetype-only sides so the LLM reasons from them rather than refusing
    a_note = _coverage_note(insights_a, a)
    b_note = _coverage_note(insights_b, b)

    # Stat context: pre-computed anomalies + query-time range comparison
    stat_ctx = _stat_context_block(a, b)
    stat_block = f"\nStat context:\n{stat_ctx}\n" if stat_ctx else ""

    if mode == "matchup":
        system = MATCHUP_SYSTEM.format(champion_a=a, champion_b=b)
        user = MATCHUP_USER.format(
            question=question, champion_a=a, champion_b=b,
            insights_a=fmt_a, insights_b=fmt_b,
            note_a=a_note, note_b=b_note,
        ) + stat_block
    else:
        system = SYNERGY_SYSTEM.format(champion_a=a, champion_b=b)
        user = SYNERGY_USER.format(
            question=question, champion_a=a, champion_b=b,
            insights_a=fmt_a, insights_b=fmt_b,
            note_a=a_note, note_b=b_note,
        ) + stat_block

    generated = llm_chat(system=system, user=user, temperature=0.2)

    ability_block = _ability_reference_block([a, b])
    if ability_block:
        generated = ability_block + "\n\n" + generated

    if show_sources:
        generated += f"\n\n---\nSources for {a}:\n" + _sources_block(insights_a, prefix="  ")
        generated += f"\nSources for {b}:\n" + _sources_block(insights_b, prefix="  ")

    return generated


def _answer_aoe2_duo(
    question: str,
    intent: dict,
    top_k: int,
    show_sources: bool,
) -> str:
    a, b = intent["a"], intent["b"]
    insights_a, insights_b, general_insights = _retrieve_aoe2_duo(question, a, b, top_k=top_k)

    if not insights_a and not insights_b and not general_insights:
        return f"No insights found for {a} or {b}. Make sure embed.py has been run."

    fmt_a = _format_insights(insights_a)
    fmt_b = _format_insights(insights_b)
    fmt_general = _format_insights(general_insights)

    generated = llm_chat(
        system=AOE2_MATCHUP_SYSTEM.format(subject_a=a, subject_b=b),
        user=AOE2_MATCHUP_USER.format(
            question=question,
            subject_a=a,
            subject_b=b,
            insights_a=fmt_a,
            insights_b=fmt_b,
            general_insights=fmt_general,
        ),
        temperature=0.2,
    )

    if show_sources:
        generated += f"\n\n---\nSources for {a}:\n" + _sources_block(insights_a, prefix="  ")
        generated += f"\nSources for {b}:\n" + _sources_block(insights_b, prefix="  ")
        if general_insights:
            generated += "\nShared AoE2 context:\n" + _sources_block(general_insights, prefix="  ")

    return generated


def _sources_block(insights: list[dict], prefix: str = "") -> str:
    lines = []
    for r in insights:
        layer = f" | {r['retrieval_layer']}" if r.get("retrieval_layer") else ""
        source = f" | {_source_label(r)}" if r.get("source") else ""
        source_weight = f" | src-w {float(r.get('source_weight') or 1.0):.2f}"
        lines.append(
            f"{prefix}[{r['score']:.2f} | conf {r['confidence']:.2f}{source_weight}{layer}{source}]"
            f" ({r['insight_type']}) {r['text'][:100]}{'...' if len(r['text']) > 100 else ''}"
        )
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Query the LoL coaching knowledge base")
    parser.add_argument("question", nargs="+", help="Your question")
    parser.add_argument("--game", default=DEFAULT_GAME, help="Game namespace (lol, aoe2)")
    parser.add_argument("--role", help="Filter by role (mid/top/jungle/adc/support)")
    parser.add_argument("--champion", help="Force a specific champion filter (skips intent detection)")
    parser.add_argument("--subject", help="Generic subject filter (champion, civ, strategy, etc.)")
    parser.add_argument("--type", dest="insight_type",
                        help="Filter by insight type (principles/laning_tips/macro_advice/etc.)")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of insights to retrieve")
    parser.add_argument("--retrieve-only", action="store_true",
                        help="Show retrieved insights without generating an answer")
    parser.add_argument("--intent", action="store_true",
                        help="Show detected intent and exit")
    args = parser.parse_args()

    question = " ".join(args.question)
    print(f"\nQuestion: {question}")

    if args.intent:
        intent = detect_aoe2_intent(question) if normalize_game(args.game) == "aoe2" else detect_intent(question)
        print(f"Intent: {intent}")
        return

    if args.role:
        print(f"Filter: role={args.role}")
    if args.champion:
        print(f"Filter: champion={args.champion}")
    if args.subject:
        print(f"Filter: subject={args.subject}")
    if args.insight_type:
        print(f"Filter: type={args.insight_type}")
    print(f"Filter: game={normalize_game(args.game)}")
    print()

    if args.retrieve_only:
        profile = _aoe2_query_profile(question) if normalize_game(args.game) == "aoe2" else {
            "preferred_types": None,
            "situation_tags": None,
        }
        results = retrieve(
            question,
            role=args.role,
            champion=args.champion,
            subject=args.subject or args.champion,
            insight_type=args.insight_type,
            preferred_types=profile["preferred_types"],
            situation_tags=profile["situation_tags"],
            game=args.game,
            top_k=args.top_k,
        )
        for r in results:
            layer = f" | {r['retrieval_layer']}" if r.get("retrieval_layer") else ""
            label = r.get("subject") or r.get("champion") or r.get("subject_type") or "-"
            print(
                f"[{r['score']:.3f} | conf {r['confidence']:.2f}{layer}] "
                f"({r['insight_type']} | {label} | {r.get('game') or normalize_game(args.game)}) {r['text']}"
            )
        return

    result = answer(
        question,
        role=args.role,
        champion=args.champion,
        subject=args.subject or args.champion,
        insight_type=args.insight_type,
        game=args.game,
        top_k=args.top_k,
    )
    print(result)


if __name__ == "__main__":
    main()
