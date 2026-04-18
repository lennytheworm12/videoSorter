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
import logging
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from pipeline.embed import load_all_vectors
from pipeline.champion_crossref import get_archetype_insights
from core.llm import chat as llm_chat

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 12
RRF_K = 60


# ── Prompts ───────────────────────────────────────────────────────────────────

GENERAL_SYSTEM = """
You are a League of Legends coaching assistant. You have access to insights
extracted from a real coach's video library. Answer using ONLY the provided
coaching insights — do not add advice not grounded in the retrieved insights.

- Lead with the core principle or mental model if one is present
- Follow with specific actionable tips
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

Insights about {champion_a}: {note_a}
{insights_a}

Insights about {champion_b}: {note_b}
{insights_b}

Using the insights above, explain how {champion_a} should play against {champion_b}.
If a side only has archetype data, reason from the shared playstyle — do not refuse to answer.
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


# ── Intent detection ──────────────────────────────────────────────────────────

_CHAMPION_LOOKUP: dict[str, str] | None = None  # {normalized_lowercase: canonical}

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
        _CHAMPION_LOOKUP = lookup
    return _CHAMPION_LOOKUP


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
    insight_type: str | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Return top_k most relevant insights via RRF (semantic + BM25).

    Layer 1: direct video insights (filtered by champion/role/type if given).
    Layer 2: generalizable archetype insights blended into remaining slots.
    """
    ids, texts, metadata, matrix = load_all_vectors(
        role=role,
        champion=champion,
        insight_type=insight_type,
    )
    if not ids:
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

    fused_indices = _rrf_fuse([semantic_ranked, bm25_ranked])[:top_k]

    confidences = np.array([m.get("confidence") or 0.5 for m in metadata])
    source_scores = np.array([m.get("source_score") or 0.5 for m in metadata])
    combined = (0.6 * confidences + 0.4 * source_scores)
    fused_indices.sort(
        key=lambda i: (matrix[i] @ query_vec) * (0.5 + 0.5 * float(combined[i])),
        reverse=True,
    )

    results = []
    for i in fused_indices[:top_k]:
        results.append({
            "text": texts[i],
            "insight_type": metadata[i]["insight_type"],
            "role": metadata[i]["role"],
            "champion": metadata[i]["champion"],
            "score": round(float(cosine_scores[i]), 4),
            "confidence": round(float(confidences[i]), 4),
            "retrieval_layer": "direct",
        })

    if champion:
        _blend_archetype_insights(results, champion, top_k)

    return results


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


# ── Duo retrieval (matchup / synergy) ─────────────────────────────────────────

def retrieve_duo(
    question: str,
    champion_a: str,
    champion_b: str,
    role: str | None = None,
    top_k: int = TOP_K,
) -> tuple[list[dict], list[dict]]:
    """
    Retrieve insights for two champions independently.
    champion_a gets role filter (the player's champion); champion_b is unfiltered.

    If a champion has no direct video coverage, falls back to archetype retrieval
    so the LLM gets same-archetype insights rather than nothing (which causes
    hallucination from the opponent's data).

    Returns (insights_a, insights_b).
    """
    per_side = max(top_k // 2, 6)

    insights_a = retrieve(question, champion=champion_a, role=role, top_k=per_side)
    if not insights_a:
        insights_a = _archetype_fallback(champion_a, per_side)

    insights_b = retrieve(question, champion=champion_b, top_k=per_side)
    if not insights_b:
        insights_b = _archetype_fallback(champion_b, per_side)

    return insights_a, insights_b


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
        layer = " [archetype]" if r.get("retrieval_layer") == "archetype" else ""
        lines.append(f"  - [{r['insight_type']}{layer}] {r['text']}")
    return "\n".join(lines)


# ── Answer ────────────────────────────────────────────────────────────────────

def answer(
    question: str,
    role: str | None = None,
    champion: str | None = None,
    insight_type: str | None = None,
    top_k: int = TOP_K,
    show_sources: bool = True,
) -> str:
    """
    Full RAG pipeline with automatic intent routing:
      - Matchup (X into/vs Y) → retrieve both sides, synthesize interaction
      - Synergy (X with/and Y) → retrieve both sides, synthesize combo
      - General               → single retrieve + answer
    """
    # Only auto-detect intent when the user hasn't manually specified a champion
    if not champion:
        intent = detect_intent(question)
    else:
        intent = {"type": "general"}

    if intent["type"] in ("matchup", "synergy"):
        return _answer_duo(question, intent, role=role, top_k=top_k, show_sources=show_sources)

    # ── General / single-champion path ────────────────────────────────────────
    insights = retrieve(question, role=role, champion=champion,
                        insight_type=insight_type, top_k=top_k)
    if not insights:
        return "No relevant insights found. Make sure embed.py has been run."

    formatted = "\n".join(
        f"{i + 1}. [{r['insight_type']} | {r['role']}"
        + (f" | {r['champion']}" if r['champion'] else "")
        + f"] {r['text']}"
        for i, r in enumerate(insights)
    )
    generated = llm_chat(
        system=GENERAL_SYSTEM,
        user=GENERAL_USER.format(question=question, insights=formatted),
        temperature=0.2,
    )

    if show_sources:
        generated += _sources_block(insights)
    return generated


def _answer_duo(
    question: str,
    intent: dict,
    role: str | None,
    top_k: int,
    show_sources: bool,
) -> str:
    a, b = intent["a"], intent["b"]
    mode = intent["type"]

    insights_a, insights_b = retrieve_duo(question, a, b, role=role, top_k=top_k)

    if not insights_a and not insights_b:
        return f"No insights found for {a} or {b}. Make sure embed.py has been run."

    fmt_a = _format_insights(insights_a)
    fmt_b = _format_insights(insights_b)

    # Note archetype-only sides so the LLM reasons from them rather than refusing
    a_note = _coverage_note(insights_a, a)
    b_note = _coverage_note(insights_b, b)

    if mode == "matchup":
        system = MATCHUP_SYSTEM.format(champion_a=a, champion_b=b)
        user = MATCHUP_USER.format(
            question=question, champion_a=a, champion_b=b,
            insights_a=fmt_a, insights_b=fmt_b,
            note_a=a_note, note_b=b_note,
        )
    else:
        system = SYNERGY_SYSTEM.format(champion_a=a, champion_b=b)
        user = SYNERGY_USER.format(
            question=question, champion_a=a, champion_b=b,
            insights_a=fmt_a, insights_b=fmt_b,
            note_a=a_note, note_b=b_note,
        )

    generated = llm_chat(system=system, user=user, temperature=0.2)

    if show_sources:
        generated += f"\n\n---\nSources for {a}:\n" + _sources_block(insights_a, prefix="  ")
        generated += f"\nSources for {b}:\n" + _sources_block(insights_b, prefix="  ")

    return generated


def _sources_block(insights: list[dict], prefix: str = "") -> str:
    lines = []
    for r in insights:
        layer = f" | {r['retrieval_layer']}" if r.get("retrieval_layer") == "archetype" else ""
        lines.append(
            f"{prefix}[{r['score']:.2f} | conf {r['confidence']:.2f}{layer}]"
            f" ({r['insight_type']}) {r['text'][:100]}{'...' if len(r['text']) > 100 else ''}"
        )
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Query the LoL coaching knowledge base")
    parser.add_argument("question", nargs="+", help="Your question")
    parser.add_argument("--role", help="Filter by role (mid/top/jungle/adc/support)")
    parser.add_argument("--champion", help="Force a specific champion filter (skips intent detection)")
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
        intent = detect_intent(question)
        print(f"Intent: {intent}")
        return

    if args.role:
        print(f"Filter: role={args.role}")
    if args.champion:
        print(f"Filter: champion={args.champion}")
    if args.insight_type:
        print(f"Filter: type={args.insight_type}")
    print()

    if args.retrieve_only:
        results = retrieve(question, role=args.role, champion=args.champion,
                           insight_type=args.insight_type, top_k=args.top_k)
        for r in results:
            layer = f" | {r['retrieval_layer']}" if r.get("retrieval_layer") == "archetype" else ""
            print(f"[{r['score']:.3f} | conf {r['confidence']:.2f}{layer}] ({r['insight_type']}) {r['text']}")
        return

    result = answer(question, role=args.role, champion=args.champion,
                    insight_type=args.insight_type, top_k=args.top_k)
    print(result)


if __name__ == "__main__":
    main()
