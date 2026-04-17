"""
PHASE 5: Query the knowledge base using RRF (semantic + BM25) + RAG.

Retrieval uses Reciprocal Rank Fusion over two ranked lists:
  1. Semantic search  — cosine similarity on sentence-transformer embeddings
  2. BM25 keyword     — exact term overlap, good for champion/item names

RRF formula: score(d) = Σ 1 / (k + rank_i(d))  where k=60

Usage:
    python -m retrieval.query "how do I play Cassiopeia against poke mages?"
    python -m retrieval.query "what does the coach say about wave management?" --role mid
    python -m retrieval.query "when should I take teleport vs ignite?" --type principles
"""

import re
import sys
import math
import logging
import argparse
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from pipeline.embed import load_all_vectors
from core.llm import chat as llm_chat

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 12
RRF_K = 60  # standard RRF smoothing constant

RAG_SYSTEM_PROMPT = """
You are a League of Legends coaching assistant. You have access to insights
extracted from a real coach's video library. Your job is to answer the player's
question using ONLY the provided coaching insights — do not add advice that
isn't grounded in the retrieved insights.

ANSWERING MATCHUP QUESTIONS (when the question asks how to play X into Y or X vs Y):
Structure your answer in two phases:

**Laning Phase**
- Champion identity and win condition in this matchup
- Specific trading patterns, wave management, and kill setup tips
- What to avoid (e.g. ability timing, positioning mistakes)

**Post-Lane / Teamfights**
- When and how to roam or transition
- Teamfight role, engage/disengage decisions
- Win condition execution in mid/late game

ANSWERING GENERAL QUESTIONS:
- Lead with the core principle or mental model if one is present
- Follow with specific actionable tips
- If insights contradict each other, acknowledge both perspectives

Always:
- Be concise — players want clear, direct coaching advice
- Only use sections that have relevant insights — skip a section rather than pad it
- If the retrieved insights only cover one phase, say so rather than inventing the other
""".strip()

RAG_USER_PROMPT = """
Player question: {question}

Relevant coaching insights retrieved from the video library:
{insights}

Answer the question using only the insights above. If this is a matchup question,
split into Laning Phase and Post-Lane sections. Only include a section if the
insights actually support it — do not invent advice not present above.
""".strip()


# ── BM25 ──────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop empty tokens."""
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _bm25_scores(query_tokens: list[str], corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """
    Compute BM25 scores for all documents in corpus against query_tokens.
    Returns a float array of shape (len(corpus),).
    """
    n = len(corpus)
    if n == 0:
        return np.array([], dtype=np.float32)

    doc_lens = np.array([len(d) for d in corpus], dtype=np.float32)
    avgdl = doc_lens.mean() if doc_lens.mean() > 0 else 1.0

    # Document frequency per query term
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
    """
    Merge multiple ranked lists (each a list of original indices, best first)
    using Reciprocal Rank Fusion. Returns indices sorted by fused score desc.
    """
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused, key=lambda i: fused[i], reverse=True)


# ── retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    question: str,
    role: str | None = None,
    champion: str | None = None,
    insight_type: str | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Embed the question and return the top_k most relevant insights using RRF
    over semantic (cosine) and keyword (BM25) search.

    Optional filters (role, champion, insight_type) narrow the pool first.

    Returns a list of dicts:
        text         — the insight string
        insight_type — category (principles, laning_tips, etc.)
        role         — which role channel it came from
        champion     — champion the video was about (if known)
        score        — cosine similarity (for reference)
        confidence   — confidence weight from score_clusters
    """
    ids, texts, metadata, matrix = load_all_vectors(
        role=role,
        champion=champion,
        insight_type=insight_type,
    )

    if not ids:
        return []

    n = len(ids)
    fetch_k = min(n, max(top_k * 3, 50))  # fetch more than needed before fusing

    # ── 1. Semantic search ────────────────────────────────────────────────────
    model = SentenceTransformer(MODEL_NAME)
    query_vec = model.encode(question, convert_to_numpy=True, normalize_embeddings=True)
    cosine_scores = matrix @ query_vec
    semantic_ranked = np.argsort(cosine_scores)[::-1][:fetch_k].tolist()

    # ── 2. BM25 keyword search ────────────────────────────────────────────────
    corpus_tokens = [_tokenize(t) for t in texts]
    query_tokens = _tokenize(question)
    bm25 = _bm25_scores(query_tokens, corpus_tokens)
    bm25_ranked = np.argsort(bm25)[::-1][:fetch_k].tolist()

    # ── 3. RRF fusion ─────────────────────────────────────────────────────────
    fused_indices = _rrf_fuse([semantic_ranked, bm25_ranked])[:top_k]

    # ── 4. Apply confidence reranking ─────────────────────────────────────────
    # Combined weight: cluster confidence + source grounding score
    # source_score is a soft signal — low values down-weight but don't exclude
    confidences = np.array([m.get("confidence") or 0.5 for m in metadata])
    source_scores = np.array([m.get("source_score") or 0.5 for m in metadata])
    combined = (0.6 * confidences + 0.4 * source_scores)
    fused_indices.sort(key=lambda i: (matrix[i] @ query_vec) * (0.5 + 0.5 * float(combined[i])), reverse=True)

    results = []
    for i in fused_indices[:top_k]:
        results.append({
            "text": texts[i],
            "insight_type": metadata[i]["insight_type"],
            "role": metadata[i]["role"],
            "champion": metadata[i]["champion"],
            "score": round(float(cosine_scores[i]), 4),
            "confidence": round(float(confidences[i]), 4),
        })

    return results


def answer(
    question: str,
    role: str | None = None,
    champion: str | None = None,
    insight_type: str | None = None,
    top_k: int = TOP_K,
    show_sources: bool = True,
) -> str:
    """
    Full RAG pipeline:
      1. Retrieve top_k insights via RRF (semantic + BM25)
      2. Feed them as context to the LLM
      3. Return a structured answer grounded in the coach's actual advice
    """
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

    prompt = RAG_USER_PROMPT.format(question=question, insights=formatted)
    generated = llm_chat(system=RAG_SYSTEM_PROMPT, user=prompt, temperature=0.2)

    if show_sources:
        sources = "\n\n---\nSources retrieved:\n" + "\n".join(
            f"  [{r['score']:.2f} | conf {r['confidence']:.2f}] ({r['insight_type']}) {r['text'][:100]}{'...' if len(r['text']) > 100 else ''}"
            for r in insights
        )
        return generated + sources

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the LoL coaching knowledge base")
    parser.add_argument("question", nargs="+", help="Your question")
    parser.add_argument("--role", help="Filter by role (mid/top/jungle/adc/support)")
    parser.add_argument("--champion", help="Filter by champion name")
    parser.add_argument("--type", dest="insight_type",
                        help="Filter by insight type (principles/laning_tips/macro_advice/etc.)")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of insights to retrieve")
    parser.add_argument("--retrieve-only", action="store_true",
                        help="Show retrieved insights without generating an answer")
    args = parser.parse_args()

    question = " ".join(args.question)
    print(f"\nQuestion: {question}")
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
        for i, r in enumerate(results):
            print(f"[{r['score']:.3f} | conf {r['confidence']:.2f}] ({r['insight_type']}) {r['text']}")
        return

    result = answer(question, role=args.role, champion=args.champion,
                    insight_type=args.insight_type, top_k=args.top_k)
    print(result)


if __name__ == "__main__":
    main()
