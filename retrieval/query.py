"""
PHASE 5: Query the knowledge base using semantic search + RAG.

Embeds a natural language question, finds the most semantically similar
insights from the coach's video library, then uses Gemma 4 to synthesize
a structured answer grounded entirely in what the coach actually said.

Usage:
    python query.py "how do I play Cassiopeia against poke mages?"
    python query.py "what does the coach say about wave management?" --role mid
    python query.py "when should I take teleport vs ignite?" --type principles
"""

import sys
import logging
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from pipeline.embed import load_all_vectors
from core.llm import chat as llm_chat

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 12

RAG_SYSTEM_PROMPT = """
You are a League of Legends coaching assistant. You have access to insights
extracted from a real coach's video library. Your job is to answer the player's
question using ONLY the provided coaching insights — do not add advice that
isn't grounded in the retrieved insights.

Structure your answer clearly:
- Lead with the core principle or mental model if one is present
- Follow with specific actionable tips
- If insights contradict each other, acknowledge both perspectives
- Be concise — players want clear, direct coaching advice
""".strip()

RAG_USER_PROMPT = """
Player question: {question}

Relevant coaching insights retrieved from the video library:
{insights}

Answer the question using only the insights above. Reference the insight content
directly rather than speaking in generalities.
""".strip()


def cosine_search(
    query_vector: np.ndarray,
    matrix: np.ndarray,
    confidences: np.ndarray | None = None,
    top_k: int = TOP_K,
) -> list[int]:
    """
    Return indices of the top_k most similar rows in matrix to query_vector.
    Both query_vector and matrix rows must be pre-normalised (done in embed.py).

    If confidences is provided (per-insight 0-1 score), the final ranking uses:
        final_score = cosine_similarity * (0.5 + 0.5 * confidence)
    This down-weights likely-hallucinated insights without fully excluding them.
    """
    if matrix.shape[0] == 0:
        return []
    scores = matrix @ query_vector  # cosine similarity, shape: (n,)
    if confidences is not None:
        scores = scores * (0.5 + 0.5 * confidences)
    top_indices = np.argsort(scores)[::-1]
    return top_indices[:top_k].tolist()


def retrieve(
    question: str,
    role: str | None = None,
    champion: str | None = None,
    insight_type: str | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Embed the question and return the top_k most semantically similar insights.
    Optional filters (role, champion, insight_type) narrow the pool first.

    Returns a list of dicts:
        text         — the insight string
        insight_type — category (principles, laning_tips, etc.)
        role         — which role channel it came from
        champion     — champion the video was about (if known)
        score        — cosine similarity 0-1 (higher = more relevant)
    """
    ids, texts, metadata, matrix = load_all_vectors(
        role=role,
        champion=champion,
        insight_type=insight_type,
    )

    if not ids:
        return []

    model = SentenceTransformer(MODEL_NAME)
    query_vec = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Use confidence scores to down-weight likely hallucinations
    confidences = np.array([m.get("confidence") or 0.5 for m in metadata])
    top_indices = cosine_search(query_vec, matrix, confidences=confidences, top_k=top_k)

    results = []
    for i in top_indices:
        cosine = float(matrix[i] @ query_vec)
        conf = float(confidences[i])
        results.append({
            "text": texts[i],
            "insight_type": metadata[i]["insight_type"],
            "role": metadata[i]["role"],
            "champion": metadata[i]["champion"],
            "score": round(cosine, 4),
            "confidence": round(conf, 4),
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
      1. Retrieve top_k semantically relevant insights
      2. Feed them as context to Gemma 4
      3. Return a structured answer grounded in the coach's actual advice

    Args:
        question     — natural language question
        role         — filter by role (mid, top, jungle, adc, support)
        champion     — filter by champion name
        insight_type — filter by category (principles, laning_tips, etc.)
        top_k        — how many insights to retrieve (more = broader context)
        show_sources — append the retrieved insights as sources at the end
    """
    insights = retrieve(question, role=role, champion=champion,
                        insight_type=insight_type, top_k=top_k)

    if not insights:
        return "No relevant insights found. Make sure embed.py has been run."

    # Format retrieved insights as a numbered list for the LLM
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
            f"  [{r['score']:.2f}] ({r['insight_type']}) {r['text'][:100]}{'...' if len(r['text']) > 100 else ''}"
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
            print(f"[{r['score']:.3f}] ({r['insight_type']}) {r['text']}")
        return

    result = answer(question, role=args.role, champion=args.champion,
                    insight_type=args.insight_type, top_k=args.top_k)
    print(result)


if __name__ == "__main__":
    main()
