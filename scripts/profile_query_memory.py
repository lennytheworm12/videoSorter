"""Profile retrieval memory for different embedding backends."""

from __future__ import annotations

import argparse
import os

from core.memory_debug import rss_mb


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile retrieval memory usage")
    parser.add_argument("--question", default="how do i beat a ranged team as illaoi")
    parser.add_argument("--game", default="lol")
    parser.add_argument("--role")
    parser.add_argument("--champion")
    parser.add_argument("--subject")
    parser.add_argument("--type", dest="insight_type")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vector-backend", choices=["sqlite", "supabase"], default=None)
    parser.add_argument("--embedding-backend", choices=["local", "hf_remote", "bm25_only"], default=None)
    args = parser.parse_args()

    if args.vector_backend:
        os.environ["VECTOR_BACKEND"] = args.vector_backend
    if args.embedding_backend:
        os.environ["EMBEDDING_BACKEND"] = args.embedding_backend
    os.environ.setdefault("DEBUG_MEMORY", "true")

    print(f"[memory] rss before retrieval import: {rss_mb():.1f}MB")
    from retrieval.query import current_retrieval_mode, retrieve

    print(f"[memory] rss after retrieval import: {rss_mb():.1f}MB")
    results = retrieve(
        args.question,
        role=args.role,
        champion=args.champion,
        subject=args.subject,
        insight_type=args.insight_type,
        game=args.game,
        top_k=args.top_k,
    )
    print(f"[memory] rss after retrieve: {rss_mb():.1f}MB")
    print(f"retrieval_mode={current_retrieval_mode()}")
    print(f"results={len(results)}")


if __name__ == "__main__":
    main()
