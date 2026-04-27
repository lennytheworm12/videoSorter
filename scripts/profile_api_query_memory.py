"""Profile full hosted API query memory usage."""

from __future__ import annotations

import argparse
import os

from starlette.requests import Request

from core.memory_debug import rss_mb


def _fake_request() -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "headers": [],
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile full API query memory usage")
    parser.add_argument("--question", default="how do i beat a ranged team as illaoi")
    parser.add_argument("--game", default="lol")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vector-backend", choices=["sqlite", "supabase"], default=None)
    parser.add_argument("--embedding-backend", choices=["local", "hf_remote", "bm25_only"], default=None)
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument("--split-detail", action="store_true")
    args = parser.parse_args()

    if args.vector_backend:
        os.environ["VECTOR_BACKEND"] = args.vector_backend
    if args.embedding_backend:
        os.environ["EMBEDDING_BACKEND"] = args.embedding_backend
    os.environ.setdefault("DEBUG_MEMORY", "true")

    print(f"[memory] rss before api import: {rss_mb():.1f}MB")
    from api.main import QueryRequest, query

    print(f"[memory] rss after api import: {rss_mb():.1f}MB")
    req = QueryRequest(
        question=args.question,
        game=args.game,
        top_k=args.top_k,
        show_sources=args.show_sources,
        split_detail=args.split_detail,
    )
    response = query(req, _fake_request(), {"id": "memory-profiler"})
    print(f"[memory] rss after api query: {rss_mb():.1f}MB")
    print(f"normalized_question={response.normalized_question}")
    print(f"retrieval_mode={response.metadata.get('retrieval_mode')}")
    print(f"answer_chars={len(response.answer)}")
    print(f"sources={len(response.sources)}")


if __name__ == "__main__":
    main()
