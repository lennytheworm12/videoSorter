"""Hosted vector search for Supabase pgvector."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any


def enabled() -> bool:
    return os.environ.get("VECTOR_BACKEND", "sqlite").strip().lower() == "supabase"


def _database_url() -> str:
    url = os.environ.get("SUPABASE_DATABASE_URL") or os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError("SUPABASE_DATABASE_URL is required when VECTOR_BACKEND=supabase")
    return url


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


def search_insights(
    query_vector: Sequence[float],
    *,
    game: str,
    role: str | None = None,
    champion: str | None = None,
    subject: str | None = None,
    insight_type: str | None = None,
    top_k: int = 35,
) -> list[dict[str, Any]]:
    """Return direct insight hits from Supabase's match_insights RPC."""
    import psycopg
    from psycopg.rows import dict_row

    general_aoe2_only = game == "aoe2" and subject is None
    with psycopg.connect(_database_url(), row_factory=dict_row) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM public.match_insights(
                %s::vector,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
            """,
            (
                _vector_literal(query_vector),
                top_k,
                game,
                role,
                champion,
                subject,
                insight_type,
                general_aoe2_only,
            ),
        ).fetchall()
    return [dict(row) for row in rows]
