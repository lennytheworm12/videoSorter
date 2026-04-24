"""Hosted vector search for Supabase pgvector."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from core.env import load_project_env

load_project_env()


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


def search_keyword_candidates(
    query_text: str,
    *,
    game: str,
    role: str | None = None,
    champion: str | None = None,
    subject: str | None = None,
    insight_type: str | None = None,
    top_k: int = 120,
) -> list[dict[str, Any]]:
    """Return lexical candidates from Supabase without requiring query embeddings."""
    import psycopg
    from psycopg.rows import dict_row

    general_aoe2_only = game == "aoe2" and subject is None
    params: list[Any] = [query_text]
    filters = ["v.game = %s"]
    params.append(game)

    if role:
        filters.append("v.role = %s")
        params.append(role)
    if champion:
        filters.append("LOWER(v.champion) = LOWER(%s)")
        params.append(champion)
    if subject:
        filters.append(
            """
            LOWER(
                CASE
                    WHEN i.subject_type IS NOT NULL THEN i.subject
                    ELSE COALESCE(v.subject, v.champion)
                END
            ) = LOWER(%s)
            """
        )
        params.append(subject)
    elif general_aoe2_only:
        filters.append("COALESCE(i.subject_type, 'general') = 'general'")
    if insight_type:
        filters.append("i.insight_type = %s")
        params.append(insight_type)

    params.append(query_text)
    params.append(top_k)

    with psycopg.connect(_database_url(), row_factory=dict_row) as conn:
        rows = conn.execute(
            f"""
            SELECT
                i.source_db,
                i.local_id,
                i.video_id,
                i.text,
                i.insight_type,
                v.role,
                CASE
                    WHEN i.subject_type IS NOT NULL THEN i.subject
                    ELSE COALESCE(v.subject, v.champion)
                END AS subject,
                i.subject AS insight_subject,
                i.subject_type,
                v.champion,
                v.game,
                v.rank,
                v.website_rating,
                COALESCE(v.source, 'discord') AS source,
                i.confidence,
                i.source_score,
                ts_rank_cd(
                    to_tsvector('english', COALESCE(i.text, '')),
                    websearch_to_tsquery('english', %s)
                ) AS lexical_score
            FROM public.insights i
            JOIN public.videos v
              ON v.source_db = i.source_db
             AND v.video_id = i.video_id
            WHERE {' AND '.join(filters)}
              AND to_tsvector('english', COALESCE(i.text, '')) @@ websearch_to_tsquery('english', %s)
            ORDER BY lexical_score DESC, COALESCE(i.confidence, i.source_score, 0.5) DESC
            LIMIT %s
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]
