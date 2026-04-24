"""Shared helpers for runtime configuration stored in Supabase/Postgres."""

from __future__ import annotations

import json
import os
from typing import Any

import psycopg
from psycopg.rows import dict_row

from core.env import load_project_env

load_project_env()


def _database_url() -> str:
    url = os.environ.get("SUPABASE_DATABASE_URL") or os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError("SUPABASE_DATABASE_URL is required")
    return url


def upsert_runtime_config(key: str, value: dict[str, Any]) -> None:
    with psycopg.connect(_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.runtime_config (key, value, updated_at)
                VALUES (%s, %s::jsonb, now())
                ON CONFLICT (key)
                DO UPDATE SET value = EXCLUDED.value, updated_at = now()
                """,
                (key, json.dumps(value)),
            )
        conn.commit()


def get_runtime_config(key: str) -> dict[str, Any] | None:
    with psycopg.connect(_database_url(), row_factory=dict_row) as conn:
        row = conn.execute(
            "SELECT value FROM public.runtime_config WHERE key = %s",
            (key,),
        ).fetchone()
    if not row:
        return None
    value = row["value"]
    if isinstance(value, dict):
        return value
    return json.loads(value)
