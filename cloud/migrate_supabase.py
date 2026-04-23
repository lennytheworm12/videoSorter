"""Sync local SQLite content DBs into Supabase Postgres/pgvector.

Required env var:
    SUPABASE_DATABASE_URL=postgresql://...

Examples:
    uv run python -m cloud.migrate_supabase --dry-run
    uv run python -m cloud.migrate_supabase --apply
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sqlite3
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from core.db_paths import all_content_db_paths
from core.env import load_project_env


VECTOR_DIMENSION = 384

load_project_env()


VIDEO_COLUMNS = (
    "source_db",
    "video_id",
    "video_url",
    "video_title",
    "description",
    "game",
    "role",
    "subject",
    "champion",
    "rank",
    "website_rating",
    "message_timestamp",
    "status",
    "transcription",
    "source",
    "created_at",
)

INSIGHT_COLUMNS = (
    "source_db",
    "local_id",
    "video_id",
    "insight_type",
    "text",
    "subject",
    "subject_type",
    "source_score",
    "cluster_score",
    "confidence",
    "repetition_count",
    "is_duplicate",
    "embedding",
    "created_at",
)


def source_db_name(path: str | pathlib.Path) -> str:
    return pathlib.Path(path).stem


def embedding_blob_to_vector(blob: bytes | memoryview | None) -> list[float] | None:
    if blob is None:
        return None
    vector = np.frombuffer(bytes(blob), dtype=np.float32)
    if vector.size != VECTOR_DIMENSION:
        raise ValueError(f"expected {VECTOR_DIMENSION}-dim embedding, got {vector.size}")
    return vector.astype(float).tolist()


def _row_value(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    return row[key] if key in row.keys() else default


def _connect_sqlite_readonly(path: str | pathlib.Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def iter_video_payloads(db_path: str | pathlib.Path) -> Iterator[dict[str, Any]]:
    source_db = source_db_name(db_path)
    with _connect_sqlite_readonly(db_path) as conn:
        for row in conn.execute("SELECT * FROM videos"):
            yield {
                "source_db": source_db,
                "video_id": row["video_id"],
                "video_url": row["video_url"],
                "video_title": _row_value(row, "video_title"),
                "description": _row_value(row, "description"),
                "game": _row_value(row, "game") or "lol",
                "role": _row_value(row, "role") or "unknown",
                "subject": _row_value(row, "subject"),
                "champion": _row_value(row, "champion"),
                "rank": _row_value(row, "rank"),
                "website_rating": _row_value(row, "website_rating"),
                "message_timestamp": _row_value(row, "message_timestamp"),
                "status": _row_value(row, "status"),
                "transcription": _row_value(row, "transcription"),
                "source": _row_value(row, "source") or "discord",
                "created_at": _row_value(row, "created_at"),
            }


def iter_insight_payloads(db_path: str | pathlib.Path) -> Iterator[dict[str, Any]]:
    source_db = source_db_name(db_path)
    with _connect_sqlite_readonly(db_path) as conn:
        for row in conn.execute("SELECT * FROM insights WHERE embedding IS NOT NULL"):
            yield {
                "source_db": source_db,
                "local_id": row["id"],
                "video_id": row["video_id"],
                "insight_type": row["insight_type"],
                "text": row["text"],
                "subject": _row_value(row, "subject"),
                "subject_type": _row_value(row, "subject_type"),
                "source_score": _row_value(row, "source_score"),
                "cluster_score": _row_value(row, "cluster_score"),
                "confidence": _row_value(row, "confidence"),
                "repetition_count": _row_value(row, "repetition_count", 1),
                "is_duplicate": bool(_row_value(row, "is_duplicate", 0)),
                "embedding": embedding_blob_to_vector(_row_value(row, "embedding")),
                "created_at": _row_value(row, "created_at"),
            }


def batched(items: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _vector_literal(values: list[float] | None) -> str | None:
    if values is None:
        return None
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def _execute_values(cur: Any, table: str, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    from psycopg import sql

    placeholders = []
    values: list[Any] = []
    for row in rows:
        row_placeholders = []
        for column in columns:
            value = row[column]
            if column == "embedding":
                value = _vector_literal(value)
                row_placeholders.append(sql.SQL("{}::vector").format(sql.Placeholder()))
            else:
                row_placeholders.append(sql.Placeholder())
            values.append(value)
        placeholders.append(sql.SQL("(") + sql.SQL(", ").join(row_placeholders) + sql.SQL(")"))

    assignments = [
        sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(column), sql.Identifier(column))
        for column in columns
        if column not in {"source_db", "video_id", "local_id"}
    ]
    conflict_key = (
        sql.SQL("(source_db, video_id)")
        if table == "videos"
        else sql.SQL("(source_db, local_id)")
    )
    query = (
        sql.SQL("INSERT INTO public.{} ({}) VALUES {} ON CONFLICT {} DO UPDATE SET {}")
        .format(
            sql.Identifier(table),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(placeholders),
            conflict_key,
            sql.SQL(", ").join(assignments),
        )
    )
    cur.execute(query, values)


def sync_to_supabase(db_paths: list[str], batch_size: int = 500, dry_run: bool = True) -> None:
    counts: dict[str, dict[str, int]] = {}
    for db_path in db_paths:
        if not pathlib.Path(db_path).exists():
            continue
        videos = sum(1 for _ in iter_video_payloads(db_path))
        insights = sum(1 for _ in iter_insight_payloads(db_path))
        counts[db_path] = {"videos": videos, "insights": insights}

    print("Local rows to sync:")
    for db_path, stat in counts.items():
        print(f"  {db_path}: {stat['videos']} videos, {stat['insights']} embedded insights")
    if dry_run:
        print("Dry run only. Re-run with --apply to write to Supabase.")
        return

    database_url = os.environ.get("SUPABASE_DATABASE_URL") or os.environ.get("SUPABASE_DB_URL")
    if not database_url:
        raise RuntimeError("SUPABASE_DATABASE_URL is required for --apply")

    import psycopg

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for db_path in db_paths:
                if not pathlib.Path(db_path).exists():
                    continue
                for batch in batched(iter_video_payloads(db_path), batch_size):
                    _execute_values(cur, "videos", VIDEO_COLUMNS, batch)
                conn.commit()
                print(f"Synced videos from {db_path}")

                for batch in batched(iter_insight_payloads(db_path), batch_size):
                    _execute_values(cur, "insights", INSIGHT_COLUMNS, batch)
                conn.commit()
                print(f"Synced embedded insights from {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync local SQLite DBs to Supabase pgvector")
    parser.add_argument("--apply", action="store_true", help="Write rows to Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Print counts only")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--db", action="append", dest="db_paths", help="SQLite DB path; repeatable")
    args = parser.parse_args()

    dry_run = not args.apply
    db_paths = args.db_paths or all_content_db_paths()
    sync_to_supabase(db_paths=db_paths, batch_size=args.batch_size, dry_run=dry_run)


if __name__ == "__main__":
    main()
