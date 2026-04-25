"""Lightweight helpers for loading stored insight vectors from SQLite."""

from __future__ import annotations

import pathlib

import numpy as np

from core.db_paths import all_content_db_paths

_ALL_DBS: list[str] | None = None


def _db_paths() -> list[str]:
    return list(_ALL_DBS) if _ALL_DBS is not None else all_content_db_paths()


def _load_vectors_from_db(
    db_path: str,
    role: str | None,
    champion: str | None,
    insight_type: str | None,
    game: str | None = None,
    subject: str | None = None,
) -> tuple[list[int], list[str], list[dict], list[np.ndarray]]:
    import core.database as _db
    import pathlib as _pl

    _db.DB_PATH = _pl.Path(db_path)

    effective_subject_sql = """
        CASE
            WHEN i.subject_type IS NOT NULL THEN i.subject
            ELSE COALESCE(v.subject, v.champion)
        END
    """

    query = """
        SELECT i.id, i.text, i.insight_type, i.embedding,
               i.subject AS raw_subject, i.subject_type, i.confidence, i.source_score,
               v.video_id, v.game, v.role,
               """ + effective_subject_sql + """ AS subject,
               v.champion, v.rank, v.website_rating,
               COALESCE(v.source, 'discord') AS source
        FROM insights i
        JOIN videos v ON i.video_id = v.video_id
        WHERE i.embedding IS NOT NULL
    """
    params: list = []
    if game:
        query += " AND v.game = ?"
        params.append(game)
    if role:
        query += " AND v.role = ?"
        params.append(role)
    if subject:
        query += " AND LOWER(" + effective_subject_sql + ") = LOWER(?)"
        params.append(subject)
    elif game == "aoe2":
        query += " AND COALESCE(i.subject_type, 'general') = 'general'"
    if champion:
        query += " AND LOWER(v.champion) = LOWER(?)"
        params.append(champion)
    if insight_type:
        query += " AND i.insight_type = ?"
        params.append(insight_type)

    with _db.get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    ids, texts, metadata, vectors = [], [], [], []
    for row in rows:
        ids.append(f"{db_path}:{row['id']}")
        texts.append(row["text"])
        metadata.append({
            "video_id": row["video_id"],
            "game": row["game"],
            "role": row["role"],
            "subject": row["subject"],
            "insight_subject": row["raw_subject"],
            "subject_type": row["subject_type"],
            "champion": row["champion"],
            "rank": row["rank"],
            "website_rating": row["website_rating"],
            "insight_type": row["insight_type"],
            "confidence": row["confidence"],
            "source_score": row["source_score"],
            "source": row["source"],
        })
        vectors.append(np.frombuffer(row["embedding"], dtype=np.float32))
    return ids, texts, metadata, vectors


def load_all_vectors(
    role: str | None = None,
    champion: str | None = None,
    insight_type: str | None = None,
    game: str | None = None,
    subject: str | None = None,
) -> tuple[list, list[str], list[dict], np.ndarray]:
    all_ids, all_texts, all_meta, all_vecs = [], [], [], []

    for db_path in _db_paths():
        if not pathlib.Path(db_path).exists():
            continue
        ids, texts, meta, vecs = _load_vectors_from_db(
            db_path, role, champion, insight_type, game=game, subject=subject
        )
        all_ids.extend(ids)
        all_texts.extend(texts)
        all_meta.extend(meta)
        all_vecs.extend(vecs)

    if not all_vecs:
        return [], [], [], np.empty((0, 384), dtype=np.float32)

    matrix = np.stack(all_vecs)
    return all_ids, all_texts, all_meta, matrix
