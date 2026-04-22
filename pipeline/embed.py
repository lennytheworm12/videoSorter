"""
PHASE 4: Embed insights into vectors using sentence-transformers.

Converts each insight string into a 384-dimensional vector that captures
semantic meaning. Similar insights produce similar vectors regardless of
exact wording — this is what makes semantic search work in query.py.

Run after analyze.py:
    python embed.py

Safe to re-run — skips insights that already have a vector stored.
"""

import logging
import pathlib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from core.database import get_connection, init_db

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_NAME = "all-MiniLM-L6-v2"  # 22MB, fast, great for semantic search


def _add_embedding_column() -> None:
    """Add embedding column to insights table if it doesn't exist yet."""
    with get_connection() as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(insights)").fetchall()]
        if "embedding" not in cols:
            conn.execute("ALTER TABLE insights ADD COLUMN embedding BLOB")
            conn.commit()


def embed_insights() -> None:
    _add_embedding_column()

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, text FROM insights WHERE embedding IS NULL"
        ).fetchall()

    if not rows:
        print("No insights to embed — all already have vectors.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {MODEL_NAME} (device={device})")
    model = SentenceTransformer(MODEL_NAME, device=device)

    ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]

    print(f"Embedding {len(texts)} insights…")
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # pre-normalise so dot product = cosine similarity
    )

    with get_connection() as conn:
        for insight_id, vector in zip(ids, vectors):
            blob = vector.astype(np.float32).tobytes()
            conn.execute(
                "UPDATE insights SET embedding = ? WHERE id = ?",
                (blob, insight_id),
            )
        conn.commit()

    print(f"Done — {len(ids)} insights embedded.")


_ALL_DBS = ["videos.db", "guide_test.db"]


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

    query = """
        SELECT i.id, i.text, i.insight_type, i.embedding,
               i.confidence, i.source_score, v.video_id, v.game, v.role, COALESCE(v.subject, v.champion) AS subject,
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
        query += " AND LOWER(COALESCE(v.subject, v.champion)) = LOWER(?)"
        params.append(subject)
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
    """
    Load insight IDs, texts, metadata, and vectors from all DBs.
    Merges videos.db and guide_test.db transparently.

    Returns:
        ids       — list of insight row IDs (prefixed with db path)
        texts     — list of insight strings
        metadata  — list of dicts with video_id, role, champion, insight_type
        matrix    — numpy array shape (n, 384), pre-normalised
    """
    all_ids, all_texts, all_meta, all_vecs = [], [], [], []

    for db_path in _ALL_DBS:
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


if __name__ == "__main__":
    import pathlib
    import core.database as _db
    for _path in ["videos.db", "guide_test.db"]:
        _db.DB_PATH = pathlib.Path(_path)
        print(f"\n--- {_path} ---")
        init_db()
        embed_insights()
