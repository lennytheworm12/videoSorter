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


def load_all_vectors(
    role: str | None = None,
    champion: str | None = None,
    insight_type: str | None = None,
) -> tuple[list[int], list[str], list[dict], np.ndarray]:
    """
    Load insight IDs, texts, metadata, and vectors from DB.
    Optional filters narrow the pool before returning.

    Returns:
        ids       — list of insight row IDs
        texts     — list of insight strings
        metadata  — list of dicts with video_id, role, champion, insight_type
        matrix    — numpy array shape (n, 384), pre-normalised
    """
    query = """
        SELECT i.id, i.text, i.insight_type, i.embedding,
               i.confidence, v.video_id, v.role, v.champion
        FROM insights i
        JOIN videos v ON i.video_id = v.video_id
        WHERE i.embedding IS NOT NULL
    """
    params = []
    if role:
        query += " AND v.role = ?"
        params.append(role)
    if champion:
        query += " AND LOWER(v.champion) = LOWER(?)"
        params.append(champion)
    if insight_type:
        query += " AND i.insight_type = ?"
        params.append(insight_type)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        return [], [], [], np.empty((0, 384), dtype=np.float32)

    ids, texts, metadata, vectors = [], [], [], []
    for row in rows:
        ids.append(row["id"])
        texts.append(row["text"])
        metadata.append({
            "video_id": row["video_id"],
            "role": row["role"],
            "champion": row["champion"],
            "insight_type": row["insight_type"],
            "confidence": row["confidence"],
        })
        vectors.append(np.frombuffer(row["embedding"], dtype=np.float32))

    matrix = np.stack(vectors)
    return ids, texts, metadata, matrix


if __name__ == "__main__":
    init_db()
    embed_insights()
