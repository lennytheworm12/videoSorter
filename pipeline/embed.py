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
import core.embedded_vectors as embedded_vectors
from core.db_paths import all_content_db_paths

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


_ALL_DBS: list[str] | None = None


def _db_paths() -> list[str]:
    return list(_ALL_DBS) if _ALL_DBS is not None else all_content_db_paths()


def load_all_vectors(
    role: str | None = None,
    champion: str | None = None,
    insight_type: str | None = None,
    game: str | None = None,
    subject: str | None = None,
) -> tuple[list, list[str], list[dict], np.ndarray]:
    embedded_vectors._ALL_DBS = list(_ALL_DBS) if _ALL_DBS is not None else None
    return embedded_vectors.load_all_vectors(
        role=role,
        champion=champion,
        insight_type=insight_type,
        game=game,
        subject=subject,
    )


if __name__ == "__main__":
    import pathlib
    import core.database as _db
    for _path in _db_paths():
        _db.DB_PATH = pathlib.Path(_path)
        print(f"\n--- {_path} ---")
        init_db()
        embed_insights()
