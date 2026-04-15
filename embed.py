"""
PHASE 4: Embed insights into vectors using sentence-transformers.

PURPOSE:
    Converts each insight string from the DB into a 384-dimensional numerical
    vector that captures its semantic meaning. Similar insights end up with
    similar vectors regardless of exact wording — this is what makes search work.

HOW IT WORKS:
    1. Load all insights with status='analyzed' from the DB
    2. Run each insight string through sentence-transformers (all-MiniLM-L6-v2)
       — a small BERT model that runs locally on your GPU
    3. Store the resulting vector as a binary blob back in the DB alongside the insight

MODEL CHOICE:
    all-MiniLM-L6-v2 — 22MB, fast, good quality for semantic search
    Can swap to all-mpnet-base-v2 for higher quality at 2x the size/time

STORAGE:
    Vectors stored as numpy binary blobs in the insights table.
    At ~50K insights × 384 dims × 4 bytes = ~75MB total — fits in memory easily.
    No separate vector DB needed at this scale.

DEPENDENCIES TO ADD:
    sentence-transformers>=3.0.0
"""

# TODO: import sentence_transformers
# TODO: import numpy as np
# from database import get_connection

# MODEL_NAME = "all-MiniLM-L6-v2"


def embed_insights() -> None:
    """
    Load all un-embedded insights from DB, generate vectors, store back.
    Safe to re-run — skips insights that already have a vector.
    """
    # TODO:
    # 1. Add `embedding BLOB` column to insights table if not exists
    # 2. Load SentenceTransformer(MODEL_NAME) — will use GPU automatically if available
    # 3. Fetch all insights WHERE embedding IS NULL
    # 4. Batch encode: model.encode(texts, batch_size=64, show_progress_bar=True)
    # 5. For each insight, serialize vector: embedding.astype(np.float32).tobytes()
    # 6. UPDATE insights SET embedding = ? WHERE id = ?
    pass


def load_all_vectors():
    """
    Load all insight IDs, text, metadata, and vectors from DB into memory.
    Returns arrays ready for cosine similarity search.
    """
    # TODO:
    # 1. SELECT id, video_id, insight_type, text, embedding FROM insights
    # 2. Deserialize blobs: np.frombuffer(row['embedding'], dtype=np.float32)
    # 3. Stack into a matrix: shape (n_insights, 384)
    # 4. Return (ids, texts, metadata, matrix)
    pass


if __name__ == "__main__":
    embed_insights()
