"""
PHASE 6: Compute cross-video cluster scores and final confidence for all insights.

Run this once after consolidate.py has processed all videos:
    python -m pipeline.score_clusters

What it does:
  For each embedded insight, count how many OTHER insights (from different videos)
  have cosine similarity above a threshold. Insights that recur across videos are
  real coach advice. Insights that appear only once are either unique tips or
  hallucinations — the source_score disambiguates between them.

  confidence = 0.6 * source_score + 0.4 * cluster_score

  At query time, confidence is used to weight retrieval so hallucinated outliers
  sink below well-grounded, recurring insights.
"""

import numpy as np
import logging
import pathlib
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import core.database as _db
from core.database import get_connection, init_db, update_cluster_scores
from core.db_paths import all_content_db_paths

# Cosine similarity threshold to count an insight as a "cluster neighbour"
# 0.70 catches paraphrased versions of the same advice across videos
SIMILARITY_THRESHOLD = 0.70

# Number of cross-video neighbours needed to reach full cluster confidence
# 3 is appropriate for partial datasets; raise to 5 once all roles are processed
MAX_NEIGHBOURS = 3

_ALL_DBS: list[str] | None = None


def _db_paths() -> list[str]:
    return list(_ALL_DBS) if _ALL_DBS is not None else all_content_db_paths()


def _scorable_rows() -> list:
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT i.id, i.video_id, i.insight_type, i.text, i.subject, i.subject_type,
                   i.embedding, i.source_score, v.game
            FROM insights i
            JOIN videos v ON v.video_id = i.video_id
            WHERE embedding IS NOT NULL
              AND COALESCE(i.is_duplicate, 0) = 0
            """
        ).fetchall()


def compute_cluster_scores() -> None:
    print("Loading embedded insights…")
    rows = _scorable_rows()

    if not rows:
        print("No scorable embedded insights found. Run embed.py and consolidate.py first.")
        return

    updates = []
    by_game: dict[str, list] = {}
    for row in rows:
        by_game.setdefault(row["game"] or "unknown", []).append(row)

    print(f"  {len(rows)} insights loaded across {len(by_game)} game bucket(s)")
    print("Computing pairwise cosine similarities…")
    print("Scoring clusters…")

    for game, game_rows in by_game.items():
        print(f"  - {game}: {len(game_rows)} insights")
        ids = [r["id"] for r in game_rows]
        video_ids = [r["video_id"] for r in game_rows]
        src_scores = [r["source_score"] for r in game_rows]
        vectors = np.array(
            [np.frombuffer(r["embedding"], dtype=np.float32) for r in game_rows]
        )
        sim_matrix = vectors @ vectors.T

        for i in range(len(ids)):
            neighbours = sum(
                1
                for j, sim in enumerate(sim_matrix[i])
                if j != i
                and video_ids[j] != video_ids[i]
                and sim >= SIMILARITY_THRESHOLD
            )

            cluster_score = min(neighbours / MAX_NEIGHBOURS, 1.0)
            src = src_scores[i] if src_scores[i] is not None else 0.5
            confidence = round(0.6 * src + 0.4 * cluster_score, 4)
            updates.append((round(cluster_score, 4), confidence, ids[i]))

    print(f"Updating {len(updates)} rows…")
    update_cluster_scores(updates)

    # Print a summary
    confidences = [u[1] for u in updates]
    low    = sum(1 for c in confidences if c < 0.30)
    medium = sum(1 for c in confidences if 0.30 <= c < 0.60)
    high   = sum(1 for c in confidences if c >= 0.60)

    print(f"\nConfidence distribution:")
    print(f"  High  (≥0.60): {high:4d}  — well-grounded, recurring advice")
    print(f"  Medium (0.30-0.60): {medium:4d}  — moderate confidence")
    print(f"  Low   (<0.30): {low:4d}  — likely hallucinated or isolated")
    print("\nDone. Re-run after adding new videos and re-running embed.py + consolidate.py.")


def score_all_databases() -> None:
    for db_path in _db_paths():
        if not pathlib.Path(db_path).exists():
            continue
        _db.DB_PATH = pathlib.Path(db_path)
        print(f"\n--- {db_path} ---")
        init_db()
        compute_cluster_scores()


if __name__ == "__main__":
    score_all_databases()
