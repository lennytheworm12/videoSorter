"""
PHASE 4.5: Within-video insight deduplication and repetition scoring.

Run after embed.py and before score_clusters.py:
    python -m pipeline.consolidate

What it does:
  Within each video, finds clusters of semantically similar insights
  (cosine similarity >= threshold). For each cluster, keeps the highest
  source_score insight as the representative and marks the rest as duplicates.
  Stores repetition_count on the representative — how many times this point
  was made across chunks of the same video.

  repetition_count feeds within_video_weight in the confidence formula:
      within_video_weight = log(1 + repetition_count) / log(MAX_REPS + 1)

  A coach who repeats a point 3 times in one session is stressing it.
  That signal should rank above a point mentioned once.
"""

import math
import logging
import numpy as np
import pathlib
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import core.database as _db
from core.database import get_connection
from core.db_paths import all_content_db_paths

# Cosine similarity threshold to treat two insights as the same point
DEDUP_THRESHOLD = 0.88

# repetition_count at which within_video_weight saturates to 1.0
MAX_REPS = 4

_ALL_DBS: list[str] | None = None


def within_video_weight(repetition_count: int) -> float:
    return math.log(1 + repetition_count) / math.log(MAX_REPS + 1)


def _add_columns() -> None:
    with get_connection() as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(insights)").fetchall()]
        if "repetition_count" not in cols:
            conn.execute("ALTER TABLE insights ADD COLUMN repetition_count INTEGER DEFAULT 1")
        if "is_duplicate" not in cols:
            conn.execute("ALTER TABLE insights ADD COLUMN is_duplicate INTEGER DEFAULT 0")
        conn.commit()


def _db_paths() -> list[str]:
    return list(_ALL_DBS) if _ALL_DBS is not None else all_content_db_paths()


def consolidate() -> None:
    _add_columns()

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT i.id, i.video_id, i.text, i.source_score, i.embedding
            FROM insights i
            WHERE i.embedding IS NOT NULL AND i.is_duplicate = 0
            ORDER BY i.video_id
        """).fetchall()

    if not rows:
        print("No embedded insights found. Run embed.py first.")
        return

    print(f"Loaded {len(rows)} insights across all videos")

    # Group by video
    by_video: dict[str, list] = {}
    for r in rows:
        by_video.setdefault(r["video_id"], []).append(r)

    total_dupes = 0
    total_merged = 0
    updates_rep: list[tuple[int, int, int]] = []   # (repetition_count, id)
    updates_dupe: list[tuple[int]] = []              # (id,) to mark as duplicate

    for video_id, insights in by_video.items():
        if len(insights) < 2:
            continue

        vecs = np.array([
            np.frombuffer(r["embedding"], dtype=np.float32) for r in insights
        ])
        sim = vecs @ vecs.T  # (N, N) cosine similarity

        visited = set()
        for i, row_i in enumerate(insights):
            if i in visited:
                continue
            # Find all insights in this video similar enough to be the same point
            cluster = [
                j for j in range(len(insights))
                if j != i and j not in visited and sim[i, j] >= DEDUP_THRESHOLD
            ]
            if not cluster:
                continue

            # All members including i form a cluster
            all_members = [i] + cluster
            visited.update(all_members)

            # Pick the representative: highest source_score
            rep_idx = max(all_members, key=lambda k: insights[k]["source_score"] or 0)
            rep_id = insights[rep_idx]["id"]
            rep_count = len(all_members)

            updates_rep.append((rep_count, rep_id))
            for j in all_members:
                if j != rep_idx:
                    updates_dupe.append((insights[j]["id"],))

            total_merged += 1
            total_dupes += len(cluster)

    if updates_rep or updates_dupe:
        with get_connection() as conn:
            if updates_rep:
                conn.executemany(
                    "UPDATE insights SET repetition_count = ? WHERE id = ?",
                    updates_rep,
                )
            if updates_dupe:
                conn.executemany(
                    "UPDATE insights SET is_duplicate = 1 WHERE id = ?",
                    updates_dupe,
                )
            conn.commit()

    # Summary
    print(f"\nConsolidation complete:")
    print(f"  Videos processed : {len(by_video)}")
    print(f"  Clusters found   : {total_merged}")
    print(f"  Duplicates marked: {total_dupes}")
    print(f"  Insights kept    : {len(rows) - total_dupes}")

    # Distribution of repetition counts
    if updates_rep:
        counts = [r[0] for r in updates_rep]
        print(f"\nRepetition count distribution (stressed points):")
        for n in sorted(set(counts)):
            w = within_video_weight(n)
            print(f"  repeated {n}x : {counts.count(n):4d} clusters  (weight={w:.2f})")


def consolidate_all_databases() -> None:
    for db_path in _db_paths():
        if not pathlib.Path(db_path).exists():
            continue
        _db.DB_PATH = pathlib.Path(db_path)
        print(f"\n--- {db_path} ---")
        consolidate()


if __name__ == "__main__":
    consolidate_all_databases()
