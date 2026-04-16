"""
PHASE 6: Compute cross-video cluster scores and final confidence for all insights.

Run this once after embed.py has processed all videos:
    python score_clusters.py

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
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from database import get_all_insights_with_embeddings, update_cluster_scores

# Cosine similarity threshold to count an insight as a "cluster neighbour"
SIMILARITY_THRESHOLD = 0.75

# Number of cross-video neighbours needed to reach full cluster confidence
# e.g. 5 similar insights in other videos → cluster_score = 1.0
MAX_NEIGHBOURS = 5


def compute_cluster_scores() -> None:
    print("Loading embedded insights…")
    rows = get_all_insights_with_embeddings()

    if not rows:
        print("No embedded insights found. Run embed.py first.")
        return

    print(f"  {len(rows)} insights loaded")

    ids        = [r["id"]        for r in rows]
    video_ids  = [r["video_id"]  for r in rows]
    src_scores = [r["source_score"] for r in rows]

    # Reconstruct the embedding matrix
    vectors = np.array(
        [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    )  # shape: (N, 384)

    print("Computing pairwise cosine similarities…")
    # Pre-normalised vectors → dot product = cosine similarity
    # Compute the full N×N similarity matrix in one shot
    sim_matrix = vectors @ vectors.T  # (N, N)

    print("Scoring clusters…")
    updates = []
    for i in range(len(ids)):
        # Count neighbours in OTHER videos above the threshold
        neighbours = sum(
            1
            for j, sim in enumerate(sim_matrix[i])
            if j != i
            and video_ids[j] != video_ids[i]
            and sim >= SIMILARITY_THRESHOLD
        )

        cluster_score = min(neighbours / MAX_NEIGHBOURS, 1.0)

        # source_score may be None for insights analyzed before this feature
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
    print("\nDone. Re-run after adding new videos and re-running embed.py.")


if __name__ == "__main__":
    compute_cluster_scores()
