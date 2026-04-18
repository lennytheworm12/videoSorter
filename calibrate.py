"""
Fit confidence weights from eval_ratings using logistic regression.

Uses rated query answers to learn w1 (source_score) and w2 (cluster_score)
weights that predict answer quality, replacing the hardcoded 0.6/0.4 split.

Usage:
    uv run python calibrate.py          # fit and print calibrated weights
    uv run python calibrate.py --apply  # fit and write weights to config

Requires 50+ ratings in eval_ratings for stable results.
"""

import json
import argparse
from core.database import get_connection, init_db


def load_training_data() -> tuple[list, list]:
    """
    For each rated query, expand into one row per retrieved insight.
    X = [source_score, cluster_score]
    y = 1 if answer_quality >= 4, else 0
    """
    with get_connection() as conn:
        ratings = conn.execute("""
            SELECT answer_good, retrieved_insight_ids
            FROM eval_ratings
            WHERE answer_good IS NOT NULL
              AND retrieved_insight_ids IS NOT NULL
        """).fetchall()

    X, y = [], []
    for row in ratings:
        label = row["answer_good"]
        try:
            ids = json.loads(row["retrieved_insight_ids"])
        except Exception:
            continue
        if not ids:
            continue

        with get_connection() as conn:
            insights = conn.execute(
                f"SELECT source_score, cluster_score FROM insights WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            ).fetchall()

        for ins in insights:
            src = ins["source_score"] or 0.5
            cls = ins["cluster_score"] or 0.5
            X.append([src, cls])
            y.append(label)

    return X, y


def calibrate() -> dict:
    try:
        from sklearn.linear_model import LogisticRegression
        import numpy as np
    except ImportError:
        print("scikit-learn not installed. Run: uv add scikit-learn")
        return {}

    X, y = load_training_data()

    if len(X) < 50:
        print(f"Only {len(X)} data points — need 50+ for stable calibration. Keep rating!")
        return {}

    X_arr = np.array(X)
    y_arr = np.array(y)

    model = LogisticRegression(max_iter=500)
    model.fit(X_arr, y_arr)

    coefs = model.coef_[0]
    total = sum(abs(c) for c in coefs)
    w1 = round(abs(coefs[0]) / total, 3)  # source_score weight
    w2 = round(abs(coefs[1]) / total, 3)  # cluster_score weight

    pos_rate = sum(y_arr) / len(y_arr)
    train_acc = model.score(X_arr, y_arr)

    print(f"\nCalibration results ({len(X)} data points):")
    print(f"  w1 (source_score)  = {w1:.3f}  (was 0.600)")
    print(f"  w2 (cluster_score) = {w2:.3f}  (was 0.400)")
    print(f"  Good answer rate   = {pos_rate:.1%}")
    print(f"  Training accuracy  = {train_acc:.1%}")

    return {"w1": w1, "w2": w2}


def apply_weights(weights: dict) -> None:
    """Write calibrated weights to weights.json for use by score_clusters.py."""
    import pathlib
    path = pathlib.Path("weights.json")
    path.write_text(json.dumps(weights, indent=2))
    print(f"\nWeights written to {path}")
    print("Re-run: uv run python -m pipeline.score_clusters to apply them.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate confidence weights from ratings")
    parser.add_argument("--apply", action="store_true", help="Write calibrated weights to weights.json")
    args = parser.parse_args()

    init_db()
    weights = calibrate()

    if weights and args.apply:
        apply_weights(weights)


if __name__ == "__main__":
    main()
