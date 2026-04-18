"""
Interactive rating harness for the LoL coaching knowledge base.

Two binary questions per answer:
  1. Good or bad?  (g/b)
  2. Did the confidence score align?  (y/n)

     good  + aligned     → best signal  (high-conf correct)
     bad   + misaligned  → worst signal (high-conf hallucination)
     bad   + aligned     → acceptable   (low-conf, at least it flagged uncertainty)

Ratings feed calibrate.py which fits w1/w2 weights for source_score and cluster_score.

Usage:
    uv run python eval.py                          # rate all unrated eval_queries
    uv run python eval.py "cassiopeia into zed"    # rate a one-off question
    uv run python eval.py --add                    # add a question to eval_queries
    uv run python eval.py --stats                  # show rating distribution
"""

import json
import argparse
from core.database import get_connection, init_db
from retrieval.query import answer, retrieve, detect_intent


# ── DB helpers ────────────────────────────────────────────────────────────────

def add_eval_query(question: str, expected_answer: str = "", notes: str = "") -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO eval_queries (question, expected_answer, notes) VALUES (?, ?, ?)",
            (question, expected_answer, notes),
        )
        conn.commit()
        return cur.lastrowid


def get_unrated_queries() -> list:
    with get_connection() as conn:
        return conn.execute("""
            SELECT eq.id, eq.question, eq.expected_answer
            FROM eval_queries eq
            LEFT JOIN eval_ratings er ON er.query_id = eq.id
            WHERE er.id IS NULL
        """).fetchall()


def save_rating(
    question: str,
    intent: dict,
    answer_good: int,
    confidence_aligned: int,
    insight_ids: list[int],
    generated_answer: str,
    query_id: int | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO eval_ratings
                (query_id, question, intent_type, champion_a, champion_b,
                 answer_good, confidence_aligned, retrieved_insight_ids, generated_answer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_id,
                question,
                intent.get("type"),
                intent.get("a"),
                intent.get("b"),
                answer_good,
                confidence_aligned,
                json.dumps(insight_ids),
                generated_answer,
            ),
        )
        conn.commit()


# ── Rating loop ────────────────────────────────────────────────────────────────

def _prompt_gb(prompt: str) -> int:
    while True:
        raw = input(prompt).strip().lower()
        if raw in ("g", "good", "y", "1"):
            return 1
        if raw in ("b", "bad", "n", "0"):
            return 0
        print("  Enter g (good) or b (bad).")


def _prompt_yn(prompt: str) -> int:
    while True:
        raw = input(prompt).strip().lower()
        if raw in ("y", "yes", "1"):
            return 1
        if raw in ("n", "no", "0"):
            return 0
        print("  Enter y or n.")


def rate_question(question: str, query_id: int | None = None, expected: str = "") -> None:
    intent = detect_intent(question)

    print(f"\n{'─' * 70}")
    print(f"Question : {question}")
    print(f"Intent   : {intent['type']}", end="")
    if intent["type"] != "general":
        print(f"  ({intent.get('a')} vs {intent.get('b')})", end="")
    print()

    if expected:
        print(f"Expected : {expected}")

    print("\nRetrieving and generating answer...\n")

    # Retrieve insights (to show sources and get IDs)
    if intent["type"] in ("matchup", "synergy"):
        from retrieval.query import retrieve_duo
        insights_a, insights_b = retrieve_duo(question, intent["a"], intent["b"])
        all_insights = insights_a + insights_b
    else:
        all_insights = retrieve(question)

    # Generate answer
    generated = answer(question, show_sources=False)

    print(generated)

    print("\n--- Retrieved insights (for confidence check) ---")
    for i, r in enumerate(all_insights):
        layer = f" | {r['retrieval_layer']}" if r.get("retrieval_layer") == "archetype" else ""
        champ = f" | {r['champion']}" if r.get("champion") else ""
        print(
            f"  {i+1}. [score={r['score']:.2f} conf={r['confidence']:.2f}{layer}]"
            f" ({r['insight_type']}{champ})"
        )
        print(f"     {r['text'][:120]}{'...' if len(r['text']) > 120 else ''}")

    print()
    answer_good = _prompt_gb("Answer good or bad? (g/b): ")

    # Explain the confidence alignment question based on the answer
    if answer_good:
        print("  → Did the TOP insights (highest confidence) contain the correct advice?")
    else:
        print("  → Were the TOP insights (highest confidence) high-scoring but WRONG/irrelevant?")
        print("    (high-conf + wrong = worst signal — model was confidently hallucinating)")

    confidence_aligned = _prompt_yn("Confidence aligned with quality? (y/n): ")

    insight_ids = [r.get("id") for r in all_insights if r.get("id")]
    save_rating(
        question=question,
        intent=intent,
        answer_good=answer_good,
        confidence_aligned=confidence_aligned,
        insight_ids=insight_ids,
        generated_answer=generated,
        query_id=query_id,
    )

    label = "GOOD" if answer_good else "BAD"
    align = "aligned" if confidence_aligned else "MISALIGNED"
    print(f"  Saved — {label}, confidence {align}")


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats() -> None:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM eval_ratings").fetchone()[0]
        if total == 0:
            print("No ratings yet.")
            return

        good_pct = conn.execute(
            "SELECT 100.0 * SUM(answer_good) / COUNT(*) FROM eval_ratings"
        ).fetchone()[0]
        aligned_pct = conn.execute(
            "SELECT 100.0 * SUM(confidence_aligned) / COUNT(*) FROM eval_ratings"
        ).fetchone()[0]

        # Worst case: high-conf + wrong
        bad_misaligned = conn.execute(
            "SELECT COUNT(*) FROM eval_ratings WHERE answer_good = 0 AND confidence_aligned = 0"
        ).fetchone()[0]

        by_intent = conn.execute("""
            SELECT intent_type,
                   COUNT(*) as n,
                   100.0 * SUM(answer_good) / COUNT(*) as good_pct
            FROM eval_ratings
            GROUP BY intent_type
        """).fetchall()

    print(f"\nTotal ratings      : {total}")
    print(f"Good answers       : {good_pct:.0f}%")
    print(f"Confidence aligned : {aligned_pct:.0f}%")
    print(f"High-conf wrong    : {bad_misaligned}  ← calibration failures\n")

    print("By intent type:")
    for r in by_intent:
        itype = r["intent_type"] or "general"
        print(f"  {itype:<12} {r['n']} ratings   {r['good_pct']:.0f}% good")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive answer rating harness")
    parser.add_argument("question", nargs="*", help="Rate a one-off question directly")
    parser.add_argument("--add", action="store_true", help="Add a question to eval_queries")
    parser.add_argument("--stats", action="store_true", help="Show rating distribution")
    args = parser.parse_args()

    init_db()

    if args.stats:
        print_stats()
        return

    if args.add:
        question = input("Question: ").strip()
        expected = input("Expected answer (optional, press enter to skip): ").strip()
        notes = input("Notes/source (optional): ").strip()
        qid = add_eval_query(question, expected, notes)
        print(f"Added query #{qid}: {question}")
        return

    if args.question:
        rate_question(" ".join(args.question))
        return

    # Rate all unrated eval_queries
    pending = get_unrated_queries()
    if not pending:
        print("No unrated queries. Use --add to add questions, or pass one directly.")
        return

    print(f"{len(pending)} unrated question(s) in the queue.")
    for row in pending:
        try:
            rate_question(row["question"], query_id=row["id"], expected=row["expected_answer"] or "")
        except KeyboardInterrupt:
            print("\nStopped.")
            break

    print_stats()


if __name__ == "__main__":
    main()
