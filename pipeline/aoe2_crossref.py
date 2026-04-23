"""
AoE2 civilization cross-reference pipeline.

Builds reusable transfer metadata so civ-specific AoE2 insights can be reused
for similar civilizations when the situation also matches the player's query.

Three steps:
  1. Civilization vectors — mean-pool embedded civ-specific AoE2 insights
  2. Applicability labeling — Gemini labels civ insights as transferable/specific
     and assigns reusable situation tags
  3. Query helper — retrieval can pull transferable insights from similar civs

Usage:
    python -m pipeline.aoe2_crossref
    python -m pipeline.aoe2_crossref --status
    python -m pipeline.aoe2_crossref --vectors
    python -m pipeline.aoe2_crossref --label
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable

import numpy as np

from core.db_paths import activate_knowledge_db

activate_knowledge_db()

from core.database import get_connection, init_db
from core.game_registry import canonical_aoe2_civilization
from core.llm import chat as llm_chat

SIMILARITY_THRESHOLD = 0.45
MAX_NEIGHBORS = 6

VECTOR_INSIGHT_TYPES = (
    "civilization_identity",
    "matchup_advice",
    "unit_compositions",
    "build_orders",
    "feudal_age",
    "castle_age",
    "imperial_age",
    "map_control",
)

TRANSFERABLE_INSIGHT_TYPES = (
    "civilization_identity",
    "game_mechanics",
    "micro",
    "principles",
    "build_orders",
    "dark_age",
    "feudal_age",
    "castle_age",
    "imperial_age",
    "economy_macro",
    "scouting",
    "unit_compositions",
    "map_control",
    "matchup_advice",
    "general_advice",
)

SITUATION_TAGS = (
    "dark_age",
    "feudal_pressure",
    "castle_timing",
    "imperial_transition",
    "boom",
    "defense",
    "scouting",
    "economy",
    "map_control",
    "cavalry",
    "archers",
    "infantry",
    "siege",
    "monks",
    "raiding",
    "tech_switch",
)

LABEL_SYSTEM = """
You are an Age of Empires II strategy analyst. Decide whether a civ-specific
insight transfers cleanly to other similar civilizations and tag the gameplay
situations it applies to.

Return valid JSON only.
""".strip()

LABEL_PROMPT = """
Civilization: {subject}
Insight type: {insight_type}
Insight: "{text}"

Allowed tags:
{tag_list}

Decide whether this insight is:
- "transferable" if it can help other similar civilizations in the same kind of spot
- "specific" if it depends on this civilization's unique bonuses, units, or tech tree

Tag rules:
- Use only tags from the allowed list
- Include only tags directly supported by the insight
- Use [] when no tag is strongly supported

Return exactly:
{{"scope":"transferable"|"specific","situation_tags":["tag"]}}
""".strip()


def _init_tables() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS aoe2_civilization_vectors (
                subject    TEXT PRIMARY KEY,
                vector     BLOB NOT NULL,
                n_insights INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS aoe2_crossref_insights (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_id     INTEGER NOT NULL REFERENCES insights(id),
                subject        TEXT NOT NULL,
                scope          TEXT NOT NULL,
                situation_tags TEXT NOT NULL,
                created_at     TEXT DEFAULT (datetime('now')),
                UNIQUE(insight_id)
            )
        """)
        conn.commit()


def build_civilization_vectors() -> dict[str, np.ndarray]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT i.embedding, i.subject
            FROM insights i
            JOIN videos v ON i.video_id = v.video_id
            WHERE v.game = 'aoe2'
              AND i.embedding IS NOT NULL
              AND i.subject_type = 'civ'
              AND i.subject IS NOT NULL
              AND i.insight_type IN ({placeholders})
            """.format(placeholders=",".join("?" for _ in VECTOR_INSIGHT_TYPES)),
            VECTOR_INSIGHT_TYPES,
        ).fetchall()

    if not rows:
        print("  No civ-specific AoE2 embeddings found — run analyze + embed first.")
        return {}

    groups: dict[str, list[np.ndarray]] = {}
    for row in rows:
        subject = canonical_aoe2_civilization(row["subject"]) or row["subject"]
        vec = np.frombuffer(row["embedding"], dtype=np.float32)
        groups.setdefault(subject, []).append(vec)

    vectors: dict[str, np.ndarray] = {}
    upserts: list[tuple[str, bytes, int]] = []
    for subject, vecs in groups.items():
        mean_vec = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm
        vectors[subject] = mean_vec
        upserts.append((subject, mean_vec.tobytes(), len(vecs)))

    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO aoe2_civilization_vectors (subject, vector, n_insights)
            VALUES (?, ?, ?)
            ON CONFLICT(subject) DO UPDATE SET
                vector = excluded.vector,
                n_insights = excluded.n_insights
            """,
            upserts,
        )
        conn.commit()

    print(f"  Built {len(vectors)} civilization vector(s).")
    return vectors


def load_civilization_vectors() -> dict[str, np.ndarray]:
    with get_connection() as conn:
        try:
            rows = conn.execute(
                "SELECT subject, vector FROM aoe2_civilization_vectors"
            ).fetchall()
        except Exception:
            return {}
    return {
        row["subject"]: np.frombuffer(row["vector"], dtype=np.float32)
        for row in rows
    }


def _normalize_tags(tags: Iterable[str] | None) -> list[str]:
    if not tags:
        return []
    allowed = set(SITUATION_TAGS)
    clean: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if not isinstance(tag, str):
            continue
        tag = tag.strip().lower()
        if tag in allowed and tag not in seen:
            clean.append(tag)
            seen.add(tag)
    return clean


def label_applicable_insights(dry_run: bool = False) -> None:
    with get_connection() as conn:
        done = {
            row["insight_id"]
            for row in conn.execute(
                "SELECT insight_id FROM aoe2_crossref_insights"
            ).fetchall()
        }
        rows = conn.execute(
            """
            SELECT i.id, i.text, i.insight_type, i.subject
            FROM insights i
            JOIN videos v ON i.video_id = v.video_id
            WHERE v.game = 'aoe2'
              AND i.subject_type = 'civ'
              AND i.subject IS NOT NULL
              AND i.insight_type IN ({placeholders})
            """.format(placeholders=",".join("?" for _ in TRANSFERABLE_INSIGHT_TYPES)),
            TRANSFERABLE_INSIGHT_TYPES,
        ).fetchall()

    pending = [row for row in rows if row["id"] not in done]
    if not pending:
        print("  All AoE2 civ insights already labeled.")
        return

    print(f"  Labeling {len(pending)} AoE2 insight(s)…")
    if dry_run:
        print("  [dry-run] skipping LLM calls.")
        return

    tag_list = "\n".join(f"- {tag}" for tag in SITUATION_TAGS)
    batch: list[tuple[int, str, str, str]] = []
    total_saved = 0
    total_transferable = 0
    total_skipped = 0

    def _flush(items: list[tuple[int, str, str, str]]) -> None:
        nonlocal total_saved, total_transferable
        with get_connection() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO aoe2_crossref_insights
                    (insight_id, subject, scope, situation_tags)
                VALUES (?, ?, ?, ?)
                """,
                items,
            )
            conn.commit()
        total_saved += len(items)
        total_transferable += sum(1 for _, _, scope, _ in items if scope == "transferable")

    for index, row in enumerate(pending, 1):
        prompt = LABEL_PROMPT.format(
            subject=row["subject"],
            insight_type=row["insight_type"],
            text=row["text"],
            tag_list=tag_list,
        )
        try:
            raw = llm_chat(system=LABEL_SYSTEM, user=prompt, temperature=0.0)
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
            scope = data.get("scope")
            if scope not in {"transferable", "specific"}:
                total_skipped += 1
                continue
            tags = json.dumps(_normalize_tags(data.get("situation_tags")), sort_keys=True)
        except Exception as exc:
            print(f"    [error] insight {row['id']}: {exc}")
            total_skipped += 1
            continue

        batch.append((row["id"], row["subject"], scope, tags))
        if len(batch) >= 50:
            _flush(batch)
            batch = []

        if index % 50 == 0:
            print(
                f"    {index}/{len(pending)} processed… "
                f"({total_saved} labeled, {total_transferable} transferable, {total_skipped} skipped)"
            )

    if batch:
        _flush(batch)

    print(
        f"  Done — {total_transferable}/{total_saved} labeled transferable. "
        f"{total_skipped} skipped."
    )


def get_applicable_insights(
    subject: str,
    preferred_types: list[str] | None = None,
    situation_tags: list[str] | None = None,
    top_k: int = 10,
) -> list[dict]:
    target = canonical_aoe2_civilization(subject) or subject
    vectors = load_civilization_vectors()
    if not vectors:
        return []

    query_vec = None
    for key, value in vectors.items():
        if key.lower() == target.lower():
            query_vec = value
            target = key
            break
    if query_vec is None:
        return []

    neighbor_scores = sorted(
        [
            (key, float(query_vec @ value))
            for key, value in vectors.items()
            if key.lower() != target.lower()
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    neighbor_scores = [
        (key, score)
        for key, score in neighbor_scores
        if score >= SIMILARITY_THRESHOLD
    ][:MAX_NEIGHBORS]
    if not neighbor_scores:
        return []
    allowed_subjects = {key for key, _ in neighbor_scores}
    similarity_map = {key.lower(): score for key, score in neighbor_scores}

    query_tags = set(_normalize_tags(situation_tags))
    preferred_map = {
        insight_type: max(0.0, 0.12 - (index * 0.02))
        for index, insight_type in enumerate(preferred_types or [])
    }

    with get_connection() as conn:
        try:
            rows = conn.execute(
                """
                SELECT i.id, i.text, i.insight_type, i.confidence, i.source_score,
                       x.subject AS source_subject, x.scope, x.situation_tags,
                       cv.vector
                FROM aoe2_crossref_insights x
                JOIN insights i ON x.insight_id = i.id
                LEFT JOIN aoe2_civilization_vectors cv ON LOWER(cv.subject) = LOWER(x.subject)
                WHERE x.scope = 'transferable'
                  AND LOWER(x.subject) != LOWER(?)
                """,
                (target,),
            ).fetchall()
        except Exception:
            return []

    results: list[dict] = []
    for row in rows:
        source_subject = canonical_aoe2_civilization(row["source_subject"]) or row["source_subject"]
        if source_subject not in allowed_subjects or not row["vector"]:
            continue
        similarity = similarity_map.get(source_subject.lower())
        if similarity is None:
            peer_vec = np.frombuffer(row["vector"], dtype=np.float32)
            similarity = float(query_vec @ peer_vec)

        row_tags = set(_normalize_tags(json.loads(row["situation_tags"] or "[]")))
        overlap = len(query_tags & row_tags)
        tag_score = overlap / max(len(query_tags), 1) if query_tags else 0.0
        combined_conf = 0.6 * float(row["confidence"] or 0.5) + 0.4 * float(row["source_score"] or 0.5)
        preferred_bonus = preferred_map.get(row["insight_type"], 0.0)
        transfer_score = round(
            (0.55 * similarity)
            + (0.20 * combined_conf)
            + (0.20 * tag_score)
            + preferred_bonus,
            4,
        )
        results.append({
            "id": row["id"],
            "text": row["text"],
            "insight_type": row["insight_type"],
            "subject": source_subject,
            "subject_type": "civ",
            "source_subject": source_subject,
            "situation_tags": sorted(row_tags),
            "role": None,
            "champion": None,
            "game": "aoe2",
            "rank": None,
            "website_rating": None,
            "source": "aoe2_crossref",
            "source_weight": 0.9,
            "score": round(similarity, 4),
            "confidence": round(combined_conf, 4),
            "transfer_score": transfer_score,
            "retrieval_layer": "aoe2_crossref",
        })

    results.sort(
        key=lambda item: (
            item["transfer_score"],
            item["score"],
            item["confidence"],
        ),
        reverse=True,
    )
    return results[:top_k]


def print_status() -> None:
    with get_connection() as conn:
        try:
            n_vectors = conn.execute(
                "SELECT COUNT(*) FROM aoe2_civilization_vectors"
            ).fetchone()[0]
            n_labeled = conn.execute(
                "SELECT COUNT(*) FROM aoe2_crossref_insights"
            ).fetchone()[0]
            n_transferable = conn.execute(
                "SELECT COUNT(*) FROM aoe2_crossref_insights WHERE scope = 'transferable'"
            ).fetchone()[0]
        except Exception:
            n_vectors = 0
            n_labeled = 0
            n_transferable = 0

    print(f"\nAoE2 civilization vectors: {n_vectors}")
    print(f"AoE2 labeled insights:     {n_labeled} ({n_transferable} transferable)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AoE2 civilization cross-reference pipeline")
    parser.add_argument("--status", action="store_true", help="Show coverage and exit")
    parser.add_argument("--vectors", action="store_true", help="Only build civilization vectors")
    parser.add_argument("--label", action="store_true", help="Only run applicability labeling")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls during labeling")
    args = parser.parse_args()

    init_db()
    _init_tables()

    if args.status:
        print_status()
        return

    if not args.label:
        print("Step 1 — Building civilization vectors…")
        build_civilization_vectors()

    if not args.vectors:
        print("Step 2 — Labeling transferable civ insights…")
        label_applicable_insights(dry_run=args.dry_run)

    print_status()


if __name__ == "__main__":
    main()
