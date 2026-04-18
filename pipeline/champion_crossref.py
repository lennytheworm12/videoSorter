"""
Champion cross-reference pipeline.

Builds a knowledge bridge so insights from champion X can inform questions
about champion Y in the same archetype, even if no direct video exists for Y.

Champion similarity is role-agnostic: Braum played ADC is still an engage
tank and should match Leona/Nautilus regardless of role. Each champion gets
one vector (all champion_identity insights pooled) and one archetype.

Three steps:
  1. Champion vectors — mean-pool ALL champion_identity embeddings per champion
  2. KNN within archetypes — cosine similarity to rank archetype peers (no role gate)
  3. Insight generalization — Gemini labels each champion_identity insight as
     'specific' (stay with this champion) or 'generalizable' (transfer to archetype)

Output stored in two new DB tables:
  champion_vectors   — one row per champion with mean-pooled embedding
  crossref_insights  — generalizable insights tagged with their archetype

Usage:
    python -m pipeline.champion_crossref              # full run
    python -m pipeline.champion_crossref --status     # show coverage
    python -m pipeline.champion_crossref --vectors    # only build champion vectors
    python -m pipeline.champion_crossref --generalize # only run generalization pass
"""

import argparse
import json
import numpy as np
from core.database import get_connection, init_db
from core.llm import chat as llm_chat

# Minimum cosine similarity to count as a meaningful archetype neighbor
KNN_THRESHOLD = 0.50
# Max neighbors to store per champion
MAX_NEIGHBORS = 5


# ── DB setup ──────────────────────────────────────────────────────────────────

def _init_tables() -> None:
    with get_connection() as conn:
        # Migrate old role-keyed champion_vectors if it exists
        cols = {r[1] for r in conn.execute("PRAGMA table_info(champion_vectors)").fetchall()}
        if cols and "role" in cols:
            conn.execute("DROP TABLE champion_vectors")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS champion_vectors (
                champion   TEXT PRIMARY KEY,
                archetype  TEXT,
                vector     BLOB NOT NULL,
                n_insights INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        # Migrate old role-keyed crossref_insights if it exists
        cols = {r[1] for r in conn.execute("PRAGMA table_info(crossref_insights)").fetchall()}
        if cols and "role" in cols:
            conn.execute("DROP TABLE crossref_insights")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS crossref_insights (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_id   INTEGER NOT NULL REFERENCES insights(id),
                champion     TEXT NOT NULL,
                archetype    TEXT NOT NULL,
                scope        TEXT NOT NULL,  -- 'specific' or 'generalizable'
                created_at   TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()


# ── Step 1: Champion vectors ──────────────────────────────────────────────────

def build_champion_vectors() -> dict[str, np.ndarray]:
    """
    Mean-pool ALL champion_identity insight embeddings per champion (role-agnostic).
    Returns dict mapping champion → normalised mean vector.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT i.embedding, v.champion
            FROM insights i
            JOIN videos v ON i.video_id = v.video_id
            WHERE i.insight_type = 'champion_identity'
              AND i.embedding IS NOT NULL
              AND v.champion IS NOT NULL
        """).fetchall()

    if not rows:
        print("  No champion_identity embeddings found — run embed.py first.")
        return {}

    # Group vectors by champion (ignore role)
    groups: dict[str, list[np.ndarray]] = {}
    for row in rows:
        vec = np.frombuffer(row["embedding"], dtype=np.float32)
        groups.setdefault(row["champion"], []).append(vec)

    # Load archetype map (champion → archetype, role-agnostic: pick first match)
    with get_connection() as conn:
        arch_rows = conn.execute(
            "SELECT champion, archetype FROM champion_archetypes"
        ).fetchall()
    archetype_map: dict[str, str] = {}
    for r in arch_rows:
        archetype_map.setdefault(r["champion"].lower(), r["archetype"])

    # Mean-pool and normalise
    champion_vectors: dict[str, np.ndarray] = {}
    upserts = []
    for champion, vecs in groups.items():
        mean_vec = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm
        champion_vectors[champion] = mean_vec
        archetype = archetype_map.get(champion.lower())
        upserts.append((champion, archetype, mean_vec.tobytes(), len(vecs)))

    with get_connection() as conn:
        conn.executemany("""
            INSERT INTO champion_vectors (champion, archetype, vector, n_insights)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(champion) DO UPDATE SET
                archetype  = excluded.archetype,
                vector     = excluded.vector,
                n_insights = excluded.n_insights
        """, upserts)
        conn.commit()

    print(f"  Built {len(champion_vectors)} champion vector(s).")
    return champion_vectors


def load_champion_vectors() -> dict[str, np.ndarray]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT champion, vector FROM champion_vectors"
        ).fetchall()
    return {
        r["champion"]: np.frombuffer(r["vector"], dtype=np.float32)
        for r in rows
    }


# ── Step 2: KNN within archetypes ─────────────────────────────────────────────

def compute_archetype_neighbors() -> dict[str, list[dict]]:
    """
    For each champion with a vector, find the most similar champions
    in the same archetype ranked by cosine similarity (role-agnostic).

    Returns dict mapping champion → list of neighbor dicts:
        {champion, archetype, similarity}
    """
    vectors = load_champion_vectors()
    if not vectors:
        print("  No champion vectors found — run --vectors first.")
        return {}

    # Load archetype per champion (role-agnostic: first entry wins)
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT champion, archetype FROM champion_archetypes"
        ).fetchall()
    archetype_map: dict[str, str] = {}
    for r in rows:
        archetype_map.setdefault(r["champion"].lower(), r["archetype"])

    neighbors: dict[str, list[dict]] = {}

    for champ, vec in vectors.items():
        archetype = archetype_map.get(champ.lower())
        if not archetype:
            continue

        # Find all same-archetype champions that also have vectors (no role gate)
        peers = [
            (c, v)
            for c, v in vectors.items()
            if archetype_map.get(c.lower()) == archetype
            and c.lower() != champ.lower()
        ]

        if not peers:
            neighbors[champ] = []
            continue

        scored = sorted(
            [(c, float(vec @ v)) for c, v in peers],
            key=lambda x: x[1],
            reverse=True,
        )

        neighbors[champ] = [
            {"champion": c, "archetype": archetype, "similarity": round(sim, 4)}
            for c, sim in scored
            if sim >= KNN_THRESHOLD
        ][:MAX_NEIGHBORS]

    return neighbors


# ── Step 3: Insight generalization ────────────────────────────────────────────

GENERALIZE_SYSTEM = """
You are a League of Legends coaching expert. Given a coaching insight about a
specific champion, decide if it generalizes to all champions of the same archetype
or is too champion-specific to transfer.

Return valid JSON only — no markdown, no explanation.
""".strip()

GENERALIZE_PROMPT = """
Champion: {champion} ({role})
Archetype: {archetype}
Insight: "{text}"

Is this insight generalizable to all {archetype} champions, or is it specific to {champion}?

Rules:
- 'generalizable' if the advice applies to the archetype's general playstyle
- 'specific' if it relies on {champion}'s unique abilities, stats, or kit

Return: {{"scope": "generalizable"}} or {{"scope": "specific"}}
""".strip()


def generalize_insights(dry_run: bool = False) -> None:
    """
    Label each champion_identity insight as 'specific' or 'generalizable'.
    Skips insights already labeled in crossref_insights.
    """
    # Get already-processed insight IDs
    with get_connection() as conn:
        done = {
            r[0] for r in conn.execute(
                "SELECT insight_id FROM crossref_insights"
            ).fetchall()
        }

    # Load champion_identity insights with archetype info (role-agnostic: first match)
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT i.id, i.text, v.champion, v.role,
                   (SELECT ca.archetype FROM champion_archetypes ca
                    WHERE LOWER(ca.champion) = LOWER(v.champion)
                    LIMIT 1) AS archetype
            FROM insights i
            JOIN videos v ON i.video_id = v.video_id
            WHERE i.insight_type = 'champion_identity'
              AND v.champion IS NOT NULL
        """).fetchall()

    pending = [r for r in rows if r["id"] not in done]
    if not pending:
        print("  All champion_identity insights already labeled.")
        return

    print(f"  Labeling {len(pending)} insight(s)…")
    if dry_run:
        print("  [dry-run] skipping LLM calls.")
        return

    BATCH_SIZE = 50
    batch: list[tuple] = []
    total_saved = 0
    total_gen = 0

    def _flush(b: list[tuple]) -> None:
        nonlocal total_saved, total_gen
        with get_connection() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO crossref_insights
                    (insight_id, champion, archetype, scope)
                VALUES (?, ?, ?, ?)
            """, b)
            conn.commit()
        total_saved += len(b)
        total_gen += sum(1 for _, _, _, s in b if s == "generalizable")

    for i, row in enumerate(pending, 1):
        prompt = GENERALIZE_PROMPT.format(
            champion=row["champion"],
            role=row["role"],
            archetype=row["archetype"],
            text=row["text"],
        )
        try:
            raw = llm_chat(system=GENERALIZE_SYSTEM, user=prompt, temperature=0.0)
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
            scope = data.get("scope", "specific")
            if scope not in ("specific", "generalizable"):
                scope = "specific"
        except Exception as e:
            print(f"    [error] insight {row['id']}: {e}")
            scope = "specific"

        batch.append((
            row["id"], row["champion"], row["archetype"] or "unknown", scope
        ))

        if len(batch) >= BATCH_SIZE:
            _flush(batch)
            batch = []

        if i % 50 == 0:
            print(f"    {i}/{len(pending)} labeled… ({total_gen} generalizable so far)")

    if batch:
        _flush(batch)

    print(f"  Done — {total_gen}/{total_saved} labeled generalizable.")


# ── Status ────────────────────────────────────────────────────────────────────

def print_status() -> None:
    with get_connection() as conn:
        n_vectors = conn.execute("SELECT COUNT(*) FROM champion_vectors").fetchone()[0]
        n_labeled = conn.execute("SELECT COUNT(*) FROM crossref_insights").fetchone()[0]
        n_gen = conn.execute(
            "SELECT COUNT(*) FROM crossref_insights WHERE scope = 'generalizable'"
        ).fetchone()[0]

        top_archetypes = conn.execute("""
            SELECT archetype, COUNT(*) as n
            FROM crossref_insights WHERE scope = 'generalizable'
            GROUP BY archetype ORDER BY n DESC LIMIT 10
        """).fetchall()

    print(f"\nChampion vectors:  {n_vectors}")
    print(f"Labeled insights:  {n_labeled}  ({n_gen} generalizable)\n")
    if top_archetypes:
        print("Top generalizable archetypes:")
        for r in top_archetypes:
            print(f"  {r['archetype']:<25} {r['n']} insights")
    print()


# ── Query helper (used by retrieval) ──────────────────────────────────────────

def get_archetype_insights(
    champion: str,
    role: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Return generalizable insights from same-archetype champions,
    ranked by (similarity × confidence). Role-agnostic: Braum ADC
    still matches Leona/Nautilus because they share the Warden archetype.

    Used by query.py as the layer-2 archetype fallback.
    """
    # Get this champion's archetype (role-agnostic: first match)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT archetype FROM champion_archetypes WHERE LOWER(champion) = LOWER(?) LIMIT 1",
            (champion,),
        ).fetchone()
    if not row:
        return []
    archetype = row["archetype"]

    # Load champion vectors for similarity scoring
    vectors = load_champion_vectors()
    query_vec = vectors.get(champion)
    if query_vec is None:
        # Try case-insensitive lookup
        for k, v in vectors.items():
            if k.lower() == champion.lower():
                query_vec = v
                break

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT i.id, i.text, i.insight_type, i.confidence, i.source_score,
                   ci.champion as src_champion, ci.archetype, ci.scope,
                   cv.vector
            FROM crossref_insights ci
            JOIN insights i ON ci.insight_id = i.id
            LEFT JOIN champion_vectors cv ON LOWER(cv.champion) = LOWER(ci.champion)
            WHERE ci.archetype = ?
              AND ci.scope = 'generalizable'
              AND LOWER(ci.champion) != LOWER(?)
        """, (archetype, champion)).fetchall()

    if not rows:
        return []

    results = []
    for row in rows:
        similarity = 1.0
        if query_vec is not None and row["vector"]:
            peer_vec = np.frombuffer(row["vector"], dtype=np.float32)
            similarity = float(query_vec @ peer_vec)

        conf = row["confidence"] or 0.5
        transfer_score = round(similarity * conf, 4)

        results.append({
            "id": row["id"],
            "text": row["text"],
            "insight_type": row["insight_type"],
            "source_champion": row["src_champion"],
            "archetype": row["archetype"],
            "similarity": round(similarity, 4),
            "confidence": conf,
            "transfer_score": transfer_score,
            "retrieval_layer": "archetype",
        })

    results.sort(key=lambda x: x["transfer_score"], reverse=True)
    return results[:top_k]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Champion cross-reference pipeline")
    parser.add_argument("--status",     action="store_true", help="Show coverage and exit")
    parser.add_argument("--vectors",    action="store_true", help="Only build champion vectors")
    parser.add_argument("--generalize", action="store_true", help="Only run generalization labeling")
    parser.add_argument("--dry-run",    action="store_true", help="Skip LLM calls")
    args = parser.parse_args()

    init_db()
    _init_tables()

    if args.status:
        print_status()
        return

    if not args.generalize:
        print("Step 1 — Building champion vectors…")
        build_champion_vectors()

        print("Step 2 — Computing archetype neighbors…")
        neighbors = compute_archetype_neighbors()
        covered = sum(1 for n in neighbors.values() if n)
        total_neighbors = sum(len(n) for n in neighbors.values())
        print(f"  {covered}/{len(neighbors)} champions have archetype neighbors "
              f"({total_neighbors} total neighbor pairs)")

    if not args.vectors:
        print("Step 3 — Generalizing champion_identity insights…")
        generalize_insights(dry_run=args.dry_run)

    print_status()


if __name__ == "__main__":
    main()
