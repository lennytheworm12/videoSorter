"""
Dump Aatrox insights from knowledge.db for before/after comparison.

Usage:
    uv run python snapshots/dump_guide_insights.py [champion]
"""
import sys

from core.db_paths import activate_knowledge_db

activate_knowledge_db()

from core.database import get_connection

champion = sys.argv[1] if len(sys.argv) > 1 else "Aatrox"

with get_connection() as conn:
    rows = conn.execute("""
        SELECT i.insight_type, i.text, i.source_score, i.repetition_count,
               v.video_title, v.video_id
        FROM insights i
        JOIN videos v ON i.video_id = v.video_id
        WHERE v.champion = ?
        ORDER BY i.insight_type, i.source_score DESC NULLS LAST
    """, (champion,)).fetchall()

lines = [f"Guide insights for {champion} ({len(rows)} total)\n"]
current_type = None
for r in rows:
    if r["insight_type"] != current_type:
        current_type = r["insight_type"]
        lines.append(f"\n=== {current_type} ===")
    score = f" (score={r['source_score']:.2f})" if r["source_score"] else ""
    rep   = f" [rep={r['repetition_count']}]" if r["repetition_count"] and r["repetition_count"] > 1 else ""
    src   = f"\n      src: {r['video_title'][:60]}" if r["video_title"] else ""
    lines.append(f"  - {r['text']}{score}{rep}{src}")

out = "\n".join(lines)
print(out)

safe = champion.lower().replace(" ", "_").replace("'", "")
path = f"snapshots/{safe}_after.txt"
with open(path, "w") as f:
    f.write(out)
print(f"\nSaved {path}")
