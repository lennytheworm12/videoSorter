"""
Synthesise ability_windows insights for each champion by combining:
  - Data Dragon ability descriptions + mechanical property tags (from champion_abilities)
  - Existing video insights (champion_mechanics, matchup_advice, laning_tips)

ability_windows are coaching cues that tie ability mechanics to decision-making:
  "After landing E (grounded), the ~1.5s window before it expires is when you
   commit your W poison + Q — the enemy cannot dash away."

These are NOT re-extracted from transcripts — they are LLM-synthesized from
structured ability data + real coach insights, so they live at the intersection
of game knowledge and observed coaching patterns.

Usage:
    uv run python -m pipeline.ability_enrich                   # all champions
    uv run python -m pipeline.ability_enrich --champion "Cassiopeia"
    uv run python -m pipeline.ability_enrich --status          # show coverage
    uv run python -m pipeline.ability_enrich --dry-run         # preview without saving
"""

import json
import textwrap
import argparse
from core.database import get_connection, init_db
from core.llm import chat as llm_chat

# Synthetic video_id for ability_windows rows so they're clearly not real videos
_ABILITY_VIDEO_ID = "__ability_enrichment__"

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert League of Legends coach and game analyst.
    Your job is to synthesize precise ability interaction cues — called "ability_windows" —
    by combining a champion's ability descriptions with patterns observed by real coaches.

    An ability_window insight describes a moment in gameplay tied to a specific ability:
    - When a CC window opens (and for how long) and what follow-up is possible
    - What abilities or summoner spells can interrupt/escape a given situation
    - Cooldown patterns that create trade opportunities or vulnerability
    - Dash/blink vs grounded — which escapes get cut off
    - Channels and how they can be interrupted
    - Combo sequences tied to ability properties (silence → cannot retaliate, stun → free damage)

    Rules:
    1. Every insight must reference at least one named ability (by slot or name).
    2. Be specific: name the window, name the follow-up, name the limitation.
    3. Ground claims in the ability data provided — do not invent cooldowns or effects.
    4. If real coach insights are provided, build on them rather than repeating them verbatim.
    5. Return valid JSON only. No markdown, no text outside the JSON.
    6. Skip trivial observations ("E slows enemies") — only insights a player would miss
       without coaching: timing edges, counter-play, combo sequencing.

    Output format:
    {
        "ability_windows": [
            "insight text here",
            ...
        ]
    }

    Return 4–8 high-quality insights. Return [] if the champion's kit offers nothing
    non-obvious (rare).
""").strip()

ENRICH_PROMPT = textwrap.dedent("""
    Champion: {champion}

    ABILITY DATA (from Riot Data Dragon):
    {ability_block}

    EXISTING COACH INSIGHTS (observed from real videos — use these as grounding):
    {insight_block}

    Synthesize ability_windows insights for this champion. Focus on:
    - CC combo chains (what can you do inside the CC window?)
    - Mobility counters (which abilities ground, silence, or interrupt dashes/channels?)
    - Trade windows (when are they forced out of a dash/escape, leaving them open?)
    - QSS/Cleanse interactions (which effects get cleansed, which don't?)
    - Key cooldowns that create vulnerability when spent

    Return only JSON in the format shown in the system prompt.
""").strip()


def _get_ability_block(champion: str) -> str:
    """Format ability data as a readable block for the prompt."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT ability_slot, name, description, cooldown, range, properties
            FROM champion_abilities
            WHERE champion = ?
            ORDER BY CASE ability_slot
                WHEN 'P' THEN 0 WHEN 'Q' THEN 1 WHEN 'W' THEN 2
                WHEN 'E' THEN 3 WHEN 'R' THEN 4 ELSE 5 END
            """,
            (champion,)
        ).fetchall()

    if not rows:
        return "(no ability data)"

    lines = []
    for row in rows:
        props = json.loads(row["properties"] or "[]")
        prop_str = f"  [tags: {', '.join(props)}]" if props else ""
        cd_str = f" | CD: {row['cooldown']}s" if row["cooldown"] else ""
        rng_str = f" | Range: {row['range']}" if row["range"] and row["range"] != "self" else ""
        lines.append(f"[{row['ability_slot']}] {row['name']}{cd_str}{rng_str}")
        lines.append(f"  {row['description'][:300]}")
        if prop_str:
            lines.append(prop_str)
        lines.append("")
    return "\n".join(lines).strip()


def _get_insight_block(champion: str, max_insights: int = 20) -> str:
    """Pull the most relevant existing insights for this champion."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT i.insight_type, i.text
            FROM insights i
            JOIN videos v ON i.video_id = v.video_id
            WHERE v.champion = ?
              AND i.insight_type IN (
                  'champion_mechanics', 'matchup_advice', 'laning_tips',
                  'champion_identity', 'teamfight_tips'
              )
              AND i.is_duplicate = 0
            ORDER BY i.source_score DESC NULLS LAST
            LIMIT ?
            """,
            (champion, max_insights)
        ).fetchall()

    if not rows:
        return "(no video insights available)"

    lines = []
    for row in rows:
        lines.append(f"[{row['insight_type']}] {row['text']}")
    return "\n".join(lines)


def _existing_windows(champion: str) -> int:
    """Count already-generated ability_windows for this champion."""
    with get_connection() as conn:
        r = conn.execute(
            """
            SELECT COUNT(*) FROM insights i
            JOIN videos v ON i.video_id = v.video_id
            WHERE v.champion = ? AND i.insight_type = 'ability_windows'
            """,
            (champion,)
        ).fetchone()
    return r[0]


def _ensure_synthetic_video(champion: str) -> str:
    """
    Ensure a synthetic video row exists for this champion's ability enrichment.
    Returns the video_id to use.
    """
    safe = champion.lower().replace(" ", "_").replace("'", "")
    vid_id = f"__ability__{safe}__"
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO videos
                (video_id, video_url, role, message_timestamp, video_title, champion)
            VALUES (?, '', 'ability_enrichment', '', ?, ?)
            """,
            (vid_id, f"Ability enrichment: {champion}", champion)
        )
        conn.commit()
    return vid_id


def enrich_champion(champion: str, dry_run: bool = False) -> list[str]:
    """
    Generate ability_windows insights for one champion.
    Returns the list of insight strings.
    """
    ability_block = _get_ability_block(champion)
    insight_block = _get_insight_block(champion)

    prompt = ENRICH_PROMPT.format(
        champion=champion,
        ability_block=ability_block,
        insight_block=insight_block,
    )

    raw = llm_chat(SYSTEM_PROMPT, prompt)

    try:
        parsed = json.loads(raw)
        windows = parsed.get("ability_windows", [])
        if not isinstance(windows, list):
            windows = []
    except json.JSONDecodeError:
        # Try to extract JSON block from response
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                windows = parsed.get("ability_windows", [])
            except Exception:
                windows = []
        else:
            windows = []

    if dry_run:
        print(f"\n  [dry-run] {champion}: {len(windows)} windows")
        for w in windows:
            print(f"    • {w}")
        return windows

    if windows:
        vid_id = _ensure_synthetic_video(champion)
        with get_connection() as conn:
            for w in windows:
                if isinstance(w, str) and w.strip():
                    conn.execute(
                        "INSERT INTO insights (video_id, insight_type, text) VALUES (?, ?, ?)",
                        (vid_id, "ability_windows", w.strip())
                    )
            conn.commit()

    return windows


# ── Status ────────────────────────────────────────────────────────────────────

def print_status() -> None:
    with get_connection() as conn:
        n = conn.execute(
            "SELECT COUNT(DISTINCT champion) FROM videos WHERE role = 'ability_enrichment'"
        ).fetchone()[0]
        total = conn.execute(
            """
            SELECT COUNT(*) FROM insights i
            JOIN videos v ON i.video_id = v.video_id
            WHERE v.role = 'ability_enrichment' AND i.insight_type = 'ability_windows'
            """
        ).fetchone()[0]

    print(f"\nability_windows: {n} champions enriched, {total} total insights\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize ability_windows insights")
    parser.add_argument("--champion", help="Enrich a single champion only")
    parser.add_argument("--status", action="store_true", help="Show coverage")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--force", action="store_true",
                        help="Re-generate even if windows already exist")
    args = parser.parse_args()

    init_db()

    if args.status:
        print_status()
        return

    if args.champion:
        champions = [args.champion]
    else:
        # Only enrich champions we actually have video data for
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT champion FROM videos WHERE champion IS NOT NULL AND role != 'ability_enrichment'"
            ).fetchall()
        champions = [r[0] for r in rows if r[0]]

    print(f"Enriching {len(champions)} champion(s)…")
    total_new = 0
    skipped = 0
    errors = 0

    for i, champion in enumerate(champions, 1):
        if not args.force and not args.dry_run:
            existing = _existing_windows(champion)
            if existing > 0:
                skipped += 1
                continue

        try:
            windows = enrich_champion(champion, dry_run=args.dry_run)
            total_new += len(windows)
            if not args.dry_run:
                print(f"  [{i}/{len(champions)}] {champion}: {len(windows)} windows")
        except Exception as e:
            print(f"  [error] {champion}: {e}")
            errors += 1

    if not args.dry_run:
        print(f"\nDone — {total_new} ability_windows inserted, {skipped} skipped (already done), {errors} errors.")
        print_status()


if __name__ == "__main__":
    main()
