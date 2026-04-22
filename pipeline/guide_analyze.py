"""
LLM analysis for guide-style sources in guide_test.db.

Supported sources:
  - youtube_guide   : transcribed educational videos
  - mobafire_guide  : cleaned written guide text saved directly into transcription

Usage:
    uv run python -m pipeline.guide_analyze               # all guide sources
    uv run python -m pipeline.guide_analyze --champion Aatrox
    uv run python -m pipeline.guide_analyze --source mobafire_guide
    uv run python -m pipeline.guide_analyze --dry-run     # print insights, don't save
"""

import os
import json
import time
import random
import textwrap
import argparse
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Write to isolated test DB until guide prompts are validated
os.environ.setdefault("DB_PATH", "guide_test.db")

from core.database import get_connection, set_status, insert_insight, init_db
from core.champions import correct_names
from core.game_registry import DEFAULT_GAME, SUPPORTED_GAMES, analysis_spec, normalize_game
from core.llm import chat as llm_chat, BACKEND

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_WINDOW_WORDS  = 40
_WINDOW_STRIDE = 20

# ~130 wpm speaking rate × 60 min = 7800; use 9000 to give a comfortable buffer
WORDS_PER_HOUR = 9_000

VIDEO_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert League of Legends analyst extracting actionable insights
    from YouTube guide and educational videos.

    SOURCE FILTERING — critical:
    These videos are solo presentations or educational gameplay breakdowns —
    there is no coach/student dynamic. The presenter is speaking directly to the
    viewer. Extract insights from any deliberate educational content.

    SKIP entirely:
      - Channel intros/outros ("welcome back", "don't forget to subscribe", "smash like")
      - Sponsor reads and promotional segments
      - Generic hype ("this champion is broken right now", "you NEED to play this")
      - Play-by-play narration with no educational content ("okay so here I walk up")
      - Obvious/baseline tips any player would know ("last hit minions for gold")
      - Patch-specific or seasonal content: current meta picks, item tier lists, "right
        now X is broken", rune page recommendations tied to a patch, anything that will
        be wrong next season. We want evergreen fundamentals only.
      - Rank-gated advice that only applies to a specific skill bracket: "at your elo",
        "in low elo you can get away with", "high elo players will punish this",
        "this only works below Diamond", "beginners should". If advice is only correct
        for some rank tiers, discard it — we serve players of all levels.

    TRANSCRIPT CONTEXT:
    Auto-generated captions — no punctuation, champion/item names often misspelled.
    Always output the correct LoL name (e.g. "kaisa" → "Kai'Sa", "khazix" → "Kha'Zix").

    INSIGHT CATEGORIES — definitions:

    champion_identity: This champion's unique strategic role, win conditions, power
    spikes, and what differentiates it from others. Must be specific to this champion.

    game_mechanics: ONLY advice about the game client — keybindings, settings, camera,
    mouse hardware. Almost always []. Do NOT put in-game decisions here.

    principles: Broad strategic mental models — wave theory, matchup archetypes,
    resource trading — that the video explicitly frames as general rules.

    laning_tips: Specific laning-phase decisions — wave management, trading patterns,
    positioning, recall timing. Champion-context is fine.

    champion_mechanics: How to use this champion's abilities — combos, sequencing,
    cooldown management, power spike windows, ability-specific interactions.

    matchup_advice: How to play against a specific champion or class. Must include
    both the condition (what the enemy does) and the adjustment required.

    champion_matchups: Direct champion-vs-champion notes where the enemy champion
    is explicitly named (for example "Against Fiora..." or "Versus Malphite...").
    Use this only for concrete champion-specific matchup guidance, not broad class
    advice like "against ranged champions".

    macro_advice: Post-laning — roaming, objectives, side lane, Teleport decisions,
    win condition execution.

    teamfight_tips: Positioning, target selection, engage/disengage, ability usage
    in team fights or skirmishes.

    vision_control: Ward placement, when to ward, contesting enemy vision.

    itemization: ONLY timeless build reasoning tied to a champion identity or matchup
    condition — e.g. "Rush Serylda's Grudge against tanks because the slow enables
    follow-up R". Skip specific starter items, rune pages, patch-tier-list picks, or
    anything prefaced with "right now" / "this season" / "currently". If the reasoning
    would be wrong next patch, do not include it.

    general_advice: Mindset and broadly applicable advice that doesn't fit above.
    Keep sparse.

    OUTPUT RULES:
    1. Return valid JSON only. No markdown fences, no text outside the JSON.
    2. Empty category = [] — never write a string.
    3. Each insight must be a complete standalone sentence, immediately applicable.
    4. Use correct LoL spelling for all names.
    5. Do not invent advice not present in the transcript.
    6. Prefer depth over breadth: one specific insight beats three vague ones.

    PHRASING RULES — critical for cross-video consistency:
    7. Write every insight as a second-person coaching instruction ("Use Q when...",
       "Avoid trading before...", "Build X against...") — not as a video observation
       ("the player uses Q", "in this clip he builds").
    8. Use canonical references: ability slots (Q, W, E, R), level numbers (level 5,
       level 6), item full names (Serylda's Grudge, not "that slow item").
    9. State the WHY concisely in the same sentence: "Avoid Q3 in lane unless you
       will secure damage — using all three charges puts Q on a 24s cooldown instead
       of 9s." A reason makes semantically similar tips from different videos cluster
       together naturally.
    10. Do not anchor to the specific video: no "as shown here", "in this game",
        "the streamer", or time references. The insight should read identically
        whether extracted from a 10-minute or 2-hour video.
    11. Evergreen over current: if an insight would be invalidated by a balance patch
        or season reset, discard it. Prioritise champion identity, mechanics, and
        decision-making patterns that hold across patches.
""").strip()

WRITTEN_GUIDE_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert League of Legends analyst extracting actionable insights
    from written champion guides.

    SOURCE FILTERING — critical:
    This is a written guide page, not a transcript. Treat section headings,
    matchup notes, and explanatory paragraphs as the source of truth.

    SKIP entirely:
      - Navigation chrome, comments, votes, "more guides", advertisements
      - Raw build tables, rune blocks, item lists, ability-order rows, and stat widgets
        unless the guide explicitly explains WHY a choice is correct
      - Author bio unless it contains evergreen champion-specific reasoning
      - Patch/season/current-meta claims, tier-list framing, or "right now" advice
      - Rank-gated advice that only works at low elo / high elo / for beginners
      - Generic filler that is not specific enough to coach off of

    PRIORITISE:
      - Champion identity and win conditions
      - Ability usage, trading patterns, lane plans, matchup adaptations
      - Teamfight role, side lane vs grouping logic, and execution details
      - Explanatory text around itemization when the reasoning is evergreen

    INSIGHT CATEGORIES — definitions:

    champion_identity: This champion's unique strategic role, win conditions, power
    spikes, and what differentiates it from others. Must be specific to this champion.

    game_mechanics: ONLY advice about the game client — keybindings, settings, camera,
    mouse hardware. Almost always [].

    principles: Broad strategic mental models and matchup logic that the guide frames
    as general rules, not just one isolated example.

    laning_tips: Specific laning-phase decisions — wave management, trading patterns,
    positioning, spacing, recall timing.

    champion_mechanics: How to use this champion's abilities — combos, sequencing,
    spacing, cooldown usage, level spikes, and ability-specific interactions.

    matchup_advice: How to play against a specific champion or class. Must include
    both the enemy condition and the needed adjustment.

    champion_matchups: Direct champion-vs-champion notes where the opposing champion
    is explicitly named. Keep only actionable matchup guidance that would help answer
    "{champion} into specific enemy champion" queries. If the enemy champion is not
    named directly, use matchup_advice instead.

    macro_advice: Post-laning — roaming, objective setup, side lane management,
    tempo, reset timing, split push vs group.

    teamfight_tips: Positioning, engage/disengage, target priority, execution in
    skirmishes and team fights.

    vision_control: Ward placement, timing, denial, and map setup with vision.

    itemization: ONLY evergreen item reasoning tied to identity, matchup, or enemy
    profile. Skip static full builds with no explanation.

    general_advice: Mindset or broadly useful advice that does not fit above.
    Keep sparse.

    OUTPUT RULES:
    1. Return valid JSON only. No markdown fences, no text outside the JSON.
    2. Empty category = [] — never write a string.
    3. Each insight must be a complete standalone sentence and immediately usable.
    4. Use correct LoL spelling for all names.
    5. Do not invent advice not present in the guide.
    6. Prefer direct second-person coaching instructions with the WHY included.
    7. Do not restate raw build tables or stat blocks unless the guide explains them.
""").strip()

EXTRACTION_PROMPT = textwrap.dedent("""
    Extract actionable insights from this League of Legends guide video transcript.

    Video info:
    - Champion: {champion}
    - Role: {role}
    - Title: {title}
    - Hour chunk: {chunk_label}

    Transcript (auto-generated captions):
    ---
    {transcript_chunk}
    ---

    Return exactly this JSON structure. Use [] for categories with no insights.
    Each insight is an object: {{"text": "...", "emphasis": 1|2|3}}
    (1=mentioned once, 2=a few times, 3=repeatedly stressed)

    Remember: write insights as second-person coaching instructions with the WHY
    included. "Use Q when the enemy commits to a last-hit animation — they cannot
    dodge during that window" not "the player uses Q on CS timing".

    Evergreen only: skip patch-specific builds, meta picks, seasonal tier lists, and
    anything prefaced with "right now" or "this season". Champion mechanics, identity,
    and decision-making patterns that hold across patches are the priority.

    {{
        "champion_identity": [],
        "game_mechanics": [],
        "principles": [],
        "laning_tips": [],
        "champion_mechanics": [],
        "champion_matchups": [],
        "matchup_advice": [],
        "macro_advice": [],
        "teamfight_tips": [],
        "vision_control": [],
        "itemization": [],
        "general_advice": []
    }}
""").strip()


_embed_model: SentenceTransformer | None = None
_EMBED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=_EMBED_DEVICE)
    return _embed_model


def _embed_chunk_windows(chunk: str) -> np.ndarray:
    words = chunk.split()
    windows = [
        " ".join(words[i: i + _WINDOW_WORDS])
        for i in range(0, max(1, len(words) - _WINDOW_WORDS + 1), _WINDOW_STRIDE)
    ]
    if not windows:
        windows = [chunk[:500]]
    model = _get_embed_model()
    return model.encode(windows, convert_to_numpy=True, normalize_embeddings=True)


def score_source_grounding(insight_text: str, window_matrix: np.ndarray) -> float:
    model = _get_embed_model()
    vec = model.encode(insight_text, convert_to_numpy=True, normalize_embeddings=True)
    return float((window_matrix @ vec).max())


def chunk_by_hour(transcript: str) -> list[str]:
    """
    Split a transcript into ~1-hour chunks, breaking at the last sentence-ending
    punctuation (. ! ?) within 500 words of the target size. Falls back to a
    hard word-count split if no punctuation is found. Each chunk overlaps the
    previous by 150 words so context isn't lost at boundaries.
    """
    import re as _re

    OVERLAP  = 150   # words of overlap between chunks
    SCAN     = 500   # words before target to search for a sentence end
    MIN_STEP = WORDS_PER_HOUR - SCAN  # minimum words advanced per chunk (~8500)

    words = transcript.split()
    total = len(words)
    if total <= WORDS_PER_HOUR:
        return [transcript]

    chunks: list[str] = []
    start = 0
    while start < total:
        target = min(start + WORDS_PER_HOUR, total)

        if target < total:
            # Scan the last SCAN words before the target for a sentence boundary
            scan_from = target - SCAN
            window = " ".join(words[scan_from:target])
            matches = list(_re.finditer(r'[.!?]', window))
            if matches:
                # Walk words in the window to find which word index the match lands on
                char_pos = matches[-1].start()
                running = 0
                split_offset = len(words[scan_from:target]) - 1  # fallback: end of window
                for wi, w in enumerate(words[scan_from:target]):
                    running += len(w) + 1
                    if running > char_pos:
                        split_offset = wi
                        break
                candidate = scan_from + split_offset + 1
                # Only accept if it keeps us making meaningful forward progress
                target = candidate if candidate >= start + MIN_STEP else target

        chunks.append(" ".join(words[start:target]))

        # Advance by (chunk size - overlap), but always move forward
        next_start = target - OVERLAP
        start = next_start if next_start > start else target

    # Absorb a trivially small tail chunk into the previous one
    MIN_CHUNK = OVERLAP * 3  # anything under ~450 words isn't worth a separate LLM call
    if len(chunks) > 1 and len(chunks[-1].split()) < MIN_CHUNK:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()

    return chunks or [transcript]


def _call_llm(
    chunk: str,
    subject: str,
    role: str,
    title: str,
    chunk_label: str,
    source: str,
    game: str,
) -> dict:
    # Escape curly braces in user content before .format() so guide text with
    # {Q}/{W} ability notation or JSON examples doesn't get parsed as format fields.
    spec = analysis_spec(game, source)
    safe_chunk = chunk.replace("{", "{{").replace("}", "}}")
    prompt = spec.extraction_prompt.format(
        subject=subject or "unknown",
        role=role or "unknown",
        title=title or "",
        chunk_label=chunk_label,
        transcript_chunk=safe_chunk,
    )
    system = spec.system_prompt
    if spec.reference_block:
        system += "\n\nREFERENCE LIST:\n" + spec.reference_block

    t0 = time.time()
    raw = llm_chat(system=system, user=prompt, temperature=0.1, max_tokens=4096)
    print(f"      [{BACKEND}] {time.time() - t0:.1f}s | {len(chunk.split()):,} words in", end=" ")

    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(0))
            except Exception:
                result = {}
        else:
            result = {}

    normalised_result = {key: result.get(key, []) for key in spec.insight_types}

    # Normalise items to (text, emphasis) tuples
    for key, items in normalised_result.items():
        if not isinstance(items, list):
            normalised_result[key] = []
            continue
        normalised = []
        for item in items:
            if isinstance(item, str) and item.strip():
                normalised.append((correct_names(item.strip()), 1))
            elif isinstance(item, dict) and item.get("text", "").strip():
                text = correct_names(item["text"].strip())
                emphasis = int(item.get("emphasis", 1))
                normalised.append((text, emphasis))
        normalised_result[key] = normalised

    return normalised_result


def _already_analyzed(video_id: str) -> bool:
    with get_connection() as conn:
        return conn.execute(
            "SELECT 1 FROM insights WHERE video_id = ? LIMIT 1", (video_id,)
        ).fetchone() is not None


def _get_pending_guide_videos(
    subject: str | None = None,
    source: str | None = None,
    game: str | None = None,
) -> list:
    with get_connection() as conn:
        source_clause = "source IN ('youtube_guide', 'mobafire_guide')"
        params: list = []
        if source:
            source_clause = "source = ?"
            params.append(source)
        game_clause = ""
        if game:
            game_clause = " AND game = ?"
            params.append(normalize_game(game))
        if subject:
            return conn.execute(
                """
                SELECT * FROM videos
                WHERE status = 'transcribed'
                  AND """ + source_clause + """
                  """ + game_clause + """
                  AND LOWER(COALESCE(subject, champion)) = LOWER(?)
                ORDER BY COALESCE(subject, champion), video_id
                """,
                params + [subject]
            ).fetchall()
        return conn.execute(
            """
            SELECT * FROM videos
            WHERE status = 'transcribed'
              AND """ + source_clause + """
              """ + game_clause + """
            ORDER BY game, COALESCE(subject, champion), video_id
            """,
            params,
        ).fetchall()


def analyze_video(video: dict, dry_run: bool = False) -> int:
    """Analyze one guide video. Returns number of insights saved (or found in dry_run)."""
    video_id = video["video_id"]
    game      = normalize_game(video["game"] or DEFAULT_GAME)
    subject   = video["subject"] or video["champion"] or "unknown"
    role      = video["role"] or "unknown"
    title     = video["video_title"] or ""
    source    = video["source"] or "youtube_guide"
    transcript = video["transcription"] or ""

    chunks = chunk_by_hour(transcript)
    total_words = len(transcript.split())
    print(f"  {total_words:,} words → {len(chunks)} hour-chunk(s)")

    aggregated: dict[str, list] = {}

    for i, chunk in enumerate(chunks):
        chunk_label = f"hour {i + 1} of {len(chunks)}"
        print(f"  Analyzing {chunk_label}…", end=" ", flush=True)
        result = _call_llm(chunk, subject, role, title, chunk_label, source, game)

        window_matrix = _embed_chunk_windows(chunk)
        n_this_chunk = 0
        for insight_type, items in result.items():
            aggregated.setdefault(insight_type, [])
            for text, emphasis in items:
                score = score_source_grounding(text, window_matrix)
                aggregated[insight_type].append((text, emphasis, score))
                n_this_chunk += 1
        print(f"→ {n_this_chunk} insights")

    total = 0
    for insight_type, items in aggregated.items():
        for text, emphasis, source_score in items:
            if not text.strip():
                continue
            if dry_run:
                print(f"    [{insight_type}] {text}")
            else:
                insert_insight(video_id, insight_type, text.strip(), source_score, repetition_count=emphasis)
            total += 1

    if not dry_run:
        set_status(video_id, "analyzed")

    return total


def run(
    subject: str | None = None,
    dry_run: bool = False,
    source: str | None = None,
    game: str | None = None,
) -> None:
    videos = _get_pending_guide_videos(subject, source=source, game=game)
    print(f"Guide entries to analyze: {len(videos)}")

    for video in videos:
        video_id = video["video_id"]
        subject_name = video["subject"] or video["champion"] or "?"
        game_name = normalize_game(video["game"] or DEFAULT_GAME)

        if _already_analyzed(video_id):
            print(f"[skip] {video_id} already analyzed")
            set_status(video_id, "analyzed")
            continue

        print(
            f"\n[{game_name}] [{video['source']}] "
            f"[{subject_name}] {video_id} | {(video['video_title'] or '')[:60]}"
        )
        try:
            n = analyze_video(dict(video), dry_run=dry_run)
            print(f"  → {n} insights {'(dry-run, not saved)' if dry_run else 'saved'}")
            delay = random.uniform(3.0, 5.0)
            print(f"  Waiting {delay:.1f}s…")
            time.sleep(delay)
        except Exception as e:
            print(f"  [error] {e}")
            if not dry_run:
                set_status(video_id, "error")


def _reset_for_reanalysis(subject: str | None, source: str | None, game: str | None) -> None:
    """Delete existing insights and reset status to transcribed so run() picks them up."""
    with get_connection() as conn:
        source_clause = "source IN ('youtube_guide', 'mobafire_guide')"
        params: list = []
        if source:
            source_clause = "source = ?"
            params.append(source)
        game_clause = ""
        if game:
            game_clause = " AND game = ?"
            params.append(normalize_game(game))
        if subject:
            video_ids = [r["video_id"] for r in conn.execute(
                "SELECT video_id FROM videos WHERE " + source_clause + game_clause + " AND LOWER(COALESCE(subject, champion)) = LOWER(?)",
                params + [subject]
            ).fetchall()]
        else:
            video_ids = [r["video_id"] for r in conn.execute(
                "SELECT video_id FROM videos WHERE " + source_clause + game_clause,
                params
            ).fetchall()]

        if not video_ids:
            return

        placeholders = ",".join("?" * len(video_ids))
        n_insights = conn.execute(
            f"DELETE FROM insights WHERE video_id IN ({placeholders})", video_ids
        ).rowcount
        conn.execute(
            f"UPDATE videos SET status='transcribed' WHERE video_id IN ({placeholders})",
            video_ids
        )
        conn.commit()
    label = subject or "all subjects"
    print(f"Reset {len(video_ids)} video(s) for {label} ({n_insights} insights deleted)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze guide-style entries in guide_test.db")
    parser.add_argument("--game", choices=sorted(SUPPORTED_GAMES), default=DEFAULT_GAME,
                        help="Game namespace to analyze")
    parser.add_argument("--subject", help="Only analyze videos for this subject (champion, civ, strategy)")
    parser.add_argument("--champion", help="Compatibility alias for --subject")
    parser.add_argument("--source", choices=["youtube_guide", "mobafire_guide"],
                        help="Limit to one guide source")
    parser.add_argument("--dry-run", action="store_true", help="Print insights, don't save")
    parser.add_argument("--reanalyze", action="store_true",
                        help="Delete existing insights and re-run (use with --champion or alone for all)")
    args = parser.parse_args()

    init_db()
    subject = args.subject or args.champion
    if args.reanalyze:
        _reset_for_reanalysis(subject, args.source, args.game)
    run(subject=subject, dry_run=args.dry_run, source=args.source, game=args.game)


if __name__ == "__main__":
    main()
