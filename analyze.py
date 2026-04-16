"""
Runs local LLM analysis on transcribed videos to extract structured LoL insights.

Requires Ollama running locally:
    ollama serve
    ollama pull llama3.1:8b

Run after transcribing:
    python analyze.py

Safe to re-run — already-analyzed videos are skipped.
"""

import json
import os
import textwrap
import numpy as np
from dotenv import load_dotenv
import ollama
from sentence_transformers import SentenceTransformer
from database import get_videos_by_status, set_status, insert_insight, get_connection
from champions import correct_names, champion_names_for_prompt

load_dotenv()

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Window size (words) and stride for sliding-window source grounding
_WINDOW_WORDS = 40
_WINDOW_STRIDE = 20

# ~4 chars per token. Keep chunks at 20K chars (~5K tokens) so the full prompt
# (system + champion list + extraction prompt + chunk) stays within Gemma's
# 16K context window and leaves ~6K tokens for the JSON output.
CHUNK_CHARS = 10_000

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert League of Legends (LoL) coach analyzing transcripts from coaching
    sessions and gameplay educational videos.

    SOURCE FILTERING — critical:
    These transcripts are from coaching sessions with a coach and a student (client).
    ONLY extract insights from the COACH's explanations and advice.
    IGNORE the student entirely:
      - Student gameplay narration: "ok I'm going to push here", "should I go in?"
      - Student questions that the coach doesn't answer in this excerpt
      - Student reactions: "oh ok", "yeah I see", "that makes sense"
    The coach is the one explaining WHY — the student is the one playing and asking.
    If you cannot tell who is speaking, only extract the statement if it reads as
    deliberate coaching advice (explaining a concept, giving a reason, correcting a mistake).

    TRANSCRIPT CONTEXT:
    These are auto-generated captions — no punctuation, and champion/item names are often
    misspelled by the speech-to-text system. Common errors:
      "Malahar", "malazar", "malasar" = Malzahar
      "Scarner", "scar", "skarn" = Skarner
      "cass" or "cassio" = Cassiopeia
      "vlad" = Vladimir
      "QSS" = Quicksilver Sash (item)
      "TP" = Teleport (summoner spell)
    Correct these in your output — always use the proper LoL name.

    INSIGHT CATEGORIES — read all definitions carefully before categorizing:

    champion_identity: The strategic role and win condition of the SPECIFIC CHAMPION
    being coached — not general LoL advice. Must name or clearly describe this champion's
    unique game plan: what it is trying to do, when it is strong/weak, what winning
    looks like for it specifically. Ask yourself: "would this apply equally to any other
    champion?" If yes, it does not belong here.
      IMPORTANT: only extract statements explicitly made about this champion in the
      transcript — do not infer win conditions from general LoL knowledge.

    game_mechanics: ONLY advice about the game CLIENT ITSELF — keybindings, settings,
    cursor behaviour, camera configuration, or mouse/input hardware technique.
    The test: if you stripped out all champion names and game context, would this tip
    still make complete sense as standalone PC/client advice? If yes → game_mechanics.
    If no → it belongs somewhere else.
      YES: "Increase your camera move speed in settings so you can pan faster"
      YES: "Click close to your character rather than far away for finer cursor control"
      NO: wave management, Teleport decisions, trading, warding, rotations — those are
          in-game decisions, NOT client settings, regardless of how mechanical they sound.
      This category is almost always empty — [] is correct for most videos.

    principles: Strategic mental models and the underlying WHY behind decisions.
    The coach is explaining LoL logic that applies broadly — wave state theory, matchup
    archetypes, resource trading, macro timing. A tip that also appears in laning_tips
    may belong here too if the coach frames it as a general rule, not just a situational cue.

    laning_tips: Specific actionable decisions during laning phase — wave management,
    trading patterns, positioning, recall timing. Champion-context is fine here.
    Overlap with principles is expected and acceptable: a wave management rule can be
    both a laning_tip (applied here) and a principle (the underlying logic).

    champion_mechanics: How to use THIS champion's abilities — combos, power spike
    windows, ability sequencing, E/Q/R usage patterns, cooldown management.

    matchup_advice: How to play against a specific champion or champion archetype.
    Must include both the condition (what the enemy does) and the required adjustment.

    macro_advice: Post-laning decisions — when to roam, objective priority, side lane
    management, Teleport usage, team coordination, win condition execution mid/late.
    Teleport decisions belong here, NOT in game_mechanics.

    teamfight_tips: Positioning, target selection, engage/disengage decisions, ability
    usage within a team fight or skirmish.

    vision_control: Ward placement, when to ward, how to contest enemy vision.
    Statements about map awareness or minimap habits belong here only if they are
    specifically about vision — not general awareness advice.

    itemization: Item choices, build order, and summoner spell selection with reasoning.

    general_advice: Mindset, mental approach, and broadly applicable advice that does
    not fit any specific category above. Keep this sparse — most advice belongs
    somewhere more specific.

    WHAT TO IGNORE ENTIRELY:
      Vague: "you should ward more", "play safer", "trade better"
      Play-by-play: "ok so here he walks up", "yeah he misses that CS"
      Meta-commentary: "that was a good play", "I can see you've improved"
      Unanswered student questions with no coaching response

    OUTPUT RULES — follow exactly:
    1. Return valid JSON only. No markdown fences, no text outside the JSON object.
    2. Empty category = [] — NEVER write a string like "no insights found".
       An empty list [] is the only valid empty value.
    3. Each insight must be a complete standalone sentence. Someone reading it without
       watching the video must understand and apply it immediately.
    4. Always use correct LoL spelling for champion names, item names, and game terms.
    5. Do not invent advice not explicitly stated in the transcript.
    6. Be selective — skip vague, redundant, or student-narration statements.
       Prefer depth over breadth: one well-explained insight beats three vague ones.
""").strip()


EXTRACTION_PROMPT = textwrap.dedent("""
    Extract actionable coaching insights from this League of Legends transcript excerpt.

    Video info:
    - Role: {role}
    - Champion: {champion}
    - Description: {description}

    Transcript (auto-generated captions, may contain name spelling errors):
    ---
    {transcript_chunk}
    ---

    Return exactly this JSON structure. Use [] for any category with no insights found.
    No text before or after the JSON.

    {{
        "champion_identity": [],
        "game_mechanics": [],
        "principles": [],
        "laning_tips": [],
        "champion_mechanics": [],
        "matchup_advice": [],
        "macro_advice": [],
        "teamfight_tips": [],
        "vision_control": [],
        "itemization": [],
        "general_advice": []
    }}
""").strip()


_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _embed_chunk_windows(chunk: str) -> np.ndarray:
    """
    Split a transcript chunk into overlapping word windows and embed them.
    Returns a (n_windows, 384) normalised float32 matrix.
    Used once per chunk so all insights in that chunk can be scored cheaply.
    """
    words = chunk.split()
    windows = [
        " ".join(words[i : i + _WINDOW_WORDS])
        for i in range(0, max(1, len(words) - _WINDOW_WORDS + 1), _WINDOW_STRIDE)
    ]
    if not windows:
        windows = [chunk[:500]]
    model = _get_embed_model()
    return model.encode(windows, convert_to_numpy=True, normalize_embeddings=True)


def score_source_grounding(insight_text: str, window_matrix: np.ndarray) -> float:
    """
    Return the max cosine similarity between an insight and any window of its
    source transcript chunk.  Pre-normalised vectors → dot product = cosine sim.

    Score interpretation:
      > 0.50  likely grounded in the transcript
      0.30-0.50  uncertain
      < 0.30  likely hallucinated from model priors
    """
    model = _get_embed_model()
    vec = model.encode(insight_text, convert_to_numpy=True, normalize_embeddings=True)
    return float((window_matrix @ vec).max())


def chunk_transcript(transcript: str) -> list[str]:
    """
    Split a transcript into chunks up to CHUNK_CHARS each, always breaking at
    a sentence boundary (. ! ?) when one exists near the target size.
    Falls back to word boundary if no punctuation found (common in raw captions).

    A 200-char lookahead window is searched before the hard limit, so chunks
    can be slightly shorter than CHUNK_CHARS but never cut mid-sentence.
    """
    SENTENCE_ENDINGS = {'.', '!', '?'}
    LOOKAHEAD = 200  # chars before the hard limit to search for a sentence end

    chunks = []
    start = 0
    n = len(transcript)

    while start < n:
        end = min(start + CHUNK_CHARS, n)

        if end < n:
            # Search backwards from end for a sentence boundary
            window_start = max(start, end - LOOKAHEAD)
            split_at = None
            for i in range(end - 1, window_start - 1, -1):
                if transcript[i] in SENTENCE_ENDINGS:
                    split_at = i + 1  # include the punctuation in this chunk
                    break

            # No sentence boundary found — fall back to last word boundary
            if split_at is None:
                space = transcript.rfind(' ', window_start, end)
                split_at = space + 1 if space != -1 else end

            end = split_at

        chunks.append(transcript[start:end].strip())
        start = end

    return [c for c in chunks if c]


def _call_ollama(chunk: str, role: str, champion: str | None, description: str | None, model: str) -> dict[str, list[str]]:
    """Single LLM call for one chunk. Returns parsed dict or raises ValueError on bad JSON."""
    prompt = EXTRACTION_PROMPT.format(
        role=role,
        champion=champion or "unknown",
        description=description or "no description",
        transcript_chunk=chunk,
    )

    # Append full champion list to system prompt so the model knows every name
    system = SYSTEM_PROMPT + "\n\nFULL CHAMPION LIST (use exact spelling from this list):\n" + champion_names_for_prompt()

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0.1,  # low temp = more consistent structured output
            "num_ctx": 16384,     # context window size
            "num_predict": 4096,  # max output tokens — enough for ~80 insights
        },
    )

    # Support both dict and object style (ollama SDK changed between versions)
    msg = response["message"] if isinstance(response, dict) else response.message
    raw = (msg["content"] if isinstance(msg, dict) else msg.content).strip()

    # Strip markdown code fences if the model wraps output despite instructions
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse failed: {raw[:200]}") from exc


def _merge_results(a: dict, b: dict) -> dict:
    """Merge two insight dicts by extending each category's list."""
    merged = dict(a)
    for key, items in b.items():
        if key in merged:
            merged[key] = merged[key] + (items if isinstance(items, list) else [])
        else:
            merged[key] = items if isinstance(items, list) else []
    return merged


def extract_insights_from_chunk(
    chunk: str,
    role: str,
    champion: str | None,
    description: str | None,
    model: str,
    _depth: int = 0,
) -> dict[str, list[str]]:
    """
    Extract insights from a transcript chunk.

    If the model's JSON output is truncated (parse failure), the chunk is split
    in half and each half is retried independently — up to 2 levels deep (quartering
    the original chunk). This handles dense chunks without shrinking all chunks globally.
    """
    MAX_DEPTH = 2
    MIN_CHARS = 2_000  # don't split below this; just skip

    try:
        result = _call_ollama(chunk, role, champion, description, model)
    except ValueError as exc:
        if _depth >= MAX_DEPTH or len(chunk) < MIN_CHARS:
            print(f"    [warn] {exc} (chunk too small to split further, skipping)")
            return {}

        mid = len(chunk) // 2
        # Split at nearest word boundary
        split_at = chunk.rfind(" ", mid - 200, mid + 200)
        if split_at == -1:
            split_at = mid

        half_a, half_b = chunk[:split_at], chunk[split_at:]
        print(f"    [retry] splitting chunk in half ({len(half_a):,} + {len(half_b):,} chars, depth={_depth + 1})")

        result_a = extract_insights_from_chunk(half_a, role, champion, description, model, _depth + 1)
        result_b = extract_insights_from_chunk(half_b, role, champion, description, model, _depth + 1)
        result = _merge_results(result_a, result_b)

    # Post-process: correct champion name misspellings in every insight string
    for key, items in result.items():
        if isinstance(items, list):
            result[key] = [
                correct_names(item) if isinstance(item, str) else item
                for item in items
            ]
    return result


def already_analyzed(video_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM insights WHERE video_id = ? LIMIT 1", (video_id,)
        ).fetchone()
    return row is not None


def run() -> None:
    videos = get_videos_by_status("transcribed")
    print(f"Videos to analyze: {len(videos)}")

    for video in videos:
        video_id = video["video_id"]
        role = video["role"]
        champion = video["champion"]
        description = video["description"]
        transcript = video["transcription"]

        if already_analyzed(video_id):
            print(f"[skip] {video_id} already analyzed")
            set_status(video_id, "analyzed")
            continue

        print(f"\n[{role}] {video_id} | champion={champion or '?'}")
        chunks = chunk_transcript(transcript)
        print(f"  {len(transcript.split())} words → {len(chunks)} chunk(s)")

        # Aggregate insights across all chunks
        aggregated: dict[str, list[str]] = {}

        for i, chunk in enumerate(chunks):
            print(f"  Analyzing chunk {i + 1}/{len(chunks)}…")
            try:
                result = extract_insights_from_chunk(chunk, role, champion, description, OLLAMA_MODEL)
            except Exception as e:
                print(f"  [error] chunk {i + 1}: {e}")
                continue

            # Pre-embed this chunk's windows once; reuse for all insights
            window_matrix = _embed_chunk_windows(chunk)

            for insight_type, items in result.items():
                if insight_type not in aggregated:
                    aggregated[insight_type] = []
                for item in items:
                    if isinstance(item, str):
                        score = score_source_grounding(item, window_matrix)
                        aggregated[insight_type].append((item, score))

        # Persist to DB
        total = 0
        flagged = 0
        for insight_type, items in aggregated.items():
            for item, source_score in items:
                if item.strip():
                    insert_insight(video_id, insight_type, item.strip(), source_score)
                    total += 1
                    if source_score < 0.30:
                        flagged += 1

        set_status(video_id, "analyzed")
        print(f"  Saved {total} insights ({flagged} low source-score, possible hallucinations)")


if __name__ == "__main__":
    run()
