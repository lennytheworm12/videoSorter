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
from dotenv import load_dotenv
import ollama
from database import get_videos_by_status, set_status, insert_insight, get_connection
from champions import correct_names, champion_names_for_prompt

load_dotenv()

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# ~4 chars per token. Keep chunks at 32K chars (~8K tokens) so the full prompt
# stays well within Gemma 4 e2b's 16K context window after adding prompt overhead.
CHUNK_CHARS = 32_000

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert League of Legends (LoL) coach analyzing transcripts from coaching
    sessions and gameplay educational videos.

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

    WHAT COUNTS AS A GOOD INSIGHT:
    Good — specific, actionable, contains a clear condition and action:
      "Buy QSS before your second item when facing Malzahar — his suppression removes
       you from the fight and you cannot react without it"
      "When ahead on Cassiopeia, slow push the wave before backing so it crashes under
       their tower and denies your opponent 15-20 CS while you shop"
      "Take Teleport over Ignite against poke mages like Orianna or Syndra — you need
       the ability to recover from losing trades and impact side lanes"

    Also extract FIRST-PRINCIPLES reasoning — the underlying logic behind WHY tips work.
    These are mental models and matchup archetypes, not just actions:
      "Poke mages beat all-in champions by winning the attrition game — your entire
       gameplan in this archetype is survival to a powerspike, which changes your wave
       positioning, recall timing, and trade patterns throughout laning phase"
      "When you are the scaling champion in a losing lane matchup, the correct mental
       model is to think in terms of waves: every decision should be about minimising
       CS loss and denying the enemy a free recall rather than trying to win trades"
      "Understanding when to freeze vs slow push comes down to who has priority —
       if you have priority you push and roam, if you don't you freeze to negate ganks"
    These belong in the 'principles' category.

    Bad — ignore these entirely:
      Vague: "you should ward more", "play safer", "trade better"
      Play-by-play commentary: "ok so here he walks up", "yeah he misses that CS"
      Meta-commentary: "that was a good play", "I can see you've improved"
      Unanswered student questions with no coaching response

    OUTPUT RULES — follow exactly:
    1. Return valid JSON only. No markdown fences, no text outside the JSON object.
    2. Empty category = [] — NEVER write a string like "no insights found" or
       "the transcript does not contain...". An empty list [] is the only valid empty value.
    3. Each insight must be a complete standalone sentence. Someone reading it without
       watching the video must be able to understand and apply it immediately.
    4. Always use correct LoL spelling for champion names, item names, and game terms.
    5. Do not invent advice that is not explicitly stated in the transcript.
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

    'principles' = the underlying WHY and mental models (matchup archetypes, wave
    state logic, win conditions, game phase reasoning). These explain the logic
    behind multiple tips and are the most valuable insights to capture.

    {{
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


def chunk_transcript(transcript: str) -> list[str]:
    """Split a transcript into chunks that fit comfortably in the LLM context."""
    chunks = []
    for i in range(0, len(transcript), CHUNK_CHARS):
        chunks.append(transcript[i : i + CHUNK_CHARS])
    return chunks


def extract_insights_from_chunk(
    chunk: str,
    role: str,
    champion: str | None,
    description: str | None,
    model: str,
) -> dict[str, list[str]]:
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
            "num_ctx": 16384,     # context window — 1hr transcript fits comfortably
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
        result = json.loads(raw)
    except json.JSONDecodeError:
        print(f"    [warn] JSON parse failed, raw output:\n{raw[:300]}")
        return {}

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

            for insight_type, items in result.items():
                if insight_type not in aggregated:
                    aggregated[insight_type] = []
                aggregated[insight_type].extend(items)

        # Persist to DB
        total = 0
        for insight_type, items in aggregated.items():
            for item in items:
                if item.strip():
                    insert_insight(video_id, insight_type, item.strip())
                    total += 1

        set_status(video_id, "analyzed")
        print(f"  Saved {total} insights")


if __name__ == "__main__":
    run()
