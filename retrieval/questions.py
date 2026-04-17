"""
Question normalization and routing layer.

Sits between the user and query.py. Takes a raw question, uses Gemma to:
  1. Rewrite it into the nearest canonical form the knowledge base answers best
  2. Extract filters (role, champion) automatically
  3. Route it to the most relevant insight_type(s)

This means vague questions like "how do i not feed?" get redirected to
well-formed questions like "How do I play safely when losing lane?" which
retrieve much more relevant insights.

Usage:
    python questions.py "how do i not feed in mid"
    python questions.py "cassio tips"
"""

import json
import os
import argparse
from core.llm import chat as llm_chat

# ---------------------------------------------------------------------------
# Canonical question templates
# Each entry: (template, insight_types, description)
# insight_types drives which DB categories get prioritised in retrieval
# ---------------------------------------------------------------------------
CANONICAL_QUESTIONS = [
    # Principles / mental models
    ("What is the core win condition for [champion]?",
     ["principles", "champion_mechanics"],
     "Champion win condition and game plan"),

    ("What is the general gameplan when playing a [archetype] champion in [role]?",
     ["principles", "macro_advice"],
     "Archetype-based gameplan (scaling, poke, all-in, etc.)"),

    ("How does [matchup type] beat [matchup type] in lane?",
     ["principles", "matchup_advice"],
     "Matchup archetype — why one type beats another"),

    # Laning
    ("How do I manage the wave when I am losing lane?",
     ["laning_tips", "matchup_advice", "principles"],
     "Wave management from behind"),

    ("How do I play safely in lane against [champion] or [champion type]?",
     ["laning_tips", "matchup_advice"],
     "Safe laning vs threatening opponents"),

    ("When should I freeze, slow push, or fast push the wave?",
     ["laning_tips", "principles"],
     "Wave management decision making"),

    ("How do I set up a kill or all-in in lane?",
     ["laning_tips", "champion_mechanics"],
     "Kill setup and trading patterns"),

    ("When should I back and how do I time my recalls?",
     ["laning_tips", "macro_advice"],
     "Recall timing and wave setup"),

    # Matchups
    ("How do I play [champion] against poke/mage champions?",
     ["matchup_advice", "laning_tips", "itemization"],
     "Playing vs poke/mage archetype"),

    ("How do I play [champion] against assassins?",
     ["matchup_advice", "laning_tips", "itemization"],
     "Playing vs assassin archetype"),

    ("How do I play [champion] against tanks or bruisers?",
     ["matchup_advice", "laning_tips"],
     "Playing vs tank/bruiser archetype"),

    ("What is [champion]'s specific gameplan against [champion]?",
     ["matchup_advice", "principles"],
     "Champion vs champion matchup gameplan — always name both champions"),

    # Macro
    ("When should I roam and when should I stay in lane?",
     ["macro_advice", "principles"],
     "Roam vs stay decision making"),

    ("How do I convert a lane lead into a game win?",
     ["macro_advice", "principles"],
     "Snowballing a lead"),

    ("What objectives should I prioritise after winning a fight?",
     ["macro_advice"],
     "Post-fight objective priority"),

    ("How do I play when my team is losing other lanes?",
     ["macro_advice", "principles"],
     "Playing with losing teammates"),

    # Champion mechanics
    ("How do I use [champion]'s abilities effectively in lane?",
     ["champion_mechanics", "laning_tips"],
     "Champion ability usage in lane"),

    ("What are [champion]'s power spikes and how do I play around them?",
     ["champion_mechanics", "principles"],
     "Power spike timing"),

    ("How do I play [champion] in teamfights?",
     ["teamfight_tips", "champion_mechanics"],
     "Champion teamfight role"),

    # Itemization
    ("What items should I buy on [champion] and when?",
     ["itemization", "principles"],
     "Item build and timing"),

    ("When should I take Teleport vs Ignite?",
     ["principles", "laning_tips", "matchup_advice"],
     "Summoner spell choice"),

    # Vision
    ("Where and when should I ward as a [role]?",
     ["vision_control", "macro_advice"],
     "Vision placement and timing"),

    # Mindset / general
    ("How do I play from behind when I am losing the game?",
     ["general_advice", "principles", "macro_advice"],
     "Playing from behind"),

    ("How do I identify and execute my win condition?",
     ["principles", "macro_advice"],
     "Win condition identification"),
]

# Flat list of template strings for the prompt
_TEMPLATE_LIST = "\n".join(
    f"{i + 1}. {t}" for i, (t, _, _) in enumerate(CANONICAL_QUESTIONS)
)

# ---------------------------------------------------------------------------
# Normalization prompt
# ---------------------------------------------------------------------------
NORMALIZE_SYSTEM = """
You are a League of Legends coaching assistant that helps players ask better questions.
Your job is to understand what a player is really asking and map it to the best form
for searching a coaching knowledge base.

You must return valid JSON only — no markdown, no explanation outside the JSON.
""".strip()

NORMALIZE_PROMPT = """
A player asked: "{question}"

Here are the canonical question templates this knowledge base answers best:
{templates}

Return a JSON object with these fields:
{{
    "normalized": "the best matching canonical template with ALL placeholders filled in — always include the player's champion name if mentioned, never drop it",
    "champion": "champion name if mentioned or implied, else null",
    "role": "one of: top, jungle, mid, adc, support — if mentioned or implied, else null",
    "insight_types": ["ordered list of 2-3 most relevant insight type keys from: principles, laning_tips, champion_mechanics, matchup_advice, macro_advice, teamfight_tips, vision_control, itemization, general_advice"],
    "reasoning": "one sentence explaining the mapping"
}}

Insight type keys to use:
- principles: mental models, archetypes, win conditions, the WHY behind tips
- laning_tips: specific laning phase actions
- champion_mechanics: abilities, combos, power spikes
- matchup_advice: how to play vs specific champions or champion types
- macro_advice: roaming, objectives, map decisions
- teamfight_tips: teamfight positioning and execution
- vision_control: ward placement and vision timing
- itemization: item builds and summoner spells
- general_advice: mindset, applies broadly
""".strip()


def normalize(question: str) -> dict:
    """
    Takes a raw player question and returns:
      - normalized: cleaner, canonical form
      - champion: extracted champion name or None
      - role: extracted role or None
      - insight_types: ordered list of relevant categories
      - reasoning: why it was mapped this way
    """
    prompt = NORMALIZE_PROMPT.format(
        question=question,
        templates=_TEMPLATE_LIST,
    )

    raw = llm_chat(system=NORMALIZE_SYSTEM, user=prompt, temperature=0.1)

    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return question as-is with no filters
        return {
            "normalized": question,
            "champion": None,
            "role": None,
            "insight_types": ["general_advice"],
            "reasoning": "Could not parse normalization response",
        }


def ask(question: str, top_k: int = 12, show_sources: bool = True) -> str:
    """
    Full pipeline: normalize question → extract filters → query knowledge base.
    Drop-in replacement for query.answer() with smarter routing.
    """
    from retrieval.query import answer

    print(f"Normalizing question…")
    parsed = normalize(question)

    print(f"  → {parsed['normalized']}")
    if parsed.get("reasoning"):
        print(f"  ({parsed['reasoning']})")
    print()

    # Role is used as a DB filter (narrows the pool to the right role).
    # Champion is NOT used as a DB filter — it would eliminate results when no
    # videos exist for that specific champion. Instead it's already baked into
    # the normalized question text, so semantic search surfaces it naturally.
    # insight_type filter is skipped too — too restrictive on small datasets.
    return answer(
        question=parsed["normalized"],
        role=parsed.get("role"),
        top_k=top_k,
        show_sources=show_sources,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the LoL coaching knowledge base")
    parser.add_argument("question", nargs="+", help="Your question in plain English")
    parser.add_argument("--raw", action="store_true",
                        help="Skip normalization, pass question directly to query.py")
    parser.add_argument("--normalize-only", action="store_true",
                        help="Show normalization result without querying")
    args = parser.parse_args()

    question = " ".join(args.question)
    print(f"\nQuestion: {question}\n")

    if args.normalize_only:
        result = normalize(question)
        import json as _json
        print(_json.dumps(result, indent=2))
        return

    if args.raw:
        from retrieval.query import answer
        print(answer(question, show_sources=True))
        return

    print(ask(question))


if __name__ == "__main__":
    main()
