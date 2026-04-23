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
import re
from core.llm import chat as llm_chat
from core.game_registry import (
    AOE2_CIVILIZATIONS,
    DEFAULT_GAME,
    canonical_aoe2_civilization,
    normalize_game,
)

# ---------------------------------------------------------------------------
# Canonical question templates
# Each entry: (template, insight_types, description)
# insight_types drives which DB categories get prioritised in retrieval
# ---------------------------------------------------------------------------
CANONICAL_QUESTIONS = [
    # Principles / mental models
    ("What is the core win condition for [champion]?",
     ["champion_identity", "principles", "champion_mechanics"],
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
     ["champion_matchups", "matchup_advice", "laning_tips"],
     "Playing vs poke/mage archetype"),

    ("How do I play [champion] against assassins?",
     ["champion_matchups", "matchup_advice", "laning_tips"],
     "Playing vs assassin archetype"),

    ("How do I play [champion] against tanks or bruisers?",
     ["champion_matchups", "matchup_advice", "laning_tips"],
     "Playing vs tank/bruiser archetype"),

    ("What is [champion]'s specific gameplan against [champion]?",
     ["champion_matchups", "matchup_advice", "principles"],
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

AOE2_CANONICAL_QUESTIONS = [
    ("How should I play [civilization] from opening through win condition?",
     ["civilization_identity", "build_orders", "dark_age", "feudal_age", "castle_age", "imperial_age", "economy_macro", "unit_compositions", "map_control"],
     "Civilization overview with opening and age-by-age game plan"),

    ("What is the core identity and win condition of [civilization]?",
     ["civilization_identity", "principles", "unit_compositions"],
     "Civilization identity and broad game plan"),

    ("What is a solid opening or build order for [civilization]?",
     ["build_orders", "dark_age", "feudal_age"],
     "Civilization opening and early structure"),

    ("How should I play the Dark Age and Feudal Age more cleanly?",
     ["dark_age", "feudal_age", "economy_macro"],
     "Early game execution"),

    ("How should I transition into Castle Age and Imperial Age?",
     ["castle_age", "imperial_age", "economy_macro"],
     "Mid and late age transitions"),

    ("How should I balance economy and production in this spot?",
     ["economy_macro", "build_orders", "principles"],
     "Economy and production scaling"),

    ("What should I scout for, and how should I adapt when I see it?",
     ["scouting", "principles", "matchup_advice"],
     "Scouting tells and response logic"),

    ("What unit composition should I aim for, and when should I transition?",
     ["unit_compositions", "castle_age", "imperial_age"],
     "Army composition and transitions"),

    ("How should I control the map and secure resources or relics?",
     ["map_control", "economy_macro", "principles"],
     "Map control and territory"),

    ("How should I micro my army and control fights more effectively?",
     ["micro", "unit_compositions", "principles"],
     "Army control and fight execution"),

    ("How should I defend against early aggression or pressure?",
     ["scouting", "economy_macro", "micro"],
     "Defense and stabilizing under pressure"),

    ("How should I execute an early attack or timing push?",
     ["build_orders", "feudal_age", "micro"],
     "Early aggression and attack execution"),

    ("How does [civilization] play into [civilization]?",
     ["matchup_advice", "civilization_identity", "unit_compositions"],
     "Civilization-versus-civilization adjustment"),

    ("What fundamental AoE2 habits should a beginner focus on first?",
     ["principles", "economy_macro", "scouting"],
     "Beginner fundamentals and clean execution"),
]

_AOE2_TEMPLATE_LIST = "\n".join(
    f"{i + 1}. {t}" for i, (t, _, _) in enumerate(AOE2_CANONICAL_QUESTIONS)
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
    "insight_types": ["ordered list of 2-3 most relevant insight type keys from: champion_identity, champion_matchups, principles, laning_tips, champion_mechanics, matchup_advice, macro_advice, teamfight_tips, vision_control, itemization, general_advice"],
    "reasoning": "one sentence explaining the mapping"
}}

Insight type keys to use:
- champion_identity: champion-specific win condition, role in fights, broad strategic identity
- champion_matchups: named champion-vs-champion matchup notes, especially from written guides
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

AOE2_NORMALIZE_SYSTEM = """
You are an Age of Empires II coaching assistant that helps players ask better questions.
Your job is to normalize a player's question for a strategy knowledge base and extract
the civilization filter when one is clearly present.

Return valid JSON only.
""".strip()

AOE2_NORMALIZE_PROMPT = """
A player asked: "{question}"

Known civilizations:
{civilizations}

Here are the canonical question templates this knowledge base answers best:
{templates}

Return a JSON object with these fields:
{{
    "normalized": "the best matching canonical template, with civilization names filled in when clearly present",
    "subject": "exact civilization name if one is clearly present, else null",
    "role": null,
    "insight_types": ["ordered list of 2-3 most relevant insight type keys from: civilization_identity, game_mechanics, controls_settings, micro, principles, build_orders, dark_age, feudal_age, castle_age, imperial_age, economy_macro, scouting, unit_compositions, map_control, matchup_advice, general_advice"],
    "reasoning": "one sentence explaining the mapping"
}}

Rules:
- Use "subject" only for a named civilization, not a strategy archetype.
- If the question is about general fundamentals, leave subject as null.
- For broad questions like "how should I play [civilization]" or "[civilization] guide",
  prefer the full opening-through-win-condition template, not identity-only.
- Prefer the most specific canonical phrasing that fits the player's intent.
- Use "controls_settings" for hotkeys, control groups, camera setup, UI usage,
  and command efficiency.
- Use "game_mechanics" for wildlife behavior, armor classes, counters, attack
  bonuses, conversion, projectile behavior, pathing quirks, and general unit or
  building interaction rules.
- Use "micro" for fight execution and unit control rather than settings/setup.
""".strip()


def _extract_aoe2_subject(question: str) -> str | None:
    q = question.lower()
    matches: list[tuple[int, str]] = []
    for civ in AOE2_CIVILIZATIONS:
        pos = q.find(civ.lower())
        if pos != -1:
            matches.append((pos, civ))
    if not matches:
        for token in question.replace("/", " ").replace("-", " ").split():
            civ = canonical_aoe2_civilization(token)
            if civ:
                return civ
        return None
    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def _is_aoe2_civ_overview_question(question: str, subject: str | None) -> bool:
    if not subject:
        return False
    return bool(
        re.search(
            r"\b(how (?:should|do) i play|how to play|playstyle|guide|gameplan)\b",
            question.lower(),
        )
    )


def _is_detail_question(question: str) -> bool:
    return bool(
        re.search(
            r"\b(detail|detailed|in[- ]?depth|step by step|full guide|explain more)\b",
            question.lower(),
        )
    )


def normalize(question: str, game: str = DEFAULT_GAME) -> dict:
    """
    Takes a raw player question and returns:
      - normalized: cleaner, canonical form
      - champion: extracted champion name or None
      - role: extracted role or None
      - insight_types: ordered list of relevant categories
      - reasoning: why it was mapped this way
    """
    game = normalize_game(game)

    if game == "aoe2":
        detected_subject = _extract_aoe2_subject(question)
        prompt = AOE2_NORMALIZE_PROMPT.format(
            question=question,
            civilizations=", ".join(AOE2_CIVILIZATIONS),
            templates=_AOE2_TEMPLATE_LIST,
        )
        raw = llm_chat(system=AOE2_NORMALIZE_SYSTEM, user=prompt, temperature=0.1)
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        try:
            data = json.loads(raw)
            if data.get("subject"):
                data["subject"] = canonical_aoe2_civilization(data["subject"]) or detected_subject
            else:
                data["subject"] = detected_subject
            if _is_aoe2_civ_overview_question(question, data.get("subject")):
                detail_suffix = " in detail" if _is_detail_question(question) else ""
                data["normalized"] = (
                    f"How should I play {data['subject']} from opening through win condition{detail_suffix}?"
                )
                data["insight_types"] = [
                    "civilization_identity",
                    "build_orders",
                    "dark_age",
                    "feudal_age",
                    "castle_age",
                    "imperial_age",
                    "economy_macro",
                    "unit_compositions",
                    "map_control",
                ]
            data.setdefault("role", None)
            data.setdefault("insight_types", ["general_advice"])
            data.setdefault("reasoning", "Mapped via AoE2 normalization prompt")
            data.setdefault("normalized", question)
            return data
        except json.JSONDecodeError:
            return {
                "normalized": question,
                "subject": detected_subject,
                "role": None,
                "insight_types": ["general_advice"],
                "reasoning": "Could not parse AoE2 normalization response",
            }

    if game != "lol":
        return {
            "normalized": question,
            "champion": None,
            "role": None,
            "insight_types": ["general_advice"],
            "reasoning": f"{game} currently uses raw question pass-through normalization",
        }

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


def ask(question: str, top_k: int = 12, show_sources: bool = True, game: str = DEFAULT_GAME) -> str:
    """
    Full pipeline: normalize question → extract filters → query knowledge base.
    Drop-in replacement for query.answer() with smarter routing.
    """
    from retrieval.query import answer

    print(f"Normalizing question…")
    parsed = normalize(question, game=game)

    print(f"  → {parsed['normalized']}")
    if parsed.get("reasoning"):
        print(f"  ({parsed['reasoning']})")
    print()

    # Role is used as a DB filter (narrows the pool to the right role).
    # Champion is NOT used as a DB filter — it would eliminate results when no
    # videos exist for that specific champion. Instead it's already baked into
    # the normalized question text, so semantic search surfaces it naturally.
    # insight_type filter is skipped too — too restrictive on small datasets.
    subject = parsed.get("subject") if normalize_game(game) == "aoe2" else None
    return answer(
        question=parsed["normalized"],
        role=parsed.get("role"),
        subject=subject,
        game=game,
        top_k=top_k,
        show_sources=show_sources,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the LoL coaching knowledge base")
    parser.add_argument("question", nargs="+", help="Your question in plain English")
    parser.add_argument("--game", default=DEFAULT_GAME, help="Game namespace (lol, aoe2)")
    parser.add_argument("--raw", action="store_true",
                        help="Skip normalization, pass question directly to query.py")
    parser.add_argument("--normalize-only", action="store_true",
                        help="Show normalization result without querying")
    args = parser.parse_args()

    question = " ".join(args.question)
    print(f"\nQuestion: {question}\n")
    game = normalize_game(args.game)

    if args.normalize_only:
        result = normalize(question, game=game)
        import json as _json
        print(_json.dumps(result, indent=2))
        return

    if args.raw:
        from retrieval.query import answer
        print(answer(question, game=game, show_sources=True))
        return

    print(ask(question, game=game))


if __name__ == "__main__":
    main()
