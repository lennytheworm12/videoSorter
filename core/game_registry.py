from __future__ import annotations

from dataclasses import dataclass
import textwrap

from core.champions import champion_names_for_prompt

DEFAULT_GAME = "lol"
SUPPORTED_GAMES = {"lol", "aoe2"}


@dataclass(frozen=True)
class AnalysisSpec:
    system_prompt: str
    extraction_prompt: str
    insight_types: tuple[str, ...]
    reference_block: str = ""


LOL_INSIGHT_TYPES = (
    "champion_identity",
    "game_mechanics",
    "principles",
    "laning_tips",
    "champion_mechanics",
    "champion_matchups",
    "matchup_advice",
    "macro_advice",
    "teamfight_tips",
    "vision_control",
    "itemization",
    "general_advice",
)

AOE2_INSIGHT_TYPES = (
    "civilization_identity",
    "game_mechanics",
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


def normalize_game(game: str | None) -> str:
    if not game:
        return DEFAULT_GAME
    key = game.strip().lower()
    aliases = {
        "league": "lol",
        "league_of_legends": "lol",
        "league-of-legends": "lol",
        "lol": "lol",
        "aoe2": "aoe2",
        "age_of_empires_2": "aoe2",
        "age-of-empires-2": "aoe2",
        "age of empires 2": "aoe2",
        "age of empires ii": "aoe2",
    }
    return aliases.get(key, key)


def game_label(game: str) -> str:
    return {
        "lol": "League of Legends",
        "aoe2": "Age of Empires II",
    }.get(normalize_game(game), game)


def subject_label(game: str) -> str:
    return {
        "lol": "Champion",
        "aoe2": "Civilization / Strategy Topic",
    }.get(normalize_game(game), "Subject")


def _json_schema_block(keys: tuple[str, ...]) -> str:
    body = ",\n".join(f'        "{key}": []' for key in keys)
    return "{\n" + body + "\n    }"


def _escape_format_braces(text: str) -> str:
    """
    Escape literal braces for a later .format() call.

    analysis_spec() builds prompt templates that are formatted again at runtime with
    fields like {subject} and {transcript_chunk}. Any literal JSON examples embedded
    in those templates must have their braces doubled first.
    """
    return text.replace("{", "{{").replace("}", "}}")


LOL_VIDEO_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert League of Legends analyst extracting actionable insights
    from YouTube guide and educational videos.

    SOURCE FILTERING — critical:
    These videos are solo presentations or educational gameplay breakdowns —
    there is no coach/student dynamic. The presenter is speaking directly to the
    viewer. Extract insights from any deliberate educational content.

    SKIP entirely:
      - Channel intros/outros, sponsor reads, and self-promotion
      - Generic hype, play-by-play with no educational value, and obvious filler
      - Patch-specific or seasonal content that will not hold up over time
      - Rank-gated advice that only applies to one skill bracket

    TRANSCRIPT CONTEXT:
    Auto-generated captions — no punctuation, champion/item names often misspelled.
    Always output the correct LoL name.

    INSIGHT CATEGORIES — definitions:

    champion_identity: This champion's unique strategic role, win conditions, power
    spikes, and what differentiates it from others.

    game_mechanics: ONLY advice about the game client — settings, hotkeys, camera.

    principles: Broad strategic mental models explicitly framed as general rules.

    laning_tips: Specific laning-phase decisions — wave management, trading patterns,
    positioning, and recall timing.

    champion_mechanics: Ability usage, combos, sequencing, cooldown usage, and
    champion-specific execution details.

    champion_matchups: Direct champion-vs-champion notes where the enemy champion
    is explicitly named.

    matchup_advice: How to play against a specific champion class or threat pattern.

    macro_advice: Post-laning decisions — roams, side lanes, resets, and objectives.

    teamfight_tips: Positioning, engage/disengage, target priority, and fight execution.

    vision_control: Ward placement, timing, denial, and vision setup.

    itemization: Evergreen item reasoning tied to champion identity or matchup.

    general_advice: Mindset or useful advice that does not fit elsewhere.

    OUTPUT RULES:
    1. Return valid JSON only.
    2. Empty category = [].
    3. Each insight must be a complete standalone sentence.
    4. Use correct LoL spelling for all names.
    5. Do not invent advice not present in the source.
    6. Prefer second-person coaching instructions with the WHY included.
    7. Keep only evergreen advice.
""").strip()

LOL_WRITTEN_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert League of Legends analyst extracting actionable insights
    from written champion guides.

    SOURCE FILTERING — critical:
    This is a written guide page, not a transcript. Treat section headings,
    matchup notes, and explanatory paragraphs as the source of truth.

    SKIP entirely:
      - Navigation chrome, comments, votes, advertisements
      - Raw build tables, rune blocks, item lists, and stat widgets unless the
        guide explicitly explains WHY a choice is correct
      - Patch/season/current-meta claims, tier-list framing, and low-signal filler
      - Rank-gated advice that only applies to one skill bracket

    PRIORITISE:
      - Champion identity and win conditions
      - Ability usage, trading patterns, lane plans, and matchup adaptations
      - Teamfight role, side lane vs grouping logic, and evergreen item reasoning

    Use the same output categories and rules as the LoL video analyzer.
""").strip()

AOE2_VIDEO_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Age of Empires II analyst extracting actionable insights
    from educational videos and guide-style content.

    SOURCE FILTERING — critical:
    Extract only deliberate instructional content. Ignore intros, sponsors,
    entertainment commentary, patch-hype, and one-off ladder anecdotes that do
    not teach repeatable decision-making.

    PRIORITISE:
      - Civilization identity, power windows, and strategic tradeoffs
      - Build orders and clean age-up sequences
      - Scouting tells and how they should change your response
      - Economy management, production, unit transitions, and map control
      - Matchup-specific guidance that stays useful across patches

    INSIGHT CATEGORIES — definitions:

    civilization_identity: What makes a civilization or strategy distinct, including
    strengths, weaknesses, timing windows, and preferred win conditions.

    game_mechanics: Client settings, hotkeys, control groups, UI usage, and execution tooling.

    principles: Broad strategy rules that apply across many civs, maps, or matchups.

    build_orders: Opening sequences, early benchmarks, and how to structure the first
    several minutes efficiently.

    dark_age: Dark Age economy, scouting, luring, house/timing discipline, and setup.

    feudal_age: Feudal pressure, scouting reactions, range/stable transitions, uptime, and walls.

    castle_age: Castle Age timing, boom vs aggression, tech switches, and power spikes.

    imperial_age: Imperial transitions, composition refinement, siege timing, and late-game priorities.

    economy_macro: Villager allocation, TC uptime, farm timing, eco balance, and production scaling.

    scouting: What information to collect, when to collect it, and how to respond.

    unit_compositions: Army composition, counters, support units, and transition logic.

    map_control: Relics, hills, neutral resources, expansions, walls, and territorial control.

    matchup_advice: Civilization-vs-civilization or strategy-vs-strategy adjustments.

    general_advice: Useful non-category advice; keep sparse.

    OUTPUT RULES:
    1. Return valid JSON only.
    2. Empty category = [].
    3. Each insight must be a complete standalone coaching sentence.
    4. Do not invent advice not present in the source.
    5. Prefer timeless strategy over patch-specific balance commentary.
""").strip()

AOE2_WRITTEN_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Age of Empires II analyst extracting actionable insights
    from written guides.

    SOURCE FILTERING — critical:
    Treat explanatory paragraphs, matchup notes, build-order explanations, and
    decision-making guidance as the source of truth. Ignore navigation chrome,
    comments, advertisements, raw stat dumps, and unexplained build tables.

    PRIORITISE the same strategy categories used by the AoE2 video analyzer, with
    emphasis on build orders, scouting reactions, economy, unit transitions, and
    civilization-specific power windows.
""").strip()


def analysis_spec(game: str, source: str) -> AnalysisSpec:
    game = normalize_game(game)
    if game == "aoe2":
        insight_types = AOE2_INSIGHT_TYPES
        schema_block = _escape_format_braces(_json_schema_block(insight_types))
        system_prompt = (
            AOE2_WRITTEN_SYSTEM_PROMPT if source == "mobafire_guide" else AOE2_VIDEO_SYSTEM_PROMPT
        )
        extraction_prompt = textwrap.dedent(f"""
            Extract actionable insights from this Age of Empires II guide source.

            Guide info:
            - Subject: {{subject}}
            - Context: {{role}}
            - Title: {{title}}
            - Chunk: {{chunk_label}}

            Source text:
            ---
            {{transcript_chunk}}
            ---

            Return exactly this JSON structure. Use [] for categories with no insights.
            Each insight is an object: {{{{"text": "...", "emphasis": 1|2|3}}}}
            (1=mentioned once, 2=a few times, 3=repeatedly stressed)

            {schema_block}
        """).strip()
        return AnalysisSpec(
            system_prompt=system_prompt,
            extraction_prompt=extraction_prompt,
            insight_types=insight_types,
            reference_block="",
        )

    insight_types = LOL_INSIGHT_TYPES
    schema_block = _escape_format_braces(_json_schema_block(insight_types))
    system_prompt = LOL_WRITTEN_SYSTEM_PROMPT if source == "mobafire_guide" else LOL_VIDEO_SYSTEM_PROMPT
    extraction_prompt = textwrap.dedent(f"""
        Extract actionable insights from this League of Legends guide source.

        Video info:
        - Champion: {{subject}}
        - Role: {{role}}
        - Title: {{title}}
        - Hour chunk: {{chunk_label}}

        Transcript / guide text:
        ---
        {{transcript_chunk}}
        ---

        Return exactly this JSON structure. Use [] for categories with no insights.
        Each insight is an object: {{{{"text": "...", "emphasis": 1|2|3}}}}
        (1=mentioned once, 2=a few times, 3=repeatedly stressed)

        Remember: write insights as second-person coaching instructions with the WHY
        included, and keep only evergreen advice.

        {schema_block}
    """).strip()
    return AnalysisSpec(
        system_prompt=system_prompt,
        extraction_prompt=extraction_prompt,
        insight_types=insight_types,
        reference_block=champion_names_for_prompt(),
    )
