from __future__ import annotations

from dataclasses import dataclass
import re

from core.champions import champion_names_for_prompt
from prompts.aoe2 import (
    AOE2_GUIDE_COACHING_SYSTEM_PROMPT,
    AOE2_GUIDE_VIDEO_SYSTEM_PROMPT,
    AOE2_GUIDE_WRITTEN_SYSTEM_PROMPT,
    build_aoe2_guide_extraction_prompt,
)
from prompts.lol import (
    LOL_GUIDE_VIDEO_SYSTEM_PROMPT,
    LOL_GUIDE_WRITTEN_SYSTEM_PROMPT,
    build_lol_guide_extraction_prompt,
)

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
    "ability_windows",
    "champion_mechanics",
    "champion_matchups",
    "matchup_advice",
    "macro_advice",
    "teamfight_tips",
    "vision_control",
    "itemization",
    "general_advice",
)

LOL_WRITTEN_INSIGHT_TYPES = (
    "champion_identity",
    "principles",
    "laning_tips",
    "ability_windows",
    "champion_mechanics",
    "champion_matchups",
    "matchup_advice",
    "macro_advice",
)

AOE2_INSIGHT_TYPES = (
    "civilization_identity",
    "game_mechanics",
    "controls_settings",
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

AOE2_CIVILIZATIONS = (
    "Achaemenids",
    "Armenians",
    "Athenians",
    "Aztecs",
    "Bengalis",
    "Berbers",
    "Bohemians",
    "Britons",
    "Bulgarians",
    "Burgundians",
    "Burmese",
    "Byzantines",
    "Celts",
    "Chinese",
    "Cumans",
    "Dravidians",
    "Ethiopians",
    "Franks",
    "Georgians",
    "Goths",
    "Gurjaras",
    "Hindustanis",
    "Huns",
    "Incas",
    "Italians",
    "Japanese",
    "Jurchens",
    "Khitans",
    "Khmer",
    "Koreans",
    "Lac Viet",
    "Lithuanians",
    "Magyars",
    "Malay",
    "Malians",
    "Mayans",
    "Mapuche",
    "Mongols",
    "Muisca",
    "Persians",
    "Poles",
    "Portuguese",
    "Romans",
    "Saracens",
    "Shu",
    "Sicilians",
    "Slavs",
    "Spanish",
    "Spartans",
    "Tatars",
    "Teutons",
    "Turks",
    "Tupi",
    "Vietnamese",
    "Vikings",
    "Wei",
    "Wu",
)

AOE2_CIV_ALIASES = {
    "bizantines": "Byzantines",
    "byzantine": "Byzantines",
    "frank": "Franks",
    "gurjara": "Gurjaras",
    "hindustani": "Hindustanis",
    "hun": "Huns",
    "inca": "Incas",
    "italian": "Italians",
    "jurchen": "Jurchens",
    "khitan": "Khitans",
    "lạc việt": "Lac Viet",
    "lacviet": "Lac Viet",
    "lithuanian": "Lithuanians",
    "magyar": "Magyars",
    "malian": "Malians",
    "mapuche": "Mapuche",
    "maya": "Mayans",
    "mayan": "Mayans",
    "mongol": "Mongols",
    "muisca": "Muisca",
    "persian": "Persians",
    "pole": "Poles",
    "portuguese": "Portuguese",
    "roman": "Romans",
    "saracen": "Saracens",
    "slav": "Slavs",
    "teuton": "Teutons",
    "turk": "Turks",
    "tupi": "Tupi",
    "viking": "Vikings",
}


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
        "aoe2": "Civilization",
    }.get(normalize_game(game), "Subject")


def aoe2_civilization_names_for_prompt() -> str:
    return "\n".join(f"- {name}" for name in AOE2_CIVILIZATIONS)


def canonical_aoe2_civilization(name: str | None) -> str | None:
    if not name:
        return None
    key = re.sub(r"[^a-z0-9 ]", "", name.strip().lower())
    if not key:
        return None
    lookup = {
        re.sub(r"[^a-z0-9 ]", "", civilization.lower()): civilization
        for civilization in AOE2_CIVILIZATIONS
    }
    lookup.update({
        re.sub(r"[^a-z0-9 ]", "", alias.lower()): civilization
        for alias, civilization in AOE2_CIV_ALIASES.items()
    })
    return lookup.get(key)

def analysis_spec(game: str, source: str) -> AnalysisSpec:
    game = normalize_game(game)
    if game == "aoe2":
        insight_types = AOE2_INSIGHT_TYPES
        if source == "aoe2_wiki":
            system_prompt = AOE2_GUIDE_WRITTEN_SYSTEM_PROMPT
        elif source == "aoe2_coaching":
            system_prompt = AOE2_GUIDE_COACHING_SYSTEM_PROMPT
        else:
            system_prompt = AOE2_GUIDE_VIDEO_SYSTEM_PROMPT
        extraction_prompt = build_aoe2_guide_extraction_prompt(insight_types)
        return AnalysisSpec(
            system_prompt=system_prompt,
            extraction_prompt=extraction_prompt,
            insight_types=insight_types,
            reference_block=aoe2_civilization_names_for_prompt(),
        )

    insight_types = LOL_WRITTEN_INSIGHT_TYPES if source == "mobafire_guide" else LOL_INSIGHT_TYPES
    system_prompt = LOL_GUIDE_WRITTEN_SYSTEM_PROMPT if source == "mobafire_guide" else LOL_GUIDE_VIDEO_SYSTEM_PROMPT
    extraction_prompt = build_lol_guide_extraction_prompt(insight_types)
    return AnalysisSpec(
        system_prompt=system_prompt,
        extraction_prompt=extraction_prompt,
        insight_types=insight_types,
        reference_block=champion_names_for_prompt(),
    )
