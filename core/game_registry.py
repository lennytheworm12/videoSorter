from __future__ import annotations

from dataclasses import dataclass
import re
from difflib import get_close_matches

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
    "armenian": "Armenians",
    "athenian": "Athenians",
    "aztec": "Aztecs",
    "bengali": "Bengalis",
    "berber": "Berbers",
    "bizantines": "Byzantines",
    "byz": "Byzantines",
    "byzants": "Byzantines",
    "byzintines": "Byzantines",
    "bohemian": "Bohemians",
    "briton": "Britons",
    "bulgarian": "Bulgarians",
    "burgundian": "Burgundians",
    "burmese": "Burmese",
    "byzantine": "Byzantines",
    "byzantins": "Byzantines",
    "celt": "Celts",
    "chinese": "Chinese",
    "cuman": "Cumans",
    "dravidians": "Dravidians",
    "ethiopian": "Ethiopians",
    "frank": "Franks",
    "georgian": "Georgians",
    "goth": "Goths",
    "gurjara": "Gurjaras",
    "hindustan": "Hindustanis",
    "hindustans": "Hindustanis",
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
    "malya": "Malay",
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
    "viet": "Vietnamese",
    "vietnam": "Vietnamese",
    "vietnamese": "Vietnamese",
}

_AOE2_CIV_LOOKUP: dict[str, str] | None = None


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
    lookup = aoe2_civilization_lookup()
    return lookup.get(key)


def aoe2_civilization_lookup() -> dict[str, str]:
    global _AOE2_CIV_LOOKUP
    if _AOE2_CIV_LOOKUP is None:
        lookup: dict[str, str] = {}
        for civilization in AOE2_CIVILIZATIONS:
            key = re.sub(r"[^a-z0-9 ]", "", civilization.lower())
            compact = key.replace(" ", "")
            lookup[key] = civilization
            lookup[compact] = civilization
            if key.endswith("s"):
                lookup[key[:-1]] = civilization
                lookup[compact[:-1]] = civilization
        for alias, civilization in AOE2_CIV_ALIASES.items():
            key = re.sub(r"[^a-z0-9 ]", "", alias.lower())
            lookup[key] = civilization
            lookup[key.replace(" ", "")] = civilization
        _AOE2_CIV_LOOKUP = lookup
    return _AOE2_CIV_LOOKUP


def find_aoe2_civilizations(text: str, max_distance: int = 1) -> list[tuple[int, str]]:
    """
    Find civ mentions with exact alias matches plus conservative typo fallback.

    The typo fallback only considers single-token civ names/aliases and allows a
    one-edit near match, which catches common query typos without turning random
    strategy words into civilization names.
    """
    q = text.lower()
    q_norm = re.sub(r"[^a-z0-9 ]", " ", q)
    q_norm = re.sub(r"\s+", " ", q_norm).strip()
    lookup = aoe2_civilization_lookup()

    found: list[tuple[int, str]] = []
    spans: list[tuple[int, int]] = []

    def add(start: int, end: int, civilization: str) -> None:
        if any(s <= start < e or s < end <= e for s, e in spans):
            return
        if any(existing == civilization for _, existing in found):
            return
        spans.append((start, end))
        found.append((start, civilization))

    for key in sorted(lookup, key=len, reverse=True):
        if not key:
            continue
        for search_q in (q, q_norm):
            match = re.search(r"\b" + re.escape(key) + r"\b", search_q)
            if match:
                add(match.start(), match.end(), lookup[key])
                break

    # Conservative typo fallback for single-token civ names and aliases.
    token_matches = list(re.finditer(r"\b[a-z][a-z0-9]{3,14}\b", q_norm))
    single_token_lookup = {
        key: value
        for key, value in lookup.items()
        if " " not in key and len(key) >= 4
    }
    single_keys = list(single_token_lookup)
    for match in token_matches:
        token = match.group(0)
        if token in single_token_lookup:
            continue
        close = get_close_matches(token, single_keys, n=1, cutoff=0.84 if max_distance <= 1 else 0.78)
        if close:
            add(match.start(), match.end(), single_token_lookup[close[0]])

    found.sort(key=lambda item: item[0])
    return found


def analysis_spec(game: str, source: str) -> AnalysisSpec:
    game = normalize_game(game)
    if game == "aoe2":
        insight_types = AOE2_INSIGHT_TYPES
        if source in {"aoe2_wiki", "aoe2_pdf"}:
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
