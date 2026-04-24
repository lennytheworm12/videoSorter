"""
Champion name registry and transcription error correction.

Fetches the full champion list from Riot Data Dragon and caches it locally.
Builds regex patterns to correct auto-caption misspellings in transcripts
and LLM output.
"""

import json
import re
import pathlib
import urllib.request
from functools import lru_cache

CACHE_PATH = pathlib.Path(__file__).with_name("champion_cache.json")
DDRAGON_VERSIONS = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_CHAMPIONS = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"

# Hand-coded corrections for names that auto-captions consistently mangle.
# Key = lowercase canonical name, value = list of known bad transcriptions.
KNOWN_ERRORS: dict[str, list[str]] = {
    "malzahar":       ["malahar", "malazar", "malasar", "malazhar", "malzar"],
    "karthus":        ["karfus", "carthus", "karthas"],
    "skarner":        ["scarner", "skarn", "scarner"],
    "cassiopeia":     ["casiopeia", "casio peia", "cassio peia", "cassio"],
    "sejuani":        ["sejauni", "sejuani", "swani", "sejwani"],
    "aatrox":         ["atrox", "aatrocks", "a trox"],
    "azir":           ["azeer", "azer", "a zeer"],
    "bel'veth":       ["bel veth", "belveth", "belvef", "bel vef"],
    "cho'gath":       ["chogath", "cho gath", "chogat", "cho gat"],
    "dr. mundo":      ["dr mundo", "doctor mundo", "mundo"],
    "fiddlesticks":   ["fiddle sticks", "fiddlestick"],
    "gragas":         ["gragus", "gragis"],
    "jarvan iv":      ["jarvan 4", "jarvan the fourth", "jarvan"],
    "k'sante":        ["ksante", "k sante", "k'sant"],
    "kai'sa":         ["kaisa", "kai sa", "kaysa"],
    "kha'zix":        ["khazix", "kha zix", "kazix"],
    "kog'maw":        ["kogmaw", "kog maw", "cog maw"],
    "lee sin":        ["lesin", "lee sen", "leesen"],
    "master yi":      ["masteryi", "master e"],
    "miss fortune":   ["misfortune", "miss f", "ms fortune"],
    "nunu & willump": ["nunu", "nunu and willump"],
    "rek'sai":        ["reksai", "rek sai", "rec sai"],
    "renata glasc":   ["renata glasc", "renata"],
    "tahm kench":     ["tam kench", "tahm ken", "tom kench"],
    "twisted fate":   ["twisted faith", "twist of fate", "tf"],
    "vel'koz":        ["velkoz", "vel koz", "vel cos"],
    "wukong":         ["wu kong", "monkey king"],
    "xin zhao":       ["xin jo", "xin zao", "shin zhao"],
    "yorick":         ["yorik", "yoric"],
}


def _fetch_champion_names() -> list[str]:
    """Fetch current champion names from Riot Data Dragon."""
    with urllib.request.urlopen(DDRAGON_VERSIONS, timeout=5) as r:
        version = json.loads(r.read())[0]
    url = DDRAGON_CHAMPIONS.format(version=version)
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.loads(r.read())
    names = [v["name"] for v in data["data"].values()]
    return sorted(names)


def load_champion_names(force_refresh: bool = False) -> list[str]:
    """Return champion names, using local cache when available."""
    if CACHE_PATH.exists() and not force_refresh:
        return json.loads(CACHE_PATH.read_text())
    try:
        names = _fetch_champion_names()
        CACHE_PATH.write_text(json.dumps(names, indent=2))
        print(f"[champions] Fetched {len(names)} champions from Data Dragon, cached to {CACHE_PATH}")
        return names
    except Exception as e:
        print(f"[champions] Data Dragon fetch failed ({e}), using cache if available")
        if CACHE_PATH.exists():
            return json.loads(CACHE_PATH.read_text())
        raise RuntimeError("No champion cache and Data Dragon unreachable") from e


def _name_variants(name: str) -> list[str]:
    """
    Generate transcription variants for a champion name automatically.
    Handles apostrophes, dots, multi-word names, and common phonetic drift.
    """
    variants = set()
    low = name.lower()

    # Remove apostrophes: Kha'Zix → khazix
    no_apos = re.sub(r"[''`]", "", low)
    variants.add(no_apos)

    # Remove dots: Dr. Mundo → dr mundo
    no_dot = re.sub(r"\.", "", low).strip()
    variants.add(no_dot)

    # Remove all punctuation
    no_punct = re.sub(r"[^a-z0-9 ]", "", low).strip()
    variants.add(no_punct)

    # Collapse spaces: Miss Fortune → missfortune
    no_space = no_punct.replace(" ", "")
    variants.add(no_space)

    # Remove the original lowercase (we want variants only)
    variants.discard(low)
    variants.discard("")
    return list(variants)


@lru_cache(maxsize=1)
def _build_correction_map() -> list[tuple[re.Pattern, str]]:
    """
    Build a list of (pattern, replacement) tuples sorted longest-pattern-first
    to prevent partial match clobbering.
    """
    names = load_champion_names()
    entries: list[tuple[str, str]] = []  # (bad_form, canonical)

    for canonical in names:
        # Auto-generated variants
        for variant in _name_variants(canonical):
            entries.append((variant, canonical))

        # Hand-coded overrides
        low = canonical.lower()
        for bad in KNOWN_ERRORS.get(low, []):
            entries.append((bad, canonical))

    # Deduplicate, longest pattern first (avoids "lee" matching before "lee sin")
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for bad, good in sorted(entries, key=lambda x: -len(x[0])):
        if bad not in seen and bad.lower() != good.lower():
            seen.add(bad)
            deduped.append((bad, good))

    # Compile to regex patterns with word boundaries
    compiled = []
    for bad, good in deduped:
        try:
            pat = re.compile(r"\b" + re.escape(bad) + r"\b", re.IGNORECASE)
            compiled.append((pat, good))
        except re.error:
            pass

    return compiled


def correct_names(text: str) -> str:
    """
    Apply champion name corrections to a piece of text.
    Replaces known auto-caption misspellings with the canonical LoL name.
    """
    for pattern, replacement in _build_correction_map():
        text = pattern.sub(replacement, text)
    return text


def get_all_champion_names() -> list[str]:
    """Return the full list of canonical champion names."""
    return load_champion_names()


def champion_names_for_prompt() -> str:
    """Return a compact string of all champion names for injection into prompts."""
    return ", ".join(load_champion_names())


@lru_cache(maxsize=1)
def _build_title_patterns() -> list[tuple[re.Pattern, str]]:
    """
    Build (pattern, canonical) pairs for extracting champion names from video titles.
    Includes all canonical names + their known variants, longest first.
    """
    names = load_champion_names()
    entries: list[tuple[str, str]] = []

    for canonical in names:
        entries.append((canonical, canonical))
        for variant in _name_variants(canonical):
            entries.append((variant, canonical))
        for bad in KNOWN_ERRORS.get(canonical.lower(), []):
            entries.append((bad, canonical))

    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for raw, good in sorted(entries, key=lambda x: -len(x[0])):
        low = raw.lower()
        if low not in seen and low != good.lower():
            seen.add(low)
            deduped.append((raw, good))
        elif low == good.lower() and low not in seen:
            seen.add(low)
            deduped.append((raw, good))

    compiled = []
    for raw, good in deduped:
        try:
            pat = re.compile(r"\b" + re.escape(raw) + r"\b", re.IGNORECASE)
            compiled.append((pat, good))
        except re.error:
            pass
    return compiled


def extract_champion_from_title(title: str) -> str | None:
    """
    Find the first champion name mentioned in a video title.
    Returns the canonical name (e.g. "Kai'Sa") or None if not found.

    Examples:
        "Platinum 2 Kaisa ADC Coaching"   → "Kai'Sa"
        "Master Cassiopeia vs Zed"         → "Cassiopeia"
        "General wave management tips"     → None
    """
    if not title:
        return None
    for pattern, canonical in _build_title_patterns():
        if pattern.search(title):
            return canonical
    return None


if __name__ == "__main__":
    # Quick test
    names = load_champion_names()
    print(f"Total champions: {len(names)}")

    test_cases = [
        "buy QSS against malahar or scarner",
        "cassio is strong vs twisted faith",
        "lesin is a good pick this patch",
        "khazix can one shot kaisa in the jungle",
        "chogath is broken in top lane",
    ]
    print("\nCorrection test:")
    for t in test_cases:
        print(f"  IN : {t}")
        print(f"  OUT: {correct_names(t)}")
        print()
