"""
Pull champion ability data from Riot Data Dragon and store in champion_abilities.

Tags each ability with mechanical properties using a synonym dictionary so
ability_enrich.py and retrieval/query.py can reason about cooldown windows,
CC interactions, and mobility counters without needing an LLM call.

Usage:
    uv run python -m pipeline.ability_scrape              # pull all champions
    uv run python -m pipeline.ability_scrape --champion "Cassiopeia"
    uv run python -m pipeline.ability_scrape --status     # show coverage
"""

import json
import re
import argparse
import urllib.request
from core.database import get_connection, init_db
from core.champions import load_champion_names

DDRAGON_VERSIONS = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_CHAMPION = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion/{key}.json"
DDRAGON_CHAMPIONS = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"


# ── Property synonym dictionary ───────────────────────────────────────────────
# Each entry: (tag, [synonyms to search for in ability description])
# Sorted from most specific to most general within each category.

PROPERTY_PATTERNS: list[tuple[str, list[str]]] = [
    # ── Hard CC ───────────────────────────────────────────────────────────────
    ("suppression",     ["suppressed", "suppression", "unable to act"]),
    ("stasis",          ["stasis", "becomes invulnerable and untargetable", "suspended in"]),
    ("sleep",           ["asleep", "sleeping", " sleep"]),
    ("polymorph",       ["polymorph", "polymorphed", "transformed into"]),
    ("airborne",        ["knock up", "knocks up", "knocked up", "knockup",
                         "knock back", "knocks back", "knocked back", "knockback",
                         " pulled ", "flings", "launches", "tosses", "airborne"]),
    ("stun",            ["stun", "stunned", "unable to move or act"]),
    ("charm",           ["charm", "charmed", "walk toward"]),
    ("taunt",           ["taunt", "taunted", "forced to attack"]),
    ("fear",            ["fear", "feared", "flee", "forced to flee", "walk away"]),

    # ── Soft CC ───────────────────────────────────────────────────────────────
    ("grounded",        ["grounded", "cannot use movement abilities",
                         "disables dashes", "prevents dashes", "their movement abilities"]),
    ("root",            ["root", "rooted", "snare", "snared",
                         "unable to move", "immobilizes", "immobilized"]),
    ("silence",         ["silenc", "cannot cast", "unable to cast"]),
    ("disarm",          ["disarm", "disarmed", "cannot attack", "prevents basic attacks"]),
    ("blind",           [" blind", "blinded", " miss ", "attacks miss"]),
    ("slow",            [" slow", "slows", "slowed", "reduces movement speed",
                         "movement speed is reduced"]),
    ("cripple",         ["cripple", "attack speed slow", "reduces attack speed",
                         "attack speed is reduced"]),

    # ── Mobility ──────────────────────────────────────────────────────────────
    ("blink",           ["blink", "blinks", "instantly relocates", "instantly moves to",
                         "teleports to target location", "repositions instantly"]),
    ("dash",            [" dash", "dashes", " leaps", " lunges", " charges toward",
                         "rushes to", "propels himself", "propels her"]),
    ("teleport",        ["teleports to an allied", "global teleport", "channels and teleports"]),
    ("haste",           ["movement speed", "gains movement speed", "increases movement speed",
                         "bonus movement speed"]),

    # ── Survivability ─────────────────────────────────────────────────────────
    ("unstoppable",     ["unstoppable", "immune to crowd control",
                         "cannot be interrupted", "cc immune", "ignores crowd control"]),
    ("invulnerable",    ["invulnerable", "immune to damage", "takes no damage"]),
    ("untargetable",    ["untargetable", "cannot be targeted",
                         "invulnerable to targeted", "becomes untargetable"]),
    ("spell_shield",    ["spell shield", "blocks the next", "negates the next ability"]),
    ("shield",          [" shield", "shields", "grants a shield", "creates a barrier",
                         "barrier that absorbs"]),
    ("heal",            [" heal", "heals", "restores health", "regenerates health",
                         "recovers health"]),

    # ── Utility ───────────────────────────────────────────────────────────────
    ("stealth",         ["invisible", "invisibility", "camouflage",
                         "hidden from", "stealths", "enter stealth"]),
    ("true_sight",      ["true sight", "reveals", "cannot be hidden", "ignores stealth"]),
    ("grievous_wounds", ["grievous wounds", "reduces healing", "healing reduction",
                         "reduced healing"]),
    ("shred",           ["shred", "reduces armor", "reduces magic resistance",
                         "armor reduction", "magic resistance reduction"]),
    ("empowered_attack", ["empowers", "next attack", "auto-attack reset",
                          "enhances his next", "enhances her next",
                          "next basic attack deals"]),
    ("execute",         ["executes", "execute", "deals bonus damage to low"]),
    ("true_damage",     ["true damage", "deals true damage"]),
    ("on_hit",          ["on-hit", "on hit", "each attack applies", "basic attacks apply"]),

    # ── Engine / Delivery ─────────────────────────────────────────────────────
    ("channel",         ["channel", "channels", "while channeling", "channeled"]),
    ("terrain",         ["creates terrain", " wall", "impassable terrain",
                         "blocks pathing", "terrain is created"]),
    ("trap",            [" trap", " mine", "places a", "triggered when", "activates when stepped"]),
    ("projectile",      ["projectile", " fires ", " fires a", "launches a",
                         " shoots ", " throws "]),
    ("aoe",             [" area", " radius", "nearby enemies", "surrounding enemies",
                         "all enemies in", "in a cone", "in an area"]),
]

# Compile all patterns once
_COMPILED: list[tuple[str, list[re.Pattern]]] = [
    (tag, [re.compile(re.escape(s), re.IGNORECASE) for s in synonyms])
    for tag, synonyms in PROPERTY_PATTERNS
]


def tag_properties(description: str) -> list[str]:
    """Return a list of property tags present in an ability description."""
    if not description:
        return []
    tags = []
    for tag, patterns in _COMPILED:
        if any(p.search(description) for p in patterns):
            tags.append(tag)
    return tags


# ── Data Dragon helpers ───────────────────────────────────────────────────────

def _get_latest_version() -> str:
    with urllib.request.urlopen(DDRAGON_VERSIONS, timeout=5) as r:
        return json.loads(r.read())[0]


def _build_name_key_map(version: str) -> dict[str, str]:
    """Fetch champion list and return {canonical_name: ddragon_key} mapping."""
    url = DDRAGON_CHAMPIONS.format(version=version)
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.loads(r.read())
    return {v["name"]: champ_id for champ_id, v in data["data"].items()}


def _fetch_champion_data(version: str, key: str) -> dict:
    """Fetch full champion JSON from Data Dragon. key = Data Dragon key (e.g. 'Kaisa')."""
    url = DDRAGON_CHAMPION.format(version=version, key=key)
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.loads(r.read())
    return data["data"][key]


def _format_values(vals) -> str:
    """Format a list of rank values as slash-joined string."""
    if not vals:
        return ""
    if isinstance(vals, list):
        # Deduplicate identical values (e.g. [0,0,0,0,0] → "0")
        unique = list(dict.fromkeys(str(int(v) if v == int(v) else v) for v in vals))
        return "/".join(unique) if len(unique) > 1 else unique[0]
    return str(vals)


def scrape_champion(version: str, champion: str, name_key_map: dict[str, str] | None = None) -> list[tuple]:
    """
    Pull ability data for one champion and return rows ready for DB insert.
    Returns list of (champion, slot, name, description, cooldown, range, cost, properties_json)
    """
    key = (name_key_map or {}).get(champion) or re.sub(r"[^a-zA-Z0-9]", "", champion)
    try:
        data = _fetch_champion_data(version, key)
    except Exception as e:
        print(f"  [error] {champion} (key={key}): {e}")
        return []

    rows = []

    # Passive
    passive = data.get("passive", {})
    p_desc = passive.get("description", "")
    rows.append((
        champion, "P",
        passive.get("name", ""),
        p_desc, "", "", "",
        json.dumps(tag_properties(p_desc)),
    ))

    # Q W E R
    slots = ["Q", "W", "E", "R"]
    for slot, spell in zip(slots, data.get("spells", [])):
        desc = spell.get("description", "")
        cooldown = _format_values(spell.get("cooldownBurn", ""))
        rng = _format_values(spell.get("rangeBurn", ""))
        cost = _format_values(spell.get("costBurn", ""))
        rows.append((
            champion, slot,
            spell.get("name", ""),
            desc, cooldown, rng, cost,
            json.dumps(tag_properties(desc)),
        ))

    return rows


def upsert_abilities(rows: list[tuple]) -> None:
    with get_connection() as conn:
        conn.executemany("""
            INSERT INTO champion_abilities
                (champion, ability_slot, name, description, cooldown, range, cost, properties)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(champion, ability_slot) DO UPDATE SET
                name        = excluded.name,
                description = excluded.description,
                cooldown    = excluded.cooldown,
                range       = excluded.range,
                cost        = excluded.cost,
                properties  = excluded.properties
        """, rows)
        conn.commit()


# ── Status ────────────────────────────────────────────────────────────────────

def print_status() -> None:
    with get_connection() as conn:
        n_champions = conn.execute(
            "SELECT COUNT(DISTINCT champion) FROM champion_abilities"
        ).fetchone()[0]
        n_abilities = conn.execute(
            "SELECT COUNT(*) FROM champion_abilities"
        ).fetchone()[0]

        # Top property tags
        all_props = conn.execute(
            "SELECT properties FROM champion_abilities WHERE properties IS NOT NULL"
        ).fetchall()

    from collections import Counter
    tag_counts: Counter = Counter()
    for row in all_props:
        try:
            tags = json.loads(row[0])
            tag_counts.update(tags)
        except Exception:
            pass

    print(f"\nchampion_abilities: {n_champions} champions, {n_abilities} abilities\n")
    print("Top property tags:")
    for tag, count in tag_counts.most_common(20):
        print(f"  {tag:<20} {count}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pull champion ability data from Data Dragon")
    parser.add_argument("--champion", help="Pull a single champion only")
    parser.add_argument("--status", action="store_true", help="Show coverage and tag distribution")
    args = parser.parse_args()

    init_db()

    if args.status:
        print_status()
        return

    print("Fetching latest Data Dragon version…")
    version = _get_latest_version()
    print(f"Version: {version}")

    name_key_map = _build_name_key_map(version)

    if args.champion:
        champions = [args.champion]
    else:
        champions = load_champion_names()

    print(f"Pulling {len(champions)} champion(s)…")
    total_rows = 0
    errors = 0
    for i, champion in enumerate(champions, 1):
        rows = scrape_champion(version, champion, name_key_map)
        if rows:
            upsert_abilities(rows)
            total_rows += len(rows)
            if i % 20 == 0:
                print(f"  {i}/{len(champions)} done…")
        else:
            errors += 1

    print(f"\nDone — {total_rows} ability rows upserted, {errors} errors.")
    print_status()


if __name__ == "__main__":
    main()
