"""
Populate champion_archetypes table from empirical blog data + Gemini fill-in.

Data sources:
  - machineloling.com Season 16 encyclopedia (8.6M diamond+ games, empirical)
  - Gemini fills in any champion in our DB not covered by the blog data

Usage:
    python -m pipeline.champion_archetypes              # populate / refresh
    python -m pipeline.champion_archetypes --status     # show current table
    python -m pipeline.champion_archetypes --fill-gaps  # only run Gemini fill-in
"""

import argparse
import json
from core.database import get_connection, init_db
from core.champions import load_champion_names
from core.llm import chat as llm_chat

# ── Empirical data from machineloling.com Season 16 encyclopedia ──────────────
# Source: 8.6M diamond+ games, WR-corrected deltas, Ward's hierarchical clustering

EMPIRICAL: list[tuple[str, str, str]] = [
    # (champion, role, archetype)

    # SUPPORT
    ("Braum",      "support", "Warden"),
    ("Taric",      "support", "Warden"),
    ("Alistar",    "support", "Warden"),
    ("Rell",       "support", "Warden"),
    ("Rakan",      "support", "Warden"),
    ("Leona",      "support", "Warden"),
    ("Nautilus",   "support", "Playmaker"),
    ("Blitzcrank", "support", "Playmaker"),
    ("Pyke",       "support", "Playmaker"),
    ("Tahm Kench", "support", "Playmaker"),
    ("Thresh",     "support", "Playmaker"),
    ("Maokai",     "support", "Playmaker"),
    ("Poppy",      "support", "Playmaker"),
    ("Pantheon",   "support", "Playmaker"),
    ("Elise",      "support", "Playmaker"),
    ("Morgana",    "support", "Mage"),
    ("Brand",      "support", "Mage"),
    ("Neeko",      "support", "Mage"),
    ("Swain",      "support", "Mage"),
    ("Zyra",       "support", "Mage"),
    ("Mel",        "support", "Mage"),
    ("Xerath",     "support", "Mage"),
    ("Lux",        "support", "Mage"),
    ("Vel'Koz",    "support", "Mage"),
    ("Yuumi",      "support", "Cat"),
    ("Senna",      "support", "Enchanter"),
    ("Seraphine",  "support", "Enchanter"),
    ("Karma",      "support", "Enchanter"),
    ("Bard",       "support", "Enchanter"),
    ("Zilean",     "support", "Enchanter"),
    ("Lulu",       "support", "Enchanter"),
    ("Nami",       "support", "Enchanter"),
    ("Janna",      "support", "Enchanter"),
    ("Sona",       "support", "Enchanter"),
    ("Milio",      "support", "Enchanter"),
    ("Soraka",     "support", "Enchanter"),

    # ADC
    ("Ziggs",      "adc", "Poke"),
    ("Ezreal",     "adc", "Poke"),
    ("Jhin",       "adc", "Poke"),
    ("Miss Fortune","adc", "Long-Range Immobile"),
    ("Ashe",       "adc", "Long-Range Immobile"),
    ("Varus",      "adc", "Long-Range Immobile"),
    ("Draven",     "adc", "Long-Range Immobile"),
    ("Kog'Maw",    "adc", "Long-Range Immobile"),
    ("Jinx",       "adc", "Long-Range Immobile"),
    ("Caitlyn",    "adc", "Long-Range Immobile"),
    ("Aphelios",   "adc", "Long-Range Immobile"),
    ("Twitch",     "adc", "Short-Range Mobile"),
    ("Samira",     "adc", "Short-Range Mobile"),
    ("Kai'Sa",     "adc", "Short-Range Mobile"),
    ("Vayne",      "adc", "Short-Range Mobile"),
    ("Zeri",       "adc", "Short-Range Mobile"),
    ("Lucian",     "adc", "Short-Range Mobile"),
    ("Tristana",   "adc", "Short-Range Mobile"),
    ("Nilah",      "adc", "Anti-Melee"),
    ("Xayah",      "adc", "Anti-Melee"),
    ("Yumira",     "adc", "Anti-Melee"),
    ("Corki",      "adc", "Anti-Melee"),
    ("Sivir",      "adc", "Anti-Melee"),
    ("Smolder",    "adc", "Anti-Melee"),
    ("Swain",      "adc", "Anti-Melee"),

    # MID
    ("Yasuo",        "mid", "Yasuo"),
    ("Naafiri",      "mid", "Melee Assassin"),
    ("Zed",          "mid", "Melee Assassin"),
    ("Qiyana",       "mid", "Melee Assassin"),
    ("Talon",        "mid", "Melee Assassin"),
    ("Yone",         "mid", "Melee Assassin"),
    ("Irelia",       "mid", "Melee Assassin"),
    ("Katarina",     "mid", "Melee Assassin"),
    ("Akshan",       "mid", "Melee Assassin"),
    ("Kassadin",     "mid", "Melee Assassin"),
    ("Akali",        "mid", "Melee Assassin"),
    ("Diana",        "mid", "Melee Assassin"),
    ("Fizz",         "mid", "Melee Assassin"),
    ("Ekko",         "mid", "Melee Assassin"),
    ("Xerath",       "mid", "True Mage"),
    ("Aurelion Sol", "mid", "True Mage"),
    ("Veigar",       "mid", "True Mage"),
    ("Taliyah",      "mid", "True Mage"),
    ("Syndra",       "mid", "True Mage"),
    ("Lux",          "mid", "True Mage"),
    ("Hwei",         "mid", "True Mage"),
    ("Vel'Koz",      "mid", "True Mage"),
    ("Ahri",         "mid", "Ranged Assassin"),
    ("Aurora",       "mid", "Ranged Assassin"),
    ("LeBlanc",      "mid", "Ranged Assassin"),
    ("Zoe",          "mid", "Ranged Assassin"),
    ("Lissandra",    "mid", "Anti-Assassin"),
    ("Vex",          "mid", "Anti-Assassin"),
    ("Twisted Fate", "mid", "Anti-Assassin"),
    ("Malzahar",     "mid", "Anti-Assassin"),
    ("Mel",          "mid", "Battle Mage"),
    ("Ryze",         "mid", "Battle Mage"),
    ("Vladimir",     "mid", "Battle Mage"),
    ("Cassiopeia",   "mid", "Battle Mage"),
    ("Viktor",       "mid", "Battle Mage"),
    ("Orianna",      "mid", "Battle Mage"),
    ("Anivia",       "mid", "Battle Mage"),
    ("Azir",         "mid", "Battle Mage"),
    ("Sylas",        "mid", "Anti-Melee"),
    ("Galio",        "mid", "Anti-Melee"),

    # JUNGLE
    ("Zed",       "jungle", "Farming"),
    ("Kindred",   "jungle", "Farming"),
    ("Jayce",     "jungle", "Farming"),
    ("Graves",    "jungle", "Farming"),
    ("Karthus",   "jungle", "Farming"),
    ("Taliyah",   "jungle", "Farming"),
    ("Rengar",    "jungle", "Assassin"),
    ("Nidalee",   "jungle", "Assassin"),
    ("Talon",     "jungle", "Assassin"),
    ("Fiddlesticks", "jungle", "Assassin"),
    ("Shaco",     "jungle", "Assassin"),
    ("Elise",     "jungle", "Assassin"),
    ("Kha'Zix",   "jungle", "Assassin"),
    ("Qiyana",    "jungle", "Assassin"),
    ("Sejuani",   "jungle", "Ult-Tank"),
    ("Malphite",  "jungle", "Ult-Tank"),
    ("Rammus",    "jungle", "Ult-Tank"),
    ("Nunu",      "jungle", "Ult-Tank"),
    ("Amumu",     "jungle", "Ult-Tank"),
    ("Lee Sin",   "jungle", "Fighter"),
    ("Rek'Sai",   "jungle", "Fighter"),
    ("Vi",        "jungle", "Fighter"),
    ("Briar",     "jungle", "Fighter"),
    ("Warwick",   "jungle", "Fighter"),
    ("Wukong",    "jungle", "Heavy-Fighter"),
    ("Zac",       "jungle", "Heavy-Fighter"),
    ("Jax",       "jungle", "Heavy-Fighter"),
    ("Volibear",  "jungle", "Heavy-Fighter"),
    ("Zaahen",    "jungle", "Heavy-Fighter"),
    ("Dr. Mundo", "jungle", "Diver"),
    ("Nocturne",  "jungle", "Diver"),
    ("Xin Zhao",  "jungle", "Diver"),
    ("Jarvan IV", "jungle", "Diver"),
    ("Diana",     "jungle", "Diver"),
    ("Hecarim",   "jungle", "Diver"),
    ("Ivern",     "jungle", "Slippery"),
    ("Ekko",      "jungle", "Slippery"),
    ("Lillia",    "jungle", "Slippery"),
    ("Gwen",      "jungle", "Slippery"),
    ("Sylas",     "jungle", "Slippery"),
    ("Udyr",      "jungle", "Slippery"),
    ("Viego",     "jungle", "Team-Assassin"),
    ("Master Yi", "jungle", "Team-Assassin"),
    ("Ambessa",   "jungle", "Team-Assassin"),
    ("Kayn",      "jungle", "Team-Assassin"),
    ("Naafiri",   "jungle", "Team-Assassin"),
    ("Bel'Veth",  "jungle", "Team-Assassin"),

    # TOP (from correlation matrix image)
    ("Varus",      "top", "Ranged"),
    ("Teemo",      "top", "Ranged"),
    ("Kennen",     "top", "Ranged"),
    ("Vladimir",   "top", "Ranged"),
    ("Akali",      "top", "Ranged"),
    ("Jayce",      "top", "Ranged"),
    ("Ryze",       "top", "Ranged"),
    ("Rumble",     "top", "Ranged"),
    ("Gnar",       "top", "Ranged"),
    ("Gragas",     "top", "Ranged"),
    ("Pantheon",   "top", "Ranged"),
    ("Kayle",      "top", "Ranged"),
    ("Vayne",      "top", "Ranged"),
    ("Riven",      "top", "Duelist"),
    ("Fiora",      "top", "Duelist"),
    ("Ambessa",    "top", "Duelist"),
    ("Yone",       "top", "Duelist"),
    ("Irelia",     "top", "Duelist"),
    ("Yasuo",      "top", "Duelist"),
    ("Gwen",       "top", "Duelist"),
    ("Yorick",     "top", "Duelist"),
    ("Gangplank",  "top", "Unique"),
    ("Olaf",       "top", "Unique"),
    ("Jax",        "top", "Unique"),
    ("Volibear",   "top", "Unique"),
    ("Warwick",    "top", "Unique"),
    ("Garen",      "top", "Unique"),
    ("Camille",    "top", "Unique"),
    ("Poppy",      "top", "Unique"),
    ("Nasus",      "top", "Unique"),
    ("Urgot",      "top", "Unique"),
    ("Tahm Kench", "top", "Unique"),
    ("Cho'Gath",   "top", "Unique"),
    ("Singed",     "top", "Unique"),
    ("Renekton",   "top", "Unique"),
    ("Illaoi",     "top", "Unique"),
    ("K'Sante",    "top", "Unique"),
    ("Shen",       "top", "Unique"),
    ("Tryndamere", "top", "Unique"),
    ("Aatrox",     "top", "Unique"),
    ("Kled",       "top", "Unique"),
    ("Mordekaiser","top", "Unique"),
    ("Darius",     "top", "Unique"),
    ("Zaahen",     "top", "Unique"),
    ("Sett",       "top", "Unique"),
    ("Malphite",   "top", "True_Tank"),
    ("Dr. Mundo",  "top", "True_Tank"),
    ("Ornn",       "top", "True_Tank"),
    ("Sion",       "top", "True_Tank"),
]


# ── DB helpers ────────────────────────────────────────────────────────────────

def upsert_archetypes(rows: list[tuple[str, str, str, str]]) -> None:
    """Insert or replace (champion, role, archetype, source) rows."""
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO champion_archetypes (champion, role, archetype, source)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(champion, role) DO UPDATE SET
                archetype = excluded.archetype,
                source    = excluded.source
            """,
            rows,
        )
        conn.commit()


def get_covered() -> set[tuple[str, str]]:
    """Return set of (champion, role) pairs already in the table."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT champion, role FROM champion_archetypes"
        ).fetchall()
    return {(r["champion"], r["role"]) for r in rows}


def get_uncovered_champions() -> list[tuple[str, str]]:
    """
    Return (champion, role) pairs that appear in our videos DB
    but have no archetype entry yet.
    """
    covered = get_covered()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT v.champion, v.role
            FROM videos v
            WHERE v.champion IS NOT NULL AND v.champion != ''
            """
        ).fetchall()
    return [
        (r["champion"], r["role"])
        for r in rows
        if (r["champion"], r["role"]) not in covered
    ]


# ── Gemini fill-in ────────────────────────────────────────────────────────────

FILL_SYSTEM = """
You are a League of Legends expert. Given a champion and their role,
classify them into the most appropriate archetype from the provided list.
Return valid JSON only — no markdown, no explanation.
""".strip()

ARCHETYPES_BY_ROLE = {
    "support": ["Warden", "Playmaker", "Mage", "Enchanter", "Cat"],
    "adc":     ["Poke", "Long-Range Immobile", "Short-Range Mobile", "Anti-Melee"],
    "mid":     ["Melee Assassin", "True Mage", "Ranged Assassin", "Anti-Assassin", "Battle Mage", "Anti-Melee", "Yasuo"],
    "jungle":  ["Farming", "Assassin", "Ult-Tank", "Fighter", "Heavy-Fighter", "Diver", "Slippery", "Team-Assassin"],
    "top":     ["Ranged", "Duelist", "Unique", "True_Tank"],
}


def fill_gaps_with_llm(pairs: list[tuple[str, str]]) -> list[tuple[str, str, str, str]]:
    """
    Ask Gemini to classify each (champion, role) pair not covered by blog data.
    Returns list of (champion, role, archetype, 'inferred') tuples.
    """
    if not pairs:
        return []

    results = []
    for champion, role in pairs:
        archetypes = ARCHETYPES_BY_ROLE.get(role, [])
        if not archetypes:
            print(f"  [skip] unknown role '{role}' for {champion}")
            continue

        prompt = f"""
Champion: {champion}
Role: {role}
Available archetypes: {', '.join(archetypes)}

Return JSON: {{"archetype": "<one of the archetypes above>"}}
""".strip()

        try:
            raw = llm_chat(system=FILL_SYSTEM, user=prompt, temperature=0.0)
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
            archetype = data["archetype"]
            results.append((champion, role, archetype, "inferred"))
            print(f"  {champion} ({role}) → {archetype}")
        except Exception as e:
            print(f"  [error] {champion} ({role}): {e}")

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def print_status() -> None:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, archetype, source, COUNT(*) as n
            FROM champion_archetypes
            GROUP BY role, archetype, source
            ORDER BY role, archetype
            """
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM champion_archetypes"
        ).fetchone()[0]

    print(f"\nChampion archetypes: {total} total\n")
    current_role = None
    for r in rows:
        if r["role"] != current_role:
            current_role = r["role"]
            print(f"  {current_role.upper()}")
        tag = "" if r["source"] == "empirical" else " (inferred)"
        print(f"    {r['archetype']:<25} {r['n']:>3} champions{tag}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate champion archetypes table")
    parser.add_argument("--status", action="store_true", help="Show current table and exit")
    parser.add_argument("--fill-gaps", action="store_true", help="Only run Gemini fill-in for uncovered champions")
    args = parser.parse_args()

    init_db()

    if args.status:
        print_status()
        return

    if not args.fill_gaps:
        # Load empirical data
        rows = [(c, r, a, "empirical") for c, r, a in EMPIRICAL]
        print(f"Loading {len(rows)} empirical entries from blog data…")
        upsert_archetypes(rows)
        print("Done.")

    # Fill gaps for champions in our videos DB
    gaps = get_uncovered_champions()
    if gaps:
        print(f"\n{len(gaps)} champion/role pair(s) in videos DB not covered by blog data:")
        for c, r in gaps:
            print(f"  {c} ({r})")
        print("\nAsking Gemini to classify…")
        inferred = fill_gaps_with_llm(gaps)
        if inferred:
            upsert_archetypes(inferred)
            print(f"\n{len(inferred)} inferred entries added.")
    else:
        print("\nNo gaps — all champions in videos DB are covered.")

    print_status()


if __name__ == "__main__":
    main()
