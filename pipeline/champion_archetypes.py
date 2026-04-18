"""
Populate champion_archetypes table from empirical blog data + Gemini fill-in.

One archetype per champion — role is not stored. A champion's playstyle identity
(e.g. Braum = Warden) is what matters for cross-referencing, regardless of what
role they happen to be played in a given video.

Data sources:
  - machineloling.com Season 16 encyclopedia (8.6M diamond+ games, empirical)
    Multi-role champions keep their primary/most-played role archetype.
  - Gemini fills in any champion not covered by the blog data

Usage:
    python -m pipeline.champion_archetypes              # populate / refresh
    python -m pipeline.champion_archetypes --status     # show current table
    python -m pipeline.champion_archetypes --fill-gaps  # only fill uncovered champions in videos DB
    python -m pipeline.champion_archetypes --fill-all   # seed ALL Data Dragon champions
"""

import argparse
import json
from core.database import get_connection, init_db
from core.champions import load_champion_names
from core.llm import chat as llm_chat

# ── Empirical data from machineloling.com Season 16 ──────────────────────────
# One entry per champion — primary role archetype chosen for multi-role champs.

EMPIRICAL: list[tuple[str, str]] = [
    # (champion, archetype)

    # SUPPORT
    ("Braum",        "Warden"),
    ("Taric",        "Warden"),
    ("Alistar",      "Warden"),
    ("Rell",         "Warden"),
    ("Rakan",        "Warden"),
    ("Leona",        "Warden"),
    ("Nautilus",     "Playmaker"),
    ("Blitzcrank",   "Playmaker"),
    ("Pyke",         "Playmaker"),
    ("Tahm Kench",   "Playmaker"),
    ("Thresh",       "Playmaker"),
    ("Maokai",       "Playmaker"),
    ("Poppy",        "Playmaker"),
    ("Pantheon",     "Playmaker"),
    ("Elise",        "Assassin"),       # primary: jungle Assassin
    ("Morgana",      "Mage"),
    ("Brand",        "Mage"),
    ("Neeko",        "Mage"),
    ("Swain",        "Mage"),
    ("Zyra",         "Mage"),
    ("Mel",          "Battle Mage"),    # primary: mid
    ("Xerath",       "True Mage"),      # primary: mid
    ("Lux",          "Mage"),
    ("Vel'Koz",      "Mage"),
    ("Yuumi",        "Cat"),
    ("Senna",        "Enchanter"),
    ("Seraphine",    "Enchanter"),
    ("Karma",        "Enchanter"),
    ("Bard",         "Enchanter"),
    ("Zilean",       "Enchanter"),
    ("Lulu",         "Enchanter"),
    ("Nami",         "Enchanter"),
    ("Janna",        "Enchanter"),
    ("Sona",         "Enchanter"),
    ("Milio",        "Enchanter"),
    ("Soraka",       "Enchanter"),

    # ADC
    ("Ziggs",        "Poke"),
    ("Ezreal",       "Poke"),
    ("Jhin",         "Poke"),
    ("Miss Fortune", "Long-Range Immobile"),
    ("Ashe",         "Long-Range Immobile"),
    ("Varus",        "Long-Range Immobile"),
    ("Draven",       "Long-Range Immobile"),
    ("Kog'Maw",      "Long-Range Immobile"),
    ("Jinx",         "Long-Range Immobile"),
    ("Caitlyn",      "Long-Range Immobile"),
    ("Aphelios",     "Long-Range Immobile"),
    ("Twitch",       "Short-Range Mobile"),
    ("Samira",       "Short-Range Mobile"),
    ("Kai'Sa",       "Short-Range Mobile"),
    ("Vayne",        "Short-Range Mobile"),
    ("Zeri",         "Short-Range Mobile"),
    ("Lucian",       "Short-Range Mobile"),
    ("Tristana",     "Short-Range Mobile"),
    ("Nilah",        "Anti-Melee"),
    ("Xayah",        "Anti-Melee"),
    ("Yunara",       "Anti-Melee"),
    ("Corki",        "Anti-Melee"),
    ("Sivir",        "Anti-Melee"),
    ("Smolder",      "Anti-Melee"),

    # MID
    ("Yasuo",        "Yasuo"),
    ("Naafiri",      "Melee Assassin"),
    ("Zed",          "Melee Assassin"),
    ("Qiyana",       "Melee Assassin"),
    ("Talon",        "Melee Assassin"),
    ("Yone",         "Melee Assassin"),
    ("Irelia",       "Melee Assassin"),
    ("Katarina",     "Melee Assassin"),
    ("Akshan",       "Melee Assassin"),
    ("Kassadin",     "Melee Assassin"),
    ("Akali",        "Melee Assassin"),
    ("Diana",        "Melee Assassin"),
    ("Fizz",         "Melee Assassin"),
    ("Ekko",         "Melee Assassin"),
    ("Aurelion Sol", "True Mage"),
    ("Veigar",       "True Mage"),
    ("Taliyah",      "True Mage"),
    ("Syndra",       "True Mage"),
    ("Hwei",         "True Mage"),
    ("Ahri",         "Ranged Assassin"),
    ("Aurora",       "Ranged Assassin"),
    ("LeBlanc",      "Ranged Assassin"),
    ("Zoe",          "Ranged Assassin"),
    ("Lissandra",    "Anti-Assassin"),
    ("Vex",          "Anti-Assassin"),
    ("Twisted Fate", "Anti-Assassin"),
    ("Malzahar",     "Anti-Assassin"),
    ("Ryze",         "Battle Mage"),
    ("Vladimir",     "Battle Mage"),
    ("Cassiopeia",   "Battle Mage"),
    ("Viktor",       "Battle Mage"),
    ("Orianna",      "Battle Mage"),
    ("Anivia",       "Battle Mage"),
    ("Azir",         "Battle Mage"),
    ("Sylas",        "Anti-Melee"),
    ("Galio",        "Anti-Melee"),

    # JUNGLE
    ("Kindred",      "Farming"),
    ("Jayce",        "Farming"),
    ("Graves",       "Farming"),
    ("Karthus",      "Farming"),
    ("Rengar",       "Assassin"),
    ("Nidalee",      "Assassin"),
    ("Fiddlesticks", "Assassin"),
    ("Shaco",        "Assassin"),
    ("Kha'Zix",      "Assassin"),
    ("Sejuani",      "Ult-Tank"),
    ("Malphite",     "Ult-Tank"),
    ("Rammus",       "Ult-Tank"),
    ("Nunu & Willump","Ult-Tank"),
    ("Amumu",        "Ult-Tank"),
    ("Lee Sin",      "Fighter"),
    ("Rek'Sai",      "Fighter"),
    ("Vi",           "Fighter"),
    ("Briar",        "Fighter"),
    ("Warwick",      "Fighter"),
    ("Wukong",       "Heavy-Fighter"),
    ("Zac",          "Heavy-Fighter"),
    ("Jax",          "Heavy-Fighter"),
    ("Volibear",     "Heavy-Fighter"),
    ("Zaahen",       "Heavy-Fighter"),
    ("Dr. Mundo",    "Diver"),
    ("Nocturne",     "Diver"),
    ("Xin Zhao",     "Diver"),
    ("Jarvan IV",    "Diver"),
    ("Hecarim",      "Diver"),
    ("Ivern",        "Slippery"),
    ("Lillia",       "Slippery"),
    ("Gwen",         "Slippery"),
    ("Udyr",         "Slippery"),
    ("Viego",        "Team-Assassin"),
    ("Master Yi",    "Team-Assassin"),
    ("Ambessa",      "Team-Assassin"),
    ("Kayn",         "Team-Assassin"),
    ("Bel'Veth",     "Team-Assassin"),

    # TOP
    ("Teemo",        "Ranged"),
    ("Kennen",       "Ranged"),
    ("Jayce",        "Ranged"),          # top Ranged (jungle Farming kept above, upsert handles dedup)
    ("Rumble",       "Ranged"),
    ("Gnar",         "Ranged"),
    ("Gragas",       "Ranged"),
    ("Kayle",        "Ranged"),
    ("Riven",        "Duelist"),
    ("Fiora",        "Duelist"),
    ("Yorick",       "Duelist"),
    ("Gangplank",    "Unique"),
    ("Olaf",         "Unique"),
    ("Garen",        "Unique"),
    ("Camille",      "Unique"),
    ("Nasus",        "Unique"),
    ("Urgot",        "Unique"),
    ("Cho'Gath",     "Unique"),
    ("Singed",       "Unique"),
    ("Renekton",     "Unique"),
    ("Illaoi",       "Unique"),
    ("K'Sante",      "Unique"),
    ("Shen",         "Unique"),
    ("Tryndamere",   "Unique"),
    ("Aatrox",       "Unique"),
    ("Kled",         "Unique"),
    ("Mordekaiser",  "Unique"),
    ("Darius",       "Unique"),
    ("Sett",         "Unique"),
    ("Ornn",         "True_Tank"),
    ("Sion",         "True_Tank"),
]

# All valid archetypes (role-agnostic pool for Gemini fill-in)
ALL_ARCHETYPES = [
    "Warden", "Playmaker", "Mage", "Enchanter", "Cat",
    "Poke", "Long-Range Immobile", "Short-Range Mobile", "Anti-Melee",
    "Melee Assassin", "True Mage", "Ranged Assassin", "Anti-Assassin",
    "Battle Mage", "Yasuo",
    "Farming", "Assassin", "Ult-Tank", "Fighter", "Heavy-Fighter",
    "Diver", "Slippery", "Team-Assassin",
    "Ranged", "Duelist", "Unique", "True_Tank",
]


# ── DB helpers ────────────────────────────────────────────────────────────────

def _migrate_if_needed() -> None:
    """Migrate old (champion, role) schema to single-entry (champion) schema."""
    with get_connection() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(champion_archetypes)").fetchall()}
        if "role" not in cols:
            return  # already migrated

        print("  Migrating champion_archetypes to role-agnostic schema…")
        # Collapse to one row per champion: prefer empirical, then first entry
        rows = conn.execute("""
            SELECT champion, archetype, source
            FROM champion_archetypes
            ORDER BY CASE source WHEN 'empirical' THEN 0 ELSE 1 END, champion
        """).fetchall()

        seen: dict[str, tuple[str, str]] = {}
        for r in rows:
            key = r["champion"].lower()
            if key not in seen:
                seen[key] = (r["champion"], r["archetype"], r["source"])

        conn.execute("DROP TABLE champion_archetypes")
        conn.execute("""
            CREATE TABLE champion_archetypes (
                champion   TEXT PRIMARY KEY,
                archetype  TEXT NOT NULL,
                source     TEXT DEFAULT 'empirical',
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.executemany(
            "INSERT INTO champion_archetypes (champion, archetype, source) VALUES (?, ?, ?)",
            [(c, a, s) for c, a, s in seen.values()]
        )
        conn.commit()
        print(f"  Migrated {len(seen)} champions.")


def upsert_archetypes(rows: list[tuple[str, str, str]]) -> None:
    """Insert or update (champion, archetype, source) rows."""
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO champion_archetypes (champion, archetype, source)
            VALUES (?, ?, ?)
            ON CONFLICT(champion) DO UPDATE SET
                archetype = excluded.archetype,
                source    = excluded.source
            """,
            rows,
        )
        conn.commit()


def get_covered() -> set[str]:
    """Return set of champion names already in the table."""
    with get_connection() as conn:
        rows = conn.execute("SELECT champion FROM champion_archetypes").fetchall()
    return {r["champion"].lower() for r in rows}


def get_uncovered_champions() -> list[str]:
    """Return champion names that appear in our videos DB but have no archetype entry."""
    covered = get_covered()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT champion FROM videos WHERE champion IS NOT NULL AND champion != ''"
        ).fetchall()
    return [r["champion"] for r in rows if r["champion"].lower() not in covered]


def get_all_uncovered_champions() -> list[str]:
    """Return champion names from Data Dragon not yet in the archetypes table."""
    covered = get_covered()
    return [n for n in load_champion_names() if n.lower() not in covered]


# ── Gemini fill-in ────────────────────────────────────────────────────────────

FILL_SYSTEM = """
You are a League of Legends expert. Given a champion name, classify them into
their primary playstyle archetype. Return valid JSON only — no markdown, no explanation.
""".strip()


def fill_with_llm(champion_names: list[str]) -> list[tuple[str, str, str]]:
    """
    Ask Gemini to classify each champion by archetype (role-agnostic).
    Returns list of (champion, archetype, 'inferred') tuples.
    """
    if not champion_names:
        return []

    archetypes_str = ", ".join(ALL_ARCHETYPES)
    results = []
    for i, champion in enumerate(champion_names, 1):
        prompt = f"""
Champion: {champion}
Available archetypes: {archetypes_str}

What is this champion's primary playstyle archetype?
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
            if archetype not in ALL_ARCHETYPES:
                print(f"  [skip] {champion}: unknown archetype '{archetype}'")
                continue
            results.append((champion, archetype, "inferred"))
            print(f"  [{i}/{len(champion_names)}] {champion} → {archetype}")
        except Exception as e:
            print(f"  [error] {champion}: {e}")

    return results


# ── Status ────────────────────────────────────────────────────────────────────

def print_status() -> None:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT archetype, source, COUNT(*) as n FROM champion_archetypes GROUP BY archetype, source ORDER BY archetype"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM champion_archetypes").fetchone()[0]

    print(f"\nChampion archetypes: {total} unique champions\n")
    for r in rows:
        tag = "" if r["source"] == "empirical" else " (inferred)"
        print(f"  {r['archetype']:<25} {r['n']:>3} champions{tag}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Populate champion archetypes table")
    parser.add_argument("--status",    action="store_true", help="Show current table and exit")
    parser.add_argument("--fill-gaps", action="store_true", help="Only fill champions in videos DB not yet covered")
    parser.add_argument("--fill-all",  action="store_true", help="Seed ALL Data Dragon champions")
    args = parser.parse_args()

    init_db()
    _migrate_if_needed()

    if args.status:
        print_status()
        return

    if not args.fill_gaps and not args.fill_all:
        rows = [(c, a, "empirical") for c, a in EMPIRICAL]
        print(f"Loading {len(rows)} empirical entries…")
        upsert_archetypes(rows)
        print("Done.")

    missing = get_all_uncovered_champions() if args.fill_all else get_uncovered_champions()
    if missing:
        label = "Data Dragon" if args.fill_all else "videos DB"
        print(f"\n{len(missing)} champion(s) from {label} not yet classified:")
        for n in missing:
            print(f"  {n}")
        print("\nAsking Gemini to classify…")
        inferred = fill_with_llm(missing)
        if inferred:
            upsert_archetypes(inferred)
            print(f"\n{len(inferred)} entries added.")
    else:
        print("\nAll champions covered.")

    print_status()


if __name__ == "__main__":
    main()
