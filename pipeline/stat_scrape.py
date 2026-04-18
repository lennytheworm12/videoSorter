"""
Pull champion base stats from Riot Data Dragon and compute anomaly notes.

For each champion, compares key stats (range, HP, armor, AD, movespeed,
scaling) against their role peer group. Stats that are ≥1.5 std devs
from the peer mean get human-readable notes stored in champion_stat_notes.

Range notes are context-aware: a melee champion with high mobility (Zed,
Akali, Katarina) does not get a "low range" warning because they close gaps
with dashes/blinks. Only immobile low-range champions get the note.

These notes are injected into matchup/synergy query responses so the LLM
can reason about early fragility, range mismatches, and scaling edges.

Usage:
    uv run python -m pipeline.stat_scrape           # full pull + compute
    uv run python -m pipeline.stat_scrape --status  # show anomaly summary
    uv run python -m pipeline.stat_scrape --notes "Caitlyn"   # show notes for one champ
"""

import json
import statistics
import argparse
import urllib.request
from core.database import get_connection, init_db
from core.champions import load_champion_names

DDRAGON_VERSIONS  = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_CHAMPIONS = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"

# Threshold: |z_score| >= this → generate a note
Z_THRESH = 1.5

# Champions whose range stat needs special context because the base stat
# doesn't reflect their actual gameplay pattern.
# Note: always shown regardless of z-score.
RANGE_SPECIAL_CASES = {
    "Kayle":    (
        "starts melee (range 175) — transforms to ranged at level 6 and AOE at level 11. "
        "She is extremely fragile and range-disadvantaged before level 6; safe farming and "
        "avoiding extended early trades is critical to reaching her power spikes."
    ),
    "Jayce":    "alternates melee (range 125) and ranged (range 500) forms — his effective range depends on stance.",
    "Nidalee":  "alternates melee (range 125) and ranged (range 525) forms — human form range applies in poke patterns.",
    "Gnar":     "alternates ranged (Mini Gnar, range 400–700) and melee (Mega Gnar) forms based on rage.",
    "Elise":    "alternates ranged (range 575) and melee spider forms — human form range is the trading range.",
}

# Stats to evaluate for anomalies: (db_column, human_label)
ANOMALY_CHECKS: list[tuple[str, str]] = [
    ("attack_range",  "attack range"),
    ("hp",            "base HP"),
    ("armor",         "base armor"),
    ("attack_damage", "base attack damage"),
    ("movespeed",     "base movement speed"),
    ("hp_level",      "HP-per-level scaling"),
    ("armor_level",   "armor-per-level scaling"),
    ("ad_level",      "AD-per-level scaling"),
]


# ── Data Dragon ───────────────────────────────────────────────────────────────

def _get_version() -> str:
    with urllib.request.urlopen(DDRAGON_VERSIONS, timeout=5) as r:
        return json.loads(r.read())[0]


def _fetch_all_stats(version: str) -> dict[str, dict]:
    """Fetch champion summary list → {canonical_name: stats_dict}."""
    url = DDRAGON_CHAMPIONS.format(version=version)
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read())
    return {v["name"]: v["stats"] for champ_id, v in data["data"].items()}


# ── DB helpers ────────────────────────────────────────────────────────────────

def upsert_stats(rows: list[tuple]) -> None:
    with get_connection() as conn:
        conn.executemany("""
            INSERT INTO champion_stats
                (champion, hp, hp_level, armor, armor_level, mr, mr_level,
                 attack_range, attack_damage, ad_level, attack_speed, as_level, movespeed)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(champion) DO UPDATE SET
                hp=excluded.hp, hp_level=excluded.hp_level,
                armor=excluded.armor, armor_level=excluded.armor_level,
                mr=excluded.mr, mr_level=excluded.mr_level,
                attack_range=excluded.attack_range,
                attack_damage=excluded.attack_damage, ad_level=excluded.ad_level,
                attack_speed=excluded.attack_speed, as_level=excluded.as_level,
                movespeed=excluded.movespeed,
                scraped_at=datetime('now')
        """, rows)
        conn.commit()


def upsert_notes(rows: list[tuple]) -> None:
    with get_connection() as conn:
        conn.executemany("""
            INSERT INTO champion_stat_notes (champion, stat_key, note, z_score, comparison_group)
            VALUES (?,?,?,?,?)
            ON CONFLICT(champion, stat_key) DO UPDATE SET
                note=excluded.note, z_score=excluded.z_score,
                comparison_group=excluded.comparison_group
        """, rows)
        conn.commit()


# ── Mobility context ──────────────────────────────────────────────────────────

def _build_mobility_map() -> dict[str, bool]:
    """
    Returns {champion: has_mobility} where has_mobility = True if the champion
    has any dash, blink, or teleport tagged in champion_abilities.
    Mobile champions skip the "low range" warning since they close gaps by design.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT champion, properties FROM champion_abilities WHERE properties IS NOT NULL"
        ).fetchall()
    mobility_tags = {"dash", "blink", "teleport"}
    has_mobility: dict[str, bool] = {}
    for r in rows:
        try:
            props = set(json.loads(r["properties"]))
        except Exception:
            props = set()
        if props & mobility_tags:
            has_mobility[r["champion"]] = True
    return has_mobility


def _build_scaling_map() -> dict[str, bool]:
    """
    Returns {champion: is_high_scaler} based on hp_level z-score vs all champions.
    High-scaling champions get extra emphasis in low-range notes.
    """
    with get_connection() as conn:
        rows = conn.execute("SELECT champion, hp_level FROM champion_stats WHERE hp_level IS NOT NULL").fetchall()
    vals = [r["hp_level"] for r in rows]
    if len(vals) < 3:
        return {}
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals)
    result = {}
    for r in rows:
        z = (r["hp_level"] - mean) / stdev if stdev > 0 else 0.0
        result[r["champion"]] = z > 1.0
    return result


# ── Peer groups ───────────────────────────────────────────────────────────────

def _build_role_peer_groups() -> dict[str, list[str]]:
    """
    Group champions by role for range + armor comparison (more meaningful than archetype
    since range is a gameplay decision per role, not per playstyle).
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT champion, role FROM champion_archetypes"
        ).fetchall()
    groups: dict[str, list[str]] = {}
    for r in rows:
        groups.setdefault(r["role"], []).append(r["champion"])
    return groups


def _build_archetype_peer_groups() -> dict[str, list[str]]:
    """
    Group by archetype for HP/armor/scaling comparisons (same strategic class).
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT champion, archetype FROM champion_archetypes"
        ).fetchall()
    groups: dict[str, list[str]] = {}
    for r in rows:
        groups.setdefault(r["archetype"], []).append(r["champion"])
    return groups


def _build_primary_role_map() -> dict[str, str]:
    """
    Returns {champion: primary_role} where primary role is determined by:
    1. Role with most videos in our DB (reflects actual coaching coverage)
    2. Fallback: first role in champion_archetypes
    """
    with get_connection() as conn:
        video_rows = conn.execute("""
            SELECT champion, role, COUNT(*) as cnt
            FROM videos
            WHERE champion IS NOT NULL AND role NOT IN ('ability_enrichment')
            GROUP BY champion, role
        """).fetchall()
        arch_rows = conn.execute(
            "SELECT DISTINCT champion, role FROM champion_archetypes"
        ).fetchall()

    # Build video-based primary role (most videos wins)
    video_counts: dict[str, dict[str, int]] = {}
    for r in video_rows:
        video_counts.setdefault(r["champion"], {})[r["role"]] = r["cnt"]

    # Build fallback from archetypes (just use first)
    arch_fallback: dict[str, str] = {}
    for r in arch_rows:
        if r["champion"] not in arch_fallback:
            arch_fallback[r["champion"]] = r["role"]

    result: dict[str, str] = {}
    all_champs = {r["champion"] for r in arch_rows}
    for champ in all_champs:
        if champ in video_counts:
            result[champ] = max(video_counts[champ], key=video_counts[champ].get)
        elif champ in arch_fallback:
            result[champ] = arch_fallback[champ]
    return result


# ── Z-score helpers ───────────────────────────────────────────────────────────

def _z(value: float, values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    return (value - mean) / stdev if stdev > 0 else 0.0


def _pct(value: float, mean: float) -> str:
    if mean == 0:
        return ""
    diff = round((value - mean) / mean * 100)
    return f"+{diff}%" if diff >= 0 else f"{diff}%"


# ── Note generation ───────────────────────────────────────────────────────────

def _range_note(
    champion: str,
    value: float,
    peer_vals: list[float],
    group_name: str,
    has_mob: bool,
    is_scaler: bool,
) -> str | None:
    """
    Generate a range note.

    Low-range notes are context-dependent (depends entirely on the opponent) and
    produce too many false positives — a melee assassin with low range is by design,
    a utility mage with a ranged poke ability doesn't care about auto range, etc.

    We only generate:
      1. Special case notes (Kayle, Jayce, etc.) — mechanically specific facts
      2. High range anomalies — always an advantage regardless of matchup
    """
    if champion in RANGE_SPECIAL_CASES:
        return RANGE_SPECIAL_CASES[champion]

    z = _z(value, peer_vals)
    if z < Z_THRESH:
        return None

    # High range only
    mean = statistics.mean(peer_vals)
    pct = _pct(value, mean)
    return (
        f"{champion} has notably high attack range ({int(value)} vs "
        f"{group_name} avg {int(mean)}, {pct}) — can auto-attack from outside "
        f"most opponents' reach, making auto trades and poke heavily one-sided."
    )


def _stat_note(
    champion: str,
    col: str,
    label: str,
    value: float,
    peer_vals: list[float],
    group_name: str,
) -> str | None:
    """Generate anomaly note for non-range stats."""
    z = _z(value, peer_vals)
    if abs(z) < Z_THRESH:
        return None

    mean = statistics.mean(peer_vals)
    pct = _pct(value, mean)
    hi = z > 0

    if col == "hp":
        if not hi:
            return (
                f"{champion} has below-average base HP ({int(value)} vs "
                f"{group_name} avg {int(mean)}, {pct}) — early burst and all-ins "
                f"are more effective before their scaling or itemization compensates."
            )
        else:
            return (
                f"{champion} has high base HP ({int(value)} vs {group_name} avg "
                f"{int(mean)}, {pct}) — burst windows are shorter; favour sustained "
                f"damage or percent-health effects if available."
            )

    if col == "armor":
        if not hi:
            return (
                f"{champion} has very low base armor ({round(value,1)} vs "
                f"{group_name} avg {round(mean,1)}, {pct}) — physical damage "
                f"and AD assassins deal disproportionately more early."
            )
        else:
            return (
                f"{champion} has high base armor ({round(value,1)} vs {group_name} "
                f"avg {round(mean,1)}, {pct}) — AD damage early is less effective; "
                f"prioritise magic damage or armor penetration."
            )

    if col == "movespeed":
        if hi:
            return (
                f"{champion} has above-average base movement speed ({int(value)} vs "
                f"{group_name} avg {int(mean)}, {pct}) — harder to kite and easier "
                f"for them to follow up after CC or gap-closers."
            )
        else:
            return (
                f"{champion} has below-average base movement speed ({int(value)} vs "
                f"{group_name} avg {int(mean)}, {pct}) — more susceptible to kiting "
                f"and easier to disengage from in extended fights."
            )

    if col == "hp_level":
        if hi:
            return (
                f"{champion} has high HP-per-level scaling ({round(value,1)} vs "
                f"{group_name} avg {round(mean,1)}, {pct}) — becomes significantly "
                f"tankier as the game progresses; snowball early or deny scaling."
            )
        else:
            return (
                f"{champion} has low HP-per-level scaling ({round(value,1)} vs "
                f"{group_name} avg {round(mean,1)}, {pct}) — stays relatively "
                f"squishy even at higher levels."
            )

    if col == "ad_level":
        if hi:
            return (
                f"{champion} has high base AD scaling ({round(value,1)} per level vs "
                f"{group_name} avg {round(mean,1)}, {pct}) — their auto-attacks "
                f"scale significantly; prioritise ability-based burst before autos dominate."
            )
        else:
            return (
                f"{champion} has low base AD scaling ({round(value,1)} per level vs "
                f"{group_name} avg {round(mean,1)}, {pct}) — relies more on "
                f"abilities than autos for late-game damage output."
            )

    if col == "armor_level":
        if hi:
            return (
                f"{champion} has high armor-per-level ({round(value,1)} vs "
                f"{group_name} avg {round(mean,1)}, {pct}) — becomes progressively "
                f"harder to burst with physical damage."
            )
        else:
            return (
                f"{champion} has low armor-per-level ({round(value,1)} vs "
                f"{group_name} avg {round(mean,1)}, {pct}) — remains vulnerable "
                f"to AD damage even into the late game."
            )

    if col == "attack_damage":
        if hi:
            return (
                f"{champion} has high base attack damage ({round(value,1)} vs "
                f"{group_name} avg {round(mean,1)}, {pct}) — their early autos "
                f"hit noticeably harder than expected."
            )
        else:
            return (
                f"{champion} has low base attack damage ({round(value,1)} vs "
                f"{group_name} avg {round(mean,1)}, {pct}) — early auto trades "
                f"are less threatening than typical."
            )

    # Generic fallback
    direction = "high" if hi else "low"
    return (
        f"{champion} has {direction} {label} ({round(value,1)} vs "
        f"{group_name} avg {round(mean,1)}, {pct})."
    )


# ── Compute all notes ─────────────────────────────────────────────────────────

def compute_stat_notes() -> None:
    """Compute anomaly notes for all champions and write to champion_stat_notes."""
    # Context maps
    mobility_map = _build_mobility_map()
    scaling_map = _build_scaling_map()
    role_groups = _build_role_peer_groups()      # for range
    arch_groups = _build_archetype_peer_groups() # for other stats
    primary_role_map = _build_primary_role_map() # video-coverage-based primary role

    # Build champion → archetypes list (for fallback)
    with get_connection() as conn:
        arch_rows = conn.execute(
            "SELECT DISTINCT champion, archetype FROM champion_archetypes"
        ).fetchall()
    champ_to_archetypes: dict[str, list[str]] = {}
    for r in arch_rows:
        champ_to_archetypes.setdefault(r["champion"], []).append(r["archetype"])

    # Load all stats
    with get_connection() as conn:
        stat_rows = conn.execute("SELECT * FROM champion_stats").fetchall()
    stats_by_champ: dict[str, dict] = {r["champion"]: dict(r) for r in stat_rows}

    # Pre-build value lists per group for each stat col
    # range: primary role group (video-coverage based); other stats: archetype group
    def _peer_values(col: str, champ: str) -> tuple[list[float], str]:
        """Return (peer_value_list, group_label) for comparison."""
        if col == "attack_range":
            role = primary_role_map.get(champ)
            if role and role in role_groups:
                peers = [
                    stats_by_champ[c][col]
                    for c in role_groups[role]
                    if c in stats_by_champ and stats_by_champ[c].get(col) is not None
                ]
                return peers, role.upper()
        else:
            archs = champ_to_archetypes.get(champ, [])
            best_arch = max(archs, key=lambda a: len(arch_groups.get(a, [])), default=None)
            if best_arch:
                peers = [
                    stats_by_champ[c][col]
                    for c in arch_groups.get(best_arch, [])
                    if c in stats_by_champ and stats_by_champ[c].get(col) is not None
                ]
                return peers, best_arch
        # fallback: all champions
        peers = [s[col] for s in stats_by_champ.values() if s.get(col) is not None]
        return peers, "all champions"

    note_rows = []
    for champ, s in stats_by_champ.items():
        has_mob = mobility_map.get(champ, False)
        is_scaler = scaling_map.get(champ, False)

        for col, label in ANOMALY_CHECKS:
            val = s.get(col)
            if val is None:
                continue

            peer_vals, group_label = _peer_values(col, champ)

            if col == "attack_range":
                note = _range_note(champ, val, peer_vals, group_label, has_mob, is_scaler)
            else:
                note = _stat_note(champ, col, label, val, peer_vals, group_label)

            if note:
                z = _z(val, peer_vals)
                note_rows.append((champ, col, note, round(z, 3), group_label))

    # Delete stale notes before re-inserting
    with get_connection() as conn:
        conn.execute("DELETE FROM champion_stat_notes")
        conn.commit()
    upsert_notes(note_rows)
    print(f"Computed {len(note_rows)} stat anomaly notes across {len(stats_by_champ)} champions.")


# ── Status / inspection ───────────────────────────────────────────────────────

def print_notes(champion: str) -> None:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT stat_key, z_score, note FROM champion_stat_notes WHERE champion = ? ORDER BY ABS(z_score) DESC",
            (champion,)
        ).fetchall()
    if not rows:
        print(f"No anomaly notes for {champion}.")
        return
    print(f"\nStat notes for {champion}:")
    for r in rows:
        print(f"  [{r['stat_key']:16} z={r['z_score']:+.2f}] {r['note']}")
    print()


def print_status() -> None:
    with get_connection() as conn:
        n_champs = conn.execute("SELECT COUNT(*) FROM champion_stats").fetchone()[0]
        n_notes = conn.execute("SELECT COUNT(*) FROM champion_stat_notes").fetchone()[0]
        top = conn.execute("""
            SELECT stat_key, COUNT(*) as cnt
            FROM champion_stat_notes GROUP BY stat_key ORDER BY cnt DESC
        """).fetchall()
        extreme = conn.execute("""
            SELECT champion, stat_key, z_score, note
            FROM champion_stat_notes ORDER BY ABS(z_score) DESC LIMIT 10
        """).fetchall()

    print(f"\nchampion_stats: {n_champs} champions")
    print(f"champion_stat_notes: {n_notes} anomalies\n")
    print("Anomalies by stat:")
    for r in top:
        print(f"  {r['stat_key']:<20} {r['cnt']}")
    print("\nMost extreme anomalies:")
    for r in extreme:
        print(f"  [{r['stat_key']:16} z={r['z_score']:+.2f}] {r['champion']}: {r['note'][:90]}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pull champion stats and compute anomaly notes")
    parser.add_argument("--status", action="store_true", help="Show anomaly summary")
    parser.add_argument("--notes", metavar="CHAMPION", help="Show anomaly notes for one champion")
    args = parser.parse_args()

    init_db()

    if args.status:
        print_status()
        return

    if args.notes:
        print_notes(args.notes)
        return

    print("Fetching Data Dragon version…")
    version = _get_version()
    print(f"Version: {version}")

    print("Fetching all champion stats…")
    all_stats = _fetch_all_stats(version)
    canonical_names = set(load_champion_names())

    rows = []
    for name, s in all_stats.items():
        if name not in canonical_names:
            continue
        rows.append((
            name,
            s.get("hp"), s.get("hpperlevel"),
            s.get("armor"), s.get("armorperlevel"),
            s.get("spellblock"), s.get("spellblockperlevel"),
            s.get("attackrange"),
            s.get("attackdamage"), s.get("attackdamageperlevel"),
            s.get("attackspeed"), s.get("attackspeedperlevel"),
            s.get("movespeed"),
        ))

    upsert_stats(rows)
    print(f"Stored stats for {len(rows)} champions.")

    print("Computing anomaly notes…")
    compute_stat_notes()
    print_status()


if __name__ == "__main__":
    main()
