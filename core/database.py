"""SQLite database setup and insert/query helpers for videoSorter."""

import os
import sqlite3
import pathlib
from typing import Optional

# Override with DB_PATH env var to target a different database file.
# Used by the YouTube guide pipeline (guide_test.db) to avoid mixing
# unvalidated guide insights with the main coaching dataset.
DB_PATH = pathlib.Path(os.environ.get("DB_PATH", "videos.db"))


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets you access columns by name
    return conn


def init_db() -> None:
    """Create tables if they don't exist yet."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id          TEXT PRIMARY KEY,
                video_url         TEXT NOT NULL,
                video_title       TEXT,
                description       TEXT,
                game              TEXT DEFAULT 'lol',
                role              TEXT NOT NULL,
                subject           TEXT,
                champion          TEXT,
                rank              TEXT,
                website_rating    REAL,
                message_timestamp TEXT,
                status            TEXT DEFAULT 'pending',
                transcription     TEXT,
                created_at        TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS insights (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id      TEXT NOT NULL REFERENCES videos(video_id),
                insight_type  TEXT NOT NULL,
                text          TEXT NOT NULL,
                source_score  REAL DEFAULT NULL,
                cluster_score REAL DEFAULT NULL,
                confidence    REAL DEFAULT NULL,
                created_at    TEXT DEFAULT (datetime('now'))
            )
        """)
        # Add columns to existing DBs that predate this schema
        for col, typedef in [
            ("source_score",      "REAL DEFAULT NULL"),
            ("cluster_score",     "REAL DEFAULT NULL"),
            ("confidence",        "REAL DEFAULT NULL"),
            ("repetition_count",  "INTEGER DEFAULT 1"),
            ("is_duplicate",      "INTEGER DEFAULT 0"),
        ]:
            try:
                conn.execute(f"ALTER TABLE insights ADD COLUMN {col} {typedef}")
            except Exception:
                pass
        for col, typedef in [
            ("source", "TEXT DEFAULT 'discord'"),
            ("game", "TEXT DEFAULT 'lol'"),
            ("subject", "TEXT DEFAULT NULL"),
            ("website_rating", "REAL DEFAULT NULL"),
        ]:
            try:
                conn.execute(f"ALTER TABLE videos ADD COLUMN {col} {typedef}")
            except Exception:
                pass  # column already exists
        try:
            conn.execute(
                "UPDATE videos SET game = 'lol' WHERE game IS NULL OR TRIM(game) = ''"
            )
        except Exception:
            pass
        try:
            conn.execute(
                """
                UPDATE videos
                SET subject = champion
                WHERE (subject IS NULL OR TRIM(subject) = '')
                  AND champion IS NOT NULL
                  AND TRIM(champion) != ''
                """
            )
        except Exception:
            pass
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_descriptions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                role            TEXT NOT NULL,
                description     TEXT NOT NULL,
                message_timestamp TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS champion_archetypes (
                champion   TEXT NOT NULL,
                role       TEXT NOT NULL,
                archetype  TEXT NOT NULL,
                source     TEXT DEFAULT 'empirical',
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (champion, role)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS champion_abilities (
                champion      TEXT NOT NULL,
                ability_slot  TEXT NOT NULL,
                name          TEXT,
                description   TEXT,
                cooldown      TEXT,
                range         TEXT,
                cost          TEXT,
                properties    TEXT,
                PRIMARY KEY (champion, ability_slot)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS champion_stats (
                champion       TEXT PRIMARY KEY,
                hp             REAL, hp_level      REAL,
                armor          REAL, armor_level   REAL,
                mr             REAL, mr_level      REAL,
                attack_range   REAL,
                attack_damage  REAL, ad_level      REAL,
                attack_speed   REAL, as_level      REAL,
                movespeed      REAL,
                scraped_at     TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS champion_stat_notes (
                champion         TEXT NOT NULL,
                stat_key         TEXT NOT NULL,
                note             TEXT NOT NULL,
                z_score          REAL NOT NULL,
                comparison_group TEXT NOT NULL,
                PRIMARY KEY (champion, stat_key)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_queries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                question        TEXT NOT NULL,
                expected_answer TEXT,
                notes           TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_ratings (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id              INTEGER REFERENCES eval_queries(id),
                question              TEXT NOT NULL,
                intent_type           TEXT,
                champion_a            TEXT,
                champion_b            TEXT,
                answer_good           INTEGER NOT NULL,  -- 1=good, 0=bad
                confidence_aligned    INTEGER NOT NULL,  -- 1=aligned, 0=misaligned
                retrieved_insight_ids TEXT,
                generated_answer      TEXT,
                retrieval_method      TEXT DEFAULT 'rrf',
                shown_order           INTEGER,
                rated_at              TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()


def insert_video(
    video_id: str,
    video_url: str,
    role: str,
    message_timestamp: str,
    video_title: Optional[str] = None,
    description: Optional[str] = None,
    game: str = "lol",
    subject: Optional[str] = None,
    champion: Optional[str] = None,
    rank: Optional[str] = None,
    website_rating: float | None = None,
    source: str = "discord",
) -> None:
    """Insert a video row, ignoring duplicates (same video_id)."""
    if subject is None:
        subject = champion
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO videos
                (video_id, video_url, video_title, description, game, role, subject, champion, rank, website_rating, message_timestamp, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_id,
                video_url,
                video_title,
                description,
                game,
                role,
                subject,
                champion,
                rank,
                website_rating,
                message_timestamp,
                source,
            ),
        )
        conn.commit()


def insert_pending_description(role: str, description: str, message_timestamp: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO pending_descriptions (role, description, message_timestamp) VALUES (?, ?, ?)",
            (role, description, message_timestamp),
        )
        conn.commit()


def set_status(video_id: str, status: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE videos SET status = ? WHERE video_id = ?", (status, video_id))
        conn.commit()


def set_transcription(video_id: str, transcription: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE videos SET transcription = ?, status = 'transcribed' WHERE video_id = ?",
            (transcription, video_id),
        )
        conn.commit()


def insert_insight(
    video_id: str,
    insight_type: str,
    text: str,
    source_score: float | None = None,
    repetition_count: int = 1,
) -> int:
    """Insert an insight and return its row id."""
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO insights (video_id, insight_type, text, source_score, repetition_count) VALUES (?, ?, ?, ?, ?)",
            (video_id, insight_type, text, source_score, repetition_count),
        )
        conn.commit()
        return cur.lastrowid


def update_cluster_scores(scores: list[tuple[float, float, int]]) -> None:
    """
    Bulk-update cluster_score and confidence for a list of insights.
    scores: list of (cluster_score, confidence, insight_id)
    """
    with get_connection() as conn:
        conn.executemany(
            "UPDATE insights SET cluster_score = ?, confidence = ? WHERE id = ?",
            scores,
        )
        conn.commit()


def get_all_insights_with_embeddings() -> list:
    """Return all insights that have an embedding stored."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT id, video_id, insight_type, text, embedding, source_score
            FROM insights
            WHERE embedding IS NOT NULL
            """
        ).fetchall()


def get_videos_by_status(status: str) -> list:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM videos WHERE status = ?", (status,)
        ).fetchall()


def try_fill_descriptions() -> None:
    """
    For any video with no description, look for a pending_description
    in the same role within 2 hours of the video's message_timestamp.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT video_id, role, message_timestamp FROM videos WHERE description IS NULL OR description = ''"
        ).fetchall()

        for row in rows:
            match = conn.execute(
                """
                SELECT id, description FROM pending_descriptions
                WHERE role = ?
                  AND ABS(
                      strftime('%s', message_timestamp) - strftime('%s', ?)
                  ) <= 7200
                ORDER BY ABS(
                    strftime('%s', message_timestamp) - strftime('%s', ?)
                )
                LIMIT 1
                """,
                (row["role"], row["message_timestamp"], row["message_timestamp"]),
            ).fetchone()

            if match:
                conn.execute(
                    "UPDATE videos SET description = ? WHERE video_id = ?",
                    (match["description"], row["video_id"]),
                )
                conn.execute(
                    "DELETE FROM pending_descriptions WHERE id = ?",
                    (match["id"],),
                )

        conn.commit()
