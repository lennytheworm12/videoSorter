"""SQLite database setup and insert/query helpers for videoSorter."""

import sqlite3
import pathlib
from typing import Optional

DB_PATH = pathlib.Path("videos.db")


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
                role              TEXT NOT NULL,
                champion          TEXT,
                rank              TEXT,
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
                pass  # column already exists
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
                champion   TEXT PRIMARY KEY,
                archetype  TEXT NOT NULL,
                source     TEXT DEFAULT 'empirical',
                created_at TEXT DEFAULT (datetime('now'))
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
    champion: Optional[str] = None,
    rank: Optional[str] = None,
) -> None:
    """Insert a video row, ignoring duplicates (same video_id)."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO videos
                (video_id, video_url, video_title, description, role, champion, rank, message_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (video_id, video_url, video_title, description, role, champion, rank, message_timestamp),
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
