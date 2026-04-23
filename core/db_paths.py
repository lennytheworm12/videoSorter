"""Helpers for the project's primary and secondary SQLite database paths."""

from __future__ import annotations

import os
import pathlib

PRIMARY_DB_NAME = "videos.db"
DEFAULT_KNOWLEDGE_DB_NAME = "knowledge.db"
LEGACY_KNOWLEDGE_DB_NAME = "guide_test.db"


def primary_db_path() -> pathlib.Path:
    return pathlib.Path(PRIMARY_DB_NAME)


def knowledge_db_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get("KNOWLEDGE_DB_PATH", DEFAULT_KNOWLEDGE_DB_NAME))


def migrate_legacy_knowledge_db() -> pathlib.Path:
    """
    Rename the legacy secondary DB to the canonical knowledge DB name.

    This is intentionally conservative:
      - if `knowledge.db` already exists, leave both files untouched
      - if a non-default KNOWLEDGE_DB_PATH is configured, migrate into that path
      - if no legacy DB exists, do nothing
    """
    target = knowledge_db_path()
    legacy = pathlib.Path(LEGACY_KNOWLEDGE_DB_NAME)
    if target != legacy and not target.exists() and legacy.exists():
        legacy.replace(target)
    return target


def activate_knowledge_db() -> pathlib.Path:
    """
    Set DB_PATH to the secondary knowledge DB unless the caller already set it.
    """
    if "DB_PATH" in os.environ:
        return pathlib.Path(os.environ["DB_PATH"])
    target = migrate_legacy_knowledge_db()
    os.environ["DB_PATH"] = str(target)
    return target


def all_content_db_paths() -> list[str]:
    target = migrate_legacy_knowledge_db()
    ordered = [str(primary_db_path()), str(target)]
    seen: set[str] = set()
    result: list[str] = []
    for path in ordered:
        if path not in seen:
            result.append(path)
            seen.add(path)
    return result
