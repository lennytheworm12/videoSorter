"""Helpers for building reusable prompt templates."""

from __future__ import annotations


def json_schema_block(keys: tuple[str, ...]) -> str:
    body = ",\n".join(f'        "{key}": []' for key in keys)
    return "{\n" + body + "\n    }"


def escape_format_braces(text: str) -> str:
    """Escape literal braces for a later .format() call."""
    return text.replace("{", "{{").replace("}", "}}")

