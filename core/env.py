"""Environment loading helpers for local development."""

from __future__ import annotations

import pathlib

from dotenv import load_dotenv


_LOADED = False


def project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def load_project_env() -> None:
    """Load local `.env` files once per process."""
    global _LOADED
    if _LOADED:
        return
    root = project_root()
    load_dotenv(root / ".env", override=False)
    load_dotenv(root / ".env.local", override=False)
    _LOADED = True
