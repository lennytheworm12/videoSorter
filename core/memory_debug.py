"""Small process memory helpers for backend diagnostics."""

from __future__ import annotations

import logging
import os
import resource


def rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def debug_enabled() -> bool:
    return os.environ.get("DEBUG_MEMORY", "false").strip().lower() in {"1", "true", "yes"}


def log_memory(label: str) -> None:
    if not debug_enabled():
        return
    logging.warning("[memory] %s rss=%.1fMB", label, rss_mb())
