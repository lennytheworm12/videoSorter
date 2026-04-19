"""
Transcribe pending videos in guide_test.db.

Thin wrapper around pipeline.transcribe that targets the guide test database
instead of the main videos.db.

Usage:
    uv run python -m pipeline.guide_transcribe
"""

import os
os.environ.setdefault("DB_PATH", "guide_test.db")

from pipeline.transcribe import run

if __name__ == "__main__":
    run()
