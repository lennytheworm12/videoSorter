"""Compatibility wrapper for the moved retry CLI.

Prefer:
    uv run python -m scripts.retry_no_transcript
"""

from scripts.retry_no_transcript import main


if __name__ == "__main__":
    main()
