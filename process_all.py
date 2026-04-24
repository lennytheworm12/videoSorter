"""Compatibility wrapper for the moved pipeline CLI.

Prefer:
    uv run python -m scripts.process_all
"""

from scripts.process_all import main


if __name__ == "__main__":
    main()
