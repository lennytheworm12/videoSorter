"""Compatibility wrapper for the moved evaluation CLI.

Prefer:
    uv run python -m scripts.eval
"""

from scripts.eval import main


if __name__ == "__main__":
    main()
