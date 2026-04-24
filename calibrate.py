"""Compatibility wrapper for the moved calibration CLI.

Prefer:
    uv run python -m scripts.calibrate
"""

from scripts.calibrate import main


if __name__ == "__main__":
    main()
