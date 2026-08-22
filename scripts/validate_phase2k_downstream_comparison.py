#!/usr/bin/env python3
"""Validate an already-built Phase 2K downstream-comparison v2 JSON against a
finalized Phase 2K output directory.

The comparison is validated fail-closed against the exact generated records,
finalized human packet, human review summary, and (for live builds) the
completed transformation audit.  No downstream model, provider call, or
human review is run; the rows must already exist in the supplied JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2k_contextual_reconstruction import (
    OUTPUT_FILENAMES,
    load_json_strict,
)
from pipeline.phase2k_downstream_comparison import (
    DOWNSTREAM_COMPARISON_SCHEMA_VERSION,
    validate_downstream_comparison,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Phase 2K downstream-comparison v2 JSON.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--downstream-comparison", type=Path, required=True,
    )
    parser.add_argument(
        "--completed-audit", type=Path, default=None,
        help=(
            "Completed transformation-audit packet; defaults to the finalized "
            "audit inside the output directory for live builds."
        ),
    )
    args = parser.parse_args(argv)

    try:
        records = load_json_strict(
            args.output_dir / OUTPUT_FILENAMES["records"], label="records",
        )
        finalized = load_json_strict(
            args.output_dir / OUTPUT_FILENAMES["finalized_packet"],
            label="finalized human packet",
        )
        human_summary = load_json_strict(
            args.output_dir / OUTPUT_FILENAMES["human_summary"],
            label="human review summary",
        )
        comparison = load_json_strict(
            args.downstream_comparison, label="downstream comparison",
        )
        completed_audit = None
        if records.get("mode") == "live":
            audit_path = args.completed_audit or (
                args.output_dir / OUTPUT_FILENAMES["finalized_transformation_audit"]
            )
            completed_audit = load_json_strict(
                audit_path, label="completed transformation audit",
            )
        validated = validate_downstream_comparison(
            comparison,
            label="downstream comparison",
            records_obj=records,
            finalized_packet=finalized,
            human_summary=human_summary,
            completed_audit=completed_audit,
        )
        print(json.dumps({
            "schema_version": validated["schema_version"],
            "valid": True,
            "decision": validated["decision"],
            "diagnosis": validated["diagnosis"],
            "window_count": validated["dataset_binding"]["window_count"],
            "target_count": validated["semantic_target_contract"]["target_count"],
        }, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(
            f"[phase2k] downstream comparison validation failed: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
