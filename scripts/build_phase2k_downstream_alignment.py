#!/usr/bin/env python3
"""Build or validate the Phase 2K downstream semantic-target alignment packet.

The blank packet is scorer/model blind and is built only after every Phase 2K
gate required for downstream reruns has passed:

- the Phase 2K records must be a live build (no-provider mode is rejected);
- every D record must be GENERATED with a sealed semantic-polish subobject
  (placeholders and missing polish are rejected);
- the finalized human review artifacts must recompute to a PASSED review gate;
- the completed transformation audit must validate against the blank audit
  and records;
- the Phase 2J reviewed packet and candidate-coverage artifact must bind to
  the exact 311 KEEP endpoints with the versioned 263/48 boundary rule.

The packet never runs providers, never runs Phase 2F/2H, and never contains
downstream predictions or model results.  ``--validate-only`` revalidates an
existing packet against the same current sources without writing anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2k_downstream_alignment import (
    ALIGNMENT_PACKET_SCHEMA_VERSION,
    TARGET_COUNT,
    TARGET_WINDOW_COUNT,
    build_downstream_alignment_packet,
    load_json_strict,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEWED_PACKET = ROOT / "data/phase2j/reviewed-endpoint-annotation-packet-v1.json"
DEFAULT_COVERAGE = ROOT / "data/phase2j/candidate-coverage-v1.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build (or --validate-only) the scorer-blind Phase 2K downstream "
            "semantic-target alignment packet."
        ),
    )
    parser.add_argument(
        "--phase2k-dir", type=Path, required=True,
        help="Finalized live Phase 2K output directory.",
    )
    parser.add_argument(
        "--reviewed-packet", type=Path, default=DEFAULT_REVIEWED_PACKET,
        help="Phase 2J reviewed endpoint annotation packet (immutable input).",
    )
    parser.add_argument(
        "--coverage", type=Path, default=DEFAULT_COVERAGE,
        help="Phase 2J candidate coverage artifact (immutable input).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help=(
            "New alignment packet path (must not exist); with "
            "--validate-only, the existing packet path to revalidate."
        ),
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Revalidate an existing alignment packet against current sources.",
    )
    args = parser.parse_args(argv)

    for label, path in (
        ("phase2k output directory", args.phase2k_dir),
        ("Phase 2J reviewed packet", args.reviewed_packet),
        ("Phase 2J candidate coverage", args.coverage),
    ):
        if not Path(path).exists():
            parser.error(f"{label} does not exist: {path}")

    try:
        if args.validate_only:
            if not args.output.is_file():
                parser.error(
                    "--validate-only requires an existing packet path: "
                    f"{args.output}",
                )
            existing = load_json_strict(
                args.output,
                label="phase2k downstream alignment packet",
            )
            if existing.get("schema_version") != ALIGNMENT_PACKET_SCHEMA_VERSION:
                raise ValueError(
                    "existing alignment packet schema version is invalid",
                )
            fresh = build_downstream_alignment_packet(
                phase2k_dir=args.phase2k_dir,
                reviewed_packet_path=args.reviewed_packet,
                coverage_path=args.coverage,
            )
            if existing != fresh:
                raise ValueError(
                    "existing alignment packet does not match a fresh build "
                    "from the current sources",
                )
            packet = existing
        else:
            if args.output.exists():
                parser.error(
                    f"output path already exists; refusing to overwrite: "
                    f"{args.output}",
                )
            packet = build_downstream_alignment_packet(
                phase2k_dir=args.phase2k_dir,
                reviewed_packet_path=args.reviewed_packet,
                coverage_path=args.coverage,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    packet, sort_keys=True, indent=2, ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )
        print(json.dumps({
            "schema_version": ALIGNMENT_PACKET_SCHEMA_VERSION,
            "content_sha256": packet["content_sha256"],
            "release_gate": packet["release_gate"],
            "path": str(args.output),
            "window_count": TARGET_WINDOW_COUNT,
            "target_count": TARGET_COUNT,
            "item_count": len(packet["items"]),
            "validate_only": args.validate_only,
        }, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"[phase2k-alignment] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
