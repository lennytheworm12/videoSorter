#!/usr/bin/env python3
"""Apply human alignment decisions to the blank Phase 2K alignment packet.

This CLI imports a compact decisions JSON mapping every ``alignment_id`` to
``state``/``polished_spans``/``reviewer``/``completed_at``/``notes``, fails
closed on partial maps, invalid states, invalid spans, duplicate spans,
cross-target duplicate spans, and content tampering, then writes a finalized
packet plus the deterministic summary to new caller-selected paths.  No
decision is ever inferred or fabricated.

The finalized packet is cryptographically rebound to the current live Phase
2K output and Phase 2J sources (``--phase2k-dir`` plus the reviewed packet
and coverage artifact): both the blank packet and the finalized packet are
validated against those sources before anything is written, so a canonical
but forged packet is rejected.
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
    ALIGNMENT_SUMMARY_SCHEMA_VERSION,
    build_alignment_summary,
    finalize_downstream_alignment_packet,
    load_alignment_inputs,
    load_json_strict,
    validate_alignment_summary,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEWED_PACKET = ROOT / "data/phase2j/reviewed-endpoint-annotation-packet-v1.json"
DEFAULT_COVERAGE = ROOT / "data/phase2j/candidate-coverage-v1.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the Phase 2K downstream semantic-target alignment "
            "packet with compact human decisions."
        ),
    )
    parser.add_argument(
        "--phase2k-dir", type=Path, required=True,
        help=(
            "Finalized live Phase 2K output directory that produced the "
            "blank packet."
        ),
    )
    parser.add_argument(
        "--reviewed-packet", type=Path, default=DEFAULT_REVIEWED_PACKET,
        help="Phase 2J reviewed endpoint annotation packet (immutable input).",
    )
    parser.add_argument(
        "--coverage", type=Path, default=DEFAULT_COVERAGE,
        help="Phase 2J candidate coverage artifact (immutable input).",
    )
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)

    for label, path in (
        ("phase2k output directory", args.phase2k_dir),
        ("Phase 2J reviewed packet", args.reviewed_packet),
        ("Phase 2J candidate coverage", args.coverage),
        ("blank alignment packet", args.packet),
        ("decisions", args.decisions),
    ):
        if not Path(path).exists():
            parser.error(f"{label} does not exist: {path}")
    for label, path in (
        ("finalized alignment packet", args.output),
        ("alignment summary", args.summary),
    ):
        if Path(path).exists():
            parser.error(
                f"{label} output path already exists; refusing to overwrite: "
                f"{path}",
            )

    try:
        inputs = load_alignment_inputs(
            phase2k_dir=args.phase2k_dir,
            reviewed_packet_path=args.reviewed_packet,
            coverage_path=args.coverage,
        )
        packet = load_json_strict(
            args.packet, label="phase2k blank downstream alignment packet",
        )
        if packet.get("schema_version") != ALIGNMENT_PACKET_SCHEMA_VERSION:
            raise ValueError("blank alignment packet schema version is invalid")
        decisions = load_json_strict(
            args.decisions, label="phase2k alignment decisions",
        )
        for key, value in decisions.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                raise ValueError(
                    "alignment decisions must map alignment IDs to objects",
                )
        finalized = finalize_downstream_alignment_packet(
            packet, decisions, bindings=inputs,
        )
        summary = build_alignment_summary(finalized)
        validate_alignment_summary(summary, finalized=finalized)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                finalized, sort_keys=True, indent=2, ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        args.summary.write_text(
            json.dumps(
                summary, sort_keys=True, indent=2, ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "schema_version": ALIGNMENT_PACKET_SCHEMA_VERSION,
            "summary_schema_version": ALIGNMENT_SUMMARY_SCHEMA_VERSION,
            "finalized_packet": str(args.output),
            "summary": str(args.summary),
            "content_sha256": finalized["content_sha256"],
            "release_gate": finalized["release_gate"],
            "item_count": len(finalized["items"]),
            "total": summary["total"],
        }, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"[phase2k-alignment] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
