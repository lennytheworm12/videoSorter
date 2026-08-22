#!/usr/bin/env python3
"""Import a completed Phase 2J adjudication export into a reviewed packet.

Reads the locked blank annotation packet, the locked selection manifest, the
original Human Pass A session, the generated adjudication packet, and the
completed ``phase2j-adjudication-export-v2`` REVIEW MATERIAL export, then
writes a separate reviewed canonical annotation packet.  Every input is
strictly loaded (duplicate-key rejection) and cross-validated; the blank
packet is never overwritten by default.

The build fails closed: invalid input or a conflicting deterministic output
leaves preexisting bytes untouched.  ``--validate-only`` validates an existing
reviewed output against the inputs without writing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2j_adjudication_import import (
    REVIEWED_PACKET_FILENAME,
    build_reviewed_packet,
    load_reviewed_packet,
    serialize_reviewed_packet,
    summarize_reviewed_packet,
)
from pipeline.phase2j_adjudication import load_adjudication_packet
from pipeline.phase2j_annotation_packet import load_annotation_packet
from pipeline.phase2j_source_selection import (
    file_sha256,
    load_selection_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET = ROOT / "data/phase2j/endpoint-annotation-packet-v1.json"
DEFAULT_MANIFEST = ROOT / "data/phase2j/window-selection-manifest-v1.json"
DEFAULT_HUMAN = Path(
    "/mnt/c/Users/bphan/Downloads/phase2j-review-session-3f766b08.json",
)
DEFAULT_ADJUDICATION = ROOT / "data/phase2j/phase2j-adjudication-packet-v1.json"
DEFAULT_OUTPUT = ROOT / "data/phase2j" / REVIEWED_PACKET_FILENAME


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _summary_for(reviewed, args: argparse.Namespace) -> dict:
    manifest = load_selection_manifest(args.manifest)
    adjudication = load_adjudication_packet(args.adjudication_packet)
    blank_packet = load_annotation_packet(args.packet, manifest=manifest)
    return summarize_reviewed_packet(
        reviewed,
        blank_packet_sha256=blank_packet["content_sha256"],
        manifest_sha256=manifest["content_sha256"],
        human_session_sha256=adjudication["human_session_sha256"],
        adjudication_packet_sha256=adjudication["content_sha256"],
        sol_review_sha256=adjudication["sol_review_sha256"],
        export_sha256=file_sha256(args.export),
    )


def _build(args: argparse.Namespace) -> int:
    reviewed = build_reviewed_packet(
        blank_packet_path=args.packet,
        manifest_path=args.manifest,
        human_session_path=args.human,
        adjudication_packet_path=args.adjudication_packet,
        export_path=args.export,
    )
    serialized = serialize_reviewed_packet(reviewed)
    output_path = Path(args.output)
    existing = (
        output_path.read_text(encoding="utf-8") if output_path.is_file() else None
    )
    if existing is not None and existing != serialized:
        raise ValueError(
            "phase2j preexisting reviewed packet does not match the "
            "deterministic import; refusing to overwrite",
        )
    _write_atomic(output_path, serialized)
    print(json.dumps(_summary_for(reviewed, args), sort_keys=True, indent=2))
    return 0


def _validate_only(args: argparse.Namespace) -> int:
    manifest = load_selection_manifest(args.manifest)
    reviewed = load_reviewed_packet(args.output, manifest=manifest)
    fresh = build_reviewed_packet(
        blank_packet_path=args.packet,
        manifest_path=args.manifest,
        human_session_path=args.human,
        adjudication_packet_path=args.adjudication_packet,
        export_path=args.export,
    )
    if Path(args.output).read_text(encoding="utf-8") != serialize_reviewed_packet(fresh):
        raise ValueError(
            "phase2j existing reviewed packet does not match the deterministic import",
        )
    print(json.dumps(_summary_for(reviewed, args), sort_keys=True, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import a completed Phase 2J adjudication export into a reviewed packet.",
    )
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument(
        "--adjudication-packet", type=Path, default=DEFAULT_ADJUDICATION,
    )
    parser.add_argument(
        "--export", type=Path, required=True,
        help="Completed phase2j-adjudication-export-v2 JSON (required).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=(
            "Reviewed packet output path (default: "
            "data/phase2j/reviewed-endpoint-annotation-packet-v1.json)."
        ),
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate the existing reviewed output and inputs without writing.",
    )
    args = parser.parse_args(argv)
    for label, path in (
        ("locked packet", args.packet),
        ("selection manifest", args.manifest),
        ("human session", args.human),
        ("adjudication packet", args.adjudication_packet),
        ("adjudication export", args.export),
    ):
        if not Path(path).is_file():
            parser.error(f"{label} input does not exist: {path}")
    try:
        if args.validate_only:
            return _validate_only(args)
        return _build(args)
    except (OSError, ValueError) as exc:
        print(f"[phase2j-import] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
