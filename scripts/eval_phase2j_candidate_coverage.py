#!/usr/bin/env python3
"""Emit the Phase 2J frozen candidate-coverage artifact.

Reads the locked selection manifest and the reviewed scorer-blind annotation
packet, regenerates every frozen mention candidate catalog with the exact
Bronze source/window contract, maps every gold-eligible endpoint to its exact
local Bronze candidate span, and writes a self-verifying deterministic
candidate-coverage artifact.  This is discovery coverage only: no model is
scored, no syntax is parsed, nothing is tuned, and candidate generation is
never modified.

The build fails closed: invalid input or a conflicting deterministic output
leaves preexisting bytes untouched.  ``--validate-only`` validates an existing
coverage artifact against the inputs without writing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2j_candidate_coverage import (
    COVERAGE_FILENAME,
    build_candidate_coverage,
    load_candidate_coverage,
    serialize_candidate_coverage,
    summarize_candidate_coverage,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/phase2j/window-selection-manifest-v1.json"
DEFAULT_PACKET = ROOT / "data/phase2j/reviewed-endpoint-annotation-packet-v1.json"
DEFAULT_OUTPUT = ROOT / "data/phase2j" / COVERAGE_FILENAME


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build(args: argparse.Namespace) -> int:
    artifact = build_candidate_coverage(
        manifest_path=args.manifest, packet_path=args.packet,
    )
    serialized = serialize_candidate_coverage(artifact)
    output_path = Path(args.output)
    existing = (
        output_path.read_text(encoding="utf-8") if output_path.is_file() else None
    )
    if existing is not None and existing != serialized:
        raise ValueError(
            "phase2j preexisting candidate coverage does not match the "
            "deterministic build; refusing to overwrite",
        )
    _write_atomic(output_path, serialized)
    print(json.dumps(summarize_candidate_coverage(artifact), sort_keys=True, indent=2))
    return 0


def _validate_only(args: argparse.Namespace) -> int:
    artifact = load_candidate_coverage(
        args.output, manifest_path=args.manifest, packet_path=args.packet,
    )
    if Path(args.output).read_text(encoding="utf-8") \
            != serialize_candidate_coverage(artifact):
        raise ValueError(
            "phase2j existing candidate coverage does not match the "
            "deterministic artifact",
        )
    print(json.dumps(summarize_candidate_coverage(artifact), sort_keys=True, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit the Phase 2J frozen candidate-coverage artifact.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help="Locked phase2j selection manifest (default: data/phase2j/...).",
    )
    parser.add_argument(
        "--packet", type=Path, default=DEFAULT_PACKET,
        help="Reviewed phase2j annotation packet (default: data/phase2j/...).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=(
            "Candidate coverage output path (default: "
            "data/phase2j/candidate-coverage-v1.json)."
        ),
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate the existing coverage output and inputs without writing.",
    )
    args = parser.parse_args(argv)
    for label, path in (
        ("selection manifest", args.manifest),
        ("reviewed packet", args.packet),
    ):
        if not Path(path).is_file():
            parser.error(f"{label} input does not exist: {path}")
    try:
        if args.validate_only:
            return _validate_only(args)
        return _build(args)
    except (OSError, ValueError) as exc:
        print(f"[phase2j-coverage] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
