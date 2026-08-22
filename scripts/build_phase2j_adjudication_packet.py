#!/usr/bin/env python3
"""Build the Phase 2J post-Pass-A human-vs-Sol adjudication packet.

Reads the locked Phase 2J annotation packet, the completed human Pass A review
session, and the sealed independent Sol navigation/audit review, then writes a
deterministic sanitized adjudication packet to ``data/phase2j/``.  The packet
contains only what the human adjudication UI needs: the locked Bronze windows,
sanitized Human/Sol endpoint alternatives, connected-component classifications,
input hashes, and schema versions.  It contains no model scores, predictions,
ranks, candidate data, reviewer identity, or packet-internal machine fields.

Sol proposals are a second opinion and are never auto-promoted to gold.  The
adjudication output remains REVIEW MATERIAL until a separately validated
canonical import/finalizer runs.

Existing outputs are validated against the fresh deterministic build and the
build fails closed (no writes) on any mismatch.  ``--validate-only`` performs
the same strict validation without writing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2j_adjudication import (
    build_adjudication_packet,
    load_adjudication_packet,
    validate_adjudication_packet,
)
from pipeline.phase2j_annotation_packet import load_annotation_packet
from pipeline.phase2j_source_selection import canonical_sha256, file_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET = ROOT / "data/phase2j/endpoint-annotation-packet-v1.json"
DEFAULT_HUMAN = Path(
    "/mnt/c/Users/bphan/Downloads/phase2j-review-session-3f766b08.json",
)
DEFAULT_SOL = Path("/tmp/phase2j-sol-high-independent-review-v1.json")
DEFAULT_OUTPUT_DIR = ROOT / "data" / "phase2j"
OUTPUT_FILENAME = "phase2j-adjudication-packet-v1.json"


def _serialize(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_inputs(
    *,
    packet_path: Path,
    human_path: Path,
    sol_path: Path,
) -> tuple[dict, dict, dict]:
    packet = load_annotation_packet(packet_path)
    human = load_adjudication_packet(human_path)
    sol = load_adjudication_packet(sol_path)
    # Rebuild the human/Sol objects strictly before passing them to the builder;
    # the builder itself re-validates every field against the locked packet.
    return packet, human, sol


def _build_fresh(
    *,
    packet_path: Path,
    human_path: Path,
    sol_path: Path,
) -> tuple[dict, str]:
    packet, human, sol = _load_inputs(
        packet_path=packet_path, human_path=human_path, sol_path=sol_path,
    )
    built = build_adjudication_packet(
        packet,
        human,
        sol,
        human_session_path=human_path,
        sol_review_path=sol_path,
    )
    validate_adjudication_packet(built)
    return built, _serialize(built)


def _summary(packet: dict) -> dict:
    totals = packet["totals"]
    return {
        "schema_version": packet["schema_version"],
        "content_sha256": packet["content_sha256"],
        "packet_sha256": packet["packet_sha256"],
        "human_session_sha256": packet["human_session_sha256"],
        "sol_review_sha256": packet["sol_review_sha256"],
        "windows": totals["windows"],
        "components": totals["components"],
        "exact_agreements": totals["exact_agreements"],
        "type_disagreements": totals["type_disagreements"],
        "boundary_disagreements": totals["boundary_disagreements"],
        "sol_only": totals["sol_only"],
        "human_only": totals["human_only"],
        "human_endpoints": totals["human_endpoints"],
        "sol_endpoints": totals["sol_endpoints"],
        "visibility_gate": packet["visibility_gate"],
    }


def _build(args: argparse.Namespace) -> int:
    packet, serialized = _build_fresh(
        packet_path=args.packet,
        human_path=args.human,
        sol_path=args.sol,
    )
    output_path = args.output_dir / args.output_name
    existing = (
        output_path.read_text(encoding="utf-8") if output_path.is_file() else None
    )
    if existing is not None and existing != serialized:
        raise ValueError(
            "phase2j preexisting adjudication packet does not match the "
            "deterministic build; refusing to overwrite",
        )
    _write_atomic(output_path, serialized)
    print(json.dumps(_summary(packet), sort_keys=True, indent=2))
    return 0


def _validate_only(args: argparse.Namespace) -> int:
    packet = load_adjudication_packet(args.output_dir / args.output_name)
    validate_adjudication_packet(packet)
    _, fresh = _build_fresh(
        packet_path=args.packet,
        human_path=args.human,
        sol_path=args.sol,
    )
    if (args.output_dir / args.output_name).read_text(encoding="utf-8") != fresh:
        raise ValueError(
            "phase2j existing adjudication packet does not match the "
            "deterministic build",
        )
    print(json.dumps(_summary(packet), sort_keys=True, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the Phase 2J post-Pass-A human-vs-Sol adjudication packet.",
    )
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--sol", type=Path, default=DEFAULT_SOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-name", default=OUTPUT_FILENAME,
        help="Adjudication packet filename (default: phase2j-adjudication-packet-v1.json).",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate the existing packet and inputs without writing.",
    )
    args = parser.parse_args(argv)
    for label, path in (
        ("locked packet", args.packet),
        ("human session", args.human),
        ("Sol review", args.sol),
    ):
        if not Path(path).is_file():
            parser.error(f"{label} input does not exist: {path}")
    try:
        if args.validate_only:
            return _validate_only(args)
        return _build(args)
    except (OSError, ValueError) as exc:
        print(f"[phase2j-adjudication] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
