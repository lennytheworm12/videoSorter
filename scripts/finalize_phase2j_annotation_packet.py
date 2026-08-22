#!/usr/bin/env python3
"""Finalize a Phase 2J annotation packet after human review edits.

The existing ``content_sha256`` is treated as stale input: the canonical
content hash is recomputed in memory, the full packet is validated against the
locked selection manifest with the strengthened cross-binding validator, and
only after successful validation is the packet atomically rewritten in
canonical pretty JSON.  ``--check-only`` requires the existing hash to already
be correct and validates without writing anything.

The release gate is never unlocked and no scoring, parser inference, Feature
B/C, Logistic/LightGBM, or label/gold creation is run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2j_annotation_packet import (
    is_window_gold_eligible,
    validate_annotation_packet,
)
from pipeline.phase2j_source_selection import (
    canonical_sha256,
    load_selection_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET = ROOT / "data/phase2j/endpoint-annotation-packet-v1.json"
DEFAULT_MANIFEST = ROOT / "data/phase2j/window-selection-manifest-v1.json"


def _load_json_strict(path: Path, *, label: str) -> dict[str, Any]:
    """Strict JSON object load with duplicate-key rejection."""
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} JSON contains duplicate keys")
            value[key] = item
        return value

    try:
        body = json.loads(
            Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
        )
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} JSON is unavailable or malformed") from exc
    if not isinstance(body, dict):
        raise ValueError(f"{label} must be a JSON object")
    return body


def recompute_packet_hash(packet: dict[str, Any]) -> str:
    """Canonical content hash of the packet excluding its self-hash."""
    inner = {key: value for key, value in packet.items() if key != "content_sha256"}
    return canonical_sha256(inner)


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


def _status_counts(packet: dict[str, Any], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in packet["records"]:
        status = record[key]["status"]
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _summary(packet: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    window_counts: dict[str, int] = {}
    endpoint_count = 0
    gold_eligible = 0
    for record in packet["records"]:
        status = record["window_status"]
        window_counts[status] = window_counts.get(status, 0) + 1
        endpoint_count += len(record["endpoints"])
        if is_window_gold_eligible(record):
            gold_eligible += 1
    return {
        "packet_sha256": packet["content_sha256"],
        "manifest_sha256": manifest["content_sha256"],
        "release_gate": packet["release_gate"],
        "window_statuses": dict(sorted(window_counts.items())),
        "pass_a_statuses": _status_counts(packet, "pass_a"),
        "pass_b_statuses": _status_counts(packet, "pass_b"),
        "endpoint_count": endpoint_count,
        "gold_eligible_windows": gold_eligible,
    }


def _finalize(args: argparse.Namespace) -> int:
    manifest = load_selection_manifest(Path(args.manifest))
    packet = _load_json_strict(Path(args.packet), label="phase2j annotation packet")
    recomputed = recompute_packet_hash(packet)
    if args.check_only:
        if packet.get("content_sha256") != recomputed:
            raise ValueError(
                "phase2j annotation packet content hash is stale; "
                "rerun without --check-only to finalize",
            )
        validate_annotation_packet(packet, manifest=manifest)
        print(json.dumps(_summary(packet, manifest), sort_keys=True, indent=2))
        return 0
    updated = {"content_sha256": recomputed, **{
        key: value for key, value in packet.items() if key != "content_sha256"
    }}
    validate_annotation_packet(updated, manifest=manifest)
    _write_atomic(Path(args.packet), _serialize(updated))
    print(json.dumps(_summary(updated, manifest), sort_keys=True, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Finalize a Phase 2J annotation packet after human review.",
    )
    parser.add_argument(
        "--packet", type=Path, default=DEFAULT_PACKET,
        help="Annotation packet JSON to finalize (default: data/phase2j/endpoint-annotation-packet-v1.json).",
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help="Locked selection manifest JSON (default: data/phase2j/window-selection-manifest-v1.json).",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Require the existing content hash to be correct and validate without writing.",
    )
    args = parser.parse_args(argv)
    for label, path in (("packet", args.packet), ("manifest", args.manifest)):
        if not Path(path).is_file():
            parser.error(f"{label} input does not exist: {path}")
    try:
        return _finalize(args)
    except (OSError, ValueError) as exc:
        print(f"[phase2j-finalize] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
