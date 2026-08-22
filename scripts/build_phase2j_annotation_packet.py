#!/usr/bin/env python3
"""Build the Phase 2J pre-annotation checkpoint artifacts (deterministic).

Writes ``data/phase2j/window-selection-manifest-v1.json`` and
``data/phase2j/endpoint-annotation-packet-v1.json`` from the retained pool and
the legacy five-window Phase 2H/2I benchmark.  The build is model-blind,
source-exact, deterministic (seed ``20260817``), and never runs scoring,
parser inference, Feature B/C, Logistic, LightGBM, or label/gold creation.

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

from pipeline.phase2j_annotation_packet import (
    PACKET_SCHEMA_VERSION,
    build_annotation_packet,
    load_annotation_packet,
    validate_annotation_packet,
)
from pipeline.phase2j_source_selection import (
    PARTITION_EXPANDED_DEV,
    PARTITION_FROZEN_REPLICATION,
    SELECTION_SCHEMA_VERSION,
    build_selection_manifest,
    load_legacy_benchmark,
    load_legacy_manifest,
    load_selection_manifest,
    validate_selection_manifest,
    verify_selection_manifest_inputs,
)
from pipeline.semantic_ir_pool import load_semantic_window_pool


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT / "data/semantic_ir_window_pool_v1.json"
DEFAULT_LEGACY_MANIFEST = ROOT / "data/semantic_ir_legacy_manifest_v1.json"
DEFAULT_LEGACY_BENCHMARK = ROOT / "data/semantic_ir_legacy_failure_v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "phase2j"
MANIFEST_FILENAME = "window-selection-manifest-v1.json"
PACKET_FILENAME = "endpoint-annotation-packet-v1.json"


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


def _build_fresh(
    *,
    pool_path: Path,
    legacy_manifest_path: Path,
    legacy_benchmark_path: Path,
) -> tuple[dict, dict, str, str]:
    pool = load_semantic_window_pool(pool_path)
    legacy_manifest = load_legacy_manifest(legacy_manifest_path)
    legacy_benchmark = load_legacy_benchmark(
        legacy_benchmark_path, manifest=legacy_manifest,
    )
    manifest = build_selection_manifest(
        pool=pool,
        pool_path=pool_path,
        legacy_manifest=legacy_manifest,
        legacy_manifest_path=legacy_manifest_path,
        legacy_benchmark=legacy_benchmark,
        legacy_benchmark_path=legacy_benchmark_path,
    )
    verify_selection_manifest_inputs(
        manifest,
        pool_path=pool_path,
        legacy_manifest_path=legacy_manifest_path,
        legacy_benchmark_path=legacy_benchmark_path,
        verify_catalogs=True,
        reproduce_selection=True,
    )
    packet = build_annotation_packet(manifest)
    validate_annotation_packet(packet, manifest=manifest)
    return manifest, packet, _serialize(manifest), _serialize(packet)


def _validate_existing(
    *,
    output_dir: Path,
    pool_path: Path,
    legacy_manifest_path: Path,
    legacy_benchmark_path: Path,
) -> tuple[dict, dict]:
    manifest_path = output_dir / MANIFEST_FILENAME
    packet_path = output_dir / PACKET_FILENAME
    if not manifest_path.is_file() or not packet_path.is_file():
        raise ValueError(
            "phase2j output set is incomplete; expected "
            + str(manifest_path) + " and " + str(packet_path),
        )
    manifest = load_selection_manifest(manifest_path)
    verify_selection_manifest_inputs(
        manifest,
        pool_path=pool_path,
        legacy_manifest_path=legacy_manifest_path,
        legacy_benchmark_path=legacy_benchmark_path,
        verify_catalogs=True,
        reproduce_selection=True,
    )
    packet = load_annotation_packet(packet_path, manifest=manifest)
    return manifest, packet


def _summary(manifest: dict, packet: dict) -> dict:
    selected = manifest["selected"]
    partitions = manifest["partition_counts"]
    return {
        "manifest": str(manifest["content_sha256"]),
        "packet": str(packet["content_sha256"]),
        "windows": len(selected),
        "video_source_groups": len({item["source_group_id"] for item in selected}),
        "expanded_dev": int(partitions[PARTITION_EXPANDED_DEV]),
        "frozen_replication": int(partitions[PARTITION_FROZEN_REPLICATION]),
        "candidate_total": int(manifest["diversity_summary"]["candidate_count"]),
        "release_gate": str(manifest["release_gate"]),
    }


def _build(args: argparse.Namespace) -> int:
    manifest, packet, manifest_bytes, packet_bytes = _build_fresh(
        pool_path=args.pool,
        legacy_manifest_path=args.legacy_manifest,
        legacy_benchmark_path=args.legacy_benchmark,
    )
    manifest_path = args.output_dir / MANIFEST_FILENAME
    packet_path = args.output_dir / PACKET_FILENAME
    existing_manifest = manifest_path.read_text(encoding="utf-8") if manifest_path.is_file() else None
    existing_packet = packet_path.read_text(encoding="utf-8") if packet_path.is_file() else None
    if (existing_manifest is None) != (existing_packet is None):
        raise ValueError(
            "phase2j preexisting output set is incomplete; refusing to overwrite",
        )
    if existing_manifest is not None and (
        existing_manifest != manifest_bytes or existing_packet != packet_bytes
    ):
        raise ValueError(
            "phase2j preexisting output does not match the deterministic "
            "build; refusing to overwrite",
        )
    _write_atomic(manifest_path, manifest_bytes)
    _write_atomic(packet_path, packet_bytes)
    print(json.dumps(_summary(manifest, packet), sort_keys=True, indent=2))
    return 0


def _validate_only(args: argparse.Namespace) -> int:
    manifest, packet = _validate_existing(
        output_dir=args.output_dir,
        pool_path=args.pool,
        legacy_manifest_path=args.legacy_manifest,
        legacy_benchmark_path=args.legacy_benchmark,
    )
    print(json.dumps(_summary(manifest, packet), sort_keys=True, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the Phase 2J pre-annotation checkpoint artifacts.",
    )
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--legacy-manifest", type=Path, default=DEFAULT_LEGACY_MANIFEST)
    parser.add_argument("--legacy-benchmark", type=Path, default=DEFAULT_LEGACY_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate the existing manifest and packet without writing.",
    )
    args = parser.parse_args(argv)
    for label, path in (
        ("pool", args.pool),
        ("legacy manifest", args.legacy_manifest),
        ("legacy benchmark", args.legacy_benchmark),
    ):
        if not Path(path).is_file():
            parser.error(f"{label} input does not exist: {path}")
    try:
        if args.validate_only:
            return _validate_only(args)
        return _build(args)
    except (OSError, ValueError) as exc:
        print(f"[phase2j] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
