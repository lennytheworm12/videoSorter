#!/usr/bin/env python3
"""Build or validate the isolated Phase 2K contextual-reconstruction artifacts.

The deterministic no-provider mode validates the frozen Phase 2J inputs and
the read-only transcript DB against all 30 exact targets, then emits the
A/B/C/D human-review records, exact context-radius entries, the blinded human
review packet (strictly blind: neutral presentations only, with the exact
condition/radius provenance retained in the separate mapping), and the
frozen-input manifest.  It never calls a provider and never overwrites
Bronze.

``--live`` additionally runs the injected ``core.llm`` backend through the
mechanical-cleanup / adaptive-diagnostic / contextual-reconstruction /
semantic-polish stages and writes every diagnostic attempt.  Live calls use
``temperature=0.0``,
``max_tokens=8192``, and ``thinking="disabled"``, and the run seals a
secret-free exact inference-config snapshot (provider, model, endpoint when
available, temperature, max_tokens, thinking, purpose) into every
model-call/attempt/failure artifact.  Output always goes to a new directory
(the build fails if it already exists); live runs should point
``--output-dir`` outside the repository for immutable archives.  Provider
responses are cached under ``--cache-dir`` keyed by the exact prompt +
inference-config + schema hash plus the ordered attempt index/kind for the
mechanical, sufficiency, contextual-reconstruction, and semantic-polish
stages, so changing model/thinking/temperature or a correction attempt's
exact prompt invalidates the cache.

``--validate-only`` revalidates an existing output directory deterministically
without any API calls.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2k_contextual_reconstruction import (
    OUTPUT_FILENAMES,
    PIPELINE_VERSION,
    build_phase2k_outputs,
    validate_output_directory,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/phase2j/window-selection-manifest-v1.json"
DEFAULT_PACKET = ROOT / "data/phase2j/reviewed-endpoint-annotation-packet-v1.json"
DEFAULT_DOC = ROOT / "docs/phase2j-independent-source-replication.md"
DEFAULT_DB = (
    Path("/home/bphan944/PersonalProjects/videoSorter-homework-archive/videos.db")
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "phase2k"
DEFAULT_CACHE_DIR = ROOT / "data" / "phase2k-response-cache-v3"

LIVE_CHAT_KWARGS = {
    "temperature": 0.0,
    "max_tokens": 8192,
    "thinking": "disabled",
}
LIVE_INFERENCE_PURPOSE = "phase2k-live-reconstruction"


def _live_inference_config(core_llm: object) -> dict:
    """Secret-free exact inference snapshot for the active core.llm backend."""
    endpoint = None
    if getattr(core_llm, "BACKEND", None) == "deepseek":
        endpoint = getattr(core_llm, "_DEEPSEEK_BASE_URL", None)
    return {
        "provider": core_llm.BACKEND,
        "model": core_llm.MODEL,
        "endpoint": endpoint,
        "temperature": LIVE_CHAT_KWARGS["temperature"],
        "max_tokens": LIVE_CHAT_KWARGS["max_tokens"],
        "thinking": LIVE_CHAT_KWARGS["thinking"],
        "purpose": LIVE_INFERENCE_PURPOSE,
    }


def _live_chat_factory() -> tuple[object, dict]:
    """Return (chat, sealed inference config) for the current core.llm."""
    from core import llm as core_llm

    def chat(system: str, user: str) -> str:
        return core_llm.chat(system=system, user=user, **LIVE_CHAT_KWARGS)

    return chat, _live_inference_config(core_llm)


def _print_summary(result: dict) -> None:
    print(json.dumps({
        "pipeline_version": PIPELINE_VERSION,
        "mode": result["mode"],
        "inference_config_hash": result.get("inference_config_hash"),
        "output_dir": str(result["output_dir"]),
        "window_count": result["window_count"],
        "frozen_input_manifest_sha256": result.get(
            "frozen_manifest_sha256",
            result.get("frozen_input_manifest_sha256"),
        ),
        "records_sha256": result["records_sha256"],
        "human_packet_sha256": result["human_packet_sha256"],
        "human_mapping_sha256": result["human_mapping_sha256"],
        "transformation_audit_sha256": result.get(
            "transformation_audit_sha256",
        ),
        "window_failure_count": result.get("window_failure_count"),
        "raw_response_count": result.get("raw_response_count"),
    }, sort_keys=True, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build or validate the isolated Phase 2K contextual-reconstruction "
            "artifacts."
        ),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reviewed-packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help=(
            "Provider response cache (live mode); keyed by the exact prompt "
            "+ inference-config + schema hash."
        ),
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Run the injected core.llm backend through all model stages.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate an existing Phase 2K output directory without API calls.",
    )
    args = parser.parse_args(argv)
    for label, path in (
        ("manifest", args.manifest),
        ("reviewed packet", args.reviewed_packet),
        ("doc", args.doc),
        ("transcript DB", args.db),
    ):
        if not Path(path).is_file():
            parser.error(f"{label} input does not exist: {path}")
    try:
        if args.validate_only:
            result = validate_output_directory(
                output_dir=args.output_dir,
                manifest_path=args.manifest,
                packet_path=args.reviewed_packet,
                db_path=args.db,
            )
        else:
            chat = None
            inference_config = None
            if args.live:
                chat, inference_config = _live_chat_factory()
            result = build_phase2k_outputs(
                manifest_path=args.manifest,
                packet_path=args.reviewed_packet,
                db_path=args.db,
                doc_path=args.doc,
                output_dir=args.output_dir,
                mode="live" if args.live else "no_provider",
                chat=chat,
                cache_dir=args.cache_dir if args.live else None,
                inference_config=inference_config,
            )
        _print_summary(result)
        return 0
    except (OSError, ValueError) as exc:
        print(f"[phase2k] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
