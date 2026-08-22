#!/usr/bin/env python3
"""Finalize the Phase 2K downstream comparison from validated evidence.

Consumes a validated rerun evidence directory and explicit human-supplied
closeout decision, diagnosis, and note, calls ``build_downstream_comparison``,
and validates the v2 envelope against the exact Phase 2K records, finalized
human packet, human review summary, and completed transformation audit before
writing it.  The diagnosis and note are never inferred.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2k_contextual_reconstruction import ROOT
from pipeline.phase2k_downstream_comparison import (
    DOWNSTREAM_DIAGNOSIS_VALUES,
    FINAL_CLOSEOUT_STATUSES,
)
from pipeline.phase2k_downstream_rerun import finalize_phase2k_downstream_rerun


DEFAULT_REVIEWED_PACKET = ROOT / "data/phase2j/reviewed-endpoint-annotation-packet-v1.json"
DEFAULT_COVERAGE = ROOT / "data/phase2j/candidate-coverage-v1.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Finalize the Phase 2K downstream-comparison v2.",
    )
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--phase2k-dir", type=Path, required=True)
    parser.add_argument("--alignment-packet", type=Path, required=True)
    parser.add_argument("--alignment-summary", type=Path, required=True)
    parser.add_argument(
        "--reviewed-packet", type=Path, default=DEFAULT_REVIEWED_PACKET,
    )
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument(
        "--decision", required=True,
        choices=sorted(FINAL_CLOSEOUT_STATUSES),
    )
    parser.add_argument(
        "--diagnosis", required=True,
        choices=sorted(DOWNSTREAM_DIAGNOSIS_VALUES),
    )
    parser.add_argument("--note", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    for label, path in (
        ("evidence directory", args.evidence_dir),
        ("phase2k output directory", args.phase2k_dir),
        ("alignment packet", args.alignment_packet),
        ("alignment summary", args.alignment_summary),
        ("Phase 2J reviewed packet", args.reviewed_packet),
        ("Phase 2J candidate coverage", args.coverage),
    ):
        if not Path(path).exists():
            parser.error(f"{label} does not exist: {path}")
    if args.output.exists():
        parser.error(f"output path already exists: {args.output}")
    if not args.note.strip():
        parser.error("--note must be non-empty")

    try:
        comparison = finalize_phase2k_downstream_rerun(
            evidence_dir=args.evidence_dir,
            phase2k_dir=args.phase2k_dir,
            alignment_packet_path=args.alignment_packet,
            alignment_summary_path=args.alignment_summary,
            reviewed_packet_path=args.reviewed_packet,
            coverage_path=args.coverage,
            decision=args.decision,
            diagnosis=args.diagnosis,
            note=args.note,
        )
        body = json.dumps(
            comparison, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ) + "\n"
        parent = args.output.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=args.output.name + ".tmp-", dir=parent,
        )
        os.close(temporary_fd)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(body, encoding="utf-8")
            os.replace(temporary, args.output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        print(json.dumps({
            "status": "FINALIZED",
            "output": str(args.output),
            "content_sha256": comparison["content_sha256"],
            "decision": comparison["decision"],
            "diagnosis": comparison["diagnosis"],
            "window_count": comparison["dataset_binding"]["window_count"],
            "target_count": comparison["semantic_target_contract"][
                "target_count"
            ],
        }, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(
            f"[phase2k] downstream rerun finalization failed: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
