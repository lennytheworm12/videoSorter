#!/usr/bin/env python3
"""Import completed Phase 2K human reviews, transformation audits, and
closeout inputs.

The review packet produced by ``build_phase2k_reconstruction.py`` is
downstream-result-blind and never carries human scores.  This CLI accepts a
completed-reviews JSON object keyed by review item ID, accepts a completed
transformation-audit packet for live builds, refuses any incomplete input,
and writes a finalized packet plus deterministic summary artifacts.  No
human score or audit decision is ever fabricated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2k_contextual_reconstruction import (
    COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION,
    FINAL_CLOSEOUT_STATUSES,
    HUMAN_MAPPING_SCHEMA_VERSION,
    HUMAN_PACKET_SCHEMA_VERSION,
    RECORDS_SCHEMA_VERSION,
    TRANSFORMATION_AUDIT_SCHEMA_VERSION,
    OUTPUT_FILENAMES,
    build_closeout_status,
    build_count_report_skeleton,
    import_completed_human_reviews,
    load_json_strict,
    summarize_human_reviews,
    summarize_transformation_audits,
    validate_completed_transformation_audits,
    validate_downstream_comparison,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "phase2k"


def _load_reviews(path: Path) -> dict:
    reviews = load_json_strict(path, label="completed human reviews")
    for key, value in reviews.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ValueError("completed human reviews must map item IDs to objects")
    return reviews


def _count_report(
    *,
    transformation_summary: dict | None,
    human_summary: dict,
) -> dict:
    """Fill the exact skeleton with computed values; every value is real."""
    report = build_count_report_skeleton()
    report["review_items"] = human_summary["overall"]["item_count"]
    report["windows"] = human_summary["overall"]["window_count"]
    if transformation_summary is None:
        return report
    report["asr"] = transformation_summary["asr"]
    report["entity"] = transformation_summary["entity"]
    report["ability_ownership"] = transformation_summary["ability_ownership"]
    report["unsupported"] = transformation_summary["unsupported"]
    report["polish_preservation"] = {
        "modality_preserved_rate": transformation_summary[
            "polish_preservation"
        ]["modality_preserved"],
        "negation_preserved_rate": transformation_summary[
            "polish_preservation"
        ]["negation_preserved"],
        "uncertainty_preserved_rate": transformation_summary[
            "polish_preservation"
        ]["uncertainty_preserved"],
        "approved_statements": transformation_summary[
            "polish_preservation"
        ]["approved_statements"],
    }
    report["first_failures"] = transformation_summary["first_failures"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Import completed Phase 2K human reviews and transformation "
            "audits, then summarize/close out."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reviews", type=Path, required=True,
        help="JSON object mapping review_item_id -> scores/reviewer/completed_at.",
    )
    parser.add_argument(
        "--audits", type=Path, default=None,
        help="Completed transformation-audit packet (required for live builds).",
    )
    parser.add_argument(
        "--downstream-comparison", type=Path, default=None,
        help="Completed downstream comparison JSON (required for closeout).",
    )
    parser.add_argument(
        "--closeout-decision", type=str, choices=FINAL_CLOSEOUT_STATUSES,
        default=None,
        help="Allowed final Notion status after all inputs are complete.",
    )
    parser.add_argument("--reviewer", type=str, required=True)
    parser.add_argument("--completed-at", type=str, required=True)
    args = parser.parse_args(argv)

    for label, path in (
        ("reviews", args.reviews),
        ("audits", args.audits),
        ("downstream comparison", args.downstream_comparison),
    ):
        if path is not None and not Path(path).is_file():
            parser.error(f"{label} input does not exist: {path}")

    packet_path = args.output_dir / OUTPUT_FILENAMES["human_packet"]
    mapping_path = args.output_dir / OUTPUT_FILENAMES["human_mapping"]
    records_path = args.output_dir / OUTPUT_FILENAMES["records"]
    for label, path in (
        ("blank human packet", packet_path),
        ("human mapping", mapping_path),
        ("records", records_path),
    ):
        if not Path(path).is_file():
            parser.error(f"{label} does not exist: {path}")

    try:
        packet = load_json_strict(packet_path, label="phase2k blank human packet")
        mapping = load_json_strict(mapping_path, label="phase2k human mapping")
        records_obj = load_json_strict(records_path, label="phase2k records")
        if packet.get("schema_version") != HUMAN_PACKET_SCHEMA_VERSION:
            raise ValueError("blank human packet schema version is invalid")
        if mapping.get("schema_version") != HUMAN_MAPPING_SCHEMA_VERSION:
            raise ValueError("human mapping schema version is invalid")
        if records_obj.get("schema_version") != RECORDS_SCHEMA_VERSION:
            raise ValueError("records schema version is invalid")
        if mapping.get("content_sha256") != packet.get("blinding", {}).get(
            "mapping_sha256",
        ):
            raise ValueError("human mapping is not bound to the packet")

        mode = records_obj.get("mode")
        if mode not in ("no_provider", "live"):
            raise ValueError("records mode is invalid")
        if mode == "live":
            if args.audits is None:
                raise ValueError("live builds require a completed transformation audit")
            template = load_json_strict(
                args.output_dir / OUTPUT_FILENAMES["transformation_audit"],
                label="phase2k transformation audit template",
            )
            if template.get("schema_version") != TRANSFORMATION_AUDIT_SCHEMA_VERSION:
                raise ValueError("transformation audit template schema is invalid")
            completed_audit = load_json_strict(
                args.audits, label="completed transformation audit",
            )
            if completed_audit.get("schema_version") != (
                COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION
            ):
                raise ValueError(
                    "completed transformation audit schema version is invalid",
                )
            completed_audit = validate_completed_transformation_audits(
                template,
                completed_audit,
                records_obj=records_obj,
            )
        elif args.audits is not None:
            raise ValueError(
                "no-provider builds do not have a transformation audit",
            )

        reviews = _load_reviews(args.reviews)
        finalized = import_completed_human_reviews(
            packet,
            reviews,
            reviewer=args.reviewer,
            completed_at=args.completed_at,
        )
        human_summary = summarize_human_reviews(
            finalized,
            mapping=mapping,
            records_file=records_obj,
        )
        gate_status = human_summary["review_gate"]["status"]
        if gate_status not in ("PASSED", "FAILED"):
            raise ValueError("human review gate status is invalid")
        review_gate_passed = gate_status == "PASSED"
        if args.downstream_comparison is not None and not review_gate_passed:
            raise ValueError(
                "downstream comparison cannot be accepted until the human "
                "review gate is PASSED",
            )
        if args.closeout_decision is not None and not review_gate_passed:
            raise ValueError(
                "--closeout-decision requires the human review gate to be PASSED",
            )

        finalized_path = args.output_dir / OUTPUT_FILENAMES["finalized_packet"]
        human_summary_path = args.output_dir / OUTPUT_FILENAMES["human_summary"]
        finalized_path.write_text(
            json.dumps(finalized, sort_keys=True, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        human_summary_path.write_text(
            json.dumps(human_summary, sort_keys=True, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

        transformation_summary = None
        if mode == "live":
            transformation_summary = summarize_transformation_audits(
                completed_audit,
                records_obj=records_obj,
            )
            finalized_audit_path = (
                args.output_dir / OUTPUT_FILENAMES["finalized_transformation_audit"]
            )
            transformation_summary_path = (
                args.output_dir / OUTPUT_FILENAMES["transformation_summary"]
            )
            finalized_audit_path.write_text(
                json.dumps(
                    completed_audit,
                    sort_keys=True,
                    indent=2,
                    ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )
            transformation_summary_path.write_text(
                json.dumps(
                    transformation_summary,
                    sort_keys=True,
                    indent=2,
                    ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )

        downstream = None
        if args.downstream_comparison is not None:
            downstream = validate_downstream_comparison(
                load_json_strict(
                    args.downstream_comparison,
                    label="downstream comparison",
                ),
                label="downstream comparison",
                records_obj=records_obj,
                finalized_packet=finalized,
                human_summary=human_summary,
                completed_audit=completed_audit if mode == "live" else None,
            )
        downstream_complete = downstream is not None
        if downstream_complete and args.closeout_decision is None:
            raise ValueError(
                "downstream comparison is complete; --closeout-decision is required",
            )
        if (
            downstream is not None
            and args.closeout_decision is not None
            and downstream["decision"] != args.closeout_decision
        ):
            raise ValueError(
                "--closeout-decision must match the downstream comparison decision",
            )
        if not downstream_complete and args.closeout_decision is not None:
            raise ValueError(
                "--closeout-decision requires a completed downstream comparison",
            )
        report = _count_report(
            transformation_summary=transformation_summary,
            human_summary=human_summary,
        )
        closeout = build_closeout_status(
            human_review_complete=True,
            downstream_comparison_complete=downstream_complete,
            closeout_decision=args.closeout_decision,
            count_report=report,
            downstream_comparison=downstream,
            human_review_gate_passed=review_gate_passed,
        )
        closeout_path = args.output_dir / OUTPUT_FILENAMES["closeout_status"]
        closeout_path.write_text(
            json.dumps(closeout, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print(json.dumps({
            "finalized_packet": str(finalized_path),
            "summary": str(human_summary_path),
            "transformation_summary": (
                str(args.output_dir / OUTPUT_FILENAMES["transformation_summary"])
                if transformation_summary is not None
                else None
            ),
            "closeout": str(closeout_path),
            "closeout_status": closeout["status"],
            "item_count": human_summary["overall"]["item_count"],
            "window_count": human_summary["overall"]["window_count"],
            "review_gate_status": gate_status,
            "diagnosis": (
                downstream["diagnosis"]
                if downstream is not None
                else None
            ),
        }, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"[phase2k] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
