#!/usr/bin/env python3
"""Build, validate, finalize, and gate the Phase 2J context ablation.

Subcommands:

  build  Freeze the 10-case selection, the shared extraction instructions,
         and the A/B condition payloads from the frozen manifest, the
         archived read-only transcript DB, and the lexical vocabulary.
         The default build reaches ``ready_for_sol`` with no model calls.
         When ``--outputs`` is supplied, validated extraction outputs are
         bound and the blinded human review packet and sealed mapping are
         generated (``review_packet`` mode).

  validate  Deterministically revalidate an existing output directory
            against the frozen manifest/packet/DB/vocabulary.

  finalize  Import completed human reviews (with explicit human
            attestation) and freeze the Sol comparison summary
            (MATERIAL / NOT_MATERIAL).

  emit-deepseek-run  Emit the DeepSeek B run packet; gate-locked until a
                     frozen MATERIAL Sol summary exists and the full output
                     directory validates.

  import-deepseek-run  Import validated DeepSeek B outputs; same gate.

No model calls are made by any subcommand.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2j_context_ablation import (
    DEFAULT_DB_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PACKET_PATH,
    DEFAULT_VOCABULARY_PATH,
    OUTPUT_FILENAMES,
    PIPELINE_VERSION,
    build_deepseek_run_packet,
    build_phase2j_context_ablation_outputs,
    finalize_materiality_outputs,
    import_deepseek_run_outputs,
    load_json_strict,
    validate_deepseek_import_artifact,
    validate_deepseek_run_packet,
    validate_output_directory,
)


ROOT = Path(__file__).resolve().parents[1]


def _print_json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, indent=2))


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--reviewed-packet", type=Path, default=DEFAULT_PACKET_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--vocabulary", type=Path, default=DEFAULT_VOCABULARY_PATH,
    )


def _cmd_build(args: argparse.Namespace) -> int:
    for label, path in (
        ("manifest", args.manifest),
        ("reviewed packet", args.reviewed_packet),
        ("transcript DB", args.db),
        ("vocabulary", args.vocabulary),
    ):
        if not Path(path).is_file():
            raise ValueError(f"{label} input does not exist: {path}")
    if args.outputs is not None and not Path(args.outputs).is_file():
        raise ValueError(f"extraction outputs input does not exist: {args.outputs}")
    result = build_phase2j_context_ablation_outputs(
        manifest_path=args.manifest,
        packet_path=args.reviewed_packet,
        db_path=args.db,
        output_dir=args.output_dir,
        outputs_path=args.outputs,
        vocabulary_path=args.vocabulary,
    )
    _print_json({
        "command": "build",
        "pipeline_version": PIPELINE_VERSION,
        "mode": result["mode"],
        "output_dir": str(result["output_dir"]),
        "selection_sha256": result["selection_sha256"],
        "instructions_sha256": result["instructions_sha256"],
        "payloads_sha256": result["payloads_sha256"],
        "outputs_sha256": result["outputs_sha256"],
        "human_packet_sha256": result["human_packet_sha256"],
        "human_mapping_sha256": result["human_mapping_sha256"],
        "build_summary_sha256": result["build_summary_sha256"],
        "selected_case_ids": result["selected_case_ids"],
        "selected_window_ids": result["selected_window_ids"],
    })
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    result = validate_output_directory(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        packet_path=args.reviewed_packet,
        db_path=args.db,
        vocabulary_path=args.vocabulary,
    )
    _print_json({
        "command": "validate",
        "valid": True,
        "output_dir": str(result["output_dir"]),
        **{key: value for key, value in result.items()
           if key not in {"valid", "output_dir"}},
    })
    return 0


def _authoritative_validation(args: argparse.Namespace) -> None:
    """Run the authoritative output-directory validation before gated ops."""
    validate_output_directory(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        packet_path=args.reviewed_packet,
        db_path=args.db,
        vocabulary_path=args.vocabulary,
    )


def _cmd_finalize(args: argparse.Namespace) -> int:
    if not Path(args.reviews).is_file():
        raise ValueError(f"completed reviews input does not exist: {args.reviews}")
    _authoritative_validation(args)
    result = finalize_materiality_outputs(
        output_dir=args.output_dir,
        reviews_path=args.reviews,
        frozen_at=args.frozen_at,
    )
    _print_json({
        "command": "finalize",
        "decision": result["decision"],
        "materiality": result["materiality"],
        "finalized_packet_sha256": result["finalized_packet_sha256"],
        "materiality_summary_sha256": result["materiality_summary_sha256"],
        "output_dir": str(result["output_dir"]),
    })
    return 0


def _cmd_emit_deepseek(args: argparse.Namespace) -> int:
    _authoritative_validation(args)
    artifacts = _load_output_artifacts(args.output_dir)
    summary = artifacts.get("materiality_summary")
    payloads = artifacts.get("payloads")
    if summary is None or payloads is None:
        raise ValueError(
            "emit-deepseek-run requires payloads and a frozen materiality summary",
        )
    run_packet = build_deepseek_run_packet(
        summary=summary,
        payloads_artifact=payloads,
    )
    validate_deepseek_run_packet(
        run_packet, summary=summary, payloads_artifact=payloads,
    )
    path = args.output_dir / OUTPUT_FILENAMES["deepseek_run"]
    _write_json(path, run_packet)
    _print_json({
        "command": "emit-deepseek-run",
        "release_gate": run_packet["release_gate"],
        "materiality_summary_sha256": run_packet["materiality_summary_sha256"],
        "deepseek_run_sha256": run_packet["content_sha256"],
        "path": str(path),
    })
    return 0


def _cmd_import_deepseek(args: argparse.Namespace) -> int:
    _authoritative_validation(args)
    artifacts = _load_output_artifacts(args.output_dir)
    summary = artifacts.get("materiality_summary")
    payloads = artifacts.get("payloads")
    if summary is None or payloads is None:
        raise ValueError(
            "import-deepseek-run requires payloads and a frozen materiality summary",
        )
    run_packet = load_json_strict(
        args.run_packet, label="deepseek run packet",
    )
    validate_deepseek_run_packet(
        run_packet, summary=summary, payloads_artifact=payloads,
    )
    outputs_bundle = load_json_strict(args.outputs, label="deepseek outputs")
    if not isinstance(outputs_bundle, dict) or not isinstance(
        outputs_bundle.get("cases"), dict,
    ):
        raise ValueError("deepseek outputs must map case_id -> B output")
    import_artifact = import_deepseek_run_outputs(
        summary=summary,
        run_packet=run_packet,
        outputs_by_case=outputs_bundle["cases"],
        payloads_artifact=payloads,
    )
    validate_deepseek_import_artifact(
        import_artifact,
        summary=summary,
        run_packet=run_packet,
        payloads_artifact=payloads,
    )
    path = args.output_dir / OUTPUT_FILENAMES["deepseek_import"]
    _write_json(path, import_artifact)
    _print_json({
        "command": "import-deepseek-run",
        "release_gate": import_artifact["release_gate"],
        "deepseek_import_sha256": import_artifact["content_sha256"],
        "path": str(path),
    })
    return 0


def _load_output_artifacts(output_dir: Path) -> dict:
    result = {}
    for key, filename in OUTPUT_FILENAMES.items():
        path = Path(output_dir) / filename
        if path.is_file():
            result[key] = load_json_strict(path, label=f"phase2j {key} artifact")
    return result


def _write_json(path: Path, value: object) -> None:
    """Same-directory atomic write: temp file + os.replace, cleanup on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build/validate/finalize the isolated Phase 2J source-grounded "
            "semantic-extraction ablation harness (no model calls)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="Freeze selection/instructions/A-B payloads.",
    )
    _common_parser(build_parser)
    build_parser.add_argument(
        "--outputs", type=Path, default=None,
        help=(
            "Validated extraction outputs bundle (requires payloads); when "
            "supplied, also builds the blinded human review packet/mapping."
        ),
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Revalidate an existing output directory.",
    )
    _common_parser(validate_parser)

    finalize_parser = subparsers.add_parser(
        "finalize", help="Import completed reviews and freeze the summary.",
    )
    _common_parser(finalize_parser)
    finalize_parser.add_argument(
        "--reviews", type=Path, required=True,
        help="Completed-reviews JSON artifact with explicit human attestation.",
    )
    finalize_parser.add_argument(
        "--frozen-at", type=str, required=True,
        help="ISO-8601 frozen timestamp for the Sol comparison summary.",
    )

    emit_parser = subparsers.add_parser(
        "emit-deepseek-run", help="Emit the gate-locked DeepSeek B run packet.",
    )
    _common_parser(emit_parser)

    import_parser = subparsers.add_parser(
        "import-deepseek-run", help="Import gate-locked DeepSeek B outputs.",
    )
    _common_parser(import_parser)
    import_parser.add_argument(
        "--run-packet", type=Path, required=True,
        help="DeepSeek run packet emitted after a MATERIAL summary.",
    )
    import_parser.add_argument(
        "--outputs", type=Path, required=True,
        help="JSON object mapping case_id -> DeepSeek B extraction output.",
    )

    args = parser.parse_args(argv)
    try:
        handlers = {
            "build": _cmd_build,
            "validate": _cmd_validate,
            "finalize": _cmd_finalize,
            "emit-deepseek-run": _cmd_emit_deepseek,
            "import-deepseek-run": _cmd_import_deepseek,
        }
        return handlers[args.command](args)
    except (OSError, ValueError) as exc:
        print(f"[phase2j-context-ablation] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
