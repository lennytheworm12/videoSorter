#!/usr/bin/env python3
"""Phase 2J context-ablation multi-agent (multi_agent_v1) transport.

This script is the audited, non-exec transport for the 20 independent Phase
2J GPT-5.6 Sol calls (10 frozen cases x conditions A/B).  The parent spawns
20 fresh non-forked ``multi_agent_v1`` default agents with
``requested_model=gpt-5.6-sol`` and ``reasoning_effort=high``; each agent's
initial message is exactly the canonical wrapper prompt plus the canonical
JSON serialization of ONLY its inner condition payload, produced by the
``prompt`` subcommand.

Subcommands:

  init     Validate the frozen artifacts, write the canonical intermediate
           response schema, and create a fresh run manifest with exactly 20
           pending calls.
  prompt   Fail-closed validate the manifest/artifacts and print the exact
           experiment user message (canonical wrapper prompt + canonical
           inner payload JSON) for one call to stdout, with no other text.
  ingest   Strict-parse and validate a staged response file for one call,
           atomically persist the raw response under the run dir, and
           atomically update only that manifest call.
  status   Validate the manifest and all completed raw responses and report
           per-call evidence without executing anything.
  import   Require all 20 valid completed calls, deterministically import
           them into the standard extraction-outputs bundle, and record and
           strictly validate the final output hashes in the manifest.

Timestamp requirements are waived for this transport.  The manifest
truthfully records that the ``multi_agent_v1`` backend identity is
requested/recorded but not cryptographically proven, and that the
surrounding subagent system envelope is transport-provided while the
experiment user message is the canonical wrapper prompt plus the canonical
inner payload.  No codex argv/CLI execution is claimed and no model calls
are made by this program itself.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2j_context_ablation import (
    CONDITION_CODES,
    DEFAULT_DB_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PACKET_PATH,
    DEFAULT_VOCABULARY_PATH,
    OUTPUT_FILENAMES,
    PIPELINE_VERSION,
    build_outputs_bundle,
    build_sol_intermediate_schema,
    canonical_sha256,
    file_sha256,
    import_sol_intermediate_response,
    load_json_strict,
    normalize_path_locator,
    text_sha256,
    validate_output_directory,
    validate_sol_intermediate_response,
)
from scripts.run_phase2j_context_ablation_sol import SOL_WRAPPER_PROMPT


ROOT = Path(__file__).resolve().parents[1]

TRANSPORT = "multi_agent_v1"
REQUESTED_MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "high"
MULTI_AGENT_RUN_SCHEMA_VERSION = "phase2j-context-ablation-multi-agent-run-v2"
DEFAULT_RUN_DIR = DEFAULT_OUTPUT_DIR / "sol_multi_agent_run_v2"

MANIFEST_FILENAME = "manifest.json"
SCHEMA_FILENAME = "intermediate-schema.json"
RAW_SUBDIR = "raw"

MANIFEST_KEYS = (
    "schema_version", "purpose", "transport", "requested_model",
    "reasoning_effort", "wrapper_sha256", "intermediate_schema_sha256",
    "instructions_sha256", "payloads_sha256", "run_dir", "calls",
    "final_outputs", "content_sha256",
)

FINAL_OUTPUTS_KEYS = (
    "outputs_path", "outputs_sha256", "outputs_file_sha256", "by_call",
)

CALL_KEYS = (
    "case_id", "condition", "payload_sha256", "prompt_sha256", "agent_id",
    "status", "attempts", "raw_response_path", "raw_response_sha256",
    "last_error",
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def _serialize(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(data)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, value: object) -> None:
    _write_bytes_atomic(path, _serialize(value).encode("utf-8"))


def _recompute_content_hash(obj: Mapping[str, Any], *, label: str) -> str:
    digest = canonical_sha256({
        key: value for key, value in obj.items() if key != "content_sha256"
    })
    if obj.get("content_sha256") != digest:
        raise ValueError(f"{label} content_sha256 does not match canonical content")
    return digest


def _require_exact_keys(value: object, expected: tuple[str, ...], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{label} key set is invalid")


def _call_key(case_id: str, condition: str) -> str:
    return f"{case_id}:{condition}"


def _raw_filename(case_id: str, condition: str) -> str:
    return f"{case_id.replace(':', '-')}-{condition}.raw.json"


def _raw_path_for_call(run_dir: Path, case_id: str, condition: str) -> Path:
    return run_dir / RAW_SUBDIR / _raw_filename(case_id, condition)


def _relative_to_run_dir(path: Path, run_dir: Path) -> str:
    try:
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path {path} is outside the run directory") from exc


def _validate_frozen_artifacts(args: argparse.Namespace) -> None:
    """Authoritative frozen-input and output-directory validation."""
    validate_output_directory(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        packet_path=args.reviewed_packet,
        db_path=args.db,
        vocabulary_path=args.vocabulary,
    )


def _load_payloads(output_dir: Path) -> dict[str, Any]:
    return load_json_strict(
        output_dir / OUTPUT_FILENAMES["payloads"], label="payloads",
    )


def _load_instructions(output_dir: Path) -> dict[str, Any]:
    artifact = load_json_strict(
        output_dir / OUTPUT_FILENAMES["instructions"], label="instructions",
    )
    return {
        key: value for key, value in artifact.items()
        if key != "content_sha256"
    }


def _schema_bytes() -> bytes:
    return _serialize(build_sol_intermediate_schema()).encode("utf-8")


def _ensure_schema_file(path: Path, *, force: bool) -> None:
    """Write the canonical schema; fail closed on a mismatched existing file."""
    canonical = _schema_bytes()
    if path.is_file():
        if path.read_bytes() == canonical:
            return
        if not force:
            raise ValueError(
                f"existing schema file does not match the canonical schema: "
                f"{path}; use --force to replace it",
            )
    _write_bytes_atomic(path, canonical)


def _new_run_manifest(
    *,
    run_dir: Path,
    payloads: Mapping[str, Any],
) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for payload_case in payloads["cases"]:
        for condition in CONDITION_CODES:
            payload = payload_case[condition]
            calls.append({
                "case_id": payload_case["case_id"],
                "condition": condition,
                "payload_sha256": payload["content_sha256"],
                "prompt_sha256": None,
                "agent_id": None,
                "status": "pending",
                "attempts": 0,
                "raw_response_path": None,
                "raw_response_sha256": None,
                "last_error": None,
            })
    manifest = {
        "schema_version": MULTI_AGENT_RUN_SCHEMA_VERSION,
        "purpose": (
            "Phase 2J context-ablation multi_agent_v1 run manifest.  "
            "Records the requested model (gpt-5.6-sol) and reasoning effort "
            "(high), the exact shared wrapper prompt hash, the intermediate "
            "response schema hash, the instructions/payload hashes, per-call "
            "prompt and raw-response hashes, agent IDs, and final output "
            "hashes.  The multi_agent_v1 backend identity is "
            "requested/recorded but not cryptographically proven; the "
            "surrounding subagent system envelope is transport-provided, "
            "while the experiment user message is the canonical wrapper "
            "prompt plus the canonical inner condition payload.  No codex "
            "argv/CLI execution is claimed."
        ),
        "transport": TRANSPORT,
        "requested_model": REQUESTED_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "wrapper_sha256": text_sha256(SOL_WRAPPER_PROMPT),
        "intermediate_schema_sha256": canonical_sha256(
            build_sol_intermediate_schema(),
        ),
        "instructions_sha256": payloads["instructions_sha256"],
        "payloads_sha256": payloads["content_sha256"],
        "run_dir": normalize_path_locator(run_dir),
        "calls": calls,
        "final_outputs": None,
    }
    return {
        **manifest,
        "content_sha256": canonical_sha256(manifest),
    }


def _validate_manifest_calls(
    manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    payloads: Mapping[str, Any],
) -> None:
    calls = manifest["calls"]
    if not isinstance(calls, list) or len(calls) != 20:
        raise ValueError("multi-agent run manifest must contain exactly 20 calls")
    expected = [
        (
            payload_case["case_id"],
            condition,
            payload_case[condition]["content_sha256"],
        )
        for payload_case in payloads["cases"]
        for condition in CONDITION_CODES
    ]
    resolved_run_dir = run_dir.resolve()
    for index, call in enumerate(calls):
        if not isinstance(call, Mapping):
            raise ValueError("multi-agent run manifest call must be an object")
        _require_exact_keys(call, CALL_KEYS, "multi-agent run manifest call")
        expected_case_id, expected_condition, expected_payload_sha256 = expected[index]
        if (
            call["case_id"] != expected_case_id
            or call["condition"] != expected_condition
        ):
            raise ValueError("multi-agent run manifest call order is misaligned")
        if call["payload_sha256"] != expected_payload_sha256:
            raise ValueError("multi-agent run manifest call payload hash is invalid")
        status = call["status"]
        if status not in {"pending", "completed", "failed"}:
            raise ValueError("multi-agent run manifest call status is invalid")
        if (
            isinstance(call["attempts"], bool)
            or not isinstance(call["attempts"], int)
            or call["attempts"] < 0
        ):
            raise ValueError("multi-agent run manifest call attempts is invalid")
        if call["last_error"] is not None and not isinstance(
            call["last_error"], str,
        ):
            raise ValueError("multi-agent run manifest call last_error is invalid")
        for name in (
            "prompt_sha256", "agent_id", "raw_response_path",
            "raw_response_sha256",
        ):
            value = call[name]
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"multi-agent run manifest call {name} is invalid",
                )
        if status == "completed":
            for name in (
                "prompt_sha256", "agent_id", "raw_response_path",
                "raw_response_sha256",
            ):
                if not isinstance(call[name], str) or not call[name]:
                    raise ValueError(
                        f"multi-agent run manifest completed call {name} is missing",
                    )
            for name in ("prompt_sha256", "raw_response_sha256"):
                if HEX64.fullmatch(call[name]) is None:
                    raise ValueError(
                        f"multi-agent run manifest completed call {name} is malformed",
                    )
            recorded = call["raw_response_path"]
            if Path(recorded).is_absolute():
                raise ValueError(
                    "multi-agent run manifest raw_response_path must be "
                    "relative to the run dir",
                )
            resolved_raw = (run_dir / recorded).resolve()
            if not resolved_raw.is_relative_to(resolved_run_dir):
                raise ValueError(
                    "multi-agent run manifest raw_response_path escapes the run dir",
                )
            if not resolved_raw.is_file():
                raise ValueError(
                    "multi-agent run manifest raw response is missing: "
                    f"{resolved_raw}",
                )
            if file_sha256(resolved_raw) != call["raw_response_sha256"]:
                raise ValueError(
                    "multi-agent run manifest raw response hash does not match",
                )
        elif status == "pending":
            for name in (
                "prompt_sha256", "agent_id", "raw_response_path",
                "raw_response_sha256",
            ):
                if call[name] is not None:
                    raise ValueError(
                        f"multi-agent run manifest pending call {name} must be null",
                    )
        else:  # failed
            if not isinstance(call["last_error"], str) or not call["last_error"]:
                raise ValueError(
                    "multi-agent run manifest failed call requires last_error",
                )
            for name in ("raw_response_path", "raw_response_sha256"):
                if call[name] is not None:
                    raise ValueError(
                        f"multi-agent run manifest failed call {name} must be null",
                    )


def _payload_by_call(
    payloads: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        _call_key(payload_case["case_id"], condition): payload_case[condition]
        for payload_case in payloads["cases"]
        for condition in CONDITION_CODES
    }


def _validate_completed_raw_responses(
    manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    payloads: Mapping[str, Any],
) -> None:
    """Re-validate every completed raw response against its exact call."""
    payload_by_call = _payload_by_call(payloads)
    for call in manifest["calls"]:
        if call["status"] != "completed":
            continue
        key = _call_key(call["case_id"], call["condition"])
        raw_path = (run_dir / call["raw_response_path"]).resolve()
        response = load_json_strict(raw_path, label=f"raw response {key}")
        validate_sol_intermediate_response(
            response,
            case_id=call["case_id"],
            condition=call["condition"],
            payload=payload_by_call[key],
        )


def _resolve_recorded_path(recorded: str) -> Path:
    """Resolve a canonical path locator back to an absolute filesystem path."""
    path = Path(recorded)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _imported_output_hashes(
    manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    payloads: Mapping[str, Any],
) -> dict[str, str]:
    """Deterministically recompute the imported extraction output hashes."""
    payload_by_call = _payload_by_call(payloads)
    hashes: dict[str, str] = {}
    for call in manifest["calls"]:
        key = _call_key(call["case_id"], call["condition"])
        raw_path = (run_dir / call["raw_response_path"]).resolve()
        if not raw_path.is_file():
            raise ValueError(f"raw response for {key} is missing: {raw_path}")
        if file_sha256(raw_path) != call["raw_response_sha256"]:
            raise ValueError(
                f"raw response hash for {key} does not match the manifest",
            )
        response = load_json_strict(raw_path, label=f"raw response {key}")
        output = import_sol_intermediate_response(
            response,
            case_id=call["case_id"],
            condition=call["condition"],
            payload=payload_by_call[key],
        )
        hashes[key] = output["content_sha256"]
    return hashes


def _validate_final_outputs(
    manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    payloads: Mapping[str, Any],
) -> None:
    """Validate the canonical pre/post-import final_outputs record."""
    final_outputs = manifest["final_outputs"]
    if final_outputs is None:
        return
    if not isinstance(final_outputs, Mapping):
        raise ValueError(
            "multi-agent run manifest final_outputs must be null or an object",
        )
    _require_exact_keys(
        final_outputs,
        FINAL_OUTPUTS_KEYS,
        "multi-agent run manifest final outputs",
    )
    outputs_path = final_outputs["outputs_path"]
    if not isinstance(outputs_path, str) or not outputs_path:
        raise ValueError(
            "multi-agent run manifest final outputs outputs_path is invalid",
        )
    for label in ("outputs_sha256", "outputs_file_sha256"):
        value = final_outputs[label]
        if not isinstance(value, str) or HEX64.fullmatch(value) is None:
            raise ValueError(
                f"multi-agent run manifest final outputs {label} is invalid",
            )
    expected_keys = [
        _call_key(call["case_id"], call["condition"])
        for call in manifest["calls"]
    ]
    by_call = final_outputs["by_call"]
    if not isinstance(by_call, Mapping) or set(by_call) != set(expected_keys):
        raise ValueError(
            "multi-agent run manifest final outputs by_call key set is invalid",
        )
    for key, value in by_call.items():
        if not isinstance(value, str) or HEX64.fullmatch(value) is None:
            raise ValueError(
                "multi-agent run manifest final outputs by_call hash for "
                f"{key} is invalid",
            )
    for call in manifest["calls"]:
        if call["status"] != "completed":
            raise ValueError(
                "multi-agent run manifest final outputs require all calls completed",
            )
    imported_hashes = _imported_output_hashes(
        manifest,
        run_dir=run_dir,
        payloads=payloads,
    )
    for key in expected_keys:
        if by_call[key] != imported_hashes[key]:
            raise ValueError(
                "multi-agent run manifest final outputs by_call does not match "
                f"the imported extraction output for {key}",
            )
    outputs_file = _resolve_recorded_path(outputs_path)
    if not outputs_file.is_file():
        raise ValueError(
            "multi-agent run manifest final outputs file is missing: "
            f"{outputs_file}",
        )
    if file_sha256(outputs_file) != final_outputs["outputs_file_sha256"]:
        raise ValueError(
            "multi-agent run manifest final outputs file hash does not match "
            "the current outputs artifact",
        )
    bundle = load_json_strict(outputs_file, label="outputs bundle")
    if bundle.get("content_sha256") != final_outputs["outputs_sha256"]:
        raise ValueError(
            "multi-agent run manifest final outputs content hash does not "
            "match the current outputs artifact",
        )
    cases = bundle.get("cases")
    if not isinstance(cases, list) or len(cases) != len(payloads["cases"]):
        raise ValueError(
            "multi-agent run manifest final outputs bundle cases are invalid",
        )
    for case in cases:
        if not isinstance(case, Mapping) or not isinstance(
            case.get("A"), Mapping,
        ) or not isinstance(case.get("B"), Mapping):
            raise ValueError(
                "multi-agent run manifest final outputs bundle case is invalid",
            )
        for condition in CONDITION_CODES:
            key = _call_key(case["case_id"], condition)
            if case[condition].get("content_sha256") != by_call.get(key):
                raise ValueError(
                    "multi-agent run manifest final outputs by_call does not "
                    "match the current outputs artifact",
                )


def _validate_run_manifest(
    manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    payloads: Mapping[str, Any],
) -> None:
    _require_exact_keys(manifest, MANIFEST_KEYS, "multi-agent run manifest")
    if manifest["schema_version"] != MULTI_AGENT_RUN_SCHEMA_VERSION:
        raise ValueError("multi-agent run manifest schema version is invalid")
    if manifest["transport"] != TRANSPORT:
        raise ValueError(
            "multi-agent run manifest transport is invalid; only "
            f"{TRANSPORT!r} is accepted",
        )
    if manifest["requested_model"] != REQUESTED_MODEL:
        raise ValueError(
            "multi-agent run manifest requests model "
            f"{manifest['requested_model']!r}; only {REQUESTED_MODEL!r} is "
            "accepted",
        )
    if manifest["reasoning_effort"] != REASONING_EFFORT:
        raise ValueError(
            "multi-agent run manifest reasoning_effort is invalid; only "
            f"{REASONING_EFFORT!r} is accepted",
        )
    if manifest["wrapper_sha256"] != text_sha256(SOL_WRAPPER_PROMPT):
        raise ValueError("multi-agent run manifest wrapper hash is not canonical")
    expected_schema_hash = canonical_sha256(build_sol_intermediate_schema())
    if manifest["intermediate_schema_sha256"] != expected_schema_hash:
        raise ValueError("multi-agent run manifest schema hash is invalid")
    if manifest["instructions_sha256"] != payloads["instructions_sha256"]:
        raise ValueError("multi-agent run manifest instructions hash is invalid")
    if manifest["payloads_sha256"] != payloads["content_sha256"]:
        raise ValueError("multi-agent run manifest payloads hash is invalid")
    if manifest["run_dir"] != normalize_path_locator(run_dir):
        raise ValueError("multi-agent run manifest run_dir is invalid")
    _recompute_content_hash(manifest, label="multi-agent run manifest")
    _validate_manifest_calls(manifest, run_dir=run_dir, payloads=payloads)
    _validate_completed_raw_responses(manifest, run_dir=run_dir, payloads=payloads)
    _validate_final_outputs(manifest, run_dir=run_dir, payloads=payloads)


def _prompt_for_payload(payload: Mapping[str, Any]) -> tuple[str, str]:
    text = SOL_WRAPPER_PROMPT + "\n\n" + _canonical_json(payload)
    return text, text_sha256(text)


def _payload_for_call(
    payloads: Mapping[str, Any],
    case_id: str,
    condition: str,
) -> Mapping[str, Any]:
    if condition not in CONDITION_CODES:
        raise ValueError(f"condition {condition!r} is not canonical")
    for payload_case in payloads["cases"]:
        if payload_case["case_id"] == case_id:
            return payload_case[condition]
    raise ValueError(f"case id {case_id!r} is not in the canonical payloads")


def _find_call(
    manifest: Mapping[str, Any],
    case_id: str,
    condition: str,
) -> dict[str, Any]:
    for call in manifest["calls"]:
        if call["case_id"] == case_id and call["condition"] == condition:
            return call
    raise ValueError(
        f"call {_call_key(case_id, condition)} is not in the run manifest",
    )


def _cmd_init(args: argparse.Namespace) -> int:
    _validate_frozen_artifacts(args)
    payloads = _load_payloads(args.output_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    schema_path = run_dir / SCHEMA_FILENAME
    _ensure_schema_file(schema_path, force=args.force)
    manifest_path = run_dir / MANIFEST_FILENAME
    if manifest_path.is_file() and not args.force:
        raise ValueError(
            f"multi-agent run manifest already exists: {manifest_path}; "
            "use --force to replace it",
        )
    manifest = _new_run_manifest(run_dir=run_dir, payloads=payloads)
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps({
        "command": "init",
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": MULTI_AGENT_RUN_SCHEMA_VERSION,
        "transport": TRANSPORT,
        "requested_model": REQUESTED_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "wrapper_sha256": manifest["wrapper_sha256"],
        "intermediate_schema_sha256": manifest["intermediate_schema_sha256"],
        "instructions_sha256": manifest["instructions_sha256"],
        "payloads_sha256": manifest["payloads_sha256"],
        "run_dir": manifest["run_dir"],
        "manifest_path": str(manifest_path),
        "calls": 20,
        "pending": 20,
    }, sort_keys=True, indent=2))
    return 0


def _cmd_prompt(args: argparse.Namespace) -> int:
    _validate_frozen_artifacts(args)
    payloads = _load_payloads(args.output_dir)
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(f"multi-agent run manifest does not exist: {manifest_path}")
    manifest = load_json_strict(manifest_path, label="multi-agent run manifest")
    _validate_run_manifest(manifest, run_dir=run_dir, payloads=payloads)
    payload = _payload_for_call(payloads, args.case_id, args.condition)
    prompt_text, _ = _prompt_for_payload(payload)
    sys.stdout.write(prompt_text)
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    if not isinstance(args.agent_id, str) or not args.agent_id.strip():
        raise ValueError("--agent-id must be a non-empty string")
    response_path = Path(args.response)
    if not response_path.is_file():
        raise ValueError(f"staged response file does not exist: {response_path}")
    _validate_frozen_artifacts(args)
    payloads = _load_payloads(args.output_dir)
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(f"multi-agent run manifest does not exist: {manifest_path}")
    manifest = load_json_strict(manifest_path, label="multi-agent run manifest")
    _validate_run_manifest(manifest, run_dir=run_dir, payloads=payloads)
    payload = _payload_for_call(payloads, args.case_id, args.condition)
    call = _find_call(manifest, args.case_id, args.condition)
    key = _call_key(args.case_id, args.condition)
    if call["status"] == "completed" and not args.force:
        raise ValueError(
            f"call {key} already has valid completed evidence; use --force "
            "to replace it",
        )
    response = load_json_strict(
        response_path, label=f"staged response {key}",
    )
    validate_sol_intermediate_response(
        response,
        case_id=args.case_id,
        condition=args.condition,
        payload=payload,
    )
    prompt_text, prompt_sha256 = _prompt_for_payload(payload)
    raw_path = _raw_path_for_call(run_dir, args.case_id, args.condition)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_atomic(raw_path, response_path.read_bytes())
    attempts = call["attempts"] + 1
    call.update({
        "prompt_sha256": prompt_sha256,
        "agent_id": args.agent_id,
        "status": "completed",
        "attempts": attempts,
        "raw_response_path": _relative_to_run_dir(raw_path, run_dir),
        "raw_response_sha256": file_sha256(raw_path),
        "last_error": None,
    })
    manifest["content_sha256"] = canonical_sha256({
        item_key: value for item_key, value in manifest.items()
        if item_key != "content_sha256"
    })
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps({
        "command": "ingest",
        "key": key,
        "status": "completed",
        "attempts": attempts,
        "prompt_sha256": prompt_sha256,
        "raw_response_path": call["raw_response_path"],
        "raw_response_sha256": call["raw_response_sha256"],
        "manifest_path": str(manifest_path),
    }, sort_keys=True, indent=2))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    _validate_frozen_artifacts(args)
    payloads = _load_payloads(args.output_dir)
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        print(json.dumps({
            "command": "status",
            "run_dir": str(run_dir),
            "manifest_path": None,
            "pending": 20,
            "completed": 0,
            "failed": 0,
        }, sort_keys=True, indent=2))
        return 0
    manifest = load_json_strict(manifest_path, label="multi-agent run manifest")
    _validate_run_manifest(manifest, run_dir=run_dir, payloads=payloads)
    print(json.dumps({
        "command": "status",
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "pending": sum(
            (1 for call in manifest["calls"] if call["status"] == "pending"),
        ),
        "completed": sum(
            (1 for call in manifest["calls"] if call["status"] == "completed"),
        ),
        "failed": sum(
            (1 for call in manifest["calls"] if call["status"] == "failed"),
        ),
        "calls": [
            {
                "key": _call_key(call["case_id"], call["condition"]),
                "status": call["status"],
                "agent_id": call["agent_id"],
                "prompt_sha256": call["prompt_sha256"],
                "raw_response_sha256": call["raw_response_sha256"],
                "attempts": call["attempts"],
            }
            for call in manifest["calls"]
        ],
    }, sort_keys=True, indent=2))
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    _validate_frozen_artifacts(args)
    payloads = _load_payloads(args.output_dir)
    instructions = _load_instructions(args.output_dir)
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(f"multi-agent run manifest does not exist: {manifest_path}")
    manifest = load_json_strict(manifest_path, label="multi-agent run manifest")
    _validate_run_manifest(manifest, run_dir=run_dir, payloads=payloads)
    payload_by_call = _payload_by_call(payloads)
    outputs_by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for call in manifest["calls"]:
        key = _call_key(call["case_id"], call["condition"])
        if call["status"] != "completed":
            raise ValueError(f"call {key} is not completed")
        raw_path = (run_dir / call["raw_response_path"]).resolve()
        if not raw_path.is_file():
            raise ValueError(f"raw response for {key} is missing: {raw_path}")
        if file_sha256(raw_path) != call["raw_response_sha256"]:
            raise ValueError(
                f"raw response hash for {key} does not match the manifest",
            )
        response = load_json_strict(raw_path, label=f"raw response {key}")
        outputs_by_case.setdefault(call["case_id"], {})[call["condition"]] = (
            import_sol_intermediate_response(
                response,
                case_id=call["case_id"],
                condition=call["condition"],
                payload=payload_by_call[key],
            )
        )
    bundle = build_outputs_bundle(
        payloads_artifact=payloads,
        instructions=instructions,
        outputs_by_case=outputs_by_case,
    )
    out_path = (
        Path(args.outputs)
        if args.outputs is not None
        else args.output_dir / OUTPUT_FILENAMES["outputs"]
    )
    if out_path.is_file() and not args.force:
        raise ValueError(
            f"outputs already exist at {out_path}; use --force to replace",
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(out_path, bundle)
    manifest["final_outputs"] = {
        "outputs_path": normalize_path_locator(out_path),
        "outputs_sha256": bundle["content_sha256"],
        "outputs_file_sha256": file_sha256(out_path),
        "by_call": {
            _call_key(case_id, condition): output["content_sha256"]
            for case_id, conditions in outputs_by_case.items()
            for condition, output in conditions.items()
        },
    }
    manifest["content_sha256"] = canonical_sha256({
        item_key: value for item_key, value in manifest.items()
        if item_key != "content_sha256"
    })
    _validate_run_manifest(manifest, run_dir=run_dir, payloads=payloads)
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps({
        "command": "import",
        "outputs_path": str(out_path),
        "outputs_sha256": bundle["content_sha256"],
        "outputs_file_sha256": manifest["final_outputs"]["outputs_file_sha256"],
        "imported_calls": 20,
        "manifest_path": str(manifest_path),
    }, sort_keys=True, indent=2))
    return 0


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--reviewed-packet", type=Path, default=DEFAULT_PACKET_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--vocabulary", type=Path, default=DEFAULT_VOCABULARY_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, prompt, ingest, and import the 20 independent Phase 2J "
            "context-ablation GPT-5.6 Sol calls via the audited "
            "multi_agent_v1 transport (no model calls made by this program "
            "itself)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", help="Create a fresh 20-call run manifest.",
    )
    _common_parser(init_parser)
    init_parser.add_argument(
        "--force", action="store_true",
        help="Replace an existing manifest/schema in the run dir.",
    )

    prompt_parser = subparsers.add_parser(
        "prompt", help="Print the exact canonical user message for one call.",
    )
    _common_parser(prompt_parser)
    prompt_parser.add_argument("--case-id", required=True)
    prompt_parser.add_argument("--condition", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Validate and record one staged response.",
    )
    _common_parser(ingest_parser)
    ingest_parser.add_argument("--case-id", required=True)
    ingest_parser.add_argument("--condition", required=True)
    ingest_parser.add_argument("--agent-id", required=True)
    ingest_parser.add_argument("--response", required=True, type=Path)
    ingest_parser.add_argument(
        "--force", action="store_true",
        help="Replace valid completed evidence for exactly this call.",
    )

    import_parser = subparsers.add_parser(
        "import", help="Import all 20 raw responses and assemble outputs.",
    )
    _common_parser(import_parser)
    import_parser.add_argument(
        "--outputs", type=Path, default=None,
        help=(
            "Output bundle path (default: the standard extraction-outputs "
            "filename in --output-dir)."
        ),
    )
    import_parser.add_argument(
        "--force", action="store_true",
        help="Allow replacing an existing outputs bundle.",
    )

    status_parser = subparsers.add_parser(
        "status", help="Report per-call run status.",
    )
    _common_parser(status_parser)

    args = parser.parse_args(argv)
    try:
        handlers = {
            "init": _cmd_init,
            "prompt": _cmd_prompt,
            "ingest": _cmd_ingest,
            "status": _cmd_status,
            "import": _cmd_import,
        }
        return handlers[args.command](args)
    except (OSError, ValueError) as exc:
        print(
            f"[phase2j-context-ablation-multi-agent] error: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
