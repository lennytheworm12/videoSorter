#!/usr/bin/env python3
"""Execute and import the Phase 2J context-ablation Sol calls.

The primary agent runs exactly 20 independent GPT-5.6 Sol calls (10 frozen
cases x conditions A/B) against the already validated DB-only v2 payload
artifact.  Each call receives one canonical wrapper prompt plus the canonical
JSON serialization of ONLY the inner condition payload, runs in an isolated
temp workspace with ``--ephemeral``, and must return a strict intermediate
response matching the canonical JSON Schema (exact contiguous quotes plus
zero-based occurrence indexes; no model-supplied character offsets).

Subcommands:

  schema   Validate the frozen artifacts and write the canonical
           intermediate response JSON Schema into the run directory.
  run      Validate the frozen artifacts, create/validate the run manifest,
           and execute all pending calls.  Valid existing raw responses for
           the exact payload/prompt/config are reused; mismatched or
           malformed results fail closed unless ``--force`` is supplied.
  import   Require all 20 completed raw responses, deterministically import
           them into validated extraction outputs, assemble the standard
           outputs bundle, and record final output hashes in the manifest.
  status   Report per-call run status without executing anything.

No model outputs are claimed to be deterministic.  The manifest records the
requested model/config, codex CLI version, exact argv template, wrapper/
schema/payload/prompt hashes, raw-response hashes, and final output hashes as
requested-model evidence only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
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


ROOT = Path(__file__).resolve().parents[1]

SOL_MODEL = "gpt-5.6-sol"
SOL_REASONING_EFFORT = "high"
SOL_RUN_SCHEMA_VERSION = "phase2j-context-ablation-sol-run-v2"
DEFAULT_RUN_DIR = DEFAULT_OUTPUT_DIR / "sol_run_v2"

MANIFEST_FILENAME = "manifest.json"
SCHEMA_FILENAME = "intermediate-schema.json"
RAW_SUBDIR = "raw"
LOGS_SUBDIR = "logs"

MAX_WORKERS_DEFAULT = 2
MAX_WORKERS_LIMIT = 4

# Canonical wrapper prompt, byte-identical for every call.  Only the appended
# canonical JSON payload differs between calls.
SOL_WRAPPER_PROMPT = (
    "You are a deterministic extraction worker for a controlled "
    "source-grounding experiment.  The supplied JSON payload embeds the "
    "exact extraction instructions and the isolated condition context.\n"
    "\n"
    "Follow the embedded instructions exactly.  Return ONLY a single JSON "
    "object matching the supplied output schema.  Do not add commentary, "
    "markdown, explanations, or any text outside the JSON object.\n"
    "\n"
    "The output schema is strict: every object has "
    "additionalProperties=false, empty field lists are allowed, and every "
    "source reference must carry an exact contiguous quote and a zero-based "
    "occurrence_index.  Occurrence indexes are counted among all exact, "
    "non-overlapping substring matches of the quote in the condition "
    "source: condition A counts matches in the supplied Bronze target; "
    "condition B counts matches in the supplied full transcript.  Do not "
    "compute, estimate, or return character offsets; the importer resolves "
    "offsets mechanically.\n"
    "\n"
    "Forbidden:\n"
    "- No tool use, shell commands, file reads, web/external lookups, or "
    "network access.  Everything you need is in the supplied payload.\n"
    "- No cross-case inference; do not use knowledge of other cases.\n"
    "- No mechanical cleaning, contextual rewriting, semantic polish, or "
    "strategic abstraction of the extracted text or quotes.\n"
    "- In condition B, the transcript context may only resolve the "
    "semantics of the supplied target; do not contribute unrelated facts "
    "from the transcript.\n"
    "- Do not include provenance, video identity, URLs, metadata beyond "
    "the supplied payload, or any keys not allowed by the schema.\n"
    "\n"
    "Return exactly one valid JSON object now."
)

# Exact argv template with placeholders; recorded verbatim in the manifest.
ARGV_TEMPLATE = [
    "codex", "exec",
    "--ephemeral",
    "--ignore-user-config",
    "--ignore-rules",
    "-m", SOL_MODEL,
    "-c", f'model_reasoning_effort="{SOL_REASONING_EFFORT}"',
    "-s", "read-only",
    "-C", "{work_dir}",
    "--skip-git-repo-check",
    "--output-schema", "{schema_path}",
    "-o", "{raw_path}",
    "-",
]

MANIFEST_KEYS = (
    "schema_version", "purpose", "requested_model",
    "model_reasoning_effort", "codex_cli_version", "argv_template",
    "wrapper_sha256", "intermediate_schema_sha256", "instructions_sha256",
    "payloads_sha256", "run_dir", "calls", "final_outputs",
    "content_sha256",
)

FINAL_OUTPUTS_KEYS = (
    "outputs_path", "outputs_sha256", "outputs_file_sha256", "by_call",
)

CALL_KEYS = (
    "case_id", "condition", "payload_sha256", "prompt_sha256", "status",
    "raw_response_path", "raw_response_sha256", "attempts", "last_error",
    "log_path", "temp_dir", "started_at", "completed_at",
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


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _recompute_content_hash(obj: Mapping[str, Any], *, label: str) -> str:
    digest = canonical_sha256({
        key: value for key, value in obj.items() if key != "content_sha256"
    })
    if obj.get("content_sha256") != digest:
        raise ValueError(f"{label} content_sha256 does not match canonical content")
    return digest


def _codex_cli_version() -> str:
    completed = subprocess.run(
        ["codex", "--version"],
        capture_output=True,
        check=False,
        shell=False,
    )
    combined = (
        (completed.stdout or b"").decode("utf-8", errors="replace")
        + "\n"
        + (completed.stderr or b"").decode("utf-8", errors="replace")
    )
    match = re.search(r"codex-cli\s+v?([0-9]+\.[0-9]+\.[0-9]+)", combined)
    if match is None:
        raise ValueError(
            "could not determine codex CLI version from `codex --version`",
        )
    return match.group(1)


def _codex_argv(
    *,
    work_dir: Path,
    schema_path: Path,
    raw_path: Path,
) -> list[str]:
    return [
        (
            arg.format(
                work_dir=work_dir,
                schema_path=schema_path,
                raw_path=raw_path,
            )
            if "{work_dir}" in arg or "{schema_path}" in arg or "{raw_path}" in arg
            else arg
        )
        for arg in ARGV_TEMPLATE
    ]


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
    codex_cli_version: str,
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
                "status": "pending",
                "raw_response_path": None,
                "raw_response_sha256": None,
                "attempts": 0,
                "last_error": None,
                "log_path": None,
                "temp_dir": None,
                "started_at": None,
                "completed_at": None,
            })
    manifest = {
        "schema_version": SOL_RUN_SCHEMA_VERSION,
        "purpose": (
            "Phase 2J context-ablation Sol run manifest.  Records requested "
            "model/config, codex CLI version, exact argv template, wrapper/"
            "schema/payload hashes, per-call prompt and raw-response "
            "hashes, and final output hashes.  Does not claim model-output "
            "determinism."
        ),
        "requested_model": SOL_MODEL,
        "model_reasoning_effort": SOL_REASONING_EFFORT,
        "codex_cli_version": codex_cli_version,
        "argv_template": list(ARGV_TEMPLATE),
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


def _require_exact_keys(value: object, expected: tuple[str, ...], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{label} key set is invalid")


def _validate_manifest_calls(
    manifest: Mapping[str, Any],
    *,
    payloads: Mapping[str, Any],
) -> None:
    calls = manifest["calls"]
    if not isinstance(calls, list) or len(calls) != 20:
        raise ValueError("sol run manifest must contain exactly 20 calls")
    expected = [
        (
            payload_case["case_id"],
            condition,
            payload_case[condition]["content_sha256"],
        )
        for payload_case in payloads["cases"]
        for condition in CONDITION_CODES
    ]
    for index, call in enumerate(calls):
        if not isinstance(call, Mapping):
            raise ValueError("sol run manifest call must be an object")
        _require_exact_keys(call, CALL_KEYS, "sol run manifest call")
        expected_case_id, expected_condition, expected_payload_sha256 = expected[index]
        if (
            call["case_id"] != expected_case_id
            or call["condition"] != expected_condition
        ):
            raise ValueError("sol run manifest call order is misaligned")
        if call["payload_sha256"] != expected_payload_sha256:
            raise ValueError("sol run manifest call payload hash is invalid")
        if call["status"] not in {"pending", "completed", "failed"}:
            raise ValueError("sol run manifest call status is invalid")
        if call["prompt_sha256"] is not None and not isinstance(
            call["prompt_sha256"], str,
        ):
            raise ValueError("sol run manifest prompt_sha256 is invalid")
        if call["raw_response_path"] is not None and not isinstance(
            call["raw_response_path"], str,
        ):
            raise ValueError("sol run manifest raw_response_path is invalid")
        if call["raw_response_sha256"] is not None and not isinstance(
            call["raw_response_sha256"], str,
        ):
            raise ValueError("sol run manifest raw_response_sha256 is invalid")
        if not isinstance(call["attempts"], int) or call["attempts"] < 0:
            raise ValueError("sol run manifest call attempts is invalid")


def _validate_run_manifest(
    manifest: Mapping[str, Any],
    *,
    codex_cli_version: str,
    run_dir: Path,
    payloads: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        manifest,
        MANIFEST_KEYS,
        "sol run manifest",
    )
    if manifest["schema_version"] != SOL_RUN_SCHEMA_VERSION:
        raise ValueError("sol run manifest schema version is invalid")
    if manifest["requested_model"] != SOL_MODEL:
        raise ValueError(
            f"sol run manifest requests model {manifest['requested_model']!r}; "
            f"only {SOL_MODEL!r} is accepted",
        )
    if manifest["model_reasoning_effort"] != SOL_REASONING_EFFORT:
        raise ValueError(
            "sol run manifest model_reasoning_effort is invalid; only "
            f"{SOL_REASONING_EFFORT!r} is accepted",
        )
    if manifest["codex_cli_version"] != codex_cli_version:
        raise ValueError(
            "sol run manifest codex CLI version "
            f"{manifest['codex_cli_version']!r} does not match the current "
            f"CLI {codex_cli_version!r}",
        )
    if manifest["argv_template"] != ARGV_TEMPLATE:
        raise ValueError("sol run manifest argv template is not canonical")
    if manifest["wrapper_sha256"] != text_sha256(SOL_WRAPPER_PROMPT):
        raise ValueError("sol run manifest wrapper hash is not canonical")
    expected_schema_hash = canonical_sha256(build_sol_intermediate_schema())
    if manifest["intermediate_schema_sha256"] != expected_schema_hash:
        raise ValueError("sol run manifest intermediate schema hash is invalid")
    if manifest["instructions_sha256"] != payloads["instructions_sha256"]:
        raise ValueError("sol run manifest instructions hash is invalid")
    if manifest["payloads_sha256"] != payloads["content_sha256"]:
        raise ValueError("sol run manifest payloads hash is invalid")
    if manifest["run_dir"] != normalize_path_locator(run_dir):
        raise ValueError("sol run manifest run_dir is invalid")
    _recompute_content_hash(manifest, label="sol run manifest")
    _validate_manifest_calls(manifest, payloads=payloads)
    _validate_final_outputs(manifest, run_dir=run_dir, payloads=payloads)


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
    payload_by_call = {
        _call_key(payload_case["case_id"], condition): payload_case[condition]
        for payload_case in payloads["cases"]
        for condition in CONDITION_CODES
    }
    hashes: dict[str, str] = {}
    for call in manifest["calls"]:
        key = _call_key(call["case_id"], call["condition"])
        raw_response_path = call["raw_response_path"]
        if not isinstance(raw_response_path, str):
            raise ValueError(
                f"sol run manifest final outputs require a raw response for "
                f"{key}",
            )
        raw_path = run_dir / raw_response_path
        if not raw_path.is_file():
            raise ValueError(
                f"raw response for {key} is missing: {raw_path}",
            )
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
    """Validate the canonical pre/post-import final_outputs record.

    ``None`` is the canonical pre-import representation.  Once present, the
    record must carry exactly the four final-output keys, valid
    64-lowercase-hex hashes, all 20 canonical case/condition keys whose
    values match the deterministically re-imported extraction outputs, and
    must bind to the current outputs artifact and file.
    """
    final_outputs = manifest["final_outputs"]
    if final_outputs is None:
        return
    if not isinstance(final_outputs, Mapping):
        raise ValueError(
            "sol run manifest final_outputs must be null or an object",
        )
    _require_exact_keys(
        final_outputs,
        FINAL_OUTPUTS_KEYS,
        "sol run manifest final outputs",
    )
    outputs_path = final_outputs["outputs_path"]
    if not isinstance(outputs_path, str) or not outputs_path:
        raise ValueError(
            "sol run manifest final outputs outputs_path is invalid",
        )
    for label in ("outputs_sha256", "outputs_file_sha256"):
        value = final_outputs[label]
        if not isinstance(value, str) or HEX64.fullmatch(value) is None:
            raise ValueError(
                f"sol run manifest final outputs {label} is invalid",
            )
    expected_keys = [
        _call_key(call["case_id"], call["condition"])
        for call in manifest["calls"]
    ]
    by_call = final_outputs["by_call"]
    if not isinstance(by_call, Mapping) or set(by_call) != set(expected_keys):
        raise ValueError(
            "sol run manifest final outputs by_call key set is invalid",
        )
    for key, value in by_call.items():
        if not isinstance(value, str) or HEX64.fullmatch(value) is None:
            raise ValueError(
                "sol run manifest final outputs by_call hash for "
                f"{key} is invalid",
            )
    for call in manifest["calls"]:
        if call["status"] != "completed":
            raise ValueError(
                "sol run manifest final outputs require all calls completed",
            )
    imported_hashes = _imported_output_hashes(
        manifest,
        run_dir=run_dir,
        payloads=payloads,
    )
    for key in expected_keys:
        if by_call[key] != imported_hashes[key]:
            raise ValueError(
                "sol run manifest final outputs by_call does not match the "
                f"imported extraction output for {key}",
            )
    outputs_file = _resolve_recorded_path(outputs_path)
    if not outputs_file.is_file():
        raise ValueError(
            "sol run manifest final outputs file is missing: "
            f"{outputs_file}",
        )
    if file_sha256(outputs_file) != final_outputs["outputs_file_sha256"]:
        raise ValueError(
            "sol run manifest final outputs file hash does not match the "
            "current outputs artifact",
        )
    bundle = load_json_strict(outputs_file, label="outputs bundle")
    if bundle.get("content_sha256") != final_outputs["outputs_sha256"]:
        raise ValueError(
            "sol run manifest final outputs content hash does not match the "
            "current outputs artifact",
        )
    cases = bundle.get("cases")
    if not isinstance(cases, list) or len(cases) != len(payloads["cases"]):
        raise ValueError(
            "sol run manifest final outputs bundle cases are invalid",
        )
    for case in cases:
        if not isinstance(case, Mapping) or not isinstance(
            case.get("A"), Mapping,
        ) or not isinstance(case.get("B"), Mapping):
            raise ValueError(
                "sol run manifest final outputs bundle case is invalid",
            )
        for condition in CONDITION_CODES:
            key = _call_key(case["case_id"], condition)
            if case[condition].get("content_sha256") != by_call.get(key):
                raise ValueError(
                    "sol run manifest final outputs by_call does not match "
                    "the current outputs artifact",
                )

def _prompt_for_payload(payload: Mapping[str, Any]) -> tuple[str, str]:
    text = SOL_WRAPPER_PROMPT + "\n\n" + _canonical_json(payload)
    return text, text_sha256(text)


def _existing_raw_decision(
    call: Mapping[str, Any],
    *,
    run_dir: Path,
    payload: Mapping[str, Any],
    force: bool,
) -> tuple[bool, dict[str, Any] | None]:
    """Decide whether an existing raw response can be reused.

    Returns (reuse, parsed_response).  Raises ValueError when an existing
    artifact is malformed or mismatched and ``force`` is not set.
    """
    raw_path = _raw_path_for_call(run_dir, call["case_id"], call["condition"])
    if not raw_path.is_file():
        if call["status"] == "completed" and not force:
            raise ValueError(
                f"call {_call_key(call['case_id'], call['condition'])} is "
                f"recorded completed but its raw response is missing: "
                f"{raw_path}",
            )
        return False, None
    current_hash = file_sha256(raw_path)
    if call["status"] == "completed":
        recorded_path = raw_path.resolve().relative_to(run_dir.resolve()).as_posix()
        if call["raw_response_path"] != recorded_path or \
                call["raw_response_sha256"] != current_hash:
            if not force:
                raise ValueError(
                    f"existing raw response for "
                    f"{_call_key(call['case_id'], call['condition'])} does "
                    f"not match the manifest record; use --force to rerun",
                )
            return False, None
    prompt_text, prompt_sha256 = _prompt_for_payload(payload)
    if call["prompt_sha256"] is not None and call["prompt_sha256"] != prompt_sha256:
        if not force:
            raise ValueError(
                f"existing raw response for "
                f"{_call_key(call['case_id'], call['condition'])} does not "
                f"match the current prompt binding; use --force to rerun",
            )
        return False, None
    try:
        response = load_json_strict(raw_path, label="raw sol response")
        validate_sol_intermediate_response(
            response,
            case_id=call["case_id"],
            condition=call["condition"],
            payload=payload,
        )
    except ValueError:
        if not force:
            raise ValueError(
                f"existing raw response for "
                f"{_call_key(call['case_id'], call['condition'])} is "
                f"malformed or unbound to its payload; use --force to rerun",
            )
        return False, None
    return True, response


def _write_call_log(
    *,
    run_dir: Path,
    case_id: str,
    condition: str,
    attempt: int,
    argv: list[str],
    completed: subprocess.CompletedProcess,
) -> Path:
    log_path = (
        run_dir / LOGS_SUBDIR
        / f"{_raw_filename(case_id, condition)}.attempt-{attempt}.log"
    )

    def _cap(data: bytes, limit: int = 64 * 1024) -> str:
        text = data.decode("utf-8", errors="replace")
        if len(text) > limit:
            text = text[:limit] + f"\n...[truncated {len(data)} bytes]"
        return text

    body = (
        f"argv={json.dumps(argv)}\n"
        f"returncode={completed.returncode}\n"
        "--- stdout ---\n"
        f"{_cap(completed.stdout or b'')}\n"
        "--- stderr ---\n"
        f"{_cap(completed.stderr or b'')}\n"
    )
    _write_bytes_atomic(log_path, body.encode("utf-8"))
    return log_path


def _execute_one_call(
    call: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    run_dir: Path,
    retries: int,
) -> dict[str, Any]:
    case_id = call["case_id"]
    condition = call["condition"]
    prompt_text, prompt_sha256 = _prompt_for_payload(payload)
    stdin_bytes = prompt_text.encode("utf-8")
    raw_path = _raw_path_for_call(run_dir, case_id, condition)
    started_at = _now_iso()
    attempts = 0
    last_error: str | None = None
    last_log_path: Path | None = None
    last_temp_dir: Path | None = None
    while attempts <= retries:
        attempts += 1
        work_dir = Path(tempfile.mkdtemp(prefix="phase2j-sol-"))
        schema_copy = work_dir / "response-schema.json"
        schema_copy.write_bytes(_schema_bytes())
        temp_raw = work_dir / "raw-response.json"
        argv = _codex_argv(
            work_dir=work_dir,
            schema_path=schema_copy,
            raw_path=temp_raw,
        )
        completed = subprocess.run(
            argv,
            input=stdin_bytes,
            capture_output=True,
            cwd=work_dir,
            check=False,
            shell=False,
        )
        log_path = _write_call_log(
            run_dir=run_dir,
            case_id=case_id,
            condition=condition,
            attempt=attempts,
            argv=argv,
            completed=completed,
        )
        last_log_path = log_path
        if completed.returncode != 0:
            last_error = (
                f"codex exec returned {completed.returncode}; see {log_path}"
            )
            last_temp_dir = work_dir
            continue
        if not temp_raw.is_file():
            last_error = (
                f"codex exec succeeded but wrote no raw response file; "
                f"see {log_path}"
            )
            last_temp_dir = work_dir
            continue
        try:
            response = load_json_strict(temp_raw, label="raw sol response")
            validate_sol_intermediate_response(
                response,
                case_id=case_id,
                condition=condition,
                payload=payload,
            )
        except ValueError as exc:
            last_error = (
                f"codex exec response failed validation: {exc}; see {log_path}"
            )
            last_temp_dir = work_dir
            continue
        _write_bytes_atomic(raw_path, temp_raw.read_bytes())
        shutil.rmtree(work_dir, ignore_errors=True)
        return {
            "case_id": case_id,
            "condition": condition,
            "payload_sha256": payload["content_sha256"],
            "prompt_sha256": prompt_sha256,
            "status": "completed",
            "raw_response_path": _relative_to_run_dir(raw_path, run_dir),
            "raw_response_sha256": file_sha256(raw_path),
            "attempts": attempts,
            "last_error": None,
            "log_path": None,
            "temp_dir": None,
            "started_at": started_at,
            "completed_at": _now_iso(),
        }
    return {
        "case_id": case_id,
        "condition": condition,
        "payload_sha256": payload["content_sha256"],
        "prompt_sha256": prompt_sha256,
        "status": "failed",
        "raw_response_path": None,
        "raw_response_sha256": None,
        "attempts": attempts,
        "last_error": last_error,
        "log_path": _relative_to_run_dir(last_log_path, run_dir)
        if last_log_path is not None else None,
        "temp_dir": str(last_temp_dir) if last_temp_dir is not None else None,
        "started_at": started_at,
        "completed_at": _now_iso(),
    }


def _merge_call_result(
    manifest: dict[str, Any],
    call: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    key = _call_key(call["case_id"], call["condition"])
    for entry in manifest["calls"]:
        if _call_key(entry["case_id"], entry["condition"]) == key:
            entry.update(dict(result))
            return
    raise ValueError(f"call {key} is missing from the run manifest")


def _cmd_schema(args: argparse.Namespace) -> int:
    _validate_frozen_artifacts(args)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    schema_path = run_dir / SCHEMA_FILENAME
    _ensure_schema_file(schema_path, force=args.force)
    print(json.dumps({
        "command": "schema",
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": build_sol_intermediate_schema()["schema_version"],
        "schema_sha256": canonical_sha256(build_sol_intermediate_schema()),
        "schema_path": str(schema_path),
        "run_dir": str(run_dir),
    }, sort_keys=True, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if not 1 <= args.max_workers <= MAX_WORKERS_LIMIT:
        raise ValueError(
            f"--max-workers must be between 1 and {MAX_WORKERS_LIMIT}",
        )
    if args.retries < 0:
        raise ValueError("--retries must be >= 0")
    _validate_frozen_artifacts(args)
    payloads = _load_payloads(args.output_dir)
    codex_cli_version = _codex_cli_version()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    schema_path = run_dir / SCHEMA_FILENAME
    _ensure_schema_file(schema_path, force=args.force)
    manifest_path = run_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest = load_json_strict(manifest_path, label="sol run manifest")
        _validate_run_manifest(
            manifest,
            codex_cli_version=codex_cli_version,
            run_dir=run_dir,
            payloads=payloads,
        )
    else:
        manifest = _new_run_manifest(
            codex_cli_version=codex_cli_version,
            run_dir=run_dir,
            payloads=payloads,
        )
        _write_json_atomic(manifest_path, manifest)
    payload_by_call = {
        _call_key(payload_case["case_id"], condition): payload_case[condition]
        for payload_case in payloads["cases"]
        for condition in CONDITION_CODES
    }
    to_run: list[dict[str, Any]] = []
    for call in manifest["calls"]:
        payload = payload_by_call[_call_key(call["case_id"], call["condition"])]
        reuse, _ = _existing_raw_decision(
            call,
            run_dir=run_dir,
            payload=payload,
            force=args.force,
        )
        if not reuse:
            to_run.append(call)
    if not to_run:
        print(json.dumps({
            "command": "run",
            "pending": 0,
            "completed": 20,
            "message": "all calls already have valid raw responses",
        }, sort_keys=True, indent=2))
        return 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers,
    ) as pool:
        futures = {
            pool.submit(
                _execute_one_call,
                call,
                payload=payload_by_call[
                    _call_key(call["case_id"], call["condition"])
                ],
                run_dir=run_dir,
                retries=args.retries,
            ): call
            for call in to_run
        }
        for future in concurrent.futures.as_completed(futures):
            call = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - record unexpected failure
                result = {
                    "status": "failed",
                    "attempts": 1,
                    "last_error": f"unexpected runner failure: {exc}",
                    "prompt_sha256": None,
                    "raw_response_path": None,
                    "raw_response_sha256": None,
                    "log_path": None,
                    "temp_dir": None,
                    "started_at": _now_iso(),
                    "completed_at": _now_iso(),
                }
            _merge_call_result(manifest, call, result)
            manifest["content_sha256"] = canonical_sha256({
                key: value for key, value in manifest.items()
                if key != "content_sha256"
            })
            _write_json_atomic(manifest_path, manifest)
    completed = sum(
        1 for call in manifest["calls"] if call["status"] == "completed"
    )
    failed = [
        _call_key(call["case_id"], call["condition"])
        for call in manifest["calls"] if call["status"] == "failed"
    ]
    summary = {
        "command": "run",
        "pending": 20 - completed - len(failed),
        "completed": completed,
        "failed": failed,
        "manifest_path": str(manifest_path),
        "run_dir": str(run_dir),
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    if failed:
        print(
            "[phase2j-context-ablation-sol] the following calls failed: "
            + ", ".join(failed),
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    _validate_frozen_artifacts(args)
    payloads = _load_payloads(args.output_dir)
    instructions = _load_instructions(args.output_dir)
    codex_cli_version = _codex_cli_version()
    run_dir = Path(args.run_dir)
    schema_path = run_dir / SCHEMA_FILENAME
    _ensure_schema_file(schema_path, force=False)
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(f"sol run manifest does not exist: {manifest_path}")
    manifest = load_json_strict(manifest_path, label="sol run manifest")
    _validate_run_manifest(
        manifest,
        codex_cli_version=codex_cli_version,
        run_dir=run_dir,
        payloads=payloads,
    )
    payload_by_call = {
        _call_key(payload_case["case_id"], condition): payload_case[condition]
        for payload_case in payloads["cases"]
        for condition in CONDITION_CODES
    }
    outputs_by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for call in manifest["calls"]:
        key = _call_key(call["case_id"], call["condition"])
        if call["status"] != "completed":
            raise ValueError(f"call {key} is not completed")
        raw_path = run_dir / call["raw_response_path"]
        if not raw_path.is_file():
            raise ValueError(f"raw response for {key} is missing: {raw_path}")
        if file_sha256(raw_path) != call["raw_response_sha256"]:
            raise ValueError(
                f"raw response hash for {key} does not match the manifest",
            )
        response = load_json_strict(raw_path, label=f"raw response {key}")
        payload = payload_by_call[key]
        outputs_by_case.setdefault(call["case_id"], {})[call["condition"]] = (
            import_sol_intermediate_response(
                response,
                case_id=call["case_id"],
                condition=call["condition"],
                payload=payload,
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
    final_outputs = {
        "outputs_path": normalize_path_locator(out_path),
        "outputs_sha256": bundle["content_sha256"],
        "outputs_file_sha256": file_sha256(out_path),
        "by_call": {
            _call_key(case_id, condition): output["content_sha256"]
            for case_id, conditions in outputs_by_case.items()
            for condition, output in conditions.items()
        },
    }
    manifest["final_outputs"] = final_outputs
    manifest["content_sha256"] = canonical_sha256({
        key: value for key, value in manifest.items()
        if key != "content_sha256"
    })
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps({
        "command": "import",
        "outputs_path": str(out_path),
        "outputs_sha256": bundle["content_sha256"],
        "outputs_file_sha256": final_outputs["outputs_file_sha256"],
        "imported_calls": 20,
        "manifest_path": str(manifest_path),
    }, sort_keys=True, indent=2))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    _validate_frozen_artifacts(args)
    payloads = _load_payloads(args.output_dir)
    codex_cli_version = _codex_cli_version()
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
    manifest = load_json_strict(manifest_path, label="sol run manifest")
    _validate_run_manifest(
        manifest,
        codex_cli_version=codex_cli_version,
        run_dir=run_dir,
        payloads=payloads,
    )
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
                "raw_response_sha256": call["raw_response_sha256"],
                "attempts": call["attempts"],
            }
            for call in manifest["calls"]
        ],
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
            "Execute/import the 20 independent Phase 2J context-ablation Sol "
            "calls against the validated DB-only v2 payloads (no model calls "
            "made by this program itself)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema_parser = subparsers.add_parser(
        "schema", help="Write the canonical intermediate response schema.",
    )
    _common_parser(schema_parser)
    schema_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser(
        "run", help="Run all pending Sol calls and record raw responses.",
    )
    _common_parser(run_parser)
    run_parser.add_argument(
        "--max-workers", type=int, default=MAX_WORKERS_DEFAULT,
        help=(
            f"Maximum parallel codex exec processes (default "
            f"{MAX_WORKERS_DEFAULT}, bounded 1..{MAX_WORKERS_LIMIT})."
        ),
    )
    run_parser.add_argument(
        "--retries", type=int, default=0,
        help=(
            "Additional identical attempts per failing call (same prompt and "
            "config; default 0)."
        ),
    )
    run_parser.add_argument(
        "--force", action="store_true",
        help=(
            "Replace only mismatched/malformed per-call artifacts in the "
            "configured run dir (valid completed results are still reused)."
        ),
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
            "schema": _cmd_schema,
            "run": _cmd_run,
            "import": _cmd_import,
            "status": _cmd_status,
        }
        return handlers[args.command](args)
    except (OSError, ValueError) as exc:
        print(f"[phase2j-context-ablation-sol] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
