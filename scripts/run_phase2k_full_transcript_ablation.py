#!/usr/bin/env python3
"""Execute and evaluate the Phase 2K full-transcript context ablation.

Both conditions use 0x Alpha through the OpenCode CLI
(``opencode-go/ox-alpha-free``).  Condition A receives only the isolated
Bronze target plus useful reliable metadata plus the League vocabulary.
Condition B additionally receives the FULL ordered transcript with the
target's character offsets.  The prompt, schema, vocabulary, target,
metadata policy, parsing, and evaluation contract are byte-identical across
conditions; only discourse context differs.

Subcommands:

  build     Freeze selection + instructions + A/B payloads (no model calls).
  init      Validate frozen artifacts and create the 20-pending-call run
            manifest (10 cases x conditions A/B).
  prompt    Print the exact experiment user message bytes for one call.
  run       Execute all pending calls through ``opencode run``.  Valid
            existing raw responses for the exact prompt/model/config are
            reused; mismatches fail closed unless ``--force``.
  status    Report per-call status without executing anything.
  import    Require all 20 valid completed calls and deterministically
            assemble the validated extraction outputs bundle.
  review    Generate the human-review packet artifact and its Markdown
            rendering from the outputs bundle.
  evaluate  Import completed structured reviews and freeze the aggregate
            evaluation summary with the preregistered decision gate.

No model outputs are claimed to be deterministic.  The run manifest records
the requested model, OpenCode CLI version, argv template, wrapper/schema/
payload/prompt hashes, and raw-response hashes as requested-model evidence
only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2j_context_ablation import (
    _reject_constant,
    _unique_pairs,
    canonical_sha256,
    file_sha256,
    load_json_strict,
    load_lexical_vocabulary,
    normalize_path_locator,
    open_transcript_db,
    text_sha256,
)
from pipeline.phase2k_full_transcript_ablation import (
    CONDITION_CODES,
    DEFAULT_DB_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VOCABULARY_PATH,
    PIPELINE_VERSION,
    RUN_SCHEMA_VERSION,
    ArtifactError,
    build_build_summary,
    build_case_vocabulary,
    build_condition_payloads,
    build_extraction_instructions,
    build_intermediate_schema,
    build_outputs_bundle,
    build_payloads_artifact,
    build_review_packet,
    build_selection_artifact,
    champion_abilities_for_transcript,
    compute_evaluation_summary,
    fetch_source_rows,
    import_intermediate_response,
    load_phase2j_manifest,
    select_phase2k_cases,
    validate_completed_reviews,
    validate_instructions_artifact,
    validate_intermediate_response,
    validate_outputs_bundle,
    validate_payloads_artifact,
    validate_review_packet,
    validate_selection_artifact,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL = "opencode-go/ox-alpha-free"
MODEL_VARIANT = None

DEFAULT_RUN_DIR = DEFAULT_OUTPUT_DIR / "opencode_run_v1"
MANIFEST_FILENAME = "manifest.json"
SCHEMA_FILENAME = "intermediate-schema.json"
RAW_SUBDIR = "raw"

SELECTION_FILENAME = "phase2k-context-ablation-selection-v1.json"
INSTRUCTIONS_FILENAME = (
    "phase2k-context-ablation-extraction-instructions-v1.json"
)
PAYLOADS_FILENAME = "phase2k-context-ablation-condition-payloads-v1.json"
OUTPUTS_FILENAME = "phase2k-context-ablation-extraction-outputs-v1.json"
BUILD_SUMMARY_FILENAME = "phase2k-context-ablation-build-summary-v1.json"
REVIEW_PACKET_FILENAME = "phase2k-context-ablation-review-packet-v1.json"
REVIEW_MARKDOWN_FILENAME = "phase2k-context-ablation-review-v1.md"
EVALUATION_SUMMARY_FILENAME = (
    "phase2k-context-ablation-evaluation-summary-v1.json"
)

MAX_WORKERS_LIMIT = 4
CALL_TIMEOUT_SECONDS = 3600

WRAPPER_PROMPT = (
    "You are a deterministic extraction worker for a controlled "
    "source-grounding experiment.  The supplied JSON payload embeds the "
    "exact extraction instructions and one experimental condition context.\n"
    "\n"
    "Follow the embedded instructions exactly.  Return ONLY a single JSON "
    "object matching the output contract embedded in the instructions.  Do "
    "not add commentary, markdown, explanations, or any text outside the "
    "JSON object.\n"
    "\n"
    "Every source citation must be an exact contiguous quote copied "
    "verbatim from the supplied condition source text, together with a "
    "zero-based occurrence_index counted among all exact non-overlapping "
    "substring matches of that quote in the condition source."
)


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _write_json_atomic(path: Path, value: object) -> None:
    _write_text_atomic(path, json.dumps(value, indent=1) + "\n")


def extract_response_json(text: str) -> dict:
    """Deterministic strict parse of a raw model response.

    The response must be a single JSON object.  One optional markdown fence
    pair (```json ... ``` or ``` ... ```) is stripped before parsing.
    Literal control characters inside strings are tolerated (``strict=False``)
    because ASR-derived quotes frequently contain them; duplicate keys and
    non-finite constants are still rejected, and every downstream semantic
    constraint remains enforced.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline < 0:
            raise ArtifactError("response fence is unterminated")
        header = stripped[:first_newline].strip()
        if header not in ("```", "```json"):
            raise ArtifactError("response fence header is not allowed")
        if not stripped.endswith("```"):
            raise ArtifactError("response fence is unterminated")
        stripped = stripped[first_newline + 1:-len("```")].strip()
    try:
        body = json.loads(
            stripped,
            object_pairs_hook=_unique_pairs("model response"),
            parse_constant=_reject_constant,
            strict=False,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"model response JSON is malformed: {exc}") from exc
    if not isinstance(body, dict):
        raise ArtifactError("model response must be a JSON object")
    return body


def call_prompt_bytes(payload: dict) -> bytes:
    message = WRAPPER_PROMPT + "\n\n" + canonical_json(payload)
    return message.encode("utf-8")


# ---------------------------------------------------------------------------
# Frozen artifact loading and validation (DB-backed)
# ---------------------------------------------------------------------------


def load_frozen(
    *,
    output_dir: Path,
    db_path: Path,
    vocabulary_path: Path,
) -> dict:
    selection_path = output_dir / SELECTION_FILENAME
    instructions_path = output_dir / INSTRUCTIONS_FILENAME
    payloads_path = output_dir / PAYLOADS_FILENAME
    for path in (selection_path, instructions_path, payloads_path):
        if not path.exists():
            raise SystemExit(
                f"frozen artifact {path} is missing; run the build subcommand",
            )
    selection = load_json_strict(selection_path, label="selection")
    instructions = load_json_strict(instructions_path, label="instructions")
    payloads = load_json_strict(payloads_path, label="payloads")
    manifest = load_phase2j_manifest(DEFAULT_MANIFEST_PATH)
    lexical_vocabulary = load_lexical_vocabulary(vocabulary_path)
    connection = open_transcript_db(db_path)
    try:
        validate_selection_artifact(
            selection,
            manifest_path=DEFAULT_MANIFEST_PATH,
            manifest=manifest,
            db_path=db_path,
            connection=connection,
        )
        validate_instructions_artifact(instructions)
        validate_payloads_artifact(
            payloads,
            selection=selection,
            instructions=instructions,
            lexical_vocabulary=lexical_vocabulary,
            manifest=manifest,
            connection=connection,
        )
    finally:
        connection.close()
    return {
        "output_dir": output_dir,
        "db_path": db_path,
        "vocabulary_path": vocabulary_path,
        "manifest": manifest,
        "lexical_vocabulary": lexical_vocabulary,
        "selection": selection,
        "instructions": instructions,
        "payloads": payloads,
    }


def cmd_build(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    db_path = Path(args.db)
    vocabulary_path = Path(args.vocabulary)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_phase2j_manifest(DEFAULT_MANIFEST_PATH)
    lexical_vocabulary = load_lexical_vocabulary(vocabulary_path)
    connection = open_transcript_db(db_path)
    try:
        source_rows = fetch_source_rows(connection, manifest["selected"])
        cases = select_phase2k_cases(manifest, source_rows=source_rows)
        selection = build_selection_artifact(
            manifest_path=DEFAULT_MANIFEST_PATH,
            manifest=manifest,
            db_path=db_path,
            cases=cases,
        )
        validate_selection_artifact(
            selection,
            manifest_path=DEFAULT_MANIFEST_PATH,
            manifest=manifest,
            db_path=db_path,
            connection=connection,
        )
        instructions = build_extraction_instructions()
        validate_instructions_artifact(instructions)
        vocabulary_by_case = {}
        for case in cases:
            case_id = case["case_id"]
            row = source_rows[case["upstream_source_id"]]
            champion_data = champion_abilities_for_transcript(
                connection,
                metadata_champion=row["champion"],
                transcript=row["transcript"],
                video_id=row["video_id"],
            )
            vocabulary_by_case[case_id] = build_case_vocabulary(
                case_id=case_id,
                lexical_vocabulary=lexical_vocabulary,
                champion_data=champion_data,
            )
        payload_cases, provenance_by_case = build_condition_payloads(
            cases=cases,
            source_rows=source_rows,
            vocabulary_by_case=vocabulary_by_case,
            instructions=instructions,
        )
        payloads = build_payloads_artifact(
            selection=selection,
            instructions=instructions,
            payload_cases=payload_cases,
            provenance_by_case=provenance_by_case,
        )
        validate_payloads_artifact(
            payloads,
            selection=selection,
            instructions=instructions,
            lexical_vocabulary=lexical_vocabulary,
            manifest=manifest,
            connection=connection,
        )
    finally:
        connection.close()
    summary = build_build_summary(
        output_dir=output_dir,
        selection=selection,
        instructions=instructions,
        payloads=payloads,
        mode="ready_for_opencode",
    )
    _write_json_atomic(output_dir / SELECTION_FILENAME, selection)
    _write_json_atomic(output_dir / INSTRUCTIONS_FILENAME, instructions)
    _write_json_atomic(output_dir / PAYLOADS_FILENAME, payloads)
    _write_json_atomic(output_dir / BUILD_SUMMARY_FILENAME, summary)
    print(f"built {len(payloads['cases'])} case pairs in {output_dir}")
    print(f"mode: {summary['mode']}")
    return 0


# ---------------------------------------------------------------------------
# Run manifest handling
# ---------------------------------------------------------------------------


def _all_calls(payloads: dict) -> list[dict]:
    calls = []
    for pair in payloads["cases"]:
        for condition in CONDITION_CODES:
            payload = pair[condition]
            prompt_sha = text_sha256(call_prompt_bytes(payload).decode("utf-8"))
            calls.append({
                "case_id": pair["case_id"],
                "condition": condition,
                "payload_sha256": payload["content_sha256"],
                "prompt_sha256": prompt_sha,
                "status": "pending",
            })
    return calls


def _capture_opencode_version() -> str:
    try:
        proc = subprocess.run(
            ["opencode", "--version"],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    return (proc.stdout or proc.stderr).strip() or "unavailable"


def _load_run_manifest(run_dir: Path) -> dict | None:
    path = run_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    manifest = load_json_strict(path, label="run manifest")
    if manifest["schema_version"] != RUN_SCHEMA_VERSION:
        raise ArtifactError("run manifest schema version is invalid")
    return manifest


def cmd_init(args: argparse.Namespace) -> int:
    frozen = load_frozen(
        output_dir=Path(args.output_dir),
        db_path=Path(args.db),
        vocabulary_path=Path(args.vocabulary),
    )
    payloads = frozen["payloads"]
    run_dir = Path(args.run_dir)
    existing = _load_run_manifest(run_dir)
    if existing is not None:
        if existing["payloads_sha256"] != payloads["content_sha256"]:
            raise SystemExit(
                "existing run manifest binds different frozen payloads; "
                "use a new run directory",
            )
        if existing["requested_model"] != MODEL:
            raise SystemExit(
                "existing run manifest binds a different requested model",
            )
        print(f"run manifest already exists at {run_dir}")
        return 0
    schema = build_intermediate_schema()
    opencode_version = _capture_opencode_version()
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "purpose": (
            "Phase 2K full-transcript ablation OpenCode CLI run manifest.  "
            "The backend identity is requested/recorded but not "
            "cryptographically proven."
        ),
        "transport": "opencode_cli_v1",
        "requested_model": MODEL,
        "model_variant": MODEL_VARIANT,
        "opencode_cli_version": opencode_version,
        "argv_template": ["opencode", "run", "--model", MODEL],
        "prompt_transport": "stdin",
        "wrapper_prompt_sha256": text_sha256(WRAPPER_PROMPT),
        "intermediate_schema_sha256": canonical_sha256(schema),
        "pipeline_version": PIPELINE_VERSION,
        "payloads_sha256": payloads["content_sha256"],
        "instructions_sha256": payloads["instructions_sha256"],
        "final_outputs": None,
        "calls": _all_calls(payloads),
    }
    _write_json_atomic(run_dir / SCHEMA_FILENAME, schema)
    _write_json_atomic(run_dir / MANIFEST_FILENAME, manifest)
    pending = sum(1 for call in manifest["calls"] if call["status"] == "pending")
    print(f"initialized {len(manifest['calls'])}-call manifest "
          f"({pending} pending) in {run_dir}")
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    frozen = load_frozen(
        output_dir=Path(args.output_dir),
        db_path=Path(args.db),
        vocabulary_path=Path(args.vocabulary),
    )
    payload = _find_payload(frozen["payloads"], args.case_id, args.condition)
    sys.stdout.buffer.write(call_prompt_bytes(payload))
    sys.stdout.buffer.flush()
    return 0


def _find_payload(payloads: dict, case_id: str, condition: str) -> dict:
    if condition not in CONDITION_CODES:
        raise SystemExit(f"unknown condition {condition!r}")
    for pair in payloads["cases"]:
        if pair["case_id"] == case_id:
            return pair[condition]
    raise SystemExit(f"unknown case id {case_id!r}")


class RunContext:
    def __init__(self, *, frozen: dict, run_dir: Path, force: bool) -> None:
        self.frozen = frozen
        self.run_dir = run_dir
        self.force = force
        self.lock = threading.Lock()

    def raw_path(self, case_id: str, condition: str) -> Path:
        return self.run_dir / RAW_SUBDIR / f"{case_id}_{condition}.json"


def _execute_call(context: RunContext, call: dict) -> None:
    case_id = call["case_id"]
    condition = call["condition"]
    payload = _find_payload(
        context.frozen["payloads"], case_id, condition,
    )
    expected_prompt_sha = text_sha256(
        call_prompt_bytes(payload).decode("utf-8"),
    )
    if expected_prompt_sha != call["prompt_sha256"]:
        raise SystemExit(
            f"prompt hash drift for {case_id}/{condition}; rebuild required",
        )
    prompt_bytes = call_prompt_bytes(payload)
    with tempfile.TemporaryDirectory(prefix="p2k-ablation-") as workspace:
        argv = ["opencode", "run", "--model", MODEL]
        try:
            proc = subprocess.run(
                argv,
                input=prompt_bytes,
                capture_output=True,
                cwd=workspace,
                timeout=CALL_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"call {case_id}/{condition} timed out",
            ) from exc
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or b"").decode("utf-8", "replace")[-2000:]
        raise RuntimeError(
            f"call {case_id}/{condition} failed with exit code "
            f"{proc.returncode}: {stderr_tail}",
        )
    stdout_text = proc.stdout.decode("utf-8")
    parsed = extract_response_json(stdout_text)
    # Full import-time grounding check before recording anything.
    import_intermediate_response(
        parsed, case_id=case_id, condition=condition, payload=payload,
    )
    raw_file = context.raw_path(case_id, condition)
    _write_text_atomic(raw_file, stdout_text)
    with context.lock:
        manifest = _load_run_manifest(context.run_dir)
        target = next(
            entry for entry in manifest["calls"]
            if entry["case_id"] == case_id
            and entry["condition"] == condition
        )
        target.update({
            "status": "completed",
            "raw_path": str(raw_file.relative_to(context.run_dir)),
            "raw_response_sha256": file_sha256(raw_file),
            "completed_at": _now_iso(),
        })
        _write_json_atomic(context.run_dir / MANIFEST_FILENAME, manifest)


def _validated_existing_call(context: RunContext, call: dict) -> bool:
    """Return True when the recorded completed call remains valid evidence.

    Validation includes full import-time quote resolution, so a recorded
    response whose citations do not byte-resolve against its source is
    treated as invalid and must be re-executed with ``--force``.
    """
    if call.get("status") != "completed":
        return False
    payload = _find_payload(
        context.frozen["payloads"], call["case_id"], call["condition"],
    )
    current_prompt_sha = text_sha256(
        call_prompt_bytes(payload).decode("utf-8"),
    )
    if current_prompt_sha != call.get("prompt_sha256"):
        raise SystemExit(
            f"completed call {call['case_id']}/{call['condition']} binds a "
            "different prompt; the cache is stale.  Use a fresh run "
            "directory or --force.",
        )
    raw_file = context.run_dir / call["raw_path"]
    resolved_parent = raw_file.resolve().parent
    if context.run_dir.resolve() not in resolved_parent.parents:
        raise SystemExit(
            f"recorded raw path escapes the run dir: {call['raw_path']}",
        )
    if not raw_file.exists():
        return False
    if file_sha256(raw_file) != call.get("raw_response_sha256"):
        if not context.force:
            raise SystemExit(
                f"raw response for {call['case_id']}/{call['condition']} "
                "changed on disk; use --force to replace it",
            )
        return False
    stdout_text = raw_file.read_text(encoding="utf-8")
    try:
        parsed = extract_response_json(stdout_text)
        import_intermediate_response(
            parsed,
            case_id=call["case_id"],
            condition=call["condition"],
            payload=payload,
        )
    except (ValueError, ArtifactError):
        if not context.force:
            raise SystemExit(
                f"recorded raw response for {call['case_id']}/"
                f"{call['condition']} fails citation grounding; rerun with "
                "--force to replace exactly this call",
            )
        return False
    return True


def cmd_run(args: argparse.Namespace) -> int:
    frozen = load_frozen(
        output_dir=Path(args.output_dir),
        db_path=Path(args.db),
        vocabulary_path=Path(args.vocabulary),
    )
    run_dir = Path(args.run_dir)
    manifest = _load_run_manifest(run_dir)
    if manifest is None:
        raise SystemExit("run manifest missing; run the init subcommand")
    if manifest["payloads_sha256"] != frozen["payloads"]["content_sha256"]:
        raise SystemExit("run manifest does not bind the frozen payloads")
    if manifest["requested_model"] != MODEL:
        raise SystemExit("run manifest binds a different requested model")
    workers = max(1, min(int(args.max_workers), MAX_WORKERS_LIMIT))
    context = RunContext(run_dir=run_dir, frozen=frozen, force=args.force)
    pending_calls = []
    for call in manifest["calls"]:
        is_completed = call.get("status") == "completed"
        valid = _validated_existing_call(context, call)
        if valid:
            continue
        if is_completed and not args.force:
            continue
        pending_calls.append(dict(call))
    if not pending_calls:
        print("no pending calls; all completed calls validated")
        return 0
    failures: list[str] = []
    if workers == 1:
        for call in pending_calls:
            try:
                _execute_call(context, call)
                print(f"completed {call['case_id']}/{call['condition']}")
            except Exception as exc:  # noqa: BLE001 - runner reports and stops
                failures.append(
                    f"{call['case_id']}/{call['condition']}: {exc}",
                )
                break
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                (call, pool.submit(_execute_call, context, call))
                for call in pending_calls
            ]
            for call, future in futures:
                try:
                    future.result()
                    print(f"completed {call['case_id']}/{call['condition']}")
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        f"{call['case_id']}/{call['condition']}: {exc}",
                    )
                    break
            pool.shutdown(wait=True, cancel_futures=True)
    if failures:
        for failure in failures:
            print(f"FAILURE {failure}", file=sys.stderr)
        return 1
    refreshed = _load_run_manifest(run_dir)
    remaining = sum(
        1 for call in refreshed["calls"] if call["status"] == "pending"
    )
    print(f"remaining pending calls: {remaining}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    manifest = _load_run_manifest(run_dir)
    if manifest is None:
        raise SystemExit("run manifest missing")
    completed = 0
    for call in manifest["calls"]:
        state = call.get("status", "pending")
        if state == "completed":
            completed += 1
        print(
            f"{call['case_id']}/{call['condition']}: {state}",
        )
    total = len(manifest["calls"])
    print(f"{completed}/{total} completed")
    if manifest.get("final_outputs"):
        print(f"final outputs: {manifest['final_outputs']}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    frozen = load_frozen(
        output_dir=Path(args.output_dir),
        db_path=Path(args.db),
        vocabulary_path=Path(args.vocabulary),
    )
    run_dir = Path(args.run_dir)
    manifest = _load_run_manifest(run_dir)
    if manifest is None:
        raise SystemExit("run manifest missing")
    payloads = frozen["payloads"]
    outputs_by_call: dict[tuple[str, str], dict] = {}
    by_call_hashes: dict[str, str] = {}
    for call in manifest["calls"]:
        case_id = call["case_id"]
        condition = call["condition"]
        if call.get("status") != "completed":
            raise SystemExit(
                f"incomplete call {case_id}/{condition}; run first",
            )
        raw_file = run_dir / call["raw_path"]
        if file_sha256(raw_file) != call["raw_response_sha256"]:
            raise SystemExit(
                f"raw response hash mismatch for {case_id}/{condition}",
            )
        stdout_text = raw_file.read_text(encoding="utf-8")
        parsed = extract_response_json(stdout_text)
        payload = _find_payload(payloads, case_id, condition)
        validate_intermediate_response(
            parsed, case_id=case_id, condition=condition, payload=payload,
        )
        output = import_intermediate_response(
            parsed, case_id=case_id, condition=condition, payload=payload,
        )
        outputs_by_call[(case_id, condition)] = output
        by_call_hashes[f"{case_id}:{condition}"] = {
            "raw_response_sha256": call["raw_response_sha256"],
            "prompt_sha256": call["prompt_sha256"],
            "payload_sha256": call["payload_sha256"],
            "output_content_sha256": output["content_sha256"],
        }
    bundle = build_outputs_bundle(
        payloads=payloads,
        outputs_by_call=outputs_by_call,
        by_call_evidence=by_call_hashes,
    )
    validate_outputs_bundle(bundle, payloads=payloads)
    output_path = Path(args.output_dir) / OUTPUTS_FILENAME
    _write_json_atomic(output_path, bundle)
    with context_lock():
        refreshed = _load_run_manifest(run_dir)
        refreshed["final_outputs"] = {
            "path": normalize_path_locator(output_path),
            "file_sha256": file_sha256(output_path),
            "content_sha256": bundle["content_sha256"],
            "imported_at": _now_iso(),
        }
        _write_json_atomic(run_dir / MANIFEST_FILENAME, refreshed)
    print(f"wrote {output_path}")
    return 0


_LOCK = threading.Lock()


def context_lock() -> threading.Lock:
    return _LOCK


# ---------------------------------------------------------------------------
# Review packet generation
# ---------------------------------------------------------------------------


def _render_field_table(
    title: str,
    rendered: list[dict],
    source: str,
) -> list[str]:
    lines = [f"### {title}", ""]
    for field_entry in rendered:
        field = field_entry["field"]
        items = field_entry["items"]
        lines.append(f"#### {field} ({len(items)} items)")
        if not items:
            lines.append("")
            lines.append("(none)")
            lines.append("")
            continue
        for index, item in enumerate(items, 1):
            lines.append(
                f"- [{index}] {item['extraction_text']} "
                f"(resolution: {item['resolution_status']}"
                + (
                    f"; relation: {item['relation_type']}"
                    if "relation_type" in item else ""
                ) + ")",
            )
            for citation in item["citations"]:
                verified = (
                    "byte-exact" if citation["verified_byte_exact"]
                    else "NOT-BYTE-EXACT"
                )
                excerpt = citation["quote"]
                if len(excerpt) > 120:
                    excerpt = excerpt[:117] + "..."
                lines.append(
                    f"  - cite[{citation['char_start']}:"
                    f"{citation['char_end']}] {verified}: {excerpt!r}",
                )
            if "span" in item:
                lines.append(
                    f"  - span [{item['span']['char_start']}:"
                    f"{item['span']['char_end']}]",
                )
        lines.append("")
    return lines


def render_review_markdown(packet: dict) -> str:
    lines = [
        "# Phase 2K Full-Transcript Ablation — Human Review Packet",
        "",
        packet["purpose"],
        "",
        "Scoring scales:",
        "",
        f"- correctness: {', '.join(packet['scoring_scales']['correctness'])}",
        "- unsupported_inference: "
        + ", ".join(packet["scoring_scales"]["unsupported_inference"]),
        "- source_grounding: "
        + ", ".join(packet["scoring_scales"]["source_grounding"]),
        "",
        "Strict success = " + packet["strict_success_definition"],
        "",
        "Do NOT score prose quality.  A more fluent answer earns nothing "
        "unless its semantic recovery is actually better.",
        "",
    ]
    for case in packet["cases"]:
        bronze = case["target"]["bronze_text"]
        lines += [
            "---",
            "",
            f"## TARGET {case['case_id']} "
            f"(selection rank {case['selection_rank']})",
            "",
            "### Exact Bronze",
            "",
            "```text",
            bronze,
            "```",
            "",
            f"Location in full transcript: "
            f"[{case['target']['char_start']}:{case['target']['char_end']}]",
            "",
            f"Metadata supplied: "
            f"{', '.join(case['metadata_fields_supplied'])}",
            "",
        ]
        section_a = case["condition_A_isolated_bronze"]
        section_b = case["condition_B_full_transcript"]
        lines += _render_field_table(
            "Condition A — Isolated Bronze",
            section_a["structured_extraction"],
            bronze,
        )
        transcript = None  # transcript intentionally not rendered in full here
        del transcript
        lines += [
            "### Condition B — Full Transcript",
            "(B citations resolve against the FULL ordered transcript; "
            "byte-exactness was verified mechanically at import time.)",
            "",
        ]
        lines += _render_field_table(
            "Condition B — structured extraction",
            section_b["structured_extraction"],
            "transcript",
        )
        lines += [
            f"Raw response binding A: "
            f"`{section_a['raw_response_binding']['payload_sha256']}`",
            f"Raw response binding B: "
            f"`{section_b['raw_response_binding']['payload_sha256']}`",
            "",
        ]
    return "\n".join(lines)


def cmd_review(args: argparse.Namespace) -> int:
    frozen = load_frozen(
        output_dir=Path(args.output_dir),
        db_path=Path(args.db),
        vocabulary_path=Path(args.vocabulary),
    )
    outputs = load_json_strict(
        Path(args.output_dir) / OUTPUTS_FILENAME, label="outputs bundle",
    )
    validate_outputs_bundle(outputs, payloads=frozen["payloads"])
    packet = build_review_packet(
        payloads=frozen["payloads"], outputs=outputs,
    )
    validate_review_packet(packet)
    packet_path = Path(args.output_dir) / REVIEW_PACKET_FILENAME
    markdown_path = Path(args.output_dir) / REVIEW_MARKDOWN_FILENAME
    _write_json_atomic(packet_path, packet)
    _write_text_atomic(markdown_path, render_review_markdown(packet))
    print(f"wrote {packet_path}")
    print(f"wrote {markdown_path}")
    return 0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def cmd_evaluate(args: argparse.Namespace) -> int:
    frozen = load_frozen(
        output_dir=Path(args.output_dir),
        db_path=Path(args.db),
        vocabulary_path=Path(args.vocabulary),
    )
    packet = load_json_strict(
        Path(args.output_dir) / REVIEW_PACKET_FILENAME,
        label="review packet",
    )
    validate_review_packet(packet)
    completed = load_json_strict(
        Path(args.reviews), label="completed reviews",
    )
    validate_completed_reviews(completed, review_packet=packet)
    summary = compute_evaluation_summary(
        review_packet=packet, completed_reviews=completed,
    )
    output_path = Path(args.output_dir) / EVALUATION_SUMMARY_FILENAME
    _write_json_atomic(output_path, summary)
    decision = summary["decision_gate"]
    print(f"wrote {output_path}")
    print(f"decision gate: {decision}")
    counts = summary["target_verdict_counts"]
    print(
        f"targets: B wins {counts['B_STRICTLY_WINS']}, "
        f"A wins {counts['A_STRICTLY_WINS']}, "
        f"ties {counts['TIE']}",
    )
    return 0


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help="read-only archived transcript DB",
    )
    parser.add_argument(
        "--vocabulary",
        default=str(DEFAULT_VOCABULARY_PATH),
        help="League lexical vocabulary v2 snapshot",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser(
        "build", help="freeze selection/instructions/payloads",
    )
    p_build.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    add_common_arguments(p_build)
    p_build.set_defaults(func=cmd_build)

    p_init = subparsers.add_parser(
        "init", help="create the 20-pending-call run manifest",
    )
    p_init.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p_init.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    add_common_arguments(p_init)
    p_init.set_defaults(func=cmd_init)

    p_prompt = subparsers.add_parser(
        "prompt", help="print the exact user-message bytes for one call",
    )
    p_prompt.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p_prompt.add_argument("--case-id", required=True)
    p_prompt.add_argument("--condition", required=True, choices=list(CONDITION_CODES))
    add_common_arguments(p_prompt)
    p_prompt.set_defaults(func=cmd_prompt)

    p_run = subparsers.add_parser(
        "run", help="execute pending calls through the OpenCode CLI",
    )
    p_run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p_run.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    p_run.add_argument("--max-workers", default=1, type=int)
    p_run.add_argument("--retries", default=0, type=int)
    p_run.add_argument(
        "--force",
        action="store_true",
        help="replace invalid/mismatched completed calls",
    )
    add_common_arguments(p_run)
    p_run.set_defaults(func=cmd_run)

    p_status = subparsers.add_parser(
        "status", help="report per-call status",
    )
    p_status.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    p_status.set_defaults(func=cmd_status)

    p_import = subparsers.add_parser(
        "import", help="assemble the validated outputs bundle",
    )
    p_import.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p_import.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    add_common_arguments(p_import)
    p_import.set_defaults(func=cmd_import)

    p_review = subparsers.add_parser(
        "review", help="generate the human-review packet artifacts",
    )
    p_review.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    add_common_arguments(p_review)
    p_review.set_defaults(func=cmd_review)

    p_eval = subparsers.add_parser(
        "evaluate", help="freeze the aggregate evaluation summary",
    )
    p_eval.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p_eval.add_argument("--reviews", required=True)
    add_common_arguments(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
