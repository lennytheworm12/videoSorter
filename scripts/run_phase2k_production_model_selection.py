#!/usr/bin/env python3
"""Phase 2K production-model selection over the frozen full-context benchmark.

Conditions over the SAME frozen 10-target Phase 2K benchmark and the SAME
Condition B inputs (full transcript + marked target + metadata + vocabulary +
byte-identical extraction instructions):

  P   one independent DeepSeek V4 Pro extraction per target
  F   one independent DeepSeek V4 Flash extraction per target
  FV  five genuinely independent V4 Flash generator calls per target plus one
      selection-only Flash verifier call that MUST choose exactly one of the
      five existing candidates (no merging, rewriting, repair, or sixth
      answer)

Transport policy (frozen before any live benchmark call):

  default   OpenCode CLI with the opencode-go provider models.
  fallback  If the OpenCode CLI transport fails, calls fall back to the
            direct DeepSeek HTTP chat-completions API authenticated with a
            key resolved at runtime from DEEPSEEK_API_KEY (environment first,
            then the project ``.env`` file).  The key value is never logged,
            printed, or persisted anywhere by this runner; only the fact that
            a transport was used is recorded.

Subcommands: verify-frozen | init | prompt | run | status | import | review |
evaluate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2j_context_ablation import (
    canonical_sha256,
    file_sha256,
    text_sha256,
)
from pipeline.phase2k_full_transcript_ablation import (
    ArtifactError,
    DEFAULT_OUTPUT_DIR,
    REVIEW_FIELDS,
    SEMANTIC_FIELDS,
    import_intermediate_response,
)
from pipeline.phase2k_production_model_selection import (
    CALLS_PER_TARGET,
    CANDIDATE_COUNT,
    CONDITION_MODELS,
    DEFAULT_PROD_SEL_DIR,
    MODEL_FLASH,
    MODEL_PRO,
    NEW_CONDITION_CODES,
    PIPELINE_VERSION,
    RUN_SCHEMA_VERSION,
    SELECTION_REPORT_SCHEMA_VERSION,
    VERIFIER_ORDER_SALT,
    VERIFIER_RESPONSE_SCHEMA_VERSION,
    build_verifier_payload,
    candidate_call_id,
    check_selection_integrity,
    compute_condition_metrics,
    deterministic_candidate_order,
    evaluate_condition_gate,
    evaluate_verifier_usefulness,
    select_production_model,
    validate_verifier_response,
    verifier_call_id,
)
from scripts.run_phase2k_full_transcript_ablation import (
    WRAPPER_PROMPT,
    _capture_opencode_version,
    _now_iso,
    _write_json_atomic,
    _write_text_atomic,
    call_prompt_bytes,
    canonical_json,
    extract_response_json,
    load_frozen,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUTS_SCHEMA_VERSION = "phase2k-production-model-selection-outputs-v1"
CONDITION_RUN_DIRS = {"P": "pro_v1", "F": "flash_v1", "FV": "flash_verifier_v1"}
MANIFEST_FILENAME = "manifest.json"
RAW_SUBDIR = "raw"
MAX_WORKERS_LIMIT = 4
CALL_TIMEOUT_SECONDS = 3600

TRANSPORT_OPENCODE = "opencode_cli"
TRANSPORT_API = "deepseek_api"

VERIFIER_WRAPPER_PROMPT = (
    "You are a strict selection-only verifier worker for a controlled "
    "source-grounding experiment.  The supplied JSON payload contains the "
    "source context, the target passage metadata binding hashes, and five "
    "candidate extractions presented in a specific order.\n"
    "\n"
    "Select exactly ONE candidate that best preserves source-supported "
    "semantics of the target passage.  Return ONLY a single JSON object with "
    "exactly these keys: schema_version, case_id, selected_candidate_id, "
    "rationale (plus optionally criteria_scores).  Do not merge candidates, "
    "rewrite or repair any candidate, add new semantic claims, or produce "
    "your own extraction.  Prefer correct abstention and honestly unresolved "
    "references over plausible but unsupported League inference.\n"
    "\n"
    "schema_version must be exactly '" + VERIFIER_RESPONSE_SCHEMA_VERSION + "'.  "
    "selected_candidate_id must be one of the supplied candidate_id values."
)

DEEPSEEK_API_BASE_DEFAULT = "https://api.deepseek.com"


def _load_env_file() -> dict:
    """Parse KEY=VALUE lines from the project .env without logging values."""
    env: dict[str, str] = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def api_model_id(model: str) -> str:
    overrides = {
        MODEL_PRO: os.environ.get("DEEPSEEK_MODEL_PRO"),
        MODEL_FLASH: os.environ.get("DEEPSEEK_MODEL_FLASH"),
    }
    override = overrides.get(model)
    if override:
        return override
    return model.split("/")[-1]


class ModelTransport:
    """OpenCode CLI transport with a direct DeepSeek API fallback."""

    def __init__(self) -> None:
        self.env_file = _load_env_file()
        self.opencode_available = shutil.which("opencode") is not None

    def _api_key(self) -> str | None:
        return (
            os.environ.get("DEEPSEEK_API_KEY")
            or self.env_file.get("DEEPSEEK_API_KEY")
        )

    def _via_opencode(self, model: str, prompt_bytes: bytes) -> str:
        with tempfile.TemporaryDirectory(prefix="p2k-prodsel-") as workspace:
            proc = subprocess.run(
                ["opencode", "run", "--model", model],
                input=prompt_bytes,
                capture_output=True,
                cwd=workspace,
                timeout=CALL_TIMEOUT_SECONDS,
                check=False,
            )
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or b"").decode("utf-8", "replace")[-500:]
            raise RuntimeError(
                f"opencode exit {proc.returncode}: {stderr_tail}",
            )
        text = proc.stdout.decode("utf-8")
        if not text.strip():
            raise RuntimeError("opencode produced empty output")
        return text

    def _via_api(self, model: str, prompt_bytes: bytes) -> str:
        key = self._api_key()
        if not key:
            raise RuntimeError(
                "DeepSeek API fallback unavailable: DEEPSEEK_API_KEY not set",
            )
        base = os.environ.get(
            "DEEPSEEK_API_BASE", DEEPSEEK_API_BASE_DEFAULT,
        ).rstrip("/")
        body = json.dumps({
            "model": api_model_id(model),
            "messages": [{
                "role": "user",
                "content": prompt_bytes.decode("utf-8"),
            }],
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            base + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=CALL_TIMEOUT_SECONDS) as response:
            payload = json.load(response)
        return payload["choices"][0]["message"]["content"]

    def available_transports(self, mode: str) -> list[str]:
        ordered: list[str] = []
        if mode in ("auto", TRANSPORT_OPENCODE) and self.opencode_available:
            ordered.append(TRANSPORT_OPENCODE)
        if mode in ("auto", TRANSPORT_API) and self._api_key():
            ordered.append(TRANSPORT_API)
        if mode in (TRANSPORT_OPENCODE, TRANSPORT_API) and mode not in ordered:
            ordered.append(mode)
        return ordered

    def execute(self, *, model: str, prompt_bytes: bytes, mode: str = "auto"):
        attempts = self.available_transports(mode)
        if not attempts:
            raise RuntimeError(
                "no transport available (OpenCode CLI missing and no "
                "DEEPSEEK_API_KEY configured)",
            )
        errors: list[str] = []
        for transport in attempts:
            try:
                if transport == TRANSPORT_OPENCODE:
                    return self._via_opencode(model, prompt_bytes), transport
                return self._via_api(model, prompt_bytes), transport
            except Exception as exc:  # noqa: BLE001 - try next transport
                errors.append(f"{transport}: {exc}")
        raise RuntimeError("all transports failed :: " + " | ".join(errors))


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def condition_run_dir(args: argparse.Namespace, condition: str) -> Path:
    base = getattr(args, "run_dir", None)
    if base:
        return Path(base)
    run_base = getattr(args, "run_base", None)
    if run_base:
        return Path(run_base) / CONDITION_RUN_DIRS[condition]
    return DEFAULT_PROD_SEL_DIR / CONDITION_RUN_DIRS[condition]


def _load_run_manifest(run_dir: Path) -> dict | None:
    path = run_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ArtifactError("run manifest schema version is invalid")
    return manifest


def _save_manifest(run_dir: Path, manifest: dict) -> None:
    _write_json_atomic(run_dir / MANIFEST_FILENAME, manifest)


def raw_path_for(run_dir: Path, call_id: str) -> Path:
    safe = call_id.replace("/", "__")
    return run_dir / RAW_SUBDIR / f"{safe}.json"


# ---------------------------------------------------------------------------
# verify-frozen
# ---------------------------------------------------------------------------


def cmd_verify_frozen(args: argparse.Namespace) -> int:
    frozen = load_frozen(
        output_dir=Path(args.output_dir),
        db_path=Path(args.db),
        vocabulary_path=Path(args.vocabulary),
    )
    payloads_hash = frozen["payloads"]["content_sha256"]
    print(f"frozen payloads content_sha256: {payloads_hash}")
    ox_manifest_path = (
        DEFAULT_OUTPUT_DIR / "opencode_run_v1" / MANIFEST_FILENAME
    )
    if ox_manifest_path.exists():
        ox_manifest = json.loads(
            ox_manifest_path.read_text(encoding="utf-8"),
        )
        bound = ox_manifest["payloads_sha256"]
        if bound != payloads_hash:
            raise SystemExit(
                "FROZEN BENCHMARK MISMATCH: baseline run binds "
                f"{bound} but frozen artifacts hash to {payloads_hash}",
            )
        print(f"baseline OX run binds the same payloads ({bound})")
    else:
        print("baseline OX manifest not found; skipped cross-binding check")
    for pair in frozen["payloads"]["cases"]:
        print(
            f"  {pair['case_id']}: bronze sha "
            f"{pair['B']['target']['bronze_text_sha256'][:16]}...",
        )
    return 0


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def _all_calls(condition: str, payloads: dict) -> list[dict]:
    calls: list[dict] = []
    for pair in payloads["cases"]:
        case_id = pair["case_id"]
        prompt_sha = text_sha256(call_prompt_bytes(pair["B"]).decode("utf-8"))
        if condition in ("P", "F"):
            calls.append({
                "call_id": f"{case_id}/{condition}",
                "case_id": case_id,
                "condition": condition,
                "role": "generator",
                "candidate_id": None,
                "prompt_sha256": prompt_sha,
                "status": "pending",
                "transport": None,
                "retries": 0,
                "parse_failures": 0,
                "latency_seconds": None,
            })
        else:
            for index in range(1, CANDIDATE_COUNT + 1):
                calls.append({
                    "call_id": candidate_call_id(case_id, index),
                    "case_id": case_id,
                    "condition": condition,
                    "role": "generator",
                    "candidate_id": f"candidate_{index}",
                    "prompt_sha256": prompt_sha,
                    "status": "pending",
                    "transport": None,
                    "retries": 0,
                    "parse_failures": 0,
                    "latency_seconds": None,
                })
    return calls


def cmd_init(args: argparse.Namespace) -> int:
    condition = args.condition
    if condition not in NEW_CONDITION_CODES:
        raise SystemExit(f"unknown condition {condition!r}")
    frozen = load_frozen(
        output_dir=Path(args.output_dir),
        db_path=Path(args.db),
        vocabulary_path=Path(args.vocabulary),
    )
    payloads = frozen["payloads"]
    run_dir = condition_run_dir(args, condition)
    existing = _load_run_manifest(run_dir)
    if existing is not None:
        if existing["payloads_sha256"] != payloads["content_sha256"]:
            raise SystemExit(
                "existing run manifest binds different frozen payloads",
            )
        if existing["requested_model"] != CONDITION_MODELS[condition]:
            raise SystemExit(
                "existing run manifest binds a different requested model",
            )
        print(f"run manifest already exists at {run_dir}")
        return 0
    transport: ModelTransport | None = None
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "purpose": (
            f"Phase 2K production-model selection condition {condition}. "
            "Backend identity is requested/recorded but not cryptographically "
            "proven."
        ),
        "pipeline_version": PIPELINE_VERSION,
        "condition": condition,
        "requested_model": CONDITION_MODELS[condition],
        "verifier_model": (
            CONDITION_MODELS["FV"] if condition == "FV" else None
        ),
        "opencode_cli_version": _capture_opencode_version(),
        "argv_template_openai_style": [
            "opencode", "run", "--model", CONDITION_MODELS[condition],
        ],
        "prompt_transport": "stdin",
        "wrapper_prompt_sha256": text_sha256(WRAPPER_PROMPT),
        "calls_per_target": CALLS_PER_TARGET[condition],
        "payloads_sha256": payloads["content_sha256"],
        "instructions_sha256": payloads["instructions_sha256"],
        "final_outputs": None,
        "calls": _all_calls(condition, payloads),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_manifest(run_dir, manifest)
    pending = sum(1 for c in manifest["calls"] if c["status"] == "pending")
    print(
        f"initialized {len(manifest['calls'])}-call manifest "
        f"({pending} pending) for condition {condition} in {run_dir}"
    )
    return 0


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


def _find_call(manifest: dict, call_id: str) -> dict:
    for call in manifest["calls"]:
        if call["call_id"] == call_id:
            return call
    raise SystemExit(f"unknown call id {call_id!r}")


def _find_payload_by_case(payloads: dict, case_id: str) -> dict:
    for pair in payloads["cases"]:
        if pair["case_id"] == case_id:
            return pair["B"]
    raise SystemExit(f"unknown case id {case_id!r}")


def _build_verifier_prompt_bytes(
    frozen: dict,
    run_dir: Path,
    call: dict,
) -> bytes:
    payload_b = _find_payload_by_case(frozen["payloads"], call["case_id"])
    order = deterministic_candidate_order(call["case_id"])
    candidate_responses: dict[str, dict] = {}
    for candidate_id in order:
        candidate_call = next(
            c for c in frozen["manifest_calls_index"]
            if c["case_id"] == call["case_id"]
            and c.get("candidate_id") == candidate_id
        )
        if candidate_call.get("status") != "completed":
            raise SystemExit(
                f"candidate {candidate_id} for {call['case_id']} is not "
                "completed; verifier cannot run yet",
            )
        raw_file = run_dir / candidate_call["raw_path"]
        candidate_responses[candidate_id] = extract_response_json(
            raw_file.read_text(encoding="utf-8"),
        )
    verifier_payload = build_verifier_payload(
        payload_b=payload_b,
        candidate_responses=candidate_responses,
        candidate_order=order,
    )
    message = VERIFIER_WRAPPER_PROMPT + "\n\n" + canonical_json(verifier_payload)
    return message.encode("utf-8")


def cmd_prompt(args: argparse.Namespace) -> int:
    frozen_ctx = load_frozen(
        output_dir=Path(args.output_dir),
        db_path=Path(args.db),
        vocabulary_path=Path(args.vocabulary),
    )
    run_dir = Path(args.run_dir)
    manifest = _load_run_manifest(run_dir)
    if manifest is None:
        raise SystemExit("run manifest missing; run init first")
    call = _find_call(manifest, args.call_id)
    payload_b = _find_payload_by_case(frozen_ctx["payloads"], call["case_id"])
    if call["role"] == "verifier":
        manifest_calls_index = manifest["calls"]
        frozen_view = {
            "payloads": frozen_ctx["payloads"],
            "manifest_calls_index": manifest_calls_index,
        }
        prompt_bytes = _build_verifier_prompt_bytes(
            frozen_view, run_dir, call,
        )
    else:
        prompt_bytes = call_prompt_bytes(payload_b)
    sys.stdout.buffer.write(prompt_bytes)
    sys.stdout.buffer.flush()
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class RunContext:
    def __init__(
        self,
        *,
        frozen: dict,
        run_dir: Path,
        manifest: dict,
        force: bool,
        transport_mode: str,
    ) -> None:
        self.frozen = frozen
        self.run_dir = run_dir
        self.manifest = manifest
        self.force = force
        self.transport_mode = transport_mode
        self.lock = threading.Lock()
        self.transport = ModelTransport()

    def payload_b(self, case_id: str) -> dict:
        return _find_payload_by_case(self.frozen["payloads"], case_id)


def _candidates_complete(context: RunContext, case_id: str) -> bool:
    candidates = [
        c for c in context.manifest["calls"]
        if c["case_id"] == case_id
        and c.get("candidate_id") is not None
    ]
    return len(candidates) == CANDIDATE_COUNT and all(
        c.get("status") == "completed" for c in candidates
    )


def _ensure_verifier_calls(context: RunContext) -> list[dict]:
    """Append verifier calls for cases whose five candidates are complete."""
    added: list[dict] = []
    seen_cases = {
        c["case_id"]: c for c in context.manifest["calls"]
        if c.get("role") == "verifier"
    }
    case_ids = sorted({
        c["case_id"] for c in context.manifest["calls"]
        if c.get("candidate_id") is not None
    })
    for case_id in case_ids:
        if case_id in seen_cases:
            continue
        if not _candidates_complete(context, case_id):
            continue
        order = deterministic_candidate_order(case_id)
        verifier_payload_binding = canonical_sha256([
            VERIFIER_ORDER_SALT, case_id,
        ])
        call = {
            "call_id": verifier_call_id(case_id),
            "case_id": case_id,
            "condition": "FV",
            "role": "verifier",
            "candidate_id": None,
            "candidate_order": order,
            "candidate_order_sha256": verifier_payload_binding,
            "prompt_sha256": None,
            "status": "pending",
            "transport": None,
            "retries": 0,
            "parse_failures": 0,
            "latency_seconds": None,
        }
        context.manifest["calls"].append(call)
        added.append(call)
    if added:
        _save_manifest(context.run_dir, context.manifest)
    return added


def _execute_call(context: RunContext, call: dict) -> None:
    case_id = call["case_id"]
    payload_b = context.payload_b(case_id)
    if call["role"] == "verifier":
        view = {
            "payloads": context.frozen["payloads"],
            "manifest_calls_index": context.manifest["calls"],
        }
        prompt_bytes = _build_verifier_prompt_bytes(view, context.run_dir, call)
        expected_sha = text_sha256(prompt_bytes.decode("utf-8"))
        if call.get("prompt_sha256") not in (None, expected_sha):
            raise SystemExit(
                f"verifier prompt drift for {call['call_id']}; rebuild required",
            )
        model = CONDITION_MODELS["FV"]
    else:
        prompt_bytes = call_prompt_bytes(payload_b)
        expected_sha = text_sha256(prompt_bytes.decode("utf-8"))
        if expected_sha != call["prompt_sha256"]:
            raise SystemExit(
                f"prompt hash drift for {call['call_id']}; rebuild required",
            )
        model = CONDITION_MODELS[call["condition"]]
    retries_allowed = int(getattr(context, "retries", 0))
    parse_failures = 0
    last_error: Exception | None = None
    started_iso = _now_iso()
    for attempt in range(retries_allowed + 1):
        started = time.monotonic()
        try:
            stdout_text, used_transport = context.transport.execute(
                model=model,
                prompt_bytes=prompt_bytes,
                mode=context.transport_mode,
            )
            latency = round(time.monotonic() - started, 3)
            parsed = extract_response_json(stdout_text)
            if call["role"] == "verifier":
                validate_verifier_response(
                    parsed,
                    case_id=case_id,
                    candidate_order=call["candidate_order"],
                )
            else:
                import_intermediate_response(
                    parsed,
                    case_id=case_id,
                    condition="B",
                    payload=payload_b,
                )
            break
        except Exception as exc:  # noqa: BLE001 - retry loop reports failures
            parse_failures += 1
            last_error = exc
            time.sleep(2)
    else:
        raise RuntimeError(
            f"call {call['call_id']} failed after "
            f"{parse_failures} attempt(s): {last_error}",
        )
    raw_file = raw_path_for(context.run_dir, call["call_id"])
    _write_text_atomic(raw_file, stdout_text)
    with context.lock:
        fresh = _load_run_manifest(context.run_dir)
        target = next(
            c for c in fresh["calls"] if c["call_id"] == call["call_id"]
        )
        target.update({
            "status": "completed",
            "raw_path": str(raw_file.relative_to(context.run_dir)),
            "raw_response_sha256": file_sha256(raw_file),
            "prompt_sha256": expected_sha,
            "transport": used_transport,
            "retries": attempt,
            "parse_failures": parse_failures,
            "started_at": started_iso,
            "completed_at": _now_iso(),
            "latency_seconds": latency,
        })
        _save_manifest(context.run_dir, fresh)
        context.manifest = fresh


def _validated_existing_call(context: RunContext, call: dict) -> bool:
    if call.get("status") != "completed":
        return False
    raw_file = context.run_dir / call["raw_path"]
    if context.run_dir.resolve() not in raw_file.resolve().parent.parents:
        raise SystemExit(
            f"recorded raw path escapes the run dir: {call['raw_path']}",
        )
    if file_sha256(raw_file) != call.get("raw_response_sha256"):
        if not context.force:
            raise SystemExit(
                f"raw response for {call['call_id']} changed on disk; "
                "use --force to replace it",
            )
        return False
    try:
        stdout_text = raw_file.read_text(encoding="utf-8")
        parsed = extract_response_json(stdout_text)
        if call["role"] == "verifier":
            validate_verifier_response(
                parsed,
                case_id=call["case_id"],
                candidate_order=call["candidate_order"],
            )
        else:
            import_intermediate_response(
                parsed,
                case_id=call["case_id"],
                condition="B",
                payload=context.payload_b(call["case_id"]),
            )
    except (ValueError, ArtifactError):
        if not context.force:
            raise SystemExit(
                f"recorded raw response for {call['call_id']} no longer "
                "validates; use --force to replace exactly this call",
            )
        return False
    return True


def cmd_run(args: argparse.Namespace) -> int:
    frozen = load_frozen(
        output_dir=Path(args.output_dir),
        db_path=Path(args.db),
        vocabulary_path=Path(args.vocabulary),
    )
    run_dir = condition_run_dir(args, args.condition)
    manifest = _load_run_manifest(run_dir)
    if manifest is None:
        raise SystemExit("run manifest missing; run init first")
    if manifest["requested_model"] != CONDITION_MODELS[args.condition]:
        raise SystemExit("run manifest binds a different requested model")
    workers = max(1, min(int(args.max_workers), MAX_WORKERS_LIMIT))
    context = RunContext(
        frozen=frozen,
        run_dir=run_dir,
        manifest=manifest,
        force=args.force,
        transport_mode=args.transport,
    )
    context.retries = int(args.retries)

    progress = True
    while progress:
        progress = False
        _ensure_verifier_calls(context)
        pending = [
            c for c in context.manifest["calls"]
            if c.get("status") == "pending"
            and not (
                c["role"] == "verifier"
                and not _candidates_complete(context, c["case_id"])
            )
        ]
        runnable = []
        for call in pending:
            if call.get("status") == "completed" and _validated_existing_call(
                context, call,
            ):
                continue
            if call["role"] == "generator" and call.get("status") == "completed":
                continue
            runnable.append(dict(call))
        if not runnable:
            break
        failures: list[str] = []
        if workers == 1:
            for call in runnable:
                try:
                    _execute_call(context, call)
                    print(f"completed {call['call_id']}")
                    progress = True
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{call['call_id']}: {exc}")
                    break
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    (call, pool.submit(_execute_call, context, call))
                    for call in runnable
                ]
                for call, future in futures:
                    try:
                        future.result()
                        print(f"completed {call['call_id']}")
                        progress = True
                    except Exception as exc:  # noqa: BLE001
                        failures.append(f"{call['call_id']}: {exc}")
                        break
                pool.shutdown(wait=True, cancel_futures=True)
        if failures:
            for failure in failures:
                print(f"FAILURE {failure}", file=sys.stderr)
            return 1
    refreshed = _load_run_manifest(run_dir)
    remaining = sum(
        1 for c in refreshed["calls"] if c.get("status") == "pending"
    )
    blocked = sum(
        1 for c in refreshed["calls"]
        if c.get("status") == "pending" and c["role"] == "verifier"
        and not _candidates_complete(context, c["case_id"])
    ) if refreshed["condition"] == "FV" else 0
    print(f"remaining pending calls: {remaining} (blocked verifiers: {blocked})")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = condition_run_dir(args, args.condition)
    manifest = _load_run_manifest(run_dir)
    if manifest is None:
        raise SystemExit("run manifest missing")
    completed = 0
    for call in manifest["calls"]:
        state = call.get("status", "pending")
        if state == "completed":
            completed += 1
        print(f"{call['call_id']}: {state}")
    total = len(manifest["calls"])
    print(f"{completed}/{total} completed "
          f"(model {manifest['requested_model']})")
    return 0


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------


def cmd_import(args: argparse.Namespace) -> int:
    frozen = load_frozen(
        output_dir=Path(args.output_dir),
        db_path=Path(args.db),
        vocabulary_path=Path(args.vocabulary),
    )
    run_dir = condition_run_dir(args, args.condition)
    manifest = _load_run_manifest(run_dir)
    if manifest is None:
        raise SystemExit("run manifest missing")
    condition = manifest["condition"]
    cases_out: list[dict] = []
    for pair in frozen["payloads"]["cases"]:
        case_id = pair["case_id"]
        payload_b = pair["B"]
        if condition in ("P", "F"):
            call = next(
                c for c in manifest["calls"] if c["case_id"] == case_id
            )
            if call.get("status") != "completed":
                raise SystemExit(f"incomplete call {call['call_id']}")
            raw_file = run_dir / call["raw_path"]
            if file_sha256(raw_file) != call["raw_response_sha256"]:
                raise SystemExit(f"raw hash mismatch for {call['call_id']}")
            response = extract_response_json(
                raw_file.read_text(encoding="utf-8"),
            )
            output = import_intermediate_response(
                response, case_id=case_id, condition="B", payload=payload_b,
            )
            cases_out.append({
                "case_id": case_id,
                "final_output": output,
                "raw_path": call["raw_path"],
            })
        else:
            candidate_calls = [
                c for c in manifest["calls"]
                if c["case_id"] == case_id
                and c.get("candidate_id") is not None
            ]
            verifier_call = next(
                c for c in manifest["calls"]
                if c["case_id"] == case_id and c.get("role") == "verifier"
            )
            for call in candidate_calls + [verifier_call]:
                if call.get("status") != "completed":
                    raise SystemExit(f"incomplete call {call['call_id']}")
            candidate_outputs: dict[str, dict] = {}
            candidate_raw_paths: dict[str, str] = {}
            for call in candidate_calls:
                raw_file = run_dir / call["raw_path"]
                if file_sha256(raw_file) != call["raw_response_sha256"]:
                    raise SystemExit(f"raw hash mismatch for {call['call_id']}")
                response = extract_response_json(
                    raw_file.read_text(encoding="utf-8"),
                )
                output = import_intermediate_response(
                    response, case_id=case_id, condition="B",
                    payload=payload_b,
                )
                candidate_outputs[call["candidate_id"]] = output
                candidate_raw_paths[call["candidate_id"]] = call["raw_path"]
            verifier_raw = run_dir / verifier_call["raw_path"]
            if file_sha256(verifier_raw) != verifier_call["raw_response_sha256"]:
                raise SystemExit(
                    f"verifier raw hash mismatch for {case_id}",
                )
            verifier_response = extract_response_json(
                verifier_raw.read_text(encoding="utf-8"),
            )
            selected = validate_verifier_response(
                verifier_response,
                case_id=case_id,
                candidate_order=verifier_call["candidate_order"],
            )
            final_output = candidate_outputs[selected]
            check_selection_integrity(
                final_output=final_output,
                candidate_outputs=candidate_outputs,
                selected_candidate_id=selected,
            )
            cases_out.append({
                "case_id": case_id,
                "final_output": final_output,
                "selected_candidate_id": selected,
                "candidate_order": verifier_call["candidate_order"],
                "candidate_outputs": candidate_outputs,
                "candidate_raw_paths": candidate_raw_paths,
                "verifier_rationale": verifier_response["rationale"],
                "verifier_criteria_scores": verifier_response.get(
                    "criteria_scores",
                ),
                "verifier_raw_path": verifier_call["raw_path"],
            })
    artifact = {
        "schema_version": OUTPUTS_SCHEMA_VERSION,
        "purpose": (
            f"Validated Phase 2K production-model selection outputs for "
            f"condition {condition}."
        ),
        "condition": condition,
        "requested_model": manifest["requested_model"],
        "payloads_sha256": frozen["payloads"]["content_sha256"],
        "instructions_sha256": frozen["payloads"]["instructions_sha256"],
        "cases": cases_out,
    }
    artifact["content_sha256"] = canonical_sha256(artifact)
    out_name = f"phase2k-prodsel-{condition.lower()}-outputs-v1.json"
    output_path = Path(args.output_dir) / out_name
    _write_json_atomic(output_path, artifact)
    manifest["final_outputs"] = {
        "path": str(output_path.relative_to(ROOT)),
        "file_sha256": file_sha256(output_path),
        "imported_at": _now_iso(),
    }
    _save_manifest(run_dir, manifest)
    print(f"wrote {output_path}")
    return 0


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


def _render_output_fields(output: dict) -> list[str]:
    lines: list[str] = []
    fields = output["fields"]
    source_note = "(citations verified byte-exact at import)"
    for field in SEMANTIC_FIELDS:
        items = fields[field]
        lines.append(f"#### {field} ({len(items)} items)")
        if not items:
            lines.append("")
            lines.append("(none)")
            lines.append("")
            continue
        for index, item in enumerate(items, 1):
            extra = ""
            if "relation_type" in item:
                extra += f"; relation: {item['relation_type']}"
            lines.append(
                f"- [{index}] {item['extraction_text']} "
                f"(resolution: {item['resolution_status']}{extra})",
            )
            for reference in item["source_references"]:
                excerpt = reference["quote"]
                if len(excerpt) > 110:
                    excerpt = excerpt[:107] + "..."
                span = reference["source_range"]
                lines.append(
                    f"  - cite[{span['char_start']}:{span['char_end']}] "
                    f"{source_note}: {excerpt!r}",
                )
        lines.append("")
    return lines


def cmd_review(args: argparse.Namespace) -> int:
    sections: list[str] = [
        "# Phase 2K Production-Model Selection — Review Packet",
        "",
        "Same frozen 10-target benchmark and scoring contract as the "
        "completed Phase 2K ablation. Baseline reference: OX = 0x Alpha "
        "full-context (109/110). Reviewer note: scoring remains AI-based.",
        "",
    ]
    for condition in NEW_CONDITION_CODES:
        path = Path(args.output_dir) / (
            f"phase2k-prodsel-{condition.lower()}-outputs-v1.json"
        )
        if not path.exists():
            continue
        artifact = json.loads(path.read_text(encoding="utf-8"))
        sections += [
            "---",
            "",
            f"## Condition {condition} "
            f"(model {artifact['requested_model']})",
            "",
        ]
        for entry in artifact["cases"]:
            sections += [
                f"### TARGET {entry['case_id']}",
                "",
            ]
            if condition == "FV":
                sections += [
                    f"Selected candidate: {entry['selected_candidate_id']} "
                    f"(presentation order: "
                    f"{', '.join(entry['candidate_order'])})",
                    "",
                    f"Verifier rationale: {entry['verifier_rationale']}",
                    "",
                ]
            sections += _render_output_fields(entry["final_output"])
    markdown_path = Path(args.output_dir) / (
        "phase2k-prodsel-review-v1.md"
    )
    _write_text_atomic(markdown_path, "\n".join(sections))
    print(f"wrote {markdown_path}")
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def _load_baseline_metrics() -> dict:
    path = (
        DEFAULT_OUTPUT_DIR
        / "phase2k-context-ablation-evaluation-summary-v1.json"
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    per_field = {
        field: {"successes": stats["B_strict_successes"], "total": 10}
        for field, stats in summary["per_field"].items()
    }
    per_target = {
        entry["case_id"]: {
            "successes": entry["B_strict_successes"],
            "total": 11,
        }
        for entry in summary["per_target"]
    }
    return {
        "condition": "OX",
        "total_strict_successes": sum(
            t["successes"] for t in per_target.values()
        ),
        "total_judgments": 110,
        "per_field": per_field,
        "per_target": per_target,
        "unsupported_field_count": 0,
        "major_unsupported_count": 0,
        "grounding_failure_count": 0,
        "unresolved_field_successes": None,
    }


def _collect_cost(args: argparse.Namespace) -> dict:
    cost: dict[str, dict] = {}
    for condition in NEW_CONDITION_CODES:
        run_dir = condition_run_dir(args, condition)
        manifest = _load_run_manifest(run_dir)
        if manifest is None:
            continue
        calls = manifest["calls"]
        completed = [c for c in calls if c.get("status") == "completed"]
        latencies = [
            c["latency_seconds"] for c in completed
            if isinstance(c.get("latency_seconds"), (int, float))
        ]
        transports = sorted({
            c.get("transport") for c in completed if c.get("transport")
        })
        cost[condition] = {
            "model": manifest["requested_model"],
            "calls_per_target": manifest["calls_per_target"],
            "total_calls": len(calls),
            "completed_calls": len(completed),
            "total_wall_clock_seconds": round(sum(latencies), 1),
            "avg_latency_seconds_per_call": (
                round(sum(latencies) / len(latencies), 1)
                if latencies else None
            ),
            "wall_seconds_per_target": (
                round(sum(latencies) / 10, 1) if latencies else None
            ),
            "total_retries": sum(c.get("retries", 0) for c in completed),
            "total_parse_failures": sum(
                c.get("parse_failures", 0) for c in completed
            ),
            "transports_used": transports,
            "tokens": "unavailable via OpenCode CLI transport",
            "provider_cost_usd": "unavailable",
        }
    return cost


def compute_oracle(
    *,
    candidate_reviews: dict,
    flash_outputs: dict,
) -> dict | None:
    if not candidate_reviews:
        return None
    oracle_total = 0
    selected_total = 0
    regret_total = 0
    per_case = {}
    for entry in flash_outputs["cases"]:
        case_id = entry["case_id"]
        selected_id = entry["selected_candidate_id"]
        candidate_scores = {}
        for candidate_id in entry["candidate_outputs"]:
            count = 0
            for field in REVIEW_FIELDS:
                review_entry = candidate_reviews.get(
                    f"{case_id}:{candidate_id}:{field}",
                )
                if review_entry is None:
                    return None
                if (
                    review_entry["correctness"] in ("CORRECT", "ABSENT_CORRECTLY")
                    and review_entry["unsupported_inference"] == "NONE"
                    and review_entry["source_grounding"] in (
                        "GROUNDED", "NOT_APPLICABLE",
                    )
                ):
                    count += 1
            candidate_scores[candidate_id] = count
        best = max(candidate_scores.values())
        selected_score = candidate_scores[selected_id]
        oracle_total += best
        selected_total += selected_score
        regret_total += best - selected_score
        per_case[case_id] = {
            "candidate_strict_counts": candidate_scores,
            "oracle_best": best,
            "selected": selected_score,
            "regret": best - selected_score,
        }
    return {
        "oracle_best_of_5_total": oracle_total,
        "verifier_selected_total": selected_total,
        "verifier_regret_total": regret_total,
        "per_case": per_case,
    }


def cmd_evaluate(args: argparse.Namespace) -> int:
    frozen = load_frozen(
        output_dir=Path(args.output_dir),
        db_path=Path(args.db),
        vocabulary_path=Path(args.vocabulary),
    )
    reviews_doc = json.loads(Path(args.reviews).read_text(encoding="utf-8"))
    reviews = reviews_doc["reviews"]
    case_ids = [pair["case_id"] for pair in frozen["payloads"]["cases"]]
    metrics = {
        condition: compute_condition_metrics(
            {
                key: entry
                for key, entry in reviews.items()
                if f":{condition}:" in key
            },
            condition=condition,
            case_ids=case_ids,
        )
        for condition in NEW_CONDITION_CODES
    }
    gates = {
        condition: evaluate_condition_gate(metrics[condition])
        for condition in NEW_CONDITION_CODES
    }
    usefulness = evaluate_verifier_usefulness(
        flash_metrics=metrics["F"],
        flash_verifier_metrics=metrics["FV"],
    )
    flash_outputs = json.loads(
        (Path(args.output_dir) / "phase2k-prodsel-fv-outputs-v1.json").read_text(
            encoding="utf-8",
        ),
    )
    oracle = compute_oracle(
        candidate_reviews=reviews_doc.get("candidate_reviews") or {},
        flash_outputs=flash_outputs,
    )
    cost = _collect_cost(args)
    cost_per_target = {
        condition: cost[condition]["wall_seconds_per_target"]
        for condition in NEW_CONDITION_CODES
        if condition in cost
    }
    selection = select_production_model(
        gates=gates,
        cost_per_target_seconds=cost_per_target,
    )
    baseline = _load_baseline_metrics()
    report = {
        "schema_version": SELECTION_REPORT_SCHEMA_VERSION,
        "purpose": (
            "Phase 2K production-model selection report over the frozen "
            "full-context benchmark."
        ),
        "pipeline_version": PIPELINE_VERSION,
        "baseline_ox": baseline,
        "metrics_by_condition": metrics,
        "gates": gates,
        "verifier_usefulness": usefulness,
        "oracle_analysis": oracle,
        "cost_accounting": cost,
        "production_selection": selection,
        "reviewer_kind": reviews_doc.get("reviewer_kind"),
        "reviewer_identity": reviews_doc.get("reviewer_identity"),
    }
    report["content_sha256"] = canonical_sha256(report)
    report_path = (
        Path(args.output_dir)
        / "phase2k-production-model-selection-report-v1.json"
    )
    _write_json_atomic(report_path, report)

    lines = [
        "# Phase 2K Production-Model Selection Report",
        "",
        "| Condition | Strict | Unsupported | Grounding | Gate | Calls/target |",
        "|---|---:|---:|---:|---|---:|",
        f"| OX Alpha baseline | {baseline['total_strict_successes']}/110 | 0 | 0 "
        f"| baseline | 1 |",
    ]
    for condition in NEW_CONDITION_CODES:
        m = metrics[condition]
        g = gates[condition]["outcome"]
        lines.append(
            f"| {condition} ({CONDITION_MODELS[condition].split('/')[-1]}) "
            f"| {m['total_strict_successes']}/110 "
            f"| {m['unsupported_field_count']} "
            f"| {m['grounding_failure_count']} | {g} "
            f"| {CALLS_PER_TARGET[condition]} |",
        )
    lines += [
        "",
        f"Verifier gate: **{usefulness['decision']}** "
        f"(delta {usefulness['delta_strict_successes']:+d}, "
        f"+{usefulness['improved_targets']}/-{usefulness['worsened_targets']} targets)",
        "",
        f"Production recommendation: **{selection['recommendation']}**",
        "",
        "## Per-field strict successes (of 10)",
        "",
        "| Field | OX | P | F | FV |",
        "|---|---:|---:|---:|---:|",
    ]
    for field in REVIEW_FIELDS:
        row = [field, str(baseline["per_field"][field]["successes"])]
        for condition in NEW_CONDITION_CODES:
            row.append(str(metrics[condition]["per_field"][field]["successes"]))
        lines.append("| " + " | ".join(row) + " |")
    if oracle is not None:
        lines += [
            "",
            "## Oracle Best-of-5 analysis",
            "",
            f"- Oracle best-of-5 total: {oracle['oracle_best_of_5_total']}/110",
            f"- Verifier-selected total: {oracle['verifier_selected_total']}/110",
            f"- Verifier regret: {oracle['verifier_regret_total']}",
        ]
    lines += ["", "## Cost accounting", ""]
    for condition, entry in cost.items():
        lines.append(
            f"- {condition}: {entry['completed_calls']} calls, "
            f"{entry['wall_seconds_per_target']}s/target, "
            f"retries {entry['total_retries']}, parse failures "
            f"{entry['total_parse_failures']}, transports "
            f"{','.join(entry['transports_used'])}",
        )
    markdown_path = (
        Path(args.output_dir)
        / "phase2k-production-model-selection-report-v1.md"
    )
    _write_text_atomic(markdown_path, "\n".join(lines) + "\n")
    print(f"wrote {report_path}")
    print(f"wrote {markdown_path}")
    print(f"production recommendation: {selection['recommendation']}")
    print(f"verifier gate: {usefulness['decision']}")
    return 0


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        default="/home/bphan944/PersonalProjects/videoSorter-homework-archive/videos.db",
    )
    parser.add_argument(
        "--vocabulary",
        default=str(Path(DEFAULT_OUTPUT_DIR).parent
                    / "phase2k_support/league_lexical_vocabulary_v2.json"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_frozen = subparsers.add_parser(
        "verify-frozen", help="verify the frozen Phase 2K benchmark",
    )
    p_frozen.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    add_common_arguments(p_frozen)
    p_frozen.set_defaults(func=cmd_verify_frozen)

    p_init = subparsers.add_parser(
        "init", help="create a per-condition run manifest",
    )
    p_init.add_argument("--condition", required=True,
                        choices=list(NEW_CONDITION_CODES))
    p_init.add_argument("--run-dir", default=None)
    p_init.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    add_common_arguments(p_init)
    p_init.set_defaults(func=cmd_init)

    p_prompt = subparsers.add_parser(
        "prompt", help="print exact user-message bytes for one call",
    )
    p_prompt.add_argument("--run-dir", required=True)
    p_prompt.add_argument("--call-id", required=True)
    p_prompt.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    add_common_arguments(p_prompt)
    p_prompt.set_defaults(func=cmd_prompt)

    p_run = subparsers.add_parser(
        "run", help="execute pending calls through the configured transport",
    )
    p_run.add_argument("--condition", required=True,
                       choices=list(NEW_CONDITION_CODES))
    p_run.add_argument("--run-dir", default=None)
    p_run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p_run.add_argument("--max-workers", default=1, type=int)
    p_run.add_argument("--retries", default=1, type=int)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument(
        "--transport",
        default="auto",
        choices=["auto", TRANSPORT_OPENCODE, TRANSPORT_API],
    )
    add_common_arguments(p_run)
    p_run.set_defaults(func=cmd_run)

    p_status = subparsers.add_parser("status", help="report call status")
    p_status.add_argument("--condition", required=True,
                          choices=list(NEW_CONDITION_CODES))
    p_status.add_argument("--run-dir", default=None)
    p_status.set_defaults(func=cmd_status)

    p_import = subparsers.add_parser(
        "import", help="assemble validated per-condition outputs",
    )
    p_import.add_argument("--condition", required=True,
                          choices=list(NEW_CONDITION_CODES))
    p_import.add_argument("--run-dir", default=None)
    p_import.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    add_common_arguments(p_import)
    p_import.set_defaults(func=cmd_import)

    p_review = subparsers.add_parser(
        "review", help="generate the combined review markdown packet",
    )
    p_review.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p_review.set_defaults(func=cmd_review)

    p_eval = subparsers.add_parser(
        "evaluate", help="compute gates/report from completed reviews",
    )
    p_eval.add_argument("--reviews", required=True)
    p_eval.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p_eval.add_argument("--run-base", default=str(DEFAULT_PROD_SEL_DIR))
    add_common_arguments(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
