"""Focused tests for the Phase 2J context-ablation Sol runner.

All codex invocations are mocked; no real model, network, or codex calls
are made.  The tests cover the exact 20 independent calls, prompt wrapper
identity/hashes, isolation/temp behavior, resumability and fail-closed
binding, failure evidence, assembly validation, and model/config manifest
evidence.
"""

from __future__ import annotations

import json
import io
import re
import subprocess
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

import scripts.run_phase2j_context_ablation_sol as runner
from pipeline.phase2j_context_ablation import (
    OUTPUT_FILENAMES,
    build_sol_intermediate_schema,
    canonical_sha256,
    load_json_strict,
    text_sha256,
    validate_extraction_outputs_bundle,
)
from tests.test_phase2j_context_ablation import _build_full_fixture


SEMANTIC_FIELDS = [
    "actors", "ability_resource_references", "event", "condition",
    "advice_action", "consequence", "uncertainty", "supporting_source_ranges",
]

WRAPPER_PREFIX = runner.SOL_WRAPPER_PROMPT.encode("utf-8") + b"\n\n"


class FakeCodex:
    """Mock ``codex`` CLI: answers --version and simulates codex exec."""

    def __init__(
        self,
        *,
        version: str = "0.147.0",
        fail_calls: set[str] | None = None,
        malformed_calls: set[str] | None = None,
        fail_first_attempt: set[str] | None = None,
    ):
        self.version = version
        self.fail_calls = set(fail_calls or [])
        self.malformed_calls = set(malformed_calls or [])
        self.fail_first_attempt = set(fail_first_attempt or [])
        self.exec_argv: list[list[str]] = []
        self.stdin_by_key: dict[str, bytes] = {}
        self.schema_files: list[bytes] = []
        self.attempts_log: list[tuple[str, list[str], bytes]] = []
        self.attempts: dict[str, int] = {}
        self.version_calls = 0
        self.lock = threading.Lock()

    def __call__(
        self,
        argv: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        if "--version" in argv:
            self.version_calls += 1
            return subprocess.CompletedProcess(
                argv, 0,
                stdout=f"codex-cli {self.version}\n".encode("utf-8"),
                stderr=b"",
            )
        stdin: bytes = kwargs["input"]
        payload = json.loads(stdin[len(WRAPPER_PREFIX):].decode("utf-8"))
        key = f"{payload['case_id']}:{payload['condition']}"
        with self.lock:
            argv_copy = list(argv)
            self.exec_argv.append(argv_copy)
            self.schema_files.append(
                Path(argv[argv.index("--output-schema") + 1]).read_bytes(),
            )
            self.stdin_by_key[key] = stdin
            attempt = self.attempts.get(key, 0) + 1
            self.attempts[key] = attempt
            self.attempts_log.append((key, argv_copy, stdin))
        out_path = None
        for index, arg in enumerate(argv):
            if arg == "-o":
                out_path = Path(argv[index + 1])
        if key in self.fail_first_attempt and attempt == 1:
            return subprocess.CompletedProcess(
                argv, 1, stdout=b"", stderr=b"first attempt failure",
            )
        if key in self.fail_calls:
            return subprocess.CompletedProcess(
                argv, 1, stdout=b"", stderr=b"boom",
            )
        source = (
            payload["target"]["bronze_text"]
            if payload["condition"] == "A" else payload["transcript"]
        )
        quote = re.search(r"\S+", source).group()
        fields = {field: [] for field in SEMANTIC_FIELDS}
        fields["actors"] = [{
            "extraction_text": "coach",
            "resolution_status": "literal_explicit",
            "source_references": [
                {"quote": quote, "occurrence_index": 0},
            ],
        }]
        fields["supporting_source_ranges"] = [{
            "extraction_text": "range",
            "resolution_status": "literal_explicit",
            "source_references": [
                {"quote": quote, "occurrence_index": 0},
            ],
        }]
        response = {
            "schema_version": build_sol_intermediate_schema()["schema_version"],
            "case_id": payload["case_id"],
            "condition": payload["condition"],
            "payload_sha256": payload["content_sha256"],
            "instructions_sha256": payload["instructions_sha256"],
            "fields": fields,
        }
        body = json.dumps(response)
        if key in self.malformed_calls:
            body = "{not valid json"
        if out_path is None:
            raise AssertionError("codex exec argv has no -o path")
        out_path.write_text(body, encoding="utf-8")
        return subprocess.CompletedProcess(
            argv, 0, stdout=b"ok", stderr=b"",
        )


def _base_args(root: Path, fixture: dict[str, Any], command: str) -> list[str]:
    return [
        command,
        "--manifest", str(fixture["manifest_path"]),
        "--reviewed-packet", str(fixture["packet_path"]),
        "--db", str(fixture["db_path"]),
        "--output-dir", str(fixture["output_dir"]),
        "--run-dir", str(root / "sol_run"),
    ]


class SolRunnerTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.fixture = _build_full_fixture(self.root)
        self.run_dir = self.root / "sol_run"
        self.payloads = load_json_strict(
            self.fixture["output_dir"] / OUTPUT_FILENAMES["payloads"],
            label="payloads",
        )

    def tearDown(self):
        self._temporary.cleanup()

    def _args(self, command: str, **extra: Any) -> list[str]:
        args = _base_args(self.root, self.fixture, command)
        for key, value in extra.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            else:
                args.append(flag)
                args.append(str(value))
        return args

    def _run(self, fake: FakeCodex, **extra: Any) -> int:
        with mock.patch.object(runner.subprocess, "run", side_effect=fake):
            return runner.main(self._args("run", **extra))

    def _import(self, fake: FakeCodex, **extra: Any) -> int:
        with mock.patch.object(runner.subprocess, "run", side_effect=fake):
            return runner.main(self._args("import", **extra))

    def _expected_payloads(self) -> dict[str, dict[str, Any]]:
        return {
            f"{payload_case['case_id']}:{condition}": payload_case[condition]
            for payload_case in self.payloads["cases"]
            for condition in ("A", "B")
        }

    def _manifest(self) -> dict[str, Any]:
        return json.loads((self.run_dir / "manifest.json").read_text(
            encoding="utf-8",
        ))

    def test_run_executes_exactly_20_isolated_calls_with_canonical_config(self):
        fake = FakeCodex()
        code = self._run(fake, max_workers=4)
        self.assertEqual(code, 0)
        self.assertEqual(len(fake.exec_argv), 20)
        schema_canonical = runner._schema_bytes()
        for schema_bytes in fake.schema_files:
            self.assertEqual(schema_bytes, schema_canonical)
        work_dirs: set[str] = set()
        for argv in fake.exec_argv:
            self.assertEqual(argv[0], "codex")
            self.assertEqual(argv[1], "exec")
            for flag in (
                "--ephemeral", "--ignore-user-config", "--ignore-rules",
                "--skip-git-repo-check",
            ):
                self.assertIn(flag, argv)
            self.assertEqual(argv[argv.index("-m") + 1], "gpt-5.6-sol")
            self.assertEqual(
                argv[argv.index("-c") + 1],
                'model_reasoning_effort="high"',
            )
            self.assertEqual(argv[argv.index("-s") + 1], "read-only")
            self.assertEqual(argv[-1], "-")
            work_dirs.add(argv[argv.index("-C") + 1])
            self.assertTrue(
                Path(argv[argv.index("--output-schema") + 1]).is_absolute(),
            )
        self.assertEqual(len(work_dirs), 20)
        for work_dir in work_dirs:
            self.assertFalse(Path(work_dir).exists())  # cleaned after success
        raw_files = list((self.run_dir / "raw").glob("*.raw.json"))
        self.assertEqual(len(raw_files), 20)
        expected = self._expected_payloads()
        self.assertEqual(set(fake.stdin_by_key), set(expected))
        for key, stdin in fake.stdin_by_key.items():
            self.assertTrue(stdin.startswith(WRAPPER_PREFIX))
            parsed = json.loads(stdin[len(WRAPPER_PREFIX):].decode("utf-8"))
            self.assertEqual(parsed, expected[key])

    def test_no_sibling_or_outer_provenance_leakage(self):
        fake = FakeCodex()
        self.assertEqual(self._run(fake, max_workers=4), 0)
        expected = self._expected_payloads()
        sibling_case_ids = [
            payload_case["case_id"]
            for payload_case in self.payloads["cases"]
        ]
        for key, stdin in fake.stdin_by_key.items():
            text = stdin.decode("utf-8")
            parsed = json.loads(stdin[len(WRAPPER_PREFIX):].decode("utf-8"))
            self.assertEqual(parsed, expected[key])
            for forbidden in (
                '"provenance_by_case"', '"video_url"', '"window_id"',
                '"selection_manifest_sha256"', '"reviewed-endpoint"',
            ):
                self.assertNotIn(forbidden, text)
            case_id = key.rsplit(":", 1)[0]
            for sibling in sibling_case_ids:
                if sibling != case_id:
                    self.assertNotIn(sibling, text)
            if key.endswith(":A"):
                self.assertNotIn('"transcript"', text)
                self.assertNotIn('"metadata"', text)
                self.assertNotIn('"vocabulary"', text)

    def test_manifest_records_model_config_and_hashes(self):
        fake = FakeCodex()
        self.assertEqual(self._run(fake, max_workers=2), 0)
        manifest = self._manifest()
        self.assertEqual(manifest["requested_model"], "gpt-5.6-sol")
        self.assertEqual(manifest["model_reasoning_effort"], "high")
        self.assertEqual(manifest["codex_cli_version"], "0.147.0")
        self.assertEqual(manifest["argv_template"], runner.ARGV_TEMPLATE)
        self.assertEqual(
            manifest["wrapper_sha256"], text_sha256(runner.SOL_WRAPPER_PROMPT),
        )
        self.assertEqual(
            manifest["intermediate_schema_sha256"],
            canonical_sha256(build_sol_intermediate_schema()),
        )
        self.assertEqual(
            manifest["payloads_sha256"], self.payloads["content_sha256"],
        )
        self.assertEqual(
            manifest["instructions_sha256"], self.payloads["instructions_sha256"],
        )
        expected = self._expected_payloads()
        for call in manifest["calls"]:
            key = f"{call['case_id']}:{call['condition']}"
            payload = expected[key]
            self.assertEqual(call["status"], "completed")
            self.assertEqual(call["payload_sha256"], payload["content_sha256"])
            self.assertEqual(
                call["prompt_sha256"],
                text_sha256(
                    runner.SOL_WRAPPER_PROMPT + "\n\n"
                    + runner._canonical_json(payload),
                ),
            )
            raw_path = self.run_dir / call["raw_response_path"]
            self.assertTrue(raw_path.is_file())
            self.assertEqual(call["raw_response_sha256"], runner.file_sha256(raw_path))
            self.assertGreater(call["attempts"], 0)

    def test_resume_reuses_valid_raw_responses(self):
        first = FakeCodex()
        self.assertEqual(self._run(first, max_workers=4), 0)
        self.assertEqual(len(first.exec_argv), 20)
        second = FakeCodex()
        self.assertEqual(self._run(second, max_workers=4), 0)
        self.assertEqual(len(second.exec_argv), 0)  # all reused, no exec calls
        manifest = self._manifest()
        self.assertEqual(
            [call["status"] for call in manifest["calls"]],
            ["completed"] * 20,
        )

    def test_tampered_raw_fails_closed_and_force_reruns_only_that_call(self):
        fake = FakeCodex()
        self.assertEqual(self._run(fake, max_workers=4), 0)
        raw_path = self.run_dir / "raw" / "p2ja-case-0001-A.raw.json"
        original = raw_path.read_text(encoding="utf-8")
        raw_path.write_text(
            original.replace('"occurrence_index": 0', '"occurrence_index": 99'),
            encoding="utf-8",
        )
        self.assertEqual(self._run(fake, max_workers=4), 1)  # fail closed
        before = len(fake.exec_argv)
        self.assertEqual(
            self._run(fake, max_workers=4, force=True), 0,
        )
        self.assertEqual(len(fake.exec_argv), before + 1)
        self.assertEqual(
            fake.attempts["p2ja:case:0001:A"],
            2,  # original run + forced rerun
        )
        self.assertEqual(
            self._import(fake, force=True), 0,
        )

    def test_malformed_raw_fails_closed(self):
        fake = FakeCodex()
        self.assertEqual(self._run(fake, max_workers=4), 0)
        raw_path = self.run_dir / "raw" / "p2ja-case-0005-B.raw.json"
        raw_path.write_text("{broken", encoding="utf-8")
        self.assertEqual(self._run(fake, max_workers=4), 1)
        before = len(fake.exec_argv)
        self.assertEqual(self._run(fake, max_workers=4, force=True), 0)
        self.assertEqual(len(fake.exec_argv), before + 1)
        self.assertEqual(self._import(fake, force=True), 0)

    def test_subprocess_failure_leaves_evidence_and_exits_nonzero(self):
        fake = FakeCodex(fail_calls={"p2ja:case:0003:B"})
        self.assertEqual(self._run(fake, max_workers=4), 1)
        manifest = self._manifest()
        by_key = {
            f"{call['case_id']}:{call['condition']}": call
            for call in manifest["calls"]
        }
        failed = by_key["p2ja:case:0003:B"]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["attempts"], 1)
        self.assertIn("codex exec returned 1", failed["last_error"])
        self.assertIsNotNone(failed["log_path"])
        self.assertTrue((self.run_dir / failed["log_path"]).is_file())
        self.assertIsNotNone(failed["temp_dir"])
        self.assertTrue(Path(failed["temp_dir"]).is_dir())  # evidence retained
        self.assertEqual(
            sum(1 for call in manifest["calls"] if call["status"] == "completed"),
            19,
        )
        self.assertEqual(self._import(fake), 1)  # incomplete run cannot import

    def test_retry_repeats_exact_same_prompt_and_config(self):
        fake = FakeCodex(fail_first_attempt={"p2ja:case:0002:A"})
        self.assertEqual(
            self._run(fake, max_workers=1, retries=1), 0,
        )
        self.assertEqual(len(fake.exec_argv), 21)  # 19 one-shot + 2 attempts
        entries = [
            entry for entry in fake.attempts_log
            if entry[0] == "p2ja:case:0002:A"
        ]
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0][2], entries[1][2])  # identical stdin bytes

        def _strip_paths(argv: list[str]) -> list[str]:
            remaining: list[str] = []
            index = 0
            while index < len(argv):
                if argv[index] in {"-C", "--output-schema", "-o"}:
                    index += 2
                    continue
                remaining.append(argv[index])
                index += 1
            return remaining

        self.assertEqual(
            _strip_paths(entries[0][1]),
            _strip_paths(entries[1][1]),
        )
        manifest = self._manifest()
        call = next(
            call for call in manifest["calls"]
            if call["case_id"] == "p2ja:case:0002" and call["condition"] == "A"
        )
        self.assertEqual(call["status"], "completed")
        self.assertEqual(call["attempts"], 2)

    def test_import_assembles_validated_bundle_and_fails_closed(self):
        fake = FakeCodex()
        self.assertEqual(self._run(fake, max_workers=4), 0)
        # The frozen fixture already contains an outputs file: fail closed.
        self.assertEqual(self._import(fake), 1)
        bundle_path = self.root / "sol-bundle.json"
        self.assertEqual(
            self._import(fake, outputs=bundle_path), 0,
        )
        self.assertEqual(
            self._import(fake, outputs=bundle_path), 1,  # exists -> fail closed
        )
        bundle = load_json_strict(bundle_path, label="outputs bundle")
        instructions = {
            key: value for key, value in load_json_strict(
                self.fixture["output_dir"] / OUTPUT_FILENAMES["instructions"],
                label="instructions",
            ).items()
            if key != "content_sha256"
        }
        validate_extraction_outputs_bundle(
            bundle,
            payloads_artifact=self.payloads,
            instructions=instructions,
        )
        manifest = self._manifest()
        final_outputs = manifest["final_outputs"]
        self.assertEqual(final_outputs["outputs_sha256"], bundle["content_sha256"])
        self.assertEqual(len(final_outputs["by_call"]), 20)
        # Deleting one raw response makes import fail closed.
        raw_path = self.run_dir / "raw" / "p2ja-case-0007-A.raw.json"
        raw_path.unlink()
        self.assertEqual(
            self._import(fake, outputs=self.root / "sol-bundle-2.json"), 1,
        )

    def test_status_and_run_validate_manifest_after_import(self):
        fake = FakeCodex()
        self.assertEqual(self._run(fake, max_workers=4), 0)
        bundle_path = self.root / "sol-bundle.json"
        self.assertEqual(self._import(fake, outputs=bundle_path), 0)
        manifest = self._manifest()
        self.assertIn("final_outputs", manifest)
        self.assertIsNotNone(manifest["final_outputs"])
        # status validates the imported manifest and reports completion.
        with mock.patch.object(runner.subprocess, "run", side_effect=fake):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.assertEqual(runner.main(self._args("status")), 0)
        report = json.loads(buffer.getvalue())
        self.assertEqual(report["completed"], 20)
        self.assertEqual(report["pending"], 0)
        # run validates the imported manifest and reuses every raw response.
        self.assertEqual(self._run(fake, max_workers=4), 0)
        self.assertEqual(len(fake.exec_argv), 20)  # no new model calls

    def test_post_import_manifest_tamper_rejected(self):
        fake = FakeCodex()
        self.assertEqual(self._run(fake, max_workers=4), 0)
        bundle_path = self.root / "sol-bundle.json"
        self.assertEqual(self._import(fake, outputs=bundle_path), 0)
        valid_manifest = self._manifest()

        def _write(manifest: dict[str, Any]) -> None:
            (self.run_dir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8",
            )

        def _status_code() -> int:
            with mock.patch.object(runner.subprocess, "run", side_effect=fake):
                return runner.main(self._args("status"))

        # Tampering with a recorded by_call hash fails closed even when the
        # manifest content hash is recomputed consistently.
        manifest = json.loads(json.dumps(valid_manifest))
        tampered_key = next(iter(manifest["final_outputs"]["by_call"]))
        manifest["final_outputs"]["by_call"][tampered_key] = "0" * 64
        manifest["content_sha256"] = canonical_sha256({
            key: value for key, value in manifest.items()
            if key != "content_sha256"
        })
        _write(manifest)
        self.assertEqual(_status_code(), 1)
        self.assertEqual(self._run(fake, max_workers=4), 1)

        # Tampering with the current outputs artifact bytes fails closed.
        _write(valid_manifest)
        bundle_bytes = bundle_path.read_bytes()
        bundle_path.write_bytes(bundle_bytes.replace(b'"purpose"', b'"purose"'))
        self.assertEqual(_status_code(), 1)
        self.assertEqual(self._import(fake), 1)
        bundle_path.write_bytes(bundle_bytes)

        # Tampering with a raw response after import fails closed via the
        # recomputed imported output hashes.
        raw_path = self.run_dir / "raw" / "p2ja-case-0002-B.raw.json"
        original = raw_path.read_text(encoding="utf-8")
        raw_path.write_text(
            original.replace('"occurrence_index": 0', '"occurrence_index": 1'),
            encoding="utf-8",
        )
        self.assertEqual(_status_code(), 1)
        raw_path.write_text(original, encoding="utf-8")

        # Dropping final_outputs (or the outputs artifact) fails closed.
        manifest = json.loads(json.dumps(valid_manifest))
        del manifest["final_outputs"]
        manifest["content_sha256"] = canonical_sha256({
            key: value for key, value in manifest.items()
            if key != "content_sha256"
        })
        _write(manifest)
        self.assertEqual(_status_code(), 1)
        _write(valid_manifest)
        bundle_path.unlink()
        self.assertEqual(_status_code(), 1)

    def test_manifest_claiming_other_model_or_config_rejected(self):
        fake = FakeCodex()
        self.assertEqual(self._run(fake, max_workers=4), 0)
        manifest = self._manifest()
        manifest["requested_model"] = "gpt-5.6-flash"
        manifest["content_sha256"] = canonical_sha256({
            key: value for key, value in manifest.items()
            if key != "content_sha256"
        })
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        self.assertEqual(self._run(fake, max_workers=4), 1)
        self.assertEqual(self._import(fake), 1)

    def test_max_workers_and_retries_bounds(self):
        fake = FakeCodex()
        for bad in (0, 5):
            self.assertEqual(self._run(fake, max_workers=bad), 1)
        self.assertEqual(self._run(fake, retries=-1), 1)

    def test_schema_subcommand_writes_canonical_schema_and_fails_closed(self):
        fake = FakeCodex()
        args = self._args("schema")
        with mock.patch.object(runner.subprocess, "run", side_effect=fake):
            self.assertEqual(runner.main(args), 0)
        schema_path = self.run_dir / "intermediate-schema.json"
        self.assertEqual(schema_path.read_bytes(), runner._schema_bytes())
        with mock.patch.object(runner.subprocess, "run", side_effect=fake):
            self.assertEqual(runner.main(args), 0)  # unchanged: ok
        schema_path.write_text("{}", encoding="utf-8")
        with mock.patch.object(runner.subprocess, "run", side_effect=fake):
            self.assertEqual(runner.main(args), 1)  # mismatch fails closed
        with mock.patch.object(runner.subprocess, "run", side_effect=fake):
            self.assertEqual(runner.main(args + ["--force"]), 0)
        self.assertEqual(schema_path.read_bytes(), runner._schema_bytes())

    def test_manifest_rejects_changed_argv_template(self):
        fake = FakeCodex()
        self.assertEqual(self._run(fake, max_workers=4), 0)
        manifest = self._manifest()
        manifest["argv_template"] = ["codex", "exec", "--different"]
        manifest["content_sha256"] = canonical_sha256({
            key: value for key, value in manifest.items()
            if key != "content_sha256"
        })
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        self.assertEqual(self._run(fake, max_workers=4), 1)

    def test_status_reports_pending_then_completed(self):
        fake = FakeCodex()
        with mock.patch.object(runner.subprocess, "run", side_effect=fake):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.assertEqual(runner.main(self._args("status")), 0)
        self.assertEqual(json.loads(buffer.getvalue())["pending"], 20)
        self.assertEqual(self._run(fake, max_workers=4), 0)
        with mock.patch.object(runner.subprocess, "run", side_effect=fake):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.assertEqual(runner.main(self._args("status")), 0)
        report = json.loads(buffer.getvalue())
        self.assertEqual(report["completed"], 20)
        self.assertEqual(report["pending"], 0)


if __name__ == "__main__":
    unittest.main()
