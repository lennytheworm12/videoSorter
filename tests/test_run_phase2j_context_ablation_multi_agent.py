"""Focused tests for the Phase 2J multi-agent (multi_agent_v1) transport.

No real model, network, or codex calls are made.  The tests cover the exact
20 pending calls, byte-identical prompt identity and payload isolation,
fail-closed ingest binding (case/condition/payload/instructions hashes,
quotes, occurrence indexes, extra keys, malformed JSON), atomic
manifest/raw mutation semantics, path-traversal rejection, deterministic
import, and post-import tamper rejection.
"""

from __future__ import annotations

import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import scripts.run_phase2j_context_ablation_multi_agent as runner
import scripts.run_phase2j_context_ablation_sol as sol_runner
from pipeline.phase2j_context_ablation import (
    OUTPUT_FILENAMES,
    build_sol_intermediate_schema,
    canonical_sha256,
    load_json_strict,
    normalize_path_locator,
    text_sha256,
    validate_extraction_outputs_bundle,
)
from tests.test_phase2j_context_ablation import _build_full_fixture


SEMANTIC_FIELDS = [
    "actors", "ability_resource_references", "event", "condition",
    "advice_action", "consequence", "uncertainty", "supporting_source_ranges",
]


def _intermediate_response(
    payload: dict[str, Any],
    *,
    case_id: str | None = None,
    condition: str | None = None,
    quote: str | None = None,
    occurrence_index: int | None = None,
    extra_keys: bool = False,
) -> dict[str, Any]:
    source = (
        payload["target"]["bronze_text"]
        if payload["condition"] == "A"
        else payload["transcript"]
    )
    first_token = re.search(r"\S+", source)
    if first_token is None:
        raise AssertionError("payload source has no tokens")
    fields: dict[str, Any] = {}
    for field in SEMANTIC_FIELDS:
        fields[field] = []
    fields["actors"] = [{
        "extraction_text": "coach",
        "resolution_status": "literal_explicit",
        "source_references": [{
            "quote": quote if quote is not None else first_token.group(),
            "occurrence_index": 0
            if occurrence_index is None else occurrence_index,
        }],
    }]
    fields["supporting_source_ranges"] = [{
        "extraction_text": "target range",
        "resolution_status": "literal_explicit",
        "source_references": [{
            "quote": quote if quote is not None else first_token.group(),
            "occurrence_index": 0
            if occurrence_index is None else occurrence_index,
        }],
    }]
    response = {
        "schema_version": build_sol_intermediate_schema()["schema_version"],
        "case_id": case_id if case_id is not None else payload["case_id"],
        "condition": condition if condition is not None else payload["condition"],
        "payload_sha256": payload["content_sha256"],
        "instructions_sha256": payload["instructions_sha256"],
        "fields": fields,
    }
    if extra_keys:
        response["sneaky"] = True
    return response


class MultiAgentTransportTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.fixture = _build_full_fixture(self.root)
        self.run_dir = self.root / "sol_multi_agent_run_v2"
        self.payloads = load_json_strict(
            self.fixture["output_dir"] / OUTPUT_FILENAMES["payloads"],
            label="payloads",
        )

    def tearDown(self):
        self._temporary.cleanup()

    def _args(self, command: str, **extra: Any) -> list[str]:
        args = [
            command,
            "--manifest", str(self.fixture["manifest_path"]),
            "--reviewed-packet", str(self.fixture["packet_path"]),
            "--db", str(self.fixture["db_path"]),
            "--output-dir", str(self.fixture["output_dir"]),
            "--run-dir", str(self.run_dir),
        ]
        for key, value in extra.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    args.append(flag)
            else:
                args.append(flag)
                args.append(str(value))
        return args

    def _call(self, command: str, **extra: Any) -> int:
        return runner.main(self._args(command, **extra))

    def _manifest(self) -> dict[str, Any]:
        return json.loads((self.run_dir / "manifest.json").read_text(
            encoding="utf-8",
        ))

    def _expected_payloads(self) -> dict[str, dict[str, Any]]:
        return {
            f"{payload_case['case_id']}:{condition}": payload_case[condition]
            for payload_case in self.payloads["cases"]
            for condition in ("A", "B")
        }

    def _expected_keys(self) -> list[str]:
        return [
            f"{payload_case['case_id']}:{condition}"
            for payload_case in self.payloads["cases"]
            for condition in ("A", "B")
        ]

    def _stage(
        self,
        key: str,
        payload: dict[str, Any],
        *,
        body: str | None = None,
        **overrides: Any,
    ) -> Path:
        staged_dir = self.root / "staged"
        staged_dir.mkdir(parents=True, exist_ok=True)
        path = staged_dir / f"{key.replace(':', '-')}.json"
        if body is not None:
            path.write_text(body, encoding="utf-8")
        else:
            path.write_text(json.dumps(_intermediate_response(
                payload,
                **overrides,
            )), encoding="utf-8")
        return path

    def _ingest_all(self, *, agent_prefix: str = "agent") -> None:
        expected = self._expected_payloads()
        for index, key in enumerate(self._expected_keys(), 1):
            staged = self._stage(key, expected[key])
            case_id, condition = key.rsplit(":", 1)
            code = self._call(
                "ingest",
                case_id=case_id,
                condition=condition,
                agent_id=f"{agent_prefix}-{index:02d}",
                response=staged,
            )
            self.assertEqual(code, 0, key)

    def test_init_creates_exact_20_calls_and_strict_metadata(self):
        self.assertEqual(self._call("init"), 0)
        manifest = self._manifest()
        self.assertEqual(set(manifest), set(runner.MANIFEST_KEYS))
        self.assertEqual(
            manifest["schema_version"],
            runner.MULTI_AGENT_RUN_SCHEMA_VERSION,
        )
        self.assertEqual(manifest["transport"], "multi_agent_v1")
        self.assertEqual(manifest["requested_model"], "gpt-5.6-sol")
        self.assertEqual(manifest["reasoning_effort"], "high")
        self.assertEqual(
            manifest["wrapper_sha256"],
            text_sha256(sol_runner.SOL_WRAPPER_PROMPT),
        )
        self.assertEqual(
            manifest["intermediate_schema_sha256"],
            canonical_sha256(build_sol_intermediate_schema()),
        )
        self.assertEqual(
            manifest["instructions_sha256"],
            self.payloads["instructions_sha256"],
        )
        self.assertEqual(
            manifest["payloads_sha256"], self.payloads["content_sha256"],
        )
        self.assertEqual(
            manifest["run_dir"], normalize_path_locator(self.run_dir),
        )
        self.assertEqual(
            manifest["content_sha256"],
            canonical_sha256({
                key: value for key, value in manifest.items()
                if key != "content_sha256"
            }),
        )
        self.assertIsNone(manifest["final_outputs"])
        self.assertIn("not cryptographically proven", manifest["purpose"])
        self.assertIn("transport-provided", manifest["purpose"])
        self.assertIn(
            "canonical wrapper prompt plus the canonical inner condition "
            "payload",
            manifest["purpose"],
        )
        self.assertNotIn("argv_template", manifest)
        self.assertNotIn("codex_cli_version", manifest)
        for timestamp_key in ("started_at", "completed_at", "created_at"):
            self.assertNotIn(timestamp_key, manifest)
        calls = manifest["calls"]
        self.assertEqual(len(calls), 20)
        expected = self._expected_payloads()
        for index, call in enumerate(calls):
            self.assertEqual(set(call), set(runner.CALL_KEYS))
            self.assertEqual(
                (call["case_id"], call["condition"]),
                (self._expected_keys()[index].rsplit(":", 1)[0],
                 self._expected_keys()[index].rsplit(":", 1)[1]),
            )
            key = self._expected_keys()[index]
            self.assertEqual(
                call["payload_sha256"], expected[key]["content_sha256"],
            )
            self.assertEqual(call["status"], "pending")
            self.assertEqual(call["attempts"], 0)
            self.assertIsNone(call["prompt_sha256"])
            self.assertIsNone(call["agent_id"])
            self.assertIsNone(call["raw_response_path"])
            self.assertIsNone(call["raw_response_sha256"])
            self.assertIsNone(call["last_error"])
            for timestamp_key in ("started_at", "completed_at", "timestamp"):
                self.assertNotIn(timestamp_key, call)
        schema_path = self.run_dir / "intermediate-schema.json"
        self.assertEqual(schema_path.read_bytes(), runner._schema_bytes())
        # Re-init without --force fails closed; with --force it replaces.
        self.assertEqual(self._call("init"), 1)
        self.assertEqual(self._call("init", force=True), 0)

    def test_prompt_byte_identity_and_payload_isolation(self):
        self.assertEqual(self._call("init"), 0)
        expected = self._expected_payloads()
        sibling_case_ids = [
            payload_case["case_id"]
            for payload_case in self.payloads["cases"]
        ]
        for key, payload in expected.items():
            case_id, condition = key.rsplit(":", 1)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = self._call(
                    "prompt", case_id=case_id, condition=condition,
                )
            self.assertEqual(code, 0)
            text = buffer.getvalue()
            self.assertEqual(
                text,
                sol_runner.SOL_WRAPPER_PROMPT + "\n\n"
                + runner._canonical_json(payload),
            )
            parsed = json.loads(text[len(sol_runner.SOL_WRAPPER_PROMPT) + 2:])
            self.assertEqual(parsed, payload)
            for forbidden in (
                '"provenance_by_case"', '"video_url"', '"window_id"',
                '"selection_manifest_sha256"', '"reviewed-endpoint"',
            ):
                self.assertNotIn(forbidden, text)
            for sibling in sibling_case_ids:
                if sibling != case_id:
                    self.assertNotIn(sibling, text)
            if condition == "A":
                self.assertNotIn('"transcript"', text)
                self.assertNotIn('"metadata"', text)
                self.assertNotIn('"vocabulary"', text)
        # Unknown/noncanonical calls fail closed with empty stdout.
        for case_id, condition in (
            ("p2ja:case:9999", "A"),
            ("p2ja:case:0001", "C"),
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = self._call(
                    "prompt", case_id=case_id, condition=condition,
                )
            self.assertEqual(code, 1)
            self.assertEqual(buffer.getvalue(), "")

    def test_successful_ingest_status_import_and_post_import_validation(self):
        self.assertEqual(self._call("init"), 0)
        self._ingest_all()
        manifest = self._manifest()
        expected = self._expected_payloads()
        for index, call in enumerate(manifest["calls"]):
            self.assertEqual(call["status"], "completed")
            self.assertEqual(call["attempts"], 1)
            key = self._expected_keys()[index]
            self.assertEqual(call["agent_id"], f"agent-{index + 1:02d}")
            self.assertEqual(
                call["prompt_sha256"],
                text_sha256(
                    sol_runner.SOL_WRAPPER_PROMPT + "\n\n"
                    + runner._canonical_json(expected[key]),
                ),
            )
            raw_path = self.run_dir / call["raw_response_path"]
            self.assertTrue(raw_path.is_file())
            self.assertEqual(call["raw_response_sha256"], runner.file_sha256(raw_path))
            self.assertNotIn("..", call["raw_response_path"])
        raw_files = list((self.run_dir / "raw").glob("*.raw.json"))
        self.assertEqual(len(raw_files), 20)
        # status validates the manifest and every completed raw response.
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(self._call("status"), 0)
        report = json.loads(buffer.getvalue())
        self.assertEqual(report["completed"], 20)
        self.assertEqual(report["pending"], 0)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(
            [entry["status"] for entry in report["calls"]],
            ["completed"] * 20,
        )
        # import requires an explicit outputs path or --force because the
        # frozen fixture already contains the standard outputs artifact.
        self.assertEqual(self._call("import"), 1)
        bundle_path = self.root / "multi-agent-bundle.json"
        self.assertEqual(
            self._call("import", outputs=bundle_path), 0,
        )
        self.assertEqual(
            self._call("import", outputs=bundle_path), 1,  # exists -> fail closed
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
        self.assertEqual(
            final_outputs["outputs_sha256"], bundle["content_sha256"],
        )
        self.assertEqual(
            final_outputs["outputs_file_sha256"], runner.file_sha256(bundle_path),
        )
        self.assertEqual(
            set(final_outputs["by_call"]), set(self._expected_keys()),
        )
        self.assertEqual(len(final_outputs["by_call"]), 20)
        for case in bundle["cases"]:
            for condition in ("A", "B"):
                key = f"{case['case_id']}:{condition}"
                self.assertEqual(
                    final_outputs["by_call"][key],
                    case[condition]["content_sha256"],
                )
        # Post-import status validates the full manifest and all outputs.
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(self._call("status"), 0)
        self.assertEqual(json.loads(buffer.getvalue())["completed"], 20)

    def test_prompt_for_completed_call_still_outputs_exact_canonical_prompt(self):
        self.assertEqual(self._call("init"), 0)
        key = "p2ja:case:0001:A"
        payload = self._expected_payloads()[key]
        staged = self._stage(key, payload)
        self.assertEqual(
            self._call(
                "ingest",
                case_id="p2ja:case:0001",
                condition="A",
                agent_id="audit-agent",
                response=staged,
            ),
            0,
        )
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(
                self._call(
                    "prompt", case_id="p2ja:case:0001", condition="A",
                ),
                0,
            )
        self.assertEqual(
            buffer.getvalue(),
            sol_runner.SOL_WRAPPER_PROMPT + "\n\n"
            + runner._canonical_json(payload),
        )

    def test_ingest_refuses_replacing_completed_evidence_without_force(self):
        self.assertEqual(self._call("init"), 0)
        key = "p2ja:case:0002:B"
        payload = self._expected_payloads()[key]
        staged = self._stage(key, payload)
        args = {
            "case_id": "p2ja:case:0002",
            "condition": "B",
            "agent_id": "agent-one",
            "response": staged,
        }
        self.assertEqual(self._call("ingest", **args), 0)
        manifest_before = self._manifest()
        raw_before = (self.run_dir / "raw" / "p2ja-case-0002-B.raw.json").read_bytes()
        self.assertEqual(self._call("ingest", **args), 1)  # no force -> fail
        self.assertEqual(self._manifest(), manifest_before)
        self.assertEqual(
            (self.run_dir / "raw" / "p2ja-case-0002-B.raw.json").read_bytes(),
            raw_before,
        )
        self.assertEqual(
            self._call("ingest", force=True, **args), 0,
        )
        manifest_after = self._manifest()
        call = next(
            call for call in manifest_after["calls"]
            if call["case_id"] == "p2ja:case:0002" and call["condition"] == "B"
        )
        self.assertEqual(call["status"], "completed")
        self.assertEqual(call["attempts"], 2)

    def test_failures_do_not_mutate_manifest_or_raw(self):
        self.assertEqual(self._call("init"), 0)
        expected = self._expected_payloads()
        key = "p2ja:case:0003:A"
        payload = expected[key]
        manifest_before = (self.run_dir / "manifest.json").read_bytes()
        scenarios: list[tuple[str, Any]] = []
        staged = self._stage(key, payload)
        scenarios.append(("missing response file", {
            "response": self.root / "does-not-exist.json",
        }))
        scenarios.append(("malformed JSON", {
            "response": self._stage(key, payload, body="{broken"),
        }))
        scenarios.append(("non-object JSON", {
            "response": self._stage(key, payload, body="[1, 2, 3]"),
        }))
        scenarios.append(("wrong case_id", {
            "response": self._stage(
                key, payload,
                case_id="p2ja:case:0004",
            ),
        }))
        scenarios.append(("wrong condition", {
            "response": self._stage(
                key, payload,
                condition="B",
            ),
        }))
        wrong_payload_hash = dict(_intermediate_response(payload))
        wrong_payload_hash["payload_sha256"] = "0" * 64
        scenarios.append(("wrong payload hash", {
            "response": self._stage(key, payload, body=json.dumps(
                wrong_payload_hash,
            )),
        }))
        wrong_instructions = dict(_intermediate_response(payload))
        wrong_instructions["instructions_sha256"] = "0" * 64
        scenarios.append(("wrong instructions hash", {
            "response": self._stage(key, payload, body=json.dumps(
                wrong_instructions,
            )),
        }))
        scenarios.append(("bad quote", {
            "response": self._stage(
                key, payload, quote="quote-that-cannot-exist-xyzzy",
            ),
        }))
        scenarios.append(("bad occurrence index", {
            "response": self._stage(
                key, payload, occurrence_index=999,
            ),
        }))
        scenarios.append(("unknown extra keys", {
            "response": self._stage(key, payload, extra_keys=True),
        }))
        for label, overrides in scenarios:
            code = self._call(
                "ingest",
                case_id="p2ja:case:0003",
                condition="A",
                agent_id="bad-agent",
                **overrides,
            )
            self.assertEqual(code, 1, label)
            self.assertEqual(
                (self.run_dir / "manifest.json").read_bytes(),
                manifest_before,
                label,
            )
        self.assertEqual(
            list((self.run_dir / "raw").glob("*")), [], "no raw files written",
        )
        # A valid ingest still succeeds afterwards.
        staged = self._stage(key, payload)
        self.assertEqual(
            self._call(
                "ingest",
                case_id="p2ja:case:0003",
                condition="A",
                agent_id="good-agent",
                response=staged,
            ),
            0,
        )

    def test_tampered_manifest_raw_output_by_call_rejected(self):
        self.assertEqual(self._call("init"), 0)
        self._ingest_all()
        bundle_path = self.root / "multi-agent-bundle.json"
        self.assertEqual(self._call("import", outputs=bundle_path), 0)
        valid_manifest = self._manifest()

        def _write(manifest: dict[str, Any]) -> None:
            (self.run_dir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8",
            )

        def _status_code() -> int:
            return self._call("status")

        # Pre-import raw tampering fails closed via raw re-validation.
        raw_path = self.run_dir / "raw" / "p2ja-case-0001-A.raw.json"
        original_raw = raw_path.read_text(encoding="utf-8")
        raw_path.write_text(
            original_raw.replace('"occurrence_index": 0', '"occurrence_index": 1'),
            encoding="utf-8",
        )
        self.assertEqual(_status_code(), 1)
        raw_path.write_text(original_raw, encoding="utf-8")

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
        self.assertEqual(self._call("import", outputs=bundle_path), 1)

        # Tampering with the current outputs artifact bytes fails closed.
        _write(valid_manifest)
        bundle_bytes = bundle_path.read_bytes()
        bundle_path.write_bytes(bundle_bytes.replace(b'"purpose"', b'"purose"'))
        self.assertEqual(_status_code(), 1)
        self.assertEqual(self._call("import", outputs=bundle_path), 1)
        bundle_path.write_bytes(bundle_bytes)

        # Tampering with a raw response after import fails closed via the
        # recomputed imported output hashes.
        original_raw = raw_path.read_text(encoding="utf-8")
        raw_path.write_text(
            original_raw.replace('"occurrence_index": 0', '"occurrence_index": 1'),
            encoding="utf-8",
        )
        self.assertEqual(_status_code(), 1)
        raw_path.write_text(original_raw, encoding="utf-8")

        # Dropping final_outputs (a required key) fails closed.
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

    def test_path_traversal_in_recorded_raw_path_rejected(self):
        self.assertEqual(self._call("init"), 0)
        key = "p2ja:case:0005:A"
        payload = self._expected_payloads()[key]
        staged = self._stage(key, payload)
        self.assertEqual(
            self._call(
                "ingest",
                case_id="p2ja:case:0005",
                condition="A",
                agent_id="agent",
                response=staged,
            ),
            0,
        )
        manifest = self._manifest()
        valid_calls = json.loads(json.dumps(manifest["calls"]))
        call = next(
            call for call in manifest["calls"]
            if call["case_id"] == "p2ja:case:0005" and call["condition"] == "A"
        )
        call["raw_response_path"] = "../escaped.raw.json"
        manifest["content_sha256"] = canonical_sha256({
            key: value for key, value in manifest.items()
            if key != "content_sha256"
        })
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        self.assertEqual(self._call("status"), 1)
        # An absolute recorded path is also rejected.
        manifest["calls"] = json.loads(json.dumps(valid_calls))
        call = next(
            call for call in manifest["calls"]
            if call["case_id"] == "p2ja:case:0005" and call["condition"] == "A"
        )
        call["raw_response_path"] = "/etc/passwd"
        manifest["content_sha256"] = canonical_sha256({
            key: value for key, value in manifest.items()
            if key != "content_sha256"
        })
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        self.assertEqual(self._call("status"), 1)

    def test_status_reports_pending_then_completed(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(self._call("status"), 0)
        self.assertEqual(json.loads(buffer.getvalue())["pending"], 20)
        self.assertEqual(self._call("init"), 0)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(self._call("status"), 0)
        self.assertEqual(json.loads(buffer.getvalue())["pending"], 20)


if __name__ == "__main__":
    unittest.main()
