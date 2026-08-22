"""CLI tests for the Phase 2K build and human-review finalize scripts."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

import scripts.build_phase2k_reconstruction as build_cli
import scripts.finalize_phase2k_human_review as finalize_cli
from pipeline.phase2k_contextual_reconstruction import (
    HUMAN_SCORE_FIELDS,
    INFERENCE_CONFIG_VERSION,
    NO_PROVIDER_INFERENCE_CONFIG,
    canonical_sha256,
    inference_config_hash,
    load_json_strict,
)
from tests._phase2k_helpers import build_fixture
from tests.test_phase2k_contextual_reconstruction import (
    _mechanical_raw,
    _polish_raw,
    _reconstruction_raw,
    _sufficiency_raw,
)


CLI_TEST_INFERENCE_CONFIG = {
    "provider": "test-backend",
    "model": "test-model",
    "endpoint": "https://example.test/endpoint",
    "temperature": 0.0,
    "max_tokens": 8192,
    "thinking": "disabled",
    "purpose": "phase2k-test-live",
}


def _live_chat() -> tuple[Any, dict[str, Any]]:
    calls: list[dict[str, str]] = []

    def chat(system: str, user: str) -> str:
        calls.append({"system": system, "user": user})
        payload = json.loads(user)
        task = payload.get("task")
        if task == "mechanical_cleanup":
            return _mechanical_raw({
                **payload["target"],
                "source_text": payload["target"]["bronze_text"],
                "upstream_start": payload["target"]["upstream_start"],
                "upstream_end": payload["target"]["upstream_end"],
                "upstream_content_sha256": payload["target"]["upstream_content_sha256"],
                "canonical_record_sha256": payload["target"]["canonical_record_sha256"],
                "metadata": {
                    key: payload["metadata"][key]["value"]
                    for key in ("champion", "role", "video_title")
                    if key in payload["metadata"]
                },
            })
        if task == "semantic_sufficiency":
            champion = payload["metadata"].get("champion", {}).get("value", "Lux")
            return _sufficiency_raw("SUFFICIENT", champion=champion)
        if task == "reconstruction":
            bronze = payload["target"]["bronze_text"]
            selected = {
                "source_text": bronze,
                "upstream_start": payload["target"]["upstream_start"],
                "upstream_end": payload["target"]["upstream_end"],
                "upstream_content_sha256": payload["target"]["upstream_content_sha256"],
                "canonical_record_sha256": payload["target"]["canonical_record_sha256"],
                "window_id": payload["target"]["window_id"],
                "source_group_id": payload["target"]["source_group_id"],
                "metadata": {
                    key: payload["metadata"][key]["value"]
                    for key in ("champion", "role", "video_title")
                    if key in payload["metadata"]
                },
            }
            return _reconstruction_raw(
                cleaned=bronze,
                bronze=bronze,
                base_offset=payload["target"]["upstream_start"],
                selected=selected,
            )
        if task == "semantic_polish":
            bronze = payload["target"]["bronze_text"]
            selected = {
                "source_text": bronze,
                "upstream_start": payload["target"]["upstream_start"],
                "upstream_end": payload["target"]["upstream_end"],
                "upstream_content_sha256": payload["target"]["upstream_content_sha256"],
                "canonical_record_sha256": payload["target"]["canonical_record_sha256"],
                "window_id": payload["target"]["window_id"],
                "source_group_id": payload["target"]["source_group_id"],
                "metadata": {
                    key: payload["metadata"][key]["value"]
                    for key in ("champion", "role", "video_title")
                    if key in payload["metadata"]
                },
            }
            return _polish_raw(selected, payload["reconstruction"])
        raise AssertionError(f"unknown task {task}")

    chat.calls = calls  # type: ignore[attr-defined]
    return chat, CLI_TEST_INFERENCE_CONFIG


class Phase2KBuildCliTests(unittest.TestCase):
    def test_no_provider_cli_builds_deterministic_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            output = root / "phase2k"
            with self.assertRaises(SystemExit):
                build_cli.main([
                    "--manifest", str(manifest_path),
                    "--reviewed-packet", str(packet_path),
                    "--doc", str(root / "missing-doc.md"),
                    "--db", str(db_path),
                    "--output-dir", str(output),
                ])  # missing doc is rejected by argparse
            doc_path = root / "replication.md"
            doc_path.write_text("frozen replication doc", encoding="utf-8")
            code = build_cli.main([
                "--manifest", str(manifest_path),
                "--reviewed-packet", str(packet_path),
                "--doc", str(doc_path),
                "--db", str(db_path),
                "--output-dir", str(output),
            ])
            self.assertEqual(code, 0)
            records = load_json_strict(
                output / build_cli.OUTPUT_FILENAMES["records"], label="records",
            )
            self.assertEqual(records["mode"], "no_provider")
            self.assertEqual(len(records["records"]), 120)
            self.assertEqual(records["schema_version"], "phase2k-reconstruction-records-v7")
            self.assertEqual(
                records["inference_config"],
                NO_PROVIDER_INFERENCE_CONFIG,
            )
            self.assertEqual(
                records["inference_config_hash"],
                inference_config_hash(NO_PROVIDER_INFERENCE_CONFIG),
            )
            self.assertFalse(any(
                record["record_type"] == "D"
                and record["content"]["generation_status"] == "GENERATED"
                for record in records["records"]
            ))
            # Output dir must fail if it already exists.
            code = build_cli.main([
                "--manifest", str(manifest_path),
                "--reviewed-packet", str(packet_path),
                "--doc", str(doc_path),
                "--db", str(db_path),
                "--output-dir", str(output),
            ])
            self.assertEqual(code, 1)
            # Validate-only on the existing output passes.
            code = build_cli.main([
                "--manifest", str(manifest_path),
                "--reviewed-packet", str(packet_path),
                "--doc", str(doc_path),
                "--db", str(db_path),
                "--output-dir", str(output),
                "--validate-only",
            ])
            self.assertEqual(code, 0)

    def test_live_cli_builds_full_abcd_with_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            original_factory = build_cli._live_chat_factory
            build_cli._live_chat_factory = _live_chat
            try:
                output = root / "phase2k-live"
                cache_dir = root / "cache"
                code = build_cli.main([
                    "--manifest", str(manifest_path),
                    "--reviewed-packet", str(packet_path),
                    "--db", str(db_path),
                    "--output-dir", str(output),
                    "--cache-dir", str(cache_dir),
                    "--live",
                ])
            finally:
                build_cli._live_chat_factory = original_factory
            self.assertEqual(code, 0)
            records = load_json_strict(
                output / build_cli.OUTPUT_FILENAMES["records"], label="records",
            )
            self.assertEqual(records["mode"], "live")
            self.assertEqual(
                records["inference_config"],
                CLI_TEST_INFERENCE_CONFIG,
            )
            self.assertEqual(
                records["inference_config_hash"],
                inference_config_hash(CLI_TEST_INFERENCE_CONFIG),
            )
            self.assertEqual(
                records["inference_config"]["thinking"],
                "disabled",
            )
            self.assertEqual(
                records["inference_config"]["temperature"],
                0.0,
            )
            self.assertEqual(
                records["inference_config"]["max_tokens"],
                8192,
            )
            summary = load_json_strict(
                output / build_cli.OUTPUT_FILENAMES["build_summary"],
                label="build summary",
            )
            self.assertEqual(
                summary["inference_config_hash"],
                records["inference_config_hash"],
            )
            d_record = next(
                record
                for record in records["records"]
                if record["record_type"] == "D"
            )
            self.assertEqual(
                d_record["content"]["model_call"]["inference_config_hash"],
                records["inference_config_hash"],
            )
            self.assertEqual(
                d_record["content"]["model_call"]["inference_config_version"],
                INFERENCE_CONFIG_VERSION,
            )
            d_records = [
                record for record in records["records"]
                if record["record_type"] == "D"
            ]
            self.assertEqual(len(d_records), 30)
            self.assertTrue(all(
                record["content"]["generation_status"] == "GENERATED"
                for record in d_records
            ))
            self.assertTrue(any(
                path.name == "r1.json"
                for path in (output / "attempts").rglob("*.json")
            ))
            human = load_json_strict(
                output / build_cli.OUTPUT_FILENAMES["human_packet"],
                label="human packet",
            )
            mapping = load_json_strict(
                output / build_cli.OUTPUT_FILENAMES["human_mapping"],
                label="human mapping",
            )
            self.assertIn(
                "D",
                {
                    mapping["labels"][item["blinded_label"]]["condition_code"]
                    for item in human["review_items"]
                },
            )
            code = build_cli.main([
                "--manifest", str(manifest_path),
                "--reviewed-packet", str(packet_path),
                "--db", str(db_path),
                "--output-dir", str(output),
                "--cache-dir", str(cache_dir),
                "--validate-only",
            ])
            self.assertEqual(code, 0)

    def test_live_factory_seals_thinking_disabled_config(self):
        fake_core = types.ModuleType("core")
        fake_llm = types.ModuleType("core.llm")
        fake_llm.BACKEND = "deepseek"
        fake_llm.MODEL = "deepseek-v4-flash-test"
        fake_llm._DEEPSEEK_BASE_URL = "https://api.deepseek.example"
        calls: list[dict[str, Any]] = []

        def fake_chat(**kwargs: Any) -> str:
            calls.append(kwargs)
            return '{"ok": true}'

        fake_llm.chat = fake_chat
        fake_core.llm = fake_llm
        old_core = sys.modules.get("core")
        old_llm = sys.modules.get("core.llm")
        sys.modules["core"] = fake_core
        sys.modules["core.llm"] = fake_llm
        try:
            chat, config = build_cli._live_chat_factory()
            self.assertEqual(config["provider"], "deepseek")
            self.assertEqual(config["model"], "deepseek-v4-flash-test")
            self.assertEqual(
                config["endpoint"], "https://api.deepseek.example",
            )
            self.assertEqual(config["temperature"], 0.0)
            self.assertEqual(config["max_tokens"], 8192)
            self.assertEqual(config["thinking"], "disabled")
            self.assertEqual(
                config["purpose"], build_cli.LIVE_INFERENCE_PURPOSE,
            )
            self.assertFalse(any(
                name
                in {
                    "api_key", "apikey", "authorization", "bearer",
                    "password", "passwd", "secret", "credential",
                    "private_key",
                }
                for name in config
            ))
            chat("system", "user")
            self.assertEqual(calls, [{
                "system": "system",
                "user": "user",
                "temperature": 0.0,
                "max_tokens": 8192,
                "thinking": "disabled",
            }])
            self.assertEqual(
                build_cli.LIVE_CHAT_KWARGS,
                {"temperature": 0.0, "max_tokens": 8192, "thinking": "disabled"},
            )
        finally:
            if old_core is None:
                sys.modules.pop("core", None)
            else:
                sys.modules["core"] = old_core
            if old_llm is None:
                sys.modules.pop("core.llm", None)
            else:
                sys.modules["core.llm"] = old_llm


class Phase2KFinalizeCliTests(unittest.TestCase):
    def _build_packet(self, root: Path) -> tuple[Path, Path, Path, Path]:
        manifest_path, packet_path, db_path = build_fixture(root)
        output = root / "phase2k"
        self.assertEqual(build_cli.main([
            "--manifest", str(manifest_path),
            "--reviewed-packet", str(packet_path),
            "--db", str(db_path),
            "--output-dir", str(output),
        ]), 0)
        return (
            output,
            output / build_cli.OUTPUT_FILENAMES["human_packet"],
            output / build_cli.OUTPUT_FILENAMES["human_mapping"],
            output / build_cli.OUTPUT_FILENAMES["records"],
        )

    def test_finalize_refuses_incomplete_and_summarizes_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, packet_path, mapping_path, records_path = self._build_packet(root)
            packet = load_json_strict(packet_path, label="packet")
            item_ids = [item["review_item_id"] for item in packet["review_items"]]
            incomplete = {
                item_id: {
                    "scores": {
                        field: 4 if field != "causality" else None
                        for field in HUMAN_SCORE_FIELDS
                    },
                    "reviewer": "human",
                    "completed_at": "2026-08-19T00:00:00.000Z",
                    "notes": [],
                }
                for item_id in item_ids
            }
            reviews_path = root / "incomplete-reviews.json"
            reviews_path.write_text(json.dumps(incomplete), encoding="utf-8")
            code = finalize_cli.main([
                "--output-dir", str(output),
                "--reviews", str(reviews_path),
                "--reviewer", "human",
                "--completed-at", "2026-08-19T00:00:00.000Z",
            ])
            self.assertEqual(code, 1)
            self.assertFalse(
                (output / finalize_cli.OUTPUT_FILENAMES["finalized_packet"]).exists(),
            )

            complete = {
                item_id: {
                    "scores": {
                        field: index % 6
                        for index, field in enumerate(HUMAN_SCORE_FIELDS)
                    },
                    "reviewer": "human",
                    "completed_at": "2026-08-19T00:00:00.000Z",
                    "notes": [],
                }
                for item_id in item_ids
            }
            reviews_path = root / "complete-reviews.json"
            reviews_path.write_text(json.dumps(complete), encoding="utf-8")
            code = finalize_cli.main([
                "--output-dir", str(output),
                "--reviews", str(reviews_path),
                "--reviewer", "human",
                "--completed-at", "2026-08-19T00:00:00.000Z",
            ])
            self.assertEqual(code, 0)
            finalized = load_json_strict(
                output / finalize_cli.OUTPUT_FILENAMES["finalized_packet"],
                label="finalized",
            )
            self.assertEqual(finalized["release_gate"], "REVIEWED")
            summary = load_json_strict(
                output / finalize_cli.OUTPUT_FILENAMES["human_summary"],
                label="summary",
            )
            self.assertEqual(
                summary["overall"]["item_count"], len(item_ids),
            )
            self.assertIn("by_condition", summary)
            # Blank official packet is untouched.
            official = load_json_strict(packet_path, label="official")
            for item in official["review_items"]:
                self.assertTrue(all(value is None for value in item["scores"].values()))


if __name__ == "__main__":
    unittest.main()
