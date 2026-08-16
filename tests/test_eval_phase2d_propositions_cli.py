from __future__ import annotations

import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE = "scripts.eval_phase2d_propositions"


class Phase2DPropositionCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = importlib.import_module(MODULE)

    def test_rejects_blank_model_and_duplicate_modes(self) -> None:
        with self.assertRaises(SystemExit):
            self.module.main(["--db", "unused.db", "--live", "--model", "   "])
        with self.assertRaises(SystemExit):
            self.module.main(["--db", "unused.db", "--live", "--model", "custom", "--mode", "insight", "--mode", "insight"])

    def test_requires_deepseek_without_an_explicit_model(self) -> None:
        with patch.object(self.module, "BACKEND", "gemini"):
            with self.assertRaises(SystemExit):
                self.module.main(["--db", "unused.db", "--live"])

    def test_labels_explicit_model_as_custom_and_writes_requested_artifact(self) -> None:
        output = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        output.close()
        try:
            with (
                patch.object(self.module, "load_development_cases", return_value=()),
                patch.object(self.module, "evaluate_source_modes", return_value={"metrics": {}}),
            ):
                self.module.main(["--db", "unused.db", "--live", "--model", "custom-model", "--mode", "insight", "--json-output", output.name])
            payload = json.loads(Path(output.name).read_text())
            self.assertEqual(payload["model"]["variant"], "custom")
            self.assertEqual(payload["model"]["model"], "custom-model")
        finally:
            Path(output.name).unlink()

    def test_records_provider_config_and_prompt_version_metadata(self) -> None:
        output = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        output.close()
        try:
            with (
                patch.object(self.module, "load_development_cases", return_value=()),
                patch.object(self.module, "evaluate_source_modes", return_value={"metrics": {}}),
                patch.object(self.module, "BACKEND", "deepseek"),
            ):
                self.module.main(["--db", "unused.db", "--live", "--model", "custom-model", "--mode", "insight", "--json-output", output.name])
            payload = json.loads(Path(output.name).read_text())
            model = payload["model"]
            self.assertEqual(model["provider"], "deepseek")
            self.assertEqual(model["thinking"], "disabled")
            self.assertEqual(model["max_tokens"], 512)
            self.assertEqual(model["prompt_version"], self.module.SPAN_FIRST_PROMPT_VERSION)
            self.assertEqual(model["variant"], "custom")
            self.assertEqual(payload["config"]["modes"], ["insight"])
            self.assertEqual(payload["config"]["fixture"], "data/relation_extraction_phase2d_dev_v0.json")
            self.assertEqual(payload["config"]["live"], True)
        finally:
            Path(output.name).unlink()

    def test_artifact_hash_is_deterministic_and_excludes_itself(self) -> None:
        first_output = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        first_output.close()
        second_output = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        second_output.close()
        try:
            args = ["--db", "unused.db", "--live", "--model", "custom-model", "--mode", "insight"]
            with (
                patch.object(self.module, "load_development_cases", return_value=()),
                patch.object(self.module, "evaluate_source_modes", return_value={"metrics": {}}),
            ):
                self.module.main(args + ["--json-output", first_output.name])
                self.module.main(args + ["--json-output", second_output.name])
            first = json.loads(Path(first_output.name).read_text())
            second = json.loads(Path(second_output.name).read_text())
            self.assertEqual(first["content_sha256"], second["content_sha256"])
            without_hash = {key: value for key, value in first.items() if key != "content_sha256"}
            recomputed = hashlib.sha256(json.dumps(without_hash, indent=2, sort_keys=True).encode("utf-8")).hexdigest()
            self.assertEqual(recomputed, first["content_sha256"])
            self.assertNotIn("content_sha256", json.dumps(without_hash))
            self.assertNotIn("content_sha256", str(without_hash))
        finally:
            Path(first_output.name).unlink()
            Path(second_output.name).unlink()

    def test_cli_always_passes_default_trusted_held_out_fixture(self) -> None:
        with (
            patch.object(self.module, "load_development_cases", return_value=()) as loader,
            patch.object(self.module, "evaluate_source_modes", return_value={"metrics": {}}),
            patch.object(self.module, "BACKEND", "deepseek"),
        ):
            self.module.main(["--db", "unused.db", "--live", "--model", "custom-model", "--mode", "insight"])
        kwargs = loader.call_args.kwargs
        self.assertEqual(kwargs["held_out_path"], self.module.DEFAULT_HELD_OUT_FIXTURE)

    def test_cli_fails_closed_on_development_fixture_contract_violation(self) -> None:
        fixture = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        label = {
            "subject_source": "a", "predicate_source": "b", "effect_source": "c",
            "semantic_field_token_groups": {"subject": [["a"]], "predicate": [["b"]], "effect": [["c"]]},
        }
        fixture.write(json.dumps({"cases": [{
            "id": "multi", "insight_id": "cli-dev-multi", "source_video_id": "cli-v-multi",
            "eligible": True, "expected_propositions": [label, label],
        }]}))
        fixture.close()
        try:
            with self.assertRaisesRegex(ValueError, "exactly one expected proposition"):
                self.module.main([
                    "--db", "unused.db", "--live", "--model", "custom",
                    "--fixture", fixture.name, "--mode", "insight",
                ])
        finally:
            Path(fixture.name).unlink()

    def test_cli_fails_closed_on_malformed_held_out_fixture(self) -> None:
        held = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        held.write('{"cases":"not-a-list"}')
        held.close()
        fixture = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        fixture.write(json.dumps({"cases": [{
            "id": "ok", "insight_id": "cli-dev-ok", "source_video_id": "cli-v-ok",
            "eligible": False, "expected_propositions": [],
        }]}))
        fixture.close()
        try:
            with patch.object(self.module, "DEFAULT_HELD_OUT_FIXTURE", Path(held.name)):
                with self.assertRaisesRegex(ValueError, "frozen held-out fixture"):
                    self.module.main([
                        "--db", "unused.db", "--live", "--model", "custom",
                        "--fixture", fixture.name, "--mode", "insight",
                    ])
        finally:
            Path(held.name).unlink()
            Path(fixture.name).unlink()



if __name__ == "__main__":
    unittest.main()
