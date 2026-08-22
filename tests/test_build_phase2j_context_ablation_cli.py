"""CLI tests for the Phase 2J source-grounded build/finalize/gate scripts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import scripts.build_phase2j_context_ablation as cli
from pipeline.phase2j_context_ablation import (
    CONDITION_CODES,
    OUTPUT_FILENAMES,
    canonical_sha256,
    load_json_strict,
)
from tests._phase2j_context_ablation_helpers import (
    build_fixture,
    make_completed_reviews,
    make_valid_output,
)


def _write_outputs_bundle(output_dir: Path, root: Path) -> Path:
    payloads = load_json_strict(
        output_dir / OUTPUT_FILENAMES["payloads"], label="payloads",
    )
    outputs = []
    for payload_case in payloads["cases"]:
        outputs.append({
            "case_id": payload_case["case_id"],
            "A": make_valid_output(
                payload_case["A"],
                case_id=payload_case["case_id"],
                condition="A",
            ),
            "B": make_valid_output(
                payload_case["B"],
                case_id=payload_case["case_id"],
                condition="B",
            ),
        })
    bundle = {
        "schema_version": "phase2j-context-ablation-extraction-outputs-v2",
        "purpose": "synthetic outputs",
        "release_gate": "LOCKED",
        "payloads_sha256": payloads["content_sha256"],
        "instructions_sha256": payloads["instructions_sha256"],
        "cases": outputs,
    }
    bundle = {
        **bundle,
        "content_sha256": canonical_sha256(bundle),
    }
    outputs_path = root / "outputs.json"
    outputs_path.write_text(json.dumps(bundle), encoding="utf-8")
    return outputs_path


def _material_reviews(output_dir: Path) -> dict[str, Any]:
    packet = load_json_strict(
        output_dir / OUTPUT_FILENAMES["human_packet"], label="packet",
    )
    mapping = load_json_strict(
        output_dir / OUTPUT_FILENAMES["human_mapping"], label="mapping",
    )
    by_condition: dict[str, dict[str, list[str]]] = {
        code: {f"p2ja:case:{rank:04d}": [] for rank in range(1, 11)}
        for code in CONDITION_CODES
    }
    for review_item_id, entry in mapping["entries"].items():
        by_condition[entry["condition_code"]][entry["case_id"]].append(
            review_item_id,
        )
    a = [8, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    b = [0, 8, 8, 8, 8, 0, 0, 0, 0, 0]
    success_by_item = {}
    for rank in range(1, 11):
        case_id = f"p2ja:case:{rank:04d}"
        for index, item_id in enumerate(by_condition["A"][case_id]):
            success_by_item[item_id] = index < a[rank - 1]
        for index, item_id in enumerate(by_condition["B"][case_id]):
            success_by_item[item_id] = index < b[rank - 1]
    return make_completed_reviews(
        packet, success_by_item=success_by_item,
    )


class Phase2JContextAblationCliTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.manifest_path, self.packet_path, self.db_path, self.transcripts = (
            build_fixture(self.root)
        )
        self.output_dir = self.root / "out"

    def tearDown(self):
        self._temporary.cleanup()

    def _base_args(self, command: str) -> list[str]:
        return [
            command,
            "--manifest", str(self.manifest_path),
            "--reviewed-packet", str(self.packet_path),
            "--db", str(self.db_path),
            "--output-dir", str(self.output_dir),
        ]

    def _assert_no_temp_files(self) -> None:
        leftovers = [
            path for path in self.output_dir.iterdir()
            if ".tmp-" in path.name
        ]
        self.assertEqual(leftovers, [])

    def test_default_build_reaches_ready_for_sol_with_payloads(self):
        code = cli.main(self._base_args("build"))
        self.assertEqual(code, 0)
        self._assert_no_temp_files()
        selection = load_json_strict(
            self.output_dir / OUTPUT_FILENAMES["selection"], label="selection",
        )
        instructions = load_json_strict(
            self.output_dir / OUTPUT_FILENAMES["instructions"],
            label="instructions",
        )
        payloads = load_json_strict(
            self.output_dir / OUTPUT_FILENAMES["payloads"], label="payloads",
        )
        summary = load_json_strict(
            self.output_dir / OUTPUT_FILENAMES["build_summary"],
            label="build summary",
        )
        self.assertEqual(len(selection["cases"]), 10)
        self.assertEqual(len(payloads["cases"]), 10)
        self.assertIn("supporting_source_ranges", instructions["fields"])
        self.assertEqual(summary["mode"], "ready_for_sol")
        self.assertEqual(
            cli.main(self._base_args("validate")),
            0,
        )

    def test_build_with_outputs_generates_blinded_review_packet(self):
        self.assertEqual(cli.main(self._base_args("build")), 0)
        outputs_path = _write_outputs_bundle(self.output_dir, self.root)
        args = self._base_args("build") + [
            "--outputs", str(outputs_path),
        ]
        self.assertEqual(cli.main(args), 0)
        self._assert_no_temp_files()
        packet = load_json_strict(
            self.output_dir / OUTPUT_FILENAMES["human_packet"], label="packet",
        )
        self.assertEqual(len(packet["review_items"]), 160)
        self.assertEqual(len(packet["source_evidence"]), 10)
        self.assertNotIn('"seed"', json.dumps(packet))
        mapping = load_json_strict(
            self.output_dir / OUTPUT_FILENAMES["human_mapping"], label="mapping",
        )
        self.assertEqual(
            mapping["content_sha256"],
            packet["blinding"]["mapping_sha256"],
        )
        self.assertEqual(cli.main(self._base_args("validate")), 0)

    def test_build_rejects_missing_inputs(self):
        code = cli.main([
            "build",
            "--manifest", str(self.root / "missing.json"),
            "--reviewed-packet", str(self.packet_path),
            "--db", str(self.db_path),
            "--output-dir", str(self.output_dir),
        ])
        self.assertEqual(code, 1)

    def test_validate_rejects_incomplete_directory(self):
        self.assertEqual(cli.main(self._base_args("build")), 0)
        (self.output_dir / OUTPUT_FILENAMES["payloads"]).unlink()
        self.assertEqual(cli.main(self._base_args("validate")), 1)

    def test_finalize_freeze_material_summary(self):
        self.assertEqual(cli.main(self._base_args("build")), 0)
        outputs_path = _write_outputs_bundle(self.output_dir, self.root)
        self.assertEqual(
            cli.main(self._base_args("build") + [
                "--outputs", str(outputs_path),
            ]),
            0,
        )
        reviews = _material_reviews(self.output_dir)
        reviews_path = self.root / "reviews.json"
        reviews_path.write_text(json.dumps(reviews), encoding="utf-8")
        self.assertEqual(
            cli.main(self._base_args("finalize") + [
                "--reviews", str(reviews_path),
                "--frozen-at", "2026-08-20T00:00:00Z",
            ]),
            0,
        )
        summary = load_json_strict(
            self.output_dir / OUTPUT_FILENAMES["materiality_summary"],
            label="materiality summary",
        )
        self.assertEqual(summary["decision"], "MATERIAL")
        self.assertEqual(summary["release_gate"], "LOCKED")
        self.assertEqual(cli.main(self._base_args("validate")), 0)

    def test_finalize_rejects_incomplete_reviews(self):
        self.assertEqual(cli.main(self._base_args("build")), 0)
        outputs_path = _write_outputs_bundle(self.output_dir, self.root)
        self.assertEqual(
            cli.main(self._base_args("build") + [
                "--outputs", str(outputs_path),
            ]),
            0,
        )
        reviews_path = self.root / "reviews.json"
        reviews_path.write_text(json.dumps({
            "schema_version": "phase2j-context-ablation-completed-reviews-v2",
            "reviewer_kind": "human",
            "human_review_attested": True,
            "attestation_statement": "I attest to this review.",
            "reviewer": "tester",
            "completed_at": "2026-08-20T00:00:00Z",
            "reviews": {},
            "content_sha256": canonical_sha256({
                "schema_version": "phase2j-context-ablation-completed-reviews-v2",
                "reviewer_kind": "human",
                "human_review_attested": True,
                "attestation_statement": "I attest to this review.",
                "reviewer": "tester",
                "completed_at": "2026-08-20T00:00:00Z",
                "reviews": {},
            }),
        }), encoding="utf-8")
        self.assertEqual(
            cli.main(self._base_args("finalize") + [
                "--reviews", str(reviews_path),
                "--frozen-at", "2026-08-20T00:00:00Z",
            ]),
            1,
        )

    def test_finalize_rejects_non_human_attestation(self):
        self.assertEqual(cli.main(self._base_args("build")), 0)
        outputs_path = _write_outputs_bundle(self.output_dir, self.root)
        self.assertEqual(
            cli.main(self._base_args("build") + [
                "--outputs", str(outputs_path),
            ]),
            0,
        )
        reviews = _material_reviews(self.output_dir)
        reviews["reviewer_kind"] = "bot"
        reviews["content_sha256"] = canonical_sha256({
            key: value for key, value in reviews.items()
            if key != "content_sha256"
        })
        reviews_path = self.root / "reviews.json"
        reviews_path.write_text(json.dumps(reviews), encoding="utf-8")
        self.assertEqual(
            cli.main(self._base_args("finalize") + [
                "--reviews", str(reviews_path),
                "--frozen-at", "2026-08-20T00:00:00Z",
            ]),
            1,
        )

    def test_deepseek_emit_and_import_gate(self):
        # Gate-locked: no frozen summary yet.
        self.assertEqual(cli.main(self._base_args("emit-deepseek-run")), 1)
        self.assertEqual(cli.main(self._base_args("build")), 0)
        outputs_path = _write_outputs_bundle(self.output_dir, self.root)
        self.assertEqual(
            cli.main(self._base_args("build") + [
                "--outputs", str(outputs_path),
            ]),
            0,
        )
        reviews_path = self.root / "reviews.json"
        reviews_path.write_text(json.dumps(
            _material_reviews(self.output_dir),
        ), encoding="utf-8")
        self.assertEqual(
            cli.main(self._base_args("finalize") + [
                "--reviews", str(reviews_path),
                "--frozen-at", "2026-08-20T00:00:00Z",
            ]),
            0,
        )
        self.assertEqual(cli.main(self._base_args("emit-deepseek-run")), 0)
        self._assert_no_temp_files()
        run_packet = load_json_strict(
            self.output_dir / OUTPUT_FILENAMES["deepseek_run"],
            label="deepseek run",
        )
        self.assertEqual(run_packet["condition"], "B")
        payloads = load_json_strict(
            self.output_dir / OUTPUT_FILENAMES["payloads"], label="payloads",
        )
        deepseek_outputs = {
            payload_case["case_id"]: make_valid_output(
                payload_case["B"],
                case_id=payload_case["case_id"],
                condition="B",
            )
            for payload_case in payloads["cases"]
        }
        deepseek_path = self.root / "deepseek-outputs.json"
        deepseek_path.write_text(json.dumps({
            "cases": deepseek_outputs,
        }), encoding="utf-8")
        self.assertEqual(
            cli.main(self._base_args("import-deepseek-run") + [
                "--run-packet", str(
                    self.output_dir / OUTPUT_FILENAMES["deepseek_run"],
                ),
                "--outputs", str(deepseek_path),
            ]),
            0,
        )
        self.assertTrue(
            (self.output_dir / OUTPUT_FILENAMES["deepseek_import"]).is_file(),
        )
        bad = dict(deepseek_outputs)
        first_case = payloads["cases"][0]["case_id"]
        bad[first_case] = make_valid_output(
            payloads["cases"][0]["A"],
            case_id=first_case,
            condition="A",
        )
        bad_path = self.root / "deepseek-bad.json"
        bad_path.write_text(json.dumps({"cases": bad}), encoding="utf-8")
        self.assertEqual(
            cli.main(self._base_args("import-deepseek-run") + [
                "--run-packet", str(
                    self.output_dir / OUTPUT_FILENAMES["deepseek_run"],
                ),
                "--outputs", str(bad_path),
            ]),
            1,
        )

    def test_fabricated_material_summary_cannot_unlock_deepseek(self):
        self.assertEqual(cli.main(self._base_args("build")), 0)
        outputs_path = _write_outputs_bundle(self.output_dir, self.root)
        self.assertEqual(
            cli.main(self._base_args("build") + [
                "--outputs", str(outputs_path),
            ]),
            0,
        )
        # Fabricate a self-consistent-looking MATERIAL summary that is not
        # bound to the real artifact chain.
        fabricated = {
            "schema_version": "phase2j-context-ablation-materiality-summary-v2",
            "purpose": "fabricated",
            "release_gate": "LOCKED",
            "decision": "MATERIAL",
            "frozen_at": "2026-08-20T00:00:00Z",
            "preregistered_policy": {},
            "input_hashes": {},
            "materiality": {"decision": "MATERIAL"},
        }
        fabricated = {
            **fabricated,
            "content_sha256": canonical_sha256(fabricated),
        }
        summary_path = self.output_dir / OUTPUT_FILENAMES["materiality_summary"]
        summary_path.write_text(json.dumps(fabricated), encoding="utf-8")
        self.assertEqual(cli.main(self._base_args("validate")), 1)
        self.assertEqual(cli.main(self._base_args("emit-deepseek-run")), 1)
        self.assertFalse(
            (self.output_dir / OUTPUT_FILENAMES["deepseek_run"]).exists(),
        )

    def test_tampered_finalized_chain_blocks_emit(self):
        self.assertEqual(cli.main(self._base_args("build")), 0)
        outputs_path = _write_outputs_bundle(self.output_dir, self.root)
        self.assertEqual(
            cli.main(self._base_args("build") + [
                "--outputs", str(outputs_path),
            ]),
            0,
        )
        reviews_path = self.root / "reviews.json"
        reviews_path.write_text(json.dumps(
            _material_reviews(self.output_dir),
        ), encoding="utf-8")
        self.assertEqual(
            cli.main(self._base_args("finalize") + [
                "--reviews", str(reviews_path),
                "--frozen-at", "2026-08-20T00:00:00Z",
            ]),
            0,
        )
        summary_path = self.output_dir / OUTPUT_FILENAMES["materiality_summary"]
        summary = load_json_strict(summary_path, label="materiality summary")
        tampered = json.loads(json.dumps(summary))
        tampered["decision"] = "NOT_MATERIAL"
        tampered["content_sha256"] = canonical_sha256({
            key: value for key, value in tampered.items()
            if key != "content_sha256"
        })
        summary_path.write_text(json.dumps(tampered), encoding="utf-8")
        self.assertEqual(cli.main(self._base_args("emit-deepseek-run")), 1)


if __name__ == "__main__":
    unittest.main()
