"""Focused contract tests for the finished Phase 2K migration.

The older behavior tests in ``test_phase2k_contextual_reconstruction.py``
remain valid; this file exercises the Notion-alignment contracts that were
still unfinished at the handoff boundary: strict Pass 1 separation, separate
reconstruction/polish passes, transformation audits, lineage/raw-response
integrity, closeout gating, and failure bookkeeping.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import scripts.finalize_phase2k_human_review as finalize_cli
from pipeline.phase2k_downstream_comparison import (
    COMPARISON_METRIC_NAMES,
    DISCRIMINATIVE_ARCHITECTURE_FAMILY,
    DOWNSTREAM_COMPARISON_SCHEMA_VERSION,
    DOWNSTREAM_DIAGNOSIS_VALUES,
    GENERATIVE_ARCHITECTURE_FAMILY,
    build_downstream_comparison,
)
from pipeline.phase2k_contextual_reconstruction import (
    AUDIT_ERROR_TAXONOMY,
    AUDIT_OPERATION_DECISIONS,
    COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION,
    FINAL_CLOSEOUT_STATUSES,
    HUMAN_SCORE_FIELDS,
    NOT_APPLICABLE,
    POLISH_MAX_CORRECTIONS,
    ProviderCorrectionExhausted,
    RELEASE_GATE_REVIEWED,
    TRANSFORMATION_AUDIT_SCHEMA_VERSION,
    build_closeout_status,
    build_count_report_skeleton,
    build_metadata_adapter,
    build_record_c,
    build_transformation_audit,
    canonical_sha256,
    import_completed_human_reviews,
    load_json_strict,
    run_mechanical_cleanup,
    run_polish,
    run_reconstruction,
    summarize_human_reviews,
    summarize_transformation_audits,
    supplied_facts,
    text_sha256,
    validate_completed_transformation_audits,
    validate_downstream_comparison,
    validate_output_directory,
    validate_transformation_audit_packet,
    _render_context_presentation,
)
from tests._phase2k_helpers import build_fixture
from tests.test_phase2k_contextual_reconstruction import (
    TEST_LIVE_INFERENCE_CONFIG,
    CountingChat,
    _bronze_span,
    _live_factory_chat,
    _mechanical_raw,
    _polish_raw,
    _reconstruction_raw,
    _selected,
    _slots,
    _sufficiency_raw,
    _uncertainty,
)


class Phase2KContractTests(unittest.TestCase):
    def test_mechanical_envelope_and_recursive_semantic_rejection(self):
        selected = _selected("He hit R.", champion="Lux")
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([_mechanical_raw(selected)]),
            config_hash=canonical_sha256({"v": 1}),
        )
        # The provider envelope is compact; full provenance is sealed by the
        # harness, not echoed by the model.
        raw = _mechanical_raw(selected)
        parsed = json.loads(raw)
        self.assertEqual(
            set(parsed),
            {"schema_version", "clean_text", "repairs", "uncertainties", "rationale"},
        )
        provenance = result["provenance"]
        self.assertEqual(provenance["task_kind"], "TEXT_RESTORATION")
        target = provenance["target"]
        for key in (
            "window_id", "source_group_id", "canonical_record_sha256",
            "upstream_start", "upstream_end", "upstream_content_sha256",
            "bronze_text", "bronze_text_sha256",
        ):
            self.assertIn(key, target)
        for semantic_field in (
            "entities", "champion_binding", "ability_owner", "events",
        ):
            bad = json.loads(raw)
            bad["repairs"] = [{
                "original_text": "He",
                "replacement": "H",
                "repair_type": "SPELLING",
                "confidence": "HIGH",
                "rationale": "prohibited semantic field",
                semantic_field: "prohibited",
            }]
            with self.assertRaises(ProviderCorrectionExhausted) as caught:
                run_mechanical_cleanup(
                    selected,
                    chat=CountingChat([
                        json.dumps(bad), json.dumps(bad), json.dumps(bad),
                        json.dumps(bad),
                    ]),
                    config_hash=canonical_sha256({"v": 1}),
                )
            self.assertIn(
                "semantic extraction field",
                caught.exception.attempts[0]["error"],
            )
        # A semantic field nested inside an uncertainty proposal is also
        # rejected fail-closed.
        nested = json.loads(raw)
        nested["uncertainties"] = [{
            "surface_text": "He",
            "uncertainty_type": "ASR_ALTERNATIVES",
            "alternatives": [{"text": "he", "confidence": "LOW"}],
            "note": "prohibited",
            "semantic": {"entity": "Lux"},
        }]
        with self.assertRaises(ProviderCorrectionExhausted) as caught:
            run_mechanical_cleanup(
                selected,
                chat=CountingChat([
                    json.dumps(nested), json.dumps(nested), json.dumps(nested),
                    json.dumps(nested),
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        self.assertIn(
            "semantic extraction field",
            caught.exception.attempts[0]["error"],
        )
        # The normalized envelope never carries semantic keys.
        self.assertNotIn("entities", result["repairs"])
        self.assertEqual(result["provenance"]["task_kind"], "TEXT_RESTORATION")

    def test_hs_uncertainty_is_retained_exactly(self):
        text = "here you are dead and have to flash if HS one more"
        selected = _selected(text, champion="Viktor")
        start = text.index("HS")
        raw = _mechanical_raw(
            selected,
            uncertainties=[_uncertainty(
                selected,
                local_start=start,
                local_end=start + 2,
                alternatives=[
                    {"text": "his", "confidence": "MEDIUM"},
                    {"text": "has", "confidence": "MEDIUM"},
                    {"text": "HS", "confidence": "LOW"},
                ],
            )],
        )
        result = run_mechanical_cleanup(
            selected,
            chat=CountingChat([raw]),
            config_hash=canonical_sha256({"v": 1}),
        )
        self.assertEqual(result["mechanical_cleaned_text"], text)
        self.assertEqual(result["repairs"], [])
        self.assertEqual(result["uncertainties"][0]["text"], "HS")
        self.assertEqual(
            [item["text"] for item in result["uncertainties"][0]["alternatives"]],
            ["his", "has", "HS"],
        )

    def test_reconstruction_and_polish_are_separate_and_reject_orphans(self):
        text = "He hit R. More context."
        selected = _selected("He hit R.", champion="Lux")
        context = {
            "schema_version": "phase2k-context-v1",
            "context_id": "p2k:ctx:w:r2",
            "window_id": selected["window_id"],
            "source_group_id": selected["source_group_id"],
            "radius": "r2",
            "target": {
                "window_id": selected["window_id"],
                "source_absolute_start": 0,
                "source_absolute_end": len(selected["source_text"]),
                "text": selected["source_text"],
                "text_sha256": text_sha256(selected["source_text"]),
                "char_length": len(selected["source_text"]),
            },
            "requested": {"previous_segments": 1, "following_segments": 1},
            "actual": {},
            "segments": [
                {
                    "segment_id": "target:test",
                    "segment_ordinal": None,
                    "kind": "target",
                    "is_partial": False,
                    "source_absolute_start": 0,
                    "source_absolute_end": len(selected["source_text"]),
                    "text": selected["source_text"],
                },
            ],
            "previous_stop_reason": "SOURCE_BOUNDARY",
            "following_stop_reason": "SOURCE_BOUNDARY",
            "stop_reason": "SOURCE_BOUNDARY",
            "source_boundaries": {
                "context_start": 0,
                "context_end": len(selected["source_text"]),
            },
            "content_sha256": canonical_sha256({
                "segments": [{
                    "segment_id": "target:test",
                    "segment_ordinal": None,
                    "kind": "target",
                    "is_partial": False,
                    "source_absolute_start": 0,
                    "source_absolute_end": len(selected["source_text"]),
                    "text": selected["source_text"],
                }],
                "target": {
                    "window_id": selected["window_id"],
                    "source_absolute_start": 0,
                    "source_absolute_end": len(selected["source_text"]),
                    "text": selected["source_text"],
                    "text_sha256": text_sha256(selected["source_text"]),
                    "char_length": len(selected["source_text"]),
                },
            }),
        }
        diagnostic = {
            "decision": "SUFFICIENT",
            "response": {
                "parsed": {
                    "slots": _slots(decision="SUFFICIENT"),
                    "metadata_conflicts": [],
                },
            },
        }
        base = selected["upstream_start"]
        valid = _reconstruction_raw(
            cleaned=selected["source_text"],
            bronze=selected["source_text"],
            base_offset=base,
            selected=selected,
        )
        combined = json.loads(valid)
        combined["paraphrase_text"] = "Lux hits R"
        with self.assertRaises(ValueError):
            run_reconstruction(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=selected["source_text"],
                final_diagnostic=diagnostic,
                chat=CountingChat([
                    json.dumps(combined),
                    json.dumps(combined),
                    json.dumps(combined),
                    json.dumps(combined),
                ]),
                config_hash=canonical_sha256({"v": 1}),
            )
        reconstruction = run_reconstruction(
            selected,
            transcript=text,
            context=context,
            mechanical_cleaned_text=selected["source_text"],
            final_diagnostic=diagnostic,
            chat=CountingChat([valid]),
            config_hash=canonical_sha256({"v": 1}),
        )
        orphan = _polish_raw(
            selected,
            reconstruction,
            statements=[{
                "text": selected["source_text"],
                "modality_preserved": True,
                "negation_preserved": True,
                "uncertainty_preserved": True,
                "evidence_quotes": [selected["source_text"]],
                "reconstruction_operation_ids": ["not-in-reconstruction"],
                "support_mode": "RECONSTRUCTION_DERIVED",
                "unchanged_source_quote": None,
            }],
        )
        with self.assertRaises(ValueError):
            run_polish(
                selected,
                transcript=text,
                context=context,
                mechanical_cleaned_text=selected["source_text"],
                reconstruction=reconstruction,
                chat=CountingChat([orphan, orphan, orphan, orphan]),
                config_hash=canonical_sha256({"v": 1}),
            )

    def test_c_presentation_uses_mechanical_target_without_losing_bronze(self):
        text = "before. he hit R. after."
        selected = _selected(text, champion="Lux", target="he hit R.")
        start = text.index("he hit R.")
        mechanical = run_mechanical_cleanup(
            selected,
            chat=CountingChat([_mechanical_raw(
                selected,
                repairs=[{
                    "repair_id": "cap",
                    "target_local_start": 0,
                    "target_local_end": 2,
                    "original_text": "he",
                    "replacement": "He",
                    "repair_type": "CAPITALIZATION",
                    "confidence": "HIGH",
                    "evidence_spans": [],
                    "rationale": "start of target",
                }],
            )]),
            config_hash=canonical_sha256({"v": 1}),
        )
        context = {
            "segments": [
                {
                    "segment_id": "seg:prev",
                    "segment_ordinal": 1,
                    "kind": "previous",
                    "is_partial": False,
                    "source_absolute_start": 0,
                    "source_absolute_end": len("before."),
                    "text": "before.",
                },
                {
                    "segment_id": "target:test",
                    "segment_ordinal": None,
                    "kind": "target",
                    "is_partial": False,
                    "source_absolute_start": start,
                    "source_absolute_end": start + len("he hit R."),
                    "text": "he hit R.",
                },
                {
                    "segment_id": "seg:after",
                    "segment_ordinal": 2,
                    "kind": "following",
                    "is_partial": False,
                    "source_absolute_start": start + len("he hit R.") + 1,
                    "source_absolute_end": len(text),
                    "text": "after.",
                },
            ],
        }
        record = build_record_c(
            selected,
            context,
            mechanical["mechanical_cleaned_text"],
        )
        rendered = _render_context_presentation(
            context, mechanical["mechanical_cleaned_text"],
        )
        self.assertIn("before.", rendered)
        self.assertIn("⟪TARGET⟫He hit R.⟪/TARGET⟫", rendered)
        self.assertNotIn("he hit R.", rendered.replace("He hit R.", ""))
        self.assertEqual(
            record["content"]["presentation_target"]["text"],
            "He hit R.",
        )
        self.assertEqual(
            record["content"]["presentation_target"]["bronze_target_sha256"],
            text_sha256(selected["source_text"]),
        )
        self.assertEqual(record["target"]["text"], selected["source_text"])

    def test_metadata_is_field_level_provenance_and_title_is_not_inference(self):
        selected = _selected("He hit R.", champion="Lux")
        adapter = build_metadata_adapter(selected)
        self.assertFalse(adapter["video_title"]["inference_allowed"])
        self.assertTrue(adapter["champion"]["inference_allowed"])
        self.assertNotIn("video_title", supplied_facts(adapter))
        self.assertIn("champion", supplied_facts(adapter))
        self.assertNotIn("Video Lux", supplied_facts(adapter))
        from pipeline.phase2k_contextual_reconstruction import build_mechanical_prompt
        _, user = build_mechanical_prompt(selected)
        payload = json.loads(user)
        self.assertTrue(
            payload["metadata_policy"]["video_title_is_provenance_only"],
        )
        self.assertNotIn("video_title", payload["supplied_facts"])

    def test_raw_response_tamper_missing_and_orphan_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            output = root / "live"
            result = build_live(root, manifest_path, packet_path, db_path, output)
            records = load_json_strict(
                output / "phase2k-reconstruction-records-v7.json",
                label="records",
            )
            raw_file_name = next(
                record["content"]["model_call"]["raw_response_path"]
                for record in records["records"]
                if record["record_type"] == "B"
                and record["content"].get("model_call") is not None
            )
            raw_file = output / "raw_responses" / raw_file_name
            pristine = raw_file.read_text(encoding="utf-8")

            raw_file.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_output_directory(
                    output_dir=output,
                    manifest_path=manifest_path,
                    packet_path=packet_path,
                    db_path=db_path,
                )
            raw_file.write_text(pristine, encoding="utf-8")

            raw_file.unlink()
            with self.assertRaises(ValueError):
                validate_output_directory(
                    output_dir=output,
                    manifest_path=manifest_path,
                    packet_path=packet_path,
                    db_path=db_path,
                )
            raw_file.write_text(pristine, encoding="utf-8")

            orphan = output / "raw_responses" / ("0" * 64 + ".txt")
            orphan.write_text("orphan", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_output_directory(
                    output_dir=output,
                    manifest_path=manifest_path,
                    packet_path=packet_path,
                    db_path=db_path,
                )
            orphan.unlink()
            validate_output_directory(
                output_dir=output,
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
            )

    def test_transformation_audit_blank_first_failure_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            output = root / "live"
            chat, _ = _failing_reconstruction_chat()
            build_live(
                root, manifest_path, packet_path, db_path, output,
                chat=chat,
            )
            records = load_json_strict(
                output / "phase2k-reconstruction-records-v7.json",
                label="records",
            )
            audit = load_json_strict(
                output / "phase2k-transformation-audit-packet-v2.json",
                label="audit",
            )
            validate_transformation_audit_packet(audit, records_obj=records)
            self.assertEqual(
                audit["schema_version"], TRANSFORMATION_AUDIT_SCHEMA_VERSION,
            )
            self.assertEqual(audit["release_gate"], "AWAITING_HUMAN_REVIEW")
            self.assertEqual(len(audit["window_audits"]), 30)
            self.assertTrue(all(
                window["first_failure"] is not None
                and window["first_failure"]["stage"] == "reconstruction"
                and window["first_reconstruction_failure"]
                == window["first_failure"]
                for window in audit["window_audits"]
            ))
            for window in audit["window_audits"]:
                for category in (
                    "mechanical_repairs", "contextual_repairs",
                    "entity_bindings", "pronoun_bindings",
                    "reference_bindings", "ability_bindings",
                    "polished_statements",
                ):
                    for operation in window["operations"][category]:
                        self.assertIsNone(operation["decision"])

    def test_completed_audit_validation_and_synthetic_metrics(self):
        template, records = synthetic_audit_fixture()
        validate_transformation_audit_packet(template, records_obj=records)
        completed = copy.deepcopy(template)
        completed["schema_version"] = COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION
        completed["release_gate"] = RELEASE_GATE_REVIEWED
        for window in completed["window_audits"]:
            for repair in window["operations"]["mechanical_repairs"]:
                repair["decision"] = "APPROVE"
                repair["error_taxonomy"] = None
            for repair in window["operations"]["contextual_repairs"]:
                repair["decision"] = "APPROVE"
                repair["error_taxonomy"] = None
            for category in (
                "entity_bindings", "pronoun_bindings",
                "reference_bindings", "ability_bindings",
            ):
                for binding in window["operations"][category]:
                    binding["decision"] = "APPROVE"
                    binding["error_taxonomy"] = None
            for statement in window["operations"]["polished_statements"]:
                statement["decision"] = "APPROVE"
                statement["supported"] = True
                statement["uncertainty_preserved"] = True
                statement["negation_preserved"] = True
                statement["modality_preserved"] = True
                statement["causality_invented"] = False
                statement["source_detail_dropped"] = False
                statement["error_taxonomy"] = None
        completed["content_sha256"] = canonical_sha256({
            key: value for key, value in completed.items()
            if key != "content_sha256"
        })
        validated = validate_completed_transformation_audits(
            template, completed, records_obj=records,
        )
        metrics = summarize_transformation_audits(validated, records_obj=records)
        self.assertEqual(metrics["asr"]["proposed"], 2)
        self.assertEqual(metrics["asr"]["approved"], 2)
        self.assertEqual(metrics["entity"]["precision"], 1.0)
        self.assertEqual(metrics["entity"]["required_resolvable_recall"], 1.0)
        self.assertEqual(metrics["ability_ownership"]["accuracy"], 1.0)
        self.assertEqual(
            metrics["polish_preservation"]["modality_preserved"], 1.0,
        )
        incomplete = copy.deepcopy(completed)
        incomplete["window_audits"][0]["operations"]["mechanical_repairs"][0][
            "decision"
        ] = None
        incomplete["content_sha256"] = canonical_sha256({
            key: value for key, value in incomplete.items()
            if key != "content_sha256"
        })
        with self.assertRaises(ValueError):
            validate_completed_transformation_audits(
                template, incomplete, records_obj=records,
            )

    @staticmethod
    def _recompute_audit_hash(audit: dict[str, Any]) -> dict[str, Any]:
        audit["content_sha256"] = canonical_sha256({
            key: value for key, value in audit.items()
            if key != "content_sha256"
        })
        return audit

    @staticmethod
    def _completed_audit_from_template(
        template: Mapping[str, Any],
    ) -> dict[str, Any]:
        completed = copy.deepcopy(dict(template))
        completed["schema_version"] = COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION
        completed["release_gate"] = RELEASE_GATE_REVIEWED
        for window in completed["window_audits"]:
            for repair in window["operations"]["mechanical_repairs"]:
                repair["decision"] = "APPROVE"
                repair["error_taxonomy"] = None
            for repair in window["operations"]["contextual_repairs"]:
                repair["decision"] = "APPROVE"
                repair["error_taxonomy"] = None
            for category in (
                "entity_bindings", "pronoun_bindings",
                "reference_bindings", "ability_bindings",
            ):
                for binding in window["operations"][category]:
                    binding["decision"] = "APPROVE"
                    binding["error_taxonomy"] = None
            for statement in window["operations"]["polished_statements"]:
                statement["decision"] = "APPROVE"
                statement["supported"] = True
                statement["uncertainty_preserved"] = True
                statement["negation_preserved"] = True
                statement["modality_preserved"] = True
                statement["causality_invented"] = False
                statement["source_detail_dropped"] = False
                statement["error_taxonomy"] = None
        return Phase2KContractTests._recompute_audit_hash(completed)

    def test_blank_audit_binding_mention_contract_rejects_malformed_spans(self):
        template, records = synthetic_audit_fixture()
        validate_transformation_audit_packet(template, records_obj=records)
        valid_mention = template["window_audits"][0]["operations"][
            "entity_bindings"
        ][0]["mention"]
        self.assertEqual(
            set(valid_mention),
            {
                "target_local_start", "target_local_end",
                "source_absolute_start", "source_absolute_end", "text",
            },
        )
        base = (
            valid_mention["source_absolute_start"]
            - valid_mention["target_local_start"]
        )
        malformed = {
            "string mention": "He",
            "missing field": {
                "target_local_start": 0,
                "target_local_end": 2,
                "source_absolute_start": base,
                "source_absolute_end": base + 2,
            },
            "extra field": dict(valid_mention, extra=True),
            "boolean offset": dict(valid_mention, target_local_start=True),
            "non-integer offset": dict(valid_mention, target_local_end=2.0),
            "negative start": dict(valid_mention, target_local_start=-1),
            "reversed offsets": dict(
                valid_mention,
                target_local_start=2,
                target_local_end=0,
                source_absolute_start=base + 2,
                source_absolute_end=base,
            ),
            "absolute start disagreement": dict(
                valid_mention,
                source_absolute_start=base + 1,
            ),
            "absolute end disagreement": dict(
                valid_mention,
                source_absolute_end=base + 1,
            ),
            "wrong text length": dict(valid_mention, text="H"),
            "outside bronze target": dict(
                valid_mention,
                target_local_end=len(
                    records["records"][0]["target"]["text"],
                ) + 1,
                source_absolute_end=(
                    base + len(records["records"][0]["target"]["text"]) + 1
                ),
            ),
        }
        for label, mention in malformed.items():
            mutated = copy.deepcopy(template)
            mutated["window_audits"][0]["operations"]["entity_bindings"][0][
                "mention"
            ] = mention
            mutated = Phase2KContractTests._recompute_audit_hash(mutated)
            with self.subTest(label=label):
                with self.assertRaises(ValueError) as caught:
                    validate_transformation_audit_packet(
                        mutated, records_obj=records,
                    )
                self.assertIn("mention", str(caught.exception))

    def test_completed_audit_rejects_string_mention_and_mention_mismatch(self):
        template, records = synthetic_audit_fixture()
        validate_transformation_audit_packet(template, records_obj=records)
        completed = self._completed_audit_from_template(template)
        validate_completed_transformation_audits(
            template, completed, records_obj=records,
        )
        string_mention = copy.deepcopy(completed)
        string_mention["window_audits"][0]["operations"]["entity_bindings"][0][
            "mention"
        ] = "He"
        string_mention = self._recompute_audit_hash(string_mention)
        with self.assertRaises(ValueError) as caught:
            validate_completed_transformation_audits(
                template, string_mention, records_obj=records,
            )
        self.assertIn("mention", str(caught.exception))
        valid_mention = template["window_audits"][0]["operations"][
            "entity_bindings"
        ][0]["mention"]
        base = (
            valid_mention["source_absolute_start"]
            - valid_mention["target_local_start"]
        )
        mismatched = copy.deepcopy(completed)
        mismatched["window_audits"][0]["operations"]["entity_bindings"][0][
            "mention"
        ] = {
            "target_local_start": 0,
            "target_local_end": 3,
            "source_absolute_start": base,
            "source_absolute_end": base + 3,
            "text": "He ",
        }
        mismatched = self._recompute_audit_hash(mismatched)
        with self.assertRaises(ValueError) as caught:
            validate_completed_transformation_audits(
                template, mismatched, records_obj=records,
            )
        self.assertIn("mention", str(caught.exception))

    def test_human_gate_declines_when_any_d_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            output = root / "live"
            chat, _ = _failing_reconstruction_chat()
            build_live(
                root, manifest_path, packet_path, db_path, output, chat=chat,
            )
            packet = load_json_strict(
                output / "phase2k-human-review-packet-v2.json", label="packet",
            )
            mapping = load_json_strict(
                output / "phase2k-human-review-mapping-v2.json", label="mapping",
            )
            records = load_json_strict(
                output / "phase2k-reconstruction-records-v7.json", label="records",
            )
            reviews = complete_reviews(packet, mapping)
            finalized = import_completed_human_reviews(
                packet,
                reviews,
                reviewer="human",
                completed_at="2026-08-19T00:00:00.000Z",
            )
            summary = summarize_human_reviews(
                finalized, mapping=mapping, records_file=records,
            )
            gate = summary["review_gate"]
            self.assertEqual(gate["status"], "FAILED")
            self.assertTrue(any(
                reason["criterion"] == "d_items_available"
                and not reason["passed"]
                for reason in gate["reasons"]
            ))

    def test_polish_exhaustion_placeholder_retains_reconstruction_and_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            output = root / "live"
            good = _live_factory_chat()

            def chat(system: str, user: str) -> str:
                payload = json.loads(user)
                if payload.get("task") in (
                    "semantic_polish", "semantic_polish_correction",
                ):
                    return "{always broken polish json"
                return good(system, user)

            build_live(
                root, manifest_path, packet_path, db_path, output, chat=chat,
            )
            records = load_json_strict(
                output / "phase2k-reconstruction-records-v7.json", label="records",
            )
            content = next(
                record["content"]
                for record in records["records"]
                if record["record_type"] == "D"
            )
            self.assertEqual(content["generation_status"], "NOT_GENERATED")
            self.assertTrue(content["is_placeholder"])
            self.assertEqual(content["failure"]["stage"], "semantic_polish")
            self.assertEqual(
                content["failure"]["attempt_count"], POLISH_MAX_CORRECTIONS + 1,
            )
            self.assertEqual(
                len(content["failure"]["attempts"]),
                POLISH_MAX_CORRECTIONS + 1,
            )
            self.assertIsNotNone(content["reconstruction"])
            self.assertIsNone(content["semantic_polish"])
            self.assertEqual(
                content["reconstruction"]["generation_status"], "GENERATED",
            )
            validate_output_directory(
                output_dir=output,
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
            )
            audit = load_json_strict(
                output / "phase2k-transformation-audit-packet-v2.json",
                label="audit",
            )
            window = audit["window_audits"][0]
            self.assertEqual(window["first_failure"]["stage"], "semantic_polish")
            self.assertIsNone(window["first_reconstruction_failure"])
            self.assertTrue(window["operations"]["pronoun_bindings"])
            self.assertEqual(window["operations"]["polished_statements"], [])

    def test_lineage_hashes_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            output = root / "phase2k"
            from pipeline.phase2k_contextual_reconstruction import (
                OUTPUT_FILENAMES,
                build_phase2k_outputs,
            )
            build_phase2k_outputs(
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
                doc_path=None,
                output_dir=output,
                mode="no_provider",
            )
            records = load_json_strict(
                output / OUTPUT_FILENAMES["records"], label="records",
            )
            summary = load_json_strict(
                output / OUTPUT_FILENAMES["build_summary"], label="summary",
            )
            lineage = records["lineage"]
            self.assertIn("repo_commit", lineage["repo"])
            self.assertIn("repo_dirty", lineage["repo"])
            self.assertTrue(lineage["implementation"]["files"])
            self.assertEqual(summary["config_hash"], records["config_hash"])
            self.assertEqual(summary["vocabulary_hash"], records["vocabulary_hash"])

    def test_closeout_gating_and_skeleton(self):
        skeleton = build_count_report_skeleton()
        self.assertIsNone(skeleton["windows"])
        waiting_human = build_closeout_status(
            human_review_complete=False,
            downstream_comparison_complete=False,
        )
        self.assertEqual(waiting_human["status"], "WAITING_FOR_HUMAN_REVIEW")
        self.assertFalse(waiting_human["inputs_complete"])
        self.assertIsNone(waiting_human["downstream_comparison"])
        gate_failed = build_closeout_status(
            human_review_complete=True,
            downstream_comparison_complete=False,
            human_review_gate_passed=False,
        )
        self.assertEqual(gate_failed["status"], "WAITING_FOR_HUMAN_REVIEW")
        self.assertFalse(gate_failed["inputs_complete"])
        with self.assertRaises(ValueError):
            build_closeout_status(
                human_review_complete=True,
                downstream_comparison_complete=True,
                closeout_decision="INCONCLUSIVE",
                downstream_comparison={"decision": "INCONCLUSIVE"},
                human_review_gate_passed=False,
            )
        waiting_downstream = build_closeout_status(
            human_review_complete=True,
            downstream_comparison_complete=False,
            human_review_gate_passed=True,
        )
        self.assertEqual(
            waiting_downstream["status"], "WAITING_FOR_DOWNSTREAM",
        )
        self.assertIsNone(waiting_downstream["downstream_comparison"])
        with self.assertRaises(ValueError):
            build_closeout_status(
                human_review_complete=True,
                downstream_comparison_complete=True,
            )
        for decision in FINAL_CLOSEOUT_STATUSES:
            closed = build_closeout_status(
                human_review_complete=True,
                downstream_comparison_complete=True,
                closeout_decision=decision,
                downstream_comparison={"decision": decision},
            )
            self.assertEqual(closed["status"], decision)
            self.assertTrue(closed["inputs_complete"])
            self.assertEqual(
                closed["downstream_comparison"]["decision"], decision,
            )
        with self.assertRaises(ValueError):
            build_closeout_status(
                human_review_complete=True,
                downstream_comparison_complete=True,
                closeout_decision="INCONCLUSIVE",
                downstream_comparison={"decision": "CONTEXT_ALONE_SUFFICIENT"},
            )
        with self.assertRaises(ValueError):
            validate_downstream_comparison(
                {
                    "schema_version": "phase2k-downstream-comparison-v1",
                    "comparison_complete": True,
                    "decision": "INCONCLUSIVE",
                    "note": "legacy v1 declaration",
                },
                label="downstream",
                records_obj={},
                finalized_packet={},
                human_summary={},
            )

    def test_finalizer_requires_completed_audits_and_downstream_for_closeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = finalize_live_state(root)
            output = state["output"]
            # Live builds refuse to finalize without a completed audit.
            code = finalize_cli.main([
                "--output-dir", str(output),
                "--reviews", str(state["reviews_path"]),
                "--reviewer", "human",
                "--completed-at", "2026-08-19T00:00:00.000Z",
            ])
            self.assertEqual(code, 1)
            downstream_path = root / "downstream.json"
            downstream_path.write_text(
                json.dumps(build_v2_comparison(state), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            code = finalize_cli.main([
                "--output-dir", str(output),
                "--reviews", str(state["reviews_path"]),
                "--audits", str(state["audit_path"]),
                "--reviewer", "human",
                "--completed-at", "2026-08-19T00:00:00.000Z",
                "--downstream-comparison", str(downstream_path),
                "--closeout-decision", "CONTEXTUAL_POLISH_VALIDATED",
            ])
            self.assertEqual(code, 0)
            closeout = load_json_strict(
                output / "phase2k-closeout-status-v2.json", label="closeout",
            )
            self.assertEqual(closeout["status"], "CONTEXTUAL_POLISH_VALIDATED")
            self.assertTrue(closeout["inputs_complete"])
            self.assertEqual(
                closeout["downstream_comparison"]["schema_version"],
                DOWNSTREAM_COMPARISON_SCHEMA_VERSION,
            )
            self.assertEqual(
                closeout["downstream_comparison"]["diagnosis"], "MIXED",
            )
            self.assertIsNotNone(closeout["count_report"]["windows"])
            validate_output_directory(
                output_dir=output,
                manifest_path=state["manifest_path"],
                packet_path=state["packet_path"],
                db_path=state["db_path"],
            )


class Phase2KDownstreamV2Tests(unittest.TestCase):
    def test_diagnosis_vocabulary_covers_required_phase2k_interpretations(self):
        self.assertTrue({
            "RAW_REPRESENTATION_BOTTLENECK",
            "GENERATIVE_FAILURE_SUBSTANTIALLY_LOSSY_INPUT",
            "INPUT_QUALITY_AND_GENERATIVE_SPARSE_DISCRIMINATION_BOTTLENECK",
            "CONTEXTUAL_POLISH_NOT_EXPLANATORY",
            "DOWNSTREAM_SEMANTIC_EXTRACTION_FAILURE_BOUNDARY",
        } <= DOWNSTREAM_DIAGNOSIS_VALUES)

    def _validate(self, comparison: Mapping[str, Any], state: Mapping[str, Any]):
        return validate_downstream_comparison(
            comparison,
            label="downstream comparison",
            records_obj=state["records"],
            finalized_packet=state["finalized"],
            human_summary=state["human_summary"],
            completed_audit=state["completed_audit"],
        )

    def test_valid_v2_passes_and_all_metrics_deltas_reconcile(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = finalize_live_state(Path(temporary))
            comparison = build_v2_comparison(state)
            validated = self._validate(comparison, state)
            self.assertEqual(
                validated["schema_version"],
                DOWNSTREAM_COMPARISON_SCHEMA_VERSION,
            )
            self.assertTrue(validated["comparison_complete"])
            self.assertEqual(
                validated["decision"], "CONTEXTUAL_POLISH_VALIDATED",
            )
            self.assertEqual(validated["diagnosis"], "MIXED")
            self.assertEqual(
                validated["dataset_binding"]["window_count"], 30,
            )
            self.assertEqual(
                validated["semantic_target_contract"]["target_count"], 60,
            )
            generative = validated["architectures"]["generative"]
            raw_rows = generative["raw"]["rows"]
            true_positive = sum(
                row["true_positive_count"] for row in raw_rows
            )
            false_positive = sum(
                row["false_positive_count"] for row in raw_rows
            )
            false_negative = sum(
                row["false_negative_count"] for row in raw_rows
            )
            self.assertEqual(
                generative["raw"]["metrics"]["precision"]["rate"],
                round(
                    true_positive / (true_positive + false_positive),
                    4,
                ),
            )
            self.assertEqual(
                generative["raw"]["metrics"]["f1"]["numerator"],
                2 * true_positive,
            )
            self.assertEqual(
                generative["raw"]["metrics"]["f1"]["denominator"],
                2 * true_positive + false_positive + false_negative,
            )
            for name in COMPARISON_METRIC_NAMES:
                raw_rate = generative["raw"]["metrics"][name]["rate"]
                polished_rate = generative["polished"]["metrics"][name]["rate"]
                expected = (
                    None
                    if raw_rate is None or polished_rate is None
                    else round(polished_rate - raw_rate, 4)
                )
                self.assertEqual(generative["deltas"][name], expected)

    def test_v1_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = finalize_live_state(Path(temporary))
            comparison = build_v2_comparison(state)
            comparison["schema_version"] = "phase2k-downstream-comparison-v1"
            rehash(comparison)
            with self.assertRaises(ValueError):
                self._validate(comparison, state)
            with self.assertRaises(ValueError):
                validate_downstream_comparison(
                    {
                        "schema_version": "phase2k-downstream-comparison-v1",
                        "comparison_complete": True,
                        "decision": "INCONCLUSIVE",
                        "note": "legacy v1 declaration",
                    },
                    label="downstream",
                    records_obj=state["records"],
                    finalized_packet=state["finalized"],
                    human_summary=state["human_summary"],
                    completed_audit=state["completed_audit"],
                )

    def test_content_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = finalize_live_state(Path(temporary))
            comparison = build_v2_comparison(state)
            comparison["content_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                self._validate(comparison, state)

    def test_dataset_binding_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = finalize_live_state(Path(temporary))
            comparison = build_v2_comparison(state)
            binding = comparison["dataset_binding"]
            for key in (
                "phase2k_records_sha256",
                "finalized_human_packet_sha256",
                "human_summary_sha256",
                "completed_transformation_audit_sha256",
                "window_ids_sha256",
                "window_count",
                "human_review_gate_status",
            ):
                with self.subTest(binding_key=key):
                    tampered = copy.deepcopy(comparison)
                    if key == "window_count":
                        tampered["dataset_binding"][key] = binding[key] + 1
                    elif key == "human_review_gate_status":
                        tampered["dataset_binding"][key] = "FAILED"
                    else:
                        tampered["dataset_binding"][key] = "0" * 64
                    rehash(tampered)
                    with self.assertRaises(ValueError):
                        self._validate(tampered, state)

    def test_missing_extra_duplicate_window_order_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = finalize_live_state(Path(temporary))
            comparison = build_v2_comparison(state)
            for name, mutate in (
                (
                    "missing",
                    lambda rows: rows.pop(0),
                ),
                (
                    "extra",
                    lambda rows: rows.append(copy.deepcopy(rows[-1])),
                ),
                (
                    "duplicate",
                    lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
                ),
                (
                    "order",
                    lambda rows: rows.__setitem__(slice(0, 2), rows[1::-1]),
                ),
            ):
                with self.subTest(mutation=name):
                    tampered = copy.deepcopy(comparison)
                    mutate(tampered["architectures"]["generative"]["raw"]["rows"])
                    rehash(tampered)
                    with self.assertRaises(ValueError):
                        self._validate(tampered, state)

    def test_raw_polished_target_counts_differ_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = finalize_live_state(Path(temporary))
            comparison = build_v2_comparison(state)
            tampered = copy.deepcopy(comparison)
            row = tampered["architectures"]["generative"]["polished"]["rows"][0]
            row.update({
                "target_count": 3,
                "true_positive_count": 2,
                "false_negative_count": 1,
                "false_positive_count": 1,
                "output_count": 3,
                "provenance_valid_count": 1,
            })
            rehash(tampered)
            with self.assertRaises(ValueError):
                self._validate(tampered, state)

    def test_count_invariants_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = finalize_live_state(Path(temporary))
            comparison = build_v2_comparison(state)
            for name, mutate in (
                (
                    "tp_plus_fn_ne_target",
                    lambda row: row.update({
                        "false_negative_count": row["false_negative_count"] + 1,
                    }),
                ),
                (
                    "tp_plus_fp_ne_output",
                    lambda row: row.update({
                        "false_positive_count": row["false_positive_count"] + 1,
                    }),
                ),
                (
                    "provenance_exceeds_output",
                    lambda row: row.update({
                        "provenance_valid_count": row["output_count"] + 1,
                    }),
                ),
                (
                    "negative_count",
                    lambda row: row.update({"true_positive_count": -1}),
                ),
            ):
                with self.subTest(invariant=name):
                    tampered = copy.deepcopy(comparison)
                    mutate(
                        tampered["architectures"]["generative"]["raw"]["rows"][0],
                    )
                    rehash(tampered)
                    with self.assertRaises(ValueError):
                        self._validate(tampered, state)

    def test_booleans_as_ints_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = finalize_live_state(Path(temporary))
            comparison = build_v2_comparison(state)
            tampered = copy.deepcopy(comparison)
            tampered["architectures"]["generative"]["raw"]["rows"][0][
                "true_positive_count"
            ] = True
            rehash(tampered)
            with self.assertRaises(ValueError):
                self._validate(tampered, state)
            tampered = copy.deepcopy(comparison)
            tampered["architectures"]["generative"]["raw"]["rows"][0][
                "abstained"
            ] = 1
            rehash(tampered)
            with self.assertRaises(ValueError):
                self._validate(tampered, state)

    def test_metric_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = finalize_live_state(Path(temporary))
            comparison = build_v2_comparison(state)
            for name in COMPARISON_METRIC_NAMES:
                for part in ("numerator", "denominator", "rate"):
                    with self.subTest(metric=name, part=part):
                        tampered = copy.deepcopy(comparison)
                        metric = tampered["architectures"]["generative"][
                            "raw"
                        ]["metrics"][name]
                        if part == "numerator":
                            metric["numerator"] += 1
                        elif part == "denominator":
                            metric["denominator"] += 1
                        else:
                            metric["rate"] = (
                                0.5
                                if metric["rate"] is None
                                else metric["rate"] + 0.1
                            )
                        rehash(tampered)
                        with self.assertRaises(ValueError):
                            self._validate(tampered, state)

    def test_delta_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = finalize_live_state(Path(temporary))
            comparison = build_v2_comparison(state)
            tampered = copy.deepcopy(comparison)
            tampered["architectures"]["generative"]["deltas"]["precision"] += 0.1
            rehash(tampered)
            with self.assertRaises(ValueError):
                self._validate(tampered, state)
            tampered = copy.deepcopy(comparison)
            tampered["architectures"]["generative"]["deltas"]["recall"] = None
            rehash(tampered)
            with self.assertRaises(ValueError):
                self._validate(tampered, state)

    def test_config_adapter_invariants_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = finalize_live_state(Path(temporary))
            comparison = build_v2_comparison(state)
            for name, mutate in (
                (
                    "same_adapter",
                    lambda arch: arch.update({
                        "polished_input_adapter_sha256": (
                            arch["raw_input_adapter_sha256"]
                        ),
                    }),
                ),
                (
                    "bad_family",
                    lambda arch: arch.update({"family": "WRONG_FAMILY"}),
                ),
                (
                    "bad_hex",
                    lambda arch: arch.update({
                        "model_or_scorer_config_sha256": "z" * 64,
                    }),
                ),
                (
                    "swapped_representation",
                    lambda arch: arch.update({
                        "raw": {
                            **arch["raw"],
                            "input_representation": "CONTEXTUAL_POLISH",
                        },
                    }),
                ),
            ):
                with self.subTest(invariant=name):
                    tampered = copy.deepcopy(comparison)
                    mutate(tampered["architectures"]["generative"])
                    rehash(tampered)
                    with self.assertRaises(ValueError):
                        self._validate(tampered, state)

    def test_review_gate_failed_prevents_downstream_and_closeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = finalize_live_state(root, reviews_factory=complete_reviews)
            output = state["output"]
            closeout = load_json_strict(
                output / "phase2k-closeout-status-v2.json", label="closeout",
            )
            self.assertEqual(closeout["status"], "WAITING_FOR_HUMAN_REVIEW")
            self.assertFalse(closeout["inputs_complete"])
            downstream_path = root / "downstream.json"
            downstream_path.write_text(
                json.dumps(build_v2_comparison(state), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            for extra in (
                ["--downstream-comparison", str(downstream_path)],
                [
                    "--downstream-comparison", str(downstream_path),
                    "--closeout-decision", "CONTEXTUAL_POLISH_VALIDATED",
                ],
            ):
                with self.subTest(extra=extra):
                    code = finalize_cli.main([
                        "--output-dir", str(output),
                        "--reviews", str(state["reviews_path"]),
                        "--audits", str(state["audit_path"]),
                        "--reviewer", "human",
                        "--completed-at", "2026-08-19T00:00:00.000Z",
                        *extra,
                    ])
                    self.assertEqual(code, 1)
            closeout = load_json_strict(
                output / "phase2k-closeout-status-v2.json", label="closeout",
            )
            self.assertEqual(closeout["status"], "WAITING_FOR_HUMAN_REVIEW")
            self.assertIsNone(closeout["downstream_comparison"])

    def test_finalizer_waiting_then_closes_with_passed_gate_and_v2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = finalize_live_state(root)
            output = state["output"]
            closeout = load_json_strict(
                output / "phase2k-closeout-status-v2.json", label="closeout",
            )
            self.assertEqual(closeout["status"], "WAITING_FOR_DOWNSTREAM")
            self.assertFalse(closeout["inputs_complete"])
            self.assertIsNone(closeout["downstream_comparison"])
            downstream_path = root / "downstream.json"
            downstream_path.write_text(
                json.dumps(
                    build_v2_comparison(
                        state, decision="CONTEXT_ALONE_SUFFICIENT",
                    ),
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
            # A closeout decision that does not match the comparison is refused.
            code = finalize_cli.main([
                "--output-dir", str(output),
                "--reviews", str(state["reviews_path"]),
                "--audits", str(state["audit_path"]),
                "--reviewer", "human",
                "--completed-at", "2026-08-19T00:00:00.000Z",
                "--downstream-comparison", str(downstream_path),
                "--closeout-decision", "INCONCLUSIVE",
            ])
            self.assertEqual(code, 1)
            code = finalize_cli.main([
                "--output-dir", str(output),
                "--reviews", str(state["reviews_path"]),
                "--audits", str(state["audit_path"]),
                "--reviewer", "human",
                "--completed-at", "2026-08-19T00:00:00.000Z",
                "--downstream-comparison", str(downstream_path),
                "--closeout-decision", "CONTEXT_ALONE_SUFFICIENT",
            ])
            self.assertEqual(code, 0)
            closeout = load_json_strict(
                output / "phase2k-closeout-status-v2.json", label="closeout",
            )
            self.assertEqual(closeout["status"], "CONTEXT_ALONE_SUFFICIENT")
            self.assertTrue(closeout["inputs_complete"])
            embedded = closeout["downstream_comparison"]
            self.assertEqual(
                embedded["schema_version"],
                DOWNSTREAM_COMPARISON_SCHEMA_VERSION,
            )
            self.assertEqual(embedded["diagnosis"], "MIXED")
            self.assertIn(
                "deltas",
                embedded["architectures"]["generative"],
            )
            self.assertEqual(
                embedded["architectures"]["generative"]["deltas"]["recall"],
                round(
                    embedded["architectures"]["generative"]["polished"][
                        "metrics"
                    ]["recall"]["rate"]
                    - embedded["architectures"]["generative"]["raw"][
                        "metrics"
                    ]["recall"]["rate"],
                    4,
                ),
            )
            validate_output_directory(
                output_dir=output,
                manifest_path=state["manifest_path"],
                packet_path=state["packet_path"],
                db_path=state["db_path"],
            )

    def test_no_phase2j_artifacts_changed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path = build_fixture(root)
            manifest_before = manifest_path.read_bytes()
            packet_before = packet_path.read_bytes()
            state = finalize_live_state(
                root,
                manifest_path=manifest_path,
                packet_path=packet_path,
                db_path=db_path,
            )
            downstream_path = root / "downstream.json"
            downstream_path.write_text(
                json.dumps(build_v2_comparison(state), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            code = finalize_cli.main([
                "--output-dir", str(state["output"]),
                "--reviews", str(state["reviews_path"]),
                "--audits", str(state["audit_path"]),
                "--reviewer", "human",
                "--completed-at", "2026-08-19T00:00:00.000Z",
                "--downstream-comparison", str(downstream_path),
                "--closeout-decision", "CONTEXTUAL_POLISH_VALIDATED",
            ])
            self.assertEqual(code, 0)
            self.assertEqual(manifest_path.read_bytes(), manifest_before)
            self.assertEqual(packet_path.read_bytes(), packet_before)

    def test_standalone_v2_validation_script(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = finalize_live_state(root)
            downstream_path = root / "downstream.json"
            downstream_path.write_text(
                json.dumps(build_v2_comparison(state), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            import scripts.validate_phase2k_downstream_comparison as validate_cli

            code = validate_cli.main([
                "--output-dir", str(state["output"]),
                "--downstream-comparison", str(downstream_path),
            ])
            self.assertEqual(code, 0)
            tampered = build_v2_comparison(state)
            tampered["content_sha256"] = "0" * 64
            tampered_path = root / "tampered.json"
            tampered_path.write_text(
                json.dumps(tampered, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            code = validate_cli.main([
                "--output-dir", str(state["output"]),
                "--downstream-comparison", str(tampered_path),
            ])
            self.assertEqual(code, 1)


def build_live(
    root: Path,
    manifest_path: Path,
    packet_path: Path,
    db_path: Path,
    output: Path,
    *,
    chat: Any | None = None,
) -> Any:
    from pipeline.phase2k_contextual_reconstruction import build_phase2k_outputs
    return build_phase2k_outputs(
        manifest_path=manifest_path,
        packet_path=packet_path,
        db_path=db_path,
        doc_path=None,
        output_dir=output,
        mode="live",
        chat=chat if chat is not None else _live_factory_chat(),
        inference_config=TEST_LIVE_INFERENCE_CONFIG,
    )


def _failing_reconstruction_chat() -> tuple[Any, dict[str, Any]]:
    good = _live_factory_chat()

    def chat(system: str, user: str) -> str:
        payload = json.loads(user)
        if payload.get("task") == "reconstruction":
            raise ValueError("synthetic reconstruction failure")
        return good(system, user)

    return chat, TEST_LIVE_INFERENCE_CONFIG


def synthetic_audit_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    selected = _selected("He hit R.", champion="Lux")
    target = {
        "window_id": selected["window_id"],
        "source_group_id": selected["source_group_id"],
        "source_absolute_start": selected["upstream_start"],
        "source_absolute_end": selected["upstream_end"],
        "text": selected["source_text"],
        "text_sha256": canonical_sha256(selected["source_text"]),
        "char_length": len(selected["source_text"]),
    }
    record_a = {
        "record_id": f"p2k:rec:{selected['window_id']}:A",
        "record_type": "A",
        "window_id": selected["window_id"],
        "target": target,
        "content": {
            "kind": "raw_bronze",
            "text": selected["source_text"],
            "text_sha256": canonical_sha256(selected["source_text"]),
            "char_length": len(selected["source_text"]),
        },
    }
    record_a["canonical_record_sha256"] = canonical_sha256(record_a)
    record_b = {
        "record_id": f"p2k:rec:{selected['window_id']}:B",
        "record_type": "B",
        "window_id": selected["window_id"],
        "target": target,
        "content": {
            "kind": "mechanical_clean",
            "generation_status": "GENERATED",
            "clean_text": selected["source_text"],
            "text": selected["source_text"],
            "repairs": [{
                "repair_id": "mech",
                "target_local_start": 0,
                "target_local_end": 2,
                "original_text": "He",
                "replacement": "He",
                "repair_type": "ASR_HOMOPHONE",
                "confidence": "HIGH",
                "evidence_spans": [],
                "rationale": "test",
            }],
            "repair_count": 1,
        },
    }
    record_b["canonical_record_sha256"] = canonical_sha256(record_b)
    reconstruction = {
        "clean_target_transcript": selected["source_text"],
        "contextual_repairs": [{
            "repair_id": "ctx",
            "target_local_start": 0,
            "target_local_end": 2,
            "original_text": "He",
            "replacement": "He",
            "repair_type": "CONTEXTUAL_ASR",
            "confidence": "HIGH",
            "evidence_spans": [],
            "rationale": "test",
        }],
        "bindings": [{
            "binding_id": "entity",
            "slot": "champion_identities",
            "mention": _bronze_span(selected, local_start=0, local_end=2),
            "resolved_candidate": "Lux",
            "resolved_status": "RESOLVED",
            "confidence": "HIGH",
            "evidence_spans": [],
            "alternatives": [],
            "metadata_contributed": True,
            "rationale": "test",
        }, {
            "binding_id": "ability",
            "slot": "ability_ownership",
            "mention": _bronze_span(selected, local_start=0, local_end=2),
            "resolved_candidate": "Lux",
            "resolved_status": "RESOLVED",
            "confidence": "HIGH",
            "evidence_spans": [],
            "alternatives": [],
            "metadata_contributed": True,
            "rationale": "test",
        }],
        "unresolved_alternatives": [],
        "generation_status": "GENERATED",
        "model_call": None,
    }
    polish = {
        "statements": [{
            "statement_id": "stmt",
            "text": selected["source_text"],
            "evidence_spans": [],
            "reconstruction_operation_ids": ["ctx"],
            "support_mode": "RECONSTRUCTION_DERIVED",
            "unchanged_source_quote": None,
        }],
        "unsupported_claims": [],
        "generation_status": "GENERATED",
        "model_call": None,
    }
    d_content = {
        "kind": "reconstruction",
        "generation_status": "GENERATED",
        "clean_target_transcript": selected["source_text"],
        "contextual_repairs": reconstruction["contextual_repairs"],
        "bindings": reconstruction["bindings"],
        "unresolved_alternatives": [],
        "reconstruction": reconstruction,
        "semantic_polish": polish,
        "model_calls": {},
        "failure": None,
        "counts": {},
    }
    record_d = {
        "record_id": f"p2k:rec:{selected['window_id']}:D",
        "record_type": "D",
        "window_id": selected["window_id"],
        "target": target,
        "content": d_content,
    }
    record_d["canonical_record_sha256"] = canonical_sha256(record_d)
    records = [record_a, record_b, record_d]
    records_obj = {
        "content_sha256": canonical_sha256({
            "records": records,
        }),
        "records": records,
    }
    audit = build_transformation_audit(
        records,
        {},
        records_sha256=records_obj["content_sha256"],
    )
    return audit, records_obj


def complete_reviews(
    packet: Mapping[str, Any],
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    reviews = {}
    for index, item in enumerate(packet["review_items"]):
        condition_code = mapping["labels"][
            item["blinded_label"]
        ]["condition_code"]
        scores = {
            field: (4 if field not in (
                "unsupported_invention", "remaining_ambiguity",
            ) else 0)
            for field in HUMAN_SCORE_FIELDS
        }
        if condition_code == "A":
            scores["asr_repair_correctness"] = NOT_APPLICABLE
        reviews[item["review_item_id"]] = {
            "scores": scores,
            "reviewer": "human",
            "completed_at": "2026-08-19T00:00:00.000Z",
            "notes": [],
        }
    return reviews


def fill_completed_audit(audit: dict[str, Any]) -> None:
    for window in audit["window_audits"]:
        for category, operations in window["operations"].items():
            for operation in operations:
                operation["decision"] = "APPROVE"
                if category.endswith("repairs"):
                    operation["error_taxonomy"] = None
                elif category.endswith("bindings"):
                    operation["error_taxonomy"] = None
                else:
                    operation["supported"] = True
                    operation["uncertainty_preserved"] = True
                    operation["negation_preserved"] = True
                    operation["modality_preserved"] = True
                    operation["causality_invented"] = False
                    operation["source_detail_dropped"] = False
                    operation["error_taxonomy"] = None


def passing_reviews(
    packet: Mapping[str, Any],
    mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Reviews whose pre-registered gate deterministically passes."""
    reviews = {}
    for item in packet["review_items"]:
        condition_code = mapping["labels"][
            item["blinded_label"]
        ]["condition_code"]
        is_a = condition_code == "A"
        is_d = condition_code == "D"
        scores = {}
        for field in HUMAN_SCORE_FIELDS:
            if is_a and field == "asr_repair_correctness":
                scores[field] = NOT_APPLICABLE
            elif field in ("unsupported_invention", "remaining_ambiguity"):
                scores[field] = 0
            elif is_d:
                scores[field] = 5
            else:
                scores[field] = 4
        reviews[item["review_item_id"]] = {
            "scores": scores,
            "reviewer": "human",
            "completed_at": "2026-08-19T00:00:00.000Z",
            "notes": [],
        }
    return reviews


def finalize_live_state(
    root: Path,
    *,
    reviews_factory: Any = passing_reviews,
    manifest_path: Path | None = None,
    packet_path: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Build a live Phase 2K output and finalize it without a comparison."""
    if manifest_path is None or packet_path is None or db_path is None:
        manifest_path, packet_path, db_path = build_fixture(root)
    output = root / "live"
    if not (output / "phase2k-reconstruction-records-v7.json").exists():
        build_live(root, manifest_path, packet_path, db_path, output)
    packet = load_json_strict(
        output / "phase2k-human-review-packet-v2.json", label="packet",
    )
    mapping = load_json_strict(
        output / "phase2k-human-review-mapping-v2.json", label="mapping",
    )
    reviews = reviews_factory(packet, mapping)
    reviews_path = root / "reviews.json"
    reviews_path.write_text(json.dumps(reviews), encoding="utf-8")
    audit_template = load_json_strict(
        output / "phase2k-transformation-audit-packet-v2.json", label="audit",
    )
    completed_audit = copy.deepcopy(audit_template)
    completed_audit["schema_version"] = (
        COMPLETED_TRANSFORMATION_AUDIT_SCHEMA_VERSION
    )
    completed_audit["release_gate"] = RELEASE_GATE_REVIEWED
    fill_completed_audit(completed_audit)
    completed_audit["content_sha256"] = canonical_sha256({
        key: value for key, value in completed_audit.items()
        if key != "content_sha256"
    })
    audit_path = root / "audits.json"
    audit_path.write_text(
        json.dumps(completed_audit, sort_keys=True) + "\n", encoding="utf-8",
    )
    code = finalize_cli.main([
        "--output-dir", str(output),
        "--reviews", str(reviews_path),
        "--audits", str(audit_path),
        "--reviewer", "human",
        "--completed-at", "2026-08-19T00:00:00.000Z",
    ])
    if code != 0:
        raise AssertionError("finalize without downstream should succeed")
    records = load_json_strict(
        output / "phase2k-reconstruction-records-v7.json", label="records",
    )
    finalized = load_json_strict(
        output / "phase2k-human-review-packet-v2-finalized.json",
        label="finalized",
    )
    human_summary = load_json_strict(
        output / "phase2k-human-review-summary-v1.json", label="summary",
    )
    return {
        "root": root,
        "manifest_path": manifest_path,
        "packet_path": packet_path,
        "db_path": db_path,
        "output": output,
        "reviews_path": reviews_path,
        "audit_path": audit_path,
        "records": records,
        "finalized": finalized,
        "human_summary": human_summary,
        "completed_audit": completed_audit,
        "window_ids": sorted({
            record["window_id"] for record in records["records"]
        }),
    }


def v2_rows(window_ids: list[str], *, polished: bool) -> list[dict[str, Any]]:
    """Synthetic per-window rows consistent with every v2 invariant."""
    rows = []
    for index, window_id in enumerate(window_ids):
        target_count = 2
        true_positive = 1 + (index % 2)
        false_negative = target_count - true_positive
        false_positive = index % 3
        output_count = true_positive + false_positive
        provenance_valid_count = index % 2
        abstained = index % 5 == 0
        if polished:
            true_positive = min(target_count, true_positive + 1)
            false_negative = target_count - true_positive
            false_positive += 1
            output_count = true_positive + false_positive
            provenance_valid_count = min(
                output_count, provenance_valid_count + 1,
            )
            abstained = False
        rows.append({
            "window_id": window_id,
            "target_count": target_count,
            "true_positive_count": true_positive,
            "false_positive_count": false_positive,
            "false_negative_count": false_negative,
            "output_count": output_count,
            "provenance_valid_count": provenance_valid_count,
            "abstained": abstained,
            "output_sha256": text_sha256(
                f"{'polished' if polished else 'raw'}:{window_id}",
            ),
        })
    return rows


def v2_architecture(
    window_ids: list[str],
    *,
    family: str,
    raw_adapter: str,
    polished_adapter: str,
) -> dict[str, Any]:
    return {
        "family": family,
        "semantic_contract_sha256": "ab" * 32,
        "model_or_scorer_config_sha256": "cd" * 32,
        "evaluation_contract_sha256": "ef" * 32,
        "raw_input_adapter_sha256": raw_adapter,
        "polished_input_adapter_sha256": polished_adapter,
        "raw": {
            "input_representation": "RAW_BRONZE",
            "output_artifact_sha256": text_sha256("raw-artifact"),
            "rows": v2_rows(window_ids, polished=False),
        },
        "polished": {
            "input_representation": "CONTEXTUAL_POLISH",
            "output_artifact_sha256": text_sha256("polished-artifact"),
            "rows": v2_rows(window_ids, polished=True),
        },
    }


def build_v2_comparison(
    state: Mapping[str, Any],
    *,
    decision: str = "CONTEXTUAL_POLISH_VALIDATED",
    diagnosis: str = "MIXED",
    note: str = "synthetic v2 downstream comparison for Phase 2K tests",
) -> dict[str, Any]:
    window_ids = state["window_ids"]
    dataset_binding = {
        "phase2k_records_sha256": state["records"]["content_sha256"],
        "finalized_human_packet_sha256": state["finalized"]["content_sha256"],
        "human_summary_sha256": canonical_sha256(state["human_summary"]),
        "completed_transformation_audit_sha256": (
            state["completed_audit"]["content_sha256"]
        ),
        "window_ids_sha256": canonical_sha256(window_ids),
        "window_count": len(window_ids),
        "human_review_gate_status": "PASSED",
    }
    contract = {
        "contract_version": "phase2k-semantic-target-contract-v1",
        "contract_sha256": "12" * 32,
        "target_count": 2 * len(window_ids),
        "boundary_rule": "exact Phase 2J window boundaries",
    }
    architectures = {
        "generative": v2_architecture(
            window_ids,
            family=GENERATIVE_ARCHITECTURE_FAMILY,
            raw_adapter="11" * 32,
            polished_adapter="22" * 32,
        ),
        "discriminative": v2_architecture(
            window_ids,
            family=DISCRIMINATIVE_ARCHITECTURE_FAMILY,
            raw_adapter="33" * 32,
            polished_adapter="44" * 32,
        ),
    }
    return build_downstream_comparison(
        dataset_binding=dataset_binding,
        semantic_target_contract=contract,
        architectures=architectures,
        decision=decision,
        diagnosis=diagnosis,
        note=note,
    )


def rehash(comparison: dict[str, Any]) -> dict[str, Any]:
    comparison["content_sha256"] = canonical_sha256({
        key: value for key, value in comparison.items()
        if key != "content_sha256"
    })
    return comparison


if __name__ == "__main__":
    unittest.main()
