"""Focused tests for the isolated Phase 2J source-grounded core module."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from pipeline.phase2j_context_ablation import (
    CONDITION_CODES,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_VOCABULARY_PATH,
    MATERIALITY_POLICY,
    OUTPUT_SCHEMA_VERSION,
    PAYLOAD_SCHEMA_VERSION,
    SEMANTIC_FIELDS,
    build_condition_payloads,
    build_deepseek_run_packet,
    build_extraction_instructions,
    build_human_review_packet,
    build_materiality_summary,
    build_outputs_bundle,
    build_payloads_artifact,
    build_selection_artifact,
    build_phase2j_context_ablation_outputs,
    build_sol_intermediate_schema,
    canonical_sha256,
    compute_materiality,
    import_completed_reviews,
    import_deepseek_run_outputs,
    import_sol_intermediate_response,
    load_json_strict,
    load_lexical_vocabulary,
    load_phase2j_manifest,
    fetch_transcript_rows,
    open_transcript_db,
    require_frozen_material_summary,
    select_cases,
    text_sha256,
    validate_deepseek_import_artifact,
    validate_extraction_output,
    validate_extraction_outputs_bundle,
    validate_human_review_mapping,
    validate_human_review_packet,
    validate_materiality_summary,
    validate_payloads_artifact,
    validate_selection_artifact,
    validate_sol_intermediate_response,
    validate_sol_intermediate_schema,
    validate_phase2j_frozen_inputs,
)
from tests._phase2j_context_ablation_helpers import (
    build_fixture,
    expected_selected_case_ids,
    make_completed_reviews,
    make_outputs_bundle,
    make_valid_output,
)


REAL_MANIFEST = DEFAULT_MANIFEST_PATH


def _build_full_fixture(root: Path) -> dict[str, Any]:
    """Build selection + instructions + payloads + outputs in one temp tree."""
    manifest_path, packet_path, db_path, transcripts = build_fixture(root)
    result = build_phase2j_context_ablation_outputs(
        manifest_path=manifest_path,
        packet_path=packet_path,
        db_path=db_path,
        output_dir=root / "out",
    )
    output_dir = root / "out"
    payloads = load_json_strict(
        output_dir / "phase2j-context-ablation-condition-payloads-v2.json",
        label="payloads",
    )
    outputs = make_outputs_bundle(payloads)
    bundle_path = output_dir / "phase2j-context-ablation-extraction-outputs-v2.json"
    bundle_path.write_text(json.dumps(outputs), encoding="utf-8")
    manifest, packet = validate_phase2j_frozen_inputs(
        manifest_path, packet_path,
    )
    result.update({
        "manifest_path": manifest_path,
        "packet_path": packet_path,
        "db_path": db_path,
        "manifest": manifest,
        "packet": packet,
        "transcripts": transcripts,
        "output_dir": output_dir,
        "payloads": payloads,
        "outputs": outputs,
    })
    return result


class SelectionTests(unittest.TestCase):
    def test_real_manifest_selects_exact_ten_cases(self):
        manifest = load_phase2j_manifest(REAL_MANIFEST)
        cases = select_cases(manifest)
        self.assertEqual(len(cases), 10)
        self.assertEqual(
            [case["case_id"] for case in cases],
            [f"p2ja:case:{rank:04d}" for rank in range(1, 11)],
        )
        expected_windows = [
            "pool:GKciadsuvlM:w00025-02893996a53ef809f95d",
            "pool:wTxMNxJczec:w00160-1cbf79226c5b15917cf1",
            "pool:AOxq_mddftk:w00070-c95460613f77ca936b6b",
            "pool:CLFXg110j1E:w00282-4d9609f4f74990026b44",
            "pool:n2RuZ0vwkE4:w00288-4d7b8fc17b8fcff31c07",
            "pool:jDxTIHR0Jq0:w00022-9f512d74e6921ba711b0",
            "pool:JEFr-hMEkPQ:w00050-b964f6fe25278236e19b",
            "pool:euxH8dbAH7c:w00386-b134366e14fa5a758525",
            "pool:DIWlwK1VArk:w00283-a3e96e923c748d643aa5",
            "pool:ZYIPtq-OZDI:w00120-839f6a5e88b097fd4463",
        ]
        self.assertEqual([case["window_id"] for case in cases], expected_windows)
        self.assertEqual(
            [case["difficulty_score"] for case in cases],
            [19, 19, 18, 18, 16, 16, 16, 15, 15, 15],
        )
        self.assertEqual(
            sorted(cases[0]["contributing_tags"]),
            sorted({
                "punctuation_poor", "omitted_actor", "pronoun",
                "multiple_abilities", "cause_chain", "uncertainty",
                "contradiction", "explicit_cause", "contrast",
                "advice_explanation", "resource_exchange",
            }),
        )

    def test_synthetic_selection_deterministic_and_ranked(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, packet_path, db_path, _ = build_fixture(
                Path(temporary),
            )
            manifest, packet = validate_phase2j_frozen_inputs(
                manifest_path, packet_path,
            )
            connection = open_transcript_db(db_path)
            try:
                rows = fetch_transcript_rows(connection, manifest["selected"])
            finally:
                connection.close()
            cases = select_cases(manifest, transcript_rows=rows)
            self.assertEqual(
                [case["case_id"] for case in cases],
                expected_selected_case_ids(),
            )
            self.assertEqual(
                [case["selection_rank"] for case in cases],
                list(range(1, 11)),
            )
            for index in range(1, len(cases)):
                previous = cases[index - 1]
                current = cases[index]
                if previous["difficulty_score"] == current["difficulty_score"]:
                    self.assertGreaterEqual(
                        previous["phenomenon_count"],
                        current["phenomenon_count"],
                    )
            reselected = select_cases(manifest)
            self.assertEqual(
                [case["window_id"] for case in reselected],
                [case["window_id"] for case in cases],
            )

    def test_selection_artifact_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path, _ = build_fixture(root)
            manifest, packet = validate_phase2j_frozen_inputs(
                manifest_path, packet_path,
            )
            connection = open_transcript_db(db_path)
            try:
                rows = fetch_transcript_rows(connection, manifest["selected"])
            finally:
                connection.close()
            cases = select_cases(manifest, transcript_rows=rows)
            artifact = build_selection_artifact(
                manifest_path=manifest_path,
                packet_path=packet_path,
                manifest=manifest,
                packet=packet,
                cases=cases,
                db_path=db_path,
            )
            validate_selection_artifact(
                artifact,
                manifest_path=manifest_path,
                packet_path=packet_path,
                manifest=manifest,
                packet=packet,
                db_path=db_path,
            )
            tampered = json.loads(json.dumps(artifact))
            tampered["cases"][0]["difficulty_score"] += 1
            with self.assertRaises(ValueError):
                validate_selection_artifact(
                    tampered,
                    manifest_path=manifest_path,
                    packet_path=packet_path,
                    manifest=manifest,
                    packet=packet,
                    db_path=db_path,
                )

    def test_self_rehashed_tampered_selection_rejected(self):
        """Recomputing every content hash cannot defeat independent selection."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, packet_path, db_path, _ = build_fixture(root)
            manifest, packet = validate_phase2j_frozen_inputs(
                manifest_path, packet_path,
            )
            cases = select_cases(manifest)
            artifact = build_selection_artifact(
                manifest_path=manifest_path,
                packet_path=packet_path,
                manifest=manifest,
                packet=packet,
                cases=cases,
                db_path=db_path,
            )
            tampered = json.loads(json.dumps(artifact))
            tampered["cases"][0]["metadata"] = {
                "champion": "TAMPERED",
                "role": "mid",
                "video_title": "Tampered title",
            }
            tampered["content_sha256"] = canonical_sha256({
                key: value for key, value in tampered.items()
                if key != "content_sha256"
            })
            with self.assertRaises(ValueError):
                validate_selection_artifact(
                    tampered,
                    manifest_path=manifest_path,
                    packet_path=packet_path,
                    manifest=manifest,
                    packet=packet,
                    db_path=db_path,
                )


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.fixture = _build_full_fixture(self.root)
        self.payloads = self.fixture["payloads"]
        self.outputs = self.fixture["outputs"]

    def tearDown(self):
        self._temporary.cleanup()

    def test_a_payload_strict_allowlist_and_isolation(self):
        payload_a = self.payloads["cases"][0]["A"]
        self.assertEqual(
            set(payload_a),
            {
                "schema_version", "condition", "case_id", "selection_rank",
                "target", "instructions", "instructions_sha256",
                "content_sha256",
            },
        )
        self.assertEqual(
            set(payload_a["target"]),
            {"bronze_text", "bronze_text_sha256", "bronze_char_length"},
        )
        text = json.dumps(payload_a)
        for forbidden in (
            "transcript", "metadata", "vocabulary", "champion_abilities",
            "video_id", "video_url", "window_id", "source_group_id",
            "full_transcript", "phenomena", "partition", "captions",
            "timestamps", "target_char_start", "target_char_end",
        ):
            self.assertNotIn(f'"{forbidden}"', text)
        bronze = payload_a["target"]["bronze_text"]
        self.assertEqual(
            text_sha256(bronze),
            payload_a["target"]["bronze_text_sha256"],
        )

    def test_a_has_no_transcript_prefix_or_suffix_leak(self):
        case = self.payloads["cases"][0]
        provenance = self.payloads["provenance_by_case"][case["case_id"]]
        transcript = self.fixture["transcripts"][provenance["video_id"]]
        bronze = case["A"]["target"]["bronze_text"]
        self.assertIn(bronze, transcript)
        start = provenance["target_char_start"]
        end = provenance["target_char_end"]
        self.assertEqual(transcript[start:end], bronze)
        prefix_word = transcript[max(0, start - 12):start].split()[-1] if start > 0 else ""
        suffix_word = transcript[end:end + 12].split()[0] if end < len(transcript) else ""
        payload_text = json.dumps(case["A"])
        if prefix_word:
            self.assertNotIn(prefix_word, payload_text)
        if suffix_word:
            self.assertNotIn(suffix_word, payload_text)

    def test_a_b_target_byte_identical(self):
        for payload_case in self.payloads["cases"]:
            self.assertEqual(
                payload_case["A"]["target"],
                payload_case["B"]["target"],
            )
            self.assertEqual(
                payload_case["A"]["instructions"],
                payload_case["B"]["instructions"],
            )
            self.assertEqual(
                payload_case["A"]["instructions_sha256"],
                payload_case["B"]["instructions_sha256"],
            )

    def test_b_contains_transcript_offsets_metadata_vocabulary(self):
        for payload_case in self.payloads["cases"]:
            payload_b = payload_case["B"]
            self.assertEqual(
                set(payload_b),
                {
                    "schema_version", "condition", "case_id", "selection_rank",
                    "target", "transcript", "target_char_start",
                    "target_char_end", "metadata", "vocabulary",
                    "vocabulary_sha256", "instructions", "instructions_sha256",
                    "content_sha256",
                },
            )
            self.assertEqual(
                set(payload_b["metadata"]),
                {"video_title", "champion", "role", "rank", "game"},
            )
            transcript = payload_b["transcript"]
            start = payload_b["target_char_start"]
            end = payload_b["target_char_end"]
            self.assertEqual(
                transcript[start:end],
                payload_b["target"]["bronze_text"],
            )
            self.assertIn("lexical_vocabulary", payload_b["vocabulary"])
            self.assertIn("champion_abilities", payload_b["vocabulary"])
            self.assertGreater(payload_b["vocabulary"]["ability_row_count"], 0)
            self.assertNotIn("video_url", json.dumps(payload_b))

    def test_vocabulary_no_source_identity_and_no_strategic_data(self):
        payload_b = self.payloads["cases"][0]["B"]
        vocabulary = payload_b["vocabulary"]
        champions = {entry["champion"] for entry in vocabulary["champions"]}
        metadata_champion = payload_b["metadata"]["champion"]
        self.assertIn(metadata_champion, champions)
        self.assertIn("Jinx", champions)  # literally present in every transcript
        for ability in vocabulary["champion_abilities"]:
            provenance = ability["provenance"]
            self.assertEqual(provenance["source"], "champion_abilities")
            self.assertTrue(provenance["selection_reasons"])
            self.assertNotIn("video_id", provenance)
        text = json.dumps(vocabulary)
        for forbidden in (
            "archetypes", "fingerprints", "strategic_relations", "labels",
            "phase2k", "video_id", "video_url",
        ):
            self.assertNotIn(f'"{forbidden}"', text)

    def test_provenance_at_outer_artifact_level(self):
        provenance = self.payloads["provenance_by_case"]
        self.assertEqual(len(provenance), 10)
        for case_id, entry in provenance.items():
            self.assertEqual(
                set(entry),
                {
                    "video_id", "video_url", "source_group_id", "window_id",
                    "full_transcript_sha256", "full_transcript_char_length",
                    "target_char_start", "target_char_end",
                    "vocabulary_sha256",
                },
            )
            self.assertTrue(entry["video_url"].startswith("https://"))

    def test_shared_instructions_hash_identical(self):
        instructions = build_extraction_instructions()
        instructions_sha256 = canonical_sha256(instructions)
        for payload_case in self.payloads["cases"]:
            self.assertEqual(
                payload_case["A"]["instructions_sha256"], instructions_sha256,
            )
            self.assertEqual(
                payload_case["B"]["instructions_sha256"], instructions_sha256,
            )
            self.assertEqual(
                payload_case["A"]["instructions"],
                payload_case["B"]["instructions"],
            )

    def test_payloads_artifact_validates_canonically(self):
        instructions_artifact = load_json_strict(
            self.root / "out/phase2j-context-ablation-extraction-instructions-v2.json",
            label="instructions",
        )
        instructions = {
            key: value for key, value in instructions_artifact.items()
            if key != "content_sha256"
        }
        validate_payloads_artifact(
            self.payloads,
            selection=load_json_strict(
                self.root / "out/phase2j-context-ablation-selection-v1.json",
                label="selection",
            ),
            instructions=instructions,
            lexical_vocabulary=load_lexical_vocabulary(DEFAULT_VOCABULARY_PATH),
            manifest_path=self.fixture["manifest_path"],
            packet_path=self.fixture["packet_path"],
            manifest=self.fixture["manifest"],
            packet=self.fixture["packet"],
            db_path=self.fixture["db_path"],
        )

    def test_self_rehashed_tampered_payload_rejected(self):
        payloads = json.loads(json.dumps(self.payloads))
        case = payloads["cases"][0]
        tampered_b = json.loads(json.dumps(case["B"]))
        tampered_b["transcript"] = tampered_b["transcript"] + " TAMPERED"
        tampered_b["content_sha256"] = canonical_sha256({
            key: value for key, value in tampered_b.items()
            if key != "content_sha256"
        })
        case["B"] = tampered_b
        case["content_sha256"] = canonical_sha256({
            key: value for key, value in case.items()
            if key != "content_sha256"
        })
        payloads["content_sha256"] = canonical_sha256({
            key: value for key, value in payloads.items()
            if key != "content_sha256"
        })
        instructions_artifact = load_json_strict(
            self.root / "out/phase2j-context-ablation-extraction-instructions-v2.json",
            label="instructions",
        )
        instructions = {
            key: value for key, value in instructions_artifact.items()
            if key != "content_sha256"
        }
        with self.assertRaises(ValueError):
            validate_payloads_artifact(
                payloads,
                selection=load_json_strict(
                    self.root / "out/phase2j-context-ablation-selection-v1.json",
                    label="selection",
                ),
                instructions=instructions,
                lexical_vocabulary=load_lexical_vocabulary(
                    DEFAULT_VOCABULARY_PATH,
                ),
                manifest_path=self.fixture["manifest_path"],
                packet_path=self.fixture["packet_path"],
                manifest=self.fixture["manifest"],
                packet=self.fixture["packet"],
                db_path=self.fixture["db_path"],
            )

    def test_wrong_case_order_in_payloads_rejected(self):
        payloads = json.loads(json.dumps(self.payloads))
        payloads["cases"][0], payloads["cases"][1] = (
            payloads["cases"][1], payloads["cases"][0],
        )
        payloads["cases"][0]["selection_rank"] = 1
        payloads["cases"][1]["selection_rank"] = 2
        payloads["content_sha256"] = canonical_sha256({
            key: value for key, value in payloads.items()
            if key != "content_sha256"
        })
        instructions_artifact = load_json_strict(
            self.root / "out/phase2j-context-ablation-extraction-instructions-v2.json",
            label="instructions",
        )
        instructions = {
            key: value for key, value in instructions_artifact.items()
            if key != "content_sha256"
        }
        with self.assertRaises(ValueError):
            validate_payloads_artifact(
                payloads,
                selection=load_json_strict(
                    self.root / "out/phase2j-context-ablation-selection-v1.json",
                    label="selection",
                ),
                instructions=instructions,
                lexical_vocabulary=load_lexical_vocabulary(
                    DEFAULT_VOCABULARY_PATH,
                ),
                manifest_path=self.fixture["manifest_path"],
                packet_path=self.fixture["packet_path"],
                manifest=self.fixture["manifest"],
                packet=self.fixture["packet"],
                db_path=self.fixture["db_path"],
            )


class OutputValidationTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.fixture = _build_full_fixture(self.root)
        self.payloads = self.fixture["payloads"]
        self.outputs = self.fixture["outputs"]
        self.payload_case = self.payloads["cases"][0]
        self.case_id = self.payload_case["case_id"]

    def tearDown(self):
        self._temporary.cleanup()

    def _valid_output(self, condition: str = "A"):
        return make_valid_output(
            self.payload_case[condition],
            case_id=self.case_id,
            condition=condition,
        )

    def test_valid_outputs_pass(self):
        validate_extraction_outputs_bundle(
            self.outputs,
            payloads_artifact=self.payloads,
            instructions=build_extraction_instructions(),
        )

    def test_wrong_case_order_and_id_rejected(self):
        bundle = json.loads(json.dumps(self.outputs))
        first = bundle["cases"][0]
        second = bundle["cases"][1]
        bundle["cases"][0], bundle["cases"][1] = second, first
        with self.assertRaises(ValueError):
            validate_extraction_outputs_bundle(
                bundle,
                payloads_artifact=self.payloads,
                instructions=build_extraction_instructions(),
            )
        bundle = json.loads(json.dumps(self.outputs))
        bundle["cases"][0]["case_id"] = "p2ja:case:0099"
        with self.assertRaises(ValueError):
            validate_extraction_outputs_bundle(
                bundle,
                payloads_artifact=self.payloads,
                instructions=build_extraction_instructions(),
            )

    def test_unknown_and_missing_keys_rejected(self):
        output = self._valid_output()
        with self.assertRaises(ValueError):
            validate_extraction_output(
                {**output, "extra_key": 1},
                case_id=self.case_id,
                condition="A",
                payload=self.payload_case["A"],
            )
        missing = {key: value for key, value in output.items() if key != "fields"}
        with self.assertRaises(ValueError):
            validate_extraction_output(
                missing,
                case_id=self.case_id,
                condition="A",
                payload=self.payload_case["A"],
            )

    def test_citation_outside_supplied_source_rejected(self):
        output = self._valid_output()
        output["fields"]["actors"][0]["source_references"] = [{
            "quote": "coach",
            "source_range": {"char_start": 0, "char_end": 999999},
        }]
        with self.assertRaises(ValueError):
            validate_extraction_output(
                output,
                case_id=self.case_id,
                condition="A",
                payload=self.payload_case["A"],
            )

    def test_non_exact_quote_rejected(self):
        output = self._valid_output()
        reference = output["fields"]["actors"][0]["source_references"][0]
        reference["quote"] = "not an exact slice at all"
        with self.assertRaises(ValueError):
            validate_extraction_output(
                output,
                case_id=self.case_id,
                condition="A",
                payload=self.payload_case["A"],
            )

    def test_malformed_source_range_rejected(self):
        output = self._valid_output()
        item = output["fields"]["supporting_source_ranges"][0]
        item["source_range"] = {"char_start": 5, "char_end": 1}
        with self.assertRaises(ValueError):
            validate_extraction_output(
                output,
                case_id=self.case_id,
                condition="A",
                payload=self.payload_case["A"],
            )
        output = self._valid_output()
        output["fields"]["supporting_source_ranges"][0]["source_range"] = {
            "char_start": -1,
            "char_end": 100,
        }
        with self.assertRaises(ValueError):
            validate_extraction_output(
                output,
                case_id=self.case_id,
                condition="A",
                payload=self.payload_case["A"],
            )

    def test_supporting_range_outside_cited_union_rejected(self):
        output = self._valid_output()
        item = output["fields"]["supporting_source_ranges"][0]
        source = self.payload_case["A"]["target"]["bronze_text"]
        quote = source.split(" ")[0]
        item["source_references"] = [{
            "quote": quote,
            "source_range": {"char_start": 0, "char_end": len(quote)},
        }]
        item["source_range"] = {
            "char_start": 0,
            "char_end": len(quote) + 5,
        }
        with self.assertRaises(ValueError):
            validate_extraction_output(
                output,
                case_id=self.case_id,
                condition="A",
                payload=self.payload_case["A"],
            )

    def test_a_offsets_are_into_bronze_not_full_transcript(self):
        """An offset valid in the full transcript but outside Bronze fails A."""
        output = self._valid_output("A")
        bronze = self.payload_case["A"]["target"]["bronze_text"]
        full = self.payload_case["B"]["transcript"]
        outside = full[len(bronze):len(bronze) + 5]
        output["fields"]["actors"][0]["source_references"] = [{
            "quote": outside,
            "source_range": {
                "char_start": len(bronze),
                "char_end": len(bronze) + len(outside),
            },
        }]
        with self.assertRaises(ValueError):
            validate_extraction_output(
                output,
                case_id=self.case_id,
                condition="A",
                payload=self.payload_case["A"],
            )

    def test_b_offsets_into_full_transcript_validate(self):
        output = self._valid_output("B")
        validate_extraction_output(
            output,
            case_id=self.case_id,
            condition="B",
            payload=self.payload_case["B"],
        )
        full = self.payload_case["B"]["transcript"]
        quote = full.split(" ")[0]
        self.assertEqual(output["fields"]["actors"][0]["source_references"][0][
            "quote"
        ], quote)

    def test_duplicate_item_ids_rejected(self):
        output = self._valid_output()
        output["fields"]["actors"].append(json.loads(json.dumps(
            output["fields"]["actors"][0],
        )))
        with self.assertRaises(ValueError):
            validate_extraction_output(
                output,
                case_id=self.case_id,
                condition="A",
                payload=self.payload_case["A"],
            )

    def test_leaked_fields_rejected(self):
        output = self._valid_output()
        output["fields"]["event"] = [{
            "item_id": f"{self.case_id}:A:event:0001",
            "extraction_text": "leak",
            "resolution_status": "literal_explicit",
            "source_references": [],
            "metadata": {"champion": "Lux"},
        }]
        with self.assertRaises(ValueError):
            validate_extraction_output(
                output,
                case_id=self.case_id,
                condition="A",
                payload=self.payload_case["A"],
            )

    def test_wrong_condition_rejected(self):
        output = self._valid_output("A")
        with self.assertRaises(ValueError):
            validate_extraction_output(
                output,
                case_id=self.case_id,
                condition="B",
                payload=self.payload_case["B"],
            )

    def test_semantic_fields_use_supporting_source_ranges(self):
        self.assertIn("supporting_source_ranges", SEMANTIC_FIELDS)
        self.assertNotIn("supporting_timestamp_ranges", SEMANTIC_FIELDS)


class SolIntermediateTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.fixture = _build_full_fixture(self.root)
        self.payloads = self.fixture["payloads"]
        self.payload_case = self.payloads["cases"][0]
        self.case_id = self.payload_case["case_id"]

    def tearDown(self):
        self._temporary.cleanup()

    def _intermediate(
        self,
        *,
        condition: str = "A",
        fields: Mapping[str, list[Any]] | None = None,
    ) -> dict[str, Any]:
        payload = self.payload_case[condition]
        source = (
            payload["target"]["bronze_text"]
            if condition == "A" else payload["transcript"]
        )
        first = re.search(r"\S+", source).group()
        base_fields = {
            field: [] for field in SEMANTIC_FIELDS
        }
        base_fields["actors"] = [{
            "extraction_text": "coach",
            "resolution_status": "literal_explicit",
            "source_references": [
                {"quote": first, "occurrence_index": 0},
            ],
        }]
        base_fields["supporting_source_ranges"] = [{
            "extraction_text": "target range",
            "resolution_status": "literal_explicit",
            "source_references": [
                {"quote": first, "occurrence_index": 0},
            ],
        }]
        if fields is not None:
            base_fields = dict(fields)
        response = {
            "schema_version": build_sol_intermediate_schema()["schema_version"],
            "case_id": self.case_id,
            "condition": condition,
            "payload_sha256": payload["content_sha256"],
            "instructions_sha256": payload["instructions_sha256"],
            "fields": base_fields,
        }
        return response

    def test_instructions_require_quotes_and_occurrence_indexes_not_offsets(self):
        instructions = build_extraction_instructions()
        self.assertIn("occurrence_index", instructions["reference_rule"])
        self.assertIn("occurrence_index", instructions["source_range_rule"])
        self.assertIn("occurrence_index", " ".join(instructions["output_rules"]))
        self.assertIn(
            "Do not count, estimate, or return character offsets",
            instructions["reference_rule"],
        )
        self.assertIn("importer", instructions["reference_rule"])
        self.assertIn("importer derives", instructions["source_range_rule"])
        self.assertNotIn("item_id", " ".join(instructions["output_rules"]))
        for flag in (
            "direct_extraction_only", "no_mechanical_clean",
            "no_contextual_rewriting", "no_semantic_polish",
            "no_strategic_abstraction",
        ):
            self.assertIs(instructions[flag], True)

    def test_intermediate_schema_is_closed_with_exactly_eight_fields(self):
        schema = build_sol_intermediate_schema()
        validate_sol_intermediate_schema(schema)
        self.assertIs(schema["additionalProperties"], False)
        fields = schema["properties"]["fields"]
        self.assertEqual(fields["required"], list(SEMANTIC_FIELDS))
        self.assertEqual(set(fields["properties"]), set(SEMANTIC_FIELDS))
        self.assertIs(fields["additionalProperties"], False)
        item = fields["properties"]["actors"]["items"]
        self.assertEqual(
            set(item["required"]),
            {"extraction_text", "resolution_status", "source_references"},
        )
        self.assertIs(item["additionalProperties"], False)
        reference = item["properties"]["source_references"]["items"]
        self.assertEqual(
            set(reference["required"]), {"quote", "occurrence_index"},
        )
        self.assertIs(reference["additionalProperties"], False)
        self.assertNotIn("source_range", reference["properties"])
        self.assertNotIn("item_id", item["properties"])

    def test_unique_quote_resolves_to_byte_exact_range(self):
        payload = self.payload_case["A"]
        source = payload["target"]["bronze_text"]
        response = self._intermediate(condition="A")
        output = import_sol_intermediate_response(
            response,
            case_id=self.case_id,
            condition="A",
            payload=payload,
        )
        quote = response["fields"]["actors"][0]["source_references"][0]["quote"]
        expected_start = source.find(quote)
        reference = output["fields"]["actors"][0]["source_references"][0]
        self.assertEqual(
            reference["source_range"],
            {"char_start": expected_start, "char_end": expected_start + len(quote)},
        )
        self.assertEqual(source[expected_start:expected_start + len(quote)], quote)
        self.assertEqual(
            output["fields"]["supporting_source_ranges"][0]["source_range"],
            dict(reference["source_range"]),
        )
        self.assertEqual(output["fields"]["actors"][0]["item_id"],
                         f"{self.case_id}:A:actors:0001")

    def test_repeated_quote_occurrence_index_selection(self):
        source = "alpha beta alpha gamma alpha delta"
        payload = {
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "condition": "A",
            "case_id": self.case_id,
            "target": {"bronze_text": source},
            "content_sha256": "1" * 64,
            "instructions_sha256": "2" * 64,
        }
        expected_starts: list[int] = []
        start = 0
        while True:
            index = source.find("alpha", start)
            if index < 0:
                break
            expected_starts.append(index)
            start = index + len("alpha")
        self.assertEqual(expected_starts, [0, 11, 23])
        fields = {field: [] for field in SEMANTIC_FIELDS}
        fields["actors"] = [{
            "extraction_text": "coach",
            "resolution_status": "literal_explicit",
            "source_references": [
                {"quote": "alpha", "occurrence_index": 1},
            ],
        }]
        response = self._intermediate(condition="A", fields=fields)
        response["payload_sha256"] = payload["content_sha256"]
        response["instructions_sha256"] = payload["instructions_sha256"]
        output = import_sol_intermediate_response(
            response,
            case_id=self.case_id,
            condition="A",
            payload=payload,
        )
        reference = output["fields"]["actors"][0]["source_references"][0]
        self.assertEqual(reference["source_range"]["char_start"], 11)
        self.assertEqual(
            source[reference["source_range"]["char_start"]:
                   reference["source_range"]["char_end"]],
            "alpha",
        )

    def test_out_of_range_occurrence_rejected(self):
        payload = self.payload_case["A"]
        response = self._intermediate(condition="A")
        response["fields"]["actors"][0]["source_references"][0][
            "occurrence_index"
        ] = 999999
        with self.assertRaises(ValueError):
            import_sol_intermediate_response(
                response,
                case_id=self.case_id,
                condition="A",
                payload=payload,
            )

    def test_missing_quote_rejected(self):
        payload = self.payload_case["A"]
        response = self._intermediate(condition="A")
        response["fields"]["actors"][0]["source_references"][0][
            "quote"
        ] = "quote-that-does-not-exist-anywhere"
        with self.assertRaises(ValueError):
            import_sol_intermediate_response(
                response,
                case_id=self.case_id,
                condition="A",
                payload=payload,
            )

    def test_hash_and_condition_binding_rejected(self):
        payload = self.payload_case["A"]
        response = self._intermediate(condition="A")
        response["payload_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_sol_intermediate_response(
                response,
                case_id=self.case_id,
                condition="A",
                payload=payload,
            )
        response = self._intermediate(condition="A")
        response["instructions_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_sol_intermediate_response(
                response,
                case_id=self.case_id,
                condition="A",
                payload=payload,
            )
        response = self._intermediate(condition="A")
        with self.assertRaises(ValueError):
            validate_sol_intermediate_response(
                response,
                case_id=self.case_id,
                condition="B",
                payload=self.payload_case["B"],
            )

    def test_model_supplied_offset_rejected(self):
        payload = self.payload_case["A"]
        response = self._intermediate(condition="A")
        response["fields"]["supporting_source_ranges"][0]["source_range"] = {
            "char_start": 0,
            "char_end": 1,
        }
        with self.assertRaises(ValueError):
            validate_sol_intermediate_response(
                response,
                case_id=self.case_id,
                condition="A",
                payload=payload,
            )

    def test_all_eight_fields_import_preserving_order_and_count(self):
        payload = self.payload_case["A"]
        source = payload["target"]["bronze_text"]
        first = re.search(r"\S+", source).group()
        fields = {}
        for index, field in enumerate(SEMANTIC_FIELDS):
            fields[field] = [{
                "extraction_text": f"item-{field}",
                "resolution_status": "literal_explicit",
                "source_references": [
                    {"quote": first, "occurrence_index": 0},
                ],
            }]
        fields["supporting_source_ranges"] = [{
            "extraction_text": "range",
            "resolution_status": "literal_explicit",
            "source_references": [
                {"quote": first, "occurrence_index": 0},
            ],
        }]
        response = self._intermediate(condition="A", fields=fields)
        output = import_sol_intermediate_response(
            response,
            case_id=self.case_id,
            condition="A",
            payload=payload,
        )
        validate_extraction_output(
            output,
            case_id=self.case_id,
            condition="A",
            payload=payload,
        )
        self.assertEqual(
            [field for field in output["fields"]],
            list(SEMANTIC_FIELDS),
        )
        for field in SEMANTIC_FIELDS:
            self.assertEqual(
                [item["extraction_text"] for item in output["fields"][field]],
                [item["extraction_text"] for item in response["fields"][field]],
            )
            self.assertEqual(
                [item["resolution_status"] for item in output["fields"][field]],
                [item["resolution_status"] for item in response["fields"][field]],
            )
            for item_index, item in enumerate(output["fields"][field], 1):
                self.assertEqual(
                    item["item_id"],
                    f"{self.case_id}:A:{field}:{item_index:04d}",
                )


class ReviewPacketTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.fixture = _build_full_fixture(self.root)
        self.payloads = self.fixture["payloads"]
        self.outputs = self.fixture["outputs"]
        self.packet, self.mapping = build_human_review_packet(
            outputs_artifact=self.outputs,
            payloads_artifact=self.payloads,
            transcript_rows={
                entry["video_id"]: {
                    "transcript": self.fixture["transcripts"][entry["video_id"]],
                    "transcript_sha256": text_sha256(
                        self.fixture["transcripts"][entry["video_id"]],
                    ),
                }
                for entry in self.payloads["provenance_by_case"].values()
            },
        )

    def tearDown(self):
        self._temporary.cleanup()

    def test_packet_shape_blinding_and_shared_source_evidence(self):
        validate_human_review_packet(self.packet, require_blank=True)
        validate_human_review_mapping(
            self.mapping,
            packet=self.packet,
            outputs_artifact=self.outputs,
            payloads_artifact=self.payloads,
        )
        self.assertEqual(len(self.packet["review_items"]), 160)
        self.assertEqual(len(self.packet["source_evidence"]), 10)
        self.assertEqual(self.packet["release_gate"], "AWAITING_HUMAN_REVIEW")
        self.assertNotIn("condition_code", json.dumps(self.packet))
        self.assertNotIn('"seed"', json.dumps(self.packet))
        for item in self.packet["review_items"]:
            index = item["presentation"]["source_evidence_index"]
            evidence = self.packet["source_evidence"][index]
            self.assertEqual(evidence["case_id"], item["case_id"])
            entry = self.mapping["entries"][item["review_item_id"]]
            self.assertEqual(entry["blinded_label"], item["blinded_label"])
            self.assertEqual(entry["case_id"], item["case_id"])
            self.assertEqual(entry["field"], item["field"])

    def test_shared_source_evidence_holds_full_transcript_and_target(self):
        for entry in self.packet["source_evidence"]:
            self.assertEqual(
                entry["transcript_sha256"],
                text_sha256(entry["transcript"]),
            )
            self.assertEqual(
                entry["transcript"][
                    entry["target_char_start"]:entry["target_char_end"]
                ],
                entry["target_text"],
            )

    def test_review_item_order_is_shuffled_not_condition_grouped(self):
        by_case: dict[str, list[str]] = {
            f"p2ja:case:{rank:04d}": [] for rank in range(1, 11)
        }
        for item in self.packet["review_items"]:
            by_case[item["case_id"]].append(
                self.mapping["entries"][item["review_item_id"]]["condition_code"],
            )
        for case_id, conditions in by_case.items():
            self.assertEqual(len(conditions), 16)
            self.assertNotEqual(
                conditions,
                ["A"] * 8 + ["B"] * 8,
            )
            self.assertNotEqual(
                conditions,
                ["B"] * 8 + ["A"] * 8,
            )
            self.assertEqual(conditions.count("A"), 8)
            self.assertEqual(conditions.count("B"), 8)

    def test_deterministic_blinding_without_public_seed(self):
        packet2, mapping2 = build_human_review_packet(
            outputs_artifact=self.outputs,
            payloads_artifact=self.payloads,
            transcript_rows={
                entry["video_id"]: {
                    "transcript": self.fixture["transcripts"][entry["video_id"]],
                    "transcript_sha256": text_sha256(
                        self.fixture["transcripts"][entry["video_id"]],
                    ),
                }
                for entry in self.payloads["provenance_by_case"].values()
            },
        )
        self.assertEqual(packet2["content_sha256"], self.packet["content_sha256"])
        self.assertEqual(mapping2["content_sha256"], self.mapping["content_sha256"])
        self.assertNotIn("seed", self.packet["blinding"])

    def test_leaked_condition_key_rejected(self):
        packet = json.loads(json.dumps(self.packet))
        packet["review_items"][0]["condition_code"] = "A"
        with self.assertRaises(ValueError):
            validate_human_review_packet(packet, require_blank=True)

    def test_mapping_tamper_rejected(self):
        mapping = json.loads(json.dumps(self.mapping))
        entry = next(iter(mapping["entries"].values()))
        entry["condition_code"] = "B" if entry["condition_code"] == "A" else "A"
        mapping["content_sha256"] = canonical_sha256({
            key: value for key, value in mapping.items()
            if key != "content_sha256"
        })
        with self.assertRaises(ValueError):
            validate_human_review_mapping(
                mapping,
                packet=self.packet,
                outputs_artifact=self.outputs,
                payloads_artifact=self.payloads,
            )

    def test_self_rehashed_semantic_swap_rejected(self):
        """Swapping A<->B condition codes and rehashing cannot pass."""
        mapping = json.loads(json.dumps(self.mapping))
        case_id = "p2ja:case:0001"
        field = "actors"
        entries_for_pair = [
            entry for entry in mapping["entries"].values()
            if entry["case_id"] == case_id and entry["field"] == field
        ]
        self.assertEqual(len(entries_for_pair), 2)
        by_condition = {entry["condition_code"]: entry for entry in entries_for_pair}
        by_condition["A"]["condition_code"] = "B"
        by_condition["B"]["condition_code"] = "A"
        mapping["content_sha256"] = canonical_sha256({
            key: value for key, value in mapping.items()
            if key != "content_sha256"
        })
        with self.assertRaises(ValueError):
            validate_human_review_mapping(
                mapping,
                packet=self.packet,
                outputs_artifact=self.outputs,
                payloads_artifact=self.payloads,
            )


class MaterialityTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.fixture = _build_full_fixture(self.root)
        self.payloads = self.fixture["payloads"]
        self.outputs = self.fixture["outputs"]
        self.packet, self.mapping = build_human_review_packet(
            outputs_artifact=self.outputs,
            payloads_artifact=self.payloads,
            transcript_rows={
                entry["video_id"]: {
                    "transcript": self.fixture["transcripts"][entry["video_id"]],
                    "transcript_sha256": text_sha256(
                        self.fixture["transcripts"][entry["video_id"]],
                    ),
                }
                for entry in self.payloads["provenance_by_case"].values()
            },
        )

    def tearDown(self):
        self._temporary.cleanup()

    def _run_materiality(
        self,
        a_per_case: list[int],
        b_per_case: list[int],
        major_b_items: set[str] | None = None,
    ):
        by_condition: dict[str, dict[str, list[str]]] = {
            code: {f"p2ja:case:{rank:04d}": [] for rank in range(1, 11)}
            for code in CONDITION_CODES
        }
        entries = self.mapping["entries"]
        for review_item_id, entry in entries.items():
            by_condition[entry["condition_code"]][entry["case_id"]].append(
                review_item_id,
            )
        success_by_item = {}
        for rank in range(1, 11):
            case_id = f"p2ja:case:{rank:04d}"
            for index, item_id in enumerate(by_condition["A"][case_id]):
                success_by_item[item_id] = index < a_per_case[rank - 1]
            for index, item_id in enumerate(by_condition["B"][case_id]):
                success_by_item[item_id] = index < b_per_case[rank - 1]
        completed = make_completed_reviews(
            self.packet,
            success_by_item=success_by_item,
            major_by_item=set(major_b_items or []),
        )
        finalized = import_completed_reviews(self.packet, completed)
        return compute_materiality(finalized, self.mapping), completed

    def test_all_fail_is_not_material(self):
        materiality, _ = self._run_materiality([0] * 10, [0] * 10)
        self.assertEqual(materiality["decision"], "NOT_MATERIAL")
        self.assertEqual(materiality["strict_success"], {"A": 0, "B": 0, "delta": 0})
        self.assertEqual(materiality["paired_field_judgments"], 160)

    def test_material_thresholds(self):
        a = [8, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        b = [0, 8, 8, 8, 8, 0, 0, 0, 0, 0]
        materiality, _ = self._run_materiality(a, b)
        self.assertEqual(materiality["decision"], "MATERIAL")
        self.assertEqual(materiality["strict_success"]["A"], 8)
        self.assertEqual(materiality["strict_success"]["B"], 32)
        self.assertEqual(materiality["strict_success"]["delta"], 24)
        self.assertEqual(materiality["case_wins"], {"A": 1, "B": 4, "ties": 5})

    def test_delta_below_threshold_not_material(self):
        a = [8, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        b = [0, 8, 8, 3, 0, 0, 0, 0, 0, 0]
        materiality, _ = self._run_materiality(a, b)
        self.assertEqual(materiality["decision"], "NOT_MATERIAL")
        self.assertEqual(materiality["strict_success"]["delta"], 11)

    def test_case_wins_below_threshold_not_material(self):
        a = [8, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        b = [0, 8, 8, 8, 0, 0, 0, 0, 0, 0]
        materiality, _ = self._run_materiality(a, b)
        self.assertEqual(materiality["decision"], "NOT_MATERIAL")
        self.assertEqual(materiality["case_wins"]["B"], 3)

    def test_a_wins_too_many_not_material(self):
        a = [8, 8, 0, 0, 0, 0, 0, 0, 0, 0]
        b = [0, 0, 8, 8, 8, 8, 0, 0, 0, 0]
        materiality, _ = self._run_materiality(a, b)
        self.assertEqual(materiality["decision"], "NOT_MATERIAL")
        self.assertEqual(materiality["case_wins"]["A"], 2)

    def test_major_unsupported_increase_not_material(self):
        entries = self.mapping["entries"]
        b_major = next(
            item_id for item_id, entry in entries.items()
            if entry["condition_code"] == "B"
        )
        a = [8, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        b = [0, 8, 8, 8, 8, 0, 0, 0, 0, 0]
        materiality, _ = self._run_materiality(
            a, b, major_b_items={b_major},
        )
        self.assertEqual(materiality["decision"], "NOT_MATERIAL")
        self.assertEqual(
            materiality["major_unsupported_inference"]["B"],
            1,
        )
        self.assertGreater(
            materiality["major_unsupported_inference"]["B"],
            materiality["major_unsupported_inference"]["A"],
        )

    def test_missing_human_attestation_rejected(self):
        completed = make_completed_reviews(self.packet, success_by_item={})
        del completed["reviewer_kind"]
        del completed["human_review_attested"]
        del completed["attestation_statement"]
        completed["content_sha256"] = canonical_sha256({
            key: value for key, value in completed.items()
            if key != "content_sha256"
        })
        with self.assertRaises(ValueError):
            import_completed_reviews(self.packet, completed)

    def test_non_human_attestation_rejected(self):
        completed = make_completed_reviews(
            self.packet,
            success_by_item={},
            reviewer_kind="bot",
        )
        with self.assertRaises(ValueError):
            import_completed_reviews(self.packet, completed)

    def test_attestation_false_rejected(self):
        completed = make_completed_reviews(
            self.packet,
            success_by_item={},
            human_review_attested=False,
        )
        with self.assertRaises(ValueError):
            import_completed_reviews(self.packet, completed)

    def test_finalized_packet_carries_explicit_attestation(self):
        completed = make_completed_reviews(self.packet, success_by_item={})
        finalized = import_completed_reviews(self.packet, completed)
        attestation = finalized["review_attestation"]
        self.assertEqual(attestation["reviewer_kind"], "human")
        self.assertIs(attestation["human_review_attested"], True)
        self.assertTrue(attestation["attestation_statement"])

    def test_summary_freeze_and_tamper(self):
        a = [8, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        b = [0, 8, 8, 8, 8, 0, 0, 0, 0, 0]
        materiality, completed = self._run_materiality(a, b)
        finalized = import_completed_reviews(self.packet, completed)
        selection = load_json_strict(
            self.root / "out/phase2j-context-ablation-selection-v1.json",
            label="selection",
        )
        instructions = load_json_strict(
            self.root / "out/phase2j-context-ablation-extraction-instructions-v2.json",
            label="instructions",
        )
        summary = build_materiality_summary(
            selection=selection,
            instructions=instructions,
            payloads=self.payloads,
            outputs=self.outputs,
            packet=self.packet,
            mapping=self.mapping,
            finalized_packet=finalized,
            completed=completed,
            materiality=materiality,
            frozen_at="2026-08-20T00:00:00Z",
        )
        validate_materiality_summary(
            summary,
            selection=selection,
            instructions=instructions,
            payloads=self.payloads,
            outputs=self.outputs,
            packet=self.packet,
            mapping=self.mapping,
            finalized_packet=finalized,
            completed=completed,
        )
        self.assertEqual(summary["decision"], "MATERIAL")
        self.assertEqual(summary["release_gate"], "LOCKED")
        self.assertEqual(summary["preregistered_policy"], MATERIALITY_POLICY)
        tampered = json.loads(json.dumps(summary))
        tampered["decision"] = "NOT_MATERIAL"
        with self.assertRaises(ValueError):
            validate_materiality_summary(
                tampered,
                selection=selection,
                instructions=instructions,
                payloads=self.payloads,
                outputs=self.outputs,
                packet=self.packet,
                mapping=self.mapping,
                finalized_packet=finalized,
                completed=completed,
            )

    def test_materiality_policy_uses_source_grounding(self):
        self.assertIn(
            "source_grounding",
            MATERIALITY_POLICY["strict_success"],
        )
        self.assertNotIn(
            "timestamp_grounding",
            MATERIALITY_POLICY["strict_success"],
        )


class DeepSeekGateTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.fixture = _build_full_fixture(self.root)
        self.payloads = self.fixture["payloads"]
        self.outputs = self.fixture["outputs"]
        self.packet, self.mapping = build_human_review_packet(
            outputs_artifact=self.outputs,
            payloads_artifact=self.payloads,
            transcript_rows={
                entry["video_id"]: {
                    "transcript": self.fixture["transcripts"][entry["video_id"]],
                    "transcript_sha256": text_sha256(
                        self.fixture["transcripts"][entry["video_id"]],
                    ),
                }
                for entry in self.payloads["provenance_by_case"].values()
            },
        )

    def tearDown(self):
        self._temporary.cleanup()

    def _material_summary(self) -> dict[str, Any]:
        a = [8, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        b = [0, 8, 8, 8, 8, 0, 0, 0, 0, 0]
        by_condition: dict[str, dict[str, list[str]]] = {
            code: {f"p2ja:case:{rank:04d}": [] for rank in range(1, 11)}
            for code in CONDITION_CODES
        }
        for review_item_id, entry in self.mapping["entries"].items():
            by_condition[entry["condition_code"]][entry["case_id"]].append(
                review_item_id,
            )
        success_by_item = {}
        for rank in range(1, 11):
            case_id = f"p2ja:case:{rank:04d}"
            for index, item_id in enumerate(by_condition["A"][case_id]):
                success_by_item[item_id] = index < a[rank - 1]
            for index, item_id in enumerate(by_condition["B"][case_id]):
                success_by_item[item_id] = index < b[rank - 1]
        completed = make_completed_reviews(
            self.packet, success_by_item=success_by_item,
        )
        finalized = import_completed_reviews(self.packet, completed)
        materiality = compute_materiality(finalized, self.mapping)
        selection = load_json_strict(
            self.root / "out/phase2j-context-ablation-selection-v1.json",
            label="selection",
        )
        instructions = load_json_strict(
            self.root / "out/phase2j-context-ablation-extraction-instructions-v2.json",
            label="instructions",
        )
        return build_materiality_summary(
            selection=selection,
            instructions=instructions,
            payloads=self.payloads,
            outputs=self.outputs,
            packet=self.packet,
            mapping=self.mapping,
            finalized_packet=finalized,
            completed=completed,
            materiality=materiality,
            frozen_at="2026-08-20T00:00:00Z",
        )

    def test_emit_requires_material_summary(self):
        with self.assertRaises(ValueError):
            build_deepseek_run_packet(
                summary={"release_gate": "AWAITING_HUMAN_REVIEW"},
                payloads_artifact=self.payloads,
            )
        summary = self._material_summary()
        self.assertEqual(summary["decision"], "MATERIAL")
        run_packet = build_deepseek_run_packet(
            summary=summary,
            payloads_artifact=self.payloads,
        )
        self.assertEqual(run_packet["condition"], "B")
        self.assertEqual(len(run_packet["cases"]), 10)
        self.assertEqual(
            run_packet["materiality_summary_sha256"],
            summary["content_sha256"],
        )

    def test_non_material_summary_stays_locked(self):
        summary = self._material_summary()
        tampered = json.loads(json.dumps(summary))
        tampered["decision"] = "NOT_MATERIAL"
        with self.assertRaises(ValueError):
            build_deepseek_run_packet(
                summary=tampered,
                payloads_artifact=self.payloads,
            )

    def test_import_requires_material_and_validated_outputs(self):
        summary = self._material_summary()
        run_packet = build_deepseek_run_packet(
            summary=summary,
            payloads_artifact=self.payloads,
        )
        outputs_by_case = {}
        for output_case, payload_case in zip(
            self.outputs["cases"], self.payloads["cases"],
        ):
            outputs_by_case[output_case["case_id"]] = make_valid_output(
                payload_case["B"],
                case_id=output_case["case_id"],
                condition="B",
            )
        imported = import_deepseek_run_outputs(
            summary=summary,
            run_packet=run_packet,
            outputs_by_case=outputs_by_case,
            payloads_artifact=self.payloads,
        )
        self.assertEqual(imported["release_gate"], "LOCKED")
        self.assertEqual(len(imported["cases"]), 10)
        validate_deepseek_import_artifact(
            imported,
            summary=summary,
            run_packet=run_packet,
            payloads_artifact=self.payloads,
        )
        bad_outputs = dict(outputs_by_case)
        bad_outputs["p2ja:case:0001"]["condition"] = "A"
        with self.assertRaises(ValueError):
            import_deepseek_run_outputs(
                summary=summary,
                run_packet=run_packet,
                outputs_by_case=bad_outputs,
                payloads_artifact=self.payloads,
            )

    def test_import_rejects_extra_and_missing_cases(self):
        summary = self._material_summary()
        run_packet = build_deepseek_run_packet(
            summary=summary,
            payloads_artifact=self.payloads,
        )
        outputs_by_case = {}
        for output_case, payload_case in zip(
            self.outputs["cases"], self.payloads["cases"],
        ):
            outputs_by_case[output_case["case_id"]] = make_valid_output(
                payload_case["B"],
                case_id=output_case["case_id"],
                condition="B",
            )
        with_extra = dict(outputs_by_case)
        with_extra["p2ja:case:0099"] = make_valid_output(
            self.payloads["cases"][0]["B"],
            case_id="p2ja:case:0099",
            condition="B",
        )
        with self.assertRaises(ValueError):
            import_deepseek_run_outputs(
                summary=summary,
                run_packet=run_packet,
                outputs_by_case=with_extra,
                payloads_artifact=self.payloads,
            )
        with_missing = {
            case_id: output
            for case_id, output in outputs_by_case.items()
            if case_id != "p2ja:case:0001"
        }
        with self.assertRaises(ValueError):
            import_deepseek_run_outputs(
                summary=summary,
                run_packet=run_packet,
                outputs_by_case=with_missing,
                payloads_artifact=self.payloads,
            )


if __name__ == "__main__":
    unittest.main()
