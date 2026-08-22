"""Focused tests for the Phase 2K full-transcript context ablation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline.phase2j_context_ablation import (
    canonical_sha256,
    file_sha256,
    text_sha256,
)
from pipeline.phase2k_full_transcript_ablation import (
    ArtifactError,
    INTERMEDIATE_SCHEMA_VERSION,
    RELATION_TYPES,
    SEMANTIC_FIELDS,
    build_build_summary,
    build_condition_payloads,
    build_extraction_instructions,
    build_intermediate_schema,
    build_outputs_bundle,
    build_payloads_artifact,
    build_review_packet,
    build_selection_artifact,
    compute_evaluation_summary,
    fetch_source_rows,
    import_intermediate_response,
    select_phase2k_cases,
    validate_completed_reviews,
    validate_extraction_output,
    validate_instructions_artifact,
    validate_intermediate_response,
    validate_outputs_bundle,
    validate_payload_pair_isolation,
    validate_payloads_artifact,
    validate_review_packet,
)
from pipeline.phase2j_context_ablation import load_lexical_vocabulary
from scripts.run_phase2k_full_transcript_ablation import (
    WRAPPER_PROMPT,
    call_prompt_bytes,
    cmd_build,
    cmd_evaluate,
    cmd_import,
    cmd_init,
    cmd_review,
    cmd_run,
    extract_response_json,
)
from tests._phase2k_full_transcript_ablation_helpers import (
    build_phase2k_fixture,
    expected_selected_window_ids,
    make_reviews,
    make_valid_intermediate_response,
)

REVIEW_FIELDS = SEMANTIC_FIELDS


_FIXTURE: dict | None = None


def fixture() -> dict:
    global _FIXTURE
    if _FIXTURE is None:
        _FIXTURE = build_phase2k_fixture(
            Path(tempfile.mkdtemp(prefix="p2k-ablation-tests-")),
        )
        _FIXTURE["selection_artifact"] = build_selection_artifact(
            manifest_path=_FIXTURE["manifest_path"],
            manifest=_FIXTURE["manifest"],
            db_path=_FIXTURE["db_path"],
            cases=_FIXTURE["cases"],
        )
        _FIXTURE["payloads_artifact"] = build_payloads_artifact(
            selection=_FIXTURE["selection_artifact"],
            instructions=_FIXTURE["instructions"],
            payload_cases=_FIXTURE["payload_cases"],
            provenance_by_case=_FIXTURE["provenance_by_case"],
        )
    return _FIXTURE


class SelectionTests(unittest.TestCase):
    def test_selection_is_deterministic(self) -> None:
        data = fixture()
        from pipeline.phase2j_context_ablation import open_transcript_db

        connection = open_transcript_db(data["db_path"])
        try:
            rows_a = fetch_source_rows(connection, data["manifest"]["selected"])
            rows_b = fetch_source_rows(connection, data["manifest"]["selected"])
            cases_a = select_phase2k_cases(data["manifest"], source_rows=rows_a)
            cases_b = select_phase2k_cases(data["manifest"], source_rows=rows_b)
        finally:
            connection.close()
        self.assertEqual(cases_a, cases_b)

    def test_selection_matches_expected_top10_window_ids(self) -> None:
        data = fixture()
        expected = expected_selected_window_ids()
        observed = [case["upstream_source_id"] for case in data["cases"]]
        self.assertEqual(observed, expected)
        self.assertEqual(len(observed), 10)

    def test_case_ids_are_p2k_prefixed_and_traceable(self) -> None:
        data = fixture()
        for rank, case in enumerate(data["cases"], 1):
            self.assertEqual(case["case_id"], f"p2k:case:{rank:04d}")
            self.assertEqual(
                case["phase2j_case_id"], f"p2ja:case:{rank:04d}",
            )

    def test_target_bronze_text_is_unchanged_from_manifest(self) -> None:
        data = fixture()
        selected = data["manifest"]["selected"]
        for case in data["cases"]:
            manifest_row = selected[case["manifest_index"]]
            self.assertEqual(
                case["bronze_text_sha256"],
                manifest_row["source_text_sha256"],
            )
            self.assertEqual(case["window_id"], manifest_row["window_id"])
            self.assertEqual(
                case["upstream_source_id"],
                manifest_row["upstream_source_id"],
            )


class PayloadIsolationTests(unittest.TestCase):
    def test_shared_fields_are_byte_identical_across_conditions(self) -> None:
        for pair in fixture()["payload_cases"]:
            payload_a, payload_b = pair["A"], pair["B"]
            shared_keys = (
                "schema_version", "case_id", "selection_rank", "target",
                "metadata", "metadata_fields_supplied",
                "vocabulary", "vocabulary_sha256",
                "instructions", "instructions_sha256",
            )
            for key in shared_keys:
                self.assertEqual(
                    canonical_sha256(payload_a[key]),
                    canonical_sha256(payload_b[key]),
                    f"key {key!r} differs between A and B",
                )

    def test_condition_b_only_adds_discourse_context(self) -> None:
        for pair in fixture()["payload_cases"]:
            payload_a, payload_b = pair["A"], pair["B"]
            self.assertNotIn("transcript", payload_a)
            self.assertNotIn("target_char_start", payload_a)
            self.assertNotIn("target_char_end", payload_a)
            extra = set(payload_b) - set(payload_a)
            self.assertEqual(
                extra,
                {"transcript", "target_char_start", "target_char_end"},
            )
            validate_payload_pair_isolation(pair)

    def test_target_location_valid_in_full_transcript(self) -> None:
        for pair in fixture()["payload_cases"]:
            payload_b = pair["B"]
            bronze = payload_b["target"]["bronze_text"]
            start = payload_b["target_char_start"]
            end = payload_b["target_char_end"]
            transcript = payload_b["transcript"]
            self.assertTrue(0 <= start < end <= len(transcript))
            self.assertEqual(transcript[start:end], bronze)

    def test_no_cleaning_applied_transcript_is_byte_exact(self) -> None:
        data = fixture()
        transcripts = data["transcripts"]
        for pair in data["payload_cases"]:
            source_id = data["provenance_by_case"][pair["case_id"]]["video_id"]
            self.assertEqual(
                pair["B"]["transcript"],
                transcripts[source_id],
                "condition B transcript must be the raw archived transcript",
            )

    def test_condition_a_rejects_identity_leak(self) -> None:
        data = fixture()
        instructions = data["instructions"]
        lexical = load_lexical_vocabulary(data["vocabulary_path"])
        from pipeline.phase2j_context_ablation import open_transcript_db

        connection = open_transcript_db(data["db_path"])
        try:
            pair = data["payload_cases"][0]
            tampered = json.loads(json.dumps(pair["A"]))
            del tampered["content_sha256"]
            tampered["video_url"] = "https://www.youtube.com/watch?v=x"
            envelope = {
                **tampered,
                "content_sha256": canonical_sha256(tampered),
            }
            with self.assertRaises(ValueError):
                validate_payloads_artifact(
                    {
                        "schema_version": "phase2k-full-transcript-ablation-condition-payloads-v1",
                        "purpose": "x",
                        "release_gate": "LOCKED",
                        "pipeline_version": envelope["schema_version"].replace("payload-v1", "payloads-v1"),
                        "selection_sha256": "0" * 64,
                        "instructions_sha256": envelope["instructions_sha256"],
                        "provenance_by_case": {},
                        "cases": [{"case_id": pair["case_id"], "selection_rank": 1, "A": envelope, "B": pair["B"]}],
                        "content_sha256": "",
                    },
                    selection={"content_sha256": "0" * 64, "cases": []},
                    instructions=instructions,
                    lexical_vocabulary=lexical,
                    manifest=data["manifest"],
                    connection=connection,
                )
        finally:
            connection.close()

    def test_metadata_policy_records_supplied_fields(self) -> None:
        for pair in fixture()["payload_cases"]:
            payload = pair["A"]
            metadata = payload["metadata"]
            supplied = payload["metadata_fields_supplied"]
            non_null = sorted(
                key
                for key, value in metadata.items()
                if isinstance(value, str) and value.strip()
            )
            self.assertEqual(sorted(supplied), non_null)
            # The synthetic DB always supplies descriptions.
            self.assertIn("description", supplied)


class ParsingTests(unittest.TestCase):
    def _import_pair(self, pair: dict) -> dict:
        output = import_intermediate_response(
            make_valid_intermediate_response(pair["A"]),
            case_id=pair["case_id"],
            condition="A",
            payload=pair["A"],
        )
        validate_extraction_output(
            output,
            case_id=pair["case_id"],
            condition="A",
            payload=pair["A"],
        )
        return output

    def test_valid_response_imports_with_resolved_ranges(self) -> None:
        pair = fixture()["payload_cases"][0]
        payload = pair["A"]
        response = make_valid_intermediate_response(payload)
        validate_intermediate_response(
            response,
            case_id=payload["case_id"],
            condition="A",
            payload=payload,
        )
        output = self._import_pair(pair)
        bronze = payload["target"]["bronze_text"]
        item = output["fields"]["actors_entities"][0]
        reference = item["source_references"][0]
        start = reference["source_range"]["char_start"]
        end = reference["source_range"]["char_end"]
        self.assertEqual(bronze[start:end], reference["quote"])
        self.assertTrue(item["item_id"].endswith("actors_entities:0001"))

    def test_unresolved_status_survives_parsing(self) -> None:
        pair = fixture()["payload_cases"][1]
        output = self._import_pair(pair)
        binding = output["fields"]["reference_bindings"][0]
        self.assertEqual(binding["resolution_status"], "unresolved")

    def test_support_span_bounding_range_derived(self) -> None:
        pair = fixture()["payload_cases"][2]
        output = self._import_pair(pair)
        span_item = output["fields"]["supporting_source_spans"][0]
        starts = [
            ref["source_range"]["char_start"]
            for ref in span_item["source_references"]
        ]
        ends = [
            ref["source_range"]["char_end"]
            for ref in span_item["source_references"]
        ]
        self.assertEqual(span_item["source_range"]["char_start"], min(starts))
        self.assertEqual(span_item["source_range"]["char_end"], max(ends))

    def test_missing_field_rejected(self) -> None:
        pair = fixture()["payload_cases"][0]
        payload = pair["A"]
        response = make_valid_intermediate_response(payload)
        del response["fields"]
        with self.assertRaises(ValueError):
            validate_intermediate_response(
                response,
                case_id=payload["case_id"],
                condition="A",
                payload=payload,
            )

    def test_extra_key_rejected(self) -> None:
        pair = fixture()["payload_cases"][0]
        payload = pair["A"]
        response = make_valid_intermediate_response(payload)
        response["surprise"] = True
        with self.assertRaises(ValueError):
            validate_intermediate_response(
                response,
                case_id=payload["case_id"],
                condition="A",
                payload=payload,
            )

    def test_relation_type_vocabulary_enforced(self) -> None:
        pair = fixture()["payload_cases"][0]
        payload = pair["A"]
        response = make_valid_intermediate_response(payload)
        response["fields"]["explicit_relationships"][0]["relation_type"] = (
            "BEATDOWN"
        )
        with self.assertRaises(ValueError):
            validate_intermediate_response(
                response,
                case_id=payload["case_id"],
                condition="A",
                payload=payload,
            )
        self.assertIn("NEGATES", RELATION_TYPES)
        self.assertNotIn("BEATDOWN", RELATION_TYPES)

    def test_quote_occurrence_out_of_range_rejected(self) -> None:
        pair = fixture()["payload_cases"][0]
        payload = pair["A"]
        response = make_valid_intermediate_response(payload)
        response["fields"]["actors_entities"][0]["source_references"][0][
            "occurrence_index"
        ] = 99
        with self.assertRaises(ValueError):
            import_intermediate_response(
                response,
                case_id=payload["case_id"],
                condition="A",
                payload=payload,
            )

    def test_wrong_payload_binding_rejected(self) -> None:
        pair = fixture()["payload_cases"][3]
        payload_a, payload_b = pair["A"], pair["B"]
        response = make_valid_intermediate_response(payload_a)
        with self.assertRaises(ValueError):
            validate_intermediate_response(
                response,
                case_id=payload_a["case_id"],
                condition="B",
                payload=payload_b,
            )

    def test_extract_response_json_fences_and_failures(self) -> None:
        good = json.dumps({"a": 1})
        self.assertEqual(extract_response_json(good), {"a": 1})
        self.assertEqual(
            extract_response_json("```json\n" + good + "\n```"), {"a": 1},
        )
        self.assertEqual(
            extract_response_json("```\n" + good + "\n```"), {"a": 1},
        )
        with self.assertRaises(ArtifactError):
            extract_response_json("```yaml\n" + good + "\n```")
        with self.assertRaises(ValueError):
            extract_response_json("this is not json")
        with self.assertRaises(ValueError):
            extract_response_json(good + " trailing text")


class ProvenanceTests(unittest.TestCase):
    def test_outputs_bundle_carries_bindings(self) -> None:
        data = fixture()
        outputs_by_call = {}
        for pair in data["payload_cases"]:
            for condition in ("A", "B"):
                output = import_intermediate_response(
                    make_valid_intermediate_response(pair[condition]),
                    case_id=pair["case_id"],
                    condition=condition,
                    payload=pair[condition],
                )
                outputs_by_call[(pair["case_id"], condition)] = output
        bundle = build_outputs_bundle(
            payloads=data["payloads_artifact"],
            outputs_by_call=outputs_by_call,
        )
        validate_outputs_bundle(bundle, payloads=data["payloads_artifact"])
        self.assertEqual(len(bundle["cases"]), 10)
        for entry in bundle["cases"]:
            for condition in ("A", "B"):
                output = entry[condition]
                self.assertEqual(
                    output["payload_sha256"],
                    data["payloads_artifact"]["cases"][
                        int(entry["case_id"][-4:]) - 1
                    ][condition]["content_sha256"],
                )
                self.assertEqual(
                    output["instructions_sha256"],
                    canonical_sha256(data["instructions"]),
                )

    def test_build_summary_lists_artifacts(self) -> None:
        data = fixture()
        summary = build_build_summary(
            output_dir=Path(tempfile.mkdtemp(prefix="p2k-summary-")),
            selection=data["selection_artifact"],
            instructions=data["instructions"],
            payloads=data["payloads_artifact"],
            mode="ready_for_opencode",
        )
        self.assertEqual(summary["mode"], "ready_for_opencode")
        self.assertEqual(len(summary["artifacts"]), 3)
        self.assertEqual(len(summary["selected_case_ids"]), 10)

    def test_intermediate_schema_is_canonical(self) -> None:
        from pipeline.phase2k_full_transcript_ablation import (
            validate_intermediate_schema,
        )

        schema = build_intermediate_schema()
        validate_intermediate_schema(schema)
        tampered = json.loads(json.dumps(schema))
        tampered["title"] = "tampered"
        with self.assertRaises(ArtifactError):
            validate_intermediate_schema(tampered)

    def test_instructions_validation_catches_tamper(self) -> None:
        data = fixture()
        instructions = data["instructions"]
        validate_instructions_artifact(instructions)
        tampered = json.loads(json.dumps(instructions))
        del tampered["content_sha256"]
        tampered["forbidden_processing"] = ["nothing"]
        envelope = {**tampered, "content_sha256": canonical_sha256(tampered)}
        with self.assertRaises(ArtifactError):
            validate_instructions_artifact(envelope)

    def test_prompt_bytes_bind_wrapper_and_payload(self) -> None:
        pair = fixture()["payload_cases"][0]
        prompt_bytes = call_prompt_bytes(pair["A"])
        text = prompt_bytes.decode("utf-8")
        self.assertTrue(text.startswith(WRAPPER_PROMPT + "\n\n"))
        embedded = text[len(WRAPPER_PROMPT) + 2:]
        self.assertEqual(embedded, json.dumps(
            pair["A"], sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ))


def all_success_fields(prefix: str) -> set[str]:
    return {
        f"{case['case_id']}:{prefix}:{field}"
        for case in fixture()["cases"]
        for field in REVIEW_FIELDS
    }


class EvaluationGateTests(unittest.TestCase):
    def _packet_and_bundle(self) -> tuple[dict, dict]:
        data = fixture()
        outputs_by_call = {}
        for pair in data["payload_cases"]:
            for condition in ("A", "B"):
                outputs_by_call[(pair["case_id"], condition)] = (
                    import_intermediate_response(
                        make_valid_intermediate_response(pair[condition]),
                        case_id=pair["case_id"],
                        condition=condition,
                        payload=pair[condition],
                    )
                )
        bundle = build_outputs_bundle(
            payloads=data["payloads_artifact"],
            outputs_by_call=outputs_by_call,
        )
        packet = build_review_packet(
            payloads=data["payloads_artifact"], outputs=bundle,
        )
        validate_review_packet(packet)
        return data, packet

    def test_material_outcome_full_context_wins(self) -> None:
        _, packet = self._packet_and_bundle()
        reviews = make_reviews(
            packet,
            b_success_fields=all_success_fields("B"),
        )
        validate_completed_reviews(reviews, review_packet=packet)
        summary = compute_evaluation_summary(
            review_packet=packet, completed_reviews=reviews,
        )
        self.assertEqual(
            summary["decision_gate"],
            "ISOLATED_BRONZE_WAS_THE_WRONG_SEMANTIC_UNIT",
        )

    def test_weak_outcome_small_improvement(self) -> None:
        _, packet = self._packet_and_bundle()
        b_all = all_success_fields("B")
        a_all = all_success_fields("A")
        weak_a = {
            key for key in a_all if not key.endswith(":states")
        }
        reviews = make_reviews(
            packet, a_success_fields=weak_a, b_success_fields=b_all,
        )
        summary = compute_evaluation_summary(
            review_packet=packet, completed_reviews=reviews,
        )
        self.assertEqual(
            summary["decision_gate"],
            "GLOBAL_CONTEXT_HELPS_BUT_DOES_NOT_SOLVE_SOURCE_RECOVERY",
        )

    def test_no_improvement_outcome(self) -> None:
        _, packet = self._packet_and_bundle()
        same_b = all_success_fields("B")
        same_a = {key.replace(":B:", ":A:") for key in same_b}
        reviews = make_reviews(
            packet, a_success_fields=same_a, b_success_fields=same_b,
        )
        summary = compute_evaluation_summary(
            review_packet=packet, completed_reviews=reviews,
        )
        self.assertEqual(
            summary["decision_gate"],
            "FULL_TRANSCRIPT_CONTEXT_DOES_NOT_EXPLAIN_THE_FAILURE",
        )

    def test_agent_reviewer_kind_accepted_human_requires_attestation(
        self,
    ) -> None:
        _, packet = self._packet_and_bundle()
        agent_reviews = make_reviews(packet, reviewer_kind="agent")
        validate_completed_reviews(agent_reviews, review_packet=packet)
        human_missing = make_reviews(packet, reviewer_kind="human")
        del human_missing["attestation_statement"]
        recomputed = canonical_sha256({
            key: value
            for key, value in human_missing.items()
            if key != "content_sha256"
        })
        human_missing["content_sha256"] = recomputed
        with self.assertRaises(ValueError):
            validate_completed_reviews(human_missing, review_packet=packet)

    def test_tampered_summary_fails_validation(self) -> None:
        from pipeline.phase2k_full_transcript_ablation import (
            validate_evaluation_summary,
        )

        _, packet = self._packet_and_bundle()
        reviews = make_reviews(
            packet, b_success_fields=all_success_fields("B"),
        )
        summary = compute_evaluation_summary(
            review_packet=packet, completed_reviews=reviews,
        )
        validate_evaluation_summary(
            summary, review_packet=packet, completed_reviews=reviews,
        )
        tampered = json.loads(json.dumps(summary))
        tampered["decision_gate"] = "TAMPERED"
        del tampered["content_sha256"]
        tampered["content_sha256"] = canonical_sha256(tampered)
        with self.assertRaises(ArtifactError):
            validate_evaluation_summary(
                tampered, review_packet=packet, completed_reviews=reviews,
            )


class EndToEndCliTests(unittest.TestCase):
    """Offline CLI flow: build → init → simulate → run/import/review/evaluate."""

    def test_full_offline_pipeline_flow(self) -> None:
        data = fixture()
        script = __import__(
            "scripts.run_phase2k_full_transcript_ablation",
            fromlist=["run_phase2k_full_transcript_ablation"],
        )
        workspace = Path(tempfile.mkdtemp(prefix="p2k-cli-"))
        out_dir = workspace / "out"
        run_dir = workspace / "run"

        def args(**kwargs: object) -> object:
            return mock.Mock(**{
                "output_dir": str(out_dir),
                "db": str(data["db_path"]),
                "vocabulary": str(data["vocabulary_path"]),
                "run_dir": str(run_dir),
                "max_workers": 1,
                "retries": 0,
                "force": False,
                **kwargs,
            })

        with mock.patch.object(
            script,
            "DEFAULT_MANIFEST_PATH",
            data["manifest_path"],
        ):
            cmd_build(args())
            cmd_init(args(run_dir=str(run_dir)))

            payloads = json.loads(
                (out_dir / script.PAYLOADS_FILENAME).read_text(encoding="utf-8"),
            )
            manifest = json.loads(
                (run_dir / script.MANIFEST_FILENAME).read_text(encoding="utf-8"),
            )
            self.assertEqual(len(manifest["calls"]), 20)
            self.assertEqual(manifest["requested_model"], script.MODEL)
            by_id = {pair["case_id"]: pair for pair in payloads["cases"]}
            raw_dir = run_dir / script.RAW_SUBDIR
            raw_dir.mkdir(parents=True, exist_ok=True)
            for call in manifest["calls"]:
                payload = by_id[call["case_id"]][call["condition"]]
                response = make_valid_intermediate_response(payload)
                raw_text = json.dumps(response)
                raw_file = raw_dir / (
                    f"{call['case_id']}_{call['condition']}.json"
                )
                raw_file.write_text(raw_text, encoding="utf-8")
                call.update({
                    "status": "completed",
                    "raw_path": str(raw_file.relative_to(run_dir)),
                    "raw_response_sha256": file_sha256(raw_file),
                    "completed_at": "2026-08-22T00:00:00Z",
                })
            (run_dir / script.MANIFEST_FILENAME).write_text(
                json.dumps(manifest), encoding="utf-8",
            )

            # Resume: a completed valid call must never be re-executed.
            with mock.patch.object(
                script.subprocess,
                "run",
                side_effect=AssertionError("opencode must not be called"),
            ):
                cmd_run(args(run_dir=str(run_dir)))
            status = cmd_status_capture(script, run_dir)
            self.assertIn("20/20 completed", status)

            # Cache invalidation: tampering one raw file fails closed.
            victim = manifest["calls"][0]
            victim_raw = run_dir / victim["raw_path"]
            original = victim_raw.read_text(encoding="utf-8")
            victim_raw.write_text(original.replace('"', "'"), encoding="utf-8")
            with mock.patch.object(
                script.subprocess,
                "run",
                side_effect=AssertionError("opencode must not be called"),
            ):
                with self.assertRaises(SystemExit):
                    cmd_run(args(run_dir=str(run_dir)))
            # --force replaces the exact per-call artifact via opencode.
            def fake_opencode(argv, **kwargs):  # noqa: ANN001
                payload = by_id[victim["case_id"]][victim["condition"]]
                response = make_valid_intermediate_response(payload)
                return mock.Mock(returncode=0, stdout=json.dumps(response).encode())

            with mock.patch.object(script.subprocess, "run", fake_opencode):
                cmd_run(args(run_dir=str(run_dir), force=True))

            # Prompt drift fails closed.
            drifted = json.loads(
                (run_dir / script.MANIFEST_FILENAME).read_text(encoding="utf-8"),
            )
            drifted["calls"][1]["prompt_sha256"] = "f" * 64
            (run_dir / script.MANIFEST_FILENAME).write_text(
                json.dumps(drifted), encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                cmd_run(args(run_dir=str(run_dir)))

            cmd_import(args(run_dir=str(run_dir)))
            outputs_path = out_dir / script.OUTPUTS_FILENAME
            self.assertTrue(outputs_path.exists())

            cmd_review(args())
            packet_path = out_dir / script.REVIEW_PACKET_FILENAME
            markdown_path = out_dir / script.REVIEW_MARKDOWN_FILENAME
            self.assertTrue(packet_path.exists())
            self.assertTrue(markdown_path.exists())

            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            reviews = make_reviews(
                packet, b_success_fields=all_success_fields("B"),
            )
            reviews_path = workspace / "reviews.json"
            reviews_path.write_text(json.dumps(reviews), encoding="utf-8")
            cmd_evaluate(args(reviews=str(reviews_path)))
            summary = json.loads(
                (out_dir / script.EVALUATION_SUMMARY_FILENAME).read_text(
                    encoding="utf-8",
                ),
            )
            self.assertEqual(
                summary["decision_gate"],
                "ISOLATED_BRONZE_WAS_THE_WRONG_SEMANTIC_UNIT",
            )


def cmd_status_capture(script: object, run_dir: Path) -> str:
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        script.cmd_status(mock.Mock(run_dir=str(run_dir)))
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
