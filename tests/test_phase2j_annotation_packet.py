import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.phase2j_annotation_packet import (
    AUDIT_CHECKS,
    PACKET_SCHEMA_VERSION,
    _validate_forbidden_content,
    build_annotation_packet,
    is_packet_gold_eligible,
    is_window_gold_eligible,
    load_annotation_packet,
    token_table,
    validate_annotation_packet,
)
from pipeline.phase2j_source_selection import canonical_sha256
from tests._phase2j_helpers import (
    rehash_manifest,
    rehash_packet,
    rehash_record,
    write_standard_phase2j_inputs,
)


def _build_packet(root: Path):
    pool_path, manifest_path, benchmark_path, _, _ = write_standard_phase2j_inputs(root)
    from pipeline.semantic_ir_pool import load_semantic_window_pool
    from pipeline.phase2j_source_selection import (
        build_selection_manifest,
        load_legacy_benchmark,
        load_legacy_manifest,
    )

    pool = load_semantic_window_pool(pool_path)
    legacy_manifest = load_legacy_manifest(manifest_path)
    legacy_benchmark = load_legacy_benchmark(benchmark_path, manifest=legacy_manifest)
    manifest = build_selection_manifest(
        pool=pool,
        pool_path=pool_path,
        legacy_manifest=legacy_manifest,
        legacy_manifest_path=manifest_path,
        legacy_benchmark=legacy_benchmark,
        legacy_benchmark_path=benchmark_path,
    )
    packet = build_annotation_packet(manifest)
    validate_annotation_packet(packet, manifest=manifest)
    return manifest, packet


def _reviewed_record(record):
    value = copy.deepcopy(dict(record))
    value["window_status"] = "REVIEWED"
    value["pass_a"] = {
        "status": "COMPLETE",
        "reviewer": "reviewer-a",
        "completed_at": "2026-08-18T00:00:00Z",
        "notes": [],
        "endpoint_count": 0,
    }
    value["pass_b"] = {
        "status": "COMPLETE",
        "reviewer": "reviewer-b",
        "completed_at": "2026-08-18T01:00:00Z",
        "notes": [],
        "audit_checks": {key: True for key in AUDIT_CHECKS},
    }
    return value


def _in_review_record(
    record,
    *,
    pass_a_status,
    pass_b_status=None,
    endpoints=(),
    ambiguity_flagged=False,
    exclusion_flagged=False,
):
    """Build a schema-shaped IN_REVIEW record for transition tests."""
    value = copy.deepcopy(dict(record))
    value["window_status"] = "IN_REVIEW"
    pass_a = {
        "status": pass_a_status,
        "reviewer": None,
        "completed_at": None,
        "notes": [],
        "endpoint_count": len(endpoints),
    }
    if pass_a_status == "COMPLETE":
        pass_a["reviewer"] = "reviewer-a"
        pass_a["completed_at"] = "2026-08-18T00:00:00Z"
    value["pass_a"] = pass_a
    if pass_b_status is None:
        pass_b_status = (
            "LOCKED_AWAITING_PASS_A"
            if pass_a_status != "COMPLETE"
            else "PENDING"
        )
    value["pass_b"] = {
        "status": pass_b_status,
        "reviewer": None,
        "completed_at": None,
        "notes": [],
        "audit_checks": {key: False for key in AUDIT_CHECKS},
    }
    value["endpoints"] = list(endpoints)
    value["ambiguity_controls"] = {
        "flagged": ambiguity_flagged,
        "notes": [] if not ambiguity_flagged else ["ambiguous"],
    }
    value["exclusion_controls"] = {
        "flagged": exclusion_flagged,
        "notes": [] if not exclusion_flagged else ["excluded"],
    }
    return value


def _keep_endpoint(window_id, index, char_start, char_end, token_start, token_end, **extra):
    return {
        "endpoint_id": f"p2j:{window_id}:ep:{index:04d}",
        "bronze_text": "",
        "char_start": char_start,
        "char_end": char_end,
        "token_start": token_start,
        "token_end": token_end,
        "node_type": "ENTITY",
        "ambiguity_state": "NONE",
        "disposition": "KEEP",
        "adjudication_requested": False,
        "notes": "",
        "pass_provenance": "PASS_A",
        **extra,
    }


def _endpoint_from_tokens(record, token_start, token_end, *, index=1, **extra):
    """Build a valid endpoint entry from token indices in the record's table."""
    tokens = record["tokens"]
    char_start = tokens[token_start]["start"]
    char_end = tokens[token_end]["end"]
    return _keep_endpoint(
        record["window_id"], index, char_start, char_end,
        token_start, token_end,
        bronze_text=record["bronze_text"][char_start:char_end],
        **extra,
    )


def _packet_with_record(packet, index, record):
    value = copy.deepcopy(dict(packet))
    records = [dict(item) for item in value["records"]]
    records[index - 1] = record
    value["records"] = records
    return rehash_packet(value)


class Phase2JAnnotationPacketTests(unittest.TestCase):
    def test_blank_packet_schema_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, packet = _build_packet(Path(temporary))
            self.assertEqual(packet["schema_version"], PACKET_SCHEMA_VERSION)
            self.assertEqual(packet["release_gate"], "LOCKED")
            self.assertEqual(packet["selection_manifest_sha256"], manifest["content_sha256"])
            self.assertEqual(len(packet["records"]), 30)
            for record in packet["records"]:
                self.assertEqual(record["window_status"], "UNREVIEWED")
                self.assertEqual(record["endpoints"], [])
                self.assertEqual(record["pass_a"]["status"], "PENDING")
                self.assertEqual(record["pass_b"]["status"], "LOCKED_AWAITING_PASS_A")
                self.assertEqual(record["bronze_char_length"], len(record["bronze_text"]))
                self.assertTrue(record["tokens"])
                self.assertEqual(record["tokens"][-1]["end"], len(record["bronze_text"]))
            self.assertFalse(is_packet_gold_eligible(packet))
            self.assertFalse(any(is_window_gold_eligible(r) for r in packet["records"]))

    def test_exact_bronze_token_span_validation_and_bool_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, packet = _build_packet(Path(temporary))
            record = _reviewed_record(packet["records"][0])
            endpoint = _endpoint_from_tokens(record, 0, 1)
            record["endpoints"] = [endpoint]
            record["pass_a"]["endpoint_count"] = 1
            valid = _packet_with_record(packet, 1, record)
            validate_annotation_packet(valid)
            self.assertTrue(is_window_gold_eligible(valid["records"][0]))

            def with_endpoint(mutator):
                value = copy.deepcopy(record)
                value["endpoints"] = [dict(endpoint)]
                value["pass_a"]["endpoint_count"] = 1
                mutator(value["endpoints"][0])
                return _packet_with_record(packet, 1, value)

            bad_span = with_endpoint(lambda item: item.update({"char_start": 5, "char_end": 2}))
            with self.assertRaisesRegex(ValueError, "character offsets"):
                validate_annotation_packet(bad_span)
            bad_token = with_endpoint(lambda item: item.update({"token_end": 2}))
            with self.assertRaisesRegex(ValueError, "token boundaries"):
                validate_annotation_packet(bad_token)
            bad_text = with_endpoint(lambda item: item.update({"bronze_text": "Lux"}))
            with self.assertRaisesRegex(ValueError, "exact Bronze slice"):
                validate_annotation_packet(bad_text)
            bool_offset = with_endpoint(lambda item: item.update({"char_start": True}))
            with self.assertRaisesRegex(ValueError, "character offsets"):
                validate_annotation_packet(bool_offset)
            bool_token = with_endpoint(lambda item: item.update({"token_start": False}))
            with self.assertRaisesRegex(ValueError, "token offsets"):
                validate_annotation_packet(bool_token)

    def test_two_pass_transition_and_gold_eligibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, packet = _build_packet(Path(temporary))
            record = _reviewed_record(packet["records"][0])
            # Pass B cannot complete before valid Pass A.
            record["pass_a"]["status"] = "PENDING"
            record["pass_a"]["reviewer"] = None
            record["pass_a"]["completed_at"] = None
            with self.assertRaisesRegex(ValueError, "cannot start before pass_a"):
                validate_annotation_packet(_packet_with_record(packet, 1, record))

            # REVIEWED requires both passes complete.
            record = _reviewed_record(packet["records"][0])
            record["pass_b"]["status"] = "PENDING"
            record["pass_b"]["reviewer"] = None
            record["pass_b"]["completed_at"] = None
            with self.assertRaisesRegex(ValueError, "clean two-pass completion"):
                validate_annotation_packet(_packet_with_record(packet, 1, record))

            # AMBIGUOUS endpoint requires an AMBIGUOUS flagged window.
            record = _reviewed_record(packet["records"][0])
            record["window_status"] = "AMBIGUOUS"
            record["pass_b"]["status"] = "PENDING"
            record["pass_b"]["reviewer"] = None
            record["pass_b"]["completed_at"] = None
            record["ambiguity_controls"]["flagged"] = True
            endpoint = _endpoint_from_tokens(
                record, 0, 1,
                ambiguity_state="AMBIGUOUS", disposition="AMBIGUOUS",
                adjudication_requested=False, pass_provenance="PASS_A",
            )
            record["endpoints"] = [endpoint]
            record["pass_a"]["endpoint_count"] = 1
            ambiguous = _packet_with_record(packet, 1, record)
            validate_annotation_packet(ambiguous)
            self.assertFalse(is_window_gold_eligible(ambiguous["records"][0]))

            # EXCLUDED windows must be empty and flagged, never gold-eligible.
            record = _reviewed_record(packet["records"][0])
            record["window_status"] = "EXCLUDED"
            record["pass_b"]["status"] = "PENDING"
            record["pass_b"]["reviewer"] = None
            record["pass_b"]["completed_at"] = None
            record["exclusion_controls"]["flagged"] = True
            excluded = _packet_with_record(packet, 1, record)
            validate_annotation_packet(excluded)
            self.assertFalse(is_window_gold_eligible(excluded["records"][0]))

            # KEEP entry must not carry ambiguity/adjudication (no silent KEEP).
            record = _reviewed_record(packet["records"][0])
            endpoint = _endpoint_from_tokens(record, 0, 1, ambiguity_state="AMBIGUOUS")
            record["endpoints"] = [endpoint]
            record["pass_a"]["endpoint_count"] = 1
            with self.assertRaisesRegex(ValueError, "cannot carry ambiguity"):
                validate_annotation_packet(_packet_with_record(packet, 1, record))

    def test_pass_b_endpoint_provenance_requires_active_pass_b(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, packet = _build_packet(Path(temporary))
            record = _reviewed_record(packet["records"][0])
            record["pass_b"].update({
                "status": "PENDING",
                "reviewer": None,
                "completed_at": None,
            })
            endpoint = _endpoint_from_tokens(
                record, 0, 1, pass_provenance="PASS_B",
            )
            record["endpoints"] = [endpoint]
            record["pass_a"]["endpoint_count"] = 1
            with self.assertRaisesRegex(ValueError, "requires an active pass_b"):
                validate_annotation_packet(_packet_with_record(packet, 1, record))

    def test_duplicate_and_overlapping_endpoint_handling(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, packet = _build_packet(Path(temporary))
            record = _reviewed_record(packet["records"][0])
            first = _endpoint_from_tokens(record, 0, 1, index=1)
            second = _endpoint_from_tokens(record, 1, 2, index=2)
            record["endpoints"] = [first, second]
            record["pass_a"]["endpoint_count"] = 2
            with self.assertRaisesRegex(ValueError, "duplicate/overlapping"):
                validate_annotation_packet(_packet_with_record(packet, 1, record))

            # Explicit adjudication permits overlap, requires adjudication window.
            record = _reviewed_record(packet["records"][0])
            record["window_status"] = "ADJUDICATION_REQUIRED"
            record["pass_b"]["status"] = "PENDING"
            record["pass_b"]["reviewer"] = None
            record["pass_b"]["completed_at"] = None
            first = dict(first)
            first.update({
                "disposition": "ADJUDICATION_REQUIRED",
                "adjudication_requested": True,
                "ambiguity_state": "AMBIGUOUS",
                "pass_provenance": "PASS_B",
            })
            second = dict(second)
            second.update({
                "disposition": "ADJUDICATION_REQUIRED",
                "adjudication_requested": True,
                "ambiguity_state": "AMBIGUOUS",
                "pass_provenance": "PASS_B",
            })
            record["endpoints"] = [first, second]
            record["pass_a"]["endpoint_count"] = 2
            record["pass_b"]["status"] = "IN_PROGRESS"
            record["pass_b"]["reviewer"] = None
            record["pass_b"]["completed_at"] = None
            adjudicated = _packet_with_record(packet, 1, record)
            validate_annotation_packet(adjudicated)
            self.assertFalse(is_window_gold_eligible(adjudicated["records"][0]))

            # Exact duplicate span also rejected without adjudication.
            record = _reviewed_record(packet["records"][0])
            duplicate = _endpoint_from_tokens(record, 0, 1, index=2)
            record["endpoints"] = [_endpoint_from_tokens(record, 0, 1), duplicate]
            record["pass_a"]["endpoint_count"] = 2
            with self.assertRaisesRegex(ValueError, "duplicate/overlapping"):
                validate_annotation_packet(_packet_with_record(packet, 1, record))

    def test_recursive_forbidden_scorer_model_fields(self):
        _validate_forbidden_content({"nested": {"bronze_text": "x"}})
        with self.assertRaisesRegex(ValueError, "forbidden key"):
            _validate_forbidden_content({"nested": {"confidence": 0.5}})
        with self.assertRaisesRegex(ValueError, "forbidden key"):
            _validate_forbidden_content({"a": {"b": [{"model_suggestion": "x"}]}})
        with self.assertRaisesRegex(ValueError, "forbidden key"):
            _validate_forbidden_content({"rules": {"label": "KEEP"}})
        with self.assertRaisesRegex(ValueError, "floating-point"):
            _validate_forbidden_content({"records": [{"bronze_char_length": 1.5}]})
        with self.assertRaisesRegex(ValueError, "floating-point"):
            _validate_forbidden_content({"nested": {"count": 0.5}})

    def test_canonical_self_hash_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, packet = _build_packet(Path(temporary))
            output = Path(temporary) / "packet.json"
            output.write_text(json.dumps(packet, sort_keys=True), encoding="utf-8")
            self.assertEqual(load_annotation_packet(output, manifest=manifest), packet)

            broken_hash = copy.deepcopy(dict(packet))
            broken_hash["content_sha256"] = "d" * 64
            with self.assertRaisesRegex(ValueError, "content hash"):
                validate_annotation_packet(broken_hash, manifest=manifest)

            text_tampered = copy.deepcopy(dict(packet))
            records = [dict(item) for item in text_tampered["records"]]
            records[0]["bronze_text"] = records[0]["bronze_text"] + " fabricated"
            records[0]["upstream_end"] = (
                records[0]["upstream_start"] + len(records[0]["bronze_text"])
            )
            records[0]["bronze_char_length"] = len(records[0]["bronze_text"])
            records[0]["bronze_text_sha256"] = hashlib.sha256(
                records[0]["bronze_text"].encode("utf-8"),
            ).hexdigest()
            text_tampered["records"] = records
            text_tampered = rehash_packet(text_tampered)
            with self.assertRaisesRegex(ValueError, "token table"):
                validate_annotation_packet(text_tampered, manifest=manifest)

    def test_candidate_catalog_bound_not_exposed(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, packet = _build_packet(Path(temporary))
            self.assertNotIn("candidates", packet)
            catalog = packet["candidate_catalog"]
            self.assertEqual(set(catalog), {"count", "per_window"})
            self.assertEqual(catalog["count"], manifest["diversity_summary"]["candidate_count"])
            for window_id, binding in catalog["per_window"].items():
                self.assertEqual(set(binding), {"count", "catalog_sha256"})
            self.assertEqual(
                set(catalog["per_window"]),
                {item["window_id"] for item in manifest["selected"]},
            )

    def test_packet_bound_to_selection_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, packet = _build_packet(Path(temporary))
            other = copy.deepcopy(dict(manifest))
            other["purpose"] = other["purpose"] + " (tampered)"
            other = rehash_manifest(other)
            with self.assertRaisesRegex(ValueError, "not bound"):
                validate_annotation_packet(packet, manifest=other)

    def test_in_review_pass_a_in_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, packet = _build_packet(Path(temporary))
            record = _in_review_record(packet["records"][0], pass_a_status="IN_PROGRESS")
            candidate = _packet_with_record(packet, 1, record)
            validate_annotation_packet(candidate, manifest=manifest)
            self.assertEqual(candidate["records"][0]["pass_b"]["status"], "LOCKED_AWAITING_PASS_A")
            self.assertIsNone(candidate["records"][0]["pass_a"]["reviewer"])

    def test_in_review_pass_a_complete_pass_b_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, packet = _build_packet(Path(temporary))
            record = _in_review_record(packet["records"][0], pass_a_status="COMPLETE")
            candidate = _packet_with_record(packet, 1, record)
            validate_annotation_packet(candidate, manifest=manifest)
            reviewed = candidate["records"][0]
            self.assertEqual(reviewed["pass_a"]["status"], "COMPLETE")
            self.assertEqual(reviewed["pass_b"]["status"], "PENDING")
            self.assertIsNone(reviewed["pass_b"]["reviewer"])
            self.assertIsNone(reviewed["pass_b"]["completed_at"])

    def test_in_review_pass_b_in_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, packet = _build_packet(Path(temporary))
            record = _in_review_record(
                packet["records"][0],
                pass_a_status="COMPLETE",
                pass_b_status="IN_PROGRESS",
            )
            endpoint = _endpoint_from_tokens(record, 0, 1)
            record["endpoints"] = [endpoint]
            record["pass_a"]["endpoint_count"] = 1
            candidate = _packet_with_record(packet, 1, record)
            validate_annotation_packet(candidate, manifest=manifest)
            self.assertEqual(candidate["records"][0]["pass_b"]["status"], "IN_PROGRESS")

    def test_in_review_invalid_combinations(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, packet = _build_packet(Path(temporary))
            base = packet["records"][0]

            with self.assertRaisesRegex(ValueError, "active or complete pass_a"):
                record = _in_review_record(base, pass_a_status="PENDING")
                validate_annotation_packet(
                    _packet_with_record(packet, 1, record), manifest=manifest,
                )

            with self.assertRaisesRegex(ValueError, "complete pass_b"):
                record = _in_review_record(
                    base,
                    pass_a_status="COMPLETE",
                    pass_b_status="COMPLETE",
                )
                record["pass_b"]["reviewer"] = "reviewer-b"
                record["pass_b"]["completed_at"] = "2026-08-18T01:00:00Z"
                record["pass_b"]["audit_checks"] = {
                    key: True for key in AUDIT_CHECKS
                }
                validate_annotation_packet(
                    _packet_with_record(packet, 1, record), manifest=manifest,
                )

            with self.assertRaisesRegex(ValueError, "clean KEEP-only"):
                record = _in_review_record(base, pass_a_status="COMPLETE")
                endpoint = _endpoint_from_tokens(
                    record, 0, 1,
                    ambiguity_state="AMBIGUOUS", disposition="AMBIGUOUS",
                    adjudication_requested=False,
                )
                record["endpoints"] = [endpoint]
                record["pass_a"]["endpoint_count"] = 1
                validate_annotation_packet(
                    _packet_with_record(packet, 1, record), manifest=manifest,
                )

            with self.assertRaisesRegex(
                ValueError,
                "clean KEEP-only|KEEP endpoint cannot carry|ADJUDICATION_REQUIRED endpoint requires",
            ):
                record = _in_review_record(base, pass_a_status="COMPLETE")
                endpoint = _endpoint_from_tokens(
                    record, 0, 1, adjudication_requested=True,
                )
                record["endpoints"] = [endpoint]
                record["pass_a"]["endpoint_count"] = 1
                validate_annotation_packet(
                    _packet_with_record(packet, 1, record), manifest=manifest,
                )

            with self.assertRaisesRegex(ValueError, "clean KEEP-only"):
                record = _in_review_record(
                    base, pass_a_status="COMPLETE", ambiguity_flagged=True,
                )
                validate_annotation_packet(
                    _packet_with_record(packet, 1, record), manifest=manifest,
                )

            with self.assertRaisesRegex(ValueError, "clean KEEP-only"):
                record = _in_review_record(
                    base, pass_a_status="COMPLETE", exclusion_flagged=True,
                )
                validate_annotation_packet(
                    _packet_with_record(packet, 1, record), manifest=manifest,
                )

    def test_in_review_never_gold_eligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, packet = _build_packet(Path(temporary))
            record = _in_review_record(
                packet["records"][0],
                pass_a_status="COMPLETE",
                pass_b_status="IN_PROGRESS",
            )
            endpoint = _endpoint_from_tokens(record, 0, 1)
            record["endpoints"] = [endpoint]
            record["pass_a"]["endpoint_count"] = 1
            candidate = _packet_with_record(packet, 1, record)
            validate_annotation_packet(candidate, manifest=manifest)
            self.assertFalse(is_window_gold_eligible(candidate["records"][0]))
            self.assertFalse(is_packet_gold_eligible(candidate))

    def test_manifest_binding_rejects_internally_consistent_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, packet = _build_packet(Path(temporary))
            manifest_records = [dict(item) for item in manifest["selected"]]

            # Bronze text fully retokenized, rehashed, and re-offset: only the
            # manifest Bronze binding can catch it.
            tampered = copy.deepcopy(dict(packet))
            records = [dict(item) for item in tampered["records"]]
            record = records[0]
            new_text = record["bronze_text"] + " fabricated"
            record["bronze_text"] = new_text
            record["bronze_text_sha256"] = hashlib.sha256(
                new_text.encode("utf-8"),
            ).hexdigest()
            record["bronze_char_length"] = len(new_text)
            record["upstream_end"] = record["upstream_start"] + len(new_text)
            record["tokens"] = token_table(new_text)
            tampered["records"] = records
            tampered = rehash_packet(tampered)
            with self.assertRaisesRegex(
                ValueError,
                "Bronze text contradicts|contradicts the selection manifest",
            ):
                validate_annotation_packet(tampered, manifest=manifest)

            # Offsets shifted while keeping the text length consistent.
            tampered = copy.deepcopy(dict(packet))
            records = [dict(item) for item in tampered["records"]]
            record = records[0]
            record["upstream_start"] += 1
            record["upstream_end"] += 1
            tampered["records"] = records
            tampered = rehash_packet(tampered)
            with self.assertRaisesRegex(ValueError, "contradicts the selection manifest"):
                validate_annotation_packet(tampered, manifest=manifest)

            # Source group/upstream identity changed consistently.
            tampered = copy.deepcopy(dict(packet))
            records = [dict(item) for item in tampered["records"]]
            record = records[0]
            record["source_group_id"] = "video:other-video"
            record["upstream_source_id"] = "other-video"
            tampered["records"] = records
            tampered = rehash_packet(tampered)
            with self.assertRaisesRegex(ValueError, "contradicts the selection manifest"):
                validate_annotation_packet(tampered, manifest=manifest)

            # Partition swapped to the other valid partition.
            tampered = copy.deepcopy(dict(packet))
            records = [dict(item) for item in tampered["records"]]
            record = records[0]
            record["partition"] = (
                "FROZEN_REPLICATION"
                if record["partition"] == "EXPANDED_DEV"
                else "EXPANDED_DEV"
            )
            tampered["records"] = records
            tampered = rehash_packet(tampered)
            with self.assertRaisesRegex(ValueError, "contradicts the selection manifest"):
                validate_annotation_packet(tampered, manifest=manifest)

            # Record order swapped with record_index renumbered so the only
            # remaining violation is the manifest order binding.
            tampered = copy.deepcopy(dict(packet))
            records = [dict(item) for item in tampered["records"]]
            records[0], records[1] = records[1], records[0]
            records[0]["record_index"] = 1
            records[1]["record_index"] = 2
            tampered["records"] = records
            tampered = rehash_packet(tampered)
            with self.assertRaisesRegex(ValueError, "record order must match"):
                validate_annotation_packet(tampered, manifest=manifest)

            self.assertEqual(
                [item["window_id"] for item in packet["records"]],
                [item["window_id"] for item in manifest_records],
            )


if __name__ == "__main__":
    unittest.main()
