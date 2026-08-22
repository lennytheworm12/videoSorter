"""Focused tests for the Phase 2J frozen candidate-coverage gate."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.phase2j_adjudication import build_adjudication_packet
from pipeline.phase2j_adjudication_import import build_reviewed_packet
from pipeline.phase2j_candidate_coverage import (
    COVERAGE_SCHEMA_VERSION,
    ERROR_CODE,
    MISSING_LONGER_SPAN_ONLY,
    MISSING_MIXED_BOUNDARY_MISMATCH,
    MISSING_NO_OVERLAPPING_CANDIDATE,
    MISSING_PARTIAL_OVERLAP_ONLY,
    MISSING_SHORTER_FRAGMENT_ONLY,
    build_candidate_coverage,
    classify_missing_span,
    load_candidate_coverage,
    serialize_candidate_coverage,
    validate_candidate_coverage,
)
from pipeline.phase2j_source_selection import canonical_sha256
from pipeline.semantic_mentions import generate_mention_candidates
from tests._phase2j_helpers import (
    build_human_session,
    build_sol_review,
    write_human_session,
    write_sol_review,
    write_standard_phase2j_inputs,
)
from tests.test_phase2j_adjudication import _build_packet, _known_spans
from tests.test_phase2j_adjudication_import import _build_export, _write


def _write_manifest(root: Path):
    """Write the locked selection manifest file for the shared packet fixture."""
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
    manifest_file = root / "window-selection-manifest-v1.json"
    manifest_file.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _fixture(root: Path, *, human_spans=None, sol_spans=None):
    """Write every Phase 2J input and import a synthetic reviewed gold packet."""
    packet = _build_packet(root)
    blank_path = _write(root / "blank-packet.json", dict(packet))
    manifest_path = root / "window-selection-manifest-v1.json"
    manifest = _write_manifest(root)
    human_path = root / "human-session.json"
    sol_path = root / "sol-review.json"
    write_human_session(human_path, packet, spans_for_window=human_spans)
    write_sol_review(sol_path, packet, spans_for_window=sol_spans)
    human = build_human_session(packet, spans_for_window=human_spans)
    sol = build_sol_review(packet, spans_for_window=sol_spans)
    adjudication = build_adjudication_packet(
        packet, human, sol,
        human_session_path=human_path,
        sol_review_path=sol_path,
    )
    adjudication_path = _write(root / "adjudication.json", adjudication)
    export_path = _write(root / "export.json", _build_export(adjudication))
    reviewed = build_reviewed_packet(
        blank_packet_path=blank_path,
        manifest_path=manifest_path,
        human_session_path=human_path,
        adjudication_packet_path=adjudication_path,
        export_path=export_path,
    )
    reviewed_path = _write(root / "reviewed.json", reviewed)
    return {
        "root": root,
        "manifest_path": manifest_path,
        "packet_path": reviewed_path,
        "manifest": manifest,
        "packet": reviewed,
    }


def _build(fixture):
    return build_candidate_coverage(
        manifest_path=fixture["manifest_path"],
        packet_path=fixture["packet_path"],
    )


def _rehash(value):
    inner = {key: item for key, item in value.items() if key != "content_sha256"}
    return {**dict(value), "content_sha256": canonical_sha256(inner)}


class Phase2JCandidateCoverageTests(unittest.TestCase):
    def test_exact_coverage_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            artifact = _build(fixture)
            aggregate = artifact["coverage"]["aggregate"]
            gold = sum(len(record["endpoints"]) for record in fixture["packet"]["records"])
            self.assertEqual(aggregate["hit_count"], aggregate["denominator"])
            self.assertEqual(aggregate["denominator"], gold)
            self.assertEqual(aggregate["rate"], 1.0)
            self.assertEqual(artifact["missing_endpoints"], [])
            self.assertEqual(len(artifact["covered_endpoints"]), gold)
            self.assertEqual(
                artifact["coverage"]["total_candidates"],
                fixture["manifest"]["diversity_summary"]["candidate_count"],
            )
            for record in artifact["covered_endpoints"]:
                self.assertTrue(record["candidate_id"].startswith("transcript:"))
                self.assertTrue(record["candidate_alias"].startswith("C"))
                self.assertEqual(
                    record["candidate_generator_version"],
                    fixture["manifest"]["candidate_generator_version"],
                )
                self.assertRegex(record["candidate_catalog_sha256"], r"[0-9a-f]{64}")
            output = fixture["root"] / "coverage.json"
            output.write_text(serialize_candidate_coverage(artifact), encoding="utf-8")
            loaded = load_candidate_coverage(
                output,
                manifest_path=fixture["manifest_path"],
                packet_path=fixture["packet_path"],
            )
            self.assertEqual(loaded["content_sha256"], artifact["content_sha256"])

    def test_classify_missing_all_categories(self):
        cases = [
            (MISSING_NO_OVERLAPPING_CANDIDATE, [(0, 5), (22, 30), (30, 35)]),
            (MISSING_LONGER_SPAN_ONLY, [(5, 25)]),
            (MISSING_LONGER_SPAN_ONLY, [(10, 25), (5, 20)]),
            (MISSING_SHORTER_FRAGMENT_ONLY, [(12, 18)]),
            (MISSING_SHORTER_FRAGMENT_ONLY, [(10, 15), (15, 20)]),
            (MISSING_PARTIAL_OVERLAP_ONLY, [(5, 15)]),
            (MISSING_PARTIAL_OVERLAP_ONLY, [(15, 25)]),
            (MISSING_PARTIAL_OVERLAP_ONLY, [(5, 15), (15, 25)]),
            (MISSING_MIXED_BOUNDARY_MISMATCH, [(5, 25), (12, 18)]),
            (MISSING_MIXED_BOUNDARY_MISMATCH, [(12, 18), (5, 15)]),
        ]
        for category, spans in cases:
            self.assertEqual(
                classify_missing_span(10, 20, spans)[0], category,
            )
        self.assertIsNone(classify_missing_span(10, 20, [(10, 20)])[0])
        self.assertEqual(
            classify_missing_span(10, 20, [(0, 5)])[0],
            MISSING_NO_OVERLAPPING_CANDIDATE,
        )

    def test_end_to_end_missing_classification_diagnostics(self):
        def human_spans(record):
            if "uses W," in record["bronze_text"]:
                return [(8, 8, "STATE")]
            return _known_spans(record)[0]

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary), human_spans=human_spans)
            artifact = _build(fixture)
            self.assertTrue(artifact["missing_endpoints"])
            self.assertTrue(artifact["covered_endpoints"])
            gold = sum(len(record["endpoints"]) for record in fixture["packet"]["records"])
            self.assertEqual(
                artifact["coverage"]["aggregate"]["denominator"], gold,
            )
            self.assertEqual(
                artifact["coverage"]["aggregate"]["hit_count"]
                + len(artifact["missing_endpoints"]),
                gold,
            )
            missing_windows = [
                record for record in fixture["packet"]["records"]
                if "uses W," in record["bronze_text"]
            ]
            self.assertEqual(len(artifact["missing_endpoints"]), len(missing_windows))
            zero_hit_groups = {
                record["source_group_id"]
                for record in artifact["missing_endpoints"]
                if artifact["coverage"]["per_source_group"][
                    record["source_group_id"]
                ]["hit_count"] == 0
            }
            self.assertTrue(zero_hit_groups)
            self.assertIn("STATE", artifact["coverage"]["per_node_type"])
            bronze_by_window = {
                record["window_id"]: record["bronze_text"]
                for record in fixture["packet"]["records"]
            }
            manifest_by_window = {
                record["window_id"]: record
                for record in fixture["manifest"]["selected"]
            }
            for missing in artifact["missing_endpoints"]:
                self.assertEqual(missing["error_code"], ERROR_CODE)
                self.assertEqual(
                    missing["failure_category"], MISSING_MIXED_BOUNDARY_MISMATCH,
                )
                text = bronze_by_window[missing["window_id"]]
                manifest_record = manifest_by_window[missing["window_id"]]
                self.assertEqual(
                    missing["bronze_text"],
                    text[missing["char_start"]:missing["char_end"]],
                )
                self.assertEqual(missing["partition"], manifest_record["partition"])
                self.assertEqual(
                    missing["source_group_id"], manifest_record["source_group_id"],
                )
                self.assertEqual(
                    missing["role"], manifest_record["metadata"]["role"],
                )
                self.assertGreater(missing["overlap_count"], 0)
                self.assertEqual(missing["overlap_count"], len(missing["overlaps"]))
                self.assertTrue(all(
                    overlap["candidate_id"] for overlap in missing["overlaps"]
                ))
                self.assertTrue(all(
                    overlap["candidate_alias"] for overlap in missing["overlaps"]
                ))
                for overlap in missing["overlaps"]:
                    self.assertEqual(
                        overlap["text"], text[overlap["start"]:overlap["end"]],
                    )
                    self.assertEqual(
                        overlap["absolute_end"] - overlap["absolute_start"],
                        overlap["end"] - overlap["start"],
                    )

    def test_role_source_partition_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            artifact = _build(fixture)
            coverage = artifact["coverage"]
            aggregate = coverage["aggregate"]
            self.assertEqual(
                set(coverage["per_partition"]),
                {"EXPANDED_DEV", "FROZEN_REPLICATION"},
            )
            self.assertEqual(
                sum(item["denominator"] for item in coverage["per_partition"].values()),
                aggregate["denominator"],
            )
            self.assertEqual(
                sum(item["hit_count"] for item in coverage["per_partition"].values()),
                aggregate["hit_count"],
            )
            manifest_by_window = {
                record["window_id"]: record
                for record in fixture["manifest"]["selected"]
            }
            groups = {
                manifest_by_window[record["window_id"]]["source_group_id"]
                for record in fixture["packet"]["records"]
            }
            roles = {
                manifest_by_window[record["window_id"]]["metadata"]["role"]
                for record in fixture["packet"]["records"]
            }
            node_types = {
                endpoint["node_type"]
                for record in fixture["packet"]["records"]
                for endpoint in record["endpoints"]
            }
            self.assertEqual(set(coverage["per_source_group"]), groups)
            self.assertEqual(set(coverage["per_role"]), roles)
            self.assertEqual(
                set(coverage["per_node_type"]),
                {value if value is not None else "null" for value in node_types},
            )
            for section in ("per_source_group", "per_role", "per_node_type"):
                self.assertEqual(
                    sum(item["denominator"] for item in coverage[section].values()),
                    aggregate["denominator"],
                )
                self.assertEqual(
                    sum(item["hit_count"] for item in coverage[section].values()),
                    aggregate["hit_count"],
                )

    def test_deterministic_artifact_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            first = _build(fixture)
            second = _build(fixture)
            self.assertEqual(
                serialize_candidate_coverage(first),
                serialize_candidate_coverage(second),
            )
            self.assertEqual(first["schema_version"], COVERAGE_SCHEMA_VERSION)
            inner = {
                key: value for key, value in first.items()
                if key != "content_sha256"
            }
            self.assertEqual(first["content_sha256"], canonical_sha256(inner))
            self.assertEqual(first["scoring_absence"]["model_scoring"], "ABSENT")
            self.assertEqual(first["scoring_absence"]["model_predictions"], "ABSENT")
            self.assertEqual(first["scoring_absence"]["thresholds"], "ABSENT")

    def test_tampering_and_hash_binding_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            artifact = _build(fixture)
            bad = copy.deepcopy(dict(artifact))
            bad["coverage"]["aggregate"]["hit_count"] += 1
            with self.assertRaisesRegex(ValueError, "content hash"):
                validate_candidate_coverage(bad)
            bad = _rehash(bad)
            with self.assertRaisesRegex(ValueError, "aggregate metric"):
                validate_candidate_coverage(bad)
            bad = copy.deepcopy(dict(artifact))
            bad["covered_endpoints"][0]["candidate_id"] = "bogus"
            bad = _rehash(bad)
            with self.assertRaisesRegex(ValueError, "deterministic regeneration"):
                validate_candidate_coverage(
                    bad,
                    manifest=fixture["manifest"],
                    packet=fixture["packet"],
                    manifest_path=fixture["manifest_path"],
                    packet_path=fixture["packet_path"],
                )
            bad = copy.deepcopy(dict(artifact))
            bad["selection_manifest"]["file_sha256"] = "0" * 64
            bad = _rehash(bad)
            with self.assertRaisesRegex(ValueError, "file hash"):
                validate_candidate_coverage(
                    bad,
                    manifest=fixture["manifest"],
                    packet=fixture["packet"],
                    manifest_path=fixture["manifest_path"],
                    packet_path=fixture["packet_path"],
                )

    def test_duplicate_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            artifact = _build(fixture)
            body = serialize_candidate_coverage(artifact)
            duplicated = (
                '{\n  "schema_version": "phase2j-candidate-coverage-v1",\n'
                + body[body.index("\n") + 1:]
            )
            path = fixture["root"] / "duplicate-coverage.json"
            path.write_text(duplicated, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate keys"):
                load_candidate_coverage(path)

    def test_boolean_overlap_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            artifact = _build(fixture)
            covered = artifact["covered_endpoints"].pop()
            artifact["missing_endpoints"].append({
                "endpoint_id": covered["endpoint_id"],
                "window_id": covered["window_id"],
                "source_group_id": covered["source_group_id"],
                "partition": covered["partition"],
                "role": covered["role"],
                "node_type": covered["node_type"],
                "char_start": covered["char_start"],
                "char_end": covered["char_end"],
                "absolute_start": covered["absolute_start"],
                "absolute_end": covered["absolute_end"],
                "bronze_text": covered["bronze_text"],
                "error_code": ERROR_CODE,
                "failure_category": MISSING_NO_OVERLAPPING_CANDIDATE,
                "overlap_count": False,
                "overlaps": [],
            })
            artifact = _rehash(artifact)
            with self.assertRaisesRegex(ValueError, "overlap diagnostics"):
                validate_candidate_coverage(artifact)

    def test_catalog_binding_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))

            def drop_last(window):
                candidates = generate_mention_candidates(window)
                return candidates[:-1]

            with patch(
                "pipeline.phase2j_candidate_coverage.generate_mention_candidates",
                side_effect=drop_last,
            ):
                with self.assertRaisesRegex(ValueError, "candidate_catalog_binding"):
                    _build(fixture)

    def test_duplicate_candidate_span_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))

            def duplicate_first(window):
                candidates = generate_mention_candidates(window)
                return candidates + (candidates[0],)

            with patch(
                "pipeline.phase2j_candidate_coverage.generate_mention_candidates",
                side_effect=duplicate_first,
            ):
                with self.assertRaisesRegex(ValueError, "duplicate span"):
                    _build(fixture)

    def test_gold_eligibility_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            packet = copy.deepcopy(dict(fixture["packet"]))
            record = packet["records"][0]
            record["window_status"] = "EXCLUDED"
            record["pass_a"]["endpoint_count"] = 0
            record["pass_b"]["status"] = "PENDING"
            record["pass_b"]["reviewer"] = None
            record["pass_b"]["completed_at"] = None
            record["pass_b"]["notes"] = []
            record["endpoints"] = []
            record["exclusion_controls"] = {"flagged": True, "notes": ["excluded"]}
            packet = _rehash(packet)
            path = _write(fixture["root"] / "reviewed-excluded.json", packet)
            with self.assertRaisesRegex(ValueError, "not fully gold eligible"):
                build_candidate_coverage(
                    manifest_path=fixture["manifest_path"], packet_path=path,
                )


if __name__ == "__main__":
    unittest.main()
