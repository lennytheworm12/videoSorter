"""Focused tests for the Phase 2J human-vs-Sol adjudication packet builder."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.phase2j_adjudication import (
    ADJUDICATION_PACKET_SCHEMA_VERSION,
    ADJUDICATION_VERSION,
    HUMAN_SESSION_SCHEMA_VERSION,
    OUTPUT_FORBIDDEN_KEYS,
    SOL_REVIEW_SCHEMA_VERSION,
    VISIBILITY_GATE,
    build_adjudication_packet,
    build_components,
    validate_adjudication_packet,
)
from pipeline.phase2j_source_selection import canonical_sha256, file_sha256
from tests._phase2j_helpers import (
    build_human_session,
    build_sol_review,
    write_human_session,
    write_sol_review,
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
    legacy_benchmark = load_legacy_benchmark(
        benchmark_path, manifest=legacy_manifest,
    )
    manifest = build_selection_manifest(
        pool=pool,
        pool_path=pool_path,
        legacy_manifest=legacy_manifest,
        legacy_manifest_path=manifest_path,
        legacy_benchmark=legacy_benchmark,
        legacy_benchmark_path=benchmark_path,
    )
    from pipeline.phase2j_annotation_packet import (
        build_annotation_packet,
        validate_annotation_packet,
    )

    packet = build_annotation_packet(manifest)
    validate_annotation_packet(packet, manifest=manifest)
    return packet


def _known_spans(record):
    """Per-window spans producing one of every component classification.

    Five Human spans and four Sol spans form exactly five connected
    components per window: an exact agreement, a Sol-bridged boundary
    disagreement spanning two nonoverlapping Human spans, a type
    disagreement, one Human-only span, and one Sol-only span.  All spans
    stay inside every fixture window (the shortest window has 23 tokens).
    """
    human = [
        (0, 1, "ENTITY"),      # exact agreement with Sol (0, 1, ENTITY)
        (2, 3, "ACTION"),      # boundary disagreement, Human part 1
        (5, 5, "STATE"),       # boundary disagreement, Human part 2
        (9, 9, "QUANTITY"),    # type disagreement with Sol (9, 9, STATE)
        (12, 12, "EVENT"),     # Human-only
    ]
    sol = [
        (0, 1, "ENTITY"),      # exact agreement
        (3, 5, "TIME"),        # bridges Human (2, 3) and (5, 5)
        (9, 9, "STATE"),       # type disagreement
        (13, 13, "EVENT"),     # Sol-only
    ]
    return human, sol


def _build_fixture(root: Path):
    packet = _build_packet(root)
    human_path = root / "human-session.json"
    sol_path = root / "sol-review.json"
    write_human_session(
        human_path, packet, spans_for_window=lambda record: _known_spans(record)[0],
    )
    write_sol_review(
        sol_path, packet, spans_for_window=lambda record: _known_spans(record)[1],
    )
    human = build_human_session(
        packet, spans_for_window=lambda record: _known_spans(record)[0],
    )
    sol = build_sol_review(
        packet, spans_for_window=lambda record: _known_spans(record)[1],
    )
    return packet, human, sol, human_path, sol_path


def _serialize(value: dict) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


class Phase2JAdjudicationPacketTests(unittest.TestCase):
    def test_build_is_deterministic_and_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human, sol, human_path, sol_path = _build_fixture(root)
            first = build_adjudication_packet(
                packet, human, sol,
                human_session_path=human_path,
                sol_review_path=sol_path,
            )
            second = build_adjudication_packet(
                packet, human, sol,
                human_session_path=human_path,
                sol_review_path=sol_path,
            )
            validate_adjudication_packet(first)
            self.assertEqual(first, second)
            self.assertEqual(_serialize(first), _serialize(second))
            inner = {key: value for key, value in first.items() if key != "content_sha256"}
            self.assertEqual(first["content_sha256"], canonical_sha256(inner))
            self.assertEqual(first["schema_version"], ADJUDICATION_PACKET_SCHEMA_VERSION)
            self.assertEqual(first["adjudication_version"], ADJUDICATION_VERSION)
            self.assertEqual(first["visibility_gate"], VISIBILITY_GATE)
            self.assertEqual(first["human_session_schema_version"], HUMAN_SESSION_SCHEMA_VERSION)
            self.assertEqual(first["sol_review_schema_version"], SOL_REVIEW_SCHEMA_VERSION)
            self.assertEqual(
                first["human_session_sha256"], file_sha256(human_path),
            )
            self.assertEqual(
                first["sol_review_sha256"], file_sha256(sol_path),
            )

    def test_component_classification_totals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human, sol, human_path, sol_path = _build_fixture(root)
            built = build_adjudication_packet(
                packet, human, sol,
                human_session_path=human_path,
                sol_review_path=sol_path,
            )
            totals = built["totals"]
            self.assertEqual(totals["windows"], 30)
            self.assertEqual(totals["exact_agreements"], 30)
            self.assertEqual(totals["type_disagreements"], 30)
            self.assertEqual(totals["boundary_disagreements"], 30)
            self.assertEqual(totals["sol_only"], 30)
            self.assertEqual(totals["human_only"], 30)
            self.assertEqual(totals["components"], 150)
            self.assertEqual(totals["human_endpoints"], 150)
            self.assertEqual(totals["sol_endpoints"], 120)
            self.assertEqual(
                sum(len(record["components"]) for record in built["records"]),
                totals["components"],
            )

    def test_output_contains_no_forbidden_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human, sol, human_path, sol_path = _build_fixture(root)
            built = build_adjudication_packet(
                packet, human, sol,
                human_session_path=human_path,
                sol_review_path=sol_path,
            )
            text = _serialize(built)
            for banned in (
                "reviewer_name", "reviewer_model", "reasoning_effort",
                "score", "candidate", "partition", "champion", "role",
                "video_title", "pass_a", "pass_b", "exported_at",
            ):
                self.assertNotIn(f'"{banned}"', text, banned)
            self.assertNotIn("test-reviewer", text)

            def scan(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        self.assertNotIn(key.casefold(), OUTPUT_FORBIDDEN_KEYS, key)
                        scan(item)
                elif isinstance(value, list):
                    for item in value:
                        scan(item)

            scan(built)

    def test_human_packet_binding_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human, sol, human_path, sol_path = _build_fixture(root)
            human = copy.deepcopy(human)
            human["packet_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                build_adjudication_packet(
                    packet, human, sol,
                    human_session_path=human_path,
                    sol_review_path=sol_path,
                )

    def test_human_window_order_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human, sol, human_path, sol_path = _build_fixture(root)
            human = copy.deepcopy(human)
            human["records"] = [human["records"][1], human["records"][0]] + human["records"][2:]
            with self.assertRaises(ValueError):
                build_adjudication_packet(
                    packet, human, sol,
                    human_session_path=human_path,
                    sol_review_path=sol_path,
                )

    def test_human_endpoint_exact_slice_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human, sol, human_path, sol_path = _build_fixture(root)
            human = copy.deepcopy(human)
            human["records"][0]["endpoints"][0]["exact_bronze_text"] = "tampered"
            with self.assertRaises(ValueError):
                build_adjudication_packet(
                    packet, human, sol,
                    human_session_path=human_path,
                    sol_review_path=sol_path,
                )

    def test_human_internal_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human, sol, human_path, sol_path = _build_fixture(root)
            human = copy.deepcopy(human)
            record = human["records"][0]
            first = record["endpoints"][0]
            record["endpoints"].append({**first, "endpoint_id": "p2j:review:" + record["window_id"] + ":ep:9999"})
            with self.assertRaises(ValueError):
                build_adjudication_packet(
                    packet, human, sol,
                    human_session_path=human_path,
                    sol_review_path=sol_path,
                )

    def test_human_non_keep_endpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human, sol, human_path, sol_path = _build_fixture(root)
            human = copy.deepcopy(human)
            human["records"][0]["endpoints"][0]["disposition"] = "AMBIGUOUS"
            with self.assertRaises(ValueError):
                build_adjudication_packet(
                    packet, human, sol,
                    human_session_path=human_path,
                    sol_review_path=sol_path,
                )

    def test_sol_content_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human, sol, human_path, sol_path = _build_fixture(root)
            sol = copy.deepcopy(sol)
            sol["content_sha256"] = "1" * 64
            with self.assertRaises(ValueError):
                build_adjudication_packet(
                    packet, human, sol,
                    human_session_path=human_path,
                    sol_review_path=sol_path,
                )

    def test_sol_packet_binding_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human, sol, human_path, sol_path = _build_fixture(root)
            sol = copy.deepcopy(sol)
            sol["blank_packet_sha256"] = "2" * 64
            with self.assertRaises(ValueError):
                build_adjudication_packet(
                    packet, human, sol,
                    human_session_path=human_path,
                    sol_review_path=sol_path,
                )

    def test_sol_endpoint_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human, sol, human_path, sol_path = _build_fixture(root)
            sol = copy.deepcopy(sol)
            record = sol["records"][0]
            first = record["proposed_endpoints"][0]
            record["proposed_endpoints"].append({**first})
            with self.assertRaises(ValueError):
                build_adjudication_packet(
                    packet, human, sol,
                    human_session_path=human_path,
                    sol_review_path=sol_path,
                )

    def test_sol_non_gold_provenance_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human, sol, human_path, sol_path = _build_fixture(root)
            sol = copy.deepcopy(sol)
            sol["records"][0]["proposed_endpoints"][0]["pass_provenance"] = "GOLD"
            with self.assertRaises(ValueError):
                build_adjudication_packet(
                    packet, human, sol,
                    human_session_path=human_path,
                    sol_review_path=sol_path,
                )

    def test_build_components_overlap_semantics(self):
        def endpoint(endpoint_id, token_start, token_end, node_type="ENTITY"):
            return {
                "endpoint_id": endpoint_id,
                "token_start": token_start,
                "token_end": token_end,
                "node_type": node_type,
                "char_start": 0,
                "char_end": 1,
                "exact_bronze_text": "x",
            }

        human = [
            endpoint("h1", 0, 1),
            endpoint("h2", 4, 5),
            endpoint("h3", 9, 9),
        ]
        sol = [
            endpoint("s1", 1, 2),
            endpoint("s2", 7, 8),
            endpoint("s3", 9, 9, "ACTION"),
        ]
        components = build_components("w1", human, sol)
        by_id = {component["component_id"]: component for component in components}
        self.assertEqual(len(components), 4)
        # Components must be deterministically numbered by Bronze source
        # position (earliest covered token), never by annotation side order:
        # the Sol-only span at tokens 7-8 precedes the shared span at token 9.
        self.assertEqual(
            [component["component_id"] for component in components],
            [
                "p2j:adjudicate:w1:c:0001",
                "p2j:adjudicate:w1:c:0002",
                "p2j:adjudicate:w1:c:0003",
                "p2j:adjudicate:w1:c:0004",
            ],
        )
        first = by_id["p2j:adjudicate:w1:c:0001"]
        self.assertEqual(first["classification"], "BOUNDARY_DISAGREEMENT")
        self.assertEqual(set(first["human_endpoint_ids"]), {"h1"})
        self.assertEqual(set(first["sol_endpoint_ids"]), {"s1"})
        second = by_id["p2j:adjudicate:w1:c:0002"]
        self.assertEqual(second["classification"], "HUMAN_ONLY")
        self.assertEqual(second["human_endpoint_ids"], ["h2"])
        third = by_id["p2j:adjudicate:w1:c:0003"]
        self.assertEqual(third["classification"], "SOL_ONLY")
        self.assertEqual(third["sol_endpoint_ids"], ["s2"])
        fourth = by_id["p2j:adjudicate:w1:c:0004"]
        self.assertEqual(fourth["classification"], "TYPE_DISAGREEMENT")
        self.assertEqual(fourth["human_endpoint_ids"], ["h3"])
        self.assertEqual(fourth["sol_endpoint_ids"], ["s3"])

    def test_validate_packet_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human, sol, human_path, sol_path = _build_fixture(root)
            built = build_adjudication_packet(
                packet, human, sol,
                human_session_path=human_path,
                sol_review_path=sol_path,
            )
            tampered = copy.deepcopy(built)
            tampered["totals"]["exact_agreements"] = 1
            with self.assertRaises(ValueError):
                validate_adjudication_packet(tampered)

            tampered = copy.deepcopy(built)
            tampered["records"][0]["components"][0]["classification"] = "SOL_ONLY"
            with self.assertRaises(ValueError):
                validate_adjudication_packet(tampered)

            tampered = copy.deepcopy(built)
            tampered["records"][0]["components"][0]["component_id"] = "p2j:adjudicate:x:c:0001"
            with self.assertRaises(ValueError):
                validate_adjudication_packet(tampered)

            tampered = copy.deepcopy(built)
            tampered["records"][0]["human_endpoints"][0]["node_type"] = "NOPE"
            with self.assertRaises(ValueError):
                validate_adjudication_packet(tampered)

            tampered = copy.deepcopy(built)
            tampered["records"][0]["sol_endpoints"][0]["sol_rationale"] = ""
            with self.assertRaises(ValueError):
                validate_adjudication_packet(tampered)

            tampered = copy.deepcopy(built)
            tampered["records"] = tampered["records"][:29]
            with self.assertRaises(ValueError):
                validate_adjudication_packet(tampered)

    def test_real_packet_totals_match_evidence(self):
        packet_path = Path(__file__).resolve().parents[1] / "data/phase2j" / (
            "phase2j-adjudication-packet-v1.json"
        )
        if not packet_path.is_file():
            self.skipTest("generated adjudication packet is not present")
        from pipeline.phase2j_adjudication import load_adjudication_packet

        packet = load_adjudication_packet(packet_path)
        validate_adjudication_packet(packet)
        self.assertEqual(
            packet["packet_sha256"],
            "3f766b08696ed512063d999c75877001d77b03db136f8edae78e631e1725c62a",
        )
        self.assertEqual(
            packet["human_session_sha256"],
            "85437bfcc737ed71380f26581883f08bf4be4853d861ff055db642e338d1a471",
        )
        self.assertEqual(
            packet["sol_review_sha256"],
            "6ef4ccbff8f9512b9119d314050acd5aaa87b927c37ee83372fcec92edd1cd8c",
        )
        totals = packet["totals"]
        self.assertEqual(totals["components"], 326)
        self.assertEqual(totals["exact_agreements"], 49)
        self.assertEqual(totals["type_disagreements"], 16)
        self.assertEqual(totals["boundary_disagreements"], 87)
        self.assertEqual(totals["sol_only"], 174)
        self.assertEqual(totals["human_only"], 0)
        self.assertEqual(totals["human_endpoints"], 166)
        self.assertEqual(totals["sol_endpoints"], 338)
        self.assertEqual(totals["windows"], 30)


if __name__ == "__main__":
    unittest.main()
