"""Focused tests for the Phase 2J post-adjudication canonical import gate."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.phase2j_adjudication import (
    ADJUDICATION_VERSION,
    build_adjudication_packet,
)
from pipeline.phase2j_adjudication_import import (
    ADJUDICATION_EXPORT_SCHEMA_VERSION,
    build_reviewed_packet,
    derive_resolved_endpoints,
    serialize_reviewed_packet,
    validate_adjudication_export,
)
from pipeline.phase2j_annotation_packet import (
    FORBIDDEN_KEYS,
    RELEASE_GATE,
    is_packet_gold_eligible,
    is_window_gold_eligible,
    load_annotation_packet,
    validate_annotation_packet,
)
from pipeline.phase2j_source_selection import (
    canonical_sha256,
    load_selection_manifest,
)
from tests._phase2j_helpers import (
    build_human_session,
    build_sol_review,
    write_human_session,
    write_sol_review,
    write_standard_phase2j_inputs,
)
from tests.test_phase2j_adjudication import _build_packet, _known_spans


AUDIT_KEYS = ("boundaries", "omissions", "roles", "duplicates", "ambiguity")


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
    _write(manifest_file, dict(manifest))
    return manifest


def _serialize(value: dict) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _write(path: Path, value: dict) -> Path:
    path.write_text(_serialize(value), encoding="utf-8")
    return path


def _rehash(value: dict) -> dict:
    """Rebuild a packet's canonical content hash after tampering its inner."""
    inner = {key: item for key, item in value.items() if key != "content_sha256"}
    return {"content_sha256": canonical_sha256(inner), **inner}


def _all_true_audit_checks() -> dict:
    return {key: True for key in AUDIT_KEYS}


def _export_components(
    adjudication_record,
    *,
    accept_sol_component_ids=(),
    custom_component_id=None,
    unresolved_component_id=None,
    drop_all=False,
):
    """Deterministic default export component entries for one window."""
    components = []
    for component in adjudication_record["components"]:
        classification = component["classification"]
        component_id = component["component_id"]
        if component_id == unresolved_component_id:
            components.append({
                "component_id": component_id,
                "classification": classification,
                "decision": None,
                "resolved_by": "WINDOW_AMBIGUOUS",
            })
            continue
        if drop_all:
            decision = {"kind": "DROP"}
            resolved_by = "DROP"
        elif classification == "EXACT_AGREEMENT":
            if component_id == custom_component_id:
                decision = {
                    "kind": "CUSTOM",
                    "token_start": 0,
                    "token_end": 0,
                    "node_type": "EVENT",
                }
                resolved_by = "CUSTOM"
            else:
                decision = {"kind": "KEEP_HUMAN_SET"}
                resolved_by = "PRE_RESOLVED"
        elif classification == "HUMAN_ONLY":
            decision = {"kind": "KEEP_HUMAN_SET"}
            resolved_by = "HUMAN_SET"
        elif classification == "SOL_ONLY":
            if component_id in accept_sol_component_ids:
                decision = {"kind": "KEEP_SOL_SET"}
                resolved_by = "SOL_SET"
            else:
                decision = {"kind": "DROP"}
                resolved_by = "DROP"
        elif classification in {"TYPE_DISAGREEMENT", "BOUNDARY_DISAGREEMENT"}:
            if component_id in accept_sol_component_ids:
                decision = {"kind": "KEEP_SOL_SET"}
                resolved_by = "SOL_SET"
            else:
                decision = {"kind": "KEEP_HUMAN_SET"}
                resolved_by = "HUMAN_SET"
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected classification {classification}")
        components.append({
            "component_id": component_id,
            "classification": classification,
            "decision": decision,
            "resolved_by": resolved_by,
        })
    return components


def _export_record(
    adjudication_record,
    *,
    index,
    outcome="CLEAN",
    note="",
    **kwargs,
):
    components = _export_components(adjudication_record, **kwargs)
    record = {
        "record_index": index,
        "window_id": adjudication_record["window_id"],
        "outcome": outcome,
        "note": note,
        "components": components,
        "resolved_endpoints": [],
    }
    record["resolved_endpoints"] = derive_resolved_endpoints(
        adjudication_record, record,
    )
    return record


def _build_export(
    adjudication,
    *,
    reviewer="import-reviewer",
    exported_at="2026-08-18T12:00:00Z",
    audit_checks=None,
    record_mutators=None,
):
    records = []
    for index, adjudication_record in enumerate(adjudication["records"], 1):
        mutator = (record_mutators or {}).get(index)
        record = _export_record(adjudication_record, index=index)
        if mutator:
            mutator(record, adjudication_record)
        records.append(record)
    return {
        "schema_version": ADJUDICATION_EXPORT_SCHEMA_VERSION,
        "adjudication_version": ADJUDICATION_VERSION,
        "packet_schema_version": "phase2j-endpoint-annotation-packet-v1",
        "adjudication_packet_sha256": adjudication["content_sha256"],
        "packet_sha256": adjudication["packet_sha256"],
        "human_session_sha256": adjudication["human_session_sha256"],
        "sol_review_sha256": adjudication["sol_review_sha256"],
        "status_label": "REVIEW_MATERIAL",
        "reviewer_name": reviewer,
        "exported_at": exported_at,
        "audit_checks": (
            audit_checks if audit_checks is not None else _all_true_audit_checks()
        ),
        "records": records,
    }


class Phase2JAdjudicationImportTests(unittest.TestCase):
    def _fixture(self, root: Path):
        """Write every input; return paths and parsed values."""
        packet = _build_packet(root)
        blank_path = _write(root / "blank-packet.json", dict(packet))
        manifest = _write_manifest(root)
        human_path = root / "human-session.json"
        sol_path = root / "sol-review.json"
        write_human_session(
            human_path, packet,
            spans_for_window=lambda record: _known_spans(record)[0],
        )
        write_sol_review(
            sol_path, packet,
            spans_for_window=lambda record: _known_spans(record)[1],
        )
        human = build_human_session(
            packet, spans_for_window=lambda record: _known_spans(record)[0],
        )
        sol = build_sol_review(
            packet, spans_for_window=lambda record: _known_spans(record)[1],
        )
        adjudication = build_adjudication_packet(
            packet, human, sol,
            human_session_path=human_path,
            sol_review_path=sol_path,
        )
        adjudication_path = _write(root / "adjudication.json", adjudication)
        export_path = _write(root / "export.json", _build_export(adjudication))
        return {
            "root": root,
            "packet": packet,
            "manifest": manifest,
            "human_path": human_path,
            "human": human,
            "sol_path": sol_path,
            "adjudication": adjudication,
            "adjudication_path": adjudication_path,
            "export_path": export_path,
            "blank_path": blank_path,
        }

    def _build(self, fixture, **kwargs):
        options = {
            "blank_packet_path": fixture["blank_path"],
            "manifest_path": fixture["root"] / "window-selection-manifest-v1.json",
            "human_session_path": fixture["human_path"],
            "adjudication_packet_path": fixture["adjudication_path"],
            "export_path": fixture["export_path"],
        }
        options.update(kwargs)
        return build_reviewed_packet(**options)

    def test_valid_clean_import_is_deterministic_and_canonically_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            reviewed = self._build(fixture)
            manifest = fixture["manifest"]
            validate_annotation_packet(reviewed, manifest=manifest)
            self.assertEqual(reviewed["release_gate"], RELEASE_GATE)
            self.assertIn("two-pass", reviewed["purpose"])
            self.assertIn("never gold", reviewed["purpose"])
            self.assertEqual(reviewed["content_sha256"], canonical_sha256({
                key: value for key, value in reviewed.items()
                if key != "content_sha256"
            }))
            self.assertTrue(is_packet_gold_eligible(reviewed))
            for index, record in enumerate(reviewed["records"], 1):
                self.assertEqual(record["window_status"], "REVIEWED")
                self.assertEqual(record["pass_a"]["status"], "COMPLETE")
                self.assertEqual(
                    record["pass_a"]["reviewer"],
                    fixture["human"]["records"][index - 1]["reviewer_name"],
                )
                self.assertEqual(
                    record["pass_a"]["endpoint_count"], len(record["endpoints"]),
                )
                self.assertIn("reconciliation", record["pass_a"]["notes"][0])
                self.assertEqual(record["pass_b"]["status"], "COMPLETE")
                self.assertEqual(record["pass_b"]["reviewer"], "import-reviewer")
                self.assertTrue(all(record["pass_b"]["audit_checks"].values()))
                self.assertEqual(record["ambiguity_controls"]["flagged"], False)
                self.assertEqual(record["exclusion_controls"]["flagged"], False)
                for position, endpoint in enumerate(record["endpoints"], 1):
                    self.assertEqual(
                        endpoint["endpoint_id"],
                        f"p2j:{record['window_id']}:ep:{str(position).zfill(4)}",
                    )
                    self.assertEqual(endpoint["disposition"], "KEEP")
                    self.assertEqual(endpoint["ambiguity_state"], "NONE")
                    self.assertFalse(endpoint["adjudication_requested"])
                    self.assertIn(
                        "adjudicated ", endpoint["notes"],
                    )
                ordered = sorted(
                    record["endpoints"],
                    key=lambda item: (
                        item["char_start"], item["char_end"], item["endpoint_id"],
                    ),
                )
                self.assertEqual(ordered, record["endpoints"])
            # Determinism.
            self.assertEqual(
                serialize_reviewed_packet(reviewed),
                serialize_reviewed_packet(self._build(fixture)),
            )

    def test_exact_input_hash_and_cross_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            adjudication = copy.deepcopy(dict(fixture["adjudication"]))
            adjudication["packet_sha256"] = "0" * 64
            adjudication = _rehash(adjudication)
            tampered_adjudication = _write(
                fixture["root"] / "adjudication-tampered.json", adjudication,
            )
            with self.assertRaisesRegex(ValueError, "not bound"):
                self._build(
                    fixture,
                    adjudication_packet_path=tampered_adjudication,
                )

            adjudication = copy.deepcopy(dict(fixture["adjudication"]))
            adjudication["human_session_sha256"] = "1" * 64
            adjudication = _rehash(adjudication)
            tampered_adjudication = _write(
                fixture["root"] / "adjudication-human-hash.json", adjudication,
            )
            with self.assertRaisesRegex(ValueError, "file hash"):
                self._build(
                    fixture,
                    adjudication_packet_path=tampered_adjudication,
                )

            export = json.loads(fixture["export_path"].read_text(encoding="utf-8"))
            export["adjudication_packet_sha256"] = "2" * 64
            tampered_export = _write(
                fixture["root"] / "export-wrong-hash.json", export,
            )
            with self.assertRaisesRegex(ValueError, "not bound"):
                self._build(fixture, export_path=tampered_export)

    def test_duplicate_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            body = fixture["export_path"].read_text(encoding="utf-8")
            duplicated = (
                '{\n  "schema_version": "phase2j-adjudication-export-v2",\n'
                + body[body.index("\n") + 1:]
            )
            duplicated_path = fixture["root"] / "export-duplicate.json"
            duplicated_path.write_text(duplicated, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate keys"):
                self._build(fixture, export_path=duplicated_path)

    def test_audit_checks_must_be_exact_and_all_true(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            adjudication = fixture["adjudication"]
            for missing in AUDIT_KEYS:
                checks = _all_true_audit_checks()
                del checks[missing]
                export = _build_export(adjudication, audit_checks=checks)
                with self.assertRaisesRegex(ValueError, "audit_checks"):
                    validate_adjudication_export(export, adjudication)
            checks = _all_true_audit_checks()
            checks["boundaries"] = False
            with self.assertRaisesRegex(ValueError, "every check to be true"):
                validate_adjudication_export(
                    _build_export(adjudication, audit_checks=checks), adjudication,
                )
            checks = _all_true_audit_checks()
            checks["invented"] = True
            with self.assertRaisesRegex(ValueError, "exactly the five"):
                validate_adjudication_export(
                    _build_export(adjudication, audit_checks=checks), adjudication,
                )
            checks = _all_true_audit_checks()
            checks["roles"] = "yes"
            with self.assertRaisesRegex(ValueError, "boolean"):
                validate_adjudication_export(
                    _build_export(adjudication, audit_checks=checks), adjudication,
                )

    def test_component_decision_and_derived_endpoint_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            adjudication = fixture["adjudication"]
            record_0 = adjudication["records"][0]
            exact_id = next(
                component["component_id"]
                for component in record_0["components"]
                if component["classification"] == "EXACT_AGREEMENT"
            )

            def disallowed_decision(record, adjudication_record):
                for entry in record["components"]:
                    if entry["component_id"] == exact_id:
                        entry["decision"] = {"kind": "KEEP_SOL_SET"}
                        entry["resolved_by"] = "SOL_SET"

            export = _build_export(
                adjudication, record_mutators={1: disallowed_decision},
            )
            with self.assertRaisesRegex(ValueError, "not allowed"):
                validate_adjudication_export(export, adjudication)

            def mismatched_resolved_by(record, adjudication_record):
                for entry in record["components"]:
                    if entry["component_id"] == exact_id:
                        entry["resolved_by"] = "CUSTOM"

            export = _build_export(
                adjudication, record_mutators={1: mismatched_resolved_by},
            )
            with self.assertRaisesRegex(ValueError, "resolved_by"):
                validate_adjudication_export(export, adjudication)

            def reordered_components(record, adjudication_record):
                record["components"] = list(reversed(record["components"]))

            export = _build_export(
                adjudication, record_mutators={1: reordered_components},
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_adjudication_export(export, adjudication)

            export = _build_export(adjudication)
            export["records"][0]["resolved_endpoints"][0]["exact_bronze_text"] = (
                "tampered"
            )
            with self.assertRaisesRegex(ValueError, "Bronze slice"):
                validate_adjudication_export(export, adjudication)

    def test_unknown_and_extra_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            export = json.loads(fixture["export_path"].read_text(encoding="utf-8"))
            export["invented"] = True
            with self.assertRaisesRegex(ValueError, "envelope"):
                validate_adjudication_export(export, fixture["adjudication"])
            export = json.loads(fixture["export_path"].read_text(encoding="utf-8"))
            export["records"][0]["extra"] = True
            with self.assertRaisesRegex(ValueError, "record 1 is invalid"):
                validate_adjudication_export(export, fixture["adjudication"])
            export = json.loads(fixture["export_path"].read_text(encoding="utf-8"))
            export["records"][0]["components"][0]["extra"] = True
            with self.assertRaisesRegex(ValueError, "component"):
                validate_adjudication_export(export, fixture["adjudication"])

    def test_reviewer_and_timestamp_requirements(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            adjudication = fixture["adjudication"]
            for reviewer in ("", "   "):
                with self.assertRaisesRegex(ValueError, "reviewer_name"):
                    validate_adjudication_export(
                        _build_export(adjudication, reviewer=reviewer),
                        adjudication,
                    )
            with self.assertRaisesRegex(ValueError, "exported_at"):
                validate_adjudication_export(
                    _build_export(adjudication, exported_at=""), adjudication,
                )

    def test_exact_bronze_span_and_no_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            export = json.loads(fixture["export_path"].read_text(encoding="utf-8"))
            record_0 = export["records"][0]
            endpoint = record_0["resolved_endpoints"][0]
            endpoint["char_start"] = endpoint["char_start"] + 1
            with self.assertRaisesRegex(ValueError, "bounds are inconsistent"):
                validate_adjudication_export(export, fixture["adjudication"])

            export = json.loads(fixture["export_path"].read_text(encoding="utf-8"))
            export["records"][0]["resolved_endpoints"][0]["char_end"] = (
                export["records"][0]["resolved_endpoints"][0]["char_start"] + 1
            )
            with self.assertRaisesRegex(ValueError, "bounds|derived"):
                validate_adjudication_export(export, fixture["adjudication"])

            adjudication = fixture["adjudication"]
            record_0 = adjudication["records"][0]

            def overlapping_custom(record, adjudication_record):
                # Second component's custom span overlaps the exact-agreement
                # human span at token 0-1.
                for entry in record["components"]:
                    if entry["classification"] == "EXACT_AGREEMENT":
                        entry["decision"] = {"kind": "DROP"}
                        entry["resolved_by"] = "DROP"
                    if entry["classification"] == "TYPE_DISAGREEMENT":
                        entry["decision"] = {
                            "kind": "CUSTOM",
                            "token_start": 2,
                            "token_end": 3,
                            "node_type": "EVENT",
                        }
                        entry["resolved_by"] = "CUSTOM"

            export = _build_export(
                adjudication, record_mutators={1: overlapping_custom},
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_adjudication_export(export, adjudication)

    def test_sol_requires_explicit_choice_and_provenance_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            adjudication = fixture["adjudication"]
            record_0 = adjudication["records"][0]
            sol_only_id = next(
                component["component_id"]
                for component in record_0["components"]
                if component["classification"] == "SOL_ONLY"
            )
            boundary_id = next(
                component["component_id"]
                for component in record_0["components"]
                if component["classification"] == "BOUNDARY_DISAGREEMENT"
            )
            def accept_sol(record, adjudication_record):
                for entry in record["components"]:
                    if entry["component_id"] in (sol_only_id, boundary_id):
                        entry["decision"] = {"kind": "KEEP_SOL_SET"}
                        entry["resolved_by"] = "SOL_SET"
                record["resolved_endpoints"] = derive_resolved_endpoints(
                    adjudication_record, record,
                )

            export = _build_export(
                adjudication, record_mutators={1: accept_sol},
            )
            validate_adjudication_export(export, adjudication)
            export_path = _write(fixture["root"] / "export-sol.json", export)
            reviewed = self._build(fixture, export_path=export_path)
            record = reviewed["records"][0]
            pass_b_ids = [
                endpoint["endpoint_id"]
                for endpoint in record["endpoints"]
                if endpoint["pass_provenance"] == "PASS_B"
            ]
            self.assertEqual(len(pass_b_ids), 2)
            pass_a_ids = [
                endpoint["endpoint_id"]
                for endpoint in record["endpoints"]
                if endpoint["pass_provenance"] == "PASS_A"
            ]
            self.assertTrue(pass_a_ids)
            for endpoint in record["endpoints"]:
                if endpoint["pass_provenance"] == "PASS_B":
                    self.assertIn("SOL", endpoint["notes"])
                elif endpoint["pass_provenance"] == "PASS_A":
                    self.assertTrue(
                        "source HUMAN" in endpoint["notes"]
                        or "source SHARED" in endpoint["notes"],
                    )

    def test_undetermined_maps_to_none(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet = _build_packet(root)
            manifest = _write_manifest(root)
            blank_path = _write(root / "blank-packet.json", dict(packet))

            def spans_with_undetermined(record):
                human, sol = _known_spans(record)
                if record["record_index"] == 1:
                    human = list(human)
                    human[0] = (0, 1, "UNDETERMINED")
                return human, sol

            human_path = root / "human-session.json"
            sol_path = root / "sol-review.json"
            write_human_session(
                human_path, packet,
                spans_for_window=lambda record: spans_with_undetermined(record)[0],
            )
            write_sol_review(
                sol_path, packet,
                spans_for_window=lambda record: spans_with_undetermined(record)[1],
            )
            human = build_human_session(
                packet,
                spans_for_window=lambda record: spans_with_undetermined(record)[0],
            )
            sol = build_sol_review(
                packet,
                spans_for_window=lambda record: spans_with_undetermined(record)[1],
            )
            adjudication = build_adjudication_packet(
                packet, human, sol,
                human_session_path=human_path,
                sol_review_path=sol_path,
            )
            adjudication_path = _write(
                root / "adjudication.json", adjudication,
            )
            # Record 1's first component becomes a type disagreement; keep the
            # human set so the UNDETERMINED type flows through the export.
            export = _build_export(adjudication)
            export_path = _write(root / "export.json", export)
            reviewed = build_reviewed_packet(
                blank_packet_path=blank_path,
                manifest_path=root / "window-selection-manifest-v1.json",
                human_session_path=human_path,
                adjudication_packet_path=adjudication_path,
                export_path=export_path,
            )
            record = reviewed["records"][0]
            self.assertTrue(any(
                endpoint["node_type"] is None for endpoint in record["endpoints"]
            ))
            validate_annotation_packet(reviewed, manifest=manifest)

    def test_clean_zero_endpoints_and_unresolved_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            adjudication = fixture["adjudication"]
            export = _build_export(
                adjudication,
                record_mutators={
                    1: lambda record, adj: [
                        entry.update({"decision": {"kind": "DROP"}, "resolved_by": "DROP"})
                        for entry in record["components"]
                    ],
                },
            )
            with self.assertRaisesRegex(ValueError, "zero endpoints"):
                validate_adjudication_export(export, adjudication)

            export = _build_export(
                adjudication,
                record_mutators={
                    1: lambda record, adj: record["components"][0].update(
                        {"decision": None, "resolved_by": "WINDOW_AMBIGUOUS"},
                    ),
                },
            )
            with self.assertRaisesRegex(ValueError, "unresolved"):
                validate_adjudication_export(export, adjudication)

    def test_ambiguous_and_excluded_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            adjudication = fixture["adjudication"]

            def ambiguous_record(record, adjudication_record):
                record["outcome"] = "AMBIGUOUS"
                record["note"] = "genuinely unclear context"
                for entry in record["components"]:
                    if entry["classification"] != "EXACT_AGREEMENT":
                        entry["decision"] = None
                        entry["resolved_by"] = "WINDOW_AMBIGUOUS"
                record["resolved_endpoints"] = derive_resolved_endpoints(
                    adjudication_record, record,
                )

            def excluded_record(record, adjudication_record):
                record["outcome"] = "EXCLUDED"
                record["note"] = "unusable ASR"
                for entry in record["components"]:
                    entry["decision"] = None
                    entry["resolved_by"] = "WINDOW_EXCLUDED"
                record["resolved_endpoints"] = derive_resolved_endpoints(
                    adjudication_record, record,
                )

            export = _build_export(
                adjudication,
                record_mutators={1: ambiguous_record, 2: excluded_record},
            )
            validate_adjudication_export(export, adjudication)
            export_path = _write(fixture["root"] / "export-mixed.json", export)
            reviewed = self._build(fixture, export_path=export_path)
            validate_annotation_packet(reviewed, manifest=fixture["manifest"])
            ambiguous = reviewed["records"][0]
            excluded = reviewed["records"][1]
            self.assertEqual(ambiguous["window_status"], "AMBIGUOUS")
            self.assertEqual(ambiguous["pass_b"]["status"], "IN_PROGRESS")
            self.assertIsNone(ambiguous["pass_b"]["reviewer"])
            self.assertIsNone(ambiguous["pass_b"]["completed_at"])
            self.assertTrue(ambiguous["ambiguity_controls"]["flagged"])
            self.assertIn("genuinely unclear", ambiguous["ambiguity_controls"]["notes"][0])
            self.assertTrue(ambiguous["endpoints"])
            self.assertFalse(is_window_gold_eligible(ambiguous))
            self.assertEqual(excluded["window_status"], "EXCLUDED")
            self.assertEqual(excluded["endpoints"], [])
            self.assertTrue(excluded["exclusion_controls"]["flagged"])
            self.assertIn("unusable ASR", excluded["exclusion_controls"]["notes"][0])
            self.assertEqual(excluded["pass_b"]["status"], "COMPLETE")
            self.assertEqual(excluded["pass_b"]["reviewer"], "import-reviewer")
            self.assertFalse(is_window_gold_eligible(excluded))
            self.assertFalse(is_packet_gold_eligible(reviewed))

    def test_release_gate_locked_and_no_forbidden_model_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            reviewed = self._build(fixture)
            self.assertEqual(reviewed["release_gate"], "LOCKED")
            validate_annotation_packet(reviewed, manifest=fixture["manifest"])

            def forbidden_keys(value):
                found = []
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key.casefold() in FORBIDDEN_KEYS:
                            found.append(key)
                        found.extend(forbidden_keys(item))
                elif isinstance(value, list):
                    for item in value:
                        found.extend(forbidden_keys(item))
                return found

            self.assertEqual(forbidden_keys(dict(reviewed)), [])
            self.assertEqual(reviewed["candidate_catalog"], fixture["packet"]["candidate_catalog"])
            self.assertEqual(reviewed["rules"], fixture["packet"]["rules"])
            self.assertEqual(
                reviewed["selection_manifest_sha256"],
                fixture["manifest"]["content_sha256"],
            )
            for record in reviewed["records"]:
                self.assertIn(
                    record["partition"],
                    {"EXPANDED_DEV", "FROZEN_REPLICATION"},
                )


if __name__ == "__main__":
    unittest.main()
