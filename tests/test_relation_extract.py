import json
import sqlite3
import tempfile
import unittest

from core.champions import canonical_champion_name
from pipeline.relation_extract import (
    EvidenceItem,
    ExtractionPacket,
    compile_candidates,
    extract_relations,
    extract_relation_trace,
    parse_model_response,
    packet_from_insight_ids,
    _positive_env_int,
    _model_env,
)


class RelationExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.packet = ExtractionPacket(
            evidence=(
                EvidenceItem(
                    insight_id="4798",
                    source_id="z5IXabhMLzQ",
                    text="Thresh Flay denies continued contact.",
                    confidence=0.62,
                ),
            ),
            ability_aliases={"Thresh Flay": "Thresh E", "Flay": "Thresh E", "Thresh E": "Thresh E"},
        )

    def test_canonical_champion_aliases_reuse_existing_registry(self) -> None:
        self.assertEqual(canonical_champion_name("kaisa"), "Kai'Sa")
        self.assertIsNone(canonical_champion_name("not a champion"))

    def test_aliases_and_evidence_provenance_compile_to_phase1_relation(self) -> None:
        decision = compile_candidates(self.packet, [_candidate(condition=None)])[0]

        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.relation.subject_key, "Thresh E")
        self.assertEqual(decision.relation.relation_type, "denies")
        self.assertEqual(decision.relation.object_key, "continuity")
        self.assertIsNone(decision.relation.condition)
        self.assertEqual(decision.relation.evidence_refs[0].insight_id, "4798")
        self.assertEqual(decision.relation.data_version, "strategic-relations-v0")
        self.assertGreater(decision.relation.confidence, 0.6)
        self.assertEqual(
            {(item.field, item.source_text, item.canonical_value) for item in decision.relation.alignments},
            {
                ("subject", "Flay", "Thresh E"),
                ("predicate", "denies", "denies"),
                ("object", "continued contact", "continuity"),
            },
        )

    def test_missing_fabricated_or_mismatched_source_grounding_is_rejected(self) -> None:
        cases = [
            ({"grounding": None}, "missing source grounding"),
            ({"grounding": {
                "subject": {"source_text": "Flay", "evidence_id": "4798"},
                "predicate": {"source_text": "denies", "evidence_id": "4798"},
                "object": {"source_text": "stays on target", "evidence_id": "4798"},
            }}, "object source phrase is not present"),
            ({"grounding": {
                "subject": {"source_text": "Flay", "evidence_id": "4798"},
                "predicate": {"source_text": "denies", "evidence_id": "4798"},
                "object": {"source_text": "Flay", "evidence_id": "4798"},
            }}, "object source phrase does not support"),
        ]
        for override, expected in cases:
            with self.subTest(expected=expected):
                decision = compile_candidates(self.packet, [_candidate(**override)])[0]
                self.assertEqual(decision.status, "rejected")
                self.assertIn(expected, decision.warnings[0])

    def test_parenthetical_ability_alias_resolves_and_unsupported_concepts_are_removed(self) -> None:
        packet = ExtractionPacket(
            evidence=(EvidenceItem("4798", "video-1", "Flay breaks continued contact.", confidence=0.8),),
            ability_aliases=self.packet.ability_aliases,
        )
        candidate = _candidate(
            subject="Flay (E)",
            concepts=["tempo", "intermittent pressure"],
            condition=None,
            _source_predicate="breaks",
            _source_subject="Flay",
        )
        decision = compile_candidates(packet, [candidate])[0]

        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.relation.subject_key, "Thresh E")
        self.assertEqual(decision.relation.concepts, ("continuity",))
        self.assertIn("removed unsupported strategic concepts", decision.warnings[0])

    def test_direct_concept_endpoint_must_be_supported_by_source_or_alias(self) -> None:
        rejected = compile_candidates(self.packet, [_candidate(
            object="access", object_type="concept", concepts=["access"], _source_predicate="denies"
        )])[0]
        self.assertEqual(rejected.status, "rejected")
        self.assertIn("unsupported strategic concept", rejected.warnings[0])

        packet = ExtractionPacket(
            evidence=(EvidenceItem("4798", "video-1", "Flay breaks continued contact.", confidence=0.8),),
            ability_aliases=self.packet.ability_aliases,
        )
        accepted = compile_candidates(packet, [_candidate(condition=None, _source_predicate="breaks")])[0]
        self.assertEqual(accepted.status, "accepted")
        self.assertIn("continuity", accepted.relation.concepts)

    def test_semantic_evidence_cue_can_ground_existing_concept_without_literal_name(self) -> None:
        packet = ExtractionPacket(
            evidence=(EvidenceItem("lux", "video", "Lux E denies forward access before you walk up to farm.", confidence=.8),),
            ability_aliases={"E": "Lux E"},
        )
        candidate = _candidate(subject="E", object="access", object_type="concept", concepts=["access"], evidence_ids=["lux"], condition="before you walk up to farm", _source_predicate="denies", _source_object="forward access")
        decision = compile_candidates(packet, [candidate])[0]
        self.assertEqual(decision.status, "accepted")

    def test_condition_must_be_grounded_in_cited_evidence(self) -> None:
        decision = compile_candidates(self.packet, [_candidate(condition="while Flay is available")])[0]
        self.assertEqual(decision.status, "rejected")
        self.assertIn("condition is not supported", decision.warnings[0])

    def test_condition_support_preserves_negation_order_and_single_evidence_scope(self) -> None:
        packet = ExtractionPacket(
            evidence=(
                EvidenceItem("one", "video", "Lux E denies access: do not walk up to farm while Lux E is available.", confidence=.8),
                EvidenceItem("two", "video", "The target is isolated.", confidence=.8),
            ), ability_aliases={"E": "Lux E"},
        )
        base = _candidate(subject="E", object="access", object_type="concept", concepts=["access"], evidence_ids=["one"], condition="do not walk up to farm while Lux E is available")
        self.assertEqual(compile_candidates(packet, [base])[0].status, "accepted")
        for condition, ids in (("walk up to farm while Lux E is available", ["one"]), ("Lux E is available before walk up to farm", ["one"]), ("walk up to farm target isolated", ["one", "two"])):
            with self.subTest(condition=condition):
                decision = compile_candidates(packet, [_candidate(subject="E", object="access", object_type="concept", concepts=["access"], evidence_ids=ids, condition=condition)])[0]
                self.assertEqual(decision.status, "rejected")

    def test_condition_anchor_cannot_be_unrelated_nearby_evidence_text(self) -> None:
        packet = ExtractionPacket(
            evidence=(EvidenceItem(
                "one", "video", "Flay denies continued contact while available. Lux was nearby.", confidence=.8
            ),),
            ability_aliases=self.packet.ability_aliases,
        )
        candidate = _candidate(
            evidence_ids=["one"], condition="while available", _source_condition="Lux"
        )
        decision = compile_candidates(packet, [candidate])[0]
        self.assertEqual(decision.status, "rejected")
        self.assertIn("condition source phrase does not support", decision.warnings[0])

    def test_condition_changes_stable_identity_and_is_not_overmerged(self) -> None:
        packet = ExtractionPacket((EvidenceItem("4798", "video", "Flay denies continued contact while available, after Tristana lands W.", confidence=.8),), ability_aliases=self.packet.ability_aliases)
        first = compile_candidates(packet, [_candidate(
            condition="while Flay is available", _source_condition="while available"
        )])[0].relation
        second = compile_candidates(packet, [_candidate(condition="after Tristana lands W")])[0].relation

        self.assertNotEqual(first.id, second.id)
        self.assertNotEqual(first.stable_key(), second.stable_key())

    def test_unknown_concept_verb_and_evidence_are_rejected(self) -> None:
        for override, expected in [
            ({"object": "made_up_concept"}, "unknown entity"),
            ({"relation_type": "sort of helps"}, "unknown entity"),
            ({"evidence_ids": ["not-in-packet"]}, "unknown evidence_id"),
        ]:
            candidate = _candidate(**override)
            with self.subTest(override=override):
                decision = compile_candidates(self.packet, [candidate])[0]
                self.assertEqual(decision.status, "rejected")
                self.assertIn(expected, decision.warnings[0])

    def test_malformed_qualifiers_unknown_abilities_and_duplicate_evidence_are_rejected(self) -> None:
        for override, expected in [
            ({"condition": ["after Q misses"]}, "condition must be"),
            ({"effect": {"event": "x"}}, "effect must be"),
            ({"subject": "Thresh Banana"}, "unknown entity"),
            ({"evidence_ids": ["4798", "4798"]}, "duplicate evidence_id"),
            ({"relation_type": "counters"}, "unknown entity"),
            ({"object_type": "event", "object": "invented artifact"}, "unknown entity"),
        ]:
            with self.subTest(override=override):
                decision = compile_candidates(self.packet, [_candidate(**override)])[0]
                self.assertEqual(decision.status, "rejected")
                self.assertIn(expected, decision.warnings[0])

    def test_packet_registered_state_is_the_only_allowed_state_node(self) -> None:
        packet = ExtractionPacket(
            evidence=self.packet.evidence,
            ability_aliases=self.packet.ability_aliases,
            entity_aliases={"state": {"after q misses": "after Q misses"}},
        )
        packet = ExtractionPacket(
            evidence=(EvidenceItem("4798", "video", "Thresh Flay denies continued contact after Q misses.", confidence=.62),),
            ability_aliases=self.packet.ability_aliases,
            entity_aliases={"state": {"after q misses": "after Q misses"}},
        )
        candidate = _candidate(object_type="state", object="after Q misses", concepts=["access"], condition=None, _source_object="after Q misses")
        decision = compile_candidates(packet, [candidate])[0]
        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.relation.object_key, "after Q misses")

    def test_source_type_and_patch_sensitivity_are_not_weakened(self) -> None:
        packet = ExtractionPacket(
            evidence=(EvidenceItem("guide-1", "guide-article", "Flay denies continued contact.", source_type="guide", confidence=0.8, patch_sensitivity="high"),),
            ability_aliases={"Flay": "Thresh E"},
        )
        decision = compile_candidates(packet, [_candidate(evidence_ids=["guide-1"], patch_sensitivity="low", condition=None)])[0]
        self.assertEqual(decision.relation.evidence_refs[0].source_type, "guide")
        self.assertEqual(decision.relation.patch_sensitivity, "high")

    def test_empty_relations_is_valid_and_malformed_response_is_rejected(self) -> None:
        self.assertEqual(parse_model_response('{"relations": []}'), [])
        self.assertEqual(parse_model_response('```json\n{"relations": []}\n```'), [])
        for raw in ("", "[]", '{"relations": ["bad"]}', '{"other": []}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_model_response(raw)

    def test_json_fence_with_surrounding_prose_is_not_accepted(self) -> None:
        for raw in (
            'Here is the output:\n```json\n{"relations": []}\n```',
            '```json\n{"relations": []}\n```\n```json\n{"relations": []}\n```',
            '```json\n{"relations": []}\n```\ntrailing text',
            '```python\n{"relations": []}\n```',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_model_response(raw)

    def test_low_confidence_relation_is_held_for_review(self) -> None:
        candidate = _candidate(extraction_confidence=0.1)
        decision = extract_relations(
            self.packet,
            lambda **_: json.dumps({"relations": [candidate]}),
            acceptance_threshold=0.6,
        )[0]
        self.assertEqual(decision.status, "review")
        self.assertIsNotNone(decision.relation)

    def test_same_condition_opposite_relations_are_quarantined_for_review(self) -> None:
        packet = ExtractionPacket(
            evidence=(EvidenceItem("4798", "video", "Flay creates and denies continued contact.", confidence=.8),),
            ability_aliases=self.packet.ability_aliases,
        )
        creates = _candidate(relation_type="creates", condition=None, _source_predicate="creates")
        denies = _candidate(relation_type="denies", condition=None)
        decisions = compile_candidates(packet, [creates, denies])

        self.assertEqual([item.status for item in decisions], ["review", "review"])
        self.assertTrue(all("contradictory" in item.warnings[-1] for item in decisions))

    def test_distinct_conditions_do_not_trigger_contradiction_review(self) -> None:
        packet = ExtractionPacket((EvidenceItem("4798", "video", "Flay creates continued contact when unavailable and denies it while available.", confidence=.8),), ability_aliases=self.packet.ability_aliases)
        creates = _candidate(
            relation_type="creates", condition="when Flay is unavailable",
            _source_predicate="creates", _source_condition="when unavailable",
        )
        denies = _candidate(
            relation_type="denies", condition="while Flay is available",
            _source_condition="while available",
        )
        decisions = compile_candidates(packet, [creates, denies])

        self.assertEqual([item.status for item in decisions], ["accepted", "accepted"])

    def test_model_is_prompted_only_with_packet_evidence_and_constraints(self) -> None:
        captured = {}
        extract_relations(
            self.packet,
            lambda **kwargs: captured.update(kwargs) or '{"relations": []}',
        )
        self.assertIn("SOURCE EVIDENCE", captured["user"])
        self.assertIn("evidence_id=4798", captured["user"])
        self.assertIn("Allowed relation types", captured["user"])
        self.assertIn("Do not use League knowledge", captured["system"])
        self.assertIn("Copy condition wording", captured["system"])
        self.assertIn("exact source phrase", captured["user"])
        self.assertEqual(captured["max_tokens"], 4096)
        self.assertEqual(captured["thinking"], "disabled")

    def test_explicit_relation_model_is_forwarded_to_existing_provider_adapter(self) -> None:
        captured = {}
        extract_relations(
            self.packet,
            lambda **kwargs: captured.update(kwargs) or '{"relations": []}',
            model="deepseek-v4-pro",
        )
        self.assertEqual(captured["model"], "deepseek-v4-pro")

    def test_model_failure_propagates_without_creating_a_decision(self) -> None:
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            extract_relations(
                self.packet,
                lambda **_: (_ for _ in ()).throw(TimeoutError("timed out")),
            )

    def test_trace_retains_malformed_response_and_parsing_stage(self) -> None:
        trace = extract_relation_trace(self.packet, lambda **_: "not-json")
        self.assertEqual(trace.failure_stage, "parsing")
        self.assertEqual(trace.raw_response, "not-json")
        self.assertFalse(trace.decisions)

    def test_output_budget_config_rejects_malformed_or_non_positive_values(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"RELATION_EXTRACTION_MAX_TOKENS": "8192"}):
            self.assertEqual(_positive_env_int("RELATION_EXTRACTION_MAX_TOKENS", 4096), 8192)
        for value in ("0", "-1", "not-a-number"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"RELATION_EXTRACTION_MAX_TOKENS": value}):
                with self.assertRaisesRegex(RuntimeError, "RELATION_EXTRACTION_MAX_TOKENS"):
                    _positive_env_int("RELATION_EXTRACTION_MAX_TOKENS", 4096)

    def test_relation_model_config_rejects_blank_value(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"DEEPSEEK_RELATION_PRO_MODEL": "  "}):
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_RELATION_PRO_MODEL"):
                _model_env("DEEPSEEK_RELATION_PRO_MODEL", "deepseek-v4-pro")

    def test_packet_loader_uses_explicit_insights_and_existing_ability_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/evidence.db"
            conn = sqlite3.connect(path)
            conn.executescript(
                """
                CREATE TABLE videos (video_id TEXT PRIMARY KEY, champion TEXT, subject TEXT);
                CREATE TABLE insights (id INTEGER PRIMARY KEY, video_id TEXT, text TEXT, source_score REAL, cluster_score REAL, confidence REAL);
                CREATE TABLE champion_abilities (champion TEXT, ability_slot TEXT, name TEXT);
                INSERT INTO videos VALUES ('video-1', 'Thresh', NULL);
                INSERT INTO insights VALUES (9, 'video-1', 'Hold Flay for the engage.', .8, .6, .7);
                INSERT INTO champion_abilities VALUES ('Thresh', 'E', 'Flay');
                INSERT INTO champion_abilities VALUES ('Thresh', 'Q', 'Death Sentence');
                """
            )
            conn.commit()
            conn.close()
            packet = packet_from_insight_ids(path, ["9"])
            with self.assertRaisesRegex(ValueError, "unknown insight IDs"):
                packet_from_insight_ids(path, ["missing"])

        self.assertEqual(packet.evidence[0].source_id, "video-1")
        self.assertEqual(packet.ability_aliases["Flay"], "Thresh E")
        self.assertNotIn("Thresh Q", packet.ability_aliases.values())

    def test_packet_loader_resolves_bare_slot_only_for_one_grounded_champion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/evidence.db"
            conn = sqlite3.connect(path)
            conn.executescript("""
                CREATE TABLE videos (video_id TEXT PRIMARY KEY, champion TEXT, subject TEXT);
                CREATE TABLE insights (id INTEGER PRIMARY KEY, video_id TEXT, text TEXT, source_score REAL, cluster_score REAL, confidence REAL);
                CREATE TABLE champion_abilities (champion TEXT, ability_slot TEXT, name TEXT);
                INSERT INTO videos VALUES ('video-1', 'Lux', NULL);
                INSERT INTO insights VALUES (9, 'video-1', 'Bait her E before walking up.', .8, .6, .7);
                INSERT INTO champion_abilities VALUES ('Lux', 'E', 'Lucent Singularity');
            """)
            conn.commit(); conn.close()
            packet = packet_from_insight_ids(path, ["9"])
        self.assertEqual(packet.ability_aliases["E"], "Lux E")

    def test_packet_loader_does_not_resolve_bare_slot_for_multiple_champions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/evidence.db"
            conn = sqlite3.connect(path)
            conn.executescript("""
                CREATE TABLE videos (video_id TEXT PRIMARY KEY, champion TEXT, subject TEXT);
                CREATE TABLE insights (id INTEGER PRIMARY KEY, video_id TEXT, text TEXT, source_score REAL, cluster_score REAL, confidence REAL);
                CREATE TABLE champion_abilities (champion TEXT, ability_slot TEXT, name TEXT);
                INSERT INTO videos VALUES ('video-1', 'Lux', 'Lux versus Syndra');
                INSERT INTO insights VALUES (9, 'video-1', 'Bait her E before walking up.', .8, .6, .7);
                INSERT INTO champion_abilities VALUES ('Lux', 'E', 'Lucent Singularity');
                INSERT INTO champion_abilities VALUES ('Syndra', 'E', 'Scatter the Weak');
            """)
            conn.commit(); conn.close()
            packet = packet_from_insight_ids(path, ["9"])
        self.assertNotIn("E", packet.ability_aliases)

    def test_packet_loader_does_not_allow_unmentioned_abilities_for_named_champion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/evidence.db"
            conn = sqlite3.connect(path)
            conn.executescript(
                """
                CREATE TABLE videos (video_id TEXT PRIMARY KEY, champion TEXT, subject TEXT);
                CREATE TABLE insights (id INTEGER PRIMARY KEY, video_id TEXT, text TEXT, source_score REAL, cluster_score REAL, confidence REAL);
                CREATE TABLE champion_abilities (champion TEXT, ability_slot TEXT, name TEXT);
                INSERT INTO videos VALUES ('video-1', 'Thresh', NULL);
                INSERT INTO insights VALUES (9, 'video-1', 'Thresh should wait before engaging.', .8, .6, .7);
                INSERT INTO champion_abilities VALUES ('Thresh', 'E', 'Flay');
                INSERT INTO champion_abilities VALUES ('Thresh', 'Q', 'Death Sentence');
                """
            )
            conn.commit()
            conn.close()
            packet = packet_from_insight_ids(path, ["9"])

        self.assertFalse(packet.ability_aliases)


def _candidate(**overrides):
    source_subject = overrides.pop("_source_subject", None)
    source_predicate = overrides.pop("_source_predicate", "denies")
    source_object = overrides.pop("_source_object", None)
    source_condition = overrides.pop("_source_condition", None)
    value = {
        "subject": "Flay",
        "subject_type": "ability",
        "relation_type": "breaks",
        "object": "continued contact",
        "object_type": "concept",
        "condition": None,
        "effect": "the enemy cannot sustain contact",
        "concepts": ["continued contact", "threat preservation"],
        "provenance_type": "coach_supported_inference",
        "evidence_ids": ["4798"],
        "extraction_confidence": 0.9,
        "patch_sensitivity": "low",
    }
    value.update(overrides)
    if "grounding" not in overrides:
        value["grounding"] = {
            "subject": {"source_text": source_subject or value["subject"], "evidence_id": value["evidence_ids"][0]},
            "predicate": {"source_text": source_predicate, "evidence_id": value["evidence_ids"][0]},
            "object": {"source_text": source_object or value["object"], "evidence_id": value["evidence_ids"][0]},
            "condition": (
                {"source_text": source_condition or value["condition"], "evidence_id": value["evidence_ids"][0]}
                if isinstance(value["condition"], str)
                else None
            ),
        }
    return value


if __name__ == "__main__":
    unittest.main()
