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
        decision = compile_candidates(self.packet, [_candidate()])[0]

        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.relation.subject_key, "Thresh E")
        self.assertEqual(decision.relation.relation_type, "denies")
        self.assertEqual(decision.relation.object_key, "continuity")
        self.assertEqual(decision.relation.condition, "while Flay is available")
        self.assertEqual(decision.relation.evidence_refs[0].insight_id, "4798")
        self.assertEqual(decision.relation.data_version, "strategic-relations-v0")
        self.assertGreater(decision.relation.confidence, 0.6)

    def test_parenthetical_ability_alias_resolves_and_unsupported_concepts_are_removed(self) -> None:
        packet = ExtractionPacket(
            evidence=(EvidenceItem("4798", "video-1", "Flay breaks continued contact.", confidence=0.8),),
            ability_aliases=self.packet.ability_aliases,
        )
        candidate = _candidate(subject="Flay (E)", concepts=["tempo", "intermittent pressure"])
        decision = compile_candidates(packet, [candidate])[0]

        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.relation.subject_key, "Thresh E")
        self.assertEqual(decision.relation.concepts, ("continuity",))
        self.assertIn("removed unsupported strategic concepts", decision.warnings[0])

    def test_direct_concept_endpoint_must_be_supported_by_source_or_alias(self) -> None:
        rejected = compile_candidates(self.packet, [_candidate(object="access", object_type="concept", concepts=["access"])])[0]
        self.assertEqual(rejected.status, "rejected")
        self.assertIn("unsupported strategic concept", rejected.warnings[0])

        packet = ExtractionPacket(
            evidence=(EvidenceItem("4798", "video-1", "Flay breaks continued contact.", confidence=0.8),),
            ability_aliases=self.packet.ability_aliases,
        )
        accepted = compile_candidates(packet, [_candidate()])[0]
        self.assertEqual(accepted.status, "accepted")
        self.assertIn("continuity", accepted.relation.concepts)

    def test_condition_changes_stable_identity_and_is_not_overmerged(self) -> None:
        first = compile_candidates(self.packet, [_candidate(condition="while Flay is available")])[0].relation
        second = compile_candidates(self.packet, [_candidate(condition="after Tristana lands W")])[0].relation

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
        candidate = _candidate(object_type="state", object="after Q misses", concepts=["access"])
        decision = compile_candidates(packet, [candidate])[0]
        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.relation.object_key, "after Q misses")

    def test_source_type_and_patch_sensitivity_are_not_weakened(self) -> None:
        packet = ExtractionPacket(
            evidence=(EvidenceItem("guide-1", "guide-article", "Flay denies continued contact.", source_type="guide", confidence=0.8, patch_sensitivity="high"),),
            ability_aliases={"Flay": "Thresh E"},
        )
        decision = compile_candidates(packet, [_candidate(evidence_ids=["guide-1"], patch_sensitivity="low")])[0]
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
        creates = _candidate(relation_type="creates")
        denies = _candidate(relation_type="denies")
        decisions = compile_candidates(self.packet, [creates, denies])

        self.assertEqual([item.status for item in decisions], ["review", "review"])
        self.assertTrue(all("contradictory" in item.warnings[-1] for item in decisions))

    def test_distinct_conditions_do_not_trigger_contradiction_review(self) -> None:
        creates = _candidate(relation_type="creates", condition="when Flay is unavailable")
        denies = _candidate(relation_type="denies", condition="while Flay is available")
        decisions = compile_candidates(self.packet, [creates, denies])

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
    value = {
        "subject": "Flay",
        "subject_type": "ability",
        "relation_type": "breaks",
        "object": "continued contact",
        "object_type": "concept",
        "condition": "while Flay is available",
        "effect": "the enemy cannot sustain contact",
        "concepts": ["continued contact", "threat preservation"],
        "provenance_type": "coach_supported_inference",
        "evidence_ids": ["4798"],
        "extraction_confidence": 0.9,
        "patch_sensitivity": "low",
    }
    value.update(overrides)
    return value


if __name__ == "__main__":
    unittest.main()
