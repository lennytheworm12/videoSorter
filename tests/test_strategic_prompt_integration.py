import unittest
from unittest import mock

import retrieval.query as query
from retrieval.strategic_context import StrategicContext, format_strategic_context


EVIDENCE = [{
    "text": "Use trap pressure to control the lane.",
    "insight_type": "laning_tips",
    "role": "adc",
    "champion": "Caitlyn",
    "source": "discord",
    "source_weight": 1.6,
    "score": 0.9,
    "confidence": 0.8,
}]

STRUCTURED = StrategicContext(
    fingerprints=({
        "champion": "Caitlyn", "persistent_advantages": ("persistent lane control",),
        "conditional_advantages": (), "dependencies": ("access",),
        "access_tools": ("trap zone",), "continuity_requirements": (),
        "failure_modes": (), "confidence": 0.9,
        "evidence_refs": ({"source_type": "fixture", "source_id": "caitlyn"},),
    },),
    relations=({
        "subject_key": "Caitlyn", "relation_type": "creates", "object_key": "persistent control",
        "condition": "traps cover the wave approach", "effect": "the mage pays resources to contest space",
        "concepts": ("persistent_control", "resource_budget"), "confidence": 0.9,
        "evidence_refs": ({"source_type": "fixture", "source_id": "caitlyn-mage"},),
    },),
)


class StrategicPromptIntegrationTests(unittest.TestCase):
    def test_formatter_separates_causal_structure_and_provenance(self):
        rendered = format_strategic_context(STRUCTURED)
        self.assertIn("Fingerprints:", rendered)
        self.assertIn("Causal relations:", rendered)
        self.assertIn("fixture:caitlyn", rendered)
        self.assertIn("fixture:caitlyn-mage", rendered)

    def test_general_answer_appends_structured_context_without_changing_evidence(self):
        with mock.patch.object(query, "retrieve", return_value=EVIDENCE), mock.patch.object(query, "build_strategic_context", return_value=STRUCTURED), mock.patch.object(query, "_stat_context_block", return_value=""), mock.patch.object(query, "_ability_reference_block", return_value=""), mock.patch.object(query, "llm_chat", return_value="answer") as chat:
            result = query.answer("Caitlyn versus mage", champion="Caitlyn", show_sources=False)
        self.assertEqual(result, "answer")
        user = chat.call_args.kwargs["user"]
        self.assertIn("=== Retrieved coaching evidence ===", user)
        self.assertIn(EVIDENCE[0]["text"], user)
        self.assertIn("=== Derived strategic context (not raw coaching evidence) ===", user)
        self.assertIn("Caitlyn creates persistent control", user)

    def test_baseline_flag_omits_strategic_context_but_keeps_same_evidence(self):
        with mock.patch.object(query, "retrieve", return_value=EVIDENCE), mock.patch.object(query, "build_strategic_context") as strategic, mock.patch.object(query, "_stat_context_block", return_value=""), mock.patch.object(query, "_ability_reference_block", return_value=""), mock.patch.object(query, "llm_chat", return_value="answer") as chat:
            query.answer("Caitlyn versus mage", champion="Caitlyn", show_sources=False, include_strategic_context=False)
        strategic.assert_not_called()
        user = chat.call_args.kwargs["user"]
        self.assertIn(EVIDENCE[0]["text"], user)
        self.assertNotIn("Derived strategic context", user)

    def test_unavailable_strategic_store_falls_back_to_evidence_only(self):
        with mock.patch.object(query, "retrieve", return_value=EVIDENCE), mock.patch.object(query, "build_strategic_context", side_effect=RuntimeError("unavailable")), mock.patch.object(query, "_stat_context_block", return_value=""), mock.patch.object(query, "_ability_reference_block", return_value=""), mock.patch.object(query, "llm_chat", return_value="answer") as chat:
            result = query.answer("Caitlyn versus mage", champion="Caitlyn", show_sources=False)
        self.assertEqual(result, "answer")
        self.assertNotIn("Derived strategic context", chat.call_args.kwargs["user"])

    def test_empty_strategic_context_is_not_rendered(self):
        with mock.patch.object(query, "retrieve", return_value=EVIDENCE), mock.patch.object(query, "build_strategic_context", return_value=StrategicContext()), mock.patch.object(query, "_stat_context_block", return_value=""), mock.patch.object(query, "_ability_reference_block", return_value=""), mock.patch.object(query, "llm_chat", return_value="answer") as chat:
            query.answer("Caitlyn versus mage", champion="Caitlyn", show_sources=False)
        self.assertIn(EVIDENCE[0]["text"], chat.call_args.kwargs["user"])
        self.assertNotIn("Derived strategic context", chat.call_args.kwargs["user"])

    def test_duo_answer_uses_one_derived_block_for_both_entities(self):
        with mock.patch.object(query, "retrieve_duo", return_value=(EVIDENCE, EVIDENCE)), mock.patch.object(query, "build_strategic_context", return_value=STRUCTURED) as strategic, mock.patch.object(query, "_stat_context_block", return_value=""), mock.patch.object(query, "_ability_reference_block", return_value=""), mock.patch.object(query, "llm_chat", return_value="answer") as chat:
            result = query._answer_duo("Caitlyn vs Mage", {"type": "matchup", "a": "Caitlyn", "b": "Mage"}, role=None, game="lol", top_k=4, show_sources=False, include_strategic_context=True)
        self.assertEqual(result, "answer")
        strategic.assert_called_once_with("Caitlyn vs Mage", ("Caitlyn", "Mage"), db_paths=None)
        self.assertIn("Derived strategic context", chat.call_args.kwargs["user"])

    def test_duo_baseline_flag_omits_strategic_context(self):
        with mock.patch.object(query, "retrieve_duo", return_value=(EVIDENCE, EVIDENCE)), mock.patch.object(query, "build_strategic_context") as strategic, mock.patch.object(query, "_stat_context_block", return_value=""), mock.patch.object(query, "_ability_reference_block", return_value=""), mock.patch.object(query, "llm_chat", return_value="answer") as chat:
            query._answer_duo("Caitlyn vs Mage", {"type": "matchup", "a": "Caitlyn", "b": "Mage"}, role=None, game="lol", top_k=4, show_sources=False, include_strategic_context=False)
        strategic.assert_not_called()
        self.assertIn(EVIDENCE[0]["text"], chat.call_args.kwargs["user"])
        self.assertNotIn("Derived strategic context", chat.call_args.kwargs["user"])


if __name__ == "__main__":
    unittest.main()
