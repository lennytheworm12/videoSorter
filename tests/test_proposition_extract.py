"""Deterministic Phase 2D tests for source-mode proposition extraction."""

from __future__ import annotations

import json
import unittest

from pipeline.proposition_extract import PropositionPacket, extract_grounded_propositions, parse_grounded_propositions
from pipeline.source_windows import SourceWindow


def _window(text: str = "After Lux misses Q she cannot stop you walking forward.") -> SourceWindow:
    return SourceWindow("1", "video", "Walk forward after Lux Q misses.", text, 20, 20 + len(text), "lexical_window", .8)


def _packet(mode: str) -> PropositionPacket:
    return PropositionPacket("1", "video", "Walk forward after Lux Q misses.", mode, _window())


def _response(source: str, text: str) -> str:
    values = {"subject_source": "Lux", "predicate_source": "cannot stop", "effect_source": "you walking forward", "condition_source": "After Lux misses Q"}
    grounding = {}
    for field, phrase in (("subject", values["subject_source"]), ("predicate", values["predicate_source"]), ("effect", values["effect_source"]), ("condition", values["condition_source"])):
        grounding[field] = {"source": source}
    return json.dumps({"propositions": [{**values, "grounding": grounding}]})


class PropositionExtractionTests(unittest.TestCase):
    def test_modes_expose_only_their_allowed_source_text(self) -> None:
        for mode, expected, absent in (("insight", "Walk forward", "cannot stop"), ("transcript", "cannot stop", "Walk forward"), ("combined", "Walk forward", None)):
            with self.subTest(mode=mode):
                prompt = _packet(mode).prompt()
                self.assertIn(expected, prompt)
                if absent:
                    self.assertNotIn(absent, prompt)

    def test_prompt_exposes_concrete_grounding_source_enums(self) -> None:
        self.assertIn('Allowed grounding source values: ["insight"]', _packet("insight").prompt())
        self.assertIn('Allowed grounding source values: ["transcript"]', _packet("transcript").prompt())
        self.assertIn('Allowed grounding source values: ["insight", "transcript"]', _packet("combined").prompt())
        self.assertNotIn('"source":"insight|transcript"', _packet("combined").prompt())

    def test_validates_transcript_field_spans(self) -> None:
        text = _window().transcript_window
        result = parse_grounded_propositions(_response("transcript", text), _packet("transcript"))
        self.assertEqual(result[0].proposition.effect_source, "you walking forward")
        self.assertEqual({item.source_kind for item in result[0].alignments}, {"transcript"})
        self.assertTrue(all(item.absolute_start is not None for item in result[0].alignments))
        self.assertTrue(all(item.absolute_start > item.start for item in result[0].alignments))

    def test_rejects_fabricated_or_misaligned_source_phrase(self) -> None:
        text = _window().transcript_window
        payload = json.loads(_response("transcript", text))
        payload["propositions"][0]["effect_source"] = "invented advantage"
        with self.assertRaisesRegex(ValueError, "exact source phrase"):
            parse_grounded_propositions(json.dumps(payload), _packet("transcript"))

    def test_combined_mode_rejects_mixed_source_causal_fields(self) -> None:
        packet = _packet("combined")
        transcript = _window().transcript_window
        insight = packet.insight_text
        payload = json.loads(_response("transcript", transcript))
        phrase = "Walk forward"
        payload["propositions"][0]["subject_source"] = phrase
        payload["propositions"][0]["grounding"]["subject"] = {"source": "insight"}
        with self.assertRaisesRegex(ValueError, "one coherent source"):
            parse_grounded_propositions(json.dumps(payload), packet)

    def test_combined_mode_rejects_condition_from_a_different_source(self) -> None:
        packet = _packet("combined")
        transcript = _window().transcript_window
        insight = packet.insight_text
        payload = json.loads(_response("transcript", transcript))
        phrase = "after Lux Q misses."
        payload["propositions"][0]["condition_source"] = phrase
        payload["propositions"][0]["grounding"]["condition"] = {"source": "insight"}
        with self.assertRaisesRegex(ValueError, "one coherent source"):
            parse_grounded_propositions(json.dumps(payload), packet)

    def test_allows_safe_zero_and_rejects_malformed_output(self) -> None:
        self.assertEqual(parse_grounded_propositions('{"propositions": []}', _packet("combined")), ())
        with self.assertRaisesRegex(ValueError, "malformed"):
            parse_grounded_propositions("not json", _packet("combined"))

    def test_rejects_ambiguous_or_model_supplied_character_offsets(self) -> None:
        packet = PropositionPacket("1", "video", "Flay stops Flay.", "insight")
        raw = json.dumps({"propositions": [{
            "subject_source": "Flay", "predicate_source": "stops", "effect_source": "Flay",
            "condition_source": None,
            "grounding": {"subject": {"source": "insight"}, "predicate": {"source": "insight"}, "effect": {"source": "insight"}, "condition": None},
        }]})
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            parse_grounded_propositions(raw, packet)
        payload = json.loads(_response("insight", _packet("insight").insight_text))
        payload["propositions"][0]["grounding"]["subject"]["start"] = 0
        with self.assertRaisesRegex(ValueError, "invalid source grounding"):
            parse_grounded_propositions(json.dumps(payload), _packet("insight"))

    def test_rejects_quoted_phrases_found_only_inside_larger_tokens(self) -> None:
        for source_text, phrase in (("Flayed the target", "Flay"), ("Qiyana uses W", "Q"), ("Kai’Sa uses W", "Kai")):
            with self.subTest(source_text=source_text):
                packet = PropositionPacket("1", "video", source_text, "insight")
                raw = json.dumps({"propositions": [{
                    "subject_source": phrase, "predicate_source": "uses", "effect_source": "W",
                    "condition_source": None,
                    "grounding": {"subject": {"source": "insight"}, "predicate": {"source": "insight"}, "effect": {"source": "insight"}, "condition": None},
                }]})
                with self.assertRaisesRegex(ValueError, "exact source phrase"):
                    parse_grounded_propositions(raw, packet)

    def test_rejects_transcript_modes_without_verified_window(self) -> None:
        unverified = SourceWindow("1", "video", "insight", "bronze", 0, 6, "unverified_external_span", 0.0)
        packet = PropositionPacket("1", "video", "insight", "transcript", unverified)
        with self.assertRaisesRegex(ValueError, "verified source window"):
            packet.prompt()

    def test_model_call_has_no_ontology_contract(self) -> None:
        text = _window().transcript_window
        calls = []
        result = extract_grounded_propositions(
            _packet("transcript"), lambda **kwargs: calls.append(kwargs) or _response("transcript", text),
            thinking="disabled",
        )
        self.assertEqual(len(result), 1)
        self.assertNotIn("continuity", calls[0]["system"].lower())
        self.assertNotIn("denies", calls[0]["system"].lower())
        self.assertEqual(calls[0]["thinking"], "disabled")


if __name__ == "__main__":
    unittest.main()
