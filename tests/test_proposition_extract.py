"""Deterministic Phase 2E tests for span-first grounded proposition extraction."""

from __future__ import annotations

import json
import inspect
import unittest

from pipeline.proposition_extract import (
    ClauseCandidate,
    DIRECTION_SYSTEM,
    EVIDENCE_LOCALIZATION_SYSTEM,
    InventedOntologyContent,
    PROPOSITION_SYSTEM,
    SLOT_SYSTEMS,
    ExtractedProposition,
    OntologyNormalization,
    PropositionPacket,
    ProviderCallError,
    SemanticSlot,
    SourceAlignment,
    SourceSemanticFrame,
    UnsupportedSourceSlot,
    assemble_grounded_proposition,
    clause_evidence_prompt,
    coalesce_selected_evidence,
    enumerate_clause_candidates,
    extract_grounded_propositions,
    extract_span_first_propositions,
    parse_causal_direction,
    parse_candidate_evidence_selection,
    parse_evidence_selection,
    parse_grounded_propositions,
    parse_ontology_normalization,
    parse_semantic_slot,
)
from pipeline.relation_extract import GroundedProposition
from pipeline.source_windows import SourceWindow


TRANSCRIPT = "After Lux misses Q she cannot stop you walking forward."
INSIGHT = "Walk forward after Lux Q misses."


def _window(text: str = TRANSCRIPT, start: int = 20) -> SourceWindow:
    return SourceWindow("1", "video", INSIGHT, text, start, start + len(text), "lexical_window", 0.8)


def _packet(mode: str, *, window: SourceWindow | None = None) -> PropositionPacket:
    return PropositionPacket("1", "video", INSIGHT, mode, _window() if window is None else window)


def _legacy_response(source: str) -> str:
    values = {
        "subject_source": "Lux",
        "predicate_source": "cannot stop",
        "effect_source": "you walking forward",
        "condition_source": "After Lux misses Q",
    }
    grounding = {field: {"source": source} for field in ("subject", "predicate", "effect", "condition")}
    return json.dumps({"propositions": [{**values, "grounding": grounding}]})


def _selection_raw(source: str = "transcript", spans=("After Lux misses Q she cannot stop you walking forward",)) -> str:
    return json.dumps({"source": source, "evidence_spans": list(spans)})


def _candidate_selection_raw(
    source: str = "transcript", candidate_ids=("transcript:c001",),
) -> str:
    return json.dumps({"source": source, "candidate_ids": list(candidate_ids)})


def _live_responses(*, condition="NONE", direction="actor_event_causes_effect",
                    normalization=None) -> list[str]:
    if normalization is None:
        normalization = {"actor_concept": None, "event_relation": None, "effect_concept": None}
    return [
        _candidate_selection_raw(),
        json.dumps({"actor": "Lux"}),
        json.dumps({"event": "cannot stop"}),
        json.dumps({"effect": "you walking forward"}),
        json.dumps({"condition": condition}),
        json.dumps({"causal_direction": direction}),
        json.dumps(normalization),
    ]


def _scripted_chat(responses):
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        index = len(calls) - 1
        if index >= len(responses):
            raise AssertionError(f"unexpected model call #{index + 1}")
        return responses[index]

    return chat, calls


def _scripted_chat_with_provider_failure(responses, fail_at, error=None):
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        index = len(calls) - 1
        if index == fail_at:
            raise error if error is not None else RuntimeError("provider unavailable")
        if index >= len(responses):
            raise AssertionError(f"unexpected model call #{index + 1}")
        return responses[index]

    return chat, calls


def _transcript_span(phrase: str, *, window_start: int = 20) -> SourceAlignment:
    local = TRANSCRIPT.index(phrase)
    return SourceAlignment(
        "transcript", local, local + len(phrase), phrase,
        window_start + local, window_start + local + len(phrase),
    )


def _frame(direction="actor_event_causes_effect", *, with_condition=True,
           normalization=None) -> SourceSemanticFrame:
    if normalization is None:
        normalization = OntologyNormalization(None, None, None)
    evidence = _transcript_span("After Lux misses Q she cannot stop you walking forward")
    actor = SemanticSlot("actor", _transcript_span("Lux"))
    event = SemanticSlot("event", _transcript_span("cannot stop"))
    effect = SemanticSlot("effect", _transcript_span("you walking forward"))
    condition = SemanticSlot("condition", _transcript_span("After Lux misses Q")) if with_condition else None
    return SourceSemanticFrame(
        evidence_spans=(evidence,), actor=actor, event=event, effect=effect,
        condition=condition, causal_direction=direction, normalization=normalization,
    )


class LegacyCompatibilityTests(unittest.TestCase):
    """The retained one-pass parser keeps its strict Phase 2D contract."""

    def test_modes_expose_only_their_allowed_source_text(self) -> None:
        for mode, expected, absent in (("insight", "Walk forward", "cannot stop"), ("transcript", "cannot stop", "Walk forward"), ("combined", "Walk forward", None)):
            with self.subTest(mode=mode):
                prompt = _packet(mode).prompt()
                self.assertIn(expected, prompt)
                if absent:
                    self.assertNotIn(absent, prompt)

    def test_legacy_prompt_preserves_coaching_mechanism_contract(self) -> None:
        self.assertIn("action/resource", PROPOSITION_SYSTEM)
        self.assertIn("should", PROPOSITION_SYSTEM)

    def test_validates_transcript_field_spans(self) -> None:
        result = parse_grounded_propositions(_legacy_response("transcript"), _packet("transcript"))
        self.assertEqual(result[0].proposition.effect_source, "you walking forward")
        self.assertEqual({item.source_kind for item in result[0].alignments}, {"transcript"})
        self.assertTrue(all(item.absolute_start is not None for item in result[0].alignments))
        self.assertTrue(all(item.absolute_start > item.start for item in result[0].alignments))

    def test_rejects_fabricated_or_misaligned_source_phrase(self) -> None:
        payload = json.loads(_legacy_response("transcript"))
        payload["propositions"][0]["effect_source"] = "invented advantage"
        with self.assertRaisesRegex(ValueError, "exact source phrase"):
            parse_grounded_propositions(json.dumps(payload), _packet("transcript"))

    def test_combined_mode_rejects_mixed_source_causal_fields(self) -> None:
        packet = _packet("combined")
        payload = json.loads(_legacy_response("transcript"))
        payload["propositions"][0]["subject_source"] = "Walk forward"
        payload["propositions"][0]["grounding"]["subject"] = {"source": "insight"}
        with self.assertRaisesRegex(ValueError, "one coherent source"):
            parse_grounded_propositions(json.dumps(payload), packet)

    def test_combined_mode_rejects_condition_from_a_different_source(self) -> None:
        packet = _packet("combined")
        payload = json.loads(_legacy_response("transcript"))
        payload["propositions"][0]["condition_source"] = "after Lux Q misses."
        payload["propositions"][0]["grounding"]["condition"] = {"source": "insight"}
        with self.assertRaisesRegex(ValueError, "one coherent source"):
            parse_grounded_propositions(json.dumps(payload), packet)

    def test_allows_safe_zero_and_rejects_malformed_output(self) -> None:
        self.assertEqual(parse_grounded_propositions('{"propositions": []}', _packet("combined")), ())
        with self.assertRaisesRegex(ValueError, "malformed"):
            parse_grounded_propositions("not json", _packet("combined"))

    def test_rejects_null_condition_with_grounding(self) -> None:
        payload = json.loads(_legacy_response("transcript"))
        payload["propositions"][0]["condition_source"] = None
        payload["propositions"][0]["grounding"]["condition"] = {"source": "transcript"}
        with self.assertRaisesRegex(ValueError, "null condition cannot have grounding"):
            parse_grounded_propositions(json.dumps(payload), _packet("transcript"))

    def test_rejects_ambiguous_or_model_supplied_character_offsets(self) -> None:
        packet = PropositionPacket("1", "video", "Flay stops Flay.", "insight")
        raw = json.dumps({"propositions": [{
            "subject_source": "Flay", "predicate_source": "stops", "effect_source": "Flay",
            "condition_source": None,
            "grounding": {"subject": {"source": "insight"}, "predicate": {"source": "insight"}, "effect": {"source": "insight"}, "condition": None},
        }]})
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            parse_grounded_propositions(raw, packet)
        payload = json.loads(_legacy_response("insight"))
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


class EvidenceLocalizationTests(unittest.TestCase):
    def test_localization_prompt_exposes_concrete_source_enums(self) -> None:
        self.assertIn('Allowed source values: ["insight"]', _packet("insight").prompt())
        self.assertIn('Allowed source values: ["transcript"]', _packet("transcript").prompt())
        self.assertIn('Allowed source values: ["insight", "transcript"]', _packet("combined").prompt())
        self.assertNotIn('"source":"insight|transcript"', _packet("combined").prompt())

    def test_evidence_selection_exact_provenance_and_absolute_offsets(self) -> None:
        packet = _packet("transcript")
        phrase = "After Lux misses Q she cannot stop you walking forward"
        spans, parsed = parse_evidence_selection(_selection_raw(), packet)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].source_kind, "transcript")
        self.assertEqual(spans[0].source_text, phrase)
        self.assertEqual(spans[0].start, TRANSCRIPT.index(phrase))
        self.assertEqual(spans[0].end, TRANSCRIPT.index(phrase) + len(phrase))
        self.assertEqual(spans[0].absolute_start, 20 + TRANSCRIPT.index(phrase))
        self.assertEqual(spans[0].absolute_end, 20 + TRANSCRIPT.index(phrase) + len(phrase))
        self.assertEqual(parsed, {"source": "transcript", "evidence_spans": [phrase]})

        two = _selection_raw(spans=("After Lux misses Q", "she cannot stop you walking forward"))
        spans, _ = parse_evidence_selection(two, packet)
        expected = [("After Lux misses Q", TRANSCRIPT.index("After Lux misses Q")), ("she cannot stop you walking forward", TRANSCRIPT.index("she cannot stop you walking forward"))]
        self.assertEqual([(item.source_text, item.start) for item in spans], expected)
        self.assertTrue(all(item.absolute_start == 20 + item.start for item in spans))

    def test_evidence_selection_rejects_nested_nonminimal_spans(self) -> None:
        nested = _selection_raw(spans=("After Lux misses Q she cannot stop you walking forward", "After Lux misses Q"))
        with self.assertRaisesRegex(ValueError, "nested non-minimal"):
            parse_evidence_selection(nested, _packet("transcript"))

    def test_evidence_selection_rejects_duplicate_spans(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_evidence_selection(_selection_raw(spans=("After Lux misses Q", "After Lux misses Q")), _packet("transcript"))

    def test_evidence_selection_rejects_ambiguous_source_text(self) -> None:
        packet = PropositionPacket("1", "video", "Flay stops Flay.", "insight")
        raw = json.dumps({"source": "insight", "evidence_spans": ["Flay"]})
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            parse_evidence_selection(raw, packet)

    def test_evidence_selection_rejects_invalid_source_or_span_shapes(self) -> None:
        packet = _packet("transcript")
        invalid = (
            json.dumps({"source": "insight", "evidence_spans": ["Walk forward"]}),
            json.dumps({"source": "transcript", "evidence_spans": []}),
            json.dumps({"source": None, "evidence_spans": ["text"]}),
            json.dumps({"source": "transcript", "evidence_spans": ["a", "b", "c"]}),
            json.dumps({"source": "transcript", "evidence_spans": [1]}),
            json.dumps({"source": "transcript", "evidence_spans": ["   "]}),
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_evidence_selection(raw, packet)

    def test_evidence_selection_safe_refusal_when_required_evidence_absent(self) -> None:
        raw = json.dumps({"source": None, "evidence_spans": []})
        spans, parsed = parse_evidence_selection(raw, _packet("transcript"))
        self.assertEqual(spans, ())
        self.assertEqual(parsed, {"source": None, "evidence_spans": []})

    def test_evidence_selection_rejects_extra_fields(self) -> None:
        raw = json.dumps({"source": "transcript", "evidence_spans": ["After Lux misses Q"], "extra": 1})
        with self.assertRaisesRegex(ValueError, "requires source and evidence_spans"):
            parse_evidence_selection(raw, _packet("transcript"))


class ClauseCandidateLocalizationTests(unittest.TestCase):
    def test_catalog_is_deterministic_ordered_and_stably_identified(self) -> None:
        text = "Pressure first. When Q misses walk up because they cannot answer."
        packet = _packet("transcript", window=_window(text, start=900))
        first = enumerate_clause_candidates(packet)
        second = enumerate_clause_candidates(packet)
        self.assertEqual(first, second)
        self.assertEqual(
            [item.candidate_id for item in first],
            [f"transcript:c{index:03d}" for index in range(1, len(first) + 1)],
        )
        self.assertEqual(
            [item.alignment.start for item in first],
            sorted(item.alignment.start for item in first),
        )

    def test_punctuation_poor_text_uses_bounded_overlapping_windows(self) -> None:
        text = " ".join(f"word{index}" for index in range(80))
        packet = _packet("transcript", window=_window(text))
        catalog = enumerate_clause_candidates(packet)
        self.assertGreater(len(catalog), 1)
        self.assertLessEqual(len(catalog), 20)
        self.assertTrue(all(len(item.alignment.source_text.split()) <= 32 for item in catalog))
        self.assertTrue(any(
            later.alignment.start < earlier.alignment.end
            for earlier, later in zip(catalog, catalog[1:])
        ))

    def test_catalog_preserves_exact_local_and_absolute_offsets(self) -> None:
        text = "First clause. When Q misses walk forward."
        packet = _packet("transcript", window=_window(text, start=1234))
        for candidate in enumerate_clause_candidates(packet):
            span = candidate.alignment
            self.assertEqual(span.source_text, text[span.start:span.end])
            self.assertEqual(span.absolute_start, 1234 + span.start)
            self.assertEqual(span.absolute_end, 1234 + span.end)

    def test_trigger_words_stay_with_the_following_candidate(self) -> None:
        text = "Hold space when Q misses walk forward because they cannot answer"
        packet = _packet("transcript", window=_window(text))
        values = [item.alignment.source_text.lower() for item in enumerate_clause_candidates(packet)]
        self.assertTrue(any(value.startswith("when ") for value in values))
        self.assertTrue(any(value.startswith("because ") for value in values))
        self.assertFalse(any(value.endswith(" when") or value.endswith(" because") for value in values))

    def test_combined_catalog_keeps_sources_and_id_namespaces_separate(self) -> None:
        packet = _packet("combined")
        catalog = enumerate_clause_candidates(packet)
        insight = [item for item in catalog if item.alignment.source_kind == "insight"]
        transcript = [item for item in catalog if item.alignment.source_kind == "transcript"]
        self.assertTrue(insight)
        self.assertTrue(transcript)
        self.assertTrue(all(item.candidate_id.startswith("insight:c") for item in insight))
        self.assertTrue(all(item.candidate_id.startswith("transcript:c") for item in transcript))
        self.assertEqual(insight[0].candidate_id, "insight:c001")
        self.assertEqual(transcript[0].candidate_id, "transcript:c001")

    def test_prompt_requires_ids_and_never_requests_quoted_spans_or_offsets(self) -> None:
        packet = _packet("transcript")
        catalog = enumerate_clause_candidates(packet)
        prompt = clause_evidence_prompt(packet, catalog)
        self.assertIn("transcript:c001", prompt)
        self.assertIn("candidate_ids", prompt)
        self.assertNotIn("evidence_spans", prompt)
        self.assertIn("Never quote", EVIDENCE_LOCALIZATION_SYSTEM)
        self.assertIn("character\noffsets", EVIDENCE_LOCALIZATION_SYSTEM)

    def test_id_selection_derives_evidence_and_supports_safe_null(self) -> None:
        packet = _packet("transcript")
        catalog = enumerate_clause_candidates(packet)
        spans, parsed = parse_candidate_evidence_selection(
            _candidate_selection_raw(), catalog, packet,
        )
        self.assertEqual(spans, (catalog[0].alignment,))
        self.assertEqual(parsed, {"source": "transcript", "candidate_ids": ["transcript:c001"]})
        empty, parsed = parse_candidate_evidence_selection(
            json.dumps({"source": None, "candidate_ids": []}), catalog, packet,
        )
        self.assertEqual(empty, ())
        self.assertEqual(parsed, {"source": None, "candidate_ids": []})

    def test_id_selection_rejects_malformed_unknown_duplicate_and_wrong_counts(self) -> None:
        text = "First useful clause. Second useful clause. Third useful clause."
        packet = _packet("transcript", window=_window(text))
        catalog = enumerate_clause_candidates(packet)
        invalid = (
            "not json",
            json.dumps({"source": "transcript", "candidate_ids": []}),
            json.dumps({"source": None, "candidate_ids": ["transcript:c001"]}),
            json.dumps({"source": "transcript", "candidate_ids": ["transcript:c999"]}),
            json.dumps({"source": "transcript", "candidate_ids": ["transcript:c001", "transcript:c001"]}),
            json.dumps({"source": "transcript", "candidate_ids": ["transcript:c001", "transcript:c002", "transcript:c003"]}),
            json.dumps({"source": "transcript", "candidate_ids": [1]}),
            json.dumps({"source": "transcript", "candidate_ids": ["transcript:c001"], "extra": True}),
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_candidate_evidence_selection(raw, catalog, packet)

    def test_id_selection_rejects_mixed_or_misdeclared_sources(self) -> None:
        packet = _packet("combined")
        catalog = enumerate_clause_candidates(packet)
        insight_id = next(item.candidate_id for item in catalog if item.alignment.source_kind == "insight")
        transcript_id = next(item.candidate_id for item in catalog if item.alignment.source_kind == "transcript")
        for raw in (
            json.dumps({"source": "transcript", "candidate_ids": [insight_id]}),
            json.dumps({"source": "transcript", "candidate_ids": [insight_id, transcript_id]}),
        ):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "sources"):
                    parse_candidate_evidence_selection(raw, catalog, packet)

    def test_coalesces_overlap_and_whitespace_touch_but_never_semantic_gaps(self) -> None:
        source = "abcd efgh GAP ijkl"
        packet = PropositionPacket("1", "video", source, "insight")
        overlap = coalesce_selected_evidence((
            SourceAlignment("insight", 0, 9, source[0:9]),
            SourceAlignment("insight", 5, 13, source[5:13]),
        ), packet)
        self.assertEqual([(item.start, item.end, item.source_text) for item in overlap], [(0, 13, source[0:13])])
        touching = coalesce_selected_evidence((
            SourceAlignment("insight", 0, 4, source[0:4]),
            SourceAlignment("insight", 5, 9, source[5:9]),
        ), packet)
        self.assertEqual([(item.start, item.end, item.source_text) for item in touching], [(0, 9, source[0:9])])
        gapped = coalesce_selected_evidence((
            SourceAlignment("insight", 0, 4, source[0:4]),
            SourceAlignment("insight", 14, 18, source[14:18]),
        ), packet)
        self.assertEqual(len(gapped), 2)

    def test_catalog_has_no_fixture_or_ontology_specific_rules(self) -> None:
        implementation = inspect.getsource(enumerate_clause_candidates).lower()
        for forbidden in ("sweeper", "gwen", "caitlyn", "hook", "continuity", "wave_obligation"):
            self.assertNotIn(forbidden, implementation)


class SemanticSlotTests(unittest.TestCase):
    def test_slot_exact_grounding_within_evidence(self) -> None:
        packet = _packet("transcript")
        spans, _ = parse_evidence_selection(_selection_raw(), packet)
        slot, parsed = parse_semantic_slot(json.dumps({"actor": "Lux"}), "actor", spans, packet)
        self.assertIsNotNone(slot)
        assert slot is not None
        self.assertEqual(slot.role, "actor")
        self.assertEqual(slot.text, "Lux")
        self.assertEqual(slot.alignment.start, TRANSCRIPT.index("Lux"))
        self.assertEqual(slot.alignment.absolute_start, 20 + TRANSCRIPT.index("Lux"))
        self.assertEqual(parsed, {"actor": "Lux"})

    def test_slot_rejects_phrase_outside_selected_evidence(self) -> None:
        packet = _packet("transcript")
        spans, _ = parse_evidence_selection(_selection_raw(spans=("After Lux misses Q",)), packet)
        with self.assertRaisesRegex(UnsupportedSourceSlot, "outside selected evidence"):
            parse_semantic_slot(json.dumps({"effect": "you walking forward"}), "effect", spans, packet)

    def test_slot_rejects_ambiguous_source_phrase(self) -> None:
        packet = PropositionPacket("1", "video", "Flay stops Flay.", "insight")
        spans, _ = parse_evidence_selection(json.dumps({"source": "insight", "evidence_spans": ["Flay stops Flay."]}), packet)
        with self.assertRaisesRegex(UnsupportedSourceSlot, "unambiguous"):
            parse_semantic_slot(json.dumps({"actor": "Flay"}), "actor", spans, packet)

    def test_slot_repeated_phrase_outside_evidence_resolves_selected_occurrence(self) -> None:
        transcript = "Lux is strong. After Lux misses Q she cannot stop you walking forward."
        window = _window(transcript, start=100)
        packet = _packet("transcript", window=window)
        span_text = "After Lux misses Q she cannot stop you walking forward"
        spans, _ = parse_evidence_selection(_selection_raw(spans=(span_text,)), packet)
        slot, _ = parse_semantic_slot(json.dumps({"actor": "Lux"}), "actor", spans, packet)
        self.assertIsNotNone(slot)
        assert slot is not None
        outside_start = transcript.index("Lux")
        selected_start = transcript.index(span_text) + span_text.index("Lux")
        self.assertNotEqual(selected_start, outside_start)
        self.assertEqual(slot.text, "Lux")
        self.assertEqual(slot.alignment.start, selected_start)
        self.assertEqual(slot.alignment.end, selected_start + len("Lux"))
        self.assertEqual(slot.alignment.absolute_start, 100 + selected_start)
        self.assertEqual(slot.alignment.absolute_end, 100 + selected_start + len("Lux"))

    def test_slot_phrase_repeated_twice_within_evidence_fails_closed(self) -> None:
        transcript = "After Lux misses Q she cannot stop you cannot stop walking forward."
        window = _window(transcript, start=100)
        packet = _packet("transcript", window=window)
        spans, _ = parse_evidence_selection(_selection_raw(spans=(transcript,)), packet)
        with self.assertRaisesRegex(UnsupportedSourceSlot, "unambiguous"):
            parse_semantic_slot(json.dumps({"event": "cannot stop"}), "event", spans, packet)

    def test_slot_phrase_in_two_selected_spans_fails_closed(self) -> None:
        transcript = "After Lux misses Q she cannot stop you walking forward. You cannot stop Lux."
        window = _window(transcript, start=100)
        packet = _packet("transcript", window=window)
        spans, _ = parse_evidence_selection(
            _selection_raw(spans=("After Lux misses Q she cannot stop", "You cannot stop Lux.")),
            packet,
        )
        with self.assertRaisesRegex(UnsupportedSourceSlot, "unambiguous"):
            parse_semantic_slot(json.dumps({"event": "cannot stop"}), "event", spans, packet)

    def test_slot_overlapping_selected_spans_do_not_double_count_occurrence(self) -> None:
        packet = _packet("transcript")
        spans, _ = parse_evidence_selection(
            _selection_raw(spans=("After Lux misses Q she", "she cannot stop you walking forward")),
            packet,
        )
        slot, _ = parse_semantic_slot(json.dumps({"actor": "she"}), "actor", spans, packet)
        self.assertIsNotNone(slot)
        assert slot is not None
        self.assertEqual(slot.text, "she")
        self.assertEqual(slot.alignment.start, TRANSCRIPT.index("she"))
        self.assertEqual(slot.alignment.absolute_start, 20 + TRANSCRIPT.index("she"))

    def test_slot_resolved_occurrence_uses_transcript_absolute_offsets(self) -> None:
        transcript = "Lux is strong. After Lux misses Q she cannot stop you walking forward."
        window = _window(transcript, start=1234)
        packet = _packet("transcript", window=window)
        span_text = "After Lux misses Q she cannot stop you walking forward"
        spans, _ = parse_evidence_selection(_selection_raw(spans=(span_text,)), packet)
        slot, _ = parse_semantic_slot(json.dumps({"effect": "you walking forward"}), "effect", spans, packet)
        self.assertIsNotNone(slot)
        assert slot is not None
        local = transcript.index("you walking forward")
        self.assertEqual(slot.alignment.start, local)
        self.assertEqual(slot.alignment.end, local + len("you walking forward"))
        self.assertEqual(slot.alignment.absolute_start, 1234 + local)
        self.assertEqual(slot.alignment.absolute_end, 1234 + local + len("you walking forward"))

    def test_condition_null_and_none_are_safe_absences(self) -> None:
        packet = _packet("transcript")
        spans, _ = parse_evidence_selection(_selection_raw(), packet)
        for raw in ('{"condition": null}', '{"condition": "NONE"}'):
            with self.subTest(raw=raw):
                slot, parsed = parse_semantic_slot(raw, "condition", spans, packet)
                self.assertIsNone(slot)
                self.assertEqual(parsed, json.loads(raw))

    def test_condition_malformed_values_fail_closed(self) -> None:
        packet = _packet("transcript")
        spans, _ = parse_evidence_selection(_selection_raw(), packet)
        cases = (
            ('{"condition": ""}', ValueError),
            ('{"condition": 3}', ValueError),
            ('{"condition": "NONE", "extra": 1}', ValueError),
            ('{"condition": "made up"}', UnsupportedSourceSlot),
        )
        for raw, error in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(error):
                    parse_semantic_slot(raw, "condition", spans, packet)

    def test_non_condition_slot_cannot_be_null(self) -> None:
        packet = _packet("transcript")
        spans, _ = parse_evidence_selection(_selection_raw(), packet)
        with self.assertRaisesRegex(ValueError, "exact source phrase"):
            parse_semantic_slot('{"actor": null}', "actor", spans, packet)

    def test_slot_requires_selected_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires selected evidence"):
            parse_semantic_slot('{"actor": "Lux"}', "actor", (), _packet("transcript"))

    def test_slot_rejects_mixed_source_evidence(self) -> None:
        packet = _packet("combined")
        spans = (
            SourceAlignment("insight", 0, 4, "Walk", None, None),
            SourceAlignment("transcript", 0, 5, "After", 20, 25),
        )
        with self.assertRaisesRegex(ValueError, "one coherent source"):
            parse_semantic_slot('{"actor": "Lux"}', "actor", spans, packet)

    def test_slot_strips_surrounding_whitespace_before_grounding(self) -> None:
        packet = _packet("transcript")
        spans, _ = parse_evidence_selection(_selection_raw(), packet)
        slot, _ = parse_semantic_slot('{"actor": "  Lux  "}', "actor", spans, packet)
        assert slot is not None
        self.assertEqual(slot.text, "Lux")
        self.assertEqual(slot.alignment.start, TRANSCRIPT.index("Lux"))

    def test_unknown_slot_role_fails_closed(self) -> None:
        packet = _packet("transcript")
        spans, _ = parse_evidence_selection(_selection_raw(), packet)
        with self.assertRaisesRegex(ValueError, "unknown semantic slot"):
            parse_semantic_slot('{"banana": "Lux"}', "banana", spans, packet)


class DirectionNormalizationTests(unittest.TestCase):
    def test_direction_accepts_exactly_the_closed_labels(self) -> None:
        labels = ("actor_event_causes_effect", "effect_causes_actor_event", "association_only", "temporal_sequence_only", "insufficient_causal_claim")
        for label in labels:
            with self.subTest(label=label):
                parsed, body = parse_causal_direction(json.dumps({"causal_direction": label}))
                self.assertEqual(parsed, label)
                self.assertEqual(body, {"causal_direction": label})

    def test_direction_rejects_unknown_label_and_malformed_shapes(self) -> None:
        invalid = (
            '{"causal_direction": "maybe"}',
            '{"causal_direction": "actor_event_causes_effect", "extra": 1}',
            '{"causal_direction": 3}',
            '{"actor_event_causes_effect": true}',
            "[]",
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_causal_direction(raw)

    def test_normalization_accepts_closed_ids_and_safe_abstention(self) -> None:
        norm, body = parse_ontology_normalization('{"actor_concept": null, "event_relation": null, "effect_concept": null}')
        self.assertEqual(norm, OntologyNormalization(None, None, None))
        self.assertEqual(body, {"actor_concept": None, "event_relation": None, "effect_concept": None})
        norm, _ = parse_ontology_normalization('{"actor_concept": "continuity", "event_relation": "denies", "effect_concept": "access"}')
        self.assertEqual(norm, OntologyNormalization("continuity", "denies", "access"))

    def test_normalization_rejects_invented_ids_with_explicit_taxonomy(self) -> None:
        cases = (
            ('{"actor_concept": "made_up", "event_relation": null, "effect_concept": null}', {"made_up": 1}, "actor concept"),
            ('{"actor_concept": null, "event_relation": "made_up", "effect_concept": null}', {"made_up": 1}, "event relation"),
            ('{"actor_concept": null, "event_relation": null, "effect_concept": "made_up"}', {"made_up": 1}, "effect concept"),
            ('{"actor_concept": "made_up", "event_relation": "made_up", "effect_concept": null}', {"made_up": 2}, "actor concept"),
            ('{"actor_concept": "bogus", "event_relation": "other", "effect_concept": null}', {"bogus": 1, "other": 1}, "actor concept"),
        )
        for raw, invented, message in cases:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(InventedOntologyContent, message) as caught:
                    parse_ontology_normalization(raw)
                self.assertEqual(caught.exception.invented, invented)
                self.assertEqual(caught.exception.count, sum(invented.values()))

    def test_normalization_malformed_and_partial_output_fails_closed_as_value_error(self) -> None:
        invalid = (
            "not json",
            "null",
            "[]",
            '{"actor_concept": null, "event_relation": null}',
            '{"actor_concept": null, "event_relation": null, "effect_concept": null, "extra": 1}',
            '{"actor_concept": 3, "event_relation": null, "effect_concept": null}',
            '{"actor_concept": null, "event_relation": [], "effect_concept": null}',
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_ontology_normalization(raw)


class AssemblyTests(unittest.TestCase):
    def test_forward_causality_assembles_deterministically(self) -> None:
        frame = _frame()
        first = assemble_grounded_proposition(frame, "1")
        second = assemble_grounded_proposition(frame, "1")
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first, second)
        proposition = first.proposition
        self.assertEqual(proposition.subject_source, "Lux")
        self.assertEqual(proposition.predicate_source, "cannot stop")
        self.assertEqual(proposition.effect_source, "you walking forward")
        self.assertEqual(proposition.condition_source, "After Lux misses Q")
        self.assertEqual(proposition.evidence_ids, ("1",))
        self.assertEqual([item.field for item in first.alignments], ["subject", "predicate", "effect", "condition"])
        self.assertEqual([item.source_text for item in first.alignments], ["Lux", "cannot stop", "you walking forward", "After Lux misses Q"])
        self.assertEqual({item.source_kind for item in first.alignments}, {"transcript"})
        self.assertTrue(all(item.absolute_start is not None for item in first.alignments))
        self.assertEqual(first.alignments[1].absolute_start, 20 + TRANSCRIPT.index("cannot stop"))

    def test_assembly_abstains_on_reversed_direction(self) -> None:
        frame = _frame(direction="effect_causes_actor_event")
        self.assertIsNone(assemble_grounded_proposition(frame, "1"))
        self.assertEqual(frame.actor.text, "Lux")
        self.assertEqual(frame.effect.text, "you walking forward")

    def test_assembly_abstains_on_non_causal_directions(self) -> None:
        for direction in ("association_only", "temporal_sequence_only", "insufficient_causal_claim"):
            with self.subTest(direction=direction):
                self.assertIsNone(assemble_grounded_proposition(_frame(direction=direction), "1"))

    def test_assembly_condition_absent_produces_null_condition(self) -> None:
        extracted = assemble_grounded_proposition(_frame(with_condition=False), "1")
        assert extracted is not None
        self.assertIsNone(extracted.proposition.condition_source)
        self.assertEqual([item.field for item in extracted.alignments], ["subject", "predicate", "effect"])

    def test_assembly_accepts_abstained_normalization(self) -> None:
        extracted = assemble_grounded_proposition(_frame(normalization=OntologyNormalization(None, None, None)), "1")
        self.assertIsNotNone(extracted)

    def test_assembly_rejects_cross_source_slots(self) -> None:
        frame = _frame()
        actor = SemanticSlot("actor", SourceAlignment("insight", INSIGHT.index("Lux"), INSIGHT.index("Lux") + 3, "Lux"))
        frame = SourceSemanticFrame(
            evidence_spans=frame.evidence_spans, actor=actor, event=frame.event, effect=frame.effect,
            condition=frame.condition, causal_direction="actor_event_causes_effect",
            normalization=frame.normalization,
        )
        with self.assertRaisesRegex(ValueError, "one coherent source"):
            assemble_grounded_proposition(frame, "1")

    def test_assembly_rejects_slot_outside_evidence_spans(self) -> None:
        evidence = _transcript_span("After Lux misses Q")
        frame = SourceSemanticFrame(
            evidence_spans=(evidence,),
            actor=SemanticSlot("actor", _transcript_span("Lux")),
            event=SemanticSlot("event", _transcript_span("cannot stop")),
            effect=SemanticSlot("effect", _transcript_span("you walking forward")),
            condition=None,
            causal_direction="actor_event_causes_effect",
            normalization=OntologyNormalization(None, None, None),
        )
        with self.assertRaisesRegex(ValueError, "outside selected evidence"):
            assemble_grounded_proposition(frame, "1")

    def test_assembly_rejects_empty_or_mixed_evidence_spans(self) -> None:
        frame = _frame()
        empty = SourceSemanticFrame(
            evidence_spans=(), actor=frame.actor, event=frame.event, effect=frame.effect,
            condition=frame.condition, causal_direction="actor_event_causes_effect",
            normalization=frame.normalization,
        )
        with self.assertRaisesRegex(ValueError, "evidence spans"):
            assemble_grounded_proposition(empty, "1")
        mixed = SourceSemanticFrame(
            evidence_spans=(SourceAlignment("insight", 0, 4, "Walk", None, None), _transcript_span("After Lux misses Q")),
            actor=frame.actor, event=frame.event, effect=frame.effect,
            condition=frame.condition, causal_direction="actor_event_causes_effect",
            normalization=frame.normalization,
        )
        with self.assertRaisesRegex(ValueError, "evidence spans"):
            assemble_grounded_proposition(mixed, "1")


class SpanFirstPipelineTests(unittest.TestCase):
    def test_pipeline_assembles_only_supported_forward_causality(self) -> None:
        chat, calls = _scripted_chat(_live_responses())
        result = extract_span_first_propositions(_packet("transcript"), chat, thinking="disabled")
        self.assertEqual(len(result.propositions), 1)
        self.assertEqual(len(result.frames), 1)
        frame = result.frames[0]
        self.assertEqual(frame.causal_direction, "actor_event_causes_effect")
        self.assertEqual(frame.normalization, OntologyNormalization(None, None, None))
        self.assertIsNone(frame.normalization_failure)
        self.assertIsNone(result.failure_stage)
        self.assertEqual(result.unsupported_slot_count, 0)
        self.assertEqual(len(calls), 7)

    def test_pipeline_refuses_when_required_evidence_absent(self) -> None:
        chat, calls = _scripted_chat([json.dumps({"source": None, "candidate_ids": []})])
        result = extract_span_first_propositions(_packet("transcript"), chat)
        self.assertEqual(result.propositions, ())
        self.assertEqual(result.frames, ())
        self.assertEqual(result.evidence_spans, ())
        self.assertIsNone(result.failure_stage)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.artifacts[0].stage, "evidence_localization")
        self.assertEqual(result.artifacts[0].parsed_output, {"source": None, "candidate_ids": []})

    def test_pipeline_condition_null_or_none_assembles_without_condition(self) -> None:
        for condition in (None, "NONE"):
            with self.subTest(condition=condition):
                chat, _ = _scripted_chat(_live_responses(condition=condition))
                result = extract_span_first_propositions(_packet("transcript"), chat)
                self.assertEqual(len(result.propositions), 1)
                proposition = result.propositions[0]
                self.assertIsNone(proposition.proposition.condition_source)
                self.assertEqual([item.field for item in proposition.alignments], ["subject", "predicate", "effect"])
                self.assertIsNone(result.frames[0].condition)

    def test_pipeline_non_forward_direction_keeps_frame_without_proposition(self) -> None:
        chat, _ = _scripted_chat(_live_responses(direction="effect_causes_actor_event"))
        result = extract_span_first_propositions(_packet("transcript"), chat)
        self.assertEqual(result.propositions, ())
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].causal_direction, "effect_causes_actor_event")
        self.assertEqual(result.causal_direction, "effect_causes_actor_event")
        self.assertIsNone(result.failure_stage)

    def test_pipeline_malformed_localization_fails_closed(self) -> None:
        chat, calls = _scripted_chat(["not json"])
        result = extract_span_first_propositions(_packet("transcript"), chat)
        self.assertEqual(result.propositions, ())
        self.assertEqual(result.failure_stage, "evidence_localization")
        self.assertEqual(result.artifacts[0].failure, "ValueError")
        self.assertEqual(len(calls), 1)

    def test_pipeline_retains_candidate_catalog_on_every_localization_outcome(self) -> None:
        expected = enumerate_clause_candidates(_packet("transcript"))
        abstain, _ = _scripted_chat([json.dumps({"source": None, "candidate_ids": []})])
        malformed, _ = _scripted_chat(["not json"])
        provider, _ = _scripted_chat_with_provider_failure([], fail_at=0)
        for name, chat in (("abstain", abstain), ("malformed", malformed), ("provider", provider)):
            with self.subTest(name=name):
                result = extract_span_first_propositions(_packet("transcript"), chat)
                self.assertEqual(result.candidate_catalog, expected)
                self.assertEqual(
                    result.to_artifact_dict()["candidate_catalog"],
                    [{"candidate_id": item.candidate_id, "alignment": {
                        "source_kind": item.alignment.source_kind,
                        "start": item.alignment.start,
                        "end": item.alignment.end,
                        "source_text": item.alignment.source_text,
                        "absolute_start": item.alignment.absolute_start,
                        "absolute_end": item.alignment.absolute_end,
                    }} for item in expected],
                )

    def test_pipeline_slot_failure_retains_spans_and_prior_slots(self) -> None:
        chat, calls = _scripted_chat([
            _candidate_selection_raw(),
            json.dumps({"actor": "Lux"}),
            json.dumps({"event": "not in evidence"}),
        ])
        result = extract_span_first_propositions(_packet("transcript"), chat)
        self.assertEqual(result.failure_stage, "event_extraction")
        self.assertEqual(result.unsupported_slot_count, 1)
        self.assertEqual(result.propositions, ())
        self.assertEqual(result.frames, ())
        self.assertEqual(len(result.evidence_spans), 1)
        self.assertEqual(result.evidence_spans[0].source_text, TRANSCRIPT)
        self.assertEqual(set(result.slots), {"actor"})
        assert result.slots["actor"] is not None
        self.assertEqual(result.slots["actor"].text, "Lux")
        self.assertIsNone(result.causal_direction)
        self.assertEqual(len(calls), 3)
        payload = result.to_artifact_dict()
        self.assertEqual(payload["failure_stage"], "event_extraction")
        self.assertEqual(payload["unsupported_slot_count"], 1)
        self.assertEqual(len(payload["selected_evidence_spans"]), 1)
        self.assertEqual([entry["role"] for entry in payload["recovered_slots"]], ["actor"])
        self.assertEqual(payload["semantic_frames"], [])
        self.assertEqual(payload["assembled_propositions"], [])

    def test_pipeline_malformed_slot_output_is_not_unsupported_source_slot(self) -> None:
        chat, calls = _scripted_chat([
            _candidate_selection_raw(),
            json.dumps({"actor": "Lux"}),
            "not json",
        ])
        result = extract_span_first_propositions(_packet("transcript"), chat)
        self.assertEqual(result.failure_stage, "event_extraction")
        self.assertEqual(result.unsupported_slot_count, 0)
        self.assertEqual(result.invented_ontology_count, 0)
        self.assertEqual(result.invented_ontology_taxonomy, {})
        self.assertEqual(result.propositions, ())
        self.assertEqual(result.artifacts[-1].failure, "ValueError")
        self.assertEqual(len(calls), 3)

    def test_pipeline_slot_outside_span_is_unsupported_source_slot(self) -> None:
        transcript = "After Lux misses Q she cannot stop. you walking forward."
        packet = _packet("transcript", window=_window(transcript))
        chat, calls = _scripted_chat([
            _candidate_selection_raw(),
            json.dumps({"actor": "Lux"}),
            json.dumps({"event": "cannot stop"}),
            json.dumps({"effect": "you walking forward"}),
        ])
        result = extract_span_first_propositions(packet, chat)
        self.assertEqual(result.failure_stage, "effect_extraction")
        self.assertEqual(result.unsupported_slot_count, 1)
        self.assertEqual(result.invented_ontology_count, 0)
        self.assertEqual(result.invented_ontology_taxonomy, {})
        self.assertEqual(result.artifacts[-1].failure, "UnsupportedSourceSlot")
        self.assertEqual(len(calls), 4)

    def test_pipeline_invalid_direction_fails_closed_and_retains_slots(self) -> None:
        responses = _live_responses()
        responses[5] = json.dumps({"causal_direction": "maybe"})
        chat, calls = _scripted_chat(responses)
        result = extract_span_first_propositions(_packet("transcript"), chat)
        self.assertEqual(result.failure_stage, "causal_direction")
        self.assertEqual(result.propositions, ())
        self.assertEqual(len(result.evidence_spans), 1)
        self.assertEqual(set(result.slots), {"actor", "event", "effect", "condition"})
        self.assertIsNone(result.causal_direction)
        self.assertEqual(len(calls), 6)

    def test_pipeline_normalization_failure_keeps_recovered_frame(self) -> None:
        responses = _live_responses(condition="After Lux misses Q")
        responses[6] = json.dumps({"actor_concept": "made_up", "event_relation": None, "effect_concept": None})
        chat, calls = _scripted_chat(responses)
        result = extract_span_first_propositions(_packet("transcript"), chat)
        self.assertEqual(result.failure_stage, "ontology_normalization")
        self.assertEqual(result.propositions, ())
        self.assertEqual(len(result.frames), 1)
        frame = result.frames[0]
        self.assertIsNone(frame.normalization)
        self.assertEqual(frame.normalization_failure, "InventedOntologyContent")
        self.assertEqual(frame.actor.text, "Lux")
        self.assertEqual(frame.event.text, "cannot stop")
        self.assertEqual(frame.effect.text, "you walking forward")
        assert frame.condition is not None
        self.assertEqual(frame.condition.text, "After Lux misses Q")
        self.assertEqual(frame.causal_direction, "actor_event_causes_effect")
        self.assertEqual(result.causal_direction, "actor_event_causes_effect")
        self.assertEqual(set(result.slots), {"actor", "event", "effect", "condition"})
        self.assertEqual(result.artifacts[-1].stage, "ontology_normalization")
        self.assertEqual(result.artifacts[-1].failure, "InventedOntologyContent")
        self.assertEqual(result.invented_ontology_count, 1)
        self.assertEqual(result.invented_ontology_taxonomy, {"made_up": 1})
        self.assertEqual(result.unsupported_slot_count, 0)
        payload = result.to_artifact_dict()
        self.assertEqual(payload["invented_ontology_count"], 1)
        self.assertEqual(payload["invented_ontology_taxonomy"], {"made_up": 1})
        self.assertEqual(len(calls), 7)

    def test_pipeline_normalization_malformed_json_is_not_invented_ontology(self) -> None:
        responses = _live_responses(condition="After Lux misses Q")
        responses[6] = "not json"
        chat, calls = _scripted_chat(responses)
        result = extract_span_first_propositions(_packet("transcript"), chat)
        self.assertEqual(result.failure_stage, "ontology_normalization")
        self.assertEqual(result.invented_ontology_count, 0)
        self.assertEqual(result.invented_ontology_taxonomy, {})
        self.assertEqual(result.unsupported_slot_count, 0)
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].normalization_failure, "ValueError")
        self.assertEqual(result.artifacts[-1].stage, "ontology_normalization")
        self.assertEqual(result.artifacts[-1].failure, "ValueError")
        self.assertEqual(result.causal_direction, "actor_event_causes_effect")
        self.assertEqual(len(result.evidence_spans), 1)
        self.assertEqual(set(result.slots), {"actor", "event", "effect", "condition"})
        self.assertEqual(len(calls), 7)
        payload = result.to_artifact_dict()
        self.assertEqual(payload["failure_stage"], "ontology_normalization")
        self.assertEqual(payload["invented_ontology_count"], 0)
        self.assertEqual(payload["invented_ontology_taxonomy"], {})

    def test_pipeline_normalization_wrong_shape_is_not_invented_ontology(self) -> None:
        responses = _live_responses(condition="After Lux misses Q")
        responses[6] = json.dumps({"actor_concept": None, "event_relation": None})
        chat, calls = _scripted_chat(responses)
        result = extract_span_first_propositions(_packet("transcript"), chat)
        self.assertEqual(result.failure_stage, "ontology_normalization")
        self.assertEqual(result.invented_ontology_count, 0)
        self.assertEqual(result.invented_ontology_taxonomy, {})
        self.assertEqual(result.frames[0].normalization_failure, "ValueError")
        self.assertEqual(result.artifacts[-1].failure, "ValueError")
        self.assertEqual(len(calls), 7)

    def test_pipeline_invented_normalization_records_all_invented_fields(self) -> None:
        responses = _live_responses(condition="After Lux misses Q")
        responses[6] = json.dumps({"actor_concept": "made_up", "event_relation": "made_up", "effect_concept": None})
        chat, calls = _scripted_chat(responses)
        result = extract_span_first_propositions(_packet("transcript"), chat)
        self.assertEqual(result.failure_stage, "ontology_normalization")
        self.assertEqual(result.invented_ontology_count, 2)
        self.assertEqual(result.invented_ontology_taxonomy, {"made_up": 2})
        self.assertEqual(result.frames[0].normalization_failure, "InventedOntologyContent")
        self.assertEqual(result.artifacts[-1].failure, "InventedOntologyContent")
        payload = result.to_artifact_dict()
        self.assertEqual(payload["invented_ontology_count"], 2)
        self.assertEqual(payload["invented_ontology_taxonomy"], {"made_up": 2})
        self.assertEqual(len(calls), 7)

    def test_pipeline_normalization_abstention_keeps_frame_and_assembles(self) -> None:
        chat, _ = _scripted_chat(_live_responses())
        result = extract_span_first_propositions(_packet("transcript"), chat)
        self.assertEqual(len(result.propositions), 1)
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].normalization, OntologyNormalization(None, None, None))
        self.assertIsNone(result.failure_stage)

    def test_pipeline_retains_raw_outputs_and_parsed_intermediates(self) -> None:
        responses = _live_responses(condition="After Lux misses Q")
        chat, _ = _scripted_chat(responses)
        result = extract_span_first_propositions(_packet("transcript"), chat)
        stages = ["evidence_localization", "actor_extraction", "event_extraction", "effect_extraction", "condition_extraction", "causal_direction", "ontology_normalization"]
        self.assertEqual([item.stage for item in result.artifacts], stages)
        for item, raw in zip(result.artifacts, responses):
            self.assertEqual(item.raw_output, raw)
            self.assertEqual(item.parsed_output, json.loads(raw))
            self.assertIsNone(item.failure)
        payload = result.to_artifact_dict()
        self.assertIsNone(payload["failure_stage"])
        self.assertEqual(len(payload["candidate_catalog"]), 1)
        self.assertEqual(payload["candidate_catalog"][0]["candidate_id"], "transcript:c001")
        self.assertEqual(payload["unsupported_slot_count"], 0)
        self.assertEqual(payload["causal_direction"], "actor_event_causes_effect")
        self.assertEqual(len(payload["raw_stage_outputs"]), 7)
        self.assertEqual(len(payload["selected_evidence_spans"]), 1)
        self.assertEqual([entry["role"] for entry in payload["recovered_slots"]], ["actor", "event", "effect", "condition"])
        self.assertEqual(len(payload["semantic_frames"]), 1)
        self.assertEqual(len(payload["assembled_propositions"]), 1)
        json.dumps(payload)

    def test_pipeline_multi_digit_transcript_offsets(self) -> None:
        window = _window(start=1234)
        chat, _ = _scripted_chat(_live_responses(condition="After Lux misses Q"))
        result = extract_span_first_propositions(_packet("transcript", window=window), chat)
        proposition = result.propositions[0]
        for field, phrase in (("subject", "Lux"), ("predicate", "cannot stop"), ("effect", "you walking forward"), ("condition", "After Lux misses Q")):
            alignment = next(item for item in proposition.alignments if item.field == field)
            self.assertEqual(alignment.absolute_start, 1234 + TRANSCRIPT.index(phrase))
            self.assertEqual(alignment.absolute_end, 1234 + TRANSCRIPT.index(phrase) + len(phrase))
        self.assertEqual(result.frames[0].evidence_spans[0].absolute_start, 1234)

    def test_pipeline_rejects_unverified_transcript_window(self) -> None:
        unverified = SourceWindow("1", "video", INSIGHT, "bronze", 0, 6, "unverified_external_span", 0.0)
        packet = PropositionPacket("1", "video", INSIGHT, "transcript", unverified)
        with self.assertRaisesRegex(ValueError, "verified source window"):
            extract_span_first_propositions(packet, lambda **kwargs: "unused")

    def test_max_tokens_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            extract_span_first_propositions(_packet("transcript"), lambda **kwargs: "", max_tokens=0)

    def test_wrapper_returns_only_assembled_propositions(self) -> None:
        chat, calls = _scripted_chat(_live_responses())
        result = extract_grounded_propositions(_packet("transcript"), chat, thinking="disabled")
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], ExtractedProposition)
        self.assertEqual(len(calls), 7)

    def test_span_first_calls_have_no_ontology_contract(self) -> None:
        chat, calls = _scripted_chat(_live_responses())
        extract_span_first_propositions(_packet("transcript"), chat, thinking="disabled")
        for index, call in enumerate(calls):
            with self.subTest(stage=index):
                system = call["system"].lower()
                for term in ("continuity", "denies", "creates", "strategic concept"):
                    self.assertNotIn(term, system)
                if index == 0:
                    self.assertIn("interpret ontology concepts", system)
                elif index < 6:
                    self.assertNotIn("ontology", system)
                else:
                    self.assertIn("closed ontology", system)
                self.assertEqual(call["thinking"], "disabled")
                self.assertEqual(call["temperature"], 0.0)

    def test_localization_and_slot_prompts_allow_mechanisms_without_ontology_terms(self) -> None:
        self.assertIn("action, resource, or state", EVIDENCE_LOCALIZATION_SYSTEM)
        self.assertIn("consequence", EVIDENCE_LOCALIZATION_SYSTEM)
        self.assertIn("do X to achieve Y", EVIDENCE_LOCALIZATION_SYSTEM)
        for system in (*SLOT_SYSTEMS.values(), DIRECTION_SYSTEM):
            for term in ("continuity", "denies", "creates", "ontology"):
                self.assertNotIn(term, system)

    def test_direction_prompt_exposes_recovered_slots_and_allowed_labels(self) -> None:
        chat, calls = _scripted_chat(_live_responses(condition="After Lux misses Q"))
        extract_span_first_propositions(_packet("transcript"), chat)
        direction_user = calls[5]["user"]
        self.assertIn("actor: Lux", direction_user)
        self.assertIn("event: cannot stop", direction_user)
        self.assertIn("effect: you walking forward", direction_user)
        self.assertIn("condition: After Lux misses Q", direction_user)
        self.assertIn("actor_event_causes_effect", direction_user)

    def test_slot_prompts_quote_only_selected_evidence(self) -> None:
        chat, calls = _scripted_chat(_live_responses())
        extract_span_first_propositions(_packet("combined"), chat)
        actor_user = calls[1]["user"]
        self.assertIn("SELECTED EVIDENCE", actor_user)
        self.assertIn("After Lux misses Q she cannot stop you walking forward", actor_user)
        self.assertNotIn(INSIGHT, actor_user)

    def test_normalization_prompt_exposes_closed_ontology(self) -> None:
        chat, calls = _scripted_chat(_live_responses())
        extract_span_first_propositions(_packet("transcript"), chat)
        user = calls[6]["user"]
        self.assertIn("Allowed strategic concepts", user)
        self.assertIn("continuity", user)
        self.assertIn("Allowed event relations", user)
        self.assertIn("denies", user)


class ProviderFailureTests(unittest.TestCase):
    """Provider call exceptions fail closed at the current stage and retain state."""

    def test_provider_failure_at_localization_has_no_prior_state(self) -> None:
        chat, calls = _scripted_chat_with_provider_failure([], fail_at=0)
        result = extract_span_first_propositions(_packet("transcript"), chat)
        self.assertEqual(result.failure_stage, "evidence_localization")
        self.assertEqual(result.propositions, ())
        self.assertEqual(result.frames, ())
        self.assertEqual(result.evidence_spans, ())
        self.assertEqual(result.slots, {})
        self.assertIsNone(result.causal_direction)
        self.assertEqual(result.unsupported_slot_count, 0)
        self.assertEqual(result.invented_ontology_count, 0)
        self.assertEqual(result.invented_ontology_taxonomy, {})
        self.assertEqual(len(result.artifacts), 1)
        artifact = result.artifacts[0]
        self.assertEqual(artifact.stage, "evidence_localization")
        self.assertIsNone(artifact.raw_output)
        self.assertIsNone(artifact.parsed_output)
        self.assertEqual(artifact.failure, ProviderCallError.__name__)
        self.assertEqual(len(calls), 1)
        payload = result.to_artifact_dict()
        self.assertEqual(payload["failure_stage"], "evidence_localization")
        self.assertEqual(payload["raw_stage_outputs"][0]["raw_output"], None)
        self.assertEqual(payload["selected_evidence_spans"], [])
        self.assertEqual(payload["recovered_slots"], [])
        json.dumps(payload)

    def test_provider_failure_after_actor_retains_spans_and_actor_slot(self) -> None:
        chat, calls = _scripted_chat_with_provider_failure(
            [_candidate_selection_raw(), json.dumps({"actor": "Lux"})], fail_at=2,
        )
        result = extract_span_first_propositions(_packet("transcript"), chat)
        self.assertEqual(result.failure_stage, "event_extraction")
        self.assertEqual(result.propositions, ())
        self.assertEqual(result.frames, ())
        self.assertEqual(len(result.evidence_spans), 1)
        self.assertEqual(result.evidence_spans[0].source_text, TRANSCRIPT)
        self.assertEqual(set(result.slots), {"actor"})
        assert result.slots["actor"] is not None
        self.assertEqual(result.slots["actor"].text, "Lux")
        self.assertIsNone(result.causal_direction)
        self.assertEqual(result.unsupported_slot_count, 0)
        self.assertEqual(result.invented_ontology_count, 0)
        self.assertEqual(result.invented_ontology_taxonomy, {})
        self.assertEqual([item.stage for item in result.artifacts], ["evidence_localization", "actor_extraction", "event_extraction"])
        self.assertEqual(result.artifacts[0].raw_output, _candidate_selection_raw())
        self.assertEqual(result.artifacts[0].failure, None)
        self.assertEqual(result.artifacts[1].raw_output, json.dumps({"actor": "Lux"}))
        self.assertEqual(result.artifacts[1].failure, None)
        failing = result.artifacts[2]
        self.assertIsNone(failing.raw_output)
        self.assertIsNone(failing.parsed_output)
        self.assertEqual(failing.failure, ProviderCallError.__name__)
        self.assertEqual(len(calls), 3)

    def test_provider_failure_at_direction_retains_all_slots(self) -> None:
        responses = _live_responses(condition="After Lux misses Q")
        chat, calls = _scripted_chat_with_provider_failure(responses, fail_at=5)
        result = extract_span_first_propositions(_packet("transcript"), chat)
        self.assertEqual(result.failure_stage, "causal_direction")
        self.assertEqual(result.propositions, ())
        self.assertEqual(result.frames, ())
        self.assertEqual(len(result.evidence_spans), 1)
        self.assertEqual(set(result.slots), {"actor", "event", "effect", "condition"})
        self.assertIsNone(result.causal_direction)
        self.assertEqual(result.unsupported_slot_count, 0)
        self.assertEqual(result.invented_ontology_count, 0)
        self.assertEqual(result.invented_ontology_taxonomy, {})
        self.assertEqual(len(result.artifacts), 6)
        self.assertEqual(result.artifacts[-1].stage, "causal_direction")
        self.assertIsNone(result.artifacts[-1].raw_output)
        self.assertIsNone(result.artifacts[-1].parsed_output)
        self.assertEqual(result.artifacts[-1].failure, ProviderCallError.__name__)
        self.assertEqual(len(calls), 6)

    def test_provider_failure_at_normalization_keeps_frame_without_invented_count(self) -> None:
        responses = _live_responses(condition="After Lux misses Q")
        chat, calls = _scripted_chat_with_provider_failure(responses, fail_at=6)
        result = extract_span_first_propositions(_packet("transcript"), chat)
        self.assertEqual(result.failure_stage, "ontology_normalization")
        self.assertEqual(result.propositions, ())
        self.assertEqual(len(result.frames), 1)
        frame = result.frames[0]
        self.assertIsNone(frame.normalization)
        self.assertEqual(frame.normalization_failure, ProviderCallError.__name__)
        self.assertEqual(frame.actor.text, "Lux")
        self.assertEqual(frame.event.text, "cannot stop")
        self.assertEqual(frame.effect.text, "you walking forward")
        assert frame.condition is not None
        self.assertEqual(frame.condition.text, "After Lux misses Q")
        self.assertEqual(frame.causal_direction, "actor_event_causes_effect")
        self.assertEqual(result.causal_direction, "actor_event_causes_effect")
        self.assertEqual(set(result.slots), {"actor", "event", "effect", "condition"})
        self.assertEqual(result.unsupported_slot_count, 0)
        self.assertEqual(result.invented_ontology_count, 0)
        self.assertEqual(result.invented_ontology_taxonomy, {})
        self.assertEqual(len(result.artifacts), 7)
        failing = result.artifacts[-1]
        self.assertEqual(failing.stage, "ontology_normalization")
        self.assertIsNone(failing.raw_output)
        self.assertIsNone(failing.parsed_output)
        self.assertEqual(failing.failure, ProviderCallError.__name__)
        self.assertEqual(len(calls), 7)
        payload = result.to_artifact_dict()
        self.assertEqual(payload["failure_stage"], "ontology_normalization")
        self.assertEqual(payload["invented_ontology_count"], 0)
        self.assertEqual(payload["invented_ontology_taxonomy"], {})
        self.assertEqual(payload["unsupported_slot_count"], 0)
        self.assertEqual(payload["causal_direction"], "actor_event_causes_effect")
        self.assertEqual(payload["semantic_frames"][0]["normalization"], None)
        self.assertEqual(payload["semantic_frames"][0]["normalization_failure"], ProviderCallError.__name__)
        json.dumps(payload)

    def test_any_provider_exception_maps_to_provider_call_error(self) -> None:
        responses = _live_responses()
        chat, calls = _scripted_chat_with_provider_failure(
            responses, fail_at=4, error=TimeoutError("provider timed out"),
        )
        result = extract_span_first_propositions(_packet("transcript"), chat)
        self.assertEqual(result.failure_stage, "condition_extraction")
        self.assertEqual(result.artifacts[-1].failure, ProviderCallError.__name__)
        self.assertIsNone(result.artifacts[-1].raw_output)
        self.assertEqual(result.unsupported_slot_count, 0)
        self.assertEqual(len(calls), 5)


if __name__ == "__main__":
    unittest.main()
