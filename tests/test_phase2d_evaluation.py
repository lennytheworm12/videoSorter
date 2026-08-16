from __future__ import annotations

from dataclasses import asdict, replace
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from pipeline.phase2d_evaluation import (
    DEFAULT_HELD_OUT_FIXTURE,
    _tokenize,
    evaluate_source_modes,
    load_development_cases,
)
from pipeline.proposition_extract import (
    ClauseCandidate,
    ExtractedProposition,
    OntologyNormalization,
    PropositionAlignment,
    PropositionPacket,
    SemanticSlot,
    SourceAlignment,
    SourceSemanticFrame,
    StageAExtraction,
    StageArtifact,
    assemble_grounded_proposition,
)
from pipeline.relation_extract import GroundedProposition
from pipeline.source_windows import SourceWindowResolver


def _stage_extraction(
    source: str,
    *,
    actor: str,
    event: str,
    effect: str,
    condition: str | None = None,
    direction: str = "actor_event_causes_effect",
    evidence_span: str | None = None,
    normalization: OntologyNormalization | None = None,
    normalization_failure: str | None = None,
) -> StageAExtraction:
    """Build a deterministic StageAExtraction whose slots quote the source."""
    def slot(role: str, phrase: str) -> SemanticSlot:
        start = source.index(phrase)
        return SemanticSlot(role, SourceAlignment("insight", start, start + len(phrase), phrase))

    span_text = evidence_span if evidence_span is not None else source.strip()
    span_start = source.index(span_text)
    evidence = SourceAlignment("insight", span_start, span_start + len(span_text), span_text)
    condition_slot = slot("condition", condition) if condition else None
    frame_normalization = (
        None if normalization_failure is not None
        else normalization if normalization is not None else OntologyNormalization(None, None, None)
    )
    frame = SourceSemanticFrame(
        (evidence,),
        slot("actor", actor),
        slot("event", event),
        slot("effect", effect),
        condition_slot,
        direction,  # type: ignore[arg-type]
        frame_normalization,
        normalization_failure,
    )
    artifact = StageArtifact(
        "evidence_localization",
        json.dumps({"source": "insight", "evidence_spans": [span_text]}),
        {"source": "insight", "evidence_spans": [span_text]},
    )
    if direction != "actor_event_causes_effect" or normalization_failure is not None:
        failure_stage = "ontology_normalization" if normalization_failure is not None else None
        return StageAExtraction(
            (), (frame,), (artifact,), failure_stage,
            candidate_catalog=(ClauseCandidate("insight:c001", evidence),),
        )
    proposition = GroundedProposition(actor, event, effect, condition, ("1",))
    slot_fields = (("subject", frame.actor), ("predicate", frame.event), ("effect", frame.effect))
    if frame.condition is not None:
        slot_fields += (("condition", frame.condition),)
    alignments = tuple(
        PropositionAlignment(field, value.alignment.source_kind, value.alignment.start, value.alignment.end, value.text)
        for field, value in slot_fields
    )
    return StageAExtraction(
        (ExtractedProposition(proposition, alignments),), (frame,), (artifact,),
        candidate_catalog=(ClauseCandidate("insight:c001", evidence),),
    )


class SourceModeEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp.close()
        with sqlite3.connect(self.temp.name) as conn:
            conn.executescript("""
                CREATE TABLE videos (video_id TEXT PRIMARY KEY, transcription TEXT);
                CREATE TABLE insights (id TEXT PRIMARY KEY, video_id TEXT, text TEXT);
                INSERT INTO videos VALUES ('v1', 'Coach says Flay prevents Tristana from staying on target after entry.');
                INSERT INTO insights VALUES ('1', 'v1', 'Flay prevents Tristana from staying on target after entry.');
                INSERT INTO insights VALUES ('2', 'v1', 'Generic advice only.');
            """)
        self.resolver = SourceWindowResolver(self.temp.name)
        self.cases = (
            {"id": "positive", "insight_id": "1", "source_video_id": "v1", "eligible": True,
             "expected_propositions": [{"subject_source": "Flay", "predicate_source": "prevents", "effect_source": "staying on target", "condition_source": "after entry", "condition_operator": "after", "semantic_field_token_groups": {"subject": [["Flay"]], "predicate": [["prevent", "prevents"]], "effect": [["staying"]], "condition": [["after"], ["entry"]]}, "expected_normalization": {"actor_concept": None, "event_relation": None, "effect_concept": None}, "normalization_rationale": "No reviewed closed-ontology mapping for this synthetic source frame."}]},
            {"id": "safe-zero", "insight_id": "2", "source_video_id": "v1", "eligible": False, "expected_propositions": []},
        )

    def tearDown(self) -> None:
        import os
        os.unlink(self.temp.name)

    def test_scores_modes_and_keeps_unavailable_separate(self) -> None:
        def extractor(packet: PropositionPacket):
            if packet.evidence_id == "2":
                return ()
            text = packet.sources()[0].text
            def aligned(field, phrase):
                start = text.index(phrase)
                return PropositionAlignment(field, packet.sources()[0].kind, start, start + len(phrase), phrase)
            return (ExtractedProposition(
                GroundedProposition("Flay", "prevents", "staying on target", "after entry", ("1",)),
                tuple(aligned(field, phrase) for field, phrase in (("subject", "Flay"), ("predicate", "prevents"), ("effect", "staying on target"), ("condition", "after entry"))),
            ),)
        result = evaluate_source_modes(self.cases, resolver=self.resolver, extractor=extractor)
        self.assertEqual(result["metrics"]["insight"]["proposition_recall"], 1.0)
        self.assertEqual(result["metrics"]["insight"]["safe_zero_accuracy"], 1.0)
        self.assertEqual(result["metrics"]["transcript"]["unavailable_case_count"], 1)
        self.assertEqual(result["metrics"]["combined"]["unavailable_case_count"], 1)

    def test_reports_extractor_failure_without_counting_it_as_safe_zero(self) -> None:
        result = evaluate_source_modes(
            self.cases, resolver=self.resolver, extractor=lambda packet: (_ for _ in ()).throw(RuntimeError("down")), modes=("insight",),
        )
        self.assertEqual(result["metrics"]["insight"]["failure_case_count"], 2)
        self.assertEqual(result["metrics"]["insight"]["safe_zero_accuracy"], 0.0)

    def test_does_not_count_an_unaligned_mock_as_a_grounded_match(self) -> None:
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver,
            extractor=lambda packet: (ExtractedProposition(GroundedProposition("Flay", "prevents", "staying on target", "after entry", ("1",)), ()),),
            modes=("insight",),
        )
        self.assertEqual(result["metrics"]["insight"]["proposition_recall"], 0.0)
        self.assertEqual(result["metrics"]["insight"]["unsupported_proposition_rate"], 1.0)

    def test_does_not_count_fabricated_alignment_offsets_or_evidence(self) -> None:
        def fake(packet):
            return (ExtractedProposition(
                GroundedProposition("Flay", "prevents", "staying on target", "after entry", ("wrong",)),
                tuple(PropositionAlignment(field, "insight", 999, 1000, phrase) for field, phrase in (("subject", "Flay"), ("predicate", "prevents"), ("effect", "staying on target"), ("condition", "after entry"))),
            ),)
        result = evaluate_source_modes(self.cases[:1], resolver=self.resolver, extractor=fake, modes=("insight",))
        self.assertEqual(result["metrics"]["insight"]["proposition_recall"], 0.0)

    def test_does_not_count_boolean_alignment_offsets(self) -> None:
        def fake(packet):
            text = packet.insight_text
            def aligned(field, phrase):
                start = False if field == "subject" else text.index(phrase)
                return PropositionAlignment(field, "insight", start, start + len(phrase), phrase)
            return (ExtractedProposition(
                GroundedProposition("Flay", "prevents", "staying on target", "after entry", ("1",)),
                tuple(aligned(field, phrase) for field, phrase in (("subject", "Flay"), ("predicate", "prevents"), ("effect", "staying on target"), ("condition", "after entry"))),
            ),)
        result = evaluate_source_modes(self.cases[:1], resolver=self.resolver, extractor=fake, modes=("insight",))
        self.assertEqual(result["metrics"]["insight"]["proposition_recall"], 0.0)

    def test_stage_a_fabricated_offsets_are_ungrounded(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        def slot(role: str, phrase: str) -> SemanticSlot:
            return SemanticSlot(role, SourceAlignment("insight", 999, 999 + len(phrase), phrase))
        span_text = source.strip()
        evidence = SourceAlignment("insight", 0, len(span_text), span_text)
        frame = SourceSemanticFrame(
            (evidence,),
            slot("actor", "Flay"),
            slot("event", "prevents"),
            slot("effect", "staying on target"),
            slot("condition", "after entry"),
            "actor_event_causes_effect",
            OntologyNormalization(None, None, None),
        )
        artifacts = (StageArtifact(
            "evidence_localization", '{"source":"insight","evidence_spans":["' + span_text + '"]}',
            {"source": "insight", "evidence_spans": [span_text]},
        ),)
        extraction = StageAExtraction(
            (), (frame,), artifacts, evidence_spans=(evidence,),
            slots={"actor": frame.actor, "event": frame.event, "effect": frame.effect, "condition": frame.condition},
            causal_direction="actor_event_causes_effect",
        )
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=lambda packet: extraction, modes=("insight",),
        )
        entry = result["cases"][0]["modes"][0]
        scores = entry["slot_scores"]
        for slot_name in ("evidence_span", "actor", "event", "effect", "condition", "causal_direction", "semantic_proposition"):
            self.assertEqual(scores[slot_name], {"hit_count": 0, "expected_count": 1})
        comparison = entry["comparisons"][0]
        self.assertFalse(comparison["actor_hit"])
        self.assertFalse(comparison["semantic_proposition_hit"])
        self.assertEqual(result["metrics"]["insight"]["proposition_recall"], 0.0)

    def test_stage_a_mixed_source_slots_are_ungrounded(self) -> None:
        def extractor(packet: PropositionPacket):
            insight = packet.sources()[0].text
            transcript = packet.sources()[1].text
            window = packet.source_window
            assert window is not None and window.window_start is not None
            def insight_slot(role: str, phrase: str) -> SemanticSlot:
                start = insight.index(phrase)
                return SemanticSlot(role, SourceAlignment("insight", start, start + len(phrase), phrase))
            def transcript_slot(role: str, phrase: str) -> SemanticSlot:
                start = transcript.index(phrase)
                return SemanticSlot(role, SourceAlignment(
                    "transcript", start, start + len(phrase), phrase,
                    window.window_start + start, window.window_start + start + len(phrase),
                ))
            span_text = transcript.strip()
            evidence = SourceAlignment(
                "transcript", 0, len(span_text), span_text,
                window.window_start, window.window_start + len(span_text),
            )
            frame = SourceSemanticFrame(
                (evidence,),
                insight_slot("actor", "Flay"),
                transcript_slot("event", "prevents"),
                transcript_slot("effect", "staying on target"),
                transcript_slot("condition", "after entry"),
                "actor_event_causes_effect",
                OntologyNormalization(None, None, None),
            )
            return StageAExtraction((), (frame,), ())
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=extractor, modes=("combined",),
        )
        scores = result["cases"][0]["modes"][0]["slot_scores"]
        for slot_name in ("evidence_span", "actor", "event", "effect", "condition", "semantic_proposition"):
            self.assertEqual(scores[slot_name], {"hit_count": 0, "expected_count": 1})

    def test_stage_a_transcript_absolute_offset_mismatch_is_ungrounded(self) -> None:
        def extractor(packet: PropositionPacket):
            transcript = packet.sources()[1].text
            window = packet.source_window
            assert window is not None and window.window_start is not None
            def transcript_slot(role: str, phrase: str) -> SemanticSlot:
                start = transcript.index(phrase)
                return SemanticSlot(role, SourceAlignment(
                    "transcript", start, start + len(phrase), phrase,
                    window.window_start + start + 1, window.window_start + start + len(phrase) + 1,
                ))
            span_text = transcript.strip()
            evidence = SourceAlignment(
                "transcript", 0, len(span_text), span_text,
                window.window_start, window.window_start + len(span_text),
            )
            frame = SourceSemanticFrame(
                (evidence,),
                transcript_slot("actor", "Flay"),
                transcript_slot("event", "prevents"),
                transcript_slot("effect", "staying on target"),
                transcript_slot("condition", "after entry"),
                "actor_event_causes_effect",
                OntologyNormalization(None, None, None),
            )
            return StageAExtraction((), (frame,), ())
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=extractor, modes=("combined",),
        )
        scores = result["cases"][0]["modes"][0]["slot_scores"]
        for slot_name in ("evidence_span", "actor", "event", "effect", "condition", "semantic_proposition"):
            self.assertEqual(scores[slot_name], {"hit_count": 0, "expected_count": 1})

    def test_stage_a_actual_and_frame_span_mismatch_is_ungrounded(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        def slot(role: str, phrase: str) -> SemanticSlot:
            start = source.index(phrase)
            return SemanticSlot(role, SourceAlignment("insight", start, start + len(phrase), phrase))
        full_span = SourceAlignment("insight", 0, len(source), source)
        partial_span = SourceAlignment("insight", 0, len("Flay prevents Tristana"), "Flay prevents Tristana")
        frame = SourceSemanticFrame(
            (full_span,),
            slot("actor", "Flay"),
            slot("event", "prevents"),
            slot("effect", "staying on target"),
            slot("condition", "after entry"),
            "actor_event_causes_effect",
            OntologyNormalization(None, None, None),
        )
        extraction = StageAExtraction(
            (), (frame,), (), evidence_spans=(partial_span,),
            slots={"actor": frame.actor, "event": frame.event, "effect": frame.effect, "condition": frame.condition},
            causal_direction="actor_event_causes_effect",
        )
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=lambda packet: extraction, modes=("insight",),
        )
        scores = result["cases"][0]["modes"][0]["slot_scores"]
        for slot_name in ("evidence_span", "actor", "event", "effect", "condition", "semantic_proposition"):
            self.assertEqual(scores[slot_name], {"hit_count": 0, "expected_count": 1})

    def test_unavailable_eligible_source_has_coverage_not_perfect_quality(self) -> None:
        unavailable = ({**self.cases[0], "source_video_id": "wrong-video"},)
        result = evaluate_source_modes(unavailable, resolver=self.resolver, extractor=lambda packet: (), modes=("transcript",))
        self.assertEqual(result["metrics"]["transcript"]["eligible_source_coverage"], 0.0)
        self.assertIsNone(result["metrics"]["transcript"]["proposition_recall"])

    def test_rejects_inconsistent_safe_zero_fixture_labels(self) -> None:
        fixture = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        fixture.write('{"cases":[{"id":"bad","insight_id":"1","source_video_id":"v1","eligible":false,"expected_propositions":[{}]}]}')
        fixture.close()
        try:
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                load_development_cases(fixture.name)
        finally:
            Path(fixture.name).unlink()

    def test_scores_grounded_role_preserving_alternate_as_semantic_match(self) -> None:
        def extractor(packet):
            text = packet.insight_text
            values = (("subject", "Flay"), ("predicate", "prevents"), ("effect", "staying on target after entry"), ("condition", "after entry"))
            return (ExtractedProposition(
                GroundedProposition("Flay", "prevents", "staying on target after entry", "after entry", ("1",)),
                tuple(PropositionAlignment(field, "insight", text.index(phrase), text.index(phrase) + len(phrase), phrase) for field, phrase in values),
            ),)
        result = evaluate_source_modes(self.cases[:1], resolver=self.resolver, extractor=extractor, modes=("insight",))
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["proposition_recall"], 1.0)
        self.assertEqual(metrics["exact_source_proposition_recall"], 0.0)

    def test_does_not_score_reversed_grounded_roles_as_semantic_match(self) -> None:
        def extractor(packet):
            text = packet.insight_text
            values = (("subject", "staying on target"), ("predicate", "prevents"), ("effect", "Flay"), ("condition", "after entry"))
            return (ExtractedProposition(
                GroundedProposition("staying on target", "prevents", "Flay", "after entry", ("1",)),
                tuple(PropositionAlignment(field, "insight", text.index(phrase), text.index(phrase) + len(phrase), phrase) for field, phrase in values),
            ),)
        result = evaluate_source_modes(self.cases[:1], resolver=self.resolver, extractor=extractor, modes=("insight",))
        self.assertEqual(result["metrics"]["insight"]["proposition_recall"], 0.0)

    def test_rejects_development_fixture_with_held_out_overlap(self) -> None:
        held = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        dev = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        held.write('{"cases":[{"evidence":[{"insight_id":"1","source_id":"held-v1"}]}]}'); held.close()
        dev.write('{"cases":[{"id":"bad","insight_id":"1","source_video_id":"v1","eligible":false,"expected_propositions":[]}]}'); dev.close()
        try:
            with self.assertRaisesRegex(ValueError, "overlaps"):
                load_development_cases(dev.name, held_out_path=held.name)
        finally:
            Path(dev.name).unlink(); Path(held.name).unlink()

    def test_rejects_partial_semantic_roles_and_held_out_source_overlap(self) -> None:
        held = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        dev = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        held.write('{"cases":[{"evidence":[{"insight_id":"other","source_id":"v1"}]}]}'); held.close()
        dev.write('{"cases":[{"id":"bad","insight_id":"1","source_video_id":"v1","eligible":true,"expected_propositions":[{"semantic_field_token_groups":{"subject":[["x"]]}}]}]}'); dev.close()
        try:
            with self.assertRaisesRegex(ValueError, "invalid semantic"):
                load_development_cases(dev.name, held_out_path=held.name)
            Path(dev.name).write_text('{"cases":[{"id":"bad","insight_id":"1","source_video_id":"v1","eligible":false,"expected_propositions":[]}]}')
            with self.assertRaisesRegex(ValueError, "source IDs"):
                load_development_cases(dev.name, held_out_path=held.name)
        finally:
            Path(dev.name).unlink(); Path(held.name).unlink()

    def test_rejects_held_out_overlap_without_fixture_metadata(self) -> None:
        held = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        dev = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        held.write('{"cases":[{"evidence":[{"insight_id":"1","source_id":"other-source"}]}]}'); held.close()
        dev.write('{"cases":[{"id":"bad","insight_id":"1","source_video_id":"v1","eligible":false,"expected_propositions":[]}]}'); dev.close()
        try:
            with self.assertRaisesRegex(ValueError, "overlaps"):
                load_development_cases(dev.name, held_out_path=held.name)
        finally:
            Path(dev.name).unlink(); Path(held.name).unlink()

    def test_errors_when_frozen_held_out_fixture_unavailable(self) -> None:
        fixture = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        fixture.write('{"cases":[{"id":"ok","insight_id":"1","source_video_id":"v1","eligible":false,"expected_propositions":[]}]}')
        fixture.close()
        try:
            with self.assertRaisesRegex(ValueError, "cannot load frozen held-out fixture"):
                load_development_cases(fixture.name, held_out_path=fixture.name + ".missing")
        finally:
            Path(fixture.name).unlink()

    def test_load_uses_repository_frozen_held_out_fixture_by_default(self) -> None:
        dev = Path(__file__).resolve().parent.parent / "data" / "relation_extraction_phase2d_dev_v0.json"
        self.assertTrue(Path(DEFAULT_HELD_OUT_FIXTURE).is_file())
        cases = load_development_cases(dev)
        self.assertTrue(cases)

    def test_requires_condition_semantic_group_and_normalizes_contractions(self) -> None:
        fixture = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        fixture.write('{"cases":[{"id":"bad","insight_id":"1","source_video_id":"v1","eligible":true,"expected_propositions":[{"condition_source":"after entry","semantic_field_token_groups":{"subject":[["x"]],"predicate":[["y"]],"effect":[["z"]]}}]}]}')
        fixture.close()
        try:
            with self.assertRaisesRegex(ValueError, "invalid semantic"):
                load_development_cases(fixture.name)
        finally:
            Path(fixture.name).unlink()
        self.assertEqual(_tokenize("can’t commit"), ("cant", "commit"))
        self.assertEqual(_tokenize("can commit"), ("can", "commit"))

    def test_rejects_missing_or_reversed_condition_operator(self) -> None:
        fixture = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        fixture.write('{"cases":[{"id":"bad","insight_id":"1","source_video_id":"v1","eligible":true,"expected_propositions":[{"condition_source":"if target is isolated","semantic_field_token_groups":{"subject":[["x"]],"predicate":[["y"]],"effect":[["z"]],"condition":[["isolated"]]}}]}]}')
        fixture.close()
        try:
            with self.assertRaisesRegex(ValueError, "condition operator"):
                load_development_cases(fixture.name)
        finally:
            Path(fixture.name).unlink()

    def test_requires_reviewed_closed_normalization_labels(self) -> None:
        base = {
            "subject_source": "x", "predicate_source": "y", "effect_source": "z",
            "condition_source": None,
            "semantic_field_token_groups": {
                "subject": [["x"]], "predicate": [["y"]], "effect": [["z"]],
            },
            "normalization_rationale": "Reviewed synthetic label.",
        }
        invalid = (
            ({**base}, "expected_normalization"),
            ({**base, "expected_normalization": {"actor_concept": "invented", "event_relation": None, "effect_concept": None}}, "actor concept"),
            ({**base, "expected_normalization": {"actor_concept": None, "event_relation": "invented", "effect_concept": None}}, "event relation"),
            ({**base, "expected_normalization": {"actor_concept": None, "event_relation": None, "effect_concept": "invented"}}, "effect concept"),
            ({**base, "expected_normalization": {"actor_concept": None, "event_relation": None, "effect_concept": None}, "normalization_rationale": ""}, "normalization rationale"),
        )
        for expected, message in invalid:
            with self.subTest(message=message):
                fixture = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
                json.dump({"cases": [{
                    "id": "bad", "insight_id": "1", "source_video_id": "v1",
                    "eligible": True, "expected_propositions": [expected],
                }]}, fixture)
                fixture.close()
                try:
                    with self.assertRaisesRegex(ValueError, message):
                        load_development_cases(fixture.name)
                finally:
                    Path(fixture.name).unlink()

    def test_stage_a_result_scores_all_slots_and_retains_artifacts(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        extraction = _stage_extraction(
            source, actor="Flay", event="prevents", effect="staying on target", condition="after entry",
        )
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=lambda packet: extraction, modes=("insight",),
        )
        entry = result["cases"][0]["modes"][0]
        self.assertEqual(entry["status"], "completed")
        self.assertEqual(entry["matched_count"], 1)
        self.assertEqual(entry["exact_matched_count"], 1)
        self.assertEqual(len(entry["artifacts"]), 1)
        self.assertEqual(entry["artifacts"][0]["stage"], "evidence_localization")
        self.assertEqual(entry["candidate_catalog"], [{
            "candidate_id": "insight:c001",
            "alignment": entry["evidence_spans"][0],
        }])
        self.assertEqual(entry["candidate_catalog_coverage"], {
            "hit_count": 1,
            "expected_count": 1,
            "catalog_count": 1,
            "valid_candidate_count": 1,
            "invalid_candidate_count": 0,
            "comparisons": [{
                "expected_index": 0,
                "covered": True,
                "source_kind": "insight",
                "candidate_ids": ["insight:c001"],
                "coalesced_spans": [entry["evidence_spans"][0]],
            }],
        })
        self.assertEqual(entry["evidence_spans"][0]["source_text"], source.strip())
        self.assertEqual(entry["semantic_frames"][0]["causal_direction"], "actor_event_causes_effect")
        self.assertEqual(entry["semantic_frames"][0]["normalization"], {"actor_concept": None, "event_relation": None, "effect_concept": None})
        self.assertEqual(entry["propositions"][0]["proposition"]["subject_source"], "Flay")
        self.assertEqual(entry["comparisons"][0]["actor_hit"], True)
        self.assertEqual(entry["comparisons"][0]["condition_operator_hit"], True)
        self.assertEqual(entry["comparisons"][0]["normalization_hit"], True)
        self.assertEqual(entry["comparisons"][0]["produced_normalization"], {"actor_concept": None, "event_relation": None, "effect_concept": None})
        self.assertEqual(entry["comparisons"][0]["semantic_proposition_hit"], True)
        self.assertIsNone(entry["comparisons"][0]["first_failed_transformation"])
        for slot in ("evidence_span", "actor", "event", "effect", "condition", "causal_direction", "normalization", "semantic_proposition", "exact_decomposition"):
            self.assertEqual(entry["slot_scores"][slot], {"hit_count": 1, "expected_count": 1})
        self.assertEqual(entry["slot_scores"]["unsupported_slots"], {"count": 0})
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["proposition_recall"], 1.0)
        self.assertEqual(metrics["slot_recall"]["actor"], {"hit_count": 1, "denominator": 1, "recall": 1.0})
        self.assertEqual(metrics["slot_recall"]["normalization"], {"hit_count": 1, "denominator": 1, "recall": 1.0})
        self.assertEqual(metrics["normalization_stage"], {"completed_count": 1, "abstained_count": 1, "mapped_count": 0, "failed_count": 0, "denominator": 1, "reached_count": 1})
        self.assertEqual(metrics["slot_reached"]["actor"], {"reached_count": 1, "hit_count": 1, "denominator": 1, "accuracy_when_reached": 1.0})
        self.assertEqual(metrics["slot_reached"]["normalization"], {"reached_count": 1, "hit_count": 1, "denominator": 1, "accuracy_when_reached": 1.0})
        self.assertEqual(metrics["slot_reached"]["exact_decomposition"], {"reached_count": 1, "hit_count": 1, "denominator": 1, "accuracy_when_reached": 1.0})
        self.assertEqual(metrics["candidate_catalog_coverage"], {
            "hit_count": 1, "denominator": 1, "recall": 1.0,
            "evaluated_entry_count": 1, "eligible_entry_count": 1,
            "complete": True,
        })
        self.assertEqual(metrics["unsupported_slot_total"], 0)

    def test_candidate_catalog_coverage_accepts_two_grounded_spans(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        extraction = _stage_extraction(
            source, actor="Flay", event="prevents", effect="staying on target",
            condition="after entry",
        )

        def candidate(candidate_id: str, phrase: str) -> ClauseCandidate:
            start = source.index(phrase)
            return ClauseCandidate(
                candidate_id,
                SourceAlignment("insight", start, start + len(phrase), phrase),
            )

        extraction = replace(extraction, candidate_catalog=(
            candidate("insight:c001", "Flay prevents"),
            candidate("insight:c002", "staying on target after entry"),
        ))
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver,
            extractor=lambda packet: extraction, modes=("insight",),
        )
        coverage = result["cases"][0]["modes"][0]["candidate_catalog_coverage"]
        self.assertEqual(coverage["hit_count"], 1)
        self.assertEqual(
            coverage["comparisons"][0]["candidate_ids"],
            ["insight:c001", "insight:c002"],
        )
        self.assertEqual(len(coverage["comparisons"][0]["coalesced_spans"]), 2)

    def test_candidate_catalog_coverage_rejects_three_candidate_requirement(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        extraction = _stage_extraction(
            source, actor="Flay", event="prevents", effect="staying on target",
            condition="after entry",
        )

        def candidate(candidate_id: str, phrase: str) -> ClauseCandidate:
            start = source.index(phrase)
            return ClauseCandidate(
                candidate_id,
                SourceAlignment("insight", start, start + len(phrase), phrase),
            )

        extraction = replace(extraction, candidate_catalog=(
            candidate("insight:c001", "Flay prevents"),
            candidate("insight:c002", "staying on target"),
            candidate("insight:c003", "after entry"),
        ))
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver,
            extractor=lambda packet: extraction, modes=("insight",),
        )
        coverage = result["cases"][0]["modes"][0]["candidate_catalog_coverage"]
        self.assertEqual(coverage["hit_count"], 0)
        self.assertFalse(coverage["comparisons"][0]["covered"])
        self.assertEqual(coverage["comparisons"][0]["candidate_ids"], [])

    def test_candidate_catalog_coverage_excludes_ungrounded_and_duplicate_ids(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        extraction = _stage_extraction(
            source, actor="Flay", event="prevents", effect="staying on target",
            condition="after entry",
        )
        full = SourceAlignment("insight", 0, len(source), source)
        fabricated = SourceAlignment("insight", 1, len(source), source)
        extraction = replace(extraction, candidate_catalog=(
            ClauseCandidate("insight:c001", full),
            ClauseCandidate("insight:c001", full),
            ClauseCandidate("insight:c002", fabricated),
        ))
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver,
            extractor=lambda packet: extraction, modes=("insight",),
        )
        coverage = result["cases"][0]["modes"][0]["candidate_catalog_coverage"]
        self.assertEqual(coverage["catalog_count"], 3)
        self.assertEqual(coverage["valid_candidate_count"], 0)
        self.assertEqual(coverage["invalid_candidate_count"], 3)
        self.assertEqual(coverage["hit_count"], 0)

    def test_normalization_recall_scores_exact_reviewed_mapping(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        expected = dict(self.cases[0]["expected_propositions"][0])
        expected["expected_normalization"] = {
            "actor_concept": "access", "event_relation": "denies",
            "effect_concept": "continuity",
        }
        expected["normalization_rationale"] = "Synthetic closed-ontology scoring fixture."
        case = ({**self.cases[0], "expected_propositions": [expected]},)

        correct = _stage_extraction(
            source, actor="Flay", event="prevents", effect="staying on target",
            condition="after entry",
            normalization=OntologyNormalization("access", "denies", "continuity"),
        )
        correct_result = evaluate_source_modes(
            case, resolver=self.resolver, extractor=lambda packet: correct,
            modes=("insight",),
        )
        correct_entry = correct_result["cases"][0]["modes"][0]
        self.assertTrue(correct_entry["comparisons"][0]["normalization_hit"])
        self.assertEqual(
            correct_result["metrics"]["insight"]["slot_recall"]["normalization"],
            {"hit_count": 1, "denominator": 1, "recall": 1.0},
        )

        wrong = _stage_extraction(
            source, actor="Flay", event="prevents", effect="staying on target",
            condition="after entry",
            normalization=OntologyNormalization("access", "enables", "continuity"),
        )
        wrong_result = evaluate_source_modes(
            case, resolver=self.resolver, extractor=lambda packet: wrong,
            modes=("insight",),
        )
        wrong_entry = wrong_result["cases"][0]["modes"][0]
        self.assertFalse(wrong_entry["comparisons"][0]["normalization_hit"])
        self.assertEqual(
            wrong_entry["comparisons"][0]["first_failed_transformation"],
            "ontology_normalization",
        )
        self.assertTrue(wrong_entry["comparisons"][0]["semantic_proposition_hit"])
        self.assertEqual(
            wrong_result["metrics"]["insight"]["slot_recall"]["normalization"],
            {"hit_count": 0, "denominator": 1, "recall": 0.0},
        )

    def test_stage_a_reversed_roles_are_not_a_semantic_match(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        extraction = _stage_extraction(
            source, actor="staying on target", event="prevents", effect="Flay", condition="after entry",
        )
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=lambda packet: extraction, modes=("insight",),
        )
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["proposition_recall"], 0.0)
        self.assertEqual(metrics["slot_recall"]["actor"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["effect"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["semantic_proposition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["causal_direction"], {"hit_count": 1, "denominator": 1, "recall": 1.0})

    def test_stage_a_reversed_direction_produces_no_proposition(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        extraction = _stage_extraction(
            source, actor="Flay", event="prevents", effect="staying on target",
            condition="after entry", direction="effect_causes_actor_event",
        )
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=lambda packet: extraction, modes=("insight",),
        )
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["proposition_recall"], 0.0)
        self.assertEqual(metrics["slot_recall"]["causal_direction"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["semantic_proposition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        for slot in ("actor", "event", "effect", "condition"):
            self.assertEqual(metrics["slot_recall"][slot], {"hit_count": 1, "denominator": 1, "recall": 1.0})
        comparison = result["cases"][0]["modes"][0]["comparisons"][0]
        self.assertEqual(comparison["matched"], False)
        self.assertTrue(comparison["actor_hit"])
        self.assertTrue(comparison["event_hit"])
        self.assertTrue(comparison["effect_hit"])
        self.assertTrue(comparison["condition_hit"])
        self.assertFalse(comparison["causal_direction_hit"])
        self.assertFalse(comparison["semantic_proposition_hit"])
        self.assertEqual(comparison["first_failed_transformation"], "causal_direction")

    def test_stage_a_partial_slot_matches_are_scored_separately(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        extraction = _stage_extraction(
            source, actor="Flay", event="Tristana", effect="staying on target", condition="after entry",
        )
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=lambda packet: extraction, modes=("insight",),
        )
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["proposition_recall"], 0.0)
        self.assertEqual(metrics["slot_recall"]["actor"], {"hit_count": 1, "denominator": 1, "recall": 1.0})
        self.assertEqual(metrics["slot_recall"]["event"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["effect"], {"hit_count": 1, "denominator": 1, "recall": 1.0})
        comparison = result["cases"][0]["modes"][0]["comparisons"][0]
        self.assertEqual(comparison["matched"], False)
        self.assertEqual(comparison["actor_hit"], True)
        self.assertEqual(comparison["event_hit"], False)

    def test_stage_a_condition_operator_mismatch_misses_official_condition_and_semantic(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        extraction = _stage_extraction(
            source, actor="Flay", event="prevents", effect="staying on target",
            condition="target after entry",
        )
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=lambda packet: extraction, modes=("insight",),
        )
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["proposition_recall"], 0.0)
        self.assertEqual(metrics["slot_recall"]["condition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["semantic_proposition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        comparison = result["cases"][0]["modes"][0]["comparisons"][0]
        self.assertEqual(comparison["condition_hit"], True)
        self.assertEqual(comparison["condition_operator_hit"], False)
        self.assertFalse(comparison["semantic_proposition_hit"])
        self.assertEqual(comparison["matched"], False)

    def test_stage_a_slots_outside_selected_spans_are_ungrounded(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        extraction = _stage_extraction(
            source, actor="Flay", event="prevents", effect="staying on target",
            condition="after entry", evidence_span="Flay prevents Tristana",
        )
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=lambda packet: extraction, modes=("insight",),
        )
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["proposition_recall"], 0.0)
        self.assertEqual(metrics["unsupported_proposition_rate"], 1.0)
        self.assertEqual(metrics["slot_recall"]["evidence_span"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        for slot in ("actor", "event", "effect", "condition", "semantic_proposition"):
            self.assertEqual(metrics["slot_recall"][slot], {"hit_count": 0, "denominator": 1, "recall": 0.0})

    def test_stage_a_unsupported_slot_failure_is_preserved_not_safe_zero(self) -> None:
        artifact = StageArtifact("event_extraction", '{"event":"tosses"}', None, "UnsupportedSourceSlot")
        extraction = StageAExtraction((), (), (artifact,), "event_extraction", 1)
        result = evaluate_source_modes(
            self.cases, resolver=self.resolver,
            extractor=lambda packet: extraction if packet.evidence_id == "1" else StageAExtraction((), (), ()),
            modes=("insight",),
        )
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["failure_case_count"], 1)
        self.assertEqual(metrics["unsupported_slot_total"], 1)
        self.assertEqual(metrics["safe_zero_accuracy"], 1.0)
        self.assertEqual(result["cases"][0]["modes"][0]["status"], "failure")
        self.assertEqual(result["cases"][0]["modes"][0]["reason"], "event_extraction")
        self.assertEqual(metrics["slot_recall"]["semantic_proposition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["assembled_proposition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})

    def test_stage_a_normalization_failure_keeps_slot_direction_diagnostics(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        def slot(role: str, phrase: str) -> SemanticSlot:
            start = source.index(phrase)
            return SemanticSlot(role, SourceAlignment("insight", start, start + len(phrase), phrase))
        span_text = source.strip()
        evidence = SourceAlignment("insight", 0, len(span_text), span_text)
        frame = SourceSemanticFrame(
            (evidence,),
            slot("actor", "Flay"),
            slot("event", "prevents"),
            slot("effect", "staying on target"),
            slot("condition", "after entry"),
            "actor_event_causes_effect",
            None,
            "ValueError",
        )
        slots = {"actor": frame.actor, "event": frame.event, "effect": frame.effect, "condition": frame.condition}
        artifacts = (
            StageArtifact("evidence_localization", '{"source":"insight","evidence_spans":["' + span_text + '"]}', {"source": "insight", "evidence_spans": [span_text]}),
            StageArtifact("actor_extraction", '{"actor":"Flay"}', {"actor": "Flay"}),
            StageArtifact("event_extraction", '{"event":"prevents"}', {"event": "prevents"}),
            StageArtifact("effect_extraction", '{"effect":"staying on target"}', {"effect": "staying on target"}),
            StageArtifact("condition_extraction", '{"condition":"after entry"}', {"condition": "after entry"}),
            StageArtifact("causal_direction", '{"causal_direction":"actor_event_causes_effect"}', {"causal_direction": "actor_event_causes_effect"}),
            StageArtifact("ontology_normalization", '{"actor_concept":"invented"}', None, "ValueError"),
        )
        extraction = StageAExtraction(
            propositions=(), frames=(frame,), artifacts=artifacts,
            failure_stage="ontology_normalization", unsupported_slot_count=0,
            evidence_spans=(evidence,), slots=slots,
            causal_direction="actor_event_causes_effect",
        )
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=lambda packet: extraction, modes=("insight",),
        )
        entry = result["cases"][0]["modes"][0]
        self.assertEqual(entry["status"], "failure")
        self.assertEqual(entry["reason"], "ontology_normalization")
        self.assertEqual(entry["first_failure"], {"stage": "ontology_normalization", "type": "ValueError"})
        self.assertEqual(
            entry["reached_stages"],
            ["evidence_localization", "actor_extraction", "event_extraction", "effect_extraction", "condition_extraction", "causal_direction"],
        )
        self.assertEqual(entry["evidence_spans"], [asdict(evidence)])
        self.assertEqual([item["role"] for item in entry["recovered_slots"]], ["actor", "event", "effect", "condition"])
        self.assertEqual(len(entry["semantic_frames"]), 1)
        scores = entry["slot_scores"]
        for slot_name in ("actor", "event", "effect", "condition"):
            self.assertEqual(scores[slot_name], {"hit_count": 1, "expected_count": 1})
        self.assertEqual(scores["causal_direction"], {"hit_count": 1, "expected_count": 1})
        self.assertEqual(scores["normalization_stage"]["completed_count"], 0)
        self.assertEqual(scores["normalization_stage"]["failed_count"], 1)
        self.assertEqual(scores["invented_slots"], {"count": 0})
        comparison = entry["comparisons"][0]
        self.assertTrue(comparison["actor_hit"])
        self.assertTrue(comparison["causal_direction_hit"])
        self.assertTrue(comparison["normalization_failed"])
        self.assertFalse(comparison["normalization_completed"])
        self.assertEqual(comparison["matched"], False)
        self.assertEqual(result["metrics"]["insight"]["failure_case_count"], 1)
        self.assertEqual(result["metrics"]["insight"]["invented_slot_total"], 0)

    def test_stage_a_partial_slot_failure_retains_preceding_diagnostics(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        def slot(role: str, phrase: str) -> SemanticSlot:
            start = source.index(phrase)
            return SemanticSlot(role, SourceAlignment("insight", start, start + len(phrase), phrase))
        span_text = "Flay prevents Tristana"
        evidence = SourceAlignment("insight", 0, len(span_text), span_text)
        actor = slot("actor", "Flay")
        event = slot("event", "prevents")
        artifacts = (
            StageArtifact("evidence_localization", '{"source":"insight","evidence_spans":["' + span_text + '"]}', {"source": "insight", "evidence_spans": [span_text]}),
            StageArtifact("actor_extraction", '{"actor":"Flay"}', {"actor": "Flay"}),
            StageArtifact("event_extraction", '{"event":"prevents"}', {"event": "prevents"}),
            StageArtifact("effect_extraction", '{"effect":"tosses"}', None, "UnsupportedSourceSlot"),
        )
        extraction = StageAExtraction(
            propositions=(), frames=(), artifacts=artifacts,
            failure_stage="effect_extraction", unsupported_slot_count=1,
            evidence_spans=(evidence,), slots={"actor": actor, "event": event},
        )
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=lambda packet: extraction, modes=("insight",),
        )
        entry = result["cases"][0]["modes"][0]
        self.assertEqual(entry["status"], "failure")
        self.assertEqual(entry["reason"], "effect_extraction")
        self.assertEqual(entry["first_failure"], {"stage": "effect_extraction", "type": "UnsupportedSourceSlot"})
        self.assertEqual(entry["reached_stages"], ["evidence_localization", "actor_extraction", "event_extraction"])
        self.assertEqual(entry["evidence_spans"], [asdict(evidence)])
        self.assertEqual([item["role"] for item in entry["recovered_slots"]], ["actor", "event"])
        scores = entry["slot_scores"]
        self.assertEqual(scores["actor"], {"hit_count": 1, "expected_count": 1})
        self.assertEqual(scores["event"], {"hit_count": 1, "expected_count": 1})
        self.assertEqual(scores["effect"], {"hit_count": 0, "expected_count": 1})
        self.assertEqual(scores["condition"], {"hit_count": 0, "expected_count": 1})
        self.assertEqual(scores["unsupported_slots"], {"count": 1})
        self.assertEqual(scores["slot_reached"]["actor"], {"reached_count": 1, "hit_count": 1, "accuracy_when_reached": 1.0})
        self.assertEqual(scores["slot_reached"]["effect"], {"reached_count": 0, "hit_count": 0, "accuracy_when_reached": None})
        self.assertEqual(scores["slot_reached"]["condition"], {"reached_count": 0, "hit_count": 0, "accuracy_when_reached": None})
        comparison = entry["comparisons"][0]
        self.assertTrue(comparison["actor_hit"])
        self.assertTrue(comparison["event_hit"])
        self.assertFalse(comparison["effect_hit"])
        self.assertFalse(comparison["condition_hit"])
        self.assertFalse(comparison["semantic_proposition_hit"])
        self.assertFalse(comparison["evidence_span_hit"])
        self.assertEqual(comparison["first_failed_transformation"], "evidence_localization")
        self.assertEqual(comparison["matched"], False)
        self.assertIn("expected", comparison)
        self.assertEqual(result["metrics"]["insight"]["unsupported_slot_total"], 1)

    def test_stage_a_malformed_normalization_is_not_invented(self) -> None:
        artifact = StageArtifact("ontology_normalization", '{"actor_concept":"invented"}', None, "ValueError")
        extraction = StageAExtraction((), (), (artifact,), "ontology_normalization", 0)
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=lambda packet: extraction, modes=("insight",),
        )
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["invented_slot_total"], 0)
        self.assertEqual(metrics["failure_case_count"], 1)
        self.assertEqual(result["cases"][0]["modes"][0]["slot_scores"]["invented_slots"], {"count": 0})

    def test_stage_a_explicit_invented_ontology_taxonomy_is_consumed(self) -> None:
        artifact = StageArtifact("ontology_normalization", '{"actor_concept":"bogus"}', None, "ValueError")
        extraction = StageAExtraction(
            propositions=(), frames=(), artifacts=(artifact,),
            failure_stage="ontology_normalization", unsupported_slot_count=0,
            invented_ontology_count=2, invented_ontology_taxonomy={"bogus": 1, "other": 1},
        )
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=lambda packet: extraction, modes=("insight",),
        )
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["invented_slot_total"], 2)
        entry = result["cases"][0]["modes"][0]
        self.assertEqual(entry["slot_scores"]["invented_slots"], {"count": 2, "taxonomy": {"bogus": 1, "other": 1}})
        self.assertEqual(entry["first_failure"], {"stage": "ontology_normalization", "type": "ValueError"})

    def test_stage_a_correct_frame_with_normalization_failure_scores_semantic_hit(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        extraction = _stage_extraction(
            source, actor="Flay", event="prevents", effect="staying on target",
            condition="after entry", normalization_failure="ValueError",
        )
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=lambda packet: extraction, modes=("insight",),
        )
        entry = result["cases"][0]["modes"][0]
        self.assertEqual(entry["status"], "failure")
        self.assertEqual(entry["reason"], "ontology_normalization")
        self.assertEqual(entry["propositions"], [])
        scores = entry["slot_scores"]
        self.assertEqual(scores["semantic_proposition"], {"hit_count": 1, "expected_count": 1})
        self.assertEqual(scores["assembled_proposition"], {"hit_count": 0, "expected_count": 1})
        self.assertEqual(scores["exact_decomposition"], {"hit_count": 0, "expected_count": 1})
        self.assertEqual(scores["normalization_stage"]["completed_count"], 0)
        self.assertEqual(scores["normalization_stage"]["failed_count"], 1)
        comparison = entry["comparisons"][0]
        self.assertFalse(comparison["matched"])
        self.assertTrue(comparison["actor_hit"])
        self.assertTrue(comparison["causal_direction_hit"])
        self.assertTrue(comparison["normalization_failed"])
        self.assertFalse(comparison["normalization_completed"])
        self.assertTrue(comparison["semantic_proposition_hit"])
        self.assertEqual(comparison["first_failed_transformation"], "ontology_normalization")
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["failure_case_count"], 1)
        self.assertEqual(metrics["proposition_recall"], 0.0)
        self.assertEqual(metrics["slot_recall"]["semantic_proposition"], {"hit_count": 1, "denominator": 1, "recall": 1.0})
        self.assertEqual(metrics["slot_recall"]["assembled_proposition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})

    def test_stage_a_malformed_provider_failure_counts_semantic_miss_not_unavailable(self) -> None:
        def extractor(packet: PropositionPacket):
            raise RuntimeError("malformed provider payload")

        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=extractor, modes=("insight",),
        )
        entry = result["cases"][0]["modes"][0]
        self.assertEqual(entry["status"], "failure")
        self.assertEqual(entry["reason"], "RuntimeError")
        self.assertEqual(entry["predicted_count"], 0)
        for slot in ("evidence_span", "actor", "event", "effect", "condition", "causal_direction", "semantic_proposition", "assembled_proposition", "exact_decomposition"):
            self.assertEqual(entry["slot_scores"][slot], {"hit_count": 0, "expected_count": 1})
        self.assertEqual(entry["slot_scores"]["semantic_proposition"], {"hit_count": 0, "expected_count": 1})
        self.assertEqual(entry["slot_scores"]["normalization_stage"], {"denominator": 1, "reached_count": 0, "completed_count": 0, "abstained_count": 0, "mapped_count": 0, "failed_count": 0})
        self.assertEqual(entry["slot_scores"]["slot_reached"]["actor"], {"reached_count": 0, "hit_count": 0, "accuracy_when_reached": None})
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["failure_case_count"], 1)
        self.assertEqual(metrics["unavailable_case_count"], 0)
        self.assertEqual(metrics["slot_recall"]["semantic_proposition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["exact_decomposition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["actor"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["evidence_span"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["causal_direction"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_reached"]["actor"], {"reached_count": 0, "hit_count": 0, "denominator": 0, "accuracy_when_reached": None})
        self.assertEqual(metrics["normalization_stage"]["denominator"], 1)
        self.assertEqual(metrics["normalization_stage"]["reached_count"], 0)

    def test_proposition_recall_denominator_includes_provider_failures(self) -> None:
        with sqlite3.connect(self.temp.name) as conn:
            conn.execute("INSERT INTO insights VALUES ('8', 'v1', 'Flay prevents Tristana from staying on target after entry.')")
        label = self.cases[0]["expected_propositions"][0]
        cases = (
            {"id": "ok", "insight_id": "1", "source_video_id": "v1", "eligible": True, "expected_propositions": [label]},
            {"id": "down", "insight_id": "8", "source_video_id": "v1", "eligible": True, "expected_propositions": [label]},
        )
        def extractor(packet: PropositionPacket):
            if packet.evidence_id != "1":
                raise RuntimeError("provider down")
            text = packet.sources()[0].text
            values = (("subject", "Flay"), ("predicate", "prevents"), ("effect", "staying on target"), ("condition", "after entry"))
            return (ExtractedProposition(
                GroundedProposition("Flay", "prevents", "staying on target", "after entry", ("1",)),
                tuple(PropositionAlignment(field, "insight", text.index(phrase), text.index(phrase) + len(phrase), phrase) for field, phrase in values),
            ),)
        result = evaluate_source_modes(cases, resolver=self.resolver, extractor=extractor, modes=("insight",))
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["completed_case_count"], 1)
        self.assertEqual(metrics["failure_case_count"], 1)
        self.assertEqual(metrics["eligible_case_count"], 2)
        self.assertEqual(metrics["proposition_recall"], 0.5)
        self.assertEqual(metrics["exact_source_proposition_recall"], 0.5)
        self.assertEqual(metrics["slot_recall"]["semantic_proposition"], {"hit_count": 1, "denominator": 2, "recall": 0.5})

    def test_safe_zero_accuracy_denominator_includes_provider_failures(self) -> None:
        with sqlite3.connect(self.temp.name) as conn:
            conn.execute("INSERT INTO insights VALUES ('9', 'v1', 'Generic advice only.')")
        cases = (
            {"id": "sz-ok", "insight_id": "2", "source_video_id": "v1", "eligible": False, "expected_propositions": []},
            {"id": "sz-down", "insight_id": "9", "source_video_id": "v1", "eligible": False, "expected_propositions": []},
        )
        def extractor(packet: PropositionPacket):
            if packet.evidence_id != "2":
                raise RuntimeError("provider down")
            return ()
        result = evaluate_source_modes(cases, resolver=self.resolver, extractor=extractor, modes=("insight",))
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["completed_case_count"], 1)
        self.assertEqual(metrics["failure_case_count"], 1)
        self.assertEqual(metrics["safe_zero_accuracy"], 0.5)

    def test_stage_a_five_conceptual_scenarios_aggregate_slot_recall(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        label = {
            "subject_source": "Flay", "predicate_source": "prevents", "effect_source": "staying on target",
            "condition_source": "after entry", "condition_operator": "after",
            "semantic_field_token_groups": {"subject": [["Flay"]], "predicate": [["prevent", "prevents"]], "effect": [["staying"]], "condition": [["after"], ["entry"]]},
        }
        with sqlite3.connect(self.temp.name) as conn:
            for insight_id in ("3", "4", "5", "6", "7"):
                conn.execute(
                    "INSERT INTO insights VALUES (?, 'v1', 'Flay prevents Tristana from staying on target after entry.')",
                    (insight_id,),
                )
        completed = _stage_extraction(
            source, actor="Flay", event="prevents", effect="staying on target", condition="after entry",
        )
        direction = _stage_extraction(
            source, actor="Flay", event="prevents", effect="staying on target",
            condition="after entry", direction="effect_causes_actor_event",
        )
        normalization = _stage_extraction(
            source, actor="Flay", event="prevents", effect="staying on target",
            condition="after entry", normalization_failure="ValueError",
        )

        def slot(role: str, phrase: str) -> SemanticSlot:
            start = source.index(phrase)
            return SemanticSlot(role, SourceAlignment("insight", start, start + len(phrase), phrase))

        span_text = "Flay prevents Tristana"
        evidence = SourceAlignment("insight", 0, len(span_text), span_text)
        partial = StageAExtraction(
            propositions=(), frames=(), artifacts=(
                StageArtifact("evidence_localization", '{"source":"insight","evidence_spans":["' + span_text + '"]}', {"source": "insight", "evidence_spans": [span_text]}),
                StageArtifact("actor_extraction", '{"actor":"Flay"}', {"actor": "Flay"}),
                StageArtifact("event_extraction", '{"event":"prevents"}', {"event": "prevents"}),
                StageArtifact("effect_extraction", '{"effect":"tosses"}', None, "UnsupportedSourceSlot"),
            ),
            failure_stage="effect_extraction", unsupported_slot_count=1,
            evidence_spans=(evidence,), slots={"actor": slot("actor", "Flay"), "event": slot("event", "prevents")},
        )
        localize = StageAExtraction(
            propositions=(), frames=(), artifacts=(
                StageArtifact("evidence_localization", '{"source":null,"evidence_spans":[]}', None, "NoSourceWindow"),
            ),
            failure_stage="evidence_localization",
        )
        by_id = {"1": completed, "3": direction, "4": normalization, "5": partial, "7": localize}
        cases = tuple(
            {"id": case_id, "insight_id": insight_id, "source_video_id": source_video_id,
             "eligible": True, "expected_propositions": [label]}
            for case_id, insight_id, source_video_id in (
                ("completed", "1", "v1"),
                ("direction", "3", "v1"),
                ("normalization", "4", "v1"),
                ("partial", "5", "v1"),
                ("localize", "7", "v1"),
                ("unavailable", "6", "wrong-video"),
            )
        )
        run = lambda case_set: evaluate_source_modes(
            case_set, resolver=self.resolver, extractor=lambda packet: by_id[packet.evidence_id],
            modes=("combined",),
        )["metrics"]["combined"]

        metrics = run(tuple(case for case in cases if case["id"] != "localize"))
        self.assertEqual(metrics["completed_case_count"], 2)
        self.assertEqual(metrics["failure_case_count"], 2)
        self.assertEqual(metrics["unavailable_case_count"], 1)
        self.assertEqual(metrics["eligible_source_coverage"], 0.8)
        self.assertEqual(metrics["slot_recall"]["evidence_span"], {"hit_count": 3, "denominator": 4, "recall": 0.75})
        self.assertEqual(metrics["slot_recall"]["actor"], {"hit_count": 4, "denominator": 4, "recall": 1.0})
        self.assertEqual(metrics["slot_recall"]["event"], {"hit_count": 4, "denominator": 4, "recall": 1.0})
        self.assertEqual(metrics["slot_recall"]["effect"], {"hit_count": 3, "denominator": 4, "recall": 0.75})
        self.assertEqual(metrics["slot_recall"]["condition"], {"hit_count": 3, "denominator": 4, "recall": 0.75})
        self.assertEqual(metrics["slot_recall"]["causal_direction"], {"hit_count": 2, "denominator": 4, "recall": 0.5})
        self.assertEqual(metrics["slot_recall"]["semantic_proposition"], {"hit_count": 2, "denominator": 4, "recall": 0.5})
        self.assertEqual(metrics["slot_recall"]["assembled_proposition"], {"hit_count": 1, "denominator": 4, "recall": 0.25})
        self.assertEqual(metrics["slot_recall"]["exact_decomposition"], {"hit_count": 1, "denominator": 4, "recall": 0.25})
        self.assertEqual(metrics["normalization_stage"], {"completed_count": 2, "abstained_count": 2, "mapped_count": 0, "failed_count": 1, "denominator": 4, "reached_count": 3})
        self.assertEqual(metrics["slot_reached"]["actor"], {"reached_count": 4, "hit_count": 4, "denominator": 4, "accuracy_when_reached": 1.0})
        self.assertEqual(metrics["slot_reached"]["effect"], {"reached_count": 3, "hit_count": 3, "denominator": 3, "accuracy_when_reached": 1.0})
        self.assertEqual(metrics["slot_reached"]["condition"], {"reached_count": 3, "hit_count": 3, "denominator": 3, "accuracy_when_reached": 1.0})
        self.assertEqual(metrics["slot_reached"]["causal_direction"], {"reached_count": 3, "hit_count": 2, "denominator": 3, "accuracy_when_reached": 2 / 3})
        self.assertEqual(metrics["slot_reached"]["semantic_proposition"], {"reached_count": 3, "hit_count": 2, "denominator": 3, "accuracy_when_reached": 2 / 3})
        self.assertEqual(metrics["slot_reached"]["assembled_proposition"], {"reached_count": 1, "hit_count": 1, "denominator": 1, "accuracy_when_reached": 1.0})

        metrics = run(cases)
        self.assertEqual(metrics["failure_case_count"], 3)
        self.assertEqual(metrics["eligible_source_coverage"], 5 / 6)
        self.assertEqual(metrics["slot_recall"]["evidence_span"], {"hit_count": 3, "denominator": 5, "recall": 0.6})
        self.assertEqual(metrics["slot_recall"]["actor"], {"hit_count": 4, "denominator": 5, "recall": 0.8})
        self.assertEqual(metrics["slot_recall"]["event"], {"hit_count": 4, "denominator": 5, "recall": 0.8})
        self.assertEqual(metrics["slot_recall"]["effect"], {"hit_count": 3, "denominator": 5, "recall": 0.6})
        self.assertEqual(metrics["slot_recall"]["condition"], {"hit_count": 3, "denominator": 5, "recall": 0.6})
        self.assertEqual(metrics["slot_recall"]["causal_direction"], {"hit_count": 2, "denominator": 5, "recall": 0.4})
        self.assertEqual(metrics["slot_recall"]["semantic_proposition"], {"hit_count": 2, "denominator": 5, "recall": 0.4})
        self.assertEqual(metrics["slot_recall"]["assembled_proposition"], {"hit_count": 1, "denominator": 5, "recall": 0.2})
        self.assertEqual(metrics["slot_recall"]["exact_decomposition"], {"hit_count": 1, "denominator": 5, "recall": 0.2})
        self.assertEqual(metrics["normalization_stage"], {"completed_count": 2, "abstained_count": 2, "mapped_count": 0, "failed_count": 1, "denominator": 5, "reached_count": 3})
        for slot in ("actor", "event"):
            self.assertEqual(metrics["slot_reached"][slot], {"reached_count": 4, "hit_count": 4, "denominator": 4, "accuracy_when_reached": 1.0})
        for slot in ("effect", "condition"):
            self.assertEqual(metrics["slot_reached"][slot], {"reached_count": 3, "hit_count": 3, "denominator": 3, "accuracy_when_reached": 1.0})

    def test_five_positives_with_one_localization_failure_report_actor_4_of_5(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        label = {
            "subject_source": "Flay", "predicate_source": "prevents", "effect_source": "staying on target",
            "condition_source": "after entry", "condition_operator": "after",
            "semantic_field_token_groups": {"subject": [["Flay"]], "predicate": [["prevent", "prevents"]], "effect": [["staying"]], "condition": [["after"], ["entry"]]},
        }
        with sqlite3.connect(self.temp.name) as conn:
            for insight_id in ("8", "9", "10", "11"):
                conn.execute(
                    "INSERT INTO insights VALUES (?, 'v1', 'Flay prevents Tristana from staying on target after entry.')",
                    (insight_id,),
                )
        completed = _stage_extraction(
            source, actor="Flay", event="prevents", effect="staying on target", condition="after entry",
        )
        localize = StageAExtraction(
            propositions=(), frames=(), artifacts=(
                StageArtifact("evidence_localization", '{"source":null,"evidence_spans":[]}', None, "NoSourceWindow"),
            ),
            failure_stage="evidence_localization",
        )
        by_id = {"1": completed, "8": completed, "9": completed, "10": localize, "11": completed}
        cases = tuple(
            {"id": case_id, "insight_id": insight_id, "source_video_id": "v1",
             "eligible": True, "expected_propositions": [label]}
            for case_id, insight_id in (
                ("c1", "1"), ("c2", "8"), ("c3", "9"), ("c4", "10"), ("c5", "11"),
            )
        )
        result = evaluate_source_modes(
            cases, resolver=self.resolver, extractor=lambda packet: by_id[packet.evidence_id],
            modes=("combined",),
        )
        metrics = result["metrics"]["combined"]
        self.assertEqual(metrics["unavailable_case_count"], 0)
        self.assertEqual(metrics["eligible_source_coverage"], 1.0)
        self.assertEqual(metrics["slot_recall"]["actor"], {"hit_count": 4, "denominator": 5, "recall": 0.8})
        self.assertEqual(metrics["slot_recall"]["event"], {"hit_count": 4, "denominator": 5, "recall": 0.8})
        self.assertEqual(metrics["slot_recall"]["effect"], {"hit_count": 4, "denominator": 5, "recall": 0.8})
        self.assertEqual(metrics["slot_recall"]["condition"], {"hit_count": 4, "denominator": 5, "recall": 0.8})
        self.assertEqual(metrics["slot_recall"]["semantic_proposition"], {"hit_count": 4, "denominator": 5, "recall": 0.8})
        self.assertEqual(metrics["slot_recall"]["causal_direction"], {"hit_count": 4, "denominator": 5, "recall": 0.8})
        self.assertEqual(metrics["slot_reached"]["actor"], {"reached_count": 4, "hit_count": 4, "denominator": 4, "accuracy_when_reached": 1.0})
        self.assertEqual(metrics["normalization_stage"]["denominator"], 5)
        self.assertEqual(metrics["normalization_stage"]["reached_count"], 4)
        failure_entry = next(
            entry for case in result["cases"] for entry in case["modes"]
            if entry["status"] == "failure"
        )
        self.assertEqual(failure_entry["first_failure"], {"stage": "evidence_localization", "type": "NoSourceWindow"})
        self.assertFalse(failure_entry["coherence"]["coherent"])
        self.assertEqual(failure_entry["comparisons"][0]["first_failed_transformation"], "evidence_localization")
        self.assertFalse(failure_entry["comparisons"][0]["semantic_proposition_hit"])

    def test_eligible_evidence_localization_abstention_is_first_failure(self) -> None:
        artifact = StageArtifact(
            "evidence_localization", '{"source":null,"evidence_spans":[]}',
            {"source": None, "evidence_spans": []},
        )
        extraction = StageAExtraction((), (), (artifact,))
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver, extractor=lambda packet: extraction, modes=("insight",),
        )
        entry = result["cases"][0]["modes"][0]
        self.assertEqual(entry["status"], "completed")
        self.assertTrue(entry["coherence"]["coherent"])
        self.assertIsNone(entry["first_failure"])
        scores = entry["slot_scores"]
        for slot_name in ("evidence_span", "actor", "event", "effect", "condition", "semantic_proposition"):
            self.assertEqual(scores[slot_name], {"hit_count": 0, "expected_count": 1})
        comparison = entry["comparisons"][0]
        self.assertFalse(comparison["evidence_span_hit"])
        self.assertFalse(comparison["semantic_proposition_hit"])
        self.assertEqual(comparison["first_failed_transformation"], "evidence_localization")
        self.assertEqual(result["metrics"]["insight"]["proposition_recall"], 0.0)

    def test_safe_zero_evidence_localization_abstention_remains_successful(self) -> None:
        artifact = StageArtifact(
            "evidence_localization", '{"source":null,"evidence_spans":[]}',
            {"source": None, "evidence_spans": []},
        )
        extraction = StageAExtraction((), (), (artifact,))
        result = evaluate_source_modes(
            self.cases, resolver=self.resolver, extractor=lambda packet: extraction, modes=("insight",),
        )
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["safe_zero_accuracy"], 1.0)
        self.assertEqual(metrics["failure_case_count"], 0)

    def test_legacy_tuple_still_reports_slot_recall_counts(self) -> None:
        def extractor(packet: PropositionPacket):
            text = packet.sources()[0].text
            values = (("subject", "Flay"), ("predicate", "prevents"), ("effect", "staying on target"), ("condition", "after entry"))
            return (ExtractedProposition(
                GroundedProposition("Flay", "prevents", "staying on target", "after entry", ("1",)),
                tuple(PropositionAlignment(field, "insight", text.index(phrase), text.index(phrase) + len(phrase), phrase) for field, phrase in values),
            ),)
        result = evaluate_source_modes(self.cases[:1], resolver=self.resolver, extractor=extractor, modes=("insight",))
        metrics = result["metrics"]["insight"]
        for slot in ("actor", "event", "effect", "condition", "semantic_proposition", "exact_decomposition"):
            self.assertEqual(metrics["slot_recall"][slot], {"hit_count": 1, "denominator": 1, "recall": 1.0})
        self.assertEqual(metrics["slot_recall"]["evidence_span"], {"hit_count": 0, "denominator": 0, "recall": None})
        self.assertEqual(metrics["slot_recall"]["normalization"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["normalization_stage"]["denominator"], 0)

    def test_stage_a_contradictory_direction_zeroes_direction_and_semantic(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        def slot(role: str, phrase: str) -> SemanticSlot:
            start = source.index(phrase)
            return SemanticSlot(role, SourceAlignment("insight", start, start + len(phrase), phrase))
        evidence = SourceAlignment("insight", 0, len(source), source)
        slots = {
            "actor": slot("actor", "Flay"),
            "event": slot("event", "prevents"),
            "effect": slot("effect", "staying on target"),
            "condition": slot("condition", "after entry"),
        }
        for frame_direction, actual_direction in (
            ("effect_causes_actor_event", "actor_event_causes_effect"),
            ("actor_event_causes_effect", "effect_causes_actor_event"),
        ):
            with self.subTest(frame_direction=frame_direction, actual_direction=actual_direction):
                frame = SourceSemanticFrame(
                    (evidence,), slots["actor"], slots["event"], slots["effect"], slots["condition"],
                    frame_direction,  # type: ignore[arg-type]
                    OntologyNormalization(None, None, None),
                )
                extraction = StageAExtraction(
                    (), (frame,), (), evidence_spans=(evidence,), slots=slots,
                    causal_direction=actual_direction,
                )
                result = evaluate_source_modes(
                    self.cases[:1], resolver=self.resolver,
                    extractor=lambda packet: extraction, modes=("insight",),
                )
                entry = result["cases"][0]["modes"][0]
                scores = entry["slot_scores"]
                self.assertFalse(entry["coherence"]["coherent"])
                self.assertEqual(entry["coherence"]["direction_consistent"], False)
                for slot_name in ("actor", "event", "effect", "condition"):
                    self.assertEqual(scores[slot_name], {"hit_count": 1, "expected_count": 1})
                for slot_name in ("causal_direction", "semantic_proposition", "assembled_proposition", "exact_decomposition"):
                    self.assertEqual(scores[slot_name], {"hit_count": 0, "expected_count": 1})
                self.assertEqual(
                    scores["slot_reached"]["causal_direction"],
                    {"reached_count": 1, "hit_count": 0, "accuracy_when_reached": 0.0},
                )
                comparison = entry["comparisons"][0]
                self.assertFalse(comparison["causal_direction_hit"])
                self.assertFalse(comparison["semantic_proposition_hit"])
                self.assertTrue(comparison["actor_hit"])
                self.assertEqual(comparison["first_failed_transformation"], "causal_direction")
                metrics = result["metrics"]["insight"]
                self.assertEqual(metrics["slot_recall"]["causal_direction"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
                self.assertEqual(metrics["slot_recall"]["semantic_proposition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
                self.assertEqual(metrics["slot_recall"]["actor"], {"hit_count": 1, "denominator": 1, "recall": 1.0})

    def test_stage_a_slot_disagreement_zeroes_affected_slot_and_semantic(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        def slot(role: str, phrase: str) -> SemanticSlot:
            start = source.index(phrase)
            return SemanticSlot(role, SourceAlignment("insight", start, start + len(phrase), phrase))
        evidence = SourceAlignment("insight", 0, len(source), source)
        frame = SourceSemanticFrame(
            (evidence,), slot("actor", "Flay"), slot("event", "prevents"),
            slot("effect", "staying on target"), slot("condition", "after entry"),
            "actor_event_causes_effect", OntologyNormalization(None, None, None),
        )
        assembled = assemble_grounded_proposition(frame, "1")
        extraction = StageAExtraction(
            (assembled,), (frame,), (), evidence_spans=(evidence,),
            slots={
                "actor": slot("actor", "Tristana"),
                "event": slot("event", "prevents"),
                "effect": slot("effect", "staying on target"),
                "condition": slot("condition", "after entry"),
            },
            causal_direction="actor_event_causes_effect",
        )
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver,
            extractor=lambda packet: extraction, modes=("insight",),
        )
        entry = result["cases"][0]["modes"][0]
        scores = entry["slot_scores"]
        self.assertEqual(entry["coherence"]["coherent"], False)
        self.assertEqual(entry["coherence"]["slots_consistent"], False)
        self.assertEqual(entry["coherence"]["slot_conflicts"], ["actor"])
        self.assertEqual(entry["coherence"]["direction_consistent"], True)
        self.assertEqual(entry["coherence"]["assembly_consistent"], True)
        self.assertEqual(entry["matched_count"], 0)
        self.assertEqual(entry["exact_matched_count"], 0)
        self.assertEqual(entry["predicted_count"], 1)
        self.assertEqual(entry["false_positive_count"], 1)
        self.assertEqual(entry["missed_count"], 1)
        self.assertEqual(scores["actor"], {"hit_count": 0, "expected_count": 1})
        for slot_name in ("event", "effect", "condition", "causal_direction", "evidence_span"):
            self.assertEqual(scores[slot_name], {"hit_count": 1, "expected_count": 1})
        self.assertEqual(scores["semantic_proposition"], {"hit_count": 0, "expected_count": 1})
        self.assertEqual(scores["assembled_proposition"], {"hit_count": 0, "expected_count": 1})
        self.assertEqual(scores["exact_decomposition"], {"hit_count": 0, "expected_count": 1})
        self.assertEqual(
            scores["slot_reached"]["actor"],
            {"reached_count": 1, "hit_count": 0, "accuracy_when_reached": 0.0},
        )
        comparison = entry["comparisons"][0]
        self.assertFalse(comparison["actor_hit"])
        self.assertTrue(comparison["event_hit"])
        self.assertFalse(comparison["semantic_proposition_hit"])
        self.assertFalse(comparison["matched"])
        self.assertFalse(comparison["exact"])
        self.assertEqual(comparison["first_failed_transformation"], "actor_extraction")
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["slot_recall"]["actor"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["semantic_proposition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["assembled_proposition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["exact_decomposition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["proposition_recall"], 0.0)
        self.assertEqual(metrics["exact_source_proposition_recall"], 0.0)

    def test_stage_a_direction_contradiction_blocks_frame_credit(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        def slot(role: str, phrase: str) -> SemanticSlot:
            start = source.index(phrase)
            return SemanticSlot(role, SourceAlignment("insight", start, start + len(phrase), phrase))
        evidence = SourceAlignment("insight", 0, len(source), source)
        frame = SourceSemanticFrame(
            (evidence,), slot("actor", "Flay"), slot("event", "prevents"),
            slot("effect", "staying on target"), slot("condition", "after entry"),
            "actor_event_causes_effect", OntologyNormalization(None, None, None),
        )
        assembled = assemble_grounded_proposition(frame, "1")
        self.assertIsNotNone(assembled)
        extraction = StageAExtraction(
            (assembled,), (frame,), (), evidence_spans=(evidence,),
            slots={
                "actor": slot("actor", "Flay"),
                "event": slot("event", "prevents"),
                "effect": slot("effect", "staying on target"),
                "condition": slot("condition", "after entry"),
            },
            causal_direction="effect_causes_actor_event",
        )
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver,
            extractor=lambda packet: extraction, modes=("insight",),
        )
        entry = result["cases"][0]["modes"][0]
        scores = entry["slot_scores"]
        self.assertEqual(entry["coherence"]["coherent"], False)
        self.assertEqual(entry["coherence"]["grounded"], True)
        self.assertEqual(entry["coherence"]["slots_consistent"], True)
        self.assertEqual(entry["coherence"]["direction_consistent"], False)
        self.assertEqual(entry["coherence"]["assembly_consistent"], True)
        self.assertEqual(entry["predicted_count"], 1)
        self.assertEqual(entry["matched_count"], 0)
        self.assertEqual(entry["exact_matched_count"], 0)
        self.assertEqual(entry["false_positive_count"], 1)
        self.assertEqual(entry["missed_count"], 1)
        for slot_name in ("actor", "event", "effect", "condition", "evidence_span"):
            self.assertEqual(scores[slot_name], {"hit_count": 1, "expected_count": 1})
        for slot_name in ("causal_direction", "semantic_proposition", "assembled_proposition", "exact_decomposition"):
            self.assertEqual(scores[slot_name], {"hit_count": 0, "expected_count": 1})
        self.assertEqual(
            scores["slot_reached"]["causal_direction"],
            {"reached_count": 1, "hit_count": 0, "accuracy_when_reached": 0.0},
        )
        comparison = entry["comparisons"][0]
        self.assertTrue(comparison["actor_hit"])
        self.assertTrue(comparison["event_hit"])
        self.assertFalse(comparison["causal_direction_hit"])
        self.assertFalse(comparison["semantic_proposition_hit"])
        self.assertFalse(comparison["matched"])
        self.assertFalse(comparison["exact"])
        self.assertEqual(comparison["first_failed_transformation"], "causal_direction")
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["proposition_recall"], 0.0)
        self.assertEqual(metrics["exact_source_proposition_recall"], 0.0)
        self.assertEqual(metrics["unsupported_proposition_rate"], 1.0)
        self.assertEqual(metrics["slot_recall"]["causal_direction"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["semantic_proposition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["assembled_proposition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})
        self.assertEqual(metrics["slot_recall"]["exact_decomposition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})

    def test_stage_a_proposition_not_deterministic_from_frame_is_suppressed(self) -> None:
        sentence = "Flay prevents Tristana from staying on target after entry."
        source = sentence + " " + sentence
        with sqlite3.connect(self.temp.name) as conn:
            conn.execute("INSERT INTO insights VALUES ('11', 'v1', ?)", (source,))
        case = ({"id": "repeat", "insight_id": "11", "source_video_id": "v1", "eligible": True,
                 "expected_propositions": [self.cases[0]["expected_propositions"][0]]},)

        def slot_at(role: str, phrase: str, start: int) -> SemanticSlot:
            return SemanticSlot(role, SourceAlignment("insight", start, start + len(phrase), phrase))

        def second(phrase: str) -> int:
            return source.index(phrase, source.index(phrase) + len(phrase))

        evidence = SourceAlignment("insight", 0, len(source), source)
        frame = SourceSemanticFrame(
            (evidence,),
            slot_at("actor", "Flay", source.index("Flay")),
            slot_at("event", "prevents", source.index("prevents")),
            slot_at("effect", "staying on target", source.index("staying on target")),
            slot_at("condition", "after entry", source.index("after entry")),
            "actor_event_causes_effect", OntologyNormalization(None, None, None),
        )
        fabricated = ExtractedProposition(
            GroundedProposition("Flay", "prevents", "staying on target", "after entry", ("11",)),
            tuple(
                PropositionAlignment(field, "insight", second(phrase), second(phrase) + len(phrase), phrase)
                for field, phrase in (("subject", "Flay"), ("predicate", "prevents"), ("effect", "staying on target"), ("condition", "after entry"))
            ),
        )
        extraction = StageAExtraction(
            (fabricated,), (frame,), (), evidence_spans=(evidence,),
            slots={
                "actor": slot_at("actor", "Flay", source.index("Flay")),
                "event": slot_at("event", "prevents", source.index("prevents")),
                "effect": slot_at("effect", "staying on target", source.index("staying on target")),
                "condition": slot_at("condition", "after entry", source.index("after entry")),
            },
            causal_direction="actor_event_causes_effect",
        )
        result = evaluate_source_modes(
            case, resolver=self.resolver, extractor=lambda packet: extraction, modes=("insight",),
        )
        entry = result["cases"][0]["modes"][0]
        scores = entry["slot_scores"]
        self.assertFalse(entry["coherence"]["coherent"])
        self.assertEqual(entry["coherence"]["assembly_consistent"], False)
        self.assertEqual(entry["matched_count"], 0)
        self.assertEqual(entry["exact_matched_count"], 0)
        self.assertEqual(entry["predicted_count"], 1)
        self.assertEqual(entry["false_positive_count"], 1)
        self.assertEqual(entry["missed_count"], 1)
        for slot_name in ("actor", "event", "effect", "condition", "causal_direction", "evidence_span", "semantic_proposition"):
            self.assertEqual(scores[slot_name], {"hit_count": 1, "expected_count": 1})
        for slot_name in ("assembled_proposition", "exact_decomposition"):
            self.assertEqual(scores[slot_name], {"hit_count": 0, "expected_count": 1})
        comparison = entry["comparisons"][0]
        self.assertFalse(comparison["matched"])
        self.assertTrue(comparison["semantic_proposition_hit"])
        metrics = result["metrics"]["insight"]
        self.assertEqual(metrics["proposition_recall"], 0.0)
        self.assertEqual(metrics["unsupported_proposition_rate"], 1.0)
        self.assertEqual(metrics["slot_recall"]["semantic_proposition"], {"hit_count": 1, "denominator": 1, "recall": 1.0})
        self.assertEqual(metrics["slot_recall"]["assembled_proposition"], {"hit_count": 0, "denominator": 1, "recall": 0.0})

    def test_stage_a_span_mismatch_reports_coherence_break(self) -> None:
        source = "Flay prevents Tristana from staying on target after entry."
        def slot(role: str, phrase: str) -> SemanticSlot:
            start = source.index(phrase)
            return SemanticSlot(role, SourceAlignment("insight", start, start + len(phrase), phrase))
        full_span = SourceAlignment("insight", 0, len(source), source)
        partial_span = SourceAlignment("insight", 0, len("Flay prevents Tristana"), "Flay prevents Tristana")
        frame = SourceSemanticFrame(
            (full_span,), slot("actor", "Flay"), slot("event", "prevents"),
            slot("effect", "staying on target"), slot("condition", "after entry"),
            "actor_event_causes_effect", OntologyNormalization(None, None, None),
        )
        extraction = StageAExtraction(
            (), (frame,), (), evidence_spans=(partial_span,),
            slots={
                "actor": slot("actor", "Flay"), "event": slot("event", "prevents"),
                "effect": slot("effect", "staying on target"), "condition": slot("condition", "after entry"),
            },
            causal_direction="actor_event_causes_effect",
        )
        result = evaluate_source_modes(
            self.cases[:1], resolver=self.resolver,
            extractor=lambda packet: extraction, modes=("insight",),
        )
        entry = result["cases"][0]["modes"][0]
        self.assertFalse(entry["coherence"]["coherent"])
        self.assertFalse(entry["coherence"]["grounded"])
        for slot_name in ("evidence_span", "actor", "event", "effect", "condition", "causal_direction", "semantic_proposition"):
            self.assertEqual(entry["slot_scores"][slot_name], {"hit_count": 0, "expected_count": 1})

    def test_requires_exactly_one_expected_proposition_per_eligible_case(self) -> None:
        label = {
            "subject_source": "Flay", "predicate_source": "prevents", "effect_source": "staying on target",
            "semantic_field_token_groups": {"subject": [["Flay"]], "predicate": [["prevent", "prevents"]], "effect": [["staying"]]},
            "expected_normalization": {"actor_concept": None, "event_relation": None, "effect_concept": None},
            "normalization_rationale": "Synthetic contract label.",
        }
        fixture = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        fixture.close()
        try:
            Path(fixture.name).write_text(json.dumps({"cases": [{
                "id": "multi", "insight_id": "dev-contract-1", "source_video_id": "dev-contract-v1",
                "eligible": True, "expected_propositions": [label, label],
            }]}))
            with self.assertRaisesRegex(ValueError, "exactly one expected proposition"):
                load_development_cases(fixture.name)
            Path(fixture.name).write_text(json.dumps({"cases": [{
                "id": "none", "insight_id": "dev-contract-1", "source_video_id": "dev-contract-v1",
                "eligible": True, "expected_propositions": [],
            }]}))
            with self.assertRaisesRegex(ValueError, "exactly one expected proposition"):
                load_development_cases(fixture.name)
            Path(fixture.name).write_text(json.dumps({"cases": [{
                "id": "one", "insight_id": "dev-contract-1", "source_video_id": "dev-contract-v1",
                "eligible": True, "expected_propositions": [label],
            }]}))
            self.assertEqual(len(load_development_cases(fixture.name)), 1)
        finally:
            Path(fixture.name).unlink()

    def test_development_fixture_satisfies_five_case_x_of_5_contract(self) -> None:
        dev = Path(__file__).resolve().parent.parent / "data" / "relation_extraction_phase2d_dev_v0.json"
        cases = load_development_cases(dev)
        eligible = [case for case in cases if case["eligible"]]
        ineligible = [case for case in cases if not case["eligible"]]
        self.assertEqual(len(eligible), 5)
        self.assertTrue(all(len(case["expected_propositions"]) == 1 for case in eligible))
        self.assertEqual(len(ineligible), 2)
        self.assertTrue(all(case["expected_propositions"] == [] for case in ineligible))

    def test_rejects_malformed_held_out_structures(self) -> None:
        dev = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        dev.write(json.dumps({"cases": [{
            "id": "ok", "insight_id": "dev-held-1", "source_video_id": "dev-held-v1",
            "eligible": False, "expected_propositions": [],
        }]}))
        dev.close()
        held = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        held.close()
        malformed = (
            ("[]", "top-level"),
            ('"text"', "top-level"),
            ('{"cases":"not-a-list"}', "cases"),
            ('{"cases":[]}', "cases"),
            ('{"cases":[1]}', "each case"),
            ('{"cases":[{}]}', "evidence"),
            ('{"cases":[{"evidence":"not-a-list"}]}', "evidence"),
            ('{"cases":[{"evidence":[]}]}', "evidence"),
            ('{"cases":[{"evidence":[1]}]}', "evidence item"),
            ('{"cases":[{"evidence":[{"source_id":"v1"}]}]}', "insight_id"),
            ('{"cases":[{"evidence":[{"insight_id":"1"}]}]}', "source_id"),
            ('{"cases":[{"evidence":[{"insight_id":"","source_id":"v1"}]}]}', "insight_id"),
            ('{"cases":[{"evidence":[{"insight_id":"1","source_id":""}]}]}', "source_id"),
            ('{"cases":[{"evidence":[{"insight_id":7,"source_id":"v1"}]}]}', "insight_id"),
        )
        try:
            for payload, message in malformed:
                with self.subTest(payload=payload):
                    Path(held.name).write_text(payload)
                    with self.assertRaisesRegex(ValueError, message):
                        load_development_cases(dev.name, held_out_path=held.name)
        finally:
            Path(dev.name).unlink()
            Path(held.name).unlink()



if __name__ == "__main__":
    unittest.main()
