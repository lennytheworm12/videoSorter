from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from pipeline.phase2d_evaluation import evaluate_source_modes, load_development_cases
from pipeline.proposition_extract import ExtractedProposition, PropositionAlignment, PropositionPacket
from pipeline.relation_extract import GroundedProposition
from pipeline.source_windows import SourceWindowResolver


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
             "expected_propositions": [{"subject_source": "Flay", "predicate_source": "prevents", "effect_source": "staying on target", "condition_source": "after entry"}]},
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


if __name__ == "__main__":
    unittest.main()
