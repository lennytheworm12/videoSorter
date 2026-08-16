from __future__ import annotations

import unittest

from pipeline.candidate_generation import generate_candidates
from pipeline.constrained_mapper import MappingSelection
from pipeline.phase2d_metrics import CanonicalReference, candidate_coverage, mapper_result, primary_failure, summarize_cases
from pipeline.relation_extract import GroundedProposition


class Phase2DMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.proposition = GroundedProposition("Flay", "prevents", "staying on target", "after entry", ("e1",))
        self.candidates = generate_candidates(self.proposition, ability_aliases={"Flay": "Thresh E"})
        self.reference = CanonicalReference("ability:Thresh E", "denies", "continuity", True, "after entry")

    def test_separates_candidate_coverage_from_mapper_selection(self) -> None:
        coverage = candidate_coverage(self.candidates, self.reference)
        self.assertTrue(coverage["full_triple"])
        selection = MappingSelection("mapped", "ability:Thresh E", "denies", "continuity", 0, .8)
        self.assertTrue(mapper_result(selection, self.reference, self.candidates)["full_triple"])

    def test_attributes_missing_candidate_before_mapper(self) -> None:
        reference = CanonicalReference("ability:Thresh E", "denies", "unknown_concept", True)
        coverage = candidate_coverage(self.candidates, reference)
        self.assertEqual(primary_failure(coverage, None), "object_candidate_miss")

    def test_attributes_wrong_mapped_id_and_explicit_mapper_failure(self) -> None:
        coverage = candidate_coverage(self.candidates, self.reference)
        wrong = MappingSelection("mapped", "ability:Thresh E", "denies", "access", 0, .8)
        self.assertEqual(primary_failure(coverage, wrong, self.reference, candidates=self.candidates), "mapper_misselection")
        self.assertEqual(primary_failure(coverage, None, mapper_failure="timeout"), "other:timeout")

    def test_correct_conditional_mapping_does_not_report_a_failure(self) -> None:
        coverage = candidate_coverage(self.candidates, self.reference)
        correct = MappingSelection("mapped", "ability:Thresh E", "denies", "continuity", 0, .8)
        self.assertIsNone(primary_failure(coverage, correct, self.reference, candidates=self.candidates))

    def test_reports_conditional_mapper_accuracy_without_dividing_by_zero(self) -> None:
        coverage = candidate_coverage(self.candidates, self.reference)
        correct = mapper_result(MappingSelection("mapped", "ability:Thresh E", "denies", "continuity", 0, .8), self.reference, self.candidates)
        missing = {**coverage, "full_triple": False}
        result = summarize_cases(((coverage, correct), (missing, None)))
        self.assertEqual(result["full_triple_candidate_coverage"], .5)
        self.assertEqual(result["mapper_accuracy_given_candidate_coverage"], 1.0)

    def test_missing_mapper_output_counts_against_overall_accuracy(self) -> None:
        coverage = candidate_coverage(self.candidates, self.reference)
        correct = mapper_result(MappingSelection("mapped", "ability:Thresh E", "denies", "continuity", 0, .8), self.reference, self.candidates)
        result = summarize_cases(((coverage, correct), (coverage, None)))
        self.assertEqual(result["end_to_end_mapping_success_rate"], .5)
        self.assertEqual(result["mapper_accuracy_given_candidate_coverage"], .5)

    def test_requires_the_expected_condition_candidate_and_selection(self) -> None:
        wrong = CanonicalReference("ability:Thresh E", "denies", "continuity", True, "while held")
        coverage = candidate_coverage(self.candidates, wrong)
        self.assertFalse(coverage["condition"])
        selection = MappingSelection("mapped", "ability:Thresh E", "denies", "continuity", 0, .8)
        self.assertFalse(mapper_result(selection, wrong, self.candidates)["condition"])
        malformed = MappingSelection("mapped", "ability:Thresh E", "denies", "continuity", -1, .8)
        self.assertFalse(mapper_result(malformed, self.reference, self.candidates)["condition"])

    def test_rejects_unknown_mapper_failure_taxonomy(self) -> None:
        coverage = candidate_coverage(self.candidates, self.reference)
        with self.assertRaisesRegex(ValueError, "unknown mapper failure"):
            primary_failure(coverage, None, mapper_failure="whatever")


if __name__ == "__main__":
    unittest.main()
