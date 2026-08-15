import unittest
from dataclasses import replace

from scripts.eval_relation_extraction import evaluate_cases, load_cases


class RelationExtractionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases("data/relation_extraction_validation_v0.json")

    def test_reference_cases_are_valid_and_score_perfect_when_replayed(self):
        from pipeline.relation_extract import ExtractionDecision

        result = evaluate_cases(
            self.cases,
            lambda packet: tuple(
                ExtractionDecision({}, expected.relation, "accepted")
                for expected in next(case.expected for case in self.cases if case.packet == packet)
            ),
        )
        self.assertEqual(result["expected_relation_count"], 3)
        self.assertEqual(result["metrics"]["relation_precision"], 1.0)
        self.assertEqual(result["metrics"]["relation_recall"], 1.0)
        self.assertEqual(result["metrics"]["condition_preservation"], 1.0)
        self.assertEqual(result["metrics"]["provenance_correctness"], 1.0)

    def test_condition_loss_is_not_counted_as_semantic_match(self):
        def lose_condition(packet):
            case = next(case for case in self.cases if case.packet == packet)
            if not case.expected:
                return case.expected
            reference = case.expected[0].relation
            from pipeline.relation_extract import ExtractionDecision
            return (ExtractionDecision(
                {}, replace(reference, condition=None), "accepted"),
            )

        result = evaluate_cases(self.cases, lose_condition)
        self.assertLess(result["metrics"]["relation_recall"], 1.0)
        self.assertLess(result["metrics"]["condition_preservation"], 1.0)

    def test_condition_cues_allow_equivalent_prose_but_require_qualifiers(self):
        from scripts.eval_relation_extraction import _condition_matches

        self.assertTrue(_condition_matches("Time it at the apex of the dash", ("apex", "dash")))
        self.assertFalse(_condition_matches("Time it at the apex", ("apex", "dash")))
        self.assertFalse(_condition_matches("when available", ("e",)))
        self.assertTrue(_condition_matches("after E misses", ("e", "miss")))

    def test_phase_2b_labeled_set_has_23_real_source_cases(self):
        cases = load_cases("data/relation_extraction_phase2b_v0.json")

        self.assertEqual(len(cases), 23)
        self.assertEqual(sum(len(case.expected) for case in cases), 18)

    def test_runtime_packet_coverage_warns_for_unexposed_expected_ability(self):
        from scripts.eval_relation_extraction import _unresolved_reference_entities

        cases = load_cases("data/relation_extraction_phase2b_v0.json")
        lux = next(case for case in cases if case.id == "lux-e-denies-farming-access")
        self.assertIn(
            "runtime packet does not expose ability alias: Lux E",
            _unresolved_reference_entities(list(lux.expected), replace(lux.packet, ability_aliases={})),
        )

    def test_same_triple_with_distinct_conditions_matches_the_compatible_reference(self):
        from pipeline.relation_extract import ExtractionDecision

        cases = load_cases("data/relation_extraction_phase2b_v0.json")
        tristana = next(case for case in cases if case.id == "tristana-access-conditions")
        second = tristana.expected[1].relation
        result = evaluate_cases(
            (tristana,), lambda _: (ExtractionDecision({}, second, "accepted"),),
        )

        self.assertEqual(result["metrics"]["true_positive"], 1)
        self.assertEqual(result["metrics"]["false_negative"], 1)

    def test_trace_failure_is_reported_without_aborting_other_cases(self):
        from pipeline.relation_extract import ExtractionTrace

        result = evaluate_cases(self.cases, lambda _: ExtractionTrace("bad", failure_stage="parsing", failure_type="ValueError", failure_message="bad JSON"))
        self.assertEqual(len(result["cases"]), len(self.cases))
        self.assertTrue(all(case["failure"]["stage"] == "parsing" for case in result["cases"]))
        self.assertEqual(result["failure_attribution"]["failure_stages"]["parsing"], len(self.cases))

    def test_false_negative_and_review_metrics_are_reported(self):
        from pipeline.relation_extract import ExtractionDecision

        def review_first_relation(packet):
            case = next(case for case in self.cases if case.packet == packet)
            return tuple(
                ExtractionDecision({}, expected.relation, "review", ("confidence below threshold 0.60",))
                for expected in case.expected
            )

        result = evaluate_cases(self.cases, review_first_relation)
        self.assertEqual(result["metrics"]["true_positive"], 0)
        self.assertEqual(result["metrics"]["false_negative"], 3)
        self.assertEqual(result["metrics"]["review_matches"], 3)
        self.assertEqual(result["failure_attribution"]["decision_reasons"]["confidence below threshold 0.60"], 3)

    def test_live_evaluation_requires_source_database(self):
        from unittest import mock
        from scripts import eval_relation_extraction

        with mock.patch("sys.argv", ["eval_relation_extraction", "--live"]), self.assertRaises(SystemExit):
            eval_relation_extraction.main()

    def test_unknown_case_id_is_rejected(self):
        from unittest import mock
        from scripts import eval_relation_extraction

        with mock.patch("sys.argv", ["eval_relation_extraction", "--case-id", "missing"]), self.assertRaises(SystemExit):
            eval_relation_extraction.main()

    def test_case_id_selection_is_repeatable_and_preserves_fixture_order(self):
        from unittest import mock
        from scripts import eval_relation_extraction

        selected = (self.cases[2].id, self.cases[0].id)
        with mock.patch("sys.argv", ["eval_relation_extraction", "--live", "--db", "videos.db", "--case-id", selected[0], "--case-id", selected[1]]), \
             mock.patch.object(eval_relation_extraction, "load_cases", return_value=self.cases), \
             mock.patch.object(eval_relation_extraction, "evaluate_cases", return_value={}) as evaluate:
            eval_relation_extraction.main()

        self.assertEqual(
            tuple(case.id for case in evaluate.call_args.args[0]),
            (self.cases[0].id, self.cases[2].id),
        )

    def test_variant_selection_uses_configured_relation_model(self):
        from unittest import mock
        from scripts import eval_relation_extraction

        with mock.patch("sys.argv", ["eval_relation_extraction", "--live", "--db", "videos.db", "--variant", "pro"]), \
             mock.patch.object(eval_relation_extraction, "BACKEND", "deepseek"), \
             mock.patch.object(eval_relation_extraction, "load_cases", return_value=()), \
             mock.patch.object(eval_relation_extraction, "evaluate_cases", return_value={} ) as evaluate:
            eval_relation_extraction.main()

        extractor = evaluate.call_args.args[1]
        with mock.patch.object(eval_relation_extraction, "extract_relation_trace", return_value=() ) as extract:
            extractor(object())
        self.assertEqual(extract.call_args.kwargs["model"], eval_relation_extraction.RELATION_PRO_MODEL)

    def test_custom_model_is_not_labeled_as_flash_or_pro_benchmark(self):
        from unittest import mock
        from scripts import eval_relation_extraction

        with mock.patch("sys.argv", ["eval_relation_extraction", "--live", "--db", "videos.db", "--model", "other-model"]), \
             mock.patch.object(eval_relation_extraction, "load_cases", return_value=()), \
             mock.patch.object(eval_relation_extraction, "evaluate_cases", return_value={}) as evaluate:
            eval_relation_extraction.main()

        extractor = evaluate.call_args.args[1]
        with mock.patch.object(eval_relation_extraction, "extract_relation_trace", return_value=()) as extract:
            extractor(object())
        self.assertEqual(extract.call_args.kwargs["model"], "other-model")


if __name__ == "__main__":
    unittest.main()
