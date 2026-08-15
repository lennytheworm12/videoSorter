import unittest

from scripts.eval_relation_extraction import evaluate_cases, load_cases


class RelationExtractionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases("data/relation_extraction_validation_v0.json")

    def test_reference_cases_are_valid_and_score_perfect_when_replayed(self):
        result = evaluate_cases(self.cases, lambda packet: next(case.expected for case in self.cases if case.packet == packet))
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
            relation = case.expected[0].relation
            raw = dict(case.expected[0].raw)
            raw["condition"] = None
            from pipeline.relation_extract import compile_candidates
            return compile_candidates(packet, [raw])

        result = evaluate_cases(self.cases, lose_condition)
        self.assertLess(result["metrics"]["relation_recall"], 1.0)
        self.assertLess(result["metrics"]["condition_preservation"], 1.0)

    def test_trace_failure_is_reported_without_aborting_other_cases(self):
        from pipeline.relation_extract import ExtractionTrace

        result = evaluate_cases(self.cases, lambda _: ExtractionTrace("bad", failure_stage="parsing", failure_type="ValueError", failure_message="bad JSON"))
        self.assertEqual(len(result["cases"]), len(self.cases))
        self.assertTrue(all(case["failure"]["stage"] == "parsing" for case in result["cases"]))

    def test_live_evaluation_requires_source_database(self):
        from unittest import mock
        from scripts import eval_relation_extraction

        with mock.patch("sys.argv", ["eval_relation_extraction", "--live"]), self.assertRaises(SystemExit):
            eval_relation_extraction.main()

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
