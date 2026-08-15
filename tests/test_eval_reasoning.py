import unittest
from unittest import mock

from scripts import eval_reasoning


EVIDENCE = [{
    "text": "Keep a range advantage until the opponent spends a key spell.",
    "insight_type": "laning_tips",
    "role": "adc",
    "champion": "Caitlyn",
    "source": "discord",
    "source_weight": 1.0,
    "score": 0.8,
    "confidence": 0.8,
}]


class EvalReasoningTests(unittest.TestCase):
    def test_phase_1_cases_cover_required_reasoning_questions(self):
        ids = {case.id for case in eval_reasoning.PHASE_1_CASES}
        self.assertEqual(
            ids,
            {
                "caitlyn-vs-mage",
                "yunara-thresh-vs-tristana-yuumi",
                "kaisa-conditional-access",
                "sylas-hp-joust-second-rotation",
            },
        )
        team_case = next(case for case in eval_reasoning.PHASE_1_CASES if case.mode == "team_matchup")
        self.assertIn(" vs ", team_case.question)

    def test_evaluate_case_uses_one_evidence_snapshot_for_both_variants(self):
        case = eval_reasoning.PHASE_1_CASES[0]
        with mock.patch.object(eval_reasoning.query, "retrieve", return_value=EVIDENCE) as retrieve, mock.patch.object(
            eval_reasoning.query,
            "answer",
            side_effect=["baseline", "structured"],
        ) as answer:
            result = eval_reasoning.evaluate_case(
                case,
                strategic_db_paths=("fixture.db",),
                top_k=3,
                run_model=True,
            )

        retrieve.assert_called_once_with(case.question, champion="Caitlyn", top_k=3)
        self.assertEqual(result["base_evidence"], {"general": EVIDENCE})
        self.assertEqual(result["baseline_answer"], "baseline")
        self.assertEqual(result["structured_answer"], "structured")
        self.assertEqual(answer.call_count, 2)
        self.assertFalse(answer.call_args_list[0].kwargs["include_strategic_context"])
        self.assertTrue(answer.call_args_list[1].kwargs["include_strategic_context"])
        self.assertEqual(answer.call_args_list[0].kwargs["strategic_db_paths"], ("fixture.db",))
        self.assertEqual(answer.call_args_list[1].kwargs["strategic_db_paths"], ("fixture.db",))

    def test_team_case_snapshots_all_four_champions_once(self):
        case = eval_reasoning.PHASE_1_CASES[1]
        with mock.patch.object(
            eval_reasoning.query,
            "_retrieve_team_matchup",
            return_value=({"Yunara": EVIDENCE, "Thresh": EVIDENCE}, {"Tristana": EVIDENCE, "Yuumi": EVIDENCE}),
        ) as retrieve_team, mock.patch.object(
            eval_reasoning.query,
            "answer",
            side_effect=["baseline", "structured"],
        ) as answer:
            result = eval_reasoning.evaluate_case(
                case,
                strategic_db_paths=("fixture.db",),
                top_k=4,
                run_model=True,
            )

        retrieve_team.assert_called_once_with(
            case.question,
            ("Yunara", "Thresh"),
            ("Tristana", "Yuumi"),
            role=None,
            game="lol",
            top_k=4,
        )
        self.assertEqual(
            result["base_evidence"],
            {"allies": {"Yunara": EVIDENCE, "Thresh": EVIDENCE}, "enemies": {"Tristana": EVIDENCE, "Yuumi": EVIDENCE}},
        )
        self.assertEqual(answer.call_count, 2)
        self.assertFalse(answer.call_args_list[0].kwargs["include_strategic_context"])
        self.assertTrue(answer.call_args_list[1].kwargs["include_strategic_context"])

    def test_dry_run_keeps_answers_empty_and_rendering_comparable(self):
        case = eval_reasoning.PHASE_1_CASES[0]
        with mock.patch.object(eval_reasoning.query, "retrieve", return_value=EVIDENCE):
            result = eval_reasoning.evaluate_case(
                case,
                strategic_db_paths=("fixture.db",),
                run_model=False,
            )

        rendered = eval_reasoning.render_comparison(result)
        self.assertIsNone(result["baseline_answer"])
        self.assertIsNone(result["structured_answer"])
        self.assertIn("### Baseline: RAG evidence only", rendered)
        self.assertIn("### Structured: same RAG evidence + derived strategic context", rendered)
        self.assertIn("shared by both variants", rendered)

    def test_unknown_case_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown evaluation case"):
            eval_reasoning._selected_cases(["missing"])

    def test_non_positive_top_k_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            eval_reasoning.evaluate_case(
                eval_reasoning.PHASE_1_CASES[0],
                strategic_db_paths=("fixture.db",),
                top_k=0,
            )

    def test_team_case_requires_enough_budget_for_each_member(self):
        with self.assertRaisesRegex(ValueError, "at least 4"):
            eval_reasoning.evaluate_case(
                eval_reasoning.PHASE_1_CASES[1],
                strategic_db_paths=("fixture.db",),
                top_k=3,
            )


if __name__ == "__main__":
    unittest.main()
