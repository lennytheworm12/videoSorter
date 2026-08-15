import unittest
from unittest import mock

import retrieval.query as query
from retrieval.strategic_context import StrategicContext


EVIDENCE = [{
    "text": "Keep peel available until the enemy commits.",
    "insight_type": "laning_tips",
    "role": "support",
    "champion": "Thresh",
    "source": "discord",
    "source_weight": 1.0,
    "score": 0.8,
    "confidence": 0.8,
}]


class TeamMatchupTests(unittest.TestCase):
    def test_detect_intent_preserves_two_champion_teams_around_vs(self):
        lookup = {
            "yunara": "Yunara",
            "thresh": "Thresh",
            "tristana": "Tristana",
            "yuumi": "Yuumi",
        }
        with mock.patch.object(query, "_get_champion_lookup", return_value=lookup):
            intent = query.detect_intent("Yunara and Thresh vs Tristana and Yuumi")

        self.assertEqual(
            intent,
            {
                "type": "team_matchup",
                "allies": ("Yunara", "Thresh"),
                "enemies": ("Tristana", "Yuumi"),
            },
        )

    def test_public_answer_routes_the_representative_question_to_team_matchup(self):
        lookup = {
            "yunara": "Yunara",
            "thresh": "Thresh",
            "tristana": "Tristana",
            "yuumi": "Yuumi",
        }
        question = "How should Yunara and Thresh play vs Tristana and Yuumi through access versus continuity?"
        with mock.patch.object(query, "_get_champion_lookup", return_value=lookup), mock.patch.object(
            query,
            "_answer_team_matchup",
            return_value="team answer",
        ) as answer_team:
            result = query.answer(question, show_sources=False)

        self.assertEqual(result, "team answer")
        self.assertEqual(answer_team.call_args.args[1]["type"], "team_matchup")

    def test_team_answer_keeps_teams_separate_and_passes_all_entities_to_context(self):
        intent = {
            "type": "team_matchup",
            "allies": ("Yunara", "Thresh"),
            "enemies": ("Tristana", "Yuumi"),
        }
        teams = ({"Yunara": EVIDENCE, "Thresh": EVIDENCE}, {"Tristana": EVIDENCE, "Yuumi": EVIDENCE})
        with mock.patch.object(query, "_retrieve_team_matchup", return_value=teams), mock.patch.object(
            query,
            "build_strategic_context",
            return_value=StrategicContext(),
        ) as strategic, mock.patch.object(query, "_ability_reference_block", return_value=""), mock.patch.object(
            query,
            "llm_chat",
            return_value="answer",
        ) as chat:
            result = query._answer_team_matchup(
                "Yunara and Thresh vs Tristana and Yuumi",
                intent,
                role=None,
                game="lol",
                top_k=8,
                show_sources=False,
                include_strategic_context=True,
            )

        self.assertEqual(result, "answer")
        strategic.assert_called_once_with(
            "Yunara and Thresh vs Tristana and Yuumi",
            ("Yunara", "Thresh", "Tristana", "Yuumi"),
            db_paths=None,
        )
        user = chat.call_args.kwargs["user"]
        self.assertIn("=== Allied pair: Yunara / Thresh ===", user)
        self.assertIn("=== Enemy pair: Tristana / Yuumi ===", user)
        self.assertIn(EVIDENCE[0]["text"], user)

    def test_team_retrieval_rejects_oversized_groups(self):
        with self.assertRaisesRegex(ValueError, "exactly two champions"):
            query._retrieve_team_matchup(
                "three versus two",
                ("Yunara", "Thresh", "Caitlyn"),
                ("Tristana", "Yuumi"),
                role=None,
                game="lol",
                top_k=8,
            )

    def test_team_retrieval_requires_global_budget_for_all_four_members(self):
        with self.assertRaisesRegex(ValueError, "at least 4"):
            query._retrieve_team_matchup(
                "two versus two",
                ("Yunara", "Thresh"),
                ("Tristana", "Yuumi"),
                role=None,
                game="lol",
                top_k=3,
            )

    def test_detect_intent_does_not_route_oversized_groups_to_team_matchup(self):
        lookup = {
            "yunara": "Yunara",
            "thresh": "Thresh",
            "caitlyn": "Caitlyn",
            "tristana": "Tristana",
            "yuumi": "Yuumi",
        }
        with mock.patch.object(query, "_get_champion_lookup", return_value=lookup):
            intent = query.detect_intent("Yunara, Thresh, and Caitlyn vs Tristana and Yuumi")

        self.assertNotEqual(intent["type"], "team_matchup")


if __name__ == "__main__":
    unittest.main()
