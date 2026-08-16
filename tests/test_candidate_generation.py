from __future__ import annotations

import unittest

from pipeline.candidate_generation import generate_candidates
from pipeline.relation_extract import GroundedProposition


def _proposition(subject="Flay", predicate="prevents", effect="staying on target", condition="after Tristana jumps"):
    return GroundedProposition(subject, predicate, effect, condition, ("1",))


class CandidateGenerationTests(unittest.TestCase):
    def test_generates_legal_alias_candidates_with_reasons(self) -> None:
        result = generate_candidates(_proposition(), ability_aliases={"Flay": "Thresh E"})
        self.assertEqual(result.subject[0].id, "ability:Thresh E")
        self.assertEqual(result.relation[0].id, "denies")
        self.assertEqual(result.object[0].id, "continuity")
        self.assertEqual(result.object[0].reason, "semantic_alias:staying on target")

    def test_preserves_multiple_relation_candidates_without_new_verbs(self) -> None:
        result = generate_candidates(_proposition(predicate="prevents"))
        self.assertEqual([item.id for item in result.relation], ["denies", "reduces", "increases_cost_of"])

    def test_unknown_subject_is_not_invented(self) -> None:
        result = generate_candidates(_proposition(subject="mystery mechanism"), ability_aliases={"E": "Lux E", "W": "Lux W"})
        self.assertEqual(result.subject, ())

    def test_generic_capability_words_do_not_imply_relation_direction(self) -> None:
        self.assertEqual(generate_candidates(_proposition(predicate="cannot")).relation, ())
        self.assertEqual(generate_candidates(_proposition(predicate="let")).relation, ())
        self.assertEqual(generate_candidates(_proposition(predicate="does not prevent")).relation, ())
        self.assertEqual(generate_candidates(_proposition(predicate="does not enable")).relation, ())
        self.assertEqual(generate_candidates(_proposition(predicate="cannot prevent")).relation, ())
        self.assertEqual(generate_candidates(_proposition(predicate="doesn't prevent")).relation, ())
        self.assertEqual(generate_candidates(_proposition(predicate="is not able to prevent")).relation, ())
        self.assertEqual(generate_candidates(_proposition(predicate="it is never able to enable")).relation, ())
        self.assertEqual(generate_candidates(_proposition(predicate="didn't prevent")).relation, ())
        self.assertEqual(generate_candidates(_proposition(predicate="don't enable")).relation, ())
        self.assertEqual(generate_candidates(_proposition(predicate="hasn't prevented")).relation, ())
        self.assertEqual(generate_candidates(_proposition(predicate="fails to prevent")).relation, ())

    def test_condition_preserves_source_and_normalizes_only_known_miss_event(self) -> None:
        result = generate_candidates(_proposition(condition="after Lux Q misses"), ability_aliases={"Lux Q": "Lux Q"})
        self.assertEqual(result.condition[0].source_text, "after Lux Q misses")
        self.assertEqual(result.condition[0].event, "missed")
        self.assertEqual(result.condition[0].derived_state, "temporarily_unavailable")
        unknown = generate_candidates(_proposition(condition="after enemy misses"), ability_aliases={"E": "Lux E"})
        self.assertIsNone(unknown.condition[0].event)

    def test_unmappable_effect_has_no_arbitrary_concept_candidates(self) -> None:
        result = generate_candidates(_proposition(effect="bananas orchestra", predicate="observes"))
        self.assertEqual(result.object, ())
        self.assertEqual(result.relation, ())

    def test_top_k_is_bounded(self) -> None:
        result = generate_candidates(_proposition(effect="access territory tempo conversion"), top_k_concepts=2)
        self.assertLessEqual(len(result.object), 2)
        with self.assertRaisesRegex(ValueError, "top_k"):
            generate_candidates(_proposition(), top_k_concepts=0)
