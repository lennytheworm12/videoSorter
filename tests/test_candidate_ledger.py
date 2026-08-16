from __future__ import annotations

import unittest

from pipeline.candidate_ledger import CandidateLedger, LedgerStatus
from pipeline.candidate_generation import generate_candidates
from pipeline.candidate_generation import CandidateSet, CanonicalCandidate
from pipeline.constrained_mapper import MappingSelection
from pipeline.relation_extract import GroundedProposition


def _prop(condition="after entry", evidence="e1", predicate="prevents"):
    return GroundedProposition("Flay", predicate, "staying on target", condition, (evidence,))


def _mapped(relation="denies", confidence=.9):
    return MappingSelection("mapped", "ability:Thresh E", relation, "continuity", 0, confidence)

def _candidates(prop, relation="denies"):
    return generate_candidates(prop, ability_aliases={"Flay": "Thresh E"})

def _ledger(**sources):
    return CandidateLedger(evidence_sources=sources or {"e1": "v1"}, ability_aliases={"Flay": "Thresh E"})


class CandidateLedgerTests(unittest.TestCase):
    def test_requires_authoritative_evidence_catalog(self) -> None:
        with self.assertRaisesRegex(ValueError, "immutable evidence source catalog"):
            CandidateLedger(evidence_sources={}, ability_aliases={"Flay": "Thresh E"})

    def test_unmapped_is_preserved_and_never_trusted(self) -> None:
        ledger = _ledger(e1="v1", e2="v2")
        prop = _prop()
        entry = ledger.record(prop, MappingSelection("unmapped"), evidence_id="e1", source_video_id="v1", candidates=_candidates(prop))
        self.assertEqual(entry.status, LedgerStatus.PROVISIONAL_UNMAPPED)
        self.assertEqual(entry.evidence_ids, ("e1",))
        no_prop = _prop(condition=None, evidence="e2")
        no_condition = ledger.record(no_prop, MappingSelection("unmapped"), evidence_id="e2", source_video_id="v2", candidates=_candidates(no_prop))
        self.assertEqual(no_condition.status, LedgerStatus.PROVISIONAL_UNMAPPED)

    def test_independent_video_support_promotes_but_repeat_model_sample_does_not(self) -> None:
        ledger = _ledger(e1="v1", e2="v2")
        prop = _prop()
        entry = ledger.record(prop, _mapped(), evidence_id="e1", source_video_id="v1", candidates=_candidates(prop), model_id="flash")
        ledger.record(prop, _mapped(), evidence_id="e1", source_video_id="v1", candidates=_candidates(prop), model_id="pro")
        self.assertEqual(entry.independent_video_count, 1)
        self.assertEqual(entry.status, LedgerStatus.PROVISIONAL_MAPPED)
        prop2 = _prop(evidence="e2"); ledger.record(prop2, _mapped(), evidence_id="e2", source_video_id="v2", candidates=_candidates(prop2))
        self.assertEqual(entry.status, LedgerStatus.TRUSTED)
        self.assertEqual(entry.evidence_ids, ("e1", "e2"))

    def test_low_confidence_and_distinct_conditions_do_not_promote_or_contradict(self) -> None:
        ledger = _ledger(e1="v1", e2="v2", e3="v3")
        first = _prop(); second = _prop(evidence="e2")
        low = ledger.record(first, _mapped(confidence=.5), evidence_id="e1", source_video_id="v1", candidates=_candidates(first))
        ledger.record(second, _mapped(confidence=.5), evidence_id="e2", source_video_id="v2", candidates=_candidates(second))
        self.assertEqual(low.status, LedgerStatus.PROVISIONAL_MAPPED)
        third = _prop("while held", "e3", "creates"); other = ledger.record(third, _mapped("creates"), evidence_id="e3", source_video_id="v3", candidates=_candidates(third))
        self.assertNotEqual(other.status, LedgerStatus.CONTRADICTED)

    def test_same_condition_conflicting_relations_are_preserved_as_contradicted(self) -> None:
        ledger = _ledger(e1="v1", e2="v2")
        one = _prop(); two = _prop(evidence="e2", predicate="creates")
        first = ledger.record(one, _mapped("denies"), evidence_id="e1", source_video_id="v1", candidates=_candidates(one))
        second = ledger.record(two, _mapped("creates"), evidence_id="e2", source_video_id="v2", candidates=_candidates(two))
        self.assertEqual(first.status, LedgerStatus.CONTRADICTED)
        self.assertEqual(second.status, LedgerStatus.CONTRADICTED)

    def test_no_relation_is_not_stored_and_provenance_is_required(self) -> None:
        ledger = _ledger(e1="v1", e2="v2")
        prop = _prop()
        outcome = ledger.record(prop, MappingSelection("no_relation"), evidence_id="e1", source_video_id="v1", candidates=_candidates(prop))
        self.assertEqual(outcome.status, LedgerStatus.NO_RELATION)
        duplicate = ledger.record(prop, MappingSelection("no_relation"), evidence_id="e1", source_video_id="v1", candidates=_candidates(prop))
        self.assertEqual(len(duplicate.evidence), 1)
        other = _prop("while held", "e2")
        other_outcome = ledger.record(other, MappingSelection("no_relation"), evidence_id="e2", source_video_id="v2", candidates=_candidates(other))
        self.assertNotEqual(outcome.key(), other_outcome.key())
        with self.assertRaisesRegex(ValueError, "provenanced"):
            ledger.record(prop, MappingSelection("unmapped"), evidence_id="wrong", source_video_id="v1", candidates=_candidates(prop))

    def test_rejected_is_inspectable_and_never_promoted(self) -> None:
        ledger = CandidateLedger(evidence_sources={"e1": "v1"}, ability_aliases={"Flay": "Thresh E"}, trusted_min_independent_videos=1)
        prop = _prop()
        rejected = ledger.record_rejected(prop, evidence_id="e1", source_video_id="v1", reason="invalid mapper ID")
        self.assertEqual(rejected.status, LedgerStatus.REJECTED)
        self.assertEqual(rejected.rejection_reason, "invalid mapper ID")
        self.assertEqual(rejected.evidence_ids, ("e1",))

    def test_rejects_forged_source_diversity_and_freeform_selection(self) -> None:
        ledger = _ledger(e1="v1"); prop = _prop(); candidates = _candidates(prop)
        with self.assertRaisesRegex(ValueError, "immutable"):
            ledger.record(prop, _mapped(), evidence_id="e1", source_video_id="v2", candidates=candidates)
        with self.assertRaisesRegex(ValueError, "candidate set"):
            ledger.record(prop, MappingSelection("mapped", "free", "free", "free", None, .9), evidence_id="e1", source_video_id="v1", candidates=candidates)

    def test_rejects_invalid_direct_status_and_confidence(self) -> None:
        ledger = _ledger(e1="v1"); prop = _prop(); candidates = _candidates(prop)
        with self.assertRaisesRegex(ValueError, "invalid confidence"):
            ledger.record(prop, _mapped(confidence=2.0), evidence_id="e1", source_video_id="v1", candidates=candidates)
        with self.assertRaisesRegex(ValueError, "invalid status"):
            ledger.record(prop, MappingSelection("invalid"), evidence_id="e1", source_video_id="v1", candidates=candidates)

    def test_rejects_candidate_set_from_another_proposition(self) -> None:
        ledger = _ledger(e1="v1", e2="v2")
        first, second = _prop(), _prop(evidence="e2", predicate="creates")
        with self.assertRaisesRegex(ValueError, "does not belong"):
            ledger.record(second, _mapped("creates"), evidence_id="e2", source_video_id="v2", candidates=_candidates(first))

    def test_rejects_fabricated_candidate_set_with_matching_signature(self) -> None:
        ledger = _ledger(e1="v1"); prop = _prop(); generated = _candidates(prop)
        forged = CandidateSet(
            proposition_signature=generated.proposition_signature,
            ability_aliases=generated.ability_aliases,
            top_k_concepts=generated.top_k_concepts,
            subject=(CanonicalCandidate("free_subject", 1.0, "fabricated", "ability"),),
            relation=(CanonicalCandidate("free_relation", 1.0, "fabricated"),),
            object=(CanonicalCandidate("free_object", 1.0, "fabricated", "concept"),),
            condition=generated.condition,
        )
        with self.assertRaisesRegex(ValueError, "deterministically generated"):
            ledger.record(prop, MappingSelection("mapped", "free_subject", "free_relation", "free_object", 0, .9), evidence_id="e1", source_video_id="v1", candidates=forged)

    def test_rejects_candidate_policy_injected_by_candidate_set(self) -> None:
        ledger = _ledger(e1="v1"); prop = _prop(); generated = _candidates(prop)
        forged = CandidateSet(
            proposition_signature=generated.proposition_signature,
            ability_aliases=(("Flay", "Invented Ability"),),
            top_k_concepts=generated.top_k_concepts,
            subject=(CanonicalCandidate("ability:Invented Ability", 1.0, "ability_alias", "ability"),),
            relation=generated.relation, object=generated.object, condition=generated.condition,
        )
        with self.assertRaisesRegex(ValueError, "approved generation policy"):
            ledger.record(prop, MappingSelection("mapped", "ability:Invented Ability", "denies", "continuity", 0, .9), evidence_id="e1", source_video_id="v1", candidates=forged)


if __name__ == "__main__":
    unittest.main()
