from __future__ import annotations

import json
import unittest

from pipeline.candidate_generation import generate_candidates
from pipeline.constrained_mapper import mapper_prompt, parse_mapping_selection, select_mapping
from pipeline.relation_extract import GroundedProposition


class ConstrainedMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.proposition = GroundedProposition("Flay", "prevents", "staying on target", "after Tristana jumps", ("1",))
        self.candidates = generate_candidates(self.proposition, ability_aliases={"Flay": "Thresh E"})

    def test_accepts_only_ids_from_matching_candidate_sets(self) -> None:
        raw = json.dumps({"mapping_status":"mapped", "subject_id":"ability:Thresh E", "relation_id":"denies", "object_id":"continuity", "condition_index":0, "confidence":.8})
        result = parse_mapping_selection(raw, self.candidates)
        self.assertEqual(result.status, "mapped")
        self.assertEqual(result.object_id, "continuity")

    def test_supports_unmapped_and_no_relation_without_forced_choice(self) -> None:
        self.assertEqual(parse_mapping_selection('{"mapping_status":"unmapped","subject_id":null,"relation_id":null,"object_id":null,"condition_index":null,"confidence":null}', self.candidates).status, "unmapped")
        self.assertEqual(parse_mapping_selection('{"mapping_status":"no_relation","subject_id":null,"relation_id":null,"object_id":null,"condition_index":null,"confidence":null}', self.candidates).status, "no_relation")

    def test_rejects_freeform_or_wrong_slot_ids(self) -> None:
        payload = {"mapping_status":"mapped", "subject_id":"Thresh E", "relation_id":"denies", "object_id":"continuity", "condition_index":0, "confidence":None}
        with self.assertRaisesRegex(ValueError, "invalid subject"):
            parse_mapping_selection(json.dumps(payload), self.candidates)
        payload["subject_id"] = "ability:Thresh E"
        payload["object_id"] = "unknown_concept"
        with self.assertRaisesRegex(ValueError, "invalid object"):
            parse_mapping_selection(json.dumps(payload), self.candidates)

    def test_rejects_selection_with_unmapped_and_bad_condition(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not select"):
            parse_mapping_selection('{"mapping_status":"unmapped","subject_id":"ability:Thresh E","relation_id":null,"object_id":null,"condition_index":null,"confidence":null}', self.candidates)
        payload = {"mapping_status":"mapped", "subject_id":"ability:Thresh E", "relation_id":"denies", "object_id":"continuity", "condition_index":99, "confidence":None}
        with self.assertRaisesRegex(ValueError, "condition"):
            parse_mapping_selection(json.dumps(payload), self.candidates)

    def test_prompt_and_model_path_expose_candidate_ids_not_ontology_freeform_contract(self) -> None:
        prompt = mapper_prompt(self.proposition, self.candidates)
        self.assertIn("ability:Thresh E", prompt)
        self.assertIn("unmapped", prompt)
        self.assertIn("confidence must be null", prompt)
        raw = '{"mapping_status":"no_relation","subject_id":null,"relation_id":null,"object_id":null,"condition_index":null,"confidence":null}'
        calls = []
        self.assertEqual(select_mapping(self.proposition, self.candidates, lambda **kwargs: calls.append(kwargs) or raw).status, "no_relation")
        self.assertNotIn("canonical key", calls[0]["user"])

    def test_rejects_extra_or_duplicate_fields_and_nonmapped_confidence(self) -> None:
        base = '{"mapping_status":"unmapped","subject_id":null,"relation_id":null,"object_id":null,"condition_index":null,"confidence":null'
        with self.assertRaisesRegex(ValueError, "unknown or missing"):
            parse_mapping_selection(base + ',"freeform_concept":"continuity"}', self.candidates)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_mapping_selection(base + ',"mapping_status":"mapped"}', self.candidates)
        with self.assertRaisesRegex(ValueError, "must not select"):
            parse_mapping_selection(base[:-4] + '0.9}', self.candidates)

    def test_rejects_wrong_slots_and_noninteger_conditions(self) -> None:
        payload = {"mapping_status":"mapped", "subject_id":"continuity", "relation_id":"denies", "object_id":"continuity", "condition_index":0, "confidence":None}
        with self.assertRaisesRegex(ValueError, "invalid subject"):
            parse_mapping_selection(json.dumps(payload), self.candidates)
        payload["subject_id"] = "ability:Thresh E"
        for index in (True, 0.0):
            payload["condition_index"] = index
            with self.assertRaisesRegex(ValueError, "condition"):
                parse_mapping_selection(json.dumps(payload), self.candidates)

    def test_empty_candidates_allow_only_nonselection(self) -> None:
        empty = generate_candidates(GroundedProposition("unknown", "observes", "bananas", None, ("1",)))
        with self.assertRaisesRegex(ValueError, "invalid subject"):
            parse_mapping_selection('{"mapping_status":"mapped","subject_id":"x","relation_id":"y","object_id":"z","condition_index":null,"confidence":null}', empty)


if __name__ == "__main__":
    unittest.main()
