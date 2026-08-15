import copy
from pathlib import Path
import unittest

from core.ontology import ONTOLOGY_VERSION, STRATEGIC_CONCEPTS
from core.strategic_types import (
    CURRENT_STRATEGIC_DATA_VERSION,
    EvidenceRef,
    StrategicRelation,
    StrategicValidationError,
    dedupe_relations,
    load_strategic_fixture,
)


FIXTURE_PATH = Path("data/strategic_fixtures_v0.json")


class StrategicTypesTests(unittest.TestCase):
    def test_ontology_v0_contains_design_doc_initial_concepts(self) -> None:
        expected = {
            "access",
            "continuity",
            "range_asymmetry",
            "territory",
            "persistent_pressure",
            "intermittent_pressure",
            "threat_preservation",
            "resource_exchange",
            "combat_compression",
            "combat_expansion",
            "isolation",
            "wave_obligation",
            "initiative",
            "role_transfer",
            "conversion",
            "reset",
            "tempo",
            "local_numbers",
            "default_trajectory",
            "winning_line",
        }

        self.assertEqual(set(STRATEGIC_CONCEPTS), expected)
        self.assertEqual(ONTOLOGY_VERSION, "strategic-ontology-v0")

    def test_fixture_loads_and_preserves_provenance(self) -> None:
        fixture = load_strategic_fixture(FIXTURE_PATH)

        self.assertEqual(fixture.ontology_version, ONTOLOGY_VERSION)
        self.assertEqual(fixture.data_version, CURRENT_STRATEGIC_DATA_VERSION)
        self.assertGreaterEqual(len(fixture.fingerprints), 6)
        self.assertGreaterEqual(len(fixture.relations), 18)
        self.assertGreaterEqual(len(fixture.principles), 5)
        self.assertTrue(all(fp.evidence_refs for fp in fixture.fingerprints))
        self.assertTrue(all(rel.evidence_refs for rel in fixture.relations))
        self.assertTrue(all(principle.evidence_refs for principle in fixture.principles))

    def test_fixture_includes_required_phase1_entities(self) -> None:
        fixture = load_strategic_fixture(FIXTURE_PATH)
        champions = {fingerprint.champion for fingerprint in fixture.fingerprints}
        relation_subjects = {
            relation.subject_key.lower() for relation in fixture.relations
        }

        for champion in {"Caitlyn", "Kai'Sa", "Yunara", "Tristana", "Thresh", "Sylas"}:
            self.assertIn(champion, champions)
        self.assertIn("artillery_mage", relation_subjects)

    def test_invalid_relation_rejects_unknown_concepts(self) -> None:
        relation_data = _valid_relation_data()
        relation_data["concepts"] = ["nonexistent_concept"]

        with self.assertRaisesRegex(StrategicValidationError, "unknown concept"):
            StrategicRelation.from_dict(relation_data)

    def test_invalid_relation_rejects_unknown_relation_type(self) -> None:
        relation_data = _valid_relation_data()
        relation_data["relation_type"] = "sort_of_helps"

        with self.assertRaisesRegex(StrategicValidationError, "unknown relation_type"):
            StrategicRelation.from_dict(relation_data)

    def test_invalid_relation_rejects_missing_provenance(self) -> None:
        relation_data = _valid_relation_data()
        relation_data["evidence_refs"] = []

        with self.assertRaisesRegex(StrategicValidationError, "must include evidence_refs"):
            StrategicRelation.from_dict(relation_data)

    def test_stale_relation_version_is_rejected(self) -> None:
        relation_data = _valid_relation_data()
        relation_data["data_version"] = "strategic-fixtures-v99"

        with self.assertRaisesRegex(StrategicValidationError, "unsupported strategic data version"):
            StrategicRelation.from_dict(relation_data)

    def test_non_numeric_confidence_raises_domain_error(self) -> None:
        relation_data = _valid_relation_data()
        relation_data["confidence"] = "not-a-number"

        with self.assertRaisesRegex(StrategicValidationError, "confidence must be numeric"):
            StrategicRelation.from_dict(relation_data)

    def test_duplicate_evidence_refs_are_rejected(self) -> None:
        relation_data = _valid_relation_data()
        relation_data["evidence_refs"].append(copy.deepcopy(relation_data["evidence_refs"][0]))

        with self.assertRaisesRegex(StrategicValidationError, "duplicate evidence ref"):
            StrategicRelation.from_dict(relation_data)

    def test_confidence_must_be_between_zero_and_one(self) -> None:
        relation_data = _valid_relation_data()
        relation_data["confidence"] = 1.5

        with self.assertRaisesRegex(StrategicValidationError, "confidence"):
            StrategicRelation.from_dict(relation_data)

    def test_duplicate_relations_are_deduped_by_stable_key_and_merge_evidence(self) -> None:
        first = StrategicRelation.from_dict(_valid_relation_data())
        duplicate_data = _valid_relation_data()
        duplicate_data["id"] = "rel-duplicate-id"
        duplicate_data["confidence"] = 0.95
        duplicate_data["evidence_refs"] = [
            {
                "source_type": "manual_analysis",
                "source_id": "phase1-test-extra",
                "insight_id": "thresh-test-extra",
            }
        ]
        duplicate = StrategicRelation.from_dict(duplicate_data)

        deduped = dedupe_relations([first, duplicate])

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].id, first.id)
        self.assertEqual(deduped[0].confidence, 0.95)
        self.assertEqual(len(deduped[0].evidence_refs), 2)

    def test_contradictory_relations_can_coexist_when_conditions_differ(self) -> None:
        hold = StrategicRelation.from_dict(_valid_relation_data())
        spend_data = _valid_relation_data()
        spend_data.update(
            {
                "id": "rel-thresh-flay-can-enable-punish",
                "subject_key": "Thresh Flay",
                "relation_type": "enables",
                "object_key": "conversion",
                "object_type": "concept",
                "condition": "when Tristana cannot punish the cooldown window",
                "effect": "Thresh can spend Flay for immediate lane value",
                "concepts": ["conversion", "resource_exchange"],
            }
        )
        spend = StrategicRelation.from_dict(spend_data)

        deduped = dedupe_relations([hold, spend])

        self.assertEqual(len(deduped), 2)

    def test_stale_fingerprint_version_is_rejected(self) -> None:
        fixture = load_strategic_fixture(FIXTURE_PATH)
        data = {
            "champion": fixture.fingerprints[0].champion,
            "preferred_states": list(fixture.fingerprints[0].preferred_states),
            "evidence_refs": [
                {
                    "source_type": ref.source_type,
                    "source_id": ref.source_id,
                    "insight_id": ref.insight_id,
                }
                for ref in fixture.fingerprints[0].evidence_refs
            ],
            "confidence": fixture.fingerprints[0].confidence,
            "data_version": "strategic-fixtures-v99",
        }

        from core.strategic_types import ChampionFingerprint

        with self.assertRaisesRegex(StrategicValidationError, "unsupported strategic data version"):
            ChampionFingerprint.from_dict(data)

    def test_unknown_fingerprint_dependency_is_rejected(self) -> None:
        fixture = load_strategic_fixture(FIXTURE_PATH)
        data = {
            "champion": "Test Champion",
            "dependencies": ["access denial"],
            "evidence_refs": [
                {
                    "source_type": ref.source_type,
                    "source_id": ref.source_id,
                    "insight_id": ref.insight_id,
                }
                for ref in fixture.fingerprints[0].evidence_refs
            ],
            "confidence": 0.7,
        }

        from core.strategic_types import ChampionFingerprint

        with self.assertRaisesRegex(StrategicValidationError, "unknown fingerprint dependency"):
            ChampionFingerprint.from_dict(data)

    def test_duplicate_principle_evidence_refs_are_rejected(self) -> None:
        from core.strategic_types import CompiledPrinciple

        ref = {
            "source_type": "manual_analysis",
            "source_id": "phase1-principle-test",
            "insight_id": "principle-test",
        }
        with self.assertRaisesRegex(StrategicValidationError, "duplicate evidence ref"):
            CompiledPrinciple.from_dict(
                {
                    "id": "principle-test",
                    "title": "Test principle",
                    "summary": "A useful strategic summary.",
                    "concepts": ["access"],
                    "evidence_refs": [ref, copy.deepcopy(ref)],
                    "confidence": 0.7,
                }
            )


def _valid_relation_data() -> dict:
    return {
        "id": "rel-thresh-flay-denies-continuity-test",
        "subject_type": "ability",
        "subject_key": "Thresh Flay",
        "relation_type": "denies",
        "object_type": "concept",
        "object_key": "continuity",
        "condition": "when held for committed enemy access",
        "effect": "breaks contact before damage converts",
        "confidence": 0.9,
        "provenance_type": "manual_fixture",
        "patch_sensitivity": "low",
        "concepts": ["continuity", "threat_preservation"],
        "evidence_refs": [
            {
                "source_type": "manual_analysis",
                "source_id": "phase1-test",
                "insight_id": "thresh-test",
            }
        ],
    }


class EvidenceRefTests(unittest.TestCase):
    def test_evidence_ref_requires_source_type_and_id(self) -> None:
        with self.assertRaisesRegex(StrategicValidationError, "source_type"):
            EvidenceRef.from_dict({"source_id": "x"})
        with self.assertRaisesRegex(StrategicValidationError, "source_id"):
            EvidenceRef.from_dict({"source_type": "manual_analysis"})


if __name__ == "__main__":
    unittest.main()
