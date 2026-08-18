import copy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import scripts.eval_phase2f_semantic_ir as runner
from scripts.eval_phase2f_semantic_ir import (
    LEGACY_CASE_CHECKSUM_DENOMINATORS, LEGACY_GATE_VERSION,
    LEGACY_BENCHMARK_SHA256, LEGACY_METRIC_DENOMINATORS, REFERENCE_ENDPOINT, REFERENCE_MODEL,
    REFERENCE_THINKING,
    evaluate_legacy_gate, reference_config,
)


def _metric(hit, denominator):
    return {
        "hit_count": hit,
        "denominator": denominator,
        "rate": hit / denominator if denominator else None,
    }


def _passing_evaluation():
    cardinalities = {
        "wave-reset-after-kill": (5, 4, 1, 2, 12),
        "push-poke-wave-crash": (5, 4, 1, 2, 12),
        "sweeper-limits-mid-play": (7, 6, 1, 0, 14),
        "mid-push-prevents-side-collapse": (5, 4, 2, 1, 12),
        "unwarded-bush-hook-risk": (11, 6, 5, 3, 25),
    }
    cases = []
    for case_id, (mentions, edges, qualifiers, references, facts) in cardinalities.items():
        metrics = {
            "mention_candidate_coverage": _metric(mentions, mentions),
            "mention_selection_recall": _metric(mentions, mentions),
            "mention_type_recall": _metric(mentions, mentions),
            "edge_pair_coverage": _metric(edges, edges),
            "edge_recall": _metric(edges, edges),
            "qualifier_candidate_coverage": _metric(qualifiers, qualifiers),
            "qualifier_recall": _metric(qualifiers, qualifiers),
            "reference_candidate_coverage": _metric(references, references),
            "reference_recall": _metric(references, references),
            "semantic_completeness": _metric(facts, facts),
            "semantic_checksum": _metric(facts, facts),
        }
        cases.append({
            "case_id": case_id, "status": "PARTIAL" if references else "OK",
            "source_span_validity": True, "edge_provenance_traceability": True,
            "failures": [
                {
                    "code": "INSUFFICIENT_EVIDENCE", "fact_id": f"reference:{index}",
                    "critical": False, "stage": "coreference",
                    "detail": "INSUFFICIENT_EVIDENCE",
                } for index in range(references)
            ],
            "first_loss": "INSUFFICIENT_EVIDENCE" if references else None,
            "questions": [{
                "answerable_from_bronze": True, "answerable_from_ir": True,
                "missing_requirements": [],
            } for _ in range(facts)],
            "recovered_facts": [f"fact:{index}" for index in range(facts)],
            "dimensions": {"semantic_completeness": _metric(facts, facts)},
            "critical_dimensions": {"semantic_completeness": _metric(facts, facts)},
            "mention_families": {"ACTION": _metric(mentions, mentions)},
            **metrics,
        })
    value = {
        "benchmark_schema_version": runner.BENCHMARK_SCHEMA_VERSION,
        "benchmark_content_sha256": LEGACY_BENCHMARK_SHA256,
        "split": "LEGACY_FAILURE", "case_count": 5,
        "source_span_validity": _metric(5, 5),
        "edge_provenance_traceability": _metric(5, 5),
        "failure_counts": {"INSUFFICIENT_EVIDENCE": 8},
        "cases": cases,
    }
    value.update({
        name: _metric(denominator, denominator)
        for name, denominator in LEGACY_METRIC_DENOMINATORS.items()
    })
    return value


def _gate(value, *, recomputed=None):
    with patch.object(
        runner, "evaluate_semantic_benchmark",
        return_value=copy.deepcopy(value if recomputed is None else recomputed),
    ):
        return evaluate_legacy_gate(value, benchmark=object(), runs={})


class Phase2FSemanticIRRunnerTests(unittest.TestCase):
    def test_reference_configuration_matches_the_preregistered_strong_model(self):
        config = reference_config()
        self.assertEqual(config.model, REFERENCE_MODEL)
        self.assertEqual(config.thinking, REFERENCE_THINKING)
        self.assertEqual(config.mention_partition_size, 600)
        self.assertEqual(config.provider_mapping()["provider"], "deepseek")
        self.assertEqual(config.provider_mapping()["endpoint"], REFERENCE_ENDPOINT)

    def test_legacy_gate_requires_every_fact_and_proof_dimension(self):
        result = _gate(_passing_evaluation())
        self.assertEqual(result["gate_version"], LEGACY_GATE_VERSION)
        self.assertTrue(result["passed"])
        self.assertEqual(result["reasons"], [])

        missed = _passing_evaluation()
        missed["cases"][2]["semantic_checksum"] = _metric(13, 14)
        missed["semantic_checksum"] = _metric(74, 75)
        failed = _gate(missed)
        self.assertFalse(failed["passed"])
        self.assertTrue(any(
            "sweeper-limits-mid-play semantic checksum" in item
            for item in failed["reasons"]
        ))

    def test_provider_parse_and_zero_denominator_cannot_pass(self):
        value = _passing_evaluation()
        value["cases"][0]["failures"].extend([
            {
                "code": "PROVIDER_FAILURE", "fact_id": "provider:1", "critical": False,
                "stage": "mentions", "detail": "MentionProviderError:TimeoutError",
            },
            {
                "code": "MODEL_PARSE_FAILURE", "fact_id": "parse:1", "critical": False,
                "stage": "mentions", "detail": "ValueError",
            },
        ])
        value["failure_counts"].update({"PROVIDER_FAILURE": 1, "MODEL_PARSE_FAILURE": 1})
        value["reference_recall"] = _metric(0, 0)
        result = _gate(value)
        self.assertFalse(result["passed"])
        self.assertTrue(any("reference_recall" in item for item in result["reasons"]))
        self.assertTrue(any("PROVIDER_FAILURE" in item for item in result["reasons"]))
        self.assertTrue(any("MODEL_PARSE_FAILURE" in item for item in result["reasons"]))

    def test_fake_denominators_rates_case_ids_and_unknown_taxonomy_fail_closed(self):
        for mutation in (
            "one-of-one", "lying-rate", "boolean-rate", "denominator-transfer",
            "fake-case", "unknown-failure",
        ):
            with self.subTest(mutation=mutation):
                value = _passing_evaluation()
                if mutation == "one-of-one":
                    value["mention_selection_recall"] = _metric(1, 1)
                elif mutation == "lying-rate":
                    value["mention_selection_recall"] = {
                        "hit_count": 0, "denominator": 33, "rate": 1.0,
                    }
                elif mutation == "boolean-rate":
                    value["mention_selection_recall"]["rate"] = True
                elif mutation == "denominator-transfer":
                    first, second = value["cases"][:2]
                    for name in LEGACY_METRIC_DENOMINATORS:
                        if name == "semantic_checksum":
                            continue
                        moved = first[name]["denominator"]
                        first[name] = _metric(0, 0)
                        current = second[name]["denominator"]
                        second[name] = _metric(current + moved, current + moved)
                elif mutation == "fake-case":
                    value["cases"][0]["case_id"] = "fake-case"
                else:
                    value["cases"][0]["failures"].append({
                        "code": "BOGUS", "fact_id": "bad:1", "critical": False,
                        "stage": "mentions", "detail": "BOGUS",
                    })
                    value["failure_counts"]["BOGUS"] = 1
                self.assertFalse(_gate(value)["passed"])

    def test_input_evaluation_is_not_mutated(self):
        value = _passing_evaluation()
        retained = copy.deepcopy(value)
        _gate(value)
        self.assertEqual(value, retained)

    def test_gate_rejects_any_evaluation_not_reconstructed_from_typed_runs(self):
        genuine = _passing_evaluation()
        forged = copy.deepcopy(genuine)
        forged["benchmark_content_sha256"] = "0" * 64
        forged["cases"][0]["questions"] = []
        forged["cases"][0]["dimensions"] = {"entity_recovery": _metric(0, 999)}
        self.assertFalse(_gate(forged, recomputed=genuine)["passed"])

    def test_direct_cli_bootstraps_repo_imports(self):
        script = Path(runner.__file__).resolve()
        result = subprocess.run(
            [sys.executable, str(script), "--help"], cwd="/tmp",
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--live", result.stdout)

    def test_published_failed_gate_returns_nonzero(self):
        fake_args = [
            "eval_phase2f_semantic_ir.py", "--db", "db", "--phase2d", "p2d",
            "--phase2e-artifact", "p2e", "--manifest", "manifest",
            "--benchmark", "benchmark", "--pool", "pool", "--output", "out", "--live",
        ]
        aggregate = {
            "content_sha256": "a" * 64,
            "gate": {"passed": False, "reasons": ["semantic loss"]},
        }
        with patch.object(sys, "argv", fake_args), \
                patch.object(runner, "validate_inputs", return_value=({}, {}, {}, object())), \
                patch.object(runner, "run_live", return_value=(Path("out"), aggregate)):
            self.assertEqual(runner.main(), 2)


if __name__ == "__main__":
    unittest.main()
