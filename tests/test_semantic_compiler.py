import json
import re
import unittest
from dataclasses import replace
from unittest.mock import patch

from pipeline.semantic_compiler import (
    COMPILER_ORCHESTRATION_VERSION_LEGACY,
    SemanticCompilerConfig, _run_integrity_sha256, compile_source_semantic_ir,
)
import pipeline.semantic_compiler as semantic_compiler
from pipeline.semantic_coreference import COREFERENCE_SYSTEM
from pipeline.semantic_edges import EDGE_CLASSIFIER_SYSTEM
from pipeline.semantic_ir import AmbiguityState, EdgeType
from pipeline.semantic_mentions import MENTION_SELECTION_SYSTEM
from pipeline.semantic_qualifiers import QUALIFIER_SYSTEM
from pipeline.semantic_source import BronzeSource, window_from_exact_span


TEXT = "Lux walks. She waits."


def _window():
    source = BronzeSource("transcript:compiler", TEXT, speaker="coach")
    return window_from_exact_span(source, 0, len(TEXT))


def _none_qualifiers():
    return {
        field: {
            "status": "NONE", "value": None, "cue_ids": [],
            "candidate_values": [], "confidence": 0.0,
        }
        for field in (
            "polarity", "modality", "temporal_scope", "conditionality",
            "comparative_degree", "uncertainty", "restriction",
        )
    }


class ScriptedSemanticModel:
    mention_types = {"Lux": "ENTITY", "walks": "ACTION", "She": "ENTITY", "waits": "ACTION"}

    def __call__(self, **kwargs):
        system, user = kwargs["system"], kwargs["user"]
        if system == MENTION_SELECTION_SYSTEM:
            payload = user.split("CANDIDATES:\n", 1)[1].split("\nSelect every", 1)[0]
            selections = []
            for candidate in json.loads(payload):
                node_type = self.mention_types.get(candidate["text"])
                if node_type:
                    selections.append({
                        "candidate_id": candidate["id"], "node_type": node_type,
                        "confidence": 0.95, "ambiguity": "NONE",
                    })
            return json.dumps({
                "status": "OK" if selections else "NONE", "mentions": selections,
            })
        if system == QUALIFIER_SYSTEM:
            return json.dumps({"status": "NONE", "qualifiers": _none_qualifiers()})
        if system == COREFERENCE_SYSTEM:
            reference = json.loads(user.split("REFERENCE: ", 1)[1].split("\n", 1)[0])
            targets = json.loads(user.split("POSSIBLE TARGETS: ", 1)[1].split("\n", 1)[0])
            if reference["source_text"] == "She":
                lux = next(item for item in targets if item["source_text"] == "Lux")
                return json.dumps({
                    "status": "RESOLVED", "target_node_id": lux["node_id"],
                    "candidate_node_ids": [], "confidence": 0.95,
                })
            return json.dumps({
                "status": "UNKNOWN", "target_node_id": None,
                "candidate_node_ids": [], "confidence": 0.0,
            })
        if system == EDGE_CLASSIFIER_SYSTEM:
            source = re.search(r"\nA [A-Z_]+: (.*)\n", user).group(1)
            target = re.search(r"\nB [A-Z_]+: (.*)\n", user).group(1)
            allowed = json.loads(user.split("Allowed directed A->B relations: ", 1)[1].split("\n", 1)[0])
            edge_types = []
            if source == "Lux" and target in {"walks", "waits"} and "ACTOR" in allowed:
                edge_types = ["ACTOR"]
            elif source == "She" and target == "Lux" and "REFERS_TO" in allowed:
                edge_types = ["REFERS_TO"]
            return json.dumps({
                "status": "SUPPORTED" if edge_types else "NO_RELATION",
                "edge_types": edge_types, "confidence": 0.95 if edge_types else 0.0,
                "ambiguity": "NONE",
            })
        raise AssertionError("unexpected semantic compiler system prompt")


class ContradictoryReferenceModel(ScriptedSemanticModel):
    def __call__(self, **kwargs):
        if kwargs["system"] == EDGE_CLASSIFIER_SYSTEM:
            return json.dumps({
                "status": "NO_RELATION", "edge_types": [], "confidence": 0.0,
                "ambiguity": "NONE",
            })
        return super().__call__(**kwargs)


class EdgeOutcomeModel(ScriptedSemanticModel):
    def __init__(self, outcome):
        self.outcome = outcome

    def __call__(self, **kwargs):
        if kwargs["system"] != EDGE_CLASSIFIER_SYSTEM:
            return super().__call__(**kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        if self.outcome == "PARSE":
            return "not-json"
        ambiguity = {
            "UNKNOWN": "UNKNOWN", "AMBIGUOUS": "AMBIGUOUS",
            "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_EVIDENCE",
        }[self.outcome]
        return json.dumps({
            "status": self.outcome, "edge_types": [], "confidence": 0.0,
            "ambiguity": ambiguity,
        })


class SemanticCompilerTests(unittest.TestCase):
    def test_legacy_orchestration_is_deserialization_only_and_prompt_version_is_bound(self):
        legacy = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            version=COMPILER_ORCHESTRATION_VERSION_LEGACY,
        )
        with self.assertRaisesRegex(ValueError, "deserialization-only"):
            compile_source_semantic_ir(_window(), ScriptedSemanticModel(), config=legacy)

        current = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )
        run = compile_source_semantic_ir(_window(), ScriptedSemanticModel(), config=current)
        forged_config = replace(current, version=COMPILER_ORCHESTRATION_VERSION_LEGACY)
        forged = replace(
            run, config=forged_config, version=COMPILER_ORCHESTRATION_VERSION_LEGACY,
        )
        forged = replace(forged, integrity_sha256=_run_integrity_sha256(forged))
        with self.assertRaisesRegex(ValueError, "mention prompt versions disagree"):
            forged.validate()

        small_config = replace(current, mention_partition_size=2)
        partitioned_run = compile_source_semantic_ir(
            _window(), ScriptedSemanticModel(), config=small_config,
        )
        self.assertGreater(len(partitioned_run.mention_selection.partition_results), 1)
        reversed_results = tuple(reversed(partitioned_run.mention_selection.partition_results))
        repartitioned = replace(
            partitioned_run.mention_selection,
            partition_results=reversed_results,
            mentions=tuple(
                mention for result in reversed_results for mention in result.mentions
            ),
        )
        forged = replace(partitioned_run, mention_selection=repartitioned)
        forged = replace(forged, integrity_sha256=_run_integrity_sha256(forged))
        with self.assertRaisesRegex(ValueError, "focal mention partitions are not deterministic"):
            forged.validate()

    def test_end_to_end_run_retains_every_catalog_and_builds_graph(self):
        window = _window()
        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )
        run = compile_source_semantic_ir(window, ScriptedSemanticModel(), config=config)
        self.assertEqual(run.status, "OK")
        self.assertIsNotNone(run.graph)
        self.assertEqual(len(run.mention_nodes), 4)
        self.assertEqual(len(run.qualifier_runs), 4)
        self.assertEqual(len(run.coreference.candidate_sets), 1)
        self.assertEqual(len(run.coreference.decisions), 1)
        she = next(node for node in run.graph.nodes if node.source_span.text == "She")
        lux = next(node for node in run.graph.nodes if node.source_span.text == "Lux")
        self.assertEqual(she.ambiguity, AmbiguityState.NONE)
        self.assertEqual(she.referent_candidate_node_ids, (lux.node_id,))
        triples = {
            (edge.edge_type, edge.source_node_id, edge.target_node_id)
            for edge in run.graph.edges
        }
        self.assertIn((EdgeType.REFERS_TO, she.node_id, lux.node_id), triples)
        self.assertEqual(sum(item[0] is EdgeType.REFERS_TO for item in triples), 1)
        self.assertEqual(len(run.edge_classification.results), len(run.edge_classification.pairs))
        self.assertFalse(run.failures)

    def test_provider_failure_remains_partial_without_erasing_source_graph(self):
        window = _window()
        model = ScriptedSemanticModel()

        def fail_qualifiers(**kwargs):
            if kwargs["system"] == QUALIFIER_SYSTEM:
                raise TimeoutError("offline")
            return model(**kwargs)

        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )
        run = compile_source_semantic_ir(window, fail_qualifiers, config=config)
        self.assertEqual(run.status, "PARTIAL")
        self.assertIsNotNone(run.graph)
        failures = [item for item in run.failures if item.stage == "qualifiers"]
        self.assertEqual(len(failures), 4)
        self.assertTrue(all(item.code == "PROVIDER_FAILURE" for item in failures))
        for node in run.qualified_nodes:
            self.assertTrue(any(":qualifiers:" in item.decision_id for item in node.additional_provenance))

    def test_config_runtime_types_fail_closed(self):
        for kwargs in (
            {"mention_partition_size": True},
            {"edge_max_segment_distance": 1.5},
            {"thinking": ""},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SemanticCompilerConfig.create(
                    "reference-pro", provider_configuration={}, **kwargs,
                )

    def test_reference_disagreement_is_retained_and_cannot_be_ok(self):
        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )
        run = compile_source_semantic_ir(
            _window(), ContradictoryReferenceModel(), config=config,
        )
        self.assertEqual(run.status, "PARTIAL")
        self.assertTrue(any(
            item.code == "REFERENCE_RESOLUTION_ERROR" for item in run.failures
        ))
        self.assertEqual(sum(
            edge.edge_type is EdgeType.REFERS_TO for edge in run.merged_edges
        ), 1)
        run.validate()

    def test_nested_provider_config_is_canonical_and_caller_mutation_safe(self):
        routing = ["primary"]
        supplied = {"provider": "scripted", "routing": routing}
        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration=supplied,
            mention_partition_size=1000,
        )
        routing.append("tampered")
        supplied["provider"] = "other"
        self.assertEqual(config.provider_mapping(), {
            "provider": "scripted", "routing": ["primary"],
        })
        run = compile_source_semantic_ir(
            _window(), ScriptedSemanticModel(), config=config,
        )
        run.validate()
        with self.assertRaises(ValueError):
            SemanticCompilerConfig.create(
                "reference-pro", provider_configuration={"bad": float("nan")},
            )

    def test_post_run_parsed_output_mutation_is_detected(self):
        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )
        run = compile_source_semantic_ir(
            _window(), ScriptedSemanticModel(), config=config,
        )
        parsed = run.mention_selection.partition_results[0].parsed_output
        self.assertIsInstance(parsed, dict)
        parsed["status"] = "NONE"
        with self.assertRaises(ValueError):
            run.validate()

    def test_unexpected_stage_failure_returns_sealed_partial_run(self):
        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )
        with patch(
            "pipeline.semantic_compiler.classify_node_qualifiers",
            side_effect=RuntimeError("injected"),
        ):
            run = compile_source_semantic_ir(
                _window(), ScriptedSemanticModel(), config=config,
            )
        self.assertEqual(run.status, "FAILURE")
        self.assertIsNotNone(run.mention_selection)
        self.assertTrue(run.mention_catalog)
        self.assertTrue(run.mention_nodes)
        self.assertEqual(run.qualifier_runs, ())
        self.assertEqual(run.failures[-1].stage, "qualifiers")
        run.validate()

    def test_malformed_aliases_fail_before_model_calls(self):
        calls = []
        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={}, mention_partition_size=1000,
        )
        with self.assertRaises(ValueError):
            compile_source_semantic_ir(
                _window(), lambda **kwargs: calls.append(kwargs), config=config,
                entity_aliases=(object(),),
            )
        self.assertEqual(calls, [])

    def test_no_corroboration_is_not_mislabeled_as_reference_error(self):
        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )
        for outcome in (
            TimeoutError("offline"), "PARSE", "UNKNOWN", "AMBIGUOUS",
            "INSUFFICIENT_EVIDENCE",
        ):
            with self.subTest(outcome=repr(outcome)):
                run = compile_source_semantic_ir(
                    _window(), EdgeOutcomeModel(outcome), config=config,
                )
                self.assertFalse(any(
                    item.code == "REFERENCE_RESOLUTION_ERROR" for item in run.failures
                ))

    def test_every_deterministic_catalog_failure_returns_a_sealed_prefix(self):
        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )
        targets = (
            ("generate_mention_candidates", "mention_catalog"),
            ("generate_qualifier_candidates", "qualifier_catalog"),
            ("generate_coreference_candidate_sets", "coreference_catalog"),
            ("generate_candidate_edge_pairs", "edge_catalog"),
        )
        for symbol, stage in targets:
            with self.subTest(stage=stage), patch(
                "pipeline.semantic_compiler." + symbol,
                side_effect=RuntimeError("injected"),
            ):
                run = compile_source_semantic_ir(
                    _window(), ScriptedSemanticModel(), config=config,
                )
                self.assertEqual(run.status, "FAILURE")
                self.assertEqual(run.failures[-1].stage, stage)
                self.assertRegex(run.integrity_sha256, r"^[0-9a-f]{64}$")

    def test_qualifier_decision_survives_deterministic_application_failure(self):
        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )
        with patch(
            "pipeline.semantic_compiler.apply_node_qualifiers",
            side_effect=RuntimeError("injected"),
        ):
            run = compile_source_semantic_ir(
                _window(), ScriptedSemanticModel(), config=config,
            )
        self.assertEqual(run.status, "FAILURE")
        self.assertEqual(len(run.qualifier_runs), 1)
        attempt = run.qualifier_runs[0]
        self.assertTrue(attempt.result.raw_output)
        self.assertIsNone(attempt.output_node)
        self.assertIn("RuntimeError", attempt.application_failure)
        run.validate()

    def test_disallowed_and_unknown_edges_and_bad_reference_target_are_typed(self):
        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )

        class BadEdge(ScriptedSemanticModel):
            def __init__(self, edge_type):
                self.edge_type = edge_type

            def __call__(self, **kwargs):
                if kwargs["system"] == EDGE_CLASSIFIER_SYSTEM:
                    return json.dumps({
                        "status": "SUPPORTED", "edge_types": [self.edge_type],
                        "confidence": 0.95, "ambiguity": "NONE",
                    })
                return super().__call__(**kwargs)

        for edge_type in ("ACTOR", "MADE_UP_RELATION"):
            with self.subTest(edge_type=edge_type):
                run = compile_source_semantic_ir(
                    _window(), BadEdge(edge_type), config=config,
                )
                self.assertIn("UNSUPPORTED_EDGE", {item.code for item in run.failures})

        class BadReference(ScriptedSemanticModel):
            def __call__(self, **kwargs):
                if kwargs["system"] == COREFERENCE_SYSTEM:
                    return json.dumps({
                        "status": "RESOLVED", "target_node_id": "node:not-offered",
                        "candidate_node_ids": [], "confidence": 0.95,
                    })
                return super().__call__(**kwargs)

        run = compile_source_semantic_ir(_window(), BadReference(), config=config)
        self.assertIn("REFERENCE_RESOLUTION_ERROR", {item.code for item in run.failures})

    def test_resealed_terminal_run_cannot_smuggle_downstream_state(self):
        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )
        success = compile_source_semantic_ir(
            _window(), ScriptedSemanticModel(), config=config,
        )
        with patch(
            "pipeline.semantic_compiler.generate_qualifier_candidates",
            side_effect=RuntimeError("injected"),
        ):
            failed = compile_source_semantic_ir(
                _window(), ScriptedSemanticModel(), config=config,
            )
        forged = replace(failed, coreference=success.coreference)
        forged = replace(forged, integrity_sha256=_run_integrity_sha256(forged))
        with self.assertRaises(ValueError):
            forged.validate()

    def test_partial_coreference_and_edge_attempt_failures_are_aggregated(self):
        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )

        model = ScriptedSemanticModel()

        def coref_timeout(**kwargs):
            if kwargs["system"] == COREFERENCE_SYSTEM:
                raise TimeoutError("offline")
            return model(**kwargs)

        with patch(
            "pipeline.semantic_compiler.assemble_coreference_catalog",
            side_effect=RuntimeError("aggregate injected"),
        ):
            run = compile_source_semantic_ir(_window(), coref_timeout, config=config)
        self.assertTrue(run.coreference_decisions)
        self.assertIn("PROVIDER_FAILURE", {item.code for item in run.failures})
        self.assertEqual(run.failures[-1].stage, "coreference")
        run.validate()

        original = semantic_compiler.classify_edge_pair
        calls = 0

        def fail_after_one(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("classifier injected")
            # chat is positional in compiler orchestration.
            values = list(args)
            values[3] = EdgeOutcomeModel("UNKNOWN")
            return original(*values, **kwargs)

        with patch("pipeline.semantic_compiler.classify_edge_pair", side_effect=fail_after_one):
            run = compile_source_semantic_ir(_window(), model, config=config)
        self.assertEqual(len(run.edge_results), 1)
        self.assertIn("UNKNOWN", {item.code for item in run.failures})
        self.assertEqual(run.failures[-1].stage, "edges")
        run.validate()

    def test_terminal_attempt_raw_output_tampering_is_reconstructively_rejected(self):
        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )
        with patch(
            "pipeline.semantic_compiler.apply_node_qualifiers",
            side_effect=RuntimeError("application injected"),
        ):
            qualifier_run = compile_source_semantic_ir(
                _window(), ScriptedSemanticModel(), config=config,
            )
        attempt = qualifier_run.qualifier_runs[0]
        forged_result = replace(attempt.result, raw_output="forged-not-json")
        forged_attempt = replace(attempt, result=forged_result)
        forged = replace(qualifier_run, qualifier_runs=(forged_attempt,))
        forged = replace(forged, integrity_sha256=_run_integrity_sha256(forged))
        with self.assertRaises(ValueError):
            forged.validate()

    def test_terminal_coreference_prefix_and_failure_taxonomy_are_closed(self):
        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )
        with patch(
            "pipeline.semantic_compiler.assemble_coreference_catalog",
            side_effect=RuntimeError("aggregate injected"),
        ):
            run = compile_source_semantic_ir(
                _window(), ScriptedSemanticModel(), config=config,
            )
        forged = replace(
            run, coreference_decisions=run.coreference_decisions + run.coreference_decisions,
        )
        forged = replace(forged, integrity_sha256=_run_integrity_sha256(forged))
        with self.assertRaises(ValueError):
            forged.validate()

        with self.assertRaises(ValueError):
            replace(run.failures[-1], code="BOGUS_TAXONOMY", detail="forged")

        with patch(
            "pipeline.semantic_compiler.apply_node_qualifiers",
            side_effect=RuntimeError("application injected"),
        ):
            qualifier_run = compile_source_semantic_ir(
                _window(), ScriptedSemanticModel(), config=config,
            )
        wrong = replace(
            qualifier_run,
            failures=qualifier_run.failures[:-1] + (
                replace(qualifier_run.failures[-1], code="MODEL_PARSE_FAILURE"),
            ),
        )
        wrong = replace(wrong, integrity_sha256=_run_integrity_sha256(wrong))
        with self.assertRaises(ValueError):
            wrong.validate()

        with patch(
            "pipeline.semantic_compiler.assemble_edge_catalog_classification",
            side_effect=RuntimeError("aggregate injected"),
        ):
            edge_run = compile_source_semantic_ir(
                _window(), ScriptedSemanticModel(), config=config,
            )
        first = replace(edge_run.edge_results[0], raw_output="forged-not-json")
        forged = replace(edge_run, edge_results=(first,) + edge_run.edge_results[1:])
        forged = replace(forged, integrity_sha256=_run_integrity_sha256(forged))
        with self.assertRaises(ValueError):
            forged.validate()


if __name__ == "__main__":
    unittest.main()
