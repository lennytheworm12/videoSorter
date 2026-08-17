import json
import unittest
from dataclasses import replace

from pipeline.semantic_coreference import (
    apply_coreference_decision,
    assemble_coreference_catalog,
    classify_coreference,
    classify_coreference_catalog,
    coreference_prompt,
    generate_coreference_candidate_sets,
    parse_coreference_decision,
)
from pipeline.semantic_ir import (
    AmbiguityState, EdgeType, ModelDecisionProvenance, NodeType, SemanticGraph,
    SemanticNode, SourceSpan, content_sha256,
)
from pipeline.semantic_source import BronzeSource, window_from_exact_span


TEXT = "Lux misses Q. She cannot stop your advance, so respect her."


def _window(text=TEXT):
    source = BronzeSource("transcript:coref", text, speaker="coach")
    return window_from_exact_span(source, 0, len(text))


def _node(window, node_type, text, occurrence=0):
    starts = []
    cursor = 0
    while True:
        start = window.text.find(text, cursor)
        if start < 0:
            break
        starts.append(start)
        cursor = start + 1
    start = starts[occurrence]
    span = SourceSpan(
        window.source_id, window.window_id, start, start + len(text), text,
        window.source_start + start, window.source_start + start + len(text), window.speaker,
    )
    provenance = ModelDecisionProvenance.create(
        f"mention:{text}:{occurrence}", "reference", "mention-v1", configuration={},
        model_input={"window": window.window_id}, model_output={"text": text, "start": start},
    )
    return SemanticNode(node_type, span, provenance, confidence=0.9)


def _raw(status, *, target=None, candidates=(), confidence=0.0):
    return json.dumps({
        "status": status, "target_node_id": target,
        "candidate_node_ids": list(candidates), "confidence": confidence,
    })


class SemanticCoreferenceTests(unittest.TestCase):
    def test_candidate_sets_are_complete_and_retain_reference_without_targets(self):
        window = _window()
        lux = _node(window, NodeType.ENTITY, "Lux")
        ability = _node(window, NodeType.ABILITY_OR_RESOURCE, "Q")
        she = _node(window, NodeType.ENTITY, "She")
        action = _node(window, NodeType.ACTION, "respect her")
        nodes = (lux, ability, she, action)
        sets = generate_coreference_candidate_sets(window, nodes)
        self.assertEqual(len(sets), 1)
        candidate_set = sets[0]
        self.assertEqual(candidate_set.source_node_id, she.node_id)
        self.assertEqual(set(candidate_set.target_node_ids), {lux.node_id, ability.node_id, action.node_id})
        candidate_set.validate(window, nodes)
        with self.assertRaises(ValueError):
            replace(candidate_set, target_node_ids=(lux.node_id,)).validate(window, nodes)

        reference_only = _window("She waits.")
        lone_she = _node(reference_only, NodeType.ENTITY, "She")
        zero = generate_coreference_candidate_sets(reference_only, (lone_she,))
        self.assertEqual(len(zero), 1)
        self.assertEqual(zero[0].target_node_ids, ())
        decision = classify_coreference(
            reference_only, (lone_she,), zero[0], lambda **kwargs: _raw("UNKNOWN"),
            model="pro", configuration={},
        )
        applied = apply_coreference_decision(reference_only, (lone_she,), zero[0], decision)
        self.assertEqual(applied.nodes[0].ambiguity, AmbiguityState.UNKNOWN)

    def test_overlapping_submention_is_not_a_possible_antecedent(self):
        window = _window("Lux controls this wave.")
        lux = _node(window, NodeType.ENTITY, "Lux")
        reference = _node(window, NodeType.ENTITY, "this wave")
        nested = _node(window, NodeType.ENTITY, "wave")
        candidate_set = generate_coreference_candidate_sets(window, (lux, reference, nested))[0]
        self.assertIn(lux.node_id, candidate_set.target_node_ids)
        self.assertNotIn(nested.node_id, candidate_set.target_node_ids)

    def test_prompt_grounding_disambiguates_repeated_reference_occurrences(self):
        window = _window("It sees it near Lux.")
        first = _node(window, NodeType.ENTITY, "It", 0)
        second = _node(window, NodeType.ENTITY, "it", 0)
        lux = _node(window, NodeType.ENTITY, "Lux")
        nodes = (first, second, lux)
        sets = generate_coreference_candidate_sets(window, nodes)
        self.assertEqual(len(sets), 2)
        node_by_id = {node.node_id: node for node in nodes}
        prompts = [coreference_prompt(window, item, node_by_id) for item in sets]
        for item, prompt in zip(sets, prompts):
            source = node_by_id[item.source_node_id]
            self.assertIn(window.text, prompt)
            self.assertIn(source.node_id, prompt)
            self.assertIn(f'"start":{source.source_span.local_start}', prompt)
            self.assertIn(f'"end":{source.source_span.local_end}', prompt)
        self.assertNotEqual(prompts[0], prompts[1])

    def test_resolved_reference_is_unambiguous_and_builds_proof_edge(self):
        window = _window()
        lux = _node(window, NodeType.ENTITY, "Lux")
        she = _node(window, NodeType.ENTITY, "She")
        nodes = (lux, she)
        candidate_set = generate_coreference_candidate_sets(window, nodes)[0]
        decision = classify_coreference(
            window, nodes, candidate_set,
            lambda **kwargs: _raw("RESOLVED", target=lux.node_id, confidence=0.95),
            model="reference-pro", configuration={"provider": "deepseek"},
        )
        applied = apply_coreference_decision(window, nodes, candidate_set, decision)
        updated = next(node for node in applied.nodes if node.source_span.text == "She")
        self.assertEqual(updated.referent_candidates, (lux.source_span,))
        self.assertEqual(updated.referent_candidate_node_ids, (lux.node_id,))
        self.assertEqual(updated.ambiguity, AmbiguityState.NONE)
        self.assertEqual(applied.edges[0].edge_type, EdgeType.REFERS_TO)
        graph = SemanticGraph.from_source_window(window, applied.nodes, applied.edges)
        self.assertEqual(len(graph.edges), 1)

    def test_ambiguous_reference_retains_node_bound_candidates_without_edge(self):
        window = _window("Lux and Morgana move. She waits.")
        lux = _node(window, NodeType.ENTITY, "Lux")
        morgana = _node(window, NodeType.ENTITY, "Morgana")
        she = _node(window, NodeType.ENTITY, "She")
        nodes = (lux, morgana, she)
        candidate_set = generate_coreference_candidate_sets(window, nodes)[0]
        decision = classify_coreference(
            window, nodes, candidate_set,
            lambda **kwargs: _raw(
                "AMBIGUOUS", candidates=(lux.node_id, morgana.node_id), confidence=0.5,
            ), model="pro", configuration={},
        )
        applied = apply_coreference_decision(window, nodes, candidate_set, decision)
        updated = next(node for node in applied.nodes if node.source_span.text == "She")
        self.assertEqual(updated.ambiguity, AmbiguityState.MULTIPLE_CANDIDATES)
        self.assertEqual(set(updated.referent_candidate_node_ids), {lux.node_id, morgana.node_id})
        self.assertFalse(applied.edges)
        SemanticGraph.from_source_window(window, applied.nodes)

    def test_parser_rejects_guess_low_confidence_smuggling_and_duplicate_keys(self):
        window = _window()
        lux = _node(window, NodeType.ENTITY, "Lux")
        she = _node(window, NodeType.ENTITY, "She")
        candidate_set = generate_coreference_candidate_sets(window, (lux, she))[0]
        invalid = (
            {"status": "RESOLVED", "target_node_id": "node_missing", "candidate_node_ids": [], "confidence": 0.9},
            {"status": "RESOLVED", "target_node_id": lux.node_id, "candidate_node_ids": [], "confidence": 0.1},
            {"status": "AMBIGUOUS", "target_node_id": None, "candidate_node_ids": [lux.node_id], "confidence": 0.5},
            {"status": "UNKNOWN", "target_node_id": lux.node_id, "candidate_node_ids": [], "confidence": 0.2},
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_coreference_decision(json.dumps(value), candidate_set)
        with self.assertRaises(ValueError):
            parse_coreference_decision(
                '{"status":"NONE","status":"RESOLVED","target_node_id":null,'
                '"candidate_node_ids":[],"confidence":0}', candidate_set,
            )

    def test_provider_parse_failure_is_validated_and_survives_in_ir(self):
        window = _window()
        lux = _node(window, NodeType.ENTITY, "Lux")
        she = _node(window, NodeType.ENTITY, "She")
        nodes = (lux, she)
        candidate_set = generate_coreference_candidate_sets(window, nodes)[0]

        def fail(**kwargs):
            raise RuntimeError("offline")

        provider = classify_coreference(window, nodes, candidate_set, fail, model="pro", configuration={})
        parsed = classify_coreference(
            window, nodes, candidate_set, lambda **kwargs: b"{}", model="pro", configuration={},
        )
        self.assertTrue(provider.failure.startswith("CoreferenceProviderError:"))
        self.assertEqual(parsed.failure, "ValueError")
        for decision in (provider, parsed):
            applied = apply_coreference_decision(window, nodes, candidate_set, decision)
            updated = next(node for node in applied.nodes if node.source_span.text == "She")
            self.assertEqual(updated.ambiguity, AmbiguityState.INSUFFICIENT_EVIDENCE)
            self.assertEqual(len(updated.additional_provenance), 1)
        forged = replace(
            provider, status="RESOLVED", target_node_id=lux.node_id, confidence=0.9,
        )
        with self.assertRaisesRegex(ValueError, "failed coreference"):
            apply_coreference_decision(window, nodes, candidate_set, forged)

    def test_request_shape_temperature_and_hash_fail_closed(self):
        window = _window()
        lux = _node(window, NodeType.ENTITY, "Lux")
        she = _node(window, NodeType.ENTITY, "She")
        nodes = (lux, she)
        candidate_set = generate_coreference_candidate_sets(window, nodes)[0]
        valid = classify_coreference(
            window, nodes, candidate_set,
            lambda **kwargs: _raw("RESOLVED", target=lux.node_id, confidence=0.9),
            model="pro", configuration={},
        )
        request = json.loads(valid.request_json)
        request["extra"] = "evil"
        with self.assertRaisesRegex(ValueError, "invalid shape"):
            apply_coreference_decision(
                window, nodes, candidate_set, replace(valid, request_json=json.dumps(request)),
            )
        request = json.loads(valid.request_json)
        request["temperature"] = False
        effective = {key: request[key] for key in (
            "caller_configuration", "temperature", "max_tokens", "model", "thinking",
            "prompt_version", "candidate_version", "max_segment_distance",
        )}
        forged = replace(
            valid, request_json=json.dumps(request),
            configuration_sha256=content_sha256(effective),
        )
        with self.assertRaisesRegex(ValueError, "temperature"):
            apply_coreference_decision(window, nodes, candidate_set, forged)

    def test_catalog_requires_every_reference_decision(self):
        window = _window("It moves. She waits. Lux watches.")
        it = _node(window, NodeType.ENTITY, "It")
        she = _node(window, NodeType.ENTITY, "She")
        lux = _node(window, NodeType.ENTITY, "Lux")
        nodes = (it, she, lux)
        sets = generate_coreference_candidate_sets(window, nodes)
        self.assertEqual(len(sets), 2)
        decisions = tuple(classify_coreference(
            window, nodes, item, lambda **kwargs: _raw("UNKNOWN"),
            model="pro", configuration={},
        ) for item in sets)
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            assemble_coreference_catalog(window, nodes, sets, decisions[:1])
        with self.assertRaisesRegex(ValueError, "complete deterministic"):
            classify_coreference_catalog(
                window, nodes, sets[:1], lambda **kwargs: _raw("UNKNOWN"),
                model="pro", configuration={},
            )
        run = assemble_coreference_catalog(window, nodes, sets, decisions)
        self.assertEqual(run.status, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(len(run.abstentions), 2)
        SemanticGraph.from_source_window(window, run.nodes, run.edges)

    def test_demonstrative_reference_np_and_possessive_pronoun_are_distinguished(self):
        window = _window("Lux spends mana. Their ability returns.")
        lux = _node(window, NodeType.ENTITY, "Lux")
        mana = _node(window, NodeType.ABILITY_OR_RESOURCE, "mana")
        possessor = _node(window, NodeType.ENTITY, "Their")
        ability = _node(window, NodeType.ABILITY_OR_RESOURCE, "Their ability")
        sets = generate_coreference_candidate_sets(window, (lux, mana, possessor, ability))
        self.assertEqual(len(sets), 1)
        self.assertEqual(sets[0].source_node_id, possessor.node_id)

    def test_reflexive_reciprocal_and_former_latter_families_are_detected(self):
        for text, reference_text in (
            ("Lux shields herself.", "herself"),
            ("Lux and Morgana help each other.", "each other"),
            ("Lux or Morgana moves; the latter waits.", "the latter"),
            ("You guys rotate while Lux waits.", "You guys"),
            ("You   guys rotate while Lux waits.", "You   guys"),
        ):
            with self.subTest(reference=reference_text):
                window = _window(text)
                lux = _node(window, NodeType.ENTITY, "Lux")
                reference = _node(window, NodeType.ENTITY, reference_text)
                sets = generate_coreference_candidate_sets(window, (lux, reference))
                self.assertEqual(len(sets), 1)
                self.assertEqual(sets[0].source_node_id, reference.node_id)

    def test_resolved_graph_requires_matching_edge_and_decision_provenance(self):
        window = _window()
        lux = _node(window, NodeType.ENTITY, "Lux")
        she = _node(window, NodeType.ENTITY, "She")
        nodes = (lux, she)
        candidate_set = generate_coreference_candidate_sets(window, nodes)[0]
        decision = classify_coreference(
            window, nodes, candidate_set,
            lambda **kwargs: _raw("RESOLVED", target=lux.node_id, confidence=0.95),
            model="pro", configuration={},
        )
        applied = apply_coreference_decision(window, nodes, candidate_set, decision)
        with self.assertRaisesRegex(ValueError, "requires exactly one"):
            SemanticGraph.from_source_window(window, applied.nodes, ())
        forged = replace(applied.edges[0], provenance=lux.provenance)
        with self.assertRaisesRegex(ValueError, "retained candidates"):
            SemanticGraph.from_source_window(window, applied.nodes, (forged,))


if __name__ == "__main__":
    unittest.main()
