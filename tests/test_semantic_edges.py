import json
import unittest
from dataclasses import replace

from pipeline.semantic_edges import (
    candidate_pair_coverage,
    assemble_semantic_graph,
    classify_edge_catalog,
    classify_edge_pair,
    generate_candidate_edge_pairs,
    parse_edge_classification,
)
from pipeline.semantic_ir import (
    AmbiguityState,
    EdgeType,
    ModelDecisionProvenance,
    NodeType,
    SemanticNode,
    SourceSpan,
    content_sha256,
)
from pipeline.semantic_source import BronzeSource, window_from_exact_span


TEXT = "When Lux misses Q, you can walk forward because she can't stop your advance. Once Q comes back, respect her again."


def _context(text=TEXT):
    source = BronzeSource("transcript:lux", text, speaker="coach")
    return window_from_exact_span(source, 0, len(text))


def _node(window, node_type, text):
    start = window.text.index(text)
    span = SourceSpan(window.source_id, window.window_id, start, start + len(text), text,
                      window.source_start + start, window.source_start + start + len(text), window.speaker)
    provenance = ModelDecisionProvenance.create(
        "mention:" + text, "reference-model", "mention-v0", configuration={"t": 0},
        model_input={"window": window.window_id}, model_output={"text": text}, candidate_ids=("candidate:" + text,),
    )
    return SemanticNode(node_type, span, provenance)


class SemanticEdgeTests(unittest.TestCase):
    def test_pair_catalog_includes_role_cause_condition_and_termination_directions(self):
        window = _context()
        lux = _node(window, NodeType.ENTITY, "Lux")
        miss = _node(window, NodeType.EVENT, "misses Q")
        walk = _node(window, NodeType.ACTION, "walk forward")
        cannot = _node(window, NodeType.STATE, "can't stop your advance")
        returns = _node(window, NodeType.EVENT, "Q comes back")
        nodes = (lux, miss, walk, cannot, returns)
        pairs = generate_candidate_edge_pairs(window, nodes)
        offered = {(p.source_node_id, p.target_node_id): set(p.allowed_edge_types) for p in pairs}
        self.assertIn(EdgeType.ACTOR, offered[(lux.node_id, miss.node_id)])
        self.assertIn(EdgeType.CONDITION, offered[(miss.node_id, walk.node_id)])
        self.assertIn(EdgeType.CAUSES, offered[(miss.node_id, cannot.node_id)])
        self.assertIn(EdgeType.TERMINATES, offered[(returns.node_id, cannot.node_id)])

    def test_pruning_reports_gold_pair_loss_separately(self):
        window = _context()
        first = _node(window, NodeType.EVENT, "misses Q")
        last = _node(window, NodeType.ACTION, "respect her again")
        pairs = generate_candidate_edge_pairs(window, (first, last), max_segment_distance=0)
        report = candidate_pair_coverage(
            pairs, ((first.node_id, last.node_id, EdgeType.CAUSES),),
            window=window, nodes=(first, last),
        )
        self.assertEqual(report, {"hit_count": 0, "denominator": 1, "recall": 0.0})

    def test_classification_accepts_one_allowed_edge_and_builds_proof(self):
        window = _context()
        cause = _node(window, NodeType.EVENT, "misses Q")
        effect = _node(window, NodeType.STATE, "can't stop your advance")
        pair = next(p for p in generate_candidate_edge_pairs(window, (cause, effect))
                    if p.source_node_id == cause.node_id and p.target_node_id == effect.node_id)
        raw = json.dumps({"status": "SUPPORTED", "edge_types": ["CAUSES"], "confidence": 0.93, "ambiguity": "NONE"})
        result = classify_edge_pair(pair, window, {cause.node_id: cause, effect.node_id: effect},
                                    lambda **kwargs: raw, model="deepseek-v4-pro", configuration={"thinking": True})
        self.assertEqual(result.status, "SUPPORTED")
        self.assertEqual(result.edge.edge_type, EdgeType.CAUSES)
        self.assertEqual(result.edge.evidence[0].text, pair.evidence_span.text)
        self.assertEqual(result.raw_output, raw)

    def test_no_relation_and_ambiguous_do_not_smuggle_edges(self):
        window = _context()
        left = _node(window, NodeType.EVENT, "misses Q")
        right = _node(window, NodeType.ACTION, "walk forward")
        pair = next(p for p in generate_candidate_edge_pairs(window, (left, right))
                    if p.source_node_id == left.node_id)
        for status, ambiguity in (("NO_RELATION", "NONE"), ("AMBIGUOUS", "AMBIGUOUS")):
            raw = json.dumps({"status": status, "edge_types": [], "confidence": 0.4, "ambiguity": ambiguity})
            result = classify_edge_pair(pair, window, {left.node_id: left, right.node_id: right},
                                        lambda **kwargs: raw, model="pro", configuration={})
            self.assertIsNone(result.edge)

    def test_parser_rejects_reversed_unsupported_relation_unknown_type_and_duplicate_key(self):
        window = _context()
        entity = _node(window, NodeType.ENTITY, "Lux")
        event = _node(window, NodeType.EVENT, "misses Q")
        pair = next(p for p in generate_candidate_edge_pairs(window, (entity, event))
                    if p.source_node_id == entity.node_id)
        invalid = [
            {"status": "SUPPORTED", "edge_types": ["CAUSES"], "confidence": 0.9, "ambiguity": "NONE"},
            {"status": "SUPPORTED", "edge_types": ["DENIES"], "confidence": 0.9, "ambiguity": "NONE"},
            {"status": "NO_RELATION", "edge_types": ["ACTOR"], "confidence": 0.5, "ambiguity": "NONE"},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_edge_classification(json.dumps(value), pair)
        with self.assertRaises(ValueError):
            parse_edge_classification('{"status":"SUPPORTED","status":"NO_RELATION","edge_types":[],"confidence":0.1,"ambiguity":"NONE"}', pair)

    def test_provider_and_parse_failure_remain_distinct(self):
        window = _context()
        left = _node(window, NodeType.EVENT, "misses Q")
        right = _node(window, NodeType.ACTION, "walk forward")
        pair = next(p for p in generate_candidate_edge_pairs(window, (left, right)) if p.source_node_id == left.node_id)
        nodes = {left.node_id: left, right.node_id: right}

        def fail(**kwargs):
            raise RuntimeError("offline")

        provider = classify_edge_pair(pair, window, nodes, fail, model="pro", configuration={})
        parsed = classify_edge_pair(pair, window, nodes, lambda **kwargs: "not-json", model="pro", configuration={})
        self.assertTrue(provider.failure.startswith("EdgeProviderError:"))
        self.assertEqual(parsed.failure, "ValueError")
        self.assertEqual(parsed.raw_output, "not-json")
        byte_output = classify_edge_pair(
            pair, window, nodes,
            lambda **kwargs: b'{"status":"NO_RELATION","edge_types":[],"confidence":1,"ambiguity":"NONE"}',
            model="pro", configuration={},
        )
        self.assertEqual(byte_output.failure, "ValueError")

    def test_condition_evidence_retains_leading_discourse_cue(self):
        window = _context()
        condition = _node(window, NodeType.EVENT, "misses Q")
        action = _node(window, NodeType.ACTION, "walk forward")
        pair = next(
            item for item in generate_candidate_edge_pairs(window, (condition, action))
            if item.source_node_id == condition.node_id and item.target_node_id == action.node_id
        )
        self.assertTrue(pair.evidence_span.text.startswith("When "))
        self.assertIn("When Lux misses Q", pair.evidence_span.text)

    def test_pair_validation_reconstructs_mapping_signatures_evidence_and_config(self):
        window = _context()
        entity = _node(window, NodeType.ENTITY, "Lux")
        event = _node(window, NodeType.EVENT, "misses Q")
        pair = next(item for item in generate_candidate_edge_pairs(window, (entity, event))
                    if item.source_node_id == entity.node_id)
        nodes = {entity.node_id: entity, event.node_id: event}
        for forged in (
            replace(pair, allowed_edge_types=(EdgeType.CAUSES,)),
            replace(pair, character_distance=999),
            replace(pair, evidence_span=entity.source_span),
            replace(pair, max_segment_distance=99),
        ):
            with self.subTest(forged=forged), self.assertRaises(ValueError):
                forged.validate(window, nodes)
        with self.assertRaisesRegex(ValueError, "mapping key"):
            pair.validate(window, {entity.node_id: event, event.node_id: entity})

    def test_pair_signatures_cover_reference_resource_condition_and_time_termination(self):
        text = "If it has 3 stacks in river, the action requires mana until noon."
        window = _context(text)
        resource = _node(window, NodeType.ABILITY_OR_RESOURCE, "mana")
        base_pronoun = _node(window, NodeType.ENTITY, "it")
        pronoun = SemanticNode(
            NodeType.ENTITY, base_pronoun.source_span, base_pronoun.provenance,
            ambiguity=AmbiguityState.NONE,
            referent_candidates=(resource.source_span,),
            referent_candidate_node_ids=(resource.node_id,),
        )
        quantity = _node(window, NodeType.QUANTITY, "3 stacks")
        location = _node(window, NodeType.LOCATION_OR_SPACE, "river")
        action = _node(window, NodeType.ACTION, "action")
        time = _node(window, NodeType.TIME, "noon")
        state = _node(window, NodeType.STATE, "has 3 stacks")
        pairs = generate_candidate_edge_pairs(window, (pronoun, quantity, location, action, resource, time, state))
        offered = {(item.source_node_id, item.target_node_id): set(item.allowed_edge_types) for item in pairs}
        self.assertIn(EdgeType.REFERS_TO, offered[(pronoun.node_id, resource.node_id)])
        self.assertIn(EdgeType.REQUIRES, offered[(action.node_id, resource.node_id)])
        self.assertIn(EdgeType.CONDITION, offered[(quantity.node_id, action.node_id)])
        self.assertIn(EdgeType.CONDITION, offered[(location.node_id, action.node_id)])
        self.assertIn(EdgeType.CONDITION, offered[(resource.node_id, action.node_id)])
        self.assertIn(EdgeType.TERMINATES, offered[(time.node_id, state.node_id)])

    def test_multi_relation_status_matrix_and_confidence_gate(self):
        window = _context()
        left = _node(window, NodeType.EVENT, "misses Q")
        right = _node(window, NodeType.STATE, "can't stop your advance")
        pair = next(item for item in generate_candidate_edge_pairs(window, (left, right))
                    if item.source_node_id == left.node_id)
        raw = json.dumps({
            "status": "SUPPORTED", "edge_types": ["CAUSES", "ENABLES"],
            "confidence": 0.9, "ambiguity": "NONE",
        })
        result = classify_edge_pair(
            pair, window, {left.node_id: left, right.node_id: right}, lambda **kwargs: raw,
            model="pro", configuration={},
        )
        self.assertEqual({edge.edge_type for edge in result.edges}, {EdgeType.CAUSES, EdgeType.ENABLES})
        invalid = (
            {"status": "SUPPORTED", "edge_types": ["CAUSES"], "confidence": 0.0, "ambiguity": "NONE"},
            {"status": "AMBIGUOUS", "edge_types": [], "confidence": 0.4, "ambiguity": "NONE"},
            {"status": "NO_RELATION", "edge_types": [], "confidence": 1.0, "ambiguity": "MULTIPLE_CANDIDATES"},
            {"status": "UNKNOWN", "edge_types": [], "confidence": 0.3, "ambiguity": "NONE"},
            {"status": "INSUFFICIENT_EVIDENCE", "edge_types": [], "confidence": 0.2, "ambiguity": "UNKNOWN"},
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_edge_classification(json.dumps(value), pair)

    def test_catalog_assembly_retains_requests_no_relation_and_rejects_tampering(self):
        window = _context("A causes B.")
        left = _node(window, NodeType.EVENT, "A")
        right = _node(window, NodeType.OUTCOME, "B")
        pairs = generate_candidate_edge_pairs(window, (left, right))

        def classify(**kwargs):
            if "A OUTCOME: B" in kwargs["user"]:
                return json.dumps({
                    "status": "NO_RELATION", "edge_types": [],
                    "confidence": 0.95, "ambiguity": "NONE",
                })
            return json.dumps({
                "status": "SUPPORTED", "edge_types": ["CAUSES"],
                "confidence": 0.95, "ambiguity": "NONE",
            })

        run = classify_edge_catalog(
            window, (left, right), pairs, classify, model="reference-pro",
            configuration={"provider": "deepseek"}, max_tokens=300, thinking="enabled",
        )
        graph = assemble_semantic_graph(window, (left, right), run)
        self.assertEqual(len(graph.edges), 1)
        request = json.loads(run.results[0].request_json)
        self.assertEqual((request["max_tokens"], request["thinking"]), (300, "enabled"))
        forged_edge = replace(run.edges[0], confidence=0.99)
        with self.assertRaisesRegex(ValueError, "aggregate semantic edges"):
            assemble_semantic_graph(window, (left, right), replace(run, edges=(forged_edge,) + run.edges[1:]))

    def test_runtime_pruning_and_coverage_labels_fail_closed(self):
        window = _context()
        left = _node(window, NodeType.EVENT, "misses Q")
        right = _node(window, NodeType.ACTION, "walk forward")
        for value in (True, 1.5, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                generate_candidate_edge_pairs(window, (left, right), max_segment_distance=value)
        with self.assertRaises(ValueError):
            generate_candidate_edge_pairs(window, [left, right])
        pairs = generate_candidate_edge_pairs(window, (left, right))
        label = (left.node_id, right.node_id, EdgeType.CAUSES)
        with self.assertRaisesRegex(ValueError, "unique"):
            candidate_pair_coverage(pairs, (label, label), window=window, nodes=(left, right))

    def test_unresolved_coreference_without_candidates_is_not_enumerated(self):
        window = _context("Lux moves because she can act.")
        lux = _node(window, NodeType.ENTITY, "Lux")
        raw_she = _node(window, NodeType.ENTITY, "she")
        she = replace(raw_she, ambiguity=AmbiguityState.AMBIGUOUS)
        pairs = generate_candidate_edge_pairs(window, (lux, she))
        offered = {
            edge_type for pair in pairs if pair.source_node_id == she.node_id
            for edge_type in pair.allowed_edge_types
        }
        self.assertNotIn(EdgeType.REFERS_TO, offered)

    def test_edge_catalog_cannot_drop_pairs_or_forge_failure_taxonomy(self):
        window = _context("A causes B.")
        left = _node(window, NodeType.EVENT, "A")
        right = _node(window, NodeType.OUTCOME, "B")
        pairs = generate_candidate_edge_pairs(window, (left, right))
        with self.assertRaisesRegex(ValueError, "complete deterministic"):
            classify_edge_catalog(
                window, (left, right), pairs[:1], lambda **kwargs: "{}",
                model="pro", configuration={},
            )

        def fail(**kwargs):
            raise TimeoutError("offline")

        run = classify_edge_catalog(window, (left, right), pairs, fail, model="pro", configuration={})
        forged_result = replace(run.results[0], status="SUPPORTED")
        forged = replace(run, results=(forged_result,) + run.results[1:])
        with self.assertRaisesRegex(ValueError, "failed edge decision"):
            assemble_semantic_graph(window, (left, right), forged)

        invalid_latency = replace(run.results[0], latency_ms={"forged": True})
        forged = replace(run, results=(invalid_latency,) + run.results[1:])
        with self.assertRaisesRegex(ValueError, "latency"):
            assemble_semantic_graph(window, (left, right), forged)

    def test_edge_request_constants_and_incompatible_relations_fail_closed(self):
        window = _context("A happens before B.")
        left = _node(window, NodeType.EVENT, "A")
        right = _node(window, NodeType.EVENT, "B")
        pairs = generate_candidate_edge_pairs(window, (left, right))
        pair = next(item for item in pairs if item.source_node_id == left.node_id)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            parse_edge_classification(json.dumps({
                "status": "SUPPORTED", "edge_types": ["TEMPORAL_BEFORE", "TEMPORAL_AFTER"],
                "confidence": 0.9, "ambiguity": "NONE",
            }), pair)

        no_relation = json.dumps({
            "status": "NO_RELATION", "edge_types": [], "confidence": 0.9, "ambiguity": "NONE",
        })
        run = classify_edge_catalog(
            window, (left, right), pairs, lambda **kwargs: no_relation,
            model="pro", configuration={},
        )
        result = run.results[0]
        request = json.loads(result.request_json)
        request["prompt_version"] = "evil-v99"
        effective = {key: request[key] for key in (
            "caller_configuration", "temperature", "max_tokens", "model", "thinking",
            "prompt_version", "pair_version", "max_character_distance", "max_segment_distance",
        )}
        forged_result = replace(
            result, request_json=json.dumps(request, sort_keys=True, separators=(",", ":")),
            configuration_sha256=content_sha256(effective),
        )
        forged = replace(run, results=(forged_result,) + run.results[1:])
        with self.assertRaisesRegex(ValueError, "constants/configuration"):
            assemble_semantic_graph(window, (left, right), forged)
        request = json.loads(result.request_json); request["temperature"] = False
        effective = {key: request[key] for key in (
            "caller_configuration", "temperature", "max_tokens", "model", "thinking",
            "prompt_version", "pair_version", "max_character_distance", "max_segment_distance",
        )}
        forged_result = replace(
            result, request_json=json.dumps(request, sort_keys=True, separators=(",", ":")),
            configuration_sha256=content_sha256(effective),
        )
        with self.assertRaisesRegex(ValueError, "constants/configuration"):
            assemble_semantic_graph(window, (left, right), replace(
                run, results=(forged_result,) + run.results[1:],
            ))


if __name__ == "__main__":
    unittest.main()
