"""Focused contract tests for the source-semantic IR boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
import unittest

from pipeline.semantic_ir import (
    AmbiguityState,
    Conditionality,
    EdgeType,
    GroundedValue,
    Modality,
    ModelDecisionProvenance,
    NodeType,
    Polarity,
    QualifierAmbiguity,
    QualifierAmbiguityState,
    QualifierCue,
    QualifierKind,
    SemanticEdge,
    SemanticGraph,
    SemanticNode,
    SemanticQualifiers,
    SourceSpan,
    TemporalScope,
    Uncertainty,
    edge_type_supports,
)
from pipeline.semantic_source import PASS0_VERSION


SOURCE = "If Mira might not open the gate, rain and wind cause two delays and flooding."
SOURCE_ID = "transcript:recording-1"
SOURCE_PROVENANCE = "1" * 64
WINDOW_SUFFIX = hashlib.sha256(
    f"{SOURCE_PROVENANCE}:100:{100 + len(SOURCE)}:{PASS0_VERSION}".encode()
).hexdigest()[:12]
WINDOW_ID = f"{SOURCE_ID}:w0004-{WINDOW_SUFFIX}"


def _span(text: str, *, source: str = SOURCE_ID, window: str = WINDOW_ID) -> SourceSpan:
    start = SOURCE.index(text)
    return SourceSpan(source, window, start, start + len(text), text, 100 + start, 100 + start + len(text),
                      "coach", 12.0, 14.0)


def _provenance(decision: str, candidates: tuple[str, ...] = ()) -> ModelDecisionProvenance:
    return ModelDecisionProvenance.create(
        decision, "reference-model@2026-08", "pass1-v1",
        configuration={"temperature": 0, "schema": 2}, model_input={"window": WINDOW_ID},
        model_output={"decision": decision}, candidate_ids=candidates,
    )


def _node(kind: NodeType, text: str, decision: str | None = None, **kwargs) -> SemanticNode:
    return SemanticNode(kind, _span(text), _provenance(decision or f"select-{text}"), **kwargs)


def _graph(nodes, edges=()) -> SemanticGraph:
    return SemanticGraph(
        SOURCE_ID, WINDOW_ID, "transcript", 100, 100 + len(SOURCE), SOURCE,
        hashlib.sha256(SOURCE.encode()).hexdigest(), SOURCE_PROVENANCE, PASS0_VERSION,
        "coach", 12.0, 14.0,
        tuple(nodes), tuple(edges),
    )


class SourceSpanTests(unittest.TestCase):
    def test_exact_offsets_round_trip_all_source_metadata(self) -> None:
        span = _span("Mira")
        self.assertEqual(span.text, SOURCE[span.local_start:span.local_end])
        self.assertEqual((span.absolute_start, span.absolute_end), (103, 107))
        self.assertEqual((span.speaker, span.start_timestamp, span.end_timestamp), ("coach", 12.0, 14.0))
        self.assertEqual(SourceSpan.from_dict(span.to_dict()), span)

    def test_rejects_invalid_or_inexact_spans(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            SourceSpan("s", "w", True, 3, "abc")
        with self.assertRaisesRegex(ValueError, "text length"):
            SourceSpan("s", "w", 0, 2, "wrong")
        with self.assertRaisesRegex(ValueError, "supplied together"):
            SourceSpan("s", "w", 0, 1, "x", absolute_start=9)
        bad = SourceSpan(SOURCE_ID, WINDOW_ID, 0, 2, "xx")
        with self.assertRaisesRegex(ValueError, "exactly match"):
            _graph((SemanticNode(NodeType.ENTITY, bad, _provenance("bad")),))

    def test_source_objects_are_immutable(self) -> None:
        span = _span("Mira")
        with self.assertRaises(FrozenInstanceError):
            span.text = "changed"  # type: ignore[misc]


class SemanticGraphTests(unittest.TestCase):
    def test_all_declared_general_types_are_exact(self) -> None:
        self.assertEqual(
            set(NodeType),
            {NodeType.ENTITY, NodeType.ABILITY_OR_RESOURCE, NodeType.EVENT, NodeType.ACTION,
             NodeType.STATE, NodeType.OUTCOME, NodeType.QUANTITY, NodeType.TIME,
             NodeType.LOCATION_OR_SPACE},
        )
        self.assertEqual(
            {item.value for item in EdgeType},
            {"ACTOR", "TARGET", "OBJECT", "EXPERIENCER", "CAUSES", "ENABLES", "PREVENTS",
             "REQUIRES", "CONDITION", "PURPOSE", "RESULT", "TEMPORAL_BEFORE", "TEMPORAL_AFTER",
             "TEMPORAL_UNTIL", "TERMINATES", "CONTRASTS_WITH", "NEGATES", "MODIFIES", "REFERS_TO"},
        )

    def test_negation_modality_and_condition_remain_first_class(self) -> None:
        qualifiers = SemanticQualifiers(
            polarity=Polarity.NEGATIVE, negated=True, modality=Modality.POSSIBLE,
            temporal_scope=TemporalScope.FUTURE, conditionality=Conditionality.CONDITIONAL,
            uncertainty=Uncertainty.POSSIBLE,
            cues=tuple(QualifierCue(kind, _span("If Mira might not")) for kind in (
                QualifierKind.POLARITY, QualifierKind.MODALITY, QualifierKind.TEMPORAL_SCOPE,
                QualifierKind.CONDITIONALITY, QualifierKind.UNCERTAINTY,
            )),
        )
        action = _node(NodeType.ACTION, "not open", qualifiers=qualifiers)
        condition = _node(NodeType.STATE, "If Mira might not")
        condition_edge = SemanticEdge(
            EdgeType.CONDITION, condition.node_id, action.node_id,
            (_span("If Mira might not open"),), _provenance("condition-edge"),
        )
        graph = _graph((action, condition), (condition_edge,))
        encoded = next(
            item["qualifiers"] for item in graph.to_artifact()["nodes"]
            if item["node_id"] == action.node_id
        )
        self.assertEqual(encoded["polarity"], "NEGATIVE")
        self.assertTrue(encoded["negated"])
        self.assertEqual(encoded["modality"], "POSSIBLE")
        self.assertEqual(encoded["conditionality"], "CONDITIONAL")
        with self.assertRaisesRegex(ValueError, "must agree"):
            SemanticQualifiers(
                polarity=Polarity.NEGATIVE, negated=False,
                cues=(QualifierCue(QualifierKind.POLARITY, _span("not")),),
            )
        with self.assertRaisesRegex(ValueError, "qualifier cues"):
            SemanticQualifiers(modality=Modality.POSSIBLE)

        with self.assertRaisesRegex(ValueError, "cannot replace"):
            _graph((action,))

    def test_relational_temporal_qualifier_requires_graph_structure(self) -> None:
        text = "Move before noon."
        source_id = "transcript:temporal"
        provenance_hash = "2" * 64
        suffix = hashlib.sha256(
            f"{provenance_hash}:0:{len(text)}:{PASS0_VERSION}".encode()
        ).hexdigest()[:12]
        window_id = f"{source_id}:w0001-{suffix}"

        def span(value: str) -> SourceSpan:
            start = text.index(value)
            return SourceSpan(source_id, window_id, start, start + len(value), value,
                              start, start + len(value))

        action = SemanticNode(
            NodeType.ACTION, span("Move"), _provenance("temporal-action"),
            qualifiers=SemanticQualifiers(
                temporal_scope=TemporalScope.BOUNDED,
                cues=(QualifierCue(QualifierKind.TEMPORAL_SCOPE, span("before")),),
            ),
        )
        time = SemanticNode(NodeType.TIME, span("noon"), _provenance("temporal-time"))

        def graph(edges=()):
            return SemanticGraph(
                source_id, window_id, "transcript", 0, len(text), text,
                hashlib.sha256(text.encode()).hexdigest(), provenance_hash, PASS0_VERSION,
                None, None, None, (action, time), tuple(edges),
            )

        with self.assertRaisesRegex(ValueError, "cannot replace"):
            graph()
        edge = SemanticEdge(
            EdgeType.TEMPORAL_BEFORE, action.node_id, time.node_id, (span(text),),
            _provenance("temporal-edge"),
        )
        graph((edge,))

    def test_unresolved_qualifier_is_grounded_and_round_trips(self) -> None:
        cue = QualifierCue(QualifierKind.MODALITY, _span("might"))
        ambiguity = QualifierAmbiguity(
            QualifierKind.MODALITY, QualifierAmbiguityState.AMBIGUOUS,
            (cue,), ("COUNTERFACTUAL", "POSSIBLE"), 0.4,
        )
        qualifiers = SemanticQualifiers(ambiguities=(ambiguity,))
        self.assertEqual(SemanticQualifiers.from_dict(qualifiers.to_dict()), qualifiers)
        self.assertIn(cue.span, qualifiers.spans())
        with self.assertRaisesRegex(ValueError, "cannot also be unresolved"):
            SemanticQualifiers(
                modality=Modality.POSSIBLE, cues=(cue,), ambiguities=(ambiguity,),
            )

    def test_multiple_causes_and_effects_are_not_flattened_to_a_tuple(self) -> None:
        cause = _node(NodeType.EVENT, "rain and wind")
        delays = _node(NodeType.OUTCOME, "two delays")
        flooding = _node(NodeType.OUTCOME, "flooding")
        evidence = (_span("rain and wind cause two delays and flooding"),)
        edges = (
            SemanticEdge(EdgeType.CAUSES, cause.node_id, delays.node_id, evidence, _provenance("delay")),
            SemanticEdge(EdgeType.CAUSES, cause.node_id, flooding.node_id, evidence, _provenance("flood")),
        )
        graph = _graph((cause, delays, flooding), edges)
        self.assertEqual(sum(edge.edge_type is EdgeType.CAUSES for edge in graph.edges), 2)
        self.assertEqual(len({edge.target_node_id for edge in graph.edges}), 2)

    def test_edges_require_proof_and_graph_rejects_unknown_endpoints(self) -> None:
        rain = _node(NodeType.EVENT, "rain")
        delays = _node(NodeType.OUTCOME, "two delays")
        with self.assertRaisesRegex(ValueError, "evidence"):
            SemanticEdge(EdgeType.CAUSES, rain.node_id, delays.node_id, (), _provenance("empty-proof"))
        edge = SemanticEdge(EdgeType.CAUSES, rain.node_id, "node_missing", (_span("rain and wind cause two delays"),),
                            _provenance("unknown-endpoint"))
        with self.assertRaisesRegex(ValueError, "endpoint"):
            _graph((rain, delays), (edge,))

    def test_edge_direction_types_and_endpoint_covering_evidence_fail_closed(self) -> None:
        rain = _node(NodeType.EVENT, "rain")
        delays = _node(NodeType.OUTCOME, "two delays")
        irrelevant = SemanticEdge(
            EdgeType.CAUSES, rain.node_id, delays.node_id, (_span("cause"),), _provenance("irrelevant"),
        )
        with self.assertRaisesRegex(ValueError, "cover both"):
            _graph((rain, delays), (irrelevant,))
        backwards_actor = SemanticEdge(
            EdgeType.ACTOR, rain.node_id, delays.node_id,
            (_span("rain and wind cause two delays"),), _provenance("wrong-role"),
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            _graph((rain, delays), (backwards_actor,))

    def test_graph_binds_absolute_offsets_and_context_metadata(self) -> None:
        node = _node(NodeType.ENTITY, "Mira")
        forged = SemanticNode(
            NodeType.ENTITY,
            SourceSpan(SOURCE_ID, WINDOW_ID, 3, 7, "Mira", 999, 1003, "coach", 12.0, 14.0),
            _provenance("forged"),
        )
        with self.assertRaisesRegex(ValueError, "absolute offsets"):
            _graph((forged,))
        wrong_speaker = SemanticNode(NodeType.ENTITY, _span("Mira"), _provenance("speaker"))
        object.__setattr__(wrong_speaker.source_span, "speaker", "other")
        with self.assertRaisesRegex(ValueError, "speaker"):
            _graph((wrong_speaker,))
        self.assertEqual(_graph((node,)).bronze_source_sha256, hashlib.sha256(SOURCE.encode()).hexdigest())

    def test_graph_rejects_inconsistent_pass0_identity(self) -> None:
        graph = _graph((_node(NodeType.ENTITY, "Mira"),))
        artifact = graph.to_artifact()
        artifact["window_id"] = "arbitrary-window"
        artifact["content_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Pass 0 provenance"):
            SemanticGraph.from_artifact(artifact)

    def test_time_and_action_condition_signatures_are_expressive(self) -> None:
        self.assertTrue(edge_type_supports(
            EdgeType.TEMPORAL_BEFORE, NodeType.EVENT, NodeType.TIME,
        ))
        self.assertTrue(edge_type_supports(
            EdgeType.CONDITION, NodeType.ACTION, NodeType.EVENT,
        ))
        self.assertTrue(edge_type_supports(
            EdgeType.MODIFIES, NodeType.LOCATION_OR_SPACE, NodeType.ACTION,
        ))
        self.assertFalse(edge_type_supports(
            EdgeType.MODIFIES, NodeType.LOCATION_OR_SPACE, NodeType.ENTITY,
        ))

    def test_cross_source_nodes_and_edge_evidence_fail_closed(self) -> None:
        local = _node(NodeType.EVENT, "rain")
        foreign_span = _span("wind", source="transcript:recording-2")
        foreign = SemanticNode(NodeType.EVENT, foreign_span, _provenance("foreign"))
        edge = SemanticEdge(EdgeType.CAUSES, local.node_id, foreign.node_id, (_span("cause"),),
                            _provenance("cross-source"))
        with self.assertRaisesRegex(ValueError, "different source"):
            _graph((local, foreign), (edge,))

        outcome = _node(NodeType.OUTCOME, "two delays")
        foreign_evidence = _span("cause", source="transcript:recording-2")
        edge = SemanticEdge(EdgeType.CAUSES, local.node_id, outcome.node_id, (foreign_evidence,),
                            _provenance("foreign-proof"))
        with self.assertRaisesRegex(ValueError, "different source"):
            _graph((local, outcome), (edge,))

    def test_unresolved_coreference_is_preserved_without_a_fake_edge(self) -> None:
        mira = _node(NodeType.ENTITY, "Mira")
        gate = _node(NodeType.ABILITY_OR_RESOURCE, "gate")
        mention = SemanticNode(
            NodeType.ENTITY, _span("wind"), _provenance("ambiguous-reference"),
            ambiguity=AmbiguityState.MULTIPLE_CANDIDATES,
            referent_candidates=(_span("Mira"), _span("gate")),
            referent_candidate_node_ids=(mira.node_id, gate.node_id),
        )
        graph = _graph((mention, mira, gate))
        self.assertFalse(graph.edges)
        restored = next(node for node in graph.nodes if node.node_id == mention.node_id)
        self.assertEqual(restored.ambiguity, AmbiguityState.MULTIPLE_CANDIDATES)
        self.assertEqual({item.text for item in restored.referent_candidates}, {"Mira", "gate"})

        invalid = SemanticNode(
            NodeType.EVENT, _span("wind"), _provenance("invalid-reference"),
            ambiguity=AmbiguityState.AMBIGUOUS,
            referent_candidates=(_span("Mira"),),
            referent_candidate_node_ids=("node_missing",),
        )
        with self.assertRaisesRegex(ValueError, "selected target"):
            _graph((invalid, mira))
        rain = _node(NodeType.EVENT, "rain")
        event_reference = SemanticNode(
            NodeType.EVENT, _span("wind"), _provenance("event-reference"),
            ambiguity=AmbiguityState.AMBIGUOUS, referent_candidates=(_span("rain"),),
            referent_candidate_node_ids=(rain.node_id,),
        )
        self.assertEqual(
            next(node for node in _graph((event_reference, rain)).nodes if node.node_id == event_reference.node_id)
            .referent_candidates[0].text,
            "rain",
        )
        unresolved = SemanticNode(
            NodeType.ENTITY, _span("wind"), _provenance("unresolved"),
            ambiguity=AmbiguityState.AMBIGUOUS,
        )
        invented = SemanticEdge(
            EdgeType.REFERS_TO, unresolved.node_id, mira.node_id,
            (_span("Mira might not open the gate, rain and wind"),),
            _provenance("invented-reference"), confidence=0.9,
        )
        with self.assertRaisesRegex(ValueError, "retained candidates"):
            _graph((unresolved, mira), (invented,))

    def test_stable_ids_hashing_order_and_artifact_round_trip(self) -> None:
        rain = _node(NodeType.EVENT, "rain")
        same_mention_new_decision = SemanticNode(NodeType.EVENT, _span("rain"), _provenance("rain-retry"))
        self.assertEqual(rain.node_id, same_mention_new_decision.node_id)
        wind = _node(NodeType.EVENT, "wind")
        delays = _node(NodeType.OUTCOME, "two delays")
        evidence = (_span("rain and wind cause two delays"),)
        first = SemanticEdge(EdgeType.CAUSES, rain.node_id, delays.node_id, evidence, _provenance("rain-causes"))
        second = SemanticEdge(EdgeType.CAUSES, wind.node_id, delays.node_id, evidence, _provenance("wind-causes"))
        graph_a = _graph((rain, wind, delays), (first, second))
        graph_b = _graph((delays, wind, rain), (second, first))
        self.assertEqual(graph_a.content_hash, graph_b.content_hash)
        self.assertEqual(graph_a.to_json(), graph_b.to_json())
        restored = SemanticGraph.from_json(graph_a.to_json())
        self.assertEqual(restored, graph_a)
        self.assertEqual(restored.content_hash, graph_a.content_hash)

    def test_tampering_with_provenance_or_content_hash_is_detected(self) -> None:
        graph = _graph((_node(NodeType.ENTITY, "Mira"),))
        artifact = json.loads(graph.to_json())
        artifact["nodes"][0]["provenance"]["output_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "content hash"):
            SemanticGraph.from_artifact(artifact)

    def test_duplicate_json_keys_and_normalizing_type_coercion_are_rejected(self) -> None:
        graph = _graph((_node(NodeType.ENTITY, "Mira"),))
        payload = graph.to_json()
        duplicate = payload[:-1] + ',"content_hash":"' + graph.content_hash + '"}'
        with self.assertRaisesRegex(ValueError, "duplicate keys"):
            SemanticGraph.from_json(duplicate)
        with self.assertRaisesRegex(ValueError, "closed non-domain"):
            GroundedValue("two delays", _span("two delays"), 2, "wave_obligation")
        with self.assertRaisesRegex(ValueError, "must be a string"):
            GroundedValue.from_dict({
                "text": 2, "span": _span("two delays").to_dict(),
                "normalized_value": 2, "unit": "count",
            })

        artifact = json.loads(graph.to_json())
        artifact["content_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "content hash"):
            SemanticGraph.from_artifact(artifact)

    def test_schema_has_no_domain_ontology_or_fixed_proposition_slots(self) -> None:
        forbidden = {"access", "continuity", "tempo", "initiative", "wave_obligation"}
        schema_terms = {name.lower() for name in NodeType.__members__} | {
            name.lower() for name in EdgeType.__members__
        }
        self.assertTrue(forbidden.isdisjoint(schema_terms))
        artifact = _graph((_node(NodeType.ENTITY, "Mira"),)).to_artifact()
        self.assertTrue({"subject", "predicate", "effect", "condition"}.isdisjoint(artifact))

    def test_confidence_and_compiler_version_fail_closed(self) -> None:
        for confidence in (True, -0.1, 1.1, float("nan")):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                SemanticNode(NodeType.ENTITY, _span("Mira"), _provenance("bad"), confidence=confidence)
        with self.assertRaisesRegex(ValueError, "compiler_version"):
            SemanticNode(
                NodeType.ENTITY, _span("Mira"), _provenance("bad-version"),
                compiler_version="prompt-v0",
            )


if __name__ == "__main__":
    unittest.main()
