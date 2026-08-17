import json
import unittest
from dataclasses import replace

from pipeline.semantic_ir import (
    Conditionality, ModelDecisionProvenance, Modality, NodeType, Polarity,
    QualifierAmbiguityState, QualifierKind, Restriction, SemanticGraph, SemanticNode,
    SemanticQualifiers, SourceSpan, TemporalScope,
    Uncertainty,
)
from pipeline.semantic_qualifiers import (
    apply_node_qualifiers,
    classify_node_qualifiers,
    generate_qualifier_candidates,
    parse_qualifier_selection,
    qualifier_candidate_coverage,
    qualifier_candidates_for_node,
    qualifier_prompt,
)
from pipeline.semantic_source import BronzeSource, window_from_exact_span


TEXT = "If Lux might not walk forward before noon, she usually waits longer."


def _window(text=TEXT):
    source = BronzeSource("transcript:qualifiers", text, speaker="coach")
    return window_from_exact_span(source, 0, len(text))


def _node(window, node_type=NodeType.ACTION, text="walk forward"):
    start = window.text.index(text)
    span = SourceSpan(
        window.source_id, window.window_id, start, start + len(text), text,
        window.source_start + start, window.source_start + start + len(text), window.speaker,
    )
    provenance = ModelDecisionProvenance.create(
        "mention:" + text, "reference", "mention-v1", configuration={},
        model_input={"window": window.window_id}, model_output={"text": text},
    )
    return SemanticNode(node_type, span, provenance, confidence=0.9)


def _empty_fields():
    return {field: {
        "status": "NONE", "value": None, "cue_ids": [],
        "candidate_values": [], "confidence": 0.0,
    } for field in (
        "polarity", "modality", "temporal_scope", "conditionality",
        "comparative_degree", "uncertainty", "restriction",
    )}


def _asserted(value, cue_id, confidence=0.9):
    return {
        "status": "ASSERTED", "value": value, "cue_ids": [cue_id],
        "candidate_values": [], "confidence": confidence,
    }


class SemanticQualifierTests(unittest.TestCase):
    def test_catalog_covers_condition_negation_modality_time_comparison_uncertainty(self):
        window = _window()
        catalog = generate_qualifier_candidates(window)
        offered = {(item.source_text.lower(), item.kind.value) for item in catalog}
        self.assertIn(("if", "CONDITIONALITY"), offered)
        self.assertIn(("might", "MODALITY"), offered)
        self.assertIn(("not", "POLARITY"), offered)
        self.assertIn(("before", "TEMPORAL_SCOPE"), offered)
        self.assertIn(("usually", "UNCERTAINTY"), offered)
        self.assertIn(("longer", "COMPARATIVE_DEGREE"), offered)

    def test_node_local_catalog_keeps_leading_condition_cues(self):
        window = _window()
        node = _node(window)
        catalog = qualifier_candidates_for_node(window, node, generate_qualifier_candidates(window))
        self.assertIn("If", {item.source_text for item in catalog})
        self.assertIn("not", {item.source_text for item in catalog})

    def test_constrained_selection_applies_grounded_qualifiers_and_provenance(self):
        window = _window()
        node = _node(window)
        catalog = qualifier_candidates_for_node(window, node, generate_qualifier_candidates(window))
        by_key = {(item.source_text.lower(), item.kind.value): item for item in catalog}
        fields = _empty_fields()
        fields["polarity"] = _asserted("NEGATIVE", by_key[("not", "POLARITY")].candidate_id)
        fields["modality"] = _asserted("POSSIBLE", by_key[("might", "MODALITY")].candidate_id)
        fields["conditionality"] = _asserted("CONDITIONAL", by_key[("if", "CONDITIONALITY")].candidate_id)
        fields["temporal_scope"] = _asserted("BOUNDED", by_key[("before", "TEMPORAL_SCOPE")].candidate_id)
        fields["uncertainty"] = _asserted("POSSIBLE", by_key[("might", "UNCERTAINTY")].candidate_id)
        raw = json.dumps({"status": "OK", "qualifiers": fields})
        result = classify_node_qualifiers(
            window, node, catalog, lambda **kwargs: raw, model="reference-pro",
            configuration={"provider": "deepseek"}, thinking="enabled",
        )
        qualified = apply_node_qualifiers(window, node, catalog, result)
        self.assertEqual(qualified.qualifiers.polarity, Polarity.NEGATIVE)
        self.assertTrue(qualified.qualifiers.negated)
        self.assertEqual(qualified.qualifiers.modality, Modality.POSSIBLE)
        self.assertEqual(qualified.qualifiers.conditionality, Conditionality.CONDITIONAL)
        self.assertEqual(qualified.qualifiers.temporal_scope, TemporalScope.BOUNDED)
        self.assertEqual(qualified.qualifiers.uncertainty, Uncertainty.POSSIBLE)
        self.assertEqual({cue.span.text.lower() for cue in qualified.qualifiers.cues}, {"not", "might", "if", "before"})
        self.assertEqual(len(qualified.additional_provenance), 1)
        self.assertEqual(qualified.node_id, node.node_id)

    def test_parser_rejects_wrong_kind_unknown_free_form_and_non_ok_smuggling(self):
        window = _window()
        catalog = generate_qualifier_candidates(window)
        by_key = {(item.source_text.lower(), item.kind.value): item for item in catalog}
        variants = []
        fields = _empty_fields(); fields["polarity"] = _asserted(
            "NEGATIVE", by_key[("might", "MODALITY")].candidate_id,
        )
        variants.append({"status": "OK", "qualifiers": fields})
        fields = _empty_fields(); fields["modality"] = _asserted(
            "ACCESS", by_key[("might", "MODALITY")].candidate_id,
        )
        variants.append({"status": "OK", "qualifiers": fields})
        fields = _empty_fields(); fields["modality"] = _asserted(
            "UNKNOWN", by_key[("might", "MODALITY")].candidate_id,
        )
        variants.append({"status": "OK", "qualifiers": fields})
        fields = _empty_fields(); fields["polarity"] = _asserted(
            "NEGATIVE", by_key[("not", "POLARITY")].candidate_id,
        )
        variants.append({"status": "AMBIGUOUS", "qualifiers": fields})
        fields = _empty_fields(); fields["polarity"] = _asserted(
            "POSITIVE", by_key[("not", "POLARITY")].candidate_id,
        )
        variants.append({"status": "OK", "qualifiers": fields})
        fields = _empty_fields(); fields["modality"] = _asserted(
            "OBLIGATORY", by_key[("might", "MODALITY")].candidate_id,
        )
        variants.append({"status": "OK", "qualifiers": fields})
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_qualifier_selection(json.dumps(value), catalog)

    def test_provider_parse_failure_and_request_tampering_remain_distinct(self):
        window = _window()
        node = _node(window)
        catalog = qualifier_candidates_for_node(window, node, generate_qualifier_candidates(window))

        def fail(**kwargs):
            raise RuntimeError("offline")

        provider = classify_node_qualifiers(window, node, catalog, fail, model="pro", configuration={})
        parsed = classify_node_qualifiers(window, node, catalog, lambda **kwargs: "not-json", model="pro", configuration={})
        self.assertTrue(provider.failure.startswith("QualifierProviderError:"))
        self.assertEqual(parsed.failure, "ValueError")

        fields = _empty_fields()
        cue = next(item for item in catalog if item.kind.value == "POLARITY")
        fields["polarity"] = _asserted("NEGATIVE", cue.candidate_id)
        raw = json.dumps({"status": "OK", "qualifiers": fields})
        valid = classify_node_qualifiers(window, node, catalog, lambda **kwargs: raw, model="pro", configuration={})
        request = json.loads(valid.request_json); request["prompt_version"] = "evil"
        forged = replace(valid, request_json=json.dumps(request))
        with self.assertRaisesRegex(ValueError, "request constants"):
            apply_node_qualifiers(window, node, catalog, forged)

        provider_node = apply_node_qualifiers(window, node, catalog, provider)
        parse_node = apply_node_qualifiers(window, node, catalog, parsed)
        self.assertEqual(len(provider_node.additional_provenance), 1)
        self.assertEqual(len(parse_node.additional_provenance), 1)
        with self.assertRaises(ValueError):
            replace(provider, status="EVIL")
        forged_failure = replace(provider, failure="TypeError", raw_output="{}")
        with self.assertRaisesRegex(ValueError, "taxonomy"):
            apply_node_qualifiers(window, node, catalog, forged_failure)

        byte_result = classify_node_qualifiers(
            window, node, catalog, lambda **kwargs: b'{"status":"NONE"}',
            model="pro", configuration={},
        )
        self.assertEqual(byte_result.failure, "ValueError")
        apply_node_qualifiers(window, node, catalog, byte_result)
        empty_result = classify_node_qualifiers(
            window, node, catalog, lambda **kwargs: "",
            model="pro", configuration={},
        )
        self.assertEqual(empty_result.failure, "ValueError")
        empty_node = apply_node_qualifiers(window, node, catalog, empty_result)
        self.assertEqual(len(empty_node.additional_provenance), 1)

        request = json.loads(valid.request_json)
        request["temperature"] = False
        effective = {key: request[key] for key in (
            "caller_configuration", "temperature", "max_tokens", "model", "thinking",
            "prompt_version", "catalog_version",
        )}
        from pipeline.semantic_ir import content_sha256
        false_temperature = replace(
            valid, request_json=json.dumps(request, sort_keys=True, separators=(",", ":")),
            configuration_sha256=content_sha256(effective),
        )
        with self.assertRaisesRegex(ValueError, "request constants"):
            apply_node_qualifiers(window, node, catalog, false_temperature)

    def test_candidate_coverage_reports_all_buckets_and_negation_separately(self):
        window = _window()
        catalog = generate_qualifier_candidates(window)
        labels = []
        for text, label in (("If", "CONDITIONALITY"), ("not", "NEGATION"), ("might", "MODALITY")):
            start = window.text.index(text)
            labels.append((start, start + len(text), label))
        report = qualifier_candidate_coverage(window, catalog, labels)
        self.assertEqual(report["conditionality"]["recall"], 1.0)
        self.assertEqual(report["negation"]["recall"], 1.0)
        self.assertEqual(report["modality"]["recall"], 1.0)
        self.assertIn("uncertainty", report)

    def test_expanded_ascii_unicode_negation_and_modality_catalog(self):
        text = (
            "You didn't contest and shouldn't walk up. You can’t contest without Q. "
            "You cannot stay. Provided Lux misses Q, act roughly equal."
        )
        window = _window(text)
        offered = {(item.source_text.casefold(), item.kind) for item in generate_qualifier_candidates(window)}
        for cue in ("didn't", "shouldn't", "can’t", "without", "cannot"):
            self.assertIn((cue, QualifierKind.POLARITY), offered)
        for cue in ("shouldn't", "can’t", "cannot"):
            self.assertIn((cue, QualifierKind.MODALITY), offered)
        self.assertIn(("provided", QualifierKind.CONDITIONALITY), offered)
        self.assertIn(("roughly", QualifierKind.UNCERTAINTY), offered)
        self.assertIn(("equal", QualifierKind.COMPARATIVE_DEGREE), offered)

        asr = _window("You dont walk, wont contest, and can only move while Q is down; still wait.")
        offered = {(item.source_text.casefold(), item.kind) for item in generate_qualifier_candidates(asr)}
        self.assertIn(("dont", QualifierKind.POLARITY), offered)
        self.assertIn(("wont", QualifierKind.POLARITY), offered)
        self.assertIn(("wont", QualifierKind.TEMPORAL_SCOPE), offered)
        self.assertIn(("only", QualifierKind.RESTRICTION), offered)
        self.assertNotIn(("only", QualifierKind.CONDITIONALITY), offered)
        self.assertIn(("while", QualifierKind.TEMPORAL_SCOPE), offered)
        self.assertIn(("still", QualifierKind.TEMPORAL_SCOPE), offered)

    def test_repeated_cues_are_disambiguated_by_offsets_and_full_context(self):
        text = "If Lux misses Q, walk forward. If she doesn't, stay back."
        window = _window(text)
        node = _node(window, text="stay back")
        catalog = qualifier_candidates_for_node(window, node, generate_qualifier_candidates(window))
        if_cues = [item for item in catalog if item.source_text.casefold() == "if"]
        self.assertEqual(len(if_cues), 2)
        self.assertNotEqual(if_cues[0].start, if_cues[1].start)
        prompt = qualifier_prompt(window, node, catalog)
        self.assertIn(text, prompt)
        self.assertIn(f'"start":{if_cues[0].start}', prompt)
        self.assertIn(f'"start":{if_cues[1].start}', prompt)

    def test_catalog_must_be_complete_window_and_node_local_derivation(self):
        window = _window()
        node = _node(window)
        full = generate_qualifier_candidates(window)
        with self.assertRaisesRegex(ValueError, "complete deterministic window"):
            qualifier_candidates_for_node(window, node, full[:-1])
        local = qualifier_candidates_for_node(window, node, full)
        fields = _empty_fields()
        raw = json.dumps({"status": "NONE", "qualifiers": fields})
        with self.assertRaisesRegex(ValueError, "complete deterministic node-local"):
            classify_node_qualifiers(
                window, node, local[:-1], lambda **kwargs: raw,
                model="pro", configuration={},
            )
        with self.assertRaisesRegex(ValueError, "complete deterministic window"):
            qualifier_candidate_coverage(window, full[:-1], ())
        with self.assertRaisesRegex(ValueError, "complete deterministic node-local"):
            classify_node_qualifiers(
                window, node, tuple(reversed(local)), lambda **kwargs: raw,
                model="pro", configuration={},
            )

        other = _window("If Lux waits.")
        foreign_node = _node(other, text="waits")
        with self.assertRaises(ValueError):
            classify_node_qualifiers(
                window, foreign_node, local, lambda **kwargs: raw,
                model="pro", configuration={},
            )

    def test_mixed_asserted_and_ambiguous_fields_are_retained(self):
        text = "Lux might move farther than before."
        window = _window(text)
        node = _node(window, text="move")
        catalog = qualifier_candidates_for_node(window, node, generate_qualifier_candidates(window))
        by_key = {(item.source_text.casefold(), item.kind.value): item for item in catalog}
        fields = _empty_fields()
        fields["modality"] = _asserted("POSSIBLE", by_key[("might", "MODALITY")].candidate_id)
        fields["comparative_degree"] = {
            "status": "AMBIGUOUS", "value": None,
            "cue_ids": [by_key[("than", "COMPARATIVE_DEGREE")].candidate_id],
            "candidate_values": ["GREATER", "LESS"], "confidence": 0.4,
        }
        raw = json.dumps({"status": "OK", "qualifiers": fields})
        result = classify_node_qualifiers(
            window, node, catalog, lambda **kwargs: raw,
            model="pro", configuration={},
        )
        qualified = apply_node_qualifiers(window, node, catalog, result)
        self.assertEqual(qualified.qualifiers.modality, Modality.POSSIBLE)
        self.assertEqual(len(qualified.qualifiers.ambiguities), 1)
        ambiguity = qualified.qualifiers.ambiguities[0]
        self.assertEqual(ambiguity.state, QualifierAmbiguityState.AMBIGUOUS)
        self.assertEqual(set(ambiguity.candidate_values), {"GREATER", "LESS"})

        second = classify_node_qualifiers(
            window, qualified, catalog, lambda **kwargs: raw,
            model="other-pro", configuration={},
        )
        with self.assertRaisesRegex(ValueError, "only one qualifier decision"):
            apply_node_qualifiers(window, qualified, catalog, second)

    def test_counterfactual_and_comparative_values_remain_reachable(self):
        text = "If Lux had hit Q, you would have lived farther than before."
        window = _window(text)
        node = _node(window, text="would have lived")
        catalog = qualifier_candidates_for_node(window, node, generate_qualifier_candidates(window))
        by_key = {(item.source_text.casefold(), item.kind.value): item for item in catalog}
        fields = _empty_fields()
        fields["conditionality"] = _asserted(
            "COUNTERFACTUAL", by_key[("if", "CONDITIONALITY")].candidate_id,
        )
        fields["comparative_degree"] = _asserted(
            "GREATER", by_key[("farther", "COMPARATIVE_DEGREE")].candidate_id,
        )
        raw = json.dumps({"status": "OK", "qualifiers": fields})
        status, parsed, _ = parse_qualifier_selection(raw, catalog)
        self.assertEqual(status, "OK")
        self.assertEqual(dict(parsed)["conditionality"].value, "COUNTERFACTUAL")
        self.assertEqual(dict(parsed)["comparative_degree"].value, "GREATER")

    def test_standalone_only_and_not_only_preserve_focus_without_fake_condition(self):
        conditional = _window("Only if Lux misses Q can you walk.")
        conditional_catalog = generate_qualifier_candidates(conditional)
        self.assertIn(
            ("only if", QualifierKind.CONDITIONALITY),
            {(item.source_text.casefold(), item.kind) for item in conditional_catalog},
        )
        self.assertNotIn(
            ("only", QualifierKind.RESTRICTION),
            {(item.source_text.casefold(), item.kind) for item in conditional_catalog},
        )
        for text, cue_text, value in (
            ("Only Lux can walk.", "Only", "EXCLUSIVE"),
            ("Not only can Lux walk, she can run.", "Not only", "ADDITIVE"),
        ):
            with self.subTest(text=text):
                window = _window(text)
                node = _node(window, text="walk")
                catalog = qualifier_candidates_for_node(
                    window, node, generate_qualifier_candidates(window),
                )
                cue = next(
                    item for item in catalog
                    if item.source_text.casefold() == cue_text.casefold()
                    and item.kind is QualifierKind.RESTRICTION
                )
                fields = _empty_fields()
                fields["restriction"] = _asserted(value, cue.candidate_id)
                raw = json.dumps({"status": "OK", "qualifiers": fields})
                result = classify_node_qualifiers(
                    window, node, catalog, lambda **kwargs: raw,
                    model="pro", configuration={},
                )
                qualified = apply_node_qualifiers(window, node, catalog, result)
                self.assertEqual(qualified.qualifiers.restriction, Restriction(value))
                self.assertEqual(qualified.qualifiers.conditionality, Conditionality.UNKNOWN)
                self.assertEqual(
                    SemanticQualifiers.from_dict(qualified.qualifiers.to_dict()), qualified.qualifiers,
                )
                SemanticGraph.from_source_window(window, (qualified,))

    def test_prompt_disclaims_condition_flattening_and_assigns_numeric_ownership(self):
        window = _window()
        node = _node(window)
        catalog = qualifier_candidates_for_node(window, node, generate_qualifier_candidates(window))
        prompt = qualifier_prompt(window, node, catalog)
        self.assertIn("never replaces the antecedent/anchor mention", prompt)
        self.assertIn("CONDITION/TEMPORAL graph edge", prompt)
        self.assertIn("Quantities and durations are represented by source nodes", prompt)
        self.assertNotIn('"duration":', prompt)
        self.assertNotIn('"quantity":', prompt)


if __name__ == "__main__":
    unittest.main()
