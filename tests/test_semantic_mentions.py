import json
from dataclasses import replace
import unittest

from pipeline.semantic_mentions import (
    MentionSelection,
    candidate_coverage,
    assemble_semantic_nodes,
    generate_mention_candidates,
    parse_mention_selection,
    partition_candidate_catalog,
    select_mentions,
    select_mention_catalog,
)
from pipeline.semantic_source import BronzeSource, window_from_exact_span


def _window(text: str):
    source = BronzeSource("transcript:test", text)
    return window_from_exact_span(source, 0, len(text))


class SemanticMentionTests(unittest.TestCase):
    def test_catalog_contains_exact_pronoun_ability_event_action_state_and_time(self):
        window = _window("When Lux misses Q, you can walk forward because she can't stop your advance.")
        catalog = generate_mention_candidates(window, entity_aliases=("Lux",), ability_aliases=("Lux Q", "Q"))
        texts = {item.source_text: set(item.type_hints) for item in catalog}
        self.assertIn("Lux", texts)
        self.assertIn("misses Q", texts)
        self.assertIn("you", texts)
        self.assertIn("walk forward", texts)
        self.assertIn("can't stop your advance", texts)
        self.assertIn("When Lux misses Q", texts)
        self.assertIn("ENTITY", texts["you"])
        self.assertIn("ABILITY_OR_RESOURCE", texts["Q"])

    def test_repeated_identical_phrase_produces_distinct_exact_candidates(self):
        window = _window("ward bush, then ward bush again.")
        catalog = generate_mention_candidates(window)
        wards = [item for item in catalog if item.source_text == "ward bush"]
        self.assertEqual(len(wards), 2)
        self.assertNotEqual(wards[0].candidate_id, wards[1].candidate_id)
        self.assertNotEqual(wards[0].start, wards[1].start)

    def test_catalog_is_deterministic_and_source_anchored(self):
        window = _window("If Q misses, don't walk up.")
        first = generate_mention_candidates(window)
        second = generate_mention_candidates(window)
        self.assertEqual(first, second)
        for candidate in first:
            candidate.validate(window)

    def test_catalog_covers_long_source_mentions_up_to_versioned_pass0_bound(self):
        text = " ".join(f"word{index}" for index in range(20))
        window = _window(text)
        catalog = generate_mention_candidates(window)
        self.assertIn(text, {item.source_text for item in catalog})
        with self.assertRaises(ValueError):
            generate_mention_candidates(window, max_ngram_words=10)

    def test_candidate_identity_and_runtime_shapes_are_fail_closed(self):
        window = _window("walk forward")
        candidate = next(item for item in generate_mention_candidates(window) if item.source_text == "walk")
        for invalid in (
            replace(candidate, candidate_id="forged"),
            replace(candidate, start=False),
            replace(candidate, type_hints=list(candidate.type_hints)),
            replace(candidate, segment_ids=()),
            replace(candidate, version="old"),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                invalid.validate(window)

    def test_partition_drops_nothing(self):
        catalog = generate_mention_candidates(_window("one two three four five six seven"))
        parts = partition_candidate_catalog(catalog, max_candidates=5)
        self.assertEqual(tuple(item for part in parts for item in part), catalog)

    def test_parse_selection_accepts_unresolved_pronoun_without_guessing_reference(self):
        window = _window("When Lux misses Q, she cannot stop you.")
        catalog = generate_mention_candidates(window, entity_aliases=("Lux",), ability_aliases=("Q",))
        by_text = {item.source_text: item for item in catalog}
        raw = json.dumps({"status": "OK", "mentions": [{
            "candidate_id": by_text["she"].candidate_id,
            "node_type": "ENTITY",
            "confidence": 0.55,
            "ambiguity": "AMBIGUOUS",
        }]})
        status, mentions, _ = parse_mention_selection(raw, catalog)
        self.assertEqual(status, "OK")
        self.assertEqual(mentions[0].ambiguity, "AMBIGUOUS")

    def test_parse_rejects_unknown_id_type_duplicate_key_and_bad_confidence(self):
        catalog = generate_mention_candidates(_window("walk forward"))
        base = {"status": "OK", "mentions": [{
            "candidate_id": catalog[0].candidate_id, "node_type": "ACTION", "confidence": 0.9,
            "ambiguity": "NONE",
        }]}
        variants = []
        unknown = json.loads(json.dumps(base)); unknown["mentions"][0]["candidate_id"] = "missing"
        variants.append(json.dumps(unknown))
        bad_type = json.loads(json.dumps(base)); bad_type["mentions"][0]["node_type"] = "ACCESS"
        variants.append(json.dumps(bad_type))
        bad_conf = json.loads(json.dumps(base)); bad_conf["mentions"][0]["confidence"] = 2
        variants.append(json.dumps(bad_conf))
        variants.append('{"status":"OK","status":"NONE","mentions":[]}')
        for raw in variants:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_mention_selection(raw, catalog)

    def test_non_ok_status_cannot_smuggle_mentions(self):
        catalog = generate_mention_candidates(_window("walk"))
        raw = json.dumps({"status": "AMBIGUOUS", "mentions": [{
            "candidate_id": catalog[0].candidate_id, "node_type": "ACTION", "confidence": 0.5,
            "ambiguity": "AMBIGUOUS",
        }]})
        with self.assertRaises(ValueError):
            parse_mention_selection(raw, catalog)

    def test_provider_failure_is_distinct_from_parse_failure(self):
        window = _window("walk forward")
        catalog = generate_mention_candidates(window)

        def fail(**kwargs):
            raise RuntimeError("offline")

        provider = select_mentions(window, catalog, fail)
        parsed = select_mentions(window, catalog, lambda **kwargs: "not json")
        self.assertTrue(provider.failure.startswith("MentionProviderError:"))
        self.assertEqual(parsed.failure, "ValueError")
        self.assertEqual(parsed.raw_output, "not json")

    def test_coverage_is_reported_by_family(self):
        window = _window("When Lux misses Q, walk forward.")
        catalog = generate_mention_candidates(window)
        spans = []
        for text, node_type in (("Lux", "ENTITY"), ("Q", "ABILITY_OR_RESOURCE"), ("walk forward", "ACTION")):
            start = window.text.index(text)
            spans.append((start, start + len(text), node_type))
        report = candidate_coverage(catalog, spans, window=window)
        self.assertEqual(report["entity"]["recall"], 1.0)
        self.assertEqual(report["ability_resource"]["recall"], 1.0)
        self.assertEqual(report["action_event"]["recall"], 1.0)
        self.assertEqual(report["negation"], {"hit_count": 0, "denominator": 0, "recall": 0.0})

    def test_arbitrary_qualifiers_and_duplicate_typing_are_rejected(self):
        catalog = generate_mention_candidates(_window("walk forward"))
        item = catalog[0]
        with self.assertRaises(ValueError):
            parse_mention_selection(json.dumps({"status": "OK", "mentions": [{
                "candidate_id": item.candidate_id, "node_type": "ACTION", "confidence": 0.8,
                "ambiguity": "NONE", "qualifiers": {"concept": "access"},
            }]}), catalog)
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            parse_mention_selection(json.dumps({"status": "OK", "mentions": [
                {"candidate_id": item.candidate_id, "node_type": "ACTION", "confidence": 0.8, "ambiguity": "NONE"},
                {"candidate_id": item.candidate_id, "node_type": "EVENT", "confidence": 0.7, "ambiguity": "UNKNOWN"},
            ]}), catalog)

    def test_partition_aggregation_retains_catalog_failures_and_builds_exact_nodes(self):
        window = _window("Lux walks forward then she stops")
        catalog = generate_mention_candidates(window, entity_aliases=("Lux",))

        def choose(**kwargs):
            ids = [entry["id"] for entry in json.loads(kwargs["user"].split("CANDIDATES:\n", 1)[1].split("\nSelect", 1)[0])]
            return json.dumps({"status": "OK", "mentions": [{
                "candidate_id": ids[0], "node_type": "ENTITY", "confidence": 0.75, "ambiguity": "NONE",
            }]})

        result = select_mention_catalog(
            window, catalog, choose, model="reference-model", configuration={"temperature": 0},
            max_candidates=10,
        )
        self.assertEqual(result.status, "OK")
        self.assertEqual(result.catalog, catalog)
        self.assertGreater(len(result.partition_results), 1)
        nodes = assemble_semantic_nodes(window, result)
        self.assertEqual(len(nodes), len(result.partition_results))
        self.assertTrue(all(node.confidence == 0.75 for node in nodes))
        self.assertTrue(all(node.source_span.text == window.text[node.source_span.start:node.source_span.end] for node in nodes))
        request = json.loads(result.partition_results[0].request_json)
        self.assertEqual(request["system"].split(".")[0], "Return strict JSON only")
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["max_tokens"], 2048)

    def test_partition_abstention_remains_partial_and_aggregate_tampering_rejects(self):
        window = _window("Lux walks forward then she stops")
        catalog = generate_mention_candidates(window, entity_aliases=("Lux",))
        calls = 0

        def partial(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                candidate_id = json.loads(
                    kwargs["user"].split("CANDIDATES:\n", 1)[1].split("\nSelect", 1)[0]
                )[0]["id"]
                return json.dumps({"status": "OK", "mentions": [{
                    "candidate_id": candidate_id, "node_type": "ENTITY",
                    "confidence": 0.1, "ambiguity": "NONE",
                }]})
            return json.dumps({"status": "AMBIGUOUS", "mentions": []})

        result = select_mention_catalog(
            window, catalog, partial, model="reference", configuration={}, max_candidates=10,
        )
        self.assertEqual(result.status, "PARTIAL")
        self.assertTrue(result.abstentions)
        forged = replace(result, mentions=(MentionSelection(
            result.mentions[0].candidate_id, "ACTION", 0.99, "NONE",
        ),))
        with self.assertRaisesRegex(ValueError, "contradict"):
            assemble_semantic_nodes(window, forged)

        partition = result.partition_results[0]
        forged_mention = MentionSelection(partition.mentions[0].candidate_id, "ACTION", 0.99, "NONE")
        forged_partition = replace(
            partition, mentions=(forged_mention,),
            parsed_output={"status": "OK", "mentions": [{
                "candidate_id": forged_mention.candidate_id, "node_type": "ACTION",
                "confidence": 0.99, "ambiguity": "NONE",
            }]},
        )
        forged_results = (forged_partition,) + result.partition_results[1:]
        forged_run = replace(result, partition_results=forged_results, mentions=(forged_mention,))
        with self.assertRaisesRegex(ValueError, "raw output contradicts"):
            assemble_semantic_nodes(window, forged_run)

    def test_selection_rejects_candidates_from_another_window(self):
        first = _window("walk")
        second = _window("stop")
        catalog = generate_mention_candidates(first)
        with self.assertRaises(ValueError):
            select_mentions(second, catalog, lambda **kwargs: "{}")

    def test_unicode_percent_and_possessive_alias_candidates(self):
        window = _window("Lux's chance is 15% near éclair, not éLuxé.")
        catalog = generate_mention_candidates(window, entity_aliases=("Lux",))
        texts = {item.source_text for item in catalog}
        self.assertIn("Lux", texts)
        self.assertIn("15%", texts)
        self.assertIn("éclair", texts)
        self.assertNotIn("éLux", texts)


if __name__ == "__main__":
    unittest.main()
