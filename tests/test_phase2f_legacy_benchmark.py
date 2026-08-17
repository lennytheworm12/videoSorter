import hashlib
import json
from pathlib import Path
import unittest

from pipeline.semantic_edges import generate_candidate_edge_pairs
from pipeline.semantic_coreference import generate_coreference_candidate_sets
from pipeline.semantic_ir import ModelDecisionProvenance, SemanticNode, SourceSpan
from pipeline.semantic_ir_evaluation import load_semantic_benchmark
from pipeline.semantic_mentions import (
    MENTION_MAX_FOCAL_STARTS_PER_REQUEST,
    generate_mention_candidates,
    partition_candidate_catalog,
)
from pipeline.semantic_qualifiers import generate_qualifier_candidates
from pipeline.semantic_source import BronzeSource, window_from_exact_span


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/semantic_ir_legacy_manifest_v1.json"
BENCHMARK = ROOT / "data/semantic_ir_legacy_failure_v1.json"


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


class Phase2FLegacyBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.benchmark_raw = json.loads(BENCHMARK.read_text(encoding="utf-8"))
        cls.benchmark = load_semantic_benchmark(
            BENCHMARK, expected_split="LEGACY_FAILURE",
            expected_content_sha256=cls.benchmark_raw["content_sha256"],
            expected_pool_manifest_sha256=cls.manifest["content_sha256"],
        )

    def test_manifest_and_benchmark_are_hash_locked_to_valid_phase2e_evidence(self):
        manifest_inner = {
            key: value for key, value in self.manifest.items() if key != "content_sha256"
        }
        self.assertEqual(self.manifest["content_sha256"], _canonical_sha256(manifest_inner))
        self.assertEqual(
            self.manifest["input_hashes"]["phase2e_artifact_inner_sha256"],
            "04c185aaf324251b4733e76c87b2c71ea3946497f79a8956f268e88f28e2e17b",
        )
        self.assertEqual(
            self.manifest["input_hashes"]["phase2e_artifact_file_sha256"],
            "02725fb163ef752c98f51a070652ef5418a5b0d4916363d1c61c3071e957c808",
        )
        self.assertEqual(
            self.benchmark.pool_manifest_sha256, self.manifest["content_sha256"],
        )
        self.assertEqual(len(self.benchmark.cases), 5)

    def test_every_gold_span_is_exact_and_in_the_deterministic_catalog(self):
        manifest_by_case = {
            item["case_id"]: item for item in self.manifest["windows"]
        }
        for case in self.benchmark.cases:
            with self.subTest(case=case.case_id):
                retained = manifest_by_case[case.case_id]
                self.assertEqual(retained["source_text"], case.source_text)
                source = BronzeSource(case.source_id, case.source_text)
                window = window_from_exact_span(source, 0, len(source.text))
                mention_catalog = generate_mention_candidates(window)
                candidate_spans = {(item.start, item.end) for item in mention_catalog}
                for mention in case.mentions:
                    self.assertTrue(any(
                        span in candidate_spans for span in mention.acceptable_spans
                    ), mention.mention_id)
                    for start, end in mention.acceptable_spans:
                        self.assertEqual(case.source_text[start:end], source.text[start:end])
                qualifier_catalog = generate_qualifier_candidates(window)
                qualifier_spans = {
                    (item.kind.value, item.start, item.end) for item in qualifier_catalog
                }
                for qualifier in case.qualifiers:
                    self.assertTrue(all(
                        (qualifier.field.upper(), start, end) in qualifier_spans
                        for start, end in qualifier.cue_spans
                    ), qualifier.qualifier_id)

    def test_focal_requests_cover_every_gold_span_once_without_splitting_starts(self):
        covered = 0
        for case in self.benchmark.cases:
            window = window_from_exact_span(
                BronzeSource(case.source_id, case.source_text), 0, len(case.source_text),
            )
            catalog = generate_mention_candidates(window)
            partitions = partition_candidate_catalog(catalog, max_candidates=600)
            self.assertEqual(tuple(item for part in partitions for item in part), catalog)
            self.assertTrue(all(
                len({item.start for item in part}) <= MENTION_MAX_FOCAL_STARTS_PER_REQUEST
                for part in partitions
            ))
            start_partition = {}
            for partition_index, partition in enumerate(partitions):
                for candidate in partition:
                    previous = start_partition.setdefault(candidate.start, partition_index)
                    self.assertEqual(previous, partition_index)
            for mention in case.mentions:
                matching_partitions = {
                    partition_index
                    for partition_index, partition in enumerate(partitions)
                    if any(
                        (item.start, item.end) in mention.acceptable_spans
                        for item in partition
                    )
                }
                self.assertEqual(len(matching_partitions), 1, (case.case_id, mention.mention_id))
                covered += 1
        self.assertEqual(covered, 33)

    def test_every_gold_edge_is_offered_before_model_classification(self):
        for case in self.benchmark.cases:
            with self.subTest(case=case.case_id):
                source = BronzeSource(case.source_id, case.source_text)
                window = window_from_exact_span(source, 0, len(source.text))
                nodes = []
                node_by_mention = {}
                for mention in case.mentions:
                    start, end = mention.acceptable_spans[0]
                    span = SourceSpan(
                        window.source_id, window.window_id, start, end,
                        window.text[start:end], start, end,
                    )
                    provenance = ModelDecisionProvenance.create(
                        "gold:" + mention.mention_id, "reviewed-gold", "legacy-v1",
                        configuration={"fixture": case.case_id},
                        model_input={"span": [start, end]}, model_output={"reviewed": True},
                        candidate_ids=("gold:" + mention.mention_id,),
                    )
                    node = SemanticNode(mention.node_types[0], span, provenance)
                    nodes.append(node)
                    node_by_mention[mention.mention_id] = node
                pairs = generate_candidate_edge_pairs(window, tuple(nodes))
                offered = {
                    (pair.source_node_id, pair.target_node_id, edge_type)
                    for pair in pairs for edge_type in pair.allowed_edge_types
                }
                for edge in case.edges:
                    self.assertTrue(any(
                        (
                            node_by_mention[edge.source_mention_id].node_id,
                            node_by_mention[edge.target_mention_id].node_id,
                            edge_type,
                        ) in offered
                        for edge_type in edge.edge_types
                    ), edge.edge_id)

    def test_every_accepted_gold_type_alternative_can_reach_its_gold_edge(self):
        for case in self.benchmark.cases:
            source = BronzeSource(case.source_id, case.source_text)
            window = window_from_exact_span(source, 0, len(source.text))
            mentions = {item.mention_id: item for item in case.mentions}
            for edge in case.edges:
                left = mentions[edge.source_mention_id]
                right = mentions[edge.target_mention_id]
                for left_type in left.node_types:
                    for right_type in right.node_types:
                        nodes = []
                        for mention, node_type in ((left, left_type), (right, right_type)):
                            start, end = mention.acceptable_spans[0]
                            provenance = ModelDecisionProvenance.create(
                                "gold:" + mention.mention_id + ":" + node_type.value,
                                "reviewed-gold", "legacy-v1",
                                configuration={"fixture": case.case_id},
                                model_input={"span": [start, end]},
                                model_output={"reviewed": True},
                                candidate_ids=("gold:" + mention.mention_id,),
                            )
                            nodes.append(SemanticNode(
                                node_type,
                                SourceSpan(
                                    window.source_id, window.window_id, start, end,
                                    window.text[start:end], start, end,
                                ),
                                provenance,
                            ))
                        offered = {
                            item
                            for pair in generate_candidate_edge_pairs(window, tuple(nodes))
                            if pair.source_node_id == nodes[0].node_id
                            and pair.target_node_id == nodes[1].node_id
                            for item in pair.allowed_edge_types
                        }
                        self.assertTrue(
                            set(edge.edge_types) & offered,
                            (case.case_id, edge.edge_id, left_type.value, right_type.value),
                        )

    def test_gold_preserves_removal_conjunction_and_negative_condition_semantics(self):
        cases = {item.case_id: item for item in self.benchmark.cases}
        sweeper = cases["sweeper-limits-mid-play"]
        sweeper_mentions = {item.mention_id: item for item in sweeper.mentions}
        sweeper_edges = {
            (item.source_mention_id, item.target_mention_id, tuple(t.value for t in item.edge_types))
            for item in sweeper.edges
        }
        self.assertEqual(
            sweeper.source_text[sweeper_mentions["remove_play"].acceptable_spans[0][0]:
                                 sweeper_mentions["remove_play"].acceptable_spans[0][1]],
            "remove their ability to play on Mid",
        )
        self.assertIn(("use", "remove_play", ("PURPOSE",)), sweeper_edges)
        self.assertIn(("play_ability", "remove_play", ("OBJECT",)), sweeper_edges)
        self.assertNotIn(("use", "play_ability", ("PURPOSE",)), sweeper_edges)

        hook = cases["unwarded-bush-hook-risk"]
        hook_mentions = {item.mention_id: item for item in hook.mentions}
        hook_edges = {
            (item.source_mention_id, item.target_mention_id, tuple(t.value for t in item.edge_types))
            for item in hook.edges
        }
        condition_start, condition_end = hook_mentions["win_condition"].acceptable_spans[0]
        self.assertEqual(
            hook.source_text[condition_start:condition_end],
            "get hooked and land double Q on them",
        )
        self.assertIn(("win_condition", "win", ("CONDITION",)), hook_edges)
        self.assertIn(("step_hook", "ward", ("CONDITION",)), hook_edges)
        self.assertNotIn(("step_hook", "ward", ("PREVENTS",)), hook_edges)
        negative = next(item for item in hook.qualifiers if item.qualifier_id == "avoid_hook_negative")
        self.assertEqual(negative.value, "NEGATIVE")
        self.assertEqual(
            [hook.source_text[start:end] for start, end in negative.cue_spans], ["without"],
        )

    def test_gold_includes_actor_scope_reference_status_and_type_alternatives(self):
        cases = {item.case_id: item for item in self.benchmark.cases}
        wave_edges = {
            (item.source_mention_id, item.target_mention_id, tuple(t.value for t in item.edge_types))
            for item in cases["wave-reset-after-kill"].edges
        }
        self.assertIn(("kill_actor", "kill", ("ACTOR",)), wave_edges)
        push_edges = {
            (item.source_mention_id, item.target_mention_id, tuple(t.value for t in item.edge_types))
            for item in cases["push-poke-wave-crash"].edges
        }
        self.assertIn(("crash_actor", "crash", ("ACTOR",)), push_edges)
        for case in cases.values():
            referenced = {item.source_mention_id for item in case.references}
            for mention in case.mentions:
                text = case.source_text[
                    mention.acceptable_spans[0][0]:mention.acceptable_spans[0][1]
                ].casefold()
                if text in {"you", "you guys"}:
                    self.assertIn(mention.mention_id, referenced, (case.case_id, mention.mention_id))
            for reference in case.references:
                self.assertEqual(reference.status, "INSUFFICIENT_EVIDENCE")
                self.assertFalse(reference.target_mention_ids)
        push = next(
            item for item in cases["mid-push-prevents-side-collapse"].mentions
            if item.mention_id == "push"
        )
        self.assertEqual(
            {item.value for item in push.node_types}, {"ACTION", "OUTCOME", "STATE"},
        )

    def test_every_reviewed_reference_has_a_deterministic_candidate_set(self):
        for case in self.benchmark.cases:
            source = BronzeSource(case.source_id, case.source_text)
            window = window_from_exact_span(source, 0, len(source.text))
            nodes = []
            node_by_mention = {}
            for mention in case.mentions:
                start, end = mention.acceptable_spans[0]
                span = SourceSpan(
                    window.source_id, window.window_id, start, end,
                    window.text[start:end], start, end,
                )
                provenance = ModelDecisionProvenance.create(
                    "gold:" + mention.mention_id, "reviewed-gold", "legacy-v1",
                    configuration={"fixture": case.case_id},
                    model_input={"span": [start, end]}, model_output={"reviewed": True},
                    candidate_ids=("gold:" + mention.mention_id,),
                )
                node = SemanticNode(mention.node_types[0], span, provenance)
                nodes.append(node)
                node_by_mention[mention.mention_id] = node
            sets = generate_coreference_candidate_sets(window, tuple(nodes))
            source_ids = {item.source_node_id for item in sets}
            for reference in case.references:
                self.assertIn(
                    node_by_mention[reference.source_mention_id].node_id, source_ids,
                    (case.case_id, reference.reference_id),
                )


if __name__ == "__main__":
    unittest.main()
