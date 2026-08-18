import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pipeline.semantic_compiler import SemanticCompilerConfig, compile_source_semantic_ir
import pipeline.semantic_compiler as semantic_compiler
from pipeline.semantic_coreference import COREFERENCE_SYSTEM
from pipeline.semantic_ir import EdgeType, NodeType
from pipeline.semantic_ir_evaluation import (
    BENCHMARK_SCHEMA_VERSION, GoldEdge, GoldMention, GoldQualifier, GoldReference,
    SemanticBenchmark, SemanticBenchmarkCase, SemanticQuestion,
    evaluate_semantic_benchmark, evaluate_semantic_case,
    load_semantic_benchmark, verify_benchmark_isolation,
    _case_to_dict, _first_loss,
)
from pipeline.semantic_ir_pool import (
    POOL_PHENOMENA, POOL_SCHEMA_VERSION, detect_pool_phenomena,
)
from tests.test_semantic_compiler import ScriptedSemanticModel, _window


def _run():
    config = SemanticCompilerConfig.create(
        "reference-pro", provider_configuration={"provider": "scripted"},
        mention_partition_size=1000,
    )
    return compile_source_semantic_ir(_window(), ScriptedSemanticModel(), config=config)


def _case(*, exhaustive=True):
    mentions = (
        GoldMention("lux", (NodeType.ENTITY,), ((0, 3),), True),
        GoldMention("walks", (NodeType.ACTION,), ((4, 9),), True),
        GoldMention("she", (NodeType.ENTITY,), ((11, 14),), True),
        GoldMention("waits", (NodeType.ACTION,), ((15, 20),), True),
    )
    edges = (
        GoldEdge("actor_walks", "lux", "walks", (EdgeType.ACTOR,), True),
        GoldEdge("actor_waits", "lux", "waits", (EdgeType.ACTOR,), True),
    )
    references = (GoldReference("she_lux", "she", "RESOLVED", ("lux",), True),)
    questions = (
        SemanticQuestion("lux", "Who is named?", "entity_recovery", ("mention:lux",), True),
        SemanticQuestion("walks", "What action occurs?", "action_recovery", ("mention:walks",), True),
        SemanticQuestion("she", "What reference expression occurs?", "entity_recovery", ("mention:she",), True),
        SemanticQuestion("waits", "What later action occurs?", "action_recovery", ("mention:waits",), True),
        SemanticQuestion("who_walks", "Who walks?", "actor_target_roles", ("edge:actor_walks",), True),
        SemanticQuestion("who_waits", "Who waits?", "actor_target_roles", ("edge:actor_waits",), True),
        SemanticQuestion("who_is_she", "What does She refer to?", "coreference", ("reference:she_lux",), True),
    )
    return SemanticBenchmarkCase(
        "case-1", "DEV", "transcript:compiler", "transcript", "Lux walks. She waits.",
        "compiler", 0, 21, ("roles", "pronouns"), exhaustive,
        mentions, edges, (), references, questions,
    )


_RICH = (
    "If Lux misses Q and when Ahri uses W, you should push two waves before dragon "
    "because mana is lower and therefore they cannot contest. Push river instead, "
    "but maybe do not wait but move behind tower now."
)


def _test_pool_manifest(primary_id, text, start=0):
    champion_names = ["Ahri", "Garen", "Jinx", "Lee Sin", "Lux"]
    sources = [
        (primary_id, text, start),
        ("coverage-rich", _RICH, 0),
        ("coverage-implicit", "You should hold the lane while the tower is dangerous and your team prepares the next careful play nearby today", 0),
        ("coverage-fact", "Jinx has a long attack range and deals steady damage to targets in the bottom lane during ordinary team fights around the map", 0),
        ("coverage-asr", "push the wave move river hold vision save flash track cooldown take space respect range keep health use wards avoid danger and look for angles", 0),
    ]
    # Preserve a single record when the primary is already the rich coverage source.
    unique = []
    seen = set()
    for source_id, source_text, source_start in sources:
        if source_id in seen:
            continue
        seen.add(source_id)
        unique.append((source_id, source_text, source_start))
    windows = []
    counts = {key: 0 for key in POOL_PHENOMENA}
    for index, (source_id, source_text, source_start) in enumerate(unique, 1):
        source_end = source_start + len(source_text)
        phenomena = list(detect_pool_phenomena(source_text, champion_names))
        for phenomenon in phenomena:
            counts[phenomenon] += 1
        identity = hashlib.sha256(
            f"{source_id}:{source_start}:{source_end}:{source_text}".encode(),
        ).hexdigest()[:20]
        windows.append({
            "pool_index": index, "window_id": f"pool:{source_id}:w00001-{identity}",
            "source_id": "transcript:" + source_id, "source_kind": "transcript",
            "upstream_source_id": source_id, "upstream_start": source_start,
            "upstream_end": source_end, "source_text": source_text,
            "token_offset": 0, "source_window_ordinal": 1,
            "source_text_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
            "upstream_content_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
            "phenomena": phenomena,
            "metadata": {"video_title": "Fixture", "role": "mid", "champion": "Lux"},
        })
    windows.sort(key=lambda item: item["window_id"])
    for index, item in enumerate(windows, 1):
        item["pool_index"] = index
    inner = {
        "schema_version": POOL_SCHEMA_VERSION, "purpose": "Test-only source-exact pool",
        "selection_policy": {
            "target_count": len(windows), "target_words": 48, "stride_words": 40,
            "minimum_per_phenomenon": 1, "one_window_per_upstream_source": True,
            "champion_names": champion_names,
            "excluded_phase2b_sources": [], "excluded_phase2d_sources": [],
        },
        "input_hashes": {
            "database_sha256": "d" * 64, "frozen_fixture_sha256": "f" * 64,
            "development_fixture_sha256": "e" * 64,
        },
        "phenomenon_counts": dict(sorted(counts.items())), "windows": windows,
    }
    digest = hashlib.sha256(json.dumps(
        inner, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    return {"content_sha256": digest, **inner}


def _fixture_pool(start=0):
    return _test_pool_manifest("fixture", "Lux acts when Q lands because it enables movement.", start)


def _fixture_payload(split="DEV", start=0):
    text = "Lux acts when Q lands because it enables movement."
    lux = [text.index("Lux"), text.index("Lux") + len("Lux")]
    action = [text.index("acts"), text.index("acts") + len("acts")]
    event = [text.index("Q lands"), text.index("Q lands") + len("Q lands")]
    case = {
        "id": "fixture-case", "split": split, "source_id": "transcript:fixture",
        "source_kind": "transcript", "source_text": text,
        "upstream_source_id": "fixture", "upstream_start": start,
        "upstream_end": start + len(text), "phenomena": ["direct_advice"],
        "exhaustive": True,
        "mentions": [
            {"id": "lux", "node_types": ["ENTITY"], "acceptable_spans": [lux], "critical": True},
            {"id": "action", "node_types": ["ACTION"], "acceptable_spans": [action], "critical": True},
            {"id": "event", "node_types": ["EVENT"], "acceptable_spans": [event], "critical": True},
        ],
        "edges": [
            {"id": "condition", "source": "event", "target": "action", "edge_types": ["CONDITION"], "critical": True},
            {"id": "cause", "source": "event", "target": "action", "edge_types": ["ENABLES"], "critical": True},
        ],
        "qualifiers": [], "references": [],
        "questions": [
            {"id": "q-entity", "prompt": "Who is named?", "dimension": "entity_recovery", "requires": ["mention:lux"], "critical": True},
            {"id": "q-action", "prompt": "What action occurs?", "dimension": "action_recovery", "requires": ["mention:action"], "critical": True},
            {"id": "q-event", "prompt": "What event occurs?", "dimension": "event_recovery", "requires": ["mention:event"], "critical": True},
            {"id": "q-condition", "prompt": "What is conditional?", "dimension": "condition_recovery", "requires": ["edge:condition"], "critical": True},
            {"id": "q-cause", "prompt": "What enables the action?", "dimension": "causal_edges", "requires": ["edge:cause"], "critical": True},
        ],
    }
    inner = {
        "schema_version": BENCHMARK_SCHEMA_VERSION, "split": split,
        "purpose": "Reviewed test fixture",
        "pool_manifest_sha256": _fixture_pool(start)["content_sha256"],
        "cases": [case],
    }
    digest = hashlib.sha256(json.dumps(
        inner, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    return {"content_sha256": digest, **inner}


def _pool_for_case(case):
    return _test_pool_manifest(
        case.upstream_source_id, case.source_text, case.upstream_start,
    )


def _benchmark_for_case(case):
    pool = _pool_for_case(case)
    inner = {
        "schema_version": BENCHMARK_SCHEMA_VERSION, "split": case.split,
        "purpose": "Typed aggregate test", "pool_manifest_sha256": pool["content_sha256"],
        "cases": [_case_to_dict(case)],
    }
    digest = hashlib.sha256(json.dumps(
        inner, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    return SemanticBenchmark(case.split, inner["purpose"], pool["content_sha256"], (case,), digest), pool


class SemanticIREvaluationTests(unittest.TestCase):
    def test_perfect_case_separates_catalog_selection_types_edges_and_checksum(self):
        result = evaluate_semantic_case(_case(), _run())
        for field in (
            "mention_candidate_coverage", "mention_selection_recall", "mention_type_recall",
            "edge_pair_coverage", "edge_recall", "reference_candidate_coverage",
            "reference_recall", "semantic_completeness", "semantic_checksum",
        ):
            self.assertEqual(result[field]["rate"], 1.0, field)
        self.assertTrue(result["source_span_validity"])
        self.assertTrue(result["edge_provenance_traceability"])
        self.assertEqual(result["unsupported_nodes"], [])
        self.assertEqual(result["unsupported_edges"], [])
        self.assertIsNone(result["first_loss"])

    def test_candidate_selection_type_and_condition_losses_are_localized(self):
        run = _run()
        base = _case(exhaustive=False)
        period = GoldMention("period", (NodeType.EVENT,), ((9, 10),), True)
        wrong_type = GoldMention("she_event", (NodeType.EVENT,), ((11, 14),), True)
        condition = GoldEdge("condition", "walks", "waits", (EdgeType.CONDITION,), True)
        case = SemanticBenchmarkCase(
            "losses", "DEV", base.source_id, base.source_kind, base.source_text,
            base.upstream_source_id, base.upstream_start, base.upstream_end,
            ("condition",), False, (base.mentions[0], base.mentions[1], base.mentions[3], period, wrong_type),
            (condition,), (), (),
            (
                SemanticQuestion("lux", "Who?", "entity_recovery", ("mention:lux",), True),
                SemanticQuestion("walks", "What action?", "action_recovery", ("mention:walks",), True),
                SemanticQuestion("waits", "What later action?", "action_recovery", ("mention:waits",), True),
                SemanticQuestion("period", "What event?", "event_recovery", ("mention:period",), True),
                SemanticQuestion("she_event", "What other event?", "event_recovery", ("mention:she_event",), True),
                SemanticQuestion("condition_q", "Under what condition?", "condition_recovery", ("edge:condition",), True),
            ),
        )
        result = evaluate_semantic_case(case, run)
        codes = {item["code"] for item in result["failures"]}
        self.assertIn("MENTION_CANDIDATE_MISSING", codes)
        self.assertIn("MENTION_TYPE_ERROR", codes)
        self.assertIn("CONDITION_LOSS", codes)
        self.assertEqual(result["mention_selection_recall"]["hit_count"], 4)
        self.assertEqual(result["mention_type_recall"]["hit_count"], 3)
        self.assertEqual(result["edge_pair_coverage"]["denominator"], 1)

    def test_nonexhaustive_case_never_labels_unreviewed_output_as_invention(self):
        base = _case(exhaustive=False)
        case = SemanticBenchmarkCase(
            "partial-label", "DEV", base.source_id, base.source_kind, base.source_text,
            base.upstream_source_id, base.upstream_start, base.upstream_end,
            base.phenomena, False, (base.mentions[0],), (), (), (),
            (SemanticQuestion("entity", "Who?", "entity_recovery", ("mention:lux",), True),),
        )
        result = evaluate_semantic_case(case, _run())
        self.assertEqual(result["unsupported_nodes"], [])
        self.assertTrue(result["unscored_nodes"])
        self.assertNotIn("UNSUPPORTED_NODE", {item["code"] for item in result["failures"]})

    def test_partial_compiler_run_is_scored_without_crashing_and_keeps_failure(self):
        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )
        with patch(
            "pipeline.semantic_compiler.classify_node_qualifiers",
            side_effect=RuntimeError("injected"),
        ):
            run = compile_source_semantic_ir(_window(), ScriptedSemanticModel(), config=config)
        result = evaluate_semantic_case(_case(exhaustive=False), run)
        self.assertEqual(result["status"], "FAILURE")
        self.assertIn("ASSEMBLY_FAILURE", {item["code"] for item in result["failures"]})
        self.assertEqual(result["mention_type_recall"]["rate"], 1.0)
        self.assertEqual(result["edge_recall"]["rate"], 0.0)

    def test_benchmark_aggregate_keeps_independent_denominators(self):
        case = replace(_case(), split="LEGACY_FAILURE")
        benchmark, pool = _benchmark_for_case(case)
        result = evaluate_semantic_benchmark(
            benchmark, {case.case_id: _run()},
            expected_content_sha256=benchmark.content_sha256, pool_manifest=pool,
        )
        self.assertEqual(result["semantic_checksum"]["rate"], 1.0)
        self.assertEqual(result["mention_candidate_coverage"]["denominator"], 4)
        self.assertEqual(result["edge_pair_coverage"]["denominator"], 2)
        self.assertEqual(result["dimensions"]["coreference"]["rate"], 1.0)
        self.assertEqual(result["mention_families"]["ENTITY"]["denominator"], 2)

    def test_fixture_hash_shapes_split_and_closed_qualifier_values_fail_closed(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dev.json"
            payload = _fixture_payload()
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_semantic_benchmark(
                path, expected_split="DEV",
                expected_content_sha256=payload["content_sha256"],
                expected_pool_manifest_sha256=payload["pool_manifest_sha256"],
                pool_manifest=_fixture_pool(),
            )
            self.assertEqual(len(loaded.cases), 1)
            tampered = copy.deepcopy(payload)
            tampered["purpose"] = "changed"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_semantic_benchmark(
                    path, expected_split="DEV",
                    expected_content_sha256=payload["content_sha256"],
                    expected_pool_manifest_sha256=payload["pool_manifest_sha256"],
                    pool_manifest=_fixture_pool(),
                )
            with self.assertRaises(ValueError):
                load_semantic_benchmark(
                    path, expected_split="DEV", expected_content_sha256="b" * 64,
                    expected_pool_manifest_sha256=payload["pool_manifest_sha256"],
                    pool_manifest=_fixture_pool(),
                )

            bad = _fixture_payload()
            bad["cases"][0]["qualifiers"] = [{
                "id": "q", "mention": "lux", "field": "polarity",
                "value": "INVENTED", "cue_spans": [[0, 3]], "critical": True,
            }]
            inner = {key: value for key, value in bad.items() if key != "content_sha256"}
            bad["content_sha256"] = hashlib.sha256(json.dumps(
                inner, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode()).hexdigest()
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_semantic_benchmark(
                    path, expected_split="DEV",
                    expected_content_sha256=bad["content_sha256"],
                    expected_pool_manifest_sha256=bad["pool_manifest_sha256"],
                    pool_manifest=_fixture_pool(),
                )

    def test_split_isolation_rejects_any_overlapping_upstream_span(self):
        with TemporaryDirectory() as directory:
            dev_path = Path(directory) / "dev.json"
            dev_path.write_text(json.dumps(_fixture_payload("DEV", 0)), encoding="utf-8")
            dev_payload = _fixture_payload("DEV", 0)
            dev = load_semantic_benchmark(
                dev_path, expected_split="DEV",
                expected_content_sha256=dev_payload["content_sha256"],
                expected_pool_manifest_sha256=dev_payload["pool_manifest_sha256"],
                pool_manifest=_fixture_pool(0),
            )
            frozen_case = replace(
                dev.cases[0], case_id="frozen-case", split="FROZEN_EVAL",
                source_id="subtitle:fixture", source_kind="subtitle",
            )
            frozen = SemanticBenchmark(
                "FROZEN_EVAL", "frozen", dev.pool_manifest_sha256,
                (frozen_case,), "f" * 64,
            )
            with self.assertRaises(ValueError):
                verify_benchmark_isolation(dev, frozen)
            with self.assertRaises(ValueError):
                load_semantic_benchmark(
                    dev_path, expected_split="DEV",
                    expected_content_sha256=dev_payload["content_sha256"],
                    expected_pool_manifest_sha256=dev_payload["pool_manifest_sha256"],
                    pool_manifest=_fixture_pool(0),
                    prohibited_upstream_sources={"fixture"},
                )

    def test_provider_failure_cannot_receive_reference_or_checksum_credit(self):
        model = ScriptedSemanticModel()

        def timeout_coreference(**kwargs):
            if kwargs["system"] == COREFERENCE_SYSTEM:
                raise TimeoutError("offline")
            return model(**kwargs)

        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )
        run = compile_source_semantic_ir(_window(), timeout_coreference, config=config)
        base = _case(exhaustive=False)
        reference = GoldReference("unresolved", "she", "INSUFFICIENT_EVIDENCE", (), True)
        case = SemanticBenchmarkCase(
            "provider-ref", "DEV", base.source_id, base.source_kind, base.source_text,
            base.upstream_source_id, base.upstream_start, base.upstream_end,
            ("pronouns",), False, base.mentions, (), (), (reference,),
            (
                SemanticQuestion("lux", "Who?", "entity_recovery", ("mention:lux",), True),
                SemanticQuestion("walks", "What action?", "action_recovery", ("mention:walks",), True),
                SemanticQuestion("she", "What expression?", "entity_recovery", ("mention:she",), True),
                SemanticQuestion("waits", "What later action?", "action_recovery", ("mention:waits",), True),
                SemanticQuestion("reference", "Can the reference be resolved?", "coreference", ("reference:unresolved",), True),
            ),
        )
        result = evaluate_semantic_case(case, run)
        self.assertEqual(result["reference_recall"]["rate"], 0.0)
        self.assertFalse(next(
            item["answerable_from_ir"] for item in result["questions"]
            if item["question_id"] == "reference"
        ))
        self.assertLess(result["semantic_checksum"]["rate"], 1.0)
        self.assertEqual(result["first_loss"], "PROVIDER_FAILURE")

    def test_partial_qualifier_prefix_does_not_erase_selected_mentions(self):
        original = semantic_compiler.classify_node_qualifiers
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected")
            return original(*args, **kwargs)

        config = SemanticCompilerConfig.create(
            "reference-pro", provider_configuration={"provider": "scripted"},
            mention_partition_size=1000,
        )
        with patch(
            "pipeline.semantic_compiler.classify_node_qualifiers",
            side_effect=fail_second,
        ):
            run = compile_source_semantic_ir(
                _window(), ScriptedSemanticModel(), config=config,
            )
        result = evaluate_semantic_case(_case(exhaustive=False), run)
        self.assertEqual(len(run.qualified_nodes), 1)
        self.assertEqual(result["mention_selection_recall"]["rate"], 1.0)
        self.assertEqual(result["mention_type_recall"]["rate"], 1.0)

    def test_checksum_fixture_cannot_omit_reviewed_facts_or_reseal_without_lock(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dev.json"
            payload = _fixture_payload()
            payload["cases"][0]["edges"] = [{
                "id": "self", "source": "lux", "target": "lux",
                "edge_types": ["MODIFIES"], "critical": False,
            }]
            # Use a second mention so the edge is structurally valid, while no
            # question names the new reviewed fact.
            payload["cases"][0]["mentions"].append({
                "id": "walks", "node_types": ["ACTION"],
                "acceptable_spans": [[4, 9]], "critical": False,
            })
            payload["cases"][0]["edges"][0]["target"] = "walks"
            inner = {key: value for key, value in payload.items() if key != "content_sha256"}
            payload["content_sha256"] = hashlib.sha256(json.dumps(
                inner, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode()).hexdigest()
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_semantic_benchmark(
                    path, expected_split="DEV",
                    expected_content_sha256=payload["content_sha256"],
                    expected_pool_manifest_sha256=payload["pool_manifest_sha256"],
                    pool_manifest=_fixture_pool(),
                )

    def test_checksum_rejects_duplicate_easy_questions_and_dimension_games(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dev.json"
            for mutation in ("duplicate", "wrong-dimension"):
                payload = _fixture_payload()
                if mutation == "duplicate":
                    duplicate = copy.deepcopy(payload["cases"][0]["questions"][0])
                    duplicate["id"] = "duplicate-easy"
                    payload["cases"][0]["questions"].append(duplicate)
                else:
                    causal = next(
                        item for item in payload["cases"][0]["questions"]
                        if item["id"] == "q-cause"
                    )
                    causal["dimension"] = "entity_recovery"
                inner = {key: value for key, value in payload.items() if key != "content_sha256"}
                payload["content_sha256"] = hashlib.sha256(json.dumps(
                    inner, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                ).encode()).hexdigest()
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    load_semantic_benchmark(
                        path, expected_split="DEV",
                        expected_content_sha256=payload["content_sha256"],
                        expected_pool_manifest_sha256=payload["pool_manifest_sha256"],
                        pool_manifest=_fixture_pool(),
                    )

    def test_fixture_window_must_be_a_member_of_locked_pool(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dev.json"
            payload = _fixture_payload()
            case = payload["cases"][0]
            case["source_id"] = "transcript:not-in-pool"
            case["upstream_source_id"] = "not-in-pool"
            inner = {key: value for key, value in payload.items() if key != "content_sha256"}
            payload["content_sha256"] = hashlib.sha256(json.dumps(
                inner, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode()).hexdigest()
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside its locked pool"):
                load_semantic_benchmark(
                    path, expected_split="DEV",
                    expected_content_sha256=payload["content_sha256"],
                    expected_pool_manifest_sha256=payload["pool_manifest_sha256"],
                    pool_manifest=_fixture_pool(),
                )

    def test_first_loss_follows_stage_chronology_not_global_error_kind(self):
        failures = [
            {"code": "MENTION_SELECTION_MISS", "fact_id": "mention:action", "stage": None},
            {"code": "PROVIDER_FAILURE", "fact_id": None, "stage": "qualifiers"},
            {"code": "EDGE_PAIR_NOT_ENUMERATED", "fact_id": "edge:cause", "stage": None},
        ]
        self.assertEqual(_first_loss(failures), "MENTION_SELECTION_MISS")
        self.assertEqual(_first_loss([
            {"code": "QUALIFIER_CANDIDATE_MISSING", "fact_id": "qualifier:q", "stage": None},
            {"code": "PROVIDER_FAILURE", "fact_id": None, "stage": "qualifiers"},
        ]), "QUALIFIER_CANDIDATE_MISSING")

    def test_source_loss_retains_dimension_and_family_denominators(self):
        run = _run()
        case = _case(exhaustive=False)
        shifted = replace(
            case, source_id="transcript:other", upstream_source_id="other",
        )
        result = evaluate_semantic_case(shifted, run)
        self.assertEqual(result["semantic_checksum"]["hit_count"], 0)
        self.assertEqual(result["semantic_checksum"]["denominator"], len(case.questions))
        self.assertEqual(result["dimensions"]["entity_recovery"]["denominator"], 2)
        self.assertEqual(result["critical_dimensions"]["entity_recovery"]["denominator"], 2)
        self.assertEqual(result["mention_families"]["ENTITY"]["denominator"], 2)

    def test_semantic_duplicate_alternatives_and_missing_critical_dimensions_reject(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dev.json"
            for mutation in ("edge-order", "noncritical"):
                payload = _fixture_payload()
                if mutation == "edge-order":
                    first = payload["cases"][0]["edges"][0]
                    first["edge_types"] = ["ACTOR", "CONDITION"]
                    duplicate = copy.deepcopy(first)
                    duplicate["id"] = "same-edge-reordered"
                    duplicate["edge_types"] = ["CONDITION", "ACTOR"]
                    payload["cases"][0]["edges"].append(duplicate)
                    payload["cases"][0]["questions"].append({
                        "id": "q-duplicate", "prompt": "Duplicate semantic edge?",
                        "dimension": "condition_recovery",
                        "requires": ["edge:same-edge-reordered"], "critical": True,
                    })
                else:
                    for fact in payload["cases"][0]["mentions"] + payload["cases"][0]["edges"]:
                        fact["critical"] = False
                    for question in payload["cases"][0]["questions"]:
                        question["critical"] = False
                inner = {key: value for key, value in payload.items() if key != "content_sha256"}
                payload["content_sha256"] = hashlib.sha256(json.dumps(
                    inner, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                ).encode()).hexdigest()
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    load_semantic_benchmark(
                        path, expected_split="DEV",
                        expected_content_sha256=payload["content_sha256"],
                        expected_pool_manifest_sha256=payload["pool_manifest_sha256"],
                        pool_manifest=_fixture_pool(),
                    )

    def test_overlapping_edge_alternatives_and_duplicate_qualifier_field_reject(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dev.json"
            for mutation in ("overlap", "qualifier-field"):
                payload = _fixture_payload()
                if mutation == "overlap":
                    payload["cases"][0]["edges"].append({
                        "id": "overlap", "source": "event", "target": "action",
                        "edge_types": ["CAUSES", "ENABLES"], "critical": True,
                    })
                    payload["cases"][0]["questions"].append({
                        "id": "q-overlap", "prompt": "What causes the action?",
                        "dimension": "causal_edges", "requires": ["edge:overlap"],
                        "critical": True,
                    })
                else:
                    text = payload["cases"][0]["source_text"]
                    cue = [text.index("when"), text.index("when") + 4]
                    for identifier, value in (("one", "CONDITIONAL"), ("two", "HYPOTHETICAL")):
                        payload["cases"][0]["qualifiers"].append({
                            "id": identifier, "mention": "event",
                            "field": "conditionality", "value": value,
                            "cue_spans": [cue], "critical": True,
                        })
                        payload["cases"][0]["questions"].append({
                            "id": "q-" + identifier, "prompt": "What condition applies?",
                            "dimension": "condition_recovery",
                            "requires": ["qualifier:" + identifier], "critical": True,
                        })
                inner = {key: value for key, value in payload.items() if key != "content_sha256"}
                payload["content_sha256"] = hashlib.sha256(json.dumps(
                    inner, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                ).encode()).hexdigest()
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    load_semantic_benchmark(
                        path, expected_split="DEV",
                        expected_content_sha256=payload["content_sha256"],
                        expected_pool_manifest_sha256=payload["pool_manifest_sha256"],
                        pool_manifest=_fixture_pool(),
                    )

    def test_missing_qualifier_candidate_is_the_first_localized_loss(self):
        base = _case(exhaustive=False)
        qualifier = GoldQualifier(
            "negative_lux", "lux", "polarity", "NEGATIVE", ((0, 3),), True,
        )
        case = SemanticBenchmarkCase(
            "qualifier-catalog-loss", "DEV", base.source_id, base.source_kind,
            base.source_text, base.upstream_source_id, base.upstream_start,
            base.upstream_end, ("negation",), False, (base.mentions[0],), (),
            (qualifier,), (), (
                SemanticQuestion("lux", "Who?", "entity_recovery", ("mention:lux",), True),
                SemanticQuestion("negative", "What is negated?", "negation", ("qualifier:negative_lux",), True),
            ),
        )
        result = evaluate_semantic_case(case, _run())
        self.assertEqual(result["qualifier_candidate_coverage"]["rate"], 0.0)
        self.assertEqual(result["first_loss"], "QUALIFIER_CANDIDATE_MISSING")

    def test_unmeasurable_invention_rate_is_explicitly_none(self):
        case = _case(exhaustive=False)
        result = evaluate_semantic_case(case, _run())
        self.assertEqual(result["unsupported_node_rate"]["denominator"], 0)
        self.assertIsNone(result["unsupported_node_rate"]["rate"])


if __name__ == "__main__":
    unittest.main()
