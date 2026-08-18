import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.phase2g_endpoint_recovery import (
    BENCHMARK_CONTENT_SHA256,
    CONDITIONS,
    FAILURE_CODES,
    FAILURE_PRECEDENCE,
    GATE_THRESHOLDS,
    RAW_BRONZE,
    REFERENCE_MODEL,
    REFERENCE_THINKING,
    RUN_VERSION,
    build_aggregate,
    build_case_experiment,
    build_request,
    canonical_sha256,
    classify_endpoint,
    compare_artifacts,
    condition_aggregate,
    evaluate_case,
    load_benchmark,
    parse_model_response,
    promotion_gate,
    publish_artifact,
    resolve_parsed_payload,
    run_case_condition,
    run_experiment,
    validate_experiment_coverage,
)
from pipeline.phase2g_silver import (
    load_silver_fixture,
    validate_fixture_against_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/semantic_ir_legacy_failure_v1.json"
FIXTURE = ROOT / "data/phase2g_silver_v1.json"


def _alias_by_span(experiment):
    return {
        (record["start"], record["end"]): record["alias"]
        for record in experiment["catalog"]
    }


def _perfect_response(experiment, *, include_invented=False):
    alias_by_span = _alias_by_span(experiment)
    response = {"endpoint_selections": {}, "reference_statuses": {}}
    for task in experiment["endpoint_tasks"]:
        span = task["gold_spans"][0]
        aliases = [alias_by_span[span]]
        if include_invented:
            aliases.append("X9999")
        response["endpoint_selections"][task["task_id"]] = {
            "roles": {task["gold_node_types"][0]: aliases},
            "status": "NONE",
        }
    for task in experiment["status_tasks"]:
        response["reference_statuses"][task["task_id"]] = {
            "status": "UNKNOWN",
            "targets": [],
        }
    return response


def _perfect_chat(experiments):
    by_bronze = {
        experiment["bronze_text"]: experiment for experiment in experiments.values()
    }

    def chat(system, user, **kwargs):
        experiment = next(
            item for bronze, item in by_bronze.items() if bronze in user
        )
        return json.dumps(_perfect_response(experiment))

    return chat


class Phase2GEndpointRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = load_benchmark(BENCHMARK)
        cls.fixture = load_silver_fixture(FIXTURE)
        validate_fixture_against_benchmark(cls.benchmark, cls.fixture)
        cls.experiments = {
            case["id"]: build_case_experiment(case)
            for case in cls.benchmark["cases"]
        }
        cls.coverage = validate_experiment_coverage(cls.experiments)

    def test_benchmark_and_coverage_are_locked(self):
        self.assertEqual(
            self.benchmark["content_sha256"], BENCHMARK_CONTENT_SHA256,
        )
        self.assertEqual(
            self.coverage["candidate_coverage"],
            {"hit_count": 33, "denominator": 33, "rate": 1.0},
        )
        expected = {
            "wave-reset-after-kill": (5, 2),
            "push-poke-wave-crash": (5, 2),
            "sweeper-limits-mid-play": (7, 0),
            "mid-push-prevents-side-collapse": (5, 1),
            "unwarded-bush-hook-risk": (11, 3),
        }
        for case_id, (endpoints, statuses) in expected.items():
            experiment = self.experiments[case_id]
            self.assertEqual(experiment["expected_endpoint_count"], endpoints)
            self.assertEqual(experiment["expected_status_count"], statuses)

    def test_aliases_are_compact_stable_and_sorted(self):
        rebuilt = {
            case["id"]: build_case_experiment(case)
            for case in self.benchmark["cases"]
        }
        for case_id, experiment in self.experiments.items():
            with self.subTest(case=case_id):
                self.assertEqual(
                    experiment["catalog"], rebuilt[case_id]["catalog"],
                )
                aliases = [record["alias"] for record in experiment["catalog"]]
                self.assertEqual(aliases, sorted(aliases))
                self.assertEqual(aliases[0], "C0001")
                spans = [(record["start"], record["end"]) for record in experiment["catalog"]]
                self.assertEqual(spans, sorted(spans))
                for record in experiment["catalog"]:
                    self.assertEqual(
                        experiment["bronze_text"][record["start"]:record["end"]],
                        record["text"],
                    )

    def test_same_aliases_offsets_and_bronze_text_in_all_three_conditions(self):
        case = self.benchmark["cases"][0]
        experiment = self.experiments[case["id"]]
        prompts = {}
        for condition in CONDITIONS:
            text = (
                case["source_text"] if condition == RAW_BRONZE
                else self.fixture["cases"][case["id"]][
                    "mechanical" if condition == "MECHANICAL_SILVER" else "resolved"
                ]["text"]
            )
            prompts[condition] = build_request(
                experiment, text, condition=condition,
            )["user"]
        catalog_json = json.dumps(
            [[record["alias"], record["start"], record["end"], record["text"]]
             for record in experiment["catalog"]],
            ensure_ascii=False, separators=(",", ":"),
        )
        for condition in CONDITIONS:
            self.assertIn(catalog_json, prompts[condition])
            self.assertIn(experiment["bronze_text"], prompts[condition])
            catalog_block = prompts[condition].split(
                "CANDIDATE CATALOG (complete rows [alias,start,end,exact authoritative bronze text]):",
                1,
            )[1].split("\n\nENDPOINT TASKS", 1)[0]
            self.assertEqual(catalog_block.strip(), catalog_json)
            first_row = json.loads(catalog_block)[0]
            self.assertEqual(first_row[1:3], [
                experiment["catalog"][0]["start"],
                experiment["catalog"][0]["end"],
            ])
        self.assertNotEqual(prompts[RAW_BRONZE], prompts["MECHANICAL_SILVER"])
        self.assertNotEqual(prompts["MECHANICAL_SILVER"], prompts["RESOLVED_SILVER"])

    def test_catalog_records_include_phase2f_ids_offsets_and_provenance(self):
        for case in self.benchmark["cases"]:
            with self.subTest(case=case["id"]):
                experiment = self.experiments[case["id"]]
                upstream_start = case["upstream_start"]
                for record in experiment["catalog"]:
                    self.assertEqual(record["alias"][0], "C")
                    self.assertTrue(record["candidate_id"].startswith(
                        record["window_id"] + ":m",
                    ))
                    start, end = record["start"], record["end"]
                    self.assertGreater(end, start)
                    self.assertLessEqual(end, len(case["source_text"]))
                    self.assertEqual(record["text"], case["source_text"][start:end])
                    self.assertEqual(
                        record["absolute_start"], upstream_start + start,
                    )
                    self.assertEqual(
                        record["absolute_end"], upstream_start + end,
                    )
                    self.assertTrue(record["segment_ids"])
                self.assertEqual(
                    experiment["catalog_sha256"],
                    canonical_sha256(experiment["catalog"]),
                )

    def test_per_case_artifact_records_include_catalog_and_expected_tasks(self):
        for case in self.benchmark["cases"]:
            with self.subTest(case=case["id"]):
                experiment = self.experiments[case["id"]]
                record = run_case_condition(
                    experiment, case, RAW_BRONZE, self.fixture,
                    lambda system, user, **kwargs: json.dumps(
                        _perfect_response(experiment),
                    ),
                )
                self.assertEqual(record["catalog"], experiment["catalog"])
                self.assertEqual(
                    record["expected_endpoint_tasks"],
                    experiment["endpoint_tasks"],
                )
                self.assertEqual(
                    record["expected_status_tasks"],
                    experiment["status_tasks"],
                )
                self.assertIn("catalog", record)
                for catalog_record in record["catalog"]:
                    for field in (
                        "alias", "candidate_id", "start", "end",
                        "absolute_start", "absolute_end", "text", "segment_ids",
                    ):
                        self.assertIn(field, catalog_record)

    def test_requests_are_deterministic_across_reruns(self):
        case = self.benchmark["cases"][0]
        experiment = self.experiments[case["id"]]
        first = build_request(
            experiment, case["source_text"], condition=RAW_BRONZE,
        )
        second = build_request(
            experiment, case["source_text"], condition=RAW_BRONZE,
        )
        self.assertEqual(first["request_sha256"], second["request_sha256"])
        self.assertEqual(first["model"], REFERENCE_MODEL)
        self.assertEqual(first["thinking"], REFERENCE_THINKING)
        self.assertEqual(first["temperature"], 0.0)

    def test_parser_accepts_strict_json_and_single_fence(self):
        experiment = self.experiments["wave-reset-after-kill"]
        response = _perfect_response(experiment)
        bare = parse_model_response(json.dumps(response), experiment)
        self.assertEqual(bare, response)
        fenced = parse_model_response(
            "```json\n" + json.dumps(response) + "\n```", experiment,
        )
        self.assertEqual(fenced, response)
        plain_fence = parse_model_response(
            "```\n" + json.dumps(response) + "\n```", experiment,
        )
        self.assertEqual(plain_fence, response)

    def test_parser_rejects_unknown_keys_types_statuses_and_roles(self):
        experiment = self.experiments["wave-reset-after-kill"]
        base = _perfect_response(experiment)
        first_task = next(iter(base["endpoint_selections"]))
        mutations = [
            (dict(base, extra="x"), "unknown root key"),
            (
                {
                    **base,
                    "endpoint_selections": {
                        **base["endpoint_selections"],
                        first_task: {
                            "roles": {"NOT_A_ROLE": ["C0001"]},
                            "status": "NONE",
                        },
                    },
                },
                "unknown role",
            ),
            (
                {
                    **base,
                    "endpoint_selections": {
                        **base["endpoint_selections"],
                        first_task: {
                            "roles": {"ENTITY": ["C0001"]},
                            "status": "MAYBE",
                        },
                    },
                },
                "unknown status",
            ),
            (
                {
                    **base,
                    "endpoint_selections": {
                        **base["endpoint_selections"],
                        "ep-99": {"roles": {"ENTITY": ["C0001"]}, "status": "NONE"},
                    },
                },
                "unknown endpoint task id",
            ),
            (
                {
                    **base,
                    "reference_statuses": {
                        **base["reference_statuses"],
                        "st-99": {"status": "UNKNOWN", "targets": []},
                    },
                },
                "unknown reference task id",
            ),
            (
                {
                    **base,
                    "endpoint_selections": {
                        **base["endpoint_selections"],
                        first_task: {
                            "roles": {"ENTITY": [7]},
                            "status": "NONE",
                        },
                    },
                },
                "non-string alias",
            ),
            (
                {
                    **base,
                    "endpoint_selections": {
                        **base["endpoint_selections"],
                        first_task: {
                            "roles": {"ENTITY": ["C0001"]},
                            "status": "NONE",
                            "offsets": [0, 5],
                        },
                    },
                },
                "model-supplied offsets must be rejected",
            ),
        ]
        for mutation, label in mutations:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    parse_model_response(json.dumps(mutation), experiment)

    def test_parser_rejects_multiple_fences_and_surrounding_text(self):
        experiment = self.experiments["wave-reset-after-kill"]
        response = _perfect_response(experiment)
        payloads = (
            "```json\n" + json.dumps(response) + "\n```\n```json\n{}\n```",
            "prefix " + json.dumps(response) + " ```",
            "```json\n" + json.dumps(response) + "\n``` trailing",
            "",
        )
        for payload in payloads:
            with self.subTest(payload=payload[:20]):
                with self.assertRaises(ValueError):
                    parse_model_response(payload, experiment)

    def test_parser_accepts_multiple_candidates_per_role(self):
        experiment = self.experiments["wave-reset-after-kill"]
        response = _perfect_response(experiment)
        task_id = experiment["endpoint_tasks"][0]["task_id"]
        aliases = [record["alias"] for record in experiment["catalog"][:2]]
        response["endpoint_selections"][task_id] = {
            "roles": {"ENTITY": aliases, "EVENT": aliases},
            "status": "NONE",
        }
        parsed = parse_model_response(json.dumps(response), experiment)
        self.assertEqual(
            parsed["endpoint_selections"][task_id]["roles"]["ENTITY"], aliases,
        )

    def test_parser_rejects_duplicate_json_keys_at_any_level(self):
        experiment = self.experiments["wave-reset-after-kill"]
        payloads = [
            (
                '{"endpoint_selections":{},"endpoint_selections":{},'
                '"reference_statuses":{}}',
                "duplicate root key",
            ),
            (
                '{"endpoint_selections":{"ep-01":{"roles":'
                '{"ENTITY":["C0001"],"ENTITY":["C0001"]},"status":"NONE"}},'
                '"reference_statuses":{}}',
                "duplicate role key",
            ),
            (
                '{"endpoint_selections":{},"reference_statuses":{"st-01":'
                '{"status":"UNKNOWN","targets":[],"targets":[]}}}',
                "duplicate reference entry key",
            ),
        ]
        for payload, label in payloads:
            with self.subTest(label=label):
                with self.assertRaises(ValueError) as caught:
                    parse_model_response(payload, experiment)
                self.assertIn("duplicate JSON key", str(caught.exception))

    def test_parser_requires_exact_root_and_task_id_sets(self):
        experiment = self.experiments["wave-reset-after-kill"]
        base = _perfect_response(experiment)
        first_task = next(iter(base["endpoint_selections"]))
        first_status = next(iter(base["reference_statuses"]))
        endpoint_entry = base["endpoint_selections"][first_task]
        status_entry = base["reference_statuses"][first_status]
        mutations = [
            (
                {key: value for key, value in base.items()
                 if key != "endpoint_selections"},
                "missing root endpoint_selections",
            ),
            (
                {**base, "endpoint_selections": {}},
                "missing all endpoint task ids",
            ),
            (
                {
                    **base,
                    "endpoint_selections": {
                        key: value for key, value in
                        base["endpoint_selections"].items()
                        if key != first_task
                    },
                },
                "missing one endpoint task id",
            ),
            (
                {**base, "reference_statuses": {}},
                "missing all reference task ids",
            ),
            (
                {
                    **base,
                    "reference_statuses": {
                        key: value for key, value in
                        base["reference_statuses"].items()
                        if key != first_status
                    },
                },
                "missing one reference task id",
            ),
            (
                {
                    **base,
                    "endpoint_selections": {
                        **base["endpoint_selections"],
                        "ep-99": endpoint_entry,
                    },
                },
                "extra endpoint task id",
            ),
            (
                {
                    **base,
                    "reference_statuses": {
                        **base["reference_statuses"],
                        "st-99": status_entry,
                    },
                },
                "extra reference task id",
            ),
        ]
        for mutation, label in mutations:
            with self.subTest(label=label):
                with self.assertRaises(ValueError) as caught:
                    parse_model_response(json.dumps(mutation), experiment)
                self.assertIn("must be exactly", str(caught.exception))

    def test_parser_requires_exact_entry_keys(self):
        experiment = self.experiments["wave-reset-after-kill"]
        base = _perfect_response(experiment)
        first_task = next(iter(base["endpoint_selections"]))
        first_status = next(iter(base["reference_statuses"]))
        mutations = [
            (
                {
                    **base,
                    "endpoint_selections": {
                        **base["endpoint_selections"],
                        first_task: {"status": "NONE"},
                    },
                },
                "endpoint entry missing roles",
            ),
            (
                {
                    **base,
                    "endpoint_selections": {
                        **base["endpoint_selections"],
                        first_task: {"roles": {"ENTITY": ["C0001"]}},
                    },
                },
                "endpoint entry missing status",
            ),
            (
                {
                    **base,
                    "reference_statuses": {
                        **base["reference_statuses"],
                        first_status: {"targets": []},
                    },
                },
                "reference entry missing status",
            ),
            (
                {
                    **base,
                    "reference_statuses": {
                        **base["reference_statuses"],
                        first_status: {"status": "UNKNOWN"},
                    },
                },
                "reference entry missing targets",
            ),
        ]
        for mutation, label in mutations:
            with self.subTest(label=label):
                with self.assertRaises(ValueError) as caught:
                    parse_model_response(json.dumps(mutation), experiment)
                self.assertIn("keys must be exactly", str(caught.exception))

    def test_parser_rejects_empty_roles_and_duplicate_in_role_aliases(self):
        experiment = self.experiments["wave-reset-after-kill"]
        base = _perfect_response(experiment)
        first_task = next(iter(base["endpoint_selections"]))
        alias = experiment["catalog"][0]["alias"]
        mutations = [
            (
                {"roles": {"ENTITY": []}, "status": "UNKNOWN"},
                "empty role array",
            ),
            (
                {"roles": {"ENTITY": [alias, alias]}, "status": "NONE"},
                "duplicate aliases within one role",
            ),
            (
                {"roles": {"ENTITY": [alias, 7]}, "status": "NONE"},
                "non-string alias in role",
            ),
        ]
        for entry, label in mutations:
            with self.subTest(label=label):
                mutated = {
                    **base,
                    "endpoint_selections": {
                        **base["endpoint_selections"],
                        first_task: entry,
                    },
                }
                with self.assertRaises(ValueError):
                    parse_model_response(json.dumps(mutated), experiment)
        # The same alias under two different roles is a legitimate choice.
        valid = {
            **base,
            "endpoint_selections": {
                **base["endpoint_selections"],
                first_task: {
                    "roles": {"ENTITY": [alias], "EVENT": [alias]},
                    "status": "NONE",
                },
            },
        }
        parsed = parse_model_response(json.dumps(valid), experiment)
        self.assertEqual(
            parsed["endpoint_selections"][first_task]["roles"]["ENTITY"],
            [alias],
        )

    def test_parser_enforces_status_selection_consistency(self):
        experiment = self.experiments["wave-reset-after-kill"]
        base = _perfect_response(experiment)
        first_task = next(iter(base["endpoint_selections"]))
        first_status = next(iter(base["reference_statuses"]))
        alias = experiment["catalog"][0]["alias"]

        def endpoint_entry(roles, status):
            return {"roles": roles, "status": status}

        def with_endpoint(entry):
            return {
                **base,
                "endpoint_selections": {
                    **base["endpoint_selections"], first_task: entry,
                },
            }

        for entry, label in (
            (endpoint_entry({}, "NONE"), "endpoint NONE without candidates"),
            (
                endpoint_entry({"ENTITY": [alias]}, "UNKNOWN"),
                "endpoint UNKNOWN with candidates",
            ),
            (
                endpoint_entry({"ENTITY": [alias]}, "AMBIGUOUS"),
                "endpoint AMBIGUOUS with candidates",
            ),
        ):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    parse_model_response(
                        json.dumps(with_endpoint(entry)), experiment,
                    )

        for entry, label in (
            (endpoint_entry({}, "UNKNOWN"), "endpoint UNKNOWN abstains"),
            (endpoint_entry({}, "AMBIGUOUS"), "endpoint AMBIGUOUS abstains"),
            (
                endpoint_entry({"ENTITY": [alias]}, "NONE"),
                "endpoint NONE with candidate",
            ),
        ):
            with self.subTest(label=label):
                parse_model_response(json.dumps(with_endpoint(entry)), experiment)

        def with_reference(entry):
            return {
                **base,
                "reference_statuses": {
                    **base["reference_statuses"], first_status: entry,
                },
            }

        for entry, label in (
            (
                {"status": "UNKNOWN", "targets": [alias]},
                "reference UNKNOWN with targets",
            ),
            (
                {"status": "AMBIGUOUS", "targets": [alias]},
                "reference AMBIGUOUS with targets",
            ),
        ):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    parse_model_response(
                        json.dumps(with_reference(entry)), experiment,
                    )

        for entry, label in (
            ({"status": "NONE", "targets": []}, "reference NONE no targets"),
            (
                {"status": "NONE", "targets": [alias]},
                "reference NONE with targets",
            ),
            (
                {"status": "UNKNOWN", "targets": []},
                "reference UNKNOWN no targets",
            ),
        ):
            with self.subTest(label=label):
                parse_model_response(json.dumps(with_reference(entry)), experiment)

    def test_invented_ids_are_retained_as_diagnostics_not_dropped(self):
        experiment = self.experiments["wave-reset-after-kill"]
        response = _perfect_response(experiment, include_invented=True)
        parsed = parse_model_response(json.dumps(response), experiment)
        resolution = resolve_parsed_payload(experiment, parsed)
        invented = [
            assignment
            for task in resolution["endpoints"].values()
            for assignment in task["assignments"]
            if not assignment["known"]
        ]
        self.assertEqual(
            [a["alias"] for a in invented], ["X9999"] * 5,
        )
        evaluation = evaluate_case(
            experiment, resolution, parser_failed=False,
        )
        self.assertEqual(evaluation["invented_selections"], 5)
        self.assertIn("X9999", evaluation["first_failures"][0]["detail"])

    def test_perfect_response_scores_full_metrics(self):
        case_results = []
        for case_id, experiment in self.experiments.items():
            parsed = parse_model_response(
                json.dumps(_perfect_response(experiment)), experiment,
            )
            resolution = resolve_parsed_payload(experiment, parsed)
            case_results.append(evaluate_case(experiment, resolution, parser_failed=False))
        aggregate = condition_aggregate(RAW_BRONZE, case_results)
        for name in (
            "candidate_coverage", "endpoint_recall", "endpoint_precision",
            "role_accuracy", "status_accuracy", "parseability",
        ):
            metric = aggregate[name]
            self.assertEqual(metric["hit_count"], metric["denominator"], name)
            self.assertEqual(metric["rate"], 1.0, name)
        self.assertEqual(aggregate["unsupported_selections"], 0)
        self.assertEqual(aggregate["invented_selections"], 0)
        self.assertEqual(
            aggregate["source_alignment_violations"]["hit_count"], 0,
        )
        self.assertTrue(aggregate["gate"]["passed"])

    def test_metric_denominator_reconciliation(self):
        case_results = []
        for experiment in self.experiments.values():
            parsed = parse_model_response(
                json.dumps(_perfect_response(experiment)), experiment,
            )
            resolution = resolve_parsed_payload(experiment, parsed)
            case_results.append(evaluate_case(experiment, resolution, parser_failed=False))
        self.assertEqual(
            sum(item["endpoint_recall"]["denominator"] for item in case_results), 33,
        )
        self.assertEqual(
            sum(item["candidate_coverage"]["denominator"] for item in case_results), 33,
        )
        self.assertEqual(
            sum(item["status_accuracy"]["denominator"] for item in case_results), 8,
        )
        self.assertEqual(
            sum(item["endpoint_precision"]["denominator"] for item in case_results),
            sum(item["endpoint_precision"]["hit_count"] for item in case_results),
        )

    def test_precision_counts_wrong_and_invented_assignments(self):
        experiment = self.experiments["wave-reset-after-kill"]
        response = _perfect_response(experiment, include_invented=True)
        task_id = experiment["endpoint_tasks"][1]["task_id"]
        wrong_alias = experiment["catalog"][0]["alias"]
        response["endpoint_selections"][task_id]["roles"]["ENTITY"].append(wrong_alias)
        parsed = parse_model_response(json.dumps(response), experiment)
        resolution = resolve_parsed_payload(experiment, parsed)
        evaluation = evaluate_case(experiment, resolution, parser_failed=False)
        self.assertEqual(evaluation["unsupported_selections"], 1)
        self.assertEqual(evaluation["invented_selections"], 5)
        self.assertEqual(
            evaluation["endpoint_precision"]["hit_count"],
            evaluation["endpoint_precision"]["denominator"] - 6,
        )

    def test_candidate_coverage_is_recomputed_from_catalog_spans(self):
        original = self.experiments["wave-reset-after-kill"]
        experiment = copy.deepcopy(original)
        task = experiment["endpoint_tasks"][0]
        removed_spans = {tuple(span) for span in task["gold_spans"]}
        experiment["catalog"] = [
            record for record in experiment["catalog"]
            if (record["start"], record["end"]) not in removed_spans
        ]
        parsed = parse_model_response(
            json.dumps(_perfect_response(original)), original,
        )
        resolution = resolve_parsed_payload(original, parsed)
        evaluation = evaluate_case(experiment, resolution, parser_failed=False)
        self.assertEqual(
            evaluation["candidate_coverage"],
            {"hit_count": 4, "denominator": 5, "rate": 0.8},
        )

    def test_role_accuracy_is_task_level_without_duplicate_inflation(self):
        experiment = self.experiments["wave-reset-after-kill"]
        response = _perfect_response(experiment)
        task = experiment["endpoint_tasks"][0]
        alias = _alias_by_span(experiment)[task["gold_spans"][0]]
        entry = response["endpoint_selections"][task["task_id"]]
        entry["roles"]["TIME"] = [alias]
        parsed = parse_model_response(json.dumps(response), experiment)
        resolution = resolve_parsed_payload(experiment, parsed)
        evaluation = evaluate_case(experiment, resolution, parser_failed=False)
        # 5 recalled tasks; task 0 carries a wrong-role correct-span assignment,
        # so the task is not role-correct and is counted exactly once.
        self.assertEqual(
            evaluation["role_accuracy"],
            {"hit_count": 4, "denominator": 5, "rate": 0.8},
        )
        # Both correct-span assignments still count toward endpoint precision.
        self.assertEqual(evaluation["endpoint_precision"]["hit_count"], 6)
        self.assertEqual(evaluation["endpoint_precision"]["denominator"], 6)

    def test_right_id_plus_extra_wrong_candidate_is_wrong_candidate_selected(self):
        experiment = self.experiments["wave-reset-after-kill"]
        response = _perfect_response(experiment)
        task = experiment["endpoint_tasks"][0]
        alias_by_span = _alias_by_span(experiment)
        gold_alias = alias_by_span[task["gold_spans"][0]]
        other_alias = next(
            record["alias"] for record in experiment["catalog"]
            if record["alias"] != gold_alias
        )
        response["endpoint_selections"][task["task_id"]]["roles"]["ENTITY"] = [
            gold_alias, other_alias,
        ]
        parsed = parse_model_response(json.dumps(response), experiment)
        resolution = resolve_parsed_payload(experiment, parsed)
        evaluation = evaluate_case(experiment, resolution, parser_failed=False)
        failure = next(
            item for item in evaluation["first_failures"]
            if item["task_id"] == task["task_id"]
        )
        self.assertEqual(failure["code"], "WRONG_CANDIDATE_SELECTED")
        self.assertEqual(
            len([
                item for item in evaluation["first_failures"]
                if item["task_id"] == task["task_id"]
            ]),
            1,
        )

        # Invented IDs outrank the extra-wrong-candidate failure.
        response["endpoint_selections"][task["task_id"]]["roles"]["ENTITY"].append(
            "X9999",
        )
        parsed = parse_model_response(json.dumps(response), experiment)
        resolution = resolve_parsed_payload(experiment, parsed)
        evaluation = evaluate_case(experiment, resolution, parser_failed=False)
        failure = next(
            item for item in evaluation["first_failures"]
            if item["task_id"] == task["task_id"]
        )
        self.assertEqual(failure["code"], "MODEL_INVENTED")

    def test_reference_targets_count_into_unsupported_invented_and_status(self):
        experiment = self.experiments["wave-reset-after-kill"]
        parsed = _perfect_response(experiment)
        status_task = experiment["status_tasks"][0]
        parsed["reference_statuses"][status_task["task_id"]] = {
            "status": "UNKNOWN",
            "targets": [experiment["catalog"][0]["alias"], "X9999"],
        }
        resolution = resolve_parsed_payload(experiment, parsed)
        evaluation = evaluate_case(experiment, resolution, parser_failed=False)
        self.assertEqual(evaluation["unsupported_selections"], 1)
        self.assertEqual(evaluation["invented_selections"], 1)
        # Endpoint precision denominator stays endpoint-only (5 assignments),
        # while the overall selection denominator includes both targets.
        self.assertEqual(evaluation["endpoint_precision"]["denominator"], 5)
        self.assertEqual(evaluation["overall_selections"], 7)
        self.assertEqual(evaluation["unsupported_or_invented_rate"], 2 / 7)
        detail = next(
            item for item in evaluation["status_details"]
            if item["task_id"] == status_task["task_id"]
        )
        self.assertFalse(detail["correct"])
        self.assertIn("UNKNOWN gold must not carry target IDs", detail["reason"])
        # The untouched status task remains correct.
        self.assertEqual(evaluation["status_accuracy"]["hit_count"], 1)
        self.assertEqual(evaluation["status_accuracy"]["denominator"], 2)

    def test_alignment_validation_inspects_every_known_selection(self):
        experiment = self.experiments["wave-reset-after-kill"]
        parsed = _perfect_response(experiment)
        endpoint_task = experiment["endpoint_tasks"][0]
        status_task = experiment["status_tasks"][0]
        alias = experiment["catalog"][0]["alias"]
        parsed["reference_statuses"][status_task["task_id"]] = {
            "status": "NONE",
            "targets": [alias, "X9999"],
        }
        resolution = resolve_parsed_payload(experiment, parsed)
        resolution["endpoints"][endpoint_task["task_id"]][
            "assignments"
        ][0]["text"] = "tampered endpoint"
        resolution["statuses"][status_task["task_id"]][
            "targets"
        ][0]["text"] = "tampered reference"
        evaluation = evaluate_case(experiment, resolution, parser_failed=False)
        self.assertEqual(
            evaluation["source_alignment_violations"]["hit_count"], 2,
        )
        # Every known endpoint assignment (5) plus the one known reference
        # target (1) is inspected; the invented target is invention evidence.
        self.assertEqual(
            evaluation["source_alignment_violations"]["denominator"], 6,
        )
        self.assertEqual(evaluation["invented_selections"], 1)

    def test_failure_taxonomy_every_feasible_branch_and_precedence(self):
        experiment = copy.deepcopy(self.experiments["wave-reset-after-kill"])
        alias_by_span = _alias_by_span(experiment)
        task = experiment["endpoint_tasks"][0]
        gold_span = task["gold_spans"][0]
        gold_alias = alias_by_span[gold_span]
        other_alias = next(
            record["alias"] for record in experiment["catalog"]
            if (record["start"], record["end"]) != gold_span
        )

        def resolution(roles, status="NONE", alias=None):
            assignments = []
            if alias:
                assignments.append({
                    "alias": alias, "role": list(roles)[0], "known": True,
                    "start": gold_span[0], "end": gold_span[1], "text": "you",
                })
            return {"status": status, "assignments": assignments}

        branches = [
            ("PARSER_FAILURE", True, True, {"status": "NONE", "assignments": []}),
            ("MODEL_ABSTAINED", False, True, {"status": "NONE", "assignments": []}),
            (
                "SOURCE_AMBIGUOUS", False, True,
                {"status": "AMBIGUOUS", "assignments": []},
            ),
            (
                "REFERENCE_UNRESOLVED", False, True,
                {"status": "UNKNOWN", "assignments": []},
            ),
        ]
        for code, parser_failed, gold_in_catalog, task_resolution in branches:
            with self.subTest(code=code):
                correct, got, _ = classify_endpoint(
                    task, task_resolution,
                    parser_failed=parser_failed, gold_in_catalog=gold_in_catalog,
                )
                self.assertFalse(correct)
                self.assertEqual(got, code)

        wrong_resolution = resolution({"ENTITY"}, alias=other_alias)
        wrong_resolution["assignments"][0]["start"] = 0
        wrong_resolution["assignments"][0]["end"] = 7
        wrong_resolution["assignments"][0]["text"] = "already"
        correct, got, _ = classify_endpoint(
            task, wrong_resolution, parser_failed=False, gold_in_catalog=True,
        )
        self.assertEqual(got, "WRONG_CANDIDATE_SELECTED")

        invented_resolution = {
            "status": "NONE",
            "assignments": [{
                "alias": "X1", "role": "ENTITY", "known": False,
                "start": None, "end": None, "text": None,
            }],
        }
        correct, got, _ = classify_endpoint(
            task, invented_resolution, parser_failed=False, gold_in_catalog=True,
        )
        self.assertEqual(got, "MODEL_INVENTED")

        right_wrong_role = resolution({"TIME"}, alias=gold_alias)
        correct, got, _ = classify_endpoint(
            task, right_wrong_role, parser_failed=False, gold_in_catalog=True,
        )
        self.assertEqual(got, "RIGHT_CANDIDATE_WRONG_ROLE")

        precedence = {
            "status": "NONE",
            "assignments": [
                {
                    "alias": gold_alias, "role": "TIME", "known": True,
                    "start": gold_span[0], "end": gold_span[1], "text": "you",
                },
                {
                    "alias": "X1", "role": "ENTITY", "known": False,
                    "start": None, "end": None, "text": None,
                },
            ],
        }
        correct, got, _ = classify_endpoint(
            task, precedence, parser_failed=False, gold_in_catalog=True,
        )
        self.assertEqual(got, "RIGHT_CANDIDATE_WRONG_ROLE")

        correct, got, _ = classify_endpoint(
            task, wrong_resolution, parser_failed=False, gold_in_catalog=False,
        )
        self.assertEqual(got, "CANDIDATE_MISSING")
        self.assertEqual(set(FAILURE_CODES), set(FAILURE_PRECEDENCE))

    def test_promotion_gate_thresholds_and_single_condition_pass(self):
        passing = {
            "candidate_coverage": {"hit_count": 33, "denominator": 33, "rate": 1.0},
            "endpoint_recall": {"hit_count": 30, "denominator": 33, "rate": 30 / 33},
            "endpoint_precision": {"hit_count": 30, "denominator": 33, "rate": 30 / 33},
            "role_accuracy": {"hit_count": 28, "denominator": 30, "rate": 28 / 30},
            "status_accuracy": {"hit_count": 8, "denominator": 8, "rate": 1.0},
            "parseability": {"hit_count": 5, "denominator": 5, "rate": 1.0},
            "unsupported_or_invented_rate": 1 / 33,
            "source_alignment_violations": {"hit_count": 0, "denominator": 33, "rate": 0.0},
        }
        self.assertTrue(promotion_gate(passing)["passed"])
        for mutation in (
            {"endpoint_recall": {"hit_count": 26, "denominator": 33, "rate": 26 / 33}},
            {"unsupported_or_invented_rate": 0.1},
            {"source_alignment_violations": {"hit_count": 1, "denominator": 33, "rate": 1 / 33}},
            {"endpoint_precision": {"hit_count": 29, "denominator": 33, "rate": 29 / 33}},
            {"role_accuracy": {"hit_count": 25, "denominator": 30, "rate": 25 / 30}},
        ):
            failed = copy.deepcopy(passing)
            failed.update(mutation)
            self.assertFalse(promotion_gate(failed)["passed"])
        self.assertEqual(GATE_THRESHOLDS["endpoint_recall"], 0.90)

    def test_mocked_three_condition_runner_makes_exactly_15_calls(self):
        calls = []

        def chat(system, user, **kwargs):
            calls.append((system, user, kwargs))
            experiment = next(
                experiment for experiment in self.experiments.values()
                if experiment["bronze_text"] in user
            )
            return json.dumps(_perfect_response(experiment))

        result = run_experiment(self.benchmark, self.fixture, chat)
        self.assertEqual(len(calls), 15)
        for condition in CONDITIONS:
            self.assertEqual(
                len(result["conditions"][condition]["cases"]), 5,
            )
            self.assertTrue(
                result["conditions"][condition]["metrics"]["gate"]["passed"],
            )
        self.assertEqual(
            result["promotion_gate"]["satisfied_conditions"], list(CONDITIONS),
        )
        for _, _, kwargs in calls:
            self.assertEqual(kwargs["temperature"], 0.0)
            self.assertEqual(kwargs["model"], REFERENCE_MODEL)
            self.assertEqual(kwargs["thinking"], REFERENCE_THINKING)

    def test_runner_records_parser_and_provider_failures(self):
        def failing_chat(system, user, **kwargs):
            return "not json at all ```"

        result = run_experiment(self.benchmark, self.fixture, failing_chat)
        for condition in CONDITIONS:
            for case_record in result["conditions"][condition]["cases"].values():
                metrics = case_record["metrics"]
                self.assertEqual(metrics["parseable"]["rate"], 0.0)
                self.assertEqual(metrics["endpoint_recall"]["hit_count"], 0)
                self.assertTrue(case_record["parse_error"])
                self.assertEqual(
                    set(metrics["failure_counts"]),
                    set(FAILURE_CODES),
                )
                self.assertEqual(
                    metrics["failure_counts"]["PARSER_FAILURE"],
                    metrics["expected_endpoint_count"],
                )

        def provider_chat(system, user, **kwargs):
            raise TimeoutError("no provider bytes")

        result = run_experiment(self.benchmark, self.fixture, provider_chat)
        first = next(iter(result["conditions"][RAW_BRONZE]["cases"].values()))
        self.assertTrue(first["provider_failure"].startswith("TimeoutError"))
        self.assertEqual(
            first["metrics"]["first_failures"][0]["code"], "OTHER",
        )
        self.assertIn(
            "PROVIDER_FAILURE", first["metrics"]["first_failures"][0]["detail"],
        )

    def test_artifact_is_hash_locked_and_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_experiment(
                self.benchmark, self.fixture, _perfect_chat(self.experiments),
            )
            aggregate = build_aggregate(
                BENCHMARK, FIXTURE, result, repo=ROOT, provider="deepseek",
            )
            inner = {
                key: value for key, value in aggregate.items()
                if key != "content_sha256"
            }
            self.assertEqual(
                aggregate["content_sha256"], canonical_sha256(inner),
            )
            self.assertEqual(aggregate["run_version"], RUN_VERSION)
            self.assertEqual(
                aggregate["input_hashes"]["benchmark_content_sha256"],
                BENCHMARK_CONTENT_SHA256,
            )
            output = Path(tmp) / "phase2g-run"
            publish_artifact(output, aggregate)
            self.assertTrue((output / "phase2g-endpoint-recovery.json").exists())
            self.assertTrue((output / "MANIFEST.json").exists())
            for condition in CONDITIONS:
                self.assertTrue((output / "conditions" / condition).exists())
            manifest = json.loads(
                (output / "MANIFEST.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(len(manifest["files"]), 1 + 15)
            for entry in manifest["files"]:
                path = output / entry["path"]
                self.assertEqual(
                    entry["file_sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            with self.assertRaises(ValueError):
                publish_artifact(output, aggregate)
            bad = Path(tmp) / "bad-run"
            with patch(
                "pipeline.phase2g_endpoint_recovery.os.replace",
                side_effect=OSError("boom"),
            ):
                with self.assertRaises(OSError):
                    publish_artifact(bad, aggregate)
            self.assertFalse(bad.exists())

    def test_compare_helper_allows_timestamps_and_raw_hashes_but_flags_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_experiment(
                self.benchmark, self.fixture, _perfect_chat(self.experiments),
            )
            aggregate_a = build_aggregate(
                BENCHMARK, FIXTURE, result, repo=ROOT, provider="deepseek",
                created_at="2026-01-01T00:00:00Z",
            )
            aggregate_b = build_aggregate(
                BENCHMARK, FIXTURE, result, repo=ROOT, provider="deepseek",
                created_at="2026-08-16T00:00:00Z",
            )
            case_id = next(iter(aggregate_b["conditions"][RAW_BRONZE]["cases"]))
            case_record = aggregate_b["conditions"][RAW_BRONZE]["cases"][case_id]
            case_record["raw_response"] = "same semantics, different bytes"
            case_record["raw_response_sha256"] = hashlib.sha256(
                b"same semantics, different bytes",
            ).hexdigest()
            left = Path(tmp) / "left"
            right = Path(tmp) / "right"
            publish_artifact(left, aggregate_a)
            publish_artifact(right, aggregate_b)
            self.assertEqual(compare_artifacts(left, right), [])

            wrong = copy.deepcopy(aggregate_b)
            wrong["conditions"][RAW_BRONZE]["metrics"]["endpoint_recall"]["rate"] = 0.5
            wrong_dir = Path(tmp) / "wrong"
            publish_artifact(wrong_dir, wrong)
            differences = compare_artifacts(left, wrong_dir)
            self.assertTrue(
                any("RAW_BRONZE: metrics differ" in item for item in differences),
            )


if __name__ == "__main__":
    unittest.main()
