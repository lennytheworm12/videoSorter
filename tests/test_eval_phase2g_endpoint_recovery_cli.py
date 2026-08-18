import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.eval_phase2g_endpoint_recovery as cli
from pipeline.phase2g_endpoint_recovery import (
    CONDITIONS,
    REFERENCE_ENDPOINT,
    REFERENCE_MODEL,
    REFERENCE_THINKING,
    build_aggregate,
    publish_artifact,
    run_experiment,
)
from pipeline.phase2g_silver import (
    load_silver_fixture,
    validate_fixture_against_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


def _experiments():
    benchmark = cli.load_benchmark(cli.DEFAULT_BENCHMARK)
    fixture = load_silver_fixture(cli.DEFAULT_SILVER)
    validate_fixture_against_benchmark(benchmark, fixture)
    from pipeline.phase2g_endpoint_recovery import build_case_experiment
    return benchmark, fixture, {
        case["id"]: build_case_experiment(case) for case in benchmark["cases"]
    }


def _perfect_response(experiment):
    alias_by_span = {
        (record["start"], record["end"]): record["alias"]
        for record in experiment["catalog"]
    }
    response = {"endpoint_selections": {}, "reference_statuses": {}}
    for task in experiment["endpoint_tasks"]:
        span = task["gold_spans"][0]
        response["endpoint_selections"][task["task_id"]] = {
            "roles": {task["gold_node_types"][0]: [alias_by_span[span]]},
            "status": "NONE",
        }
    for task in experiment["status_tasks"]:
        response["reference_statuses"][task["task_id"]] = {
            "status": "UNKNOWN", "targets": [],
        }
    return json.dumps(response)


class Phase2GEvalCliTests(unittest.TestCase):
    def test_no_provider_validation_runs_by_default(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main([
                "--benchmark", str(cli.DEFAULT_BENCHMARK),
                "--silver", str(cli.DEFAULT_SILVER),
            ])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("no-provider validation passed", text)
        self.assertIn("33/33", text)
        self.assertIn("status tasks: 8/8", text)
        self.assertIn("15 deterministic case-level requests", text)
        self.assertIn("artifact records carry full catalogs", text)
        with patch.object(cli, "run_experiment", side_effect=AssertionError("provider called")):
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main([
                    "--benchmark", str(cli.DEFAULT_BENCHMARK),
                    "--silver", str(cli.DEFAULT_SILVER),
                ])

    def test_no_provider_validation_enforces_artifact_catalog_fields(self):
        original = cli.build_case_experiment

        def stripped(case):
            experiment = original(case)
            experiment["catalog"] = [
                {key: value for key, value in record.items()
                 if key != "absolute_start"}
                for record in experiment["catalog"]
            ]
            return experiment

        with patch.object(
            cli, "build_case_experiment", side_effect=stripped,
        ):
            with self.assertRaises(ValueError):
                cli.main([
                    "--benchmark", str(cli.DEFAULT_BENCHMARK),
                    "--silver", str(cli.DEFAULT_SILVER),
                ])

    def test_output_requires_live_and_live_requires_output(self):
        with self.assertRaises(SystemExit):
            cli.main(["--output", "/tmp/phase2g-should-fail"])
        with self.assertRaises(SystemExit):
            cli.main(["--live"])

    def test_live_rejects_repo_output_and_existing_directories(self):
        with self.assertRaises(SystemExit):
            cli.main([
                "--live", "--output", str(ROOT / "phase2g-in-repo"),
            ])
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "exists"
            existing.mkdir()
            with self.assertRaises(SystemExit):
                cli.main([
                    "--live", "--output", str(existing),
                ])

    def test_live_mocked_runner_publishes_15_call_artifact(self):
        benchmark, fixture, experiments = _experiments()
        calls = []

        def fake_chat(system, user, **kwargs):
            calls.append(kwargs)
            experiment = next(
                item for item in experiments.values()
                if item["bronze_text"] in user
            )
            return _perfect_response(experiment)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "phase2g-live"
            with patch("core.llm.BACKEND", "deepseek"), patch(
                "core.llm._DEEPSEEK_BASE_URL", REFERENCE_ENDPOINT,
            ), patch("core.llm.chat", side_effect=fake_chat):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = cli.main([
                        "--live", "--output", str(output),
                        "--benchmark", str(cli.DEFAULT_BENCHMARK),
                        "--silver", str(cli.DEFAULT_SILVER),
                    ])
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 15)
            self.assertTrue((output / "phase2g-endpoint-recovery.json").exists())
            self.assertTrue((output / "MANIFEST.json").exists())
            for condition in CONDITIONS:
                self.assertTrue((output / "conditions" / condition).exists())
            aggregate = json.loads(
                (output / "phase2g-endpoint-recovery.json").read_text(encoding="utf-8"),
            )
            self.assertTrue(aggregate["promotion_gate"]["passed"])
            self.assertEqual(
                aggregate["promotion_gate"]["satisfied_conditions"], list(CONDITIONS),
            )
            self.assertEqual(aggregate["definition"]["model"], REFERENCE_MODEL)
            self.assertEqual(aggregate["definition"]["thinking"], REFERENCE_THINKING)
            for _, kwargs in zip(calls, calls):
                self.assertEqual(kwargs["temperature"], 0.0)
            text = out.getvalue()
            self.assertIn("promotion gate passed=True", text)

    def test_live_requires_official_deepseek_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("core.llm.BACKEND", "gemini"):
                with self.assertRaises(SystemExit):
                    cli.main([
                        "--live", "--output", str(Path(tmp) / "out"),
                    ])
            with patch("core.llm.BACKEND", "deepseek"), patch(
                "core.llm._DEEPSEEK_BASE_URL", "https://example.invalid",
            ):
                with self.assertRaises(SystemExit):
                    cli.main([
                        "--live", "--output", str(Path(tmp) / "out"),
                    ])

    def test_compare_mode_matches_clean_rerun_and_flags_score_changes(self):
        benchmark, fixture, experiments = _experiments()

        def fake_chat(system, user, **kwargs):
            experiment = next(
                item for item in experiments.values()
                if item["bronze_text"] in user
            )
            return _perfect_response(experiment)

        with tempfile.TemporaryDirectory() as tmp:
            result = run_experiment(benchmark, fixture, fake_chat)
            first = build_aggregate(
                cli.DEFAULT_BENCHMARK, cli.DEFAULT_SILVER, result,
                repo=ROOT, provider="deepseek", created_at="2026-01-01T00:00:00Z",
            )
            second = build_aggregate(
                cli.DEFAULT_BENCHMARK, cli.DEFAULT_SILVER, result,
                repo=ROOT, provider="deepseek", created_at="2026-08-16T00:00:00Z",
            )
            left = Path(tmp) / "left"
            right = Path(tmp) / "right"
            publish_artifact(left, first)
            publish_artifact(right, second)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cli.main([
                    "--compare-left", str(left),
                    "--compare-right", str(right),
                ])
            self.assertEqual(code, 0)
            self.assertIn("artifacts match", out.getvalue())

            wrong = json.loads(
                (right / "phase2g-endpoint-recovery.json").read_text(encoding="utf-8"),
            )
            wrong["conditions"][CONDITIONS[0]]["metrics"]["endpoint_recall"]["rate"] = 0.2
            wrong_dir = Path(tmp) / "wrong"
            publish_artifact(wrong_dir, wrong)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cli.main([
                    "--compare-left", str(left),
                    "--compare-right", str(wrong_dir),
                ])
            self.assertEqual(code, 1)
            self.assertIn("metrics differ", out.getvalue())

    def test_compare_flags_missing_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                cli.main([
                    "--compare-left", str(Path(tmp) / "a"),
                ])


if __name__ == "__main__":
    unittest.main()
