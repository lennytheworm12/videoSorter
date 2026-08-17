import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import scripts.eval_phase2h_endpoint_scoring as cli
from pipeline.phase2h_endpoint_scoring import (
    RUN_VERSION,
    build_aggregate,
    build_dataset,
    build_window_table,
    canonical_sha256,
    classify_all_errors,
    compute_cell_metrics,
    compute_rankings,
    load_benchmark,
    publish_artifact,
    run_cv,
    strongest_features,
)


ROOT = Path(__file__).resolve().parents[1]


class Phase2HEvalCliTests(unittest.TestCase):
    def test_offline_run_prints_summary_without_publishing(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main([
                "--benchmark", str(cli.DEFAULT_BENCHMARK),
                "--cell", "logistic_A",
            ])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn(RUN_VERSION, text)
        self.assertIn("33/33", text)
        self.assertIn("logistic_A", text)
        self.assertIn("leave-one-window-out", text)

    def test_output_requires_clean_tree_and_rejects_repo_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_output = ROOT / "phase2h-in-repo"
            with self.assertRaises(SystemExit):
                cli.main([
                    "--output", str(repo_output),
                    "--cell", "logistic_A",
                ])
            existing = Path(tmp) / "exists"
            existing.mkdir()
            with patch(
                "scripts.eval_phase2h_endpoint_scoring._repository_dirty",
                return_value=False,
            ):
                with self.assertRaises(SystemExit):
                    cli.main([
                        "--output", str(existing),
                        "--cell", "logistic_A",
                    ])
            with patch(
                "scripts.eval_phase2h_endpoint_scoring._repository_dirty",
                return_value=True,
            ):
                with self.assertRaises(SystemExit) as caught:
                    cli.main([
                        "--output", str(Path(tmp) / "dirty-tree"),
                        "--cell", "logistic_A",
                    ])
                self.assertIn("clean committed tree", str(caught.exception))

    def test_publish_from_clean_tree_writes_immutable_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "phase2h-live"
            with patch(
                "scripts.eval_phase2h_endpoint_scoring._repository_dirty",
                return_value=False,
            ), patch(
                "pipeline.phase2h_endpoint_scoring._git_state",
                return_value=("testcommit", False),
            ):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = cli.main([
                        "--output", str(output),
                        "--benchmark", str(cli.DEFAULT_BENCHMARK),
                        "--cell", "logistic_A",
                    ])
            self.assertEqual(code, 0)
            self.assertTrue(
                (output / "phase2h-endpoint-scoring.json").exists(),
            )
            self.assertTrue((output / "MANIFEST.json").exists())
            self.assertEqual(len(list((output / "windows").iterdir())), 5)
            aggregate = json.loads(
                (output / "phase2h-endpoint-scoring.json").read_text(
                    encoding="utf-8",
                ),
            )
            self.assertEqual(aggregate["run_version"], RUN_VERSION)
            self.assertFalse(aggregate["repository_dirty"])
            self.assertEqual(aggregate["git_commit"], "testcommit")
            self.assertEqual(
                aggregate["metrics"]["logistic_A"]["label_keep_count"], 33,
            )
            inner = {
                key: value for key, value in aggregate.items()
                if key != "content_sha256"
            }
            self.assertEqual(
                aggregate["content_sha256"], canonical_sha256(inner),
            )
            manifest = json.loads(
                (output / "MANIFEST.json").read_text(encoding="utf-8"),
            )
            for entry in manifest["files"]:
                path = output / entry["path"]
                self.assertEqual(
                    entry["file_sha256"],
                    __import__("hashlib").sha256(
                        path.read_bytes(),
                    ).hexdigest(),
                )
            text = out.getvalue()
            self.assertIn("published immutable artifact", text)
            with self.assertRaises(SystemExit):
                cli.main(["--output", str(output), "--cell", "logistic_A"])

    def test_compare_mode_matches_clean_rerun_and_flags_changes(self):
        benchmark = load_benchmark(cli.DEFAULT_BENCHMARK)
        dataset = build_dataset(benchmark)
        cv = run_cv(dataset, cells=("logistic_A",))
        rankings = compute_rankings(dataset, cv["oof_scores"], cells=("logistic_A",))
        errors = classify_all_errors(dataset, rankings, cells=("logistic_A",))
        metrics = {
            "logistic_A": compute_cell_metrics(
                dataset, rankings, "logistic_A",
            ),
        }
        result = {
            "dataset": dataset,
            "folds": cv["folds"],
            "fit_scope": cv["fit_scope"],
            "rankings": rankings,
            "errors": errors,
            "metrics": metrics,
            "strongest_features": strongest_features(
                cv["fitted_models"], cells=("logistic_A",),
            ),
            "cells": ["logistic_A"],
        }

        def tables_and_hashes():
            tables = {
                window_id: build_window_table(
                    dataset, rankings, errors, window_id,
                    cells=("logistic_A",),
                )
                for window_id in sorted(dataset["windows"])
            }
            return tables, {
                window_id: canonical_sha256(table)
                for window_id, table in tables.items()
            }

        with tempfile.TemporaryDirectory() as tmp:
            first_tables, first_hashes = tables_and_hashes()
            second_tables, second_hashes = tables_and_hashes()
            with patch(
                "pipeline.phase2h_endpoint_scoring._git_state",
                return_value=("commit", False),
            ):
                first = build_aggregate(
                    cli.DEFAULT_BENCHMARK, result, repo=ROOT,
                    created_at="2026-01-01T00:00:00Z",
                    window_table_hashes=first_hashes,
                )
                second = build_aggregate(
                    cli.DEFAULT_BENCHMARK, result, repo=ROOT,
                    created_at="2026-08-16T00:00:00Z",
                    window_table_hashes=second_hashes,
                )
            left = Path(tmp) / "left"
            right = Path(tmp) / "right"
            publish_artifact(left, first, first_tables)
            publish_artifact(right, second, second_tables)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cli.main([
                    "--compare-left", str(left),
                    "--compare-right", str(right),
                ])
            self.assertEqual(code, 0)
            self.assertIn("artifacts match", out.getvalue())

            wrong = json.loads(
                (right / "phase2h-endpoint-scoring.json").read_text(
                    encoding="utf-8",
                ),
            )
            wrong["metrics"]["logistic_A"]["recall"]["rate"] = 0.1
            wrong_dir = Path(tmp) / "wrong"
            publish_artifact(wrong_dir, wrong, second_tables)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cli.main([
                    "--compare-left", str(left),
                    "--compare-right", str(wrong_dir),
                ])
            self.assertEqual(code, 1)
            self.assertIn("differ", out.getvalue())

    def test_compare_requires_both_sides(self):
        with self.assertRaises(SystemExit):
            cli.main(["--compare-left", "/tmp/phase2h-left"])

    def test_duplicate_cells_are_rejected(self):
        with self.assertRaises(SystemExit) as caught:
            cli.main(["--cell", "logistic_A", "--cell", "logistic_A"])
        self.assertIn("duplicate --cell", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
