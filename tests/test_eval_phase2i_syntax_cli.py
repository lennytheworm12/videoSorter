import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import scripts.eval_phase2i_syntax as cli
from pipeline.phase2g_silver import canonical_sha256
from pipeline.phase2i_endpoint_scoring import (
    CELLS_C,
    RUN_VERSION,
    build_aggregate_c,
    build_phase2i_window_table,
    close_phase2h_baseline,
    load_phase2h_baseline,
    publish_phase2i_artifact,
    run_experiment_c,
)
from pipeline.phase2i_syntax import verify_assets_provenance
from tests._phase2i_helpers import experiment_result as _shared_experiment


ROOT = Path(__file__).resolve().parents[1]
TEST_GIT_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
    text=True, capture_output=True,
).stdout.strip()

def _experiment():
    return _shared_experiment()


class Phase2IEvalCliTests(unittest.TestCase):
    def test_offline_run_prints_summary_without_publishing(self):
        if not cli.DEFAULT_ASSETS.is_dir():
            self.skipTest("parser assets not present")
        out = io.StringIO()
        with patch(
            "scripts.eval_phase2i_syntax.run_experiment_c",
            side_effect=lambda *args, **kwargs: _experiment(),
        ), contextlib.redirect_stdout(out):
            code = cli.main([
                "--cell", "logistic_C",
                "--assets-dir", str(cli.DEFAULT_ASSETS),
            ])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn(RUN_VERSION, text)
        self.assertIn("logistic_C", text)
        self.assertIn("leave-one-window-out", text)
        self.assertIn("universally missed gold endpoints: 7", text)

    def test_output_requires_clean_tree_and_rejects_repo_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_output = ROOT / "phase2i-in-repo"
            with self.assertRaises(SystemExit):
                cli.main(["--output", str(repo_output)])
            existing = Path(tmp) / "exists"
            existing.mkdir()
            with patch(
                "scripts.eval_phase2i_syntax._repository_dirty",
                return_value=False,
            ):
                with self.assertRaises(SystemExit):
                    cli.main([
                        "--output", str(existing),
                    ])
            with patch(
                "scripts.eval_phase2i_syntax._repository_dirty",
                return_value=True,
            ):
                with self.assertRaises(SystemExit) as caught:
                    cli.main([
                        "--output", str(Path(tmp) / "dirty-tree"),
                    ])
                self.assertIn(
                    "clean committed tree", str(caught.exception),
                )

    def test_publish_requires_both_c_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "scripts.eval_phase2i_syntax._repository_dirty",
                return_value=False,
            ):
                with self.assertRaises(SystemExit) as caught:
                    cli.main([
                        "--output", str(Path(tmp) / "subset"),
                        "--cell", "logistic_C",
                    ])
                self.assertIn("both C cells", str(caught.exception))

    def test_publish_from_clean_tree_writes_immutable_artifact(self):
        if not cli.DEFAULT_ASSETS.is_dir():
            self.skipTest("parser assets not present")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "phase2i-live"
            with patch(
                "scripts.eval_phase2i_syntax._repository_dirty",
                return_value=False,
            ), patch(
                "pipeline.phase2i_endpoint_scoring._git_state_c",
                return_value=(TEST_GIT_COMMIT, False),
            ), patch(
                "scripts.eval_phase2i_syntax.run_experiment_c",
                side_effect=lambda *args, **kwargs: _experiment(),
            ):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = cli.main([
                        "--output", str(output),
                        "--assets-dir", str(cli.DEFAULT_ASSETS),
                    ])
            self.assertEqual(code, 0)
            self.assertTrue(
                (output / "phase2i-syntax-features.json").exists(),
            )
            self.assertTrue((output / "MANIFEST.json").exists())
            self.assertEqual(len(list((output / "windows").iterdir())), 5)
            self.assertEqual(len(list((output / "parser").iterdir())), 5)
            aggregate = json.loads(
                (output / "phase2i-syntax-features.json").read_text(
                    encoding="utf-8",
                ),
            )
            self.assertEqual(aggregate["run_version"], RUN_VERSION)
            self.assertFalse(aggregate["repository_dirty"])
            self.assertEqual(aggregate["git_commit"], TEST_GIT_COMMIT)
            self.assertEqual(
                aggregate["definition"]["cells"], list(CELLS_C),
            )
            self.assertEqual(
                set(aggregate["metrics"]), set(CELLS_C),
            )
            self.assertEqual(
                aggregate["metrics"]["logistic_C"]["label_keep_count"], 33,
            )
            self.assertEqual(
                aggregate["metrics"]["lightgbm_C"]["label_keep_count"], 33,
            )
            self.assertEqual(
                aggregate["universally_missed_validation"]["validated"],
                True,
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
            self.assertEqual(len(manifest["files"]), 1 + 5 + 5)
            for entry in manifest["files"]:
                path = output / entry["path"]
                self.assertEqual(
                    entry["file_sha256"],
                    __import__("hashlib").sha256(
                        path.read_bytes(),
                    ).hexdigest(),
                )
            window_table = json.loads(
                (
                    output / "windows"
                    / "mid-push-prevents-side-collapse.json"
                ).read_text(encoding="utf-8"),
            )
            predictions = window_table["candidates"][0]["predictions"]
            self.assertIn("logistic_C", predictions)
            self.assertIn("lightgbm_C", predictions)
            text = out.getvalue()
            self.assertIn("published immutable artifact", text)
            with self.assertRaises(SystemExit):
                cli.main([
                    "--output", str(output),
                ])

    def test_publish_detects_head_change_during_build(self):
        if not cli.DEFAULT_ASSETS.is_dir():
            self.skipTest("parser assets not present")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "phase2i-head-change"
            # The before-lock is read from the real repository HEAD, while
            # the aggregate reports a different commit, so the CLI must
            # abort deterministically without touching the real repo.
            with patch(
                "scripts.eval_phase2i_syntax._repository_dirty",
                return_value=False,
            ), patch(
                "pipeline.phase2i_endpoint_scoring._git_state_c",
                return_value=("0" * 40, False),
            ), patch(
                "scripts.eval_phase2i_syntax.run_experiment_c",
                side_effect=lambda *args, **kwargs: _experiment(),
            ):
                with self.assertRaises(SystemExit) as caught:
                    cli.main([
                        "--output", str(output),
                        "--assets-dir", str(cli.DEFAULT_ASSETS),
                    ])
            self.assertIn(
                "repository HEAD changed", str(caught.exception),
            )
            self.assertFalse(output.exists())

    def test_compare_mode_matches_deterministic_rerun(self):
        if not cli.DEFAULT_ASSETS.is_dir():
            self.skipTest("parser assets not present")
        result = _experiment()
        baseline = load_phase2h_baseline(cli.DEFAULT_ARCHIVE)
        try:
            tables = {}
            hashes = {}
            parse_tables = {}
            for window_id in sorted(result["dataset"]["windows"]):
                table = build_phase2i_window_table(
                    result["dataset"], result["rankings"],
                    result["errors"],
                    baseline["window_tables"][window_id],
                    result["parses"][window_id],
                    result["candidate_syntax"][window_id],
                    window_id,
                    cells=CELLS_C,
                )
                tables[window_id] = table
                hashes[window_id] = canonical_sha256(table)
                parse_tables[window_id] = (
                    result["parses"][window_id].to_dict()
                )
        finally:
            close_phase2h_baseline(baseline)
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left"
            right = Path(tmp) / "right"
            provenance = verify_assets_provenance(cli.DEFAULT_ASSETS)
            self.assertTrue(provenance["verified"])
            with patch(
                "pipeline.phase2i_endpoint_scoring._git_state_c",
                return_value=(TEST_GIT_COMMIT, False),
            ):
                first = build_aggregate_c(
                    cli.DEFAULT_BENCHMARK,
                    result,
                    repo=ROOT,
                    created_at="2026-01-01T00:00:00Z",
                    window_table_hashes=hashes,
                    assets_provenance=provenance,
                )
                second = build_aggregate_c(
                    cli.DEFAULT_BENCHMARK,
                    result,
                    repo=ROOT,
                    created_at="2026-08-16T00:00:00Z",
                    window_table_hashes=hashes,
                    assets_provenance=provenance,
                )
            publish_phase2i_artifact(
                left, first, tables, parse_tables,
                benchmark_path=cli.DEFAULT_BENCHMARK,
                baseline_archive=cli.DEFAULT_ARCHIVE,
            )
            publish_phase2i_artifact(
                right, second, tables, parse_tables,
                benchmark_path=cli.DEFAULT_BENCHMARK,
                baseline_archive=cli.DEFAULT_ARCHIVE,
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cli.main([
                    "--compare-left", str(left),
                    "--compare-right", str(right),
                ])
            self.assertEqual(code, 0)
            self.assertIn("artifacts match", out.getvalue())

    def test_compare_requires_both_sides(self):
        with self.assertRaises(SystemExit):
            cli.main(["--compare-left", "/tmp/phase2i-left"])

    def test_duplicate_cells_are_rejected(self):
        with self.assertRaises(SystemExit) as caught:
            cli.main(["--cell", "logistic_C", "--cell", "logistic_C"])
        self.assertIn("duplicate --cell", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
