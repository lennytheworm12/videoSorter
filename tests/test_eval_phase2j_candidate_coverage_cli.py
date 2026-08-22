"""CLI tests for the Phase 2J candidate-coverage evaluator."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import scripts.eval_phase2j_candidate_coverage as cli
from pipeline.phase2j_candidate_coverage import (
    load_candidate_coverage,
)
from tests.test_phase2j_candidate_coverage import _fixture
from tests.test_phase2j_adjudication_import import _serialize


def _args(fixture, output, *, extra=()):
    return [
        "--manifest", str(fixture["manifest_path"]),
        "--packet", str(fixture["packet_path"]),
        "--output", str(output),
        *extra,
    ]


class Phase2JCandidateCoverageCliTests(unittest.TestCase):
    def test_cli_builds_artifact_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _fixture(root)
            output = root / "coverage.json"
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                code = cli.main(_args(fixture, output))
            self.assertEqual(code, 0)
            self.assertTrue(output.is_file())
            summary = json.loads(captured.getvalue())
            loaded = load_candidate_coverage(
                output,
                manifest_path=fixture["manifest_path"],
                packet_path=fixture["packet_path"],
            )
            self.assertEqual(
                summary["coverage_content_sha256"], loaded["content_sha256"],
            )
            self.assertEqual(summary["aggregate"], loaded["coverage"]["aggregate"])
            self.assertEqual(
                summary["total_candidates"], loaded["coverage"]["total_candidates"],
            )
            self.assertEqual(
                summary["per_partition"], loaded["coverage"]["per_partition"],
            )
            first_bytes = output.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                rerun = cli.main(_args(fixture, output))
            self.assertEqual(rerun, 0)
            self.assertEqual(output.read_bytes(), first_bytes)
            with contextlib.redirect_stdout(io.StringIO()):
                validate = cli.main(
                    _args(fixture, output, extra=["--validate-only"]),
                )
            self.assertEqual(validate, 0)

    def test_cli_fails_closed_on_conflicting_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _fixture(root)
            output = root / "coverage.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"tampered": true}\n', encoding="utf-8")
            original = output.read_bytes()
            code = cli.main(_args(fixture, output))
            self.assertEqual(code, 1)
            self.assertEqual(output.read_bytes(), original)

    def test_cli_validate_only_rejects_tampered_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _fixture(root)
            output = root / "coverage.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(_args(fixture, output)), 0)
            artifact = json.loads(output.read_text(encoding="utf-8"))
            artifact["purpose"] = "tampered"
            from pipeline.phase2j_source_selection import canonical_sha256
            inner = {
                key: value for key, value in artifact.items()
                if key != "content_sha256"
            }
            artifact["content_sha256"] = canonical_sha256(inner)
            output.write_text(_serialize(artifact), encoding="utf-8")
            tampered_bytes = output.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                code = cli.main(
                    _args(fixture, output, extra=["--validate-only"]),
                )
            self.assertEqual(code, 1)
            self.assertEqual(output.read_bytes(), tampered_bytes)

    def test_cli_rejects_missing_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _fixture(root)
            missing = root / "missing-manifest.json"
            with self.assertRaises(SystemExit):
                cli.main([
                    "--manifest", str(missing),
                    "--packet", str(fixture["packet_path"]),
                    "--output", str(root / "coverage.json"),
                ])
            self.assertFalse((root / "coverage.json").exists())


if __name__ == "__main__":
    unittest.main()
