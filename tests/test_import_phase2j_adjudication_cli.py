"""CLI tests for the Phase 2J post-adjudication canonical import."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import scripts.import_phase2j_adjudication as cli
from pipeline.phase2j_adjudication_import import (
    REVIEWED_PACKET_FILENAME,
    serialize_reviewed_packet,
)
from pipeline.phase2j_annotation_packet import load_annotation_packet
from pipeline.phase2j_source_selection import load_selection_manifest
from tests.test_phase2j_adjudication_import import (
    _build_export,
    _serialize,
    _write,
)
from tests.test_phase2j_adjudication import _build_packet, _known_spans
from tests._phase2j_helpers import (
    build_human_session,
    build_sol_review,
    write_human_session,
    write_sol_review,
    write_standard_phase2j_inputs,
)


def _write_manifest(root: Path):
    """Write the locked selection manifest file for the shared packet fixture."""
    pool_path, manifest_path, benchmark_path, _, _ = write_standard_phase2j_inputs(root)
    from pipeline.semantic_ir_pool import load_semantic_window_pool
    from pipeline.phase2j_source_selection import (
        build_selection_manifest,
        load_legacy_benchmark,
        load_legacy_manifest,
    )

    pool = load_semantic_window_pool(pool_path)
    legacy_manifest = load_legacy_manifest(manifest_path)
    legacy_benchmark = load_legacy_benchmark(benchmark_path, manifest=legacy_manifest)
    manifest = build_selection_manifest(
        pool=pool,
        pool_path=pool_path,
        legacy_manifest=legacy_manifest,
        legacy_manifest_path=manifest_path,
        legacy_benchmark=legacy_benchmark,
        legacy_benchmark_path=benchmark_path,
    )
    manifest_file = root / "window-selection-manifest-v1.json"
    manifest_file.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _inputs(root: Path):
    """Write all import inputs; return the CLI argument path set."""
    packet = _build_packet(root)
    _write_manifest(root)
    blank_path = _write(root / "blank-packet.json", dict(packet))
    human_path = root / "human-session.json"
    sol_path = root / "sol-review.json"
    write_human_session(
        human_path, packet,
        spans_for_window=lambda record: _known_spans(record)[0],
    )
    write_sol_review(
        sol_path, packet,
        spans_for_window=lambda record: _known_spans(record)[1],
    )
    from pipeline.phase2j_adjudication import build_adjudication_packet
    human = build_human_session(
        packet, spans_for_window=lambda record: _known_spans(record)[0],
    )
    sol = build_sol_review(
        packet, spans_for_window=lambda record: _known_spans(record)[1],
    )
    adjudication = build_adjudication_packet(
        packet, human, sol,
        human_session_path=human_path,
        sol_review_path=sol_path,
    )
    adjudication_path = _write(root / "adjudication.json", adjudication)
    export_path = _write(root / "export.json", _build_export(adjudication))
    output = root / "phase2j" / REVIEWED_PACKET_FILENAME
    return {
        "packet": blank_path,
        "manifest": root / "window-selection-manifest-v1.json",
        "human": human_path,
        "adjudication_packet": adjudication_path,
        "export": export_path,
        "output": output,
        "adjudication": adjudication,
    }


def _args(inputs, **overrides):
    values = dict(inputs)
    values.update(overrides)
    return [
        "--packet", str(values["packet"]),
        "--manifest", str(values["manifest"]),
        "--human", str(values["human"]),
        "--adjudication-packet", str(values["adjudication_packet"]),
        "--export", str(values["export"]),
        "--output", str(values["output"]),
    ]


class Phase2JImportCliTests(unittest.TestCase):
    def test_cli_imports_reviewed_packet_and_prints_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _inputs(root)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli.main(_args(inputs))
            self.assertEqual(code, 0)
            self.assertTrue(inputs["output"].is_file())
            manifest = load_selection_manifest(inputs["manifest"])
            reviewed = load_annotation_packet(
                inputs["output"], manifest=manifest,
            )
            self.assertEqual(reviewed["release_gate"], "LOCKED")
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["release_gate"], "LOCKED")
            self.assertEqual(summary["reviewed_packet_sha256"], reviewed["content_sha256"])
            self.assertEqual(summary["window_statuses"]["REVIEWED"], 30)
            self.assertEqual(summary["pass_a_statuses"]["COMPLETE"], 30)
            self.assertEqual(summary["pass_b_statuses"]["COMPLETE"], 30)
            self.assertEqual(summary["gold_eligible_windows"], 30)
            self.assertEqual(summary["endpoint_count"], reviewed["records"][0]["pass_a"]["endpoint_count"] * 30)
            self.assertEqual(summary["outcomes"], {"CLEAN": 30})
            self.assertIn(
                summary["export_sha256"][:12],
                reviewed["records"][0]["pass_b"]["notes"][0],
            )

            first_bytes = inputs["output"].read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                rerun = cli.main(_args(inputs))
            self.assertEqual(rerun, 0)
            self.assertEqual(inputs["output"].read_bytes(), first_bytes)

    def test_cli_fails_closed_on_conflicting_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _inputs(root)
            inputs["output"].parent.mkdir(parents=True, exist_ok=True)
            inputs["output"].write_text('{"tampered": true}\n', encoding="utf-8")
            original = inputs["output"].read_bytes()
            code = cli.main(_args(inputs))
            self.assertEqual(code, 1)
            self.assertEqual(inputs["output"].read_bytes(), original)

    def test_cli_validate_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _inputs(root)
            self.assertEqual(cli.main(_args(inputs)), 0)
            with contextlib.redirect_stdout(io.StringIO()):
                code = cli.main(_args(inputs) + ["--validate-only"])
            self.assertEqual(code, 0)

            reviewed = json.loads(inputs["output"].read_text(encoding="utf-8"))
            reviewed["purpose"] = "tampered"
            inputs["output"].write_text(_serialize(reviewed), encoding="utf-8")
            tampered_bytes = inputs["output"].read_bytes()
            self.assertEqual(
                cli.main(_args(inputs) + ["--validate-only"]), 1,
            )
            self.assertEqual(inputs["output"].read_bytes(), tampered_bytes)

    def test_cli_rejects_invalid_export_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _inputs(root)
            export = json.loads(inputs["export"].read_text(encoding="utf-8"))
            export["audit_checks"]["boundaries"] = False
            invalid_export = _write(root / "invalid-export.json", export)
            self.assertEqual(
                cli.main(_args(inputs, export=invalid_export)), 1,
            )
            self.assertFalse(inputs["output"].exists())

    def test_cli_rejects_missing_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = _inputs(root)
            missing = root / "missing-export.json"
            with self.assertRaises(SystemExit):
                cli.main(_args(inputs, export=missing))


if __name__ == "__main__":
    unittest.main()
