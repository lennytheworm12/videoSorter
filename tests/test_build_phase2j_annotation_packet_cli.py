import json
from pathlib import Path
import tempfile
import unittest

import scripts.build_phase2j_annotation_packet as cli
from pipeline.phase2j_annotation_packet import (
    PACKET_SCHEMA_VERSION,
    load_annotation_packet,
)
from pipeline.phase2j_source_selection import (
    SELECTION_SCHEMA_VERSION,
    load_selection_manifest,
)
from tests._phase2j_helpers import write_standard_phase2j_inputs


class Phase2JBuildCliTests(unittest.TestCase):
    def _inputs(self, root: Path):
        pool_path, manifest_path, benchmark_path, _, _ = write_standard_phase2j_inputs(root)
        return pool_path, manifest_path, benchmark_path

    def test_cli_builds_deterministic_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool_path, manifest_path, benchmark_path = self._inputs(root)
            output = root / "phase2j"
            code = cli.main([
                "--pool", str(pool_path),
                "--legacy-manifest", str(manifest_path),
                "--legacy-benchmark", str(benchmark_path),
                "--output-dir", str(output),
            ])
            self.assertEqual(code, 0)
            manifest_file = output / cli.MANIFEST_FILENAME
            packet_file = output / cli.PACKET_FILENAME
            self.assertTrue(manifest_file.is_file())
            self.assertTrue(packet_file.is_file())
            manifest = load_selection_manifest(manifest_file)
            packet = load_annotation_packet(packet_file, manifest=manifest)
            self.assertEqual(manifest["schema_version"], SELECTION_SCHEMA_VERSION)
            self.assertEqual(packet["schema_version"], PACKET_SCHEMA_VERSION)
            self.assertEqual(packet["selection_manifest_sha256"], manifest["content_sha256"])
            self.assertEqual(manifest["release_gate"], "LOCKED")
            self.assertEqual(packet["release_gate"], "LOCKED")
            first_manifest = manifest_file.read_bytes()
            first_packet = packet_file.read_bytes()

            rerun = cli.main([
                "--pool", str(pool_path),
                "--legacy-manifest", str(manifest_path),
                "--legacy-benchmark", str(benchmark_path),
                "--output-dir", str(output),
            ])
            self.assertEqual(rerun, 0)
            self.assertEqual(manifest_file.read_bytes(), first_manifest)
            self.assertEqual(packet_file.read_bytes(), first_packet)

    def test_cli_fails_closed_on_mismatched_preexisting_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool_path, manifest_path, benchmark_path = self._inputs(root)
            output = root / "phase2j"
            self.assertEqual(cli.main([
                "--pool", str(pool_path),
                "--legacy-manifest", str(manifest_path),
                "--legacy-benchmark", str(benchmark_path),
                "--output-dir", str(output),
            ]), 0)
            manifest_file = output / cli.MANIFEST_FILENAME
            packet_file = output / cli.PACKET_FILENAME
            tampered = json.loads(manifest_file.read_text(encoding="utf-8"))
            tampered["purpose"] = "tampered"
            tampered_bytes = (
                json.dumps(tampered, sort_keys=True, indent=2) + "\n"
            ).encode("utf-8")
            manifest_file.write_bytes(tampered_bytes)
            packet_bytes = packet_file.read_bytes()

            code = cli.main([
                "--pool", str(pool_path),
                "--legacy-manifest", str(manifest_path),
                "--legacy-benchmark", str(benchmark_path),
                "--output-dir", str(output),
            ])
            self.assertEqual(code, 1)
            self.assertEqual(manifest_file.read_bytes(), tampered_bytes)
            self.assertEqual(packet_file.read_bytes(), packet_bytes)

    def test_cli_fails_closed_on_incomplete_output_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool_path, manifest_path, benchmark_path = self._inputs(root)
            output = root / "phase2j"
            self.assertEqual(cli.main([
                "--pool", str(pool_path),
                "--legacy-manifest", str(manifest_path),
                "--legacy-benchmark", str(benchmark_path),
                "--output-dir", str(output),
            ]), 0)
            manifest_file = output / cli.MANIFEST_FILENAME
            packet_file = output / cli.PACKET_FILENAME
            manifest_bytes = manifest_file.read_bytes()
            packet_file.unlink()
            code = cli.main([
                "--pool", str(pool_path),
                "--legacy-manifest", str(manifest_path),
                "--legacy-benchmark", str(benchmark_path),
                "--output-dir", str(output),
            ])
            self.assertEqual(code, 1)
            self.assertEqual(manifest_file.read_bytes(), manifest_bytes)
            self.assertFalse(packet_file.exists())

    def test_cli_validate_only_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool_path, manifest_path, benchmark_path = self._inputs(root)
            output = root / "phase2j"
            self.assertEqual(cli.main([
                "--pool", str(pool_path),
                "--legacy-manifest", str(manifest_path),
                "--legacy-benchmark", str(benchmark_path),
                "--output-dir", str(output),
            ]), 0)
            self.assertEqual(cli.main([
                "--pool", str(pool_path),
                "--legacy-manifest", str(manifest_path),
                "--legacy-benchmark", str(benchmark_path),
                "--output-dir", str(output),
                "--validate-only",
            ]), 0)
            packet_file = output / cli.PACKET_FILENAME
            tampered = json.loads(packet_file.read_text(encoding="utf-8"))
            tampered["purpose"] = "tampered"
            packet_file.write_text(
                json.dumps(tampered, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(cli.main([
                "--pool", str(pool_path),
                "--legacy-manifest", str(manifest_path),
                "--legacy-benchmark", str(benchmark_path),
                "--output-dir", str(output),
                "--validate-only",
            ]), 1)


if __name__ == "__main__":
    unittest.main()
