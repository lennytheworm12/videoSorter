"""CLI tests for the Phase 2J adjudication packet builder."""

import json
from pathlib import Path
import tempfile
import unittest

import scripts.build_phase2j_adjudication_packet as cli
from pipeline.phase2j_adjudication import (
    ADJUDICATION_PACKET_SCHEMA_VERSION,
    load_adjudication_packet,
    validate_adjudication_packet,
)
from tests._phase2j_helpers import (
    write_human_session,
    write_sol_review,
    write_standard_phase2j_inputs,
)
from tests.test_phase2j_adjudication import _build_packet, _known_spans


class Phase2JAdjudicationCliTests(unittest.TestCase):
    def _inputs(self, root: Path):
        packet = _build_packet(root)
        human_path = root / "human-session.json"
        sol_path = root / "sol-review.json"
        write_human_session(
            human_path, packet, spans_for_window=lambda record: _known_spans(record)[0],
        )
        write_sol_review(
            sol_path, packet, spans_for_window=lambda record: _known_spans(record)[1],
        )
        return packet, human_path, sol_path

    def test_cli_builds_deterministic_packet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human_path, sol_path = self._inputs(root)
            packet_path = root / "locked-packet.json"
            packet_path.write_text(
                json.dumps(packet, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            output = root / "phase2j"
            code = cli.main([
                "--packet", str(packet_path),
                "--human", str(human_path),
                "--sol", str(sol_path),
                "--output-dir", str(output),
            ])
            self.assertEqual(code, 0)
            output_file = output / cli.OUTPUT_FILENAME
            self.assertTrue(output_file.is_file())
            built = load_adjudication_packet(output_file)
            validate_adjudication_packet(built)
            self.assertEqual(built["schema_version"], ADJUDICATION_PACKET_SCHEMA_VERSION)
            self.assertEqual(built["packet_sha256"], packet["content_sha256"])
            first_bytes = output_file.read_bytes()

            rerun = cli.main([
                "--packet", str(packet_path),
                "--human", str(human_path),
                "--sol", str(sol_path),
                "--output-dir", str(output),
            ])
            self.assertEqual(rerun, 0)
            self.assertEqual(output_file.read_bytes(), first_bytes)

    def test_cli_fails_closed_on_tampered_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human_path, sol_path = self._inputs(root)
            packet_path = root / "locked-packet.json"
            packet_path.write_text(
                json.dumps(packet, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            output = root / "phase2j"
            self.assertEqual(cli.main([
                "--packet", str(packet_path),
                "--human", str(human_path),
                "--sol", str(sol_path),
                "--output-dir", str(output),
            ]), 0)
            output_file = output / cli.OUTPUT_FILENAME
            tampered = json.loads(output_file.read_text(encoding="utf-8"))
            tampered["purpose"] = "tampered"
            tampered_bytes = (
                json.dumps(tampered, sort_keys=True, indent=2) + "\n"
            ).encode("utf-8")
            output_file.write_bytes(tampered_bytes)

            code = cli.main([
                "--packet", str(packet_path),
                "--human", str(human_path),
                "--sol", str(sol_path),
                "--output-dir", str(output),
            ])
            self.assertEqual(code, 1)
            self.assertEqual(output_file.read_bytes(), tampered_bytes)

    def test_cli_validate_only_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human_path, sol_path = self._inputs(root)
            packet_path = root / "locked-packet.json"
            packet_path.write_text(
                json.dumps(packet, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            output = root / "phase2j"
            self.assertEqual(cli.main([
                "--packet", str(packet_path),
                "--human", str(human_path),
                "--sol", str(sol_path),
                "--output-dir", str(output),
            ]), 0)
            self.assertEqual(cli.main([
                "--packet", str(packet_path),
                "--human", str(human_path),
                "--sol", str(sol_path),
                "--output-dir", str(output),
                "--validate-only",
            ]), 0)
            output_file = output / cli.OUTPUT_FILENAME
            tampered = json.loads(output_file.read_text(encoding="utf-8"))
            tampered["totals"]["sol_only"] = 0
            output_file.write_text(
                json.dumps(tampered, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(cli.main([
                "--packet", str(packet_path),
                "--human", str(human_path),
                "--sol", str(sol_path),
                "--output-dir", str(output),
                "--validate-only",
            ]), 1)

    def test_cli_rejects_missing_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet, human_path, sol_path = self._inputs(root)
            missing = root / "missing.json"
            with self.assertRaises(SystemExit):
                cli.main([
                    "--packet", str(missing),
                    "--human", str(human_path),
                    "--sol", str(sol_path),
                    "--output-dir", str(root / "phase2j"),
                ])


if __name__ == "__main__":
    unittest.main()
