import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import scripts.finalize_phase2j_annotation_packet as cli
from pipeline.phase2j_annotation_packet import (
    build_annotation_packet,
    load_annotation_packet,
    token_table,
)
from pipeline.phase2j_source_selection import canonical_sha256
from tests._phase2j_helpers import write_standard_phase2j_inputs


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_standard(root: Path):
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
    return manifest, manifest_file


@contextlib.contextmanager
def _setup():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest, manifest_file = _write_standard(root)
        packet_file = root / "endpoint-annotation-packet-v1.json"
        yield manifest, manifest_file, packet_file


def _write_packet(path: Path, packet) -> bytes:
    text = json.dumps(packet, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return text.encode("utf-8")


def _valid_human_edit(blank_packet):
    """Mark record 1 as a clean completed Pass A / pending Pass B review."""
    packet = copy.deepcopy(dict(blank_packet))
    record = copy.deepcopy(dict(packet["records"][0]))
    record["window_status"] = "IN_REVIEW"
    record["pass_a"] = {
        "status": "COMPLETE",
        "reviewer": "reviewer-a",
        "completed_at": "2026-08-18T00:00:00Z",
        "notes": [],
        "endpoint_count": 1,
    }
    tokens = record["tokens"]
    char_start = tokens[0]["start"]
    char_end = tokens[1]["end"]
    record["endpoints"] = [{
        "endpoint_id": f"p2j:{record['window_id']}:ep:0001",
        "bronze_text": record["bronze_text"][char_start:char_end],
        "char_start": char_start,
        "char_end": char_end,
        "token_start": 0,
        "token_end": 1,
        "node_type": "ENTITY",
        "ambiguity_state": "NONE",
        "disposition": "KEEP",
        "adjudication_requested": False,
        "notes": "",
        "pass_provenance": "PASS_A",
    }]
    record["pass_b"] = {
        "status": "PENDING",
        "reviewer": None,
        "completed_at": None,
        "notes": [],
        "audit_checks": {key: False for key in (
            "boundaries", "omissions", "roles", "duplicates", "ambiguity",
        )},
    }
    records = [dict(item) for item in packet["records"]]
    records[0] = record
    packet["records"] = records
    # content_sha256 intentionally left stale.
    return packet


def _fully_consistent_bronze_tamper(packet, record_index=0):
    """Tamper Bronze text and rebuild every derived field plus the self-hash."""
    tampered = copy.deepcopy(dict(packet))
    records = [dict(item) for item in tampered["records"]]
    record = records[record_index]
    record["bronze_text"] = record["bronze_text"] + " fabricated"
    record["bronze_text_sha256"] = _sha256_text(record["bronze_text"])
    record["bronze_char_length"] = len(record["bronze_text"])
    record["upstream_end"] = record["upstream_start"] + len(record["bronze_text"])
    record["tokens"] = token_table(record["bronze_text"])
    tampered["records"] = records
    inner = {key: value for key, value in tampered.items() if key != "content_sha256"}
    tampered["content_sha256"] = canonical_sha256(inner)
    return tampered


class Phase2JFinalizeCliTests(unittest.TestCase):
    def test_finalization_recomputes_stale_hash_for_valid_edit(self):
        with _setup() as (manifest, manifest_file, packet_file):
            blank = build_annotation_packet(manifest)
            edited = _valid_human_edit(blank)
            _write_packet(packet_file, edited)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli.main([
                    "--packet", str(packet_file),
                    "--manifest", str(manifest_file),
                ])
            self.assertEqual(code, 0)
            reloaded = load_annotation_packet(packet_file, manifest=manifest)
            self.assertEqual(reloaded["content_sha256"], canonical_sha256({
                key: value for key, value in reloaded.items() if key != "content_sha256"
            }))
            self.assertEqual(reloaded["records"][0]["window_status"], "IN_REVIEW")
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["window_statuses"]["IN_REVIEW"], 1)
            self.assertEqual(summary["pass_a_statuses"]["COMPLETE"], 1)
            self.assertEqual(summary["pass_b_statuses"]["PENDING"], 1)
            self.assertEqual(summary["endpoint_count"], 1)
            self.assertEqual(summary["gold_eligible_windows"], 0)
            self.assertEqual(summary["release_gate"], "LOCKED")

    def test_invalid_edit_leaves_original_bytes_untouched(self):
        with _setup() as (manifest, manifest_file, packet_file):
            blank = build_annotation_packet(manifest)
            tampered = _fully_consistent_bronze_tamper(blank)
            original = _write_packet(packet_file, tampered)

            code = cli.main([
                "--packet", str(packet_file),
                "--manifest", str(manifest_file),
            ])
            self.assertEqual(code, 1)
            self.assertEqual(packet_file.read_bytes(), original)

    def test_check_only_requires_correct_hash_and_never_writes(self):
        with _setup() as (manifest, manifest_file, packet_file):
            blank = build_annotation_packet(manifest)
            blank_bytes = _write_packet(packet_file, blank)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli.main([
                    "--packet", str(packet_file),
                    "--manifest", str(manifest_file),
                    "--check-only",
                ])
            self.assertEqual(code, 0)
            self.assertEqual(packet_file.read_bytes(), blank_bytes)
            self.assertEqual(json.loads(output.getvalue())["gold_eligible_windows"], 0)

            # A valid human edit with a stale hash must fail check-only.
            edited = _valid_human_edit(blank)
            edited_bytes = _write_packet(packet_file, edited)
            code = cli.main([
                "--packet", str(packet_file),
                "--manifest", str(manifest_file),
                "--check-only",
            ])
            self.assertEqual(code, 1)
            self.assertEqual(packet_file.read_bytes(), edited_bytes)

            # An internally consistent Bronze tamper with a correct self-hash
            # must still fail check-only via the manifest cross-binding.
            bad = _fully_consistent_bronze_tamper(blank)
            bad_bytes = _write_packet(packet_file, bad)
            code = cli.main([
                "--packet", str(packet_file),
                "--manifest", str(manifest_file),
                "--check-only",
            ])
            self.assertEqual(code, 1)
            self.assertEqual(packet_file.read_bytes(), bad_bytes)

    def test_duplicate_key_rejection_fails_closed(self):
        with _setup() as (manifest, manifest_file, packet_file):
            blank = build_annotation_packet(manifest)
            inner = {
                key: value for key, value in blank.items() if key != "content_sha256"
            }
            base = json.dumps(inner, sort_keys=True, indent=2, ensure_ascii=False)
            duplicated_text = (
                '{\n  "schema_version": "' + blank["schema_version"] + '",\n'
                + base[1:] + "\n"
            )
            original = duplicated_text.encode("utf-8")
            packet_file.write_bytes(original)

            code = cli.main([
                "--packet", str(packet_file),
                "--manifest", str(manifest_file),
            ])
            self.assertEqual(code, 1)
            self.assertEqual(packet_file.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
