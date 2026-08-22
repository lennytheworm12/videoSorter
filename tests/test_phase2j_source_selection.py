import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pipeline.phase2j_source_selection import (
    PARTITION_EXPANDED_DEV,
    PARTITION_FROZEN_REPLICATION,
    SELECTION_SCHEMA_VERSION,
    SELECTION_SEED,
    TARGET_WINDOW_COUNT,
    _selection_tie_key,
    asr_punctuation_band,
    build_selection_manifest,
    legacy_source_exclusions,
    load_legacy_benchmark,
    load_legacy_manifest,
    load_selection_manifest,
    marginal_diversity_score,
    select_windows,
    validate_selection_manifest,
    verify_selection_manifest_catalogs,
    verify_selection_manifest_inputs,
)
from pipeline.semantic_ir_pool import load_semantic_window_pool
from tests._phase2j_helpers import (
    ROOT,
    rehash_manifest,
    rehash_record,
    write_legacy_benchmark,
    write_legacy_manifest,
    write_pool,
    write_standard_phase2j_inputs,
)


FORBIDDEN_MODEL_KEYS = frozenset({
    "score", "scores", "probability", "probabilities", "confidence", "rank",
    "ranks", "prediction", "predictions", "label", "labels",
    "syntax_importance", "feature_importance", "error_taxonomy",
    "model_suggestion", "logits", "proba",
})


class Phase2JSourceSelectionTests(unittest.TestCase):
    def test_preregistered_diversity_score_and_tie_break_are_pinned(self):
        record = {
            "window_id": "pool:source:w00001-example",
            "phenomena": ["pronoun", "simple_fact"],
            "metadata": {"role": "mid", "champion": "Lux"},
        }
        self.assertEqual(asr_punctuation_band(record["phenomena"]), "PUNCTUATED")
        self.assertEqual(asr_punctuation_band(["punctuation_poor"]), "PUNCTUATION_POOR")
        self.assertEqual(
            marginal_diversity_score(
                record,
                phenomenon_counts={"pronoun": 1, "simple_fact": 0},
                role_counts={"mid": 1},
                asr_band_counts={"PUNCTUATED": 2},
                selected_champions=set(),
            ),
            23,  # (2 phenomena * 8) + 4 role + 2 ASR band + 1 champion
        )
        self.assertEqual(
            marginal_diversity_score(
                record,
                phenomenon_counts={"pronoun": 2, "simple_fact": 2},
                role_counts={"mid": 2},
                asr_band_counts={"PUNCTUATED": 3},
                selected_champions={"Lux"},
            ),
            0,
        )
        expected_digest = hashlib.sha256(
            f"{SELECTION_SEED}:{record['window_id']}".encode("utf-8"),
        ).hexdigest()
        self.assertEqual(
            _selection_tie_key(record),
            (expected_digest, record["window_id"]),
        )

    def _manifest(self, root: Path):
        pool_path, manifest_path, benchmark_path, _, _ = write_standard_phase2j_inputs(root)
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
        verify_selection_manifest_inputs(
            manifest,
            pool_path=pool_path,
            legacy_manifest_path=manifest_path,
            legacy_benchmark_path=benchmark_path,
            verify_catalogs=True,
            reproduce_selection=True,
        )
        return manifest, pool_path, manifest_path, benchmark_path

    def test_deterministic_manifest_and_stable_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, pool_path, manifest_path, benchmark_path = self._manifest(root)
            second, _, _, _ = self._manifest(root)
            self.assertEqual(first, second)
            output = root / "manifest.json"
            output.write_text(json.dumps(first, sort_keys=True), encoding="utf-8")
            self.assertEqual(load_selection_manifest(output), first)
            self.assertEqual(
                {item["window_id"] for item in first["selected"]},
                {item["window_id"] for item in second["selected"]},
            )

    def test_exact_30_windows_and_30_video_source_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _, _ = self._manifest(Path(temporary))
            selected = manifest["selected"]
            self.assertEqual(len(selected), TARGET_WINDOW_COUNT)
            self.assertEqual(len({item["window_id"] for item in selected}), 30)
            self.assertEqual(len({item["upstream_source_id"] for item in selected}), 30)
            groups = {item["source_group_id"] for item in selected}
            self.assertEqual(len(groups), 30)
            self.assertTrue(all(
                item["source_group_id"] == "video:" + item["upstream_source_id"]
                for item in selected
            ))
            self.assertEqual(manifest["partition_counts"], {
                PARTITION_EXPANDED_DEV: 24,
                PARTITION_FROZEN_REPLICATION: 6,
            })

    def test_legacy_sources_excluded_even_when_pool_retains_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool_path, manifest_path, benchmark_path, legacy_manifest, legacy_benchmark = (
                write_standard_phase2j_inputs(root)
            )
            exclusions = legacy_source_exclusions(legacy_manifest, legacy_benchmark)
            self.assertEqual(
                set(exclusions),
                {"legacy-1", "legacy-2", "outside-3", "outside-4", "outside-5"},
            )
            pool = load_semantic_window_pool(pool_path)
            # The retained pool itself still contains the legacy sources.
            self.assertIn(
                "legacy-1",
                {item["upstream_source_id"] for item in pool["windows"]},
            )
            manifest = build_selection_manifest(
                pool=pool,
                pool_path=pool_path,
                legacy_manifest=legacy_manifest,
                legacy_manifest_path=manifest_path,
                legacy_benchmark=legacy_benchmark,
                legacy_benchmark_path=benchmark_path,
            )
            selected_sources = {item["upstream_source_id"] for item in manifest["selected"]}
            self.assertFalse(selected_sources & set(exclusions))
            self.assertEqual(manifest["legacy_source_exclusions"], list(exclusions))

    def test_selection_fails_when_fewer_than_30_eligible_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pool_path = root / "small-pool.json"
            write_pool(pool_path, [f"src-{index:02d}" for index in range(1, 26)])
            pool = load_semantic_window_pool(pool_path)
            with self.assertRaisesRegex(ValueError, "cannot provide 30"):
                select_windows(pool, excluded_sources=())

    def test_partition_split_is_deterministic_without_group_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _, _ = self._manifest(Path(temporary))
            dev = {item["source_group_id"] for item in manifest["selected"]
                   if item["partition"] == PARTITION_EXPANDED_DEV}
            frozen = {item["source_group_id"] for item in manifest["selected"]
                      if item["partition"] == PARTITION_FROZEN_REPLICATION}
            self.assertEqual(len(dev), 24)
            self.assertEqual(len(frozen), 6)
            self.assertFalse(dev & frozen)
            rerun, _, _, _ = self._manifest(Path(temporary))
            rerun_groups = {
                (item["source_group_id"], item["partition"])
                for item in rerun["selected"]
            }
            self.assertEqual(
                {(item["source_group_id"], item["partition"]) for item in manifest["selected"]},
                rerun_groups,
            )

    def test_frozen_release_remains_locked_and_checkpoint_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _, _ = self._manifest(Path(temporary))
            self.assertEqual(manifest["release_gate"], "LOCKED")
            self.assertEqual(manifest["checkpoint"], "PRE_ANNOTATION_CHECKPOINT")
            self.assertEqual(manifest["schema_version"], SELECTION_SCHEMA_VERSION)
            self.assertEqual(manifest["selection_policy"]["seed"], SELECTION_SEED)

    def test_duplicate_source_and_window_identity_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _, _ = self._manifest(Path(temporary))
            # Duplicate window identity across two otherwise distinct records.
            renamed = copy.deepcopy(dict(manifest))
            records = [dict(item) for item in renamed["selected"]]
            records[1]["window_id"] = records[0]["window_id"]
            records[1] = rehash_record(records[1])
            renamed["selected"] = records
            renamed = rehash_manifest(renamed)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_selection_manifest(renamed)

            # Duplicate upstream video source group across two records.
            duplicated = copy.deepcopy(dict(manifest))
            records = [dict(item) for item in duplicated["selected"]]
            records[1]["upstream_source_id"] = records[0]["upstream_source_id"]
            records[1]["source_group_id"] = records[0]["source_group_id"]
            records[1] = rehash_record(records[1])
            duplicated["selected"] = records
            duplicated = rehash_manifest(duplicated)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_selection_manifest(duplicated)

    def test_input_source_and_candidate_hash_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, pool_path, manifest_path, benchmark_path = self._manifest(root)
            tampered = copy.deepcopy(dict(manifest))
            tampered["input_hashes"]["pool_content_sha256"] = "a" * 64
            tampered = rehash_manifest(tampered)
            with self.assertRaisesRegex(ValueError, "input files do not match"):
                verify_selection_manifest_inputs(
                    tampered,
                    pool_path=pool_path,
                    legacy_manifest_path=manifest_path,
                    legacy_benchmark_path=benchmark_path,
                    verify_catalogs=False,
                )

            source_tampered = copy.deepcopy(dict(manifest))
            records = [dict(item) for item in source_tampered["selected"]]
            records[0]["source_text_sha256"] = "b" * 64
            records[0] = rehash_record(records[0])
            source_tampered["selected"] = records
            source_tampered = rehash_manifest(source_tampered)
            with self.assertRaisesRegex(ValueError, "source text hash"):
                validate_selection_manifest(source_tampered)

            catalog_tampered = copy.deepcopy(dict(manifest))
            records = [dict(item) for item in catalog_tampered["selected"]]
            records[0]["candidate_catalog_sha256"] = "c" * 64
            records[0] = rehash_record(records[0])
            catalog_tampered["selected"] = records
            catalog_tampered = rehash_manifest(catalog_tampered)
            validate_selection_manifest(catalog_tampered)
            with self.assertRaisesRegex(ValueError, "not reproducible"):
                verify_selection_manifest_catalogs(catalog_tampered)

    def test_selection_contains_no_model_score_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _, _ = self._manifest(Path(temporary))

            def scan(value, path=""):
                if isinstance(value, dict):
                    for key, item in value.items():
                        self.assertNotIn(
                            key.casefold(), FORBIDDEN_MODEL_KEYS,
                            f"forbidden key {key!r} at {path}",
                        )
                        scan(item, path + "/" + key)
                elif isinstance(value, list):
                    for index, item in enumerate(value):
                        scan(item, f"{path}[{index}]")

            scan(manifest["selection_policy"])
            for record in manifest["selected"]:
                scan(record)
            self.assertNotIn("candidates", manifest)

    def test_retained_pool_validator_continues_to_pass(self):
        pool = load_semantic_window_pool(ROOT / "data/semantic_ir_window_pool_v1.json")
        self.assertEqual(len(pool["windows"]), 300)
        self.assertEqual(
            len({item["upstream_source_id"] for item in pool["windows"]}), 300,
        )


if __name__ == "__main__":
    unittest.main()
