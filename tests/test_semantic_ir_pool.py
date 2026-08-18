import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from pipeline.semantic_ir_pool import (
    POOL_PHENOMENA,
    build_semantic_window_pool,
    detect_pool_phenomena,
    load_semantic_window_pool,
    validate_semantic_window_pool,
    verify_semantic_window_pool_inputs,
)


class SemanticIRPoolTests(unittest.TestCase):
    def _inputs(self, root: Path):
        database = root / "pool.db"
        with sqlite3.connect(database) as connection:
            connection.execute(
                """CREATE TABLE videos (
                    video_id TEXT PRIMARY KEY, video_title TEXT, role TEXT,
                    champion TEXT, game TEXT, transcription TEXT
                )"""
            )
            records = (
                (
                    "rich", "Rich", "mid", "Lux", "lol",
                    "If Lux misses Q and when Ahri uses W, you should push two waves "
                    "before dragon because mana is lower and therefore they cannot contest. "
                    "Push river instead, but maybe do not wait but move behind tower now.",
                ),
                (
                    "implicit", "Implicit", "top", "Garen", "lol",
                    "You should hold the lane and keep the tower safe since the enemy is "
                    "dangerous nearby while your team prepares the next careful play.",
                ),
                (
                    "fact", "Fact", "bot", "Jinx", "lol",
                    "Jinx has a long attack range and deals steady damage to targets in the "
                    "bottom lane during ordinary team fights around the map.",
                ),
                (
                    "asr", "ASR", "jungle", "Lee Sin", "lol",
                    "push the wave move river hold vision save flash track cooldown take space "
                    "respect range keep health use wards avoid danger and look for angles",
                ),
                # A short row contributes to the corpus-derived champion catalog but
                # is intentionally too short to become a candidate window.
                ("catalog-ahri", "Catalog", "mid", "Ahri", "lol", "short"),
            )
            connection.executemany("INSERT INTO videos VALUES (?, ?, ?, ?, ?, ?)", records)
        frozen = root / "frozen.json"
        development = root / "development.json"
        frozen.write_text(json.dumps({"cases": []}), encoding="utf-8")
        development.write_text(json.dumps({"cases": []}), encoding="utf-8")
        return database, frozen, development

    def test_detection_covers_declared_general_phenomena(self):
        rich = (
            "If Lux misses Q and when Ahri uses W, you should push two waves before dragon "
            "because mana is lower and therefore they cannot contest. Push river instead, "
            "but maybe do not wait but move behind tower now."
        )
        values = detect_pool_phenomena(rich, ("Lux", "Ahri", "Garen"))
        for expected in (
            "direct_advice", "advice_explanation", "explicit_cause", "conditional",
            "nested_condition", "temporal", "negation", "modality", "comparison",
            "pronoun", "omitted_actor", "multiple_champions", "multiple_abilities",
            "wave_reasoning", "resource_exchange", "cause_chain", "multi_sentence",
            "contrast", "contradiction", "uncertainty", "quantity", "location_or_space",
        ):
            self.assertIn(expected, values)

    def test_pool_is_deterministic_source_isolated_and_self_validating(self):
        with tempfile.TemporaryDirectory() as temporary:
            database, frozen, development = self._inputs(Path(temporary))
            kwargs = dict(
                frozen_fixture=frozen, development_fixture=development,
                target_count=4, target_words=48, stride_words=40,
                minimum_per_phenomenon=1,
            )
            first = build_semantic_window_pool(database, **kwargs)
            second = build_semantic_window_pool(database, **kwargs)
            self.assertEqual(first, second)
            validate_semantic_window_pool(first)
            verify_semantic_window_pool_inputs(
                first, db_path=database, frozen_fixture=frozen,
                development_fixture=development, reproduce_selection=True,
            )
            self.assertEqual(len(first["windows"]), 4)
            self.assertEqual(
                len({item["upstream_source_id"] for item in first["windows"]}), 4,
            )
            self.assertEqual(set(first["phenomenon_counts"]), set(POOL_PHENOMENA))
            self.assertTrue(all(first["phenomenon_counts"].values()))
            path = Path(temporary) / "pool.json"
            path.write_text(json.dumps(first), encoding="utf-8")
            self.assertEqual(load_semantic_window_pool(path), first)

    def test_excluded_source_and_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            database, frozen, development = self._inputs(Path(temporary))
            development.write_text(json.dumps({"cases": [{
                "source_video_id": "rich", "evidence": [],
            }]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "phenomenon coverage"):
                build_semantic_window_pool(
                    database, frozen_fixture=frozen, development_fixture=development,
                    target_count=3, target_words=48, minimum_per_phenomenon=1,
                )

            development.write_text(json.dumps({"cases": []}), encoding="utf-8")
            pool = build_semantic_window_pool(
                database, frozen_fixture=frozen, development_fixture=development,
                target_count=4, target_words=48, minimum_per_phenomenon=1,
            )
            tampered = copy.deepcopy(pool)
            tampered["windows"][0]["source_text"] += " fabricated"
            with self.assertRaisesRegex(ValueError, "content hash"):
                validate_semantic_window_pool(tampered)

    def test_parameters_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            database, frozen, development = self._inputs(Path(temporary))
            for bad in (True, 0, -1, 1.5):
                with self.subTest(value=bad):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        build_semantic_window_pool(
                            database, frozen_fixture=frozen,
                            development_fixture=development, target_count=bad,
                        )


if __name__ == "__main__":
    unittest.main()
