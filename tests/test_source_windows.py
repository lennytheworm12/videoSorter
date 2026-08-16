"""Deterministic Phase 2D tests for bronze transcript window resolution."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from pipeline.source_windows import SourceWindowResolver


class SourceWindowResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "windows.db"
        with sqlite3.connect(self.db) as conn:
            conn.executescript("""
                CREATE TABLE videos (video_id TEXT PRIMARY KEY, transcription TEXT);
                CREATE TABLE insights (id INTEGER PRIMARY KEY, video_id TEXT, text TEXT);
            """)
            conn.execute("INSERT INTO videos VALUES (?, ?)", ("v1", "Before exact source phrase after. The coach says Flay prevents staying on target after Tristana jumps. End."))
            conn.execute("INSERT INTO videos VALUES (?, ?)", ("v2", "Repeat exact source phrase. filler. Repeat exact source phrase."))
            conn.execute("INSERT INTO videos VALUES (?, ?)", ("v3", "After Lux misses Q she cannot stop you walking forward, so take space."))
            conn.execute("INSERT INTO videos VALUES (?, ?)", ("empty", ""))
            conn.executemany("INSERT INTO insights VALUES (?, ?, ?)", [
                (1, "v1", "Flay prevents staying on target after Tristana jumps."),
                (2, "v2", "Repeat exact source phrase."),
                (3, "v3", "After Lux Q misses, walk forward because she cannot stop your advance."),
                (4, "empty", "anything"),
                (5, "v3", "unrelated bananas orchestra"),
            ])

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_resolves_unique_exact_text(self) -> None:
        result = SourceWindowResolver(str(self.db)).resolve("1")
        self.assertEqual(result.alignment_method, "exact_text")
        self.assertEqual(result.alignment_score, 1.0)
        self.assertIn("Flay prevents", result.transcript_window)
        self.assertEqual(len(result.exact_source_spans), 1)

    def test_preserves_multiple_exact_locations_as_ambiguous(self) -> None:
        result = SourceWindowResolver(str(self.db)).resolve("2")
        self.assertEqual(result.alignment_method, "ambiguous_exact")
        self.assertEqual(len(result.exact_source_spans), 2)
        self.assertFalse(result.resolved)

    def test_uses_explicit_span_when_available(self) -> None:
        result = SourceWindowResolver(str(self.db)).resolve(
            "3", source_span=(0, 17), source_span_verified=True,
        )
        self.assertEqual(result.alignment_method, "explicit_span")
        self.assertEqual(result.exact_source_spans, ((0, 17),))

    def test_marks_unverified_external_span_as_non_exact(self) -> None:
        result = SourceWindowResolver(str(self.db)).resolve("3", source_span=(0, 17))
        self.assertEqual(result.alignment_method, "unverified_external_span")
        self.assertEqual(result.alignment_score, 0.0)
        self.assertFalse(result.resolved)

    def test_lexically_resolves_paraphrased_insight_in_own_video(self) -> None:
        result = SourceWindowResolver(str(self.db)).resolve("3")
        self.assertEqual(result.alignment_method, "lexical_window")
        self.assertIn("Lux misses Q", result.transcript_window)
        self.assertGreater(result.alignment_score, 0.12)

    def test_rejects_wrong_expected_source_without_searching_elsewhere(self) -> None:
        result = SourceWindowResolver(str(self.db)).resolve("1", expected_source_id="v3")
        self.assertEqual(result.alignment_method, "source_mismatch")
        self.assertFalse(result.resolved)

    def test_handles_missing_or_empty_transcript(self) -> None:
        result = SourceWindowResolver(str(self.db)).resolve("4")
        self.assertEqual(result.alignment_method, "transcript_missing")
        self.assertFalse(result.resolved)

    def test_reports_unresolved_nonmatching_summary_with_candidates(self) -> None:
        result = SourceWindowResolver(str(self.db)).resolve("5")
        self.assertEqual(result.alignment_method, "unresolved")
        self.assertFalse(result.resolved)

    def test_does_not_resolve_a_single_generic_token_overlap(self) -> None:
        with sqlite3.connect(self.db) as conn:
            conn.execute("INSERT INTO videos VALUES (?, ?)", ("weak", "The enemy appears in an unrelated transcript."))
            conn.execute("INSERT INTO insights VALUES (?, ?, ?)", (6, "weak", "enemy bananas orchestra"))
        result = SourceWindowResolver(str(self.db)).resolve("6")
        self.assertEqual(result.alignment_method, "unresolved")
        self.assertFalse(result.resolved)

    def test_quarantines_near_tied_lexical_windows(self) -> None:
        with sqlite3.connect(self.db) as conn:
            conn.execute("INSERT INTO videos VALUES (?, ?)", ("tie", "Lux misses Q take space Lux misses Q take space"))
            conn.execute("INSERT INTO insights VALUES (?, ?, ?)", (7, "tie", "After Lux misses Q, claim space"))
        result = SourceWindowResolver(str(self.db), window_words=5, window_stride=5).resolve("7")
        self.assertEqual(result.alignment_method, "ambiguous_lexical")

    def test_rejects_invalid_span_and_unknown_insight(self) -> None:
        resolver = SourceWindowResolver(str(self.db))
        self.assertEqual(resolver.resolve("1", source_span=(-1, 2)).alignment_method, "invalid_source_span")
        self.assertEqual(resolver.resolve("1", source_span=(True, 3)).alignment_method, "invalid_source_span")
        with self.assertRaisesRegex(ValueError, "unknown insight"):
            resolver.resolve("999")


if __name__ == "__main__":
    unittest.main()
