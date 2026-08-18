import unittest
from dataclasses import replace

from pipeline.semantic_source import (
    BronzeSource,
    build_context_windows,
    segment_window,
    window_from_exact_span,
)


class SemanticSourceTests(unittest.TestCase):
    def test_source_id_requires_namespace(self):
        with self.assertRaises(ValueError):
            BronzeSource("video", "valid text")

    def test_exact_window_round_trips_source_prefix_offsets_and_hash(self):
        source = BronzeSource("transcript:vid-1", "lead When Lux misses Q, walk forward. tail", speaker="coach")
        window = window_from_exact_span(source, 5, 39)
        self.assertEqual(window.text, source.text[5:39])
        self.assertEqual(window.reconstruct(), window.text)
        self.assertTrue(window.window_id.startswith("transcript:vid-1:w"))
        self.assertEqual(window.speaker, "coach")
        window.validate(source)

    def test_window_rejects_boolean_and_out_of_bounds_offsets(self):
        source = BronzeSource("transcript:v", "one two")
        for start, end in ((True, 3), (0, 100), (3, 2)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                window_from_exact_span(source, start, end)

    def test_punctuation_poor_source_gets_nonoverlapping_fallback_segments(self):
        text = " ".join(f"word{index}" for index in range(70))
        source = BronzeSource("transcript:poor", text)
        window = window_from_exact_span(source, 0, len(text))
        self.assertEqual([segment.kind for segment in window.segments], ["fallback", "fallback", "fallback"])
        self.assertTrue(all(segment.source_text == window.text[segment.start:segment.end] for segment in window.segments))

    def test_sentence_and_discourse_boundaries_are_hints_not_answers(self):
        source = BronzeSource(
            "transcript:lux",
            "When Lux misses Q, you can walk forward because she can't stop you. Once Q comes back, respect her.",
        )
        window = window_from_exact_span(source, 0, len(source.text))
        texts = [segment.source_text for segment in window.segments]
        self.assertGreaterEqual(len(texts), 3)
        self.assertIn("because she can't stop you.", texts)
        self.assertFalse(any("access" in text.lower() or "continuity" in text.lower() for text in texts))

    def test_multiple_context_windows_are_stable_and_overlap_exact_bronze(self):
        source = BronzeSource("transcript:long", " ".join(f"t{index}" for index in range(30)))
        first = build_context_windows(source, target_words=10, overlap_words=2)
        second = build_context_windows(source, target_words=10, overlap_words=2)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 1)
        for window in first:
            window.validate(source)
            self.assertEqual(source.text[window.source_start:window.source_end], window.text)

    def test_exact_span_ids_are_bound_to_span_and_index_is_validated(self):
        source = BronzeSource("transcript:ids", "one two three four")
        left = window_from_exact_span(source, 0, 7)
        right = window_from_exact_span(source, 8, len(source.text))
        self.assertNotEqual(left.window_id, right.window_id)
        self.assertTrue(set(item.segment_id for item in left.segments).isdisjoint(
            item.segment_id for item in right.segments
        ))
        for index in (0, -1, True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                window_from_exact_span(source, 0, 3, index=index)

    def test_sentence_closing_delimiters_are_not_discarded(self):
        text = 'He said “go!”)] Next move.'
        source = BronzeSource("transcript:punct", text)
        window = window_from_exact_span(source, 0, len(text))
        self.assertIn('He said “go!”)]', [segment.source_text for segment in window.segments])
        window.validate(source)

    def test_sparse_punctuation_still_bounds_long_fallback_segments(self):
        text = "Start. " + " ".join(f"word{index}" for index in range(100))
        source = BronzeSource("transcript:sparse", text)
        window = window_from_exact_span(source, 0, len(text))
        self.assertGreater(len(window.segments), 3)
        self.assertLessEqual(max(len(segment.source_text.split()) for segment in window.segments), 32)

    def test_leading_whitespace_span_has_contiguous_segment_ids(self):
        source = BronzeSource("transcript:space", "x because y.")
        window = window_from_exact_span(source, 1, len(source.text))
        self.assertEqual([item.segment_id.rsplit(":", 1)[1] for item in window.segments], ["s001"])
        window.validate(source)

    def test_timestamps_are_preserved_not_interpolated(self):
        source = BronzeSource("transcript:time", "one two three", start_ms=1000, end_ms=4000)
        window = window_from_exact_span(source, 4, 13)
        self.assertEqual((window.start_ms, window.end_ms), (1000, 4000))

    def test_malformed_timestamp_and_metadata_fail_closed(self):
        with self.assertRaises(ValueError):
            BronzeSource("transcript:v", "text", start_ms=1)
        with self.assertRaises(ValueError):
            BronzeSource("transcript:v", "text", metadata=(("x", "1"), ("x", "2")))
        for kwargs in (
            {"speaker": ""},
            {"start_ms": False, "end_ms": True},
            {"metadata": [("x", "1")]},
            {"metadata": (("", "1"),)},
            {"metadata": (("x", object()),)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BronzeSource("transcript:v", "text", **kwargs)

    def test_window_validation_rejects_missing_segments_and_context_mutation(self):
        source = BronzeSource(
            "transcript:ctx", "speaker text.", speaker="coach", start_ms=100, end_ms=200,
            metadata=(("language", "en"),),
        )
        window = window_from_exact_span(source, 0, len(source.text))
        for changed in (
            replace(window, segments=()),
            replace(window, speaker="student"),
            replace(window, start_ms=50, end_ms=60),
            replace(window, metadata=(("language", "ko"),)),
            replace(window, version="wrong"),
            replace(window, source_content_sha256="0" * 64),
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                changed.validate(source)

    def test_segment_validation_detects_text_fabrication(self):
        source = BronzeSource("transcript:v", "First. Second.")
        window = window_from_exact_span(source, 0, len(source.text))
        segment = window.segments[0]
        object.__setattr__(segment, "source_text", "fabricated")
        with self.assertRaises(ValueError):
            segment.validate(window)

    def test_segment_window_rejects_invalid_fallback_size(self):
        source = BronzeSource("transcript:v", "one two three")
        window = window_from_exact_span(source, 0, len(source.text))
        with self.assertRaises(ValueError):
            segment_window(window, fallback_words=1)

    def test_segmentation_is_bound_to_the_versioned_algorithm(self):
        source = BronzeSource("transcript:stable", "First. Second.")
        window = window_from_exact_span(source, 0, len(source.text))
        forged = replace(
            window.segments[0], kind="fallback", end=len(window.text),
            absolute_end=len(window.text), source_text=window.text,
        )
        with self.assertRaises(ValueError):
            replace(window, segments=(forged,)).validate(source)
        with self.assertRaises(ValueError):
            segment_window(window, fallback_words=16)

    def test_segment_runtime_types_fail_closed(self):
        source = BronzeSource("transcript:types", "One sentence.")
        window = window_from_exact_span(source, 0, len(source.text))
        segment = window.segments[0]
        for bad in (
            replace(segment, kind="PROPOSITION"),
            replace(segment, start=False),
            replace(segment, absolute_end=float(segment.absolute_end)),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(window, segments=(bad,)).validate(source)
        with self.assertRaises(ValueError):
            replace(window, segments=list(window.segments)).validate(source)

    def test_word_count_parameters_require_real_integers(self):
        source = BronzeSource("transcript:params", "one two three")
        for target in (True, 2.5, "3"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                build_context_windows(source, target_words=target)  # type: ignore[arg-type]
        window = window_from_exact_span(source, 0, len(source.text))
        for fallback in (True, 2.5, "32"):
            with self.subTest(fallback=fallback), self.assertRaises(ValueError):
                segment_window(window, fallback_words=fallback)  # type: ignore[arg-type]

    def test_namespaced_ids_reject_empty_or_whitespace_components(self):
        for source_id in (":", "a:", " : ", "a\n:b"):
            with self.subTest(source_id=source_id), self.assertRaises(ValueError):
                BronzeSource(source_id, "text")

    def test_window_identity_binds_contextual_bronze_provenance(self):
        first = BronzeSource("transcript:same", "same text", speaker="A", start_ms=0, end_ms=1)
        second = BronzeSource("transcript:same", "same text", speaker="B", start_ms=2, end_ms=3)
        first_window = window_from_exact_span(first, 0, len(first.text))
        second_window = window_from_exact_span(second, 0, len(second.text))
        self.assertNotEqual(first.provenance_sha256, second.provenance_sha256)
        self.assertNotEqual(first_window.window_id, second_window.window_id)
        with self.assertRaises(ValueError):
            first_window.validate(second)
        with self.assertRaises(ValueError):
            replace(first_window, speaker="other").validate()


if __name__ == "__main__":
    unittest.main()
