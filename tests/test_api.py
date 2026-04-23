import unittest

from api.main import _split_answer_sources


class ApiTests(unittest.TestCase):
    def test_split_answer_sources_returns_body_and_source_lines(self) -> None:
        body, sources = _split_answer_sources(
            "Answer text\n\n---\nSources by section:\n  Core:\n    [0.1] row"
        )

        self.assertEqual(body, "Answer text")
        self.assertEqual(sources[0], "Sources by section:")
        self.assertIn("[0.1] row", sources[-1])

    def test_split_answer_sources_handles_answer_without_sources(self) -> None:
        body, sources = _split_answer_sources("Answer only")

        self.assertEqual(body, "Answer only")
        self.assertEqual(sources, [])


if __name__ == "__main__":
    unittest.main()
