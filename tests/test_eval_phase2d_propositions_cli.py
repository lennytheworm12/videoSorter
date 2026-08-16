from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE = "scripts.eval_phase2d_propositions"


class Phase2DPropositionCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = importlib.import_module(MODULE)

    def test_rejects_blank_model_and_duplicate_modes(self) -> None:
        with self.assertRaises(SystemExit):
            self.module.main(["--db", "unused.db", "--live", "--model", "   "])
        with self.assertRaises(SystemExit):
            self.module.main(["--db", "unused.db", "--live", "--model", "custom", "--mode", "insight", "--mode", "insight"])

    def test_requires_deepseek_without_an_explicit_model(self) -> None:
        with patch.object(self.module, "BACKEND", "gemini"):
            with self.assertRaises(SystemExit):
                self.module.main(["--db", "unused.db", "--live"])

    def test_labels_explicit_model_as_custom_and_writes_requested_artifact(self) -> None:
        output = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        output.close()
        try:
            with (
                patch.object(self.module, "load_development_cases", return_value=()),
                patch.object(self.module, "evaluate_source_modes", return_value={"metrics": {}}),
            ):
                self.module.main(["--db", "unused.db", "--live", "--model", "custom-model", "--mode", "insight", "--json-output", output.name])
            payload = json.loads(Path(output.name).read_text())
            self.assertEqual(payload["model"]["variant"], "custom")
            self.assertEqual(payload["model"]["model"], "custom-model")
        finally:
            Path(output.name).unlink()


if __name__ == "__main__":
    unittest.main()
