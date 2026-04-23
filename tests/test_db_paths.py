import os
import pathlib
import tempfile
import unittest
from unittest import mock

from core import db_paths


class DbPathsTests(unittest.TestCase):
    def test_migrate_legacy_knowledge_db_renames_guide_test_when_target_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            legacy = tmp_path / db_paths.LEGACY_KNOWLEDGE_DB_NAME
            legacy.write_text("legacy", encoding="utf-8")

            with mock.patch.dict(os.environ, {}, clear=True):
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmp_path)
                    target = db_paths.migrate_legacy_knowledge_db()
                    self.assertEqual(target.resolve(), (tmp_path / db_paths.DEFAULT_KNOWLEDGE_DB_NAME).resolve())
                    self.assertTrue(target.exists())
                    self.assertFalse(legacy.exists())
                    self.assertEqual(target.read_text(encoding="utf-8"), "legacy")
                finally:
                    os.chdir(old_cwd)

    def test_activate_knowledge_db_preserves_explicit_db_path(self) -> None:
        with mock.patch.dict(os.environ, {"DB_PATH": "custom.db"}, clear=True):
            self.assertEqual(db_paths.activate_knowledge_db(), pathlib.Path("custom.db"))
            self.assertEqual(os.environ["DB_PATH"], "custom.db")


if __name__ == "__main__":
    unittest.main()
