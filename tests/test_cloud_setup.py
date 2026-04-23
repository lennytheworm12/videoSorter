import io
import os
import pathlib
import unittest
from contextlib import redirect_stdout
from unittest import mock

from cloud import check_setup
from core import env as env_mod


class CloudSetupTests(unittest.TestCase):
    def test_check_setup_reports_missing_values(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            output = io.StringIO()
            with redirect_stdout(output):
                code = check_setup.check_setup()

        self.assertEqual(code, 1)
        rendered = output.getvalue()
        self.assertIn("SUPABASE_DATABASE_URL", rendered)
        self.assertIn("apps/web/.env.local", rendered)

    def test_load_project_env_uses_repo_root(self) -> None:
        root = env_mod.project_root()
        self.assertTrue((root / "pyproject.toml").exists())
        self.assertTrue(isinstance(root, pathlib.Path))


if __name__ == "__main__":
    unittest.main()
