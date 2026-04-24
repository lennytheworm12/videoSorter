import io
import os
import pathlib
import tempfile
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
        self.assertIn("Fill the backend Supabase vars in .env.", rendered)

    def test_load_project_env_uses_repo_root(self) -> None:
        root = env_mod.project_root()
        self.assertTrue((root / "pyproject.toml").exists())
        self.assertTrue(isinstance(root, pathlib.Path))

    def test_check_setup_reads_frontend_env_file_and_publishable_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "supabase").mkdir()
            (root / "supabase" / "schema.sql").write_text("select 1;")
            (root / "apps" / "web").mkdir(parents=True)
            (root / "apps" / "web" / ".env.local").write_text(
                "\n".join(
                    [
                        "NEXT_PUBLIC_SUPABASE_URL=https://example.supabase.co",
                        "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=sb_publishable_x",
                        "NEXT_PUBLIC_QUERY_API_URL=http://localhost:8000",
                    ]
                )
            )
            env = {
                "SUPABASE_DATABASE_URL": "postgresql://x",
                "SUPABASE_URL": "https://example.supabase.co",
                "SUPABASE_ANON_KEY": "sb_publishable_x",
                "VECTOR_BACKEND": "supabase",
                "REQUIRE_AUTH": "true",
            }
            output = io.StringIO()
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
                check_setup,
                "project_root",
                return_value=root,
            ):
                with redirect_stdout(output):
                    code = check_setup.check_setup()

        self.assertEqual(code, 0)
        rendered = output.getvalue()
        self.assertIn("ok      NEXT_PUBLIC_SUPABASE_URL", rendered)
        self.assertIn("ok      NEXT_PUBLIC_SUPABASE_ANON_KEY", rendered)
        self.assertIn("ok      NEXT_PUBLIC_QUERY_API_URL", rendered)


if __name__ == "__main__":
    unittest.main()
