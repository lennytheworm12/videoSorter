"""Check local cloud MVP setup status without mutating anything.

Examples:
    uv run python -m cloud.check_setup
"""

from __future__ import annotations

import pathlib
import os

from core.env import load_project_env, project_root

load_project_env()


BACKEND_VARS = (
    "SUPABASE_DATABASE_URL",
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "VECTOR_BACKEND",
    "REQUIRE_AUTH",
)

FRONTEND_VARS = (
    "NEXT_PUBLIC_SUPABASE_URL",
    "NEXT_PUBLIC_SUPABASE_ANON_KEY",
    "NEXT_PUBLIC_QUERY_API_URL",
)


def _read_env_file(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _status_line(name: str, value: str | None) -> str:
    return f"  {'ok' if value else 'missing':7} {name}"


def check_setup() -> int:
    root = project_root()
    frontend_env = root / "apps" / "web" / ".env.local"
    schema = root / "supabase" / "schema.sql"
    frontend_env_values = _read_env_file(frontend_env)

    print("Cloud MVP setup status\n")
    print(f"Root .env:              {'found' if (root / '.env').exists() else 'missing'}")
    print(f"Frontend .env.local:    {'found' if frontend_env.exists() else 'missing'}")
    print(f"Supabase schema file:   {'found' if schema.exists() else 'missing'}")

    print("\nBackend env vars:")
    backend_missing = 0
    for name in BACKEND_VARS:
        value = os.environ.get(name)
        print(_status_line(name, value))
        backend_missing += int(not value)

    print("\nFrontend env vars:")
    frontend_missing = 0
    for name in FRONTEND_VARS:
        value = os.environ.get(name) or frontend_env_values.get(name)
        if name == "NEXT_PUBLIC_SUPABASE_ANON_KEY":
            value = value or frontend_env_values.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")
        print(_status_line(name, value))
        frontend_missing += int(not value)

    example_env = root / "docs" / "examples" / "cloud.env.example"

    print("\nNext steps:")
    if not schema.exists():
        print("  - Ensure supabase/schema.sql exists in the repo.")
    print("  - Run supabase/schema.sql in the Supabase SQL editor.")
    if backend_missing:
        print(f"  - Fill the backend Supabase vars in .env using {example_env.relative_to(root)}.")
    if frontend_missing:
        print("  - Fill apps/web/.env.local from apps/web/.env.local.example.")
    if not backend_missing:
        print("  - Preview sync with: uv run python -m cloud.migrate_supabase --dry-run")
        print("  - Apply sync with:   uv run python -m cloud.migrate_supabase --apply")
        print("  - Start API with:    uv run uvicorn api.main:app --reload")
    if not frontend_missing:
        print("  - Start web with:    cd apps/web && npm run dev")

    return 0 if backend_missing == 0 and frontend_missing == 0 and schema.exists() else 1


def main() -> None:
    raise SystemExit(check_setup())


if __name__ == "__main__":
    main()
