# Cloud Query MVP

This keeps local SQLite ingestion as the source workflow, then syncs embedded
rows into Supabase for hosted query serving.

## 1. Supabase

1. Create a Supabase project.
2. Run `supabase/schema.sql` in the Supabase SQL editor.
3. Copy the project database connection string and auth keys.

Required backend env:

```bash
SUPABASE_DATABASE_URL='postgresql://...'
SUPABASE_URL='https://your-project.supabase.co'
SUPABASE_ANON_KEY='...'
VECTOR_BACKEND=supabase
REQUIRE_AUTH=true
```

## 2. Sync Local Data

Preview row counts:

```bash
uv run python -m cloud.migrate_supabase --dry-run
```

Write to Supabase:

```bash
uv run python -m cloud.migrate_supabase --apply
```

## 3. Run The Query API

```bash
uv run uvicorn api.main:app --reload
```

The API exposes:

```text
GET  /health
POST /api/query
```

## 4. Run The Frontend

Required frontend env in `apps/web/.env.local`:

```bash
NEXT_PUBLIC_SUPABASE_URL='https://your-project.supabase.co'
NEXT_PUBLIC_SUPABASE_ANON_KEY='...'
NEXT_PUBLIC_QUERY_API_URL='http://localhost:8000'
```

Run:

```bash
cd apps/web
npm install
npm run dev
```

The default detailed AoE2 path uses one LLM call. Enable the UI checkbox or API
field `split_detail=true` only when testing the two-call splice path.
