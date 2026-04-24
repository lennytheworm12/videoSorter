# Deployment And Cloud Setup

This project keeps local SQLite ingestion as the source workflow, then syncs
embedded rows into Supabase for hosted query serving.

## 1. Supabase

1. Create a Supabase project.
2. Run `supabase/schema.sql` in the Supabase SQL editor.
3. Copy the project database connection string and auth keys.
4. If you re-enable auth later, configure the relevant provider and redirect
   URLs in Supabase Auth.
5. `runtime_config` is also used for dynamic frontend discovery of your current
   ngrok-backed strong backend.

Backend env shape:

```bash
cp docs/examples/cloud.env.example .env.cloud.local
# then copy the values you need into your main .env
SUPABASE_DATABASE_URL='postgresql://...'
SUPABASE_URL='https://your-project.supabase.co'
SUPABASE_ANON_KEY='...'
VECTOR_BACKEND=supabase
EMBEDDING_BACKEND=hf_remote
REQUIRE_AUTH=false
DAILY_QUERY_LIMIT=100
MAX_ACTIVE_QUERIES=8
MAX_QUEUED_QUERIES=8
QUEUE_WAIT_TIMEOUT_SECONDS=20
BACKEND_LABEL='Render fallback backend'
BACKEND_QUALITY='fallback'
HF_TOKEN='...'
```

Check local setup status:

```bash
uv run python -m cloud.check_setup
```

## 2. Sync Local Data Into Supabase

Preview row counts:

```bash
uv run python -m cloud.migrate_supabase --dry-run
```

Write to Supabase:

```bash
uv run python -m cloud.migrate_supabase --apply
```

Use the Supabase `Session Pooler` connection string for
`SUPABASE_DATABASE_URL`, not the direct `db.<project>.supabase.co:5432` host.

## 3. Run Or Deploy The Backend

Local run:

```bash
uv run uvicorn api.main:app --reload
```

The API exposes:

```text
GET  /health
POST /api/query
```

Render deployment:

1. Create a new Web Service from this repository.
2. Use [`render.yaml`](../render.yaml) as the source of truth.
3. Set the unsynced env vars:
   - `SUPABASE_DATABASE_URL`
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
   - `HF_TOKEN`
   - `GOOGLE_API_KEY`
   - optional `GOOGLE_API_KEY_TWO`
   - `CORS_ORIGINS`
4. In public mode, keep:
   - `REQUIRE_AUTH=false`
   - `DAILY_QUERY_LIMIT=100`
   - `EMBEDDING_BACKEND=hf_remote`
   - `BACKEND_LABEL=Render fallback backend`
   - `BACKEND_QUALITY=fallback`
5. Set `CORS_ORIGINS` to the GitHub Pages site origin.
6. If you run a stronger home backend, give it its own public URL and set:
   - `BACKEND_LABEL=Home strong backend`
   - `BACKEND_QUALITY=strong`
   - `EMBEDDING_BACKEND=local`
   - `MAX_ACTIVE_QUERIES=8`

### Publishing a changing ngrok URL

If you do not own a domain and use free ngrok, publish the current tunnel URL
into Supabase:

```bash
uv run python -m cloud.ngrok_publish --watch
```

This watches `http://127.0.0.1:4040/api/tunnels` and updates the public
`runtime_config` row used by the frontend.

## 4. Frontend

Local frontend env:

```bash
cp apps/web/.env.local.example apps/web/.env.local
NEXT_PUBLIC_SUPABASE_URL='https://your-project.supabase.co'
NEXT_PUBLIC_SUPABASE_ANON_KEY='...'
NEXT_PUBLIC_FALLBACK_QUERY_API_URL='http://localhost:8000'
NEXT_PUBLIC_BASE_PATH=''
```

Run locally:

```bash
cd apps/web
npm install
npm run dev
```

GitHub Pages deployment:

1. Enable GitHub Pages for the repo using `GitHub Actions`.
2. Add repository variables:
   - `NEXT_PUBLIC_SUPABASE_URL`
   - `NEXT_PUBLIC_FALLBACK_QUERY_API_URL`
3. Add repository secret:
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
4. Point `NEXT_PUBLIC_FALLBACK_QUERY_API_URL` at the Render backend root URL.
5. Optionally set `NEXT_PUBLIC_PRIMARY_QUERY_API_URL` only if you want a static
   override; otherwise the frontend will discover the current ngrok URL from
   Supabase at runtime.

The workflow in [`.github/workflows/deploy-web.yml`](../.github/workflows/deploy-web.yml)
builds `apps/web/out` and publishes it to GitHub Pages.
