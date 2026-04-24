# Cloud Query MVP

This keeps local SQLite ingestion as the source workflow, then syncs embedded
rows into Supabase for hosted query serving.

## 1. Supabase

1. Create a Supabase project.
2. Run `supabase/schema.sql` in the Supabase SQL editor.
3. Copy the project database connection string and auth keys.

Required backend env:

```bash
cp .env.cloud.example .env.cloud.local
# then copy the values you need into your main .env
SUPABASE_DATABASE_URL='postgresql://...'
SUPABASE_URL='https://your-project.supabase.co'
SUPABASE_ANON_KEY='...'
VECTOR_BACKEND=supabase
REQUIRE_AUTH=true
```

Check what is still missing:

```bash
uv run python -m cloud.check_setup
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

## 3b. Deploy The Backend On Render

The repo includes [`render.yaml`](/home/bphan944/PersonalProjects/videoSorter/render.yaml:1).

In Render:

1. Create a new Blueprint or Web Service from this GitHub repo.
2. Set the unsynced env vars:
   - `SUPABASE_DATABASE_URL`
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
   - `GOOGLE_API_KEY`
   - `GOOGLE_API_KEY_TWO` (optional fallback)
   - `CORS_ORIGINS`
3. Set `CORS_ORIGINS` to your GitHub Pages site URL.

The backend will fail fast on startup if hosted env vars are missing.

## 4. Run The Frontend

Required frontend env in `apps/web/.env.local`:

```bash
cp apps/web/.env.local.example apps/web/.env.local
NEXT_PUBLIC_SUPABASE_URL='https://your-project.supabase.co'
NEXT_PUBLIC_SUPABASE_ANON_KEY='...'
NEXT_PUBLIC_QUERY_API_URL='http://localhost:8000'
NEXT_PUBLIC_BASE_PATH=''
```

Run:

```bash
cd apps/web
npm install
npm run dev
```

## 4b. Deploy The Frontend On GitHub Pages

The repo includes [`.github/workflows/deploy-web.yml`](/home/bphan944/PersonalProjects/videoSorter/.github/workflows/deploy-web.yml:1).

In GitHub:

1. Enable GitHub Pages for this repo.
2. Add repository variables:
   - `NEXT_PUBLIC_SUPABASE_URL`
   - `NEXT_PUBLIC_QUERY_API_URL`
3. Add repository secret:
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`

The workflow builds the static export with a repo-name base path and deploys `apps/web/out`.

The default detailed AoE2 path uses one LLM call. Enable the UI checkbox or API
field `split_detail=true` only when testing the two-call splice path.
