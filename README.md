# videoSorter

`videoSorter` is a multi-game knowledge pipeline and query system for two use
cases:

- `League of Legends`: ingest private/local coaching video sources and written guides,
  extract matchup and gameplay insights, and answer champion-specific questions.
- `Age of Empires II`: ingest YouTube coaching/guides, wiki references, and PDF
  material, then answer civilization and strategy questions.

The project now supports both local CLI usage and a hosted stack built around:

- `GitHub Pages` for the frontend
- `Render` for the FastAPI backend
- `Supabase` for hosted Postgres + `pgvector`

## What The Repo Contains

- Scrapers for local/private coaching sources, YouTube, MOBAFire, AoE2 wiki pages, and PDFs
- Analysis pipelines that chunk content, extract structured insights, embed them,
  score them, and build cross-reference layers
- Retrieval + answer generation for local CLI use and the hosted API
- A static Next.js frontend in `apps/web`

## Quick Start

### 1. Local Python setup

```bash
uv sync
```

Create a local `.env` with at least your Gemini key:

```bash
GOOGLE_API_KEY=...
GOOGLE_API_KEY_TWO=...
```

### 2. Common local commands

Ask a question from the CLI:

```bash
uv run python -m retrieval.questions "How should I play Aatrox into Darius?"
uv run python -m retrieval.questions --game aoe2 "How should I play Khmer in detail?"
```

Run the main LoL processing pipeline:

```bash
uv run python -m scripts.process_all
```

Run hosted-stack checks locally:

```bash
uv run python -m cloud.check_setup
uv run uvicorn api.main:app --reload
cd apps/web && npm run dev
```

### 3. Start the personal home backend with ngrok

Use this when you want the GitHub Pages frontend to discover and use your
strong backend running on your own machine. Run each block in a separate
terminal from the repo root.

Terminal 1: start the FastAPI backend on port `8000`.

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2: expose that backend through ngrok.

```bash
ngrok http 8000
```

Terminal 3: publish the current ngrok HTTPS URL into Supabase
`runtime_config`, so the frontend can discover it.

```bash
uv run python -m cloud.ngrok_publish --watch
```

Check that the backend is live:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:4040/api/tunnels
uv run python -m cloud.ngrok_publish --print-only
```

The `.env` file needs the usual backend values, including
`GOOGLE_API_KEY`, `SUPABASE_DATABASE_URL`, `SUPABASE_URL`,
`SUPABASE_ANON_KEY`, `VECTOR_BACKEND=supabase`, `REQUIRE_AUTH=false`,
`BACKEND_LABEL=Home strong backend`, and `BACKEND_QUALITY=strong`.
With free ngrok, the public URL changes each time; keep the publish command
running while the backend is online.

If `uv run python -m cloud.ngrok_publish --watch` fails with a Supabase
`tenant/user ... not found` error, ngrok is not the problem. Replace
`SUPABASE_DATABASE_URL` in `.env` with a fresh connection string from the
current Supabase project:

1. Open Supabase Dashboard -> Project Settings -> Database.
2. Copy the current pooler connection string, preferably the Session Pooler
   URI for local development.
3. Replace the password placeholder and save it as `SUPABASE_DATABASE_URL`.
4. Re-run:

```bash
uv run python -m cloud.check_setup
uv run python -m cloud.ngrok_publish
uv run python -m cloud.ngrok_publish --watch
```

You can still use the home backend manually while Supabase publishing is
broken. Run `uv run python -m cloud.ngrok_publish --print-only` or read the
Forwarding URL from the `ngrok http 8000` terminal, then use that
`https://...ngrok...` URL as the query API base URL. A hosted GitHub Pages
build cannot discover a new free-ngrok URL by itself unless the URL is either
published to Supabase or baked into `NEXT_PUBLIC_PRIMARY_QUERY_API_URL` at
build time.

## Repo Layout

```text
api/         FastAPI query backend
apps/web/    Next.js frontend
cloud/       Supabase sync + hosted vector store helpers
core/        shared database, env, registry, and champion utilities
data/        tracked input lists and other small static inputs
docs/        command reference and deployment/setup guides
pipeline/    transcription, analysis, embedding, scoring, crossref jobs
prompts/     prompt modules for LoL, AoE2, and shared prompt logic
retrieval/   question normalization, retrieval, and answer generation
scrape/      ingestion scripts for Discord, YouTube, AoE2 wiki, PDFs, guides
scripts/     utility CLIs moved out of the repository root
supabase/    hosted database schema
tests/       automated test coverage
```

The root still includes thin compatibility wrappers like `process_all.py` so
older commands continue to work, but the preferred entrypoints now live under
`scripts/`.

## Main Workflows

### League of Legends

1. Ingest local/private source material into the local databases.
2. Transcribe or parse source material.
3. Analyze into typed insights.
4. Embed, deduplicate, and cluster-score those insights.
5. Build champion cross-reference data.
6. Query through the CLI, API, or web frontend.

### Age of Empires II

1. Import video URLs, wiki references, and PDFs.
2. Transcribe or parse source material.
3. Analyze into AoE2-specific insights.
4. Embed, score, and build civilization cross-reference data.
5. Query through the CLI, API, or web frontend.

## Docs

- [Command reference](docs/commands.md)
- [Deployment and cloud setup](docs/deployment.md)
- [Incremental GitHub workflow](docs/development-workflow.md)
- [Supabase schema](supabase/schema.sql)

## Hosted Stack Notes

- The GitHub Pages frontend is static and queries the Render-hosted FastAPI API.
- The frontend can be configured with both a strong primary backend and a weaker fallback backend.
- The strong backend can be discovered dynamically from Supabase when you use a rotating ngrok URL on your home machine.
- The hosted backend can run in public mode with a daily query cap.
- Use the Supabase `Session Pooler` URI for hosted database access.
- Local scrape/session artifacts such as browser state, cookies, proxies, and raw Discord exports are intentionally untracked and should stay out of Git.

## Development Notes

- Retired course notebooks, reports, and export helpers are intentionally
  excluded from the project repository.
- Develop on a focused branch, commit a validated logical slice, and push that
  branch after each checkpoint; see the incremental GitHub workflow above.
- `apps/web/tsconfig.tsbuildinfo` is a generated artifact and should stay
  untracked.
