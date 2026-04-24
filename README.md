# videoSorter

`videoSorter` is a multi-game knowledge pipeline and query system for two use
cases:

- `League of Legends`: ingest Discord-shared coaching videos and written guides,
  extract matchup and gameplay insights, and answer champion-specific questions.
- `Age of Empires II`: ingest YouTube coaching/guides, wiki references, and PDF
  material, then answer civilization and strategy questions.

The project now supports both local CLI usage and a hosted stack built around:

- `GitHub Pages` for the frontend
- `Render` for the FastAPI backend
- `Supabase` for hosted Postgres + `pgvector`

## What The Repo Contains

- Scrapers for Discord, YouTube, MOBAFire, AoE2 wiki pages, and PDFs
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

1. Scrape Discord / guide sources into the local databases.
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
- [Supabase schema](supabase/schema.sql)

## Hosted Stack Notes

- The GitHub Pages frontend is static and queries the Render-hosted FastAPI API.
- The frontend can be configured with both a strong primary backend and a weaker fallback backend.
- The hosted backend can run in public mode with a daily query cap.
- Use the Supabase `Session Pooler` URI for hosted database access.

## Development Notes

- `HOMEWORK_README.md` and `homework_data_cleaning.ipynb` are unrelated local
  files and are intentionally not part of the project layout.
- `apps/web/tsconfig.tsbuildinfo` is a generated artifact and should stay
  untracked.
