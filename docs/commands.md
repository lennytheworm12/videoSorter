# Command Reference

All commands assume the project root as the working directory.

## Scraping And Imports

```bash
# One-time local setup: save a browser login session for scraping
# This creates local-only state files; do not commit them.
uv run python scrape/login_and_save_state.py

# Scrape private/local thread sources for a role
uv run python -m scrape.main --role support

# Scrape AoE2 wiki reference pages
uv run python -m scrape.aoe2_wiki_scrape
uv run python -m scrape.aoe2_wiki_scrape --limit 25

# Import AoE2 YouTube videos/playlists from a plain text file
uv run python -m scrape.aoe2_video_import
uv run python -m scrape.aoe2_video_import --input data/aoe2_video_urls.txt
uv run python -m scrape.aoe2_video_import --review

# Structured AoE2 imports
uv run python -m scrape.aoe2_import --input data/aoe2_sources.json --source aoe2_wiki

# Import a written AoE2 PDF guide
uv run python -m scrape.aoe2_pdf_import "/mnt/c/Users/bphan/Downloads/hera-strategy-guide-2025-12 (1).pdf" --title "Hera Strategy Guide 2025"

# Debug scrape results
uv run python -m scrape.debug_scrape

# Scrape top-rated MOBAFire written guides
uv run python -m scrape.mobafire_scrape --limit 3
uv run python -m scrape.mobafire_scrape --champion Aatrox
uv run python -m scrape.mobafire_scrape --reanalyze --limit 3
uv run python -m scrape.mobafire_scrape --reanalyze --champion Aatrox --limit 3
uv run python -m scrape.mobafire_scrape --status
```

## Processing Pipelines

```bash
# LoL: transcribe + analyze all pending/transcribed videos
uv run python -m scripts.process_all
uv run python -m scripts.process_all --role support
uv run python -m scripts.process_all --role support top
uv run python -m scripts.process_all --reanalyze --role support
uv run python -m scripts.process_all --video VIDEO_ID
uv run python -m scripts.process_all --status

# Legacy compatibility wrappers still work
uv run python process_all.py --status

# Champion archetypes
uv run python -m pipeline.champion_archetypes
uv run python -m pipeline.champion_archetypes --status
uv run python -m pipeline.champion_archetypes --fill-gaps
uv run python -m pipeline.champion_archetypes --fill-all

# Embedding, dedupe, confidence
uv run python -m pipeline.embed
uv run python -m pipeline.consolidate
uv run python -m pipeline.score_clusters

# Champion / civilization cross-reference
uv run python -m pipeline.champion_crossref
uv run python -m pipeline.champion_crossref --status
uv run python -m pipeline.champion_crossref --vectors
uv run python -m pipeline.champion_crossref --generalize
uv run python -m pipeline.champion_crossref --dry-run

uv run python -m pipeline.aoe2_crossref
uv run python -m pipeline.aoe2_crossref --status
uv run python -m pipeline.aoe2_crossref --vectors
uv run python -m pipeline.aoe2_crossref --label
```

## Maintenance Utilities

```bash
# Retry videos that failed transcription
uv run python -m scripts.retry_no_transcript
uv run python -m scripts.retry_no_transcript --dry-run
uv run python -m scripts.retry_no_transcript --role top

# Analyze guide sources already stored in knowledge.db
uv run python -m pipeline.guide_analyze
uv run python -m pipeline.guide_analyze --source mobafire_guide
uv run python -m pipeline.guide_analyze --champion Aatrox --source mobafire_guide

# AoE2 guide transcription / analysis
uv run python -m pipeline.guide_transcribe
uv run python -m pipeline.guide_analyze --game aoe2
uv run python -m pipeline.guide_analyze --game aoe2 --source aoe2_coaching
uv run python -m pipeline.guide_analyze --game aoe2 --source aoe2_wiki
uv run python -m pipeline.guide_analyze --game aoe2 --source aoe2_pdf

# Confidence calibration / evaluation
uv run python -m scripts.calibrate
uv run python -m scripts.calibrate --apply
uv run python -m scripts.eval --stats
uv run python -m scripts.eval "cassiopeia into zed"

# Retrieval memory profiling
uv run python -m scripts.profile_query_memory --vector-backend supabase --embedding-backend hf_remote --question "how do i beat a ranged team as illaoi"
uv run python -m scripts.profile_query_memory --vector-backend supabase --embedding-backend bm25_only --question "how do i beat a ranged team as illaoi"
```

## Querying

```bash
# General query flow
uv run python -m retrieval.questions "your question here"
uv run python -m retrieval.questions "how do i play safe in lane" --role support
uv run python -m retrieval.questions "your question" --raw
uv run python -m retrieval.questions "your question" --normalize-only
uv run python -m retrieval.questions "your question" --top-k 20

# AoE2-specific examples
uv run python -m retrieval.questions --game aoe2 "how should I open with Franks?"
uv run python -m retrieval.questions --game aoe2 "how do I defend early pressure with better micro?"
uv run python -m retrieval.questions --game aoe2 --split-detail "how should I play Khmer in detail"
```

## Hosted Stack

```bash
# Local hosted-stack checks
uv run python -m cloud.check_setup
uv run python -m cloud.migrate_supabase --dry-run
uv run python -m cloud.migrate_supabase --apply
uv run uvicorn api.main:app --reload

# Frontend
cd apps/web
npm install
npm run dev
```

Deployment/config references:

```text
render.yaml
.github/workflows/deploy-web.yml
docs/examples/cloud.env.example
apps/web/.env.local.example
```
