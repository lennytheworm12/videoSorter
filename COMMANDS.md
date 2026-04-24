# Commands

All commands use `uv run` from the project root.

---

## Scraping

```bash
# One-time: save Discord login session
uv run python scrape/login_and_save_state.py

# Scrape Discord threads for a role and insert videos into DB
uv run python -m scrape.main --role support

# Scrape a portal-core AoE2 wiki reference set into knowledge.db
uv run python -m scrape.aoe2_wiki_scrape
uv run python -m scrape.aoe2_wiki_scrape --limit 25

# Import AoE2 YouTube videos/playlists from a plain txt file of URLs
# Plain lines are auto-classified per expanded video.
# Optional overrides:
#   video|https://www.youtube.com/watch?v=...
#   coaching|https://www.youtube.com/watch?v=...
uv run python -m scrape.aoe2_video_import
uv run python -m scrape.aoe2_video_import --input data/aoe2_video_urls.txt
uv run python -m scrape.aoe2_video_import --review

# Structured AoE2 import fallback (CSV/JSON)
uv run python -m scrape.aoe2_import --input data/aoe2_sources.json --source aoe2_wiki

# Import a written AoE2 PDF guide into knowledge.db
uv run python -m scrape.aoe2_pdf_import "/mnt/c/Users/bphan/Downloads/hera-strategy-guide-2025-12 (1).pdf" --title "Hera Strategy Guide 2025"

# Check what was scraped (debug)
uv run python -m scrape.debug_scrape

# Scrape top-rated MOBAFire written guides per champion into knowledge.db
uv run python -m scrape.mobafire_scrape --limit 3
uv run python -m scrape.mobafire_scrape --champion Aatrox
uv run python -m scrape.mobafire_scrape --reanalyze --limit 3
uv run python -m scrape.mobafire_scrape --reanalyze --champion Aatrox --limit 3
uv run python -m scrape.mobafire_scrape --status
```

---

## Processing pipeline (in order)

```bash
# 1. Transcribe + analyze all unprocessed videos
uv run python process_all.py

# Filter by one or more roles
uv run python process_all.py --role support
uv run python process_all.py --role support top

# Re-analyze already-analyzed videos (keeps transcripts, clears insights)
uv run python process_all.py --role support --reanalyze

# Re-process a single video by ID
uv run python process_all.py --video VIDEO_ID

# Print counts by role and status, then exit
uv run python process_all.py --status

# 2. Build champion archetype map (run once, re-run after new champions added)
uv run python -m pipeline.champion_archetypes
uv run python -m pipeline.champion_archetypes --status   # show table
uv run python -m pipeline.champion_archetypes --fill-gaps # only Gemini fill-in for champions in videos DB
uv run python -m pipeline.champion_archetypes --fill-all  # seed ALL Data Dragon champions (empty buckets)

# 3. Embed all analyzed insights into vectors
uv run python -m pipeline.embed

# 3. Deduplicate similar insights within each video
uv run python -m pipeline.consolidate

# 4. Score insight clusters and compute final confidence weights
uv run python -m pipeline.score_clusters

# 5. Build champion cross-reference (archetype-based layer-2 retrieval)
uv run python -m pipeline.champion_crossref              # full run (vectors + generalize)
uv run python -m pipeline.champion_crossref --status     # show coverage
uv run python -m pipeline.champion_crossref --vectors    # only build champion vectors
uv run python -m pipeline.champion_crossref --generalize # only label generalizable insights
uv run python -m pipeline.champion_crossref --dry-run    # skip LLM calls (test run)

# 5b. Build AoE2 civilization cross-reference (similar civs + situation transfer)
uv run python -m pipeline.aoe2_crossref
uv run python -m pipeline.aoe2_crossref --status
uv run python -m pipeline.aoe2_crossref --vectors
uv run python -m pipeline.aoe2_crossref --label
```

---

## Maintenance

```bash
# Retry videos that failed transcription (may have been proxy/rate-limit errors)
uv run python retry_no_transcript.py

# Preview how many would be retried without actually running
uv run python retry_no_transcript.py --dry-run

# Retry a specific role only
uv run python retry_no_transcript.py --role top

# Analyze guide sources already stored in knowledge.db
uv run python -m pipeline.guide_analyze
uv run python -m pipeline.guide_analyze --source mobafire_guide
uv run python -m pipeline.guide_analyze --champion Aatrox --source mobafire_guide

# Transcribe / analyze AoE2 guide rows in knowledge.db
uv run python -m pipeline.guide_transcribe
uv run python -m pipeline.guide_analyze --game aoe2
uv run python -m pipeline.guide_analyze --game aoe2 --source aoe2_coaching
uv run python -m pipeline.guide_analyze --game aoe2 --source aoe2_wiki
uv run python -m pipeline.guide_analyze --game aoe2 --source aoe2_pdf
```

---

## Querying

```bash
# Ask a question (normalizes + retrieves + generates answer)
uv run python -m retrieval.questions "your question here"

# Filter by role
uv run python -m retrieval.questions "how do i play safe in lane" --role support

# Skip normalization, query directly
uv run python -m retrieval.questions "your question" --raw

# Show normalization result without querying
uv run python -m retrieval.questions "your question" --normalize-only

# Return more results (default: 12)
uv run python -m retrieval.questions "your question" --top-k 20

# Ask AoE2-specific questions
uv run python -m retrieval.questions --game aoe2 "how should I open with Franks?"
uv run python -m retrieval.questions --game aoe2 "how do I defend early pressure with better micro?"

# Optional: test the split detailed AoE2 answer path
uv run python -m retrieval.questions --game aoe2 --split-detail "how should I play Khmer in detail"
```

---

## Cloud MVP

```bash
# Check local cloud/frontend setup status
uv run python -m cloud.check_setup

# Preview how many rows will be synced to Supabase
uv run python -m cloud.migrate_supabase --dry-run

# After running supabase/schema.sql and setting SUPABASE_DATABASE_URL
uv run python -m cloud.migrate_supabase --apply

# Start the backend query API
uv run uvicorn api.main:app --reload

# Start the frontend
cd apps/web
npm install
npm run dev
```

Hosted deployment files:

```text
render.yaml
.github/workflows/deploy-web.yml
```

Hosted setup notes:

```text
- In Supabase Auth, enable Google and add redirect URLs for local dev and GitHub Pages.
- In Render, use the Supabase Session Pooler URI for SUPABASE_DATABASE_URL.
- In GitHub repo variables, set NEXT_PUBLIC_QUERY_API_URL to the Render backend root URL.
```

---

## Environment variables (.env)

| Variable              | Description                                              |
|-----------------------|----------------------------------------------------------|
| `GOOGLE_API_KEY`      | Primary Gemini key (free tier or paid)                   |
| `GOOGLE_CLOUD_API_KEY`| Fallback Gemini key — auto-used when primary hits quota  |
| `LLM_MODEL`           | Override model (default: `gemini-3.1-flash-lite-preview`)|
| `PROXY_LIST`          | Comma-separated proxies (`host:port:user:pass,...`)       |

Proxy list can also be stored in `proxies.txt` (one per line) in the project root — takes priority over `PROXY_LIST`.
