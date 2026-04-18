# Commands

All commands use `uv run` from the project root.

---

## Scraping

```bash
# One-time: save Discord login session
uv run python scrape/login_and_save_state.py

# Scrape Discord threads for a role and insert videos into DB
uv run python -m scrape.main --role support

# Check what was scraped (debug)
uv run python -m scrape.debug_scrape
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
uv run python -m pipeline.champion_archetypes --fill-gaps # only Gemini fill-in

# 3. Embed all analyzed insights into vectors
uv run python -m pipeline.embed

# 3. Deduplicate similar insights within each video
uv run python -m pipeline.consolidate

# 4. Score insight clusters and compute final confidence weights
uv run python -m pipeline.score_clusters
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
