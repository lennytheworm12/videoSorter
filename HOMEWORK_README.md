# Data Cleaning & Preparation — Assignment Writeup

## What This Assignment Is

This notebook identifies a real-world dataset, imports it using pandas, and systematically cleans it in preparation for downstream analysis. Each cleaning step is documented with the issue found, the rationale for the chosen method, and before/after row counts.

---

## Background — What Is This Data?

**League of Legends (LoL)** is a multiplayer online game where two teams of five players compete on a map. Each player controls a character called a **champion**, specializing in one of five positions called **roles** (top lane, jungle, mid lane, ADC, support). The game has a large competitive scene and a coaching ecosystem where professional coaches analyze gameplay footage and give players feedback.

**What this data will be used for:** A RAG (Retrieval-Augmented Generation) coaching assistant — a system that answers player questions like *"How do I play Aatrox into Darius?"* or *"When should I roam as a mid laner?"* by retrieving relevant coaching advice from real videos and synthesizing a response with an LLM.

---

## The Dataset

Two sources were scraped into SQLite databases:

| Source | Description | Rows |
|---|---|---|
| **Discord coaching server** | A professional team coach shared YouTube VOD links in a Discord server, each with a champion label and role. Links were extracted and transcripts fetched via YouTube's caption API. | 494 |
| **YouTube guide videos** | Educational guide videos were scraped using `yt-dlp` across all 172 playable champions, filtered to guides >5 minutes long. | 687 |
| **Extracted insights** | Each video transcript was passed through an LLM (Gemini) which extracted structured coaching tips (laning advice, champion mechanics, matchup advice, etc.) with a grounding score. | 12,000+ |

**Combined total: 1,181 video rows** — well above the 1,000 row requirement.

---

## Data Cleaning Summary

| Step | Issue | Method | Rows Affected |
|---|---|---|---|
| 1 | Discord messages with no YouTube link had empty `video_url` | Drop rows — no URL means no video, no transcript, nothing to analyze | ~108 |
| 2 | Some videos had no `champion` label | Drop rows — champion is required for retrieval; cannot be inferred from title | ~11 |
| 3 | Videos where YouTube captions were unavailable (`no_transcript`) | Exclude from analysis — fabricating transcript content is not viable | ~109 |
| 4 | Non-English video titles (French, Korean, Spanish, etc.) | Drop rows — non-English transcripts produce garbled LLM output | varies |
| 5 | Off-topic game titles (Legends of Runeterra, TFT, Wild Rift) | Drop rows — same champion names appear across Riot titles; keyword blocklist applied | varies |
| 6 | `message_timestamp` stored as TEXT instead of datetime | Type conversion via `pd.to_datetime(errors='coerce')` — preserves all rows | all Discord rows |
| 7 | Insights with near-zero `source_score` (LLM hallucinations) | Filter rows below 0.40 threshold — these insights don't correspond to anything in the transcript | ~1,100 |

---

## Files

- `homework_data_cleaning.ipynb` — the working notebook with all cleaning code and documentation
- `videos.db` — Discord coaching session videos
- `guide_test.db` — YouTube guide videos and extracted insights
