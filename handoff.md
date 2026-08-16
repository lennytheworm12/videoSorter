# Handoff: Repair Phase 2D grounded-proposition extraction before any Phase 3 work

## 1. Project Context & Goal
- **Strategic Objective:** Build a source-grounded causal-relation compiler for videoSorter so cheap answer models can use compiled strategic knowledge alongside existing RAG. Phase 2D is stopped because Stage A cannot yet recover reliable causal propositions from verified bronze transcript windows.
- **Tech Stack:** Python 3.12, `uv`, SQLite (`videos.db`), optional Supabase/Postgres, DeepSeek OpenAI-compatible API, `unittest`.
- **Hard Constraints:** Do not start Phase 3/fingerprint automation; do not weaken provenance, entity grounding, condition preservation, ontology closure, or trusted-relation promotion; do not tune against `data/relation_extraction_phase2b_v0.json`; no corpus-wide reparse/backfill; use focused commits pushed directly to `main`; obtain an independent sub-agent review for each meaningful boundary.

## 2. Current State
- **Git Status:** `/tmp/videoSorter-main` is a clean detached worktree at `163ddf4` (`Record Phase 2D Stage A stop gate`), already pushed to `origin/main`.
- **System Health:** `uv run python -m unittest tests.test_phase2d_evaluation tests.test_proposition_extract tests.test_eval_phase2d_propositions_cli tests.test_phase2d_metrics tests.test_candidate_ledger tests.test_constrained_mapper tests.test_candidate_generation tests.test_relation_extract tests.test_source_windows` passes: 116 tests.
- **Completed Items:** M12 resolver; M13 source-mode proposition extraction; M14 closed-world candidate generation; M15 ID-only mapper; M16 provisional-only ledger; M17 evaluation/CLI; durable repo plan and Notion stop-gate update. Bronze audit: 385/494 videos have transcripts; 7,755/8,495 insights map to nonempty local transcript text.

## 3. Decisions Made (Do Not Re-Litigate)
- **Phase 2D data stays separate from trusted relations:** `CandidateLedger` is in-memory/provisional and never writes `StrategicRelation`; only independently supported, validated mappings can become trusted later.
- **Source alignment is strict and deterministic:** Stage A must quote exact source phrases and choose a source label; `pipeline.proposition_extract` derives unique token-bounded spans and rejects paraphrases, ambiguous matches, cross-source propositions, fabricated evidence, and model-supplied offsets.
- **Canonical mapping is closed-world:** `pipeline.candidate_generation` creates legal candidate IDs; `pipeline.constrained_mapper` may only select those IDs or return `unmapped`/`no_relation`.
- **The Phase 2B fixture is held out:** `data/relation_extraction_phase2b_v0.json` is not for prompt, alias, top-k, threshold, or candidate tuning. Development fixture loading enforces no held-out insight-ID or source-video-ID overlap.
- **Do not spend on Pro yet:** Flash Stage A must first produce enough valid propositions to make candidate/mapper comparisons interpretable. Pro was intentionally not run in Phase 2D.

## 4. Failed Attempts & Dead Ends
- **Free-form Phase 2/2B relation compilation:** Flash accepted 0/18 held-out references; Pro peaked at 3/18 before stricter safe grounding variants returned 0. Do not loosen validator thresholds to change this result.
- **Initial Phase 2D Flash calls:** Omitting DeepSeek non-thinking returned empty content. Stage A now forwards `RELATION_EXTRACTION_DEEPSEEK_THINKING=disabled` for DeepSeek only.
- **LLM-supplied offsets and ambiguous source enums:** Flash echoed `insight|transcript` and generated invalid spans. The prompt now lists valid source labels per packet, and code derives spans; do not restore model-generated offsets.
- **Current Stage A prompt repair:** Even with verified local transcript windows containing the reviewed mechanism, Flash transcript-only scored 0/5 semantic and 0/5 exact recall. It returned four safe zeros and one source-grounded but causally misstructured proposition. This is the current blocker.
- **Pooled semantic-token scoring:** It accepted reversed causal roles and lost condition polarity. Development scoring now requires reviewed subject/predicate/effect groups, condition groups, and leading condition operators; do not revert to pooled matching.

## 5. Next Immediate Steps
1. **Read the durable stop gate and inspect the five development bronze windows:** Start with `docs/compiled-reasoning-implementation-plan.md`, `data/relation_extraction_phase2d_dev_v0.json`, `pipeline/proposition_extract.py`, and `/tmp/phase2d-dev-flash-transcript-coaching-repair.json`. Form a narrow redesign hypothesis for Stage A that preserves exact source grounding while recovering actor, causal direction, effect, and condition.
2. **Implement and review only the Stage A repair:** Add deterministic tests and run the development transcript-only evaluation first. Do not run the held-out fixture, candidate mapper, ledger promotion, Pro, or Phase 3 until development Stage A has useful high-precision recall.

## 6. Context Files
- `docs/compiled-reasoning-implementation-plan.md`: Durable architecture, phase history, exact stop-gate rationale, commands, and constraints.
- `handoff.md`: This takeover record.
- `pipeline/source_windows.py`: Read-only insight-to-own-video bronze transcript resolver.
- `pipeline/proposition_extract.py`: Current Stage A contract and exact source-span validation; immediate repair target.
- `pipeline/phase2d_evaluation.py`: Source-mode, semantic-role, condition, and held-out-separation metrics.
- `scripts/eval_phase2d_propositions.py`: Read-only live Stage A evaluation CLI.
- `data/relation_extraction_phase2d_dev_v0.json`: Non-overlapping development labels; use for iteration only.
- `data/relation_extraction_phase2b_v0.json`: Frozen 18-positive Phase 2B held-out fixture; do not tune on it.
- `pipeline/candidate_generation.py`, `pipeline/constrained_mapper.py`, `pipeline/candidate_ledger.py`: Completed downstream Phase 2D components; do not change until Stage A passes its gate.
- `tests/test_proposition_extract.py`, `tests/test_phase2d_evaluation.py`: Required safety/metric regressions for the immediate Stage A boundary.
- `/tmp/phase2d-dev-flash-source-modes-scored.json`: Latest all-mode Flash artifact, SHA-256 `515a151d15fbba3c3122695f0258ce9dc40c12a373c4149749eeaafd0f0f7f82`.
- `/tmp/phase2d-dev-flash-transcript-coaching-repair.json`: Latest transcript-only Flash artifact, SHA-256 `a11c8590b6a1dcd6f6544974a40c609c1ec0c40bdfc7eaf42c42e63ec989e740`.
