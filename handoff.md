# Handoff: Phase 2E Span-First Stage A Validated; Gate Failed at Clause and Role Selection

## 1. Project Context & Goal
- **Strategic Objective:** Build a source-grounded causal-relation compiler for videoSorter so cheap answer models can use compiled strategic knowledge alongside existing RAG. Phase 2E restructured Phase 2D Stage A into a span-first, 7-call decomposition so a weak model performs small, observable, source-grounded decisions while deterministic code preserves causal structure.
- **Tech Stack:** Python 3.12, `uv`, SQLite (`videos.db`), optional Supabase/Postgres, DeepSeek OpenAI-compatible API, `unittest`.
- **Hard Constraints:** Do not start Phase 3/fingerprint automation; do not weaken source grounding, condition validation, provenance, or trusted-relation validation; do not tune against the frozen `data/relation_extraction_phase2b_v0.json` held-out fixture; do not persist or promote candidate relations; do not use V4 Pro to bypass a broken Flash architecture; use focused commits pushed to `main` once Git is writable; obtain an independent sub-agent review for each meaningful boundary.

## 2. Current State
- **Worktree:** `/tmp/videoSorter-main` is a shared linked worktree with read-only Git metadata, so it displays the Phase 2E implementation diffs because the commits live in the separate writable publication clone, not in this worktree:
  - `pipeline/proposition_extract.py`: span-first 7-call Stage A (`extract_span_first_propositions`, `SPAN_FIRST_PROMPT_VERSION=phase2e-span-first-v1`): evidence localization, actor, event, effect, condition, causal direction, ontology normalization; deterministic span derivation; per-stage `StageArtifact` provenance; `ProviderCallError`/parse failure taxonomy; safe `NONE`; no proposition when required evidence is absent.
  - `pipeline/phase2d_evaluation.py`: slot-level evaluator (actor/event/effect/condition/direction/normalization/evidence/semantic/assembled/exact) with X/5 denominators counting every source-available eligible case (unreached stage = miss); mandatory held-out separation via trusted `DEFAULT_HELD_OUT_FIXTURE` for every fixture, including arbitrary/metadata-less files; direction derived from reviewed role labels; exact reviewed normalization scoring plus stage diagnostics.
  - `scripts/eval_phase2d_propositions.py`: read-only live CLI; records provider/config/prompt-version metadata; deterministic `content_sha256` artifacts.
  - `tests/test_proposition_extract.py`, `tests/test_phase2d_evaluation.py`, `tests/test_eval_phase2d_propositions_cli.py`: new provenance, condition, direction, reversal, offset, malformed-output, `NONE`, assembly, and refusal regressions.
  - `data/relation_extraction_phase2d_dev_v0.json`: five conservative, reviewed, evaluation-only normalization labels with explicit nulls and rationales; no held-out labels were inspected or changed.
  - `handoff.md` and `docs/compiled-reasoning-implementation-plan.md`: this takeover record and the durable Phase 2E plan section (code boundary, deterministic validation, valid Flash artifact, gate decision, next architecture move).
- **System Health:** Broader current evidence: `.venv/bin/python -m pytest tests --ignore=tests/test_auth.py -q` passes **404 tests with 115 subtests** (`404 passed, 115 subtests passed in 25.44s`). Focused Phase 2 evidence retained: `unittest` across the nine Phase 2 modules (`test_phase2d_evaluation`, `test_proposition_extract`, `test_eval_phase2d_propositions_cli`, `test_phase2d_metrics`, `test_candidate_ledger`, `test_constrained_mapper`, `test_candidate_generation`, `test_relation_extract`, `test_source_windows`) passes: **222 tests** (`Ran 222 tests ... OK`).
- **Live run status:** The clean Flash transcript run is valid and **failed the Phase 2E gate** at 0/5 semantic and 0/5 exact recall (see section 5).
- **Git:** The three implementation commits are published on GitHub `main`: core extraction + test (`5bcffcf`, `Add span-first semantic proposition extraction`), evaluator/CLI/tests + handoff (`da1ad9f`, `Add Phase 2E semantic evaluation and handoff`), and reviewed normalization labels/scoring/tests/docs (`b63d5e1`, `Add reviewed Phase 2E normalization scoring`).

## 3. Decisions Made (Do Not Re-Litigate)
- **Span-first 7-call Stage A:** the model only localizes evidence, fills narrow slots, classifies direction, and normalizes; deterministic code derives unique token-bounded spans and assembles propositions. Model-supplied offsets remain rejected.
- **Strict provenance/failure taxonomy/artifacts:** every stage keeps raw output, parsed output, and failure class; ungrounded or fabricated frames score no evidence/slot/semantic hit; artifacts carry deterministic `content_sha256` plus full model/provider/config metadata.
- **Slot-level evaluator with mandatory held-out separation:** official recall denominators count every source-available eligible case; an unreached stage is a visible miss, never a denominator exclusion. Held-out separation is enforced for all development fixtures, and an unavailable frozen fixture is an error.
- **Scope of the Phase 2E claim:** the code boundary and deterministic validation are complete, and a valid live model-quality measurement now exists. It proves the span-first-v1 architecture does not meet the gate; Phase 2 remains in Stage A. Normalization recall uses the same X/5 denominator through reviewed development-only labels.
- **No downstream progression:** the 0/5 semantic result requires lower-level decomposition. No Pro, frozen held-out quality run, candidate mapping, ledger promotion, or Phase 3 work was performed.

## 4. Independent Review Closure
- **Selected-evidence-local semantic uniqueness:** semantic-slot phrase uniqueness is checked only within the selected evidence span, not packet-wide; exact offsets are preserved, and multiple selected occurrences fail the case instead of silently choosing one.
- **Defensive final credit:** assembled/exact credit is granted only after grounded, slot-consistent, direction-consistent, deterministic frame assembly; per-slot diagnostics remain preserved for every case.
- **Single-proposition eligibility:** eligible development cases enforce exactly one expected proposition, keeping the official gate at X/5.
- **Fail-closed held-out validation:** the frozen held-out fixture schema is validated fail-closed before any overlap checks are performed.

## 5. Valid Flash Run and Gate Decision
- Command:
  ```bash
  LLM_PROVIDER=deepseek .venv/bin/python -m scripts.eval_phase2d_propositions \
    --live --variant flash --db videos.db --mode transcript \
    --json-output /tmp/phase2e-dev-flash-transcript-span-first-network-final.json
  ```
- Artifact inner `content_sha256`: `e3a769b61dc4699d6e65bdc5572eb86e1741832abe1c84839a68e98564f55017`; file SHA-256: `6527419b5b905964e42e6c4cbebc9b6a200ce03710e4f6d2f57161a5c1035fd8`.
- Valid result: 7 cases, 5 eligible, 2 unavailable safe-zero cases, full eligible source coverage, 2 completed cases, and 3 stage failures.
- Official X/5 results: semantic 0/5, exact 0/5, evidence span 2/5, direction 1/5, and actor/event/effect/condition/normalization 0/5 each. Unsupported proposition rate: 1.0.
- First-loss evidence: Flash chose broad or adjacent clauses and then assigned grammatical `you` or the first nearby action as the causal actor. Three cases stopped at evidence or actor validation; both completed frames preserved source text but represented the wrong causal roles.
- Decision: **PHASE 2E FAIL — CONTINUE STAGE A DECOMPOSITION.** This is model-quality evidence, not an environmental failure.

## 6. Next Immediate Steps
1. Add deterministic, source-local clause candidate enumeration before semantic extraction; expose stable candidate IDs and exact offsets.
2. Ask Flash to select the smallest mechanism-bearing clause or linked clause pair by ID, then run actor/event/effect/condition extraction only inside that selected boundary.
3. Preserve the current five-case development fixture, reviewed normalization labels, X/5 metrics, failure traces, and frozen held-out separation; do not add case-specific aliases.
4. Rerun Flash only after focused deterministic tests pass. Do not run Pro, the held-out fixture, candidate mapping, ledger promotion, or Phase 3 until Stage A reaches the preregistered gate.

## 7. Context Files
- `docs/compiled-reasoning-implementation-plan.md`: Durable architecture, Phase 2E code boundary, valid Flash gate result, first-loss diagnosis, and next decomposition contract.
- `handoff.md`: This takeover record.
- `pipeline/proposition_extract.py`: Span-first 7-call Stage A contract and deterministic span validation.
- `pipeline/phase2d_evaluation.py`: Slot-level Phase 2E evaluation and mandatory held-out separation.
- `scripts/eval_phase2d_propositions.py`: Read-only live Stage A evaluation CLI with deterministic artifact hashes.
- `data/relation_extraction_phase2d_dev_v0.json`: Non-overlapping development labels; iteration only.
- `data/relation_extraction_phase2b_v0.json`: Frozen 18-positive Phase 2B held-out fixture; do not tune on it.
- `pipeline/candidate_generation.py`, `pipeline/constrained_mapper.py`, `pipeline/candidate_ledger.py`: Completed downstream Phase 2D components; do not change until Stage A passes its gate.
- `tests/test_proposition_extract.py`, `tests/test_phase2d_evaluation.py`, `tests/test_eval_phase2d_propositions_cli.py`: Required Phase 2E safety/metric regressions (222 tests total across the nine Phase 2 modules).
- `/tmp/phase2e-dev-flash-transcript-span-first-network-final.json`: valid Flash model-quality artifact; inner `content_sha256` `e3a769b61dc4699d6e65bdc5572eb86e1741832abe1c84839a68e98564f55017`, file SHA-256 `6527419b5b905964e42e6c4cbebc9b6a200ce03710e4f6d2f57161a5c1035fd8`.
- `/tmp/phase2d-dev-flash-source-modes-scored.json` (SHA-256 `515a151d15fbba3c3122695f0258ce9dc40c12a373c4149749eeaafd0f0f7f82`) and `/tmp/phase2d-dev-flash-transcript-coaching-repair.json` (SHA-256 `a11c8590b6a1dcd6f6544974a40c609c1ec0c40bdfc7eaf42c42e63ec989e740`): valid retained Phase 2D model-quality artifacts documenting the earlier 0/5 semantic and 0/5 exact baseline.
