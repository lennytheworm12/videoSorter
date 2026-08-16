# Handoff: Phase 2E Span-First Stage A Code Boundary Implemented; Live Model-Quality Gate Pending

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
  - `handoff.md` and `docs/compiled-reasoning-implementation-plan.md`: this takeover record and the durable Phase 2E plan section (code boundary, deterministic validation, test evidence, invalid-artifact record, rerun command).
- **System Health:** Broader current evidence: `.venv/bin/python -m pytest tests --ignore=tests/test_auth.py -q` passes **404 tests with 115 subtests** (`404 passed, 115 subtests passed in 25.44s`). Focused Phase 2 evidence retained: `unittest` across the nine Phase 2 modules (`test_phase2d_evaluation`, `test_proposition_extract`, `test_eval_phase2d_propositions_cli`, `test_phase2d_metrics`, `test_candidate_ledger`, `test_constrained_mapper`, `test_candidate_generation`, `test_relation_extract`, `test_source_windows`) passes: **222 tests** (`Ran 222 tests ... OK`).
- **Live run status:** The attempted clean Flash transcript run is **INVALID AS A MODEL QUALITY RESULT**; do not report PASS/FAIL model quality (see section 4).
- **Git:** The writable publication clone `/tmp/videoSorter-phase2e-publish` is based on `91dc157` and carries the work as three focused commits: core extraction + test (`5bcffcf`, `Add span-first semantic proposition extraction`), evaluator/CLI/tests + handoff (`da1ad9f`, `Add Phase 2E semantic evaluation and handoff`), and reviewed normalization labels/scoring/tests/docs. GitHub publication remains pending: DNS is still blocked in this managed sandbox, so nothing has been pushed.

## 3. Decisions Made (Do Not Re-Litigate)
- **Span-first 7-call Stage A:** the model only localizes evidence, fills narrow slots, classifies direction, and normalizes; deterministic code derives unique token-bounded spans and assembles propositions. Model-supplied offsets remain rejected.
- **Strict provenance/failure taxonomy/artifacts:** every stage keeps raw output, parsed output, and failure class; ungrounded or fabricated frames score no evidence/slot/semantic hit; artifacts carry deterministic `content_sha256` plus full model/provider/config metadata.
- **Slot-level evaluator with mandatory held-out separation:** official recall denominators count every source-available eligible case; an unreached stage is a visible miss, never a denominator exclusion. Held-out separation is enforced for all development fixtures, and an unavailable frozen fixture is an error.
- **Scope of the Phase 2E claim:** the code boundary is implemented and deterministic validation is complete; the live model-quality gate is pending a network-enabled run. Normalization recall is now available on the same X/5 denominator as other slots through reviewed development-only labels.
- **No model-quality claim from the blocked run:** the artifact records an environmental provider failure, not a Phase 2E result. First valid quality measurement requires rerunning in a network-enabled environment.

## 4. Independent Review Closure
- **Selected-evidence-local semantic uniqueness:** semantic-slot phrase uniqueness is checked only within the selected evidence span, not packet-wide; exact offsets are preserved, and multiple selected occurrences fail the case instead of silently choosing one.
- **Defensive final credit:** assembled/exact credit is granted only after grounded, slot-consistent, direction-consistent, deterministic frame assembly; per-slot diagnostics remain preserved for every case.
- **Single-proposition eligibility:** eligible development cases enforce exactly one expected proposition, keeping the official gate at X/5.
- **Fail-closed held-out validation:** the frozen held-out fixture schema is validated fail-closed before any overlap checks are performed.

## 5. Current Live-Run Blocker (Environmental)
- Attempted command:
  ```bash
  LLM_PROVIDER=deepseek .venv/bin/python -m scripts.eval_phase2d_propositions \
    --live --variant flash --db videos.db --mode transcript \
    --json-output /tmp/phase2e-dev-flash-transcript-span-first.json
  ```
- Latest artifact SHA-256 `c9ee4745622fcd47587617ce440b7ecedd89c6e6215c88163779ccb6e5c8f1df`
  (inner `content_sha256`): 7 cases, 5 eligible, 2 unavailable (ambiguous
  lexical). All five eligible cases failed at the first provider call
  (`evidence_localization`, `ProviderCallError`) before any raw output, because
  this managed sandbox blocks network/DNS. The artifact schema reports a
  normalization denominator of 5, but provider failures are not a model-quality
  result. No Pro, held-out, candidate-mapping, ledger-promotion, or downstream
  Phase 2 work was run.
- Next step is to rerun the exact command in a network-enabled environment; do
  not treat the blocked artifact as evidence about the architecture.

## 6. Next Immediate Steps
1. Rerun the exact command above in a network-enabled environment and inspect per-case artifacts (reached stages, `first_failure`, raw/parsed stage outputs), not just aggregate metrics.
2. Apply the Phase 2E gate on the valid run: >=4/5 semantic proposition recall (with slot diagnostics) justifies continuing Phase 2 evaluation into candidate/mapping experiments; 3/5 justifies one targeted repair only if a localized bottleneck is identified; 0-2/5 requires further decomposition before another end-to-end run.
3. Push the three focused commits from `/tmp/videoSorter-phase2e-publish` to `main` once network access is restored: (a) core extraction + tests; (b) evaluator/CLI/tests + handoff; and (c) reviewed normalization labels/scoring/tests/docs. Keep all run artifacts with hashes.
4. Do not run Pro, the held-out fixture, candidate mapping, ledger promotion, or Phase 3 until the gate passes on a valid run.

## 7. Context Files
- `docs/compiled-reasoning-implementation-plan.md`: Durable architecture, Phase 2E code-boundary status (span-first 7-call architecture, provenance/failure taxonomy, slot-level evaluator, mandatory held-out separation, 222 focused + 404 broader tests, invalid-artifact record, exact rerun command).
- `handoff.md`: This takeover record.
- `pipeline/proposition_extract.py`: Span-first 7-call Stage A contract and deterministic span validation.
- `pipeline/phase2d_evaluation.py`: Slot-level Phase 2E evaluation and mandatory held-out separation.
- `scripts/eval_phase2d_propositions.py`: Read-only live Stage A evaluation CLI with deterministic artifact hashes.
- `data/relation_extraction_phase2d_dev_v0.json`: Non-overlapping development labels; iteration only.
- `data/relation_extraction_phase2b_v0.json`: Frozen 18-positive Phase 2B held-out fixture; do not tune on it.
- `pipeline/candidate_generation.py`, `pipeline/constrained_mapper.py`, `pipeline/candidate_ledger.py`: Completed downstream Phase 2D components; do not change until Stage A passes its gate.
- `tests/test_proposition_extract.py`, `tests/test_phase2d_evaluation.py`, `tests/test_eval_phase2d_propositions_cli.py`: Required Phase 2E safety/metric regressions (222 tests total across the nine Phase 2 modules).
- `/tmp/phase2e-dev-flash-transcript-span-first-final-retry.json`: **INVALID as a model quality result** because the provider failed before producing raw output; SHA-256 `c9ee4745622fcd47587617ce440b7ecedd89c6e6215c88163779ccb6e5c8f1df`.
- `/tmp/phase2d-dev-flash-source-modes-scored.json` (SHA-256 `515a151d15fbba3c3122695f0258ce9dc40c12a373c4149749eeaafd0f0f7f82`) and `/tmp/phase2d-dev-flash-transcript-coaching-repair.json` (SHA-256 `a11c8590b6a1dcd6f6544974a40c609c1ec0c40bdfc7eaf42c42e63ec989e740`): **valid retained Phase 2D model-quality artifacts** documenting 0/5 semantic and 0/5 exact proposition recall after repairs. Only the Phase 2E artifact above is invalid as a quality result.
