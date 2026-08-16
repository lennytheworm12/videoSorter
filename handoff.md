# Handoff: Phase 2E Clause-First v2 Implemented; Valid Flash Gate Pending

## 1. Goal and Guardrails

Phase 2E must demonstrate at least **4/5 semantic proposition recall** on the
existing five-case development slice by decomposing Stage A into grounded
evidence selection, source-close actor/event/effect/condition extraction,
explicit causal direction, separate ontology normalization, and deterministic
assembly. Exact recall and every slot recall remain independently inspectable.

Do not weaken grounding, conditions, provenance, or trusted-relation checks;
do not tune on the frozen Phase 2B held-out fixture; do not run V4 Pro to mask a
Flash architecture failure; and do not proceed to candidate mapping, ledger
promotion, persistence, fingerprints, or Phase 3 until the Stage A gate passes.

## 2. Authoritative State

- Publication clone: `/tmp/videoSorter-phase2e-publish`, branch `publish-main`.
- Shared review worktree: `/tmp/videoSorter-main`; its Phase 2E source/test
  files are synchronized with the publication clone, but its Git metadata is
  read-only and it intentionally displays the implementation as local diffs.
- Evidence database: `/home/bphan944/PersonalProjects/videoSorter/videos.db`.
- GitHub `main` currently ends at `2f47f8f`. The unpublished series contains
  these implementation commits plus this handoff update:
  - `f1f38c4` — `Add clause-first evidence candidate selection`
  - `6624221` — `Retain clause candidates in Phase 2E artifacts`
  - `fc80ade` — `Measure Phase 2E clause catalog coverage`

The visible primary workspace is on an unrelated homework branch with user
changes. Do not copy, reset, commit, or otherwise disturb that branch.

## 3. Implemented Architecture

`pipeline/proposition_extract.py` now uses prompt version
`phase2e-clause-first-v2`:

1. Deterministically enumerate exact source-local clause candidates with
   stable IDs (`insight:cNNN` / `transcript:cNNN`). Sentence/discourse
   boundaries are primary; punctuation-poor regions use overlapping 32-token
   windows with stride 16. Catalogs are capped at 20 candidates per source by
   deterministic even coverage.
2. Flash selects one or two candidate IDs from one source. It cannot generate
   evidence text or offsets.
3. Code validates IDs, derives exact local/transcript-absolute offsets, merges
   only overlap or whitespace adjacency, and preserves real gaps.
4. Actor, event, effect, and condition are selected as exact source spans;
   condition supports safe null/`NONE`.
5. Causal direction is explicitly classified.
6. Ontology normalization runs only after the source-semantic frame exists.
7. Final proposition assembly is deterministic and fails closed.

Every success, abstention, malformed response, and provider failure retains
the candidate catalog plus per-stage raw/parsed output, selected spans,
recovered slots, direction, normalization, frames, and assembled propositions.

`pipeline/phase2d_evaluation.py` additionally measures deterministic candidate
catalog coverage before model selection. A reviewed proposition is covered
only when one or two unique, grounded candidates from one source can contain
all exact reviewed source fields after the same production coalescing logic.
Duplicate IDs and fabricated offsets fail closed. This diagnostic is separate
from semantic model credit.

## 4. Verification

- Focused nine-module Phase 2 suite: **237 tests passed**.
- Broad suite: **419 tests and 128 subtests passed**
  (`419 passed, 128 subtests passed in 20.10s`).
- The actual five eligible transcript cases have deterministic candidate
  catalog coverage **5/5**:
  - wave reset: `transcript:c009`
  - push/poke: `transcript:c005`
  - sweeper: `transcript:c004` + `transcript:c006`
  - mid push: `transcript:c006` + `transcript:c007`
  - hook risk: `transcript:c003` + `transcript:c008`

This proves candidate generation preserves every reviewed mechanism under the
one/two-ID contract. It does **not** prove Flash will select or decompose those
candidates correctly.

## 5. Model-Quality Evidence

The retained v1 artifact remains the latest valid model-quality result:

- `/tmp/phase2e-dev-flash-transcript-span-first-network-final.json`
- inner `content_sha256`:
  `e3a769b61dc4699d6e65bdc5572eb86e1741832abe1c84839a68e98564f55017`
- file SHA-256:
  `6527419b5b905964e42e6c4cbebc9b6a200ce03710e4f6d2f57161a5c1035fd8`
- result: semantic **0/5**, exact **0/5**, evidence **2/5**,
  direction **1/5**, all other required slots **0/5**.

That valid v1 result justified the lower-level clause-first v2 architecture.
It is not evidence about v2 quality.

The newest v2 retry artifact is:

- `/tmp/phase2e-clause-first-v2-catalog-network-retry.json`
- inner `content_sha256`:
  `ecfce104b38fe57e3f8db767057254177a150b93a377b05ab1f12781fe61f0b2`
- file SHA-256:
  `7e1a4cbb62bf638ff3d83b2106ab7a8682837f2f2ef00bbab1afa89ff1b7b957`
- candidate catalog coverage: **5/5**, complete
- all five eligible calls failed at evidence localization with
  `ProviderCallError` before raw model output.

Therefore this retry is an inspectable infrastructure-failure artifact, **not
a valid 0/5 v2 quality result**. Restricted outbound networking is the current
remaining gate. Earlier v2 network-failure artifacts are likewise invalid for
quality claims.

## 6. Next Justified Action

1. In a network-capable run, execute Flash transcript-only on the unchanged
   five-case development fixture and retain the JSON artifact.
2. If the first valid v2 run reaches 4–5/5 semantic recall, perform a clean
   second run to prove reproducibility, then report the complete Phase 2E gate.
3. If it reaches 3/5, make only one principled repair supported by a localized
   slot bottleneck. If it reaches 0–2/5, do not keep tuning this decomposition;
   move to entity/action/consequence mention selection and pairwise causality.
4. Push `publish-main:main` once GitHub is reachable. Update Notion only after
   its connector is reauthenticated; the previous token was expired.

Do not claim PASS, TARGETED REPAIR, or ARCHITECTURE STILL FAILING for v2 until
a valid provider run exists. Do not use the network-failure zeros as semantic
recall evidence.
