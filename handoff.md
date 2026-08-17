# Handoff: Phase 2E and Phase 2F Closed at Their Architecture Gates

## 1. Architectural decision

Phase 2E clause-first v2 failed its preregistered Stage A gate. The
proposition-first and clause-first architectures are closed; do not repair their
parser, add fixture-specific aliases, tune their prompts further, or run V4 Pro
through the same proposition boundary to rescue the result.

Phase 2F then implemented a clean-room, source-preserving compiler from bronze
transcript windows to source-anchored semantic mentions and general semantic
edges. Its valid strong-model legacy gate failed at mention selection, after
complete deterministic candidate coverage. Phase 2F is also closed. Do not
begin canonical claims, League ontology normalization, strategic relations,
fingerprints, production persistence, corpus backfill, Phase 3, or Flash
optimization. Further work requires a new explicit goal to redesign Pass 1.

Architectural source of truth: Notion page **Ground-Up Semantic Compiler
Architecture — Source-Preserving Bronze → Strategic Knowledge**
(`3bef8ba7-8bf3-8148-b4fd-e9ee584d3c30`) and its parent design page
(`3bbf8ba7-8bf3-811d-9b96-cc7c4d2df5b4`).

## 2. Authoritative Phase 2E result

Valid Flash artifact:
`/tmp/phase2e-clause-first-v2-valid-run1.json`

The artifact's windows, catalogs, and reviewed expectations were regenerated
and matched exactly against commit
`1b3063edd84237c32a391564e461416ec992c308`, the configured development
fixture, and the configured local database during Phase 2F reconciliation.

- canonical inner `content_sha256`:
  `04c185aaf324251b4733e76c87b2c71ea3946497f79a8956f268e88f28e2e17b`
- file SHA-256:
  `02725fb163ef752c98f51a070652ef5418a5b0d4916363d1c61c3071e957c808`
- semantic proposition recall: **0/5**
- exact decomposition recall: **0/5**
- every required semantic slot: **0/5**
- deterministic candidate-catalog coverage: **5/5**
- unsupported or invented slots: **0**
- five raw provider outputs; **0** completed eligible cases and **5**
  `ValueError` failures at evidence localization;
- two additional fixture cases unavailable because their lexical source-window
  resolutions were ambiguous; downstream semantic stages never ran.

All five raw outputs omitted the required source prefix (`c009` rather than
`transcript:c009`). That envelope defect makes the production parser reject the
selections, but diagnostic prefix canonicalization is sufficient to locate the
semantic loss without granting model credit:

- wave reset: reviewed mechanism selected;
- push/poke: reviewed mechanism plus one irrelevant clause selected;
- sweeper: wrong clause pair;
- mid push: wrong clause pair;
- hook risk: wrong clause pair.

Official parse-valid clause selection is **0/5**. After diagnostic-only prefix
expansion, reviewed-mechanism containment is **2/5**; exact/minimal selection
would be at most **1/5** because push/poke included irrelevant `c008`.
Therefore parser robustness is not the central failure and fixing the prefix
would not repair the architecture.

## 3. Known first-loss boundary

```text
DETERMINISTIC SOURCE COVERAGE
        5/5
          ↓
DIAGNOSTIC REVIEWED-MECHANISM CONTAINMENT
        2/5 (exact/minimal <= 1/5)
          ↓
PROPOSITION EXTRACTION
        unusable
```

Record this precisely:

- candidate generation succeeded;
- clause selection failed;
- the first semantic loss boundary is known;
- Phase 2E is an architecture failure, not an unfinished experiment.

## 4. Preserved historical negative evidence

The valid artifact supersedes only the earlier statement that v2 quality was
pending. It does not replace or reclassify prior evidence:

- valid span-first v1 model-quality failure:
  `/tmp/phase2e-dev-flash-transcript-span-first-network-final.json`; inner hash
  `e3a769b61dc4699d6e65bdc5572eb86e1741832abe1c84839a68e98564f55017`;
  file hash
  `6527419b5b905964e42e6c4cbebc9b6a200ce03710e4f6d2f57161a5c1035fd8`;
  semantic/exact **0/5**, evidence **2/5**, direction **1/5**, all other
  required slots **0/5**.
- clause-first v2 provider-failure retry:
  `/tmp/phase2e-clause-first-v2-catalog-network-retry.json`; inner hash
  `ecfce104b38fe57e3f8db767057254177a150b93a377b05ab1f12781fe61f0b2`;
  file hash
  `7e1a4cbb62bf638ff3d83b2106ab7a8682837f2f2ef00bbab1afa89ff1b7b957`;
  catalog coverage **5/5**, but all eligible calls failed with
  `ProviderCallError` before raw output. This remains infrastructure evidence,
  not a quality result.

## 5. Repository state and isolation

- GitHub `main` at Phase 2F start:
  `1b3063edd84237c32a391564e461416ec992c308`.
- Authoritative clean clone at Phase 2F start:
  `/tmp/videoSorter-phase2e-publish`.
- Evidence database:
  `/home/bphan944/PersonalProjects/videoSorter/videos.db`.
- The visible primary workspace is on an unrelated homework branch with user
  changes. Do not reset, overwrite, copy Phase 2 files into, or commit those
  changes.
- Preserve all Phase 2D/2E implementation, tests, fixtures, and artifacts as
  historical evidence. Phase 2F uses a separate module boundary.
- Audit limitation: the valid v2 JSON is temporary and does not embed a commit,
  timestamp, fixture hash, or database hash. Its current linkage was verified
  by regeneration, not by the artifact hash alone. Phase 2F artifacts must
  embed revision and input-content hashes.

## 6. Phase 2F lower-boundary contract

The new target is:

```text
bronze transcript
  -> deterministic source segmentation/context
  -> source-anchored mention candidates
  -> constrained mention selection and typing
  -> bounded candidate mention pairs
  -> general semantic edge classification
  -> proof-carrying source-semantic IR graph
```

Bronze is immutable. Model-supplied arbitrary offsets are untrusted;
deterministic code resolves selected IDs to exact spans. Conditions, time,
negation, modality, ambiguity, and unresolved references are first-class.
Every node and edge must trace to exact source evidence. `UNKNOWN`,
`AMBIGUOUS`, `INSUFFICIENT_EVIDENCE`, and no-relation are valid outputs.

Pass 1 must not contain League strategic concepts such as `access`,
`continuity`, `tempo`, `initiative`, or `wave_obligation`. It must not emit a
mandatory actor/predicate/effect/condition tuple.

## 7. Evaluation discipline

- Retain the five Phase 2E cases as a legacy failure regression set, not the
  only benchmark.
- Keep `data/relation_extraction_phase2b_v0.json` frozen and unchanged.
- Build non-overlapping semantic-IR `DEV` and `FROZEN_EVAL` fixtures; fail
  closed when overlap cannot be verified.
- Measure deterministic mention-catalog and candidate-edge-pair coverage before
  model selection/classification.
- Tune only on DEV. Preregister the representation gate before one frozen run.
- Use a strong configured reference model to test representation viability;
  cheap-model optimization is a future goal.
- Provider failure and model-quality failure remain distinct in artifacts and
  denominators.

Final Phase 2F recommendation must be exactly one of:

```text
SEMANTIC IR VIABLE — READY TO DESIGN PASS 2
SEMANTIC IR VIABLE WITH SPECIFIC LIMITATIONS — REPAIR BEFORE PASS 2
SEMANTIC IR NOT VIABLE — REDESIGN PASS 1
```

## 8. Phase 2F closed at the source-semantic representation gate

The clean-room Pass 0/Pass 1 implementation now lives in separate
`pipeline/semantic_*` modules and covers typed nodes/edges, deterministic
mentions, qualifiers, coreference, pairwise relations, graph orchestration,
proof-carrying run artifacts, semantic-checksum evaluation, and isolated pool
construction. Independent adversarial reviews have passed the source, schema,
mention, qualifier, coreference, edge, compiler, artifact, evaluator, and pool
boundaries after their findings were repaired. The current non-browser suite
passes 574 tests and 249 subtests; `tests/test_auth.py` remains an unrelated
Chromium import-time environment problem.

The representative pool contains 300 exact windows from 300 distinct videos,
covers every declared phenomenon at least eight times, and has content SHA-256
`9b89c6d6c6c8070eba48d6db47254e156c1b2591c1480a60f98a1e8d789491c2`.
External reproduction against `videos.db` and the Phase 2B/2D exclusions
matched exactly.

The repaired legacy benchmark contains 33 reviewed mentions, 24 reviewed
edges, 10 grounded qualifiers, 8 unresolved-reference judgments, and 75
semantic-checksum questions. It is locked at content SHA-256
`a17674b6e2c491f0d7a1600dde0cfb8cc533d1d17db8633d8d94b2de9a57c1dd`;
its exact-source manifest is
`cf86dde955f4cbeee091f38aab8293256b0c48f809c969384185a330ee511241`.
The first strong-model configuration and strict five-case thresholds are
preregistered in the implementation plan. A clean committed attempt at
`a0feefd50013722c943976a9131eb545f364178c` reached the provider boundary, but
all 30 mention calls failed as `MentionProviderError:URLError` because the
execution environment could not resolve `api.deepseek.com`. It returned no
model bytes and is not a semantic-quality run. The complete reconstructible
negative artifact is retained at
`data/phase2f_artifacts/phase2f-legacy-pro-run1.tar.gz` (SHA-256
`e6c2122a2b91c2b70d9775f2c108c26c82cdfff2f5cea9b3c5f60dbbc4146330`).
After network access was enabled, the first valid strong-model legacy-development
run on the locked `LEGACY_FAILURE` split completed at clean revision
`b5317c6bd90572e052ab85f399e339c4de83a4e8` against the official endpoint with
`deepseek-v4-pro`. It is a genuine semantic-quality failure, not provider
evidence. Deterministic mention coverage was 33/33, but reviewed exact mention
selection and typing were 0/33; qualifier recall was 0/10, edge recall 0/24,
reference recovery 0/8, and semantic checksum 0/75. All 1,949 model calls
returned bytes and there were zero provider failures. Safe diagnostic removal
of one complete Markdown fence still recovers 0/33 reviewed mentions; only one
reviewed candidate ID appears anywhere in all retained raw mention output.

The first semantic loss is therefore:

```text
DETERMINISTIC EXACT MENTION CATALOG        33/33
        ↓
FLAT MODEL-FACING MENTION SELECTION         0/33
        ↓
DOWNSTREAM REVIEWED SOURCE SEMANTICS        0/75
```

The immutable run is retained at
`data/phase2f_artifacts/phase2f-legacy-pro-run2.tar.gz` (archive SHA-256
`b17cde9d7dc909c317aac81be08e9ed4860f91231d5568aeb6ee515a1fd67183`, aggregate
inner/file SHA-256 `b0a030765217f2dcb52634d31eec171b307541308012945f87864cf7d5697492` /
`ad3801a9fc23a23837fe0ad078273a2744fb9640bbd826172d359af4654cf547`). The old
interface exposed 3,248–3,344 overlapping n-grams per window in six 600-item
flat partitions, produced 43k–143k-character prompts, and encouraged clause-sized
proxy spans; 11/30 responses also truncated. Parser tolerance does not repair
the semantic miss.

Per the preregistered stop rule, do not build the broader reviewed benchmark or
inspect/run FROZEN_EVAL yet. A general Pass 1A development repair is now
implemented and independently approved: keep the exhaustive catalog as the
coverage oracle, but group all
end alternatives by exact source-start anchor, expose exactly one anchor per
request with compact aliases and exact offsets, ask only for atomic mentions
beginning at those anchors, hide heuristic type hints, and safely accept only a
single complete JSON fence. No completed repaired-run artifact exists. A full
rerun process was stopped during its first case and the atomic runner published
nothing; that process note is not representation evidence. Structural review
of the unevaluated repair cannot overturn the valid 0/33 and 0/75 strong-model
result.

Phase 2F is now closed at the source-semantic representation gate. The final
report is [docs/phase2f-semantic-ir-stop-gate.md](docs/phase2f-semantic-ir-stop-gate.md).
Formal DEV/FROZEN gold was not created, frozen labels were not inspected, and
the frozen run was not consumed because the preregistered legacy gate failed.
Do not begin Pass 2 or any later compiler pass without a new explicit goal.

```text
SEMANTIC IR NOT VIABLE — REDESIGN PASS 1
```

## 9. Phase 2G endpoint-recovery ablation closed

Phase 2G replaced source-text regeneration with compact candidate-ID selection
and held the 33 reviewed endpoints fixed across Raw Bronze, Mechanical Silver,
and Resolved Silver. The experiment is implemented at clean commit `64baf2b`.
The final report is
[docs/phase2g-endpoint-recovery-ablation.md](docs/phase2g-endpoint-recovery-ablation.md).

Two clean 15-call runs completed against the official DeepSeek endpoint with
`deepseek-v4-pro`, thinking disabled, and no provider failures. Deterministic
coverage stayed 33/33. No condition passed. Across the two runs, endpoint
recall ranged from 4/33 to 10/33, precision from 1.3% to 11.3%, reviewed
reference-status accuracy was always 0/8, and broad wrong-candidate selection
dominated. Mechanical and Resolved Silver produced no stable material lift.
Every known selection remained exactly traceable to Bronze.

The clean rerun reproduced the failed gate and diagnosis but not exact score or
failure distributions; this negative reproducibility evidence is retained.
Phase 2G is therefore closed as:

```text
TARGETED NEXT INTERVENTION
DIAGNOSIS: MODEL-CAPABILITY BOTTLENECK
```

Exactly one next action is authorized: rerun the frozen candidate-ID benchmark
with the same `deepseek-v4-pro` teacher and thinking enabled. Do not tune the
prompt, candidate catalog, Silver fixture, reviewed endpoints, parser, ontology,
or downstream graph stages before that intervention.
