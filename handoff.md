# Handoff: Phase 2J Stopped at the Human Annotation Checkpoint

**Current checkpoint:** Phase 2J has deterministically selected 30 new Bronze
windows from 30 independently recorded video source groups and produced a
blank scorer-blind two-pass annotation packet. The source-group split is fixed
at 24 Expanded DEV / 6 Frozen Replication and remains `LOCKED`. No new gold,
candidate-coverage result, B/C prediction, or Phase 2J disposition exists yet.
Human Pass A is complete (30 windows / 166 endpoints); Pass B is completed in
the `/phase2j-adjudicate/` route as the explicit five-check human audit
attestation, and the canonical importer is ready. No reviewed packet exists
until a completed `phase2j-adjudication-export-v2` is imported. See sections
12–13.

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

## 10. Phase 2H discriminative endpoint scoring closed

Phase 2H (initial implementation `65e7100`, corrected clean experiment
`a754991`) replaced generative recovery with a fully offline,
deterministic candidate-level binary KEEP/DROP endpoint scorer over the exact
frozen Phase 2F/2G candidate universe: 5-window grouped leave-one-window-out,
16,624 candidates, 33 KEEP, 16,591 DROP, 33/33 coverage. Fixed threshold 0.5;
class-balanced L2 logistic and conservative LightGBM on Feature Set A
(geometry/provenance) and Feature Set B (A plus bounded lexical/cue features).
No generative calls, no syntax/UD work, no roles, no edges, and no graph work.
The final report is
[docs/phase2h-discriminative-endpoint-scoring.md](docs/phase2h-discriminative-endpoint-scoring.md).

Final status:

```text
WEAK RANKING SIGNAL ONLY
```

Gate 1 failed; Gate 2 was not triggered. The signal is real but weak and
inconsistent: logistic B reaches the best pooled ranking (P 8.772%, R 30.303%,
AP 0.081275, AUC 0.939701, R@10 18.182%, median gold rank 92); logistic A
reaches 78.788% recall only by flooding 1,907 selections at 1.363% precision;
LightGBM B reaches P 2.757%, R 33.333%, AP 0.050952, AUC 0.932765, R@10
12.121%, median gold rank 157; LightGBM A reaches P 2.513%, R 30.303%, AP
0.026943, AUC 0.861030. Only 4/33 gold endpoints are selected by all four
cells, 7/33 by none, and every B/LightGBM true positive lies inside logistic
A's 26-hit recall set. LightGBM did not establish nonlinear superiority.

Reproducibility: two clean runs at `a754991` (`repository_dirty: false`)
produce identical dataset, folds, metrics, feature findings, and error
taxonomies; definition SHA-256
`75dfaca522195ccd953825317c72e9780781c6c2b45b19f2655638d544c4a459`; benchmark
content SHA-256
`a17674b6e2c491f0d7a1600dde0cfb8cc533d1d17db8633d8d94b2de9a57c1dd`. Archives:
`data/phase2h_artifacts/phase2h-endpoint-scoring-run1.tar.gz` (SHA-256
`22aaab162f6122691f577bc95746a0b7b1da9834706766b746a29737a5e46380`) and
`run2.tar.gz` (SHA-256
`02b3e62030169c3c394b10ca3440ea1433bd270d2fe207f68ac1d7a6e165d817`).

Testing/review: focused Phase 2H suite passes 46 tests + 16 subtests; a fresh
independent review of the corrected state returned APPROVE; the broad non-auth
suite passes 681 tests + 366 subtests.

Exactly one next intervention is justified: a bounded Feature Set C
UD/syntactic ablation with the same frozen candidate universe, labels, folds,
threshold, two model families, and metrics, retaining A/B as frozen
comparators and stopping if parser integration becomes substantial. Do not
expand the data or run generative interventions now. The
Phase 2G thinking-enabled generative rerun recommendation is superseded by
this result; Phase 2G history and its diagnosis remain preserved.

## 11. Phase 2I UD / syntactic feature ablation closed

Phase 2I is implemented at clean commit `9a88c3a`. It added only Feature Set C:
the frozen Phase 2H geometry + lexical matrix plus deterministic Stanza 1.14.0
English EWT UD/syntactic evidence. Bronze text, all 16,624 candidate rows, 33
KEEP labels, 33/33 endpoint coverage, five grouped folds, threshold 0.5, seed,
model configurations, and the archived B predictions remained unchanged. No
generative model, candidate-bound change, SRL, role model, semantic edge, graph,
ontology, or Phase 3 work was performed. The final report is
[docs/phase2i-ud-syntactic-feature-ablation.md](docs/phase2i-ud-syntactic-feature-ablation.md).

Final status:

```text
SYNTACTIC SIGNAL FOUND
```

Logistic C regressed relative to Logistic B: P 5.882%, R 18.182%, F1 0.08889,
AP 0.049788, AUC 0.929188, R@10 4/33, median rank 89, and 102 selections.
LightGBM C supplied the positive evidence: versus LightGBM B, precision rose
from 2.757% to 7.692%, recall from 33.333% to 36.364%, F1 from 0.05093 to
0.12698, AUC from 0.932765 to 0.950458, median gold rank improved from 157 to
96, mean rank from 226.24 to 170.09, and selections fell from 399 to 156. AP
fell from 0.050952 to 0.041251 and R@10 remained 4/33, so top-of-list ranking
is still unstable.

The LightGBM gains were distributed rather than isolated: precision, F1, AUC,
selection efficiency, and mean rank improved in all five held-out windows.
Syntax accounted for 31.5%–83.1% of gain by fold and reduced generic action,
generic entity, filler, cue-prior, and overlap false positives. Five of the
seven universally missed Phase 2H endpoints moved upward but none crossed the
fixed threshold. Parser projection produced 597 EXACT and 16,027
TOKEN_ALIGNED candidates, with no partial, unaligned, ambiguous-status, or
objective parser/alignment errors.

The signal is a five-window proof, not a production model decision. Severe
train/held-out AP gaps and 132–187 held-out syntax OOV values per fold remain.
A fresh independent post-fix review returned `ACCEPT`. The Phase 2I syntax and
CLI suite passed 74 tests; the endpoint/artifact suite passed 49 tests and 8
subtests after two corrected expectation checks passed 2 tests and 5 subtests;
the frozen Phase 2H regression passed 46 tests and 16 subtests; and six new
adversarial provenance/JSON/Git/manifest checks passed.

Two clean official runs matched on deterministic inputs, parser tables,
candidate scores/ranks, metrics, deltas, diagnostics, importances, and content
hashes. Retained archives:

- `data/phase2i_artifacts/phase2i-syntax-features-run1.tar.gz`, SHA-256
  `ed1c489c8ce273adb59b6321017d03c89015c8afad2f3e3e6cade813458cc4ad`;
- `data/phase2i_artifacts/phase2i-syntax-features-run2.tar.gz`, SHA-256
  `6fd83bf9f0bfafbf58f43eac3b3203fc2a1ac7e0292bf6fad52b36f6fe89afb5`.

Exactly one next intervention is justified: expand the reviewed candidate-level
dataset with substantially more independent source windows while preserving
group-level source isolation. Do not begin it without a new explicit goal. Do
not tune the current models, redesign candidate bounds, add SRL, train semantic
edges, or start Phase 3 first.

## 12. Phase 2J pre-annotation checkpoint

Phase 2J changes only data quantity and source independence. The current
uncommitted checkpoint adds deterministic source selection, a locked 24/6
source-group split, strict artifact/input hashes, and a scorer-blind two-pass
annotation packet. The implementation and reviewer instructions are documented
in [docs/phase2j-annotation-checkpoint.md](docs/phase2j-annotation-checkpoint.md).

Official pre-annotation artifacts:

- `data/phase2j/window-selection-manifest-v1.json`, canonical content SHA-256
  `4d19b29db9bf7b31baca24b8b32ee1c082830bdf692309e2c65662cb313382b9`;
- `data/phase2j/endpoint-annotation-packet-v1.json`, canonical content SHA-256
  `3f766b08696ed512063d999c75877001d77b03db136f8edae78e631e1725c62a`.

The manifest contains exactly 30 windows from 30 distinct `video:` source
groups, excludes the three legacy upstream videos, fixes 24 groups as
`EXPANDED_DEV` and 6 as `FROZEN_REPLICATION`, and binds 30,788 frozen candidate
rows by count/hash only. The packet contains zero endpoint annotations and zero
reviewer signatures. No parser, Feature B/C, Logistic, LightGBM, or scoring
path ran.

Validation: 34 focused Phase 2J tests pass after directly pinning the
preregistered +8/+4/+2/+1 diversity arithmetic, seeded tie-break, ASR-band
mapping, Pass-B provenance gate, clean `IN_REVIEW` transitions, immutable
manifest/Bronze binding, and atomic reviewed-packet finalization. The combined
Phase 2J/source/candidate run passes 78 tests plus 45 subtests; frozen Phase 2H
regression passes 46 tests plus 16
subtests; Phase 2I syntax tests pass 62 with 4 asset-dependent skips. A fresh
independent audit returned `ACCEPT WITH NON-BLOCKING LIMITATIONS` and confirmed
the artifacts are safe to hand to human review.

Human review is available in the Notion workspace
[Phase 2J — Human Endpoint Review Workspace](https://app.notion.com/p/3c0f8ba78bf38133b6e9c3b61e0db22e?pvs=204),
which contains all 30 exact-Bronze windows, indexed tokens, editable endpoint
tables, Pass A/Pass B properties, and scorer-blind queue/audit views.
Reviewer-facing champion, role, and video-title clues are hidden: annotations
must preserve literal Bronze mentions and may not resolve identities or repair
ASR from metadata/game knowledge. Windows can be marked `AMBIGUOUS` when only
some spans are defensible or `EXCLUDED` with no endpoints when ASR/context loss
makes endpoint discovery unreliable. Notion entries remain human review
material until imported and validated against the locked repository manifest.

For Pass A, the local Quizlet-style span review route is now preferred:
`http://localhost:3000/phase2j-review/`. It displays one Bronze window at a
time; drag-selection snaps to token boundaries and opens an endpoint-type
picker, with highlights, removal/undo, deck navigation, clean/ambiguous/
excluded outcomes, packet-hash-bound browser autosave, and validated JSON
backup export/import. The static page receives no split, candidate, model,
champion, role, video-title, or Sol-review fields. From `apps/web`, validation
passes 41 Jest tests and `npm run build`; serve the exported production UI with
`python3 -m http.server 3000 --bind 0.0.0.0 --directory out` and open
`http://localhost:3000/phase2j-review/` from the host browser. Its JSON export
is review material and must be imported/validated before it can affect the
canonical packet.

A parallel `gpt-5.6-sol` high-reasoning review of the same 30 windows is sealed
at `/tmp/phase2j-sol-high-independent-review-v1.json` until human Pass A is
complete. It is audit/navigation material, never gold. Its 30 identities and
338 exact token/Bronze spans validate; file SHA-256 is
`6ef4ccbff8f9512b9119d314050acd5aaa87b927c37ee83372fcec92edd1cd8c`
and canonical content SHA-256 is
`8025d05c1bbe4f5b8c5c38d3689b96f69019087a3390e33cbfe98d2865ea0e53`.

This is the historical pre-annotation checkpoint. It is superseded by the
completed import and final coverage-gate closeout in section 14.

## 13. Phase 2J-A post-Pass-A adjudication and canonical import gate (REVIEW MATERIAL)

Human Pass A is complete (30 windows / 166 endpoints). The sealed
`gpt-5.6-sol` High review is therefore **revealed for explicit human
adjudication only**; it is a navigation/audit second opinion and is **never
gold and never auto-promoted**. The human remains the baseline; every
disagreement must be resolved explicitly before an export is accepted. The
adjudication plus the explicit five-check human attestation **is** the Pass-B
audit; Sol remains only a second opinion. At this historical checkpoint, the
adjudication export remained `REVIEW_MATERIAL` until validation by the canonical
importer below. The completed import and final gate are recorded in section 14.

The deterministic sanitized packet is
`data/phase2j/phase2j-adjudication-packet-v1.json` (content SHA-256
`13aaa1a9d6ecdba2d16b722109373e26494467e1b14d21d26458b93c8750015b`), built by
`uv run python scripts/build_phase2j_adjudication_packet.py` from the locked
packet, the human export, and `/tmp/phase2j-sol-high-independent-review-v1.json`.
It binds the human/Sol file hashes, strips reviewer identity and all
model/candidate/packet-internal fields, and classifies 326 connected
components: 49 exact agreements, 16 type disagreements, 87 boundary
disagreements, 174 Sol-only, 0 Human-only (166 human / 338 Sol endpoints).

The local adjudication route is `http://localhost:3000/phase2j-adjudicate/`
(built by `cd apps/web && npm run build`, served from `out`). It reads only the
generated packet at build time; the `/phase2j-review/` Pass A route is
unchanged. The UI provides neutral Human/Sol overlays, per-component decisions
(Human, Sol, custom exact span/type, or drop), `CLEAN`/`AMBIGUOUS`/`EXCLUDED`
window outcomes with required notes, explicit “Keep my Pass A choices”,
progress, autosave, JSON import/export, reset, and a global Pass-B audit
attestation. Exports use schema `phase2j-adjudication-export-v2`; component
decision state remains localStorage-compatible, while the five audit checks
persist separately and are restored from a validated export. Exact-agreement
components are pre-kept by default but remain editable and drop-able, because
agreement is evidence, not proof. Export is blocked until every component in a
`CLEAN` window is resolved, `AMBIGUOUS`/`EXCLUDED` windows carry a required
note (`EXCLUDED` clears all endpoints), the resolved endpoint set is
duplicate/overlap-free, and the five-check attestation (boundaries, omissions,
roles, duplicates, ambiguity) is all true.

The canonical importer (`pipeline/phase2j_adjudication_import.py` +
`scripts/import_phase2j_adjudication.py`) rebuilds a separate reviewed packet
from the locked blank packet, the locked manifest, the original human Pass-A
session, the generated adjudication packet, and the completed export v2. It
fails closed: it independently validates every input hash, schema,
record/window/component order, Bronze slice, component decision and
`resolved_by` semantics, derived endpoint fields, audit checks, and
overlap-freedom. The default output is
`data/phase2j/reviewed-endpoint-annotation-packet-v1.json`; the blank packet
is never overwritten, and `release_gate` stays `LOCKED`. Sol enters only via
an explicit human `KEEP_SOL_SET`/`CUSTOM` decision (`HUMAN`/`SHARED` map to
`PASS_A`, `SOL`/`CUSTOM` map to `PASS_B`); `UNDETERMINED` maps to canonical
`null` rather than inventing a type. `CLEAN` maps to `REVIEWED` + Pass B
`COMPLETE`; `AMBIGUOUS` maps to `AMBIGUOUS` + Pass B `IN_PROGRESS` (non-gold);
`EXCLUDED` maps to `EXCLUDED` + empty endpoints (non-gold).

```bash
python3 scripts/import_phase2j_adjudication.py \
  --export /path/to/phase2j-adjudication-export-13aaa1a9.json

python3 scripts/import_phase2j_adjudication.py \
  --export /path/to/phase2j-adjudication-export-13aaa1a9.json \
  --validate-only
```

Use `python3` to avoid `uv` network sync. After import, check the reviewed
output and assess sizing; only eligible reviewed windows may proceed to
candidate coverage.

Test evidence: the new importer tests pass **18 tests**
(`tests/test_phase2j_adjudication_import.py` + `tests/test_import_phase2j_adjudication_cli.py`);
the adjudication web suite passes **29 focused Jest tests**, the full web
suite passes **70 tests**, and `tsc`/`build` pass. Combined Python totals
across the full Phase 2J suite have not been supplied and are not claimed
here. See `docs/phase2j-annotation-checkpoint.md` sections 6.3–6.4 for
commands and totals.

This is the historical post-Pass-A gate. The completed import and final
coverage-gate closeout are recorded in section 14.

## 14. Phase 2J final closeout — exact-boundary gate failure

Phase 2J is closed with the required disposition:

`ANNOTATION CONTRACT NOT STABLE`

The completed scorer-blind two-pass import produced 30/30 reviewed windows and
311 gold-eligible endpoints, exceeding the 150-endpoint sizing target. The
reviewed packet content SHA-256 is
`c239070e107e0848e8d26918d33ece5fa978f9ce48e0f43a2e65b67cd622365d`.

The frozen 30,788-candidate coverage gate found 263/311 exact matches
(84.566%): Expanded DEV was 216/243 (88.889%) and Frozen Replication was 47/68
(69.118%). The coverage artifact content SHA-256 is
`1ac837aae4a4411837d2277f23ce613f531ffb5dec57e449e0a7fb4c14a2daa2`.

All 48 misses are terminal-punctuation boundary mismatches: each has a frozen
candidate with the same start and semantic text, ending exactly one character
before the reviewed gold span (28 periods, 20 commas). There are no semantic
no-overlap misses. The whitespace-token annotation workflow retained terminal
punctuation while the frozen candidate generator excluded it, and the gold
contract did not define the governing convention. Phase 2J therefore stopped
before any B/C parser/model run; no syntactic replication claim is made.

An independent read-only `gpt-5.6-sol` high-reasoning audit reproduced source
isolation, hash lineage, eligibility, arithmetic, deterministic coverage, and
the punctuation-only diagnosis, and accepted this disposition. Full evidence
is in
[docs/phase2j-independent-source-replication.md](docs/phase2j-independent-source-replication.md).

Exactly one next intervention is justified: perform a versioned, scorer-blind
terminal-punctuation boundary correction, with human Pass-B re-adjudication
limited to the 48 affected endpoints, then regenerate exact candidate coverage
while keeping the candidate generator frozen. Do not begin it without a new
explicit goal.

## 15. Phase 2K contextual reconstruction — live dataset complete, human review pending

Phase 2K is now implemented as an isolated experiment over immutable Phase 2J
inputs. Pass 1 is strictly text restoration: its JSON envelope contains only
`clean_text`, explicit repairs, uncertainties, and provenance. Semantic
endpoints, entities, events, relations, and champion bindings are excluded
from mechanical cleanup and occur only after semantic-sufficiency diagnosis
and deterministic adaptive context expansion.

Implemented tooling covers ordered source retrieval, conservative mechanical
repair, adaptive sufficiency decisions, source-faithful reconstruction,
semantic polish, complete repair/binding provenance, ambiguity preservation,
A/B/C/D human review, context-radius auditing, source-bound downstream target
alignment, comparison-v2, and a gate-locked paired Phase 2F/Phase 2H evidence
runner. The runner aborts atomically on typed provider failures, binds exact
compiler aliases/config/input lineage, and deterministically replays Phase 2H
folds, fit scope, scores, and metrics during validation.

Current verification evidence:

- downstream rerun/alignment/contract suite: **87 tests + 98 subtests passed**;
- reconstruction/build-CLI suite after the v15 repair: **125 tests + 4
  subtests passed**;
- Phase 2K contract suite after updating the stale four-attempt fixture:
  **31 tests + 50 subtests passed**;
- rerun module and both CLIs pass `py_compile`;
- focused provider, D-integrity, family-matching, alias, and tamper regressions
  are included in `tests/test_phase2k_downstream_rerun.py`.

This does **not** close Phase 2K. The authoritative live reconstruction dataset
is now `/tmp/phase2k-live-v15-current`: it deep-validates with **30/30 D
records generated, zero placeholders, and zero window failures**. The frozen
Phase 2J input hash remains
`36d7f02f7fc74ea6ecc9d72028aa58cfa9967cc7ed7a972ba864bce7a1f4004a`.
The records SHA-256 is
`c544c12b0527c1452b5b558a612c14b273c1f50e1eed8a4cbaaca1e800a0d5f8`,
the blind human packet SHA-256 is
`4ffaee566c428fae05f678752f4943fd7e26a5943bcf0ab2d3a15fe253c0c303`,
and the blank transformation-audit SHA-256 is
`0348773ece8959b68ee16177d44c8b0ad12fb3ba46fbc3babfc03673bf305ed3`.
The live output contains 137 distinct raw provider responses.

The v15 local repair conservatively omits only context-only proposals from
normalized *target* bindings while preserving them in `raw_compact` and
counting the omission. Mentions absent from both target and context still fail
closed, and omitted bindings cannot license clean-text entities. Semantic
polish remains strict; its v3 correction prompt now receives the exact actual
text and exact source quote plus source-exact / reconstruction-derived repair
instructions. Base provider prompts and response schemas are unchanged, so
the existing cache remains reusable.

The active gate is scorer-blind human review: 270 representation/radius items
and 254 transformation operations (35 mechanical repairs, 37 contextual
repairs, 23 entity bindings, 8 pronoun bindings, 4 reference bindings, 1
ability binding, and 146 polished statements). Start the local client-only UI:

```bash
cd /home/bphan944/PersonalProjects/videoSorter
cd apps/web
npm run dev
```

Open `http://localhost:3000/phase2k-review` and import
`/tmp/phase2k-live-v15-current/phase2k-human-review-packet-v2.json`. Open
`http://localhost:3000/phase2k-audit` and import
`/tmp/phase2k-live-v15-current/phase2k-transformation-audit-packet-v2.json`.
Both routes keep packet data in the browser, autosave hash-bound progress, and
refuse a final export until every required human field is complete.

The build summary's transformation-audit hash is authoritative for the newly
built blank audit. The current `--validate-only` summary reports that field as
`null` when no finalized human audit exists; a zero exit still means the
output directory passed deep validation.

After both completed browser exports exist, run
`scripts/finalize_phase2k_human_review.py`, require a `PASSED` human gate,
finalize source-bound target alignment, run the paired unchanged
generative/discriminative architectures, and produce the empirical diagnosis.
Until those steps are complete, Phase 2K cannot answer whether earlier failure
was primarily input representation, model capability, or candidate
discrimination.

## 16. Phase 2K v16 lexical cleanup — implementation complete, live rebuild pending

Human review of the v15 packet is paused. A bounded champion-name lexical
normalization layer is now implemented for Pass 1 without changing its
text-restoration-only boundary. Exact word-boundary hints never mutate text;
the provider must return every eligible occurrence as an explicit,
Bronze-span-bound `DOMAIN_SPELLING` repair. Direct, guarded, and exact
champion-metadata-licensed rules are versioned in
`data/phase2k_support/league_lexical_vocabulary_v2.json`. General fuzzy
matching is forbidden, and common false matches (`like`, `then`, `when`,
`ward`, `well`) plus automatic `Soie -> Zoe` fail closed.

Lineage is now pipeline v7, config v3, records v7, build summary v5,
vocabulary v2, and mechanical base/correction prompts v4. The old v15 cache
cannot satisfy the new mechanical lineage. Validation passes **168 tests + 54
subtests** across reconstruction/CLI/contracts and **23 tests + 15 subtests**
for downstream alignment compatibility. A deterministic scan of the 30
frozen targets finds 17 eligible repairs across 7 windows.

The next step is a fresh live build and deterministic validation:

```bash
.venv/bin/python scripts/build_phase2k_reconstruction.py \
  --live \
  --output-dir /tmp/phase2k-live-v16-current \
  --cache-dir /tmp/phase2k-live-v10-sEzueV/cache

.venv/bin/python scripts/build_phase2k_reconstruction.py \
  --validate-only \
  --output-dir /tmp/phase2k-live-v16-current
```

Do not resume browser review using the v15 packet hashes. After v16 validates
with 30 windows and zero failures, import its new human-review and
transformation-audit packets and continue the existing acceptance gates.
