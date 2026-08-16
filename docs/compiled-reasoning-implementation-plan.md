# Compiled Reasoning Implementation Plan

Last updated: 2026-08-15

## Architectural Objective

The architectural target is the Notion design document **videoSorter Design -
From Naive RAG to Compiled Strategic Reasoning** (page
`3bbf8ba7-8bf3-811d-9b96-cc7c4d2df5b4`). The implementation hypothesis is:

> Explicit strategic fingerprints, causal relations, and compiled principles
> let the same inexpensive answer model produce materially better
> first-principles League analysis than RAG evidence alone.

Notion is the source of architectural rationale. This file is the durable,
executable record of scope, gates, implementation status, tests, and risks.

## Constraints

- Retain vector and lexical retrieval; add relational structure alongside it.
- Use the current SQLite/Postgres/Supabase architecture. No graph database for
  the MVP.
- Keep source-grounded evidence (`videos` and `insights`) separate from derived
  strategic knowledge.
- Preserve evidence provenance for every relation, fingerprint, and principle.
- Make only small, reversible extensions. Do not add expensive synthesis before
  structured context proves useful.
- The query pipeline must retrieve and expose causal structure, not merely
  persist it.
- Empty or unavailable strategic data must preserve existing RAG behavior.

## Repository Mapping

Existing extensions used rather than duplicated:

- Evidence and local persistence: `core/database.py`.
- Hosted persistence: `supabase/schema.sql`, `cloud/migrate_supabase.py`.
- Vector/lexical retrieval: `core/embedded_vectors.py`, `cloud/vector_store.py`,
  and `retrieval/query.py`.
- Query normalization and routing: `retrieval/questions.py`.
- Existing derived game-data layers: `pipeline/champion_crossref.py` and
  `pipeline/ability_enrich.py`.

New, narrowly scoped strategic modules:

- `core/ontology.py`, `core/strategic_types.py`: validated derived domain data.
- `retrieval/strategic_context.py`: optional bounded strategic retrieval and
  prompt serialization.
- `data/strategic_fixtures_v0.json`: manually authored Phase 1 fixture.
- `scripts/eval_reasoning.py`: repeatable baseline-versus-structured harness.

## Phase Gates

### Phase 1: Manual Representation MVP

**Status: Complete. Gate passed; stop here.**

Purpose: prove the representation manually before automating extraction,
fingerprinting, hybrid retrieval, or synthesis.

Acceptance criteria:

- Ontology v0, validated relations/fingerprints/principles, and provenance exist.
- Fixtures cover Caitlyn, Kai'Sa, Yunara, Tristana, Thresh, Sylas, and artillery
  mage context.
- Derived objects persist separately from raw evidence with version/confidence
  filtering and provenance.
- Query-time retrieval adds a bounded useful causal neighborhood while raw RAG
  evidence remains intact.
- Baseline and structured answers use the same model and the same RAG snapshot.
- The four representative questions produce inspectable side-by-side output.
- Tests cover malformed data, duplicate/contradictory relations, stale data,
  missing data, confidence, cycles/hop limits, and RAG fallback.
- Each meaningful boundary has independent review, fixes, tests, and a focused
  commit pushed to `main`.

Decision rule: proceed only if structured context increases causal specificity,
state/resource reasoning, winning-line clarity, role dynamics, and opponent
adaptation without replacing evidence grounding.

### Phase 2: Automated Relation Extraction

**Status: In progress.** The constrained compiler, source packet loader,
canonical validation, persistence, inspector, and small reference evaluator are
implemented locally pending independent review and live-model validation. No
fingerprint automation has started.

Gate: accept only if a cheap model produces provenanced, ontology-valid,
condition-preserving relations close enough to the small manual references at a
useful precision/recall level. Review/reject output must never enter the graph.

#### Phase 2 M1: Compiler Contract and Persistence

**Complete. Commit:** `0ed16f3` (`main`).

- Added source-grounded `EvidenceItem`/`ExtractionPacket`, strict structured
  response parsing, deterministic canonicalization, confidence composition,
  condition-safe identities, and accepted/review/rejected decisions.
- Preserves the Phase 1 `StrategicRelation` object and strategic tables, adding
  an automated relation data version and ontology-version-safe uniqueness.
- Automated relations may only be persisted through accepted compiler decisions;
  all have one or more evidence references. Manual Phase 1 fixtures remain
  separate and usable.
- Tests/review: 64 deterministic tests passed. Independent reviewers corrected
  arbitrary-node acceptance, unsafe inverse aliasing, and legacy SQLite FK
  migration behavior before approval.

#### Phase 2 M2: Explicit Evidence Workflows and Evaluation

**Status: Validated stop gate. NOT READY — fix Phase 2 first.**

- Added explicit insight-ID packet construction, default dry-run extraction,
  opt-in persistence, relation inspection, and a four-case source-grounded
  reference evaluator. Automated accepted relations now participate in the
  existing bounded relation context; fingerprints and principles remain manual.
- Independent review approved after fixes for provenance merging, source-only
  ability aliases, ontology filtering, and cross-cluster contradiction review.
- Live DeepSeek V4 Flash evaluation on 2026-08-15 was dry-run only and made no
  persistence changes. It returned zero accepted relations against three
  reference edges (precision/recall: 0.00). The compiler safely rejected
  unregistered `action`/`mechanic` nodes, stripped unsupported concepts, and
  quarantined low-confidence valid edges, but this means the automation is not
  reliable enough to build fingerprints on top of it.
- Stop here. Before retrying the gate, improve the constrained response schema
  and reference evidence quality, add model-output normalization for supported
  compound ability aliases only, and recalibrate the confidence threshold from
  an audited sample. Do not start Phase 3.

### Phase 3: Automated Champion Fingerprints

**Deferred.** Assemble versioned fingerprints from high-confidence relations,
track coverage/freshness/contradictions, and compare against the manual fixture.

### Phase 2B: Diagnose and Repair Relation Extraction

**Status: Complete stop gate. NOT READY — continue Phase 2 repair. Do not start Phase 3.**

Purpose: locate semantic signal loss in the existing safe compiler before
changing validation or confidence behavior. The target is useful recall with
high precision, conditions, and provenance intact, not a larger accepted count.

#### M1: Reproduce Original Failures

**Complete.** [Baseline traces](phase2b-baseline-traces.md) reproduce the
three Flash non-thinking failures unchanged. Findings: two target relations
survive canonicalization but are routed to review by confidence; the reset case
correctly rejects invented `action`/`mechanic` nodes. Dedupe is not implicated.

#### M2: Fair Flash/Pro Experiment Plumbing

**Complete.** Added `DEEPSEEK_RELATION_FLASH_MODEL` and
`DEEPSEEK_RELATION_PRO_MODEL`, plus evaluator `--variant flash|pro`. Variants
require the DeepSeek provider, custom models are explicitly labeled, and no
prompt, packet, thinking mode, output budget, ontology, or validator changes
between benchmark variants. Independent review approved the boundary.

Next: build a larger reviewed corpus before diagnosing confidence or changing
semantic abstraction.

#### M3: Labeled Corpus, Model Comparison, and Grounding Repair

**Complete. Commits:** `1539f2c`, `a61ccb4`, `ed73d38`, `9153dbd` (`main`).

- Added a 23-case, source-grounded Phase 2B set with 18 positive reference
  edges, four negative cases, and three review-only ambiguity cases. The
  evaluator now reports TP/FP/FN, F1, review matches, decision reasons, and
  runtime packet entity-coverage warnings without treating condition prose as
  exact identity.
- Flash non-thinking: 0/18 accepted true positives. Pro non-thinking initially
  recovered 3/18 with no measured false positives, demonstrating higher raw
  extraction capability, but post-repair reruns remained at 0 accepted true
  positives. Pro thinking at 512 output tokens produced only empty final
  content, so it is not a valid quality comparison.
- The repair added single-champion bare-slot aliases, a narrow evidence-cue
  table for existing concepts, and ordered single-evidence condition grounding
  with negation preservation. Independent review rejected broad cue matching
  and unordered condition matching; those defects were fixed before commit.
- The post-repair benchmark exposed a remaining conflict: strict qualifier
  grounding correctly rejects alias/paraphrase conditions emitted by the model,
  while accepting them would weaken the source-boundary. A prompt requiring
  source condition wording did not restore useful accepted recall.

**Decision:** do not automate fingerprints or backfill relations. The compiler
is safe but not reliable enough for Phase 3. The next Phase 2 effort should
evaluate a two-stage grounded-proposition representation or a condition-aware
semantic entailment reviewer on a held-out corpus, with the same provenance and
review quarantine invariants.

### Phase 2C: Source-Aligned Semantic Grounding

**Status: In progress. Do not start Phase 3.**

Hypothesis: a canonical strategic term need not be literal source wording when
each canonical field retains an inspectable, evidence-bound source-to-canonical
mapping. Concrete entity grounding remains strict; semantic entailment and
ontology abstraction are separate validation steps.

#### M1: Preserve Phase 2B Baseline

**Complete.** The Phase 2B artifacts remain the comparison baseline: Flash
non-thinking accepted 0/18 reference true positives, Pro non-thinking peaked
at 3/18 with precision 1.00 and recall 0.17, and stricter grounding variants
accepted none. No historical run has been replaced or reclassified.

#### M2: Alignment Domain and Persistence

**Complete. Commit pending.** Added `RelationAlignment` to the existing
`StrategicRelation` model. It preserves field, source text/span, evidence ID,
canonical target, mapping type/confidence, and mapping version. SQLite,
Supabase schema/migration payloads, relation merging, and the relation
inspector retain it. Legacy databases receive an empty alignment list.

Invariants: an alignment must reference relation evidence and target the exact
canonical relation field. Reruns merge using `(field, evidence_id,
canonical_value)` and deterministically retain the highest-confidence mapping;
the merged relation is validated before persistence. Existing Phase 1 manual
relations remain valid with no alignments.

Tests: 49 focused persistence, inspector, domain, and cloud-migration tests
pass. Independent review found an initial invalid alignment-merge state and a
missing Supabase propagation path; both were fixed and re-reviewed approved.
This milestone is storage and inspection only. The extractor has not yet begun
to emit alignments; that is the next isolated behavior change.

#### M3: Strict Source-Aligned Extraction Contract

**Complete. Commit pending.** The extraction prompt now requires source text
and an evidence ID for the subject, predicate, object, and any condition. The
compiler rejects a missing, fabricated, uncited, or canonically mismatched
anchor. Accepted automated relations persist `RelationAlignment` records for
all of those fields.

Condition anchors must be literal evidence wording, name an ordered
negation-compatible part of the canonical condition, and the full canonical
condition must remain supported by the same cited evidence. This rejects an
unrelated nearby phrase being attached as condition provenance. The previous
strategic-context integration fixture was updated to the new candidate
contract.

Tests: 93 focused extractor, context, persistence, inspector, domain, and
cloud-migration tests pass. Independent review found an initially unpropagated
context fixture and a condition-alignment provenance bypass; both were fixed
and approved. This remains intentionally strict: nonliteral semantic concept
and condition aliases are still rejected until the next alias/entailment
milestone.

#### M4–M5: Deterministic Entity and Semantic Alias Mapping

**Complete. Commit pending.** Reused the existing champion/ability registry and
relation-normalization boundary. Curated source aliases now map selected
evidence phrases such as `staying on your ADC` to the existing `continuity`
concept, and explicit causal verbs such as `prevents` to `denies`. All aliases
are inspectable constants in `core/relation_normalization.py`, resolve only to
ontology-v0 values, and retain the literal source anchor in the stored
alignment.

Safety decisions: ambiguous capability and generic coaching verbs (`cannot`,
`can't`, `let`, `open`) are deliberately not deterministic relation aliases.
Predicate aliases reject local prefix negation (`do not allow`, `nothing
prevents`) and an immediate negated complement (`allows no`, `stops no one`),
so no reverse edge is fabricated from a bare verb. The exact positive
`Thresh E --denies--> continuity` source-alias case and these adversarial
negation cases are covered. 97 focused tests pass; independent review required
and approved both alias-safety repairs.

#### M6–M7: Structured Source Events for Conditions

**Complete. Commit pending.** Added the optional `ConditionEvent` to the
existing `StrategicRelation` representation. Phase 2C currently accepts only
the audited mapping `missed -> temporarily_unavailable`. It retains the exact
source phrase, evidence ID, canonical ability, temporal operator, event, and
derived state alongside the legacy condition string. Entity and temporal words
must appear in the event phrase; unknown events and unsupported states reject.

`condition_event_json` is additive in SQLite, Supabase, and cloud migration.
Legacy SQLite strategic tables rebuild their derived relation identity to add
the field while retaining evidence and alignment JSON. Event identity is part
of the domain stable key, generated relation ID, SQLite/Supabase unique keys,
and persistence match query. Retrieval and the relation inspector expose the
source event and derived state to the answer model and operator. 82 focused
tests pass. Independent review found and verified fixes for source binding,
event scope, retrieval visibility, legacy identity migration, and an
event-distinct ID collision.

#### M8: Source-Aligned Prompt Diagnostic

**Complete. Commit pending.** The source-aligned prompt now enumerates every
allowed entity type, prohibits free-form `action`/`attribute`/`opportunity`
nodes, and permits non-core entities only when they occur in the packet's
explicit alias registry. A deterministic test preserves truncated JSON as a
parsing failure. Independent review approved the boundary.

Live dry-run benchmark at a shared 1024-token output limit: Flash and Pro both
accepted 0/18 reference relations. The larger limit eliminated Flash parsing
failures, but not canonical failures. Flash emitted 23 unknown
entity/concept/relation candidates and four subject-alignment mismatches; Pro
emitted 21 and four respectively. This supports the Phase 2C Case C diagnosis:
one call is overloaded with grounded proposition extraction and custom ontology
abstraction. The validator is not being relaxed; the next experiment is the
optional grounded-proposition fallback.

#### M9: Optional Grounded-Proposition Fallback

**Complete. Commit pending.** The fallback is explicitly two-stage and does
not change the default one-pass compiler. Stage A extracts source-only causal
propositions and rejects any phrase not present in cited evidence. Stage B sees
only those validated propositions plus ontology and alias constraints, never
the raw evidence packet; its output still passes the existing source-aligned
candidate validator. Both stage budgets are configurable and bounded
separately (512 and 768 defaults). Threshold validation occurs before model
calls, and Stage A/B raw failures remain in the trace. Mocked tests cover a
successful source-preserving path, fabricated source rejection, threshold
short-circuiting, and Stage B raw-evidence isolation. Independent review
approved the boundary.

### Phase 4: Hybrid Vector + Graph Retrieval

**Deferred.** Expand from vector/lexical seeds with bounded confidence,
concept, entity, hop, and freshness filters.

### Phase 5: Strong Offline Synthesis

**Deferred.** Add gated, high-value offline synthesis only after structured
context continues to prove useful.

### Phase 6: Demand, Cache, and Invalidation

**Deferred.** Add content-hash caching, demand triggers, budgets, and
incremental invalidation only after synthesis is validated.

### Phase 7: Expanded Evaluation and Calibration

**Deferred.** Build a larger golden set and cost/quality A-B-C-D evaluation.

## Phase 1 Milestones

### M1: Domain Model and Fixture

**Complete. Commit:** `7a63887` (`main`).

- Added ontology v0 (20 concepts), typed `StrategicRelation`,
  `ChampionFingerprint`, `CompiledPrinciple`, and evidence references.
- Added the seven-entity manual fixture.
- Invariants: non-speculative derived data requires provenance; confidence is
  finite and bounded; stale versions are rejected; duplicate relations merge
  evidence; conditioned contradictions remain distinct.
- Tests: `uv run pytest tests/test_strategic_types.py` (16 passed at milestone).
- Independent review fixed version validation, canonical dependency handling,
  evidence-loss on duplicate merge, and malformed-confidence behavior.
- Existing RAG behavior: unchanged.

### M2: Minimal Derived Persistence

**Complete. Commit:** `a9b4d96` (`main`).

- Extended existing local and hosted schemas with additive strategic tables and
  explicit provenance tables; raw `insights` remain unchanged.
- Added typed fixture persistence, FK-safe evidence merging, version handling,
  and optional hosted strategic sync.
- Tests: strategic persistence/migration/retrieval focused suite (41 passed);
  adjacent regression suite (59 passed).
- Independent review fixed nested validation, FK enforcement, stale provenance,
  forward migration, hosted cache replacement, and normalized duplicate keys.
- Existing RAG behavior: empty strategic tables are a no-op.

### M3: Strategic Retrieval and Context Builder

**Complete. Commits:** `0a615c2`, `0969074` (`main`).

- Added bounded SQLite strategic retrieval for fingerprints, compiled
  principles, and a relevant causal subgraph.
- Seeds entities from question text (including aliases/possessives), retains
  provenance, bounds global relation count/hops, filters stale/low-confidence
  data, and handles cycles/malformed rows safely.
- Tests cover unknown entities, missing fingerprints, empty neighborhoods,
  confidence filtering, stale data, duplicate/contradictory relations, cycles,
  and unchanged base retrieval fallback.
- Independent review corrected global relation bounds and question entity seeding.
- Existing RAG behavior: unchanged when no strategic data is present.

### M4: Query-Time Prompt Integration

**Complete. Commit:** `2e17322` (`main`).

- Added explicitly separated prompt sections for retrieved coaching evidence and
  derived strategic context. Derived context is optional and cannot replace raw
  evidence.
- Added an opt-out flag for baseline evaluation and a 2v2 matchup route that
  passes all four strategic entities.
- Tests cover prompt separation, failures/no-op behavior, and existing answer
  flow.
- Independent review approved after routing and fallback fixes.
- Existing RAG behavior: baseline path remains available.

### M5: Evaluation Harness and Phase Gate

**Complete. Commit:** `d5c7641` (`main`).

Purpose: compare the same answer model with identical source evidence, once
without and once with structured context. The output retains both answers and
the exact base evidence snapshot in JSON for manual scoring of grounding,
causal depth, state model, resource reasoning, winning line, role dynamics,
specificity, opponent adaptation, and teaching quality.

Inputs/outputs:

- `scripts/eval_reasoning.py --live --top-k 8 --json-output <path>`.
- Four cases: Caitlyn/artillery mage; Yunara + Thresh vs Tristana + Yuumi;
  Kai'Sa conditional access/concentration; Sylas HP joust/second rotation.
- Default execution is deterministic/plumbing-only; `--live` is opt-in.

Tests and review:

- Focused strategic/query/evaluation suite: 50 passed.
- Adjacent API/migration/publishing regression suite: 23 passed.
- Independent reviewer approved M5 after inspecting implementation, deriving
  edge cases, and running the focused suite.

### DeepSeek Answer Provider

**Complete. Commits:** `b485591`, `43a4f36` (`main`).

- Added DeepSeek's OpenAI-compatible chat provider and explicit
  `LLM_PROVIDER=deepseek` selection, preserving automatic provider fallback
  when unset.
- Hosted runtime validation now matches the provider contract for Gemini,
  DeepSeek, Ollama, and auto selection.
- Tests cover request shape, malformed/empty responses, precedence, missing
  credentials, hosted validation, and `.env` isolation: 43 passed.
- Independent review found and verified fixes for hosted credential validation
  and test isolation; final review approved.

## Phase 1 Evaluation Result

Run on 2026-08-15:

```text
LLM_PROVIDER=deepseek VECTOR_BACKEND=sqlite \
  uv run python -m scripts.eval_reasoning --live --top-k 8 \
  --json-output /tmp/phase1-reasoning-eval.json
```

Model: `deepseek:deepseek-v4-flash`. The harness used one local SQLite RAG
snapshot per case for both answers. The saved JSON was 42,835 bytes.

Observed comparison:

- **Caitlyn vs artillery mage:** baseline repeated kit/range advice; structured
  answer explained persistent low-cost pressure versus intermittent spell
  windows, initiative after misses, and the resource budget.
- **Yunara + Thresh vs Tristana + Yuumi:** baseline gave generic wave/lantern
  advice; structured answer separated access from continuity, identified
  Thresh's Flay as the continuation denial, and described Yuumi's amplification
  of a successful entry.
- **Kai'Sa:** baseline described an opportunistic all-in identity; structured
  answer added the conditional commitment gate, plasma-mark access, damage
  concentration, and failure states (numbers disadvantage or spread Q damage).
- **Sylas:** baseline explicitly lacked the requested framing; structured answer
  supplied access versus continuation, a second-rotation/HP-joust model, and a
  decision sequence for commitment.

Gate decision: **supported.** In all four cases the structured answer added the
target causal/state/resource relationships while preserving the same raw
evidence section. This is evidence that structured context reduces query-time
reasoning work for this small manual fixture. It is not evidence of broad
champion coverage, extraction quality, or production calibration.

## Risks and Hold

- The configured Supabase connection is stale (`ENOTFOUND` tenant/user). The
  live Phase 1 evaluation therefore used the existing local SQLite evidence
  store with `VECTOR_BACKEND=sqlite`; the harness remains valid because each
  pair used the identical snapshot.
- Hosted strategic schema/sync have structural tests but have not been executed
  against a live Supabase project.
- The fixture is intentionally small and manually authored. Do not claim
  generalized quality from these four comparisons.
- Add `LLM_PROVIDER=deepseek` to the local `.env` to make DeepSeek default;
  `DEEPSEEK_API_KEY` and `DEEPSEEK_MODEL=deepseek-v4-flash` are already present.

**Current hold:** Phase 1 is complete and the implementation is intentionally
stopped. Do not begin Phase 2 without an explicit user request.

## Testing and Git Policy

For each meaningful behavior change: define invariants and edge cases, add
deterministic tests, keep live LLM tests opt-in, obtain independent review,
fix findings, retest, then make one focused commit and push it to `main`.
Never mix unrelated dirty-worktree changes into those commits.

## Fresh Session Bootstrap

```text
/model 5.6 terra

Read:
1. @docs/compiled-reasoning-implementation-plan.md
2. the Notion page "videoSorter Design - From Naive RAG to Compiled Strategic Reasoning"
3. the current repository state.

Then set /goal to continue this plan exactly as documented.
Do not proceed past a phase gate without satisfying its acceptance criteria.
```
