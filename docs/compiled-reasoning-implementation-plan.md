# Compiled Reasoning Implementation Plan

Last updated: 2026-08-16

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

#### M10: Fallback Schema Repair

**Complete. Commit pending.** Stage B now supplies the exact accepted-relation
JSON contract, requires `evidence_ids` and separate subject/predicate/object
grounding, and forbids alternate `source`/`target` structures. It remains
isolated from raw evidence and supports every allowed entity type. Independent
review approved it. A live Flash fallback rerun removed the missing-evidence-ID
failure mode, but still accepted 0/18 references: one grounded-proposition
failure, 15 unknown entity/concept/relation rejections, two subject-alignment
mismatches, and one unsupported-concept rejection. The fallback is safe but
has not demonstrated useful recall.

#### M11: Causal Eligibility Audit

**Complete. Commit pending.** A deterministic 50-record slice of real
`videos.db` insights was manually classified in
`docs/phase2c-causal-eligibility-sample.md`: 23 explicit A mechanisms, 16
implicit-but-recoverable B mechanisms, 11 advice-only C records, and no D
noise records. A/B eligible share is 78%; C/D safe-zero share is 22%. This
small contiguous three-video audit shows that causal material exists beyond
the benchmark, but is not corpus-representative and does not isolate compiler
failure from ontology coverage, alias coverage, or insight granularity. The
per-record rationale is preserved; a stratified, independently labeled sample
is required before deciding whether upstream extraction needs rework.

### Phase 2C Stop Gate

**Decision: NOT READY — CONTINUE PHASE 2 REPAIR. Do not start Phase 3.**

The hard safety properties are implemented: raw evidence stays separate,
accepted relations retain alignment and provenance, concrete entities remain
strictly grounded, conditions/events survive persistence, and unsupported or
free-form candidates reject. However, the required useful-recall gate failed.
On the 18-reference live set, Flash and Pro one-pass source-aligned extraction
both accepted 0/18 at a shared 1024-token limit. The optional two-stage Flash
fallback also accepted 0/18 after schema repair. Its final trace contains one
grounded-proposition failure, 15 unknown entity/concept/relation rejections,
two subject-alignment mismatches, and one unsupported-concept rejection. The
small audit provides insufficient evidence to blame the compiler alone or to
justify a blanket corpus reparse.

Required next Phase 2 work: replace free-form Stage B generation with a
stronger schema-constrained mapper or deterministic proposal-to-ontology
adapter, then re-run the held-out benchmark with precision/recall and review
bucket metrics. Do not build automated fingerprints on this output.

### Phase 2D: High-Recall Candidate Mapping + Bronze Transcript Recompilation

**Status: Stopped at M13 gate. NOT READY — CONTINUE PHASE 2 REPAIR. Do not start Phase 3.** The 18-positive-reference
Phase 2B fixture is frozen as held-out evaluation. Phase 2D uses a separate
development set and does not tune prompts, aliases, candidate ranking, or
thresholds on the held-out IDs.

#### M12: Bronze Source-Window Resolver

**Complete. Commit pending.** Added a read-only resolver from an insight ID to
the transcript for that insight's own `video_id`. Bronze source text remains
in `videos.transcription`; insight summaries remain in `insights.text`; there
are no retained per-insight timestamps, speaker segments, or source spans in
the existing schema. Resolution therefore prefers externally verified spans
when supplied, then unique exact text, then bounded lexical retrieval within
the same transcript. It never searches a different video.

`SourceWindow` exposes the insight, video ID, character window, method, score,
exact locations, and lexical candidate locations. Only verified explicit spans,
unique exact matches, and unambiguous multi-token lexical matches are marked
resolved. Multiple exact matches, near-tied lexical matches, and caller spans
without verified metadata remain inspectable but cannot become provenance.

Bronze audit: 385 of 494 videos contain transcript text. Of 8,495 insights,
7,755 map to a nonempty local transcript and 740 do not. Historical transcript
ingestion stripped VTT timing and speaker segments, so Phase 2D can recover
character windows but cannot reconstruct timestamps or speakers from the
stored bronze text.

Tests: `uv run python -m unittest tests.test_source_windows` (11 passing).
Independent review found weak single-token lexical matches, unverified span
overclaiming, boolean span bounds, and ambiguous-window status leakage; all
were fixed and re-reviewed approved. The next milestone is a non-overlapping
development fixture and source-mode proposition benchmark.

#### M13: Source-Mode Grounded Proposition Extraction

**Complete. Commit pending.** Added a Phase 2D-only proposition extractor that
reuses the existing cheap-model callable and `GroundedProposition` fields, but
does not select ontology concepts, canonical entities, relation types, or
persist relations. It supports `insight`, `transcript`, and `combined` source
modes. Every non-null proposition field cites an exact source span. Transcript
alignments retain both local window offsets and absolute offsets into the
immutable bronze transcript.

Combined mode is deliberately conservative: all subject, predicate, effect,
and condition fields in one proposition must come from the same source text.
This prevents a model from stitching individually real but causally unrelated
insight and transcript fragments into a new claim. A grounded proposition can
therefore be retained for later provisional mapping without being made a
canonical or trusted relation.

`data/relation_extraction_phase2d_dev_v0.json` is a separate seven-case
development fixture from three non-held-out videos. Five causal cases require
verified transcript windows; two advice-only safe-zero cases remain insight
mode because their transcript locations are ambiguous. It has zero insight-ID
overlap with the frozen Phase 2B held-out fixture.

Tests: `uv run python -m unittest tests.test_proposition_extract
tests.test_relation_extract tests.test_source_windows` (59 passing).
Independent review found three issues: missing absolute transcript offsets,
cross-source causal-field stitching, and an underspecified development source
mode. All were fixed with regressions; a follow-up review additionally caught
cross-source condition stitching, which was fixed and rerun. The next
milestone adds legal canonical candidate generation; no live model run has
occurred during M13 tuning.

#### M14: Auditable Legal Candidate Generation

**Complete. Commit pending.** Added deterministic candidate generation before
any ontology mapper call. It produces scored, reasoned legal IDs for subjects
from complete-token champion/ability aliases; relation types from the existing
closed vocabulary; concepts from curated aliases and ontology-description
overlap; and source-preserving conditions. No candidate generator can create a
new ontology node. Empty candidate sets are retained for grounded but
unmappable propositions.

Safety rules: one-character ability aliases must match complete tokens, generic
capability words (`cannot`, `let`) do not imply a causal direction, and a broad
set of negation/failure constructions suppresses directional relation
candidates. A `missed -> temporarily_unavailable` condition is inferred only
when exactly one known ability alias is present as complete tokens. Candidate
sets include scores and reasons so later evaluation can distinguish missing
candidates from mapper mistakes.

Tests: `uv run python -m unittest tests.test_candidate_generation
tests.test_proposition_extract tests.test_relation_extract
tests.test_source_windows` (66 passing). Independent review found substring
ability-alias poisoning, generic directional verb inference, incomplete
negation handling, and condition-event alias poisoning. All were fixed with
adversarial regressions and the final review approved. The next milestone is
an ID-only mapper; it must select only these candidates or `UNMAPPED`.

**Follow-up correction (commit `9a770c1`):** Object candidates now include
source-grounded ability and champion IDs as well as ontology concepts. The
existing `StrategicRelation` contract allows those object types, so limiting
the mapper to concepts made legitimate ability-to-ability relations impossible
to select. The generator exposes an entity target only when its alias or
champion name occurs in the effect source; it does not introduce an entity
from metadata alone. Independent review verified that the expanded object list
does not affect relation candidates or bypass ledger regeneration.

#### M15: Schema-Constrained ID-Only Mapper

**Complete. Commit pending.** Added an ID-only mapper that receives a grounded
proposition and M14 candidate sets, then returns exactly one of `mapped`,
`unmapped`, or `no_relation`. A mapped response must select subject, relation,
and object IDs from their own candidate lists; any free-form, cross-slot,
unknown, duplicate, missing, or extra field rejects. Conditions use an index
into the generated condition candidates. `unmapped` and `no_relation` must
make no selection and carry `null` confidence.

The mapper does not import or create `StrategicRelation`, and does not persist
anything. This preserves the Phase 2D distinction between a high-recall
proposition/candidate layer and trusted compiled knowledge.

Tests: `uv run python -m unittest tests.test_constrained_mapper
tests.test_candidate_generation tests.test_proposition_extract
tests.test_relation_extract tests.test_source_windows` (74 passing).
Independent review found non-closed/duplicate JSON fields, ambiguous
non-selection confidence, missing wrong-slot and empty-set coverage, and a
prompt/parser confidence contradiction. All were fixed with regressions; the
final review approved. Next: add the candidate relation ledger and aggregation
states without promoting provisional output into existing strategic relations.

#### M16: Provisional Candidate Ledger and Aggregation

**Complete. Commit pending.** Added a lightweight in-memory candidate ledger;
it does not touch the existing `StrategicRelation` persistence path. It retains
`trusted`, `provisional_mapped`, `provisional_unmapped`, `contradicted`,
`rejected`, and inspectable `no_relation` outcomes. All recorded evidence must
be present in an immutable evidence-ID-to-video catalog injected when the
ledger is created.

Mapped selections are revalidated against a deterministically regenerated
candidate set. The ledger owns the approved ability aliases and concept top-k;
candidate metadata must match those values and the exact proposition signature.
This prevents caller-supplied videos, arbitrary selections, forged candidates,
or self-attested aliases from creating trusted graph knowledge. Only mapped
hypotheses with two independent registered source videos and confidence at or
above the configured threshold can be trusted. Repeated model samples from one
video do not create source diversity. Different conditions coexist; only
structurally conflicting relations under the same condition become
`contradicted`.

Tests: `uv run python -m unittest tests.test_candidate_ledger
tests.test_constrained_mapper tests.test_candidate_generation
tests.test_proposition_extract tests.test_relation_extract
tests.test_source_windows` (86 passing). Independent review found and verified
fixes for forged source diversity, direct/free-form mapper selections,
uninspectable no-relation output, confidence/status bypasses, candidate reuse,
candidate-set fabrication, and self-attested alias/top-k policy. Final review
approved. Next: implement the evaluation harness, freeze configuration, and
run development source-mode ablation before the one held-out checkpoint.

#### M17: Evaluation Attribution and Candidate Coverage Metrics

**Complete. Commit:** `04bf11b` (`main`). Added deterministic metrics that
separate subject, predicate, object, and condition candidate coverage from
mapper selection. The report now distinguishes end-to-end mapping success from
mapper accuracy conditional on a full legal candidate set. Conditions score
only when the candidate and selected candidate index match the expected
source-preserved condition, not merely because a condition is non-null.

Failure categories remain closed: candidate-slot misses precede mapper
misselection, while structured output, parsing, provider, timeout, and invalid
selection failures are explicitly classified. This ensures a failed source
window/candidate generator cannot be misreported as an LLM mapper failure.

Tests: `uv run python -m unittest tests.test_phase2d_metrics
tests.test_candidate_generation tests.test_constrained_mapper
tests.test_candidate_ledger` (37 passing before final review). No model calls,
source mutation, persistence, or trusted-relation promotion occur in this
boundary.

**Source-mode evaluator (commit pending):** Added the development-only
`pipeline.phase2d_evaluation` harness. It resolves each case through the
read-only M12 resolver and evaluates identical Stage A extraction in
`insight`, `transcript`, and `combined` modes. An unresolved bronze window is
reported as unavailable and contributes to source coverage, never as a safe
zero or a perfect quality score. Exact source-aligned proposition labels are
matched as multisets, and evaluator-side validation rechecks evidence IDs,
field completeness, span type/bounds/slices, transcript absolute offsets, and
single-source coherence before a mocked or bypassed output can count as a true
positive. Fixture validation prohibits noneligible/safe-zero cases from
containing expected propositions.

Tests: `uv run python -m unittest tests.test_phase2d_evaluation
tests.test_phase2d_metrics tests.test_proposition_extract
tests.test_source_windows` (34 passing). Independent review found and fixed
ungrounded mock matches, unavailable-case perfect metrics, inconsistent
safe-zero labels, fabricated offsets/evidence, and boolean offsets. This is
only the Stage A/source-ablation measurement harness; canonical candidate
coverage and mapper evaluation remain next.

Operator command (dry-run; no relation or ledger persistence):

```bash
LLM_PROVIDER=deepseek uv run python -m scripts.eval_phase2d_propositions \
  --live --variant flash --db videos.db \
  --json-output /tmp/phase2d-dev-flash-source-modes.json
```

The CLI rejects blank explicit models, duplicate source modes, and use of a
default relation variant with a non-DeepSeek backend. Its artifact labels an
explicit `--model` as `custom` rather than misreporting it as Flash or Pro.

**Live configuration repair (commit pending):** The first Stage A Flash
dry-run exposed six `DeepSeek returned empty chat content` failures. This was a
provider-mode wiring issue: the Phase 2D proposition extractor had not passed
the existing `RELATION_EXTRACTION_DEEPSEEK_THINKING=disabled` configuration
used by the Phase 2 relation path. Stage A now forwards that same mode only
when the selected backend is DeepSeek. This changes no prompt, ontology,
validation, source text, persistence path, or confidence threshold; the
pre-repair artifact remains retained as an invalid provider-configuration
baseline rather than a quality score.

**Prompt-contract repair (commit pending):** A raw Flash trace then showed
that Stage A echoed `insight|transcript`, a literal placeholder inadvertently
shown as a grounding source value in the response shape. The packet now lists
the concrete valid source enum for its mode (`["insight"]`, `["transcript"]`,
or both in combined mode). Parsing still rejects any source not present in the
packet and still requires all fields in a proposition to use one source. This
is an unambiguous structured-output contract correction, not an alias,
candidate, validation, ontology, or benchmark-label change.

**Deterministic span repair (commit pending):** Subsequent raw output showed
that Flash was asked to perform both semantic extraction and brittle character
offset bookkeeping. Stage A now requires byte-identical source phrases and a
permitted source label; it computes a unique, token-bounded span
deterministically. It rejects paraphrases, missing phrases, duplicate source
locations, model-supplied offsets, embedded token fragments (including Unicode
apostrophe names), and mixed-source propositions. The accepted provenance is
therefore stricter and more inspectable while removing an unnecessary index
generation task from the model.

**Development scoring repair (commit pending):** The first valid Stage A run
showed that byte-exact field decomposition is too strict for proposition
quality measurement: a model can preserve source spans yet choose a different
actor/predicate/effect segmentation. The development fixture now has manually
reviewed, role-specific semantic token groups. A match still requires valid
source alignment plus subject, predicate, and effect evidence in their own
roles; conditioned references additionally require condition evidence and the
same leading operator. Exact decomposition recall remains a separately
reported stricter metric. The loader verifies no insight ID or source video ID
overlaps the declared frozen Phase 2B fixture. This applies only to the
non-overlapping development fixture and does not alter prompts, aliases,
candidate generation, or held-out labels.

**Stage A instruction repair (commit pending):** Bronze-window inspection
showed explicit action-plus-effect mechanisms, but Flash returned zero because
the extractor did not explicitly recognize coaching constructions such as
"use X to remove Y". The generic Stage A contract now permits such a
proposition only when the supplied text itself contains both the action or
resource and the stated strategic effect. It still requires exact source
quotes, retains zero output for advice without a mechanism, and introduces no
ontology term, champion fact, or benchmark-specific example.

#### Phase 2D M13 Stop Gate: Stage A Did Not Reach Useful Recall

**Stopped after development-only validation.** The source resolver itself is
working for the five positive development cases: all resolve to their own
bronze transcript window, and manual inspection confirms that those windows
contain the labeled mechanisms. The configured Flash model nevertheless
achieved **0/5 semantic proposition recall** and **0 exact decomposition
recall** in transcript-only evaluation after all source-contract repairs.
It returned four safe zeros and one source-grounded but causally misstructured
proposition. The combined and insight modes also scored 0 recall; their
nonzero outputs were source-grounded spans but did not retain the reviewed
actor/predicate/effect/condition structure. This is a Stage A failure, not
evidence that conditions, provenance, or the trusted relation validator should
be weakened.

Retained local dry-run artifacts:

- `/tmp/phase2d-dev-flash-source-modes-scored.json`
  (`sha256:515a151d15fbba3c3122695f0258ce9dc40c12a373c4149749eeaafd0f0f7f82`)
- `/tmp/phase2d-dev-flash-transcript-coaching-repair.json`
  (`sha256:a11c8590b6a1dcd6f6544974a40c609c1ec0c40bdfc7eaf42c42e63ec989e740`)

The first provider run before non-thinking forwarding is retained as an invalid
configuration baseline: V4 Flash returned empty final content. It must not be
treated as a quality result. V4 Pro was intentionally not tested: Phase 2D's
controlled-comparison rule defers Pro until Stage A works well enough to make
candidate and mapper measurements meaningful. The frozen Phase 2B held-out
fixture was not used for prompt, alias, threshold, or candidate tuning, and no
candidate mapping, ledger promotion, persistence, or Phase 3 fingerprint work
was performed from this failed Stage A output.

Next justified Phase 2 work: redesign or decompose grounded proposition
extraction so that it can reliably recover actor, causal direction, effect,
and condition from the same verified window before evaluating candidate sets,
mapper selection, or any stronger model. Do not progress to M14-M17 as though
the current Stage A output were useful.

### Phase 2E: Span-First / Clause-First Stage A Restructure

**Closed as an architecture failure.** The five eligible development cases
carry conservative, reviewed labels. The valid clause-first v2 run establishes
that deterministic source coverage succeeded while model mechanism-clause
selection failed. Do not continue tuning proposition-first or clause-first
Stage A.

#### Architecture: Span-First 7-Call Stage A

Stage A is restructured from one difficult weak-model semantic-generation act
into a sequence of seven narrow, observable, source-grounded model calls
(`prompt_version` `phase2e-span-first-v1`,
`pipeline.proposition_extract.extract_span_first_propositions`):

1. evidence localization: select the smallest exact source span(s) and one
   permitted source label;
2. actor slot; 3. event slot; 4. effect slot; 5. condition slot (explicit
   `NONE` when absent);
6. causal direction classification; 7. ontology normalization.

Deterministic code derives unique token-bounded source spans and assembles the
final proposition; the model never supplies character offsets. A failure in any
stage returns no proposition for that case rather than inventing one.

#### Provenance, Failure Taxonomy, and Artifacts

- Every stage retains a `StageArtifact` (stage, raw provider output, parsed
  output, failure class); `StageAExtraction` carries frames, slots, evidence
  spans, `unsupported_slot_count`, and the first `failure_stage`.
- The failure taxonomy includes `ProviderCallError` plus per-stage structured
  output parse/validation failures; ungrounded or fabricated frames score no
  evidence, slot, or semantic hit.
- Run artifacts are JSON files with deterministic `content_sha256` and full
  model/provider/configuration metadata (backend, model, variant, thinking,
  max_tokens, prompt version, fixture, held-out fixture, db, live).

#### Slot-Level Evaluator and Mandatory Held-Out Separation

`pipeline.phase2d_evaluation` measures slot-level recall (actor, event, effect,
condition, causal direction, normalization, evidence span, semantic
proposition, assembled proposition, exact decomposition) with X/5 denominators
that count every source-available eligible case: an unreached stage is a miss,
never a denominator exclusion. Causal direction expectations are derived from
the reviewed subject/predicate/effect role labels. The five development
normalization labels are evaluation-only, require an exact closed-ontology
triple match, preserve explicit nulls where the source does not directly
support a canonical mapping, and include a review rationale. Held-out
separation is mandatory: every development fixture, including arbitrary or
metadata-less files, is compared against the frozen Phase 2B fixture resolved
from a trusted explicit path, and an unavailable frozen fixture is an error.
Normalization-stage reached/completed/abstained/mapped/failed counts remain
available alongside recall.

#### Tests

Broader current evidence: `.venv/bin/python -m pytest tests
--ignore=tests/test_auth.py -q` passes **419 tests with 128 subtests**
(`419 passed, 128 subtests passed in 20.10s`). Focused Phase 2 evidence
retained: `unittest` across the Phase 2 evaluation modules
(`test_phase2d_evaluation`, `test_proposition_extract`,
`test_eval_phase2d_propositions_cli`, `test_phase2d_metrics`,
`test_candidate_ledger`, `test_constrained_mapper`,
`test_candidate_generation`, `test_relation_extract`,
`test_source_windows`): **237 tests passing** (`Ran 237 tests ... OK`). New
coverage includes evidence-span provenance, condition vs no-condition, causal
direction, actor/effect reversal, multiple-digit/source-span offsets,
malformed/partial model output, safe `NONE`, deterministic proposition
assembly, source spans containing multiple possible actors/events, and refusal
to construct a proposition when required evidence is absent. It also covers
fail-closed normalization-label validation, exact reviewed normalization
scoring, and first-loss attribution at ontology normalization.

#### Independent Review Closure

Independent review of the Phase 2E deterministic boundary is closed on four
points: (1) semantic-slot phrase uniqueness is selected-evidence-local rather
than packet-wide, preserving exact offsets and failing on multiple selected
occurrences; (2) the evaluator defensively requires grounded, slot-consistent,
direction-consistent, deterministic frame assembly before granting final
assembled/exact credit, while preserving per-slot diagnostics; (3) eligible
development cases enforce exactly one expected proposition, so the official
gate stays X/5; (4) the frozen held-out fixture schema validates fail-closed
before overlap checks run.

#### Valid Live Run: Phase 2E Gate Failed

Clean Flash transcript-only run:

```bash
LLM_PROVIDER=deepseek .venv/bin/python -m scripts.eval_phase2d_propositions \
  --live --variant flash --db videos.db --mode transcript \
  --json-output /tmp/phase2e-dev-flash-transcript-span-first-network-final.json
```

Artifact inner `content_sha256`
`e3a769b61dc4699d6e65bdc5572eb86e1741832abe1c84839a68e98564f55017`
(file SHA-256
`6527419b5b905964e42e6c4cbebc9b6a200ce03710e4f6d2f57161a5c1035fd8`)
is a **valid model-quality result**: 7 cases, 5 eligible, 2 unavailable safe-zero
cases, full eligible source coverage, and successful provider output.

The result is **Phase 2E FAIL — CONTINUE STAGE A DECOMPOSITION**:

- semantic proposition recall: **0/5**;
- exact decomposition recall: **0/5**;
- evidence-span recall: **2/5**;
- causal-direction recall: **1/5**;
- actor, event, effect, condition, and normalization recall: **0/5** each;
- two cases completed all seven calls, while three stopped at evidence or actor
  validation; unsupported proposition rate was **1.0**.

The trace localizes the earliest architectural loss to clause selection and
role assignment. Flash often selected a broad or adjacent coaching clause, then
treated the grammatical `you` or the first nearby action as the causal actor.
Examples include selecting `you portal away` instead of the sweeper mechanism,
and returning `you` as the actor for the push/poke and hook-risk cases. This is
not a normalization-only problem and does not justify aliases, relaxed
grounding, Pro, candidate mapping, or held-out evaluation.

Per the preregistered 0-2/5 rule, the next Stage A experiment must move lower:
deterministically enumerate source-local clause candidates, have Flash choose
the smallest mechanism-bearing clause or linked clause pair from IDs, and only
then extract actor/event/effect/condition within that selected boundary. Keep
the same five development cases, metrics, exact-span provenance, and frozen
held-out isolation.

#### Clause-First v2 Implementation and Deterministic Boundary

That lower-level architecture is now implemented under prompt version
`phase2e-clause-first-v2`. Deterministic code enumerates stable source-local
candidate IDs from sentence/discourse boundaries and overlapping 32-token
windows for punctuation-poor regions, bounded to 20 candidates per source.
Flash may select only one or two IDs from one source; code derives all offsets,
coalesces only overlap or whitespace adjacency, and preserves real gaps. The
later source-slot, direction, normalization, and deterministic-assembly stages
remain unchanged. Candidate catalogs survive success, abstention, malformed
output, and provider failure in both core and evaluator artifacts.

The evaluator now reports candidate-catalog coverage independently of model
selection. It applies the exact production one/two-ID and same-source contract,
rejects duplicate IDs or ungrounded alignments, runs production coalescing, and
checks that all reviewed exact source fields fit within the resulting spans.
On the unchanged five eligible transcript cases this deterministic boundary is
**5/5 complete**: wave reset and push/poke each need one candidate; sweeper,
mid-push, and hook-risk are coverable by two candidates. This establishes that
v2 candidate generation itself has not discarded any reviewed mechanism. It
does not grant semantic model credit.

The retained v2 retry artifact
`/tmp/phase2e-clause-first-v2-catalog-network-retry.json` has inner
`content_sha256`
`ecfce104b38fe57e3f8db767057254177a150b93a377b05ab1f12781fe61f0b2`
and file SHA-256
`7e1a4cbb62bf638ff3d83b2106ab7a8682837f2f2ef00bbab1afa89ff1b7b957`.
It records complete 5/5 catalog coverage, but all five model calls failed at
evidence localization with `ProviderCallError` before raw output because the
execution environment had no outbound network path. It is therefore an
infrastructure-failure artifact, **not** a valid v2 0/5 model-quality result.
That provider-failure artifact remains historical infrastructure evidence. It
was later followed by the valid model-quality artifact
`/tmp/phase2e-clause-first-v2-valid-run1.json` with canonical inner hash
`04c185aaf324251b4733e76c87b2c71ea3946497f79a8956f268e88f28e2e17b`
and file SHA-256
`02725fb163ef752c98f51a070652ef5418a5b0d4916363d1c61c3071e957c808`.
Its windows, catalogs, and reviewed expectations were regenerated exactly from
commit `1b3063edd84237c32a391564e461416ec992c308`, the configured development
fixture, and the configured database during Phase 2F reconciliation. The
artifact itself does not embed those revision/input hashes, so future Phase 2F
artifacts must do so.

The valid result is semantic proposition **0/5**, exact decomposition **0/5**,
all required semantic slots **0/5**, candidate-catalog coverage **5/5**, and
zero unsupported/invented slots. It contains five raw provider outputs, zero
completed eligible cases, five evidence-localization `ValueError` failures,
and two unavailable ambiguous-lexical cases; downstream semantic stages never
ran. All five outputs omitted the required `transcript:` prefix, so official
parse-valid clause selection is **0/5**. Diagnostic prefix canonicalization is
used only to locate the loss boundary: wave reset selected the reviewed
mechanism; push/poke selected it plus irrelevant `c008`; sweeper, mid push, and
hook risk selected the wrong clause pairs. Thus reviewed-mechanism containment
is **2/5**, while exact/minimal selection is at most **1/5**. Fixing the parser
cannot turn this into a viable architecture.

Authoritative Phase 2E conclusion:

```text
deterministic candidate generation: 5/5
    -> diagnostic mechanism containment: 2/5 (exact/minimal <= 1/5)
    -> proposition extraction: unusable
```

Candidate generation succeeded, clause selection failed, and the first loss
boundary is known. Official semantic recall is **0/5**, so per the
preregistered 0–2/5 semantic-recall rule Phase 2E is **ARCHITECTURE STILL
FAILING**. No Pro, frozen held-out quality run, downstream mapping, ledger
promotion, persistence, or Phase 3 work was performed.

#### Publication Status

The v1 code boundary and valid failure record were published directly to
GitHub `main` through `2f47f8f`. The focused v1 boundary is:
(a) core extraction + tests,
`5bcffcf` ("Add span-first semantic proposition extraction"); (b) evaluator,
CLI, tests, and handoff, `da1ad9f` ("Add Phase 2E semantic evaluation and
handoff"); and (c) reviewed normalization labels, scoring, tests, and final
documentation, `b63d5e1` ("Add reviewed Phase 2E normalization scoring"); and
(d) the valid v1 gate record, `2f47f8f` ("Record Phase 2E Flash gate result").

Clause-first v2 was published to GitHub `main` as `f1f38c4` (candidate
enumeration/selection), `6624221` (artifact retention), `fc80ade`
(candidate-catalog coverage diagnostics), and `1b3063e` (the pre-run validation
boundary). The valid artifact above closes the experiment without rewriting
the earlier provider-failure history.

### Phase 2F: Ground-Up Source Semantic Compiler

**Authorized and in progress. Stop after Pass 1 representation proof.**

Architectural source of truth: Notion page **Ground-Up Semantic Compiler
Architecture — Source-Preserving Bronze → Strategic Knowledge**. Phase 2F
implements only:

```text
bronze
  -> Pass 0 deterministic segmentation/context
  -> source-anchored semantic mentions
  -> general semantic relations between mentions
  -> proof-carrying source-semantic IR graph
```

It does not implement canonical source claims, League ontology normalization,
strategic relations, motifs, fingerprints, production graph persistence,
corpus backfill, Phase 3, or Flash optimization.

#### Representation invariants

- Bronze text is immutable and round-trippable from exact deterministic spans.
- Model output selects stable source-local IDs; arbitrary model offsets are
  never trusted.
- Node types are limited initially to `ENTITY`, `ABILITY_OR_RESOURCE`, `EVENT`,
  `ACTION`, `STATE`, `OUTCOME`, `QUANTITY`, `TIME`, and `LOCATION_OR_SPACE`.
- General edges begin with roles, causal/enable/prevent/require relations,
  condition/purpose/result, temporal relations/termination, contrast,
  negation/modification, and reference.
- Conditions, time, negation, modality, uncertainty, and unresolved references
  remain first-class rather than being flattened into proposition strings.
- Every accepted node and edge carries exact source and model/configuration
  provenance. `UNKNOWN`, `AMBIGUOUS`, `INSUFFICIENT_EVIDENCE`, and no-relation
  are valid outputs.
- No Pass 1 object may contain hidden strategic concepts such as `access`,
  `continuity`, `tempo`, `initiative`, or `wave_obligation`.

#### Benchmark isolation and preregistered gate

- Keep the five Phase 2E cases as a legacy failure regression set.
- Keep `data/relation_extraction_phase2b_v0.json` frozen and unchanged.
- Build a 200–500-window representative pool, then a separately reviewed
  30–50-window benchmark with non-overlapping `DEV` and `FROZEN_EVAL` metadata.
- Fail closed when source-span overlap cannot be verified. Tune only on DEV and
  run the frozen strong-model evaluation once.
- Report deterministic mention-catalog coverage and candidate-edge-pair
  coverage separately from model selection/classification.

The frozen thresholds must be finalized before its run. Initial preregistered
hard-safety thresholds are: accepted node source anchoring **100%**, accepted
edge/node/evidence provenance **100%**, fabricated offsets **0**, and hidden
ontology normalization **0**. Initial semantic thresholds are DEV checksum
**>= 0.90**, frozen checksum **>= 0.85**, unsupported node/edge invention
**<= 0.05**, and no critical entity/event/condition/causal dimension below
**0.80**. Any revised exact threshold must be justified from DEV evidence and
committed before frozen labels/results are inspected.

##### Legacy five-case strong-reference gate — preregistered before first live run

The first live Phase 2F decision uses the reviewed five-case regression fixture
`data/semantic_ir_legacy_failure_v1.json`, locked at content SHA-256
`a17674b6e2c491f0d7a1600dde0cfb8cc533d1d17db8633d8d94b2de9a57c1dd`.
Its exact-source manifest is locked at
`cf86dde955f4cbeee091f38aab8293256b0c48f809c969384185a330ee511241`.
The manifest is reproducible from the primary `videos.db`, the immutable Phase
2D development fixture, and the valid Phase 2E artifact; the latter retains
the verified inner/file hashes recorded above. The five cases are development
regressions, not the later frozen representation set.

The reference compiler configuration is DeepSeek `deepseek-v4-pro` through the
existing provider abstraction and official `https://api.deepseek.com` endpoint,
with provider thinking explicitly `disabled`,
temperature `0`, mention partitions of 600 candidates, and retained output
limits of 2048 mention / 512 qualifier / 256 coreference / 256 edge tokens.
Coreference is limited to two Pass 0 segments; edge pairs are limited to 600
characters and two Pass 0 segments. Entity hints come from the hash-locked
representative pool's deterministic champion catalog. Ability/resource hints
are the general source vocabulary `Q`, `W`, `E`, `R`, `ult`, `ultimate`,
`Flash`, `Teleport`, `Ignite`, `Exhaust`, `Ward`, and `Sweeper`. The runner must
verify a clean, committed worktree before the first provider call so the
retained revision binds the executed code. These hints do not normalize or
type model output.

The gate is intentionally strict because every reviewed fact was selected as
necessary to reconstruct the five mechanisms:

- accepted node source anchoring and edge provenance traceability: 5/5 cases;
- fabricated offsets: zero; hidden strategic/domain normalization: zero;
- deterministic mention candidate coverage, qualifier-cue coverage, and
  endpoint-reached edge-pair coverage: 100%;
- mention selection/type, reviewed qualifier, reviewed reference, and reviewed
  edge recall: 100%;
- semantic checksum: 100% in every case and in aggregate;
- provider failures, model parse failures, and assembly failures: zero.

The fixture is deliberately non-exhaustive, so unscored extra nodes/edges are
reported but cannot be called unsupported invention. The broader exhaustive
DEV/FROZEN benchmark owns invention-rate acceptance. A failed legacy run may
drive a general Pass 1 repair and another development run, but Phase 2F must
not advance to broad annotation/evaluation while any of the five reviewed
mechanisms remains semantically incomplete. Thresholds above do not change
after observing live output.

#### Phase 2F milestone record

**Milestone 0 — complete (`cacb15d`).** Hypothesis: the valid clause-first v2
artifact can close Phase 2E without erasing earlier negative evidence. The
handoff and this plan now retain the official 0/5 parse-valid/semantic result,
the diagnostic-only 2/5 reviewed-mechanism containment (exact/minimal <= 1/5),
the verified inner/file hashes, and the earlier v1/provider-failure artifacts.
Independent artifact review regenerated all seven windows and five catalogs at
`1b3063e` and confirmed the first loss as deterministic catalog 5/5 -> Flash
clause selection failure -> unusable proposition extraction. No unresolved
Phase 2E quality run remains, so the clean-room boundary is justified.

**Milestone 1 — complete.** Hypothesis: stable source/provider/test utilities
can be reused without inheriting proposition assumptions. Repository mapping
kept exact source loading, offset handling, provider injection, failure
retention, hashing, and test utilities; it explicitly rejected
`SourceSemanticFrame`, `GroundedProposition`, fixed slot scoring, normalization,
and proposition ledgers as Pass 1 IR. New work is isolated in
`pipeline/semantic_source.py`, `pipeline/semantic_ir.py`, and later sibling
modules. Production retrieval and persistence remain unchanged.

**Milestone 2 — complete (`b3c9dc6`).** Hypothesis: a general graph can express
bronze meaning without a mandatory proposition tuple. The typed schema retains
the nine source node families, the preregistered general edge vocabulary,
exact local/absolute spans, source context, confidence, ambiguity, typed
field-specific qualifier cues, unresolved exact referent candidates, and
model/config digests. Graph validation binds the supported Pass 0 version and
provenance-derived window ID, enforces directed endpoint signatures, and
requires one exact evidence span jointly covering both endpoints. Tests also
cover multiple effects, time/condition expressiveness, strict JSON, prohibited
domain units, malformed offsets, and artifact round trips. Independent review
found and drove fixes for arbitrary absolute offsets, ungrounded qualifiers,
recursive/entity-only coreference, irrelevant edge proof, and inconsistent
Pass 0 identity; its final verdict found no schema blocker. Raw failed attempts
and independently recomputable model digests remain assigned to the later run
artifact boundary, not silently claimed by the graph alone.

**Milestone 3 — complete (`cf32d40`).** Hypothesis: Pass 0 can be entirely
deterministic while retaining exact contextual bronze truth. The implementation
uses provenance-bound window IDs, exact source/local offsets, versioned
sentence/discourse hints, bounded 32-word punctuation-poor fallback segments,
canonical context/provenance hashes, and exact deterministic regeneration in
validation. Twenty focused tests cover reconstruction, malformed inputs,
timestamps, source prefixes, sparse punctuation, closing delimiters, leading
whitespace, multiple windows, runtime type attacks, algorithm/config stability,
and contextual identity. Independent review plus 1,000 randomized windows
found no remaining source-truth or stability defect after fixes. There is no
model or development evaluation in Pass 0; first loss is therefore not in this
boundary, and mention-catalog work remains justified.

**Milestone 4 — complete (`614e110`, with later catalog-version hardening).**
Hypothesis: a broad source-local catalog can guarantee
reviewed mention availability before model selection is judged. The catalog
uses versioned, span-hashed IDs and exhaustive within-segment spans up to the
Pass 0 bound, including repeated, overlapping, Unicode, percent, possessive,
alias, negation, and temporal forms. Required coverage buckets are emitted even
at zero denominator and require a validated window. Partitioned selection now
retains complete catalogs, raw/parsed decisions, semantic abstentions,
failures, and the effective request; exact node assembly revalidates every
partition and never trusts model offsets. Independent review drove fixes for
hidden free-form qualifiers, cross-window candidates, swallowed partition
abstentions, contradictory assembly provenance, incomplete request identity,
and optional metric validation. Pass 0 segments remain provenance/context
hints rather than semantic-loss boundaries: the current catalog also includes
bounded cross-segment spans, with every intersecting segment retained. The
reviewed legacy fixture has deterministic mention coverage 33/33. The known
risk is quadratic overlapping-candidate volume; the live runner reports the
complete catalogs and uses larger, fixed strong-model partitions without
discarding candidates. Catalog size, redundant selections, and cost remain
explicit DEV measurements rather than hidden success assumptions.

**Milestones 5–7 — implementation complete; live quality pending.** Constrained
mention selection retains every partition and effective request, while node
assembly resolves only offered IDs to deterministic spans. Directed edge-pair
generation retains full discourse evidence, configuration-bound distance
pruning, compatible general-semantic signatures, and a complete catalog.
Pairwise classification supports abstention/no-relation and rejects unsupported
or contradictory labels. Independent reviews drove repairs for swallowed
partition abstentions, contradictory raw/parsed selections, cue loss, forged
pair fields, incomplete catalogs, false request provenance, coreference guesses,
and failure taxonomy. The repaired legacy fixture has endpoint/type-reachable
candidate edge coverage 24/24; strong-model selection/classification quality is
still unmeasured until the preregistered live gate.

**Milestones 8–9 — implementation complete and independently reviewed.** Typed
qualifiers preserve polarity, modality, temporal scope, conditionality,
comparison, uncertainty, and focus/restriction using exact cue spans. Dedicated
coreference preserves zero-target unresolved references, ambiguity candidates,
and proof-carrying `REFERS_TO` only after supported resolution. Graph assembly
requires source-local nodes, exact evidence, resolved-reference edges, retained
decision provenance, and relational condition/temporal structure rather than
flattened strings. The orchestration run is reconstructively sealed across
mentions, qualifiers, coreference, edges, failures, and safe partial outcomes.
Independent adversarial reviews found and closed application-result loss,
provider/model conflation, mutable configuration, suffix smuggling, incomplete
decision prefixes, invented reference resolution, and unbound terminal failure
codes.

**Milestone 10 — complete and independently reviewed.** The evaluator scores
candidate coverage separately from selection, type, edge-pair enumeration,
edge classification, qualifiers, coreference, provenance, unsupported
invention, dimensions, and semantic checksum. Gold facts map one-to-one to
questions; duplicate/intersecting gold, split overlap, double-credited edges,
zero-denominator hiding, failures excluded from metrics, and wrong first-loss
ordering fail closed. Run artifacts reconstruct typed compiler state, bind exact
source/window/input hashes, retain raw outputs and provider/config identity, and
reject inner/outer reseal attacks.

**Milestone 11 — reviewed fixture and preregistration complete; live run
blocked by provider connectivity, not semantically evaluated.** The five Phase
2E failures are reconstructed from the verified
artifact and primary bronze database. The current fixture contains 33 mentions,
24 edges, 10 qualifiers, 8 explicit unresolved-reference judgments, and 75
one-fact semantic-checksum questions. Independent gold review corrected a
polarity-reversing sweeper annotation, an invented hook relation, lost
conjunction, missing actor/reference facts, and brittle types before any model
output was observed. The strict gate and strong-model configuration are locked
above.

The first clean committed attempt ran at revision
`a0feefd50013722c943976a9131eb545f364178c` on
`2026-08-17T00:17:39.171617Z`. All 30 mention-partition calls failed before
returning model bytes as `MentionProviderError:URLError`; a separate read-only
probe localized this to DNS (`gaierror: Temporary failure in name resolution`)
for the official DeepSeek endpoint in the execution environment. The artifact
therefore is **provider-failure evidence, not a strong-model quality result**.
Deterministic mention coverage remained 33/33 and qualifier-cue coverage 10/10;
mention selection, endpoint-reached pair enumeration, and all downstream
semantic checks were not reached. Every case's chronological first loss is
`PROVIDER_FAILURE`.

The reconstructible run is retained as
`data/phase2f_artifacts/phase2f-legacy-pro-run1.tar.gz`, deterministic archive
SHA-256 `e6c2122a2b91c2b70d9775f2c108c26c82cdfff2f5cea9b3c5f60dbbc4146330`.
Its aggregate inner/file hashes are
`80be66cc48b9f2e7685a3da2effa3e190cc1cd8cae5aa9c392a6ba1e693c782a` /
`68e85a6d9b69265ffe8793b27f0c8a44235857be4b8134a1670c3d48c2d4fe1c`.
The strict gate correctly failed and the CLI returned status 2. Do not count
this attempt as the once-only semantic reference run; retry the unchanged
committed configuration only when the official provider is reachable.

**Milestone 11 — first valid strong-model development result: FAILED at Pass
1A; general repair authorized.** After network access was enabled, the locked
five-case command completed against `https://api.deepseek.com` with
`deepseek-v4-pro`, thinking disabled, from clean revision
`b5317c6bd90572e052ab85f399e339c4de83a4e8`. This was a valid legacy-development
run on the locked `LEGACY_FAILURE` split, not the once-only frozen run and not a
provider failure. Its aggregate inner
and file hashes are
`b0a030765217f2dcb52634d31eec171b307541308012945f87864cf7d5697492` and
`ad3801a9fc23a23837fe0ad078273a2744fb9640bbd826172d359af4654cf547`.

Preregistered results:

- deterministic exact mention candidate coverage: 33/33;
- exact reviewed mention selection and compatible typing: 0/33;
- qualifier recall: 0/10;
- reviewed edge-pair/edge recall: 0/24 (endpoints were not recovered, so this
  is upstream-cascaded and does not evaluate pair pruning/classification);
- unresolved-reference recovery: 0/8;
- semantic completeness/checksum: 0/75, with every case at zero;
- provider failures: 0 across 30 mention, 80 qualifier, and 1,839 edge calls.

Official first loss is mention-stage `MODEL_PARSE_FAILURE`, but the semantic
diagnostic localizes the architecture failure independently of parsing. Only
7/30 mention responses were strict JSON, 12 more were complete JSON inside a
single Markdown fence, and 11 truncated. Safely unwrapping only complete fences
raises parseable responses to 19/30 but reviewed exact selection remains 0/33.
Only 1/33 reviewed candidate IDs occurs anywhere in all retained raw output.
The old interface exposed 3,248–3,344 overlapping n-grams per roughly
500-character window in six flat 600-candidate slices, repeated the full source
while asking each partial slice to recover every mention, omitted offsets, and
showed incomplete heuristic type hints. The model selected long clause-sized
proxy spans (111-character median across emitted IDs) instead of atomic
mentions. This reproduces the prohibited selection bottleneck one level lower.

The immutable negative evidence is archived at
`data/phase2f_artifacts/phase2f-legacy-pro-run2.tar.gz`, deterministic archive
SHA-256 `b17cde9d7dc909c317aac81be08e9ed4860f91231d5568aeb6ee515a1fd67183`.
The archive reconstructs all five typed artifacts and is regression-tested.
Preserve the earlier DNS/provider archive separately.

Hypothesis for the single allowed general DEV repair: a strong model can select
atomic exact mentions when each request asks only about a bounded set of exact
source-start anchors, while the exhaustive catalog remains the independent
coverage oracle. Invariant: every catalog candidate appears exactly once; all
end alternatives for one start remain together; a request contains exactly
one start so every abstention is attributable; candidates expose exact offsets
and compact request-local aliases;
model-visible heuristic type hints are absent; source text/offsets remain
deterministically resolved; `NONE`, ambiguity, and nested same-start mentions
remain possible. Exactly one complete JSON fence may be canonicalized while raw
bytes remain retained; this transport repair is not counted as semantic repair.

Do not build the 30–50 reviewed DEV/FROZEN subsets, inspect frozen labels, or
run frozen evaluation until the repaired interface passes all five locked
legacy mechanisms. After deterministic tests and independent diff review, rerun
only the legacy development gate. Thresholds, gold, and denominators remain
unchanged.

Implementation checkpoint before the rerun: `pipeline/semantic_mentions.py`
now uses one complete source-start group per model request, with no change to
the exhaustive candidate generator; `pipeline/semantic_compiler.py` binds the
new orchestration/prompt versions and regenerates the retained partition layout.
Across the locked cases this yields 117–120 requests/window, at most 32 exact
end alternatives and fewer than 7,000 prompt characters per request. This is a
deliberate representation-proof/cost tradeoff; batching is future cheapification.
The historical v1 selection/orchestration path remains deserializable only so
both negative archives continue to reconstruct.

Deterministic tests prove all 33 reviewed spans occur in exactly one focal
request, repeated phrases remain offset-distinguishable, aliases resolve only
to offered stable IDs, nested same-start mentions survive, type hints are not
model-visible, exact single-fence handling is versioned, and partition/prompt
tampering fails reconstruction. Independent diff review found and drove fixes
for three issues: mention-boundary ambiguity had been incorrectly overloaded
onto coreference `MULTIPLE_CANDIDATES`; legacy compiler versions were not bound
to legacy prompt versions; and a multi-focus response could not attribute a
mixed abstention. The final one-focus design closes all three. Reviewer suites
passed 56 focused and 148 semantic tests plus a 100-window randomized partition
audit. Although structural review justified a staged legacy-development probe,
no completed repaired-run artifact exists. A subsequently started process was
interrupted during its first case and the atomic runner published nothing; that
process note is not representation evidence. The repair therefore remains
semantically unproved and broader DEV/FROZEN work is stopped.

**Milestone 12 — representative pool complete; reviewed subsets stopped.** A
deterministic, source-exact pool of 300 windows from 300 distinct coaching
videos covers all 25 declared routing phenomena at least eight times, with zero
Phase 2B/2D source overlap. Its content hash is
`9b89c6d6c6c8070eba48d6db47254e156c1b2591c1480a60f98a1e8d789491c2`.
Independent external verification rebuilt exact equality from `videos.db` and
the exclusion fixtures and rejected source, policy, phenomenon, hash, and
exclusion tampering. Phenomenon tags are deterministic routing aids, not gold.
Manual 30–50-window DEV/FROZEN annotation begins only after the five-case gate.

**Milestones 13–17 — stopped by the legacy gate.** The valid strong-model
legacy-development result did not pass Milestone 11, so the representative pool
was not manually annotated into formal DEV/FROZEN gold, frozen labels were not
inspected, and the once-only frozen evaluation was not run. The reviewed focal
repair remains structurally valid but semantically unevaluated and therefore
cannot count as representation evidence. See
`docs/phase2f-semantic-ir-stop-gate.md` for the complete evidence, first-loss
analysis, milestone report, and answers to all final questions.

Final Phase 2F recommendation:

```text
SEMANTIC IR NOT VIABLE — REDESIGN PASS 1
```

The final Phase 2F recommendation must be exactly one of:

```text
SEMANTIC IR VIABLE — READY TO DESIGN PASS 2
SEMANTIC IR VIABLE WITH SPECIFIC LIMITATIONS — REPAIR BEFORE PASS 2
SEMANTIC IR NOT VIABLE — REDESIGN PASS 1
```

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
