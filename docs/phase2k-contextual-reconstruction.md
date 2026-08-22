# Phase 2K — Contextual Reconstruction

## Status and scope

Phase 2K is an isolated upstream core that freezes the Phase 2J reviewed
windows and produces human-review artifacts. It is not final downstream
extraction: no matchup answers, strategy ontology, LightGBM tuning, or
candidate-generator redesign is emitted or run by the reconstruction
builder. The Phase 2F generative and Phase 2H discriminative reruns are not
executed by the builder, but they are required after human review passes:
the mandatory post-review, pre-rerun alignment gate below prepares the exact
raw/polished targets those reruns consume. Phase 2K never edits Phase 2J
code/data, never corrects Phase 2J scoring, and never overwrites Bronze.

Implementation status as of 2026-08-19: the reconstruction, adaptive-context,
A/B/C/D review, radius-audit, downstream-alignment, comparison-v2, and
gate-locked paired-rerun tooling are implemented. The paired-rerun repair
suite passes 87 tests plus 98 subtests, including replay and resealed-tamper
coverage. The Phase 2K v16 champion-name lexical normalization upgrade
adds the v2 lexical vocabulary (direct/guarded/metadata-licensed/never
champion alias rules) with deterministic exact word-boundary hint
detection, mandatory explicit `DOMAIN_SPELLING` repairs, and the v4
mechanical prompts; the focused reconstruction/build-CLI/contract suites
pass 168 tests plus 54 subtests. This is engineering evidence only: no fresh
authoritative live 30-window reconstruction, completed human review, paired
model rerun, or empirical Phase 2K diagnosis is claimed yet.

The frozen authoritative inputs are:

- `data/phase2j/window-selection-manifest-v1.json`
- `data/phase2j/reviewed-endpoint-annotation-packet-v1.json`
- `docs/phase2j-independent-source-replication.md`
- the read-only SQLite transcript DB supplied by CLI, normally
  `/home/bphan944/PersonalProjects/videoSorter-homework-archive/videos.db`.

All 30 selected windows round-trip their full transcript SHA-256 and exact
upstream offsets.

## Pass separation

Pass 1 is **text restoration only**, not semantic extraction. The provider
envelope is deliberately compact: it carries only the fields that require
judgment, exactly:

- `schema_version`
- `clean_text` — the exact restored target text
- `repairs` — ordered proposals with quoted `original_text`/`replacement`,
  closed repair type, confidence, and rationale
- `uncertainties` — at most 8 proposals with exact quoted `surface_text`,
  alternatives, closed uncertainty type, and a note
- `rationale` — one short paragraph

The provider never emits provenance, hashes, metadata copies, offsets, span
coordinates, IDs, or evidence fields. The harness treats every model-supplied
coordinate and deterministic lineage field as untrusted and seals them
itself. The stored/normalized mechanical object then carries exactly the
allowed envelope: `clean_text`, explicit normalized repairs, explicit
normalized uncertainties, and full deterministic provenance (plus
`schema_version` as the envelope discriminator). No semantic output exists
at this stage.

Structural semantic-extraction keys (`entities`, `events`, `relations`,
`claims`, `bindings`, `champion`, `ability_owner`, and similar) are
rejected recursively anywhere in the provider response, fail-closed.
Mechanical repair types never include entity, pronoun, champion,
ability-ownership, or referent resolution.  Rationale text may contain
lexical definitions (for example, explaining that `pryo` is a common
mishearing of `prio`, i.e. priority, a standard League term), but a
semantic endpoint/extraction list disguised as rationale (labels such as
`entities:`, `events:`, `bindings:`, or phrases such as `extracted
entities`) is rejected with a narrow deterministic guard.

### Champion-name lexical normalization (Phase 2K v16)

Pass 1 remains text restoration only. The v2 lexical vocabulary
(`data/phase2k_support/league_lexical_vocabulary_v2.json`) adds explicit
champion alias spelling rules as lexical data only, never semantic entity or
binding output:

- **Direct aliases** are unconditional exact word-boundary hints:
  `Kale->Kayle` (capital-initial only; lowercase grocery `kale` is the
  documented negative via the `initial_capital` match rule), `Brier->Briar`,
  `Milo->Milio`, `Morana->Morgana`, `raan->Rakan`, `atrox->Aatrox`,
  `Talia->Taliyah`, `Nocturn->Nocturne`. Matching is case-insensitive while
  the canonical replacement is fixed.
- **Guarded aliases** (`pike->Pyke`, `rise`/`rice->Ryze`, `Sig->Ziggs`)
  require champion-shaped local syntax: a nearby `versus`/`vs`, a
  play/pick/ban/draft/counter/main/lane verb or role word (`support`, `mid`,
  `top`, `jungle`, `adc`, `matchup`, `mirror`), or the comparative
  `stronger/weaker/better/worse/... than` pattern with a champion context
  token on the comparator side. Ordinary usage — fishing pike, grocery kale,
  ordinary rise/rice, a signature `sig` — is never hinted.
- **Metadata-licensed** `darus->Darius` requires the supplied champion
  metadata to contain `Darius`; without it (`Varus` is a competing lexical
  possibility) `darus` is never hinted and stays unresolved.
- **Never/uncertainty-only** surfaces (`like->Pyke`, `then/when->Shen`,
  `ward->Bard`, `well->Rell`, `Soie->Zoe`) never produce hints; `Soie` may
  only be recorded as a `DOMAIN_TOKEN_UNCERTAIN` uncertainty.

The harness computes the deterministic hint list per target (exact
word-boundary detection, no edit distance/fuzzy matching) and includes it in
the mechanical prompt as `lexical_hints`. It never mutates `clean_text` and
never fabricates repairs. Every listed occurrence must appear as an explicit
provider `DOMAIN_SPELLING` repair with the exact Bronze quote and the exact
canonical replacement; missing, wrong-span, wrong-replacement, or extra
unlicensed `DOMAIN_SPELLING` repairs fail strict validation and feed the
correction flow. The mechanical response schema remains v3 (envelope
structure unchanged); the mechanical base prompt and correction prompt bump
to v4, the vocabulary schema/data bump to v2, the pipeline/config/records/
build-summary versions bump, and cache keys bind the new prompt/config
lineage so old v15 responses cannot be unsafely reused.

## Deterministic binding and sealing

Repair and uncertainty proposals quote exact text from Bronze; the harness
never trusts model offsets. Binding is deterministic:

- proposals are ordered, so the k-th proposal quoting a given original text
  binds to the k-th occurrence of that exact text in Bronze;
- every difference between Bronze and `clean_text` must be represented by
  exactly one bound, non-overlapping repair, and applying the bound repairs
  must reproduce `clean_text` exactly;
- a proposal whose quoted text cannot be bound to a unique Bronze slice,
  a repeated original with an ambiguous single proposal, or overlapping
  repairs all fail closed;
- a no-op proposal whose replacement equals its original text fails closed;
- every punctuation insertion/deletion must be represented by a full
  non-empty replacement span that includes the adjacent word;
- exact target-local and source-absolute spans, evidence spans, deterministic
  IDs, and full provenance are recomputed from Bronze and the sealed frozen
  inputs;
- the validated raw proposals are preserved verbatim as `raw_proposals` next
  to the content-addressed raw response, forming the audit trail between the
  raw response and the normalized record.

## Bounded provider corrections

Mechanical cleanup, the sufficiency diagnostic, contextual reconstruction,
and semantic polish each run one initial provider attempt plus strict
correction attempts.  Correction budgets are stage-specific: the
sufficiency diagnostic and contextual reconstruction keep at most two
corrections, while mechanical cleanup and semantic polish allow at most
three corrections so a malformed intermediate correction response can be
recovered without weakening validation.  A correction prompt embeds the
original task/schema, the exact prior raw response verbatim, the concise
validator error, and an exact response template, and asks for a complete
corrected JSON object only.  Validation stays strict: a failed response is
never normalized as valid merely because it is close, and the harness never
fabricates or auto-adds missing repair/binding/statement judgments.

- Every raw attempt is content-addressed under `raw_responses/<sha256>.txt`
  and is referenced by its ordered attempt record (status, attempt index,
  exact error, prompt/schema versions, and raw-response linkage).
- Cache keys bind the exact correction prompt hash plus attempt
  index/kind, prompt version, schema, and inference config, so a correction
  can never silently reuse or overwrite a different attempt.
- When a correction response is byte-identical to an earlier failed
  response, the next correction prompt adds explicit repeat guidance
  ("Identical repeat responses are rejected") and demands a materially
  different corrected JSON object; the repeat guidance changes the prompt
  hash, so the version lineage of the stronger second correction is
  content-addressed and auditable.
- On success the ordered attempt history is retained in the generated B
  record, the reconstruction/polish D subobjects (`attempts`), or the
  diagnostic attempt files.
- On exhaustion the exact final failure and the full ordered attempt
  history are preserved in the per-window `failure.json` placeholder.
- Raw-response tamper, missing-file, and orphan validation covers every
  retry raw file, including retries that only appear in failure artifacts.
- A source-absent binding mention error aggregates every unique absent
  `mention_text` in one bounded error and instructs the provider to remove
  each listed entire binding, so one correction can drop all invented
  mentions instead of consuming one attempt per error.  `NONE`-sentinel
  and context-only omission are the only no-claim omissions; valid,
  normalized, and grouped bindings are unaffected.
- When an evidence quote is genuinely absent but has a unique
  whitespace-skeleton match (every non-whitespace character appears in
  order inside one exact contiguous source slice, with one distinct slice
  text), the validation error suggests that exact slice as a verbatim
  replacement `evidence_quote`.  This is diagnostic guidance only: the
  malformed quote still fails closed and is never normalized or coerced.
  Zero skeleton matches or multiple distinct slice texts yield no
  suggestion.

## Sufficiency diagnostic contract

The sufficiency provider contract is a compact explicit schema
(`phase2k-sufficiency-response-v2`).  The model never counts absolute
offsets.  It supplies:

- `decision` — one of the closed sufficiency decisions;
- `slots` — all 11 exact `SLOT_KEYS`, each with `status`, `candidates`,
  categorical `confidence` (HIGH/MEDIUM/LOW, never floats), and
  `evidence_quotes`;
- compact candidates with `candidate`, `confidence`, and `evidence_quotes`;
- `metadata_conflicts` and `rationale`;
- the full exact response template for every slot is included in the prompt,
  with `RESOLVED` + `NONE` candidates for genuinely not-applicable slots.

Deterministic code binds every evidence quote to an exact source span inside
the supplied context (left-to-right ordered binding; a quote must be an
exact contiguous substring of one segment, and a repeated surface must be
quoted once per intended occurrence).  Binding fails closed when a quote is
absent, spans a segment boundary, or is ambiguous.  The final normalized
sufficiency response retains the conceptual fields required downstream:
`decision`, all 11 slots with `status`/`candidates`/`confidence`/
`evidence_spans` (exact sealed offsets), `metadata_conflicts`, and
`rationale`.  Raw compact proposals are preserved next to the normalized
diagnostic output as the audit trail, and the adaptive decisions stay
strict: `SUFFICIENT` iff every slot is `RESOLVED`, with unchanged
max-context rules.

Uncertainties are capped at 8. Only genuine unresolved alternatives that
could materially affect text restoration belong there; clear standard words,
intentional speaker/turn markers such as `>>`, and already-correct domain
tokens must not be listed, and an empty list is preferred when none.

The adversarial examples are part of the prompt and tests:

- `wed on the wave` may be repaired to `W'd on the wave` because the local
  phrase plus domain metadata determines the surface form without resolving
  an owner.
- `HS` remains unchanged when it could mean `his`, `has`, or a champion
  tag; it is recorded under `uncertainties` with explicit alternatives.

## Reconstruction and semantic polish

Contextual reconstruction runs only after the adaptive sufficiency
diagnostic is sealed. Its compact provider response
(`phase2k-reconstruction-response-v4`) contains judgment fields only:

- `schema_version`
- `clean_target_transcript` — must equal the complete non-overlapping
  application of `contextual_repairs` to Bronze
- `contextual_repairs` — proposals with `original_text` (exact Bronze
  quote), non-empty/different `replacement` with one narrow exception
  (`FILLER` deletions may use an empty `replacement` to delete a non-lexical
  filler plus its surrounding whitespace, for example `" MH" -> ""`; no
  other contextual repair type may use an empty replacement and generic
  deletions are never allowed), closed `repair_type`, categorical
  `confidence`, `evidence_quotes` (exact context quotes), and `rationale`
- `bindings` — proposals with `slot`, `mention_text` (exact Bronze quote),
  `resolved_candidate`, `resolved_status`, `confidence`, `evidence_quotes`,
  `alternatives` (`candidate`/`evidence_quotes`/`note`),
  `metadata_contributed`, and `rationale`
- `unresolved_alternatives` — proposals with `slot`, `mention_text`,
  `alternatives` (`candidate`/`confidence`/`evidence_quotes`),
  `evidence_quotes`, and `note`
- `rationale`

The provider never emits IDs, offsets, hashes, metadata copies, or
provenance. Deterministic code binds `original_text`/`mention_text` to exact
Bronze spans and `evidence_quotes` to exact context spans using ordered
occurrence rules (a repeated surface must be quoted once per intended
occurrence or a longer unique span), then seals repair/binding/alternative
IDs, source offsets, and the full provenance from the frozen inputs plus
the model rationale. Absent, unrepresented, no-op, overlapping, or ambiguous
proposals fail closed rather than being silently accepted or invented.

Contextual repair evidence is bound only after the repair spans are sealed,
and a narrow repair-anchored rescue applies to repair evidence quotes only:
when an exact evidence quote occurs multiple times in the supplied context
and the all-context occurrence-count rule would reject the one-quote
evidence list, the quote is accepted only if exactly one exact occurrence
contains the repair's source-absolute span, and that exact context span is
stored.  Zero or multiple containing occurrences preserve the existing
fail-closed ambiguity/absence errors.  The anchor never applies to binding,
statement, or general evidence ambiguity rules, and a malformed quote (for
example `"[ <NBSP>__<NBSP> ]"` instead of the exact
`"[\u00a0__\u00a0]"`) still fails with exact-quote guidance and is never
coerced.

Repair proposals are assigned to Bronze spans by a bounded deterministic
global assignment, not by a first-match shortcut.  The harness searches for
non-overlapping span selections that bind **every** provider repair proposal
(repeated identical proposals bind left-to-right to distinct occurrences)
and whose ordered replacement application reproduces
`clean_target_transcript` exactly.  A proposal is never silently dropped:
when a provider supplies a redundant or overlapping repair it must remove
that repair explicitly or merge overlapping edits into one exact Bronze
span in a correction response.  Zero valid selections fail closed with the
actionable diff feedback, overlapping candidate spans fail with explicit
merge guidance ("merge overlapping edits into one exact Bronze span... do
not drop or duplicate repairs"), an unbindable/redundant proposal fails with
explicit remove-entire-repair guidance, and more than one valid selection
fails closed as ambiguous (the provider must disambiguate with longer or
distinct quotes).  The global assignment therefore chooses the intended
occurrence even when the same short original (for example `e`) appears many
times: the unique selection reproducing the clean transcript is sealed with
exact Bronze slices and source offsets.

Reconstruction `mention_text` bindings are grouped by semantic assertion
(`mention_text` + `slot` + `resolved_candidate` + `resolved_status` for
bindings; `mention_text` + `slot` for unresolved alternatives). Inside one
assertion group the k-th proposal binds left-to-right to the k-th
occurrence, while different groups (for example different slots or
assertions) may share the same deterministic occurrence — a repeated
addressed-player `you` or `she` may legitimately appear once in several
slots. Supplying fewer proposals than the total source occurrence count is
valid; supplying more proposals than occurrences inside one assertion group
still fails closed. The compact prompt states this contract explicitly.

For compact reconstruction `mention_text` and context `evidence_quotes`
only, exact binding remains first choice and a deterministic
surface-normalized fallback is permitted when the exact surface is absent
or its exact occurrence count cannot be matched. The fallback ignores only
capitalization, punctuation, Unicode-space variants, and whitespace runs;
it maps back to one exact contiguous Bronze/context source span and must be
unique (zero or multiple normalized matches fail closed). Spelling/lexical
changes, edit distance, synonyms, champion names, and ASR substitutions
never normalized-match (`W'd` vs `wed` and `pryo` vs `prio` stay distinct).
The normalized artifact always stores the exact source slice and exact
offsets while the raw compact envelope preserves the provider quote. The
fallback is never applied to Pass 1 `original_text`/mechanical repairs,
Polish evidence quotes, or sufficiency evidence quotes, which stay exact.

Entity, pronoun, ability-ownership, and reference resolutions require a
matching `RESOLVED` binding over the exact same mention span whose candidate
is licensed by the final diagnostic slots and/or metadata. Unresolved
mentions are never rewritten. For `ENTITY_RESOLUTION` repairs only, one
`RESOLVED` binding in an allowed entity slot (`champion_identities` or
`principal_actors`) may additionally license repeated identical repair
occurrences when the binding mention source surface and the repair
`original_text` are equal under strict surface normalization and the
binding `resolved_candidate` and the repair `replacement` are equal under
strict surface normalization (one Aatrox binding can license every
`atrox` → `Aatrox` canonicalization repair). Pronoun, reference, and
ability-ownership repairs stay exact-span licensed because repeated
pronouns/mentions can have different referents, and a different candidate
never licenses a repair. A narrow exact-span composite license additionally
applies to `ENTITY_RESOLUTION` when the binding mention span exactly equals
the repair span with exact Bronze text, the resolved candidate appears as a
complete normalized token sequence inside the replacement, and the
replacement differs from the original only by replacing one contiguous
token sequence with that candidate while all surrounding normalized tokens
remain identical in order (licensing exact-span `this darus` → `this
Darius` with candidate `Darius`). Broad rewrites, substring-inside-token
candidate matches, wrong candidates, changed surrounding words, and
non-exact mention spans stay rejected. Literal words that already appear in Bronze or an
exact quoted evidence span may remain in the source-faithful reconstruction
(including strategy words such as priority or pressure), while newly
introduced ontology abstractions not licensed by exact source evidence are
rejected. The no-final-strategy-ontology rule is preserved.

Composed candidate licensing applies narrowly to the reference-resolution
slots `pronouns`, `discourse_refs`, and `unresolved_asr` only.  A binding in
one of those slots validates when the binding mention itself is licensed by
a candidate in that slot's own diagnostic candidates (exact surface match,
contracted mention stem, or an explicit equation/annotation mention side such
as `she/her = enemy mid laner; you/your = Veigar player`) **and** the
resolved referent is licensed by the entity/principal-actor diagnostic
candidates or supplied metadata (exact surface match or the
equation/annotation resolution side).  `ability_ownership` is intentionally
direct-only: the data contract stores ownership as full-phrase candidates
(for example `Ignite is owned by the player (Lucian)`), so composed
licensing never applies there.  An n2-style `we` binding is rejected when
`we` appears on no diagnostic mention side, while the same payload's `e` →
`E` repair still binds the intended `uses e` occurrence before/independently
of the invalid semantic binding.

Binding proposals that carry no real resolution claim are omitted narrowly
and only under the documented sentinel rule: an explicit `NONE` placeholder
(`mention_text="NONE"` and `resolved_candidate="NONE"`) or a proposal whose
`resolved_candidate` is `NONE` is preserved verbatim in `raw_compact` and
omitted from the normalized semantic bindings, with the omission counted in
`omitted_binding_count`.  Output validation audits that the raw `bindings`
count equals the normalized binding count plus `omitted_binding_count`.
Other source-absent mentions (for example `your queue`) are not sentinels:
they still fail closed with explicit remove-entire-binding guidance and are
never normalized.

There is one additional narrow, deterministic context-only omission rule
for binding proposals: a proposal whose `mention_text` has zero exact or
surface-normalized matches in the target Bronze but at least one exact or
surface-normalized match inside the supplied ordered context refers only to
the surrounding context, never to a target-Bronze mention.  Target bindings
must refer to target Bronze mentions, so such a proposal is conservatively
omitted from the normalized target bindings while its verbatim original
remains in `raw_compact` and the omission is included in
`omitted_binding_count` (keeping the raw-count audit invariant).  A
context-only mention is never used to license entity text, and the
downstream clean-text/entity validation still runs and fails if the clean
transcript depends on the omitted binding.  Proposals with any target match
keep the normal deterministic binding path and still fail closed on
count/ambiguity errors; proposals absent from both the target and the
supplied context (for example `your queue`) are not context-only and keep
failing with the existing remove-entire-binding guidance.  The live
`enemy mid` proposal from pool:n2RuZ0vwkE4:w00288 is the motivating case:
the phrase occurs exactly in a preceding context segment but not in the
target Bronze, so it is omitted rather than accepted as a target binding.

Contextual `WHITESPACE` is the single closed repair type for Unicode
whitespace normalization. When Bronze contains a Unicode whitespace
(for example a non-breaking space inside a `[ __ ]` mask) that the provider
cannot reproduce, a `WHITESPACE` proposal may quote the regular-space form
as both `original_text` and `replacement` only when that exact quote is
absent from Bronze, a deterministic whitespace-skeleton matcher maps it to
a unique Bronze slice that differs bytewise only in Unicode whitespace, the
sealed exact Bronze slice differs from the replacement, and the complete
repair application reproduces `clean_target_transcript`. The sealed repair
stores the exact Bronze slice, the regular-space replacement, exact
offsets/evidence/provenance, and the `WHITESPACE` type; the raw compact
envelope preserves the provider's regular-space text. True no-ops (an
exact identical slice), arbitrary punctuation/case changes, and other
unrepresented edits still fail closed.

A whitespace-only `WHITESPACE` proposal (for example
`original_text="\u00a0"` with `replacement=" "`) is the narrow explicit rule
for replacing one Unicode whitespace character with another.  The harness
binds whitespace-only proposals left-to-right to the whitespace-only Bronze
slices of the same length, and repeated identical proposals bind to distinct
successive whitespace slices.  The proposal must differ bytewise only in
Unicode whitespace, the evidence quote must be an exact contiguous context
span, and the complete repair application must reproduce
`clean_target_transcript`; true no-ops and non-whitespace differences are
never accepted.  This rule is strict about evidence: the old AOxq correction
payload that quoted `"[ \u00a0__\u00a0 ]"` (regular spaces around NBSPs) is
rejected because that quote is genuinely absent — the Bronze slice is
`"[\u00a0__\u00a0]"` — and the failure stays actionable with exact-quote
guidance.

Named-entity validation uses a dedicated lexical token extractor: word
tokens are stripped of surrounding punctuation and contractions such as
`I'm`/`It's` are never entity tokens. A capitalized named entity is licensed
case-insensitively when its lexical token appears in the exact Bronze text,
in exact sealed reconstruction evidence, in metadata values, or in
resolved/alternative binding candidates. Ordinary sentence-initial
capitalization of words already present in the licensed text is therefore
not a false positive, while a genuinely new capitalized named entity absent
from every license source still fails closed. Reconstruction licenses
against Bronze plus all exact reconstruction evidence; each Polish
statement licenses against Bronze plus that statement's exact evidence and
metadata/bindings. Strategy-abstraction validation remains separate and
strict.

Semantic polish runs separately after reconstruction is sealed. Its compact
response (`phase2k-semantic-polish-response-v2`) contains `schema_version`,
`statements`, `unsupported_claims`, and `rationale`. Each compact statement
carries judgment fields only: `text`, preservation attestations for
modality/negation/uncertainty, `evidence_quotes` (exact Bronze quotes),
`reconstruction_operation_ids` (existing repair/binding IDs, optional),
closed `support_mode`, and `unchanged_source_quote` (string or null).

Deterministic code binds evidence and unchanged quotes to exact Bronze
spans and seals statement IDs. Support modes are validated exactly:
`UNCHANGED_EXACT` requires `text` equal the unchanged source quote;
`RECONSTRUCTION_DERIVED` requires at least one valid reconstruction
operation ID; `EVIDENCE_PARAPHRASE` requires exact Bronze evidence and
allows zero operation IDs, so evidence-grounded paraphrase works even when
reconstruction made no repair/binding. Every statement requires evidence,
unknown operation IDs fail, and `unsupported_claims` keep the closed reason
enum. Named entities/claims must remain licensed; human audit judges
preservation versus invention.

Repaired text must stay `RECONSTRUCTION_DERIVED` with `unchanged_source_quote`
null, reference the operation IDs that support the change, and quote the
exact Bronze original text in `evidence_quotes` (never the repaired text).
When a reconstruction-derived evidence quote equals repaired text but is
absent from Bronze and strict surface-normalized matching identifies one
exact Bronze slice, the validation error suggests that exact Bronze
`evidence_quote` verbatim; the suggestion is guidance only and the response
still fails closed.  Zero or multiple normalized slices yield no suggestion.
On a text/unchanged-source mismatch, the correction prompt instructs the
provider with the exact data and two explicit actions: the validator error
carries the JSON-safe exact actual statement text and the exact
`unchanged_source_quote`, and instructs the provider to either (a) safest —
set the statement `text` byte-exactly equal to that unchanged quote (copy
the quoted value verbatim), or (b) use `support_mode`
`RECONSTRUCTION_DERIVED` with `unchanged_source_quote` null and at least one
valid `reconstruction_operation_id` supporting the change.  Repaired text
is never relabeled as `UNCHANGED_EXACT`, and `UNCHANGED_EXACT` validation
is never weakened.

Reconstruction uses one initial provider attempt plus at most three strict
correction attempts (four total calls); semantic polish uses one initial
provider attempt plus at most three strict correction attempts so a
malformed intermediate correction can be recovered. The sealed D subobjects
retain the ordered attempt history and the verbatim raw compact proposals
(`attempts` and `raw_compact`) next to the content-addressed raw responses.
A GENERATED D requires both passes; a failure placeholder retains the exact
failure stage and full attempt history (and keeps the sealed reconstruction
when only semantic polish failed).

When the mechanical or reconstruction clean text does not equal the
deterministic application of the supplied repair list, the validator error
embedded in the correction prompt carries concise bounded diff feedback:
the applied text versus the requested clean text plus the ordered
non-equal `difflib` opcodes with exact differing substrings and positions
(sizes bounded so errors/prompts cannot explode). Each opcode is also
expanded to the surrounding full word in the applied and requested texts
with a concrete diagnostic suggestion such as
`original_text="exhaust" replacement="Exhaust"` (or the minimal full
replaceable span including adjacent punctuation for
insertions/deletions). Suggestions are diagnostic only — the provider must
still return the explicit repair and nothing is auto-created. The
reconstruction correction prompt explicitly instructs the provider to use
that feedback, return every missing change as a full non-empty replacement
span, and use `WHITESPACE` for Unicode whitespace differences. Exhaustion
remains fail-closed with the full ordered attempt history preserved.

## A/B/C/D records

All record conditions share the exact Bronze target identity:

- **A** — exact isolated raw Bronze.
- **B** — exact mechanical target. Zero edits are allowed.
- **C** — exact ordered enlarged context; the reviewer-facing target is
  replaced with B.clean_text when B exists, while the record keeps the exact
  Bronze target hash/identity.
- **D** — `reconstruction` and `semantic_polish` subobjects, each with its
  own sealed model call, prompt/schema versions, raw response path/hash, and
  counts. D never masquerades as B or Bronze.

No-provider D is an explicit `NOT_GENERATED` placeholder carrying the
mechanical/Bronze text. Live D with a per-window failure is also a
placeholder, not a valid generated D. The human promotion gate fails when
any D is unavailable.

## Context and metadata lineage

Segments are deterministic with exact offsets and bounded fallback. The
adaptive radius stages are target-only, ±2, ±5, ±10, and bounded local
episode (±40), with per-side hard caps.

Metadata is field-level and provenance-bound:

- `source` points to the frozen Phase 2J manifest field;
- `provenance_hash` binds field/value/canonical record;
- `reliability` is `SUPPLIED_FACT`;
- `inference_allowed` is true for champion/role and false for
  provenance-only `video_title`.

`video_title` is never passed as a model-visible supplied fact and no
title-based matchup inference exists.

Build lineage records repo HEAD/dirty status, implementation file hashes,
the vocabulary snapshot path/file/content hashes, config hash, and the
sealed secret-free inference-config hash. Cache keys include the prompt,
inference config, and response schema hashes.

The pipeline version was bumped for the v6 contract, the reconstruction
response schema was bumped to v4, the semantic-polish prompt/response
schema was bumped to v2, the transformation-audit packet was bumped to v2,
and live inference seals `max_tokens=8192`, so stale cache entries cannot be
reused as current.

The v6 robustness repair bumped the mechanical correction prompt to v2 and
the reconstruction prompt to v6 / reconstruction correction prompt to v4:
correction prompts embed bounded ordered diff feedback for
clean-text/application mismatches with full-word diagnostic suggestions and
instruct the provider to return every missing change as a full non-empty
replacement span. Reconstruction `mention_text` binding is grouped per
semantic assertion (different slots may share a mention span; fewer
proposals than occurrences is valid), `ENTITY_RESOLUTION` repairs may be
licensed across repeated identical surfaces by one equivalent entity
binding, and the closed contextual `WHITESPACE` repair type normalizes
Unicode whitespace while storing the exact Bronze slice.  The Phase 2K
contract hardening adds the bounded global repair assignment that binds
every proposal (never silently dropping one; redundant/overlapping repairs
must be removed or merged by a provider correction and ambiguous
assignments fail closed), the whitespace-only NBSP→space rule with exact
evidence, the narrow `NONE`-sentinel omission with `omitted_binding_count`
and `raw_compact` count audit, composed reference licensing for
pronouns/discourse refs/unresolved ASR (direct-only for ability ownership),
and byte-identical-repeat correction guidance.  These are validation and
prompt-contract behaviors, not response-schema changes, so the reconstruction
response schema remains v4.  The final live-fix pass bumped the
reconstruction prompt to v7 and the reconstruction correction prompt to v5
so live calls cannot reuse stale cached responses: the correction prompt
now instructs the provider to use any suggested exact `evidence_quote`
verbatim and to remove every listed source-absent binding in one
correction, the main prompt requires byte-exact evidence quotes, and the
validator reports the exact-span composite entity license plus the
aggregated source-absent binding guidance described above.

The live #7 fix pass keeps the reconstruction response schema at v4 and
bumps the reconstruction prompt to v8 and its correction prompt to v6
(FILLER-only empty replacements plus repair-anchored exact evidence),
the mechanical correction prompt to v3 and the semantic-polish prompt to
v3 / correction prompt to v2 (stage-specific three-correction budgets and
stronger polish guidance), so live caches cannot reuse stale correction
outputs.  Mechanical cleanup and semantic polish now allow one initial
attempt plus at most three corrections; the sufficiency diagnostic keeps
the global default of at most two corrections.  The docs above describe
the repair-anchored exact-evidence rescue, the FILLER-only
empty-replacement exception, and the reconstruction-derived polish
evidence guidance.

The Phase 2K v14 reconstruction-correction hardening bumps only the
reconstruction correction prompt to v7 and gives contextual reconstruction
its own correction budget of one initial attempt plus at most three
corrections (four total calls), while every other stage keeps its existing
budget.  The v7 correction prompt requires every `evidence_quotes` field
at every nesting level (contextual repairs, bindings, binding alternatives,
unresolved alternatives and their alternatives) to be a JSON array of exact
quote strings, never a scalar/string/object/null, even for a single quote;
it requires `resolved_candidate` on RESOLVED bindings to copy one complete
licensed diagnostic candidate or metadata value byte-for-byte (case,
descriptors, and parenthetical qualifiers included), copying the chosen
full exact value from any allowed-candidate list in `validator_error`
instead of inventing a nicer label; and it keeps exact NBSP/Unicode
whitespace quoting with explicit `WHITESPACE` repairs.  A structured
`correction_rules` list is embedded in the correction user payload as
prompt-side guidance only (no semantic outputs or provider-controlled
provenance).  Because the correction prompt version participates in the
content-addressed cache key, v7 corrections cannot reuse or overwrite v6
cached responses, and no cache files or v13 output are deleted or mutated.

The Phase 2K v15 live repair keeps the pipeline version
(`phase2k-contextual-reconstruction-v6`), the response schemas, and every
base prompt version unchanged so cached successful base calls remain
reusable; it bumps only the semantic-polish correction prompt from v2 to v3
(`phase2k-semantic-polish-correction-prompt-v3`).  The v15 change adds the
narrow deterministic context-only binding-omission rule described above
(a mention with zero target-Bronze matches but at least one exact or
surface-normalized match in the supplied ordered context is omitted from
normalized target bindings, preserved verbatim in `raw_compact`, and
counted in `omitted_binding_count`), and makes an `UNCHANGED_EXACT` polish
text mismatch report the exact actual text and exact unchanged quote with
explicit source-exact / `RECONSTRUCTION_DERIVED` repair instructions.  The
v3 correction prompt participates in the content-addressed correction cache
key, so stale v2 correction responses cannot be reused while base polish
prompt-v3 cache entries remain valid.

The Phase 2K v16 champion-name lexical normalization upgrade bumps the
pipeline version to `phase2k-contextual-reconstruction-v7`, config version to
`phase2k-config-v3`, the lexical vocabulary to v2
(`phase2k-league-lexical-vocabulary-v2`), the mechanical cleanup base prompt
to v4 (`phase2k-mechanical-cleanup-prompt-v4`), and the mechanical correction
prompt to v4 (`phase2k-mechanical-cleanup-correction-prompt-v4`). The
mechanical response schema stays v3 because its envelope structure is
unchanged. Records and build-summary schemas bump to v7/v5 so the new B
lexical-hint snapshot and lineage are versioned. `_config_hash` binds the
config/mechanical prompt versions and the v2 vocabulary hash, and cache keys
bind prompt version + config hash, so old v15 cached responses cannot be
reused for the v16 mechanical cleanup. The v2 vocabulary path is
`data/phase2k_support/league_lexical_vocabulary_v2.json`; the v1 file is
preserved untouched.

## Raw response integrity

Every live provider call writes its exact raw response under
`raw_responses/<sha256>.txt` and records that content-addressed path/hash in
its model call. Output validation rejects:

- missing referenced raw files;
- tampered raw files;
- unreferenced orphan raw files;
- model calls whose recorded inference-config snapshot/hash does not match
  the sealed top-level config.

## Build outputs

- `phase2k-frozen-input-manifest-v1.json`
- `phase2k-reconstruction-records-v7.json`
- `phase2k-human-review-packet-v2.json`
- `phase2k-human-review-mapping-v2.json`
- `phase2k-build-summary-v5.json`
- `phase2k-transformation-audit-packet-v2.json` (live only)
- `raw_responses/` and `attempts/` (live only)

The human packet is strictly blind. The mapping is separate and bound by
hash. C presentations display exact surrounding context plus the mechanical
target while retaining the Bronze identity in the mapping/record.

## Transformation audit

The live build emits a blank, downstream-result-blind transformation audit
for:

- mechanical and contextual repairs;
- entity, pronoun, reference, and ability-ownership bindings;
- polished statements.

Each operation has an exact ID, evidence, and blank human decision fields.
Repairs also have blank `corrected_replacement` and `error_taxonomy`; the
packet includes the closed error taxonomy and decision set. Each window
records `first_failure` and the reconstruction-specific
`first_reconstruction_failure`. Polished statements carry their sealed
`support_mode` and `unchanged_source_quote` alongside the existing
evidence/operation references. Validation binds the audit to records and
rejects missing/extra operations. Binding operations carry the canonical
five-field Bronze span `mention` object (`target_local_start`,
`target_local_end`, `source_absolute_start`, `source_absolute_end`, `text`),
and validation checks that mention against the window Bronze target exactly
like reconstruction bindings; string mentions and inconsistent spans are
rejected.

## Human review and finalization

`scripts/finalize_phase2k_human_review.py` requires:

- complete blind representation reviews;
- a complete transformation audit for live builds.

The web app provides two client-only operator surfaces. Start it from
`apps/web` with `npm run dev`; packet files stay in the browser and are not
uploaded:

- `/phase2k-review` imports `phase2k-human-review-packet-v2.json`, autosaves a
  packet-hash-bound session, and exports the complete reviews map accepted by
  `--reviews`. It never loads the separate condition/radius mapping.
- `/phase2k-audit` imports
  `phase2k-transformation-audit-packet-v2.json`, verifies the packet's
  canonical hash, autosaves a records-hash-bound session, and exports the
  completed audit accepted by `--audits`. It never displays downstream model
  results.

Both surfaces refuse final export until every required human field is
explicitly completed. Scores, reviewer identity, timestamps, decisions, and
statement attestations are never prefilled. Work-in-progress session exports
are backups only and are not accepted as finalized human evidence.

For a live output directory, finalize after both browser exports exist:

```bash
.venv/bin/python scripts/finalize_phase2k_human_review.py \
  --output-dir /path/to/sealed-phase2k-live-output \
  --reviews /path/to/phase2k-completed-reviews.json \
  --audits /path/to/phase2k-completed-transformation-audit.json \
  --reviewer "Human reviewer" \
  --completed-at "2026-08-19T00:00:00Z"
```

Do not import or finalize the interrupted/stale live attempts. Run
`build_phase2k_reconstruction.py --validate-only` against the chosen sealed
output first; the browser audit validator intentionally rejects artifacts
that no longer satisfy the current Python contract.

It writes:

- `phase2k-human-review-packet-v2-finalized.json`
- `phase2k-human-review-summary-v1.json`
- `phase2k-transformation-audit-packet-v2-finalized.json` (live)
- `phase2k-transformation-audit-summary-v1.json` (live)
- `phase2k-closeout-status-v2.json`

Deterministic transformation metrics include ASR approved/proposed by type
and confidence, entity precision and required-resolvable recall,
ability-owner accuracy, unsupported rate, modality/negation/uncertainty
preservation, first-failure counts, and operation decisions. No score or
audit decision is fabricated; the synthetic tests provide all values.

## Downstream semantic-target alignment (mandatory post-review gate)

After the finalized human review gate is `PASSED` and the completed
transformation audit validates, Phase 2K emits the scorer-blind, post-review
semantic-target alignment packet that the paired Phase 2F generative and
Phase 2H discriminative reruns require. This is contract/tooling only: it
never runs providers, never runs Phase 2F/2H scoring, and never fabricates
human decisions. The reruns themselves are required downstream after this
gate passes; the alignment builder does not execute them.

The builder is fail-closed:

- the Phase 2K records must be a live build — no-provider mode is rejected;
- every D record must be `GENERATED` with a sealed `semantic_polish`
  subobject — placeholders, `NOT_GENERATED` D records, and missing polish
  are rejected;
- the finalized human packet/summary must recompute to a `PASSED` review
  gate via the existing validators, and stale/invalid records are rejected
  through the existing completed-audit/summary validation;
- the completed transformation audit must validate against the blank audit
  and records;
- the frozen input manifest in the Phase 2K output directory must resolve
  the Phase 2J window-selection manifest and the read-only transcript DB
  (repo-relative locators resolve from the repository root; absolute paths
  stay absolute), and the whole live output must pass the Phase 2K core
  `validate_output_directory` deep validation before any alignment input is
  accepted — current pipeline/prompt/config schema versions, sealed
  reconstruction/polish validation, diagnostic attempts, raw response
  files, and provider lineage are all rechecked, so a self-consistent stale
  output is rejected, not merely an internally inconsistent one;
- the Phase 2J reviewed packet and candidate-coverage artifact must bind by
  content/file hash.

Build the blank packet only from a finalized live output directory:

```bash
.venv/bin/python scripts/build_phase2k_downstream_alignment.py \
  --phase2k-dir /path/to/sealed-phase2k-live-output \
  --reviewed-packet data/phase2j/reviewed-endpoint-annotation-packet-v1.json \
  --coverage data/phase2j/candidate-coverage-v1.json \
  --output /path/to/phase2k-downstream-alignment-packet-v1.json
```

`--validate-only` revalidates an existing packet against the same current
sources without writing anything.

### Packet contract

The packet is a canonical hash envelope
(`phase2k-downstream-alignment-packet-v1`) with exact top-level keys
`schema_version`, `content_sha256`, `purpose`, `release_gate`,
`dataset_binding`, `boundary_rule`, and `items`. The blank packet is
`AWAITING_HUMAN_REVIEW` with null/empty decisions; finalization sets
`REVIEWED` and never fabricates a decision.

`dataset_binding` cryptographically binds the Phase 2K records content hash,
the Phase 2J reviewed packet and coverage content hashes, the finalized
human packet content hash, the recomputed human summary canonical hash, the
completed transformation audit content hash, the sorted 30-window ID
hash/count, the 311 target count, and `human_review_gate_status: PASSED`.

One ordered item is emitted per Phase 2J KEEP endpoint (311 items, ordered
by records/window order then endpoint position). Each item preserves the
exact endpoint ID, window ID, and node type from the reviewed packet —
including the single real endpoint whose `node_type` is `null`
(`p2j:pool:MjHLNnOPgn8:w00190-ad0cc2adb93f3e63133b:ep:0008`); validators
accept `null` only as inherited from the bound reviewed packet. Raw Bronze
target identity is exact, and the sealed D `clean_target_transcript` plus
`semantic_polish.polished_text` are retained with SHA-256 hashes. The
endpoint-to-polish alignment is never inferred from broad statement
evidence; only the sealed D fields are used.

### Versioned boundary rule

The exact 263 covered endpoints keep their exact reviewed spans. For the
exact 48 artifact-identified missing endpoints only
(`MIXED_BOUNDARY_MISMATCH` / `CANDIDATE_GENERATION_MISS`, each ending in one
terminal `.` (28) or `,` (20), with exactly one overlap candidate at the
same start/end-1/text without terminal punctuation), the raw evaluation span
drops exactly that one terminal character: start unchanged, `end - 1`, text
without the terminal punctuation. This is declared as
`phase2k-target-boundary-rule-v1-phase2j-terminal-punctuation`. The reviewed
packet and coverage artifact are never mutated, and all 311 endpoint
identities remain unchanged.

### Finalization and summary

```bash
.venv/bin/python scripts/finalize_phase2k_downstream_alignment.py \
  --phase2k-dir /path/to/sealed-phase2k-live-output \
  --reviewed-packet data/phase2j/reviewed-endpoint-annotation-packet-v1.json \
  --coverage data/phase2j/candidate-coverage-v1.json \
  --packet /path/to/phase2k-downstream-alignment-packet-v1.json \
  --decisions /path/to/phase2k-alignment-decisions.json \
  --output /path/to/phase2k-downstream-alignment-packet-v1-finalized.json \
  --summary /path/to/phase2k-downstream-alignment-summary-v1.json
```

`--phase2k-dir` is required, and `--reviewed-packet`/`--coverage` default to
the same immutable Phase 2J artifacts as the build CLI. The finalizer loads
and validates the current alignment inputs, then validates the blank packet
against those sources (`require_blank=True`), applies the decisions, and
validates the finalized packet against the same sources
(`require_blank=False`) before writing anything. A canonical-but-forged
blank packet — one whose hashes are internally consistent but whose
dataset binding or sealed display/source content does not match the current
live output — is rejected, so self-contained validation is no longer
sufficient for the production CLI.

The compact decisions map is keyed by `alignment_id` with exact
`state`/`polished_spans`/`reviewer`/`completed_at`/`notes` fields and must
cover every alignment ID. Final states are `ALIGNED` (one or more spans),
`ABSENT` (zero spans), `AMBIGUOUS` (zero or more spans), and
`MULTIPLE_CANDIDATES` (at least two spans); all require a non-empty reviewer
and completed timestamp. Spans are exact half-open slices of the sealed
polished text, positive integers (bools rejected), unique, and
deterministically sorted. Within a window, two different endpoint IDs may
not both claim the exact same `(start, end)` in `ALIGNED`/
`MULTIPLE_CANDIDATES`: one selected output cannot count as two targets, so a
reviewer must leave one compressed target `AMBIGUOUS`/`ABSENT` rather than
silently collapsing target identity.

The summary (`phase2k-downstream-alignment-summary-v1`) reports total 311,
counts/rates by state, node type, and window, boundary-correction counts,
and unresolved `ABSENT`/`AMBIGUOUS` targets, and hash-binds the finalized
packet. `ABSENT`/`AMBIGUOUS` items remain targets for later evaluation; they
are not deleted or excluded. The packet is scorer/model blind: no downstream
predictions, model results, scores, thresholds, or semantic extraction ever
appear in it.

## Gate-locked paired downstream rerun (evidence producer)

Once the finalized alignment packet is `REVIEWED` and its summary validates,
the paired Phase 2F generative and Phase 2H discriminative rerun may run over
the exact same 30 windows under `RAW_BRONZE` vs `CONTEXTUAL_POLISH`.  The
producer is `pipeline/phase2k_downstream_rerun.py` with CLI
`scripts/run_phase2k_downstream_rerun.py`; it is downstream only and never
performs mechanical cleanup, never edits Phase 2J/Phase 2K artifacts, and
never runs either semantic architecture or fabricates result rows before
every gate passes.

### Required inputs and gates

The production entry requires the Phase 2K output directory, the finalized
alignment packet, the alignment summary, the Phase 2J reviewed packet, and
the candidate-coverage artifact.  It reuses the alignment module's
source-bound loading/validation and additionally requires:

- deep-validated live Phase 2K records with finalized A/B/C/D human review
  recomputing to a `PASSED` gate and a validated completed transformation
  audit;
- a finalized alignment packet whose release gate is `REVIEWED`, whose
  dataset binding matches the current Phase 2K/2J sources exactly, and which
  carries all 30 windows and all 311 endpoint IDs;
- the alignment summary recomputing to that exact packet;
- exact raw/polished representation text: the raw adapter text must equal the
  sealed D/A Bronze target text (verified against the A content and source
  hashes), and the polished adapter text must equal the sealed D
  `semantic_polish.polished_text`.

### Deterministic representation adapters

Both representations are built with `BronzeSource` +
`window_from_exact_span`, exactly like the Phase 2J coverage regeneration.
The raw adapter is `transcript:<upstream_source_id>` over the exact Bronze
window text; the polished adapter is `polished:<phase2k-window-id>` over the
sealed polished text.  Every accepted target span is re-verified as an exact
slice of the corresponding adapter window.  Adapter descriptors (version,
representation, window identity, source identity/kind, text SHA-256, span)
are canonical-hashed; raw and polished adapter hashes always differ and are
bound into every evidence envelope.

### Target contract

`RAW_BRONZE` accepts each alignment item's `bronze_target`
`evaluation_start`/`evaluation_end`/`evaluation_text`, including the versioned
48 terminal-punctuation corrections.  `CONTEXTUAL_POLISH` accepts only
finalized alignment spans: `ALIGNED` one or more exact spans,
`MULTIPLE_CANDIDATES` two or more exact alternative spans for one target, and
`ABSENT`/`AMBIGUOUS` no accepted positive span (the target stays in
`target_count` and the FN denominator).  No target is deleted, and no same
exact polished span may belong to two targets (enforced by the alignment
validator).  The semantic target contract binds the deterministic ordered
311-item target list; per-window `target_count` is identical across every
raw/polished architecture cell and sums to 311.

### Generative (Phase 2F) cell

`compile_source_semantic_ir` runs on each representation window with one
identical `SemanticCompilerConfig` and chat callable.  Endpoint outputs are
evaluated from `mention_nodes` by exact local span; when the target
`node_type` is non-null the generative output must carry that exact type;
an output with `node_type: null` cannot wildcard a typed target. A target
whose type is null remains a wildcard. Matching is deterministic
one-target-at-most-one-TP: a second node matching another
alternative of the same target is FP.  All compiler output nodes count as
outputs; source-provenance validity (exact window slice) is checked and
reported.  Full typed run evidence and raw provider lineage are preserved
through the existing Phase 2F semantic run artifacts.

The compiler execution identity includes the exact normalized entity and
ability alias sets as well as the frozen compiler config. The live CLI loads
champion names from `data/semantic_ir_window_pool_v1.json` and uses the frozen
ability aliases `Q`, `W`, `E`, `R`, `ult`, `ultimate`, `Flash`, `Teleport`,
`Ignite`, `Exhaust`, `Ward`, and `Sweeper`. A typed `PROVIDER_FAILURE` aborts
the whole transaction; no partial evidence directory is published.

The live CLI requires an explicit `--live` flag and loads the provider the
same safe way as `scripts/eval_phase2f_semantic_ir.py` (official DeepSeek
endpoint check); default/preflight mode never calls a provider.

### Discriminative (Phase 2H) cell

Candidates are reproduced with the frozen Phase 2F
`generate_mention_candidates` API and custom `CandidateRow` datasets are
constructed for both representations.  Training labels are `KEEP` for every
accepted alternative exact span and `DROP` otherwise.  The same grouped
leave-one-window-out folds run over all four fixed Phase 2H cells with the
existing `run_cv`/`compute_rankings` APIs and `KEEP_THRESHOLD=0.5`; no tuning
is performed.  The primary cell is declared before execution (default
`logistic_B`); every cell's scores/results are preserved as supplementary
evidence, while comparison-v2 rows use only the declared primary.  Selected
candidates are evaluated with the same one-target-at-most-one-TP rule
(exact span; no predicted node type), and candidate provenance is exact by
construction.

Every discriminative artifact retains the complete fold and fit-scope
evidence. Validation rebuilds the exact bound candidate dataset and
deterministically replays grouped leave-one-window-out CV and rankings,
comparing every fold, fit scope, cell score, cell metric, and summary row.
Canonical resealing therefore cannot legitimize altered scorer evidence.

### Emitted artifacts

The live run publishes immutable canonical-hash envelopes into a directory
that must not already exist, built in a temporary sibling and atomically
renamed only after complete validation:

- `preflight-input-contract.json` — sealed input contract (dataset/
  semantic-target bindings, adapters, compiler/scorer configs, gate evidence)
  with no predictions;
- `generative-raw.json` / `generative-polished.json` — full Phase 2F artifacts
  with typed run evidence per window, rows, and matching details;
- `discriminative-raw.json` / `discriminative-polished.json` — full all-cell
  Phase 2H artifacts with per-cell scores, rows, and matching details;
- `comparison-input.json` — evidence artifact carrying the two v2
  architecture builder blocks and exact bindings, with
  `output_artifact_sha256` bound to the actual corresponding envelope content
  hash, but no decision/diagnosis/note.

`validate_rerun_evidence` reloads every file and recomputes hashes, rows,
configs, and adapters from the current Phase 2K sources, rejecting tampering,
swapped raw/polished artifacts, a changed primary scorer, stale gates, and
target mismatch.  A partial live/provider failure never publishes a complete
result directory.

```bash
.venv/bin/python scripts/run_phase2k_downstream_rerun.py \
  --phase2k-dir /path/to/sealed-phase2k-live-output \
  --alignment-packet /path/to/phase2k-downstream-alignment-packet-v1-finalized.json \
  --alignment-summary /path/to/phase2k-downstream-alignment-summary-v1.json \
  --reviewed-packet data/phase2j/reviewed-endpoint-annotation-packet-v1.json \
  --coverage data/phase2j/candidate-coverage-v1.json

.venv/bin/python scripts/run_phase2k_downstream_rerun.py \
  --phase2k-dir /path/to/sealed-phase2k-live-output \
  --alignment-packet /path/to/phase2k-downstream-alignment-packet-v1-finalized.json \
  --alignment-summary /path/to/phase2k-downstream-alignment-summary-v1.json \
  --output /path/to/phase2k-downstream-evidence \
  --live --primary-cell logistic_B
```

### Explicit finalization

`scripts/finalize_phase2k_downstream_rerun.py` consumes the validated
evidence directory plus explicit human-supplied `--decision`, `--diagnosis`,
and `--note` arguments, calls `build_downstream_comparison`, validates the v2
envelope against the exact Phase 2K records, finalized human packet, human
review summary, and completed transformation audit, and writes the
`phase2k-downstream-comparison-v2` file.  The diagnosis and note are never
inferred, and the existing enum values are enforced.

```bash
.venv/bin/python scripts/finalize_phase2k_downstream_rerun.py \
  --evidence-dir /path/to/phase2k-downstream-evidence \
  --phase2k-dir /path/to/sealed-phase2k-live-output \
  --alignment-packet /path/to/phase2k-downstream-alignment-packet-v1-finalized.json \
  --alignment-summary /path/to/phase2k-downstream-alignment-summary-v1.json \
  --decision CONTEXTUAL_POLISH_VALIDATED \
  --diagnosis MIXED \
  --note "Human interpretation bound to the measured evidence." \
  --output /path/to/downstream-comparison-v2.json
```

## Downstream comparison v2 (evidence-bearing closeout contract)

Closeout decisions are accepted only from a canonical
`phase2k-downstream-comparison-v2` envelope.  The comparison is one content
hash envelope: `content_sha256` is the canonical hash over every other
top-level field, and the top level contains exactly:

- `schema_version` — `phase2k-downstream-comparison-v2`;
- `content_sha256` — canonical hash over all other top-level fields;
- `comparison_complete` — `true`;
- `dataset_binding` — `phase2k_records_sha256` (must match the records
  artifact), `finalized_human_packet_sha256`, `human_summary_sha256`
  (canonical hash of the exact generated summary), completed
  transformation-audit hash (live builds), sorted window-ID hash, window
  count, and `human_review_gate_status: PASSED`;
- `semantic_target_contract` — contract version/hash, target count, and
  boundary rule shared by every raw/polished cell;
- `architectures` — `generative` (Phase 2F semantic IR) and `discriminative`
  (Phase 2H endpoint scoring), each with frozen semantic contract,
  model/scorer config, evaluation contract, and distinct raw/polished
  input-adapter hashes, plus raw/polished cells and recomputed deltas;
- `decision` — one of the final closeout statuses;
- `diagnosis` — one of the preregistered bottleneck diagnosis values;
- `note` — nonempty human interpretation.

Every raw/polished cell contains exactly one row per dataset window with
`true_positive_count`, `false_positive_count`, `false_negative_count`,
`output_count`, `provenance_valid_count`, `abstained`, and `output_sha256`.
Per-window invariants are enforced (`tp + fn = target_count`,
`tp + fp = output_count`, `provenance_valid_count <= output_count`,
nonnegative integer counts), and raw/polished cells across both
architectures must share identical ordered window IDs and per-window target
counts whose sum equals the semantic target contract.

The pipeline recomputes precision, recall, F1, unsupported rate,
provenance-valid rate, and abstention rate from those rows, recomputes every
raw-vs-polished delta, and fails closed if any count, rate, delta, hash, or
dataset binding does not reconcile.  No arbitrary metric-to-diagnosis
threshold is applied: the decision and diagnosis remain human empirical
interpretation, now bound to exact measured evidence.

The v2 contract is a measurement contract, not an evaluator.  It does not
run downstream models and it never fabricates rows: architecture-specific
rerun artifacts and per-window rows must still be produced and supplied
after human review passes.  Phase 2K also does not modify Phase 2J artifacts
or claim that Phase 2J's original exact-boundary result changed.

An already-built v2 JSON can be checked against a finalized output directory
with:

```bash
.venv/bin/python scripts/validate_phase2k_downstream_comparison.py \
  --output-dir /path/to/phase2k-output \
  --downstream-comparison /path/to/downstream-comparison-v2.json
```

## Closeout gate

Closeout is:

- `WAITING_FOR_HUMAN_REVIEW` until human reviews and (for live builds)
  transformation audits are complete and the human review gate is `PASSED`;
  a `FAILED` gate keeps the phase non-closed and rejects any downstream
  comparison;
- `WAITING_FOR_DOWNSTREAM` once the review gate passes but no valid
  downstream comparison has been supplied;
- a final status only after a valid `phase2k-downstream-comparison-v2` whose
  `decision` matches `--closeout-decision`.

Final Notion statuses are restricted to
`CONTEXTUAL_POLISH_VALIDATED`, `CONTEXT_ALONE_SUFFICIENT`,
`POLISH_UNSAFE_OVER_RECONSTRUCTING`,
`NO_MATERIAL_REPRESENTATION_GAIN`, and `INCONCLUSIVE`, and are only accepted
after both inputs are complete. The closeout artifact includes the exact
count-report skeleton plus the validated downstream comparison (evidence,
metrics, deltas, diagnosis); missing values stay null.

## Validation

Focused tests:

```bash
.venv/bin/python -m pytest \
  tests/test_phase2k_contextual_reconstruction.py \
  tests/test_build_phase2k_reconstruction_cli.py \
  tests/test_phase2k_contracts.py \
  tests/test_phase2k_downstream_alignment.py \
  tests/test_phase2k_downstream_rerun.py -q
```

No-provider build/validate:

```bash
.venv/bin/python scripts/build_phase2k_reconstruction.py \
  --manifest data/phase2j/window-selection-manifest-v1.json \
  --reviewed-packet data/phase2j/reviewed-endpoint-annotation-packet-v1.json \
  --doc docs/phase2j-independent-source-replication.md \
  --db /home/bphan944/PersonalProjects/videoSorter-homework-archive/videos.db \
  --output-dir /tmp/phase2k-validation-output

.venv/bin/python scripts/build_phase2k_reconstruction.py \
  --manifest data/phase2j/window-selection-manifest-v1.json \
  --reviewed-packet data/phase2j/reviewed-endpoint-annotation-packet-v1.json \
  --doc docs/phase2j-independent-source-replication.md \
  --db /home/bphan944/PersonalProjects/videoSorter-homework-archive/videos.db \
  --output-dir /tmp/phase2k-validation-output --validate-only
```

Relative and absolute input spellings produce identical hashes.

## Non-goals

No Phase 2J edits or result changes; no strategic ontology; no matchup
answers/fingerprints; no actual human-score fabrication; the reconstruction
builder does not execute the downstream generative/discriminative rerun
inside Phase 2K. The downstream rerun module is also downstream only: it does
not alter mechanical cleanup or perform semantic extraction during Pass 1.
Those reruns are required after the post-review alignment gate passes, and
the v2 comparison remains a required external evidence input, never
fabricated.
