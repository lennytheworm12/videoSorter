# Phase 2J: Pre-Annotation Checkpoint (Model-Blind Source Selection)

## Status: STOP AT HUMAN ANNOTATION CHECKPOINT

```text
PRE_ANNOTATION_CHECKPOINT
```

Phase 2J has **no result status yet**. No Phase 2H/2I-style final disposition
is assigned before the two-pass human annotation and subsequent evaluation.
This checkpoint only produces a deterministic, model-blind selection and a
scorer-blind blank annotation packet. It runs **no scoring, no parser
inference, no Feature B/C, no Logistic/LightGBM, and no label/gold creation**.

**No valid new endpoint gold exists yet.** The 33 reviewed endpoints in the
legacy Phase 2G/2H/2I benchmark are regression-only; they are not reused as
gold for Phase 2J, and no synthetic/LLM/model gold was created. Stop after
producing the blank packet and reporting; do not evaluate.

## 1. Scope and source-of-truth contract

- Phase 2J changes only **independently recorded reviewed data**: the retained
  model-blind pool `data/semantic_ir_window_pool_v1.json`.
- The legacy five Phase 2H/2I windows (the Phase 2F legacy-failure benchmark)
  are **regression-only and excluded** from selection. Their upstream/source
  video IDs are loaded explicitly from
  `data/semantic_ir_legacy_manifest_v1.json` +
  `data/semantic_ir_legacy_failure_v1.json` and excluded even though the pool
  already excludes them.
- Source groups derive **exactly** from the pool window
  `upstream_source_id`:

  ```text
  source_group_id = "video:" + upstream_source_id
  ```

  Grouping never uses champion, role, timestamp, or window identity.
- The retained pool is one window per upstream video, is not gold, and is
  validated with the existing loader/validator
  (`pipeline.semantic_ir_pool.load_semantic_window_pool`).

## 2. Model-blind selection policy (preregistered, seed `20260817`)

Selection chooses exactly **30 windows from 30 distinct video source groups**
from the retained pool minus the legacy exclusions. Eligibility uses only:

- pool/source identity, exact offsets, and text hashes;
- ordinary metadata (video title, role, champion);
- the frozen lexical phenomenon tags;
- the diagnostic ASR punctuation band derived from the frozen
  `punctuation_poor` tag:

  ```text
  ASR band = PUNCTUATION_POOR  if "punctuation_poor" in phenomena
           = PUNCTUATED        otherwise
  ```

No model predictions, scores, ranks, uncertainty, syntax importance, labels,
or error taxonomy influence selection or appear in annotation-facing content.

Greedy marginal diversity preference (this is a preference, **not a claim of
corpus balance**; actual distributions are recorded in the manifest):

| Signal | Points | While selected count is |
| --- | ---: | --- |
| Frozen phenomenon present | +8 | per phenomenon with count < 2 |
| Role (metadata) present | +4 | per role with count < 2 |
| ASR punctuation band present | +2 | per band with count < 3 |
| Champion (metadata) not yet represented | +1 | per unrepresented champion |

Stable tie-break: `sha256("20260817:" + window_id)` ascending, then
`window_id` ascending. The selection is fully deterministic and rerunnable;
the manifest records the policy verbatim.

## 3. Outputs

Two official artifacts under `data/phase2j/`:

| File | Contents |
| --- | --- |
| `window-selection-manifest-v1.json` | selection policy, input/content hashes, legacy exclusions, 30 selected records (source group, window ID, upstream video ID, exact upstream offsets, Bronze text SHA, metadata, phenomena, ASR band, partition, candidate catalog count/hash), partition counts, actual diversity summaries, release gate `LOCKED`, candidate generator version, canonical content SHA-256 |
| `endpoint-annotation-packet-v1.json` | 30 blank annotation records (exact Bronze text + SHA, char length, whitespace-token table, blank endpoint list, Pass A/Pass B review records, ambiguity/exclusion controls, reviewer notes), rules, candidate catalog count/hash binding only, canonical content SHA-256 |

Selection totals (deterministic, seed `20260817`):

- 30 windows / 30 distinct video source groups; no source group crosses
  partitions.
- Partitions: **24 EXPANDED_DEV**, **6 FROZEN_REPLICATION** (Frozen release
  remains `LOCKED`). Frozen is a machine-integrity partition label; reviewers
  must not treat Frozen records differently.
- Legacy source exclusions: `3nKrtwpZ6sQ`, `uAdWuLPYn-0`, `z5IXabhMLzQ`.
- Candidate catalog: frozen Phase 2F generator
  `phase2f-mention-catalog-v3-cross-segment-ngrams-32`, **30,788** candidates
  total across the 30 windows (944–1,050 per window). The catalog is
  regenerated only to bind identity/count/hash and is never scored or exposed
  to reviewers.

Actual diversity summary from the official manifest:

- Phenomenon counts: `simple_fact` 2, `multiple_champions` 3,
  `multiple_abilities` 6, `omitted_actor` 8, `implicit_cause` 11,
  `contradiction` 12, `multi_sentence` 12, `cause_chain` 15,
  `comparison`/`resource_exchange`/`wave_reasoning` 16, `nested_condition` 17,
  `temporal`/`uncertainty`/`punctuation_poor` 18, `advice_explanation`/
  `explicit_cause`/`location_or_space` 19, `quantity` 23,
  `conditional`/`contrast`/`direct_advice` 25, `modality`/`negation` 26,
  `pronoun` 28.
- Roles: mid 10, adc 7, top 7, jungle 4, support 2.
- ASR bands: `PUNCTUATION_POOR` 18, `PUNCTUATED` 12.
- Champions: 30 distinct champions, one selected window each.

Content hashes (official build):

```text
manifest: 4d19b29db9bf7b31baca24b8b32ee1c082830bdf692309e2c65662cb313382b9
packet:   3f766b08696ed512063d999c75877001d77b03db136f8edae78e631e1725c62a
```

## 4. Two-pass reviewer instructions

The packet is a blank two-pass annotation instrument. Window statuses:
`UNREVIEWED`, `IN_REVIEW`, `REVIEWED`, `AMBIGUOUS`, `EXCLUDED`,
`ADJUDICATION_REQUIRED`.
Endpoint dispositions: `KEEP`, `AMBIGUOUS`, `EXCLUDED`,
`ADJUDICATION_REQUIRED`.

**Pass A — endpoint discovery.** For each window, mark every existing source
endpoint that must be preserved: exact character span, exact whitespace-token
span, the frozen source-semantic node type it plays (one of `ENTITY`,
`ABILITY_OR_RESOURCE`, `EVENT`, `ACTION`, `STATE`, `OUTCOME`, `QUANTITY`,
`TIME`, `LOCATION_OR_SPACE`, or undetermined), ambiguity state, and notes.
Offsets must slice the exact Bronze text; token boundaries must match
character boundaries; booleans are never valid offsets.

**Pass B — blinded audit.** After Pass A validates, the adjudication workflow
completes Pass B as an explicit human audit attestation: checking boundaries,
omissions, roles, duplicates, and ambiguity against Bronze. Pass B cannot
complete before a valid Pass A. A packet is **gold-eligible only when both
passes complete** on a `REVIEWED` window with every endpoint `KEEP`, no
ambiguity/exclusion flags, and every Pass B audit check true.

### 4.1 Exact two-pass field transitions

Pass A record fields:

| Pass A status | Valid preconditions | `reviewer` / `completed_at` | `endpoint_count` |
| --- | --- | --- | --- |
| `PENDING` | window `UNREVIEWED` | must be `null` | `0` |
| `IN_PROGRESS` | window `IN_REVIEW`; Pass B `LOCKED_AWAITING_PASS_A` | must be `null` (unsigned) | equals current endpoint list length |
| `COMPLETE` | window `IN_REVIEW` or final; Pass B `PENDING`/`IN_PROGRESS`/`COMPLETE` | non-empty strings required | equals endpoint list length |

Pass B record fields:

| Pass B status | Valid preconditions | `reviewer` / `completed_at` | `audit_checks` |
| --- | --- | --- | --- |
| `LOCKED_AWAITING_PASS_A` | Pass A not `COMPLETE` | must be `null` | all `false` |
| `PENDING` | Pass A `COMPLETE` | must be `null` | may be partial |
| `IN_PROGRESS` | Pass A `COMPLETE` | must be `null` (unsigned) | may be partial |
| `COMPLETE` | Pass A `COMPLETE`; every audit check `true` | non-empty strings required | all `true` |

Window status transitions:

| Window status | Valid pass state | Endpoints / flags | Gold-eligible |
| --- | --- | --- | --- |
| `UNREVIEWED` | Pass A `PENDING`; Pass B `LOCKED_AWAITING_PASS_A` | none; no flags | no |
| `IN_REVIEW` | Pass A `IN_PROGRESS` + Pass B `LOCKED_AWAITING_PASS_A`, **or** Pass A `COMPLETE` + Pass B `PENDING`/`IN_PROGRESS` | only `KEEP` dispositions; no ambiguity/exclusion/adjudication flags | **never** |
| `REVIEWED` | both passes `COMPLETE` | every endpoint `KEEP`; no flags; all audit checks `true` | yes |
| `AMBIGUOUS` | Pass A `COMPLETE`; Pass B open | ambiguity flag set; Pass B left open | no |
| `EXCLUDED` | Pass A `COMPLETE`; Pass B open | exclusion flag set; endpoints empty | no |
| `ADJUDICATION_REQUIRED` | Pass A `COMPLETE`; Pass B open | adjudication entries present; Pass B left open | no |

`IN_REVIEW` is the only clean intermediate status: it lets a completed,
signed Pass A result be saved before Pass B starts, and lets Pass B run
without prematurely reaching a final status. Rejected combinations include
`IN_REVIEW` with Pass A `PENDING`, with Pass B `COMPLETE`, or with any
non-`KEEP`/adjudication/ambiguity/exclusion state.

**Timestamps.** The retained pool has **no timestamp field**. Review
timestamps are therefore unavailable rather than inferred: `completed_at` is
recorded only at the moment a pass is marked `COMPLETE`, and pending or
in-progress passes keep `completed_at` `null`. No timestamp is derived from
file metadata or any pool field.

Rules:

- `AMBIGUOUS`, `EXCLUDED`, and `ADJUDICATION_REQUIRED` entries never silently
  become `KEEP`; they block gold eligibility and require the matching window
  status.
- Duplicate or overlapping endpoint annotations are rejected unless every
  overlapping pair is explicitly marked `adjudication_requested=true`; such a
  window must be `ADJUDICATION_REQUIRED` and is never gold-eligible.
- The annotation-facing packet must contain **no probabilities, scores,
  ranks, selected/predicted labels, syntax features/importances, model error
  taxonomy, or model suggestions**; recursive forbidden-key validation
  enforces this, and the frozen candidate catalog is bound by version, count,
  and hash only (never shown as suggestions).

### 4.2 Human-facing Notion workspace

The preferred review interface is
[Phase 2J — Human Endpoint Review Workspace](https://app.notion.com/p/3c0f8ba78bf38133b6e9c3b61e0db22e?pvs=204).
Its database contains exactly 30 pages, one per locked source window, with:

- exact Bronze text and an indexed whitespace-token view;
- an embedded endpoint-annotation database using inclusive token indices and
  dropdowns for type, ambiguity, disposition, and pass provenance;
- Pass A/Pass B properties and completion dates;
- the five Pass B audit checkboxes;
- filtered Pass A queue, Pass B audit, and status-board views.

The human-facing views omit partition labels, candidate counts, candidate
rows, and every scorer/model field. Reviewer-facing titles are neutral
(`NN · Bronze window`); champion, role, and video-title clues remain only in
the locked repository manifest for later diversity diagnostics. An endpoint
is an exact Bronze mention rather than a resolved identity: pronouns and
ability keys may remain unresolved, and the reviewer must not repair ASR or
infer names from metadata/game knowledge. Partially recoverable windows may
be marked `AMBIGUOUS`; wholly unreliable ASR/context-truncated windows must be
empty and `EXCLUDED`. Neither state is gold-eligible.

The review-queue data source is
`collection://57f4e89f-65ba-43e4-a0b1-511efb5691e7`; the endpoint-annotation
data source is `collection://74a9853b-87ac-4664-8378-688fc9a2db71`. Notion is
a review UI, not the final gold artifact: after review, Codex must fetch all
30 pages, compute character offsets from the accepted token spans, bind every
identity and Bronze field to the locked manifest, and run the packet finalizer
before candidate coverage or scoring.

### 4.3 Local span-selection review interface

The static Next.js route `/phase2j-review/` is the preferred Pass A span-entry
surface when reviewing locally. It reads the locked packet at build time and
serializes only record/window/source-group identity, exact Bronze text/hash,
character length, and token offsets. It does not serialize partition labels,
candidate metadata, champion/role/video-title metadata, model fields, or Sol
proposals.

The interface shows one window at a time. Dragging across Bronze snaps to
inclusive token boundaries and opens an endpoint-type picker; accepting a type
derives the exact character slice automatically. Duplicate and overlapping
spans fail closed. Browser-local autosave is bound to the packet content hash,
and JSON export/import provides a validated backup. The exported session is
review material, not canonical gold; Codex must validate and import it into the
locked packet after Pass A.

`CLEAN`, `AMBIGUOUS`, and `EXCLUDED` outcomes can all be human-signed with a
reviewer and date. Ambiguous/excluded windows require a note; excluded windows
must contain no endpoints. Pass B is completed separately in the
`/phase2j-adjudicate/` route as the explicit five-check human audit
attestation (section 6).

Build and serve the static UI from WSL:

```bash
cd apps/web
npm test -- --runInBand
npm run build
python3 -m http.server 3000 --bind 0.0.0.0 --directory out
```

Then open `http://localhost:3000/phase2j-review/` in the host browser.

### 4.4 Parallel Sol High review

A scorer-blind Sol High pass may run in parallel as **sealed navigation/audit
material only**. It must not see human answers, partitions, candidate rows, or
B/C outputs, and its endpoint proposals must remain hidden until human Pass A
is complete. Sol output is never gold and cannot sign either human pass.

After human Pass A, compare the sealed proposals with the human annotations.
The human reviewer must adjudicate every disagreement and possible omission
against Bronze, then complete the explicit five-check Pass-B audit
attestation in the `/phase2j-adjudicate/` route (section 6). The adjudication
plus that attestation **is** the Pass-B audit; Sol remains only a second
opinion. This avoids circularly defining the benchmark by an LLM while still
using Sol to reduce oversight.

The independent `gpt-5.6-sol` high-reasoning pass completed over all 30
windows and is retained outside the repository at
`/tmp/phase2j-sol-high-independent-review-v1.json`. It contains 338 proposed
mentions, remains sealed until human Pass A completes, and is explicitly
`NOT GOLD`. Independent structural validation confirmed all 30 identities,
every proposed token range and exact Bronze slice, and its canonical content
hash. File SHA-256 is
`6ef4ccbff8f9512b9119d314050acd5aaa87b927c37ee83372fcec92edd1cd8c`;
canonical content SHA-256 is
`8025d05c1bbe4f5b8c5c38d3689b96f69019087a3390e33cbfe98d2865ea0e53`.

## 5. Commands

```bash
# Build (deterministic; fails closed if preexisting outputs mismatch)
uv run python scripts/build_phase2j_annotation_packet.py

# Build with explicit inputs / output directory
uv run python scripts/build_phase2j_annotation_packet.py \
  --pool data/semantic_ir_window_pool_v1.json \
  --legacy-manifest data/semantic_ir_legacy_manifest_v1.json \
  --legacy-benchmark data/semantic_ir_legacy_failure_v1.json \
  --output-dir data/phase2j

# Validate-only (strict load + input/catalog re-verification, no writes)
uv run python scripts/build_phase2j_annotation_packet.py --validate-only

# Finalize a human-edited packet: recompute content_sha256, validate against
# the locked manifest, then atomically rewrite canonical pretty JSON
uv run python scripts/finalize_phase2j_annotation_packet.py

# Explicit inputs for finalization
uv run python scripts/finalize_phase2j_annotation_packet.py \
  --packet data/phase2j/endpoint-annotation-packet-v1.json \
  --manifest data/phase2j/window-selection-manifest-v1.json

# Check-only: existing content hash must already be correct; validates with no writes
uv run python scripts/finalize_phase2j_annotation_packet.py --check-only

# Focused tests
uv run pytest tests/test_phase2j_source_selection.py \
  tests/test_phase2j_annotation_packet.py \
  tests/test_build_phase2j_annotation_packet_cli.py \
  tests/test_finalize_phase2j_annotation_packet_cli.py
```

Do not commit or push until the parent technical lead reviews the checkpoint.

## 6. Post-Pass-A human-vs-Sol adjudication (REVIEW MATERIAL only)

**Historical state (superseded by section 7):** Human Pass A is complete (30 windows / 166 endpoints,
exported 2026-08-19 as `phase2j-review-session-3f766b08.json`), so the sealed
Sol High review may now be **revealed for explicit human adjudication only**.
Sol (`/tmp/phase2j-sol-high-independent-review-v1.json`, 30 windows / 338
proposals) is a navigation/audit **second opinion, never gold**. Nothing
auto-promotes Sol; the adjudication export remains `REVIEW_MATERIAL` until a
completed export is validated by the canonical importer (section 6.4). The
adjudication plus the explicit five-check human attestation **is** the Pass-B
audit; Sol remains only a second opinion. **Do not run model evaluation yet**
and do not treat any adjudicated export as canonical gold. No real reviewed
packet exists until a completed export is supplied and imported.

### 6.1 Generated adjudication packet

`data/phase2j/phase2j-adjudication-packet-v1.json` is a deterministic,
sanitized packet built from the locked annotation packet, the human session,
and the Sol review. It contains only the locked Bronze windows, sanitized
Human/Sol endpoint alternatives, connected-component classifications, input
hashes, and schema versions. It contains no reviewer identity, model ids,
scores, predictions, ranks, candidate data, or packet-internal fields.

Connected components use inclusive token intervals
(`left.start <= right.end && right.start <= left.end`). Official totals:

```text
components: 326 | exact agreements: 49 | type disagreements: 16
boundary disagreements: 87 | Sol-only: 174 | Human-only: 0
human endpoints: 166 | Sol proposals: 338
```

```text
packet content SHA-256: 13aaa1a9d6ecdba2d16b722109373e26494467e1b14d21d26458b93c8750015b
packet file  SHA-256: 074224d1e96e0a612d9bf9dcd5daccb5c1260c5c82b1ab76cdcd5770ebcb51a6
human file    SHA-256: 85437bfcc737ed71380f26581883f08bf4be4853d861ff055db642e338d1a471
Sol file      SHA-256: 6ef4ccbff8f9512b9119d314050acd5aaa87b927c37ee83372fcec92edd1cd8c
locked packet SHA-256: 3f766b08696ed512063d999c75877001d77b03db136f8edae78e631e1725c62a
```

### 6.2 How to build and use the adjudication route

```bash
# Generate (deterministic; fails closed if the existing packet differs)
uv run python scripts/build_phase2j_adjudication_packet.py

# Validate-only (no writes)
uv run python scripts/build_phase2j_adjudication_packet.py --validate-only

# Build the static web app from apps/web
cd apps/web && npm run build

# Serve the exported UI and open http://localhost:3000/phase2j-adjudicate/
python3 -m http.server 3000 --bind 0.0.0.0 --directory out
```

The route reads only `data/phase2j/phase2j-adjudication-packet-v1.json` at
build time. The UI shows one Bronze window at a time with neutral Human Pass A
and Sol overlays, per-component decisions (Human type/set, Sol type/set,
custom exact span/type, or drop), per-window `CLEAN`/`AMBIGUOUS`/`EXCLUDED`
outcomes with required notes, an explicit “Keep my Pass A choices” action,
progress, browser autosave, JSON import/export, reset, and a global Pass-B
audit attestation. Exports use schema `phase2j-adjudication-export-v2`;
component decision state remains localStorage-compatible, while the five audit
checks persist separately (`phase2j-adjudication-audit:v1`) and are restored
from a validated export. Exact-agreement components are pre-kept by default
but remain editable and drop-able, because agreement is evidence, not proof.
Export is blocked until every component in a `CLEAN` window is resolved,
`AMBIGUOUS`/`EXCLUDED` windows carry a required note (`EXCLUDED` clears all
endpoints), the resolved endpoint set is duplicate/overlap-free, and the
explicit five-check Pass-B attestation (boundaries, omissions, roles,
duplicates, ambiguity) is all true. The exported file is **REVIEW MATERIAL**,
not gold.

### 6.3 Focused tests

```bash
uv run pytest tests/test_phase2j_adjudication.py \
  tests/test_build_phase2j_adjudication_packet_cli.py \
  tests/test_phase2j_adjudication_import.py \
  tests/test_import_phase2j_adjudication_cli.py
cd apps/web && npx jest lib/phase2j-adjudication.test.ts
```

Current evidence: the new importer tests pass **18 tests** (13 in
`tests/test_phase2j_adjudication_import.py` plus 5 in
`tests/test_import_phase2j_adjudication_cli.py`). The adjudication web suite
passes **29 focused Jest tests**; the full web suite passes **70 tests** and
`tsc`/`build` pass. Combined Python totals across the full Phase 2J suite have
not been supplied and are not claimed here.

### 6.4 Canonical importer (fail closed)

`pipeline/phase2j_adjudication_import.py` +
`scripts/import_phase2j_adjudication.py` turn a completed
`phase2j-adjudication-export-v2` REVIEW MATERIAL export into a separate
reviewed canonical packet:

```bash
python3 scripts/import_phase2j_adjudication.py \
  --export /path/to/phase2j-adjudication-export-13aaa1a9.json

python3 scripts/import_phase2j_adjudication.py \
  --export /path/to/phase2j-adjudication-export-13aaa1a9.json \
  --validate-only
```

Use `python3` to avoid `uv` network sync. `--validate-only` checks the
existing reviewed output against the inputs without writing.

Inputs are the locked blank packet, the locked selection manifest, the
original human Pass-A session, the generated adjudication packet, and the
completed user export v2. The default output is a separate file,
`data/phase2j/reviewed-endpoint-annotation-packet-v1.json`; the blank packet
is never overwritten. The importer independently validates every input hash,
schema, record/window/component order, Bronze slice, component decision and
`resolved_by` semantics, derived endpoint fields, audit checks, and
overlap-freedom, then rebuilds canonical records from the locked blank packet
(rules, candidate binding, and `release_gate=LOCKED` preserved).

Mapping rules:

- Sol enters a reviewed packet only through an explicit human `KEEP_SOL_SET`
  or `CUSTOM` decision. `HUMAN`/`SHARED` provenance maps to `PASS_A`;
  `SOL`/`CUSTOM` maps to `PASS_B`. `UNDETERMINED` maps to canonical
  `node_type` `null` rather than inventing a type.
- `CLEAN` maps to window `REVIEWED` with Pass B `COMPLETE`; `AMBIGUOUS` maps
  to window `AMBIGUOUS` with Pass B `IN_PROGRESS` and is non-gold; `EXCLUDED`
  maps to window `EXCLUDED` with empty endpoints and is non-gold.
- At this historical checkpoint the release gate stayed `LOCKED` pending a
  completed export. The completed import and final coverage gate are recorded
  in section 7.

After import, check the reviewed output and assess sizing; only eligible
reviewed windows may proceed to candidate coverage.

## 7. Final coverage-gate closeout

The completed adjudication export was imported and deterministically
revalidated. It produced 30/30 reviewed windows and 311 gold-eligible
endpoints. The frozen candidate-coverage evaluator then found 263/311 exact
matches overall, 216/243 on Expanded DEV, and 47/68 on Frozen Replication.

Every one of the 48 misses differs from an existing frozen candidate only by
one trailing punctuation character (28 periods and 20 commas). This exposed an
unstated exact-boundary convention: whitespace-token review retained terminal
punctuation while candidate spans excluded it. No B/C scoring was run.

Final disposition: `ANNOTATION CONTRACT NOT STABLE`.

Exactly one next intervention is permitted: perform a versioned, scorer-blind
terminal-punctuation boundary correction, with human Pass-B re-adjudication
limited to the 48 affected endpoints, then regenerate exact candidate coverage
with the generator still frozen. See
[phase2j-independent-source-replication.md](phase2j-independent-source-replication.md).
