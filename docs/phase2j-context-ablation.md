# Phase 2J — Source-Grounded Semantic-Extraction Ablation (v2)

## Status and scope

Phase 2J context ablation is an isolated, model-free harness that prepares
and validates a controlled 10-example experiment:

  A  GPT-5.6 Sol receives only the isolated Bronze target plus the exact
     byte-identical extraction instructions.  No full transcript, no
     metadata, no vocabulary, no source/video identity, and no surrounding
     text.

  B  GPT-5.6 Sol receives the exact same Bronze target, the full archived
     transcript with the target's character offsets, useful ordinary
     metadata only, and the League champion/ability vocabulary, plus the
     exact same byte-identical extraction instructions.

Both conditions directly extract actors, ability/resource references,
event, condition, advice/action, consequence, uncertainty, and supporting
source ranges.  There is no mechanical cleaning, contextual rewriting,
semantic polish, or strategic abstraction anywhere in the pipeline.

Implementation status as of 2026-08-20: the source-grounded v2 harness is
implemented and tested.  Selection, extraction instructions, and the A/B
condition payloads are generated directly from the frozen manifest, the
read-only transcript DB, and the lexical vocabulary — no captions or
timestamps are required or invented.  The default build reaches
`ready_for_sol` with no model calls.

The harness is isolated from Phase 2K: it never imports or edits Phase 2K
code/data, never runs Phase 2J scoring, and never touches Phase 2K
implementation or artifacts.

## Frozen authoritative inputs

- `data/phase2j/window-selection-manifest-v1.json`
- `data/phase2j/reviewed-endpoint-annotation-packet-v1.json`
- the read-only SQLite transcript DB supplied by CLI, normally
  `/home/bphan944/PersonalProjects/videoSorter-homework-archive/videos.db`
- the Phase 2K lexical vocabulary snapshot
  `data/phase2k_support/league_lexical_vocabulary_v2.json` (schema
  `phase2k-league-lexical-vocabulary-v2`) as the lexical half of the B
  vocabulary.

The DB `videos.transcription` field is the archived full transcript plain
text.  Transcript text is read directly from the read-only DB; full
transcripts must byte-for-byte match the frozen manifest
`upstream_content_sha256` hashes.

## Case selection (frozen, manifest tags only)

The 10 controlled cases are frozen from the frozen Phase 2J manifest tags
only.  The preregistered difficulty weight table is:

| Tag | Weight |
| --- | --- |
| punctuation_poor | 3 |
| omitted_actor | 3 |
| pronoun | 2 |
| multiple_abilities | 2 |
| multiple_champions | 2 |
| nested_condition | 2 |
| cause_chain | 2 |
| uncertainty | 2 |
| contradiction | 1 |
| implicit_cause | 1 |
| explicit_cause | 1 |
| conditional | 1 |
| temporal | 1 |
| contrast | 1 |
| advice_explanation | 1 |
| resource_exchange | 1 |
| wave_reasoning | 1 |
| multi_sentence | 1 |

Score = sum of the weights of the frozen manifest tags present in the
window.  Sort is descending score, then descending total phenomenon count,
then original manifest selected order; the first 10 are selected.  The
selection artifact records score, contributing tags, selection rank, and
all input hashes per case.

Selection validation independently recomputes the canonical top-10 from the
frozen manifest plus the read-only DB — never from the artifact's own
cases — and requires exact equality.  A self-rehashed tampered case cannot
pass.

Explicitly excluded selection signals: Phase 2K results, model
predictions, human semantic outputs, endpoint counts, partition, and gold
labels.

## A/B payload boundaries

- **A payload** (model-visible) contains exactly: `schema_version`,
  `condition`, `case_id`, `selection_rank`, `target` (`bronze_text`,
  `bronze_text_sha256`, `bronze_char_length`), the byte-identical
  `instructions` object and `instructions_sha256`, and `content_sha256`.
  No transcript, metadata, vocabulary, video identity, title, URL,
  retrieval-handle hashes, or surrounding text.
- **B payload** (model-visible) contains the exact same `target` object
  (byte-identical to A), the full `transcript`, `target_char_start` /
  `target_char_end` locating the Bronze target in that transcript,
  ordinary metadata (`video_title`, `champion`, `role`, `rank`, `game`),
  the B vocabulary plus `vocabulary_sha256`, the byte-identical
  `instructions` object and `instructions_sha256`, and `content_sha256`.
  No video URL appears in any model-visible payload.
- Source identity provenance (`video_id`, `video_url`, `source_group_id`,
  `window_id`, full-transcript hash/length, target offsets, vocabulary
  hash) lives at the outer `payloads` artifact level under
  `provenance_by_case`, not inside the model-visible payloads.  Vocabulary
  provenance records the `champion_abilities` source and selection reasons
  without exposing source video identity.
- The B vocabulary is the lexical vocabulary v2 plus DB
  `champion_abilities` rows for champions named in metadata and any
  canonical champion names literally present (exact word boundary) in the
  full transcript.  No archetypes, fingerprints, strategic relations,
  labels, or Phase 2K generated bindings are included.
- The shared extraction instructions are a separate canonical object; both
  conditions embed the byte-identical object and bind to its exact
  SHA-256.  `validate_payloads_artifact` rebuilds the exact canonical
  payloads from the frozen manifest, DB, instructions, and vocabulary and
  requires exact equality, so self-rehashed tampering fails.

## Source grounding (no timestamps)

Citation grounding is exact source grounding only:

- Every `source_reference` carries a `quote` and an integer
  `source_range` `{"char_start", "char_end"}`.
- The quote must byte-for-byte equal the supplied source slice at that
  range: `source[char_start:char_end] == quote`.
- Condition A offsets are into the supplied Bronze target; condition B
  offsets are into the supplied full transcript.
- The eighth semantic field is `supporting_source_ranges`; its items carry
  an item-level `source_range` contained within the union of their cited
  references.
- The human review field formerly named `timestamp_grounding` is
  `source_grounding` with values `GROUNDED | PARTIAL | UNGROUNDED |
  NOT_APPLICABLE`.

Offsets are computed mechanically, never by the model.  The model-visible
intermediate Sol response returns each source reference as an exact
contiguous `quote` plus a zero-based `occurrence_index` (counted among all
exact, non-overlapping substring matches in the condition source).  The
deterministic importer resolves each occurrence to its byte-exact
`[char_start, char_end)` range and derives the item-level
`supporting_source_ranges` range as the minimal bounding range of the
resolved references.  No timestamp or caption requirement remains anywhere
in the harness.

## Sol execution and import runner

`scripts/run_phase2j_context_ablation_sol.py` executes exactly 20
independent GPT-5.6 Sol calls (10 frozen cases x conditions A/B) against
the already validated DB-only v2 payload artifact and imports the raw
responses into the standard extraction-outputs bundle consumed by the
`build --outputs` path.

Every call uses the identical wrapper prompt and configuration
(`--ephemeral --ignore-user-config --ignore-rules -m gpt-5.6-sol -c
model_reasoning_effort="high" -s read-only --skip-git-repo-check`), runs in
its own isolated temp workspace containing only the response schema and
output necessities, receives stdin consisting of the canonical wrapper
prompt plus the canonical JSON serialization of ONLY the inner condition
payload, and must return a strict intermediate response matching the
canonical JSON Schema (closed objects, exact quotes plus zero-based
occurrence indexes, no model-supplied character offsets).  Outer
provenance, repo paths, the sibling condition, sealed mapping, and other
cases are never exposed.  No full-history or session continuation is used
across calls.

```bash
# Optional: validate frozen artifacts and write the intermediate schema.
.venv/bin/python scripts/run_phase2j_context_ablation_sol.py schema \
  --output-dir data/phase2j_context_ablation

# Execute all 20 independent calls (resumable; default run dir
# data/phase2j_context_ablation/sol_run_v2).
.venv/bin/python scripts/run_phase2j_context_ablation_sol.py run \
  --output-dir data/phase2j_context_ablation

# Deterministically import all 20 raw responses and write the standard
# extraction outputs bundle (phase2j-context-ablation-extraction-outputs-v2.json).
.venv/bin/python scripts/run_phase2j_context_ablation_sol.py import \
  --output-dir data/phase2j_context_ablation
```

Resumability and safety:

- Raw intermediate responses are stored one file per case/condition under
  the configured run directory (default `data/phase2j_context_ablation/
  sol_run_v2`) with atomic writes.
- An existing valid raw response for the exact payload/prompt/config is
  reused.  A mismatched or malformed existing result fails closed; only
  `--force` replaces the exact per-call artifact in the configured run
  directory (valid completed results are still reused).
- A subprocess failure leaves its log and temp workspace as evidence and
  exits nonzero.  `--retries N` may only repeat the exact same prompt and
  config.  `--max-workers` defaults to 2 and is bounded to 1..4.
- The run manifest records the requested model (`gpt-5.6-sol`), reasoning
  effort (`high`), codex CLI version, exact argv template, wrapper hash,
  intermediate schema hash, payload/prompt/raw-response hashes, and final
  output hashes, and rejects a manifest claiming any other model/config.
This is requested-model evidence only; the runner cannot
cryptographically prove backend identity.

## Multi-agent transport runner (multi_agent_v1)

`scripts/run_phase2j_context_ablation_multi_agent.py` is the audited,
additive transport for the same 20 Phase 2J context-ablation calls when the
workspace sandbox blocks `codex exec` (e.g. `ab.chatgpt.com`).  It makes no
model calls itself and does not claim codex argv/CLI execution.  The parent
spawns 20 fresh non-forked `multi_agent_v1` default agents with
`requested_model=gpt-5.6-sol` and `reasoning_effort=high`; each agent's
initial message is exactly the canonical wrapper prompt plus the canonical
JSON serialization of ONLY its inner condition payload, produced by the
`prompt` subcommand.

```bash
# Create the separate canonical 20-pending-call manifest (default run dir
# data/phase2j_context_ablation/sol_multi_agent_run_v2).
.venv/bin/python scripts/run_phase2j_context_ablation_multi_agent.py init \
  --output-dir data/phase2j_context_ablation

# Print the exact experiment user message for one call (stdout only).
.venv/bin/python scripts/run_phase2j_context_ablation_multi_agent.py prompt \
  --output-dir data/phase2j_context_ablation \
  --case-id p2ja:case:0001 --condition A

# Strict-parse, validate, and atomically record one staged response.
.venv/bin/python scripts/run_phase2j_context_ablation_multi_agent.py ingest \
  --output-dir data/phase2j_context_ablation \
  --case-id p2ja:case:0001 --condition A \
  --agent-id agent-01 --response /tmp/staged-response.json

# Validate the manifest and all completed raw responses.
.venv/bin/python scripts/run_phase2j_context_ablation_multi_agent.py status \
  --output-dir data/phase2j_context_ablation

# Require all 20 valid completed calls and assemble the standard outputs
# bundle (phase2j-context-ablation-extraction-outputs-v2.json).
.venv/bin/python scripts/run_phase2j_context_ablation_multi_agent.py import \
  --output-dir data/phase2j_context_ablation
```

Safety and audit properties:

- `init` writes a separate canonical manifest with exactly 20 pending calls
  and records the transport (`multi_agent_v1`), requested model
  (`gpt-5.6-sol`), reasoning effort (`high`), exact shared wrapper hash,
  intermediate schema hash, instructions/payload hashes, run dir, and a
  null `final_outputs`.
- `prompt` prints the exact byte string `SOL_WRAPPER_PROMPT + "\n\n" +
  canonical-inner-payload` with no other stdout text, for pending or
  completed calls alike; unknown or noncanonical calls fail closed.
- `ingest` strict-parses the staged response, validates it against the
  exact case/condition/payload via `validate_sol_intermediate_response`,
  atomically persists the raw bytes under the run dir, and atomically
  updates only that manifest call.  Any validation failure leaves the
  manifest and run dir untouched; replacing valid completed evidence
  requires `--force`.
- `status` validates the manifest, every completed raw response, and any
  recorded final outputs; `import` requires all 20 valid completed calls,
  deterministically imports them with
  `import_sol_intermediate_response`/`build_outputs_bundle`, and records
  and strictly validates the exact 20 by-call hashes plus output
  file/content hashes.  Tampering with the manifest, raw responses,
  outputs artifact, or by-call hashes fails closed.
- The recorded raw paths are always generated and validated inside the run
  directory; recorded paths that are absolute or escape the run dir are
  rejected.
- Timestamps are not part of this transport contract.  The manifest
  purpose discloses that the `multi_agent_v1` backend identity is
  requested/recorded but not cryptographically proven, and that the
  surrounding subagent system envelope is transport-provided while the
  experiment user message is the canonical wrapper prompt plus the
  canonical inner payload.

## Extraction output schema and validation

Every validated output carries `case_id`, `condition`, `payload_sha256`,
`instructions_sha256`, and the eight fields.  Every extracted item has:

- `item_id` (`{case_id}:{condition}:{field}:{NNNN}`, sequential)
- `extraction_text` (concise)
- `resolution_status`: `literal_explicit`, `context_resolved`,
  `vocabulary_supported`, or `unresolved`
- `source_references`: at least one reference with a byte-exact quote and
  integer character range
- `source_range` for `supporting_source_ranges` items

Validators reject: unknown/missing keys, wrong example order/IDs,
citations outside the supplied source, quotes that are not byte-for-byte
equal to the cited source slice, malformed/out-of-range source ranges,
duplicate item IDs, and leaked output fields.  Empty lists and
`unresolved`/uncertainty are allowed; unsupported guessing is forbidden.

## Blinded human review packet

The blank review packet is generated from validated A and B outputs.  Per
example/condition/field it requires:

- `correctness`: `CORRECT | PARTIAL | INCORRECT | ABSENT_CORRECTLY`
- `unsupported_inference`: `NONE | MINOR | MAJOR`
- `source_grounding`: `GROUNDED | PARTIAL | UNGROUNDED | NOT_APPLICABLE`
- `notes`

10 cases × 2 conditions × 8 fields = 160 review items.  The packet contains
one shared, condition-neutral full-transcript `source_evidence` entry per
case (including the target text/hash and its character offsets), and both A
and B items point to the identical shared evidence for their case, so a
human can verify context-resolved outputs and every cited quote.

Condition labels are blinded with deterministic private constants; the
review-item order is deterministically shuffled after construction so
condition ordering is not revealed.  No condition code and no blinding seed
appear anywhere in the reviewer-visible packet.  The label-to-condition
mapping is retained in a separate sealed artifact bound only by hash.
Mapping validation compares every entry against its packet item, requires
exactly one A and one B per case+field, and (given outputs/payloads)
recomputes the canonical presentation and output hashes so a self-rehashed
semantic tamper is rejected.

## Human attestation

Completed reviews must explicitly attest to human review:

- `reviewer_kind` exactly `"human"`
- `human_review_attested` exactly `true`
- a concise non-empty `attestation_statement`

Non-human or absent attestations are rejected, and the finalized packet
carries the attestation in `review_attestation` so the frozen artifact
makes the required human assertion explicit.

## Preregistered materiality decision

Before any model outputs, a field is a strict success iff:

- `correctness` in `{CORRECT, ABSENT_CORRECTLY}`
- `unsupported_inference` = `NONE`
- `source_grounding` in `{GROUNDED, NOT_APPLICABLE}`

There are 80 paired field judgments per condition (10 cases × 8 fields).
Full context is **MATERIAL** iff all four hold:

1. B strict-success fields − A strict-success fields ≥ 12
2. B strictly wins ≥ 4 cases by per-case strict-success count
3. A strictly wins ≤ 1 case
4. B has no increase in MAJOR unsupported-inference judgments vs A

Otherwise the decision is **NOT_MATERIAL**.  Finalization imports the
completed human reviews, recomputes every statistic deterministically, and
freezes a Sol comparison summary with input/output/review hashes.

## DeepSeek B gate

DeepSeek B remains gate-locked until a frozen MATERIAL Sol summary exists
**and** the full output directory authoritatively validates against the
frozen manifest, packet, DB, and vocabulary.  `finalize`,
`emit-deepseek-run`, and `import-deepseek-run` all run
`validate_output_directory` first, so a self-consistent fabricated MATERIAL
summary cannot unlock DeepSeek.  DeepSeek import rejects extra or missing
cases and substantively validates case identities, order, and outputs
against the run packet and payloads.

## CLI

```bash
python scripts/build_phase2j_context_ablation.py build \
  --manifest data/phase2j/window-selection-manifest-v1.json \
  --reviewed-packet data/phase2j/reviewed-endpoint-annotation-packet-v1.json \
  --db /home/bphan944/PersonalProjects/videoSorter-homework-archive/videos.db \
  --output-dir data/phase2j_context_ablation
```

Subcommands:

- `build` — freeze selection + extraction instructions + A/B payloads
  (`ready_for_sol`); with `--outputs`, also bind validated outputs and
  generate the blinded human review packet and mapping
  (`review_packet`).
- `validate` — deterministic revalidation of an output directory against
  manifest/packet/DB/vocabulary; fails closed on partial artifact
  combinations and validates any present build summary and DeepSeek
  artifacts.
- `finalize --reviews ... --frozen-at ...` — import completed human
  reviews (attestation required) and freeze the materiality summary.
- `emit-deepseek-run` / `import-deepseek-run` — gate-locked DeepSeek B
  run/import.

All CLI writes are atomic (same-directory temp file + `os.replace` with
cleanup on failure).

## Generated artifacts

`data/phase2j_context_ablation/` contains the frozen v2 artifacts:

- `phase2j-context-ablation-selection-v1.json`
- `phase2j-context-ablation-extraction-instructions-v2.json`
- `phase2j-context-ablation-condition-payloads-v2.json`
- `phase2j-context-ablation-build-summary-v2.json`

The obsolete caption-request/build-summary v1 artifacts were removed when
the timestamp/caption direction was replaced by direct DB transcript
source grounding.
