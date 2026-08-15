# Phase 2 Relation Extraction

## Scope

Phase 2 compiles explicit, source-grounded `insights` into derived
`StrategicRelation` objects. It does not automate champion fingerprints,
expand the ontology, synthesize principles, backfill the corpus, or add a graph
database.

```text
explicit insight IDs -> ExtractionPacket -> cheap structured model
-> schema/canonical validation -> accepted/review/rejected decision
-> condition-safe dedupe -> optional strategic_relations persistence
-> relation_evidence provenance
```

Raw evidence remains immutable. The compiler reads only evidence named in its
packet and existing ability metadata for champions explicitly found in that
evidence. It does not add unrelated retrieval context.

## Contract and Invariants

An accepted relation is the existing `StrategicRelation` model with
`data_version="strategic-relations-v0"` and the current ontology version.
Required normalized fields are subject/type, relation type, object/type,
condition/effect, concepts, provenance, patch sensitivity, confidence, and
evidence references.

- Valid relation types and concepts come from `core/ontology.py`; unknown
  values are rejected, never silently added.
- Champion names use the existing champion registry. Ability aliases come from
  existing `champion_abilities`; non-concept state/event/archetype nodes require
  explicit packet aliases.
- Conditions are part of the stable relation identity. Different conditions do
  not merge.
- Accepted automated provenance is only `source_claim` or
  `coach_supported_inference`; each accepted relation has packet evidence IDs.
- Confidence combines extraction confidence (55%), source evidence quality
  (35%), and canonicalization certainty (10%). The threshold is configurable;
  lower valid output is review-only. Invalid output is rejected.
- Persistence accepts only `accepted` compiler decisions and merges evidence
  references without losing provenance. Re-running the same accepted decision
  is idempotent.
- Manual Phase 1 relations remain valid and are available alongside automated
  relations in bounded strategic context retrieval.

## Operations

Dry run is the default and never writes:

```bash
LLM_PROVIDER=deepseek uv run python -m scripts.extract_relations \
  --db videos.db --insight-id 4807 --json-output /tmp/relation-dry-run.json
```

Use `--apply` only after inspecting the dry-run result. It persists accepted
relations only. The JSON includes the source evidence, raw model proposal,
canonical relation, confidence components, validation warnings, and persistence
action.

Inspect stored derived relations and source links:

```bash
uv run python -m scripts.inspect_relations --db videos.db --champion Thresh
uv run python -m scripts.inspect_relations --db videos.db --concept access
uv run python -m scripts.inspect_relations --db videos.db --evidence-id 4807
uv run python -m scripts.inspect_relations --review-file /tmp/relation-dry-run.json
```

Validate the small reference corpus without model calls, then optionally run
the configured model without persistence:

```bash
uv run python -m scripts.eval_relation_extraction
LLM_PROVIDER=deepseek uv run python -m scripts.eval_relation_extraction \
  --live --db videos.db --json-output /tmp/phase2-relation-eval.json
```

## Extending Vocabulary

Add a concept alias in `core/relation_normalization.py` only when it maps to an
existing `core/ontology.py` concept. Add entity aliases through source metadata
or a packet's explicit aliases. Add a relation type only after ontology review:
update `RELATION_TYPES`, canonical aliases, validation tests, prompt schema,
and documentation together. Phase 2 deliberately has no automatic vocabulary
expansion.

## Known Limits

The current evaluator is a small audit corpus and uses exact canonical
subject/type/object/condition comparison; prose effects are inspectable but not
identity. It is not a production backfill evaluator. Distinct conditional
relations remain separately stored; same-condition opposing relation types are
quarantined for review rather than persisted. Phase 2 does not resolve
strategic disagreements.
