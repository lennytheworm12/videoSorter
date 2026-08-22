# Phase 2J — Independent-Source Dataset Expansion & Syntactic Signal Replication

## Final disposition

`ANNOTATION CONTRACT NOT STABLE`

Phase 2J stopped at the preregistered candidate-coverage gate. No Feature B or
Feature C predictions were inspected, no Phase 2J parser/scorer run occurred,
and no claim about syntactic-signal replication is made.

## Independent-source dataset

The locked selection manifest contains 30 new windows from 30 independently
recorded `video:` source groups. The three legacy Phase 2H/2I source videos are
excluded. Source groups are isolated into 24 `EXPANDED_DEV` and 6
`FROZEN_REPLICATION` groups; no source appears in both partitions. Selection
preceded annotation and scoring and used only source metadata, preregistered
phenomenon/ASR strata, and deterministic seeded tie-breaking.

The completed scorer-blind two-pass review produced:

- 30/30 `REVIEWED` windows;
- 311 gold-eligible `KEEP` endpoints;
- 0 ambiguous or excluded windows;
- complete Pass A and Pass B state for every window;
- all five Pass B audit checks true.

The retained reviewed packet is
`data/phase2j/reviewed-endpoint-annotation-packet-v1.json`:

- canonical content SHA-256:
  `c239070e107e0848e8d26918d33ece5fa978f9ce48e0f43a2e65b67cd622365d`;
- file SHA-256:
  `149f6043f1547c9769ac24fcbe8c839195a9906a071c3accde866eff25a3362f`.

The 311 eligible endpoints exceed the staged 150-endpoint target, so no
additional model-blind source selection was required.

## Frozen candidate-coverage gate

The evaluator in `pipeline/phase2j_candidate_coverage.py` regenerates the
frozen Phase 2F mention catalog without changing its configuration, verifies
every per-window count/hash against both the manifest and reviewed packet, and
matches gold by exact local Bronze `(char_start, char_end)` only. It contains
no parser, feature, probability, rank, prediction, threshold, or model path.

The retained coverage artifact is
`data/phase2j/candidate-coverage-v1.json`:

- canonical content SHA-256:
  `1ac837aae4a4411837d2277f23ce613f531ffb5dec57e449e0a7fb4c14a2daa2`;
- file SHA-256:
  `8c08a5e5efc674c7fb1540cc11dc45e3864f66bea70f9a0213cab582d6fa1b29`;
- frozen candidate rows: 30,788.

Exact coverage was:

| Partition | Covered | Gold | Exact recall |
|---|---:|---:|---:|
| Expanded DEV | 216 | 243 | 88.889% |
| Frozen Replication | 47 | 68 | 69.118% |
| Aggregate | 263 | 311 | 84.566% |

The legacy five-window benchmark had 33/33 exact coverage. The expanded result
therefore failed the exact-span gate before scorer evaluation.

## Root cause

All 48 misses occur in 12 punctuated source windows. Every miss has a frozen
candidate with:

- the same semantic text;
- the same start offset;
- an end offset exactly one character before the reviewed gold end.

The 48 extra terminal characters are 28 periods and 20 commas. There are no
no-overlap or semantic-discovery misses. The annotation workflow used
whitespace-token boundaries and retained terminal punctuation, while the
frozen candidate generator excluded it. The gold contract did not specify
which convention governed exact endpoint boundaries.

Calling this a semantic candidate-generation failure would be misleading: the
semantic spans exist, but the exact gold/candidate boundary contract is not
stable. Scoring B/C on the current labels would confound selection quality
with this unresolved boundary rule, so Phase 2J correctly stopped before
Expanded DEV or Frozen Replication model evaluation.

## Independent review

An independent read-only `gpt-5.6-sol` high-reasoning audit reproduced the
manifest, source isolation, input/hash lineage, reviewed-packet eligibility,
coverage arithmetic, deterministic regeneration, and punctuation-only
diagnosis. It confirmed that no Phase 2J model-scoring artifact exists and
recommended the same disposition. Confidence was high, with the limitation
that the current worktree is uncommitted and therefore the new artifacts do
not yet have clean-commit provenance.

Repository state recorded at closeout:

- HEAD: `1f52e6ba3ab66f2190ca0c97e27b63c5e092c984`;
- worktree: dirty/uncommitted;
- candidate-coverage focused suite: 15 tests passed;
- deterministic artifact `--validate-only`: passed.

## Exactly one next intervention

Perform a versioned, scorer-blind terminal-punctuation boundary correction,
with human Pass-B re-adjudication limited to the 48 affected endpoints, then
regenerate exact candidate coverage while keeping the candidate generator
frozen.

Do not begin that intervention without a new explicit goal. Do not run B/C,
tune models, alter Feature B/C, or modify candidate generation first.
