# Phase 2I — UD / Syntactic Feature Ablation

## Final status

```text
SYNTACTIC SIGNAL FOUND
```

Exactly one next justified architecture intervention:

```text
EXPAND THE REVIEWED CANDIDATE-LEVEL DATASET WITH SUBSTANTIALLY MORE
INDEPENDENT SOURCE WINDOWS, PRESERVING GROUP-LEVEL SOURCE ISOLATION.
```

Do not begin that intervention, semantic edges, candidate-bound redesign, or
Phase 3 without a new explicit goal.

## Central answer

Yes, bounded Universal Dependencies evidence exposed useful held-out endpoint
signal that the frozen geometry + lexical Feature Set B did not. The signal is
model-dependent: Logistic C regressed, while LightGBM C materially improved
sparse threshold discrimination and broad ranking efficiency across all five
held-out windows.

This is proof of syntactic signal on a five-window micro-benchmark, not proof
of production generalization or a production classifier choice.

## Frozen experiment contract

- Bronze windows: 5, unchanged.
- Candidate rows: 16,624, unchanged.
- KEEP: 33; DROP: 16,591.
- Reviewed endpoint coverage: 33/33.
- Splits: the exact five Phase 2H grouped leave-one-window-out folds.
- Threshold: 0.5.
- Seed: 20260817.
- Models: the frozen Phase 2H Logistic B and LightGBM B configurations.
- New cells only: Logistic C and LightGBM C.
- Feature C: the exact inherited B matrix prefix plus deterministic UD/syntax
  columns.
- No LLM, generative selector, candidate changes, SRL, role model, semantic
  edge model, graph decoder, or ontology work.

The pre-training immutability audit compared every window and candidate in
exact order against the archived Phase 2H run-1 tables, including candidate
IDs, text, spans, offsets, labels, exclusions, ambiguity, generator metadata,
gold metadata, Bronze locks, and catalog hashes. It passed with no problems.

## Parser and alignment

Parser:

- Stanza 1.14.0;
- English EWT;
- processors: `tokenize,mwt,pos,lemma,depparse`;
- CPU;
- local assets only;
- `DownloadMethod.NONE` during evaluation;
- no parser tuning on benchmark labels.

Locked parser-asset manifest:

```text
ee9b1a3a22e29ac0ddafcafbe00ef742c803094014cccd0e1d2a43b3f38ae357
```

Candidate alignment over all 16,624 rows:

| Status | Count |
|---|---:|
| EXACT | 597 |
| TOKEN_ALIGNED | 16,027 |
| PARTIAL_BOUNDARY | 0 |
| UNALIGNED | 0 |
| AMBIGUOUS | 0 |

There were zero objective parser/alignment errors. Diagnostic ambiguity such
as multiple candidate heads remained explicit evidence for overcomplete spans;
it was not treated as parser blame or as a hard endpoint rule. Bronze text,
candidate IDs, spans, and offsets were never changed.

## Model configuration

Logistic C reused the Phase 2H Logistic B contract:

- binary L2 logistic regression;
- `C=1.0`;
- `solver=lbfgs`;
- `class_weight=balanced`;
- `max_iter=300`;
- fixed seed.

LightGBM C reused the Phase 2H LightGBM B contract:

- binary objective;
- 120 estimators;
- 7 leaves;
- max depth 3;
- minimum child support 20;
- learning rate 0.1;
- balanced class weights;
- deterministic row-wise training;
- fixed seed and two worker threads.

All B token vocabularies, syntax categorical vocabularies, scaling, class
weights, and training-derived statistics were fit on the four training
windows only. The held-out window was transformed once and never used for
threshold selection or model tuning.

## Aggregate results

### Logistic: B versus C

| Metric | Logistic B | Logistic C | Delta C−B |
|---|---:|---:|---:|
| Precision | 8.772% | 5.882% | -2.890 pp |
| Recall | 30.303% | 18.182% | -12.121 pp |
| F1 | 0.13605 | 0.08889 | -0.04717 |
| Average precision | 0.081275 | 0.049788 | -0.031487 |
| ROC AUC | 0.939701 | 0.929188 | -0.010513 |
| Recall@10 | 6/33 | 4/33 | -2 endpoints |
| Median gold rank | 92 | 89 | -3 |
| Mean gold rank | 208.12 | 236.67 | +28.55 |
| Selected | 114 | 102 | -12 |

Logistic C is not a successful syntax result. Its tiny median-rank improvement
does not compensate for worse recall, precision, F1, AP, AUC, recall@10, and
mean rank.

### LightGBM: B versus C

| Metric | LightGBM B | LightGBM C | Delta C−B |
|---|---:|---:|---:|
| Precision | 2.757% | 7.692% | +4.935 pp |
| Recall | 33.333% | 36.364% | +3.030 pp |
| F1 | 0.05093 | 0.12698 | +0.07606 |
| Average precision | 0.050952 | 0.041251 | -0.009700 |
| ROC AUC | 0.932765 | 0.950458 | +0.017693 |
| Recall@10 | 4/33 | 4/33 | 0 |
| Median gold rank | 157 | 96 | -61 |
| Mean gold rank | 226.24 | 170.09 | -56.15 |
| Selected | 399 | 156 | -243 |

LightGBM C recovered 12/33 gold endpoints with 144 false positives. LightGBM
B recovered 11/33 with 388 false positives. Syntax therefore raised recall
slightly while removing 62% of selected candidates and nearly tripling
precision. The result is not uniformly better: pooled AP fell, recall@1/3/5
fell, and recall@10 was unchanged.

## Per-window LightGBM evidence

| Held-out window | Precision B→C | Recall B→C | F1 B→C | AP B→C | AUC B→C | Selected B→C | Median rank B→C |
|---|---:|---:|---:|---:|---:|---:|---:|
| mid-push-prevents-side-collapse | 2.50%→4.00% | 60.0%→40.0% | .048→.073 | .0555→.0671 | .9228→.9365 | 120→50 | 103→53 |
| push-poke-wave-crash | 1.92%→13.04% | 20.0%→60.0% | .035→.214 | .0265→.1093 | .9556→.9864 | 52→23 | 78→16 |
| sweeper-limits-mid-play | 2.06%→7.41% | 28.6%→28.6% | .038→.118 | .0231→.0385 | .9190→.9223 | 97→27 | 272→147 |
| unwarded-bush-hook-risk | 2.60%→7.89% | 18.2%→27.3% | .045→.122 | .0635→.0468 | .9396→.9502 | 77→38 | 171→182 |
| wave-reset-after-kill | 5.66%→11.11% | 60.0%→40.0% | .103→.174 | .2813→.1047 | .9271→.9668 | 53→18 | 52→96 |

The key consistency evidence is not one lucky fold:

- precision improved in 5/5 windows;
- F1 improved in 5/5;
- ROC AUC improved in 5/5;
- selected candidates decreased in 5/5;
- mean gold rank improved in 5/5;
- median gold rank improved in 3/5;
- AP improved in 3/5;
- recall improved in 2/5, tied in 1/5, and fell in 2/5.

That pattern establishes held-out syntactic discrimination while also showing
that top-of-list ranking and calibration remain unstable.

## Are syntax features actually used?

Yes.

LightGBM syntax features contributed between 31.5% and 83.1% of total gain in
each fold. The strongest aggregate syntax features included:

- external-governor count;
- parser-word count;
- crossing incoming dependency context;
- internal `nsubj:VERB:PRON` context;
- clause-marker count;
- boundary-head count;
- internal predicate subject;
- scope governor outside the candidate.

The inherited `contained_by_count` geometry feature remained the strongest
single LightGBM feature, so syntax did not eliminate the overcomplete-candidate
geometry problem. It added substantial information alongside it.

Logistic C also used syntax. Strong negative syntax coefficients included
pronoun-governor `ccomp`, internal `nsubj:VERB:PRON`, and external governors
such as `know` and `have`. Strong positive coefficients included oblique
presence, subtree fraction, external auxiliaries, and verb-initial structure.
The coefficient pattern confirms that syntax can suppress some pronoun/action
distractors, but the linear model could not combine the evidence reliably.

## Error changes

LightGBM C reduced total classification errors by 245 relative to LightGBM B.
False-positive taxonomy changes included:

| Error code | B | C | Delta |
|---|---:|---:|---:|
| DISCOURSE_FILLER | 29 | 4 | -25 |
| GENERIC_ACTION_DISTRACTOR | 64 | 30 | -34 |
| GENERIC_ENTITY_DISTRACTOR | 57 | 20 | -37 |
| OVERLAPPING_LONGER_SPAN | 53 | 17 | -36 |
| OVERLAPPING_SHORTER_FRAGMENT | 55 | 20 | -35 |
| WRONG_CUE_PRIOR | 11 | 3 | -8 |
| PRONOUN_DISTRACTOR | 27 | 28 | +1 |
| GOLD_RANKED_LOW | 22 | 21 | -1 |
| PARSER_FEATURE_ERROR | 0 | 0 | 0 |

Syntax materially reduced generic-action, generic-entity, filler, cue-prior,
and both overlap error families for LightGBM. It did not solve pronoun
distractors. Logistic C reduced generic actions and pronouns but increased
longer-overlap errors and threshold misses; this matches its aggregate
regression.

No non-maximum suppression or candidate pruning was added. The overlap result
is diagnostic evidence only.

## Seven endpoints missed by every Phase 2H cell

All seven remained below threshold. Under LightGBM C:

- 5/7 moved upward in rank;
- 2/7 moved downward;
- the largest upward moves were 437, 275, 213, 125, and 71 positions;
- no movement was attributable to an objective parser/alignment error.

Representative upward movements:

- `pull the wave up again`: rank 715 → 278;
- `run into Tower and just die`: 458 → 183;
- `win level one`: 510 → 297;
- `around mid`: 272 → 147.

Syntax therefore introduced new ranking information, but not enough to recover
the hardest endpoints at threshold 0.5.

## Memorization and micro-benchmark limitation

Both models show severe train/held-out gaps. Per-fold training AP is roughly
0.89–1.00 for Logistic C and 0.82–1.00 for LightGBM C, while held-out AP is
far lower. The benchmark has only 22–28 positives in each training fold, and
syntax vocabularies have 132–187 held-out OOV values per fold.

The result therefore proves a useful information source, not stable corpus
generalization. It does not justify choosing LightGBM as a production
architecture, tuning it further, adding edges, or moving to Phase 3.

## Reproducibility

Implementation commit:

```text
9a88c3aa6e2eacbbcbe279bcbbbfe82495c25264
```

Key locks:

- benchmark content SHA-256:
  `a17674b6e2c491f0d7a1600dde0cfb8cc533d1d17db8633d8d94b2de9a57c1dd`;
- Phase 2H run-1 archive SHA-256:
  `22aaab162f6122691f577bc95746a0b7b1da9834706766b746a29737a5e46380`;
- Phase 2H aggregate SHA-256:
  `3a890de5f429056bae9d9932ce7f1985d9315e20655b4239be5acee4174edee2`;
- Phase 2I definition SHA-256:
  `265327b332942aa8a922b1bcb4960c4492692882a4960c5072b58af4b5646f1f`;
- parser assets manifest SHA-256:
  `ee9b1a3a22e29ac0ddafcafbe00ef742c803094014cccd0e1d2a43b3f38ae357`.

Runtime versions:

```text
Python 3.10.12
scikit-learn 1.7.2
LightGBM 4.7.0
NumPy 2.2.6
SciPy 1.15.3
Stanza 1.14.0
Torch 2.11.0
```

Two clean official artifacts matched on every deterministic input, parser
table, candidate score, rank, metric, delta, diagnostic, importance, and
content hash. `created_at` and hashes transitively dependent on it differ by
design.

Retained archives:

| Artifact | Archive SHA-256 | Aggregate content SHA-256 |
|---|---|---|
| `data/phase2i_artifacts/phase2i-syntax-features-run1.tar.gz` | `ed1c489c8ce273adb59b6321017d03c89015c8afad2f3e3e6cade813458cc4ad` | `f83282d49730b6c5bcb7d6dcfa7db7c18b8c9140633da1580c9c25398e8f45f3` |
| `data/phase2i_artifacts/phase2i-syntax-features-run2.tar.gz` | `6fd83bf9f0bfafbf58f43eac3b3203fc2a1ac7e0292bf6fad52b36f6fe89afb5` | `45046361130a59717e11bd843451c62d281ad04e50fdc4471f80ec0fd1933a8f` |

## Validation and independent review

- Fresh independent post-fix review: `ACCEPT`, no substantive blocker.
- Phase 2I syntax + CLI: 74 tests passed.
- Full endpoint/artifact invocation: 49 tests and 8 subtests passed; three
  assertions expected a later validation layer after strict JSON began
  rejecting the same malformed artifacts earlier.
- Targeted corrected assertions: 2 tests and 5 subtests passed. No production
  code changed after the full endpoint invocation.
- Frozen Phase 2H regression: 46 tests and 16 subtests passed.
- Additional manifest/Git/JSON/HEAD adversarial battery: 6 tests passed.
- Compilation, `git diff --check`, and `uv lock --check`: clean.
- Official artifacts self-verified and the strict comparison command passed.

## Interpretation

The central question is answered positively but narrowly:

```text
BOUNDED UD/SYNTACTIC STRUCTURE ADDS REAL HELD-OUT DISCRIMINATIVE SIGNAL,
MOST CLEARLY THROUGH NONLINEAR INTERACTIONS IN LIGHTGBM.
```

The signal is strong enough to justify collecting more independent labels,
not strong enough to justify production promotion. Dataset expansion is the
single next intervention because it directly addresses the measured
memorization/OOV limitation and is required before model choice, tuning,
candidate-bound redesign, richer semantic features, or edge learning.
