# Phase 2H: Discriminative Semantic Endpoint Scoring — Closeout Report

## Final status

```text
WEAK RANKING SIGNAL ONLY
```

Gate 1 failed and Gate 2 was not triggered. Initial implementation: commit
`65e7100` (`Add Phase 2H discriminative endpoint scoring`). The final clean
experiment, including the parser-taxonomy correction, ran at commit `a754991`.
Two immutable offline runs of `phase2h-endpoint-scoring-v1` completed there
(`repository_dirty: false`); run 1 is the authoritative artifact at
`/tmp/phase2h-endpoint-scoring-v1-final-run1`, and both runs are archived under
`data/phase2h_artifacts/`.

## 1. Scope and non-goals

Phase 2H is a candidate-level binary KEEP/DROP semantic endpoint scorer over
the exact exhaustive source-grounded candidate universe produced by the frozen
Phase 2F candidate generator (`phase2f-mention-catalog-v3-cross-segment-ngrams-32`),
using the same locked five-case Phase 2F/2G benchmark, the same 33 reviewed
gold endpoints, and the same labels as Phase 2G. It is fully offline and
deterministic:

- **No generative calls.** `no_llm: true`; no LLM/API requests, no provider
  bytes, no DeepSeek labels or predictions in any feature.
- **No syntax work.** Optional UD-syntax integration is explicitly out of scope
  for this first implementation.
- **No roles, edges, or graph work.** `no_semantic_edges: true`; no candidate
  pairs, relation classification, node typing, or downstream graph stages.
- **No changes to Phase 2F candidate generation or Phase 2G gold labels.** The
  candidate universe, benchmark, and labels are frozen.

The four fixed cells share identical grouped leave-one-window-out folds:

| Cell | Feature set | Model |
| --- | --- | --- |
| `logistic_A` | A: geometry/provenance (27 dense features) | Class-balanced L2 logistic regression |
| `logistic_B` | B: A + bounded lexical/cue features + sparse word 1–2 n-grams + boundary tokens | Class-balanced L2 logistic regression |
| `lightgbm_A` | A | Conservative deterministic LightGBM |
| `lightgbm_B` | B | Conservative deterministic LightGBM |

Fixed settings: KEEP threshold `0.5`; seed `20260817`; L2 logistic
(`lbfgs`, `C=1.0`, `class_weight=balanced`, `max_iter=300`, `tol=1e-4`);
conservative LightGBM (`n_estimators=120`, `num_leaves=7`, `max_depth=3`,
`min_child_samples=20`, `learning_rate=0.1`, `reg_alpha=0.001`,
`reg_lambda=1.0`, `deterministic=true`, `force_row_wise=true`, gain
importance).

## 2. Evaluation design and leakage controls

- **Folds:** 5-window grouped leave-one-window-out with `group_key=window`; each
  fold trains on the other four windows and scores the held-out window. Fold
  guard: never train/test a fold whose training windows have no positive
  examples.
- **Source overlap audit:** the five windows are disjoint source spans. Three
  come from the same source video at non-overlapping offsets, which is not
  span leakage but further limits the diversity of this micro-benchmark.
- **Fit scope:** every preprocessing statistic — scaler, sparse vocabulary,
  class weights, and model fit — is computed from training windows only. The
  B-cell fit-scope audit records actual held-out OOV token types absent from
  the fitted vocabulary (count, sorted list, deterministic SHA-256); fold 0
  reports 29 OOV token types with SHA-256
  `b97fa7c2ce76580cc4238dbe1eed8b4b67c483b5c2deca93d182be1e13dae5b5` and a
  training-only vocabulary of 599 terms.
- **Threshold and ranking:** fixed threshold `0.5`; 1-based ranks inside each
  held-out window; ties broken by descending score then deterministic
  candidate-catalog order (ascending start/end).
- **Prohibited features:** case IDs, source IDs, window IDs, candidate IDs,
  mention IDs, question text, gold roles/types, label-derived values, and
  DeepSeek labels/predictions. Identifiers and gold are retained only as
  provenance.
- **Phase 2G metadata:** the clean Phase 2G run-2 Raw/Mechanical/Resolved
  recall and precision values are recorded as fixed report metadata only and
  are never used as labels.
- **Benchmark lock:** the experiment rejects any benchmark whose content does
  not match the preregistered lock `a17674b6e2c491f0d7a1600dde0cfb8cc533d1d17db8633d8d94b2de9a57c1dd`
  (the Phase 2F legacy-benchmark content SHA-256, reused for Phase 2G/2H).

## 3. Dataset

The frozen five-case benchmark resolves to 16,624 candidates: 33 KEEP and
16,591 DROP (pooled prevalence 0.1985%). Candidate coverage is 33/33
(100%) for all five windows:

| Window | Candidates | KEEP | Coverage |
| --- | ---: | ---: | ---: |
| mid-push-prevents-side-collapse | 3,344 | 5 | 5/5 |
| push-poke-wave-crash | 3,248 | 5 | 5/5 |
| sweeper-limits-mid-play | 3,344 | 7 | 7/7 |
| unwarded-bush-hook-risk | 3,344 | 11 | 11/11 |
| wave-reset-after-kill | 3,344 | 5 | 5/5 |
| **Total** | **16,624** | **33** | **33/33** |

## 4. Pooled results (candidate-level, out-of-fold)

Each candidate is scored exactly once, when its window was held out.

| Metric | `logistic_A` | `logistic_B` | `lightgbm_A` | `lightgbm_B` |
| --- | ---: | ---: | ---: | ---: |
| Selected (score ≥ 0.5) | 1,907 | 114 | 398 | 399 |
| TP / FP / FN | 26 / 1,881 / 7 | 10 / 104 / 23 | 10 / 388 / 23 | 11 / 388 / 22 |
| Precision | 1.363% | 8.772% | 2.513% | 2.757% |
| Recall | 78.788% | 30.303% | 30.303% | 33.333% |
| F1 | 0.02680 | 0.13605 | 0.04640 | 0.05093 |
| Average precision (AP) | 0.020927 | 0.081275 | 0.026943 | 0.050952 |
| ROC AUC | 0.929158 | 0.939701 | 0.861030 | 0.932765 |
| Recall@1 / @3 / @5 / @10 | 0 / 0 / 0 / 0 | 0 / 3 / 4 / 6 | 0 / 2 / 2 / 2 | 1 / 2 / 2 / 4 |
| Recall@10 (rate) | 0.0% | 18.182% | 6.061% | 12.121% |
| Precision@10 (rate) | 0.0% | 12.0% | 4.0% | 8.0% |
| Gold rank mean / median | 233.48 / 137 | 208.12 / 92 | 452.36 / 179 | 226.24 / 157 |
| Overlap-cluster rank mean / median | 18.03 / 10 | 20.0 / 9 | 54.36 / 15 | 21.55 / 9 |

Baselines: all-DROP gives precision 0, recall 0, F1 0; all-KEEP gives precision
0.1985%, recall 100%, F1 0.3962%. Every cell has AP materially above the
0.1985% prevalence baseline and ROC AUC above chance, but top-K behavior is
weak and `logistic_A` has no top-10 hits. The models therefore expose a real
but weak and inconsistent discriminative signal.

## 5. Per-fold results

### 5.1 `logistic_A`

| Fold (held-out window) | Sel | TP | FP | FN | P% | R% | AP | AUC | R@1/3/5/10 | P@10% | Gold med | OC med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| mid-push-prevents-side-collapse | 361 | 4 | 357 | 1 | 1.108 | 80.0 | 0.028791 | 0.925726 | 0/0/0/0 | 0.0 | 132 | 11 |
| push-poke-wave-crash | 443 | 5 | 438 | 0 | 1.129 | 100.0 | 0.030757 | 0.957879 | 0/0/0/0 | 0.0 | 137 | 6 |
| sweeper-limits-mid-play | 482 | 5 | 477 | 2 | 1.037 | 71.429 | 0.026720 | 0.902950 | 0/0/0/0 | 0.0 | 199 | 25 |
| unwarded-bush-hook-risk | 318 | 9 | 309 | 2 | 2.830 | 81.818 | 0.036442 | 0.940212 | 0/0/0/0 | 0.0 | 205 | 10 |
| wave-reset-after-kill | 303 | 3 | 300 | 2 | 0.990 | 60.0 | 0.021241 | 0.929081 | 0/0/0/0 | 0.0 | 78 | 3 |

### 5.2 `logistic_B`

| Fold (held-out window) | Sel | TP | FP | FN | P% | R% | AP | AUC | R@1/3/5/10 | P@10% | Gold med | OC med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| mid-push-prevents-side-collapse | 35 | 1 | 34 | 4 | 2.857 | 20.0 | 0.049182 | 0.913627 | 0/0/0/1 | 10.0 | 92 | 6 |
| push-poke-wave-crash | 16 | 3 | 13 | 2 | 18.75 | 60.0 | 0.293781 | 0.982855 | 0/2/2/2 | 20.0 | 13 | 2 |
| sweeper-limits-mid-play | 10 | 0 | 10 | 7 | 0.0 | 0.0 | 0.027948 | 0.912625 | 0/0/0/0 | 0.0 | 125 | 19 |
| unwarded-bush-hook-risk | 36 | 4 | 32 | 7 | 11.111 | 36.364 | 0.105514 | 0.938985 | 0/1/1/1 | 10.0 | 128 | 9 |
| wave-reset-after-kill | 17 | 2 | 15 | 3 | 11.765 | 40.0 | 0.125569 | 0.956214 | 0/0/1/2 | 20.0 | 44 | 6 |

### 5.3 `lightgbm_A`

| Fold (held-out window) | Sel | TP | FP | FN | P% | R% | AP | AUC | R@1/3/5/10 | P@10% | Gold med | OC med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| mid-push-prevents-side-collapse | 106 | 3 | 103 | 2 | 2.830 | 60.0 | 0.136114 | 0.931536 | 0/1/1/1 | 10.0 | 35 | 10 |
| push-poke-wave-crash | 67 | 1 | 66 | 4 | 1.493 | 20.0 | 0.018239 | 0.938267 | 0/0/0/0 | 0.0 | 195 | 27 |
| sweeper-limits-mid-play | 100 | 2 | 98 | 5 | 2.0 | 28.571 | 0.018898 | 0.912839 | 0/0/0/0 | 0.0 | 147 | 16 |
| unwarded-bush-hook-risk | 50 | 2 | 48 | 9 | 4.0 | 18.182 | 0.024798 | 0.810040 | 0/0/0/0 | 0.0 | 240 | 14 |
| wave-reset-after-kill | 75 | 2 | 73 | 3 | 2.667 | 40.0 | 0.081978 | 0.781671 | 0/1/1/1 | 10.0 | 113 | 4 |

### 5.4 `lightgbm_B`

| Fold (held-out window) | Sel | TP | FP | FN | P% | R% | AP | AUC | R@1/3/5/10 | P@10% | Gold med | OC med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| mid-push-prevents-side-collapse | 120 | 3 | 117 | 2 | 2.5 | 60.0 | 0.055531 | 0.922761 | 0/0/0/1 | 10.0 | 103 | 9 |
| push-poke-wave-crash | 52 | 1 | 51 | 4 | 1.923 | 20.0 | 0.026524 | 0.955597 | 0/0/0/0 | 0.0 | 78 | 7 |
| sweeper-limits-mid-play | 97 | 2 | 95 | 5 | 2.062 | 28.571 | 0.023089 | 0.919025 | 0/0/0/0 | 0.0 | 272 | 27 |
| unwarded-bush-hook-risk | 77 | 2 | 75 | 9 | 2.597 | 18.182 | 0.063541 | 0.939612 | 0/1/1/1 | 10.0 | 171 | 4 |
| wave-reset-after-kill | 53 | 3 | 50 | 2 | 5.660 | 60.0 | 0.281347 | 0.927074 | 1/1/1/2 | 20.0 | 52 | 6 |

## 6. Overlap diagnostics

Gold endpoints sit inside large overlap clusters (median cluster size 518–656
across windows) of containing/contained candidate variants. The models rank
gold inside those clusters relatively well (pooled overlap-cluster rank
median 9–15) but rank gold far too low overall (gold rank median 92–179),
meaning the dominant distractors are spread across the whole window, with
overlap-cluster distractors contributing materially.

Pooled totals of overlap-cluster distractors outranking gold:

| Cell | Containing distractors outranking | Contained distractors outranking | Total |
| --- | ---: | ---: | ---: |
| `logistic_A` | 26 | 113 | 139 |
| `logistic_B` | 43 | 100 | 143 |
| `lightgbm_A` | 686 | 118 | 804 |
| `lightgbm_B` | 36 | 112 | 148 |

`lightgbm_A` is qualitatively worse here (notably 445 containing distractors
outranking gold in unwarded-bush-hook-risk), matching its degraded pooled
overlap-cluster rank (median 15, mean 54.36) and AUC 0.861.

Cross-cell true-positive consistency computed from the immutable window
tables (out-of-fold predictions):

- Pooled true positives: `logistic_A` 26, `logistic_B` 10, `lightgbm_A` 10,
  `lightgbm_B` 11.
- Every true positive found by `logistic_B`, `lightgbm_A`, or `lightgbm_B` is
  also found by `logistic_A` (intersections 10/10, 10/10, 11/11). The B
  features and LightGBM added no new gold endpoint to `logistic_A`'s recall.
- Only 4 of 33 gold endpoints are selected by all four cells; 18/33 are
  selected by at least two cells; 8/33 by exactly one cell; 7/33 are missed by
  every cell.
- Predicted-KEEP sets diverge across cells (pairwise Jaccard 5.6%–33.3%),
  with the best agreement between the two LightGBM cells (199/598 ≈ 33.3%).

The same candidate can therefore be a hit for one cell and a miss for another,
which is the signature of a real but weak and inconsistent ranking signal
rather than a robust discriminative pattern.

## 7. Feature findings

### 7.1 `logistic_A` coefficients (mean across folds, folds-seen in parentheses)

Top positive: `normalized_length` (3.470, 5), `char_len` (2.725, 2),
`contained_by_count` (2.564, 5), `overlap_count` (1.465, 3),
`hint_EVENT` (1.002, 5), `hint_LOCATION_OR_SPACE` (0.960, 5),
`hint_ABILITY_OR_RESOURCE` (0.785, 5), `ends_at_segment_boundary` (0.687, 5),
`normalized_end` (0.630, 5), `hint_ENTITY` (0.272, 4).

Top negative: `contains_count` (−6.652, 5), `token_count` (−3.683, 5),
`hint_TIME` (−2.626, 5), `segment_count` (−1.835, 5),
`starts_at_segment_boundary` (−1.678, 5), `hint_STATE` (−0.960, 5),
`char_len` (−0.905, 3), `overlap_count` (−0.514, 2), `hint_QUANTITY` (−0.507, 1),
`normalized_start` (−0.413, 5).

Interpretation: the pure-geometry model learns that contained spans and longer,
segment-aligned candidates are more endpoint-like while container spans and
time/state-tagged candidates are not. `char_len` and `overlap_count` flip sign
across folds (folds-seen 2–3), a warning of instability.

### 7.2 `logistic_B` coefficients

Top positive: `head=you` (3.666, 5), `first=you` (3.009, 5),
`contained_by_count` (2.596, 5), `last=one` (2.553, 4), `first=enemy` (2.460, 4),
`first=pull` (2.295, 4), `last=mid` (2.172, 2), `first=deep` (2.151, 4),
`first=get` (2.105, 3), `ngram=enemy team` (2.065, 4).

Top negative: `ngram=to` (−1.924, 5), `first=mid` (−1.713, 3),
`hint_STATE` (−1.638, 4), `ngram=and` (−1.615, 4), `ngram=you get` (−1.490, 4),
`ngram=to remove` (−1.487, 4), `ngram=die to` (−1.475, 4), `ngram=to pull`
(−1.428, 3), `ngram=you` (−1.414, 3), `ngram=mid and` (−1.393, 2).

Interpretation: sparse lexical evidence concentrates on a small set of
window-specific surface cues (`you`, `enemy`, `pull`, `mid`), which is why B
precision improves (8.772%) but recall collapses (30.303%) and the model is
not portable across the five windows (push-poke reaches AP 0.294, sweeper
drops to 0.0 recall).

### 7.3 `lightgbm_A` gain importances

Top: `contained_by_count` (42,500), `normalized_start` (12,648), `token_count`
(7,153), `normalized_length` (5,096), `hint_STATE` (4,877), `normalized_end`
(4,456), `char_len` (3,742), `ends_at_segment_boundary` (2,295),
`segment_count` (2,194), `overlap_count` (1,761), `hint_TIME` (1,388),
`hint_ENTITY` (1,077), `hint_LOCATION_OR_SPACE` (737), `type_hint_count` (487),
`starts_at_segment_boundary` (434), `hint_EVENT` (344),
`hint_ABILITY_OR_RESOURCE` (316), `hint_ACTION` (156), `has_digit` (0),
`has_percent` (0).

### 7.4 `lightgbm_B` gain importances

Top: `contained_by_count` (40,015), `token_count` (11,910),
`normalized_length` (7,691), `normalized_start` (6,466), `hint_STATE` (4,747),
`segment_count` (4,101), `first=enemy` (4,061), `last=again` (3,112),
`last=mid` (2,160), `cue_disfluency` (2,137), `first=win` (1,637),
`ngram=to` (1,627), `ends_at_segment_boundary` (1,603), `last=one` (1,499),
`last=them` (1,414), `has_quote` (1,407), `normalized_end` (1,251),
`char_len` (1,172), `hint_TIME` (1,168), `first=you` (1,157).

Both tree models are dominated by the same overlap/containment and
length/position geometry as the logistic cells; sparse lexical terms appear
only at lower importance and are not stable across folds (many folds-seen 1–2).
`has_digit`/`has_percent` receive zero gain in `lightgbm_A`.

## 8. Error analysis

Each misclassified candidate (false positive or false negative) receives
exactly one taxonomy code. Totals per cell (`correct` includes all correctly
scored candidates):

| Code | `logistic_A` | `logistic_B` | `lightgbm_A` | `lightgbm_B` |
| --- | ---: | ---: | ---: | ---: |
| Correct (TP+TN) | 14,736 | 16,497 | 16,213 | 16,214 |
| PARSER_FEATURE_ERROR | 0 | 0 | 0 | 0 |
| OVERLAPPING_LONGER_SPAN | 224 | 3 | 38 | 53 |
| OVERLAPPING_SHORTER_FRAGMENT | 173 | 8 | 52 | 55 |
| PRONOUN_DISTRACTOR | 54 | 29 | 22 | 27 |
| DISCOURSE_FILLER | 446 | 6 | 91 | 29 |
| GENERIC_ACTION_DISTRACTOR | 177 | 33 | 43 | 64 |
| GENERIC_ENTITY_DISTRACTOR | 337 | 7 | 43 | 57 |
| WRONG_CUE_PRIOR | 42 | 1 | 7 | 11 |
| SOURCE_POSITION_BIAS | 9 | 2 | 0 | 0 |
| GOLD_RANKED_LOW | 7 | 23 | 23 | 22 |
| GOLD_RANKED_HIGH_THRESHOLD_MISS | 0 | 0 | 0 | 0 |
| OTHER | 419 | 15 | 92 | 92 |

At the fixed 0.5 threshold, the recall loss is exactly the
`GOLD_RANKED_LOW` counts (7/23/23/22); no gold endpoint was in the top-5 of its
window yet below threshold (`GOLD_RANKED_HIGH_THRESHOLD_MISS` = 0).

`logistic_A`'s broad flood (1,907 selected) is dominated by discourse fillers,
generic entities, and overlapping longer spans — lexical surface plausibility
without endpoint semantics. `logistic_B` trades that flood for strictness and
leaves most gold endpoints below threshold (`GOLD_RANKED_LOW` 23), which is
why its precision is best but recall collapses. LightGBM cells sit between the
two: they contain false-positive volume (388 each) while still missing 22–23
gold endpoints.

## 9. Interpretation: real but inconsistent/weak ranking

The signal is real: pooled AP (0.021–0.081) and AUC (0.861–0.940) are far
above chance and above both trivial baselines, and overlap-cluster ranks
(median 9–15) show the models can often locate gold within its local cluster.

The signal is weak and inconsistent:

- Best top-K behavior is still poor: pooled R@10 is at most 6/33 (18.182%),
  and every gold endpoint has a median rank of 92–179 across cells, far below
  any usable selection horizon.
- The four cells' hits disagree: only 4/33 gold endpoints are selected by all
  cells, only 18/33 by at least two, and 7/33 by none; predicted-KEEP sets have
  pairwise Jaccard similarity as low as 5.6%.
- `logistic_A` (the highest-recall cell) achieves recall only by flooding 1,907
  selections at 1.363% precision; `logistic_B` (the highest-precision cell)
  covers just 10/33 gold. No cell is close to both usable precision and recall.
- Coefficients and importances show fold instability: surface cues such as
  `char_len`, `overlap_count`, `last=mid`, and several lexical terms flip
  sign/importance or appear in only 1–2 folds; the B vocabulary is trained on
  583–599 terms per fold and every fold reports 28–34 held-out OOV token
  types absent from the fitted vocabulary.

**LightGBM did not establish nonlinear superiority.** Comparing logistic vs
LightGBM on identical folds and features: AP is lower for `lightgbm_B`
(0.050952 vs 0.081275) and only marginally higher for `lightgbm_A`
(0.026943 vs 0.020927) while its AUC drops materially (0.861030 vs 0.929158);
R@10 is lower for `lightgbm_B` (12.121% vs 18.182%) and only 6.061% for
`lightgbm_A`; median gold ranks are worse for both tree cells (179/157 vs
137/92); and `lightgbm_A` lets 686 containing distractors outrank gold versus
26 for `logistic_A`. The tree models add no gold endpoint beyond
`logistic_A`'s recall set. There is no evidence that nonlinearity or the
bounded lexical features capture the endpoint concept.

## 10. Gates

- **Gate 1 (ranking-signal gate): failed.** No cell demonstrated a usable
  discriminative ranking signal: best pooled precision is 8.772% at 30.303%
  recall, best pooled R@10 is 18.182%, best AP is 0.081275, and gold median
  ranks are 92–179.
- **Gate 2 (dataset expansion): not triggered.** Because Gate 1 failed, this
  five-window result does not trigger dataset expansion as the next action.
  The LightGBM-vs-logistic comparison remains informational evidence and does
  not justify continued nonlinear modeling.

```text
WEAK RANKING SIGNAL ONLY
```

## 11. Reproducibility

Two clean offline runs at commit `a7549918fdef5dda0c5d10da8d52164d550d6142`
(`repository_dirty: false`)
produce identical scientific content; only the run timestamp differs.

- Definition SHA-256: `75dfaca522195ccd953825317c72e9780781c6c2b45b19f2655638d544c4a459`
  (identical).
- Benchmark input hashes (identical): content
  `a17674b6e2c491f0d7a1600dde0cfb8cc533d1d17db8633d8d94b2de9a57c1dd`; file
  `21a79651245c3a093c83bf725594fe7af9906f10e9449ab34af060986332c854`.
- Dataset summary, folds, per-window candidate tables, all metrics, feature
  findings, and error taxonomies: bit-for-bit identical between runs.
- Dependencies (identical): python 3.10.12, scikit-learn 1.7.2, lightgbm
  4.7.0, numpy 2.2.6, scipy 1.15.3.
- Aggregate inner `content_sha256`: run1
  `3a890de5f429056bae9d9932ce7f1985d9315e20655b4239be5acee4174edee2`; run2
  `9e62226a0ccb716bde74c186786038d8546980863eba37e8c18acdc24967e7ed`
  (differ only because `created_at` is inside the aggregate body).
- Aggregate file SHA-256: run1
  `bf6a591edc683491e3597013c83a74c774257c9dda2e910709117353de4b7816`; run2
  `1081e5b4da892c0f7d32ef28469dad4903f74597040c94ccf09f834adee8b655`.
- Manifest SHA-256: run1
  `b80fc6e7916dd473b024b9b59c48ccef38d72c9ef296e1438bc267c861ae2569`; run2
  `ea4962221e41fc3f378958ae646444c905f36f6692ff59dac54f581aa4c130e2`.
- Window-file SHA-256 (identical across runs): mid-push
  `a66ad0301f3491099419c8e2bb0c749a81bef666bfbec4d0ace9a317f4e5199a`;
  push-poke `ab89f82e53d80216f3ee0ff63dc09df6c0afe2a76a2158bcfd8dfcf7aa673555`;
  sweeper `00b7029ae0991c18137e1c897a8d583da77a4cbea22f60894b29fa8d51af18b8`;
  unwarded `6895d98ad61aaf3fb2b790f8e9e75bbf584dcf3de987f9bf42f2d3806dee394b`;
  wave-reset `a38a886650a9fda1b3b409a7fb41a611450c87ac28c5c14d5ac314e3ac4e5c58`.
- Archived clean runs:
  `data/phase2h_artifacts/phase2h-endpoint-scoring-run1.tar.gz` SHA-256
  `22aaab162f6122691f577bc95746a0b7b1da9834706766b746a29737a5e46380`;
  `data/phase2h_artifacts/phase2h-endpoint-scoring-run2.tar.gz` SHA-256
  `02b3e62030169c3c394b10ca3440ea1433bd270d2fe207f68ac1d7a6e165d817`.
- Per-window candidate-table locks recorded in the aggregate: mid-push
  `7c044f18dae95ab106df9c0406d7d562fe1be1ce419aa622e267d2e174e7ff72`; push-poke
  `4f655580c9dd6843ce64a44676a7b6b8b5a2a64b3c061cb72c119f4e8fa109b6`; sweeper
  `450041aabaa26bd90403242556a7825819139ffda2d7329408f9c2f1b9a9c896`; unwarded
  `6d862626ad9f572477eab6287ca249c9abcaef3dd98e3bc7ba83c1b81066a6b2`; wave-reset
  `c56cc8ace72d697c9cf22b768201a0edbb69e985b33a50ddda58f2143bfd89cf`.

Artifact integrity is enforced by the pipeline (aggregate self-hash plus
per-window locks against the MANIFEST and the aggregate) and by the
`compare-artifacts` CLI, which flags any change in inputs, scores, metrics, or
bytes.

## 12. Testing and review evidence

- Focused Phase 2H suite: `tests/test_phase2h_endpoint_scoring.py` and
  `tests/test_eval_phase2h_endpoint_scoring_cli.py` —
  **46 tests + 16 subtests passed** after the taxonomy correction (locally
  verified: `46 passed, 16 subtests passed in 54.87s`) using
  `./.venv/bin/python -m pytest tests/test_phase2h_endpoint_scoring.py tests/test_eval_phase2h_endpoint_scoring_cli.py -q`.
- Coverage: dataset contract and benchmark tamper rejection, four-cell
  determinism (including a cross-process determinism test), fold and fit-scope
  audits (scaler/class weights/vocabulary training-windows-only, OOV
  accounting), metrics and R@K/P@K with explicit denominators, overlap
  diagnostics, error taxonomy, strongest-features aggregation, hash-locked
  immutable artifact publishing, and CLI compare/publish behavior.
- Fresh independent review after the parser-taxonomy correction: **APPROVE**.
  The reviewer independently recomputed label, split, metric, overlap, and
  artifact-integrity invariants and found no blocking issue.
- Broad repository suite: **681 tests + 366 subtests passed** in 137.36s via
  `./.venv/bin/python -m pytest tests --ignore=tests/test_auth.py -q`.

## 13. Exactly one next justified intervention

**Bounded Feature Set C UD/syntactic ablation**, run only if implemented as a
bounded ablation with the same frozen candidate universe, the same 33 labels,
the same five grouped leave-one-window-out folds, the same fixed threshold
(0.5), the same two model families, and the same metrics and artifact format
as Phase 2H; retain A/B as frozen comparators. Stop the intervention if parser
integration becomes substantial
(anything beyond bounded, deterministic syntactic feature extraction for the
existing candidates). No other intervention — including data expansion, new
generative calls, roles/edges/graph construction, or tuning of the candidate
universe, labels, folds, or threshold — is justified now.

## 14. Supersession note

Phase 2G history is preserved, including its `MODEL-CAPABILITY BOTTLENECK`
diagnosis and its recommendation to rerun the frozen candidate-ID benchmark
with `deepseek-v4-pro` thinking enabled. That thinking-enabled generative
recommendation is now **superseded** by Phase 2H: the next step is the
bounded deterministic Feature Set C UD/syntactic ablation above, not a
generative thinking-enabled rerun.
