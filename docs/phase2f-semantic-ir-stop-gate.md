# Phase 2F Stop-Gate Report — Bronze to Source-Anchored Semantic IR

## Decision

Phase 2F did not prove recoverable source-semantic IR. The strong-reference
legacy-development gate failed before reviewed graph semantics could be evaluated:
deterministic exact mention coverage was complete, but the compiler accepted
none of the 33 reviewed exact mentions and the compiled IR preserved none of
the 75 reviewed semantic facts.

```text
SEMANTIC IR NOT VIABLE — REDESIGN PASS 1
```

This decision stops the project before canonical source claims, domain
normalization, strategic relations, production persistence, corpus backfill,
Flash optimization, or Phase 3.

## Scope and stop rationale

The evaluated split was the five-case locked `LEGACY_FAILURE` development gate,
not formal `DEV` or `FROZEN_EVAL`. The goal preregistered this gate before the
30–50-window reviewed benchmark: if the new IR still lost the known five
mechanisms, broader annotation and frozen evaluation were not justified.

That stop condition occurred. A general focal-anchor mention repair was then
implemented, deterministically tested, and independently approved, but no
completed repaired-run artifact exists. A subsequently started process was
interrupted during its first case and the atomic runner published nothing; this
is a process note, not evidence supporting the recommendation. The unevaluated
repair is not positive evidence and cannot overturn the valid run2 failure.
Formal DEV/FROZEN gold
was not created, frozen labels were not inspected, and the once-only frozen run
was not consumed.

## Authoritative evidence

### Closed Phase 2E boundary

- Valid clause-first v2 artifact:
  `/tmp/phase2e-clause-first-v2-valid-run1.json`
- Inner SHA-256:
  `04c185aaf324251b4733e76c87b2c71ea3946497f79a8956f268e88f28e2e17b`
- File SHA-256:
  `02725fb163ef752c98f51a070652ef5418a5b0d4916363d1c61c3071e957c808`
- Result: deterministic clause coverage 5/5; official semantic recall 0/5;
  diagnostic mechanism containment after envelope-only normalization 2/5.

### Phase 2F provider-only attempt

- Archive:
  `data/phase2f_artifacts/phase2f-legacy-pro-run1.tar.gz`
- Archive SHA-256:
  `e6c2122a2b91c2b70d9775f2c108c26c82cdfff2f5cea9b3c5f60dbbc4146330`
- Classification: infrastructure evidence only. All 30 mention requests failed
  before model bytes because DNS was unavailable. It is not a semantic result.

### Phase 2F valid semantic-quality attempt

- Archive:
  `data/phase2f_artifacts/phase2f-legacy-pro-run2.tar.gz`
- Deterministic archive SHA-256:
  `b17cde9d7dc909c317aac81be08e9ed4860f91231d5568aeb6ee515a1fd67183`
- Aggregate inner SHA-256:
  `b0a030765217f2dcb52634d31eec171b307541308012945f87864cf7d5697492`
- Aggregate file SHA-256:
  `ad3801a9fc23a23837fe0ad078273a2744fb9640bbd826172d359af4654cf547`
- Clean revision:
  `b5317c6bd90572e052ab85f399e339c4de83a4e8`
- Provider/model: official `https://api.deepseek.com`, `deepseek-v4-pro`,
  thinking disabled.
- Calls with nonempty model output: 30 mention, 80 qualifier, 1,839 edge;
  provider failures: zero.

Locked evaluation inputs:

- legacy benchmark content SHA-256:
  `a17674b6e2c491f0d7a1600dde0cfb8cc533d1d17db8633d8d94b2de9a57c1dd`;
- exact-source manifest content SHA-256:
  `cf86dde955f4cbeee091f38aab8293256b0c48f809c969384185a330ee511241`;
- representative 300-window pool content SHA-256:
  `9b89c6d6c6c8070eba48d6db47254e156c1b2591c1480a60f98a1e8d789491c2`.

Every archived case reconstructs as a typed `SemanticRunArtifact`, validates,
round-trips to identical canonical bytes, and matches its indexed hashes.

## Preregistered layer results

| Layer | Result | Interpretation |
|---|---:|---|
| Exact mention candidate coverage | 33/33 | Deterministic catalog succeeded |
| Exact mention selection | 0/33 | First semantic loss |
| Compatible mention typing | 0/33 | Cascaded from selection |
| Qualifier candidate coverage | 10/10 | Deterministic cue catalog succeeded |
| Qualifier recovery | 0/10 | Reviewed mention nodes absent |
| Reviewed edge-pair/edge recovery | 0/24 | Endpoints absent; classifier viability undetermined |
| Reference candidate/recovery | 0/8 | Reviewed reference nodes absent |
| Semantic completeness/checksum | 0/75 | No reviewed fact survived |
| Source anchoring | 5/5 cases | Accepted spans remained exact bronze slices |
| Edge provenance | 5/5 cases | Accepted edges remained traceable |
| Fabricated source offsets | 0 | Hard-safety gate passed |
| Hidden Pass 1 ontology normalization | 0 | No strategic/domain concepts inserted |
| Provider failures | 0 | Result is model/architecture evidence |

Every case had a zero checksum, with locked denominators 12, 12, 14, 12, and
25. Unsupported-node and unsupported-edge rates remain unknown because the
legacy fixture is intentionally non-exhaustive: 80 produced nodes and 1,292
produced edges were unscored. They must not be treated as evidence of low
invention.

### Required dimension scores

| Dimension | Score |
|---|---:|
| Entity recovery | 0/9 |
| Ability/resource recovery | 0/3 |
| Event recovery | 0/3 |
| Action recovery | 0/14 |
| State/outcome recovery | 0/3 |
| Actor/target roles | 0/12 |
| Condition recovery | 0/10 |
| Negation | 0/3 |
| Modality | 0/2 |
| Causal edges | 0/5 |
| Coreference | 0/8 |
| Location/space | 0/1 |
| Semantic-completeness dimension | 0/2 |
| Temporal edges/termination | Not labeled in the legacy gold |

The aggregate failure taxonomy is retained rather than collapsed into one
score: `ASSEMBLY_FAILURE` 32, `CONDITION_LOSS` 5,
`MENTION_SELECTION_MISS` 33, `MODALITY_LOSS` 2,
`MODEL_PARSE_FAILURE` 86, `NEGATION_LOSS` 3, `QUALIFIER_LOSS` 1, and
`TEMPORAL_LOSS` 1. These counts include downstream cascades and are not
independent first losses. In particular, `TEMPORAL_LOSS` is a compiler assembly
failure on an unscored proxy node, not reviewed temporal-recall evidence.

## First-loss analysis

Official evaluator first loss was `MODEL_PARSE_FAILURE` in mention selection
for all five cases. Parser robustness was not central:

- 7/30 mention outputs were strict JSON;
- 12/30 were complete JSON inside one Markdown fence;
- 11/30 were truncated or otherwise incomplete;
- accepting only the complete fence envelope raises parseable outputs to 19/30
  but exact reviewed mention recovery remains 0/33;
- only one reviewed candidate ID appears anywhere in all retained raw mention
  output, and that response was incomplete.

The evaluated model interface exposed 3,248–3,344 overlapping n-grams per
roughly 500-character window in six flat partitions of up to 600 candidates.
Each partial partition repeated the whole source and asked for every mention,
omitted offsets, exposed long opaque IDs, and showed incomplete heuristic type
hints. The model predominantly selected clause-sized proxy spans; emitted IDs
had a median span length of 111 characters. The architecture therefore
recreated the Phase 2E selection bottleneck one level lower:

```text
DETERMINISTIC EXACT SOURCE COVERAGE       33/33
        ↓
MODEL-FACING MENTION BOUNDARY              0/33
        ↓
REVIEWED SEMANTIC FACTS                    0/75
```

The independently approved but unevaluated repair retains the exhaustive
catalog as a coverage oracle and changes model requests to exact focal starts,
local aliases, offsets, atomic-span instructions, and versioned fence handling.
Its structural correctness does not establish semantic recoverability.

## Milestone report

- **Hypothesis:** low-level source mentions plus bounded relation
  classification are more recoverable than direct proposition selection.
- **Architecture change:** separate Pass 0 source windows, typed source nodes,
  qualifiers, explicit unresolved references, bounded directed edge pairs,
  graph assembly, semantic checksum, and proof-carrying artifacts were added in
  independent `pipeline/semantic_*` modules.
- **Files changed:** `pipeline/semantic_source.py`, `semantic_ir.py`,
  `semantic_mentions.py`, `semantic_qualifiers.py`, `semantic_coreference.py`,
  `semantic_edges.py`, `semantic_compiler.py`, `semantic_ir_evaluation.py`,
  `semantic_ir_artifact.py`, and `semantic_ir_pool.py`;
  `scripts/build_semantic_ir_pool.py` and `scripts/eval_phase2f_semantic_ir.py`;
  locked fixtures and archives under
  `data/`; focused `tests/test_semantic_*` and `tests/test_phase2f_*`; and this
  plan, report, and `handoff.md`.
- **Invariant:** bronze is immutable; every accepted node is an exact source
  slice; every edge retains exact endpoints/evidence and model/config
  provenance; Pass 1 contains no strategic ontology or proposition tuple.
- **Deterministic tests:** the final semantic and Phase 2F suite passed 166
  tests and 129 subtests. The focal repair additionally passed a 100-window,
  eight-budget randomized partition audit.
- **Development evaluation:** the valid locked five-case strong-model run
  failed at 0/33 mention selection and 0/75 semantic checksum.
- **First loss:** model-facing mention selection, after successful deterministic
  candidate generation.
- **Reviewer findings:** reviewers localized broad proxy selection, fence and
  truncation noise, missing offsets, misleading type hints, ambiguity lifecycle
  errors, orchestration/prompt version gaps, and non-attributable multi-focus
  abstention.
- **Fixes:** exact focal grouping, local aliases/offsets, type-hint removal,
  strict version binding, single-fence canonicalization, and one-focus
  attribution were implemented and independently approved.
- **Unresolved issue:** no completed strong-model result demonstrates that the
  repaired boundary recovers mentions; downstream qualifier, coreference, and
  pairwise relation quality therefore remains unproved.
- **Next milestone justified:** no. Pass 2 and formal DEV/FROZEN evaluation are
  not justified by current evidence.

## Answers to the 18 final Phase 2F questions

1. **Can the IR express important bronze meaning without one proposition
   tuple?** The typed graph and reviewed gold can encode it, so schema
   expressivity is plausible. The compiler did not recover that graph, so the
   representation proof failed.
2. **Can source mentions be recovered reliably?** No: 0/33 reviewed exact
   mentions were recovered.
3. **Does deterministic generation cover reviewed mentions?** Yes: 33/33.
4. **Can a strong model select the correct mentions?** Not through the
   evaluated interface: accepted recovery was 0/33. The lone exact reviewed ID
   visible in raw output occurred inside an incomplete response and was not a
   valid model decision.
5. **Can pairwise classification recover causal, temporal, and role edges?**
   Undetermined. Reviewed causal/role endpoints were absent, so their zero
   recovery is upstream-cascaded and not a valid classifier-quality measure.
   Temporal edges were not labeled in this legacy gold at all.
6. **Does the architecture avoid the clause-selection bottleneck?** No. The
   evaluated flat mention interface reproduced it with broad proxy spans.
7. **Are actors/targets recovered without blindly assigning “you”?** The schema
   avoids automatic rewriting, but reviewed actor/target recovery was zero.
8. **Are conditions preserved?** No reviewed condition survived into the IR.
9. **Is negation preserved?** Grounded qualifier candidates existed, but no
   reviewed negation survived compilation.
10. **Are temporal termination events preserved?** Not evaluated. The locked
    legacy benchmark contains no reviewed temporal/`TERMINATES` fact.
11. **Is unresolved coreference represented rather than guessed?** The schema
    and deterministic candidate layer support it, but recovery was 0/8 and no
    model-quality proof was obtained.
12. **Can every accepted node/edge be traced to bronze?** Yes for accepted
    artifacts: exact-source and provenance safety held. This safety result does
    not compensate for zero reviewed recall.
13. **What percentage of bronze-answerable questions remain answerable from
    IR?** 0/75, or 0%.
14. **What information is still lost?** All reviewed actors, events, actions,
    states/outcomes, conditions, qualifiers, references, and causal/role facts
    in the five cases. Temporal preservation remains unknown rather than lost.
15. **What causes the first remaining loss?** The evaluated model-facing
    mention boundary: exhaustive overlapping candidates, flat partial
    partitions, broad proxy selection, omitted offsets, misleading hints, and
    output truncation.
16. **Is the representation strong enough to justify Pass 2 canonical
    claims?** No.
17. **Which operations look deterministic enough for future work?** Pass 0
    segmentation/context, exact-span candidate generation, stable IDs, offset
    resolution, source hashing, artifact reconstruction, and provenance
    validation.
18. **Which operations still require strong semantic judgment?** Atomic mention
    boundary selection and typing remain unproved; qualifier scope and
    reference resolution remain unproved; reviewed causal/role endpoints were
    not reached; and temporal classification had no legacy gold against which
    to measure quality.

## Completion audit against the Phase 2F goal

| Requirement group | Authoritative evidence | Disposition |
|---|---|---|
| Phase 2E closure and history | Verified v2 artifact hashes, regenerated 5/5 catalogs, handoff and plan history, preserved Phase 2E code/tests | Complete |
| Clean-room Pass 0/Pass 1 boundary | Separate `pipeline/semantic_*` modules and 166 passing semantic/Phase 2F tests | Implemented and reviewed |
| Bronze immutability and proof | Exact `SourceSpan` validation, stable IDs/hashes, typed artifact reconstruction | Hard-safety gate passed |
| Deterministic mention catalog | Locked benchmark and run2 evaluation | 33/33 coverage passed |
| Strong-model representation proof | Reconstructible run2 archive | Failed at 0/33 mentions and 0/75 checksum |
| Representative 200–500 pool | `data/semantic_ir_window_pool_v1.json`, independently reproduced from inputs | 300 windows complete |
| Formal reviewed DEV/FROZEN subsets | Preregistered legacy stop rule and failed gate | Intentionally not entered |
| Frozen evaluation | Failed prerequisite; no labels inspected and no run published | Correctly not consumed |
| Independent review | Boundary-specific reviews plus final stop-gate review | Complete |
| Pass 2+, normalization, strategic relations, production persistence, Flash optimization | Repository/module review and documented stop | Not started, as required |
| Final recommendation | This report | Complete |

The no-provider runner still reconstructs and hash-locks all inputs and reports
`VALIDATED_NO_PROVIDER_CALL`. Artifact tests separately reconstruct the DNS-only
run and the valid semantic-quality run. No conditional milestone is relabeled
as successful merely because it was skipped after the failure.

## Required next action

Any future continuation requires a new explicit goal to redesign Pass 1. It
must not proceed to Pass 2, domain normalization, strategic relations, corpus
backfill, or cheap-model optimization. A future design should preregister an
early mention-only stop gate and a strict inference-call budget before invoking
a strong provider.
