# Phase 2G Report — Semantic Endpoint Recovery Ablation

## Decision

```text
Phase 2G: TARGETED NEXT INTERVENTION

Candidate coverage:
33/33

Raw Bronze + candidate IDs:
  endpoint recall: 8/33 (24.2%)
  precision: 8/397 (2.0%)
  role accuracy: 8/8 (100.0%)

Mechanical Silver + candidate IDs:
  endpoint recall: 4/33 (12.1%)
  precision: 4/311 (1.3%)
  role accuracy: 4/4 (100.0%)

Resolved Silver + candidate IDs:
  endpoint recall: 10/33 (30.3%)
  precision: 10/765 (1.3%)
  role accuracy: 10/10 (100.0%)

Failure distribution:
  WRONG_CANDIDATE_SELECTED: 87
  MODEL_INVENTED: 6
  all other first-failure codes: 0

Diagnosis:
MODEL-CAPABILITY BOTTLENECK

Next justified action:
Run the frozen candidate-ID benchmark with the same deepseek-v4-pro teacher
and thinking enabled. Do not change the prompt, catalog, Silver fixture,
reviewed endpoints, parser, or grounding rules.
```

The scores above are the second clean run, whose 15 responses were all
parseable. The first clean run is retained separately and is reported below.
Neither run passed any endpoint-promotion condition.

## Central answer

Removing exact-span generation recovered some reviewed endpoints: Raw Bronze
rose from Phase 2F's accepted 0/33 to 6/33 and 8/33 across the two clean runs.
Exact reproduction was therefore a contributing problem, but it was not the
dominant sufficient explanation. Candidate-ID selection remained far below the
90% recall and precision gates.

Mechanical cleanup and explicit linguistic reference resolution did not
produce a stable, material improvement. Mechanical Silver moved from 8/33 in
run 1 to 4/33 in run 2; Resolved Silver moved from 7/33 to 10/33. The direction
and size of the differences were not reproducible, and every condition stayed
at or below 30.3% recall.

The dominant observed failure was candidate discrimination. The model selected
large numbers of valid but task-wrong candidate IDs. In run 2, 98.2% of Raw,
98.9% of Mechanical, and 98.8% of Resolved selections were unsupported or
invented for their task. When an exact endpoint was selected, its low-level
role was always compatible with the reviewed role. The model can type the
occasional correct endpoint; it cannot reliably isolate that endpoint from the
complete source-covered catalog under the tested non-thinking configuration.

The reference-status result was also uniformly negative: 0/8 in every
condition and both runs. The model treated the reviewed unresolved references
as unambiguous instead of returning `UNKNOWN`. Resolved Silver did not repair
that behavior.

Per the preregistered Outcome D rule, the evidence supports a tested-model
semantic discrimination ceiling after exact-span generation, basic cleanup,
and simple reference reconstruction have all failed as sufficient solutions.
This diagnosis applies to `deepseek-v4-pro` with thinking disabled on this
interface and benchmark; it is not a claim about all models.

## Controlled experiment

The implementation is commit `64baf2b69e4d1c92dd681b61ac9c29c131e39c79`.
Both live runs recorded a clean worktree at that revision and used:

- locked benchmark content SHA-256
  `a17674b6e2c491f0d7a1600dde0cfb8cc533d1d17db8633d8d94b2de9a57c1dd`;
- locked Silver fixture content SHA-256
  `4ae3f1bd167f1bebb27ce3d27118833d7c869bb28c4b076d98735182a9fb5a41`;
- the complete Phase 2F candidate generator, compact immutable aliases, exact
  candidate text, and disambiguating local offsets;
- identical 33 endpoint tasks and 8 reference-status tasks across conditions;
- official `https://api.deepseek.com`, `deepseek-v4-pro`, thinking disabled,
  temperature 0, and 4,096 output tokens;
- one case-level request for each of five cases in each of three conditions,
  for 15 calls per run;
- no strategic concepts, causal edges, graph construction, ontology tuning,
  held-out labels, or weakened grounding.

Every candidate artifact retains its Phase 2F candidate ID, window-local and
upstream absolute Bronze offsets, exact Bronze text, segment provenance, and
compact alias. Every known selection resolved through that catalog. Model
text and offsets were never accepted as source authority.

## Condition measurements

### Clean run 1

| Condition | Candidate coverage | Endpoint recall | Endpoint precision | Role accuracy | UNKNOWN/status accuracy | Parseability | Unsupported | Invented | Unsupported/invented rate | Alignment violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw Bronze | 33/33 | 6/33 | 6/53 | 6/6 | 0/8 | 4/5 | 65 | 0 | 91.5% | 0/71 |
| Mechanical Silver | 33/33 | 8/33 | 8/257 | 8/8 | 0/8 | 4/5 | 268 | 0 | 97.1% | 0/276 |
| Resolved Silver | 33/33 | 7/33 | 7/80 | 7/7 | 0/8 | 4/5 | 90 | 0 | 92.8% | 0/97 |

Run 1 first failures:

| Condition | Wrong candidate | Parser failure | Other codes |
|---|---:|---:|---:|
| Raw Bronze | 19 | 11 | 0 |
| Mechanical Silver | 26 | 5 | 0 |
| Resolved Silver | 19 | 11 | 0 |

All 15 provider calls returned nonempty model bytes. There were zero provider
failures. One case per condition failed strict parsing; the raw responses are
retained and were not converted into semantic credit.

### Clean run 2

| Condition | Candidate coverage | Endpoint recall | Endpoint precision | Role accuracy | UNKNOWN/status accuracy | Parseability | Unsupported | Invented | Unsupported/invented rate | Alignment violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw Bronze | 33/33 | 8/33 | 8/397 | 8/8 | 0/8 | 5/5 | 435 | 1 | 98.2% | 0/443 |
| Mechanical Silver | 33/33 | 4/33 | 4/311 | 4/4 | 0/8 | 5/5 | 376 | 0 | 98.9% | 0/380 |
| Resolved Silver | 33/33 | 10/33 | 10/765 | 10/10 | 0/8 | 5/5 | 802 | 5 | 98.8% | 0/812 |

Run 2 first failures:

| Condition | Wrong candidate | Model invented | Other codes |
|---|---:|---:|---:|
| Raw Bronze | 29 | 1 | 0 |
| Mechanical Silver | 32 | 0 | 0 |
| Resolved Silver | 26 | 5 | 0 |

All 15 provider calls returned nonempty model bytes, all 15 responses parsed,
and there were zero provider failures. The cleaner parse result did not improve
semantic precision.

## Reproducibility disposition

The deterministic experiment inputs reproduced exactly: revision, benchmark,
Silver fixture, candidate catalogs, task definitions, model/configuration, and
all request hashes matched. The promotion decision and diagnosis also
reproduced: no condition passed, all endpoint recalls remained at or below
30.3%, all precisions remained at or below 11.3%, status accuracy remained
0/8, and broad wrong-candidate selection dominated.

The strict score/failure-distribution comparator did not pass because the
provider returned materially different selection volumes and exact scores at
temperature 0. This is retained negative reproducibility evidence; exact-score
reproduction is not claimed. It independently prevents a Phase 2G endpoint
gate pass even though the already-failing semantic thresholds are decisive.

## Promotion-gate audit

| Requirement | Run 1 | Run 2 | Disposition |
|---|---:|---:|---|
| Deterministic candidate coverage = 100% | 33/33 all conditions | 33/33 all conditions | Pass |
| Endpoint recall >= 90% | best 8/33 | best 10/33 | Fail |
| Endpoint precision >= 90% | best 6/53 | best 8/397 | Fail |
| Role accuracy >= 85% | 100% of recalled | 100% of recalled | Pass, conditional on sparse recall |
| Unsupported/invented <= 5% | best 91.5% | best 98.2% | Fail |
| Accepted source alignment | zero known-selection violations | zero known-selection violations | Pass |
| Parseability | 4/5 each condition | 5/5 each condition | Improved, not semantic rescue |
| UNKNOWN/NONE/AMBIGUOUS accuracy | 0/8 all conditions | 0/8 all conditions | Fail |
| Clean rerun reproduces exact scores | — | strict comparator failed | Fail |
| No ontology/held-out/grounding weakening | verified | verified | Pass |

The endpoint gate fails. Typed semantic nodes, bounded semantic edges, Pass 2,
and Phase 3 remain unauthorized.

## Retained artifacts

- Run 1 archive:
  `data/phase2g_artifacts/phase2g-endpoint-run1.tar.gz`;
  archive SHA-256
  `ef83bbc131dc228c3369334d47e3986bd7a955cd9f609526b55a4c15c5d1253e`;
  aggregate canonical content SHA-256
  `c9b55a55fbdc72d31c2b042a582df56047d8d8711a3c58640f2e54d0f5a50374`;
  aggregate file SHA-256
  `58a8d8366d30a0d9a85968109a47f3999587efb1efb48bd05e57a9bb73009341`.
- Run 2 archive:
  `data/phase2g_artifacts/phase2g-endpoint-run2.tar.gz`;
  archive SHA-256
  `79fe660abe02aa12a2464e56522664915c9f1ed34c68092b66ab186ac9a20a8e`;
  aggregate canonical content SHA-256
  `d0225841660b571d3134b0bb883fcb987778af2f2f75117e2ef4c1e15499da76`;
  aggregate file SHA-256
  `6817a1a3f2fc237ab9b12dc0c76ff0bd69f4ee0e757222abf8260a31a03e5242`.

Each archive includes the aggregate report, 15 condition/case artifacts, and a
file-hash manifest. The artifacts retain input representations, full candidate
catalogs, exact Bronze offsets, all Silver transformations, raw responses,
parsed selections, deterministic resolutions, expected-versus-selected
reports, role metrics, failure taxonomies, model/configuration metadata, and
reproducibility hashes.

## Validation

- Focused Phase 2G plus inherited Phase 2F source/candidate suite:
  98 tests and 144 subtests passed.
- Full non-browser repository suite:
  635 tests and 350 subtests passed.
- `tests/test_auth.py` remains the unrelated Chromium import-time environment
  exclusion already recorded by Phase 2F.

## Exactly one next intervention

Freeze this candidate-ID interface, benchmark, Silver fixture, parser, and
metrics. Run the same 15-call matrix with `deepseek-v4-pro` thinking enabled.
This tests the prescribed Outcome D teacher intervention without moving the
failure boundary, changing the gold, or redesigning downstream graph stages.
