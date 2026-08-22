# Phase 2K Full-Transcript Context Ablation — Aggregate Report

Experiment: same model (0x Alpha / `opencode-go/ox-alpha-free` via OpenCode),
same Bronze target, same task/schema/vocabulary/metadata policy —
Condition A receives only the isolated Bronze target (+metadata+vocabulary);
Condition B additionally receives the FULL ordered transcript with the target's
character offsets. 20 live calls (10 frozen targets × 2 conditions).

Reviewer of record: agent reviewer 0x Alpha (scoping statement embedded in the
completed-reviews artifact). Correctness is graded against session-level ground
truth; honest-but-unresolved where the session determines the answer counts as
PARTIAL recovery; genuinely indeterminate tokens left unresolved are not
penalized. Prose quality is not scored. All 220 citations across both
conditions passed mechanical byte-exact grounding at import time.

## Headline

```text
A (isolated Bronze):   78 / 110 strict-success field judgments
B (full transcript): 109 / 110 strict-success field judgments

Target verdicts: B strictly wins 10 / 10, A wins 0, ties 0
Unsupported inference: NONE on all 110 judgments in BOTH conditions
Grounding failures:    0 in both conditions

Decision gate (all 5 preregistered criteria met):
ISOLATED_BRONZE_WAS_THE_WRONG_SEMANTIC_UNIT
```

## Per-field results (strict successes out of 10 cases)

| Field | A | B | net |
|---|---:|---:|---:|
| actors_entities | 3 | 10 | +7 |
| reference_bindings | 2 | 10 | +8 |
| abilities_resources | 6 | 10 | +4 |
| events_actions | 8 | 9 | +1 |
| states | 6 | 10 | +4 |
| conditions | 10 | 10 | 0 |
| recommended_advice | 8 | 10 | +2 |
| consequences_outcomes | 8 | 10 | +2 |
| explicit_relationships | 9 | 10 | +1 |
| uncertainty_unresolved | 10 | 10 | 0 |
| supporting_source_spans | 10 | 10 | 0 |

Hypothesis focus fields (actors, abilities, events, conditions, relations):
improvement in actors (+7), abilities (+4), and relations/events (+1 each);
conditions were already recoverable in isolation (0).

## Per-target results

| Target | A strict | B strict | verdict | decisive B-only recoveries |
|---|---:|---:|---|---|
| p2k:case:0001 | 9 | 10 | B wins | champion Karthus; SM→Smite; Harvest→Dark Harvest |
| p2k:case:0002 | 8 | 11 | B wins | 'after this' antecedent; cloth armor repair; speaker/student roles |
| p2k:case:0003 | 10 | 11 | B wins | truncated-clause completion (Executioner's/Trinity/Hydra logic) |
| p2k:case:0004 | 7 | 11 | B wins | 'one one'→Briar–Talia 1v1; ongoing invade state; advice purpose |
| p2k:case:0005 | 8 | 11 | B wins | 'she'→enemy Syndra; 'no spell' completion; push-away causality |
| p2k:case:0006 | 6 | 11 | B wins | habit=step-back-after-farming; 'queue'=Mel's Q; counterfactual |
| p2k:case:0007 | 9 | 11 | B wins | First Strike rationale; Nami W/heal confirmation follow-up |
| p2k:case:0008 | 7 | 11 | B wins | ban target=Ambessa (A mis-bound to Riven); 'they all in me' |
| p2k:case:0009 | 9 | 11 | B wins | 'she'=Mel via three-options enumeration; elimination completes |
| p2k:case:0010 | 8 | 11 | B wins | matchup identity (Varus vs Blitzcrank/Kalista); 'push'=wave push |

## Failure taxonomy (why isolated Bronze fails)

1. **Entity identity outside the window.** The coached champion or lane
   opponents are often named only before/after the target passage (Karthus,
   Syndra, Mel, Varus/Blitzcrank/Kalista, Ambessa). Isolated extraction either
   leaves them unnamed or — worst case (case 0008) — binds pronouns to the
   wrong champion (ban target read as Riven instead of Ambessa).
2. **Discourse-resolvable references.** Pronouns and compressed phrases whose
   antecedents sit in adjacent dialogue (`she`, `he`, `it`, `this`, `one one`
   = "the Briar-vs-Talia 1v1", `after this`). Condition A must leave these
   unresolved; Condition B resolves them with citations.
3. **Truncated-clause completions.** Windows cut mid-sentence; the completion
   typically follows within a few words (`but then she has no` → `spell`;
   `is fine because` → can't-buy-Executioner's reasoning; `like if` → `they
   all in me`). Only full-transcript access recovers the intended semantics.
4. **Context-licensed ASR repairs.** Tokens unrepairable from the lexical
   vocabulary alone become recoverable from nearby transcript statements
   (`SM`→Smite, `Harvest`→Dark Harvest, `clo armor`→cloth armor,
   `queue`→Mel's Q).
5. **What full context does NOT fix.** Genuinely indeterminate ASR tokens stay
   unresolved under both conditions (`freeb`, `twoo`, `the sa`, `MH`, `IIA`,
   `way`, `Synindra`). Audio-assisted span repair remains a separate,
   still-open problem.

## Validity notes

- Same prompt wrapper, instructions object, schema, vocabulary, target bytes,
  and metadata in both conditions (verified byte-identical at validation).
- No Mechanical Clean / contextual reconstruction / polish ran anywhere; B's
  transcript equals the archived DB transcription byte-for-byte.
- Grounding was mechanically enforced: every citation byte-exact (NBSP-tolerant
  matching documented in the module), ranges computed deterministically.
- One caveat: the reviewer of record is an AI (the implementing agent), not a
  blind human. The scoring policy is preregistered in the design doc and the
  raw evidence is fully preserved for independent re-review.
