# Phase 2B Baseline Failure Traces

Run date: 2026-08-15. Configuration was unchanged from the Phase 2 stop gate:
`deepseek-v4-flash`, non-thinking mode, `RELATION_EXTRACTION_MAX_TOKENS=512`,
ontology `strategic-ontology-v0`, prompt
`strategic-relation-extraction-v0`, acceptance threshold `0.60`, and no
persistence. Raw command output is retained under `/tmp/phase2b-flash-*.json`.

## 13426: Thresh Flay versus Rocket Jump

Source evidence explicitly says that Flay can interrupt Tristana Rocket Jump
when timed at the dash apex. The packet exposes `Flay -> Thresh E` and
`Rocket Jump -> Tristana W`.

The model emitted `Flay (E) --denies--> Rocket Jump` with the apex condition.
Entity canonicalization resolved both aliases; relation type canonicalized to
`denies`; evidence ID/provenance validated; and condition/effect survived. The
model also emitted a second validly grounded relation for Zac E, which is not a
false positive by itself but is outside the reference target.

Disposition: **REVIEW**, not reject. Confidence was `0.59 = .55*.70 +
.35*.30 + .10*1.0`, just below the `0.60` threshold. The model-proposed
`intermittent_pressure` concept was removed because it was not evidence-bound.
No dedupe ran.

Attribution: `CONFIDENCE_GATE_TOO_STRICT_OR_UNCALIBRATED` for the target edge;
`CONCEPT_MAPPING_FAILURE` only for the unsupported auxiliary concept.

## 13334: Explosive Charge and Rocket Jump reset

Source evidence says Rocket Jump resets when Explosive Charge is fully stacked
or on a kill. The packet exposes `Explosive Charge -> Tristana E` and
`Rocket Jump -> Tristana W`.

The model emitted `Rocket Jump --enables--> secondary jump` with
`object_type=action`, then `Explosive Charge --enables--> Rocket Jump reset`
with `object_type=mechanic`. Both types/nodes are deliberately absent from the
ontology and packet, so entity validation rejected both before confidence or
dedupe.

Disposition: **REJECT**. This is a correct safety rejection, but demonstrates
that the one-stage prompt asks the model to invent an intermediate effect node
rather than return a supported canonical ability-to-ability relation.

Attribution: `MODEL_WRONG_OBJECT`, `MODEL_UNSUPPORTED_INVENTION`, and
`ENTITY_NORMALIZATION_FAILURE` (correct rejection). No concept endpoint was
involved; no dedupe ran.

## 13612: Sylas Abduct buffers Kingslayer

Source evidence explicitly says Abduct travel permits a buffered Kingslayer,
whose arrival timing prevents a flash. The packet exposes `Abduct -> Sylas E`
and `Kingslayer -> Sylas W`.

The model emitted the correct ability-to-ability `enables` relation, preserved
the mid-dash condition, cited the supplied evidence, and canonicalized both
entities. It added `tempo` and `combat_compression`; these were removed because
the source does not support those abstractions. The grounded relation itself
remained valid.

Disposition: **REVIEW**, not reject. Confidence was `0.37 = .55*.30 +
.35*.30 + .10*1.0`, far below threshold because both model and source scores
were `0.30`. No dedupe ran.

Attribution: `CONFIDENCE_GATE_CORRECT_REJECTION_OR_UNCALIBRATED` pending the
labeled benchmark; `CONCEPT_MAPPING_FAILURE` only for auxiliary unsupported
concepts.

## Initial Conclusion

The Phase 2 failure is not one bottleneck. The current evidence shows:

1. A strict entity validator correctly blocks invented intermediate nodes.
2. Valid relation structure can survive the compiler but be concentrated in the
   review bucket due to source/model confidence calibration.
3. Current concept validation is literal/alias based. It correctly rejects
   unsupported auxiliary concepts here, but Phase 2B must test whether it also
   incorrectly rejects semantically entailed abstraction endpoints.

No validator threshold or ontology change is justified from these three cases.
