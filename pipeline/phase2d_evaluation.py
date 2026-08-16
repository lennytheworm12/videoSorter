"""Development-only Phase 2E span-first Stage A proposition evaluation.

This module measures source recovery and Stage A independently.  It does not
invoke canonical mapping, write a ledger, or alter bronze/evidence records.

The evaluator accepts either the Phase 2E :class:`StageAExtraction` result
(retaining raw stage outputs, selected evidence spans, parsed semantic
frames, normalization, and deterministic propositions) or the legacy tuple of
:class:`ExtractedProposition` values.  The primary Phase 2E semantic
proposition score measures the grounded recovered actor/event/effect/
condition slots plus the forward causal direction directly from the semantic
frame or stage slots, independently of ontology normalization and
proposition assembly; final assembled proposition recall is reported
separately.  Slot-level scores are diagnostics with explicit hit counts and
denominators suitable for X/5 reporting.  Every official slot recall
denominator counts every source-available eligible expected proposition:
an unreached stage is a miss, never a denominator exclusion, so a provider,
localizer, or partial-stage failure is visible as lost recall rather than
hidden.  Reached counts and conditional accuracy-when-reached are reported
separately under ``slot_reached`` so first-loss analysis remains available
without blurring the official X/5 gate.

Stage A frames are defensively validated against the proposition packet:
slots and evidence spans must quote their exact source text and offsets,
transcript spans must carry verified absolute offsets, all spans and slots
must share one source, the actual and frame spans must agree, and every slot
must fall inside a selected evidence span; ungrounded frames score no
evidence/slot/semantic hit.  Condition official recall and the semantic
proposition additionally require the reviewed leading condition operator when
one is present.  Held-out separation is mandatory for every development
fixture, including arbitrary or metadata-less files: development insight and
source IDs are always compared to the repository frozen Phase 2B fixture
resolved from a trusted explicit path, and an unavailable frozen fixture is an
error.  Top-level proposition recall/exact/eligible-case and safe-zero
accuracy denominators count every source-available eligible entry, so provider
and stage failures lower the official metrics instead of vanishing.

Stage A results must be internally coherent before scoring: stage slots,
selected evidence spans, causal direction, and the semantic frame must agree
wherever they overlap, and ``actual.propositions`` must equal the deterministic
assembly of the validated frame via
:func:`pipeline.proposition_extract.assemble_grounded_proposition`.  A slot or
direction contradiction zeroes the affected official metrics (never inflates),
and propositions that are not the frame's deterministic assembly are
suppressed from assembled/exact scoring and reported as unsupported output,
while partial-stage diagnostics and normalization-failure semantic scoring
remain available.  The per-mode ``coherence`` block exposes the overall
``coherent`` boolean (grounded, no slot conflicts, direction consistent, and
deterministic assembly consistent) alongside the component flags so callers
can gate final credit on the entire Stage A result without re-deriving it.

The development fixture carries reviewed closed-ontology normalization labels,
including explicit nulls where no canonical concept or relation is directly
supported.  Normalization recall therefore uses the same source-available X/5
denominator as the semantic slots.  The normalization-stage diagnostic also
exposes reached, completed, abstained, mapped, and failed counts.  Causal
direction expectations for eligible cases are derived from the reviewed
subject/predicate/effect role labels (the reviewed causal actor produces the
reviewed effect), not from separate direction labels.
"""

from __future__ import annotations

from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from core.ontology import RELATION_TYPES, STRATEGIC_CONCEPTS
from pipeline.proposition_extract import (
    ClauseCandidate,
    ExtractedProposition,
    PropositionPacket,
    SourceAlignment,
    SourceMode,
    SourceSemanticFrame,
    StageAExtraction,
    assemble_grounded_proposition,
    coalesce_selected_evidence,
)
from pipeline.source_windows import SourceWindow, SourceWindowResolver


ExtractionResult = tuple[ExtractedProposition, ...] | StageAExtraction

_CONDITION_OPERATORS = frozenset(("if", "when", "after", "before", "while", "unless", "until", "once", "around"))
_SLOT_FIELDS = ("subject", "predicate", "effect", "condition")
_SLOT_SCORE_FIELDS = (("actor", "subject"), ("event", "predicate"), ("effect", "effect"), ("condition", "condition"))
_SLOT_RECALL_NAMES = (
    "evidence_span", "actor", "event", "effect", "condition", "causal_direction",
    "normalization", "semantic_proposition", "assembled_proposition", "exact_decomposition",
)
_TRANSFORMATION_STAGES = (
    "evidence_localization", "actor_extraction", "event_extraction",
    "effect_extraction", "condition_extraction", "causal_direction",
    "ontology_normalization", "proposition_assembly", "exact_decomposition",
)
_STAGE_FLAG_KEYS = {
    "evidence_localization": "evidence_span_hit",
    "actor_extraction": "actor_hit",
    "event_extraction": "event_hit",
    "effect_extraction": "effect_hit",
    "condition_extraction": "condition_hit",
    "causal_direction": "causal_direction_hit",
    "ontology_normalization": "normalization_hit",
    "proposition_assembly": "matched",
    "exact_decomposition": "exact",
}
_EXPECTED_CAUSAL_DIRECTION = "actor_event_causes_effect"
DEFAULT_HELD_OUT_FIXTURE = (
    Path(__file__).resolve().parent.parent / "data" / "relation_extraction_phase2b_v0.json"
)


def load_development_cases(
    path: str | Path, *, held_out_path: str | Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Load the separately maintained Phase 2D development-only fixture.

    Held-out separation is mandatory: the development fixture is always
    compared against the repository frozen Phase 2B fixture resolved from a
    trusted explicit path (never a fixture-controlled basename), and an
    unavailable frozen fixture is an error.  Callers that need temp-fixture
    testability may pass their own trusted ``held_out_path`` explicitly.
    """
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, Mapping) else None
    if not isinstance(cases, list):
        raise ValueError("Phase 2D fixture requires a cases list")
    result = []
    for case in cases:
        if not isinstance(case, Mapping) or not isinstance(case.get("id"), str):
            raise ValueError("Phase 2D fixture case requires an ID")
        if not isinstance(case.get("insight_id"), str) or not isinstance(case.get("source_video_id"), str):
            raise ValueError(f"Phase 2D fixture case {case['id']} requires source identifiers")
        if not isinstance(case.get("eligible"), bool) or not isinstance(case.get("expected_propositions"), list):
            raise ValueError(f"Phase 2D fixture case {case['id']} has invalid proposition labels")
        if case["eligible"]:
            if len(case["expected_propositions"]) != 1:
                raise ValueError(
                    f"Phase 2D fixture case {case['id']} has inconsistent eligible/expected-proposition labels: "
                    "eligible cases require exactly one expected proposition"
                )
        elif case["expected_propositions"]:
            raise ValueError(
                f"Phase 2D fixture case {case['id']} has inconsistent eligible/expected-proposition labels: "
                "ineligible cases require zero expected propositions"
            )
        for expected in case["expected_propositions"]:
            groups = expected.get("semantic_field_token_groups")
            if groups is not None and (
                not isinstance(groups, Mapping)
                or not groups
                or not {"subject", "predicate", "effect"}.issubset(groups)
                or (expected.get("condition_source") is not None and "condition" not in groups)
                or any(field not in {"subject", "predicate", "effect", "condition"} or not isinstance(values, list) or not values or any(not isinstance(group, list) or not group or not all(isinstance(token, str) and token for token in group) for group in values) for field, values in groups.items())
            ):
                raise ValueError(f"Phase 2D fixture case {case['id']} has invalid semantic field token groups")
            operator = expected.get("condition_operator")
            if operator is not None and (not isinstance(operator, str) or operator not in _CONDITION_OPERATORS):
                raise ValueError(f"Phase 2D fixture case {case['id']} has invalid condition operator")
            condition = expected.get("condition_source")
            if isinstance(condition, str) and _tokenize(condition) and _tokenize(condition)[0] in _CONDITION_OPERATORS and operator != _tokenize(condition)[0]:
                raise ValueError(f"Phase 2D fixture case {case['id']} must preserve its condition operator")
            normalization = expected.get("expected_normalization")
            if not isinstance(normalization, Mapping) or set(normalization) != {
                "actor_concept", "event_relation", "effect_concept",
            }:
                raise ValueError(
                    f"Phase 2D fixture case {case['id']} requires reviewed expected_normalization labels"
                )
            actor_concept = normalization.get("actor_concept")
            event_relation = normalization.get("event_relation")
            effect_concept = normalization.get("effect_concept")
            if actor_concept is not None and actor_concept not in STRATEGIC_CONCEPTS:
                raise ValueError(f"Phase 2D fixture case {case['id']} has invalid normalized actor concept")
            if event_relation is not None and event_relation not in RELATION_TYPES:
                raise ValueError(f"Phase 2D fixture case {case['id']} has invalid normalized event relation")
            if effect_concept is not None and effect_concept not in STRATEGIC_CONCEPTS:
                raise ValueError(f"Phase 2D fixture case {case['id']} has invalid normalized effect concept")
            rationale = expected.get("normalization_rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError(f"Phase 2D fixture case {case['id']} requires a normalization rationale")
        result.append(dict(case))
    _validate_held_out_separation(
        result, Path(held_out_path) if held_out_path is not None else DEFAULT_HELD_OUT_FIXTURE,
    )
    return tuple(result)


def evaluate_source_modes(
    cases: Iterable[Mapping[str, Any]], *, resolver: SourceWindowResolver,
    extractor: Callable[[PropositionPacket], ExtractionResult],
    modes: tuple[SourceMode, ...] = ("insight", "transcript", "combined"),
) -> dict[str, Any]:
    """Evaluate mocked or live Stage A extraction without persistence.

    A transcript/combined mode is *unavailable*, not a safe zero, when the
    resolver cannot verify a local bronze window.
    """
    output = []
    for case in cases:
        window = resolver.resolve(str(case["insight_id"]), expected_source_id=str(case["source_video_id"]))
        entries = []
        for mode in modes:
            if mode in {"transcript", "combined"} and not window.resolved:
                entries.append({"mode": mode, "status": "unavailable", "reason": window.alignment_method})
                continue
            packet = PropositionPacket(
                evidence_id=str(case["insight_id"]), source_video_id=str(case["source_video_id"]),
                insight_text=window.insight_text, mode=mode, source_window=window if mode != "insight" else None,
            )
            try:
                actual = extractor(packet)
            except Exception as exc:  # Provider/parser failures are reported, not transformed into zero relations.
                expected = list(case["expected_propositions"])
                entries.append({
                    "mode": mode, "status": "failure", "reason": type(exc).__name__,
                    "predicted_count": 0, "matched_count": 0, "exact_matched_count": 0,
                    "expected_count": len(expected), "false_positive_count": 0,
                    "missed_count": len(expected), "propositions": [],
                    "slot_scores": _provider_failure_slots(len(expected)),
                })
                continue
            entries.append(_score_mode(mode, case, packet, actual))
        output.append({
            "case_id": case["id"], "eligible": case["eligible"],
            "source_window": _window_json(window), "modes": entries,
        })
    return {"cases": output, "metrics": _summarize_source_modes(output, modes)}


def _score_mode(
    mode: SourceMode, case: Mapping[str, Any], packet: PropositionPacket,
    actual: ExtractionResult,
) -> dict[str, Any]:
    if isinstance(actual, StageAExtraction):
        return _score_stage_a_mode(mode, case, packet, actual)
    return _score_legacy_mode(mode, case, packet, actual)


def _score_legacy_mode(
    mode: SourceMode, case: Mapping[str, Any], packet: PropositionPacket,
    actual: tuple[ExtractedProposition, ...],
) -> dict[str, Any]:
    expected = list(case["expected_propositions"])
    produced = list(actual)
    pairings = _pair(expected, produced, packet)
    partial = _pair_partial(expected, produced, packet)
    matched_count = len(pairings)
    exact_count = sum(1 for index in pairings if _matches(produced[pairings[index]], expected[index], packet))
    return {
        "mode": mode, "status": "completed", "predicted_count": len(produced),
        "matched_count": matched_count, "exact_matched_count": exact_count, "expected_count": len(expected),
        "false_positive_count": len(produced) - matched_count,
        "missed_count": len(expected) - matched_count,
        "propositions": [_proposition_json(item) for item in produced],
        "slot_scores": _slot_scores(expected, pairings, partial, produced, packet),
        "comparisons": _comparisons(expected, pairings, partial, produced, packet),
    }


def _score_stage_a_mode(
    mode: SourceMode, case: Mapping[str, Any], packet: PropositionPacket,
    actual: StageAExtraction,
) -> dict[str, Any]:
    expected = list(case["expected_propositions"])
    produced = list(actual.propositions)
    frame = actual.frames[0] if actual.frames else None
    grounded = _valid_stage_grounding(actual, frame, packet)
    slot_conflicts = _stage_slot_conflicts(actual, frame)
    direction_consistent = _stage_direction_consistent(actual, frame)
    direction = (
        _stage_resolved_direction(actual, frame)
        if grounded and direction_consistent else None
    )
    direction_reached = grounded and (
        actual.causal_direction is not None
        or (frame is not None and frame.causal_direction is not None)
    )
    assembly_consistent = _stage_assembly_consistent(actual, frame, packet)
    coherent = (
        actual.failure_stage is None
        and grounded
        and not slot_conflicts
        and direction_consistent
        and assembly_consistent
    )
    recovered = _recovered_slots(actual, frame) if grounded else {}
    for role in slot_conflicts:
        recovered.pop(role, None)
    span_texts = _stage_span_texts(actual, frame) if grounded else ()
    slot_reached_overrides: dict[str, bool] = {}
    stage_reached_overrides: dict[str, bool] = {}
    if grounded:
        for role in ("actor", "event", "effect", "condition"):
            if role in slot_conflicts:
                slot_reached_overrides[role] = True
                stage_reached_overrides[role + "_extraction"] = True
        slot_reached_overrides["causal_direction"] = direction_reached
        stage_reached_overrides["causal_direction"] = direction_reached
    if coherent:
        pairings = _pair(expected, produced, packet)
        partial = _pair_partial(expected, produced, packet)
    elif grounded:
        pairings = {}
        partial = _pair_partial(expected, produced, packet)
    else:
        pairings = {}
        partial = {}
    matched_count = len(pairings)
    exact_count = sum(1 for index in pairings if _matches(produced[pairings[index]], expected[index], packet))
    candidate_catalog_coverage = _candidate_catalog_coverage(
        expected, actual.candidate_catalog, packet,
    )
    return {
        "mode": mode,
        "status": "completed" if actual.failure_stage is None else "failure",
        "reason": actual.failure_stage,
        "predicted_count": len(produced),
        "matched_count": matched_count, "exact_matched_count": exact_count, "expected_count": len(expected),
        "false_positive_count": len(produced) - matched_count,
        "missed_count": len(expected) - matched_count,
        "propositions": [_proposition_json(item) for item in produced],
        "coherence": {
            "coherent": coherent,
            "grounded": grounded,
            "slots_consistent": not slot_conflicts,
            "slot_conflicts": sorted(slot_conflicts),
            "direction_consistent": direction_consistent,
            "assembly_consistent": assembly_consistent,
        },
        "artifacts": [asdict(item) for item in actual.artifacts],
        "candidate_catalog": [asdict(item) for item in actual.candidate_catalog],
        "candidate_catalog_coverage": candidate_catalog_coverage,
        "evidence_spans": _stage_evidence_spans(actual, frame),
        "recovered_slots": _stage_slot_entries(actual, frame),
        "semantic_frames": [asdict(item) for item in actual.frames],
        "reached_stages": [artifact.stage for artifact in actual.artifacts if artifact.failure is None],
        "first_failure": _first_failure(actual),
        "slot_scores": _slot_scores(
            expected, pairings, partial, produced, packet, frame=frame,
            recovered=recovered, direction=direction,
            span_texts=span_texts,
            unsupported=actual.unsupported_slot_count, invented=_invented_slots(actual),
            reached_overrides=slot_reached_overrides,
        ),
        "comparisons": _comparisons(
            expected, pairings, partial, produced, packet, frame=frame, recovered=recovered,
            direction=direction, span_texts=span_texts, failure_stage=actual.failure_stage,
            evidence_localization_ran=_evidence_localization_ran(actual),
            reached_overrides=stage_reached_overrides,
        ),
    }


def _stage_slot_conflicts(
    actual: StageAExtraction, frame: SourceSemanticFrame | None,
) -> frozenset[str]:
    """Roles whose stage slots disagree with the semantic frame.

    Slots are compared only where they overlap: a role present in both
    ``actual.slots`` and ``frame`` must carry the identical alignment, and a
    role asserted by one side while the other explicitly lacks it is a
    contradiction.  Missing keys (unreached stages) are not conflicts.
    """
    if frame is None or not actual.slots:
        return frozenset()
    frame_slots = {
        "actor": frame.actor,
        "event": frame.event,
        "effect": frame.effect,
    }
    if frame.condition is not None:
        frame_slots["condition"] = frame.condition
    conflicts: set[str] = set()
    for role, frame_slot in frame_slots.items():
        actual_slot = actual.slots.get(role)
        if actual_slot is None:
            if role in actual.slots:
                conflicts.add(role)
        elif actual_slot != frame_slot:
            conflicts.add(role)
    for role, actual_slot in actual.slots.items():
        if actual_slot is not None and role not in frame_slots:
            conflicts.add(role)
    return frozenset(conflicts)


def _stage_direction_consistent(
    actual: StageAExtraction, frame: SourceSemanticFrame | None,
) -> bool:
    """Both present causal direction labels must agree; overlap is required."""
    if frame is None or actual.causal_direction is None or frame.causal_direction is None:
        return True
    return actual.causal_direction == frame.causal_direction


def _stage_resolved_direction(
    actual: StageAExtraction, frame: SourceSemanticFrame | None,
) -> str | None:
    """Resolved causal direction; None when present labels contradict."""
    if frame is None:
        return actual.causal_direction
    if actual.causal_direction is None:
        return frame.causal_direction
    if actual.causal_direction != frame.causal_direction:
        return None
    return actual.causal_direction


def _stage_assembly_consistent(
    actual: StageAExtraction, frame: SourceSemanticFrame | None, packet: PropositionPacket,
) -> bool:
    """Actual propositions must equal the frame's deterministic assembly.

    A produced proposition that is not the output of
    :func:`pipeline.proposition_extract.assemble_grounded_proposition` on the
    validated frame (including any proposition produced without a frame, or a
    missing proposition the frame would deterministically assemble) is
    suppressed from assembled/exact scoring.
    """
    if frame is None:
        return not actual.propositions
    try:
        assembled = assemble_grounded_proposition(frame, packet.evidence_id)
    except ValueError:
        assembled = None
    expected_assembly = (assembled,) if assembled is not None else ()
    return tuple(actual.propositions) == expected_assembly


def _pair(
    expected: list[Mapping[str, Any]], produced: list[ExtractedProposition],
    packet: PropositionPacket,
) -> dict[int, int]:
    """Greedily pair produced propositions to unmatched expected labels."""
    remaining = set(range(len(expected)))
    pairings: dict[int, int] = {}
    for index, item in enumerate(produced):
        match_index = next((item_index for item_index in remaining if _semantic_match(item, expected[item_index], packet)), None)
        if match_index is not None:
            remaining.discard(match_index)
            pairings[match_index] = index
    return pairings


def _pair_partial(
    expected: list[Mapping[str, Any]], produced: list[ExtractedProposition],
    packet: PropositionPacket,
) -> dict[int, int]:
    """Pair expected labels to the best partially matching produced item."""
    remaining = set(range(len(expected)))
    partial: dict[int, int] = {}
    for index, item in enumerate(produced):
        best_index: int | None = None
        best_key = (-1, -1)
        for item_index in remaining:
            hits = sum(1 for _, field in _SLOT_SCORE_FIELDS if _field_semantic_hit(item, expected[item_index], field))
            exact = sum(1 for _, field in _SLOT_SCORE_FIELDS if _field_exact_hit(item, expected[item_index], field))
            key = (hits, exact)
            if key > best_key:
                best_key = key
                best_index = item_index
        if best_index is not None and best_key[0] > 0:
            remaining.discard(best_index)
            partial[best_index] = index
    return partial


def _comparisons(
    expected: list[Mapping[str, Any]], pairings: Mapping[int, int],
    partial: Mapping[int, int], produced: list[ExtractedProposition], packet: PropositionPacket,
    frame: SourceSemanticFrame | None = None,
    recovered: Mapping[str, str | None] | None = None,
    *,
    direction: str | None = None,
    span_texts: Iterable[str] = (),
    failure_stage: str | None = None,
    evidence_localization_ran: bool = False,
    reached_overrides: Mapping[str, bool] | None = None,
) -> list[dict[str, Any]]:
    comparisons = []
    for index in range(len(expected)):
        full_item = produced[pairings[index]] if index in pairings else None
        item = produced[partial[index]] if index in partial else full_item
        comparison: dict[str, Any] = {
            "expected_index": index,
            "matched": full_item is not None,
            "exact": full_item is not None and _matches(full_item, expected[index], packet),
            "produced": _proposition_json(item) if item is not None else None,
            "expected": expected[index],
        }
        if recovered is not None:
            semantic_hit = _comparison_semantic_proposition_hit(recovered, direction, expected[index])
            for slot, field in _SLOT_SCORE_FIELDS:
                comparison[slot + "_hit"] = slot in recovered and _slot_field_semantic_hit(recovered[slot], expected[index], field)
                comparison[slot + "_exact"] = slot in recovered and _slot_field_exact_hit(recovered[slot], expected[index], field)
            comparison["condition_operator_hit"] = (
                _condition_operator_text_hit(recovered["condition"], expected[index])
                if "condition" in recovered else False
            )
            comparison["evidence_span_hit"] = _evidence_span_hit(expected[index], span_texts)
            comparison["causal_direction_hit"] = direction == _EXPECTED_CAUSAL_DIRECTION
            comparison["normalization_completed"] = frame is not None and frame.normalization is not None
            comparison["normalization_abstained"] = frame is not None and _normalization_abstained(frame)
            comparison["normalization_failed"] = frame is not None and frame.normalization_failure is not None
            comparison["normalization_hit"] = _normalization_hit(frame, expected[index])
            comparison["produced_normalization"] = (
                asdict(frame.normalization)
                if frame is not None and frame.normalization is not None else None
            )
            reached = _stage_reached_flags(
                frame=frame, recovered=recovered, direction=direction,
                span_texts=span_texts, produced=produced,
                evidence_localization_ran=evidence_localization_ran,
                reached_overrides=reached_overrides,
            )
        else:
            semantic_hit = full_item is not None
            for slot, field in _SLOT_SCORE_FIELDS:
                comparison[slot + "_hit"] = item is not None and _field_semantic_hit(item, expected[index], field)
                comparison[slot + "_exact"] = item is not None and _field_exact_hit(item, expected[index], field)
            comparison["condition_operator_hit"] = item is not None and _condition_operator_hit(item, expected[index])
            comparison["normalization_hit"] = False
            comparison["produced_normalization"] = None
            reached = {
                "evidence_localization": False,
                "actor_extraction": item is not None,
                "event_extraction": item is not None,
                "effect_extraction": item is not None,
                "condition_extraction": item is not None,
                "causal_direction": False,
                "ontology_normalization": False,
                "proposition_assembly": len(produced) > 0,
                "exact_decomposition": len(produced) > 0,
            }
        comparison["semantic_proposition_hit"] = semantic_hit
        comparison["first_failed_transformation"] = _first_failed_transformation(
            comparison, reached=reached, failure_stage=failure_stage,
        )
        comparisons.append(comparison)
    return comparisons


def _slot_scores(
    expected: list[Mapping[str, Any]], pairings: Mapping[int, int],
    partial: Mapping[int, int], produced: list[ExtractedProposition], packet: PropositionPacket,
    frame: SourceSemanticFrame | None = None, *,
    recovered: Mapping[str, str | None] | None = None, unsupported: int = 0,
    invented: Mapping[str, Any] | None = None, direction: str | None = None,
    span_texts: Iterable[str] = (),
    reached_overrides: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    for slot, field in _SLOT_SCORE_FIELDS:
        if recovered is not None:
            scores[slot] = _recovered_slot_score(expected, recovered, slot, field)
        else:
            scores[slot] = {
                "hit_count": sum(
                    1 for index in range(len(expected))
                    if index in partial and _official_slot_hit(
                        getattr(produced[partial[index]].proposition, field + "_source"),
                        expected[index], field,
                    )
                ),
                "expected_count": len(expected),
            }
    if recovered is not None:
        scores["semantic_proposition"] = _grounded_semantic_proposition_score(expected, recovered, direction)
    else:
        scores["semantic_proposition"] = {"hit_count": len(pairings), "expected_count": len(expected)}
    scores["assembled_proposition"] = {"hit_count": len(pairings), "expected_count": len(expected)}
    scores["exact_decomposition"] = {
        "hit_count": sum(
            1 for index in pairings if _matches(produced[pairings[index]], expected[index], packet)
        ),
        "expected_count": len(expected),
    }
    scores["normalization"] = {
        "hit_count": sum(1 for label in expected if _normalization_hit(frame, label)),
        "expected_count": len(expected),
    }
    if recovered is not None:
        scores["evidence_span"] = {
            "hit_count": sum(1 for index in range(len(expected)) if _evidence_span_hit(expected[index], span_texts)),
            "expected_count": len(expected),
        }
        scores["causal_direction"] = {
            "hit_count": len(expected) if direction == _EXPECTED_CAUSAL_DIRECTION else 0,
            "expected_count": len(expected),
        }
        scores["normalization_stage"] = {
            "denominator": len(expected),
            "reached_count": len(expected) if frame is not None else 0,
            "completed_count": len(expected) if frame is not None and frame.normalization is not None else 0,
            "abstained_count": len(expected) if frame is not None and _normalization_abstained(frame) else 0,
            "mapped_count": len(expected) if frame is not None and frame.normalization is not None and not _normalization_abstained(frame) else 0,
            "failed_count": len(expected) if frame is not None and frame.normalization_failure is not None else 0,
        }
        slot_reached = {
            "evidence_span": bool(tuple(span_texts)),
            "actor": "actor" in recovered,
            "event": "event" in recovered,
            "effect": "effect" in recovered,
            "condition": "condition" in recovered,
            "causal_direction": direction is not None,
            "normalization": frame is not None,
            "semantic_proposition": (
                all(role in recovered for role, _ in _SLOT_SCORE_FIELDS)
                and direction is not None
            ),
            "assembled_proposition": len(produced) > 0,
            "exact_decomposition": len(produced) > 0,
        }
        if reached_overrides:
            slot_reached.update(reached_overrides)
        scores["slot_reached"] = _slot_reached_diagnostics(scores, reached=slot_reached)
    scores["unsupported_slots"] = {"count": unsupported}
    scores["invented_slots"] = invented if invented is not None else {"count": 0}
    return scores


def _grounded_semantic_proposition_score(
    expected: list[Mapping[str, Any]], recovered: Mapping[str, str | None],
    direction: str | None,
) -> dict[str, int]:
    """Primary Phase 2E semantic proposition score from the recovered frame.

    A proposition is a semantic hit only when every reviewed role slot is
    recovered with a semantic hit and the causal direction is forward.  This
    score is independent of ontology normalization and deterministic
    proposition assembly, so a correct frame followed by a normalization
    failure still scores a semantic hit, while reversed causal direction or a
    partial slot failure remains a miss.
    """
    if direction != _EXPECTED_CAUSAL_DIRECTION:
        return {"hit_count": 0, "expected_count": len(expected)}
    return {
        "hit_count": sum(
            1 for label in expected
            if all(
                role in recovered and _official_slot_hit(recovered[role], label, field)
                for role, field in _SLOT_SCORE_FIELDS
            )
        ),
        "expected_count": len(expected),
    }


def _comparison_semantic_proposition_hit(
    recovered: Mapping[str, str | None], direction: str | None,
    expected_label: Mapping[str, Any],
) -> bool:
    if direction != _EXPECTED_CAUSAL_DIRECTION:
        return False
    return all(
        role in recovered and _official_slot_hit(recovered[role], expected_label, field)
        for role, field in _SLOT_SCORE_FIELDS
    )


def _first_failed_transformation(
    comparison: Mapping[str, Any], *, reached: Mapping[str, bool],
    failure_stage: str | None,
) -> str | None:
    """First semantic transformation lost for one expected proposition.

    Stages are walked in pipeline order; a transformation is reported as lost
    only when its stage was reached (or is the explicit pipeline failure
    stage), so later never-reached stages are not misattributed as the first
    loss.  Returns ``None`` when every reached transformation passed.
    """
    for stage in _TRANSFORMATION_STAGES:
        if failure_stage is not None and failure_stage == stage:
            return stage
        failed = _stage_failed(comparison, stage)
        if failed is None or not failed or not reached.get(stage, False):
            continue
        return stage
    return None


def _stage_failed(comparison: Mapping[str, Any], stage: str) -> bool | None:
    """Whether the comparison flags mark ``stage`` as lost; None if not evaluable."""
    flag = comparison.get(_STAGE_FLAG_KEYS[stage])
    if flag is None:
        return None
    return not bool(flag)  # *_hit / matched / exact are True on success


def _stage_reached_flags(
    *, frame: SourceSemanticFrame | None, recovered: Mapping[str, str | None],
    direction: str | None, span_texts: Iterable[str], produced: list[ExtractedProposition],
    evidence_localization_ran: bool = False,
    reached_overrides: Mapping[str, bool] | None = None,
) -> dict[str, bool]:
    """Whether each pipeline stage ran for this extraction.

    ``reached_overrides`` reports stages that ran but produced contradictory
    output (for example a causal direction that disagrees with the frame), so
    first-loss attribution still lands on the contradictory stage.
    """
    reached = {
        "evidence_localization": bool(tuple(span_texts)) or evidence_localization_ran,
        "actor_extraction": "actor" in recovered,
        "event_extraction": "event" in recovered,
        "effect_extraction": "effect" in recovered,
        "condition_extraction": "condition" in recovered,
        "causal_direction": direction is not None,
        "ontology_normalization": frame is not None,
        "proposition_assembly": len(produced) > 0,
        "exact_decomposition": len(produced) > 0,
    }
    if reached_overrides:
        reached.update(reached_overrides)
    return reached


def _slot_reached_diagnostics(
    scores: Mapping[str, Any], *, reached: Mapping[str, bool],
) -> dict[str, Any]:
    """Conditional stage diagnostics: reached counts and accuracy when reached."""
    diagnostics: dict[str, Any] = {}
    for slot in _SLOT_RECALL_NAMES:
        if not reached[slot]:
            diagnostics[slot] = {
                "reached_count": 0, "hit_count": 0, "accuracy_when_reached": None,
            }
            continue
        hit_count = int(scores[slot]["hit_count"])
        reached_count = int(scores[slot]["expected_count"])
        diagnostics[slot] = {
            "reached_count": reached_count,
            "hit_count": hit_count,
            "accuracy_when_reached": hit_count / reached_count if reached_count else None,
        }
    return diagnostics


def _provider_failure_slots(expected_count: int) -> dict[str, Any]:
    """Slot scores for a source-available provider/parser failure.

    No Stage A stage ran, so every official slot counts every source-available
    expected proposition as a miss (unreached stages are misses, not
    denominator exclusions); the reached diagnostics expose zero reached
    counts so first-loss analysis stays available.
    """
    return {
        "evidence_span": {"hit_count": 0, "expected_count": expected_count},
        "actor": {"hit_count": 0, "expected_count": expected_count},
        "event": {"hit_count": 0, "expected_count": expected_count},
        "effect": {"hit_count": 0, "expected_count": expected_count},
        "condition": {"hit_count": 0, "expected_count": expected_count},
        "causal_direction": {"hit_count": 0, "expected_count": expected_count},
        "normalization": {"hit_count": 0, "expected_count": expected_count},
        "semantic_proposition": {"hit_count": 0, "expected_count": expected_count},
        "assembled_proposition": {"hit_count": 0, "expected_count": expected_count},
        "exact_decomposition": {"hit_count": 0, "expected_count": expected_count},
        "normalization_stage": {
            "denominator": expected_count, "reached_count": 0,
            "completed_count": 0, "abstained_count": 0, "mapped_count": 0,
            "failed_count": 0,
        },
        "slot_reached": {
            slot: {"reached_count": 0, "hit_count": 0, "accuracy_when_reached": None}
            for slot in _SLOT_RECALL_NAMES
        },
        "unsupported_slots": {"count": 0},
        "invented_slots": {"count": 0},
    }


def _recovered_slots(
    actual: StageAExtraction, frame: SourceSemanticFrame | None,
) -> dict[str, str | None]:
    """Recovered slot text by role; key presence means the slot stage ran."""
    if frame is not None:
        return {
            "actor": frame.actor.text,
            "event": frame.event.text,
            "effect": frame.effect.text,
            "condition": frame.condition.text if frame.condition is not None else None,
        }
    return {role: (slot.text if slot is not None else None) for role, slot in actual.slots.items()}


def _recovered_slot_score(
    expected: list[Mapping[str, Any]], recovered: Mapping[str, str | None],
    role: str, field: str,
) -> dict[str, int]:
    """Score a recovered semantic slot directly against reviewed token groups.

    The official denominator always counts every source-available expected
    proposition; an unreached slot stage is a miss, never an exclusion.
    """
    return {
        "hit_count": sum(
            1 for label in expected
            if role in recovered and _official_slot_hit(recovered[role], label, field)
        ),
        "expected_count": len(expected),
    }


def _official_slot_hit(value: str | None, expected: Mapping[str, Any], field: str) -> bool:
    """Official slot hit: reviewed semantic tokens plus the condition operator.

    Official condition recall requires the reviewed leading condition operator
    when one is present, not merely overlapping semantic tokens; comparison
    ``*_hit`` diagnostics keep reporting the looser token-level hit.
    """
    if not _slot_field_semantic_hit(value, expected, field):
        return False
    if field == "condition":
        return _condition_operator_text_hit(value, expected)
    return True


def _stage_evidence_spans(
    actual: StageAExtraction, frame: SourceSemanticFrame | None,
) -> list[dict[str, Any]]:
    spans = actual.evidence_spans or (frame.evidence_spans if frame is not None else ())
    return [asdict(span) for span in spans]


def _stage_span_texts(
    actual: StageAExtraction, frame: SourceSemanticFrame | None,
) -> tuple[str, ...]:
    spans = actual.evidence_spans or (frame.evidence_spans if frame is not None else ())
    return tuple(span.source_text for span in spans)


def _stage_slot_entries(
    actual: StageAExtraction, frame: SourceSemanticFrame | None,
) -> list[dict[str, Any]]:
    if actual.slots:
        return [
            {"role": role, "slot": asdict(slot) if slot is not None else None}
            for role, slot in actual.slots.items()
        ]
    if frame is not None:
        return [
            {"role": "actor", "slot": asdict(frame.actor)},
            {"role": "event", "slot": asdict(frame.event)},
            {"role": "effect", "slot": asdict(frame.effect)},
            {"role": "condition", "slot": asdict(frame.condition) if frame.condition is not None else None},
        ]
    return []


def _first_failure(actual: StageAExtraction) -> dict[str, str | None] | None:
    if actual.failure_stage is None:
        return None
    failure_type = next(
        (artifact.failure for artifact in actual.artifacts if artifact.stage == actual.failure_stage),
        None,
    )
    return {"stage": actual.failure_stage, "type": failure_type}


def _invented_slots(actual: StageAExtraction) -> dict[str, Any]:
    """Consume the core Stage A explicit invented-ontology count.

    The core contract exposes ``invented_slot_count`` and
    ``invented_slot_taxonomy``; malformed normalization (an ordinary
    ``ValueError``) is never inferred to be invented ontology content.
    """
    count = getattr(actual, "invented_slot_count", None)
    if count is None:
        count = getattr(actual, "invented_ontology_count", 0)
    score: dict[str, Any] = {"count": int(count or 0)}
    taxonomy = getattr(actual, "invented_slot_taxonomy", None)
    if taxonomy is None:
        taxonomy = getattr(actual, "invented_ontology_taxonomy", None)
    if taxonomy:
        score["taxonomy"] = {str(key): int(value) for key, value in dict(taxonomy).items()}
    return score


def _field_semantic_hit(actual: ExtractedProposition, expected: Mapping[str, Any], field: str) -> bool:
    return _slot_field_semantic_hit(getattr(actual.proposition, field + "_source"), expected, field)


def _slot_field_semantic_hit(value: str | None, expected: Mapping[str, Any], field: str) -> bool:
    expected_source = expected.get(field + "_source")
    if expected_source is None:
        return value is None
    if value is None:
        return False
    groups = expected.get("semantic_field_token_groups")
    if not groups:
        return _normalize(value) == _normalize(expected_source)
    field_groups = groups.get(field)
    if not field_groups:
        return _normalize(value) == _normalize(expected_source)
    tokens = set(_tokenize(value))
    return all(tokens & set(_tokenize(" ".join(group))) for group in field_groups)


def _field_exact_hit(actual: ExtractedProposition, expected: Mapping[str, Any], field: str) -> bool:
    return _slot_field_exact_hit(getattr(actual.proposition, field + "_source"), expected, field)


def _slot_field_exact_hit(value: str | None, expected: Mapping[str, Any], field: str) -> bool:
    return _normalize(value) == _normalize(expected.get(field + "_source"))


def _condition_operator_hit(actual: ExtractedProposition, expected: Mapping[str, Any]) -> bool:
    return _condition_operator_text_hit(actual.proposition.condition_source, expected)


def _condition_operator_text_hit(condition_source: str | None, expected: Mapping[str, Any]) -> bool:
    operator = expected.get("condition_operator")
    if operator is None:
        return True
    return _tokenize(condition_source or "")[:1] == (operator,)


def _evidence_span_hit(expected: Mapping[str, Any], span_texts: Iterable[str]) -> bool:
    span_texts = tuple(span_texts)
    if not span_texts:
        return False
    return all(
        expected.get(field + "_source") is None
        or any(expected.get(field + "_source") in text for text in span_texts)
        for field in _SLOT_FIELDS
    )


def _candidate_catalog_coverage(
    expected: list[Mapping[str, Any]], candidates: tuple[ClauseCandidate, ...],
    packet: PropositionPacket,
) -> dict[str, Any]:
    """Measure whether one/two selectable candidates can contain each label.

    This is a deterministic pre-model diagnostic, not semantic model credit.
    It mirrors the evidence-localizer contract: at most two IDs, one source,
    followed by the same overlap/adjacency coalescing used in live Stage A.
    A reviewed source field must occur wholly inside one resulting evidence
    span.  Duplicate IDs and ungrounded alignments are excluded fail-closed.
    """
    id_counts: dict[str, int] = {}
    for candidate in candidates:
        if isinstance(candidate.candidate_id, str):
            id_counts[candidate.candidate_id] = id_counts.get(candidate.candidate_id, 0) + 1
    sources = {source.kind: source.text for source in packet.sources()}
    valid = tuple(
        candidate for candidate in candidates
        if isinstance(candidate.candidate_id, str)
        and bool(candidate.candidate_id)
        and id_counts[candidate.candidate_id] == 1
        and _alignment_grounded(candidate.alignment, sources, packet)
    )

    comparisons_output: list[dict[str, Any]] = []
    for expected_index, label in enumerate(expected):
        covering: tuple[ClauseCandidate, ...] | None = None
        covering_spans: tuple[SourceAlignment, ...] = ()
        for size in (1, 2):
            for selected in combinations(valid, size):
                if len({candidate.alignment.source_kind for candidate in selected}) != 1:
                    continue
                try:
                    spans = coalesce_selected_evidence(
                        tuple(candidate.alignment for candidate in selected), packet,
                    )
                except ValueError:
                    continue
                if _catalog_spans_cover_label(label, spans):
                    covering = selected
                    covering_spans = spans
                    break
            if covering is not None:
                break
        comparisons_output.append({
            "expected_index": expected_index,
            "covered": covering is not None,
            "source_kind": (
                covering[0].alignment.source_kind if covering is not None else None
            ),
            "candidate_ids": (
                [candidate.candidate_id for candidate in covering]
                if covering is not None else []
            ),
            "coalesced_spans": [asdict(span) for span in covering_spans],
        })
    return {
        "hit_count": sum(item["covered"] for item in comparisons_output),
        "expected_count": len(expected),
        "catalog_count": len(candidates),
        "valid_candidate_count": len(valid),
        "invalid_candidate_count": len(candidates) - len(valid),
        "comparisons": comparisons_output,
    }


def _catalog_spans_cover_label(
    expected: Mapping[str, Any], spans: tuple[SourceAlignment, ...],
) -> bool:
    """Whether exact reviewed source fields fit inside selected evidence."""
    return all(
        expected.get(field + "_source") is None
        or any(expected[field + "_source"] in span.source_text for span in spans)
        for field in _SLOT_FIELDS
    )


def _normalization_abstained(frame: SourceSemanticFrame) -> bool:
    normalization = frame.normalization
    if normalization is None:
        return False
    return (
        normalization.actor_concept is None
        and normalization.event_relation is None
        and normalization.effect_concept is None
    )


def _normalization_hit(
    frame: SourceSemanticFrame | None, expected: Mapping[str, Any],
) -> bool:
    """Exact reviewed closed-ontology match, including intentional nulls."""
    reviewed = expected.get("expected_normalization")
    return (
        frame is not None
        and frame.normalization is not None
        and isinstance(reviewed, Mapping)
        and asdict(frame.normalization) == dict(reviewed)
    )


def _matches(actual: ExtractedProposition, expected: Mapping[str, Any], packet: PropositionPacket) -> bool:
    proposition = actual.proposition
    return _has_valid_grounding(actual, packet) and all(
        _normalize(getattr(proposition, field + "_source")) == _normalize(expected.get(field + "_source"))
        for field in ("subject", "predicate", "effect", "condition")
    )


def _semantic_match(actual: ExtractedProposition, expected: Mapping[str, Any], packet: PropositionPacket) -> bool:
    """Match reviewed causal mechanism labels without accepting ungrounded output."""
    if not _has_valid_grounding(actual, packet):
        return False
    groups = expected.get("semantic_field_token_groups")
    if not groups:
        return _matches(actual, expected, packet)
    for field, field_groups in groups.items():
        value = getattr(actual.proposition, field + "_source")
        if value is None:
            return False
        tokens = set(_tokenize(value))
        if not all(tokens & set(_tokenize(" ".join(group))) for group in field_groups):
            return False
    operator = expected.get("condition_operator")
    if operator is not None and _tokenize(actual.proposition.condition_source or "")[:1] != (operator,):
        return False
    return True


def _tokenize(value: str) -> tuple[str, ...]:
    import re
    normalized = value.lower().replace("’", "'").replace("‘", "'").replace("`", "'")
    normalized = re.sub(r"([a-z0-9])'([a-z0-9])", r"\1\2", normalized)
    return tuple(re.findall(r"[a-z0-9]+", normalized))


def _validate_held_out_separation(cases: list[dict[str, Any]], held_out_path: Path) -> None:
    """Mandatory overlap check against the trusted frozen held-out fixture.

    The trusted path is explicit and repository-owned; arbitrary development
    fixtures cannot suppress or redirect this check through metadata, and an
    unavailable frozen fixture is an error rather than a silent skip.  The
    frozen fixture schema fails closed: a structurally malformed (but valid
    JSON) fixture raises before any overlap set is computed, so a damaged
    held-out fixture can never silently yield an empty overlap.
    """
    try:
        held_out_payload = json.loads(held_out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Phase 2D fixture cannot load frozen held-out fixture: " + str(held_out_path)
        ) from exc
    held_out_ids, held_out_sources = _held_out_identifiers(held_out_payload, held_out_path)
    overlap = {str(case["insight_id"]) for case in cases} & held_out_ids
    if overlap:
        raise ValueError("Phase 2D development fixture overlaps frozen held-out insight IDs: " + ", ".join(sorted(overlap)))
    source_overlap = {str(case["source_video_id"]) for case in cases} & held_out_sources
    if source_overlap:
        raise ValueError("Phase 2D development fixture overlaps frozen held-out source IDs: " + ", ".join(sorted(source_overlap)))


def _held_out_identifiers(
    payload: Any, held_out_path: Path,
) -> tuple[set[str], set[str]]:
    """Extract usable held-out insight/source IDs after fail-closed validation.

    The frozen Phase 2B schema requires a top-level mapping, a non-empty
    ``cases`` list, each case a mapping with a non-empty ``evidence`` list,
    and each evidence record a mapping carrying usable (non-empty string)
    ``insight_id`` and ``source_id`` values.  Any structural violation is an
    error instead of silently contributing an empty identifier set.
    """
    def invalid(reason: str) -> ValueError:
        return ValueError(
            "Phase 2D fixture cannot use frozen held-out fixture " + str(held_out_path) + ": " + reason
        )

    if not isinstance(payload, Mapping):
        raise invalid("top-level JSON must be a mapping with a cases list")
    held_cases = payload.get("cases")
    if not isinstance(held_cases, list) or not held_cases:
        raise invalid("cases must be a non-empty list")
    held_out_ids: set[str] = set()
    held_out_sources: set[str] = set()
    for case in held_cases:
        if not isinstance(case, Mapping):
            raise invalid("each case must be a mapping")
        evidence = case.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise invalid("each case requires a non-empty evidence list")
        for item in evidence:
            if not isinstance(item, Mapping):
                raise invalid("each evidence item must be a mapping")
            insight_id = item.get("insight_id")
            source_id = item.get("source_id")
            if not isinstance(insight_id, str) or not insight_id.strip():
                raise invalid("each evidence item requires a usable insight_id")
            if not isinstance(source_id, str) or not source_id.strip():
                raise invalid("each evidence item requires a usable source_id")
            held_out_ids.add(insight_id)
            held_out_sources.add(source_id)
    return held_out_ids, held_out_sources


def _has_valid_grounding(actual: ExtractedProposition, packet: PropositionPacket) -> bool:
    """Defend evaluation against mocked or bypassed ungrounded outputs."""
    values = {
        "subject": actual.proposition.subject_source,
        "predicate": actual.proposition.predicate_source,
        "effect": actual.proposition.effect_source,
    }
    if actual.proposition.condition_source is not None:
        values["condition"] = actual.proposition.condition_source
    if actual.proposition.evidence_ids != (packet.evidence_id,):
        return False
    if len(actual.alignments) != len(values) or {item.field for item in actual.alignments} != set(values):
        return False
    sources = {item.kind: item.text for item in packet.sources()}
    seen = set()
    for alignment in actual.alignments:
        if alignment.field in seen or alignment.source_kind not in sources:
            return False
        seen.add(alignment.field)
        source = sources[alignment.source_kind]
        if (
            alignment.source_text != values[alignment.field]
            or isinstance(alignment.start, bool)
            or isinstance(alignment.end, bool)
            or alignment.start < 0
            or alignment.end <= alignment.start
            or alignment.end > len(source)
            or source[alignment.start:alignment.end] != alignment.source_text
        ):
            return False
        if alignment.source_kind == "transcript":
            if packet.source_window is None or alignment.absolute_start != packet.source_window.window_start + alignment.start or alignment.absolute_end != packet.source_window.window_start + alignment.end:
                return False
        elif alignment.absolute_start is not None or alignment.absolute_end is not None:
            return False
    if len({item.source_kind for item in actual.alignments}) != 1:
        return False
    return True


def _evidence_localization_ran(actual: StageAExtraction) -> bool:
    """Whether the evidence-localization stage produced an artifact."""
    return any(artifact.stage == "evidence_localization" for artifact in actual.artifacts)


def _valid_stage_grounding(
    actual: StageAExtraction, frame: SourceSemanticFrame | None, packet: PropositionPacket,
) -> bool:
    """Defensively validate Stage A frame provenance against the packet.

    A frame is grounded only when every recovered slot and selected evidence
    span quotes its packet source at exact text offsets, transcript spans
    carry verified absolute offsets, all spans and slots share one source, the
    actual and frame span sets agree when both are present, and every
    recovered slot falls inside a selected evidence span.  An ungrounded frame
    cannot score any evidence/slot/semantic hit.
    """
    slots = _scored_slots(actual, frame)
    spans = _scored_spans(actual, frame)
    alignments = tuple(spans) + tuple(slot.alignment for slot in slots)
    if not alignments:
        return True
    sources = {source.kind: source.text for source in packet.sources()}
    kinds = {alignment.source_kind for alignment in alignments}
    if not kinds.issubset(sources) or len(kinds) != 1:
        return False
    for alignment in alignments:
        if not _alignment_grounded(alignment, sources, packet):
            return False
    if not _spans_agree(actual, frame):
        return False
    if not spans:
        return False
    return all(_inside_selected_span(slot.alignment, spans) for slot in slots)


def _scored_slots(
    actual: StageAExtraction, frame: SourceSemanticFrame | None,
) -> tuple[SemanticSlot, ...]:
    if frame is not None:
        slots = [frame.actor, frame.event, frame.effect]
        if frame.condition is not None:
            slots.append(frame.condition)
        return tuple(slots)
    return tuple(slot for slot in actual.slots.values() if slot is not None)


def _scored_spans(
    actual: StageAExtraction, frame: SourceSemanticFrame | None,
) -> tuple[SourceAlignment, ...]:
    if actual.evidence_spans:
        return tuple(actual.evidence_spans)
    if frame is not None:
        return tuple(frame.evidence_spans)
    return ()


def _spans_agree(actual: StageAExtraction, frame: SourceSemanticFrame | None) -> bool:
    if frame is None or not actual.evidence_spans or not frame.evidence_spans:
        return True
    return actual.evidence_spans == frame.evidence_spans


def _alignment_grounded(
    alignment: SourceAlignment | PropositionAlignment,
    sources: Mapping[str, str], packet: PropositionPacket,
) -> bool:
    """Validate one span/slot alignment against its packet source text."""
    source = sources.get(alignment.source_kind)
    if source is None:
        return False
    if isinstance(alignment.start, bool) or isinstance(alignment.end, bool):
        return False
    if alignment.start < 0 or alignment.end <= alignment.start or alignment.end > len(source):
        return False
    if source[alignment.start:alignment.end] != alignment.source_text:
        return False
    if alignment.source_kind == "transcript":
        window = packet.source_window
        if window is None or window.window_start is None:
            return False
        expected_start = window.window_start + alignment.start
        expected_end = window.window_start + alignment.end
        if alignment.absolute_start != expected_start or alignment.absolute_end != expected_end:
            return False
    elif alignment.absolute_start is not None or alignment.absolute_end is not None:
        return False
    return True


def _inside_selected_span(
    alignment: SourceAlignment, spans: tuple[SourceAlignment, ...],
) -> bool:
    return any(
        span.source_kind == alignment.source_kind
        and span.start <= alignment.start
        and alignment.end <= span.end
        for span in spans
    )


def _normalize(value: object) -> str | None:
    return " ".join(str(value).lower().split()) if value is not None else None


def _proposition_json(value: ExtractedProposition) -> dict[str, Any]:
    return {
        "proposition": asdict(value.proposition),
        "alignments": [asdict(item) for item in value.alignments],
    }


def _window_json(window: SourceWindow) -> dict[str, Any]:
    return {
        "alignment_method": window.alignment_method, "alignment_score": window.alignment_score,
        "resolved": window.resolved, "window_start": window.window_start, "window_end": window.window_end,
    }


def _summarize_source_modes(cases: list[dict[str, Any]], modes: tuple[SourceMode, ...]) -> dict[str, dict[str, Any]]:
    summary = {}
    for mode in modes:
        entries = [entry for case in cases for entry in case["modes"] if entry["mode"] == mode]
        completed = [entry for entry in entries if entry["status"] == "completed"]
        source_available = [entry for case in cases if case["eligible"] for entry in case["modes"] if entry["mode"] == mode and entry["status"] != "unavailable"]
        safe_zero = [entry for case in cases if not case["eligible"] for entry in case["modes"] if entry["mode"] == mode and entry["status"] != "unavailable"]
        eligible_entries = [entry for case in cases if case["eligible"] for entry in case["modes"] if entry["mode"] == mode and entry["status"] != "unavailable"]
        tp = sum(item["matched_count"] for item in completed)
        exact_tp = sum(item["exact_matched_count"] for item in completed)
        fp = sum(item["false_positive_count"] for item in completed)
        fn = sum(item["missed_count"] for item in eligible_entries)
        eligible_entry_count = sum(1 for case in cases if case["eligible"] for item in case["modes"] if item["mode"] == mode)
        slot_recall = {}
        for slot in _SLOT_RECALL_NAMES:
            scored = [entry for entry in eligible_entries if slot in entry.get("slot_scores", {})]
            denominator = sum(entry["slot_scores"][slot]["expected_count"] for entry in scored)
            hits = sum(entry["slot_scores"][slot]["hit_count"] for entry in scored)
            slot_recall[slot] = {
                "hit_count": hits, "denominator": denominator,
                "recall": hits / denominator if denominator else None,
            }
        slot_reached = {}
        for slot in _SLOT_RECALL_NAMES:
            reached_entries = [
                entry for entry in eligible_entries
                if "slot_reached" in entry.get("slot_scores", {}) and slot in entry["slot_scores"]["slot_reached"]
            ]
            reached_count = sum(entry["slot_scores"]["slot_reached"][slot]["reached_count"] for entry in reached_entries)
            hits = sum(entry["slot_scores"]["slot_reached"][slot]["hit_count"] for entry in reached_entries)
            slot_reached[slot] = {
                "reached_count": reached_count, "hit_count": hits,
                "denominator": reached_count,
                "accuracy_when_reached": hits / reached_count if reached_count else None,
            }
        normalization_scored = [entry for entry in eligible_entries if "normalization_stage" in entry.get("slot_scores", {})]
        normalization_denominator = sum(entry["slot_scores"]["normalization_stage"]["denominator"] for entry in normalization_scored)
        normalization_stage = {
            "denominator": normalization_denominator,
            "reached_count": sum(entry["slot_scores"]["normalization_stage"]["reached_count"] for entry in normalization_scored),
            "completed_count": sum(entry["slot_scores"]["normalization_stage"]["completed_count"] for entry in normalization_scored),
            "abstained_count": sum(entry["slot_scores"]["normalization_stage"]["abstained_count"] for entry in normalization_scored),
            "mapped_count": sum(entry["slot_scores"]["normalization_stage"]["mapped_count"] for entry in normalization_scored),
            "failed_count": sum(entry["slot_scores"]["normalization_stage"]["failed_count"] for entry in normalization_scored),
        }
        candidate_catalog_scored = [
            entry for entry in eligible_entries
            if "candidate_catalog_coverage" in entry
        ]
        candidate_catalog_denominator = sum(
            entry["candidate_catalog_coverage"]["expected_count"]
            for entry in candidate_catalog_scored
        )
        candidate_catalog_hits = sum(
            entry["candidate_catalog_coverage"]["hit_count"]
            for entry in candidate_catalog_scored
        )
        eligible_expected_count = sum(entry["expected_count"] for entry in eligible_entries)
        candidate_catalog_coverage = {
            "hit_count": candidate_catalog_hits,
            "denominator": candidate_catalog_denominator,
            "recall": (
                candidate_catalog_hits / candidate_catalog_denominator
                if candidate_catalog_denominator else None
            ),
            "evaluated_entry_count": len(candidate_catalog_scored),
            "eligible_entry_count": len(eligible_entries),
            "complete": (
                candidate_catalog_denominator == eligible_expected_count
                and len(candidate_catalog_scored) == len(eligible_entries)
            ),
        }
        summary[mode] = {
            "case_count": len(entries), "completed_case_count": len(completed),
            "unavailable_case_count": sum(item["status"] == "unavailable" for item in entries),
            "failure_case_count": sum(item["status"] == "failure" for item in entries),
            "eligible_source_coverage": len(source_available) / eligible_entry_count if eligible_entry_count else None,
            "proposition_precision": tp / (tp + fp) if tp + fp else (0.0 if eligible_entries else None),
            "proposition_recall": tp / (tp + fn) if tp + fn else (0.0 if eligible_entries else None),
            "exact_source_proposition_recall": exact_tp / (tp + fn) if tp + fn else (0.0 if eligible_entries else None),
            "unsupported_proposition_rate": fp / max(tp + fp, 1),
            "safe_zero_accuracy": sum(item["status"] == "completed" and item["predicted_count"] == 0 for item in safe_zero) / len(safe_zero) if safe_zero else 0.0,
            "eligible_case_count": len(eligible_entries),
            "slot_recall": slot_recall,
            "slot_reached": slot_reached,
            "normalization_stage": normalization_stage,
            "candidate_catalog_coverage": candidate_catalog_coverage,
            "unsupported_slot_total": sum(entry.get("slot_scores", {}).get("unsupported_slots", {}).get("count", 0) for entry in eligible_entries),
            "invented_slot_total": sum(entry.get("slot_scores", {}).get("invented_slots", {}).get("count", 0) for entry in eligible_entries),
        }
    return summary
