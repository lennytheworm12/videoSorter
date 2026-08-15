"""Evaluate Phase 2 extraction against a small source-grounded reference set.

Default mode validates the dataset only. ``--live`` invokes the configured cheap
relation-extraction model; it never persists its output.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from core.llm import BACKEND, MODEL, chat
from core.strategic_types import EvidenceRef, StrategicRelation
from pipeline.relation_extract import (
    RELATION_FLASH_MODEL,
    RELATION_PRO_MODEL,
    EvidenceItem,
    ExtractionDecision,
    ExtractionPacket,
    compile_candidates,
    extract_relation_trace,
    extract_relations,
    packet_from_insight_ids,
)


@dataclass(frozen=True)
class ExtractionCase:
    id: str
    packet: ExtractionPacket
    expected: tuple["ExpectedRelation", ...]


@dataclass(frozen=True)
class ExpectedRelation:
    """Human-reviewed reference relation and the condition cues it must retain."""

    relation: StrategicRelation
    required_condition_terms: tuple[str, ...] = ()


def load_cases(path: str | Path, *, source_db: str | Path | None = None) -> tuple[ExtractionCase, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    for raw_case in payload["cases"]:
        packet = (
            packet_from_insight_ids(str(source_db), [item["insight_id"] for item in raw_case["evidence"]])
            if source_db
            else ExtractionPacket(
                evidence=tuple(EvidenceItem(**item) for item in raw_case["evidence"]),
                ability_aliases=raw_case.get("ability_aliases", {}),
                entity_aliases=raw_case.get("entity_aliases", {}),
                ontology_version=payload["ontology_version"], prompt_version=payload["prompt_version"],
            )
        )
        expected = tuple(
            _reference_relation(raw_case["id"], index, raw, packet)
            for index, raw in enumerate(raw_case.get("expected", []), start=1)
        )
        cases.append(ExtractionCase(raw_case["id"], packet, expected))
    return tuple(cases)


def evaluate_cases(
    cases: tuple[ExtractionCase, ...],
    extractor: Callable[[ExtractionPacket], tuple[ExtractionDecision, ...] | object],
) -> dict[str, Any]:
    """Compare normalized relation structure without treating prose effects as identity."""
    outputs = []
    matched = expected_total = accepted_total = review_total = rejected_total = 0
    subject_correct = type_correct = object_correct = condition_correct = provenance_correct = 0
    review_matches = 0
    disposition_reasons: Counter[str] = Counter()
    failure_stages: Counter[str] = Counter()
    unresolved_reference_entities = 0
    for case in cases:
        trace = extractor(case.packet)
        failure = None
        if hasattr(trace, "decisions") and hasattr(trace, "failure_stage"):
            decisions = trace.decisions
            failure = {
                "stage": trace.failure_stage, "type": trace.failure_type,
                "message": trace.failure_message, "latency_ms": trace.latency_ms,
                "raw_response": trace.raw_response,
            }
        else:
            decisions = trace
        if failure and failure["stage"]:
            failure_stages[failure["stage"]] += 1
        expected = list(case.expected)
        unresolved = _unresolved_reference_entities(expected, case.packet)
        unresolved_reference_entities += len(unresolved)
        accepted = [item.relation for item in decisions if item.status == "accepted" and item.relation]
        reviewed = [item.relation for item in decisions if item.status == "review" and item.relation]
        review_total += len(reviewed)
        rejected_total += sum(item.status == "rejected" for item in decisions)
        for decision in decisions:
            for warning in decision.warnings:
                disposition_reasons[warning] += 1
        expected_total += len(expected)
        accepted_total += len(accepted)
        unmatched = list(expected)
        case_matches = 0
        for actual in accepted:
            match = next(
                (
                    item for item in unmatched
                    if _triple(item.relation) == _triple(actual)
                    and _condition_matches(
                        actual.condition, item.required_condition_terms, item.relation.condition,
                    )
                ),
                None,
            )
            if match is None:
                continue
            reference = match.relation
            subject_correct += actual.subject_key == reference.subject_key and actual.subject_type == reference.subject_type
            type_correct += actual.relation_type == reference.relation_type
            object_correct += actual.object_key == reference.object_key and actual.object_type == reference.object_type
            condition_correct += 1
            provenance_correct += {ref.insight_id for ref in actual.evidence_refs} == {ref.insight_id for ref in reference.evidence_refs}
            unmatched.remove(match)
            matched += 1
            case_matches += 1
        review_matches += sum(
            any(
                _triple(reference.relation) == _triple(actual)
                and _condition_matches(actual.condition, reference.required_condition_terms, reference.relation.condition)
                for reference in expected
            )
            for actual in reviewed
        )
        outputs.append({
            "case_id": case.id, "expected": [_expected_json(item) for item in expected],
            "decisions": [_decision_json(item) for item in decisions], "matched": case_matches,
            "failure": failure,
            "missed_expected": [_expected_json(item) for item in unmatched],
            "reference_packet_warnings": unresolved,
        })
    precision = matched / accepted_total if accepted_total else (1.0 if not expected_total else 0.0)
    recall = matched / expected_total if expected_total else 1.0
    denominator = max(accepted_total, 1)
    false_positives = accepted_total - matched
    false_negatives = expected_total - matched
    f1_denominator = 2 * matched + false_positives + false_negatives
    return {
        "metrics": {
            "true_positive": matched,
            "false_positive": false_positives,
            "false_negative": false_negatives,
            "relation_precision": precision, "relation_recall": recall,
            "f1": (2 * matched / f1_denominator) if f1_denominator else 1.0,
            "subject_correctness": subject_correct / denominator,
            "relation_type_correctness": type_correct / denominator,
            "object_correctness": object_correct / denominator,
            "condition_preservation": condition_correct / denominator,
            "provenance_correctness": provenance_correct / denominator,
            "unsupported_inference_rate": (accepted_total - matched) / denominator,
            "canonicalization_quality": matched / expected_total if expected_total else 1.0,
            "overmerge_or_undermerge_count": abs(accepted_total - expected_total),
            "review_matches": review_matches,
            "review_count": review_total,
            "rejected_count": rejected_total,
            "unresolved_reference_entity_count": unresolved_reference_entities,
        },
        "expected_relation_count": expected_total, "accepted_relation_count": accepted_total,
        "cases": outputs,
        "failure_attribution": {
            "failure_stages": dict(sorted(failure_stages.items())),
            "decision_reasons": dict(sorted(disposition_reasons.items())),
        },
    }


def _triple(relation) -> tuple[str, str, str, str, str]:
    return (relation.subject_type, relation.subject_key, relation.relation_type, relation.object_type, relation.object_key)


def _semantic_key(relation) -> tuple[str, str, str, str, str, str | None]:
    return (*_triple(relation), relation.condition)


def _condition_matches(
    condition: str | None, required_terms: tuple[str, ...], reference: str | None = None,
) -> bool:
    """Require labeled qualifiers but do not equate semantically identical prose."""
    if not required_terms:
        return condition == reference
    if not condition:
        return False
    tokens = re.findall(r"[a-z0-9]+", condition.lower())
    return all(_contains_token_phrase(tokens, term) for term in required_terms)


def _contains_token_phrase(tokens: list[str], phrase: str) -> bool:
    expected = re.findall(r"[a-z0-9]+", phrase.lower())
    if not expected:
        return False
    width = len(expected)
    for index in range(len(tokens) - width + 1):
        actual = tokens[index:index + width]
        if actual[:-1] == expected[:-1] and (
            actual[-1] == expected[-1]
            or (len(expected[-1]) >= 3 and actual[-1].startswith(expected[-1]))
        ):
            return True
    return False


def _unresolved_reference_entities(
    expected: list[ExpectedRelation], packet: ExtractionPacket,
) -> list[str]:
    """Expose labels the production packet cannot currently canonicalize."""
    known_abilities = set(packet.ability_aliases.values())
    warnings = []
    for item in expected:
        relation = item.relation
        for entity_type, key in (
            (relation.subject_type, relation.subject_key),
            (relation.object_type, relation.object_key),
        ):
            if entity_type == "ability" and key not in known_abilities:
                warnings.append(f"runtime packet does not expose ability alias: {key}")
    return warnings


def _reference_relation(
    case_id: str, index: int, raw: Mapping[str, Any], packet: ExtractionPacket,
) -> ExpectedRelation:
    evidence_ids = tuple(raw.get("evidence_ids", ()))
    evidence_by_id = {item.insight_id: item for item in packet.evidence}
    if not evidence_ids or any(item not in evidence_by_id for item in evidence_ids):
        raise ValueError(f"validation case {case_id} has missing or unknown reference evidence")
    refs = tuple(
        EvidenceRef(
            evidence_by_id[item].source_type,
            evidence_by_id[item].source_id,
            item,
            evidence_by_id[item].text,
        )
        for item in evidence_ids
    )
    relation = StrategicRelation(
        id=f"reference-{case_id}-{index}",
        subject_type=raw["subject_type"], subject_key=raw["subject"],
        relation_type=raw["relation_type"], object_type=raw["object_type"],
        object_key=raw["object"], confidence=1.0,
        provenance_type=raw.get("provenance_type", "source_claim"), evidence_refs=refs,
        condition=raw.get("condition"), effect=raw.get("effect"),
        concepts=tuple(raw.get("concepts", ())),
        patch_sensitivity=raw.get("patch_sensitivity", "low"),
    )
    relation.validate()
    return ExpectedRelation(relation, tuple(raw.get("required_condition_terms", ())))


def _relation_json(relation):
    return asdict(relation)


def _expected_json(expected: ExpectedRelation):
    rendered = _relation_json(expected.relation)
    rendered["required_condition_terms"] = list(expected.required_condition_terms)
    return rendered


def _decision_json(decision):
    return {
        "status": decision.status, "warnings": list(decision.warnings),
        "raw": dict(decision.raw), "confidence_components": dict(decision.confidence_components),
        "relation": _relation_json(decision.relation) if decision.relation else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 relation extraction")
    parser.add_argument("--fixture", type=Path, default=Path("data/relation_extraction_validation_v0.json"))
    parser.add_argument("--db", type=Path, help="Source SQLite DB; required with --live to reuse production packet aliases")
    parser.add_argument("--live", action="store_true", help="Call the configured extraction model; never writes relations")
    parser.add_argument("--variant", choices=("flash", "pro"), help="DeepSeek relation model variant for a fair live comparison")
    parser.add_argument("--case-id", action="append", help="Evaluate only this fixture case; repeatable")
    parser.add_argument("--model", help="Explicit model identifier; overrides --variant")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.live and not args.db:
        parser.error("--live requires --db so evaluation uses the production packet loader")
    cases = load_cases(args.fixture, source_db=args.db)
    if args.case_id:
        requested = set(args.case_id)
        cases = tuple(case for case in cases if case.id in requested)
        missing = requested - {case.id for case in cases}
        if missing:
            parser.error("unknown --case-id: " + ", ".join(sorted(missing)))
    if args.variant and BACKEND != "deepseek":
        parser.error("--variant requires LLM_PROVIDER=deepseek")
    if args.model:
        selected_model, model_label = args.model, "custom"
    elif args.variant == "flash":
        selected_model, model_label = RELATION_FLASH_MODEL, "flash"
    elif args.variant == "pro":
        selected_model, model_label = RELATION_PRO_MODEL, "pro"
    else:
        selected_model, model_label = None, "provider_default"
    if args.live:
        result = evaluate_cases(cases, lambda packet: extract_relation_trace(packet, chat, model=selected_model))
    else:
        result = {"status": "validated_fixture_only", "case_count": len(cases), "message": "Pass --live to make model calls."}
    result["model"] = {"backend": BACKEND, "model": selected_model or MODEL, "variant": model_label} if args.live else None
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
