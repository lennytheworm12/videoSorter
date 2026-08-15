"""Evaluate Phase 2 extraction against a small source-grounded reference set.

Default mode validates the dataset only. ``--live`` invokes the configured cheap
relation-extraction model; it never persists its output.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from core.llm import BACKEND, MODEL, chat
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
    expected: tuple[ExtractionDecision, ...]


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
        expected = compile_candidates(packet, raw_case.get("expected", []))
        if any(decision.status != "accepted" for decision in expected):
            raise ValueError(f"validation case {raw_case['id']} has invalid reference relation")
        cases.append(ExtractionCase(raw_case["id"], packet, expected))
    return tuple(cases)


def evaluate_cases(
    cases: tuple[ExtractionCase, ...],
    extractor: Callable[[ExtractionPacket], tuple[ExtractionDecision, ...] | object],
) -> dict[str, Any]:
    """Compare normalized relation structure without treating prose effects as identity."""
    outputs = []
    matched = expected_total = accepted_total = 0
    subject_correct = type_correct = object_correct = condition_correct = provenance_correct = 0
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
        expected = [item.relation for item in case.expected]
        accepted = [item.relation for item in decisions if item.status == "accepted" and item.relation]
        expected_total += len(expected)
        accepted_total += len(accepted)
        unmatched = list(expected)
        case_matches = 0
        for actual in accepted:
            match = next((item for item in unmatched if _triple(item) == _triple(actual)), None)
            if match is None:
                continue
            subject_correct += actual.subject_key == match.subject_key and actual.subject_type == match.subject_type
            type_correct += actual.relation_type == match.relation_type
            object_correct += actual.object_key == match.object_key and actual.object_type == match.object_type
            condition_correct += actual.condition == match.condition
            provenance_correct += {ref.insight_id for ref in actual.evidence_refs} == {ref.insight_id for ref in match.evidence_refs}
            if _semantic_key(actual) == _semantic_key(match):
                unmatched.remove(match)
                matched += 1
                case_matches += 1
        outputs.append({
            "case_id": case.id, "expected": [_relation_json(item) for item in expected],
            "decisions": [_decision_json(item) for item in decisions], "matched": case_matches,
            "failure": failure,
        })
    precision = matched / accepted_total if accepted_total else (1.0 if not expected_total else 0.0)
    recall = matched / expected_total if expected_total else 1.0
    denominator = max(accepted_total, 1)
    return {
        "metrics": {
            "relation_precision": precision, "relation_recall": recall,
            "subject_correctness": subject_correct / denominator,
            "relation_type_correctness": type_correct / denominator,
            "object_correctness": object_correct / denominator,
            "condition_preservation": condition_correct / denominator,
            "provenance_correctness": provenance_correct / denominator,
            "unsupported_inference_rate": (accepted_total - matched) / denominator,
            "canonicalization_quality": matched / expected_total if expected_total else 1.0,
            "overmerge_or_undermerge_count": abs(accepted_total - expected_total),
        },
        "expected_relation_count": expected_total, "accepted_relation_count": accepted_total,
        "cases": outputs,
    }


def _triple(relation) -> tuple[str, str, str, str, str]:
    return (relation.subject_type, relation.subject_key, relation.relation_type, relation.object_type, relation.object_key)


def _semantic_key(relation) -> tuple[str, str, str, str, str, str | None]:
    return (*_triple(relation), relation.condition)


def _relation_json(relation):
    return asdict(relation)


def _decision_json(decision):
    return {"status": decision.status, "warnings": list(decision.warnings), "relation": _relation_json(decision.relation) if decision.relation else None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 relation extraction")
    parser.add_argument("--fixture", type=Path, default=Path("data/relation_extraction_validation_v0.json"))
    parser.add_argument("--db", type=Path, help="Source SQLite DB; required with --live to reuse production packet aliases")
    parser.add_argument("--live", action="store_true", help="Call the configured extraction model; never writes relations")
    parser.add_argument("--variant", choices=("flash", "pro"), help="DeepSeek relation model variant for a fair live comparison")
    parser.add_argument("--model", help="Explicit model identifier; overrides --variant")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.live and not args.db:
        parser.error("--live requires --db so evaluation uses the production packet loader")
    cases = load_cases(args.fixture, source_db=args.db)
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
