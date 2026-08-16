"""Development-only Phase 2D source-mode proposition evaluation.

This module measures source recovery and Stage A independently.  It does not
invoke canonical mapping, write a ledger, or alter bronze/evidence records.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from pipeline.proposition_extract import ExtractedProposition, PropositionPacket, SourceMode
from pipeline.source_windows import SourceWindow, SourceWindowResolver


def load_development_cases(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load the separately maintained Phase 2D development-only fixture."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
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
        if case["eligible"] != bool(case["expected_propositions"]):
            raise ValueError(f"Phase 2D fixture case {case['id']} has inconsistent eligible/safe-zero labels")
        result.append(dict(case))
    return tuple(result)


def evaluate_source_modes(
    cases: Iterable[Mapping[str, Any]], *, resolver: SourceWindowResolver,
    extractor: Callable[[PropositionPacket], tuple[ExtractedProposition, ...]],
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
                entries.append({"mode": mode, "status": "failure", "reason": type(exc).__name__})
                continue
            entries.append(_score_mode(mode, case, packet, actual))
        output.append({
            "case_id": case["id"], "eligible": case["eligible"],
            "source_window": _window_json(window), "modes": entries,
        })
    return {"cases": output, "metrics": _summarize_source_modes(output, modes)}


def _score_mode(
    mode: SourceMode, case: Mapping[str, Any], packet: PropositionPacket,
    actual: tuple[ExtractedProposition, ...],
) -> dict[str, Any]:
    expected = list(case["expected_propositions"])
    unmatched = list(expected)
    matches = []
    for item in actual:
        matched = next((value for value in unmatched if _matches(item, value, packet)), None)
        if matched is not None:
            unmatched.remove(matched)
            matches.append(matched)
    return {
        "mode": mode, "status": "completed", "predicted_count": len(actual),
        "matched_count": len(matches), "expected_count": len(expected),
        "false_positive_count": len(actual) - len(matches),
        "missed_count": len(unmatched), "propositions": [_proposition_json(item) for item in actual],
    }


def _matches(actual: ExtractedProposition, expected: Mapping[str, Any], packet: PropositionPacket) -> bool:
    proposition = actual.proposition
    return _has_valid_grounding(actual, packet) and all(
        _normalize(getattr(proposition, field + "_source")) == _normalize(expected.get(field + "_source"))
        for field in ("subject", "predicate", "effect", "condition")
    )


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


def _summarize_source_modes(cases: list[dict[str, Any]], modes: tuple[SourceMode, ...]) -> dict[str, dict[str, float | int | None]]:
    summary = {}
    for mode in modes:
        entries = [entry for case in cases for entry in case["modes"] if entry["mode"] == mode]
        completed = [entry for entry in entries if entry["status"] == "completed"]
        eligible = [entry for case in cases if case["eligible"] for entry in case["modes"] if entry["mode"] == mode and entry["status"] == "completed"]
        source_available = [entry for case in cases if case["eligible"] for entry in case["modes"] if entry["mode"] == mode and entry["status"] != "unavailable"]
        safe_zero = [entry for case in cases if not case["eligible"] for entry in case["modes"] if entry["mode"] == mode and entry["status"] == "completed"]
        tp = sum(item["matched_count"] for item in completed)
        fp = sum(item["false_positive_count"] for item in completed)
        fn = sum(item["missed_count"] for item in completed)
        eligible_entry_count = sum(1 for case in cases if case["eligible"] for item in case["modes"] if item["mode"] == mode)
        summary[mode] = {
            "case_count": len(entries), "completed_case_count": len(completed),
            "unavailable_case_count": sum(item["status"] == "unavailable" for item in entries),
            "failure_case_count": sum(item["status"] == "failure" for item in entries),
            "eligible_source_coverage": len(source_available) / eligible_entry_count if eligible_entry_count else None,
            "proposition_precision": tp / (tp + fp) if tp + fp else (0.0 if eligible else None),
            "proposition_recall": tp / (tp + fn) if tp + fn else (0.0 if eligible else None),
            "unsupported_proposition_rate": fp / max(tp + fp, 1),
            "safe_zero_accuracy": sum(item["predicted_count"] == 0 for item in safe_zero) / len(safe_zero) if safe_zero else 0.0,
            "eligible_case_count": len(eligible),
        }
    return summary
